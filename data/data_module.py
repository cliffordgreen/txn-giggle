import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Import scaler
# Keep Dataset for type hints if needed elsewhere, but not directly used for loader
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch.nn.utils.rnn import pad_sequence # For padding sequences
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
import pytorch_lightning as pl
import time # For timing setup steps
import os

class SingleBatchIterable:
    """An iterable wrapper that yields a single batch once per epoch."""
    def __init__(self, batch):
        if batch is None:
             raise ValueError("SingleBatchIterable cannot be initialized with None batch.")
        self.batch = batch
        self.yielded = False

    def __iter__(self):
        # Reset yielded flag at the start of each iteration (epoch)
        self.yielded = False
        return self

    def __next__(self):
        # Yield the batch only once per iteration
        if not self.yielded:
            self.yielded = True
            return self.batch
        else:
            # Signal the end of the iteration (epoch)
            raise StopIteration

    def __len__(self):
        # Define the length as 1 (one batch per epoch)
        # This helps Lightning estimate steps, etc.
        return 1


class TransactionDataModule(pl.LightningDataModule):
    """
    Data module for transaction classification using NeighborLoader for graph sampling.
    Processes graph, sequence, and text data during setup. Applies scaling.
    """
    def __init__(
        self,
        transactions_df: pd.DataFrame,
        batch_size: int = 128,
        num_workers: int = 4,
        # Sequence params
        max_seq_length: int = 50,
        # Text params
        text_model_name: str = 'bert-base-uncased',
        text_max_length: int = 128,
        # Graph sampling params
        num_neighbors: List[int] = [15, 10],
        # Data split params
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        perform_overfit_test: bool = False
    ):
        super().__init__()
        # --- Store initial configuration ---
        print("Initializing TransactionDataModule...")
        # Make a copy to avoid modifying the original DataFrame passed in
        self.transactions_df = transactions_df.copy()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.text_model_name = text_model_name
        self.text_max_length = text_max_length
        self.num_neighbors = num_neighbors
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # --- Placeholders for processed data ---
        self.tokenizer = None
        self.graph_data: Optional[HeteroData] = None
        self.node_feature_dims: Dict[str, int] = {}
        self.edge_feature_dims: Dict[Tuple[str, str, str], int] = {}
        self.sequence_feature_dim: Optional[int] = None
        self.scalers: Dict[str, StandardScaler] = {} # Store node scalers
        self.seq_scalers: Dict[str, StandardScaler] = {} # Store sequence feat scalers
        self.edge_scalers: Dict[str, StandardScaler] = {} # Store edge feat scalers
        self.perform_overfit_test = perform_overfit_test
        self.first_train_batch = None # To store the single batch

        # --- Pre-process incoming DataFrame ---
        start_time = time.time()
        print("Pre-processing DataFrame (datetime, category IDs, time features)...")
        if not pd.api.types.is_datetime64_any_dtype(self.transactions_df['timestamp']):
            try:
                # Attempt conversion with error coercion
                self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'], errors='coerce')
                # Check if conversion failed for any rows
                if self.transactions_df['timestamp'].isnull().any():
                     print(f"[WARN] Coerced {self.transactions_df['timestamp'].isnull().sum()} invalid timestamps to NaT.")
                     # Consider filling NaT here if necessary, e.g., with median or a default
                     # self.transactions_df['timestamp'].fillna(self.transactions_df['timestamp'].median(), inplace=True)
                     # For now, let subsequent steps handle potential NaNs if applicable
                print("Converted 'timestamp' column to datetime.")
            except Exception as e:
                raise ValueError(f"Failed to convert 'timestamp' column to datetime: {e}")

        # Add time features needed for nodes/sequences early
        self.transactions_df['hour'] = self.transactions_df['timestamp'].dt.hour
        self.transactions_df['weekday'] = self.transactions_df['timestamp'].dt.weekday

        # Handle potential NaNs in hour/weekday if timestamp conversion failed
        if self.transactions_df['hour'].isnull().any():
             print("[WARN] Filling NaN 'hour' values (likely from NaT timestamps) with 0.")
             self.transactions_df['hour'].fillna(0, inplace=True)
        if self.transactions_df['weekday'].isnull().any():
             print("[WARN] Filling NaN 'weekday' values (likely from NaT timestamps) with 0.")
             self.transactions_df['weekday'].fillna(0, inplace=True)

        # Ensure types after potential filling
        self.transactions_df['hour'] = self.transactions_df['hour'].astype(int)
        self.transactions_df['weekday'] = self.transactions_df['weekday'].astype(int)

        # --- Category ID Handling ---
        self.category_id_map = None
        if 'category_id' in self.transactions_df.columns:
            # Fill NaNs before checking type or factorizing
            if self.transactions_df['category_id'].isnull().any():
                print(f"[WARN] 'category_id' contains NaNs. Filling with 'UNKNOWN_CAT'.")
                self.transactions_df['category_id'] = self.transactions_df['category_id'].fillna('UNKNOWN_CAT')
            
            # Check if column seems to contain strings (like 'CATxxxx')
            if self.transactions_df['category_id'].dtype == 'object':
                print("'category_id' column contains strings. Factorizing...")
                # Factorize converts strings to integers (0, 1, 2...)
                # It returns the integer codes and the unique category strings
                codes, uniques = pd.factorize(self.transactions_df['category_id'], sort=True)
                self.transactions_df['category_id'] = codes # Assign integer codes back to the column
                self.category_id_map = {code: unique_val for code, unique_val in enumerate(uniques)}
                print(f"Factorized 'category_id' into {len(uniques)} unique integer IDs.")
                # print(f"Category ID Mapping (first 5): {list(self.category_id_map.items())[:5]}")
            else:
                # If not object type, attempt conversion to int (original logic)
                try:
                    self.transactions_df['category_id'] = self.transactions_df['category_id'].astype(int)
                except ValueError as e:
                    print(f"[ERROR] Could not convert 'category_id' to int: {e}")
                    raise e
        else:
            print("[WARN] 'category_id' column not found.")

        # --- User Category ID Handling (similar logic) ---
        self.user_category_id_map = None
        if 'user_category_id' in self.transactions_df.columns:
            if self.transactions_df['user_category_id'].isnull().any():
                print(f"[WARN] 'user_category_id' contains NaNs. Filling with 'UNKNOWN_USER_CAT'.")
                self.transactions_df['user_category_id'] = self.transactions_df['user_category_id'].fillna('UNKNOWN_USER_CAT')
            
            if self.transactions_df['user_category_id'].dtype == 'object':
                print("'user_category_id' column contains strings. Factorizing...")
                codes, uniques = pd.factorize(self.transactions_df['user_category_id'], sort=True)
                self.transactions_df['user_category_id'] = codes
                self.user_category_id_map = {code: unique_val for code, unique_val in enumerate(uniques)}
                print(f"Factorized 'user_category_id' into {len(uniques)} unique integer IDs.")
            else:
                try:
                    self.transactions_df['user_category_id'] = self.transactions_df['user_category_id'].astype(int)
                except ValueError as e:
                    print(f"[ERROR] Could not convert 'user_category_id' to int: {e}")
                    raise e
        # else: user_category_id might be optional

        # Ensure user_id exists for splitting and sequence building
        if 'user_id' not in self.transactions_df.columns:
             raise ValueError("DataFrame must contain a 'user_id' column for splitting and sequence preparation.")

        print(f"DataFrame pre-processing finished in {time.time() - start_time:.2f}s")


    def prepare_data(self):
        # Download tokenizer models if not cached
        print(f"Downloading/loading tokenizer: {self.text_model_name}")
        try:
             _ = AutoTokenizer.from_pretrained(self.text_model_name)
             print("Tokenizer ready.")
        except Exception as e:
             print(f"[ERROR] Failed to download/load tokenizer '{self.text_model_name}': {e}")
             raise e

    def setup(self, stage: Optional[str] = None):
        """
        Build graph, process features (incl. scaling), tokenize text, prepare sequences, create masks.
        """
        print(f"\n--- Starting Data Setup for stage: {stage} ---")
        setup_start_time = time.time()

        if self.tokenizer is None:
            print("Initializing tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            print("Tokenizer initialized.")

        # --- Build Graph & Calculate Raw Features ---
        # Avoid rebuilding if already done (check if graph_data exists)
        if self.graph_data is None:
            print("Calculating raw node features (vectorized)...")
            start_time = time.time()
            raw_features = self._calculate_raw_features()
            print(f"Raw node features calculated in {time.time() - start_time:.2f}s")

            # --- Fit Scalers ---
            # NOTE: Fitting on the entire dataset here for simplicity.
            # For strictness, fit only on the training portion AFTER splitting.
            # This requires passing the train_mask or train indices to _fit_scalers.
            print("Fitting scalers on all calculated raw features...")
            start_time = time.time()
            self._fit_scalers(raw_features)
            print(f"Scalers fitted in {time.time() - start_time:.2f}s")

            # --- Build Graph with Scaled Features ---
            print("Building graph structure with scaled features...")
            start_time = time.time()
            # Pass raw features so _build_graph can apply the fitted scalers
            self._build_graph_and_edges(raw_features)
            print(f"Graph structure and scaled features built in {time.time() - start_time:.2f}s")

            # --- Prepare and Add Sequences (with scaling) ---
            print("Preparing sequence data (including scaling)...")
            start_time = time.time()
            self._prepare_and_add_sequences()
            print(f"Sequence data prepared in {time.time() - start_time:.2f}s")

            # --- Prepare and Add Text Data ---
            print("Preparing text data...")
            start_time = time.time()
            self._prepare_and_add_text()
            print(f"Text data prepared in {time.time() - start_time:.2f}s")

            # --- Split Data and Add Masks ---
            # Splitting needs to happen before fitting scalers ideally,
            # but we fit on all data for now. Add masks after graph is built.
            print("Splitting data by user and adding masks...")
            start_time = time.time()
            self._split_data_and_add_masks()
            print(f"Data split and masks added in {time.time() - start_time:.2f}s")

            print("\n--- Data Setup Summary ---")
            print(f"Graph: {self.graph_data}")
            print(f"Node Feature Dims (Scaled): {self.node_feature_dims}")
            print(f"Edge Feature Dims (Scaled where applicable): {self.edge_feature_dims}")
            print(f"Sequence Feature Dim (Scaled): {self.sequence_feature_dim}")
            if 'transaction' in self.graph_data:
                 print(f"Num Train Nodes: {self.graph_data['transaction'].train_mask.sum().item()}")
                 print(f"Num Val Nodes: {self.graph_data['transaction'].val_mask.sum().item()}")
                 print(f"Num Test Nodes: {self.graph_data['transaction'].test_mask.sum().item()}")
            print("--- End Data Setup Summary ---")
        else:
             print("Graph data already exists. Skipping build process.")

        if stage == 'fit' and self.perform_overfit_test and self.first_train_batch is None:
            print("\n[INFO] Overfitting Test: Getting the first training batch...")
            temp_loader = self._create_loader(self.graph_data['transaction'].train_mask, shuffle=False) # No shuffle needed
            if temp_loader:
              self.first_train_batch = next(iter(temp_loader))
              print("[INFO] Stored the first training batch for overfitting.")
        else:
              print("[WARN] Could not get first batch for overfitting test (train set empty?).")

        print(f"--- Data Setup finished for stage: {stage} in {time.time() - setup_start_time:.2f}s ---")


    def _calculate_raw_features(self) -> Dict[str, np.ndarray]:
        """Calculates raw numerical features BEFORE scaling for nodes."""
        df = self.transactions_df
        raw_features_dict = {}
        print("Calculating raw transaction features (vectorized)...")

        # --- Transaction Features (Vectorized - Already Done) ---
        self.tx_feat_cols_to_scale = ['amount']
        self.tx_feat_cols_no_scale = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        hour = df['hour'].fillna(0).astype(int)
        day = df['weekday'].fillna(0).astype(int)
        amount = df['amount'].fillna(0.0)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        tx_features_array = np.column_stack([
            amount, hour_sin, hour_cos, day_sin, day_cos
        ])
        raw_features_dict['transaction'] = tx_features_array.astype(np.float64)
        print(f"  Raw transaction features calculated. Shape: {raw_features_dict['transaction'].shape}")

        # Define aggregations with explicit output names
        agg_funcs_named = {
            'amount_mean': ('amount', 'mean'),
            'amount_std': ('amount', lambda x: x.std(ddof=0)),
            'amount_max': ('amount', 'max'),
            'amount_min': ('amount', 'min'),
            'amount_count': ('amount', 'count'),
            'amount_median': ('amount', 'median'),
            'amount_q25': ('amount', lambda x: x.quantile(0.25)),
            'amount_q75': ('amount', lambda x: x.quantile(0.75))
        }

        # --- Merchant Features (Vectorized with Named Aggregation) ---
        print("Calculating raw merchant features (vectorized)...")
        merchant_ids = df['merchant_name'].dropna().unique()
        merchant_map = {name: i for i, name in enumerate(merchant_ids)}
        num_merchants = len(merchant_map)

        if num_merchants > 0:
            # Use named aggregations directly
            merchant_stats = df[pd.notna(df['merchant_name'])].groupby('merchant_name').agg(**agg_funcs_named)
            # Columns already have the desired names (e.g., 'amount_mean', 'amount_std')
            merchant_stats = merchant_stats.fillna(0) # Fill any NaNs 
            merchant_stats = merchant_stats.reindex(merchant_ids, fill_value=0) # Ensure order and all merchants
            # Get column names in the desired order
            final_merchant_cols = list(agg_funcs_named.keys())
            raw_features_dict['merchant'] = merchant_stats[final_merchant_cols].values.astype(np.float64)
        else:
             print("No valid merchants found. Creating empty merchant features.")
             raw_features_dict['merchant'] = np.zeros((0, 8), dtype=np.float64)
        print(f"  Raw merchant features calculated. Shape: {raw_features_dict['merchant'].shape}")

        # --- Category Features (Vectorized with Named Aggregation) ---
        print("Calculating raw category features (vectorized)...")
        valid_categories = df['category_id'].dropna().unique()
        valid_categories = [c for c in valid_categories if isinstance(c, (int, np.integer)) and c != -1]
        # No need for category_map here as we use category_id directly for groupby/reindex
        num_categories = len(valid_categories)

        if num_categories > 0:
            # Use named aggregations directly
            category_stats = df[df['category_id'].isin(valid_categories)].groupby('category_id').agg(**agg_funcs_named)
            category_stats = category_stats.fillna(0) # Fill NaNs
            category_stats = category_stats.reindex(valid_categories, fill_value=0) # Ensure order and all categories
            # Get column names in the desired order
            final_category_cols = list(agg_funcs_named.keys())
            raw_features_dict['category'] = category_stats[final_category_cols].values.astype(np.float64)
        else:
            print("No valid categories found. Creating empty category features.")
            raw_features_dict['category'] = np.zeros((0, 8), dtype=np.float64)
        print(f"  Raw category features calculated. Shape: {raw_features_dict['category'].shape}")

        return raw_features_dict


    def _fit_scalers(self, raw_features: Dict[str, np.ndarray]):
        """Fits StandardScaler on the raw numerical features.
           NOTE: Ideally fit only on training data. Fitting on all data here for simplicity."""

        # --- Fit Node Scalers ---
        for node_type, features in raw_features.items():
            # Check if features array is not empty and has more than 0 samples
            if features.size > 0 and features.shape[0] > 0:
                if node_type == 'transaction':
                    # Scale only the designated columns (e.g., 'amount')
                    num_cols_to_scale = len(self.tx_feat_cols_to_scale)
                    if features.shape[1] >= num_cols_to_scale and num_cols_to_scale > 0:
                         scaler = StandardScaler()
                         # Fit only on the columns to be scaled
                         scaler.fit(features[:, :num_cols_to_scale])
                         self.scalers[node_type] = scaler
                         print(f"  Fitted scaler for 'transaction' features (first {num_cols_to_scale} cols). Mean: {scaler.mean_}, Scale: {scaler.scale_}")
                    else:
                         print(f"  [WARN] Not enough columns in transaction features to scale or no columns designated.")
                else: # Scale all features for merchant/category
                    scaler = StandardScaler()
                    scaler.fit(features)
                    self.scalers[node_type] = scaler
                    print(f"  Fitted scaler for '{node_type}' features. Mean: {scaler.mean_}, Scale: {scaler.scale_}")
            else:
                print(f"  Skipping scaler fitting for empty features: '{node_type}'")


        # --- Fit Sequence Scalers ---
        # Need to calculate raw sequences first to fit scalers properly.
        # This is complex without easy access to train split here.
        # Workaround 1: Reuse node 'amount' scaler for sequence 'amount'.
        # Workaround 2: Calculate all time deltas and fit scaler (less pure).
        # Workaround 3: Skip scaling sequence features for now.

        # Workaround 1: Reuse transaction amount scaler
        if 'transaction' in self.scalers:
            tx_scaler = self.scalers['transaction']
            # Assuming 'amount' is the first feature in self.tx_feat_cols_to_scale
            if len(tx_scaler.mean_) >= 1:
                 amount_seq_scaler = StandardScaler()
                 amount_seq_scaler.mean_ = tx_scaler.mean_[0:1]
                 amount_seq_scaler.scale_ = tx_scaler.scale_[0:1]
                 self.seq_scalers['amount'] = amount_seq_scaler
                 print("  Created sequence scaler for 'amount' based on transaction scaler.")

        # Fit scaler for time_delta (Placeholder - requires generating time deltas)
        # For now, we will skip scaling time_delta, but add a placeholder
        self.seq_scalers['time_delta'] = None # Indicate no scaler fitted yet
        print("  [INFO] Skipping scaler fitting for sequence 'time_delta' (requires refinement).")

        # --- Fit Edge Scalers ---
        # Fit scaler for amount distance 'dist' feature in similar_amount edge
        # Placeholder - requires generating distances first.
        # Using raw amount std dev as a proxy for distance scale - very rough approximation
        if 'transaction' in self.scalers:
             amount_dist_scaler_proxy = StandardScaler()
             # Use std dev of amounts as a rough scale estimate for distances
             amount_dist_scaler_proxy.mean_ = np.array([0.0]) # Assume mean distance is 0? Risky.
             amount_dist_scaler_proxy.scale_ = self.scalers['transaction'].scale_[0:1] # Use amount std dev
             self.edge_scalers['amount_dist'] = amount_dist_scaler_proxy
             print("  Created proxy edge scaler for 'amount_dist' based on transaction amount std dev.")
        else:
             self.edge_scalers['amount_dist'] = None
             print("  [INFO] Skipping scaler fitting for edge 'amount_dist'.")


    def _build_graph_and_edges(self, raw_features: Dict[str, np.ndarray]):
        """Builds HeteroData object with SCALED node features and calculated edge features."""
        data = HeteroData()
        df = self.transactions_df

        # --- Node Mapping and Basic Info ---
        tx_map = {idx: i for i, idx in enumerate(df.index)}
        merchants = df['merchant_name'].dropna().unique()
        merchant_map = {name: i for i, name in enumerate(merchants)}
        valid_categories = df['category_id'].dropna().unique()
        valid_categories = [c for c in valid_categories if isinstance(c, (int, np.integer)) and c != -1]
        category_map = {cat_id: i for i, cat_id in enumerate(valid_categories)}

        # --- Add SCALED Node Features ---
        print("Adding scaled node features to graph...")
        for node_type, raw_feat_array in raw_features.items():
            if raw_feat_array.size > 0: # Check if features exist
                if node_type in self.scalers:
                    scaler = self.scalers[node_type]
                    if node_type == 'transaction':
                         # Scale designated columns and concatenate
                         num_cols_to_scale = len(self.tx_feat_cols_to_scale)
                         # Add epsilon here for stability
                         scaled_part = (raw_feat_array[:, :num_cols_to_scale] - scaler.mean_) / (scaler.scale_ + 1e-8)
                         non_scaled_part = raw_feat_array[:, num_cols_to_scale:]
                         final_features = np.concatenate([scaled_part, non_scaled_part], axis=1)
                    else:
                         # Add epsilon here for stability
                         final_features = (raw_feat_array - scaler.mean_) / (scaler.scale_ + 1e-8)

                    data[node_type].x = torch.tensor(final_features, dtype=torch.float)
                    self.node_feature_dims[node_type] = data[node_type].x.shape[1]
                    # print(f"  Added scaled features for '{node_type}'. Shape: {data[node_type].x.shape}")
                    # Debug: Print stats after scaling
                    # print(f"    Scaled Stats: min={torch.min(data[node_type].x).item():.4f}, max={torch.max(data[node_type].x).item():.4f}")

                else: # Scaler wasn't fitted (e.g., empty raw features)
                    # Add raw features if they exist but weren't scaled (shouldn't happen often with current logic)
                    print(f"  [WARN] Adding raw features for '{node_type}' as scaler was not found/fitted.")
                    data[node_type].x = torch.tensor(raw_feat_array, dtype=torch.float)
                    self.node_feature_dims[node_type] = data[node_type].x.shape[1]
            # Else: If raw features were empty, data[node_type].x remains empty / unset

        # Add labels and original index (no change)
        data['transaction'].original_index = torch.tensor(list(tx_map.keys()), dtype=torch.long)
        data['transaction'].y_global = torch.tensor(df['category_id'].values, dtype=torch.long)
        user_cat_series = df['user_category_id'].fillna(-1).astype(int) # Already processed in __init__
        data['transaction'].y_user = torch.tensor(user_cat_series.values, dtype=torch.long)


        # --- Edge Creation & (Potentially Scaled) Features ---
        print("Creating edges and edge features...")
        edge_index_dict = {}
        edge_attr_dict = {}
        df_row_map = df.set_index(pd.Index(tx_map.keys())) # Map tx_node_idx back to row data

        # 1. Transaction -> Merchant (belongs_to) - Z-score is already scaled
        edge_list = []
        attr_list = []
        merchant_stats = df.groupby('merchant_name')['amount'].agg(['mean', 'std']).fillna(0)
        for idx, row in df.iterrows():
            if pd.notna(row['merchant_name']) and row['merchant_name'] in merchant_map:
                tx_node_idx = tx_map[idx]
                merchant_node_idx = merchant_map[row['merchant_name']]
                edge_list.append([tx_node_idx, merchant_node_idx])
                stats = merchant_stats.loc[row['merchant_name']]
                amount_zscore = (row['amount'] - stats['mean']) / (stats['std'] + 1e-8) # Added epsilon here too
                attr_list.append([amount_zscore]) # Keep as list for consistent shape
        if edge_list:
            edge_index_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(attr_list, dtype=torch.float)
            self.edge_feature_dims[('transaction', 'belongs_to', 'merchant')] = 1

        # 2. Merchant -> Category (categorized_as) - Confidence is 0-1
        edge_list = []
        attr_list = []
        merchant_primary_category = df.groupby('merchant_name')['category_id'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1)
        merchant_category_confidence = df.groupby('merchant_name')['category_id'].agg(lambda x: x.value_counts(normalize=True).max() if not x.empty else 0)
        for merchant_name, primary_cat_id in merchant_primary_category.items():
            # Check if primary_cat_id is in our valid category map
            if pd.notna(merchant_name) and merchant_name in merchant_map and pd.notna(primary_cat_id) and primary_cat_id in category_map:
                merchant_node_idx = merchant_map[merchant_name]
                category_node_idx = category_map[primary_cat_id]
                edge_list.append([merchant_node_idx, category_node_idx])
                confidence = merchant_category_confidence.get(merchant_name, 0)
                attr_list.append([confidence])
        if edge_list:
            edge_index_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(attr_list, dtype=torch.float)
            self.edge_feature_dims[('merchant', 'categorized_as', 'category')] = 1


        # 3. Transaction -> Transaction (temporal) - Time diff norm is 0-1 (days)
        edge_list = []
        attr_list = []
        # Use already sorted df if timestamp conversion worked
        df_sorted = df.sort_values('timestamp') if 'timestamp' in df else df
        tx_map_sorted = {idx: i for i, idx in enumerate(df_sorted.index)}
        num_transactions = data['transaction'].num_nodes # Use num nodes from graph object

        for i in range(num_transactions):
            row_i = df_sorted.iloc[i]
            ts_i = row_i['timestamp']
            if pd.isna(ts_i): continue # Skip if timestamp is invalid
            orig_idx_i = df_sorted.index[i]
            tx_node_i = tx_map[orig_idx_i]

            for k in range(1, 6): # Limit window size
                j = i + k
                if j >= num_transactions: break
                row_j = df_sorted.iloc[j]
                ts_j = row_j['timestamp']
                if pd.isna(ts_j): continue

                time_diff_seconds = abs((ts_i - ts_j).total_seconds())
                if time_diff_seconds <= 86400 * 1: # 1 day window
                    orig_idx_j = df_sorted.index[j]
                    tx_node_j = tx_map[orig_idx_j]
                    edge_list.extend([[tx_node_i, tx_node_j], [tx_node_j, tx_node_i]])
                    time_diff_norm = min(time_diff_seconds / 86400.0, 1.0) # Normalize and cap at 1 day
                    attr_list.extend([[time_diff_norm, 1.0], [time_diff_norm, 0.0]]) # Diff + direction
        if edge_list:
            edge_index_dict[('transaction', 'temporal', 'transaction')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr_dict[('transaction', 'temporal', 'transaction')] = torch.tensor(attr_list, dtype=torch.float)
            self.edge_feature_dims[('transaction', 'temporal', 'transaction')] = 2


        # 4. Transaction -> Transaction (similar_amount)
        edge_list = []
        raw_dist_list = []
        amount_ratio_list = []
        direction_list = []

        # Use raw amounts for NN calculation
        # Access raw amount from the original calculation (assuming it's first col)
        raw_tx_amounts = raw_features['transaction'][:, 0]

        if num_transactions > 1: # k-NN needs at least 2 points
            try:
                 from sklearn.neighbors import NearestNeighbors
                 k_neighbors = min(5, num_transactions - 1) # Adjust k if fewer nodes than requested
                 nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='minkowski', p=1, algorithm='auto')
                 nn.fit(raw_tx_amounts.reshape(-1, 1))
                 distances, indices = nn.kneighbors(raw_tx_amounts.reshape(-1, 1))

                 for i in range(num_transactions):
                     for k in range(1, k_neighbors + 1):
                         j = indices[i, k]
                         dist = distances[i, k]
                         tx_node_i = i # Assumes node indices 0..N-1 match amount array order
                         tx_node_j = j

                         edge_list.extend([[tx_node_i, tx_node_j], [tx_node_j, tx_node_i]])
                         raw_dist_list.extend([dist, dist]) # Store raw dist for scaling
                         # Ratio uses raw amounts
                         amount_i = raw_tx_amounts[i]
                         amount_j = raw_tx_amounts[j]
                         amount_ratio = min(amount_i, amount_j) / (max(amount_i, amount_j) + 1e-8) if max(amount_i, amount_j) > 1e-8 else 1.0 # Added epsilon
                         amount_ratio_list.extend([amount_ratio, amount_ratio])
                         direction_list.extend([1.0, 0.0]) # Direction bit

            except ImportError:
                 print("[WARN] scikit-learn not found. Skipping similar_amount edge creation.")

        if edge_list:
            # Scale the collected distances using the fitted scaler (if available)
            if 'amount_dist' in self.edge_scalers and self.edge_scalers['amount_dist'] is not None:
                scaler = self.edge_scalers['amount_dist']
                raw_dist_array = np.array(raw_dist_list, dtype=np.float64).reshape(-1, 1)
                # Add epsilon here for stability
                scaled_dist = ((raw_dist_array - scaler.mean_) / (scaler.scale_ + 1e-8)).flatten()
                print(f"  Applied scaler to 'amount_dist' edge feature.")
            else:
                print(f"  [WARN] Scaler for 'amount_dist' not found. Using raw distances.")
                scaled_dist = np.array(raw_dist_list) # Use raw distances

            # Combine scaled distance, ratio, direction
            final_attr_list = [[scaled_dist[idx], amount_ratio_list[idx], direction_list[idx]]
                               for idx in range(len(scaled_dist))]

            edge_index_dict[('transaction', 'similar_amount', 'transaction')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr_dict[('transaction', 'similar_amount', 'transaction')] = torch.tensor(final_attr_list, dtype=torch.float)
            self.edge_feature_dims[('transaction', 'similar_amount', 'transaction')] = 3


        # --- Assign final edge index and attributes to HeteroData ---
        print("Assigning final edge data to graph object...")
        for edge_type, index_tensor in edge_index_dict.items():
            data[edge_type].edge_index = index_tensor
            if edge_type in edge_attr_dict:
                data[edge_type].edge_attr = edge_attr_dict[edge_type]

        self.graph_data = data
        print("Graph building complete.")


    def _prepare_and_add_sequences(self):
        """Prepare sequence data (features of previous transactions) using SCALED features and cyclical time encodings."""
        if self.graph_data is None: raise RuntimeError("Graph data not built.")
        print("Preparing sequence features...")
        df_sorted = self.transactions_df.sort_values(['user_id', 'timestamp'])
        user_groups = df_sorted.groupby('user_id')

        all_seq_features = []
        all_seq_lengths = []
        # Define raw feature columns needed from DataFrame for sequence items
        # self.seq_raw_feature_cols = ['amount', 'weekday', 'hour'] # Old
        # Define the final dimension after processing/scaling
        # OLD: self.sequence_feature_dim = 4 # Scaled Amount, Weekday, Hour, Scaled/Transformed TimeDelta
        self.sequence_feature_dim = 6 # Scaled Amount, day_sin, day_cos, hour_sin, hour_cos, Scaled_TimeDelta

        original_to_sorted_pos = {idx: i for i, idx in enumerate(df_sorted.index)}

        # Pre-calculate time deltas to fit scaler (simplified: using all deltas)
        all_time_deltas = []
        print("  Calculating all time deltas for scaler fitting (simplified)...")
        for user_id, user_group_df in user_groups:
             timestamps = user_group_df['timestamp']
             time_diffs = timestamps.diff().dt.total_seconds().fillna(0).clip(lower=0) # Diff from previous, fill first with 0
             all_time_deltas.extend(time_diffs.tolist())

        if all_time_deltas:
             # Fit scaler for time delta (consider log transform + scale for skewed data)
             time_delta_array = np.array(all_time_deltas).reshape(-1, 1)
             # Option: Log transform before scaling
             # time_delta_array_log = np.log1p(time_delta_array)
             # scaler_td = StandardScaler().fit(time_delta_array_log)
             # No log transform for now:
             scaler_td = StandardScaler().fit(time_delta_array)
             self.seq_scalers['time_delta'] = scaler_td
             print(f"  Fitted time_delta scaler. Mean: {scaler_td.mean_}, Scale: {scaler_td.scale_}")

        print("  Generating sequences for each transaction...")
        # Iterate via original graph node order
        for orig_idx in self.graph_data['transaction'].original_index.tolist():
            current_pos_in_sorted = original_to_sorted_pos.get(orig_idx)
            if current_pos_in_sorted is None: continue # Should not happen if mapping is correct
            current_row = df_sorted.iloc[current_pos_in_sorted]
            user_id = current_row['user_id']

            # user_group_df = user_groups.get_group(user_id) # Can be slow repeated access
            # Find index within the sorted df
            try:
                 user_group_indices = df_sorted[df_sorted['user_id'] == user_id].index
                 current_pos_in_group = user_group_indices.get_loc(orig_idx)
            except KeyError:
                 continue # Current transaction not found in user group (shouldn't happen)


            start_idx_in_sorted = max(0, current_pos_in_sorted - self.max_seq_length)
            # Select rows from the globally sorted df based on position
            prev_txs_df = df_sorted.iloc[start_idx_in_sorted:current_pos_in_sorted]
            # Filter only for the current user's transactions within that window
            prev_txs_group = prev_txs_df[prev_txs_df['user_id'] == user_id]


            seq_features_for_tx = []
            if not prev_txs_group.empty:
                current_time = current_row['timestamp']
                for _, prev_row in prev_txs_group.iterrows():
                    # Calculate time delta
                    time_delta = 0.0
                    if pd.notna(current_time) and pd.notna(prev_row['timestamp']):
                         time_delta = (current_time - prev_row['timestamp']).total_seconds()

                    # Scale Amount
                    amount = prev_row['amount'] if pd.notna(prev_row['amount']) else 0.0
                    if 'amount' in self.seq_scalers:
                        scaler_a = self.seq_scalers['amount']
                        # Add epsilon here for stability
                        amount = (amount - scaler_a.mean_[0]) / (scaler_a.scale_[0] + 1e-8)

                    # Scale Time Delta
                    time_delta_val = time_delta
                    if 'time_delta' in self.seq_scalers and self.seq_scalers['time_delta'] is not None:
                         scaler_td = self.seq_scalers['time_delta']
                         # Apply same potential transform (log) if used in fitting
                         # time_delta_log = np.log1p(time_delta)
                         # Add epsilon here for stability
                         time_delta_val = (time_delta - scaler_td.mean_[0]) / (scaler_td.scale_[0] + 1e-8)

                    # Use CYCLICAL weekday/hour features
                    hour = prev_row['hour'] if pd.notna(prev_row['hour']) else 0
                    day = prev_row['weekday'] if pd.notna(prev_row['weekday']) else 0
                    hour_sin = np.sin(2 * np.pi * hour / 24) if pd.notna(hour) else 0.0
                    hour_cos = np.cos(2 * np.pi * hour / 24) if pd.notna(hour) else 1.0 # Cos(0)=1
                    day_sin = np.sin(2 * np.pi * day / 7) if pd.notna(day) else 0.0
                    day_cos = np.cos(2 * np.pi * day / 7) if pd.notna(day) else 1.0 # Cos(0)=1

                    # Order: Amount, day_sin, day_cos, hour_sin, hour_cos, TimeDelta
                    scaled_feat = [amount, day_sin, day_cos, hour_sin, hour_cos, time_delta_val]
                    seq_features_for_tx.append(scaled_feat)

            # Convert to tensor
            if seq_features_for_tx:
                seq_tensor = torch.tensor(seq_features_for_tx, dtype=torch.float)
                seq_len = len(seq_tensor)
            else:
                # Handle case with no previous transactions
                # Use sequence feature dim calculated earlier (now 6)
                seq_tensor = torch.zeros((0, self.sequence_feature_dim), dtype=torch.float)
                seq_len = 0

            all_seq_features.append(seq_tensor)
            all_seq_lengths.append(seq_len)

        # --- Add check for zero-length sequences ---
        num_zero_length = sum(1 for length in all_seq_lengths if length == 0)
        total_sequences = len(all_seq_lengths)
        if total_sequences > 0:
             print(f"  [INFO] Generated {num_zero_length} zero-length sequences out of {total_sequences} ({num_zero_length/total_sequences:.2%}).")
        # ---------------------------------------------

        # Pad sequences
        if not all_seq_features:
             print("[WARN] No sequence features generated.")
             padded_sequences = torch.empty((self.graph_data['transaction'].num_nodes, 0, self.sequence_feature_dim), dtype=torch.float)
        else:
             padded_sequences = pad_sequence(all_seq_features, batch_first=True, padding_value=0.0)

        # Add to graph data object
        self.graph_data['transaction'].seq_features = padded_sequences
        self.graph_data['transaction'].seq_lengths = torch.tensor(all_seq_lengths, dtype=torch.long)
        print(f"Added sequence features to graph. Padded shape: {padded_sequences.shape}, Expected Dim: {self.sequence_feature_dim}")


    def _prepare_and_add_text(self):
       # (No changes needed here based on scaling)
        if self.graph_data is None or self.tokenizer is None:
            raise RuntimeError("Graph data or tokenizer not initialized.")
        print("Tokenizing text fields...")
        text_fields_to_process = ['raw_description', 'memo', 'merchant_name']
        processed_tokens = {}
        start_time_text = time.time()
        for field in text_fields_to_process:
            if field not in self.transactions_df.columns:
                 print(f"[WARN] Text field '{field}' not found in DataFrame. Skipping.")
                 continue
            texts = self.transactions_df[field].fillna('').astype(str).tolist()
            # Replace empty strings with a single space, as some tokenizers treat "" differently
            texts = [t if t.strip() else " " for t in texts]

            print(f"  Tokenizing field: '{field}' for {len(texts)} transactions...")
            tokens = self.tokenizer(
                texts, padding='max_length', truncation=True,
                max_length=self.text_max_length, return_tensors='pt'
            )
            processed_tokens[f'{field}_input_ids'] = tokens['input_ids']
            processed_tokens[f'{field}_attention_mask'] = tokens['attention_mask']
            # print(f"  Tokenized '{field}'. Shape: {tokens['input_ids'].shape}") # Reduce verbosity

        for key, tensor in processed_tokens.items():
            self.graph_data['transaction'][key] = tensor
        print(f"Added tokenized text features to graph in {time.time() - start_time_text:.2f}s.")

    def _split_data_and_add_masks(self):
        # (No changes needed here based on scaling)
        if self.graph_data is None: raise RuntimeError("Graph data not built.")
        print("Creating train/val/test masks based on user ID...")
        num_transactions = self.graph_data['transaction'].num_nodes
        try:
             node_idx_to_user_id = self.transactions_df['user_id'].values
             if len(node_idx_to_user_id) != num_transactions:
                  # This can happen if df index wasn't 0..N-1 originally
                  # Re-fetch based on original_index stored in graph
                  orig_indices = self.graph_data['transaction'].original_index.tolist()
                  node_idx_to_user_id = self.transactions_df.loc[orig_indices, 'user_id'].values
                  if len(node_idx_to_user_id) != num_transactions:
                       raise ValueError("Mismatch between num_transactions and user_id mapping length even after re-indexing.")
        except KeyError:
             raise KeyError("DataFrame must contain 'user_id' column for splitting.")

        unique_users = np.unique(node_idx_to_user_id)
        np.random.shuffle(unique_users)
        n_users = len(unique_users)
        n_val = int(n_users * self.val_ratio)
        n_test = int(n_users * self.test_ratio)

        # Ensure at least one user in val/test if possible and ratios > 0
        if self.val_ratio > 0 and n_val == 0 and n_users > 1:
             n_val = 1
             print(f"[INFO] Adjusted n_val to {n_val} due to small number of users ({n_users}).")
        if self.test_ratio > 0 and n_test == 0 and n_users > (1 + n_val):
             n_test = 1
             print(f"[INFO] Adjusted n_test to {n_test} due to small number of users ({n_users}).")
        # Ensure train set is not empty if possible
        if n_val + n_test >= n_users and n_users > 0:
             print(f"[WARN] val_ratio + test_ratio >= 1.0, train set might be empty!")
             # Prioritize val/test splits if specified
             n_test = max(0, n_users - n_val)
             n_train = 0
        else:
             n_train = n_users - n_val - n_test

        print(f"Splitting {n_users} users into: Train={n_train}, Val={n_val}, Test={n_test}")

        val_users = set(unique_users[:n_val])
        test_users = set(unique_users[n_val : n_val + n_test])
        # Handle edge case where n_train could be 0
        train_users = set(unique_users[n_val + n_test :]) if n_train > 0 else set()


        train_mask = torch.zeros(num_transactions, dtype=torch.bool)
        val_mask = torch.zeros(num_transactions, dtype=torch.bool)
        test_mask = torch.zeros(num_transactions, dtype=torch.bool)
        for i in range(num_transactions):
            user_id = node_idx_to_user_id[i]
            if user_id in train_users: train_mask[i] = True
            elif user_id in val_users: val_mask[i] = True
            elif user_id in test_users: test_mask[i] = True

        self.graph_data['transaction'].train_mask = train_mask
        self.graph_data['transaction'].val_mask = val_mask
        self.graph_data['transaction'].test_mask = test_mask
        # --- START: Save Val/Test sets to CSV ---
        print("Attempting to save validation and test set node info to CSV...")
        try:
            # Ensure output directory exists
            save_dir = "training/output/run1"
            os.makedirs(save_dir, exist_ok=True)
    
            # Get necessary data (convert to numpy for easier indexing)
            print(self.graph_data['transaction'].y_global.cpu())#.numpy()
            # Use the node_idx_to_user_id mapping derived above
            # all_original_indices = self.graph_data['transaction'].original_index.cpu().numpy() # We already used this
    
            print( val_mask.cpu())#.numpy()
            print( test_mask.cpu())#.numpy()
    
            # --- Save Validation Set ---
            if np.any(val_mask_np):
                val_node_indices_0_N = np.where(val_mask_np)[0] # Get indices 0..N-1
                val_true_labels = all_labels[val_mask_np]
                val_user_ids = node_idx_to_user_id[val_mask_np]
                # Get original DF indices corresponding to these nodes
                val_original_indices = orig_indices[val_mask_np]
    
                val_df = pd.DataFrame({
                    'node_index': val_node_indices_0_N,
                    'original_df_index': val_original_indices,
                    'user_id': val_user_ids,
                    'true_category_id': val_true_labels
                })
                val_save_path = os.path.join(save_dir, 'validation_set_nodes.csv')
                val_df.to_csv(val_save_path, index=False)
                print(f"Saved validation set node info ({len(val_df)} nodes) to {val_save_path}")
            else:
                print("Validation set is empty. Skipping CSV save.")
    
            # --- Save Test Set ---
            if np.any(test_mask_np):
                test_node_indices_0_N = np.where(test_mask_np)[0] # Get indices 0..N-1
                test_true_labels = all_labels[test_mask_np]
                test_user_ids = node_idx_to_user_id[test_mask_np]
                 # Get original DF indices corresponding to these nodes
                test_original_indices = orig_indices[test_mask_np]
    
                test_df = pd.DataFrame({
                    'node_index': test_node_indices_0_N,
                    'original_df_index': test_original_indices,
                    'user_id': test_user_ids,
                    'true_category_id': test_true_labels
                })
                test_save_path = os.path.join(save_dir, 'test_set_nodes.csv')
                test_df.to_csv(test_save_path, index=False)
                print(f"Saved test set node info ({len(test_df)} nodes) to {test_save_path}")
            else:
                 print("Test set is empty. Skipping CSV save.")
    
        except Exception as e:
            print(f"[ERROR] Failed to save validation/test set info to CSV: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc() # Print full traceback for saving errors
        # --- END: Save Val/Test sets to CSV ---


        
        print("Masks added.")


    def _create_loader(self, mask: torch.Tensor, shuffle: bool) -> Optional[NeighborLoader]:
        # (No changes needed here based on scaling)
        if self.graph_data is None: raise RuntimeError("Graph data not loaded. Call setup() first.")
        num_seed_nodes = mask.sum().item()
        if num_seed_nodes == 0:
            print(f"[WARN] Mask for loader has zero nodes selected. Returning None.")
            return None

        input_nodes = ('transaction', mask)
        
        # Reverted: NeighborLoader should handle feature propagation automatically.
        # Explicitly listing features caused TypeError.
        # node_attrs = [
        #     'x', 'y_global', 'y_user', 'seq_features', 'seq_lengths',
        #     'raw_description_input_ids', 'raw_description_attention_mask',
        #     'memo_input_ids', 'memo_attention_mask',
        #     'merchant_name_input_ids', 'merchant_name_attention_mask'
        # ]
        # edge_attrs = ['edge_attr']

        loader = NeighborLoader(
            self.graph_data,
            num_neighbors=self.num_neighbors, # List of neighbors per hop [-1] for all
            shuffle=shuffle,
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            # Removed node_features and edge_features arguments
            num_workers=self.num_workers, # Use original request
            persistent_workers=(self.num_workers > 0),
        )
        print(f"Created NeighborLoader with {num_seed_nodes} seed nodes.")
        return loader

    # --- Dataloader Methods ---
    def train_dataloader(self) -> Optional[Union[NeighborLoader, SingleBatchIterable]]: # Adjust return type hint
        print("Creating train NeighborLoader / Overfit Batch...")
        if self.graph_data is None or not hasattr(self.graph_data['transaction'], 'train_mask'):
             self.setup('fit')
        if self.graph_data is None: return None
    
        if self.perform_overfit_test:
            if self.first_train_batch:
                print("[INFO] Overfitting Test: Returning SingleBatchIterable.")
                # --- MODIFICATION ---
                # Return an instance of the wrapper class instead of a list
                return SingleBatchIterable(self.first_train_batch)
                # --- END MODIFICATION ---
            else:
                print("[WARN] Overfitting Test: First batch not available, cannot overfit.")
                # Return None or raise an error if the batch wasn't captured in setup
                return None
        else:
            # Original behavior: return the full NeighborLoader
            # Ensure _create_loader handles potential empty train_mask
            return self._create_loader(self.graph_data['transaction'].train_mask, shuffle=True)
    
    # def train_dataloader(self) -> Optional[NeighborLoader]:
    #     print("Creating train NeighborLoader...")
    #     # Ensure setup has run for the 'fit' stage
    #     if self.graph_data is None or not hasattr(self.graph_data['transaction'], 'train_mask'):
    #          self.setup('fit')
    #     if self.graph_data is None: return None # Check again if setup failed
    #     # if self.perform_overfit_test:
    #     #     if self.first_train_batch:
    #     #         print("[INFO] Overfitting Test: Returning list containing only the first batch.")
    #     #         # Return the single batch wrapped in a list to mimic an iterable dataloader
    #     #         # Lightning handles this simple case.
    #     #         return [self.first_train_batch]
    #     #     else:
    #     #         print("[WARN] Overfitting Test: First batch not available, cannot overfit.")
    #     #         return None # Or raise error
    #     else:
    #         # Original behavior: return the full NeighborLoader
    #         return self._create_loader(self.graph_data['transaction'].train_mask, shuffle=True)
        
    #     return self._create_loader(self.graph_data['transaction'].train_mask, shuffle=True)

    def val_dataloader(self) -> Optional[NeighborLoader]:
        print("Creating validation NeighborLoader...")
        if self.graph_data is None or not hasattr(self.graph_data['transaction'], 'val_mask'):
             self.setup('fit') # Use 'fit' stage for validation too
        if self.graph_data is None: return None
        return self._create_loader(self.graph_data['transaction'].val_mask, shuffle=False)

    def test_dataloader(self) -> Optional[NeighborLoader]:
        print("Creating test NeighborLoader...")
        if self.graph_data is None or not hasattr(self.graph_data['transaction'], 'test_mask'):
             self.setup('test')
        if self.graph_data is None: return None
        return self._create_loader(self.graph_data['transaction'].test_mask, shuffle=False)


