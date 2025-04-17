import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader 
from torch_geometric.data import HeteroData, Batch 
from torch_geometric.loader import HGTLoader 
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union, Any
import pytorch_lightning as pl
import time 
import os
import functools 
from sklearn.neighbors import NearestNeighbors # Need this for similar_amount

# --- SingleBatchIterable Class (Keep if used for testing/overfitting) ---
class SingleBatchIterable:
    """An iterable wrapper that yields a single batch once per epoch."""
    def __init__(self, batch):
        if batch is None:
             raise ValueError("SingleBatchIterable cannot be initialized with None batch.")
        self.batch = batch
        self.yielded = False

    def __iter__(self):
        self.yielded = False
        return self

    def __next__(self):
        if not self.yielded:
            self.yielded = True
            return self.batch
        else:
            raise StopIteration

    def __len__(self):
        return 1

# --- V2 DataModule with HGTLoader --- 
class TransactionDataModuleV2(pl.LightningDataModule):
    """
    DataModule V2: Prepares HeteroData graph and uses HGTLoader.
    """
    def __init__(self,
                 transactions_df: pd.DataFrame,
                 batch_size: int = 32,
                 num_workers: int = 0, 
                 max_seq_length: int = 50,
                 text_model_name: str = 'bert-base-uncased',
                 text_max_length: int = 128,
                 num_hgt_layers: int = 2, 
                 hgt_num_samples: Optional[Dict[str, List[int]]] = None, 
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 use_sequence_encoder: bool = True,
                 use_text_encoder: bool = True,
                 use_gnn_encoder: bool = True,
                 use_scheduleC_label: bool = False
                 ):
        super().__init__()
        self.transactions_df = transactions_df.copy()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.text_model_name = text_model_name
        self.text_max_length = text_max_length
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.use_sequence_encoder = use_sequence_encoder
        self.use_text_encoder = use_text_encoder
        self.use_gnn_encoder = use_gnn_encoder
        self.use_scheduleC_label = use_scheduleC_label

        # Define known base node types
        self.node_types_in_graph = ['transaction', 'merchant', 'category']
        # Add new node types if GNN is used
        if self.use_gnn_encoder:
            self.node_types_in_graph.extend(['mcc', 'sic'])

        # Store/Create HGT sampling config
        self.num_hgt_layers = num_hgt_layers
        self.hgt_num_samples = hgt_num_samples
        if self.hgt_num_samples is None and self.use_gnn_encoder:
            # Simple default: Sample 15 neighbors in first layer, 10 in second
            default_samples_per_layer = [15, 10]
            # Use the defined node types list
            self.hgt_num_samples = {ntype: default_samples_per_layer[:self.num_hgt_layers] for ntype in self.node_types_in_graph}
            print(f"[WARN] hgt_num_samples not provided. Using default based on num_hgt_layers={self.num_hgt_layers} "
                  f"for node types {self.node_types_in_graph}: {self.hgt_num_samples}")
        elif self.hgt_num_samples is not None and self.use_gnn_encoder:
             # Validate provided samples
             for ntype in self.node_types_in_graph:
                  if ntype not in self.hgt_num_samples:
                      print(f"[WARN] User-provided hgt_num_samples missing node type '{ntype}'. Setting samples to [0]*{self.num_hgt_layers}.")
                      self.hgt_num_samples[ntype] = [0] * self.num_hgt_layers
                  elif len(self.hgt_num_samples[ntype]) != self.num_hgt_layers:
                      raise ValueError(f"Length of user-provided hgt_num_samples for '{ntype}' ({len(self.hgt_num_samples[ntype])}) "
                                       f"must match num_hgt_layers ({self.num_hgt_layers})")
        elif not self.use_gnn_encoder:
             self.hgt_num_samples = {} # No sampling needed if GNN is off

        print(f"Initializing TransactionDataModuleV2 (HGTLoader={self.use_gnn_encoder})...")
        print(f"  Modality Flags: GNN={self.use_gnn_encoder}, Sequence={self.use_sequence_encoder}, Text={self.use_text_encoder}")
        print(f"  Optional Label: ScheduleC={self.use_scheduleC_label}")

        # Placeholders
        self.tokenizer = None
        self.full_graph_data: Optional[HeteroData] = None
        self.node_feature_dims: Dict[str, int] = {}
        self.edge_feature_dims: Dict[Tuple[str, str, str], int] = {}
        self.sequence_feature_dim: Optional[int] = None
        self.scalers: Dict[str, StandardScaler] = {}
        self.seq_scalers: Dict[str, StandardScaler] = {}
        self.edge_scalers: Dict[str, StandardScaler] = {}
        self.user_map = None
        self.num_users = 0
        self.num_global_classes = 0
        self.num_user_classes = 0
        self.train_indices: Optional[torch.Tensor] = None
        self.val_indices: Optional[torch.Tensor] = None
        self.test_indices: Optional[torch.Tensor] = None

        # --- Pre-process DataFrame --- 
        start_time = time.time()
        print("Pre-processing DataFrame (datetime, category IDs, user IDs, time features)...")
        # Timestamp handling
        timestamp_col = 'books_create_timestamp' # Using the correct column name
        if timestamp_col not in self.transactions_df.columns:
            print(f"[WARN] Timestamp column '{timestamp_col}' not found. Using default date.")
            self.transactions_df['timestamp'] = pd.Timestamp('2020-01-01')
        else:
            print(f"Converting '{timestamp_col}' column to datetime...")
            self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df[timestamp_col], errors='coerce')
            if self.transactions_df['timestamp'].isnull().any():
                 print(f"[WARN] Coerced {self.transactions_df['timestamp'].isnull().sum()} invalid timestamps to NaT.")
                 median_date = self.transactions_df['timestamp'].dropna().median()
                 if pd.isna(median_date): median_date = pd.Timestamp('2020-01-01')
                 self.transactions_df['timestamp'].fillna(median_date, inplace=True)
            print("Timestamp conversion/handling complete.")
        # Time features
        self.transactions_df['hour'] = self.transactions_df['timestamp'].dt.hour.fillna(0).astype(int)
        self.transactions_df['weekday'] = self.transactions_df['timestamp'].dt.weekday.fillna(0).astype(int)

        # Category ID handling (factorize if object)
        self.category_id_map = None
        if 'category_id' in self.transactions_df.columns:
            if self.transactions_df['category_id'].isnull().any():
                self.transactions_df['category_id'] = self.transactions_df['category_id'].fillna('UNKNOWN_CAT')
            if self.transactions_df['category_id'].dtype == 'object':
                codes, uniques = pd.factorize(self.transactions_df['category_id'], sort=True)
                self.transactions_df['category_id'] = codes
                self.category_id_map = {code: unique_val for code, unique_val in enumerate(uniques)}
                print(f"Factorized 'category_id' into {len(uniques)} unique IDs.")
            else: 
                 self.transactions_df['category_id'] = self.transactions_df['category_id'].astype(int)
            self.num_global_classes = self.transactions_df['category_id'].max() + 1
        else:
            raise ValueError("Missing required column: 'category_id'")

        # User Category ID handling (Simplified: assume numeric or NaN, fill with -1)
        self.user_category_id_map = None 
        if 'user_category_id' in self.transactions_df.columns:
            try:
                if self.transactions_df['user_category_id'].isnull().any():
                    print(f"[WARN] 'user_category_id' contains NaNs. Filling with -1 before int conversion.")
                    self.transactions_df['user_category_id'] = self.transactions_df['user_category_id'].fillna(-1)
                self.transactions_df['user_category_id'] = self.transactions_df['user_category_id'].astype(int)
                valid_labels = self.transactions_df['user_category_id'][self.transactions_df['user_category_id'] >= 0]
                if not valid_labels.empty:
                    self.num_user_classes = valid_labels.max() + 1
                else:
                    self.num_user_classes = 0
            except (ValueError, TypeError) as e:
                print(f"[ERROR] Could not convert 'user_category_id' to int: {e}. Setting num_user_classes to 0.")
                self.num_user_classes = 0
        else:
             print("[WARN] Optional column 'user_category_id' not found. Setting num_user_classes to 0.")
             self.num_user_classes = 0
             
        # User ID handling (factorize for embedding index)
        if 'user_id' not in self.transactions_df.columns:
            raise ValueError("Missing required column: 'user_id'")
        user_codes, user_uniques = pd.factorize(self.transactions_df['user_id'], sort=True)
        self.transactions_df['user_id_code'] = user_codes 
        self.user_map = {code: uid for code, uid in enumerate(user_uniques)}
        self.num_users = len(user_uniques)
        
        print(f"Final Class Counts: Global={self.num_global_classes}, User={self.num_user_classes}, Users={self.num_users}")
        print(f"DataFrame pre-processing finished in {time.time() - start_time:.2f}s")

    def prepare_data(self):
        if self.use_text_encoder:
            print(f"Downloading/loading tokenizer: {self.text_model_name}")
            try:
                 _ = AutoTokenizer.from_pretrained(self.text_model_name)
                 print("Tokenizer ready.")
            except Exception as e:
                 print(f"[ERROR] Failed to download/load tokenizer '{self.text_model_name}': {e}")
                 raise e

    def setup(self, stage: Optional[str] = None):
        if self.full_graph_data is not None and self.train_indices is not None: 
             print("DataModuleV2 already set up.")
             return

        print(f"--- Starting DataModuleV2 Setup for stage: {stage} ---")
        setup_start_time = time.time()
        
        # Initialize tokenizer FIRST if text encoder is used
        if self.tokenizer is None and self.use_text_encoder:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            print("Tokenizer initialized.")

        # Ensure the DataFrame is sorted for consistent splitting and processing
        self.transactions_df.sort_values(['user_id_code', 'timestamp'], inplace=True)
        self.transactions_df.reset_index(drop=True, inplace=True) # Reset index after sort

        # **Step 1: Split data indices (time-based within users)**
        print("Splitting data indices (time-based within users)...")
        self.train_indices, self.val_indices, self.test_indices = self._split_data_indices()
        print(f"Split complete: #Train={len(self.train_indices)}, #Val={len(self.val_indices)}, #Test={len(self.test_indices)}")

        # **Step 2: Build the full graph structure and calculate raw node features**
        # Raw features might be needed by multiple components
        print("Calculating raw features for graph nodes...")
        raw_features = {} # Initialize raw_features
        if self.use_gnn_encoder:
            raw_features = self._calculate_raw_features() # Calculate based on full df

        # **Step 3: Fit scalers ONLY on TRAINING data**
        print("Fitting scalers on TRAINING data...")
        self._fit_scalers(raw_features, self.train_indices) # Pass train_indices

        # **Step 4: Build Graph with SCALED features (using fitted scalers)**
        print("Building full graph data with scaled features...")
        if self.use_gnn_encoder:
             # Pass train_indices to scaling part within build_graph
            self._build_graph_and_edges(raw_features, self.train_indices)
        else:
            self._build_minimal_graph() # Contains essential info like labels, user_id_code

        # **Step 5: Prepare and Add Sequences (uses full graph data)**
        if self.use_sequence_encoder:
            # Pass train_indices for fitting sequence scalers
            self._prepare_and_add_sequences(self.train_indices) 

        # Assign masks to the graph data AFTER it's built
        self._assign_masks_to_graph()
            
        print(f"--- DataModuleV2 Setup finished in {time.time() - setup_start_time:.2f}s ---")

    # --- Helper Methods for Setup ---
    def _calculate_raw_features(self) -> Dict[str, np.ndarray]:
        df = self.transactions_df
        raw_features_dict = {}
        print("Calculating raw transaction features (vectorized)...")
        self.tx_feat_cols_to_scale = ['amount']
        self.tx_feat_cols_no_scale = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        hour = df['hour'].fillna(0).astype(int)
        day = df['weekday'].fillna(0).astype(int)
        amount = df['amount'].fillna(0.0)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        tx_features_array = np.column_stack([amount, hour_sin, hour_cos, day_sin, day_cos])
        raw_features_dict['transaction'] = tx_features_array.astype(np.float64)
        print(f"  Raw transaction features calculated. Shape: {raw_features_dict['transaction'].shape}")
        agg_funcs_named = {
            'amount_mean': ('amount', 'mean'), 'amount_std': ('amount', lambda x: x.std(ddof=0)),
            'amount_max': ('amount', 'max'), 'amount_min': ('amount', 'min'),
            'amount_count': ('amount', 'count'), 'amount_median': ('amount', 'median'),
            'amount_q25': ('amount', lambda x: x.quantile(0.25)), 'amount_q75': ('amount', lambda x: x.quantile(0.75))
        }
        print("Calculating raw merchant features (vectorized)...")
        merchant_ids = df['merchant_name'].dropna().unique()
        num_merchants = len(merchant_ids)
        if num_merchants > 0:
            merchant_stats = df[pd.notna(df['merchant_name'])].groupby('merchant_name').agg(**agg_funcs_named)
            merchant_stats = merchant_stats.fillna(0).reindex(merchant_ids, fill_value=0)
            final_merchant_cols = list(agg_funcs_named.keys())
            raw_features_dict['merchant'] = merchant_stats[final_merchant_cols].values.astype(np.float64)
        else:
            raw_features_dict['merchant'] = np.zeros((0, 8), dtype=np.float64)
        print(f"  Raw merchant features calculated. Shape: {raw_features_dict['merchant'].shape}")
        print("Calculating raw category features (vectorized)...")
        valid_categories = df['category_id'].dropna().unique()
        valid_categories = [c for c in valid_categories if isinstance(c, (int, np.integer)) and c >= 0]
        num_categories = len(valid_categories)
        if num_categories > 0:
            category_stats = df[df['category_id'].isin(valid_categories)].groupby('category_id').agg(**agg_funcs_named)
            category_stats = category_stats.fillna(0).reindex(valid_categories, fill_value=0)
            final_category_cols = list(agg_funcs_named.keys())
            raw_features_dict['category'] = category_stats[final_category_cols].values.astype(np.float64)
        else:
            raw_features_dict['category'] = np.zeros((0, 8), dtype=np.float64)
        print(f"  Raw category features calculated. Shape: {raw_features_dict['category'].shape}")
        return raw_features_dict

    def _fit_scalers(self, raw_features: Dict[str, np.ndarray], train_indices: torch.Tensor):
        print("Fitting scalers (using only training data where applicable)...")
        train_indices_np = train_indices.numpy()

        for node_type, features in raw_features.items():
            if features.size > 0 and features.shape[0] > 0:
                # Fit node scalers only on the training subset of nodes
                if node_type == 'transaction':
                    num_cols_to_scale = len(self.tx_feat_cols_to_scale)
                    if features.shape[1] >= num_cols_to_scale and num_cols_to_scale > 0:
                        scaler = StandardScaler()
                        # Ensure train_indices_np are valid indices for features array
                        valid_train_indices = train_indices_np[train_indices_np < features.shape[0]]
                        if len(valid_train_indices) > 0:
                             scaler.fit(features[valid_train_indices, :num_cols_to_scale])
                             self.scalers[node_type] = scaler
                             print(f"  Fitted scaler for 'transaction' features (first {num_cols_to_scale} cols) using {len(valid_train_indices)} training samples.")
                        else:
                             print(f"  [WARN] No valid training indices found for fitting 'transaction' scaler.")
                             self.scalers[node_type] = None # Indicate scaler couldn't be fitted
                    else:
                         print(f"  [WARN] Not enough columns in transaction features to scale.")
                         self.scalers[node_type] = None
                elif node_type in ['merchant', 'category']: 
                    # NOTE: Merchant/Category nodes don't directly map to train_indices.
                    # We fit these on ALL unique merchants/categories found in the raw data.
                    # This is still a form of leakage, but harder to avoid without complex logic
                    # to only consider merchants/categories touched by training transactions.
                    # For now, we keep fitting these globally.
                    scaler = StandardScaler()
                    scaler.fit(features)
                    self.scalers[node_type] = scaler
                    print(f"  Fitted scaler GLOBALLY for '{node_type}'.")
            else:
                print(f"  Skipping scaler fitting for empty features: '{node_type}'")
                self.scalers[node_type] = None

        # Defer sequence scaler fitting to _prepare_and_add_sequences
        self.seq_scalers['time_delta'] = None 

        # Edge scaler for amount_dist uses transaction scaler's scale value
        if self.use_gnn_encoder and 'transaction' in self.scalers and self.scalers['transaction'] is not None:
             amount_dist_scaler_proxy = StandardScaler()
             amount_dist_scaler_proxy.mean_ = np.array([0.0])
             # Use scale_ from the fitted transaction scaler
             scale_val = np.maximum(self.scalers['transaction'].scale_[0:1], 1e-8) 
             amount_dist_scaler_proxy.scale_ = scale_val
             self.edge_scalers['amount_dist'] = amount_dist_scaler_proxy
             print("  Created proxy edge scaler for 'amount_dist' based on train transaction scaler.")
        else:
             self.edge_scalers['amount_dist'] = None
             print("  [INFO] Skipping scaler fitting for edge 'amount_dist' (transaction scaler not available).")

    def _build_graph_and_edges(self, raw_features: Dict[str, np.ndarray], train_indices: torch.Tensor):
        print("Building graph structure and applying SCALED features...")
        data = HeteroData()
        df = self.transactions_df # Use the sorted, reset_index df
        num_transactions = len(df)
        # Create node index map based on the DataFrame's index (0 to N-1)
        tx_map = {idx: i for i, idx in enumerate(df.index)} 
        
        # Map merchant names to unique integer node indices
        merchant_ids = df['merchant_name'].dropna().unique()
        merchant_map = {name: i for i, name in enumerate(merchant_ids)}
        num_merchants = len(merchant_map)
        
        # Map valid category IDs to unique integer node indices
        valid_categories = df['category_id'].dropna().unique()
        valid_categories = [c for c in valid_categories if isinstance(c, (int, np.integer)) and c >= 0]
        category_map = {cat_id: i for i, cat_id in enumerate(valid_categories)}
        num_categories = len(category_map)

        data['transaction'].num_nodes = num_transactions
        if num_merchants > 0: data['merchant'].num_nodes = num_merchants
        if num_categories > 0: data['category'].num_nodes = num_categories

        print("  Adding scaled node features...")
        for node_type, raw_feat_array in raw_features.items():
            if raw_feat_array.size > 0: 
                final_features = raw_feat_array.copy() # Start with raw features
                if node_type in self.scalers and self.scalers[node_type] is not None:
                    scaler = self.scalers[node_type]
                    if node_type == 'transaction':
                        num_cols_to_scale = len(self.tx_feat_cols_to_scale)
                        # Apply scaler fitted on training data TO ALL transaction nodes
                        scaled_part = (raw_feat_array[:, :num_cols_to_scale] - scaler.mean_) / (scaler.scale_ + 1e-8)
                        non_scaled_part = raw_feat_array[:, num_cols_to_scale:]
                        final_features = np.concatenate([scaled_part, non_scaled_part], axis=1)
                        print(f"    Applied TRAIN-fitted scaler to ALL 'transaction' nodes.")
                    else: # Apply globally fitted scaler to merchant/category nodes
                        final_features = (raw_feat_array - scaler.mean_) / (scaler.scale_ + 1e-8)
                        print(f"    Applied GLOBALLY-fitted scaler to '{node_type}' nodes.")
                    
                    data[node_type].x = torch.tensor(final_features, dtype=torch.float)
                    self.node_feature_dims[node_type] = data[node_type].x.shape[1]
                else: 
                    # Use raw features if scaler wasn't fitted or is None
                    print(f"    Using RAW features for '{node_type}' (scaler not available or fitted).")
                    data[node_type].x = torch.tensor(raw_feat_array, dtype=torch.float)
                    self.node_feature_dims[node_type] = data[node_type].x.shape[1]
            else:
                 print(f"    Skipping node features for empty raw features: '{node_type}'")
        
        print("  Adding labels, user IDs, original indices...")
        # Use df.index directly as original_index if df index is 0 to N-1
        data['transaction'].original_index = torch.tensor(df.index.values, dtype=torch.long) 
        data['transaction'].y_global = torch.tensor(df['category_id'].values, dtype=torch.long)
        if 'user_category_id' in df:
            data['transaction'].y_user = torch.tensor(df['user_category_id'].values, dtype=torch.long)
        data['transaction'].user_id_code = torch.tensor(df['user_id_code'].values, dtype=torch.long)
        
        text_cols = ['raw_description', 'memo', 'merchant_name']
        present_text_cols = [col for col in text_cols if col in df.columns]
        if present_text_cols:
            data['transaction']._raw_text = df[present_text_cols].astype(str).apply(lambda x: ' || '.join(x), axis=1).tolist()
        else:
            data['transaction']._raw_text = ['' for _ in range(len(df))] 

        print("  Creating edges and edge features...")
        edge_index_dict = {}
        edge_attr_dict = {}
        
        # Edge calculations remain largely the same, using the node indices from maps
        # 1. Transaction -> Merchant
        if num_merchants > 0:
            edge_list, attr_list = [], []
            merchant_stats_map = df.groupby('merchant_name')['amount'].agg(['mean', 'std']).fillna(0).to_dict('index')
            for idx, row in df.iterrows():
                merchant_name = row['merchant_name']
                if pd.notna(merchant_name) and merchant_name in merchant_map:
                    tx_node_idx = tx_map[idx]; merchant_node_idx = merchant_map[merchant_name]
                    edge_list.append([tx_node_idx, merchant_node_idx])
                    stats = merchant_stats_map.get(merchant_name, {'mean': 0, 'std': 0})
                    amount_zscore = (row['amount'] - stats['mean']) / (stats['std'] + 1e-8)
                    attr_list.append([amount_zscore])
            if edge_list:
                edge_index_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_attr_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(attr_list, dtype=torch.float)
                self.edge_feature_dims[('transaction', 'belongs_to', 'merchant')] = 1

        # 2. Merchant -> Category
        if num_merchants > 0 and num_categories > 0:
            edge_list, attr_list = [], []
            merchant_groups = df.groupby('merchant_name')['category_id']
            merchant_primary_category = merchant_groups.agg(lambda x: x.mode()[0] if not x.mode().empty else -1)
            merchant_category_confidence = merchant_groups.agg(lambda x: x.value_counts(normalize=True).max() if not x.empty else 0)
            for merchant_name, primary_cat_id in merchant_primary_category.items():
                if merchant_name in merchant_map and primary_cat_id in category_map:
                    merchant_node_idx = merchant_map[merchant_name]; category_node_idx = category_map[primary_cat_id]
                    edge_list.append([merchant_node_idx, category_node_idx])
                    attr_list.append([merchant_category_confidence.get(merchant_name, 0)])
            if edge_list:
                edge_index_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_attr_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(attr_list, dtype=torch.float)
                self.edge_feature_dims[('merchant', 'categorized_as', 'category')] = 1

        # 3. Transaction -> Transaction (temporal) 
        edge_list, attr_list = [], []
        # df is already sorted by user, then timestamp
        # tx_map maps df index (0..N-1) to node index (0..N-1)
        for i in range(num_transactions):
            ts_i = df.iloc[i]['timestamp']
            user_i = df.iloc[i]['user_id_code']
            if pd.isna(ts_i): continue 
            tx_node_i = tx_map[df.index[i]] # Get node index for row i
            # Look ahead only within the same user
            for k in range(1, 6): 
                j = i + k
                if j >= num_transactions: break
                user_j = df.iloc[j]['user_id_code']
                if user_j != user_i: break # Stop if we reach next user
                ts_j = df.iloc[j]['timestamp']
                if pd.isna(ts_j): continue
                time_diff_seconds = abs((ts_i - ts_j).total_seconds())
                if time_diff_seconds <= 86400 * 1: 
                    tx_node_j = tx_map[df.index[j]] # Get node index for row j
                    edge_list.extend([[tx_node_i, tx_node_j], [tx_node_j, tx_node_i]])
                    time_diff_norm = min(time_diff_seconds / 86400.0, 1.0)
                    attr_list.extend([[time_diff_norm, 1.0], [time_diff_norm, 0.0]]) # Add direction
                else: # Optimization: if time diff > 1 day, subsequent diffs will also be > 1 day
                    break 
        if edge_list:
            edge_index_dict[('transaction', 'temporal', 'transaction')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr_dict[('transaction', 'temporal', 'transaction')] = torch.tensor(attr_list, dtype=torch.float)
            self.edge_feature_dims[('transaction', 'temporal', 'transaction')] = 2

        # 4. Transaction -> Transaction (similar_amount)
        edge_list, raw_dist_list, amount_ratio_list, direction_list = [], [], [], []
        # Get raw amounts from the correct place (assuming 'transaction' node features exist)
        raw_tx_amounts = data['transaction'].x[:, 0].numpy() if 'transaction' in data and data['transaction'].x is not None else np.array([])

        if num_transactions > 1 and raw_tx_amounts.size > 0:
            try:
                 k_neighbors = min(5, num_transactions - 1)
                 nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='minkowski', p=1, algorithm='auto')
                 nn.fit(raw_tx_amounts.reshape(-1, 1))
                 distances, indices = nn.kneighbors(raw_tx_amounts.reshape(-1, 1))
                 for i in range(num_transactions):
                     tx_node_i = tx_map[df.index[i]] 
                     # Only connect nodes within the same user? - NO, KNN is global amount similarity
                     for k in range(1, k_neighbors + 1):
                         j_pos = indices[i, k]
                         if j_pos < num_transactions: 
                            tx_node_j = tx_map[df.index[j_pos]]
                            # Ensure i != j_pos to avoid self-loops from KNN
                            if tx_node_i == tx_node_j: continue 
                            dist = distances[i, k]
                            edge_list.extend([[tx_node_i, tx_node_j], [tx_node_j, tx_node_i]])
                            raw_dist_list.extend([dist, dist])
                            amount_i = raw_tx_amounts[i]; amount_j = raw_tx_amounts[j_pos]
                            amount_ratio = min(amount_i, amount_j) / (max(amount_i, amount_j) + 1e-8) if max(amount_i, amount_j) > 1e-8 else 1.0
                            amount_ratio_list.extend([amount_ratio, amount_ratio])
                            direction_list.extend([1.0, 0.0])
            except Exception as e_knn:
                 print(f"[ERROR] KNN for similar_amount failed: {e_knn}")
        
        if edge_list:
            scaled_dist_values = None
            # Use the proxy scaler created in _fit_scalers (based on train transaction scale)
            if 'amount_dist' in self.edge_scalers and self.edge_scalers['amount_dist'] is not None:
                try:
                    scaler = self.edge_scalers['amount_dist']
                    raw_dist_array = np.array(raw_dist_list, dtype=np.float64).reshape(-1, 1)
                    # Apply scaling: (dist - 0) / scale
                    scaled_dist = (raw_dist_array / (scaler.scale_ + 1e-8)).flatten() 
                    print("  Applied TRAIN-based scaler to 'amount_dist' edge feature.")
                    scaled_dist_values = scaled_dist
                except Exception as e_scale:
                    print(f"[WARN] Failed to scale amount_dist: {e_scale}. Using raw distances.")
            if scaled_dist_values is None:
                scaled_dist_values = np.array(raw_dist_list) # Fallback to raw
            
            # Combine attributes
            if len(scaled_dist_values) == len(amount_ratio_list) == len(direction_list):
                final_attr_list = [[scaled_dist_values[idx], amount_ratio_list[idx], direction_list[idx]]
                                   for idx in range(len(scaled_dist_values))]
                edge_index_dict[('transaction', 'similar_amount', 'transaction')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_attr_dict[('transaction', 'similar_amount', 'transaction')] = torch.tensor(final_attr_list, dtype=torch.float)
                self.edge_feature_dims[('transaction', 'similar_amount', 'transaction')] = 3
            else:
                print(f"[WARN] Length mismatch in similar_amount attributes.")

        # Assign edges to graph
        print("  Assigning final edge data...")
        for edge_type, index_tensor in edge_index_dict.items():
            data[edge_type].edge_index = index_tensor
            if edge_type in edge_attr_dict:
                data[edge_type].edge_attr = edge_attr_dict[edge_type]
        self.full_graph_data = data 
        print("Graph building complete.")

    def _build_minimal_graph(self):
        print("[INFO] Building minimal graph data...")
        self.full_graph_data = HeteroData()
        df = self.transactions_df
        num_nodes = len(df)
        self.full_graph_data['transaction'].num_nodes = num_nodes
        self.full_graph_data['transaction'].original_index = torch.tensor(df.index.values, dtype=torch.long)
        self.full_graph_data['transaction'].y_global = torch.tensor(df['category_id'].values, dtype=torch.long)
        if 'user_category_id' in df:
            self.full_graph_data['transaction'].y_user = torch.tensor(df['user_category_id'].values, dtype=torch.long)
        self.full_graph_data['transaction'].user_id_code = torch.tensor(df['user_id_code'].values, dtype=torch.long)
        text_cols = ['raw_description', 'memo', 'merchant_name']
        present_text_cols = [col for col in text_cols if col in df.columns]
        if present_text_cols:
            self.full_graph_data['transaction']._raw_text = df[present_text_cols].astype(str).apply(lambda x: ' || '.join(x), axis=1).tolist()
        else:
             self.full_graph_data['transaction']._raw_text = ['' for _ in range(num_nodes)]
        print("Built minimal graph data.")

    def _prepare_and_add_sequences(self, train_indices: torch.Tensor):
        if self.full_graph_data is None: raise RuntimeError("Graph data not built.")
        if not self.use_sequence_encoder: 
            print("[INFO] Skipping sequence preparation as use_sequence_encoder=False")
            return 
        if hasattr(self.full_graph_data['transaction'], 'seq_features'): return 
             
        print("Preparing sequence features...")
        # df should already be sorted by user_id_code, timestamp from setup()
        df_sorted = self.transactions_df 
        train_indices_np = train_indices.numpy()

        # --- Fit sequence scalers only on training data portions ---
        print("  Fitting sequence scalers on TRAINING data...")
        all_train_time_deltas = []
        # Need to iterate through sequences corresponding to train_indices
        original_to_current_pos = {idx: i for i, idx in enumerate(df_sorted.index)}

        for node_idx in train_indices_np:
             orig_idx = node_idx # Assuming node index matches DataFrame index after reset
             current_pos_in_sorted = original_to_current_pos.get(orig_idx)
             if current_pos_in_sorted is None: continue

             current_row = df_sorted.iloc[current_pos_in_sorted]
             user_id_code = current_row['user_id_code']; current_time = current_row['timestamp']
             start_idx_in_sorted = max(0, current_pos_in_sorted - self.max_seq_length)
             prev_txs_window_df = df_sorted.iloc[start_idx_in_sorted:current_pos_in_sorted]
             prev_txs_user_df = prev_txs_window_df[prev_txs_window_df['user_id_code'] == user_id_code]

             if not prev_txs_user_df.empty:
                 # Calculate time deltas within this training sequence
                 if pd.notna(current_time):
                     time_diffs = (current_time - prev_txs_user_df['timestamp']).dt.total_seconds()
                     valid_time_diffs = time_diffs[pd.notna(time_diffs)].clip(lower=0).tolist()
                     all_train_time_deltas.extend(valid_time_diffs)
        
        # Fit time_delta scaler
        if all_train_time_deltas:
             time_delta_array = np.array(all_train_time_deltas).reshape(-1, 1)
             scaler_td = StandardScaler().fit(time_delta_array)
             self.seq_scalers['time_delta'] = scaler_td
             print(f"    Fitted time_delta scaler on training sequences.")
        else:
             print(f"    [WARN] No valid time deltas found in training sequences to fit scaler.")
             self.seq_scalers['time_delta'] = None

        # --- Generate sequences for ALL transactions (apply fitted scalers) ---
        print("  Generating sequences for ALL transactions...")
        all_seq_features = []
        all_seq_cat_features = [] # New list for categorical features
        all_seq_lengths = []
        self.sequence_feature_dim = 6 # Real features
        self.num_regions = 0 # Initialize cardinality for region_id

        # Factorize region_id if it exists and is not numeric, handle NaN
        region_map = None
        if 'region_id' in df_sorted.columns:
            df_sorted['region_id_proc'] = df_sorted['region_id'].fillna(-1) # Fill NaN with -1
            # Factorize for consistent 0-based indexing, handles both string/object and numeric
            codes, uniques = pd.factorize(df_sorted['region_id_proc'], sort=True)
            df_sorted['region_id_proc'] = codes
            region_map = {code: val for code, val in enumerate(uniques)}
            self.num_regions = len(uniques)
            print(f"  Processed 'region_id'. Cardinality: {self.num_regions}")
        else:
            print("  [WARN] 'region_id' column not found. Skipping sequence categorical feature.")
            df_sorted['region_id_proc'] = 0 # Assign dummy value if column missing
            self.num_regions = 1 # Only the dummy value

        original_to_sorted_pos = {idx: i for i, idx in enumerate(df_sorted.index)}

        for node_idx in range(num_transactions):
            orig_idx = orig_indices_tensor[node_idx].item()
            current_pos_in_sorted = original_to_sorted_pos.get(orig_idx)
            if current_pos_in_sorted is None: 
                 seq_tensor = torch.zeros((0, self.sequence_feature_dim), dtype=torch.float); seq_len = 0
                 all_seq_features.append(seq_tensor); all_seq_cat_features.append(torch.zeros((0, 1), dtype=torch.long)); all_seq_lengths.append(seq_len)
                 continue
                 
            current_row = df_sorted.iloc[current_pos_in_sorted]
            user_id_code = current_row['user_id_code']; current_time = current_row['timestamp']
            start_idx_in_sorted = max(0, current_pos_in_sorted - self.max_seq_length)
            prev_txs_window_df = df_sorted.iloc[start_idx_in_sorted:current_pos_in_sorted]
            prev_txs_user_df = prev_txs_window_df[prev_txs_window_df['user_id_code'] == user_id_code]

            seq_features_for_tx = []
            seq_cat_features_for_tx = [] # New list for current sequence

            if not prev_txs_user_df.empty:
                for _, prev_row in prev_txs_user_df.iterrows():
                    time_delta = 0.0
                    if pd.notna(current_time) and pd.notna(prev_row['timestamp']):
                         time_delta_seconds = (current_time - prev_row['timestamp']).total_seconds()
                         time_delta = max(0.0, time_delta_seconds) 
                    amount = prev_row['amount'] if pd.notna(prev_row['amount']) else 0.0
                    if 'amount' in self.seq_scalers:
                        scaler_a = self.seq_scalers['amount']
                        amount = (amount - scaler_a.mean_[0]) / (scaler_a.scale_[0] + 1e-8)
                    time_delta_val = time_delta
                    if 'time_delta' in self.seq_scalers and self.seq_scalers['time_delta'] is not None:
                         scaler_td = self.seq_scalers['time_delta']
                         time_delta_val = (time_delta - scaler_td.mean_[0]) / (scaler_td.scale_[0] + 1e-8)
                    hour = prev_row['hour']; day = prev_row['weekday']
                    hour_sin = np.sin(2*np.pi*hour/24); hour_cos = np.cos(2*np.pi*hour/24)
                    day_sin = np.sin(2*np.pi*day/7); day_cos = np.cos(2*np.pi*day/7)
                    scaled_feat = [amount, day_sin, day_cos, hour_sin, hour_cos, time_delta_val]
                    seq_features_for_tx.append(scaled_feat)

                    # --- Add Categorical Feature (region_id) --- 
                    region_code = prev_row['region_id_proc']
                    seq_cat_features_for_tx.append([region_code]) # Append as a list/singleton feature

            if seq_features_for_tx: # Check if real features were added
                seq_tensor = torch.tensor(seq_features_for_tx, dtype=torch.float)
                seq_cat_tensor = torch.tensor(seq_cat_features_for_tx, dtype=torch.long) # Use long for categorical IDs
                
                # Apply max_seq_length limit
                if seq_tensor.shape[0] > self.max_seq_length:
                     seq_tensor = seq_tensor[-self.max_seq_length:, :]
                     seq_cat_tensor = seq_cat_tensor[-self.max_seq_length:, :]
                seq_len = len(seq_tensor)
            else:
                seq_tensor = torch.zeros((0, self.sequence_feature_dim), dtype=torch.float)
                seq_cat_tensor = torch.zeros((0, 1), dtype=torch.long) # Match cat feature dim (1)
                seq_len = 0
                
            all_seq_features.append(seq_tensor)
            all_seq_cat_features.append(seq_cat_tensor) # Add padded cat tensor
            all_seq_lengths.append(seq_len)

        max_len_found = max(all_seq_lengths) if all_seq_lengths else 0
        print(f"  Max sequence length found: {max_len_found}")
        if not all_seq_features:
             padded_sequences = torch.empty((num_transactions, 0, self.sequence_feature_dim), dtype=torch.float)
        else:
             padded_sequences = pad_sequence(all_seq_features, batch_first=True, padding_value=0.0)
        
        # Pad categorical features (padding_value usually 0, assuming 0 is a valid code or reserved)
        if not all_seq_cat_features:
             padded_cat_sequences = torch.empty((num_transactions, 0, 1), dtype=torch.long)
        else:
             padded_cat_sequences = pad_sequence(all_seq_cat_features, batch_first=True, padding_value=0) # Pad with 0

        self.full_graph_data['transaction'].seq_features = padded_sequences
        self.full_graph_data['transaction'].seq_cat_features = padded_cat_sequences # Add cat features
        self.full_graph_data['transaction'].seq_lengths = torch.tensor(all_seq_lengths, dtype=torch.long)
        print(f"Added sequence features. Padded reals shape: {padded_sequences.shape}, Padded cats shape: {padded_cat_sequences.shape}")

    def _split_data_indices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        df = self.transactions_df # Assumes sorted by user, timestamp
        print("Creating train/val/test indices: Time-based split within each user...")
        
        all_train_indices = []
        all_val_indices = []
        all_test_indices = []

        # Group by user
        user_groups = df.groupby('user_id_code', sort=False) # Use sort=False as df is already sorted

        for user_code, group in user_groups:
            n_transactions = len(group)
            indices = group.index.tolist() # Get DataFrame indices (0 to N-1)

            if n_transactions < 3: # Assign all to train if too few transactions
                all_train_indices.extend(indices)
            else:
                n_train = int(n_transactions * (1.0 - self.val_ratio - self.test_ratio))
                n_val = int(n_transactions * self.val_ratio)
                # Ensure n_train and n_val are at least 1 if ratios > 0
                if self.val_ratio > 0 and n_val == 0: n_val = 1
                if (1.0 - self.val_ratio - self.test_ratio) > 0 and n_train == 0: n_train = 1
                # Ensure n_train + n_val doesn't exceed total
                if n_train + n_val >= n_transactions:
                    n_val = max(0, n_transactions - n_train) # Prioritize training data

                n_test = n_transactions - n_train - n_val

                # Split indices based on time order (since df is sorted)
                all_train_indices.extend(indices[:n_train])
                all_val_indices.extend(indices[n_train : n_train + n_val])
                all_test_indices.extend(indices[n_train + n_val :])

        # Convert lists to tensors
        train_indices_tensor = torch.tensor(sorted(all_train_indices), dtype=torch.long)
        val_indices_tensor = torch.tensor(sorted(all_val_indices), dtype=torch.long)
        test_indices_tensor = torch.tensor(sorted(all_test_indices), dtype=torch.long)

        # Store split sizes
        self.num_train_samples = len(train_indices_tensor)
        self.num_val_samples = len(val_indices_tensor)
        self.num_test_samples = len(test_indices_tensor)

        print(f"Split Method: Time-based within users (approx ratios: "
              f"Tr={1-self.val_ratio-self.test_ratio:.2f}, "
              f"V={self.val_ratio:.2f}, Te={self.test_ratio:.2f})")
        
        # Return the tensors containing the DataFrame indices for each split
        return train_indices_tensor, val_indices_tensor, test_indices_tensor

    # New helper to assign masks AFTER graph is built
    def _assign_masks_to_graph(self):
        if self.full_graph_data is None or not hasattr(self.full_graph_data['transaction'], 'num_nodes'):
            print("[WARN] Cannot assign masks, graph data not available.")
            return
        
        num_nodes = self.full_graph_data['transaction'].num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if self.train_indices is not None:
            train_mask[self.train_indices] = True
        if self.val_indices is not None:
            val_mask[self.val_indices] = True
        if self.test_indices is not None:
            test_mask[self.test_indices] = True

        self.full_graph_data['transaction'].train_mask = train_mask
        self.full_graph_data['transaction'].val_mask = val_mask
        self.full_graph_data['transaction'].test_mask = test_mask
        print("Assigned train/val/test masks to graph data.")
        
    # --- Dataloader Methods using HGTLoader --- 
    def train_dataloader(self) -> HGTLoader:
        print("Creating train HGTLoader...")
        if self.train_indices is None: self.setup('fit')
        if self.full_graph_data is None: raise RuntimeError("Graph data not available.")
        if self.train_indices is None or len(self.train_indices) == 0: raise ValueError("Training set empty.")
        valid_hgt_num_samples = {k:v for k,v in self.hgt_num_samples.items() if k in self.full_graph_data.node_types}
        if not valid_hgt_num_samples: raise ValueError("No valid node types for HGT sampling.")
        return HGTLoader(self.full_graph_data.cpu(), num_samples=valid_hgt_num_samples, shuffle=True,
                         input_nodes=('transaction', self.train_indices.cpu()), batch_size=self.batch_size,
                         num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def val_dataloader(self) -> HGTLoader:
        print("Creating val HGTLoader...")
        if self.val_indices is None: self.setup('fit')
        if self.full_graph_data is None: raise RuntimeError("Graph data not available.")
        if self.val_indices is None or len(self.val_indices) == 0: raise ValueError("Validation set empty.")
        valid_hgt_num_samples = {k:v for k,v in self.hgt_num_samples.items() if k in self.full_graph_data.node_types}
        if not valid_hgt_num_samples: raise ValueError("No valid node types for HGT sampling.")
        return HGTLoader(self.full_graph_data.cpu(), num_samples=valid_hgt_num_samples, shuffle=False,
                         input_nodes=('transaction', self.val_indices.cpu()), batch_size=self.batch_size,
                         num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def test_dataloader(self) -> HGTLoader:
        print("Creating test HGTLoader...")
        if self.test_indices is None: self.setup('test')
        if self.full_graph_data is None: raise RuntimeError("Graph data not available.")
        if self.test_indices is None or len(self.test_indices) == 0: raise ValueError("Test set empty.")
        valid_hgt_num_samples = {k:v for k,v in self.hgt_num_samples.items() if k in self.full_graph_data.node_types}
        if not valid_hgt_num_samples: raise ValueError("No valid node types for HGT sampling.")
        return HGTLoader(self.full_graph_data.cpu(), num_samples=valid_hgt_num_samples, shuffle=False,
                         input_nodes=('transaction', self.test_indices.cpu()), batch_size=self.batch_size,
                         num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))


