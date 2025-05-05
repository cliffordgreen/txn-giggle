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
import json # Added for parsing chart_of_accounts if needed
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
    Adapts to new data format with company_name, txn_accepted_category_id_str, etc.
    Removes 'category' node type.
    """
    def __init__(self,
                 transactions_df_ref: pd.DataFrame, # Changed from transactions_df
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
                 # --- Arguments for loading pre-fitted state --- 
                 fitted_scalers: Optional[Dict[str, StandardScaler]] = None,
                 fitted_seq_scalers: Optional[Dict[str, StandardScaler]] = None,
                 fitted_edge_scalers: Optional[Dict[str, StandardScaler]] = None,
                 fitted_user_map: Optional[Dict[int, Any]] = None,
                 fitted_category_id_map: Optional[Dict[int, Any]] = None
                 ):
        super().__init__()
        # Expects preprocessed df from train_new.load_data
        self.transactions_df = transactions_df_ref.copy() 
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

        # Store/Create HGT sampling config
        self.num_hgt_layers = num_hgt_layers
        self.hgt_num_samples = hgt_num_samples
        if self.hgt_num_samples is None:
            # Simple default: Sample 15 neighbors in first layer, 10 in second
            default_samples_per_layer = [15, 10]
            # Define KNOWN node types used in the graph (NO 'category')
            node_types_in_graph = ['transaction', 'merchant'] 
            self.hgt_num_samples = {ntype: default_samples_per_layer[:self.num_hgt_layers] for ntype in node_types_in_graph}
            print(f"[WARN] hgt_num_samples not provided. Using default based on num_hgt_layers={self.num_hgt_layers}: {self.hgt_num_samples}")
        else:
            # Validate provided config against expected node types
            expected_node_types = {'transaction', 'merchant'} # Only these are expected now
            for ntype, samples in self.hgt_num_samples.items():
                if ntype not in expected_node_types:
                    print(f"[WARN] hgt_num_samples contains unexpected node type '{ntype}'. It will be ignored.")
                    continue # Don't validate length if type is unused
                if len(samples) != self.num_hgt_layers:
                    raise ValueError(f"Length of hgt_num_samples for '{ntype}' ({len(samples)}) must match num_hgt_layers ({self.num_hgt_layers})")
            # Ensure required node types are present if GNN is used
            if self.use_gnn_encoder:
                for req_type in expected_node_types:
                    if req_type not in self.hgt_num_samples:
                         print(f"[WARN] hgt_num_samples missing required node type '{req_type}' for GNN. Adding default.")
                         default_samples_per_layer = [15, 10]
                         self.hgt_num_samples[req_type] = default_samples_per_layer[:self.num_hgt_layers]
        
        print(f"Initializing TransactionDataModuleV2 (HGTLoader - No Category Node)...")
        print(f"  Modality Flags: GNN={self.use_gnn_encoder}, Sequence={self.use_sequence_encoder}, Text={self.use_text_encoder}")

        # Store pre-fitted state if provided
        self.fitted_scalers = fitted_scalers
        self.fitted_seq_scalers = fitted_seq_scalers
        self.fitted_edge_scalers = fitted_edge_scalers
        self.fitted_user_map = fitted_user_map
        self.fitted_category_id_map = fitted_category_id_map
        self.is_predicting = (fitted_scalers is not None) # Flag if we are in prediction mode

        # Placeholders
        self.tokenizer = None
        self.full_graph_data: Optional[HeteroData] = None
        self.node_feature_dims: Dict[str, int] = {}
        self.edge_feature_dims: Dict[Tuple[str, str, str], int] = {}
        self.sequence_feature_dim: Optional[int] = None
        self.scalers: Dict[str, StandardScaler] = {} if fitted_scalers is None else fitted_scalers
        self.seq_scalers: Dict[str, StandardScaler] = {} if fitted_seq_scalers is None else fitted_seq_scalers
        self.edge_scalers: Dict[str, StandardScaler] = {} if fitted_edge_scalers is None else fitted_edge_scalers
        self.user_map = None if fitted_user_map is None else fitted_user_map
        self.category_id_map = None if fitted_category_id_map is None else fitted_category_id_map
        self.num_users = 0
        self.num_global_classes = 0
        self.num_user_classes = 0 # No user-specific target assumed
        self.train_indices: Optional[torch.Tensor] = None
        self.val_indices: Optional[torch.Tensor] = None
        self.test_indices: Optional[torch.Tensor] = None

        # --- Pre-process DataFrame (simplified as most done in load_data) --- 
        start_time = time.time()
        print("Pre-processing DataFrame (ID mapping)...")
        # Timestamp, amount, time features assumed present from load_data

        # Target Category ID handling (factorize the string ID)
        target_col = 'txn_accepted_category_id_str'
        if target_col not in self.transactions_df.columns:
            raise ValueError(f"Missing required target column: '{target_col}'")
        # Ensure fillna happened in load_data
        if self.transactions_df[target_col].isnull().any():
            print(f"[WARN] Target column '{target_col}' still contains NaNs after load_data. Filling with 'UNKNOWN'.")
            self.transactions_df[target_col] = self.transactions_df[target_col].fillna('UNKNOWN')
            
        # Use fitted maps if provided (predicting), otherwise factorize
        if self.is_predicting:
            print("  Using provided category_id and user maps.")
            # Map strings to known ints, use -1 for unknowns
            map_cat_str_to_int = {v: k for k, v in self.category_id_map.items()}
            map_user_str_to_int = {v: k for k, v in self.user_map.items()}
            self.transactions_df['category_id'] = self.transactions_df[target_col].map(map_cat_str_to_int).fillna(-1).astype(int)
            self.transactions_df['user_id_code'] = self.transactions_df['company_name'].map(map_user_str_to_int).fillna(-1).astype(int)
            self.num_global_classes = len(self.category_id_map)
            self.num_users = len(self.user_map)
            if (self.transactions_df['user_id_code'] == -1).any():
                print(f"[WARN] Found {(self.transactions_df['user_id_code'] == -1).sum()} users in prediction data not present in training user_map.")
        else:
            print("  Factorizing category_id and user maps from data.")
            codes, uniques = pd.factorize(self.transactions_df[target_col], sort=True)
            self.transactions_df['category_id'] = codes 
            self.category_id_map = {code: unique_val for code, unique_val in enumerate(uniques)}
            self.num_global_classes = len(uniques)
            
            user_codes, user_uniques = pd.factorize(self.transactions_df['company_name'], sort=True)
            self.transactions_df['user_id_code'] = user_codes 
            self.user_map = {code: uid for code, uid in enumerate(user_uniques)}
            self.num_users = len(user_uniques)
        
        print(f"Final Counts: Global Classes={self.num_global_classes}, Users={self.num_users}")
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
        # Added check for prediction stage
        is_predict_stage = (stage == 'predict')
        if is_predict_stage and not self.is_predicting:
            raise ValueError("Setup stage is 'predict', but no pre-fitted state was provided during __init__.")
        if not is_predict_stage and self.is_predicting:
             print("[WARN] Pre-fitted state was provided during __init__, but setup stage is not 'predict'. Using fitted state anyway.")
             # Allow using fitted state even if stage is fit/test for simplicity

        if self.full_graph_data is not None and self.train_indices is not None: # Check if already set up for fit/test
             if not is_predict_stage:
                 print("DataModuleV2 already set up for fit/test.")
                 return
             # Allow re-setup for predict stage if needed, but maybe graph exists?
             # If graph exists, maybe just assign test mask?
             if self.test_indices is not None:
                  print("DataModuleV2 graph exists, assigning predict mask.")
                  self._assign_masks_to_graph(predict_only=True) # New flag needed
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

        # **Step 1: Split data indices (or assign for prediction)**
        if is_predict_stage:
            print("Assigning all indices to test set for prediction...")
            self.train_indices = torch.tensor([], dtype=torch.long)
            self.val_indices = torch.tensor([], dtype=torch.long)
            self.test_indices = torch.arange(len(self.transactions_df), dtype=torch.long)
            print(f"Split complete (predict): #Test={len(self.test_indices)}")
        else:
            print("Splitting data indices (time-based within users)...")
            self.train_indices, self.val_indices, self.test_indices = self._split_data_indices()
            print(f"Split complete: #Train={len(self.train_indices)}, #Val={len(self.val_indices)}, #Test={len(self.test_indices)}")

        # **Step 2: Build the full graph structure and calculate raw node features**
        print("Calculating raw features for graph nodes...")
        raw_features = {} 
        if self.use_gnn_encoder:
            raw_features = self._calculate_raw_features() 

        # **Step 3: Fit scalers ONLY on TRAINING data (Skip if predicting)**
        if not self.is_predicting: # Only fit if not using pre-fitted state
            print("Fitting scalers on TRAINING data...")
            self._fit_scalers(raw_features, self.train_indices) 
        else:
            print("Skipping scaler fitting (using pre-loaded scalers).")
            # Ensure edge scaler exists if needed and not provided
            if self.use_gnn_encoder and 'amount_dist' not in self.edge_scalers: 
                 self._create_proxy_edge_scaler() # New helper needed

        # **Step 4: Build Graph with SCALED features (using fitted scalers)**
        print("Building full graph data with scaled features...")
        if self.use_gnn_encoder:
             # Pass train_indices (needed for amount_dist scaling even in predict? Check _build_graph)
             # _build_graph_and_edges uses self.scalers which are now populated
            self._build_graph_and_edges(raw_features, self.train_indices) 
        else:
            self._build_minimal_graph() 

        # **Step 5: Prepare and Add Sequences (uses full graph data)**
        if self.use_sequence_encoder:
            # Pass train_indices for fitting sequence scalers if not predicting
            # _prepare_and_add_sequences needs update to handle self.is_predicting
            self._prepare_and_add_sequences(self.train_indices) 

        # Assign masks to the graph data AFTER it's built
        self._assign_masks_to_graph(predict_only=is_predict_stage)
            
        print(f"--- DataModuleV2 Setup finished in {time.time() - setup_start_time:.2f}s ---")

    # --- Helper Methods for Setup ---
    def _calculate_raw_features(self) -> Dict[str, np.ndarray]:
        df = self.transactions_df
        raw_features_dict = {}
        print("Calculating raw transaction features (vectorized)...")
        # Features: amount, num_chart_of_accounts (scaled), time features (not scaled)
        self.tx_feat_cols_to_scale = ['amount', 'num_chart_of_accounts'] 
        self.tx_feat_cols_no_scale = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        
        # Ensure required columns exist
        if 'amount' not in df.columns: raise ValueError("Missing 'amount' column")
        if 'hour' not in df.columns: raise ValueError("Missing 'hour' column")
        if 'weekday' not in df.columns: raise ValueError("Missing 'weekday' column")
        if 'num_chart_of_accounts' not in df.columns: raise ValueError("Missing 'num_chart_of_accounts' column")

        amount = df['amount'].fillna(0.0)
        num_coa = df['num_chart_of_accounts'].fillna(0).astype(int)
        hour = df['hour'].fillna(0).astype(int)
        day = df['weekday'].fillna(0).astype(int)
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        # Stack scaled features first, then non-scaled
        tx_features_array = np.column_stack([amount, num_coa, hour_sin, hour_cos, day_sin, day_cos])
        raw_features_dict['transaction'] = tx_features_array.astype(np.float64)
        print(f"  Raw transaction features calculated. Shape: {raw_features_dict['transaction'].shape}")
        
        # Define aggregation functions (can reuse)
        agg_funcs_named = {
            'amount_mean': ('amount', 'mean'), 'amount_std': ('amount', lambda x: x.std(ddof=0)),
            'amount_max': ('amount', 'max'), 'amount_min': ('amount', 'min'),
            'amount_count': ('amount', 'count'), 'amount_median': ('amount', 'median'),
            'amount_q25': ('amount', lambda x: x.quantile(0.25)), 'amount_q75': ('amount', lambda x: x.quantile(0.75))
        }
        
        print("Calculating raw merchant features (vectorized)... using 'merchant_name'")
        merchant_col = 'merchant_name' # Use the column derived from payee
        if merchant_col not in df.columns: raise ValueError("Missing 'merchant_name' column")
        merchant_ids = df[merchant_col].dropna().unique()
        num_merchants = len(merchant_ids)
        if num_merchants > 0:
            merchant_stats = df[pd.notna(df[merchant_col])].groupby(merchant_col).agg(**agg_funcs_named)
            merchant_stats = merchant_stats.fillna(0).reindex(merchant_ids, fill_value=0)
            final_merchant_cols = list(agg_funcs_named.keys())
            raw_features_dict['merchant'] = merchant_stats[final_merchant_cols].values.astype(np.float64)
        else:
            raw_features_dict['merchant'] = np.zeros((0, 8), dtype=np.float64)
        print(f"  Raw merchant features calculated. Shape: {raw_features_dict['merchant'].shape}")
        
        # Category features are removed
        print("Skipping category features (node type removed).")
        
        return raw_features_dict

    def _fit_scalers(self, raw_features: Dict[str, np.ndarray], train_indices: torch.Tensor):
        print("Fitting scalers (using only training data where applicable)...")
        train_indices_np = train_indices.numpy()

        for node_type, features in raw_features.items():
            if features.size > 0 and features.shape[0] > 0:
                if node_type == 'transaction':
                    num_cols_to_scale = len(self.tx_feat_cols_to_scale) # amount, num_coa
                    if features.shape[1] >= num_cols_to_scale and num_cols_to_scale > 0:
                        scaler = StandardScaler()
                        # Ensure train_indices_np are valid indices for features array
                        valid_train_indices = train_indices_np[train_indices_np < features.shape[0]]
                        if len(valid_train_indices) > 0:
                             # Fit only on the columns designated for scaling
                             scaler.fit(features[valid_train_indices, :num_cols_to_scale])
                             self.scalers[node_type] = scaler
                             print(f"  Fitted scaler for 'transaction' features ({self.tx_feat_cols_to_scale}) using {len(valid_train_indices)} training samples.")
                        else:
                             print(f"  [WARN] No valid training indices found for fitting 'transaction' scaler.")
                             self.scalers[node_type] = None # Indicate scaler couldn't be fitted
                    else:
                         print(f"  [WARN] Not enough columns in transaction features ({features.shape[1]}) to scale {num_cols_to_scale} columns.")
                         self.scalers[node_type] = None
                elif node_type == 'merchant': 
                    # Fit merchant scaler globally (as before)
                    scaler = StandardScaler()
                    scaler.fit(features)
                    self.scalers[node_type] = scaler
                    print(f"  Fitted scaler GLOBALLY for '{node_type}'.")
                # No 'category' node type anymore
            else:
                print(f"  Skipping scaler fitting for empty features: '{node_type}'")
                self.scalers[node_type] = None

        # Defer sequence scaler fitting to _prepare_and_add_sequences
        self.seq_scalers['time_delta'] = None 

        # Edge scaler for amount_dist uses transaction scaler's scale value for the 'amount' column (index 0)
        if self.use_gnn_encoder and 'transaction' in self.scalers and self.scalers['transaction'] is not None:
             amount_dist_scaler_proxy = StandardScaler()
             amount_dist_scaler_proxy.mean_ = np.array([0.0])
             # Use scale_ from the fitted transaction scaler for the amount column (index 0)
             # Ensure scaler has enough dimensions before accessing index 0
             if self.scalers['transaction'].scale_.shape[0] > 0:
                 scale_val = np.maximum(self.scalers['transaction'].scale_[0:1], 1e-8)
                 amount_dist_scaler_proxy.scale_ = scale_val
                 self.edge_scalers['amount_dist'] = amount_dist_scaler_proxy
                 print("  Created proxy edge scaler for 'amount_dist' based on train transaction scaler (amount column).")
             else:
                 self.edge_scalers['amount_dist'] = None
                 print("  [WARN] Skipping scaler fitting for edge 'amount_dist' (transaction scaler scale_ is empty).")
        else:
             self.edge_scalers['amount_dist'] = None
             print("  [INFO] Skipping scaler fitting for edge 'amount_dist' (transaction scaler not available).")

        # Add creation of proxy edge scaler here as well
        self._create_proxy_edge_scaler()

    def _create_proxy_edge_scaler(self):
        if self.use_gnn_encoder and 'transaction' in self.scalers and self.scalers['transaction'] is not None:
             if self.scalers['transaction'].scale_.shape[0] > 0:
                 scale_val = np.maximum(self.scalers['transaction'].scale_[0:1], 1e-8)
                 # Only create if not already loaded
                 if 'amount_dist' not in self.edge_scalers or self.edge_scalers['amount_dist'] is None:
                      amount_dist_scaler_proxy = StandardScaler()
                      amount_dist_scaler_proxy.mean_ = np.array([0.0])
                      amount_dist_scaler_proxy.scale_ = scale_val
                      self.edge_scalers['amount_dist'] = amount_dist_scaler_proxy
                      print("  Created proxy edge scaler for 'amount_dist' based on transaction scaler (amount column).")
             # else: print warn? (Already done in fit_scalers)
        # else: print warn? (Already done in fit_scalers)

    def _build_graph_and_edges(self, raw_features: Dict[str, np.ndarray], train_indices: torch.Tensor):
        print("Building graph structure and applying SCALED features...")
        data = HeteroData()
        df = self.transactions_df # Use the sorted, reset_index df
        num_transactions = len(df)
        # Create node index map based on the DataFrame's index (0 to N-1)
        tx_map = {idx: i for i, idx in enumerate(df.index)} 
        
        # Map merchant names to unique integer node indices
        merchant_col = 'merchant_name'
        merchant_ids = df[merchant_col].dropna().unique()
        merchant_map = {name: i for i, name in enumerate(merchant_ids)}
        num_merchants = len(merchant_map)
        
        # No category map needed here

        data['transaction'].num_nodes = num_transactions
        if num_merchants > 0: data['merchant'].num_nodes = num_merchants
        # No category nodes

        print("  Adding scaled node features...")
        for node_type, raw_feat_array in raw_features.items():
            if raw_feat_array.size > 0: 
                final_features = raw_feat_array.copy() # Start with raw features
                if node_type in self.scalers and self.scalers[node_type] is not None:
                    scaler = self.scalers[node_type]
                    if node_type == 'transaction':
                        num_cols_to_scale = len(self.tx_feat_cols_to_scale)
                        # Apply scaler fitted on training data TO ALL transaction nodes
                        scaled_part = (raw_feat_array[:, :num_cols_to_scale] - scaler.mean_) / (np.maximum(scaler.scale_, 1e-8))
                        non_scaled_part = raw_feat_array[:, num_cols_to_scale:]
                        final_features = np.concatenate([scaled_part, non_scaled_part], axis=1)
                        print(f"    Applied TRAIN-fitted scaler to ALL 'transaction' nodes ({self.tx_feat_cols_to_scale}).")
                    elif node_type == 'merchant': # Apply globally fitted scaler to merchant nodes
                        final_features = (raw_feat_array - scaler.mean_) / (np.maximum(scaler.scale_, 1e-8))
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
        # Use the factorized integer 'category_id' column for labels
        data['transaction'].y_global = torch.tensor(df['category_id'].values, dtype=torch.long)
        # No y_user assumed in this version
        # data['transaction'].y_user = ... 
        data['transaction'].user_id_code = torch.tensor(df['user_id_code'].values, dtype=torch.long)
        
        # Updated text feature extraction
        text_cols = ['description', 'memo', 'merchant_name'] # Use new source columns
        # Optionally add chart of account text
        coa_text_list = []
        if 'chart_of_accounts_processed' in df.columns and self.use_text_encoder:
            print("  Including chart of accounts text...")
            for coa_json_str in df['chart_of_accounts_processed'].fillna('[]'):
                try:
                    coa_list = json.loads(coa_json_str) if isinstance(coa_json_str, str) else coa_json_str
                    # Combine account names and descriptions
                    acc_texts = [f"{acc.get('account_name', '')} {acc.get('account_description', '')}".strip() for acc in coa_list if isinstance(acc, dict)]
                    coa_text_list.append(" || ".join(filter(None, acc_texts)))
                except (json.JSONDecodeError, TypeError):
                    coa_text_list.append("") # Append empty string on error
        else:
             coa_text_list = ["" for _ in range(len(df))] # Empty strings if column missing

        # Combine transaction text and chart of accounts text
        raw_text_combined = []
        for i in range(len(df)):
            tx_parts = [str(df.iloc[i].get(col, '')) for col in text_cols]
            tx_text = ' || '.join(filter(None, tx_parts))
            coa_text = coa_text_list[i]
            combined = f"{tx_text} || CHART: {coa_text}" if coa_text else tx_text
            raw_text_combined.append(combined)
        
        data['transaction']._raw_text = raw_text_combined
        if self.use_text_encoder: print(f"  Combined raw text features created (including COA if present).")

        print("  Creating edges and edge features...")
        edge_index_dict = {}
        edge_attr_dict = {}
        
        # Edge calculations remain largely the same, using the node indices from maps
        # 1. Transaction -> Merchant
        if num_merchants > 0:
            edge_list, attr_list = [], []
            # Use merchant_name (derived from payee) for grouping
            merchant_stats_map = df.groupby(merchant_col)['amount'].agg(['mean', 'std']).fillna(0).to_dict('index')
            for idx, row in df.iterrows():
                merchant_name = row[merchant_col]
                if pd.notna(merchant_name) and merchant_name in merchant_map:
                    tx_node_idx = tx_map[idx]; merchant_node_idx = merchant_map[merchant_name]
                    edge_list.append([tx_node_idx, merchant_node_idx])
                    stats = merchant_stats_map.get(merchant_name, {'mean': 0, 'std': 0})
                    # Ensure row amount is float before calculation
                    row_amount = float(row['amount'])
                    amount_zscore = (row_amount - stats['mean']) / (stats['std'] + 1e-8)
                    attr_list.append([amount_zscore])
            if edge_list:
                edge_index_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_attr_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(attr_list, dtype=torch.float)
                self.edge_feature_dims[('transaction', 'belongs_to', 'merchant')] = 1

        # 2. Merchant -> Category - REMOVED
        # print("Skipping Merchant -> Category edges.")

        # 3. Transaction -> Transaction (temporal) 
        edge_list, attr_list = [], []
        # df is already sorted by user_id_code, then timestamp
        # tx_map maps df index (0..N-1) to node index (0..N-1)
        for i in range(num_transactions):
            ts_i = df.iloc[i]['timestamp']
            user_i = df.iloc[i]['user_id_code'] # Use the factorized user code
            if pd.isna(ts_i): continue 
            tx_node_i = tx_map[df.index[i]] # Get node index for row i
            # Look ahead only within the same user
            for k in range(1, 6): 
                j = i + k
                if j >= num_transactions: break
                user_j = df.iloc[j]['user_id_code'] # Use factorized user code
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
        # Get raw amounts from the correct place (first col of scaled feats) 
        # Need to get from original df before scaling for KNN
        raw_tx_amounts = df['amount'].fillna(0.0).values 

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
                    scaled_dist = (raw_dist_array / (np.maximum(scaler.scale_, 1e-8))).flatten() 
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
        num_transactions = len(df) # Renamed from num_nodes for clarity
        self.full_graph_data['transaction'].num_nodes = num_transactions
        self.full_graph_data['transaction'].original_index = torch.tensor(df.index.values, dtype=torch.long)
        # Use the factorized integer 'category_id' for labels
        self.full_graph_data['transaction'].y_global = torch.tensor(df['category_id'].values, dtype=torch.long)
        # No y_user
        self.full_graph_data['transaction'].user_id_code = torch.tensor(df['user_id_code'].values, dtype=torch.long)
        
        # Updated text feature extraction for minimal graph
        text_cols = ['description', 'memo', 'merchant_name'] 
        coa_text_list = []
        if 'chart_of_accounts_processed' in df.columns and self.use_text_encoder:
            for coa_json_str in df['chart_of_accounts_processed'].fillna('[]'):
                try:
                    coa_list = json.loads(coa_json_str) if isinstance(coa_json_str, str) else coa_json_str
                    acc_texts = [f"{acc.get('account_name', '')} {acc.get('account_description', '')}".strip() for acc in coa_list if isinstance(acc, dict)]
                    coa_text_list.append(" || ".join(filter(None, acc_texts)))
                except (json.JSONDecodeError, TypeError):
                    coa_text_list.append("") 
        else:
            coa_text_list = ["" for _ in range(len(df))]

        raw_text_combined = []
        for i in range(len(df)):
            tx_parts = [str(df.iloc[i].get(col, '')) for col in text_cols]
            tx_text = ' || '.join(filter(None, tx_parts))
            coa_text = coa_text_list[i]
            combined = f"{tx_text} || CHART: {coa_text}" if coa_text else tx_text
            raw_text_combined.append(combined)
        
        self.full_graph_data['transaction']._raw_text = raw_text_combined
        print("Built minimal graph data.")

    def _prepare_and_add_sequences(self, train_indices: torch.Tensor):
        if self.full_graph_data is None: raise RuntimeError("Graph data not built.")
        if not self.use_sequence_encoder: 
            print("[INFO] Skipping sequence preparation as use_sequence_encoder=False")
            return 
        if hasattr(self.full_graph_data['transaction'], 'seq_features'): return 
             
        print("Preparing sequence features...")
        df_sorted = self.transactions_df 
        num_transactions = len(df_sorted)
        train_indices_np = train_indices.numpy()

        # --- Fit sequence scalers only if NOT predicting --- 
        if not self.is_predicting:
            print("  Fitting sequence scalers on TRAINING data...")
            all_train_time_deltas = []
            for current_pos_in_sorted in train_indices_np:
                 # Ensure index is valid
                 if current_pos_in_sorted >= num_transactions: continue

                 current_row = df_sorted.iloc[current_pos_in_sorted]
                 user_id_code = current_row['user_id_code']; current_time = current_row['timestamp']
                 start_idx_in_sorted = max(0, current_pos_in_sorted - self.max_seq_length)
                 prev_txs_window_df = df_sorted.iloc[start_idx_in_sorted:current_pos_in_sorted]
                 # Filter window to only include transactions from the same user
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
        else:
            print("  Skipping sequence scaler fitting (using pre-loaded scaler).")
            # Ensure the key exists even if None was loaded
            if 'time_delta' not in self.seq_scalers:
                 self.seq_scalers['time_delta'] = None

        # --- Generate sequences for ALL transactions (apply fitted/loaded scalers) ---
        print("  Generating sequences for ALL transactions...")
        all_seq_features = []
        all_seq_lengths = []
        self.sequence_feature_dim = 6 

        for node_idx in range(num_transactions):
            # We iterate through node indices 0..N-1, which correspond to df_sorted rows
            current_pos_in_sorted = node_idx 
                 
            current_row = df_sorted.iloc[current_pos_in_sorted]
            user_id_code = current_row['user_id_code']; current_time = current_row['timestamp']
            start_idx_in_sorted = max(0, current_pos_in_sorted - self.max_seq_length)
            prev_txs_window_df = df_sorted.iloc[start_idx_in_sorted:current_pos_in_sorted]
            # Filter window to only include transactions from the same user
            prev_txs_user_df = prev_txs_window_df[prev_txs_window_df['user_id_code'] == user_id_code]

            seq_features_for_tx = []

            if not prev_txs_user_df.empty:
                for _, prev_row in prev_txs_user_df.iterrows():
                    time_delta = 0.0
                    if pd.notna(current_time) and pd.notna(prev_row['timestamp']):
                         time_delta_seconds = (current_time - prev_row['timestamp']).total_seconds()
                         time_delta = max(0.0, time_delta_seconds) 
                    amount = prev_row['amount'] if pd.notna(prev_row['amount']) else 0.0
                    # Amount scaling for sequences - should we fit a separate scaler?
                    # For now, don't scale amount in sequences, only time_delta.
                    time_delta_val = time_delta
                    # Check if scaler exists and is not None before applying
                    if 'time_delta' in self.seq_scalers and self.seq_scalers['time_delta'] is not None:
                         scaler_td = self.seq_scalers['time_delta']
                         time_delta_val = (time_delta - scaler_td.mean_[0]) / (np.maximum(scaler_td.scale_[0], 1e-8))
                    hour = prev_row['hour']; day = prev_row['weekday']
                    hour_sin = np.sin(2*np.pi*hour/24); hour_cos = np.cos(2*np.pi*hour/24)
                    day_sin = np.sin(2*np.pi*day/7); day_cos = np.cos(2*np.pi*day/7)
                    scaled_feat = [amount, day_sin, day_cos, hour_sin, hour_cos, time_delta_val]
                    seq_features_for_tx.append(scaled_feat)

            if seq_features_for_tx: # Check if real features were added
                seq_tensor = torch.tensor(seq_features_for_tx, dtype=torch.float)
                # Apply max_seq_length limit - window slicing already limits input rows
                # but final check ensures tensor length constraint
                if seq_tensor.shape[0] > self.max_seq_length:
                    seq_tensor = seq_tensor[-self.max_seq_length:, :]
                seq_len = len(seq_tensor)
            else:
                seq_tensor = torch.zeros((0, self.sequence_feature_dim), dtype=torch.float)
                seq_len = 0
                
            all_seq_features.append(seq_tensor)
            all_seq_lengths.append(seq_len)

        max_len_found = max(all_seq_lengths) if all_seq_lengths else 0
        print(f"  Max sequence length found: {max_len_found}")
        if not all_seq_features:
             padded_sequences = torch.empty((num_transactions, 0, self.sequence_feature_dim), dtype=torch.float)
        else:
             padded_sequences = pad_sequence(all_seq_features, batch_first=True, padding_value=0.0)
        
        # No padding for categorical features

        self.full_graph_data['transaction'].seq_features = padded_sequences
        self.full_graph_data['transaction'].seq_lengths = torch.tensor(all_seq_lengths, dtype=torch.long)
        print(f"Added sequence features. Padded reals shape: {padded_sequences.shape}") # Removed cat shape log

    def _split_data_indices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        df = self.transactions_df # Assumes sorted by user_id_code, timestamp
        print("Creating train/val/test indices: Time-based split within each user...")
        
        all_train_indices = []
        all_val_indices = []
        all_test_indices = []

        # Group by user_id_code
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
    def _assign_masks_to_graph(self, predict_only: bool = False):
        if self.full_graph_data is None or not hasattr(self.full_graph_data['transaction'], 'num_nodes'):
            print("[WARN] Cannot assign masks, graph data not available.")
            return
        
        num_nodes = self.full_graph_data['transaction'].num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if predict_only:
            # Only assign test mask using self.test_indices (which covers all nodes)
            if self.test_indices is not None:
                 test_mask[self.test_indices] = True
                 print("Assigned predict mask (all nodes) to graph data.")
            else:
                 print("[WARN] predict_only=True but test_indices is None.")
        else:
            # Assign train/val/test masks as before for fit/test stages
            if self.train_indices is not None:
                train_mask[self.train_indices] = True
            if self.val_indices is not None:
                val_mask[self.val_indices] = True
            if self.test_indices is not None:
                test_mask[self.test_indices] = True
            print("Assigned train/val/test masks to graph data.")

        self.full_graph_data['transaction'].train_mask = train_mask
        self.full_graph_data['transaction'].val_mask = val_mask
        self.full_graph_data['transaction'].test_mask = test_mask
        
    # --- Dataloader Methods using HGTLoader --- 
    def train_dataloader(self) -> HGTLoader:
        print("Creating train HGTLoader...")
        if self.train_indices is None: self.setup('fit')
        if self.full_graph_data is None: raise RuntimeError("Graph data not available.")
        if self.train_indices is None or len(self.train_indices) == 0: raise ValueError("Training set empty.")
        # Filter hgt_num_samples to only include node types actually present in the graph
        valid_hgt_num_samples = {k:v for k,v in self.hgt_num_samples.items() if k in self.full_graph_data.node_types}
        if not valid_hgt_num_samples: 
             print("[WARN] No valid node types for HGT sampling specified or detected. GNN might not function correctly.")
             # If GNN is enabled, we need *something* here, even if it's just transaction nodes.
             if self.use_gnn_encoder and 'transaction' in self.full_graph_data.node_types:
                 print("  Defaulting to sampling only for 'transaction' node.")
                 valid_hgt_num_samples = {'transaction': self.hgt_num_samples.get('transaction', [15, 10][:self.num_hgt_layers])}
             else:
                 raise ValueError("No valid node types for HGT sampling found.") # Cannot proceed

        # Move graph data to CPU before passing to loader if it's not already
        graph_data_cpu = self.full_graph_data.cpu()
        return HGTLoader(graph_data_cpu, num_samples=valid_hgt_num_samples, shuffle=True,
                         input_nodes=('transaction', self.train_indices.cpu()), batch_size=self.batch_size,
                         num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def val_dataloader(self) -> HGTLoader:
        print("Creating val HGTLoader...")
        if self.val_indices is None: self.setup('fit')
        if self.full_graph_data is None: raise RuntimeError("Graph data not available.")
        if self.val_indices is None or len(self.val_indices) == 0: raise ValueError("Validation set empty.")
        valid_hgt_num_samples = {k:v for k,v in self.hgt_num_samples.items() if k in self.full_graph_data.node_types}
        if not valid_hgt_num_samples: 
            print("[WARN] No valid node types for HGT sampling specified or detected during validation.")
            if self.use_gnn_encoder and 'transaction' in self.full_graph_data.node_types:
                 print("  Defaulting to sampling only for 'transaction' node.")
                 valid_hgt_num_samples = {'transaction': self.hgt_num_samples.get('transaction', [15, 10][:self.num_hgt_layers])}
            else:
                 raise ValueError("No valid node types for HGT sampling found.")
        graph_data_cpu = self.full_graph_data.cpu()
        return HGTLoader(graph_data_cpu, num_samples=valid_hgt_num_samples, shuffle=False,
                         input_nodes=('transaction', self.val_indices.cpu()), batch_size=self.batch_size,
                         num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def test_dataloader(self) -> HGTLoader:
        print("Creating test HGTLoader...")
        if self.test_indices is None: self.setup('test')
        if self.full_graph_data is None: raise RuntimeError("Graph data not available.")
        if self.test_indices is None or len(self.test_indices) == 0: raise ValueError("Test set empty.")
        valid_hgt_num_samples = {k:v for k,v in self.hgt_num_samples.items() if k in self.full_graph_data.node_types}
        if not valid_hgt_num_samples: 
            print("[WARN] No valid node types for HGT sampling specified or detected during testing.")
            if self.use_gnn_encoder and 'transaction' in self.full_graph_data.node_types:
                 print("  Defaulting to sampling only for 'transaction' node.")
                 valid_hgt_num_samples = {'transaction': self.hgt_num_samples.get('transaction', [15, 10][:self.num_hgt_layers])}
            else:
                 raise ValueError("No valid node types for HGT sampling found.")
        graph_data_cpu = self.full_graph_data.cpu()
        return HGTLoader(graph_data_cpu, num_samples=valid_hgt_num_samples, shuffle=False,
                         input_nodes=('transaction', self.test_indices.cpu()), batch_size=self.batch_size,
                         num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    # Add predict_dataloader
    def predict_dataloader(self) -> HGTLoader:
        print("Creating predict HGTLoader...")
        if self.test_indices is None: self.setup('predict') # Ensure setup is called
        if self.full_graph_data is None: raise RuntimeError("Graph data not available.")
        if self.test_indices is None or len(self.test_indices) == 0: 
             # If test_indices is still empty after setup('predict'), something went wrong
             raise ValueError("Prediction set empty or test indices not assigned correctly during setup.")
             
        # Use the same sampling logic as test_dataloader
        valid_hgt_num_samples = {k:v for k,v in self.hgt_num_samples.items() if k in self.full_graph_data.node_types}
        if not valid_hgt_num_samples: 
            print("[WARN] No valid node types for HGT sampling specified or detected during prediction.")
            if self.use_gnn_encoder and 'transaction' in self.full_graph_data.node_types:
                 print("  Defaulting to sampling only for 'transaction' node.")
                 valid_hgt_num_samples = {'transaction': self.hgt_num_samples.get('transaction', [15, 10][:self.num_hgt_layers])}
            else:
                 raise ValueError("No valid node types for HGT sampling found.")
                 
        graph_data_cpu = self.full_graph_data.cpu()
        # Input nodes should be the test_indices, which contain all nodes for predict stage
        return HGTLoader(graph_data_cpu, num_samples=valid_hgt_num_samples, shuffle=False,
                         input_nodes=('transaction', self.test_indices.cpu()), batch_size=self.batch_size,
                         num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))


