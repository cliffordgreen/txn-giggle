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

# --- TextInjectingDataLoader Class ---
class TextInjectingDataLoader:
    """Wrapper around HGTLoader that injects text data into each batch."""
    
    def __init__(self, hgt_loader: HGTLoader, transaction_raw_text: List[str], use_text_encoder: bool):
        self.hgt_loader = hgt_loader
        self.transaction_raw_text = transaction_raw_text
        self.use_text_encoder = use_text_encoder
        
    def __iter__(self):
        for batch in self.hgt_loader:
            if self.use_text_encoder and 'transaction' in batch.node_types:
                self._inject_text_data(batch)
            yield batch
    
    def _inject_text_data(self, batch):
        """Inject text data into the batch based on transaction indices."""
        try:
            # Get the transaction node store
            tx_store = batch['transaction']
            
            # Method 1: Try to use input_id (seed nodes for the batch)
            batch_indices = None
            if hasattr(tx_store, 'input_id'):
                try:
                    batch_indices = tx_store.input_id.cpu().numpy()
                except Exception:
                    pass
            
            # Method 2: Try to use batch_size + original_index
            if batch_indices is None and hasattr(tx_store, 'batch_size') and hasattr(tx_store, 'original_index'):
                try:
                    batch_size = tx_store.batch_size
                    if batch_size is not None and batch_size > 0 and tx_store.original_index.shape[0] >= batch_size:
                        batch_indices = tx_store.original_index[:batch_size].cpu().numpy()
                except Exception:
                    pass
            
            # Method 3: Try to use all original_index if we can't determine batch_size
            if batch_indices is None and hasattr(tx_store, 'original_index'):
                try:
                    # This might include neighbor nodes too, but we'll limit by available data
                    batch_indices = tx_store.original_index.cpu().numpy()
                except Exception:
                    pass
            
            # Inject text data if we have valid indices
            if batch_indices is not None:
                # Ensure indices are within bounds and get corresponding text
                valid_indices = [idx for idx in batch_indices if 0 <= idx < len(self.transaction_raw_text)]
                batch_text = [self.transaction_raw_text[idx] for idx in valid_indices]
                
                # Add text data to the transaction store
                tx_store._raw_text = batch_text
                
                # Debug info
                if len(batch_text) != len(batch_indices):
                    print(f"[INFO] Text injection: {len(batch_text)} texts for {len(batch_indices)} indices "
                          f"(some indices out of bounds)")
            else:
                print(f"[WARN] Could not determine batch indices for text injection")
                tx_store._raw_text = []
                
        except Exception as e:
            print(f"[ERROR] Failed to inject text data into batch: {e}")
            # Ensure _raw_text exists even if empty
            if 'transaction' in batch.node_types:
                batch['transaction']._raw_text = []
    
    def __len__(self):
        return len(self.hgt_loader)

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
                 val_ratio: float = 0.05,
                 test_ratio: float = 0.05,
                 use_sequence_encoder: bool = True,
                 use_text_encoder: bool = True,
                 use_gnn_encoder: bool = True,
                 use_coa_text_features: bool = True,
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
        self.use_coa_text_features = use_coa_text_features

        # Store/Create HGT sampling config
        self.num_hgt_layers = num_hgt_layers
        self.hgt_num_samples = hgt_num_samples
        if self.hgt_num_samples is None:
            # Simple default: Sample 15 neighbors in first layer, 10 in second
            default_samples_per_layer = [15, 10]
            # Define KNOWN node types used in the graph (including 'category')
            node_types_in_graph = ['transaction', 'merchant', 'category'] 
            self.hgt_num_samples = {ntype: default_samples_per_layer[:self.num_hgt_layers] for ntype in node_types_in_graph}
            print(f"[WARN] hgt_num_samples not provided. Using default based on num_hgt_layers={self.num_hgt_layers}: {self.hgt_num_samples}")
        else:
            # Validate provided config against expected node types
            expected_node_types = {'transaction', 'merchant', 'category'} # All expected node types
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
        self.use_precomputed_scalers = (fitted_scalers is not None)  # Flag for pre-computed scalers

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
            self.transactions_df[target_col] = self.transactions_df[target_col].fillna('UNKNOWN').infer_objects(copy=False)
            
        # Use fitted maps if provided (predicting), otherwise factorize
        if self.is_predicting:
            print("  Using provided category_id and user maps.")
            # Map strings to known ints, use -1 for unknowns
            map_cat_str_to_int = {v: k for k, v in self.category_id_map.items()}
            map_user_str_to_int = {v: k for k, v in self.user_map.items()}
            self.transactions_df['category_id'] = self.transactions_df[target_col].map(map_cat_str_to_int).fillna(-1).infer_objects(copy=False).astype(int)
            self.transactions_df['user_id_code'] = self.transactions_df['company_name'].map(map_user_str_to_int).fillna(-1).infer_objects(copy=False).astype(int)
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
            
            if self.use_precomputed_scalers and 'user_map' in self.original_stats:
                # Use pre-computed user mapping (format: {user_name: user_id})
                self.user_map = self.original_stats['user_map']
                self.num_users = len(self.user_map)
                # Apply the mapping to create user_id_code column
                self.transactions_df['user_id_code'] = self.transactions_df['company_name'].map(self.user_map)
                # Handle any unmapped users (shouldn't happen in streaming if stats are correct)
                unmapped_mask = self.transactions_df['user_id_code'].isna()
                if unmapped_mask.any():
                    print(f"[WARN] Found {unmapped_mask.sum()} unmapped users in streaming data")
                    self.transactions_df.loc[unmapped_mask, 'user_id_code'] = 0  # Map to first user as fallback
            else:
                # Create new user mapping (format: {user_name: user_id})
                user_codes, user_uniques = pd.factorize(self.transactions_df['company_name'], sort=True)
                self.transactions_df['user_id_code'] = user_codes 
                self.user_map = {uid: code for code, uid in enumerate(user_uniques)}  # {user_name: user_id}
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

        # **Step 3: Fit scalers ONLY on TRAINING data (Skip if using pre-computed)**
        if not self.use_precomputed_scalers: # Only fit if not using pre-computed scalers
            print("Fitting scalers on TRAINING data...")
            self._fit_scalers(raw_features, self.train_indices) 
        else:
            print("Using pre-computed scalers from statistics pass...")
            # Load pre-computed scalers
            self._load_precomputed_scalers()
            # Ensure edge scaler exists if needed and not provided
            if self.use_gnn_encoder and 'amount_dist' not in self.edge_scalers: 
                 self._create_proxy_edge_scaler()

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
        # Features: amount (scaled), num_chart_of_accounts (log-normalized), COA features (scaled), similar txn features (scaled), time features (not scaled)
        self.tx_feat_cols_to_scale = ['amount']  # Continuous features
        self.tx_feat_cols_to_normalize = ['num_chart_of_accounts']  # Count features (log transform)
        self.tx_feat_cols_coa_scale = ['coa_size', 'hierarchy_depth', 'type_diversity', 
                                       'code_complexity', 'naming_consistency', 
                                       'description_richness', 'balance_diversity', 'recent_accounts']  # COA features
        self.tx_feat_cols_similar_scale = ['num_similar_txns', 'avg_similarity', 'max_similarity', 'similarity_std',
                                          'category_diversity', 'avg_similar_amount', 'unique_categories', 'top_similarity']  # Similar txn features  
        self.tx_feat_cols_no_scale = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        
        # Ensure required columns exist
        if 'amount' not in df.columns: raise ValueError("Missing 'amount' column")
        if 'hour' not in df.columns: raise ValueError("Missing 'hour' column")
        if 'weekday' not in df.columns: raise ValueError("Missing 'weekday' column")
        if 'num_chart_of_accounts' not in df.columns: raise ValueError("Missing 'num_chart_of_accounts' column")

        amount = df['amount'].fillna(0.0)
        num_coa = df['num_chart_of_accounts'].fillna(0).astype(int)
        # Apply log transformation to count features (log1p to handle 0 values)
        num_coa_log = np.log1p(num_coa)
        hour = df['hour'].fillna(0).astype(int)
        day = df['weekday'].fillna(0).astype(int)
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        # Extract COA features
        coa_features = self._extract_transaction_coa_features(df)
        print(f"  COA features extracted. Shape: {coa_features.shape}")
        
        # Extract similar transaction features
        similar_txn_features = self._extract_similar_txn_features(df)
        print(f"  Similar transaction features extracted. Shape: {similar_txn_features.shape}")
        
        # Stack: continuous (to scale), log-normalized counts (to scale), COA features (to scale), similar txn features (to scale), cyclical (no scale)
        tx_features_array = np.column_stack([amount, num_coa_log, coa_features, similar_txn_features, hour_sin, hour_cos, day_sin, day_cos])
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
        
        # Category features calculation
        print("Calculating raw category features (vectorized)...")
        valid_categories = df['category_id'].dropna().unique()
        valid_categories = [c for c in valid_categories if isinstance(c, (int, np.integer)) and c != -1]
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

    def _extract_transaction_coa_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract COA features for each transaction."""
        print("Extracting transaction-level COA features (vectorized)...")
        
        def analyze_transaction_coa(coa_json_str) -> List[float]:
            """Analyze COA structure for a single transaction."""
            try:
                # Safe null check that works with both scalars and arrays
                if coa_json_str is None or str(coa_json_str).lower() in ['nan', 'none', '', 'null']:
                    return self._default_coa_features()
                    
                coa_list = json.loads(coa_json_str) if isinstance(coa_json_str, str) else coa_json_str
                if not isinstance(coa_list, list) or len(coa_list) == 0:
                    return self._default_coa_features()
                
                # Extract basic info
                account_codes = [acc.get('account_code', '') for acc in coa_list if isinstance(acc, dict)]
                account_names = [acc.get('account_name', '') for acc in coa_list if isinstance(acc, dict)]
                account_descriptions = [acc.get('account_description', '') for acc in coa_list if isinstance(acc, dict)]
                
                return [
                    len(coa_list),  # COA size at this point in time
                    self._calculate_hierarchy_depth(account_codes),
                    self._calculate_account_type_diversity(account_codes, account_names),
                    self._calculate_code_complexity(account_codes),
                    self._calculate_naming_consistency(account_names),
                    self._calculate_description_richness(account_descriptions),
                    self._calculate_account_balance_diversity(coa_list),
                    self._calculate_recent_accounts_indicator(coa_list)
                ]
                
            except (json.JSONDecodeError, TypeError, AttributeError):
                return self._default_coa_features()
        
        # Apply vectorized processing
        if 'chart_of_accounts_processed' in df.columns:
            coa_feature_series = df['chart_of_accounts_processed'].apply(analyze_transaction_coa)
            return np.array(coa_feature_series.tolist())
        else:
            # Return default features if COA column missing
            return np.array([self._default_coa_features() for _ in range(len(df))])

    def _default_coa_features(self) -> List[float]:
        """Default features for missing/invalid COA data."""
        return [0.0] * 8  # 8 COA features

    def _calculate_hierarchy_depth(self, account_codes: List[str]) -> float:
        """Calculate average hierarchy depth from account codes."""
        if not account_codes:
            return 0.0
        
        depths = []
        for code in account_codes:
            code_str = str(code).strip()
            if not code_str:
                continue
                
            # Heuristic: count digits for depth (e.g., 1000=1, 1100=2, 1110=3)
            if code_str.isdigit():
                # For numeric codes, estimate depth by significant digits
                depth = len(code_str.rstrip('0')) if code_str.rstrip('0') else 1
            else:
                # For alphanumeric codes, count separators + 1
                depth = code_str.count('-') + code_str.count('.') + code_str.count('_') + 1
            depths.append(min(depth, 10))  # Cap at reasonable depth
        
        return np.mean(depths) if depths else 0.0

    def _calculate_account_type_diversity(self, account_codes: List[str], account_names: List[str]) -> float:
        """Calculate diversity of account types (assets, liabilities, etc.)."""
        account_types = set()
        
        for code, name in zip(account_codes, account_names):
            account_type = self._infer_account_type(code, name)
            account_types.add(account_type)
        
        # Return normalized diversity (0-1)
        max_types = 6  # asset, liability, equity, revenue, expense, other
        return len(account_types) / max_types

    def _infer_account_type(self, code: str, name: str) -> str:
        """Infer account type from code and name patterns."""
        code_str = str(code).lower().strip()
        name_str = str(name).lower().strip()
        
        # Standard account code ranges (first digit)
        if code_str and code_str[0].isdigit():
            first_digit = code_str[0]
            if first_digit == '1': return 'asset'
            elif first_digit == '2': return 'liability' 
            elif first_digit == '3': return 'equity'
            elif first_digit == '4': return 'revenue'
            elif first_digit in ['5', '6', '7']: return 'expense'
        
        # Name-based inference
        asset_keywords = ['cash', 'bank', 'receivable', 'inventory', 'equipment', 'asset']
        liability_keywords = ['payable', 'loan', 'debt', 'accrued', 'liability']
        equity_keywords = ['equity', 'capital', 'retained', 'stock']
        revenue_keywords = ['sales', 'revenue', 'income', 'service', 'fees']
        expense_keywords = ['expense', 'cost', 'salary', 'rent', 'utilities', 'depreciation']
        
        if any(kw in name_str for kw in asset_keywords): return 'asset'
        elif any(kw in name_str for kw in liability_keywords): return 'liability'
        elif any(kw in name_str for kw in equity_keywords): return 'equity'
        elif any(kw in name_str for kw in revenue_keywords): return 'revenue'
        elif any(kw in name_str for kw in expense_keywords): return 'expense'
        
        return 'other'

    def _calculate_code_complexity(self, account_codes: List[str]) -> float:
        """Calculate complexity/structure of account codes."""
        if not account_codes:
            return 0.0
        
        complexities = []
        for code in account_codes:
            code_str = str(code).strip()
            if not code_str:
                continue
                
            # Complexity indicators
            length_score = min(len(code_str) / 10.0, 1.0)  # Normalize length
            separator_score = min((code_str.count('-') + code_str.count('.') + code_str.count('_')) / 3.0, 1.0)
            alphanumeric_score = 0.2 if any(c.isalpha() for c in code_str) and any(c.isdigit() for c in code_str) else 0.0
            
            complexity = (length_score + separator_score + alphanumeric_score) / 3.0
            complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.0

    def _calculate_naming_consistency(self, account_names: List[str]) -> float:
        """Calculate consistency of naming conventions."""
        if len(account_names) < 2:
            return 1.0  # Perfect consistency for 0-1 accounts
        
        # Simple heuristic: consistency in capitalization and word patterns
        capitalization_patterns = set()
        word_counts = []
        
        for name in account_names:
            name_str = str(name).strip()
            if not name_str:
                continue
                
            # Capitalization pattern
            if name_str.isupper():
                capitalization_patterns.add('upper')
            elif name_str.islower():
                capitalization_patterns.add('lower')
            elif name_str.istitle():
                capitalization_patterns.add('title')
            else:
                capitalization_patterns.add('mixed')
            
            # Word count
            word_counts.append(len(name_str.split()))
        
        # Consistency scores
        cap_consistency = 1.0 - (len(capitalization_patterns) - 1) / 3.0  # Normalize to 0-1
        word_count_consistency = 1.0 - (np.std(word_counts) / (np.mean(word_counts) + 1e-8)) if word_counts else 1.0
        word_count_consistency = max(0.0, min(1.0, word_count_consistency))
        
        return (cap_consistency + word_count_consistency) / 2.0

    def _calculate_description_richness(self, descriptions: List[str]) -> float:
        """Calculate richness of account descriptions."""
        if not descriptions:
            return 0.0
        
        total_chars = 0
        non_empty_count = 0
        
        for desc in descriptions:
            desc_str = str(desc).strip()
            if desc_str:
                total_chars += len(desc_str)
                non_empty_count += 1
        
        if non_empty_count == 0:
            return 0.0
        
        # Average description length (normalized)
        avg_length = total_chars / non_empty_count
        return min(avg_length / 100.0, 1.0)  # Normalize to [0, 1], cap at 100 chars

    def _calculate_account_balance_diversity(self, coa_list: List[Dict]) -> float:
        """Calculate diversity of account balances."""
        balances = []
        for acc in coa_list:
            if isinstance(acc, dict) and 'balance' in acc:
                try:
                    balance = float(acc['balance'])
                    balances.append(abs(balance))
                except (ValueError, TypeError):
                    continue
        
        if len(balances) < 2:
            return 0.0
        
        # Coefficient of variation (normalized diversity)
        mean_balance = np.mean(balances)
        if mean_balance < 1e-8:
            return 0.0
        
        cv = np.std(balances) / mean_balance
        return min(cv / 2.0, 1.0)  # Normalize and cap

    def _calculate_recent_accounts_indicator(self, coa_list: List[Dict]) -> float:
        """Estimate proportion of recently added accounts."""
        if not coa_list:
            return 0.0
        
        recent_indicators = 0
        total_accounts = len(coa_list)
        
        for acc in coa_list:
            if not isinstance(acc, dict):
                continue
                
            acc_code = str(acc.get('account_code', '')).lower()
            acc_name = str(acc.get('account_name', '')).lower()
            
            # Heuristics for recent/temporary accounts
            recent_patterns = [
                acc_code.endswith('99'), acc_code.endswith('00'),  # Common for new accounts
                'new' in acc_name, 'temp' in acc_name, 'misc' in acc_name,
                'other' in acc_name, 'suspense' in acc_name
            ]
            
            if any(recent_patterns):
                recent_indicators += 1
        
        return recent_indicators / total_accounts

    def _extract_similar_txn_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract aggregated features from similar transactions data."""
        features = []
        
        # Check if similar_txns_processed column exists
        if 'similar_txns_processed' not in df.columns:
            print("  [INFO] similar_txns_processed column not found, using zero features for all transactions")
            return np.zeros((len(df), 8), dtype=np.float32)
        
        for similar_txns_str in df['similar_txns_processed']:
            if pd.isna(similar_txns_str) or similar_txns_str == '[]':
                # No similar transactions - use zero features
                features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
                
            try:
                similar_txns = json.loads(similar_txns_str)
                if not similar_txns:
                    features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                
                # Extract similarity scores and categories
                similarities = [txn.get('similarity', 0.0) for txn in similar_txns]
                categories = [txn.get('category_id') for txn in similar_txns if txn.get('category_id') is not None]
                amounts = [txn.get('amount', 0.0) for txn in similar_txns]
                
                # Feature 1: Number of similar transactions
                num_similar = len(similar_txns)
                
                # Feature 2: Average similarity score
                avg_similarity = np.mean(similarities) if similarities else 0.0
                
                # Feature 3: Maximum similarity score
                max_similarity = max(similarities) if similarities else 0.0
                
                # Feature 4: Similarity score standard deviation
                similarity_std = np.std(similarities) if len(similarities) > 1 else 0.0
                
                # Feature 5: Category diversity (number of unique categories)
                unique_categories = len(set(categories)) if categories else 0
                
                # Feature 6: Average amount of similar transactions
                avg_similar_amount = np.mean(amounts) if amounts else 0.0
                
                # Feature 7: Category diversity ratio (unique/total)
                category_diversity = unique_categories / num_similar if num_similar > 0 else 0.0
                
                # Feature 8: Top similarity (95th percentile or max if < 20 samples)
                top_similarity = np.percentile(similarities, 95) if len(similarities) >= 20 else max_similarity
                
                features.append([
                    num_similar, avg_similarity, max_similarity, similarity_std,
                    category_diversity, avg_similar_amount, unique_categories, top_similarity
                ])
                
            except (json.JSONDecodeError, KeyError, ValueError):
                # Malformed data - use zero features
                features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)

    def _extract_text_features(self, df: pd.DataFrame) -> List[str]:
        """Extract and combine text features including COA text (DRY principle)."""
        text_cols = ['description', 'memo', 'merchant_name']
        
        # Extract COA text using vectorized operations
        coa_text_series = self._extract_coa_text_vectorized(df)
        
        # Vectorized transaction text extraction
        tx_text_parts = df[text_cols].fillna('').astype(str)
        tx_text_series = tx_text_parts.apply(lambda row: ' || '.join(filter(None, row)), axis=1)
        
        # Combine transaction and COA text
        combined_text = []
        for tx_text, coa_text in zip(tx_text_series, coa_text_series):
            combined = f"{tx_text} || CHART: {coa_text}" if coa_text else tx_text
            combined_text.append(combined)
            
        return combined_text

    def _extract_coa_text_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Extract COA text features using vectorized pandas operations."""
        if not (self.use_text_encoder and self.use_coa_text_features and 'chart_of_accounts_processed' in df.columns):
            return pd.Series([""] * len(df))
        
        print("  Including chart of accounts text (vectorized)...")
        
        def parse_coa_text(coa_json_str):
            """Parse individual COA JSON string."""
            try:
                if pd.isna(coa_json_str):
                    return ""
                coa_list = json.loads(coa_json_str) if isinstance(coa_json_str, str) else coa_json_str
                if not isinstance(coa_list, list):
                    return ""
                # Combine account names and descriptions
                acc_texts = [
                    f"{acc.get('account_name', '')} {acc.get('account_description', '')}".strip() 
                    for acc in coa_list if isinstance(acc, dict)
                ]
                return " || ".join(filter(None, acc_texts))
            except (json.JSONDecodeError, TypeError, AttributeError):
                return ""
        
        # Apply vectorized parsing
        return df['chart_of_accounts_processed'].apply(parse_coa_text)

    def _fit_scalers(self, raw_features: Dict[str, np.ndarray], train_indices: torch.Tensor):
        print("Fitting scalers (using only training data where applicable)...")
        train_indices_np = train_indices.numpy()

        for node_type, features in raw_features.items():
            if features.size > 0 and features.shape[0] > 0:
                if node_type == 'transaction':
                    num_cols_to_scale = (len(self.tx_feat_cols_to_scale) + 
                                        len(self.tx_feat_cols_to_normalize) + 
                                        len(self.tx_feat_cols_coa_scale) + 
                                        len(self.tx_feat_cols_similar_scale))  # amount + num_coa_log + COA features + similar txn features
                    if features.shape[1] >= num_cols_to_scale and num_cols_to_scale > 0:
                        scaler = StandardScaler()
                        # Ensure train_indices_np are valid indices for features array
                        valid_train_indices = train_indices_np[train_indices_np < features.shape[0]]
                        if len(valid_train_indices) > 0:
                             # Fit on both continuous and log-normalized features
                             scaler.fit(features[valid_train_indices, :num_cols_to_scale])
                             self.scalers[node_type] = scaler
                             print(f"  Fitted scaler for 'transaction' features ({self.tx_feat_cols_to_scale + self.tx_feat_cols_to_normalize + self.tx_feat_cols_coa_scale + self.tx_feat_cols_similar_scale}) using {len(valid_train_indices)} training samples.")
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
                elif node_type == 'category':
                    # Fit category scaler globally
                    scaler = StandardScaler()
                    scaler.fit(features)
                    self.scalers[node_type] = scaler
                    print(f"  Fitted scaler GLOBALLY for '{node_type}'.")
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

    def _load_precomputed_scalers(self):
        """Load pre-computed scalers from fitted_scalers."""
        if self.fitted_scalers and 'transaction' in self.fitted_scalers:
            # Convert pre-computed scaler parameters to StandardScaler-like objects
            scaler_params = self.fitted_scalers['transaction']
            
            # Create a mock StandardScaler object
            from sklearn.preprocessing import StandardScaler
            mock_scaler = StandardScaler()
            mock_scaler.mean_ = scaler_params['mean_']
            mock_scaler.scale_ = scaler_params['scale_']
            
            self.scalers['transaction'] = mock_scaler
            print(f"  Loaded pre-computed transaction scaler with {len(scaler_params['mean_'])} features")
        
        # Load other scalers if available
        if self.fitted_scalers:
            for scaler_name, scaler_params in self.fitted_scalers.items():
                if scaler_name != 'transaction' and isinstance(scaler_params, dict):
                    mock_scaler = StandardScaler()
                    mock_scaler.mean_ = scaler_params['mean_']
                    mock_scaler.scale_ = scaler_params['scale_']
                    self.scalers[scaler_name] = mock_scaler

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
        
        # Create category map
        valid_categories = df['category_id'].dropna().unique()
        valid_categories = [c for c in valid_categories if isinstance(c, (int, np.integer)) and c != -1]
        category_map = {cat_id: i for i, cat_id in enumerate(valid_categories)}
        num_categories = len(category_map)

        # Store node counts as class attributes instead of directly in graph data
        # to avoid PyTorch Geometric trying to move integers to device
        self._num_transaction_nodes = num_transactions
        self._num_merchant_nodes = num_merchants if num_merchants > 0 else 0
        self._num_category_nodes = num_categories if num_categories > 0 else 0
        
        print(f"  Graph node counts: transaction={num_transactions}, merchant={num_merchants}, category={num_categories}")

        print("  Adding scaled node features...")
        for node_type, raw_feat_array in raw_features.items():
            if raw_feat_array.size > 0: 
                final_features = raw_feat_array.copy() # Start with raw features
                if node_type in self.scalers and self.scalers[node_type] is not None:
                    scaler = self.scalers[node_type]
                    if node_type == 'transaction':
                        num_cols_to_scale = (len(self.tx_feat_cols_to_scale) + 
                                            len(self.tx_feat_cols_to_normalize) + 
                                            len(self.tx_feat_cols_coa_scale) + 
                                            len(self.tx_feat_cols_similar_scale))
                        # Apply scaler fitted on training data TO ALL transaction nodes
                        scaled_part = (raw_feat_array[:, :num_cols_to_scale] - scaler.mean_) / (np.maximum(scaler.scale_, 1e-8))
                        non_scaled_part = raw_feat_array[:, num_cols_to_scale:]
                        final_features = np.concatenate([scaled_part, non_scaled_part], axis=1)
                        print(f"    Applied TRAIN-fitted scaler to ALL 'transaction' nodes ({self.tx_feat_cols_to_scale + self.tx_feat_cols_to_normalize + self.tx_feat_cols_coa_scale + self.tx_feat_cols_similar_scale}).")
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
        
        # Extract text features using refactored method
        raw_text_combined = self._extract_text_features(df)
        
        # Store raw text as class attribute to avoid device transfer issues
        self._transaction_raw_text = raw_text_combined
        if self.use_text_encoder: print(f"  Combined raw text features created (COA included: {self.use_coa_text_features and 'chart_of_accounts_processed' in df.columns}).")

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
                    tx_node_idx = tx_map[idx]
                    merchant_node_idx = merchant_map[merchant_name]
                    
                    # Additional bounds validation
                    if (0 <= tx_node_idx < num_transactions and 
                        0 <= merchant_node_idx < num_merchants):
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

        # 2. Merchant -> Category (categorized_as)
        if num_merchants > 0 and num_categories > 0:
            edge_list, attr_list = [], []
            merchant_primary_category = df.groupby(merchant_col)['category_id'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1)
            merchant_category_confidence = df.groupby(merchant_col)['category_id'].agg(lambda x: x.value_counts(normalize=True).max() if not x.empty else 0)
            
            for merchant_name, primary_cat_id in merchant_primary_category.items():
                if (pd.notna(merchant_name) and merchant_name in merchant_map and 
                    pd.notna(primary_cat_id) and primary_cat_id != -1 and primary_cat_id in category_map):
                    merchant_node_idx = merchant_map[merchant_name]
                    category_node_idx = category_map[primary_cat_id]
                    
                    # Additional bounds validation
                    if (0 <= merchant_node_idx < num_merchants and 
                        0 <= category_node_idx < num_categories):
                        edge_list.append([merchant_node_idx, category_node_idx])
                        confidence = merchant_category_confidence.get(merchant_name, 0)
                        attr_list.append([confidence])
                    
            if edge_list:
                edge_index_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_attr_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(attr_list, dtype=torch.float)
                self.edge_feature_dims[('merchant', 'categorized_as', 'category')] = 1

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
                    
                    # Additional bounds validation
                    if (0 <= tx_node_i < num_transactions and 
                        0 <= tx_node_j < num_transactions and 
                        tx_node_i != tx_node_j):
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
                         if j_pos < num_transactions and j_pos < len(df): 
                            tx_node_j = tx_map[df.index[j_pos]]
                            
                            # Additional bounds validation
                            if (0 <= tx_node_i < num_transactions and 
                                0 <= tx_node_j < num_transactions and 
                                tx_node_i != tx_node_j):
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
        # Store node count as class attribute to avoid device transfer issues
        self._num_transaction_nodes = num_transactions
        self.full_graph_data['transaction'].original_index = torch.tensor(df.index.values, dtype=torch.long)
        # Use the factorized integer 'category_id' for labels
        self.full_graph_data['transaction'].y_global = torch.tensor(df['category_id'].values, dtype=torch.long)
        # No y_user
        self.full_graph_data['transaction'].user_id_code = torch.tensor(df['user_id_code'].values, dtype=torch.long)
        
        # Extract text features using refactored method
        raw_text_combined = self._extract_text_features(df)
        
        # Store raw text as class attribute to avoid device transfer issues  
        self._transaction_raw_text = raw_text_combined
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
    def train_dataloader(self) -> DataLoader:
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
        hgt_loader = HGTLoader(graph_data_cpu, num_samples=valid_hgt_num_samples, shuffle=True,
                               input_nodes=('transaction', self.train_indices.cpu()), batch_size=self.batch_size,
                               num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))
        
        # Wrap the HGTLoader to inject text data
        return TextInjectingDataLoader(hgt_loader, self._transaction_raw_text, self.use_text_encoder)

    def val_dataloader(self) -> DataLoader:
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
        hgt_loader = HGTLoader(graph_data_cpu, num_samples=valid_hgt_num_samples, shuffle=False,
                               input_nodes=('transaction', self.val_indices.cpu()), batch_size=self.batch_size,
                               num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))
        
        # Wrap the HGTLoader to inject text data
        return TextInjectingDataLoader(hgt_loader, self._transaction_raw_text, self.use_text_encoder)

    def test_dataloader(self) -> DataLoader:
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
        hgt_loader = HGTLoader(graph_data_cpu, num_samples=valid_hgt_num_samples, shuffle=False,
                               input_nodes=('transaction', self.test_indices.cpu()), batch_size=self.batch_size,
                               num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))
        
        # Wrap the HGTLoader to inject text data
        return TextInjectingDataLoader(hgt_loader, self._transaction_raw_text, self.use_text_encoder)

    # Add predict_dataloader
    def predict_dataloader(self) -> DataLoader:
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
        hgt_loader = HGTLoader(graph_data_cpu, num_samples=valid_hgt_num_samples, shuffle=False,
                               input_nodes=('transaction', self.test_indices.cpu()), batch_size=self.batch_size,
                               num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))
        
        # Wrap the HGTLoader to inject text data
        return TextInjectingDataLoader(hgt_loader, self._transaction_raw_text, self.use_text_encoder)


