import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # For splitting file paths
from torch.utils.data import Dataset, DataLoader, IterableDataset # Added IterableDataset
from torch_geometric.data import HeteroData, Batch 
# from torch_geometric.loader import HGTLoader # Will be removed
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator # Added Iterator
from collections import defaultdict # Added
import pytorch_lightning as pl
import time 
import os
import functools 
import json 
from sklearn.neighbors import NearestNeighbors 
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.dataset as ds # Added for reading arrow datasets in old code
import pyarrow.parquet # Explicitly import parquet module

# --- ArrowFilesIterableDataset ---
class ArrowFilesIterableDataset(IterableDataset):
    def __init__(self, 
                 file_paths: List[str], 
                 data_module_ref: 'TransactionDataModuleV2', 
                 chunk_size: int = 1, # Number of files to process into one HeteroData object
                 stage: str = 'fit'):
        super().__init__()
        self.file_paths = file_paths
        self.data_module_ref = data_module_ref
        self.chunk_size = chunk_size
        self.stage = stage
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        
        # Store references to pre-fitted/pre-calculated items from the main DataModule
        self.tokenizer = self.data_module_ref.tokenizer
        self.user_map = self.data_module_ref.user_map
        self.category_id_map = self.data_module_ref.category_id_map
        self.scalers = self.data_module_ref.scalers
        self.seq_scalers = self.data_module_ref.seq_scalers
        self.edge_scalers = self.data_module_ref.edge_scalers
        self.user_coa_texts = self.data_module_ref.user_coa_texts
        self.text_max_length = self.data_module_ref.text_max_length
        self.coa_text_max_length = self.data_module_ref.coa_text_max_length
        self.max_seq_length = self.data_module_ref.max_seq_length
        self.tx_feat_cols_to_scale = self.data_module_ref.tx_feat_cols_to_scale
        self.merchant_agg_funcs_named = self.data_module_ref.merchant_agg_funcs_named
        self.use_gnn_encoder = self.data_module_ref.use_gnn_encoder # To know if merchant nodes/edges are needed
        
        if self.tokenizer is None and (self.data_module_ref.use_text_encoder or self.data_module_ref.use_coa_text_features):
            raise RuntimeError(f"Tokenizer not initialized in DataModule before creating ArrowFilesIterableDataset for stage {self.stage}")
        if self.user_map is None or self.category_id_map is None:
            raise RuntimeError(f"User/Category maps not ready in DataModule before creating ArrowFilesIterableDataset for stage {self.stage}")
        # Scalers might be empty if predicting and no numeric features were scaled, so check stage context
        if not self.scalers and self.stage != 'predict' and any(self.data_module_ref.node_feature_dims.get(ntype, 0) > 0 for ntype in ['transaction', 'merchant']):
             print(f"[WARN] ArrowFilesIterableDataset (stage {self.stage}): Scalers might not be fully initialized for all expected node types.")

    def _process_single_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Reads a single Parquet file and performs initial per-file preprocessing."""
        try:
            print(f"[DEBUG _process_single_file] Processing file: {os.path.basename(file_path)}")
            table = pa.parquet.read_table(file_path) 
            df_single = table.to_pandas(split_blocks=True, self_destruct=True) 
            print(f"[DEBUG _process_single_file] Initial df_single shape: {df_single.shape}, Columns: {df_single.columns.tolist()}")

            if df_single.empty:
                print(f"[DEBUG _process_single_file] df_single is empty after reading. Skipping.")
                return None

            required_cols_base = ['company_name']
            # Determine which form the data is in: nested JSON column or flat columns
            has_nested_txn_col = 'target_transaction_processed' in df_single.columns
            has_flat_txn_cols = all(col in df_single.columns for col in ['txn_amount', 'txn_created_date'])
            has_category_str = 'txn_accepted_category_id_str' in df_single.columns
            has_category_int = 'txn_accepted_category_id' in df_single.columns

            # If neither the nested column nor the flat columns exist, skip file
            if not has_nested_txn_col and not has_flat_txn_cols:
                print(f"[DEBUG _process_single_file] Skipping file – no recognised transaction columns (nested or flat).")
                return None

            # Category column check (require either str or int form)
            if not has_category_str and not has_category_int:
                print(f"[DEBUG _process_single_file] Skipping file – missing category column (txn_accepted_category_id[_str]).")
                return None

            # Ensure company_name column exists
            if 'company_name' not in df_single.columns:
                print(f"[DEBUG _process_single_file] Skipping file – missing 'company_name' column.")
                return None

            # ------------------------------------------------------------------
            # 1. Handle category column (ensure string version exists)
            # ------------------------------------------------------------------
            if not has_category_str and has_category_int:
                df_single['txn_accepted_category_id_str'] = df_single['txn_accepted_category_id'].astype(str)
                print(f"[DEBUG _process_single_file] Created 'txn_accepted_category_id_str' from integer column.")

            # ------------------------------------------------------------------
            # 2. Extract / harmonise transaction fields
            # ------------------------------------------------------------------
            extracted_data = []

            if has_nested_txn_col:
                # Existing logic for nested JSON column -----------------------
                print(f"[DEBUG _process_single_file] Processing nested 'target_transaction_processed' column.")
                for i, row_tuple in enumerate(df_single.itertuples()):
                    detailed_log = i < 2
                    txn_dict = {}
                    target_txn_raw = getattr(row_tuple, 'target_transaction_processed')
                    if isinstance(target_txn_raw, dict):
                        txn_dict = target_txn_raw
                    elif isinstance(target_txn_raw, str) and target_txn_raw.strip():
                        try:
                            txn_dict = json.loads(target_txn_raw)
                            if not isinstance(txn_dict, dict):
                                txn_dict = {}
                        except json.JSONDecodeError:
                            txn_dict = {}
                    row_extracted = {
                        'amount': float(txn_dict.get('amount', 0.0)),
                        'timestamp': pd.to_datetime(txn_dict.get('created_date'), errors='coerce'),
                        'description': str(txn_dict.get('description', '')),
                        'memo': str(txn_dict.get('memo', '')),
                        'merchant_name': str(txn_dict.get('payee', ''))
                    }
                    extracted_data.append(row_extracted)
            else:
                # Flat schema ---------------------------------------------------
                print(f"[DEBUG _process_single_file] Processing flat column schema.")
                rename_map = {
                    'txn_amount': 'amount',
                    'txn_description': 'description',
                    'txn_memo': 'memo',
                    'txn_payee': 'merchant_name'
                }
                df_single_flat = df_single.rename(columns={k: v for k, v in rename_map.items() if k in df_single.columns})
                # Timestamps
                if 'txn_created_date' in df_single_flat.columns:
                    df_single_flat['timestamp'] = pd.to_datetime(df_single_flat['txn_created_date'], errors='coerce')
                elif 'txn_ofx_create_date' in df_single_flat.columns:
                    df_single_flat['timestamp'] = pd.to_datetime(df_single_flat['txn_ofx_create_date'], errors='coerce')
                else:
                    df_single_flat['timestamp'] = pd.NaT
                # Ensure mandatory columns exist even if NaN so later logic can run
                for col in ['amount', 'description', 'memo', 'merchant_name']:
                    if col not in df_single_flat.columns:
                        df_single_flat[col] = '' if col in ['description', 'memo', 'merchant_name'] else 0.0
                extracted_data = df_single_flat[['amount', 'timestamp', 'description', 'memo', 'merchant_name']].to_dict('records')

            extracted_df = pd.DataFrame(extracted_data, index=df_single.index)

            # ------------------------------------------------------------------
            # 3. Combine with original dataframe (drop nested JSON column if exists)
            # ------------------------------------------------------------------
            df_processed = pd.concat([
                df_single.drop(columns=['target_transaction_processed'], errors='ignore'),
                extracted_df
            ], axis=1)

            # ------------------------------------------------------------------
            # 4. Chart of accounts handling ------------------------------------
            # ------------------------------------------------------------------
            if 'chart_of_accounts_processed' not in df_processed.columns:
                if 'chart_of_accounts' in df_processed.columns:
                    def _parse_coa(val):
                        if isinstance(val, list):
                            return val
                        if isinstance(val, str) and val.strip():
                            try:
                                loaded = json.loads(val)
                                return loaded if isinstance(loaded, list) else []
                            except json.JSONDecodeError:
                                return []
                        return []
                    df_processed['chart_of_accounts_processed'] = df_processed['chart_of_accounts'].apply(_parse_coa)
                    print("[DEBUG _process_single_file] Parsed 'chart_of_accounts' into 'chart_of_accounts_processed'.")
                else:
                    df_processed['chart_of_accounts_processed'] = [[] for _ in range(len(df_processed))]
                    print("[DEBUG _process_single_file] Added empty 'chart_of_accounts_processed' column (not present in file).")
            # num_chart_of_accounts
            if 'num_chart_of_accounts' not in df_processed.columns:
                df_processed['num_chart_of_accounts'] = df_processed['chart_of_accounts_processed'].apply(lambda x: len(x) if isinstance(x, list) else 0)

            df_processed['weekday'] = df_processed['timestamp'].dt.weekday.fillna(0).astype(int)
            df_processed['hour'] = df_processed['timestamp'].dt.hour.fillna(0).astype(int)
            df_processed['txn_accepted_category_id_str'] = df_processed['txn_accepted_category_id_str'].fillna('UNKNOWN').astype(str)
            df_processed['company_name'] = df_processed['company_name'].fillna('UNKNOWN').astype(str)

            default_cols = {
                'industry_name': ('UNKNOWN', str), 
                'num_chart_of_accounts': (0, int),
            }
            for col_name, (default_val, dtype) in default_cols.items():
                 if col_name not in df_processed.columns: 
                     df_processed[col_name] = default_val
                     print(f"[DEBUG _process_single_file] Column '{col_name}' missing, added with default: {default_val}")
                 else:
                     if dtype == int:
                         df_processed[col_name] = pd.to_numeric(df_processed[col_name],errors='coerce').fillna(default_val)
                     else: # str
                         df_processed[col_name] = df_processed[col_name].fillna(default_val)
                 df_processed[col_name] = df_processed[col_name].astype(dtype)

            cols_to_keep_static = self.data_module_ref.cols_to_keep_after_single_file_proc
            final_df = df_processed[[col for col in cols_to_keep_static if col in df_processed.columns]]
            print(f"[DEBUG _process_single_file] Final df shape before returning: {final_df.shape}. Kept columns: {final_df.columns.tolist()}")
            if final_df.empty:
                print(f"[DEBUG _process_single_file] FINAL DATAFRAME IS EMPTY.")
            return final_df

        except Exception as e:
            print(f"[CRITICAL ERROR] _process_single_file unhandled exception for {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_chunk_to_heterodata(self, current_chunk_file_paths: List[str]) -> Optional[HeteroData]:
        chunk_dfs = []
        for fp in current_chunk_file_paths:
            df_file = self._process_single_file(fp)
            if df_file is not None and not df_file.empty:
                chunk_dfs.append(df_file)
        if not chunk_dfs:
            print(f"[DEBUG _process_chunk_to_heterodata] Stage: {self.stage}, Files: {current_chunk_file_paths}, RESULT: EARLY EXIT (no valid dfs in chunk)")
            return None
        chunk_df = pd.concat(chunk_dfs, ignore_index=True)
        if chunk_df.empty: return None

        map_user_str_to_int = {v: k for k, v in self.user_map.items()}
        map_cat_str_to_int = {v: k for k, v in self.category_id_map.items()}
        unknown_user_code = map_user_str_to_int.get('UNKNOWN', -1)
        unknown_category_code = map_cat_str_to_int.get('UNKNOWN', -1)
        chunk_df['user_id_code'] = chunk_df['company_name'].map(map_user_str_to_int).fillna(unknown_user_code).astype(int)
        chunk_df['category_id'] = chunk_df['txn_accepted_category_id_str'].map(map_cat_str_to_int).fillna(unknown_category_code).astype(int)
        chunk_df.sort_values(['user_id_code', 'timestamp'], inplace=True)
        chunk_df.reset_index(drop=True, inplace=True)

        graph_chunk = HeteroData()
        num_transactions_chunk = len(chunk_df)
        graph_chunk['transaction'].num_nodes = num_transactions_chunk
        
        # Raw features for transaction
        df = chunk_df
        for col in ['amount', 'hour', 'weekday', 'num_chart_of_accounts']:
            if col not in df.columns: df[col] = 0.0 if col =='amount' else 0 # default values
        amount = df['amount'].fillna(0.0); num_coa = df['num_chart_of_accounts'].fillna(0).astype(int)
        hour = df['hour'].fillna(0).astype(int); day = df['weekday'].fillna(0).astype(int)
        hour_sin = np.sin(2*np.pi*hour/24); hour_cos = np.cos(2*np.pi*hour/24)
        day_sin = np.sin(2*np.pi*day/7); day_cos = np.cos(2*np.pi*day/7)
        raw_tx_features = np.column_stack([amount, num_coa, hour_sin, hour_cos, day_sin, day_cos]).astype(np.float64)
        
        # Scale transaction features
        final_tx_features = raw_tx_features.copy()
        tx_scaler = self.scalers.get('transaction')
        if tx_scaler is not None:
            num_cols_to_scale = len(self.tx_feat_cols_to_scale)
            # Ensure raw_tx_features has enough columns before slicing
            if raw_tx_features.shape[1] >= num_cols_to_scale:
                scaled_part = (raw_tx_features[:, :num_cols_to_scale] - tx_scaler.mean_) / (np.maximum(tx_scaler.scale_, 1e-8))
                final_tx_features = np.concatenate([scaled_part, raw_tx_features[:, num_cols_to_scale:]], axis=1)
            else:
                print(f"[WARN] _process_chunk_to_heterodata: raw_tx_features ({raw_tx_features.shape}) has fewer columns than tx_feat_cols_to_scale ({num_cols_to_scale}). Using raw features for transactions.")
        graph_chunk['transaction'].x = torch.tensor(final_tx_features, dtype=torch.float)

        # Merchant features and nodes (if GNN is used)
        if self.use_gnn_encoder:
            merchant_col = 'merchant_name'
            if merchant_col in df.columns:
                merchants_in_chunk = df[merchant_col].dropna().unique()
                merchant_map_chunk = {name: i for i, name in enumerate(merchants_in_chunk)}
                num_merchants_in_chunk = len(merchant_map_chunk)
                if num_merchants_in_chunk > 0:
                    graph_chunk['merchant'].num_nodes = num_merchants_in_chunk
                    merchant_stats_chunk_df = df[pd.notna(df[merchant_col])].groupby(merchant_col).agg(**self.merchant_agg_funcs_named)
                    merchant_stats_chunk_df = merchant_stats_chunk_df.fillna(0).reindex(merchants_in_chunk, fill_value=0)
                    raw_merchant_features = merchant_stats_chunk_df[list(self.merchant_agg_funcs_named.keys())].values.astype(np.float64)
                    
                    final_merchant_features = raw_merchant_features.copy()
                    merchant_scaler = self.scalers.get('merchant')
                    if merchant_scaler is not None:
                        final_merchant_features = (raw_merchant_features - merchant_scaler.mean_) / (np.maximum(merchant_scaler.scale_, 1e-8))
                    graph_chunk['merchant'].x = torch.tensor(final_merchant_features, dtype=torch.float)
                else: # No merchants found in this chunk after dropna/unique
                    graph_chunk['merchant'].num_nodes = 0
                    graph_chunk['merchant'].x = torch.empty((0, len(self.merchant_agg_funcs_named) if self.merchant_agg_funcs_named else 8), dtype=torch.float) # default to 8 if funcs_named is empty
            else: # No merchant column in the dataframe chunk
                graph_chunk['merchant'].num_nodes = 0
                graph_chunk['merchant'].x = torch.empty((0, len(self.merchant_agg_funcs_named) if self.merchant_agg_funcs_named else 8), dtype=torch.float)

        graph_chunk['transaction'].y_global = torch.tensor(chunk_df['category_id'].values, dtype=torch.long)
        graph_chunk['transaction'].user_id_code = torch.tensor(chunk_df['user_id_code'].values, dtype=torch.long) # Global user IDs
        graph_chunk['transaction'].original_index = torch.tensor(chunk_df.index.values, dtype=torch.long) # Index within this chunk_df

        if self.data_module_ref.use_text_encoder:
            text_cols = ['description', 'memo', 'merchant_name']
            raw_tx_texts_chunk = chunk_df[text_cols].astype(str).agg(' || '.join, axis=1).tolist()
            tokenized_tx = self.tokenizer(raw_tx_texts_chunk, padding='max_length', truncation=True, max_length=self.text_max_length, return_tensors='pt')
            graph_chunk['transaction'].input_ids = tokenized_tx['input_ids']
            graph_chunk['transaction'].attention_mask = tokenized_tx['attention_mask']

        if self.data_module_ref.use_sequence_encoder:
            all_seq_features_chunk, all_seq_lengths_chunk = [], []
            sequence_feature_dim = 6
            for idx in range(num_transactions_chunk):
                row = chunk_df.iloc[idx]; user = row['user_id_code']; time = row['timestamp']
                start = max(0, idx - self.max_seq_length)
                prev_txs = chunk_df.iloc[start:idx][chunk_df.iloc[start:idx]['user_id_code'] == user]
                seq_feats = []
                if not prev_txs.empty:
                    for _, p_row in prev_txs.iterrows():
                        td = (time - p_row['timestamp']).total_seconds() if pd.notna(time) and pd.notna(p_row['timestamp']) else 0
                        td_scaled = td; scaler_td = self.seq_scalers.get('time_delta')
                        if scaler_td: td_scaled = (td - scaler_td.mean_[0]) / (np.maximum(scaler_td.scale_[0],1e-8))
                        h, d = p_row['hour'], p_row['weekday']
                        seq_feats.append([p_row['amount'],np.sin(2*np.pi*d/7),np.cos(2*np.pi*d/7),np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),td_scaled])
                all_seq_lengths_chunk.append(len(seq_feats))
                all_seq_features_chunk.append(torch.tensor(seq_feats[-self.max_seq_length:], dtype=torch.float) if seq_feats else torch.empty((0,sequence_feature_dim),dtype=torch.float))
            graph_chunk['transaction'].seq_features = pad_sequence(all_seq_features_chunk, batch_first=True, padding_value=0.0)
            graph_chunk['transaction'].seq_lengths = torch.tensor(all_seq_lengths_chunk, dtype=torch.long)

        if self.use_gnn_encoder and graph_chunk.get('merchant') and graph_chunk['merchant'].num_nodes > 0:
            tx_map_chunk = {orig_idx: i for i, orig_idx in enumerate(chunk_df.index)}
            edge_list_tx_merch, attr_list_tx_merch = [], []
            merchant_stats_map = chunk_df.groupby(merchant_col)['amount'].agg(['mean','std']).fillna(0).to_dict('index')
            for idx_in_chunkdf, row_data in chunk_df.iterrows():
                merch_name = row_data[merchant_col]
                if pd.notna(merch_name) and merch_name in merchant_map_chunk:
                    tx_node_local = tx_map_chunk[idx_in_chunkdf]
                    merch_node_local = merchant_map_chunk[merch_name]
                    edge_list_tx_merch.append([tx_node_local, merch_node_local])
                    stats = merchant_stats_map.get(merch_name, {'mean':0,'std':0})
                    attr_list_tx_merch.append([(float(row_data['amount']) - stats['mean'])/(stats['std']+1e-8 if stats['std'] != 0 else 1e-8)]) # Avoid div by zero if std is exactly 0
            if edge_list_tx_merch:
                graph_chunk['transaction', 'belongs_to', 'merchant'].edge_index = torch.tensor(edge_list_tx_merch, dtype=torch.long).t().contiguous()
                graph_chunk['transaction', 'belongs_to', 'merchant'].edge_attr = torch.tensor(attr_list_tx_merch, dtype=torch.float)
        
        mask = torch.ones(num_transactions_chunk, dtype=torch.bool)
        if self.stage == 'fit': graph_chunk['transaction'].train_mask = mask
        elif self.stage == 'validate': graph_chunk['transaction'].val_mask = mask
        else: graph_chunk['transaction'].test_mask = mask # Covers test and predict

        if graph_chunk is None or not hasattr(graph_chunk, 'num_nodes') or graph_chunk.num_nodes == 0:
            print(f"[DEBUG _process_chunk_to_heterodata] Stage: {self.stage}, Files: {current_chunk_file_paths}, RESULT: EMPTY Graph (num_nodes=0 or None)")
        else:
            print(f"[DEBUG _process_chunk_to_heterodata] Stage: {self.stage}, Files: {current_chunk_file_paths}, RESULT: Graph with num_nodes={graph_chunk.num_nodes}")

        return graph_chunk

    def __iter__(self) -> Iterator[HeteroData]:
        if self.stage == 'validate':
            print(f"[DEBUG __iter__ VAL] Entered __iter__ for validation. Files: {self.file_paths}")

        worker_info = torch.utils.data.get_worker_info()
        files_to_process_by_worker = self.file_paths
        if worker_info is not None:
            per_worker = int(np.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            files_to_process_by_worker = self.file_paths[iter_start:iter_end]

        for i in range(0, len(files_to_process_by_worker), self.chunk_size):
            current_chunk_file_paths = files_to_process_by_worker[i : i + self.chunk_size]
            if not current_chunk_file_paths: continue
            if self.stage == 'validate':
                print(f"[DEBUG __iter__ VAL] About to call _process_chunk_to_heterodata for chunk: {current_chunk_file_paths}")
            graph_data = self._process_chunk_to_heterodata(current_chunk_file_paths)
            if graph_data is not None and graph_data['transaction'].num_nodes > 0:
                yield graph_data

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

# --- V2 DataModule with IterableDataset --- 
class TransactionDataModuleV2(pl.LightningDataModule):
    """
    DataModule V2: Processes Arrow files iteratively using ArrowFilesIterableDataset.
    Handles per-chunk graph creation, feature engineering, and global state management 
    (scalers, maps) fitted on training data.
    """
    def __init__(self, 
                 file_paths: List[str], 
                 batch_size: int = 32,
                 iterable_dataset_chunk_size: int = 1,
                 shuffle_files_before_split: bool = True,
                 num_workers: int = 0, 
                 max_seq_length: int = 50,
                 text_model_name: str = 'bert-base-uncased',
                 text_max_length: int = 128,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 use_sequence_encoder: bool = True,
                 use_text_encoder: bool = True,
                 use_gnn_encoder: bool = True,
                 use_coa_text_features: bool = True,
                 coa_text_max_length: int = 256,
                 fitted_scalers: Optional[Dict[str, StandardScaler]] = None,
                 fitted_seq_scalers: Optional[Dict[str, StandardScaler]] = None,
                 fitted_edge_scalers: Optional[Dict[str, StandardScaler]] = None,
                 fitted_user_map: Optional[Dict[int, Any]] = None,
                 fitted_category_id_map: Optional[Dict[int, Any]] = None,
                 model_config: Optional[Dict] = None,
                 include_edge_types: Optional[List[str]] = None
                 ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                'file_paths', 'fitted_scalers', 'user_map', 
                'category_id_map', 'user_coa_texts', 'global_user_coa_tensors',
                'graph_metadata' # Avoid saving potentially large tensors/data in hparams
            ]
        )
        self._model_config = model_config # Store model_config

        self.all_file_paths = sorted(list(set(file_paths))) # Ensure unique and sorted
        self.is_predicting = (fitted_scalers is not None and fitted_user_map is not None and fitted_category_id_map is not None)

        if shuffle_files_before_split and not self.is_predicting:
            np.random.RandomState(seed=42).shuffle(self.all_file_paths)

        self.batch_size = batch_size
        self.iterable_dataset_chunk_size = iterable_dataset_chunk_size
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
        self.coa_text_max_length = coa_text_max_length

        self.fitted_scalers = fitted_scalers
        self.fitted_seq_scalers = fitted_seq_scalers
        self.fitted_edge_scalers = fitted_edge_scalers
        self.fitted_user_map = fitted_user_map
        self.fitted_category_id_map = fitted_category_id_map
        
        self.tokenizer = None
        self.node_feature_dims: Dict[str, int] = {}
        self.edge_feature_dims: Dict[Tuple[str, str, str], int] = {}
        self.sequence_feature_dim: Optional[int] = 6 
        
        self.scalers: Dict[str, StandardScaler] = {} if fitted_scalers is None else fitted_scalers
        self.seq_scalers: Dict[str, StandardScaler] = {} if fitted_seq_scalers is None else fitted_seq_scalers
        self.edge_scalers: Dict[str, StandardScaler] = {} if fitted_edge_scalers is None else fitted_edge_scalers
        self.user_map = None if fitted_user_map is None else fitted_user_map
        self.category_id_map = None if fitted_category_id_map is None else fitted_category_id_map
        
        self.num_users = len(self.user_map) if self.user_map else 0
        self.num_global_classes = len(self.category_id_map) if self.category_id_map else 0
        self.user_coa_texts: Dict[int, str] = {} 
        self.raw_user_coa_data: Dict[str, Any] = {}

        # --- File Path Splitting ---
        if not self.all_file_paths:
            raise ValueError("Received an empty list of file_paths for the DataModule.")

        if self.is_predicting:
            self.train_files: List[str] = []
            self.val_files: List[str] = []
            self.predict_files: List[str] = list(self.all_file_paths)
            self.test_files: List[str] = [] 
            print(f"Prediction mode: Using all {len(self.predict_files)} files for prediction set.")
        else:
            self.predict_files = []
            # Ensure val_ratio and test_ratio are floats for comparison
            current_val_ratio = float(self.val_ratio) if self.val_ratio is not None else 0.1 # Default to 0.1 if None
            current_test_ratio = float(self.test_ratio) if self.test_ratio is not None else 0.1 # Default to 0.1 if None

            if len(self.all_file_paths) == 1:
                print(f"[INFO] Only 1 file available. Using it for training. Validation and testing will be skipped.")
                self.train_files = list(self.all_file_paths)
                self.val_files = []
                self.test_files = []
            elif len(self.all_file_paths) == 2:
                if current_val_ratio > 0 and current_test_ratio == 0:
                    print(f"[INFO] 2 files available. Assigning 1 for training and 1 for validation.")
                    # Ensure consistent assignment for reproducibility if shuffle_files_before_split is False
                    self.train_files = [self.all_file_paths[0]]
                    self.val_files = [self.all_file_paths[1]]
                    self.test_files = []
                elif current_test_ratio > 0 and current_val_ratio == 0:
                    print(f"[INFO] 2 files available. Assigning 1 for training and 1 for testing.")
                    self.train_files = [self.all_file_paths[0]]
                    self.test_files = [self.all_file_paths[1]]
                    self.val_files = []
                elif current_val_ratio > 0 and current_test_ratio > 0: # Both want a share, prioritize test
                    print(f"[INFO] 2 files available with val_ratio > 0 and test_ratio > 0. Assigning 1 for training and 1 for testing (test takes precedence over val). Val will be empty.")
                    self.train_files = [self.all_file_paths[0]]
                    self.test_files = [self.all_file_paths[1]]
                    self.val_files = []
                else: # Both ratios are 0 or invalid
                    print(f"[INFO] 2 files available. Both val_ratio and test_ratio are 0. Assigning both for training.")
                    self.train_files = list(self.all_file_paths)
                    self.val_files = []
                    self.test_files = []
            elif len(self.all_file_paths) < 3 and (current_val_ratio > 0 or current_test_ratio > 0): # This case should be covered by len == 1 or 2 now
                 print(f"[WARN] Not enough files ({len(self.all_file_paths)}) for a full train/val/test split with ratios val={current_val_ratio}, test={current_test_ratio}. Using all for training. This path should ideally not be hit if len 1 or 2 handled.")
                 self.train_files = list(self.all_file_paths)
                 self.val_files = [] 
                 self.test_files = []
            elif current_val_ratio == 0 and current_test_ratio == 0 : # Use all for training if ratios are zero
                 self.train_files = list(self.all_file_paths)
                 self.val_files = []
                 self.test_files = []
            else: # Proceed with splitting for 3+ files
                if not (0 <= current_val_ratio < 1 and 0 <= current_test_ratio < 1 and (current_val_ratio + current_test_ratio) < 1):
                    raise ValueError(f"Invalid val_ratio ({current_val_ratio}) or test_ratio ({current_test_ratio}). Sum must be < 1.")

                train_intermediate_files, self.test_files = train_test_split(
                    self.all_file_paths, test_size=current_test_ratio, random_state=42, shuffle=False) 
                
                if not train_intermediate_files: 
                     self.train_files = []
                     self.val_files = []
                     if not self.test_files: 
                          print("[WARN] No files left for train, val, or test after splitting.")
                elif current_val_ratio == 0: 
                    self.train_files = train_intermediate_files
                    self.val_files = []
                else:
                    if (1.0 - current_test_ratio) <= 0 : 
                        effective_val_ratio = 0 
                        if train_intermediate_files: 
                             self.train_files = train_intermediate_files
                             self.val_files = []
                        else: 
                             self.train_files = []
                             self.val_files = []
                    elif not train_intermediate_files: 
                         self.train_files = []
                         self.val_files = []
                    else:
                        effective_val_ratio = current_val_ratio / (1.0 - current_test_ratio)
                        if effective_val_ratio >= 1.0 and train_intermediate_files: 
                             self.train_files = []
                             self.val_files = train_intermediate_files
                        elif effective_val_ratio == 0 and train_intermediate_files: 
                             self.train_files = train_intermediate_files
                             self.val_files = []
                        elif train_intermediate_files: 
                             self.train_files, self.val_files = train_test_split(
                                 train_intermediate_files, test_size=effective_val_ratio, random_state=42, shuffle=False)
                        else: 
                             self.train_files = []
                             self.val_files = []
            
            if not self.train_files and not self.is_predicting and (self.val_files or self.test_files):
                print("[WARN] Training file list is empty after split, but val/test files exist. This might indicate an issue with ratios or dataset size.")


        print(f"File splits: Train={len(self.train_files)}, Val={len(self.val_files)}, Test={len(self.test_files)}, Predict={len(self.predict_files)}")
        
        self.tx_feat_cols_to_scale = ['amount', 'num_chart_of_accounts'] 
        self.merchant_agg_funcs_named = {
            'amount_mean': ('amount', 'mean'), 'amount_std': ('amount', lambda x: x.std(ddof=0)),
            'amount_max': ('amount', 'max'), 'amount_min': ('amount', 'min'),
            'amount_count': ('amount', 'count'), 'amount_median': ('amount', 'median'),
            'amount_q25': ('amount', lambda x: x.quantile(0.25)), 'amount_q75': ('amount', lambda x: x.quantile(0.75))
        }
        # Static list of columns to keep after _process_single_file
        self.cols_to_keep_after_single_file_proc = [
            'txn_accepted_category_id_str', 'company_name', 'industry_name', 
            'num_chart_of_accounts', 'chart_of_accounts_processed', 
            'amount', 'timestamp', 'description', 'memo', 'merchant_name', 
            'weekday', 'hour'
        ]

        print(f"TransactionDataModuleV2 (Iterable) initialized. Iterable chunk size: {self.iterable_dataset_chunk_size}")
        print(f"  Modality Flags: GNN={self.use_gnn_encoder}, Sequence={self.use_sequence_encoder}, Text={self.use_text_encoder}, COA_Text={self.use_coa_text_features}")

    def _process_single_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Reads a single Parquet file and performs initial per-file preprocessing."""
        try:
            print(f"[DEBUG _process_single_file] Processing file: {os.path.basename(file_path)}")
            table = pa.parquet.read_table(file_path) 
            df_single = table.to_pandas(split_blocks=True, self_destruct=True) 
            print(f"[DEBUG _process_single_file] Initial df_single shape: {df_single.shape}, Columns: {df_single.columns.tolist()}")

            if df_single.empty:
                print(f"[DEBUG _process_single_file] df_single is empty after reading. Skipping.")
                return None

            required_cols_base = ['company_name']
            # Determine which form the data is in: nested JSON column or flat columns
            has_nested_txn_col = 'target_transaction_processed' in df_single.columns
            has_flat_txn_cols = all(col in df_single.columns for col in ['txn_amount', 'txn_created_date'])
            has_category_str = 'txn_accepted_category_id_str' in df_single.columns
            has_category_int = 'txn_accepted_category_id' in df_single.columns

            # If neither the nested column nor the flat columns exist, skip file
            if not has_nested_txn_col and not has_flat_txn_cols:
                print(f"[DEBUG _process_single_file] Skipping file – no recognised transaction columns (nested or flat).")
                return None

            # Category column check (require either str or int form)
            if not has_category_str and not has_category_int:
                print(f"[DEBUG _process_single_file] Skipping file – missing category column (txn_accepted_category_id[_str]).")
                return None

            # Ensure company_name column exists
            if 'company_name' not in df_single.columns:
                print(f"[DEBUG _process_single_file] Skipping file – missing 'company_name' column.")
                return None

            # ------------------------------------------------------------------
            # 1. Handle category column (ensure string version exists)
            # ------------------------------------------------------------------
            if not has_category_str and has_category_int:
                df_single['txn_accepted_category_id_str'] = df_single['txn_accepted_category_id'].astype(str)
                print(f"[DEBUG _process_single_file] Created 'txn_accepted_category_id_str' from integer column.")

            # ------------------------------------------------------------------
            # 2. Extract / harmonise transaction fields
            # ------------------------------------------------------------------
            extracted_data = []

            if has_nested_txn_col:
                # Existing logic for nested JSON column -----------------------
                print(f"[DEBUG _process_single_file] Processing nested 'target_transaction_processed' column.")
                for i, row_tuple in enumerate(df_single.itertuples()):
                    detailed_log = i < 2
                    txn_dict = {}
                    target_txn_raw = getattr(row_tuple, 'target_transaction_processed')
                    if isinstance(target_txn_raw, dict):
                        txn_dict = target_txn_raw
                    elif isinstance(target_txn_raw, str) and target_txn_raw.strip():
                        try:
                            txn_dict = json.loads(target_txn_raw)
                            if not isinstance(txn_dict, dict):
                                txn_dict = {}
                        except json.JSONDecodeError:
                            txn_dict = {}
                    row_extracted = {
                        'amount': float(txn_dict.get('amount', 0.0)),
                        'timestamp': pd.to_datetime(txn_dict.get('created_date'), errors='coerce'),
                        'description': str(txn_dict.get('description', '')),
                        'memo': str(txn_dict.get('memo', '')),
                        'merchant_name': str(txn_dict.get('payee', ''))
                    }
                    extracted_data.append(row_extracted)
            else:
                # Flat schema ---------------------------------------------------
                print(f"[DEBUG _process_single_file] Processing flat column schema.")
                rename_map = {
                    'txn_amount': 'amount',
                    'txn_description': 'description',
                    'txn_memo': 'memo',
                    'txn_payee': 'merchant_name'
                }
                df_single_flat = df_single.rename(columns={k: v for k, v in rename_map.items() if k in df_single.columns})
                # Timestamps
                if 'txn_created_date' in df_single_flat.columns:
                    df_single_flat['timestamp'] = pd.to_datetime(df_single_flat['txn_created_date'], errors='coerce')
                elif 'txn_ofx_create_date' in df_single_flat.columns:
                    df_single_flat['timestamp'] = pd.to_datetime(df_single_flat['txn_ofx_create_date'], errors='coerce')
                else:
                    df_single_flat['timestamp'] = pd.NaT
                # Ensure mandatory columns exist even if NaN so later logic can run
                for col in ['amount', 'description', 'memo', 'merchant_name']:
                    if col not in df_single_flat.columns:
                        df_single_flat[col] = '' if col in ['description', 'memo', 'merchant_name'] else 0.0
                extracted_data = df_single_flat[['amount', 'timestamp', 'description', 'memo', 'merchant_name']].to_dict('records')

            extracted_df = pd.DataFrame(extracted_data, index=df_single.index)

            # ------------------------------------------------------------------
            # 3. Combine with original dataframe (drop nested JSON column if exists)
            # ------------------------------------------------------------------
            df_processed = pd.concat([
                df_single.drop(columns=['target_transaction_processed'], errors='ignore'),
                extracted_df
            ], axis=1)

            # ------------------------------------------------------------------
            # 4. Chart of accounts handling ------------------------------------
            # ------------------------------------------------------------------
            if 'chart_of_accounts_processed' not in df_processed.columns:
                if 'chart_of_accounts' in df_processed.columns:
                    def _parse_coa(val):
                        if isinstance(val, list):
                            return val
                        if isinstance(val, str) and val.strip():
                            try:
                                loaded = json.loads(val)
                                return loaded if isinstance(loaded, list) else []
                            except json.JSONDecodeError:
                                return []
                        return []
                    df_processed['chart_of_accounts_processed'] = df_processed['chart_of_accounts'].apply(_parse_coa)
                    print("[DEBUG _process_single_file] Parsed 'chart_of_accounts' into 'chart_of_accounts_processed'.")
                else:
                    df_processed['chart_of_accounts_processed'] = [[] for _ in range(len(df_processed))]
                    print("[DEBUG _process_single_file] Added empty 'chart_of_accounts_processed' column (not present in file).")
            # num_chart_of_accounts
            if 'num_chart_of_accounts' not in df_processed.columns:
                df_processed['num_chart_of_accounts'] = df_processed['chart_of_accounts_processed'].apply(lambda x: len(x) if isinstance(x, list) else 0)

            df_processed['weekday'] = df_processed['timestamp'].dt.weekday.fillna(0).astype(int)
            df_processed['hour'] = df_processed['timestamp'].dt.hour.fillna(0).astype(int)
            df_processed['txn_accepted_category_id_str'] = df_processed['txn_accepted_category_id_str'].fillna('UNKNOWN').astype(str)
            df_processed['company_name'] = df_processed['company_name'].fillna('UNKNOWN').astype(str)

            default_cols = {
                'industry_name': ('UNKNOWN', str), 
                'num_chart_of_accounts': (0, int),
            }
            for col_name, (default_val, dtype) in default_cols.items():
                 if col_name not in df_processed.columns: 
                     df_processed[col_name] = default_val
                     print(f"[DEBUG _process_single_file] Column '{col_name}' missing, added with default: {default_val}")
                 else:
                     if dtype == int:
                         df_processed[col_name] = pd.to_numeric(df_processed[col_name],errors='coerce').fillna(default_val)
                     else: # str
                         df_processed[col_name] = df_processed[col_name].fillna(default_val)
                 df_processed[col_name] = df_processed[col_name].astype(dtype)

            cols_to_keep_static = self.cols_to_keep_after_single_file_proc
            final_df = df_processed[[col for col in cols_to_keep_static if col in df_processed.columns]]
            print(f"[DEBUG _process_single_file] Final df shape before returning: {final_df.shape}. Kept columns: {final_df.columns.tolist()}")
            if final_df.empty:
                print(f"[DEBUG _process_single_file] FINAL DATAFRAME IS EMPTY.")
            return final_df

        except Exception as e:
            print(f"[CRITICAL ERROR] _process_single_file unhandled exception for {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def prepare_data(self):
        # This method is for downloading, and one-time global setup like fitting scalers, building maps.
        # Pass 1: Iterate through files to fit scalers and build global maps (user_map, category_id_map)
        # and collect raw COA data.
        if self.is_predicting: # Check if we are in prediction mode with pre-fitted state
            print("Prepare_data: In prediction mode with pre-fitted state. Skipping scaler/map fitting.")
            if self.fitted_user_map: self.num_users = len(self.fitted_user_map)
            if self.fitted_category_id_map: self.num_global_classes = len(self.fitted_category_id_map)
            if self.use_coa_text_features and not self.user_coa_texts and self.num_users > 0: 
                # If predicting and COA is used but texts not loaded, initialize empty dict for safety
                print("[WARN] Prediction mode: use_coa_text_features=True, but self.user_coa_texts is empty. Initializing empty COA texts.")
                self.user_coa_texts = {i: "" for i in range(self.num_users)}
            if self.tokenizer is None and (self.use_text_encoder or self.use_coa_text_features):
                 self._init_tokenizer()
            return
        
        # Proceed with fitting if not in predict mode or if essential fitted state is missing
        print(f"--- Starting DataModuleV2 Prepare Data (Pass 1: Scalers, Maps, COA Collection) ---")
        prepare_start_time = time.time()

        if self.tokenizer is None and (self.use_text_encoder or self.use_coa_text_features):
            self._init_tokenizer()

        all_transaction_features_to_scale_collector = [] 
        all_merchant_transactions_collector: Dict[str, List[float]] = defaultdict(list) 
        all_sequence_time_deltas_collector = []
        category_id_strings_collector = set()
        user_company_names_collector = set()
        self.raw_user_coa_data = {} 

        # self.tx_feat_cols_to_scale and self.merchant_agg_funcs_named are already set in __init__

        files_for_pass1 = self.train_files # CRITICAL: Use only training files for fitting
        if not files_for_pass1:
            print("[WARN] No training files available for Pass 1 (scaler fitting, map building). Scalers/maps might be empty or suboptimal.")
            # Optionally, could raise an error or use all_file_paths with a strong warning if this path is critical
            # For now, proceed, and scalers/maps might end up empty if no train_files.
        else:
            print(f"Iterating over {len(files_for_pass1)} training files for Pass 1 (scaler fitting, map building)...")
        
        for file_idx, file_path in enumerate(files_for_pass1):
            if file_idx % 200 == 0 and file_idx > 0: 
                 print(f"    Processed {file_idx}/{len(files_for_pass1)} training files for Pass 1...")
            
            df_file = self._process_single_file(file_path)
            if df_file is None or df_file.empty:
                continue

            if not df_file[self.tx_feat_cols_to_scale].empty:
                 all_transaction_features_to_scale_collector.append(df_file[self.tx_feat_cols_to_scale].values)

            # Collect raw data for merchant features
            for _, row in df_file.iterrows():
                merchant_name_str = str(row['merchant_name'])
                amount_val = float(row['amount'])
                if merchant_name_str != 'UNKNOWN' and pd.notna(amount_val):
                    all_merchant_transactions_collector[merchant_name_str].append(amount_val)

            # Collect raw data for sequence time delta scaler (intra-file, per-user)
            df_file_sorted_for_seq = df_file.sort_values(['company_name', 'timestamp'])
            for _, user_group in df_file_sorted_for_seq.groupby('company_name', sort=False):
                if len(user_group) > 1:
                    time_diffs_seconds = user_group['timestamp'].diff().dt.total_seconds()
                    # Keep only positive time differences (current - previous)
                    valid_time_diffs = time_diffs_seconds[time_diffs_seconds > 0].tolist()
                    all_sequence_time_deltas_collector.extend(valid_time_diffs)

            category_id_strings_collector.update(df_file['txn_accepted_category_id_str'].unique())
            user_company_names_collector.update(df_file['company_name'].unique())

            if self.use_coa_text_features:
                for _, row in df_file.iterrows():
                    company_name_str = str(row['company_name'])
                    if company_name_str != 'UNKNOWN' and company_name_str not in self.raw_user_coa_data:
                        self.raw_user_coa_data[company_name_str] = row['chart_of_accounts_processed'] 
        
        if not self.is_predicting: # This block should only run if we are fitting, not using pre-fitted state
            self._fit_scalers_from_collected(
                all_transaction_features_to_scale_collector,
                all_merchant_transactions_collector, 
                all_sequence_time_deltas_collector
            )
            self._create_mappings_from_collected(category_id_strings_collector, user_company_names_collector)
            self._process_collected_coa_text()
        
        print(f"Final Counts after Pass 1: Global Classes={self.num_global_classes}, Users={self.num_users}")
        print(f"--- DataModuleV2 Prepare Data (Pass 1) finished in {time.time() - prepare_start_time:.2f}s ---")

    def _init_tokenizer(self):
        print(f"Downloading/loading tokenizer: {self.text_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, force_download=True)
            print("Tokenizer ready.")
        except Exception as e:
            print(f"[ERROR] Failed to download/load tokenizer '{self.text_model_name}': {e}")
            raise e

    def _fit_scalers_from_collected(self, 
                                    all_transaction_features_to_scale_collector,
                                    all_merchant_transactions_collector: Dict[str, List[float]],
                                    all_sequence_time_deltas_collector: List[float]):
        print("Fitting scalers based on collected data...")
        # 1. Transaction Feature Scaler
        if all_transaction_features_to_scale_collector:
            all_tx_scale_np = np.concatenate(all_transaction_features_to_scale_collector, axis=0)
            if all_tx_scale_np.shape[0] > 0 and all_tx_scale_np.shape[1] == len(self.tx_feat_cols_to_scale):
                tx_scaler = StandardScaler().fit(all_tx_scale_np)
                self.scalers['transaction'] = tx_scaler
                print(f"  Fitted scaler for 'transaction' features ({self.tx_feat_cols_to_scale}).")
                # Create proxy edge scaler after transaction scaler is fitted
                self._create_proxy_edge_scaler() 
            else:
                print(f"  [WARN] Not enough data or mismatched columns for transaction scaler. Data shape: {all_tx_scale_np.shape if isinstance(all_tx_scale_np, np.ndarray) else 'N/A'}")
                self.scalers['transaction'] = None
        else:
            print("  [WARN] No data collected for transaction feature scaler.")
            self.scalers['transaction'] = None
        
        # 2. Merchant Feature Scaler
        if all_merchant_transactions_collector:
            merchant_names_fit = list(all_merchant_transactions_collector.keys())
            merchant_features_to_scale_list = []
            for merchant_name_item in merchant_names_fit:
                amounts = pd.Series(all_merchant_transactions_collector[merchant_name_item])
                if not amounts.empty:
                    stats = [
                        amounts.mean(),
                        amounts.std(ddof=0),
                        amounts.max(),
                        amounts.min(),
                        amounts.count(),
                        amounts.median(),
                        amounts.quantile(0.25),
                        amounts.quantile(0.75)
                    ]
                    merchant_features_to_scale_list.append(stats)
            
            if merchant_features_to_scale_list:
                merchant_features_np = np.array(merchant_features_to_scale_list, dtype=np.float64)
                # Fill any NaNs that might result from single-transaction merchants (e.g. std becomes NaN)
                merchant_features_np = np.nan_to_num(merchant_features_np, nan=0.0)
                if merchant_features_np.shape[0] > 0:
                    merchant_scaler = StandardScaler().fit(merchant_features_np)
                    self.scalers['merchant'] = merchant_scaler
                    # Store the merchant names in the order their features were used for fitting the scaler
                    # This isn't strictly necessary if _calculate_raw_features re-computes and re-orders
                    # but good for consistency check if needed.
                    # self.fitted_merchant_order_for_scaler = merchant_names_fit 
                    print(f"  Fitted scaler for 'merchant' features. Input shape: {merchant_features_np.shape}")
                else:
                    print("  [WARN] No valid merchant features computed for scaler fitting.")
                    self.scalers['merchant'] = None
            else:
                print("  [WARN] No merchant transaction data processed for scaler fitting.")
                self.scalers['merchant'] = None

        # 3. Edge attribute scaler (merchant amount diff) -------------------
        edge_attr_values = []  # Collect normalized differences to fit scaler
        for merchant_name_item, amounts_list in all_merchant_transactions_collector.items():
            if len(amounts_list) < 2:
                continue  # std would be zero
            mean_val = np.mean(amounts_list)
            std_val = np.std(amounts_list, ddof=0)
            if std_val == 0:
                continue
            normalized_diffs = [(amt - mean_val) / std_val for amt in amounts_list]
            edge_attr_values.extend(normalized_diffs)

        if edge_attr_values:
            edge_attr_np = np.array(edge_attr_values).reshape(-1, 1)
            edge_scaler = StandardScaler().fit(edge_attr_np)
            edge_key = ('transaction', 'belongs_to', 'merchant')
            self.edge_scalers[edge_key] = edge_scaler
            print(f"  Fitted scaler for edge attribute {edge_key}. Input shape: {edge_attr_np.shape}")
        else:
            print("  [WARN] Not enough data to fit edge attribute scaler. Edge attributes will remain unscaled.")
            self.edge_scalers[('transaction', 'belongs_to', 'merchant')] = None

        # 4. Sequence Time Delta Scaler
        if all_sequence_time_deltas_collector:
            time_delta_array = np.array(all_sequence_time_deltas_collector).reshape(-1, 1)
            if time_delta_array.shape[0] > 0:
                time_delta_scaler = StandardScaler().fit(time_delta_array)
                self.seq_scalers['time_delta'] = time_delta_scaler
                print(f"  Fitted scaler for sequence 'time_delta'. Input shape: {time_delta_array.shape}")
            else:
                print("  [WARN] No valid time deltas collected for sequence scaler fitting.")
                self.seq_scalers['time_delta'] = None
        else:
            print("  [WARN] No time deltas collected for sequence scaler fitting.")
            self.seq_scalers['time_delta'] = None

        # self.scalers['merchant'] = None # Removed old placeholder
        # self.seq_scalers['time_delta'] = None # Removed old placeholder
        # print("  Merchant and sequence scaler fitting need full integration for iterative loading.") # Removed old message

    def _create_mappings_from_collected(self, category_id_strings_collector, user_company_names_collector):
        print("Creating ID mappings...")
        sorted_cat_strings = sorted(list(s for s in category_id_strings_collector if pd.notna(s) and s != 'UNKNOWN'))
        self.category_id_map = {code: name for code, name in enumerate(sorted_cat_strings)}
        if 'UNKNOWN' not in self.category_id_map.values(): # Ensure UNKNOWN is in map if present
            unknown_code = len(self.category_id_map)
            self.category_id_map[unknown_code] = 'UNKNOWN'
        self.num_global_classes = len(self.category_id_map)
        print(f"  Created category_id_map: {self.num_global_classes} classes (incl. UNKNOWN if present).")

        sorted_user_names = sorted(list(s for s in user_company_names_collector if pd.notna(s) and s != 'UNKNOWN'))
        self.user_map = {code: name for code, name in enumerate(sorted_user_names)}
        if 'UNKNOWN' not in self.user_map.values(): # Ensure UNKNOWN is in map
            unknown_user_code = len(self.user_map)
            self.user_map[unknown_user_code] = 'UNKNOWN'
        self.num_users = len(self.user_map)
        print(f"  Created user_map: {self.num_users} users (incl. UNKNOWN if present).")

    def _process_collected_coa_text(self):
        """Consolidates COA entries and tokenizes them for all users."""
        if not hasattr(self, 'all_user_coa_data_collector') or not self.all_user_coa_data_collector:
            print("[WARN] _process_collected_coa_text: No COA data collected. Skipping COA processing.")
            self.user_coa_texts = {}
            self.global_user_coa_tensors = None # Ensure it's defined
            return

        if self.tokenizer is None:
            self._init_tokenizer() # Ensure tokenizer is initialized

        print(f"[INFO] Processing collected COA text for {len(self.all_user_coa_data_collector)} users.")
        self.user_coa_texts = {}
        processed_coa_for_tokenization = []
        user_ids_for_tokenization = [] # Keep track of user_id_code for ordering

        # First, consolidate texts for each user
        for user_id_code, coa_entries in self.all_user_coa_data_collector.items():
            full_text_parts = []
            for entry in coa_entries:
                name = entry.get('name', '')
                description = entry.get('description', '')
                tax_type = entry.get('tax_type', '')
                # Only include non-empty parts
                parts = [p for p in [name, description, tax_type] if isinstance(p, str) and p.strip()]
                if parts:
                    full_text_parts.append("; ".join(parts))
            
            consolidated_text = " || ".join(full_text_parts) if full_text_parts else "" # Use || as separator like transaction text
            self.user_coa_texts[user_id_code] = consolidated_text
            # Add to list for batch tokenization, ensuring order matches user_map if possible
            # or at least a consistent order for the buffer
            processed_coa_for_tokenization.append(consolidated_text)
            user_ids_for_tokenization.append(user_id_code)

        if not processed_coa_for_tokenization:
            print("[INFO] No actual COA text to tokenize after consolidation.")
            self.global_user_coa_tensors = None
            return

        print(f"[INFO] Tokenizing COA texts for {len(processed_coa_for_tokenization)} users with max_length={self.coa_text_max_length}")
        # Tokenize all collected COA texts at once
        try:
            tokenized_coa = self.tokenizer(
                processed_coa_for_tokenization,
                padding='max_length', 
                truncation=True, 
                max_length=self.coa_text_max_length, 
                return_tensors='pt'
            )
            self.global_user_coa_tensors = {
                'input_ids': tokenized_coa['input_ids'],
                'attention_mask': tokenized_coa['attention_mask']
            }
            # To ensure the tensors in global_user_coa_tensors are indexed by user_id_code correctly later,
            # we need a mapping from the original user_id_code to its row index in these tensors.
            # The simplest way is to sort user_ids_for_tokenization and re-index the tensors accordingly
            # if the model strictly requires user_id_code to be a direct index.
            # However, the model will receive user_tokenized_coa_tensors and index into them using user_id_codes from the batch.
            # The current AdvancedTransactionCategorizationModel expects user_tokenized_coa_tensors to be a dict of tensors
            # where the first dimension corresponds to user_id_code if user_id_codes are contiguous and 0-indexed.
            # Let's assume user_map gives contiguous 0-indexed user_id_codes.
            # We need to ensure the order in global_user_coa_tensors matches the order of user_map.
            
            num_mapped_users = len(self.user_map)
            # Create placeholder tensors based on the full user map size
            final_coa_input_ids = torch.zeros((num_mapped_users, self.coa_text_max_length), dtype=torch.long)
            final_coa_attention_mask = torch.zeros((num_mapped_users, self.coa_text_max_length), dtype=torch.long)

            # Fill these tensors using the user_id_code from user_ids_for_tokenization as index
            for i, user_id_code in enumerate(user_ids_for_tokenization):
                if user_id_code < num_mapped_users: # Safety check
                    final_coa_input_ids[user_id_code] = tokenized_coa['input_ids'][i]
                    final_coa_attention_mask[user_id_code] = tokenized_coa['attention_mask'][i]
                else:
                    print(f"[WARN] User ID code {user_id_code} from COA data is out of bounds for user_map size {num_mapped_users}. Skipping.")

            self.global_user_coa_tensors = {
                'input_ids': final_coa_input_ids,
                'attention_mask': final_coa_attention_mask
            }

            print(f"[INFO] Global COA tensors created. Shapes: input_ids {self.global_user_coa_tensors['input_ids'].shape}, attention_mask {self.global_user_coa_tensors['attention_mask'].shape}")

        except Exception as e:
            print(f"[ERROR] Failed to tokenize global COA texts: {e}")
            import traceback
            traceback.print_exc()
            self.global_user_coa_tensors = None

    def _create_proxy_edge_scaler(self):
        """Creates a dummy edge scaler so that downstream code expecting an entry in self.edge_scalers will not fail.
        Currently, the only edge attribute is the normalized merchant amount diff which is already on a comparable
        scale.  We therefore create an identity StandardScaler (mean=0, scale=1) for that single-dimensional edge
        feature and store it under the canonical edge key.
        """
        from sklearn.preprocessing import StandardScaler
        edge_key = ('transaction', 'belongs_to', 'merchant')
        if edge_key in self.edge_scalers:
            return  # already exists
        dummy_scaler = StandardScaler()
        # Manually set fitted attributes for a 1-d feature so that transform can be called if needed.
        dummy_scaler.mean_ = np.array([0.0])
        dummy_scaler.scale_ = np.array([1.0])
        self.edge_scalers[edge_key] = dummy_scaler
        print(f"[INFO] Created proxy edge scaler for edge {edge_key}.")

    def setup(self, stage: Optional[str] = None):
        if not self.is_predicting: # Fit/Test stage
            if self.user_map is None or not self.scalers or self.num_global_classes == 0 : # Check if prepare_data needs to be run
                print(f"Prepare_data (Pass 1) not yet run or incomplete for stage '{stage}'. Running it now using train_files...")
                self.prepare_data() # This uses self.train_files
            else:
                print(f"Prepare_data (Pass 1) already run for stage '{stage}'. Using existing scalers/maps.")
                # Ensure num_users and num_global_classes are set if prepare_data was run previously
                if self.user_map and self.num_users == 0: self.num_users = len(self.user_map)
                if self.category_id_map and self.num_global_classes == 0: self.num_global_classes = len(self.category_id_map)

        else: # is_predicting is True
            print(f"Setup for stage '{stage}' (predict_mode=True). Using pre-fitted state.")
            if not self.fitted_scalers or not self.fitted_user_map or not self.fitted_category_id_map:
                raise ValueError("Predict stage: Pre-fitted scalers, user_map, and category_id_map are required.")
            # Ensure num_users, num_global_classes are set from fitted maps
            self.num_users = len(self.fitted_user_map) if self.fitted_user_map else 0
            self.num_global_classes = len(self.fitted_category_id_map) if self.fitted_category_id_map else 0
            
            if self.tokenizer is None and (self.use_text_encoder or self.use_coa_text_features):
                 self._init_tokenizer()
            # For COA text in prediction: user_coa_texts should have been loaded by `load_state`.
            # If not loaded and COA features are used, initialize to empty to avoid errors.
            if self.use_coa_text_features and not self.user_coa_texts and self.num_users > 0:
                 print("[WARN] Predict mode: use_coa_text_features=True, but self.user_coa_texts is empty. Initializing empty COA texts.")
                 self.user_coa_texts = {i: "" for i in range(self.num_users)}

        # Ensure COA texts are processed and global COA tensors are created if needed
        if self.use_coa_text_features and not hasattr(self, 'global_user_coa_tensors'):
             if not hasattr(self, 'user_coa_texts') or not self.user_coa_texts:
                print("[WARN] setup: COA texts not processed in prepare_data. global_user_coa_tensors might be missing.")
             # This implies _process_collected_coa_text should have been called in prepare_data
             # If it wasn't, or if it failed, global_user_coa_tensors might be None.
             # For safety, let's call it here if user_coa_texts exists from a previous run but tensors are missing.
             # However, _process_collected_coa_text relies on all_user_coa_data_collector from prepare_data.
             # So, this logic should primarily be in prepare_data.
             # Here, we just check.
             if self.global_user_coa_tensors is None:
                 print("[WARN] setup: global_user_coa_tensors is None. COA features in model might not work.")

        # --- Define Feature Dimensions and Metadata for the Model ---
        # These should be determined AFTER scalers/maps are fitted in prepare_data
        self.node_feature_dims = {}
        self.edge_feature_dims = {}
        self.sequence_feature_dim = 0 # Set to 0 as sequence encoder is removed

        # Transaction node features
        # Base: amount, num_coa, hour_sin, hour_cos, day_sin, day_cos (6 features)
        # If scaled, dim remains same. This is the dim of graph_chunk['transaction'].x
        # From _process_chunk_to_heterodata: raw_tx_features = np.column_stack([amount, num_coa, hour_sin, hour_cos, day_sin, day_cos])
        self.node_feature_dims['transaction'] = 6 # Update if more base features are added
        # If tx_feat_cols_to_scale changes, this might need adjustment if it affects final feature vector length

        # User node features (not explicitly created as separate nodes in HeteroData for GNN input, but model uses user_id_code)
        # The user embedding itself is handled by the model.
        # If COA text is used, its embedding is concatenated by the model.
        self.node_feature_dims['user'] = self._model_config.get('user_embed_dim', 64) # Placeholder, actual user node data for HGT might be different
        if self.use_coa_text_features:
            # The COA text feature is handled by the text_encoder within the model, not added to HGT's user node input here.
            # HGT user node input dim doesn't change due to COA, but the effective user feature dim in fusion does.
            pass 

        # Merchant node features (if GNN is used)
        if self.use_gnn_encoder:
            # Based on merchant_agg_funcs_named in _process_chunk_to_heterodata
            # Example: ['amount_mean', 'amount_std', 'amount_count', 'time_std', 'unique_users']
            # The number of features is len(self.merchant_agg_funcs_named)
            if self.merchant_agg_funcs_named:
                self.node_feature_dims['merchant'] = len(self.merchant_agg_funcs_named)
            else: # Default fallback if merchant_agg_funcs_named is somehow empty but GNN is on
                self.node_feature_dims['merchant'] = 8 # Matching default in _process_chunk_to_heterodata
        
        # Sequence features (REMOVED)
        # if self.use_sequence_encoder:
            # self.sequence_feature_dim = 6 # amount, day_sin, day_cos, hour_sin, hour_cos, time_delta_scaled

        # Edge features
        if self.use_gnn_encoder: 
            # ('transaction', 'belongs_to', 'merchant') has 1 edge attribute: normalized amount diff
            self.edge_feature_dims[('transaction', 'belongs_to', 'merchant')] = 1
            # Add other edge types if they exist and have features

        # --- Define Graph Metadata for HGT --- 
        node_types = ['transaction', 'user'] # 'user' is implicit via user_id_code for embedding lookup
        edge_types_list = []
        if self.use_gnn_encoder and 'merchant' in self.node_feature_dims:
            node_types.append('merchant')
            edge_types_list.append(('transaction', 'belongs_to', 'merchant'))
            # If users were explicit nodes linked to transactions for GNN:
            # node_types.append('user') 
            # edge_types_list.append(('user', 'performs', 'transaction'))
            # edge_types_list.append(('transaction', 'performed_by', 'user'))

        self.graph_metadata = {
            'node_types': node_types,
            'edge_types': edge_types_list
        }
        print(f"[INFO] DataModule setup complete for stage: {stage}. Node dims: {self.node_feature_dims}, Edge dims: {self.edge_feature_dims}, Graph Metadata: {self.graph_metadata}")
        if self.use_coa_text_features:
            if self.global_user_coa_tensors:
                print(f"Global COA Tensors available. input_ids shape: {self.global_user_coa_tensors['input_ids'].shape}")
            else:
                print("[WARN] Global COA Tensors are None after setup. COA features may not work.")

    def _get_dataloader(self, file_list: List[str], stage: str) -> Optional[DataLoader]:
        if not file_list:
            # For 'validate' stage, PyTorch Lightning handles a None dataloader by skipping validation.
            if stage == 'validate': 
                print(f"[INFO] No files provided for stage '{stage}'. Returning None for DataLoader (validation will be skipped).")
                return None
            # For other stages like 'fit' or 'test', missing files is usually an error.
            # However, predict might also have an empty list if no files are meant for prediction.
            # Let's allow predict to also return None if file_list is empty and handle it in predict_dataloader.
            if stage == 'predict':
                print(f"[INFO] No files provided for stage '{stage}'. Returning None for DataLoader.")
                return None
            raise ValueError(f"No files provided for DataLoader stage '{stage}', and it's not 'validate' or 'predict'. File list is empty.")

        dataset = ArrowFilesIterableDataset(
            file_paths=file_list, 
            data_module_ref=self, 
            chunk_size=self.iterable_dataset_chunk_size,
            stage=stage
        )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, # This batches the HeteroData objects yielded by dataset
            num_workers=self.num_workers,
            collate_fn=Batch.from_data_list, # PyG's collate function for HeteroData
            persistent_workers=(self.num_workers > 0) # Ensure this is only True if num_workers > 0
        )

    # --- Dataloader Methods using IterableDataset --- 
    def train_dataloader(self) -> DataLoader:
        print(f"Creating Iterable train DataLoader (chunk_size={self.iterable_dataset_chunk_size}, dl_batch_size={self.batch_size})...")
        if not self.train_files:
             # This case should ideally be handled by robust file splitting or raise error in __init__.
             # Raising error here if no train files, as training is not possible.
             raise ValueError("No training files specified for train_dataloader. Training cannot proceed.")
        return self._get_dataloader(self.train_files, 'fit')

    def val_dataloader(self) -> Optional[DataLoader]:
        # Reverted temporary debugging changes.
        # Original logic:
        if not self.val_files:
            # print("[DEBUG DATAMODULE] val_files is empty. Attempting to use train_files for validation for debugging purposes.")
            # if not self.train_files:
            #     print("[DEBUG DATAMODULE] train_files is also empty. No validation data available.")
            #     return None
            # print(f"[DEBUG DATAMODULE] Using first train file for validation: {self.train_files[:1]}")
            # return self._get_dataloader(self.train_files[:1], stage='validate')
            print("[INFO] No validation files specified. val_dataloader will be None, and validation will be skipped.")
            return None
        return self._get_dataloader(self.val_files, stage='validate')

    def test_dataloader(self) -> Optional[DataLoader]: # Can be None if no test files
        print(f"Creating Iterable test DataLoader (chunk_size={self.iterable_dataset_chunk_size}, dl_batch_size={self.batch_size})...")
        files_to_use = self.test_files # By default use test_files
        
        if self.is_predicting: # If in full prediction mode, test_dataloader might use predict_files if test_files is empty
            if not self.test_files and self.predict_files:
                print("[INFO] test_dataloader in prediction mode: No test_files, using predict_files instead.")
                files_to_use = self.predict_files
            elif not self.test_files and not self.predict_files:
                print("[WARN] test_dataloader: No test_files or predict_files available. Returning None.")
                return None
        elif not self.test_files:
             print("[WARN] test_dataloader: No test_files available. Returning None.")
             return None
             
        if not files_to_use: # Final check
            print("[WARN] test_dataloader: No files to use after logic. Returning None.")
            return None
        return self._get_dataloader(files_to_use, 'test')

    def predict_dataloader(self) -> Optional[DataLoader]: # Can be None if no predict files
        print(f"Creating Iterable predict DataLoader (chunk_size={self.iterable_dataset_chunk_size}, dl_batch_size={self.batch_size})...")
        if not self.predict_files:
            print("[WARN] No predict_files specified for predict_dataloader. Returning None.")
            return None
        return self._get_dataloader(self.predict_files, 'predict')
        
    def get_state(self) -> Dict[str, Any]:
        # Method to get serializable state for saving (scalers, maps, config)
        # Ensure scalers are serializable (they are by default with pickle)
        return {
            'fitted_scalers': self.scalers,
            'fitted_seq_scalers': self.seq_scalers,
            'fitted_edge_scalers': self.edge_scalers,
            'fitted_user_map': self.user_map,
            'fitted_category_id_map': self.category_id_map,
            'user_coa_texts': self.user_coa_texts, # Essential for COA features in predict
            'num_users': self.num_users,
            'num_global_classes': self.num_global_classes,
            # Store relevant config used by IterableDataset too
            'text_model_name': self.text_model_name,
            'text_max_length': self.text_max_length,
            'coa_text_max_length': self.coa_text_max_length,
            'max_seq_length': self.max_seq_length,
            # Modality flags are also important for consistent behavior in predict
            'use_sequence_encoder': self.use_sequence_encoder,
            'use_text_encoder': self.use_text_encoder,
            'use_gnn_encoder': self.use_gnn_encoder,
            'use_coa_text_features': self.use_coa_text_features,
            # Store expected feature dimensions if determined statically
            'node_feature_dims': self.node_feature_dims,
            'edge_feature_dims': self.edge_feature_dims,
            'sequence_feature_dim': self.sequence_feature_dim
        }

    @classmethod
    def load_state(cls, state: Dict[str, Any], all_file_paths_for_predict: List[str], batch_size: int, iterable_dataset_chunk_size: int, num_workers: int) -> 'TransactionDataModuleV2':
        # Create a new instance in prediction mode
        # Pass relevant config from state if needed by __init__
        dm = cls(
            file_paths=all_file_paths_for_predict, 
            batch_size=batch_size,
            iterable_dataset_chunk_size=iterable_dataset_chunk_size,
            num_workers=num_workers,
            val_ratio=0, test_ratio=0, 
            shuffle_files_before_split=False, 
            text_model_name=state.get('text_model_name', 'bert-base-uncased'),
            text_max_length=state.get('text_max_length', 128),
            coa_text_max_length=state.get('coa_text_max_length', 256),
            max_seq_length=state.get('max_seq_length', 50),
            use_sequence_encoder=state.get('use_sequence_encoder', True),
            use_text_encoder=state.get('use_text_encoder', True),
            use_gnn_encoder=state.get('use_gnn_encoder', True), 
            use_coa_text_features=state.get('use_coa_text_features', True),
            fitted_scalers=state.get('fitted_scalers'), # Use .get for safety
            fitted_seq_scalers=state.get('fitted_seq_scalers'),
            fitted_edge_scalers=state.get('fitted_edge_scalers'),
            fitted_user_map=state.get('fitted_user_map'),
            fitted_category_id_map=state.get('fitted_category_id_map')
        )
        dm.user_coa_texts = state.get('user_coa_texts', {})
        # Ensure num_users and num_global_classes are correctly set from loaded maps
        dm.num_users = len(dm.user_map) if dm.user_map else 0
        dm.num_global_classes = len(dm.category_id_map) if dm.category_id_map else 0
        
        dm.is_predicting = True 
        
        dm.node_feature_dims = state.get('node_feature_dims', {})
        dm.edge_feature_dims = state.get('edge_feature_dims', {})
        dm.sequence_feature_dim = state.get('sequence_feature_dim')

        dm.setup('predict') 
        return dm

# End of TransactionDataModuleV2


