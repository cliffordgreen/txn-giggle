import os
import argparse

# Set environment variables for CUDA debugging and memory optimization
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA for better error reporting
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Enable device-side assertions for debugging
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Reduce memory fragmentation
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism to avoid fork warnings
import pandas as pd
import pyarrow.dataset as ds # Added for reading arrow datasets
import json # Added for parsing JSON strings
import glob # Added for finding files
from tqdm import tqdm # Added for progress bar
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_forecasting.metrics import MAE # Use MAE for TFT init placeholder
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch._dynamo  # Import for compilation compatibility
import yaml # For loading potential YAML configs
from typing import Optional, Dict, List, Any, Tuple
import pyarrow as pa # Added for ArrowInvalid check
import pyarrow.ipc as ipc # Use ipc explicitly for stream reading
import numpy as np # Added
import pickle # Added for saving state

# Use the V2 DataModule
from data.data_module_v2 import TransactionDataModuleV2, SingleBatchIterable 
# Import the new advanced model
from models.advanced_transaction_classifier import AdvancedTransactionCategorizationModel

# Helper functions for two-pass loading
def collect_statistics_pass(data_dir: str, max_files_for_stats: Optional[int] = None, add_unknowns: bool = False) -> Dict[str, Any]:
    """Pass 1: Collect statistics needed for scaling without loading full data."""
    print(f"=== PASS 1: Collecting Statistics from {data_dir} ===")
    
    arrow_files_pattern = os.path.join(data_dir, '**/*.arrow')
    arrow_files = sorted(glob.glob(arrow_files_pattern, recursive=True))
    
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in directory: {data_dir}")
    
    # Limit files for statistics if specified
    if max_files_for_stats is not None:
        arrow_files = arrow_files[:max_files_for_stats]
        print(f"Limited to {len(arrow_files)} files for statistics collection")
    
    # Initialize statistics collectors
    stats = {
        'amount_stats': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')},
        'num_coa_stats': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
        'coa_feature_stats': {
            'coa_size': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'hierarchy_depth': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'type_diversity': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'code_complexity': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'naming_consistency': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'description_richness': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'balance_diversity': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'recent_accounts': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
        },
        'category_counts': {},
        'user_counts': {},
        'total_transactions_seen': 0
    }
    
    print(f"Processing {len(arrow_files)} files for statistics...")
    
    for file_path in tqdm(arrow_files, desc="Computing statistics"):
        try:
            with ipc.open_stream(file_path) as reader:
                table = reader.read_all()
                df_chunk = table.to_pandas()
            
            # Process each row for statistics
            for _, row in df_chunk.iterrows():
                try:
                    # Extract target and user info
                    if isinstance(row['target_transaction_processed'], dict):
                        txn_dict = row['target_transaction_processed']
                    else:
                        txn_dict = json.loads(row['target_transaction_processed'])
                    
                    amount = float(txn_dict.get('amount', 0.0))
                    category_str = str(row.get('txn_accepted_category_id_str', 'UNKNOWN'))
                    company_name = str(row.get('company_name', 'UNKNOWN'))
                    num_coa = pd.to_numeric(row.get('num_chart_of_accounts', 0), errors='coerce')
                    if pd.isna(num_coa):
                        num_coa = 0
                    
                    # Update amount statistics
                    stats['amount_stats']['sum'] += amount
                    stats['amount_stats']['sum_sq'] += amount**2
                    stats['amount_stats']['count'] += 1
                    stats['amount_stats']['min'] = min(stats['amount_stats']['min'], amount)
                    stats['amount_stats']['max'] = max(stats['amount_stats']['max'], amount)
                    
                    # Update num_coa statistics (will be log-transformed)
                    num_coa_log = np.log1p(float(num_coa))
                    stats['num_coa_stats']['sum'] += num_coa_log
                    stats['num_coa_stats']['sum_sq'] += num_coa_log**2
                    stats['num_coa_stats']['count'] += 1
                    
                    # Extract and update COA feature statistics
                    coa_features = extract_coa_features_single(row.get('chart_of_accounts_processed', '[]'))
                    for feat_name, feat_val in zip(stats['coa_feature_stats'].keys(), coa_features):
                        stats['coa_feature_stats'][feat_name]['sum'] += feat_val
                        stats['coa_feature_stats'][feat_name]['sum_sq'] += feat_val**2
                        stats['coa_feature_stats'][feat_name]['count'] += 1
                    
                    # Count categories and users
                    stats['category_counts'][category_str] = stats['category_counts'].get(category_str, 0) + 1
                    stats['user_counts'][company_name] = stats['user_counts'].get(company_name, 0) + 1
                    stats['total_transactions_seen'] += 1
                    
                except Exception as e:
                    # print(f"[DEBUG] Skipping row due to error: {e}") # Optional debug
                    continue  # Skip problematic rows
                    
        except Exception as e:
            print(f"[WARN] Failed to process file {os.path.basename(file_path)} for stats: {e}")
            continue
    
    print(f"Statistics collected from {stats['total_transactions_seen']:,} transactions")
    
    # Add unknown tokens if requested, before creating maps
    if add_unknowns:
        if '<UNKNOWN_CATEGORY>' not in stats['category_counts']:
            stats['category_counts']['<UNKNOWN_CATEGORY>'] = 1 # Add with a count of 1 to ensure it's in the map
            print("Added <UNKNOWN_CATEGORY> to category_counts.")
        if '<UNKNOWN_USER>' not in stats['user_counts']:
            stats['user_counts']['<UNKNOWN_USER>'] = 1 # Add with a count of 1
            print("Added <UNKNOWN_USER> to user_counts.")
        print("Completed adding unknown tokens if they were missing.")

    # Compute scalers from statistics
    stats['scalers'] = compute_scalers_from_stats(stats)
    # These maps are {name: code_int}
    stats['category_map'] = {cat: idx for idx, cat in enumerate(sorted(stats['category_counts'].keys()))} 
    stats['user_map'] = {user: idx for idx, user in enumerate(sorted(stats['user_counts'].keys()))}       
    
    print(f"Found {len(stats['category_map'])} unique categories, {len(stats['user_map'])} unique users (including unknowns if added).")
    return stats

def extract_coa_features_single(coa_json_str) -> List[float]:
    """Extract COA features for a single transaction (lightweight version)."""
    try:
        if pd.isna(coa_json_str):
            return [0.0] * 8
            
        coa_list = json.loads(coa_json_str) if isinstance(coa_json_str, str) else coa_json_str
        if not isinstance(coa_list, list) or len(coa_list) == 0:
            return [0.0] * 8
        
        # Simplified feature extraction for statistics
        account_codes = [acc.get('account_code', '') for acc in coa_list if isinstance(acc, dict)]
        account_names = [acc.get('account_name', '') for acc in coa_list if isinstance(acc, dict)]
        
        # Basic features
        coa_size = len(coa_list)
        hierarchy_depth = np.mean([len(str(code).rstrip('0')) if str(code).isdigit() and str(code).rstrip('0') else 1 
                                  for code in account_codes]) if account_codes else 0.0
        
        # Simplified versions of other features
        type_diversity = len(set(str(code)[0] if str(code) and str(code)[0].isdigit() else 'other' 
                               for code in account_codes)) / 6.0 if account_codes else 0.0
        code_complexity = np.mean([len(str(code)) / 10.0 for code in account_codes]) if account_codes else 0.0
        naming_consistency = 0.5  # Placeholder for statistics
        description_richness = 0.5  # Placeholder for statistics  
        balance_diversity = 0.5  # Placeholder for statistics
        recent_accounts = 0.1  # Placeholder for statistics
        
        return [coa_size, hierarchy_depth, type_diversity, code_complexity, 
                naming_consistency, description_richness, balance_diversity, recent_accounts]
        
    except Exception:
        return [0.0] * 8

def compute_scalers_from_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Compute StandardScaler parameters from collected statistics."""
    scalers = {}
    
    # Compute scaler for all scalable transaction features
    all_means = []
    all_stds = []
    
    # Amount
    if stats['amount_stats']['count'] > 0:
        amount_mean = stats['amount_stats']['sum'] / stats['amount_stats']['count']
        amount_var = (stats['amount_stats']['sum_sq'] / stats['amount_stats']['count']) - amount_mean**2
        amount_std = np.sqrt(max(amount_var, 1e-8))
        all_means.append(amount_mean)
        all_stds.append(amount_std)
    
    # Num COA (log-transformed)
    if stats['num_coa_stats']['count'] > 0:
        coa_mean = stats['num_coa_stats']['sum'] / stats['num_coa_stats']['count']
        coa_var = (stats['num_coa_stats']['sum_sq'] / stats['num_coa_stats']['count']) - coa_mean**2
        coa_std = np.sqrt(max(coa_var, 1e-8))
        all_means.append(coa_mean)
        all_stds.append(coa_std)
    
    # COA features
    for feat_name, feat_stats in stats['coa_feature_stats'].items():
        if feat_stats['count'] > 0:
            feat_mean = feat_stats['sum'] / feat_stats['count']
            feat_var = (feat_stats['sum_sq'] / feat_stats['count']) - feat_mean**2
            feat_std = np.sqrt(max(feat_var, 1e-8))
            all_means.append(feat_mean)
            all_stds.append(feat_std)
    
    # Create scaler parameters
    scalers['transaction'] = {
        'mean_': np.array(all_means),
        'scale_': np.array(all_stds)
    }
    
    print(f"Computed scalers: {len(all_means)} features")
    return scalers

def load_data_with_scalers(data_dir: str, stats: Dict[str, Any], max_transactions: int = 1000000) -> pd.DataFrame:
    """Pass 2: Load and process data using pre-computed scalers."""
    print(f"=== PASS 2: Loading {max_transactions:,} transactions with pre-computed scalers ===")
    
    arrow_files_pattern = os.path.join(data_dir, '**/*.arrow')
    arrow_files = sorted(glob.glob(arrow_files_pattern, recursive=True))
    
    processed_chunks = []
    total_loaded = 0
    
    for file_path in tqdm(arrow_files, desc="Loading data"):
        if total_loaded >= max_transactions:
            break
            
        try:
            # Load and process chunk
            with ipc.open_stream(file_path) as reader:
                table = reader.read_all()
                df_chunk = table.to_pandas()
            
            # Process chunk using existing logic but with pre-computed mappings
            # This df_processed will have user_id_code based on stats['user_map'] which might be incomplete
            df_processed = process_chunk_with_mappings(df_chunk, stats)
            
            # Limit chunk size if needed
            remaining = max_transactions - total_loaded
            if len(df_processed) > remaining:
                df_processed = df_processed.head(remaining)
            
            if len(df_processed) > 0:
                processed_chunks.append(df_processed)
                total_loaded += len(df_processed)
                
                if total_loaded % 50000 == 0:
                    print(f"Loaded {total_loaded:,} / {max_transactions:,} transactions")
                
        except Exception as e:
            print(f"[WARN] Failed to process file {os.path.basename(file_path)}: {e}")
            continue
    
    if not processed_chunks:
        raise ValueError("No valid data chunks were loaded")
    
    df = pd.concat(processed_chunks, ignore_index=True) # Renamed from final_df to df
    
    # Ensure clean index for graph building
    df.reset_index(drop=True, inplace=True)
    
    # Verify index is continuous
    if not df.index.equals(pd.RangeIndex(len(df))):
        print(f"[WARN] Index discontinuity in streaming, forcing reset...")
        df.index = pd.RangeIndex(len(df))
    
    print(f"Final dataset: {len(df):,} transactions")
    return df # Return df instead of final_df

def process_chunk_with_mappings(df_chunk: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """Process a data chunk using pre-computed category and user mappings."""
    extracted_data = []
    
    for _, row in df_chunk.iterrows():
        try:
            # Extract transaction data
            if isinstance(row['target_transaction_processed'], dict):
                txn_dict = row['target_transaction_processed']
            else:
                txn_dict = json.loads(row['target_transaction_processed'])
            
            # Map category and user using pre-computed mappings
            category_str = str(row.get('txn_accepted_category_id_str', 'UNKNOWN'))
            company_name = str(row.get('company_name', 'UNKNOWN'))
            
            category_id = stats['category_map'].get(category_str, -1)
            user_id_code = stats['user_map'].get(company_name, -1)
            
            extracted_data.append({
                'amount': float(txn_dict.get('amount', 0.0)),
                'timestamp': pd.to_datetime(txn_dict.get('created_date'), errors='coerce'),
                'description': str(txn_dict.get('description', '')),
                'memo': str(txn_dict.get('memo', '')),
                'merchant_name': str(txn_dict.get('payee', '')),
                'txn_accepted_category_id_str': category_str,
                'company_name': company_name,
                'category_id': category_id,
                'user_id_code': user_id_code,
                'industry_name': str(row.get('industry_name', 'UNKNOWN')),
                'num_chart_of_accounts': pd.to_numeric(row.get('num_chart_of_accounts', 0), errors='coerce'),
                'chart_of_accounts_processed': row.get('chart_of_accounts_processed', '[]')
            })
            
        except Exception as e:
            continue  # Skip problematic rows
    
    if not extracted_data:
        return pd.DataFrame()
    
    df_processed = pd.DataFrame(extracted_data)
    
    # Handle timestamps
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
    median_date = df_processed['timestamp'].dropna().median()
    if pd.isna(median_date):
        median_date = pd.Timestamp('2020-01-01')
    df_processed['timestamp'] = df_processed['timestamp'].fillna(median_date)
    
    # Add time features
    df_processed['weekday'] = df_processed['timestamp'].dt.weekday
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['num_chart_of_accounts'] = df_processed['num_chart_of_accounts'].fillna(0)
    
    return df_processed

def process_chunk_with_global_maps(df_raw_chunk: pd.DataFrame, global_stats: Dict[str, Any]) -> pd.DataFrame:
    """
    Process a raw data chunk using pre-computed global category and user mappings.
    Maps unknown entities to specific <UNKNOWN_...> IDs.
    """
    extracted_data = []
    
    # Get the pre-defined unknown IDs from the global maps
    # These maps are {name: code_int}
    unknown_user_id = global_stats['user_map'].get('<UNKNOWN_USER>')
    unknown_category_id = global_stats['category_map'].get('<UNKNOWN_CATEGORY>')

    if unknown_user_id is None:
        raise ValueError("'<UNKNOWN_USER>' not found in global_stats['user_map']. Ensure it was added during collect_statistics_pass.")
    if unknown_category_id is None:
        raise ValueError("'<UNKNOWN_CATEGORY>' not found in global_stats['category_map']. Ensure it was added during collect_statistics_pass.")

    for _, row in df_raw_chunk.iterrows():
        try:
            # Extract transaction data
            if isinstance(row['target_transaction_processed'], dict):
                txn_dict = row['target_transaction_processed']
            else:
                txn_dict = json.loads(row['target_transaction_processed'])
            
            raw_category_str = str(row.get('txn_accepted_category_id_str', '<UNKNOWN_CATEGORY>'))
            raw_company_name = str(row.get('company_name', '<UNKNOWN_USER>'))
            
            # Map category and user using global mappings, defaulting to <UNKNOWN_...> ID
            category_id = global_stats['category_map'].get(raw_category_str, unknown_category_id)
            user_id_code = global_stats['user_map'].get(raw_company_name, unknown_user_id)
            
            extracted_data.append({
                'amount': float(txn_dict.get('amount', 0.0)),
                'timestamp': pd.to_datetime(txn_dict.get('created_date'), errors='coerce'),
                'description': str(txn_dict.get('description', '')),
                'memo': str(txn_dict.get('memo', '')),
                'merchant_name': str(txn_dict.get('payee', '')),
                # Store the original category string and the mapped ID
                'txn_accepted_category_id_str': raw_category_str, 
                'category_id': category_id,
                # Store the original company name and the mapped ID
                'company_name': raw_company_name,
                'user_id_code': user_id_code,
                'industry_name': str(row.get('industry_name', 'UNKNOWN')),
                'num_chart_of_accounts': pd.to_numeric(row.get('num_chart_of_accounts', 0), errors='coerce'),
                'chart_of_accounts_processed': row.get('chart_of_accounts_processed', '[]')
            })
            
        except Exception as e:
            # print(f"[DEBUG] Skipping row in process_chunk_with_global_maps due to error: {e}") # Optional
            continue  # Skip problematic rows
    
    if not extracted_data:
        return pd.DataFrame()
    
    df_processed = pd.DataFrame(extracted_data)
    
    # Handle timestamps
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
    # Fill NaNs with a placeholder or strategy if necessary, e.g., overall median from global_stats
    # For simplicity here, using a fixed date if all in chunk are NaT.
    # A more robust approach would be to use global_stats['median_timestamp'] if calculated.
    median_date_chunk = df_processed['timestamp'].dropna().median()
    if pd.isna(median_date_chunk):
        median_date_chunk = pd.Timestamp('2020-01-01') 
    df_processed['timestamp'] = df_processed['timestamp'].fillna(median_date_chunk)
    
    # Add time features
    df_processed['weekday'] = df_processed['timestamp'].dt.weekday
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['num_chart_of_accounts'] = df_processed['num_chart_of_accounts'].fillna(0)
    
    return df_processed

def load_data_chunk_iteratively(
    all_arrow_files: List[str],
    current_file_idx: int,
    current_row_offset_in_file: int,
    max_transactions_per_chunk: int,
    global_stats: Dict[str, Any]
) -> Tuple[Optional[pd.DataFrame], int, int, bool]:
    """
    Loads and processes one chunk of data iteratively from a list of Arrow files.

    Args:
        all_arrow_files: Sorted list of all Arrow file paths.
        current_file_idx: Index of the Arrow file to start reading from.
        current_row_offset_in_file: Row offset within the starting Arrow file.
        max_transactions_per_chunk: The desired number of transactions for this chunk.
        global_stats: Statistics dictionary containing global scalers and maps.

    Returns:
        A tuple containing:
        - DataFrame for the current chunk (or None if no more data).
        - Next file index to resume from.
        - Next row offset within that file.
        - Boolean flag indicating if more data might be available.
    """
    processed_chunks_for_current_df = []
    loaded_in_current_df = 0
    
    original_start_file_idx = current_file_idx
    original_start_row_offset = current_row_offset_in_file

    for file_idx in range(current_file_idx, len(all_arrow_files)):
        file_path = all_arrow_files[file_idx]
        
        if loaded_in_current_df >= max_transactions_per_chunk:
            break 
            
        try:
            with ipc.open_stream(file_path) as reader:
                table = reader.read_all()
                df_raw_file_chunk = table.to_pandas()

            # Apply row offset if this is the first file being processed in this call
            if file_idx == original_start_file_idx and current_row_offset_in_file > 0:
                if current_row_offset_in_file >= len(df_raw_file_chunk):
                    # Offset is beyond this file, move to next file
                    current_row_offset_in_file = 0 # Reset for next file
                    continue 
                df_raw_file_chunk = df_raw_file_chunk.iloc[current_row_offset_in_file:]
            
            # Process this part of the file
            df_processed_part = process_chunk_with_global_maps(df_raw_file_chunk, global_stats)
            
            if df_processed_part.empty:
                if file_idx == original_start_file_idx: # Reset offset if we skipped the rest of the starting file
                     current_row_offset_in_file = 0
                continue

            # How many can we add to the current DF?
            can_add = max_transactions_per_chunk - loaded_in_current_df
            
            if len(df_processed_part) > can_add:
                df_to_add = df_processed_part.head(can_add)
                rows_taken_from_processed_part = can_add
                # Estimate rows taken from raw chunk to update offset (approximate if processing filters rows)
                # This approximation assumes process_chunk_with_global_maps doesn't drastically change row count.
                # A more precise way would be to track original indices if vital.
                raw_rows_estimate_taken = rows_taken_from_processed_part 
                current_row_offset_in_file += raw_rows_estimate_taken 
            else:
                df_to_add = df_processed_part
                rows_taken_from_processed_part = len(df_to_add)
                current_row_offset_in_file = 0 # Moved to next file or finished this one
            
            processed_chunks_for_current_df.append(df_to_add)
            loaded_in_current_df += len(df_to_add)
            
            # If we took all rows from df_raw_file_chunk (after offset), reset offset for next file
            if rows_taken_from_processed_part >= len(df_raw_file_chunk): # or if df_to_add was the whole df_processed_part
                 current_row_offset_in_file = 0


            if loaded_in_current_df >= max_transactions_per_chunk:
                # Update current_file_idx for the next call
                # If current_row_offset_in_file is non-zero, it means we stopped mid-file
                # otherwise, we finished this file and should start the next one.
                if current_row_offset_in_file == 0:
                    current_file_idx = file_idx + 1
                else:
                    current_file_idx = file_idx 
                break # Filled the chunk

        except Exception as e:
            print(f"[WARN] Failed to process file {os.path.basename(file_path)} during iterative loading: {e}")
            current_row_offset_in_file = 0 # Skip to next file on error
            continue # Move to the next file
    
    if not processed_chunks_for_current_df:
        return None, current_file_idx, current_row_offset_in_file, False # No more data

    final_df_chunk = pd.concat(processed_chunks_for_current_df, ignore_index=True)
    final_df_chunk.reset_index(drop=True, inplace=True)
    
    # Determine if more data might be available
    more_data_available = (current_file_idx < len(all_arrow_files)) or \
                          (current_file_idx == len(all_arrow_files) -1 and current_row_offset_in_file > 0 and current_row_offset_in_file < len(df_raw_file_chunk))


    print(f"Loaded chunk of {len(final_df_chunk)} transactions. Next file index: {current_file_idx}, Next offset: {current_row_offset_in_file}")
    return final_df_chunk, current_file_idx, current_row_offset_in_file, more_data_available

def train_advanced_streaming(
    # Data/Output  
    data_dir: str,
    output_dir: str,
    # Memory Management
    max_transactions: int = 1000000, # Max records per chunk in memory
    max_files_for_stats: Optional[int] = None, 
    # Training params
    batch_size: int = 32,
    num_workers: int = 0,
    # max_epochs CLI arg is not directly used by iterative trainer for epochs_per_chunk
    # Instead, we now have num_overall_epochs and epochs_per_chunk
    num_overall_epochs: int = 1, # New: Number of full passes over the entire dataset via chunks
    epochs_per_chunk: int = 1, 
    total_chunks_to_process: Optional[int] = None, # Limit total chunks processed *within one overall epoch*
    seed: int = 42,
    model_config: dict = {}, 
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    mtl_weights: dict = {'global': 1.0, 'user': 0.0},
    focal_loss_alpha: float = 0.25,
    focal_loss_gamma: float = 2.0,
    accelerator: str = 'auto',
    precision: str = 'bf16-mixed',
    hgt_num_samples: Optional[Dict[str, List[int]]] = None,
    early_stopping_patience_per_chunk: int = 3 
):
    """Train AdvancedTransactionCategorizationModel iteratively on chunks of a large dataset,
       for a specified number of overall epochs through the entire dataset."""
    torch.set_float32_matmul_precision('high') 
    pl.seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    # === PASS 1: Collect GLOBAL Statistics (ONCE at the very beginning) ===
    print("="*50)
    print("Starting GLOBAL statistics collection pass (will scan all relevant files)...")
    print("="*50)
    if max_files_for_stats is not None:
        print(f"[INFO] max_files_for_stats is set to {max_files_for_stats}. For full dataset stats, this should ideally be None or cover all files.")
    
    global_stats = collect_statistics_pass(data_dir, max_files_for_stats=max_files_for_stats, add_unknowns=True) 
    
    global_user_map_name_to_code = global_stats['user_map']
    global_category_map_name_to_code = global_stats['category_map']
    global_num_users = len(global_user_map_name_to_code)
    global_num_global_classes = len(global_category_map_name_to_code)
    global_num_user_classes = 0 

    print(f"Global Stats: #Users={global_num_users} (incl. unknown), #GlobalClasses={global_num_global_classes} (incl. unknown)")
    print("="*50)

    # === GNN Configuration Dry Run (ONCE at the very beginning) ===
    print("Starting GNN configuration dry run...")
    all_arrow_files_for_dry_run = sorted(glob.glob(os.path.join(data_dir, '**/*.arrow'), recursive=True))
    if not all_arrow_files_for_dry_run: # Check needed here as well
        raise FileNotFoundError(f"No .arrow files found in {data_dir} for GNN dry run.")

    dry_run_chunk_df, _, _, _ = load_data_chunk_iteratively(
        all_arrow_files=all_arrow_files_for_dry_run, # Use the full list
        current_file_idx=0,
        current_row_offset_in_file=0,
        max_transactions_per_chunk=min(10000, max_transactions // 10 if max_transactions > 1000 else 1000),
        global_stats=global_stats
    )
    if dry_run_chunk_df is None or dry_run_chunk_df.empty:
        raise ValueError("Dry run failed: Could not load sample data for metadata determination")
    print(f"Dry run loaded {len(dry_run_chunk_df)} transactions for metadata extraction")
    
    temp_global_category_map_code_to_name = {v: k for k, v in global_category_map_name_to_code.items()}
    temp_data_module = TransactionDataModuleV2(
        transactions_df_ref=dry_run_chunk_df, batch_size=min(32, len(dry_run_chunk_df)), num_workers=0,
        text_model_name=model_config.get('text_encoder_params', {}).get('model_name', 'ProsusAI/finbert'),
        max_seq_length=model_config.get('max_seq_length', 50),
        text_max_length=model_config.get('text_encoder_params', {}).get('max_length', 128),
        use_sequence_encoder=model_config.get('use_sequence_encoder', False),
        use_gnn_encoder=model_config.get('use_gnn_encoder', True),   
        use_text_encoder=model_config.get('use_text_encoder', True),     
        use_coa_text_features=model_config.get('use_coa_text_features', False),
        num_hgt_layers = model_config.get('graph_encoder_params', {}).get('num_layers', 2),
        hgt_num_samples = hgt_num_samples,
        fitted_scalers=global_stats['scalers'],
        fitted_category_id_map=temp_global_category_map_code_to_name,
        fitted_user_map=global_user_map_name_to_code 
    )
    print("Setting up temporary DataModule for GNN config...")
    temp_data_module.setup('fit')
    initial_graph_metadata = temp_data_module.full_graph_data.metadata()
    initial_node_feature_dims = temp_data_module.node_feature_dims
    del temp_data_module, dry_run_chunk_df, temp_global_category_map_code_to_name
    print("GNN configuration dry run complete.")
    print(f"  Initial Graph Metadata: {initial_graph_metadata}")
    print(f"  Initial Node Feature Dims: {initial_node_feature_dims}")
    print("="*50)

    # === Initialize Main Model (ONCE) ===
    main_model_config = model_config.copy()
    main_model_config['num_users'] = global_num_users
    main_model_config['num_global_classes'] = global_num_global_classes
    main_model_config['num_user_classes'] = global_num_user_classes
    if 'graph_encoder_params' not in main_model_config: main_model_config['graph_encoder_params'] = {}
    main_model_config['graph_encoder_params']['metadata'] = initial_graph_metadata
    main_model_config['graph_encoder_params']['in_channels'] = initial_node_feature_dims
    
    # Ensure fusion parameter keys exist before assignment
    if 'fusion_params' not in main_model_config: main_model_config['fusion_params'] = {}
    if main_model_config.get('use_gnn_encoder', True) and 'graph_encoder_params' in main_model_config:
        main_model_config['fusion_params']['graph_dim'] = main_model_config['graph_encoder_params'].get('out_channels', 128)
    if main_model_config.get('use_sequence_encoder', False) and 'sequence_encoder_params' in main_model_config:
        main_model_config['fusion_params']['seq_dim'] = main_model_config.get('sequence_encoder_params',{}).get('output_dim', 128)
    if main_model_config.get('use_text_encoder', True) and 'text_encoder_params' in main_model_config:
        text_proj_dim = main_model_config.get('text_encoder_params',{}).get('projection_dim', 0)
        finbert_hidden_size = 768 
        text_out_dim_for_fusion = text_proj_dim if text_proj_dim > 0 else finbert_hidden_size
        main_model_config['fusion_params']['text_dim'] = text_out_dim_for_fusion
    if 'user_embed_dim' in main_model_config: # Ensure user_embed_dim is in config
        main_model_config['fusion_params']['user_dim'] = main_model_config['user_embed_dim']
    else: # Add a default if not present, or raise error
        main_model_config['user_embed_dim'] = 64 # Example default
        main_model_config['fusion_params']['user_dim'] = 64
        print("[WARN] 'user_embed_dim' not found in model_config, defaulted to 64.")


    print("Initializing AdvancedTransactionCategorizationModel globally...")
    model = AdvancedTransactionCategorizationModel(
        model_config=main_model_config, learning_rate=learning_rate, weight_decay=weight_decay,
        mtl_weights=mtl_weights, focal_loss_alpha=focal_loss_alpha, focal_loss_gamma=focal_loss_gamma
    )
    print("Global model initialized.")
    print("="*50)

    # --- Script Resumption Logic ---
    # TODO: Implement more robust script resumption. For now, this tracks the last overall checkpoint.
    # This should ideally be loaded *into* the model object if resuming.
    latest_overall_model_checkpoint_to_resume_script = None 

    overall_model_checkpoints_dir = os.path.join(output_dir, "overall_model_checkpoints")
    os.makedirs(overall_model_checkpoints_dir, exist_ok=True)
    
    all_arrow_files_full_dataset = sorted(glob.glob(os.path.join(data_dir, '**/*.arrow'), recursive=True))
    if not all_arrow_files_full_dataset:
        raise FileNotFoundError(f"No .arrow files found in {data_dir} for iterative training.")

    # === Outer Loop for Overall Epochs ===
    for overall_epoch_num in range(1, num_overall_epochs + 1):
        print(f"\n{'='*25} Starting Overall Epoch {overall_epoch_num}/{num_overall_epochs} {'='*25}")
        
        # Reset chunk iteration state for each overall epoch
        current_file_idx = 0
        current_row_offset_in_file = 0 
        more_data_to_load = True
        chunk_iteration_in_epoch = 0 # Renamed to avoid confusion with a global chunk_number

        # Inner Loop for processing chunks within the current overall epoch
        while more_data_to_load:
            chunk_iteration_in_epoch += 1
            current_global_chunk_num = ((overall_epoch_num - 1) * (len(all_arrow_files_full_dataset) * 10000 // max_transactions +1 )) + chunk_iteration_in_epoch # Approx for logging

            if total_chunks_to_process is not None and chunk_iteration_in_epoch > total_chunks_to_process:
                print(f"Reached total_chunks_to_process limit ({total_chunks_to_process}) for overall_epoch {overall_epoch_num}.")
                more_data_to_load = False # Stop processing chunks for this overall epoch
                break 
            
            print(f"\n--- Overall Epoch {overall_epoch_num}, Processing Chunk {chunk_iteration_in_epoch} ---")

            df_chunk, next_file_idx, next_row_offset, more_data_available = load_data_chunk_iteratively(
                all_arrow_files=all_arrow_files_full_dataset,
                current_file_idx=current_file_idx,
                current_row_offset_in_file=current_row_offset_in_file,
                max_transactions_per_chunk=max_transactions,
                global_stats=global_stats
            )

            if df_chunk is None or df_chunk.empty:
                print(f"No more data loaded for chunk {chunk_iteration_in_epoch} in overall epoch {overall_epoch_num}. Ending this overall epoch.")
                break # End of data for this overall epoch

            print(f"Chunk {chunk_iteration_in_epoch} (Overall Epoch {overall_epoch_num}) loaded with {len(df_chunk)} transactions.")
            current_file_idx = next_file_idx
            current_row_offset_in_file = next_row_offset
            # more_data_to_load is primarily for this inner loop; outer loop controls overall epochs.
            # If load_data_chunk_iteratively says no more data, this inner loop for the current overall epoch ends.
            if not more_data_available:
                 more_data_to_load = False


            # --- DataModule for the current chunk ---
            chunk_global_category_map_code_to_name = {v: k for k, v in global_category_map_name_to_code.items()} # {code:name}
            
            chunk_data_module = TransactionDataModuleV2(
                transactions_df_ref=df_chunk, batch_size=batch_size, num_workers=num_workers,
                text_model_name=main_model_config.get('text_encoder_params',{}).get('model_name', 'ProsusAI/finbert'),
                max_seq_length=main_model_config.get('max_seq_length', 50),
                text_max_length=main_model_config.get('text_encoder_params',{}).get('max_length', 128),
                use_sequence_encoder=main_model_config.get('use_sequence_encoder', False),
                use_gnn_encoder=main_model_config.get('use_gnn_encoder', True),   
                use_text_encoder=main_model_config.get('use_text_encoder', True),     
                use_coa_text_features=main_model_config.get('use_coa_text_features', False),
                num_hgt_layers = main_model_config.get('graph_encoder_params',{}).get('num_layers', 2),
                hgt_num_samples = hgt_num_samples, 
                fitted_scalers=global_stats['scalers'],
                fitted_category_id_map=chunk_global_category_map_code_to_name, 
                fitted_user_map=global_user_map_name_to_code 
            )
            print(f"Setting up DataModule for chunk {chunk_iteration_in_epoch} (Overall Epoch {overall_epoch_num})...")
            chunk_data_module.setup('fit') 
            
            # --- Trainer for the current chunk ---
            per_chunk_artifacts_dir = os.path.join(output_dir, "per_chunk_artifacts", f"overall_epoch_{overall_epoch_num}", f"chunk_{chunk_iteration_in_epoch}")
            os.makedirs(per_chunk_artifacts_dir, exist_ok=True)

            per_chunk_checkpoint_callback = ModelCheckpoint(
                dirpath=per_chunk_artifacts_dir, 
                filename=f'model-oe{overall_epoch_num}-c{chunk_iteration_in_epoch}-best-{{epoch:02d}}-{{val_loss:.2f}}',
                save_top_k=1, monitor='val_loss', mode='min'
            )
            
            trainer_early_stop_patience = main_model_config.get('trainer_params',{}).get('early_stopping_patience_per_chunk', early_stopping_patience_per_chunk)
            chunk_early_stop_callback = EarlyStopping(
                monitor='val_loss', patience=trainer_early_stop_patience, mode='min', verbose=True
            )
            
            chunk_logger = TensorBoardLogger(
                save_dir=os.path.join(output_dir, "tensorboard_logs_per_chunk"), 
                name=f"oe{overall_epoch_num}_chunk_{chunk_iteration_in_epoch}"
            )
            
            num_batches_in_chunk = (len(df_chunk) // batch_size) + (1 if len(df_chunk) % batch_size != 0 else 0)
            val_check_interval_steps = max(1, num_batches_in_chunk // 4) if epochs_per_chunk > 0 else num_batches_in_chunk 
            log_steps = main_model_config.get('trainer_params',{}).get('log_every_n_steps', max(1, num_batches_in_chunk // 20))

            trainer = pl.Trainer(
                max_epochs=epochs_per_chunk, 
                accelerator=accelerator, precision=precision, devices=1, 
                callbacks=[per_chunk_checkpoint_callback, chunk_early_stop_callback],
                logger=chunk_logger, log_every_n_steps=log_steps,
                val_check_interval=val_check_interval_steps,
                accumulate_grad_batches=main_model_config.get('trainer_params',{}).get('accumulate_grad_batches', 4),
            )

            print(f"Fitting model on chunk {chunk_iteration_in_epoch} (Overall Epoch {overall_epoch_num}) for {epochs_per_chunk} epoch(s)...")
            
            # For the very first chunk of the very first overall epoch, potentially resume script state.
            # Otherwise, train the 'model' instance which holds weights from previous chunks/overall epochs.
            ckpt_path_for_this_fit = None
            if overall_epoch_num == 1 and chunk_iteration_in_epoch == 1 and \
               latest_overall_model_checkpoint_to_resume_script and \
               os.path.exists(latest_overall_model_checkpoint_to_resume_script):
                print(f"Resuming trainer state for first chunk of first overall epoch from: {latest_overall_model_checkpoint_to_resume_script}")
                ckpt_path_for_this_fit = latest_overall_model_checkpoint_to_resume_script
            
            trainer.fit(model, datamodule=chunk_data_module, ckpt_path=ckpt_path_for_this_fit) 
            
            best_model_this_chunk_fit = per_chunk_checkpoint_callback.best_model_path 
            if best_model_this_chunk_fit and os.path.exists(best_model_this_chunk_fit):
                print(f"Best model during chunk {chunk_iteration_in_epoch} (OE {overall_epoch_num}) training passes saved to: {best_model_this_chunk_fit}")
            else:
                print(f"[INFO] No new 'best' checkpoint saved by ModelCheckpoint for this chunk's fit.")

            # Save the overall model state after training on this chunk.
            current_overall_ckpt_filename = f"overall_model_oe{overall_epoch_num}_after_chunk_{chunk_iteration_in_epoch}.ckpt"
            current_overall_ckpt_path = os.path.join(overall_model_checkpoints_dir, current_overall_ckpt_filename)
            trainer.save_checkpoint(current_overall_ckpt_path)
            latest_overall_model_checkpoint_to_resume_script = current_overall_ckpt_path 
            print(f"Overall model state checkpoint saved to: {latest_overall_model_checkpoint_to_resume_script}")
            
            del df_chunk, chunk_data_module, trainer, per_chunk_checkpoint_callback, chunk_early_stop_callback, chunk_logger
            if torch.cuda.is_available() and accelerator in ['gpu', 'cuda']:
                torch.cuda.empty_cache()
        # End of inner while loop (chunks for current overall epoch)
        print(f"--- Completed Overall Epoch {overall_epoch_num}/{num_overall_epochs} ---")
    # End of outer for loop (overall epochs)

    print("="*50)
    print("Finished processing all overall epochs and their chunks.")
    print("="*50)

    final_model_save_path = os.path.join(output_dir, "final_trained_model_after_all_epochs.ckpt")
    if latest_overall_model_checkpoint_to_resume_script and os.path.exists(latest_overall_model_checkpoint_to_resume_script):
        # Copy the very last "overall" checkpoint to a fixed final name
        import shutil
        shutil.copy(latest_overall_model_checkpoint_to_resume_script, final_model_save_path)
        print(f"Final trained model (copied from last overall checkpoint) saved to: {final_model_save_path}")
    elif model: # If model exists but no checkpoints were made (e.g. dry run, 1 chunk, no save_top_k match)
        # This trainer instance is from the last chunk, might not be ideal but better than nothing.
        # A better approach would be to reinstantiate a trainer just for saving if needed.
        # For simplicity, we'll assume latest_overall_model_checkpoint_to_resume_script is preferred.
        print(f"[WARN] No overall checkpoint path found to copy as final model. Attempting to save current model state if possible.")
        # Need a trainer to save, the last one was deleted. Re-create a simple one.
        simple_trainer_for_save = pl.Trainer(accelerator='cpu', devices=1) # Simple trainer just to save
        simple_trainer_for_save.save_checkpoint(final_model_save_path, weights_only=True) # Or full
        print(f"Saved current model state to {final_model_save_path}. This might not be from a val checkpoint.")


    print("Saving final (global) preprocessing state...")
    state_to_save = {
        'scalers': global_stats['scalers'],
        'global_category_map_name_to_code': global_category_map_name_to_code, 
        'global_user_map_name_to_code': global_user_map_name_to_code,
        'model_config_used_for_init': main_model_config 
    }
    global_state_save_path = os.path.join(output_dir, 'global_preprocessing_and_model_config_state.pkl')
    try:
        with open(global_state_save_path, 'wb') as f:
            pickle.dump(state_to_save, f)
        print(f"Global preprocessing and model config state saved to: {global_state_save_path}")
    except Exception as save_e:
        print(f"[ERROR] Failed to save global state: {save_e}")

    print("Iterative streaming training script finished.")

def train_advanced(
    # Data/Output
    data_dir: str, # Changed from data_path
    output_dir: str,
    # Basic Training Params
    batch_size: int = 32,
    num_workers: int = 0,
    max_epochs: int = 50,
    seed: int = 42,
    # Model Hyperparameters (passed via dicts)
    model_config: dict = {},
    # Training Strategy Params
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    mtl_weights: dict = {'global': 1.0, 'user': 0.0},
    focal_loss_alpha: float = 0.25,
    focal_loss_gamma: float = 2.0,
    # Add trainer specific args if needed (accelerator, precision etc.)
    accelerator: str = 'auto',
    precision: str = 'bf16-mixed',
    # <<< HGTLoader specific config >>>
    hgt_num_samples: Optional[Dict[str, List[int]]] = None,
    # num_hgt_layers is derived from model_config now
    # Add max_files_to_process argument
    max_files_to_process: Optional[int] = None 
):
    """Train the advanced transaction classifier."""
    # Set precision for Tensor Cores before seeding
    torch.set_float32_matmul_precision('high') 
    pl.seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data --- 
    # Pass the new argument to load_data
    df = load_data(data_dir, max_files_to_process=max_files_to_process)

    # --- Calculate necessary dims from data --- 
    num_global_classes = df['txn_accepted_category_id_str'].nunique()
    num_user_classes = 0 # Adjust if user-specific task is defined later
    # Need num_users for embedding layer, use 'company_name'
    num_users = df['company_name'].nunique() # Estimate here, refined in DataModule
    print(f"Derived from data: #Global={num_global_classes}, #User={num_user_classes}, #Users={num_users}")
    
    # --- Data Module V2 --- 
    print("Initializing TransactionDataModuleV2...")
    # --- Get encoder flags from config --- 
    use_seq = model_config.get('use_sequence_encoder', True) # Default to True if missing
    use_graph = model_config.get('use_graph_encoder', True)   # Default to True
    use_text = model_config.get('use_text_encoder', True)     # Default to True
    # Get the new flag for COA text features
    use_coa_text = model_config.get('use_coa_text_features', True) # Default to True

    num_hgt_layers = model_config['graph_encoder_params'].get('num_layers', 2) 
    data_module = TransactionDataModuleV2(
        transactions_df_ref=df,
        batch_size=batch_size,
        num_workers=num_workers,
        num_hgt_layers=num_hgt_layers,
        hgt_num_samples=hgt_num_samples,
        text_model_name=model_config['text_encoder_params'].get('model_name', 'ProsusAI/finbert'),
        max_seq_length=model_config.get('max_seq_length', 50),
        text_max_length=model_config['text_encoder_params'].get('max_length', 128),
        use_sequence_encoder=use_seq,
        use_gnn_encoder=use_graph,
        use_text_encoder=use_text,
        use_coa_text_features=use_coa_text # Pass the new flag
        # Add other arguments like fitted_scalers if this is also used for predict.py context
        # For now, assuming this specific snippet is from train_new.py for training context
    )
    print("Setting up DataModuleV2...")
    data_module.setup('fit') 
    
    # --- Update Model Config with dynamic values AFTER setup --- 
    # Get class counts from DataModule instance
    num_global_classes = data_module.num_global_classes
    num_user_classes = data_module.num_user_classes
    num_users = data_module.num_users
    print(f"DataModule counts: #Global={num_global_classes}, #User={num_user_classes}, #Users={num_users}")

    # Required by HGT
    model_config['graph_encoder_params']['in_channels'] = data_module.node_feature_dims
    model_config['graph_encoder_params']['metadata'] = data_module.full_graph_data.metadata() 
    # Required by User Embedding
    model_config['num_users'] = num_users
    # Required by Classifiers
    model_config['num_global_classes'] = num_global_classes
    model_config['num_user_classes'] = num_user_classes
    # Update fusion dims based on actual encoder output dims
    model_config['fusion_params']['graph_dim'] = model_config['graph_encoder_params'].get('out_channels', 128)
    model_config['fusion_params']['seq_dim'] = model_config['sequence_encoder_params'].get('output_dim', 128)
    text_proj_dim = model_config['text_encoder_params'].get('projection_dim', 0)
    finbert_hidden_size = 768 # Placeholder - ideally get from loaded model config
    text_out_dim_for_fusion = text_proj_dim if text_proj_dim > 0 else finbert_hidden_size
    model_config['fusion_params']['text_dim'] = text_out_dim_for_fusion
    model_config['fusion_params']['user_dim'] = model_config['user_embed_dim']
    
    # --- Update TFT Categorical Config with dynamic cardinality ---
    # Only run if sequence encoder is enabled
    if use_seq:
        if (tft_params := model_config.get('sequence_encoder_params', {}).get('tft_params')) and \
           hasattr(data_module, 'num_regions') and 'region_id' in tft_params.get('categorical_groups', {}):
            print(f"[INFO] Updating TFT categorical_groups for 'region_id' with cardinality: {data_module.num_regions}")
            tft_params['categorical_groups']['region_id'] = data_module.num_regions
        elif tft_params:
            print("[WARN] Could not dynamically update TFT categorical_groups for 'region_id'. Check config and DataModule.")

    # --- Replace TFT loss placeholder from YAML with actual Metric instance --- 
    # Only run if sequence encoder is enabled
    if use_seq:
        if (tft_params := model_config.get('sequence_encoder_params', {}).get('tft_params')) and \
           tft_params.get('loss') == "MAE_placeholder":
            print("[INFO] Replacing TFT loss placeholder with MAE() instance.")
            tft_params['loss'] = MAE()
        elif tft_params:
            print(f"[WARN] TFT loss in config is not the placeholder string: {tft_params.get('loss')}")
    
    # --- Create Model --- 
    print("Initializing AdvancedTransactionCategorizationModel...")
    model = AdvancedTransactionCategorizationModel(
        model_config=model_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mtl_weights=mtl_weights,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        transactions_df_ref=df
    )
    
    # Disable torch.compile due to CUDA compatibility issues with PyTorch Geometric
    print("[INFO] Skipping torch.compile due to CUDA compatibility issues with PyTorch Geometric")
    # No longer needed if we fetch labels within _step from df/graph ref passed here
    
    # --- Callbacks & Logger --- 
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='adv_model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1, # Save only the best
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10, # Adjust as needed
            mode='min'
        )
    ]
    logger = TensorBoardLogger(save_dir=output_dir, name='adv_logs')

    # --- Trainer --- 
    print("Initializing Trainer...")
    # Calculate validation interval as a factor of total training batches
    total_samples = len(df)
    batches_per_epoch = (total_samples // batch_size) + (1 if total_samples % batch_size != 0 else 0)
    # Check validation every 10% of an epoch, minimum 50 steps, maximum 1000 steps
    val_check_interval = max(50, min(1000, batches_per_epoch // 10))
    print(f"[INFO] Calculated val_check_interval: {val_check_interval} (batches_per_epoch: {batches_per_epoch})")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision=precision,
        devices=1, # Assuming single device for now
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=min(val_check_interval, 500),  # Log more frequently than validation
        val_check_interval=val_check_interval,
        accumulate_grad_batches=4,  # Effective batch size = batch_size * 4
        # gradient_clip_val=1 # Optional
    )

    # --- Train --- 
    print("Starting Training...")
    try:
        trainer.fit(model, datamodule=data_module) # Use datamodule argument
    except Exception as e:
        print(f"!!! ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        # Optionally save state or raise
        raise e
    finally:
        # --- Save Preprocessing State (Scalers and Mappings) ---
        print("Attempting to save preprocessing state...")
        state_to_save = {
            'scalers': data_module.scalers,
            'seq_scalers': data_module.seq_scalers,
            'edge_scalers': data_module.edge_scalers,
            'user_map': data_module.user_map,
            'category_id_map': data_module.category_id_map
        }
        save_path = os.path.join(output_dir, 'preprocessing_state.pkl')
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(state_to_save, f)
            print(f"Preprocessing state saved successfully to: {save_path}")
        except Exception as save_e:
            print(f"[ERROR] Failed to save preprocessing state to {save_path}: {save_e}")

    # --- Test --- 
    print("Starting Testing...")
    try:
        # Load best checkpoint automatically by Trainer
        test_results = trainer.test(datamodule=data_module, ckpt_path='best') 
        print("Test Results:", test_results)
        # Save test results to file
        if test_results:
            results_df = pd.DataFrame(test_results)
            results_df.to_csv(os.path.join(output_dir, 'adv_test_results.csv'), index=False)
    except Exception as e:
        print(f"!!! ERROR during testing: {e}")
        import traceback
        traceback.print_exc()

    print("Training script finished.")

# --- Main Execution Block --- 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Advanced Transaction Classifier')
    
    # --- Data/Output Arguments --- 
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing transaction data Arrow files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints and logs')
    parser.add_argument('--config_path', type=str, default='config/model_config.yaml', help='Path to YAML model configuration file')
    parser.add_argument('--max_files_to_process', type=int, default=None, 
                        help='Optional: Process only the first N files found in data_dir for testing.')
    
    # --- Memory Management Arguments (for streaming) ---
    parser.add_argument('--streaming', action='store_true', 
                        help='Use iterative chunk-based streaming approach for large datasets')
    parser.add_argument('--max_transactions', type=int, default=500000,
                        help='Maximum number of transactions per chunk (streaming mode)')
    parser.add_argument('--max_files_for_stats', type=int, default=None,
                        help='Maximum number of files to use for statistics collection (streaming mode)')
    
    # --- Iterative Training Arguments (for streaming) ---
    parser.add_argument('--num_overall_epochs', type=int, default=1,
                        help='Number of full passes over the entire dataset via chunks (streaming mode)')
    parser.add_argument('--epochs_per_chunk', type=int, default=1,
                        help='Number of epochs to train on each chunk (streaming mode)')
    parser.add_argument('--total_chunks_to_process', type=int, default=None,
                        help='Maximum number of chunks to process for testing/debugging (streaming mode)')
    parser.add_argument('--early_stopping_patience_per_chunk', type=int, default=3,
                        help='Early stopping patience for each chunk training (streaming mode)')

    # --- Training Arguments --- 
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers (set to 0 for debugging)')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--accelerator', type=str, default='auto', help='Trainer accelerator (cpu, gpu, auto)')
    parser.add_argument('--precision', type=str, default='32', help='Trainer precision (e.g., 16, 32, bf16)')
    
    # <<< HGT Sampler Args >>>
    # Allow passing sampling config via CLI (complex, maybe better in YAML)
    # Example: --hgt_samples "{'transaction':[15,10], 'user':[10,5]}"
    parser.add_argument('--hgt_samples', type=str, default=None, 
                        help='JSON/YAML string defining num_samples per node type per layer for HGTLoader')
    
    # --- MTL/Focal Loss Arguments --- 
    parser.add_argument('--mtl_weight_global', type=float, default=1.0, help='Weight for global loss')
    parser.add_argument('--mtl_weight_user', type=float, default=0.0, help='Weight for user loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Alpha for Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for Focal Loss')

    args = parser.parse_args()

    # --- Load Model Config from YAML --- 
    print(f"Loading model configuration from: {args.config_path}")
    try:
        with open(args.config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        print("Model configuration loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found at {args.config_path}. Using default placeholders.")
        # Define placeholder default config if file not found
        model_config = {
            'user_embed_dim': 64,
            'graph_encoder_params': {'hidden_channels': 128, 'out_channels': 128, 'num_heads': 4, 'num_layers': 2},
            'sequence_encoder_params': {'output_dim': 128, 'tft_params': {}}, # Add necessary TFT params here
            'text_encoder_params': {'model_name': 'ProsusAI/finbert', 'projection_dim': 128},
            'fusion_params': {'hidden_dim': 128, 'output_dim': 256, 'dropout': 0.1}
        }
    except Exception as e:
        print(f"[ERROR] Failed to load or parse config file {args.config_path}: {e}")
        raise

    # --- Parse HGT Samples from CLI arg --- 
    hgt_samples_dict = None
    if args.hgt_samples:
        try:
            # Try parsing as JSON first, then YAML
            try: 
                import json
                hgt_samples_dict = json.loads(args.hgt_samples.replace("'", '"')) # Allow single quotes
            except json.JSONDecodeError:
                import yaml
                hgt_samples_dict = yaml.safe_load(args.hgt_samples)
            print(f"Parsed HGT samples from CLI: {hgt_samples_dict}")
        except Exception as e:
            print(f"[WARN] Failed to parse --hgt_samples argument: {e}. Using default.")
            hgt_samples_dict = None # Fallback to default in DataModule init

    # --- Run Training --- 
    if args.streaming:
        print("Using streaming training approach...")
        train_advanced_streaming(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_transactions=args.max_transactions,
            max_files_for_stats=args.max_files_for_stats,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_overall_epochs=args.num_overall_epochs,
            epochs_per_chunk=args.epochs_per_chunk,
            total_chunks_to_process=args.total_chunks_to_process,
            seed=args.seed,
            model_config=model_config,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            mtl_weights={'global': args.mtl_weight_global, 'user': args.mtl_weight_user},
            focal_loss_alpha=args.focal_alpha,
            focal_loss_gamma=args.focal_gamma,
            accelerator=args.accelerator,
            precision=args.precision,
            hgt_num_samples=hgt_samples_dict,
            early_stopping_patience_per_chunk=args.early_stopping_patience_per_chunk
        )
    else:
        print("Using traditional training approach...")
        train_advanced(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_epochs=args.max_epochs,
            seed=args.seed,
            model_config=model_config,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            mtl_weights={'global': args.mtl_weight_global, 'user': args.mtl_weight_user},
            focal_loss_alpha=args.focal_alpha,
            focal_loss_gamma=args.focal_gamma,
            accelerator=args.accelerator,
            precision=args.precision,
            hgt_num_samples=hgt_samples_dict,
            max_files_to_process=args.max_files_to_process
        ) 