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
    max_transactions: int = 1000000,
    max_files_for_stats: Optional[int] = None,
    # Training params (same as before)
    batch_size: int = 32,
    num_workers: int = 0,
    max_epochs: int = 50,
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
    epochs_per_chunk: int = 1, # New parameter: how many epochs to train on each 500k chunk
    total_chunks_to_process: Optional[int] = None # New: Limit total chunks for testing
):
    """Train using iterative chunk-based approach for very large datasets."""
    torch.set_float32_matmul_precision('high') 
    pl.seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    # === PASS 1: Collect GLOBAL Statistics (ONCE) ===
    print("Starting GLOBAL statistics collection pass...")
    # CRITICAL: Ensure max_files_for_stats=None to scan all files for global stats
    global_stats = collect_statistics_pass(data_dir, max_files_for_stats=None, add_unknowns=True) 
    
    # Global parameters for model initialization
    # These maps are {name: code_int}
    global_user_map = global_stats['user_map']
    global_category_map_name_to_code = global_stats['category_map']
    
    global_num_users = len(global_user_map)
    global_num_global_classes = len(global_category_map_name_to_code)
    global_num_user_classes = 0 # Assuming no user-specific classes for now

    print(f"Global Stats: #Users={global_num_users}, #GlobalClasses={global_num_global_classes}")

    # === Initialize Model (ONCE) ===
    # Update model_config with GLOBAL counts
    current_model_config = model_config.copy() # Use a copy to avoid modifying the input dict directly
    # graph_encoder_params will be set per-chunk by DataModule, but num_users/classes are global
    current_model_config['num_users'] = global_num_users
    current_model_config['num_global_classes'] = global_num_global_classes
    current_model_config['num_user_classes'] = global_num_user_classes
    # Fusion dims also need to be set based on what encoders are active and their output dims
    # This part of model_config setup needs to be robust
    current_model_config['fusion_params']['graph_dim'] = current_model_config['graph_encoder_params'].get('out_channels', 128)
    current_model_config['fusion_params']['seq_dim'] = current_model_config['sequence_encoder_params'].get('output_dim', 128) # Ensure this key exists if seq encoder used
    text_proj_dim = current_model_config['text_encoder_params'].get('projection_dim', 0)
    finbert_hidden_size = 768 # Default for finbert
    text_out_dim_for_fusion = text_proj_dim if text_proj_dim > 0 else finbert_hidden_size
    current_model_config['fusion_params']['text_dim'] = text_out_dim_for_fusion
    current_model_config['fusion_params']['user_dim'] = current_model_config['user_embed_dim']

    # === Perform Dry Run to Get HGT Metadata ===
    print("Performing dry run to determine HGT metadata and node feature dimensions...")
    
    # Load a small sample to get metadata
    dry_run_chunk, _, _, _ = load_data_chunk_iteratively(
        all_arrow_files=sorted(glob.glob(os.path.join(data_dir, '**/*.arrow'), recursive=True)),
        current_file_idx=0,
        current_row_offset_in_file=0,
        max_transactions_per_chunk=min(10000, max_transactions // 10),  # Small sample for dry run
        global_stats=global_stats
    )
    
    if dry_run_chunk is None or dry_run_chunk.empty:
        raise ValueError("Dry run failed: Could not load sample data for metadata determination")
    
    print(f"Dry run loaded {len(dry_run_chunk)} transactions for metadata extraction")
    
    # Create temporary DataModule to extract metadata
    global_category_map_code_to_name = {v: k for k, v in global_category_map_name_to_code.items()}
    
    temp_data_module = TransactionDataModuleV2(
        transactions_df_ref=dry_run_chunk,
        batch_size=32,  # Small batch for dry run
        num_workers=0,  # No workers for dry run
        text_model_name=current_model_config['text_encoder_params'].get('model_name', 'ProsusAI/finbert'),
        max_seq_length=current_model_config.get('max_seq_length', 50),
        text_max_length=current_model_config['text_encoder_params'].get('max_length', 128),
        use_sequence_encoder=current_model_config.get('use_sequence_encoder', False),
        use_gnn_encoder=current_model_config.get('use_gnn_encoder', True),
        use_text_encoder=current_model_config.get('use_text_encoder', True),
        use_coa_text_features=current_model_config.get('use_coa_text_features', False),
        num_hgt_layers=current_model_config['graph_encoder_params'].get('num_layers', 2),
        hgt_num_samples=hgt_num_samples,
        fitted_scalers=global_stats['scalers'],
        fitted_category_id_map=global_category_map_code_to_name,
        fitted_user_map=global_user_map
    )
    
    print("Setting up temporary DataModule for metadata extraction...")
    temp_data_module.setup('fit')
    
    # Extract metadata
    initial_node_feature_dims = temp_data_module.node_feature_dims
    initial_graph_metadata = temp_data_module.full_graph_data.metadata()
    
    print(f"Extracted node_feature_dims: {initial_node_feature_dims}")
    print(f"Extracted graph metadata: {initial_graph_metadata}")
    
    # Clean up temporary DataModule
    del temp_data_module
    del dry_run_chunk
    
    # Ensure graph_encoder_params exists and is a dict
    if not isinstance(current_model_config.get('graph_encoder_params'), dict):
        print("[DEBUG] 'graph_encoder_params' was not a dict or not found in current_model_config. Initializing.")
        current_model_config['graph_encoder_params'] = {}
    
    # Set GNN params from dry run
    print(f"[DEBUG] Assigning initial_graph_metadata: {type(initial_graph_metadata)}")
    current_model_config['graph_encoder_params']['metadata'] = initial_graph_metadata
    
    print(f"[DEBUG] Assigning initial_node_feature_dims: {initial_node_feature_dims} (type: {type(initial_node_feature_dims)}) to 'in_channels'")
    current_model_config['graph_encoder_params']['in_channels'] = initial_node_feature_dims
    
    print(f"[DEBUG] current_model_config['graph_encoder_params'] content after setting 'in_channels':")
    print(f"         {current_model_config['graph_encoder_params']}")
    
    if 'in_channels' in current_model_config['graph_encoder_params']:
        print(f"[DEBUG] Key 'in_channels' IS PRESENT in current_model_config['graph_encoder_params'].")
        print(f"         Value: {current_model_config['graph_encoder_params']['in_channels']}")
    else:
        print(f"[DEBUG] Key 'in_channels' IS MISSING from current_model_config['graph_encoder_params'] just before model init!")
        print(f"         Keys present: {current_model_config['graph_encoder_params'].keys()}")

    print("Initializing AdvancedTransactionCategorizationModel globally...")
    model = AdvancedTransactionCategorizationModel(
        model_config=current_model_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mtl_weights=mtl_weights,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        # transactions_df_ref is not really used by the model if data comes from dataloader
    )

    # === Iterative Training Loop ===
    all_arrow_files = sorted(glob.glob(os.path.join(data_dir, '**/*.arrow'), recursive=True))
    if not all_arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {data_dir} for iterative training.")

    current_file_idx = 0
    current_row_offset = 0
    more_data_to_load = True
    chunk_number = 0
    last_checkpoint_path = None

    while more_data_to_load:
        if total_chunks_to_process is not None and chunk_number >= total_chunks_to_process:
            print(f"Reached maximum number of chunks to process: {total_chunks_to_process}.")
            break
        chunk_number += 1
        print(f"\n--- Processing Chunk {chunk_number} ---")

        df_chunk, next_file_idx, next_row_offset, more_data_available_after_this_chunk = load_data_chunk_iteratively(
            all_arrow_files=all_arrow_files,
            current_file_idx=current_file_idx,
            current_row_offset_in_file=current_row_offset,
            max_transactions_per_chunk=max_transactions, # Your 500k limit
            global_stats=global_stats
        )

        current_file_idx = next_file_idx
        current_row_offset = next_row_offset
        more_data_to_load = more_data_available_after_this_chunk

        if df_chunk is None or df_chunk.empty:
            print("No more data to load or empty chunk returned.")
            break

        print(f"Chunk {chunk_number} loaded with {len(df_chunk)} transactions.")
        
        # DataModule for the current chunk
        # DataModuleV2 expects fitted_category_id_map as {code: name_str}
        # global_category_map_name_to_code is {name: code}
        # So we need to invert it for the DataModule or change DataModule
        # For now, let's invert it here:
        global_category_map_code_to_name = {v: k for k, v in global_category_map_name_to_code.items()}

        data_module = TransactionDataModuleV2(
            transactions_df_ref=df_chunk,
            batch_size=batch_size,
            num_workers=num_workers,
            # ... (other DataModule params like text_model_name, max_seq_length etc.) ...
            # Make sure these are passed correctly:
            text_model_name=current_model_config['text_encoder_params'].get('model_name', 'ProsusAI/finbert'),
            max_seq_length=current_model_config.get('max_seq_length', 50), # Get from main config
            text_max_length=current_model_config['text_encoder_params'].get('max_length', 128),
            use_sequence_encoder=current_model_config.get('use_sequence_encoder', False),
            use_gnn_encoder=current_model_config.get('use_gnn_encoder', True),   
            use_text_encoder=current_model_config.get('use_text_encoder', True),     
            use_coa_text_features=current_model_config.get('use_coa_text_features', False),
            num_hgt_layers = current_model_config['graph_encoder_params'].get('num_layers', 2),
            hgt_num_samples = hgt_num_samples, # From main args

            fitted_scalers=global_stats['scalers'],
            fitted_category_id_map=global_category_map_code_to_name, # {code:name}
            fitted_user_map=global_user_map # {name:code} - check DataModuleV2 consumes this correctly
        )
        
        print(f"Setting up DataModule for chunk {chunk_number}...")
        data_module.setup('fit') 
        
        # Update graph-specific parts of model_config if they change per chunk (e.g. metadata from HeteroData)
        # This is tricky because GNN metadata/in_channels depend on the *current chunk's graph structure*
        # If node types or feature dims can vary wildly per chunk, this is complex.
        # Assuming for now that the *types* of nodes/edges are consistent enough for global GNN init.
        # The DataModule must provide consistent node_feature_dims keys.
        # If HGT metadata changes, the model cannot be simply resumed.
        # For now, assume metadata from the first chunk's setup (or a global one) is okay.
        # It's safer if the model's GNN part is initialized with metadata from global_stats or a representative first chunk.
        # The `in_channels` for HGT must match what DataModule produces.
        
        # Let's assume metadata is stable, and we set it once during model init
        # If model.graph_encoder.metadata is not set, it needs to be.
        # And model_config['graph_encoder_params']['in_channels'] needs to be accurate.
        # This might require a "dry run" of data_module.setup() on a small sample with global_stats
        # just to get metadata and node_feature_dims before initializing the main model.
        # For now, the model was initialized with a placeholder metadata. Let's try to update it IF POSSIBLE,
        # but this is a complex aspect of iterative GNN training.
        # The safest is to ensure the GNN config used for the *single model instance* is compatible with all chunks.
        # This means `data_module.node_feature_dims` and `data_module.full_graph_data.metadata()` from any chunk
        # must be compatible with the one-time initialized HGT.
        # This usually means all possible node types and their feature dimensions are known upfront.

        # Trainer for the current chunk
        # Checkpoint callback needs to be specific for this chunk or managed globally
        chunk_output_dir = os.path.join(output_dir, f"chunk_{chunk_number}")
        os.makedirs(chunk_output_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=chunk_output_dir, # Save chunk-specific checkpoints
            filename=f'model-chunk{chunk_number}-{{epoch:02d}}-{{val_loss:.2f}}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )
        # Early stopping might be per chunk or global; per-chunk is simpler here.
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=True) # Increased patience
        
        # Logger can also be per chunk
        logger = TensorBoardLogger(save_dir=os.path.join(output_dir, "tensorboard_logs"), name=f"chunk_{chunk_number}")

        trainer = pl.Trainer(
            max_epochs=epochs_per_chunk, 
            accelerator=accelerator,
            precision=precision,
            devices=1, # Assuming single device
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            log_every_n_steps=min(val_check_interval if 'val_check_interval' in locals() else 100, 50), 
            val_check_interval=0.5, # Or some fraction of steps in the chunk
            accumulate_grad_batches=4, # From original args
        )

        print(f"Fitting model on chunk {chunk_number}...")
        trainer.fit(model, datamodule=data_module, ckpt_path=last_checkpoint_path)
        
        # Update last_checkpoint_path for the next iteration
        last_checkpoint_path = checkpoint_callback.best_model_path
        if not last_checkpoint_path or not os.path.exists(last_checkpoint_path):
            print(f"[WARN] Best model path from checkpoint_callback for chunk {chunk_number} is invalid: {last_checkpoint_path}. Resuming may fail.")
            # Fallback or error handling needed here if checkpoints are critical for resumption
            # For simplicity, if no checkpoint, next iteration will train from current model state in memory.

        print(f"Finished training on chunk {chunk_number}. Best model for this chunk: {last_checkpoint_path}")

        # Optional: Clean up older non-best checkpoints for this chunk if needed to save space

    print("Finished processing all data chunks.")

    # --- Final Steps (e.g., Save final model, preprocessing state, Test) ---
    final_model_save_path = os.path.join(output_dir, "final_trained_model.ckpt")
    trainer.save_checkpoint(final_model_save_path) # Save the very last state
    print(f"Final trained model saved to {final_model_save_path}")

    print("Saving final (global) preprocessing state...")
    state_to_save = {
        'scalers': global_stats['scalers'],
        'category_id_map_name_to_code': global_category_map_name_to_code, # {name:code}
        'user_map_name_to_code': global_user_map, # {name:code}
        # Add other relevant global stats if needed
    }
    save_path = os.path.join(output_dir, 'global_preprocessing_state.pkl')
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(state_to_save, f)
        print(f"Global preprocessing state saved to: {save_path}")
    except Exception as save_e:
        print(f"[ERROR] Failed to save global preprocessing state: {save_e}")

    # Testing (optional, on a held-out test set or a final validation chunk)
    # This would require loading a test chunk and using trainer.test()
    # ...

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
    parser.add_argument('--epochs_per_chunk', type=int, default=1,
                        help='Number of epochs to train on each chunk (streaming mode)')
    parser.add_argument('--total_chunks_to_process', type=int, default=None,
                        help='Maximum number of chunks to process for testing/debugging (streaming mode)')

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
            epochs_per_chunk=args.epochs_per_chunk,
            total_chunks_to_process=args.total_chunks_to_process
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