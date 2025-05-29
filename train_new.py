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
from typing import Optional, Dict, List, Any
import pyarrow as pa # Added for ArrowInvalid check
import pyarrow.ipc as ipc # Use ipc explicitly for stream reading
import numpy as np # Added
import pickle # Added for saving state

# Use the V2 DataModule
from data.data_module_v2 import TransactionDataModuleV2, SingleBatchIterable 
# Import the new advanced model
from models.advanced_transaction_classifier import AdvancedTransactionCategorizationModel

# Helper functions for two-pass loading
def collect_statistics_pass(data_dir: str, max_files_for_stats: Optional[int] = None) -> Dict[str, Any]:
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
                    continue  # Skip problematic rows
                    
        except Exception as e:
            print(f"[WARN] Failed to process file {os.path.basename(file_path)}: {e}")
            continue
    
    print(f"Statistics collected from {stats['total_transactions_seen']:,} transactions")
    
    # Compute scalers from statistics
    stats['scalers'] = compute_scalers_from_stats(stats)
    stats['category_map'] = {cat: idx for idx, cat in enumerate(sorted(stats['category_counts'].keys()))}
    stats['user_map'] = {user: idx for idx, user in enumerate(sorted(stats['user_counts'].keys()))}
    
    print(f"Found {len(stats['category_map'])} unique categories, {len(stats['user_map'])} unique users")
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
    
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    # Ensure clean index for graph building
    final_df.reset_index(drop=True, inplace=True)
    
    # Verify index is continuous
    if not final_df.index.equals(pd.RangeIndex(len(final_df))):
        print(f"[WARN] Index discontinuity in streaming, forcing reset...")
        final_df.index = pd.RangeIndex(len(final_df))
    
    print(f"Final dataset: {len(final_df):,} transactions")
    return final_df

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

# Original load_data function (now deprecated)
def load_data(data_dir: str, max_files_to_process: Optional[int] = None) -> pd.DataFrame:
    """Loads and preprocesses data from multiple Arrow streaming files, one by one.
    Processes up to max_files_to_process if specified.
    """
    print(f"Loading data from directory: {data_dir}")
    arrow_files_pattern = os.path.join(data_dir, '**/*.arrow')
    # Sort files for deterministic behavior when using max_files_to_process
    arrow_files = sorted(glob.glob(arrow_files_pattern, recursive=True))
    
    if not arrow_files:
        if not os.path.isdir(data_dir):
             raise FileNotFoundError(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"No .arrow files found in directory: {data_dir}")

    # Limit number of files if max_files_to_process is set
    files_to_process = arrow_files
    if max_files_to_process is not None and max_files_to_process > 0:
        print(f"Limiting processing to the first {max_files_to_process} files found.")
        files_to_process = arrow_files[:max_files_to_process]
    
    print(f"Found {len(arrow_files)} arrow files. Processing {len(files_to_process)} file(s) file by file...")
    
    processed_dfs = []
    skipped_files = []
    
    for file_path in tqdm(files_to_process, desc="Processing Arrow files"):
        try:
            # 1. Read one Arrow stream file
            with ipc.open_stream(file_path) as reader:
                 table = reader.read_all()
                 df_single = table.to_pandas()
            
            # 2. Perform initial preprocessing on this single DataFrame
            required_cols = ['target_transaction_processed', 'txn_accepted_category_id_str', 'company_name']
            if not all(col in df_single.columns for col in required_cols):
                print(f"[WARN] Skipping file {os.path.basename(file_path)} due to missing required columns.")
                skipped_files.append(os.path.basename(file_path))
                continue # Skip to next file
                
            extracted_data = []
            for _, row in df_single.iterrows(): # Process rows within the small df
                try:
                    if isinstance(row['target_transaction_processed'], dict):
                        txn_dict = row['target_transaction_processed']
                    else:
                        txn_dict = json.loads(row['target_transaction_processed'])
                    
                    extracted_data.append({
                        'amount': float(txn_dict.get('amount', 0.0)),
                        'timestamp': pd.to_datetime(txn_dict.get('created_date'), errors='coerce'),
                        'description': str(txn_dict.get('description', '')),
                        'memo': str(txn_dict.get('memo', '')),
                        'merchant_name': str(txn_dict.get('payee', ''))
                    })
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    # print(f"[WARN] Error parsing target_transaction_processed in file {os.path.basename(file_path)}, row {row.name}: {e}. Using defaults.")
                    extracted_data.append({'amount': 0.0, 'timestamp': pd.NaT, 'description': '', 'memo': '', 'merchant_name': ''})
            
            extracted_df = pd.DataFrame(extracted_data, index=df_single.index)
            df_processed_single = pd.concat([df_single, extracted_df], axis=1)

            # Handle timestamps within the small df
            if df_processed_single['timestamp'].isnull().any():
                median_date = df_processed_single['timestamp'].dropna().median()
                if pd.isna(median_date):
                    median_date = pd.Timestamp('2020-01-01') 
                df_processed_single['timestamp'].fillna(median_date, inplace=True)

            # Extract time features within the small df
            df_processed_single['weekday'] = df_processed_single['timestamp'].dt.weekday
            df_processed_single['hour'] = df_processed_single['timestamp'].dt.hour
            
            # Ensure other columns exist and fill NaNs within the small df
            df_processed_single['txn_accepted_category_id_str'] = df_processed_single['txn_accepted_category_id_str'].fillna('UNKNOWN').astype(str)
            df_processed_single['company_name'] = df_processed_single['company_name'].fillna('UNKNOWN').astype(str)
            if 'industry_name' in df_processed_single.columns:
                 df_processed_single['industry_name'] = df_processed_single['industry_name'].fillna('UNKNOWN').astype(str)
            else: df_processed_single['industry_name'] = 'UNKNOWN'
            if 'num_chart_of_accounts' in df_processed_single.columns:
                 df_processed_single['num_chart_of_accounts'] = pd.to_numeric(df_processed_single['num_chart_of_accounts'], errors='coerce').fillna(0).astype(int)
            else: df_processed_single['num_chart_of_accounts'] = 0
            # Don't drop target_transaction_processed yet, might be needed later? Keep it for now.
            
            # Select only the columns needed downstream to potentially save memory before append
            cols_to_keep = [
                'txn_accepted_category_id_str', 'company_name', 'industry_name', 
                'num_chart_of_accounts', 'chart_of_accounts_processed', # Keep COA for DataModule
                'amount', 'timestamp', 'description', 'memo', 'merchant_name', 
                'weekday', 'hour'
                # Add any other original columns if they are used by DataModule/Model
            ]
            # Filter df_processed_single to keep only necessary columns
            df_filtered_single = df_processed_single[[col for col in cols_to_keep if col in df_processed_single.columns]]
            
            processed_dfs.append(df_filtered_single)

        except pa.lib.ArrowInvalid as e:
            print(f"[WARN] Skipping invalid Arrow stream file: {os.path.basename(file_path)} - Reason: {e}")
            skipped_files.append(os.path.basename(file_path))
        except Exception as e:
            print(f"[WARN] Skipping file {os.path.basename(file_path)} due to unexpected error: {e}")
            skipped_files.append(os.path.basename(file_path))

    if not processed_dfs:
        raise ValueError(f"No valid Arrow files could be processed from {data_dir}. Skipped files: {skipped_files}")

    # 3. Concatenate all *processed* DataFrames
    print(f"Concatenating {len(processed_dfs)} processed DataFrames..." )
    print("[WARNING] This step loads the full processed dataset into memory!")
    df_combined = pd.concat(processed_dfs, ignore_index=True)
    
    num_processed = len(processed_dfs)
    num_skipped = len(skipped_files)
    print(f"Data loading and initial processing complete: {len(df_combined)} records from {num_processed} files ({num_skipped} files skipped)." )
    if skipped_files:
         print(f"Skipped files: {skipped_files}")

    # No further processing needed here, return the combined df
    print("Preprocessing finished.") # Renamed log message
    return df_combined

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
    hgt_num_samples: Optional[Dict[str, List[int]]] = None
):
    """Train using two-pass approach for large datasets."""
    torch.set_float32_matmul_precision('high') 
    pl.seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    # === PASS 1: Collect Statistics ===
    print("Starting two-pass training approach...")
    stats = collect_statistics_pass(data_dir, max_files_for_stats)
    
    # === PASS 2: Load Data Subset ===
    df = load_data_with_scalers(data_dir, stats, max_transactions)
    
    # DEBUG: Check DataFrame index integrity
    print(f"[DEBUG] df.index range: {df.index.min()} to {df.index.max()}, len={len(df)}")
    print(f"[DEBUG] df.index is continuous: {df.index.equals(pd.RangeIndex(len(df)))}")
    print(f"[DEBUG] df.index dtype: {df.index.dtype}")
    if not df.index.equals(pd.RangeIndex(len(df))):
        print(f"[DEBUG] Index discontinuity detected! Expected 0-{len(df)-1}, but got min={df.index.min()}, max={df.index.max()}")
        print(f"[DEBUG] First 10 index values: {df.index[:10].tolist()}")
        print(f"[DEBUG] Last 10 index values: {df.index[-10:].tolist()}")
    
    # === Continue with existing training logic ===
    num_global_classes = len(stats['category_map'])
    num_user_classes = 0 # Global-only prediction
    num_users = len(stats['user_map'])
    print(f"Dataset stats: #Global={num_global_classes}, #Users={num_users}, #Transactions={len(df):,}")

    # DEBUGGING: Check for -1 user_id_codes
    if 'user_id_code' in df.columns and (df['user_id_code'] == -1).any():
        print(f"[!!!! DEBUG !!!!] Found {(df['user_id_code'] == -1).sum()} transactions with user_id_code == -1 in the DataFrame 'df'.")
        problematic_companies = df.loc[df['user_id_code'] == -1, 'company_name'].unique()
        print(f"[!!!! DEBUG !!!!] Unique company_names that mapped to -1 (first 10): {problematic_companies[:10]}")
        # Check if these problematic_companies are in stats['user_map']
        for comp in problematic_companies[:5]: # Check a few
            # Ensure comp is a string for the dictionary lookup
            comp_str = str(comp)
            if comp_str in stats['user_map']:
                print(f"[!!!! DEBUG !!!!] Problematic company '{comp_str}' IS in stats['user_map'] with ID {stats['user_map'][comp_str]} but df['user_id_code'] was -1. This is unexpected.")
            else:
                print(f"[!!!! DEBUG !!!!] Problematic company '{comp_str}' IS NOT in stats['user_map']. This is the primary issue.")
    else:
        print("[!!!! DEBUG !!!!] No user_id_code == -1 found in DataFrame 'df'.")
    
    # Create DataModule with pre-computed scalers
    print("Initializing TransactionDataModuleV2 with pre-computed scalers...")
    use_seq = model_config.get('use_sequence_encoder', False)
    use_graph = model_config.get('use_graph_encoder', True)   
    use_text = model_config.get('use_text_encoder', True)     
    use_coa_text = model_config.get('use_coa_text_features', False)

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
        use_coa_text_features=use_coa_text,
        # Pass pre-computed scalers
        fitted_scalers=stats['scalers'],
        fitted_category_id_map=stats['category_map'],
        fitted_user_map=stats['user_map']
    )
    
    print("Setting up DataModuleV2...")
    data_module.setup('fit') 
    
    # Continue with model creation and training (same as before)
    # Update model config with dynamic values
    model_config['graph_encoder_params']['in_channels'] = data_module.node_feature_dims
    model_config['graph_encoder_params']['metadata'] = data_module.full_graph_data.metadata() 
    model_config['num_users'] = num_users
    model_config['num_global_classes'] = num_global_classes
    model_config['num_user_classes'] = num_user_classes
    
    # Update fusion dims
    model_config['fusion_params']['graph_dim'] = model_config['graph_encoder_params'].get('out_channels', 128)
    model_config['fusion_params']['seq_dim'] = model_config['sequence_encoder_params'].get('output_dim', 128)
    text_proj_dim = model_config['text_encoder_params'].get('projection_dim', 0)
    finbert_hidden_size = 768
    text_out_dim_for_fusion = text_proj_dim if text_proj_dim > 0 else finbert_hidden_size
    model_config['fusion_params']['text_dim'] = text_out_dim_for_fusion
    model_config['fusion_params']['user_dim'] = model_config['user_embed_dim']
    
    # Create model
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
    
    # Training setup (same as before)
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='adv_model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
    logger = TensorBoardLogger(save_dir=output_dir, name='adv_logs')

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
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=min(val_check_interval, 500),  # Log more frequently than validation
        val_check_interval=val_check_interval,
        accumulate_grad_batches=4,  # Effective batch size = batch_size * 4
    )

    # Train
    print("Starting Training...")
    try:
        trainer.fit(model, datamodule=data_module)
    except RuntimeError as e:
        if "CUDA error" in str(e):
            print(f"!!! CUDA ERROR during training: {e}")
            print("This may be due to:")
            print("1. Out of GPU memory - try reducing batch_size")
            print("2. Invalid CUDA operations - check data types and tensor operations")
            print("3. Hardware/driver issues - try restarting the training process")
            
            # Try to clear CUDA cache
            try:
                torch.cuda.empty_cache()
                print("CUDA cache cleared")
            except:
                pass
        else:
            print(f"!!! RUNTIME ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise e
    except Exception as e:
        print(f"!!! UNEXPECTED ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Save preprocessing state including original statistics
        print("Saving preprocessing state...")
        state_to_save = {
            'scalers': stats['scalers'],
            'category_id_map': stats['category_map'],
            'user_map': stats['user_map'],
            'original_stats': stats,  # Save original statistics for reference
            'max_transactions_used': len(df)
        }
        save_path = os.path.join(output_dir, 'preprocessing_state.pkl')
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(state_to_save, f)
            print(f"Preprocessing state saved to: {save_path}")
        except Exception as save_e:
            print(f"[ERROR] Failed to save preprocessing state: {save_e}")

    # Test
    print("Starting Testing...")
    try:
        test_results = trainer.test(datamodule=data_module, ckpt_path='best') 
        print("Test Results:", test_results)
        if test_results:
            results_df = pd.DataFrame(test_results)
            results_df.to_csv(os.path.join(output_dir, 'adv_test_results.csv'), index=False)
    except Exception as e:
        print(f"!!! ERROR during testing: {e}")
        import traceback
        traceback.print_exc()

    print("Streaming training script finished.")

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
                        help='Use two-pass streaming approach for large datasets')
    parser.add_argument('--max_transactions', type=int, default=1000000,
                        help='Maximum number of transactions to load (streaming mode)')
    parser.add_argument('--max_files_for_stats', type=int, default=None,
                        help='Maximum number of files to use for statistics collection (streaming mode)')

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
            hgt_num_samples=hgt_samples_dict
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