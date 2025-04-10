import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import json
import argparse
import os # Added for checking file existence

def load_data(data_path: str) -> pd.DataFrame:
    """Load transaction data from CSV or parquet file."""
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
         raise FileNotFoundError(f"Input data file not found: {data_path}")
         
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        # Attempt to read CSV, handling potential low_memory warnings
        try:
             df = pd.read_csv(data_path, low_memory=False)
        except Exception as e:
             print(f"Error reading CSV file: {e}")
             raise
    else:
        raise ValueError(f"Unsupported file type: {data_path}. Please use .csv or .parquet")

    # Identify timestamp column
    if 'posted_date' in df.columns:
        timestamp_col = 'posted_date'
    elif 'books_create_timestamp' in df.columns:
        timestamp_col = 'books_create_timestamp'
    else:
        # Handle case where neither preferred timestamp column exists
        print("[WARN] Neither 'posted_date' nor 'books_create_timestamp' found.")
        # Check for any column containing 'time' or 'date'
        potential_ts_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if potential_ts_cols:
            timestamp_col = potential_ts_cols[0] # Use the first potential match
            print(f"Using column '{timestamp_col}' as timestamp.")
        else:
             # If absolutely no suitable column, raise error
             raise ValueError("Could not identify a suitable timestamp column.")

    # Rename the identified column to 'timestamp' for consistency if needed
    if timestamp_col != 'timestamp':
        print(f"Renaming column '{timestamp_col}' to 'timestamp'.")
        df['timestamp'] = df[timestamp_col]
        # Optionally drop the original column if needed
        # df = df.drop(columns=[timestamp_col])
    
    print(f"Identified '{timestamp_col}' as the timestamp column.")
    return df

def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    label_encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical features using LabelEncoder."""
    print("Encoding categorical features...")
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_columns:
        if col not in df.columns:
             print(f"  [WARN] Categorical column '{col}' not found. Skipping encoding.")
             continue
             
        # Fill NaN before encoding - crucial!
        # Use 'unknown' as the fill value for consistency
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
             print(f"  Filling {nan_count} NaNs in '{col}' with 'unknown'.")
             df[col] = df[col].fillna('unknown')
             
        # Ensure string type before encoding
        df[col] = df[col].astype(str)
        
        if col not in label_encoders:
            print(f"  Fitting LabelEncoder for: {col}")
            label_encoders[col] = LabelEncoder()
            # Fit and transform
            df[col] = label_encoders[col].fit_transform(df[col])
        else:
            # This part is more relevant for applying existing encoders to new data
            # For initial processing, fit_transform is sufficient.
            # Let's simplify this for clarity in the initial preprocess script.
            print(f"  LabelEncoder already exists for {col}. Refitting.") # Or raise error?
            label_encoders[col] = LabelEncoder() # Re-initialize
            df[col] = label_encoders[col].fit_transform(df[col]) # Refit

    return df, label_encoders

def preprocess_data(
    data_path: str,
    output_path: str,
    label_mapping_path: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Preprocess transaction data: Load, identify timestamp, encode specified categoricals."""
    # Load data
    df = load_data(data_path)
    
    # Define default categorical columns to attempt encoding
    if categorical_columns is None:
        # List potential categorical columns
        default_categorical_columns = [
            'merchant_name',
            'merchant_city',
            'merchant_state',
            'description', # Keep raw text, maybe don't encode here? depends on downstream
            'memo',        # Keep raw text, maybe don't encode here? depends on downstream
            'mcc_name',
            'account_type_id',
            'tax_account_type',
            'company_name',
            'industry_name',
            'region_name',
            'language_name',
            'category_name', # Name corresponding to category_id
            'category_id',     # Target label (global) - Ensure this is handled correctly (factorized if string)
            'user_category_id' # Target label (user-specific, if exists)
            # Add other known categoricals like 'locale', 'transaction_type' etc. if needed
        ]
        # Filter out columns that don't exist in the dataframe
        categorical_columns = [col for col in default_categorical_columns if col in df.columns]
        print(f"Using categorical columns for encoding: {categorical_columns}")
    
    # Encode categorical features (handles NaN filling for these columns)
    df, label_encoders = encode_categorical_features(df, categorical_columns)
    
    # Save minimally preprocessed data
    print(f"Saving minimally preprocessed data to {output_path}...")
    try:
         df.to_csv(output_path, index=False)
    except Exception as e:
         print(f"Error saving output file {output_path}: {e}")
         raise
         
    # Save label mappings if provided
    if label_mapping_path:
        print(f"Saving label mappings to {label_mapping_path}...")
        label_mappings = {}
        for col, encoder in label_encoders.items():
            if hasattr(encoder, 'classes_'): # Check if encoder was actually fitted
                 try:
                      # Create mapping from original class label (string) to integer code
                      mapping = {str(class_): int(idx) for idx, class_ in enumerate(encoder.classes_)}
                      label_mappings[col] = mapping
                 except Exception as map_err:
                      print(f"  [WARN] Error creating mapping for '{col}': {map_err}")
            else:
                 print(f"  [WARN] Could not generate mapping for '{col}' (likely wasn't fitted).")
        
        try:
             with open(label_mapping_path, 'w') as f:
                 json.dump(label_mappings, f, indent=2)
        except Exception as e:
             print(f"Error writing label mapping file {label_mapping_path}: {e}")
             # Decide whether to raise or just warn
             
    return df, label_encoders

def main(
    input_dir: str,
    output_path: str,
    label_mapping_path: Optional[str] = None,
    min_user_txns: int = 10, # Min transactions per user to be considered
    min_company_txns: int = 50, # Min transactions per company to be considered
    min_category_samples: int = 100, # Min samples per category in final dataset
    target_sample_size: int = 2_000_000 # Target size for the output dataset
):
    """Main function for sampling, balancing, and preprocessing."""
    try:
        # Create the final dataset using the new workflow
        create_sampled_dataset(
            input_dir=input_dir,
            output_path=output_path,
            label_mapping_path=label_mapping_path,
            min_user_txns=min_user_txns,
            min_company_txns=min_company_txns,
            min_category_samples=min_category_samples,
            target_sample_size=target_sample_size
        )
        print(f"\nPreprocessing and sampling complete. Output saved to {output_path}")
        if label_mapping_path:
            print(f"Label mappings saved to {label_mapping_path}")
    except Exception as e:
        print(f"\n--- PREPROCESSING FAILED ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("--------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample, balance, and preprocess transaction data from Parquet files.')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input Parquet files.')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save the final sampled and preprocessed data (CSV).')
    parser.add_argument('--label_mapping_path', type=str, default=None,
                      help='Path to save label mappings for encoded columns (JSON).')
    parser.add_argument('--min_user_txns', type=int, default=10,
                      help='Minimum transactions per user to include.')
    parser.add_argument('--min_company_txns', type=int, default=50,
                      help='Minimum transactions per company to include.')
    parser.add_argument('--min_category_samples', type=int, default=100,
                      help='Minimum samples per category_id in the final dataset.')
    parser.add_argument('--target_sample_size', type=int, default=2_000_000,
                      help='Approximate target number of transactions in the final dataset.')

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_path=args.output_path,
        label_mapping_path=args.label_mapping_path,
        min_user_txns=args.min_user_txns,
        min_company_txns=args.min_company_txns,
        min_category_samples=args.min_category_samples,
        target_sample_size=args.target_sample_size
    )

# --- NEW Sampling Workflow Functions --- 

def _scan_parquet_files(input_dir: str) -> pd.DataFrame:
    """Scans parquet files to get counts of users, companies, and categories."""
    print(f"Scanning Parquet files in: {input_dir}")
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.parquet')]
    if not all_files:
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")

    # Define columns needed for scanning
    scan_cols = ['user_id', 'company_id', 'category_id'] 
    # Add a unique identifier if possible, otherwise use index later
    # if 'transaction_id' in known_columns: scan_cols.append('transaction_id')

    all_metadata = []
    total_txns = 0
    for i, file_path in enumerate(all_files):
        try:
            # Read only necessary columns
            df_chunk = pd.read_parquet(file_path, columns=scan_cols)
            # Ensure category_id is treated consistently (handle potential strings/NaNs if needed)
            # Convert to string first to handle mixed types, then fillna, then maybe back to int if needed elsewhere
            df_chunk['category_id'] = df_chunk['category_id'].astype(str).fillna('-1') 
            df_chunk['company_id'] = df_chunk['company_id'].astype(str).fillna('unknown')
            df_chunk['user_id'] = df_chunk['user_id'].astype(str).fillna('unknown')
            
            # Store the file path along with the data
            df_chunk['_source_file'] = file_path 
            all_metadata.append(df_chunk)
            total_txns += len(df_chunk)
            print(f"  Scanned {i+1}/{len(all_files)}: {os.path.basename(file_path)} ({len(df_chunk)} rows)")
        except Exception as e:
            print(f"  [WARN] Failed to read or process {file_path}: {e}")
    
    if not all_metadata:
         raise ValueError("Failed to load metadata from any Parquet file.")

    print(f"Scanning complete. Total transactions found: {total_txns}")
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    # Ensure category_id is treated as a consistent type for grouping/counting
    metadata_df['category_id'] = metadata_df['category_id'].astype(str) 
    return metadata_df

def _filter_users_companies(metadata_df: pd.DataFrame, min_user_txns: int, min_company_txns: int) -> pd.DataFrame:
    """Filters metadata based on minimum transaction counts per user and company."""
    print("Filtering users and companies based on minimum transaction counts...")
    user_counts = metadata_df['user_id'].value_counts()
    company_counts = metadata_df['company_id'].value_counts()

    valid_users = user_counts[user_counts >= min_user_txns].index
    valid_companies = company_counts[company_counts >= min_company_txns].index

    print(f"  Keeping {len(valid_users)} users (>= {min_user_txns} txns).")
    print(f"  Keeping {len(valid_companies)} companies (>= {min_company_txns} txns).")

    filtered_df = metadata_df[
        metadata_df['user_id'].isin(valid_users) & 
        metadata_df['company_id'].isin(valid_companies)
    ].copy() # Add copy to avoid SettingWithCopyWarning
    print(f"  Metadata reduced to {len(filtered_df)} transactions after user/company filtering.")
    return filtered_df

def _calculate_sampling_numbers(filtered_df: pd.DataFrame, min_category_samples: int, target_sample_size: int) -> pd.Series:
    """Calculates how many samples to take per category_id."""
    print("Calculating samples needed per category...")
    # Ensure category_id is string for consistent grouping, even if originally numeric
    category_counts = filtered_df['category_id'].astype(str).value_counts()
    num_categories = len(category_counts)
    print(f"  Found {num_categories} unique categories after filtering.")

    # --- Stratified Sampling Logic --- 
    # Strategy: 
    # 1. Guarantee `min_category_samples` for each category (up to its available count).
    # 2. Distribute the remaining `target_sample_size` proportionally among categories.
    
    # Step 1: Base minimum samples
    samples_per_cat = category_counts.clip(upper=min_category_samples)
    guaranteed_samples = samples_per_cat.sum()
    print(f"  Guaranteed minimum samples: {guaranteed_samples}")

    # Step 2: Remaining samples to distribute
    remaining_target = max(0, target_sample_size - guaranteed_samples)
    print(f"  Remaining target samples to distribute proportionally: {remaining_target}")

    if remaining_target > 0 and category_counts.sum() > 0:
        # Calculate proportions based on counts *after* filtering
        proportions = category_counts / category_counts.sum()
        # Calculate additional samples needed based on proportion
        additional_samples_ideal = (proportions * remaining_target).round().astype(int)
        
        # Distribute additional samples, ensuring we don't exceed available count
        for cat_id, additional in additional_samples_ideal.items():
            available = category_counts[cat_id]
            current_alloc = samples_per_cat[cat_id]
            can_take = available - current_alloc # How many more can this category provide?
            take_now = min(additional, can_take)
            samples_per_cat[cat_id] += take_now

        # Check if we allocated enough, handle rounding differences if needed
        total_allocated = samples_per_cat.sum()
        print(f"  Total samples allocated after proportional distribution: {total_allocated}")
        # Simple adjustment: If under target due to rounding, add difference to largest cats? (Or other strategy)
        shortfall = target_sample_size - total_allocated
        if shortfall > 0:
             print(f"    Adjusting for rounding shortfall of {shortfall} samples...")
             # Distribute shortfall to categories that still have available samples
             sorted_cats = category_counts.sort_values(ascending=False).index
             for cat_id in sorted_cats:
                  if shortfall <= 0: break
                  available = category_counts[cat_id]
                  current_alloc = samples_per_cat[cat_id]
                  can_take = available - current_alloc
                  take_now = min(shortfall, can_take)
                  if take_now > 0:
                      samples_per_cat[cat_id] += take_now
                      shortfall -= take_now
             print(f"  Final allocated samples after adjustment: {samples_per_cat.sum()}")
    elif remaining_target <= 0:
         print("  Target size met or exceeded by minimums, no proportional allocation needed.")
    else:
         print("  No transactions left after filtering, cannot allocate proportionally.")

    # Ensure final counts don't exceed available
    samples_per_cat = samples_per_cat.clip(upper=category_counts)

    print(f"  Final calculated samples per category (top 5):\n{samples_per_cat.head()}")
    return samples_per_cat

def _sample_transactions(filtered_df: pd.DataFrame, samples_per_cat: pd.Series) -> pd.DataFrame:
    """Performs stratified sampling based on calculated numbers per category."""
    print("Sampling transactions based on calculated numbers...")
    
    def safe_sample(group, n):
        group_size = len(group)
        sample_n = min(n, group_size)
        if sample_n <= 0: 
             return pd.DataFrame(columns=group.columns) # Return empty frame if n=0
        return group.sample(n=sample_n, replace=False, random_state=42) 

    # Apply sampling per category
    # Convert category_id to string for consistent grouping with samples_per_cat index
    sampled_indices = filtered_df.groupby(filtered_df['category_id'].astype(str), group_keys=False).apply(
        lambda x: safe_sample(x, int(samples_per_cat.get(x.name, 0))) # Ensure n is int
    ).index

    sampled_metadata = filtered_df.loc[sampled_indices]
    print(f"  Sampled {len(sampled_metadata)} transactions.")
    print(f"  Category distribution in sample (top 5):\n{sampled_metadata['category_id'].value_counts().head()}")
    return sampled_metadata

def _load_full_data(sampled_metadata: pd.DataFrame) -> pd.DataFrame:
    """Loads the full data for the sampled transactions from their source files."""
    print("Loading full data for sampled transactions...")
    if '_source_file' not in sampled_metadata.columns:
        raise ValueError("'_source_file' column missing from sampled metadata.")
        
    source_files = sampled_metadata['_source_file'].unique()
    all_sampled_data = []

    # Prepare lookup: group sampled indices by source file
    grouped_indices = sampled_metadata.groupby('_source_file').groups

    for i, file_path in enumerate(source_files):
        print(f"  Loading from source file {i+1}/{len(source_files)}: {os.path.basename(file_path)}")
        try:
            # Read the full file
            df_full = pd.read_parquet(file_path)
            # Get the indices that belong to this file from the sampled metadata
            indices_to_load = grouped_indices[file_path]
            # Select rows based on the original index
            df_selected = df_full.loc[df_full.index.isin(indices_to_load)]
            all_sampled_data.append(df_selected)
            print(f"    Loaded {len(df_selected)} rows.")
        except Exception as e:
            print(f"  [WARN] Failed to load full data from {file_path}: {e}")
    
    if not all_sampled_data:
         raise ValueError("Failed to load full data for any sampled transaction.")

    final_df = pd.concat(all_sampled_data)
    print(f"Loaded full data for {len(final_df)} transactions.")
    # Reindex based on the sampled metadata to ensure correct order/rows
    final_df = final_df.reindex(sampled_metadata.index)
    
    if len(final_df) != len(sampled_metadata):
         print(f"[WARN] Final DataFrame size ({len(final_df)}) differs from sampled metadata size ({len(sampled_metadata)}) after reindexing. Check for duplicate indices or loading issues.")
    
    return final_df

def create_sampled_dataset(
    input_dir: str,
    output_path: str,
    label_mapping_path: Optional[str] = None,
    min_user_txns: int = 10,
    min_company_txns: int = 50,
    min_category_samples: int = 100,
    target_sample_size: int = 2_000_000
):
    """Orchestrates the sampling and preprocessing workflow."""
    # 1. Scan files for metadata
    metadata_df = _scan_parquet_files(input_dir)
    
    # 2. Filter users/companies
    filtered_df = _filter_users_companies(metadata_df, min_user_txns, min_company_txns)
    if filtered_df.empty:
        print("[ERROR] No transactions remaining after filtering users/companies. Aborting.")
        return

    # 3. Calculate sampling numbers per category
    samples_per_cat = _calculate_sampling_numbers(filtered_df, min_category_samples, target_sample_size)

    # 4. Sample transaction metadata
    sampled_metadata = _sample_transactions(filtered_df, samples_per_cat)
    if sampled_metadata.empty:
        print("[ERROR] No transactions selected during sampling. Aborting.")
        return
        
    # 5. Load full data for the sample
    final_df = _load_full_data(sampled_metadata)
    
    # 6. Final Processing (Timestamp ID, Categorical Encoding)
    print("Performing final processing on sampled data...")
    # Identify timestamp (assuming load_data logic can be reused or adapted)
    # Find timestamp column (reusing part of load_data logic)
    if 'posted_date' in final_df.columns:
        timestamp_col = 'posted_date'
    elif 'books_create_timestamp' in final_df.columns:
        timestamp_col = 'books_create_timestamp'
    else:
        potential_ts_cols = [col for col in final_df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if potential_ts_cols:
            timestamp_col = potential_ts_cols[0]
        else:
            # Fallback: If still no timestamp, maybe skip or use index?
            print("[WARN] Could not identify timestamp column in final sampled data. Timestamp-dependent features in DataModule might fail.")
            timestamp_col = None # Indicate no timestamp column found
            
    if timestamp_col and timestamp_col != 'timestamp':
        print(f"  Renaming '{timestamp_col}' to 'timestamp'.")
        final_df['timestamp'] = final_df[timestamp_col]

    # Define categoricals for encoding (reuse logic from old preprocess_data)
    default_categorical_columns = [
            'merchant_name',
            'merchant_city',
            'merchant_state',
            'description',
            'memo',
            'mcc_name',
            'account_type_id',
            'tax_account_type',
            'company_name',
            'industry_name',
            'region_name',
            'language_name',
            'category_name',
            'category_id',
            'user_category_id'
    ]
    categorical_columns_to_encode = [col for col in default_categorical_columns if col in final_df.columns]
    print(f"  Encoding categorical columns: {categorical_columns_to_encode}")
    final_df, label_encoders = encode_categorical_features(final_df, categorical_columns_to_encode)
    
    # 7. Save final dataset and mappings
    print(f"Saving final sampled dataset ({len(final_df)} rows) to {output_path}...")
    final_df.to_csv(output_path, index=False)
    
    if label_mapping_path:
        print(f"Saving label mappings to {label_mapping_path}...")
        label_mappings = {}
        for col, encoder in label_encoders.items():
            if hasattr(encoder, 'classes_'):
                try:
                    mapping = {str(class_): int(idx) for idx, class_ in enumerate(encoder.classes_)}
                    label_mappings[col] = mapping
                except Exception as map_err:
                    print(f"  [WARN] Error creating mapping for '{col}': {map_err}")
            else:
                print(f"  [WARN] Could not generate mapping for '{col}'.")
        try:
            with open(label_mapping_path, 'w') as f:
                json.dump(label_mappings, f, indent=2)
        except Exception as e:
            print(f"Error writing label mapping file {label_mapping_path}: {e}")


# --- OLD function (kept for reference or single file use if needed) ---
# def preprocess_data(
#     data_path: str,
#     output_path: str,
#     label_mapping_path: Optional[str] = None,
#     categorical_columns: Optional[List[str]] = None
# ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
#     """Preprocess transaction data: Load, identify timestamp, encode specified categoricals."""
#     # ... (previous implementation) ...


