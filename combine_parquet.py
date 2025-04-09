import pandas as pd
import dask.dataframe as dd
import numpy as np
import glob
import os
import argparse
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process and balance transaction data with user-centric and category-aware sampling.')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing parquet files to process')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save processed files')
    parser.add_argument('--file_pattern', type=str, default='*.parquet',
                        help='Pattern to match parquet files (e.g., "transaction_*.parquet")')
    parser.add_argument('--target_size', type=int, default=2500000,
                        help='Target size for final dataset (total train+val+test)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proportion of data for training set')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Proportion of data for validation set')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (sample only a few files)')
    parser.add_argument('--balance_categories', action='store_true',
                        help='Balance training data by category')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def log_info(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_required_columns():
    """Define the columns needed for the DataModule."""
    essential_columns = [
        'user_id',              # Essential for user-based splitting
        'category_id',          # Target variable 1
        'user_category_id',     # Target variable 2
        'timestamp',            # For time features and sorting
        'amount',               # For node/edge/sequence features
        'merchant_name',        # For merchant nodes and text
        'raw_description',      # For text features (or 'description')
        'memo',                 # For text features
    ]
    
    # Optional fallback columns
    optional_columns = [
        'posted_date',          # Potential timestamp fallback
        'books_create_timestamp'  # Potential timestamp fallback
    ]
    
    return essential_columns, optional_columns

def main():
    """Main function to process transaction data."""
    start_time = time.time()
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get required columns
    essential_columns, optional_columns = get_required_columns()
    all_columns = essential_columns + optional_columns
    
    # Find all parquet files
    parquet_path = os.path.join(args.input_dir, args.file_pattern)
    all_files = sorted(glob.glob(parquet_path))
    
    if not all_files:
        raise ValueError(f"No parquet files found matching '{parquet_path}'")
    
    log_info(f"Found {len(all_files)} parquet files")
    
    # Debug mode: use only a few files
    if args.debug:
        all_files = all_files[:5]
        log_info(f"Debug mode: Using only {len(all_files)} files")
    
    # Step 1: Get all unique users and categories using Dask
    log_info("Step 1: Finding all unique users and categories...")
    
    # Read all parquet files with Dask, selecting only needed columns for user/category identification
    ddf = dd.read_parquet(
        all_files,
        columns=['user_id', 'category_id'],
        engine='pyarrow'
    )
    
    # Get unique users
    log_info("Extracting unique user IDs...")
    unique_users = ddf['user_id'].drop_duplicates().compute().values
    log_info(f"Found {len(unique_users)} unique users")
    
    # Get unique categories
    log_info("Extracting unique category IDs...")
    unique_categories = ddf['category_id'].drop_duplicates().compute().values
    log_info(f"Found {len(unique_categories)} unique categories")
    
    # Step 2: Split users into train/val/test sets
    log_info("Step 2: Splitting users into train/val/test sets...")
    np.random.seed(args.seed)
    
    # Calculate split ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    
    # Split users into train and temp (val+test)
    train_users, temp_users = train_test_split(
        unique_users,
        test_size=(args.val_ratio + test_ratio),
        random_state=args.seed
    )
    
    # Split temp into val and test
    val_users, test_users = train_test_split(
        temp_users,
        test_size=test_ratio / (args.val_ratio + test_ratio),
        random_state=args.seed
    )
    
    log_info(f"User split: Train={len(train_users)}, Val={len(val_users)}, Test={len(test_users)}")
    
    # Convert to sets for faster lookup
    train_users_set = set(train_users)
    val_users_set = set(val_users)
    test_users_set = set(test_users)
    
    # Step 3: Read all files with Dask and filter by user sets
    log_info("Step 3: Filtering data by user sets and sampling transactions...")
    
    # Target sizes
    target_train_size = int(args.target_size * args.train_ratio)
    target_val_size = int(args.target_size * args.val_ratio)
    target_test_size = args.target_size - target_train_size - target_val_size
    
    log_info(f"Target sizes: Train={target_train_size}, Val={target_val_size}, Test={target_test_size}, Total={args.target_size}")
    
    # Read with all required columns
    ddf_full = dd.read_parquet(
        all_files,
        columns=all_columns,
        engine='pyarrow'
    )
    
    # Create user ID indicators (map operation)
    ddf_full = ddf_full.assign(
        is_train_user=ddf_full['user_id'].isin(list(train_users_set)),
        is_val_user=ddf_full['user_id'].isin(list(val_users_set)),
        is_test_user=ddf_full['user_id'].isin(list(test_users_set))
    )
    
    # Step 3.1: Improve category and user coverage using category stratification
    log_info("Step 3.1: Stratifying by category to improve diversity...")
    
    # Create filtered dataframes with more than required rows per user set
    # This ensures we have enough data to sample and get good coverage
    train_factor = 2.0  # Sample 2x the target size initially to ensure diversity
    val_factor = 1.5
    test_factor = 1.5
    
    # Define a Dask stratified sampling function (equivalent to df.groupby(x).head(n))
    def stratified_sample(ddf, target_size, is_user_col, stratify_col='category_id', samples_per_category=10):
        # Get only the rows for this user set
        filtered_ddf = ddf[ddf[is_user_col]]
        
        # Strategy: take samples_per_category rows for each category first
        # to ensure category diversity
        
        # This is simulated with a dummy column as Dask doesn't have a direct equivalent
        # of pandas' groupby().head()
        filtered_ddf = filtered_ddf.map_partitions(
            lambda pdf: pdf.assign(
                _rank=pdf.groupby(stratify_col).cumcount()
            )
        )
        
        # Keep only the first samples_per_category rows per category
        ddf_stratified = filtered_ddf[filtered_ddf['_rank'] < samples_per_category]
        
        # Add random sample to reach target_size if needed
        ddf_stratified_count = ddf_stratified.shape[0].compute()
        if ddf_stratified_count < target_size:
            remaining_size = target_size - ddf_stratified_count
            ddf_random = filtered_ddf[filtered_ddf['_rank'] >= samples_per_category].sample(
                frac=min(1.0, remaining_size / filtered_ddf.shape[0].compute()), 
                random_state=args.seed
            )
            ddf_result = dd.concat([ddf_stratified, ddf_random])
        else:
            ddf_result = ddf_stratified
        
        return ddf_result
    
    # Apply stratified sampling to each user set
    train_ddf = stratified_sample(
        ddf_full, 
        int(target_train_size * train_factor),  # Target size with factor
        'is_train_user',
        samples_per_category=20  # Take more samples per category for training
    )
    
    val_ddf = stratified_sample(
        ddf_full, 
        int(target_val_size * val_factor),
        'is_val_user',
        samples_per_category=10
    )
    
    test_ddf = stratified_sample(
        ddf_full, 
        int(target_test_size * test_factor),
        'is_test_user',
        samples_per_category=10
    )
    
    # Step 4: Compute and finalize the datasets (move from Dask to Pandas)
    log_info("Step 4: Computing and finalizing datasets...")
    
    log_info("Computing train dataframe...")
    train_df = train_ddf.compute()
    log_info(f"Computed train dataframe: {len(train_df)} rows")
    
    log_info("Computing validation dataframe...")
    val_df = val_ddf.compute()
    log_info(f"Computed validation dataframe: {len(val_df)} rows")
    
    log_info("Computing test dataframe...")
    test_df = test_ddf.compute()
    log_info(f"Computed test dataframe: {len(test_df)} rows")
    
    # Drop temporary columns
    for df in [train_df, val_df, test_df]:
        for col in ['is_train_user', 'is_val_user', 'is_test_user', '_rank']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
    
    # Step 5: Sample to target sizes if needed
    log_info("Step 5: Sampling to target sizes...")
    
    # Ensure val_df and test_df don't exceed target sizes
    if len(val_df) > target_val_size:
        val_df = val_df.sample(n=target_val_size, random_state=args.seed)
        log_info(f"Sampled validation dataframe to {len(val_df)} rows")
    
    if len(test_df) > target_test_size:
        test_df = test_df.sample(n=target_test_size, random_state=args.seed)
        log_info(f"Sampled test dataframe to {len(test_df)} rows")
    
    # Step 6: Balance the training set by category if requested
    if args.balance_categories:
        log_info("Step 6: Balancing training set by category...")
        
        # Check how many categories are in the training set
        train_categories = train_df['category_id'].nunique()
        log_info(f"Training set has {train_categories} unique categories")
        
        # Determine max samples per category for balancing
        # (Target slightly less than the total target to account for potential small categories)
        target_train_balanced = min(target_train_size, len(train_df))
        max_per_class = int(0.95 * target_train_balanced / train_categories)
        log_info(f"Using max {max_per_class} samples per category")
        
        # Balance by undersampling majority classes
        train_df_balanced = train_df.groupby('category_id', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_per_class), random_state=args.seed)
        )
        
        # Check if we need to add more samples to reach target size
        if len(train_df_balanced) < target_train_size:
            # Find rows in train_df that are not in train_df_balanced
            missing_indices = list(set(train_df.index) - set(train_df_balanced.index))
            if missing_indices:
                remaining_df = train_df.loc[missing_indices]
                additional_samples = min(len(remaining_df), target_train_size - len(train_df_balanced))
                additional_df = remaining_df.sample(n=additional_samples, random_state=args.seed)
                train_df_balanced = pd.concat([train_df_balanced, additional_df])
        
        log_info(f"Balanced training set has {len(train_df_balanced)} rows")
        train_df = train_df_balanced
    else:
        # Just sample to target size if not balancing
        if len(train_df) > target_train_size:
            train_df = train_df.sample(n=target_train_size, random_state=args.seed)
            log_info(f"Sampled training dataframe to {len(train_df)} rows")
    
    # Step 7: Save the datasets
    log_info("Step 7: Saving datasets...")
    
    train_df.to_parquet(os.path.join(args.output_dir, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(args.output_dir, 'val.parquet'), index=False)
    test_df.to_parquet(os.path.join(args.output_dir, 'test.parquet'), index=False)
    
    # Also save a combined version for compatibility with previous code
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_parquet(os.path.join(args.output_dir, 'combined.parquet'), index=False)
    combined_df.to_csv(os.path.join(args.output_dir, 'combined.csv'), index=False)
    
    # Save metadata
    metadata = {
        'total_rows': len(combined_df),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'unique_users': {
            'total': combined_df['user_id'].nunique(),
            'train': train_df['user_id'].nunique(),
            'val': val_df['user_id'].nunique(),
            'test': test_df['user_id'].nunique()
        },
        'unique_categories': {
            'total': combined_df['category_id'].nunique(),
            'train': train_df['category_id'].nunique(),
            'val': val_df['category_id'].nunique(),
            'test': test_df['category_id'].nunique()
        },
        'processing_time_seconds': time.time() - start_time
    }
    
    # Write metadata to text file for easy viewing
    with open(os.path.join(args.output_dir, 'metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    log_info(f"Processing complete. Total time: {(time.time() - start_time)/60:.2f} minutes")
    log_info(f"Results saved to {args.output_dir}")
    
    # Return the paths to the saved files
    return {
        'train': os.path.join(args.output_dir, 'train.parquet'),
        'val': os.path.join(args.output_dir, 'val.parquet'),
        'test': os.path.join(args.output_dir, 'test.parquet'),
        'combined': os.path.join(args.output_dir, 'combined.parquet')
    }

if __name__ == "__main__":
    main() 