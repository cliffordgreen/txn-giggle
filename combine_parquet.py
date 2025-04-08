import pandas as pd
import glob
import os

# Define the directory containing the parquet files
parquet_dir = 'Test_1k'
output_csv = 'combined_1k.csv'
num_files_to_combine = 5

try:
    # Find parquet files
    files = sorted(glob.glob(os.path.join(parquet_dir, 'transaction_data_*.parquet')))

    if not files:
        print(f"Error: No parquet files found matching 'transaction_data_*.parquet' in '{parquet_dir}'")
        exit(1)

    # Select the first N files
    files_to_process = files[:num_files_to_combine]
    print(f"Found {len(files)} files in '{parquet_dir}'. Processing the first {len(files_to_process)}...")

    # Read and concatenate
    df_list = []
    for f in files_to_process:
        print(f"  Reading {f}...")
        try:
            df_list.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  Error reading {f}: {e}")
            exit(1)

    if not df_list:
        print("Error: No dataframes were successfully read.")
        exit(1)

    print("Concatenating dataframes...")
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save to CSV
    print(f"Saving combined dataframe with {len(combined_df)} rows to '{output_csv}'...")
    combined_df.to_csv(output_csv, index=False)

    print(f"Successfully combined {len(files_to_process)} files from '{parquet_dir}' into '{output_csv}'.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1) 