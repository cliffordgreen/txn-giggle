import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple
import time
import argparse
import os
from sklearn.neighbors import NearestNeighbors # Import for NN calculation

# Helper function to calculate cyclical features (can be shared/imported)
def calculate_cyclical_features(df: pd.DataFrame, col: str, max_val: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates sin/cos features for a cyclical column."""
    values = df[col].fillna(0).astype(float)
    sin_feat = np.sin(2 * np.pi * values / max_val)
    cos_feat = np.cos(2 * np.pi * values / max_val)
    return sin_feat, cos_feat

def build_hetero_graph(df: pd.DataFrame, text_cols: List[str]) -> HeteroData:
    """
    Builds the raw HeteroData object from the DataFrame.
    Features are calculated but NOT scaled here.
    Timestamps are stored as seconds since epoch.
    """
    print("--- Starting HeteroGraph Construction ---")
    start_time = time.time()
    data = HeteroData()

    # --- Pre-processing (minimal, ensure necessary cols) ---
    print("Performing minimal preprocessing (timestamps, basic fills)...")
    timestamp_col_found = None
    if 'timestamp' in df.columns:
        timestamp_col_found = 'timestamp'
    elif 'posted_date' in df.columns:
        timestamp_col_found = 'posted_date'
        df['timestamp'] = pd.to_datetime(df[timestamp_col_found], errors='coerce')
    elif 'books_create_timestamp' in df.columns:
        timestamp_col_found = 'books_create_timestamp'
        df['timestamp'] = pd.to_datetime(df[timestamp_col_found], errors='coerce')
    else:
        raise ValueError("Requires a timestamp column ('timestamp', 'posted_date', or 'books_create_timestamp')")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Handle NaT timestamps
    if df['timestamp'].isnull().any():
         print(f"[WARN] Found {df['timestamp'].isnull().sum()} invalid timestamps. Filling with median.")
         median_ts = df['timestamp'].dropna().median()
         if pd.isna(median_ts): median_ts = pd.Timestamp.now() # Fallback if all are invalid
         df['timestamp'].fillna(median_ts, inplace=True)

    # Store timestamp as seconds since epoch (float) for graph storage
    # Using a common reference point (Unix epoch)
    epoch = pd.Timestamp("1970-01-01")
    df['timestamp_epoch'] = (df['timestamp'] - epoch) // pd.Timedelta('1s')


    # Fill critical NaNs needed for features/indexing
    df['amount'] = df['amount'].fillna(0.0)
    df['merchant_name'] = df['merchant_name'].fillna('UNKNOWN_MERCHANT')
    df['category_id'] = df['category_id'].fillna(-1) # Use -1 for unknown categories
    df['user_category_id'] = df['user_category_id'].fillna(-1)
    df['user_id'] = df['user_id'].fillna('UNKNOWN_USER')

    # Calculate cyclical features once
    df['hour'] = df['timestamp'].dt.hour.fillna(0)
    df['weekday'] = df['timestamp'].dt.weekday.fillna(0)
    df['hour_sin'], df['hour_cos'] = calculate_cyclical_features(df, 'hour', 24)
    df['day_sin'], df['day_cos'] = calculate_cyclical_features(df, 'weekday', 7)

    # --- Node Mapping ---
    print("Creating node mappings...")
    tx_indices = df.index # Use original DF index
    tx_map = {idx: i for i, idx in enumerate(tx_indices)}
    num_transactions = len(tx_map)

    merchants = df['merchant_name'].unique()
    merchant_map = {name: i for i, name in enumerate(merchants)}
    num_merchants = len(merchant_map)

    # Ensure category IDs are treated consistently (factorize if needed)
    # Factorize strings or explicitly map integers to ensure 0..N-1 range
    codes_global, uniques_global = pd.factorize(df['category_id'].astype(str), sort=True)
    df['category_id_code'] = codes_global
    category_map = {code: i for i, code in enumerate(range(len(uniques_global)))} # Maps 0..N-1 to 0..N-1
    num_categories = len(uniques_global)
    print(f"Factorized 'category_id' into {num_categories} codes.")


    print(f"  Nodes: Transactions={num_transactions}, Merchants={num_merchants}, Categories={num_categories}")

    # --- Calculate Raw Node Features ---
    print("Calculating raw node features...")
    # 1. Transaction Features (Amount + Cyclical Time)
    tx_features = df[['amount', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']].values.astype(np.float32)
    data['transaction'].x = torch.from_numpy(tx_features)
    data['transaction'].num_nodes = num_transactions # Explicitly set num_nodes

    # Store labels and original index
    data['transaction'].original_index = torch.tensor(tx_indices.tolist(), dtype=torch.long)
    data['transaction'].y_global = torch.tensor(df['category_id_code'].values, dtype=torch.long) # Use factorized codes

    # Handle user category ID similarly
    codes_user, uniques_user = pd.factorize(df['user_category_id'].astype(str), sort=True)
    data['transaction'].y_user = torch.tensor(codes_user, dtype=torch.long)
    print(f"Factorized 'user_category_id' into {len(uniques_user)} codes.")


    # Store raw text for later tokenization in DataModule
    for col in text_cols:
        if col in df:
            # Store as a list associated with the 'transaction' node store
             data['transaction'][f'_raw_{col}'] = df[col].fillna('').astype(str).tolist()
        else:
             print(f"[WARN] Text column '{col}' not found in DataFrame.")
             data['transaction'][f'_raw_{col}'] = [''] * num_transactions

    # Store user IDs needed for splitting and sequences
    data['transaction'].user_id_code, _ = pd.factorize(df['user_id'], sort=True)
    data['transaction'].user_id_code = torch.tensor(data['transaction'].user_id_code, dtype=torch.long)

    # --- Store Timestamp (as float tensor) ---
    data['transaction'].timestamp = torch.tensor(df['timestamp_epoch'].values, dtype=torch.float)


    # 2. Merchant Features (Aggregations)
    agg_funcs = {'amount': ['mean', 'std', 'max', 'min', 'count', 'median']}
    merchant_stats = df.groupby('merchant_name').agg(agg_funcs)
    merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns.values] # Flatten multi-index
    merchant_stats = merchant_stats.fillna(0)
    merchant_stats = merchant_stats.reindex(merchants, fill_value=0) # Ensure order and all merchants
    data['merchant'].x = torch.from_numpy(merchant_stats.values.astype(np.float32))
    data['merchant'].num_nodes = num_merchants

    # 3. Category Features (Aggregations)
    # Use the factorized category_id_code for grouping
    category_stats = df.groupby('category_id_code').agg(agg_funcs)
    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]
    category_stats = category_stats.fillna(0)
    # Reindex using the range of factorized codes (0 to N-1)
    category_stats = category_stats.reindex(range(num_categories), fill_value=0)
    data['category'].x = torch.from_numpy(category_stats.values.astype(np.float32))
    data['category'].num_nodes = num_categories

    print("Raw node features calculated.")

    # --- Calculate Raw Edge Features and Indices ---
    print("Calculating raw edge features and indices...")
    edge_index_dict = {}
    edge_attr_dict = {}

    # 1. Transaction -> Merchant (belongs_to) - Z-score (Raw, needs scaling later)
    edge_list = []
    attr_list = []
    # Calculate stats needed for z-score (mean/std per merchant)
    merchant_mean_std = df.groupby('merchant_name')['amount'].agg(['mean', 'std']).fillna(0)
    # Ensure std is not zero for division stability later
    merchant_mean_std['std'] = merchant_mean_std['std'].replace(0, 1e-6) # Replace 0 std with small epsilon
    for idx, row in df.iterrows():
        if row['merchant_name'] != 'UNKNOWN_MERCHANT': # Exclude unknown merchant links
            tx_node_idx = tx_map[idx]
            merchant_node_idx = merchant_map[row['merchant_name']]
            edge_list.append([tx_node_idx, merchant_node_idx])
            stats = merchant_mean_std.loc[row['merchant_name']]
            # Calculate raw z-score feature components (amount, mean, std)
            attr_list.append([row['amount'], stats['mean'], stats['std']])
    if edge_list:
        edge_index_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Store raw components for later scaling
        edge_attr_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(attr_list, dtype=torch.float)

    # 2. Merchant -> Category (categorized_as) - Confidence (0-1, no scaling needed)
    edge_list = []
    attr_list = []
    # Group by merchant_name, find the mode of factorized category_id_code
    merchant_primary_category_code = df.groupby('merchant_name')['category_id_code'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1)
    # Calculate confidence based on factorized codes
    merchant_category_confidence = df.groupby('merchant_name')['category_id_code'].agg(lambda x: x.value_counts(normalize=True).max() if not x.empty else 0)
    for merchant_name, primary_cat_code in merchant_primary_category_code.items():
        # Map merchant_name to its node index
        # Map factorized category code to its node index (which is the code itself in this setup)
        if merchant_name in merchant_map and primary_cat_code in category_map: # Use category_map keys (0..N-1)
            merchant_node_idx = merchant_map[merchant_name]
            category_node_idx = category_map[primary_cat_code] # primary_cat_code is already 0..N-1
            edge_list.append([merchant_node_idx, category_node_idx])
            confidence = merchant_category_confidence.get(merchant_name, 0)
            attr_list.append([confidence]) # Confidence is already scaled (0-1)
    if edge_list:
        edge_index_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(attr_list, dtype=torch.float)


    # 3. Transaction -> Transaction (temporal) - Raw time diff (seconds), direction
    edge_list = []
    attr_list = []
    df_sorted_time = df.sort_values('timestamp') # Use original datetime timestamp for sorting
    tx_map_sorted = {idx: tx_map[idx] for idx in df_sorted_time.index} # Map sorted index to original node index

    for i in range(num_transactions):
        row_i = df_sorted_time.iloc[i]
        ts_i = row_i['timestamp']
        if pd.isna(ts_i): continue
        tx_node_i = tx_map_sorted[df_sorted_time.index[i]]

        for k in range(1, 6): # Limit window size
            j = i + k
            if j >= num_transactions: break
            row_j = df_sorted_time.iloc[j]
            ts_j = row_j['timestamp']
            if pd.isna(ts_j): continue

            time_diff_seconds = abs((ts_i - ts_j).total_seconds())
            if time_diff_seconds <= 86400 * 1: # 1 day window
                tx_node_j = tx_map_sorted[df_sorted_time.index[j]]
                edge_list.extend([[tx_node_i, tx_node_j], [tx_node_j, tx_node_i]])
                # Store raw time difference and direction bit
                attr_list.extend([[time_diff_seconds, 1.0], [time_diff_seconds, 0.0]])
    if edge_list:
        edge_index_dict[('transaction', 'temporal', 'transaction')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr_dict[('transaction', 'temporal', 'transaction')] = torch.tensor(attr_list, dtype=torch.float)

    # 4. Transaction -> Transaction (similar_amount) - Raw amount diff, ratio, direction
    edge_list = []
    attr_list = []
    raw_tx_amounts = data['transaction'].x[:, 0].numpy() # Get raw amounts

    if num_transactions > 1:
        k_neighbors = min(5, num_transactions - 1)
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='minkowski', p=1, algorithm='auto')
        nn.fit(raw_tx_amounts.reshape(-1, 1))
        distances, indices = nn.kneighbors(raw_tx_amounts.reshape(-1, 1))

        for i in range(num_transactions):
            for k in range(1, k_neighbors + 1):
                j = indices[i, k]
                dist = distances[i, k] # This is the raw amount difference (L1 norm)
                tx_node_i = i
                tx_node_j = j

                amount_i = raw_tx_amounts[i]
                amount_j = raw_tx_amounts[j]
                amount_ratio = min(amount_i, amount_j) / (max(amount_i, amount_j) + 1e-8) if max(amount_i, amount_j) > 1e-8 else 1.0

                edge_list.extend([[tx_node_i, tx_node_j], [tx_node_j, tx_node_i]])
                # Store raw distance, ratio, direction
                attr_list.extend([[dist, amount_ratio, 1.0], [dist, amount_ratio, 0.0]])

    if edge_list:
        edge_index_dict[('transaction', 'similar_amount', 'transaction')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr_dict[('transaction', 'similar_amount', 'transaction')] = torch.tensor(attr_list, dtype=torch.float)

    print("Raw edge features and indices calculated.")

    # --- Assign final edge data ---
    print("Assigning final edge data to graph object...")
    for edge_type, index_tensor in edge_index_dict.items():
        data[edge_type].edge_index = index_tensor
        if edge_type in edge_attr_dict:
            data[edge_type].edge_attr = edge_attr_dict[edge_type]

    print(f"--- HeteroGraph Construction Finished in {time.time() - start_time:.2f}s ---")
    print("Graph Summary:")
    print(data)
    # Example validation check
    if ('transaction', 'belongs_to', 'merchant') in data.edge_types:
         max_tx_idx = data['transaction'].num_nodes - 1
         max_merch_idx = data['merchant'].num_nodes - 1
         if data['transaction','belongs_to','merchant'].edge_index.numel() > 0: # Check if edges exist
             assert data['transaction','belongs_to','merchant'].edge_index[0].max() <= max_tx_idx, f"Max tx index {data['transaction','belongs_to','merchant'].edge_index[0].max()} > {max_tx_idx}"
             assert data['transaction','belongs_to','merchant'].edge_index[1].max() <= max_merch_idx, f"Max merch index {data['transaction','belongs_to','merchant'].edge_index[1].max()} > {max_merch_idx}"
             print("Edge index validation check passed.")
         else:
             print("No ('transaction', 'belongs_to', 'merchant') edges found, skipping validation check.")

    return data

def main():
    parser = argparse.ArgumentParser(description="Build HeteroData graph from transaction data.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input transaction CSV file (e.g., combined_5.csv).')
    parser.add_argument('--output_path', type=str, default='config/graph_data.pt', help='Path to save the output HeteroData object.')
    # Add relevant text columns from your dataset
    parser.add_argument('--text_cols', nargs='+', default=['raw_description', 'memo', 'merchant_name'], help='List of raw text columns to include.')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir: # Check if directory part is not empty
        os.makedirs(output_dir, exist_ok=True)


    print(f"Loading data from: {args.data_path}")
    try:
        # Adjust dtype warning handling or specify dtypes if known
        df = pd.read_csv(args.data_path, low_memory=False)
        print(f"Loaded DataFrame with shape: {df.shape}")
    except FileNotFoundError:
        print(f"[ERROR] Data file not found at {args.data_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # Build the graph
    graph_data = build_hetero_graph(df, args.text_cols)

    # Save the graph data
    print(f"Saving graph data object to: {args.output_path}")
    try:
        torch.save(graph_data, args.output_path)
        print("Graph data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save graph data: {e}")

if __name__ == '__main__':
    main()
