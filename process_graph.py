import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import time
import argparse
import os

# --- Helper Functions (Adapted from refactored DataModule) ---

def _split_data_and_add_masks(graph_data: HeteroData, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """Creates train/val/test masks based on user ID codes stored in the graph."""
    print("Creating train/val/test masks...")
    if graph_data is None or 'user_id_code' not in graph_data['transaction']:
        raise ValueError("Graph data or 'user_id_code' not found in graph['transaction'].")

    node_user_codes = graph_data['transaction'].user_id_code.numpy()
    num_transactions = graph_data['transaction'].num_nodes
    unique_user_codes = np.unique(node_user_codes)
    np.random.seed(seed) # Ensure consistent split
    np.random.shuffle(unique_user_codes)

    n_users = len(unique_user_codes)
    n_val = int(n_users * val_ratio)
    n_test = int(n_users * test_ratio)

    # Adjust counts similar to original logic
    if val_ratio > 0 and n_val == 0 and n_users > 1: n_val = 1
    if test_ratio > 0 and n_test == 0 and n_users > (1 + n_val): n_test = 1
    if n_val + n_test >= n_users and n_users > 0:
        n_test = max(0, n_users - n_val) # Prioritize val
        n_train = 0
    else:
        n_train = n_users - n_val - n_test

    print(f"Splitting {n_users} user codes into: Train={n_train}, Val={n_val}, Test={n_test}")

    val_user_codes = set(unique_user_codes[:n_val])
    test_user_codes = set(unique_user_codes[n_val : n_val + n_test])
    train_user_codes = set(unique_user_codes[n_val + n_test :]) if n_train > 0 else set()

    train_mask = torch.zeros(num_transactions, dtype=torch.bool)
    val_mask = torch.zeros(num_transactions, dtype=torch.bool)
    test_mask = torch.zeros(num_transactions, dtype=torch.bool)

    for i in range(num_transactions):
        user_code = node_user_codes[i]
        if user_code in train_user_codes: train_mask[i] = True
        elif user_code in val_user_codes: val_mask[i] = True
        elif user_code in test_user_codes: test_mask[i] = True

    graph_data['transaction'].train_mask = train_mask
    graph_data['transaction'].val_mask = val_mask
    graph_data['transaction'].test_mask = test_mask
    print("Masks added to graph_data['transaction'].")
    return graph_data # Return modified graph

def _fit_train_scalers(graph_data: HeteroData) -> Tuple[Dict[str, StandardScaler], Dict[str, StandardScaler]]:
    """Fits StandardScaler instances using ONLY the training data split."""
    if 'train_mask' not in graph_data['transaction']:
        raise ValueError("train_mask not available in graph data.")

    train_mask = graph_data['transaction'].train_mask
    if train_mask.sum() == 0:
        print("[WARN] No training nodes found. Returning empty scalers.")
        return {}, {}

    print(f"Fitting scalers on {train_mask.sum()} training nodes/edges...")
    node_scalers = {}
    edge_scalers = {}

    # --- Fit Node Scalers ---
    node_type = 'transaction'
    if 'x' in graph_data[node_type] and graph_data[node_type].x.numel() > 0:
        features = graph_data[node_type].x[train_mask]
        if features.shape[0] > 0 : # Ensure features exist for train nodes
            scaler = StandardScaler()
            # Assuming amount is the first column, scale only that
            if features.shape[1] > 0:
                 scaler.fit(features[:, 0:1].numpy()) # Fit on amount column
                 node_scalers[node_type] = scaler
                 print(f"  Fitted scaler for '{node_type}' (col 0). Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")
            else: print(f"  [WARN] No feature columns for '{node_type}'.")
        else: print(f"  [WARN] No training features found for '{node_type}'.")

    # --- Fit Edge Scalers ---
    # Fit scaler for ('transaction', 'belongs_to', 'merchant') z-score components
    edge_type_z = ('transaction', 'belongs_to', 'merchant')
    if edge_type_z in graph_data.edge_types and 'edge_attr' in graph_data[edge_type_z]:
        edge_index = graph_data[edge_type_z].edge_index
        src_nodes = edge_index[0]
        edge_train_mask = train_mask[src_nodes]
        if edge_train_mask.sum() > 0:
             train_edge_attrs = graph_data[edge_type_z].edge_attr[edge_train_mask]
             if train_edge_attrs.shape[1] == 3: # Check if components are stored
                 scaler = StandardScaler()
                 scaler.fit(train_edge_attrs.numpy())
                 edge_scalers['z_score_components'] = scaler
                 print(f"  Fitted scaler for '{edge_type_z}' components. Mean: {scaler.mean_}, Scale: {scaler.scale_}")
             else: print(f"  [WARN] Incorrect number of attributes for '{edge_type_z}' edge scaler.")
        else: print(f"  [WARN] No training edges found for '{edge_type_z}'.")

    # Fit scaler for ('transaction', 'temporal', 'transaction') raw time diff
    edge_type_t = ('transaction', 'temporal', 'transaction')
    if edge_type_t in graph_data.edge_types and 'edge_attr' in graph_data[edge_type_t]:
         edge_index = graph_data[edge_type_t].edge_index
         src_nodes = edge_index[0]
         edge_train_mask = train_mask[src_nodes]
         if edge_train_mask.sum() > 0:
              train_edge_attrs = graph_data[edge_type_t].edge_attr[edge_train_mask]
              if train_edge_attrs.shape[1] >= 1: # Check if time diff is present
                  scaler = StandardScaler()
                  scaler.fit(train_edge_attrs[:, 0:1].numpy())
                  edge_scalers['temporal_diff'] = scaler
                  print(f"  Fitted scaler for '{edge_type_t}' (time diff). Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")
              else: print(f"  [WARN] Not enough attributes for '{edge_type_t}' edge scaler.")
         else: print(f"  [WARN] No training edges found for '{edge_type_t}'.")

    # Fit scaler for ('transaction', 'similar_amount', 'transaction') raw amount diff
    edge_type_a = ('transaction', 'similar_amount', 'transaction')
    if edge_type_a in graph_data.edge_types and 'edge_attr' in graph_data[edge_type_a]:
         edge_index = graph_data[edge_type_a].edge_index
         src_nodes = edge_index[0]
         edge_train_mask = train_mask[src_nodes]
         if edge_train_mask.sum() > 0:
              train_edge_attrs = graph_data[edge_type_a].edge_attr[edge_train_mask]
              if train_edge_attrs.shape[1] >= 1: # Check if amount diff is present
                  scaler = StandardScaler()
                  scaler.fit(train_edge_attrs[:, 0:1].numpy())
                  edge_scalers['amount_diff'] = scaler
                  print(f"  Fitted scaler for '{edge_type_a}' (amount diff). Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")
              else: print(f"  [WARN] Not enough attributes for '{edge_type_a}' edge scaler.")
         else: print(f"  [WARN] No training edges found for '{edge_type_a}'.")

    return node_scalers, edge_scalers


def _apply_scalers(graph_data: HeteroData, node_scalers: Dict, edge_scalers: Dict):
    """Applies the fitted scalers to the node and edge features of the entire graph."""
    epsilon = 1e-8 # For numerical stability
    print("Applying fitted scalers to graph features...")

    # --- Apply Node Scalers ---
    node_type = 'transaction'
    if node_type in node_scalers and 'x' in graph_data[node_type]:
        scaler = node_scalers[node_type]
        features = graph_data[node_type].x # Operate directly on tensor
        # Scale only the first column (amount)
        scaled_amount = (features[:, 0:1] - torch.from_numpy(scaler.mean_.astype(np.float32))) / (torch.from_numpy(scaler.scale_.astype(np.float32)) + epsilon)
        # Combine with non-scaled features (cyclical time)
        graph_data[node_type].x = torch.cat([
            scaled_amount,
            features[:, 1:]
        ], dim=1)
        print(f"  Applied scaler to '{node_type}' node features (col 0).")

    # --- Apply Edge Scalers ---
    # Apply scaler to ('transaction', 'belongs_to', 'merchant') z-score components
    edge_type_z = ('transaction', 'belongs_to', 'merchant')
    if edge_type_z in graph_data.edge_types and 'z_score_components' in edge_scalers:
        # This edge type doesn't need scaling applied here, the final z-score is calculated from raw components
        scaler = edge_scalers['z_score_components'] # We still need the fitted scaler info potentially, but don't transform edge_attr here.
        raw_attrs = graph_data[edge_type_z].edge_attr
        if raw_attrs.shape[1] == 3: # Check if components are present
            # Calculate final z-score using raw components
            amount, mean, std = raw_attrs[:, 0], raw_attrs[:, 1], raw_attrs[:, 2]
            final_z_score = (amount - mean) / (std + epsilon)
            graph_data[edge_type_z].edge_attr = final_z_score.unsqueeze(-1) # Add feature dim
            print(f"  Calculated final z-score for '{edge_type_z}' edges (using raw components).")
        else: print(f"  [WARN] Skipping z-score calculation for '{edge_type_z}' due to unexpected attribute shape.")

    # Apply scaler to ('transaction', 'temporal', 'transaction') time diff
    edge_type_t = ('transaction', 'temporal', 'transaction')
    if edge_type_t in graph_data.edge_types and 'temporal_diff' in edge_scalers:
         scaler = edge_scalers['temporal_diff']
         raw_attrs = graph_data[edge_type_t].edge_attr
         if raw_attrs.shape[1] >= 1:
             time_diff_scaled = (raw_attrs[:, 0:1] - torch.from_numpy(scaler.mean_.astype(np.float32))) / (torch.from_numpy(scaler.scale_.astype(np.float32)) + epsilon)
             # Combine scaled diff with original direction bit
             graph_data[edge_type_t].edge_attr = torch.cat([
                 time_diff_scaled,
                 raw_attrs[:, 1:2] # Direction bit
             ], dim=1)
             print(f"  Applied scaler to '{edge_type_t}' edge features (time diff).")
         else: print(f"  [WARN] Skipping scaling for '{edge_type_t}' due to unexpected attribute shape.")


    # Apply scaler to ('transaction', 'similar_amount', 'transaction') amount diff
    edge_type_a = ('transaction', 'similar_amount', 'transaction')
    if edge_type_a in graph_data.edge_types and 'amount_diff' in edge_scalers:
         scaler = edge_scalers['amount_diff']
         raw_attrs = graph_data[edge_type_a].edge_attr
         if raw_attrs.shape[1] >= 1:
             amount_diff_scaled = (raw_attrs[:, 0:1] - torch.from_numpy(scaler.mean_.astype(np.float32))) / (torch.from_numpy(scaler.scale_.astype(np.float32)) + epsilon)
             # Combine scaled diff with ratio and direction
             graph_data[edge_type_a].edge_attr = torch.cat([
                 amount_diff_scaled,
                 raw_attrs[:, 1:] # Ratio and direction
             ], dim=1)
             print(f"  Applied scaler to '{edge_type_a}' edge features (amount diff).")
         else: print(f"  [WARN] Skipping scaling for '{edge_type_a}' due to unexpected attribute shape.")

    return graph_data # Return modified graph


def _prepare_and_add_sequences(graph_data: HeteroData, max_seq_length: int):
    """Prepare sequence data using SCALED features from the graph."""
    print("Preparing sequence features...")
    if 'x' not in graph_data['transaction']:
        print("[WARN] Scaled transaction features 'x' not found. Skipping sequence preparation.")
        return graph_data
    # --- Check for Timestamp ---
    if 'timestamp' not in graph_data['transaction']:
         # Attempt to load from original DataFrame if index available
         if 'original_index' in graph_data['transaction']:
              print("[INFO] 'timestamp' not on graph, attempting fallback load from raw data using original_index (requires raw data access).")
              # THIS REQUIRES ACCESS TO THE ORIGINAL DATAFRAME - MAJOR CHANGE NEEDED
              # For now, add a placeholder or raise error. Adding placeholder:
              print("[WARN] Cannot load original timestamps. Sequence time deltas will be inaccurate (using zeros).")
              timestamps_tensor = torch.zeros(graph_data['transaction'].num_nodes, dtype=torch.long) # Dummy tensor
         else:
              print("[WARN] 'timestamp' and 'original_index' not found on transaction nodes. Skipping sequence preparation.")
              return graph_data
    else:
         # Ensure timestamp is numeric (e.g., seconds since epoch) before converting
         if not torch.is_tensor(graph_data['transaction'].timestamp) or not graph_data['transaction'].timestamp.is_floating_point():
             if torch.is_tensor(graph_data['transaction'].timestamp) and graph_data['transaction'].timestamp.dtype == torch.int64:
                 print("[INFO] Converting int64 timestamp tensor to float for processing.")
                 timestamps_tensor = graph_data['transaction'].timestamp.float() # Convert to float if int
             else:
                 print("[WARN] Timestamp format unexpected. Sequence time deltas may be inaccurate.")
                 # Fallback: use zeros if cannot process
                 timestamps_tensor = torch.zeros(graph_data['transaction'].num_nodes, dtype=torch.float)
         else:
             timestamps_tensor = graph_data['transaction'].timestamp # Assumes it's already float seconds


    if 'user_id_code' not in graph_data['transaction']:
        print("[WARN] 'user_id_code' attribute not found on transaction nodes. Skipping sequence preparation.")
        return graph_data

    num_transactions = graph_data['transaction'].num_nodes
    user_codes = graph_data['transaction'].user_id_code.numpy()
    scaled_node_features = graph_data['transaction'].x.numpy() # Use already scaled features

    # Infer sequence dim from node features + 1 (for time delta)
    expected_seq_dim = scaled_node_features.shape[1] + 1

    all_seq_features = []
    all_seq_lengths = []

    # Create DataFrame view for easier sorting/grouping
    temp_df_data = {
         'node_idx': np.arange(num_transactions),
         'user_code': user_codes,
         'timestamp': pd.to_datetime(timestamps_tensor.numpy(), errors='coerce', unit='s'), # Convert tensor (assuming seconds since epoch)
         # Include all scaled node features
         **{f'feat_{i}': scaled_node_features[:, i] for i in range(scaled_node_features.shape[1])}
    }
    temp_df = pd.DataFrame(temp_df_data)
    # Check for NaT timestamps after conversion
    if temp_df['timestamp'].isnull().any():
        print(f"[WARN] {temp_df['timestamp'].isnull().sum()} timestamps became NaT after conversion. Filling with median.")
        median_ts = temp_df['timestamp'].median()
        if pd.isna(median_ts): median_ts = pd.Timestamp.now() # Fallback if all are NaT
        temp_df['timestamp'].fillna(median_ts, inplace=True)

    temp_df.sort_values(['user_code', 'timestamp'], inplace=True)
    user_groups = temp_df.groupby('user_code')
    node_idx_map = {node_idx: pos for pos, node_idx in enumerate(temp_df['node_idx'])}

    # --- Fit Time Delta Scaler on Train Split---
    seq_time_scaler = StandardScaler()
    all_time_deltas_train = []
    if 'train_mask' in graph_data['transaction']:
        train_mask_np = graph_data['transaction'].train_mask.numpy()
        print("  Calculating training time deltas for sequence scaler...")
        for user_code, group in user_groups:
            user_node_indices = group['node_idx'].values
            if np.any(train_mask_np[user_node_indices]):
                time_diffs = group['timestamp'].diff().dt.total_seconds().fillna(0).clip(lower=0)
                current_node_is_train = train_mask_np[group['node_idx'].values]
                all_time_deltas_train.extend(time_diffs[current_node_is_train].tolist())

        if all_time_deltas_train:
            seq_time_scaler.fit(np.array(all_time_deltas_train).reshape(-1, 1))
            print(f"  Fitted sequence time_delta scaler on train data. Mean: {seq_time_scaler.mean_[0]:.4f}, Scale: {seq_time_scaler.scale_[0]:.4f}")
        else:
            print("[WARN] No training time deltas found for sequence scaler. Using dummy (0 mean, 1 scale).")
            seq_time_scaler.mean_ = np.array([0.0])
            seq_time_scaler.scale_ = np.array([1.0])
    else:
        print("[WARN] Train mask not found. Using dummy time delta scaler.")
        seq_time_scaler.mean_ = np.array([0.0])
        seq_time_scaler.scale_ = np.array([1.0])


    print("  Generating sequences for each transaction...")
    feature_cols = [f'feat_{i}' for i in range(scaled_node_features.shape[1])]
    epsilon = 1e-8

    # Iterate through nodes in their original 0..N-1 order
    for node_idx in range(num_transactions):
        sorted_pos = node_idx_map.get(node_idx)
        if sorted_pos is None: continue

        current_row = temp_df.iloc[sorted_pos]
        user_code = current_row['user_code']
        current_time = current_row['timestamp']

        start_idx_in_sorted = max(0, sorted_pos - max_seq_length)
        prev_txs_df_sorted = temp_df.iloc[start_idx_in_sorted:sorted_pos]
        prev_txs_group = prev_txs_df_sorted[prev_txs_df_sorted['user_code'] == user_code]

        seq_features_for_tx = np.array([]) # Initialize as empty numpy array
        if not prev_txs_group.empty:
            # Calculate time deltas relative to current transaction
            time_deltas = (current_time - prev_txs_group['timestamp']).dt.total_seconds().fillna(0).clip(lower=0)
            # Scale time deltas
            time_deltas_scaled = (time_deltas.values.reshape(-1, 1) - seq_time_scaler.mean_) / (seq_time_scaler.scale_ + epsilon)
            # Get existing scaled features
            prev_features = prev_txs_group[feature_cols].values
            # Combine existing features with scaled time delta
            seq_features_for_tx = np.column_stack([prev_features, time_deltas_scaled])


        if seq_features_for_tx.size > 0:
            seq_tensor = torch.from_numpy(seq_features_for_tx.astype(np.float32))
            seq_len = len(seq_tensor)
        else:
            seq_tensor = torch.zeros((0, expected_seq_dim), dtype=torch.float)
            seq_len = 0

        all_seq_features.append(seq_tensor)
        all_seq_lengths.append(seq_len)
    # --- End Loop ---

    num_zero_length = sum(1 for length in all_seq_lengths if length == 0)
    total_sequences = len(all_seq_lengths)
    if total_sequences > 0: print(f"  [INFO] Generated {num_zero_length} zero-length sequences out of {total_sequences} ({num_zero_length/total_sequences:.2%}).")

    if not all_seq_features:
        print("[WARN] No sequence features generated.")
        padded_sequences = torch.empty((num_transactions, 0, expected_seq_dim), dtype=torch.float)
    else:
        padded_sequences = pad_sequence(all_seq_features, batch_first=True, padding_value=0.0)

    graph_data['transaction'].seq_features = padded_sequences
    graph_data['transaction'].seq_lengths = torch.tensor(all_seq_lengths, dtype=torch.long)
    print(f"Added sequence features to graph. Padded shape: {padded_sequences.shape}")
    return graph_data


def _prepare_and_add_text(graph_data: HeteroData, tokenizer_name: str, text_max_length: int):
    """Tokenizes raw text stored on the graph nodes."""
    print(f"Tokenizing text fields using tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer '{tokenizer_name}': {e}")
        return graph_data # Return unmodified graph

    start_time_text = time.time()

    text_keys_found = [k for k in graph_data['transaction'].keys() if k.startswith('_raw_')]
    if not text_keys_found:
         print("[WARN] No raw text fields (starting with '_raw_') found. Skipping tokenization.")
         return graph_data

    processed_tokens = {}
    num_transactions = graph_data['transaction'].num_nodes

    for raw_key in text_keys_found:
        field_name = raw_key.replace('_raw_', '')
        if raw_key in graph_data['transaction']:
            texts = graph_data['transaction'][raw_key]
            if not isinstance(texts, list):
                 print(f"[WARN] Skipping text field '{field_name}' - expected a list, found {type(texts)}.")
                 continue
            if len(texts) != num_transactions:
                 print(f"[WARN] Length mismatch for text field '{field_name}' ({len(texts)}) vs num_transactions ({num_transactions}). Skipping.")
                 continue

            print(f"  Tokenizing field: '{field_name}' ({len(texts)} items)...")
            texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]

            # Batch tokenization
            batch_size_tok = 10000
            all_input_ids = []
            all_attention_masks = []
            for i in range(0, num_transactions, batch_size_tok):
                batch_texts = texts[i : i + batch_size_tok]
                tokens = tokenizer(
                    batch_texts, padding='max_length', truncation=True,
                    max_length=text_max_length, return_tensors='pt'
                )
                all_input_ids.append(tokens['input_ids'])
                all_attention_masks.append(tokens['attention_mask'])
                # if (i // batch_size_tok) % 10 == 0: print(f"    Tokenized {i + len(batch_texts)} / {num_transactions}")

            processed_tokens[f'{field_name}_input_ids'] = torch.cat(all_input_ids, dim=0)
            processed_tokens[f'{field_name}_attention_mask'] = torch.cat(all_attention_masks, dim=0)
            print(f"  Finished tokenizing '{field_name}'. Shape: {processed_tokens[f'{field_name}_input_ids'].shape}")

            # --- Remove raw text field after tokenization ---
            try:
                del graph_data['transaction'][raw_key]
                print(f"  Removed raw text field '{raw_key}'.")
            except KeyError:
                print(f"  [WARN] Could not remove raw text field '{raw_key}'.")
            # --------------------------------------------

    # Add tokenized tensors back to the graph data
    for key, tensor in processed_tokens.items():
        graph_data['transaction'][key] = tensor

    print(f"Added tokenized text features to graph in {time.time() - start_time_text:.2f}s.")
    return graph_data


# --- Main Processing Function ---
def process_graph_data(raw_graph_path: str,
                       output_path: str,
                       val_ratio: float,
                       test_ratio: float,
                       seed: int,
                       max_seq_length: int,
                       tokenizer_name: str,
                       text_max_length: int,
                       prepare_sequences: bool,
                       prepare_text: bool):
    """Loads raw graph, processes it, and saves the result."""
    print(f"--- Starting Graph Processing ---")
    start_time_total = time.time()

    # 1. Load Raw Graph
    print(f"Loading raw graph from: {raw_graph_path}")
    if not os.path.exists(raw_graph_path):
        raise FileNotFoundError(f"Raw graph file not found: {raw_graph_path}")
    try:
        graph_data = torch.load(raw_graph_path)
        if not isinstance(graph_data, HeteroData):
             raise TypeError("Loaded file is not a HeteroData object.")
        print("Raw graph loaded successfully.")
        print(f"Initial Graph: {graph_data}")
    except Exception as e:
        print(f"[ERROR] Failed to load raw graph: {e}")
        raise

    # 2. Split Data (Add Masks)
    graph_data = _split_data_and_add_masks(graph_data, val_ratio, test_ratio, seed)

    # 3. Fit Scalers (on Train Split)
    node_scalers, edge_scalers = _fit_train_scalers(graph_data)

    # 4. Apply Scalers (to Full Graph)
    graph_data = _apply_scalers(graph_data, node_scalers, edge_scalers)

    # 5. Prepare Sequences (if enabled)
    if prepare_sequences:
        graph_data = _prepare_and_add_sequences(graph_data, max_seq_length)
    else:
        print("[INFO] Skipping sequence preparation as per arguments.")

    # 6. Prepare Text (if enabled)
    if prepare_text:
        graph_data = _prepare_and_add_text(graph_data, tokenizer_name, text_max_length)
    else:
        print("[INFO] Skipping text preparation as per arguments.")

    # 7. Final Validation (Optional)
    print("\n--- Final Processed Graph Summary ---")
    print(graph_data)
    # Example check: Ensure seq_features exist if sequences were prepared
    if prepare_sequences and 'seq_features' not in graph_data['transaction']:
         print("[WARN] Sequence features missing after sequence preparation step!")
    # Example check: Ensure tokenized text exists if text was prepared
    if prepare_text and not any(k.endswith('_input_ids') for k in graph_data['transaction'].keys()):
         print("[WARN] Tokenized text features missing after text preparation step!")


    # 8. Save Processed Graph
    print(f"\nSaving processed graph data object to: {output_path}")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Check if directory part is not empty
            os.makedirs(output_dir, exist_ok=True)
        torch.save(graph_data, output_path)
        print("Processed graph data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save processed graph data: {e}")
        raise

    print(f"--- Graph Processing Finished in {time.time() - start_time_total:.2f}s ---")


# --- Script Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process raw HeteroData graph: add masks, scale features, prepare sequences/text.")
    parser.add_argument('--raw_graph_path', type=str, default='config/graph_data.pt', help='Path to the raw graph data file (.pt) created by build_graph.py.')
    parser.add_argument('--output_path', type=str, default='config/processed_graph_data.pt', help='Path to save the processed graph data file (.pt).')
    # Split params
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Fraction of users for validation split.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Fraction of users for test split.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting.')
    # Sequence params
    parser.add_argument('--max_seq_length', type=int, default=50, help='Max sequence length for historical transactions.')
    parser.add_argument('--prepare_sequences', action='store_true', help='Flag to enable sequence preparation.')
    # Text params
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='HuggingFace tokenizer name.')
    parser.add_argument('--text_max_length', type=int, default=128, help='Max length for text tokenization.')
    parser.add_argument('--prepare_text', action='store_true', help='Flag to enable text tokenization.')

    args = parser.parse_args()

    process_graph_data(
        raw_graph_path=args.raw_graph_path,
        output_path=args.output_path,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        tokenizer_name=args.tokenizer_name,
        text_max_length=args.text_max_length,
        prepare_sequences=args.prepare_sequences,
        prepare_text=args.prepare_text
    )
