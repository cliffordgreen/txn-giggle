import argparse
import pandas as pd
import torch
import os

# Import the V2 DataModule capable of building the graph
from data.data_module_v2 import TransactionDataModuleV2

def main(args):
    print(f"--- Generating Graph Metadata from: {args.data_path} ---")

    # --- 1. Load Data ---
    print(f"Loading transaction data...")
    try:
        # Load with low_memory=False to handle potential DtypeWarnings if necessary
        transactions_df = pd.read_csv(args.data_path, low_memory=False)
        print(f"Loaded DataFrame with shape: {transactions_df.shape}")
        # Basic check for user_id
        if args.user_id_col not in transactions_df.columns:
             raise ValueError(f"User ID column '{args.user_id_col}' not found in data.")
        # Add checks for other essential columns if needed by DataModuleV2
    except FileNotFoundError:
        print(f"[ERROR] Data file not found at {args.data_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # --- 2. Initialize DataModule V2 (configured to build graph) ---
    print("Initializing TransactionDataModuleV2...")
    # Minimal config needed for graph building - adjust if your V2 requires more
    data_module = TransactionDataModuleV2(
        transactions_df=transactions_df,
        batch_size=args.batch_size, # Needed but value doesn't matter much here
        num_workers=0,
        num_hgt_layers=args.num_hgt_layers, # Needed for default sample calculation if not provided
        # Ensure graph building is enabled
        use_gnn_encoder=True,
        use_sequence_encoder=False, # Disable others to speed up setup
        use_text_encoder=False,
        val_ratio=0.01, # Minimal split just for setup logic
        test_ratio=0.01
    )

    # --- 3. Run Setup to Build Graph ---
    print("Running DataModuleV2 setup to build graph and extract metadata...")
    try:
        data_module.setup(stage='fit') # 'fit' stage typically triggers graph building
        print("DataModule setup complete.")
    except Exception as e:
        print(f"[ERROR] Failed during DataModuleV2 setup: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Extract Metadata ---
    if not hasattr(data_module, 'full_graph_data') or data_module.full_graph_data is None:
        print("[ERROR] DataModuleV2 did not build the graph ('full_graph_data' not found). Cannot extract metadata.")
        return

    try:
        metadata = data_module.full_graph_data.metadata()
        print(f"Successfully extracted metadata: {metadata}")
        if hasattr(data_module, 'node_feature_dims') and data_module.node_feature_dims:
             node_dims = data_module.node_feature_dims
             print(f"Successfully extracted node_feature_dims: {node_dims}")
        else:
             print("[ERROR] Could not extract 'node_feature_dims' from DataModuleV2.")
             return
    except Exception as e:
        print(f"[ERROR] Failed to extract metadata or node_dims from graph: {e}")
        return

    # --- 5. Save Full Graph Data ---
    output_path = args.output_path
    print(f"Saving full graph data object to: {output_path}")
    # Save the entire HeteroData object
    data_to_save = data_module.full_graph_data
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(data_to_save, output_path)
        print("Full graph data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save graph data to {output_path}: {e}")

    print("--- Metadata Generation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and save graph metadata for HGT.')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the transaction data CSV file.')
    parser.add_argument('--output_path', type=str, default='config/graph_metadata.pt', help='Path to save the generated metadata file.')
    parser.add_argument('--user_id_col', type=str, default='user_id', help='Column name for user IDs.')
    parser.add_argument('--num_hgt_layers', type=int, default=2, help='Number of HGT layers (needed for DataModuleV2 init).')
    parser.add_argument('--batch_size', type=int, default=32, help='Temporary batch size for DataModuleV2 init.')
    # Add any other arguments absolutely required by TransactionDataModuleV2 __init__ or setup

    args = parser.parse_args()
    main(args) 