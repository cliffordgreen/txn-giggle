#train.py
import os
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.data import HeteroData

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models.transaction_classifier import TransactionClassifier, ModelConfig
from data.data_module import TransactionDataModule
from typing import Dict, Optional
import torch
import sys

# print("Script Started")

torch.set_float32_matmul_precision('medium')  # Use 'medium' if you encounter numerical instability

# Enable CUDA benchmarking to optimize kernels for your specific model
torch.backends.cudnn.benchmark = True

# If your model architecture is static (fixed input sizes)
torch.backends.cudnn.deterministic = False

def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess transaction data."""
    # Load data with low_memory=False to avoid dtype warnings
    df = pd.read_csv(data_path, low_memory=False)
    #df = df.head(400)
    # Step 1: Convert timestamps with error handling
    try:
        # Convert to datetime with lenient parsing
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Check for NaT values after conversion
        nat_count = df['timestamp'].isna().sum()
        if nat_count > 0:
            print(f"Warning: {nat_count} rows have invalid timestamps that couldn't be parsed")
            
            # For demonstration, show some examples of problematic values
            if 'timestamp_original' not in df.columns:
                df['timestamp_original'] = df['timestamp']
            
            # Fill NaT values with a default timestamp to avoid errors
            # Use the median date as a reasonable default
            median_date = df['timestamp'].dropna().median()
            if pd.isna(median_date):  # If all dates are NaT
                median_date = pd.Timestamp('2020-01-01')
                
            print(f"Filling NaT timestamps with {median_date}")
            df['timestamp'] = df['timestamp'].fillna(median_date)
            
    except Exception as e:
        print(f"Error during timestamp conversion: {e}")
        # Fallback to a default date if conversion totally fails
        print("Using default timestamp for all rows")
        df['timestamp'] = pd.Timestamp('2020-01-01')
    
    # Step 2: Safely extract temporal features
    # - Only extract features from valid timestamps
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour
    
    # Step 3: For the timestamp() method specifically
    # - Create a safe version that handles NaT values
    def safe_timestamp(ts):
        try:
            if pd.isna(ts):
                return None  # or a default value like 0
            return ts.timestamp()
        except (ValueError, AttributeError):
            return None  # or a default value
    
    # Apply the safe function if you need Unix timestamps
    df['unix_timestamp'] = df['timestamp'].apply(safe_timestamp)
    
    # Handle missing values in other columns
    df['raw_description'] = df['raw_description'].fillna('')
    df['memo'] = df['memo'].fillna('')
    df['merchant_name'] = df['merchant_name'].fillna('')

    print(f"Data loaded: {len(df)} records")
    # print("Data Loaded")
    return df
    
def train(
    data_path: str,
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    warmup_steps: int = 50,
    max_seq_length: int = 50,
    graph_neighbors: Optional[Dict[str, int]] = None,
    text_model_name: str = 'bert-base-uncased',
    text_max_length: int = 128,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    use_sequence_encoder: bool = True,
    use_text_encoder: bool = True,
    use_gnn_encoder: bool = True,
    gnn_only_test_mode: bool = False
):
    """Train the transaction classifier."""
    # print("Train function started")
    pl.seed_everything(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # print("Creating DataModule")
    # Create data module
    data_module = TransactionDataModule(
        transactions_df=df,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_length=max_seq_length,
        num_neighbors=[15,10],
        #graph_neighbors=graph_neighbors,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        perform_overfit_test=False
    )
    # print("Running DataModule setup")
    data_module.setup('fit')    
    node_dims = data_module.node_feature_dims
    # Get sequence dimension AFTER setup
    sequence_dim = data_module.sequence_feature_dim 
    if sequence_dim is None:
        raise ValueError("DataModule sequence_feature_dim is None after setup.")
    # Get edge dimensions AFTER setup
    edge_dims = data_module.edge_feature_dims
    # Get the full graph data object AFTER setup
    full_graph_data = data_module.graph_data
    if full_graph_data is None:
        raise ValueError("DataModule graph_data is None after setup.")
    # print("DataModule setup complete")

    try:
        # Get node types from the keys of the calculated feature dimensions
        node_types_list = list(node_dims.keys())
        # Get edge types from the fully constructed graph_data object
        if data_module.graph_data is not None:
             # Ensure graph_data has edge_types populated correctly
             if hasattr(data_module.graph_data, 'edge_types'):
                  edge_types_list = list(data_module.graph_data.edge_types)
             else:
                  # Fallback: try getting keys from edge_index_dict if edge_types attr missing
                  edge_types_list = list(data_module.graph_data.edge_index_dict.keys())
    
             # Ensure edge_types_list contains tuples, not just strings if keys were used directly
             if not all(isinstance(et, tuple) for et in edge_types_list):
                  raise TypeError("Inferred edge types are not all tuples.")
    
        else:
             raise ValueError("DataModule graph_data is None after setup, cannot get edge types.")

        # Create the metadata tuple
        gnn_metadata = (node_types_list, edge_types_list)
        # print(f"Constructed GNN Metadata: NodeTypes={node_types_list}, EdgeTypes={edge_types_list}") # Debug print
    
    except Exception as e_meta:
         print(f"[ERROR] Failed to construct GNN metadata from DataModule: {e_meta}")
         raise e_meta # Stop execution if metadata cannot be created
    
    # print("\n--- Checking First Batch Data ---")
    # 1. Ensure setup has run (it usually runs automatically before dataloaders are needed)
    #    You might need to call it explicitly if running this outside the normal Pytorch Lightning flow.
    if data_module.graph_data is None:
        print("Running data_module.setup('fit')...") # or setup('test') if appropriate
        data_module.setup('fit') # 'fit' stage usually creates all masks
    
    # 2. Check if graph data and necessary attributes exist
    if data_module.graph_data is None:
        print("Error: graph_data not found after setup.")
    elif 'transaction' not in data_module.graph_data:
        print("Error: 'transaction' node store not found in graph_data.")
    elif not hasattr(data_module.graph_data['transaction'], 'y_global'):
        print("Error: 'y_global' not found in graph_data['transaction'].")
    elif not hasattr(data_module.graph_data['transaction'], 'val_mask'):
        print("Error: 'val_mask' not found in graph_data['transaction'].")
    elif not hasattr(data_module.graph_data['transaction'], 'test_mask'):
        print("Error: 'test_mask' not found in graph_data['transaction'].")
    else:
        # 3. Get the full tensor of labels
        all_labels = data_module.graph_data['transaction'].y_global
    
        # 4. Get the boolean masks
        val_mask = data_module.graph_data['transaction'].val_mask
        test_mask = data_module.graph_data['transaction'].test_mask
    
        # 5. Apply the masks to get the labels for each set
        val_labels = all_labels[val_mask]
        test_labels = all_labels[test_mask]
    
        # 6. Print or analyze the labels
        print(f"\n--- Validation Set Labels ({len(val_labels)} nodes) ---")
        # Convert to numpy for easier analysis like unique counts
        val_labels_np = val_labels.cpu().numpy()
        unique_val_cats, val_counts = np.unique(val_labels_np, return_counts=True)
        print("Unique Categories and Counts:")
        for cat, count in zip(unique_val_cats, val_counts):
            print(f"  Category {cat}: {count} nodes")
        # print("First 10 val labels:", val_labels[:10]) # Print sample
    
        print(f"\n--- Test Set Labels ({len(test_labels)} nodes) ---")
        test_labels_np = test_labels.cpu().numpy()
        unique_test_cats, test_counts = np.unique(test_labels_np, return_counts=True)
        print("Unique Categories and Counts:")
        for cat, count in zip(unique_test_cats, test_counts):
            print(f"  Category {cat}: {count} nodes")
        # print("First 10 test labels:", test_labels[:10]) # Print sample

    
    try:
        train_loader = data_module.train_dataloader()
        first_batch = next(iter(train_loader))
        print(f"First batch type: {type(first_batch)}")
    
        # --- Check x_dict (Node Features) ---
        print("Checking x_dict:")
        if hasattr(first_batch, 'x_dict') and first_batch.x_dict:
            for node_type, x in first_batch.x_dict.items():
                print(f"  Node Type '{node_type}':")
                print(f"    Shape={x.shape}, dtype={x.dtype}")
                # Check stats only for floating point tensors
                if torch.is_floating_point(x):
                    has_nan = torch.isnan(x).any().item()
                    has_inf = torch.isinf(x).any().item()
                    # Safely calculate min/max only if tensor is not empty and has no NaNs/Infs
                    if x.numel() > 0 and not (has_nan or has_inf):
                        # Using try-except just in case min/max fails unexpectedly on valid tensor
                        try:
                            min_val = torch.min(x).item()
                            max_val = torch.max(x).item()
                        except RuntimeError:
                            min_val = 'Error calculating'
                            max_val = 'Error calculating'
                    else:
                        min_val = 'N/A (empty/nan/inf)'
                        max_val = 'N/A (empty/nan/inf)'
                    # Print without specific float formatting to avoid errors with 'N/A'
                    print(f"    hasNaN={has_nan}, hasInf={has_inf}, min={min_val}, max={max_val}")
                else:
                    print(f"    Non-floating point tensor.")
        else:
            print("  Batch has no x_dict or it's empty.")
    
        # --- Check Edge Attributes (Handles both common storage methods) ---
        print("Checking edge attributes:")
        edge_attr_data_found = False
    
        # Method 1: Check batch.edge_attr_dict (Older PyG style)
        if hasattr(first_batch, 'edge_attr_dict') and first_batch.edge_attr_dict is not None:
            print("  Checking batch.edge_attr_dict:")
            # Check if the dictionary itself is not empty
            if first_batch.edge_attr_dict:
                 edge_attr_data_found = True
                 for edge_type, edge_attr in first_batch.edge_attr_dict.items(): # Correct variable names
                     print(f"    Edge Type '{edge_type}':") # Use edge_type (key)
                     print(f"      Shape={edge_attr.shape}, dtype={edge_attr.dtype}")
                     if torch.is_floating_point(edge_attr):
                         has_nan = torch.isnan(edge_attr).any().item()
                         has_inf = torch.isinf(edge_attr).any().item()
                         # Safely calculate min/max
                         if edge_attr.numel() > 0 and not (has_nan or has_inf):
                             try:
                                 min_val = torch.min(edge_attr).item()
                                 max_val = torch.max(edge_attr).item()
                             except RuntimeError:
                                 min_val = 'Error calculating'
                                 max_val = 'Error calculating'
                         else:
                             min_val = 'N/A (empty/nan/inf)'
                             max_val = 'N/A (empty/nan/inf)'
                         # Print without specific float formatting
                         print(f"      hasNaN={has_nan}, hasInf={has_inf}, min={min_val}, max={max_val}")
                     else:
                         print(f"      Non-floating point tensor.")
            else:
                 print("  batch.edge_attr_dict is empty.")
    
    
        # Method 2: Check batch[edge_type].edge_attr (Newer PyG style)
        printed_method2_header = False
        if hasattr(first_batch, 'edge_types'):
            for edge_type_tuple in first_batch.edge_types:
                if hasattr(first_batch[edge_type_tuple], 'edge_attr'):
                    if not edge_attr_data_found and not printed_method2_header:
                        # Print header only once if method 1 didn't find anything or doesn't exist
                        print("  Checking batch[edge_type].edge_attr:")
                        printed_method2_header = True
                    edge_attr_data_found = True
                    edge_attr = first_batch[edge_type_tuple].edge_attr
    
                    print(f"    Edge Type '{edge_type_tuple}':") # Use edge_type_tuple
                    print(f"      Shape={edge_attr.shape}, dtype={edge_attr.dtype}")
                    if torch.is_floating_point(edge_attr):
                        has_nan = torch.isnan(edge_attr).any().item()
                        has_inf = torch.isinf(edge_attr).any().item()
                        # Safely calculate min/max
                        if edge_attr.numel() > 0 and not (has_nan or has_inf):
                             try:
                                 min_val = torch.min(edge_attr).item()
                                 max_val = torch.max(edge_attr).item()
                             except RuntimeError:
                                 min_val = 'Error calculating'
                                 max_val = 'Error calculating'
                        else:
                            min_val = 'N/A (empty/nan/inf)'
                            max_val = 'N/A (empty/nan/inf)'
                        # Print without specific float formatting
                        print(f"      hasNaN={has_nan}, hasInf={has_inf}, min={min_val}, max={max_val}")
                    else:
                        print(f"      Non-floating point tensor.")
    
        # Final check if no attributes were found anywhere
        if not edge_attr_data_found:
            print("  No edge attributes found via batch.edge_attr_dict or batch[edge_type].edge_attr.")
    
    except Exception as e:
        print(f"!!! ERROR during First Batch Check: {type(e).__name__}: {e}")
        # Optionally re-raise if you want the script to stop on check error
        # raise e
    
    # print("--- End First Batch Check --- \nStarting model creation...")
    
    # print("--- End First Batch Check --- \nStarting training...")
    
    # print("Creating Model")
    # Create model
    # print(f"\n--- Creating Model ---")
    # print(f"  Using Sequence Input Dim: {sequence_dim}") # Add print
    # print(f"  Using Edge Input Dims: {edge_dims}") # Add print
    model = TransactionClassifier(
        num_classes=df['category_id'].nunique(),
        gnn_hidden_channels=256,
        gnn_out_channels = 256,
        gnn_node_input_dims = node_dims,
        gnn_edge_input_dims = edge_dims, # Pass edge dimensions
        gnn_num_layers=2,
        gnn_heads=4,
        seq_input_dim=sequence_dim, # Pass the correct dimension
        seq_hidden_size=256,
        seq_num_layers=2,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_sequence_encoder=use_sequence_encoder,
        use_text_encoder=use_text_encoder,
        use_gnn_encoder=use_gnn_encoder,
        gnn_only_test_mode=gnn_only_test_mode,
        gnn_metadata=gnn_metadata,
        full_graph_data_ref=full_graph_data # Pass reference to full graph
    )
    # print("Model Created")
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='logs'
    )
    
    # print("Creating Trainer")
    trainer = pl.Trainer(
        max_epochs=1000,
        # Restore original logic: Use CUDA if available, otherwise CPU
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,   # Explicitly enable
        enable_model_summary=True,  # Show model architecture
        log_every_n_steps=1,
        precision='32',
        min_epochs=100,
       # num_sanity_val_steps=1 ,
        gradient_clip_val=1 
    )
    # print("Trainer Created")
    
    # print("Starting Trainer.fit()")
    # Train model
    try:
        trainer.fit(model, data_module)
        # print("Trainer.fit() finished")
    except Exception as e_fit:
        print(f"\n!!! ERROR during trainer.fit(): {type(e_fit).__name__}: {e_fit}")
        import traceback
        traceback.print_exc()
        # Optionally re-raise or exit
        # raise e_fit 
        sys.exit(1) # Exit if fit fails
    
    # print("Starting Trainer.test()")
    # Test model
    test_results = None # Initialize
    try:
        test_results = trainer.test(model, data_module)
        # print("Trainer.test() finished")
    except Exception as e_test:
        print(f"\n!!! ERROR during trainer.test(): {type(e_test).__name__}: {e_test}")
        import traceback
        traceback.print_exc()
        # Continue to saving part if possible, or exit

    # Save test results
    if test_results and len(test_results) > 0:
        # Convert test results to DataFrame
        results_df = pd.DataFrame([test_results[0]])
        results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
        
        # Get model predictions on test set
        model.eval()
        test_predictions = []
        test_dataloader = data_module.test_dataloader()
        
        # with torch.no_grad():
        #     for batch in test_dataloader:
        #         global_logits, user_logits, _ = model(batch)
        #         global_preds = torch.argmax(global_logits, dim=1)
                
        #         # Get true labels
        #         true_labels = batch['labels']
                
        #         # Add predictions and true labels to list
        #         for i, pred in enumerate(global_preds):
        #             test_predictions.append({
        #                 'batch_idx': i,
        #                 'predicted_category': pred.item(),
        #                 'true_category': true_labels[i].item()
        #             })
        model.eval()
        
        print("\n--- Starting Manual Prediction Generation (Simplified) ---")
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        print(f"Using device: {device}")
        
        test_predictions = []
        try:
            with torch.no_grad():
                for i_batch, batch in enumerate(test_dataloader):
                    # --- Get Labels from ORIGINAL CPU batch first ---
                    true_labels_cpu = None
                    batch_size_this = 0
                    if 'transaction' in batch and hasattr(batch['transaction'], 'y_global'):
                         try:
                              batch_size_this = batch['transaction'].batch_size
                              # Slice labels on CPU
                              true_labels_cpu = batch['transaction'].y_global[:batch_size_this].cpu()
                         except Exception as e_label_cpu:
                              print(f"[WARN] Batch {i_batch}: Error accessing/slicing CPU labels: {e_label_cpu}")
        
                    # Move batch to device for model
                    try:
                        batch_on_device = batch.to(device)
                    except Exception as e_move:
                        print(f"[ERROR] Failed to move batch {i_batch} to device {device}: {e_move}")
                        continue
        
                    # Get model predictions
                    try:
                        global_logits, _, _ = model(batch_on_device)
                        global_preds_cpu = torch.argmax(global_logits, dim=1).cpu()
                    except Exception as e_model:
                        print(f"[ERROR] Model forward pass failed for batch {i_batch}: {e_model}")
                        continue
        
                    # Append results
                    if len(global_preds_cpu) != batch_size_this and true_labels_cpu is not None :
                         print(f"[WARN] Batch {i_batch}: Pred count {len(global_preds_cpu)} != Batch size {batch_size_this}. Appending without labels.")
                         true_labels_cpu = None # Avoid index errors
        
                    for i in range(len(global_preds_cpu)):
                         label_item = true_labels_cpu[i].item() if true_labels_cpu is not None and i < len(true_labels_cpu) else None
                         test_predictions.append({
                             'pred_index_in_batch': i,
                             'predicted_category': global_preds_cpu[i].item(),
                             'true_category': label_item
                         })
        
        except Exception as e_loop:
             print(f"\n!!! ERROR during Manual Prediction Loop execution: {type(e_loop).__name__}: {e_loop}")
             import traceback
             traceback.print_exc()
        
        print(f"\n--- Finished Manual Prediction Generation ({len(test_predictions)} predictions collected) ---")
        # (Save predictions logic follows)
                
        # --- Optional: Save the collected predictions ---
        if test_predictions:
            try:
                pred_df = pd.DataFrame(test_predictions)
                save_path = os.path.join(output_dir, 'test_predictions_manual.csv')
                pred_df.to_csv(save_path, index=False)
                print(f"Saved {len(pred_df)} manual predictions to {save_path}")
            except Exception as e_save:
                print(f"[ERROR] Failed to save predictions to CSV: {e_save}")
        else:
            print("No predictions were collected in the manual loop to save.")
        # with torch.no_grad():
        #     for batch in test_dataloader:
        #         global_logits, user_logits, _ = model(batch) # Assuming model is on correct device
        #         global_preds = torch.argmax(global_logits, dim=1)
        
        #         # --- FIX: Get true labels correctly ---
        #         # Original failing line:
        #         # true_labels = batch['labels']
        #         # Corrected lines:
        #         if 'transaction' in batch and hasattr(batch['transaction'], 'y_global'):
        #             # Get labels for the seed nodes in this batch
        #             true_labels = batch['transaction'].y_global[:batch['transaction'].batch_size]
        #         else:
        #             # Handle cases where labels might be missing in the batch (shouldn't happen in test)
        #             print("[WARN] Could not find 'transaction.y_global' in test batch. Skipping label extraction for this batch.")
        #             true_labels = None # Or handle appropriately
        
        #         # Add predictions and true labels to list
        #         if true_labels is not None: # Only proceed if labels were found
        #             for i, pred in enumerate(global_preds):
        #                  # Ensure index i is valid for true_labels
        #                  if i < len(true_labels):
        #                       test_predictions.append({
        #                           'batch_idx': i, # This index is within the batch
        #                           'predicted_category': pred.item(),
        #                           'true_category': true_labels[i].item() # Access label corresponding to prediction i
        #                       })
        #                  else:
        #                       print(f"[WARN] Index mismatch: pred index {i} >= true_labels length {len(true_labels)}")
        #         # else: predictions were generated but labels weren't available for comparison

        
        # Save predictions
        pred_df = pd.DataFrame(test_predictions)
        if len(pred_df) > 0:
            pred_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)

    # print("Script Finished")

if __name__ == '__main__':
    # print("Running __main__ block")
    import argparse
    
    parser = argparse.ArgumentParser(description='Train transaction classifier')
    parser.add_argument('--gnn_only', action='store_true',
                    help='Run model in GNN-only mode for debugging.')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to transaction data CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save model checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=100,
                      help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay for optimizer')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                      help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--max_seq_length', type=int, default=50,
                      help='Maximum sequence length for user history')
    parser.add_argument('--text_model_name', type=str, default='bert-base-uncased',
                      help='Name of pretrained text model')
    parser.add_argument('--text_max_length', type=int, default=128,
                      help='Maximum sequence length for text inputs')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                      help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                      help='Ratio of test data')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--no_sequence', action='store_true',
                        help='Disable the sequence encoder modality.')
    parser.add_argument('--no_text', action='store_true',
                        help='Disable the text encoder modality.')
    parser.add_argument('--no_gnn', action='store_true',
                        help='Disable the GNN encoder modality.')
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        text_model_name=args.text_model_name,
        text_max_length=args.text_max_length,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_sequence_encoder=not args.no_sequence,
        use_text_encoder=not args.no_text,
        use_gnn_encoder=not args.no_gnn,
        gnn_only_test_mode=args.gnn_only
    ) 