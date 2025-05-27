import os
import argparse
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
import yaml # For loading potential YAML configs
from typing import Optional, Dict, List
import pyarrow as pa # Added for ArrowInvalid check
import pyarrow.ipc as ipc # Use ipc explicitly for stream reading
import numpy as np # Added
import pickle # Added for saving state

# Use the V2 DataModule
from data.data_module_v2 import TransactionDataModuleV2 # Removed SingleBatchIterable as it might not be compatible with new flow
# Import the new advanced model
from models.advanced_transaction_classifier import AdvancedTransactionCategorizationModel

def load_file_paths(data_dir: str, max_files_to_process: Optional[int] = None) -> List[str]:
    """Scans a directory for .arrow files and returns a list of paths to process."""
    print(f"Scanning for .arrow files in directory: {data_dir}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    arrow_files_pattern = os.path.join(data_dir, '**/*.parquet')
    all_arrow_files = sorted(glob.glob(arrow_files_pattern, recursive=True))
    
    if not all_arrow_files:
        raise FileNotFoundError(f"No .parquet files found in directory: {data_dir}")

    files_to_process = all_arrow_files
    if max_files_to_process is not None and max_files_to_process > 0:
        print(f"Limiting processing to the first {max_files_to_process} .parquet files found out of {len(all_arrow_files)} total.")
        files_to_process = all_arrow_files[:max_files_to_process]
    else:
        print(f"Found {len(all_arrow_files)} .parquet files to process.")
        
    return files_to_process

# Helper function to load data (similar to old train.py)
# def load_data(data_dir: str, max_files_to_process: Optional[int] = None) -> pd.DataFrame:
#     """Loads and preprocesses data from multiple Arrow streaming files, one by one.
#     Processes up to max_files_to_process if specified.
#     """
#     print(f"Loading data from directory: {data_dir}")
#     arrow_files_pattern = os.path.join(data_dir, '**/*.arrow')
#     # Sort files for deterministic behavior when using max_files_to_process
#     arrow_files = sorted(glob.glob(arrow_files_pattern, recursive=True))
    
#     if not arrow_files:
#         if not os.path.isdir(data_dir):
#              raise FileNotFoundError(f"Data directory not found: {data_dir}")
#         raise FileNotFoundError(f"No .arrow files found in directory: {data_dir}")

#     # Limit number of files if max_files_to_process is set
#     files_to_process = arrow_files
#     if max_files_to_process is not None and max_files_to_process > 0:
#         print(f"Limiting processing to the first {max_files_to_process} files found.")
#         files_to_process = arrow_files[:max_files_to_process]
    
#     print(f"Found {len(arrow_files)} arrow files. Processing {len(files_to_process)} file(s) file by file...")
    
#     processed_dfs = []
#     skipped_files = []
    
#     for file_path in tqdm(files_to_process, desc="Processing Arrow files"):
#         try:
#             # 1. Read one Arrow stream file
#             with ipc.open_stream(file_path) as reader:
#                  table = reader.read_all()
#                  df_single = table.to_pandas()
            
#             # 2. Perform initial preprocessing on this single DataFrame
#             required_cols = ['target_transaction_processed', 'txn_accepted_category_id_str', 'company_name']
#             if not all(col in df_single.columns for col in required_cols):
#                 print(f"[WARN] Skipping file {os.path.basename(file_path)} due to missing required columns.")
#                 skipped_files.append(os.path.basename(file_path))
#                 continue # Skip to next file
                
#             extracted_data = []
#             for _, row in df_single.iterrows(): # Process rows within the small df
#                 try:
#                     if isinstance(row['target_transaction_processed'], dict):
#                         txn_dict = row['target_transaction_processed']
#                     else:
#                         txn_dict = json.loads(row['target_transaction_processed'])
                    
#                     extracted_data.append({
#                         'amount': float(txn_dict.get('amount', 0.0)),
#                         'timestamp': pd.to_datetime(txn_dict.get('created_date'), errors='coerce'),
#                         'description': str(txn_dict.get('description', '')),
#                         'memo': str(txn_dict.get('memo', '')),
#                         'merchant_name': str(txn_dict.get('payee', ''))
#                     })
#                 except (json.JSONDecodeError, TypeError, ValueError) as e:
#                     # print(f"[WARN] Error parsing target_transaction_processed in file {os.path.basename(file_path)}, row {row.name}: {e}. Using defaults.")
#                     extracted_data.append({'amount': 0.0, 'timestamp': pd.NaT, 'description': '', 'memo': '', 'merchant_name': ''})
            
#             extracted_df = pd.DataFrame(extracted_data, index=df_single.index)
#             df_processed_single = pd.concat([df_single, extracted_df], axis=1)

#             # Handle timestamps within the small df
#             if df_processed_single['timestamp'].isnull().any():
#                 median_date = df_processed_single['timestamp'].dropna().median()
#                 if pd.isna(median_date):
#                     median_date = pd.Timestamp('2020-01-01') 
#                 df_processed_single['timestamp'].fillna(median_date, inplace=True)

#             # Extract time features within the small df
#             df_processed_single['weekday'] = df_processed_single['timestamp'].dt.weekday
#             df_processed_single['hour'] = df_processed_single['timestamp'].dt.hour
            
#             # Ensure other columns exist and fill NaNs within the small df
#             df_processed_single['txn_accepted_category_id_str'] = df_processed_single['txn_accepted_category_id_str'].fillna('UNKNOWN').astype(str)
#             df_processed_single['company_name'] = df_processed_single['company_name'].fillna('UNKNOWN').astype(str)
#             if 'industry_name' in df_processed_single.columns:
#                  df_processed_single['industry_name'] = df_processed_single['industry_name'].fillna('UNKNOWN').astype(str)
#             else: df_processed_single['industry_name'] = 'UNKNOWN'
#             if 'num_chart_of_accounts' in df_processed_single.columns:
#                  df_processed_single['num_chart_of_accounts'] = pd.to_numeric(df_processed_single['num_chart_of_accounts'], errors='coerce').fillna(0).astype(int)
#             else: df_processed_single['num_chart_of_accounts'] = 0
#             # Don't drop target_transaction_processed yet, might be needed later? Keep it for now.
            
#             # Select only the columns needed downstream to potentially save memory before append
#             cols_to_keep = [
#                 'txn_accepted_category_id_str', 'company_name', 'industry_name', 
#                 'num_chart_of_accounts', 'chart_of_accounts_processed', # Keep COA for DataModule
#                 'amount', 'timestamp', 'description', 'memo', 'merchant_name', 
#                 'weekday', 'hour'
#                 # Add any other original columns if they are used by DataModule/Model
#             ]
#             # Filter df_processed_single to keep only necessary columns
#             df_filtered_single = df_processed_single[[col for col in cols_to_keep if col in df_processed_single.columns]]
            
#             processed_dfs.append(df_filtered_single)

#         except pa.lib.ArrowInvalid as e:
#             print(f"[WARN] Skipping invalid Arrow stream file: {os.path.basename(file_path)} - Reason: {e}")
#             skipped_files.append(os.path.basename(file_path))
#         except Exception as e:
#             print(f"[WARN] Skipping file {os.path.basename(file_path)} due to unexpected error: {e}")
#             skipped_files.append(os.path.basename(file_path))

#     if not processed_dfs:
#         raise ValueError(f"No valid Arrow files could be processed from {data_dir}. Skipped files: {skipped_files}")

#     # 3. Concatenate all *processed* DataFrames
#     print(f"Concatenating {len(processed_dfs)} processed DataFrames..." )
#     print("[WARNING] This step loads the full processed dataset into memory!")
#     df_combined = pd.concat(processed_dfs, ignore_index=True)
    
#     num_processed = len(processed_dfs)
#     num_skipped = len(skipped_files)
#     print(f"Data loading and initial processing complete: {len(df_combined)} records from {num_processed} files ({num_skipped} files skipped)." )
#     if skipped_files:
#          print(f"Skipped files: {skipped_files}")

#     # No further processing needed here, return the combined df
#     print("Preprocessing finished.") # Renamed log message
#     return df_combined

def train_advanced(
    # Data/Output
    data_dir: str, # Changed from data_path
    output_dir: str,
    # Basic Training Params
    batch_size: int = 32,
    num_workers: int = 0,
    max_epochs: int = 50,
    seed: int = 42,
    val_ratio: float = 0.1, # Add val_ratio
    test_ratio: float = 0.1, # Add test_ratio
    # Model Hyperparameters (passed via dicts)
    model_config: dict = {},
    # Training Strategy Params
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    mtl_weights: dict = {'global': 0.5, 'user': 0.5},
    focal_loss_alpha: float = 0.25,
    focal_loss_gamma: float = 2.0,
    # Add trainer specific args if needed (accelerator, precision etc.)
    accelerator: str = 'auto',
    precision: str = '32',
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
    # df = load_data(data_dir, max_files_to_process=max_files_to_process)
    train_file_paths = load_file_paths(data_dir, max_files_to_process=max_files_to_process)
    
    # For now, we won't have a pre-loaded df for these initial calculations.
    # These will need to be derived INSIDE the DataModule after it processes data.
    # num_global_classes = df['txn_accepted_category_id_str'].nunique()
    # num_user_classes = 0 # Adjust if user-specific task is defined later
    # num_users = df['company_name'].nunique() # Estimate here, refined in DataModule
    # print(f"Derived from data: #Global={num_global_classes}, #User={num_user_classes}, #Users={num_users}")
    
    # --- Data Module V2 --- 
    print("Initializing TransactionDataModuleV2...")
    # --- Get encoder flags from config --- 
    use_seq = model_config.get('use_sequence_encoder', True) # Default to True if missing
    use_graph = model_config.get('use_graph_encoder', True)   # Default to True
    use_text = model_config.get('use_text_encoder', True)     # Default to True
    # Get the new flag for COA text features
    use_coa_text = model_config.get('use_coa_text_features', True) # Default to True
    # Get COA text max length
    coa_text_max_length = model_config.get('coa_text_max_length', 256) # Default if missing

    num_hgt_layers = model_config['graph_encoder_params'].get('num_layers', 2) 
    data_module = TransactionDataModuleV2(
        # transactions_df_ref=df, # This will be replaced by file paths
        file_paths=train_file_paths, # Pass the list of file paths
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=val_ratio, # Use the passed val_ratio
        test_ratio=test_ratio, # Use the passed test_ratio
        text_model_name=model_config['text_encoder_params'].get('model_name', 'ProsusAI/finbert'),
        max_seq_length=model_config.get('max_seq_length', 50),
        text_max_length=model_config['text_encoder_params'].get('max_length', 128),
        use_sequence_encoder=use_seq,
        use_gnn_encoder=use_graph,
        use_text_encoder=use_text,
        use_coa_text_features=use_coa_text, # Pass the new flag
        coa_text_max_length=coa_text_max_length, # Pass the new length parameter
        model_config=model_config # Pass the full model_config
        # Add other arguments like fitted_scalers if this is also used for predict.py context
        # For now, assuming this specific snippet is from train_new.py for training context
    )
    print("Setting up DataModuleV2...")
    data_module.setup('fit') 
    
    # --- Print info about val_files ---
    if hasattr(data_module, 'val_files'):
        print(f"[INFO] Number of validation files in DataModule: {len(data_module.val_files)}")
        if not data_module.val_files:
            print("[WARN] Validation file list is empty. 'val_loss' might not be logged, affecting ModelCheckpoint.")
    else:
        print("[WARN] DataModule does not have 'val_files' attribute to check.")

    # --- Update Model Config with dynamic values AFTER setup --- 
    # Get class counts from DataModule instance
    num_global_classes = data_module.num_global_classes
    # num_user_classes = data_module.num_user_classes # User specific head removed from model
    num_users = data_module.num_users
    print(f"DataModule counts: #Global={num_global_classes}, #Users={num_users}")

    # Get feature dimensions and other necessary data from DataModule
    node_feature_dims = data_module.node_feature_dims
    edge_feature_dims = data_module.edge_feature_dims
    # graph_metadata = data_module.graph_metadata # This was yielding a dict

    # --- Reconstruct the graph_metadata tuple explicitly --- 
    # Assuming data_module.graph_metadata has become a dict {'node_types': [], 'edge_types': []}
    # This is a workaround for an observed issue where it's not a tuple as expected.
    dm_graph_meta_dict = data_module.graph_metadata 
    if not isinstance(dm_graph_meta_dict, dict) or 'node_types' not in dm_graph_meta_dict or 'edge_types' not in dm_graph_meta_dict:
        raise ValueError(f"data_module.graph_metadata is not the expected dictionary. Got: {dm_graph_meta_dict}")
    graph_metadata_tuple_for_model = (
        dm_graph_meta_dict['node_types'],
        dm_graph_meta_dict['edge_types']
    )
    print(f"[DEBUG train_new.py] Reconstructed graph_metadata_tuple_for_model: {graph_metadata_tuple_for_model}, type: {type(graph_metadata_tuple_for_model)}")

    # Get COA tensors if COA features are enabled
    user_tokenized_coa_tensors = None
    if use_coa_text: # use_coa_text is defined above from model_config
        if hasattr(data_module, 'global_user_coa_tensors') and data_module.global_user_coa_tensors is not None:
            user_tokenized_coa_tensors = data_module.global_user_coa_tensors
            print(f"Retrieved global_user_coa_tensors from DataModule. input_ids shape: {user_tokenized_coa_tensors['input_ids'].shape}")
        else:
            print("[WARN] COA features enabled in config, but global_user_coa_tensors not found or is None in DataModule.")

    # Update model_config with graph_metadata from DataModule if HGT is used
    if model_config.get('use_gnn_encoder', True):
        if hasattr(data_module, 'graph_metadata') and data_module.graph_metadata:
            model_config['graph_encoder_params']['metadata'] = data_module.graph_metadata
            print(f"Updated model_config with graph_metadata from DataModule: {data_module.graph_metadata}")
        elif 'metadata' not in model_config['graph_encoder_params']:
            # Fallback: if GNN is on, but DM didn't provide metadata and it's not in config already, this is an issue.
            raise ValueError("GNN encoder is enabled, but graph_metadata is not available from DataModule and not pre-set in model_config['graph_encoder_params']['metadata'].")
        else:
            print("GNN encoder is enabled, using pre-set graph_metadata from model_config.")

    # --- Model Instantiation --- 
    print("Initializing AdvancedTransactionCategorizationModel...")
    model = AdvancedTransactionCategorizationModel(
        model_config=model_config, 
        # Feature dimensions and counts (from DataModule after setup)
        node_feature_dims=node_feature_dims,
        edge_feature_dims=edge_feature_dims,
        num_users=num_users,
        num_global_classes=num_global_classes,
        graph_metadata=graph_metadata_tuple_for_model, # Use the reconstructed tuple
        # User COA tokenized tensors (from DataModule)
        user_tokenized_coa_tensors=user_tokenized_coa_tensors, 
        # Training params
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mtl_weights=mtl_weights,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma
        # transactions_df_ref=df # Remove this, model should not need the full df
    )
    # No longer needed if we fetch labels within _step from df/graph ref passed here
    
    # --- Callbacks & Logger --- 
    print("Setting up Callbacks...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"), # Explicit subdirectory
        filename='best_model', # Simpler filename, PL will add epoch/step if needed or we can customize more
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False # Keep filename clean
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss', # Monitor validation loss
        patience=10,        # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'
    )
    logger = TensorBoardLogger(save_dir=output_dir, name='adv_logs')

    # --- Trainer --- 
    print("Initializing Trainer...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision=precision,
        devices=1, # Assuming single device for now
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=1, # Log every batch
        val_check_interval=1.0, # Validate at the end of every epoch
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
        # It's often good to still try to save config/state if training fails mid-way
        # but after data_module setup and model_config finalization.
        # The finally block handles this.
        raise e
    finally:
        # --- Save Final Model Configuration ---
        print("Attempting to save final model configuration...")
        final_config_path = os.path.join(output_dir, 'final_model_config.yaml')
        try:
            # Ensure all serializable types (e.g., numpy arrays might need conversion to lists)
            # For now, assuming model_config is directly serializable by PyYAML
            with open(final_config_path, 'w') as f:
                yaml.dump(model_config, f, sort_keys=False)
            print(f"Final model configuration saved successfully to: {final_config_path}")
        except Exception as config_save_e:
            print(f"[ERROR] Failed to save final model configuration to {final_config_path}: {config_save_e}")

        # --- Save Preprocessing State (Scalers and Mappings) ---
        print("Attempting to save preprocessing state from DataModule...")
        # This assumes data_module has a method `get_state()` that returns a serializable dict
        # and that such a method is implemented in TransactionDataModuleV2.
        # If not, the direct attribute access like before is an alternative,
        # but a get_state() method is cleaner.
        
        # For now, sticking to the direct attribute access as per existing code.
        # If data_module might not be fully initialized in case of early error,
        # add checks.
        if hasattr(data_module, 'scalers') and hasattr(data_module, 'user_map'): # Basic check
            state_to_save = {
                'scalers': data_module.scalers if hasattr(data_module, 'scalers') else None,
                'seq_scalers': data_module.seq_scalers if hasattr(data_module, 'seq_scalers') else None,
                'edge_scalers': data_module.edge_scalers if hasattr(data_module, 'edge_scalers') else None,
                'user_map': data_module.user_map if hasattr(data_module, 'user_map') else None,
                'category_id_map': data_module.category_id_map if hasattr(data_module, 'category_id_map') else None,
                # Add other relevant state from DataModule
                'num_global_classes': data_module.num_global_classes if hasattr(data_module, 'num_global_classes') else None,
                'num_users': data_module.num_users if hasattr(data_module, 'num_users') else None,
                'node_feature_dims': data_module.node_feature_dims if hasattr(data_module, 'node_feature_dims') else None,
                'text_model_name': data_module.text_model_name if hasattr(data_module, 'text_model_name') else None,
                'max_seq_length': data_module.max_seq_length if hasattr(data_module, 'max_seq_length') else None,
                'text_max_length': data_module.text_max_length if hasattr(data_module, 'text_max_length') else None,
            }
            save_path = os.path.join(output_dir, 'data_module_preprocessing_state.pkl')
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(state_to_save, f)
                print(f"DataModule preprocessing state saved successfully to: {save_path}")
            except Exception as save_e:
                print(f"[ERROR] Failed to save DataModule preprocessing state to {save_path}: {save_e}")
        else:
            print("[WARN] DataModule does not seem to have the expected attributes for saving state (e.g., 'scalers', 'user_map'). Skipping state saving.")

        # Save the final model state (hyperparameters and datamodule state)
        final_model_path = os.path.join(output_dir, "final_model_with_state.ckpt_custom")
        dm_state = data_module.get_state()
        # Ensure all components of dm_state are serializable (especially custom objects or scalers if not default)
        # For PyTorch tensors in dm_state (like global_user_coa_tensors if we decide to save them via dm_state):
        # Convert to CPU before saving if they are on GPU
        if dm_state.get('global_user_coa_tensors') is not None: # Example if we add it to get_state
            if isinstance(dm_state['global_user_coa_tensors'], dict):
                for key in dm_state['global_user_coa_tensors']:
                    if isinstance(dm_state['global_user_coa_tensors'][key], torch.Tensor):
                        dm_state['global_user_coa_tensors'][key] = dm_state['global_user_coa_tensors'][key].cpu()
            elif isinstance(dm_state['global_user_coa_tensors'], torch.Tensor):
                dm_state['global_user_coa_tensors'] = dm_state['global_user_coa_tensors'].cpu()
            
        model_state_dict = model.state_dict()
        # Convert tensors in model_state_dict to CPU
        for key in model_state_dict:
            if isinstance(model_state_dict[key], torch.Tensor):
                model_state_dict[key] = model_state_dict[key].cpu()

        # Get model's hyperparameters (already saved by Lightning, but can include for explicitness)
        model_hparams = model.hparams 
        # Convert tensors in model_hparams (if any complex objects were saved)
        # For simple hparams like lr, gamma, it's usually fine.

        saved_object = {
            'model_hyperparameters': model_hparams,
            'model_state_dict': model_state_dict,
            'datamodule_state': dm_state,
            'config': model_config # Save the original runtime config for reference
        }
        try:
            torch.save(saved_object, final_model_path)
            print(f"Saved final model with DataModule state to {final_model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save final model with state: {e}")
            # Fallback: try saving just the Lightning checkpoint if custom save fails
            # lightning_ckpt_path = os.path.join(output_dir, "final_model_lightning.ckpt")
            # trainer.save_checkpoint(lightning_ckpt_path)
            # print(f"Saved standard Lightning checkpoint to {lightning_ckpt_path} as fallback.")

        print(f"Training finished. Best model checkpoint: {checkpoint_callback.best_model_path}")

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

    # --- Training Arguments --- 
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers (set to 0 for debugging)')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--accelerator', type=str, default='auto', help='Trainer accelerator (cpu, gpu, auto)')
    parser.add_argument('--precision', type=str, default='32', help='Trainer precision (e.g., 16, 32, bf16)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Proportion of data to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Proportion of data to use for testing')
    
    # <<< HGT Sampler Args >>>
    # Allow passing sampling config via CLI (complex, maybe better in YAML)
    # Example: --hgt_samples "{'transaction':[15,10], 'user':[10,5]}"
    parser.add_argument('--hgt_samples', type=str, default=None, 
                        help='JSON/YAML string defining num_samples per node type per layer for HGTLoader')
    
    # --- MTL/Focal Loss Arguments --- 
    parser.add_argument('--mtl_weight_global', type=float, default=0.5, help='Weight for global loss')
    parser.add_argument('--mtl_weight_user', type=float, default=0.5, help='Weight for user loss')
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
    train_advanced(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        seed=args.seed,
        val_ratio=args.val_ratio, # Pass to train_advanced
        test_ratio=args.test_ratio, # Pass to train_advanced
        model_config=model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mtl_weights={'global': args.mtl_weight_global, 'user': args.mtl_weight_user},
        focal_loss_alpha=args.focal_alpha,
        focal_loss_gamma=args.focal_gamma,
        accelerator=args.accelerator,
        precision=args.precision,
        hgt_num_samples=hgt_samples_dict, # Pass parsed dict or None
        # Pass max_files_to_process to load_data via train_advanced
        max_files_to_process=args.max_files_to_process
    ) 