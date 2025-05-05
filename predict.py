import os
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import pickle # To load saved scalers/mappings later
import numpy as np # Added for concatenate
from sklearn.metrics import accuracy_score, classification_report

# Import necessary components from our project
from train_new import load_data # Reuse data loading function
from data.data_module_v2 import TransactionDataModuleV2
from models.advanced_transaction_classifier import AdvancedTransactionCategorizationModel
import yaml


def predict(
    checkpoint_path: str,
    data_dir: str,
    config_path: str,
    state_path: str, # Added path for saved state
    output_csv: Optional[str] = None,
    batch_size: int = 64, # Can often use larger batch size for inference
    num_workers: int = 0,
):
    """Load a trained model and make predictions on new data."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        print("Model configuration loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found at {config_path}. Cannot proceed.")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load or parse config file {config_path}: {e}")
        raise

    # --- Load Preprocessing State --- 
    print(f"Loading preprocessing state from: {state_path}")
    try:
        with open(state_path, 'rb') as f:
            preprocessing_state = pickle.load(f)
        print("Preprocessing state loaded successfully.")
        # Extract components
        fitted_scalers = preprocessing_state.get('scalers')
        fitted_seq_scalers = preprocessing_state.get('seq_scalers')
        fitted_edge_scalers = preprocessing_state.get('edge_scalers')
        fitted_user_map = preprocessing_state.get('user_map')
        fitted_category_id_map = preprocessing_state.get('category_id_map')
        # Basic validation
        if not all([fitted_scalers is not None, fitted_seq_scalers is not None, fitted_edge_scalers is not None, 
                    fitted_user_map is not None, fitted_category_id_map is not None]):
             raise ValueError("Loaded preprocessing state is missing required keys.")
    except FileNotFoundError:
        print(f"[ERROR] Preprocessing state file not found at {state_path}. Cannot proceed.")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load or parse preprocessing state file {state_path}: {e}")
        raise

    print(f"Loading data from directory: {data_dir}")
    # This loads the data we want to predict on
    predict_df = load_data(data_dir) # Assuming load_data doesn't need max_files here

    print("Initializing DataModule for prediction...")
    # Pass the loaded state to DataModuleV2 constructor
    # (Need to modify DataModuleV2.__init__ next)
    data_module = TransactionDataModuleV2(
        transactions_df_ref=predict_df, # Pass prediction data
        batch_size=batch_size,
        num_workers=num_workers,
        # Static params from config
        num_hgt_layers=model_config['graph_encoder_params'].get('num_layers', 2), 
        hgt_num_samples=None, # Not typically needed for prediction graph structure?
        text_model_name=model_config['text_encoder_params'].get('model_name', 'ProsusAI/finbert'),
        max_seq_length=model_config.get('max_seq_length', 50),
        text_max_length=model_config['text_encoder_params'].get('max_length', 128),
        use_sequence_encoder=model_config.get('use_sequence_encoder', True),
        use_gnn_encoder=model_config.get('use_graph_encoder', True),
        use_text_encoder=model_config.get('use_text_encoder', True),
        # Pass fitted state
        fitted_scalers=fitted_scalers, 
        fitted_seq_scalers=fitted_seq_scalers,
        fitted_edge_scalers=fitted_edge_scalers,
        fitted_user_map=fitted_user_map,
        fitted_category_id_map=fitted_category_id_map
    )
    
    # Call setup - it will need modification to use the fitted state
    print("Setting up DataModule using loaded state..." )
    data_module.setup('predict') # Using 'predict' stage - DataModule needs to handle this

    print(f"Loading model from checkpoint: {checkpoint_path}")
    # It's safer to update config with counts derived from loaded maps
    model_config['num_users'] = len(fitted_user_map)
    model_config['num_global_classes'] = len(fitted_category_id_map)
    model_config['num_user_classes'] = 0 # Assuming still 0
    # Potentially update metadata if it changed (though unlikely if structure is same)
    # model_config['graph_encoder_params']['metadata'] = data_module.full_graph_data.metadata() 

    try:
        # Pass updated model_config
        model = AdvancedTransactionCategorizationModel.load_from_checkpoint(
            checkpoint_path, 
            map_location='cpu',
            model_config=model_config 
        )
        print("Model loaded successfully.")
    except Exception as e:
         print(f"[ERROR] Failed to load model from checkpoint {checkpoint_path}: {e}")
         print("Ensure the checkpoint/config/state are compatible.")
         raise

    # --- Trainer Initialization ---
    # Minimal trainer for prediction
    trainer = pl.Trainer(
        accelerator='auto', 
        devices=1,
        logger=False # No logging needed for prediction
    )

    print("Starting prediction...")
    # trainer.predict returns a list of outputs per batch
    predictions_batches = trainer.predict(model, datamodule=data_module)

    # --- Process Predictions ---
    print("Processing predictions...")
    all_global_preds = []
    # predictions_batches is a list of tuples (global_logits, user_logits) per batch
    for batch_output in predictions_batches:
        global_logits, _ = batch_output # We only care about global predictions for now
        if global_logits is not None:
            preds = torch.argmax(global_logits, dim=1)
            all_global_preds.append(preds.cpu().numpy())
        else:
             print("[WARN] Found None in batch predictions output.")

    if not all_global_preds:
        print("[ERROR] No valid predictions were generated.")
        return
        
    predictions = np.concatenate(all_global_preds)

    # --- Get True Labels ---
    # Need to ensure predict_df aligns with the order of predictions
    # HGTLoader processes in order specified by input_nodes (which is all nodes in 'test' setup)
    # So, the order should match the predict_df if it was sorted correctly in setup.
    true_labels_str = predict_df['txn_accepted_category_id_str'].iloc[:len(predictions)] # Slice to match prediction count
    
    # Use the loaded category_id map for mapping
    category_id_map = fitted_category_id_map 
    if category_id_map is None:
         print("[ERROR] Loaded Category ID map is None. Cannot calculate accuracy correctly.")
         return
         
    # Convert true string labels to integer labels based on the *training* mapping
    try:
        # Create inverse map for string lookup
        map_str_to_int = {v: k for k, v in category_id_map.items()}
        true_labels = true_labels_str.map(map_str_to_int).fillna(-1).astype(int) # Map strings to ints, handle unknowns
        # Filter out any labels that weren't in the training map (-1)
        valid_indices = (true_labels != -1)
        true_labels_filtered = true_labels[valid_indices]
        predictions_filtered = predictions[valid_indices]
    except Exception as e:
         print(f"[ERROR] Failed to map true labels using category map: {e}")
         return

    if len(true_labels_filtered) == 0:
        print("[ERROR] No valid true labels found after mapping. Cannot calculate accuracy.")
        return

    # --- Calculate Metrics ---
    accuracy = accuracy_score(true_labels_filtered, predictions_filtered)
    print(f"\n--- Evaluation Results ---")
    print(f"Accuracy on prediction set: {accuracy:.4f}")
    
    # Optional: More detailed report
    try:
        # Need inverse map from int back to string for report labels
        map_int_to_str = {k: v for k, v in category_id_map.items()}
        report = classification_report(
            true_labels_filtered, 
            predictions_filtered, 
            labels=list(map_int_to_str.keys()), # Use integer labels
            target_names=list(map_int_to_str.values()), # Use string names
            zero_division=0
        )
        print("\nClassification Report:")
        print(report)
    except Exception as e:
        print(f"\n[WARN] Could not generate classification report: {e}")

    # --- Save Results (Optional) ---
    if output_csv:
        print(f"Saving predictions to: {output_csv}")
        results_df = pd.DataFrame({
            'true_label_str': true_labels_str, # Original string labels
            'true_label_int': true_labels,     # Mapped integer labels (-1 if unknown)
            'predicted_label_int': predictions # Model output integer labels
        })
        # Add predicted string label using the loaded map
        map_int_to_str = {v: k for k, v in category_id_map.items()} # Use loaded map
        results_df['predicted_label_str'] = results_df['predicted_label_int'].map(map_int_to_str).fillna('UNKNOWN_PRED')
        
        results_df.to_csv(output_csv, index=False)
        print("Predictions saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using a trained AdvancedTransactionCategorizationModel')
    
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the saved model checkpoint (.ckpt file)')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the directory containing prediction data (.arrow files)')
    parser.add_argument('--config_path', type=str, required=True, 
                        help='Path to the model configuration YAML file used during training')
    parser.add_argument('--state_path', type=str, required=True, 
                        help='Path to the saved preprocessing state file (preprocessing_state.pkl)') # Added state_path arg
    parser.add_argument('--output_csv', type=str, default=None, 
                        help='Optional path to save predictions and true labels to a CSV file')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for prediction')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of data loading workers')

    args = parser.parse_args()

    predict(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        config_path=args.config_path,
        state_path=args.state_path, # Pass state_path
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    ) 