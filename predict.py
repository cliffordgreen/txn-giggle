import os
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import pickle # To load saved scalers/mappings later
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
    output_csv: Optional[str] = None,
    batch_size: int = 64, # Can often use larger batch size for inference
    num_workers: int = 0,
    # Add args for saved scalers/mappings later
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

    print(f"Loading data from directory: {data_dir}")
    # This loads the data we want to predict on
    predict_df = load_data(data_dir) 

    # TODO: Load saved scalers and mappings from training run
    # e.g., with open('training_output/scalers.pkl', 'rb') as f: fitted_scalers = pickle.load(f)
    #       with open('training_output/mappings.pkl', 'rb') as f: fitted_mappings = pickle.load(f)
    
    print("Initializing DataModule for prediction...")
    # NOTE: We need to modify DataModuleV2 to accept fitted scalers/mappings
    # For now, placeholder initialization - this will likely fail or misuse data
    # We assume model_config contains necessary static parameters like text_model_name etc.
    # but dynamic ones like num_classes/users might be needed from checkpoint hparams.
    data_module = TransactionDataModuleV2(
        transactions_df_ref=predict_df, # Pass prediction data
        batch_size=batch_size,
        num_workers=num_workers,
        num_hgt_layers=model_config['graph_encoder_params'].get('num_layers', 2), # Example static param
        hgt_num_samples=None, # Or load from config/train state if needed
        text_model_name=model_config['text_encoder_params'].get('model_name', 'ProsusAI/finbert'),
        max_seq_length=model_config.get('max_seq_length', 50),
        text_max_length=model_config['text_encoder_params'].get('max_length', 128),
        use_sequence_encoder=model_config.get('use_sequence_encoder', True),
        use_gnn_encoder=model_config.get('use_graph_encoder', True),
        use_text_encoder=model_config.get('use_text_encoder', True),
        # TODO: Pass fitted_scalers and fitted_mappings here after loading
        # fitted_scalers=fitted_scalers, 
        # fitted_user_map=fitted_mappings['user_map'],
        # fitted_category_map=fitted_mappings['category_map']
    )
    # TODO: Call a modified setup method, e.g., data_module.setup('predict')
    # This setup should apply the loaded scalers/mappings, not fit new ones.
    # For now, using 'test' stage setup as a placeholder
    print("Setting up DataModule (placeholder - needs modification)..." )
    data_module.setup('test') 

    print(f"Loading model from checkpoint: {checkpoint_path}")
    # Load the model - might need to pass the config if not saved in checkpoint hparams
    # Or update config with hparams from checkpoint after loading? Check Lightning docs.
    # model = AdvancedTransactionCategorizationModel.load_from_checkpoint(checkpoint_path, map_location='cpu') # Load to CPU initially
    # Let's try passing the config directly, assuming it contains necessary static info
    # The dynamic parts (num_classes, etc.) should ideally be in hparams or loaded separately.
    try:
        model = AdvancedTransactionCategorizationModel.load_from_checkpoint(
            checkpoint_path, 
            map_location='cpu', # Load to CPU initially
            model_config=model_config # Pass the loaded config
            # We might need to update model_config with num_classes etc. AFTER loading checkpoint hparams
        )
        print("Model loaded successfully.")
    except Exception as e:
         print(f"[ERROR] Failed to load model from checkpoint {checkpoint_path}: {e}")
         print("Ensure the checkpoint is compatible and the model_config is correct.")
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
    
    # We need the integer mapping used during training to compare predictions (integers)
    # TODO: Load the category_id_map from the saved training state
    # category_id_map_inverse = {v: k for k, v in fitted_mappings['category_map'].items()} # Example
    category_id_map = data_module.category_id_map # Placeholder - uses map from predict_df! Needs fix.
    if category_id_map is None:
         print("[ERROR] Category ID map not found in DataModule. Cannot calculate accuracy correctly.")
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
        # Add predicted string label using the map
        map_int_to_str = {k: v for k, v in category_id_map.items()}
        results_df['predicted_label_str'] = results_df['predicted_label_int'].map(map_int_to_str).fillna('UNKNOWN_PRED')
        
        # Optionally include original data columns if needed
        # results_df = pd.concat([predict_df.iloc[:len(predictions)].reset_index(drop=True), results_df], axis=1)
        
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
    parser.add_argument('--output_csv', type=str, default=None, 
                        help='Optional path to save predictions and true labels to a CSV file')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for prediction')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of data loading workers')
    # TODO: Add arguments for paths to saved scalers/mappings

    args = parser.parse_args()

    predict(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        config_path=args.config_path,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Pass loaded scaler/mapping paths here later
    ) 