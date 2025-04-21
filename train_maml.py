import torch
import pytorch_lightning as pl
import pandas as pd
import argparse
import os
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml

# Import your custom modules
from models.advanced_transaction_classifier import AdvancedTransactionCategorizationModel # Assuming MAML modifications
from data.maml_data_module import MAMLTransactionDataModule # Assuming you rename the file or class

def main(args):
    print("--- Starting MAML Training Script ---")
    print(f"Arguments: {args}")

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # --- 0. Load Raw Transaction Data ---
    print(f"Loading raw transaction data from: {args.data_path}")
    try:
        # Handle potential dtype warnings if needed
        transactions_df = pd.read_csv(args.data_path, low_memory=False)
        print(f"Loaded raw DataFrame with shape: {transactions_df.shape}")
        # Optional: Basic validation (e.g., check required columns like user_id, labels)
        if args.user_id_col not in transactions_df.columns:
             raise ValueError(f"Required user ID column '{args.user_id_col}' not found in {args.data_path}")
        if args.user_label_col not in transactions_df.columns:
             print(f"[WARN] Specified MAML target label column '{args.user_label_col}' not found. Check data or args.")
             # Potentially raise error or proceed depending on requirements
    except FileNotFoundError:
        print(f"[ERROR] Raw data file not found at {args.data_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load raw data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 1. Load Base Configuration ---
    print(f"Loading base model configuration from: {args.config_path}")
    try:
        with open(args.config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        print("Base model configuration loaded successfully.")
    except FileNotFoundError:
        print(f"[WARN] Base configuration file not found at {args.config_path}. Using defaults/command-line args only.")
        base_config = {} # Start with empty dict if file not found
    except Exception as e:
        print(f"[ERROR] Failed to load or parse base config file {args.config_path}: {e}")
        return

    # --- 2. Initialize MAML DataModule ---
    print("Initializing MAMLTransactionDataModule...")
    maml_dm = MAMLTransactionDataModule(
        transactions_df=transactions_df, # Pass the loaded raw DataFrame
        user_id_column=args.user_id_col,
        label_column=args.global_label_col, # Base label (if needed)
        user_label_column=args.user_label_col, # User-specific label for MAML adaptation
        # timestamp_column=args.timestamp_col, # Timestamp handling now inside DataModule
        K_shot=args.k_shot,
        Q_query=args.q_query,
        meta_batch_size=args.meta_batch_size,
        num_workers=args.num_workers,
        meta_val_ratio=args.meta_val_ratio,
        meta_test_ratio=args.meta_test_ratio,
        seed=args.seed
        # Removed modality flags - MAML DM doesn't need them directly
    )

    # --- Setup DataModule FIRST to get data-dependent config values ---
    print("Setting up MAML DataModule to derive config...")
    try:
        maml_dm.setup(stage='fit') # Call setup explicitly
        print("MAML DataModule setup complete.")
    except Exception as e:
        print(f"[ERROR] Failed during MAML DataModule setup: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. Initialize Model (Configured for MAML) ---
    print("Configuring model with values derived from DataModule...")

    # Get necessary dims/counts from the setup DataModule instance
    num_users = maml_dm.num_users # Should still be calculated during user split
    num_maml_target_classes = maml_dm.num_maml_classes # Should be calculated during label processing

    # --- Construct Final Model Config ---
    model_config = base_config.copy()

    # Override specific keys with data-derived or essential MAML values
    model_config['use_gnn_encoder'] = args.use_gnn
    model_config['use_sequence_encoder'] = args.use_sequence
    model_config['use_text_encoder'] = args.use_text
    model_config['num_users'] = num_users
    model_config['user_embed_dim'] = model_config.get('user_embed_dim', 64) # Keep configurable
    model_config['num_user_classes'] = num_maml_target_classes # Set user classes based on MAML target
    model_config['num_global_classes'] = 0 # Assume global head is not adapted/used in MAML

    # Add/Update sub-configs based on DataModule results and flags
    # These might need adjustment if dimensions/metadata aren't passed from MAML DM
    # Best practice: Load processed graph here to get final dims/metadata for model config
    print("Loading processed graph data to extract final dims/metadata...")
    processed_graph_path = 'config/processed_graph_data.pt' # Or wherever you save it
    if not os.path.exists(processed_graph_path):
        raise FileNotFoundError(f"Processed graph not found at {processed_graph_path}. Run build_graph.py and process_graph.py first.")
    try:
        processed_graph_data_for_config = torch.load(processed_graph_path)
        print("Processed graph loaded for config.")
    except Exception as e:
        print(f"[ERROR] Failed loading processed graph for config: {e}")
        # Decide how to handle this: exit, use defaults, etc.
        return # Exit for now

    if args.use_gnn:
        print("Updating model config with GNN parameters from processed graph...")
        if 'graph_encoder_params' not in model_config: model_config['graph_encoder_params'] = {}
        try:
             model_config['graph_encoder_params']['metadata'] = processed_graph_data_for_config.metadata()
             # Get node feature dimensions after processing/scaling
             model_config['graph_encoder_params']['in_channels'] = {
                 ntype: store['x'].shape[1]
                 for ntype, store in processed_graph_data_for_config.node_items() if 'x' in store
             }
             # Get edge feature dimensions after processing/scaling
             model_config['graph_encoder_params']['edge_input_dims'] = {
                 etype: store['edge_attr'].shape[1]
                 for etype, store in processed_graph_data_for_config.edge_items() if 'edge_attr' in store
             }
        except Exception as e:
             print(f"[WARN] Could not extract all GNN parameters from processed graph: {e}. Model init might fail.")


    if args.use_sequence:
         if 'sequence_encoder_params' not in model_config: model_config['sequence_encoder_params'] = {}
         try:
             # Assuming sequence feature dim needed is stored/inferable from processed graph
             seq_dim = processed_graph_data_for_config['transaction'].seq_features.shape[-1]
             # Pass this to sequence model config if required
             # model_config['sequence_encoder_params']['input_dim'] = seq_dim # Example
             print(f"Sequence dimension from processed graph: {seq_dim}")
         except Exception as e:
             print(f"[WARN] Could not extract sequence dimension from processed graph: {e}")


    if args.use_text:
        if 'text_encoder_params' not in model_config: model_config['text_encoder_params'] = {}
        model_config['text_encoder_params']['finetune'] = False # Override, typically false for MAML
        if 'model_name' not in model_config['text_encoder_params']:
            # Get tokenizer name from args if possible, else default
            tokenizer_name = args.tokenizer_name if hasattr(args, 'tokenizer_name') else 'ProsusAI/finbert'
            model_config['text_encoder_params']['model_name'] = tokenizer_name

    if 'fusion_params' not in model_config: model_config['fusion_params'] = {}

    print(f"Final Model Config (before model init): {model_config}")

    print("Initializing AdvancedTransactionCategorizationModel for MAML...")
    maml_target_label = 'user' if args.user_label_col else 'global' # Should match DataModule target

    # Load the *actual* processed graph data to pass to the model
    # Re-load here or use the one loaded for config? Re-load for clarity.
    print("Loading PROCESSED graph data for MAML feature fetching...")
    processed_graph_path = 'config/processed_graph_data.pt' # Or wherever you save it
    if not os.path.exists(processed_graph_path):
        # This check might be redundant if loaded above, but safe
        raise FileNotFoundError(f"Processed graph not found at {processed_graph_path}. Run process_graph.py first.")
    processed_graph_data = torch.load(processed_graph_path)
    print("Processed graph loaded for model.")


    model = AdvancedTransactionCategorizationModel(
        model_config=model_config,
        learning_rate=args.meta_lr, # Use meta_lr for the outer loop optimizer
        weight_decay=args.weight_decay,
        focal_loss_alpha=args.focal_alpha,
        focal_loss_gamma=args.focal_gamma,
        use_maml=True,
        inner_lr=args.inner_lr,
        adaptation_steps=args.adaptation_steps,
        maml_head_label_type=maml_target_label,
        # Pass the loaded processed graph object
        full_data_ref=processed_graph_data
    )

    # --- Load Pre-trained Checkpoint (Manual, Non-Strict) ---
    if args.load_pretrained_ckpt:
        if os.path.exists(args.load_pretrained_ckpt):
            print(f"Loading weights MANUALLY from pre-trained checkpoint (strict=False): {args.load_pretrained_ckpt}")
            try:
                checkpoint = torch.load(args.load_pretrained_ckpt, map_location=model.device)
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
                if missing_keys: print(f"[WARN] Checkpoint load missing keys: {missing_keys}")
                if unexpected_keys: print(f"[INFO] Checkpoint load ignored unexpected keys: {unexpected_keys}")
                print("Manual non-strict weight loading finished.")
            except Exception as e:
                 print(f"[ERROR] Failed to manually load checkpoint {args.load_pretrained_ckpt}: {e}")
                 print("[WARN] Proceeding without loading pre-trained weights.")
        else:
            print(f"[WARN] Pre-trained checkpoint not found at: {args.load_pretrained_ckpt}. Starting from scratch.")


    # --- 4. Configure Trainer ( Largely unchanged ) ---
    print("Configuring PyTorch Lightning Trainer...")
    # Logger
    logger = None
    if args.logger == 'tensorboard':
        logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    elif args.logger == 'wandb':
        try:
             logger = WandbLogger(project=args.wandb_project, name=args.experiment_name, log_model=True)
        except ImportError:
             print("[WARN] wandb logger selected but wandb is not installed. Disabling logger. Install with: pip install wandb")
             logger = None
    print(f"Using logger: {'wandb' if logger else args.logger}")

    # Callbacks
    callbacks = []
    checkpoint_monitor_metric = f'val/meta_query_acc_{maml_target_label}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, args.experiment_name, 'checkpoints'),
        filename=f'{{epoch}}-{{step}}-{{{checkpoint_monitor_metric}:.4f}}',
        monitor=checkpoint_monitor_metric,
        mode='max',
        save_top_k=args.save_top_k,
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    if args.early_stopping_patience > 0:
        early_stop_monitor_metric = 'train/meta_outer_loss'
        early_stopping_callback = EarlyStopping(
            monitor=early_stop_monitor_metric,
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True,
            check_finite=True
        )
        callbacks.append(early_stopping_callback)

    print(f"Callbacks: {[type(cb).__name__ for cb in callbacks]}")

    # Trainer instance
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0 # Keep disabled for MAML
    )

    # --- 5. Start Meta-Training ---
    print("--- Starting Meta-Training --- ")
    try:
        trainer.fit(model, datamodule=maml_dm) # Pass the MAML datamodule
        print("--- Meta-Training Finished --- ")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 6. Start Meta-Testing (Optional) ---
    if args.run_test:
        print("--- Starting Meta-Testing --- ")
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            try:
                 test_model = AdvancedTransactionCategorizationModel.load_from_checkpoint(
                     best_model_path,
                     hparams_file=None,
                     # Provide essential arguments NOT saved in hparams, including the graph ref
                     full_data_ref=processed_graph_data # Pass the loaded processed graph again
                 )
                 maml_dm.setup('test')
                 trainer.test(test_model, datamodule=maml_dm)
            except Exception as e:
                 print(f"[ERROR] Failed to load best model or run test: {e}")
                 traceback.print_exc()
        else:
            print("[WARN] No best model checkpoint found or path invalid. Testing with last trained model.")
            try:
                 maml_dm.setup('test')
                 trainer.test(model, datamodule=maml_dm)
            except Exception as e:
                 print(f"[ERROR] Failed to run test with last model: {e}")
                 traceback.print_exc()
        print("--- Meta-Testing Finished --- ")
    else:
        print("Skipping meta-testing.")

    print("--- MAML Training Script Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML Training Script for Transaction Categorization (V2 - Refactored Data)')

    # --- Data Args ---
    parser.add_argument('--data_path', type=str, required=True, help='Path to the RAW transaction CSV file.')
    parser.add_argument('--graph_data_path', type=str, default='config/graph_data.pt', help='Path to the pre-built graph data file (.pt). Used for checks.')
    parser.add_argument('--user_id_col', type=str, default='user_id', help='Column name for user IDs (used for splitting).')
    parser.add_argument('--global_label_col', type=str, default='category_id', help='Column name for global category labels (if needed).')
    parser.add_argument('--user_label_col', type=str, default='user_category_id', help='Column name for user-specific category labels (MAML target).')

    # --- MAML DataModule Args ---
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support examples per task (K).')
    parser.add_argument('--q_query', type=int, default=10, help='Number of query examples per task (Q).')
    parser.add_argument('--meta_batch_size', type=int, default=16, help='Number of tasks per meta-batch.')
    parser.add_argument('--meta_val_ratio', type=float, default=0.15, help='Fraction of users for meta-validation.')
    parser.add_argument('--meta_test_ratio', type=float, default=0.15, help='Fraction of users for meta-testing.')

    # --- MAML Model Args ---
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Learning rate for the inner loop adaptation.')
    parser.add_argument('--adaptation_steps', type=int, default=1, help='Number of gradient steps in the inner loop.')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='Learning rate for the outer loop (meta-optimizer).')

    # --- Base Model Config Args ---
    parser.add_argument('--config_path', type=str, default='config/model_config.yaml', help='Path to YAML base model configuration file.')
    parser.add_argument('--use_gnn', action='store_true', help='Flag to enable GNN encoder.')
    parser.add_argument('--use_sequence', action='store_true', help='Flag to enable sequence encoder.')
    parser.add_argument('--use_text', action='store_true', help='Flag to enable text encoder.')
    parser.add_argument('--tokenizer_name', type=str, default='ProsusAI/finbert', help='HuggingFace tokenizer name (used if use_text=True).')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for meta-optimizer.')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')

    # --- Trainer Args ---
    parser.add_argument('--accelerator', type=str, default='auto', choices=['cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto'], help='Hardware accelerator.')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices or specific device IDs (e.g., "1", "0,1").')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of meta-training epochs.')
    parser.add_argument('--precision', type=str, default='32-true', help='Training precision (e.g., 32-true, 16-mixed).')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers.')
    parser.add_argument('--log_dir', type=str, default='./maml_logs', help='Directory for logs and checkpoints.')
    parser.add_argument('--experiment_name', type=str, default='maml_transaction_exp', help='Name for the experiment run.')
    parser.add_argument('--logger', type=str, default='tensorboard', choices=['tensorboard', 'wandb', 'none'], help='Logger to use.')
    parser.add_argument('--wandb_project', type=str, default='MAML-Transactions', help='WandB project name (if using wandb).')
    parser.add_argument('--save_top_k', type=int, default=1, help='Save top K model checkpoints based on monitor metric.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping. 0 to disable.')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='Log metrics every N training steps.')
    parser.add_argument('--run_test', action='store_true', help='Run testing phase after training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--load_pretrained_ckpt', type=str, default=None, help='Path to a pre-trained model checkpoint to initialize MAML.')

    args = parser.parse_args()

    # --- Add check for graph_data_path AND processed_graph_data.pt ---
    processed_graph_path = 'config/processed_graph_data.pt' # Define path
    # Check raw graph only if GNN is used (as a proxy for build step)
    if args.use_gnn and not os.path.exists(args.graph_data_path):
         print(f"[ERROR] --use_gnn is True, but the raw graph data file '{args.graph_data_path}' was not found.")
         print(f"Please run 'python build_graph.py --data_path {args.data_path} --output_path {args.graph_data_path}' first.")
         exit(1)
    # Check processed graph existence (needed by MAML model regardless of GNN flag?)
    if not os.path.exists(processed_graph_path):
         print(f"[ERROR] Processed graph data file '{processed_graph_path}' not found.")
         print(f"Please run 'python process_graph.py --raw_graph_path {args.graph_data_path} --output_path {processed_graph_path} [--prepare_sequences] [--prepare_text]' first.")
         exit(1) # Exit if processed graph is needed but not found

    main(args) 