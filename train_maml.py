import torch
import pytorch_lightning as pl
import pandas as pd
import argparse
import os
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import your custom modules
from models.advanced_transaction_classifier import AdvancedTransactionCategorizationModel # Assuming MAML modifications
from data.maml_data_module import MAMLTransactionDataModule

def main(args):
    print("--- Starting MAML Training Script ---")
    print(f"Arguments: {args}")

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # --- 1. Load Data ---
    print(f"Loading transaction data from: {args.data_path}")
    try:
        transactions_df = pd.read_csv(args.data_path)
        # Optional: Add basic data validation here (e.g., check required columns)
        print(f"Loaded DataFrame with shape: {transactions_df.shape}")
    except FileNotFoundError:
        print(f"[ERROR] Data file not found at {args.data_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # --- 2. Initialize MAML DataModule ---
    print("Initializing MAMLTransactionDataModule...")
    maml_dm = MAMLTransactionDataModule(
        transactions_df=transactions_df,
        user_id_column=args.user_id_col,
        label_column=args.global_label_col, # Base label
        user_label_column=args.user_label_col, # User-specific label for MAML adaptation
        timestamp_column=args.timestamp_col,
        K_shot=args.k_shot,
        Q_query=args.q_query,
        meta_batch_size=args.meta_batch_size,
        num_workers=args.num_workers,
        meta_val_ratio=args.meta_val_ratio,
        meta_test_ratio=args.meta_test_ratio,
        seed=args.seed
    )
    # Run setup to split users and prepare datasets
    # Setup is implicitly called by the Trainer, but can be called manually:
    # maml_dm.setup()

    # --- 3. Initialize Model (Configured for MAML) ---
    # TODO: Define model_config based on your architecture needs
    # This needs careful construction based on the actual feature dimensions
    # derived from the data processing (e.g., in a full pipeline)
    # For now, using placeholders - replace with actual values.
    print("Configuring model...")

    # Placeholder: Get necessary dims/counts from DataModule after setup
    # These would typically be determined *after* processing the data fully
    # in a setup phase that calculates them (like in DataModuleV2).
    # Here, we'll use dummy values or rely on defaults in the model.
    # maml_dm.setup() # Call setup if needed to calculate these
    # num_users = maml_dm.num_users
    # num_user_classes = maml_dm.num_user_classes # Based on maml_label_column?
    # num_global_classes = maml_dm.num_global_classes
    # metadata = maml_dm.metadata # If graph features are involved

    # Example Placeholder Configuration (Replace with actual derived values)
    model_config = {
        'use_gnn_encoder': args.use_gnn,
        'use_sequence_encoder': args.use_sequence,
        'use_text_encoder': args.use_text,
        'num_users': 1000, # Placeholder - should come from data
        'user_embed_dim': 64,
        'num_global_classes': 20, # Placeholder
        'num_user_classes': 10, # Placeholder - Corresponds to user_label_col cardinality
        'graph_encoder_params': { # Only if use_gnn is True
            'in_channels': -1,
            'hidden_channels': 64,
            'out_channels': 64,
            'metadata': ([('transaction', 'merchant'), ('merchant', 'category'), ...], # Placeholder
                         [('transaction', 'to', 'merchant'), ...]), # Placeholder
            'num_heads': 4,
            'num_layers': 2
        },
        'sequence_encoder_params': { # Only if use_sequence is True
             'output_dim': 64,
             'tft_params': {
                # Required TFT params like time_idx, target, group_ids etc.
                # Need to be configured based on MAML data structure/features
             },
        },
        'text_encoder_params': { # Only if use_text is True
             'model_name': 'ProsusAI/finbert',
             'projection_dim': 64
        },
        'fusion_params': {
             'hidden_dim': 128,
             'output_dim': 128 # This feeds into the MAML head
        }
        # Add MAML specific flags IF they are part of model_config
        # 'use_maml': True # Better to pass directly as argument
    }

    print("Initializing AdvancedTransactionCategorizationModel for MAML...")
    model = AdvancedTransactionCategorizationModel(
        model_config=model_config,
        learning_rate=args.meta_lr, # Use meta_lr for the outer loop optimizer
        weight_decay=args.weight_decay,
        focal_loss_alpha=args.focal_alpha,
        focal_loss_gamma=args.focal_gamma,
        # Add MAML specific hyperparameters
        use_maml=True,
        inner_lr=args.inner_lr,
        adaptation_steps=args.adaptation_steps,
        maml_head_label_type='user' # Assuming MAML adapts the user_specific_head
    )

    # --- 4. Configure Trainer --- 
    print("Configuring PyTorch Lightning Trainer...")

    # Logger
    logger = None
    if args.logger == 'tensorboard':
        logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    elif args.logger == 'wandb':
        logger = WandbLogger(project=args.wandb_project, name=args.experiment_name, log_model=True)
        # Watch model - BE CAREFUL with MAML due to manual optimization
        # logger.watch(model, log='all', log_freq=100)
    print(f"Using logger: {args.logger}")

    # Callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, args.experiment_name, 'checkpoints'),
        filename='{epoch}-{step}-{val_meta_acc:.4f}', # Log meta-validation accuracy
        monitor='val/meta_query_acc', # Monitor meta-validation query accuracy
        mode='max',
        save_top_k=args.save_top_k,
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    if args.early_stopping_patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor='val/meta_query_acc', # Monitor meta-validation query accuracy
            patience=args.early_stopping_patience,
            mode='max',
            verbose=True
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
        # MAML requires manual optimization, so gradient clipping handled manually if needed
        # gradient_clip_val=args.grad_clip, 
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True, # Ensure reproducibility
        # Checkpointing is handled by the callback
        enable_checkpointing=True,
        # Progress bar
        enable_progress_bar=True
        # MAML requires manual optimization handled within the LightningModule
        # automatic_optimization=False # This should be set *inside* the LightningModule
    )

    # --- 5. Start Meta-Training ---
    print("--- Starting Meta-Training --- ")
    try:
        trainer.fit(model, datamodule=maml_dm)
        print("--- Meta-Training Finished --- ")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return # Exit if training fails

    # --- 6. Start Meta-Testing (Optional) ---
    if args.run_test:
        print("--- Starting Meta-Testing --- ")
        # Load best model checkpoint if needed
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            # Pass hparams_file=None if hyperparameters are saved in checkpoint
            # Ensure model class can load MAML hparams correctly
            test_model = AdvancedTransactionCategorizationModel.load_from_checkpoint(
                 best_model_path, 
                 # Pass necessary args if not saved in hparams
                 # model_config=model_config, 
                 # use_maml=True, inner_lr=args.inner_lr, ... 
                 hparams_file=None 
            )
            trainer.test(test_model, datamodule=maml_dm)
        else:
            print("[WARN] No best model checkpoint found. Testing with last trained model.")
            trainer.test(model, datamodule=maml_dm) # Test with the model state after fit
        print("--- Meta-Testing Finished --- ")
    else:
        print("Skipping meta-testing.")

    print("--- MAML Training Script Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML Training Script for Transaction Categorization')

    # --- Data Args ---
    parser.add_argument('--data_path', type=str, required=True, help='Path to the transaction CSV file.')
    parser.add_argument('--user_id_col', type=str, default='user_id', help='Column name for user IDs.')
    parser.add_argument('--global_label_col', type=str, default='category_id', help='Column name for global category labels.')
    parser.add_argument('--user_label_col', type=str, default='user_category_id', help='Column name for user-specific category labels (MAML target).')
    parser.add_argument('--timestamp_col', type=str, default='books_create_timestamp', help='Column name for timestamps.')

    # --- MAML DataModule Args ---
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support examples per task (K).')
    parser.add_argument('--q_query', type=int, default=10, help='Number of query examples per task (Q). -1 for all remaining.')
    parser.add_argument('--meta_batch_size', type=int, default=16, help='Number of tasks per meta-batch.')
    parser.add_argument('--meta_val_ratio', type=float, default=0.15, help='Fraction of users for meta-validation.')
    parser.add_argument('--meta_test_ratio', type=float, default=0.15, help='Fraction of users for meta-testing.')

    # --- MAML Model Args ---
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Learning rate for the inner loop adaptation.')
    parser.add_argument('--adaptation_steps', type=int, default=1, help='Number of gradient steps in the inner loop.')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='Learning rate for the outer loop (meta-optimizer).')

    # --- Base Model Config Args (Placeholders - adjust as needed) ---
    parser.add_argument('--use_gnn', action='store_true', help='Flag to enable GNN encoder.')
    parser.add_argument('--use_sequence', action='store_true', help='Flag to enable sequence encoder.')
    parser.add_argument('--use_text', action='store_true', help='Flag to enable text encoder.')
    # Add more args here to build model_config if not using a separate config file
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
    # parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient clipping value. 0 to disable. (Manual handling needed for MAML)')
    parser.add_argument('--run_test', action='store_true', help='Run testing phase after training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    main(args) 