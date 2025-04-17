import os
import argparse
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_forecasting.metrics import MAE # Use MAE for TFT init placeholder
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml # For loading potential YAML configs
from typing import Optional, Dict, List

# Use the V2 DataModule
from data.data_module_v2 import TransactionDataModuleV2, SingleBatchIterable 
# Import the new advanced model
from models.advanced_transaction_classifier import AdvancedTransactionCategorizationModel

# Helper function to load data (similar to old train.py)
def load_data(data_path: str) -> pd.DataFrame:
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    try:
        # Handle potential low_memory warning
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Data loaded successfully: {len(df)} records")
        # Basic preprocessing (handle missing essential columns)
        if 'user_id' not in df.columns:
            raise ValueError("Missing required column: 'user_id'")
        if 'category_id' not in df.columns:
             raise ValueError("Missing required column: 'category_id' (for global labels)")
        if 'user_category_id' not in df.columns:
             print("[WARN] 'user_category_id' column not found. User-specific task might fail.")
             # Optionally create a dummy column if needed downstream
             # df['user_category_id'] = df['category_id'] # Example: Use global as user 

        # Handle timestamp (use actual column name)
        timestamp_col = 'books_create_timestamp' # <<< Use correct column name
        if timestamp_col not in df.columns:
            # Try fallback 'posted_date'
            if 'posted_date' in df.columns:
                 timestamp_col = 'posted_date'
                 print(f"[INFO] Using fallback timestamp column '{timestamp_col}'")
            else:
                 print(f"[WARN] Timestamp columns ('books_create_timestamp', 'posted_date') not found. Using default date.")
                 df['timestamp'] = pd.Timestamp('2020-01-01') # Still create standard 'timestamp' col
                 timestamp_col = None # Mark that no original timestamp found

        if timestamp_col: # Process if a timestamp column was found
            print(f"Converting '{timestamp_col}' column to datetime...")
            # Create the standard 'timestamp' column from the source column
            df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
            if df['timestamp'].isnull().any():
                 print(f"[WARN] Coerced {df['timestamp'].isnull().sum()} invalid timestamps to NaT.")
                 median_date = df['timestamp'].dropna().median()
                 if pd.isna(median_date): median_date = pd.Timestamp('2020-01-01')
                 print(f"Filling NaT timestamps with {median_date}")
                 df['timestamp'].fillna(median_date, inplace=True)
            print("Timestamp conversion/handling complete.")
        
        # Extract time features from the standard 'timestamp' column
        if 'timestamp' in df.columns:
            df['weekday'] = df['timestamp'].dt.weekday
            df['hour'] = df['timestamp'].dt.hour
        else: # Create dummy columns if no timestamp
             df['weekday'] = 0
             df['hour'] = 0

        # Handle text fields (replace if different names are used)
        for col in ['raw_description', 'memo', 'merchant_name']:
            if col in df:
                df[col] = df[col].fillna('')
            else:
                print(f"[WARN] Text column '{col}' not found. Filling with empty strings.")
                df[col] = ''

    except Exception as e:
        print(f"Error loading or processing data from {data_path}: {e}")
        raise
    return df

def train_advanced(
    # Data/Output
    data_path: str,
    output_dir: str,
    # Basic Training Params
    batch_size: int = 32,
    num_workers: int = 0,
    max_epochs: int = 50,
    seed: int = 42,
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
    use_scheduleC_label: bool = False,
):
    """Train the advanced transaction classifier."""
    pl.seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data --- 
    df = load_data(data_path)

    # --- Calculate necessary dims from data --- 
    num_global_classes = df['category_id'].nunique()
    num_user_classes = df['user_category_id'].nunique() if 'user_category_id' in df else num_global_classes
    # Need num_users for embedding layer, get from processed df in DataModule init
    num_users = df['user_id'].nunique() # Estimate here, can be refined if needed
    print(f"Derived from data: #Global={num_global_classes}, #User={num_user_classes}, #Users={num_users}")
    
    # --- Data Module V2 --- 
    print("Initializing TransactionDataModuleV2...")
    # --- Get encoder flags from config --- 
    use_seq = model_config.get('use_sequence_encoder', True)
    use_graph = model_config.get('use_graph_encoder', True)
    use_text = model_config.get('use_text_encoder', True)

    num_hgt_layers = model_config['graph_encoder_params'].get('num_layers', 2) 
    data_module = TransactionDataModuleV2(
        transactions_df=df,
        batch_size=batch_size,
        num_workers=num_workers,
        num_hgt_layers=num_hgt_layers,
        hgt_num_samples=hgt_num_samples,
        text_model_name=model_config['text_encoder_params'].get('model_name', 'ProsusAI/finbert'),
        max_seq_length=model_config.get('max_seq_length', 50),
        text_max_length=model_config['text_encoder_params'].get('max_length', 128),
        # Pass flags to DataModule
        use_sequence_encoder=use_seq,
        use_gnn_encoder=use_graph,
        use_text_encoder=use_text,
        use_scheduleC_label=use_scheduleC_label
    )
    print("Setting up DataModuleV2...")
    data_module.setup('fit') 
    
    # --- Update Model Config with dynamic values AFTER setup --- 
    # Get class counts from DataModule instance
    num_global_classes = data_module.num_global_classes
    num_user_classes = data_module.num_user_classes
    num_users = data_module.num_users
    num_scheduleC_classes = data_module.num_scheduleC_classes
    print(f"DataModule counts: #Global={num_global_classes}, #User={num_user_classes}, #Users={num_users}, #SchedC={num_scheduleC_classes}")

    # Required by HGT
    model_config['graph_encoder_params']['in_channels'] = data_module.node_feature_dims
    model_config['graph_encoder_params']['metadata'] = data_module.full_graph_data.metadata() 
    # Required by User Embedding
    model_config['num_users'] = num_users
    # Required by Classifiers
    model_config['num_global_classes'] = num_global_classes
    model_config['num_user_classes'] = num_user_classes
    model_config['num_scheduleC_classes'] = num_scheduleC_classes
    # Update fusion dims based on actual encoder output dims
    model_config['fusion_params']['graph_dim'] = model_config['graph_encoder_params'].get('out_channels', 128)
    model_config['fusion_params']['seq_dim'] = model_config['sequence_encoder_params'].get('output_dim', 128)
    text_proj_dim = model_config['text_encoder_params'].get('projection_dim', 0)
    finbert_hidden_size = 768 # Placeholder - ideally get from loaded model config
    text_out_dim_for_fusion = text_proj_dim if text_proj_dim > 0 else finbert_hidden_size
    model_config['fusion_params']['text_dim'] = text_out_dim_for_fusion
    model_config['fusion_params']['user_dim'] = model_config['user_embed_dim']
    
    # --- Update TFT Categorical Config with dynamic cardinality ---
    # Only run if sequence encoder is enabled
    if use_seq:
        if (tft_params := model_config.get('sequence_encoder_params', {}).get('tft_params')) and \
           hasattr(data_module, 'num_regions') and 'region_id' in tft_params.get('categorical_groups', {}):
            print(f"[INFO] Updating TFT categorical_groups for 'region_id' with cardinality: {data_module.num_regions}")
            tft_params['categorical_groups']['region_id'] = data_module.num_regions
        elif tft_params:
            print("[WARN] Could not dynamically update TFT categorical_groups for 'region_id'. Check config and DataModule.")

    # --- Replace TFT loss placeholder from YAML with actual Metric instance --- 
    # Only run if sequence encoder is enabled
    if use_seq:
        if (tft_params := model_config.get('sequence_encoder_params', {}).get('tft_params')) and \
           tft_params.get('loss') == "MAE_placeholder":
            print("[INFO] Replacing TFT loss placeholder with MAE() instance.")
            tft_params['loss'] = MAE()
        elif tft_params:
            print(f"[WARN] TFT loss in config is not the placeholder string: {tft_params.get('loss')}")
    
    # --- Create Model --- 
    print("Initializing AdvancedTransactionCategorizationModel...")
    model = AdvancedTransactionCategorizationModel(
        model_config=model_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mtl_weights=mtl_weights,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        transactions_df_ref=df
    )
    # No longer needed if we fetch labels within _step from df/graph ref passed here
    
    # --- Callbacks & Logger --- 
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='adv_model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1, # Save only the best
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10, # Adjust as needed
            mode='min'
        )
    ]
    logger = TensorBoardLogger(save_dir=output_dir, name='adv_logs')

    # --- Trainer --- 
    print("Initializing Trainer...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision=precision,
        devices=1, # Assuming single device for now
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1000,
        val_check_interval=1000,# Log less frequently
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
        raise e

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
    parser.add_argument('--data_path', type=str, required=True, help='Path to transaction data CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints and logs')
    parser.add_argument('--config_path', type=str, default='config/model_config.yaml', help='Path to YAML model configuration file')

    # --- Training Arguments --- 
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers (set to 0 for debugging)')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--accelerator', type=str, default='auto', help='Trainer accelerator (cpu, gpu, auto)')
    parser.add_argument('--precision', type=str, default='32', help='Trainer precision (e.g., 16, 32, bf16)')
    
    # <<< HGT Sampler Args >>>
    # Allow passing sampling config via CLI (complex, maybe better in YAML)
    # Example: --hgt_samples "{'transaction':[15,10], 'user':[10,5]}"
    parser.add_argument('--hgt_samples', type=str, default=None, 
                        help='JSON/YAML string defining num_samples per node type per layer for HGTLoader')
    
    # --- MTL/Focal Loss Arguments --- 
    parser.add_argument('--mtl_weight_global', type=float, default=0.5, help='Weight for global loss')
    parser.add_argument('--mtl_weight_user', type=float, default=0.5, help='Weight for user loss')
    parser.add_argument('--mtl_weight_scheduleC', type=float, default=0.0, # New argument, default 0
                        help='Weight for Schedule C loss (only used if --use_scheduleC_label is set)')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Alpha for Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for Focal Loss')
    parser.add_argument('--use_scheduleC_label', action='store_true', # Use flag
                        help='Enable Schedule C classification task')

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

    # --- Construct MTL Weights Dictionary ---
    mtl_weights = {}
    if args.mtl_weight_global > 0:
        mtl_weights['global'] = args.mtl_weight_global
    if args.mtl_weight_user > 0:
        mtl_weights['user'] = args.mtl_weight_user
    # Add scheduleC weight only if the label is used AND weight > 0
    if args.use_scheduleC_label and args.mtl_weight_scheduleC > 0:
        mtl_weights['scheduleC'] = args.mtl_weight_scheduleC
    print(f"Using MTL weights: {mtl_weights}")
    if not mtl_weights:
        print("[WARN] No MTL weights > 0 specified. Model might not train effectively.")

    # --- Run Training --- 
    train_advanced(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        seed=args.seed,
        model_config=model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mtl_weights=mtl_weights,
        focal_loss_alpha=args.focal_alpha,
        focal_loss_gamma=args.focal_gamma,
        accelerator=args.accelerator,
        precision=args.precision,
        hgt_num_samples=hgt_samples_dict, # Pass parsed dict or None
        use_scheduleC_label=args.use_scheduleC_label,
    ) 