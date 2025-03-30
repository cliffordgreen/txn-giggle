import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models.transaction_classifier import TransactionClassifier, ModelConfig
from data.data_module import TransactionDataModule
from typing import Dict, Optional
import torch

def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess transaction data."""
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract temporal features
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour
    
    # Handle missing values
    df['description'] = df['description'].fillna('')
    df['memo'] = df['memo'].fillna('')
    df['merchant_name'] = df['merchant_name'].fillna('')
    
    return df

def train(
    data_path: str,
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    warmup_steps: int = 1000,
    max_seq_length: int = 50,
    graph_neighbors: Optional[Dict[str, int]] = None,
    text_model_name: str = 'bert-base-uncased',
    text_max_length: int = 128,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Train the transaction classifier."""
    # Set random seed
    pl.seed_everything(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Create data module
    data_module = TransactionDataModule(
        transactions_df=df,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_length=max_seq_length,
        graph_neighbors=graph_neighbors,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Create model
    model = TransactionClassifier(
        num_classes=df['category_id'].nunique(),
        gnn_hidden_channels=256,
        gnn_num_layers=3,
        gnn_heads=4,
        seq_hidden_size=256,
        seq_num_layers=2,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps
    )
    
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
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    test_results = trainer.test(model, data_module)
    
    # Save test results
    if test_results and len(test_results) > 0:
        # Convert test results to DataFrame
        results_df = pd.DataFrame([test_results[0]])
        results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
        
        # Get model predictions on test set
        model.eval()
        test_predictions = []
        test_dataloader = data_module.test_dataloader()
        
        with torch.no_grad():
            for batch in test_dataloader:
                global_logits, user_logits, _ = model(batch)
                global_preds = torch.argmax(global_logits, dim=1)
                
                # Get true labels
                true_labels = batch['labels']
                
                # Add predictions and true labels to list
                for i, pred in enumerate(global_preds):
                    test_predictions.append({
                        'batch_idx': i,
                        'predicted_category': pred.item(),
                        'true_category': true_labels[i].item()
                    })
        
        # Save predictions
        pred_df = pd.DataFrame(test_predictions)
        if len(pred_df) > 0:
            pred_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train transaction classifier')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to transaction data CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save model checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
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
        seed=args.seed
    ) 