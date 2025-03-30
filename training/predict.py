import os
import pandas as pd
import torch
import pytorch_lightning as pl
from models.transaction_classifier import TransactionClassifier
from data.data_module import TransactionDataModule
from typing import Dict, List, Optional, Tuple
import numpy as np

def load_model(
    checkpoint_path: str,
    num_global_classes: int,
    num_user_classes: int,
    device: str = 'auto'
) -> TransactionClassifier:
    """Load trained model from checkpoint."""
    # Create model instance
    model = TransactionClassifier(
        # GNN parameters
        gnn_input_dim=4,  # amount, timestamp, weekday, hour
        gnn_hidden_dim=256,
        gnn_num_layers=3,
        gnn_edge_types=['same_merchant', 'same_company', 'similar_amount'],
        
        # Sequence parameters
        seq_input_dim=4,  # same as gnn_input_dim
        seq_hidden_dim=256,
        
        # Fusion parameters
        fusion_hidden_dim=512,
        num_global_classes=num_global_classes,
        num_user_classes=num_user_classes,
        
        # Optional parameters
        text_model_name='bert-base-uncased',
        text_max_length=128,
        dropout=0.2,
        learning_rate=1e-4
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model

def predict(
    model: TransactionClassifier,
    data_module: TransactionDataModule,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'auto'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make predictions on the dataset."""
    # Create trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        logger=False
    )
    
    # Create dataloader
    dataloader = data_module.test_dataloader()
    
    # Collect predictions
    global_preds = []
    user_preds = []
    global_probs = []
    user_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Get predictions
            outputs = model(batch)
            
            # Get predicted classes
            global_pred = outputs['global_logits'].argmax(dim=1)
            user_pred = outputs['user_logits'].argmax(dim=1)
            
            # Get probabilities
            global_prob = torch.softmax(outputs['global_logits'], dim=1)
            user_prob = torch.softmax(outputs['user_logits'], dim=1)
            
            # Append to lists
            global_preds.append(global_pred.cpu().numpy())
            user_preds.append(user_pred.cpu().numpy())
            global_probs.append(global_prob.cpu().numpy())
            user_probs.append(user_prob.cpu().numpy())
    
    # Concatenate predictions
    global_preds = np.concatenate(global_preds)
    user_preds = np.concatenate(user_preds)
    global_probs = np.concatenate(global_probs)
    user_probs = np.concatenate(user_probs)
    
    return global_preds, user_preds, global_probs, user_probs

def main(
    checkpoint_path: str,
    data_path: str,
    output_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'auto'
):
    """Main function for making predictions."""
    # Load data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create data module
    data_module = TransactionDataModule(
        transactions_df=df,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_length=50,
        text_model_name='bert-base-uncased',
        text_max_length=128,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Setup data module
    data_module.setup()
    
    # Load model
    model = load_model(
        checkpoint_path=checkpoint_path,
        num_global_classes=df['category_id'].nunique(),
        num_user_classes=df['user_category_id'].nunique(),
        device=device
    )
    
    # Make predictions
    global_preds, user_preds, global_probs, user_probs = predict(
        model=model,
        data_module=data_module,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'transaction_id': df['transaction_id'],
        'predicted_global_category': global_preds,
        'predicted_user_category': user_preds,
        'global_category_probabilities': list(global_probs),
        'user_category_probabilities': list(user_probs)
    })
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to transaction data CSV file')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for prediction')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use for prediction')
    
    args = parser.parse_args()
    
    main(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    ) 