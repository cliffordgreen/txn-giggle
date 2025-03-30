import os
import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models.transaction_classifier import TransactionClassifier
from data.data_module import TransactionDataModule
from typing import Dict, Optional
import json

def objective(
    trial: optuna.Trial,
    data_module: TransactionDataModule,
    output_dir: str,
    num_global_classes: int,
    num_user_classes: int
) -> float:
    """Objective function for hyperparameter optimization."""
    # Define hyperparameters to optimize
    gnn_hidden_dim = trial.suggest_int('gnn_hidden_dim', 64, 512)
    gnn_num_layers = trial.suggest_int('gnn_num_layers', 1, 5)
    gnn_dropout = trial.suggest_float('gnn_dropout', 0.0, 0.5)
    
    seq_hidden_dim = trial.suggest_int('seq_hidden_dim', 64, 512)
    seq_num_layers = trial.suggest_int('seq_num_layers', 1, 5)
    seq_dropout = trial.suggest_float('seq_dropout', 0.0, 0.5)
    
    fusion_hidden_dim = trial.suggest_int('fusion_hidden_dim', 128, 1024)
    fusion_dropout = trial.suggest_float('fusion_dropout', 0.0, 0.5)
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    warmup_steps = trial.suggest_int('warmup_steps', 100, 5000)
    
    # Create model
    model = TransactionClassifier(
        num_global_classes=num_global_classes,
        num_user_classes=num_user_classes,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_dropout=gnn_dropout,
        seq_hidden_dim=seq_hidden_dim,
        seq_num_layers=seq_num_layers,
        seq_dropout=seq_dropout,
        text_model_name='bert-base-uncased',
        text_max_length=128,
        text_field_weights={
            'description': 1.0,
            'memo': 0.5,
            'merchant': 0.3
        },
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename=f'trial-{trial.number}-{{val_loss:.2f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name=f'trial-{trial.number}'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=20,  # Reduced epochs for faster tuning
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Get validation loss
    val_loss = trainer.callback_metrics['val_loss']
    
    return val_loss

def tune_hyperparameters(
    data_path: str,
    output_dir: str,
    n_trials: int = 100,
    batch_size: int = 32,
    num_workers: int = 4,
    max_seq_length: int = 50,
    text_model_name: str = 'bert-base-uncased',
    text_max_length: int = 128,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Tune model hyperparameters using Optuna."""
    # Set random seed
    pl.seed_everything(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Create data module
    data_module = TransactionDataModule(
        transactions_df=df,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_length=max_seq_length,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Setup data module
    data_module.setup()
    
    # Create study
    study = optuna.create_study(direction='minimize')
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial=trial,
            data_module=data_module,
            output_dir=output_dir,
            num_global_classes=df['category_id'].nunique(),
            num_user_classes=df['user_category_id'].nunique()
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Save best parameters
    best_params = study.best_params
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Print results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(output_dir, 'optimization_history.html'))
    
    # Plot parameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(output_dir, 'param_importances.html'))
    
    # Plot parallel coordinate
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(os.path.join(output_dir, 'parallel_coordinate.html'))

def main(
    data_path: str,
    output_dir: str,
    n_trials: int = 100,
    batch_size: int = 32,
    num_workers: int = 4,
    max_seq_length: int = 50,
    text_model_name: str = 'bert-base-uncased',
    text_max_length: int = 128,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Main function for hyperparameter tuning."""
    tune_hyperparameters(
        data_path=data_path,
        output_dir=output_dir,
        n_trials=n_trials,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_length=max_seq_length,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune model hyperparameters')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to transaction data CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save tuning results')
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of trials for optimization')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
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
    
    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_length=args.max_seq_length,
        text_model_name=args.text_model_name,
        text_max_length=args.text_max_length,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    ) 