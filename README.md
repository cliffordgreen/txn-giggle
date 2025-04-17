# Transaction Classification System

A state-of-the-art multi-modal transaction classification model that predicts both global and user-specific categories for financial transactions. The system combines:

- Graph Neural Networks (GNN) for cross-transaction relationships
- Sequential models for per-user transaction patterns
- Transformer-based text encoding for transaction descriptions
- Multi-task learning for category prediction

## Features

- Heterogeneous graph modeling of transaction relationships
- Temporal pattern recognition via LSTM/GRU
- BERT-based text understanding
- Robust handling of missing data
- Multi-modal feature fusion
- Multi-task learning for category prediction

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `models/`: Core model implementations
  - `gnn.py`: Graph Neural Network components
  - `sequence.py`: Sequential model components
  - `text.py`: Text encoding components
  - `fusion.py`: Feature fusion and classification
  - `transaction_classifier.py`: Main model class

- `data/`: Data handling and preprocessing
  - `data_module.py`: PyTorch Lightning DataModule
  - `preprocessing.py`: Data preprocessing utilities

- `training/`: Training scripts and utilities
  - `train.py`: Main training script
  - `config.py`: Training configuration

## Usage

1. Prepare your transaction data in the required format
2. Configure training parameters in `training/config.py`
3. Run training:
```bash
python training/train.py
```

## Model Architecture

The model combines three main components:

1. **GNN Component**: Models relationships between transactions using a heterogeneous graph structure
2. **Sequential Model**: Captures temporal patterns in user transaction history
3. **Text Encoder**: Processes transaction descriptions using a transformer model

Features are fused using an attention mechanism and fed into multi-task classification heads.

## Performance

The model is designed to achieve high accuracy and F1 scores by leveraging:
- Cross-transaction relationships via graph structure
- User-specific temporal patterns
- Rich semantic information from text descriptions
- Multi-task learning for improved generalization

## License

MIT License

## Standard Training (Multi-Task Learning)

To train the model using the standard multi-task learning setup (predicting both global and user-specific categories simultaneously if configured), use the `train_new.py` script.

```bash
python train_new.py \
    --data_path path/to/your/transactions.csv \
    --output_dir ./standard_training_output \\
    --config_path config/model_config.yaml \\
    --batch_size 64 \\
    --max_epochs 100 \\
    --learning_rate 1e-4 \\
    --weight_decay 1e-5 \\
    --mtl_weight_global 0.5 \\
    --mtl_weight_user 0.5 \\
    # --use_scheduleC_label # Add this flag if schedule C task is desired
    # --mtl_weight_scheduleC 0.1 # Add weight if schedule C is used
    --accelerator gpu \\
    --precision 32 \\
    --num_workers 4 \
    # Add other relevant arguments from train_new.py --help
```

Refer to `train_new.py --help` for a full list of arguments and their descriptions. The model configuration (encoder details, fusion parameters, etc.) is primarily controlled via the YAML file specified by `--config_path`.


## Meta-Learning for Cold-Start Adaptation (MAML)

To address the user-specific cold-start problem (adapting the model quickly to new users with few transactions), this project implements Model-Agnostic Meta-Learning (MAML). The goal is to learn a model initialization (specifically for the user-specific classification head) that can be rapidly fine-tuned using a small support set (K-shot) of transactions from a new user.

The meta-learning process is handled by the `train_maml.py` script and the `data/maml_data_module.py` data loader.

### MAML Training Usage

The `train_maml.py` script orchestrates the meta-training process. It requires a dataset and MAML-specific hyperparameters.

```bash
python train_maml.py \\
    --data_path path/to/your/transactions.csv \\
    --user_id_col user_id \\
    --user_label_col user_category_id \\
    --timestamp_col books_create_timestamp \\
    --k_shot 5 \\
    --q_query 10 \\
    --meta_batch_size 16 \\
    --inner_lr 0.01 \\
    --adaptation_steps 1 \\
    --meta_lr 1e-4 \\
    --max_epochs 50 \\
    --accelerator gpu \\
    --devices 1 \\
    --log_dir ./maml_logs \\
    --experiment_name maml_training_run \\
    # Specify which base encoders to use if applicable (defaults may vary)
    # --use_sequence \\
    # --use_text \\
    # Add other relevant arguments from train_maml.py --help
```

**Key MAML Arguments:**

*   `--data_path`: Path to the transaction data (CSV format expected).
*   `--user_id_col`: Column containing unique user identifiers.
*   `--user_label_col`: Column containing the user-specific labels that the MAML head will adapt to (e.g., `user_category_id`).
*   `--k_shot`: Number of support examples per user task (K).
*   `--q_query`: Number of query examples per user task (Q). Used for calculating the meta-loss.
*   `--meta_batch_size`: Number of user tasks to process in each meta-batch.
*   `--inner_lr`: Learning rate for the inner loop adaptation steps.
*   `--adaptation_steps`: Number of gradient steps performed in the inner loop for each task.
*   `--meta_lr`: Learning rate for the outer loop meta-optimizer (updates the initial head parameters).

Refer to `train_maml.py --help` for all available arguments. Note that the base model configuration (encoders, fusion) is currently defined with placeholders within `train_maml.py` and should be adjusted based on your specific data and requirements. Ideally, this would load from a config file similar to the standard training script.

**Important:** MAML training requires the `learn2learn` library. Ensure it's installed (`pip install learn2learn`).


## Evaluation 