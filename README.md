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