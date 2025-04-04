# Transaction Classification Model

This project implements a multi-modal transaction classification model that leverages:
1. Graph structure of transactions and merchants
2. Sequential history of user transactions
3. Text descriptions and memos

## Key Features

### Fusion Mechanisms
The model supports three different fusion mechanisms to combine information from different modalities:
- **Multi-Task Fusion**: Trains separate classifiers for each modality and combines them with a weighted sum
- **Attention Fusion**: Uses an attention mechanism to dynamically weight the importance of each modality
- **Gating Fusion**: Uses gate values to control information flow from each modality

### Sequence Handling
- Enhanced time delta encoding to capture temporal patterns
- Attention-weighted representation of transaction history
- Normalization of large time gaps

### Graph Structure
- Heterogeneous graph with transaction, merchant, and category nodes
- Multiple edge types (belongs_to, categorized_as, temporal, similar_amount)
- Message passing via graph neural networks

### Text Processing
- Multi-field encoding of transaction descriptions and memos
- Pretrained language model for semantic understanding
- Field-specific weights to handle different text fields

## Results

The modality weights for different fusion approaches (based on synthetic data):

| Fusion Type | Graph Weight | Sequence Weight | Text Weight | Accuracy |
|-------------|--------------|----------------|-------------|----------|
| Multi-Task  | 0.1798       | 0.6588         | 0.1613      | 0.0000   |
| Attention   | 0.3847       | 0.3872         | 0.2281      | 0.0000   |
| Gating      | 0.4382       | 0.3980         | 0.7514      | 0.0417   |

*Note: These results are based on synthetic data and don't reflect real transaction patterns.*

## Future Work
- Train with real transaction data
- Add user personalization features
- Implement contrastive learning for better representations
- Support for more input modalities (location, device, etc.)

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