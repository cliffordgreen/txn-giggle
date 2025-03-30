import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from models.gnn import HeteroGNNEncoder, GNNPredictor
from models.text import MultiFieldTextEncoder
from models.sequence import SequenceEncoder
import pytorch_lightning as pl

class TransactionClassifier(pl.LightningModule):
    """Multi-modal transaction classifier using heterogeneous graph, text, and sequence data."""
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_sequence_layers: int = 2,
        text_model_name: str = 'bert-base-uncased',
        text_max_length: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize text encoder
        self.text_encoder = MultiFieldTextEncoder(
            model_name=text_model_name,
            max_length=text_max_length,
            dropout=dropout
        )
        
        # Initialize sequence encoder
        self.sequence_encoder = SequenceEncoder(
            input_dim=4,  # amount, timestamp, weekday, hour
            hidden_dim=hidden_dim,
            num_layers=num_sequence_layers,
            dropout=dropout
        )
        
        # Initialize GNN encoder
        self.gnn_encoder = HeteroGNNEncoder(
            input_channels_dict={
                'transaction': 4,  # amount, timestamp, weekday, hour
                'merchant': 5,     # mean, std, max, min, count
                'category': 5      # mean, std, max, min, count
            },
            hidden_channels=hidden_dim,
            num_layers=num_gnn_layers,
            edge_types=[
                ('transaction', 'belongs_to', 'merchant'),
                ('merchant', 'categorized_as', 'category'),
                ('transaction', 'temporal', 'transaction'),
                ('transaction', 'similar_amount', 'transaction')
            ],
            edge_feature_dims={
                ('transaction', 'belongs_to', 'merchant'): 1,  # Basic edge weight
                ('merchant', 'categorized_as', 'category'): 1,
                ('transaction', 'temporal', 'transaction'): 1,  # Time difference
                ('transaction', 'similar_amount', 'transaction'): 1  # Amount difference
            },
            dropout=dropout
        )
        
        # Initialize GNN predictor
        self.gnn_predictor = GNNPredictor(
            gnn=self.gnn_encoder,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Modality-specific attention
        self.modality_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
        text_features: torch.Tensor,
        seq_data: torch.Tensor,
        seq_length: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            graph_data: Dictionary containing graph data (node features, edge indices, edge attributes)
            text_features: Pre-computed text features
            seq_data: Sequence data tensor
            seq_length: Sequence length tensor
            
        Returns:
            Class logits
        """
        # Get sequence embeddings
        seq_output, seq_hidden = self.sequence_encoder(seq_data, seq_length)
        seq_embeddings = self.sequence_encoder.get_last_hidden(seq_hidden)
        
        # Get graph embeddings
        graph_embeddings, _ = self.gnn_predictor(
            graph_data['x_dict'],
            graph_data['edge_index_dict'],
            graph_data['edge_attr_dict'],
            return_embeddings=True
        )
        
        # Combine embeddings
        embeddings = torch.stack([
            text_features,
            seq_embeddings,
            graph_embeddings
        ], dim=1)  # [batch_size, 3, hidden_dim]
        
        # Apply modality attention
        attention_scores = self.modality_attention(embeddings)  # [batch_size, 3, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 3, 1]
        fused_embeddings = torch.sum(attention_weights * embeddings, dim=1)  # [batch_size, hidden_dim]
        
        # Final classification
        logits = self.classifier(fused_embeddings)
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Forward pass
        logits = self(
            batch['graph_data'],
            batch['text_features'],
            batch['seq_data'],
            batch['seq_length']
        )
        
        # Calculate loss (using both global and user-specific labels)
        loss_global = F.cross_entropy(logits, batch['labels_global'])
        loss_user = F.cross_entropy(logits, batch['labels_user'])
        loss = loss_global + loss_user
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_loss_global', loss_global)
        self.log('train_loss_user', loss_user)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Forward pass
        logits = self(
            batch['graph_data'],
            batch['text_features'],
            batch['seq_data'],
            batch['seq_length']
        )
        
        # Calculate loss
        loss_global = F.cross_entropy(logits, batch['labels_global'])
        loss_user = F.cross_entropy(logits, batch['labels_user'])
        loss = loss_global + loss_user
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc_global = (preds == batch['labels_global']).float().mean()
        acc_user = (preds == batch['labels_user']).float().mean()
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_loss_global', loss_global)
        self.log('val_loss_user', loss_user)
        self.log('val_acc_global', acc_global)
        self.log('val_acc_user', acc_user)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        # Forward pass
        logits = self(
            batch['graph_data'],
            batch['text_features'],
            batch['seq_data'],
            batch['seq_length']
        )
        
        # Calculate loss
        loss_global = F.cross_entropy(logits, batch['labels_global'])
        loss_user = F.cross_entropy(logits, batch['labels_user'])
        loss = loss_global + loss_user
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc_global = (preds == batch['labels_global']).float().mean()
        acc_user = (preds == batch['labels_user']).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_loss_global', loss_global)
        self.log('test_loss_user', loss_user)
        self.log('test_acc_global', acc_global)
        self.log('test_acc_user', acc_user)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        } 