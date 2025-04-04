import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

from .gnn import HeteroGNNEncoder
from .sequence import SequenceEncoder
from .text import MultiFieldTextEncoder
from .fusion import MultiTaskFusion, AttentionFusion, GatingFusion

@dataclass
class ModelConfig:
    """Configuration for the transaction classifier model."""
    @dataclass
    class GNNConfig:
        input_dim: int
        hidden_dim: int
        output_dim: int
        num_layers: int
        dropout: float
        edge_types: list

    @dataclass
    class SequentialConfig:
        input_dim: int
        hidden_dim: int
        num_layers: int
        dropout: float

    @dataclass
    class TextConfig:
        model_name: str
        max_length: int
        hidden_dim: int
        dropout: float
        field_weights: Optional[Dict[str, float]] = None

    @dataclass
    class ClassifierConfig:
        hidden_dim: int
        dropout: float

    num_classes: int
    gnn: GNNConfig
    sequential: SequentialConfig
    text: TextConfig
    classifier: ClassifierConfig

class TransactionClassifier(pl.LightningModule):
    """Transaction classifier with multi-modal fusion."""
    
    def __init__(
        self,
        num_classes: int,
        gnn_hidden_channels: int = 256,
        gnn_num_layers: int = 3,
        gnn_heads: int = 4,
        seq_hidden_size: int = 256,
        seq_num_layers: int = 2,
        text_model_name: str = 'bert-base-uncased',
        text_max_length: int = 128,
        fusion_type: str = 'gating',  # Options: 'attention', 'gating', 'multi_task'
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 100,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # GNN encoder
        self.gnn_encoder = HeteroGNNEncoder(
            in_channels={
                'transaction': 6,  # amount, hour_sin, hour_cos, day_sin, day_cos, timestamp
                'merchant': 8,     # statistical features
                'category': 8      # statistical features
            },
            hidden_channels=gnn_hidden_channels,
            out_channels=gnn_hidden_channels,
            edge_types=[
                ('transaction', 'belongs_to', 'merchant'),
                ('merchant', 'categorized_as', 'category'),
                ('transaction', 'temporal', 'transaction'),
                ('transaction', 'similar_amount', 'transaction')
            ],
            num_layers=gnn_num_layers,
            heads=gnn_heads
        )
        
        # Sequential encoder
        self.seq_encoder = SequenceEncoder(
            input_dim=4,  # amount, timestamp, weekday, hour
            hidden_dim=seq_hidden_size,
            num_layers=seq_num_layers
        )
        
        # Text encoder
        self.text_encoder = MultiFieldTextEncoder(
            model_name=text_model_name,
            max_length=text_max_length
        )
        
        # Get text encoder hidden size
        text_hidden_size = 256  # Actual output size from the text encoder
        
        # Define input dimensions for each modality
        self.modality_dims = {
            'graph': gnn_hidden_channels,
            'sequence': seq_hidden_size,
            'text': text_hidden_size
        }
        
        # Create the appropriate fusion module based on the requested type
        if fusion_type == 'attention':
            self.fusion_module = AttentionFusion(
                input_dims=self.modality_dims,
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout
            )
            # Add classification heads for global and user categories
            self.global_classifier = nn.Linear(fusion_hidden_dim, num_classes)
            self.user_classifier = nn.Linear(fusion_hidden_dim, num_classes)
            self._fusion_type = 'attention'
            
        elif fusion_type == 'gating':
            self.fusion_module = GatingFusion(
                input_dims=self.modality_dims,
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout
            )
            # Add classification heads for global and user categories
            self.global_classifier = nn.Linear(fusion_hidden_dim, num_classes)
            self.user_classifier = nn.Linear(fusion_hidden_dim, num_classes)
            self._fusion_type = 'gating'
            
        else:  # Default to multi_task fusion
            self.fusion_module = MultiTaskFusion(
                input_dims=self.modality_dims,
                hidden_dim=fusion_hidden_dim,
                num_global_classes=num_classes,
                num_user_classes=num_classes,  # Temporarily set to same as global until we have user categories
                dropout=fusion_dropout
            )
            self._fusion_type = 'multi_task'
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with advanced multi-modal fusion.
        
        Returns:
            Tuple of (global_logits, user_logits, attention_weights)
        """
        # Process GNN features
        gnn_out = self.gnn_encoder(
            batch['node_features'],
            batch['edge_index'],
            batch['edge_attr']
        )
        gnn_features = gnn_out['transaction']
        
        # Process sequential features with improved time delta handling
        seq_output, seq_hidden, seq_attn = self.seq_encoder(
            batch['sequential_features'],
            lengths=batch.get('seq_lengths')  # Use sequence lengths if available
        )
        
        # Use attention-weighted representation instead of just last hidden state
        # This captures important patterns across the entire sequence
        seq_features = seq_attn
        
        # Process text features
        text_features = self.text_encoder(batch['input_ids'])
        
        # Create modality embeddings dictionary
        embeddings = {
            'graph': gnn_features,
            'sequence': seq_features,
            'text': text_features
        }
        
        # Use the appropriate fusion mechanism based on the selected type
        if self._fusion_type == 'multi_task':
            # MultiTaskFusion returns global logits, user logits, and attention weights directly
            global_logits, user_logits, fusion_weights = self.fusion_module(embeddings)
        else:
            # AttentionFusion and GatingFusion return fused embeddings and weights
            fused_embeddings, fusion_weights = self.fusion_module(embeddings)
            # Apply classification heads
            global_logits = self.global_classifier(fused_embeddings)
            user_logits = self.user_classifier(fused_embeddings)
        
        return global_logits, user_logits, fusion_weights
    
    def _log_modality_weights(self, attention_weights, phase='train'):
        """Log modality weights for analysis."""
        # Handle different return types from different fusion methods
        if isinstance(attention_weights, dict):
            # GatingFusion returns a dictionary of weights
            modality_weights = {k: v.mean().item() for k, v in attention_weights.items()}
            
            # Log each modality weight
            for modality, weight in modality_weights.items():
                self.log(f'{phase}_weight_{modality}', weight)
            
            # Convert to tensor for dominant modality calculation
            weight_values = torch.tensor([modality_weights[k] for k in self.modality_dims.keys()])
            dominant_idx = weight_values.argmax().item()
        else:
            # Average across the batch dimension for tensor weights
            avg_weights = attention_weights.mean(dim=0)
            
            # Get weights for each modality
            modality_weights = {k: v.item() for k, v in zip(self.modality_dims.keys(), avg_weights)}
            
            # Log each modality weight
            for modality, weight in modality_weights.items():
                self.log(f'{phase}_weight_{modality}', weight)
            
            # Log the dominant modality
            dominant_idx = avg_weights.argmax().item()
            
        # Log dominant modality index
        dominant_modality = list(self.modality_dims.keys())[dominant_idx]
        self.log(f'{phase}_dominant_modality', dominant_idx, prog_bar=True)
        
        return modality_weights
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step with multi-task learning."""
        # Forward pass to get logits for both tasks
        global_logits, user_logits, attention_weights = self(batch)
        
        # Log modality attention weights
        self._log_modality_weights(attention_weights, phase='train')
        
        # Compute losses for both tasks
        global_loss = self.criterion(global_logits, batch['labels'])
        
        # When user labels are available, add this loss component
        if 'user_labels' in batch:
            user_loss = self.criterion(user_logits, batch['user_labels'])
            total_loss = global_loss + user_loss
            self.log('train_user_loss', user_loss)
            self.log('train_user_accuracy', (user_logits.argmax(dim=-1) == batch['user_labels']).float().mean())
        else:
            total_loss = global_loss
        
        # Log metrics
        self.log('train_global_loss', global_loss)
        self.log('train_loss', total_loss)
        self.log('train_global_accuracy', (global_logits.argmax(dim=-1) == batch['labels']).float().mean())
        
        return {'loss': total_loss}
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step with metrics computation for multi-task learning."""
        # Forward pass to get logits for both tasks
        global_logits, user_logits, attention_weights = self(batch)
        
        # Log modality attention weights
        self._log_modality_weights(attention_weights, phase='val')
        
        # Compute losses for both tasks
        global_loss = self.criterion(global_logits, batch['labels'])
        
        # When user labels are available, add this loss component
        if 'user_labels' in batch:
            user_loss = self.criterion(user_logits, batch['user_labels'])
            total_loss = global_loss + user_loss
            self.log('val_user_loss', user_loss)
            self.log('val_user_accuracy', (user_logits.argmax(dim=-1) == batch['user_labels']).float().mean())
        else:
            total_loss = global_loss
        
        # Log metrics
        self.log('val_global_loss', global_loss)
        self.log('val_loss', total_loss)
        self.log('val_global_accuracy', (global_logits.argmax(dim=-1) == batch['labels']).float().mean())
        
        return {'val_loss': total_loss}
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step with metrics computation for multi-task learning."""
        # Forward pass to get logits for both tasks
        global_logits, user_logits, attention_weights = self(batch)
        
        # Log modality attention weights
        modality_weights = self._log_modality_weights(attention_weights, phase='test')
        
        # Compute losses for both tasks
        global_loss = self.criterion(global_logits, batch['labels'])
        
        # When user labels are available, add this loss component
        if 'user_labels' in batch:
            user_loss = self.criterion(user_logits, batch['user_labels'])
            total_loss = global_loss + user_loss
            self.log('test_user_loss', user_loss)
            self.log('test_user_accuracy', (user_logits.argmax(dim=-1) == batch['user_labels']).float().mean())
        else:
            total_loss = global_loss
        
        # Compute accuracy
        global_accuracy = (global_logits.argmax(dim=-1) == batch['labels']).float().mean()
        
        # Log metrics
        self.log('test_global_loss', global_loss)
        self.log('test_loss', total_loss)
        self.log('test_global_accuracy', global_accuracy)
        
        return {
            'test_loss': total_loss,
            'test_accuracy': global_accuracy
        }
    
    def train_dataloader(self):
        """Return the training dataloader."""
        return self.trainer.datamodule.train_dataloader()
    
    def val_dataloader(self):
        """Return the validation dataloader."""
        return self.trainer.datamodule.val_dataloader()
    
    def test_dataloader(self):
        """Return the test dataloader."""
        return self.trainer.datamodule.test_dataloader()
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer with warmup."""
        # Create optimizer with different learning rates for BERT and other parameters
        param_groups = [
            {'params': self.text_encoder.parameters(), 'lr': self.hparams.learning_rate * 0.1},
            {'params': self.gnn_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.seq_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.fusion_module.parameters(), 'lr': self.hparams.learning_rate},
        ]
        
        # Add classifier parameters based on fusion type
        if self._fusion_type == 'multi_task':
            # MultiTaskFusion has its own classifiers internally
            pass
        else:
            # Add separate classifier heads for attention and gating fusion
            param_groups.append({'params': self.global_classifier.parameters(), 'lr': self.hparams.learning_rate})
            param_groups.append({'params': self.user_classifier.parameters(), 'lr': self.hparams.learning_rate})
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        # Create scheduler with appropriate number of parameter groups
        max_lr = [self.hparams.learning_rate * 0.1]  # BERT
        max_lr.extend([self.hparams.learning_rate] * (len(param_groups) - 1))  # All other components
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.train_dataloader()),
            pct_start=0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        } 