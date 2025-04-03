import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from torch.optim.lr_scheduler import OneCycleLR
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
        fusion_type: str = 'multi_task',  # Options: 'attention', 'gating', 'multi_task'
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
        criterion_weight = None
        if class_weights is not None:
            if not instance(classifier_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.regiester_buffer("criterio_weight_buffer",class_weights)
            criterion_weight = self.criterion_weight_buffer
            print(f"Registered criterion weights buffer on device:{self.criterion_weight_buffer.device}")
        else:
            pass
                
                            
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Loss function with class weights
    def forward(self, batch: Dict[str, Any], graph_data: 'HeteroData') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using full graph data and batch-specific inputs.

        Args:
            batch (Dict[str, Any]): Dictionary containing batch-specific data like
                                    'sequential_features', 'text_features', 'labels', 'batch_indices'.
            graph_data (HeteroData): The complete heterogeneous graph data object.

        Returns:
            Tuple of (global_logits, user_logits, fusion_weights)
        """
        # 1. Process GNN features using the FULL graph_data
        # The gnn_encoder needs the feature dict (x_dict) and edge dicts
        # Ensure graph_data contains the necessary keys ('x_dict', edge index/attr dicts)
        # Access node features via .x_dict, edge indices via .edge_index_dict etc.
        gnn_out_dict = self.gnn_encoder(
            graph_data.x_dict,
            graph_data.edge_index_dict,
            graph_data.edge_attr_dict  # Pass edge_attr_dict if your GNN uses it
        )
        # gnn_out_dict contains features for ALL nodes in the graph.
        # Select the features for the transaction nodes IN THIS BATCH.
        # batch['batch_indices'] holds the original indices for the transactions in this batch.
        batch_tx_indices = batch['batch_indices']
        gnn_features = gnn_out_dict['transaction'][batch_tx_indices] # Shape: [batch_size, gnn_hidden_channels]

        # 2. Process sequential features (remains the same)
        # Assuming 'sequential_features' and 'seq_lengths' are in the batch from collate_fn
        seq_output, seq_hidden, seq_attn = self.seq_encoder(
            batch['seq_features'], # Use the correct key from your collate_fn
            lengths=batch.get('seq_lengths')
        )
        seq_features = seq_attn # Or whichever output you use

        # 3. Process text features (remains the same)
        # Assuming 'text_features' dict with input_ids/attention_mask is in the batch
        # Adjust key access based on your collate_fn output structure
        # Example: If collate_fn returns {'text_features': {'desc_input_ids': ..., 'memo_input_ids': ...}}
        # You'll need to adapt how MultiFieldTextEncoder takes input or adjust collate_fn.
        # Assuming MultiFieldTextEncoder handles the dict input:
        text_features = self.text_encoder(batch['text_features']) # Key from collate_fn

        # 4. Fusion (remains mostly the same)
        embeddings = {
            'graph': gnn_features, # Now correctly sized [batch_size, dim]
            'sequence': seq_features,
            'text': text_features
        }

        if self._fusion_type == 'multi_task':
            global_logits, user_logits, fusion_weights = self.fusion_module(embeddings)
        else:
            fused_embeddings, fusion_weights = self.fusion_module(embeddings)
            global_logits = self.global_classifier(fused_embeddings)
            user_logits = self.user_classifier(fused_embeddings)

        return global_logits, user_logits, fusion_weights

    # MODIFY training_step, validation_step, test_step to get graph_data

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step with loss computation for multi-task learning."""
        # --- Get the full graph data ---
        # Access it via the trainer's datamodule attribute
        if not hasattr(self.trainer.datamodule, 'graph_data') or self.trainer.datamodule.graph_data is None:
             raise RuntimeError("Graph data not found in DataModule. Ensure setup() was run.")
        graph_data = self.trainer.datamodule.graph_data
        # ---

        # Pass graph_data to the forward method
        global_logits, user_logits, attention_weights = self(batch, graph_data)

        # Log modality attention weights (check if attention_weights is valid)
        if attention_weights is not None and attention_weights.numel() > 0 :
             avg_weights = {k: v.mean().item() for k, v in zip(self.modality_dims.keys(), attention_weights.mean(dim=0))}
             for key, value in avg_weights.items():
                 self.log(f'train_weight_{key}', value, on_step=False, on_epoch=True) # Log per epoch usually better

        # Compute losses
        # Use the correct label keys from your collate_fn
        global_loss = self.criterion(global_logits, batch['labels_global']) # Key from collate_fn

        total_loss = global_loss
        # Check for user-specific labels
        if 'labels_user' in batch:
            user_loss = self.criterion(user_logits, batch['labels_user']) # Key from collate_fn
            total_loss = total_loss + user_loss # Add user loss if needed/desired
            self.log('train_user_loss', user_loss, on_step=False, on_epoch=True)
            # Ensure user labels are on the same device and are Long type
            user_preds = user_logits.argmax(dim=-1)
            user_acc = (user_preds == batch['labels_user']).float().mean()
            self.log('train_user_accuracy', user_acc, on_step=False, on_epoch=True)

        # Log metrics
        global_preds = global_logits.argmax(dim=-1)
        global_acc = (global_preds == batch['labels_global']).float().mean()
        self.log('train_global_loss', global_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True) # Log total loss with prog bar
        self.log('train_global_accuracy', global_acc, on_step=True, on_epoch=True, prog_bar=True)

        return {'loss': total_loss}


    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step with metrics computation."""
        # --- Get the full graph data ---
        graph_data = self.trainer.datamodule.graph_data
        # ---

        # Pass graph_data to the forward method
        global_logits, user_logits, attention_weights = self(batch, graph_data)

        # Log modality attention weights
        if attention_weights is not None and attention_weights.numel() > 0 :
             avg_weights = {k: v.mean().item() for k, v in zip(self.modality_dims.keys(), attention_weights.mean(dim=0))}
             for key, value in avg_weights.items():
                 self.log(f'val_weight_{key}', value, on_step=False, on_epoch=True)

        # Compute losses
        global_loss = self.criterion(global_logits, batch['labels_global'])
        total_loss = global_loss
        if 'labels_user' in batch:
            user_loss = self.criterion(user_logits, batch['labels_user'])
            total_loss = total_loss + user_loss
            self.log('val_user_loss', user_loss, on_step=False, on_epoch=True)
            user_preds = user_logits.argmax(dim=-1)
            user_acc = (user_preds == batch['labels_user']).float().mean()
            self.log('val_user_accuracy', user_acc, on_step=False, on_epoch=True)

        # Log metrics
        global_preds = global_logits.argmax(dim=-1)
        global_acc = (global_preds == batch['labels_global']).float().mean()
        self.log('val_global_loss', global_loss, on_step=False, on_epoch=True)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_global_accuracy', global_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': total_loss} # Must return dictionary

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step with metrics computation."""
        # --- Get the full graph data ---
        graph_data = self.trainer.datamodule.graph_data
        # ---

        # Pass graph_data to the forward method
        global_logits, user_logits, attention_weights = self(batch, graph_data)

        # Log modality attention weights
        if attention_weights is not None and attention_weights.numel() > 0:
             avg_weights = {k: v.mean().item() for k, v in zip(self.modality_dims.keys(), attention_weights.mean(dim=0))}
             for key, value in avg_weights.items():
                 self.log(f'test_weight_{key}', value) # Logged once at the end by default

        # Compute losses
        global_loss = self.criterion(global_logits, batch['labels_global'])
        total_loss = global_loss
        if 'labels_user' in batch:
            user_loss = self.criterion(user_logits, batch['labels_user'])
            total_loss = total_loss + user_loss
            self.log('test_user_loss', user_loss)
            user_preds = user_logits.argmax(dim=-1)
            user_acc = (user_preds == batch['labels_user']).float().mean()
            self.log('test_user_accuracy', user_acc)

        # Log metrics
        global_preds = global_logits.argmax(dim=-1)
        global_acc = (global_preds == batch['labels_global']).float().mean()
        self.log('test_global_loss', global_loss)
        self.log('test_loss', total_loss)
        self.log('test_global_accuracy', global_acc)

        return {'test_loss': total_loss, 'test_accuracy': global_acc}
    # def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Forward pass with advanced multi-modal fusion.
        
    #     Returns:
    #         Tuple of (global_logits, user_logits, attention_weights)
    #     """

    #     print(f"Batch keys: {batch.keys()}")
    #     # Process GNN features
    #     gnn_out = self.gnn_encoder(
    #         batch['node_features'],
    #         batch['edge_index'],
    #         batch['edge_attr']
    #     )
    #     gnn_features = gnn_out['transaction']
        
    #     # Process sequential features with improved time delta handling
    #     seq_output, seq_hidden, seq_attn = self.seq_encoder(
    #         batch['sequential_features'],
    #         lengths=batch.get('seq_lengths')  # Use sequence lengths if available
    #     )
        
    #     # Use attention-weighted representation instead of just last hidden state
    #     # This captures important patterns across the entire sequence
    #     seq_features = seq_attn
        
    #     # Process text features
    #     text_features = self.text_encoder(batch['input_ids'])
        
    #     # Create modality embeddings dictionary
    #     embeddings = {
    #         'graph': gnn_features,
    #         'sequence': seq_features,
    #         'text': text_features
    #     }
        
    #     # Use the appropriate fusion mechanism based on the selected type
    #     if self._fusion_type == 'multi_task':
    #         # MultiTaskFusion returns global logits, user logits, and attention weights directly
    #         global_logits, user_logits, fusion_weights = self.fusion_module(embeddings)
    #     else:
    #         # AttentionFusion and GatingFusion return fused embeddings and weights
    #         fused_embeddings, fusion_weights = self.fusion_module(embeddings)
    #         # Apply classification heads
    #         global_logits = self.global_classifier(fused_embeddings)
    #         user_logits = self.user_classifier(fused_embeddings)
        
    #     return global_logits, user_logits, fusion_weights
    
    # def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
    #     """Training step with loss computation for multi-task learning."""
    #     # Forward pass to get logits for both tasks
    #     global_logits, user_logits, attention_weights = self(batch)
        
    #     # Log modality attention weights
    #     avg_weights = {k: v.mean().item() for k, v in zip(self.modality_dims.keys(), attention_weights.mean(dim=0))}
    #     for key, value in avg_weights.items():
    #         self.log(f'train_weight_{key}', value)
        
    #     # Compute losses for both tasks
    #     global_loss = self.criterion(global_logits, batch['labels'])
        
    #     # When user labels are available, add this loss component
    #     if 'user_labels' in batch:
    #         user_loss = self.criterion(user_logits, batch['user_labels'])
    #         total_loss = global_loss + user_loss
    #         self.log('train_user_loss', user_loss)
    #         self.log('train_user_accuracy', (user_logits.argmax(dim=-1) == batch['user_labels']).float().mean())
    #     else:
    #         total_loss = global_loss
        
    #     # Log metrics
    #     self.log('train_global_loss', global_loss)
    #     self.log('train_loss', total_loss)
    #     self.log('train_global_accuracy', (global_logits.argmax(dim=-1) == batch['labels']).float().mean())
        
    #     return {'loss': total_loss}
    
    # def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
    #     """Validation step with metrics computation for multi-task learning."""
    #     # Forward pass to get logits for both tasks
    #     global_logits, user_logits, attention_weights = self(batch)
        
    #     # Log modality attention weights
    #     avg_weights = {k: v.mean().item() for k, v in zip(self.modality_dims.keys(), attention_weights.mean(dim=0))}
    #     for key, value in avg_weights.items():
    #         self.log(f'val_weight_{key}', value)
        
    #     # Compute losses for both tasks
    #     global_loss = self.criterion(global_logits, batch['labels'])
        
    #     # When user labels are available, add this loss component
    #     if 'user_labels' in batch:
    #         user_loss = self.criterion(user_logits, batch['user_labels'])
    #         total_loss = global_loss + user_loss
    #         self.log('val_user_loss', user_loss)
    #         self.log('val_user_accuracy', (user_logits.argmax(dim=-1) == batch['user_labels']).float().mean())
    #     else:
    #         total_loss = global_loss
        
    #     # Log metrics
    #     self.log('val_global_loss', global_loss)
    #     self.log('val_loss', total_loss)
    #     self.log('val_global_accuracy', (global_logits.argmax(dim=-1) == batch['labels']).float().mean())
        
    #     return {'val_loss': total_loss}
    
    # def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
    #     """Test step with metrics computation for multi-task learning."""
    #     # Forward pass to get logits for both tasks
    #     global_logits, user_logits, attention_weights = self(batch)
        
    #     # Log modality attention weights
    #     avg_weights = {k: v.mean().item() for k, v in zip(self.modality_dims.keys(), attention_weights.mean(dim=0))}
    #     for key, value in avg_weights.items():
    #         self.log(f'test_weight_{key}', value)
        
    #     # Compute losses for both tasks
    #     global_loss = self.criterion(global_logits, batch['labels'])
        
    #     # When user labels are available, add this loss component
    #     if 'user_labels' in batch:
    #         user_loss = self.criterion(user_logits, batch['user_labels'])
    #         total_loss = global_loss + user_loss
    #         self.log('test_user_loss', user_loss)
    #         self.log('test_user_accuracy', (user_logits.argmax(dim=-1) == batch['user_labels']).float().mean())
    #     else:
    #         total_loss = global_loss
        
    #     # Compute accuracy
    #     global_accuracy = (global_logits.argmax(dim=-1) == batch['labels']).float().mean()
        
    #     # Log metrics
    #     self.log('test_global_loss', global_loss)
    #     self.log('test_loss', total_loss)
    #     self.log('test_global_accuracy', global_accuracy)
        
    #     return {
    #         'test_loss': total_loss,
    #         'test_accuracy': global_accuracy
    #     }
    
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

        total_steps = 0 # Check if the trainer attribute exists and is calculated 
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches: 
            # estimated_stepping_batches accounts for max_epochs, accumulation, etc. 
            total_steps = self.trainer.estimated_stepping_batches 
            print(f"[INFO] Using self.trainer.estimated_stepping_batches: {total_steps}") 
        else:
            print("[WARNING] self.trainer.estimated_stepping_batches not available or zero.") 
            print("[WARNING] Attempting manual calculation (less recommended for OneCycleLR).") 
            # Manual calculation as a fallback: 
            if not self.trainer.datamodule or not hasattr(self.trainer.datamodule, 'train_dataloader') or not self.trainer.datamodule.train_dataloader(): 
                raise RuntimeError("Train dataloader not available for optimizer setup fallback. Ensure datamodule.setup() was called before trainer.fit().")
            train_loader = self.trainer.datamodule.train_dataloader() 
            len_train_loader = len(train_loader) 
            accumulate_grad_batches = self.trainer.accumulate_grad_batches or 1 
            num_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None and self.trainer.max_epochs > 0 else 1 
            steps_per_epoch = len_train_loader // accumulate_grad_batches 
            if steps_per_epoch == 0: steps_per_epoch = 1 # Avoid zero division 
            total_steps = steps_per_epoch * num_epochs 
            print(f"[INFO] Manually calculated total_steps: {total_steps}") 
            # Override with max_steps if set 
            if self.trainer.max_steps is not None and self.trainer.max_steps > 0: 
                print(f"[INFO] Overriding total_steps with trainer.max_steps: {self.trainer.max_steps}") 
                total_steps = self.trainer.max_steps 
                
        if total_steps <= 0: 
            raise ValueError(f"Could not determine a valid number of total training steps ({total_steps}) for OneCycleLR. Check trainer configuration (max_epochs/max_steps) and ensure dataloader is not empty.") 
        # --- Setup OneCycleLR using total_steps --- # Create scheduler with appropriate number of parameter groups for max_lr 
        max_lr_list = [self.hparams.learning_rate * 0.1] # BERT # Calculate how many non-BERT groups were added 
        num_other_groups = len(param_groups) - 1 
        max_lr_list.extend([self.hparams.learning_rate] * num_other_groups) 
        # All other components 
        print(f"[INFO] Setting up OneCycleLR: total_steps={total_steps}, max_lr={max_lr_list}") 
        
        scheduler = OneCycleLR( 
            optimizer, 
            max_lr=max_lr_list, 
            total_steps=total_steps, # Use total_steps here 
            pct_start=self.hparams.get('pct_start', 0.1), # Use hparams.get for optional pct_start 
            # Add other relevant OneCycleLR parameters if needed: 
            # anneal_strategy='cos', 
            # div_factor=25.0,
            # final_div_factor=10000.0,
        ) 
        # # Create scheduler with appropriate number of parameter groups
        # max_lr = [self.hparams.learning_rate * 0.1]  # BERT
        # max_lr.extend([self.hparams.learning_rate] * (len(param_groups) - 1))  # All other components
        
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=max_lr,
        #     epochs=self.trainer.max_epochs,
        #     pct_start=0.1
        # )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        } 