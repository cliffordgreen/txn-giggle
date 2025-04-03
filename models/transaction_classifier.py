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
    """
    Transaction classifier with multi-modal fusion, designed for use with
    PyG NeighborLoader (or similar graph samplers).
    """
    def __init__(
        self,
        num_classes: int,
        # --- GNN Params ---
        gnn_node_input_dims: Dict[str, int], # Input dim for EACH node type
        gnn_edge_input_dims: Optional[Dict[Tuple[str, str, str], int]] = None, # Optional edge feature dims
        gnn_hidden_channels: int = 256,
        gnn_out_channels: int = 256, # Output dim of GNN for 'transaction' nodes
        gnn_num_layers: int = 2, # Adjusted default, 3 can be deep for sampling
        gnn_heads: int = 4,
        # Edge types MUST match those in your HeteroData object
        gnn_edge_types: List[Tuple[str, str, str]] = [
            ('transaction', 'belongs_to', 'merchant'),
            ('merchant', 'categorized_as', 'category'),
            ('transaction', 'temporal', 'transaction'),
            ('transaction', 'similar_amount', 'transaction'),
            # ('merchant', 'rev_belongs_to', 'transaction'), # Add reverse edges if needed by GNNConv
            # ('category', 'rev_categorized_as', 'merchant')
         ],
        # --- Sequence Params ---
        seq_input_dim: int = 4, # amount, timestamp, weekday, hour
        seq_hidden_size: int = 256,
        seq_num_layers: int = 2,
        # --- Text Params ---
        text_model_name: str = 'bert-base-uncased', # Or a smaller one like 'distilbert-base-uncased'
        text_max_length: int = 128,
        text_out_dim: int = 256, # Desired output dimension after pooling/projection
        # --- Fusion Params ---
        fusion_type: str = 'multi_task', # Options: 'attention', 'gating', 'multi_task'
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        # --- Training Params ---
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        # pct_start for OneCycleLR scheduler (default 0.1 = 10% warmup)
        scheduler_pct_start: float = 0.1,
        class_weights: Optional[List[float]] = None # Expect list, will convert to tensor
    ):
        super().__init__()
        # Use save_hyperparameters() to automatically log hyperparameters
        # Note: Dictionaries and complex objects might not be saved correctly by default.
        # It's often better to pass simple types here if relying on automatic logging/loading.
        self.save_hyperparameters(ignore=['class_weights']) # Ignore weights from saving if they are large

        # GNN encoder
        # Ensure HeteroGNNEncoder can handle heterogeneous input/output channels
        # and process edge_attr_dict if gnn_edge_input_dims is provided
        self.gnn_encoder = HeteroGNNEncoder(
            in_channels=gnn_node_input_dims,
            hidden_channels=gnn_hidden_channels,
            out_channels=gnn_out_channels, # GNN output dim
            edge_types=gnn_edge_types,
            num_layers=gnn_num_layers,
            heads=gnn_heads,
            # Pass other relevant args like dropout, activation, etc.
        )

        # Sequential encoder
        self.seq_encoder = SequenceEncoder(
            input_dim=seq_input_dim,
            hidden_dim=seq_hidden_size,
            num_layers=seq_num_layers,
            # Pass other relevant args like dropout, bidirectional, etc.
        )

        # Text encoder
        self.text_encoder = MultiFieldTextEncoder(
            model_name=text_model_name,
            max_length=text_max_length,
            # Add an argument to specify output dimension if the base model's dim differs
            # output_dim=text_out_dim, # Assuming MultiFieldTextEncoder handles pooling/projection
        )
        # Get the actual text embedding dimension (might need adjustment)
        # This depends on how MultiFieldTextEncoder pools the transformer output.
        # If it uses CLS token of bert-base, it's 768. Let's assume it projects to text_out_dim.
        text_hidden_size = text_out_dim

        # Define input dimensions for each modality for fusion
        self.modality_dims = {
            'graph': gnn_out_channels, # Output dim from GNN for transaction nodes
            'sequence': seq_hidden_size, # Output dim from sequence encoder
            'text': text_hidden_size # Output dim from text encoder
        }
        self._fusion_type = fusion_type

        # Create the appropriate fusion module based on the requested type
        if fusion_type == 'attention':
            self.fusion_module = AttentionFusion(
                input_dims=self.modality_dims,
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout
            )
            # Add classification heads
            self.global_classifier = nn.Linear(fusion_hidden_dim, num_classes)
            # Assuming user categories are the same for now
            self.user_classifier = nn.Linear(fusion_hidden_dim, num_classes)

        elif fusion_type == 'gating':
            self.fusion_module = GatingFusion(
                input_dims=self.modality_dims,
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout
            )
             # Add classification heads
            self.global_classifier = nn.Linear(fusion_hidden_dim, num_classes)
            self.user_classifier = nn.Linear(fusion_hidden_dim, num_classes)

        else: # Default to multi_task fusion
            self.fusion_module = MultiTaskFusion(
                input_dims=self.modality_dims,
                hidden_dim=fusion_hidden_dim,
                num_global_classes=num_classes,
                # Temporarily set to same as global until we have user categories
                num_user_classes=num_classes,
                dropout=fusion_dropout
            )
            # MultiTaskFusion includes classifiers internally
            self.global_classifier = None
            self.user_classifier = None


        # Loss function with class weights (handle potential device placement)
        criterion_weight = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            # Register as buffer to ensure it moves with the model (e.g., .to(device))
            self.register_buffer("criterion_weight_buffer", weight_tensor, persistent=False)
            criterion_weight = self.criterion_weight_buffer
            print(f"Using criterion weights: {self.criterion_weight_buffer}")

        self.criterion = nn.CrossEntropyLoss(weight=criterion_weight)


    def forward(self, batch: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using a subgraph (`batch`) generated by NeighborLoader.

        Args:
            batch (HeteroData): A PyG HeteroData object representing the sampled
                                subgraph and containing features for nodes/edges
                                within that subgraph. It should also contain sequence
                                and text features corresponding to the 'seed'
                                transaction nodes, and their labels.

        Returns:
            Tuple of (global_logits, user_logits, fusion_weights)
        """
        # 1. Process GNN features using the SUBGRAPH batch
        # Pass the features and structure dictionaries directly from the batch
        gnn_out_dict = self.gnn_encoder(
            batch.x_dict,
            batch.edge_index_dict,
            getattr(batch, 'edge_attr_dict', None) # Pass edge_attr if available
        )

        # Extract features for the SEED 'transaction' nodes.
        # NeighborLoader typically puts the seed nodes first in the specified node type tensor.
        # The number of seed nodes corresponds to the batch size used in the loader.
        num_seed_nodes = batch['transaction'].batch_size # Get batch size from the subgraph metadata
        if 'transaction' not in gnn_out_dict:
             raise ValueError("GNN Encoder did not return embeddings for 'transaction' nodes.")
        if gnn_out_dict['transaction'].shape[0] < num_seed_nodes:
             raise ValueError(f"GNN output tensor for 'transaction' has fewer nodes ({gnn_out_dict['transaction'].shape[0]}) than the expected batch size ({num_seed_nodes}). Check NeighborLoader output.")

        gnn_features = gnn_out_dict['transaction'][:num_seed_nodes] # Shape: [batch_size, gnn_out_channels]


        # 2. Process sequential features
        # Assume 'seq_features' and maybe 'seq_lengths' are directly in the batch object,
        # aligned with the seed transaction nodes. Needs correct setup in DataModule/collate_fn.
        if 'seq_features' not in batch:
            raise KeyError("Batch object missing 'seq_features'. Ensure DataModule provides it.")

        # Ensure seq_features has the correct shape [batch_size, seq_len, feat_dim]
        seq_output, seq_hidden, seq_attn_repr = self.seq_encoder(
            batch.seq_features, # Shape: [batch_size, seq_len, seq_input_dim]
            lengths=getattr(batch, 'seq_lengths', None) # Optional: lengths if sequences are padded
        )
        # Use the attention-based/final representation from the sequence encoder
        seq_features = seq_attn_repr # Shape: [batch_size, seq_hidden_size]


        # 3. Process text features
        # Assume 'text_features' (a dict like {'field': {'input_ids': ..., 'mask': ...}})
        # is in the batch object, aligned with the seed transaction nodes.
        if 'text_features' not in batch:
            raise KeyError("Batch object missing 'text_features'. Ensure DataModule provides it.")

        text_features = self.text_encoder(batch.text_features) # Shape: [batch_size, text_out_dim]


        # 4. Fusion
        embeddings = {
            'graph': gnn_features,
            'sequence': seq_features,
            'text': text_features
        }

        global_logits, user_logits, fusion_weights = None, None, None
        if self._fusion_type == 'multi_task':
            global_logits, user_logits, fusion_weights = self.fusion_module(embeddings)
        elif self._fusion_type in ['attention', 'gating']:
            fused_embeddings, fusion_weights = self.fusion_module(embeddings)
            global_logits = self.global_classifier(fused_embeddings)
            user_logits = self.user_classifier(fused_embeddings)
        else:
            raise ValueError(f"Unsupported fusion_type: {self._fusion_type}")

        return global_logits, user_logits, fusion_weights

    def _common_step(self, batch: HeteroData, batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        """Common logic for training, validation, and test steps."""

        # Get predictions and fusion weights
        global_logits, user_logits, fusion_weights = self(batch)

        # --- Loss Calculation ---
        # Assume labels for seed nodes are stored in the batch object.
        # Common key from NeighborLoader output is batch[node_type].y
        # Adapt 'labels_global' and 'labels_user' based on your actual data structure.
        if 'y' not in batch['transaction']:
             raise KeyError("Batch object missing 'transaction.y' for labels. Ensure DataModule provides labels for seed nodes.")

        # Use the labels corresponding to the seed nodes
        labels_global = batch['transaction'].y[:batch['transaction'].batch_size]
        # Ensure labels are long type
        labels_global = labels_global.long()

        global_loss = self.criterion(global_logits, labels_global)
        total_loss = global_loss

        # --- Optional User-Specific Loss --- (Requires 'labels_user' in batch)
        user_loss = torch.tensor(0.0, device=self.device) # Default zero loss
        if 'labels_user' in batch: # Check if user labels exist in the batch
             labels_user = batch.labels_user # Assuming shape [batch_size]
             labels_user = labels_user.long()
             if user_logits is not None:
                 user_loss = self.criterion(user_logits, labels_user)
                 total_loss = total_loss + user_loss # Add user loss if computed
                 self.log(f'{stage}_user_loss', user_loss, on_step=(stage=='train'), on_epoch=True, batch_size=batch['transaction'].batch_size)
                 # Log user accuracy
                 user_preds = user_logits.argmax(dim=-1)
                 user_acc = (user_preds == labels_user).float().mean()
                 self.log(f'{stage}_user_accuracy', user_acc, on_step=False, on_epoch=True, batch_size=batch['transaction'].batch_size)


        # --- Logging ---
        # Log losses
        self.log(f'{stage}_global_loss', global_loss, on_step=(stage=='train'), on_epoch=True, batch_size=batch['transaction'].batch_size)
        self.log(f'{stage}_loss', total_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, batch_size=batch['transaction'].batch_size)

        # Log global accuracy
        global_preds = global_logits.argmax(dim=-1)
        global_acc = (global_preds == labels_global).float().mean()
        self.log(f'{stage}_global_accuracy', global_acc, on_step=(stage=='train'), on_epoch=True, prog_bar=True, batch_size=batch['transaction'].batch_size)

        # Log modality fusion weights (if available)
        if fusion_weights is not None and fusion_weights.numel() > 0 :
            # Ensure weights are detached and averaged correctly
            avg_weights = fusion_weights.detach().mean(dim=0)
            if avg_weights.numel() == len(self.modality_dims): # Check if weights correspond to modalities
                for i, key in enumerate(self.modality_dims.keys()):
                    self.log(f'{stage}_weight_{key}', avg_weights[i].item(), on_step=False, on_epoch=True, batch_size=batch['transaction'].batch_size)


        return {'loss': total_loss, f'{stage}_accuracy': global_acc}


    def training_step(self, batch: HeteroData, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step(batch, batch_idx, stage='train')

    def validation_step(self, batch: HeteroData, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step(batch, batch_idx, stage='val')

    def test_step(self, batch: HeteroData, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._common_step(batch, batch_idx, stage='test')

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer (AdamW) and LR scheduler (OneCycleLR)."""
        
        # --- Parameter Groups for Differential Learning Rates ---
        # Give text encoder parameters a potentially lower LR
        text_params = list(self.text_encoder.parameters())
        other_params = []
        other_params.extend(list(self.gnn_encoder.parameters()))
        other_params.extend(list(self.seq_encoder.parameters()))
        other_params.extend(list(self.fusion_module.parameters()))

        # Add separate classifier heads only if not using MultiTaskFusion
        if self.global_classifier is not None:
            other_params.extend(list(self.global_classifier.parameters()))
        if self.user_classifier is not None:
            other_params.extend(list(self.user_classifier.parameters()))

        # Filter out parameters that require gradients
        text_params = [p for p in text_params if p.requires_grad]
        other_params = [p for p in other_params if p.requires_grad]

        param_groups = []
        if text_params:
             param_groups.append({'params': text_params, 'lr': self.hparams.learning_rate * 0.1}) # 10% of base LR for text encoder
             print(f"Optimizer: Applying LR factor of 0.1 to {len(text_params)} text encoder parameters.")
        if other_params:
            param_groups.append({'params': other_params, 'lr': self.hparams.learning_rate}) # Base LR for others
            print(f"Optimizer: Applying base LR ({self.hparams.learning_rate}) to {len(other_params)} other parameters.")

        if not param_groups:
             raise ValueError("No parameters requiring gradients found for the optimizer.")
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)

        # --- OneCycleLR Scheduler ---
        # total_steps is calculated by Lightning Trainer. Accessing it early can be tricky.
        # Lightning >= 1.5 automatically calculates total_steps for OneCycleLR.
        # If using older versions, you might need the manual calculation, but prefer automatic.
        
        # Define max_lr for each parameter group
        max_lr_list = []
        if text_params:
             max_lr_list.append(self.hparams.learning_rate * 0.1)
        if other_params:
             max_lr_list.append(self.hparams.learning_rate)
             
        if not max_lr_list:
            raise ValueError("Cannot configure scheduler, max_lr_list is empty.")

        print(f"[INFO] Setting up OneCycleLR: max_lr={max_lr_list}, pct_start={self.hparams.scheduler_pct_start}")
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr_list,
            # total_steps will be inferred by Trainer, do not set explicitly unless needed for older PL versions
            # total_steps=self.trainer.estimated_stepping_batches, # Preferred way if available & needed
            pct_start=self.hparams.scheduler_pct_start,
            anneal_strategy='cos', # Common annealing strategy
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step', # Run scheduler every step
                'frequency': 1
            }
        }
