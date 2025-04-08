import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for multiple modalities."""
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int, dropout: float):
        super().__init__()
        self.input_dims = input_dims
        self.modality_names = list(input_dims.keys())
        self.num_modalities = len(input_dims)
        
        if not self.modality_names:
            raise ValueError("AttentionFusion requires at least one modality in input_dims.")

        # Modality projections to common space
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim) 
            for name, dim in input_dims.items()
        })
        
        # Attention weights computation
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, 
        embeddings: Dict[str, torch.Tensor], 
        mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            embeddings: Dictionary mapping modality names to embeddings.
                        Keys MUST be a subset of self.modality_names used during init.
            mask: Optional modality masks.
        Returns:
            Tuple of (fused_representation, attention_weights).
        """
        projected_embeddings = []
        available_modalities = list(embeddings.keys())

        if not available_modalities:
            raise ValueError("AttentionFusion forward received empty embeddings dictionary.")

        # Project available embeddings
        for name in available_modalities:
            if name not in self.projections:
                # This shouldn't happen if TransactionClassifier builds modality_dims correctly
                raise KeyError(f"Modality '{name}' provided in embeddings but no projection layer found.")
            projected_embeddings.append(self.projections[name](embeddings[name]))
        
        # Stack projected embeddings: [batch_size, num_available_modalities, hidden_dim]
        stacked_embeddings = torch.stack(projected_embeddings, dim=1)
        
        # Compute attention scores
        attention_scores = self.attention_layer(stacked_embeddings) # [batch_size, num_available_modalities, 1]
        
        # --- Masking (Optional, based on `mask` input) ---
        # If you have masks indicating validity of entire modalities per sample:
        if mask is not None:
            # Build mask tensor matching attention_scores shape
            modality_mask = torch.stack([mask[name] for name in available_modalities], dim=1)
            # Unsqueeze to [batch_size, num_available_modalities, 1] for broadcasting
            modality_mask = modality_mask.unsqueeze(-1)
            attention_scores = attention_scores.masked_fill(modality_mask == 0, float('-inf'))
        # -----------------------------------------------

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1) # [batch_size, num_available_modalities, 1]
        
        # Apply attention
        fused = torch.sum(attention_weights * stacked_embeddings, dim=1) # [batch_size, hidden_dim]
        
        # Post-fusion processing
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        
        # Return attention weights matching the order of modalities in the input `embeddings` dict
        # Squeeze the last dimension: [batch_size, num_available_modalities]
        return fused, attention_weights.squeeze(-1)

class MultiTaskFusion(nn.Module):
    """Multi-task fusion with separate classification heads."""
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int,
        num_global_classes: int,
        num_user_classes: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Fusion layer
        self.fusion = AttentionFusion(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Global category classifier
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_global_classes)
        )
        
        # User category classifier
        self.user_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_user_classes)
        )
        
    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the multi-task fusion.
        
        Args:
            embeddings: Dictionary mapping modality names to their embeddings
            mask: Optional dictionary mapping modality names to attention masks
            
        Returns:
            Tuple of:
            - Global category logits [batch_size, num_global_classes]
            - User category logits [batch_size, num_user_classes]
            - Attention weights [batch_size, num_modalities]
        """
        # Fuse embeddings
        fused, attn_weights = self.fusion(embeddings, mask)
        
        # Get predictions
        global_logits = self.global_classifier(fused)
        user_logits = self.user_classifier(fused)
        
        return global_logits, user_logits, attn_weights

class GatingFusion(nn.Module):
    """Gating-based fusion with learnable gates for each modality."""
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dims = input_dims
        
        # Project each modality
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in input_dims.items()
        })
        
        # Gating networks
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for name in input_dims.keys()
        })
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the gating fusion.
        
        Args:
            embeddings: Dictionary mapping modality names to their embeddings
            mask: Optional dictionary mapping modality names to attention masks
            
        Returns:
            Tuple of:
            - Fused embedding [batch_size, hidden_dim]
            - Gate values [batch_size, num_modalities]
        """
        # Project each modality
        projected = {}
        for name, emb in embeddings.items():
            projected[name] = self.projections[name](emb)
        
        # Compute gates
        gates = {}
        for name, emb in projected.items():
            gates[name] = self.gates[name](emb)
        
        # Apply gates
        fused = torch.zeros_like(projected[list(projected.keys())[0]])
        for name, emb in projected.items():
            fused = fused + gates[name] * emb
        
        # Final projection
        fused = self.final_proj(fused)
        
        return fused, gates 