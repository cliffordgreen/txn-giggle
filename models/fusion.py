import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class AttentionFusion(nn.Module):
    """Attention-based fusion of multiple modalities."""
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dims = input_dims
        
        # Project each modality to same dimension
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in input_dims.items()
        })
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(input_dims)),
            nn.Softmax(dim=-1)
        )
        
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
        Forward pass of the attention fusion.
        
        Args:
            embeddings: Dictionary mapping modality names to their embeddings
            mask: Optional dictionary mapping modality names to attention masks
            
        Returns:
            Tuple of:
            - Fused embedding [batch_size, hidden_dim]
            - Attention weights [batch_size, num_modalities]
        """
        # Project each modality
        projected = {}
        for name, emb in embeddings.items():
            projected[name] = self.projections[name](emb)
        
        # Concatenate all projected embeddings
        concat = torch.cat(list(projected.values()), dim=-1)
        
        # Compute attention weights
        attn_weights = self.attention(concat)
        
        # Weighted sum
        fused = torch.zeros_like(projected[list(projected.keys())[0]])
        for i, (name, emb) in enumerate(projected.items()):
            fused = fused + attn_weights[:, i].unsqueeze(-1) * emb
        
        # Final projection
        fused = self.final_proj(fused)
        
        return fused, attn_weights

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