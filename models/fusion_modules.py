import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

class AttentionFusion(nn.Module):
    """ 
    Attention-based fusion using MLP attention weights over concatenated projections.
    Handles potentially missing modalities in the input dictionary.
    """
    def __init__(self, 
                 modality_dims: Dict[str, int], # Dict mapping modality name -> input dim
                 hidden_dim: int, 
                 output_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.modality_dims = modality_dims
        # Store modality names in a fixed order for consistent processing
        self.all_modality_names = sorted(list(modality_dims.keys())) 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if not self.all_modality_names:
            raise ValueError("AttentionFusion requires at least one modality in modality_dims.")

        # Modality projections
        self.projections = nn.ModuleDict()
        total_projected_dim = 0
        for name in self.all_modality_names:
            dim = modality_dims[name]
            self.projections[name] = nn.Linear(dim, hidden_dim)
            total_projected_dim += hidden_dim
        
        # Attention weights computation (MLP on concatenated projected features)
        self.attention_mlp = nn.Sequential(
            nn.Linear(total_projected_dim, hidden_dim), 
            nn.Tanh(),
            nn.Linear(hidden_dim, len(self.all_modality_names)), # Output one weight per *potential* modality
            # Use LogSoftmax + exp for numerical stability if needed, but Softmax is standard
            nn.Softmax(dim=-1) 
        )
        
        # Final processing layers
        self.dropout = nn.Dropout(dropout)
        self.final_proj = nn.Linear(hidden_dim, output_dim) # Weighted sum is hidden_dim
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, 
                embeddings: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            embeddings: Dictionary mapping modality names to embeddings.
                        Keys MUST be a subset of self.all_modality_names.
        Returns:
            Tuple of (fused_representation, attention_weights).
        """
        available_modalities = list(embeddings.keys())
        batch_size = -1
        device = None

        if not available_modalities:
            raise ValueError("AttentionFusion forward received empty embeddings dictionary.")

        # Determine batch size and device from the first available embedding
        first_key = available_modalities[0]
        batch_size = embeddings[first_key].shape[0]
        device = embeddings[first_key].device

        projected_embeddings_list = [] # List to store projected embeddings in fixed order
        present_mask = [] # Boolean mask indicating which modalities were present
        
        for name in self.all_modality_names:
            if name in embeddings:
                if embeddings[name].shape[0] != batch_size:
                     raise ValueError(f"Inconsistent batch sizes in fusion: { {k:v.shape[0] for k,v in embeddings.items()} }")
                projected = F.relu(self.projections[name](embeddings[name]))
                projected_embeddings_list.append(projected)
                present_mask.append(True)
            else:
                # Use placeholder zeros for missing modalities 
                projected_embeddings_list.append(torch.zeros(batch_size, self.hidden_dim, device=device))
                present_mask.append(False)

        # Concatenate projected embeddings (including placeholders)
        concat_embeds = torch.cat(projected_embeddings_list, dim=-1)
        
        # Calculate attention weights over all potential modalities
        # Shape: [batch_size, num_all_modalities]
        attn_weights_all = self.attention_mlp(concat_embeds) 
        
        # Weighted sum using projected embeddings (placeholders will have zero contribution)
        # Stack embeddings used for sum: [batch_size, num_all_modalities, hidden_dim]
        stacked_for_sum = torch.stack(projected_embeddings_list, dim=1)
        # Apply weights (unsqueeze for broadcasting): [batch_size, num_all_modalities, 1]
        # weighted_sum shape: [batch_size, hidden_dim]
        weighted_sum = torch.sum(stacked_for_sum * attn_weights_all.unsqueeze(-1), dim=1)

        # Final processing
        fused_representation = self.dropout(weighted_sum)
        fused_representation = self.final_proj(fused_representation) 
        fused_representation = self.layer_norm(fused_representation)

        # Return fused representation and the attention weights over ALL modalities
        return fused_representation, attn_weights_all 

# TODO: Implement GatingFusion and MultiTaskFusion if needed based on config 