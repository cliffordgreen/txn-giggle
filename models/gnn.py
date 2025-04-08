import torch
import torch.nn as nn
import torch.nn.functional as F
# Use standard PyG layers
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.nn import LayerNorm # Use PyG LayerNorm for graph data
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class HeteroGNNEncoder(torch.nn.Module):
    """Heterogeneous GNN encoder using HeteroConv and GATv2Conv layers."""
    
    def __init__(self, 
                 in_channels: Dict[str, int], 
                 edge_input_dims: Dict[Tuple[str, str, str], int], 
                 hidden_channels: int, 
                 out_channels: int,
                 edge_types: List[Tuple[str, str, str]], 
                 num_layers: int = 2, # Reduced default depth slightly
                 heads: int = 4, 
                 dropout: float = 0.1): # Added dropout argument
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(in_channels.keys())

        # --- Input Projections --- 
        # Project each node type features to the hidden dimension
        self.input_proj = nn.ModuleDict()
        for node_type, in_dim in in_channels.items():
            self.input_proj[node_type] = Linear(in_dim, hidden_channels)

        # --- HeteroConv Layers --- 
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() # LayerNorm for each layer's output
        for i in range(num_layers):
            # Determine input channels for this layer
            current_in_channels = hidden_channels
            is_last_layer = (i == num_layers - 1)
            concat_heads = not is_last_layer
            # Determine output channels for this layer's GATv2Convs
            # If concatenating heads, output per head is hidden_channels // heads
            # If not concatenating (last layer), output is the final out_channels
            current_out_channels_per_head = hidden_channels // heads if concat_heads else out_channels

            conv_dict = {}
            for edge_type in edge_types:
                src_type, _, dst_type = edge_type
                edge_dim = edge_input_dims.get(edge_type, -1) # Get edge dim, -1 if no features
                
                # Input for GATv2Conv can be tuple or single int
                gat_in_channels = (current_in_channels, current_in_channels) # Assuming hidden_dim for both src/dst
                
                conv_dict[edge_type] = GATv2Conv(
                    in_channels=gat_in_channels,
                    out_channels=current_out_channels_per_head, # Corrected output dim per head 
                    heads=heads,
                    concat=concat_heads, 
                    dropout=dropout,
                    edge_dim=edge_dim if edge_dim > 0 else None, # Pass edge_dim only if features exist
                    add_self_loops=False # Often handled by specific edge types or globally
                )

            # Create HeteroConv layer for this depth
            # Use sum aggregation by default
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            # LayerNorm dimension matches the output dim of the HeteroConv layer
            norm_dim = hidden_channels if concat_heads else out_channels
            self.norms.append(LayerNorm(norm_dim)) 

        # Note: No explicit output projection needed if last layer outputs `out_channels` directly
        # self.out_proj = nn.ModuleDict({nt: Linear(...) for nt in node_types}) if needed
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None
               ) -> Dict[str, torch.Tensor]:
        
        # 1. Apply initial projections
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.input_proj[node_type](x).relu()
            x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)

        # 2. Apply HeteroConv layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Prepare edge features for this layer
            current_edge_attr_dict = {}
            if edge_attr_dict:
                 for edge_type, layer in conv.convs.items():
                      if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and edge_type in edge_attr_dict:
                           current_edge_attr_dict[edge_type] = edge_attr_dict[edge_type]
            
            # Apply convolution
            # Residual connection might be tricky with changing dims/concat, skip for now
            x_dict_update = conv(x_dict, edge_index_dict, edge_attr_dict=current_edge_attr_dict)
            
            # Apply LayerNorm and Activation
            for node_type, x_update in x_dict_update.items():
                x_dict[node_type] = norm(x_update) 
                if i < self.num_layers - 1: # Apply activation for all but last layer
                   x_dict[node_type] = x_dict[node_type].relu()
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        
        # 3. Final Output (already projected if last layer had concat=False)
        return x_dict 