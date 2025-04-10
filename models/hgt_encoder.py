import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear # Use PyG HGTConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple

class HGT(nn.Module):
    """ HGT Encoder based on torch_geometric.nn.HGTConv. """
    def __init__(self, 
                 in_channels: Dict[str, int], # Dict mapping node_type to input feature dim
                 hidden_channels: int, 
                 out_channels: int, 
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]], # (node_types, edge_types)
                 num_heads: int, 
                 num_layers: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.metadata = metadata
        node_types = metadata[0]

        # Initial linear projection for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            in_dim = in_channels.get(node_type, -1)
            if in_dim <= 0: # Handle cases where a node type might have no input features initially
                 print(f"[WARN] HGT: Input dimension for node type '{node_type}' is {in_dim}. Using LazyLinear.")
                 self.lin_dict[node_type] = Linear(-1, hidden_channels)
            else:
                 self.lin_dict[node_type] = Linear(in_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads)
            self.convs.append(conv)

        # Final linear projection only if out_channels differs
        self.out_lin_dict = nn.ModuleDict()
        if hidden_channels != out_channels:
             print(f"[INFO] HGT: Adding final projection from {hidden_channels} to {out_channels}")
             for node_type in node_types:
                 self.out_lin_dict[node_type] = Linear(hidden_channels, out_channels)
        else:
             for node_type in node_types:
                 self.out_lin_dict[node_type] = nn.Identity()


    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                # edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        
        # Get node types expected from metadata
        node_types_in_metadata = self.metadata[0]
        # Determine device from model parameters (safer than accessing before check)
        device = None 
        
        # Apply initial projection and activation, ensuring all types exist
        projected_x_dict = {}
        for node_type in node_types_in_metadata: 
            if node_type in x_dict and x_dict[node_type].numel() > 0: 
                 x = x_dict[node_type]
                 if device is None: device = x.device # Get device from first valid tensor
                 # TODO: Add Relative Temporal Encoding (RTE) here if needed
                 try:
                      projected_x = self.lin_dict[node_type](x).relu()
                      projected_x_dict[node_type] = F.dropout(projected_x, p=0.1, training=self.training)
                 except Exception as e_proj:
                      print(f"[ERROR] HGT initial projection failed for node type '{node_type}' with shape {x.shape}: {e_proj}")
                      raise e_proj
            else:
                 # If node type missing or empty in input batch, add placeholder 
                 # Get device from parameters if not yet set
                 if device is None: device = next(self.parameters()).device
                 projected_x_dict[node_type] = torch.empty((0, self.hidden_channels), device=device, dtype=torch.float) # Use float dtype

        x_dict_processed = projected_x_dict # Use this potentially completed dict

        # Apply HGT layers
        for i, conv in enumerate(self.convs):
            try:
                # Pass the dict containing placeholders
                x_dict_processed = conv(x_dict_processed, edge_index_dict)
                
                # Apply LayerNorm/Activation/Dropout between layers
                if i < self.num_layers - 1:
                    for node_type in x_dict_processed:
                         # Important: Apply to the *output* of the conv before next loop
                         x_dict_processed[node_type] = F.relu(x_dict_processed[node_type]) 
                         x_dict_processed[node_type] = F.dropout(x_dict_processed[node_type], p=0.1, training=self.training)
            except Exception as e:
                 print(f"[ERROR] HGTConv layer {i} failed: {e}")
                 print("x_dict shapes:", {k: v.shape for k, v in x_dict_processed.items()})
                 print("edge_index_dict keys:", list(edge_index_dict.keys()))
                 raise e

        # Apply final projection (optional)
        out_dict = {}
        for node_type, x in x_dict_processed.items(): # Use the final processed dict
             out_dict[node_type] = self.out_lin_dict[node_type](x)

        return out_dict 