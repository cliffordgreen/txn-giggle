import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, HeteroConv
from torch_geometric.nn.conv import MessagePassing
from typing import Dict, List, Optional, Tuple, Union

class HeteroGNNLayer(torch.nn.Module):
    """Heterogeneous GNN layer with attention and edge feature handling."""
    
    def __init__(self, in_channels: Dict[str, int], out_channels: int, edge_types: List[Tuple[str, str, str]], heads: int = 4):
        super().__init__()
        self.edge_types = edge_types
        self.heads = heads
        
        # Extract unique node types from edge types
        node_types = set()
        for src_type, _, dst_type in edge_types:
            node_types.add(src_type)
            node_types.add(dst_type)
        
        # Dimension per head
        dim_per_head = out_channels // heads
        
        # Input projections for each node type
        self.node_proj = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(in_channels[node_type], out_channels)
            for node_type in node_types
        })
        
        # Edge feature projections for each edge type
        self.edge_proj = torch.nn.ModuleDict({
            str(edge_type): torch.nn.Linear(1, out_channels)  # Edge features are 1-dimensional
            for edge_type in edge_types
        })
        
        # Attention layers for each edge type
        self.attention = torch.nn.ModuleDict({
            str(edge_type): torch.nn.MultiheadAttention(embed_dim=dim_per_head, num_heads=heads, batch_first=True)
            for edge_type in edge_types
        })
        
        # Output projections for each node type
        self.out_proj = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(out_channels, out_channels)
            for node_type in node_types
        })
        
        # Layer normalization
        self.norm = torch.nn.LayerNorm(out_channels)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with attention and edge features."""
        out_dict = {}
        
        # Initialize output for each node type
        for node_type in x_dict:
            out_dict[node_type] = torch.zeros_like(x_dict[node_type])
        
        # Project node features
        x_proj = {
            node_type: self.node_proj[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Process each edge type
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict or edge_type not in edge_attr_dict:
                continue  # Skip missing edge types
                
            src_type, edge_name, dst_type = edge_type
            edge_key = str(edge_type)
            edge_index = edge_index_dict[edge_type]
            edge_attr = edge_attr_dict[edge_type]
            
            # Project edge features
            edge_proj = self.edge_proj[edge_key](edge_attr)
            
            # Get source and target node features
            src = x_proj[src_type][edge_index[0]]
            dst = x_proj[dst_type][edge_index[1]]
            
            # Reshape for attention: [num_edges, heads, dim_per_head]
            dim_per_head = src.size(1) // self.heads
            src_reshaped = src.reshape(-1, self.heads, dim_per_head)
            dst_reshaped = dst.reshape(-1, self.heads, dim_per_head)
            edge_reshaped = edge_proj.reshape(-1, self.heads, dim_per_head)
            
            # Apply attention separately for each edge
            messages = []
            for i in range(src_reshaped.size(0)):
                # Add batch dimension [1, heads, dim_per_head]
                s = src_reshaped[i:i+1]
                d = dst_reshaped[i:i+1]
                e = edge_reshaped[i:i+1]
                
                # Compute attention
                attn_output, _ = self.attention[edge_key](
                    query=d,
                    key=s,
                    value=e
                )
                messages.append(attn_output)
                
            if messages:
                # Combine messages and reshape back
                messages = torch.cat(messages, dim=0)
                messages = messages.reshape(-1, src.size(1))
                
                # Aggregate messages
                for i, j in enumerate(edge_index[1]):
                    out_dict[dst_type][j] += messages[i]
        
        # Apply self-loops for node types without messages
        for node_type, x in x_dict.items():
            if torch.all(out_dict[node_type] == 0):
                out_dict[node_type] = x_proj[node_type]
        
        # Project outputs and apply normalization
        out_dict = {
            node_type: self.norm(self.out_proj[node_type](out))
            for node_type, out in out_dict.items()
        }
        
        return out_dict

class HeteroGNNEncoder(torch.nn.Module):
    """Heterogeneous GNN encoder with multiple layers."""
    
    def __init__(self, in_channels: Dict[str, int], hidden_channels: int, out_channels: int,
                 edge_types: List[Tuple[str, str, str]], num_layers: int = 3, heads: int = 4):
        super().__init__()
        self.num_layers = num_layers
        
        # Input projections
        self.input_proj = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(in_channels[node_type], hidden_channels)
            for node_type in in_channels
        })
        
        # GNN layers
        self.layers = torch.nn.ModuleList([
            HeteroGNNLayer(
                in_channels={node_type: hidden_channels for node_type in in_channels},
                out_channels=hidden_channels,
                edge_types=edge_types,
                heads=heads
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.out_proj = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(hidden_channels, out_channels)
            for node_type in in_channels
        })
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the GNN layers."""
        # Project input features
        x_dict = {
            node_type: self.input_proj[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Process through GNN layers
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
        
        # Project to output channels
        x_dict = {
            node_type: self.out_proj[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        return x_dict

class GNNPredictor(nn.Module):
    """GNN-based predictor that can be used for node classification."""
    def __init__(
        self,
        gnn: HeteroGNNEncoder,
        num_classes: int,
        hidden_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.gnn = gnn
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the GNN predictor.
        
        Args:
            x_dict: Dictionary mapping node types to node feature matrices
            edge_index_dict: Dictionary mapping edge types to edge indices
            edge_attr_dict: Dictionary mapping edge types to edge attributes
            return_embeddings: Whether to return embeddings along with logits
            
        Returns:
            If return_embeddings is True:
                Tuple of (embeddings [num_nodes, hidden_dim], logits [num_nodes, num_classes])
            Otherwise:
                Class logits [num_nodes, num_classes]
        """
        # Get node embeddings from GNN
        h_dict = self.gnn(x_dict, edge_index_dict, edge_attr_dict)
        
        # Only use transaction node embeddings for classification
        h = h_dict['transaction']
        logits = self.classifier(h)
        
        if return_embeddings:
            return h, logits
        return logits 