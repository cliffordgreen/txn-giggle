import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import learn2learn as l2l
import logging
import traceback # Added for detailed error printing

# Import components from other files in the 'models' directory
from .hgt_encoder import HGT
from .tft_encoder import PytorchForecastingTFTWrapper
from .finbert_encoder import FinBERTEmbedder
from .fusion_modules import AttentionFusion # Assuming AttentionFusion is desired
from .losses import FocalLoss

# Assume HeteroData comes from torch_geometric
from torch_geometric.data import HeteroData

class AdvancedTransactionCategorizationModel(pl.LightningModule):
    """ Multi-modal transaction categorization using HGT, TFT, FinBERT,
        User Embeddings, Attention Fusion, Focal Loss, and MTL.
        Includes optional Schedule C prediction head.
        Includes MAML adaptation capability.
    """
    def __init__(self,
                 # Config Dictionaries
                 model_config: Dict[str, Any], # Contains keys like graph_encoder_params etc.
                 # Training Config
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 mtl_weights: Dict[str, float] = {'global': 0.5, 'user': 0.5}, # Can include 'scheduleC'
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 # --- MAML Specific Config ---
                 use_maml: bool = False, # Flag to enable MAML mode
                 inner_lr: float = 0.01, # Inner loop learning rate
                 adaptation_steps: int = 1, # K_adapt: Inner loop adaptation steps
                 maml_head_label_type: str = 'user', # Which head/label to adapt ('user' or 'global')
                 meta_lr: float = 1e-4, # <<< ADD meta_lr explicitly for saving >>>
                 # <<< MODIFIED: Expecting the FULL HeteroData graph object >>>
                 full_data_ref: Optional[HeteroData] = None,
                 ):
        super().__init__()
        # Store simple hyperparameters automatically
        # <<< FIX: Add meta_lr to hparams >>>
        self.save_hyperparameters('learning_rate', 'weight_decay',
                                'mtl_weights', 'focal_loss_alpha', 'focal_loss_gamma',
                                'use_maml', 'inner_lr', 'adaptation_steps', 'maml_head_label_type',
                                'meta_lr') # Save meta_lr

        # <<< Store flag reliably for runtime checks >>>
        # Set the runtime flag based on the init argument *after* hparams are saved/loaded
        self._use_maml_runtime_flag = use_maml
        # Don't necessarily overwrite hparams.use_maml here if loading from checkpoint
        # self.hparams.use_maml = use_maml

        # --- MAML Setup ---
        if self._use_maml_runtime_flag:
             print("[INFO] MAML Mode Enabled.")
             self.automatic_optimization = False # Essential for MAML's manual optimization
             # Check if the target MAML head exists
             maml_target_head_classes_key = f"num_{self.hparams.maml_head_label_type}_classes"
             if model_config.get(maml_target_head_classes_key, 0) == 0:
                  raise ValueError(f"MAML target is '{self.hparams.maml_head_label_type}', but {maml_target_head_classes_key} is 0 in model_config.")
             print(f"[INFO] MAML will adapt the '{self.hparams.maml_head_label_type}' head.")
             # Check if full_data_ref is provided for MAML feature fetching
             if full_data_ref is None:
                  print("[WARN] MAML mode enabled but full_data_ref (processed graph) was not provided. Feature fetching will fail.")
        else:
             print("[INFO] MAML Mode Disabled.")


        # Store complex configs manually
        self._model_config = model_config
        self._graph_config = model_config.get('graph_encoder_params', {}) # Handle missing key
        self._sequence_config = model_config.get('sequence_encoder_params', {})
        self._text_config = model_config.get('text_encoder_params', {})
        self._fusion_config = model_config.get('fusion_params', {})

        # Extract required counts/dims from the config for convenience
        num_global_classes = model_config.get('num_global_classes', 0)
        num_user_classes = model_config.get('num_user_classes', 0)
        num_users = model_config.get('num_users', 0)
        user_embed_dim = model_config.get('user_embed_dim', 64) # Provide default
        # New: Schedule C classes and head flag
        num_scheduleC_classes = model_config.get('num_scheduleC_classes', 0)
        self.use_scheduleC_head = num_scheduleC_classes > 0

        # Store references (NOT saved as hparams)
        self._full_data_ref = full_data_ref # Store reference for MAML feature fetching

        # <<< Store modality flags from config >>>
        self.use_gnn_encoder = model_config.get('use_gnn_encoder', True)
        self.use_sequence_encoder = model_config.get('use_sequence_encoder', True)
        self.use_text_encoder = model_config.get('use_text_encoder', True)

        # --- 1. Encoders ---
        self.graph_encoder = None
        graph_out_dim = 0
        if self.use_gnn_encoder:
             if not self._graph_config:
                 print("[WARN] GNN encoder enabled but 'graph_encoder_params' missing in config. Skipping GNN init.")
                 self.use_gnn_encoder = False # Disable if config missing
             elif 'metadata' not in self._graph_config:
                 print("[WARN] GNN encoder enabled but 'metadata' missing in graph_encoder_params. Skipping GNN init.")
                 self.use_gnn_encoder = False # Disable if metadata missing
             else:
                 hgt_metadata = self._graph_config['metadata']
                 print(f"[INFO] Initializing HGT with metadata: Nodes={hgt_metadata[0]}, Edges={hgt_metadata[1]}")
                 in_channels_config = self._graph_config.get('in_channels', -1)
                 self.graph_encoder = HGT(
                    in_channels=in_channels_config,
                    hidden_channels=self._graph_config.get('hidden_channels', 64),
                    out_channels=self._graph_config.get('out_channels', 64),
                    metadata=hgt_metadata,
                    num_heads=self._graph_config.get('num_heads', 4),
                    num_layers=self._graph_config.get('num_layers', 2)
                 )
                 graph_out_dim = self._graph_config['out_channels']
                 print(f"[INFO] HGT Encoder Initialized. Output Dim: {graph_out_dim}")
        else:
             print("[INFO] GNN Encoder is disabled via config.")

        self.sequence_encoder = None
        seq_out_dim = 0
        if self.use_sequence_encoder:
            if not self._sequence_config:
                 print("[WARN] Sequence encoder enabled but 'sequence_encoder_params' missing. Skipping init.")
                 self.use_sequence_encoder = False
            else:
                 self.sequence_encoder = PytorchForecastingTFTWrapper(
                    output_dim=self._sequence_config.get('output_dim', 64),
                    tft_params=self._sequence_config.get('tft_params', {}),
                    embedding_source_key=self._sequence_config.get('embedding_source_key', 'encoder_variables')
                 )
                 seq_out_dim = self.sequence_encoder.get_output_dim()
                 print(f"[INFO] TFT Wrapper Initialized. Output Dim: {seq_out_dim}")
        else:
             print("[INFO] Sequence Encoder is disabled via config.")

        self.text_encoder = None
        text_out_dim = 0
        if self.use_text_encoder:
             if not self._text_config:
                 print("[WARN] Text encoder enabled but 'text_encoder_params' missing. Skipping init.")
                 self.use_text_encoder = False
             else:
                 self.text_encoder = FinBERTEmbedder(
                    model_name=self._text_config.get('model_name', 'ProsusAI/finbert'),
                    pooling_strategy=self._text_config.get('pooling_strategy', 'mean'),
                    finetune=self._text_config.get('finetune', True),
                    projection_dim=self._text_config.get('projection_dim', 0)
                 )
                 text_out_dim = self.text_encoder.get_output_dim()
                 print(f"[INFO] FinBERT Encoder Initialized. Output Dim: {text_out_dim}")
        else:
             print("[INFO] Text Encoder is disabled via config.")

        self.user_embedding = None
        if num_users > 0 and user_embed_dim > 0:
             self.user_embedding = nn.Embedding(num_users, user_embed_dim)
             print(f"[INFO] User Embedding Initialized. Num Users: {num_users}, Output Dim: {user_embed_dim}")
        else:
             print(f"[INFO] User Embedding skipped (num_users={num_users}, user_embed_dim={user_embed_dim}).")
             user_embed_dim = 0

        # --- 2. Fusion Module ---
        fusion_input_dims = {}
        if self.use_gnn_encoder and self.graph_encoder: fusion_input_dims['graph'] = graph_out_dim
        if self.use_sequence_encoder and self.sequence_encoder: fusion_input_dims['sequence'] = seq_out_dim
        if self.use_text_encoder and self.text_encoder: fusion_input_dims['text'] = text_out_dim
        if self.user_embedding: fusion_input_dims['user'] = user_embed_dim

        if not fusion_input_dims: raise ValueError("No modalities enabled or configured correctly.")
        if not self._fusion_config: raise ValueError("'fusion_params' missing from model_config.")

        fusion_output_dim = self._fusion_config.get('output_dim', 128)
        self.fusion_module = AttentionFusion(
            modality_dims=fusion_input_dims,
            hidden_dim=self._fusion_config.get('hidden_dim', 128),
            output_dim=fusion_output_dim,
            dropout=self._fusion_config.get('dropout', 0.1)
        )
        print(f"[INFO] Attention Fusion Initialized. Input Dims: {fusion_input_dims}, Output Dim: {fusion_output_dim}")

        # --- 3. Classification Heads ---
        self.global_head = None
        if num_global_classes > 0:
             self.global_head = nn.Linear(fusion_output_dim, num_global_classes)
             print(f"[INFO] Global Classifier Initialized: Output Classes={num_global_classes}")
        else: print("[INFO] Global classification head skipped (num_global_classes=0).")

        self.user_specific_head = None
        if num_user_classes > 0:
             self.user_specific_head = nn.Linear(fusion_output_dim, num_user_classes)
             print(f"[INFO] User Classifier Initialized: Output Classes={num_user_classes}")
        else: print("[INFO] User classification head skipped (num_user_classes=0).")

        self.scheduleC_head = None
        if self.use_scheduleC_head:
             self.scheduleC_head = nn.Linear(fusion_output_dim, num_scheduleC_classes)
             print(f"[INFO] Schedule C Classifier Initialized: Output Classes={num_scheduleC_classes}")
        else: print("[INFO] Schedule C classification head skipped (num_scheduleC_classes=0).")

        # --- 4. Loss Functions ---
        self.focal_loss_global = None
        if self.global_head:
             self.focal_loss_global = FocalLoss(alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma, num_classes=num_global_classes)
             print("[INFO] Global Focal Loss Initialized.")
        self.focal_loss_user = None
        if self.user_specific_head:
             self.focal_loss_user = FocalLoss(alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma, num_classes=num_user_classes)
             print("[INFO] User Focal Loss Initialized.")
        self.focal_loss_scheduleC = None
        if self.scheduleC_head:
             self.focal_loss_scheduleC = FocalLoss(alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma, num_classes=num_scheduleC_classes)
             print("[INFO] Schedule C Focal Loss Initialized.")

    # --- Forward Pass (No MAML logic here) ---
    def forward(self,
                graph_batch: Optional[HeteroData] = None,
                sequence_batch: Optional[Any] = None,
                text_batch: Optional[List[str]] = None,
                user_ids: Optional[torch.Tensor] = None,
                batch_size: Optional[int] = None
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        embeddings_to_fuse = {}
        graph_embed = None
        if self.graph_encoder and graph_batch:
            try:
                node_embeddings_dict = self.graph_encoder(graph_batch.x_dict, graph_batch.edge_index_dict)
                if 'transaction' in node_embeddings_dict:
                     if batch_size is None:
                          if user_ids is not None: batch_size = user_ids.shape[0]
                          elif text_batch is not None: batch_size = len(text_batch)
                          elif hasattr(graph_batch['transaction'], 'input_id'): batch_size = graph_batch['transaction'].input_id.shape[0]
                          elif hasattr(graph_batch['transaction'], 'batch_size'): batch_size = graph_batch['transaction'].batch_size
                          else: batch_size = graph_batch['transaction'].num_nodes
                     num_nodes_in_output = node_embeddings_dict['transaction'].shape[0]
                     slice_len = min(batch_size, num_nodes_in_output) if batch_size is not None else num_nodes_in_output
                     if slice_len > 0: graph_embed = node_embeddings_dict['transaction'][:slice_len]
                     if graph_embed is not None: embeddings_to_fuse['graph'] = graph_embed.to(self.device)
                     elif batch_size > 0: print("[WARN] Forward: HGT 'transaction' embedding is None or empty after slicing.")
                else: print("[WARN] Forward: HGT output missing 'transaction' embeddings.")
            except Exception as e: print(f"[ERROR] HGT Encoder forward failed: {e}"); traceback.print_exc()
        if self.sequence_encoder and sequence_batch is not None:
            try:
                seq_embed = self.sequence_encoder(sequence_batch, device=self.device)
                if seq_embed is not None and seq_embed.shape[0] > 0: embeddings_to_fuse['sequence'] = seq_embed.to(self.device)
                elif batch_size is not None and batch_size > 0: print("[WARN] Forward: TFT output is None or empty.")
            except Exception as e: print(f"[ERROR] TFT Encoder forward failed: {e}"); traceback.print_exc()
        if self.text_encoder and text_batch is not None:
            try:
                if text_batch:
                     text_embed = self.text_encoder(text_batch)
                     if text_embed is not None and text_embed.shape[0] > 0: embeddings_to_fuse['text'] = text_embed.to(self.device)
                     elif batch_size is not None and batch_size > 0: print("[WARN] Forward: FinBERT output is None or empty.")
                else: print("[WARN] Forward: Received empty text_batch list.")
            except Exception as e: print(f"[ERROR] FinBERT Encoder forward failed: {e}"); traceback.print_exc()
        if self.user_embedding and user_ids is not None:
             try:
                 user_ids_clamped = torch.clamp(user_ids, 0, self.user_embedding.num_embeddings - 1)
                 user_embed = self.user_embedding(user_ids_clamped.to(self.device))
                 if user_embed is not None and user_embed.shape[0] > 0: embeddings_to_fuse['user'] = user_embed.to(self.device)
                 elif batch_size is not None and batch_size > 0: print("[WARN] Forward: User embedding is None or empty.")
             except Exception as e: print(f"[ERROR] User Embedding forward failed: {e}"); traceback.print_exc()
        if not embeddings_to_fuse: print("[ERROR] Forward: No embeddings available for fusion."); return None, None, None
        ref_batch_size = next(iter(embeddings_to_fuse.values())).shape[0]
        if ref_batch_size == 0: print("[WARN] Forward: Fusion input batch size is 0."); return None, None, None
        for name, emb in embeddings_to_fuse.items():
             if emb.shape[0] != ref_batch_size: print(f"[ERROR] Forward: Mismatched batch size for {name}."); return None, None, None
             if emb.device != self.device: print(f"[ERROR] Forward: Embedding {name} on wrong device."); return None, None, None
        fused_representation = None
        try:
            fused_representation, _ = self.fusion_module(embeddings_to_fuse)
        except Exception as e: print(f"[ERROR] Fusion module forward failed: {e}"); traceback.print_exc(); return None, None, None
        global_logits, user_specific_logits, scheduleC_logits = None, None, None
        if fused_representation is None: print("[ERROR] Forward: Fused representation is None."); return None, None, None
        try:
            if self.global_head: global_logits = self.global_head(fused_representation)
            if self.user_specific_head: user_specific_logits = self.user_specific_head(fused_representation)
            if self.scheduleC_head: scheduleC_logits = self.scheduleC_head(fused_representation)
        except Exception as e: print(f"[ERROR] Classifier head forward failed: {e}"); traceback.print_exc()
        return global_logits, user_specific_logits, scheduleC_logits


    # --- MAML Specific Methods ---
    def _get_features_for_indices(self, transaction_indices: torch.Tensor
                                 ) -> Tuple[Optional[HeteroData], Optional[Any], Optional[List[str]], Optional[torch.Tensor]]:
        """
        Fetches input features for specific transaction indices from self._full_data_ref.
        Performs k-hop subgraph sampling for GNN if enabled.
        Retrieves sequence and text data for the specified indices.
        """
        if self._full_data_ref is None or not isinstance(self._full_data_ref, HeteroData):
            print("[ERROR] _get_features_for_indices: _full_data_ref (processed graph) is None or invalid.")
            return None, None, None, None
        full_graph = self._full_data_ref

        if transaction_indices is None or transaction_indices.numel() == 0:
             print("[WARN] _get_features_for_indices: Received empty or None indices.")
             return None, None, None, None

        # Ensure indices are long and on CPU for indexing/sampling logic
        seed_indices_tensor = transaction_indices.cpu().long()
        indices_np = seed_indices_tensor.numpy()
        device = self.device # Target device for final user_ids tensor

        # --- Initialize outputs ---
        graph_batch = None # Will hold the sampled subgraph
        sequence_batch = None
        text_batch = None
        user_ids = None

        try:
            # Check node existence for safety against transaction indices
            num_tx_nodes = full_graph['transaction'].num_nodes
            if np.any(indices_np >= num_tx_nodes):
                 offending_indices = indices_np[indices_np >= num_tx_nodes]
                 print(f"[ERROR] _get_features_for_indices: Indices out of bounds. Max tx node index: {num_tx_nodes-1}. Offending: {offending_indices}")
                 return None, None, None, None

            # --- 1. Subgraph Sampling (if GNN enabled) ---
            if self.use_gnn_encoder:
                try:
                    print(f"[DEBUG] MAML Subgraph Sampling for {len(seed_indices_tensor)} seed nodes...")
                    k_hops = self._graph_config.get('maml_sampling_hops', 2)
                    num_neighbors = self._graph_config.get('maml_sampling_neighbors', [15, 10]) # Placeholder if needed
                    
                    # --- Manual k-Hop Subgraph Implementation for HeteroData ---
                    sampled_nodes = {'transaction': set(indices_np)} # Start with seed nodes
                    frontier = {'transaction': sampled_nodes['transaction'].copy()}
                    all_sampled_edges = {etype: set() for etype in full_graph.edge_types} # Store (src_orig, dst_orig) tuples

                    # Initialize sets for other node types
                    for ntype in full_graph.node_types:
                        if ntype != 'transaction':
                            sampled_nodes[ntype] = set()
                            frontier[ntype] = set()

                    # BFS for k hops
                    for hop in range(k_hops):
                        next_frontier = {ntype: set() for ntype in full_graph.node_types}
                        print(f"[DEBUG] Sampling Hop {hop+1}/{k_hops}...")
                        nodes_added_this_hop = 0
                        edges_added_this_hop = 0

                        for edge_type in full_graph.edge_types:
                            src_type, _, dst_type = edge_type
                            
                            # Check if source type has nodes in the current frontier
                            if frontier[src_type]:
                                # Get edges originating from frontier nodes of src_type
                                full_edge_index = full_graph[edge_type].edge_index
                                
                                # Create a mask for edges starting from the current frontier
                                # This can be slow for large frontiers/graphs - consider optimization if needed
                                frontier_nodes_tensor = torch.tensor(list(frontier[src_type]), dtype=torch.long)
                                edge_mask = torch.isin(full_edge_index[0], frontier_nodes_tensor)

                                # Get target nodes and original indices of the edges found
                                dst_nodes_indices = full_edge_index[1][edge_mask]
                                src_nodes_indices = full_edge_index[0][edge_mask] # Source nodes for the matched edges
                                
                                # Add newly found destination nodes
                                new_dst_nodes = set(dst_nodes_indices.numpy()) - sampled_nodes[dst_type]
                                sampled_nodes[dst_type].update(new_dst_nodes)
                                next_frontier[dst_type].update(new_dst_nodes)
                                nodes_added_this_hop += len(new_dst_nodes)
                                
                                # Add edges to the collection
                                current_edges = set(zip(src_nodes_indices.numpy(), dst_nodes_indices.numpy()))
                                new_edges_count = len(current_edges - all_sampled_edges[edge_type])
                                all_sampled_edges[edge_type].update(current_edges)
                                edges_added_this_hop += new_edges_count

                        print(f"[DEBUG] Hop {hop+1}: Added {nodes_added_this_hop} nodes, {edges_added_this_hop} edges.")
                        frontier = next_frontier
                        if nodes_added_this_hop == 0 and edges_added_this_hop == 0: # Optimization: stop if no new nodes/edges found
                             print(f"[DEBUG] Early stopping sampling at hop {hop+1}.")
                             break

                    # --- Construct the Subgraph HeteroData Object ---
                    graph_batch = HeteroData()
                    node_maps = {} # original_index -> new_subgraph_index

                    print("[DEBUG] Building subgraph object...")
                    # 1. Map nodes and copy features
                    for node_type, nodes_set in sampled_nodes.items():
                        if not nodes_set: continue # Skip empty node types
                        
                        sorted_nodes = sorted(list(nodes_set))
                        node_maps[node_type] = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted_nodes)}
                        graph_batch[node_type].num_nodes = len(sorted_nodes)
                        
                        if 'x' in full_graph[node_type]:
                            node_indices_tensor = torch.tensor(sorted_nodes, dtype=torch.long)
                            graph_batch[node_type].x = full_graph[node_type].x[node_indices_tensor]
                        
                        # Copy other necessary attributes (like labels, original_index for transaction)
                        if node_type == 'transaction':
                             if 'y_global' in full_graph[node_type]: graph_batch[node_type].y_global = full_graph[node_type].y_global[node_indices_tensor]
                             if 'y_user' in full_graph[node_type]: graph_batch[node_type].y_user = full_graph[node_type].y_user[node_indices_tensor]
                             if 'original_index' in full_graph[node_type]: graph_batch[node_type].original_index = full_graph[node_type].original_index[node_indices_tensor]
                             if 'user_id_code' in full_graph[node_type]: graph_batch[node_type].user_id_code = full_graph[node_type].user_id_code[node_indices_tensor]
                             # Add timestamp if needed by model architecture (e.g., some GNN layers)
                             if 'timestamp' in full_graph[node_type]: graph_batch[node_type].timestamp = full_graph[node_type].timestamp[node_indices_tensor]


                    # 2. Map edges and copy attributes
                    for edge_type, edges_set in all_sampled_edges.items():
                        if not edges_set: continue
                        
                        src_type, _, dst_type = edge_type
                        if src_type not in node_maps or dst_type not in node_maps: continue # Skip if node types not in subgraph

                        src_map = node_maps[src_type]
                        dst_map = node_maps[dst_type]
                        
                        new_edge_indices = []
                        original_edge_indices_for_attrs = [] # Store original indices to fetch edge_attr

                        # Find original indices of the sampled edges
                        full_edge_index_np = full_graph[edge_type].edge_index.T.numpy() # Transpose for easier row searching
                        edge_set_list = list(edges_set) # Convert set to list for consistent ordering? Not strictly needed.

                        # This mapping can be slow, optimize if bottleneck
                        # Create a quick lookup for edges in the original graph
                        # full_edge_tuples = set(tuple(row) for row in full_edge_index_np) # Can be large

                        # Store original edge indices corresponding to the edges in edges_set
                        # We need to map (src_orig, dst_orig) back to the row index in full_graph[edge_type].edge_index
                        # This is non-trivial. Let's simplify: ONLY copy edge_index first.
                        # If edge_attr is crucial, this needs careful implementation.

                        for src_orig, dst_orig in edges_set:
                            if src_orig in src_map and dst_orig in dst_map:
                                new_src = src_map[src_orig]
                                new_dst = dst_map[dst_orig]
                                new_edge_indices.append([new_src, new_dst])
                                # --- Simplified: Omit edge_attr copying for now ---
                                # TODO: Implement edge_attr copying if needed by finding original edge index
                                # -------------------------------------------------

                        if new_edge_indices:
                            graph_batch[edge_type].edge_index = torch.tensor(new_edge_indices, dtype=torch.long).t().contiguous()
                            # --- Simplified: Set edge_attr to None ---
                            graph_batch[edge_type].edge_attr = None
                            # -----------------------------------------
                        print(f"[DEBUG] Added {len(new_edge_indices)} edges for type {edge_type}.")


                    # 3. Store mapping for seed nodes
                    if 'transaction' in node_maps:
                         seed_map = node_maps['transaction']
                         # Filter seed indices that are actually present in the sampled subgraph
                         valid_seed_indices = [idx.item() for idx in seed_indices_tensor if idx.item() in seed_map]
                         if valid_seed_indices:
                              graph_batch['transaction'].seed_idx_in_sample = torch.tensor(
                                   [seed_map[orig_idx] for orig_idx in valid_seed_indices], dtype=torch.long
                              )
                              # Also store the original indices corresponding to these subgraph seeds
                              graph_batch['transaction'].original_seed_indices = torch.tensor(valid_seed_indices, dtype=torch.long)
                         else:
                              print("[WARN] No valid seed nodes found within the sampled subgraph!")
                              graph_batch['transaction'].seed_idx_in_sample = torch.tensor([], dtype=torch.long)
                              graph_batch['transaction'].original_seed_indices = torch.tensor([], dtype=torch.long)

                    print(f"[DEBUG] MAML Subgraph Sampling Complete. Nodes: {graph_batch.num_nodes}, Edges: {graph_batch.num_edges}")

                except ImportError:
                     print("[ERROR] NeighborSampler not available (check PyG version?). MAML GNN requires efficient sampling.")
                     graph_batch = None # Fallback if sampler fails
                except Exception as e_sample:
                    print(f"[ERROR] Subgraph sampling/preparation failed: {e_sample}")
                    traceback.print_exc()
                    graph_batch = None # Fallback to None if sampling fails
            # else: graph_batch remains None if GNN not used

            # --- 2. Fetch Features ONLY for SEED nodes (transaction_indices) ---
            #    (Sequence and Text data fetching logic remains the same)
            # --- (Code from previous correct version) ---
            if 'user_id_code' in full_graph['transaction']: user_ids = full_graph['transaction'].user_id_code[indices_np].to(device)
            if self.use_text_encoder:
                 combined_texts = []
                 raw_text_keys = [k for k in full_graph['transaction'].keys() if k.startswith('_raw_')]
                 if raw_text_keys:
                     for i in indices_np:
                         entry_texts = []
                         for key in raw_text_keys:
                             try: text_list = full_graph['transaction'][key]; entry_texts.append(str(text_list[i]))
                             except Exception: entry_texts.append("")
                         combined_texts.append(" ".join(filter(None, entry_texts)).strip())
                     text_batch = combined_texts
                 else: text_batch = [""] * len(indices_np)
            if self.use_sequence_encoder:
                 if 'seq_features' in full_graph['transaction'] and 'seq_lengths' in full_graph['transaction']:
                      seq_feat = full_graph['transaction'].seq_features[indices_np]
                      seq_len = full_graph['transaction'].seq_lengths[indices_np]
                      sequence_batch = {'sequences': seq_feat, 'lengths': seq_len}
                      if 'seq_cat_features' in full_graph['transaction']: sequence_batch['seq_cat_features'] = full_graph['transaction'].seq_cat_features[indices_np]
                 else: print("[WARN] _get_features_for_indices: Sequence data missing.")
            # --- End Sequence/Text Fetching ---

        except Exception as e:
            print(f"[ERROR] _get_features_for_indices: Failed during feature extraction: {e}")
            traceback.print_exc()
            return None, None, None, None

        return graph_batch, sequence_batch, text_batch, user_ids


    def _get_fused_representation(self,
                                  graph_batch: Optional[HeteroData] = None, # Now expects a SUBGRAPH for MAML GNN
                                  seed_indices_for_graph: Optional[torch.Tensor] = None, # Original indices (still useful for other modalities)
                                  sequence_batch: Optional[Any] = None,
                                  text_batch: Optional[List[str]] = None,
                                  user_ids: Optional[torch.Tensor] = None
                                 ) -> Optional[torch.Tensor]:
        """Internal helper to run encoders and fusion module, returning only the fused tensor."""
        device = self.device
        embeddings_to_fuse = {}
        graph_embed = None

        # Determine effective batch size from seed_indices if available, else fallback
        effective_batch_size = None
        if seed_indices_for_graph is not None:
             effective_batch_size = seed_indices_for_graph.shape[0]
        # Fallbacks (should match seed_indices size ideally)
        elif text_batch is not None: effective_batch_size = len(text_batch)
        elif sequence_batch is not None and 'sequences' in sequence_batch: effective_batch_size = sequence_batch['sequences'].shape[0]
        elif user_ids is not None: effective_batch_size = user_ids.shape[0]

        if effective_batch_size is None or effective_batch_size == 0: print("[WARN] Fuse: Cannot determine effective batch size."); return None
        if sequence_batch is not None: sequence_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sequence_batch.items()}
        if user_ids is not None: user_ids = user_ids.to(device)


        # --- 1. Graph Encoding (Operates on Subgraph if MAML) ---
        if self.graph_encoder and graph_batch is not None:
            try:
                graph_batch_dev = graph_batch.to(device) # Move subgraph to device
                # Call the graph encoder on the SUBGRAPH
                node_embeddings_dict = self.graph_encoder(graph_batch_dev.x_dict, graph_batch_dev.edge_index_dict) # Pass edge_attr if copied

                if 'transaction' in node_embeddings_dict:
                    tx_embeds_all_subgraph = node_embeddings_dict['transaction']

                    # --- Extract embeddings for original SEED nodes using the stored mapping ---
                    if hasattr(graph_batch['transaction'], 'seed_idx_in_sample'):
                         seed_idx_in_sample = graph_batch['transaction'].seed_idx_in_sample.to(device) # Get indices within subgraph
                         if seed_idx_in_sample.numel() > 0:
                              # Ensure indices are valid
                              if seed_idx_in_sample.max() < tx_embeds_all_subgraph.shape[0]:
                                   graph_embed = tx_embeds_all_subgraph[seed_idx_in_sample]
                              else:
                                   print(f"[ERROR] Fuse: seed_idx_in_sample out of bounds for subgraph GNN output!")
                                   graph_embed = None
                         else:
                              print("[WARN] Fuse: seed_idx_in_sample is empty.")
                              graph_embed = None # No valid seed nodes found in subgraph
                    else:
                         # This case should ideally not happen if _get_features_for_indices stores the mapping
                         print("[WARN] Fuse: seed_idx_in_sample not found on graph_batch. Attempting fallback slice.")
                         # Fallback: Assume seed nodes are the first N in the subgraph output
                         if tx_embeds_all_subgraph.shape[0] >= effective_batch_size:
                              graph_embed = tx_embeds_all_subgraph[:effective_batch_size]
                         else:
                              graph_embed = tx_embeds_all_subgraph # Use what's available

                    # --- Validate output size and add to fusion dict ---
                    if graph_embed is not None and graph_embed.shape[0] > 0:
                         # The number of embeddings extracted should match the original number of valid seed nodes
                         expected_seeds = graph_batch['transaction'].original_seed_indices.numel() if hasattr(graph_batch['transaction'], 'original_seed_indices') else effective_batch_size
                         if graph_embed.shape[0] != expected_seeds:
                              print(f"[WARN] Fuse: Extracted graph embedding size ({graph_embed.shape[0]}) mismatch with expected seeds ({expected_seeds}).")
                              # Handle potential size mismatch with other modalities if necessary (e.g., pad?)
                              # For now, let pre-fusion check handle strict size match.
                         embeddings_to_fuse['graph'] = graph_embed
                    elif effective_batch_size > 0:
                        print("[WARN] Fuse: Empty graph embeddings after extraction.")
                else:
                    print("[WARN] Fuse: GNN output missing 'transaction' embeddings.")
            except Exception as e:
                print(f"[ERROR] Fuse (Graph Encoder Call): {e}")
                traceback.print_exc()
        elif self.use_gnn_encoder:
            print("[WARN] Fuse: GNN is enabled but graph_batch was not provided or was None.")


        # --- 2. Sequence Encoding ---
        # ... (Sequence, Text, User Embedding logic remains the same) ...
        if self.sequence_encoder and sequence_batch is not None:
            try:
                seq_embed = self.sequence_encoder(sequence_batch, device=self.device)
                if seq_embed is not None and seq_embed.shape[0] > 0: embeddings_to_fuse['sequence'] = seq_embed
            except Exception as e: print(f"[ERROR] Fuse (Sequence): {e}")
        if self.text_encoder and text_batch is not None:
             if text_batch:
                 try:
                     text_embed = self.text_encoder(text_batch)
                     if text_embed is not None and text_embed.shape[0] > 0: embeddings_to_fuse['text'] = text_embed
                 except Exception as e: print(f"[ERROR] Fuse (Text): {e}")
        if self.user_embedding and user_ids is not None:
            try:
                user_ids_clamped = torch.clamp(user_ids, 0, self.user_embedding.num_embeddings - 1)
                user_embed = self.user_embedding(user_ids_clamped)
                if user_embed is not None and user_embed.shape[0] > 0: embeddings_to_fuse['user'] = user_embed
            except Exception as e: print(f"[ERROR] Fuse (User): {e}")

        # --- Pre-Fusion Checks & Fusion ---
        if not embeddings_to_fuse: print("[ERROR] Fuse: No embeddings available."); return None
        # Check consistency against effective_batch_size
        ref_size = next(iter(embeddings_to_fuse.values())).shape[0] # Get size from first available modality
        for name, emb in embeddings_to_fuse.items():
            if emb.shape[0] != ref_size: # Check against the size of the first modality found
                 print(f"[ERROR] Fuse: Mismatched batch sizes! {name}:{emb.shape[0]} vs reference:{ref_size}")
                 return None
            if emb.device != self.device: print(f"[ERROR] Fuse: {name} on wrong device {emb.device}"); return None
        if ref_size == 0: print("[WARN] Fuse: Reference embedding size is 0."); return None

        try:
            fused_representation, _ = self.fusion_module(embeddings_to_fuse)
            return fused_representation
        except Exception as e: print(f"[ERROR] Fusion module failed: {e}"); traceback.print_exc(); return None
        # --- End _get_fused_representation Code ---


    # --- Lightning Steps ---
    def training_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        if self._use_maml_runtime_flag:
            # --- MAML Training Logic ---
            meta_optimizer = self.optimizers()
            meta_optimizer.zero_grad()
            batch_of_tasks = None
            if isinstance(batch, dict) and 'support' in batch and 'query' in batch: batch_of_tasks = self._reconstruct_maml_batch(batch)
            elif isinstance(batch, list): batch_of_tasks = batch
            else: print(f"[ERROR] training_step: Unexpected batch type for MAML: {type(batch)}"); return None
            if batch_of_tasks is None: return None

            total_outer_loss = 0.0; total_query_acc_numerator = 0.0; total_query_samples = 0
            target_head_attr = f"{self.hparams.maml_head_label_type}_specific_head"
            target_loss_attr = f"focal_loss_{self.hparams.maml_head_label_type}"
            if not hasattr(self, target_head_attr) or getattr(self, target_head_attr) is None: print(f"[ERROR] MAML target head '{target_head_attr}' missing."); return None
            if not hasattr(self, target_loss_attr) or getattr(self, target_loss_attr) is None: print(f"[ERROR] MAML target loss '{target_loss_attr}' missing."); return None
            original_head = getattr(self, target_head_attr); loss_fn = getattr(self, target_loss_attr)
            tasks_processed_for_update = 0

            for task_data in batch_of_tasks:
                support_indices, support_labels = task_data['support']; query_indices, query_labels = task_data['query']
                support_labels = support_labels.to(self.device); query_labels = query_labels.to(self.device)
                if support_indices.numel() == 0 or query_indices.numel() == 0: continue

                learner = l2l.clone_module(original_head); inner_loop_failed = False
                for adaptation_step in range(self.hparams.adaptation_steps):
                    graph_batch_supp, seq_batch_supp, text_batch_supp, user_ids_supp = self._get_features_for_indices(support_indices)
                    if graph_batch_supp is None and seq_batch_supp is None and text_batch_supp is None and user_ids_supp is None: inner_loop_failed=True; break
                    support_fused = self._get_fused_representation(graph_batch=graph_batch_supp, seed_indices_for_graph=support_indices, sequence_batch=seq_batch_supp, text_batch=text_batch_supp, user_ids=user_ids_supp)
                    if support_fused is None: inner_loop_failed=True; break
                    support_preds = learner(support_fused); inner_loss = loss_fn(support_preds, support_labels)
                    grads = torch.autograd.grad(inner_loss, learner.parameters(), create_graph=False)
                    updates = [-self.hparams.inner_lr * g for g in grads]
                    l2l.update_module(learner, updates=updates)
                if inner_loop_failed: continue

                # Outer Loop - Ensure gradients are tracked
                graph_batch_qry, seq_batch_qry, text_batch_qry, user_ids_qry = self._get_features_for_indices(query_indices)
                if graph_batch_qry is None and seq_batch_qry is None and text_batch_qry is None and user_ids_qry is None: continue
                query_fused = self._get_fused_representation(graph_batch=graph_batch_qry, seed_indices_for_graph=query_indices, sequence_batch=seq_batch_qry, text_batch=text_batch_qry, user_ids=user_ids_qry)
                if query_fused is None: continue
                query_preds = learner(query_fused); outer_loss = loss_fn(query_preds, query_labels) # Grad tracked here

                if not torch.isnan(outer_loss).any() and not torch.isinf(outer_loss).any():
                     total_outer_loss += outer_loss; tasks_processed_for_update += 1
                     with torch.no_grad():
                          current_acc_num, current_samples = self._calculate_accuracy_numerator_denominator(query_preds, query_labels)
                          total_query_acc_numerator += current_acc_num; total_query_samples += current_samples
                else: print(f"[WARN] Outer loss NaN/Inf for user {task_data.get('user_id', 'Unknown')}.")

            if tasks_processed_for_update > 0:
                avg_outer_loss = total_outer_loss / tasks_processed_for_update
                self.manual_backward(avg_outer_loss); meta_optimizer.step()
                avg_query_acc = (total_query_acc_numerator / total_query_samples) if total_query_samples > 0 else 0.0
                log_batch_size = tasks_processed_for_update # Log per task processed for update
                self.log(f'train/meta_outer_loss', avg_outer_loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)
                self.log(f'train/meta_query_acc_{self.hparams.maml_head_label_type}_num', total_query_acc_numerator, on_step=False, on_epoch=True, batch_size=log_batch_size, sync_dist=True)
                self.log(f'train/meta_query_acc_{self.hparams.maml_head_label_type}_den', float(total_query_samples), on_step=False, on_epoch=True, batch_size=log_batch_size, sync_dist=True)
                self.log(f'train/meta_query_acc_{self.hparams.maml_head_label_type}', avg_query_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)
                return avg_outer_loss
            else: print("[WARN] No tasks processed in meta-batch. Skipping update."); return None

        else: # <<< Standard (non-MAML) training step logic integrated here >>>
            stage = 'train'
            if not isinstance(batch, HeteroData): print(f"[ERROR] {stage}_step invalid batch type: {type(batch)}"); return None
            graph_batch = batch
            batch_size = graph_batch['transaction'].batch_size if hasattr(graph_batch['transaction'], 'batch_size') else 0
            if batch_size == 0: print(f"[WARN] {stage}_step: Batch size is 0."); return None

            sequence_batch, text_batch, user_ids, global_target, user_target, scheduleC_target = None, None, None, None, None, None
            try:
                tx_store = graph_batch['transaction']
                if self.use_sequence_encoder and hasattr(tx_store, 'seq_features') and hasattr(tx_store, 'seq_lengths'):
                     if tx_store.seq_features.shape[0] >= batch_size:
                         sequence_batch = {'sequences': tx_store.seq_features[:batch_size], 'lengths': tx_store.seq_lengths[:batch_size]}
                         if hasattr(tx_store, 'seq_cat_features') and tx_store.seq_cat_features.shape[0] >= batch_size: sequence_batch['seq_cat_features'] = tx_store.seq_cat_features[:batch_size]
                if self.use_text_encoder and hasattr(tx_store, 'n_id'):
                     original_indices = tx_store.n_id[:batch_size]; _, _, text_batch, _ = self._get_features_for_indices(original_indices)
                     if text_batch is None: text_batch = [""] * batch_size
                if hasattr(tx_store, 'user_id_code'):
                     if tx_store.user_id_code.shape[0] >= batch_size: user_ids = tx_store.user_id_code[:batch_size]
                elif hasattr(tx_store, 'n_id') and self._full_data_ref and 'user_id_code' in self._full_data_ref['transaction']:
                     original_indices = tx_store.n_id[:batch_size]; user_ids = self._full_data_ref['transaction'].user_id_code[original_indices]
                if hasattr(tx_store, 'y_global') and tx_store.y_global.shape[0] >= batch_size: global_target = tx_store.y_global[:batch_size]
                if hasattr(tx_store, 'y_user') and tx_store.y_user.shape[0] >= batch_size: user_target = tx_store.y_user[:batch_size]
                if self.use_scheduleC_head and hasattr(tx_store, 'y_scheduleC') and tx_store.y_scheduleC.shape[0] >= batch_size: scheduleC_target = tx_store.y_scheduleC[:batch_size]
            except Exception as e: print(f"[ERROR] Failed during standard batch data extraction in {stage}_step: {e}"); traceback.print_exc(); return None

            target_available = (self.global_head and global_target is not None) or (self.user_specific_head and user_target is not None) or (self.scheduleC_head and scheduleC_target is not None)
            if not target_available: print(f"[WARN] {stage}_step: No target labels found."); return None
            if self.user_embedding and user_ids is None: print(f"[WARN] {stage}_step: user_ids is None but user embedding active."); return None

            global_logits, user_specific_logits, scheduleC_logits = self(graph_batch=graph_batch, sequence_batch=sequence_batch, text_batch=text_batch, user_ids=user_ids, batch_size=batch_size)
            logits_produced = global_logits is not None or user_specific_logits is not None or scheduleC_logits is not None
            if not logits_produced: print(f"[WARN] {stage}_step: No logits produced."); return None
            total_loss, loss_global, loss_user, loss_scheduleC = self._calculate_mtl_loss(global_logits, global_target, user_specific_logits, user_target, scheduleC_logits, scheduleC_target)

            global_correct, global_total = self._calculate_accuracy_numerator_denominator(global_logits, global_target)
            user_correct, user_total = self._calculate_accuracy_numerator_denominator(user_specific_logits, user_target)
            scheduleC_correct, scheduleC_total = self._calculate_accuracy_numerator_denominator(scheduleC_logits, scheduleC_target)
            log_batch_size = global_total or user_total or scheduleC_total or 0
            if log_batch_size == 0: log_batch_size = 1
            log_dict = { f'{stage}/total_loss': total_loss }
            if self.global_head: log_dict.update({f'{stage}/global_loss': loss_global, f'{stage}/acc_global_num': float(global_correct), f'{stage}/acc_global_den': float(global_total), f'{stage}/acc_global': global_correct / global_total if global_total > 0 else 0.0})
            if self.user_specific_head: log_dict.update({f'{stage}/user_loss': loss_user, f'{stage}/acc_user_num': float(user_correct), f'{stage}/acc_user_den': float(user_total), f'{stage}/acc_user': user_correct / user_total if user_total > 0 else 0.0})
            if self.scheduleC_head: log_dict.update({f'{stage}/scheduleC_loss': loss_scheduleC, f'{stage}/acc_scheduleC_num': float(scheduleC_correct), f'{stage}/acc_scheduleC_den': float(scheduleC_total), f'{stage}/acc_scheduleC': scheduleC_correct / scheduleC_total if scheduleC_total > 0 else 0.0})
            self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)
            return total_loss
            # <<< End of integrated non-MAML training logic >>>

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        stage = 'val'
        if self._use_maml_runtime_flag:
             batch_of_tasks = None
             if isinstance(batch, list): batch_of_tasks = batch
             elif isinstance(batch, dict) and 'support' in batch: batch_of_tasks = self._reconstruct_maml_batch(batch)
             if batch_of_tasks is not None: self._meta_eval_step(batch_of_tasks, batch_idx, stage=stage)
             else: print(f"[ERROR] {stage}_step: MAML batch issue: type {type(batch)}")
        else:
             self._step(batch, batch_idx, stage=stage) # Use _step for standard val

    def test_step(self, batch: Any, batch_idx: int) -> None:
        stage = 'test'
        if self._use_maml_runtime_flag:
             batch_of_tasks = None
             if isinstance(batch, list): batch_of_tasks = batch
             elif isinstance(batch, dict) and 'support' in batch: batch_of_tasks = self._reconstruct_maml_batch(batch)
             if batch_of_tasks is not None: self._meta_eval_step(batch_of_tasks, batch_idx, stage=stage)
             else: print(f"[ERROR] {stage}_step: MAML batch issue: type {type(batch)}")
        else:
             self._step(batch, batch_idx, stage=stage) # Use _step for standard test

    # --- Keep the original _step method for standard validation/testing ---
    def _step(self, batch: Any, batch_idx: int, stage: str) -> Optional[torch.Tensor]:
        """Common logic for standard (non-MAML) val/test steps (and previously train)."""
        if not isinstance(batch, HeteroData): print(f"[ERROR] {stage}_step invalid batch type: {type(batch)}"); return None
        graph_batch = batch
        batch_size = graph_batch['transaction'].batch_size if hasattr(graph_batch['transaction'], 'batch_size') else 0
        if batch_size == 0: print(f"[WARN] {stage}_step: Batch size is 0."); return None
        sequence_batch, text_batch, user_ids, global_target, user_target, scheduleC_target = None, None, None, None, None, None
        try:
            tx_store = graph_batch['transaction']
            if self.use_sequence_encoder and hasattr(tx_store, 'seq_features') and hasattr(tx_store, 'seq_lengths'):
                 if tx_store.seq_features.shape[0] >= batch_size:
                     sequence_batch = {'sequences': tx_store.seq_features[:batch_size], 'lengths': tx_store.seq_lengths[:batch_size]}
                     if hasattr(tx_store, 'seq_cat_features') and tx_store.seq_cat_features.shape[0] >= batch_size: sequence_batch['seq_cat_features'] = tx_store.seq_cat_features[:batch_size]
            if self.use_text_encoder and hasattr(tx_store, 'n_id'):
                 original_indices = tx_store.n_id[:batch_size]; _, _, text_batch, _ = self._get_features_for_indices(original_indices)
                 if text_batch is None: text_batch = [""] * batch_size
            if hasattr(tx_store, 'user_id_code'):
                 if tx_store.user_id_code.shape[0] >= batch_size: user_ids = tx_store.user_id_code[:batch_size]
            elif hasattr(tx_store, 'n_id') and self._full_data_ref and 'user_id_code' in self._full_data_ref['transaction']:
                 original_indices = tx_store.n_id[:batch_size]; user_ids = self._full_data_ref['transaction'].user_id_code[original_indices]
            if hasattr(tx_store, 'y_global') and tx_store.y_global.shape[0] >= batch_size: global_target = tx_store.y_global[:batch_size]
            if hasattr(tx_store, 'y_user') and tx_store.y_user.shape[0] >= batch_size: user_target = tx_store.y_user[:batch_size]
            if self.use_scheduleC_head and hasattr(tx_store, 'y_scheduleC') and tx_store.y_scheduleC.shape[0] >= batch_size: scheduleC_target = tx_store.y_scheduleC[:batch_size]
        except Exception as e: print(f"[ERROR] Failed during standard batch data extraction in {stage}_step: {e}"); traceback.print_exc(); return None
        target_available = (self.global_head and global_target is not None) or (self.user_specific_head and user_target is not None) or (self.scheduleC_head and scheduleC_target is not None)
        if not target_available and stage != 'predict': print(f"[WARN] {stage}_step: No target labels found."); return None
        if self.user_embedding and user_ids is None: print(f"[WARN] {stage}_step: user_ids is None but user embedding active."); return None
        global_logits, user_specific_logits, scheduleC_logits = self(graph_batch=graph_batch, sequence_batch=sequence_batch, text_batch=text_batch, user_ids=user_ids, batch_size=batch_size)
        logits_produced = global_logits is not None or user_specific_logits is not None or scheduleC_logits is not None
        if not logits_produced: print(f"[WARN] {stage}_step: No logits produced."); return None
        total_loss, loss_global, loss_user, loss_scheduleC = self._calculate_mtl_loss(global_logits, global_target, user_specific_logits, user_target, scheduleC_logits, scheduleC_target)
        global_correct, global_total = self._calculate_accuracy_numerator_denominator(global_logits, global_target)
        user_correct, user_total = self._calculate_accuracy_numerator_denominator(user_specific_logits, user_target)
        scheduleC_correct, scheduleC_total = self._calculate_accuracy_numerator_denominator(scheduleC_logits, scheduleC_target)
        log_batch_size = global_total or user_total or scheduleC_total or 0
        if log_batch_size == 0: log_batch_size = 1
        log_dict = { f'{stage}/total_loss': total_loss }
        if self.global_head: log_dict.update({f'{stage}/global_loss': loss_global, f'{stage}/acc_global_num': float(global_correct), f'{stage}/acc_global_den': float(global_total), f'{stage}/acc_global': global_correct / global_total if global_total > 0 else 0.0})
        if self.user_specific_head: log_dict.update({f'{stage}/user_loss': loss_user, f'{stage}/acc_user_num': float(user_correct), f'{stage}/acc_user_den': float(user_total), f'{stage}/acc_user': user_correct / user_total if user_total > 0 else 0.0})
        if self.scheduleC_head: log_dict.update({f'{stage}/scheduleC_loss': loss_scheduleC, f'{stage}/acc_scheduleC_num': float(scheduleC_correct), f'{stage}/acc_scheduleC_den': float(scheduleC_total), f'{stage}/acc_scheduleC': scheduleC_correct / scheduleC_total if scheduleC_total > 0 else 0.0})
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True) # Log on epoch for val/test
        return None # Return None for val/test steps


    def configure_optimizers(self):
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())
        lr = self.hparams.meta_lr if self._use_maml_runtime_flag else self.hparams.learning_rate
        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=self.hparams.weight_decay)
        print(f"[INFO] configure_optimizers: Using LR={lr:.2e} (MAML Mode: {self._use_maml_runtime_flag})")
        return optimizer

    def _meta_eval_step(self, batch_of_tasks: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], batch_idx: int, stage: str) -> None:
        total_outer_loss = 0.0; total_query_acc_numerator = 0.0; total_query_samples = 0; tasks_processed = 0
        target_head_attr = f"{self.hparams.maml_head_label_type}_specific_head"; target_loss_attr = f"focal_loss_{self.hparams.maml_head_label_type}"
        if not hasattr(self, target_head_attr) or getattr(self, target_head_attr) is None: return
        if not hasattr(self, target_loss_attr) or getattr(self, target_loss_attr) is None: return
        original_head = getattr(self, target_head_attr); loss_fn = getattr(self, target_loss_attr)
        for task_data in batch_of_tasks:
            support_indices, support_labels = task_data['support']; query_indices, query_labels = task_data['query']
            support_labels = support_labels.to(self.device); query_labels = query_labels.to(self.device)
            if support_indices.numel() == 0 or query_indices.numel() == 0: continue
            with torch.enable_grad():
                learner = l2l.clone_module(original_head); inner_loop_failed = False
                for _ in range(self.hparams.adaptation_steps):
                    graph_batch_supp, seq_batch_supp, text_batch_supp, user_ids_supp = self._get_features_for_indices(support_indices)
                    if graph_batch_supp is None and seq_batch_supp is None and text_batch_supp is None and user_ids_supp is None: inner_loop_failed=True; break
                    support_fused = self._get_fused_representation(graph_batch=graph_batch_supp, seed_indices_for_graph=support_indices, sequence_batch=seq_batch_supp, text_batch=text_batch_supp, user_ids=user_ids_supp)
                    if support_fused is None: inner_loop_failed=True; break
                    support_preds = learner(support_fused); inner_loss = loss_fn(support_preds, support_labels)
                    grads = torch.autograd.grad(inner_loss, learner.parameters(), create_graph=False)
                    updates = [-self.hparams.inner_lr * g for g in grads]
                    l2l.update_module(learner, updates=updates)
                if inner_loop_failed: continue
                with torch.no_grad():
                    graph_batch_qry, seq_batch_qry, text_batch_qry, user_ids_qry = self._get_features_for_indices(query_indices)
                    if graph_batch_qry is None and seq_batch_qry is None and text_batch_qry is None and user_ids_qry is None: continue
                    query_fused = self._get_fused_representation(graph_batch=graph_batch_qry, seed_indices_for_graph=query_indices, sequence_batch=seq_batch_qry, text_batch=text_batch_qry, user_ids=user_ids_qry)
                    if query_fused is None: continue
                    query_preds = learner(query_fused); outer_loss = loss_fn(query_preds, query_labels)
                    if not torch.isnan(outer_loss).any() and not torch.isinf(outer_loss).any():
                        total_outer_loss += outer_loss; tasks_processed += 1
                        with torch.no_grad():
                            current_acc_num, current_samples = self._calculate_accuracy_numerator_denominator(query_preds, query_labels)
                            total_query_acc_numerator += current_acc_num; total_query_samples += current_samples
                    else: print(f"[{stage}] Outer loss NaN/Inf for user {task_data.get('user_id', 'Unknown')}.")
        if tasks_processed > 0:
             avg_outer_loss = total_outer_loss / tasks_processed; avg_query_acc = (total_query_acc_numerator / total_query_samples) if total_query_samples > 0 else 0.0
             log_batch_size = tasks_processed
             log_dict = {f'{stage}/meta_outer_loss': avg_outer_loss, f'{stage}/meta_query_acc_{self.hparams.maml_head_label_type}_num': total_query_acc_numerator, f'{stage}/meta_query_acc_{self.hparams.maml_head_label_type}_den': float(total_query_samples), f'{stage}/meta_query_acc_{self.hparams.maml_head_label_type}': avg_query_acc}
             self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)
        else: print(f"[{stage}] No tasks successfully processed in evaluation batch.")

    def _reconstruct_maml_batch(self, collated_batch: Dict) -> Optional[List[Dict]]:
         try:
             if 'user_id' not in collated_batch: print("[ERROR] Cannot reconstruct MAML batch: 'user_id' key missing."); return None
             if isinstance(collated_batch['user_id'], list): num_tasks = len(collated_batch['user_id'])
             elif torch.is_tensor(collated_batch['user_id']): num_tasks = collated_batch['user_id'].shape[0]
             else: print("[ERROR] Cannot determine number of tasks from user_id field type."); return None
             if num_tasks == 0: return []
             reconstructed_batch = []
             support_data = collated_batch.get('support'); query_data = collated_batch.get('query'); user_id_batch = collated_batch['user_id']
             if not isinstance(support_data, (list, tuple)) or len(support_data) != 2: print("[ERROR] Invalid 'support' data format."); return None
             if not isinstance(query_data, (list, tuple)) or len(query_data) != 2: print("[ERROR] Invalid 'query' data format."); return None
             support_indices_batch = support_data[0]; support_labels_batch = support_data[1]; query_indices_batch = query_data[0]; query_labels_batch = query_data[1]
             if not (len(support_indices_batch) == num_tasks and len(support_labels_batch) == num_tasks and len(query_indices_batch) == num_tasks and len(query_labels_batch) == num_tasks): print("[ERROR] Mismatch between num_tasks and length of support/query data lists."); return None
             for i in range(num_tasks):
                  support_indices = support_indices_batch[i]; support_labels = support_labels_batch[i]; query_indices = query_indices_batch[i]; query_labels = query_labels_batch[i]; user_id = user_id_batch[i]
                  task = {'support': (support_indices, support_labels), 'query': (query_indices, query_labels), 'user_id': user_id}
                  reconstructed_batch.append(task)
             return reconstructed_batch
         except Exception as e: print(f"[ERROR] Failed to reconstruct MAML batch: {e}"); traceback.print_exc(); return None