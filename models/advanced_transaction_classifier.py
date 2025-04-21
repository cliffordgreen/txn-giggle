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
                 # <<< MODIFIED: Expecting the FULL HeteroData graph object >>>
                 full_data_ref: Optional[HeteroData] = None,
                 ):
        super().__init__()
        # Store simple hyperparameters automatically
        # <<< FIX: Use different key for MAML outer loop LR >>>
        self.save_hyperparameters('learning_rate', 'weight_decay',
                                'mtl_weights', 'focal_loss_alpha', 'focal_loss_gamma',
                                'use_maml', 'inner_lr', 'adaptation_steps', 'maml_head_label_type')

        # <<< Store flag reliably for runtime checks >>>
        # Set the runtime flag based on the init argument *after* hparams are saved/loaded
        self._use_maml_runtime_flag = use_maml
        self.hparams.use_maml = use_maml # Ensure hparams also reflects the init value

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
                 # <<< Pass metadata from config >>>
                 hgt_metadata = self._graph_config['metadata']
                 print(f"[INFO] Initializing HGT with metadata: Nodes={hgt_metadata[0]}, Edges={hgt_metadata[1]}")
                 # <<< Get input dims from config if provided >>>
                 # HGT can infer input dim if set to -1 and node features are heterogeneous
                 in_channels_config = self._graph_config.get('in_channels', -1)
                 self.graph_encoder = HGT(
                    in_channels=in_channels_config, # Use config value or -1 for inference
                    hidden_channels=self._graph_config.get('hidden_channels', 64),
                    out_channels=self._graph_config.get('out_channels', 64),
                    metadata=hgt_metadata,
                    num_heads=self._graph_config.get('num_heads', 4),
                    num_layers=self._graph_config.get('num_layers', 2)
                    # Add dropout if HGT supports it, e.g., dropout=self._graph_config.get('dropout', 0.1)
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
                    output_dim=self._sequence_config.get('output_dim', 64), # Provide default
                    tft_params=self._sequence_config.get('tft_params', {}),
                    # Pass sequence feature dimension if TFT requires it
                    # input_dim=model_config.get('sequence_feature_dim', None), # Example
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
                    projection_dim=self._text_config.get('projection_dim', 0) # Allow projection
                 )
                 text_out_dim = self.text_encoder.get_output_dim()
                 print(f"[INFO] FinBERT Encoder Initialized. Output Dim: {text_out_dim}")
        else:
             print("[INFO] Text Encoder is disabled via config.")

        self.user_embedding = None # Initialize
        if num_users > 0 and user_embed_dim > 0:
             self.user_embedding = nn.Embedding(num_users, user_embed_dim)
             print(f"[INFO] User Embedding Initialized. Num Users: {num_users}, Output Dim: {user_embed_dim}")
        else:
             print(f"[INFO] User Embedding skipped (num_users={num_users}, user_embed_dim={user_embed_dim}).")
             user_embed_dim = 0 # Ensure dim is 0 if not used

        # --- 2. Fusion Module ---
        fusion_input_dims = {}
        if self.use_gnn_encoder and self.graph_encoder:
             fusion_input_dims['graph'] = graph_out_dim
        if self.use_sequence_encoder and self.sequence_encoder:
             fusion_input_dims['sequence'] = seq_out_dim
        if self.use_text_encoder and self.text_encoder:
             fusion_input_dims['text'] = text_out_dim
        # Always include user embedding if it exists
        if self.user_embedding:
            fusion_input_dims['user'] = user_embed_dim

        if not fusion_input_dims:
            raise ValueError("No modalities are enabled or configured correctly. At least one encoder (graph, sequence, text) or user embedding must be active.")
        if not self._fusion_config:
             raise ValueError("'fusion_params' missing from model_config, cannot initialize fusion module.")

        fusion_output_dim = self._fusion_config.get('output_dim', 128) # Get output dim or default
        self.fusion_module = AttentionFusion(
            modality_dims=fusion_input_dims,
            hidden_dim=self._fusion_config.get('hidden_dim', 128), # Provide default
            output_dim=fusion_output_dim, # Use derived output dim
            dropout=self._fusion_config.get('dropout', 0.1)
        )
        print(f"[INFO] Attention Fusion Initialized. Input Dims: {fusion_input_dims}, Output Dim: {fusion_output_dim}")

        # --- 3. Classification Heads ---
        self.global_head = None
        if num_global_classes > 0:
             self.global_head = nn.Linear(fusion_output_dim, num_global_classes)
             print(f"[INFO] Global Classifier Initialized: Output Classes={num_global_classes}")
        else:
             print("[INFO] Global classification head skipped (num_global_classes=0).")

        self.user_specific_head = None
        if num_user_classes > 0:
             self.user_specific_head = nn.Linear(fusion_output_dim, num_user_classes)
             print(f"[INFO] User Classifier Initialized: Output Classes={num_user_classes}")
        else:
             print("[INFO] User classification head skipped (num_user_classes=0).")

        # <<< Conditional Schedule C Head >>>
        self.scheduleC_head = None
        if self.use_scheduleC_head:
             self.scheduleC_head = nn.Linear(fusion_output_dim, num_scheduleC_classes)
             print(f"[INFO] Schedule C Classifier Initialized: Output Classes={num_scheduleC_classes}")
        else:
             print("[INFO] Schedule C classification head skipped (num_scheduleC_classes=0).")

        # --- 4. Loss Functions ---
        self.focal_loss_global = None
        if self.global_head:
             self.focal_loss_global = FocalLoss(
                 alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma,
                 num_classes=num_global_classes # Use actual number of classes
             )
             print("[INFO] Global Focal Loss Initialized.")

        self.focal_loss_user = None
        if self.user_specific_head:
             self.focal_loss_user = FocalLoss(
                 alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma,
                 num_classes=num_user_classes # Use actual number of classes
             )
             print("[INFO] User Focal Loss Initialized.")

        # <<< Conditional Schedule C Loss >>>
        self.focal_loss_scheduleC = None
        if self.scheduleC_head:
             self.focal_loss_scheduleC = FocalLoss(
                 alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma,
                 num_classes=num_scheduleC_classes # Use actual number of classes
             )
             print("[INFO] Schedule C Focal Loss Initialized.")

    # --- Forward Pass (No MAML logic here) ---
    def forward(self,
                graph_batch: Optional[HeteroData] = None,
                sequence_batch: Optional[Any] = None,
                text_batch: Optional[List[str]] = None,
                user_ids: Optional[torch.Tensor] = None,
                batch_size: Optional[int] = None # Needed for slicing GNN output
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: # Added ScheduleC logits
        """Forward pass through encoders, fusion, and classification heads."""
        embeddings_to_fuse = {}
        graph_embed = None # Initialize graph_embed

        # --- 1. Graph Encoding ---
        if self.graph_encoder and graph_batch:
            try:
                # Pass both x_dict and edge_index_dict
                node_embeddings_dict = self.graph_encoder(graph_batch.x_dict, graph_batch.edge_index_dict)

                if 'transaction' in node_embeddings_dict:
                     # Try to infer batch size if not provided (e.g., from user_ids or text_batch)
                     if batch_size is None:
                          if user_ids is not None: batch_size = user_ids.shape[0]
                          elif text_batch is not None: batch_size = len(text_batch)
                          else:
                              # Check if graph_batch has 'input_id' or 'batch_size' from NeighborLoader
                              if hasattr(graph_batch['transaction'], 'input_id'):
                                   batch_size = graph_batch['transaction'].input_id.shape[0]
                              elif hasattr(graph_batch['transaction'], 'batch_size'):
                                   batch_size = graph_batch['transaction'].batch_size
                              else:
                                   # As a last resort, use the number of transaction nodes if forward is called outside Lightning context
                                   batch_size = graph_batch['transaction'].num_nodes
                                   # print("[WARN] Forward: batch_size not provided and cannot be inferred reliably. Using num_nodes as fallback.")

                     # Determine the number of nodes to slice based on the inferred/provided batch_size
                     # HGT output size might not match input batch_size directly due to sampling
                     num_nodes_in_output = node_embeddings_dict['transaction'].shape[0]

                     # Slice based on the minimum of expected batch size and available nodes
                     slice_len = min(batch_size, num_nodes_in_output) if batch_size is not None else num_nodes_in_output

                     if slice_len > 0:
                         graph_embed = node_embeddings_dict['transaction'][:slice_len]
                         if batch_size is not None and slice_len < batch_size:
                             print(f"[WARN] Forward: Sliced GNN output to {slice_len} nodes (expected {batch_size}).")
                     else:
                         graph_embed = None # No nodes to slice

                     if graph_embed is not None:
                         embeddings_to_fuse['graph'] = graph_embed.to(self.device)
                     elif batch_size > 0: # Only warn if batch_size was expected to be > 0
                          print("[WARN] Forward: HGT 'transaction' embedding is None or empty after slicing.")

                else:
                    print("[WARN] Forward: HGT output missing 'transaction' embeddings.")
            except Exception as e:
                print(f"[ERROR] HGT Encoder forward failed: {e}")
                traceback.print_exc() # Print full traceback for GNN errors

        # --- 2. Sequence Encoding ---
        if self.sequence_encoder and sequence_batch is not None:
            try:
                seq_embed = self.sequence_encoder(sequence_batch, device=self.device)
                if seq_embed is not None and seq_embed.shape[0] > 0:
                    # Basic batch size check (more robust checks in fusion stage)
                    # if graph_embed is not None and seq_embed.shape[0] != graph_embed.shape[0]:
                    #      print(f"[WARN] Forward: Sequence batch size ({seq_embed.shape[0]}) mismatch with Graph ({graph_embed.shape[0]})")
                    embeddings_to_fuse['sequence'] = seq_embed.to(self.device)
                elif batch_size is not None and batch_size > 0: # Check if expected based on batch size
                    print("[WARN] Forward: TFT output is None or empty.")
            except Exception as e:
                print(f"[ERROR] TFT Encoder forward failed: {e}")
                traceback.print_exc()

        # --- 3. Text Encoding ---
        if self.text_encoder and text_batch is not None:
            try:
                # Ensure text_batch is not empty
                if text_batch:
                     text_embed = self.text_encoder(text_batch)
                     if text_embed is not None and text_embed.shape[0] > 0:
                         embeddings_to_fuse['text'] = text_embed.to(self.device)
                     elif batch_size is not None and batch_size > 0: # Check if expected
                         print("[WARN] Forward: FinBERT output is None or empty.")
                else:
                     print("[WARN] Forward: Received empty text_batch list.")
            except Exception as e:
                print(f"[ERROR] FinBERT Encoder forward failed: {e}")
                traceback.print_exc()

        # --- 4. User Embedding ---
        if self.user_embedding and user_ids is not None:
             try:
                 user_ids_clamped = torch.clamp(user_ids, 0, self.user_embedding.num_embeddings - 1)
                 user_embed = self.user_embedding(user_ids_clamped.to(self.device))
                 if user_embed is not None and user_embed.shape[0] > 0:
                     embeddings_to_fuse['user'] = user_embed.to(self.device)
                 elif batch_size is not None and batch_size > 0: # Check if expected
                      print("[WARN] Forward: User embedding is None or empty.")
             except Exception as e:
                 print(f"[ERROR] User Embedding forward failed: {e}")
                 traceback.print_exc()


        # --- Pre-Fusion Checks ---
        if not embeddings_to_fuse:
            print("[ERROR] Forward: No embeddings available for fusion.")
            return None, None, None # Return three Nones

        # Determine reference batch size from the first available embedding
        ref_batch_size = next(iter(embeddings_to_fuse.values())).shape[0]

        if ref_batch_size == 0:
             print("[WARN] Forward: Fusion input batch size is 0. Returning None.")
             return None, None, None

        # Check consistency of batch sizes and device
        for name, emb in embeddings_to_fuse.items():
             if emb.shape[0] != ref_batch_size:
                  print(f"[ERROR] Forward: Mismatched batch size for {name}: {emb.shape[0]} vs {ref_batch_size}. Skipping fusion.")
                  return None, None, None
             if emb.device != self.device:
                  print(f"[ERROR] Forward: Embedding {name} is on wrong device: {emb.device} vs {self.device}. Skipping fusion.")
                  return None, None, None

        # --- 5. Fusion ---
        fused_representation = None
        try:
            fused_representation, _ = self.fusion_module(embeddings_to_fuse)
        except Exception as e:
             print(f"[ERROR] Fusion module forward failed: {e}")
             traceback.print_exc()
             return None, None, None # Return Nones if fusion fails

        # --- 6. Classify ---
        global_logits = None
        user_specific_logits = None
        scheduleC_logits = None # Initialize

        if fused_representation is None:
             print("[ERROR] Forward: Fused representation is None after fusion module.")
             return None, None, None

        try:
            if self.global_head:
                 global_logits = self.global_head(fused_representation)
            if self.user_specific_head:
                 user_specific_logits = self.user_specific_head(fused_representation)
            # <<< Conditional Schedule C Classification >>>
            if self.scheduleC_head:
                 scheduleC_logits = self.scheduleC_head(fused_representation)
        except Exception as e:
             print(f"[ERROR] Classifier head forward failed: {e}")
             traceback.print_exc()
             # Return Nones based on which heads exist
             return (None if self.global_head else global_logits,
                     None if self.user_specific_head else user_specific_logits,
                     None if self.scheduleC_head else scheduleC_logits)

        return global_logits, user_specific_logits, scheduleC_logits # Return all three

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

        indices_np = transaction_indices.cpu().numpy()
        device = self.device # Target device for some outputs (user_ids)

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
                 # Filter out invalid indices? Or return error? Returning error for now.
                 return None, None, None, None

            # --- 1. Subgraph Sampling (if GNN enabled) ---
            if self.use_gnn_encoder:
                try:
                    # Configuration for sampling - consider making these configurable
                    k_hops = self._graph_config.get('maml_sampling_hops', 2) # Default to 2 hops
                    num_neighbors = self._graph_config.get('maml_sampling_neighbors', [15, 10]) # Neighbors per hop
                    if len(num_neighbors) != k_hops:
                         print(f"[WARN] MAML sampling: num_neighbors length ({len(num_neighbors)}) != k_hops ({k_hops}). Adjusting.")
                         if len(num_neighbors) > k_hops: num_neighbors = num_neighbors[:k_hops]
                         else: num_neighbors = num_neighbors + [num_neighbors[-1]] * (k_hops - len(num_neighbors)) # Repeat last

                    # Use NeighborSampler for efficient k-hop sampling
                    from torch_geometric.loader import NeighborSampler

                    # Ensure indices are on CPU for sampler
                    seed_nodes_tensor = torch.from_numpy(indices_np).to('cpu').long()

                    # Need input_nodes format: (node_type, tensor_of_indices)
                    sampler = NeighborSampler(data=full_graph,
                                              num_neighbors=num_neighbors,
                                              input_nodes=('transaction', seed_nodes_tensor),
                                              # batch_size defaults to sampling all seeds at once
                                              # Other args like shuffle, num_workers usually not needed here
                                             )
                    # Sample the subgraph
                    # This returns batch_size, n_id (node ids in the sampled graph), adjs (edge info)
                    # We only need the first sample (as batch_size covers all seed nodes)
                    # sampled_data = next(iter(sampler)) # Get the single sampled subgraph data

                    # Reconstruct HeteroData from sampler output (simplified approach)
                    # Note: PyG versions might offer direct subgraph extraction. This is manual.
                    # Get all nodes involved in the k-hop neighborhood
                    # Extracting node features and edge indices requires careful mapping
                    # Let's try a simpler approach first: just extract features for seed nodes
                    # If GNN needs neighborhood, NeighborSampler approach is better.

                    # --- Simplified Feature Extraction for MAML GNN ---
                    # Option A: Pass the full graph and let forward handle slicing (less efficient)
                    # Option B: Extract only seed node features (no neighborhood aggregation)
                    # Option C: Implement proper subgraph extraction (complex)

                    # Choosing Option A for simplicity now, assuming forward pass handles slicing.
                    # If performance is an issue, revisit subgraph sampling.
                    graph_batch = full_graph # Pass the reference to the full graph
                    # Store seed indices directly for forward pass slicing
                    graph_batch.seed_indices_maml = seed_nodes_tensor

                except Exception as e_sample:
                    print(f"[ERROR] Subgraph sampling/preparation failed: {e_sample}")
                    traceback.print_exc()
                    graph_batch = None # Fallback to None if sampling fails
            # else: graph_batch remains None if GNN not used

            # --- 2. Fetch Features ONLY for SEED nodes (transaction_indices) ---
            # User IDs
            if 'user_id_code' in full_graph['transaction']:
                 user_ids = full_graph['transaction'].user_id_code[indices_np].to(device)
            else:
                 print("[WARN] _get_features_for_indices: 'user_id_code' not found in graph['transaction'].")

            # Text Data (Uses keys like '_raw_description', '_raw_memo')
            if self.use_text_encoder:
                 # Combine relevant raw text fields for the target indices
                 combined_texts = []
                 text_keys = [k for k in full_graph['transaction'].keys() if k.startswith('_raw_')]
                 if not text_keys:
                      print("[WARN] _get_features_for_indices: Text enabled but no '_raw_' fields found.")
                      text_batch = [""] * len(indices_np) # Return empty strings
                 else:
                     # Iterate through each requested index
                     for i in indices_np:
                          entry_texts = []
                          for key in text_keys:
                               try:
                                    # Access the list stored on the node and get the specific item
                                    text_list = full_graph['transaction'][key]
                                    entry_texts.append(str(text_list[i])) # Ensure string conversion
                               except IndexError:
                                    print(f"[ERROR] Index {i} out of bounds for text list '{key}' (len {len(text_list)}).")
                                    entry_texts.append("") # Append empty string on error
                               except Exception as e_text:
                                    print(f"[ERROR] Failed accessing text for index {i}, key '{key}': {e_text}")
                                    entry_texts.append("")
                          # Simple concatenation with space separator
                          combined_texts.append(" ".join(filter(None, entry_texts)).strip())
                     text_batch = combined_texts


            # Sequence Data
            if self.use_sequence_encoder:
                 if 'seq_features' in full_graph['transaction'] and 'seq_lengths' in full_graph['transaction']:
                      seq_feat = full_graph['transaction'].seq_features[indices_np]
                      seq_len = full_graph['transaction'].seq_lengths[indices_np]
                      sequence_batch = {'sequences': seq_feat, 'lengths': seq_len}
                      # Add categorical features if they exist
                      if 'seq_cat_features' in full_graph['transaction']:
                           sequence_batch['seq_cat_features'] = full_graph['transaction'].seq_cat_features[indices_np]
                 else:
                      print("[WARN] _get_features_for_indices: Sequence enabled but 'seq_features'/'seq_lengths' not found.")


        except Exception as e:
            print(f"[ERROR] _get_features_for_indices: Failed during feature extraction: {e}")
            traceback.print_exc()
            return None, None, None, None

        # Return features
        return graph_batch, sequence_batch, text_batch, user_ids


    def _get_fused_representation(self,
                                  # <<< Accepts full graph ref + seed indices OR a pre-sampled subgraph >>>
                                  graph_batch: Optional[HeteroData] = None, # Can be full graph or subgraph
                                  seed_indices_for_graph: Optional[torch.Tensor] = None, # Indices if graph_batch is full graph
                                  sequence_batch: Optional[Any] = None,
                                  text_batch: Optional[List[str]] = None,
                                  user_ids: Optional[torch.Tensor] = None
                                 ) -> Optional[torch.Tensor]:
        """Internal helper to run encoders and fusion module, returning only the fused tensor."""
        device = self.device
        embeddings_to_fuse = {}
        graph_embed = None

        # --- Determine effective batch size (number of seed nodes) ---
        effective_batch_size = None
        if seed_indices_for_graph is not None: effective_batch_size = seed_indices_for_graph.shape[0]
        elif text_batch is not None: effective_batch_size = len(text_batch)
        elif sequence_batch is not None and 'sequences' in sequence_batch: effective_batch_size = sequence_batch['sequences'].shape[0]
        elif user_ids is not None: effective_batch_size = user_ids.shape[0]

        if effective_batch_size is None or effective_batch_size == 0:
             print("[WARN] Fuse: Cannot determine effective batch size or size is 0.")
             return None

        # --- Move Inputs to Device ---
        if sequence_batch is not None: sequence_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sequence_batch.items()}
        if user_ids is not None: user_ids = user_ids.to(device)


        # --- 1. Graph Encoding (Call the actual encoder) ---
        if self.graph_encoder and graph_batch is not None:
            try:
                # Ensure graph data is on the correct device BEFORE passing to encoder
                graph_batch_dev = graph_batch.to(device)

                # Call the graph encoder
                node_embeddings_dict = self.graph_encoder(graph_batch_dev.x_dict, graph_batch_dev.edge_index_dict)

                if 'transaction' in node_embeddings_dict:
                    tx_embeds_all = node_embeddings_dict['transaction']

                    # Extract embeddings for the seed nodes specified
                    if seed_indices_for_graph is not None:
                         # Ensure indices are valid for the output embeddings
                         if seed_indices_for_graph.max() < tx_embeds_all.shape[0]:
                              graph_embed = tx_embeds_all[seed_indices_for_graph.to(device)] # Use provided indices
                         else:
                              print(f"[ERROR] Fuse: seed_indices_for_graph out of bounds for GNN output. Max index: {seed_indices_for_graph.max()}, Output shape: {tx_embeds_all.shape}")
                              graph_embed = None
                    else:
                         # If no specific indices, assume the first N embeddings correspond to the batch
                         # This requires graph_batch to be a pre-sampled subgraph where seed nodes are first
                         if tx_embeds_all.shape[0] >= effective_batch_size:
                              graph_embed = tx_embeds_all[:effective_batch_size]
                         else:
                              print(f"[WARN] Fuse: GNN output ({tx_embeds_all.shape[0]}) is smaller than effective batch size ({effective_batch_size}). Using available embeddings.")
                              graph_embed = tx_embeds_all # Use all available

                    if graph_embed is not None and graph_embed.shape[0] > 0:
                        # Final check on batch size consistency
                        if graph_embed.shape[0] != effective_batch_size:
                            print(f"[WARN] Fuse: Graph embedding size ({graph_embed.shape[0]}) mismatch with effective batch size ({effective_batch_size}).")
                            # Attempt to use graph_embed size as reference? Risky. Let pre-fusion check handle it.
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
        if self.sequence_encoder and sequence_batch is not None:
            try:
                seq_embed = self.sequence_encoder(sequence_batch, device=self.device)
                if seq_embed is not None and seq_embed.shape[0] > 0:
                    embeddings_to_fuse['sequence'] = seq_embed
            except Exception as e: print(f"[ERROR] Fuse (Sequence): {e}")

        # --- 3. Text Encoding ---
        if self.text_encoder and text_batch is not None:
             if text_batch: # Ensure not empty
                 try:
                     text_embed = self.text_encoder(text_batch) # Encoder handles device placement
                     if text_embed is not None and text_embed.shape[0] > 0:
                          embeddings_to_fuse['text'] = text_embed
                 except Exception as e: print(f"[ERROR] Fuse (Text): {e}")

        # --- 4. User Embedding ---
        if self.user_embedding and user_ids is not None:
            try:
                user_ids_clamped = torch.clamp(user_ids, 0, self.user_embedding.num_embeddings - 1)
                user_embed = self.user_embedding(user_ids_clamped) # Already on device
                if user_embed is not None and user_embed.shape[0] > 0: embeddings_to_fuse['user'] = user_embed
            except Exception as e: print(f"[ERROR] Fuse (User): {e}")

        # --- Pre-Fusion Checks & Fusion ---
        if not embeddings_to_fuse:
             print("[ERROR] Fuse: No embeddings available for fusion after processing inputs.")
             return None

        # Check consistency against effective_batch_size
        for name, emb in embeddings_to_fuse.items():
            if emb.shape[0] != effective_batch_size:
                 print(f"[ERROR] Fuse: Mismatched batch sizes! {name}:{emb.shape[0]} vs expected:{effective_batch_size}")
                 return None
            if emb.device != self.device:
                 print(f"[ERROR] Fuse: {name} on wrong device {emb.device}")
                 return None

        try:
            fused_representation, _ = self.fusion_module(embeddings_to_fuse)
            return fused_representation
        except Exception as e:
             print(f"[ERROR] Fusion module failed: {e}"); traceback.print_exc(); return None

    # --- Standard (non-MAML) _step ---
    def _step(self, batch: Any, batch_idx: int, stage: str) -> Optional[torch.Tensor]:
        """Common logic for standard (non-MAML) train/val/test steps."""
        # --- Unpack Batch & Extract Data (from NeighborLoader output) ---
        if not isinstance(batch, HeteroData):
            print(f"[ERROR] {stage}_step received unexpected batch type: {type(batch)}. Expected HeteroData.")
            return None

        graph_batch = batch # Input is the sampled subgraph from NeighborLoader
        batch_size = graph_batch['transaction'].batch_size # Get batch size from loader output

        # --- Extract Features and Labels from the Subgraph Batch ---
        sequence_batch = None
        text_batch = None
        user_ids = None
        global_target = None
        user_target = None
        scheduleC_target = None

        try:
            tx_store = graph_batch['transaction']

            # Features are already on the subgraph, just need to select the target nodes (first batch_size)
            # GNN features are handled by forward pass
            # Sequence
            if self.use_sequence_encoder and hasattr(tx_store, 'seq_features') and hasattr(tx_store, 'seq_lengths'):
                 if tx_store.seq_features.shape[0] >= batch_size:
                     sequence_batch = {
                         'sequences': tx_store.seq_features[:batch_size],
                         'lengths': tx_store.seq_lengths[:batch_size]
                     }
                     if hasattr(tx_store, 'seq_cat_features') and tx_store.seq_cat_features.shape[0] >= batch_size:
                          sequence_batch['seq_cat_features'] = tx_store.seq_cat_features[:batch_size]
                 else: print(f"[WARN] {stage}_step: Insufficient seq_features elements in batch.")

            # Text (requires retrieving raw text based on original indices)
            # This requires original indices to be passed by NeighborLoader or accessible.
            # NeighborLoader usually passes 'n_id' (original node IDs).
            if self.use_text_encoder and hasattr(tx_store, 'n_id'):
                 original_indices = tx_store.n_id[:batch_size] # Original indices of the batch nodes
                 # Fetch raw text from the full graph reference using these original indices
                 _, _, text_batch, _ = self._get_features_for_indices(original_indices)
                 if text_batch is None:
                     print(f"[WARN] {stage}_step: Failed to retrieve text for batch indices.")
                     text_batch = [""] * batch_size # Fallback
            elif self.use_text_encoder:
                 print(f"[WARN] {stage}_step: Text enabled, but cannot get original indices ('n_id') from batch.")

            # User ID
            if hasattr(tx_store, 'user_id_code'): # Assuming user_id_code is copied by loader
                 if tx_store.user_id_code.shape[0] >= batch_size:
                     user_ids = tx_store.user_id_code[:batch_size]
                 else: print(f"[WARN] {stage}_step: Insufficient user_id_code elements in batch.")
            elif hasattr(tx_store, 'n_id') and self._full_data_ref: # Fallback: get from full graph
                 original_indices = tx_store.n_id[:batch_size]
                 if 'user_id_code' in self._full_data_ref['transaction']:
                     user_ids = self._full_data_ref['transaction'].user_id_code[original_indices]
                 else: print(f"[WARN] {stage}_step: Cannot find user_id_code in full graph data.")


            # Labels
            if hasattr(tx_store, 'y_global') and tx_store.y_global.shape[0] >= batch_size:
                global_target = tx_store.y_global[:batch_size]
            if hasattr(tx_store, 'y_user') and tx_store.y_user.shape[0] >= batch_size:
                user_target = tx_store.y_user[:batch_size]
            if self.use_scheduleC_head and hasattr(tx_store, 'y_scheduleC'):
                 if tx_store.y_scheduleC.shape[0] >= batch_size:
                     scheduleC_target = tx_store.y_scheduleC[:batch_size]

        except Exception as e:
            print(f"[ERROR] Failed during standard batch data extraction in {stage}_step: {e}")
            traceback.print_exc()
            return None

        # --- Validation after extraction ---
        # ... (existing validation checks, ensure they use extracted variables) ...
        target_available = False
        if self.global_head and global_target is not None: target_available = True
        if self.user_specific_head and user_target is not None: target_available = True
        if self.scheduleC_head and scheduleC_target is not None: target_available = True
        if not target_available and stage != 'predict':
             print(f"[WARN] {stage}_step (batch {batch_idx}): No target labels found. Skipping batch.")
             return None
        if self.user_embedding and user_ids is None:
             print(f"[WARN] {stage}_step (batch {batch_idx}): user_ids is None but user embedding is active. Skipping batch.")
             return None
        # Add checks for sequence_batch / text_batch if needed

        # --- Forward pass ---
        # Pass the subgraph directly, forward pass expects HeteroData
        global_logits, user_specific_logits, scheduleC_logits = self(
            graph_batch=graph_batch, # Pass the subgraph from NeighborLoader
            sequence_batch=sequence_batch,
            text_batch=text_batch,
            user_ids=user_ids,
            batch_size=batch_size # Pass size explicitly
        )

        # --- Loss Calculation ---
        logits_produced = global_logits is not None or user_specific_logits is not None or scheduleC_logits is not None
        if not logits_produced:
             print(f"[WARN] {stage}_step (batch {batch_idx}): No logits produced by forward pass. Skipping loss calculation.")
             return None

        # Pass targets (should be on CPU or correct device for loss function)
        total_loss, loss_global, loss_user, loss_scheduleC = self._calculate_mtl_loss(
            global_logits, global_target, # Pass targets directly
            user_specific_logits, user_target,
            scheduleC_logits, scheduleC_target
        )

        # --- Accuracy Calculation ---
        # Use numerator/denominator for accurate aggregation
        global_correct, global_total = self._calculate_accuracy_numerator_denominator(global_logits, global_target)
        user_correct, user_total = self._calculate_accuracy_numerator_denominator(user_specific_logits, user_target)
        scheduleC_correct, scheduleC_total = self._calculate_accuracy_numerator_denominator(scheduleC_logits, scheduleC_target)

        # --- Logging & Return ---
        # Determine log batch size from any available target count
        log_batch_size = global_total or user_total or scheduleC_total or 0
        if log_batch_size == 0: log_batch_size = 1 # Avoid division by zero

        log_dict = { f'{stage}/total_loss': total_loss }
        # Only log metrics for active heads/losses
        if self.global_head:
             log_dict[f'{stage}/global_loss'] = loss_global
             log_dict[f'{stage}/acc_global_num'] = float(global_correct)
             log_dict[f'{stage}/acc_global_den'] = float(global_total)
             avg_acc_global = global_correct / global_total if global_total > 0 else 0.0
             log_dict[f'{stage}/acc_global'] = avg_acc_global # Log average for prog bar

        if self.user_specific_head:
             log_dict[f'{stage}/user_loss'] = loss_user
             log_dict[f'{stage}/acc_user_num'] = float(user_correct)
             log_dict[f'{stage}/acc_user_den'] = float(user_total)
             avg_acc_user = user_correct / user_total if user_total > 0 else 0.0
             log_dict[f'{stage}/acc_user'] = avg_acc_user

        if self.scheduleC_head:
             log_dict[f'{stage}/scheduleC_loss'] = loss_scheduleC
             log_dict[f'{stage}/acc_scheduleC_num'] = float(scheduleC_correct)
             log_dict[f'{stage}/acc_scheduleC_den'] = float(scheduleC_total)
             avg_acc_schedC = scheduleC_correct / scheduleC_total if scheduleC_total > 0 else 0.0
             log_dict[f'{stage}/acc_scheduleC'] = avg_acc_schedC

        # Use sync_dist=True for distributed training
        self.log_dict(log_dict, on_step=(stage=='train'), on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)

        # Return total_loss only for the training stage driver
        return total_loss if stage == 'train' else None

    # --- Lightning Hooks ---
    # training_step modified above to handle MAML case correctly

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        stage = 'val'
        if self._use_maml_runtime_flag:
             batch_of_tasks = None
             if isinstance(batch, list): batch_of_tasks = batch
             elif isinstance(batch, dict) and 'support' in batch: batch_of_tasks = self._reconstruct_maml_batch(batch)

             if batch_of_tasks:
                 self._meta_eval_step(batch_of_tasks, batch_idx, stage=stage)
             else: print(f"[ERROR] {stage}_step: Unexpected or failed reconstruction of batch type for MAML: {type(batch)}")
        else:
             self._step(batch, batch_idx, stage=stage)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        stage = 'test'
        if self._use_maml_runtime_flag:
             batch_of_tasks = None
             if isinstance(batch, list): batch_of_tasks = batch
             elif isinstance(batch, dict) and 'support' in batch: batch_of_tasks = self._reconstruct_maml_batch(batch)

             if batch_of_tasks:
                 self._meta_eval_step(batch_of_tasks, batch_idx, stage=stage)
             else: print(f"[ERROR] {stage}_step: Unexpected or failed reconstruction of batch type for MAML: {type(batch)}")
        else:
             self._step(batch, batch_idx, stage=stage)

    # _reconstruct_maml_batch added below

    def configure_optimizers(self):
        # Only optimize parameters that require grad
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())

        # Determine learning rate based on MAML mode or not
        # Use meta_lr for the outer loop MAML optimizer if specified, else use base learning_rate
        lr = self.hparams.meta_lr if self._use_maml_runtime_flag and hasattr(self.hparams, 'meta_lr') else self.hparams.learning_rate

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=lr,
            weight_decay=self.hparams.weight_decay
        )
        print(f"[INFO] configure_optimizers: Using LR={lr:.2e} (MAML Mode: {self._use_maml_runtime_flag})")
        # Add scheduler later if needed
        return optimizer

    # _meta_eval_step added below
    # _reconstruct_maml_batch added below

    # --- MAML Helper Methods ---
    def _meta_eval_step(self, batch_of_tasks: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], batch_idx: int, stage: str) -> None:
        """ Performs MAML evaluation (adaptation + query evaluation) without meta-update. """
        total_outer_loss = 0.0
        total_query_acc_numerator = 0.0
        total_query_samples = 0
        tasks_processed = 0

        target_head_attr = f"{self.hparams.maml_head_label_type}_specific_head"
        target_loss_attr = f"focal_loss_{self.hparams.maml_head_label_type}"
        if not hasattr(self, target_head_attr) or getattr(self, target_head_attr) is None:
            print(f"[{stage}] MAML target head '{target_head_attr}' not found or is None. Skipping eval batch.")
            return
        if not hasattr(self, target_loss_attr) or getattr(self, target_loss_attr) is None:
            print(f"[{stage}] MAML target loss '{target_loss_attr}' not found or is None. Skipping eval batch.")
            return

        original_head = getattr(self, target_head_attr)
        loss_fn = getattr(self, target_loss_attr)


        for task_data in batch_of_tasks:
            support_indices, support_labels = task_data['support']
            query_indices, query_labels = task_data['query']

            support_labels = support_labels.to(self.device)
            query_labels = query_labels.to(self.device)

            if support_indices.numel() == 0 or query_indices.numel() == 0:
                continue # Skip empty tasks

            # --- Inner Loop Adaptation (Enable Grads for adaptation itself) ---
            with torch.enable_grad(): # Gradients needed for FOMAML update
                learner = l2l.clone_module(original_head)

                inner_loop_failed = False
                for _ in range(self.hparams.adaptation_steps):
                    # Pass seed indices explicitly to fused representation method
                    graph_batch_supp, seq_batch_supp, text_batch_supp, user_ids_supp = self._get_features_for_indices(support_indices)
                    if graph_batch_supp is None and seq_batch_supp is None and text_batch_supp is None and user_ids_supp is None:
                         inner_loop_failed=True; break

                    support_fused = self._get_fused_representation(
                        graph_batch=graph_batch_supp, # Can be full graph
                        seed_indices_for_graph=support_indices, # Pass indices
                        sequence_batch=seq_batch_supp,
                        text_batch=text_batch_supp,
                        user_ids=user_ids_supp,
                    )
                    if support_fused is None: inner_loop_failed=True; break

                    support_preds = learner(support_fused)
                    inner_loss = loss_fn(support_preds, support_labels)

                    grads = torch.autograd.grad(inner_loss,
                                                learner.parameters(),
                                                create_graph=False) # FOMAML

                    l2l.update_module(learner, updates=grads, lr=self.hparams.inner_lr)
            # --- End Inner Loop ---
            if inner_loop_failed: print(f"[{stage}] Skipping task eval due to inner loop feature failure."); continue

            # --- Evaluate on Query Set (No Grads needed for eval) ---
            with torch.no_grad():
                graph_batch_qry, seq_batch_qry, text_batch_qry, user_ids_qry = self._get_features_for_indices(query_indices)
                if graph_batch_qry is None and seq_batch_qry is None and text_batch_qry is None and user_ids_qry is None:
                     print(f"[{stage}] Skipping task eval due to query feature failure."); continue

                query_fused = self._get_fused_representation(
                    graph_batch=graph_batch_qry, # Can be full graph
                    seed_indices_for_graph=query_indices, # Pass indices
                    sequence_batch=seq_batch_qry,
                    text_batch=text_batch_qry,
                    user_ids=user_ids_qry,
                )
                if query_fused is None: print(f"[{stage}] Skipping task eval due to query fusion failure."); continue

                query_preds = learner(query_fused)
                outer_loss = loss_fn(query_preds, query_labels)

                # --- Accumulate Results ---
                if not torch.isnan(outer_loss).any() and not torch.isinf(outer_loss).any():
                    total_outer_loss += outer_loss.item() # Accumulate scalar value
                    # Use numerator/denominator method for accuracy
                    current_acc_num, current_samples = self._calculate_accuracy_numerator_denominator(query_preds, query_labels)
                    total_query_acc_numerator += current_acc_num
                    total_query_samples += current_samples
                    tasks_processed += 1
                else:
                    print(f"[{stage}] Outer loss is NaN/Inf for user {task_data.get('user_id', 'Unknown')}. Skipping task result.")

        # --- Logging ---
        if tasks_processed > 0:
             avg_outer_loss = total_outer_loss / tasks_processed
             avg_query_acc = (total_query_acc_numerator / total_query_samples) if total_query_samples > 0 else 0.0

             log_batch_size = tasks_processed # Log per task
             log_dict = {
                 f'{stage}/meta_outer_loss': avg_outer_loss,
                 # Log accuracy numerator and denominator for correct aggregation
                 f'{stage}/meta_query_acc_{self.hparams.maml_head_label_type}_num': total_query_acc_numerator,
                 f'{stage}/meta_query_acc_{self.hparams.maml_head_label_type}_den': float(total_query_samples),
                 # Log averaged accuracy for convenience (prog bar)
                 f'{stage}/meta_query_acc_{self.hparams.maml_head_label_type}': avg_query_acc
             }
             self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)
        else:
             print(f"[{stage}] No tasks successfully processed in evaluation batch.")

        # No loss returned for eval steps

    def _reconstruct_maml_batch(self, collated_batch: Dict) -> Optional[List[Dict]]:
         """Helper to reconstruct a list of tasks from a collated batch dictionary."""
         try:
             # Determine number of tasks, check for tensor vs list structure from collate_fn
             if 'user_id' not in collated_batch:
                  print("[ERROR] Cannot reconstruct MAML batch: 'user_id' key missing.")
                  return None

             if isinstance(collated_batch['user_id'], list):
                  num_tasks = len(collated_batch['user_id'])
             elif torch.is_tensor(collated_batch['user_id']):
                  num_tasks = collated_batch['user_id'].shape[0]
             else:
                  print("[ERROR] Cannot determine number of tasks from user_id field type.")
                  return None

             if num_tasks == 0: return [] # Return empty list if no tasks

             reconstructed_batch = []
             # Check if keys exist before accessing
             support_data = collated_batch.get('support')
             query_data = collated_batch.get('query')
             user_id_batch = collated_batch['user_id']

             if not isinstance(support_data, (list, tuple)) or len(support_data) != 2:
                  print("[ERROR] Invalid 'support' data format in collated batch.")
                  return None
             if not isinstance(query_data, (list, tuple)) or len(query_data) != 2:
                  print("[ERROR] Invalid 'query' data format in collated batch.")
                  return None

             support_indices_batch = support_data[0]
             support_labels_batch = support_data[1]
             query_indices_batch = query_data[0]
             query_labels_batch = query_data[1]

             # Check if lengths match num_tasks
             if not (len(support_indices_batch) == num_tasks and
                     len(support_labels_batch) == num_tasks and
                     len(query_indices_batch) == num_tasks and
                     len(query_labels_batch) == num_tasks):
                   print("[ERROR] Mismatch between num_tasks and length of support/query data lists.")
                   return None


             for i in range(num_tasks):
                  # Access elements assuming default collate stacked tensors or kept lists
                  support_indices = support_indices_batch[i]
                  support_labels = support_labels_batch[i]
                  query_indices = query_indices_batch[i]
                  query_labels = query_labels_batch[i]
                  user_id = user_id_batch[i]
                  task = {
                      'support': (support_indices, support_labels),
                      'query': (query_indices, query_labels),
                      'user_id': user_id
                  }
                  reconstructed_batch.append(task)
             return reconstructed_batch
         except Exception as e:
             print(f"[ERROR] Failed to reconstruct MAML batch: {e}")
             print(f"Collated Batch type: {type(collated_batch)}, Batch keys: {collated_batch.keys()}")
             traceback.print_exc() # Print traceback for reconstruction errors
             return None

    def _calculate_mtl_loss(self,
                              global_logits: Optional[torch.Tensor], global_target: Optional[torch.Tensor],
                              user_logits: Optional[torch.Tensor], user_target: Optional[torch.Tensor],
                              scheduleC_logits: Optional[torch.Tensor], scheduleC_target: Optional[torch.Tensor] # Added Schedule C
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # Added Schedule C loss
         """Calculates individual and combined MTL loss."""
         loss_global = torch.tensor(0.0, device=self.device)
         loss_user = torch.tensor(0.0, device=self.device)
         loss_scheduleC = torch.tensor(0.0, device=self.device) # Initialize

         # --- Global Loss ---
         if self.focal_loss_global and global_logits is not None and global_target is not None:
             try:
                 loss_global = self.focal_loss_global(global_logits, global_target.to(global_logits.device))
             except Exception as e:
                 print(f"[ERROR] Global loss calculation failed: {e}")
                 loss_global = torch.tensor(0.0, device=self.device, requires_grad=True) # Ensure grad if error

         # --- User Loss ---
         if self.focal_loss_user and user_logits is not None and user_target is not None:
             try:
                 loss_user = self.focal_loss_user(user_logits, user_target.to(user_logits.device))
             except Exception as e:
                 print(f"[ERROR] User loss calculation failed: {e}")
                 loss_user = torch.tensor(0.0, device=self.device, requires_grad=True)

         # --- Schedule C Loss (Conditional) ---
         if self.focal_loss_scheduleC and scheduleC_logits is not None and scheduleC_target is not None:
             # Check if scheduleC weight exists
             if 'scheduleC' in self.hparams.mtl_weights:
                 try:
                      # Consider filtering out ignored indices (e.g., -1 for UNKNOWN) if needed
                      # loss_scheduleC = self.focal_loss_scheduleC(scheduleC_logits[scheduleC_target >= 0], scheduleC_target[scheduleC_target >= 0].to(scheduleC_logits.device))
                      loss_scheduleC = self.focal_loss_scheduleC(scheduleC_logits, scheduleC_target.to(scheduleC_logits.device))
                 except Exception as e:
                      print(f"[ERROR] Schedule C loss calculation failed: {e}")
                      loss_scheduleC = torch.tensor(0.0, device=self.device, requires_grad=True)
             # else: loss_scheduleC remains 0 if weight not specified

         # --- Combine Losses ---
         total_loss = torch.tensor(0.0, device=self.device)
         weight_global = self.hparams.mtl_weights.get('global', 0.0) # Default 0 if not specified
         weight_user = self.hparams.mtl_weights.get('user', 0.0)
         weight_scheduleC = self.hparams.mtl_weights.get('scheduleC', 0.0) # Default 0

         if weight_global > 0: total_loss += weight_global * loss_global
         if weight_user > 0: total_loss += weight_user * loss_user
         if weight_scheduleC > 0: total_loss += weight_scheduleC * loss_scheduleC

         # Handle potential NaN/Inf loss
         if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
              print(f"[WARN] Calculated total_loss is NaN/Inf. "
                    f"Global={loss_global.item():.4f}(w={weight_global:.2f}), "
                    f"User={loss_user.item():.4f}(w={weight_user:.2f}), "
                    f"SchedC={loss_scheduleC.item():.4f}(w={weight_scheduleC:.2f})")
              total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
              # Reset individual losses for logging consistency if total is NaN
              loss_global = torch.tensor(0.0, device=self.device)
              loss_user = torch.tensor(0.0, device=self.device)
              loss_scheduleC = torch.tensor(0.0, device=self.device)

         return total_loss, loss_global, loss_user, loss_scheduleC # Return all four

    def _calculate_accuracy(self, logits: Optional[torch.Tensor], targets: Optional[torch.Tensor]) -> torch.Tensor:
        """Helper to calculate accuracy, handles None inputs."""
        if logits is None or targets is None or logits.shape[0] == 0 or targets.shape[0] == 0:
             return torch.tensor(0.0, device=self.device)
        if logits.shape[0] != targets.shape[0]:
             print(f"[WARN] Accuracy calc: Logits batch ({logits.shape[0]}) != Targets batch ({targets.shape[0]})")
             return torch.tensor(0.0, device=self.device)
        # Check if targets contain only ignored values (e.g., -1)
        valid_targets = targets[targets >= 0] # Assuming negative values are ignored
        if valid_targets.shape[0] == 0:
            return torch.tensor(0.0, device=self.device) # No valid targets to calculate accuracy on

        with torch.no_grad():
            # Only calculate accuracy on valid targets
            logits_valid = logits[targets >= 0]
            targets_valid = valid_targets.to(logits.device).long()

            if logits_valid.shape[0] == 0: # Double check after filtering
                 return torch.tensor(0.0, device=self.device)

            preds = torch.argmax(logits_valid, dim=1)
            # Clamp targets based on the number of classes in logits AFTER filtering
            num_classes = logits_valid.shape[1]
            targets_clamped = torch.clamp(targets_valid, 0, num_classes - 1)
            correct = (preds == targets_clamped).float()
            accuracy = correct.mean()
        return accuracy

    def _calculate_accuracy_numerator_denominator(self, logits: Optional[torch.Tensor], targets: Optional[torch.Tensor]) -> Tuple[float, int]:
        """Helper to calculate sum of correct predictions and number of valid samples."""
        if logits is None or targets is None or logits.shape[0] == 0 or targets.shape[0] == 0:
             return 0.0, 0
        if logits.shape[0] != targets.shape[0]:
             print(f"[WARN] Accuracy calc: Logits batch ({logits.shape[0]}) != Targets batch ({targets.shape[0]})")
             return 0.0, 0

        # Ensure targets are on the same device as logits FOR COMPARISON
        targets_dev = targets.to(logits.device)

        # Check if targets contain only ignored values (e.g., -1)
        valid_mask = targets_dev >= 0 # Assuming negative values are ignored
        valid_targets = targets_dev[valid_mask]

        if valid_targets.numel() == 0:
            return 0.0, 0 # No valid targets to calculate accuracy on

        logits_valid = logits[valid_mask]

        if logits_valid.shape[0] == 0: # Double check after filtering
             return 0.0, 0

        with torch.no_grad():
            preds = torch.argmax(logits_valid, dim=1)
            # Clamp targets based on the number of classes in logits AFTER filtering
            num_classes = logits_valid.shape[1]
            # Ensure targets are long type for comparison
            targets_clamped = torch.clamp(valid_targets.long(), 0, num_classes - 1)
            correct_sum = (preds == targets_clamped).float().sum().item()
            num_valid_samples = valid_targets.numel()

        return correct_sum, num_valid_samples