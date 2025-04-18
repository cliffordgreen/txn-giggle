import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import learn2learn as l2l

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
                 # Optional reference to full data (needed for efficient MAML feature fetching)
                 # <<< MODIFIED: Expecting the raw DataFrame for MAML feature fetching >>>
                 full_data_ref: Optional[pd.DataFrame] = None, # Raw transactions_df
                 ):
        super().__init__()
        # Store simple hyperparameters automatically
        self.save_hyperparameters('learning_rate', 'weight_decay',
                                'mtl_weights', 'focal_loss_alpha', 'focal_loss_gamma',
                                'use_maml', 'inner_lr', 'adaptation_steps', 'maml_head_label_type')

        # <<< FIX: Explicitly set use_maml AFTER save_hyperparameters >>>
        # This ensures the value passed during init isn't overwritten by potential
        # checkpoint hparam loading during save_hyperparameters call.
        self.hparams.use_maml = use_maml # Use the argument passed to __init__

        # <<< Store flag reliably for runtime checks >>>
        self._use_maml_runtime_flag = use_maml

        # --- MAML Setup --- 
        if self._use_maml_runtime_flag:
             print("[INFO] MAML Mode Enabled.")
             self.automatic_optimization = False # Essential for MAML's manual optimization
             # Check if the target MAML head exists
             if self.hparams.maml_head_label_type == 'user' and model_config.get('num_user_classes', 0) == 0:
                  raise ValueError("MAML target is 'user', but num_user_classes is 0 in model_config.")
             if self.hparams.maml_head_label_type == 'global' and model_config.get('num_global_classes', 0) == 0:
                  raise ValueError("MAML target is 'global', but num_global_classes is 0 in model_config.")
             print(f"[INFO] MAML will adapt the '{self.hparams.maml_head_label_type}' head.")

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
        if self._use_maml_runtime_flag and self._full_data_ref is None:
             print("[WARN] MAML mode enabled but full_data_ref (raw DataFrame) was not provided to the model. Feature fetching will fail.")

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
                 self.graph_encoder = HGT(
                    in_channels=self._graph_config.get('in_channels', -1), # HGT can infer if -1
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

        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        print(f"[INFO] User Embedding Initialized. Num Users: {num_users}, Output Dim: {user_embed_dim}")

        # --- 2. Fusion Module ---
        fusion_input_dims = {}
        if self.use_gnn_encoder and self.graph_encoder:
             fusion_input_dims['graph'] = graph_out_dim
        if self.use_sequence_encoder and self.sequence_encoder:
             fusion_input_dims['sequence'] = seq_out_dim
        if self.use_text_encoder and self.text_encoder:
             fusion_input_dims['text'] = text_out_dim
        # Always include user embedding if num_users > 0
        if num_users > 0:
            fusion_input_dims['user'] = user_embed_dim
        else:
            print("[WARN] num_users is 0. User embedding will not be used in fusion.")

        if not fusion_input_dims:
            raise ValueError("No modalities are enabled or configured correctly. At least one encoder (graph, sequence, text) or user embedding must be active.")
        if not self._fusion_config:
             raise ValueError("'fusion_params' missing from model_config, cannot initialize fusion module.")

        self.fusion_module = AttentionFusion(
            modality_dims=fusion_input_dims,
            hidden_dim=self._fusion_config.get('hidden_dim', 128), # Provide default
            output_dim=self._fusion_config.get('output_dim', 128), # Provide default
            dropout=self._fusion_config.get('dropout', 0.1)
        )
        fused_dim = self._fusion_config['output_dim']
        print(f"[INFO] Attention Fusion Initialized. Input Dims: {fusion_input_dims}, Output Dim: {fused_dim}")

        # --- 3. Classification Heads ---
        self.global_head = None
        if num_global_classes > 0:
             self.global_head = nn.Linear(fused_dim, num_global_classes)
             print(f"[INFO] Global Classifier Initialized: Output Classes={num_global_classes}")
        else:
             print("[INFO] Global classification head skipped (num_global_classes=0).")

        self.user_specific_head = None
        if num_user_classes > 0:
             self.user_specific_head = nn.Linear(fused_dim, num_user_classes)
             print(f"[INFO] User Classifier Initialized: Output Classes={num_user_classes}")
        else:
             print("[INFO] User classification head skipped (num_user_classes=0).")

        # <<< Conditional Schedule C Head >>>
        self.scheduleC_head = None
        if self.use_scheduleC_head:
             self.scheduleC_head = nn.Linear(fused_dim, num_scheduleC_classes)
             print(f"[INFO] Schedule C Classifier Initialized: Output Classes={num_scheduleC_classes}")
        else:
             print("[INFO] Schedule C classification head skipped (num_scheduleC_classes=0).")

        # --- 4. Loss Functions ---
        self.focal_loss_global = None
        if self.global_head:
             self.focal_loss_global = FocalLoss(
                 alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma,
                 num_classes=num_global_classes
             )
             print("[INFO] Global Focal Loss Initialized.")

        self.focal_loss_user = None
        if self.user_specific_head:
             self.focal_loss_user = FocalLoss(
                 alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma,
                 num_classes=num_user_classes
             )
             print("[INFO] User Focal Loss Initialized.")

        # <<< Conditional Schedule C Loss >>>
        self.focal_loss_scheduleC = None
        if self.scheduleC_head:
             self.focal_loss_scheduleC = FocalLoss(
                 alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma,
                 num_classes=num_scheduleC_classes
             )
             print("[INFO] Schedule C Focal Loss Initialized.")

        # Store complex configs manually (needed if loading from checkpoint)
        self._model_config = model_config
        self._graph_config = model_config.get('graph_encoder_params', {}) # Handle missing key
        self._sequence_config = model_config.get('sequence_encoder_params', {})
        self._text_config = model_config.get('text_encoder_params', {})
        self._fusion_config = model_config.get('fusion_params', {})

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
                     if batch_size is None:
                          if user_ids is not None: batch_size = user_ids.shape[0]
                          # Add fallback using maybe _raw_text length if text_encoder is on?
                          elif self.use_text_encoder and text_batch is not None: batch_size = len(text_batch)
                          else: raise ValueError("Forward pass needs batch_size if graph_batch is provided.")

                     num_nodes_in_batch_output = node_embeddings_dict['transaction'].shape[0]
                     if batch_size > num_nodes_in_batch_output:
                         print(f"[WARN] Forward: batch_size ({batch_size}) > GNN output nodes ({num_nodes_in_batch_output}). Slicing available nodes.")
                         graph_embed = node_embeddings_dict['transaction'][:num_nodes_in_batch_output]
                     elif batch_size > 0:
                          graph_embed = node_embeddings_dict['transaction'][:batch_size]
                     # else: batch_size might be 0, graph_embed remains None

                     if graph_embed is not None and graph_embed.shape[0] > 0: # Check size > 0
                         embeddings_to_fuse['graph'] = graph_embed.to(self.device)
                     elif batch_size > 0: # Only warn if batch_size was expected to be > 0
                          print("[WARN] Forward: HGT 'transaction' embedding is None or empty after slicing.")

                else:
                    print("[WARN] Forward: HGT output missing 'transaction' embeddings.")
            except Exception as e:
                print(f"[ERROR] HGT Encoder forward failed: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for GNN errors

        # --- 2. Sequence Encoding ---
        if self.sequence_encoder and sequence_batch is not None:
            try:
                seq_embed = self.sequence_encoder(sequence_batch, device=self.device)
                if seq_embed is not None and seq_embed.shape[0] > 0:
                    # Ensure batch size matches graph if graph is used
                    if graph_embed is not None and seq_embed.shape[0] != graph_embed.shape[0]:
                         print(f"[ERROR] Forward: Sequence batch size ({seq_embed.shape[0]}) mismatch with Graph ({graph_embed.shape[0]})")
                    else:
                        embeddings_to_fuse['sequence'] = seq_embed.to(self.device)
                elif graph_embed is not None and graph_embed.shape[0] > 0: # Check if expected based on graph
                    print("[WARN] Forward: TFT output is None or empty.")
            except Exception as e:
                print(f"[ERROR] TFT Encoder forward failed: {e}")

        # --- 3. Text Encoding ---
        if self.text_encoder and text_batch is not None:
            try:
                text_embed = self.text_encoder(text_batch)
                if text_embed is not None and text_embed.shape[0] > 0:
                    # Ensure batch size matches graph if graph is used
                    ref_shape = graph_embed.shape[0] if graph_embed is not None else (seq_embed.shape[0] if 'sequence' in embeddings_to_fuse else None)
                    if ref_shape is not None and text_embed.shape[0] != ref_shape:
                         print(f"[ERROR] Forward: Text batch size ({text_embed.shape[0]}) mismatch with reference ({ref_shape})")
                    else:
                         embeddings_to_fuse['text'] = text_embed.to(self.device)
                elif ref_shape is not None and ref_shape > 0: # Check if expected
                    print("[WARN] Forward: FinBERT output is None or empty.")

            except Exception as e:
                print(f"[ERROR] FinBERT Encoder forward failed: {e}")

        # --- 4. User Embedding ---
        if self.user_embedding and user_ids is not None:
             # Check if num_users > 0 before trying to embed
            if self.user_embedding.num_embeddings > 0:
                try:
                    # Clamp user_ids to be safe
                    user_ids_clamped = torch.clamp(user_ids, 0, self.user_embedding.num_embeddings - 1)
                    user_embed = self.user_embedding(user_ids_clamped.to(self.device))
                    if user_embed is not None and user_embed.shape[0] > 0:
                        # Ensure batch size matches graph if graph is used
                        ref_shape = graph_embed.shape[0] if graph_embed is not None else (embeddings_to_fuse.get('sequence', embeddings_to_fuse.get('text', None)).shape[0] if embeddings_to_fuse else None)
                        if ref_shape is not None and user_embed.shape[0] != ref_shape:
                            print(f"[ERROR] Forward: User ID batch size ({user_embed.shape[0]}) mismatch with reference ({ref_shape})")
                        else:
                            embeddings_to_fuse['user'] = user_embed.to(self.device)
                    elif ref_shape is not None and ref_shape > 0: # Check if expected
                         print("[WARN] Forward: User embedding is None or empty.")
                except Exception as e:
                    print(f"[ERROR] User Embedding forward failed: {e}")
            else:
                print("[WARN] Forward: Skipping user embedding as num_users is 0.")


        # --- Pre-Fusion Checks ---
        if not embeddings_to_fuse:
            print("[ERROR] Forward: No embeddings available for fusion.")
            return None, None, None # Return three Nones

        ref_batch_size = None
        first_key = next(iter(embeddings_to_fuse))
        ref_batch_size = embeddings_to_fuse[first_key].shape[0]

        if ref_batch_size == 0:
             print("[WARN] Forward: Fusion input batch size is 0. Returning None.")
             return None, None, None

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
             # Return Nones based on which heads exist
             return (None if self.global_head else global_logits,
                     None if self.user_specific_head else user_specific_logits,
                     None if self.scheduleC_head else scheduleC_logits)

        return global_logits, user_specific_logits, scheduleC_logits # Return all three

    def _get_features_for_indices(self, transaction_indices: torch.Tensor) -> Tuple[Optional[HeteroData], Optional[Any], Optional[List[str]], Optional[torch.Tensor]]:
        """
        Fetches input features for specific transaction indices FROM THE STORED RAW DATAFRAME.
        Crucial for MAML where support/query sets are defined by indices.

        Args:
            transaction_indices: Tensor of *original* DataFrame indices.

        Returns:
            Tuple containing:
                - graph_batch: None (GNN feature fetching not implemented for MAML)
                - sequence_batch: Dict with basic sequence features (or None if disabled/failed)
                - text_batch: List of combined raw text strings (or None if failed)
                - user_ids: Tensor of user ID codes (or None if failed)
        """
        if self._full_data_ref is None:
            print("[ERROR] _get_features_for_indices: _full_data_ref (raw DataFrame) is None. Cannot fetch features.")
            return None, None, None, None

        if transaction_indices is None or transaction_indices.numel() == 0:
             print("[WARN] _get_features_for_indices: Received empty or None indices.")
             return None, None, None, None

        # Ensure indices are on CPU and numpy for DataFrame indexing
        try:
             indices_np = transaction_indices.cpu().numpy()
             # Retrieve corresponding rows from the original DataFrame
             df_slice = self._full_data_ref.iloc[indices_np]
        except IndexError as e:
             max_idx = self._full_data_ref.index.max()
             offending_indices = indices_np[indices_np > max_idx]
             print(f"[ERROR] _get_features_for_indices: IndexError during DataFrame lookup. Max df index: {max_idx}. Offending input indices > max: {offending_indices}. Error: {e}")
             return None, None, None, None
        except Exception as e:
             print(f"[ERROR] _get_features_for_indices: Error during DataFrame lookup for indices {indices_np}: {e}")
             return None, None, None, None

        # print(f"[DEBUG] _get_features_for_indices: Fetched {len(df_slice)} rows for {len(indices_np)} indices.")

        graph_batch = None # GNN features skipped for MAML
        sequence_batch = None
        text_batch = None
        user_ids = None

        try:
            # 1. User IDs (Assuming 'user_id_code' was added during DataModule setup/preprocessing)
            # It's safer if the DataModule adds this column to the original df passed here
            if 'user_id_code' in df_slice.columns:
                user_ids = torch.tensor(df_slice['user_id_code'].values, dtype=torch.long)
            else:
                print("[WARN] _get_features_for_indices: 'user_id_code' column not found in _full_data_ref. User embedding will be missing.")

            # 2. Text Data
            if self.use_text_encoder:
                # Combine text columns present in the dataframe slice
                text_cols_to_use = [col for col in ['raw_description', 'memo', 'merchant_name'] if col in df_slice.columns]
                if text_cols_to_use:
                    # Fill NaNs just in case, concatenate, convert to list
                    text_batch = df_slice[text_cols_to_use].fillna('').astype(str).apply(' || '.join, axis=1).tolist()
                else:
                    print("[WARN] _get_features_for_indices: No standard text columns found for text encoder.")
                    text_batch = ['' for _ in range(len(df_slice))] # Return empty strings

            # 3. Sequence Data (Basic Implementation)
            if self.use_sequence_encoder:
                # WARNING: This uses a simplified feature set, likely incompatible with pre-trained TFT.
                # Consider disabling sequence encoder for MAML (`--use_sequence False`)
                # unless this preprocessing is adapted.
                required_seq_cols = ['amount', 'timestamp'] # Need amount and timestamp
                if all(col in df_slice.columns for col in required_seq_cols):
                    # Basic features: amount, hour_sin, hour_cos, day_sin, day_cos
                    # Note: This doesn't create sequences of past transactions, just features of the indexed ones.
                    # A true sequential input for MAML would require more complex logic here.
                    amount = torch.tensor(df_slice['amount'].fillna(0.0).values, dtype=torch.float).unsqueeze(1)
                    timestamps = pd.to_datetime(df_slice['timestamp'], errors='coerce')
                    hour = timestamps.dt.hour.fillna(0).astype(int)
                    day = timestamps.dt.weekday.fillna(0).astype(int)
                    hour_sin = torch.tensor(np.sin(2 * np.pi * hour / 24), dtype=torch.float).unsqueeze(1)
                    hour_cos = torch.tensor(np.cos(2 * np.pi * hour / 24), dtype=torch.float).unsqueeze(1)
                    day_sin = torch.tensor(np.sin(2 * np.pi * day / 7), dtype=torch.float).unsqueeze(1)
                    day_cos = torch.tensor(np.cos(2 * np.pi * day / 7), dtype=torch.float).unsqueeze(1)

                    # Combine features - Shape: [batch_size, num_features]
                    seq_features = torch.cat([amount, hour_sin, hour_cos, day_sin, day_cos], dim=1)

                    # Package for compatibility (even though it's not really a sequence)
                    sequence_batch = {
                        # Treat each transaction as a sequence of length 1
                        'sequences': seq_features.unsqueeze(1), # Shape: [batch_size, 1, num_features]
                        'lengths': torch.ones(len(df_slice), dtype=torch.long)
                    }
                    # print("[WARN] _get_features_for_indices: Using simplified, non-sequential features for sequence_batch in MAML.")
                else:
                    print("[WARN] _get_features_for_indices: Missing required columns for sequence features (amount, timestamp). Skipping.")

        except KeyError as e:
             print(f"[ERROR] _get_features_for_indices: KeyError during feature extraction for indices {indices_np}. Missing column? Error: {e}")
             return None, None, None, None
        except Exception as e:
            print(f"[ERROR] _get_features_for_indices: Failed during feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

        # Return fetched features (Graph is None)
        return graph_batch, sequence_batch, text_batch, user_ids

    def get_fused_features(self,
                           graph_batch: Optional[HeteroData] = None,
                           sequence_batch: Optional[Any] = None,
                           text_batch: Optional[List[str]] = None,
                           user_ids: Optional[torch.Tensor] = None,
                           batch_size: Optional[int] = None
                           ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Fetches fused features from the model.

        Args:
            graph_batch: HeteroData object containing graph data.
            sequence_batch: Dict containing sequence data.
            text_batch: List of text strings.
            user_ids: Tensor of user IDs.
            batch_size: Optional batch size for slicing GNN output.

        Returns:
            Tuple containing:
                - global_logits: Tensor of global class logits.
                - user_specific_logits: Tensor of user-specific class logits.
                - scheduleC_logits: Tensor of schedule C class logits.
        """
        # Ensure all inputs are on the correct device
        device = self.device
        if graph_batch:
            graph_batch = graph_batch.to(device)
        if sequence_batch:
            sequence_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sequence_batch.items()}
        if text_batch:
            text_batch = [t.to(device) for t in text_batch]
        if user_ids:
            user_ids = user_ids.to(device)

        # Forward pass through the model
        global_logits, user_specific_logits, scheduleC_logits = self(
            graph_batch=graph_batch,
            sequence_batch=sequence_batch,
            text_batch=text_batch,
            user_ids=user_ids,
            batch_size=batch_size
        )

        return global_logits, user_specific_logits, scheduleC_logits

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

    def _step(self, batch: Any, batch_idx: int, stage: str) -> Optional[torch.Tensor]:
        """Common logic for train/val/test steps."""
        # --- Unpack Batch & Extract Data ---
        graph_batch = None
        sequence_batch = None
        text_batch = None
        user_ids = None
        global_target = None
        user_target = None
        scheduleC_target = None # Initialize
        batch_size = None

        # Assuming batch is HeteroData from HGTLoader if GNN is used
        if self.use_gnn_encoder:
            if not isinstance(batch, HeteroData):
                print(f"[ERROR] {stage}_step received unexpected batch type: {type(batch)} when GNN enabled.")
                return None
            graph_batch = batch # Pass the full sampled graph
        else:
             # Handle non-GNN batch format (e.g., a dictionary)
             # This needs specific implementation based on how non-GNN loader yields data
             print(f"[WARN] {stage}_step running without GNN. Assuming batch is a dictionary (implement proper handling).")
             if not isinstance(batch, dict):
                  print(f"[ERROR] {stage}_step received unexpected batch type: {type(batch)} when GNN disabled.")
                  return None
             # Example: Extract data needed for non-GNN modalities from the dict
             # sequence_batch = batch.get('sequence_data')
             # text_batch = batch.get('text_data')
             # user_ids = batch.get('user_ids')
             # global_target = batch.get('global_labels')
             # ... etc ...
             # Need to determine batch_size from the available data
             # batch_size = user_ids.shape[0] if user_ids is not None else len(text_batch) # Example
             # For now, we'll assume the required data is present in the HeteroData structure even if GNN isn't used for processing
             if not isinstance(batch, HeteroData): # Fallback check
                  print(f"[ERROR] {stage}_step requires HeteroData structure even if GNN is off for current data extraction logic.")
                  return None
             graph_batch = batch # Still use HeteroData for unpacking, just won't pass to GNN encoder


        try:
            if 'transaction' in graph_batch.node_types:
                tx_store = graph_batch['transaction']

                # Determine batch_size (number of seed nodes for HGTLoader, or total nodes if not HGT)
                # HGTLoader adds 'batch_size' attribute to the central node type store
                if hasattr(tx_store, 'batch_size') and tx_store.batch_size is not None:
                     batch_size = tx_store.batch_size
                     if batch_size == 0:
                          print(f"[WARN] {stage}_step (batch {batch_idx}): tx_store.batch_size is 0. Skipping batch.")
                          return None
                elif hasattr(tx_store, 'input_id'): # Fallback for HGTLoader if batch_size missing
                     batch_size = tx_store.input_id.shape[0]
                     if batch_size == 0: print(f"[WARN] {stage}_step (batch {batch_idx}): Determined batch_size=0 from input_id. Skipping.") ; return None
                     print(f"[INFO] {stage}_step (batch {batch_idx}): Used input_id length for batch_size: {batch_size}")
                elif hasattr(tx_store, 'num_nodes'):
                     # If not using HGTLoader, batch_size might just be the number of nodes in the batch graph
                     batch_size = tx_store.num_nodes
                     if batch_size == 0: print(f"[WARN] {stage}_step (batch {batch_idx}): Determined batch_size=0 from num_nodes. Skipping.") ; return None
                     # print(f"[INFO] {stage}_step (batch {batch_idx}): Used tx_store.num_nodes for batch_size: {batch_size}")
                else:
                     print(f"[ERROR] {stage}_step (batch {batch_idx}): Cannot determine batch_size.")
                     return None

                # --- Fetch data by slicing first batch_size elements (assumes target nodes are first) ---
                # This slicing logic is specific to HGTLoader's output where target nodes are first.
                # If using a different loader, data extraction needs adjustment.
                nodes_available = tx_store.num_nodes
                if batch_size > nodes_available:
                     print(f"[WARN] {stage}_step (batch {batch_idx}): Required batch_size ({batch_size}) > nodes available ({nodes_available}). Using available.")
                     slice_len = nodes_available
                else:
                     slice_len = batch_size

                if slice_len == 0: print(f"[WARN] {stage}_step (batch {batch_idx}): Effective batch size is 0 after checks. Skipping."); return None

                # User ID
                if hasattr(tx_store, 'user_id_code'):
                    if tx_store.user_id_code.shape[0] >= slice_len:
                        user_ids = tx_store.user_id_code[:slice_len]
                    else: print(f"[WARN] {stage}_step (batch {batch_idx}): Insufficient user_id_code elements.")

                # Text (_raw_text is Python list, handle differently)
                if self.use_text_encoder and hasattr(tx_store, '_raw_text'):
                    if tx_store._raw_text is not None and len(tx_store._raw_text) >= slice_len:
                         # _raw_text might be on the whole graph, need mapping if GNN sampled
                         if hasattr(tx_store, 'input_id'): # HGTLoader provides input_id map
                             original_indices = tx_store.input_id[:slice_len] # Get original indices of seed nodes
                             # Need the full graph's _raw_text IF _raw_text wasn't copied to batch
                             # This assumes _raw_text IS copied/sliced correctly by the loader
                             text_batch = [tx_store._raw_text[i] for i in range(slice_len)] # Simpler if loader handles it
                         elif len(tx_store._raw_text) == tx_store.num_nodes: # Assume direct mapping if no input_id
                              text_batch = tx_store._raw_text[:slice_len]
                         else:
                              print(f"[WARN] {stage}_step (batch {batch_idx}): Cannot map _raw_text, length mismatch ({len(tx_store._raw_text)}) vs nodes ({tx_store.num_nodes}).")
                    else: print(f"[WARN] {stage}_step (batch {batch_idx}): Missing or insufficient _raw_text elements.")

                # Sequence
                if self.use_sequence_encoder and hasattr(tx_store, 'seq_features') and hasattr(tx_store, 'seq_lengths'):
                    # Assume sequence features are already sliced correctly for the batch_size nodes
                    if tx_store.seq_features.shape[0] >= slice_len:
                        sequence_batch = {
                            'sequences': tx_store.seq_features[:slice_len],
                            'lengths': tx_store.seq_lengths[:slice_len]
                        }
                        # Add categorical features if they exist and match size
                        if hasattr(tx_store, 'seq_cat_features') and tx_store.seq_cat_features.shape[0] >= slice_len:
                             sequence_batch['seq_cat_features'] = tx_store.seq_cat_features[:slice_len]
                        # else: print(f"[DEBUG] {stage}_step (batch {batch_idx}): Missing or incorrectly sized seq_cat_features.")
                    else: print(f"[WARN] {stage}_step (batch {batch_idx}): Insufficient seq_features elements.")

                # Labels
                if hasattr(tx_store, 'y_global') and tx_store.y_global.shape[0] >= slice_len:
                    global_target = tx_store.y_global[:slice_len]
                # else: print(f"[DEBUG] {stage}_step (batch {batch_idx}): Cannot extract global_target.")

                if hasattr(tx_store, 'y_user') and tx_store.y_user.shape[0] >= slice_len:
                    user_target = tx_store.y_user[:slice_len]
                # else: print(f"[DEBUG] {stage}_step (batch {batch_idx}): Cannot extract user_target.")

                # <<< Conditional Schedule C Label Extraction >>>
                if self.use_scheduleC_head and hasattr(tx_store, 'y_scheduleC'):
                     if tx_store.y_scheduleC.shape[0] >= slice_len:
                         scheduleC_target = tx_store.y_scheduleC[:slice_len]
                     # else: print(f"[DEBUG] {stage}_step (batch {batch_idx}): Cannot extract scheduleC_target.")


            else: # 'transaction' node type not found
                 print(f"[WARN] {stage}_step (batch {batch_idx}): 'transaction' node type not found in batch.")
                 return None

        except Exception as e:
            print(f"[ERROR] Failed during batch data extraction in {stage}_step (batch {batch_idx}): {e}")
            import traceback
            traceback.print_exc()
            return None

        # --- Validation after extraction ---
        # Check if at least one target is available based on active heads
        target_available = False
        if self.global_head and global_target is not None: target_available = True
        if self.user_specific_head and user_target is not None: target_available = True
        if self.scheduleC_head and scheduleC_target is not None: target_available = True

        if not target_available and stage != 'predict': # Need labels for train/val/test
            print(f"[WARN] {stage}_step (batch {batch_idx}): No target labels found for active heads. Skipping batch.")
            return None
        # Check user_ids if user embedding is used
        if self.user_embedding and self.user_embedding.num_embeddings > 0 and user_ids is None:
             print(f"[WARN] {stage}_step (batch {batch_idx}): user_ids is None but user embedding is active. Skipping batch.")
             return None
        # Check other required inputs based on enabled modalities
        if self.use_sequence_encoder and sequence_batch is None:
              print(f"[WARN] {stage}_step (batch {batch_idx}): sequence_batch is None but sequence encoder is active. Skipping batch.")
              return None
        if self.use_text_encoder and text_batch is None:
              print(f"[WARN] {stage}_step (batch {batch_idx}): text_batch is None but text encoder is active. Skipping batch.")
              return None

        # --- Forward pass ---
        device = self.device
        # Move graph batch to device only if GNN is used
        if self.use_gnn_encoder and graph_batch is not None:
             graph_batch = graph_batch.to(device)
        # Move other inputs to device
        user_ids = user_ids.to(device) if user_ids is not None else None
        if isinstance(sequence_batch, dict):
             sequence_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in sequence_batch.items()}
        # Text batch remains on CPU, handled by text encoder

        # Pass determined batch_size to forward (crucial for GNN slicing)
        global_logits, user_specific_logits, scheduleC_logits = self( # Unpack third logit
            graph_batch=graph_batch if self.use_gnn_encoder else None, # Pass graph only if used
            sequence_batch=sequence_batch,
            text_batch=text_batch,
            user_ids=user_ids,
            batch_size=slice_len # Use the actual number of nodes being processed
        )

        # --- Loss Calculation ---
        # Check if any logits were produced (at least one head must be active)
        logits_produced = global_logits is not None or user_specific_logits is not None or scheduleC_logits is not None
        if not logits_produced:
             print(f"[WARN] {stage}_step (batch {batch_idx}): No logits produced by forward pass. Skipping loss calculation.")
             return None

        # Ensure targets are on CPU for the loss function (it handles moving them)
        total_loss, loss_global, loss_user, loss_scheduleC = self._calculate_mtl_loss( # Unpack fourth loss
            global_logits, global_target.cpu() if global_target is not None else None,
            user_specific_logits, user_target.cpu() if user_target is not None else None,
            scheduleC_logits, scheduleC_target.cpu() if scheduleC_target is not None else None # Pass schedule C items
        )

        # --- Accuracy Calculation ---
        acc_global = self._calculate_accuracy(global_logits, global_target)
        acc_user = self._calculate_accuracy(user_specific_logits, user_target)
        acc_scheduleC = self._calculate_accuracy(scheduleC_logits, scheduleC_target) # Calculate Schedule C Acc

        # --- Logging & Return ---
        # Determine log batch size from any available logits
        log_batch_size = 0
        if global_logits is not None: log_batch_size = global_logits.shape[0]
        elif user_specific_logits is not None: log_batch_size = user_specific_logits.shape[0]
        elif scheduleC_logits is not None: log_batch_size = scheduleC_logits.shape[0]
        if log_batch_size == 0:
            print(f"[WARN] {stage}_step (batch {batch_idx}): Log batch size is 0. Skipping logging.")
            # Still return loss for training step if calculated
            return total_loss if stage == 'train' and torch.is_tensor(total_loss) else None


        log_dict = { f'{stage}/total_loss': total_loss }
        # Only log metrics for active heads/losses
        if self.global_head:
             log_dict[f'{stage}/global_loss'] = loss_global
             log_dict[f'{stage}/acc_global'] = acc_global
        if self.user_specific_head:
             log_dict[f'{stage}/user_loss'] = loss_user
             log_dict[f'{stage}/acc_user'] = acc_user
        if self.scheduleC_head:
             log_dict[f'{stage}/scheduleC_loss'] = loss_scheduleC
             log_dict[f'{stage}/acc_scheduleC'] = acc_scheduleC

        # Use sync_dist=True for distributed training
        self.log_dict(log_dict, on_step=(stage=='train'), on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)


        # Return total_loss only for the training stage driver
        return total_loss if stage == 'train' else None

    # --- Standard Lightning Hooks ---
    def training_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        # <<< Use reliable runtime flag for dispatch >>>
        if self._use_maml_runtime_flag:
            meta_optimizer = self.optimizers() # Get meta-optimizer for manual update
            meta_optimizer.zero_grad()

            # --- Process Batch of Tasks --- 
            batch_of_tasks = None
            # Handle potential collation by default DataLoader
            if isinstance(batch, dict) and 'support' in batch and 'query' in batch:
                 # Reconstruct list of tasks if collated
                 try:
                     num_tasks = len(batch['user_id']) # Assuming user_id marks tasks
                     reconstructed_batch = []
                     for i in range(num_tasks):
                          # Adapt access based on how collate_fn might structure it
                          support_indices = batch['support'][0][i] # Assuming first dim is batch
                          support_labels = batch['support'][1][i]
                          query_indices = batch['query'][0][i]
                          query_labels = batch['query'][1][i]
                          user_id = batch['user_id'][i] # Assume user_id is indexable

                          task = {
                              'support': (support_indices, support_labels),
                              'query': (query_indices, query_labels),
                              'user_id': user_id
                          }
                          reconstructed_batch.append(task)
                     batch_of_tasks = reconstructed_batch # Use reconstructed list
                 except Exception as e:
                      print(f"[ERROR] Failed to reconstruct MAML batch in training_step: {e}")
                      print(f"Batch type: {type(batch)}, Batch keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}")
                      return None # Skip batch if reconstruction fails
            elif isinstance(batch, list): # Already a list of tasks
                 batch_of_tasks = batch
            else:
                 print(f"[ERROR] training_step: Unexpected batch type for MAML: {type(batch)}")
                 return None

            if batch_of_tasks is None: return None # Exit if batch processing failed

            # --- MAML Inner/Outer Loop Logic (Moved here from non-existent meta_training_step) --- 
            total_outer_loss = 0.0
            avg_query_acc = 0.0
            tasks_processed = 0

            target_head_attr = f"{self.hparams.maml_head_label_type}_head"
            target_loss_attr = f"focal_loss_{self.hparams.maml_head_label_type}"
            if not hasattr(self, target_head_attr) or getattr(self, target_head_attr) is None:
                print(f"[ERROR] MAML target head '{target_head_attr}' not found or is None.")
                return None
            if not hasattr(self, target_loss_attr) or getattr(self, target_loss_attr) is None:
                print(f"[ERROR] MAML target loss '{target_loss_attr}' not found or is None.")
                return None

            original_head = getattr(self, target_head_attr)
            loss_fn = getattr(self, target_loss_attr)

            for task_data in batch_of_tasks:
                support_indices, support_labels = task_data['support']
                query_indices, query_labels = task_data['query']

                support_labels = support_labels.to(self.device)
                query_labels = query_labels.to(self.device)

                if support_indices.numel() == 0 or query_indices.numel() == 0:
                    print(f"[WARN] Skipping task for user {task_data.get('user_id', 'Unknown')} due to empty support/query indices.")
                    continue

                # --- Inner Loop Adaptation ---
                learner = l2l.clone_module(original_head)
                inner_optimizer = torch.optim.SGD(learner.parameters(), lr=self.hparams.inner_lr)

                for _ in range(self.hparams.adaptation_steps):
                    graph_batch_supp, seq_batch_supp, text_batch_supp, user_ids_supp = self._get_features_for_indices(support_indices)
                    with torch.no_grad():
                        support_fused = self.get_fused_features(
                            graph_batch=graph_batch_supp, sequence_batch=seq_batch_supp,
                            text_batch=text_batch_supp, user_ids=user_ids_supp,
                            batch_size=len(support_indices)
                        )
                    if support_fused is None: print(f"[WARN] Inner Loop: Could not get fused features for support set. Skipping adapt step."); break
                    support_preds = learner(support_fused)
                    inner_loss = loss_fn(support_preds, support_labels)
                    inner_optimizer.zero_grad()
                    inner_loss.backward()
                    inner_optimizer.step()

                # --- Outer Loop Evaluation ---
                graph_batch_qry, seq_batch_qry, text_batch_qry, user_ids_qry = self._get_features_for_indices(query_indices)
                with torch.no_grad():
                    query_fused = self.get_fused_features(
                        graph_batch=graph_batch_qry, sequence_batch=seq_batch_qry,
                        text_batch=text_batch_qry, user_ids=user_ids_qry,
                        batch_size=len(query_indices)
                    )
                    if query_fused is None: print(f"[WARN] Outer Loop: Could not get fused features for query set. Skipping task."); continue
                    query_preds = learner(query_fused)
                    outer_loss = loss_fn(query_preds, query_labels)

                # --- Accumulate Results --- 
                if not torch.isnan(outer_loss).any() and not torch.isinf(outer_loss).any():
                     total_outer_loss += outer_loss
                     avg_query_acc += self._calculate_accuracy(query_preds, query_labels)
                     tasks_processed += 1
                else:
                     print(f"[WARN] Outer loss is NaN/Inf for user {task_data.get('user_id', 'Unknown')}. Skipping task.")

            # --- Meta-Update --- 
            if tasks_processed > 0:
                avg_outer_loss = total_outer_loss / tasks_processed
                avg_query_acc /= tasks_processed
                self.manual_backward(avg_outer_loss) # Calculate gradients for original_head
                meta_optimizer.step() # Update original_head
                self.log(f'train/meta_outer_loss', avg_outer_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=tasks_processed, sync_dist=True)
                self.log(f'train/meta_query_acc', avg_query_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=tasks_processed, sync_dist=True)
                return avg_outer_loss
            else:
                print("[WARN] No tasks processed in meta-batch. Skipping meta-update.")
                return None # No loss to return
        else:
            # Call renamed standard step method
            return self._standard_step(batch, batch_idx, stage='train')

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        stage = 'val'
        # <<< Use reliable runtime flag for dispatch >>>
        if self._use_maml_runtime_flag:
             # Similar batch restructuring logic as training_step if needed
             if isinstance(batch, dict) and 'support' in batch and 'query' in batch:
                  try:
                      num_tasks = len(batch['user_id'])
                      batch_of_tasks = []
                      for i in range(num_tasks):
                           # Simplified reconstruction (adapt based on actual collate behavior)
                           support_indices = batch['support'][0][i]
                           support_labels = batch['support'][1][i]
                           query_indices = batch['query'][0][i]
                           query_labels = batch['query'][1][i]
                           user_id = batch['user_id'][i]
                           batch_of_tasks.append({
                               'support': (support_indices, support_labels),
                               'query': (query_indices, query_labels),
                               'user_id': user_id
                           })
                      self._meta_eval_step(batch_of_tasks, batch_idx, stage=stage)
                  except Exception as e:
                       print(f"[ERROR] Failed to reconstruct MAML batch in validation_step: {e}")
             elif isinstance(batch, list):
                  self._meta_eval_step(batch, batch_idx, stage=stage)
             else: print(f"[ERROR] {stage}_step: Unexpected batch type for MAML: {type(batch)}")
        else:
            # Call renamed standard step method
            self._standard_step(batch, batch_idx, stage=stage)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        stage = 'test'
        # <<< Use reliable runtime flag for dispatch >>>
        if self._use_maml_runtime_flag:
             # Similar batch restructuring logic as training_step if needed
             if isinstance(batch, dict) and 'support' in batch and 'query' in batch:
                  try:
                     num_tasks = len(batch['user_id'])
                     batch_of_tasks = []
                     for i in range(num_tasks):
                          # Simplified reconstruction (adapt based on actual collate behavior)
                          support_indices = batch['support'][0][i]
                          support_labels = batch['support'][1][i]
                          query_indices = batch['query'][0][i]
                          query_labels = batch['query'][1][i]
                          user_id = batch['user_id'][i]
                          batch_of_tasks.append({
                              'support': (support_indices, support_labels),
                              'query': (query_indices, query_labels),
                              'user_id': user_id
                          })
                     self._meta_eval_step(batch_of_tasks, batch_idx, stage=stage)
                  except Exception as e:
                      print(f"[ERROR] Failed to reconstruct MAML batch in test_step: {e}")
             elif isinstance(batch, list):
                  self._meta_eval_step(batch, batch_idx, stage=stage)
             else: print(f"[ERROR] {stage}_step: Unexpected batch type for MAML: {type(batch)}")
        else:
            # Call renamed standard step method
            self._standard_step(batch, batch_idx, stage=stage)

    def configure_optimizers(self):
        # TODO: Implement differential LR (e.g., lower LR for text encoder) if needed
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), # Only optimize parameters that require grad
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        print("[INFO] configure_optimizers: Returning simple AdamW optimizer.")
        # Example Scheduler (Optional):
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/total_loss"}}
        return optimizer
        # Add scheduler later if needed 