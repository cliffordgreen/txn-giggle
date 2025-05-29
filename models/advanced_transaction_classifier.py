import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

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
    """
    def __init__(self, 
                 # Config Dictionaries
                 model_config: Dict[str, Any], # Contains keys like graph_encoder_params etc.
                 # Training Config
                 learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-5,
                 mtl_weights: Dict[str, float] = {'global': 1.0, 'user': 0.0},
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 # Add DataFrame reference
                 transactions_df_ref: pd.DataFrame = None 
                 ):
        super().__init__()
        # Store simple hyperparameters automatically
        self.save_hyperparameters('learning_rate', 'weight_decay',
                                'mtl_weights', 'focal_loss_alpha', 'focal_loss_gamma')
        # Store complex configs manually (needed if loading from checkpoint)
        # We access these via self._config_name during init
        self._model_config = model_config 
        self._graph_config = model_config['graph_encoder_params']
        self._sequence_config = model_config.get('sequence_encoder_params', {})
        self._text_config = model_config.get('text_encoder_params', {})
        self._fusion_config = model_config['fusion_params']
        
        # Extract required counts/dims from the config for convenience
        num_global_classes = model_config['num_global_classes']
        num_user_classes = model_config.get('num_user_classes', 0) # Default to 0 if missing
        num_users = model_config['num_users']
        user_embed_dim = model_config['user_embed_dim']

        # Store references (NOT saved as hparams)
        self._transactions_df_ref = transactions_df_ref
        # We still need the full graph for labels if not propagated by loader
        self._full_graph_data_ref = None # Will be set by train script

        # <<< Store modality flags from config >>>
        self.use_gnn_encoder = model_config.get('use_gnn_encoder', True)
        self.use_sequence_encoder = model_config.get('use_sequence_encoder', True)
        self.use_text_encoder = model_config.get('use_text_encoder', True)

        # --- 1. Encoders ---
        self.graph_encoder = None
        if self.use_gnn_encoder:
             self.graph_encoder = HGT(
                in_channels=self._graph_config['in_channels'],
                hidden_channels=self._graph_config['hidden_channels'], 
                out_channels=self._graph_config['out_channels'], 
                metadata=self._graph_config['metadata'], 
                num_heads=self._graph_config['num_heads'], 
                num_layers=self._graph_config['num_layers']
                # Add dropout if HGT supports it
             )
             print(f"[INFO] HGT Encoder Initialized. Output Dim: {self._graph_config['out_channels']}")

        # <<< Conditionally initialize Sequence Encoder >>>
        self.sequence_encoder = None
        seq_out_dim = 0 # Default if not used
        if self.use_sequence_encoder:
            # Ensure sequence config is not empty if used
            if not self._sequence_config:
                 raise ValueError("Sequence encoder is enabled but 'sequence_encoder_params' is missing or empty in config.")
            self.sequence_encoder = PytorchForecastingTFTWrapper(
                output_dim=self._sequence_config['output_dim'],
                tft_params=self._sequence_config.get('tft_params', {}),
                embedding_source_key=self._sequence_config.get('embedding_source_key', 'encoder_variables')
            )
            seq_out_dim = self.sequence_encoder.get_output_dim()
            print(f"[INFO] TFT Wrapper Initialized. Output Dim: {seq_out_dim}")

        # <<< Conditionally initialize Text Encoder >>>
        self.text_encoder = None
        text_out_dim = 0 # Default if not used
        if self.use_text_encoder:
             # Ensure text config is not empty if used
             if not self._text_config:
                  raise ValueError("Text encoder is enabled but 'text_encoder_params' is missing or empty in config.")
             self.text_encoder = FinBERTEmbedder(
                model_name=self._text_config.get('model_name', 'ProsusAI/finbert'),
                pooling_strategy=self._text_config.get('pooling_strategy', 'mean'),
                finetune=self._text_config.get('finetune', True),
                projection_dim=self._text_config.get('projection_dim', 0)
             )
             text_out_dim = self.text_encoder.get_output_dim()
             print(f"[INFO] FinBERT Encoder Initialized. Output Dim: {text_out_dim}")
        
        # <<< User Embedding (Always initialized if num_users > 0?) >>>
        # Check if num_users is valid
        if not isinstance(num_users, int) or num_users <= 0:
             raise ValueError(f"Invalid num_users ({num_users}). Must be a positive integer.")
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        print(f"[INFO] User Embedding Initialized. Output Dim: {user_embed_dim}")

        # --- 2. Fusion Module --- 
        # <<< Adjust fusion input dims based on active encoders >>>
        fusion_input_dims = {}
        if self.use_gnn_encoder and self.graph_encoder:
             fusion_input_dims['graph'] = self._graph_config['out_channels']
        if self.use_sequence_encoder and self.sequence_encoder:
             fusion_input_dims['sequence'] = seq_out_dim
        if self.use_text_encoder and self.text_encoder:
             fusion_input_dims['text'] = text_out_dim
        # Always include user embedding (assuming it's always used)
        fusion_input_dims['user'] = user_embed_dim
        
        # Ensure at least one modality is active
        if not fusion_input_dims:
            raise ValueError("No encoders are enabled or user embedding dim is zero. At least one input must be available for fusion.")
            
        self.fusion_module = AttentionFusion(
            modality_dims=fusion_input_dims,
            hidden_dim=self._fusion_config['hidden_dim'], 
            output_dim=self._fusion_config['output_dim'], 
            dropout=self._fusion_config.get('dropout', 0.1)
        )
        fused_dim = self._fusion_config['output_dim']
        print(f"[INFO] Attention Fusion Initialized. Output Dim: {fused_dim}")

        # --- 3. Classification Heads ---
        # Global head should always exist if num_global_classes is valid
        if not isinstance(num_global_classes, int) or num_global_classes <= 0:
            raise ValueError(f"Invalid num_global_classes ({num_global_classes}). Must be a positive integer.")
        self.global_head = nn.Linear(fused_dim, num_global_classes)
        
        # User-specific head only if num_user_classes > 0
        self.user_specific_head = None
        if isinstance(num_user_classes, int) and num_user_classes > 0:
            self.user_specific_head = nn.Linear(fused_dim, num_user_classes)
            print(f"[INFO] Classifiers Initialized: Global={num_global_classes}, User={num_user_classes}")
        else:
             print(f"[INFO] Classifiers Initialized: Global={num_global_classes}, User=DISABLED")

        # --- 4. Loss Function ---
        self.focal_loss_global = FocalLoss(
            alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma, 
            num_classes=num_global_classes # Pass num_classes for alpha tensor creation
        )
        
        # User-specific loss only if num_user_classes > 0
        self.focal_loss_user = None
        if isinstance(num_user_classes, int) and num_user_classes > 0:
            self.focal_loss_user = FocalLoss(
                alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma, 
                num_classes=num_user_classes # Pass num_classes for alpha tensor creation
            )
            print("[INFO] Focal Loss Initialized: Global, User")
        else:
             print("[INFO] Focal Loss Initialized: Global Only")

    def forward(self, 
                graph_batch: Optional[HeteroData] = None, 
                sequence_batch: Optional[Any] = None, 
                text_batch: Optional[List[str]] = None, 
                user_ids: Optional[torch.Tensor] = None,
                batch_size: Optional[int] = None
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        embeddings_to_fuse = {}

        # --- 1. Graph Encoding --- 
        if self.graph_encoder and graph_batch:
            try:
                node_embeddings_dict = self.graph_encoder(graph_batch.x_dict, graph_batch.edge_index_dict)
                
                # --- Extract embeddings for TARGET transaction nodes --- 
                # Slice the first batch_size nodes from the GNN output
                if 'transaction' in node_embeddings_dict:
                     if batch_size is None:
                          # Try to infer batch_size if not passed (e.g., from user_ids)
                          if user_ids is not None: batch_size = user_ids.shape[0]
                          else: raise ValueError("Forward pass needs batch_size if graph_batch is provided.")
                     
                     # Ensure we don't slice beyond available nodes
                     num_nodes_in_batch_output = node_embeddings_dict['transaction'].shape[0]
                     if batch_size > num_nodes_in_batch_output:
                         print(f"[WARN] Forward: batch_size ({batch_size}) > GNN output nodes ({num_nodes_in_batch_output}). Slicing available nodes.")
                         graph_embed = node_embeddings_dict['transaction'][:num_nodes_in_batch_output]
                     else:
                         graph_embed = node_embeddings_dict['transaction'][:batch_size]
                         
                     if graph_embed is not None:
                         embeddings_to_fuse['graph'] = graph_embed.to(self.device)
                else:
                    print("[WARN] Forward: HGT output missing 'transaction' embeddings.")
            except Exception as e:
                print(f"[ERROR] HGT Encoder forward failed: {e}")
                # Decide if we should raise or continue without graph embeddings

        # --- 2. Sequence Encoding --- 
        if self.sequence_encoder and sequence_batch is not None:
            try:
                seq_embed = self.sequence_encoder(sequence_batch, device=self.device)
                if seq_embed is not None:
                    embeddings_to_fuse['sequence'] = seq_embed.to(self.device)
            except Exception as e:
                print(f"[ERROR] TFT Encoder forward failed: {e}")
                
        # --- 3. Text Encoding --- 
        if self.text_encoder and text_batch is not None:
            try:
                text_embed = self.text_encoder(text_batch)
                if text_embed is not None:
                    embeddings_to_fuse['text'] = text_embed.to(self.device)
            except Exception as e:
                print(f"[ERROR] FinBERT Encoder forward failed: {e}")
                
        # --- 4. User Embedding --- 
        if self.user_embedding and user_ids is not None:
            try:
                user_ids_device = user_ids.to(self.device)
                
                # Validate user IDs are within bounds
                max_user_id = self.user_embedding.num_embeddings - 1
                min_user_id = user_ids_device.min().item()
                max_user_id_batch = user_ids_device.max().item()
                
                if min_user_id < 0 or max_user_id_batch > max_user_id:
                    raise ValueError(f"User ID out of bounds: batch range [{min_user_id}, {max_user_id_batch}] "
                                   f"exceeds embedding range [0, {max_user_id}]. "
                                   f"This indicates a data preprocessing error in user ID mapping.")
                
                user_embed = self.user_embedding(user_ids_device)
                if user_embed is not None:
                    embeddings_to_fuse['user'] = user_embed.to(self.device)
            except Exception as e:
                print(f"[ERROR] User Embedding forward failed: {e}")
                if user_ids is not None:
                    print(f"[DEBUG] user_ids shape: {user_ids.shape}")
                    print(f"[DEBUG] user_ids range: [{user_ids.min().item()}, {user_ids.max().item()}]")
                print(f"[DEBUG] embedding num_embeddings: {self.user_embedding.num_embeddings}")
                # Don't add to embeddings_to_fuse if failed
                
        # Check shapes and device consistency before fusion
        ref_batch_size = None
        if not embeddings_to_fuse:
            print("[ERROR] Forward: No embeddings available for fusion.")
            return None, None
        else:
            # Check batch sizes and device
            for name, emb in embeddings_to_fuse.items():
                 if ref_batch_size is None: ref_batch_size = emb.shape[0]
                 if emb.shape[0] != ref_batch_size:
                      print(f"[ERROR] Forward: Mismatched batch size for {name}: {emb.shape[0]} vs {ref_batch_size}")
                      return None, None
                 if emb.device != self.device:
                      print(f"[ERROR] Forward: Embedding {name} is on wrong device: {emb.device} vs {self.device}")
                      return None, None

        # --- 5. Fusion --- 
        try:
            fused_representation, _ = self.fusion_module(embeddings_to_fuse)
        except Exception as e:
             print(f"[ERROR] Fusion module forward failed: {e}")
             # Return Nones if fusion fails
             return None, None

        # --- 6. Classify --- 
        global_logits = None
        user_specific_logits = None
        try:
            if self.global_head:
                 global_logits = self.global_head(fused_representation)
            if self.user_specific_head:
                 user_specific_logits = self.user_specific_head(fused_representation)
        except Exception as e:
             print(f"[ERROR] Classifier head forward failed: {e}")
             # Return Nones if classification fails
             return None, None

        return global_logits, user_specific_logits 

    # Comment out the now unused helper method
    # def _get_labels_for_batch(self, original_indices: Optional[torch.Tensor], stage: str, batch_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     """Helper to look up labels using original indices from the full graph data."""
    #     labels_global, labels_user = None, None
    #     if original_indices is None or self._full_graph_data_ref is None:
    #          print(f"[WARN] {stage}_step (batch {batch_idx}): Cannot lookup labels, missing indices or full graph ref.")
    #          return None, None
    #     try:
    #         original_indices_cpu = original_indices.cpu().long()
    #         # Access the transaction store in the referenced full graph data
    #         tx_store = self._full_graph_data_ref['transaction'] 
    #         if hasattr(tx_store, 'y_global'):
    #             labels_global = tx_store.y_global[original_indices_cpu]
    #         if hasattr(tx_store, 'y_user'):
    #             labels_user = tx_store.y_user[original_indices_cpu]
    #     except Exception as e:
    #          print(f"[ERROR] {stage}_step (batch {batch_idx}): Error during label lookup: {e}")
    #          return None, None
    #     return labels_global, labels_user
        
    def _calculate_mtl_loss(self, 
                              global_logits: torch.Tensor, global_target: torch.Tensor, 
                              user_logits: Optional[torch.Tensor], user_target: Optional[torch.Tensor]
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         """Calculates individual and combined MTL loss."""
         loss_global = torch.tensor(0.0, device=self.device)
         if global_logits is not None and global_target is not None:
             try:
                 loss_global = self.focal_loss_global(global_logits, global_target.to(global_logits.device))
             except Exception as e:
                 print(f"[ERROR] Global loss calculation failed: {e}")
                 loss_global = torch.tensor(0.0, device=self.device, requires_grad=True) 
 
         loss_user = torch.tensor(0.0, device=self.device)
         # Check if user head and loss are initialized AND logits/targets are provided
         if self.user_specific_head is not None and self.focal_loss_user is not None and \
            user_logits is not None and user_target is not None:
             try:
                 loss_user = self.focal_loss_user(user_logits, user_target.to(user_logits.device))
             except Exception as e:
                 print(f"[ERROR] User loss calculation failed: {e}")
                 loss_user = torch.tensor(0.0, device=self.device, requires_grad=True)
                 
         weight_global = self.hparams.mtl_weights.get('global', 1.0) # Default to 1.0 if user loss is disabled
         weight_user = self.hparams.mtl_weights.get('user', 0.0) if self.user_specific_head else 0.0 # Use 0 weight if disabled
         total_loss = weight_global * loss_global + weight_user * loss_user
         
         # Handle potential NaN loss before returning
         if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
              print(f"[WARN] Calculated total_loss is NaN/Inf. Global={loss_global.item()}, User={loss_user.item()}")
              # Return a zero tensor that requires grad for training step
              total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
              # Set individual losses to 0 for logging consistency
              loss_global = torch.tensor(0.0, device=self.device)
              loss_user = torch.tensor(0.0, device=self.device)
              
         return total_loss, loss_global, loss_user
         
    def _calculate_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Helper to calculate accuracy."""
        # Ensure inputs are valid tensors on the same device
        if logits is None or targets is None or logits.shape[0] == 0 or targets.shape[0] == 0:
             return torch.tensor(0.0, device=self.device)
        if logits.shape[0] != targets.shape[0]:
             print(f"[WARN] Accuracy calc: Logits batch ({logits.shape[0]}) != Targets batch ({targets.shape[0]})")
             return torch.tensor(0.0, device=self.device)
             
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            targets_on_device = targets.to(logits.device).long()
            # Clamp targets AFTER moving to device and converting to long
            targets_clamped = torch.clamp(targets_on_device, 0, logits.shape[1]-1)
            correct = (preds == targets_clamped).float()
            accuracy = correct.mean()
        return accuracy

    def _step(self, batch: Any, batch_idx: int, stage: str) -> Optional[torch.Tensor]:
        """Common logic for train/val/test steps, assuming HGTLoader yields HeteroData."""
        # --- Unpack Batch & Extract Data (Using Slicing based on batch_size) --- 
        if not isinstance(batch, HeteroData):
            print(f"[ERROR] {stage}_step received unexpected batch type: {type(batch)}.")
            return None
        
        graph_batch = batch # Pass the full sampled graph to forward
        sequence_batch = None 
        text_batch = None
        user_ids = None
        global_target = None
        user_target = None
        batch_size = None
        
        try:
            if 'transaction' in graph_batch.node_types:
                tx_store = graph_batch['transaction']
                
                # Get batch_size (number of seed nodes)
                if hasattr(tx_store, 'batch_size'):
                     batch_size = tx_store.batch_size
                     if batch_size is None or batch_size == 0: 
                          print(f"[WARN] {stage}_step (batch {batch_idx}): tx_store.batch_size is None or 0. Cannot proceed.")
                          return None
                elif hasattr(tx_store, 'input_id'): # Fallback if batch_size missing
                     batch_size = tx_store.input_id.shape[0]
                     if batch_size == 0:
                          print(f"[WARN] {stage}_step (batch {batch_idx}): Determined batch_size is 0 from input_id. Skipping.")
                          return None
                     print(f"[INFO] {stage}_step (batch {batch_idx}): Used input_id length for batch_size: {batch_size}")
                else:
                     print(f"[ERROR] {stage}_step (batch {batch_idx}): Cannot determine batch_size.")
                     return None
                
                # --- Fetch data for the seed nodes by slicing first batch_size elements --- 
                
                # User ID 
                if self.user_embedding and hasattr(tx_store, 'user_id_code') and tx_store.user_id_code.shape[0] >= batch_size:
                    user_ids = tx_store.user_id_code[:batch_size]
                          
                # Text (use _raw_text)
                if self.use_text_encoder and hasattr(tx_store, '_raw_text'):
                    if len(tx_store._raw_text) >= batch_size:
                        text_batch = tx_store._raw_text[:batch_size]
                    else:
                         print(f"[WARN] {stage}_step (batch {batch_idx}): _raw_text length ({len(tx_store._raw_text)}) < batch_size ({batch_size}).")
                               
                # Sequence 
                if self.use_sequence_encoder and hasattr(tx_store, 'seq_features') and hasattr(tx_store, 'seq_lengths'):
                    if tx_store.seq_features.shape[0] >= batch_size:
                        sequence_batch = {
                            'sequences': tx_store.seq_features[:batch_size],
                            'lengths': tx_store.seq_lengths[:batch_size]
                        }
                        # Add categorical features if they exist
                        if hasattr(tx_store, 'seq_cat_features') and tx_store.seq_cat_features.shape[0] >= batch_size:
                             sequence_batch['seq_cat_features'] = tx_store.seq_cat_features[:batch_size]
                        else:
                             print(f"[WARN] {stage}_step (batch {batch_idx}): Missing or incorrectly sized seq_cat_features.")
                    else:
                         print(f"[WARN] {stage}_step (batch {batch_idx}): seq_features length ({tx_store.seq_features.shape[0]}) < batch_size ({batch_size}).")
                               
                # Labels 
                if hasattr(tx_store, 'y_global') and tx_store.y_global.shape[0] >= batch_size:
                    global_target = tx_store.y_global[:batch_size]
                else:
                     print(f"[WARN] {stage}_step (batch {batch_idx}): Cannot extract global_target via slicing.")
                     
                if hasattr(tx_store, 'y_user') and tx_store.y_user.shape[0] >= batch_size:
                    user_target = tx_store.y_user[:batch_size]
                # else: user_target might be optional
                          
            else:
                 print(f"[WARN] {stage}_step (batch {batch_idx}): 'transaction' node type not found in HGTLoader batch.")
                 return None 

        except Exception as e:
            print(f"[ERROR] Failed during batch data extraction in {stage}_step (batch {batch_idx}): {e}")
            import traceback
            traceback.print_exc()
            return None

        # --- Validation after extraction --- 
        if global_target is None: 
            print(f"[WARN] {stage}_step (batch {batch_idx}): global_target is None after extraction. Skipping batch.")
            return None 
        if self.user_embedding and user_ids is None:
             print(f"[WARN] {stage}_step (batch {batch_idx}): user_ids is None after extraction. Skipping batch.")
             return None
        # Add more checks as needed for text_batch, sequence_batch if they are critical

        # --- Forward pass --- 
        device = self.device
        graph_batch = graph_batch.to(device)
        user_ids = user_ids.to(device) if user_ids is not None else None
        if isinstance(sequence_batch, dict):
             sequence_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in sequence_batch.items()}
        elif isinstance(sequence_batch, torch.Tensor):
             sequence_batch = sequence_batch.to(device)
        
        # Pass determined batch_size to forward
        global_logits, user_specific_logits = self(graph_batch, sequence_batch, text_batch, user_ids, batch_size=batch_size)

        # --- Loss Calculation --- 
        if global_logits is None:
             print(f"[WARN] {stage}_step (batch {batch_idx}): global_logits is None after forward pass.")
             return None
             
        total_loss, loss_global, loss_user = self._calculate_mtl_loss(
            global_logits, global_target,
            user_specific_logits, user_target # Pass potentially None user logits/targets
        )

        # --- Accuracy Calculation --- 
        acc_global = self._calculate_accuracy(global_logits, global_target)
        acc_user = self._calculate_accuracy(user_specific_logits, user_target) # Handles None inputs

        # --- Logging & Return --- 
        log_batch_size = global_logits.shape[0]
        log_dict = {
            f'{stage}_loss': total_loss,
            f'{stage}_global_loss': loss_global,
            f'{stage}_user_loss': loss_user,
            f'{stage}_acc_global': acc_global,
            f'{stage}_acc_user': acc_user
        }
        self.log_dict(log_dict, on_step=(stage=='train'), on_epoch=True, prog_bar=(stage=='train'), batch_size=log_batch_size, sync_dist=True)

        return total_loss if stage == 'train' else None

    # --- Standard Lightning Hooks --- 
    def training_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._step(batch, batch_idx, stage='val') 

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._step(batch, batch_idx, stage='test')

    def configure_optimizers(self):
        # Placeholder: Simple optimizer for all params
        # TODO: Implement differential LR for text encoder if desired
        # TODO: Integrate MAML optimizer logic if used
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        print("[INFO] configure_optimizers: Returning simple AdamW optimizer.")
        return optimizer
        # Add scheduler later if needed 