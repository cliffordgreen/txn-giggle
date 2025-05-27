import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

# Import components from other files in the 'models' directory
from .hgt_encoder import HGT
# from .tft_encoder import PytorchForecastingTFTWrapper
from .finbert_encoder import FinBERTEmbedder
from .fusion_modules import AttentionFusion # Assuming AttentionFusion is desired
from .losses import FocalLoss

# Assume HeteroData comes from torch_geometric
from torch_geometric.data import Batch # HeteroData Batch object

class AdvancedTransactionCategorizationModel(pl.LightningModule):
    """ Multi-modal transaction categorization using HGT, TFT, FinBERT, 
        User Embeddings, Attention Fusion, Focal Loss, and MTL.
    """
    def __init__(self, 
                 # Config Dictionaries
                 model_config: Dict[str, Any], # Contains keys like graph_encoder_params etc.
                 # Feature dimensions and counts (from DataModule after setup)
                 node_feature_dims: Dict[str, int],
                 edge_feature_dims: Dict[Tuple[str,str,str], int], # For GNN metadata
                 num_users: int,
                 num_global_classes: int,
                 graph_metadata: Tuple[List[str], List[Tuple[str,str,str]]], # Moved up
                 # User COA tokenized tensors (from DataModule)
                 user_tokenized_coa_tensors: Optional[Dict[str, torch.Tensor]] = None,
                 # Training Config
                 learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-5,
                 mtl_weights: Dict[str, float] = {'global': 1.0}, # User MTL removed for now
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 # Add DataFrame reference
                 transactions_df_ref: pd.DataFrame = None 
                 ):
        super().__init__()
        # Store simple hyperparameters automatically
        self.save_hyperparameters('learning_rate', 'weight_decay',
                                'mtl_weights', 'focal_loss_alpha', 'focal_loss_gamma',
                                'node_feature_dims', 'edge_feature_dims',
                                'num_users', 'num_global_classes',
                                'graph_metadata' # Added graph_metadata
                                )
        # Store complex configs manually (needed if loading from checkpoint)
        # We access these via self._config_name during init
        self._model_config = model_config 
        self._graph_config = model_config['graph_encoder_params']
        # self._sequence_config = model_config.get('sequence_encoder_params', {}) # Removed
        self._text_config = model_config.get('text_encoder_params', {})
        self._fusion_config = model_config['fusion_params']
        self.graph_metadata_prop = graph_metadata # Store graph_metadata directly
        
        # Extract required counts/dims from the config for convenience
        self.node_feature_dims = node_feature_dims
        self.edge_feature_dims = edge_feature_dims # Used for GNN metadata
        # self.sequence_feature_dim = sequence_feature_dim
        self.num_users = num_users
        self.num_global_classes = num_global_classes

        # Store references (NOT saved as hparams)
        self._transactions_df_ref = transactions_df_ref
        # We still need the full graph for labels if not propagated by loader
        self._full_graph_data_ref = None # Will be set by train script

        # <<< Store modality flags from config >>>
        self.use_gnn_encoder = model_config.get('use_gnn_encoder', True)
        # self.use_sequence_encoder = model_config.get('use_sequence_encoder', True) # Removed
        self.use_text_encoder = model_config.get('use_text_encoder', True)
        self.use_coa_text_features = model_config.get('use_coa_text_features', True) # New flag from config

        # --- 1. Encoders ---
        self.graph_encoder = None
        if self.use_gnn_encoder:
             print(f"[DEBUG HGT INIT] metadata type: {type(self.graph_metadata_prop)}, value: {self.graph_metadata_prop}") # DEBUG PRINT
             self.graph_encoder = HGT(
                in_channels=self.node_feature_dims,
                hidden_channels=self._graph_config['hidden_channels'], 
                out_channels=self._graph_config['out_channels'], 
                metadata=self.graph_metadata_prop, # Use the stored graph_metadata tuple
                num_heads=self._graph_config['num_heads'], 
                num_layers=self._graph_config['num_layers']
                # Add dropout if HGT supports it
             )
             print(f"[INFO] HGT Encoder Initialized. Output Dim: {self._graph_config['out_channels']}")

        # <<< Conditionally initialize Sequence Encoder >>>
        self.sequence_encoder = None
        # seq_out_dim = 0 # Default if not used
        # if self.use_sequence_encoder:
            # Ensure sequence config is not empty if used
            # if not self._sequence_config:
                 # raise ValueError("Sequence encoder is enabled but 'sequence_encoder_params' is missing or empty in config.")
            # self.sequence_encoder = PytorchForecastingTFTWrapper(
                # input_dim=self.sequence_feature_dim, # New: pass input_dim per timestep
                # output_dim=self._sequence_config['output_dim'],
                # tft_params=self._sequence_config.get('tft_params', {}),
                # embedding_source_key=self._sequence_config.get('embedding_source_key', 'encoder_variables')
            # )
            # seq_out_dim = self.sequence_encoder.get_output_dim()
            # print(f"[INFO] Sequence (TFT) Encoder Initialized. Input dim per step: {self.sequence_feature_dim}, Output Dim: {seq_out_dim}")

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
             print(f"[INFO] Transaction FinBERT Encoder Initialized. Output Dim: {text_out_dim}")
        
        # <<< User Embedding (Always initialized if num_users > 0?) >>>
        # Check if num_users is valid
        if not isinstance(num_users, int) or num_users <= 0:
             raise ValueError(f"Invalid num_users ({num_users}). Must be a positive integer.")
        self.user_embedding = nn.Embedding(num_users, self._model_config['user_embed_dim'])
        print(f"[INFO] User Embedding Initialized. Output Dim: {self._model_config['user_embed_dim']}")
        
        # Effective dimension for user features going into fusion
        effective_user_dim_for_fusion = self._model_config['user_embed_dim']
        if self.use_coa_text_features and self.use_text_encoder and self.text_encoder:
            # COA text will be encoded by the main text_encoder, so its output dim is text_out_dim
            effective_user_dim_for_fusion += text_out_dim # Concatenation
            print(f"[INFO] COA text features will be used. Effective user dim for fusion: {effective_user_dim_for_fusion}")

        # Store user_tokenized_coa_tensors as buffers if provided (Option A)
        self.user_coa_input_ids_buffer = None
        self.user_coa_attention_mask_buffer = None
        if self.use_coa_text_features and user_tokenized_coa_tensors is not None:
            if 'input_ids' in user_tokenized_coa_tensors and 'attention_mask' in user_tokenized_coa_tensors:
                self.register_buffer('user_coa_input_ids_buffer', user_tokenized_coa_tensors['input_ids'])
                self.register_buffer('user_coa_attention_mask_buffer', user_tokenized_coa_tensors['attention_mask'])
                print(f"[INFO] Registered COA tokenized tensors as buffers. Shape e.g. input_ids: {self.user_coa_input_ids_buffer.shape}")
            else:
                print("[WARN] user_tokenized_coa_tensors provided but missing 'input_ids' or 'attention_mask'. COA features might not work.")
        elif self.use_coa_text_features:
            print("[WARN] COA text features enabled, but user_tokenized_coa_tensors not provided to model. COA features will be zeros.")

        # --- 2. Fusion Module --- 
        # <<< Adjust fusion input dims based on active encoders >>>
        fusion_input_dims = {}
        if self.use_gnn_encoder and self.graph_encoder:
             fusion_input_dims['graph'] = self._graph_config['out_channels']
        # if self.use_sequence_encoder and self.sequence_encoder:
             # fusion_input_dims['sequence'] = seq_out_dim
        if self.use_text_encoder and self.text_encoder: # For transaction text
             fusion_input_dims['text'] = text_out_dim
        
        # User modality uses the (potentially combined) effective user dimension
        fusion_input_dims['user'] = effective_user_dim_for_fusion
        
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
        
        # --- 4. Loss Function ---
        self.focal_loss_global = FocalLoss(
            alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma, 
            num_classes=num_global_classes # Pass num_classes for alpha tensor creation
        )
        
    def forward(self, data: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: # Return only global logits
        # data is a torch_geometric.data.Batch object
        
        # Determine the effective batch size (total number of transactions in the batch)
        num_transactions_in_batch = data['transaction'].num_nodes 
        if num_transactions_in_batch == 0:
            # Should not happen if dataloader filters empty graphs, but handle defensively
            print("[WARN] Forward pass received a batch with zero transactions.")
            # Return dummy outputs matching expected dimensions if possible
            dummy_logits = torch.empty((0, self.num_global_classes), device=self.device)
            return dummy_logits, None # No user logits

        embeddings_to_fuse = {}

        # --- 1. Graph Encoding (GNN) ---
        if self.use_gnn_encoder and self.graph_encoder:
            try:
                # x_dict and edge_index_dict are directly available from the Batch object
                # Ensure edge_attr_dict is passed if your GNN uses it and it's in the batch
                edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
                
                node_embeddings_dict = self.graph_encoder(data.x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
                if 'transaction' in node_embeddings_dict:
                    graph_embed = node_embeddings_dict['transaction']
                    if graph_embed.shape[0] == num_transactions_in_batch:
                        embeddings_to_fuse['graph'] = graph_embed
                    else:
                        print(f"[WARN] GNN output transaction nodes ({graph_embed.shape[0]}) mismatch with batch transaction nodes ({num_transactions_in_batch}). Padding/truncating or error needed.")
                        # Simple fix: use zeros if mismatch, or ensure GNN outputs correctly sized tensor for all nodes in batch.
                        # This usually indicates an issue with how GNN handles batching or node indexing.
                        # For now, let's assume HGT and other PyG GNNs correctly output for all nodes in the batch.
                        embeddings_to_fuse['graph'] = torch.zeros((num_transactions_in_batch, self.graph_encoder.out_channels), device=self.device)

                else: # GNN did not produce 'transaction' embeddings
                    print("[WARN] GNN output missing 'transaction' key. Using zeros for graph embeddings.")
                    embeddings_to_fuse['graph'] = torch.zeros((num_transactions_in_batch, self.graph_encoder.out_channels), device=self.device)
            except Exception as e:
                print(f"[ERROR] Graph encoding failed in forward: {e}")
                import traceback; traceback.print_exc();
                embeddings_to_fuse['graph'] = torch.zeros((num_transactions_in_batch, self.graph_encoder.out_channels), device=self.device)
        elif self.use_gnn_encoder: # GNN enabled but no encoder instance
             embeddings_to_fuse['graph'] = torch.zeros((num_transactions_in_batch, self._model_config.get('graph_encoder_params',{}).get('out_channels',64)), device=self.device)


        # --- 2. Text Encoding (Transaction Text) ---
        if self.use_text_encoder and self.text_encoder:
            if hasattr(data['transaction'], 'input_ids') and hasattr(data['transaction'], 'attention_mask'):
                tx_input_ids = data['transaction'].input_ids
                tx_attention_mask = data['transaction'].attention_mask
                # Expected shape: (num_transactions_in_batch, seq_len)
                if tx_input_ids.shape[0] == num_transactions_in_batch:
                    text_input_dict = {'input_ids': tx_input_ids, 'attention_mask': tx_attention_mask}
                    text_embed = self.text_encoder(text_input_dict)
                    embeddings_to_fuse['text'] = text_embed
                else:
                    print(f"[WARN] Transaction text input_ids shape ({tx_input_ids.shape[0]}) mismatch with batch transaction nodes ({num_transactions_in_batch}). Using zeros.")
                    embeddings_to_fuse['text'] = torch.zeros((num_transactions_in_batch, self.text_encoder.get_output_dim()), device=self.device)
            else:
                print("[WARN] Transaction text 'input_ids' or 'attention_mask' not found in batch. Using zeros for text embeddings.")
                embeddings_to_fuse['text'] = torch.zeros((num_transactions_in_batch, self.text_encoder.get_output_dim()), device=self.device)
        elif self.use_text_encoder: # Text enabled but no encoder
            embeddings_to_fuse['text'] = torch.zeros((num_transactions_in_batch, self._model_config.get('text_encoder_params',{}).get('projection_dim') or 768), device=self.device)


        # --- 4. User Embedding & COA Text ---
        if self.num_users > 0 and hasattr(data['transaction'], 'user_id_code'):
            user_ids_for_transactions = data['transaction'].user_id_code # Global user IDs for each transaction
            
            # Standard user embedding
            user_embed = self.user_embedding(user_ids_for_transactions) # Shape: (num_tx_batch, user_embed_dim)
            
            combined_user_features = user_embed

            if self.use_coa_text_features and self.text_encoder and \
               self.user_coa_input_ids_buffer is not None and \
               self.user_coa_attention_mask_buffer is not None:
                try:
                    # Gather pre-tokenized COA for users in this batch
                    # user_ids_for_transactions might have duplicates, which is fine.
                    coa_input_ids_batch = self.user_coa_input_ids_buffer[user_ids_for_transactions]
                    coa_attention_mask_batch = self.user_coa_attention_mask_buffer[user_ids_for_transactions]
                    
                    coa_text_input_dict = {'input_ids': coa_input_ids_batch, 'attention_mask': coa_attention_mask_batch}
                    coa_embed = self.text_encoder(coa_text_input_dict) # Shape: (num_tx_batch, coa_text_out_dim)
                    
                    combined_user_features = torch.cat([user_embed, coa_embed], dim=-1)
                except IndexError as e_coa_idx:
                    print(f"[ERROR] COA text gathering failed due to IndexError (likely user_id out of bounds for COA buffers): {e_coa_idx}. User IDs: {user_ids_for_transactions.min()}-{user_ids_for_transactions.max()}, Buffer size: {self.user_coa_input_ids_buffer.shape[0]}")
                    # Fallback: use only standard user_embed, and pad COA part with zeros
                    zeros_for_coa = torch.zeros((num_transactions_in_batch, self.text_encoder.get_output_dim()), device=self.device)
                    combined_user_features = torch.cat([user_embed, zeros_for_coa], dim=-1)
                except Exception as e_coa:
                    print(f"[ERROR] COA text encoding failed: {e_coa}")
                    import traceback; traceback.print_exc();
                    # Fallback: use only standard user_embed, and pad COA part with zeros
                    zeros_for_coa = torch.zeros((num_transactions_in_batch, self.text_encoder.get_output_dim()), device=self.device)
                    combined_user_features = torch.cat([user_embed, zeros_for_coa], dim=-1)
            elif self.use_coa_text_features: # COA enabled but encoder or buffers missing
                # Pad with zeros for the COA part if it was intended to be used
                print("[WARN] COA features enabled but encoder or tokenized buffers missing. Using zeros for COA part of user features.")
                coa_dim = self._model_config.get('text_encoder_params',{}).get('projection_dim') or self.text_encoder.get_output_dim() if self.text_encoder else 768
                zeros_for_coa = torch.zeros((num_transactions_in_batch, coa_dim), device=self.device)
                combined_user_features = torch.cat([user_embed, zeros_for_coa], dim=-1)

            embeddings_to_fuse['user'] = combined_user_features
        elif self.num_users > 0 : # User embeddings enabled but user_id_code missing on transaction
            print("[WARN] User embeddings enabled but 'user_id_code' not found on transaction. Using zeros for user features.")
            effective_user_dim_for_fusion = self._model_config['user_embed_dim']
            if self.use_coa_text_features: 
                 coa_dim = self._model_config.get('text_encoder_params',{}).get('projection_dim') or self.text_encoder.get_output_dim() if self.text_encoder else 768
                 effective_user_dim_for_fusion += coa_dim
            embeddings_to_fuse['user'] = torch.zeros((num_transactions_in_batch, effective_user_dim_for_fusion), device=self.device)


        # --- 5. Fusion ---
        if not embeddings_to_fuse:
            # This case should ideally be prevented by checks in __init__ or if num_transactions_in_batch is 0
            print("[ERROR] No embeddings available for fusion. Returning zeros.")
            global_logits = torch.zeros((num_transactions_in_batch, self.num_global_classes), device=self.device)
            return global_logits, None

        fused_embeddings = self.fusion_module(embeddings_to_fuse) # Shape: (num_tx_batch, fusion_output_dim)

        # --- 6. Classification ---
        global_logits = self.global_head(fused_embeddings)
        
        # User-specific head is removed for now
        user_logits = None 
        
        return global_logits, user_logits # Return only global logits

    def _calculate_mtl_loss(self, 
                              global_logits: torch.Tensor, global_target: torch.Tensor, 
                              user_logits: Optional[torch.Tensor], user_target: Optional[torch.Tensor] # user_target can be None
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # total_loss, global_loss, user_loss (0 if not used)
        
        global_loss = self.focal_loss_global(global_logits, global_target)
        
        user_loss = torch.tensor(0.0, device=self.device) # Default to 0
        # User-specific MTL part removed as user_specific_head is removed.
        # If re-added, this logic would be:
        # if self.user_specific_head is not None and user_logits is not None and user_target is not None and \
        #    self.hparams.mtl_weights.get('user', 0.0) > 0:
        #     user_loss = self.focal_loss_user(user_logits, user_target)
        # else:
        #     user_loss = torch.tensor(0.0, device=self.device)
            
        total_loss = self.hparams.mtl_weights.get('global', 1.0) * global_loss # + \
                     # self.hparams.mtl_weights.get('user', 0.0) * user_loss # User part removed
        
        return total_loss, global_loss, user_loss

    def _calculate_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0 or targets.numel() == 0 or logits.shape[0] != targets.shape[0]:
            return torch.tensor(0.0, device=self.device) # Or handle as appropriate
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        return acc

    def _step(self, batch: Batch, batch_idx: int, stage: str) -> Optional[torch.Tensor]:
        # batch is now a torch_geometric.data.Batch object
        
        # Targets are on the transaction nodes in the batch
        global_target = batch['transaction'].y_global 
        # User-specific target removed for now
        # user_target = batch['transaction'].y_user if hasattr(batch['transaction'], 'y_user') else None

        if global_target is None or global_target.numel() == 0:
            print(f"[WARN] {stage}_step: Global target is None or empty. Skipping batch {batch_idx}.")
            return None

        # Forward pass now only takes the Batch object
        global_logits, user_logits = self(batch) # user_logits will be None

        if global_logits.shape[0] != global_target.shape[0]:
            print(f"[ERROR] {stage}_step: Logits batch size ({global_logits.shape[0]}) != Target batch size ({global_target.shape[0]}). Skipping batch {batch_idx}.")
            return None # Or raise error

        total_loss, global_loss_val, user_loss_val = self._calculate_mtl_loss(
            global_logits, global_target, 
            user_logits, None # Pass None for user_target
        )
        
        # Log losses
        self.log(f'{stage}/total_loss', total_loss, batch_size=batch['transaction'].num_nodes, prog_bar=True)
        self.log(f'{stage}/global_loss', global_loss_val, batch_size=batch['transaction'].num_nodes)
        # self.log(f'{stage}/user_loss', user_loss_val, batch_size=batch.num_graphs) # If user MTL re-added

        # Calculate and log accuracies
        global_acc = self._calculate_accuracy(global_logits, global_target)
        self.log(f'{stage}/global_acc', global_acc, batch_size=batch['transaction'].num_nodes, prog_bar=True)
        
        # if self.user_specific_head and user_logits is not None and user_target is not None:
        #     user_acc = self._calculate_accuracy(user_logits, user_target)
        #     self.log(f'{stage}/user_acc', user_acc, batch_size=batch.num_graphs)
            
        effective_batch_size = global_logits.shape[0] 
        if effective_batch_size == 0: effective_batch_size = 1 # Avoid division by zero if somehow all were filtered

        if stage == 'val':
            print(f"[DEBUG VAL_STEP] Calculated val_loss: {total_loss.item()}, global_loss: {global_loss_val.item()}")

        self.log(f'{stage}_loss', total_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, batch_size=effective_batch_size)

        return total_loss if stage == 'train' else None

    def training_step(self, batch: Batch, batch_idx: int) -> Optional[torch.Tensor]:
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        print("[DEBUG VAL_STEP] Entered validation_step") # DEBUG PRINT
        self._step(batch, batch_idx, 'val')

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        self._step(batch, batch_idx, 'test')

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        global_logits, _ = self(batch) # user_logits is None
        
        # Get original indices if available on transaction nodes for mapping predictions back
        original_indices = batch['transaction'].original_index if hasattr(batch['transaction'], 'original_index') else None
        user_ids = batch['transaction'].user_id_code if hasattr(batch['transaction'], 'user_id_code') else None

        predictions = {'global_logits': global_logits}
        if original_indices is not None:
            predictions['original_indices'] = original_indices
        if user_ids is not None:
            predictions['user_ids'] = user_ids
            
        return predictions

    def configure_optimizers(self):
        # TODO: Add support for differential learning rates (e.g., for FinBERT)
        params_to_optimize = []
        if self.use_text_encoder and self.text_encoder and self.text_encoder.finetune:
            # Separate FinBERT params if different LR is desired
            finbert_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
            other_params = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("text_encoder.")]
            # If coa_finbert_encoder is different and finetuned, add its params too
            if self.use_coa_text_features and self.text_encoder and \
               self.text_encoder is not self.coa_finbert_encoder and self.text_encoder.finetune:
                finbert_params.extend([p for p in self.coa_finbert_encoder.parameters() if p.requires_grad])
                # Ensure other_params does not include coa_finbert_encoder params
                other_params = [p for n, p in self.named_parameters() if p.requires_grad and \
                                not n.startswith("text_encoder.") and not n.startswith("coa_finbert_encoder.")]

            # Example: Different LRs (adjust as needed)
            # For now, common optimizer for all.
            # optimizer_grouped_parameters = [
            #     {'params': finbert_params, 'lr': self.hparams.learning_rate * 0.1}, # Smaller LR for FinBERT
            #     {'params': other_params, 'lr': self.hparams.learning_rate}
            # ]
            # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=self.hparams.weight_decay)
            params_to_optimize = self.parameters() # Default: all params together
        else:
            params_to_optimize = self.parameters()

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        
        # Learning rate scheduler (optional)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/total_loss"}
        return optimizer

    # Properties to expose feature dimensions for external use (e.g., logging, analysis)
    @property
    def gnn_output_dim(self) -> int:
        return self.graph_encoder.out_channels if self.use_gnn_encoder and self.graph_encoder else 0

    @property
    def text_output_dim(self) -> int: # For transaction text
        return self.text_encoder.get_output_dim() if self.use_text_encoder and self.text_encoder else 0
    
    @property
    def coa_text_output_dim(self) -> int:
        return self.coa_finbert_encoder.get_output_dim() if self.use_coa_text_features and self.coa_finbert_encoder else 0

    @property
    def user_embedding_dim(self) -> int:
        return self.user_embedding.embedding_dim if self.user_embedding else 0

    @property
    def fusion_output_dim(self) -> int:
        return self.fusion_module.output_dim if self.fusion_module else 0


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
        if not isinstance(batch, Batch):
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