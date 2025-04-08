# models/transaction_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# import torchmetrics # Import if you add other metrics later
from dataclasses import dataclass # Keep if ModelConfig used elsewhere
from torch_geometric.data import HeteroData
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.nn import HeteroConv, GATv2Conv # Assuming GATv2Conv is used
from torch.nn import Linear, LayerNorm # Make sure Linear/LayerNorm are imported
from typing import Dict, Optional, Tuple, Any, List, Union
import sys # Keep for potential future debug exits
import traceback # Keep for error printing

# Import local modules
from .gnn import HeteroGNNEncoder
from .sequence import SequenceEncoder
from .text import MultiFieldTextEncoder
from .fusion import MultiTaskFusion, AttentionFusion, GatingFusion

# Note: ModelConfig dataclass definition is kept below,
# but it doesn't seem directly used by TransactionClassifier __init__ args.
# Remove if not needed elsewhere in your project.
@dataclass
class ModelConfig:
     """Configuration for the transaction classifier model."""
     @dataclass
     class GNNConfig:
         input_dim: int; hidden_dim: int; output_dim: int; num_layers: int; dropout: float; edge_types: list
     @dataclass
     class SequentialConfig:
         input_dim: int; hidden_dim: int; num_layers: int; dropout: float
     @dataclass
     class TextConfig:
         model_name: str; max_length: int; hidden_dim: int; dropout: float; field_weights: Optional[Dict[str, float]] = None
     @dataclass
     class ClassifierConfig:
         hidden_dim: int; dropout: float

     num_classes: int
     gnn: GNNConfig; sequential: SequentialConfig; text: TextConfig; classifier: ClassifierConfig

class TransactionClassifier(pl.LightningModule):
     def __init__(
         self,
         num_global_classes: int,
         num_user_classes: int,
         # --- GNN Params ---
         gnn_node_input_dims: Dict[str, int],
         gnn_edge_input_dims: Dict[Tuple[str, str, str], int],
         gnn_hidden_channels: int = 256,
         gnn_out_channels: int = 256, # Output dim of GNN
         gnn_num_layers: int = 2,
         gnn_heads: int = 4, # Ensure hidden_channels % heads == 0
         gnn_metadata: Optional[Tuple[List[str], List[Tuple[str, str, str]]]] = None, # Pass metadata for refactored GNN
         gnn_dropout: float = 0.1, # <<< ADDED parameter 
         # --- Sequence Params ---
         seq_input_dim: int = 6, # Default to 6 now based on DataModule
         seq_hidden_size: int = 256,
         seq_num_layers: int = 2,
         # --- Text Params ---
         text_model_name: str = 'bert-base-uncased',
         text_max_length: int = 128,
         text_out_dim: int = 256, # Desired output dim after text encoder pooling/projection
         # --- Fusion Params ---
         fusion_type: str = 'attention', # Options: 'attention', 'gating', 'multi_task'
         fusion_hidden_dim: int = 256,
         fusion_dropout: float = 0.2,
         # --- Training Params ---
         learning_rate: float = 1e-4,
         weight_decay: float = 1e-5,
         scheduler_pct_start: float = 0.1,
         class_weights: Optional[List[float]] = None,
         # --- Modality Control ---
         use_sequence_encoder: bool = True,
         use_text_encoder: bool = True,
         use_gnn_encoder: bool = True,
         # --- Debug Flag ---
         gnn_only_test_mode: bool = False,
         # --- Reference to Full Data (for label lookup) ---
         full_graph_data_ref: Optional[HeteroData] = None
     ):
         super().__init__()
         # <<< Call save_hyperparameters FIRST, without ignoring module configs yet >>>
         # This makes args available via self.hparams for module initialization.
         # We will ignore/delete problematic ones before logging later.
         self.save_hyperparameters(ignore=['class_weights', 'full_graph_data_ref']) 

         # Store the reference to the full graph data (not logged)
         self._full_graph_data = full_graph_data_ref
         if self._full_graph_data is None:
             print("[WARN] TransactionClassifier initialized without full_graph_data_ref. Label lookup fallback might fail.")

         # Save hyperparameters for logging and access via self.hparams
         # If you need class_weights later from hparams, remove 'class_weights' from ignore list.
         self.save_hyperparameters(ignore=['class_weights'])

         # Store the GNN-only mode flag
         self.gnn_only_test_mode = self.hparams.gnn_only_test_mode
         if self.gnn_only_test_mode:
             print("[INFO] TransactionClassifier running in GNN-ONLY TEST MODE.")

         # --- Cleaned-up Loss Function Initialization ---
         self.criterion = None # Initialize placeholder
         weight_tensor = None
         if class_weights is not None:
             try:
                 # Use the function *argument* directly here
                 weight_tensor = torch.tensor(class_weights, dtype=torch.float)
                 # Register as buffer AFTER super().__init__() and BEFORE using it in criterion
                 # Make sure buffer name matches what's used in _common_step if accessed directly (not needed if self.criterion used)
                 self.register_buffer("criterion_weight_buffer", weight_tensor, persistent=False)
                 print(f"[INFO] Using criterion weights (first 5): {self.criterion_weight_buffer[:5]}...")
                 self.criterion = nn.CrossEntropyLoss(weight=self.criterion_weight_buffer)
             except Exception as e:
                 print(f"[WARN] Failed to process class_weights: {e}. Using unweighted loss.")
                 self.criterion = nn.CrossEntropyLoss() # Fallback to unweighted
         else:
             print("[INFO] No class_weights provided. Using unweighted loss.")
             self.criterion = nn.CrossEntropyLoss() # Default unweighted
         # --- End Cleaned-up Loss Function Initialization ---

         # --- Initialize Encoders Conditionally (using self.hparams) ---
         self.gnn_encoder = None
         if self.hparams.use_gnn_encoder:
             if self.hparams.gnn_metadata is None or len(self.hparams.gnn_metadata) != 2:
                 raise ValueError("gnn_metadata must be provided if use_gnn_encoder is True")
             gnn_edge_types = self.hparams.gnn_metadata[1] # Now works
             self.gnn_encoder = HeteroGNNEncoder(
                 edge_types=gnn_edge_types, 
                 in_channels=self.hparams.gnn_node_input_dims, # Now works
                 edge_input_dims=self.hparams.gnn_edge_input_dims, # Now works
                 hidden_channels=self.hparams.gnn_hidden_channels,
                 out_channels=self.hparams.gnn_out_channels,
                 num_layers=self.hparams.gnn_num_layers,
                 heads=self.hparams.gnn_heads,
                 dropout=self.hparams.gnn_dropout
             )
             print("[INFO] GNN Encoder Initialized.")
         else:
             print("[INFO] GNN Encoder Disabled.")

         self.seq_encoder = None
         if self.hparams.use_sequence_encoder:
             self.seq_encoder = SequenceEncoder(
                 input_dim=self.hparams.seq_input_dim,
                 hidden_dim=self.hparams.seq_hidden_size,
                 num_layers=self.hparams.seq_num_layers,
             )
             print("[INFO] Sequence Encoder Initialized.")
         else:
             print("[INFO] Sequence Encoder Disabled.")

         self.text_encoder = None
         if self.hparams.use_text_encoder:
             self.text_encoder = MultiFieldTextEncoder(
                 model_name=self.hparams.text_model_name,
                 max_length=self.hparams.text_max_length,
             )
             print("[INFO] Text Encoder Initialized.")
             text_hidden_size = self.hparams.text_out_dim # Assume this comes from text encoder
         else:
             print("[INFO] Text Encoder Disabled.")
             text_hidden_size = 0 # No text contribution
         
         # --- Initialize Fusion Module and Classifiers (using self.hparams) ---
         # Dynamically determine modality dimensions based on enabled encoders
         self.modality_dims = {}
         if self.gnn_encoder:
             self.modality_dims['graph'] = self.hparams.gnn_out_channels
         if self.seq_encoder:
             # Seq hidden size determines the output dim used for fusion
             self.modality_dims['sequence'] = self.hparams.seq_hidden_size 
         if self.text_encoder:
             self.modality_dims['text'] = text_hidden_size

         if not self.modality_dims:
             raise ValueError("At least one encoder (GNN, Sequence, or Text) must be enabled.")
         print(f"[INFO] Active Modality Dims for Fusion: {self.modality_dims}")

         self._fusion_type = self.hparams.fusion_type
         self.fusion_module = None
         self.global_classifier = None # Classifier used after attention/gating fusion
         self.user_classifier = None   # User classifier used after attention/gating fusion

         if self._fusion_type == 'attention' or self._fusion_type == 'gating':
             FusionClass = AttentionFusion if self._fusion_type == 'attention' else GatingFusion
             self.fusion_module = FusionClass(
                 input_dims=self.modality_dims,
                 hidden_dim=self.hparams.fusion_hidden_dim,
                 dropout=self.hparams.fusion_dropout
             )
             # Use correct class counts for classifiers
             self.global_classifier = Linear(self.hparams.fusion_hidden_dim, self.hparams.num_global_classes)
             self.user_classifier = Linear(self.hparams.fusion_hidden_dim, self.hparams.num_user_classes) 

         elif self._fusion_type == 'multi_task':
             self.fusion_module = MultiTaskFusion(
                 input_dims=self.modality_dims,
                 hidden_dim=self.hparams.fusion_hidden_dim,
                 # Pass correct class counts here too
                 num_global_classes=self.hparams.num_global_classes,
                 num_user_classes=self.hparams.num_user_classes,
                 dropout=self.hparams.fusion_dropout
             )
         else:
             raise ValueError(f"Unsupported fusion_type: {self._fusion_type}")

         # --- Define Separate Classifier for GNN-Only Mode ---
         self.gnn_direct_classifier = Linear(self.hparams.gnn_out_channels, self.hparams.num_global_classes)
         print(f"[INFO] Initialized gnn_direct_classifier: Linear({self.hparams.gnn_out_channels}, {self.hparams.num_global_classes})")
         
         # <<< REMOVE problematic hparams before automatic logging happens >>>
         # These complex types cause issues with OmegaConf/YAML saving.
         if 'gnn_node_input_dims' in self.hparams: del self.hparams['gnn_node_input_dims']
         if 'gnn_edge_input_dims' in self.hparams: del self.hparams['gnn_edge_input_dims']
         if 'gnn_metadata' in self.hparams: del self.hparams['gnn_metadata']
         # Note: class_weights and full_graph_data_ref were already ignored initially.

     # ----------------------------------------------------
     # Forward Pass (Handles both GNN-only and Full mode)
     # ----------------------------------------------------
     def forward(self, batch: HeteroData) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
         """
         Forward pass. Runs either GNN-only or full multi-modal path based on init flag.
         """
         global_logits, user_logits, fusion_weights = None, None, None

         # --- 1. GNN Processing (Common to both paths) ---
         try:
             # Prepare edge attributes (project if needed, based on your GNN encoder version)
             # Use a clearer variable name for the dict retrieved from the batch
             edge_attr_input_to_gnn = None # This will be passed to GNN
             # Check if the GNN encoder instance requires edge projections (has edge_projs)
             # --- Simplified Edge Attribute Handling (Example - Adjust based on your GNN) ---
             edge_attr_input_from_batch = getattr(batch, 'edge_attr_dict', None) # Get potential top-level dict
             if edge_attr_input_from_batch is None: # If not top-level, try gathering from edge stores
                 edge_attr_input_from_batch = {}
                 if hasattr(batch, 'metadata'): # Use metadata if available
                     node_types, edge_types_tuples = batch.metadata()
                     for edge_type in edge_types_tuples:
                         if hasattr(batch[edge_type], 'edge_attr'):
                             edge_attr_input_from_batch[edge_type] = batch[edge_type].edge_attr
                 if not edge_attr_input_from_batch: # If still empty
                     edge_attr_input_from_batch = None

             edge_attr_to_pass_to_gnn = edge_attr_input_from_batch # Default: pass what we got

             # --- START ADDITION: Debug Print Before GNN Call ---
             # print("\n--- DEBUG: Edge Attrs Dict Being Passed to GNN Encoder ---")
             # if edge_attr_to_pass_to_gnn is not None:
             #     for et, ea in edge_attr_to_pass_to_gnn.items():
             #         # Print edge type and the shape/dtype of the corresponding tensor
             #         print(f"  Edge Type {et}: Shape={ea.shape if hasattr(ea, 'shape') else 'N/A'}, Dtype={ea.dtype if hasattr(ea, 'dtype') else 'N/A'}")
             # else:
             #     # Print if no edge attributes are being passed
             #     print("  edge_attr_dict passed to GNN Encoder is: None")
             # print("--- END DEBUG ---")
             # --- END ADDITION ---

             # Run GNN Encoder, passing the attributes retrieved from the batch
             gnn_out_dict = self.gnn_encoder(
                 batch.x_dict,
                 batch.edge_index_dict,
                 edge_attr_dict=edge_attr_to_pass_to_gnn # Pass the dict obtained/prepared from batch
             )

             if 'transaction' not in gnn_out_dict or gnn_out_dict['transaction'] is None:
                 raise ValueError("GNN Encoder did not return 'transaction' embeddings.")

             # Assuming the first batch_size nodes are the target ones
             # Verify this assumption based on your DataModule!
             num_seed_nodes = batch['transaction'].batch_size
             if num_seed_nodes > gnn_out_dict['transaction'].shape[0]:
                  print(f"[WARN] Forward: num_seed_nodes ({num_seed_nodes}) > available GNN transaction nodes ({gnn_out_dict['transaction'].shape[0]}). Check batch structure.")
                  num_seed_nodes = gnn_out_dict['transaction'].shape[0] # Avoid index error

             gnn_features = gnn_out_dict['transaction'][:num_seed_nodes]
             # <<< DEBUG PRINT GNN >>>
             # print(f"DEBUG Forward: GNN Features Shape: {gnn_features.shape}, HasNaN: {torch.isnan(gnn_features).any().item()}, Min: {torch.min(gnn_features).item() if gnn_features.numel() > 0 else 'N/A':.4f}, Max: {torch.max(gnn_features).item() if gnn_features.numel() > 0 else 'N/A':.4f}")

         except Exception as e:
             print(f"\n!!! ERROR during GNN processing in forward pass: {e}")
             traceback.print_exc() # Print full traceback for GNN errors
             raise e

         # --- Conditional Path ---
         if self.gnn_only_test_mode:
             # --- Path 1: GNN Only + Direct Classifier ---
             print("--- Running SIMPLIFIED forward (GNN only) ---")
             try:
                 # Use the dedicated GNN classifier defined in __init__
                 if not hasattr(self, 'gnn_direct_classifier'):
                     raise AttributeError("self.gnn_direct_classifier not found. Ensure it's defined in __init__.")

                 global_logits = self.gnn_direct_classifier(gnn_features)
                 # <<< DEBUG PRINT GNN ONLY LOGITS >>>
                 # print(f"DEBUG Forward (GNN Only): Global Logits Shape: {global_logits.shape}, HasNaN: {torch.isnan(global_logits).any().item()}")
                 user_logits = None # No user classification in this mode
                 fusion_weights = None # No fusion happening
             except Exception as e_clf:
                 print(f"\n!!! ERROR during Direct GNN Classification: {e_clf}")
                 traceback.print_exc()
                 raise e_clf

         else:
             # --- Path 2: Full Multi-modal Path ---
             # print("--- Running FULL forward pass ---") # Optional debug print

             # 2. Sequence Processing
             seq_features = None
             if self.seq_encoder:
                 try:
                    # Ensure slicing uses the potentially adjusted num_seed_nodes
                    seq_input = batch['transaction'].seq_features[:num_seed_nodes]
                    seq_lens = getattr(batch['transaction'], 'seq_lengths', None)

                    if seq_lens is not None:
                        seq_lens = seq_lens[:num_seed_nodes] # Slice sequence lengths

                        # --- START: CORRECTED LOGIC ---
                        # Check ONLY for invalid NEGATIVE lengths
                        invalid_len_mask = seq_lens < 0
                        if invalid_len_mask.any():
                            num_invalid = invalid_len_mask.sum().item()
                            # Make sure the warning message is accurate!
                            print(f"[WARN] Forward: Found {num_invalid} sequences with length < 0. Clamping to 0.")
                            # Clone to avoid modifying original tensor from batch if necessary
                            seq_lens = seq_lens.clone()
                            seq_lens[invalid_len_mask] = 0 # Clamp negatives to ZERO

                        # Optional, but good practice: Clamp lengths that might exceed the actual dimension
                        # due to potential batching artifacts (though padding usually handles this)
                        if seq_input.nelement() > 0: # Check if seq_input is not empty
                             max_allowed_len = seq_input.shape[1]
                             too_long_mask = seq_lens > max_allowed_len
                             if too_long_mask.any():
                                 print(f"[WARN] Forward: Clamping {too_long_mask.sum().item()} seq_lens > {max_allowed_len}")
                                 if not seq_lens.is_contiguous(): seq_lens = seq_lens.contiguous() # Ensure contiguous for inplace op
                                 seq_lens[too_long_mask] = max_allowed_len
                        # --- END: CORRECTED LOGIC ---

                    # Pass the CORRECTED lengths (0 preserved, negatives are 0) to the encoder
                    # The SequenceEncoder should now handle length 0 correctly via pack_padded_sequence
                    _, _, seq_features = self.seq_encoder(seq_input, lengths=seq_lens)
                    # <<< DEBUG PRINT SEQUENCE >>>
                    # print(f"DEBUG Forward: Seq Features Shape: {seq_features.shape}, HasNaN: {torch.isnan(seq_features).any().item()}, Min: {torch.min(seq_features).item() if seq_features.numel() > 0 else 'N/A':.4f}, Max: {torch.max(seq_features).item() if seq_features.numel() > 0 else 'N/A':.4f}")

                 except Exception as e:
                    print(f"\n!!! ERROR during Sequence processing: {e}")
                    traceback.print_exc()
                    raise e
             else: # seq_encoder disabled
                  # Need a placeholder tensor of correct shape if sequence modality is expected by fusion
                  # Or ensure fusion module can handle missing modalities
                  pass # Assuming fusion module handles missing keys

             # 3. Text Processing
             text_features = None
             if self.text_encoder:
                 try:
                     text_features_dict = {}
                     # Use hparams for consistency if possible, otherwise hardcoded list
                     fields = getattr(self.hparams, 'text_fields', ['description', 'memo', 'merchant_name'])
                     for field in fields:
                         input_ids_key = f'{field}_input_ids'
                         attn_mask_key = f'{field}_attention_mask'
                         if hasattr(batch['transaction'], input_ids_key) and hasattr(batch['transaction'], attn_mask_key):
                             # Move tensors to the model's device within the dict comprehension
                             # Ensure slicing uses the potentially adjusted num_seed_nodes
                             input_ids = batch['transaction'][input_ids_key][:num_seed_nodes]
                             attn_mask = batch['transaction'][attn_mask_key][:num_seed_nodes]
                             text_features_dict[field] = {
                                 'input_ids': input_ids.to(self.device),
                                 'attention_mask': attn_mask.to(self.device)
                             }
                     if not text_features_dict: raise ValueError("No text features found/constructed.")
                     text_features = self.text_encoder(text_features_dict)
                     # <<< DEBUG PRINT TEXT >>>
                     # print(f"DEBUG Forward: Text Features Shape: {text_features.shape}, HasNaN: {torch.isnan(text_features).any().item()}, Min: {torch.min(text_features).item() if text_features.numel() > 0 else 'N/A':.4f}, Max: {torch.max(text_features).item() if text_features.numel() > 0 else 'N/A':.4f}")
                 except Exception as e:
                     print(f"\n!!! ERROR during Text processing: {e}")
                     traceback.print_exc()
                     raise e
             else: # text_encoder disabled
                  pass # Assuming fusion module handles missing keys

             # 4. Fusion
             # Build embeddings dict only with available features
             embeddings = {}
             if gnn_features is not None: # Should always exist if not GNN-only mode
                 embeddings['graph'] = gnn_features
             if seq_features is not None:
                 embeddings['sequence'] = seq_features
             if text_features is not None:
                 embeddings['text'] = text_features
             
             if not embeddings:
                  raise RuntimeError("No features available for fusion in forward pass.")

             try:
                 if self._fusion_type == 'multi_task':
                     # MultiTaskFusion likely returns logits directly
                     # Ensure it can handle missing keys in embeddings dict
                     global_logits, user_logits, fusion_weights = self.fusion_module(embeddings)
                 elif self._fusion_type in ['attention', 'gating']:
                     # Attention/Gating return fused embed + weights
                     # Ensure they can handle missing keys in embeddings dict
                     fused_embeddings, fusion_weights = self.fusion_module(embeddings)
                     # <<< DEBUG PRINT FUSION >>>
                     # print(f"DEBUG Forward ({self._fusion_type}): Fused Embed Shape: {fused_embeddings.shape}, HasNaN: {torch.isnan(fused_embeddings).any().item()}, Min: {torch.min(fused_embeddings).item() if fused_embeddings.numel() > 0 else 'N/A':.4f}, Max: {torch.max(fused_embeddings).item() if fused_embeddings.numel() > 0 else 'N/A':.4f}")
                     if self.global_classifier:
                         global_logits = self.global_classifier(fused_embeddings)
                         # <<< DEBUG PRINT FINAL LOGITS >>>
                         # print(f"DEBUG Forward ({self._fusion_type}): Global Logits Shape: {global_logits.shape}, HasNaN: {torch.isnan(global_logits).any().item()}")
                     else: # Should have been caught in __init__, but safety check
                         raise AttributeError(f"Fusion type '{self._fusion_type}' requires self.global_classifier, but it is None.")
                     if self.user_classifier: # Only calculate if classifier exists
                         user_logits = self.user_classifier(fused_embeddings)
                         # <<< DEBUG PRINT FINAL LOGITS >>>
                         # print(f"DEBUG Forward ({self._fusion_type}): User Logits Shape: {user_logits.shape}, HasNaN: {torch.isnan(user_logits).any().item()}")
                 # else: # Already checked in __init__
             except Exception as e:
                 print(f"\n!!! ERROR during Fusion/Classification processing: {e}")
                 traceback.print_exc()
                 raise e

         # --- Return ---
         if global_logits is None:
             raise RuntimeError("Forward pass logic error: global_logits was not assigned.")

         return global_logits, user_logits, fusion_weights

     def training_step(self, batch: HeteroData, batch_idx: int) -> Dict[str, torch.Tensor]:
         # training_step MUST return a dictionary containing at least the 'loss' key
         step_output = self._common_step(batch, batch_idx, stage='train')
         if 'loss' not in step_output:
             # Add error handling just in case _common_step changes
             raise ValueError("_common_step did not return 'loss' required by training_step")
         return step_output # Return the dict containing 'loss'

     def validation_step(self, batch: HeteroData, batch_idx: int):
         # For validation_step, Lightning primarily uses the logged values.
         _ = self._common_step(batch, batch_idx, stage='val')
         # Return value is optional unless you need it for manual reduction

     def test_step(self, batch: HeteroData, batch_idx: int):
         # For test_step also, Lightning primarily uses the logged values.
         _ = self._common_step(batch, batch_idx, stage='test')
         # Return value is optional

     # ----------------------------------------------------
     # Configure Optimizers (FIXED)
     # ----------------------------------------------------
     def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
         """Configure optimizer (AdamW) and optional LR scheduler."""

         # --- Parameter Groups (Differential LR) ---
         text_params = list(self.text_encoder.parameters()) # Get parameters from text encoder

         # --- FIX: Correctly Populate other_params ---
         other_params = []
         other_param_sources = [self.gnn_encoder, self.seq_encoder]

         # Add fusion module parameters IF it exists and is NOT multi_task (which has internal classifiers/params)
         if self.fusion_module and self._fusion_type != 'multi_task':
             other_param_sources.append(self.fusion_module)
         # Add classifier parameters IF they exist (for attention/gating fusion)
         if self.global_classifier:
             other_param_sources.append(self.global_classifier)
         if self.user_classifier:
             other_param_sources.append(self.user_classifier)
         # Add the direct GNN classifier parameters (always initialized)
         if hasattr(self, 'gnn_direct_classifier'):
             other_param_sources.append(self.gnn_direct_classifier)

         # Iterate through the sources and collect their parameters
         print("[INFO] Collecting parameters for 'other_modules' group...")
         for module in other_param_sources:
             if module is not None: # Ensure module exists
                 module_params = list(module.parameters())
                 print(f"  - Found {len(module_params)} params in {type(module).__name__}")
                 other_params.extend(module_params)
             else:
                 print(f"  - Skipping a None module source.")
         # --- END FIX ---

         # Filter for parameters requiring gradients (as before)
         text_params = [p for p in text_params if p.requires_grad]
         other_params = [p for p in other_params if p.requires_grad]

         # Create parameter groups
         param_groups = []
         base_lr = self.hparams.learning_rate
         # Use a smaller LR for the pre-trained text encoder (optional, but common)
         if text_params:
             param_groups.append({'params': text_params, 'lr': base_lr * 0.1, 'name': 'text_encoder'})
         if other_params:
             param_groups.append({'params': other_params, 'lr': base_lr, 'name': 'other_modules'})

         if not param_groups:
             raise ValueError("No parameters requiring gradients found for the optimizer.")
         else:
             print("[INFO] Optimizer param groups created:")
             for group in param_groups:
                 param_count = sum(p.numel() for p in group['params'])
                 print(f"  - Group '{group.get('name', 'Unnamed')}': {len(group['params'])} tensors, {param_count} total parameters, LR={group['lr']}")

         # Create optimizer (as before)
         optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)

         # --- Return Optimizer Only (For Overfitting Test / No Scheduler) ---
         # --- Make sure the function returns the optimizer like this ---
         print("[INFO] configure_optimizers: Returning optimizer ONLY (scheduler disabled for now).")
         return optimizer

         # --- Original Scheduler Code (Keep Commented Out/Remove if not needed yet) ---
         # If you need the scheduler later, uncomment and return the dictionary:
         # print("[INFO] configure_optimizers: Returning optimizer AND OneCycleLR scheduler.")
         # steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs # Or calculate based on DataLoader
         # total_steps = self.trainer.estimated_stepping_batches
         # print(f"Scheduler: total_steps={total_steps}, pct_start={self.hparams.scheduler_pct_start}")
         # scheduler = OneCycleLR(
         #     optimizer,
         #     max_lr=[pg['lr'] for pg in param_groups], # Max LR per group
         #     total_steps=total_steps,
         #     pct_start=self.hparams.scheduler_pct_start,
         #     anneal_strategy='cos' # or 'linear'
         # )
         # return {
         #     "optimizer": optimizer,
         #     "lr_scheduler": {
         #         "scheduler": scheduler,
         #         "interval": "step", # Call scheduler HPT step
         #         "frequency": 1,
         #         "monitor": "train_loss", # Optional: Monitor a metric
         #      },
         # }
         # --- End Original Scheduler Code ---

     # ----------------------------------------------------
     # Common Step for Loss / Metrics
     # ----------------------------------------------------
     def _common_step(self, batch: HeteroData, batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
         """
         Common logic for training, validation, and test steps.
         Handles forward pass, loss calculation, and metric logging.
         Ensures labels match logits especially for val/test with NeighborLoader.
         """
         # --- Debug: Print batch attributes ---
         # if batch_idx == 0: 
         #     print(f"--- Attributes of batch object (stage={stage}, batch_idx={batch_idx}) ---")
         #     try:
         #         # print(dir(batch))
         #         # Also print node stores if they exist
         #         if hasattr(batch, 'node_stores'):
         #             print("Inspecting batch.node_stores:")
         #             for i, store in enumerate(batch.node_stores):
         #                 store_key = getattr(store, '_key', '[NO _key]') # Safely get key
         #                 print(f"  Store {i}: type={type(store)}, key={store_key}")
         #                 # print(f"    Attributes: {dir(store)}") # Optional: print all attrs again
         #     except Exception as e_dir:
         #         print(f"Error printing batch dir: {e_dir}")
         #     print("---------------------------------------------------------------------")
         # --- End Debug ---

         # --- Forward Pass ---
         try:
             global_logits, user_logits, fusion_weights = self.forward(batch)
         except Exception as e:
             print(f"[ERROR] Exception during {stage} forward pass (batch {batch_idx}): {e}")
             dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True if stage=='train' else False)
             return {'loss': dummy_loss} if stage == 'train' else {}

         # --- Prepare Labels using input_id and full graph lookup (Third time's the charm?) --- 
         labels_global, labels_user = None, None
         num_seed_nodes = None
         if global_logits is not None:
             num_seed_nodes = global_logits.shape[0]
         elif user_logits is not None:
             num_seed_nodes = user_logits.shape[0]

         seed_node_original_indices = None
         transaction_store = None

         # Iterate through node stores to find the 'transaction' store
         if hasattr(batch, 'node_stores'):
             for store in batch.node_stores:
                 # Check the _key attribute exists and equals 'transaction'
                 if hasattr(store, '_key') and store._key == 'transaction':
                     transaction_store = store
                     break # Stop after finding the store
         
         # Check if we found the store and if it has input_id
         if transaction_store is not None and hasattr(transaction_store, 'input_id'):
             seed_node_original_indices = transaction_store.input_id
             # Verify length matches num_seed_nodes if possible
             if num_seed_nodes is not None and len(seed_node_original_indices) != num_seed_nodes:
                 print(f"[WARN] {stage} step (batch {batch_idx}): transaction_store.input_id length ({len(seed_node_original_indices)}) != num_seed_nodes ({num_seed_nodes}). Using first N.")
                 seed_node_original_indices = seed_node_original_indices[:num_seed_nodes]
             elif num_seed_nodes is None: 
                 print(f"[WARN] {stage} step (batch {batch_idx}): num_seed_nodes is None despite logits existing?")
         
         # Handle cases where lookup failed
         else: # Covers transaction_store is None OR transaction_store lacks input_id
             if transaction_store is None:
                 print(f"[WARN] {stage} step (batch {batch_idx}): Did not find 'transaction' node store in batch.node_stores list.")
             else: # transaction_store exists but no input_id
                 print(f"[WARN] {stage} step (batch {batch_idx}): Found 'transaction' store but it's missing 'input_id'.")
             print(f"[WARN] {stage} step (batch {batch_idx}): Cannot map seed nodes for label lookup.")
             seed_node_original_indices = None # Ensure it's None if lookup failed

         # --- Fetch Labels from Full Graph using Original Indices --- 
         if seed_node_original_indices is not None and self._full_graph_data is not None:
             try:
                 original_indices_cpu = seed_node_original_indices.cpu().long()
                 
                 # Fetch Global Labels
                 if hasattr(self._full_graph_data['transaction'], 'y_global'):
                      labels_global = self._full_graph_data['transaction'].y_global[original_indices_cpu]
                      if global_logits is not None:
                           labels_global = labels_global.to(global_logits.device)
                           # <<< CLAMP IMMEDIATELY >>>
                           num_global_classes_expected = self.hparams.num_global_classes
                           labels_global = torch.clamp(labels_global, 0, num_global_classes_expected - 1)
                 else:
                      print(f"[WARN] {stage} step (batch {batch_idx}): _full_graph_data['transaction'] has no y_global.")
                      labels_global = None # Ensure it's None if not found

                 # Fetch User Labels
                 if hasattr(self._full_graph_data['transaction'], 'y_user'):
                      labels_user = self._full_graph_data['transaction'].y_user[original_indices_cpu]
                      if user_logits is not None:
                           labels_user = labels_user.to(user_logits.device)
                           # <<< CLAMP IMMEDIATELY >>>
                           num_user_classes_expected = self.hparams.num_user_classes
                           labels_user = torch.clamp(labels_user, 0, num_user_classes_expected - 1)
                 else:
                      labels_user = None # Ensure it's None if not found
                  
             except IndexError as e:
                  print(f"[ERROR] {stage} step (batch {batch_idx}): IndexError during label lookup from full graph: {e}. Original indices might be invalid.")
                  labels_global, labels_user = None, None 
             except Exception as e:
                  print(f"[ERROR] {stage} step (batch {batch_idx}): Unexpected error during label lookup: {e}")
                  labels_global, labels_user = None, None
         else:
             # Print more specific warnings
             if seed_node_original_indices is None:
                  print(f"[WARN] {stage} step (batch {batch_idx}): Cannot perform label lookup because seed node original indices (input_id) were not found.")
             if self._full_graph_data is None:
                  print(f"[WARN] {stage} step (batch {batch_idx}): Cannot perform label lookup because _full_graph_data reference is missing.")

         # Verify final label shapes match logits shapes (keep this check)
         if labels_global is not None and global_logits is not None and labels_global.shape[0] != global_logits.shape[0]:
              print(f"[ERROR] {stage} step (batch {batch_idx}): Final labels_global shape {labels_global.shape} != global_logits shape {global_logits.shape}. Resetting labels.")
              labels_global = None
         if labels_user is not None and user_logits is not None and labels_user.shape[0] != user_logits.shape[0]:
              print(f"[ERROR] {stage} step (batch {batch_idx}): Final labels_user shape {labels_user.shape} != user_logits shape {user_logits.shape}. Resetting labels.")
              labels_user = None

         # --- Loss and Metric Calculation ---
         loss = torch.tensor(0.0, device=self.device)
         log_dict = {}

         # Global Task (Now uses the already clamped labels_global)
         if global_logits is not None and labels_global is not None:
             if global_logits.shape[0] == labels_global.shape[0]:
                 try:
                     loss_global = self.criterion(global_logits, labels_global) 
                     if torch.isnan(loss_global).any() or torch.isinf(loss_global).any():
                          print(f"[WARN] {stage} step (batch {batch_idx}): NaN/Inf detected in global loss. Logits min/max: {global_logits.min():.2f}/{global_logits.max():.2f}")
                          # Handle NaN loss - maybe skip update or use a default value?
                          # For now, add 0 to total loss if NaN/Inf occurs
                          loss_global_val = 0.0
                     else:
                          loss_global_val = loss_global.item() # Get scalar value for adding if not NaN/Inf
                          loss = loss + loss_global # Add tensor loss for backprop if training

                     # Use num_seed_nodes for logging batch size if available
                     log_batch_size = num_seed_nodes if num_seed_nodes is not None else global_logits.shape[0]

                     log_dict[f'{stage}_loss_global'] = loss_global_val
                     # Calculate accuracy
                     with torch.no_grad(): # Ensure accuracy calc doesn't affect gradients
                         preds_global = torch.argmax(global_logits, dim=1)
                         # Ensure labels are long type for comparison
                         labels_global_long = labels_global.long()
                         correct_global = (preds_global == labels_global_long).float()
                         acc_global_val = correct_global.mean().item()
                     log_dict[f'{stage}_acc_global'] = acc_global_val
                 except Exception as e:
                      print(f"[ERROR] Exception during {stage} global loss/metric calculation (batch {batch_idx}): {e}")
             else:
                 print(f"[WARN] {stage} step (batch {batch_idx}): Mismatch! global_logits shape {global_logits.shape} != labels_global shape {labels_global.shape}. Skipping global loss/acc.")
         elif global_logits is not None and labels_global is None:
              print(f"[WARN] {stage} step (batch {batch_idx}): Skipping global loss/metrics because labels_global is None after lookup attempt.")

         # User Task (Now uses the already clamped labels_user)
         if user_logits is not None and labels_user is not None:
             if user_logits.shape[0] == labels_user.shape[0]:
                 try:
                     # Debug print now uses clamped labels 
                     user_label_min = labels_user.min().item()
                     user_label_max = labels_user.max().item()
                     num_user_classes_expected = self.hparams.num_user_classes
                     print(f"DEBUG User Labels (batch {batch_idx}): min={user_label_min}, max={user_label_max}, num_classes={num_user_classes_expected}")
                     # This error check is now less critical as clamping already happened, but keep for info
                     if user_label_min < 0 or user_label_max >= num_user_classes_expected:
                         print(f"[!!!WARN!!!] User labels were out of range [0, {num_user_classes_expected - 1}] before clamping!")

                     loss_user = self.criterion(user_logits, labels_user)
                     if torch.isnan(loss_user).any() or torch.isinf(loss_user).any():
                          print(f"[WARN] {stage} step (batch {batch_idx}): NaN/Inf detected in user loss. Logits min/max: {user_logits.min():.2f}/{user_logits.max():.2f}")
                          loss_user_val = 0.0
                     else:
                          loss_user_val = loss_user.item()
                          loss = loss + loss_user # Add tensor loss if training

                     log_batch_size = num_seed_nodes if num_seed_nodes is not None else user_logits.shape[0]
                     log_dict[f'{stage}_loss_user'] = loss_user_val

                     with torch.no_grad():
                         preds_user = torch.argmax(user_logits, dim=1)
                         # Ensure labels are long type for comparison
                         labels_user_long = labels_user.long()
                         correct_user = (preds_user == labels_user_long).float()
                         acc_user_val = correct_user.mean().item()
                     log_dict[f'{stage}_acc_user'] = acc_user_val
                 except Exception as e:
                      print(f"[ERROR] Exception during {stage} user loss/metric calculation (batch {batch_idx}): {e}")
             else:
                  print(f"[WARN] {stage} step (batch {batch_idx}): Mismatch! user_logits shape {user_logits.shape} != labels_user shape {labels_user.shape}. Skipping user loss/acc.")
         elif user_logits is not None and labels_user is None:
              # This might be expected if user task is optional or failed lookup
              pass

         # Log the combined loss (scalar value)
         # If loss tensor contains NaNs from calculation, log 0.0 or a placeholder
         if torch.isnan(loss).any() or torch.isinf(loss).any():
              print(f"[WARN] {stage} step (batch {batch_idx}): Combined loss tensor is NaN/Inf. Logging 0.0.")
              log_dict[f'{stage}_loss'] = 0.0 # Log scalar 0.0
              # If training, we need to return a valid loss tensor. Re-create 0.0 tensor.
              loss = torch.tensor(0.0, device=self.device, requires_grad=True if stage=='train' else False)
         else:
              log_dict[f'{stage}_loss'] = loss.item() # Log the scalar value of combined loss

         # --- Logging ---
         # Use log_dict for cleaner logging
         # Use num_seed_nodes if available, otherwise fallback to logits shape[0] or 1
         log_batch_size_fallback = 1
         if global_logits is not None:
             log_batch_size_fallback = global_logits.shape[0]
         elif user_logits is not None: # If global is None, try user
             log_batch_size_fallback = user_logits.shape[0]

         log_batch_size_final = num_seed_nodes if num_seed_nodes is not None else log_batch_size_fallback
         
         self.log_dict(log_dict, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, batch_size=log_batch_size_final, sync_dist=True)

         # Add debug print before returning
         # print(f"--- {stage} step finished for batch {batch_idx}. Loss: {loss.item() if isinstance(loss, torch.Tensor) and loss.requires_grad else 'N/A'} --- ") 

         # --- Return Value ---
         # training_step requires a dict containing 'loss' key with the loss tensor
         if stage == 'train':
             # Ensure we return the loss *tensor* for backpropagation
             return {'loss': loss}
         else:
             # validation_step and test_step don't strictly need a return value if logging is done
             return {} # Return empty dict
