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
         num_classes: int,
         # --- GNN Params ---
         gnn_node_input_dims: Dict[str, int],
         # gnn_edge_input_dims: Optional[Dict[Tuple[str, str, str], int]] = None, # Might not be needed if GNN handles internally
         gnn_hidden_channels: int = 256,
         gnn_out_channels: int = 256, # Output dim of GNN
         gnn_num_layers: int = 2,
         gnn_heads: int = 4, # Ensure hidden_channels % heads == 0
         gnn_metadata: Optional[Tuple[List[str], List[Tuple[str, str, str]]]] = None, # Pass metadata for refactored GNN
         # --- Sequence Params ---
         seq_input_dim: int = 4, # Check if this matches DataModule sequence_feature_dim
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
         # --- Debug Flag ---
         gnn_only_test_mode: bool = False # Flag to run only GNN + Classifier
     ):
         super().__init__()
         # Save hyperparameters for logging and access via self.hparams
         # If you need class_weights later from hparams, remove 'class_weights' from ignore list.
         self.save_hyperparameters(ignore=['class_weights'])

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

         # Store the GNN-only mode flag
         self.gnn_only_test_mode = self.hparams.gnn_only_test_mode
         if self.gnn_only_test_mode:
             print("[INFO] TransactionClassifier running in GNN-ONLY TEST MODE.")

         # --- Initialize Encoders ---
         node_types_list = list(self.hparams.gnn_node_input_dims.keys())

         self.gnn_encoder = HeteroGNNEncoder(
             node_types=node_types_list,
             metadata=self.hparams.gnn_metadata, # Pass metadata tuple (node_types, edge_types)
             in_channels=self.hparams.gnn_node_input_dims,
             hidden_channels=self.hparams.gnn_hidden_channels,
             out_channels=self.hparams.gnn_out_channels,
             num_layers=self.hparams.gnn_num_layers,
             heads=self.hparams.gnn_heads,
             # dropout=... # Pass GNN dropout if applicable
         )

         self.seq_encoder = SequenceEncoder(
             input_dim=self.hparams.seq_input_dim,
             hidden_dim=self.hparams.seq_hidden_size,
             num_layers=self.hparams.seq_num_layers,
             # dropout=... # Pass Sequence dropout if applicable
         )

         self.text_encoder = MultiFieldTextEncoder(
             model_name=self.hparams.text_model_name,
             max_length=self.hparams.text_max_length,
             # output_dim=self.hparams.text_out_dim # Assuming handled internally
         )
         # Use defined text_out_dim for consistency downstream
         text_hidden_size = self.hparams.text_out_dim # This assumes MultiFieldTextEncoder outputs this dim

         # --- Initialize Fusion Module and Classifiers ---
         # Dimensions for input to fusion module
         self.modality_dims = {
             'graph': self.hparams.gnn_out_channels,
             'sequence': self.hparams.seq_hidden_size,
             'text': text_hidden_size
         }
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
             # Classifier takes output of fusion module
             self.global_classifier = Linear(self.hparams.fusion_hidden_dim, self.hparams.num_classes)
             self.user_classifier = Linear(self.hparams.fusion_hidden_dim, self.hparams.num_classes) # Assuming same num_classes

         elif self._fusion_type == 'multi_task':
             self.fusion_module = MultiTaskFusion(
                 input_dims=self.modality_dims,
                 hidden_dim=self.hparams.fusion_hidden_dim,
                 num_global_classes=self.hparams.num_classes,
                 num_user_classes=self.hparams.num_classes, # Assuming same num_classes
                 dropout=self.hparams.fusion_dropout
             )
             # Classifiers are internal to MultiTaskFusion, set main ones to None
         else:
             raise ValueError(f"Unsupported fusion_type: {self._fusion_type}")

         # --- Define Separate Classifier for GNN-Only Mode ---
         # Takes GNN output directly. Initialized unconditionally.
         self.gnn_direct_classifier = Linear(self.hparams.gnn_out_channels, self.hparams.num_classes)
         print(f"[INFO] Initialized gnn_direct_classifier: Linear({self.hparams.gnn_out_channels}, {self.hparams.num_classes})")

         # --- The second redundant criterion initialization block has been removed ---

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
             print("\n--- DEBUG: Edge Attrs Dict Being Passed to GNN Encoder ---")
             if edge_attr_to_pass_to_gnn is not None:
                 for et, ea in edge_attr_to_pass_to_gnn.items():
                     # Print edge type and the shape/dtype of the corresponding tensor
                     print(f"  Edge Type {et}: Shape={ea.shape if hasattr(ea, 'shape') else 'N/A'}, Dtype={ea.dtype if hasattr(ea, 'dtype') else 'N/A'}")
             else:
                 # Print if no edge attributes are being passed
                 print("  edge_attr_dict passed to GNN Encoder is: None")
             print("--- END DEBUG ---")
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

             except Exception as e:
                print(f"\n!!! ERROR during Sequence processing: {e}")
                traceback.print_exc()
                raise e
             # try:
             #     # Ensure slicing uses the potentially adjusted num_seed_nodes
             #     seq_input = batch['transaction'].seq_features[:num_seed_nodes]
             #     seq_lens = getattr(batch['transaction'], 'seq_lengths', None)
             #     if seq_lens is not None:
             #         seq_lens = seq_lens[:num_seed_nodes]

             #         invalid_len_mask = seq_lens < 0

                     
             #         if (seq_lens <= 0).any():
             #             # This warning persists, needs fix in DataModule
             #             print(f"[WARN] Forward: Found {(seq_lens <= 0).sum().item()} sequences with length <= 0. Clamping to 1.")
             #             seq_lens = torch.clamp(seq_lens, min=1)

             #     _, _, seq_features = self.seq_encoder(seq_input, lengths=seq_lens)
             # except Exception as e:
             #     print(f"\n!!! ERROR during Sequence processing: {e}")
             #     traceback.print_exc()
             #     raise e

             # 3. Text Processing
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
             except Exception as e:
                 print(f"\n!!! ERROR during Text processing: {e}")
                 traceback.print_exc()
                 raise e

             # 4. Fusion
             embeddings = {
                 'graph': gnn_features,
                 'sequence': seq_features,
                 'text': text_features
             }
             try:
                 if self._fusion_type == 'multi_task':
                     # MultiTaskFusion likely returns logits directly
                     global_logits, user_logits, fusion_weights = self.fusion_module(embeddings)
                 elif self._fusion_type in ['attention', 'gating']:
                     # Attention/Gating return fused embed + weights, needs separate classifiers
                     fused_embeddings, fusion_weights = self.fusion_module(embeddings)
                     if self.global_classifier:
                         global_logits = self.global_classifier(fused_embeddings)
                     else: # Should have been caught in __init__, but safety check
                         raise AttributeError(f"Fusion type '{self._fusion_type}' requires self.global_classifier, but it is None.")
                     if self.user_classifier: # Only calculate if classifier exists
                         user_logits = self.user_classifier(fused_embeddings)
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
         """ Common logic for training, validation, and test steps. """
         # Get model predictions
         global_logits, user_logits, fusion_weights = self(batch)

         # --- Loss Calculation ---
         if not hasattr(batch['transaction'], 'y_global'):
             print(f"[ERROR] Stage {stage}, Batch {batch_idx}: Batch object missing 'transaction.y_global' for labels!")
             raise KeyError("Batch object missing 'transaction.y_global' for labels.")

         # Get labels for seed nodes (Ensure slicing matches forward pass)
         num_seed_nodes = batch['transaction'].batch_size # Assuming this is correct count
         # Add safety check similar to forward pass if needed
         if num_seed_nodes > global_logits.shape[0]:
              print(f"[WARN] _common_step: num_seed_nodes ({num_seed_nodes}) > global_logits ({global_logits.shape[0]}). Mismatch likely.")
              # Decide how to handle: maybe slice labels to match logits?
              # num_seed_nodes = global_logits.shape[0] # Risky if labels don't align

         labels_global = batch['transaction'].y_global[:num_seed_nodes] # Slice labels
         labels_global = labels_global.long() # Ensure Long type for CrossEntropyLoss

         # --- START DEBUGGING Loss Inputs (for overfitting test) ---
         is_gnn_only_train = (stage == 'train' and getattr(self.hparams, 'gnn_only_test_mode', False))
         if is_gnn_only_train and batch_idx % 10 == 0: # Print occasionally
             print(f"\n--- Overfit Test Loss Input Check (Batch {batch_idx}, Global Step {self.trainer.global_step if hasattr(self,'trainer') else 'N/A'}) ---")
             print(f"  Logits (Input to Loss): shape={global_logits.shape}, dtype={global_logits.dtype}, device={global_logits.device}, "
                   f"min={global_logits.min().item():.4f}, max={global_logits.max().item():.4f}, mean={global_logits.mean().item():.4f}, "
                   f"hasNaN={torch.isnan(global_logits).any().item()}, hasInf={torch.isinf(global_logits).any().item()}")
             print(f"  Labels (Input to Loss): shape={labels_global.shape}, dtype={labels_global.dtype}, device={labels_global.device}, "
                   f"min={labels_global.min().item()}, max={labels_global.max().item()}")
             try:
                 unique_labels, counts = torch.unique(labels_global, return_counts=True)
                 print(f"  Unique Labels in Batch: {unique_labels.cpu().tolist()}")
                 print(f"  Label Counts in Batch: {counts.cpu().tolist()}")
             except Exception as e_unique:
                 print(f"  Error getting unique labels: {e_unique}")
             print(f"--- End Overfit Check ---")
         # --- END DEBUGGING ---


         # Validate label range BEFORE passing to loss
         num_classes = global_logits.size(1)
         invalid_mask = (labels_global < 0) | (labels_global >= num_classes)
         if invalid_mask.any():
             num_invalid = invalid_mask.sum().item()
             print(f"[WARN] Stage {stage}, Batch {batch_idx}: Found {num_invalid} invalid labels outside range [0, {num_classes-1}]. Clamping to 0.")
             # Clamp invalid labels to 0 (or another valid index)
             labels_global = torch.where(invalid_mask, torch.zeros_like(labels_global), labels_global)


         # Calculate global loss
         try:
             # Ensure criterion is initialized
             if self.criterion is None:
                 raise RuntimeError("self.criterion was not initialized in __init__.")
             global_loss = self.criterion(global_logits, labels_global)
             if torch.isnan(global_loss) or torch.isinf(global_loss):
                 print(f"[ERROR] Stage {stage}, Batch {batch_idx}: Calculated global_loss is NaN or Inf! Check inputs/model weights.")
                 # Optionally raise error or return dummy loss if needed
                 # raise ValueError("NaN/Inf loss detected")
                 return {'loss': global_loss} # Return the NaN/Inf loss to potentially stop training
         except Exception as e_loss:
             print(f"[ERROR] Stage {stage}, Batch {batch_idx}: Error during loss calculation: {e_loss}")
             traceback.print_exc()
             raise e_loss


         # --- User-Specific Loss (Optional - Currently Inactive) ---
         user_loss = torch.tensor(0.0, device=self.device) # Default zero loss
         # if not self.gnn_only_test_mode and user_logits is not None and hasattr(batch['transaction'], 'y_user'):
         #     labels_user = batch['transaction'].y_user[:num_seed_nodes].long() # Slice labels
         #     # Add validation for user labels similar to global labels if needed
         #     user_loss = self.criterion(user_logits, labels_user) # Use same criterion or different one?
         #     self.log(f'{stage}_user_loss', user_loss, on_step=(stage=='train'), on_epoch=True, batch_size=num_seed_nodes, logger=True)
         #     # Log user accuracy etc.
         #     user_preds = user_logits.argmax(dim=-1)
         #     user_acc = (user_preds == labels_user).float().mean()
         #     self.log(f'{stage}_user_accuracy', user_acc, on_step=False, on_epoch=True, batch_size=num_seed_nodes, logger=True)


         # --- Total Loss ---
         # This is the loss value that will be used for backpropagation
         total_loss = global_loss # + user_loss # Add user_loss if active


         # --- Logging ---
         batch_size = num_seed_nodes # Use the actual number of nodes used for loss calculation
         self.log(f'{stage}_loss', total_loss, on_step=(stage=='train'), on_epoch=True, batch_size=batch_size, prog_bar=(stage=='train'), logger=True)
         if user_loss > 0: # Log constituent losses if needed
             self.log(f'{stage}_global_loss_contrib', global_loss, on_step=(stage=='train'), on_epoch=True, batch_size=batch_size, logger=True)
             self.log(f'{stage}_user_loss_contrib', user_loss, on_step=(stage=='train'), on_epoch=True, batch_size=batch_size, logger=True)
         # Log accuracy etc.
         global_preds = global_logits.argmax(dim=-1)
         global_acc = (global_preds == labels_global).float().mean()
         self.log(f'{stage}_global_accuracy', global_acc, on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True, logger=True)

         # Return dictionary - MUST contain 'loss' key for training
         output_dict = {'loss': total_loss}
         # Optionally add other items needed by hooks or callbacks
         # output_dict['logits'] = global_logits.detach()
         # output_dict['labels'] = labels_global.detach()
         return output_dict
