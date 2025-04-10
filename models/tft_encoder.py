import torch
import torch.nn as nn
from typing import Any, Dict, Optional

# --- IMPORTANT --- 
# You need to install pytorch-forecasting: pip install pytorch-forecasting
# The code below will attempt to import it.
from pytorch_forecasting.models import TemporalFusionTransformer



# Wrapper class to integrate pytorch-forecasting's TFT
class PytorchForecastingTFTWrapper(nn.Module):
    """ 
    Wrapper for TemporalFusionTransformer from pytorch-forecasting.
    Handles input/output adaptation.
    """
    def __init__(self, 
                 output_dim: int = 128, # Desired dimension for the *final embedding* output
                 tft_params: Dict[str, Any] = {}, # Params passed directly to library's TFT
                 embedding_source_key: str = 'encoder_variables' # Key in TFT output dict to use as embedding
                 ):
        super().__init__()
        print(f"[INFO] Initializing TFT Wrapper. Expecting output embedding dim: {output_dim}")
        
        self.embedding_source_key = embedding_source_key
        
        # --- Instantiate the actual TFT model --- 
        # Ensure required params for pytorch-forecasting TFT are in tft_params:
        # e.g., hidden_size, lstm_layers, num_heads, dropout, output_size, loss, 
        #       logging_metrics, time_varying_known_reals, time_varying_unknown_reals, 
        #       static_categoricals, static_reals, etc.
        # These should be defined in your model_config.yaml
        print(f"  Instantiating TemporalFusionTransformer with params: {list(tft_params.keys())}")
        try:
             # Ensure output_size matches expected internal TFT dimension if needed
             # We might need a projection layer if TFT output dim != desired output_dim
             if 'output_size' not in tft_params:
                  # TFT output_size often relates to prediction targets, not necessarily embedding dim
                  # Using a default, but this needs careful configuration.
                  tft_params['output_size'] = 1 
                  print(f"[WARN] TFT Wrapper: 'output_size' not in tft_params, defaulting to 1.")
                  
             # --- Workaround for BaseModel TypeError ---
             # 1. Identify parameters only for BaseModel init (loss, logging_metrics)
             base_model_params = {}
             if 'loss' in tft_params:
                 base_model_params['loss'] = tft_params['loss']
             # Add logging_metrics if needed

             # 2. Initialize TFT with only BaseModel parameters
             print(f"  Initializing TemporalFusionTransformer minimally with params: {list(base_model_params.keys())}")
             self.tft = TemporalFusionTransformer(**base_model_params)
             print("  Minimal initialization complete.")

             # 3. Manually set the remaining TFT-specific parameters on the instance's hparams
             # Ensure hparams namespace exists (it should after minimal init, but check)
             if not hasattr(self.tft, 'hparams'):
                  print("[WARN] self.tft missing hparams attribute after minimal init. Creating Namespace.")
                  import argparse # Import locally if needed
                  self.tft.hparams = argparse.Namespace()
             else:
                  print("  self.tft.hparams found.")
                  
             tft_specific_params_to_set = [ 
                 'hidden_size', 'lstm_layers', 'num_heads', 'dropout', 'output_size',
                 'static_categoricals', 'static_reals',
                 'time_varying_known_categoricals', 'time_varying_known_reals',
                 'time_varying_unknown_categoricals', 'time_varying_unknown_reals',
                 'categorical_groups', 'embedding_sizes' # Add other relevant params like embedding_paddings if used
             ]
             print("  Manually setting TFT hparams after minimal init...")
             for key in tft_specific_params_to_set:
                 if key in tft_params:
                     setattr(self.tft.hparams, key, tft_params[key]) 
                     print(f"    Set self.tft.hparams.{key}")
                 else:
                     print(f"    Skipping {key} (not in provided tft_params)")

             # 4. --- Attempt to rebuild network based on updated hparams ---
             try:
                 print("  Attempting self.tft._build_network()")
                 self.tft._build_network()
                 print("  _build_network() called successfully.")
             except Exception as build_err:
                 print(f"[WARN] Failed to call self.tft._build_network(): {build_err}")
                 print("  Continuing without explicit network rebuild, which may cause forward errors.")

             # Determine the actual dimension of the chosen embedding source
             # Store max_encoder_length for use in forward pass
             self.max_encoder_length = self.tft.hparams.get("max_encoder_length", 10) # Default based on TFT source
             # Use hidden_size as estimate if projection is needed later
             actual_embedding_dim = self.tft.hparams.hidden_size
             print(f"  TFT Initialized. Internal embedding dim (estimated): {actual_embedding_dim}")
             
        except TypeError as e:
             print(f"[ERROR] Failed to initialize TemporalFusionTransformer: {e}")
             print("  Ensure all required arguments for pytorch-forecasting's TFT are provided in tft_params.")
             raise e
        except NameError: # If TFT wasn't imported
             print("[ERROR] TemporalFusionTransformer class not available. Install/Import pytorch-forecasting.")
             raise

        # Optional projection layer if the extracted embedding dim doesn't match desired output_dim
        self.projection = None
        if actual_embedding_dim != output_dim:
             print(f"  Adding projection layer from TFT embedding ({actual_embedding_dim}) to desired output ({output_dim})")
             self.projection = nn.Linear(actual_embedding_dim, output_dim)
             self.output_dim = output_dim
        else:
             self.output_dim = actual_embedding_dim # Output dim is the TFT internal dim

    def forward(self, 
                sequence_batch: Dict[str, Any], # Expects dict from Collator
                device: Optional[torch.device] = None, 
                **kwargs 
                ) -> torch.Tensor:
        """
        Wrapper forward method.
        1. Reformats input dictionary.
        2. Calls underlying TFT.
        3. Extracts desired embedding.
        4. Optionally projects embedding.
        """
        # --- 1. Reformat input for pytorch-forecasting TFT --- 
        # The input `sequence_batch` comes from our DataModule V2 
        # and contains 'sequences' and 'lengths' keys.
        # We need to adapt this to the format expected by the TFT library.
        
        if not isinstance(sequence_batch, dict) or 'sequences' not in sequence_batch or 'lengths' not in sequence_batch:
             print(f"[ERROR] TFT Wrapper: Invalid input sequence_batch format. Expected dict with 'sequences' and 'lengths'. Got: {type(sequence_batch)}")
             # Try to determine batch size for fallback zeros
             bs = 1
             if isinstance(sequence_batch, dict) and 'sequences' in sequence_batch:
                  bs = sequence_batch['sequences'].shape[0]
             elif isinstance(sequence_batch, torch.Tensor):
                  bs = sequence_batch.shape[0]
             return torch.zeros(bs, self.output_dim, device=torch.device('cpu'))

        sequences = sequence_batch['sequences']
        lengths = sequence_batch['lengths']
        # Expect categorical features as well now
        sequences_cat = sequence_batch.get('seq_cat_features', None)
        
        batch_size, max_seq_len_in_batch, num_real_features = sequences.shape
        # --- Get device from the underlying TFT model's parameters --- 
        try:
            target_device = next(self.tft.parameters()).device
        except StopIteration:
             print("[WARN] TFT Wrapper forward: Could not determine device from TFT parameters. Falling back to CPU.")
             target_device = torch.device("cpu")
        # target_device = self.device # Incorrect for nn.Module
        
        # --- Check Categorical Features --- 
        num_cat_features = 0
        if sequences_cat is None:
             print("[WARN] TFT Wrapper forward: 'seq_cat_features' not found in sequence_batch. Using placeholder.")
             # Use 0 feature dimension if none provided
             sequences_cat = torch.zeros((batch_size, max_seq_len_in_batch, 0), dtype=torch.long, device=target_device) 
        elif sequences_cat.shape[0] != batch_size or sequences_cat.shape[1] != max_seq_len_in_batch:
             print(f"[ERROR] TFT Wrapper forward: Mismatched shape for seq_cat_features. Expected ({batch_size}, {max_seq_len_in_batch}, ...), Got {sequences_cat.shape}")
             return torch.zeros(batch_size, self.output_dim, device=target_device) # Fallback
        else:
             num_cat_features = sequences_cat.shape[2]

        # --- Construct dictionary based on forward() signature --- 
        # We only have encoder inputs
        x_for_tft = {
             'encoder_cont': sequences.to(target_device),
             'encoder_cat': sequences_cat.to(target_device),
             'encoder_lengths': lengths.to(target_device),

             # Decoder inputs are empty as we use TFT only as encoder
             # Ensure correct shapes: [batch, time=0, features]
             'decoder_cont': torch.zeros((batch_size, 0, num_real_features), dtype=torch.float, device=target_device),
             'decoder_cat': torch.zeros((batch_size, 0, num_cat_features), dtype=torch.long, device=target_device),
             'decoder_lengths': torch.zeros_like(lengths, device=target_device), # All decoder lengths are 0

             # Other potentially necessary keys (placeholders)
             'static_categoricals': torch.zeros((batch_size, 0), dtype=torch.long, device=target_device),
             'static_reals': torch.zeros((batch_size, 0), dtype=torch.float, device=target_device),
             # Add placeholders for time_idx and groups, as they might be needed internally
             'time_idx': torch.arange(max_seq_len_in_batch, device=target_device).unsqueeze(0).expand(batch_size, -1),
             'groups': torch.arange(batch_size, device=target_device),
             # Target might also be needed, even if just for shape consistency in some layers
             'target': torch.zeros((batch_size, max_seq_len_in_batch), dtype=torch.float, device=target_device),
        }
        
        # Input dict is now constructed entirely on the target_device
        x_for_tft_dev = x_for_tft 
        
        # --- DEBUG: Check devices before calling TFT forward ---
        print(f"--- TFT Wrapper Forward Device Check (Target: {target_device}) ---")
        for key, tensor in x_for_tft_dev.items():
            if isinstance(tensor, torch.Tensor):
                 print(f"  Input '{key}' device: {tensor.device}, shape: {tensor.shape}")
            else:
                 print(f"  Input '{key}': Not a tensor ({type(tensor)}) ")
        try:
             model_param_device = next(self.tft.parameters()).device
             print(f"  Model parameter device: {model_param_device}")
        except StopIteration:
             print("  Model has no parameters?")
        print("---------------------------------------------------------")
        
        # --- 2. Call underlying TFT --- 
        try:
            # The library TFT returns a dictionary of outputs
            tft_output_dict = self.tft(x_for_tft_dev)
        except RuntimeError as e:
             print(f"[FATAL ERROR] Runtime Error during self.tft(x_for_tft_dev): {e}")
             # Print devices again to see if anything changed?
             print("--- Devices After Error ---")
             for key, tensor in x_for_tft_dev.items():
                 if isinstance(tensor, torch.Tensor): print(f"  Input '{key}' device: {tensor.device}")
             try: model_param_device = next(self.tft.parameters()).device; print(f"  Model parameter device: {model_param_device}")
             except StopIteration: print("  Model has no parameters?")
             print("-------------------------")
             raise e # Re-raise the error
        except Exception as e:
             print(f"[FATAL ERROR] Unexpected Error during self.tft(x_for_tft_dev): {e}")
             raise e

        # --- 3. Extract Embedding --- 
        # Identify the correct tensor to use as the sequence embedding.
        # This depends on the TFT implementation. Common choices might be:
        # - Output of the final GRN before prediction heads
        # - Aggregated output of the attention layer
        # Check the keys of tft_output_dict and documentation.
        # Using key specified in self.embedding_source_key.
        if self.embedding_source_key not in tft_output_dict:
             print(f"[ERROR] TFT Wrapper: Embedding key '{self.embedding_source_key}' not found in TFT output dict.")
             print(f"  Available keys: {list(tft_output_dict.keys())}")
             # Fallback to zeros
             batch_size = len(sequence_batch.get('encoder_lengths', [0]))
             embedding = torch.zeros(batch_size, self.output_dim, device=target_device)
        else:
             embedding = tft_output_dict[self.embedding_source_key]
             # Embedding might need aggregation if it's per-timestep, e.g., take last step:
             if embedding.ndim == 3: # [batch, sequence, features]
                 print("[WARN] TFT Wrapper: Extracted embedding has 3 dims. Taking last time step.")
                 embedding = embedding[:, -1, :] # Take last time step

        # --- 4. Optional Projection --- 
        if self.projection:
             embedding = self.projection(embedding)

        return embedding

    def get_output_dim(self):
        """ Helper to get the final output dimension after potential projection. """
        return self.output_dim 