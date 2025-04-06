# models/sequence.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TemporalAttention(nn.Module):
    """
    Attention mechanism for temporal sequence modeling.
    Applies attention over the output steps of an RNN.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Simple linear attention mechanism
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False) # No bias needed for attention scores
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention to sequence.

        Args:
            x: Sequence tensor from RNN [batch_size, seq_len, hidden_dim]
            mask: Optional boolean attention mask [batch_size, seq_len]
                  (True for valid positions, False for padding)

        Returns:
            Attention-weighted context vector [batch_size, hidden_dim]
        """
        # Compute attention scores (unnormalized)
        # x shape: [batch_size, seq_len, hidden_dim]
        # scores shape: [batch_size, seq_len, 1]
        scores = self.attention_net(x)

        # Apply mask if provided to prevent attention on padding
        if mask is not None:
            # Ensure mask has the expected shape [batch_size, seq_len]
            if mask.shape != scores.shape[:-1]:
                raise ValueError(
                    f"Mask shape {mask.shape} incompatible with input sequence length {scores.shape[1]}"
                )
            # Mask out padding positions by setting scores to negative infinity
            # Unsqueeze mask to match scores shape for broadcasting: [batch_size, seq_len, 1]
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf')) # Mask value 0 means padding

        # Compute attention weights (softmax over sequence length dimension)
        # attn_weights shape: [batch_size, seq_len, 1]
        attn_weights = F.softmax(scores, dim=1)

        # Compute weighted sum context vector
        # attn_weights * x: [batch_size, seq_len, hidden_dim]
        # context shape: [batch_size, hidden_dim]
        context = torch.sum(attn_weights * x, dim=1)

        return context


class SequenceEncoder(nn.Module):
    """
    LSTM-based sequence encoder for transaction history.
    Processes sequences with features [amount, weekday, hour, time_delta].
    Uses TemporalAttention for pooling.
    """
    def __init__(
        self,
        input_dim: int, # Should be 4 from DataModule
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Validate input_dim expectation from DataModule
        expected_input_dim = 4
        if input_dim != expected_input_dim:
            # This warning helps catch mismatches during model initialization
            print(
                f"[WARN] SequenceEncoder initialized with input_dim={input_dim}, "
                f"but expected {expected_input_dim} ([amount, weekday, hour, time_delta]). "
                f"Ensure this matches DataModule output and TransactionClassifier parameters."
            )
            # Depending on severity, you might raise an error instead:
            # raise ValueError(f"SequenceEncoder expected input_dim={expected_input_dim}, got {input_dim}")

        # Normalize the 'amount' feature (index 0)
        self.amount_norm = nn.LayerNorm(1)

        # Optional: Normalize other features?
        # self.other_feature_norm = nn.LayerNorm(input_dim - 1) # Example if normalizing others

        # Define the actual input size fed into the LSTM
        # In this version, we use the normalized amount + original other features
        self.lstm_input_size = input_dim # Stays 4 if only normalizing amount

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Input tensor format: [batch_size, seq_len, feature_dim]
            dropout=dropout if num_layers > 1 else 0, # Apply dropout between LSTM layers
            bidirectional=bidirectional
        )

        # Calculate LSTM output dimension (accounts for bidirectionality)
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Layer normalization for the LSTM output sequence
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # Dropout applied to the final sequence representation before attention/output
        self.dropout = nn.Dropout(dropout)

        # Attention mechanism for weighted pooling over the output sequence
        self.attention = TemporalAttention(lstm_output_dim)


    def forward(
        self,
        x: torch.Tensor, # Expects shape [batch_size, seq_len, input_dim=4]
        lengths: Optional[torch.Tensor] = None, # Original lengths before padding [batch_size]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None # Initial hidden state
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the sequence encoder.

        Args:
            x: Input sequence tensor [batch_size, seq_len, input_dim=4]
               Feature order: [amount, weekday, hour, time_delta]
            lengths: Optional tensor of original sequence lengths [batch_size]
            hidden: Optional initial hidden state for the LSTM

        Returns:
            Tuple of:
            - lstm_out_norm: Normalized output sequence from LSTM [batch_size, seq_len, hidden_dim * num_directions]
            - (h_n, c_n): Final hidden state and cell state from LSTM
            - attn_context: Attention-weighted context vector [batch_size, hidden_dim * num_directions]
        """
        batch_size, seq_len, input_dim_actual = x.shape

        # --- Input Validation ---
        if input_dim_actual != self.input_dim:
            raise ValueError(
                f"SequenceEncoder forward received input with dim {input_dim_actual}, "
                f"but expected {self.input_dim} based on initialization."
            )

        # --- Length Validation and Preparation ---
        if lengths is None:
            # Assume all sequences have the maximum length if lengths are not provided
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)
        else:
            lengths = lengths.to(x.device) # Ensure lengths are on the same device as input

            # 1. Clamp negative lengths (should not happen, but safety check)
            invalid_mask = lengths < 0
            if invalid_mask.any():
                print(f"WARNING (SequenceEncoder): Found {invalid_mask.sum().item()} sequences with length < 0. Setting to 0.")
                lengths = lengths.clamp(min=0) # Use clamp for simplicity

            # 2. Clamp lengths exceeding the actual sequence dimension (can happen with padding)
            too_long_mask = lengths > seq_len
            if too_long_mask.any():
                print(f"WARNING (SequenceEncoder): Found {too_long_mask.sum().item()} sequences with length > seq_len={seq_len}. Clamping.")
                lengths = lengths.clamp(max=seq_len)

        # --- Feature Processing ---
        # Separate features based on expected DataModule output order
        amount = x[..., 0:1]            # Shape: [batch_size, seq_len, 1]
        # Other features (weekday, hour, time_delta)
        other_features = x[..., 1:]     # Shape: [batch_size, seq_len, input_dim-1]

        # Normalize the amount feature
        normed_amount = self.amount_norm(amount)

        # Optional: Normalize other features if needed
        # normed_other_features = self.other_feature_norm(other_features)
        # processed_x = torch.cat([normed_amount, normed_other_features], dim=-1)

        # Use original other features for now
        processed_x = torch.cat([normed_amount, other_features], dim=-1)

        # Verify processed shape
        if processed_x.shape[-1] != self.lstm_input_size:
             raise RuntimeError(
                 f"Internal Error: Processed sequence feature dim {processed_x.shape[-1]} "
                 f"!= expected LSTM input_size {self.lstm_input_size}"
             )
        attention_mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)

        # --- Prepare for RNN Packing ---
        # Move lengths to CPU as required by pack_padded_sequence
        lengths_cpu = lengths.cpu()

        lengths_for_packing = lengths_cpu.clone()
        lengths_for_packing[lengths_for_packing == 0] = 1
        if not torch.all(lengths_for_packing > 0):
             raise RuntimeError("Internal Error: Lengths passed to pack_padded_sequence are not all > 0 after clamping.")
        
        # Pack sequences (handles varying lengths efficiently for RNN)
        # `pack_padded_sequence` correctly handles sequences with length 0.
        # packed_x = nn.utils.rnn.pack_padded_sequence(
        #     processed_x, lengths_cpu, batch_first=True, enforce_sorted=False
        # )
        packed_x = nn.utils.rnn.pack_padded_sequence(
            processed_x, lengths_for_packing, batch_first=True, enforce_sorted=False # Use lengths_for_packing
        )
        # --- Apply LSTM ---
        # packed_output: PackedSequence object containing outputs for each time step
        # (h_n, c_n): Final hidden and cell states for each layer/direction
        packed_output, (h_n, c_n) = self.lstm(packed_x, hidden)

        # --- Unpack Sequences ---
        # Unpack the output sequence back into a padded tensor
        # lstm_out shape: [batch_size, seq_len, hidden_dim * num_directions]
        # `total_length=seq_len` ensures output matches original padded length
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )

        # --- Post-processing ---
        # Apply Layer Normalization to the LSTM output sequence
        lstm_out_norm = self.layer_norm(lstm_out)

        # Apply Dropout
        lstm_out_drop = self.dropout(lstm_out_norm)

        # --- Attention Pooling ---
        # Create boolean mask for attention based on original lengths
        # Shape: [batch_size, seq_len]. True for valid steps, False for padding.
        #attention_mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        print(f"DEBUG (SequenceEncoder): lstm_out shape: {lstm_out_drop.shape}, mask shape: {attention_mask.shape}")
        if lstm_out_drop.shape[:2] != attention_mask.shape:
             print(f"CRITICAL WARNING: Mismatch between lstm_out seq len {lstm_out_drop.shape[1]} and mask seq len {attention_mask.shape[1]} before attention!")
             # Recreate mask based on lstm_out shape if mismatch occurs (investigate root cause)
             attention_mask = torch.arange(lstm_out_drop.shape[1], device=x.device).expand(lstm_out_drop.shape[0], lstm_out_drop.shape[1]) < lengths.unsqueeze(1) # Use original lengths here
             print(f"Recreated mask with shape: {attention_mask.shape}")
        # Apply attention mechanism using the dropout-applied LSTM output and the mask
        # attn_context shape: [batch_size, hidden_dim * num_directions]
        attn_context = self.attention(lstm_out_drop, mask=attention_mask)

        # Return the normalized LSTM output sequence (useful for some downstream tasks),
        # the final hidden/cell states, and the attention context vector.
        return lstm_out_norm, (h_n, c_n), attn_context

    def get_last_hidden(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Extract the final hidden state from the LSTM's output."""
        # h_n shape: [num_layers * num_directions, batch_size, hidden_dim]
        h_n, _ = hidden
        if self.bidirectional:
            # Concatenate the last hidden state from the forward direction (h_n[-2])
            # and the last hidden state from the backward direction (h_n[-1])
            last_hidden = torch.cat([h_n[-2,:,:], h_n[-1,:,:]], dim=-1)
            # Shape: [batch_size, hidden_dim * 2]
        else:
            # Get the last hidden state from the single direction last layer
            last_hidden = h_n[-1,:,:]
            # Shape: [batch_size, hidden_dim]
        return last_hidden

# --- SequencePredictor (Wrapper for classification - likely unused by TransactionClassifier) ---
class SequencePredictor(nn.Module):
    """Sequence-based predictor using SequenceEncoder and a classification head."""
    def __init__(
        self,
        encoder: SequenceEncoder,
        num_classes: int,
        hidden_dim: int, # Should match encoder's hidden_dim
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = encoder

        # Determine the input dimension for the classifier based on encoder output
        encoder_output_dim = hidden_dim * (2 if encoder.bidirectional else 1)

        # Simple classification head
        self.classifier = nn.Sequential(
            # Optional: Add another layer for more capacity
            # nn.Linear(encoder_output_dim, encoder_output_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(encoder_output_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass for prediction.

        Args:
            x: Input sequence tensor [batch_size, seq_len, input_dim]
            lengths: Optional sequence lengths [batch_size]
            hidden: Optional initial hidden state

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Encode the sequence using the encoder
        # We primarily need the attention context vector for classification here
        _, _, attn_context = self.encoder(x, lengths, hidden)

        # Classify using the attention context vector
        logits = self.classifier(attn_context)
        return logits



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple

# class SequenceEncoder(nn.Module):
#     """LSTM-based sequence encoder for transaction history."""
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         num_layers: int = 2,
#         dropout: float = 0.2,
#         bidirectional: bool = False
#     ):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
        
#         if input_dim != 4:
#             print(f"[WARN] SequenceEncoder initialized with input_dim={input_dim}, but expected 4 based on DataModule.")
#         # Adjust expected_lstm_input_size accordingly or raise error if critical
        
#         # Time feature processing
#         # self.time_proj = nn.Sequential(
#         #     nn.Linear(6, 6),  # Project time features (sin/cos of hour, day, weekday)
#         #     nn.ReLU(),
#         #     nn.Linear(6, 3)   # Project back to original dim
#         # )
        
#         # Time delta embedding
#         # self.time_delta_encoder = nn.Sequential(
#         #     nn.Linear(1, 8),  # Embed time delta
#         #     nn.ReLU(),
#         #     nn.Linear(8, 4)   # Project to feature dimension
#         # )
        
#         # Amount normalization
#         self.amount_norm = nn.LayerNorm(1)
#         self.other_feature_norm = nn.LayerNorm(3)
#         self.lstm_input_size = input_dim
#         # LSTM layers - adjusted input_size to include time delta features
#         self.lstm = nn.LSTM(
#             input_size=input_dim + 4,  # +4 for time delta features
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0,
#             bidirectional=bidirectional
#         )
        
#         # Layer normalization
#         self.layer_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
        
#         # Attention mechanism for weighted pooling over sequence
#         self.attention = TemporalAttention(hidden_dim * (2 if bidirectional else 1))
        
#     def _encode_time_features(self, time_features: torch.Tensor) -> torch.Tensor:
#         """
#         Encode time features using cyclical encoding.
        
#         Args:
#             time_features: Tensor of shape [batch_size, seq_len, n] containing time features
            
#         Returns:
#             Encoded time features [batch_size, seq_len, 3]
#         """
#         # Check the shape of time features
#         if time_features.size(-1) < 3:
#             # If we only have two time features (e.g., day and hour), add a placeholder for weekday
#             # Create a zero tensor for the missing feature with the same shape as other features
#             batch_size, seq_len = time_features.shape[0], time_features.shape[1]
#             placeholder = torch.zeros((batch_size, seq_len), device=time_features.device)
            
#             # Extract available components
#             hour = time_features[..., 0]  # [batch_size, seq_len]
#             day = time_features[..., 1]
            
#             # Use placeholder for the missing feature
#             weekday = placeholder
#         else:
#             # Extract components as before
#             hour = time_features[..., 0]  # [batch_size, seq_len]
#             day = time_features[..., 1]
#             weekday = time_features[..., 2]
        
#         # Cyclical encoding
#         hour_sin = torch.sin(2 * torch.pi * hour / 24)
#         hour_cos = torch.cos(2 * torch.pi * hour / 24)
#         day_sin = torch.sin(2 * torch.pi * day / 31)
#         day_cos = torch.cos(2 * torch.pi * day / 31)
#         weekday_sin = torch.sin(2 * torch.pi * weekday / 7)
#         weekday_cos = torch.cos(2 * torch.pi * weekday / 7)
        
#         # Stack encoded features
#         encoded = torch.stack([
#             hour_sin, hour_cos,
#             day_sin, day_cos,
#             weekday_sin, weekday_cos
#         ], dim=-1)  # [batch_size, seq_len, 6]
        
#         # Project back to original dimension
#         encoded = self.time_proj(encoded)  # [batch_size, seq_len, 3]
        
#         return encoded
    
#     # def _compute_time_deltas(self, timestamps: torch.Tensor) -> torch.Tensor:
#     #     """
#     #     Compute time deltas between consecutive transactions.
        
#     #     Args:
#     #         timestamps: Tensor of shape [batch_size, seq_len] containing timestamps
            
#     #     Returns:
#     #         Time deltas [batch_size, seq_len, 1] with first position set to 0
#     #     """
#     #     # Get timestamps
#     #     # Shift to get previous timestamp (padding first with the same timestamp)
#     #     padded = torch.cat([timestamps[:, 0:1], timestamps[:, :-1]], dim=1)
        
#     #     # Compute time deltas in hours
#     #     time_deltas = (timestamps - padded) / 3600  # Assuming timestamps are in seconds
        
#     #     # Set first position to 0 (no previous transaction)
#     #     time_deltas[:, 0] = 0
        
#     #     # Ensure positive deltas
#     #     time_deltas = torch.abs(time_deltas)
        
#     #     # Apply log1p to handle large time differences better
#     #     time_deltas = torch.log1p(time_deltas)
        
#     #     # Normalize with mean and std to prevent extreme values
#     #     mean = time_deltas.mean()
#     #     std = time_deltas.std() + 1e-6  # Add small epsilon to prevent division by zero
#     #     time_deltas = (time_deltas - mean) / std
        
#     #     # Add channel dimension
#     #     return time_deltas.unsqueeze(-1)
        
#     def forward(
#         self,
#         x: torch.Tensor,
#         lengths: Optional[torch.Tensor] = None,
#         hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
#     ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
#         """
#         Forward pass of the sequence encoder.
        
#         Args:
#             x: Input sequence tensor [batch_size, seq_len, input_dim]
#             lengths: Optional sequence lengths [batch_size]
#             hidden: Optional initial hidden state
            
#         Returns:
#             Tuple of:
#             - Output sequence [batch_size, seq_len, hidden_dim * (2 if bidirectional)]
#             - Final hidden state (h_n, c_n)
#             - Attention-weighted representation [batch_size, hidden_dim * (2 if bidirectional)]
#         """
#         batch_size, seq_len, _ = x.shape
        
#         # ADDED: Validate sequence lengths early
#         if lengths is not None:
#             # Ensure lengths are on the correct device
#             lengths = lengths.to(x.device)
            
#             # Check and fix invalid lengths
#             invalid_mask = lengths <= 0
#             if invalid_mask.any():
#                 num_invalid = invalid_mask.sum().item()
#                 print(f"WARNING (SequenceEncoder): Found {num_invalid} sequences with length < 0. Setting to 0.")
#                 lengths = lengths.clone() # Avoid modifying original
#                 lengths[invalid_mask] = 0

#             # Clamp lengths > seq_len
#             too_long_mask = lengths > seq_len
#             if too_long_mask.any():
#                 num_too_long = too_long_mask.sum().item()
#                 print(f"WARNING (SequenceEncoder): Found {num_too_long} sequences with length > seq_len={seq_len}. Clamping.")
#                 if not lengths.is_contiguous(): lengths = lengths.contiguous() # Ensure contiguous before inplace op
#                 lengths[too_long_mask] = seq_len
#             # if invalid_mask.any():
#             #     num_invalid = invalid_mask.sum().item()
#             #     print(f"WARNING: Found {num_invalid} sequences with length <= 0. Setting to length 1.")
                
#             #     # Clone to avoid modifying the original tensor
#             #     lengths = lengths.clone()
                
#             #     # Fix invalid lengths
#             #     lengths[invalid_mask] = 1
            
#             # # Ensure lengths don't exceed sequence length
#             # too_long_mask = lengths > seq_len
#             # if too_long_mask.any():
#             #     num_too_long = too_long_mask.sum().item()
#             #     print(f"WARNING: Found {num_too_long} sequences with length > max_len. Clamping to {seq_len}.")
#             #     lengths[too_long_mask] = seq_len
#         else:
#             # If no lengths provided, use full sequence length
#             lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)
#         # Split input into amount, timestamp, and other time features
#         amount = x[..., 0:1]                  # [batch_size, seq_len, 1]
#         timestamps = x[..., 1]                # [batch_size, seq_len]
#         time_features = x[..., 2:]            # [batch_size, seq_len, input_dim-2]
        
#         normed_amount = self.amount_norm(amount)

#         # Compute time deltas
#         time_deltas = self._compute_time_deltas(timestamps)
        
#         # Encode time deltas
#         time_delta_features = self.time_delta_encoder(time_deltas)
        
#         # Normalize amount
#         amount = self.amount_norm(amount)
        
#         # Encode time features
#         time_encoded = self._encode_time_features(time_features)
        
#         # Combine features with time delta information
#         x = torch.cat([amount, time_encoded, time_delta_features], dim=-1)
        
#         # Create mask for attention
#         # Create mask [batch_size, seq_len] where 1 indicates valid positions
#         mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
#         for i, length in enumerate(lengths):
#             mask[i, :length] = 1
        
#         # CHANGED: Move lengths to CPU for packing with a final validation check
#         lengths_cpu = lengths.cpu()
        
#         # Final validation check right before packing
#         if (lengths_cpu <= 0).any():
#             print("CRITICAL: Still found invalid lengths before packing. Fixing again.")
#             lengths_cpu = torch.clamp(lengths_cpu, min=1)
        
#         # Pack sequences with validated lengths
#         x = nn.utils.rnn.pack_padded_sequence(
#             x, lengths_cpu, batch_first=True, enforce_sorted=False
#         )
        
#         # Apply LSTM
#         lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
#         # Unpack sequences if they were packed
#         lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
#         # Layer normalization
#         lstm_out = self.layer_norm(lstm_out)
        
#         # Dropout
#         lstm_out = self.dropout(lstm_out)
        
#         # Apply attention to get context vector
#         attn_out = self.attention(lstm_out, mask)
        
#         return lstm_out, (h_n, c_n), attn_out
    
#     def get_last_hidden(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         """Get the last hidden state."""
#         h_n, _ = hidden
#         if self.bidirectional:
#             # Concatenate forward and backward last hidden states
#             last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
#         else:
#             last_hidden = h_n[-1]
#         return last_hidden

# class SequencePredictor(nn.Module):
#     """Sequence-based predictor for transaction classification."""
#     def __init__(
#         self,
#         encoder: SequenceEncoder,
#         num_classes: int,
#         hidden_dim: int,
#         dropout: float = 0.2
#     ):
#         super().__init__()
#         self.encoder = encoder
        
#         # Output dimension depends on whether LSTM is bidirectional
#         out_dim = hidden_dim * (2 if encoder.bidirectional else 1)
        
#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(out_dim, out_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, num_classes)
#         )
        
#     def forward(
#         self,
#         x: torch.Tensor,
#         lengths: Optional[torch.Tensor] = None,
#         hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
#     ) -> torch.Tensor:
#         """
#         Forward pass of the sequence predictor.
        
#         Args:
#             x: Input sequence tensor [batch_size, seq_len, input_dim]
#             lengths: Optional sequence lengths [batch_size]
#             hidden: Optional initial hidden state
            
#         Returns:
#             Class logits [batch_size, num_classes]
#         """
#         # Get sequence embeddings
#         out, hidden, _ = self.encoder(x, lengths, hidden)
        
#         # Get last hidden state
#         last_hidden = self.encoder.get_last_hidden(hidden)
        
#         # Classify
#         logits = self.classifier(last_hidden)
#         return logits

# class TemporalAttention(nn.Module):
#     """Attention mechanism for temporal sequence modeling."""
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )
#     def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Apply attention to sequence.
        
#         Args:
#             x: Sequence tensor [batch_size, seq_len, hidden_dim]
#             mask: Optional attention mask [batch_size, seq_len]
            
#         Returns:
#             Weighted sum of sequence [batch_size, hidden_dim]
#         """
#         # Compute attention scores
#         scores = self.attention(x)  # [batch_size, seq_len, 1]
        
#         # Apply mask if provided
#         if mask is not None:
#             # ADDED: Fix mask dimension mismatch
#             # if mask.shape[1] != x.shape[1]:
#             #     print(f"WARNING: Mask shape {mask.shape} doesn't match input shape {x.shape}. Resizing mask.")
                
#             #     # Create a new mask matching the sequence length of x
#             #     new_mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
                
#             #     # Copy valid positions from old mask, up to the minimum length
#             #     min_len = min(mask.shape[1], x.shape[1])
#             #     new_mask[:, :min_len] = mask[:, :min_len]
                
#             #     # Use the resized mask
#             #     mask = new_mask
#             if mask.shape != scores.shape[:-1]:
#                  raise ValueError(f"Mask shape {mask.shape} incompatible with scores shape {scores.shape[:-1]}")
            
#             scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
#         # Softmax
#         attn = F.softmax(scores, dim=1)
        
#         # Weighted sum
#         out = torch.sum(attn * x, dim=1)
#         return out        
    # def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     """
    #     Apply attention to sequence.
        
    #     Args:
    #         x: Sequence tensor [batch_size, seq_len, hidden_dim]
    #         mask: Optional attention mask [batch_size, seq_len]
            
    #     Returns:
    #         Weighted sum of sequence [batch_size, hidden_dim]
    #     """
    #     # Compute attention scores
    #     scores = self.attention(x)  # [batch_size, seq_len, 1]
        
    #     # Apply mask if provided
    #     if mask is not None:
    #         scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
    #     # Softmax
    #     attn = F.softmax(scores, dim=1)
        
    #     # Weighted sum
    #     out = torch.sum(attn * x, dim=1)
    #     return out 