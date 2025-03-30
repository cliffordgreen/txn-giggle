import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SequenceEncoder(nn.Module):
    """LSTM-based sequence encoder for transaction history."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Time feature processing
        self.time_proj = nn.Sequential(
            nn.Linear(6, 6),  # Project time features (sin/cos of hour, day, weekday)
            nn.ReLU(),
            nn.Linear(6, 3)   # Project back to original dim
        )
        
        # Time delta embedding
        self.time_delta_encoder = nn.Sequential(
            nn.Linear(1, 8),  # Embed time delta
            nn.ReLU(),
            nn.Linear(8, 4)   # Project to feature dimension
        )
        
        # Amount normalization
        self.amount_norm = nn.LayerNorm(1)
        
        # LSTM layers - adjusted input_size to include time delta features
        self.lstm = nn.LSTM(
            input_size=input_dim + 4,  # +4 for time delta features
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism for weighted pooling over sequence
        self.attention = TemporalAttention(hidden_dim * (2 if bidirectional else 1))
        
    def _encode_time_features(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Encode time features using cyclical encoding.
        
        Args:
            time_features: Tensor of shape [batch_size, seq_len, n] containing time features
            
        Returns:
            Encoded time features [batch_size, seq_len, 3]
        """
        # Check the shape of time features
        if time_features.size(-1) < 3:
            # If we only have two time features (e.g., day and hour), add a placeholder for weekday
            # Create a zero tensor for the missing feature with the same shape as other features
            batch_size, seq_len = time_features.shape[0], time_features.shape[1]
            placeholder = torch.zeros((batch_size, seq_len), device=time_features.device)
            
            # Extract available components
            hour = time_features[..., 0]  # [batch_size, seq_len]
            day = time_features[..., 1]
            
            # Use placeholder for the missing feature
            weekday = placeholder
        else:
            # Extract components as before
            hour = time_features[..., 0]  # [batch_size, seq_len]
            day = time_features[..., 1]
            weekday = time_features[..., 2]
        
        # Cyclical encoding
        hour_sin = torch.sin(2 * torch.pi * hour / 24)
        hour_cos = torch.cos(2 * torch.pi * hour / 24)
        day_sin = torch.sin(2 * torch.pi * day / 31)
        day_cos = torch.cos(2 * torch.pi * day / 31)
        weekday_sin = torch.sin(2 * torch.pi * weekday / 7)
        weekday_cos = torch.cos(2 * torch.pi * weekday / 7)
        
        # Stack encoded features
        encoded = torch.stack([
            hour_sin, hour_cos,
            day_sin, day_cos,
            weekday_sin, weekday_cos
        ], dim=-1)  # [batch_size, seq_len, 6]
        
        # Project back to original dimension
        encoded = self.time_proj(encoded)  # [batch_size, seq_len, 3]
        
        return encoded
    
    def _compute_time_deltas(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute time deltas between consecutive transactions.
        
        Args:
            timestamps: Tensor of shape [batch_size, seq_len] containing timestamps
            
        Returns:
            Time deltas [batch_size, seq_len, 1] with first position set to 0
        """
        # Get timestamps
        # Shift to get previous timestamp (padding first with the same timestamp)
        padded = torch.cat([timestamps[:, 0:1], timestamps[:, :-1]], dim=1)
        
        # Compute time deltas in hours
        time_deltas = (timestamps - padded) / 3600  # Assuming timestamps are in seconds
        
        # Set first position to 0 (no previous transaction)
        time_deltas[:, 0] = 0
        
        # Ensure positive deltas
        time_deltas = torch.abs(time_deltas)
        
        # Apply log1p to handle large time differences better
        time_deltas = torch.log1p(time_deltas)
        
        # Normalize with mean and std to prevent extreme values
        mean = time_deltas.mean()
        std = time_deltas.std() + 1e-6  # Add small epsilon to prevent division by zero
        time_deltas = (time_deltas - mean) / std
        
        # Add channel dimension
        return time_deltas.unsqueeze(-1)
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the sequence encoder.
        
        Args:
            x: Input sequence tensor [batch_size, seq_len, input_dim]
            lengths: Optional sequence lengths [batch_size]
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of:
            - Output sequence [batch_size, seq_len, hidden_dim * (2 if bidirectional)]
            - Final hidden state (h_n, c_n)
            - Attention-weighted representation [batch_size, hidden_dim * (2 if bidirectional)]
        """
        # Split input into amount, timestamp, and other time features
        amount = x[..., 0:1]                  # [batch_size, seq_len, 1]
        timestamps = x[..., 1]                # [batch_size, seq_len]
        time_features = x[..., 2:]            # [batch_size, seq_len, input_dim-2]
        
        # Compute time deltas
        time_deltas = self._compute_time_deltas(timestamps)
        
        # Encode time deltas
        time_delta_features = self.time_delta_encoder(time_deltas)
        
        # Normalize amount
        amount = self.amount_norm(amount)
        
        # Encode time features
        time_encoded = self._encode_time_features(time_features)
        
        # Combine features with time delta information
        x = torch.cat([amount, time_encoded, time_delta_features], dim=-1)
        
        # Create mask for attention if lengths provided
        if lengths is not None:
            # Create mask [batch_size, seq_len] where 1 indicates valid positions
            mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1
        else:
            mask = None
        
        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Apply LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Unpack sequences if they were packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Apply attention to get context vector
        attn_out = self.attention(lstm_out, mask)
        
        return lstm_out, (h_n, c_n), attn_out
    
    def get_last_hidden(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Get the last hidden state."""
        h_n, _ = hidden
        if self.bidirectional:
            # Concatenate forward and backward last hidden states
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_hidden = h_n[-1]
        return last_hidden

class SequencePredictor(nn.Module):
    """Sequence-based predictor for transaction classification."""
    def __init__(
        self,
        encoder: SequenceEncoder,
        num_classes: int,
        hidden_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = encoder
        
        # Output dimension depends on whether LSTM is bidirectional
        out_dim = hidden_dim * (2 if encoder.bidirectional else 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass of the sequence predictor.
        
        Args:
            x: Input sequence tensor [batch_size, seq_len, input_dim]
            lengths: Optional sequence lengths [batch_size]
            hidden: Optional initial hidden state
            
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Get sequence embeddings
        out, hidden, _ = self.encoder(x, lengths, hidden)
        
        # Get last hidden state
        last_hidden = self.encoder.get_last_hidden(hidden)
        
        # Classify
        logits = self.classifier(last_hidden)
        return logits

class TemporalAttention(nn.Module):
    """Attention mechanism for temporal sequence modeling."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention to sequence.
        
        Args:
            x: Sequence tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Weighted sum of sequence [batch_size, hidden_dim]
        """
        # Compute attention scores
        scores = self.attention(x)  # [batch_size, seq_len, 1]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=1)
        
        # Weighted sum
        out = torch.sum(attn * x, dim=1)
        return out 