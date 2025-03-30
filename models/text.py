import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd

class TextEncoder(nn.Module):
    """Text encoder using pretrained transformer model."""
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128,
        dropout: float = 0.1,
        freeze_layers: Optional[List[int]] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.dropout = dropout
        
        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Store hidden dimension
        self.hidden_dim = self.model.config.hidden_size
        
        # Freeze specified layers
        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in self.model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)
    
    def tokenize(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input texts."""
        # Replace empty strings with a space to avoid tokenizer errors
        texts = [text if text.strip() else " " for text in texts]
        
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = self.dropout(embeddings)
        
        return embeddings

class TextPredictor(nn.Module):
    """Text-based predictor for transaction classification."""
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_layers: Optional[List[int]] = None
    ):
        super().__init__()
        self.encoder = TextEncoder(
            model_name=model_name,
            max_length=max_length,
            dropout=dropout,
            freeze_layers=freeze_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.classifier(embeddings)
        return logits

class MultiFieldTextEncoder(nn.Module):
    """Text encoder for multiple text fields using a pretrained transformer."""
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        field_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.max_length = max_length
        self.field_weights = field_weights or {}
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze transformer parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Project BERT hidden dim to our hidden dim
        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Field attention
        self.field_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, text_fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the text encoder.
        
        Args:
            text_fields: Dictionary of text fields, each with shape [batch_size, max_length]
            
        Returns:
            Text embeddings [batch_size, hidden_dim]
        """
        device = next(self.parameters()).device
        field_embeddings = []
        field_weights = []
        
        for field_name, field_tokens in text_fields.items():
            # Skip empty fields
            if field_tokens is None:
                continue
                
            # Move tokens to device
            field_tokens = field_tokens.to(device)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.model(field_tokens)
                embeddings = outputs.last_hidden_state[:, 0]  # Use [CLS] token
            
            # Project to hidden dim
            embeddings = self.projection(embeddings)
            field_embeddings.append(embeddings)
            
            # Get field weight
            weight = self.field_weights.get(field_name, 1.0)
            field_weights.append(weight)
            
        # Return zero tensor if no valid fields
        if not field_embeddings:
            return torch.zeros(text_fields[list(text_fields.keys())[0]].size(0), self.hidden_dim, device=device)
            
        # Stack field embeddings
        field_embeddings = torch.stack(field_embeddings, dim=1)  # [batch_size, num_fields, hidden_dim]
        field_weights = torch.tensor(field_weights, device=device).view(1, -1, 1)  # [1, num_fields, 1]
        
        # Apply field attention with weights
        attention_scores = self.field_attention(field_embeddings)  # [batch_size, num_fields, 1]
        attention_weights = F.softmax(attention_scores * field_weights, dim=1)  # [batch_size, num_fields, 1]
        attended_embeddings = torch.sum(attention_weights * field_embeddings, dim=1)  # [batch_size, hidden_dim]
        
        return attended_embeddings 