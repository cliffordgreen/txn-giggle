import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple

class MultiFieldTextEncoder(nn.Module):
    """Multi-field text encoder with field-specific weights and concatenation."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 128):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        
        # Initialize BERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Field-specific weights
        self.field_weights = nn.ParameterDict({
            'merchant_name': nn.Parameter(torch.ones(1)),
            'description': nn.Parameter(torch.ones(1)),
            'category_name': nn.Parameter(torch.ones(1))
        })
        
        # Field-specific projections
        self.field_projections = nn.ModuleDict({
            field: nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
            for field in self.field_weights.keys()
        })
        
        # Final projection
        self.final_projection = nn.Linear(
            self.encoder.config.hidden_size * len(self.field_weights),
            self.encoder.config.hidden_size
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.encoder.config.hidden_size)
        
    def _tokenize_field(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text field."""
        # Handle empty strings
        if not text or text.isspace():
            text = "[UNK]"
            
        # Tokenize with special tokens
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return tokens
    
    def forward(self, text_inputs: Dict[str, List[str]]) -> torch.Tensor:
        """Forward pass with field-specific processing."""
        field_embeddings = []
        
        # Process each field
        for field, texts in text_inputs.items():
            if field not in self.field_weights:
                continue
                
            # Tokenize field
            tokens = self._tokenize_field(texts[0])  # Assuming batch size 1 for now
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.encoder(**tokens)
                embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            
            # Apply field-specific projection
            embeddings = self.field_projections[field](embeddings)
            
            # Apply field-specific weight
            weight = F.softplus(self.field_weights[field])  # Ensure positive weights
            embeddings = embeddings * weight
            
            field_embeddings.append(embeddings)
        
        # Concatenate field embeddings
        if not field_embeddings:
            # Handle case where no fields are available
            return torch.zeros(1, self.encoder.config.hidden_size, device=next(self.parameters()).device)
            
        combined = torch.cat(field_embeddings, dim=-1)
        
        # Final projection and normalization
        output = self.final_projection(combined)
        output = self.layer_norm(output)
        
        return output 