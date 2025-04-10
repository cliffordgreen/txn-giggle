import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List

class FinBERTEmbedder(nn.Module):
    def __init__(self, 
                 model_name: str="yiyanghkust/finbert-tone", 
                 pooling_strategy: str="mean", 
                 finetune: bool=True,
                 projection_dim: int = 0 # Optional: Project output to this dim if > 0
                 ):
        """
        Initializes the FinBERT embedder.

        Args:
            model_name (str): Name of the FinBERT model on Hugging Face Hub.
            pooling_strategy (str): 'cls' or 'mean'.
            finetune (bool): Whether to allow fine-tuning of BERT parameters.
            projection_dim (int): If > 0, add a linear layer to project the output embedding.
        """
        super().__init__()
        print(f"Initializing FinBERTEmbedder with model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f"[ERROR] Failed to load model/tokenizer '{model_name}' from Hugging Face Hub: {e}")
            print("Make sure the model name is correct and you have an internet connection.")
            raise e
            
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in ['cls', 'mean']:
            raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}. Choose 'cls' or 'mean'.")
        
        self.finetune = finetune
        if not self.finetune:
            print("Freezing FinBERT parameters.")
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Optional output projection
        self.projection = None
        bert_hidden_dim = self.model.config.hidden_size
        self.output_dim = bert_hidden_dim # Default output dim
        if projection_dim > 0 and projection_dim != bert_hidden_dim:
             print(f"Adding projection layer from {bert_hidden_dim} to {projection_dim}")
             self.projection = nn.Linear(bert_hidden_dim, projection_dim)
             self.output_dim = projection_dim # Update output dim
        elif projection_dim > 0 and projection_dim == bert_hidden_dim:
             print(f"Projection dim matches BERT hidden dim ({bert_hidden_dim}), projection layer skipped.")

    def forward(self, text_batch: List[str]) -> torch.Tensor:
        """
        Encodes a batch of text strings.

        Args:
            text_batch (list[str]): A list or batch of text strings.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, output_dim] containing text embeddings.
        """
        # Determine device from model parameters
        device = next(self.parameters()).device

        # Replace None or empty strings with a placeholder
        processed_text_batch = [t if isinstance(t, str) and t.strip() else "[PAD]" for t in text_batch] # Use [PAD] token
        
        inputs = self.tokenizer(processed_text_batch, padding=True, truncation=True,
                                return_tensors="pt", max_length=512).to(device)

        # Pass inputs through the model
        forward_context = torch.no_grad() if (not self.finetune and not self.training) else torch.enable_grad()
        
        with forward_context:
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state

        if self.pooling_strategy == "cls":
            embedding = last_hidden_state[:, 0, :] 
        elif self.pooling_strategy == "mean":
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        
        if self.projection:
             embedding = self.projection(embedding)

        return embedding

    def get_output_dim(self):
        return self.output_dim 