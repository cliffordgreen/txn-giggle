import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Dict

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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
            self.model = AutoModel.from_pretrained(model_name, force_download=True)
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

    def forward(self, text_input: Union[List[str], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Encodes a batch of text strings or pre-tokenized input.

        Args:
            text_input (Union[List[str], Dict[str, torch.Tensor]]):
                - A list or batch of text strings.
                - OR a dictionary containing 'input_ids' and 'attention_mask' tensors.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, output_dim] containing text embeddings.
        """
        # Determine device from model parameters
        device = next(self.parameters()).device

        if isinstance(text_input, list):
            # Input is a list of strings, tokenize it
            processed_text_batch = [t if isinstance(t, str) and t.strip() else "[PAD]" for t in text_input] # Use [PAD] token
            inputs = self.tokenizer(processed_text_batch, padding=True, truncation=True,
                                    return_tensors="pt", max_length=512).to(device)
        elif isinstance(text_input, dict) and 'input_ids' in text_input and 'attention_mask' in text_input:
            # Input is pre-tokenized
            inputs = {
                'input_ids': text_input['input_ids'].to(device),
                'attention_mask': text_input['attention_mask'].to(device)
            }
            # Potentially add token_type_ids if your model uses them and they are provided
            if 'token_type_ids' in text_input:
                inputs['token_type_ids'] = text_input['token_type_ids'].to(device)
        else:
            raise ValueError("Invalid input type for FinBERTEmbedder. Expected List[str] or Dict with 'input_ids' and 'attention_mask'.")

        # Pass inputs through the model
        # The context manager for no_grad should depend on self.training, not self.finetune.
        # If self.finetune is False, parameters have requires_grad=False, so grads won't be computed anyway.
        # If self.finetune is True, we want grads during training.
        if not self.finetune:
            # If not fine-tuning, ensure all model parameters do not require gradients.
            # This is usually set in __init__, but double-check or enforce here if necessary.
            # The primary control for gradient calculation is torch.set_grad_enabled(), 
            # which is implicitly handled by model.train() and model.eval().
            # Forcing no_grad() here if not finetuning AND not training might be overly restrictive if part of a larger model that IS training.
            # It's better to rely on param.requires_grad set in __init__ and the model's training state.
            pass # Relies on requires_grad being False from __init__ if not finetune

        # Control gradient computation based on model's training state if fine-tuning
        # If not fine-tuning, grads are already disabled for BERT params by requires_grad=False
        # The self.training flag is set by pl.LightningModule.train() or .eval()
        if self.finetune and self.training:
            context_manager = torch.enable_grad()
        else:
            # No grads if not fine-tuning, or if fine-tuning but in eval mode
            context_manager = torch.no_grad()
        
        with context_manager:
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