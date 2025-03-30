import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
import pytorch_lightning as pl
from models.text import MultiFieldTextEncoder

class TransactionDataset(Dataset):
    """Dataset for transaction classification."""
    def __init__(
        self,
        transactions_df: pd.DataFrame,
        graph_data: HeteroData,
        text_encoder: 'MultiFieldTextEncoder',
        seq_data: Dict[int, torch.Tensor],
        seq_lengths: Dict[int, int],
        indices: List[int]
    ):
        self.transactions_df = transactions_df
        self.graph_data = graph_data
        self.text_encoder = text_encoder
        self.seq_data = seq_data
        self.seq_lengths = seq_lengths
        self.indices = indices
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single transaction sample."""
        # Get transaction index
        tx_idx = self.indices[idx]
        
        # Get transaction data
        tx = self.transactions_df.iloc[tx_idx]
        
        # Get sequence data
        user_id = tx['user_id']
        seq = self.seq_data[user_id]
        seq_len = self.seq_lengths[user_id]
        
        # Get text inputs
        text_inputs = {
            'description': str(tx['description']) if pd.notna(tx['description']) else '',
            'memo': str(tx['memo']) if pd.notna(tx['memo']) else '',
            'merchant': str(tx['merchant_name']) if pd.notna(tx['merchant_name']) else ''
        }
        
        # Get labels
        labels = {
            'global': torch.tensor(tx['category_id'], dtype=torch.long),
            'user': torch.tensor(tx['user_category_id'], dtype=torch.long)
        }
        
        # Get graph data
        graph_data = {
            'x_dict': {
                'transaction': self.graph_data['transaction'].x[tx_idx],
                'merchant': self.graph_data['merchant'].x,
                'category': self.graph_data['category'].x
            },
            'edge_index_dict': {
                edge_type: self.graph_data[edge_type].edge_index
                for edge_type in [
                    ('transaction', 'belongs_to', 'merchant'),
                    ('merchant', 'categorized_as', 'category'),
                    ('transaction', 'temporal', 'transaction'),
                    ('transaction', 'similar_amount', 'transaction')
                ]
            },
            'edge_attr_dict': {
                edge_type: self.graph_data[edge_type].edge_attr
                for edge_type in [
                    ('transaction', 'belongs_to', 'merchant'),
                    ('merchant', 'categorized_as', 'category'),
                    ('transaction', 'temporal', 'transaction'),
                    ('transaction', 'similar_amount', 'transaction')
                ]
            }
        }
        
        return {
            'tx_idx': tx_idx,
            'seq_data': seq,
            'seq_length': seq_len,
            'text_inputs': text_inputs,
            'labels': labels,
            'graph_data': graph_data
        }

class TransactionDataModule(pl.LightningDataModule):
    """Data module for transaction classification."""
    def __init__(
        self,
        transactions_df: pd.DataFrame,
        batch_size: int = 32,
        num_workers: int = 4,
        max_seq_length: int = 50,
        graph_neighbors: Dict[str, int] = None,
        text_model_name: str = 'bert-base-uncased',
        text_max_length: int = 128,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ):
        super().__init__()
        self.transactions_df = transactions_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.graph_neighbors = graph_neighbors or {
            'same_merchant': 10,
            'same_company': 5,
            'similar_amount': 5
        }
        self.text_model_name = text_model_name
        self.text_max_length = text_max_length
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Initialize text encoder for tokenization
        self.text_encoder = None
        self.graph_data = None
        self.seq_data = None
        self.seq_lengths = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup data for training."""
        if stage == 'fit' or stage is None:
            # Initialize text encoder
            self.text_encoder = MultiFieldTextEncoder(
                model_name=self.text_model_name,
                max_length=self.text_max_length
            )
            
            # Build graph
            self._build_graph()
            
            # Prepare sequences
            self._prepare_sequences()
            
            # Split data
            self._split_data()
            
        if stage == 'test' or stage is None:
            # For testing, we need the same setup as training
            if self.text_encoder is None:
                self.setup('fit')
    
    def _build_graph(self):
        """Build heterogeneous transaction graph with multiple node types and edge types."""
        # Create transaction node features with cyclical time encoding
        transaction_features = []
        for _, row in self.transactions_df.iterrows():
            # Cyclical encoding for time features
            hour = row['hour']
            day = row['weekday']
            
            # Hour encoding (24-hour cycle)
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            # Day encoding (7-day cycle)
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)
            
            # Combine features
            features = [
                row['amount'],  # Raw amount
                hour_sin, hour_cos,  # Cyclical hour encoding
                day_sin, day_cos,    # Cyclical day encoding
                row['timestamp'].timestamp()  # Absolute timestamp
            ]
            transaction_features.append(features)
        transaction_features = torch.tensor(transaction_features, dtype=torch.float)
        
        # Create merchant node features (using aggregated statistics)
        merchant_features = {}
        for merchant_name, group in self.transactions_df.groupby('merchant_name'):
            merchant_features[merchant_name] = torch.tensor([
                group['amount'].mean(),
                group['amount'].std(),
                group['amount'].max(),
                group['amount'].min(),
                len(group),  # transaction count
                group['amount'].median(),
                group['amount'].quantile(0.25),
                group['amount'].quantile(0.75)
            ], dtype=torch.float)
        
        # Create category node features (using aggregated statistics)
        category_features = {}
        for category_id, group in self.transactions_df.groupby('category_id'):
            category_features[category_id] = torch.tensor([
                group['amount'].mean(),
                group['amount'].std(),
                group['amount'].max(),
                group['amount'].min(),
                len(group),  # transaction count
                group['amount'].median(),
                group['amount'].quantile(0.25),
                group['amount'].quantile(0.75)
            ], dtype=torch.float)
        
        # Initialize dictionaries for edge indices and attributes
        edge_indices = {}
        edge_attrs = {}
        
        # Transaction -> Merchant edges (belongs_to)
        belongs_to_edges = []
        belongs_to_attrs = []
        for idx, row in self.transactions_df.iterrows():
            belongs_to_edges.append([idx, row['merchant_name']])
            # Enhanced edge features: amount relative to merchant's statistics
            merchant_group = self.transactions_df[self.transactions_df['merchant_name'] == row['merchant_name']]
            amount_mean = merchant_group['amount'].mean()
            amount_std = merchant_group['amount'].std()
            amount_zscore = (row['amount'] - amount_mean) / (amount_std + 1e-6)
            belongs_to_attrs.append([amount_zscore])
        if belongs_to_edges:
            edge_indices[('transaction', 'belongs_to', 'merchant')] = torch.tensor(belongs_to_edges, dtype=torch.long).t()
            edge_attrs[('transaction', 'belongs_to', 'merchant')] = torch.tensor(belongs_to_attrs, dtype=torch.float)
        
        # Merchant -> Category edges (categorized_as)
        categorized_as_edges = []
        categorized_as_attrs = []
        for merchant_name, group in self.transactions_df.groupby('merchant_name'):
            category_id = group['category_id'].iloc[0]  # Most common category
            categorized_as_edges.append([merchant_name, category_id])
            # Enhanced edge features: merchant's category confidence
            category_counts = group['category_id'].value_counts()
            category_confidence = category_counts.iloc[0] / len(group)
            categorized_as_attrs.append([category_confidence])
        if categorized_as_edges:
            edge_indices[('merchant', 'categorized_as', 'category')] = torch.tensor(categorized_as_edges, dtype=torch.long).t()
            edge_attrs[('merchant', 'categorized_as', 'category')] = torch.tensor(categorized_as_attrs, dtype=torch.float)
        
        # Transaction -> Transaction edges (temporal)
        temporal_edges = []
        temporal_attrs = []
        for i in range(len(self.transactions_df) - 1):
            for j in range(i + 1, min(i + 5, len(self.transactions_df))):
                time_diff = abs((self.transactions_df.iloc[i]['timestamp'] - 
                               self.transactions_df.iloc[j]['timestamp']).total_seconds())
                if time_diff <= 86400:  # Within 24 hours
                    temporal_edges.extend([
                        [i, j],
                        [j, i]
                    ])
                    # Enhanced temporal edge features
                    time_diff_hours = time_diff / 3600
                    time_diff_days = time_diff / 86400
                    temporal_attrs.extend([
                        [time_diff_hours, time_diff_days, 1.0],  # Forward edge
                        [time_diff_hours, time_diff_days, 0.0]   # Backward edge
                    ])
        if temporal_edges:
            edge_indices[('transaction', 'temporal', 'transaction')] = torch.tensor(temporal_edges, dtype=torch.long).t()
            edge_attrs[('transaction', 'temporal', 'transaction')] = torch.tensor(temporal_attrs, dtype=torch.float)
        
        # Transaction -> Transaction edges (similar_amount)
        similar_amount_edges = []
        similar_amount_attrs = []
        amounts = self.transactions_df['amount'].values
        for i in range(len(amounts)):
            diffs = np.abs(amounts - amounts[i])
            k_nearest = np.argsort(diffs)[1:self.graph_neighbors['similar_amount'] + 1]
            for j in k_nearest:
                similar_amount_edges.extend([
                    [i, j],
                    [j, i]
                ])
                # Enhanced amount similarity edge features
                amount_ratio = min(amounts[i], amounts[j]) / max(amounts[i], amounts[j])
                amount_diff = diffs[j]
                similar_amount_attrs.extend([
                    [amount_ratio, amount_diff, 1.0],  # Forward edge
                    [amount_ratio, amount_diff, 0.0]   # Backward edge
                ])
        if similar_amount_edges:
            edge_indices[('transaction', 'similar_amount', 'transaction')] = torch.tensor(similar_amount_edges, dtype=torch.long).t()
            edge_attrs[('transaction', 'similar_amount', 'transaction')] = torch.tensor(similar_amount_attrs, dtype=torch.float)
        
        # Create HeteroData object
        data = HeteroData()
        
        # Add node features
        data['transaction'].x = transaction_features
        data['merchant'].x = torch.stack(list(merchant_features.values()))
        data['category'].x = torch.stack(list(category_features.values()))
        
        # Add edge indices and attributes
        for edge_type in edge_indices:
            data[edge_type].edge_index = edge_indices[edge_type]
            data[edge_type].edge_attr = edge_attrs[edge_type]
        
        self.graph_data = data
    
    def _prepare_sequences(self):
        """Prepare sequence data for each user."""
        self.seq_data = {}
        self.seq_lengths = {}
        
        # Group transactions by user
        user_groups = self.transactions_df.groupby('user_id')
        
        for user_id, group in user_groups:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Create sequence features
            seq_features = []
            for _, row in group.iterrows():
                feat = [
                    row['amount'],
                    row['timestamp'].timestamp(),
                    row['weekday'],
                    row['hour']
                ]
                seq_features.append(feat)
            
            # Convert to tensor
            seq = torch.tensor(seq_features, dtype=torch.float)
            
            # Truncate if too long
            if len(seq) > self.max_seq_length:
                seq = seq[-self.max_seq_length:]
            
            self.seq_data[user_id] = seq
            self.seq_lengths[user_id] = len(seq)
    
    def _split_data(self):
        """Split data into train/val/test sets."""
        # Get unique users
        users = self.transactions_df['user_id'].unique()
        
        # Randomly split users
        np.random.shuffle(users)
        n_users = len(users)
        
        n_val = int(n_users * self.val_ratio)
        n_test = int(n_users * self.test_ratio)
        
        val_users = users[:n_val]
        test_users = users[n_val:n_val + n_test]
        train_users = users[n_val + n_test:]
        
        # Get indices for each split
        self.train_indices = self.transactions_df[
            self.transactions_df['user_id'].isin(train_users)
        ].index.tolist()
        
        self.val_indices = self.transactions_df[
            self.transactions_df['user_id'].isin(val_users)
        ].index.tolist()
        
        self.test_indices = self.transactions_df[
            self.transactions_df['user_id'].isin(test_users)
        ].index.tolist()
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        dataset = TransactionDataset(
            transactions_df=self.transactions_df,
            graph_data=self.graph_data,
            text_encoder=self.text_encoder,
            seq_data=self.seq_data,
            seq_lengths=self.seq_lengths,
            indices=self.train_indices
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Use single process for MPS compatibility
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        dataset = TransactionDataset(
            transactions_df=self.transactions_df,
            graph_data=self.graph_data,
            text_encoder=self.text_encoder,
            seq_data=self.seq_data,
            seq_lengths=self.seq_lengths,
            indices=self.val_indices
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Use single process for MPS compatibility
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        dataset = TransactionDataset(
            transactions_df=self.transactions_df,
            graph_data=self.graph_data,
            text_encoder=self.text_encoder,
            seq_data=self.seq_data,
            seq_lengths=self.seq_lengths,
            indices=self.test_indices
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Use single process for MPS compatibility
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Collate function for batching."""
        # Get batch indices
        batch_indices = [item['tx_idx'] for item in batch]
        
        # Get node features for each node type
        node_features = {
            'transaction': self.graph_data['transaction'].x[batch_indices],
            'merchant': self.graph_data['merchant'].x,
            'category': self.graph_data['category'].x
        }
        
        # Get edge indices and attributes for each edge type
        edge_indices = {}
        edge_attrs = {}
        
        # Process temporal edges
        temporal_edges = []
        temporal_attrs = []
        for i, idx in enumerate(batch_indices):
            for j, other_idx in enumerate(batch_indices):
                if i != j:
                    time_diff = abs((self.transactions_df.iloc[idx]['timestamp'] - 
                                   self.transactions_df.iloc[other_idx]['timestamp']).total_seconds())
                    if time_diff <= 86400:  # Within 24 hours
                        temporal_edges.extend([[i, j]])
                        temporal_attrs.extend([[time_diff]])
        if temporal_edges:
            edge_indices[('transaction', 'temporal', 'transaction')] = torch.tensor(temporal_edges, dtype=torch.long).t()
            edge_attrs[('transaction', 'temporal', 'transaction')] = torch.tensor(temporal_attrs, dtype=torch.float)
        
        # Process similar amount edges
        similar_amount_edges = []
        similar_amount_attrs = []
        amounts = node_features['transaction'][:, 0]  # First column is amount
        for i in range(len(amounts)):
            diffs = torch.abs(amounts - amounts[i])
            k_nearest = torch.argsort(diffs)[1:self.graph_neighbors['similar_amount'] + 1]
            for j in k_nearest:
                similar_amount_edges.extend([[i, j]])
                similar_amount_attrs.extend([[diffs[j].item()]])
        if similar_amount_edges:
            edge_indices[('transaction', 'similar_amount', 'transaction')] = torch.tensor(similar_amount_edges, dtype=torch.long).t()
            edge_attrs[('transaction', 'similar_amount', 'transaction')] = torch.tensor(similar_amount_attrs, dtype=torch.float)
        
        # Get max sequence length in batch
        max_seq_len = max(item['seq_length'] for item in batch)
        
        # Pad and stack sequence data
        seq_data = torch.zeros(len(batch), max_seq_len, batch[0]['seq_data'].size(-1))
        seq_lengths = torch.tensor([item['seq_length'] for item in batch])
        
        for i, item in enumerate(batch):
            seq_data[i, :item['seq_length']] = item['seq_data']
        
        # Process text inputs
        text_fields = {
            field: [item['text_inputs'][field] for item in batch]
            for field in batch[0]['text_inputs'].keys()
        }
        
        # Tokenize text inputs
        tokenized_fields = {}
        for field, texts in text_fields.items():
            # Replace empty strings with a space to avoid tokenizer errors
            texts = [text if text.strip() else " " for text in texts]
            # Tokenize with padding and truncation
            tokens = self.text_encoder.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.text_max_length,
                return_tensors='pt'
            )
            tokenized_fields[field] = tokens['input_ids']
        
        # Get labels
        labels = torch.stack([item['labels']['global'] for item in batch])
        
        return {
            'node_features': node_features,
            'edge_index': edge_indices,
            'edge_attr': edge_attrs,
            'sequential_features': seq_data,
            'input_ids': tokenized_fields,
            'labels': labels
        } 