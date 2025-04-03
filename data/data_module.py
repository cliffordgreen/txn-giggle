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
from scipy.spatial import KDTree
from torch.nn.utils.rnn import pad_sequence

# class TransactionDataset(Dataset):
#     """Dataset for transaction classification."""
#     def __init__(
#         self,
#         transactions_df: pd.DataFrame,
#         graph_data: HeteroData,
#         text_encoder: 'MultiFieldTextEncoder',
#         seq_data: Dict[int, torch.Tensor],
#         seq_lengths: Dict[int, int],
#         indices: List[int]
#     ):
#         self.transactions_df = transactions_df
#         self.graph_data = graph_data
#         self.text_encoder = text_encoder
#         self.seq_data = seq_data
#         self.seq_lengths = seq_lengths
#         self.indices = indices
        
#     def __len__(self) -> int:
#         return len(self.indices)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         """Get a single transaction sample."""
#         # Get transaction index
#         tx_idx = self.indices[idx]
        
#         # Get transaction data
#         tx = self.transactions_df.iloc[tx_idx]
        
#         # Get sequence data
#         user_id = tx['user_id']
#         seq = self.seq_data[user_id]
#         seq_len = self.seq_lengths[user_id]
        
#         # Get text inputs
#         text_inputs = {
#             'description': str(tx['description']) if pd.notna(tx['description']) else '',
#             'memo': str(tx['memo']) if pd.notna(tx['memo']) else '',
#             'merchant': str(tx['merchant_name']) if pd.notna(tx['merchant_name']) else ''
#         }
        
#         # Get labels
#         labels = {
#             'global': torch.tensor(tx['category_id'], dtype=torch.long),
#             'user': torch.tensor(tx['user_category_id'], dtype=torch.long)
#         }
        
#         # Get graph data
#         graph_data = {
#             'x_dict': {
#                 'transaction': self.graph_data['transaction'].x[tx_idx],
#                 'merchant': self.graph_data['merchant'].x,
#                 'category': self.graph_data['category'].x
#             },
#             'edge_index_dict': {
#                 edge_type: self.graph_data[edge_type].edge_index
#                 for edge_type in [
#                     ('transaction', 'belongs_to', 'merchant'),
#                     ('merchant', 'categorized_as', 'category'),
#                     ('transaction', 'temporal', 'transaction'),
#                     ('transaction', 'similar_amount', 'transaction')
#                 ]
#             },
#             'edge_attr_dict': {
#                 edge_type: self.graph_data[edge_type].edge_attr
#                 for edge_type in [
#                     ('transaction', 'belongs_to', 'merchant'),
#                     ('merchant', 'categorized_as', 'category'),
#                     ('transaction', 'temporal', 'transaction'),
#                     ('transaction', 'similar_amount', 'transaction')
#                 ]
#             }
#         }
        
#         return {
#             'tx_idx': tx_idx,
#             'seq_data': seq,
#             'seq_length': seq_len,
#             'text_inputs': text_inputs,
#             'labels': labels,
#             'graph_data': graph_data
#         }

# class TransactionDataModule(pl.LightningDataModule):

class TransactionDataset(Dataset):
    """Optimized Dataset for transaction classification."""
    def __init__(
        self,
        indices: List[int],
        labels_global: torch.Tensor,
        labels_user: torch.Tensor,
        seq_data_map: Dict[int, torch.Tensor], # Map user_id -> sequence tensor
        seq_lengths_map: Dict[int, int],      # Map user_id -> sequence length
        text_data: Dict[str, List[str]],      # Dict field -> List of texts for relevant indices
        user_ids: pd.Series,                  # Series of user_ids for relevant indices
    ):
        self.indices = indices # Original DataFrame indices (if needed for mapping back)
        self.labels_global = labels_global # Tensor of labels for items in this dataset split
        self.labels_user = labels_user     # Tensor of user-specific labels
        self.seq_data_map = seq_data_map
        self.seq_lengths_map = seq_lengths_map
        self.text_data = text_data         # Contains 'description', 'memo', 'merchant' lists
        self.user_ids = user_ids           # Pandas Series or numpy array of user ids

        # Store lists directly for faster access in __getitem__
        self.descriptions = self.text_data['description']
        self.memos = self.text_data['memo']
        self.merchants = self.text_data['merchant']
        # Convert user_ids Series to numpy for potentially faster lookup if large
        self.user_ids_np = self.user_ids.to_numpy()


    def __len__(self) -> int:
        # The length is the number of examples in this specific split (train/val/test)
        return len(self.labels_global)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single transaction sample efficiently."""
        # idx here is the index within *this dataset split* (0 to len(split)-1)
        original_tx_idx = self.indices[idx] # Get original DataFrame index if needed elsewhere
        user_id = self.user_ids_np[idx]

        seq = self.seq_data_map.get(user_id, torch.empty(0, dtype=torch.float)) # Handle missing user?
        seq_len = self.seq_lengths_map.get(user_id, 0)

        text_inputs = {
            'description': self.descriptions[idx],
            'memo': self.memos[idx],
            'merchant': self.merchants[idx]
        }

        labels = {
            'global': self.labels_global[idx],
            'user': self.labels_user[idx]
        }

        return {
            'tx_idx': original_tx_idx, # Original index
            'user_id': user_id,
            'seq_data': seq,
            'seq_length': seq_len, # Pass length for padding in collate_fn
            'text_inputs': text_inputs,
            'labels': labels,
        }

# --- Optimized TransactionDataModule ---
class TransactionDataModule(pl.LightningDataModule):
    """Optimized Data module for transaction classification."""
    def __init__(
        self,
        transactions_df: pd.DataFrame,
        batch_size: int = 32,
        num_workers: int = 4, # Try increasing this now
        max_seq_length: int = 50,
        graph_k_neighbors_amount: int = 5, # k for similar amount edges
        text_model_name: str = 'bert-base-uncased',
        text_max_length: int = 128,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42 # For reproducible splits
    ):
        super().__init__()
        # Make a copy to avoid modifying the original DataFrame outside the module
        self.transactions_df = transactions_df.copy()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.graph_k_neighbors_amount = graph_k_neighbors_amount
        self.text_model_name = text_model_name
        self.text_max_length = text_max_length
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # Placeholders
        self.text_encoder = None
        self.graph_data: Optional[HeteroData] = None
        self.seq_data_map: Optional[Dict[int, torch.Tensor]] = None
        self.seq_lengths_map: Optional[Dict[int, int]] = None
        self.train_indices: Optional[List[int]] = None
        self.val_indices: Optional[List[int]] = None
        self.test_indices: Optional[List[int]] = None
        self.train_dataset: Optional[TransactionDataset] = None
        self.val_dataset: Optional[TransactionDataset] = None
        self.test_dataset: Optional[TransactionDataset] = None

        # Mappings needed for graph construction and dataset creation
        self.user_id_map: Optional[Dict] = None
        self.merchant_id_map: Optional[Dict] = None
        self.category_id_map: Optional[Dict] = None
        self.num_transactions: int = 0
        self.num_merchants: int = 0
        self.num_categories: int = 0


    def setup(self, stage: Optional[str] = None):
        """Setup data for training, validation, and testing."""
        # Prevent running setup multiple times unnecessarily
        if self.graph_data is not None and self.train_dataset is not None:
            print("DataModule setup already completed.")
            return

        print("Running DataModule setup...")
        self.num_transactions = len(self.transactions_df)

        # 1. Initialize Text Encoder (needed for collate_fn)
        print("Initializing text encoder...")
        self.text_encoder = MultiFieldTextEncoder(
            model_name=self.text_model_name,
            max_length=self.text_max_length
        )

        # 2. Preprocess Data & Create Mappings (before graph building)
        print("Preprocessing data and creating mappings...")
        self._preprocess_and_map()

        # 3. Build Graph (using mappings)
        print("Building graph...")
        self._build_graph() # This now populates self.graph_data

        # 4. Prepare Sequences
        print("Preparing sequences...")
        self._prepare_sequences() # Populates self.seq_data_map, self.seq_lengths_map

        # 5. Split Data (indices for train/val/test)
        print("Splitting data...")
        self._split_data() # Populates self.train_indices, self.val_indices, self.test_indices


         # AFTER self.graph_data is created in _build_graph:
        print("DataModule setup finished graph construction.")
        # --- Move graph_data to the correct device ---
        # Determine the target device (e.g., from trainer if available, or check CUDA)
        target_device = None
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'strategy') and hasattr(self.trainer.strategy, 'root_device'):
             target_device = self.trainer.strategy.root_device
             print(f"Detected target device from trainer: {target_device}")
        elif torch.cuda.is_available():
             # Fallback if trainer isn't fully available yet, assume default GPU
             target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
             print(f"Assuming target device: {target_device}")
        else:
             target_device = torch.device("cpu")
             print(f"Using target device: {target_device}")
        if self.graph_data is not None and target_device is not None and target_device.type != 'cpu':
            try:
                print(f"Attempting to move graph_data to {target_device}...")
                # Use the .to() method for HeteroData objects
                self.graph_data = self.graph_data.to(target_device)
                # Verify one tensor within the graph data
                if 'transaction' in self.graph_data and hasattr(self.graph_data['transaction'], 'x'):
                     print(f"Moved graph_data successfully. Example tensor device: {self.graph_data['transaction'].x.device}")
                else:
                    print("Moved graph_data (structure validation pending).")
            except Exception as e:
                 print(f"[ERROR] Failed to move graph_data to {target_device}: {e}")
                 print("[ERROR] Graph operations will likely fail due to device mismatch.")
        elif self.graph_data is None:
             print("[WARNING] graph_data is None after setup.")
        else:
            print("graph_data remains on CPU.")
            # 6. Create Datasets (only requires indices and precomputed data maps)
            # We create them here to avoid redundant data slicing later
            print("Creating datasets...")
        common_kwargs = {
            "seq_data_map": self.seq_data_map,
            "seq_lengths_map": self.seq_lengths_map,
            # Pass only the necessary user_ids and text data for each split
        }

        self.train_dataset = self._create_dataset_split(self.train_indices, **common_kwargs)
        self.val_dataset = self._create_dataset_split(self.val_indices, **common_kwargs)
        self.test_dataset = self._create_dataset_split(self.test_indices, **common_kwargs)

        print("DataModule setup finished.")


    def _preprocess_and_map(self):
        """Clean data, create numerical IDs, and precompute text fields."""
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.transactions_df['timestamp']):
             self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])

        # Sort by timestamp (important for temporal logic if needed later, good practice)
        self.transactions_df.sort_values('timestamp', inplace=True)
        self.transactions_df.reset_index(drop=True, inplace=True) # Ensure index is 0..N-1

        # Create numerical mappings for nodes
        # Use factorize for efficiency and handling unseen values gracefully if needed
        merchant_codes, self.merchant_id_map = pd.factorize(self.transactions_df['merchant_name'])
        category_codes, self.category_id_map = pd.factorize(self.transactions_df['category_id'])
        # User IDs might already be integers, but ensure they are mapped if sparse/non-sequential
        user_codes, self.user_id_map = pd.factorize(self.transactions_df['user_id'])

        self.transactions_df['merchant_idx'] = merchant_codes
        self.transactions_df['category_idx'] = category_codes
        self.transactions_df['user_idx'] = user_codes # Use this if user features are added to graph

        self.num_merchants = len(self.merchant_id_map)
        self.num_categories = len(self.category_id_map)
        # self.num_users = len(self.user_id_map) # If users become nodes

        # Pre-process text fields (handle NaN) - do this once
        self.transactions_df['description_proc'] = self.transactions_df['description'].fillna('').astype(str)
        self.transactions_df['memo_proc'] = self.transactions_df['memo'].fillna('').astype(str)
        self.transactions_df['merchant_name_proc'] = self.transactions_df['merchant_name'].fillna('').astype(str)

        # Optional: Scale numerical features
        # scaler_amount = StandardScaler()
        # self.transactions_df['amount_scaled'] = scaler_amount.fit_transform(self.transactions_df[['amount']])
        # Consider scaling other numerical features if used in the graph


    def _build_graph(self):
        """Build heterogeneous transaction graph efficiently using precomputed mappings."""
        data = HeteroData()

        # --- Node Features ---
        # Transaction Node Features (Vectorized)
        print("Calculating transaction node features...")
        hours = self.transactions_df['hour'].values
        days = self.transactions_df['weekday'].values
        amounts = self.transactions_df['amount'].values.astype(np.float32)
        # Use 'timestamp' seconds since epoch directly if needed, or keep as is
        timestamps_sec = self.transactions_df['timestamp'].apply(lambda x: x.timestamp()).values.astype(np.float32)

        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        day_sin = np.sin(2 * np.pi * days / 7)
        day_cos = np.cos(2 * np.pi * days / 7)

        # Combine features - ensure all are float32 for consistency
        tx_features = np.stack([
            amounts,
            hour_sin.astype(np.float32), hour_cos.astype(np.float32),
            day_sin.astype(np.float32), day_cos.astype(np.float32),
            timestamps_sec
            # Add scaled amount if used: self.transactions_df['amount_scaled'].values
        ], axis=1)
        data['transaction'].x = torch.from_numpy(tx_features)

        # Merchant Node Features (using groupby().agg())
        # Ensure NaNs in amount are handled if they exist (e.g., fillna(0)) before agg
        print("Calculating merchant node features...")
        merchant_stats = self.transactions_df.groupby('merchant_idx')['amount'].agg(
            ['mean', 'std', 'max', 'min', 'count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        ).fillna(0) # Fill NaNs resulting from std on single-item groups etc.
        # Ensure order matches merchant_idx 0..M-1
        merchant_stats = merchant_stats.reindex(range(self.num_merchants), fill_value=0)
        data['merchant'].x = torch.tensor(merchant_stats.values, dtype=torch.float32)

        # Category Node Features (using groupby().agg())
        print("Calculating category node features...")
        category_stats = self.transactions_df.groupby('category_idx')['amount'].agg(
             ['mean', 'std', 'max', 'min', 'count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        ).fillna(0)
        category_stats = category_stats.reindex(range(self.num_categories), fill_value=0)
        data['category'].x = torch.tensor(category_stats.values, dtype=torch.float32)

        # --- Edge Indices & Features ---

        # Transaction -> Merchant edges (belongs_to)
        print("Creating (transaction, belongs_to, merchant) edges...")
        edge_index_t_m = torch.stack([
            torch.arange(self.num_transactions, dtype=torch.long),
            torch.from_numpy(self.transactions_df['merchant_idx'].values)
        ], dim=0)
        data['transaction', 'belongs_to', 'merchant'].edge_index = edge_index_t_m

        # Optional: Edge features (e.g., amount Z-score relative to merchant mean/std)
        # Precompute merchant mean/std
        # merchant_means = merchant_stats['mean'].values
        # merchant_stds = merchant_stats['std'].values
        # tx_amounts = data['transaction'].x[:, 0].numpy() # Get amounts back
        # tx_merchant_idxs = self.transactions_df['merchant_idx'].values
        # means_for_tx = merchant_means[tx_merchant_idxs]
        # stds_for_tx = merchant_stds[tx_merchant_idxs]
        # edge_attr_t_m = (tx_amounts - means_for_tx) / (stds_for_tx + 1e-6)
        # data['transaction', 'belongs_to', 'merchant'].edge_attr = torch.tensor(edge_attr_t_m, dtype=torch.float32).unsqueeze(1)


        # Merchant -> Category edges (categorized_as)
        # Find the most frequent category_idx for each merchant_idx
        print("Creating (merchant, categorized_as, category) edges...")
        merchant_to_category = self.transactions_df.groupby('merchant_idx')['category_idx'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1)
        merchant_to_category = merchant_to_category.reindex(range(self.num_merchants), fill_value=-1) # Ensure all merchants are present
        valid_map = merchant_to_category[merchant_to_category != -1]

        edge_index_m_c = torch.stack([
            torch.tensor(valid_map.index.values, dtype=torch.long), # merchant_idx
            torch.tensor(valid_map.values, dtype=torch.long)      # category_idx
        ], dim=0)
        data['merchant', 'categorized_as', 'category'].edge_index = edge_index_m_c
        # Optional: Edge features (e.g., confidence = count(mode) / count(total))


        # Transaction -> Transaction edges (temporal) - Simplified Example
        # Create edges between consecutive transactions *for the same user*
        # This is O(N) after sorting. A fixed window (like your original code)
        # or radius graph based on time could also be implemented efficiently.
        print("Creating (transaction, temporal, transaction) edges...")
        temporal_edges_src = []
        temporal_edges_dst = []
        temporal_attrs = []
        # Group by user, then find consecutive transactions within each group
        self.transactions_df['original_index'] = self.transactions_df.index # Keep track of 0..N-1 index
        for user_id, group in self.transactions_df.groupby('user_idx'):
             if len(group) > 1:
                indices = group['original_index'].values
                timestamps = group['timestamp'].values
                # Add edges between consecutive transactions
                temporal_edges_src.extend(indices[:-1])
                temporal_edges_dst.extend(indices[1:])
                # Calculate time diff in hours for potential edge feature
                time_diffs = (timestamps[1:] - timestamps[:-1]) / np.timedelta64(1, 'h')
                temporal_attrs.extend(time_diffs)

        # Add reverse edges? Depends on GNN model (MessagePassing handles this)
        # If adding reverse, remember to adjust/duplicate attributes appropriately

        if temporal_edges_src:
             edge_index_t_t_temp = torch.tensor([temporal_edges_src, temporal_edges_dst], dtype=torch.long)
             edge_attr_t_t_temp = torch.tensor(temporal_attrs, dtype=torch.float32).unsqueeze(1)
             data['transaction', 'temporal', 'transaction'].edge_index = edge_index_t_t_temp
             data['transaction', 'temporal', 'transaction'].edge_attr = edge_attr_t_t_temp


        # Transaction -> Transaction edges (similar_amount) - Using KDTree
        print("Creating (transaction, similar_amount, transaction) edges...")
        k = self.graph_k_neighbors_amount + 1 # +1 because it includes self
        tx_amounts = data['transaction'].x[:, 0].numpy().reshape(-1, 1) # KDTree needs 2D array
        kdtree = KDTree(tx_amounts)
        # Query for k nearest neighbors for all points
        distances, indices = kdtree.query(tx_amounts, k=k) # shape (N, k)

        # indices[:, 0] is the point itself, neighbors are indices[:, 1:]
        src_nodes = np.repeat(np.arange(self.num_transactions), k - 1) # Repeat self index
        dst_nodes = indices[:, 1:].flatten() # Neighbor indices
        amount_diffs = distances[:, 1:].flatten() # Distances are absolute amount diffs

        # Remove potential invalid indices if N < k
        valid_mask = dst_nodes < self.num_transactions
        src_nodes = src_nodes[valid_mask]
        dst_nodes = dst_nodes[valid_mask]
        amount_diffs = amount_diffs[valid_mask]

        edge_index_t_t_sim = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr_t_t_sim = torch.tensor(amount_diffs, dtype=torch.float32).unsqueeze(1)
        # Optional: Add amount ratio as another feature
        # amount_src = tx_amounts[src_nodes].flatten()
        # amount_dst = tx_amounts[dst_nodes].flatten()
        # ratio = np.minimum(amount_src, amount_dst) / (np.maximum(amount_src, amount_dst) + 1e-6)
        # edge_attr_t_t_sim = torch.cat([...], dim=1)


        data['transaction', 'similar_amount', 'transaction'].edge_index = edge_index_t_t_sim
        data['transaction', 'similar_amount', 'transaction'].edge_attr = edge_attr_t_t_sim

        # Validate and store graph
        # print(data)
        # data.validate() # Good practice
        self.graph_data = data


    def _prepare_sequences(self):
        """Prepare sequence data efficiently for each user."""
        self.seq_data_map = {}
        self.seq_lengths_map = {}

        # Required columns for sequence features
        feature_cols = ['amount', 'timestamp_sec', 'weekday', 'hour'] # Example features
        # Precompute timestamp seconds if not done already
        if 'timestamp_sec' not in self.transactions_df:
            self.transactions_df['timestamp_sec'] = self.transactions_df['timestamp'].apply(lambda x: x.timestamp())

        # Group by user_id (original one)
        # Sort each group by timestamp (already done globally, but safer here)
        # Apply function to create sequence tensor
        for user_id, group in self.transactions_df.groupby('user_id'):
            group = group.sort_values('timestamp')
            seq_features = group[feature_cols].values.astype(np.float32)
            seq = torch.from_numpy(seq_features)

            # Truncate/Pad (truncate here, padding in collate_fn)
            if len(seq) > self.max_seq_length:
                seq = seq[-self.max_seq_length:] # Take last 'max_seq_length' transactions

            self.seq_data_map[user_id] = seq
            self.seq_lengths_map[user_id] = len(seq)


    def _split_data(self):
        """Split data indices based on users."""
        unique_user_ids = self.transactions_df['user_id'].unique()
        n_users = len(unique_user_ids)

        n_val = int(n_users * self.val_ratio)
        n_test = int(n_users * self.test_ratio)

        # Ensure reproducibility
        np.random.seed(self.random_state)
        shuffled_users = np.random.permutation(unique_user_ids)

        val_users = set(shuffled_users[:n_val])
        test_users = set(shuffled_users[n_val : n_val + n_test])
        train_users = set(shuffled_users[n_val + n_test :])

        # Get DataFrame indices for each split much faster using boolean indexing
        self.train_indices = self.transactions_df.index[self.transactions_df['user_id'].isin(train_users)].tolist()
        self.val_indices = self.transactions_df.index[self.transactions_df['user_id'].isin(val_users)].tolist()
        self.test_indices = self.transactions_df.index[self.transactions_df['user_id'].isin(test_users)].tolist()

        print(f"Data split: Train={len(self.train_indices)}, Val={len(self.val_indices)}, Test={len(self.test_indices)}")


    def _create_dataset_split(self, indices: List[int], **kwargs) -> TransactionDataset:
        """Helper to create a TransactionDataset for a specific index split."""
        if not indices:
             return None # Handle empty splits

        # Efficiently slice the required data using the indices for the split
        split_df = self.transactions_df.iloc[indices]

        labels_global = torch.tensor(split_df['category_id'].values, dtype=torch.long) # Assuming global label is category_id
        labels_user = torch.tensor(split_df['user_category_id'].values, dtype=torch.long) # Assuming user-specific label exists

        text_data = {
             'description': split_df['description_proc'].tolist(),
             'memo': split_df['memo_proc'].tolist(),
             'merchant': split_df['merchant_name_proc'].tolist()
        }
        user_ids = split_df['user_id'] # Pass the series directly

        return TransactionDataset(
             indices=indices, # Pass original indices for this split
             labels_global=labels_global,
             labels_user=labels_user,
             text_data=text_data,
             user_ids=user_ids,
             **kwargs # Passes seq_data_map and seq_lengths_map
         )


    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Run setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True # Usually good for GPU training
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Run setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=15,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Run setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )


    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Optimized collate function for batching."""
        # Batch items returned by __getitem__

        # --- Sequence Data ---
        seq_list = [item['seq_data'] for item in batch]
        # Pad sequences dynamically based on the max length in *this batch*
        # Use batch_first=True convention
        padded_seqs = pad_sequence(seq_list, batch_first=True, padding_value=0.0)
        seq_lengths = torch.tensor([item['seq_length'] for item in batch], dtype=torch.long)

        # --- Text Data ---
        tokenized_fields = {}
        # Get all text fields from the first item (assuming structure is consistent)
        text_field_keys = batch[0]['text_inputs'].keys()

        for field in text_field_keys:
            texts = [item['text_inputs'][field] for item in batch]
            # Replace truly empty strings with a space if tokenizer requires it
            texts = [text if text.strip() else " " for text in texts]
            tokens = self.text_encoder.tokenizer(
                texts,
                padding=True,      # Pad to max length *in this batch*
                truncation=True,
                max_length=self.text_max_length,
                return_tensors='pt' # Return PyTorch tensors
            )
            # We typically need input_ids and attention_mask
            tokenized_fields[f'{field}_input_ids'] = tokens['input_ids']
            tokenized_fields[f'{field}_attention_mask'] = tokens['attention_mask']

        # --- Labels ---
        labels_global = torch.stack([item['labels']['global'] for item in batch])
        labels_user = torch.stack([item['labels']['user'] for item in batch])

        # --- Indices ---
        # These are the original DataFrame indices for the items in the batch
        batch_tx_indices = torch.tensor([item['tx_idx'] for item in batch], dtype=torch.long)
        # User IDs might be useful too
        batch_user_ids = torch.tensor([item['user_id'] for item in batch], dtype=torch.long)


        # The collated batch should contain everything the model's forward pass needs
        # Note: The graph_data itself is NOT part of the batch here.
        # The model will access the full self.graph_data stored in the DataModule.
        return {
            'batch_indices': batch_tx_indices, # Indices of transactions in this batch
            'user_ids': batch_user_ids,
            'seq_features': padded_seqs,
            'seq_lengths': seq_lengths,
            'text_features': tokenized_fields, # Contains input_ids and attention_masks per field
            'labels_global': labels_global,
            'labels_user': labels_user
            # Add any other item-specific data needed by the model
        }

    # """Data module for transaction classification."""
    # def __init__(
    #     self,
    #     transactions_df: pd.DataFrame,
    #     batch_size: int = 32,
    #     num_workers: int = 4,
    #     max_seq_length: int = 50,
    #     graph_neighbors: Dict[str, int] = None,
    #     text_model_name: str = 'bert-base-uncased',
    #     text_max_length: int = 128,
    #     val_ratio: float = 0.1,
    #     test_ratio: float = 0.1
    # ):
    #     super().__init__()
    #     self.transactions_df = transactions_df
    #     self.batch_size = batch_size
    #     self.num_workers = num_workers
    #     self.max_seq_length = max_seq_length
    #     self.graph_neighbors = graph_neighbors or {
    #         'same_merchant': 10,
    #         'same_company': 5,
    #         'similar_amount': 5
    #     }
    #     self.text_model_name = text_model_name
    #     self.text_max_length = text_max_length
    #     self.val_ratio = val_ratio
    #     self.test_ratio = test_ratio
        
    #     # Initialize text encoder for tokenization
    #     self.text_encoder = None
    #     self.graph_data = None
    #     self.seq_data = None
    #     self.seq_lengths = None
    #     self.train_indices = None
    #     self.val_indices = None
    #     self.test_indices = None
        
    # def setup(self, stage: Optional[str] = None):
    #     """Setup data for training."""
    #     if stage == 'fit' or stage is None:
    #         # Initialize text encoder
    #         self.text_encoder = MultiFieldTextEncoder(
    #             model_name=self.text_model_name,
    #             max_length=self.text_max_length
    #         )
            
    #         # Build graph
    #         self._build_graph()
            
    #         # Prepare sequences
    #         self._prepare_sequences()
            
    #         # Split data
    #         self._split_data()
            
    #     if stage == 'test' or stage is None:
    #         # For testing, we need the same setup as training
    #         if self.text_encoder is None:
    #             self.setup('fit')
    
    # def _build_graph(self):
    #     """Build heterogeneous transaction graph with multiple node types and edge types."""
    #     print("Building graph this may take a while...")
    #     # Create transaction node features with cyclical time encoding
    #     transaction_features = []
    #     for _, row in self.transactions_df.iterrows():
    #         # Cyclical encoding for time features
    #         hour = row['hour']
    #         day = row['weekday']
            
    #         # Hour encoding (24-hour cycle)
    #         hour_sin = np.sin(2 * np.pi * hour / 24)
    #         hour_cos = np.cos(2 * np.pi * hour / 24)
            
    #         # Day encoding (7-day cycle)
    #         day_sin = np.sin(2 * np.pi * day / 7)
    #         day_cos = np.cos(2 * np.pi * day / 7)
            
    #         # Combine features
    #         features = [
    #             row['amount'],  # Raw amount
    #             hour_sin, hour_cos,  # Cyclical hour encoding
    #             day_sin, day_cos,    # Cyclical day encoding
    #             row['timestamp'].timestamp()  # Absolute timestamp
    #         ]
    #         transaction_features.append(features)
    #     transaction_features = torch.tensor(transaction_features, dtype=torch.float)
    #     print("finished with cyclical")
    #     # Create merchant node features (using aggregated statistics)
    #     merchant_features = {}
    #     for merchant_name, group in self.transactions_df.groupby('merchant_name'):
    #         merchant_features[merchant_name] = torch.tensor([
    #             group['amount'].mean(),
    #             group['amount'].std(),
    #             group['amount'].max(),
    #             group['amount'].min(),
    #             len(group),  # transaction count
    #             group['amount'].median(),
    #             group['amount'].quantile(0.25),
    #             group['amount'].quantile(0.75)
    #         ], dtype=torch.float)
    #     print("finished with merchant nodes")
    #     # Create category node features (using aggregated statistics)
    #     category_features = {}
    #     for category_id, group in self.transactions_df.groupby('category_id'):
    #         category_features[category_id] = torch.tensor([
    #             group['amount'].mean(),
    #             group['amount'].std(),
    #             group['amount'].max(),
    #             group['amount'].min(),
    #             len(group),  # transaction count
    #             group['amount'].median(),
    #             group['amount'].quantile(0.25),
    #             group['amount'].quantile(0.75)
    #         ], dtype=torch.float)
    #     print("finished with catagory nodes")
    #     # Initialize dictionaries for edge indices and attributes
    #     edge_indices = {}
    #     edge_attrs = {}
        
    #     # Transaction -> Merchant edges (belongs_to)
    #     belongs_to_edges = []
    #     belongs_to_attrs = []
    #     for idx, row in self.transactions_df.iterrows():
    #         belongs_to_edges.append([idx, row['merchant_name']])
    #         # Enhanced edge features: amount relative to merchant's statistics
    #         merchant_group = self.transactions_df[self.transactions_df['merchant_name'] == row['merchant_name']]
    #         amount_mean = merchant_group['amount'].mean()
    #         amount_std = merchant_group['amount'].std()
    #         amount_zscore = (row['amount'] - amount_mean) / (amount_std + 1e-6)
    #         belongs_to_attrs.append([amount_zscore])
    #     if belongs_to_edges:
    #         edge_indices[('transaction', 'belongs_to', 'merchant')] = torch.tensor(belongs_to_edges, dtype=torch.long).t()
    #         edge_attrs[('transaction', 'belongs_to', 'merchant')] = torch.tensor(belongs_to_attrs, dtype=torch.float)
    #     print("finished with transaction-> merchent edges")

    #     # Merchant -> Category edges (categorized_as)
    #     categorized_as_edges = []
    #     categorized_as_attrs = []
    #     for merchant_name, group in self.transactions_df.groupby('merchant_name'):
    #         category_id = group['category_id'].iloc[0]  # Most common category
    #         categorized_as_edges.append([merchant_name, category_id])
    #         # Enhanced edge features: merchant's category confidence
    #         category_counts = group['category_id'].value_counts()
    #         category_confidence = category_counts.iloc[0] / len(group)
    #         categorized_as_attrs.append([category_confidence])
    #     if categorized_as_edges:
    #         edge_indices[('merchant', 'categorized_as', 'category')] = torch.tensor(categorized_as_edges, dtype=torch.long).t()
    #         edge_attrs[('merchant', 'categorized_as', 'category')] = torch.tensor(categorized_as_attrs, dtype=torch.float)
    #     print("finished with catagory-> merchent edges")
    #     # Transaction -> Transaction edges (temporal)
    #     temporal_edges = []
    #     temporal_attrs = []
    #     for i in range(len(self.transactions_df) - 1):
    #         for j in range(i + 1, min(i + 5, len(self.transactions_df))):
    #             time_diff = abs((self.transactions_df.iloc[i]['timestamp'] - 
    #                            self.transactions_df.iloc[j]['timestamp']).total_seconds())
    #             if time_diff <= 86400:  # Within 24 hours
    #                 temporal_edges.extend([
    #                     [i, j],
    #                     [j, i]
    #                 ])
    #                 # Enhanced temporal edge features
    #                 time_diff_hours = time_diff / 3600
    #                 time_diff_days = time_diff / 86400
    #                 temporal_attrs.extend([
    #                     [time_diff_hours, time_diff_days, 1.0],  # Forward edge
    #                     [time_diff_hours, time_diff_days, 0.0]   # Backward edge
    #                 ])
    #     if temporal_edges:
    #         edge_indices[('transaction', 'temporal', 'transaction')] = torch.tensor(temporal_edges, dtype=torch.long).t()
    #         edge_attrs[('transaction', 'temporal', 'transaction')] = torch.tensor(temporal_attrs, dtype=torch.float)
    #     print("finished with Transaction -> Transaction edges edges")

    #     # Transaction -> Transaction edges (similar_amount)
    #     similar_amount_edges = []
    #     similar_amount_attrs = []
    #     amounts = self.transactions_df['amount'].values
    #     for i in range(len(amounts)):
    #         diffs = np.abs(amounts - amounts[i])
    #         k_nearest = np.argsort(diffs)[1:self.graph_neighbors['similar_amount'] + 1]
    #         for j in k_nearest:
    #             similar_amount_edges.extend([
    #                 [i, j],
    #                 [j, i]
    #             ])
    #             # Enhanced amount similarity edge features
    #             amount_ratio = min(amounts[i], amounts[j]) / max(amounts[i], amounts[j])
    #             amount_diff = diffs[j]
    #             similar_amount_attrs.extend([
    #                 [amount_ratio, amount_diff, 1.0],  # Forward edge
    #                 [amount_ratio, amount_diff, 0.0]   # Backward edge
    #             ])
    #     if similar_amount_edges:
    #         edge_indices[('transaction', 'similar_amount', 'transaction')] = torch.tensor(similar_amount_edges, dtype=torch.long).t()
    #         edge_attrs[('transaction', 'similar_amount', 'transaction')] = torch.tensor(similar_amount_attrs, dtype=torch.float)
    #     print("finished with Transaction -> Transaction edges")
    #     # Create HeteroData object
    #     data = HeteroData()
        
    #     # Add node features
    #     data['transaction'].x = transaction_features
    #     data['merchant'].x = torch.stack(list(merchant_features.values()))
    #     data['category'].x = torch.stack(list(category_features.values()))
        
    #     # Add edge indices and attributes
    #     for edge_type in edge_indices:
    #         data[edge_type].edge_index = edge_indices[edge_type]
    #         data[edge_type].edge_attr = edge_attrs[edge_type]
        
    #     self.graph_data = data
    
    # def _prepare_sequences(self):
    #     """Prepare sequence data for each user."""
    #     self.seq_data = {}
    #     self.seq_lengths = {}
        
    #     # Group transactions by user
    #     user_groups = self.transactions_df.groupby('user_id')
        
    #     for user_id, group in user_groups:
    #         # Sort by timestamp
    #         group = group.sort_values('timestamp')
            
    #         # Create sequence features
    #         seq_features = []
    #         for _, row in group.iterrows():
    #             feat = [
    #                 row['amount'],
    #                 row['timestamp'].timestamp(),
    #                 row['weekday'],
    #                 row['hour']
    #             ]
    #             seq_features.append(feat)
            
    #         # Convert to tensor
    #         seq = torch.tensor(seq_features, dtype=torch.float)
            
    #         # Truncate if too long
    #         if len(seq) > self.max_seq_length:
    #             seq = seq[-self.max_seq_length:]
            
    #         self.seq_data[user_id] = seq
    #         self.seq_lengths[user_id] = len(seq)
    
    # def _split_data(self):
    #     """Split data into train/val/test sets."""
    #     # Get unique users
    #     users = self.transactions_df['user_id'].unique()
        
    #     # Randomly split users
    #     np.random.shuffle(users)
    #     n_users = len(users)
        
    #     n_val = int(n_users * self.val_ratio)
    #     n_test = int(n_users * self.test_ratio)
        
    #     val_users = users[:n_val]
    #     test_users = users[n_val:n_val + n_test]
    #     train_users = users[n_val + n_test:]
        
    #     # Get indices for each split
    #     self.train_indices = self.transactions_df[
    #         self.transactions_df['user_id'].isin(train_users)
    #     ].index.tolist()
        
    #     self.val_indices = self.transactions_df[
    #         self.transactions_df['user_id'].isin(val_users)
    #     ].index.tolist()
        
    #     self.test_indices = self.transactions_df[
    #         self.transactions_df['user_id'].isin(test_users)
    #     ].index.tolist()
    
    # def train_dataloader(self) -> DataLoader:
    #     """Get training dataloader."""
    #     dataset = TransactionDataset(
    #         transactions_df=self.transactions_df,
    #         graph_data=self.graph_data,
    #         text_encoder=self.text_encoder,
    #         seq_data=self.seq_data,
    #         seq_lengths=self.seq_lengths,
    #         indices=self.train_indices
    #     )
        
    #     return DataLoader(
    #         dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.num_workers,  # Use single process for MPS compatibility
    #         collate_fn=self._collate_fn
    #     )
    
    # def val_dataloader(self) -> DataLoader:
    #     """Get validation dataloader."""
    #     dataset = TransactionDataset(
    #         transactions_df=self.transactions_df,
    #         graph_data=self.graph_data,
    #         text_encoder=self.text_encoder,
    #         seq_data=self.seq_data,
    #         seq_lengths=self.seq_lengths,
    #         indices=self.val_indices
    #     )
        
    #     return DataLoader(
    #         dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,  # Use single process for MPS compatibility
    #         collate_fn=self._collate_fn
    #     )
    
    # def test_dataloader(self) -> DataLoader:
    #     """Get test dataloader."""
    #     dataset = TransactionDataset(
    #         transactions_df=self.transactions_df,
    #         graph_data=self.graph_data,
    #         text_encoder=self.text_encoder,
    #         seq_data=self.seq_data,
    #         seq_lengths=self.seq_lengths,
    #         indices=self.test_indices
    #     )
        
    #     return DataLoader(
    #         dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,  # Use single process for MPS compatibility
    #         collate_fn=self._collate_fn
    #     )
    
    # def _collate_fn(self, batch):
    #     """Collate function for batching."""
    #     # Get batch indices
    #     batch_indices = [item['tx_idx'] for item in batch]
        
    #     # Get node features for each node type
    #     node_features = {
    #         'transaction': self.graph_data['transaction'].x[batch_indices],
    #         'merchant': self.graph_data['merchant'].x,
    #         'category': self.graph_data['category'].x
    #     }
        
    #     # Get edge indices and attributes for each edge type
    #     edge_indices = {}
    #     edge_attrs = {}
        
    #     # Process temporal edges
    #     temporal_edges = []
    #     temporal_attrs = []
    #     max_temporal_edges_per_node = min(5, len(batch_indices) - 1) 
        
    #     for i, idx in enumerate(batch_indices):
    #         tx_time = self.transactions_df.iloc[idx]['timestamp']
    #         time_diff = []
            
    #         for j, other_idx in enumerate(batch_indices):
    #             if i != j:
    #                 # time_diff = abs((self.transactions_df.iloc[idx]['timestamp'] - 
    #                 #                self.transactions_df.iloc[other_idx]['timestamp']).total_seconds())
    #                 time_diff = abs((tx_time - 
    #                              self.transactions_df.iloc[other_idx]['timestamp']).total_seconds())
    #                 # if time_diff <= 86400:  # Within 24 hours
    #                 #     temporal_edges.extend([[i, j]])
    #                 #     temporal_attrs.extend([[time_diff]])
    #                 if time_diff <= 86400:  # Within 24 hours
    #                     time_diff.append((j, time_diff))
    #         time_diff.sort(key=lambda x:x[1])
    #         for j, diff in time_diffs[:max_temporal_edges_per_node]:
    #             temporal_edges.append([i,j])
    #             temporal_attrs.append([diff])
    #     if temporal_edges:
    #         edge_indices[('transaction', 'temporal', 'transaction')] = torch.tensor(temporal_edges, dtype=torch.long).t()
    #         edge_attrs[('transaction', 'temporal', 'transaction')] = torch.tensor(temporal_attrs, dtype=torch.float)
        
    #     # Process similar amount edges
    #     similar_amount_edges = []
    #     similar_amount_attrs = []
    #     amounts = node_features['transaction'][:, 0]  # First column is amount

    #     max_similar_edges = min(self.graph_neighbors['similar_amount'], len(batch_indices) - 1)

    #     amount_matrix = amounts.unsqueeze(1) - amounts.unsqueeze(0)
    #     amount_matrix = torch.abs(amount_matrix)

    #     mask = torch.ones_like(amount_matrix, dtype=torch.bool)
    #     mask.fill_diagonal_(False)
    #     amount_matric = amount_matrix * mask

        
    #     for i in range(len(amounts)):
    #         # diffs = torch.abs(amounts - amounts[i])
    #         # k_nearest = torch.argsort(diffs)[1:self.graph_neighbors['similar_amount'] + 1]
    #         diffs = amount_matrix[i]
    #         k_nearest = torch.topk(diffs, max_similar_edges, largest=False).indices
    #         for j in k_nearest:
    #             # similar_amount_edges.extend([[i, j]])
    #             # similar_amount_attrs.extend([[diffs[j].item()]])
    #             similar_amount_edges.extend([i, j])
    #             similar_amount_attrs.extend([diffs[j].item()])
    #     if similar_amount_edges:
    #         edge_indices[('transaction', 'similar_amount', 'transaction')] = torch.tensor(similar_amount_edges, dtype=torch.long).t()
    #         edge_attrs[('transaction', 'similar_amount', 'transaction')] = torch.tensor(similar_amount_attrs, dtype=torch.float)
        
    #     # Get max sequence length in batch
    #     max_seq_len = max(item['seq_length'] for item in batch)
        
    #     # Pad and stack sequence data
    #     seq_data = torch.zeros(len(batch), max_seq_len, batch[0]['seq_data'].size(-1))
    #     seq_lengths = torch.tensor([item['seq_length'] for item in batch])
        
    #     for i, item in enumerate(batch):
    #         seq_data[i, :item['seq_length']] = item['seq_data']
        
    #     # Process text inputs
    #     text_fields = {
    #         field: [item['text_inputs'][field] for item in batch]
    #         for field in batch[0]['text_inputs'].keys()
    #     }
        
    #     # Tokenize text inputs
    #     tokenized_fields = {}
    #     for field, texts in text_fields.items():
    #         # Replace empty strings with a space to avoid tokenizer errors
    #         texts = [text if text.strip() else " " for text in texts]
    #         # Tokenize with padding and truncation
    #         tokens = self.text_encoder.tokenizer(
    #             texts,
    #             padding=True,
    #             truncation=True,
    #             max_length=self.text_max_length,
    #             return_tensors='pt'
    #         )
    #         tokenized_fields[field] = tokens['input_ids']
        
    #     # Get labels
    #     labels = torch.stack([item['labels']['global'] for item in batch])
        
    #     return {
    #         'node_features': node_features,
    #         'edge_index': edge_indices,
    #         'edge_attr': edge_attrs,
    #         'sequential_features': seq_data,
    #         'input_ids': tokenized_fields,
    #         'labels': labels
    #     } 