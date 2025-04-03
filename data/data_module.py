import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset # Keep for type hints, but not used directly for loader
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch.nn.utils.rnn import pad_sequence # For padding sequences
from transformers import AutoTokenizer # Use tokenizer directly for setup
from typing import Dict, List, Optional, Tuple, Union
import pytorch_lightning as pl
# Removed: from models.text import MultiFieldTextEncoder # Model not needed for data prep

# Removed the old TransactionDataset class as it's replaced by NeighborLoader logic

class TransactionDataModule(pl.LightningDataModule):
    """
    Data module for transaction classification using NeighborLoader for graph sampling.
    Processes graph, sequence, and text data during setup.
    """
    def __init__(
        self,
        transactions_df: pd.DataFrame,
        batch_size: int = 128, # Adjusted default batch size
        num_workers: int = 4,
        # Sequence params
        max_seq_length: int = 50, # Max length BEFORE a transaction
        # Text params
        text_model_name: str = 'bert-base-uncased', # Used for tokenizer
        text_max_length: int = 128, # For tokenizer padding/truncation
        # Graph sampling params
        # List length should match number of GNN layers, e.g., [15, 10] for 2 layers
        num_neighbors: List[int] = [15, 10],
        # Data split params
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ):
        super().__init__()
        # --- Store initial configuration ---
        # Make a copy to avoid modifying the original DataFrame passed in
        self.transactions_df = transactions_df.copy()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.text_model_name = text_model_name
        self.text_max_length = text_max_length
        self.num_neighbors = num_neighbors
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # --- Placeholders for processed data ---
        self.tokenizer = None
        self.graph_data: Optional[HeteroData] = None
        # Node feature dimensions will be inferred during setup
        self.node_feature_dims: Dict[str, int] = {}
        # Edge feature dimensions will be inferred during setup
        self.edge_feature_dims: Dict[Tuple[str, str, str], int] = {}
        # Sequence feature dimension will be inferred during setup
        self.sequence_feature_dim: Optional[int] = None


        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.transactions_df['timestamp']):
             try:
                 self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])
                 print("Converted 'timestamp' column to datetime.")
             except Exception as e:
                 raise ValueError(f"Failed to convert 'timestamp' column to datetime: {e}")
        
        # Ensure category IDs are integers if they exist
        if 'category_id' in self.transactions_df.columns:
            self.transactions_df['category_id'] = self.transactions_df['category_id'].astype(int)
        if 'user_category_id' in self.transactions_df.columns:
             # Handle potential NaNs before converting to int
             if self.transactions_df['user_category_id'].isnull().any():
                 print(f"[WARN] 'user_category_id' contains NaNs. Filling with -1 before converting to int.")
                 self.transactions_df['user_category_id'] = self.transactions_df['user_category_id'].fillna(-1).astype(int)
             else:
                 self.transactions_df['user_category_id'] = self.transactions_df['user_category_id'].astype(int)


    def prepare_data(self):
        # Download tokenizer models if not cached
        print(f"Downloading/loading tokenizer: {self.text_model_name}")
        _ = AutoTokenizer.from_pretrained(self.text_model_name)
        print("Tokenizer ready.")

    def setup(self, stage: Optional[str] = None):
        """
        Build graph, process features, tokenize text, prepare sequences, create masks.
        This runs on each GPU in distributed settings.
        """
        print(f"Setting up data for stage: {stage}")
        # --- Initialize Tokenizer ---
        if self.tokenizer is None:
             self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)

        # --- Build Graph (Nodes and Edges) ---
        # Avoid rebuilding if already done (though setup usually runs per stage)
        if self.graph_data is None:
            print("Building graph...")
            self._build_graph()
            print("Graph built.")

            # --- Prepare and Add Sequence Data to Graph ---
            print("Preparing sequence data...")
            self._prepare_and_add_sequences()
            print("Sequence data prepared.")

            # --- Prepare and Add Text Data to Graph ---
            print("Preparing text data...")
            self._prepare_and_add_text()
            print("Text data prepared.")

            # --- Split Data and Add Masks to Graph ---
            print("Splitting data and adding masks...")
            self._split_data_and_add_masks()
            print("Data split and masks added.")
            
            print("--- Data Setup Summary ---")
            print(f"Graph: {self.graph_data}")
            print(f"Node Feature Dims: {self.node_feature_dims}")
            print(f"Edge Feature Dims: {self.edge_feature_dims}")
            print(f"Sequence Feature Dim: {self.sequence_feature_dim}")
            print(f"Num Train Nodes: {self.graph_data['transaction'].train_mask.sum()}")
            print(f"Num Val Nodes: {self.graph_data['transaction'].val_mask.sum()}")
            print(f"Num Test Nodes: {self.graph_data['transaction'].test_mask.sum()}")
            print("--- End Data Setup Summary ---")


    def _build_graph(self):
        """Build heterogeneous transaction graph with nodes and edges."""
        data = HeteroData()
        df = self.transactions_df # Use the class attribute

        # --- Node Creation & Features ---

        # Map unique IDs/names to contiguous integer indices for graph nodes
        tx_map = {idx: i for i, idx in enumerate(df.index)}
        merchants = df['merchant_name'].dropna().unique()
        merchant_map = {name: i for i, name in enumerate(merchants)}
        categories = df['category_id'].dropna().unique()
        category_map = {cat_id: i for i, cat_id in enumerate(categories)}

        num_transactions = len(tx_map)
        num_merchants = len(merchant_map)
        num_categories = len(category_map)

        # 1. Transaction Nodes
        tx_features_list = []
        for idx, row in df.iterrows():
            hour = row['hour']
            day = row['weekday']
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)
            features = [
                row['amount'], hour_sin, hour_cos, day_sin, day_cos,
                row['timestamp'].timestamp() # Absolute timestamp
            ]
            tx_features_list.append(features)
        data['transaction'].x = torch.tensor(tx_features_list, dtype=torch.float)
        self.node_feature_dims['transaction'] = data['transaction'].x.shape[1]

        # Add original indices and labels needed later
        data['transaction'].original_index = torch.tensor(list(tx_map.keys()), dtype=torch.long)
        data['transaction'].y_global = torch.tensor(df['category_id'].values, dtype=torch.long)
        # Handle user category potentially having NaNs or different dtype
        user_cat_series = df['user_category_id'].fillna(-1).astype(int) # Ensure int, fillna just in case
        data['transaction'].y_user = torch.tensor(user_cat_series.values, dtype=torch.long)


        # 2. Merchant Nodes (Aggregated features)
        merchant_features_list = [([0.0] * 8)] * num_merchants # Initialize with zeros
        for merchant_name, group in df.groupby('merchant_name'):
             if merchant_name in merchant_map: # Ensure merchant is valid
                merchant_idx = merchant_map[merchant_name]
                stats = [
                    group['amount'].mean(), group['amount'].std(ddof=0), # Use population std
                    group['amount'].max(), group['amount'].min(),
                    len(group), group['amount'].median(),
                    group['amount'].quantile(0.25), group['amount'].quantile(0.75)
                ]
                # Replace NaN std with 0 if group size is 1
                merchant_features_list[merchant_idx] = [s if not np.isnan(s) else 0.0 for s in stats]
        data['merchant'].x = torch.tensor(merchant_features_list, dtype=torch.float)
        self.node_feature_dims['merchant'] = data['merchant'].x.shape[1]


        # 3. Category Nodes (Aggregated features)
        category_features_list = [([0.0] * 8)] * num_categories # Initialize
        for category_id, group in df.groupby('category_id'):
             if category_id in category_map:
                 category_idx = category_map[category_id]
                 stats = [
                    group['amount'].mean(), group['amount'].std(ddof=0),
                    group['amount'].max(), group['amount'].min(),
                    len(group), group['amount'].median(),
                    group['amount'].quantile(0.25), group['amount'].quantile(0.75)
                ]
                 category_features_list[category_idx] = [s if not np.isnan(s) else 0.0 for s in stats]
        data['category'].x = torch.tensor(category_features_list, dtype=torch.float)
        self.node_feature_dims['category'] = data['category'].x.shape[1]

        # --- Edge Creation & Features ---
        edge_index_dict = {}
        edge_attr_dict = {}

        # 1. Transaction -> Merchant (belongs_to)
        edge_list = []
        attr_list = []
        # Precompute merchant stats for efficiency
        merchant_stats = df.groupby('merchant_name')['amount'].agg(['mean', 'std']).fillna(0)
        for idx, row in df.iterrows():
             if pd.notna(row['merchant_name']) and row['merchant_name'] in merchant_map:
                tx_node_idx = tx_map[idx]
                merchant_node_idx = merchant_map[row['merchant_name']]
                edge_list.append([tx_node_idx, merchant_node_idx])
                # Edge attributes: z-score of amount relative to merchant mean/std
                stats = merchant_stats.loc[row['merchant_name']]
                amount_mean, amount_std = stats['mean'], stats['std']
                amount_zscore = (row['amount'] - amount_mean) / (amount_std + 1e-6)
                attr_list.append([amount_zscore])
        if edge_list:
            edge_index_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr_dict[('transaction', 'belongs_to', 'merchant')] = torch.tensor(attr_list, dtype=torch.float)
            self.edge_feature_dims[('transaction', 'belongs_to', 'merchant')] = edge_attr_dict[('transaction', 'belongs_to', 'merchant')].shape[1]


        # 2. Merchant -> Category (categorized_as)
        edge_list = []
        attr_list = []
        # Find the primary category for each merchant
        merchant_primary_category = df.groupby('merchant_name')['category_id'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1)
        merchant_category_confidence = df.groupby('merchant_name')['category_id'].agg(lambda x: x.value_counts(normalize=True).max() if not x.empty else 0)

        for merchant_name, primary_cat_id in merchant_primary_category.items():
            if pd.notna(merchant_name) and merchant_name in merchant_map and pd.notna(primary_cat_id) and primary_cat_id in category_map:
                merchant_node_idx = merchant_map[merchant_name]
                category_node_idx = category_map[primary_cat_id]
                edge_list.append([merchant_node_idx, category_node_idx])
                # Edge attribute: confidence of primary category
                confidence = merchant_category_confidence.get(merchant_name, 0)
                attr_list.append([confidence])
        if edge_list:
            edge_index_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr_dict[('merchant', 'categorized_as', 'category')] = torch.tensor(attr_list, dtype=torch.float)
            self.edge_feature_dims[('merchant', 'categorized_as', 'category')] = edge_attr_dict[('merchant', 'categorized_as', 'category')].shape[1]


        # 3. Transaction -> Transaction (temporal) - More efficient approach needed for large data
        # This quadratic approach is TOO SLOW for large datasets.
        # Consider time-based windowing or k-NN on timestamps if performance is an issue.
        # For now, keeping a simplified version.
        edge_list = []
        attr_list = []
        df_sorted = df.sort_values('timestamp')
        tx_map_sorted = {idx: i for i, idx in enumerate(df_sorted.index)} # Map original index to sorted position

        for i in range(num_transactions):
            ts_i = df_sorted.iloc[i]['timestamp']
            orig_idx_i = df_sorted.index[i]
            tx_node_i = tx_map[orig_idx_i] # Use original graph node index

            # Look ahead limited window
            for k in range(1, 6): # Check next 5 transactions
                j = i + k
                if j >= num_transactions: break
                ts_j = df_sorted.iloc[j]['timestamp']
                time_diff_seconds = abs((ts_i - ts_j).total_seconds())

                if time_diff_seconds <= 86400 * 1: # 1 day window
                    orig_idx_j = df_sorted.index[j]
                    tx_node_j = tx_map[orig_idx_j] # Use original graph node index
                    
                    edge_list.extend([[tx_node_i, tx_node_j], [tx_node_j, tx_node_i]])
                    time_diff_norm = time_diff_seconds / 86400.0 # Normalize diff
                    attr_list.extend([[time_diff_norm, 1.0], [time_diff_norm, 0.0]]) # Diff + direction bit

        if edge_list:
             edge_index_dict[('transaction', 'temporal', 'transaction')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
             edge_attr_dict[('transaction', 'temporal', 'transaction')] = torch.tensor(attr_list, dtype=torch.float)
             self.edge_feature_dims[('transaction', 'temporal', 'transaction')] = edge_attr_dict[('transaction', 'temporal', 'transaction')].shape[1]
        else:
             print("[WARN] No temporal edges created.")


        # 4. Transaction -> Transaction (similar_amount) - Use k-NN for efficiency
        # This also needs optimization for large scale (e.g., approximate NN or locality sensitive hashing)
        # Using simple brute-force k-NN here for demonstration.
        # Note: This creates edges based on feature similarity, potentially dense.
        edge_list = []
        attr_list = []
        amounts = data['transaction'].x[:, 0].numpy() # First feature is amount
        from sklearn.neighbors import NearestNeighbors
        k_neighbors = 5 # Define number of neighbors
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='minkowski', p=1, algorithm='auto') # L1 distance (abs diff)
        nn.fit(amounts.reshape(-1, 1))
        distances, indices = nn.kneighbors(amounts.reshape(-1, 1))
        
        for i in range(num_transactions):
            for k in range(1, k_neighbors + 1): # Skip self (index 0)
                j = indices[i, k]
                dist = distances[i, k]
                # Ensure i, j are valid node indices from the original mapping
                # The indices from NN correspond directly to the order in 'amounts' which is 0..N-1
                # Assuming tx_map maps original df index to 0..N-1
                tx_node_i = i
                tx_node_j = j
        
                edge_list.extend([[tx_node_i, tx_node_j], [tx_node_j, tx_node_i]])
                amount_ratio = min(amounts[i], amounts[j]) / (max(amounts[i], amounts[j]) + 1e-6)
                attr_list.extend([[dist, amount_ratio, 1.0], [dist, amount_ratio, 0.0]]) # Diff, Ratio, Direction
        
        if edge_list:
             edge_index_dict[('transaction', 'similar_amount', 'transaction')] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
             edge_attr_dict[('transaction', 'similar_amount', 'transaction')] = torch.tensor(attr_list, dtype=torch.float)
             self.edge_feature_dims[('transaction', 'similar_amount', 'transaction')] = edge_attr_dict[('transaction', 'similar_amount', 'transaction')].shape[1]
        else:
             print("[WARN] No similar_amount edges created (k-NN).")


        # --- Assign to HeteroData Object ---
        for edge_type, index_tensor in edge_index_dict.items():
            data[edge_type].edge_index = index_tensor
            if edge_type in edge_attr_dict:
                data[edge_type].edge_attr = edge_attr_dict[edge_type]

        self.graph_data = data


    def _prepare_and_add_sequences(self):
        """Prepare sequence data (features of previous transactions) for each transaction."""
        if self.graph_data is None:
            raise RuntimeError("Graph data not built. Call _build_graph() first.")

        df_sorted = self.transactions_df.sort_values(['user_id', 'timestamp'])
        user_groups = df_sorted.groupby('user_id')

        all_seq_features = []
        all_seq_lengths = []
        feature_cols = ['amount', 'weekday', 'hour'] # Features for sequence model input
        self.sequence_feature_dim = len(feature_cols) + 1 # +1 for time delta

        # Create mapping from original df index to its position in the sorted df
        original_to_sorted_pos = {idx: i for i, idx in enumerate(df_sorted.index)}

        # Iterate through the transactions in the order they appear in self.graph_data['transaction']
        # which corresponds to the original self.transactions_df order before sorting.
        for orig_idx in self.graph_data['transaction'].original_index.tolist():
            
            # Find the position of this transaction in the user's sorted sequence
            current_pos_in_sorted = original_to_sorted_pos[orig_idx]
            current_row = df_sorted.iloc[current_pos_in_sorted]
            user_id = current_row['user_id']
            
            # Get the group for this user
            user_group_df = user_groups.get_group(user_id)
            
            # Find the index of the current transaction WITHIN the user's group
            current_pos_in_group = user_group_df.index.get_loc(orig_idx)

            # Select previous transactions within the max_seq_length window
            start_idx = max(0, current_pos_in_group - self.max_seq_length)
            prev_txs_group = user_group_df.iloc[start_idx:current_pos_in_group]

            seq_features_for_tx = []
            if not prev_txs_group.empty:
                current_time = current_row['timestamp']
                for _, prev_row in prev_txs_group.iterrows():
                    time_delta = (current_time - prev_row['timestamp']).total_seconds()
                    # Normalize time_delta? Maybe log scale? For now, just seconds.
                    feat = [prev_row[col] for col in feature_cols] + [time_delta]
                    seq_features_for_tx.append(feat)

            # Convert to tensor
            if seq_features_for_tx:
                seq_tensor = torch.tensor(seq_features_for_tx, dtype=torch.float)
                seq_len = len(seq_tensor)
            else:
                # Handle case with no previous transactions
                seq_tensor = torch.zeros((0, self.sequence_feature_dim), dtype=torch.float)
                seq_len = 0

            all_seq_features.append(seq_tensor)
            all_seq_lengths.append(seq_len)

        # Pad sequences for the entire dataset
        # Note: This pads all sequences to the longest sequence *in the dataset*.
        # If memory is an issue, padding per-batch in a collate_fn might be needed,
        # but NeighborLoader doesn't easily support custom collate for this.
        padded_sequences = pad_sequence(all_seq_features, batch_first=True, padding_value=0.0)

        # Add to graph data object
        self.graph_data['transaction'].seq_features = padded_sequences
        self.graph_data['transaction'].seq_lengths = torch.tensor(all_seq_lengths, dtype=torch.long)
        print(f"Added sequence features to graph. Padded shape: {padded_sequences.shape}")


    def _prepare_and_add_text(self):
        """Tokenize text fields and add input_ids/attention_mask to graph."""
        if self.graph_data is None or self.tokenizer is None:
             raise RuntimeError("Graph data or tokenizer not initialized.")

        # Define text fields to use
        text_fields_to_process = ['description', 'memo', 'merchant_name']
        processed_tokens = {}

        for field in text_fields_to_process:
             # Get text, handle potential NaNs by converting to empty string
             texts = self.transactions_df[field].fillna('').astype(str).tolist()
             # Replace truly empty strings with a space for tokenizer
             texts = [t if t.strip() else " " for t in texts]

             print(f"Tokenizing field: '{field}' for {len(texts)} transactions...")
             tokens = self.tokenizer(
                 texts,
                 padding='max_length', # Pad to max_length specified
                 truncation=True,
                 max_length=self.text_max_length,
                 return_tensors='pt' # Return PyTorch tensors
             )
             processed_tokens[f'{field}_input_ids'] = tokens['input_ids']
             processed_tokens[f'{field}_attention_mask'] = tokens['attention_mask']
             print(f"Tokenized '{field}'. Shape: {tokens['input_ids'].shape}")

        # Add tokenized tensors to the graph data object
        for key, tensor in processed_tokens.items():
             self.graph_data['transaction'][key] = tensor
        print("Added tokenized text features to graph.")

    def _split_data_and_add_masks(self):
        """Split data based on users and add boolean masks to graph_data."""
        if self.graph_data is None:
             raise RuntimeError("Graph data not built.")

        num_transactions = self.graph_data['transaction'].num_nodes
        # Ensure mapping from node index (0..N-1) back to user_id
        node_idx_to_user_id = self.transactions_df['user_id'].values # Assumes df index maps 0..N-1

        unique_users = np.unique(node_idx_to_user_id)
        np.random.shuffle(unique_users) # Shuffle users for splitting
        n_users = len(unique_users)

        n_val = int(n_users * self.val_ratio)
        n_test = int(n_users * self.test_ratio)
        if n_val == 0 or n_test == 0:
            print(f"[WARN] Very few users ({n_users}). val_ratio/test_ratio might result in 0 users for val/test sets.")
            n_val = max(1, n_val) if n_users > 1 else 0
            n_test = max(1, n_test) if n_users > 1 + n_val else 0


        val_users = set(unique_users[:n_val])
        test_users = set(unique_users[n_val : n_val + n_test])
        train_users = set(unique_users[n_val + n_test :])

        # Create boolean masks based on user assignment
        train_mask = torch.zeros(num_transactions, dtype=torch.bool)
        val_mask = torch.zeros(num_transactions, dtype=torch.bool)
        test_mask = torch.zeros(num_transactions, dtype=torch.bool)

        for i in range(num_transactions):
            user_id = node_idx_to_user_id[i]
            if user_id in train_users:
                train_mask[i] = True
            elif user_id in val_users:
                val_mask[i] = True
            elif user_id in test_users:
                test_mask[i] = True

        # Add masks to graph data
        self.graph_data['transaction'].train_mask = train_mask
        self.graph_data['transaction'].val_mask = val_mask
        self.graph_data['transaction'].test_mask = test_mask


    def _create_loader(self, mask: torch.Tensor, shuffle: bool) -> NeighborLoader:
        """Helper to create NeighborLoader for a given mask."""
        if self.graph_data is None:
            raise RuntimeError("Graph data not loaded. Call setup() first.")
        if mask.sum() == 0:
            print(f"[WARN] Mask for loader has zero nodes selected. Returning None for this dataloader.")
            return None

        # Specify the node type and the mask defining the seed nodes for this loader
        input_nodes = ('transaction', mask)

        loader = NeighborLoader(
            self.graph_data,
            num_neighbors=self.num_neighbors, # Define neighbors per layer
            shuffle=shuffle,
            batch_size=self.batch_size,
            input_nodes=input_nodes, # Seed nodes are transactions defined by the mask
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            # NeighborLoader handles collation internally for graph structure.
            # Features attached to nodes (like sequences/text tokens) should be sliced automatically.
        )
        print(f"Created NeighborLoader with {mask.sum()} seed nodes.")
        return loader

    # --- Dataloader Methods ---
    def train_dataloader(self) -> Optional[NeighborLoader]:
        print("Creating train NeighborLoader...")
        if self.graph_data is None: self.setup('fit')
        return self._create_loader(self.graph_data['transaction'].train_mask, shuffle=True)

    def val_dataloader(self) -> Optional[NeighborLoader]:
        print("Creating validation NeighborLoader...")
        if self.graph_data is None: self.setup('fit') # Ensure data is setup
        return self._create_loader(self.graph_data['transaction'].val_mask, shuffle=False)

    def test_dataloader(self) -> Optional[NeighborLoader]:
        print("Creating test NeighborLoader...")
        if self.graph_data is None: self.setup('test') # Ensure data is setup
        return self._create_loader(self.graph_data['transaction'].test_mask, shuffle=False)
