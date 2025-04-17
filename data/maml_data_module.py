import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split # For splitting users
from typing import Dict, List, Optional, Tuple, Any, Union
import random

class MAMLTaskDataset(Dataset):
    """
    A PyTorch Dataset that generates tasks for MAML.
    Each task corresponds to a single user and contains support and query sets.
    """
    def __init__(self,
                 transactions_df: pd.DataFrame,
                 user_ids: List[Any], # List of unique user IDs for this split (meta-train/val/test)
                 K_shot: int,         # Number of support examples per task
                 Q_query: int,        # Number of query examples per task (-1 for all remaining)
                 label_column: str = 'category_id', # Column containing the labels
                 user_id_column: str = 'user_id',   # Column containing user IDs
                 timestamp_column: Optional[str] = 'timestamp' # Optional: For chronological sampling
                 ):
        """
        Args:
            transactions_df: DataFrame containing all transaction data.
            user_ids: List of unique user IDs included in this dataset split.
            K_shot: Number of support examples per task.
            Q_query: Number of query examples per task. If -1, use all non-support examples.
            label_column: Name of the column containing the target labels.
            user_id_column: Name of the column containing user identifiers.
            timestamp_column: Optional name of the timestamp column for sorting.
        """
        super().__init__()
        self.df = transactions_df
        self.user_ids = user_ids
        self.K_shot = K_shot
        self.Q_query = Q_query
        self.label_column = label_column
        self.user_id_column = user_id_column
        self.timestamp_column = timestamp_column

        # Pre-filter DataFrame for faster lookups during __getitem__
        self.df_filtered = self.df[self.df[self.user_id_column].isin(self.user_ids)].copy()

        # Optional: Sort by user and time if timestamp column provided
        if self.timestamp_column and self.timestamp_column in self.df_filtered.columns:
             self.df_filtered.sort_values([self.user_id_column, self.timestamp_column], inplace=True)

        self.df_filtered.reset_index(inplace=True) # Keep original index if needed, rename to 'original_index'
        self.df_filtered.rename(columns={'index': 'original_index'}, inplace=True)

        # Group by user for efficient task sampling
        self.user_groups = self.df_filtered.groupby(self.user_id_column)
        self.user_indices = {user: group.index.tolist() for user, group in self.user_groups.groups.items()}

        # Filter users with insufficient data for K_shot + 1 query item
        min_required = self.K_shot + 1
        self.valid_user_ids = [
             uid for uid in self.user_ids if len(self.user_indices.get(uid, [])) >= min_required
        ]

        if len(self.valid_user_ids) < len(self.user_ids):
            print(f"[WARN] MAMLTaskDataset: Filtered out {len(self.user_ids) - len(self.valid_user_ids)} users "
                  f"with < {min_required} transactions.")
        if not self.valid_user_ids:
             print("[ERROR] MAMLTaskDataset: No users remaining after filtering for K_shot + 1 examples.")
             # raise ValueError("No valid users found for task creation.") # Optionally raise error


    def __len__(self) -> int:
        """Returns the number of tasks (valid users)."""
        return len(self.valid_user_ids)

    def __getitem__(self, index: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generates a single MAML task for the user at the given index.

        Returns:
            A dictionary containing:
                'support': (support_indices, support_labels)
                'query': (query_indices, query_labels)
            Where indices are the *original* DataFrame indices of the transactions.
        """
        user_id = self.valid_user_ids[index]
        user_data_indices = self.user_indices[user_id] # Get pre-grouped indices

        # Sample K_shot indices for support set
        if len(user_data_indices) < self.K_shot:
            # Should not happen due to __init__ filtering, but handle defensively
            print(f"[WARN] User {user_id} has < K_shot ({self.K_shot}) samples ({len(user_data_indices)}), sampling with replacement or skipping.")
            # Option 1: Sample with replacement (might be okay for MAML)
            support_indices_rel = random.choices(user_data_indices, k=self.K_shot)
            # Option 2: Return empty tensors or skip (would require filtering in collate_fn)
            # support_indices_rel = []
        else:
            support_indices_rel = random.sample(user_data_indices, self.K_shot)

        # Get remaining indices for the query pool
        query_pool_indices_rel = [idx for idx in user_data_indices if idx not in support_indices_rel]

        # Sample Q_query indices for the query set from the pool
        if not query_pool_indices_rel:
            # Should not happen due to __init__ filtering (needs K+1)
             print(f"[WARN] User {user_id} has no samples left for query set after taking K_shot={self.K_shot}.")
             query_indices_rel = []
        elif self.Q_query == -1 or self.Q_query >= len(query_pool_indices_rel):
            # Use all remaining if Q_query is -1 or larger than available
            query_indices_rel = query_pool_indices_rel
        else:
            query_indices_rel = random.sample(query_pool_indices_rel, self.Q_query)


        # --- Get Original DataFrame Indices and Labels ---
        # Retrieve data using the relative indices obtained from sampling
        support_df_slice = self.df_filtered.loc[support_indices_rel]
        query_df_slice = self.df_filtered.loc[query_indices_rel]

        # Extract original indices (important for model to fetch correct features)
        support_original_indices = torch.tensor(support_df_slice['original_index'].values, dtype=torch.long)
        query_original_indices = torch.tensor(query_df_slice['original_index'].values, dtype=torch.long)

        # Extract labels
        support_labels = torch.tensor(support_df_slice[self.label_column].values, dtype=torch.long)
        query_labels = torch.tensor(query_df_slice[self.label_column].values, dtype=torch.long)

        # Basic check
        if len(support_original_indices) != self.K_shot and len(user_data_indices) >= self.K_shot :
             print(f"[WARN] Mismatch in support set size for user {user_id}. Expected {self.K_shot}, Got {len(support_original_indices)}")
        if self.Q_query != -1 and len(query_original_indices) != min(self.Q_query, len(query_pool_indices_rel)) and query_pool_indices_rel:
             print(f"[WARN] Mismatch in query set size for user {user_id}. Requested {self.Q_query}, Available {len(query_pool_indices_rel)}, Got {len(query_original_indices)}")


        return {
            'support': (support_original_indices, support_labels),
            'query': (query_original_indices, query_labels),
            'user_id': user_id # Optional: Include user ID for debugging/logging
        }


class MAMLTransactionDataModule(pl.LightningDataModule):
    """
    DataModule for preparing and loading MAML tasks.
    Splits users into meta-train, meta-validation, and meta-test sets.
    """
    def __init__(self,
                 transactions_df: pd.DataFrame,
                 user_id_column: str = 'user_id',
                 label_column: str = 'category_id', # Global label
                 user_label_column: Optional[str] = 'user_category_id', # User-specific label (if used by MAML head)
                 timestamp_column: Optional[str] = 'books_create_timestamp',
                 K_shot: int = 5,
                 Q_query: int = 10, # Number of query samples per task (-1 for all remaining)
                 meta_batch_size: int = 16, # Number of tasks per batch
                 num_workers: int = 0,
                 meta_val_ratio: float = 0.15,
                 meta_test_ratio: float = 0.15,
                 seed: int = 42
                 ):
        """
        Args:
            transactions_df: DataFrame containing transaction data.
            user_id_column: Column name for user identifiers.
            label_column: Column name for the primary target labels (e.g., global category).
            user_label_column: Optional column name for user-specific labels. MAML might adapt towards this.
            timestamp_column: Optional column name for timestamps (for sorting).
            K_shot: Number of support examples per user/task.
            Q_query: Number of query examples per user/task.
            meta_batch_size: How many tasks (users) to include in a single batch.
            num_workers: Number of workers for DataLoader.
            meta_val_ratio: Fraction of USERS to use for meta-validation.
            meta_test_ratio: Fraction of USERS to use for meta-test.
            seed: Random seed for user splitting.
        """
        super().__init__()
        # Store essential configurations automatically
        self.save_hyperparameters('user_id_column', 'label_column', 'user_label_column',
                                'timestamp_column', 'K_shot', 'Q_query',
                                'meta_batch_size', 'num_workers',
                                'meta_val_ratio', 'meta_test_ratio', 'seed')

        self.transactions_df = transactions_df.copy()

        # Determine which label to use for MAML tasks
        self.maml_label_column = user_label_column if user_label_column and user_label_column in self.transactions_df else label_column
        print(f"[INFO] MAMLDataModule: Using '{self.maml_label_column}' as the target label for MAML tasks.")

        # Placeholders for datasets and user splits
        self.meta_train_users: Optional[List[Any]] = None
        self.meta_val_users: Optional[List[Any]] = None
        self.meta_test_users: Optional[List[Any]] = None
        self.train_dataset: Optional[MAMLTaskDataset] = None
        self.val_dataset: Optional[MAMLTaskDataset] = None
        self.test_dataset: Optional[MAMLTaskDataset] = None
        self.user_map: Optional[Dict[int, Any]] = None
        self.num_users: int = 0

    def setup(self, stage: Optional[str] = None):
        """
        Splits users into meta-train, meta-val, and meta-test sets.
        Creates MAMLTaskDataset for each split.
        """
        if self.train_dataset is not None and self.val_dataset is not None:
            print("MAMLDataModule already set up.")
            return

        print(f"--- Starting MAMLDataModule Setup for stage: {stage} ---")

        # --- 1. Preprocess DataFrame (Basic: ensure types, handle NaNs) ---
        # Convert timestamp
        if self.hparams.timestamp_column and self.hparams.timestamp_column in self.transactions_df.columns:
            self.transactions_df[self.hparams.timestamp_column] = pd.to_datetime(
                self.transactions_df[self.hparams.timestamp_column], errors='coerce'
            )
            # Basic NaN handling for timestamp (optional, improves sorting robustness)
            if self.transactions_df[self.hparams.timestamp_column].isnull().any():
                 median_ts = self.transactions_df[self.hparams.timestamp_column].dropna().median()
                 if pd.isna(median_ts): median_ts = pd.Timestamp('now')
                 self.transactions_df[self.hparams.timestamp_column].fillna(median_ts, inplace=True)
        else:
             print(f"[WARN] Timestamp column '{self.hparams.timestamp_column}' not found or specified. Chronological sampling within task disabled.")
             self.hparams.timestamp_column = None # Disable if not valid

        # Ensure label column exists and handle NaNs (e.g., fill with a specific value like -1 or 'UNKNOWN')
        if self.maml_label_column not in self.transactions_df.columns:
             raise ValueError(f"MAML target label column '{self.maml_label_column}' not found in DataFrame.")
        # Example NaN handling: Fill with -1, assuming labels are non-negative integers
        if self.transactions_df[self.maml_label_column].isnull().any():
             print(f"[WARN] MAML label column '{self.maml_label_column}' contains NaNs. Filling with -1.")
             self.transactions_df[self.maml_label_column] = self.transactions_df[self.maml_label_column].fillna(-1)
        # Convert labels to integer type
        try:
            self.transactions_df[self.maml_label_column] = self.transactions_df[self.maml_label_column].astype(int)
            # Calculate number of classes based on max value + 1 (assuming 0-based)
            # Filter out potential negative fill values (-1)
            valid_labels = self.transactions_df[self.maml_label_column][self.transactions_df[self.maml_label_column] >= 0]
            if not valid_labels.empty:
                 self.num_maml_classes = valid_labels.max() + 1
            else:
                 self.num_maml_classes = 0
            print(f"Calculated num_maml_classes (int): {self.num_maml_classes}")
        except ValueError:
            print(f"[WARN] Could not convert MAML label column '{self.maml_label_column}' to int. Attempting factorization.")
            codes, uniques = pd.factorize(self.transactions_df[self.maml_label_column].astype(str), sort=True)
            self.transactions_df[self.maml_label_column] = codes
            self.num_maml_classes = len(uniques)
            print(f"Factorized '{self.maml_label_column}' into {self.num_maml_classes} codes.")


        # --- 2. Get Unique Users and Split ---
        all_users = self.transactions_df[self.hparams.user_id_column].unique()
        self.num_users = len(all_users)
        if self.num_users == 0:
            raise ValueError("No users found in the DataFrame.")

        print(f"Total unique users found: {self.num_users}")

        # Split users, not transactions
        train_val_users, self.meta_test_users = train_test_split(
            all_users,
            test_size=self.hparams.meta_test_ratio,
            random_state=self.hparams.seed
        )

        # Adjust val ratio relative to the remaining users
        # Ensure minimum 1 validation user if ratio > 0 and train_val has users
        if len(train_val_users) > 0 and self.hparams.meta_val_ratio > 0 and (1.0 - self.hparams.meta_test_ratio) > 0:
             val_ratio_adjusted = self.hparams.meta_val_ratio / (1.0 - self.hparams.meta_test_ratio)
             # Ensure at least 1 user for validation if possible and requested
             num_val_users = max(1, int(len(train_val_users) * val_ratio_adjusted))
             # Ensure val users doesn't exceed total available minus 1 for training
             num_val_users = min(num_val_users, max(0, len(train_val_users) - 1))
        else:
             val_ratio_adjusted = 0 # No validation split possible/requested
             num_val_users = 0

        if val_ratio_adjusted >= 1.0 or num_val_users == 0: # Handle edge case or impossibility
            if val_ratio_adjusted > 0: # Only warn if validation was expected
                 print("[WARN] Not enough users to create validation split based on ratios. Assigning all remaining to train.")
            self.meta_train_users = train_val_users
            self.meta_val_users = np.array([]) # Empty array
        elif len(train_val_users) < 2 : # Need at least 2 users to split into train/val
             print("[WARN] Not enough users remaining after test split to create a validation split. Assigning all to train.")
             self.meta_train_users = train_val_users
             self.meta_val_users = np.array([])
        else:
             self.meta_train_users, self.meta_val_users = train_test_split(
                 train_val_users,
                 test_size=val_ratio_adjusted,
                 random_state=self.hparams.seed
             )

        # Convert to lists
        self.meta_train_users = self.meta_train_users.tolist()
        self.meta_val_users = self.meta_val_users.tolist()
        self.meta_test_users = self.meta_test_users.tolist()


        print(f"Split users: #Meta-Train={len(self.meta_train_users)}, "
              f"#Meta-Val={len(self.meta_val_users)}, #Meta-Test={len(self.meta_test_users)}")

        # --- 3. Create Datasets ---
        dataset_args = {
            'transactions_df': self.transactions_df,
            'K_shot': self.hparams.K_shot,
            'Q_query': self.hparams.Q_query,
            'label_column': self.maml_label_column,
            'user_id_column': self.hparams.user_id_column,
            'timestamp_column': self.hparams.timestamp_column
        }

        print("Creating meta-training dataset...")
        self.train_dataset = MAMLTaskDataset(user_ids=self.meta_train_users, **dataset_args)
        if not self.train_dataset or len(self.train_dataset) == 0:
             print("[WARN] Meta-training dataset is empty after user filtering.")

        print("Creating meta-validation dataset...")
        self.val_dataset = MAMLTaskDataset(user_ids=self.meta_val_users, **dataset_args)
        if not self.val_dataset or len(self.val_dataset) == 0:
             print("[WARN] Meta-validation dataset is empty after user filtering.")

        print("Creating meta-test dataset...")
        self.test_dataset = MAMLTaskDataset(user_ids=self.meta_test_users, **dataset_args)
        if not self.test_dataset or len(self.test_dataset) == 0:
             print("[WARN] Meta-test dataset is empty after user filtering.")

        print(f"--- MAMLDataModule Setup finished ---")


    # --- DataLoader Methods ---
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None: self.setup('fit')
        if not self.train_dataset or len(self.train_dataset) == 0:
             print("[WARN] Returning None for train_dataloader as dataset is empty.")
             return None # Or raise error? Returning None might break Lightning Trainer
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.meta_batch_size,
                          shuffle=True, # Shuffle tasks (users)
                          num_workers=self.hparams.num_workers,
                          pin_memory=True if self.hparams.num_workers > 0 else False,
                          drop_last=True) # Drop last if not a full batch of tasks

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None: self.setup('fit')
        if not self.val_dataset or len(self.val_dataset) == 0:
             print("[INFO] Returning None for val_dataloader as dataset is empty.")
             return None
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.meta_batch_size,
                          shuffle=False, # No need to shuffle validation tasks
                          num_workers=self.hparams.num_workers,
                          pin_memory=True if self.hparams.num_workers > 0 else False,
                          drop_last=False) # Evaluate on all validation tasks

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None: self.setup('test')
        if not self.test_dataset or len(self.test_dataset) == 0:
             print("[INFO] Returning None for test_dataloader as dataset is empty.")
             return None
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.meta_batch_size,
                          shuffle=False, # No need to shuffle test tasks
                          num_workers=self.hparams.num_workers,
                          pin_memory=True if self.hparams.num_workers > 0 else False,
                          drop_last=False) # Evaluate on all test tasks

# Example Usage (Conceptual)
if __name__ == '__main__':
    # Create dummy data
    num_users = 50
    tx_per_user = 20
    data = []
    for u in range(num_users):
        user_id = f'user_{u:03d}'
        cat_offset = u % 5 # Simple category pattern per user
        for t in range(tx_per_user):
            timestamp = pd.Timestamp('2023-01-01') + pd.Timedelta(days=t + u*5, hours=random.randint(0,23))
            category = (cat_offset + random.randint(0, 2)) % 10 # Global category
            user_category = (cat_offset + random.randint(0, 1)) % 5 # User-specific category (fewer classes)
            amount = random.uniform(5, 100)
            data.append({'user_id': user_id, 'timestamp': timestamp, 'category_id': category, 'user_category_id': user_category, 'amount': amount})

    dummy_df = pd.DataFrame(data)
    print("Dummy DataFrame info:")
    dummy_df.info()
    print(dummy_df.head())

    # Instantiate DataModule
    maml_dm = MAMLTransactionDataModule(
        transactions_df=dummy_df,
        user_id_column='user_id',
        label_column='category_id',
        user_label_column='user_category_id', # Try adapting to user-specific labels
        timestamp_column='timestamp',
        K_shot=5,
        Q_query=5, # Use 5 query samples
        meta_batch_size=4, # 4 tasks (users) per batch
        meta_val_ratio=0.2,
        meta_test_ratio=0.2,
        num_workers=0
    )

    # Run setup
    maml_dm.setup()

    # Test dataloader
    print("\nTesting Train Dataloader...")
    train_loader = maml_dm.train_dataloader()
    if train_loader:
        for i, batch in enumerate(train_loader):
            print(f"--- Meta-Batch {i} ---")
            print(f"Number of tasks in batch: {len(batch['support'][0]) if isinstance(batch['support'], tuple) else 'N/A'}") # Check format if collate changes it
            # Example access (assuming default collate_fn keeps structure)
            first_task_support_indices = batch['support'][0][0]
            first_task_support_labels = batch['support'][1][0]
            first_task_query_indices = batch['query'][0][0]
            first_task_query_labels = batch['query'][1][0]
            first_task_user = batch['user_id'][0]

            print(f"  Task 0 (User: {first_task_user}):")
            print(f"    Support Indices ({len(first_task_support_indices)}): {first_task_support_indices.numpy()}")
            print(f"    Support Labels ({len(first_task_support_labels)}): {first_task_support_labels.numpy()}")
            print(f"    Query Indices ({len(first_task_query_indices)}): {first_task_query_indices.numpy()}")
            print(f"    Query Labels ({len(first_task_query_labels)}): {first_task_query_labels.numpy()}")

            if i >= 1: # Print first 2 batches
                break
    else:
        print("Train loader is None.")

    print("\nTesting Val Dataloader...")
    val_loader = maml_dm.val_dataloader()
    if val_loader:
         try:
             first_val_batch = next(iter(val_loader))
             print("Successfully fetched one batch from val_loader.")
             # Add similar print logic as above if needed
         except StopIteration:
             print("Val loader is empty.")
         except Exception as e:
             print(f"Error fetching from val_loader: {e}")
    else:
         print("Val loader is None.")

    print("\nTesting Test Dataloader...")
    test_loader = maml_dm.test_dataloader()
    if test_loader:
         try:
             first_test_batch = next(iter(test_loader))
             print("Successfully fetched one batch from test_loader.")
             # Add similar print logic as above if needed
         except StopIteration:
             print("Test loader is empty.")
         except Exception as e:
              print(f"Error fetching from test_loader: {e}")
    else:
         print("Test loader is None.") 