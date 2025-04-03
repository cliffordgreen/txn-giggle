import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import json
import os

def load_data(data_path: str) -> pd.DataFrame:
    """Load transaction data from CSV or parquet file."""
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Use posted_date as timestamp if available
    if 'posted_date' in df.columns:
        df['timestamp'] = df['posted_date']
    else: 
        df['timestamp'] = df['books_create_timestamp'] 
    

    # Convert timestamp to datetime and handle parsing errors
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Check if there are any NaT values and inform the user
    nat_count = df['timestamp'].isna().sum()
    if nat_count > 0:
        print(f"Warning: Found {nat_count} rows with invalid timestamps. These will be removed.")
        
    # Remove rows with NaT timestamps
    df = df.dropna(subset=['timestamp'])
    df = df.drop('is_uncat_fdp_logic', axis=1)
    return df

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract temporal features
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    
    # Extract time-based features
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['is_morning'] = ((df['hour'] >= 5) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 17)).astype(int)
    df['is_evening'] = ((df['hour'] >= 17) & (df['hour'] < 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 5)).astype(int)
    
    return df

    
def extract_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from transaction amount."""
    # Log transform amount (ensure amounts are positive)
    df['amount_log'] = np.log1p(np.maximum(0, df['amount']))
    
    # Amount statistics per user
    user_amount_stats = df.groupby('user_id')['amount'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Rename columns
    user_amount_stats.columns = [
        'user_id',
        'user_amount_mean',
        'user_amount_std',
        'user_amount_min',
        'user_amount_max',
        'user_transaction_count'
    ]
    
    # Ensure std is never zero to avoid division by zero
    user_amount_stats['user_amount_std'] = user_amount_stats['user_amount_std'].replace(0, 1)
    
    # Merge with original dataframe
    df = df.merge(user_amount_stats, on='user_id', how='left')
    
    # Amount relative to user statistics (with safe division)
    df['amount_relative_to_mean'] = df['amount'] / df['user_amount_mean'].replace(0, 1)
    df['amount_relative_to_std'] = (df['amount'] - df['user_amount_mean']) / df['user_amount_std']
    
    # Handle percentile calculation safely
    try:
        df['amount_percentile'] = df.groupby('user_id')['amount'].transform(
            lambda x: pd.qcut(x, q=min(10, len(x)), labels=False, duplicates='drop') if len(x) > 1 else 0
        )
    except Exception:
        df['amount_percentile'] = 0
    
    return df
# def extract_amount_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Extract features from transaction amount."""
#     # Log transform amount
#     df['amount_log'] = np.log1p(df['amount'])
    
#     # Amount statistics per user
#     user_amount_stats = df.groupby('user_id')['amount'].agg([
#         'mean', 'std', 'min', 'max', 'count'
#     ]).reset_index()
    
#     # Rename columns
#     user_amount_stats.columns = [
#         'user_id',
#         'user_amount_mean',
#         'user_amount_std',
#         'user_amount_min',
#         'user_amount_max',
#         'user_transaction_count'
#     ]
    
#     # Merge with original dataframe
#     df = df.merge(user_amount_stats, on='user_id', how='left')
    
#     # Amount relative to user statistics
#     df['amount_relative_to_mean'] = df['amount'] / df['user_amount_mean']
#     df['amount_relative_to_std'] = (df['amount'] - df['user_amount_mean']) / df['user_amount_std']
    
#     # Amount percentiles
#     df['amount_percentile'] = df.groupby('user_id')['amount'].transform(
#         lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
#     )
    
#     return df

def extract_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from merchant information."""
    # Merchant frequency
    merchant_freq = df['merchant_name'].value_counts()
    df['merchant_frequency'] = df['merchant_name'].map(merchant_freq)
    
    # Merchant frequency per user
    user_merchant_freq = df.groupby(['user_id', 'merchant_name']).size().reset_index(name='user_merchant_frequency')
    df = df.merge(user_merchant_freq, on=['user_id', 'merchant_name'], how='left')
    
    # Merchant categories (if available)
    if 'merchant_category' in df.columns:
        df['merchant_category_frequency'] = df.groupby('merchant_category')['merchant_category'].transform('count')
    
    return df

def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from text fields."""
    # Text length features
    df['description_length'] = df['description'].str.len()
    df['memo_length'] = df['memo'].str.len()
    
    # Word count features
    df['description_word_count'] = df['description'].str.split().str.len()
    df['memo_word_count'] = df['memo'].str.split().str.len()
    
    # Character type features
    df['description_digit_count'] = df['description'].str.count(r'\d')
    df['description_uppercase_count'] = df['description'].str.count(r'[A-Z]')
    df['description_special_count'] = df['description'].str.count(r'[^a-zA-Z0-9\s]')
    
    df['memo_digit_count'] = df['memo'].str.count(r'\d')
    df['memo_uppercase_count'] = df['memo'].str.count(r'[A-Z]')
    df['memo_special_count'] = df['memo'].str.count(r'[^a-zA-Z0-9\s]')
    
    return df

def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    label_encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical features using LabelEncoder."""
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_columns:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))
        else:
            # Handle unknown categories
            known_categories = set(label_encoders[col].classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: label_encoders[col].transform([x])[0] if x in known_categories else -1
            )
    
    return df, label_encoders

# def scale_numerical_features(
#     df: pd.DataFrame,
#     numerical_columns: List[str],
#     scalers: Optional[Dict[str, StandardScaler]] = None
# ) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
#     """Scale numerical features using StandardScaler."""
#     if scalers is None:
#         scalers = {}
    
#     for col in numerical_columns:
#         if col not in scalers:
#             scalers[col] = StandardScaler()
#             df[col] = scalers[col].fit_transform(df[[col]])
#         else:
#             df[col] = scalers[col].transform(df[[col]])
    
#     return df, scalers

def scale_numerical_features(
    df: pd.DataFrame,
    numerical_columns: List[str],
    scalers: Optional[Dict[str, StandardScaler]] = None
) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """Scale numerical features using StandardScaler."""
    if scalers is None:
        scalers = {}
    
    for col in numerical_columns:
        if col not in df.columns:
            continue
            
        # Replace infinity values with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column mean or 0
        if df[col].isna().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(0 if pd.isna(mean_val) else mean_val)
        
        # Now perform scaling
        if col not in scalers:
            scalers[col] = StandardScaler()
            df[col] = scalers[col].fit_transform(df[[col]])
        else:
            df[col] = scalers[col].transform(df[[col]])
    
    return df, scalers

def preprocess_data(
    data_path: str,
    output_path: str,
    label_mapping_path: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], Dict[str, StandardScaler]]:
    """Preprocess transaction data and engineer features."""
    # Load data
    df = load_data(data_path)
    
    # Extract features
    df = extract_temporal_features(df)
    df = extract_amount_features(df)
    df = extract_merchant_features(df)
    df = extract_text_features(df)
    
    # Define default categorical columns
    if categorical_columns is None:
        categorical_columns = [
            'merchant_name',
            'merchant_city',
            'merchant_state',
            'description',
            'memo',
            'mcc_name',
            'account_type_id',
            'tax_account_type',
            'company_name',
            'industry_code',
            'region_name',
            'language_name',
            'category_name',
            'category_id',
            'user_category_id'
            
            
        ]
        # Filter out columns that don't exist in the dataframe
        categorical_columns = [col for col in categorical_columns if col in df.columns]
    
    # Define default numerical columns
    if numerical_columns is None:
        numerical_columns = [
            'amount',
            'amount_log',
            'user_amount_mean',
            'user_amount_std',
            'user_amount_min',
            'user_amount_max',
            'user_transaction_count',
            'amount_relative_to_mean',
            'amount_relative_to_std',
            'amount_percentile',
            'merchant_frequency',
            'user_merchant_frequency',
            'description_length',
            'memo_length',
            'description_word_count',
            'memo_word_count',
            'description_digit_count',
            'description_uppercase_count',
            'description_special_count',
            'memo_digit_count',
            'memo_uppercase_count',
            'memo_special_count'
        ]
    
    # Handle missing values
    df = df.fillna({
        'description': '',
        'memo': '',
        'merchant_name': 'unknown',
        'merchant_city': 'unknown',
        'merchant_state': 'unknown',
        'mcc_name': 'unknown',
        'account_type_id': 'unknown',
        'tax_account_type': 'unknown',
        'company_name': 'unknown',
        'industry_code': 'unknown',
        'region_name': 'unknown',
        'language_name': 'unknown',
        'category_name': 'unknown'
    })
    
    # Encode categorical features
    df, label_encoders = encode_categorical_features(df, categorical_columns)
    
    # Scale numerical features
    df, scalers = scale_numerical_features(df, numerical_columns)

    if output_path.endswith('/') or os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, 'processed_data.csv')
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_file = output_path
    
    # Save preprocessed data
    df.to_csv(output_file, index=False)

    
    # # Save preprocessed data
    # df.to_csv(output_path, index=False)
    
    # Save label mappings if provided
    if label_mapping_path:
        label_mappings = {}
        for col in label_encoders:
            mapping = {}
            for class_, idx in zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)):
                mapping[class_] = int(idx)  # Convert numpy.int64 to Python int
            label_mappings[col] = mapping
        
        with open(label_mapping_path, 'w') as f:
            json.dump(label_mappings, f, indent=2)
    
    return df, label_encoders, scalers

def main(
    data_path: str,
    output_path: str,
    label_mapping_path: Optional[str] = None
):
    """Main function for data preprocessing."""
    # Preprocess data
    df, label_encoders, scalers = preprocess_data(
        data_path=data_path,
        output_path=output_path,
        label_mapping_path=label_mapping_path
    )
    
    print(f"Preprocessed data saved to {output_path}")
    if label_mapping_path:
        print(f"Label mappings saved to {label_mapping_path}")
    
    print("\nFeature Statistics:")
    print(df.describe())

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess transaction data')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to input data file (CSV or parquet)')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save preprocessed data')
    parser.add_argument('--label_mapping_path', type=str, default=None,
                      help='Path to save label mappings')
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        output_path=args.output_path,
        label_mapping_path=args.label_mapping_path
    ) 