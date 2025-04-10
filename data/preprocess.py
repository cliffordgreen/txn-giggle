import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import json
import os
import glob

def load_data(data_path: str, max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Load transaction data. 
    If data_path is a directory, loads and concatenates all .parquet files within it,
    up to max_files if specified.
    If data_path is a file, loads the specified CSV or parquet file.
    """
    if os.path.isdir(data_path):
        parquet_files = sorted(glob.glob(os.path.join(data_path, '*.parquet'))) # Sort for consistency
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in directory: {data_path}")
        
        if max_files is not None and max_files > 0:
            print(f"Limiting to {max_files} files out of {len(parquet_files)} found.")
            parquet_files = parquet_files[:max_files]
        
        df_list = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded and concatenated {len(parquet_files)} parquet files from {data_path}")
        
    elif os.path.isfile(data_path):
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            print(f"Loaded single parquet file: {data_path}")
        elif data_path.endswith('.csv'):
             df = pd.read_csv(data_path)
             print(f"Loaded single CSV file: {data_path}")
        else:
             raise ValueError(f"Unsupported file type: {data_path}. Only .parquet and .csv are supported.")
    else:
        raise FileNotFoundError(f"Path not found or is not a valid file/directory: {data_path}")

    #df = df.head(1000)
    # Use posted_date as timestamp if available
    if 'posted_date' in df.columns:
        df['timestamp'] = df['posted_date']
    elif 'books_create_timestamp' in df.columns:
        df['timestamp'] = df['books_create_timestamp']
    else:
        print("Warning: No timestamp column found, generating dummy timestamps")
        start_date = pd.to_datetime('2022-01-01')
        end_date = pd.to_datetime('2023-01-01')
        time_range = (end_date - start_date).days
        random_days = np.random.randint(0, time_range, size=len(df))
        df['timestamp'] = start_date + pd.to_timedelta(random_days, unit='D')
    
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
    # Log transform amount - Check if 'amount' can be < 0. log1p handles 0 but not <= -1.
    # Consider adding: df['amount'] = df['amount'].clip(lower=0) if negative amounts are possible errors
    df['amount_log'] = np.log1p(df['amount'])

    return df    


def extract_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from merchant information."""
    # Merchant frequency
    merchant_freq = df['merchant_name'].value_counts()
    df['merchant_frequency'] = df['merchant_name'].map(merchant_freq)
    
    # Merchant categories (if available)
    if 'merchant_category' in df.columns:
        df['merchant_category_frequency'] = df.groupby('merchant_category')['merchant_category'].transform('count')
    
    return df

def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from text fields. Handles missing columns."""
    # Ensure 'description' column exists
    if 'description' not in df.columns:
        print("Warning: 'description' column not found. Creating empty column.")
        df['description'] = ''
    # Ensure 'memo' column exists
    if 'memo' not in df.columns:
        print("Warning: 'memo' column not found. Creating empty column.")
        df['memo'] = ''
        
    # Fill NaNs in text columns before processing
    df['description'] = df['description'].fillna('')
    df['memo'] = df['memo'].fillna('')

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

def scale_numerical_features(
    df: pd.DataFrame,
    numerical_columns: List[str],
    scalers: Optional[Dict[str, StandardScaler]] = None
) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """Scale numerical features using StandardScaler."""
    if scalers is None:
        scalers = {}
    
    for col in numerical_columns:
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
    numerical_columns: Optional[List[str]] = None,
    max_files: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], Dict[str, StandardScaler]]:
    """Preprocess transaction data and engineer features."""
    # Load data
    df = load_data(data_path, max_files=max_files)
    
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
            'industry_name',
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
            'merchant_frequency',
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
        # Filter numerical columns to only include those present in the DataFrame
        numerical_columns = [col for col in numerical_columns if col in df.columns]
    
    # Handle missing values only for existing columns
    default_fill_values = {
        'description': '',
        'memo': '',
        'merchant_name': 'unknown',
        'merchant_city': 'unknown',
        'merchant_state': 'unknown',
        'mcc_name': 'unknown',
        'account_type_id': 'unknown',
        'tax_account_type': 'unknown',
        'company_name': 'unknown',
        'industry_name': 'unknown',
        'region_name': 'unknown',
        'language_name': 'unknown',
        'category_name': 'unknown'
    }
    
    # Create a dictionary with fill values only for columns present in the DataFrame
    fill_values_for_existing_cols = {col: val for col, val in default_fill_values.items() if col in df.columns}
    
    df = df.fillna(fill_values_for_existing_cols)
    
    # Encode categorical features
    df, label_encoders = encode_categorical_features(df, categorical_columns)
    
    # Scale numerical features
    df, scalers = scale_numerical_features(df, numerical_columns)
    
    # Save preprocessed data
    df.to_csv(output_path, index=False)
    
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
    label_mapping_path: Optional[str] = None,
    max_files: Optional[int] = None
):
    """Main function for data preprocessing."""
    # Preprocess data
    df, label_encoders, scalers = preprocess_data(
        data_path=data_path,
        output_path=output_path,
        label_mapping_path=label_mapping_path,
        max_files=max_files
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
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of parquet files to load from a directory')
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        output_path=args.output_path,
        label_mapping_path=args.label_mapping_path,
        max_files=args.max_files
    ) 