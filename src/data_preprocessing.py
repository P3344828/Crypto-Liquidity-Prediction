"""
Data Preprocessing for Cryptocurrency Liquidity Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and merge all CSV files from the raw data directory.
    
    Args:
        data_dir: Path to the raw data directory
        
    Returns:
        Merged DataFrame containing all cryptocurrency data
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"Loaded {file.name}: {len(df)} records")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records after merging: {len(merged_df)}")
    
    return merged_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and data inconsistencies.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Rename columns for consistency
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Handle column name variations
    column_mapping = {
        'mkt_cap': 'market_cap',
        '24h_volume': 'volume_24h',
        '1h': 'change_1h',
        '24h': 'change_24h',
        '7d': 'change_7d'
    }
    df_clean = df_clean.rename(columns=column_mapping)
    
    # Convert date column to datetime
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Handle missing values
    print("\nMissing values before cleaning:")
    print(df_clean.isnull().sum())
    
    # For percentage changes, fill with 0 (no change)
    percentage_cols = ['change_1h', 'change_24h', 'change_7d']
    for col in percentage_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # For volume and market_cap, forward fill then backward fill
    numeric_cols = ['price', 'volume_24h', 'market_cap']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].ffill().bfill()
    
    # Remove rows where critical columns are still missing
    critical_cols = ['coin', 'symbol', 'price', 'volume_24h', 'market_cap']
    existing_critical = [col for col in critical_cols if col in df_clean.columns]
    df_clean = df_clean.dropna(subset=existing_critical)
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Ensure numeric columns are properly typed
    for col in numeric_cols + percentage_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    print("\nMissing values after cleaning:")
    print(df_clean.isnull().sum())
    print(f"\nFinal dataset shape: {df_clean.shape}")
    
    return df_clean


def normalize_features(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Normalize numerical features using Min-Max scaling.
    
    Args:
        df: DataFrame to normalize
        columns: List of columns to normalize. If None, normalizes all numeric columns.
        
    Returns:
        DataFrame with normalized features (original + normalized columns)
    """
    from sklearn.preprocessing import MinMaxScaler
    
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude percentage columns from normalization
        columns = [c for c in columns if 'change' not in c]
    
    scaler = MinMaxScaler()
    
    for col in columns:
        if col in df_normalized.columns:
            df_normalized[f'{col}_normalized'] = scaler.fit_transform(
                df_normalized[[col]].values
            )
    
    return df_normalized


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/crypto_processed.csv"):
    """
    Save the processed DataFrame to CSV.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the processed data
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")


def main():
    """Main function to run the data preprocessing pipeline."""
    print("=" * 60)
    print("CRYPTOCURRENCY DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading raw data...")
    df_raw = load_raw_data()
    
    # Clean data
    print("\n[2/4] Cleaning data...")
    df_clean = clean_data(df_raw)
    
    # Normalize features
    print("\n[3/4] Normalizing features...")
    df_processed = normalize_features(df_clean, columns=['price', 'volume_24h', 'market_cap'])
    
    # Save processed data
    print("\n[4/4] Saving processed data...")
    save_processed_data(df_processed)
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return df_processed


if __name__ == "__main__":
    main()
