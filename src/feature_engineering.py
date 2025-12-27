"""
Feature Engineering for Cryptocurrency Liquidity Prediction

This creates liquidity-related features for cryptocurrency analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def calculate_liquidity_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the primary liquidity metric: Volume/MarketCap ratio.
    
    Higher ratio = Higher liquidity (easier to trade without price impact)
    
    Args:
        df: DataFrame with volume_24h and market_cap columns
        
    Returns:
        DataFrame with liquidity_ratio column added
    """
    df_features = df.copy()
    
    # Primary liquidity metric: 24h Volume / Market Cap
    df_features['liquidity_ratio'] = df_features['volume_24h'] / df_features['market_cap']
    
    # Handle infinite values (when market_cap is 0)
    df_features['liquidity_ratio'] = df_features['liquidity_ratio'].replace([np.inf, -np.inf], np.nan)
    df_features['liquidity_ratio'] = df_features['liquidity_ratio'].fillna(0)
    
    print(f"Liquidity Ratio - Mean: {df_features['liquidity_ratio'].mean():.4f}, "
          f"Median: {df_features['liquidity_ratio'].median():.4f}")
    
    return df_features


def calculate_volatility_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volatility score based on price changes across different timeframes.
    Higher volatility often correlates with lower effective liquidity.
    
    Args:
        df: DataFrame with change_1h, change_24h, change_7d columns
        
    Returns:
        DataFrame with volatility_score column added
    """
    df_features = df.copy()
    
    # Get percentage change columns
    change_cols = ['change_1h', 'change_24h', 'change_7d']
    existing_cols = [col for col in change_cols if col in df_features.columns]
    
    if existing_cols:
        # Volatility as standard deviation of price changes
        df_features['volatility_score'] = df_features[existing_cols].std(axis=1)
        
        # Also calculate absolute average change
        df_features['avg_abs_change'] = df_features[existing_cols].abs().mean(axis=1)
        
        print(f"Volatility Score - Mean: {df_features['volatility_score'].mean():.4f}")
    else:
        df_features['volatility_score'] = 0
        df_features['avg_abs_change'] = 0
    
    return df_features


def calculate_turnover_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate turnover rate as a normalized measure of trading activity.
    
    Args:
        df: DataFrame with volume_24h column
        
    Returns:
        DataFrame with turnover_rate column added
    """
    df_features = df.copy()
    
    # Turnover rate: Volume relative to max volume
    max_volume = df_features['volume_24h'].max()
    if max_volume > 0:
        df_features['turnover_rate'] = df_features['volume_24h'] / max_volume
    else:
        df_features['turnover_rate'] = 0
    
    print(f"Turnover Rate - Mean: {df_features['turnover_rate'].mean():.4f}")
    
    return df_features


def calculate_market_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate market dominance as percentage of total market cap.
    
    Args:
        df: DataFrame with market_cap column
        
    Returns:
        DataFrame with market_dominance column added
    """
    df_features = df.copy()
    
    total_market_cap = df_features['market_cap'].sum()
    if total_market_cap > 0:
        df_features['market_dominance'] = (df_features['market_cap'] / total_market_cap) * 100
    else:
        df_features['market_dominance'] = 0
    
    print(f"Market Dominance - Max: {df_features['market_dominance'].max():.2f}%")
    
    return df_features


def calculate_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price momentum indicators.
    
    Args:
        df: DataFrame with price change columns
        
    Returns:
        DataFrame with momentum indicators
    """
    df_features = df.copy()
    
    # Short-term momentum (1h change)
    if 'change_1h' in df_features.columns:
        df_features['momentum_short'] = df_features['change_1h']
    
    # Medium-term momentum (24h change)
    if 'change_24h' in df_features.columns:
        df_features['momentum_medium'] = df_features['change_24h']
    
    # Long-term momentum (7d change)
    if 'change_7d' in df_features.columns:
        df_features['momentum_long'] = df_features['change_7d']
    
    # Momentum consistency (are all changes in same direction?)
    change_cols = ['change_1h', 'change_24h', 'change_7d']
    existing_cols = [col for col in change_cols if col in df_features.columns]
    
    if existing_cols:
        # Check if all changes are positive or all negative
        signs = df_features[existing_cols].apply(np.sign)
        df_features['momentum_consistency'] = (signs.nunique(axis=1) == 1).astype(int)
    
    return df_features


def calculate_liquidity_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify cryptocurrencies into liquidity categories.
    
    Categories:
    - High: Top 25% liquidity ratio
    - Medium: 25-75% liquidity ratio  
    - Low: Bottom 25% liquidity ratio
    
    Args:
        df: DataFrame with liquidity_ratio column
        
    Returns:
        DataFrame with liquidity_class column added
    """
    df_features = df.copy()
    
    if 'liquidity_ratio' in df_features.columns:
        q25 = df_features['liquidity_ratio'].quantile(0.25)
        q75 = df_features['liquidity_ratio'].quantile(0.75)
        
        conditions = [
            df_features['liquidity_ratio'] >= q75,
            df_features['liquidity_ratio'] >= q25,
            df_features['liquidity_ratio'] < q25
        ]
        choices = ['High', 'Medium', 'Low']
        
        df_features['liquidity_class'] = np.select(conditions, choices, default='Medium')
        
        print(f"\nLiquidity Classification:")
        print(df_features['liquidity_class'].value_counts())
    
    return df_features


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with all engineered features
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    print("\n[1/6] Calculating Liquidity Ratio...")
    df = calculate_liquidity_ratio(df)
    
    print("\n[2/6] Calculating Volatility Score...")
    df = calculate_volatility_score(df)
    
    print("\n[3/6] Calculating Turnover Rate...")
    df = calculate_turnover_rate(df)
    
    print("\n[4/6] Calculating Market Dominance...")
    df = calculate_market_dominance(df)
    
    print("\n[5/6] Calculating Price Momentum...")
    df = calculate_price_momentum(df)
    
    print("\n[6/6] Classifying Liquidity Levels...")
    df = calculate_liquidity_class(df)
    
    return df


def save_featured_data(df: pd.DataFrame, output_path: str = "data/processed/crypto_featured.csv"):
    """
    Save the feature-engineered DataFrame.
    
    Args:
        df: DataFrame with engineered features
        output_path: Path to save the data
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nFeatured data saved to: {output_path}")


def main():
    """Main function to run the feature engineering pipeline."""
    # Load preprocessed data
    input_path = "data/processed/crypto_processed.csv"
    
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found. Please run data_preprocessing.py first.")
        return None
    
    print("Loading preprocessed data...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    
    # Create all features
    df_featured = create_all_features(df)
    
    # Save featured data
    save_featured_data(df_featured)
    
    # Display feature summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED!")
    print("=" * 60)
    print(f"\nNew features added:")
    new_features = ['liquidity_ratio', 'volatility_score', 'avg_abs_change', 
                    'turnover_rate', 'market_dominance', 'momentum_short',
                    'momentum_medium', 'momentum_long', 'momentum_consistency',
                    'liquidity_class']
    for feat in new_features:
        if feat in df_featured.columns:
            print(f"  - {feat}")
    
    return df_featured


if __name__ == "__main__":
    main()
