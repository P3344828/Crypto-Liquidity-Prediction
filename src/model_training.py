"""
Model Training for Cryptocurrency Liquidity Prediction

Trains multiple ML models to predict cryptocurrency liquidity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost model.")


class LiquidityPredictor:
    """
    A class to train and evaluate multiple ML models for liquidity prediction.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.results = {}
        self.q25 = 0.05
        self.q75 = 0.1
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and target for model training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            X (features), y (target)
        """
        # Define feature columns
        self.feature_columns = [
            'price', 'volume_24h', 'market_cap',
            'change_1h', 'change_24h', 'change_7d',
            'volatility_score', 'avg_abs_change',
            'turnover_rate', 'market_dominance'
        ]
        
        # Filter to available columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        # Target: liquidity_ratio
        if 'liquidity_ratio' not in df.columns:
            raise ValueError("liquidity_ratio column not found. Run feature_engineering.py first.")
        
        X = df[self.feature_columns].copy()
        y = df['liquidity_ratio'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        y = y.fillna(y.median())
        
        # Save threshold values
        self.q25 = float(y.quantile(0.25))
        self.q75 = float(y.quantile(0.75))
        print(f"Calculated thresholds from data: q25={self.q25:.6f}, q75={self.q75:.6f}")
        
        print(f"Features: {len(self.feature_columns)}")
        print(f"Samples: {len(X)}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train multiple ML models.
        
        Args:
            X_train: Scaled training features
            y_train: Training target
        """
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)
        
        # 1. Linear Regression (Baseline)
        print("\n[1/5] Training Linear Regression...")
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train, y_train)
        
        # 2. Ridge Regression
        print("[2/5] Training Ridge Regression...")
        self.models['Ridge'] = Ridge(alpha=1.0, random_state=self.random_state)
        self.models['Ridge'].fit(X_train, y_train)
        
        # 3. Random Forest
        print("[3/5] Training Random Forest...")
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)
        
        # 4. Gradient Boosting
        print("[4/5] Training Gradient Boosting...")
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state
        )
        self.models['Gradient Boosting'].fit(X_train, y_train)
        
        # 5. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("[5/5] Training XGBoost...")
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.models['XGBoost'].fit(X_train, y_train)
        else:
            print("[5/5] Skipping XGBoost (not installed)")
        
        print(f"\nTrained {len(self.models)} models successfully!")
    
    def evaluate_models(self, X_test, y_test) -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Scaled test features
            y_test: Test target
            
        Returns:
            DataFrame with evaluation metrics for each model
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2
            })
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.4f}")
        
        self.results = pd.DataFrame(results)
        self.results = self.results.sort_values('R² Score', ascending=False)
        
        # Select best model
        best_idx = self.results['R² Score'].idxmax()
        self.best_model_name = self.results.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n{'=' * 60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"{'=' * 60}")
        
        return self.results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name: str = 'Random Forest'):
        """
        Perform hyperparameter tuning on the specified model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of model to tune
        """
        print(f"\n{'=' * 60}")
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print(f"{'=' * 60}")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            model = GradientBoostingRegressor(random_state=self.random_state)
        
        else:
            print(f"Tuning not implemented for {model_name}")
            return
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best CV Score (R²): {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[f'{model_name} (Tuned)'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the best model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            print(importance.to_string(index=False))
            
            return importance
        else:
            print("Feature importance not available for this model type.")
            return None
    
    def save_model(self, output_dir: str = "models"):
        """
        Save the best model and scaler.
        
        Args:
            output_dir: Directory to save model files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "best_model.joblib"
        joblib.dump(self.best_model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = output_path / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save feature columns
        features_path = output_path / "feature_columns.joblib"
        joblib.dump(self.feature_columns, features_path)
        print(f"Features saved to: {features_path}")
        
        # Save liquidity thresholds (q25 and q75)
        thresholds_path = output_path / "liquidity_thresholds.joblib"
        joblib.dump({'q25': self.q25, 'q75': self.q75}, thresholds_path)
        print(f"Liquidity thresholds saved to: {thresholds_path}")
        
        # Save results
        results_path = output_path / "model_results.csv"
        self.results.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")


def main():
    """Main function to run the model training pipeline."""
    print("=" * 60)
    print("CRYPTOCURRENCY LIQUIDITY PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load featured data
    input_path = "data/processed/crypto_featured.csv"
    
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found. Please run feature_engineering.py first.")
        return
    
    print("\nLoading featured data...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    
    # Initialize predictor
    predictor = LiquidityPredictor(random_state=42)
    
    # Prepare features
    print("\nPreparing features...")
    X, y = predictor.prepare_features(df)
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train models
    predictor.train_models(X_train, y_train)
    
    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)
    
    # Hyperparameter tuning on Random Forest
    predictor.hyperparameter_tuning(X_train, y_train, 'Random Forest')
    
    # Re-evaluate with tuned model
    print("\nRe-evaluating with tuned model...")
    results = predictor.evaluate_models(X_test, y_test)
    
    # Get feature importance
    predictor.get_feature_importance()
    
    # Save best model
    print("\nSaving model...")
    predictor.save_model()
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return predictor


if __name__ == "__main__":
    main()
