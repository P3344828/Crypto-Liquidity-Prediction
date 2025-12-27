# Low-Level Design (LLD) Document
## Cryptocurrency Liquidity Prediction System

---

## 1. Module Specifications

### 1.1 Data Preprocessing Module (`src/data_preprocessing.py`)

#### Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `load_raw_data` | `data_dir: str` | `pd.DataFrame` | Loads and merges CSVs |
| `clean_data` | `df: pd.DataFrame` | `pd.DataFrame` | Handles missing values |
| `normalize_features` | `df, columns` | `pd.DataFrame` | Min-Max normalization |
| `save_processed_data` | `df, output_path` | `None` | Saves to CSV |

#### Data Cleaning Logic
```python
# Missing value strategy
percentage_cols: fill with 0 (no change)
numeric_cols: forward fill → backward fill
critical_cols: drop rows if still missing
```

---

### 1.2 Feature Engineering Module (`src/feature_engineering.py`)

#### Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `liquidity_ratio` | `volume_24h / market_cap` | Primary target |
| `volatility_score` | `std([1h%, 24h%, 7d%])` | Risk indicator |
| `turnover_rate` | `volume / max_volume` | Trading activity |
| `market_dominance` | `(mcap / total_mcap) * 100` | Market share |
| `momentum_short` | `change_1h` | Short-term trend |
| `momentum_medium` | `change_24h` | Medium-term trend |
| `momentum_long` | `change_7d` | Long-term trend |
| `liquidity_class` | Quantile-based | Categorical target |

#### Classification Thresholds
```python
High:   liquidity_ratio >= 75th percentile
Medium: liquidity_ratio >= 25th percentile
Low:    liquidity_ratio < 25th percentile
```

---

### 1.3 Model Training Module (`src/model_training.py`)

#### LiquidityPredictor Class

```python
class LiquidityPredictor:
    def __init__(self, random_state=42)
    def prepare_features(self, df) -> (X, y)
    def split_data(self, X, y, test_size=0.2) -> splits
    def train_models(self, X_train, y_train)
    def evaluate_models(self, X_test, y_test) -> results
    def hyperparameter_tuning(self, X_train, y_train, model_name)
    def get_feature_importance(self) -> importance_df
    def save_model(self, output_dir)
```

#### Model Configurations

| Model | Key Parameters |
|-------|----------------|
| Linear Regression | Default |
| Ridge | alpha=1.0 |
| Random Forest | n_estimators=100, max_depth=10 |
| Gradient Boosting | n_estimators=100, learning_rate=0.1 |
| XGBoost | n_estimators=100, max_depth=5 |

#### Hyperparameter Grid (Random Forest)
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
```

---

### 1.4 Model Evaluation Module (`src/model_evaluation.py`)

#### Metrics Calculated

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | `sqrt(mean(y - ŷ)²)` | Average error magnitude |
| MAE | `mean(|y - ŷ|)` | Average absolute error |
| R² | `1 - SS_res/SS_tot` | Variance explained |
| MAPE | `mean(|y - ŷ|/y) * 100` | Percentage error |

#### Visualization Functions
- `plot_predictions_vs_actual()`: Scatter plot
- `plot_residuals()`: Residual analysis
- `plot_feature_importance()`: Bar chart
- `plot_model_comparison()`: Multi-metric comparison

---

### 1.5 Streamlit Application (`app/streamlit_app.py`)

#### Page Structure

```
├── Dashboard
│   ├── Key Metrics (4 cards)
│   ├── Top 10 Liquidity Chart
│   ├── Distribution Histogram
│   ├── Classification Pie Chart
│   └── Top Coins Table
├── Predict Liquidity
│   ├── Input Form
│   ├── Prediction Button
│   └── Results Display
├── Analysis
│   ├── Correlation Heatmap
│   ├── Volume vs MCap Scatter
│   └── Price Changes Boxplot
└── About
    └── Documentation
```

#### Prediction Flow
```
User Input → DataFrame → Scale Features → Model.predict() → Classify → Display
```

---

## 2. Data Structures

### Input Schema
```python
{
    'coin': str,          # Cryptocurrency name
    'symbol': str,        # Trading symbol
    'price': float,       # Current price in USD
    '1h': float,          # 1-hour change (decimal)
    '24h': float,         # 24-hour change
    '7d': float,          # 7-day change
    '24h_volume': float,  # Trading volume
    'mkt_cap': float,     # Market capitalization
    'date': str           # Date of data
}
```

### Output Schema
```python
{
    'liquidity_ratio': float,      # Predicted value
    'liquidity_class': str,        # 'High'/'Medium'/'Low'
    'confidence': float            # Model confidence
}
```

---

## 3. File Organization

```
Crypto-Liquidity-Prediction/
├── data/
│   ├── raw/                    # Original CSVs
│   │   ├── coin_gecko_2022-03-16.csv
│   │   └── coin_gecko_2022-03-17.csv
│   └── processed/
│       ├── crypto_processed.csv
│       └── crypto_featured.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.joblib
│   └── model_results.csv
├── app/
│   └── streamlit_app.py
├── docs/
│   ├── HLD.md
│   ├── LLD.md
│   ├── Pipeline_Architecture.md
│   ├── EDA_Report.md
│   └── Final_Report.md
├── reports/
│   ├── predictions_vs_actual.png
│   ├── residuals.png
│   ├── feature_importance.png
│   └── model_comparison.png
├── requirements.txt
└── README.md
```

---

## 4. Error Handling

| Scenario | Handler |
|----------|---------|
| Missing CSV files | `FileNotFoundError` with message |
| Missing columns | Filter to available columns |
| NaN in predictions | Fill with median |
| Model not found | Display setup instructions |
| Invalid user input | Streamlit validation |

---

## 5. Performance Considerations

- **Data Loading**: Cached with `@st.cache_data`
- **Model Loading**: Cached with `@st.cache_resource`
- **Batch Processing**: Vectorized Pandas operations
- **Memory**: Feature selection limits columns loaded
