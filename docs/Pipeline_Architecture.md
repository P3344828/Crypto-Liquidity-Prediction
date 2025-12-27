# Pipeline Architecture Document
## Cryptocurrency Liquidity Prediction System

---

## 1. End-to-End Pipeline Overview

```mermaid
flowchart LR
    subgraph Input
        A[Raw CSV Files]
    end
    
    subgraph Preprocessing
        B[Load Data]
        C[Clean Data]
        D[Normalize]
    end
    
    subgraph Features
        E[Liquidity Ratio]
        F[Volatility]
        G[Turnover]
        H[Momentum]
    end
    
    subgraph Training
        I[Split Data]
        J[Train Models]
        K[Evaluate]
        L[Select Best]
    end
    
    subgraph Deployment
        M[Save Model]
        N[Streamlit App]
        O[Predictions]
    end
    
    A --> B --> C --> D
    D --> E & F & G & H
    E & F & G & H --> I --> J --> K --> L --> M --> N --> O
```

---

## 2. Pipeline Stages

### Stage 1: Data Ingestion
- **Input**: `data/raw/*.csv`
- **Process**: Load and merge all CSV files
- **Output**: Combined DataFrame

### Stage 2: Data Preprocessing
- **Input**: Raw DataFrame
- **Process**: Clean, handle missing values, normalize
- **Output**: `data/processed/crypto_processed.csv`

### Stage 3: Feature Engineering
- **Input**: Preprocessed data
- **Process**: Create liquidity-related features
- **Output**: `data/processed/crypto_featured.csv`

### Stage 4: Model Training
- **Input**: Featured dataset
- **Process**: Train multiple models, tune, evaluate
- **Output**: `models/best_model.joblib`

### Stage 5: Deployment
- **Input**: Saved model + data
- **Process**: Streamlit web application
- **Output**: Interactive predictions

---

## 3. Execution Commands

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run preprocessing
python src/data_preprocessing.py

# Step 3: Run feature engineering
python src/feature_engineering.py

# Step 4: Train models
python src/model_training.py

# Step 5: Evaluate models (optional)
python src/model_evaluation.py

# Step 6: Launch web app
streamlit run app/streamlit_app.py
```

---

## 4. Data Flow Diagram

```mermaid
graph TD
    subgraph Data Storage
        R1[coin_gecko_2022-03-16.csv]
        R2[coin_gecko_2022-03-17.csv]
        P1[crypto_processed.csv]
        P2[crypto_featured.csv]
    end
    
    subgraph Model Storage
        M1[best_model.joblib]
        M2[scaler.joblib]
        M3[feature_columns.joblib]
    end
    
    subgraph Processing
        S1[data_preprocessing.py]
        S2[feature_engineering.py]
        S3[model_training.py]
    end
    
    R1 & R2 --> S1 --> P1 --> S2 --> P2 --> S3 --> M1 & M2 & M3
```

---

## 5. Feature Pipeline

| Stage | Input Features | Output Features |
|-------|---------------|-----------------|
| Raw | coin, symbol, price, 1h, 24h, 7d, volume, mcap, date | - |
| Clean | All raw | Renamed, nulls handled |
| Normalize | price, volume, mcap | + _normalized versions |
| Engineer | All cleaned | + liquidity_ratio, volatility_score, turnover_rate, etc. |

---

## 6. Model Pipeline

```mermaid
flowchart TB
    A[Featured Data] --> B[Train/Test Split 80/20]
    B --> C[StandardScaler]
    C --> D1[Linear Regression]
    C --> D2[Ridge]
    C --> D3[Random Forest]
    C --> D4[Gradient Boosting]
    C --> D5[XGBoost]
    D1 & D2 & D3 & D4 & D5 --> E[Evaluate All]
    E --> F[Compare R² Scores]
    F --> G[Select Best Model]
    G --> H[Hyperparameter Tuning]
    H --> I[Final Model]
    I --> J[Save to Disk]
```

---

