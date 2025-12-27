# 💎 Cryptocurrency Liquidity Prediction

A machine learning system to predict cryptocurrency liquidity levels for market stability analysis.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.18+-red)

---

## 📋 Project Overview

This project predicts cryptocurrency liquidity using machine learning to help traders and financial institutions:
- Detect liquidity crises early
- Assess market stability
- Make informed trading decisions

**Target Metric**: Liquidity Ratio (Volume / Market Cap)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# Step 1: Preprocess data
python src/data_preprocessing.py

# Step 2: Engineer features
python src/feature_engineering.py

# Step 3: Train models
python src/model_training.py

# Step 4: Evaluate (optional)
python src/model_evaluation.py
```

### 3. Launch Web App
```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
Crypto-Liquidity-Prediction/
├── data/
│   ├── raw/                    # Original CSV files
│   └── processed/              # Cleaned data
├── src/
│   ├── data_preprocessing.py   # Data cleaning
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # ML training
│   └── model_evaluation.py     # Evaluation
├── models/                     # Saved models
├── app/
│   └── streamlit_app.py        # Web dashboard
├── docs/
│   ├── HLD.md                  # High-Level Design
│   ├── LLD.md                  # Low-Level Design
│   ├── Pipeline_Architecture.md
│   ├── EDA_Report.md           # Analysis report
│   └── Final_Report.md         # Project summary
├── requirements.txt
└── README.md
```

---

## 📊 Features Engineered

| Feature | Description |
|---------|-------------|
| `liquidity_ratio` | Volume / Market Cap |
| `volatility_score` | Std dev of price changes |
| `turnover_rate` | Normalized volume |
| `market_dominance` | % of total market cap |
| `momentum_*` | Price change trends |
| `liquidity_class` | High/Medium/Low category |

---

## 🤖 Models Trained

- Linear Regression
- Ridge Regression
- Random Forest ⭐ (Best)
- Gradient Boosting
- XGBoost

**Best Model**: Random Forest with R² = 0.81

---

## 📈 Results

| Metric | Value |
|--------|-------|
| R² Score | 0.81 |
| RMSE | 0.0521 |
| MAE | 0.0198 |

---

## 📝 Documentation

- [High-Level Design](docs/HLD.md)
- [Low-Level Design](docs/LLD.md)
- [Pipeline Architecture](docs/Pipeline_Architecture.md)
- [EDA Report](docs/EDA_Report.md)
- [Final Report](docs/Final_Report.md)

---

## 🛠️ Technologies

- **Python 3.12**
- **Pandas, NumPy** - Data processing
- **Scikit-learn, XGBoost** - Machine learning
- **Matplotlib, Seaborn, Plotly** - Visualization
- **Streamlit** - Web application

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- Data source: CoinGecko
- Built for learning ML and data science workflows
