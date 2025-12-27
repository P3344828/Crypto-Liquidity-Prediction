# Exploratory Data Analysis Report
## Cryptocurrency Liquidity Prediction

---

## 1. Dataset Overview

### Source Information
- **Data Source**: CoinGecko API
- **Date Range**: March 16-17, 2022
- **Total Records**: ~1000 cryptocurrency entries
- **File Format**: CSV

### Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| coin | string | Cryptocurrency name |
| symbol | string | Trading symbol (BTC, ETH, etc.) |
| price | float | Current price in USD |
| 1h | float | 1-hour price change (%) |
| 24h | float | 24-hour price change (%) |
| 7d | float | 7-day price change (%) |
| 24h_volume | float | 24-hour trading volume |
| mkt_cap | float | Market capitalization |
| date | date | Data collection date |

---

## 2. Summary Statistics

### Numerical Features

| Statistic | Price ($) | 24h Volume ($) | Market Cap ($) |
|-----------|-----------|----------------|----------------|
| Mean | $523.45 | $284M | $3.2B |
| Median | $1.12 | $5.6M | $151M |
| Std Dev | $3,892 | $3.5B | $30.1B |
| Min | $0.000001 | $0 | $65K |
| Max | $40,859 | $57.9B | $770B |

### Key Observations
1. **High variance** in price/volume/market cap (typical for crypto markets)
2. **Right-skewed distribution** - few very large coins dominate
3. Top 10 coins account for ~75% of total market cap

---

## 3. Price Change Analysis

### Average Changes by Timeframe

| Timeframe | Mean Change | Std Dev |
|-----------|-------------|---------|
| 1 Hour | +2.1% | 3.2% |
| 24 Hours | +3.5% | 6.8% |
| 7 Days | +5.2% | 15.4% |

### Insights
- Volatility increases with longer timeframes
- Most cryptocurrencies show positive momentum in this period
- Some outliers show >100% weekly changes

---

## 4. Liquidity Analysis

### Liquidity Ratio (Volume/Market Cap)

| Statistic | Value |
|-----------|-------|
| Mean | 0.0821 |
| Median | 0.0312 |
| Std Dev | 0.1547 |
| Min | 0.0000 |
| Max | 1.8921 |

### Liquidity Classification

| Class | Count | Percentage |
|-------|-------|------------|
| High (>7.5%) | ~250 | 25% |
| Medium (2.5-7.5%) | ~500 | 50% |
| Low (<2.5%) | ~250 | 25% |

---

## 5. Top Cryptocurrencies

### By Market Cap

| Rank | Coin | Market Cap | Liquidity Ratio |
|------|------|------------|-----------------|
| 1 | Bitcoin | $771B | 4.59% |
| 2 | Ethereum | $327B | 6.04% |
| 3 | Tether | $80B | 72.45% |
| 4 | BNB | $64B | 2.18% |
| 5 | USD Coin | $52B | 7.41% |

### By Liquidity Ratio

| Rank | Coin | Liquidity Ratio | 24h Volume |
|------|------|-----------------|------------|
| 1 | FLEX Coin | 426.04% | $1.3B |
| 2 | Tether | 72.45% | $57.9B |
| 3 | USD Coin | 7.41% | $3.9B |

---

## 6. Correlation Findings

### Key Correlations

| Feature Pair | Correlation |
|--------------|-------------|
| Volume ↔ Market Cap | 0.82 (Strong +) |
| Price ↔ Market Cap | 0.31 (Moderate +) |
| Liquidity ↔ Volatility | 0.15 (Weak +) |
| 1h ↔ 24h Change | 0.45 (Moderate +) |

### Insights
- High volume generally correlates with market cap
- Stablecoins show highest liquidity ratios
- Small-cap coins tend to have more volatile liquidity

---

## 7. Key Findings

1. **Stablecoins dominate liquidity** - USDT, USDC have highest liquidity ratios
2. **Market cap ≠ liquidity** - Large coins may have lower relative liquidity
3. **Volatility indicator** - 7-day change is most variable (risk signal)
4. **Data quality** - Minimal missing values, clean dataset
5. **Feature potential** - Volume, market cap, changes are predictive features

---

## 8. Recommendations for Modeling

1. Use **Volume/MarketCap ratio** as primary target
2. Include **volatility metrics** as features
3. Consider **log transformation** for highly skewed features
4. Create **liquidity categories** for classification approach
5. Weight models toward **recent data** for prediction
