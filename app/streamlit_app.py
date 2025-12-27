"""
Streamlit Web Application for Cryptocurrency Liquidity Prediction
This application provides an interactive interface to visualize cryptocurrency liquidity data,
make liquidity predictions using a trained machine learning model, and explore detailed analyses.

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Crypto Liquidity Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #1e1e2e;
        padding: 1rem;
        border-radius: 10px;
    }
    .prediction-high {
        color: #00ff88;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .prediction-low {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the featured cryptocurrency data."""
    try:
        df = pd.read_csv("data/processed/crypto_featured.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the preprocessing pipeline first.")
        return None


@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    try:
        model = joblib.load("models/best_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        feature_columns = joblib.load("models/feature_columns.joblib")
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Cryptocurrency Liquidity Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and model
    df = load_data()
    model, scaler, feature_columns = load_model()
    
    if df is None:
        st.warning("⚠️ Please run the data pipeline first:")
        st.code("""
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model_training.py
        """)
        return
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.radio("Select Page", [
        "📊 Dashboard",
        "🔮 Predict Liquidity",
        "📈 Analysis",
        "ℹ️ About"
    ])
    
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "🔮 Predict Liquidity":
        show_prediction(df, model, scaler, feature_columns)
    elif page == "📈 Analysis":
        show_analysis(df)
    else:
        show_about()


def show_dashboard(df):
    """Display the main dashboard."""
    st.header("📊 Market Overview Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cryptocurrencies", len(df['coin'].unique()) if 'coin' in df.columns else len(df))
    
    with col2:
        avg_liquidity = df['liquidity_ratio'].mean() if 'liquidity_ratio' in df.columns else 0
        st.metric("Avg Liquidity Ratio", f"{avg_liquidity:.4f}")
    
    with col3:
        total_volume = df['volume_24h'].sum() if 'volume_24h' in df.columns else 0
        st.metric("Total 24h Volume", f"${total_volume/1e9:.2f}B")
    
    with col4:
        total_mcap = df['market_cap'].sum() if 'market_cap' in df.columns else 0
        st.metric("Total Market Cap", f"${total_mcap/1e12:.2f}T")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 by Liquidity Ratio")
        if 'liquidity_ratio' in df.columns and 'coin' in df.columns:
            top_10 = df.nlargest(10, 'liquidity_ratio')[['coin', 'liquidity_ratio']]
            fig = px.bar(
                top_10, 
                x='liquidity_ratio', 
                y='coin', 
                orientation='h',
                color='liquidity_ratio',
                color_continuous_scale='Viridis',
                title="Most Liquid Cryptocurrencies"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Liquidity Distribution")
        if 'liquidity_ratio' in df.columns:
            fig = px.histogram(
                df, 
                x='liquidity_ratio', 
                nbins=50,
                title="Distribution of Liquidity Ratios",
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
    
    # Liquidity Classification Pie Chart
    st.subheader("Liquidity Classification")
    if 'liquidity_class' in df.columns:
        class_counts = df['liquidity_class'].value_counts()
        fig = px.pie(
            values=class_counts.values, 
            names=class_counts.index,
            title="Distribution by Liquidity Class",
            color_discrete_sequence=['#00ff88', '#ffaa00', '#ff4444']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Data Table
    st.subheader("📋 Top Cryptocurrencies by Market Cap")
    if 'market_cap' in df.columns:
        display_cols = ['coin', 'symbol', 'price', 'volume_24h', 'market_cap', 'liquidity_ratio', 'liquidity_class']
        available_cols = [c for c in display_cols if c in df.columns]
        top_coins = df.nlargest(20, 'market_cap')[available_cols]
        st.dataframe(top_coins, width='stretch')


def show_prediction(df, model, scaler, feature_columns):
    """Show the prediction interface."""
    st.header("🔮 Liquidity Prediction")
    
    if model is None:
        st.error("Model not loaded. Please train the model first.")
        return
    
    st.markdown("### Enter cryptocurrency parameters to predict liquidity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.number_input("Price ($)", min_value=0.0, value=100.0, step=0.01)
        volume_24h = st.number_input("24h Volume ($)", min_value=0.0, value=1000000.0, step=1000.0)
        market_cap = st.number_input("Market Cap ($)", min_value=0.0, value=10000000.0, step=10000.0)
        change_1h = st.slider("1h Change (%)", -50.0, 50.0, 0.0, 0.1) / 100
        change_24h = st.slider("24h Change (%)", -50.0, 50.0, 0.0, 0.1) / 100
    
    with col2:
        change_7d = st.slider("7d Change (%)", -100.0, 100.0, 0.0, 0.1) / 100
        volatility_score = abs(np.std([change_1h, change_24h, change_7d]))
        avg_abs_change = np.mean([abs(change_1h), abs(change_24h), abs(change_7d)])
        
        if df is not None and 'volume_24h' in df.columns:
            turnover_rate = volume_24h / df['volume_24h'].max()
            market_dominance = (market_cap / df['market_cap'].sum()) * 100
        else:
            turnover_rate = 0.1
            market_dominance = 0.01
        
        st.info(f"**Calculated Volatility Score:** {volatility_score:.4f}")
        st.info(f"**Calculated Turnover Rate:** {turnover_rate:.4f}")
    
    if st.button("🔮 Predict Liquidity", type="primary"):
        # Prepare input
        input_data = {
            'price': [price],
            'volume_24h': [volume_24h],
            'market_cap': [market_cap],
            'change_1h': [change_1h],
            'change_24h': [change_24h],
            'change_7d': [change_7d],
            'volatility_score': [volatility_score],
            'avg_abs_change': [avg_abs_change],
            'turnover_rate': [turnover_rate],
            'market_dominance': [market_dominance]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Select only available features
        available_features = [col for col in feature_columns if col in input_df.columns]
        input_scaled = scaler.transform(input_df[available_features].fillna(0))
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Classify
        if prediction >= 0.1:
            liquidity_class = "High"
            class_color = "prediction-high"
        elif prediction >= 0.05:
            liquidity_class = "Medium"
            class_color = "prediction-medium"
        else:
            liquidity_class = "Low"
            class_color = "prediction-low"
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Liquidity Ratio", f"{prediction:.6f}")
        
        with col2:
            st.markdown(f"**Liquidity Class:** <span class='{class_color}'>{liquidity_class}</span>", 
                       unsafe_allow_html=True)
        
        with col3:
            actual_ratio = volume_24h / market_cap if market_cap > 0 else 0
            st.metric("Actual Volume/MCap Ratio", f"{actual_ratio:.6f}")
        
        # Interpretation
        st.markdown("### 📝 Interpretation")
        if liquidity_class == "High":
            st.success("✅ **High Liquidity**: This cryptocurrency can be easily traded with minimal price impact. Good for active trading.")
        elif liquidity_class == "Medium":
            st.warning("⚠️ **Medium Liquidity**: Moderate trading activity. Be cautious with large trades as they may impact price.")
        else:
            st.error("❌ **Low Liquidity**: Limited trading activity. Large trades may significantly impact price. Exercise caution.")


def show_analysis(df):
    """Show detailed analysis."""
    st.header("📈 Detailed Analysis")
    
    # Correlation Heatmap
    st.subheader("Feature Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_cols = ['price', 'volume_24h', 'market_cap', 'liquidity_ratio', 'volatility_score', 
                     'change_1h', 'change_24h', 'change_7d']
        available_corr = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[available_corr].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    # Volume vs Market Cap Scatter
    st.subheader("Volume vs Market Cap Analysis")
    if 'volume_24h' in df.columns and 'market_cap' in df.columns:
        fig = px.scatter(
            df,
            x='market_cap',
            y='volume_24h',
            color='liquidity_ratio' if 'liquidity_ratio' in df.columns else None,
            hover_name='coin' if 'coin' in df.columns else None,
            log_x=True,
            log_y=True,
            title="Volume vs Market Cap (Log Scale)",
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    # Price Changes Distribution
    st.subheader("Price Changes Distribution")
    change_cols = ['change_1h', 'change_24h', 'change_7d']
    available_changes = [c for c in change_cols if c in df.columns]
    
    if available_changes:
        fig = go.Figure()
        colors = ['#667eea', '#764ba2', '#f093fb']
        for i, col in enumerate(available_changes):
            fig.add_trace(go.Box(y=df[col]*100, name=col.replace('change_', ''), marker_color=colors[i]))
        
        fig.update_layout(
            title="Price Change Distribution by Timeframe",
            yaxis_title="Change (%)",
            height=400
        )
        st.plotly_chart(fig, width='stretch')


def show_about():
    """Show about page."""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Cryptocurrency Liquidity Prediction System
    
    This application predicts cryptocurrency liquidity levels using machine learning to help traders
    and financial institutions make informed decisions about market stability.
    
    ### 🎯 Features
    - **Real-time Liquidity Prediction**: Enter crypto parameters and get instant predictions
    - **Interactive Dashboard**: Visualize market liquidity patterns
    - **Detailed Analysis**: Explore correlations and trends in the data
    
    ### 📊 Liquidity Metric
    We use **Volume/Market Cap ratio** as the primary liquidity proxy:
    - **High Liquidity (>10%)**: Easy to trade, minimal price impact
    - **Medium Liquidity (5-10%)**: Moderate trading activity
    - **Low Liquidity (<5%)**: Limited trading, potential price impact
    
    ### 🤖 Model Information
    - **Algorithm**: Random Forest Regressor with hyperparameter tuning
    - **Features**: Price, Volume, Market Cap, Price Changes, Volatility, etc.
    - **Evaluation Metrics**: RMSE, MAE, R² Score
    
    ### 📝 How to Use
    1. Navigate to **Dashboard** for market overview
    2. Use **Predict Liquidity** to make predictions
    3. Explore **Analysis** for detailed insights
    
    ---
    **Built with**: Python, Streamlit, Scikit-learn, Plotly
    """)
    


if __name__ == "__main__":
    main()
