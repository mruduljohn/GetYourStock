import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Handle TensorFlow import gracefully
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    st.error(f"❌ TensorFlow not available: {str(e)}")
    st.info("💡 This might be due to Python version compatibility. TensorFlow requires Python 3.9-3.12.")

import warnings
import os
import logging
import time
from typing import Dict, Any, Tuple, Optional, List, Union
import json
from dotenv import load_dotenv
import supabase_client

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PREDICTION_DAYS = 30
TIME_STEP = 60
DATA_YEARS = 3

# Comprehensive stock lists organized by category
STOCK_CATEGORIES = {
    "🏆 Most Popular": [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("GOOGL", "Alphabet Inc. (Google)"),
        ("AMZN", "Amazon.com Inc."),
        ("TSLA", "Tesla Inc."),
        ("NVDA", "NVIDIA Corporation"),
        ("META", "Meta Platforms Inc."),
        ("NFLX", "Netflix Inc.")
    ],
    "💰 Financial": [
        ("JPM", "JPMorgan Chase & Co."),
        ("BAC", "Bank of America Corp"),
        ("WFC", "Wells Fargo & Company"),
        ("GS", "Goldman Sachs Group Inc."),
        ("MS", "Morgan Stanley"),
        ("C", "Citigroup Inc."),
        ("V", "Visa Inc."),
        ("MA", "Mastercard Inc.")
    ],
    "🏭 Industrial": [
        ("GE", "General Electric Company"),
        ("CAT", "Caterpillar Inc."),
        ("BA", "Boeing Company"),
        ("MMM", "3M Company"),
        ("HON", "Honeywell International"),
        ("UPS", "United Parcel Service"),
        ("FDX", "FedEx Corporation"),
        ("LMT", "Lockheed Martin Corp")
    ],
    "⚕️ Healthcare": [
        ("JNJ", "Johnson & Johnson"),
        ("PFE", "Pfizer Inc."),
        ("UNH", "UnitedHealth Group Inc."),
        ("ABBV", "AbbVie Inc."),
        ("TMO", "Thermo Fisher Scientific"),
        ("ABT", "Abbott Laboratories"),
        ("MRK", "Merck & Co. Inc."),
        ("CVS", "CVS Health Corporation")
    ],
    "🛒 Consumer": [
        ("PG", "Procter & Gamble Co."),
        ("KO", "Coca-Cola Company"),
        ("PEP", "PepsiCo Inc."),
        ("WMT", "Walmart Inc."),
        ("HD", "Home Depot Inc."),
        ("MCD", "McDonald's Corporation"),
        ("NKE", "Nike Inc."),
        ("SBUX", "Starbucks Corporation")
    ],
    "⚡ Energy": [
        ("XOM", "Exxon Mobil Corporation"),
        ("CVX", "Chevron Corporation"),
        ("COP", "ConocoPhillips"),
        ("SLB", "Schlumberger NV"),
        ("EOG", "EOG Resources Inc."),
        ("PXD", "Pioneer Natural Resources"),
        ("KMI", "Kinder Morgan Inc."),
        ("OXY", "Occidental Petroleum")
    ],
    "🔌 Tech & Software": [
        ("CRM", "Salesforce Inc."),
        ("ORCL", "Oracle Corporation"),
        ("ADBE", "Adobe Inc."),
        ("NOW", "ServiceNow Inc."),
        ("INTC", "Intel Corporation"),
        ("AMD", "Advanced Micro Devices"),
        ("QCOM", "Qualcomm Inc."),
        ("IBM", "International Business Machines")
    ],
    "🚗 Automotive": [
        ("F", "Ford Motor Company"),
        ("GM", "General Motors Company"),
        ("RIVN", "Rivian Automotive Inc."),
        ("LCID", "Lucid Group Inc."),
        ("NIO", "NIO Inc."),
        ("TM", "Toyota Motor Corporation"),
        ("HMC", "Honda Motor Co. Ltd."),
        ("STLA", "Stellantis N.V.")
    ],
    "🏠 Real Estate & REITs": [
        ("AMT", "American Tower Corporation"),
        ("PLD", "Prologis Inc."),
        ("CCI", "Crown Castle Inc."),
        ("EQIX", "Equinix Inc."),
        ("SPG", "Simon Property Group"),
        ("O", "Realty Income Corporation"),
        ("VTR", "Ventas Inc."),
        ("ARE", "Alexandria Real Estate")
    ],
    "📱 Communication": [
        ("T", "AT&T Inc."),
        ("VZ", "Verizon Communications"),
        ("TMUS", "T-Mobile US Inc."),
        ("CMCSA", "Comcast Corporation"),
        ("DIS", "Walt Disney Company"),
        ("NFLX", "Netflix Inc."),
        ("SPOT", "Spotify Technology"),
        ("SNAP", "Snap Inc.")
    ]
}

# Flatten all stocks for search functionality
ALL_STOCKS = {}
for category, stocks in STOCK_CATEGORIES.items():
    for symbol, name in stocks:
        ALL_STOCKS[symbol] = name

# Global variables
model = None
model_loaded = False

# Streamlit page config
st.set_page_config(
    page_title="📈 Stock Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ml_model():
    """
    Load the LSTM model with caching for better performance.
    
    Returns:
        tuple: (model, success_status)
    """
    global model, model_loaded
    
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not available")
        return None, False
    
    try:
        logger.info("Loading LSTM model...")
        
        if not os.path.exists('stock_price_model.h5'):
            logger.error("Model file stock_price_model.h5 not found")
            return None, False
            
        model = load_model('stock_price_model.h5')
        model.make_predict_function()
        model_loaded = True
        
        logger.info("LSTM model loaded successfully")
        return model, True
        
    except Exception as e:
        logger.error(f"Failed to load LSTM model: {str(e)}")
        return None, False

def validate_stock_symbol(symbol: str) -> bool:
    """Validate stock symbol format."""
    if not symbol or not isinstance(symbol, str):
        return False
        
    symbol = symbol.strip().upper()
    return (
        len(symbol) >= 1 and 
        len(symbol) <= 5 and 
        symbol.isalnum()
    )

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(stock_symbol: str) -> pd.DataFrame:
    """
    Fetch stock data with caching for better performance.
    
    Args:
        stock_symbol: Stock symbol to fetch
        
    Returns:
        pd.DataFrame: Processed stock data
    """
    logger.info(f"Fetching stock data for symbol: {stock_symbol}")
    
    if not validate_stock_symbol(stock_symbol):
        raise ValueError(f"Invalid stock symbol format: {stock_symbol}")
    
    stock_symbol = stock_symbol.strip().upper()
    
    try:
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365 * DATA_YEARS)
        
        df = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError(f"No data available for symbol: {stock_symbol}")
        
        # Process the data
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index().rename(columns={'index': 'Date'})
        
        required_columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df = df[required_columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        if len(df) < TIME_STEP:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least {TIME_STEP}")
        
        logger.info(f"Successfully fetched {len(df)} days of data for {stock_symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {stock_symbol}: {str(e)}")
        raise Exception(f"Failed to fetch data for {stock_symbol}: {str(e)}")

def prepare_data(df):
    """Prepare data for LSTM prediction."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    X = np.array([scaled_data[i:i + TIME_STEP, 0]
                  for i in range(len(scaled_data) - TIME_STEP - 1)])
    y = scaled_data[TIME_STEP + 1:, 0]

    return X.reshape(X.shape[0], TIME_STEP, 1), y, scaler

def predict_future(model, data, scaler):
    """Generate future predictions."""
    last_data = data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    future_preds = np.zeros(PREDICTION_DAYS, dtype='float32')

    for i in range(PREDICTION_DAYS):
        next_pred = model.predict(last_data, verbose=0)[0, 0]
        future_preds[i] = next_pred
        last_data = np.roll(last_data, -1, axis=1)
        last_data[0, -1, 0] = next_pred

    return scaler.inverse_transform(future_preds.reshape(-1, 1))

def create_plot(df, pred_data=None, future_data=None, title=""):
    """Create interactive Plotly figure."""
    fig = go.Figure()

    # Main price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))

    # Prediction line
    if pred_data is not None:
        fig.add_trace(go.Scatter(
            x=df.index[TIME_STEP + 1:],
            y=pred_data[:, 0],
            name='Predicted',
            line=dict(color='orange', width=2)
        ))

    # Future prediction
    if future_data is not None:
        future_dates = pd.date_range(
            start=df.index[-1],
            periods=PREDICTION_DAYS + 1
        )[1:]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_data[:, 0],
            name='30-Day Forecast',
            line=dict(color='green', width=2, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    return fig

def save_prediction_to_supabase(stock_symbol, last_price, last_date, prediction_data):
    """Save prediction to Supabase database."""
    if supabase_client.supabase:
        try:
            supabase_client.save_prediction(
                stock_symbol=stock_symbol,
                last_price=float(last_price),
                last_date=last_date,
                prediction_days=PREDICTION_DAYS,
                prediction_data=prediction_data
            )
            logger.info(f"Saved public prediction for {stock_symbol} to Supabase")
            return True
        except Exception as e:
            logger.warning(f"Failed to save prediction to Supabase: {str(e)}")
            return False
    return False

def main():
    """Main Streamlit application."""
    
    # Initialize Supabase
    supabase_client.init_supabase()
    
    # Title and description
    st.title("📈 Real-Time Stock Predictor")
    st.markdown("Predict stock prices using LSTM neural networks with public prediction history")
    
    # Check system requirements
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow is not available. This is required for stock predictions.")
        st.info("💡 **Deployment Issue**: TensorFlow requires Python 3.9-3.12. Please check your Python version.")
        st.info("🔧 **Solution**: Update your deployment to use Python 3.12 or earlier.")
        st.stop()
    
    # Load model
    model, model_success = load_ml_model()
    
    if not model_success:
        st.error("❌ Failed to load the LSTM model. Please check if 'stock_price_model.h5' exists.")
        if not os.path.exists('stock_price_model.h5'):
            st.info("📁 **Missing Model**: The model file 'stock_price_model.h5' was not found.")
            st.info("💡 **Solution**: Ensure the model file is included in your repository.")
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Stock Selection")
        
        # Selection method
        selection_method = st.radio(
            "Choose selection method:",
            ["📋 Browse Categories", "🔍 Search Stocks", "✍️ Manual Entry"],
            index=0
        )
        
        stock_symbol = None
        
        if selection_method == "📋 Browse Categories":
            # Category selection
            selected_category = st.selectbox(
                "Select Category:",
                list(STOCK_CATEGORIES.keys()),
                index=0  # Default to "Most Popular"
            )
            
            # Stock selection within category
            category_stocks = STOCK_CATEGORIES[selected_category]
            stock_options = [f"{symbol} - {name}" for symbol, name in category_stocks]
            
            selected_stock = st.selectbox(
                f"Select from {selected_category}:",
                stock_options,
                index=4 if selected_category == "🏆 Most Popular" else 0  # Default to TSLA in popular
            )
            
            if selected_stock:
                stock_symbol = selected_stock.split(" - ")[0]
                
        elif selection_method == "🔍 Search Stocks":
            # Search functionality
            search_term = st.text_input(
                "Search stocks by symbol or company name:",
                placeholder="e.g., AAPL, Apple, Microsoft..."
            )
            
            if search_term:
                search_term = search_term.upper().strip()
                
                # Find matching stocks
                matches = []
                for symbol, name in ALL_STOCKS.items():
                    if (search_term in symbol.upper() or 
                        search_term.lower() in name.lower()):
                        matches.append((symbol, name))
                
                if matches:
                    # Sort matches - exact symbol matches first
                    matches.sort(key=lambda x: (x[0] != search_term, x[0]))
                    
                    match_options = [f"{symbol} - {name}" for symbol, name in matches[:20]]  # Limit to 20 results
                    
                    selected_match = st.selectbox(
                        f"Found {len(matches)} matches:",
                        match_options,
                        index=0
                    )
                    
                    if selected_match:
                        stock_symbol = selected_match.split(" - ")[0]
                        
                    if len(matches) > 20:
                        st.info(f"Showing top 20 of {len(matches)} results. Be more specific to narrow down.")
                else:
                    st.warning("No matches found. Try a different search term.")
                    
        else:  # Manual Entry
            # Manual input with validation
            manual_symbol = st.text_input(
                "Enter Stock Symbol:",
                value="TSLA",
                help="Enter any valid stock symbol (1-5 characters)",
                max_chars=5
            ).upper().strip()
            
            if manual_symbol:
                if validate_stock_symbol(manual_symbol):
                    stock_symbol = manual_symbol
                    
                    # Show company name if known
                    if manual_symbol in ALL_STOCKS:
                        st.success(f"✅ {manual_symbol} - {ALL_STOCKS[manual_symbol]}")
                    else:
                        st.info(f"ℹ️ {manual_symbol} - Symbol not in our database but may still be valid")
                else:
                    st.error("❌ Invalid format. Use 1-5 alphanumeric characters.")
        
        # Display selected stock info
        if stock_symbol:
            st.markdown("---")
            st.markdown("**📊 Selected Stock:**")
            if stock_symbol in ALL_STOCKS:
                st.write(f"**{stock_symbol}** - {ALL_STOCKS[stock_symbol]}")
            else:
                st.write(f"**{stock_symbol}** - Custom Symbol")
            
            # Quick stats from recent data (if available)
            try:
                with st.spinner("Loading quick stats..."):
                    quick_data = yf.Ticker(stock_symbol).info
                    if quick_data and 'regularMarketPrice' in quick_data:
                        current_price = quick_data.get('regularMarketPrice', 'N/A')
                        prev_close = quick_data.get('previousClose', 'N/A')
                        
                        if current_price != 'N/A' and prev_close != 'N/A':
                            change = current_price - prev_close
                            change_pct = (change / prev_close) * 100
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Current", f"${current_price:.2f}")
                            with col2:
                                st.metric("Change", f"{change_pct:+.2f}%", f"${change:+.2f}")
            except Exception:
                pass  # Silently fail if quick stats unavailable
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button(
            "🚀 Generate Prediction", 
            type="primary",
            disabled=not stock_symbol,
            use_container_width=True
        )
        
        if not stock_symbol:
            st.warning("⚠️ Please select a stock first")
        
        st.markdown("---")
        
        # System status
        st.header("🏥 System Status")
        
        # TensorFlow status
        if TENSORFLOW_AVAILABLE:
            st.success("✅ TensorFlow: Available")
        else:
            st.error("❌ TensorFlow: Not Available")
        
        # Model status
        if model_success:
            st.success("✅ LSTM Model: Loaded")
        else:
            st.error("❌ LSTM Model: Failed")
        
        # Supabase status
        if supabase_client.supabase:
            st.success("✅ Database: Connected")
        else:
            st.warning("⚠️ Database: Offline")
        
        # Python version info
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 13):
            st.warning(f"⚠️ Python: {python_version} (TensorFlow incompatible)")
        else:
            st.success(f"✅ Python: {python_version}")
        
        # Total stocks available
        st.markdown("---")
        st.markdown(f"**📈 {len(ALL_STOCKS)} stocks available**")
        
        # Categories summary
        with st.expander("📋 Browse by Category"):
            for category, stocks in STOCK_CATEGORIES.items():
                st.write(f"**{category}:** {len(stocks)} stocks")
    
    # Main content area
    if predict_button and stock_symbol:
        if not validate_stock_symbol(stock_symbol):
            st.error("❌ Invalid stock symbol format. Please enter a valid symbol (1-5 alphanumeric characters).")
            return
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch data
            status_text.text("📊 Fetching stock data...")
            progress_bar.progress(20)
            
            df = get_stock_data(stock_symbol)
            
            # Step 2: Prepare data
            status_text.text("🔄 Preparing data for prediction...")
            progress_bar.progress(40)
            
            X, y, scaler = prepare_data(df)
            
            # Step 3: Make predictions
            status_text.text("🧠 Generating predictions...")
            progress_bar.progress(60)
            
            y_pred = model.predict(X, verbose=0)
            y_pred = scaler.inverse_transform(y_pred)
            
            # Step 4: Future predictions
            status_text.text("🔮 Forecasting future prices...")
            progress_bar.progress(80)
            
            future_prices = predict_future(
                model,
                scaler.transform(df['Close'].values.reshape(-1, 1)),
                scaler
            )
            
            # Step 5: Save to database
            status_text.text("💾 Saving to database...")
            progress_bar.progress(90)
            
            # Prepare prediction data
            last_price = df['Close'].iloc[-1]
            last_date = df.index[-1].date()
            
            prediction_data = {
                "historical": {
                    "dates": [d.strftime("%Y-%m-%d") for d in df.index[-30:].to_list()],
                    "prices": df['Close'].iloc[-30:].to_list()
                },
                "future": {
                    "dates": [(df.index[-1] + dt.timedelta(days=i+1)).strftime("%Y-%m-%d") 
                             for i in range(PREDICTION_DAYS)],
                    "prices": future_prices[:, 0].tolist()
                },
                "indicators": {
                    "sma_50": df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None,
                    "sma_200": df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else None
                }
            }
            
            # Save to Supabase
            save_prediction_to_supabase(stock_symbol, last_price, last_date, prediction_data)
            
            progress_bar.progress(100)
            status_text.text("✅ Prediction completed!")
            
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Last Price",
                    value=f"${last_price:.2f}",
                    delta=f"{((last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}%"
                )
            
            with col2:
                st.metric(
                    label="Last Date",
                    value=last_date.strftime('%Y-%m-%d')
                )
            
            # Tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📈 Price Prediction", "🔮 30-Day Forecast", "📊 Technical Analysis"])
            
            with tab1:
                st.subheader(f"{stock_symbol} Price Prediction")
                main_plot = create_plot(
                    df,
                    pred_data=y_pred,
                    title=f"{stock_symbol} Historical vs Predicted Prices"
                )
                st.plotly_chart(main_plot, use_container_width=True)
            
            with tab2:
                st.subheader(f"{stock_symbol} 30-Day Forecast")
                future_plot = create_plot(
                    df,
                    future_data=future_prices,
                    title=f"{stock_symbol} 30-Day Price Forecast"
                )
                st.plotly_chart(future_plot, use_container_width=True)
                
                # Forecast summary
                forecast_change = ((future_prices[-1, 0] - last_price) / last_price * 100)
                st.metric(
                    label="30-Day Forecast",
                    value=f"${future_prices[-1, 0]:.2f}",
                    delta=f"{forecast_change:.2f}%"
                )
            
            with tab3:
                st.subheader(f"{stock_symbol} Technical Indicators")
                
                # Calculate technical indicators
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['SMA_200'] = df['Close'].rolling(200).mean()
                
                tech_fig = go.Figure()
                tech_fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'],
                    name='Price', line=dict(color='blue', width=2)))
                
                if not df['SMA_50'].isna().all():
                    tech_fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA_50'],
                        name='50-Day SMA', line=dict(color='orange', width=1)))
                
                if not df['SMA_200'].isna().all():
                    tech_fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA_200'],
                        name='200-Day SMA', line=dict(color='red', width=1)))
                
                tech_fig.update_layout(
                    title=f"{stock_symbol} Technical Indicators",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template='plotly_white',
                    height=500
                )
                st.plotly_chart(tech_fig, use_container_width=True)
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Prediction failed: {str(e)}")
    
    # Recent predictions section
    if supabase_client.supabase:
        st.header("📊 Recent Public Predictions")
        
        try:
            success, recent_predictions, error = supabase_client.get_recent_predictions(10)
            
            if success and recent_predictions:
                # Display recent predictions in a nice format
                for pred in recent_predictions:
                    with st.expander(f"{pred['stock_symbol']} - {pred['created_at'][:10]}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Symbol", pred['stock_symbol'])
                        with col2:
                            st.metric("Last Price", f"${pred['last_price']:.2f}")
                        with col3:
                            st.metric("Prediction Days", pred['prediction_days'])
            else:
                st.info("No recent predictions available.")
        except Exception as e:
            st.warning(f"Could not load recent predictions: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Note:** This is an educational tool. Stock predictions are estimates and should not be used as financial advice."
    )
    
    # Personal branding footer
    st.markdown("---")
    
    # Create columns for centered layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <p style="font-size: 16px; color: #666; margin-bottom: 10px;">
                    Made with ❤️ by <strong>MJ</strong>
                </p>
                <div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
                    <a href="https://github.com/mruduljohn" target="_blank" style="text-decoration: none;">
                        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 16px; 
                                    background-color: #f0f2f6; border-radius: 20px; 
                                    transition: all 0.3s ease; border: 1px solid #e0e0e0;">
                            <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                            </svg>
                            <span style="color: #333; font-weight: 500;">GitHub</span>
                        </div>
                    </a>
                    <a href="https://linkedin.com/in/mruduljohnmathews" target="_blank" style="text-decoration: none;">
                        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 16px; 
                                    background-color: #f0f2f6; border-radius: 20px; 
                                    transition: all 0.3s ease; border: 1px solid #e0e0e0;">
                            <svg width="20" height="20" fill="#0077B5" viewBox="0 0 24 24">
                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                            </svg>
                            <span style="color: #333; font-weight: 500;">LinkedIn</span>
                        </div>
                    </a>
                </div>
                <p style="font-size: 12px; color: #999; margin-top: 15px;">
                    Open source • Built with Streamlit & TensorFlow
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main() 