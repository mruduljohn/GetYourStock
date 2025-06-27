import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
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
    
    # Load model
    model, model_success = load_ml_model()
    
    if not model_success:
        st.error("❌ Failed to load the LSTM model. Please check if 'stock_price_model.h5' exists.")
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Stock symbol input
        stock_symbol = st.text_input(
            "Stock Symbol",
            value="TSLA",
            help="Enter a valid stock symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        # Example symbols
        st.markdown("**Popular Symbols:**")
        example_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        selected_example = st.selectbox("Quick Select:", [""] + example_symbols)
        
        if selected_example:
            stock_symbol = selected_example
        
        # Predict button
        predict_button = st.button("🚀 Predict Stock Price", type="primary")
        
        # System status
        st.header("🏥 System Status")
        
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

if __name__ == "__main__":
    main() 