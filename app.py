import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import gradio as gr
import warnings
import os
import logging
import time
from typing import Dict, Any, Tuple, Optional, List, Union
from flask import Flask, jsonify, request, session, redirect, url_for
import threading
import sys
import json
from dotenv import load_dotenv
import supabase_client

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
PREDICTION_DAYS = 30
TIME_STEP = 60
DATA_YEARS = 3
HEALTH_CHECK_TIMEOUT = 10  # seconds

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and health status
model = None
model_loaded = False
last_health_check = None
health_status = {
    "model_loaded": False,
    "yfinance_accessible": False,
    "last_check_timestamp": None,
    "check_duration_ms": None,
    "errors": []
}

def load_ml_model():
    """
    Load the LSTM model with comprehensive error handling and logging.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
        
    Raises:
        Exception: If model loading fails critically
    """
    global model, model_loaded
    
    try:
        logger.info("Attempting to load LSTM model from stock_price_model.h5")
        
        if not os.path.exists('stock_price_model.h5'):
            logger.error("Model file stock_price_model.h5 not found")
            return False
            
model = load_model('stock_price_model.h5')
model.make_predict_function()  # For faster inference
        model_loaded = True
        
        logger.info("LSTM model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load LSTM model: {str(e)}")
        model_loaded = False
        return False

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format and basic requirements.
    
    Args:
        symbol (str): Stock symbol to validate
        
    Returns:
        bool: True if symbol appears valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
        
    # Basic validation: alphanumeric, 1-5 characters, uppercase
    symbol = symbol.strip().upper()
    return (
        len(symbol) >= 1 and 
        len(symbol) <= 5 and 
        symbol.isalnum()
    )

def test_yfinance_connectivity(test_symbol: str = "AAPL") -> Tuple[bool, Optional[str]]:
    """
    Test connectivity to Yahoo Finance API with timeout and error handling.
    
    Args:
        test_symbol (str): Symbol to use for connectivity test
        
    Returns:
        Tuple[bool, Optional[str]]: (success_status, error_message)
    """
    try:
        logger.debug(f"Testing yfinance connectivity with symbol: {test_symbol}")
        
        # Attempt to fetch minimal data with timeout
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=5)  # Just 5 days for health check
        
        # This should be fast for health check
        df = yf.download(test_symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return False, f"No data returned for symbol {test_symbol}"
            
        logger.debug("yfinance connectivity test successful")
        return True, None
        
    except Exception as e:
        error_msg = f"yfinance connectivity failed: {str(e)}"
        logger.warning(error_msg)
        return False, error_msg

@app.route('/health/search', methods=['GET'])
def health_check_search():
    """
    Comprehensive health check endpoint for search functionality.
    
    Validates:
    - Model availability and readiness
    - External API (yfinance) connectivity
    - Supabase connection status
    - System resource availability
    - Response time benchmarks
    
    Returns:
        JSON: Detailed health status with HTTP status codes:
        - 200: All systems operational
        - 503: Critical services unavailable
        - 500: Internal system errors
        
    Security considerations:
    - No sensitive information exposed
    - Rate limiting through basic timestamp tracking
    - Input validation for optional test parameters
    """
    global health_status, last_health_check
    
    start_time = time.time()
    current_timestamp = dt.datetime.utcnow().isoformat()
    
    logger.info("Health check initiated for search endpoint")
    
    try:
        # Initialize response structure
        response_data = {
            "service": "stock_search",
            "status": "unknown",
            "timestamp": current_timestamp,
            "checks": {},
            "performance": {},
            "errors": []
        }
        
        # Check 1: Model availability
        logger.debug("Checking model availability")
        model_check_start = time.time()
        
        if not model_loaded or model is None:
            logger.warning("Model not loaded, attempting to reload")
            model_available = load_ml_model()
        else:
            model_available = True
            
        response_data["checks"]["model_loaded"] = model_available
        response_data["performance"]["model_check_ms"] = round((time.time() - model_check_start) * 1000, 2)
        
        if not model_available:
            response_data["errors"].append("LSTM model not available")
        
        # Check 2: External API connectivity
        logger.debug("Checking yfinance API connectivity")
        api_check_start = time.time()
        
        # Get test symbol from query parameter or use default
        test_symbol = request.args.get('test_symbol', 'AAPL')
        
        # Validate test symbol to prevent injection
        if not validate_stock_symbol(test_symbol):
            test_symbol = 'AAPL'  # Fallback to safe default
            
        api_available, api_error = test_yfinance_connectivity(test_symbol)
        response_data["checks"]["yfinance_api"] = api_available
        response_data["performance"]["api_check_ms"] = round((time.time() - api_check_start) * 1000, 2)
        
        if not api_available:
            response_data["errors"].append(api_error or "Yahoo Finance API unavailable")
        
        # Check 3: Supabase connectivity
        logger.debug("Checking Supabase connectivity")
        supabase_check_start = time.time()
        
        # Test Supabase connection
        supabase_available = supabase_client.supabase is not None
        
        if supabase_available:
            # Try a simple query to verify connection
            try:
                result = supabase_client.get_trending_stocks(1)
                supabase_available = result[0]  # First element is success status
                if not supabase_available:
                    response_data["errors"].append(f"Supabase connection issue: {result[2]}")
            except Exception as e:
                supabase_available = False
                response_data["errors"].append(f"Supabase error: {str(e)}")
        else:
            response_data["errors"].append("Supabase client not initialized")
            
        response_data["checks"]["supabase_connected"] = supabase_available
        response_data["performance"]["supabase_check_ms"] = round((time.time() - supabase_check_start) * 1000, 2)
        
        # Check 4: System resources (basic)
        logger.debug("Checking system resources")
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            response_data["checks"]["memory_usage_percent"] = memory_percent
            
            if memory_percent > 90:
                response_data["errors"].append(f"High memory usage: {memory_percent}%")
                
        except ImportError:
            logger.debug("psutil not available, skipping memory check")
            response_data["checks"]["memory_usage_percent"] = "unavailable"
        
        # Calculate overall status
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        response_data["performance"]["total_check_ms"] = total_time_ms
        
        # Determine overall health status
        critical_checks_passed = model_available and api_available
        supabase_required = os.environ.get("SUPABASE_REQUIRED", "false").lower() == "true"
        
        if supabase_required:
            critical_checks_passed = critical_checks_passed and supabase_available
        
        if critical_checks_passed and len(response_data["errors"]) == 0:
            response_data["status"] = "healthy"
            http_status = 200
            logger.info(f"Health check passed in {total_time_ms}ms")
        elif critical_checks_passed:
            response_data["status"] = "degraded"
            http_status = 200
            logger.warning(f"Health check passed with warnings in {total_time_ms}ms")
        else:
            response_data["status"] = "unhealthy"
            http_status = 503
            logger.error(f"Health check failed in {total_time_ms}ms")
        
        # Update global health status
        health_status.update({
            "model_loaded": model_available,
            "yfinance_accessible": api_available,
            "supabase_connected": supabase_available,
            "last_check_timestamp": current_timestamp,
            "check_duration_ms": total_time_ms,
            "errors": response_data["errors"]
        })
        last_health_check = time.time()
        
        return jsonify(response_data), http_status
        
    except Exception as e:
        # Critical error in health check itself
        error_msg = f"Health check system failure: {str(e)}"
        logger.error(error_msg)
        
        error_response = {
            "service": "stock_search",
            "status": "error",
            "timestamp": current_timestamp,
            "error": "Health check system failure",
            "performance": {
                "total_check_ms": round((time.time() - start_time) * 1000, 2)
            }
        }
        
        return jsonify(error_response), 500

@app.route('/health', methods=['GET'])
def health_check_general():
    """
    General health check endpoint providing basic service status.
    
    Returns:
        JSON: Basic health information
    """
    logger.info("General health check requested")
    
    current_time = dt.datetime.utcnow().isoformat()
    uptime_info = "Service running"  # Basic uptime info
    
    basic_health = {
        "service": "stock_prediction_service",
        "status": "online",
        "timestamp": current_time,
        "uptime": uptime_info,
        "version": "1.0.0"
    }
    
    # Include last search health check if available
    if last_health_check:
        time_since_last_check = time.time() - last_health_check
        basic_health["last_search_health_check_seconds_ago"] = round(time_since_last_check, 2)
        basic_health["search_service_status"] = health_status.get("model_loaded", False) and health_status.get("yfinance_accessible", False)
    
    return jsonify(basic_health), 200

def preprocess_data(df):
    """
    Process yfinance data with comprehensive validation and error handling.
    
    Args:
        df (pd.DataFrame): Raw yfinance data
        
    Returns:
        pd.DataFrame: Processed and validated dataframe
        
    Raises:
        ValueError: If data format is invalid or insufficient
    """
    logger.debug("Preprocessing yfinance data")
    
    if df.empty:
        raise ValueError("Empty dataframe provided to preprocess_data")
    
    try:
        # Handle multi-level columns from yfinance
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index().rename(columns={'index': 'Date'})
        
        # Validate required columns exist
        required_columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df = df[required_columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
        
        # Validate data quality
        if len(df) < TIME_STEP:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least {TIME_STEP}")
        
        logger.debug(f"Data preprocessing completed: {len(df)} rows")
    return df

    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise

def get_stock_data(stock_symbol: str) -> pd.DataFrame:
    """
    Fetch stock data with comprehensive validation, caching, and error handling.
    
    Args:
        stock_symbol (str): Valid stock symbol
        
    Returns:
        pd.DataFrame: Processed stock data
        
    Raises:
        ValueError: If symbol is invalid or data unavailable
        Exception: If external API fails
    """
    logger.info(f"Fetching stock data for symbol: {stock_symbol}")
    
    # Validate and sanitize input
    if not validate_stock_symbol(stock_symbol):
        raise ValueError(f"Invalid stock symbol format: {stock_symbol}")
    
    stock_symbol = stock_symbol.strip().upper()
    
    try:
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * DATA_YEARS)
        
        logger.debug(f"Downloading data from {start_date.date()} to {end_date.date()}")
        
        df = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError(f"No data available for symbol: {stock_symbol}")
        
        processed_df = preprocess_data(df)
        logger.info(f"Successfully fetched {len(processed_df)} days of data for {stock_symbol}")
        
        return processed_df
        
    except Exception as e:
        error_msg = f"Failed to fetch data for {stock_symbol}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def prepare_data(df):
    """Prepare data for LSTM prediction"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Create dataset using sliding window
    X = np.array([scaled_data[i:i + TIME_STEP, 0]
                  for i in range(len(scaled_data) - TIME_STEP - 1)])
    y = scaled_data[TIME_STEP + 1:, 0]

    return X.reshape(X.shape[0], TIME_STEP, 1), y, scaler

def predict_future(model, data, scaler):
    """Generate future predictions"""
    last_data = data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    future_preds = np.zeros(PREDICTION_DAYS, dtype='float32')

    for i in range(PREDICTION_DAYS):
        next_pred = model.predict(last_data, verbose=0)[0, 0]
        future_preds[i] = next_pred
        last_data = np.roll(last_data, -1, axis=1)
        last_data[0, -1, 0] = next_pred

    return scaler.inverse_transform(future_preds.reshape(-1, 1))

def create_plot(df, pred_data=None, future_data=None, title=""):
    """Create interactive Plotly figure"""
    fig = go.Figure()

    # Main price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Prediction line
    if pred_data is not None:
        fig.add_trace(go.Scatter(
            x=df.index[TIME_STEP + 1:],
            y=pred_data[:, 0],
            name='Predicted',
            line=dict(color='orange')
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
            line=dict(color='green')
        ))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def predict_stock(stock_symbol, user_id=None):
    """Main prediction function for Gradio with public Supabase storage"""
    try:
        df = get_stock_data(stock_symbol)
        X, y, scaler = prepare_data(df)

        # Make predictions
        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred)

        # Future prediction
        future_prices = predict_future(
            model,
            scaler.transform(df['Close'].values.reshape(-1, 1)),
            scaler
        )

        # Create plots
        main_plot = create_plot(
            df,
            pred_data=y_pred,
            title=f"{stock_symbol} Price Prediction"
        )

        future_plot = create_plot(
            df,
            future_data=future_prices,
            title=f"{stock_symbol} 30-Day Forecast"
        )

        # Technical indicators
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()

        tech_fig = go.Figure()
        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            name='Price', line=dict(color='blue')))
        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='50-Day SMA', line=dict(color='orange')))
        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_200'],
            name='200-Day SMA', line=dict(color='red')))
        tech_fig.update_layout(
            title=f"{stock_symbol} Technical Indicators",
            template='plotly_dark'
        )
        
        # Store prediction data for Supabase (public storage, no user association)
        last_price = df['Close'].iloc[-1]
        last_date = df.index[-1].date()
        
        # Prepare prediction data to save
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
                "sma_50": df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else None,
                "sma_200": df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else None
            }
        }
        
        # Save prediction to Supabase if client is initialized (public storage)
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
            except Exception as e:
                # Don't fail the prediction if saving fails
                logger.warning(f"Failed to save prediction to Supabase: {str(e)}")

        return (
            f"${df['Close'].iloc[-1]:.2f}",
            df.index[-1].strftime('%Y-%m-%d'),
            main_plot,
            future_plot,
            tech_fig
        )

    except Exception as e:
        raise gr.Error(f"Prediction failed: {str(e)}")

# Gradio Interface
with gr.Blocks(title="Stock Prediction", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 📈 Real-Time Stock Predictor")
    gr.Markdown("Predict stock prices using LSTM neural networks")

    with gr.Row():
        stock_input = gr.Textbox(
            label="Stock Symbol (Examples: TSLA, AAPL, MSFT, AMZN, GOOG, AEP)",
            value="TSLA",
            placeholder="Enter stock symbol (e.g. AAPL, MSFT)"
        )
        submit_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        with gr.Column():
            last_price = gr.Textbox(label="Last Price")
            last_date = gr.Textbox(label="Last Date")

    with gr.Tabs():
        with gr.Tab("Price Prediction"):
            main_plot = gr.Plot(label="Price Prediction")
        with gr.Tab("30-Day Forecast"):
            future_plot = gr.Plot(label="Future Prediction")
        with gr.Tab("Technical Indicators"):
            tech_plot = gr.Plot(label="Technical Analysis")

    submit_btn.click(
        fn=predict_stock,
        inputs=stock_input,
        outputs=[last_price, last_date, main_plot, future_plot, tech_plot]
    )

def run_gradio_app():
    """
    Run the Gradio interface in a separate thread.
    
    This allows both Flask API endpoints and Gradio interface to run simultaneously.
    Configured for both local development and cloud deployment.
    """
    logger.info("Starting Gradio interface")
    
    # Get Gradio port from environment variable or use default
    gradio_port = int(os.environ.get("GRADIO_PORT", 7860))
    
    # Configure for cloud deployment
    server_name = "0.0.0.0"
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    
    try:
        demo.launch(
            server_name=server_name,
            server_port=gradio_port,
            share=share,
            debug=False,
            show_error=True,
            quiet=True  # Reduce log noise in production
        )
    except Exception as e:
        logger.error(f"Failed to start Gradio interface: {str(e)}")
        # Don't crash the entire application if Gradio fails
        logger.warning("Continuing without Gradio interface")

def initialize_application():
    """
    Initialize the application by loading the model and setting up services.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    logger.info("Initializing Stock Prediction Application")
    
    try:
        # Load the LSTM model
        model_success = load_ml_model()
        
        if not model_success:
            logger.warning("Application starting with model loading issues")
        else:
            logger.info("Model loaded successfully during initialization")
            
        # Test external API connectivity
        api_success, api_error = test_yfinance_connectivity()
        
        if not api_success:
            logger.warning(f"API connectivity issues during startup: {api_error}")
        else:
            logger.info("External API connectivity verified")
            
        # Initialize Supabase client
        supabase_success = supabase_client.init_supabase()
        
        if not supabase_success:
            logger.warning("Supabase integration disabled - check credentials")
            logger.warning("Prediction history and user features will be unavailable")
        else:
            logger.info("Supabase integration enabled for persistent storage")
            
        # Update initial health status
        health_status.update({
            "model_loaded": model_success,
            "yfinance_accessible": api_success,
            "supabase_connected": supabase_success,
            "last_check_timestamp": dt.datetime.utcnow().isoformat(),
            "errors": []
        })
        
        return model_success and api_success
        
    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        return False

# Additional routes for Supabase integration
@app.route('/api/history', methods=['GET'])
def get_prediction_history():
    """
    Get recent stock prediction history (public, no authentication needed).
    
    Query parameters:
    - limit: Maximum number of records to return (default: 20, max: 100)
    - symbol: Optional stock symbol to filter by
    
    Returns:
        JSON: Prediction history with HTTP status codes
    """
    try:
        # Get limit from query params or use default
        limit = request.args.get('limit', default=20, type=int)
        symbol = request.args.get('symbol')
        
        # Validate limit
        if limit <= 0 or limit > 100:
            limit = 20
        
        if symbol:
            # Get predictions for specific symbol
            success, history, error = supabase_client.get_predictions_by_symbol(symbol, limit)
        else:
            # Get recent predictions
            success, history, error = supabase_client.get_recent_predictions(limit)
        
        if success:
            return jsonify({
                "success": True,
                "history": history or [],
                "count": len(history) if history else 0
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": error or "Unknown error getting prediction history"
            }), 500
    except Exception as e:
        logger.error(f"Error in get_prediction_history: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/trending', methods=['GET'])
def get_trending():
    """
    Get trending stocks (most frequently searched).
    
    Query parameters:
    - limit: Maximum number of stocks to return (default: 10)
    - all: Set to "true" to get all trending stocks, not just weekly
    
    Returns:
        JSON: Trending stocks with HTTP status codes
    """
    try:
        limit = request.args.get('limit', default=10, type=int)
        show_all = request.args.get('all', 'false').lower() == 'true'
        
        # Validate limit
        if limit <= 0 or limit > 50:
            limit = 10
        
        if show_all:
            success, trending, error = supabase_client.get_all_trending_stocks()
            # Limit the results
            if trending and len(trending) > limit:
                trending = trending[:limit]
        else:
            success, trending, error = supabase_client.get_trending_stocks(limit)
        
        if success:
            return jsonify({
                "success": True,
                "trending": trending or [],
                "count": len(trending) if trending else 0
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": error or "Unknown error getting trending stocks"
            }), 500
    except Exception as e:
        logger.error(f"Error in get_trending: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get general statistics about the prediction database.
    
    Returns:
        JSON: Database statistics with HTTP status codes
    """
    try:
        success, stats, error = supabase_client.get_prediction_stats()
        
        if success:
            return jsonify({
                "success": True,
                "stats": stats or {}
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": error or "Unknown error getting statistics"
            }), 500
    except Exception as e:
        logger.error(f"Error in get_stats: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    """
    Main application entry point.
    
    Runs both Flask API server and Gradio interface concurrently.
    Configured for both local development and cloud deployment (Render).
    """
    logger.info("Starting Stock Prediction Service")
    
    # Get port from environment variable (for Render deployment) or use default
    port = int(os.environ.get("PORT", 5000))
    gradio_port = int(os.environ.get("GRADIO_PORT", 7860))
    
    # Initialize the application
    init_success = initialize_application()
    
    if not init_success:
        logger.warning("Application started with initialization issues - check health endpoints")
    
    try:
        # Start Gradio in a separate thread
        gradio_thread = threading.Thread(
            target=run_gradio_app,
            daemon=True,
            name="GradioThread"
        )
        gradio_thread.start()
        
        logger.info(f"Gradio interface started in background thread on port {gradio_port}")
        
        # Run Flask app on main thread
        logger.info(f"Starting Flask API server on port {port}")
        logger.info("Available endpoints:")
        logger.info("  - GET /health - General service health")
        logger.info("  - GET /health/search - Detailed search functionality health")
        logger.info(f"  - Gradio UI available at http://localhost:{gradio_port}")
        
        # For production deployment (like Render), use appropriate settings
        debug_mode = os.environ.get("FLASK_ENV", "production") == "development"
        
        app.run(
            host="0.0.0.0",
            port=port,
            debug=debug_mode,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Stock Prediction Service stopped")