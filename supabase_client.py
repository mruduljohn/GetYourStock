import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file (if present)
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase: Optional[Client] = None

def init_supabase() -> bool:
    """
    Initialize Supabase client with environment variables.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global supabase
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials not found in environment variables")
        return False
        
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        return False

# Public prediction storage functions

def save_prediction(
    stock_symbol: str,
    last_price: float,
    last_date: Union[str, datetime.date],
    prediction_days: int,
    prediction_data: Dict
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Save a stock prediction to the public database.
    
    Args:
        stock_symbol: Stock ticker symbol
        last_price: Last closing price
        last_date: Date of last price
        prediction_days: Number of days in prediction
        prediction_data: JSON serializable prediction data
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - Inserted record (Dict) if successful, None otherwise
        - Error message (str) if failed, None otherwise
    """
    if not supabase:
        return False, None, "Supabase client not initialized"
        
    try:
        # Convert prediction data to JSON string if needed
        if not isinstance(prediction_data, str):
            prediction_data_json = json.dumps(prediction_data)
        else:
            prediction_data_json = prediction_data
            
        # Handle date format
        if isinstance(last_date, datetime.date):
            last_date = last_date.isoformat()
            
        # Create record
        record = {
            "stock_symbol": stock_symbol,
            "last_price": last_price,
            "last_date": last_date,
            "prediction_days": prediction_days,
            "prediction_data": prediction_data_json
        }
            
        # Insert into database
        result = supabase.table("public_predictions").insert(record).execute()
        
        if result.data:
            logger.info(f"Saved public prediction for {stock_symbol}")
            return True, result.data[0], None
        else:
            logger.warning("Prediction saved but no data returned")
            return True, None, None
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to save prediction: {error_msg}")
        return False, None, error_msg

def get_recent_predictions(limit: int = 20) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
    """
    Get recent stock predictions from the public database.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - List of prediction records if successful, None otherwise
        - Error message if failed, None otherwise
    """
    if not supabase:
        return False, None, "Supabase client not initialized"
        
    try:
        result = supabase.table("public_predictions") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
            
        return True, result.data, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get recent predictions: {error_msg}")
        return False, None, error_msg

def get_predictions_by_symbol(stock_symbol: str, limit: int = 10) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
    """
    Get recent predictions for a specific stock symbol.
    
    Args:
        stock_symbol: Stock ticker symbol
        limit: Maximum number of records to return
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - List of prediction records if successful, None otherwise
        - Error message if failed, None otherwise
    """
    if not supabase:
        return False, None, "Supabase client not initialized"
        
    try:
        result = supabase.table("public_predictions") \
            .select("*") \
            .eq("stock_symbol", stock_symbol) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
            
        return True, result.data, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get predictions for {stock_symbol}: {error_msg}")
        return False, None, error_msg

# Trending stocks functions

def get_trending_stocks(limit: int = 10) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
    """
    Get the trending stocks (most searched).
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - List of trending stock records if successful, None otherwise
        - Error message if failed, None otherwise
    """
    if not supabase:
        return False, None, "Supabase client not initialized"
        
    try:
        result = supabase.table("weekly_trending_stocks") \
            .select("*") \
            .limit(limit) \
            .execute()
            
        return True, result.data, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get trending stocks: {error_msg}")
        return False, None, error_msg

def get_all_trending_stocks() -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
    """
    Get all trending stocks (not just weekly).
    
    Returns:
        Tuple containing:
        - Success status (bool)
        - List of all trending stock records if successful, None otherwise
        - Error message if failed, None otherwise
    """
    if not supabase:
        return False, None, "Supabase client not initialized"
        
    try:
        result = supabase.table("trending_stocks") \
            .select("*") \
            .order("search_count", desc=True) \
            .limit(50) \
            .execute()
            
        return True, result.data, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get all trending stocks: {error_msg}")
        return False, None, error_msg

# Statistics functions

def get_prediction_stats() -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Get basic statistics about predictions in the database.
    
    Returns:
        Tuple containing:
        - Success status (bool)
        - Statistics dict if successful, None otherwise
        - Error message if failed, None otherwise
    """
    if not supabase:
        return False, None, "Supabase client not initialized"
        
    try:
        # Get total predictions count
        total_result = supabase.table("public_predictions") \
            .select("id", count="exact") \
            .execute()
        
        # Get unique symbols count  
        symbols_result = supabase.table("public_predictions") \
            .select("stock_symbol") \
            .execute()
        
        unique_symbols = len(set(record['stock_symbol'] for record in symbols_result.data)) if symbols_result.data else 0
        
        # Get recent activity (last 24 hours)
        recent_result = supabase.table("public_predictions") \
            .select("id", count="exact") \
            .gte("created_at", (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat()) \
            .execute()
        
        stats = {
            "total_predictions": total_result.count if total_result.count else 0,
            "unique_symbols": unique_symbols,
            "predictions_last_24h": recent_result.count if recent_result.count else 0,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return True, stats, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get prediction stats: {error_msg}")
        return False, None, error_msg

# Initialize on module import
init_success = init_supabase() 