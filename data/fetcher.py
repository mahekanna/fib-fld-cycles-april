"""
Data fetcher module that handles fetching financial data from various sources.
This implementation is based on the original golden cycles implementation.
"""

import pandas as pd
import numpy as np
import os
import time
import hashlib
import pickle
import logging
from tvDatafeed import TvDatafeed, Interval

# Try to get logger from centralized logging
try:
    from utils.logging_utils import get_component_logger
    logger = get_component_logger("data.fetcher")
except ImportError:
    # Fallback to standard logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class DataFetcher:
    """Data manager for fetching and caching financial data"""
    
    def __init__(self, cache_dir="./data/cache", max_age=86400):
        """
        Initialize data manager
        
        Args:
            cache_dir: Directory for data cache
            max_age: Maximum age of cached data in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir
        self.max_age = max_age
        self.tv = None
        self.memory_cache = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize TvDatafeed client
        self._init_tv_client()
    
    def _init_tv_client(self):
        """Initialize TvDatafeed client"""
        try:
            self.tv = TvDatafeed()
            logger.info("TvDatafeed client initialized")
        except Exception as e:
            logger.error(f"Error initializing TvDatafeed client: {e}")
            self.tv = None
    
    def get_data(self, symbol, exchange="NSE", interval="daily", n_bars=5000, use_cache=True):
        """
        Get historical data for a symbol
        
        Args:
            symbol: Symbol to fetch data for
            exchange: Exchange to fetch data from
            interval: Interval to fetch data for (e.g., "daily", "15min")
            n_bars: Number of bars to fetch
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with historical data
        """
        # Convert string interval to TvDatafeed Interval enum
        interval_map = {
            # Standard formats
            "1min": Interval.in_1_minute,
            "5min": Interval.in_5_minute,
            "15min": Interval.in_15_minute,
            "30min": Interval.in_30_minute,
            "1h": Interval.in_1_hour,
            "2h": Interval.in_2_hour,
            "4h": Interval.in_4_hour,
            "daily": Interval.in_daily,
            "weekly": Interval.in_weekly,
            "monthly": Interval.in_monthly,
            
            # For backward compatibility with dashboard UI
            "1m": Interval.in_1_minute,
            "5m": Interval.in_5_minute,
            "15m": Interval.in_15_minute,
            "30m": Interval.in_30_minute,
            
            # Other common variations
            "1d": Interval.in_daily,
            "1day": Interval.in_daily,
            "1w": Interval.in_weekly,
            "1month": Interval.in_monthly
        }
        
        # Critical fix: Raise an error for unrecognized intervals instead of silently defaulting to daily
        if interval not in interval_map:
            error_msg = f"Unrecognized interval: '{interval}'. Must be one of: {list(interval_map.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        tv_interval = interval_map[interval]
        logger.info(f"Using interval: {interval} (TV interval: {tv_interval})")
        
        # Generate cache key
        cache_key = f"{symbol}_{exchange}_{interval}_{n_bars}"
        
        # Try to get from cache
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached data for {symbol} {interval}")
                return cached_data
        
        # Initialize TV client if needed
        if self.tv is None:
            self._init_tv_client()
            if self.tv is None:
                logger.error("TvDatafeed client not available")
                return None
        
        try:
            # Fetch data
            logger.info(f"Fetching {n_bars} bars of {symbol} {interval} data from {exchange}")
            data = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=tv_interval,
                n_bars=n_bars
            )
            
            if data is None or data.empty:
                logger.warning(f"No data returned for {symbol} on {exchange}")
                return None
                
            # Cache the data
            if use_cache:
                self._save_to_cache(cache_key, data)
            
            logger.info(f"Successfully fetched {len(data)} bars for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _get_cache_path(self, cache_key):
        """Get file path for cache key"""
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_hash}.pkl")
    
    def _get_from_cache(self, cache_key):
        """Get data from cache if available and not expired"""
        # Check memory cache first
        if cache_key in self.memory_cache:
            timestamp, data = self.memory_cache[cache_key]
            if time.time() - timestamp <= self.max_age:
                return data
            # Remove expired entry
            del self.memory_cache[cache_key]
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            # Check if expired
            if time.time() - os.path.getmtime(cache_path) > self.max_age:
                # Remove expired file
                os.remove(cache_path)
                return None
            
            try:
                # Load from cache
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Add to memory cache
                self.memory_cache[cache_key] = (time.time(), data)
                
                return data
            except Exception as e:
                logger.error(f"Error loading from cache: {e}")
                # Remove corrupted cache file
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        
        return None
    
    def _save_to_cache(self, cache_key, data):
        """Save data to cache"""
        try:
            # Save to memory cache
            self.memory_cache[cache_key] = (time.time(), data)
            
            # Save to file cache
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def clear_cache(self):
        """Clear all cached data"""
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear file cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def load_symbols_from_file(self, file_path):
        """
        Load symbols from a file
        
        Args:
            file_path: Path to file containing symbols
            
        Returns:
            List of symbols
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Symbol file not found: {file_path}")
                return []
            
            # Determine file type
            if file_path.endswith('.csv'):
                # CSV file
                df = pd.read_csv(file_path)
                
                # Try to find symbol column
                symbol_cols = [col for col in df.columns if any(term in col.lower() 
                              for term in ['symbol', 'ticker', 'name', 'scrip'])]
                
                if symbol_cols:
                    symbols = df[symbol_cols[0]].dropna().astype(str).str.strip().tolist()
                else:
                    # Use first column as fallback
                    symbols = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            else:
                # Text file
                with open(file_path, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
            
            # Convert to uppercase
            symbols = [s.upper() for s in symbols]
            
            logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
            return symbols
            
        except Exception as e:
            logger.error(f"Error loading symbols from {file_path}: {e}")
            return []