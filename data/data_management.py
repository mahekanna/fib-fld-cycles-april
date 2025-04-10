"""
This module provides a compatibility wrapper for the DataFetcher class.
It preserves the API interface expected by the application while using the
original implementation under the hood.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import sys
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# Import the original DataFetcher class
from .fetcher import DataFetcher as OriginalDataFetcher

# Import centralized logging system if available
try:
    from utils.logging_utils import get_component_logger
    DEFAULT_LOGGER = get_component_logger("data.data_management")
except ImportError:
    # Fallback to standard logging
    logging.basicConfig(level=logging.INFO)
    DEFAULT_LOGGER = logging.getLogger(__name__)


class DataFetcher:
    """
    Compatibility wrapper for the original DataFetcher class.
    This class provides the expected API interface for the application
    while using the original implementation for data fetching.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataFetcher.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or DEFAULT_LOGGER
        
        # Set up cache directory
        self.cache_dir = config.get('data', {}).get('cache_dir', 'data/cache')
        
        # Initialize the original DataFetcher
        self.fetcher = OriginalDataFetcher(cache_dir=self.cache_dir)
        self.logger.info("Original DataFetcher initialized")
    
    def get_data(self, 
                symbol: str, 
                exchange: Optional[str] = None,
                interval: str = 'daily',
                lookback: int = 1000,
                source: Optional[str] = None,
                use_cache: bool = True,
                force_download: bool = False,
                price_source: str = 'close') -> Optional[pd.DataFrame]:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Symbol to fetch data for
            exchange: Exchange code (optional, will use default if not specified)
            interval: Time interval ('daily', '4h', '1h', etc.)
            lookback: Number of bars to fetch
            source: Data source to use (ignored, for compatibility)
            use_cache: Whether to use cached data if available
            force_download: Whether to force download fresh data
            price_source: Price source to use ('close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4', 'hlcc4')
            
        Returns:
            DataFrame with price data or None if fetch failed
        """
        # Determine exchange if not provided
        if not exchange:
            exchange = self.config.get('general', {}).get('default_exchange', 'NSE')
        
        # Use original fetcher with renamed parameters
        data = self.fetcher.get_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=lookback,
            use_cache=use_cache and not force_download
        )
        
        if data is None:
            return None
            
        # Ensure price column exists with desired price source
        if 'price' not in data.columns:
            data = self.process_price_source(data, price_source)
        
        return data
    
    def process_price_source(self, data: pd.DataFrame, price_source: str = 'close') -> pd.DataFrame:
        """
        Process the price data according to the specified price source.
        
        Args:
            data: DataFrame with OHLCV data
            price_source: Price source to use ('close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4', 'hlcc4')
            
        Returns:
            DataFrame with processed price data including the selected price source
        """
        # Make sure we have all required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns in price data: {missing_columns}. Using 'close' as fallback.")
            # If any required column is missing, use close as the price source
            data['price'] = data['close']
            return data
            
        # Always ensure derived columns exist for cycle calculations, regardless of price source
        if 'hl2' not in data.columns:
            data['hl2'] = ((data['high'] + data['low']) / 2).round(2)
            
        if 'hlc3' not in data.columns:
            data['hlc3'] = ((data['high'] + data['low'] + data['close']) / 3).round(2)
            
        if 'ohlc4' not in data.columns:
            data['ohlc4'] = ((data['open'] + data['high'] + data['low'] + data['close']) / 4).round(2)
            
        if 'hlcc4' not in data.columns:
            # Weighted close (high, low, close, close)
            data['hlcc4'] = ((data['high'] + data['low'] + 2 * data['close']) / 4).round(2)
        
        # Calculate selected price source
        if price_source == 'close':
            data['price'] = data['close']
        elif price_source == 'open':
            data['price'] = data['open']
        elif price_source == 'high':
            data['price'] = data['high']
        elif price_source == 'low':
            data['price'] = data['low']
        elif price_source == 'hl2':
            data['price'] = data['hl2']
        elif price_source == 'hlc3':
            data['price'] = data['hlc3']
        elif price_source == 'ohlc4':
            data['price'] = data['ohlc4']
        elif price_source == 'hlcc4':
            data['price'] = data['hlcc4']
        else:
            # Default to close if price source is invalid
            self.logger.warning(f"Invalid price source: {price_source}. Using 'close' as fallback.")
            data['price'] = data['close']
            
        self.logger.info(f"Using price source: {price_source}")
            
        return data
    
    def get_historical_data(self,
                           symbol: str,
                           exchange: Optional[str] = None,
                           interval: str = 'daily',
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           lookback: int = 1000,
                           source: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get historical data between specified dates.
        
        Args:
            symbol: Symbol to fetch data for
            exchange: Exchange code
            interval: Time interval
            start_date: Start date
            end_date: End date
            lookback: Number of bars to fetch if no dates provided
            source: Data source to use
            
        Returns:
            DataFrame with price data or None if fetch failed
        """
        # Get full data first
        data = self.get_data(symbol, exchange, interval, lookback, source)
        
        if data is None:
            return None
        
        # If dates provided, filter data
        if start_date or end_date:
            if start_date:
                data = data[data.index >= start_date]
            
            if end_date:
                data = data[data.index <= end_date]
        
        return data
    
    def batch_get_data(self,
                      symbols: List[str],
                      exchange: Optional[str] = None,
                      interval: str = 'daily',
                      lookback: int = 1000,
                      source: Optional[str] = None,
                      max_workers: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols sequentially.
        
        Args:
            symbols: List of symbols to fetch
            exchange: Exchange code
            interval: Time interval
            lookback: Number of bars to fetch
            source: Data source to use
            max_workers: Maximum number of concurrent workers (ignored, for compatibility)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        # Process each symbol sequentially
        for symbol in symbols:
            try:
                data = self.get_data(symbol, exchange, interval, lookback, source)
                if data is not None:
                    results[symbol] = data
                    self.logger.info(f"Successfully fetched data for {symbol}")
                else:
                    self.logger.warning(f"Failed to fetch data for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
        
        self.logger.info(f"Batch fetch completed: {len(results)}/{len(symbols)} successful")
        return results
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            older_than_days: Only clear entries older than this many days (ignored, for compatibility)
            
        Returns:
            Number of entries cleared
        """
        # Use original fetcher's clear_cache method
        self.fetcher.clear_cache()
        return 1  # Return 1 to indicate success

# For backward compatibility, include the DataProcessor class
class DataProcessor:
    """
    Data preprocessing and normalization for market data.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_derived_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common derived price columns.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with additional derived columns
        """
        df = data.copy()
        
        # Make sure required columns exist
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            self.logger.error("Missing required columns in data")
            return df
        
        # Calculate basic price indicators
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Calculate basic returns
        df['daily_return'] = df['close'].pct_change() * 100
        
        # Calculate true range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Calculate log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def normalize_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize volume data to facilitate comparison.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with normalized volume
        """
        df = data.copy()
        
        if 'volume' not in df.columns:
            self.logger.warning("Volume column not found, skipping normalization")
            return df
        
        # Calculate average volume over 20 periods
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        
        # Normalize volume to average (> 1 means above average)
        df['volume_relative'] = df['volume'] / df['volume_sma20']
        
        return df
    
    def detect_gaps(self, data: pd.DataFrame, threshold_pct: float = 1.0) -> pd.DataFrame:
        """
        Detect price gaps between bars.
        
        Args:
            data: DataFrame with price data
            threshold_pct: Threshold percentage for gap detection
            
        Returns:
            DataFrame with gap information
        """
        df = data.copy()
        
        # Calculate gaps as percentage of previous close
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        
        # Identify significant gaps
        df['gap_up'] = df['gap_pct'] > threshold_pct
        df['gap_down'] = df['gap_pct'] < -threshold_pct
        
        return df
    
    def apply_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Preprocessed DataFrame
        """
        # Apply processing steps
        df = self.calculate_derived_columns(data)
        df = self.normalize_volume(df)
        df = self.detect_gaps(df)
        
        return df