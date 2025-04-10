"""
Real-time Data Refresher for Fibonacci Cycles Trading System

This module provides real-time data fetching capabilities for the trading system,
implementing various strategies for keeping price data up-to-date.
"""

import pandas as pd
import numpy as np
import time
import threading
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable

# Import centralized logging system if available
try:
    from utils.logging_utils import get_component_logger
    logger = get_component_logger("data.data_refresher")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import our data fetcher
from data.data_management import DataFetcher


class DataRefresher:
    """
    Real-time data refresher for keeping price data up-to-date.
    
    This class provides mechanisms for:
    1. Periodic automatic data refreshing
    2. Forced immediate refresh
    3. Refresh prioritization based on symbol and interval
    4. Callback notifications when data is updated
    """
    
    def __init__(self, config: Dict, data_fetcher: Optional[DataFetcher] = None):
        """
        Initialize the DataRefresher.
        
        Args:
            config: Configuration dictionary
            data_fetcher: Optional DataFetcher instance
        """
        self.config = config
        self.data_fetcher = data_fetcher or DataFetcher(config)
        
        # Initialize refresh intervals (in seconds)
        self.refresh_intervals = {
            '1m': 30,          # 30 seconds for 1-minute data
            '5m': 60,          # 1 minute for 5-minute data
            '15m': 60 * 3,     # 3 minutes for 15-minute data
            '30m': 60 * 5,     # 5 minutes for 30-minute data
            '1h': 60 * 10,     # 10 minutes for 1-hour data
            '4h': 60 * 30,     # 30 minutes for 4-hour data
            'daily': 60 * 60,  # 1 hour for daily data
            'weekly': 60 * 60 * 6,  # 6 hours for weekly data
            'monthly': 60 * 60 * 24  # 24 hours for monthly data
        }
        
        # Last refresh timestamps keyed by "symbol_exchange_interval"
        self.last_refresh = {}
        
        # Data cache keyed by "symbol_exchange_interval"
        self.data_cache = {}
        
        # Subscriptions keyed by "symbol_exchange_interval"
        self.subscriptions = {}
        
        # Refresh thread
        self.refresh_thread = None
        self.running = False
        self.stop_event = threading.Event()
        
        # Registered callbacks for data updates
        self.callbacks = {}
        
        # Priority symbols that should be refreshed more frequently
        self.priority_symbols = set()
        
        logger.info("DataRefresher initialized")
    
    def start_refresh_thread(self, interval: int = 10):
        """
        Start the refresh thread.
        
        Args:
            interval: Check interval in seconds
        """
        if self.refresh_thread and self.refresh_thread.is_alive():
            logger.warning("Refresh thread already running")
            return
        
        self.running = True
        self.stop_event.clear()
        self.refresh_thread = threading.Thread(
            target=self._refresh_worker, 
            args=(interval,),
            daemon=True
        )
        self.refresh_thread.start()
        logger.info(f"Refresh thread started with {interval}s check interval")
    
    def stop_refresh_thread(self):
        """Stop the refresh thread."""
        self.stop_event.set()
        self.running = False
        if self.refresh_thread:
            self.refresh_thread.join(timeout=5)
            logger.info("Refresh thread stopped")
    
    def _refresh_worker(self, interval: int = 10):
        """
        Worker function for the refresh thread.
        
        Args:
            interval: Check interval in seconds
        """
        logger.info("Refresh worker thread started")
        
        while not self.stop_event.is_set():
            try:
                self._check_and_refresh_all()
            except Exception as e:
                logger.error(f"Error in refresh worker: {e}")
            
            # Sleep for the specified interval, but check stop_event periodically
            for _ in range(interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    def _check_and_refresh_all(self):
        """Check and refresh all subscribed symbols that need updating."""
        current_time = time.time()
        
        # Get all subscriptions
        for key in list(self.subscriptions.keys()):
            try:
                symbol, exchange, interval = key.split('_')
                
                # Get last refresh time
                last_refresh = self.last_refresh.get(key, 0)
                
                # Get refresh interval
                refresh_interval = self.refresh_intervals.get(interval, 300)  # Default to 5 minutes
                
                # If symbol is priority, reduce interval
                if symbol in self.priority_symbols:
                    refresh_interval = max(refresh_interval // 2, 10)  # At least 10 seconds
                
                # Check if refresh is needed
                if current_time - last_refresh >= refresh_interval:
                    logger.info(f"Refreshing data for {symbol} ({interval})")
                    self._refresh_symbol(symbol, exchange, interval)
            except Exception as e:
                logger.error(f"Error checking/refreshing {key}: {e}")
    
    def _refresh_symbol(self, symbol: str, exchange: str, interval: str):
        """
        Refresh data for a specific symbol.
        
        Args:
            symbol: Symbol to refresh
            exchange: Exchange to fetch from
            interval: Time interval
        """
        key = f"{symbol}_{exchange}_{interval}"
        
        try:
            # Fetch fresh data without using cache
            data = self.data_fetcher.get_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                use_cache=False,
                force_download=True
            )
            
            if data is None:
                logger.warning(f"Failed to refresh data for {symbol} ({interval})")
                return
            
            # Update last refresh timestamp
            self.last_refresh[key] = time.time()
            
            # Update data cache
            self.data_cache[key] = data
            
            # Notify subscribers
            self._notify_subscribers(key, data)
            
            logger.info(f"Successfully refreshed data for {symbol} ({interval}), {len(data)} rows")
            
        except Exception as e:
            logger.error(f"Error refreshing data for {symbol} ({interval}): {e}")
    
    def _notify_subscribers(self, key: str, data: pd.DataFrame):
        """
        Notify subscribers about data updates.
        
        Args:
            key: Subscription key (symbol_exchange_interval)
            data: Updated data
        """
        callbacks = self.callbacks.get(key, [])
        
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {key}: {e}")
    
    def subscribe(self, symbol: str, exchange: str, interval: str, callback: Optional[Callable] = None):
        """
        Subscribe to data updates for a symbol.
        
        Args:
            symbol: Symbol to subscribe to
            exchange: Exchange to fetch from
            interval: Time interval
            callback: Optional callback function to receive updates
        """
        key = f"{symbol}_{exchange}_{interval}"
        
        # Add to subscriptions
        self.subscriptions[key] = {
            'symbol': symbol,
            'exchange': exchange,
            'interval': interval
        }
        
        # Register callback if provided
        if callback:
            if key not in self.callbacks:
                self.callbacks[key] = []
            self.callbacks[key].append(callback)
        
        logger.info(f"Subscribed to {symbol} ({interval})")
        
        # Perform initial refresh if needed
        if key not in self.data_cache:
            self._refresh_symbol(symbol, exchange, interval)
        elif callback and key in self.data_cache:
            # Call callback with existing data
            callback(self.data_cache[key])
    
    def unsubscribe(self, symbol: str, exchange: str, interval: str, callback: Optional[Callable] = None):
        """
        Unsubscribe from data updates.
        
        Args:
            symbol: Symbol to unsubscribe from
            exchange: Exchange
            interval: Time interval
            callback: Optional specific callback to remove (if None, remove all)
        """
        key = f"{symbol}_{exchange}_{interval}"
        
        # Remove specific callback if provided
        if callback and key in self.callbacks:
            if callback in self.callbacks[key]:
                self.callbacks[key].remove(callback)
            
            # If no more callbacks, remove the key
            if not self.callbacks[key]:
                del self.callbacks[key]
        
        # If no callback specified or no more callbacks, remove subscription
        if callback is None or key not in self.callbacks:
            if key in self.subscriptions:
                del self.subscriptions[key]
            
            if key in self.callbacks:
                del self.callbacks[key]
        
        logger.info(f"Unsubscribed from {symbol} ({interval})")
    
    def get_latest_data(self, symbol: str, exchange: str, interval: str, 
                        refresh_if_needed: bool = True, lookback: int = 1000,
                        price_source: str = 'close') -> Optional[pd.DataFrame]:
        """
        Get the latest data for a symbol, refreshing if needed.
        
        Args:
            symbol: Symbol to get data for
            exchange: Exchange to fetch from
            interval: Time interval
            refresh_if_needed: Whether to refresh data if too old
            lookback: Number of bars to return
            price_source: Price source to use
            
        Returns:
            Latest data as DataFrame
        """
        key = f"{symbol}_{exchange}_{interval}"
        
        # Check if we need to refresh
        if refresh_if_needed:
            current_time = time.time()
            last_refresh = self.last_refresh.get(key, 0)
            refresh_interval = self.refresh_intervals.get(interval, 300)
            
            if current_time - last_refresh >= refresh_interval:
                logger.info(f"Refreshing data for {symbol} due to age")
                self._refresh_symbol(symbol, exchange, interval)
        
        # Return from cache if available
        if key in self.data_cache:
            return self.data_cache[key].copy()
        
        # Otherwise, fetch and cache
        data = self.data_fetcher.get_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            lookback=lookback,
            price_source=price_source
        )
        
        if data is not None:
            self.data_cache[key] = data
            self.last_refresh[key] = time.time()
        
        return data
    
    def add_priority_symbol(self, symbol: str):
        """
        Add a symbol to the priority list for more frequent updates.
        
        Args:
            symbol: Symbol to prioritize
        """
        self.priority_symbols.add(symbol)
        logger.info(f"Added {symbol} to priority symbols")
    
    def remove_priority_symbol(self, symbol: str):
        """
        Remove a symbol from the priority list.
        
        Args:
            symbol: Symbol to remove from priorities
        """
        if symbol in self.priority_symbols:
            self.priority_symbols.remove(symbol)
            logger.info(f"Removed {symbol} from priority symbols")
    
    def refresh_all(self):
        """Force refresh all subscribed symbols."""
        for key in list(self.subscriptions.keys()):
            try:
                symbol, exchange, interval = key.split('_')
                logger.info(f"Force refreshing {symbol} ({interval})")
                self._refresh_symbol(symbol, exchange, interval)
            except Exception as e:
                logger.error(f"Error refreshing {key}: {e}")
    
    def refresh_symbol(self, symbol: str, exchange: str, interval: str):
        """
        Force refresh a specific symbol.
        
        Args:
            symbol: Symbol to refresh
            exchange: Exchange to fetch from
            interval: Time interval
        """
        logger.info(f"Force refreshing {symbol} ({interval})")
        self._refresh_symbol(symbol, exchange, interval)


# Global singleton instance
_data_refresher = None

def get_data_refresher(config: Optional[Dict] = None) -> DataRefresher:
    """
    Get the global data refresher instance.
    
    Args:
        config: Optional configuration to initialize with
        
    Returns:
        DataRefresher instance
    """
    global _data_refresher
    
    if _data_refresher is None:
        if config is None:
            # Try to load config
            try:
                from utils.config import load_config
                config = load_config()
            except ImportError:
                # Use default config
                config = {
                    'data': {
                        'cache_dir': 'data/cache'
                    }
                }
        
        _data_refresher = DataRefresher(config)
    
    return _data_refresher