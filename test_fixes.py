#!/usr/bin/env python3
"""
Test script to verify the fixes implemented in the Fibonacci Cycles Trading System.
This script checks:
1. Interval handling - ensures correct intervals are used and no silent defaulting to daily
2. Lookback parameter respect - ensures the lookback parameter is respected in visualizations
3. Data fetching with TvDatafeed - ensures data fetching works properly
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_fixes")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_interval_handling():
    """Test that the interval handling is correct and raises errors for invalid intervals."""
    logger.info("Testing interval handling...")
    
    from data.fetcher import DataFetcher
    fetcher = DataFetcher()
    
    # Test valid intervals
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', 'daily', 'weekly', 'monthly']
    for interval in valid_intervals:
        try:
            logger.info(f"Testing valid interval: {interval}")
            data = fetcher.get_data('NIFTY', 'NSE', interval, 10, use_cache=True)
            if data is not None:
                logger.info(f"✅ Successfully fetched data for interval {interval} - got {len(data)} rows")
            else:
                logger.warning(f"⚠️ No data returned for interval {interval}, but no error was raised")
        except Exception as e:
            logger.error(f"❌ Error fetching data for valid interval {interval}: {str(e)}")
    
    # Test invalid interval - should raise ValueError
    invalid_interval = 'invalid_interval'
    try:
        logger.info(f"Testing invalid interval: {invalid_interval}")
        data = fetcher.get_data('NIFTY', 'NSE', invalid_interval, 10, use_cache=True)
        logger.error(f"❌ No error raised for invalid interval {invalid_interval}")
    except ValueError as e:
        logger.info(f"✅ Correctly raised ValueError for invalid interval: {str(e)}")
    except Exception as e:
        logger.warning(f"⚠️ Raised unexpected error for invalid interval: {str(e)}")
    
    logger.info("Interval handling test completed")

def test_lookback_respect():
    """Test that the lookback parameter is respected in visualizations."""
    logger.info("Testing lookback parameter respect...")
    
    from core.scanner_system import FibCycleScanner
    from models.scan_parameters import ScanParameters
    from utils.config import load_config
    
    # Load config
    config_path = os.path.join(project_root, "config", "config.json")
    config = load_config(config_path)
    
    # Create scanner
    scanner = FibCycleScanner(config)
    
    # Test with different lookback values
    lookback_values = [100, 200, 500]
    
    for lookback in lookback_values:
        logger.info(f"Testing with lookback={lookback}")
        
        # Create scan parameters
        params = ScanParameters(
            symbol="NIFTY",
            exchange="NSE",
            interval="daily",
            lookback=lookback,
            generate_chart=True
        )
        
        # Perform scan
        result = scanner.analyze_symbol(params)
        
        # Check if the result has a chart
        if result.chart_image is not None:
            logger.info(f"✅ Successfully generated chart with lookback={lookback}")
        else:
            logger.warning(f"⚠️ No chart generated for lookback={lookback}")
        
        # Check if the stored lookback matches the requested lookback
        if result.lookback == lookback:
            logger.info(f"✅ Stored lookback value ({result.lookback}) matches requested lookback ({lookback})")
        else:
            logger.error(f"❌ Stored lookback value ({result.lookback}) differs from requested lookback ({lookback})")
    
    logger.info("Lookback parameter test completed")

def test_data_fetching():
    """Test that data fetching with TvDatafeed works properly."""
    logger.info("Testing data fetching with TvDatafeed...")
    
    from data.data_management import DataFetcher
    from utils.config import load_config
    
    # Load config
    config_path = os.path.join(project_root, "config", "config.json")
    config = load_config(config_path)
    
    # Create data fetcher
    fetcher = DataFetcher(config)
    
    # Test fetching data for different symbols and intervals
    test_cases = [
        {"symbol": "NIFTY", "exchange": "NSE", "interval": "daily", "lookback": 100},
        {"symbol": "NIFTY", "exchange": "NSE", "interval": "15m", "lookback": 100},
        {"symbol": "BANKNIFTY", "exchange": "NSE", "interval": "daily", "lookback": 100}
    ]
    
    for case in test_cases:
        symbol = case["symbol"]
        exchange = case["exchange"]
        interval = case["interval"]
        lookback = case["lookback"]
        
        logger.info(f"Testing data fetching for {symbol} ({exchange}) with interval={interval}, lookback={lookback}")
        
        try:
            # First try with cache
            data = fetcher.get_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                lookback=lookback,
                use_cache=True
            )
            
            if data is not None and not data.empty:
                logger.info(f"✅ Successfully fetched data with cache - got {len(data)} rows")
            else:
                # Try without cache
                logger.info("No data from cache, trying force download...")
                data = fetcher.get_data(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    lookback=lookback,
                    force_download=True
                )
                
                if data is not None and not data.empty:
                    logger.info(f"✅ Successfully fetched data with force download - got {len(data)} rows")
                else:
                    logger.error(f"❌ Failed to fetch data for {symbol} ({interval})")
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol} ({interval}): {str(e)}")
    
    logger.info("Data fetching test completed")

if __name__ == "__main__":
    logger.info("Starting test script for Fibonacci Cycles Trading System fixes")
    
    # Run all tests
    test_interval_handling()
    print("\n")
    test_lookback_respect()
    print("\n")
    test_data_fetching()
    
    logger.info("All tests completed")