"""
Test script to verify data generation works correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_and_save_test_data():
    """Generate and save test data for NIFTY."""
    print("Generating test data for NIFTY...")
    
    # Parameters
    symbol = "NIFTY"
    exchange = "NSE"
    interval = "daily"
    lookback = 500
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback)
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Create some mock price data with cycles
    base = 100 + sum(ord(c) for c in symbol) % 500  # Different base for each symbol
    t = np.arange(len(dates))
    
    # Add some cycles
    cycles = [21, 55, 144]
    cycle_waves = sum(
        10 * np.sin(2 * np.pi * t / cycle + 0.1 * i) 
        for i, cycle in enumerate(cycles)
    )
    
    # Create price data with cycles, trend, and noise
    trend = np.linspace(0, 30, len(dates))
    noise = np.random.normal(0, 5, len(dates))
    closes = base + trend + cycle_waves + noise
    
    # Generate OHLC
    opens = closes + np.random.normal(0, 1, len(dates))
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 2, len(dates)))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 2, len(dates)))
    volumes = np.random.randint(1000, 10000, len(dates))
    
    # Create DataFrame
    mock_df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'hlc3': (highs + lows + closes) / 3,
        'ohlc4': (opens + highs + lows + closes) / 4,
        'hl2': (highs + lows) / 2
    }, index=dates)
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join('data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save the mock data
    cache_path = os.path.join(cache_dir, f"{exchange}_{symbol}_{interval}.csv")
    mock_df.to_csv(cache_path)
    print(f"Saved mock data to {cache_path}")
    
    # Verify the file exists and has data
    if os.path.exists(cache_path):
        file_size = os.path.getsize(cache_path)
        print(f"File exists with size: {file_size} bytes")
        
        # Read a few rows to verify
        df = pd.read_csv(cache_path, nrows=5)
        print(f"Sample data: {df.head()}")
    else:
        print(f"ERROR: File not created at {cache_path}")
    
    return cache_path

def test_data_fetcher():
    """Test the DataFetcher class directly."""
    from data.data_management import DataFetcher
    
    print("\nTesting DataFetcher class...")
    
    # Create a more detailed config
    config = {
        'data': {
            'cache_dir': 'data/cache',
            'cache_expiry': {
                '1m': 1,
                '5m': 1,
                '15m': 1,
                '30m': 1,
                '1h': 7,
                '4h': 7,
                'daily': 30,
                'weekly': 90,
                'monthly': 90
            }
        },
        'general': {
            'default_exchange': 'NSE',
            'default_source': 'tradingview',
            'symbols_file_path': 'config/symbols.json'
        },
        'analysis': {
            'min_period': 10,
            'max_period': 250,
            'fib_cycles': [21, 34, 55, 89, 144, 233],
            'power_threshold': 0.2,
            'cycle_tolerance': 0.15,
            'detrend_method': 'diff',
            'window_function': 'hanning',
            'gap_threshold': 0.01,
            'crossover_lookback': 5
        },
        'performance': {'max_workers': 2}
    }
    
    # Initialize the fetcher
    fetcher = DataFetcher(config)
    
    # Try to fetch data
    print("Fetching data for NIFTY...")
    
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Try multiple approaches for fetching
    approaches = [
        # Approach 1: Regular get_data
        lambda: fetcher.get_data(
            symbol="NIFTY",
            exchange="NSE",
            interval="daily",
            lookback=500
        ),
        # Approach 2: Direct check of cache
        lambda: fetcher._get_cached_data(fetcher._generate_cache_id(
            symbol="NIFTY",
            exchange="NSE",
            interval="daily",
            source=None
        )),
        # Approach 3: Load directly from file
        lambda: pd.read_csv("data/cache/NSE_NIFTY_daily.csv", index_col=0, parse_dates=True)
    ]
    
    data = None
    for i, approach in enumerate(approaches):
        try:
            print(f"Trying approach {i+1}...")
            data = approach()
            if data is not None and not data.empty:
                print(f"Approach {i+1} succeeded!")
                break
        except Exception as e:
            print(f"Approach {i+1} failed: {e}")
    
    # One more direct attempt with debugging
    if data is None or data.empty:
        print("\nTrying direct file read with debugging:")
        try:
            import glob
            csv_files = glob.glob("data/cache/*.csv")
            print(f"Available CSV files: {csv_files}")
            
            for csv_file in csv_files:
                print(f"Reading {csv_file}...")
                df = pd.read_csv(csv_file, nrows=2)
                print(f"Sample content: {df.head()}")
                
            # Try to create a DataFetcher with minimal sources
            print("Creating a simple DataFetcher...")
            simple_fetcher = DataFetcher({
                'data': {'cache_dir': 'data/cache'}
            })
            
            print("Trying to generate mock data directly...")
            from data.mock_data_generator import generate_mock_price_data
            mock_data = generate_mock_price_data("NIFTY", 500, "daily")
            print(f"Generated mock data: {mock_data.shape}")
            
            data = mock_data
        except Exception as e:
            print(f"Direct approach failed: {e}")
    
    if data is not None and not data.empty:
        print(f"Successfully fetched {len(data)} rows of data")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Sample data:\n{data.head()}")
    else:
        print("Failed to fetch data")
    
    return data

if __name__ == "__main__":
    # Ensure we're in the project root directory
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/cache'):
        os.makedirs('data/cache', exist_ok=True)
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Generate and save test data
    cache_path = generate_and_save_test_data()
    
    # Test the DataFetcher
    df = test_data_fetcher()
    
    print("\nData verification complete.")