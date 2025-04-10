import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_mock_price_data(
    symbol: str, 
    lookback: int = 1000, 
    interval: str = "daily",
    include_cycles: bool = True
) -> pd.DataFrame:
    """
    Generate mock price data with embedded cycles for testing.
    
    Args:
        symbol: Symbol name
        lookback: Number of bars to generate
        interval: Time interval (daily, 4h, etc.)
        include_cycles: Whether to include embedded cycles
        
    Returns:
        DataFrame with mock price data
    """
    # Safety check on lookback parameter
    lookback = max(min(lookback, 2000), 100)  # Ensure lookback is between 100 and 2000
    # Determine date range based on interval
    # Fix the dates explicitly to April 5, 2024 for daily data, matching real market data
    if interval == "daily":
        end_date = datetime(2024, 4, 5)  # Fixed to April 5, 2024 for daily data
        start_date = end_date - timedelta(days=lookback)
        freq = "D"
    elif interval == "4h":
        # Fix to March 25, 2024 for intraday data, matching real market data
        end_date = datetime(2024, 3, 25, 16, 0)  # March 25, 2024 4:00 PM
        start_date = end_date - timedelta(hours=4 * lookback)
        freq = "4H"
    elif interval == "1h":
        # Fix to March 25, 2024 for intraday data
        end_date = datetime(2024, 3, 25, 16, 0)  # March 25, 2024 4:00 PM
        start_date = end_date - timedelta(hours=lookback)
        freq = "H"
    elif interval == "15min":
        # Fix to March 25, 2024 for intraday data
        end_date = datetime(2024, 3, 25, 16, 0)  # March 25, 2024 4:00 PM
        start_date = end_date - timedelta(minutes=15 * lookback)
        freq = "15min"
    # For backward compatibility, also handle "15m" format
    elif interval == "15m":
        # Fix to March 25, 2024 for intraday data
        end_date = datetime(2024, 3, 25, 16, 0)  # March 25, 2024 4:00 PM
        start_date = end_date - timedelta(minutes=15 * lookback)
        freq = "15min"
    else:
        # Default to daily with fixed date
        end_date = datetime(2024, 4, 5)  # Default to April 5, 2024
        start_date = end_date - timedelta(days=lookback)
        freq = "D"
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Base price - start at different levels based on symbol
    symbol_hash = sum(ord(c) for c in symbol)
    base_price = 100 + (symbol_hash % 900)  # Price between 100-1000
    
    # Trend component - either up, down, or sideways
    trend_type = symbol_hash % 3  # 0=up, 1=down, 2=sideways
    
    if trend_type == 0:  # Uptrend
        trend = np.linspace(0, 20, len(dates))
    elif trend_type == 1:  # Downtrend
        trend = np.linspace(0, -15, len(dates))
    else:  # Sideways
        trend = np.zeros(len(dates))
    
    # Add noise
    noise = np.random.normal(0, 1, len(dates))
    
    # Initialize price array
    close_prices = base_price + trend + noise
    
    # Add cycles if requested
    if include_cycles:
        # Add three cycles of different lengths
        cycle1 = 21  # Short cycle
        cycle2 = 55  # Medium cycle
        cycle3 = 144  # Long cycle
        
        # Generate cycle waves
        t = np.arange(len(dates))
        cycle1_wave = 5 * np.sin(2 * np.pi * t / cycle1)
        cycle2_wave = 10 * np.sin(2 * np.pi * t / cycle2 + 0.5)
        cycle3_wave = 20 * np.sin(2 * np.pi * t / cycle3 + 1.0)
        
        # Add cycles to price
        close_prices += cycle1_wave + cycle2_wave + cycle3_wave
    
    # Generate open, high, low from close
    open_prices = close_prices + np.random.normal(0, 0.5, len(dates))
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 1.0, len(dates)))
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 1.0, len(dates)))
    
    # Generate volume
    volume = np.random.randint(1000, 10000, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    # Add derived columns
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['hl2'] = (df['high'] + df['low']) / 2
    
    return df


def save_mock_data_to_cache(symbol: str, exchange: str, interval: str, data: pd.DataFrame):
    """
    Save mock data to the cache directory.
    
    Args:
        symbol: Symbol name
        exchange: Exchange code
        interval: Time interval
        data: DataFrame with price data
    """
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create filename
    cache_filename = f"{exchange}_{symbol}_{interval}.csv"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Save to cache
    data.to_csv(cache_path)
    
    print(f"Mock data saved to {cache_path}")


def clear_and_regenerate_cache():
    """
    Clear all cache files and regenerate mock data with consistent dates.
    This function removes all pickle cache files and regenerates fresh mock data.
    """
    import glob
    import os
    
    # Get the cache directory
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
    
    # Remove all pickle files
    pkl_files = glob.glob(os.path.join(cache_dir, '*.pkl'))
    for file in pkl_files:
        try:
            os.remove(file)
            print(f"Removed {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    
    # Remove all CSV files
    csv_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    for file in csv_files:
        try:
            os.remove(file)
            print(f"Removed {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    
    # Generate fresh mock data
    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
    exchange = "NSE"
    intervals = ["daily", "4h", "1h", "15min"]
    
    for symbol in symbols:
        for interval in intervals:
            data = generate_mock_price_data(symbol, lookback=1000, interval=interval)
            save_mock_data_to_cache(symbol, exchange, interval, data)
    
    print("Cache cleared and regenerated with consistent dates")


if __name__ == "__main__":
    # Generate some example mock data
    clear_and_regenerate_cache()