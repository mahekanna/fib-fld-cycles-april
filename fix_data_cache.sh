#!/bin/bash

# Emergency script to completely reset the data cache system with consistent dates
echo "EMERGENCY DATA CACHE RESET TOOL"
echo "==============================="
echo "This script will completely reset the data cache system to fix date and lookback issues"
echo "- Will set daily data to April 5, 2024"
echo "- Will set intraday data to March 25, 2024"

# Make sure we're in the project directory
cd "$(dirname "$0")"

# Stop any running processes
echo "Stopping any running dashboard processes..."
pkill -f main_dashboard.py || true

# Remove the entire market_data.db file
echo "Removing market_data.db database..."
rm -f data/cache/market_data.db
echo "✓ Database removed"

# Remove all CSV cache files
echo "Removing all CSV cache files..."
rm -f data/cache/*.csv
echo "✓ CSV files removed"

# Recreate the directory structure
echo "Recreating cache directory structure..."
mkdir -p data/cache
echo "✓ Directory structure recreated"

# Fix the mock data to have proper market pricing
echo "Creating fresh price formatted mock data..."
python -c '
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_mock_data(symbol, exchange, interval="daily", bars=1000):
    """Generate realistic price data based on typical price ranges for different markets"""
    
    # Set base price ranges based on exchange and symbol
    if exchange == "NSE":
        if symbol == "NIFTY":
            base_price = 22000
            volatility = 200
        elif symbol == "BANKNIFTY":
            base_price = 48000
            volatility = 400
        elif symbol in ["RELIANCE", "TCS", "INFY"]:
            base_price = 2500
            volatility = 50
        else:
            base_price = 1000
            volatility = 30
    elif exchange == "NYSE":
        if symbol == "AAPL":
            base_price = 170
            volatility = 3
        elif symbol == "MSFT":
            base_price = 400
            volatility = 5
        else:
            base_price = 100
            volatility = 2
    else:
        base_price = 100
        volatility = 2
    
    # Generate dates with consistent dates
    # Use April 5, 2024 for daily data and March 25, 2024 for intraday
    if interval == "daily":
        end_date = datetime(2024, 4, 5)
        start_date = end_date - timedelta(days=bars)
        freq = "B"  # Business days
    elif interval == "4h":
        end_date = datetime(2024, 3, 25, 16, 0)  # March 25, 2024 4:00 PM
        start_date = end_date - timedelta(days=bars//6)
        freq = "4H"
    elif interval == "1h":
        end_date = datetime(2024, 3, 25, 16, 0)  # March 25, 2024 4:00 PM
        start_date = end_date - timedelta(days=bars//24)
        freq = "1H"
    elif interval == "15m":
        end_date = datetime(2024, 3, 25, 16, 0)  # March 25, 2024 4:00 PM
        start_date = end_date - timedelta(days=bars//96)
        freq = "15min"
    else:
        end_date = datetime(2024, 4, 5)
        start_date = end_date - timedelta(days=bars)
        freq = "B"
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)[:bars]
    
    # Generate price data
    np.random.seed(sum(ord(c) for c in symbol))  # Consistent seed for same symbol
    
    # Base trend
    x = np.linspace(0, 1, len(dates))
    trend = base_price * (1 + 0.1 * (np.random.random() - 0.5) * x)
    
    # Add cycles
    cycles = [21, 55, 89]
    cycle_strengths = [0.02, 0.04, 0.06]
    
    for cycle, strength in zip(cycles, cycle_strengths):
        phase = np.random.random() * 2 * np.pi
        cycle_wave = base_price * strength * np.sin(2 * np.pi * np.arange(len(dates)) / cycle + phase)
        trend += cycle_wave
    
    # Add random walk component
    random_walk = np.cumsum(np.random.normal(0, volatility/10, len(dates)))
    price = trend + random_walk
    
    # Ensure price is positive
    price = np.maximum(price, base_price * 0.5)
    
    # Generate OHLC
    daily_volatility = volatility * 0.01
    
    open_price = price * (1 + np.random.normal(0, daily_volatility, len(dates)))
    high_price = np.maximum(price, open_price) * (1 + np.abs(np.random.normal(0, daily_volatility, len(dates))))
    low_price = np.minimum(price, open_price) * (1 - np.abs(np.random.normal(0, daily_volatility, len(dates))))
    close_price = price
    
    # Generate volume
    avg_volume = 1000000 if exchange == "NSE" else 5000000
    volume = np.random.normal(avg_volume, avg_volume * 0.3, len(dates))
    volume = np.abs(volume).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume
    }, index=dates)
    
    # Add derived columns
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
    df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["hl2"] = (df["high"] + df["low"]) / 2
    df["price"] = df["close"]  # Add price column directly
    
    # Round to 2 decimal places for realism
    for col in ["open", "high", "low", "close", "hlc3", "ohlc4", "hl2", "price"]:
        df[col] = df[col].round(2)
    
    return df

# Generate and save mock data
symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
exchanges = ["NSE"]
intervals = ["daily", "4h", "1h", "15m"]

# Add NYSE symbols
symbols.extend(["AAPL", "MSFT"])
exchanges.extend(["NYSE", "NYSE"])

cache_dir = "data/cache"
os.makedirs(cache_dir, exist_ok=True)

for symbol, exchange in zip(symbols, exchanges):
    for interval in intervals:
        # Skip non-daily for NYSE for simplicity
        if exchange == "NYSE" and interval != "daily":
            continue
            
        print(f"Generating {exchange}_{symbol}_{interval} data...")
        df = generate_realistic_mock_data(symbol, exchange, interval)
        
        # Save to CSV
        filename = f"{exchange}_{symbol}_{interval}.csv"
        filepath = os.path.join(cache_dir, filename)
        df.to_csv(filepath)
        print(f"Saved {len(df)} bars to {filepath}")
'

echo "✓ Fresh mock data created with proper price formatting"

# Create a flag file to indicate we've done a full reset
touch data/cache/.full_reset_completed

echo ""
echo "Data cache reset complete! You can now run the dashboard with:"
echo "./run_dashboard.sh"
echo ""
echo "Important: The first scan after this reset will fetch fresh data"
echo "and correctly respect the lookback parameter."

# Make script executable
chmod +x run_dashboard.sh