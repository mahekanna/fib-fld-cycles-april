#!/usr/bin/env python
"""
Test script for the backtesting system.
This script verifies that the backtesting framework can correctly load and process data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
logger.info(f"Project root: {project_root}")

# Dynamically import modules using absolute paths to avoid import issues
def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not find module {module_name} at {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error importing {module_name} from {file_path}: {str(e)}")
        raise

# Import config utils
try:
    config_module = import_module_from_path("config", os.path.join(project_root, "utils", "config.py"))
    load_config = config_module.load_config
    logger.info("Successfully imported config module")
except Exception as e:
    logger.error(f"Failed to import config: {str(e)}")
    sys.exit(1)

# Import backtesting framework
try:
    backtesting_module = import_module_from_path(
        "backtesting_framework", 
        os.path.join(project_root, "backtesting", "backtesting_framework.py")
    )
    BacktestEngine = backtesting_module.BacktestEngine
    BacktestParameters = backtesting_module.BacktestParameters
    logger.info("Successfully imported backtesting framework")
except Exception as e:
    logger.error(f"Failed to import backtesting framework: {str(e)}")
    sys.exit(1)

# Import data management
try:
    data_module = import_module_from_path(
        "data_management", 
        os.path.join(project_root, "data", "data_management.py")
    )
    DataFetcher = data_module.DataFetcher
    DataProcessor = data_module.DataProcessor
    logger.info("Successfully imported data management")
except Exception as e:
    logger.error(f"Failed to import data management: {str(e)}")
    sys.exit(1)

def run_simple_backtest():
    """Run a simple backtest to verify the system is working."""
    logger.info("Loading configuration...")
    config_path = os.path.join(project_root, "config", "config.json")
    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Create data fetcher and engine
    logger.info("Creating data fetcher and backtest engine...")
    data_fetcher = DataFetcher(config)
    engine = BacktestEngine(config, data_fetcher=data_fetcher)
    
    # Create backtest parameters
    symbol = "AAPL"  # Change to a symbol with data available
    exchange = "NYSE"
    interval = "daily"
    lookback = 500
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    logger.info(f"Running backtest for {symbol} on {interval} timeframe...")
    params = BacktestParameters(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        lookback=lookback,
        initial_capital=100000.0,
        position_size_pct=10.0,
        min_strength=0.3,
        take_profit_multiplier=2.0,
        trailing_stop=True,
        trailing_stop_pct=5.0
    )
    
    try:
        # Run the backtest
        results = engine.run_backtest(params)
        
        # Print results summary
        metrics = results.get('metrics', {})
        logger.info("===== Backtest Results =====")
        logger.info(f"Symbol: {results.get('symbol')}")
        logger.info(f"Period: {results.get('start_date')} to {results.get('end_date')}")
        logger.info(f"Initial Capital: ${results.get('initial_capital'):.2f}")
        logger.info(f"Final Capital: ${results.get('final_capital'):.2f}")
        logger.info(f"Total Return: {metrics.get('profit_loss_pct'):.2f}%")
        logger.info(f"Total Trades: {metrics.get('total_trades')}")
        logger.info(f"Win Rate: {metrics.get('win_rate')*100:.2f}%")
        logger.info(f"Profit Factor: {metrics.get('profit_factor'):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown_pct'):.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio'):.2f}")
        
        # Plot equity curve
        try:
            fig = engine.plot_equity_curve(results)
            plt.show()
        except Exception as plot_err:
            logger.error(f"Error plotting equity curve: {str(plot_err)}")
        
        # Return success
        return True, results
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        return False, str(e)

def test_data_fetcher():
    """Test the data fetcher to ensure it can retrieve data."""
    logger.info("Testing data fetcher...")
    config_path = os.path.join(project_root, "config", "config.json")
    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)
    fetcher = DataFetcher(config)
    
    # Try to fetch data for a few common symbols
    symbols = ["AAPL", "MSFT", "NIFTY", "RELIANCE"]
    exchanges = ["NYSE", "NYSE", "NSE", "NSE"]
    intervals = ["daily", "daily", "daily", "daily"]
    
    for symbol, exchange, interval in zip(symbols, exchanges, intervals):
        logger.info(f"Attempting to fetch {symbol} on {exchange} with {interval} interval...")
        try:
            data = fetcher.get_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                lookback=500,
                force_download=False
            )
            
            if data is not None and not data.empty:
                logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
                logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
                logger.info(f"Columns: {data.columns.tolist()}")
                
                # Add a simple price chart
                try:
                    plt.figure(figsize=(10, 6))
                    plt.plot(data.index, data['close'])
                    plt.title(f"{symbol} Price Chart")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.tight_layout()
                    plt.savefig(f"{symbol}_{interval}_test.png")
                    logger.info(f"Saved price chart to {symbol}_{interval}_test.png")
                except Exception as plot_err:
                    logger.error(f"Error plotting price chart: {str(plot_err)}")
            else:
                logger.warning(f"No data returned for {symbol}")
        
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")

if __name__ == "__main__":
    # Test data fetcher first
    logger.info("=== Testing Data Fetcher ===")
    test_data_fetcher()
    
    # Then run a backtest
    logger.info("=== Running Simple Backtest ===")
    success, results = run_simple_backtest()
    
    if success:
        logger.info("Backtest completed successfully!")
    else:
        logger.error(f"Backtest failed: {results}")