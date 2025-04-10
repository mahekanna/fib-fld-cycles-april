#!/usr/bin/env python3
"""
Example Script for Running Fibonacci Cycles Trading Strategies

This script demonstrates how to use the advanced trading strategies with 
the Fibonacci Cycles System. It loads data, detects cycles, generates signals,
and optionally runs a backtest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the strategy components
from strategies.strategy_factory import get_strategy, get_available_strategies
from strategies.backtest_engine import run_strategy_backtest

# Mock cycle detector for example purposes
class MockCycleDetector:
    """Simple mock cycle detector for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock detector."""
        self.config = config
    
    def detect_cycles(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect cycles in the data.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with detected cycles
        """
        # Generate Fibonacci cycle lengths
        cycles = [21, 34, 55, 89]
        
        # Generate mock cycle states
        cycle_states = []
        for cycle in cycles:
            # Determine if bullish based on simple moving average comparison
            if len(data) > cycle:
                sma_short = data['close'].rolling(window=cycle//2).mean().iloc[-1]
                sma_long = data['close'].rolling(window=cycle).mean().iloc[-1]
                is_bullish = sma_short > sma_long
            else:
                is_bullish = True
            
            # Generate random position in cycle (days since crossover)
            import random
            days_since = random.randint(1, cycle//2)
            
            cycle_states.append({
                'cycle_length': cycle,
                'is_bullish': is_bullish,
                'days_since_crossover': days_since
            })
        
        return {
            'cycles': cycles,
            'cycle_states': cycle_states
        }


def load_sample_data(symbol: str = "SAMPLE", rows: int = 500) -> pd.DataFrame:
    """
    Load or generate sample data for demonstration.
    
    Args:
        symbol: Symbol to use
        rows: Number of data points to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    # Try to load from file
    data_path = f"data/{symbol.lower()}_daily.csv"
    
    if os.path.exists(data_path):
        try:
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            return data
        except Exception as e:
            logger.warning(f"Error loading data: {e}")
    
    # Generate synthetic data
    logger.info("Generating synthetic price data")
    
    # Generate date range
    end_date = datetime.now()
    date_range = pd.date_range(end=end_date, periods=rows, freq='D')
    
    # Generate price series with some cyclicality
    close = 100.0
    prices = []
    for i in range(rows):
        # Random walk with cycles
        cycle1 = 2 * np.sin(i/21)  # 21-day cycle
        cycle2 = 1.5 * np.sin(i/34)  # 34-day cycle
        cycle3 = 3 * np.sin(i/55)  # 55-day cycle
        
        # Combine cycles with random walk
        change = 0.005 * (cycle1 + cycle2 + cycle3) + np.random.normal(0, 0.007)
        close *= (1 + change)
        
        # Generate OHLC based on close
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = close * (1 + np.random.normal(0, 0.003))
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 100000)
        })
    
    # Create DataFrame
    data = pd.DataFrame(prices, index=date_range)
    
    # Save for future use
    os.makedirs('data', exist_ok=True)
    data.to_csv(data_path)
    
    return data


def run_example(strategy_name: str, symbol: str = "SAMPLE", backtest: bool = True) -> None:
    """
    Run a complete example with a specified strategy.
    
    Args:
        strategy_name: Name of strategy to use
        symbol: Symbol to analyze
        backtest: Whether to run a backtest
    """
    logger.info(f"Running example with {strategy_name} strategy on {symbol}")
    
    # Load or generate data
    data = load_sample_data(symbol)
    logger.info(f"Loaded {len(data)} bars of data for {symbol}")
    
    # Load default configuration
    config = {
        'risk_per_trade': 1.0,
        'max_positions': 5,
        'use_trailing_stop': True,
        'initial_capital': 100000,
        'commission_pct': 0.1,
        'slippage_pct': 0.05,
        'min_alignment_threshold': 0.6,
        'log_level': 'INFO'
    }
    
    # Create cycle detector
    cycle_detector = MockCycleDetector(config)
    
    # Get strategy
    strategy = get_strategy(strategy_name, config)
    
    if not strategy:
        logger.error(f"Strategy '{strategy_name}' not found")
        return
    
    logger.info(f"Using strategy: {strategy.name}")
    
    # Detect cycles for most recent data point
    current_data = data.copy()
    cycle_result = cycle_detector.detect_cycles(current_data)
    
    cycles = cycle_result.get('cycles', [])
    cycle_states = cycle_result.get('cycle_states', [])
    
    if not cycles:
        logger.error("No cycles detected")
        return
    
    logger.info(f"Detected cycles: {cycles}")
    
    # Generate trading signal
    fld_crossovers = []
    for cycle in cycles:
        crossovers = strategy.detect_fld_crossovers(current_data, cycle)
        fld_crossovers.extend(crossovers)
    
    signal = strategy.generate_signal(current_data, cycles, fld_crossovers, cycle_states)
    
    # Display signal
    logger.info(f"Generated signal: {signal.get('signal', 'neutral')}")
    logger.info(f"Signal strength: {signal.get('strength', 0)}")
    logger.info(f"Confidence: {signal.get('confidence', 'low')}")
    logger.info(f"Description: {signal.get('description', '')}")
    
    # Calculate stops and targets if valid signal
    if signal.get('signal') in ['buy', 'sell']:
        direction = 'long' if signal.get('signal') == 'buy' else 'short'
        current_price = current_data['close'].iloc[-1]
        
        stop_price = strategy.set_stop_loss(current_data, signal, current_price, direction)
        target_price = strategy.set_take_profit(current_data, signal, current_price, stop_price, direction)
        
        logger.info(f"Current price: {current_price:.2f}")
        logger.info(f"Stop loss: {stop_price:.2f}")
        logger.info(f"Take profit: {target_price:.2f}")
        
        # Calculate position size
        position_size = strategy.calculate_position_size(
            config['initial_capital'], signal, current_price, stop_price
        )
        
        logger.info(f"Position size: {position_size:.2f} units")
    
    # Run backtest if requested
    if backtest:
        logger.info("Running backtest...")
        
        result = run_strategy_backtest(
            data=current_data,
            strategy_class=type(strategy),
            cycle_detector=cycle_detector,
            config=config,
            symbol=symbol,
            timeframe="daily"
        )
        
        # Display backtest results
        logger.info("Backtest completed")
        logger.info(f"Initial capital: ${result.initial_capital:.2f}")
        logger.info(f"Final capital: ${result.final_capital:.2f}")
        logger.info(f"Total return: {result.total_return_pct:.2f}%")
        logger.info(f"Max drawdown: {result.max_drawdown_pct:.2f}%")
        logger.info(f"Win rate: {result.win_rate*100:.1f}%")
        logger.info(f"Total trades: {len(result.trades)}")
        
        # Plot results
        plot_file = f"{symbol}_{strategy_name}_backtest.png"
        result.plot_results(save_path=plot_file)
        logger.info(f"Backtest chart saved to {plot_file}")
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Fibonacci Cycles Trading Strategy example')
    
    # Get available strategies
    available_strategies = get_available_strategies()
    
    parser.add_argument('--strategy', type=str, choices=available_strategies, 
                      default='rapid_cycle_fld', help='Strategy to use')
    parser.add_argument('--symbol', type=str, default='SAMPLE', help='Symbol to analyze')
    parser.add_argument('--no-backtest', dest='backtest', action='store_false', 
                      help='Skip running backtest')
    
    args = parser.parse_args()
    
    run_example(args.strategy, args.symbol, args.backtest)