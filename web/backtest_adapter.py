"""
Backtest Adapter Module - Connects the UI with the Backtesting Framework.

This adapter provides a clean interface between the web UI and the backtesting framework,
ensuring proper translation between UI parameters and backtest engine parameters,
and formatting results for display in the dashboard.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Configure logging
try:
    from utils.logging_utils import get_component_logger
    logger = get_component_logger("web.backtest_adapter")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("web.backtest_adapter")

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import backtesting classes
try:
    from backtesting.backtesting_framework import BacktestEngine, BacktestParameters, BacktestTrade
except ImportError as e:
    logger.error(f"Error importing backtesting framework: {e}")
    raise ImportError("Could not import backtesting framework. Make sure it's installed correctly.")

# Import config utilities
try:
    from utils.config import load_config
except ImportError:
    # Simple fallback
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

# Import data utilities
try:
    from data.data_management import DataFetcher
except ImportError as e:
    logger.error(f"Error importing DataFetcher: {e}")
    DataFetcher = None

# Import scanner
try:
    from core.scanner_system import FibCycleScanner
except ImportError as e:
    logger.error(f"Error importing FibCycleScanner: {e}")
    FibCycleScanner = None


class BacktestAdapter:
    """
    Adapter class that connects the web dashboard to the backtesting engine.
    
    This class handles:
    1. Translation between UI parameters and backtest engine parameters
    2. Execution of backtests through the engine
    3. Formatting results for display in the UI
    """
    
    def __init__(self, config=None):
        """
        Initialize the backtest adapter.
        
        Args:
            config: Optional configuration dictionary
        """
        # Load config if not provided
        if config is None:
            config_path = os.path.join(project_root, "config", "config.json")
            self.config = load_config(config_path)
        else:
            self.config = config
            
        # Set up components - created on first use to avoid circular imports and overhead
        self._backtest_engine = None
        self._data_fetcher = None
        self._scanner = None
        
    @property
    def backtest_engine(self):
        """Lazy initialization of backtest engine."""
        if self._backtest_engine is None:
            logger.info("Initializing backtest engine")
            self._backtest_engine = BacktestEngine(
                config=self.config,
                scanner=self.scanner,
                data_fetcher=self.data_fetcher
            )
        return self._backtest_engine
    
    @property
    def data_fetcher(self):
        """Lazy initialization of data fetcher."""
        if self._data_fetcher is None and DataFetcher is not None:
            logger.info("Initializing data fetcher")
            self._data_fetcher = DataFetcher(self.config)
        return self._data_fetcher
    
    @property
    def scanner(self):
        """Lazy initialization of scanner."""
        if self._scanner is None and FibCycleScanner is not None:
            logger.info("Initializing scanner")
            self._scanner = FibCycleScanner(self.config)
        return self._scanner
        
    def run_backtest(self, ui_params: Dict) -> Dict:
        """
        Run a backtest using parameters from the UI.
        
        Args:
            ui_params: Dictionary of UI parameters
            
        Returns:
            Dictionary of backtest results formatted for the UI
        """
        try:
            # Convert UI parameters to BacktestParameters
            backtest_params = self._parse_ui_parameters(ui_params)
            
            # Run the backtest
            logger.info(f"Running backtest for {backtest_params.symbol} ({backtest_params.interval}) "
                       f"from {backtest_params.start_date} to {backtest_params.end_date}")
            
            results = self.backtest_engine.run_backtest(backtest_params)
            
            # Check for success
            if not results.get('success', False):
                logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
                return {
                    'error': results.get('error', 'Backtest failed')
                }
            
            # Format results for the UI
            ui_results = self._format_results_for_ui(results)
            
            return ui_results
            
        except Exception as e:
            logger.error(f"Error in run_backtest: {e}", exc_info=True)
            return {
                'error': str(e)
            }
    
    def _parse_ui_parameters(self, ui_params: Dict) -> BacktestParameters:
        """
        Convert UI parameters to BacktestParameters.
        
        Args:
            ui_params: Dictionary of UI parameters
            
        Returns:
            BacktestParameters instance
        """
        # Extract required parameters
        symbol = ui_params.get('symbol', '')
        exchange = ui_params.get('exchange', 'NSE')
        interval = ui_params.get('timeframe', 'daily')
        
        # Parse dates
        try:
            start_date = ui_params.get('start_date')
            end_date = ui_params.get('end_date')
            
            if start_date:
                start_date = pd.Timestamp(start_date)
            if end_date:
                end_date = pd.Timestamp(end_date)
                
        except Exception as date_err:
            logger.warning(f"Error parsing dates: {date_err}. Using defaults.")
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
        
        # Extract numeric parameters with defaults
        initial_capital = float(ui_params.get('initial_capital', 100000))
        position_size = float(ui_params.get('position_size', 10))
        signal_threshold = float(ui_params.get('signal_threshold', 0.2))
        
        # Extract boolean parameters
        require_alignment = bool(ui_params.get('require_alignment', True))
        trailing_stop = bool(ui_params.get('trailing_stop', False))
        
        # Extract optional parameters
        strategy_type = ui_params.get('strategy', 'fib_cycle')
        trailing_stop_pct = float(ui_params.get('trailing_stop_pct', 5.0))
        take_profit_multiplier = float(ui_params.get('tp_multiplier', 2.0))
        max_positions = int(ui_params.get('max_positions', 5))
        pyramiding = int(ui_params.get('pyramiding', 0))
        
        # Calculate appropriate lookback
        if start_date and end_date:
            date_span = (end_date - start_date).days
            lookback = min(max(date_span + 10, 365), 1000)
        else:
            lookback = 365
            
        logger.info(f"Using lookback of {lookback} days for {symbol}")
        
        # Create and return parameters
        return BacktestParameters(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            lookback=lookback,
            initial_capital=initial_capital,
            position_size_pct=position_size,
            min_strength=signal_threshold,
            require_alignment=require_alignment,
            max_open_positions=max_positions,
            pyramiding=pyramiding,
            trailing_stop=trailing_stop,
            trailing_stop_pct=trailing_stop_pct,
            take_profit_multiplier=take_profit_multiplier,
            strategy_type=strategy_type
        )
    
    def _format_results_for_ui(self, results: Dict) -> Dict:
        """
        Format backtest results for the UI.
        
        Args:
            results: Raw backtest results from the engine
            
        Returns:
            Formatted results for the UI
        """
        # If there's an error, just return that
        if 'error' in results:
            return {'error': results['error']}
            
        # Extract and reformat data
        symbol = results.get('symbol', '')
        exchange = results.get('exchange', '')
        interval = results.get('interval', '')
        
        # Convert dates if needed
        start_date = results.get('start_date')
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
            
        end_date = results.get('end_date')
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
            
        # Extract financial data
        initial_capital = results.get('initial_capital', 0)
        final_capital = results.get('final_capital', 0)
        
        # Extract metrics
        metrics = results.get('metrics', {})
        
        # Process equity curve (ensure dates are serializable)
        equity_curve = results.get('equity_curve', [])
        for point in equity_curve:
            date = point.get('date')
            if hasattr(date, 'isoformat'):
                point['date'] = date.isoformat()
        
        # Process trades (ensure dates are serializable and all required fields are present)
        trades = results.get('trades', [])
        for trade in trades:
            # Format dates
            for date_field in ['entry_date', 'exit_date']:
                date = trade.get(date_field)
                if hasattr(date, 'isoformat'):
                    trade[date_field] = date.isoformat()
                    
            # Ensure required fields with defaults
            required_fields = {
                'symbol': symbol,
                'direction': 'unknown',
                'entry_price': 0,
                'exit_price': 0,
                'quantity': 0,
                'profit_loss': 0,
                'profit_loss_pct': 0,
                'exit_reason': 'unknown'
            }
            
            for field, default in required_fields.items():
                if field not in trade:
                    trade[field] = default
        
        # Build the final result dictionary
        formatted_results = {
            'symbol': symbol,
            'exchange': exchange,
            'interval': interval,
            'start_date': start_date.isoformat() if hasattr(start_date, 'isoformat') else start_date,
            'end_date': end_date.isoformat() if hasattr(end_date, 'isoformat') else end_date,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'duration': results.get('duration_days', 0),
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'execution_time': results.get('execution_time', 0)
        }
        
        return formatted_results
        
        
# Create a singleton instance for importing
adapter = BacktestAdapter()

# Helper function for direct use from UI
def run_backtest_from_ui(params: Dict) -> Dict:
    """
    Helper function to run a backtest from the UI.
    
    Args:
        params: UI parameter dictionary
        
    Returns:
        Formatted backtest results
    """
    return adapter.run_backtest(params)