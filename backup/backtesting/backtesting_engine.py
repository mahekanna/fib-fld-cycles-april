"""
Advanced backtesting engine for the Fibonacci Cycle Trading System.

This module provides a comprehensive backtesting framework for evaluating
trading strategies based on Fibonacci cycles and FLD crossovers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import json
import os
import sys
import logging
import time
import uuid

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import centralized logging
try:
    from utils.logging_utils import get_component_logger
    logger = get_component_logger("backtesting.engine")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import core components with robust error handling
try:
    from core.scanner import FibCycleScanner
    from core.cycle_detection import CycleDetector
    from core.fld_signal_generator import FLDCalculator, SignalGenerator
    from models.scan_parameters import ScanParameters
    from models.scan_result import ScanResult
    from data.data_management import DataFetcher
    from trading.trading_strategies import TradingStrategy
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    # We'll continue and try to handle imports dynamically when needed


@dataclass
class BacktestTrade:
    """Represents a completed trade in a backtest."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    exchange: str = ""
    interval: str = ""
    direction: str = ""  # 'long' or 'short'
    entry_date: Optional[datetime] = None
    entry_price: float = 0.0
    exit_date: Optional[datetime] = None
    exit_price: float = 0.0
    quantity: float = 0.0
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0
    exit_reason: str = ""
    trade_duration: int = 0  # Duration in candles or days
    entry_signal: Dict[str, Any] = field(default_factory=dict)
    exit_signal: Dict[str, Any] = field(default_factory=dict)
    cycle_state: Dict[str, Any] = field(default_factory=dict)
    risk_reward_planned: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    max_favorable_excursion: float = 0.0  # Maximum favorable price movement
    max_adverse_excursion: float = 0.0  # Maximum adverse price movement
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, (dict, list)):
                result[key] = value  # Already JSON-serializable
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestTrade':
        """Create a trade from a dictionary."""
        # Handle datetime conversion
        for date_field in ['entry_date', 'exit_date']:
            if date_field in data and data[date_field] and isinstance(data[date_field], str):
                try:
                    data[date_field] = datetime.fromisoformat(data[date_field].replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Could not parse {date_field} from {data[date_field]}")
        
        # Create instance
        return cls(**data)


@dataclass
class BacktestParameters:
    """Parameters for configuring a backtest."""
    symbol: str
    exchange: str = "NSE"
    interval: str = "daily"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100000.0
    position_size_pct: float = 10.0
    strategy_type: str = "fib_cycle"
    cycles_to_use: List[int] = field(default_factory=lambda: [21, 34, 55])
    cycle_influence: Dict[int, float] = field(default_factory=dict)
    min_strength: float = 0.3
    stop_loss_atr_multiplier: float = 1.5
    take_profit_multiplier: float = 2.0
    trailing_stop: bool = False
    trailing_stop_pct: float = 5.0
    enable_pyramid: bool = False
    max_pyramid_positions: int = 3
    allow_short: bool = True
    use_leverage: bool = False
    max_leverage: float = 1.0
    fld_margin: float = 0.02
    filter_by_market_regime: bool = False
    max_open_positions: int = 5
    trade_entry_delay: int = 0  # Bars to wait before entering a trade
    entry_order_type: str = "market"  # market, limit, stop
    exit_order_type: str = "market"  # market, limit, stop, trailing_stop
    risk_per_trade_pct: float = 1.0
    enable_risk_management: bool = True
    min_risk_reward_ratio: float = 1.5
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, (dict, list)):
                result[key] = value  # Already JSON-serializable
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestParameters':
        """Create parameters from a dictionary."""
        # Handle datetime conversion
        for date_field in ['start_date', 'end_date']:
            if date_field in data and data[date_field] and isinstance(data[date_field], str):
                try:
                    data[date_field] = datetime.fromisoformat(data[date_field].replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Could not parse {date_field} from {data[date_field]}")
        
        # Create instance
        return cls(**data)


@dataclass
class BacktestResults:
    """Contains the results of a backtest run."""
    parameters: BacktestParameters
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    drawdowns: List[Dict[str, Any]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    annual_returns: Dict[str, float] = field(default_factory=dict)
    trade_statistics: Dict[str, Any] = field(default_factory=dict)
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    mar_ratio: float = 0.0
    calmar_ratio: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    net_profit: float = 0.0
    net_profit_pct: float = 0.0
    rate_of_return: float = 0.0
    annualized_return: float = 0.0
    test_duration_days: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        result = {
            'parameters': self.parameters.to_dict(),
            'trades': [trade.to_dict() for trade in self.trades],
            'equity_curve': self.equity_curve,
            'metrics': self.metrics,
            'drawdowns': self.drawdowns,
            'monthly_returns': self.monthly_returns,
            'annual_returns': self.annual_returns,
            'trade_statistics': self.trade_statistics,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'mar_ratio': self.mar_ratio,
            'calmar_ratio': self.calmar_ratio,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_win_pct': self.avg_win_pct,
            'avg_loss_pct': self.avg_loss_pct,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'net_profit': self.net_profit,
            'net_profit_pct': self.net_profit_pct,
            'rate_of_return': self.rate_of_return,
            'annualized_return': self.annualized_return,
            'test_duration_days': self.test_duration_days,
            'custom_metrics': self.custom_metrics,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestResults':
        """Create results from a dictionary."""
        # Handle nested objects
        params = BacktestParameters.from_dict(data.pop('parameters', {}))
        
        # Convert trade dicts to BacktestTrade objects
        trades = [BacktestTrade.from_dict(trade) for trade in data.pop('trades', [])]
        
        # Create instance with basic data
        results = cls(parameters=params)
        
        # Update with remaining data from dict
        for key, value in data.items():
            if hasattr(results, key):
                setattr(results, key, value)
                
        # Assign trades after creation to avoid default factory
        results.trades = trades
        
        return results


class BacktestEngine:
    """
    Advanced backtesting engine for trading strategies.
    
    This engine simulates trading with realistic conditions including:
    - Position sizing
    - Risk management
    - Multiple entry/exit strategies
    - Detailed metrics and reporting
    """
    
    def __init__(self, config: Dict[str, Any], data_fetcher: Optional[DataFetcher] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            config: System configuration
            data_fetcher: Optional DataFetcher instance
        """
        self.config = config
        self.data_fetcher = data_fetcher or self._initialize_data_fetcher()
        self.scanner = self._initialize_scanner()
        self.cycle_detector = self._initialize_cycle_detector()
        self.fld_calculator = self._initialize_fld_calculator()
        self.signal_generator = self._initialize_signal_generator()
        
        # Initialize state
        self.current_data = None
        self.current_symbol = None
        self.current_interval = None
        self.current_exchange = None
        
        # Initialize results
        self.results = None
        
    def _initialize_data_fetcher(self) -> DataFetcher:
        """Initialize the data fetcher."""
        try:
            return DataFetcher(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize data fetcher: {e}")
            raise RuntimeError(f"Failed to initialize data fetcher: {e}")
            
    def _initialize_scanner(self) -> FibCycleScanner:
        """Initialize the scanner."""
        try:
            return FibCycleScanner(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize scanner: {e}")
            raise RuntimeError(f"Failed to initialize scanner: {e}")
            
    def _initialize_cycle_detector(self) -> CycleDetector:
        """Initialize the cycle detector."""
        try:
            return CycleDetector(
                min_periods=self.config['analysis']['min_period'],
                max_periods=self.config['analysis']['max_period'],
                fibonacci_cycles=self.config['analysis']['fib_cycles'],
                power_threshold=self.config['analysis']['power_threshold'],
                cycle_tolerance=self.config['analysis']['cycle_tolerance']
            )
        except Exception as e:
            logger.error(f"Failed to initialize cycle detector: {e}")
            raise RuntimeError(f"Failed to initialize cycle detector: {e}")
            
    def _initialize_fld_calculator(self) -> FLDCalculator:
        """Initialize the FLD calculator."""
        try:
            return FLDCalculator()
        except Exception as e:
            logger.error(f"Failed to initialize FLD calculator: {e}")
            raise RuntimeError(f"Failed to initialize FLD calculator: {e}")
    
    def _initialize_signal_generator(self) -> SignalGenerator:
        """Initialize the signal generator."""
        try:
            return SignalGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize signal generator: {e}")
            raise RuntimeError(f"Failed to initialize signal generator: {e}")
    
    def _fetch_data(self, params: BacktestParameters) -> pd.DataFrame:
        """
        Fetch data for backtesting.
        
        Args:
            params: Backtest parameters
            
        Returns:
            DataFrame with price data
        """
        logger.info(f"Fetching data for {params.symbol} ({params.interval})")
        
        try:
            # Date handling with sensible defaults
            start_date = params.start_date
            end_date = params.end_date or datetime.now()
            
            # Calculate lookback based on parameters and add margin
            lookback = 1000  # Default
            
            # Fetch the data
            data = self.data_fetcher.get_data(
                symbol=params.symbol,
                exchange=params.exchange,
                interval=params.interval,
                lookback=lookback,
                use_cache=True
            )
            
            if data is None or data.empty:
                logger.error(f"Failed to fetch data for {params.symbol}")
                raise ValueError(f"No data available for {params.symbol} ({params.exchange}, {params.interval})")
                
            # Filter by date range if specified
            if start_date:
                data = data[data.index >= start_date]
                
            if end_date:
                data = data[data.index <= end_date]
                
            # Make sure we have enough data
            if len(data) < 100:
                logger.warning(f"Limited data available for {params.symbol}: {len(data)} bars")
                
            if data.empty:
                logger.error(f"No data available in specified date range for {params.symbol}")
                raise ValueError(f"No data available in date range {start_date} to {end_date}")
                
            logger.info(f"Fetched {len(data)} bars for {params.symbol} from {data.index[0]} to {data.index[-1]}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise RuntimeError(f"Failed to fetch data: {e}")
    
    def _detect_cycles(self, data: pd.DataFrame, params: BacktestParameters) -> Tuple[List[int], Dict[int, float]]:
        """
        Detect dominant cycles in the data.
        
        Args:
            data: Price data
            params: Backtest parameters
            
        Returns:
            Tuple of (detected cycles, cycle powers)
        """
        price_series = data['close']
        
        # Use specified cycles if provided, otherwise detect them
        if params.cycles_to_use:
            logger.info(f"Using specified cycles: {params.cycles_to_use}")
            return params.cycles_to_use, {cycle: 1.0 for cycle in params.cycles_to_use}
            
        try:
            # Detect cycles
            detected_cycles, cycle_powers = self.cycle_detector.detect_dominant_cycles(price_series)
            logger.info(f"Detected cycles: {detected_cycles} with powers: {cycle_powers}")
            return detected_cycles, cycle_powers
        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")
            # Fallback to default cycles
            default_cycles = [21, 34, 55]
            logger.warning(f"Using default cycles: {default_cycles}")
            return default_cycles, {cycle: 1.0 for cycle in default_cycles}
    
    def _calculate_flds(self, data: pd.DataFrame, cycles: List[int]) -> Dict[int, pd.Series]:
        """
        Calculate FLDs for each cycle.
        
        Args:
            data: Price data
            cycles: List of cycle lengths
            
        Returns:
            Dictionary mapping cycle lengths to FLD series
        """
        try:
            flds = {}
            price_series = data['close']
            
            for cycle in cycles:
                fld = self.fld_calculator.calculate_fld(price_series, cycle)
                flds[cycle] = fld
                
            return flds
        except Exception as e:
            logger.error(f"Error calculating FLDs: {e}")
            raise RuntimeError(f"Failed to calculate FLDs: {e}")
    
    def _generate_signals(self, data: pd.DataFrame, flds: Dict[int, pd.Series], 
                         cycles: List[int], params: BacktestParameters) -> pd.DataFrame:
        """
        Generate trading signals based on FLD crossovers.
        
        Args:
            data: Price data
            flds: Dictionary of FLDs by cycle length
            cycles: List of cycle lengths
            params: Backtest parameters
            
        Returns:
            DataFrame with signals added
        """
        try:
            signals = data.copy()
            signals['signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
            signals['strength'] = 0.0
            signals['confidence'] = 'low'
            
            # Process each cycle
            for cycle in cycles:
                if cycle not in flds:
                    continue
                    
                # Generate crossover signals for this cycle
                cycle_signals = self.signal_generator.generate_signals(
                    price=data['close'],
                    fld=flds[cycle],
                    fld_margin=params.fld_margin
                )
                
                # Combine signals (simple sum for now)
                weight = params.cycle_influence.get(cycle, 1.0)
                signals['signal'] += cycle_signals['signal'] * weight
                signals['strength'] += cycle_signals['strength'] * weight
                
            # Normalize strength
            if cycles:
                total_weight = sum(params.cycle_influence.get(c, 1.0) for c in cycles)
                if total_weight > 0:
                    signals['strength'] /= total_weight
                    
            # Apply strength threshold
            signals.loc[abs(signals['strength']) < params.min_strength, 'signal'] = 0
            
            # Set confidence levels
            signals.loc[abs(signals['strength']) > 0.7, 'confidence'] = 'high'
            signals.loc[(abs(signals['strength']) > 0.4) & (abs(signals['strength']) <= 0.7), 'confidence'] = 'medium'
            
            # Normalize signal to +1/0/-1
            signals['signal'] = np.sign(signals['signal'])
            
            # If no short selling allowed, convert sell signals to 0
            if not params.allow_short:
                signals.loc[signals['signal'] < 0, 'signal'] = 0
                
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise RuntimeError(f"Failed to generate signals: {e}")
    
    def _calculate_position_sizes(self, data: pd.DataFrame, params: BacktestParameters) -> pd.DataFrame:
        """
        Calculate position sizes for each potential trade.
        
        Args:
            data: Price data with signals
            params: Backtest parameters
            
        Returns:
            DataFrame with position sizes added
        """
        try:
            # Calculate ATR for position sizing if available
            if 'high' in data.columns and 'low' in data.columns:
                # Simple ATR calculation (14-period)
                tr1 = data['high'] - data['low']
                tr2 = abs(data['high'] - data['close'].shift(1))
                tr3 = abs(data['low'] - data['close'].shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                data['atr'] = tr.rolling(window=14).mean()
            else:
                # Simple volatility estimate if high/low not available
                data['atr'] = data['close'].rolling(window=14).std()
                
            # Position size as percentage of capital
            data['position_size_pct'] = params.position_size_pct
            
            # Risk-based position sizing
            data['risk_per_trade'] = params.initial_capital * (params.risk_per_trade_pct / 100)
            
            # Calculate stop loss distances
            data['stop_loss_distance'] = data['atr'] * params.stop_loss_atr_multiplier
            
            # Calculate take profit levels based on stop loss
            data['take_profit_distance'] = data['stop_loss_distance'] * params.take_profit_multiplier
            
            # Calculate actual position sizes
            data['max_position_size'] = (params.initial_capital * (params.position_size_pct / 100)) / data['close']
            
            # Risk-based position size
            data['risk_position_size'] = data['risk_per_trade'] / data['stop_loss_distance']
            
            # Use the smaller of the two sizes
            data['position_size'] = pd.DataFrame({
                'max': data['max_position_size'],
                'risk': data['risk_position_size']
            }).min(axis=1)
            
            return data
        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}")
            raise RuntimeError(f"Failed to calculate position sizes: {e}")
    
    def _execute_backtest(self, data: pd.DataFrame, params: BacktestParameters) -> BacktestResults:
        """
        Execute the backtest.
        
        Args:
            data: Price data with signals and position sizes
            params: Backtest parameters
            
        Returns:
            BacktestResults with trades and metrics
        """
        logger.info(f"Executing backtest for {params.symbol} with {len(data)} bars")
        
        try:
            # Initialize results
            results = BacktestResults(parameters=params)
            
            # Initialize portfolio and tracking variables
            portfolio = {
                'cash': params.initial_capital,
                'equity': params.initial_capital,
                'position': 0,
                'entry_price': 0,
                'open_position_direction': None,  # 'long' or 'short'
                'entry_date': None,
                'stop_loss': 0,
                'take_profit': 0,
                'trade_count': 0,
                'trades': [],
                'equity_curve': [],
                'drawdowns': [],
                'max_equity': params.initial_capital,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'pyramiding_count': 0,
                'active_trades': [],
                'pending_orders': [],
            }
            
            # Execute the simulation, bar by bar
            for i in range(1, len(data)):
                current_bar = data.iloc[i]
                prev_bar = data.iloc[i-1]
                date = data.index[i]
                
                # Skip if too early in the dataset for indicators
                if pd.isna(current_bar['signal']) or pd.isna(current_bar['atr']):
                    continue
                
                # Update current bar with portfolio status
                current_bar_dict = current_bar.to_dict()
                current_bar_dict['date'] = date
                current_bar_dict['cash'] = portfolio['cash']
                current_bar_dict['position'] = portfolio['position']
                current_bar_dict['equity'] = portfolio['cash']
                
                # Add position value to equity if we have a position
                if portfolio['position'] != 0:
                    position_value = portfolio['position'] * current_bar['close']
                    current_bar_dict['equity'] += position_value
                    
                # Check for exit conditions if we have an open position
                if portfolio['open_position_direction'] is not None:
                    # Calculate current profit/loss
                    if portfolio['open_position_direction'] == 'long':
                        current_pl = (current_bar['close'] - portfolio['entry_price']) * portfolio['position']
                        current_pl_pct = (current_bar['close'] / portfolio['entry_price'] - 1) * 100
                    else:  # 'short'
                        current_pl = (portfolio['entry_price'] - current_bar['close']) * abs(portfolio['position'])
                        current_pl_pct = (1 - current_bar['close'] / portfolio['entry_price']) * 100
                    
                    # Check stop loss
                    stop_loss_hit = False
                    if portfolio['open_position_direction'] == 'long' and current_bar['low'] <= portfolio['stop_loss']:
                        stop_loss_hit = True
                        exit_price = max(portfolio['stop_loss'], current_bar['open'])  # Realistic slippage
                        exit_reason = "stop_loss"
                    elif portfolio['open_position_direction'] == 'short' and current_bar['high'] >= portfolio['stop_loss']:
                        stop_loss_hit = True
                        exit_price = min(portfolio['stop_loss'], current_bar['open'])  # Realistic slippage
                        exit_reason = "stop_loss"
                        
                    # Check take profit
                    take_profit_hit = False
                    if portfolio['open_position_direction'] == 'long' and current_bar['high'] >= portfolio['take_profit']:
                        take_profit_hit = True
                        exit_price = min(portfolio['take_profit'], current_bar['open'])  # Realistic slippage
                        exit_reason = "take_profit"
                    elif portfolio['open_position_direction'] == 'short' and current_bar['low'] <= portfolio['take_profit']:
                        take_profit_hit = True
                        exit_price = max(portfolio['take_profit'], current_bar['open'])  # Realistic slippage
                        exit_reason = "take_profit"
                        
                    # Check trailing stop if enabled
                    trailing_stop_hit = False
                    if params.trailing_stop:
                        if portfolio['open_position_direction'] == 'long':
                            trail_stop_level = current_bar['close'] * (1 - params.trailing_stop_pct/100)
                            if trail_stop_level > portfolio['stop_loss']:  # Move up the stop loss
                                portfolio['stop_loss'] = trail_stop_level
                        else:  # 'short'
                            trail_stop_level = current_bar['close'] * (1 + params.trailing_stop_pct/100)
                            if trail_stop_level < portfolio['stop_loss']:  # Move down the stop loss
                                portfolio['stop_loss'] = trail_stop_level
                                
                    # Check for signal reversal
                    signal_reversal = False
                    if portfolio['open_position_direction'] == 'long' and current_bar['signal'] < 0:
                        signal_reversal = True
                        exit_price = current_bar['close']
                        exit_reason = "signal_reversal"
                    elif portfolio['open_position_direction'] == 'short' and current_bar['signal'] > 0:
                        signal_reversal = True
                        exit_price = current_bar['close']
                        exit_reason = "signal_reversal"
                        
                    # Execute exit if any condition met
                    if stop_loss_hit or take_profit_hit or trailing_stop_hit or signal_reversal:
                        # Calculate profit/loss
                        if portfolio['open_position_direction'] == 'long':
                            pl = (exit_price - portfolio['entry_price']) * portfolio['position']
                            pl_pct = (exit_price / portfolio['entry_price'] - 1) * 100
                        else:  # 'short'
                            pl = (portfolio['entry_price'] - exit_price) * abs(portfolio['position'])
                            pl_pct = (1 - exit_price / portfolio['entry_price']) * 100
                            
                        # Update cash and position
                        portfolio['cash'] += exit_price * abs(portfolio['position']) + pl
                        
                        # Record the trade
                        trade = BacktestTrade(
                            symbol=params.symbol,
                            exchange=params.exchange,
                            interval=params.interval,
                            direction=portfolio['open_position_direction'],
                            entry_date=portfolio['entry_date'],
                            entry_price=portfolio['entry_price'],
                            exit_date=date,
                            exit_price=exit_price,
                            quantity=abs(portfolio['position']),
                            profit_loss=pl,
                            profit_loss_pct=pl_pct,
                            exit_reason=exit_reason,
                            trade_duration=(date - portfolio['entry_date']).days if params.interval == 'daily' else i - portfolio['entry_index']
                        )
                        
                        # Add to trades list
                        results.trades.append(trade)
                        
                        # Reset position tracking
                        portfolio['position'] = 0
                        portfolio['open_position_direction'] = None
                        portfolio['entry_price'] = 0
                        portfolio['entry_date'] = None
                        portfolio['stop_loss'] = 0
                        portfolio['take_profit'] = 0
                        portfolio['pyramiding_count'] = 0
                
                # Entry logic - only if we have no open position or pyramiding is enabled
                valid_for_entry = (
                    portfolio['open_position_direction'] is None or
                    (params.enable_pyramid and 
                     portfolio['pyramiding_count'] < params.max_pyramid_positions and
                     portfolio['open_position_direction'] == ('long' if current_bar['signal'] > 0 else 'short'))
                )
                
                if valid_for_entry:
                    # Long signal
                    if current_bar['signal'] > 0:
                        # Calculate position size
                        position_size = current_bar['position_size']
                        
                        # Adjust for available cash
                        max_affordable = portfolio['cash'] / current_bar['close']
                        position_size = min(position_size, max_affordable)
                        
                        if position_size > 0:
                            # Calculate stop loss and take profit levels
                            stop_loss = current_bar['close'] - current_bar['stop_loss_distance']
                            take_profit = current_bar['close'] + current_bar['take_profit_distance']
                            
                            # Execute the trade
                            cost = position_size * current_bar['close']
                            portfolio['cash'] -= cost
                            portfolio['position'] += position_size
                            portfolio['entry_price'] = current_bar['close']
                            portfolio['entry_date'] = date
                            portfolio['entry_index'] = i
                            portfolio['open_position_direction'] = 'long'
                            portfolio['stop_loss'] = stop_loss
                            portfolio['take_profit'] = take_profit
                            portfolio['pyramiding_count'] += 1
                            
                    # Short signal - only if allowed
                    elif current_bar['signal'] < 0 and params.allow_short:
                        # Calculate position size (negative for short)
                        position_size = -current_bar['position_size']
                        
                        # For short selling, we still need to ensure we have enough margin
                        margin_requirement = abs(position_size) * current_bar['close'] * (params.max_leverage or 1.0)
                        if margin_requirement <= portfolio['cash']:
                            # Calculate stop loss and take profit levels
                            stop_loss = current_bar['close'] + current_bar['stop_loss_distance']
                            take_profit = current_bar['close'] - current_bar['take_profit_distance']
                            
                            # Execute the trade
                            # For short, we're borrowing the shares, so no immediate cash change
                            # Just track the position and margin used
                            portfolio['position'] += position_size  # Negative for short
                            portfolio['entry_price'] = current_bar['close']
                            portfolio['entry_date'] = date
                            portfolio['entry_index'] = i
                            portfolio['open_position_direction'] = 'short'
                            portfolio['stop_loss'] = stop_loss
                            portfolio['take_profit'] = take_profit
                            portfolio['pyramiding_count'] += 1
                
                # Update equity curve
                equity = portfolio['cash']
                if portfolio['position'] != 0:
                    position_value = portfolio['position'] * current_bar['close']
                    equity += position_value
                
                portfolio['equity_curve'].append({
                    'date': date,
                    'equity': equity,
                    'cash': portfolio['cash'],
                    'position_size': portfolio['position'],
                    'position_value': portfolio['position'] * current_bar['close'] if portfolio['position'] != 0 else 0,
                    'close': current_bar['close']
                })
                
                # Update max equity and drawdown
                if equity > portfolio['max_equity']:
                    portfolio['max_equity'] = equity
                
                # Calculate drawdown
                if portfolio['max_equity'] > 0:
                    drawdown = portfolio['max_equity'] - equity
                    drawdown_pct = (drawdown / portfolio['max_equity']) * 100
                    
                    if drawdown > portfolio['max_drawdown']:
                        portfolio['max_drawdown'] = drawdown
                        portfolio['max_drawdown_pct'] = drawdown_pct
                    
                    portfolio['drawdowns'].append({
                        'date': date,
                        'equity': equity,
                        'max_equity': portfolio['max_equity'],
                        'drawdown': drawdown,
                        'drawdown_pct': drawdown_pct
                    })
            
            # If we have an open position at the end, close it at the last price
            if portfolio['open_position_direction'] is not None:
                last_bar = data.iloc[-1]
                last_date = data.index[-1]
                
                # Calculate final profit/loss
                if portfolio['open_position_direction'] == 'long':
                    pl = (last_bar['close'] - portfolio['entry_price']) * portfolio['position']
                    pl_pct = (last_bar['close'] / portfolio['entry_price'] - 1) * 100
                else:  # 'short'
                    pl = (portfolio['entry_price'] - last_bar['close']) * abs(portfolio['position'])
                    pl_pct = (1 - last_bar['close'] / portfolio['entry_price']) * 100
                
                # Record the final trade
                trade = BacktestTrade(
                    symbol=params.symbol,
                    exchange=params.exchange,
                    interval=params.interval,
                    direction=portfolio['open_position_direction'],
                    entry_date=portfolio['entry_date'],
                    entry_price=portfolio['entry_price'],
                    exit_date=last_date,
                    exit_price=last_bar['close'],
                    quantity=abs(portfolio['position']),
                    profit_loss=pl,
                    profit_loss_pct=pl_pct,
                    exit_reason="end_of_test",
                    trade_duration=(last_date - portfolio['entry_date']).days if params.interval == 'daily' else len(data) - portfolio['entry_index']
                )
                
                # Add to trades list
                results.trades.append(trade)
                
                # Update final portfolio value
                portfolio['cash'] += last_bar['close'] * abs(portfolio['position']) + pl
                portfolio['position'] = 0
            
            # Calculate metrics
            metrics = self._calculate_metrics(portfolio, results.trades, params)
            
            # Update results
            results.equity_curve = portfolio['equity_curve']
            results.drawdowns = portfolio['drawdowns']
            results.metrics = metrics
            results.trade_count = len(results.trades)
            results.win_count = sum(1 for t in results.trades if t.profit_loss > 0)
            results.loss_count = results.trade_count - results.win_count
            
            # Additional metrics
            results.win_rate = results.win_count / results.trade_count if results.trade_count > 0 else 0
            
            winning_trades = [t for t in results.trades if t.profit_loss > 0]
            losing_trades = [t for t in results.trades if t.profit_loss <= 0]
            
            results.avg_win = sum(t.profit_loss for t in winning_trades) / len(winning_trades) if winning_trades else 0
            results.avg_loss = sum(t.profit_loss for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            results.avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
            results.avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            results.largest_win = max([t.profit_loss for t in winning_trades]) if winning_trades else 0
            results.largest_loss = min([t.profit_loss for t in losing_trades]) if losing_trades else 0
            
            results.net_profit = sum(t.profit_loss for t in results.trades)
            results.net_profit_pct = (portfolio['equity_curve'][-1]['equity'] / params.initial_capital - 1) * 100
            
            results.profit_factor = (
                sum(t.profit_loss for t in winning_trades) / abs(sum(t.profit_loss for t in losing_trades))
                if losing_trades and sum(t.profit_loss for t in losing_trades) != 0
                else float('inf')
            )
            
            results.max_drawdown = portfolio['max_drawdown']
            results.max_drawdown_pct = portfolio['max_drawdown_pct']
            
            # Time-based metrics
            if data.index[-1] > data.index[0]:
                days = (data.index[-1] - data.index[0]).days
                results.test_duration_days = max(days, 1)  # Avoid division by zero
                
                # Annualized return
                results.annualized_return = ((1 + results.net_profit_pct/100) ** (365/results.test_duration_days) - 1) * 100
                
                # Calculate monthly returns
                monthly_returns = {}
                for i in range(1, len(portfolio['equity_curve'])):
                    curr = portfolio['equity_curve'][i]
                    prev = portfolio['equity_curve'][i-1]
                    
                    month_key = curr['date'].strftime('%Y-%m')
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = prev['equity']
                    
                    # Update the ending equity for this month
                    monthly_returns[month_key] = curr['equity']
                
                # Convert to percentage returns
                prev_month = None
                prev_equity = params.initial_capital
                monthly_returns_pct = {}
                
                for month in sorted(monthly_returns.keys()):
                    monthly_return = (monthly_returns[month] / prev_equity - 1) * 100
                    monthly_returns_pct[month] = monthly_return
                    prev_equity = monthly_returns[month]
                
                results.monthly_returns = monthly_returns_pct
                
                # Calculate annual returns
                annual_returns = {}
                for month, ret in monthly_returns_pct.items():
                    year = month.split('-')[0]
                    if year not in annual_returns:
                        annual_returns[year] = 100.0  # Start at 100%
                    
                    # Compound the monthly return
                    annual_returns[year] *= (1 + ret/100)
                
                # Convert to percentage returns
                for year in annual_returns:
                    annual_returns[year] = (annual_returns[year] - 1) * 100
                
                results.annual_returns = annual_returns
                
                # Risk metrics
                # Sharpe ratio calculation (simplified, using average monthly return and std dev)
                if len(monthly_returns_pct) > 1:
                    monthly_returns_list = list(monthly_returns_pct.values())
                    avg_monthly_return = sum(monthly_returns_list) / len(monthly_returns_list)
                    std_monthly_return = np.std(monthly_returns_list)
                    
                    if std_monthly_return > 0:
                        results.sharpe_ratio = (avg_monthly_return) / std_monthly_return * np.sqrt(12)
                    
                    # Sortino ratio (using only negative returns for denominator)
                    negative_returns = [r for r in monthly_returns_list if r < 0]
                    if negative_returns:
                        downside_deviation = np.sqrt(sum(r**2 for r in negative_returns) / len(negative_returns))
                        if downside_deviation > 0:
                            results.sortino_ratio = (avg_monthly_return) / downside_deviation * np.sqrt(12)
                    
                    # MAR ratio (annualized return / max drawdown)
                    if results.max_drawdown_pct > 0:
                        results.mar_ratio = results.annualized_return / results.max_drawdown_pct
                    
                    # Calmar ratio (similar to MAR but using a 3-year lookback)
                    results.calmar_ratio = results.mar_ratio  # Simplified
            
            logger.info(f"Backtest completed with {results.trade_count} trades")
            return results
            
        except Exception as e:
            logger.error(f"Error executing backtest: {e}")
            raise RuntimeError(f"Failed to execute backtest: {e}")
    
    def _calculate_metrics(self, portfolio: Dict[str, Any], trades: List[BacktestTrade], 
                         params: BacktestParameters) -> Dict[str, Any]:
        """
        Calculate backtest metrics.
        
        Args:
            portfolio: Portfolio state
            trades: List of trades
            params: Backtest parameters
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        initial_capital = params.initial_capital
        final_capital = portfolio['equity_curve'][-1]['equity'] if portfolio['equity_curve'] else initial_capital
        
        metrics['initial_capital'] = initial_capital
        metrics['final_capital'] = final_capital
        metrics['profit_loss'] = final_capital - initial_capital
        metrics['profit_loss_pct'] = (final_capital / initial_capital - 1) * 100
        metrics['max_drawdown'] = portfolio['max_drawdown']
        metrics['max_drawdown_pct'] = portfolio['max_drawdown_pct']
        
        # Trade metrics
        metrics['total_trades'] = len(trades)
        metrics['winning_trades'] = sum(1 for t in trades if t.profit_loss > 0)
        metrics['losing_trades'] = metrics['total_trades'] - metrics['winning_trades']
        
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        # Risk metrics
        if metrics['total_trades'] > 0:
            # Calculate average profit and loss
            winning_profits = [t.profit_loss for t in trades if t.profit_loss > 0]
            losing_profits = [t.profit_loss for t in trades if t.profit_loss <= 0]
            
            metrics['avg_profit'] = sum(winning_profits) / len(winning_profits) if winning_profits else 0
            metrics['avg_loss'] = sum(losing_profits) / len(losing_profits) if losing_profits else 0
            
            metrics['largest_profit'] = max(winning_profits) if winning_profits else 0
            metrics['largest_loss'] = min(losing_profits) if losing_profits else 0
            
            metrics['profit_factor'] = (
                sum(winning_profits) / abs(sum(losing_profits))
                if losing_profits and sum(losing_profits) != 0
                else float('inf')
            )
            
            # Average holding time
            metrics['avg_bars_held'] = sum(t.trade_duration for t in trades) / len(trades)
            
        # Time-based metrics
        if len(portfolio['equity_curve']) > 1:
            start_date = portfolio['equity_curve'][0]['date']
            end_date = portfolio['equity_curve'][-1]['date']
            trading_days = (end_date - start_date).days
            
            metrics['trading_days'] = trading_days
            metrics['trades_per_year'] = metrics['total_trades'] * 365 / max(trading_days, 1)
            
            # Annualized return
            if trading_days > 0:
                metrics['annualized_return'] = ((final_capital / initial_capital) ** (365 / trading_days) - 1) * 100
            else:
                metrics['annualized_return'] = 0
            
            # Sharpe ratio (simplified)
            if len(portfolio['equity_curve']) > 2:
                returns = []
                for i in range(1, len(portfolio['equity_curve'])):
                    prev_equity = portfolio['equity_curve'][i-1]['equity']
                    curr_equity = portfolio['equity_curve'][i]['equity']
                    returns.append((curr_equity / prev_equity) - 1)
                
                avg_return = sum(returns) / len(returns)
                std_return = np.std(returns) if len(returns) > 1 else 0.0001
                
                # Simple Sharpe calculation (no risk-free rate)
                metrics['sharpe_ratio'] = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
                
                # Calculate drawdowns
                drawdowns = []
                peak = initial_capital
                for e in portfolio['equity_curve']:
                    if e['equity'] > peak:
                        peak = e['equity']
                    drawdown = (peak - e['equity']) / peak * 100
                    drawdowns.append(drawdown)
                
                metrics['avg_drawdown'] = sum(drawdowns) / len(drawdowns)
                
                # Calmar ratio
                metrics['calmar_ratio'] = (
                    metrics['annualized_return'] / metrics['max_drawdown_pct']
                    if metrics['max_drawdown_pct'] > 0
                    else float('inf')
                )
        
        return metrics

    def run_backtest(self, params: BacktestParameters) -> BacktestResults:
        """
        Run a backtest with the given parameters.
        
        Args:
            params: Backtest parameters
            
        Returns:
            BacktestResults with trades and metrics
        """
        try:
            # Fetch data
            data = self._fetch_data(params)
            self.current_data = data
            self.current_symbol = params.symbol
            self.current_interval = params.interval
            self.current_exchange = params.exchange
            
            # Detect cycles
            cycles, cycle_powers = self._detect_cycles(data, params)
            
            # Calculate FLDs
            flds = self._calculate_flds(data, cycles)
            
            # Generate signals
            signals = self._generate_signals(data, flds, cycles, params)
            
            # Calculate position sizes
            strategy_data = self._calculate_position_sizes(signals, params)
            
            # Execute backtest
            results = self._execute_backtest(strategy_data, params)
            
            # Store results
            self.results = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            # Create a minimal results object to return
            results = BacktestResults(parameters=params)
            results.trade_count = 0
            results.net_profit = 0
            results.net_profit_pct = 0
            results.max_drawdown = 0
            results.max_drawdown_pct = 0
            results.metrics = {"error": str(e)}
            return results
            
    def generate_backtest_report(self, results: Optional[BacktestResults] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive backtest report.
        
        Args:
            results: Optional BacktestResults (uses stored results if None)
            
        Returns:
            Dictionary with report data
        """
        results = results or self.results
        
        if results is None:
            logger.error("No backtest results available for report generation")
            return {"error": "No backtest results available"}
        
        try:
            # Generate summary statistics
            report = {
                "strategy": results.parameters.strategy_type,
                "symbol": results.parameters.symbol,
                "exchange": results.parameters.exchange,
                "interval": results.parameters.interval,
                "start_date": results.parameters.start_date.isoformat() if results.parameters.start_date else None,
                "end_date": results.parameters.end_date.isoformat() if results.parameters.end_date else None,
                "initial_capital": results.parameters.initial_capital,
                "final_capital": results.metrics.get('final_capital', 0),
                "net_profit": results.net_profit,
                "net_profit_pct": results.net_profit_pct,
                "annualized_return": results.annualized_return,
                "trade_count": results.trade_count,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "max_drawdown": results.max_drawdown,
                "max_drawdown_pct": results.max_drawdown_pct,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "calmar_ratio": results.calmar_ratio,
                "trades": [t.to_dict() for t in results.trades],
                "equity_curve": results.equity_curve,
                "drawdowns": results.drawdowns,
                "monthly_returns": results.monthly_returns,
                "annual_returns": results.annual_returns,
                "test_duration_days": results.test_duration_days,
                "avg_win": results.avg_win,
                "avg_loss": results.avg_loss,
                "avg_win_pct": results.avg_win_pct,
                "avg_loss_pct": results.avg_loss_pct,
                "largest_win": results.largest_win,
                "largest_loss": results.largest_loss,
                "metrics": results.metrics,
                "parameters": results.parameters.to_dict(),
            }
            
            # Transaction log
            transaction_log = []
            for trade in results.trades:
                transaction_log.append({
                    "trade_id": trade.id,
                    "type": "entry",
                    "direction": trade.direction,
                    "date": trade.entry_date.isoformat() if trade.entry_date else None,
                    "price": trade.entry_price,
                    "quantity": trade.quantity,
                    "value": trade.entry_price * trade.quantity,
                })
                transaction_log.append({
                    "trade_id": trade.id,
                    "type": "exit",
                    "direction": trade.direction,
                    "date": trade.exit_date.isoformat() if trade.exit_date else None,
                    "price": trade.exit_price,
                    "quantity": trade.quantity,
                    "value": trade.exit_price * trade.quantity,
                    "profit_loss": trade.profit_loss,
                    "profit_loss_pct": trade.profit_loss_pct,
                    "exit_reason": trade.exit_reason,
                })
            
            report["transaction_log"] = sorted(transaction_log, key=lambda x: x["date"] if x["date"] else "")
            
            # Trade analysis
            if results.trades:
                # Calculate trade statistics
                report["trade_statistics"] = {
                    "avg_trade_duration": sum(t.trade_duration for t in results.trades) / len(results.trades),
                    "max_consecutive_wins": self._max_consecutive(results.trades, lambda t: t.profit_loss > 0),
                    "max_consecutive_losses": self._max_consecutive(results.trades, lambda t: t.profit_loss <= 0),
                    "total_fees": 0,  # Not implemented yet
                    "win_loss_ratio": results.avg_win / abs(results.avg_loss) if results.avg_loss != 0 else float('inf'),
                    "expectancy": results.win_rate * results.avg_win_pct + (1 - results.win_rate) * results.avg_loss_pct,
                }
                
                # Trade distribution
                long_trades = sum(1 for t in results.trades if t.direction == 'long')
                short_trades = sum(1 for t in results.trades if t.direction == 'short')
                
                report["trade_distribution"] = {
                    "long_trades": long_trades,
                    "short_trades": short_trades,
                    "long_pct": long_trades / results.trade_count * 100 if results.trade_count > 0 else 0,
                    "short_pct": short_trades / results.trade_count * 100 if results.trade_count > 0 else 0,
                }
                
                # Exit reason distribution
                exit_reasons = {}
                for trade in results.trades:
                    exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
                
                report["exit_reason_distribution"] = {
                    reason: {
                        "count": count,
                        "percentage": count / results.trade_count * 100
                    }
                    for reason, count in exit_reasons.items()
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
            return {"error": str(e)}
    
    def _max_consecutive(self, trades: List[BacktestTrade], condition_fn) -> int:
        """Calculate the maximum consecutive occurrences where a condition is true."""
        max_streak = 0
        current_streak = 0
        
        for trade in trades:
            if condition_fn(trade):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak
    
    def _make_serializable(self, results: BacktestResults) -> Dict[str, Any]:
        """Convert results to JSON-serializable format."""
        # Simply use the to_dict method
        return results.to_dict()
    
    def save_results(self, results: BacktestResults, filepath: str):
        """Save backtest results to a file."""
        try:
            serialized = self._make_serializable(results)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serialized, f, indent=2)
                
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    def load_results(self, filepath: str) -> BacktestResults:
        """Load backtest results from a file."""
        try:
            with open(filepath, 'r') as f:
                serialized = json.load(f)
                
            # Deserialize
            results = BacktestResults.from_dict(serialized)
            
            logger.info(f"Backtest results loaded from {filepath}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading backtest results: {e}")
            raise RuntimeError(f"Failed to load backtest results: {e}")