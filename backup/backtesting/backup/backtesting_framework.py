import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import time
import json

# Fix imports to work when run directly
import os
import sys

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define our BacktestTrade class here to avoid import issues
@dataclass
class BacktestTrade:
    """Class to represent a completed backtest trade."""
    symbol: str
    direction: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: float
    profit_loss: float
    profit_loss_pct: float
    exit_reason: str

# Now import from absolute paths with better error handling
try:
    from fib_cycles_system.core.scanner import FibCycleScanner
except ImportError:
    try:
        from core.scanner import FibCycleScanner
    except ImportError:
        # Last resort - load from relative path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "FibCycleScanner", 
            os.path.join(project_root, "core", "scanner.py")
        )
        scanner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scanner_module)
        FibCycleScanner = scanner_module.FibCycleScanner

try:
    from fib_cycles_system.models.scan_parameters import ScanParameters
except ImportError:
    try:
        from models.scan_parameters import ScanParameters
    except ImportError:
        # Last resort - load from relative path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ScanParameters", 
            os.path.join(project_root, "models", "scan_parameters.py")
        )
        scan_params_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scan_params_module)
        ScanParameters = scan_params_module.ScanParameters

try:
    from fib_cycles_system.models.scan_result import ScanResult
except ImportError:
    try:
        from models.scan_result import ScanResult
    except ImportError:
        # Last resort - load from relative path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ScanResult", 
            os.path.join(project_root, "models", "scan_result.py")
        )
        scan_result_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scan_result_module)
        ScanResult = scan_result_module.ScanResult

try:
    from fib_cycles_system.data.fetcher import DataFetcher
except ImportError:
    try:
        from data.fetcher import DataFetcher
    except ImportError:
        try:
            from data.data_management import DataFetcher
        except ImportError:
            # Last resort - load from relative path
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "DataFetcher", 
                os.path.join(project_root, "data", "data_management.py")
            )
            fetcher_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fetcher_module)
            DataFetcher = fetcher_module.DataFetcher


@dataclass
class BacktestParameters:
    """Parameters for backtesting."""
    symbol: str
    exchange: str
    interval: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    lookback: int = 1000
    num_cycles: int = 3
    price_source: str = "close"
    initial_capital: float = 100000.0
    position_size_pct: float = 10.0
    max_open_positions: int = 5
    trailing_stop: bool = False
    trailing_stop_pct: float = 5.0
    take_profit_multiplier: float = 2.0  # R:R ratio
    rebalance_frequency: Optional[str] = None  # 'daily', 'weekly', 'monthly'
    require_alignment: bool = True
    min_strength: float = 0.3
    pyramiding: int = 0  # Number of additional entries allowed
    strategy_type: str = "fib_cycle"  # Strategy type to use for backtesting


class BacktestEngine:
    """
    Backtesting engine for the Fibonacci Cycle Trading System.
    Tests strategy performance on historical data.
    """
    
    def __init__(self, 
                 config: Dict,
                 scanner: Optional[FibCycleScanner] = None,
                 data_fetcher: Optional[DataFetcher] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Configuration dictionary
            scanner: Optional FibCycleScanner instance
            data_fetcher: Optional DataFetcher instance
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up components
        self.scanner = scanner or FibCycleScanner(config)
        self.data_fetcher = data_fetcher or DataFetcher(config)
    
    def run_backtest(self, params: BacktestParameters) -> Dict:
        """
        Run a backtest with the specified parameters.
        
        Args:
            params: BacktestParameters instance
            
        Returns:
            Dictionary containing backtest results
        """
        self.logger.info(f"Starting backtest for {params.symbol} on {params.interval} timeframe")
        start_time = time.time()
        
        try:
            # Fetch data
            data = self.data_fetcher.get_historical_data(
                symbol=params.symbol,
                exchange=params.exchange,
                interval=params.interval,
                start_date=params.start_date,
                end_date=params.end_date,
                lookback=params.lookback
            )
            
            if data is None or data.empty:
                raise ValueError(f"No data fetched for {params.symbol}")
            
            self.logger.info(f"Fetched {len(data)} bars of historical data")
            
            # Initialize tracking variables
            equity_curve = []
            trades = []
            positions = {}  # Currently open positions
            current_capital = params.initial_capital
            available_capital = params.initial_capital
            
            # Process data bar by bar - make sure lookback is within bounds
            safe_lookback = min(params.lookback, len(data) - 1)
            if safe_lookback != params.lookback:
                self.logger.warning(f"Adjusted processing lookback from {params.lookback} to {safe_lookback} due to data size")
            
            for i in range(safe_lookback, len(data)):
                current_date = data.index[i]
                current_bar = data.iloc[i]
                
                # Slice historical data up to current bar
                historical_slice = data.iloc[:i+1].copy()
                
                # Update open positions
                update_result = self._update_positions(
                    positions=positions,
                    current_bar=current_bar,
                    current_date=current_date,
                    params=params
                )
                
                # Update capital and equity
                current_capital = update_result['capital']
                available_capital = current_capital - update_result['allocated_capital']
                
                # Add closed trades
                trades.extend(update_result['closed_trades'])
                
                # Check if we should generate signals
                should_analyze = self._should_analyze(
                    current_date=current_date,
                    params=params
                )
                
                if not should_analyze:
                    # Just record equity and continue
                    equity_curve.append({
                        'date': current_date,
                        'equity': current_capital,
                        'open_positions': len(positions)
                    })
                    continue
                
                # Can we open new positions?
                if len(positions) >= params.max_open_positions:
                    # Record equity and continue
                    equity_curve.append({
                        'date': current_date,
                        'equity': current_capital,
                        'open_positions': len(positions)
                    })
                    continue
                
                # Analyze current state and generate signals
                scan_params = ScanParameters(
                    symbol=params.symbol,
                    exchange=params.exchange,
                    interval=params.interval,
                    lookback=params.lookback,
                    num_cycles=params.num_cycles,
                    price_source=params.price_source,
                    generate_chart=False
                )
                
                # Run analysis on historical slice
                result = self._analyze_historical_slice(
                    scanner=self.scanner,
                    data=historical_slice,
                    params=scan_params
                )
                
                if not result.success:
                    # Skip this bar if analysis failed
                    self.logger.warning(f"Analysis failed at {current_date}: {result.error}")
                    continue
                
                # Check for trade signals
                if self._should_enter_position(result, params):
                    # Calculate position size
                    position_size = self._calculate_position_size(
                        available_capital=available_capital,
                        position_size_pct=params.position_size_pct,
                        result=result
                    )
                    
                    # Enter new position
                    new_position = self._enter_position(
                        result=result,
                        current_date=current_date,
                        position_size=position_size,
                        params=params
                    )
                    
                    # Add to positions dictionary
                    position_id = f"{params.symbol}_{len(trades)}"
                    positions[position_id] = new_position
                    
                    # Update available capital
                    available_capital -= new_position['allocated_capital']
                
                # Record equity
                equity_curve.append({
                    'date': current_date,
                    'equity': current_capital,
                    'open_positions': len(positions)
                })
            
            # Close any remaining open positions
            final_bar = data.iloc[-1]
            final_date = data.index[-1]
            
            for position_id, position in list(positions.items()):
                # Close at final bar price
                exit_price = final_bar['close']
                
                # Calculate P&L
                if position['direction'] == 'long':
                    profit_loss = (exit_price - position['entry_price']) * position['quantity']
                else:
                    profit_loss = (position['entry_price'] - exit_price) * position['quantity']
                
                # Create trade record
                trade = BacktestTrade(
                    symbol=params.symbol,
                    direction=position['direction'],
                    entry_date=position['entry_date'],
                    entry_price=position['entry_price'],
                    exit_date=final_date,
                    exit_price=exit_price,
                    quantity=position['quantity'],
                    profit_loss=profit_loss,
                    profit_loss_pct=(profit_loss / position['allocated_capital']) * 100,
                    exit_reason="end_of_data"
                )
                
                # Add to trades list
                trades.append(trade)
                
                # Update capital
                current_capital += position['allocated_capital'] + profit_loss
                
                # Remove from positions
                del positions[position_id]
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                equity_curve=equity_curve,
                trades=trades,
                initial_capital=params.initial_capital
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create results dictionary with safe index access
            # Make sure the lookback index doesn't exceed the available data
            safe_lookback = min(params.lookback, len(data) - 1)
            if safe_lookback != params.lookback:
                self.logger.warning(f"Adjusted lookback from {params.lookback} to {safe_lookback} due to data size")
                
            results = {
                'symbol': params.symbol,
                'interval': params.interval,
                'start_date': data.index[safe_lookback] if len(data) > safe_lookback else data.index[0],
                'end_date': data.index[-1],
                'duration': (data.index[-1] - data.index[safe_lookback if len(data) > safe_lookback else 0]).days,
                'initial_capital': params.initial_capital,
                'final_capital': current_capital,
                'trades': [t.__dict__ for t in trades],
                'metrics': metrics,
                'equity_curve': equity_curve,
                'execution_time': execution_time
            }
            
            self.logger.info(f"Backtest completed in {execution_time:.2f} seconds with "
                           f"{metrics['total_trades']} trades and "
                           f"{metrics['profit_loss_pct']:.2f}% total return")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}", exc_info=True)
            raise
    
    def _update_positions(self, 
                         positions: Dict, 
                         current_bar: pd.Series, 
                         current_date: datetime,
                         params: BacktestParameters) -> Dict:
        """
        Update open positions based on current bar data.
        
        Args:
            positions: Dictionary of open positions
            current_bar: Current price bar
            current_date: Current date
            params: Backtest parameters
            
        Returns:
            Dictionary with updated capital and closed trades
        """
        capital = 0  # Current capital allocated to positions
        closed_trades = []
        
        for position_id, position in list(positions.items()):
            # Check for stop loss or take profit
            hit_stop = False
            hit_target = False
            exit_price = None
            exit_reason = None
            
            # Get current prices
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_close = current_bar['close']
            
            if position['direction'] == 'long':
                # Check if low hit stop loss
                if current_low <= position['stop_loss']:
                    hit_stop = True
                    exit_price = position['stop_loss']
                    exit_reason = "stop_loss"
                
                # Check if high hit take profit
                elif current_high >= position['take_profit']:
                    hit_target = True
                    exit_price = position['take_profit']
                    exit_reason = "take_profit"
                
                # Update trailing stop if enabled
                elif params.trailing_stop:
                    # Calculate potential new stop
                    new_stop = current_close * (1 - params.trailing_stop_pct/100)
                    
                    # Only move stop up, never down
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
            
            else:  # Short position
                # Check if high hit stop loss
                if current_high >= position['stop_loss']:
                    hit_stop = True
                    exit_price = position['stop_loss']
                    exit_reason = "stop_loss"
                
                # Check if low hit take profit
                elif current_low <= position['take_profit']:
                    hit_target = True
                    exit_price = position['take_profit']
                    exit_reason = "take_profit"
                
                # Update trailing stop if enabled
                elif params.trailing_stop:
                    # Calculate potential new stop
                    new_stop = current_close * (1 + params.trailing_stop_pct/100)
                    
                    # Only move stop down, never up
                    if new_stop < position['stop_loss']:
                        position['stop_loss'] = new_stop
            
            # If position is closed
            if hit_stop or hit_target:
                # Calculate P&L
                if position['direction'] == 'long':
                    profit_loss = (exit_price - position['entry_price']) * position['quantity']
                else:
                    profit_loss = (position['entry_price'] - exit_price) * position['quantity']
                
                # Create trade record
                trade = BacktestTrade(
                    symbol=params.symbol,
                    direction=position['direction'],
                    entry_date=position['entry_date'],
                    entry_price=position['entry_price'],
                    exit_date=current_date,
                    exit_price=exit_price,
                    quantity=position['quantity'],
                    profit_loss=profit_loss,
                    profit_loss_pct=(profit_loss / position['allocated_capital']) * 100,
                    exit_reason=exit_reason
                )
                
                # Add to closed trades
                closed_trades.append(trade)
                
                # Remove from positions
                del positions[position_id]
            else:
                # Position still open, add to capital
                capital += position['allocated_capital']
        
        # Calculate new capital value after closed trades
        new_capital = capital
        for trade in closed_trades:
            new_capital += trade.profit_loss
        
        return {
            'capital': new_capital,
            'allocated_capital': capital,
            'closed_trades': closed_trades
        }
    
    def _should_analyze(self, current_date: datetime, params: BacktestParameters) -> bool:
        """
        Determine if analysis should be performed on the current date.
        
        Args:
            current_date: Current date
            params: Backtest parameters
            
        Returns:
            Boolean indicating whether to analyze
        """
        # Always analyze if no rebalance frequency specified
        if not params.rebalance_frequency:
            return True
        
        # Check rebalance frequency
        if params.rebalance_frequency == 'daily':
            return True
        
        elif params.rebalance_frequency == 'weekly':
            # Analyze on Mondays
            return current_date.weekday() == 0
        
        elif params.rebalance_frequency == 'monthly':
            # Analyze on first day of month
            return current_date.day == 1
        
        # Default to daily
        return True
    
    def _analyze_historical_slice(self, 
                                 scanner: FibCycleScanner,
                                 data: pd.DataFrame,
                                 params: ScanParameters) -> ScanResult:
        """
        Analyze a historical slice of data.
        
        Args:
            scanner: FibCycleScanner instance
            data: Historical price data slice
            params: Scan parameters
            
        Returns:
            ScanResult instance
        """
        # Create a copy of parameters with data slice
        modified_params = ScanParameters(
            symbol=params.symbol,
            exchange=params.exchange,
            interval=params.interval,
            lookback=params.lookback,
            num_cycles=params.num_cycles,
            price_source=params.price_source,
            generate_chart=params.generate_chart,
            custom_data=data
        )
        
        # Run analysis
        return scanner.analyze_symbol(modified_params)
    
    def _should_enter_position(self, 
                              result: ScanResult, 
                              params: BacktestParameters) -> bool:
        """
        Determine if a new position should be entered based on scan result.
        
        Args:
            result: ScanResult instance
            params: Backtest parameters
            
        Returns:
            Boolean indicating whether to enter position
        """
        # Log the signal strength and alignment for debugging
        self.logger.debug(f"Signal strength: {abs(result.signal['strength']):.4f}, min required: {params.min_strength:.4f}")
        self.logger.debug(f"Alignment: {result.signal['alignment']:.4f}, min required: {0.7 if params.require_alignment else 0:.4f}")
        
        # Check signal strength with more lenient requirements
        if abs(result.signal['strength']) < params.min_strength:
            self.logger.debug(f"Signal strength too low: {abs(result.signal['strength']):.4f} < {params.min_strength:.4f}")
            return False
        
        # Check alignment if required, with more lenient requirements
        if params.require_alignment and result.signal['alignment'] < 0.6:  # Changed from 0.7 to 0.6
            self.logger.debug(f"Cycle alignment too low: {result.signal['alignment']:.4f} < 0.6")
            return False
        
        # Check for valid signals - include weak signals too
        valid_buy_signals = ['buy', 'strong_buy', 'weak_buy']  # Added weak_buy
        valid_sell_signals = ['sell', 'strong_sell', 'weak_sell']  # Added weak_sell
        
        # Log the signal type
        self.logger.debug(f"Signal type: {result.signal['signal']}, valid: {(result.signal['signal'] in valid_buy_signals) or (result.signal['signal'] in valid_sell_signals)}")
        
        return (result.signal['signal'] in valid_buy_signals) or (result.signal['signal'] in valid_sell_signals)
    
    def _calculate_position_size(self, 
                                available_capital: float,
                                position_size_pct: float,
                                result: ScanResult) -> float:
        """
        Calculate position size based on available capital and risk.
        
        Args:
            available_capital: Available capital
            position_size_pct: Position size percentage
            result: ScanResult instance
            
        Returns:
            Position size
        """
        # Calculate position allocation
        position_allocation = available_capital * (position_size_pct / 100)
        
        # Calculate number of shares/contracts
        current_price = result.price
        
        # Round to whole number of shares
        quantity = int(position_allocation / current_price)
        
        return quantity
    
    def _enter_position(self, 
                       result: ScanResult,
                       current_date: datetime,
                       position_size: float,
                       params: BacktestParameters) -> Dict:
        """
        Enter a new position based on scan result.
        
        Args:
            result: ScanResult instance
            current_date: Current date
            position_size: Position size
            params: Backtest parameters
            
        Returns:
            Dictionary containing position information
        """
        # Determine direction from signal
        is_long = 'buy' in result.signal['signal']
        direction = 'long' if is_long else 'short'
        
        # Get entry price
        entry_price = result.price
        
        # Get stop loss and take profit
        stop_loss = result.position_guidance['stop_loss']
        take_profit = result.position_guidance['target_price']
        
        # Calculate allocated capital
        allocated_capital = position_size * entry_price
        
        # Create position dictionary
        position = {
            'symbol': result.symbol,
            'direction': direction,
            'entry_date': current_date,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'quantity': position_size,
            'allocated_capital': allocated_capital,
            'signal_strength': result.signal['strength'],
            'signal_alignment': result.signal['alignment'],
            'cycles': result.detected_cycles
        }
        
        return position
    
    def _calculate_performance_metrics(self, 
                                      equity_curve: List[Dict],
                                      trades: List[BacktestTrade],
                                      initial_capital: float) -> Dict:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            equity_curve: List of equity points
            trades: List of completed trades
            initial_capital: Initial capital
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_loss': 0.0,
                'profit_loss_pct': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'avg_profit_per_trade': 0.0,
                'avg_loss_per_trade': 0.0,
                'avg_profit_loss_pct': 0.0,
                'profit_factor': 0.0
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.profit_loss > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Profit/Loss metrics
        profit_loss = sum(t.profit_loss for t in trades)
        profit_loss_pct = (profit_loss / initial_capital) * 100
        
        # Average trade metrics
        winning_trade_profits = [t.profit_loss for t in trades if t.profit_loss > 0]
        losing_trade_losses = [abs(t.profit_loss) for t in trades if t.profit_loss < 0]
        
        avg_profit_per_trade = np.mean(winning_trade_profits) if winning_trade_profits else 0.0
        avg_loss_per_trade = np.mean(losing_trade_losses) if losing_trade_losses else 0.0
        avg_profit_loss_pct = np.mean([t.profit_loss_pct for t in trades])
        
        # Profit factor
        gross_profit = sum(t.profit_loss for t in trades if t.profit_loss > 0)
        gross_loss = sum(abs(t.profit_loss) for t in trades if t.profit_loss < 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Drawdown analysis
        equity_values = [e['equity'] for e in equity_curve]
        max_equity = initial_capital
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        
        for equity in equity_values:
            max_equity = max(max_equity, equity)
            drawdown = max_equity - equity
            drawdown_pct = (drawdown / max_equity) * 100
            
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        
        # Sharpe ratio (simplified - assuming risk-free rate of 0)
        if len(equity_values) > 1:
            equity_returns = [(equity_values[i] / equity_values[i-1]) - 1 
                           for i in range(1, len(equity_values))]
            avg_return = np.mean(equity_returns)
            std_return = np.std(equity_returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'avg_profit_per_trade': avg_profit_per_trade,
            'avg_loss_per_trade': avg_loss_per_trade,
            'avg_profit_loss_pct': avg_profit_loss_pct,
            'profit_factor': profit_factor
        }
    
    def plot_equity_curve(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot equity curve from backtest results.
        
        Args:
            results: Backtest results dictionary
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        equity_curve = results['equity_curve']
        dates = [e['date'] for e in equity_curve]
        equity = [e['equity'] for e in equity_curve]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curve
        ax.plot(dates, equity, label='Equity', color='blue', linewidth=2)
        
        # Add initial capital reference line
        ax.axhline(y=results['initial_capital'], color='gray', linestyle='--', alpha=0.7,
                 label=f"Initial Capital ({results['initial_capital']})")
        
        # Add trade markers
        for trade in results['trades']:
            # Convert to datetime if string
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            
            if isinstance(entry_date, str):
                entry_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
            
            if isinstance(exit_date, str):
                exit_date = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
            
            # Find nearest equity values
            entry_idx = np.abs(np.array(dates) - entry_date).argmin()
            exit_idx = np.abs(np.array(dates) - exit_date).argmin()
            
            entry_equity = equity[entry_idx]
            exit_equity = equity[exit_idx]
            
            # Determine color
            color = 'green' if trade['profit_loss'] > 0 else 'red'
            
            # Plot entry point
            ax.scatter(entry_date, entry_equity, color=color, marker='^', s=50, alpha=0.7)
            
            # Plot exit point
            ax.scatter(exit_date, exit_equity, color=color, marker='v', s=50, alpha=0.7)
            
            # Connect with line
            ax.plot([entry_date, exit_date], [entry_equity, exit_equity], color=color, alpha=0.3)
        
        # Format plot
        ax.set_title(f"Equity Curve - {results['symbol']} ({results['start_date']} to {results['end_date']})")
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.grid(True, alpha=0.3)
        
        # Add metrics annotation
        metrics = results['metrics']
        metrics_text = (
            f"Total Return: {metrics['profit_loss_pct']:.2f}%\n"
            f"Win Rate: {metrics['win_rate']*100:.2f}%\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
        )
        
        # Add metrics box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_results(self, 
                    results: Dict, 
                    file_path: str) -> None:
        """
        Save backtest results to a file.
        
        Args:
            results: Backtest results dictionary
            file_path: Path to save results
        """
        # Convert any non-serializable objects
        serializable_results = self._make_serializable(results)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj):
        """
        Convert non-serializable objects to serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
