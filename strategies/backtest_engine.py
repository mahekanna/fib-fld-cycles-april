"""
Backtesting Engine for Fibonacci Cycles Trading System

This module provides a framework for backtesting the cycle-based trading strategies
against historical market data. It simulates trading based on cycle detection and
signal generation from the strategy classes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Type
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import os
import uuid

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BacktestResult:
    """Class to store and analyze backtest results."""
    
    def __init__(self, symbol: str, timeframe: str, 
                 strategy_name: str, initial_capital: float):
        """
        Initialize backtest result container.
        
        Args:
            symbol: Traded symbol
            timeframe: Trading timeframe
            strategy_name: Name of the strategy used
            initial_capital: Initial capital for the test
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        self.signals = []
        self.final_capital = initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.total_return_pct = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.metrics = {}
    
    def add_trade(self, trade: Dict) -> None:
        """Add completed trade to results."""
        self.trades.append(trade)
    
    def add_equity_point(self, date: datetime, equity: float) -> None:
        """Add point to equity curve."""
        self.equity_curve.append({'date': date, 'equity': equity})
    
    def add_signal(self, date: datetime, signal: Dict) -> None:
        """Add generated signal to results."""
        self.signals.append({
            'date': date,
            'type': signal.get('signal', 'neutral'),
            'strength': signal.get('strength', 0),
            'confidence': signal.get('confidence', 'low')
        })
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics from backtest results."""
        if not self.trades or not self.equity_curve:
            return {}
        
        # Final capital
        if self.equity_curve:
            self.final_capital = self.equity_curve[-1]['equity']
        
        # Total return
        total_return = self.final_capital - self.initial_capital
        self.total_return_pct = (total_return / self.initial_capital) * 100
        
        # Drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = self.initial_capital
        drawdowns = []
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for equity in equity_values:
            peak = max(peak, equity)
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
            drawdowns.append(drawdown_pct)
            
            if drawdown_pct > max_drawdown_pct:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        self.drawdowns = drawdowns
        self.max_drawdown = max_drawdown
        self.max_drawdown_pct = max_drawdown_pct
        
        # Trade metrics
        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        losing_trades = [t for t in self.trades if t['profit_loss'] <= 0]
        
        self.win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        gross_profit = sum(t['profit_loss'] for t in winning_trades)
        gross_loss = abs(sum(t['profit_loss'] for t in losing_trades))
        
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Simple sharpe calculation
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / equity_values[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        # Compile metrics
        self.metrics = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'strategy': self.strategy_name,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': total_return,
            'total_return_pct': self.total_return_pct,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
        
        return self.metrics
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Extract data for plotting
        dates = [e['date'] for e in self.equity_curve]
        equity = [e['equity'] for e in self.equity_curve]
        
        # Plot equity curve
        ax1.plot(dates, equity, 'b-', linewidth=2)
        ax1.set_title(f"Equity Curve: {self.symbol} {self.timeframe} - {self.strategy_name}")
        ax1.set_ylabel("Equity")
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line at initial capital
        ax1.axhline(y=self.initial_capital, color='k', linestyle='--', alpha=0.5)
        
        # Plot drawdowns
        if self.drawdowns:
            # Ensure drawdowns align with dates
            if len(self.drawdowns) > len(dates):
                drawdowns_to_plot = self.drawdowns[:len(dates)]
            elif len(self.drawdowns) < len(dates):
                # Pad with zeros
                drawdowns_to_plot = self.drawdowns + [0] * (len(dates) - len(self.drawdowns))
            else:
                drawdowns_to_plot = self.drawdowns
                
            ax2.fill_between(dates, 0, drawdowns_to_plot, color='r', alpha=0.3)
            ax2.set_ylabel("Drawdown %")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()  # Invert y-axis for better visualization
        
        # Add trade markers to equity curve
        for trade in self.trades:
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            
            # Find closest equity curve points
            entry_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - entry_date).total_seconds()))
            exit_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - exit_date).total_seconds()))
            
            # Plot markers
            if trade['profit_loss'] > 0:
                color = 'green'
            else:
                color = 'red'
                
            ax1.scatter(dates[entry_idx], equity[entry_idx], color=color, marker='^', s=100, alpha=0.7)
            ax1.scatter(dates[exit_idx], equity[exit_idx], color=color, marker='v', s=100, alpha=0.7)
        
        # Add metrics table
        metrics_text = (
            f"Total Return: {self.total_return_pct:.2f}%\n"
            f"Win Rate: {self.win_rate*100:.1f}%\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Max Drawdown: {self.max_drawdown_pct:.2f}%\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Trades: {len(self.trades)}"
        )
        
        # Add text box with metrics
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    
    This class handles the simulation of trading based on cycle detection
    and signal generation from the strategy classes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtest engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Default parameters
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission_pct = config.get('commission_pct', 0.1)
        self.slippage_pct = config.get('slippage_pct', 0.05)
        self.max_positions = config.get('max_positions', 5)
        
        # Logging setup
        log_level = config.get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level))
        
        logger.info(f"Initialized BacktestEngine with {self.initial_capital} initial capital")
    
    def run_backtest(self, 
                   data: pd.DataFrame, 
                   strategy: BaseStrategy,
                   cycle_detector: Any, 
                   symbol: str = "UNKNOWN",
                   timeframe: str = "daily") -> BacktestResult:
        """
        Run a backtest using the provided data and strategy.
        
        Args:
            data: Historical price data with OHLCV columns
            strategy: Strategy instance to use for trading decisions
            cycle_detector: Cycle detection component to find cycles
            symbol: Symbol being tested
            timeframe: Timeframe of the data
            
        Returns:
            BacktestResult object with backtest results
        """
        logger.info(f"Starting backtest for {symbol} on {timeframe} timeframe")
        start_time = datetime.now()
        
        # Initialize the result object
        result = BacktestResult(symbol, timeframe, strategy.name, self.initial_capital)
        
        # Initialize backtest variables
        capital = self.initial_capital
        positions = {}  # Current open positions
        
        # Ensure we have enough data
        min_data_length = 100  # Minimum bars needed
        if len(data) < min_data_length:
            logger.error(f"Insufficient data for backtest. Need at least {min_data_length} bars.")
            return result
        
        # Process data bar by bar to avoid lookahead bias
        for i in range(min_data_length, len(data)):
            current_date = data.index[i]
            current_bar = data.iloc[i]
            
            # Get historical data up to current bar
            historical_data = data.iloc[:i+1].copy()
            
            # Update existing positions with current price
            updated_capital, positions = self._update_positions(
                capital, positions, current_bar, current_date, result
            )
            capital = updated_capital
            
            # Record equity at this point
            equity = capital + sum(pos.get('current_value', 0) for pos in positions.values())
            result.add_equity_point(current_date, equity)
            
            # Only analyze for new positions if we have capacity
            if len(positions) < self.max_positions:
                # Detect cycles using the cycle detector component
                try:
                    cycles, cycle_states = self._detect_cycles(historical_data, cycle_detector)
                    
                    if cycles:
                        # Detect FLD crossovers
                        fld_crossovers = []
                        for cycle in cycles:
                            crossovers = strategy.detect_fld_crossovers(historical_data, cycle)
                            fld_crossovers.extend(crossovers)
                        
                        # Generate trading signal
                        signal = strategy.generate_signal(
                            historical_data, cycles, fld_crossovers, cycle_states
                        )
                        
                        # Record signal
                        result.add_signal(current_date, signal)
                        
                        # Check for valid entry signal
                        if self._is_valid_entry_signal(signal):
                            # Enter new position
                            capital, position_id = self._enter_position(
                                capital, historical_data, current_date, signal, strategy, result
                            )
                            
                            # Add to positions if valid
                            if position_id:
                                positions[position_id] = positions[position_id]
                                logger.info(f"Entered new position: {position_id}")
                except Exception as e:
                    logger.error(f"Error during cycle detection/signal generation: {e}")
            
            # Update trailing stops if configured
            positions = strategy.update_stops(historical_data, positions)
        
        # Close any remaining positions at the end of the backtest
        if positions:
            logger.info(f"Closing {len(positions)} positions at end of backtest")
            
            final_bar = data.iloc[-1]
            final_date = data.index[-1]
            
            for pos_id, position in list(positions.items()):
                # Close at the last price
                exit_price = final_bar['close']
                exit_reason = "end_of_backtest"
                
                # Calculate profit/loss
                profit_loss, current_capital = self._calculate_exit_pnl(
                    position, exit_price, self.commission_pct
                )
                
                # Update capital
                capital += current_capital
                
                # Record trade
                self._record_trade_exit(
                    position, final_date, exit_price, profit_loss, exit_reason, result
                )
                
                # Remove from positions
                del positions[pos_id]
        
        # Calculate final metrics
        result.calculate_metrics()
        
        # Log completion
        duration = datetime.now() - start_time
        logger.info(f"Backtest completed in {duration.total_seconds():.2f} seconds")
        logger.info(f"Final capital: {result.final_capital:.2f}, Return: {result.total_return_pct:.2f}%")
        
        return result
    
    def _detect_cycles(self, data: pd.DataFrame, cycle_detector: Any) -> Tuple[List[int], List[Dict]]:
        """
        Detect dominant cycles in the data.
        
        Args:
            data: Historical price data
            cycle_detector: Cycle detection component
            
        Returns:
            Tuple of (cycles, cycle_states)
        """
        try:
            # This is a placeholder - you'll need to adapt this to your cycle detector
            result = cycle_detector.detect_cycles(data)
            
            # For now, return some default cycles for demo if detector not functioning
            if not result or not result.get('cycles'):
                # Default cycles - Fibonacci numbers
                cycles = [21, 34, 55]
                
                # Generate fake cycle states
                cycle_states = []
                for cycle in cycles:
                    is_bullish = data['close'].iloc[-1] > data['close'].iloc[-cycle] if len(data) > cycle else True
                    days_since = cycle // 3  # Random position in cycle
                    
                    cycle_states.append({
                        'cycle_length': cycle,
                        'is_bullish': is_bullish,
                        'days_since_crossover': days_since
                    })
                
                return cycles, cycle_states
            
            # Extract cycles and states from detector result
            cycles = result.get('cycles', [])
            cycle_states = result.get('cycle_states', [])
            
            return cycles, cycle_states
            
        except Exception as e:
            logger.error(f"Error in cycle detection: {e}")
            return [], []
    
    def _is_valid_entry_signal(self, signal: Dict) -> bool:
        """
        Check if a signal is valid for entry.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Boolean indicating if signal is valid for entry
        """
        signal_type = signal.get('signal', 'neutral')
        strength = abs(signal.get('strength', 0))
        confidence = signal.get('confidence', 'low')
        
        # Valid buy or sell signal with sufficient strength
        valid_signal = (signal_type in ['buy', 'sell']) and strength > 0.3
        
        # Minimum confidence level
        min_confidence = confidence in ['medium', 'high']
        
        return valid_signal and min_confidence
    
    def _enter_position(self, capital: float, data: pd.DataFrame, 
                      current_date: datetime, signal: Dict, 
                      strategy: BaseStrategy, result: BacktestResult) -> Tuple[float, Optional[str]]:
        """
        Enter a new position based on the signal.
        
        Args:
            capital: Available capital
            data: Historical price data
            current_date: Current date
            signal: Signal dictionary
            strategy: Strategy instance
            result: Result object to record trades
            
        Returns:
            Tuple of (updated_capital, position_id)
        """
        # Extract required data
        signal_type = signal.get('signal', 'neutral')
        current_price = data['close'].iloc[-1]
        
        # Skip if not a valid signal or no capital
        if signal_type not in ['buy', 'sell'] or capital <= 0:
            return capital, None
        
        # Determine trade direction
        direction = 'long' if signal_type == 'buy' else 'short'
        
        # Calculate stop loss
        stop_price = strategy.set_stop_loss(data, signal, current_price, direction)
        
        # Calculate take profit
        take_profit = strategy.set_take_profit(data, signal, current_price, stop_price, direction)
        
        # Calculate position size
        quantity = strategy.calculate_position_size(capital, signal, current_price, stop_price)
        
        # Apply slippage to entry price
        if direction == 'long':
            entry_price = current_price * (1 + self.slippage_pct/100)
        else:
            entry_price = current_price * (1 - self.slippage_pct/100)
        
        # Calculate initial position value and commission
        position_value = quantity * entry_price
        commission = position_value * (self.commission_pct / 100)
        
        # Check if we have enough capital
        if position_value + commission > capital:
            # Adjust quantity to fit available capital
            adjustment_factor = capital / (position_value + commission) * 0.99  # 1% extra buffer
            quantity = quantity * adjustment_factor
            position_value = quantity * entry_price
            commission = position_value * (self.commission_pct / 100)
        
        # Stop if quantity is too small
        if quantity < 0.01:
            return capital, None
        
        # Update capital
        capital -= (position_value + commission)
        
        # Generate position ID
        position_id = str(uuid.uuid4())
        
        # Create position record
        position = {
            'id': position_id,
            'symbol': result.symbol,
            'direction': direction,
            'entry_date': current_date,
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_price,
            'take_profit': take_profit,
            'initial_value': position_value,
            'current_value': position_value,
            'current_price': entry_price,
            'signal_strength': signal.get('strength', 0),
            'signal_confidence': signal.get('confidence', 'medium'),
            'cycles': signal.get('cycles', []),
            'cycle_length': signal.get('dominant_cycle') or signal.get('primary_cycle', 21)
        }
        
        logger.debug(f"New position: {direction} {quantity} units at {entry_price:.2f}, "
                   f"stop: {stop_price:.2f}, target: {take_profit:.2f}")
        
        # Add position to tracking
        positions = {position_id: position}
        
        return capital, position_id
    
    def _update_positions(self, capital: float, positions: Dict, 
                        current_bar: pd.Series, current_date: datetime,
                        result: BacktestResult) -> Tuple[float, Dict]:
        """
        Update open positions with current prices and check for exits.
        
        Args:
            capital: Current capital
            positions: Dictionary of open positions
            current_bar: Current price bar
            current_date: Current date
            result: Result object to record trades
            
        Returns:
            Tuple of (updated_capital, updated_positions)
        """
        if not positions:
            return capital, {}
        
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        
        # Copy to avoid modification during iteration
        updated_positions = positions.copy()
        
        for pos_id, position in list(updated_positions.items()):
            direction = position['direction']
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            quantity = position['quantity']
            
            # Update current price and value
            position['current_price'] = current_close
            position['current_value'] = quantity * current_close
            
            # Check for exit conditions
            hit_stop = False
            hit_target = False
            exit_price = None
            exit_reason = None
            
            # Check stop loss hit
            if direction == 'long' and current_low <= stop_loss:
                hit_stop = True
                exit_price = stop_loss
                exit_reason = "stop_loss"
            elif direction == 'short' and current_high >= stop_loss:
                hit_stop = True
                exit_price = stop_loss
                exit_reason = "stop_loss"
            
            # Check take profit hit
            if direction == 'long' and current_high >= take_profit:
                hit_target = True
                exit_price = take_profit
                exit_reason = "take_profit"
            elif direction == 'short' and current_low <= take_profit:
                hit_target = True
                exit_price = take_profit
                exit_reason = "take_profit"
            
            # Process exit if conditions met
            if hit_stop or hit_target:
                # Use exit price if set, otherwise close
                if exit_price is None:
                    exit_price = current_close
                
                # Calculate profit/loss
                profit_loss, position_value = self._calculate_exit_pnl(
                    position, exit_price, self.commission_pct
                )
                
                # Update capital
                capital += position_value
                
                # Record trade
                self._record_trade_exit(
                    position, current_date, exit_price, profit_loss, exit_reason, result
                )
                
                # Remove from positions
                del updated_positions[pos_id]
                
                logger.debug(f"Position {pos_id} closed: {exit_reason}, P&L: {profit_loss:.2f}")
        
        return capital, updated_positions
    
    def _calculate_exit_pnl(self, position: Dict, exit_price: float, 
                          commission_pct: float) -> Tuple[float, float]:
        """
        Calculate profit/loss for position exit.
        
        Args:
            position: Position dictionary
            exit_price: Exit price
            commission_pct: Commission percentage
            
        Returns:
            Tuple of (profit_loss, position_value)
        """
        direction = position['direction']
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        # Calculate raw p&l
        if direction == 'long':
            profit_loss = (exit_price - entry_price) * quantity
        else:
            profit_loss = (entry_price - exit_price) * quantity
        
        # Calculate exit commission
        exit_value = exit_price * quantity
        exit_commission = exit_value * (commission_pct / 100)
        
        # Deduct commission from p&l
        profit_loss -= exit_commission
        
        # Return position value (for capital update)
        position_value = exit_value - exit_commission
        
        return profit_loss, position_value
    
    def _record_trade_exit(self, position: Dict, exit_date: datetime, 
                         exit_price: float, profit_loss: float, exit_reason: str, 
                         result: BacktestResult) -> None:
        """
        Record a completed trade in the results.
        
        Args:
            position: Position dictionary
            exit_date: Exit date
            exit_price: Exit price
            profit_loss: Profit or loss amount
            exit_reason: Reason for exit
            result: Result object to record trade
        """
        # Calculate profit percentage
        initial_value = position['initial_value']
        profit_pct = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
        
        # Create trade record
        trade = {
            'symbol': position['symbol'],
            'direction': position['direction'],
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'exit_date': exit_date,
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_pct,
            'exit_reason': exit_reason,
            'trade_duration': (exit_date - position['entry_date']).days,
            'signal_strength': position.get('signal_strength', 0),
            'signal_confidence': position.get('signal_confidence', 'medium'),
            'cycle_length': position.get('cycle_length', 0)
        }
        
        # Add to results
        result.add_trade(trade)


def run_strategy_backtest(data: pd.DataFrame, 
                        strategy_class: Type[BaseStrategy], 
                        cycle_detector: Any,
                        config: Dict,
                        symbol: str = "UNKNOWN",
                        timeframe: str = "daily") -> BacktestResult:
    """
    Run a backtest for a specific strategy with sensible defaults.
    
    Args:
        data: Historical price data with OHLCV columns
        strategy_class: Strategy class to test
        cycle_detector: Cycle detection component
        config: Configuration dictionary
        symbol: Symbol being tested
        timeframe: Timeframe of the data
        
    Returns:
        BacktestResult object with backtest results
    """
    # Initialize the strategy
    strategy = strategy_class(config)
    
    # Initialize backtest engine
    engine = BacktestEngine(config)
    
    # Run the backtest
    result = engine.run_backtest(
        data=data,
        strategy=strategy,
        cycle_detector=cycle_detector,
        symbol=symbol,
        timeframe=timeframe
    )
    
    return result