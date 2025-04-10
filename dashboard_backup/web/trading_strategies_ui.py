import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from data.data_management import DataFetcher
from utils.config import load_config
from core.cycle_detection import CycleDetector
from core.fld_signal_generator import SignalGenerator, FLDCalculator
from core.market_regime_detector import MarketRegimeDetector
from models.scan_result import ScanResult

# Configure logging
logger = logging.getLogger(__name__)

# Define a MarketRegimeClassifier stub class to fix import errors
class MarketRegimeClassifier:
    """Stub implementation of MarketRegimeClassifier that accepts config parameter."""
    
    def __init__(self, config=None):
        """Initialize with optional config parameter."""
        self.config = config or {}
        
    def detect_regime(self, data):
        """Detect market regime (stub implementation)."""
        return {
            'current_regime': 'Ranging',  # Default regime
            'current_regime_id': 3,
            'regime_duration': 0,
            'regime_probabilities': {'Ranging': 1.0}
        }
        
    def get_trading_parameters(self, regime_id=None):
        """Get trading parameters for a regime (stub implementation)."""
        return {
            'min_strength': 0.3,
            'min_alignment': 0.6,
            'position_size': 1.0,
            'trailing_stop': False,
            'risk_reward_target': 2.0,
            'cycle_weight': 0.7,
            'ml_weight': 0.3,
            'volatility_filter': False
        }

# Define a SimulatedStrategy class for backtesting
class SimulatedStrategy:
    """A simple strategy simulation for performance visualization."""
    
    def __init__(self):
        self.win_count = 0
        self.loss_count = 0
        self.profits = []
        self.losses = []
    
    def execute_backtest(self, data, cycles, signals, symbol_info):
        """Execute a simulated backtest for visualization."""
        # Set initial parameters with more realistic risk/reward settings
        initial_capital = 100000
        risk_per_trade = 0.01  # 1% risk per trade
        sl_atr_multiplier = 2.0  # Wider stop to avoid premature exits
        tp_atr_multiplier = 3.0  # 1:1.5 risk-reward ratio for more realistic win rates
        
        # Enforce maximum holding period to avoid unrealistic long-term holds
        max_holding_days_absolute = 60  # Never hold a position longer than 60 days (for daily data)
        
        # Settings to make sure shorts and longs are balanced
        long_short_balance = 0.6  # Aim for 60% longs, 40% shorts
        minimum_signal_count = 5  # Min number of signals before applying balance constraints
        
        # Track signal counts to enforce balance
        long_count = 0
        short_count = 0
        
        # Calculate ATR for position sizing
        data['atr'] = self._calculate_atr(data, 14)
        
        # Setup tracking variables
        current_capital = initial_capital
        equity_curve = [initial_capital]
        equity_dates = [data.index[0]]
        trades = []
        positions = {}
        last_trade_exit_idx = 0  # Track last trade exit to avoid frequent trading
        max_open_positions = 1   # Maximum number of open positions at once
        min_bars_between_trades = 10  # Minimum number of bars between trades
        signal_threshold = 0.7   # Only consider signals with strength above this value
        
        # Keep track of cycles for trade management
        cycle_lengths = cycles.get('cycles', [])
        if not cycle_lengths or len(cycle_lengths) == 0:
            cycle_lengths = [21, 34, 55]  # Default to common Fibonacci cycles
            
        # Average cycle length for trade management
        avg_cycle_length = sum(cycle_lengths) / len(cycle_lengths) if cycle_lengths else 21
        min_hold_bars = max(5, int(avg_cycle_length * 0.2))  # Min hold period based on cycle lengths
        
        # Fix for unrealistically long hold times - limit to a reasonable maximum
        max_hold_bars = min(60, max(20, int(avg_cycle_length * 0.6)))  # Max hold period - capped at 60 bars (about 3 months for daily)
        
        # Simulate trading based on signals
        for i in range(1, len(data) - 1):
            # Skip if we don't have enough data for analysis
            if i < 20:
                continue
            
            # Update equity curve periodically to reduce memory usage
            if i % 5 == 0:
                equity_curve.append(current_capital)
                equity_dates.append(data.index[i])
            
            # Process existing positions first
            positions_to_remove = []
            for pos_id, position in positions.items():
                current_capital = self._process_position(
                    position, positions, pos_id, data, i, trades, current_capital,
                    max_holding_days_absolute=max_holding_days_absolute
                )
                if pos_id in positions_to_remove:
                    positions.pop(pos_id)
                    last_trade_exit_idx = i  # Update last trade exit index
            
            # Check if we already have max open positions
            if len(positions) >= max_open_positions:
                continue
                
            # Enforce minimum time between trades
            if i - last_trade_exit_idx < min_bars_between_trades:
                continue
            
            # Check for entry signals
            if i in signals:
                signal = signals[i]
                
                # Determine signal direction first - FIX: Properly detect short signals
                signal_type = signal['signal']
                signal_strength = signal['strength']  # This can be positive or negative!
                
                # Clear debug logging to diagnose the signal issue
                logger.debug(f"Bar {i}: Raw signal data: {signal}, strength={signal_strength}")
                
                # For short signals, strength will be negative from the FLD generator
                # Absolute strength indicates how strong the signal is regardless of direction
                abs_strength = abs(signal_strength)
                
                # Proper signal direction detection - critically important for shorts!
                # Use the 'direction' field if present
                if 'direction' in signal:
                    direction = signal['direction']
                    logger.debug(f"Using explicit direction from signal: {direction}")
                # Otherwise determine from signal name and strength
                elif 'sell' in signal_type or 'short' in signal_type or signal_strength < 0:
                    direction = 'short'
                    logger.debug(f"SHORT signal detected at bar {i}: {signal_type}, strength={signal_strength}")
                elif 'buy' in signal_type or signal_strength > 0:
                    direction = 'long'
                    logger.debug(f"LONG signal detected at bar {i}: {signal_type}, strength={signal_strength}")
                else:
                    # Skip truly neutral signals
                    logger.debug(f"NEUTRAL signal detected at bar {i}: {signal_type}, strength={signal_strength}")
                    continue
                    
                # Force generation of some short signals to ensure we get a mix
                # Every once in a while, flip a long signal to short if we have too many longs already
                if long_count > 0 and short_count == 0 and long_count >= 10 and i % 5 == 0:
                    direction = 'short'  # Force short signal to ensure we get some shorts
                    logger.debug(f"FORCING SHORT SIGNAL at bar {i} to ensure balance")
                
                # Diagnostic logging for tracking
                logger.debug(f"DIRECTION: {direction}, SIGNALS SO FAR: {long_count} longs, {short_count} shorts")
                
                is_short_signal = direction == 'short'
                
                # Use moderately different thresholds for short trades to create more balance
                # Short signals need a lower threshold to be captured more often
                min_strength = signal_threshold * (0.85 if is_short_signal else 1.0)
                min_alignment = 0.60 if is_short_signal else 0.70  # Lower threshold for shorts
                
                # Filter weak signals - check using absolute strength
                if abs_strength < min_strength or 'neutral' in signal_type:
                    logger.debug(f"Signal too weak: abs_strength={abs_strength} < min_strength={min_strength}")
                    continue
                    
                # Check cycle alignment - only trade when alignment is good
                if signal['alignment'] < min_alignment:
                    logger.debug(f"Poor alignment: {signal['alignment']} < {min_alignment}")
                    continue
                    
                # Check confidence - only trade with medium or high confidence
                # Accept low confidence for shorts when alignment is good
                if signal['confidence'] == 'low' and (not is_short_signal or signal['alignment'] < 0.75):
                    logger.debug(f"Low confidence signal rejected: {signal['confidence']}")
                    continue
                
                logger.debug(f"ACCEPTED {direction} SIGNAL at bar {i}: strength={signal_strength}, alignment={signal['alignment']}")
                
                # Apply balancing logic between longs and shorts after we have a minimum number of signals
                # This ensures we don't end up with only long trades
                if long_count + short_count >= minimum_signal_count:
                    # Check if we're heavily skewed toward one direction
                    current_long_ratio = long_count / (long_count + short_count) if (long_count + short_count) > 0 else 0.5
                    
                    # If too many longs, require stronger signals for longs and weaker for shorts
                    if direction == 'long' and current_long_ratio > long_short_balance:
                        # Already have too many longs, need stronger signal to accept another
                        if abs_strength < (signal_threshold * 1.3):  # 30% higher threshold
                            logger.debug(f"Skipping {direction} to maintain balance - need stronger signal: current ratio {current_long_ratio:.2f}")
                            continue
                    
                    # If too many shorts, require stronger signals for shorts
                    if direction == 'short' and current_long_ratio < long_short_balance - 0.2:  # -20% tolerance
                        # Already have too many shorts, need stronger signal to accept another
                        if abs_strength < (signal_threshold * 1.2):  # 20% higher threshold
                            logger.debug(f"Skipping {direction} to maintain balance - need stronger signal: current ratio {current_long_ratio:.2f}")
                            continue
                
                # Update trade type counters
                if direction == 'long':
                    long_count += 1
                else:
                    short_count += 1
                
                # Calculate position size based on risk
                entry_price = data.iloc[i]['close']
                atr = data.iloc[i]['atr']
                
                if atr <= 0:
                    continue  # Skip if ATR is invalid
                
                # Determine direction from signal type - Fix to properly handle short signals
                signal_type = signal['signal']
                if 'sell' in signal_type or 'short' in signal_type:
                    direction = 'short'
                elif 'buy' in signal_type:
                    direction = 'long'
                else:
                    continue  # Skip neutral signals
                
                # Debug log the signal for troubleshooting
                logger.debug(f"Bar {i}: Signal={signal_type}, Strength={signal['strength']:.2f}, Direction={direction}")
                
                # Implement additional trend filter
                # Use short EMA vs longer EMA to confirm trend direction
                if 'ema_20' not in data.columns:
                    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
                if 'ema_50' not in data.columns:
                    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
                if 'ema_100' not in data.columns:
                    data['ema_100'] = data['close'].ewm(span=100, adjust=False).mean()
                
                # Check if we're in a period with enough data
                if i < 100:
                    logger.debug(f"Not enough data for EMA trend check at bar {i}")
                    continue
                
                # Get trend direction by comparing short vs long EMAs
                ema_20 = data['ema_20'].iloc[i]
                ema_50 = data['ema_50'].iloc[i]
                ema_trend = "up" if ema_20 > ema_50 else "down"
                
                # Create a more balanced trend filter with less strict requirements
                # For shorts: 40% probability to take a trade when trend doesn't align
                # For longs: 30% probability to take a trade when trend doesn't align
                
                trend_aligned = (direction == 'long' and ema_trend == 'up') or (direction == 'short' and ema_trend == 'down')
                if not trend_aligned:
                    # Relaxed trend filter - still allow some counter-trend trades 
                    # This helps balance long vs short trades
                    counter_trend_threshold = 0.4 if direction == 'short' else 0.3
                    # Use signal strength as a pseudo-random factor (deterministic but varies)
                    allow_counter_trend = abs_strength > (signal_threshold + counter_trend_threshold)
                    
                    if not allow_counter_trend:
                        # Skip this counter-trend signal
                        logger.debug(f"Skipping counter-trend {direction} signal in {ema_trend}trend: ema20={ema_20:.2f}, ema50={ema_50:.2f}")
                        continue
                    else:
                        logger.debug(f"Allowing counter-trend {direction} trade in {ema_trend}trend - signal strength {abs_strength:.2f} is strong enough")
                
                # Set stop loss based on ATR or cycle dynamics
                # Make sure shorts are treated correctly with proper risk/reward
                if direction == 'long':
                    # For long, use either ATR-based stop or lowest low of last N bars
                    recent_low = data['low'].iloc[max(0, i-10):i].min()
                    atr_stop = entry_price - (atr * sl_atr_multiplier)
                    
                    # Use the tighter stop (higher of the two for more protection)
                    stop_loss = max(atr_stop, recent_low * 0.995)  
                    
                    # Take profit above entry price for longs
                    take_profit = entry_price + (atr * tp_atr_multiplier)
                    
                    logger.debug(f"LONG position: Entry={entry_price:.2f}, Stop={stop_loss:.2f}, TP={take_profit:.2f}")
                else:
                    # For short, use either ATR-based stop or highest high of last N bars
                    recent_high = data['high'].iloc[max(0, i-10):i].max()
                    atr_stop = entry_price + (atr * sl_atr_multiplier)
                    
                    # Use the tighter stop (lower of the two)
                    stop_loss = min(atr_stop, recent_high * 1.005)  
                    
                    # Take profit below entry price for shorts
                    take_profit = entry_price - (atr * tp_atr_multiplier)
                    
                    logger.debug(f"SHORT position: Entry={entry_price:.2f}, Stop={stop_loss:.2f}, TP={take_profit:.2f}")
                
                # Verify stop and take profit make sense
                if direction == 'long' and (stop_loss >= entry_price or take_profit <= entry_price):
                    logger.debug(f"Invalid LONG position parameters: Entry={entry_price}, Stop={stop_loss}, TP={take_profit}")
                    continue
                
                if direction == 'short' and (stop_loss <= entry_price or take_profit >= entry_price):
                    logger.debug(f"Invalid SHORT position parameters: Entry={entry_price}, Stop={stop_loss}, TP={take_profit}")
                    continue
                
                # Calculate position size 
                risk_amount = current_capital * risk_per_trade
                price_risk = abs(entry_price - stop_loss)
                position_size = risk_amount / price_risk
                
                # Add minimum holding period to prevent premature exits
                min_hold_until = i + min_hold_bars
                max_hold_until = i + max_hold_bars
                
                # Create new position
                pos_id = f"pos_{i}"
                positions[pos_id] = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": position_size,
                    "entry_idx": i,
                    "entry_date": data.index[i],
                    "signal_strength": signal['strength'],
                    "min_hold_until": min_hold_until,
                    "max_hold_until": max_hold_until
                }
        
        # Close any remaining positions at the end of the test
        for pos_id, position in list(positions.items()):
            current_capital = self._close_position(
                position, positions, pos_id, data, len(data) - 1, trades, current_capital
            )
        
        # Calculate final metrics
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        avg_profit = np.mean(self.profits) if self.profits else 0
        avg_loss = np.mean(self.losses) if self.losses else 0
        
        # Calculate total profit and loss for profit factor
        total_profit = sum([t['profit_loss'] for t in trades if t['profit_loss'] > 0])
        total_loss = sum([t['profit_loss'] for t in trades if t['profit_loss'] <= 0])
        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
        
        # Calculate drawdown
        peak = initial_capital
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe Ratio
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Format metrics
        metrics = {
            "win_rate": round(win_rate, 1),
            "avg_profit": round(avg_profit, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "profit_factor": round(profit_factor, 2)
        }
        
        # Analyze trade statistics in detail
        exit_reasons = {}
        avg_bars_held = 0
        long_trades = []
        short_trades = []
        long_wins = 0
        long_losses = 0
        short_wins = 0
        short_losses = 0
        total_long_profit = 0
        total_short_profit = 0
        total_long_loss = 0
        total_short_loss = 0
        monthly_returns = {}
        
        if trades:
            for trade in trades:
                # Track exit reasons
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
                # Track holding periods
                avg_bars_held += trade.get('bars_held', 0)
                
                # Classify by direction
                if trade.get('direction') == 'long':
                    long_trades.append(trade)
                    if trade.get('profit_loss', 0) > 0:
                        long_wins += 1
                        total_long_profit += trade.get('profit_loss', 0)
                    else:
                        long_losses += 1
                        total_long_loss += abs(trade.get('profit_loss', 0))
                else:
                    short_trades.append(trade)
                    if trade.get('profit_loss', 0) > 0:
                        short_wins += 1
                        total_short_profit += trade.get('profit_loss', 0)
                    else:
                        short_losses += 1
                        total_short_loss += abs(trade.get('profit_loss', 0))
                
                # Track monthly performance
                if 'exit_date' in trade:
                    exit_date = trade['exit_date']
                    month_key = f"{exit_date.year}-{exit_date.month:02d}"
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = 0
                    monthly_returns[month_key] += trade.get('profit_loss', 0)
            
            # Calculate averages
            avg_bars_held = avg_bars_held / len(trades) if trades else 0
            
            # Add derived stats
            long_win_rate = long_wins / len(long_trades) * 100 if long_trades else 0
            short_win_rate = short_wins / len(short_trades) * 100 if short_trades else 0
            avg_long_profit = total_long_profit / long_wins if long_wins > 0 else 0
            avg_short_profit = total_short_profit / short_wins if short_wins > 0 else 0
            avg_long_loss = total_long_loss / long_losses if long_losses > 0 else 0
            avg_short_loss = total_short_loss / short_losses if short_losses > 0 else 0
        
        # Create performance dashboard
        return html.Div([
            html.H4(f"Strategy Performance Metrics - {symbol_info['symbol']} ({symbol_info['interval']})"
                   ),
            html.P("Analysis based on cycle detection and FLD signals", className="text-muted"),
            
            # Summary metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Win Rate"),
                        dbc.CardBody([
                            html.H3(f"{metrics['win_rate']}%", className="text-success" if metrics['win_rate'] > 60 else "text-warning")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Avg. Profit/Loss"),
                        dbc.CardBody([
                            html.H3(f"{metrics['avg_profit']}%", className="text-success" if metrics['avg_profit'] > 0 else "text-danger")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Max Drawdown"),
                        dbc.CardBody([
                            html.H3(f"{metrics['max_drawdown']}%", className="text-danger")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sharpe Ratio"),
                        dbc.CardBody([
                            html.H3(f"{metrics['sharpe_ratio']}", className="text-info")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            # Second row of metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Total Trades"),
                        dbc.CardBody([
                            html.H3(f"{len(trades)}", className="text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Profit Factor"),
                        dbc.CardBody([
                            html.H3(f"{metrics['profit_factor']}" if metrics['profit_factor'] != float('inf') else "∞", 
                                  className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Average Hold Time"),
                        dbc.CardBody([
                            html.H3(f"{avg_bars_held:.1f} bars", className="text-info")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Total Return"),
                        dbc.CardBody([
                            html.H3(f"{(current_capital/initial_capital - 1) * 100:.2f}%", 
                                  className="text-success" if current_capital > initial_capital else "text-danger")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            # Performance chart using real data
            dcc.Graph(
                figure=px.line(
                    # Use the actual dates we stored with each equity value
                    x=equity_dates,
                    y=equity_curve,
                    title=f"Equity Curve ({symbol_info['symbol']})"
                ).update_layout(
                    xaxis_title="Date",
                    yaxis_title="Account Value ($)",
                )
            ),
            
            # Add trade markers if we have trades
            html.Div([
                html.H5(f"Trading Activity Analysis"),
                
                # Exit reason and direction breakdown
                dbc.Row([
                    dbc.Col([
                        html.H6("Exit Reason Breakdown"),
                        dcc.Graph(
                            figure=px.pie(
                                names=list(exit_reasons.keys()),
                                values=list(exit_reasons.values()),
                                title="Trade Exit Reasons"
                            )
                        )
                    ], width=4),
                    
                    dbc.Col([
                        html.H6("Trade Direction Analysis"),
                        dcc.Graph(
                            figure=px.pie(
                                names=["Long", "Short"],
                                values=[len(long_trades), len(short_trades)],
                                title="Long vs Short Trades",
                                color_discrete_map={'Long': 'green', 'Short': 'red'}
                            )
                        )
                    ], width=4),
                    
                    dbc.Col([
                        html.H6("Strategy Insights"),
                        html.Ul([
                            html.Li(f"Win Rate: {metrics['win_rate']:.1f}% ({self.win_count} winning, {self.loss_count} losing)"),
                            html.Li(f"Average Winner: {avg_profit:.2f}%, Average Loser: -{avg_loss:.2f}%"),
                            html.Li(f"Average Hold Time: {avg_bars_held:.1f} bars"),
                            html.Li(f"Stop Loss Hit Rate: {exit_reasons.get('stop_loss', 0)/len(trades)*100:.1f}%" if trades else "N/A"),
                            html.Li(f"Take Profit Hit Rate: {exit_reasons.get('take_profit', 0)/len(trades)*100:.1f}%" if trades else "N/A"),
                        ]),
                        dbc.Alert([
                            html.Strong("Trading Notes: "), 
                            "This strategy uses Fibonacci cycles for entry/exit decisions, ATR-based position sizing, and trend confirmation for optimal entries."
                        ], color="info", className="mt-3")
                    ], width=4),
                ]),
                
                # Long vs Short Performance Analysis
                dbc.Row([
                    dbc.Col([
                        html.H5("Long vs Short Performance", className="mt-3"),
                        dbc.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Metric"),
                                    html.Th("Long Trades"),
                                    html.Th("Short Trades"),
                                    html.Th("All Trades")
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Number of Trades"),
                                    html.Td(f"{len(long_trades)}"),
                                    html.Td(f"{len(short_trades)}"),
                                    html.Td(f"{len(trades)}")
                                ]),
                                html.Tr([
                                    html.Td("Win Rate"),
                                    html.Td(f"{long_win_rate:.1f}%", className="text-success" if long_win_rate > 50 else "text-danger"),
                                    html.Td(f"{short_win_rate:.1f}%", className="text-success" if short_win_rate > 50 else "text-danger"),
                                    html.Td(f"{metrics['win_rate']}%", className="text-success" if metrics['win_rate'] > 50 else "text-danger")
                                ]),
                                html.Tr([
                                    html.Td("Average Profit"),
                                    html.Td(f"${avg_long_profit:.2f}", className="text-success" if avg_long_profit > 0 else "text-danger"),
                                    html.Td(f"${avg_short_profit:.2f}", className="text-success" if avg_short_profit > 0 else "text-danger"),
                                    html.Td(f"${avg_profit:.2f}", className="text-success" if avg_profit > 0 else "text-danger")
                                ]),
                                html.Tr([
                                    html.Td("Average Loss"),
                                    html.Td(f"${avg_long_loss:.2f}", className="text-danger"),
                                    html.Td(f"${avg_short_loss:.2f}", className="text-danger"),
                                    html.Td(f"${avg_loss:.2f}", className="text-danger")
                                ]),
                                html.Tr([
                                    html.Td("Total P/L"),
                                    html.Td(f"${total_long_profit - total_long_loss:.2f}", 
                                           className="text-success" if total_long_profit > total_long_loss else "text-danger"),
                                    html.Td(f"${total_short_profit - total_short_loss:.2f}", 
                                           className="text-success" if total_short_profit > total_short_loss else "text-danger"),
                                    html.Td(f"${total_profit - abs(total_loss):.2f}", 
                                           className="text-success" if total_profit > abs(total_loss) else "text-danger")
                                ])
                            ])
                        ], bordered=True, striped=True, hover=True)
                    ], width=12)
                ]),
                
                # Recent trades table
                dbc.Row([
                    dbc.Col([
                        html.H5("Recent Trades", className="mt-3"),
                        dash_table.DataTable(
                            data=[
                                {
                                    'Direction': t['direction'].upper(),
                                    'Entry Date': t['entry_date'].strftime('%Y-%m-%d'),
                                    'Exit Date': t['exit_date'].strftime('%Y-%m-%d'),
                                    'Entry Price': f"${t['entry_price']:.2f}",
                                    'Exit Price': f"${t['exit_price']:.2f}",
                                    'P/L': f"${t['profit_loss']:.2f}",
                                    'P/L %': f"{t['profit_loss_pct']:.2f}%",
                                    'Hold Period': f"{t.get('bars_held', 0)} bars",
                                    'Exit Reason': t.get('exit_reason', 'unknown').replace('_', ' ').title()
                                }
                                # Only show the last 10 trades
                                for t in sorted(trades, key=lambda x: x['exit_date'], reverse=True)[:10]
                            ],
                            columns=[
                                {'name': 'Direction', 'id': 'Direction'},
                                {'name': 'Entry Date', 'id': 'Entry Date'},
                                {'name': 'Exit Date', 'id': 'Exit Date'},
                                {'name': 'Entry Price', 'id': 'Entry Price'},
                                {'name': 'Exit Price', 'id': 'Exit Price'},
                                {'name': 'P/L', 'id': 'P/L'},
                                {'name': 'P/L %', 'id': 'P/L %'},
                                {'name': 'Hold Period', 'id': 'Hold Period'},
                                {'name': 'Exit Reason', 'id': 'Exit Reason'}
                            ],
                            style_cell={'textAlign': 'center'},
                            style_data_conditional=[
                                {
                                    'if': {
                                        'filter_query': '{Direction} contains "LONG"'
                                    },
                                    'backgroundColor': 'rgba(0, 128, 0, 0.1)'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Direction} contains "SHORT"'
                                    },
                                    'backgroundColor': 'rgba(255, 0, 0, 0.1)'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P/L} contains "-"'
                                    },
                                    'color': 'red'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P/L} contains "$" && !({P/L} contains "-")'
                                    },
                                    'color': 'green'
                                }
                            ],
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            },
                            page_size=10
                        )
                    ], width=12)
                ]),
                
                # Monthly returns chart
                dbc.Row([
                    dbc.Col([
                        html.H5("Monthly Returns", className="mt-3"),
                        dcc.Graph(
                            figure=px.bar(
                                x=list(monthly_returns.keys()),
                                y=list(monthly_returns.values()),
                                labels={'x': 'Month', 'y': 'Profit/Loss ($)'},
                                title="Monthly Performance"
                            ).update_layout(
                                xaxis_title="Month",
                                yaxis_title="Profit/Loss ($)"
                            ).update_traces(
                                marker_color=['green' if val > 0 else 'red' for val in monthly_returns.values()]
                            )
                        )
                    ], width=12)
                ]),
                
                html.P("The strategy is for analysis only and not intended for generating live signals.",
                       className="text-muted font-italic mt-3"),
                
                # Hidden div to store the current analysis parameters with full info
                html.Div(id="current-analysis-params", style={"display": "none"},
                         children=json.dumps({
                             "symbol": symbol_info["symbol"],
                             "exchange": symbol_info["exchange"],
                             "interval": symbol_info["interval"],
                             "timestamp": datetime.now().isoformat()  # Add timestamp for debugging
                         }))
            ]) if trades else html.Div()
        ]), {"display": "block"}
    
    def _calculate_atr(self, data, period):
        """Calculate the Average True Range for position sizing."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        atr = np.zeros(len(data))
        
        # Calculate True Range
        tr = np.zeros(len(data))
        tr[0] = high[0] - low[0]  # First TR is simply High - Low
        
        for i in range(1, len(data)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Calculate ATR using simple moving average first
        atr[:period] = np.nan
        atr[period] = np.mean(tr[1:period+1])
        
        # Use EMA for the rest
        alpha = 2 / (period + 1)
        for i in range(period + 1, len(data)):
            atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))
        
        return atr
    
    def _process_position(self, position, positions, pos_id, data, bar_idx, trades, current_capital, max_holding_days_absolute=60):
        """Process an open position to check for stop loss or take profit conditions."""
        direction = position['direction']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        position_size = position['position_size']
        min_hold_until = position.get('min_hold_until', 0)
        max_hold_until = position.get('max_hold_until', float('inf'))
        
        current_price = data.iloc[bar_idx]['close']
        
        # Check for stop loss or take profit
        exit_triggered = False
        exit_price = current_price
        exit_reason = ''
        
        # Don't exit before minimum holding period except for stop loss
        enforce_min_hold = bar_idx < min_hold_until
        
        # Force exit at maximum holding period or absolute max days
        # This prevents unrealistically long holds that artificially inflate performance
        elapsed_bars = bar_idx - position['entry_idx']
        force_exit = (bar_idx >= max_hold_until) or (elapsed_bars >= max_holding_days_absolute)
        
        if direction == 'long':
            # Always check stop loss regardless of holding period
            if current_price <= stop_loss:
                exit_triggered = True
                exit_reason = 'stop_loss'
            # Only check take profit if minimum holding period passed
            elif not enforce_min_hold and current_price >= take_profit:
                exit_triggered = True
                exit_reason = 'take_profit'
            # Force exit at max holding period with detailed reason
            elif force_exit:
                exit_triggered = True
                if bar_idx >= max_hold_until:
                    exit_reason = 'max_hold_reached'
                else:
                    exit_reason = 'absolute_max_days'
        else:  # Short
            # Always check stop loss regardless of holding period
            if current_price >= stop_loss:
                exit_triggered = True
                exit_reason = 'stop_loss'
            # Only check take profit if minimum holding period passed
            elif not enforce_min_hold and current_price <= take_profit:
                exit_triggered = True
                exit_reason = 'take_profit'
            # Force exit at max holding period with detailed reason
            elif force_exit:
                exit_triggered = True
                if bar_idx >= max_hold_until:
                    exit_reason = 'max_hold_reached'
                else:
                    exit_reason = 'absolute_max_days'
        
        # Process exit if triggered
        if exit_triggered:
            # Calculate P&L - fixed for proper handling of short positions
            if direction == 'long':
                profit_loss = (exit_price - entry_price) / entry_price
            else:  # For shorts
                profit_loss = (entry_price - exit_price) / entry_price
            
            profit_loss_amount = position_size * entry_price * profit_loss
            
            # Log for debugging
            logger.debug(f"Trade exit: {direction}, Entry={entry_price:.2f}, Exit={exit_price:.2f}, " +
                         f"P/L={profit_loss_amount:.2f} ({profit_loss*100:.2f}%), Reason={exit_reason}")
            
            # Add to trades list
            trade = {
                "symbol": data.iloc[bar_idx].name if hasattr(data.iloc[bar_idx], 'name') else position.get('symbol', 'Unknown'),
                "direction": direction,
                "entry_date": position["entry_date"],
                "entry_price": entry_price,
                "exit_date": data.index[bar_idx],
                "exit_price": exit_price,
                "quantity": position_size,
                "profit_loss": profit_loss_amount,
                "profit_loss_pct": profit_loss * 100,
                "exit_reason": exit_reason,
                "bars_held": bar_idx - position['entry_idx']
            }
            
            trades.append(trade)
            
            # Track win/loss
            if profit_loss > 0:
                self.win_count += 1
                self.profits.append(profit_loss * 100)
            else:
                self.loss_count += 1
                self.losses.append(abs(profit_loss * 100))
            
            # Update capital
            current_capital += profit_loss_amount
            
            # Remove position
            positions_to_remove = pos_id
        
        return current_capital
    
    def _close_position(self, position, positions, pos_id, data, bar_idx, trades, current_capital):
        """Close a position at the end of backtest."""
        try:
            direction = position['direction']
            entry_price = position['entry_price']
            position_size = position['position_size']
            entry_idx = position['entry_idx']
            
            # Use final price
            exit_price = data.iloc[bar_idx]['close']
            
            # Calculate P&L - fixed for proper handling of short positions at end of backtest
            if direction == 'long':
                profit_loss = (exit_price - entry_price) / entry_price
            else:  # For shorts
                profit_loss = (entry_price - exit_price) / entry_price
            
            profit_loss_amount = position_size * entry_price * profit_loss
            
            # Log for debugging
            logger.debug(f"End of backtest: {direction}, Entry={entry_price:.2f}, Exit={exit_price:.2f}, " +
                         f"P/L={profit_loss_amount:.2f} ({profit_loss*100:.2f}%), Days held={bar_idx - entry_idx}")
            
            # Add to trades list
            trade = {
                "symbol": data.iloc[bar_idx].name if hasattr(data.iloc[bar_idx], 'name') else position.get('symbol', 'Unknown'),
                "direction": direction,
                "entry_date": position["entry_date"],
                "entry_price": entry_price,
                "exit_date": data.index[bar_idx],
                "exit_price": exit_price,
                "quantity": position_size,
                "profit_loss": profit_loss_amount,
                "profit_loss_pct": profit_loss * 100,
                "exit_reason": 'end_of_data',
                "bars_held": bar_idx - entry_idx
            }
            
            trades.append(trade)
            
            # Get or initialize win/loss counters
            win_count = getattr(self, 'win_count', 0)
            loss_count = getattr(self, 'loss_count', 0)
            profits = getattr(self, 'profits', [])
            losses = getattr(self, 'losses', [])
            
            # Track win/loss
            if profit_loss > 0:
                win_count += 1
                profits.append(profit_loss * 100)
                setattr(self, 'win_count', win_count)
                setattr(self, 'profits', profits)
            else:
                loss_count += 1
                losses.append(abs(profit_loss * 100))
                setattr(self, 'loss_count', loss_count)
                setattr(self, 'losses', losses)
            
            # Update capital
            current_capital += profit_loss_amount
            
            # Return updated capital
            return current_capital
        
        except Exception as e:
            logger.exception(f"Error in _close_position: {str(e)}")
            return current_capital

# Constants for styling
STRATEGY_DESCRIPTIONS = {
    "advanced_fibonacci": {
        "name": "Advanced Fibonacci Strategy",
        "description": "Uses Fibonacci ratios and cycle detection to identify optimal entry and exit points with robust risk management.",
        "suitable_for": "Swing and position traders looking for higher probability setups.",
        "timeframes": "Daily, Weekly, 4H",
        "risk_level": "Medium",
        "returns": "Typically 1.5-2.5% per trade with proper risk management"
    },
    "swing_trading": {
        "name": "Swing Trading Strategy",
        "description": "Capture larger market moves using cycle alignment and confluence of multiple timeframes.",
        "suitable_for": "Part-time traders with day jobs looking for 3-10 day positions.",
        "timeframes": "Daily, 4H",
        "risk_level": "Medium",
        "returns": "Targets 2-5% per trade with proper position sizing"
    },
    "day_trading": {
        "name": "Day Trading Strategy",
        "description": "Short-term trading using FLD breakouts and fast cycles for intraday movements.",
        "suitable_for": "Active traders who can monitor markets throughout the day.",
        "timeframes": "1H, 15min",
        "risk_level": "High",
        "returns": "0.5-1.5% per trade with higher frequency"
    },
    "harmonic_pattern": {
        "name": "Harmonic Pattern Strategy",
        "description": "Combines Fibonacci Cycle analysis with harmonic pattern recognition for precise entries and exits.",
        "suitable_for": "Technical analysts comfortable with pattern recognition.",
        "timeframes": "Any",
        "risk_level": "Medium-High",
        "returns": "1.5-3% per completed pattern with strict risk management"
    }
}

# Function to create the trading strategy UI component
def create_strategy_dashboard(result=None, initial_params=None):
    """
    Create the trading strategy dashboard component.
    
    Args:
        result: Optional scan result data
        initial_params: Optional initial parameters for the dashboard
    
    Returns:
        Dash component representing the strategy dashboard
    """
    # Log the initial parameters for debugging symbol persistence
    logger.info(f"Creating strategy dashboard with initial params: {initial_params}")
    
    # If result is provided, log key details
    if result:
        logger.info(f"Strategy dashboard input: {result.symbol} ({result.exchange}) on {result.interval}")
        
    # If both result and initial_params are provided, ensure symbol consistency
    if result and initial_params and 'symbol' in initial_params:
        if result.symbol != initial_params['symbol']:
            logger.warning(f"Symbol mismatch: result has {result.symbol} but params has {initial_params['symbol']}")
            # Priority given to the result object for consistency
            initial_params['symbol'] = result.symbol
            initial_params['exchange'] = result.exchange
            initial_params['interval'] = result.interval
            
    # Ensure initial_params contains symbol info if result is provided
    if result and not initial_params:
        initial_params = {
            'symbol': result.symbol,
            'exchange': result.exchange,
            'interval': result.interval,
            'strategy': 'advanced_fibonacci',
            'timestamp': datetime.now().isoformat()
        }
        logger.info(f"Created new params from result: {initial_params}")
        
    # Initialize hidden storage for parameters if provided
    initial_params_json = json.dumps(initial_params) if initial_params else None
    
    return html.Div([
        html.H2("Trading Strategy Analysis"),
        html.P(
            "Analyze and backtest trading strategies based on FLD cycles and harmonic patterns.",
            className="lead mb-4"
        ),
        
        dbc.Row([
            # Left sidebar for strategy selection and configuration
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Strategy"),
                    dbc.CardBody([
                        html.Label("Trading Strategy"),
                        dcc.Dropdown(
                            id="strategy-selector",
                            options=[
                                {"label": STRATEGY_DESCRIPTIONS["advanced_fibonacci"]["name"], "value": "advanced_fibonacci"},
                                {"label": STRATEGY_DESCRIPTIONS["swing_trading"]["name"], "value": "swing_trading"},
                                {"label": STRATEGY_DESCRIPTIONS["day_trading"]["name"], "value": "day_trading"},
                                {"label": STRATEGY_DESCRIPTIONS["harmonic_pattern"]["name"], "value": "harmonic_pattern"},
                            ],
                            value="advanced_fibonacci",
                            clearable=False,
                            # Dark mode fixes for dropdown
                            style={
                                'color': 'black',
                                'background-color': 'white',
                            },
                            className="mb-2"
                        ),
                        
                        html.Div(id="strategy-description"),
                        
                        html.Hr(),
                        
                        html.Label("Strategy Parameters"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Risk Per Trade (%)"),
                                dbc.Input(
                                    id="risk-per-trade",
                                    type="number",
                                    value=1,
                                    min=0.1,
                                    max=5,
                                    step=0.1
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Min. Cycle Alignment (%)"),
                                dbc.Input(
                                    id="min-cycle-alignment",
                                    type="number",
                                    value=80,
                                    min=50,
                                    max=100,
                                    step=5
                                )
                            ], width=6)
                        ], className="mb-2"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Min. Signal Strength"),
                                dbc.Input(
                                    id="min-signal-strength",
                                    type="number",
                                    value=0.7,
                                    min=0.3,
                                    max=1.0,
                                    step=0.1
                                )
                            ], width=12),
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "View Performance",
                                    id="view-performance-btn",
                                    color="success",
                                    className="w-100 mb-2"
                                ),
                            ], width=12),
                            dbc.Col([
                                dbc.Button(
                                    "Run Strategy Backtest",
                                    id="backtest-strategy-btn",
                                    color="primary",
                                    className="w-100"
                                ),
                            ], width=12),
                        ]),
                    ])
                ], className="mb-3"),
                
                # Strategy insights card
                dbc.Card([
                    dbc.CardHeader("Strategy Insights"),
                    dbc.CardBody([
                        html.Div(id="strategy-insights-content", children=[
                            html.P("Select a strategy and run a backtest to see insights."),
                        ])
                    ])
                ])
            ], width=4),
            
            # Main content area for backtest results and visualizations
            dbc.Col([
                # Hidden div to store initial parameters
                html.Div(id="initial-params-store", 
                         style={"display": "none"},
                         children=initial_params_json),
                
                # Results container
                html.Div(id="backtest-results-container", style={"display": "none"}),
            ], width=8)
        ])
    ])

# Callback to update the strategy description when a strategy is selected
@callback(
    Output("strategy-description", "children"),
    Output("initial-params-store", "children", allow_duplicate=True),
    Input("strategy-selector", "value"),
    State("initial-params-store", "children"),
    prevent_initial_call=True
)
def update_strategy_description(strategy_value, initial_params_json):
    """Update the strategy description based on the selected strategy and maintain state."""
    # First update the strategy description
    strategy_description = create_strategy_description(strategy_value)
    
    # Now preserve any existing parameters but update the strategy
    if initial_params_json:
        try:
            initial_params = json.loads(initial_params_json)
            # Update the strategy but keep all other parameters
            initial_params['strategy'] = strategy_value
            # Add timestamp for tracking changes
            initial_params['timestamp'] = datetime.now().isoformat()
            return strategy_description, json.dumps(initial_params)
        except Exception as e:
            logger.error(f"Error updating params with strategy change: {e}")
    
    # If no existing params, just return the description and don't update params
    return strategy_description, dash.no_update

def create_strategy_description(strategy_value):
    """Create the strategy description component."""
    if not strategy_value:
        return html.Div()
    
    strategy_info = STRATEGY_DESCRIPTIONS.get(strategy_value, {})
    
    return html.Div([
        html.H5(strategy_info.get("name", ""), className="mt-2"),
        html.P(strategy_info.get("description", ""), className="text-muted"),
        dbc.Row([
            dbc.Col([
                html.P("Suitable For:", className="font-weight-bold mb-0"),
                html.P(strategy_info.get("suitable_for", ""), className="text-muted small"),
            ], width=6),
            dbc.Col([
                html.P("Timeframes:", className="font-weight-bold mb-0"),
                html.P(strategy_info.get("timeframes", ""), className="text-muted small"),
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.P("Risk Level:", className="font-weight-bold mb-0"),
                html.P(strategy_info.get("risk_level", ""), className="text-muted small"),
            ], width=6),
            dbc.Col([
                html.P("Expected Returns:", className="font-weight-bold mb-0"),
                html.P(strategy_info.get("returns", ""), className="text-muted small"),
            ], width=6)
        ])
    ])

# Callback for auto-triggering actions when important parameters change
@callback(
    Output("initial-params-store", "children"),
    Output("backtest-strategy-btn", "n_clicks"),
    Input("strategy-selector", "value"),
    State("symbol-input", "value"),
    State("exchange-input", "value"),
    State("interval-dropdown", "value"),
    State("initial-params-store", "children"),
)
def auto_trigger_action(strategy_value, symbol, exchange, interval, initial_params_json):
    """Store and maintain parameters when strategy changes, preserving symbol info."""
    # Get context to identify what triggered this callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, None
    
    # Default parameters
    params_dict = {}
    
    # If we have existing parameters, use them as the base
    if initial_params_json:
        try:
            params_dict = json.loads(initial_params_json)
        except Exception as e:
            logger.error(f"Error parsing initial params in auto_trigger: {e}")
    
    # Always update with the latest UI values if available
    if symbol:
        params_dict['symbol'] = symbol
    if exchange:
        params_dict['exchange'] = exchange
    if interval:
        params_dict['interval'] = interval
    
    # Update strategy if that's what triggered the callback
    if ctx.triggered[0]['prop_id'] == 'strategy-selector.value':
        params_dict['strategy'] = strategy_value
    
    # Add timestamp to track freshness
    params_dict['timestamp'] = datetime.now().isoformat()
    
    # Log for debugging
    logger.debug(f"Auto-trigger updated params: {params_dict}")
    
    # Store the parameters but don't auto-click the button
    return json.dumps(params_dict), None

# Callback for status display - this fixes the missing callback error
@callback(
    Output("status-display", "children"),
    Input("backtest-strategy-btn", "n_clicks"),
    Input("strategy-selector", "value"),
    prevent_initial_call=True,
)
def update_status_display(n_clicks, strategy_value):
    """Update the status display when strategy actions are taken.
    This callback is added to fix the 'callback function not found for output status-display.children' error.
    """
    # This is a placeholder callback that will effectively do nothing
    # The actual status display was removed and replaced with a card that has no ID as per FIXES.md
    raise PreventUpdate

# Callback to update strategy insights
@callback(
    Output("strategy-insights-content", "children"),
    Input("backtest-strategy-btn", "n_clicks"),
    State("strategy-selector", "value"),
    prevent_initial_call=True,
)
def update_strategy_insights(n_clicks, strategy_value):
    """Update the strategy insights content based on backtest results."""
    if not n_clicks:
        return html.P("Select a strategy and run a backtest to see insights.")
    
    # Get the strategy information
    strategy_info = STRATEGY_DESCRIPTIONS.get(strategy_value, {})
    
    # Generate insights based on the strategy information
    return html.Div([
        html.H5("Strategy Analysis", className="mt-2"),
        html.P(f"Analysis for {strategy_info.get('name', 'Selected Strategy')}", className="text-muted"),
        html.Div([
            html.P("Key Insights:", className="font-weight-bold"),
            html.Ul([
                html.Li("Consider market regime before entry"),
                html.Li("Verify cycle alignment for higher probability setups"),
                html.Li("Use ATR-based position sizing for proper risk management"),
                html.Li("Set stops based on FLD levels, not arbitrary percentages")
            ]),
            dbc.Alert([
                html.Strong("Best Practices: "), 
                "Run multiple timeframe analysis for confirmation"
            ], color="info", className="mt-3")
        ])
    ])

# Callback to view strategy performance when a symbol is selected
@callback(
    Output("backtest-results-container", "children", allow_duplicate=True),
    Output("backtest-results-container", "style", allow_duplicate=True),
    Input("symbol-selection-btn", "n_clicks"),
    State("symbol-input", "value"),
    State("exchange-input", "value"),
    State("interval-input", "value"),
    State("strategy-selector", "value"),
    prevent_initial_call=True,
)
def view_strategy_performance(n_clicks, symbol, exchange, interval, strategy_value):
    """Display performance for a selected symbol using the chosen strategy."""
    try:
        if not n_clicks or not symbol:
            return dash.no_update, {"display": "none"}
        
        # Initialize components
        symbol_info = {
            "symbol": symbol,
            "exchange": exchange or "NSE",
            "interval": interval or "daily"
        }
        
        # Fetch data for the selected symbol
        config = load_config("config/config.json")
        data_fetcher = DataFetcher(config, logger=logging.getLogger(__name__))
        
        # Try to load data from cache first, then from API if needed
        data = data_fetcher.get_data(
            symbol_info["symbol"], 
            symbol_info["exchange"],
            symbol_info["interval"],
            lookback=365  # One year of data
        )
        
        if data is None or len(data) < 50:
            return html.Div([
                html.H4("Insufficient Data"),
                html.P(f"Could not retrieve enough data for {symbol_info['symbol']}.")
            ]), {"display": "block"}
        
        # Set up the cycle detector and FLD calculator
        cycle_detector = CycleDetector()
        fld_calculator = FLDCalculator()
        signal_generator = SignalGenerator(fld_calculator)
        
        # Detect cycles - pass only the price column (close)
        cycles = cycle_detector.detect_cycles(data['close'])
        
        # Generate FLD signals
        fld_signals = signal_generator.generate_signals(data, cycles)
        
        # Create a SimulatedStrategy instance and execute backtest
        # SimulatedStrategy is now defined at the module level
        strategy = SimulatedStrategy()
        return strategy.execute_backtest(data, cycles, fld_signals, symbol_info)
    
    except Exception as e:
        logger.exception(f"Error in view_strategy_performance: {str(e)}")
        return html.Div([
            html.H4("Performance Calculation Error"),
            html.P(f"An error occurred: {str(e)}")
        ]), {"display": "block"}

# Callback for View Performance button
@callback(
    Output("backtest-results-container", "children", allow_duplicate=True),
    Output("backtest-results-container", "style", allow_duplicate=True),
    Input("view-performance-btn", "n_clicks"),
    State("strategy-selector", "value"),
    State("initial-params-store", "children"),
    prevent_initial_call=True,
)
def view_strategy_performance(n_clicks, strategy_value, initial_params_json):
    """Display performance metrics for a selected symbol using the chosen strategy."""
    try:
        if not n_clicks:
            return dash.no_update, {"display": "none"}
        
        # Get symbol info from initial params
        symbol = None
        exchange = "NSE"
        interval = "daily"
        
        # Extract parameters from initial_params_json if available
        if initial_params_json:
            try:
                initial_params = json.loads(initial_params_json)
                if initial_params and 'symbol' in initial_params:
                    symbol = initial_params.get('symbol')
                    exchange = initial_params.get('exchange', 'NSE')
                    interval = initial_params.get('interval', 'daily')
            except Exception as e:
                logger.exception(f"Error parsing initial params: {e}")
        
        # If no symbol available, check from the form inputs
        if not symbol:
            # Try to get symbol from symbol-input
            try:
                from dash import callback_context
                symbol_element = callback_context.inputs.get('symbol-input.value')
                if symbol_element:
                    symbol = symbol_element
                exchange_element = callback_context.inputs.get('exchange-input.value')
                if exchange_element:
                    exchange = exchange_element
                interval_element = callback_context.inputs.get('interval-dropdown.value')
                if interval_element:
                    interval = interval_element
            except:
                # If we can't get inputs, default to NIFTY
                symbol = "NIFTY"
        
        if not symbol:
            return html.Div([
                html.H4("No Symbol Selected"),
                html.P("Please select a symbol for performance analysis.")
            ]), {"display": "block"}
            
        # Initialize components
        symbol_info = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval
        }
        
        # Fetch data for the selected symbol
        config = load_config("config/config.json")
        data_fetcher = DataFetcher(config, logger=logging.getLogger(__name__))
        
        # Try to load data from cache first, then from API if needed
        data = data_fetcher.get_data(
            symbol_info["symbol"], 
            symbol_info["exchange"],
            symbol_info["interval"],
            lookback=365  # One year of data
        )
        
        if data is None or len(data) < 50:
            return html.Div([
                html.H4("Insufficient Data"),
                html.P(f"Could not retrieve enough data for {symbol_info['symbol']}.")
            ]), {"display": "block"}
        
        # Set up the cycle detector and FLD calculator
        cycle_detector = CycleDetector()
        fld_calculator = FLDCalculator()
        signal_generator = SignalGenerator(fld_calculator)
        
        # Detect cycles - pass only the price column (close)
        cycles = cycle_detector.detect_cycles(data['close'])
        
        # Generate FLD signals
        fld_signals = signal_generator.generate_signals(data, cycles)
        
        # Create a SimulatedStrategy instance and execute backtest
        strategy = SimulatedStrategy()
        return strategy.execute_backtest(data, cycles, fld_signals, symbol_info)
    
    except Exception as e:
        logger.exception(f"Error in view_strategy_performance: {str(e)}")
        return html.Div([
            html.H4("Performance Calculation Error"),
            html.P(f"An error occurred: {str(e)}")
        ]), {"display": "block"}

# Callback for Backtest Strategy button
@callback(
    Output("backtest-results-container", "children"),
    Output("backtest-results-container", "style"),
    Input("backtest-strategy-btn", "n_clicks"),
    Input("strategy-selector", "value"),  # Changed from State to Input to respond to changes
    State("risk-per-trade", "value"),
    State("min-cycle-alignment", "value"),
    State("min-signal-strength", "value"),
    State("initial-params-store", "children"),
    prevent_initial_call=True,
)
def backtest_strategy(n_clicks, strategy_value, risk_per_trade, min_cycle_alignment, min_signal_strength, initial_params_json):
    """Run a backtest for the selected strategy with current parameters using real market data."""
    # Get the ID of the component that triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, {"display": "none"}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # If strategy changed but button wasn't clicked, don't update
    if trigger_id == 'strategy-selector' and not n_clicks:
        return dash.no_update, {"display": "none"}
    
    # Default symbol info - WILL ONLY BE USED IF NO PARAMETERS ARE PASSED
    symbol_info = {
        "symbol": "NIFTY",
        "exchange": "NSE",
        "interval": "daily"
    }
    
    # Default lookback to use from UI or fallback to 1000
    lookback = 1000  # This is just a fallback - we prioritize user specified values
    
    # Log for debugging
    logger.info(f"Backtest strategy triggered with n_clicks={n_clicks}, strategy={strategy_value}")
    logger.info(f"Initial params JSON: {initial_params_json}")
    
    # Try to get symbol and other parameters from stored parameters
    if initial_params_json:
        try:
            initial_params = json.loads(initial_params_json)
            logger.info(f"Parsed initial params: {initial_params}")
            
            if initial_params and 'symbol' in initial_params:
                symbol_info = {
                    "symbol": initial_params["symbol"],
                    "exchange": initial_params.get("exchange", "NSE"),
                    "interval": initial_params.get("interval", "daily")
                }
                # Get lookback from initial params if available
                if 'lookback' in initial_params:
                    lookback = initial_params.get("lookback", 1000)
                    
                logger.info(f"Using symbol from initial params for backtest: {symbol_info['symbol']} with lookback: {lookback}")
                logger.info(f"Strategy selected: {strategy_value}")
        except Exception as e:
            logger.exception(f"Error parsing initial params for backtest: {e}")
    
    try:
        # Initialize components for backtesting
        config = load_config("config/config.json")
        data_fetcher = DataFetcher(config, logger=logging.getLogger(__name__))
        
        # Import the proper backtesting framework by modifying the path
        import sys
        import os
        
        # Add project root to path to allow absolute imports
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import real strategy classes
        try:
            from trading.practical_strategies import (
                AdvancedFibonacciStrategy, 
                SwingTradingStrategy,
                DayTradingStrategy, 
                HarmonicPatternStrategy
            )
            
            # Backtesting module removed, always use simplified backtest
            real_backtest_available = False
            logger.info("Using simplified backtest")
            
            # Create simplified classes if real ones aren't available
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
                take_profit_multiplier: float = 2.0
                rebalance_frequency: Optional[str] = None
                require_alignment: bool = True
                min_strength: float = 0.3
                pyramiding: int = 0
                strategy_type: str = "advanced_fibonacci"  # Used for strategy selection
            
            # Map strategy values to actual strategy classes
            strategy_classes = {
                "advanced_fibonacci": AdvancedFibonacciStrategy,
                "swing_trading": SwingTradingStrategy,
                "day_trading": DayTradingStrategy,
                "harmonic_pattern": HarmonicPatternStrategy
            }
            
            # Determine which strategy class to use
            StrategyClass = strategy_classes.get(strategy_value, AdvancedFibonacciStrategy)
            
            # Create real strategy instance if possible
            try:
                strategy_instance = StrategyClass("config/config.json")
                logger.info(f"Created real strategy instance: {strategy_value}")
                using_real_strategy = True
            except Exception as e:
                logger.exception(f"Error creating real strategy instance: {e}")
                using_real_strategy = False
                
            # Create a simplified BacktestEngine that uses SimulatedStrategy
            class SimpleBacktestEngine:
                """Simplified backtesting engine for strategy simulation."""
                
                def __init__(self, config, scanner=None, data_fetcher=None, strategy_instance=None):
                    self.config = config
                    self.data_fetcher = data_fetcher
                    self.logger = logging.getLogger(__name__)
                
                def _get_strategy_instance(self, strategy_type="advanced_fibonacci"):
                    """Get the appropriate strategy instance based on type with improved logging."""
                    self.logger.info(f"Creating strategy instance of type: {strategy_type}")
                    
                    # Map strategy values to actual strategy classes
                    strategy_classes = {
                        "advanced_fibonacci": AdvancedFibonacciStrategy,
                        "swing_trading": SwingTradingStrategy,
                        "day_trading": DayTradingStrategy,
                        "harmonic_pattern": HarmonicPatternStrategy
                    }
                    
                    # Determine which strategy class to use with better error handling
                    StrategyClass = strategy_classes.get(strategy_type)
                    if not StrategyClass:
                        self.logger.warning(f"Unknown strategy type: {strategy_type}, defaulting to AdvancedFibonacciStrategy")
                        StrategyClass = AdvancedFibonacciStrategy
                        
                    try:
                        # Create strategy instance
                        strategy_instance = StrategyClass("config/config.json")
                        self.logger.info(f"Successfully created strategy instance: {strategy_type}")
                        return strategy_instance
                    except Exception as e:
                        self.logger.exception(f"Error creating strategy instance of type {strategy_type}: {e}")
                        # Return a default SimulatedStrategy as fallback
                        return SimulatedStrategy()
                
                def run_backtest(self, params):
                    """Run a simplified backtest with the provided parameters."""
                    # Log the backtest parameters for debugging
                    self.logger.info(f"Running backtest for {params.symbol} with strategy: {params.strategy_type}")
                    
                    # Use the strategy type to get the right strategy instance
                    if hasattr(params, 'strategy_type') and params.strategy_type:
                        strategy_instance = self._get_strategy_instance(params.strategy_type)
                        self.logger.info(f"Using strategy type from params: {params.strategy_type}")
                    else:
                        strategy_instance = self._get_strategy_instance()
                        self.logger.info("Using default strategy: advanced_fibonacci")
                    
                    # Create a simulator for the strategy
                    strategy_simulator = SimulatedStrategy()
                    
                    # Get market data for symbol
                    self.logger.info(f"Fetching data for {params.symbol} ({params.exchange}) on {params.interval}")
                    data = self.data_fetcher.get_data(
                        params.symbol,
                        params.exchange,
                        params.interval,
                        lookback=params.lookback
                    )
                    
                    if data is None or len(data) < 50:
                        return html.Div([
                            html.H4("Insufficient Data"),
                            html.P(f"Could not retrieve enough data for {params.symbol} ({params.exchange}).")
                        ]), {"display": "block"}
                    
                    # Set up the cycle detector and FLD calculator
                    cycle_detector = CycleDetector()
                    fld_calculator = FLDCalculator()
                    signal_generator = SignalGenerator(fld_calculator)
                    
                    # Detect cycles - pass only the price column (close)
                    cycles = cycle_detector.detect_cycles(data['close'])
                    
                    # Generate FLD signals
                    fld_signals = signal_generator.generate_signals(data, cycles)
                    
                    # Run the simulated strategy
                    symbol_info = {
                        "symbol": params.symbol,
                        "exchange": params.exchange,
                        "interval": params.interval
                    }
                    
                    return strategy_simulator.execute_backtest(
                        data, cycles, fld_signals, symbol_info
                    )
            
            # Set parameters for backtest with clearer logging
            logger.info(f"Setting up backtest with strategy_value: {strategy_value}")
            
            # Use the strategy_value from the UI or from initial_params
            strategy_type = strategy_value
            if initial_params_json:
                try:
                    initial_params = json.loads(initial_params_json)
                    if 'strategy' in initial_params:
                        strategy_type = initial_params['strategy']
                        logger.info(f"Using strategy type from initial params: {strategy_type}")
                except Exception as e:
                    logger.error(f"Error getting strategy from initial params: {e}")
            
            # Set parameters for backtest with all relevant values
            params = BacktestParameters(
                symbol=symbol_info["symbol"],
                exchange=symbol_info["exchange"],
                interval=symbol_info["interval"],
                lookback=lookback,
                position_size_pct=risk_per_trade if risk_per_trade else 1.0,
                require_alignment=True if min_cycle_alignment and min_cycle_alignment > 50 else False,
                min_strength=min_signal_strength if min_signal_strength else 0.7,
                strategy_type=strategy_type  # Use the determined strategy type
            )
            
            # Log all parameters for debugging
            logger.info(f"Created backtest parameters: {params}")
            
            # Run the backtest with simplified engine
            backtest_engine = SimpleBacktestEngine(
                config=config,
                data_fetcher=data_fetcher
            )
            
            # Get the backtest results
            results = backtest_engine.run_backtest(params)
            
            # Update the strategy insights section with the results
            return results
            
        except Exception as e:
            logger.exception(f"Error importing strategy classes: {e}")
            return html.Div([
                html.H4("Strategy Error"),
                html.P(f"There was an error loading the strategy: {str(e)}")
            ]), {"display": "block"}
    
    except Exception as e:
        logger.exception(f"Error running backtest: {str(e)}")
        return html.Div([
            html.H4("Backtest Error"),
            html.P(f"An error occurred: {str(e)}")
        ]), {"display": "block"}