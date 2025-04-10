import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import time
import json
import matplotlib.pyplot as plt
import os
import uuid

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use absolute imports
from core.fld_signal_generator import FLDCalculator
from core.scanner import FibCycleScanner

# ML imports - use conditionals to handle if they don't exist
try:
    from ml.ml_enhancements import MLSignalEnhancer, MarketRegimeClassifier, EnsembleSignalGenerator, AnomalyDetector
except ImportError:
    # Create stub classes if they don't exist
    class MLSignalEnhancer: pass
    class MarketRegimeClassifier: pass
    class EnsembleSignalGenerator: pass
    class AnomalyDetector: pass

try:    
    from integration.broker_integration import BrokerManager
except ImportError:
    class BrokerManager: pass


class TradingStrategy:
    """
    Base class for trading strategies using Fibonacci Harmonic Trading System.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trading strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Trading parameters
        self.risk_per_trade = self.config.get('risk_per_trade', 1.0)  # % of account
        self.max_open_positions = self.config.get('max_open_positions', 5)
        self.max_risk_per_symbol = self.config.get('max_risk_per_symbol', 2.0)  # % of account
        self.default_stop_pct = self.config.get('default_stop_pct', 2.0)  # % below entry
        
        # State tracking
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        self.signals = {}
        
        # Performance metrics
        self.equity_curve = []
        self.metrics = {}
        
        # Component initialization
        self._init_components()
    
    def _init_components(self):
        """Initialize strategy components."""
        # Initialize scanner for cycle analysis
        scanner_config = self.config.get('scanner_config', {})
        self.scanner = FibCycleScanner(scanner_config)
        
        # Initialize ML components if configured
        ml_config = self.config.get('ml_config', {})
        self.use_ml = ml_config.get('enabled', False)
        
        if self.use_ml:
            self.ensemble = EnsembleSignalGenerator(ml_config)
        else:
            self.ensemble = None
        
        # Initialize broker connection if configured
        broker_config = self.config.get('broker_config', {})
        self.use_broker = broker_config.get('enabled', False)
        
        if self.use_broker:
            self.broker_manager = BrokerManager(broker_config)
        else:
            self.broker_manager = None
    
    def load_models(self, model_dir: str) -> bool:
        """
        Load ML models from directory.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Boolean indicating success
        """
        if not self.use_ml or self.ensemble is None:
            self.logger.warning("ML enhancement not enabled, skipping model loading")
            return False
        
        try:
            self.ensemble.load_models(model_dir)
            self.logger.info(f"Successfully loaded models from {model_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def connect_broker(self) -> bool:
        """
        Connect to configured broker.
        
        Returns:
            Boolean indicating connection success
        """
        if not self.use_broker or self.broker_manager is None:
            self.logger.warning("Broker integration not enabled")
            return False
        
        try:
            # Connect to default broker
            result = self.broker_manager.connect_broker()
            
            if result:
                self.logger.info("Connected to broker successfully")
                
                # Get current positions
                broker = self.broker_manager.get_broker()
                self.positions = broker.get_positions()
                
                return True
            else:
                self.logger.error("Failed to connect to broker")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to broker: {e}")
            return False
    
    def analyze_symbol(self, 
                      symbol: str, 
                      exchange: str = 'NSE',
                      interval: str = 'daily') -> Dict:
        """
        Analyze a symbol to generate trading signals.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code
            interval: Time interval
            
        Returns:
            Dictionary with analysis results
        """
        # Create scan parameters
        params = {
            'symbol': symbol,
            'exchange': exchange,
            'interval': interval,
            'lookback': self.config.get('lookback', 1000),
            'num_cycles': self.config.get('num_cycles', 3),
            'price_source': self.config.get('price_source', 'hlc3'),
            'generate_chart': True
        }
        
        # Analyze symbol
        scan_result = self.scanner.analyze_symbol(params)
        
        if not scan_result.success:
            self.logger.error(f"Error analyzing {symbol}: {scan_result.error}")
            return {'success': False, 'error': scan_result.error}
        
        # Apply ML enhancement if enabled
        if self.use_ml and self.ensemble is not None:
            # Get feature data for ML analysis
            features = self._prepare_features_for_ml(scan_result)
            
            if features is not None:
                # Enhance signal with ML and regime awareness
                enhanced_result = self.ensemble.generate_signal(scan_result, features)
                scan_result = enhanced_result
        
        # Store signal
        self.signals[symbol] = {
            'timestamp': datetime.now(),
            'result': scan_result
        }
        
        return scan_result
    
    def _prepare_features_for_ml(self, scan_result: Dict) -> Optional[pd.DataFrame]:
        """
        Prepare features for ML prediction.
        
        Args:
            scan_result: Scan result dictionary
            
        Returns:
            DataFrame with features for ML or None if error
        """
        try:
            # Extract relevant data from scan result
            symbol = scan_result.get('symbol')
            exchange = scan_result.get('exchange')
            interval = scan_result.get('interval')
            
            # Fetch data
            data = self.scanner.data_fetcher.get_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                lookback=500  # Sufficient for feature calculation
            )
            
            if data is None or data.empty:
                self.logger.error(f"No data fetched for {symbol}")
                return None
            
            # Use feature engineering to create features
            from ..ml_enhancements.feature_engineer import FeatureEngineer
            
            feature_engineer = FeatureEngineer()
            features = feature_engineer.create_technical_features(data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features for ML: {e}")
            return None
    
    def evaluate_position(self, 
                        symbol: str,
                        position_data: Dict) -> Dict:
        """
        Evaluate an existing position against current analysis.
        
        Args:
            symbol: Symbol to evaluate
            position_data: Position data dictionary
            
        Returns:
            Position evaluation result
        """
        # Analyze current state
        result = self.analyze_symbol(symbol)
        
        if not result.get('success', False):
            return {
                'symbol': symbol,
                'action': 'hold',
                'reason': 'analysis_failed'
            }
        
        # Get signal information
        signal = result.get('signal', {})
        signal_type = signal.get('signal', 'neutral')
        position_side = position_data.get('direction', 'long')
        
        # Default to hold
        action = 'hold'
        reason = 'default'
        
        # Check for exit signals
        if position_side == 'long' and 'sell' in signal_type:
            action = 'exit'
            reason = 'reverse_signal'
        elif position_side == 'short' and 'buy' in signal_type:
            action = 'exit'
            reason = 'reverse_signal'
        
        # Check for trailing stop adjustment
        position_guidance = result.get('position_guidance', {})
        current_stop = position_data.get('stop_loss')
        suggested_stop = position_guidance.get('stop_loss')
        
        if suggested_stop is not None and current_stop is not None:
            # For long positions, only move stop up
            if position_side == 'long' and suggested_stop > current_stop:
                action = 'adjust_stop'
                reason = 'raise_stop'
            
            # For short positions, only move stop down
            elif position_side == 'short' and suggested_stop < current_stop:
                action = 'adjust_stop'
                reason = 'lower_stop'
        
        # Check for target reached
        current_price = result.get('price', 0)
        take_profit = position_data.get('take_profit')
        
        if take_profit is not None:
            if (position_side == 'long' and current_price >= take_profit) or \
               (position_side == 'short' and current_price <= take_profit):
                action = 'exit'
                reason = 'target_reached'
        
        # Create evaluation result
        evaluation = {
            'symbol': symbol,
            'action': action,
            'reason': reason,
            'position_side': position_side,
            'current_price': current_price,
            'current_stop': current_stop,
            'suggested_stop': suggested_stop,
            'take_profit': take_profit,
            'signal': signal_type,
            'signal_strength': signal.get('strength', 0),
            'timestamp': datetime.now()
        }
        
        return evaluation
    
    def calculate_position_size(self, 
                              account_value: float, 
                              entry_price: float, 
                              stop_price: float, 
                              risk_pct: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate position size based on risk percentage.
        
        Args:
            account_value: Current account value
            entry_price: Entry price
            stop_price: Stop loss price
            risk_pct: Risk percentage (default to strategy setting if None)
            
        Returns:
            Tuple of (quantity, risk_amount)
        """
        # Use default risk if not specified
        if risk_pct is None:
            risk_pct = self.risk_per_trade
        
        # Calculate risk amount
        risk_amount = account_value * (risk_pct / 100)
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share <= 0:
            self.logger.warning(f"Invalid risk per share: {risk_per_share}. Using default.")
            # Use default stop percentage
            risk_per_share = entry_price * (self.default_stop_pct / 100)
        
        # Calculate quantity
        quantity = risk_amount / risk_per_share
        
        # Round down to whole number
        quantity = np.floor(quantity)
        
        return quantity, risk_amount
    
    def execute_trade_signal(self, 
                           symbol: str, 
                           side: str, 
                           entry_price: float, 
                           stop_price: float, 
                           target_price: float, 
                           risk_pct: Optional[float] = None) -> Dict:
        """
        Execute a trade signal through connected broker.
        
        Args:
            symbol: Symbol to trade
            side: Trade side ('BUY' or 'SELL')
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Take profit price
            risk_pct: Risk percentage (default to strategy setting if None)
            
        Returns:
            Dictionary with trade execution result
        """
        if not self.use_broker or self.broker_manager is None:
            self.logger.warning("Broker integration not enabled, simulating trade")
            
            # Simulate trade
            trade_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            trade = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'risk_pct': risk_pct or self.risk_per_trade,
                'status': 'simulated',
                'timestamp': timestamp,
                'message': "Trade simulated (no broker connection)"
            }
            
            # Add to trade history
            self.trade_history.append(trade)
            
            return {
                'success': True,
                'trade': trade,
                'message': "Trade simulated (no broker connection)"
            }
        
        try:
            # Get broker
            broker = self.broker_manager.get_broker()
            
            # Get account information
            account_info = broker.get_account_info()
            account_value = float(account_info.get('portfolio_value', 0))
            
            if account_value <= 0:
                return {
                    'success': False,
                    'message': "Invalid account value"
                }
            
            # Calculate position size
            quantity, risk_amount = self.calculate_position_size(
                account_value, 
                entry_price, 
                stop_price, 
                risk_pct
            )
            
            if quantity <= 0:
                return {
                    'success': False,
                    'message': "Invalid position size calculated"
                }
            
            # Place entry order
            entry_result = broker.place_order(
                symbol=symbol,
                side=side,
                order_type='LIMIT',
                quantity=quantity,
                price=entry_price,
                time_in_force='DAY'
            )
            
            # Check for error
            if 'error' in entry_result:
                return {
                    'success': False,
                    'message': f"Error placing entry order: {entry_result['error']}"
                }
            
            entry_order_id = entry_result.get('order_id')
            
            # Place stop loss order
            stop_side = 'SELL' if side == 'BUY' else 'BUY'
            
            stop_result = broker.place_order(
                symbol=symbol,
                side=stop_side,
                order_type='STOP',
                quantity=quantity,
                stop_price=stop_price,
                time_in_force='GTC',
                parent_id=entry_order_id  # Link to entry order
            )
            
            # Place take profit order
            take_profit_result = broker.place_order(
                symbol=symbol,
                side=stop_side,
                order_type='LIMIT',
                quantity=quantity,
                price=target_price,
                time_in_force='GTC',
                parent_id=entry_order_id  # Link to entry order
            )
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'quantity': quantity,
                'risk_amount': risk_amount,
                'risk_pct': risk_pct or self.risk_per_trade,
                'entry_order_id': entry_order_id,
                'stop_order_id': stop_result.get('order_id'),
                'target_order_id': take_profit_result.get('order_id'),
                'status': 'pending',
                'timestamp': datetime.now()
            }
            
            # Add to trade history
            self.trade_history.append(trade)
            
            return {
                'success': True,
                'trade': trade,
                'entry_order': entry_result,
                'stop_order': stop_result,
                'target_order': take_profit_result
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'message': f"Error executing trade: {e}"
            }
    
    def close_position(self, symbol: str, position_id: Optional[str] = None) -> Dict:
        """
        Close an open position.
        
        Args:
            symbol: Symbol to close
            position_id: Optional position ID for multiple positions on same symbol
            
        Returns:
            Dictionary with position closing result
        """
        if not self.use_broker or self.broker_manager is None:
            self.logger.warning("Broker integration not enabled, simulating position close")
            
            # Simulate close
            close_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            result = {
                'close_id': close_id,
                'symbol': symbol,
                'status': 'simulated',
                'timestamp': timestamp,
                'message': "Position close simulated (no broker connection)"
            }
            
            return {
                'success': True,
                'result': result
            }
        
        try:
            # Get broker
            broker = self.broker_manager.get_broker()
            
            # Get position information
            positions = broker.get_positions()
            
            # Check if position exists
            if symbol not in positions:
                return {
                    'success': False,
                    'message': f"No position found for {symbol}"
                }
            
            position = positions[symbol]
            
            # Determine close side
            side = 'SELL' if position['position'] > 0 else 'BUY'
            quantity = abs(position['position'])
            
            # Place market order to close
            close_result = broker.place_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity
            )
            
            # Check for error
            if 'error' in close_result:
                return {
                    'success': False,
                    'message': f"Error closing position: {close_result['error']}"
                }
            
            return {
                'success': True,
                'result': close_result
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {
                'success': False,
                'message': f"Error closing position: {e}"
            }
    
    def update_stop_loss(self, 
                        symbol: str, 
                        new_stop_price: float, 
                        order_id: Optional[str] = None) -> Dict:
        """
        Update stop loss for an open position.
        
        Args:
            symbol: Symbol to update
            new_stop_price: New stop loss price
            order_id: Optional existing stop order ID
            
        Returns:
            Dictionary with stop loss update result
        """
        if not self.use_broker or self.broker_manager is None:
            self.logger.warning("Broker integration not enabled, simulating stop loss update")
            
            result = {
                'symbol': symbol,
                'new_stop_price': new_stop_price,
                'status': 'simulated',
                'timestamp': datetime.now(),
                'message': "Stop loss update simulated (no broker connection)"
            }
            
            return {
                'success': True,
                'result': result
            }
        
        try:
            # Get broker
            broker = self.broker_manager.get_broker()
            
            # Get orders
            orders = broker.get_orders()
            
            # Find stop order
            stop_order = None
            
            if order_id is not None:
                # Find by ID
                if order_id in orders:
                    stop_order = orders[order_id]
            else:
                # Find by symbol and type
                for order_id, order in orders.items():
                    if order['symbol'] == symbol and 'STOP' in order['order_type']:
                        stop_order = order
                        break
            
            if stop_order is None:
                return {
                    'success': False,
                    'message': f"No stop order found for {symbol}"
                }
            
            # Update stop price
            update_result = broker.modify_order(
                order_id=stop_order['order_id'],
                stop_price=new_stop_price
            )
            
            # Check for error
            if 'error' in update_result:
                return {
                    'success': False,
                    'message': f"Error updating stop loss: {update_result['error']}"
                }
            
            return {
                'success': True,
                'result': update_result
            }
            
        except Exception as e:
            self.logger.error(f"Error updating stop loss: {e}")
            return {
                'success': False,
                'message': f"Error updating stop loss: {e}"
            }
    
    def scan_watchlist(self, symbols: List[Dict]) -> List[Dict]:
        """
        Scan a watchlist of symbols for trading opportunities.
        
        Args:
            symbols: List of symbol dictionaries (with 'symbol' and 'exchange' keys)
            
        Returns:
            List of scan results
        """
        results = []
        
        for symbol_info in symbols:
            symbol = symbol_info.get('symbol')
            exchange = symbol_info.get('exchange', 'NSE')
            
            # Skip if no symbol
            if not symbol:
                continue
            
            # Analyze symbol
            result = self.analyze_symbol(symbol, exchange)
            
            # Add to results
            results.append(result)
        
        # Sort by signal strength
        sorted_results = sorted(
            [r for r in results if r.get('success', False)],
            key=lambda x: abs(x.get('signal', {}).get('strength', 0)),
            reverse=True
        )
        
        return sorted_results
    
    def generate_trading_plan(self, scan_results: List[Dict]) -> Dict:
        """
        Generate a trading plan from scan results.
        
        Args:
            scan_results: List of scan results
            
        Returns:
            Dictionary with trading plan
        """
        # Separate buy and sell signals
        buy_signals = []
        sell_signals = []
        
        for result in scan_results:
            if not result.get('success', False):
                continue
                
            signal = result.get('signal', {})
            signal_type = signal.get('signal', 'neutral')
            
            if 'buy' in signal_type:
                buy_signals.append(result)
            elif 'sell' in signal_type:
                sell_signals.append(result)
        
        # Get current positions
        current_positions = {}
        
        if self.use_broker and self.broker_manager is not None:
            try:
                broker = self.broker_manager.get_broker()
                current_positions = broker.get_positions()
            except:
                self.logger.error("Error getting current positions from broker")
        
        # Create trading plan
        trading_plan = {
            'timestamp': datetime.now(),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'current_positions': len(current_positions),
            'actions': []
        }
        
        # Process current positions
        for symbol, position in current_positions.items():
            # Skip if no position
            if position.get('position', 0) == 0:
                continue
                
            # Evaluate position
            evaluation = self.evaluate_position(symbol, position)
            
            # Add action to plan
            if evaluation['action'] != 'hold':
                trading_plan['actions'].append({
                    'symbol': symbol,
                    'action': evaluation['action'],
                    'reason': evaluation['reason'],
                    'position': position,
                    'evaluation': evaluation
                })
        
        # Process new opportunities
        available_slots = self.max_open_positions - len(current_positions)
        
        if available_slots > 0:
            # Filter to high-confidence signals
            high_confidence_signals = [
                r for r in buy_signals + sell_signals
                if r.get('signal', {}).get('confidence', 'low') in ['medium', 'high']
            ]
            
            # Sort by strength
            sorted_signals = sorted(
                high_confidence_signals,
                key=lambda x: abs(x.get('signal', {}).get('strength', 0)),
                reverse=True
            )
            
            # Add new trade opportunities
            for i, result in enumerate(sorted_signals):
                if i >= available_slots:
                    break
                    
                # Skip if already in positions
                if result.get('symbol') in current_positions:
                    continue
                
                signal = result.get('signal', {})
                position_guidance = result.get('position_guidance', {})
                
                trading_plan['actions'].append({
                    'symbol': result.get('symbol'),
                    'action': 'enter',
                    'side': 'BUY' if 'buy' in signal.get('signal', '') else 'SELL',
                    'entry_price': position_guidance.get('entry_price'),
                    'stop_price': position_guidance.get('stop_loss'),
                    'target_price': position_guidance.get('target_price'),
                    'signal': signal,
                    'position_guidance': position_guidance
                })
        
        return trading_plan
    
    def execute_trading_plan(self, trading_plan: Dict) -> Dict:
        """
        Execute a trading plan.
        
        Args:
            trading_plan: Trading plan dictionary
            
        Returns:
            Dictionary with execution results
        """
        execution_results = {
            'timestamp': datetime.now(),
            'total_actions': len(trading_plan.get('actions', [])),
            'successful_actions': 0,
            'failed_actions': 0,
            'action_results': []
        }
        
        # Process each action
        for action_item in trading_plan.get('actions', []):
            action = action_item.get('action')
            symbol = action_item.get('symbol')
            
            result = {
                'symbol': symbol,
                'action': action,
                'success': False,
                'message': ''
            }
            
            try:
                if action == 'enter':
                    # Execute new trade
                    entry_result = self.execute_trade_signal(
                        symbol=symbol,
                        side=action_item.get('side'),
                        entry_price=action_item.get('entry_price'),
                        stop_price=action_item.get('stop_price'),
                        target_price=action_item.get('target_price')
                    )
                    
                    result['success'] = entry_result.get('success', False)
                    result['message'] = entry_result.get('message', '')
                    result['trade'] = entry_result.get('trade')
                
                elif action == 'exit':
                    # Close position
                    exit_result = self.close_position(symbol)
                    
                    result['success'] = exit_result.get('success', False)
                    result['message'] = exit_result.get('message', '')
                    result['close_result'] = exit_result.get('result')
                
                elif action == 'adjust_stop':
                    # Update stop loss
                    evaluation = action_item.get('evaluation', {})
                    new_stop = evaluation.get('suggested_stop')
                    
                    if new_stop is None:
                        result['success'] = False
                        result['message'] = "Missing suggested stop price"
                    else:
                        update_result = self.update_stop_loss(symbol, new_stop)
                        
                        result['success'] = update_result.get('success', False)
                        result['message'] = update_result.get('message', '')
                        result['update_result'] = update_result.get('result')
            
            except Exception as e:
                result['success'] = False
                result['message'] = f"Error executing action: {e}"
            
            # Add to results
            execution_results['action_results'].append(result)
            
            if result['success']:
                execution_results['successful_actions'] += 1
            else:
                execution_results['failed_actions'] += 1
        
        return execution_results
    
    def update_portfolio_status(self) -> Dict:
        """
        Update portfolio status from broker.
        
        Returns:
            Dictionary with portfolio status
        """
        portfolio = {
            'timestamp': datetime.now(),
            'positions': {},
            'equity': 0,
            'cash': 0,
            'total_value': 0
        }
        
        if not self.use_broker or self.broker_manager is None:
            self.logger.warning("Broker integration not enabled, returning empty portfolio")
            return portfolio
        
        try:
            # Get broker
            broker = self.broker_manager.get_broker()
            
            # Get account information
            account_info = broker.get_account_info()
            portfolio['equity'] = float(account_info.get('equity', 0))
            portfolio['cash'] = float(account_info.get('cash', 0))
            portfolio['total_value'] = float(account_info.get('portfolio_value', 0))
            
            # Get positions
            positions = broker.get_positions()
            portfolio['positions'] = positions
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': portfolio['total_value']
            })
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio status: {e}")
            return portfolio
    
    def generate_performance_report(self) -> Dict:
        """
        Generate a performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return {
                'message': "No equity data available for reporting"
            }
        
        # Calculate basic metrics
        start_equity = self.equity_curve[0]['equity']
        current_equity = self.equity_curve[-1]['equity']
        
        absolute_return = current_equity - start_equity
        percent_return = (absolute_return / start_equity) * 100
        
        # Calculate drawdown
        max_equity = start_equity
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for point in self.equity_curve:
            equity = point['equity']
            max_equity = max(max_equity, equity)
            
            drawdown = max_equity - equity
            drawdown_pct = (drawdown / max_equity) * 100
            
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        
        # Calculate trade metrics
        winning_trades = [t for t in self.trade_history if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('profit_loss', 0) <= 0]
        
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        
        # Compile report
        report = {
            'start_date': self.equity_curve[0]['timestamp'],
            'end_date': self.equity_curve[-1]['timestamp'],
            'start_equity': start_equity,
            'current_equity': current_equity,
            'absolute_return': absolute_return,
            'percent_return': percent_return,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
        }
        
        return report
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot equity curve.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.equity_curve:
            self.logger.warning("No equity data available for plotting")
            return
        
        # Extract data
        timestamps = [p['timestamp'] for p in self.equity_curve]
        equity = [p['equity'] for p in self.equity_curve]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity, 'b-', linewidth=2)
        
        # Add trade markers
        for trade in self.trade_history:
            timestamp = trade.get('timestamp')
            
            if 'exit_timestamp' in trade and trade['exit_timestamp'] is not None:
                # Find nearest equity point to entry and exit
                entry_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - timestamp))
                exit_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - trade['exit_timestamp']))
                
                # Determine color based on profit/loss
                color = 'green' if trade.get('profit_loss', 0) > 0 else 'red'
                
                # Plot entry and exit points
                plt.scatter(timestamps[entry_idx], equity[entry_idx], marker='^', color=color, s=80, alpha=0.7)
                plt.scatter(timestamps[exit_idx], equity[exit_idx], marker='v', color=color, s=80, alpha=0.7)
        
        # Add labels and title
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Add starting point reference
        plt.axhline(y=self.equity_curve[0]['equity'], color='gray', linestyle='--', alpha=0.5)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class FibonacciCycleStrategy(TradingStrategy):
    """
    Trading strategy based on Fibonacci cycle analysis and FLD crossings.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Fibonacci cycle strategy.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.min_cycle_alignment = self.config.get('min_cycle_alignment', 0.6)
        self.min_signal_strength = self.config.get('min_signal_strength', 0.3)
        self.require_multiple_cycles = self.config.get('require_multiple_cycles', True)
        self.use_trailing_stop = self.config.get('use_trailing_stop', True)
        self.fld_confirmation = self.config.get('fld_confirmation', True)
        
        # Initialize FLD calculator
        self.fld_calculator = FLDCalculator()
    
    def analyze_symbol(self, 
                      symbol: str, 
                      exchange: str = 'NSE',
                      interval: str = 'daily') -> Dict:
        """
        Analyze a symbol using Fibonacci cycle analysis.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code
            interval: Time interval
            
        Returns:
            Dictionary with analysis results
        """
        # Use base class analysis
        result = super().analyze_symbol(symbol, exchange, interval)
        
        if not result.get('success', False):
            return result
        
        # Apply additional filters specific to Fibonacci strategy
        signal = result.get('signal', {})
        
        # Check cycle alignment
        alignment = signal.get('alignment', 0)
        if alignment < self.min_cycle_alignment:
            # Downgrade signal if alignment is low
            if signal.get('confidence', 'low') != 'low':
                signal['confidence'] = 'low'
                
                # Update signal in result
                result['signal'] = signal
                
                # Add warning
                result['warnings'] = result.get('warnings', []) + [
                    f"Low cycle alignment ({alignment:.2f} < {self.min_cycle_alignment})"
                ]
        
        # Check signal strength
        strength = abs(signal.get('strength', 0))
        if strength < self.min_signal_strength:
            # Downgrade to neutral if strength is too low
            if signal.get('signal', 'neutral') != 'neutral':
                signal['signal'] = 'neutral'
                signal['strength'] = 0
                
                # Update signal in result
                result['signal'] = signal
                
                # Add warning
                result['warnings'] = result.get('warnings', []) + [
                    f"Low signal strength ({strength:.2f} < {self.min_signal_strength})"
                ]
        
        # Check multiple cycle confirmation if required
        if self.require_multiple_cycles:
            cycle_states = result.get('cycle_states', [])
            
            if len(cycle_states) >= 2:
                # Check if at least two cycles agree on direction
                is_bullish = [state['is_bullish'] for state in cycle_states]
                bullish_count = sum(is_bullish)
                bearish_count = len(is_bullish) - bullish_count
                
                # If no agreement, downgrade signal
                if bullish_count == bearish_count:
                    if signal.get('signal', 'neutral') != 'neutral':
                        signal['signal'] = 'neutral'
                        signal['strength'] = 0
                        
                        # Update signal in result
                        result['signal'] = signal
                        
                        # Add warning
                        result['warnings'] = result.get('warnings', []) + [
                            "No agreement between cycles on direction"
                        ]
        
        # Check FLD confirmation if required
        if self.fld_confirmation:
            # Look for recent FLD crossings
            recent_crossings = False
            cycle_states = result.get('cycle_states', [])
            
            for state in cycle_states:
                days_since_cross = state.get('days_since_crossover')
                
                if days_since_cross is not None and days_since_cross <= 3:
                    recent_crossings = True
                    break
            
            if not recent_crossings:
                # No recent crossings, reduce signal strength
                signal['strength'] = signal.get('strength', 0) * 0.7
                
                # Update signal in result
                result['signal'] = signal
                
                # Add note
                result['notes'] = result.get('notes', []) + [
                    "No recent FLD crossings, reduced signal strength"
                ]
        
        # Update position guidance based on trailing stop preference
        if 'position_guidance' in result:
            position_guidance = result['position_guidance']
            position_guidance['trailing_stop'] = self.use_trailing_stop
            result['position_guidance'] = position_guidance
        
        return result
    
    def execute_trading_plan(self, trading_plan: Dict) -> Dict:
        """
        Execute a trading plan with Fibonacci-specific adjustments.
        
        Args:
            trading_plan: Trading plan dictionary
            
        Returns:
            Dictionary with execution results
        """
        # Apply Fibonacci-specific adjustments to the trading plan
        for action_item in trading_plan.get('actions', []):
            if action_item.get('action') == 'enter':
                # Check for additional Fibonacci confirmations
                signal = action_item.get('signal', {})
                
                # Only proceed with high confidence signals
                if signal.get('confidence', 'low') != 'high':
                    action_item['action'] = 'monitor'
                    action_item['reason'] = 'low_confidence'
                    continue
                
                # Check alignment threshold
                if signal.get('alignment', 0) < self.min_cycle_alignment:
                    action_item['action'] = 'monitor'
                    action_item['reason'] = 'low_alignment'
                    continue
        
        # Use base class execution
        return super().execute_trading_plan(trading_plan)


class MultiTimeframeHarmonicStrategy(TradingStrategy):
    """
    Trading strategy using multiple timeframe Fibonacci harmonic analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the multi-timeframe harmonic strategy.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.timeframes = self.config.get('timeframes', ['daily', '4h', '1h'])
        self.primary_timeframe = self.config.get('primary_timeframe', 'daily')
        self.require_alignment = self.config.get('require_timeframe_alignment', True)
        self.alignment_threshold = self.config.get('alignment_threshold', 0.7)
        self.min_signal_strength = self.config.get('min_signal_strength', 0.4)
        
        # Timeframe weights
        self.timeframe_weights = self.config.get('timeframe_weights', {
            'daily': 0.5,
            '4h': 0.3,
            '1h': 0.2
        })
    
    def analyze_symbol_multi_timeframe(self, 
                                      symbol: str, 
                                      exchange: str = 'NSE') -> Dict:
        """
        Analyze a symbol across multiple timeframes.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code
            
        Returns:
            Dictionary with multi-timeframe analysis results
        """
        # Analyze each timeframe
        timeframe_results = {}
        
        for tf in self.timeframes:
            result = super().analyze_symbol(symbol, exchange, tf)
            timeframe_results[tf] = result
        
        # Get primary timeframe result
        primary_result = timeframe_results.get(self.primary_timeframe)
        
        if not primary_result or not primary_result.get('success', False):
            return {
                'success': False,
                'error': f"Error analyzing primary timeframe {self.primary_timeframe}",
                'timeframe_results': timeframe_results
            }
        
        # Calculate timeframe alignment
        alignment_score, signals_aligned = self._calculate_timeframe_alignment(timeframe_results)
        
        # Combine signals from multiple timeframes
        combined_signal = self._combine_timeframe_signals(timeframe_results)
        
        # Apply alignment threshold if required
        if self.require_alignment and alignment_score < self.alignment_threshold:
            combined_signal['signal'] = 'neutral'
            combined_signal['strength'] = 0
            combined_signal['confidence'] = 'low'
            combined_signal['alignment'] = alignment_score
            
            primary_result['warnings'] = primary_result.get('warnings', []) + [
                f"Low timeframe alignment ({alignment_score:.2f} < {self.alignment_threshold})"
            ]
        
        # Create multi-timeframe result
        multi_tf_result = primary_result.copy()
        multi_tf_result['timeframe_results'] = timeframe_results
        multi_tf_result['timeframe_alignment'] = alignment_score
        multi_tf_result['signals_aligned'] = signals_aligned
        multi_tf_result['original_signal'] = multi_tf_result.get('signal', {}).copy()
        multi_tf_result['signal'] = combined_signal
        
        # Update position guidance based on combined signal
        if 'position_guidance' in multi_tf_result:
            position_guidance = multi_tf_result['position_guidance'].copy()
            
            # Adjust position size based on alignment
            position_size = position_guidance.get('position_size', 1.0)
            adjusted_size = position_size * alignment_score
            position_guidance['position_size'] = adjusted_size
            
            # Update in result
            multi_tf_result['position_guidance'] = position_guidance
        
        return multi_tf_result
    
    def _calculate_timeframe_alignment(self, timeframe_results: Dict) -> Tuple[float, bool]:
        """
        Calculate alignment between timeframes.
        
        Args:
            timeframe_results: Dictionary of results by timeframe
            
        Returns:
            Tuple of (alignment_score, signals_aligned)
        """
        # Count bullish, bearish, and neutral signals
        signals = {'buy': 0, 'sell': 0, 'neutral': 0}
        total_timeframes = 0
        
        for tf, result in timeframe_results.items():
            if not result.get('success', False):
                continue
                
            signal = result.get('signal', {})
            signal_type = signal.get('signal', 'neutral')
            
            if 'buy' in signal_type:
                signals['buy'] += 1
            elif 'sell' in signal_type:
                signals['sell'] += 1
            else:
                signals['neutral'] += 1
            
            total_timeframes += 1
        
        if total_timeframes == 0:
            return 0.0, False
        
        # Calculate alignment score
        max_signal = max(signals.values())
        alignment_score = max_signal / total_timeframes
        
        # Determine if signals are aligned
        signals_aligned = (signals['buy'] > 0 and signals['sell'] == 0) or \
                         (signals['sell'] > 0 and signals['buy'] == 0)
        
        return alignment_score, signals_aligned
    
    def _combine_timeframe_signals(self, timeframe_results: Dict) -> Dict:
        """
        Combine signals from multiple timeframes.
        
        Args:
            timeframe_results: Dictionary of results by timeframe
            
        Returns:
            Combined signal dictionary
        """
        # Calculate weighted signal strength
        weighted_strength = 0.0
        total_weight = 0.0
        
        for tf, result in timeframe_results.items():
            if not result.get('success', False):
                continue
                
            signal = result.get('signal', {})
            strength = signal.get('strength', 0)
            weight = self.timeframe_weights.get(tf, 1.0 / len(self.timeframes))
            
            weighted_strength += strength * weight
            total_weight += weight
        
        if total_weight == 0:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'alignment': 0
            }
        
        # Normalize weighted strength
        weighted_strength = weighted_strength / total_weight
        
        # Determine signal type based on weighted strength
        if weighted_strength > self.min_signal_strength:
            signal_type = "strong_buy" if weighted_strength > 0.7 else "buy"
        elif weighted_strength < -self.min_signal_strength:
            signal_type = "strong_sell" if weighted_strength < -0.7 else "sell"
        else:
            signal_type = "neutral"
        
        # Determine confidence based on alignment and strength
        alignment_score, signals_aligned = self._calculate_timeframe_alignment(timeframe_results)
        
        if signals_aligned and abs(weighted_strength) > 0.7:
            confidence = "high"
        elif signals_aligned and abs(weighted_strength) > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            'signal': signal_type,
            'strength': weighted_strength,
            'confidence': confidence,
            'alignment': alignment_score,
            'timeframe_weights': self.timeframe_weights
        }
    
    def analyze_symbol(self, 
                      symbol: str, 
                      exchange: str = 'NSE',
                      interval: str = None) -> Dict:
        """
        Override to perform multi-timeframe analysis.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code
            interval: Time interval (ignored for multi-timeframe analysis)
            
        Returns:
            Dictionary with analysis results
        """
        return self.analyze_symbol_multi_timeframe(symbol, exchange)
    
    def generate_trading_plan(self, scan_results: List[Dict]) -> Dict:
        """
        Generate trading plan with multi-timeframe considerations.
        
        Args:
            scan_results: List of scan results
            
        Returns:
            Dictionary with trading plan
        """
        # Only consider opportunities with high timeframe alignment
        filtered_results = []
        
        for result in scan_results:
            if not result.get('success', False):
                continue
                
            alignment = result.get('timeframe_alignment', 0)
            
            if alignment >= self.alignment_threshold:
                filtered_results.append(result)
        
        # Use filtered results for the trading plan
        return super().generate_trading_plan(filtered_results)


class MLEnhancedCycleStrategy(TradingStrategy):
    """
    Trading strategy using ML-enhanced cycle analysis with regime awareness.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ML-enhanced cycle strategy.
        
        Args:
            config: Configuration dictionary
        """
        # Ensure ML is enabled
        ml_config = config.get('ml_config', {})
        ml_config['enabled'] = True
        config['ml_config'] = ml_config
        
        super().__init__(config)
        
        # Strategy-specific parameters
        self.anomaly_detection = self.config.get('anomaly_detection', True)
        self.adaptive_sizing = self.config.get('adaptive_sizing', True)
        self.min_signal_strength = self.config.get('min_signal_strength', 0.5)
        self.min_confidence = self.config.get('min_confidence', 'medium')
        
        # Initialize components
        if self.anomaly_detection:
            self.anomaly_detector = AnomalyDetector()
        else:
            self.anomaly_detector = None
    
    def analyze_symbol(self, 
                      symbol: str, 
                      exchange: str = 'NSE',
                      interval: str = 'daily') -> Dict:
        """
        Analyze a symbol with ML enhancement and anomaly detection.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code
            interval: Time interval
            
        Returns:
            Dictionary with analysis results
        """
        # Use base class analysis
        result = super().analyze_symbol(symbol, exchange, interval)
        
        if not result.get('success', False):
            return result
        
        # Perform anomaly detection if enabled
        if self.anomaly_detection and self.anomaly_detector is not None:
            try:
                # Extract data for anomaly detection
                data = self.scanner.data_fetcher.get_data(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval
                )
                
                if data is not None and not data.empty:
                    # Fit historical data if not already done
                    if self.anomaly_detector.historical_stats == {}:
                        # Use older data for historical baseline
                        historical_window = min(252, len(data) - 20)
                        self.anomaly_detector.fit_historical_data(data.iloc[:-20].tail(historical_window))
                    
                    # Check for anomalies in recent data
                    anomalies = self.anomaly_detector.detect_anomalies(data.iloc[-20:])
                    
                    # Add anomaly information to result
                    if anomalies:
                        result['anomalies'] = anomalies
                        
                        # Add warning if significant anomalies detected
                        significant_anomalies = [a for a in anomalies if a.get('zscore', 0) > 3]
                        if significant_anomalies:
                            result['warnings'] = result.get('warnings', []) + [
                                f"Significant market anomalies detected: {len(significant_anomalies)}"
                            ]
                    
                    # Get regime change probability
                    regime_change_prob = self.anomaly_detector.get_regime_change_probability(data.iloc[-20:])
                    
                    if regime_change_prob > 0.5:
                        # Add warning about possible regime change
                        result['warnings'] = result.get('warnings', []) + [
                            f"Possible market regime change detected ({regime_change_prob:.1%} probability)"
                        ]
                        
                        # Adjust signal confidence if high regime change probability
                        if regime_change_prob > 0.7:
                            signal = result.get('signal', {})
                            if signal.get('confidence', 'low') != 'low':
                                signal['confidence'] = 'low'
                                result['signal'] = signal
            except Exception as e:
                self.logger.error(f"Error in anomaly detection: {e}")
        
        # Apply minimum signal requirements
        signal = result.get('signal', {})
        strength = abs(signal.get('strength', 0))
        confidence = signal.get('confidence', 'low')
        
        if strength < self.min_signal_strength or confidence_level(confidence) < confidence_level(self.min_confidence):
            # Downgrade to neutral if below thresholds
            signal['signal'] = 'neutral'
            signal['strength'] = 0
            signal['confidence'] = 'low'
            
            # Update signal in result
            result['signal'] = signal
            
            # Add warning
            result['warnings'] = result.get('warnings', []) + [
                f"Signal below minimum thresholds (strength: {strength:.2f}, confidence: {confidence})"
            ]
        
        # Apply adaptive position sizing based on regime
        if self.adaptive_sizing and 'position_guidance' in result and 'regime' in result:
            position_guidance = result['position_guidance']
            regime_info = result['regime']
            
            # Get regime-based position size adjustment
            regime_params = regime_info.get('parameters', {})
            position_size_factor = regime_params.get('position_size', 1.0)
            
            # Apply adjustment
            position_guidance['position_size'] = position_guidance.get('position_size', 1.0) * position_size_factor
            
            # Update in result
            result['position_guidance'] = position_guidance
        
        return result
    
    def generate_trading_plan(self, scan_results: List[Dict]) -> Dict:
        """
        Generate trading plan with ML-based risk management.
        
        Args:
            scan_results: List of scan results
            
        Returns:
            Dictionary with trading plan
        """
        # Apply ML-specific filtering
        filtered_results = []
        
        for result in scan_results:
            if not result.get('success', False):
                continue
                
            # Skip if anomalies indicate high regime change probability
            if 'anomalies' in result and len(result['anomalies']) > 3:
                continue
                
            # Skip if warnings contain regime change
            warnings = result.get('warnings', [])
            if any("regime change" in w.lower() for w in warnings):
                continue
            
            # Include result if passing filters
            filtered_results.append(result)
        
        # Use base method with filtered results
        trading_plan = super().generate_trading_plan(filtered_results)
        
        # Apply ML-specific risk management
        for action in trading_plan.get('actions', []):
            if action.get('action') == 'enter':
                # Check for regime information
                result = None
                
                for r in scan_results:
                    if r.get('symbol') == action.get('symbol'):
                        result = r
                        break
                
                if result and 'regime' in result:
                    regime_info = result['regime']
                    regime_name = regime_info.get('name', 'Unknown')
                    
                    # Adjust risk based on regime
                    if 'Volatile' in regime_name or 'Strong Downtrend' in regime_name:
                        # Reduce risk in problematic regimes
                        action['risk_pct'] = self.risk_per_trade * 0.5
                    elif 'Uptrend' in regime_name:
                        # Increase risk in favorable regimes
                        action['risk_pct'] = self.risk_per_trade * 1.2
        
        return trading_plan


def confidence_level(confidence: str) -> int:
    """Convert confidence string to numeric level."""
    levels = {'low': 0, 'medium': 1, 'high': 2}
    return levels.get(confidence.lower(), 0)


def run_strategy_backtest(strategy_class: type, config: Dict, data: pd.DataFrame, symbol: str) -> Dict:
    """
    Run a simple backtest for a strategy.
    
    Args:
        strategy_class: Strategy class to test
        config: Configuration dictionary
        data: Historical price data
        symbol: Symbol name
        
    Returns:
        Dictionary with backtest results
    """
    # Create strategy instance
    strategy = strategy_class(config)
    
    # Initialize results
    results = {
        'trades': [],
        'equity_curve': [],
        'initial_capital': config.get('initial_capital', 100000),
        'final_capital': config.get('initial_capital', 100000),
        'max_drawdown': 0,
        'max_drawdown_pct': 0,
        'win_rate': 0,
        'total_return_pct': 0
    }
    
    # Initialize variables
    capital = results['initial_capital']
    equity_curve = [{'date': data.index[0], 'equity': capital}]
    trades = []
    position = None
    
    # Process each bar
    for i in range(100, len(data)):  # Start after sufficient data for indicators
        current_date = data.index[i]
        
        # Create a slice of data up to current bar
        current_data = data.iloc[:i+1].copy()
        
        # If no position, check for entry signal
        if position is None:
            # Mock scan result
            mock_result = {
                'success': True,
                'symbol': symbol,
                'price': current_data['close'].iloc[-1],
                'signal': {'signal': 'neutral', 'strength': 0, 'confidence': 'low'},
                'cycle_states': []  # Mock cycle states
            }
            
            # Perform analysis
            # In a real backtest, this would be a proper cycle analysis on the historical slice
            
            # For simulation, just check for a simple condition
            if i > 0:
                price_change = (current_data['close'].iloc[-1] / current_data['close'].iloc[-2] - 1) * 100
                
                if price_change > 1:  # Simple condition for bullish signal
                    mock_result['signal'] = {
                        'signal': 'buy',
                        'strength': 0.7,
                        'confidence': 'medium',
                        'alignment': 0.8
                    }
                    
                    # Add position guidance
                    entry_price = current_data['close'].iloc[-1]
                    stop_price = entry_price * 0.97  # 3% stop loss
                    target_price = entry_price * 1.06  # 6% target
                    
                    mock_result['position_guidance'] = {
                        'entry_price': entry_price,
                        'stop_loss': stop_price,
                        'target_price': target_price,
                        'position_size': 1.0
                    }
                
                elif price_change < -1:  # Simple condition for bearish signal
                    mock_result['signal'] = {
                        'signal': 'sell',
                        'strength': -0.7,
                        'confidence': 'medium',
                        'alignment': 0.8
                    }
                    
                    # Add position guidance
                    entry_price = current_data['close'].iloc[-1]
                    stop_price = entry_price * 1.03  # 3% stop loss for short
                    target_price = entry_price * 0.94  # 6% target for short
                    
                    mock_result['position_guidance'] = {
                        'entry_price': entry_price,
                        'stop_loss': stop_price,
                        'target_price': target_price,
                        'position_size': 1.0
                    }
            
            # Check for entry signal
            signal = mock_result.get('signal', {})
            position_guidance = mock_result.get('position_guidance', {})
            
            if 'buy' in signal.get('signal', '') or 'sell' in signal.get('signal', ''):
                # Calculate position size
                risk_pct = config.get('risk_per_trade', 1.0)
                risk_amount = capital * (risk_pct / 100)
                
                entry_price = position_guidance.get('entry_price', mock_result.get('price', 0))
                stop_price = position_guidance.get('stop_loss')
                target_price = position_guidance.get('target_price')
                
                if entry_price > 0 and stop_price is not None:
                    # Calculate risk per share
                    risk_per_share = abs(entry_price - stop_price)
                    
                    if risk_per_share > 0:
                        # Calculate quantity
                        quantity = risk_amount / risk_per_share
                        quantity = np.floor(quantity)
                        
                        if quantity > 0:
                            # Enter position
                            position = {
                                'symbol': symbol,
                                'entry_date': current_date,
                                'entry_price': entry_price,
                                'stop_loss': stop_price,
                                'target_price': target_price,
                                'quantity': quantity,
                                'side': 'long' if 'buy' in signal.get('signal', '') else 'short',
                                'risk_amount': risk_amount
                            }
        
        # If in position, check for exit
        if position is not None:
            current_price = current_data['close'].iloc[-1]
            position_side = position['side']
            stop_loss = position['stop_loss']
            target_price = position['target_price']
            
            # Check stop loss
            stop_hit = False
            target_hit = False
            
            if position_side == 'long':
                if current_data['low'].iloc[-1] <= stop_loss:
                    stop_hit = True
                elif current_data['high'].iloc[-1] >= target_price:
                    target_hit = True
            else:  # short
                if current_data['high'].iloc[-1] >= stop_loss:
                    stop_hit = True
                elif current_data['low'].iloc[-1] <= target_price:
                    target_hit = True
            
            # Exit if stop or target hit
            if stop_hit or target_hit:
                exit_price = stop_loss if stop_hit else target_price
                
                # Calculate profit/loss
                if position_side == 'long':
                    profit_loss = (exit_price - position['entry_price']) * position['quantity']
                else:
                    profit_loss = (position['entry_price'] - exit_price) * position['quantity']
                
                # Update capital
                capital += profit_loss
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': current_date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'side': position_side,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': (profit_loss / position['risk_amount']) * 100,
                    'exit_reason': 'stop_loss' if stop_hit else 'target'
                }
                
                trades.append(trade)
                
                # Reset position
                position = None
        
        # Record equity
        equity_curve.append({
            'date': current_date,
            'equity': capital
        })
    
    # Close any open position at the end
    if position is not None:
        exit_price = data['close'].iloc[-1]
        
        # Calculate profit/loss
        if position['side'] == 'long':
            profit_loss = (exit_price - position['entry_price']) * position['quantity']
        else:
            profit_loss = (position['entry_price'] - exit_price) * position['quantity']
        
        # Update capital
        capital += profit_loss
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': data.index[-1],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'side': position['side'],
            'profit_loss': profit_loss,
            'profit_loss_pct': (profit_loss / position['risk_amount']) * 100,
            'exit_reason': 'end_of_data'
        }
        
        trades.append(trade)
    
    # Update final capital
    equity_curve.append({
        'date': data.index[-1],
        'equity': capital
    })
    
    # Calculate performance metrics
    results['trades'] = trades
    results['equity_curve'] = equity_curve
    results['final_capital'] = capital
    
    # Calculate returns
    total_return = capital - results['initial_capital']
    total_return_pct = (total_return / results['initial_capital']) * 100
    results['total_return'] = total_return
    results['total_return_pct'] = total_return_pct
    
    # Calculate drawdown
    max_equity = results['initial_capital']
    max_drawdown = 0
    max_drawdown_pct = 0
    
    for point in equity_curve:
        equity = point['equity']
        max_equity = max(max_equity, equity)
        
        drawdown = max_equity - equity
        drawdown_pct = (drawdown / max_equity) * 100
        
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    
    results['max_drawdown'] = max_drawdown
    results['max_drawdown_pct'] = max_drawdown_pct
    
    # Calculate win rate
    winning_trades = [t for t in trades if t['profit_loss'] > 0]
    results['win_rate'] = len(winning_trades) / len(trades) if trades else 0
    
    return results
