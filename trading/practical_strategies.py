import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import os

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use absolute imports
from core.scanner import FibCycleScanner
from core.fld_signal_generator import FLDCalculator
# ML imports - use conditionals to handle if they don't exist
try:
    from ml.ml_enhancements import EnsembleSignalGenerator, MarketRegimeClassifier, AnomalyDetector
except ImportError:
    # Create stub classes if they don't exist
    class EnsembleSignalGenerator: pass
    class MarketRegimeClassifier: pass
    class AnomalyDetector: pass

try:    
    from integration.broker_integration import BrokerManager
except ImportError:
    class BrokerManager: pass
    
from utils.config import load_config
from models.scan_parameters import ScanParameters
from models.scan_result import ScanResult


class AdvancedFibonacciStrategy:
    """
    Advanced Fibonacci trading strategy combining cycle analysis, FLD crossovers,
    and harmonic relationships with regime adaptation.
    """
    
    def __init__(self, config_path: str = "config/default_config.json"):
        """
        Initialize the strategy with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Init components
        self.scanner = FibCycleScanner(self.config)
        self.fld_calculator = FLDCalculator(gap_threshold=self.config['analysis']['gap_threshold'])
        self.regime_classifier = MarketRegimeClassifier(self.config.get('regime_config', {}))
        
        # ML enhancement if enabled
        self.use_ml = self.config.get('use_ml', False)
        if self.use_ml:
            self.ensemble = EnsembleSignalGenerator(self.config.get('ml_config', {}))
        else:
            self.ensemble = None
            
        # Anomaly detection
        self.anomaly_detector = AnomalyDetector(self.config.get('anomaly_config', {}))
        
        # Broker connection if enabled
        self.use_broker = self.config.get('use_broker', False)
        if self.use_broker:
            self.broker_manager = BrokerManager(self.config.get('broker_config', {}))
        else:
            self.broker_manager = None
            
        # Trading parameters
        self.risk_per_trade = self.config.get('risk_per_trade', 1.0)  # % of account
        self.max_positions = self.config.get('max_positions', 5)
        self.position_sizing_model = self.config.get('position_sizing', 'fixed')  # 'fixed', 'kelly', 'adaptive'
        
        # State tracking
        self.positions = {}
        self.pending_orders = {}
        self.trade_history = []
        self.signals = {}
        
        # Performance tracking
        self.equity_curve = []
        self.metrics = {}
    
    def connect_broker(self) -> bool:
        """
        Connect to the broker if enabled.
        
        Returns:
            Boolean indicating success
        """
        if not self.use_broker or self.broker_manager is None:
            self.logger.warning("Broker integration not enabled")
            return False
            
        try:
            success = self.broker_manager.connect_broker()
            if success:
                self.logger.info("Successfully connected to broker")
                
                # Update positions from broker
                self._sync_positions()
                
                return True
            else:
                self.logger.error("Failed to connect to broker")
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to broker: {e}")
            return False
    
    def _sync_positions(self) -> None:
        """Synchronize positions with broker."""
        if not self.use_broker or self.broker_manager is None:
            return
            
        try:
            broker = self.broker_manager.get_broker()
            if broker.connected:
                self.positions = broker.get_positions()
                self.pending_orders = broker.get_orders()
                
                self.logger.info(f"Synced {len(self.positions)} positions from broker")
        except Exception as e:
            self.logger.error(f"Error syncing positions: {e}")
    
    def analyze_symbol(self, symbol: str, exchange: str = None, interval: str = 'daily') -> ScanResult:
        """
        Analyze a symbol to generate trading signals.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code (optional)
            interval: Time interval
            
        Returns:
            ScanResult object with analysis results
        """
        # Use default exchange if not provided
        if exchange is None:
            exchange = self.config['general']['default_exchange']
            
        # Create scan parameters
        params = ScanParameters(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            lookback=self.config.get('lookback', 1000),
            num_cycles=self.config.get('num_cycles', 3),
            price_source=self.config.get('price_source', 'hlc3'),
            generate_chart=True
        )
        
        # Perform cycle analysis
        scan_result = self.scanner.analyze_symbol(params)
        
        if not scan_result.success:
            self.logger.error(f"Error analyzing {symbol}: {scan_result.error}")
            return scan_result
            
        # Apply ML enhancement if enabled
        if self.use_ml and self.ensemble is not None:
            # Get feature data
            features = self._prepare_features(symbol, exchange, interval)
            
            if features is not None:
                # Enhance signal
                enhanced_result = self.ensemble.enhance_signal(scan_result, features)
                scan_result = enhanced_result
        
        # Detect market regime
        features = self._prepare_features(symbol, exchange, interval)
        if features is not None:
            regime = self.regime_classifier.detect_regime(features)
            
            # Store regime info in result
            scan_result.regime_info = regime
            
            # Adjust signal based on regime if needed
            self._adapt_signal_to_regime(scan_result)
            
        # Store signal for tracking
        self.signals[symbol] = {
            'timestamp': datetime.now(),
            'result': scan_result
        }
        
        return scan_result
    
    def _prepare_features(self, symbol: str, exchange: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Prepare feature data for ML models and regime detection.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code
            interval: Time interval
            
        Returns:
            DataFrame with features or None if error
        """
        try:
            # Fetch data
            data = self.scanner.data_fetcher.get_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                lookback=500  # Enough for feature calculation
            )
            
            if data is None or data.empty:
                return None
                
            # Create features
            from ..ml_enhancements.feature_engineer import FeatureEngineer
            
            feature_engineer = FeatureEngineer()
            features = feature_engineer.create_technical_features(data)
            
            return features
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    def _adapt_signal_to_regime(self, scan_result: ScanResult) -> None:
        """
        Adapt signal and position guidance based on detected regime.
        
        Args:
            scan_result: ScanResult to modify
        """
        if not hasattr(scan_result, 'regime_info'):
            return
            
        regime = scan_result.regime_info
        regime_id = regime.get('current_regime_id', 3)  # Default to ranging
        
        # Get trading parameters for this regime
        trading_params = self.regime_classifier.get_trading_parameters(regime_id)
        
        # Update signal confidence if needed
        signal = scan_result.signal
        signal_type = signal.get('signal', 'neutral')
        
        # In volatile or strong downtrend regimes, reduce confidence
        if regime_id in [0, 1]:  # Volatile or Strong Downtrend
            if signal.get('confidence', 'low') == 'high':
                signal['confidence'] = 'medium'
                
            # Increase strength threshold
            min_strength = trading_params.get('min_strength', 0.3)
            if abs(signal.get('strength', 0)) < min_strength:
                signal['signal'] = 'neutral'
                signal['strength'] = 0
                
        # Update position sizing in position guidance
        if 'position_guidance' in scan_result.__dict__:
            guidance = scan_result.position_guidance
            
            # Apply position size factor from regime
            position_size_factor = trading_params.get('position_size', 1.0)
            guidance['position_size'] = guidance.get('position_size', 1.0) * position_size_factor
            
            # Apply trailing stop from regime
            guidance['trailing_stop'] = trading_params.get('trailing_stop', False)
            
            # Adjust risk/reward targets
            if 'risk' in guidance and guidance['risk'] > 0:
                risk_reward_target = trading_params.get('risk_reward_target', 2.0)
                
                if 'buy' in signal_type:
                    guidance['target_price'] = guidance['entry_price'] + guidance['risk'] * risk_reward_target
                elif 'sell' in signal_type:
                    guidance['target_price'] = guidance['entry_price'] - guidance['risk'] * risk_reward_target
                
                # Recalculate risk-reward ratio
                entry = guidance['entry_price']
                stop = guidance.get('stop_loss')
                target = guidance.get('target_price')
                
                if stop and target:
                    risk = abs(entry - stop)
                    reward = abs(target - entry)
                    
                    if risk > 0:
                        guidance['risk_reward_ratio'] = reward / risk
                
            # Update in scan result
            scan_result.position_guidance = guidance
    
    def analyze_watchlist(self, symbols: List[Dict]) -> List[ScanResult]:
        """
        Analyze multiple symbols in watchlist.
        
        Args:
            symbols: List of dictionaries with symbol and exchange keys
            
        Returns:
            List of scan results
        """
        results = []
        
        for symbol_info in symbols:
            symbol = symbol_info.get('symbol')
            exchange = symbol_info.get('exchange')
            
            if not symbol:
                continue
                
            # Analyze symbol
            result = self.analyze_symbol(symbol, exchange)
            results.append(result)
            
        # Sort by signal strength
        sorted_results = sorted(
            [r for r in results if r.success],
            key=lambda r: abs(r.signal.get('strength', 0)),
            reverse=True
        )
        
        return sorted_results
    
    def calculate_position_size(self, 
                             account_value: float, 
                             entry_price: float, 
                             stop_price: float, 
                             risk_pct: float = None,
                             confidence: str = 'medium',
                             regime_id: int = 3) -> Tuple[float, float]:
        """
        Calculate position size based on risk management rules.
        
        Args:
            account_value: Current account value
            entry_price: Entry price
            stop_price: Stop loss price
            risk_pct: Risk percentage (uses default if None)
            confidence: Signal confidence level
            regime_id: Market regime ID
            
        Returns:
            Tuple of (quantity, risk_amount)
        """
        # Use default risk if not specified
        if risk_pct is None:
            risk_pct = self.risk_per_trade
            
        # Adjust risk based on confidence
        if confidence == 'high':
            risk_pct = risk_pct * 1.2
        elif confidence == 'low':
            risk_pct = risk_pct * 0.6
            
        # Adjust risk based on regime
        trading_params = self.regime_classifier.get_trading_parameters(regime_id)
        position_size_factor = trading_params.get('position_size', 1.0)
        risk_pct = risk_pct * position_size_factor
            
        # Calculate risk amount
        risk_amount = account_value * (risk_pct / 100)
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share <= 0 or risk_per_share > entry_price * 0.1:  # Cap at 10% for safety
            # Use default risk
            risk_per_share = entry_price * 0.02  # 2% default risk
            
        # Calculate quantity
        quantity = risk_amount / risk_per_share
        
        # Apply position sizing model
        if self.position_sizing_model == 'kelly':
            # Apply Kelly Criterion adjustment if we have win rate data
            if hasattr(self, 'metrics') and 'win_rate' in self.metrics:
                win_rate = self.metrics['win_rate']
                avg_win_loss_ratio = self.metrics.get('avg_win_loss_ratio', 1.0)
                
                # Kelly fraction = win_rate - (1-win_rate)/win_loss_ratio
                kelly_fraction = win_rate - (1 - win_rate) / avg_win_loss_ratio
                
                # Apply half Kelly for safety
                kelly_fraction = max(0, kelly_fraction * 0.5)
                
                quantity = quantity * kelly_fraction
        
        # Round down to whole number for stocks
        quantity = np.floor(quantity)
        quantity = max(1, quantity)  # Minimum 1 share
        
        return quantity, risk_amount
    
    def execute_signal(self, scan_result: ScanResult) -> Dict:
        """
        Execute a trading signal.
        
        Args:
            scan_result: Scan result with signal
            
        Returns:
            Dictionary with execution results
        """
        if not scan_result.success:
            return {'success': False, 'error': 'Invalid scan result'}
            
        symbol = scan_result.symbol
        signal = scan_result.signal
        signal_type = signal.get('signal', 'neutral')
        
        # Skip neutral signals
        if 'neutral' in signal_type:
            return {'success': False, 'error': 'Neutral signal, no action taken'}
            
        # Skip if already in position for this symbol
        if symbol in self.positions:
            return {'success': False, 'error': f'Already in position for {symbol}'}
            
        # Get position guidance
        position_guidance = scan_result.position_guidance
        entry_price = position_guidance.get('entry_price')
        stop_loss = position_guidance.get('stop_loss')
        target_price = position_guidance.get('target_price')
        
        if not entry_price or not stop_loss:
            return {'success': False, 'error': 'Missing entry or stop price'}
            
        # Determine trade direction
        side = 'BUY' if 'buy' in signal_type else 'SELL'
        
        # Get account value for position sizing
        account_value = self._get_account_value()
        
        # Calculate position size
        regime_id = getattr(scan_result, 'regime_info', {}).get('current_regime_id', 3)
        quantity, risk_amount = self.calculate_position_size(
            account_value=account_value,
            entry_price=entry_price,
            stop_price=stop_loss,
            confidence=signal.get('confidence', 'medium'),
            regime_id=regime_id
        )
        
        if quantity <= 0:
            return {'success': False, 'error': 'Invalid position size'}
            
        # Execute through broker if enabled
        if self.use_broker and self.broker_manager:
            result = self._execute_broker_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price
            )
        else:
            # Simulate trade
            result = self._simulate_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price
            )
            
        if result.get('success', False):
            # Record trade
            trade = {
                'symbol': symbol,
                'entry_time': datetime.now(),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'quantity': quantity,
                'side': side,
                'risk_amount': risk_amount,
                'signal_strength': signal.get('strength', 0),
                'signal_confidence': signal.get('confidence', 'medium'),
                'status': 'open'
            }
            
            self.trade_history.append(trade)
            
            # Add to positions
            self.positions[symbol] = trade
            
            self.logger.info(f"Executed {side} signal for {symbol} at {entry_price}")
            
        return result
    
    def _execute_broker_order(self, 
                           symbol: str, 
                           side: str, 
                           quantity: float,
                           entry_price: float,
                           stop_loss: float,
                           target_price: float) -> Dict:
        """
        Execute order through broker.
        
        Args:
            symbol: Symbol to trade
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price
            
        Returns:
            Dictionary with execution results
        """
        try:
            broker = self.broker_manager.get_broker()
            
            # Place entry order
            entry_result = broker.place_order(
                symbol=symbol,
                side=side,
                order_type='LIMIT',
                quantity=quantity,
                price=entry_price,
                time_in_force='DAY'
            )
            
            if 'error' in entry_result:
                return {'success': False, 'error': f"Error placing entry order: {entry_result['error']}"}
                
            entry_order_id = entry_result.get('order_id')
            
            # Place stop loss order
            stop_side = 'SELL' if side == 'BUY' else 'BUY'
            
            stop_result = broker.place_order(
                symbol=symbol,
                side=stop_side,
                order_type='STOP',
                quantity=quantity,
                stop_price=stop_loss,
                time_in_force='GTC',
                parent_id=entry_order_id
            )
            
            # Place take profit order
            take_profit_result = broker.place_order(
                symbol=symbol,
                side=stop_side,
                order_type='LIMIT',
                quantity=quantity,
                price=target_price,
                time_in_force='GTC',
                parent_id=entry_order_id
            )
            
            return {
                'success': True,
                'entry_order': entry_result,
                'stop_order': stop_result,
                'take_profit_order': take_profit_result
            }
            
        except Exception as e:
            self.logger.error(f"Error executing broker order: {e}")
            return {'success': False, 'error': str(e)}
    
    def _simulate_order(self, 
                      symbol: str, 
                      side: str, 
                      quantity: float,
                      entry_price: float,
                      stop_loss: float,
                      target_price: float) -> Dict:
        """
        Simulate order execution for testing or paper trading.
        
        Args:
            symbol: Symbol to trade
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price
            
        Returns:
            Dictionary with simulated execution results
        """
        # Generate unique order IDs
        import uuid
        entry_order_id = str(uuid.uuid4())
        stop_order_id = str(uuid.uuid4())
        target_order_id = str(uuid.uuid4())
        
        return {
            'success': True,
            'simulated': True,
            'entry_order': {
                'order_id': entry_order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': entry_price,
                'status': 'filled',
                'filled_time': datetime.now().isoformat()
            },
            'stop_order': {
                'order_id': stop_order_id,
                'symbol': symbol,
                'side': 'SELL' if side == 'BUY' else 'BUY',
                'quantity': quantity,
                'stop_price': stop_loss,
                'status': 'working'
            },
            'take_profit_order': {
                'order_id': target_order_id,
                'symbol': symbol,
                'side': 'SELL' if side == 'BUY' else 'BUY',
                'quantity': quantity,
                'price': target_price,
                'status': 'working'
            }
        }
    
    def _get_account_value(self) -> float:
        """
        Get current account value from broker or simulation.
        
        Returns:
            Account value
        """
        if self.use_broker and self.broker_manager:
            try:
                broker = self.broker_manager.get_broker()
                account_info = broker.get_account_info()
                return float(account_info.get('portfolio_value', 100000))
            except Exception as e:
                self.logger.error(f"Error getting account value: {e}")
                return 100000  # Default value
        else:
            # For simulation, use most recent equity value or default
            if self.equity_curve:
                return self.equity_curve[-1]['equity']
            else:
                return self.config.get('initial_capital', 100000)
    
    def close_position(self, symbol: str, reason: str = 'manual') -> Dict:
        """
        Close an open position.
        
        Args:
            symbol: Symbol to close
            reason: Reason for closing
            
        Returns:
            Dictionary with results
        """
        if symbol not in self.positions:
            return {'success': False, 'error': f"No position found for {symbol}"}
            
        position = self.positions[symbol]
        
        # Execute through broker if enabled
        if self.use_broker and self.broker_manager:
            try:
                broker = self.broker_manager.get_broker()
                
                # Determine side for closing
                side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                
                # Place market order to close
                result = broker.place_order(
                    symbol=symbol,
                    side=side,
                    order_type='MARKET',
                    quantity=position['quantity']
                )
                
                if 'error' in result:
                    return {'success': False, 'error': f"Error closing position: {result['error']}"}
                    
                # Calculate profit/loss based on current price
                exit_price = result.get('price', self._get_current_price(symbol))
                
                # Record closed trade
                self._record_closed_trade(symbol, exit_price, reason)
                
                # Remove from positions
                del self.positions[symbol]
                
                return {'success': True, 'result': result}
                
            except Exception as e:
                self.logger.error(f"Error closing position with broker: {e}")
                return {'success': False, 'error': str(e)}
        else:
            # Simulate closing
            exit_price = self._get_current_price(symbol)
            
            # Record closed trade
            self._record_closed_trade(symbol, exit_price, reason)
            
            # Remove from positions
            del self.positions[symbol]
            
            return {
                'success': True,
                'simulated': True,
                'symbol': symbol,
                'exit_price': exit_price,
                'exit_time': datetime.now().isoformat()
            }
    
    def _record_closed_trade(self, symbol: str, exit_price: float, reason: str) -> None:
        """
        Record a closed trade in trade history.
        
        Args:
            symbol: Symbol that was traded
            exit_price: Exit price
            reason: Reason for exit
        """
        position = self.positions[symbol]
        
        # Calculate profit/loss
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        
        if side == 'BUY':
            profit_loss = (exit_price - entry_price) * quantity
        else:
            profit_loss = (entry_price - exit_price) * quantity
            
        # Calculate percentage gain/loss
        if entry_price > 0:
            profit_loss_pct = (profit_loss / (entry_price * quantity)) * 100
        else:
            profit_loss_pct = 0
            
        # Update trade in history
        for trade in self.trade_history:
            if trade['symbol'] == symbol and trade['status'] == 'open':
                trade['exit_time'] = datetime.now()
                trade['exit_price'] = exit_price
                trade['profit_loss'] = profit_loss
                trade['profit_loss_pct'] = profit_loss_pct
                trade['exit_reason'] = reason
                trade['status'] = 'closed'
                break
                
        # Update equity curve
        equity = self._get_account_value() + profit_loss  # Add P/L if not reflected in account yet
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': equity
        })
        
        # Update metrics
        self._update_performance_metrics()
        
        self.logger.info(f"Closed {side} position for {symbol} at {exit_price}, P/L: {profit_loss:.2f} ({profit_loss_pct:.2f}%)")
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price
        """
        if self.use_broker and self.broker_manager:
            try:
                broker = self.broker_manager.get_broker()
                quote = broker.get_market_data(symbol)
                return quote.get('last', 0) or quote.get('close', 0)
            except Exception as e:
                self.logger.error(f"Error getting current price from broker: {e}")
                
        # Fallback to data fetcher
        try:
            data = self.scanner.data_fetcher.get_data(
                symbol=symbol,
                lookback=1
            )
            
            if data is not None and not data.empty:
                return data['close'].iloc[-1]
        except Exception as e:
            self.logger.error(f"Error getting current price from data fetcher: {e}")
            
        # If we get here, use last known price from positions
        if symbol in self.positions:
            return self.positions[symbol]['entry_price']
            
        return 0
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on trade history."""
        closed_trades = [t for t in self.trade_history if t['status'] == 'closed']
        
        if not closed_trades:
            return
            
        # Calculate metrics
        winning_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit_loss', 0) <= 0]
        
        total_trades = len(closed_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate average win/loss
        avg_win = np.mean([t['profit_loss'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['profit_loss']) for t in losing_trades]) if losing_trades else 0
        
        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
        
        # Calculate profit factor
        gross_profit = sum([t['profit_loss'] for t in winning_trades])
        gross_loss = sum([abs(t['profit_loss']) for t in losing_trades])
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Calculate drawdown
        if self.equity_curve:
            max_equity = self.equity_curve[0]['equity']
            max_drawdown = 0
            max_drawdown_pct = 0
            
            for point in self.equity_curve:
                equity = point['equity']
                max_equity = max(max_equity, equity)
                
                drawdown = max_equity - equity
                drawdown_pct = (drawdown / max_equity) * 100 if max_equity > 0 else 0
                
                max_drawdown = max(max_drawdown, drawdown)
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Update metrics
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct
        }
    
    def update_stops_and_targets(self) -> Dict:
        """
        Update stop losses and targets for open positions.
        
        Returns:
            Dictionary with update results
        """
        results = {'updated': 0, 'errors': 0, 'details': []}
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            
            # Re-analyze the symbol
            scan_result = self.analyze_symbol(symbol)
            
            if not scan_result.success:
                results['details'].append({
                    'symbol': symbol,
                    'success': False,
                    'error': 'Analysis failed'
                })
                results['errors'] += 1
                continue
                
            # Check if we need to update the stop loss
            position_side = position['side']
            position_guidance = scan_result.position_guidance
            current_stop = position.get('stop_loss')
            suggested_stop = position_guidance.get('stop_loss')
            
            update_needed = False
            
            if suggested_stop and current_stop:
                # For long positions, only move stop up (trailing stop)
                if position_side == 'BUY' and suggested_stop > current_stop:
                    update_needed = True
                
                # For short positions, only move stop down (trailing stop)
                elif position_side == 'SELL' and suggested_stop < current_stop:
                    update_needed = True
            
            if update_needed:
                # Execute through broker if enabled
                if self.use_broker and self.broker_manager:
                    try:
                        result = self._update_broker_stop(symbol, suggested_stop)
                        
                        if result.get('success', False):
                            # Update position record
                            position['stop_loss'] = suggested_stop
                            results['updated'] += 1
                            results['details'].append({
                                'symbol': symbol,
                                'success': True,
                                'old_stop': current_stop,
                                'new_stop': suggested_stop,
                                'broker_result': result
                            })
                        else:
                            results['errors'] += 1
                            results['details'].append({
                                'symbol': symbol,
                                'success': False,
                                'error': result.get('error', 'Unknown error')
                            })
                    except Exception as e:
                        results['errors'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'success': False,
                            'error': str(e)
                        })
                else:
                    # Simulate stop update
                    position['stop_loss'] = suggested_stop
                    results['updated'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'success': True,
                        'old_stop': current_stop,
                        'new_stop': suggested_stop,
                        'simulated': True
                    })
                    
                    self.logger.info(f"Updated stop for {symbol} from {current_stop} to {suggested_stop}")
        
        return results
    
    def _update_broker_stop(self, symbol: str, new_stop: float) -> Dict:
        """
        Update stop loss order through broker.
        
        Args:
            symbol: Symbol to update
            new_stop: New stop price
            
        Returns:
            Dictionary with update results
        """
        try:
            broker = self.broker_manager.get_broker()
            
            # Find existing stop order
            orders = broker.get_orders()
            stop_order = None
            
            for order_id, order in orders.items():
                if order['symbol'] == symbol and 'STOP' in order['order_type']:
                    stop_order = order
                    break
            
            if not stop_order:
                return {'success': False, 'error': 'Stop order not found'}
                
            # Update stop price
            result = broker.modify_order(
                order_id=stop_order['order_id'],
                stop_price=new_stop
            )
            
            return {'success': True, 'result': result}
            
        except Exception as e:
            self.logger.error(f"Error updating broker stop: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_positions(self) -> Dict:
        """
        Check open positions for exit signals or target/stop hits.
        
        Returns:
            Dictionary with check results
        """
        results = {'closed': 0, 'errors': 0, 'details': []}
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            
            # Get current price
            current_price = self._get_current_price(symbol)
            
            if current_price <= 0:
                results['errors'] += 1
                results['details'].append({
                    'symbol': symbol,
                    'success': False,
                    'error': 'Invalid current price'
                })
                continue
                
            # Check for stop loss hit
            stop_loss = position.get('stop_loss')
            target_price = position.get('target_price')
            position_side = position['side']
            
            stop_hit = False
            target_hit = False
            
            if stop_loss:
                if position_side == 'BUY' and current_price <= stop_loss:
                    stop_hit = True
                elif position_side == 'SELL' and current_price >= stop_loss:
                    stop_hit = True
            
            if target_price:
                if position_side == 'BUY' and current_price >= target_price:
                    target_hit = True
                elif position_side == 'SELL' and current_price <= target_price:
                    target_hit = True
            
            # Check for signal reversal
            signal_reversal = False
            
            if not stop_hit and not target_hit:
                # Re-analyze the symbol
                scan_result = self.analyze_symbol(symbol)
                
                if scan_result.success:
                    signal_type = scan_result.signal.get('signal', 'neutral')
                    
                    if position_side == 'BUY' and 'sell' in signal_type:
                        signal_reversal = True
                    elif position_side == 'SELL' and 'buy' in signal_type:
                        signal_reversal = True
            
            # Close position if needed
            if stop_hit or target_hit or signal_reversal:
                reason = 'stop_loss' if stop_hit else 'target' if target_hit else 'signal_reversal'
                
                try:
                    result = self.close_position(symbol, reason)
                    
                    if result.get('success', False):
                        results['closed'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'success': True,
                            'reason': reason,
                            'exit_price': current_price,
                            'result': result
                        })
                    else:
                        results['errors'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'success': False,
                            'error': result.get('error', 'Unknown error')
                        })
                except Exception as e:
                    results['errors'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def generate_trading_plan(self, scan_results: List[ScanResult]) -> Dict:
        """
        Generate a trading plan from scan results.
        
        Args:
            scan_results: List of scan results
            
        Returns:
            Dictionary with trading plan
        """
        # Filter to successful results
        valid_results = [r for r in scan_results if r.success]
        
        # Separate by signal type
        buy_signals = []
        sell_signals = []
        
        for result in valid_results:
            signal_type = result.signal.get('signal', 'neutral')
            
            if 'buy' in signal_type:
                buy_signals.append(result)
            elif 'sell' in signal_type:
                sell_signals.append(result)
        
        # Sort by signal strength
        buy_signals.sort(key=lambda r: abs(r.signal.get('strength', 0)), reverse=True)
        sell_signals.sort(key=lambda r: abs(r.signal.get('strength', 0)), reverse=True)
        
        # Check current positions
        current_symbols = set(self.positions.keys())
        
        # Create plan
        plan = {
            'timestamp': datetime.now(),
            'positions': len(current_symbols),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'available_slots': max(0, self.max_positions - len(current_symbols)),
            'actions': []
        }
        
        # Check existing positions first
        for symbol in current_symbols:
            position = self.positions[symbol]
            
            # Find corresponding result if any
            matching_result = None
            for result in valid_results:
                if result.symbol == symbol:
                    matching_result = result
                    break
            
            if matching_result:
                signal_type = matching_result.signal.get('signal', 'neutral')
                position_side = position['side']
                
                # Check for exit signal
                if (position_side == 'BUY' and 'sell' in signal_type) or \
                   (position_side == 'SELL' and 'buy' in signal_type):
                    plan['actions'].append({
                        'symbol': symbol,
                        'action': 'close',
                        'reason': 'signal_reversal',
                        'current_price': self._get_current_price(symbol),
                        'position': position
                    })
                else:
                    # Check if stop loss update needed
                    current_stop = position.get('stop_loss')
                    suggested_stop = matching_result.position_guidance.get('stop_loss')
                    
                    if suggested_stop and current_stop:
                        if (position_side == 'BUY' and suggested_stop > current_stop) or \
                           (position_side == 'SELL' and suggested_stop < current_stop):
                            plan['actions'].append({
                                'symbol': symbol,
                                'action': 'update_stop',
                                'old_stop': current_stop,
                                'new_stop': suggested_stop,
                                'position': position
                            })
        
        # Add new entry opportunities
        if plan['available_slots'] > 0:
            # Prioritize entry signals
            entry_candidates = buy_signals + sell_signals
            
            # Filter to high-quality signals
            filtered_candidates = [
                r for r in entry_candidates 
                if r.signal.get('confidence', 'low') != 'low' and 
                abs(r.signal.get('strength', 0)) >= 0.4 and
                r.symbol not in current_symbols
            ]
            
            # Sort by strength
            filtered_candidates.sort(key=lambda r: abs(r.signal.get('strength', 0)), reverse=True)
            
            # Add top N candidates
            for i, result in enumerate(filtered_candidates):
                if i >= plan['available_slots']:
                    break
                    
                signal_type = result.signal.get('signal', 'neutral')
                side = 'BUY' if 'buy' in signal_type else 'SELL'
                
                plan['actions'].append({
                    'symbol': result.symbol,
                    'action': 'enter',
                    'side': side,
                    'entry_price': result.position_guidance.get('entry_price'),
                    'stop_loss': result.position_guidance.get('stop_loss'),
                    'target_price': result.position_guidance.get('target_price'),
                    'signal_strength': result.signal.get('strength', 0),
                    'confidence': result.signal.get('confidence', 'medium'),
                    'scan_result': result
                })
        
        return plan
    
    def execute_trading_plan(self, plan: Dict) -> Dict:
        """
        Execute a trading plan.
        
        Args:
            plan: Trading plan to execute
            
        Returns:
            Dictionary with execution results
        """
        results = {
            'timestamp': datetime.now(),
            'total_actions': len(plan.get('actions', [])),
            'successful_actions': 0,
            'failed_actions': 0,
            'action_results': []
        }
        
        # Process each action
        for action_item in plan.get('actions', []):
            action_type = action_item.get('action')
            symbol = action_item.get('symbol')
            
            result = {
                'symbol': symbol,
                'action': action_type,
                'success': False
            }
            
            try:
                # Execute based on action type
                if action_type == 'enter':
                    # Execute new position
                    scan_result = action_item.get('scan_result')
                    if scan_result:
                        entry_result = self.execute_signal(scan_result)
                        result['success'] = entry_result.get('success', False)
                        result['details'] = entry_result
                    else:
                        result['error'] = 'Missing scan result'
                
                elif action_type == 'close':
                    # Close position
                    close_result = self.close_position(symbol, action_item.get('reason', 'trading_plan'))
                    result['success'] = close_result.get('success', False)
                    result['details'] = close_result
                
                elif action_type == 'update_stop':
                    # Update stop loss
                    new_stop = action_item.get('new_stop')
                    
                    if new_stop and symbol in self.positions:
                        # Update the stop in our position tracking
                        self.positions[symbol]['stop_loss'] = new_stop
                        
                        # Update through broker if enabled
                        if self.use_broker and self.broker_manager:
                            update_result = self._update_broker_stop(symbol, new_stop)
                            result['success'] = update_result.get('success', False)
                            result['details'] = update_result
                        else:
                            # Simulated update
                            result['success'] = True
                            result['simulated'] = True
                    else:
                        result['error'] = 'Invalid stop price or position not found'
            
            except Exception as e:
                result['success'] = False
                result['error'] = str(e)
                self.logger.error(f"Error executing {action_type} for {symbol}: {e}")
            
            # Add to results
            results['action_results'].append(result)
            
            if result['success']:
                results['successful_actions'] += 1
            else:
                results['failed_actions'] += 1
        
        return results
    
    def generate_performance_report(self) -> Dict:
        """
        Generate a performance report.
        
        Returns:
            Dictionary with performance data
        """
        # Update metrics first
        self._update_performance_metrics()
        
        # Get current equity
        current_equity = self._get_account_value()
        
        # Create report
        report = {
            'timestamp': datetime.now(),
            'metrics': self.metrics,
            'open_positions': len(self.positions),
            'position_details': self.positions,
            'trade_count': len(self.trade_history),
            'current_equity': current_equity
        }
        
        # Add equity curve summary if available
        if self.equity_curve:
            initial_equity = self.equity_curve[0]['equity']
            
            report['equity'] = {
                'initial': initial_equity,
                'current': current_equity,
                'change': current_equity - initial_equity,
                'change_pct': ((current_equity / initial_equity) - 1) * 100 if initial_equity > 0 else 0,
                'points': len(self.equity_curve)
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
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity, 'b-', linewidth=2)
        
        # Add markers for trades
        closed_trades = [t for t in self.trade_history if t.get('status') == 'closed']
        
        for trade in closed_trades:
            exit_time = trade.get('exit_time')
            if not exit_time:
                continue
                
            # Find nearest equity point
            closest_idx = min(range(len(timestamps)), 
                            key=lambda i: abs((timestamps[i] - exit_time).total_seconds()))
            
            if closest_idx < len(equity):
                marker_color = 'green' if trade.get('profit_loss', 0) > 0 else 'red'
                plt.scatter(timestamps[closest_idx], equity[closest_idx], 
                           color=marker_color, marker='o', s=50, alpha=0.7)
        
        # Add labels and title
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        
        # Add metrics annotation
        if self.metrics:
            metrics_text = (
                f"Win Rate: {self.metrics.get('win_rate', 0)*100:.1f}%\n"
                f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}\n"
                f"Trades: {self.metrics.get('total_trades', 0)}\n"
                f"Max DD: {self.metrics.get('max_drawdown_pct', 0):.1f}%"
            )
            
            plt.annotate(metrics_text, xy=(0.02, 0.97), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        va='top', ha='left')
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class SwingTradingStrategy(AdvancedFibonacciStrategy):
    """
    Fibonacci-based swing trading strategy optimized for multi-day holding periods.
    Focuses on larger cycles and stronger trends for position trades.
    """
    
    def __init__(self, config_path: str = "config/swing_config.json"):
        """Initialize with swing-specific configuration."""
        super().__init__(config_path)
        
        # Swing-specific parameters
        self.min_cycle_length = self.config.get('min_cycle_length', 40)  # Focus on larger cycles
        self.min_signal_strength = self.config.get('min_signal_strength', 0.5)  # Require stronger signals
        self.min_cycle_alignment = self.config.get('min_cycle_alignment', 0.7)  # Higher alignment threshold
        self.use_weekly_confirmation = self.config.get('use_weekly_confirmation', True)
        
    def analyze_symbol(self, symbol: str, exchange: str = None, interval: str = 'daily') -> ScanResult:
        """
        Override to add swing-specific analysis enhancements.
        """
        # First get the base analysis
        result = super().analyze_symbol(symbol, exchange, interval)
        
        if not result.success:
            return result
            
        # Additional swing-specific filtering
        
        # 1. Check cycle lengths
        cycles = result.detected_cycles
        if not any(cycle >= self.min_cycle_length for cycle in cycles):
            # Downgrade signal if no large enough cycles
            result.signal['signal'] = 'neutral'
            result.signal['strength'] = 0
            result.signal['confidence'] = 'low'
            result.warnings = result.warnings + ['No suitable swing cycles detected']
            return result
            
        # 2. Check signal strength and alignment
        signal = result.signal
        strength = abs(signal.get('strength', 0))
        alignment = signal.get('alignment', 0)
        
        if strength < self.min_signal_strength or alignment < self.min_cycle_alignment:
            # Downgrade signal
            result.signal['signal'] = 'neutral'
            result.signal['strength'] = 0
            result.signal['confidence'] = 'low'
            result.warnings = result.warnings + [
                f'Insufficient signal metrics for swing trade: strength={strength:.2f}, alignment={alignment:.2f}'
            ]
            return result
            
        # 3. Add weekly confirmation if enabled
        if self.use_weekly_confirmation and interval == 'daily':
            weekly_result = super().analyze_symbol(symbol, exchange, 'weekly')
            
            if weekly_result.success:
                weekly_signal = weekly_result.signal.get('signal', 'neutral')
                daily_signal = signal.get('signal', 'neutral')
                
                # Check if weekly and daily align
                if ('buy' in daily_signal and 'sell' in weekly_signal) or \
                   ('sell' in daily_signal and 'buy' in weekly_signal):
                    # Weekly contradicts daily - downgrade
                    result.signal['confidence'] = 'low'
                    result.warnings = result.warnings + ['Weekly timeframe contradicts daily signal']
                elif ('buy' in daily_signal and 'buy' in weekly_signal) or \
                     ('sell' in daily_signal and 'sell' in weekly_signal):
                    # Weekly confirms daily - upgrade
                    if result.signal['confidence'] != 'high':
                        result.signal['confidence'] = 'high'
                    result.notes = result.notes + ['Weekly timeframe confirms daily signal']
        
        # 4. Adjust position guidance for swing trades
        if 'position_guidance' in result.__dict__:
            # Use wider stops for swing trades
            guidance = result.position_guidance
            entry_price = guidance.get('entry_price', 0)
            
            if entry_price > 0:
                signal_type = signal.get('signal', 'neutral')
                
                if 'buy' in signal_type:
                    # For long trades, use 1.5x the normal stop distance
                    normal_stop = guidance.get('stop_loss', entry_price * 0.95)
                    stop_distance = entry_price - normal_stop
                    wider_stop = entry_price - (stop_distance * 1.5)
                    guidance['stop_loss'] = wider_stop
                    
                    # Set a more ambitious target
                    guidance['target_price'] = entry_price + (stop_distance * 3)
                
                elif 'sell' in signal_type:
                    # For short trades, use 1.5x the normal stop distance
                    normal_stop = guidance.get('stop_loss', entry_price * 1.05)
                    stop_distance = normal_stop - entry_price
                    wider_stop = entry_price + (stop_distance * 1.5)
                    guidance['stop_loss'] = wider_stop
                    
                    # Set a more ambitious target
                    guidance['target_price'] = entry_price - (stop_distance * 3)
                
                # Recalculate risk-reward ratio
                if guidance.get('stop_loss') and guidance.get('target_price'):
                    risk = abs(entry_price - guidance['stop_loss'])
                    reward = abs(guidance['target_price'] - entry_price)
                    
                    if risk > 0:
                        guidance['risk_reward_ratio'] = reward / risk
                        
                result.position_guidance = guidance
        
        return result
    
    def update_stops_and_targets(self) -> Dict:
        """
        Override with swing-specific stop management.
        Uses a more conservative approach to trailing stops.
        """
        results = {'updated': 0, 'errors': 0, 'details': []}
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            
            # For swing trades, only update stops after significant movement
            entry_price = position.get('entry_price', 0)
            current_price = self._get_current_price(symbol)
            current_stop = position.get('stop_loss', 0)
            position_side = position['side']
            
            # Calculate price movement since entry
            if entry_price > 0 and current_price > 0:
                price_movement_pct = ((current_price / entry_price) - 1) * 100
                
                # Only consider updating if price has moved significantly in our favor
                significant_movement = False
                
                if position_side == 'BUY' and price_movement_pct > 5:
                    significant_movement = True
                elif position_side == 'SELL' and price_movement_pct < -5:
                    significant_movement = True
                
                if significant_movement:
                    # Calculate a conservative trailing stop
                    new_stop = current_stop
                    
                    if position_side == 'BUY':
                        # Move stop to breakeven after 5% gain
                        if current_stop < entry_price and price_movement_pct >= 5:
                            new_stop = entry_price
                        
                        # Use ATR-based trailing stop after 10% gain
                        elif price_movement_pct >= 10:
                            # Get ATR value from recent data
                            data = self.scanner.data_fetcher.get_data(
                                symbol=symbol,
                                lookback=20
                            )
                            
                            if data is not None and not data.empty:
                                # Calculate ATR
                                tr_values = []
                                for i in range(1, len(data)):
                                    high = data['high'].iloc[i]
                                    low = data['low'].iloc[i]
                                    prev_close = data['close'].iloc[i-1]
                                    
                                    tr1 = high - low
                                    tr2 = abs(high - prev_close)
                                    tr3 = abs(low - prev_close)
                                    
                                    tr_values.append(max(tr1, tr2, tr3))
                                
                                atr = sum(tr_values) / len(tr_values) if tr_values else 0
                                
                                if atr > 0:
                                    # Set stop 3 ATR below current price
                                    potential_stop = current_price - (3 * atr)
                                    # Only move stop up
                                    if potential_stop > current_stop:
                                        new_stop = potential_stop
                    
                    elif position_side == 'SELL':
                        # Move stop to breakeven after 5% gain
                        if current_stop > entry_price and price_movement_pct <= -5:
                            new_stop = entry_price
                        
                        # Use ATR-based trailing stop after 10% gain
                        elif price_movement_pct <= -10:
                            # Get ATR value from recent data
                            data = self.scanner.data_fetcher.get_data(
                                symbol=symbol,
                                lookback=20
                            )
                            
                            if data is not None and not data.empty:
                                # Calculate ATR
                                tr_values = []
                                for i in range(1, len(data)):
                                    high = data['high'].iloc[i]
                                    low = data['low'].iloc[i]
                                    prev_close = data['close'].iloc[i-1]
                                    
                                    tr1 = high - low
                                    tr2 = abs(high - prev_close)
                                    tr3 = abs(low - prev_close)
                                    
                                    tr_values.append(max(tr1, tr2, tr3))
                                
                                atr = sum(tr_values) / len(tr_values) if tr_values else 0
                                
                                if atr > 0:
                                    # Set stop 3 ATR above current price
                                    potential_stop = current_price + (3 * atr)
                                    # Only move stop down
                                    if potential_stop < current_stop:
                                        new_stop = potential_stop
                    
                    # Update stop if it changed
                    if new_stop != current_stop:
                        # Execute through broker if enabled
                        if self.use_broker and self.broker_manager:
                            try:
                                result = self._update_broker_stop(symbol, new_stop)
                                
                                if result.get('success', False):
                                    # Update position record
                                    position['stop_loss'] = new_stop
                                    results['updated'] += 1
                                    results['details'].append({
                                        'symbol': symbol,
                                        'success': True,
                                        'old_stop': current_stop,
                                        'new_stop': new_stop,
                                        'broker_result': result
                                    })
                                else:
                                    results['errors'] += 1
                                    results['details'].append({
                                        'symbol': symbol,
                                        'success': False,
                                        'error': result.get('error', 'Unknown error')
                                    })
                            except Exception as e:
                                results['errors'] += 1
                                results['details'].append({
                                    'symbol': symbol,
                                    'success': False,
                                    'error': str(e)
                                })
                        else:
                            # Simulate stop update
                            position['stop_loss'] = new_stop
                            results['updated'] += 1
                            results['details'].append({
                                'symbol': symbol,
                                'success': True,
                                'old_stop': current_stop,
                                'new_stop': new_stop,
                                'simulated': True
                            })
                            
                            self.logger.info(f"Updated stop for {symbol} from {current_stop} to {new_stop}")
        
        return results


class DayTradingStrategy(AdvancedFibonacciStrategy):
    """
    Fibonacci-based day trading strategy optimized for intraday trading.
    Focuses on smaller cycles and quicker moves for day trades.
    """
    
    def __init__(self, config_path: str = "config/daytrading_config.json"):
        """Initialize with day trading specific configuration."""
        super().__init__(config_path)
        
		# Day trading specific parameters
        self.max_cycle_length = self.config.get('max_cycle_length', 34)  # Focus on smaller cycles
        self.min_signal_strength = self.config.get('min_signal_strength', 0.4)  # Moderate signal strength
        self.use_volume_confirmation = self.config.get('use_volume_confirmation', True)
        self.tight_stops = self.config.get('tight_stops', True)
        self.use_multiple_timeframes = self.config.get('use_multiple_timeframes', True)
        self.intraday_intervals = self.config.get('intraday_intervals', ['1h', '15m', '5m'])
        
        # Position management parameters
        self.quick_profit_target = self.config.get('quick_profit_target', True)
        self.partial_profit_taking = self.config.get('partial_profit_taking', True)
        self.max_daily_trades = self.config.get('max_daily_trades', 5)
        
        # Daily tracking
        self.daily_trades = 0
        self.last_reset_day = datetime.now().date()
        
        # Reset daily trade counter at market open
        self._check_daily_reset()
    
    def _check_daily_reset(self) -> None:
        """Reset daily trade counter if it's a new day."""
        today = datetime.now().date()
        if today != self.last_reset_day:
            self.daily_trades = 0
            self.last_reset_day = today
    
    def analyze_symbol(self, symbol: str, exchange: str = None, interval: str = '15m') -> ScanResult:
        """
        Override with day trading specific analysis enhancements.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code (optional)
            interval: Time interval (default to 15-minute for day trading)
            
        Returns:
            ScanResult with analysis results
        """
        # First check daily trade limit
        self._check_daily_reset()
        if self.daily_trades >= self.max_daily_trades:
            result = ScanResult(
                symbol=symbol,
                exchange=exchange or self.config['general']['default_exchange'],
                interval=interval,
                success=True
            )
            result.signal = {'signal': 'neutral', 'strength': 0, 'confidence': 'low'}
            result.warnings = ['Daily trade limit reached']
            return result
        
        # First get the base analysis
        result = super().analyze_symbol(symbol, exchange, interval)
        
        if not result.success:
            return result
        
        # Additional day trading specific filtering
        
        # 1. Check cycle lengths
        cycles = result.detected_cycles
        if not cycles or all(cycle > self.max_cycle_length for cycle in cycles):
            # Downgrade signal if cycles are too large
            result.signal['signal'] = 'neutral'
            result.signal['strength'] = 0
            result.signal['confidence'] = 'low'
            result.warnings = result.warnings + ['No suitable intraday cycles detected']
            return result
        
        # 2. Add multi-timeframe confirmation if enabled
        if self.use_multiple_timeframes:
            signal_type = result.signal.get('signal', 'neutral')
            
            if 'neutral' not in signal_type:
                # Check higher timeframe for trend direction
                if interval != '1h' and '1h' in self.intraday_intervals:
                    hourly_result = super().analyze_symbol(symbol, exchange, '1h')
                    
                    if hourly_result.success:
                        hourly_signal = hourly_result.signal.get('signal', 'neutral')
                        
                        # Check for alignment between timeframes
                        if ('buy' in signal_type and 'sell' in hourly_signal) or \
                           ('sell' in signal_type and 'buy' in hourly_signal):
                            # Counter-trend trade - downgrade or neutralize
                            result.signal['confidence'] = 'low'
                            result.warnings = result.warnings + ['Counter-trend to hourly timeframe']
                            
                            # For day trading, we generally avoid counter-trend trades
                            result.signal['signal'] = 'neutral'
                            result.signal['strength'] = 0
                            return result
                        
                        elif ('buy' in signal_type and 'buy' in hourly_signal) or \
                             ('sell' in signal_type and 'sell' in hourly_signal):
                            # With-trend trade - upgrade confidence
                            result.signal['confidence'] = 'high'
                            result.notes = result.notes + ['Aligned with hourly timeframe trend']
        
        # 3. Volume confirmation if enabled
        if self.use_volume_confirmation:
            # Get recent volume data
            data = self.scanner.data_fetcher.get_data(
                symbol=symbol,
                exchange=exchange or self.config['general']['default_exchange'],
                interval=interval,
                lookback=20
            )
            
            if data is not None and not data.empty and 'volume' in data.columns:
                # Calculate average volume
                avg_volume = data['volume'].rolling(window=10).mean()
                latest_volume = data['volume'].iloc[-1]
                
                # Volume ratio to average
                volume_ratio = latest_volume / avg_volume.iloc[-1] if not pd.isna(avg_volume.iloc[-1]) and avg_volume.iloc[-1] > 0 else 1.0
                
                # Check if volume confirms signal
                signal_type = result.signal.get('signal', 'neutral')
                
                if 'buy' in signal_type or 'sell' in signal_type:
                    if volume_ratio < 0.8:
                        # Low volume - downgrade confidence
                        if result.signal['confidence'] != 'low':
                            result.signal['confidence'] = 'low'
                        result.warnings = result.warnings + [f'Low volume confirmation: {volume_ratio:.2f}x average']
                    elif volume_ratio > 1.5:
                        # High volume - upgrade confidence
                        if result.signal['confidence'] != 'high':
                            result.signal['confidence'] = 'high'
                        result.notes = result.notes + [f'Strong volume confirmation: {volume_ratio:.2f}x average']
        
        # 4. Adjust position guidance for day trades
        if 'position_guidance' in result.__dict__:
            guidance = result.position_guidance
            entry_price = guidance.get('entry_price', 0)
            
            if entry_price > 0:
                signal_type = result.signal.get('signal', 'neutral')
                
                # Set tighter stops for day trading
                if self.tight_stops:
                    if 'buy' in signal_type:
                        # For long trades, use tighter stop
                        normal_stop = guidance.get('stop_loss', entry_price * 0.98)
                        stop_distance = entry_price - normal_stop
                        tighter_stop = entry_price - (stop_distance * 0.7)
                        guidance['stop_loss'] = tighter_stop
                    
                    elif 'sell' in signal_type:
                        # For short trades, use tighter stop
                        normal_stop = guidance.get('stop_loss', entry_price * 1.02)
                        stop_distance = normal_stop - entry_price
                        tighter_stop = entry_price + (stop_distance * 0.7)
                        guidance['stop_loss'] = tighter_stop
                
                # Set quicker profit targets for day trading
                if self.quick_profit_target:
                    risk = abs(entry_price - guidance.get('stop_loss', entry_price * 0.98))
                    
                    if 'buy' in signal_type:
                        # For long trades, 1.5:1 reward-to-risk for quicker profits
                        guidance['target_price'] = entry_price + (risk * 1.5)
                    elif 'sell' in signal_type:
                        # For short trades, 1.5:1 reward-to-risk for quicker profits
                        guidance['target_price'] = entry_price - (risk * 1.5)
                
                # Recalculate risk-reward ratio
                if guidance.get('stop_loss') and guidance.get('target_price'):
                    risk = abs(entry_price - guidance['stop_loss'])
                    reward = abs(guidance['target_price'] - entry_price)
                    
                    if risk > 0:
                        guidance['risk_reward_ratio'] = reward / risk
                
                # Add partial targets for day trading
                if self.partial_profit_taking:
                    if 'buy' in signal_type:
                        partial_target = entry_price + (risk * 0.8)
                        guidance['partial_target'] = partial_target
                        guidance['partial_size'] = 0.5  # Close 50% at first target
                    elif 'sell' in signal_type:
                        partial_target = entry_price - (risk * 0.8)
                        guidance['partial_target'] = partial_target
                        guidance['partial_size'] = 0.5  # Close 50% at first target
                
                result.position_guidance = guidance
        
        return result
    
    def execute_signal(self, scan_result: ScanResult) -> Dict:
        """
        Override to handle day trading specific execution.
        
        Args:
            scan_result: Scan result with signal
            
        Returns:
            Dictionary with execution results
        """
        # Check daily trade limit
        self._check_daily_reset()
        if self.daily_trades >= self.max_daily_trades:
            return {
                'success': False,
                'error': f'Daily trade limit reached ({self.daily_trades}/{self.max_daily_trades})'
            }
        
        # Execute trade
        result = super().execute_signal(scan_result)
        
        # Increment daily trade counter if trade was successful
        if result.get('success', False):
            self.daily_trades += 1
        
        return result
    
    def check_positions(self) -> Dict:
        """
        Override for day trading specific position checking.
        Adds handling of partial profit taking.
        
        Returns:
            Dictionary with check results
        """
        results = {'closed': 0, 'partial': 0, 'errors': 0, 'details': []}
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            
            # Get current price
            current_price = self._get_current_price(symbol)
            
            if current_price <= 0:
                results['errors'] += 1
                results['details'].append({
                    'symbol': symbol,
                    'success': False,
                    'error': 'Invalid current price'
                })
                continue
            
            # Check for regular exit conditions
            stop_loss = position.get('stop_loss')
            target_price = position.get('target_price')
            position_side = position['side']
            
            stop_hit = False
            target_hit = False
            
            if stop_loss:
                if position_side == 'BUY' and current_price <= stop_loss:
                    stop_hit = True
                elif position_side == 'SELL' and current_price >= stop_loss:
                    stop_hit = True
            
            if target_price:
                if position_side == 'BUY' and current_price >= target_price:
                    target_hit = True
                elif position_side == 'SELL' and current_price <= target_price:
                    target_hit = True
            
            # Check for partial profit target
            partial_target = position.get('partial_target')
            partial_size = position.get('partial_size', 0.5)
            partial_hit = False
            
            if partial_target and not position.get('partial_exit_executed', False):
                if position_side == 'BUY' and current_price >= partial_target:
                    partial_hit = True
                elif position_side == 'SELL' and current_price <= partial_target:
                    partial_hit = True
            
            # Handle partial exit
            if partial_hit:
                try:
                    # Calculate partial quantity
                    quantity = position['quantity']
                    partial_quantity = quantity * partial_size
                    partial_quantity = max(1, round(partial_quantity))  # At least 1 share
                    
                    # Execute partial exit
                    if self.use_broker and self.broker_manager:
                        broker = self.broker_manager.get_broker()
                        
                        # Determine side for closing
                        side = 'SELL' if position_side == 'BUY' else 'BUY'
                        
                        # Place market order to close partial position
                        result = broker.place_order(
                            symbol=symbol,
                            side=side,
                            order_type='MARKET',
                            quantity=partial_quantity
                        )
                        
                        if 'error' in result:
                            results['errors'] += 1
                            results['details'].append({
                                'symbol': symbol,
                                'success': False,
                                'action': 'partial_exit',
                                'error': result.get('error', 'Unknown error')
                            })
                            continue
                    else:
                        # Simulate partial closing
                        result = {
                            'simulated': True,
                            'symbol': symbol,
                            'side': 'SELL' if position_side == 'BUY' else 'BUY',
                            'quantity': partial_quantity,
                            'price': current_price
                        }
                    
                    # Record partial exit
                    position['partial_exit_executed'] = True
                    position['partial_exit_price'] = current_price
                    position['partial_exit_time'] = datetime.now()
                    position['quantity'] = quantity - partial_quantity
                    
                    # Calculate partial profit
                    if position_side == 'BUY':
                        partial_profit = (current_price - position['entry_price']) * partial_quantity
                    else:
                        partial_profit = (position['entry_price'] - current_price) * partial_quantity
                    
                    position['partial_profit'] = partial_profit
                    
                    # Add to results
                    results['partial'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'success': True,
                        'action': 'partial_exit',
                        'quantity': partial_quantity,
                        'price': current_price,
                        'profit': partial_profit,
                        'result': result
                    })
                    
                    self.logger.info(f"Executed partial exit for {symbol} at {current_price}, profit: {partial_profit:.2f}")
                    
                    # Move stop to breakeven after partial exit
                    if not position.get('stop_moved_to_breakeven', False):
                        position['stop_loss'] = position['entry_price']
                        position['stop_moved_to_breakeven'] = True
                        
                        # Update stop in broker if using live trading
                        if self.use_broker and self.broker_manager:
                            try:
                                self._update_broker_stop(symbol, position['entry_price'])
                            except Exception as e:
                                self.logger.error(f"Error updating stop to breakeven: {e}")
                        
                        self.logger.info(f"Moved stop to breakeven for {symbol}")
                
                except Exception as e:
                    results['errors'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'success': False,
                        'action': 'partial_exit',
                        'error': str(e)
                    })
            
            # Handle full position exit
            if stop_hit or target_hit:
                reason = 'stop_loss' if stop_hit else 'target'
                
                try:
                    result = self.close_position(symbol, reason)
                    
                    if result.get('success', False):
                        results['closed'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'success': True,
                            'action': 'full_exit',
                            'reason': reason,
                            'exit_price': current_price,
                            'result': result
                        })
                    else:
                        results['errors'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'success': False,
                            'action': 'full_exit',
                            'error': result.get('error', 'Unknown error')
                        })
                except Exception as e:
                    results['errors'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'success': False,
                        'action': 'full_exit',
                        'error': str(e)
                    })
        
        # For day trading, check if we need to close positions at end of day
        now = datetime.now()
        market_close_approaching = now.hour >= 15 and now.minute >= 30  # 3:30 PM or later
        
        if market_close_approaching and self.positions:
            self.logger.info("Market close approaching, closing remaining positions")
            
            for symbol in list(self.positions.keys()):
                try:
                    result = self.close_position(symbol, 'end_of_day')
                    
                    if result.get('success', False):
                        results['closed'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'success': True,
                            'action': 'full_exit',
                            'reason': 'end_of_day',
                            'result': result
                        })
                    else:
                        results['errors'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'success': False,
                            'action': 'full_exit',
                            'error': result.get('error', 'Unknown error')
                        })
                except Exception as e:
                    results['errors'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'success': False,
                        'action': 'full_exit',
                        'error': str(e)
                    })
        
        return results


class HarmonicPatternStrategy(AdvancedFibonacciStrategy):
    """
    Strategy that focuses on harmonic price patterns and Fibonacci retracements.
    Identifies and trades harmonic patterns like Gartley, Butterfly, and Bat patterns.
    """
    
    def __init__(self, config_path: str = "config/harmonic_config.json"):
        """Initialize with harmonic pattern specific configuration."""
        super().__init__(config_path)
        
        # Harmonic pattern specific parameters
        self.pattern_types = self.config.get('pattern_types', ['gartley', 'butterfly', 'bat', 'crab'])
        self.min_pattern_quality = self.config.get('min_pattern_quality', 0.7)
        self.use_convergence = self.config.get('use_convergence', True)
        self.fibonacci_levels = self.config.get('fibonacci_levels', [0.382, 0.5, 0.618, 0.786, 1.272, 1.618])
        
        # Pattern detection tolerances
        self.pattern_tolerances = self.config.get('pattern_tolerances', {
            'gartley': 0.05,
            'butterfly': 0.05,
            'bat': 0.06,
            'crab': 0.06
        })
        
        # Initialize pattern detector
        self.pattern_detector = HarmonicPatternDetector(
            self.pattern_types,
            self.fibonacci_levels,
            self.pattern_tolerances
        )
    
    def analyze_symbol(self, symbol: str, exchange: str = None, interval: str = 'daily') -> ScanResult:
        """
        Override to add harmonic pattern detection.
        
        Args:
            symbol: Symbol to analyze
            exchange: Exchange code (optional)
            interval: Time interval
            
        Returns:
            ScanResult with analysis results
        """
        # First get the base analysis
        result = super().analyze_symbol(symbol, exchange, interval)
        
        if not result.success:
            return result
        
        # Get price data for pattern detection
        data = self.scanner.data_fetcher.get_data(
            symbol=symbol,
            exchange=exchange or self.config['general']['default_exchange'],
            interval=interval,
            lookback=200  # Need enough data for pattern detection
        )
        
        if data is None or data.empty:
            return result
        
        # Detect harmonic patterns
        patterns = self.pattern_detector.detect_patterns(data)
        
        # If we found patterns, analyze them
        if patterns:
            # Filter to completed or nearly completed patterns with good quality
            valid_patterns = [p for p in patterns if p['completion'] >= 0.9 and p['quality'] >= self.min_pattern_quality]
            
            if valid_patterns:
                # Sort by quality and completion
                valid_patterns.sort(key=lambda p: (p['completion'], p['quality']), reverse=True)
                
                # Get best pattern
                best_pattern = valid_patterns[0]
                
                # Check if pattern signal agrees with cycle signal
                cycle_signal = result.signal.get('signal', 'neutral')
                pattern_signal = best_pattern['pattern_type']
                
                # Harmonic patterns are bullish or bearish
                pattern_bullish = best_pattern['direction'] == 'bullish'
                cycle_bullish = 'buy' in cycle_signal
                
                # Store pattern info in result
                result.harmonic_patterns = valid_patterns
                result.best_pattern = best_pattern
                
                # If pattern and cycle signals agree, enhance confidence
                if (pattern_bullish and cycle_bullish) or (not pattern_bullish and not cycle_bullish):
                    if result.signal['confidence'] != 'high':
                        result.signal['confidence'] = 'high'
                    result.notes = result.notes + [f'Harmonic {best_pattern["pattern_type"]} pattern confirms cycle signal']
                else:
                    # If they disagree, reduce confidence or neutralize
                    if self.use_convergence:
                        # Require convergence - neutralize on disagreement
                        result.signal['signal'] = 'neutral'
                        result.signal['strength'] = 0
                        result.signal['confidence'] = 'low'
                        result.warnings = result.warnings + [f'Harmonic {best_pattern["pattern_type"]} pattern contradicts cycle signal']
                    else:
                        # Just reduce confidence on disagreement
                        result.signal['confidence'] = 'low'
                        result.warnings = result.warnings + [f'Harmonic {best_pattern["pattern_type"]} pattern contradicts cycle signal']
                
                # Use pattern for precise entry, stop, and target
                if best_pattern['completion'] >= 0.98:  # Pattern is complete
                    if 'position_guidance' in result.__dict__:
                        guidance = result.position_guidance
                        
                        # Use pattern D point as entry
                        guidance['entry_price'] = best_pattern['d_point']['price']
                        
                        # Use pattern stop and targets
                        if pattern_bullish:
                            # For bullish patterns
                            guidance['stop_loss'] = best_pattern['stop_level']
                            guidance['target_price'] = best_pattern['target_level']
                        else:
                            # For bearish patterns
                            guidance['stop_loss'] = best_pattern['stop_level']
                            guidance['target_price'] = best_pattern['target_level']
                        
                        # Recalculate risk-reward ratio
                        entry = guidance['entry_price']
                        stop = guidance['stop_loss']
                        target = guidance['target_price']
                        
                        if stop and target:
                            risk = abs(entry - stop)
                            reward = abs(target - entry)
                            
                            if risk > 0:
                                guidance['risk_reward_ratio'] = reward / risk
                        
                        result.position_guidance = guidance
        
        return result


class HarmonicPatternDetector:
    """Helper class for detecting harmonic price patterns."""
    
    def __init__(self, pattern_types, fibonacci_levels, tolerances):
        """Initialize the harmonic pattern detector."""
        self.pattern_types = pattern_types
        self.fibonacci_levels = fibonacci_levels
        self.tolerances = tolerances
        
    def detect_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect harmonic patterns in price data.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            List of detected patterns
        """
        # Detect swing highs and lows
        swing_points = self._detect_swing_points(data)
        
        # Find potential patterns
        patterns = []
        
        # We need at least 5 swing points to form a pattern (0, X, A, B, C)
        if len(swing_points) < 5:
            return patterns
        
        # Look for patterns in the most recent swing points
        for i in range(len(swing_points) - 4):
            # Get 5 consecutive swing points
            points = swing_points[i:i+5]
            
            # Check if these points form a valid pattern
            for pattern_type in self.pattern_types:
                pattern = self._check_pattern(points, pattern_type)
                if pattern and pattern['quality'] >= 0.6:  # Basic quality threshold
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_swing_points(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect significant swing highs and lows in price data.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            List of swing points
        """
        swing_points = []
        
        # Need at least 10 bars
        if len(data) < 10:
            return swing_points
        
        # Look for swing highs
        for i in range(2, len(data) - 2):
            # Check for swing high
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i-2] and
                data['high'].iloc[i] > data['high'].iloc[i+1] and
                data['high'].iloc[i] > data['high'].iloc[i+2]):
                
                swing_points.append({
                    'index': i,
                    'price': data['high'].iloc[i],
                    'date': data.index[i],
                    'type': 'high'
                })
            
            # Check for swing low
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i-2] and
                data['low'].iloc[i] < data['low'].iloc[i+1] and
                data['low'].iloc[i] < data['low'].iloc[i+2]):
                
                swing_points.append({
                    'index': i,
                    'price': data['low'].iloc[i],
                    'date': data.index[i],
                    'type': 'low'
                })
        
        # Sort by index
        swing_points.sort(key=lambda x: x['index'])
        
        return swing_points
    
    def _check_pattern(self, points: List[Dict], pattern_type: str) -> Optional[Dict]:
        """
        Check if given points form a specific harmonic pattern.
        
        Args:
            points: List of swing points
            pattern_type: Type of pattern to check
            
        Returns:
            Pattern information or None if not a valid pattern
        """
        # Ensure alternating high/low points
        if not self._check_alternating(points):
            return None
        
        # Extract X, A, B, C, D points
        x_point = points[0]
        a_point = points[1]
        b_point = points[2]
        c_point = points[3]
        d_point = points[4] if len(points) > 4 else None
        
        # Calculate ratios
        xab_ratio = self._calculate_ratio(x_point, a_point, b_point)
        abc_ratio = self._calculate_ratio(a_point, b_point, c_point)
        
        # D point may not exist yet (pattern not complete)
        completion = 1.0
        bcd_ratio = None
        xad_ratio = None
        
        if d_point:
            bcd_ratio = self._calculate_ratio(b_point, c_point, d_point)
            xad_ratio = self._calculate_ratio(x_point, a_point, d_point)
        else:
            # Estimate completion percentage
            completion = 0.75  # Pattern is 75% complete without D point
        
        # Check for pattern specific ratios
        if pattern_type == 'gartley':
            return self._check_gartley(x_point, a_point, b_point, c_point, d_point, 
                                     xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion)
        elif pattern_type == 'butterfly':
            return self._check_butterfly(x_point, a_point, b_point, c_point, d_point, 
                                       xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion)
        elif pattern_type == 'bat':
            return self._check_bat(x_point, a_point, b_point, c_point, d_point, 
                                 xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion)
        elif pattern_type == 'crab':
            return self._check_crab(x_point, a_point, b_point, c_point, d_point, 
                                  xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion)
        
        return None
    
    def _check_alternating(self, points: List[Dict]) -> bool:
        """Check if swing points alternate between highs and lows."""
        for i in range(1, len(points)):
            if points[i]['type'] == points[i-1]['type']:
                return False
        return True
    
    def _calculate_ratio(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """Calculate retracement ratio between three points."""
        range_12 = abs(point2['price'] - point1['price'])
        range_23 = abs(point3['price'] - point2['price'])
        
        if range_12 == 0:
            return 0
            
        return range_23 / range_12
    
    def _check_ratio_match(self, actual: float, expected: float, tolerance: float) -> float:
        """
        Check how closely a ratio matches an expected value within tolerance.
        Returns quality score between 0 and 1 (1 being perfect match).
        """
        if actual is None:
            return 0
            
        difference = abs(actual - expected)
        if difference <= tolerance:
            # Calculate quality as inverse of normalized difference
            quality = 1.0 - (difference / tolerance)
            return quality
        return 0
    
    def _check_gartley(self, x_point, a_point, b_point, c_point, d_point, 
                     xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion):
        """Check for Gartley pattern."""
        tolerance = self.tolerances.get('gartley', 0.05)
        
        # Expected Gartley ratios
        # XAB = 0.618
        # ABC = 0.382 or 0.886
        # BCD = 1.272 or 1.618
        # XAD = 0.786
        
        xab_quality = max(
            self._check_ratio_match(xab_ratio, 0.618, tolerance),
            0.0
        )
        
        abc_quality = max(
            self._check_ratio_match(abc_ratio, 0.382, tolerance),
            self._check_ratio_match(abc_ratio, 0.886, tolerance),
            0.0
        )
        
        # If D point exists, check BCD and XAD ratios
        if d_point:
            bcd_quality = max(
                self._check_ratio_match(bcd_ratio, 1.272, tolerance),
                self._check_ratio_match(bcd_ratio, 1.618, tolerance),
                0.0
            )
            
            xad_quality = max(
                self._check_ratio_match(xad_ratio, 0.786, tolerance),
                0.0
            )
        else:
            bcd_quality = 0.0
            xad_quality = 0.0
        
        # Calculate overall quality
        if d_point:
            quality = (xab_quality * 0.25 + abc_quality * 0.25 + 
                      bcd_quality * 0.25 + xad_quality * 0.25)
        else:
            # Without D point, only consider XAB and ABC
            quality = (xab_quality * 0.5 + abc_quality * 0.5)
        
        # Determine pattern direction
        if a_point['price'] > x_point['price']:
            # X to A is up, so pattern is bearish
            direction = 'bearish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] + (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] - (abs(a_point['price'] - d_point['price']) * 0.618)
            else:
                stop_level = None
                target_level = None
        else:
            # X to A is down, so pattern is bullish
            direction = 'bullish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] - (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] + (abs(a_point['price'] - d_point['price']) * 0.618)
            else:
                stop_level = None
                target_level = None
        
        if quality < 0.6:
            return None
            
        return {
            'pattern_type': 'gartley',
            'direction': direction,
            'quality': quality,
            'completion': completion,
            'x_point': x_point,
            'a_point': a_point,
            'b_point': b_point,
            'c_point': c_point,
            'd_point': d_point,
            'stop_level': stop_level,
            'target_level': target_level,
            'ratios': {
                'xab': xab_ratio,
                'abc': abc_ratio,
                'bcd': bcd_ratio,
                'xad': xad_ratio
            }
        }
    
    def _check_butterfly(self, x_point, a_point, b_point, c_point, d_point, 
                       xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion):
        """Check for Butterfly pattern."""
        tolerance = self.tolerances.get('butterfly', 0.05)
        
        # Expected Butterfly ratios
        # XAB = 0.786
        # ABC = 0.382 or 0.886
        # BCD = 1.618 or 2.618
        # XAD = 1.27 or 1.618
        
        xab_quality = max(
            self._check_ratio_match(xab_ratio, 0.786, tolerance),
            0.0
        )
        
        abc_quality = max(
            self._check_ratio_match(abc_ratio, 0.382, tolerance),
            self._check_ratio_match(abc_ratio, 0.886, tolerance),
            0.0
        )
        
        # If D point exists, check BCD and XAD ratios
        if d_point:
            bcd_quality = max(
                self._check_ratio_match(bcd_ratio, 1.618, tolerance),
                self._check_ratio_match(bcd_ratio, 2.618, tolerance),
                0.0
            )
            
            xad_quality = max(
                self._check_ratio_match(xad_ratio, 1.27, tolerance),
                self._check_ratio_match(xad_ratio, 1.618, tolerance),
                0.0
            )
        else:
            bcd_quality = 0.0
            xad_quality = 0.0
        
        # Calculate overall quality
        if d_point:
            quality = (xab_quality * 0.25 + abc_quality * 0.25 + 
                      bcd_quality * 0.25 + xad_quality * 0.25)
        else:
            # Without D point, only consider XAB and ABC
            quality = (xab_quality * 0.5 + abc_quality * 0.5)
        
        # Determine pattern direction
        if a_point['price'] > x_point['price']:
            # X to A is up, so pattern is bearish
            direction = 'bearish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] + (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] - (abs(a_point['price'] - d_point['price']) * 1.27)
            else:
                stop_level = None
                target_level = None
        else:
            # X to A is down, so pattern is bullish
            direction = 'bullish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] - (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] + (abs(a_point['price'] - d_point['price']) * 1.27)
            else:
                stop_level = None
                target_level = None
        
        if quality < 0.6:
            return None
            
        return {
            'pattern_type': 'butterfly',
            'direction': direction,
            'quality': quality,
            'completion': completion,
            'x_point': x_point,
            'a_point': a_point,
            'b_point': b_point,
            'c_point': c_point,
            'd_point': d_point,
            'stop_level': stop_level,
            'target_level': target_level,
            'ratios': {
                'xab': xab_ratio,
                'abc': abc_ratio,
                'bcd': bcd_ratio,
                'xad': xad_ratio
            }
        }
    
    def _check_bat(self, x_point, a_point, b_point, c_point, d_point, 
                 xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion):
        """Check for Bat pattern."""
        tolerance = self.tolerances.get('bat', 0.06)
        
        # Expected Bat ratios
        # XAB = 0.382 or 0.5
        # ABC = 0.382 or 0.886
        # BCD = 1.618 or 2.0
        # XAD = 0.886
        
        xab_quality = max(
            self._check_ratio_match(xab_ratio, 0.382, tolerance),
            self._check_ratio_match(xab_ratio, 0.5, tolerance),
            0.0
        )
        
        abc_quality = max(
            self._check_ratio_match(abc_ratio, 0.382, tolerance),
            self._check_ratio_match(abc_ratio, 0.886, tolerance),
            0.0
        )
        
        # If D point exists, check BCD and XAD ratios
        if d_point:
            bcd_quality = max(
                self._check_ratio_match(bcd_ratio, 1.618, tolerance),
                self._check_ratio_match(bcd_ratio, 2.0, tolerance),
                0.0
            )
            
            xad_quality = max(
                self._check_ratio_match(xad_ratio, 0.886, tolerance),
                0.0
            )
        else:
            bcd_quality = 0.0
            xad_quality = 0.0
        
        # Calculate overall quality
        if d_point:
            quality = (xab_quality * 0.25 + abc_quality * 0.25 + 
                      bcd_quality * 0.25 + xad_quality * 0.25)
        else:
            # Without D point, only consider XAB and ABC
            quality = (xab_quality * 0.5 + abc_quality * 0.5)
        
        # Determine pattern direction
        if a_point['price'] > x_point['price']:
            # X to A is up, so pattern is bearish
            direction = 'bearish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] + (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] - (abs(a_point['price'] - d_point['price']) * 0.886)
            else:
                stop_level = None
                target_level = None
        else:
            # X to A is down, so pattern is bullish
            direction = 'bullish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] - (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] + (abs(a_point['price'] - d_point['price']) * 0.886)
            else:
                stop_level = None
                target_level = None
        
        if quality < 0.6:
            return None
            
        return {
            'pattern_type': 'bat',
            'direction': direction,
            'quality': quality,
            'completion': completion,
            'x_point': x_point,
            'a_point': a_point,
            'b_point': b_point,
            'c_point': c_point,
            'd_point': d_point,
            'stop_level': stop_level,
            'target_level': target_level,
            'ratios': {
                'xab': xab_ratio,
                'abc': abc_ratio,
                'bcd': bcd_ratio,
                'xad': xad_ratio
            }
        }
    
    def _check_crab(self, x_point, a_point, b_point, c_point, d_point, 
                  xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion):
        """Check for Crab pattern."""
        tolerance = self.tolerances.get('crab', 0.06)
        
        # Expected Crab ratios
        # XAB = 0.382 or 0.618
        # ABC = 0.382 or 0.886
        # BCD = 2.618 or 3.618
        # XAD = 1.618
        
        xab_quality = max(
            self._check_ratio_match(xab_ratio, 0.382, tolerance),
            self._check_ratio_match(xab_ratio, 0.618, tolerance),
            0.0
        )
        
        abc_quality = max(
            self._check_ratio_match(abc_ratio, 0.382, tolerance),
            self._check_ratio_match(abc_ratio, 0.886, tolerance),
            0.0
        )
        
        # If D point exists, check BCD and XAD ratios
        if d_point:
            bcd_quality = max(
                self._check_ratio_match(bcd_ratio, 2.618, tolerance),
                self._check_ratio_match(bcd_ratio, 3.618, tolerance),
                0.0
            )
            
            xad_quality = max(
                self._check_ratio_match(xad_ratio, 1.618, tolerance),
                0.0
            )
        else:
            bcd_quality = 0.0
            xad_quality = 0.0
        
        # Calculate overall quality
        if d_point:
            quality = (xab_quality * 0.25 + abc_quality * 0.25 + 
                      bcd_quality * 0.25 + xad_quality * 0.25)
        else:
            # Without D point, only consider XAB and ABC
            quality = (xab_quality * 0.5 + abc_quality * 0.5)
        
        # Determine pattern direction
        if a_point['price'] > x_point['price']:
            # X to A is up, so pattern is bearish
            direction = 'bearish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] + (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] - (abs(a_point['price'] - d_point['price']) * 1.618)
            else:
                stop_level = None
                target_level = None
        else:
            # X to A is down, so pattern is bullish
            direction = 'bullish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] - (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] + (abs(a_point['price'] - d_point['price']) * 1.618)
            else:
                stop_level = None
                target_level = None
        
        if quality < 0.6:
            return None
            
        return {
            'pattern_type': 'crab',
            'direction': direction,
            'quality': quality,
            'completion': completion,
            'x_point': x_point,
            'a_point': a_point,
            'b_point': b_point,
            'c_point': c_point,
            'd_point': d_point,
            'stop_level': stop_level,
            'target_level': target_level,
            'ratios': {
                'xab': xab_ratio,
                'abc': abc_ratio,
                'bcd': bcd_ratio,
                'xad': xad_ratio
            }
        }
