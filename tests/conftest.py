import pytest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Create a simple ScanResult class for tests
class ScanResult:
    """Mock ScanResult class for testing"""
    def __init__(
        self, symbol, exchange, interval, timestamp, price,
        detected_cycles, cycle_powers, cycle_states, 
        harmonic_relationships, signal, position_guidance,
        price_data, success, error, harmonic_patterns=None,
        backtest_results=None
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.interval = interval
        self.timestamp = timestamp
        self.price = price
        self.detected_cycles = detected_cycles
        self.cycle_powers = cycle_powers
        self.cycle_states = cycle_states
        self.harmonic_relationships = harmonic_relationships
        self.signal = signal
        self.position_guidance = position_guidance
        self.price_data = price_data
        self.success = success
        self.error = error
        self.harmonic_patterns = harmonic_patterns or {}
        self.backtest_results = backtest_results
    
    def to_dict(self):
        """Convert result to dictionary"""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'interval': self.interval,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'detected_cycles': self.detected_cycles,
            'cycle_powers': self.cycle_powers,
            'cycle_states': self.cycle_states,
            'harmonic_relationships': self.harmonic_relationships,
            'signal': self.signal,
            'position_guidance': self.position_guidance,
            'success': self.success,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create ScanResult from dictionary"""
        result = cls(
            symbol=data['symbol'],
            exchange=data['exchange'],
            interval=data['interval'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            price=data['price'],
            detected_cycles=data['detected_cycles'],
            cycle_powers=data['cycle_powers'],
            cycle_states=data['cycle_states'],
            harmonic_relationships=data['harmonic_relationships'],
            signal=data['signal'],
            position_guidance=data['position_guidance'],
            price_data=None,  # Price data doesn't get serialized
            success=data['success'],
            error=data['error']
        )
        return result


@pytest.fixture
def mock_scan_result():
    """Create a mock ScanResult for testing"""
    # Create some price data
    dates = [datetime.now() - timedelta(days=x) for x in range(100, 0, -1)]
    prices = np.sin(np.linspace(0, 10, 100)) * 10 + 100
    price_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices - 1,
        'high': prices + 1,
        'low': prices - 2,
        'volume': np.random.rand(100) * 1000
    })
    
    # Create a mock ScanResult
    result = ScanResult(
        symbol='TEST',
        exchange='TEST',
        interval='daily',
        timestamp=datetime.now(),
        price=prices[-1],
        detected_cycles=[21, 55, 89],
        cycle_powers={21: 0.8, 55: 0.6, 89: 0.4},
        cycle_states=[
            {'cycle_length': 21, 'is_bullish': True, 'days_since_crossover': 5, 'price_to_fld_ratio': 1.05},
            {'cycle_length': 55, 'is_bullish': False, 'days_since_crossover': 10, 'price_to_fld_ratio': 0.95},
            {'cycle_length': 89, 'is_bullish': True, 'days_since_crossover': 3, 'price_to_fld_ratio': 1.02}
        ],
        harmonic_relationships={'21:55': {'ratio': 0.382, 'harmonic': 'Fibonacci', 'precision': 98.5}},
        signal={'signal': 'buy', 'strength': 0.75, 'confidence': 'high', 'alignment': 0.82},
        position_guidance={
            'entry_price': 100.0,
            'stop_loss': 95.0,
            'target_price': 110.0,
            'risk_percentage': 5.0,
            'target_percentage': 10.0,
            'risk_reward_ratio': 2.0
        },
        price_data=price_data,
        success=True,
        error=None
    )
    
    # Add harmonic pattern data
    result.harmonic_patterns = {
        'gartley': {
            'pattern_type': 'gartley',
            'quality': 0.85,
            'points': {
                'X': {'price': 90.0, 'timestamp': dates[20]},
                'A': {'price': 110.0, 'timestamp': dates[35]},
                'B': {'price': 100.0, 'timestamp': dates[50]},
                'C': {'price': 105.0, 'timestamp': dates[65]},
                'D': {'price': 97.0, 'timestamp': dates[80]}
            },
            'fibonacci_ratios': {
                'XA_retracement': 0.618,
                'AB_retracement': 0.382,
                'BC_extension': 1.272
            },
            'entry': 97.5,
            'stop_loss': 94.0,
            'targets': [105.0, 110.0]
        }
    }
    
    # Add backtest data
    result.backtest_results = {
        'strategy_name': 'Swing Trading',
        'trades': [
            {'entry_date': dates[80], 'exit_date': dates[70], 'entry_price': 98.0, 'exit_price': 103.0, 'profit_pct': 5.1, 'type': 'long'},
            {'entry_date': dates[65], 'exit_date': dates[55], 'entry_price': 104.0, 'exit_price': 99.0, 'profit_pct': -4.8, 'type': 'long'},
            {'entry_date': dates[50], 'exit_date': dates[40], 'entry_price': 96.0, 'exit_price': 101.0, 'profit_pct': 5.2, 'type': 'long'},
            {'entry_date': dates[35], 'exit_date': dates[25], 'entry_price': 103.0, 'exit_price': 98.0, 'profit_pct': -4.9, 'type': 'long'},
            {'entry_date': dates[20], 'exit_date': dates[10], 'entry_price': 97.0, 'exit_price': 105.0, 'profit_pct': 8.2, 'type': 'long'},
        ],
        'performance_metrics': {
            'total_trades': 5,
            'win_rate': 0.6,
            'avg_profit_pct': 1.76,
            'max_drawdown_pct': 9.8,
            'sharpe_ratio': 1.2,
            'profit_factor': 2.1
        },
        'equity_curve': pd.Series(np.cumsum(np.array([0, 5.1, -4.8, 5.2, -4.9, 8.2])), index=[dates[90]] + [t['exit_date'] for t in [
            {'entry_date': dates[80], 'exit_date': dates[70], 'entry_price': 98.0, 'exit_price': 103.0, 'profit_pct': 5.1, 'type': 'long'},
            {'entry_date': dates[65], 'exit_date': dates[55], 'entry_price': 104.0, 'exit_price': 99.0, 'profit_pct': -4.8, 'type': 'long'},
            {'entry_date': dates[50], 'exit_date': dates[40], 'entry_price': 96.0, 'exit_price': 101.0, 'profit_pct': 5.2, 'type': 'long'},
            {'entry_date': dates[35], 'exit_date': dates[25], 'entry_price': 103.0, 'exit_price': 98.0, 'profit_pct': -4.9, 'type': 'long'},
            {'entry_date': dates[20], 'exit_date': dates[10], 'entry_price': 97.0, 'exit_price': 105.0, 'profit_pct': 8.2, 'type': 'long'},
        ]])
    }
    
    return result


@pytest.fixture
def mock_scan_results():
    """Create multiple mock ScanResults for testing"""
    results = []
    
    for i in range(5):
        # Create some price data
        dates = [datetime.now() - timedelta(days=x) for x in range(100, 0, -1)]
        prices = np.sin(np.linspace(0, 10, 100)) * 10 + 100 + (i * 5)
        price_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': prices - 1,
            'high': prices + 1,
            'low': prices - 2,
            'volume': np.random.rand(100) * 1000
        })
        
        # Signal type alternates between buy and sell
        signal_type = "buy" if i % 2 == 0 else "sell"
        
        # Create a mock ScanResult
        result = ScanResult(
            symbol=f'TEST{i}',
            exchange='TEST',
            interval='daily',
            timestamp=datetime.now(),
            price=prices[-1],
            detected_cycles=[21, 55, 89],
            cycle_powers={21: 0.8 - (i * 0.1), 55: 0.6 - (i * 0.05), 89: 0.4 - (i * 0.02)},
            cycle_states=[
                {'cycle_length': 21, 'is_bullish': i % 2 == 0, 'days_since_crossover': 5 + i, 'price_to_fld_ratio': 1.05 - (i * 0.01)},
                {'cycle_length': 55, 'is_bullish': i % 3 == 0, 'days_since_crossover': 10 + i, 'price_to_fld_ratio': 0.95 + (i * 0.01)},
                {'cycle_length': 89, 'is_bullish': i % 4 == 0, 'days_since_crossover': 3 + i, 'price_to_fld_ratio': 1.02 - (i * 0.005)}
            ],
            harmonic_relationships={'21:55': {'ratio': 0.382, 'harmonic': 'Fibonacci', 'precision': 98.5 - (i * 1.0)}},
            signal={'signal': signal_type, 'strength': 0.75 - (i * 0.1), 'confidence': 'high' if i < 2 else ('medium' if i < 4 else 'low'), 'alignment': 0.82 - (i * 0.05)},
            position_guidance={
                'entry_price': 100.0 + (i * 5),
                'stop_loss': 95.0 + (i * 5),
                'target_price': 110.0 + (i * 5),
                'risk_percentage': 5.0,
                'target_percentage': 10.0,
                'risk_reward_ratio': 2.0 - (i * 0.2)
            },
            price_data=price_data,
            success=True,
            error=None
        )
        
        results.append(result)
    
    return results