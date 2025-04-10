from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import numpy as np
import json


def get_default_config() -> Dict:
    """
    Returns a default configuration dictionary for the system.
    
    Returns:
        Dictionary with default configuration values
    """
    return {
        "general": {
            "default_exchange": "NSE",
            "default_source": "tradingview",
            "symbols_file_path": "config/symbols.json"
        },
        "data": {
            "cache_dir": "data/cache",
            "cache_expiry": {
                "1m": 1,  # days
                "5m": 1,
                "15m": 1,
                "30m": 1,
                "1h": 7,
                "4h": 7,
                "daily": 30,
                "weekly": 90,
                "monthly": 90
            }
        },
        "tradingview": {
            "username": "",
            "password": ""
        },
        "analysis": {
            "min_period": 10,
            "max_period": 250,
            "fib_cycles": [21, 34, 55, 89, 144, 233],
            "power_threshold": 0.2,
            "cycle_tolerance": 0.15,
            "detrend_method": "diff",
            "window_function": "hanning",
            "gap_threshold": 0.01,
            "crossover_lookback": 5
        },
        "scanner": {
            "default_symbols": ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"],
            "default_exchange": "NSE",
            "default_interval": "daily",
            "default_lookback": 1000,
            "price_source": "close",
            "num_cycles": 3,
            "filter_signal": None,
            "min_confidence": None,
            "min_alignment": 0.6,
            "ranking_factor": "strength"
        },
        "backtest": {
            "default_symbols": ["NIFTY", "BANKNIFTY"],
            "start_date": "2018-01-01",
            "end_date": None,  # None for current date
            "initial_capital": 100000,
            "position_size": 0.1,
            "commission_rate": 0.0005,
            "slippage": 0.0002
        },
        "performance": {
            "max_workers": 5
        },
        "visualization": {
            "theme": "dark",
            "default_chart_height": 800,
            "default_chart_width": 1200,
            "color_palette": {
                "price": "#2962FF",
                "up_candle": "#26A69A",
                "down_candle": "#EF5350",
                "volume": "#B2B5BE",
                "cycle_21": "#FFC107",
                "cycle_34": "#FF9800",
                "cycle_55": "#FF5722",
                "cycle_89": "#E91E63",
                "cycle_144": "#9C27B0",
                "cycle_233": "#673AB7",
                "fld": "#4CAF50",
                "buy_signal": "#00E676",
                "sell_signal": "#FF3D00",
                "background": "#1E1E1E",
                "grid": "#363636",
                "text": "#E0E0E0"
            }
        },
        "telegram": {
            "bot_token": "",
            "chat_id": "",
            "enable_notifications": False,
            "notification_types": ["strong_buy", "strong_sell"]
        },
        "web": {
            "dashboard_host": "127.0.0.1",
            "dashboard_port": 8501,
            "refresh_interval": 60  # minutes
        }
    }


@dataclass
class ScanParameters:
    """
    Parameters for cycle scanning and analysis.
    """
    symbol: str
    exchange: str
    interval: str = "daily"
    lookback: int = 1000
    num_cycles: int = 3
    price_source: str = "close"
    generate_chart: bool = False
    custom_data: Optional[Any] = None


@dataclass
class CycleState:
    """
    State information for a detected cycle.
    """
    cycle_length: int
    is_bullish: bool
    days_since_crossover: Optional[int] = None
    price_to_fld_ratio: float = 1.0
    fld_value: float = 0.0
    price_value: float = 0.0
    recent_crossover: Optional[Dict] = None


@dataclass
class ScanResult:
    """
    Complete result of a cycle analysis.
    """
    # Basic information
    symbol: str
    exchange: str
    interval: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0
    
    # Price information
    price: float = 0.0
    
    # Cycle information
    detected_cycles: List[int] = field(default_factory=list)
    cycle_powers: Dict[int, float] = field(default_factory=dict)
    cycle_states: List[Dict] = field(default_factory=list)
    harmonic_relationships: Dict[str, Dict] = field(default_factory=dict)
    
    # Signal information
    signal: Dict = field(default_factory=dict)
    position_guidance: Dict = field(default_factory=dict)
    
    # Chart image (if generated)
    chart_image: Optional[Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'interval': self.interval,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error': self.error,
            'execution_time': self.execution_time,
            'price': self.price,
            'detected_cycles': self.detected_cycles,
            'cycle_powers': self.cycle_powers,
            'cycle_states': self.cycle_states,
            'harmonic_relationships': self.harmonic_relationships,
            'signal': self.signal,
            'position_guidance': self.position_guidance
        }
        
        # Chart image can't be serialized directly
        if self.chart_image is not None:
            result['has_chart'] = True
        else:
            result['has_chart'] = False
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScanResult':
        """Create from dictionary."""
        # Convert timestamp string to datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            
        # Handle chart image
        data['chart_image'] = None
            
        return cls(**data)


@dataclass
class BacktestTrade:
    """
    Information about a backtest trade.
    """
    symbol: str
    direction: str  # 'long' or 'short'
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: float
    profit_loss: float
    profit_loss_pct: float
    exit_reason: str


@dataclass
class ReportParameters:
    """
    Parameters for report generation.
    """
    title: str = "Market Analysis Report"
    subtitle: Optional[str] = None
    author: Optional[str] = None
    date: datetime = field(default_factory=datetime.now)
    symbols: List[str] = field(default_factory=list)
    intervals: List[str] = field(default_factory=list)
    include_charts: bool = True
    include_signals: bool = True
    include_cycles: bool = True
    include_harmonic_relationships: bool = True
    include_market_regime: bool = True
    output_format: str = "html"  # 'html', 'pdf', 'markdown'
    template: Optional[str] = None


@dataclass
class SignalSummary:
    """
    Summary of trading signals across multiple symbols.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    interval: str = "daily"
    
    # Signal counts by type
    buy_signals: int = 0
    sell_signals: int = 0
    neutral_signals: int = 0
    
    # Signal details
    signals: Dict[str, Dict] = field(default_factory=dict)
    
    # Market breadth indicators
    breadth: Dict = field(default_factory=dict)
    
    def add_signal(self, symbol: str, signal: Dict) -> None:
        """Add a signal to the summary."""
        self.signals[symbol] = signal
        
        # Update signal counts
        signal_type = signal.get('signal', 'neutral')
        if 'buy' in signal_type:
            self.buy_signals += 1
        elif 'sell' in signal_type:
            self.sell_signals += 1
        else:
            self.neutral_signals += 1
    
    def calculate_breadth(self) -> None:
        """Calculate market breadth indicators."""
        total_signals = len(self.signals)
        if total_signals == 0:
            return
        
        self.breadth = {
            'buy_percentage': (self.buy_signals / total_signals) * 100,
            'sell_percentage': (self.sell_signals / total_signals) * 100,
            'neutral_percentage': (self.neutral_signals / total_signals) * 100,
            'bullish_bias': True if self.buy_signals > self.sell_signals else False,
            'strength': (self.buy_signals - self.sell_signals) / total_signals
        }


@dataclass
class OptimizationResult:
    """
    Result of parameter optimization.
    """
    parameters: Dict
    metrics: Dict
    trades: List[Dict]
    equity_curve: List[Dict]
    
    # Performance metrics
    win_rate: float = 0.0
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    
    def __post_init__(self):
        """Extract key metrics after initialization."""
        if self.metrics:
            self.win_rate = self.metrics.get('win_rate', 0.0)
            self.profit_loss = self.metrics.get('profit_loss', 0.0)
            self.profit_loss_pct = self.metrics.get('profit_loss_pct', 0.0)
            self.max_drawdown_pct = self.metrics.get('max_drawdown_pct', 0.0)
            self.sharpe_ratio = self.metrics.get('sharpe_ratio', 0.0)
            self.profit_factor = self.metrics.get('profit_factor', 0.0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'parameters': self.parameters,
            'metrics': self.metrics,
            'win_rate': self.win_rate,
            'profit_loss': self.profit_loss,
            'profit_loss_pct': self.profit_loss_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor,
            'trades_count': len(self.trades)
        }


@dataclass
class MarketRegimeState:
    """
    Current market regime and state.
    """
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    regime: str = "UNKNOWN"
    regime_value: float = 0.0
    duration: int = 0
    trend_strength: float = 0.0
    volatility: float = 0.0
    oscillator_value: float = 0.0
    is_trending: bool = False
    is_volatile: bool = False
    indicators: Dict = field(default_factory=dict)
    trading_rules: Dict = field(default_factory=dict)


@dataclass
class AlertConfiguration:
    """
    Configuration for alerts and notifications.
    """
    enabled: bool = True
    telegram_enabled: bool = False
    email_enabled: bool = False
    
    # Alert types
    signal_alerts: bool = True
    crossover_alerts: bool = True
    regime_change_alerts: bool = True
    
    # Alert thresholds
    min_signal_strength: float = 0.3
    min_signal_alignment: float = 0.6
    
    # Alert destinations
    telegram_chat_id: Optional[str] = None
    email_address: Optional[str] = None
    
    # Alert formatting
    include_chart: bool = True
    include_position_guidance: bool = True
