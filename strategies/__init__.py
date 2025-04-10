"""
Fibonacci Cycles System Advanced Trading Strategies

This package contains implementations of advanced trading strategies for the
Fibonacci Cycles System as documented in the Advanced Trading Strategies Documentation.

Available strategies:
- Rapid Cycle FLD Strategy
- Multi-Cycle Confluence Strategy
- Turning Point Anticipation Strategy 
- Cycle Phase Trading Strategy

See README.md for usage documentation.
"""

# Version
__version__ = '1.0.0'

# Import factory functions for easy access
from strategies.strategy_factory import (
    get_strategy,
    get_available_strategies,
    get_strategy_by_name
)

# Import strategy classes for direct access
from strategies.rapid_cycle_fld_strategy import RapidCycleFLDStrategy
from strategies.multi_cycle_confluence_strategy import MultiCycleConfluenceStrategy
from strategies.turning_point_anticipation_strategy import TurningPointAnticipationStrategy
from strategies.cycle_phase_trading_strategy import CyclePhaseStrategy

# List of available strategies
__all__ = [
    'get_strategy',
    'get_available_strategies',
    'get_strategy_by_name',
    'RapidCycleFLDStrategy',
    'MultiCycleConfluenceStrategy',
    'TurningPointAnticipationStrategy',
    'CyclePhaseStrategy'
]