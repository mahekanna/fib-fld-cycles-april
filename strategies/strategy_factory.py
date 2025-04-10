"""
Strategy Factory for Fibonacci Cycles Trading System

This module provides a centralized registry and factory for all trading strategies
in the system. It allows for easy instantiation and registration of strategies.
"""

import logging
from typing import Dict, Type, List, Any, Optional

from strategies.base_strategy import BaseStrategy
from strategies.rapid_cycle_fld_strategy import RapidCycleFLDStrategy
from strategies.multi_cycle_confluence_strategy import MultiCycleConfluenceStrategy
from strategies.turning_point_anticipation_strategy import TurningPointAnticipationStrategy
from strategies.cycle_phase_trading_strategy import CyclePhaseStrategy

logger = logging.getLogger(__name__)

class StrategyFactory:
    """
    Factory for creating and managing trading strategies.
    
    This class maintains a registry of available strategies and provides
    methods to instantiate and retrieve them by name.
    """
    
    def __init__(self):
        """Initialize the strategy factory with registered strategies."""
        self._strategies = {}
        
        # Register all available strategies
        self.register_strategy("rapid_cycle_fld", RapidCycleFLDStrategy)
        self.register_strategy("multi_cycle_confluence", MultiCycleConfluenceStrategy)
        self.register_strategy("turning_point_anticipation", TurningPointAnticipationStrategy)
        self.register_strategy("cycle_phase", CyclePhaseStrategy)
        
        logger.info(f"Strategy factory initialized with {len(self._strategies)} strategies")
    
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy with the factory.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        self._strategies[name] = strategy_class
        logger.debug(f"Registered strategy: {name}")
    
    def create_strategy(self, name: str, config: Dict[str, Any]) -> Optional[BaseStrategy]:
        """
        Create a strategy instance by name.
        
        Args:
            name: Strategy name
            config: Configuration dictionary
            
        Returns:
            Strategy instance or None if not found
        """
        strategy_class = self._strategies.get(name)
        
        if not strategy_class:
            logger.error(f"Strategy not found: {name}")
            return None
        
        try:
            strategy = strategy_class(config)
            logger.debug(f"Created strategy instance: {name}")
            return strategy
        except Exception as e:
            logger.error(f"Error creating strategy '{name}': {e}")
            return None
    
    def get_available_strategies(self) -> List[str]:
        """
        Get a list of available strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get a strategy class by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy class or None if not found
        """
        return self._strategies.get(name)


# Create singleton instance
strategy_factory = StrategyFactory()


def get_strategy(name: str, config: Dict[str, Any]) -> Optional[BaseStrategy]:
    """
    Get a strategy instance by name (convenience function).
    
    Args:
        name: Strategy name
        config: Configuration dictionary
        
    Returns:
        Strategy instance or None if not found
    """
    return strategy_factory.create_strategy(name, config)


def get_available_strategies() -> List[str]:
    """
    Get a list of available strategy names (convenience function).
    
    Returns:
        List of strategy names
    """
    return strategy_factory.get_available_strategies()


def get_strategy_by_name(name: str) -> Optional[Type[BaseStrategy]]:
    """
    Get a strategy class by name (convenience function).
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy class or None if not found
    """
    return strategy_factory.get_strategy_class(name)