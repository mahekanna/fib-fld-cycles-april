"""
Backtesting package for the Fibonacci Cycles Trading System.

This package provides tools for backtesting trading strategies
against historical market data.
"""

# Import main classes for easier access
try:
    from .backtesting_framework import BacktestEngine, BacktestParameters
except ImportError:
    # This prevents errors when the module is imported but classes aren't available yet
    pass

__all__ = ['BacktestEngine', 'BacktestParameters']