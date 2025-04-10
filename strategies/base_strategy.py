"""
Base Strategy Module for Fibonacci Cycles Trading System

This module defines the BaseStrategy abstract class that all strategies must implement.
It provides the common interface and utilities for cycle-based strategies.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies in the Fibonacci Cycles System.
    
    This class defines the common interface that all strategies must implement,
    including methods for signal generation, position management, and backtest execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy with configuration parameters.
        
        Args:
            config: Configuration dictionary with strategy parameters
        """
        self.config = config
        self.name = self.__class__.__name__
        
        # Common parameters with defaults
        self.risk_per_trade = config.get('risk_per_trade', 1.0)  # % of account
        self.max_positions = config.get('max_positions', 5)
        self.use_trailing_stop = config.get('use_trailing_stop', True)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 2.0)
        self.take_profit_multiplier = config.get('take_profit_multiplier', 2.0)
        
        # State tracking
        self.positions = {}
        self.trade_history = []
        
        # Market metadata
        self.market_regime = "NEUTRAL"
        
        # Setup logging
        log_level = config.get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level))

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, cycles: List[int], 
                     fld_crossovers: List[Dict], cycle_states: List[Dict]) -> Dict:
        """
        Generate trading signals based on cycle analysis.
        
        Args:
            data: Price data DataFrame with OHLCV columns
            cycles: List of detected cycle lengths
            fld_crossovers: List of detected FLD crossovers
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Dictionary with signal information
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, account_value: float, signal_dict: Dict, 
                            current_price: float, stop_price: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            account_value: Current account value
            signal_dict: Signal information dictionary
            current_price: Current market price
            stop_price: Stop loss price
            
        Returns:
            Position size (quantity to trade)
        """
        pass
    
    @abstractmethod
    def set_stop_loss(self, data: pd.DataFrame, signal_dict: Dict, 
                   entry_price: float, direction: str) -> float:
        """
        Calculate stop loss level based on strategy rules.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price level
        """
        pass
    
    @abstractmethod
    def set_take_profit(self, data: pd.DataFrame, signal_dict: Dict, 
                     entry_price: float, stop_price: float, direction: str) -> float:
        """
        Calculate take profit level based on strategy rules.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            stop_price: Stop loss price
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Take profit price level
        """
        pass
    
    def update_stops(self, data: pd.DataFrame, positions: Dict) -> Dict:
        """
        Update stop loss levels for open positions.
        
        Args:
            data: Current price data
            positions: Dictionary of open positions
            
        Returns:
            Updated positions dictionary
        """
        # Default implementation uses trailing stops
        if not self.use_trailing_stop:
            return positions
            
        current_price = data['close'].iloc[-1]
        updated_positions = positions.copy()
        
        for pos_id, position in updated_positions.items():
            if position['direction'] == 'long':
                # For long positions, move stop up (never down)
                new_stop = current_price * (1 - self.trailing_stop_pct/100)
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
                    logger.debug(f"Updated trailing stop to {new_stop:.2f} for position {pos_id}")
            else:
                # For short positions, move stop down (never up)
                new_stop = current_price * (1 + self.trailing_stop_pct/100)
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
                    logger.debug(f"Updated trailing stop to {new_stop:.2f} for position {pos_id}")
                    
        return updated_positions
    
    def calculate_cycle_alignment(self, cycle_states: List[Dict]) -> float:
        """
        Calculate alignment score of multiple cycles.
        
        Args:
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Alignment score between -1.0 (completely bearish) and 1.0 (completely bullish)
        """
        if not cycle_states:
            return 0.0
            
        bullish_count = sum(1 for state in cycle_states if state.get('is_bullish', False))
        bearish_count = len(cycle_states) - bullish_count
        
        # Calculate weighted alignment
        alignment = (bullish_count - bearish_count) / len(cycle_states)
        return alignment
    
    def detect_fld_crossovers(self, data: pd.DataFrame, cycle_length: int) -> List[Dict]:
        """
        Detect FLD (Future Line of Demarcation) crossovers.
        
        Args:
            data: Price data DataFrame
            cycle_length: Cycle length to calculate FLD for
            
        Returns:
            List of crossover dictionaries
        """
        if len(data) < cycle_length * 2:
            return []
            
        # Calculate FLD (half cycle displacement)
        half_cycle = cycle_length // 2
        
        # Price crossing above/below the displaced price
        price = data['close'].values
        fld_values = np.concatenate([np.full(half_cycle, np.nan), price[:-half_cycle]])
        
        # Detect crossovers
        crossovers = []
        
        for i in range(1, len(data)):
            if i < half_cycle:
                continue
                
            # Price crossing above FLD
            if price[i] > fld_values[i] and price[i-1] <= fld_values[i-1]:
                crossovers.append({
                    'index': i,
                    'direction': 'bullish',
                    'price': price[i],
                    'fld_value': fld_values[i],
                    'date': data.index[i]
                })
            
            # Price crossing below FLD
            elif price[i] < fld_values[i] and price[i-1] >= fld_values[i-1]:
                crossovers.append({
                    'index': i,
                    'direction': 'bearish',
                    'price': price[i],
                    'fld_value': fld_values[i],
                    'date': data.index[i]
                })
                
        return crossovers
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range for risk management.
        
        Args:
            data: Price data DataFrame
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(data) < period + 1:
            # Fallback if not enough data
            return (data['high'].iloc[-1] - data['low'].iloc[-1])
            
        # Calculate True Range
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift(1))
        tr3 = abs(data['low'] - data['close'].shift(1))
        
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(period).mean().iloc[-1]
        return atr
    
    def log_signal(self, signal_dict: Dict) -> None:
        """Log the generated signal for debugging."""
        signal_type = signal_dict.get('signal', 'neutral')
        strength = signal_dict.get('strength', 0)
        confidence = signal_dict.get('confidence', 'low')
        
        logger.info(f"Signal generated: {signal_type}, strength: {strength:.2f}, confidence: {confidence}")