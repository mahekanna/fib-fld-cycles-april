"""
Rapid Cycle FLD Strategy Module

Implements the Rapid Cycle FLD Strategy for intraday trading, focusing on the shortest
detected cycle (typically 21) and crossovers with the Future Line of Demarcation (FLD).
This strategy is optimized for quick entries and exits, making it suitable for intraday
trading on 15-minute and 1-hour timeframes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class RapidCycleFLDStrategy(BaseStrategy):
    """
    Rapid Cycle FLD Strategy for intraday trading.
    
    This strategy focuses on the shortest detected cycle and FLD crossovers,
    with quick entries and exits. It's designed for high-frequency trading
    with tight risk management.
    
    Key features:
    - Uses shortest cycle (typically 21) for primary signals
    - Optimal for 15-minute and 1-hour timeframes
    - Implements tight stop-losses at 0.3 x cycle amplitude
    - Targets 1:2 risk-reward minimum
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Rapid Cycle FLD Strategy.
        
        Args:
            config: Configuration dictionary with strategy parameters
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.min_alignment_threshold = config.get('min_alignment_threshold', 0.7)
        self.stop_loss_factor = config.get('stop_loss_factor', 0.3)
        self.take_profit_factor = config.get('take_profit_factor', 0.6)  # 2x the risk
        self.min_signal_strength = config.get('min_signal_strength', 0.4)
        
        logger.info(f"Initialized {self.name} with alignment threshold: {self.min_alignment_threshold}, "
                   f"stop loss factor: {self.stop_loss_factor}")
    
    def generate_signal(self, data: pd.DataFrame, cycles: List[int], 
                     fld_crossovers: List[Dict], cycle_states: List[Dict]) -> Dict:
        """
        Generate trading signals based on rapid cycle FLD crossovers.
        
        Args:
            data: Price data DataFrame with OHLCV columns
            cycles: List of detected cycle lengths
            fld_crossovers: List of detected FLD crossovers
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Dictionary with signal information
        """
        # If no cycles detected, return neutral signal
        if not cycles or not cycle_states:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'description': "No valid cycles detected"
            }
        
        # Focus on shortest detected cycle for rapid signals
        primary_cycle = min(cycles)
        logger.debug(f"Using primary cycle of {primary_cycle} for rapid strategy")
        
        # If no fld_crossovers provided, calculate them
        if not fld_crossovers:
            fld_crossovers = self.detect_fld_crossovers(data, primary_cycle)
        
        # Check for recent crossovers (last 3 bars)
        last_3_dates = data.index[-3:]
        recent_crossovers = [c for c in fld_crossovers if c['date'] in last_3_dates]
        
        # If no recent crossovers, return neutral signal
        if not recent_crossovers:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'description': "No recent FLD crossovers detected"
            }
        
        # Get the most recent crossover
        latest_crossover = recent_crossovers[-1]
        crossover_direction = latest_crossover['direction']
        
        # Calculate cycle alignment
        alignment = self.calculate_cycle_alignment(cycle_states)
        
        # Signal type based on crossover direction and alignment
        signal_type = 'neutral'
        strength = 0
        confidence = 'low'
        
        # Generate signals when FLD crossover occurs with good alignment
        if crossover_direction == 'bullish' and alignment > self.min_alignment_threshold:
            signal_type = 'buy'
            strength = min(1.0, alignment)
            confidence = 'high' if alignment > 0.85 else 'medium'
        elif crossover_direction == 'bearish' and alignment < -self.min_alignment_threshold:
            signal_type = 'sell'
            strength = max(-1.0, -alignment)
            confidence = 'high' if alignment < -0.85 else 'medium'
        
        # Apply strength requirement
        if abs(strength) < self.min_signal_strength:
            signal_type = 'neutral'
            strength = 0
            confidence = 'low'
        
        signal = {
            'signal': signal_type,
            'strength': strength,
            'confidence': confidence,
            'alignment': alignment,
            'primary_cycle': primary_cycle,
            'recent_crossover': latest_crossover,
            'description': f"{signal_type.capitalize()} signal from {primary_cycle}-period cycle FLD crossover"
        }
        
        self.log_signal(signal)
        return signal
    
    def calculate_position_size(self, account_value: float, signal_dict: Dict, 
                            current_price: float, stop_price: float) -> float:
        """
        Calculate position size for rapid strategy with tight risk control.
        
        Args:
            account_value: Current account value
            signal_dict: Signal information dictionary
            current_price: Current market price
            stop_price: Stop loss price
            
        Returns:
            Position size (quantity to trade)
        """
        # Apply strict risk management for rapid strategy
        risk_pct = self.risk_per_trade
        
        # Adjust risk based on signal confidence
        confidence = signal_dict.get('confidence', 'medium')
        if confidence == 'high':
            risk_pct = risk_pct * 1.2
        elif confidence == 'low':
            risk_pct = risk_pct * 0.6
        
        # Calculate risk amount
        risk_amount = account_value * (risk_pct / 100)
        
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_price)
        
        # Safety check
        if risk_per_share <= 0 or risk_per_share > current_price * 0.05:
            # Default to 2% of price if invalid risk per share
            risk_per_share = current_price * 0.02
        
        # Calculate quantity
        quantity = risk_amount / risk_per_share
        
        logger.debug(f"Position size calculation: Account: {account_value}, Risk: {risk_pct}%, "
                   f"Amount: {risk_amount}, Quantity: {quantity}")
        
        return quantity
    
    def set_stop_loss(self, data: pd.DataFrame, signal_dict: Dict, 
                   entry_price: float, direction: str) -> float:
        """
        Calculate stop loss using a fraction of cycle amplitude.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price level
        """
        # Get primary cycle
        primary_cycle = signal_dict.get('primary_cycle')
        if not primary_cycle:
            # Default to ATR-based stop if cycle info missing
            atr = self.calculate_atr(data)
            return entry_price * (1 - 2 * atr / entry_price) if direction == 'long' else entry_price * (1 + 2 * atr / entry_price)
        
        # Calculate cycle amplitude (high-low range over the cycle period)
        cycle_high = data['high'].rolling(window=primary_cycle).max().iloc[-1]
        cycle_low = data['low'].rolling(window=primary_cycle).min().iloc[-1]
        cycle_amplitude = cycle_high - cycle_low
        
        # Set stop loss based on amplitude and direction
        stop_distance = cycle_amplitude * self.stop_loss_factor
        
        if direction == 'long':
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        logger.debug(f"Stop loss calculation: Entry: {entry_price}, Direction: {direction}, "
                   f"Cycle amplitude: {cycle_amplitude}, Stop: {stop_price}")
        
        return stop_price
    
    def set_take_profit(self, data: pd.DataFrame, signal_dict: Dict, 
                     entry_price: float, stop_price: float, direction: str) -> float:
        """
        Calculate take profit using risk-reward multiplier.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            stop_price: Stop loss price
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Take profit price level
        """
        # Calculate risk (distance from entry to stop)
        risk = abs(entry_price - stop_price)
        
        # Calculate take profit distance (2:1 reward-to-risk by default)
        take_profit_distance = risk * self.take_profit_factor
        
        # Set take profit based on direction
        if direction == 'long':
            take_profit = entry_price + take_profit_distance
        else:
            take_profit = entry_price - take_profit_distance
        
        logger.debug(f"Take profit calculation: Entry: {entry_price}, Stop: {stop_price}, "
                   f"Risk: {risk}, Take profit: {take_profit}")
        
        return take_profit