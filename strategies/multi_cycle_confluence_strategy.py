"""
Multi-Cycle Confluence Strategy Module

Implements the Multi-Cycle Confluence Strategy which identifies when multiple cycle FLDs
align in the same direction and enters on retracements to the primary FLD. This strategy
is ideal for range-bound markets with clear cyclical behavior and moderate volatility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MultiCycleConfluenceStrategy(BaseStrategy):
    """
    Multi-Cycle Confluence Strategy for range-bound markets.
    
    This strategy identifies when multiple cycle FLDs align in the same direction
    and enters on retracements to the primary FLD, placing stops beyond recent
    cycle extremes and targeting the next projected cycle turn.
    
    Key features:
    - Requires alignment of multiple cycles' FLDs
    - Enters on retracements to primary FLD
    - Places stops beyond recent cycle extreme
    - Targets next projected cycle turn
    - Optimal for range-bound markets with clear cycles
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Multi-Cycle Confluence Strategy.
        
        Args:
            config: Configuration dictionary with strategy parameters
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.min_cycle_count = config.get('min_cycle_count', 3)
        self.min_alignment_threshold = config.get('min_alignment_threshold', 0.8)
        self.retracement_threshold = config.get('retracement_threshold', 0.5)  # 50% retracement
        
        logger.info(f"Initialized {self.name} with min cycles: {self.min_cycle_count}, "
                   f"alignment threshold: {self.min_alignment_threshold}")
    
    def generate_signal(self, data: pd.DataFrame, cycles: List[int], 
                     fld_crossovers: List[Dict], cycle_states: List[Dict]) -> Dict:
        """
        Generate trading signals based on multi-cycle confluence.
        
        Args:
            data: Price data DataFrame with OHLCV columns
            cycles: List of detected cycle lengths
            fld_crossovers: List of detected FLD crossovers
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Dictionary with signal information
        """
        # Validate inputs
        if not cycles or len(cycles) < self.min_cycle_count or not cycle_states:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'description': f"Insufficient cycles detected (need {self.min_cycle_count})"
            }
        
        # Calculate alignment score of all cycles
        alignment = self.calculate_cycle_alignment(cycle_states)
        
        # Check if alignment meets threshold
        if abs(alignment) < self.min_alignment_threshold:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'alignment': alignment,
                'description': f"Insufficient cycle alignment: {alignment:.2f}"
            }
        
        # Determine the primary cycle (middle of detected cycles)
        sorted_cycles = sorted(cycles)
        primary_idx = len(sorted_cycles) // 2
        primary_cycle = sorted_cycles[primary_idx]
        
        # Get FLD crossovers for primary cycle if not provided
        primary_crossovers = []
        if not fld_crossovers:
            for cycle in cycles:
                cycle_crossovers = self.detect_fld_crossovers(data, cycle)
                if cycle == primary_cycle:
                    primary_crossovers = cycle_crossovers
        else:
            # Filter to get only primary cycle crossovers
            primary_crossovers = [c for c in fld_crossovers if c.get('cycle_length') == primary_cycle]
        
        # Calculate retracement to primary FLD
        current_price = data['close'].iloc[-1]
        
        # Find the most recent crossover
        recent_crossovers = sorted(primary_crossovers, key=lambda x: x['date'])[-3:]
        
        if not recent_crossovers:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'alignment': alignment,
                'description': "No recent FLD crossovers detected"
            }
        
        latest_crossover = recent_crossovers[-1]
        crossover_price = latest_crossover['price']
        crossover_direction = latest_crossover['direction']
        
        # Check if dominant direction aligns with crossover direction
        bullish_alignment = alignment > 0 and crossover_direction == 'bullish'
        bearish_alignment = alignment < 0 and crossover_direction == 'bearish'
        
        if not (bullish_alignment or bearish_alignment):
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'medium',
                'alignment': alignment,
                'description': f"Crossover direction {crossover_direction} doesn't match cycle alignment"
            }
        
        # Check for retracement to FLD (potential entry)
        fld_value = self._calculate_current_fld(data, primary_cycle)
        
        # Determine if we have a valid retracement
        signal_type = 'neutral'
        confidence = 'medium'
        retracement_detected = False
        
        if bullish_alignment:
            # For bullish alignment, check if price has retraced towards FLD
            price_change = (current_price - crossover_price) / crossover_price
            retracement = abs(price_change) < self.retracement_threshold
            pullback = current_price < crossover_price  # Pulled back from crossover
            
            if pullback and current_price >= fld_value:
                signal_type = 'buy'
                retracement_detected = True
                confidence = 'high' if alignment > 0.9 else 'medium'
        
        elif bearish_alignment:
            # For bearish alignment, check if price has retraced towards FLD
            price_change = (current_price - crossover_price) / crossover_price
            retracement = abs(price_change) < self.retracement_threshold
            pullback = current_price > crossover_price  # Pulled back from crossover
            
            if pullback and current_price <= fld_value:
                signal_type = 'sell'
                retracement_detected = True
                confidence = 'high' if alignment < -0.9 else 'medium'
        
        # Calculate signal strength based on alignment and retracement
        strength = abs(alignment)
        if not retracement_detected:
            strength = 0
            signal_type = 'neutral'
            confidence = 'low'
        
        signal = {
            'signal': signal_type,
            'strength': alignment if signal_type == 'buy' else -alignment if signal_type == 'sell' else 0,
            'confidence': confidence,
            'alignment': alignment,
            'primary_cycle': primary_cycle,
            'cycles': cycles,
            'fld_value': fld_value,
            'recent_crossover': latest_crossover,
            'description': f"{signal_type.capitalize()} signal with {len(cycles)}-cycle confluence, "
                          f"alignment: {alignment:.2f}"
        }
        
        self.log_signal(signal)
        return signal
    
    def _calculate_current_fld(self, data: pd.DataFrame, cycle_length: int) -> float:
        """
        Calculate the current FLD value for a given cycle.
        
        Args:
            data: Price data DataFrame
            cycle_length: Cycle length to calculate FLD for
            
        Returns:
            Current FLD value
        """
        if len(data) < cycle_length // 2:
            return data['close'].iloc[-1]
            
        # FLD is price displaced by half a cycle
        half_cycle = cycle_length // 2
        
        if len(data) < half_cycle:
            return data['close'].iloc[-1]
            
        return data['close'].iloc[-half_cycle-1]
    
    def calculate_position_size(self, account_value: float, signal_dict: Dict, 
                            current_price: float, stop_price: float) -> float:
        """
        Calculate position size based on confluence strength.
        
        Args:
            account_value: Current account value
            signal_dict: Signal information dictionary
            current_price: Current market price
            stop_price: Stop loss price
            
        Returns:
            Position size (quantity to trade)
        """
        # Get alignment and confidence
        alignment = signal_dict.get('alignment', 0)
        confidence = signal_dict.get('confidence', 'medium')
        
        # Base risk percentage on alignment strength
        base_risk = self.risk_per_trade
        alignment_factor = min(1.5, max(0.5, abs(alignment) * 1.25))
        
        # Adjust for confidence
        confidence_factor = 1.0
        if confidence == 'high':
            confidence_factor = 1.2
        elif confidence == 'low':
            confidence_factor = 0.7
        
        # Calculate adjusted risk percentage
        risk_pct = base_risk * alignment_factor * confidence_factor
        
        # Calculate risk amount
        risk_amount = account_value * (risk_pct / 100)
        
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_price)
        
        # Safety check
        if risk_per_share <= 0 or risk_per_share > current_price * 0.1:
            # Default to 3% of price if invalid risk per share
            risk_per_share = current_price * 0.03
        
        # Calculate quantity
        quantity = risk_amount / risk_per_share
        
        logger.debug(f"Position size calculation: Account: {account_value}, Base risk: {base_risk}%, "
                   f"Adjusted risk: {risk_pct}%, Amount: {risk_amount}, Quantity: {quantity}")
        
        return quantity
    
    def set_stop_loss(self, data: pd.DataFrame, signal_dict: Dict, 
                   entry_price: float, direction: str) -> float:
        """
        Calculate stop loss beyond recent cycle extreme.
        
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
        
        # For this strategy, we set stops beyond recent cycle extreme
        lookback = min(len(data), primary_cycle)
        
        if direction == 'long':
            # For long positions, stop goes below recent low
            recent_low = data['low'].iloc[-lookback:].min()
            buffer = (entry_price - recent_low) * 0.1  # Add 10% buffer
            stop_price = recent_low - buffer
        else:
            # For short positions, stop goes above recent high
            recent_high = data['high'].iloc[-lookback:].max()
            buffer = (recent_high - entry_price) * 0.1  # Add 10% buffer
            stop_price = recent_high + buffer
        
        logger.debug(f"Stop loss calculation: Entry: {entry_price}, Direction: {direction}, "
                   f"Stop: {stop_price}")
        
        return stop_price
    
    def set_take_profit(self, data: pd.DataFrame, signal_dict: Dict, 
                     entry_price: float, stop_price: float, direction: str) -> float:
        """
        Calculate take profit targeting next projected cycle turn.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            stop_price: Stop loss price
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Take profit price level
        """
        # Get cycles
        primary_cycle = signal_dict.get('primary_cycle')
        
        if not primary_cycle:
            # Default to risk-reward multiple if cycle info missing
            risk = abs(entry_price - stop_price)
            reward = risk * self.take_profit_multiplier
            return entry_price + reward if direction == 'long' else entry_price - reward
        
        # Calculate expected move to next cycle turn
        # For this strategy, we target the next cycle turn
        
        # Calculate average cycle range
        cycle_data = data.iloc[-primary_cycle*2:]
        if len(cycle_data) > primary_cycle:
            cycle_range = cycle_data['high'].max() - cycle_data['low'].min()
            
            # Set target based on direction and cycle range
            if direction == 'long':
                take_profit = entry_price + cycle_range * 0.7
            else:
                take_profit = entry_price - cycle_range * 0.7
                
            logger.debug(f"Take profit calculation: Entry: {entry_price}, Cycle range: {cycle_range}, "
                       f"Take profit: {take_profit}")
            
            return take_profit
        
        # Fallback to risk-reward multiple
        risk = abs(entry_price - stop_price)
        reward = risk * self.take_profit_multiplier
        return entry_price + reward if direction == 'long' else entry_price - reward