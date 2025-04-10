"""
Turning Point Anticipation Strategy Module

This strategy leverages projected cycle turns to anticipate market reversals,
monitoring approaching projected cycle turns from multiple timeframes and
confirming reversals with price action patterns. It's designed for swing trading
with longer holding periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TurningPointAnticipationStrategy(BaseStrategy):
    """
    Turning Point Anticipation Strategy for swing trading.
    
    This strategy leverages projected cycle turns to anticipate market reversals.
    It monitors approaching projected cycle turns from multiple timeframes,
    confirms reversals with price action, and holds positions through cycle
    duration with trailing stops.
    
    Key features:
    - Monitors approaching projected cycle turns
    - Confirms reversals with price action patterns
    - Enters when smaller cycle confirms direction change
    - Holds through cycle duration with trailing stops
    - Exits at next major cycle turning point projection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Turning Point Anticipation Strategy.
        
        Args:
            config: Configuration dictionary with strategy parameters
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.turn_proximity_threshold = config.get('turn_proximity_threshold', 0.15)  # Percent of cycle
        self.confirmation_bars = config.get('confirmation_bars', 2)
        self.stop_factor = config.get('stop_factor', 0.5)  # Multiple of cycle amplitude
        
        logger.info(f"Initialized {self.name} with turn proximity: {self.turn_proximity_threshold}, "
                   f"confirmation bars: {self.confirmation_bars}")
    
    def generate_signal(self, data: pd.DataFrame, cycles: List[int], 
                     fld_crossovers: List[Dict], cycle_states: List[Dict]) -> Dict:
        """
        Generate trading signals based on anticipated turning points.
        
        Args:
            data: Price data DataFrame with OHLCV columns
            cycles: List of detected cycle lengths
            fld_crossovers: List of detected FLD crossovers
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Dictionary with signal information
        """
        # Validate inputs
        if not cycles or not cycle_states:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'description': "Insufficient cycle data"
            }
        
        # Identify cycles approaching turning points
        cycles_approaching_turns = self._identify_approaching_turns(data, cycle_states)
        
        if not cycles_approaching_turns:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'description': "No cycles approaching turning points"
            }
        
        # Check for price confirmation of reversal
        confirmed_reversals = self._check_price_confirmation(data, cycles_approaching_turns)
        
        if not confirmed_reversals:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'medium',
                'approaching_turns': cycles_approaching_turns,
                'description': "Approaching turns without price confirmation"
            }
        
        # Determine signal direction based on confirmed reversals
        bullish_confirmations = [c for c in confirmed_reversals if c['direction'] == 'bullish']
        bearish_confirmations = [c for c in confirmed_reversals if c['direction'] == 'bearish']
        
        signal_type = 'neutral'
        strength = 0
        confidence = 'medium'
        
        if len(bullish_confirmations) > len(bearish_confirmations):
            signal_type = 'buy'
            strength = min(1.0, len(bullish_confirmations) / len(cycles))
            confidence = 'high' if len(bullish_confirmations) >= 2 else 'medium'
        elif len(bearish_confirmations) > len(bullish_confirmations):
            signal_type = 'sell'
            strength = -min(1.0, len(bearish_confirmations) / len(cycles))
            confidence = 'high' if len(bearish_confirmations) >= 2 else 'medium'
        
        signal = {
            'signal': signal_type,
            'strength': strength,
            'confidence': confidence,
            'approaching_turns': cycles_approaching_turns,
            'confirmed_reversals': confirmed_reversals,
            'cycles': cycles,
            'dominant_cycle': max(cycles),  # Use longest cycle for swing strategy
            'description': f"{signal_type.capitalize()} signal from {len(confirmed_reversals)} "
                          f"confirmed cycle turns out of {len(cycles)} cycles"
        }
        
        self.log_signal(signal)
        return signal
    
    def _identify_approaching_turns(self, data: pd.DataFrame, cycle_states: List[Dict]) -> List[Dict]:
        """
        Identify cycles that are approaching turning points.
        
        Args:
            data: Price data DataFrame
            cycle_states: List of cycle state dictionaries
            
        Returns:
            List of cycles approaching turns with direction
        """
        approaching_turns = []
        
        for state in cycle_states:
            cycle_length = state.get('cycle_length', 0)
            days_since_cross = state.get('days_since_crossover', 0)
            is_bullish = state.get('is_bullish', False)
            
            if cycle_length == 0 or days_since_cross is None:
                continue
            
            # Calculate how close we are to next turn (half cycle)
            half_cycle = cycle_length / 2
            proximity_to_turn = (days_since_cross % half_cycle) / half_cycle
            
            # If we're in the last 15% of the half-cycle, we're approaching a turn
            if proximity_to_turn > (1 - self.turn_proximity_threshold):
                # Current phase determines what kind of turn we're approaching
                expected_direction = 'bearish' if is_bullish else 'bullish'
                
                approaching_turns.append({
                    'cycle_length': cycle_length,
                    'days_since_cross': days_since_cross,
                    'current_phase': 'bullish' if is_bullish else 'bearish',
                    'expected_direction': expected_direction,
                    'proximity_to_turn': proximity_to_turn
                })
        
        return approaching_turns
    
    def _check_price_confirmation(self, data: pd.DataFrame, approaching_turns: List[Dict]) -> List[Dict]:
        """
        Check if price action confirms the expected reversal.
        
        Args:
            data: Price data DataFrame
            approaching_turns: List of cycles approaching turns
            
        Returns:
            List of confirmed reversals with direction
        """
        if len(data) < self.confirmation_bars + 1:
            return []
        
        confirmed_reversals = []
        
        for turn in approaching_turns:
            expected_direction = turn['expected_direction']
            
            # Extract short-term price action
            recent_data = data.iloc[-self.confirmation_bars-1:]
            
            if expected_direction == 'bullish':
                # Check for bullish reversal confirmation (higher lows, higher highs)
                lower_low = recent_data['low'].iloc[0] > recent_data['low'].iloc[1:].min()
                higher_high = recent_data['high'].iloc[-1] > recent_data['high'].iloc[:-1].max()
                higher_close = recent_data['close'].iloc[-1] > recent_data['close'].iloc[-2]
                
                if higher_close and (higher_high or not lower_low):
                    confirmed_reversals.append({
                        'cycle_length': turn['cycle_length'],
                        'direction': 'bullish',
                        'proximity_to_turn': turn['proximity_to_turn']
                    })
            
            elif expected_direction == 'bearish':
                # Check for bearish reversal confirmation (lower highs, lower lows)
                higher_high = recent_data['high'].iloc[0] < recent_data['high'].iloc[1:].max()
                lower_low = recent_data['low'].iloc[-1] < recent_data['low'].iloc[:-1].min()
                lower_close = recent_data['close'].iloc[-1] < recent_data['close'].iloc[-2]
                
                if lower_close and (lower_low or not higher_high):
                    confirmed_reversals.append({
                        'cycle_length': turn['cycle_length'],
                        'direction': 'bearish',
                        'proximity_to_turn': turn['proximity_to_turn']
                    })
        
        return confirmed_reversals
    
    def calculate_position_size(self, account_value: float, signal_dict: Dict, 
                            current_price: float, stop_price: float) -> float:
        """
        Calculate position size for swing trades with higher confidence level.
        
        Args:
            account_value: Current account value
            signal_dict: Signal information dictionary
            current_price: Current market price
            stop_price: Stop loss price
            
        Returns:
            Position size (quantity to trade)
        """
        # Use more conservative position sizing for swing trades
        risk_pct = self.risk_per_trade * 0.8  # More conservative
        
        # Adjust risk based on signal confidence
        confidence = signal_dict.get('confidence', 'medium')
        if confidence == 'high':
            risk_pct = risk_pct * 1.2
        elif confidence == 'low':
            risk_pct = risk_pct * 0.5
        
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
        
        logger.debug(f"Position size calculation: Account: {account_value}, Risk: {risk_pct}%, "
                   f"Amount: {risk_amount}, Risk/share: {risk_per_share}, Quantity: {quantity}")
        
        return quantity
    
    def set_stop_loss(self, data: pd.DataFrame, signal_dict: Dict, 
                   entry_price: float, direction: str) -> float:
        """
        Calculate stop loss at 0.5 x cycle amplitude.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price level
        """
        # Get dominant cycle length for swing strategy
        dominant_cycle = signal_dict.get('dominant_cycle')
        if not dominant_cycle:
            # Use longest detected cycle
            cycles = signal_dict.get('cycles', [])
            dominant_cycle = max(cycles) if cycles else 21  # Default
        
        # Calculate cycle amplitude
        lookback = min(len(data), dominant_cycle * 2)
        cycle_data = data.iloc[-lookback:]
        cycle_amplitude = cycle_data['high'].max() - cycle_data['low'].min()
        
        # Set stop based on amplitude and direction
        stop_distance = cycle_amplitude * self.stop_factor
        
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
        Calculate take profit targeting next major cycle turning point.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            stop_price: Stop loss price
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Take profit price level
        """
        # For swing strategy, we target the next major cycle turning point
        dominant_cycle = signal_dict.get('dominant_cycle')
        if not dominant_cycle:
            # Use longest detected cycle
            cycles = signal_dict.get('cycles', [])
            dominant_cycle = max(cycles) if cycles else 21  # Default
        
        # Calculate expected move to next cycle turn
        lookback = min(len(data), dominant_cycle * 2)
        cycle_data = data.iloc[-lookback:]
        cycle_amplitude = cycle_data['high'].max() - cycle_data['low'].min()
        
        # Calculate momentum factor (how strongly price is moving)
        recent_bars = min(5, len(data)-1)
        price_change = (data['close'].iloc[-1] - data['close'].iloc[-recent_bars-1]) / data['close'].iloc[-recent_bars-1]
        momentum_factor = min(1.5, max(0.5, abs(price_change) * 20 + 1))  # Scale to 0.5-1.5
        
        # Set target based on direction, cycle amplitude and momentum
        if direction == 'long':
            take_profit = entry_price + cycle_amplitude * momentum_factor
        else:
            take_profit = entry_price - cycle_amplitude * momentum_factor
        
        logger.debug(f"Take profit calculation: Entry: {entry_price}, Direction: {direction}, "
                   f"Cycle amplitude: {cycle_amplitude}, Momentum: {momentum_factor}, Target: {take_profit}")
        
        return take_profit
    
    def update_stops(self, data: pd.DataFrame, positions: Dict) -> Dict:
        """
        Update stop loss levels using cycle extremes.
        
        Args:
            data: Current price data
            positions: Dictionary of open positions
            
        Returns:
            Updated positions dictionary
        """
        if not self.use_trailing_stop:
            return positions
            
        updated_positions = positions.copy()
        current_price = data['close'].iloc[-1]
        
        for pos_id, position in updated_positions.items():
            direction = position['direction']
            entry_price = position['entry_price']
            cycle_length = position.get('cycle_length', 21)  # Default
            
            # Determine trailing stop threshold based on cycle
            lookback = min(10, len(data)-1)
            recent_data = data.iloc[-lookback:]
            
            if direction == 'long':
                # For long positions, use recent swing low
                recent_low = recent_data['low'].min()
                current_stop = position['stop_loss']
                
                # Only move stop up, never down
                if recent_low > current_stop:
                    position['stop_loss'] = recent_low
                    logger.debug(f"Updated trailing stop to {recent_low:.2f} for position {pos_id}")
            else:
                # For short positions, use recent swing high
                recent_high = recent_data['high'].max()
                current_stop = position['stop_loss']
                
                # Only move stop down, never up
                if recent_high < current_stop:
                    position['stop_loss'] = recent_high
                    logger.debug(f"Updated trailing stop to {recent_high:.2f} for position {pos_id}")
                    
        return updated_positions