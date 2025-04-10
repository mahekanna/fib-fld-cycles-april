"""
Cycle Phase Trading Strategy Module

This strategy implements a sophisticated approach trading different phases of identified cycles.
It focuses on the accumulation phase (entering after trough confirmation in longest cycle) and
distribution phase (scaling out as projected peak approaches), creating a comprehensive
position management approach.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class CyclePhaseStrategy(BaseStrategy):
    """
    Cycle Phase Trading Strategy for advanced cycle-based position management.
    
    This sophisticated approach trades different phases of identified cycles:
    - Accumulation Phase: Enter after trough confirmation in longest cycle
    - Distribution Phase: Begin scaling out as projected peak approaches
    
    The strategy allows for:
    - Multiple entries on shorter cycle retracements
    - Maximum exposure when all cycles align bullish
    - Complete exit when longest cycle peaks
    - Reverse position when confirmed bearish alignment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Cycle Phase Trading Strategy.
        
        Args:
            config: Configuration dictionary with strategy parameters
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.phase_threshold = config.get('phase_threshold', 0.2)  # Early phase threshold
        self.min_cycle_count = config.get('min_cycle_count', 2)
        self.position_scale_factor = config.get('position_scale_factor', 0.33)  # For scaling in/out
        
        logger.info(f"Initialized {self.name} with phase threshold: {self.phase_threshold}, "
                   f"min cycles: {self.min_cycle_count}")
    
    def generate_signal(self, data: pd.DataFrame, cycles: List[int], 
                     fld_crossovers: List[Dict], cycle_states: List[Dict]) -> Dict:
        """
        Generate trading signals based on cycle phase analysis.
        
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
        
        # Identify longest cycle for primary analysis
        longest_cycle = max(cycles)
        longest_cycle_state = None
        for state in cycle_states:
            if state.get('cycle_length') == longest_cycle:
                longest_cycle_state = state
                break
        
        if not longest_cycle_state:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 'low',
                'description': "Could not identify state of longest cycle"
            }
        
        # Calculate overall alignment
        alignment = self.calculate_cycle_alignment(cycle_states)
        
        # Analyze cycle phases
        cycle_phases = self._analyze_cycle_phases(cycle_states)
        
        # Check for accumulation phase conditions
        is_accumulation = self._check_accumulation_conditions(cycle_phases, alignment)
        
        # Check for distribution phase conditions
        is_distribution = self._check_distribution_conditions(cycle_phases, alignment)
        
        # Determine signal based on phase analysis
        signal_type = 'neutral'
        strength = 0
        confidence = 'medium'
        description = "No clear cycle phase signal"
        
        if is_accumulation:
            signal_type = 'buy'
            strength = min(1.0, (alignment + 1) / 2)  # Scale from 0-1
            confidence = 'high' if alignment > 0.7 else 'medium'
            description = f"Accumulation phase detected in {longest_cycle}-period cycle"
        elif is_distribution:
            signal_type = 'sell'
            strength = min(1.0, (-alignment + 1) / 2)  # Scale from 0-1
            confidence = 'high' if alignment < -0.7 else 'medium'
            description = f"Distribution phase detected in {longest_cycle}-period cycle"
        
        # Enhance signal with scale-in/out recommendations
        scale_recommendation = None
        if signal_type == 'buy':
            # Count early phase cycles for scale-in recommendations
            early_phase_count = sum(1 for phase in cycle_phases if phase['phase'] == 'early')
            if early_phase_count >= 2:
                scale_recommendation = 'scale_in'
        elif signal_type == 'sell':
            # Count late phase cycles for scale-out recommendations
            late_phase_count = sum(1 for phase in cycle_phases if phase['phase'] == 'late')
            if late_phase_count >= 2:
                scale_recommendation = 'scale_out'
        
        signal = {
            'signal': signal_type,
            'strength': strength if signal_type == 'buy' else -strength if signal_type == 'sell' else 0,
            'confidence': confidence,
            'alignment': alignment,
            'longest_cycle': longest_cycle,
            'cycle_phases': cycle_phases,
            'is_accumulation': is_accumulation,
            'is_distribution': is_distribution,
            'scale_recommendation': scale_recommendation,
            'description': description
        }
        
        self.log_signal(signal)
        return signal
    
    def _analyze_cycle_phases(self, cycle_states: List[Dict]) -> List[Dict]:
        """
        Analyze the phase of each cycle.
        
        Args:
            cycle_states: List of cycle state dictionaries
            
        Returns:
            List of cycle phase information
        """
        cycle_phases = []
        
        for state in cycle_states:
            cycle_length = state.get('cycle_length', 0)
            days_since_cross = state.get('days_since_crossover')
            is_bullish = state.get('is_bullish', False)
            
            if cycle_length == 0 or days_since_cross is None:
                continue
            
            # Calculate completion percentage
            completion_pct = (days_since_cross / cycle_length) * 100
            
            # Determine phase
            if completion_pct <= 20:
                phase = 'early'
            elif completion_pct <= 60:
                phase = 'mid'
            else:
                phase = 'late'
            
            # Store cycle phase information
            cycle_phases.append({
                'cycle_length': cycle_length,
                'days_since_cross': days_since_cross,
                'completion_pct': completion_pct,
                'is_bullish': is_bullish,
                'phase': phase
            })
        
        return cycle_phases
    
    def _check_accumulation_conditions(self, cycle_phases: List[Dict], alignment: float) -> bool:
        """
        Check if conditions indicate accumulation phase.
        
        Args:
            cycle_phases: List of cycle phase information
            alignment: Overall cycle alignment score
            
        Returns:
            Boolean indicating if accumulation conditions are met
        """
        if not cycle_phases:
            return False
        
        # Sort phases by cycle length (longest first)
        sorted_phases = sorted(cycle_phases, key=lambda x: x['cycle_length'], reverse=True)
        longest_cycle_phase = sorted_phases[0]
        
        # Check if longest cycle is in early phase and is bullish
        longest_is_early_bullish = (
            longest_cycle_phase['phase'] == 'early' and 
            longest_cycle_phase['is_bullish'] and
            longest_cycle_phase['completion_pct'] <= 30
        )
        
        # Check alignment
        good_alignment = alignment > 0.5
        
        # Either longest cycle is in early bullish phase or we have good bullish alignment
        return longest_is_early_bullish or good_alignment
    
    def _check_distribution_conditions(self, cycle_phases: List[Dict], alignment: float) -> bool:
        """
        Check if conditions indicate distribution phase.
        
        Args:
            cycle_phases: List of cycle phase information
            alignment: Overall cycle alignment score
            
        Returns:
            Boolean indicating if distribution conditions are met
        """
        if not cycle_phases:
            return False
        
        # Sort phases by cycle length (longest first)
        sorted_phases = sorted(cycle_phases, key=lambda x: x['cycle_length'], reverse=True)
        longest_cycle_phase = sorted_phases[0]
        
        # Check if longest cycle is in late phase and is bullish (topping) or bearish
        longest_is_late = (
            longest_cycle_phase['phase'] == 'late' and 
            longest_cycle_phase['completion_pct'] >= 70
        )
        
        # Check alignment
        bearish_alignment = alignment < -0.3
        
        # Either longest cycle is in late phase or we have bearish alignment
        return longest_is_late or bearish_alignment
    
    def calculate_position_size(self, account_value: float, signal_dict: Dict, 
                            current_price: float, stop_price: float) -> float:
        """
        Calculate position size with phase-based scaling.
        
        Args:
            account_value: Current account value
            signal_dict: Signal information dictionary
            current_price: Current market price
            stop_price: Stop loss price
            
        Returns:
            Position size (quantity to trade)
        """
        # Get signal info
        alignment = signal_dict.get('alignment', 0)
        confidence = signal_dict.get('confidence', 'medium')
        scale_recommendation = signal_dict.get('scale_recommendation')
        
        # Base risk percentage on strategy settings
        base_risk = self.risk_per_trade
        
        # Adjust for confidence
        confidence_factor = 1.0
        if confidence == 'high':
            confidence_factor = 1.2
        elif confidence == 'low':
            confidence_factor = 0.7
        
        # Adjust for scale recommendation
        scaling_factor = 1.0
        if scale_recommendation == 'scale_in':
            scaling_factor = self.position_scale_factor
        elif scale_recommendation == 'scale_out':
            scaling_factor = 1 - self.position_scale_factor
        
        # Calculate adjusted risk percentage
        risk_pct = base_risk * confidence_factor * scaling_factor
        
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
        Calculate stop loss based on cycle phase.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price level
        """
        # Get cycle phases
        cycle_phases = signal_dict.get('cycle_phases', [])
        if not cycle_phases:
            # Default to ATR-based stop if cycle info missing
            atr = self.calculate_atr(data)
            return entry_price * (1 - 2 * atr / entry_price) if direction == 'long' else entry_price * (1 + 2 * atr / entry_price)
        
        # Sort by cycle length (shortest first for stop calculation)
        sorted_phases = sorted(cycle_phases, key=lambda x: x['cycle_length'])
        shortest_cycle = sorted_phases[0]['cycle_length']
        
        # Calculate stop based on cycle length and recent volatility
        lookback = min(len(data), shortest_cycle * 2)
        cycle_data = data.iloc[-lookback:]
        
        if direction == 'long':
            # For accumulation phase, use cycle low point
            cycle_low = cycle_data['low'].min()
            buffer = (entry_price - cycle_low) * 0.1  # 10% buffer
            stop_price = max(cycle_low - buffer, entry_price * 0.95)  # Max 5% from entry
        else:
            # For distribution phase, use cycle high point
            cycle_high = cycle_data['high'].max()
            buffer = (cycle_high - entry_price) * 0.1  # 10% buffer
            stop_price = min(cycle_high + buffer, entry_price * 1.05)  # Max 5% from entry
        
        logger.debug(f"Stop loss calculation: Entry: {entry_price}, Direction: {direction}, "
                   f"Cycle length: {shortest_cycle}, Stop: {stop_price}")
        
        return stop_price
    
    def set_take_profit(self, data: pd.DataFrame, signal_dict: Dict, 
                     entry_price: float, stop_price: float, direction: str) -> float:
        """
        Calculate take profit based on cycle projection.
        
        Args:
            data: Price data DataFrame
            signal_dict: Signal information dictionary
            entry_price: Entry price of the position
            stop_price: Stop loss price
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Take profit price level
        """
        # Get cycle phases and longest cycle
        cycle_phases = signal_dict.get('cycle_phases', [])
        longest_cycle = signal_dict.get('longest_cycle')
        
        if not longest_cycle or not cycle_phases:
            # Default to risk-reward multiple if cycle info missing
            risk = abs(entry_price - stop_price)
            reward = risk * 2  # 2:1 reward-to-risk ratio
            return entry_price + reward if direction == 'long' else entry_price - reward
        
        # Sort by cycle length (longest first for projection)
        sorted_phases = sorted(cycle_phases, key=lambda x: x['cycle_length'], reverse=True)
        longest_phase = sorted_phases[0]
        completion_pct = longest_phase.get('completion_pct', 50)
        
        # Calculate expected price move based on cycle phase
        lookback = min(len(data), longest_cycle * 2)
        cycle_data = data.iloc[-lookback:]
        cycle_range = cycle_data['high'].max() - cycle_data['low'].min()
        
        # Calculate projected move based on remaining cycle
        remaining_pct = 100 - completion_pct
        if remaining_pct <= 0:
            remaining_pct = 20  # Default if near cycle end
        
        projected_move = cycle_range * (remaining_pct / 100)
        
        # Set target based on direction
        if direction == 'long':
            take_profit = entry_price + projected_move
        else:
            take_profit = entry_price - projected_move
        
        logger.debug(f"Take profit calculation: Entry: {entry_price}, Direction: {direction}, "
                   f"Cycle completion: {completion_pct}%, Projected move: {projected_move}, "
                   f"Take profit: {take_profit}")
        
        return take_profit
    
    def update_stops(self, data: pd.DataFrame, positions: Dict) -> Dict:
        """
        Update stops based on cycle phase progression.
        
        Args:
            data: Current price data
            positions: Dictionary of open positions
            
        Returns:
            Updated positions dictionary
        """
        if not self.use_trailing_stop:
            return positions
            
        updated_positions = positions.copy()
        
        for pos_id, position in updated_positions.items():
            direction = position['direction']
            entry_date = position.get('entry_date')
            
            if entry_date is None:
                continue
                
            # Calculate days in trade
            if hasattr(entry_date, 'date'):
                days_in_trade = (data.index[-1].date() - entry_date.date()).days
            else:
                # If entry_date is already a date
                days_in_trade = (data.index[-1].date() - entry_date).days
            
            # Get cycle length for this position
            cycle_length = position.get('cycle_length', 21)  # Default
            
            # Adjust trailing stop based on position duration relative to cycle
            completion_pct = (days_in_trade / cycle_length) * 100
            current_price = data['close'].iloc[-1]
            
            if direction == 'long':
                # Early phase - wide stop
                if completion_pct < 30:
                    # Use recent cycle low
                    recent_low = data['low'].iloc[-min(len(data), cycle_length):].min()
                    new_stop = recent_low
                
                # Mid phase - tighter stop
                elif completion_pct < 70:
                    # Use recent swing low (shorter lookback)
                    lookback = min(len(data), max(5, int(cycle_length * 0.3)))
                    recent_low = data['low'].iloc[-lookback:].min()
                    new_stop = recent_low
                
                # Late phase - very tight stop to protect profits
                else:
                    # Use recent support level or a percentage of current price
                    lookback = min(len(data), max(3, int(cycle_length * 0.15)))
                    recent_low = data['low'].iloc[-lookback:].min()
                    pct_stop = current_price * 0.98  # 2% below current price
                    new_stop = max(recent_low, pct_stop)
                
                # Only move stop up, never down
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
                    logger.debug(f"Updated trailing stop to {new_stop:.2f} for position {pos_id}")
            
            else:  # Short position
                # Early phase - wide stop
                if completion_pct < 30:
                    # Use recent cycle high
                    recent_high = data['high'].iloc[-min(len(data), cycle_length):].max()
                    new_stop = recent_high
                
                # Mid phase - tighter stop
                elif completion_pct < 70:
                    # Use recent swing high (shorter lookback)
                    lookback = min(len(data), max(5, int(cycle_length * 0.3)))
                    recent_high = data['high'].iloc[-lookback:].max()
                    new_stop = recent_high
                
                # Late phase - very tight stop to protect profits
                else:
                    # Use recent resistance level or a percentage of current price
                    lookback = min(len(data), max(3, int(cycle_length * 0.15)))
                    recent_high = data['high'].iloc[-lookback:].max()
                    pct_stop = current_price * 1.02  # 2% above current price
                    new_stop = min(recent_high, pct_stop)
                
                # Only move stop down, never up
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
                    logger.debug(f"Updated trailing stop to {new_stop:.2f} for position {pos_id}")
                    
        return updated_positions