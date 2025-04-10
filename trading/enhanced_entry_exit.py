"""
Enhanced Entry/Exit Strategy Module

This module implements advanced cycle-based entry and exit strategies that take into account:
1. Cycle maturity (percentage completion)
2. Optimal entry windows based on cycle phase
3. Multi-cycle alignment for entry/exit decisions
4. Position sizing recommendations based on cycle maturity
5. Trade duration planning based on cycle lengths

This module extends and enhances the position guidance functionality that would 
conceptually be in a 'position_guidance.py' file. It provides a more sophisticated
approach to trade timing and position management by analyzing cycle maturity and phase.

Key features:
- Calculates precise cycle maturity percentages based on days since crossover
- Classifies cycles into phases (Fresh Crossover, Early Cycle, Mid Cycle, etc.)
- Identifies optimal entry windows based on cycle phases
- Calculates and visualizes cycle alignment scores
- Provides dynamic position sizing based on cycle maturity
- Recommends trade duration based on cycle lengths
- Offers enhanced stop loss and target calculations
- Evaluates entry conditions with detailed confidence metrics

The EnhancedEntryExitStrategy class works in conjunction with the core analysis components
to provide actionable trading guidance optimized for the current cycle conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

# Import from other modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.scan_result import ScanResult
from core.fld_signal_generator import FLDCalculator

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedEntryExitStrategy:
    """
    Implements advanced entry and exit strategies based on cycle maturity,
    harmonic relationships, and multi-cycle alignment.
    
    This class extends the position guidance functionality with sophisticated
    cycle-based analysis that enables more precise trade timing and management.
    It complements the basic position guidance provided in the core/fld_signal_generator.py
    by adding cycle maturity analysis, adaptive position sizing, and optimal
    entry/exit window identification.
    
    Key methods include:
    - analyze(): Performs comprehensive enhanced entry/exit analysis
    - _calculate_cycle_maturity(): Analyzes cycle completion percentages
    - _calculate_entry_windows(): Determines optimal entry window quality
    - _calculate_position_sizing(): Computes recommended position size
    - _calculate_trade_duration(): Recommends optimal holding periods
    - _calculate_alignment_score(): Evaluates multi-cycle alignment
    - _evaluate_entry_conditions(): Determines if entry conditions are favorable
    - _generate_enhanced_guidance(): Creates comprehensive trading guidance
    
    This module represents an advanced layer of trading strategy that builds on
    the foundational cycle detection and signal generation capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enhanced entry/exit strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fld_calculator = FLDCalculator(
            gap_threshold=config['analysis'].get('gap_threshold', 0.01)
        )
    
    def analyze(self, result: ScanResult) -> Dict[str, Any]:
        """
        Perform enhanced entry/exit analysis on a scan result.
        
        Args:
            result: ScanResult object containing cycle and market data
            
        Returns:
            Dictionary with enhanced entry/exit recommendations
        """
        if not result.success:
            return {
                'valid': False,
                'message': f"Cannot analyze unsuccessful scan: {result.error}"
            }
        
        # Calculate cycle maturity for each detected cycle
        cycle_maturity = self._calculate_cycle_maturity(result.cycle_states)
        
        # Determine optimal entry windows
        entry_windows = self._calculate_entry_windows(cycle_maturity)
        
        # Calculate position size recommendations
        position_sizing = self._calculate_position_sizing(cycle_maturity)
        
        # Determine trade duration recommendations
        trade_duration = self._calculate_trade_duration(result.detected_cycles)
        
        # Calculate multi-cycle alignment score
        alignment_score = self._calculate_alignment_score(cycle_maturity, result.cycle_states)
        
        # Determine if conditions are favorable for entry
        entry_conditions = self._evaluate_entry_conditions(cycle_maturity, alignment_score)
        
        # Generate enhanced trade guidance
        enhanced_guidance = self._generate_enhanced_guidance(
            result, 
            cycle_maturity, 
            entry_windows, 
            position_sizing,
            trade_duration,
            alignment_score,
            entry_conditions
        )
        
        return enhanced_guidance
    
    def _calculate_cycle_maturity(self, cycle_states: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Calculate the maturity (percentage completion) of each cycle.
        
        Args:
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Dictionary mapping cycle length to maturity information
        """
        cycle_maturity = {}
        
        for state in cycle_states:
            cycle_length = state['cycle_length']
            days_since_cross = state['days_since_crossover']
            
            # Skip if crossover data is not available
            if days_since_cross is None:
                continue
            
            # Calculate percentage of cycle completed since last crossover
            completion_pct = (days_since_cross / cycle_length) * 100.0
            
            # Determine maturity phase - Only count as Fresh Crossover if it's within the last 2 days
            if completion_pct <= 15 and days_since_cross <= 2:  # Only truly fresh if within 2 days
                phase = "Fresh Crossover"
                optimal_entry = True
                entry_quality = "Excellent"
            elif completion_pct <= 25:
                phase = "Early Cycle"
                optimal_entry = True
                entry_quality = "Very Good"
            elif completion_pct <= 50:
                phase = "Mid Cycle"
                optimal_entry = state['is_bullish']  # Only optimal if in bullish phase
                entry_quality = "Good"
            elif completion_pct <= 80:
                phase = "Late Cycle"
                optimal_entry = False
                entry_quality = "Caution"
            else:
                phase = "End Cycle"
                optimal_entry = False
                entry_quality = "Avoid Entry"
            
            # Calculate expected days remaining in current phase
            days_remaining = cycle_length - days_since_cross
            
            # Store cycle maturity information
            cycle_maturity[cycle_length] = {
                'cycle_length': cycle_length,
                'days_since_cross': days_since_cross,
                'completion_pct': completion_pct,
                'phase': phase,
                'days_remaining': days_remaining,
                'optimal_entry': optimal_entry,
                'entry_quality': entry_quality,
                'is_bullish': state['is_bullish']
            }
        
        return cycle_maturity
    
    def _calculate_entry_windows(self, cycle_maturity: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate optimal entry windows based on cycle maturity.
        
        Args:
            cycle_maturity: Dictionary mapping cycle length to maturity information
            
        Returns:
            Dictionary with entry window recommendations
        """
        # Count cycles in different phases
        fresh_crossovers = sum(1 for cycle in cycle_maturity.values() 
                              if cycle['phase'] == "Fresh Crossover")
        early_cycles = sum(1 for cycle in cycle_maturity.values() 
                          if cycle['phase'] == "Early Cycle")
        mid_cycles = sum(1 for cycle in cycle_maturity.values() 
                        if cycle['phase'] == "Mid Cycle")
        late_cycles = sum(1 for cycle in cycle_maturity.values() 
                         if cycle['phase'] == "Late Cycle")
        end_cycles = sum(1 for cycle in cycle_maturity.values() 
                        if cycle['phase'] == "End Cycle")
        
        # Determine entry quality based on cycle phases
        total_cycles = len(cycle_maturity)
        
        if total_cycles == 0:
            return {
                'entry_quality': "Unknown",
                'description': "No cycle data available",
                'score': 0.0
            }
        
        fresh_pct = (fresh_crossovers / total_cycles) * 100
        early_pct = (early_cycles / total_cycles) * 100
        end_pct = (end_cycles / total_cycles) * 100
        
        # Calculate weighted entry quality score (0-10 scale)
        entry_score = (
            (fresh_crossovers * 10.0) + 
            (early_cycles * 7.5) + 
            (mid_cycles * 5.0) + 
            (late_cycles * 2.5) - 
            (end_cycles * 5.0)
        ) / total_cycles
        
        # Bound the score between 0 and 10
        entry_score = max(0, min(10, entry_score))
        
        # Determine overall entry quality
        if entry_score >= 8.5:
            quality = "Excellent"
        elif entry_score >= 7.0:
            quality = "Very Good"
        elif entry_score >= 5.0:
            quality = "Good"
        elif entry_score >= 3.0:
            quality = "Fair"
        elif entry_score >= 1.0:
            quality = "Poor"
        else:
            quality = "Avoid Entry"
        
        # Generate description
        if end_cycles > 0:
            description = (
                f"{end_cycles} cycle(s) near completion. "
                f"Consider waiting for new crossover."
            )
        elif fresh_crossovers + early_cycles == total_cycles:
            description = (
                f"All cycles in optimal entry phase. "
                f"Excellent entry opportunity."
            )
        elif fresh_crossovers > 0:
            description = (
                f"{fresh_crossovers} fresh crossover(s). "
                f"Good opportunity for entry."
            )
        else:
            description = (
                f"Mixed cycle phases. "
                f"Entry opportunity: {quality}."
            )
        
        return {
            'entry_quality': quality,
            'description': description,
            'score': entry_score,
            'fresh_crossovers': fresh_crossovers,
            'early_cycles': early_cycles,
            'mid_cycles': mid_cycles,
            'late_cycles': late_cycles,
            'end_cycles': end_cycles
        }
    
    def _calculate_position_sizing(self, cycle_maturity: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate position sizing recommendations based on cycle maturity.
        
        Args:
            cycle_maturity: Dictionary mapping cycle length to maturity information
            
        Returns:
            Dictionary with position sizing recommendations
        """
        # Default full position size
        base_position_pct = 100.0
        
        # Position size adjustments based on cycle maturity
        for cycle in cycle_maturity.values():
            phase = cycle['phase']
            
            # Adjust position size based on phase
            if phase == "End Cycle":
                # Major reduction for cycles nearing completion
                base_position_pct *= 0.25
            elif phase == "Late Cycle":
                # Significant reduction for late cycle
                base_position_pct *= 0.5
            elif phase == "Mid Cycle":
                # Moderate reduction for mid cycle
                base_position_pct *= 0.75
            # No reduction for Fresh Crossover or Early Cycle
        
        # Ensure position size is between 20% and 100%
        position_pct = max(20.0, min(100.0, base_position_pct))
        
        # Determine risk adjustment factor
        if position_pct <= 25:
            risk_text = "Minimum Position"
            stop_adjustment = 0.5  # Tighter stops for smaller positions
        elif position_pct <= 50:
            risk_text = "Reduced Position"
            stop_adjustment = 0.75
        elif position_pct <= 75:
            risk_text = "Standard Position"
            stop_adjustment = 1.0
        else:
            risk_text = "Full Position"
            stop_adjustment = 1.0
        
        return {
            'position_pct': position_pct,
            'risk_text': risk_text,
            'stop_adjustment': stop_adjustment
        }
    
    def _calculate_trade_duration(self, cycle_lengths: List[int]) -> Dict[str, Any]:
        """
        Calculate trade duration recommendations based on cycle lengths.
        
        Args:
            cycle_lengths: List of detected cycle lengths
            
        Returns:
            Dictionary with trade duration recommendations
        """
        if not cycle_lengths:
            return {
                'min_hold': 0,
                'optimal_hold': 0,
                'max_hold': 0,
                'description': "No cycle data available"
            }
        
        # Sort cycles by length
        sorted_cycles = sorted(cycle_lengths)
        
        # Use the shortest cycle for minimum hold period
        min_hold = max(1, int(sorted_cycles[0] * 0.25))
        
        # Use middle cycle (or shorter of two middle cycles) for optimal hold
        if len(sorted_cycles) % 2 == 0:
            middle_idx = len(sorted_cycles) // 2 - 1
        else:
            middle_idx = len(sorted_cycles) // 2
        
        optimal_hold = max(min_hold, int(sorted_cycles[middle_idx] * 0.5))
        
        # Use the longest cycle for maximum hold period
        # But never hold through more than 80% of longest cycle
        max_hold = max(optimal_hold, int(sorted_cycles[-1] * 0.8))
        
        return {
            'min_hold': min_hold,
            'optimal_hold': optimal_hold,
            'max_hold': max_hold,
            'description': (
                f"Min: {min_hold} days, "
                f"Optimal: {optimal_hold} days, "
                f"Max: {max_hold} days"
            )
        }
    
    def _calculate_alignment_score(self, 
                                 cycle_maturity: Dict[int, Dict[str, Any]],
                                 cycle_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate multi-cycle alignment score.
        
        Args:
            cycle_maturity: Dictionary mapping cycle length to maturity information
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Dictionary with alignment score information
        """
        if not cycle_states:
            return {
                'score': 0.0,
                'quality': "Unknown",
                'description': "No cycle data available"
            }
        
        # Count bullish and bearish cycles
        bullish_count = sum(1 for state in cycle_states if state['is_bullish'])
        bearish_count = len(cycle_states) - bullish_count
        
        # Determine dominant direction
        total_cycles = len(cycle_states)
        bullish_pct = (bullish_count / total_cycles) * 100 if total_cycles > 0 else 0
        
        if bullish_count > bearish_count:
            direction = "bullish"
            aligned_count = bullish_count
        else:
            direction = "bearish"
            aligned_count = bearish_count
        
        # Calculate alignment score (0-10 scale)
        # 10 = all cycles aligned, 0 = evenly split
        alignment_score = abs((bullish_count - bearish_count) / total_cycles) * 10.0
        
        # Determine alignment quality
        if alignment_score >= 9.5:
            quality = "Perfect Alignment"
        elif alignment_score >= 8.0:
            quality = "Strong Alignment"
        elif alignment_score >= 6.0:
            quality = "Good Alignment"
        elif alignment_score >= 4.0:
            quality = "Moderate Alignment"
        elif alignment_score >= 2.0:
            quality = "Weak Alignment"
        else:
            quality = "No Clear Alignment"
        
        # Check if any cycle is in the End phase
        end_phase_cycles = [
            cycle for cycle in cycle_maturity.values() 
            if cycle['phase'] == "End Cycle"
        ]
        
        # Generate description
        if end_phase_cycles:
            # Warn about cycles nearing completion
            longest_end_cycle = max(
                end_phase_cycles, 
                key=lambda x: x['cycle_length']
            )
            
            description = (
                f"{aligned_count}/{total_cycles} cycles aligned {direction}. "
                f"Warning: {len(end_phase_cycles)} cycle(s) near completion, "
                f"including {longest_end_cycle['cycle_length']}-day cycle "
                f"({longest_end_cycle['completion_pct']:.1f}% complete)."
            )
        else:
            # Standard alignment description
            description = (
                f"{aligned_count}/{total_cycles} cycles aligned {direction}. "
                f"{quality}."
            )
        
        return {
            'score': alignment_score,
            'quality': quality,
            'direction': direction,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'description': description
        }
    
    def _evaluate_entry_conditions(self, 
                                 cycle_maturity: Dict[int, Dict[str, Any]], 
                                 alignment_score: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate entry conditions based on cycle maturity and alignment.
        
        Args:
            cycle_maturity: Dictionary mapping cycle length to maturity information
            alignment_score: Dictionary with alignment score information
            
        Returns:
            Dictionary with entry condition evaluation
        """
        # Default values
        favorable = False
        confidence = "Low"
        warnings = []
        
        # Check if any cycles are in End Phase
        end_phase_cycles = [
            cycle for cycle in cycle_maturity.values() 
            if cycle['phase'] == "End Cycle"
        ]
        
        if end_phase_cycles:
            longest_end_cycle = max(
                end_phase_cycles, 
                key=lambda x: x['cycle_length']
            )
            warnings.append(
                f"{longest_end_cycle['cycle_length']}-day cycle is "
                f"{longest_end_cycle['completion_pct']:.1f}% complete. "
                f"Potential reversal in {longest_end_cycle['days_remaining']} days."
            )
        
        # Check for fresh crossovers
        fresh_cycles = [
            cycle for cycle in cycle_maturity.values() 
            if cycle['phase'] == "Fresh Crossover"
        ]
        
        # Determine trade direction based on cycle alignment
        direction = "bullish" if alignment_score['bullish_count'] > alignment_score['bearish_count'] else "bearish"
        
        # Determine if entry conditions are favorable
        if fresh_cycles and alignment_score['score'] >= 6.0:
            favorable = True
            
            if alignment_score['score'] >= 8.0 and len(fresh_cycles) >= 2:
                confidence = "High"
            elif alignment_score['score'] >= 6.0 or len(fresh_cycles) >= 1:
                confidence = "Medium"
            else:
                confidence = "Low"
        
        # If any end-phase cycles present, reduce confidence
        if end_phase_cycles:
            if confidence == "High":
                confidence = "Medium"
            elif confidence == "Medium":
                confidence = "Low"
            
            if len(end_phase_cycles) >= 2:
                favorable = False
        
        # Generate recommendation
        if favorable:
            dir_text = "long" if direction == "bullish" else "short"
            recommendation = f"Entry conditions are favorable for {dir_text} trade. Confidence: {confidence}."
            if warnings:
                recommendation += f" Warning: {warnings[0]}"
        else:
            if end_phase_cycles:
                recommendation = "Entry conditions not favorable due to cycles nearing completion."
            elif alignment_score['score'] < 4.0:
                recommendation = "Entry conditions not favorable due to weak cycle alignment."
            else:
                recommendation = "Entry conditions not favorable. Consider waiting for fresh crossovers."
        
        return {
            'favorable': favorable,
            'confidence': confidence,
            'warnings': warnings,
            'recommendation': recommendation,
            'direction': direction  # Add direction to the entry conditions
        }
    
    def _generate_enhanced_guidance(self,
                                  result: ScanResult,
                                  cycle_maturity: Dict[int, Dict[str, Any]],
                                  entry_windows: Dict[str, Any],
                                  position_sizing: Dict[str, Any],
                                  trade_duration: Dict[str, Any],
                                  alignment_score: Dict[str, Any],
                                  entry_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced trade guidance based on all analysis factors.
        
        Args:
            result: ScanResult object
            cycle_maturity: Dictionary mapping cycle length to maturity information
            entry_windows: Dictionary with entry window recommendations  
            position_sizing: Dictionary with position sizing recommendations
            trade_duration: Dictionary with trade duration recommendations
            alignment_score: Dictionary with alignment score information
            entry_conditions: Dictionary with entry condition evaluation
            
        Returns:
            Dictionary with enhanced trade guidance
        """
        # Use the direction directly from entry_conditions for consistency
        direction = "long" if entry_conditions.get('direction', alignment_score['direction']) == "bullish" else "short"
        
        # Adjust stop loss based on position sizing recommendation
        original_stop = result.position_guidance['stop_loss']
        original_entry = result.position_guidance['entry_price']
        current_price = result.price
        
        # Calculate adjusted stop based on position sizing
        stop_adjustment = position_sizing['stop_adjustment']
        
        # For shorts, we need to swap the original stop and entry logic
        if direction == "long":
            # For long positions, risk is below entry
            # Original logic is appropriate - stops below entry price
            stop_distance = abs(original_entry - original_stop)
            adjusted_stop = current_price - (stop_distance * stop_adjustment)
            
            # Long target above current price
            target_distance = abs(result.position_guidance['target_price'] - original_entry)
            adjusted_target = current_price + (target_distance * stop_adjustment)
        else:
            # For short positions, risk is above entry
            # Need to place stop above current price
            stop_distance = abs(original_stop - original_entry)
            adjusted_stop = current_price + (stop_distance * stop_adjustment)
            
            # Short target below current price
            target_distance = abs(original_entry - result.position_guidance['target_price'])
            adjusted_target = current_price - (target_distance * stop_adjustment)
        
        # Calculate adjusted risk-reward ratio
        if direction == "long":
            # For longs: Risk is downside, reward is upside
            adjusted_risk = current_price - adjusted_stop
            adjusted_reward = adjusted_target - current_price
            
            # Ensure values are positive
            adjusted_risk = abs(adjusted_risk)
            adjusted_reward = abs(adjusted_reward)
            
            # Convert to percentages
            adjusted_risk_pct = (adjusted_risk / current_price) * 100
            adjusted_reward_pct = (adjusted_reward / current_price) * 100
        else:
            # For shorts: Risk is upside, reward is downside
            adjusted_risk = adjusted_stop - current_price
            adjusted_reward = current_price - adjusted_target
            
            # Ensure values are positive
            adjusted_risk = abs(adjusted_risk)
            adjusted_reward = abs(adjusted_reward)
            
            # Convert to percentages
            adjusted_risk_pct = (adjusted_risk / current_price) * 100
            adjusted_reward_pct = (adjusted_reward / current_price) * 100
        
        # Calculate R/R ratio - handle division by zero
        if adjusted_risk > 0:
            adjusted_rr_ratio = adjusted_reward / adjusted_risk
        else:
            adjusted_rr_ratio = 0
        
        # Prepare cycle maturity summary
        cycle_summary = []
        for cycle_length, data in sorted(cycle_maturity.items()):
            cycle_summary.append({
                'cycle_length': cycle_length,
                'phase': data['phase'],
                'completion_pct': data['completion_pct'],
                'days_remaining': data['days_remaining'],
                'is_bullish': data['is_bullish']
            })
        
        # Check for consistency between signal and enhanced strategy
        # If Analysis Results show Neutral signal, make sure trade recommendation reflects that
        if hasattr(result, 'signal') and result.signal.get('signal', '').lower() == 'neutral':
            # Override entry conditions to ensure consistency with main signal
            entry_conditions['favorable'] = False
            entry_conditions['recommendation'] = "Entry conditions not favorable (neutral signal detected)."
        
        # Generate final enhanced guidance
        enhanced_guidance = {
            'valid': True,
            'trade_direction': direction,
            'entry_windows': entry_windows,
            'position_sizing': position_sizing,
            'trade_duration': trade_duration,
            'alignment_score': alignment_score,
            'entry_conditions': entry_conditions,
            'cycle_maturity': cycle_summary,
            'closing_price': result.price,  # Add closing price
            
            # Enhanced position guidance
            'enhanced_position_guidance': {
                'position_size_pct': position_sizing['position_pct'],
                'entry_price': current_price,
                'adjusted_stop_loss': round(adjusted_stop, 2),
                'adjusted_target': round(adjusted_target, 2),
                'adjusted_risk_pct': round(adjusted_risk_pct, 2),
                'adjusted_reward_pct': round(adjusted_reward_pct, 2),
                'adjusted_rr_ratio': round(adjusted_rr_ratio, 2),
                'min_hold_days': trade_duration['min_hold'],
                'optimal_hold_days': trade_duration['optimal_hold'],
                'max_hold_days': trade_duration['max_hold']
            }
        }
        
        return enhanced_guidance


def get_enhanced_strategy(scan_result: ScanResult, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get enhanced entry/exit strategy recommendations.
    
    Args:
        scan_result: ScanResult object containing cycle and market data
        config: Configuration dictionary
        
    Returns:
        Dictionary with enhanced entry/exit recommendations
    """
    strategy = EnhancedEntryExitStrategy(config)
    return strategy.analyze(scan_result)