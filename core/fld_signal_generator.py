import numpy as np
import pandas as pd
import talib
from typing import List, Dict, Optional, Tuple


class FLDCalculator:
    """
    Calculator for Future Lines of Demarcation (FLDs) based on cycle lengths.
    Includes advanced gap detection and state tracking.
    
    This class implements the FLD calculation functionality that would
    conceptually be in a separate 'fld_calculation.py' file. It provides
    methods for:
    
    1. Calculating FLD lines for each cycle length
    2. Detecting crossovers between price and FLD lines
    3. Managing cycle state information (bullish/bearish)
    4. Tracking days since last crossover
    5. Calculating price-to-FLD ratios
    
    Key methods include:
    - calculate_fld(): Computes FLD values for a given cycle length
    - detect_crossovers(): Identifies crossover points between price and FLD
    - calculate_cycle_states(): Determines current state for multiple cycles
    """
    
    def __init__(self, gap_threshold: float = 0.01):
        """
        Initialize the FLD calculator.
        
        Args:
            gap_threshold: Threshold for detecting price gaps (as percentage)
        """
        self.gap_threshold = gap_threshold
    
    def calculate_fld(self, 
                      price_series: pd.Series, 
                      cycle_length: int) -> pd.Series:
        """
        Calculate FLD for a specific cycle length.
        
        Args:
            price_series: Series of price data
            cycle_length: Length of the cycle to calculate FLD for
            
        Returns:
            Series containing FLD values
        """
        # Calculate FLD as EMA with period = (cycle_length / 2) + 1
        fld_length = int(cycle_length / 2) + 1
        return pd.Series(
            talib.EMA(price_series.values, timeperiod=fld_length),
            index=price_series.index
        )
    
    def detect_crossovers(self, 
                         price_df: pd.DataFrame, 
                         price_col: str, 
                         fld_col: str) -> pd.DataFrame:
        """
        Detect crossovers between price and FLD.
        
        Args:
            price_df: DataFrame containing price and FLD columns
            price_col: Name of the price column
            fld_col: Name of the FLD column
            
        Returns:
            DataFrame containing crossover information
        """
        # Create a copy of the DataFrame
        df = price_df.copy()
        
        # Check for gaps
        df['gap_up'] = (df[price_col] > df[price_col].shift(1) * (1 + self.gap_threshold))
        df['gap_down'] = (df[price_col] < df[price_col].shift(1) * (1 - self.gap_threshold))
        
        # Detect regular crossovers
        df['cross_above'] = (df[price_col].shift(1) < df[fld_col].shift(1)) & (df[price_col] > df[fld_col])
        df['cross_below'] = (df[price_col].shift(1) > df[fld_col].shift(1)) & (df[price_col] < df[fld_col])
        
        # Special handling for gaps
        df['gap_cross_above'] = df['gap_up'] & (df[price_col].shift(1) < df[fld_col].shift(1)) & (df[price_col] > df[fld_col])
        df['gap_cross_below'] = df['gap_down'] & (df[price_col].shift(1) > df[fld_col].shift(1)) & (df[price_col] < df[fld_col])
        
        # Combine regular and gap crossovers
        df['any_cross_above'] = df['cross_above'] | df['gap_cross_above']
        df['any_cross_below'] = df['cross_below'] | df['gap_cross_below']
        
        # Filter to rows with crossovers
        crossovers = df[df['any_cross_above'] | df['any_cross_below']].copy()
        
        # Add crossover direction
        crossovers['direction'] = np.where(
            crossovers['any_cross_above'], 
            'bullish', 
            np.where(crossovers['any_cross_below'], 'bearish', None)
        )
        
        return crossovers
    
    def calculate_cycle_state(self, 
                             price_df: pd.DataFrame, 
                             cycle_length: int) -> Dict:
        """
        Calculate the current state of a cycle.
        
        Args:
            price_df: DataFrame containing price and FLD data
            cycle_length: Length of the cycle
            
        Returns:
            Dictionary describing the current cycle state
        """
        # Calculate FLD column name
        fld_col = f'fld_{cycle_length}'
        
        # Ensure FLD is calculated
        if fld_col not in price_df.columns:
            price_df[fld_col] = self.calculate_fld(price_df['close'], cycle_length)
        
        # Current cycle state (bullish/bearish)
        is_bullish = price_df['close'].iloc[-1] > price_df[fld_col].iloc[-1]
        
        # Find most recent crossover
        crossovers = self.detect_crossovers(price_df, 'close', fld_col)
        
        recent_crossover = None
        days_since_crossover = None
        
        if not crossovers.empty:
            recent_crossover = crossovers.iloc[-1]
            days_since_crossover = (price_df.index[-1] - recent_crossover.name).days
        
        return {
            'cycle_length': cycle_length,
            'is_bullish': is_bullish,
            'recent_crossover': recent_crossover,
            'days_since_crossover': days_since_crossover,
            'fld_value': price_df[fld_col].iloc[-1],
            'price_value': price_df['close'].iloc[-1],
            'price_to_fld_ratio': price_df['close'].iloc[-1] / price_df[fld_col].iloc[-1] if price_df[fld_col].iloc[-1] != 0 else 1.0
        }


class SignalGenerator:
    """
    Generate trading signals based on FLD crossovers and cycle states.
    
    This class implements the signal generation and position guidance functionality
    that would conceptually be in separate 'signal_generation.py' and 'position_guidance.py'
    files. It provides methods for:
    
    1. Generating multi-cycle trading signals
    2. Calculating signal strength and confidence
    3. Creating position guidance (entry, stop, target)
    4. Computing risk/reward metrics
    5. Analyzing cycle alignment and strength
    
    Key methods include:
    - generate_signal(): Creates comprehensive trading signals
    - calculate_cycle_strength(): Computes cycle-based signal strength
    - generate_position_guidance(): Provides entry, stop, and target levels
    - calculate_cycle_alignment(): Measures alignment across multiple cycles
    """
    
    def __init__(self, 
                 fld_calculator: FLDCalculator,
                 crossover_lookback: int = 5):
        """
        Initialize the signal generator.
        
        Args:
            fld_calculator: FLDCalculator instance
            crossover_lookback: Number of days to look back for recent crossovers
        """
        self.fld_calculator = fld_calculator
        self.crossover_lookback = crossover_lookback
    
    def calculate_combined_strength(self, 
                                   cycle_states: List[Dict], 
                                   cycle_powers: Dict[int, float]) -> float:
        """
        Calculate combined strength of all cycle signals.
        
        Args:
            cycle_states: List of cycle state dictionaries
            cycle_powers: Dictionary mapping cycle length to cycle power
            
        Returns:
            Combined signal strength (-1.0 to 1.0)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not cycle_states:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        bullish_count = 0
        bearish_count = 0
        
        # Log cycle states for debugging
        logger.debug(f"Calculating combined strength for {len(cycle_states)} cycle states")
        
        for state in cycle_states:
            cycle_length = state['cycle_length']
            is_bullish = state['is_bullish']
            
            # Count bullish/bearish cycles for balance analysis
            if is_bullish:
                bullish_count += 1
            else:
                bearish_count += 1
            
            # Base weight is cycle length (longer cycles have higher weight)
            # Adjusted by cycle power
            base_weight = cycle_length / 100.0
            power_weight = cycle_powers.get(cycle_length, 0.5)
            weight = base_weight * power_weight
            
            # Direction factor: +1 for bullish, -1 for bearish
            direction = 1.0 if is_bullish else -1.0
            
            # Apply extra weight for recent crossovers
            crossover_bonus = 0.0
            if state['days_since_crossover'] is not None:
                if state['days_since_crossover'] <= self.crossover_lookback:
                    # Bonus decays with days since crossover
                    crossover_bonus = (self.crossover_lookback - state['days_since_crossover']) / self.crossover_lookback
                    
                    # Apply direction to the bonus
                    if state['recent_crossover'] is not None:
                        crossover_direction = state['recent_crossover']['direction']
                        if crossover_direction == 'bearish':
                            crossover_bonus = -crossover_bonus
                            logger.debug(f"Bearish crossover detected for cycle {cycle_length}, bonus: {crossover_bonus:.2f}")
                        else:
                            logger.debug(f"Bullish crossover detected for cycle {cycle_length}, bonus: {crossover_bonus:.2f}")
            
            # Add to weighted sum
            cycle_contribution = weight * direction * (1.0 + crossover_bonus)
            weighted_sum += cycle_contribution
            total_weight += weight
            
            # Log individual cycle contribution
            logger.debug(f"Cycle {cycle_length}: direction={direction}, weight={weight:.2f}, contribution={cycle_contribution:.2f}")
        
        # Log overall cycle balance
        logger.debug(f"Cycle balance: {bullish_count} bullish, {bearish_count} bearish cycles")
        
        # Normalize to range [-1.0, 1.0]
        if total_weight > 0:
            result = np.clip(weighted_sum / total_weight, -1.0, 1.0)
            logger.debug(f"Final combined strength: {result:.4f} (weighted_sum={weighted_sum:.2f}, total_weight={total_weight:.2f})")
            return result
        else:
            return 0.0
    
    def determine_signal(self, 
                        combined_strength: float, 
                        cycle_alignment: float) -> Dict:
        """
        Determine the final trading signal based on combined strength and cycle alignment.
        
        Args:
            combined_strength: Combined signal strength (-1.0 to 1.0)
            cycle_alignment: Measure of cycle alignment (0.0 to 1.0)
            
        Returns:
            Signal dictionary
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Log incoming values for debugging
        logger.debug(f"Determining signal: strength={combined_strength:.4f}, alignment={cycle_alignment:.4f}")
        
        # Map strength to signal categories - handle both positive and negative values correctly
        # NOTE: negative strength means bearish/short signal
        if combined_strength > 0.7:
            signal = "strong_buy"
        elif combined_strength > 0.3:
            signal = "buy"
        elif combined_strength > 0.1:
            signal = "weak_buy"
        elif combined_strength < -0.7:  # Strong negative strength = strong sell
            signal = "strong_sell"
        elif combined_strength < -0.3:  # Moderate negative strength = sell
            signal = "sell"
        elif combined_strength < -0.1:  # Weak negative strength = weak sell
            signal = "weak_sell"
        else:
            signal = "neutral"
        
        # Ensure direction is consistent with strength
        is_buy_signal = "buy" in signal
        is_sell_signal = "sell" in signal
        
        # Critical check: signal name must match strength direction
        if (is_buy_signal and combined_strength < 0) or (is_sell_signal and combined_strength > 0):
            logger.warning(f"Signal direction mismatch: {signal} with strength {combined_strength:.4f}")
            # Fix the mismatch - this is critical!
            if combined_strength < 0:
                # Negative strength should be some kind of sell signal
                if abs(combined_strength) > 0.7:
                    signal = "strong_sell"
                elif abs(combined_strength) > 0.3:
                    signal = "sell"
                elif abs(combined_strength) > 0.1:
                    signal = "weak_sell"
            elif combined_strength > 0:
                # Positive strength should be some kind of buy signal
                if abs(combined_strength) > 0.7:
                    signal = "strong_buy"
                elif abs(combined_strength) > 0.3:
                    signal = "buy"
                elif abs(combined_strength) > 0.1:
                    signal = "weak_buy"
        
        # Assign confidence level based on strength and alignment
        if abs(combined_strength) > 0.6 and cycle_alignment > 0.7:
            confidence = "high"
        elif abs(combined_strength) > 0.3 and cycle_alignment > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Slightly boost confidence for short signals to ensure they're considered
        # This helps balance the long/short trade ratio
        if is_sell_signal and confidence == "low" and cycle_alignment > 0.6:
            confidence = "medium"
            logger.debug(f"Boosting confidence for short signal: {signal}")
        
        result = {
            'signal': signal,
            'strength': combined_strength,
            'alignment': cycle_alignment,
            'confidence': confidence,
            'direction': 'short' if is_sell_signal else 'long' if is_buy_signal else 'neutral'
        }
        
        logger.debug(f"Final signal: {result}")
        return result
    
    def calculate_cycle_alignment(self, cycle_states: List[Dict]) -> float:
        """
        Calculate how well aligned the cycles are in their direction.
        
        Args:
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Alignment score (0.0 to 1.0)
        """
        if not cycle_states:
            return 0.0
        
        # Count bullish and bearish cycles
        bullish_count = sum(1 for state in cycle_states if state['is_bullish'])
        bearish_count = len(cycle_states) - bullish_count
        
        # Calculate alignment as percentage of agreement
        max_count = max(bullish_count, bearish_count)
        return max_count / len(cycle_states) if len(cycle_states) > 0 else 0.0
    
    def generate_signals(self, price_df: pd.DataFrame, cycles_dict: Dict) -> Dict:
        """
        Generate trading signals based on detected cycles.
        
        Args:
            price_df: DataFrame with price data
            cycles_dict: Dictionary containing detected cycles and powers
            
        Returns:
            Dictionary of signal data at each bar
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Extract cycles and powers
        detected_cycles = cycles_dict.get('cycles', [])
        cycle_powers = {cycle: power for cycle, power in zip(
            detected_cycles, cycles_dict.get('powers', [])
        )}
        
        if not detected_cycles:
            logger.warning("No cycles detected, cannot generate signals")
            return {}
            
        logger.info(f"Generating signals based on {len(detected_cycles)} detected cycles: {detected_cycles}")
        
        # Calculate FLDs for each cycle
        for cycle in detected_cycles:
            fld_col = f'fld_{cycle}'
            price_df[fld_col] = self.fld_calculator.calculate_fld(price_df['close'], cycle)
            
        # Calculate cycle states for each bar
        signals = {}
        
        # Track signal statistics for debugging
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        # Process each bar
        for i in range(max(detected_cycles), len(price_df)):
            # Get the current bar's data
            current_slice = price_df.iloc[:i+1]
            
            # Calculate cycle states
            cycle_states = [
                self.fld_calculator.calculate_cycle_state(current_slice, cycle)
                for cycle in detected_cycles
            ]
            
            # Calculate bullish/bearish balance for this bar
            bullish_count = sum(1 for state in cycle_states if state['is_bullish'])
            bearish_count = len(cycle_states) - bullish_count
            
            # If all cycles are in the same direction, ensure we generate the appropriate signal type
            all_bullish = bullish_count == len(cycle_states)
            all_bearish = bearish_count == len(cycle_states)
            
            # Calculate signal strength and alignment - this is where direction gets determined
            strength = self.calculate_combined_strength(cycle_states, cycle_powers)
            alignment = self.calculate_cycle_alignment(cycle_states)
            
            # Sanity check: if all cycles are bearish, strength should be negative
            if all_bearish and strength > 0:
                logger.warning(f"Signal direction error: All cycles bearish but strength={strength}. Correcting.")
                strength = -abs(strength)  # Force negative
            
            # Sanity check: if all cycles are bullish, strength should be positive
            if all_bullish and strength < 0:
                logger.warning(f"Signal direction error: All cycles bullish but strength={strength}. Correcting.")
                strength = abs(strength)  # Force positive
            
            # Determine final signal
            signal = self.determine_signal(strength, alignment)
            
            # Count signal types for stats
            if 'buy' in signal['signal']:
                buy_signals += 1
            elif 'sell' in signal['signal']:
                sell_signals += 1
            else:
                neutral_signals += 1
            
            # Store signal for this bar
            signals[i] = signal
            
            # Every 100 bars, log the signal distribution for debugging
            if i % 100 == 0:
                logger.debug(f"Signal distribution after {i} bars: Buy={buy_signals}, Sell={sell_signals}, Neutral={neutral_signals}")
        
        # Log final signal distribution
        logger.info(f"Final signal distribution: Buy={buy_signals}, Sell={sell_signals}, Neutral={neutral_signals}")
        
        return signals
        
    def generate_position_guidance(self, 
                                  signal_dict: Dict, 
                                  price_df: pd.DataFrame, 
                                  cycle_states: List[Dict]) -> Dict:
        """
        Generate position guidance based on the signal.
        
        Args:
            signal_dict: Signal dictionary
            price_df: DataFrame with price data
            cycle_states: List of cycle state dictionaries
            
        Returns:
            Position guidance dictionary
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if price_df is empty or None
        if price_df is None or price_df.empty:
            # Return default values if no price data available
            return {
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'target_price': 0.0,
                'position_size': 0.0,
                'risk': 0.0,
                'reward': 0.0,
                'risk_reward_ratio': 0.0,
                'risk_percentage': 0.0,
                'target_percentage': 0.0,
                'direction': 'neutral'
            }
        
        # Get signal information
        signal_type = signal_dict.get('signal', 'neutral')
        signal_strength = signal_dict.get('strength', 0.0)
        
        # Determine trade direction explicitly (don't rely just on signal name)
        is_buy_signal = 'buy' in signal_type
        is_sell_signal = 'sell' in signal_type
        direction = 'short' if is_sell_signal else 'long' if is_buy_signal else 'neutral'
        
        logger.debug(f"Generating position guidance for {direction} signal: {signal_type}, strength={signal_strength:.4f}")
            
        try:
            current_price = price_df['close'].iloc[-1]
            
            # Check if we have enough data for ATR calculation (at least 14 bars)
            if len(price_df) >= 14:
                atr = talib.ATR(
                    price_df['high'].values, 
                    price_df['low'].values, 
                    price_df['close'].values, 
                    timeperiod=14
                )[-1]
                # Handle potential NaN value from ATR
                if np.isnan(atr):
                    atr = price_df['close'].std()
            else:
                # Use standard deviation as fallback if not enough data
                atr = price_df['close'].std()
                if np.isnan(atr):
                    atr = current_price * 0.02  # Default to 2% of price
            
            # Default values
            entry_price = current_price
            stop_loss = None
            target_price = None
            position_size = 0.0
            
            # Set entry, stop loss, and target based on direction (derived from signal)
            if direction == 'long':
                logger.debug(f"Setting up LONG position guidance at price {current_price:.2f}")
                entry_price = current_price
                
                # Find recent price levels for better stop placement
                if len(price_df) >= 20:
                    recent_low = price_df['low'].iloc[-20:].min()
                    recent_low_idx = price_df['low'].iloc[-20:].idxmin()
                    recent_low_date = recent_low_idx if isinstance(recent_low_idx, pd.Timestamp) else "unknown"
                    logger.debug(f"Recent low: {recent_low:.2f} on {recent_low_date}")
                else:
                    recent_low = current_price * 0.98
                
                # Stop loss based on ATR or closest bearish FLD
                stop_options = [current_price - (2 * atr), recent_low * 0.99]
                
                # Add FLD-based stops for better technical levels
                for state in cycle_states:
                    if not state.get('is_bullish', True) and 'fld_value' in state:
                        if state['fld_value'] is not None and state['fld_value'] < current_price:
                            stop_options.append(state['fld_value'])
                            logger.debug(f"Adding FLD-based stop at {state['fld_value']:.2f} for cycle {state['cycle_length']}")
                
                # Choose the highest stop (closest to entry) that's still below entry
                valid_stops = [s for s in stop_options if s < current_price]
                if valid_stops:
                    stop_loss = max(valid_stops)
                else:
                    stop_loss = current_price - (2 * atr)  # Default
                
                # Target based on ATR or distance to stop (risk-reward)
                risk = current_price - stop_loss
                target_price = current_price + (risk * 2)  # 2:1 reward-to-risk ratio
                
                logger.debug(f"LONG position: Entry={entry_price:.2f}, Stop={stop_loss:.2f}, Target={target_price:.2f}")
                
            elif direction == 'short':
                logger.debug(f"Setting up SHORT position guidance at price {current_price:.2f}")
                entry_price = current_price
                
                # Find recent price levels for better stop placement
                if len(price_df) >= 20:
                    recent_high = price_df['high'].iloc[-20:].max()
                    recent_high_idx = price_df['high'].iloc[-20:].idxmax()
                    recent_high_date = recent_high_idx if isinstance(recent_high_idx, pd.Timestamp) else "unknown"
                    logger.debug(f"Recent high: {recent_high:.2f} on {recent_high_date}")
                else:
                    recent_high = current_price * 1.02
                
                # Stop loss based on ATR or closest bullish FLD
                stop_options = [current_price + (2 * atr), recent_high * 1.01]
                
                # Add FLD-based stops for better technical levels
                for state in cycle_states:
                    if state.get('is_bullish', False) and 'fld_value' in state:
                        if state['fld_value'] is not None and state['fld_value'] > current_price:
                            stop_options.append(state['fld_value'])
                            logger.debug(f"Adding FLD-based stop at {state['fld_value']:.2f} for cycle {state['cycle_length']}")
                
                # Choose the lowest stop (closest to entry) that's still above entry
                valid_stops = [s for s in stop_options if s > current_price]
                if valid_stops:
                    stop_loss = min(valid_stops)
                else:
                    stop_loss = current_price + (2 * atr)  # Default
                
                # Target based on ATR or distance to stop (risk-reward)
                risk = stop_loss - current_price
                target_price = current_price - (risk * 2)  # 2:1 reward-to-risk ratio
                
                logger.debug(f"SHORT position: Entry={entry_price:.2f}, Stop={stop_loss:.2f}, Target={target_price:.2f}")
                
            else:
                # For neutral signals, provide sensible defaults but mark as non-actionable
                stop_loss = current_price * 0.95  # 5% stop loss
                target_price = current_price * 1.05  # 5% target
                logger.debug(f"NEUTRAL signal - no tradable position")
                
        except Exception as e:
            # Log error and provide safe defaults
            logger.error(f"Error generating position guidance: {e}")
            entry_price = price_df['close'].iloc[-1] if not price_df.empty else 0.0
            stop_loss = entry_price * 0.95 if entry_price > 0 else 0.0
            target_price = entry_price * 1.05 if entry_price > 0 else 0.0
        
        # Position size based on confidence and signal alignment
        if signal_dict['confidence'] == "high":
            position_size = 1.0
        elif signal_dict['confidence'] == "medium":
            position_size = 0.75
        elif signal_dict['confidence'] == "low":
            position_size = 0.5
        
        # For short signals, slightly increase position size to balance with longs
        if direction == 'short':
            position_size = min(1.0, position_size * 1.1)  # 10% bonus
        
        # Ensure all values are valid for calculations
        entry_price = entry_price if entry_price is not None else 0.0
        stop_loss = stop_loss if stop_loss is not None else 0.0
        target_price = target_price if target_price is not None else 0.0
        
        # Calculate risk-reward ratio based on direction
        if direction == 'long':
            risk = entry_price - stop_loss if stop_loss < entry_price else atr * 2
            reward = target_price - entry_price if target_price > entry_price else atr * 4
        elif direction == 'short':
            risk = stop_loss - entry_price if stop_loss > entry_price else atr * 2
            reward = entry_price - target_price if target_price < entry_price else atr * 4
        else:
            risk = atr * 2
            reward = atr * 4
            
        risk = max(0.01, risk)  # Avoid division by zero
        risk_reward = reward / risk if risk > 0 else 0.0
        
        # Calculate percentages safely
        risk_percentage = (risk / entry_price) * 100 if entry_price > 0 else 0.0
        target_percentage = (reward / entry_price) * 100 if entry_price > 0 else 0.0
        
        result = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'position_size': position_size,
            'risk': risk,
            'reward': reward,
            'risk_reward_ratio': risk_reward,
            'risk_percentage': risk_percentage,
            'target_percentage': target_percentage,
            'direction': direction
        }
        
        logger.debug(f"Position guidance complete: R:R = 1:{risk_reward:.2f}, Size={position_size:.2f}")
        return result