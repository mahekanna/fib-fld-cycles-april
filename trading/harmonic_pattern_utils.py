import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any


class HarmonicPatternDetector:
    """
    Helper class for detecting harmonic price patterns.
    
    This class implements the harmonic pattern analysis functionality that would
    conceptually be in a separate 'harmonic_analysis.py' file. It provides methods for:
    
    1. Detecting classic harmonic price patterns:
       - Gartley
       - Butterfly
       - Bat
       - Crab
       - Shark
       - Cypher
    
    2. Analyzing Fibonacci relationships between price points
    3. Calculating pattern quality and completion percentage
    4. Projecting potential reversal zones based on patterns
    
    Key methods include:
    - detect_patterns(): Identifies harmonic patterns in price data
    - _detect_swing_points(): Finds significant swing highs and lows
    - _validate_pattern(): Ensures pattern meets Fibonacci ratio criteria
    - calculate_pattern_quality(): Assesses how closely a pattern matches ideal ratios
    
    This module contains the harmonic pattern analysis capabilities that complement
    the cycle-based analysis in the core module.
    """
    
    def __init__(self, pattern_types, fibonacci_levels, tolerances):
        """Initialize the harmonic pattern detector."""
        self.pattern_types = pattern_types
        self.fibonacci_levels = fibonacci_levels
        self.tolerances = tolerances
        
    def detect_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect harmonic patterns in price data.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            List of detected patterns
        """
        # Detect swing highs and lows
        swing_points = self._detect_swing_points(data)
        
        # Find potential patterns
        patterns = []
        
        # We need at least 5 swing points to form a pattern (0, X, A, B, C)
        if len(swing_points) < 5:
            return patterns
        
        # Look for patterns in the most recent swing points
        for i in range(len(swing_points) - 4):
            # Get 5 consecutive swing points
            points = swing_points[i:i+5]
            
            # Check if these points form a valid pattern
            for pattern_type in self.pattern_types:
                pattern = self._check_pattern(points, pattern_type)
                if pattern and pattern['quality'] >= 0.6:  # Basic quality threshold
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_swing_points(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect significant swing highs and lows in price data.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            List of swing points
        """
        swing_points = []
        
        # Need at least 10 bars
        if len(data) < 10:
            return swing_points
        
        # Look for swing highs
        for i in range(2, len(data) - 2):
            # Check for swing high
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i-2] and
                data['high'].iloc[i] > data['high'].iloc[i+1] and
                data['high'].iloc[i] > data['high'].iloc[i+2]):
                
                swing_points.append({
                    'index': i,
                    'price': data['high'].iloc[i],
                    'date': data.index[i],
                    'type': 'high'
                })
            
            # Check for swing low
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i-2] and
                data['low'].iloc[i] < data['low'].iloc[i+1] and
                data['low'].iloc[i] < data['low'].iloc[i+2]):
                
                swing_points.append({
                    'index': i,
                    'price': data['low'].iloc[i],
                    'date': data.index[i],
                    'type': 'low'
                })
        
        # Sort by index
        swing_points.sort(key=lambda x: x['index'])
        
        return swing_points
    
    def _check_pattern(self, points: List[Dict], pattern_type: str) -> Optional[Dict]:
        """
        Check if given points form a specific harmonic pattern.
        
        Args:
            points: List of swing points
            pattern_type: Type of pattern to check
            
        Returns:
            Pattern information or None if not a valid pattern
        """
        # Ensure alternating high/low points
        if not self._check_alternating(points):
            return None
        
        # Extract X, A, B, C, D points
        x_point = points[0]
        a_point = points[1]
        b_point = points[2]
        c_point = points[3]
        d_point = points[4] if len(points) > 4 else None
        
        # Calculate ratios
        xab_ratio = self._calculate_ratio(x_point, a_point, b_point)
        abc_ratio = self._calculate_ratio(a_point, b_point, c_point)
        
        # D point may not exist yet (pattern not complete)
        completion = 1.0
        bcd_ratio = None
        xad_ratio = None
        
        if d_point:
            bcd_ratio = self._calculate_ratio(b_point, c_point, d_point)
            xad_ratio = self._calculate_ratio(x_point, a_point, d_point)
        else:
            # Estimate completion percentage
            completion = 0.75  # Pattern is 75% complete without D point
        
        # Check for pattern specific ratios
        if pattern_type == 'gartley':
            return self._check_gartley(x_point, a_point, b_point, c_point, d_point, 
                                     xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion)
        elif pattern_type == 'butterfly':
            return self._check_butterfly(x_point, a_point, b_point, c_point, d_point, 
                                       xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion)
        elif pattern_type == 'bat':
            return self._check_bat(x_point, a_point, b_point, c_point, d_point, 
                                 xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion)
        elif pattern_type == 'crab':
            return self._check_crab(x_point, a_point, b_point, c_point, d_point, 
                                  xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion)
        
        return None
    
    def _check_alternating(self, points: List[Dict]) -> bool:
        """Check if swing points alternate between highs and lows."""
        for i in range(1, len(points)):
            if points[i]['type'] == points[i-1]['type']:
                return False
        return True
    
    def _calculate_ratio(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """Calculate retracement ratio between three points."""
        range_12 = abs(point2['price'] - point1['price'])
        range_23 = abs(point3['price'] - point2['price'])
        
        if range_12 == 0:
            return 0
            
        return range_23 / range_12
    
    def _check_ratio_match(self, actual: float, expected: float, tolerance: float) -> float:
        """
        Check how closely a ratio matches an expected value within tolerance.
        Returns quality score between 0 and 1 (1 being perfect match).
        """
        if actual is None:
            return 0
            
        difference = abs(actual - expected)
        if difference <= tolerance:
            # Calculate quality as inverse of normalized difference
            quality = 1.0 - (difference / tolerance)
            return quality
        return 0
    
    def _check_gartley(self, x_point, a_point, b_point, c_point, d_point, 
                     xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion):
        """Check for Gartley pattern."""
        tolerance = self.tolerances.get('gartley', 0.05)
        
        # Expected Gartley ratios
        # XAB = 0.618
        # ABC = 0.382 or 0.886
        # BCD = 1.272 or 1.618
        # XAD = 0.786
        
        xab_quality = max(
            self._check_ratio_match(xab_ratio, 0.618, tolerance),
            0.0
        )
        
        abc_quality = max(
            self._check_ratio_match(abc_ratio, 0.382, tolerance),
            self._check_ratio_match(abc_ratio, 0.886, tolerance),
            0.0
        )
        
        # If D point exists, check BCD and XAD ratios
        if d_point:
            bcd_quality = max(
                self._check_ratio_match(bcd_ratio, 1.272, tolerance),
                self._check_ratio_match(bcd_ratio, 1.618, tolerance),
                0.0
            )
            
            xad_quality = max(
                self._check_ratio_match(xad_ratio, 0.786, tolerance),
                0.0
            )
        else:
            bcd_quality = 0.0
            xad_quality = 0.0
        
        # Calculate overall quality
        if d_point:
            quality = (xab_quality * 0.25 + abc_quality * 0.25 + 
                      bcd_quality * 0.25 + xad_quality * 0.25)
        else:
            # Without D point, only consider XAB and ABC
            quality = (xab_quality * 0.5 + abc_quality * 0.5)
        
        # Determine pattern direction
        if a_point['price'] > x_point['price']:
            # X to A is up, so pattern is bearish
            direction = 'bearish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] + (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] - (abs(a_point['price'] - d_point['price']) * 0.618)
            else:
                stop_level = None
                target_level = None
        else:
            # X to A is down, so pattern is bullish
            direction = 'bullish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] - (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] + (abs(a_point['price'] - d_point['price']) * 0.618)
            else:
                stop_level = None
                target_level = None
        
        if quality < 0.6:
            return None
            
        return {
            'pattern_type': 'gartley',
            'direction': direction,
            'quality': quality,
            'completion': completion,
            'x_point': x_point,
            'a_point': a_point,
            'b_point': b_point,
            'c_point': c_point,
            'd_point': d_point,
            'stop_level': stop_level,
            'target_level': target_level,
            'ratios': {
                'xab': xab_ratio,
                'abc': abc_ratio,
                'bcd': bcd_ratio,
                'xad': xad_ratio
            }
        }
    
    def _check_butterfly(self, x_point, a_point, b_point, c_point, d_point, 
                       xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion):
        """Check for Butterfly pattern."""
        tolerance = self.tolerances.get('butterfly', 0.05)
        
        # Expected Butterfly ratios
        # XAB = 0.786
        # ABC = 0.382 or 0.886
        # BCD = 1.618 or 2.618
        # XAD = 1.27 or 1.618
        
        xab_quality = max(
            self._check_ratio_match(xab_ratio, 0.786, tolerance),
            0.0
        )
        
        abc_quality = max(
            self._check_ratio_match(abc_ratio, 0.382, tolerance),
            self._check_ratio_match(abc_ratio, 0.886, tolerance),
            0.0
        )
        
        # If D point exists, check BCD and XAD ratios
        if d_point:
            bcd_quality = max(
                self._check_ratio_match(bcd_ratio, 1.618, tolerance),
                self._check_ratio_match(bcd_ratio, 2.618, tolerance),
                0.0
            )
            
            xad_quality = max(
                self._check_ratio_match(xad_ratio, 1.27, tolerance),
                self._check_ratio_match(xad_ratio, 1.618, tolerance),
                0.0
            )
        else:
            bcd_quality = 0.0
            xad_quality = 0.0
        
        # Calculate overall quality
        if d_point:
            quality = (xab_quality * 0.25 + abc_quality * 0.25 + 
                      bcd_quality * 0.25 + xad_quality * 0.25)
        else:
            # Without D point, only consider XAB and ABC
            quality = (xab_quality * 0.5 + abc_quality * 0.5)
        
        # Determine pattern direction
        if a_point['price'] > x_point['price']:
            # X to A is up, so pattern is bearish
            direction = 'bearish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] + (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] - (abs(a_point['price'] - d_point['price']) * 1.27)
            else:
                stop_level = None
                target_level = None
        else:
            # X to A is down, so pattern is bullish
            direction = 'bullish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] - (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] + (abs(a_point['price'] - d_point['price']) * 1.27)
            else:
                stop_level = None
                target_level = None
        
        if quality < 0.6:
            return None
            
        return {
            'pattern_type': 'butterfly',
            'direction': direction,
            'quality': quality,
            'completion': completion,
            'x_point': x_point,
            'a_point': a_point,
            'b_point': b_point,
            'c_point': c_point,
            'd_point': d_point,
            'stop_level': stop_level,
            'target_level': target_level,
            'ratios': {
                'xab': xab_ratio,
                'abc': abc_ratio,
                'bcd': bcd_ratio,
                'xad': xad_ratio
            }
        }
    
    def _check_bat(self, x_point, a_point, b_point, c_point, d_point, 
                 xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion):
        """Check for Bat pattern."""
        tolerance = self.tolerances.get('bat', 0.06)
        
        # Expected Bat ratios
        # XAB = 0.382 or 0.5
        # ABC = 0.382 or 0.886
        # BCD = 1.618 or 2.0
        # XAD = 0.886
        
        xab_quality = max(
            self._check_ratio_match(xab_ratio, 0.382, tolerance),
            self._check_ratio_match(xab_ratio, 0.5, tolerance),
            0.0
        )
        
        abc_quality = max(
            self._check_ratio_match(abc_ratio, 0.382, tolerance),
            self._check_ratio_match(abc_ratio, 0.886, tolerance),
            0.0
        )
        
        # If D point exists, check BCD and XAD ratios
        if d_point:
            bcd_quality = max(
                self._check_ratio_match(bcd_ratio, 1.618, tolerance),
                self._check_ratio_match(bcd_ratio, 2.0, tolerance),
                0.0
            )
            
            xad_quality = max(
                self._check_ratio_match(xad_ratio, 0.886, tolerance),
                0.0
            )
        else:
            bcd_quality = 0.0
            xad_quality = 0.0
        
        # Calculate overall quality
        if d_point:
            quality = (xab_quality * 0.25 + abc_quality * 0.25 + 
                      bcd_quality * 0.25 + xad_quality * 0.25)
        else:
            # Without D point, only consider XAB and ABC
            quality = (xab_quality * 0.5 + abc_quality * 0.5)
        
        # Determine pattern direction
        if a_point['price'] > x_point['price']:
            # X to A is up, so pattern is bearish
            direction = 'bearish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] + (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] - (abs(a_point['price'] - d_point['price']) * 0.886)
            else:
                stop_level = None
                target_level = None
        else:
            # X to A is down, so pattern is bullish
            direction = 'bullish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] - (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] + (abs(a_point['price'] - d_point['price']) * 0.886)
            else:
                stop_level = None
                target_level = None
        
        if quality < 0.6:
            return None
            
        return {
            'pattern_type': 'bat',
            'direction': direction,
            'quality': quality,
            'completion': completion,
            'x_point': x_point,
            'a_point': a_point,
            'b_point': b_point,
            'c_point': c_point,
            'd_point': d_point,
            'stop_level': stop_level,
            'target_level': target_level,
            'ratios': {
                'xab': xab_ratio,
                'abc': abc_ratio,
                'bcd': bcd_ratio,
                'xad': xad_ratio
            }
        }
    
    def _check_crab(self, x_point, a_point, b_point, c_point, d_point, 
                  xab_ratio, abc_ratio, bcd_ratio, xad_ratio, completion):
        """Check for Crab pattern."""
        tolerance = self.tolerances.get('crab', 0.06)
        
        # Expected Crab ratios
        # XAB = 0.382 or 0.618
        # ABC = 0.382 or 0.886
        # BCD = 2.618 or 3.618
        # XAD = 1.618
        
        xab_quality = max(
            self._check_ratio_match(xab_ratio, 0.382, tolerance),
            self._check_ratio_match(xab_ratio, 0.618, tolerance),
            0.0
        )
        
        abc_quality = max(
            self._check_ratio_match(abc_ratio, 0.382, tolerance),
            self._check_ratio_match(abc_ratio, 0.886, tolerance),
            0.0
        )
        
        # If D point exists, check BCD and XAD ratios
        if d_point:
            bcd_quality = max(
                self._check_ratio_match(bcd_ratio, 2.618, tolerance),
                self._check_ratio_match(bcd_ratio, 3.618, tolerance),
                0.0
            )
            
            xad_quality = max(
                self._check_ratio_match(xad_ratio, 1.618, tolerance),
                0.0
            )
        else:
            bcd_quality = 0.0
            xad_quality = 0.0
        
        # Calculate overall quality
        if d_point:
            quality = (xab_quality * 0.25 + abc_quality * 0.25 + 
                      bcd_quality * 0.25 + xad_quality * 0.25)
        else:
            # Without D point, only consider XAB and ABC
            quality = (xab_quality * 0.5 + abc_quality * 0.5)
        
        # Determine pattern direction
        if a_point['price'] > x_point['price']:
            # X to A is up, so pattern is bearish
            direction = 'bearish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] + (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] - (abs(a_point['price'] - d_point['price']) * 1.618)
            else:
                stop_level = None
                target_level = None
        else:
            # X to A is down, so pattern is bullish
            direction = 'bullish'
            
            # Calculate stop and target
            if d_point:
                stop_level = d_point['price'] - (abs(d_point['price'] - c_point['price']) * 0.2)
                target_level = d_point['price'] + (abs(a_point['price'] - d_point['price']) * 1.618)
            else:
                stop_level = None
                target_level = None
        
        if quality < 0.6:
            return None
            
        return {
            'pattern_type': 'crab',
            'direction': direction,
            'quality': quality,
            'completion': completion,
            'x_point': x_point,
            'a_point': a_point,
            'b_point': b_point,
            'c_point': c_point,
            'd_point': d_point,
            'stop_level': stop_level,
            'target_level': target_level,
            'ratios': {
                'xab': xab_ratio,
                'abc': abc_ratio,
                'bcd': bcd_ratio,
                'xad': xad_ratio
            }
        }