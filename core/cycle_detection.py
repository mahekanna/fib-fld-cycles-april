import numpy as np
import pandas as pd
import talib
from scipy import signal
from scipy.fft import fft
from typing import List, Tuple, Dict, Optional


class CycleDetector:
    """
    Enhanced cycle detection using multi-stage FFT analysis with improved
    detrending and harmonic relationship analysis.
    
    This class implements the core functionality of:
    1. Dominant cycle detection using Fast Fourier Transform
    2. Cycle extremes detection (peaks and troughs)
    3. Harmonic relationship analysis between different cycles
    4. Synthetic cycle wave generation for visualization and projection
    5. Cycle alignment and phase optimization
    
    Key methods include:
    - detect_cycles(): Identifies dominant cycles and their power
    - detect_cycle_extremes(): Finds turning points in price based on cycles
    - analyze_harmonic_relationships(): Analyzes relationships between detected cycles
    - generate_cycle_wave(): Creates synthetic waves for visualization and projection
    - find_optimal_phase(): Optimizes alignment between price and synthetic waves
    
    This module contains cycle detection and projection functionality that would
    conceptually be in separate 'cycle_detection.py' and 'cycle_projection.py' files.
    """
    
    def __init__(self, 
                 min_periods: int = 10, 
                 max_periods: int = 250,
                 fibonacci_cycles: List[int] = [21, 34, 55, 89, 144, 233],
                 power_threshold: float = 0.2,
                 cycle_tolerance: float = 0.15,
                 detrend_method: str = "diff",
                 window_function: str = "hanning"):
        """
        Initialize the cycle detector with configuration parameters.
        
        Args:
            min_periods: Minimum cycle length to detect
            max_periods: Maximum cycle length to detect
            fibonacci_cycles: List of Fibonacci cycle lengths to consider
            power_threshold: Minimum power threshold for detected cycles
            cycle_tolerance: Tolerance for matching cycles to Fibonacci values
            detrend_method: Method for detrending ("diff", "ema", "linear")
            window_function: Window function for FFT ("hanning", "hamming", "blackman")
        """
        self.min_periods = min_periods
        self.max_periods = max_periods
        self.fibonacci_cycles = fibonacci_cycles
        self.power_threshold = power_threshold
        self.cycle_tolerance = cycle_tolerance
        self.detrend_method = detrend_method
        self.window_function = window_function
    
    def detect_cycles(self, 
                      price_series: pd.Series, 
                      num_cycles: int = 3) -> Dict:
        """
        Detect dominant cycles in the price series.
        
        Args:
            price_series: Series of price data
            num_cycles: Number of dominant cycles to detect
            
        Returns:
            Dictionary containing detected cycles, powers, and metadata
        """
        # Remove NaN values
        clean_series = price_series.dropna()
        
        # Apply detrending based on selected method
        detrended = self._detrend_price(clean_series)
        
        # Apply window function to reduce spectral leakage
        windowed = self._apply_window(detrended)
        
        # Apply FFT to find dominant cycles
        fft_values = np.abs(fft(windowed.values))
        freqs = np.fft.fftfreq(len(windowed))
        
        # Focus on positive frequencies in our range of interest
        positive_freqs_idx = np.where((freqs > 0) & 
                                    (freqs < 1/self.min_periods) & 
                                    (freqs > 1/self.max_periods))[0]
        
        # Calculate power spectrum
        power_spectrum = fft_values[positive_freqs_idx]**2
        total_power = np.sum(power_spectrum)
        normalized_power = power_spectrum / total_power if total_power > 0 else power_spectrum
        
        # Sort by power to find dominant cycles
        sorted_idx = np.argsort(normalized_power)[::-1]
        
        # Get dominant periods and their powers
        dominant_periods = np.round(1/freqs[positive_freqs_idx][sorted_idx]).astype(int)
        dominant_powers = normalized_power[sorted_idx]
        
        # Filter to unique periods within range
        filtered_periods = []
        filtered_powers = []
        seen_periods = set()
        
        for period, power in zip(dominant_periods, dominant_powers):
            if self.min_periods <= period <= self.max_periods and period not in seen_periods:
                filtered_periods.append(period)
                filtered_powers.append(power)
                seen_periods.add(period)
        
        # Match to known Fibonacci cycles when close enough
        matched_cycles = []
        matched_powers = []
        
        for period, power in zip(filtered_periods, filtered_powers):
            if power < self.power_threshold:
                continue
                
            # Check if close to a Fibonacci cycle
            for fib_cycle in self.fibonacci_cycles:
                if abs(period - fib_cycle) / fib_cycle <= self.cycle_tolerance:
                    matched_cycles.append(fib_cycle)
                    matched_powers.append(power)
                    break
            else:
                matched_cycles.append(period)
                matched_powers.append(power)
        
        # Ensure we have enough cycles
        if len(matched_cycles) < num_cycles:
            additional_cycles = [c for c in self.fibonacci_cycles 
                              if c not in matched_cycles]
            for cycle in additional_cycles:
                if len(matched_cycles) < num_cycles:
                    matched_cycles.append(cycle)
                    # Assign a lower power to added cycles
                    matched_powers.append(self.power_threshold / 2)
        
        # Sort by power and take the top num_cycles
        if matched_cycles and matched_powers:
            sorted_indices = np.argsort(matched_powers)[::-1]
            matched_cycles = [matched_cycles[i] for i in sorted_indices[:num_cycles]]
            matched_powers = [matched_powers[i] for i in sorted_indices[:num_cycles]]
        
        return {
            'cycles': matched_cycles[:num_cycles],
            'powers': matched_powers[:num_cycles],
            'all_periods': filtered_periods,
            'all_powers': filtered_powers,
            'detrend_method': self.detrend_method,
            'window_function': self.window_function
        }
    
    def detect_cycle_extremes(self, 
                             price_series: pd.Series, 
                             cycle_length: int) -> Dict:
        """
        Detect cycle extremes (peaks and troughs) for a given cycle length.
        
        Args:
            price_series: Series of price data
            cycle_length: Length of the cycle to detect extremes for
            
        Returns:
            Dictionary containing peaks, troughs, and their properties
        """
        # Use peak detection with adaptive prominence
        prominence = np.std(price_series) * 0.5
        # Allow some flexibility in peak spacing 
        distance = int(cycle_length * 0.6)
        
        # Find peaks (tops)
        peaks, peak_properties = signal.find_peaks(
            price_series.values, 
            distance=distance, 
            prominence=prominence
        )
        
        # Find troughs (bottoms)
        inverted = -price_series.values
        troughs, trough_properties = signal.find_peaks(
            inverted, 
            distance=distance, 
            prominence=prominence
        )
        
        return {
            'peaks': peaks,
            'troughs': troughs,
            'peak_properties': peak_properties,
            'trough_properties': trough_properties
        }
    
    def generate_cycle_wave(self, 
                           cycle_length: int, 
                           num_points: int, 
                           phase_shift: float = 0,
                           amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a synthetic cycle wave for visualization and projection.
        
        Args:
            cycle_length: Length of the cycle
            num_points: Number of points to generate
            phase_shift: Phase shift in radians
            amplitude: Amplitude of the wave
            
        Returns:
            Numpy array containing the synthetic wave
        """
        # Create a sine wave with the specified cycle length
        x = np.linspace(0, 2 * np.pi * (num_points / cycle_length), num_points)
        return amplitude * np.sin(x + phase_shift)
    
    def find_optimal_phase(self, 
                          price_series: pd.Series, 
                          cycle_length: int) -> float:
        """
        Find the optimal phase shift for a cycle by correlating with price.
        
        Args:
            price_series: Series of price data
            cycle_length: Length of the cycle
            
        Returns:
            Optimal phase shift in radians
        """
        # Normalize price series
        norm_price = (price_series - price_series.mean()) / price_series.std()
        
        # Generate x values for the cycle
        x_cycle = np.linspace(0, 2 * np.pi * (len(norm_price) / cycle_length), len(norm_price))
        
        # Find the best phase shift by correlating with price
        best_corr = -1
        best_phase = 0
        
        for phase in np.linspace(0, 2*np.pi, 20):
            wave = np.sin(x_cycle + phase)
            corr = np.abs(np.corrcoef(norm_price.values, wave)[0, 1])
            
            if corr > best_corr:
                best_corr = corr
                best_phase = phase
        
        return best_phase
    
    def analyze_harmonic_relationships(self, cycles: List[int]) -> Dict:
        """
        Analyze harmonic relationships between detected cycles.
        
        Args:
            cycles: List of cycle lengths
            
        Returns:
            Dictionary of harmonic relationships between cycles
        """
        relationships = {}
        
        for i, cycle1 in enumerate(cycles):
            for j, cycle2 in enumerate(cycles):
                if i < j:
                    ratio = cycle2 / cycle1
                    error = abs(ratio - round(ratio))
                    
                    # Check if close to known harmonic ratios
                    harmonic = "None"
                    if abs(ratio - 1.618) < 0.1:
                        harmonic = "Golden Ratio (1.618)"
                    elif abs(ratio - 0.618) < 0.1:
                        harmonic = "Golden Ratio (0.618)"
                    elif abs(ratio - 2) < 0.1:
                        harmonic = "Octave (2:1)"
                    elif abs(ratio - 0.5) < 0.1:
                        harmonic = "Octave (1:2)"
                    elif abs(ratio - 1.5) < 0.1:
                        harmonic = "Perfect Fifth (3:2)"
                    elif abs(ratio - 0.667) < 0.1:
                        harmonic = "Perfect Fifth (2:3)"
                    elif abs(ratio - 1.414) < 0.1:
                        harmonic = "Square Root of 2"
                    elif abs(ratio - 0.707) < 0.1:
                        harmonic = "Square Root of 1/2"
                    
                    relationships[f"{cycle1}:{cycle2}"] = {
                        'ratio': ratio,
                        'harmonic': harmonic,
                        'precision': (1-error)*100,
                        'power': 1-error
                    }
        
        return relationships
    
    def _detrend_price(self, price_series: pd.Series) -> pd.Series:
        """
        Apply detrending to the price series based on the selected method.
        
        Args:
            price_series: Series of price data
            
        Returns:
            Detrended price series
        """
        if self.detrend_method == "diff":
            # Simple differencing
            return price_series.diff().dropna()
        
        elif self.detrend_method == "ema":
            # EMA detrending
            ema_period = len(price_series) // 4
            return price_series - talib.EMA(price_series, timeperiod=ema_period)
        
        elif self.detrend_method == "linear":
            # Linear regression detrending
            x = np.arange(len(price_series))
            y = price_series.values
            
            # Linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Remove trend
            return pd.Series(y - (m*x + c), index=price_series.index)
        
        else:
            # Default to simple detrending
            return price_series - price_series.mean()
    
    def _apply_window(self, series: pd.Series) -> pd.Series:
        """
        Apply a window function to reduce spectral leakage.
        
        Args:
            series: Series of data
            
        Returns:
            Windowed data series
        """
        if self.window_function == "hanning":
            window = np.hanning(len(series))
        elif self.window_function == "hamming":
            window = np.hamming(len(series))
        elif self.window_function == "blackman":
            window = np.blackman(len(series))
        else:
            # Default to rectangular window (no windowing)
            return series
        
        return pd.Series(series.values * window, index=series.index)
