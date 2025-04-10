import pandas as pd
import numpy as np
import logging

# Import centralized logging system if available
try:
    from utils.logging_utils import get_component_logger
    DEFAULT_LOGGER = get_component_logger("core.scanner_system")
except ImportError:
    # Fallback to standard logging
    logging.basicConfig(level=logging.INFO)
    DEFAULT_LOGGER = logging.getLogger(__name__)
import time
from typing import List, Dict, Optional, Union, Tuple
import concurrent.futures
from dataclasses import dataclass

# Import our custom modules - using absolute imports
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.cycle_detection import CycleDetector
from core.fld_signal_generator import FLDCalculator, SignalGenerator 
from data.data_management import DataFetcher
from models.scan_parameters import ScanParameters
from models.scan_result import ScanResult
from visualization.price_charts import generate_plot_image


class FibCycleScanner:
    """
    Main orchestrator for Fibonacci Cycle analysis and signal generation.
    
    This class serves as the central orchestration engine for the entire system,
    implementing the scanning functionality that would conceptually be in a 
    'scanner.py' file. It provides methods for:
    
    1. Analyzing individual symbols with full cycle detection
    2. Batch scanning multiple symbols for efficient market screening
    3. Coordinating the entire analysis pipeline:
       - Data fetching
       - Cycle detection
       - FLD calculation
       - Signal generation
       - Position guidance
       - Result organization
    
    Key methods include:
    - analyze_symbol(): Performs complete analysis on a single symbol
    - scan_batch(): Analyzes multiple symbols efficiently
    - _process_data(): Handles the main data processing pipeline
    - rank_results(): Ranks batch results based on signal quality
    
    This module contains the high-level orchestration logic that integrates
    all the core analysis components into a cohesive system.
    """
    
    def __init__(self, 
                 config: Dict,
                 data_fetcher: Optional[DataFetcher] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the FibCycleScanner with configuration.
        
        Args:
            config: Configuration dictionary
            data_fetcher: Optional DataFetcher instance
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or DEFAULT_LOGGER
        
        # Create components
        self.data_fetcher = data_fetcher or DataFetcher(config)
        self.cycle_detector = CycleDetector(
            min_periods=config['analysis']['min_period'],
            max_periods=config['analysis']['max_period'],
            fibonacci_cycles=config['analysis']['fib_cycles'],
            power_threshold=config['analysis']['power_threshold'],
            cycle_tolerance=config['analysis']['cycle_tolerance'],
            detrend_method=config['analysis']['detrend_method'],
            window_function=config['analysis']['window_function']
        )
        self.fld_calculator = FLDCalculator(
            gap_threshold=config['analysis']['gap_threshold']
        )
        self.signal_generator = SignalGenerator(
            fld_calculator=self.fld_calculator,
            crossover_lookback=config['analysis']['crossover_lookback']
        )
    
    def analyze_symbol(self, 
                      parameters: ScanParameters) -> ScanResult:
        """
        Analyze a single symbol with the given parameters.
        
        Args:
            parameters: ScanParameters instance with analysis configuration
            
        Returns:
            ScanResult instance with analysis results
        """
        self.logger.info(f"Analyzing {parameters.symbol} on {parameters.exchange} "
                        f"with {parameters.interval} interval")
        
        start_time = time.time()
        
        try:
            # 1. Fetch price data - force refresh to ensure correct interval
            self.logger.info(f"Fetching data for {parameters.symbol} with interval {parameters.interval}")
            data = self.data_fetcher.get_data(
                symbol=parameters.symbol,
                exchange=parameters.exchange,
                interval=parameters.interval,
                lookback=parameters.lookback,
                price_source=parameters.price_source,
                force_download=True  # Force refresh to ensure we get correct interval data
            )
            
            if data is None or data.empty:
                self.logger.error(f"No data fetched for {parameters.symbol}")
                return ScanResult(
                    success=False,
                    symbol=parameters.symbol,
                    exchange=parameters.exchange,
                    interval=parameters.interval,
                    error="No data fetched"
                )
            
            self.logger.info(f"Successfully fetched {len(data)} bars of {parameters.symbol} data")
            
            # We'll store data in the result later when we create it
            
            # 2. Detect dominant cycles
            # Verify price source is available
            if parameters.price_source not in data.columns and parameters.price_source != 'price':
                self.logger.warning(f"Price source '{parameters.price_source}' not found in data. Checking for 'price' column.")
                if 'price' in data.columns:
                    self.logger.info(f"Using 'price' column (which should be {parameters.price_source}) for cycle detection")
                    price_series = data['price']
                else:
                    self.logger.warning(f"No 'price' column found. Falling back to 'close'")
                    price_series = data['close']
            else:
                # Use the specified price source
                if parameters.price_source == 'price':
                    self.logger.info(f"Using 'price' column for cycle detection")
                    price_series = data['price']
                else:
                    self.logger.info(f"Using '{parameters.price_source}' column for cycle detection")
                    price_series = data[parameters.price_source]
            
            # Log price series statistics for debugging
            self.logger.info(f"Price series for cycle detection - Min: {price_series.min():.2f}, Max: {price_series.max():.2f}, Mean: {price_series.mean():.2f}")
            
            # Detect cycles
            cycle_results = self.cycle_detector.detect_cycles(
                price_series=price_series,
                num_cycles=parameters.num_cycles
            )
            
            detected_cycles = cycle_results['cycles']
            cycle_powers = {cycle: power for cycle, power in zip(detected_cycles, cycle_results['powers'])}
            
            self.logger.info(f"Detected cycles for {parameters.symbol}: {detected_cycles}")
            
            # 3. Calculate FLDs and cycle states
            cycle_states = []
            
            # Log which price source is used for FLD calculations - the same used for cycle detection
            self.logger.info(f"Calculating FLDs using price source: {parameters.price_source}")
            
            for cycle_length in detected_cycles:
                # Calculate FLD
                fld_name = f'fld_{cycle_length}'
                
                # For historical compatibility, use hl2 for FLD calculations if available
                # This is important for correct FLD crossovers
                if parameters.price_source == 'hl2' or 'hl2' in data.columns:
                    self.logger.info(f"Using hl2 price for FLD calculation of cycle {cycle_length}")
                    data[fld_name] = self.fld_calculator.calculate_fld(
                        data['hl2'] if 'hl2' in data.columns else price_series, 
                        cycle_length
                    )
                else:
                    # Use same price series as cycle detection
                    self.logger.info(f"Using same price source ({parameters.price_source}) for FLD calculation of cycle {cycle_length}")
                    data[fld_name] = self.fld_calculator.calculate_fld(price_series, cycle_length)
                
                # Calculate cycle state
                cycle_state = self.fld_calculator.calculate_cycle_state(data, cycle_length)
                cycle_states.append(cycle_state)
            
            # 4. Generate synthetic cycle waves for visualization
            for cycle_length in detected_cycles:
                wave_name = f'cycle_wave_{cycle_length}'
                
                # Use all available data for visualization instead of just the last 250 points
                # For very large datasets, limit to a reasonable number of points to prevent performance issues
                sample_size = min(len(price_series), parameters.lookback)
                
                # Find optimal phase
                phase = self.cycle_detector.find_optimal_phase(
                    price_series.iloc[-sample_size:], 
                    cycle_length
                )
                
                # Generate wave
                wave = self.cycle_detector.generate_cycle_wave(
                    cycle_length=cycle_length,
                    num_points=sample_size,
                    phase_shift=phase,
                    amplitude=price_series.iloc[-sample_size:].std() * 0.25
                )
                
                # Scale and shift to price range
                price_min = price_series.iloc[-sample_size:].min()
                price_max = price_series.iloc[-sample_size:].max()
                price_mid = (price_max + price_min) / 2
                wave = wave + price_mid
                
                # Store in data
                wave_series = pd.Series(wave, index=data.index[-sample_size:])
                data[wave_name] = np.nan
                data.loc[data.index[-sample_size:], wave_name] = wave_series
            
            # 5. Analyze harmonic relationships
            harmonic_relationships = self.cycle_detector.analyze_harmonic_relationships(detected_cycles)
            
            # 6. Generate trading signal
            cycle_alignment = self.signal_generator.calculate_cycle_alignment(cycle_states)
            combined_strength = self.signal_generator.calculate_combined_strength(cycle_states, cycle_powers)
            signal_dict = self.signal_generator.determine_signal(combined_strength, cycle_alignment)
            
            # 7. Generate position guidance
            position_guidance = self.signal_generator.generate_position_guidance(
                signal_dict, 
                data, 
                cycle_states
            )
            
            # 8. Create visualization (optional)
            chart_image = None
            if parameters.generate_chart:
                # Use the requested lookback from parameters (respecting the fix)
                chart_image = generate_plot_image(
                    data=data,
                    symbol=parameters.symbol,
                    cycles=detected_cycles,
                    cycle_states=cycle_states,
                    signal=signal_dict,
                    lookback=parameters.lookback  # Use the user-defined lookback instead of hardcoded 250
                )
            
            # 9. Create result object
            execution_time = time.time() - start_time
            
            # Get the closing price - ALWAYS use the last close price to ensure consistency
            closing_price = data['close'].iloc[-1]
            self.logger.warning(f"PRICE SOURCE: Setting {parameters.symbol} price to {closing_price} from most recent 'close'")
            
            result = ScanResult(
                success=True,
                symbol=parameters.symbol,
                exchange=parameters.exchange,
                interval=parameters.interval,
                timestamp=pd.Timestamp.now(),
                execution_time=execution_time,
                lookback=parameters.lookback,  # Store the requested lookback for UI display
                price=closing_price,  # Use the verified closing price
                detected_cycles=detected_cycles,
                cycle_powers=cycle_powers,
                cycle_states=cycle_states,
                harmonic_relationships=harmonic_relationships,
                signal=signal_dict,
                position_guidance=position_guidance,
                data=data,  # Store the full data DataFrame
                chart_image=chart_image
            )
            
            self.logger.info(f"Analysis completed for {parameters.symbol} "
                           f"with signal {signal_dict['signal']} "
                           f"in {execution_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {parameters.symbol}: {str(e)}", exc_info=True)
            return ScanResult(
                success=False,
                symbol=parameters.symbol,
                exchange=parameters.exchange,
                interval=parameters.interval,
                error=str(e)
            )
    
    def scan_batch(self, 
                  parameters_list: List[ScanParameters],
                  max_workers: Optional[int] = None) -> List[ScanResult]:
        """
        Analyze multiple symbols in parallel.
        
        Args:
            parameters_list: List of ScanParameters instances
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of ScanResult instances
        """
        max_workers = max_workers or self.config['performance']['max_workers']
        results = []
        
        self.logger.info(f"Starting batch scan of {len(parameters_list)} symbols "
                        f"with {max_workers} workers")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self.analyze_symbol, params): params 
                for params in parameters_list
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed analysis for {params.symbol}")
                except Exception as e:
                    self.logger.error(f"Error in batch processing {params.symbol}: {str(e)}")
                    results.append(ScanResult(
                        success=False,
                        symbol=params.symbol,
                        exchange=params.exchange,
                        interval=params.interval,
                        error=str(e)
                    ))
        
        execution_time = time.time() - start_time
        self.logger.info(f"Batch scan completed in {execution_time:.2f} seconds")
        
        return results
    
    def filter_signals(self, 
                      results: List[ScanResult],
                      signal_type: Optional[str] = None,
                      min_confidence: Optional[str] = None,
                      min_alignment: Optional[float] = None) -> List[ScanResult]:
        """
        Filter scan results by signal criteria.
        
        Args:
            results: List of ScanResult instances
            signal_type: Optional signal type to filter for (e.g., "buy", "strong_buy")
            min_confidence: Optional minimum confidence level ("low", "medium", "high")
            min_alignment: Optional minimum cycle alignment score (0.0 to 1.0)
            
        Returns:
            Filtered list of ScanResult instances
        """
        filtered_results = results
        
        # Filter by success
        filtered_results = [r for r in filtered_results if r.success]
        
        # Filter by signal type
        if signal_type:
            filtered_results = [r for r in filtered_results if signal_type in r.signal['signal']]
        
        # Filter by confidence
        if min_confidence:
            confidence_levels = {"low": 0, "medium": 1, "high": 2}
            min_conf_level = confidence_levels.get(min_confidence, 0)
            filtered_results = [
                r for r in filtered_results 
                if confidence_levels.get(r.signal['confidence'], 0) >= min_conf_level
            ]
        
        # Filter by alignment
        if min_alignment is not None:
            filtered_results = [
                r for r in filtered_results 
                if r.signal['alignment'] >= min_alignment
            ]
        
        return filtered_results
    
    def rank_results(self, 
                    results: List[ScanResult],
                    ranking_factor: str = "strength") -> List[ScanResult]:
        """
        Rank scan results by specified factor.
        
        Args:
            results: List of ScanResult instances
            ranking_factor: Factor to rank by ("strength", "alignment", "risk_reward")
            
        Returns:
            List of ScanResult instances sorted by ranking factor
        """
        if not results:
            return []
        
        if ranking_factor == "strength":
            # Rank by absolute signal strength
            return sorted(results, key=lambda r: abs(r.signal['strength']), reverse=True)
        
        elif ranking_factor == "alignment":
            # Rank by cycle alignment
            return sorted(results, key=lambda r: r.signal['alignment'], reverse=True)
        
        elif ranking_factor == "risk_reward":
            # Rank by risk-reward ratio
            return sorted(results, key=lambda r: r.position_guidance['risk_reward_ratio'], reverse=True)
        
        else:
            # Default ranking by signal strength
            return sorted(results, key=lambda r: abs(r.signal['strength']), reverse=True)
