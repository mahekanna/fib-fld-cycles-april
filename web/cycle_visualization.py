"""Cycle visualization module for market analysis"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import signal
from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, List, Optional

# Import core cycle detection tools
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.cycle_detection import CycleDetector
from models.scan_result import ScanResult
from data.data_management import DataFetcher

# Helper functions for harmonic analysis
def _get_harmonic_name(ratio: float) -> str:
    """
    Determine harmonic name from ratio
    
    Args:
        ratio: Cycle ratio (larger:smaller)
        
    Returns:
        String name of harmonic relationship
    """
    if abs(ratio - 1.618) < 0.1:
        return "Golden Ratio (1.618)"
    elif abs(ratio - 0.618) < 0.1:
        return "Golden Ratio (0.618)"
    elif abs(ratio - 2) < 0.1:
        return "Octave (2:1)"
    elif abs(ratio - 0.5) < 0.1:
        return "Octave (1:2)"
    elif abs(ratio - 1.5) < 0.1:
        return "Perfect Fifth (3:2)"
    elif abs(ratio - 0.667) < 0.1:
        return "Perfect Fifth (2:3)"
    elif abs(ratio - 1.414) < 0.1:
        return "Square Root of 2"
    elif abs(ratio - 0.707) < 0.1:
        return "Square Root of 1/2"
    else:
        # Round to nearest 1/8
        rounded = round(ratio * 8) / 8
        num = int(rounded * 8)
        denom = 8
        # Simplify fraction
        from math import gcd
        g = gcd(num, denom)
        if g > 0:
            num, denom = num // g, denom // g
        return f"~{num}:{denom}"

def _get_harmonic_precision(ratio: float) -> float:
    """
    Calculate precision of harmonic relationship
    
    Args:
        ratio: Cycle ratio (larger:smaller)
        
    Returns:
        Precision as percentage
    """
    # Check standard harmonics
    standard_ratios = [0.5, 0.618, 0.667, 0.707, 1.0, 1.414, 1.5, 1.618, 2.0]
    closest = min(standard_ratios, key=lambda x: abs(x - ratio))
    error = abs(ratio - closest) / closest
    return (1 - error) * 100


def create_cycle_visualization(result: ScanResult, save_image: bool = False):
    """
    Create an interactive visualization of market cycles.
    
    Args:
        result: ScanResult object containing cycle data
        
    Returns:
        Dash component for cycle visualization
    """
    if not result.success:
        return html.Div([
            html.H3("Error in Cycle Analysis"),
            html.P(result.error or "Unknown error occurred")
        ])
    
    # Get the number of detected cycles for proper subplot allocation
    num_cycles = len(result.detected_cycles) if hasattr(result, 'detected_cycles') else 0
    
    # Create visualization with real data - add a separate pane for each cycle wave
    # This makes it much clearer to see individual cycles
    row_heights = [0.4]  # Price panel
    row_heights.extend([0.15] * num_cycles)  # Individual cycle panels
    row_heights.append(0.2)  # Power spectrum panel
    
    # Track cycle correlations for later display
    cycle_correlations = {}
    
    # Create subplot titles - Price panel, individual cycle panels, power spectrum
    subplot_titles = ["Price with FLDs"]
    if num_cycles > 0:
        subplot_titles.extend([f"{cycle} Cycle Wave" for cycle in result.detected_cycles])
    subplot_titles.append("Power Spectrum")
    
    fig = make_subplots(
        rows=2 + num_cycles,  # Price + individual cycle panels + power spectrum
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,  # Reduce spacing to fit more panels
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # Get price data from the result
    data = getattr(result, 'data', None)
    
    # If there's no data in the result, fetch it using the result parameters
    if data is None:
        try:
            from data.data_management import DataFetcher
            from utils.config import load_config
            
            config_path = "config/config.json"  # Default config path 
            config = load_config(config_path)  # Load config with explicit path
            data_fetcher = DataFetcher(config)
            
            data = data_fetcher.get_data(
                symbol=result.symbol,
                exchange=result.exchange,
                interval=result.interval,
                lookback=getattr(result, 'lookback', 1000)  # Use lookback from result or default to 1000
            )
        except Exception as e:
            # Log error and continue with limited visualization
            print(f"Error fetching data for visualization: {str(e)}")
    
    # If we have data, create a complete visualization
    if data is not None and not data.empty:
        # Main price plot with cycles overlay - use all available data
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add FLD lines to price chart
        colors = ['rgba(255,99,71,0.8)', 'rgba(65,105,225,0.8)', 'rgba(50,205,50,0.8)']
        for i, cycle in enumerate(result.detected_cycles):
            fld_col = f'fld_{cycle}'
            if fld_col in data.columns:
                color_idx = i % len(colors)
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[fld_col].dropna(),
                        mode='lines',
                        name=f"FLD-{cycle}",
                        line=dict(width=2, color=colors[color_idx])
                    ),
                    row=1, col=1
                )
        
        # Add cycle waves in individual panels for clearer visualization
        # Enhanced color scheme with more distinct colors
        cycle_colors = [
            'rgba(255,140,0,1)',      # Dark Orange
            'rgba(30,144,255,1)',     # Dodger Blue
            'rgba(50,205,50,1)',      # Lime Green
            'rgba(255,105,180,1)',    # Hot Pink
            'rgba(138,43,226,1)',     # Blue Violet
            'rgba(0,191,255,1)'       # Deep Sky Blue
        ]
        
        # First, check if we need to regenerate cycle waves with improved alignment
        price_source = 'close'  # Default source
        if hasattr(result, 'price_source'):
            price_source = result.price_source
        
        price_series = data[price_source].dropna()
        
        # Calculate mid-point reference for normalizing amplitude
        for i, cycle in enumerate(result.detected_cycles):
            wave_col = f'cycle_wave_{cycle}'
            color_idx = i % len(cycle_colors)
            cycle_power = result.cycle_powers.get(cycle, 0)
            
            # For our enhanced visualization, we always need to generate the future projection
            # Get the existing cycle data if it exists
            if wave_col in data.columns:
                print(f"DEBUG: Using existing cycle wave for historical data (cycle {cycle})")
                cycle_data = data[wave_col].dropna()
            else:
                print(f"DEBUG: No existing cycle wave found (cycle {cycle}) - generating it")
                # We need to generate the wave since it doesn't exist
                # This is critical for all cycles to show properly
                x_cycle_hist = np.linspace(0, 2 * np.pi * (len(data) / cycle), len(data))
                # Use a default phase shift if we haven't calculated one yet
                phase_shift_default = 0
                historical_wave_default = np.sin(x_cycle_hist + phase_shift_default)
                data[wave_col] = historical_wave_default
                cycle_data = data[wave_col].dropna()
                
            # We always regenerate for future projection, even if cycle data exists
            # The projection_col is our signal to create a future projection
            
            # Use the cycle detection helper to find peaks and troughs
            # (using np from global import)
            
            # Detect cycle peaks for phase alignment
            prominence = np.std(price_series) * 0.3
            distance = int(cycle * 0.6)  # Allow some flexibility in peak spacing
            peaks, _ = signal.find_peaks(price_series.values, distance=distance, prominence=prominence)
            
            # Align phase with recent price action
            recent_peaks = [p for p in peaks if p >= len(price_series) - min(500, len(price_series))]
            
            if len(recent_peaks) > 0:
                # Use the last peak for phase alignment
                last_peak = recent_peaks[-1]
                phase_shift = -2 * np.pi * (last_peak / cycle)  # Negative to align peaks
            else:
                # If no peaks detected, try correlation-based approach
                recent_data = price_series.iloc[-min(100, len(price_series)):]
                normalized_price = (recent_data - recent_data.mean()) / recent_data.std()
                
                best_corr = -1
                phase_shift = 0
                for phase in np.linspace(0, 2*np.pi, 20):
                    # Generate trial wave
                    x_cycle = np.linspace(0, 2 * np.pi * (len(normalized_price) / cycle), len(normalized_price))
                    wave = np.sin(x_cycle + phase)
                    corr = abs(np.corrcoef(normalized_price.values, wave)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        phase_shift = phase
            
            # Scale the wave to price range based on cycle power
            price_min = price_series.min()
            price_max = price_series.max()
            price_range = price_max - price_min
            price_mid = (price_max + price_min) / 2
            
            # Make amplitude proportional to cycle power (stronger cycles get larger amplitude)
            amplitude_factor = 0.15 + (cycle_power * 0.25)  # 15% to 40% of price range
            
            # Generate the wave for full data length PLUS future projection
            # Add 20% more points for future projection
            total_length = int(len(data) * 1.2)
            future_points = total_length - len(data)
            
            # Generate full cycle wave including future projection
            x_cycle = np.linspace(0, 2 * np.pi * (total_length / cycle), total_length)
            complete_wave = np.sin(x_cycle + phase_shift)
            
            # Split into historical and projected portions
            historical_wave = complete_wave[:len(data)]
            projected_wave = complete_wave[len(data):]
            
            # We need to create both versions:
            # 1. A normalized wave (centered at 0) for display in separate panels (MUST cross zero)
            # 2. A price-aligned wave (centered at price_mid) for correlation analysis
            
            # 1. Historical normalized wave for panel display - ensure it crosses zero
            data[wave_col] = historical_wave
            
            # 2. Store projected wave in a separate column
            projection_col = f'cycle_projection_{cycle}'
            data[projection_col] = np.nan  # Initialize with NaN
            
            # Create a date range for future projection
            future_dates = []
            if len(data.index) > 0:
                # Get the average time delta between points
                time_deltas = []
                for j in range(1, min(10, len(data.index))):
                    delta = (data.index[j] - data.index[j-1]).total_seconds()
                    time_deltas.append(delta)
                
                if time_deltas:
                    avg_delta = np.mean(time_deltas)
                    # Create future dates based on last date
                    last_date = data.index[-1]
                    future_dates = [last_date + pd.Timedelta(seconds=avg_delta * (i+1)) 
                                  for i in range(future_points)]
                    
                    # Add projection wave to dataframe with future dates
                    projection_series = pd.Series(projected_wave, index=future_dates)
                    data = pd.concat([data, pd.DataFrame({projection_col: projection_series})])
            
            # Price-scaled wave for correlation with actual price
            # This is for internal use - Scale AND center around price midpoint
            price_aligned_wave = (historical_wave * (price_range * amplitude_factor)) + price_mid
            
            # Calculate correlation with price for recent data
            recent_length = min(100, len(price_series))
            recent_price = price_series.iloc[-recent_length:]
            # Use the historical wave for correlation calculation
            recent_historical_wave = historical_wave[-recent_length:]
            
            # Normalize both series for proper correlation measurement
            norm_price = (recent_price - recent_price.mean()) / recent_price.std()
            norm_wave = (recent_historical_wave - recent_historical_wave.mean()) / recent_historical_wave.std()
            
            # Calculate correlation
            correlation = np.corrcoef(norm_price, norm_wave)[0, 1]
            cycle_correlations[cycle] = correlation
            
            # Store the next peak and trough predictions
            # Find zero crossings in the projected wave to identify future turning points
            zero_crossings_future = []
            for j in range(1, len(projected_wave)):
                if (projected_wave[j-1] <= 0 and projected_wave[j] > 0) or \
                   (projected_wave[j-1] >= 0 and projected_wave[j] < 0):
                    zero_crossings_future.append(j)
            
            # Find peaks and troughs in the projected wave
            peaks_future, _ = signal.find_peaks(projected_wave, prominence=0.1)
            troughs_future, _ = signal.find_peaks(-projected_wave, prominence=0.1)
            
            # Store projections
            # Store the next predicted turn dates
            next_turns = []
            
            # Convert peak indices to dates
            if len(peaks_future) > 0 and len(future_dates) > 0:
                for peak_idx in peaks_future:
                    if peak_idx < len(future_dates):
                        next_turns.append({
                            'type': 'peak',
                            'date': future_dates[peak_idx],
                            'cycle': cycle
                        })
                        
            # Convert trough indices to dates
            if len(troughs_future) > 0 and len(future_dates) > 0:
                for trough_idx in troughs_future:
                    if trough_idx < len(future_dates):
                        next_turns.append({
                            'type': 'trough',
                            'date': future_dates[trough_idx],
                            'cycle': cycle
                        })
            
            # Sort by date
            next_turns.sort(key=lambda x: x['date'])
            
            # Store for later use
            if not hasattr(result, 'cycle_projections'):
                result.cycle_projections = {}
            result.cycle_projections[cycle] = next_turns
            
            # Debug print to verify projections are being created
            print(f"DEBUG: Added {len(next_turns)} projections for cycle {cycle}")
            for turn_idx, turn in enumerate(next_turns[:3]):  # Print first 3
                print(f"  {turn_idx+1}. {turn['type']} on {turn['date'].strftime('%Y-%m-%d')}")

            # We've already added the projection to the dataframe above
            
            # Use the wave data (whether original or regenerated)
            cycle_data = data[wave_col].dropna()
            
            # Get correlation if not already calculated
            if cycle not in cycle_correlations and len(cycle_data) > 0:
                recent_length = min(100, len(price_series))
                if len(cycle_data) >= recent_length:
                    recent_price = price_series.iloc[-recent_length:]
                    recent_wave = cycle_data.iloc[-recent_length:]
                    
                    # Normalize for correlation
                    norm_price = (recent_price - recent_price.mean()) / recent_price.std() 
                    norm_wave = (recent_wave - recent_wave.mean()) / recent_wave.std()
                    
                    # Calculate correlation
                    correlation = np.corrcoef(norm_price, norm_wave)[0, 1]
                    cycle_correlations[cycle] = correlation
            
            # Ensure cycle data is properly normalized for its own panel
            # For better visualization and comparison between cycles, we normalize
            # the amplitude to be between -1 and 1
            if len(cycle_data) > 0:
                # Make sure we're getting a valid number
                try:
                    max_val = max(abs(cycle_data.max()), abs(cycle_data.min()))
                    if max_val > 0:
                        # Normalize to range [-1, 1] for consistent scaling across cycle panels
                        normalized_cycle_data = cycle_data / max_val
                    else:
                        # If no valid max/min, regenerate cycle data
                        print(f"DEBUG: No valid amplitude for cycle {cycle}, regenerating")
                        x_cycle_hist = np.linspace(0, 2 * np.pi * (len(data) / cycle), len(data))
                        historical_wave_default = np.sin(x_cycle_hist)
                        normalized_cycle_data = pd.Series(historical_wave_default, index=data.index[:len(historical_wave_default)])
                except Exception as e:
                    # Handle any error by regenerating the cycle data
                    print(f"DEBUG: Error normalizing cycle {cycle}: {str(e)}, regenerating")
                    x_cycle_hist = np.linspace(0, 2 * np.pi * (len(data) / cycle), len(data))
                    historical_wave_default = np.sin(x_cycle_hist)
                    normalized_cycle_data = pd.Series(historical_wave_default, index=data.index[:len(historical_wave_default)])
            else:
                # If no cycle data, regenerate it
                print(f"DEBUG: No cycle data for cycle {cycle}, regenerating")
                x_cycle_hist = np.linspace(0, 2 * np.pi * (len(data) / cycle), len(data))
                historical_wave_default = np.sin(x_cycle_hist)
                normalized_cycle_data = pd.Series(historical_wave_default, index=data.index[:len(historical_wave_default)])
                
            # Find the proper amplitude for visualization
            max_amplitude = 1.0  # Fixed reference for normalized data
            
            # Add cycle wave in its own panel (row 2+i) for better visibility
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized_cycle_data,  # Use normalized data for panel
                    mode='lines',
                    name=f"{cycle} Cycle",
                    line=dict(width=2.5, color=cycle_colors[color_idx])
                ),
                row=2+i, col=1  # Each cycle gets its own row
            )
            
            # Add projection if available
            projection_col = f'cycle_projection_{cycle}'
            if projection_col in data.columns:
                projection_data = data[projection_col].dropna()
                if not projection_data.empty:
                    # Normalize the projection data
                    if len(projection_data) > 0:
                        max_val = max(abs(projection_data.max()), abs(projection_data.min()))
                        if max_val > 0:
                            normalized_projection = projection_data / max_val
                        else:
                            normalized_projection = projection_data
                    else:
                        normalized_projection = projection_data
                    
                    # Add the projection as a dashed line
                    fig.add_trace(
                        go.Scatter(
                            x=normalized_projection.index,
                            y=normalized_projection,
                            mode='lines',
                            name=f"{cycle} Projection",
                            line=dict(
                                width=2, 
                                color=cycle_colors[color_idx],
                                dash='dash'
                            ),
                            showlegend=False
                        ),
                        row=2+i, col=1
                    )
                    
                    # Add markers for projected turns
                    if hasattr(result, 'cycle_projections') and cycle in result.cycle_projections:
                        for turn in result.cycle_projections[cycle]:
                            # Find the value at this date
                            if turn['date'] in normalized_projection.index:
                                y_val = normalized_projection.loc[turn['date']]
                                
                                # Add marker for peak or trough
                                marker_symbol = 'triangle-up' if turn['type'] == 'peak' else 'triangle-down'
                                marker_color = 'green' if turn['type'] == 'peak' else 'red'
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=[turn['date']],
                                        y=[y_val],
                                        mode='markers',
                                        marker=dict(
                                            symbol=marker_symbol,
                                            size=10,
                                            color=marker_color,
                                            line=dict(width=1, color='white')
                                        ),
                                        name=f"Projected {turn['type'].capitalize()}",
                                        showlegend=False,
                                        hovertemplate=f"Projected {turn['type'].capitalize()}: %{{x}} <extra></extra>"
                                    ),
                                    row=2+i, col=1
                                )
            
            # Add zero reference line
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=0,
                y1=0,
                line=dict(color="gray", width=1, dash="dot"),
                row=2+i, col=1
            )
            
            # Add positive threshold line (optional)
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=max_amplitude * 0.7,
                y1=max_amplitude * 0.7,
                line=dict(color="rgba(0,255,0,0.3)", width=1, dash="dot"),
                row=2+i, col=1
            )
            
            # Add negative threshold line (optional)
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=-max_amplitude * 0.7,
                y1=-max_amplitude * 0.7,
                line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dot"),
                row=2+i, col=1
            )
            
            # Add cycle information as annotation
            correlation_text = ""
            if cycle in cycle_correlations:
                correlation_text = f" | Corr: {cycle_correlations[cycle]:.2f}"
                
            fig.add_annotation(
                x=data.index[int(len(data.index) * 0.05)],  # Position at 5% from left
                y=max_amplitude * 0.8,  # Position at 80% of max amplitude
                text=f"Power: {cycle_power:.2f}{correlation_text}",
                showarrow=False,
                font=dict(color=cycle_colors[color_idx]),
                bgcolor="rgba(0,0,0,0.3)",
                row=2+i, col=1
            )
            
            # Add price peaks/troughs markers to show alignment with cycle
            # First, find important price peaks and troughs
            prominence = np.std(price_series) * 0.3
            distance = int(cycle * 0.6)  # Based on the cycle length
            price_peaks, _ = signal.find_peaks(price_series.values, distance=distance, prominence=prominence)
            price_troughs, _ = signal.find_peaks(-price_series.values, distance=distance, prominence=prominence)
            
            # Limit to a reasonable number of markers
            max_markers = 10
            if len(price_peaks) > max_markers:
                indices = np.round(np.linspace(0, len(price_peaks) - 1, max_markers)).astype(int)
                price_peaks = price_peaks[indices]
            if len(price_troughs) > max_markers:
                indices = np.round(np.linspace(0, len(price_troughs) - 1, max_markers)).astype(int)
                price_troughs = price_troughs[indices]
                
            # Add markers for price peaks (showing alignment to cycle waves)
            for peak_idx in price_peaks:
                if 0 <= peak_idx < len(data.index):
                    # Add small upward triangles at price peaks
                    fig.add_trace(
                        go.Scatter(
                            x=[data.index[peak_idx]],
                            y=[0.9],  # Position near top of panel
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=8,
                                color=cycle_colors[color_idx],
                                line=dict(width=1, color='white')
                            ),
                            name=f"Price Peak",
                            showlegend=False
                        ),
                        row=2+i, col=1
                    )
            
            # Add cycle boundary markers
            # Detect zero crossings which indicate cycle boundaries
            zero_crossings = []
            for j in range(1, len(cycle_data)):
                if (cycle_data.iloc[j-1] <= 0 and cycle_data.iloc[j] > 0) or \
                   (cycle_data.iloc[j-1] >= 0 and cycle_data.iloc[j] < 0):
                    zero_crossings.append(j)
            
            # Add vertical lines at cycle boundaries (limit to not overcrowd)
            max_markers = 10  # Maximum number of markers to show
            if len(zero_crossings) > max_markers:
                # Sample evenly
                indices = np.round(np.linspace(0, len(zero_crossings) - 1, max_markers)).astype(int)
                zero_crossings = [zero_crossings[i] for i in indices]
            
            for crossing_idx in zero_crossings:
                if 0 <= crossing_idx < len(data.index):
                    fig.add_shape(
                        type="line",
                        x0=data.index[crossing_idx],
                        x1=data.index[crossing_idx],
                        y0=-max_amplitude * 0.8,
                        y1=max_amplitude * 0.8,
                        line=dict(color=cycle_colors[color_idx], width=1, dash="dot"),
                        opacity=0.3,
                        row=2+i, col=1
                    )
        
        # Add power spectrum in the second subplot
        # Create a more comprehensive power spectrum based on cycle_powers
        cycles = list(result.cycle_powers.keys())
        powers = list(result.cycle_powers.values())
        
        # Calculate the correct row for power spectrum (after all cycle panels)
        power_spectrum_row = 2 + num_cycles
        
        # Create custom colors based on power values (stronger cycles get brighter colors)
        # Use colors that match the cycle panels for easier association
        cycle_bar_colors = []
        hover_data = []
        bar_text = []
        
        # Create Fibonacci reference cycles for comparison
        fib_cycles = [5, 8, 13, 21, 34, 55, 89, 144, 233]
        
        # Add Fibonacci markers to the power spectrum
        for fib_cycle in fib_cycles:
            if fib_cycle > min(cycles) - 10 and fib_cycle < max(cycles) + 10:
                # Add vertical line at Fibonacci cycle
                fig.add_shape(
                    type="line",
                    x0=fib_cycle,
                    x1=fib_cycle,
                    y0=0,
                    y1=max(powers) * 1.1,  # Extend above the highest bar
                    line=dict(color="rgba(255,215,0,0.3)", width=1, dash="dash"),  # Gold color
                    row=power_spectrum_row, col=1
                )
                
                # Add small annotation
                fig.add_annotation(
                    x=fib_cycle,
                    y=max(powers) * 1.05,
                    text=f"Fib {fib_cycle}",
                    showarrow=False,
                    font=dict(size=8, color="rgba(255,215,0,0.8)"),
                    row=power_spectrum_row, col=1
                )
        
        # Process each cycle for the power spectrum
        for i, cycle in enumerate(cycles):
            # Get the color index and other metrics
            try:
                # Find the index in detected_cycles to use the same color
                detected_idx = result.detected_cycles.index(cycle)
                # Use the same color as in the cycle panel, but adjust opacity based on power
                power = powers[i]
                color_idx = detected_idx % len(cycle_colors)
                base_color = cycle_colors[color_idx].replace('1)', f'{0.5 + power/2})')
                cycle_bar_colors.append(base_color)
                
                # Add correlation to hover data if available
                hover_text = f"Cycle: {cycle}<br>Power: {power:.3f}"
                if cycle in cycle_correlations:
                    hover_text += f"<br>Correlation: {cycle_correlations[cycle]:.3f}"
                    
                # Check if this cycle is close to a Fibonacci cycle
                closest_fib = min(fib_cycles, key=lambda x: abs(x - cycle))
                fib_proximity = abs(cycle - closest_fib) / closest_fib
                if fib_proximity <= 0.1:  # Within 10% of a Fibonacci cycle
                    hover_text += f"<br>Near Fibonacci: {closest_fib} ({fib_proximity*100:.1f}%)"
                    bar_text.append(f"{power:.2f} ≈{closest_fib}")
                else:
                    bar_text.append(f"{power:.2f}")
                    
                hover_data.append(hover_text)
                
            except ValueError:
                # If the cycle is not in detected_cycles, use a default color
                cycle_bar_colors.append('rgba(102,102,255,0.8)')
                hover_data.append(f"Cycle: {cycle}<br>Power: {powers[i]:.3f}")
                bar_text.append(f"{powers[i]:.2f}")
        
        # Add the power spectrum bars with enhanced information
        fig.add_trace(
            go.Bar(
                x=cycles,
                y=powers,
                name="Cycle Power",
                marker_color=cycle_bar_colors,
                marker=dict(
                    line=dict(width=1, color='rgba(255,255,255,0.4)')
                ),
                text=bar_text,  # Show power values and Fibonacci proximity
                textposition="outside",  # Position text above bars
                hovertemplate="%{customdata}",
                customdata=hover_data
            ),
            row=power_spectrum_row, col=1  # Position after all cycle panels
        )
        
        # Add harmonics of dominant cycle
        if len(result.detected_cycles) > 0:
            dominant_cycle = result.detected_cycles[0]  # First cycle is dominant
            
            # Add harmonic markers (1/2, 2x, 3x of dominant cycle)
            harmonics = [
                {"value": dominant_cycle/2, "name": "½× Harmonic"},
                {"value": dominant_cycle*2, "name": "2× Harmonic"},
                {"value": dominant_cycle*3, "name": "3× Harmonic"}
            ]
            
            for harmonic in harmonics:
                # Check if harmonic is within range of displayed cycles
                if harmonic["value"] > min(cycles) - 10 and harmonic["value"] < max(cycles) + 10:
                    # Add vertical line for harmonic
                    fig.add_shape(
                        type="line",
                        x0=harmonic["value"],
                        x1=harmonic["value"],
                        y0=0,
                        y1=max(powers) * 0.7,
                        line=dict(color="rgba(255,105,180,0.4)", width=1, dash="dot"),  # Pink
                        row=power_spectrum_row, col=1
                    )
                    
                    # Add annotation for harmonic
                    fig.add_annotation(
                        x=harmonic["value"],
                        y=max(powers) * 0.4,
                        text=harmonic["name"],
                        showarrow=False,
                        font=dict(size=8, color="rgba(255,105,180,0.8)"),
                        row=power_spectrum_row, col=1
                    )
        
        # Add threshold line to indicate significant cycle power
        fig.add_shape(
            type="line",
            x0=min(cycles) - 5,
            x1=max(cycles) + 5,
            y0=0.2,  # Threshold for significant cycle power
            y1=0.2,
            line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"),
            row=power_spectrum_row, col=1
        )
        
        # Add annotation to explain the threshold
        fig.add_annotation(
            x=min(cycles),
            y=0.22,
            text="Significance Threshold",
            showarrow=False,
            font=dict(size=10),
            row=power_spectrum_row, col=1
        )
        
        # Update subplot axis labels
        
        # Update axis labels for price chart
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        # Update power spectrum labels
        fig.update_xaxes(title_text="Cycle Length", row=power_spectrum_row, col=1)
        fig.update_yaxes(title_text="Relative Power", row=power_spectrum_row, col=1)
        
        # Note: Individual cycle panel labels are set later in a separate loop
    
    # Update layout regardless of data availability
    # Calculate appropriate height based on number of cycles
    base_height = 400  # Base height for price panel and power spectrum
    cycle_panel_height = 150  # Height per cycle panel
    total_height = base_height + (num_cycles * cycle_panel_height)
    
    fig.update_layout(
        height=total_height,  # Dynamic height based on number of cycle panels
        title=f"{result.symbol} Cycle Analysis ({result.interval})",
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=60, t=80, b=50),  # Improved margins
        paper_bgcolor='rgba(0,0,0,0.8)',
        plot_bgcolor='rgba(0,0,0,0.2)'
    )
    
    # Update y-axis titles for each cycle panel with additional info
    for i, cycle in enumerate(result.detected_cycles):
        # Get cycle power for additional context
        cycle_power = result.cycle_powers.get(cycle, 0)
        # Update the y-axis title with cycle info
        fig.update_yaxes(
            title_text=f"{cycle} Cycle (Power: {cycle_power:.2f})",
            row=2+i, col=1,
            tickfont=dict(size=10),
            title_font=dict(size=12)  # Correct property name is title_font, not titlefont
        )
    
    # Save chart image if requested
    if save_image:
        try:
            import os
            # Create directory if it doesn't exist
            os.makedirs("./assets/chart_images", exist_ok=True)
            
            # Generate filename based on symbol and interval
            filename = f"./assets/chart_images/{result.symbol}_{result.interval}_cycles.png"
            
            # Save the figure as an image
            fig.write_image(filename, width=1200, height=800, scale=2)
            print(f"Saved cycle chart image to {filename}")
        except Exception as e:
            print(f"Error saving chart image: {e}")
    
    # Create the component
    return html.Div([
        dbc.Card([
            dbc.CardHeader(html.H3("Cycle Visualization")),
            dbc.CardBody([
                html.P([
                    "The visualization includes: ",
                    html.Ul([
                        html.Li("Price chart with FLD lines (top panel)"),
                        html.Li(f"Individual cycle waves ({', '.join([str(c) for c in result.detected_cycles])}) in separate panels for clarity"),
                        html.Li("Power spectrum showing relative strength of each cycle (bottom panel)")
                    ]),
                    html.Strong("How to interpret: "),
                    html.Ul([
                        html.Li("Each cycle panel shows a normalized wave (oscillating between -1 and 1) with phase-optimized alignment to price action"),
                        html.Li("ZERO LINE CROSSINGS (center) indicate momentum shifts - these are critical turning points!"),
                        html.Li("Dashed lines show future cycle projections with predicted turns"),
                        html.Li("Triangles indicate peak (▲) and trough (▼) points - green/red markers for future projections"),
                        html.Li("Cycle power (in metrics) shows dominance in the market (>0.4 is significant)"),
                        html.Li("Correlation (Corr) measures how closely each cycle matches recent price movements"),
                        html.Li("Green/red threshold lines at ±0.7 indicate potential overbought/oversold regions")
                    ]),
                    html.Strong("Power Spectrum Features: "),
                    html.Ul([
                        html.Li("Bars show relative strength of detected cycles in price data"),
                        html.Li("Gold vertical lines mark Fibonacci cycle lengths"),
                        html.Li("Pink lines show harmonics of the dominant cycle"),
                        html.Li("Hover over bars for detailed cycle information"),
                        html.Li("Cycles close to Fibonacci values are marked with ≈ symbol")
                    ])
                ], className="text-muted mb-3"),
                dcc.Graph(figure=fig, id="cycle-graph"),
                dbc.Button(
                    "Save Chart Image", 
                    id="save-cycle-chart-btn", 
                    color="info", 
                    className="mt-2",
                    n_clicks=0
                ),
                html.Div(id="save-result-message", className="mt-2"),
                
                # Add a key/legend box to explain the visualization elements
                dbc.Card([
                    dbc.CardHeader("Visualization Key"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Span("▬▬▬▬", style={"color": cycle_colors[0]}), 
                                    " Historical cycle"
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("▬ ▬ ▬", style={"color": cycle_colors[0]}), 
                                    " Projected cycle"
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("▬▬▬▬", style={"color": "gray"}), 
                                    " Zero line (momentum shift)"
                                ], className="mb-1"),
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    html.Span("▲", style={"color": "green"}), 
                                    " Projected peak"
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("▼", style={"color": "red"}), 
                                    " Projected trough"
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("|", style={"color": "rgba(0,0,0,0.3)"}), 
                                    " Cycle boundary (zero crossing)"
                                ], className="mb-1"),
                            ], width=6),
                        ]),
                    ]),
                ], className="mt-3 mb-3", style={"fontSize": "0.9rem"}),
            ]),
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader(html.H4("Cycle Metrics")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Dominant Cycles"),
                        html.P(", ".join([f"{cycle}" for cycle in result.detected_cycles])),
                    ], width=3),
                    dbc.Col([
                        html.H5("Cycle Powers"),
                        html.P(", ".join([f"{cycle}: {power:.2f}" for cycle, power in
                                         result.cycle_powers.items()])),
                    ], width=3),
                    dbc.Col([
                        html.H5("Correlations"),
                        html.P(", ".join([f"{cycle}: {corr:.2f}" for cycle, corr in
                                         cycle_correlations.items()]) if cycle_correlations else "Not calculated"),
                    ], width=3),
                    dbc.Col([
                        html.H5("Next Turn Predictions"),
                        html.Div([
                            html.P("Upcoming cycle turns:"),
                            html.Ul([
                                html.Li([
                                    f"Cycle {cycle}: ",
                                    ", ".join([
                                        f"{turn['type'].capitalize()} on {turn['date'].strftime('%Y-%m-%d')}"
                                        for turn in result.cycle_projections.get(cycle, [])[:2]  # Show first 2 turns
                                    ]) if cycle in result.cycle_projections and result.cycle_projections[cycle] else "No projections"
                                ]) for cycle in result.detected_cycles
                            ])
                        ]) if hasattr(result, 'cycle_projections') else html.P("No turn predictions available")
                    ], width=3),
                ]),
                
                # Add harmonic relationships section
                html.Hr(),
                html.H5("Harmonic Relationships"),
                
                # Calculate harmonic relationships between cycles
                html.Div([
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("Cycles"), 
                                html.Th("Ratio"), 
                                html.Th("Harmonic"), 
                                html.Th("Precision")
                            ])
                        ),
                        html.Tbody([
                            # Generate rows for each harmonic relationship
                            *[html.Tr([
                                html.Td(f"{cycle1}:{cycle2}"),
                                html.Td(f"{(cycle2/cycle1):.3f}"),
                                html.Td(_get_harmonic_name(cycle2/cycle1)),
                                html.Td(f"{_get_harmonic_precision(cycle2/cycle1):.1f}%")
                            ]) for i, cycle1 in enumerate(result.detected_cycles) 
                               for j, cycle2 in enumerate(result.detected_cycles) if i < j]
                        ])
                    ], className="table table-sm table-striped")
                ]) if len(result.detected_cycles) > 1 else html.P("Not enough cycles to analyze harmonics"),
            ]),
        ]),
    ])