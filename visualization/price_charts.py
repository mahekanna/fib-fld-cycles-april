import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, List, Optional, Any

# Configure matplotlib to use non-interactive Agg backend to avoid thread issues
import matplotlib
matplotlib.use('Agg')  # This must be before any other matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


def generate_plot_image(data: pd.DataFrame, 
                       symbol: str, 
                       cycles: List[int],
                       cycle_states: List[Dict],
                       signal: Dict,
                       lookback: int = 250) -> str:
    """
    Generate a plot image for a scan result.
    
    Args:
        data: DataFrame with price and cycle data
        symbol: Symbol name
        cycles: List of cycle lengths
        cycle_states: List of cycle state dictionaries
        signal: Signal dictionary
        lookback: Number of bars to look back
        
    Returns:
        Base64-encoded string of the plot image
    """
    # Use only recent data based on lookback
    if len(data) > lookback:
        plot_data = data.iloc[-lookback:]
    else:
        plot_data = data
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.3)
    
    # Format the dates nicely
    date_formatter = mdates.DateFormatter('%Y-%m-%d')
    
    # Plot price chart
    ax_price = axes[0]
    ax_price.plot(plot_data.index, plot_data['close'], label='Price', color='#1f77b4')
    
    # Plot FLD lines for each cycle
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, cycle_length in enumerate(cycles[:5]):  # Limit to 5 cycles for clarity
        fld_col = f'fld_{cycle_length}'
        if fld_col in plot_data.columns:
            color = colors[i % len(colors)]
            ax_price.plot(plot_data.index, plot_data[fld_col], 
                         label=f'FLD {cycle_length}', color=color, linestyle='--')
    
    # Add signal markers if available
    if signal and 'signal' in signal:
        last_idx = plot_data.index[-1]
        last_price = plot_data['close'].iloc[-1]
        
        if 'buy' in signal['signal']:
            ax_price.scatter([last_idx], [last_price], color='green', s=100, marker='^')
        elif 'sell' in signal['signal']:
            ax_price.scatter([last_idx], [last_price], color='red', s=100, marker='v')
    
    # Add cycle wave visualizations if available
    for i, cycle_length in enumerate(cycles[:3]):  # Show waves for top 3 cycles
        wave_col = f'cycle_wave_{cycle_length}'
        if wave_col in plot_data.columns:
            color = colors[i % len(colors)]
            ax_price.plot(plot_data.index, plot_data[wave_col], 
                         label=f'Cycle {cycle_length}', color=color, alpha=0.5)
    
    # Format price chart
    ax_price.set_title(f"{symbol} - Fibonacci Cycle Analysis", fontsize=14)
    ax_price.set_ylabel("Price", fontsize=12)
    ax_price.xaxis.set_major_formatter(date_formatter)
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc='upper left')
    
    # Plot cycle powers in the bottom panel
    ax_power = axes[1]
    
    # Create a bar chart of cycle powers
    cycle_powers = []
    for cycle in cycles:
        for state in cycle_states:
            if state['cycle_length'] == cycle:
                if 'price_to_fld_ratio' in state:
                    power = abs(state['price_to_fld_ratio'] - 1) * 100
                    cycle_powers.append(power)
                else:
                    cycle_powers.append(0)
                break
        else:
            cycle_powers.append(0)
    
    # Plot cycle powers
    bars = ax_power.bar(range(len(cycles)), cycle_powers, color=colors[:len(cycles)])
    ax_power.set_xticks(range(len(cycles)))
    ax_power.set_xticklabels([str(c) for c in cycles])
    ax_power.set_xlabel("Cycle Length (bars)", fontsize=12)
    ax_power.set_ylabel("Cycle Power (%)", fontsize=12)
    ax_power.set_title("Cycle Power Analysis", fontsize=14)
    ax_power.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, power in zip(bars, cycle_powers):
        height = bar.get_height()
        ax_power.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{power:.1f}%', ha='center', va='bottom', rotation=0)
    
    # Add signal information as a text box
    signal_text = (
        f"Signal: {signal.get('signal', 'Unknown').upper()}\n"
        f"Strength: {signal.get('strength', 0):.2f}\n"
        f"Confidence: {signal.get('confidence', 'Unknown').upper()}\n"
        f"Alignment: {signal.get('alignment', 0):.2f}"
    )
    
    ax_price.text(0.02, 0.05, signal_text, transform=ax_price.transAxes,
                fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str