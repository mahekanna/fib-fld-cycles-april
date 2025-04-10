import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple, Union, Any
import io
from datetime import datetime, timedelta
import base64

# For interactive web visualizations
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CycleChartGenerator:
    """
    Generate charts for cycle analysis and visualization.
    """
    
    def __init__(self, theme: str = 'dark'):
        """
        Initialize the chart generator.
        
        Args:
            theme: Visual theme ('dark' or 'light')
        """
        self.theme = theme
        
        # Set up matplotlib style
        if theme == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'background': '#121212',
                'text': '#FFFFFF',
                'grid': '#333333',
                'price': '#FFFFFF',
                'buy': '#00FF7F',
                'sell': '#FF4500',
                'neutral': '#888888',
                'cycle_colors': ['#1E90FF', '#FF1493', '#32CD32', '#FFD700', '#9370DB']
            }
        else:
            plt.style.use('default')
            self.colors = {
                'background': '#FFFFFF',
                'text': '#000000',
                'grid': '#CCCCCC',
                'price': '#000000',
                'buy': '#008000',
                'sell': '#FF0000',
                'neutral': '#888888',
                'cycle_colors': ['#1E90FF', '#FF1493', '#32CD32', '#FFD700', '#9370DB']
            }
    
    def generate_cycle_chart(self, 
                            data: pd.DataFrame, 
                            symbol: str,
                            cycles: List[int],
                            cycle_states: List[Dict],
                            signal: Dict,
                            lookback: int = 250) -> np.ndarray:
        """
        Generate a chart showing price with cycle analysis.
        
        Args:
            data: DataFrame with price and cycle data
            symbol: Symbol name
            cycles: List of cycle lengths
            cycle_states: List of cycle state dictionaries
            signal: Signal dictionary
            lookback: Number of bars to show
            
        Returns:
            Numpy array containing the rendered image
        """
        # Create figure and axes
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Get subset of data to display
        display_data = data.iloc[-lookback:].copy()
        
        # Create main price chart
        ax1 = plt.subplot(gs[0])
        self._plot_price_with_cycles(ax1, display_data, symbol, cycles, cycle_states)
        
        # Create signal strength chart
        ax2 = plt.subplot(gs[1], sharex=ax1)
        self._plot_signal_strength(ax2, display_data, cycles, signal)
        
        # Add title
        direction = 'Bullish' if 'buy' in signal['signal'] else ('Bearish' if 'sell' in signal['signal'] else 'Neutral')
        title_color = self.colors['buy'] if 'buy' in signal['signal'] else (
            self.colors['sell'] if 'sell' in signal['signal'] else self.colors['neutral']
        )
        
        confidence = signal.get('confidence', '').upper()
        fig.suptitle(
            f"{symbol} - {direction} ({confidence}) - Strength: {signal.get('strength', 0):.2f}",
            fontsize=16, 
            color=title_color
        )
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.05)
        
        # Render to array
        fig_data = io.BytesIO()
        plt.savefig(fig_data, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        fig_data.seek(0)
        
        # Convert to array
        import matplotlib.image as mpimg
        img_array = mpimg.imread(fig_data, format='png')
        
        return img_array
    
    def generate_cycle_comparison(self,
                                 data: pd.DataFrame,
                                 symbol: str,
                                 cycles: List[int]) -> np.ndarray:
        """
        Generate a comparison chart of different cycle lengths.
        
        Args:
            data: DataFrame with price data
            symbol: Symbol name
            cycles: List of cycle lengths
            
        Returns:
            Numpy array containing the rendered image
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get recent data for normalization
        recent_data = data.iloc[-100:].copy()
        
        # Normalize price to better see cycle structure
        close_price = recent_data['close']
        norm_price = (close_price - close_price.mean()) / close_price.std()
        
        # Base time axis
        x = np.arange(len(norm_price))
        
        # Plot normalized price
        ax.plot(x, norm_price.values, color=self.colors['price'], alpha=0.7, label='Normalized Price')
        
        # Plot idealized cycles with proper amplitudes and phases
        for i, cycle_length in enumerate(cycles):
            color = self.colors['cycle_colors'][i % len(self.colors['cycle_colors'])]
            
            # Generate a synthetic cycle with appropriate length
            x_cycle = np.linspace(0, 2 * np.pi * (len(norm_price) / cycle_length), len(norm_price))
            
            # Find the best phase by correlating with price
            best_corr = -1
            best_phase = 0
            for phase in np.linspace(0, 2*np.pi, 20):
                wave = np.sin(x_cycle + phase)
                corr = np.abs(np.corrcoef(norm_price.values, wave)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_phase = phase
            
            # Generate the wave with optimal phase
            wave = np.sin(x_cycle + best_phase)
            
            # Plot
            ax.plot(x, wave, color=color, label=f'Cycle {cycle_length}', linewidth=2)
            
            # Mark cycle lengths
            for j in range(0, len(norm_price), cycle_length):
                if j < len(norm_price):
                    ax.axvline(x=j, color=color, linestyle=':', alpha=0.3)
        
        # Add a horizontal line at zero
        ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
        
        # Format plot
        ax.set_title(f"{symbol} - Normalized Cycle Comparison")
        ax.set_xlabel('Days')
        ax.set_ylabel('Normalized Value')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Render to array
        fig_data = io.BytesIO()
        plt.savefig(fig_data, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        fig_data.seek(0)
        
        # Convert to array
        import matplotlib.image as mpimg
        img_array = mpimg.imread(fig_data, format='png')
        
        return img_array
    
    def generate_harmonics_chart(self,
                                data: pd.DataFrame,
                                symbol: str,
                                harmonic_relationships: Dict) -> np.ndarray:
        """
        Generate a chart showing harmonic relationships between cycles.
        
        Args:
            data: DataFrame with price data
            symbol: Symbol name
            harmonic_relationships: Dictionary of harmonic relationships
            
        Returns:
            Numpy array containing the rendered image
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up cycle data
        cycles = set()
        for pair in harmonic_relationships.keys():
            c1, c2 = pair.split(':')
            cycles.add(int(c1))
            cycles.add(int(c2))
        
        cycles = sorted(list(cycles))
        
        # Create a grid
        x = np.arange(len(cycles))
        y = np.arange(len(cycles))
        X, Y = np.meshgrid(x, y)
        
        # Create a size and color matrix
        sizes = np.zeros((len(cycles), len(cycles)))
        colors = np.zeros((len(cycles), len(cycles)))
        annotations = []
        
        # Populate matrices
        for pair, relation in harmonic_relationships.items():
            c1, c2 = pair.split(':')
            i = cycles.index(int(c1))
            j = cycles.index(int(c2))
            
            # Size based on precision
            precision = relation.get('precision', 0)
            sizes[i, j] = precision / 100 * 500 + 50  # Scale to reasonable marker size
            
            # Color based on harmonic type
            harmonic = relation.get('harmonic', 'None')
            if harmonic == 'Golden Ratio (1.618)':
                colors[i, j] = 1.0  # Golden ratio
            elif harmonic == 'Octave (2:1)':
                colors[i, j] = 0.8  # Octave
            elif harmonic == 'Perfect Fifth (3:2)':
                colors[i, j] = 0.6  # Perfect fifth
            elif harmonic == 'Square Root of 2':
                colors[i, j] = 0.4  # Square root of 2
            else:
                colors[i, j] = 0.2  # Other
            
            # Add annotation
            annotations.append({
                'x': i,
                'y': j,
                'text': f"{relation.get('ratio', 0):.2f}",
                'precision': precision
            })
        
        # Define colormap
        cmap = LinearSegmentedColormap.from_list(
            'harmonic_cmap', 
            [(0.0, 'gray'), (0.4, 'purple'), (0.6, 'blue'), (0.8, 'green'), (1.0, 'gold')]
        )
        
        # Create scatter plot
        scatter = ax.scatter(X, Y, s=sizes, c=colors, cmap=cmap, alpha=0.7)
        
        # Add annotations
        for anno in annotations:
            if anno['precision'] > 50:  # Only annotate strong relationships
                ax.text(anno['x'], anno['y'], anno['text'], 
                       ha='center', va='center', 
                       fontsize=8, color='white',
                       bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
        
        # Set axis labels and ticks
        ax.set_xticks(np.arange(len(cycles)))
        ax.set_yticks(np.arange(len(cycles)))
        ax.set_xticklabels([str(c) for c in cycles])
        ax.set_yticklabels([str(c) for c in cycles])
        ax.set_xlabel('Cycle Length')
        ax.set_ylabel('Cycle Length')
        
        # Set title
        ax.set_title(f"{symbol} - Harmonic Relationships Between Cycles")
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Harmonic Type')
        
        # Render to array
        fig_data = io.BytesIO()
        plt.savefig(fig_data, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        fig_data.seek(0)
        
        # Convert to array
        import matplotlib.image as mpimg
        img_array = mpimg.imread(fig_data, format='png')
        
        return img_array
    
    def _plot_price_with_cycles(self,
                              ax: plt.Axes,
                              data: pd.DataFrame,
                              symbol: str,
                              cycles: List[int],
                              cycle_states: List[Dict]) -> None:
        """
        Plot price with cycle analysis on a given axis.
        
        Args:
            ax: Matplotlib Axes to plot on
            data: DataFrame with price and cycle data
            symbol: Symbol name
            cycles: List of cycle lengths
            cycle_states: List of cycle state dictionaries
        """
        # Plot price
        if 'hlc3' in data.columns:
            price_series = data['hlc3']
        else:
            price_series = data['close']
        
        ax.plot(data.index, price_series, label='Price', color=self.colors['price'], linewidth=2)
        
        # Plot cycles and FLDs
        for i, cycle_length in enumerate(cycles):
            color = self.colors['cycle_colors'][i % len(self.colors['cycle_colors'])]
            fld_name = f'fld_{cycle_length}'
            wave_name = f'cycle_wave_{cycle_length}'
            
            # Plot FLD
            if fld_name in data.columns:
                ax.plot(data.index, data[fld_name], 
                       label=f'FLD {cycle_length}', 
                       color=color,
                       linestyle='--',
                       linewidth=1.5,
                       alpha=0.8)
            
            # Plot cycle wave
            if wave_name in data.columns:
                ax.plot(data.index, data[wave_name], 
                       label=f'Cycle {cycle_length}', 
                       color=color,
                       linestyle='-',
                       linewidth=1,
                       alpha=0.5)
                
            # Mark crossovers
            if fld_name in data.columns:
                # Find crossovers
                bullish_cross = (data['close'].shift(1) < data[fld_name].shift(1)) & (data['close'] > data[fld_name])
                bearish_cross = (data['close'].shift(1) > data[fld_name].shift(1)) & (data['close'] < data[fld_name])
                
                # Mark on chart
                cross_dates_bull = data.index[bullish_cross]
                cross_dates_bear = data.index[bearish_cross]
                
                if len(cross_dates_bull) > 0:
                    ax.scatter(cross_dates_bull, data.loc[cross_dates_bull, 'close'], 
                             marker='^', color=color, s=100, alpha=0.8,
                             label=f'Bullish Cross {cycle_length}')
                
                if len(cross_dates_bear) > 0:
                    ax.scatter(cross_dates_bear, data.loc[cross_dates_bear, 'close'], 
                             marker='v', color=color, s=100, alpha=0.8,
                             label=f'Bearish Cross {cycle_length}')
        
        # Format the plot
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize='small')
        ax.set_title(f"{symbol} Price with Detected Cycles")
        
        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_signal_strength(self,
                            ax: plt.Axes,
                            data: pd.DataFrame,
                            cycles: List[int],
                            signal: Dict) -> None:
        """
        Plot signal strength on a given axis.
        
        Args:
            ax: Matplotlib Axes to plot on
            data: DataFrame with signal data
            cycles: List of cycle lengths
            signal: Signal dictionary
        """
        # Plot individual cycle signals
        for i, cycle_length in enumerate(cycles):
            color = self.colors['cycle_colors'][i % len(self.colors['cycle_colors'])]
            signal_name = f'signal_{cycle_length}'
            
            if signal_name in data.columns:
                # Plot signals as a step function (shifted up for visualization)
                shift = i * 2  # Shift each signal up to avoid overlap
                ax.step(data.index, data[signal_name] + shift, 
                      where='post', color=color, linewidth=1.5,
                      label=f'Cycle {cycle_length} Signal')
                
                # Add horizontal lines for reference
                ax.axhline(y=shift, color=color, linestyle=':', alpha=0.3)
                ax.axhline(y=shift+1, color=color, linestyle=':', alpha=0.3)
                ax.axhline(y=shift-1, color=color, linestyle=':', alpha=0.3)
        
        # Plot composite signal if available
        if 'composite_signal' in data.columns:
            ax.step(data.index, data['composite_signal'], 
                  where='post', color='white', linewidth=2, 
                  label='Composite Signal')
        
        # Add current signal
        if signal:
            strength = signal.get('strength', 0)
            alignment = signal.get('alignment', 0)
            
            # Draw current signal strength
            ax.axhline(y=strength, color='red', linestyle='-', linewidth=2, alpha=0.7,
                     label=f'Current Strength: {strength:.2f}')
            
            # Mark alignment
            ax.text(data.index[-1], strength, 
                  f"Strength: {strength:.2f}\nAlignment: {alignment:.2f}", 
                  ha='right', va='bottom',
                  bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
        
        # Format the plot
        ax.set_ylabel('Signal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize='small')
        
        # Hide x-labels (shared with main plot)
        plt.setp(ax.get_xticklabels(), visible=False)


class InteractiveCycleChartGenerator:
    """
    Generate interactive charts for web display using Plotly.
    """
    
    def __init__(self, theme: str = 'dark'):
        """
        Initialize the interactive chart generator.
        
        Args:
            theme: Visual theme ('dark' or 'light')
        """
        self.theme = theme
        
        # Set up colors
        if theme == 'dark':
            self.colors = {
                'background': '#121212',
                'paper_bg': '#1E1E1E',
                'text': '#FFFFFF',
                'grid': '#333333',
                'price': '#FFFFFF',
                'buy': '#00FF7F',
                'sell': '#FF4500',
                'neutral': '#888888',
                'cycle_colors': ['#1E90FF', '#FF1493', '#32CD32', '#FFD700', '#9370DB']
            }
            self.plot_template = 'plotly_dark'
        else:
            self.colors = {
                'background': '#FFFFFF',
                'paper_bg': '#F5F5F5',
                'text': '#000000',
                'grid': '#CCCCCC',
                'price': '#000000',
                'buy': '#008000',
                'sell': '#FF0000',
                'neutral': '#888888',
                'cycle_colors': ['#1E90FF', '#FF1493', '#32CD32', '#FFD700', '#9370DB']
            }
            self.plot_template = 'plotly'
    
    def generate_interactive_chart(self, 
                                  data: pd.DataFrame, 
                                  symbol: str,
                                  cycles: List[int],
                                  cycle_states: List[Dict],
                                  signal: Dict,
                                  lookback: int = 250) -> go.Figure:
        """
        Generate an interactive chart showing price with cycle analysis.
        
        Args:
            data: DataFrame with price and cycle data
            symbol: Symbol name
            cycles: List of cycle lengths
            cycle_states: List of cycle state dictionaries
            signal: Signal dictionary
            lookback: Number of bars to show
            
        Returns:
            Plotly Figure object
        """
        # Get subset of data to display
        display_data = data.iloc[-lookback:].copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Price with FLDs", "Signal Strength")
        )
        
        # Add price chart
        self._add_price_chart(fig, display_data, symbol, cycles, 1, 1)
        
        # Add signal chart
        self._add_signal_chart(fig, display_data, cycles, signal, 2, 1)
        
        # Update layout
        direction = 'Bullish' if 'buy' in signal['signal'] else ('Bearish' if 'sell' in signal['signal'] else 'Neutral')
        title_color = self.colors['buy'] if 'buy' in signal['signal'] else (
            self.colors['sell'] if 'sell' in signal['signal'] else self.colors['neutral']
        )
        
        confidence = signal.get('confidence', '').upper()
        
        fig.update_layout(
            title=f"{symbol} - {direction} ({confidence}) - Strength: {signal.get('strength', 0):.2f}",
            template=self.plot_template,
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['paper_bg'],
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def generate_interactive_comparison(self,
                                       data: pd.DataFrame,
                                       symbol: str,
                                       cycles: List[int]) -> go.Figure:
        """
        Generate an interactive comparison chart of different cycle lengths.
        
        Args:
            data: DataFrame with price data
            symbol: Symbol name
            cycles: List of cycle lengths
            
        Returns:
            Plotly Figure object
        """
        # Create figure
        fig = go.Figure()
        
        # Get recent data for normalization
        recent_data = data.iloc[-100:].copy()
        
        # Normalize price to better see cycle structure
        close_price = recent_data['close']
        norm_price = (close_price - close_price.mean()) / close_price.std()
        
        # Add normalized price
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=norm_price.values,
            mode='lines',
            name='Normalized Price',
            line=dict(color=self.colors['price'], width=2, dash='solid'),
            opacity=0.7
        ))
        
        # Add cycles
        for i, cycle_length in enumerate(cycles):
            color = self.colors['cycle_colors'][i % len(self.colors['cycle_colors'])]
            
            # Generate a synthetic cycle with appropriate length
            x_cycle = np.linspace(0, 2 * np.pi * (len(norm_price) / cycle_length), len(norm_price))
            
            # Find the best phase by correlating with price
            best_corr = -1
            best_phase = 0
            for phase in np.linspace(0, 2*np.pi, 20):
                wave = np.sin(x_cycle + phase)
                corr = np.abs(np.corrcoef(norm_price.values, wave)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_phase = phase
            
            # Generate the wave with optimal phase
            wave = np.sin(x_cycle + best_phase)
            
            # Add to plot
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=wave,
                mode='lines',
                name=f'Cycle {cycle_length}',
                line=dict(color=color, width=2, dash='solid')
            ))
            
            # Add vertical lines for cycle boundaries
            for j in range(0, len(recent_data), cycle_length):
                if j < len(recent_data):
                    fig.add_vline(
                        x=recent_data.index[j],
                        line=dict(color=color, width=1, dash='dot'),
                        opacity=0.3
                    )
        
        # Add a horizontal line at zero
        fig.add_hline(
            y=0,
            line=dict(color='grey', width=1, dash='solid'),
            opacity=0.3
        )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Normalized Cycle Comparison",
            xaxis_title="Date",
            yaxis_title="Normalized Value",
            template=self.plot_template,
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['paper_bg'],
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def _add_price_chart(self,
                       fig: go.Figure,
                       data: pd.DataFrame,
                       symbol: str,
                       cycles: List[int],
                       row: int,
                       col: int) -> None:
        """
        Add price chart with cycles to a Plotly figure.
        
        Args:
            fig: Plotly Figure to add to
            data: DataFrame with price and cycle data
            symbol: Symbol name
            cycles: List of cycle lengths
            row: Subplot row
            col: Subplot column
        """
        # Add price line
        if 'hlc3' in data.columns:
            price_series = data['hlc3']
        else:
            price_series = data['close']
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=price_series,
                name='Price',
                line=dict(color=self.colors['price'], width=2)
            ),
            row=row,
            col=col
        )
        
        # Add cycles and FLDs
        for i, cycle_length in enumerate(cycles):
            color = self.colors['cycle_colors'][i % len(self.colors['cycle_colors'])]
            fld_name = f'fld_{cycle_length}'
            wave_name = f'cycle_wave_{cycle_length}'
            
            # Add FLD
            if fld_name in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[fld_name],
                        name=f'FLD {cycle_length}',
                        line=dict(color=color, width=1.5, dash='dash'),
                        opacity=0.8
                    ),
                    row=row,
                    col=col
                )
            
            # Add cycle wave
            if wave_name in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[wave_name],
                        name=f'Cycle {cycle_length}',
                        line=dict(color=color, width=1),
                        opacity=0.5
                    ),
                    row=row,
                    col=col
                )
                
            # Add crossovers
            if fld_name in data.columns:
                # Find crossovers
                bullish_cross = (data['close'].shift(1) < data[fld_name].shift(1)) & (data['close'] > data[fld_name])
                bearish_cross = (data['close'].shift(1) > data[fld_name].shift(1)) & (data['close'] < data[fld_name])
                
                # Add to chart
                cross_dates_bull = data.index[bullish_cross]
                cross_prices_bull = data.loc[bullish_cross, 'close']
                
                cross_dates_bear = data.index[bearish_cross]
                cross_prices_bear = data.loc[bearish_cross, 'close']
                
                if len(cross_dates_bull) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=cross_dates_bull,
                            y=cross_prices_bull,
                            name=f'Bullish Cross {cycle_length}',
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color=color,
                                opacity=0.8
                            )
                        ),
                        row=row,
                        col=col
                    )
                
                if len(cross_dates_bear) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=cross_dates_bear,
                            y=cross_prices_bear,
                            name=f'Bearish Cross {cycle_length}',
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color=color,
                                opacity=0.8
                            )
                        ),
                        row=row,
                        col=col
                    )
    
    def _add_signal_chart(self,
                        fig: go.Figure,
                        data: pd.DataFrame,
                        cycles: List[int],
                        signal: Dict,
                        row: int,
                        col: int) -> None:
        """
        Add signal strength chart to a Plotly figure.
        
        Args:
            fig: Plotly Figure to add to
            data: DataFrame with signal data
            cycles: List of cycle lengths
            signal: Signal dictionary
            row: Subplot row
            col: Subplot column
        """
        # Add individual cycle signals
        for i, cycle_length in enumerate(cycles):
            color = self.colors['cycle_colors'][i % len(self.colors['cycle_colors'])]
            signal_name = f'signal_{cycle_length}'
            
            if signal_name in data.columns:
                # Shift each signal up to avoid overlap
                shift = i * 2
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[signal_name] + shift,
                        name=f'Cycle {cycle_length} Signal',
                        line=dict(color=color, width=1.5, shape='hv'),
                        mode='lines'
                    ),
                    row=row,
                    col=col
                )
                
                # Add reference lines
                for level in [shift-1, shift, shift+1]:
                    fig.add_shape(
                        type="line",
                        x0=data.index[0],
                        y0=level,
                        x1=data.index[-1],
                        y1=level,
                        line=dict(color=color, width=1, dash="dot"),
                        opacity=0.3,
                        row=row,
                        col=col
                    )
        
        # Add composite signal if available
        if 'composite_signal' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['composite_signal'],
                    name='Composite Signal',
                    line=dict(color='white', width=2, shape='hv'),
                    mode='lines'
                ),
                row=row,
                col=col
            )
        
        # Add current signal
        if signal:
            strength = signal.get('strength', 0)
            
            # Add horizontal line for current strength
            fig.add_shape(
                type="line",
                x0=data.index[0],
                y0=strength,
                x1=data.index[-1],
                y1=strength,
                line=dict(color='red', width=2),
                opacity=0.7,
                row=row,
                col=col
            )
            
            # Add annotation
            fig.add_annotation(
                x=data.index[-1],
                y=strength,
                text=f"Strength: {strength:.2f}",
                showarrow=True,
                arrowhead=1,
                row=row,
                col=col
            )


class CycleVisualizationExporter:
    """
    Export cycle visualizations in various formats.
    """
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize the visualization exporter.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_matplotlib_chart(self, 
                            img_array: np.ndarray,
                            filename: str) -> str:
        """
        Save a Matplotlib chart image.
        
        Args:
            img_array: Numpy array containing the image
            filename: Base filename without extension
            
        Returns:
            Path to saved file
        """
        # Create full path
        filepath = os.path.join(self.output_dir, f"{filename}.png")
        
        # Save file
        plt.imsave(filepath, img_array)
        
        return filepath
    
    def save_plotly_chart(self, 
                         fig: go.Figure,
                         filename: str,
                         formats: List[str] = ['html', 'png', 'json']) -> Dict[str, str]:
        """
        Save a Plotly chart in multiple formats.
        
        Args:
            fig: Plotly Figure to save
            filename: Base filename without extension
            formats: List of formats to save ('html', 'png', 'json', 'svg')
            
        Returns:
            Dictionary mapping formats to file paths
        """
        paths = {}
        
        # Save in each requested format
        for fmt in formats:
            if fmt == 'html':
                filepath = os.path.join(self.output_dir, f"{filename}.html")
                fig.write_html(filepath, include_plotlyjs='cdn')
                paths['html'] = filepath
            
            elif fmt == 'png':
                filepath = os.path.join(self.output_dir, f"{filename}.png")
                fig.write_image(filepath, width=1200, height=800)
                paths['png'] = filepath
            
            elif fmt == 'svg':
                filepath = os.path.join(self.output_dir, f"{filename}.svg")
                fig.write_image(filepath, width=1200, height=800)
                paths['svg'] = filepath
            
            elif fmt == 'json':
                filepath = os.path.join(self.output_dir, f"{filename}.json")
                with open(filepath, 'w') as f:
                    f.write(fig.to_json())
                paths['json'] = filepath
        
        return paths
    
    def chart_to_base64(self, 
                       img_array: np.ndarray) -> str:
        """
        Convert a chart image to base64 for embedding.
        
        Args:
            img_array: Numpy array containing the image
            
        Returns:
            Base64 encoded string
        """
        # Convert to bytes
        img_bytes = io.BytesIO()
        plt.imsave(img_bytes, img_array, format='PNG')
        img_bytes.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        
        return img_base64
