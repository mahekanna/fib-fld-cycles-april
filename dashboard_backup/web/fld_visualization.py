"""FLD visualization module for market analysis"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, List, Optional

# Import core components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.fld_signal_generator import FLDCalculator
from models.scan_result import ScanResult
from data.data_management import DataFetcher


def create_fld_visualization(result: ScanResult):
    """
    Create an interactive visualization of FLD analysis.
    
    Args:
        result: ScanResult object containing FLD data
        
    Returns:
        Dash component for FLD visualization
    """
    if not result.success:
        return html.Div([
            html.H3("Error in FLD Analysis"),
            html.P(result.error or "Unknown error occurred")
        ])
    
    # Create visualization for FLD analysis
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price with FLD Lines", "FLD Crossover Signals")
    )
    
    # Get price data from the result
    data = getattr(result, 'data', None)
    
    # If there's no data in the result, fetch it using the result parameters
    if data is None:
        try:
            from data.data_management import DataFetcher
            from utils.config import load_config
            from core.fld_signal_generator import FLDCalculator
            
            config_path = "config/config.json"  # Default config path
            config = load_config(config_path)  # Load config with explicit path
            data_fetcher = DataFetcher(config)
            fld_calculator = FLDCalculator()
            
            # Fetch data using lookback from result or default to 1000
            data = data_fetcher.get_data(
                symbol=result.symbol,
                exchange=result.exchange,
                interval=result.interval,
                lookback=getattr(result, 'lookback', 1000)  # Use lookback from result or default to 1000
            )
            
            # Calculate FLDs for each detected cycle if data was fetched successfully
            if data is not None and not data.empty:
                price_series = data['close']
                for cycle_length in result.detected_cycles:
                    fld_name = f'fld_{cycle_length}'
                    data[fld_name] = fld_calculator.calculate_fld(price_series, cycle_length)
        except Exception as e:
            # Log error and continue with limited visualization
            print(f"Error fetching data for FLD visualization: {str(e)}")
    
    # If we have data, create a complete visualization
    if data is not None and not data.empty:
        # Main price plot with FLDs - use all available data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name="Price",
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # Add FLD lines for each cycle
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
        
        # Add crossover markers if available from cycle_states
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []
        
        # In a real implementation, extract crossover points from data
        # For now, use simplified approach based on cycle state
        for state in result.cycle_states:
            if state.get('days_since_crossover') is not None:
                # Find approx index position based on days_since_crossover
                days_back = state.get('days_since_crossover', 0)
                if days_back < len(data):
                    idx = -int(days_back) - 1 if days_back > 0 else -1
                    if state.get('is_bullish', False):
                        buy_x.append(data.index[idx])
                        buy_y.append(data['close'].iloc[idx])
                    else:
                        sell_x.append(data.index[idx])
                        sell_y.append(data['close'].iloc[idx])
        
        # Add buy/sell markers
        if buy_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                ),
                row=1, col=1
            )
        
        if sell_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                ),
                row=1, col=1
            )
        
        # Second subplot - FLD to price ratios over time for fastest cycle
        if result.detected_cycles and f'fld_{result.detected_cycles[0]}' in data.columns:
            cycle = result.detected_cycles[0]
            fld_col = f'fld_{cycle}'
            
            # Calculate price to FLD ratio
            data['price_to_fld'] = data['close'] / data[fld_col]
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['price_to_fld'].dropna(),
                    mode='lines',
                    name=f"Price/FLD-{cycle} Ratio"
                ),
                row=2, col=1
            )
            
            # Add reference line at 1.0 (price = FLD)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=[1] * len(data.index),
                    mode='lines',
                    name="Equilibrium",
                    line=dict(dash='dash', color='rgba(255,255,255,0.5)')
                ),
                row=2, col=1
            )
            
            # Update second subplot axis
            fig.update_yaxes(title_text="Price/FLD Ratio", row=2, col=1)
    
    # Update layout regardless of data availability
    fig.update_layout(
        height=700,
        title=f"{result.symbol} FLD Analysis",
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # Create the component
    return html.Div([
        dbc.Card([
            dbc.CardHeader(html.H3("FLD Visualization")),
            dbc.CardBody([
                dcc.Graph(figure=fig, id="fld-graph"),
            ]),
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader(html.H4("FLD Crossover Metrics")),
            dbc.CardBody([
                # FLD Crossover States Table
                html.H5("FLD Crossover States"),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Cycle"),
                        html.Th("State"),
                        html.Th("Days Since Crossover"),
                        html.Th("Price/FLD Ratio"),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(f"{state['cycle_length']}"),
                            html.Td(
                                html.Span("Bullish", className="text-success") 
                                if state['is_bullish'] 
                                else html.Span("Bearish", className="text-danger")
                            ),
                            html.Td(f"{state['days_since_crossover']}" if state['days_since_crossover'] is not None else "N/A"),
                            html.Td(f"{state['price_to_fld_ratio']:.4f}"),
                        ])
                        for state in result.cycle_states
                    ]),
                ], bordered=True, striped=True, hover=True, size="sm"),
            ]),
        ]),
    ])