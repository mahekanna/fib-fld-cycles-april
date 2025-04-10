"""Harmonic pattern visualization module for market analysis"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.scan_result import ScanResult


def create_harmonic_visualization(result: ScanResult):
    """
    Create an interactive visualization of harmonic patterns.
    
    Args:
        result: ScanResult object containing harmonic pattern data
        
    Returns:
        Dash component for harmonic pattern visualization
    """
    if not result.success:
        return html.Div([
            html.H3("Error in Harmonic Pattern Analysis"),
            html.P(result.error or "Unknown error occurred")
        ])
    
    # Check if harmonic patterns are present in harmonic_relationships
    if not hasattr(result, 'harmonic_relationships') or not result.harmonic_relationships:
        return html.Div([
            html.H3("No harmonic patterns detected"),
            html.P("No valid harmonic patterns were found in the current data.")
        ])
    
    # Get data from the result if available
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
            print(f"Error fetching data for harmonic visualization: {str(e)}")
    
    # Create visualization with actual price data if available
    if data is not None and not data.empty:
        fig = go.Figure()
        
        # Add price candlesticks
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
        )
    else:
        # Create a placeholder visualization
        fig = go.Figure()
    
    # Update layout
    fig.update_layout(
        height=700,
        title=f"{result.symbol} Harmonic Pattern Analysis",
        template="plotly_dark",
        showlegend=True
    )
    
    # Create the component
    return html.Div([
        dbc.Card([
            dbc.CardHeader(html.H3("Harmonic Pattern Visualization")),
            dbc.CardBody([
                dcc.Graph(figure=fig, id="harmonic-graph"),
            ]),
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader(html.H4("Detected Patterns")),
            dbc.CardBody([
                # Create pattern cards based on harmonic relationships
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            f"Harmonic Relationship: {pattern_name}"
                        ),
                        dbc.CardBody([
                            html.Div([
                                html.Strong("Ratio: "),
                                html.Span(f"{data['ratio']:.3f}")
                            ], className="mb-2"),
                            html.Div([
                                html.Strong("Harmonic: "),
                                html.Span(data['harmonic'] or "None")
                            ], className="mb-2"),
                            html.Div([
                                html.Strong("Precision: "),
                                html.Span(f"{data['precision']:.2f}%")
                            ]),
                        ])
                    ], className="mb-3")
                    for pattern_name, data in result.harmonic_relationships.items()
                ]),
            ]),
        ]),
    ])