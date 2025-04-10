"""Market scanner dashboard module"""
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import logging

# Import core components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.scan_result import ScanResult
from utils.logging_utils import get_component_logger

# Configure logging
logger = get_component_logger("web.scanner_dashboard")

# Track callback registration status to avoid duplicates
# (Using globals() dict instead of module-level variable for better state tracking)


def create_scanner_dashboard(results: List[ScanResult]):
    """
    Create a comprehensive scanner dashboard from multiple scan results.
    
    Args:
        results: List of ScanResult instances
        
    Returns:
        Dash Div component with interactive dashboard
    """
    # Check if there are any results
    if not results:
        return html.Div([
            html.H3("No scan results available"),
            html.P("Please run a scan to see results here.")
        ])
    
    # Filter successful results
    valid_results = [r for r in results if r.success]
    
    if not valid_results:
        return html.Div([
            html.H3("Scanner Dashboard"),
            html.P("No valid scan results available."),
        ])
    
    # Count different signal types
    buy_signals = sum(1 for r in valid_results if 'buy' in r.signal.get('signal', ''))
    sell_signals = sum(1 for r in valid_results if 'sell' in r.signal.get('signal', ''))
    strong_buy = sum(1 for r in valid_results if 'strong_buy' in r.signal.get('signal', ''))
    strong_sell = sum(1 for r in valid_results if 'strong_sell' in r.signal.get('signal', ''))
    high_confidence = sum(1 for r in valid_results if r.signal.get('confidence') == 'high')
    
    # Create market distribution chart
    fig_distribution = go.Figure()
    
    # Add bars for signal types
    signal_labels = ['Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell']
    signal_values = [
        strong_buy,
        buy_signals - strong_buy,
        len(valid_results) - buy_signals - sell_signals,
        sell_signals - strong_sell,
        strong_sell
    ]
    signal_colors = ['darkgreen', 'green', 'gray', 'red', 'darkred']
    
    fig_distribution.add_trace(
        go.Bar(
            x=signal_labels,
            y=signal_values,
            marker_color=signal_colors,
            name="Signal Distribution"
        )
    )
    
    fig_distribution.update_layout(
        title="Market Signal Distribution",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Create cycle distribution chart - count occurrences of each cycle length
    cycle_counts = {}
    for result in valid_results:
        for cycle in result.detected_cycles:
            cycle_counts[cycle] = cycle_counts.get(cycle, 0) + 1
    
    # Sort cycles by length
    sorted_cycles = sorted(cycle_counts.keys())
    cycle_values = [cycle_counts[c] for c in sorted_cycles]
    
    fig_cycles = go.Figure()
    fig_cycles.add_trace(
        go.Bar(
            x=sorted_cycles,
            y=cycle_values,
            marker_color='royalblue',
            name="Cycle Distribution"
        )
    )
    
    fig_cycles.update_layout(
        title="Dominant Cycle Distribution",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_title="Cycle Length",
        yaxis_title="Number of Symbols"
    )
    
    # Organize results into a sortable table
    sorted_results = sorted(valid_results, key=lambda r: abs(r.signal.get('strength', 0)), reverse=True)
    
    # Create the scanner dashboard
    return html.Div([
        html.H3("Market Scanner Dashboard"),
        
        # Summary metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Symbols", className="card-title"),
                        html.H3(f"{len(valid_results)}", className="card-text text-center")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Buy Signals", className="card-title"),
                        html.H3(
                            f"{buy_signals}", 
                            className="card-text text-center text-success"
                        )
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Sell Signals", className="card-title"),
                        html.H3(
                            f"{sell_signals}", 
                            className="card-text text-center text-danger"
                        )
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("High Confidence", className="card-title"),
                        html.H3(
                            f"{high_confidence}", 
                            className="card-text text-center text-warning"
                        )
                    ])
                ])
            ], width=3),
        ], className="mb-4"),
        
        # Distribution charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_distribution),
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=fig_cycles),
            ], width=6),
        ], className="mb-4"),
        
        # Filter controls
        dbc.Row([
            dbc.Col([
                html.Label("Filter By:"),
                dbc.ButtonGroup([
                    dbc.Button("All", color="primary", id="filter-all", n_clicks=0),
                    dbc.Button("Buy Signals", color="success", id="filter-buy", n_clicks=0),
                    dbc.Button("Sell Signals", color="danger", id="filter-sell", n_clicks=0),
                    dbc.Button("High Confidence", color="warning", id="filter-high", n_clicks=0),
                ]),
            ], width=12),
        ], className="mb-3"),
        
        # Results table
        html.Div([
            html.H4(f"Scan Results ({len(valid_results)} symbols)"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Price"),
                    html.Th("Signal"),
                    html.Th("Strength"),
                    html.Th("Confidence"),
                    html.Th("Risk/Reward"),
                    html.Th("Cycles"),
                    html.Th("Actions"),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(result.symbol),
                        html.Td(
                            html.Span(f"{result.price:.2f}", className="fw-bold text-primary")
                        ),
                        html.Td(
                            dbc.Badge(
                                result.signal['signal'].replace("_", " ").upper(),
                                color="success" if "buy" in result.signal['signal'] else (
                                    "danger" if "sell" in result.signal['signal'] else "secondary"
                                ),
                                className="p-2"
                            )
                        ),
                        html.Td(
                            html.Div([
                                html.Span(f"{result.signal['strength']:.2f}"),
                                html.Div(
                                    className="progress mt-1",
                                    style={"height": "5px"},
                                    children=[
                                        html.Div(
                                            className="progress-bar bg-success" if result.signal['strength'] > 0 else "progress-bar bg-danger",
                                            style={"width": f"{min(abs(result.signal['strength']*100), 100)}%"}
                                        )
                                    ]
                                )
                            ])
                        ),
                        html.Td(
                            dbc.Badge(
                                result.signal['confidence'].upper(),
                                color={
                                    'high': 'success',
                                    'medium': 'warning',
                                    'low': 'danger'
                                }.get(result.signal['confidence'], 'secondary')
                            )
                        ),
                        html.Td(f"{result.position_guidance.get('risk_reward_ratio', 0):.2f}"),
                        html.Td(", ".join(map(str, result.detected_cycles))),
                        html.Td(
                            html.Div([
                                dbc.Button(
                                    "View", 
                                    color="primary", 
                                    size="sm", 
                                    className="me-1",
                                    id={"type": "scan-view-btn", "index": result.symbol}
                                ),
                                dbc.Button(
                                    "Perf", 
                                    color="success", 
                                    size="sm",
                                    className="me-1",
                                    id={"type": "scan-perf-btn", "index": result.symbol}
                                ),
                                # Backtest button removed
                            ], className="d-flex")
                        ),
                    ])
                    for result in sorted_results
                ]),
            ], bordered=True, striped=True, hover=True, responsive=True),
        ]),
        
        # Hidden div to store scan result data for callbacks
        html.Div(
            id="scan-result-store",
            style={"display": "none"},
            children=json.dumps({result.symbol: result.__dict__ for result in valid_results}, default=str)
        ),
    ])


def register_scanner_callbacks(app):
    """
    Register callbacks for scanner dashboard.
    
    Args:
        app: Dash application instance
    """
    # Backtest callback removed