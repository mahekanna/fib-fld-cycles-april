"""
Backtesting UI module for the Fibonacci Cycle Trading System.
This module provides a web interface for backtesting trading strategies.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the centralized logging system
from utils.logging_utils import get_component_logger

# Configure logging with component-specific logger
logger = get_component_logger("web.backtest_ui")

# Import project modules using relative imports
# These imports will be handled by main_dashboard.py setting the correct sys.path
# If run directly, this will cause an ImportError that will be caught in main_dashboard.py

# Dummy BacktestParameters class to use if real one is not available
class BacktestParameters:
    """Parameters for backtesting."""
    def __init__(self, 
                 symbol="", 
                 exchange="", 
                 interval="", 
                 start_date=None, 
                 end_date=None, 
                 initial_capital=100000.0,
                 position_size_pct=10.0,
                 min_strength=0.3,
                 take_profit_multiplier=2.0,
                 trailing_stop=False,
                 trailing_stop_pct=5.0,
                 strategy_type="fib_cycle"):
        self.symbol = symbol
        self.exchange = exchange
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.min_strength = min_strength
        self.take_profit_multiplier = take_profit_multiplier
        self.trailing_stop = trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.strategy_type = strategy_type


def create_backtest_ui(result=None, initial_params=None):
    """Create the backtesting UI component."""
    # Get symbol from result or initial params if available
    symbol = ""
    exchange = "NSE"
    interval = "daily"
    
    if result and hasattr(result, 'symbol'):
        symbol = result.symbol
        exchange = getattr(result, 'exchange', 'NSE')
        interval = getattr(result, 'interval', 'daily')
    elif initial_params:
        symbol = initial_params.get('symbol', '')
        exchange = initial_params.get('exchange', 'NSE')
        interval = initial_params.get('interval', 'daily')
    
    return html.Div([
        # Store for backtest results to prevent callback conflicts
        dcc.Store(id="backtest-results-store", storage_type="session"),
        
        dbc.Row([
            # Left side - Backtest Configuration
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Backtest Configuration"),
                    dbc.CardBody([
                        # Trading Pair Selection
                        html.Div([
                            html.Label("Symbol"),
                            dbc.Input(
                                id="backtest-symbol-input",
                                type="text",
                                placeholder="Enter symbol (e.g., NIFTY)",
                                value=symbol,
                            ),
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Label("Exchange"),
                            dcc.Dropdown(
                                id="backtest-exchange-dropdown",
                                options=[
                                    {"label": "NSE", "value": "NSE"},
                                    {"label": "BSE", "value": "BSE"},
                                    {"label": "NYSE", "value": "NYSE"},
                                    {"label": "NASDAQ", "value": "NASDAQ"},
                                ],
                                value=exchange,
                                clearable=False,
                                style={
                                    'color': 'black',
                                    'background-color': 'white',
                                },
                            ),
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Label("Timeframe"),
                            dcc.Dropdown(
                                id="backtest-timeframe-dropdown",
                                options=[
                                    {"label": "1 minute", "value": "1m"},
                                    {"label": "5 minutes", "value": "5m"},
                                    {"label": "15 minutes", "value": "15m"},
                                    {"label": "30 minutes", "value": "30m"},
                                    {"label": "1 hour", "value": "1h"},
                                    {"label": "4 hours", "value": "4h"},
                                    {"label": "Daily", "value": "daily"},
                                    {"label": "Weekly", "value": "weekly"},
                                    {"label": "Monthly", "value": "monthly"},
                                ],
                                value=interval,
                                clearable=False,
                                style={
                                    'color': 'black',
                                    'background-color': 'white',
                                },
                            ),
                        ], className="mb-3"),
                        
                        # Date Range
                        html.Div([
                            html.Label("Date Range"),
                            dbc.Row([
                                dbc.Col([
                                    dcc.DatePickerSingle(
                                        id="backtest-start-date",
                                        placeholder="Start Date",
                                        date=(datetime.now() - timedelta(days=365)).date(),
                                        min_date_allowed=(datetime.now() - timedelta(days=3*365)).date(),
                                        max_date_allowed=datetime.now().date(),
                                        display_format='YYYY-MM-DD',
                                    ),
                                ], width=6),
                                dbc.Col([
                                    dcc.DatePickerSingle(
                                        id="backtest-end-date",
                                        placeholder="End Date",
                                        date=datetime.now().date(),
                                        min_date_allowed=(datetime.now() - timedelta(days=3*365)).date(),
                                        max_date_allowed=datetime.now().date(),
                                        display_format='YYYY-MM-DD',
                                    ),
                                ], width=6),
                            ]),
                        ], className="mb-3"),
                        
                        # Strategy Selection
                        html.Div([
                            html.Label("Strategy"),
                            dcc.Dropdown(
                                id="backtest-strategy-dropdown",
                                options=[
                                    {"label": "Fibonacci Cycle", "value": "fib_cycle"},
                                    {"label": "Harmonic Patterns", "value": "harmonic"},
                                    {"label": "FLD Crossover", "value": "fld_crossover"},
                                    {"label": "Multi-Timeframe", "value": "multi_tf"},
                                ],
                                value="fib_cycle",
                                clearable=False,
                                style={
                                    'color': 'black',
                                    'background-color': 'white',
                                },
                            ),
                        ], className="mb-3"),
                        
                        # Capital and Risk Parameters
                        html.Div([
                            html.Label("Initial Capital"),
                            dbc.Input(
                                id="backtest-initial-capital",
                                type="number",
                                value=100000,
                                min=1000,
                                step=1000,
                            ),
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Label("Position Size (%)"),
                            dbc.Input(
                                id="backtest-position-size",
                                type="number",
                                value=10,
                                min=1,
                                max=100,
                                step=1,
                            ),
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Label("Signal Strength Threshold"),
                            dcc.Slider(
                                id="backtest-signal-threshold",
                                min=0.1,
                                max=0.9,
                                step=0.1,
                                value=0.3,
                                marks={i/10: str(i/10) for i in range(1, 10)},
                            ),
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Label("Take Profit Multiplier"),
                            dcc.Slider(
                                id="backtest-tp-multiplier",
                                min=1.0,
                                max=5.0,
                                step=0.5,
                                value=2.0,
                                marks={i: str(i) for i in range(1, 6)},
                            ),
                        ], className="mb-3"),
                        
                        html.Div([
                            dbc.Checkbox(
                                id="backtest-trailing-stop",
                                label="Use Trailing Stop",
                                value=False,
                            ),
                            dbc.Input(
                                id="backtest-trailing-stop-pct",
                                type="number",
                                value=5.0,
                                min=1.0,
                                max=20.0,
                                step=0.5,
                                disabled=True,
                                className="mt-2",
                                placeholder="Trailing Stop %",
                            ),
                        ], className="mb-3"),
                        
                        # Run Backtest Button
                        dbc.Button(
                            "Run Backtest",
                            id="run-backtest-button",
                            color="primary",
                            className="w-100",
                        ),
                    ]),
                ]),
            ], width=4),
            
            # Right side - Results
            dbc.Col([
                dbc.Spinner([
                    html.Div(id="backtest-results-container", children=[], style={"display": "none"}),
                ], color="primary", spinner_style={"width": "3rem", "height": "3rem"}),
            ], width=8),
        ]),
    ])


def create_backtest_results_ui(results: Dict) -> html.Div:
    """Create the UI for displaying backtest results."""
    if not results:
        return html.Div("No backtest results available.")
    
    # Check if we're using the placeholder results
    is_placeholder = results.get('placeholder', False)
    logger.info(f"Backtest results received: placeholder={is_placeholder}, symbol={results.get('symbol')}")
    logger.info(f"Result keys: {list(results.keys())}")
    
    if is_placeholder:
        # Create a direct placeholder UI that's immediately identifiable
        return html.Div([
            html.H3(f"Backtest Results - {results.get('symbol', '')} ({results.get('interval', '')})"),
            html.P(f"Period: {results.get('start_date', '')} to {results.get('end_date', '')}"),
            
            # Big, clear placeholder message
            dbc.Alert([
                html.H3("Backtesting Data Source Issue", className="alert-heading"),
                html.P([
                    "The backtesting engine has been installed, but there's an issue with data fetching. ",
                    "To use the full backtesting functionality, you'll need to install the data packages:"
                ]),
                html.Pre(
                    "pip install yfinance\n"
                    "pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git\n"
                    "pip install alpha_vantage",
                    style={"backgroundColor": "#f8f9fa", "padding": "10px"}
                ),
                html.P([
                    "You can also run the provided installer script: ",
                    html.Code("./install_data_dependencies.sh")
                ]),
                html.P([
                    "To test if the backtesting engine is working correctly, run: ",
                    html.Code("python test_backtest.py")
                ]),
            ], color="warning", style={"marginTop": "20px"}),
            
            # Debug information
            dbc.Card([
                dbc.CardHeader("Debug Information"),
                dbc.CardBody([
                    html.P("The following information can help diagnose backtesting issues:"),
                    html.Pre(
                        f"Symbol: {results.get('symbol')}\n"
                        f"Exchange: {results.get('exchange')}\n"
                        f"Interval: {results.get('interval')}\n"
                        f"Date Range: {results.get('start_date')} to {results.get('end_date')}\n"
                        f"Python Path: {sys.path}\n",
                        style={"backgroundColor": "#f8f9fa", "padding": "10px", "maxHeight": "200px", "overflow": "auto"}
                    ),
                ]),
            ], style={"marginTop": "20px"}),
        ])
    
    # Extract metrics and data for real results
    metrics = results.get('metrics', {})
    equity_curve = results.get('equity_curve', [])
    trades = results.get('trades', [])
    
    # Create equity curve figure
    fig_equity = go.Figure()
    
    if equity_curve:
        # Convert dates to datetime if they're strings
        dates = []
        equity_values = []
        for point in equity_curve:
            date = point.get('date')
            if isinstance(date, str):
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            dates.append(date)
            equity_values.append(point.get('equity'))
        
        # Plot equity curve
        fig_equity.add_trace(go.Scatter(
            x=dates,
            y=equity_values,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        # Add initial capital reference line
        fig_equity.add_shape(
            type="line",
            x0=dates[0],
            y0=results.get('initial_capital', 0),
            x1=dates[-1],
            y1=results.get('initial_capital', 0),
            line=dict(color="gray", width=2, dash="dash"),
        )
        
        # Add trade markers
        winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
        
        # Add winning trade markers
        if winning_trades:
            entry_dates = []
            exit_dates = []
            entry_equities = []
            exit_equities = []
            
            for trade in winning_trades:
                entry_date = trade.get('entry_date')
                exit_date = trade.get('exit_date')
                
                if isinstance(entry_date, str):
                    entry_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
                if isinstance(exit_date, str):
                    exit_date = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
                
                # Find nearest equity values
                entry_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - entry_date).total_seconds()))
                exit_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - exit_date).total_seconds()))
                
                entry_dates.append(entry_date)
                exit_dates.append(exit_date)
                entry_equities.append(equity_values[entry_idx])
                exit_equities.append(equity_values[exit_idx])
            
            # Add entry points
            fig_equity.add_trace(go.Scatter(
                x=entry_dates,
                y=entry_equities,
                mode='markers',
                name='Entry (Win)',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
            # Add exit points
            fig_equity.add_trace(go.Scatter(
                x=exit_dates,
                y=exit_equities,
                mode='markers',
                name='Exit (Win)',
                marker=dict(color='green', size=10, symbol='triangle-down')
            ))
        
        # Add losing trade markers (similar logic for losing trades)
        if losing_trades:
            entry_dates = []
            exit_dates = []
            entry_equities = []
            exit_equities = []
            
            for trade in losing_trades:
                entry_date = trade.get('entry_date')
                exit_date = trade.get('exit_date')
                
                if isinstance(entry_date, str):
                    entry_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
                if isinstance(exit_date, str):
                    exit_date = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
                
                # Find nearest equity values
                entry_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - entry_date).total_seconds()))
                exit_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - exit_date).total_seconds()))
                
                entry_dates.append(entry_date)
                exit_dates.append(exit_date)
                entry_equities.append(equity_values[entry_idx])
                exit_equities.append(equity_values[exit_idx])
            
            # Add entry points
            fig_equity.add_trace(go.Scatter(
                x=entry_dates,
                y=entry_equities,
                mode='markers',
                name='Entry (Loss)',
                marker=dict(color='red', size=10, symbol='triangle-up')
            ))
            
            # Add exit points
            fig_equity.add_trace(go.Scatter(
                x=exit_dates,
                y=exit_equities,
                mode='markers',
                name='Exit (Loss)',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
    
    # Update layout with better styling
    fig_equity.update_layout(
        title=f"Equity Curve - {results.get('symbol', '')} ({results.get('interval', '')})",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Create trade metrics table
    trade_metrics = [
        {'Metric': 'Total Return', 'Value': f"{metrics.get('profit_loss_pct', 0):.2f}%"},
        {'Metric': 'Total Trades', 'Value': metrics.get('total_trades', 0)},
        {'Metric': 'Win Rate', 'Value': f"{metrics.get('win_rate', 0) * 100:.2f}%"},
        {'Metric': 'Profit Factor', 'Value': f"{metrics.get('profit_factor', 0):.2f}"},
        {'Metric': 'Max Drawdown', 'Value': f"{metrics.get('max_drawdown_pct', 0):.2f}%"},
        {'Metric': 'Sharpe Ratio', 'Value': f"{metrics.get('sharpe_ratio', 0):.2f}"},
    ]
    
    # Return the complete UI
    return html.Div([
        html.H3(f"Backtest Results - {results.get('symbol', '')} ({results.get('interval', '')})"),
        html.P(f"Period: {results.get('start_date', '')} to {results.get('end_date', '')}"),
        
        # Metrics summary cards 
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Return", className="card-title"),
                        html.H3(f"{metrics.get('profit_loss_pct', 0):.2f}%", 
                               className="card-text text-primary" if metrics.get('profit_loss_pct', 0) > 0 else "card-text text-danger"),
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Win Rate", className="card-title"),
                        html.H3(f"{metrics.get('win_rate', 0) * 100:.2f}%", className="card-text"),
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Drawdown", className="card-title"),
                        html.H3(f"{metrics.get('max_drawdown_pct', 0):.2f}%", className="card-text text-danger"),
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Profit Factor", className="card-title"),
                        html.H3(f"{metrics.get('profit_factor', 0):.2f}", className="card-text"),
                    ])
                ])
            ], width=3),
        ], className="mb-4"),
        
        # Equity curve graph
        dbc.Card([
            dbc.CardHeader("Equity Curve"),
            dbc.CardBody([
                dcc.Graph(
                    figure=fig_equity,
                    config={"displayModeBar": True},
                    style={"height": "500px"}
                )
            ])
        ], className="mb-4"),
        
        # Metrics table and trade list in tabs
        dbc.Tabs([
            dbc.Tab([
                html.Div([
                    dash.dash_table.DataTable(
                        data=trade_metrics,
                        columns=[
                            {"name": "Metric", "id": "Metric"},
                            {"name": "Value", "id": "Value"},
                        ],
                        style_header={
                            'backgroundColor': '#333',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_cell={
                            'backgroundColor': '#222',
                            'color': 'white',
                            'padding': '10px',
                            'textAlign': 'left'
                        },
                        style_table={
                            'width': '100%',
                            'margin': '20px 0'
                        }
                    )
                ], style={'padding': '20px'})
            ], label="Metrics"),
            dbc.Tab([
                html.Div([
                    dash.dash_table.DataTable(
                        data=trades if trades else [],
                        columns=[
                            {"name": "Direction", "id": "direction"},
                            {"name": "Entry Date", "id": "entry_date"},
                            {"name": "Exit Date", "id": "exit_date"},
                            {"name": "Entry Price", "id": "entry_price", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Exit Price", "id": "exit_price", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "P&L", "id": "profit_loss", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "P&L %", "id": "profit_loss_pct", "type": "numeric", "format": {"specifier": ".2f%"}},
                            {"name": "Exit Reason", "id": "exit_reason"},
                        ],
                        style_header={
                            'backgroundColor': '#333',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_cell={
                            'backgroundColor': '#222',
                            'color': 'white',
                            'padding': '10px',
                            'textAlign': 'left'
                        },
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{profit_loss} > 0',
                                },
                                'backgroundColor': 'rgba(0, 128, 0, 0.2)',
                            },
                            {
                                'if': {
                                    'filter_query': '{profit_loss} <= 0',
                                },
                                'backgroundColor': 'rgba(255, 0, 0, 0.2)',
                            }
                        ],
                        page_size=10,
                        style_table={
                            'overflowX': 'auto',
                            'width': '100%',
                            'margin': '20px 0'
                        }
                    )
                ], style={'padding': '20px'})
            ], label="Trades"),
        ]),
    ])


def register_backtest_callbacks(app):
    """Register all backtesting callbacks."""
    
    @app.callback(
        Output("backtest-trailing-stop-pct", "disabled"),
        Input("backtest-trailing-stop", "value")
    )
    def toggle_trailing_stop_input(use_trailing_stop):
        """Toggle the trailing stop percentage input based on checkbox state."""
        return not use_trailing_stop
    
    # First callback: Save to store and trigger update
    @app.callback(
        Output("backtest-results-store", "data"),
        Input("run-backtest-button", "n_clicks"),
        State("backtest-symbol-input", "value"),
        State("backtest-exchange-dropdown", "value"),
        State("backtest-timeframe-dropdown", "value"),
        State("backtest-start-date", "date"),
        State("backtest-end-date", "date"),
        State("backtest-strategy-dropdown", "value"),
        State("backtest-initial-capital", "value"),
        State("backtest-position-size", "value"),
        State("backtest-signal-threshold", "value"),
        State("backtest-tp-multiplier", "value"),
        State("backtest-trailing-stop", "value"),
        State("backtest-trailing-stop-pct", "value"),
        prevent_initial_call=True
    )
    def prepare_backtest(n_clicks, symbol, exchange, timeframe, start_date, end_date, 
                       strategy, initial_capital, position_size, signal_threshold,
                       tp_multiplier, use_trailing_stop, trailing_stop_pct):
        """Prepare backtest parameters and run backtest, storing results."""
        if not n_clicks or not symbol:
            return None
            
        logger.info(f"Preparing backtest for {symbol} on {exchange}")
        
        # Run the actual backtest and store results
        try:
            # Import required components with proper error handling
            try:
                # Ensure we have the project root in path
                import os
                import sys
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                # Try to import required modules
                from backtesting.backtesting_framework import BacktestEngine, BacktestParameters
                from utils.config import load_config
                from data.data_management import DataFetcher
                from datetime import datetime
                
                # If we get here, imports worked!
                logger.info("Successfully imported backtesting modules")
                
                # First, load the configuration
                config_path = os.path.join(project_root, "config", "config.json")
                config = load_config(config_path)
                
                # Create a data fetcher and engine
                data_fetcher = DataFetcher(config)
                engine = BacktestEngine(config, data_fetcher=data_fetcher)
                
                # Parse dates - convert string dates to datetime objects
                try:
                    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
                    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
                except ValueError as date_err:
                    logger.warning(f"Date parsing error: {date_err}. Using default date range.")
                    from datetime import datetime, timedelta
                    end_date_obj = datetime.now()
                    start_date_obj = end_date_obj - timedelta(days=365)
                
                # Create backtesting parameters
                params = BacktestParameters(
                    symbol=symbol,
                    exchange=exchange,
                    interval=timeframe,
                    start_date=start_date_obj,
                    end_date=end_date_obj,
                    initial_capital=float(initial_capital) if initial_capital else 100000.0,
                    position_size_pct=float(position_size) if position_size else 10.0,
                    min_strength=float(signal_threshold) if signal_threshold else 0.3,
                    take_profit_multiplier=float(tp_multiplier) if tp_multiplier else 2.0,
                    trailing_stop=bool(use_trailing_stop),
                    trailing_stop_pct=float(trailing_stop_pct) if trailing_stop_pct else 5.0,
                    strategy_type=strategy
                )
                
                # Check if we have data for this symbol - try to get data to confirm
                try:
                    # Force real data for Indian markets unless specifically requested mock data
                    force_download = False
                    if exchange == "NSE":
                        # For Indian markets, retry harder to get real data
                        logger.info(f"Attempting to fetch real data for Indian market symbol: {symbol}")
                        # Try a few different ways to get actual data before falling back
                        try:
                            # Try to import tvdatafeed first
                            import importlib
                            try:
                                importlib.import_module('tvdatafeed')
                                logger.info("Found TradingView data feed - using for NSE data")
                                force_download = True  # Prefer fresh data for NSE
                            except ImportError:
                                logger.warning("tvdatafeed not available - falling back to cached data or yfinance")
                                # Try to alert the user about better data source
                                print(f"Warning: For optimal results with NSE, install tvdatafeed:")
                                print(f"pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git")
                        except Exception as e:
                            logger.warning(f"Error checking for tvdatafeed: {e}")
                    
                    # Try to get data
                    data = data_fetcher.get_data(
                        symbol=symbol,
                        exchange=exchange,
                        interval=timeframe,
                        lookback=365,
                        force_download=force_download
                    )
                    
                    if data is None or data.empty:
                        # Try to get cached data before generating mock data
                        logger.warning(f"No data found for {symbol}. Checking cache first...")
                        
                        # Check if data exists in cache
                        cache_path = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "cache", f"{exchange}_{symbol}_{timeframe}.csv"
                        )
                        
                        if os.path.exists(cache_path):
                            logger.info(f"Found cached data at {cache_path}. Trying to load it...")
                            try:
                                import pandas as pd
                                data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                                if not data.empty:
                                    logger.info(f"Successfully loaded cached data with {len(data)} rows")
                            except Exception as cache_err:
                                logger.error(f"Failed to load cached data: {str(cache_err)}")
                        
                        # If still no data, try to generate mock data as last resort
                        if data is None or data.empty:
                            logger.warning(f"No real data found for {symbol}. Attempting to generate mock data as last resort.")
                            try:
                                from data.mock_data_generator import generate_mock_price_data, save_mock_data_to_cache
                                mock_data = generate_mock_price_data(symbol, lookback=500, interval=timeframe)
                                save_mock_data_to_cache(symbol, exchange, timeframe, mock_data)
                                logger.info(f"Successfully generated mock data for {symbol}")
                                # Try fetching again after generating mock data
                                data = data_fetcher.get_data(
                                    symbol=symbol,
                                    exchange=exchange,
                                    interval=timeframe,
                                    lookback=365,
                                    force_download=False
                                )
                            except Exception as mock_err:
                                logger.error(f"Failed to generate mock data: {str(mock_err)}")
                                return {
                                    "error": f"No data available for {symbol} and mock data generation failed: {str(mock_err)}"
                                }
                    
                    # If we still don't have data, return an error
                    if data is None or data.empty:
                        return {
                            "error": f"No data available for {symbol} on {exchange} with {timeframe} interval."
                        }
                    
                    logger.info(f"Successfully retrieved data for {symbol}: {len(data)} rows")
                    
                except Exception as data_err:
                    logger.error(f"Error retrieving data: {str(data_err)}")
                    return {
                        "error": f"Failed to retrieve data: {str(data_err)}"
                    }
                
                # Run the backtest
                logger.info(f"Running backtest with parameters: {params.__dict__}")
                results = engine.run_backtest(params)
                
                # Process the results to ensure serializability
                serializable_results = engine._make_serializable(results)
                
                # Return the results
                logger.info(f"Backtest completed successfully. {len(results.get('trades', []))} trades.")
                return serializable_results
                
            except ImportError as ie:
                logger.warning(f"Import error in backtesting modules: {str(ie)}")
                # Fall back to placeholder data
                return {
                    "results_ready": True,
                    "symbol": symbol,
                    "exchange": exchange,
                    "interval": timeframe,
                    "start_date": start_date,
                    "end_date": end_date,
                    "placeholder": True,
                    "import_error": str(ie)
                }
                
        except Exception as e:
            logger.exception(f"Error preparing backtest: {str(e)}")
            return {"error": str(e)}
    
    # Second callback: Update UI from store - with suppress_callback_exceptions to avoid errors
    @app.callback(
        Output("backtest-results-container", "children"),
        Output("backtest-results-container", "style"),
        Input("backtest-results-store", "data"),
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def update_backtest_ui(stored_results):
        """Update the UI based on stored backtest results."""
        if not stored_results:
            return html.Div(), {"display": "none"}
        
        logger.info(f"Processing stored backtest results: {list(stored_results.keys())}")
        
        # Check for error in the stored results
        if "error" in stored_results:
            error_msg = stored_results["error"]
            logger.error(f"Backtest error from store: {error_msg}")
            return html.Div([
                html.H4("Backtest Error"),
                html.P(f"Error running backtest: {error_msg}"),
                html.Pre(error_msg, style={"whiteSpace": "pre-wrap", "backgroundColor": "#f8f9fa", "padding": "10px"})
            ]), {"display": "block"}
        
        # Process stored results
        try:
            # If placeholder flag is set, use placeholder UI
            if stored_results.get("placeholder", False):
                logger.info("Using placeholder backtest results UI")
                return create_backtest_results_ui(stored_results), {"display": "block"}
            
            # For real results, ensure we have the proper imports
            try:
                # Ensure we have the project root in path
                import os
                import sys
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                # Try to import required modules
                logger.info(f"Importing backtesting framework from: {project_root}")
                from backtesting.backtesting_framework import BacktestEngine
                logger.info("Successfully imported backtesting modules")
                
                # Real results should already be processed and in the store
                # We just need to pass them to the UI creation function
                return create_backtest_results_ui(stored_results), {"display": "block"}
                
            except ImportError as ie:
                logger.warning(f"Backtesting framework import error: {str(ie)}")
                # Even with import error, we can still display results if they're in the store
                return create_backtest_results_ui(stored_results), {"display": "block"}
            
        except Exception as e:
            error_msg = f"Error processing backtest results: {str(e)}"
            logger.exception(error_msg)
            return html.Div([
                html.H4("Backtest Results Error"),
                html.P(error_msg),
                html.Pre(str(e), style={"whiteSpace": "pre-wrap", "backgroundColor": "#f8f9fa", "padding": "10px"}),
                html.Div([
                    html.H5("Debug Information"),
                    html.Pre(
                        f"Stored Results Keys: {list(stored_results.keys())}",
                        style={"whiteSpace": "pre-wrap", "backgroundColor": "#f8f9fa", "padding": "10px"}
                    )
                ])
            ]), {"display": "block"}