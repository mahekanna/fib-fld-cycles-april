"""
Advanced Backtesting UI Module for the Fibonacci Cycle Trading System.
This module provides a comprehensive web interface for backtesting trading strategies with
detailed performance reports, trade logs, metrics, and interactive visualizations.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Dash components
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the centralized logging system
from utils.logging_utils import get_component_logger

# Configure logging with component-specific logger
logger = get_component_logger("web.advanced_backtest_ui")

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
# Try different import approaches to ensure we can get the backtesting classes
try:
    from backtesting.backtesting_framework import BacktestEngine, BacktestParameters, BacktestTrade
except ImportError:
    try:
        # Try absolute imports with project path
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from backtesting.backtesting_framework import BacktestEngine, BacktestParameters, BacktestTrade
    except ImportError as e:
        logger.error(f"Error importing backtesting modules: {e}")
        # Create fallback classes if imports fail
        from dataclasses import dataclass
        
        @dataclass
        class BacktestTrade:
            """Class to represent a completed backtest trade."""
            symbol: str
            direction: str
            entry_date: datetime
            entry_price: float
            exit_date: datetime
            exit_price: float
            quantity: float
            profit_loss: float
            profit_loss_pct: float
            exit_reason: str
            
        @dataclass
        class BacktestParameters:
            """Parameters for backtesting."""
            symbol: str
            exchange: str
            interval: str
            start_date: Optional[datetime] = None
            end_date: Optional[datetime] = None
            lookback: int = 365
            num_cycles: int = 3
            price_source: str = "close"
            initial_capital: float = 100000.0
            position_size_pct: float = 10.0
            max_open_positions: int = 5
            trailing_stop: bool = False
            trailing_stop_pct: float = 5.0
            take_profit_multiplier: float = 2.0
            rebalance_frequency: Optional[str] = None
            require_alignment: bool = True
            min_strength: float = 0.3
            pyramiding: int = 0
            strategy_type: str = "fib_cycle"
from utils.config import load_config
from data.data_management import DataFetcher
from models.scan_result import ScanResult
from core.scanner import FibCycleScanner

# Layout 
def create_advanced_backtest_ui(result=None, initial_params=None):
    """
    Create the advanced backtesting UI component.
    
    Args:
        result: Optional ScanResult instance to use for initial configuration
        initial_params: Optional dictionary of initial parameters
        
    Returns:
        Dash component representing the advanced backtesting UI
    """
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
    
    # Create the component
    return html.Div([
        # Stores for state management
        dcc.Store(id="advanced-backtest-results-store", storage_type="session"),
        dcc.Store(id="advanced-backtest-trades-store", storage_type="session"),
        dcc.Store(id="advanced-backtest-params-store", storage_type="session"),
        
        dbc.Row([
            # Configuration panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Backtest Configuration", className="card-title mb-0"),
                    ]),
                    dbc.CardBody([
                        # Symbol and Market Selection
                        html.Div([
                            html.H5("Market & Symbol", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Exchange"),
                                    dcc.Dropdown(
                                        id="advanced-backtest-exchange-dropdown",
                                        options=[
                                            {"label": "NSE (India)", "value": "NSE"},
                                            {"label": "BSE (India)", "value": "BSE"},
                                            {"label": "NYSE (US)", "value": "NYSE"},
                                            {"label": "NASDAQ (US)", "value": "NASDAQ"},
                                        ],
                                        value=exchange,
                                        clearable=False,
                                        style={'color': 'black', 'background-color': 'white'},
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Symbol"),
                                    dbc.Input(
                                        id="advanced-backtest-symbol-input",
                                        type="text",
                                        placeholder="E.g., NIFTY, RELIANCE",
                                        value=symbol,
                                    ),
                                ], width=6),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Timeframe"),
                                    dcc.Dropdown(
                                        id="advanced-backtest-timeframe-dropdown",
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
                                        style={'color': 'black', 'background-color': 'white'},
                                    ),
                                ], width=12),
                            ]),
                        ], className="mb-4"),
                        
                        # Date Range Selection
                        html.Div([
                            html.H5("Backtest Period", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Start Date"),
                                    dcc.DatePickerSingle(
                                        id="advanced-backtest-start-date",
                                        min_date_allowed=datetime(2015, 1, 1),
                                        max_date_allowed=datetime.now().date(),
                                        initial_visible_month=datetime.now().date() - timedelta(days=365),
                                        date=datetime.now().date() - timedelta(days=365),
                                        display_format='YYYY-MM-DD',
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("End Date"),
                                    dcc.DatePickerSingle(
                                        id="advanced-backtest-end-date",
                                        min_date_allowed=datetime(2015, 1, 1),
                                        max_date_allowed=datetime.now().date(),
                                        initial_visible_month=datetime.now().date(),
                                        date=datetime.now().date(),
                                        display_format='YYYY-MM-DD',
                                    ),
                                ], width=6),
                            ], className="mb-3"),
                        ], className="mb-4"),
                        
                        # Strategy Selection
                        html.Div([
                            html.H5("Strategy Configuration", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Trading Strategy"),
                                    dcc.Dropdown(
                                        id="advanced-backtest-strategy-dropdown",
                                        options=[
                                            {"label": "Fibonacci Cycle", "value": "fib_cycle"},
                                            {"label": "Harmonic Patterns", "value": "harmonic"},
                                            {"label": "FLD Crossover", "value": "fld_crossover"},
                                            {"label": "Multi-Timeframe", "value": "multi_tf"},
                                            {"label": "Enhanced Entry/Exit", "value": "enhanced"},
                                        ],
                                        value="fib_cycle",
                                        clearable=False,
                                        style={'color': 'black', 'background-color': 'white'},
                                    ),
                                ], width=12),
                            ], className="mb-3"),
                        ], className="mb-4"),
                        
                        # Capital & Position Sizing
                        html.Div([
                            html.H5("Capital & Position Sizing", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Initial Capital"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("₹"),
                                        dbc.Input(
                                            id="advanced-backtest-initial-capital",
                                            type="number",
                                            value=100000,
                                            min=1000,
                                            step=1000,
                                        ),
                                    ]),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Position Size (%)"),
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="advanced-backtest-position-size",
                                            type="number",
                                            value=10,
                                            min=1,
                                            max=100,
                                            step=1,
                                        ),
                                        dbc.InputGroupText("%"),
                                    ]),
                                ], width=6),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Max Open Positions"),
                                    dbc.Input(
                                        id="advanced-backtest-max-positions",
                                        type="number",
                                        value=5,
                                        min=1,
                                        max=20,
                                        step=1,
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Pyramiding"),
                                    dbc.Input(
                                        id="advanced-backtest-pyramiding",
                                        type="number",
                                        value=0,
                                        min=0,
                                        max=5,
                                        step=1,
                                    ),
                                ], width=6),
                            ]),
                        ], className="mb-4"),
                        
                        # Entry Conditions
                        html.Div([
                            html.H5("Entry Conditions", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Signal Strength Threshold"),
                                    dcc.Slider(
                                        id="advanced-backtest-signal-threshold",
                                        min=0.1,
                                        max=0.9,
                                        step=0.1,
                                        value=0.2,
                                        marks={i/10: str(i/10) for i in range(1, 10)},
                                    ),
                                ], width=12),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Checkbox(
                                        id="advanced-backtest-require-alignment",
                                        label="Require Cycle Alignment",
                                        value=True,
                                    ),
                                ], width=12),
                            ]),
                        ], className="mb-4"),
                        
                        # Risk Management
                        html.Div([
                            html.H5("Risk Management", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Take Profit Multiple (R:R)"),
                                    dcc.Slider(
                                        id="advanced-backtest-tp-multiplier",
                                        min=1.0,
                                        max=5.0,
                                        step=0.5,
                                        value=2.0,
                                        marks={i: str(i) for i in range(1, 6)},
                                    ),
                                ], width=12),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Checkbox(
                                        id="advanced-backtest-trailing-stop",
                                        label="Use Trailing Stop",
                                        value=False,
                                    ),
                                ], width=12),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Trailing Stop (%)"),
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="advanced-backtest-trailing-stop-pct",
                                            type="number",
                                            value=5.0,
                                            min=1.0,
                                            max=20.0,
                                            step=0.5,
                                            disabled=True,
                                        ),
                                        dbc.InputGroupText("%"),
                                    ]),
                                ], width=12),
                            ]),
                        ], className="mb-4"),
                        
                        # Run Button
                        dbc.Button(
                            "Run Advanced Backtest",
                            id="run-advanced-backtest-button",
                            color="success",
                            size="lg",
                            className="w-100 mt-3",
                        ),
                    ]),
                ]),
            ], width=3),
            
            # Results panel
            dbc.Col([
                # Container div for consistent callback target
                html.Div(id="advanced-backtest-results-container", className="h-100", children=[
                    # Spinner while loading
                    dbc.Spinner(
                        html.Div(id="advanced-backtest-results-content", className="h-100"),
                        color="primary",
                        type="border",
                        fullscreen=False,
                        spinnerClassName="spinner-lg"
                    ),
                ]),
            ], width=9),
        ]),
    ])


def create_results_placeholder():
    """Create a placeholder for backtest results."""
    return html.Div([
        html.H4("Backtest Results", className="mb-4"),
        html.Div([
            html.P("Configure your backtest parameters and click 'Run Advanced Backtest' to see results here."),
            html.Hr(),
            html.Div([
                html.Img(src="/assets/backtest_placeholder.png", style={"maxWidth": "100%"}),
            ], style={"textAlign": "center"}),
        ], className="text-center p-5")
    ])


def create_backtest_results_ui(results):
    """
    Create the UI for displaying backtest results.
    
    Args:
        results: Dictionary containing backtest results
        
    Returns:
        Dash component representing the backtest results UI
    """
    if not results:
        return create_results_placeholder()
    
    # Extract data from results
    symbol = results.get('symbol', '')
    exchange = results.get('exchange', '')
    interval = results.get('interval', '')
    start_date = results.get('start_date', '')
    end_date = results.get('end_date', '')
    initial_capital = results.get('initial_capital', 0)
    final_capital = results.get('final_capital', 0)
    metrics = results.get('metrics', {})
    trades = results.get('trades', [])
    equity_curve = results.get('equity_curve', [])
    
    # Check if error occurred
    if 'error' in results:
        return html.Div([
            html.H4("Backtest Error"),
            html.P("An error occurred during backtesting:"),
            dbc.Alert(results['error'], color="danger"),
            html.Div([
                dbc.Button(
                    "Back to Configuration",
                    color="primary",
                    id="backtest-back-to-config-btn",  # Use the same ID as the main back button
                    className="mt-3"
                )
            ], className="text-center")
        ])
    
    # Calculate performance metrics
    total_return = metrics.get('profit_loss_pct', 0)
    win_rate = metrics.get('win_rate', 0) * 100
    profit_factor = metrics.get('profit_factor', 0)
    max_drawdown = metrics.get('max_drawdown_pct', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    total_trades = metrics.get('total_trades', 0)
    
    # Create equity curve figure
    fig_equity = go.Figure()
    
    # Process equity curve data
    if equity_curve:
        # Convert dates to datetime objects if they're strings
        dates = []
        equity_values = []
        
        for point in equity_curve:
            date = point.get('date')
            if isinstance(date, str):
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            dates.append(date)
            equity_values.append(point.get('equity'))
        
        # Create the equity curve
        fig_equity.add_trace(go.Scatter(
            x=dates,
            y=equity_values,
            mode='lines',
            name='Equity',
            line=dict(color='#00b894', width=2)
        ))
        
        # Add reference line for initial capital
        fig_equity.add_shape(
            type="line",
            x0=dates[0],
            y0=initial_capital,
            x1=dates[-1],
            y1=initial_capital,
            line=dict(color="#636e72", width=1.5, dash="dash"),
        )
        
        # Add trace for high water mark
        hwm = [initial_capital]
        for i in range(1, len(equity_values)):
            hwm.append(max(hwm[-1], equity_values[i]))
        
        fig_equity.add_trace(go.Scatter(
            x=dates,
            y=hwm,
            mode='lines',
            name='High Water Mark',
            line=dict(color='#0984e3', width=1, dash='dot'),
            opacity=0.7
        ))
        
        # Add drawdown areas
        drawdowns = []
        for i in range(len(equity_values)):
            drawdowns.append((1 - equity_values[i]/hwm[i]) * 100)
        
        # Create a separate figure for drawdowns
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(
            x=dates,
            y=drawdowns,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.3)',
            line=dict(color='#e74c3c', width=1)
        ))
        
        fig_drawdown.update_layout(
            title="Drawdown (%)",
            template="plotly_dark",
            height=180,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(
                ticksuffix="%",
                autorange="reversed",  # Invert Y-axis
            ),
            showlegend=False
        )
        
        # Add trade markers
        if trades:
            # Separate winning and losing trades
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
            
            # Add markers for winning trades
            if winning_trades:
                entry_dates = []
                exit_dates = []
                
                for trade in winning_trades:
                    entry_date = trade.get('entry_date')
                    exit_date = trade.get('exit_date')
                    
                    if isinstance(entry_date, str):
                        entry_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
                    if isinstance(exit_date, str):
                        exit_date = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
                    
                    entry_dates.append(entry_date)
                    exit_dates.append(exit_date)
                
                # Find closest equity values for entry dates
                entry_values = []
                for date in entry_dates:
                    # Find closest date
                    idx = min(range(len(dates)), key=lambda i: abs((dates[i] - date).total_seconds()))
                    entry_values.append(equity_values[idx])
                
                # Find closest equity values for exit dates
                exit_values = []
                for date in exit_dates:
                    # Find closest date
                    idx = min(range(len(dates)), key=lambda i: abs((dates[i] - date).total_seconds()))
                    exit_values.append(equity_values[idx])
                
                # Add entry markers
                fig_equity.add_trace(go.Scatter(
                    x=entry_dates,
                    y=entry_values,
                    mode='markers',
                    name='Buy',
                    marker=dict(color='#00b894', size=8, symbol='triangle-up')
                ))
                
                # Add exit markers
                fig_equity.add_trace(go.Scatter(
                    x=exit_dates,
                    y=exit_values,
                    mode='markers',
                    name='Sell (Profit)',
                    marker=dict(color='#00b894', size=8, symbol='triangle-down')
                ))
            
            # Add markers for losing trades
            if losing_trades:
                entry_dates = []
                exit_dates = []
                
                for trade in losing_trades:
                    entry_date = trade.get('entry_date')
                    exit_date = trade.get('exit_date')
                    
                    if isinstance(entry_date, str):
                        entry_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
                    if isinstance(exit_date, str):
                        exit_date = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
                    
                    entry_dates.append(entry_date)
                    exit_dates.append(exit_date)
                
                # Find closest equity values for entry dates
                entry_values = []
                for date in entry_dates:
                    # Find closest date
                    idx = min(range(len(dates)), key=lambda i: abs((dates[i] - date).total_seconds()))
                    entry_values.append(equity_values[idx])
                
                # Find closest equity values for exit dates
                exit_values = []
                for date in exit_dates:
                    # Find closest date
                    idx = min(range(len(dates)), key=lambda i: abs((dates[i] - date).total_seconds()))
                    exit_values.append(equity_values[idx])
                
                # Add entry markers
                fig_equity.add_trace(go.Scatter(
                    x=entry_dates,
                    y=entry_values,
                    mode='markers',
                    name='Buy',
                    marker=dict(color='#e74c3c', size=8, symbol='triangle-up')
                ))
                
                # Add exit markers
                fig_equity.add_trace(go.Scatter(
                    x=exit_dates,
                    y=exit_values,
                    mode='markers',
                    name='Sell (Loss)',
                    marker=dict(color='#e74c3c', size=8, symbol='triangle-down')
                ))
    
    # Update equity curve layout
    fig_equity.update_layout(
        title=f"Equity Curve: {symbol}",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    # Create monthly returns heatmap with proper error handling
    try:
        monthly_returns = calculate_monthly_returns(trades, equity_curve, initial_capital)
        if monthly_returns:
            fig_monthly = create_monthly_returns_heatmap(monthly_returns)
        else:
            # Create an empty figure with a message if no monthly returns could be calculated
            fig_monthly = go.Figure()
            fig_monthly.add_annotation(
                text="No monthly return data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig_monthly.update_layout(
                template="plotly_dark",
                height=300
            )
    except Exception as e:
        logger.error(f"Error creating monthly returns heatmap: {e}")
        # Create a fallback empty figure
        fig_monthly = go.Figure()
        fig_monthly.add_annotation(
            text=f"Error generating monthly returns chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig_monthly.update_layout(
            template="plotly_dark",
            height=300
        )
    
    # Create drawdown distribution
    fig_drawdown_dist = create_drawdown_distribution(equity_curve, initial_capital)
    
    # Create performance metrics boxplots
    fig_metrics_box = create_metrics_boxplots(trades)
    
    # Process the trades for the trades table
    processed_trades = []
    if trades:
        for i, trade in enumerate(trades):
            processed_trades.append({
                'id': i + 1,
                'direction': trade.get('direction', '').upper(),
                'entry_date': format_date(trade.get('entry_date')),
                'exit_date': format_date(trade.get('exit_date')),
                'entry_price': round(trade.get('entry_price', 0), 2),
                'exit_price': round(trade.get('exit_price', 0), 2),
                'quantity': round(trade.get('quantity', 0), 2),
                'profit_loss': round(trade.get('profit_loss', 0), 2),
                'profit_loss_pct': round(trade.get('profit_loss_pct', 0), 2),  # Numeric value for sorting
                'profit_loss_pct_display': f"{round(trade.get('profit_loss_pct', 0), 2)}%",  # String with % for display
                'exit_reason': trade.get('exit_reason', '').replace('_', ' ').title()
            })
    
    # Create summary metrics
    return html.Div([
        # Header with key metrics
        dbc.Card([
            dbc.CardHeader([
                html.H4(f"Backtest Results: {symbol} ({interval})"),
                html.P(f"Period: {format_date(start_date)} to {format_date(end_date)}")
            ]),
            dbc.CardBody([
                dbc.Row([
                    # Total Return
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Total Return", className="card-title text-center"),
                                html.H3(
                                    f"{total_return:.2f}%", 
                                    className=f"card-text text-center {'text-success' if total_return > 0 else 'text-danger'}"
                                ),
                            ])
                        ]),
                    ], width=2),
                    
                    # CAGR
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("CAGR", className="card-title text-center"),
                                html.H3(
                                    f"{calculate_cagr(initial_capital, final_capital, results.get('duration', 365)):.2f}%", 
                                    className="card-text text-center"
                                ),
                            ])
                        ]),
                    ], width=2),
                    
                    # Win Rate
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Win Rate", className="card-title text-center"),
                                html.H3(
                                    f"{win_rate:.2f}%", 
                                    className="card-text text-center"
                                ),
                            ])
                        ]),
                    ], width=2),
                    
                    # Profit Factor
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Profit Factor", className="card-title text-center"),
                                html.H3(
                                    f"{profit_factor:.2f}", 
                                    className="card-text text-center"
                                ),
                            ])
                        ]),
                    ], width=2),
                    
                    # Max Drawdown
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Max Drawdown", className="card-title text-center"),
                                html.H3(
                                    f"{max_drawdown:.2f}%", 
                                    className="card-text text-danger text-center"
                                ),
                            ])
                        ]),
                    ], width=2),
                    
                    # Sharpe Ratio
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Sharpe Ratio", className="card-title text-center"),
                                html.H3(
                                    f"{sharpe_ratio:.2f}", 
                                    className="card-text text-center"
                                ),
                            ])
                        ]),
                    ], width=2),
                ]),
            ]),
        ], className="mb-4"),
        
        # Equity Curve and Drawdown
        dbc.Card([
            dbc.CardHeader("Performance Charts"),
            dbc.CardBody([
                dcc.Graph(figure=fig_equity),
                dcc.Graph(figure=fig_drawdown),
            ]),
        ], className="mb-4"),
        
        # Tabs for additional analysis
        dbc.Tabs([
            # Trades Tab
            dbc.Tab([
                html.Div([
                    html.H5(f"Trade History ({len(processed_trades)} trades)", className="mt-3 mb-3"),
                    
                    dash_table.DataTable(
                        id='backtest-trades-table',
                        columns=[
                            {'name': 'ID', 'id': 'id'},
                            {'name': 'Direction', 'id': 'direction'},
                            {'name': 'Entry Date', 'id': 'entry_date'},
                            {'name': 'Exit Date', 'id': 'exit_date'},
                            {'name': 'Entry Price', 'id': 'entry_price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'Exit Price', 'id': 'exit_price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'P&L', 'id': 'profit_loss', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'P&L %', 'id': 'profit_loss_pct_display'},
                            {'name': 'Exit Reason', 'id': 'exit_reason'},
                        ],
                        data=processed_trades,
                        style_header={
                            'backgroundColor': '#2c3e50',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        style_cell={
                            'backgroundColor': '#1e272e',
                            'color': 'white',
                            'textAlign': 'center'
                        },
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{profit_loss} > 0',
                                },
                                'backgroundColor': 'rgba(46, 213, 115, 0.2)',
                            },
                            {
                                'if': {
                                    'filter_query': '{profit_loss} < 0',
                                },
                                'backgroundColor': 'rgba(235, 77, 75, 0.2)',
                            }
                        ],
                        page_size=15,
                        sort_action='native',
                        filter_action='native',
                        export_format='csv',
                        export_headers='display',
                        style_table={
                            'overflowX': 'auto',
                            'minWidth': '100%',
                        }
                    ),
                    
                    html.Div([
                        html.H5("Trade Distribution", className="mt-4 mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=fig_metrics_box), width=12),
                        ]),
                    ]),
                ], className="p-3"),
            ], label="Trade History", tab_id="tab-trades"),
            
            # Performance Metrics Tab
            dbc.Tab([
                html.Div([
                    html.H5("Detailed Performance Metrics", className="mt-3 mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            # Create a detailed metrics table with more metrics
                            dash_table.DataTable(
                                id='detailed-metrics-table',
                                columns=[
                                    {'name': 'Metric', 'id': 'metric'},
                                    {'name': 'Value', 'id': 'value'},
                                ],
                                data=[
                                    {'metric': 'Starting Capital', 'value': f"₹{initial_capital:,.2f}"},
                                    {'metric': 'Ending Capital', 'value': f"₹{final_capital:,.2f}"},
                                    {'metric': 'Net Profit/Loss', 'value': f"₹{final_capital - initial_capital:,.2f}"},
                                    {'metric': 'Return (%)', 'value': f"{total_return:.2f}%"},
                                    {'metric': 'CAGR', 'value': f"{calculate_cagr(initial_capital, final_capital, results.get('duration', 365)):.2f}%"},
                                    {'metric': 'Total Trades', 'value': str(metrics.get('total_trades', 0))},
                                    {'metric': 'Winning Trades', 'value': str(metrics.get('winning_trades', 0))},
                                    {'metric': 'Losing Trades', 'value': str(metrics.get('losing_trades', 0))},
                                    {'metric': 'Win Rate', 'value': f"{win_rate:.2f}%"},
                                    {'metric': 'Loss Rate', 'value': f"{100 - win_rate:.2f}%"},
                                    {'metric': 'Avg. Profit per Trade', 'value': f"₹{metrics.get('avg_profit_per_trade', 0):,.2f}"},
                                    {'metric': 'Avg. Loss per Trade', 'value': f"₹{metrics.get('avg_loss_per_trade', 0):,.2f}"},
                                    {'metric': 'Profit Factor', 'value': f"{profit_factor:.2f}"},
                                    {'metric': 'Max Drawdown (%)', 'value': f"{max_drawdown:.2f}%"},
                                    {'metric': 'Max Drawdown Amount', 'value': f"₹{metrics.get('max_drawdown', 0):,.2f}"},
                                    {'metric': 'Recovery Factor', 'value': f"{calculate_recovery_factor(total_return, max_drawdown):.2f}"},
                                    {'metric': 'Sharpe Ratio', 'value': f"{sharpe_ratio:.2f}"},
                                    {'metric': 'Expectancy', 'value': f"₹{calculate_expectancy(trades):,.2f}"},
                                    {'metric': 'Expectancy Ratio', 'value': f"{calculate_expectancy_ratio(trades):.2f}"},
                                    {'metric': 'System Quality Number (SQN)', 'value': f"{calculate_sqn(trades):.2f}"},
                                ],
                                style_header={
                                    'backgroundColor': '#2c3e50',
                                    'color': 'white',
                                    'fontWeight': 'bold',
                                },
                                style_cell={
                                    'backgroundColor': '#1e272e',
                                    'color': 'white',
                                    'textAlign': 'left',
                                    'padding': '10px'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {
                                            'column_id': 'metric',
                                        },
                                        'fontWeight': 'bold',
                                    }
                                ],
                            )
                        ], width=6),
                        
                        dbc.Col([
                            html.H5("Monthly Returns", className="mb-3"),
                            dcc.Graph(figure=fig_monthly),
                            
                            html.H5("Drawdown Distribution", className="mt-4 mb-3"),
                            dcc.Graph(figure=fig_drawdown_dist),
                        ], width=6),
                    ]),
                ], className="p-3"),
            ], label="Detailed Metrics", tab_id="tab-metrics"),
            
            # Optimization Tab (placeholder for future feature)
            dbc.Tab([
                html.Div([
                    html.H5("Strategy Optimization", className="mt-3 mb-3"),
                    html.P("This feature will allow parameter optimization to find the best configuration for your strategy."),
                    
                    dbc.Alert([
                        html.H4("Coming Soon", className="alert-heading"),
                        html.P("Parameter optimization is under development and will be available in a future update."),
                        html.Hr(),
                        html.P(
                            "Strategy optimization will allow you to test multiple parameter combinations to find the optimal configuration.",
                            className="mb-0"
                        ),
                    ], color="info"),
                ], className="p-3"),
            ], label="Optimization", tab_id="tab-optimization"),
            
            # Export Tab
            dbc.Tab([
                html.Div([
                    html.H5("Export Backtest Results", className="mt-3 mb-3"),
                    html.P("Export your backtest results in different formats for further analysis or reporting."),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Export Options"),
                                dbc.CardBody([
                                    html.P("Select the data you want to export:"),
                                    
                                    dbc.Checklist(
                                        options=[
                                            {"label": "Trade History", "value": "trades"},
                                            {"label": "Equity Curve", "value": "equity"},
                                            {"label": "Performance Metrics", "value": "metrics"},
                                            {"label": "Monthly Returns", "value": "monthly"},
                                        ],
                                        value=["trades", "equity", "metrics"],
                                        id="export-options",
                                        inline=False,
                                    ),
                                    
                                    html.Hr(),
                                    
                                    html.P("Select export format:"),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "CSV", "value": "csv"},
                                            {"label": "Excel", "value": "excel"},
                                            {"label": "JSON", "value": "json"},
                                        ],
                                        value="csv",
                                        id="export-format",
                                        inline=True,
                                    ),
                                    
                                    dbc.Button(
                                        "Export Data",
                                        id="export-data-button",
                                        color="primary",
                                        className="mt-3"
                                    ),
                                    
                                    dbc.Button(
                                        "Export Report",
                                        id="export-report-button",
                                        color="success",
                                        className="mt-3 ms-2"
                                    ),
                                    
                                    html.Div(id="export-status", className="mt-3"),
                                ]),
                            ]),
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Backtest Report Preview"),
                                dbc.CardBody([
                                    html.Img(
                                        src="/assets/report_preview.png",
                                        style={"maxWidth": "100%", "border": "1px solid #ddd", "borderRadius": "5px"},
                                    ),
                                ]),
                            ]),
                        ], width=6),
                    ]),
                ], className="p-3"),
            ], label="Export", tab_id="tab-export"),
        ], active_tab="tab-trades"),
        
        # Action Buttons Row
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    "Back to Configuration",
                    id="backtest-back-to-config-btn",
                    color="secondary",
                    className="me-2",
                ),
                dbc.Button(
                    "Save Configuration",
                    id="backtest-save-config-btn",
                    color="info",
                    className="me-2",
                ),
                dbc.Button(
                    "Compare Strategies",
                    id="backtest-compare-btn",
                    color="warning",
                    className="me-2",
                ),
            ], width=12, className="mt-4 text-end"),
        ]),
    ])


def calculate_monthly_returns(trades, equity_curve, initial_capital):
    """
    Calculate monthly returns from trades and equity curve.
    
    Args:
        trades: List of trade dictionaries
        equity_curve: List of equity point dictionaries
        initial_capital: Initial capital
        
    Returns:
        Dictionary mapping (year, month) to percentage return
    """
    if not equity_curve:
        return {}
    
    # Process equity curve
    monthly_returns = {}
    
    # Convert dates to datetime objects if they're strings
    processed_curve = []
    for point in equity_curve:
        date = point.get('date')
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        processed_curve.append({
            'date': date,
            'equity': point.get('equity')
        })
    
    # Sort by date
    processed_curve = sorted(processed_curve, key=lambda x: x['date'])
    
    # Group by month and calculate returns
    month_equity = {}
    
    for point in processed_curve:
        date = point['date']
        year_month = (date.year, date.month)
        
        # For the first entry of the month, store the opening equity
        if year_month not in month_equity:
            month_equity[year_month] = {
                'open': point['equity'],
                'close': point['equity']
            }
        else:
            # Update the closing equity for the month
            month_equity[year_month]['close'] = point['equity']
    
    # Calculate monthly returns
    for year_month, equity in month_equity.items():
        # Check for zero opening equity to avoid division by zero
        if equity['open'] == 0:
            logger.warning(f"Zero opening equity detected for {year_month[0]}-{year_month[1]}. Skipping monthly return calculation.")
            continue
            
        try:
            monthly_return = (equity['close'] - equity['open']) / equity['open'] * 100
            monthly_returns[year_month] = monthly_return
        except ZeroDivisionError:
            logger.warning(f"Division by zero when calculating monthly return for {year_month[0]}-{year_month[1]}. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Error calculating monthly return for {year_month[0]}-{year_month[1]}: {e}")
            continue
    
    return monthly_returns


def create_monthly_returns_heatmap(monthly_returns):
    """
    Create a heatmap visualization of monthly returns.
    
    Args:
        monthly_returns: Dictionary mapping (year, month) to percentage return
        
    Returns:
        Plotly Figure object
    """
    if not monthly_returns:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No monthly return data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        return fig
    
    # Extract years and months
    years = sorted(set(year for year, month in monthly_returns.keys()))
    months = list(range(1, 13))
    
    # Create the data for the heatmap
    z = []
    text = []
    
    for year in years:
        year_data = []
        year_text = []
        
        for month in months:
            value = monthly_returns.get((year, month), None)
            if value is not None:
                year_data.append(value)
                year_text.append(f"{value:.2f}%")
            else:
                year_data.append(0)
                year_text.append("N/A")
        
        z.append(year_data)
        text.append(year_text)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=years,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0, 'rgb(165, 0, 38)'],
            [0.25, 'rgb(215, 48, 39)'],
            [0.45, 'rgb(244, 109, 67)'],
            [0.5, 'rgb(255, 255, 255)'],
            [0.55, 'rgb(116, 173, 209)'],
            [0.75, 'rgb(69, 117, 180)'],
            [1, 'rgb(49, 54, 149)']
        ],
        colorbar=dict(
            title="Return (%)",
            titleside="right"
        ),
        hoverinfo="text",
        hovertext=[
            [f"{month} {year}: {value:.2f}%" 
             if value is not None else f"{month} {year}: N/A"
             for month, value in zip(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                                   [monthly_returns.get((year, m)) for m in range(1, 13)])]
            for year in years
        ]
    ))
    
    # Update layout
    fig.update_layout(
        title="Monthly Returns (%)",
        template="plotly_dark",
        height=300,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(
            title="Month",
            tickvals=list(range(12)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        yaxis=dict(
            title="Year",
            autorange="reversed"
        )
    )
    
    return fig


def create_drawdown_distribution(equity_curve, initial_capital):
    """
    Create a distribution visualization of drawdowns.
    
    Args:
        equity_curve: List of equity point dictionaries
        initial_capital: Initial capital
        
    Returns:
        Plotly Figure object
    """
    if not equity_curve:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No drawdown data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        return fig
    
    # Process equity curve
    equity_values = []
    
    for point in equity_curve:
        equity = point.get('equity')
        equity_values.append(equity)
    
    # Calculate drawdowns
    hwm = [initial_capital]
    for i in range(1, len(equity_values)):
        hwm.append(max(hwm[-1], equity_values[i]))
    
    drawdowns = []
    for i in range(len(equity_values)):
        try:
            # Avoid division by zero
            if hwm[i] <= 0:
                logger.warning(f"Zero or negative high water mark at index {i}. Skipping drawdown calculation.")
                drawdowns.append(0)  # Use 0 as a fallback value
            else:
                drawdowns.append((1 - equity_values[i]/hwm[i]) * 100)
        except ZeroDivisionError:
            logger.warning(f"Division by zero when calculating drawdown at index {i}")
            drawdowns.append(0)  # Use 0 as a fallback value
        except Exception as e:
            logger.error(f"Error calculating drawdown at index {i}: {e}")
            drawdowns.append(0)  # Use 0 as a fallback value
    
    # Create the histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=drawdowns,
        histnorm='percent',
        marker=dict(
            color="rgba(255, 0, 0, 0.7)",
            line=dict(
                color="rgba(255, 0, 0, 1)",
                width=1
            )
        ),
        opacity=0.7,
        name="Drawdown Distribution"
    ))
    
    # Calculate statistics for annotations
    avg_dd = np.mean(drawdowns)
    max_dd = np.max(drawdowns)
    median_dd = np.median(drawdowns)
    
    # Add annotations
    fig.add_annotation(
        x=avg_dd,
        y=0.9,
        text=f"Avg: {avg_dd:.2f}%",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    fig.add_annotation(
        x=max_dd,
        y=0.8,
        text=f"Max: {max_dd:.2f}%",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-30
    )
    
    # Update layout
    fig.update_layout(
        title="Drawdown Distribution",
        template="plotly_dark",
        height=300,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(title="Drawdown (%)"),
        yaxis=dict(title="Frequency (%)"),
        bargap=0.1,
        showlegend=False
    )
    
    return fig


def create_metrics_boxplots(trades):
    """
    Create boxplots for key trade metrics.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Plotly Figure object
    """
    if not trades:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No trade data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        return fig
    
    # Extract trade data
    profit_loss = [t.get('profit_loss', 0) for t in trades]
    profit_loss_pct = [t.get('profit_loss_pct', 0) for t in trades]
    
    # Create the subplot figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=("P&L Distribution", "P&L % Distribution"))
    
    # Add traces
    fig.add_trace(
        go.Box(
            y=profit_loss,
            name="P&L",
            boxmean=True,
            marker=dict(color="blue")
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(
            y=profit_loss_pct,
            name="P&L %",
            boxmean=True,
            marker=dict(color="green")
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=30, r=30, t=40, b=30),
        showlegend=False
    )
    
    return fig


def calculate_cagr(initial_capital, final_capital, days):
    """
    Calculate the Compound Annual Growth Rate.
    
    Args:
        initial_capital: Initial capital
        final_capital: Final capital
        days: Number of days in the backtest
        
    Returns:
        CAGR as a percentage
    """
    try:
        # Validate inputs to avoid division by zero and negative values
        if initial_capital <= 0:
            logger.warning(f"Initial capital must be positive, got {initial_capital}. CAGR defaulting to 0.0.")
            return 0.0
        
        if days <= 0:
            logger.warning(f"Number of days must be positive, got {days}. CAGR defaulting to 0.0.")
            return 0.0
        
        years = days / 365.0
        if years <= 0:
            logger.warning(f"Calculated years must be positive, got {years}. CAGR defaulting to 0.0.")
            return 0.0
        
        # Calculate CAGR
        cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
        
        # Handle potential NaN or infinite values
        if np.isnan(cagr) or np.isinf(cagr):
            logger.warning(f"CAGR calculation resulted in invalid value: {cagr}. Defaulting to 0.0.")
            return 0.0
            
        return cagr
    except ZeroDivisionError as e:
        logger.warning(f"Division by zero when calculating CAGR: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating CAGR: {e}")
        return 0.0


def calculate_recovery_factor(total_return, max_drawdown):
    """
    Calculate the Recovery Factor.
    
    Args:
        total_return: Total return percentage
        max_drawdown: Maximum drawdown percentage
        
    Returns:
        Recovery Factor
    """
    try:
        # Check for valid max_drawdown before division
        if max_drawdown <= 0:
            logger.debug("Max drawdown is zero or negative. Recovery factor defaulting to 0.0.")
            return 0.0
        
        return abs(total_return / max_drawdown)
    except ZeroDivisionError:
        logger.warning("Division by zero when calculating recovery factor. Max drawdown cannot be zero.")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating recovery factor: {e}")
        return 0.0


def calculate_expectancy(trades):
    """
    Calculate the Expectancy.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Expectancy value
    """
    try:
        if not trades:
            logger.debug("No trades provided for expectancy calculation")
            return 0.0
        
        # Calculate the average profit/loss per trade
        total_pl = sum(t.get('profit_loss', 0) for t in trades)
        num_trades = len(trades)
        
        if num_trades == 0:
            logger.warning("Division by zero: No trades for expectancy calculation")
            return 0.0
            
        expectancy = total_pl / num_trades
        
        # Handle potential NaN or infinite values
        if np.isnan(expectancy) or np.isinf(expectancy):
            logger.warning(f"Expectancy calculation resulted in invalid value: {expectancy}. Defaulting to 0.0.")
            return 0.0
            
        return expectancy
    except ZeroDivisionError as e:
        logger.warning(f"Division by zero when calculating expectancy: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating expectancy: {e}")
        return 0.0


def calculate_expectancy_ratio(trades):
    """
    Calculate the Expectancy Ratio.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Expectancy Ratio
    """
    try:
        if not trades:
            return 0.0
        
        winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
        
        if not winning_trades or not losing_trades:
            logger.info("No winning or losing trades found. Expectancy ratio defaulting to 0.0.")
            return 0.0
        
        win_rate = len(winning_trades) / len(trades)
        
        # Check for empty lists before calculating averages
        if len(winning_trades) == 0:
            logger.warning("No winning trades for expectancy ratio calculation")
            return 0.0
            
        if len(losing_trades) == 0:
            logger.warning("No losing trades for expectancy ratio calculation")
            return 0.0
            
        avg_win = sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades)
        avg_loss = sum(abs(t.get('profit_loss', 0)) for t in losing_trades) / len(losing_trades)
        
        # Check for zero avg_loss to avoid division by zero
        if avg_loss <= 0:
            logger.warning("Average loss is zero or negative. Expectancy ratio defaulting to 0.0.")
            return 0.0
        
        return (win_rate * avg_win / avg_loss - (1 - win_rate))
    except ZeroDivisionError as e:
        logger.warning(f"Division by zero when calculating expectancy ratio: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating expectancy ratio: {e}")
        return 0.0


def calculate_sqn(trades):
    """
    Calculate the System Quality Number (SQN).
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        SQN value
    """
    try:
        if not trades or len(trades) < 2:
            logger.debug("Not enough trades to calculate SQN. Need at least 2 trades.")
            return 0.0
        
        profit_loss = [t.get('profit_loss', 0) for t in trades]
        
        mean_pl = np.mean(profit_loss)
        std_pl = np.std(profit_loss)
        
        # Check for zero or negative standard deviation to avoid division by zero
        if std_pl <= 0:
            logger.warning("Standard deviation of profit/loss is zero or negative. SQN defaulting to 0.0.")
            return 0.0
        
        return (mean_pl / std_pl) * np.sqrt(len(trades))
    except ZeroDivisionError as e:
        logger.warning(f"Division by zero when calculating SQN: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating SQN: {e}")
        return 0.0


def format_date(date_value):
    """
    Format a date value consistently.
    
    Args:
        date_value: Date value, can be string or datetime
        
    Returns:
        Formatted date string
    """
    if not date_value:
        return ""
    
    if isinstance(date_value, str):
        try:
            date_value = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
        except ValueError:
            return date_value
    
    return date_value.strftime("%Y-%m-%d")


def register_advanced_backtest_callbacks(app):
    """
    Register all advanced backtesting callbacks.
    
    Args:
        app: Dash application instance
    """
    # Toggle trailing stop input based on checkbox
    @app.callback(
        Output("advanced-backtest-trailing-stop-pct", "disabled"),
        Input("advanced-backtest-trailing-stop", "value")
    )
    def toggle_trailing_stop_input(use_trailing_stop):
        """Toggle the trailing stop percentage input based on checkbox state."""
        return not use_trailing_stop
    
    # Run backtest and store results
    @app.callback(
        Output("advanced-backtest-results-store", "data"),
        Output("advanced-backtest-params-store", "data"),
        Input("run-advanced-backtest-button", "n_clicks"),
        State("advanced-backtest-symbol-input", "value"),
        State("advanced-backtest-exchange-dropdown", "value"),
        State("advanced-backtest-timeframe-dropdown", "value"),
        State("advanced-backtest-start-date", "date"),
        State("advanced-backtest-end-date", "date"),
        State("advanced-backtest-strategy-dropdown", "value"),
        State("advanced-backtest-initial-capital", "value"),
        State("advanced-backtest-position-size", "value"),
        State("advanced-backtest-max-positions", "value"),
        State("advanced-backtest-pyramiding", "value"),
        State("advanced-backtest-signal-threshold", "value"),
        State("advanced-backtest-require-alignment", "value"),
        State("advanced-backtest-tp-multiplier", "value"),
        State("advanced-backtest-trailing-stop", "value"),
        State("advanced-backtest-trailing-stop-pct", "value"),
        prevent_initial_call=True
    )
    def run_advanced_backtest(n_clicks, symbol, exchange, timeframe, start_date, end_date, 
                         strategy, initial_capital, position_size, max_positions, pyramiding,
                         signal_threshold, require_alignment, tp_multiplier, 
                         use_trailing_stop, trailing_stop_pct):
        """Run the advanced backtest and store results."""
        if not n_clicks or not symbol:
            return None, None
        
        # Log the backtest run
        logger.info(f"Running advanced backtest for {symbol} on {exchange} [{timeframe}]")
        
        # Store the parameters
        params = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'strategy': strategy,
            'initial_capital': initial_capital,
            'position_size': position_size,
            'max_positions': max_positions,
            'pyramiding': pyramiding,
            'signal_threshold': signal_threshold,
            'require_alignment': require_alignment,
            'tp_multiplier': tp_multiplier,
            'use_trailing_stop': use_trailing_stop,
            'trailing_stop_pct': trailing_stop_pct
        }
        
        try:
            # Load config
            config_path = os.path.join(project_root, "config", "config.json")
            config = load_config(config_path)
            
            # Create data fetcher and scanner
            data_fetcher = DataFetcher(config)
            scanner = FibCycleScanner(config)
            
            # Create backtest engine
            engine = BacktestEngine(config, scanner=scanner, data_fetcher=data_fetcher)
            
            # Parse dates
            try:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
            except ValueError as date_err:
                logger.error(f"Date parsing error: {date_err}")
                start_date_obj = datetime.now() - timedelta(days=365)
                end_date_obj = datetime.now()
            
            # Calculate appropriate lookback based on date range
            if start_date_obj and end_date_obj:
                # Calculate days between dates with safety margin
                date_span = (end_date_obj - start_date_obj).days
                # Use at least 365 days of lookback but not more than 500
                lookback = min(max(date_span + 10, 365), 500)
            else:
                lookback = 365  # Default to one year
                
            logger.info(f"Using lookback of {lookback} days for backtest")
            
            # Create backtest parameters
            backtest_params = BacktestParameters(
                symbol=symbol,
                exchange=exchange,
                interval=timeframe,
                start_date=start_date_obj,
                end_date=end_date_obj,
                lookback=lookback,  # Add explicit lookback to avoid index errors
                initial_capital=float(initial_capital) if initial_capital else 100000.0,
                position_size_pct=float(position_size) if position_size else 10.0,
                max_open_positions=int(max_positions) if max_positions else 5,
                pyramiding=int(pyramiding) if pyramiding else 0,
                min_strength=float(signal_threshold) if signal_threshold else 0.2,
                require_alignment=bool(require_alignment),
                take_profit_multiplier=float(tp_multiplier) if tp_multiplier else 2.0,
                trailing_stop=bool(use_trailing_stop),
                trailing_stop_pct=float(trailing_stop_pct) if trailing_stop_pct else 5.0,
                strategy_type=strategy  # Add strategy type parameter
            )
            
            # Ensure we have data before running the backtest
            try:
                # Try to get data directly first
                data = data_fetcher.get_data(
                    symbol=symbol, 
                    exchange=exchange, 
                    interval=timeframe,
                    lookback=lookback,
                    force_download=False  # Try cache first to be faster
                )
                
                if data is None or data.empty:
                    logger.warning(f"No data found for {symbol} on {exchange} ({timeframe}). Trying alternative sources...")
                    
                    # Try with force download
                    data = data_fetcher.get_data(
                        symbol=symbol, 
                        exchange=exchange, 
                        interval=timeframe,
                        lookback=lookback,
                        force_download=True  # Force refresh from data source
                    )
                    
                    if data is None or data.empty:
                        # Check for cached file directly
                        cache_file = os.path.join(
                            project_root, "data", "cache", 
                            f"{exchange}_{symbol}_{timeframe}.csv"
                        )
                        
                        if os.path.exists(cache_file):
                            logger.info(f"Loading data from cache file: {cache_file}")
                            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        
                        # If still no data, try to generate mock data
                        if data is None or data.empty:
                            logger.warning(f"No real data available for {symbol}. Using mock data.")
                            from data.mock_data_generator import generate_mock_price_data, save_mock_data_to_cache
                            
                            # Generate mock data
                            data = generate_mock_price_data(
                                symbol=symbol,
                                lookback=lookback * 1.5,  # Generate extra data for safety
                                interval=timeframe
                            )
                            
                            # Save to cache for future use
                            save_mock_data_to_cache(symbol, exchange, timeframe, data)
                
                # Verify we have enough data
                if data is None or data.empty:
                    logger.error(f"Failed to get data for {symbol} on {exchange} ({timeframe})")
                    return {"error": f"Could not get market data for {symbol} on {exchange} ({timeframe})"}, params
                    
                # Check if we have enough data for the lookback period
                if len(data) < lookback:
                    logger.warning(f"Not enough data for requested lookback ({len(data)} < {lookback}). Adjusting parameters.")
                    backtest_params.lookback = max(100, len(data) - 10)  # Adjust lookback, keep at least 100 bars
                
                logger.info(f"Successfully retrieved {len(data)} bars of data for {symbol}")
                
                # Use the new backtest adapter instead of the old engine
                try:
                    # Import the adapter
                    from web.backtest_adapter import run_backtest_from_ui
                    
                    # Create parameters dictionary for adapter
                    adapter_params = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'start_date': start_date_obj,
                        'end_date': end_date_obj,
                        'initial_capital': initial_capital,
                        'position_size': position_size,
                        'max_positions': max_positions,
                        'pyramiding': pyramiding,
                        'signal_threshold': signal_threshold,
                        'require_alignment': require_alignment,
                        'tp_multiplier': tp_multiplier,
                        'trailing_stop': use_trailing_stop,
                        'trailing_stop_pct': trailing_stop_pct,
                        'strategy': strategy
                    }
                    
                    # Run backtest using adapter
                    results = run_backtest_from_ui(adapter_params)
                    
                    # Log completion
                    logger.info(f"Backtest completed for {symbol} with {len(results.get('trades', []))} trades")
                    
                    return results, params
                    
                except ImportError as e:
                    logger.error(f"Could not import backtest adapter: {e}")
                    logger.warning("Falling back to old backtesting engine")
                    
                    # Fall back to old engine if adapter not available
                    results = engine.run_backtest(backtest_params)
                    
                    # Make results JSON serializable
                    serializable_results = engine._make_serializable(results)
                    
                    # Log completion
                    logger.info(f"Backtest completed for {symbol} with {len(results.get('trades', []))} trades")
                    
                    return serializable_results, params
                
            except Exception as e:
                logger.error(f"Error in backtest execution: {e}", exc_info=True)
                return {"error": f"Backtest execution error: {str(e)}"}, params
            
        except Exception as e:
            # Log the error
            logger.error(f"Backtest error: {str(e)}", exc_info=True)
            
            # Return error
            return {"error": str(e)}, params
    
    # Update UI based on backtest results
    @app.callback(
        Output("advanced-backtest-results-content", "children"),
        Input("advanced-backtest-results-store", "data"),
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def update_backtest_results_ui(results):
        """Update the UI with backtest results."""
        if not results:
            return create_results_placeholder()
        
        logger.info(f"Updating UI with backtest results containing {len(results.get('trades', []))} trades")
        
        return create_backtest_results_ui(results)
    
    # Handle "Back to Configuration" button
    @app.callback(
        Output("advanced-backtest-results-container", "children", allow_duplicate=True),
        Input("backtest-back-to-config-btn", "n_clicks"),
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def back_to_configuration(n_clicks_back):
        """Go back to the results placeholder."""
        if n_clicks_back:
            return create_results_placeholder()
        
        return dash.no_update