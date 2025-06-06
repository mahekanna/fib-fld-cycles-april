"""
Fibonacci Harmonic Trading System - Main Dashboard

IMPORTANT NOTES FOR MAINTENANCE:
1. This dashboard uses callbacks with allow_duplicate=True to handle multiple callbacks
   updating the same output elements. When making changes, make sure to keep this pattern.
2. If you encounter "Output X is already in use" errors, run kill_cache_and_restart.sh
   to completely clear all Dash caches.
3. When adding new callbacks that modify existing outputs, use allow_duplicate=True
   and check the trigger using dash.callback_context
4. Each callback should only update its outputs when explicitly triggered (check ctx.triggered)
5. For deep cache clearing, use: ./kill_cache_and_restart.sh

For more information, see the docs/DASHBOARD_GUIDE.md file.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import io
import json
import glob
from typing import Dict, List, Optional
import sys
import os
import random
import logging

# Global variables for dashboard
APP_CONFIG_PATH = "config/config.json"  # Default path that will be updated in create_app

# Import the centralized logging system
from utils.logging_utils import get_component_logger

# Configure logging
logger = get_component_logger("main_dashboard")

# Add project directories to path - make this explicit
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
print(f"Project root added to path: {project_root}")

# Import core components directly from the project structure using absolute imports
from core.scanner import FibCycleScanner
from models.scan_parameters import ScanParameters
from models.scan_result import ScanResult
from utils.config import load_config
from storage.results_repository import ResultsRepository

# Import web UI modules using absolute imports - ensure these are all correctly loaded
from web.cycle_visualization import create_cycle_visualization
from web.fld_visualization import create_fld_visualization
from web.harmonic_visualization import create_harmonic_visualization
from web.scanner_dashboard import create_scanner_dashboard
from web.trading_strategies_ui import create_strategy_dashboard
from web.enhanced_entry_exit_ui import create_enhanced_entry_exit_ui
from web.advanced_strategies_ui import create_strategy_dashboard as create_advanced_strategy_dashboard
from web.advanced_strategies_ui import create_batch_advanced_signals
from web.realtime_price_updater import create_realtime_price_components, register_realtime_price_callbacks, get_realtime_price_stylesheet

# Double-check imports to make sure we're using the correct modules
logger.info("Loaded visualization modules: cycle_visualization, fld_visualization, harmonic_visualization, scanner_dashboard, trading_strategies_ui, advanced_strategies_ui")


def create_app(config_path: str) -> dash.Dash:
    """
    Create the Dash application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dash application instance
    """
    # Load configuration
    config = load_config(config_path)
    
    # Store the config path as a global variable for use in callbacks
    global APP_CONFIG_PATH
    APP_CONFIG_PATH = config_path
    
    # Initialize scanner and repository
    scanner = FibCycleScanner(config)
    repository = ResultsRepository(config)
    
    # Create Dash app with explicit assets directory to clear cache
    import shutil
    assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(assets_path, exist_ok=True)
    
    # Create Dash app with strict callback validation disabled to avoid DuplicateCallback errors
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        assets_folder=assets_path,
        prevent_initial_callbacks=True
    )
    
    # Add realtime price stylesheet
    app.index_string = app.index_string.replace(
        '</head>',
        f'<style>{get_realtime_price_stylesheet()}</style></head>'
    )
    
    # Set app title
    app.title = "Fibonacci Harmonic Trading System"
    
    # Create real-time price components
    realtime_components = create_realtime_price_components()
    
    # Register real-time price update callbacks
    register_realtime_price_callbacks(app)
    
    # Define app layout
    app.layout = html.Div([
        # Store component for shared data
        dcc.Store(id='scan-results-store'),
        dcc.Store(id='trading-data-store'),
        dcc.Store(id='navigation-store'),  # For navigation between components
        
        # Real-time price components
        realtime_components["price_store"],
        realtime_components["activation_store"],
        realtime_components["interval"],
        
        # Location component for URL-based navigation
        dcc.Location(id='url', refresh=False),
        
        # System information card - using a card with no ID to avoid any callback issues
        dbc.Card(
            dbc.CardBody([
                html.H5("System Status", className="card-title"),
                html.P("Ready to analyze market data", className="card-text")
            ]),
            style={'margin': '10px', 'backgroundColor': '#e2e3e5', 'color': '#383d41'}
        ),
        
        # Header
        dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H2("Fibonacci Harmonic Trading System", className="text-white")),
                    dbc.Col(
                        html.Div([
                            realtime_components["status"],
                            dbc.Button("Scan", id="scan-button", color="success", className="ms-auto"),
                        ], className="d-flex align-items-center"),
                        width={"size": 1},
                    ),
                ]),
            ]),
            color="primary",
            dark=True,
        ),
        
        # Main content container
        dbc.Container([
            dbc.Row([
                # Left sidebar
                dbc.Col([
                    html.Div([
                        html.H4("Analysis Parameters", className="mt-3 text-dark"),
                        
                        html.Label("Symbol", className="text-dark"),
                        dbc.Input(id="symbol-input", value="NIFTY", type="text", className="mb-2"),
                        
                        html.Label("Exchange", className="mt-2 text-dark"),
                        dbc.Input(id="exchange-input", value="NSE", type="text", className="mb-2"),
                        
                        html.Label("Interval", className="mt-2 text-dark"),
                        dcc.Dropdown(
                            id="interval-dropdown",
                            options=[
                                {"label": "Monthly", "value": "monthly"},
                                {"label": "Weekly", "value": "weekly"},
                                {"label": "Daily", "value": "daily"},
                                {"label": "4 Hour", "value": "4h"},
                                {"label": "2 Hour", "value": "2h"},
                                {"label": "1 Hour", "value": "1h"},
                                {"label": "30 Minute", "value": "30m"},
                                {"label": "15 Minute", "value": "15m"},
                                {"label": "5 Minute", "value": "5m"},
                                {"label": "1 Minute", "value": "1m"},
                            ],
                            value="daily",
                            clearable=False,
                            # Dark mode fixes for dropdown
                            style={
                                'color': 'black',
                                'background-color': 'white',
                            },
                            className="mb-2"
                        ),
                        
                        html.Label("Lookback (bars)", className="mt-2 text-dark"),
                        dbc.Input(id="lookback-input", value="1000", type="number", className="mb-2"),
                        
                        html.Label("Number of Cycles", className="mt-2 text-dark"),
                        dbc.Input(id="cycles-input", value="3", type="number", min=1, max=5, className="mb-2"),
                        
                        html.Label("Price Source", className="mt-2 text-dark"),
                        dcc.Dropdown(
                            id="price-source-dropdown",
                            options=[
                                {"label": "Close", "value": "close"},
                                {"label": "Open", "value": "open"},
                                {"label": "High", "value": "high"},
                                {"label": "Low", "value": "low"},
                                {"label": "HL2 (High+Low)/2", "value": "hl2"},
                                {"label": "HLC3 (High+Low+Close)/3", "value": "hlc3"},
                                {"label": "OHLC4 (Open+High+Low+Close)/4", "value": "ohlc4"},
                                {"label": "Weighted Close (HLCC4)", "value": "hlcc4"},
                            ],
                            value="close",
                            clearable=False,
                            # Dark mode fixes for dropdown
                            style={
                                'color': 'black',
                                'background-color': 'white',
                            },
                            className="mb-2"
                        ),
                        
                        html.Hr(),
                        
                        # Real-time price controls
                        html.H4("Real-time Prices", className="text-dark mt-2"),
                        realtime_components["controls"],
                        
                        html.Hr(),
                        
                        html.H4("Batch Scan", className="text-dark"),
                        dbc.Textarea(
                            id="symbols-textarea",
                            placeholder="Enter symbols, one per line...",
                            rows=5,
                            className="mb-2 bg-white text-dark"
                        ),
                        dbc.Button("Batch Scan", id="batch-scan-button", color="warning", className="mt-2"),
                    ], className="p-3 bg-light rounded shadow")
                ], width=3),
                
                # Main content area - Enhanced Tabs with new modules
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="Analysis Results", tab_id="tab-analysis", children=[
                            html.Div(id="analysis-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Enhanced Entry/Exit", tab_id="tab-enhanced", children=[
                            html.Div(id="enhanced-entry-exit-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Cycle Visualization", tab_id="tab-cycles", children=[
                            html.Div(id="cycle-visualization-content", className="mt-3")
                        ]),
                        dbc.Tab(label="FLD Analysis", tab_id="tab-fld", children=[
                            html.Div(id="fld-visualization-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Harmonic Patterns", tab_id="tab-harmonic", children=[
                            html.Div(id="harmonic-visualization-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Scanner Dashboard", tab_id="tab-scanner", children=[
                            html.Div(id="scanner-dashboard-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Trading Strategies", tab_id="tab-strategies", children=[
                            html.Div(id="trading-strategies-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Batch Results", tab_id="tab-batch", children=[
                            html.Div(id="batch-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Advanced Strategies", tab_id="tab-advanced-strategies", children=[
                            html.Div(id="advanced-strategies-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Batch Advanced Signals", tab_id="tab-batch-advanced", children=[
                            html.Div(id="batch-advanced-content", className="mt-3")
                        ]),
                    ], id="main-tabs", active_tab="tab-analysis"),
                ], width=9),
            ], className="mt-3"),
        ], fluid=True),
    ])
    
    # Register callbacks
    register_callbacks(app, scanner, repository)
    
    return app


def register_callbacks(app, scanner, repository):
    """
    Register all app callbacks.
    
    Args:
        app: Dash application instance
        scanner: FibCycleScanner instance
        repository: ResultsRepository instance
    """
    # Clear any previous cached callbacks to avoid conflicts - this is a complete reset
    
    # Callback for main scan functionality
    @app.callback(
        # Use 8 outputs with allow_duplicate for shared outputs
        Output("scan-results-store", "data"),
        Output("analysis-content", "children", allow_duplicate=True),
        Output("enhanced-entry-exit-content", "children", allow_duplicate=True),  # New output
        Output("cycle-visualization-content", "children", allow_duplicate=True),
        Output("fld-visualization-content", "children", allow_duplicate=True),
        Output("harmonic-visualization-content", "children", allow_duplicate=True),
        Output("scanner-dashboard-content", "children", allow_duplicate=True),
        Output("advanced-strategies-content", "children", allow_duplicate=True),  # Advanced strategies
        Output("batch-advanced-content", "children", allow_duplicate=True),      # Batch advanced signals
        # Make all parameters Inputs instead of State for responsive UI
        Input("scan-button", "n_clicks"),
        Input("symbol-input", "value"),
        Input("exchange-input", "value"),
        Input("interval-dropdown", "value"),
        Input("lookback-input", "value"),
        Input("cycles-input", "value"),
        Input("price-source-dropdown", "value"),
        prevent_initial_call=True,
    )
    def scan_symbol(n_clicks, symbol, exchange, interval, lookback, num_cycles, price_source):
        """Handle single symbol scan and update all visualization modules"""
        # Get the triggered input to ensure we only scan when the button is clicked
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'scan-button.n_clicks' or not n_clicks:
            return [dash.no_update] * 7
            
        if not symbol:
            return [dash.no_update] * 7
        
        try:
            # Create scan parameters
            params = ScanParameters(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                lookback=int(lookback),
                num_cycles=int(num_cycles),
                price_source=price_source,
                generate_chart=True
            )
            
            logger.info(f"Scanning {symbol} with {params}")
            
            # Perform scan
            result = scanner.analyze_symbol(params)
            
            # If scan failed, log the error
            if not result.success:
                logger.error(f"Scan failed for {symbol}: {result.error}")
            
            # Save result to repository
            repository.save_result(result)
            
            # Update price with real-time data if available
            try:
                from data.data_refresher import get_data_refresher
                
                # Need to create a new data refresher with the current config
                config_to_use = load_config(APP_CONFIG_PATH)
                refresher = get_data_refresher(config_to_use)
                
                # Force a refresh of the data
                refresher.refresh_symbol(symbol, exchange, interval)
                
                # Get latest data
                latest_data = refresher.get_latest_data(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    refresh_if_needed=True
                )
                
                if latest_data is not None and not latest_data.empty:
                    latest_price = latest_data['close'].iloc[-1]
                    old_price = result.price
                    
                    # Log the real-time price but don't overwrite for consistency
                    logger.warning(f"REALTIME: Got {symbol} price {latest_price:.2f} (original: {old_price:.2f})")
                    
                    # Log significant differences
                    if abs(latest_price - old_price) > 0.01:
                        logger.warning(f"PRICE DIFFERENCE: {symbol} original: {old_price:.2f}, real-time: {latest_price:.2f}")
                
                # Add symbol to priority refresh list
                refresher.add_priority_symbol(symbol)
                
                # Start refresher thread if not running
                if not refresher.running:
                    refresher.start_refresh_thread()
            except Exception as e:
                logger.error(f"Error updating real-time price: {e}")
            
            # Serialize for store
            result_data = result.to_dict()
            
            # Create content for each visualization module
            analysis_content = create_analysis_content(result)
            enhanced_entry_exit = create_enhanced_entry_exit_ui(result)  # New enhanced component
            cycle_visualization = create_cycle_visualization(result)
            fld_visualization = create_fld_visualization(result)
            harmonic_visualization = create_harmonic_visualization(result)
            scanner_dashboard = create_scanner_dashboard([result])
            
            # Add advanced strategies content
            advanced_strategies = create_advanced_strategy_dashboard(result=result)
            
            # For batch advanced signals, we need to create a list result
            logger.info("Creating batch advanced signals with app instance (single result)")
            batch_advanced_signals = create_batch_advanced_signals([result], app=app)
            
            return (
                result_data, 
                analysis_content,
                enhanced_entry_exit,  # Return the enhanced component 
                cycle_visualization, 
                fld_visualization, 
                harmonic_visualization, 
                scanner_dashboard,
                advanced_strategies,
                batch_advanced_signals
            )
            
        except Exception as e:
            # Handle any errors and return a placeholder
            logger.exception(f"Error scanning {symbol}: {str(e)}")
            
            # Create error content
            error_content = html.Div([
                html.H3(f"Error scanning {symbol}"),
                html.P(f"An error occurred: {str(e)}"),
                html.Hr(),
                html.P("Please try again or check the logs for more information.")
            ])
            
            # Return error content for all tabs
            return (
                {},  # Empty store
                error_content,  # Analysis tab
                error_content,  # Enhanced entry/exit tab
                error_content,  # Cycle visualization tab
                error_content,  # FLD visualization tab
                error_content,  # Harmonic visualization tab
                error_content,  # Scanner dashboard tab
                error_content,  # Advanced strategies tab
                error_content   # Batch advanced signals tab
            )
    
    # Callback for batch scan
    @app.callback(
        Output("batch-content", "children"),
        Output("symbols-textarea", "value", allow_duplicate=True),
        Output("batch-advanced-content", "children", allow_duplicate=True),  # Add batch advanced signals
        Input("batch-scan-button", "n_clicks"),
        Input("symbols-textarea", "value"),
        Input("exchange-input", "value"),
        Input("interval-dropdown", "value"),
        Input("lookback-input", "value"),
        Input("cycles-input", "value"),
        Input("price-source-dropdown", "value"),
        prevent_initial_call=True,
    )
    def scan_batch(n_clicks, symbols_text, exchange, interval, lookback, num_cycles, price_source):
        """Handle batch symbol scan - uses symbols.csv file if text area is empty"""
        # Get the triggered input to ensure we only scan when the button is clicked
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'batch-scan-button.n_clicks' or not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        
        symbols = []
        symbols_display = ""
        
        try:
            # Use symbols from text area if provided, otherwise read from symbols.csv
            if symbols_text and symbols_text.strip():
                # Parse symbols from text area
                symbols = [s.strip() for s in symbols_text.split("\n") if s.strip()]
                symbols_display = symbols_text  # Keep original text
            else:
                # Read from symbols.csv file
                symbols_csv_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    "config", "data", "symbols.csv"
                )
                
                if os.path.exists(symbols_csv_path):
                    try:
                        # Read CSV file
                        symbols_df = pd.read_csv(symbols_csv_path)
                        
                        # Filter by exchange if provided
                        if exchange:
                            symbols_df = symbols_df[symbols_df['exchange'] == exchange]
                            
                        # Extract symbols
                        symbols = symbols_df['symbol'].tolist()
                        
                        # Update text area with loaded symbols
                        symbols_display = "\n".join(symbols)
                        logger.info(f"Loaded {len(symbols)} symbols from {symbols_csv_path}")
                    except Exception as e:
                        logger.error(f"Error loading symbols from CSV: {str(e)}")
                        return html.Div([
                            html.H3("Batch Scan Error"),
                            html.P(f"Error loading symbols from CSV file: {str(e)}"),
                            html.Hr(),
                            html.P(f"Path: {symbols_csv_path}")
                        ]), dash.no_update, dash.no_update
                else:
                    logger.error(f"Symbols CSV file not found: {symbols_csv_path}")
                    return html.Div([
                        html.H3("Batch Scan Error"),
                        html.P("Symbols CSV file not found. Please provide symbols in the text area."),
                        html.Hr(),
                        html.P(f"Expected path: {symbols_csv_path}")
                    ]), dash.no_update, dash.no_update
            
            if not symbols:
                logger.warning("No symbols to scan")
                return html.Div([
                    html.H3("Batch Scan Error"),
                    html.P("No symbols to scan. Please provide symbols in the text area or create a symbols.csv file."),
                ]), dash.no_update, dash.no_update
            
            # Create scan parameters for each symbol
            params_list = [
                ScanParameters(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    lookback=int(lookback),
                    num_cycles=int(num_cycles),
                    price_source=price_source,
                    generate_chart=False
                )
                for symbol in symbols
            ]
            
            # Log batch scan start
            logger.info(f"Batch scan starting for {len(symbols)} symbols")
            
            # Perform batch scan
            results = scanner.scan_batch(params_list)
            
            # Update prices with real-time data if available
            try:
                from data.data_refresher import get_data_refresher
                
                # Use the current application config for data refresher
                config_to_use = load_config(APP_CONFIG_PATH)
                refresher = get_data_refresher(config_to_use)
                
                for result in results:
                    if not result.success:
                        continue
                        
                    # Force a refresh of the data
                    refresher.refresh_symbol(result.symbol, result.exchange, result.interval)
                    
                    # Get latest data
                    latest_data = refresher.get_latest_data(
                        symbol=result.symbol,
                        exchange=result.exchange,
                        interval=result.interval,
                        refresh_if_needed=True
                    )
                    
                    if latest_data is not None and not latest_data.empty:
                        latest_price = latest_data['close'].iloc[-1]
                        old_price = result.price
                        
                        # Log the real-time price but don't overwrite for consistency
                        logger.warning(f"REALTIME BATCH: Got {result.symbol} price {latest_price:.2f} (original: {old_price:.2f})")
                        
                        # Log significant differences
                        if abs(latest_price - old_price) > 0.01:
                            logger.warning(f"BATCH PRICE DIFFERENCE: {result.symbol} original: {old_price:.2f}, real-time: {latest_price:.2f}")
                    
                    # Add symbol to priority refresh list
                    refresher.add_priority_symbol(result.symbol)
                
                # Start refresher thread if not running
                if not refresher.running:
                    refresher.start_refresh_thread()
            except Exception as e:
                logger.error(f"Error updating real-time batch prices: {e}")
            
            # Save results to repository
            repository.save_batch_results(results)
            
            # Create batch results content using scanner dashboard
            batch_content = create_scanner_dashboard(results)
            
            # Log completion
            succeeded = sum(1 for r in results if r.success)
            failed = len(results) - succeeded
            logger.info(f"Batch scan complete. Success: {succeeded}, Failed: {failed}")
            
            # Make sure all results have data for advanced strategy analysis
            for result in results:
                if result.success and not hasattr(result, 'data'):
                    try:
                        # Add data to the result object for strategy analysis
                        data_fetcher = scanner.data_fetcher
                        result.data = data_fetcher.get_data(
                            symbol=result.symbol,
                            exchange=result.exchange,
                            interval=result.interval,
                            lookback=int(lookback)
                        )
                        logger.info(f"Added data to result for {result.symbol}")
                    except Exception as e:
                        logger.error(f"Error adding data to result for {result.symbol}: {e}")
            
            # Create batch advanced signals - make sure app instance is passed
            logger.info("Creating batch advanced signals with app instance")
            batch_advanced_signals = create_batch_advanced_signals(results, app=app)
            
            return batch_content, symbols_display, batch_advanced_signals
            
        except Exception as e:
            # Handle any errors and return a placeholder
            logger.exception(f"Error in batch scan: {str(e)}")
            
            # Create error content
            error_content = html.Div([
                html.H3("Batch Scan Error"),
                html.P(f"An error occurred during batch scan: {str(e)}"),
                html.Hr(),
                html.P("Please try again or check the logs for more information.")
            ])
            
            return error_content, dash.no_update, dash.no_update
    
    # Callback for filter buttons in scanner dashboard
    @app.callback(
        Output("scanner-dashboard-content", "children", allow_duplicate=True),
        Input("filter-all", "n_clicks"),
        Input("filter-buy", "n_clicks"),
        Input("filter-sell", "n_clicks"),
        Input("filter-high", "n_clicks"),
        State("scan-results-store", "data"),
        prevent_initial_call=True,
    )
    def filter_scanner_results(all_clicks, buy_clicks, sell_clicks, high_clicks, stored_results):
        """Apply filters to scanner dashboard"""
        # Skip if no stored results
        if not stored_results:
            return dash.no_update
            
        # Determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
            
        # Get the button ID that was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Create a dummy single result object for backward compatibility
        dummy_result = ScanResult.from_dict(stored_results)
        
        # Apply the appropriate filter
        if button_id == "filter-buy":
            filtered_results = [dummy_result] if 'buy' in dummy_result.signal.get('signal', '') else []
        elif button_id == "filter-sell":
            filtered_results = [dummy_result] if 'sell' in dummy_result.signal.get('signal', '') else []
        elif button_id == "filter-high":
            filtered_results = [dummy_result] if dummy_result.signal.get('confidence') == 'high' else []
        else:  # filter-all or any other case
            filtered_results = [dummy_result]
            
        # Create a new scanner dashboard with the filtered results
        return create_scanner_dashboard(filtered_results)
    
    # Pattern-matching callback for "View" buttons in the batch results
    @app.callback(
        Output("analysis-content", "children", allow_duplicate=True),
        Output("enhanced-entry-exit-content", "children", allow_duplicate=True),  # New output
        Output("cycle-visualization-content", "children", allow_duplicate=True),
        Output("fld-visualization-content", "children", allow_duplicate=True),
        Output("harmonic-visualization-content", "children", allow_duplicate=True),
        Output("main-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "batch-view-btn", "index": dash.ALL}, "n_clicks"),  # Using batch-view-btn ID type
        # Don't use batch-content as state, it's too complex and can cause issues
        prevent_initial_call=True,
    )
    def handle_batch_view_button(n_clicks_list):
        """Handle View button clicks in the batch results"""
        if not any(n_clicks_list):
            return [dash.no_update] * 6  # Updated for 6 outputs
            
        # Get the triggered button's index (symbol)
        ctx = dash.callback_context
        if not ctx.triggered:
            return [dash.no_update] * 6  # Updated for 6 outputs
            
        # Get the button ID that was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            # Parse the JSON string to get the button index (symbol)
            button_data = json.loads(button_id)
            symbol = button_data['index']
            
            # Need to analyze this symbol as batch view doesn't have stored data
            try:
                # Create scan parameters - force generate_chart=False to avoid matplotlib thread issues
                # Get lookback from input if available via UI store
                lookback_value = 1000  # Default lookback
                try:
                    lookback_element = app.layout.children[2].children[1].children[0].children[8]
                    if hasattr(lookback_element, 'value') and lookback_element.value:
                        lookback_value = int(lookback_element.value)
                except:
                    pass  # Use default if lookup fails
                
                params = ScanParameters(
                    symbol=symbol,
                    exchange="NSE",  # Default exchange
                    interval="daily",  # Default interval
                    lookback=lookback_value,  # Use user-defined value or default
                    num_cycles=3,
                    price_source='close',
                    generate_chart=False  # Important: don't generate chart to avoid thread issues
                )
                
                # Use the scanner to analyze this symbol with the application's config
                scanner_instance = FibCycleScanner(load_config(APP_CONFIG_PATH))
                result = scanner_instance.analyze_symbol(params)
                
                # Make sure data is attached to the result
                if getattr(result, 'data', None) is None:
                    from data.data_management import DataFetcher
                    data_fetcher = DataFetcher(load_config(APP_CONFIG_PATH))
                    result.data = data_fetcher.get_data(
                        symbol=symbol,
                        exchange="NSE",
                        interval="daily",
                        lookback=params.lookback
                    )
                
                # Create all visualizations for this symbol
                analysis_content = create_analysis_content(result)
                enhanced_entry_exit = create_enhanced_entry_exit_ui(result)  # Create enhanced component
                cycle_viz = create_cycle_visualization(result)
                fld_viz = create_fld_visualization(result)
                harmonic_viz = create_harmonic_visualization(result)
                
                # Return content and switch to Analysis tab
                return analysis_content, enhanced_entry_exit, cycle_viz, fld_viz, harmonic_viz, "tab-analysis"
                
            except Exception as e:
                logger.exception(f"Error loading batch data for symbol {symbol}: {str(e)}")
                error_content = html.Div([
                    html.H3(f"Error loading data for {symbol}"),
                    html.P(f"An error occurred: {str(e)}"),
                ])
                return error_content, error_content, error_content, error_content, error_content, "tab-analysis"
                
        except Exception as e:
            logger.exception(f"Error handling batch view button: {str(e)}")
            error_content = html.Div([
                html.H3("Error Viewing Symbol"),
                html.P(f"An error occurred: {str(e)}"),
            ])
            return error_content, error_content, error_content, error_content, error_content, "tab-analysis"
    
    # Pattern-matching callback for "View" buttons in the scanner dashboard
    @app.callback(
        Output("analysis-content", "children", allow_duplicate=True),
        Output("enhanced-entry-exit-content", "children", allow_duplicate=True),  # New output
        Output("cycle-visualization-content", "children", allow_duplicate=True),
        Output("fld-visualization-content", "children", allow_duplicate=True),
        Output("harmonic-visualization-content", "children", allow_duplicate=True),
        Output("main-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "scan-view-btn", "index": dash.ALL}, "n_clicks"),  # Changed to match new button ID type
        State("scan-results-store", "data"),
        prevent_initial_call=True,
    )
    def handle_view_button(n_clicks_list, stored_data):
        """Handle View button clicks in the scanner dashboard"""
        if not any(n_clicks_list) or not stored_data:
            return [dash.no_update] * 6  # Updated for 6 outputs
            
        # Get the triggered button's index (symbol)
        ctx = dash.callback_context
        if not ctx.triggered:
            return [dash.no_update] * 6  # Updated for 6 outputs
            
        # Get the button ID that was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            # Parse the JSON string to get the button index (symbol)
            button_data = json.loads(button_id)
            symbol = button_data['index']
            
            # Check if we have data for this symbol
            if symbol != stored_data.get('symbol'):
                # We need to load data for this symbol
                logger.info(f"View button clicked for {symbol}, but data is for {stored_data.get('symbol')}. Loading new data...")
                try:
                    # Create scan parameters - avoid thread issues
                    params = ScanParameters(
                        symbol=symbol,
                        exchange=stored_data.get('exchange', 'NSE'),
                        interval=stored_data.get('interval', 'daily'),
                        lookback=1000,
                        num_cycles=3,
                        price_source='close',
                        generate_chart=False  # Important: don't generate chart to avoid thread issues
                    )
                    
                    # Use the scanner to analyze this symbol with the application's config
                    scanner_instance = FibCycleScanner(load_config(APP_CONFIG_PATH))
                    result = scanner_instance.analyze_symbol(params)
                    
                    # Make sure data is attached to the result
                    if getattr(result, 'data', None) is None:
                        from data.data_management import DataFetcher
                        data_fetcher = DataFetcher(load_config(APP_CONFIG_PATH))
                        result.data = data_fetcher.get_data(
                            symbol=symbol,
                            exchange=stored_data.get('exchange', 'NSE'),
                            interval=stored_data.get('interval', 'daily'),
                            lookback=params.lookback
                        )
                    
                    # Convert to dict for UI
                    result_data = result.to_dict()
                    
                    # Create all visualizations for this symbol using the correct modules
                    analysis_content = create_analysis_content(result)
                    enhanced_entry_exit = create_enhanced_entry_exit_ui(result)  # Create enhanced component
                    cycle_viz = create_cycle_visualization(result)
                    fld_viz = create_fld_visualization(result)
                    harmonic_viz = create_harmonic_visualization(result)
                    
                    # Return content and switch to Analysis tab
                    return analysis_content, enhanced_entry_exit, cycle_viz, fld_viz, harmonic_viz, "tab-analysis"
                    
                except Exception as e:
                    logger.exception(f"Error loading data for symbol {symbol}: {str(e)}")
                    error_content = html.Div([
                        html.H3(f"Error loading data for {symbol}"),
                        html.P(f"An error occurred: {str(e)}"),
                    ])
                    return error_content, error_content, error_content, error_content, error_content, "tab-analysis"
            else:
                # We already have data for this symbol, create visualizations from stored data
                result = ScanResult.from_dict(stored_data)
                
                # Create all visualizations for this symbol
                analysis_content = create_analysis_content(result)
                enhanced_entry_exit = create_enhanced_entry_exit_ui(result)  # Create enhanced component
                cycle_viz = create_cycle_visualization(result)
                fld_viz = create_fld_visualization(result)
                harmonic_viz = create_harmonic_visualization(result)
                
                # Return content and switch to Analysis tab
                return analysis_content, enhanced_entry_exit, cycle_viz, fld_viz, harmonic_viz, "tab-analysis"
                
        except Exception as e:
            logger.exception(f"Error handling view button: {str(e)}")
            error_content = html.Div([
                html.H3("Error Viewing Symbol"),
                html.P(f"An error occurred: {str(e)}"),
            ])
            return error_content, error_content, error_content, error_content, error_content, "tab-analysis"
    
    # Callbacks for the Performance and Backtest buttons in scanner results
    @app.callback(
        Output("main-tabs", "active_tab", allow_duplicate=True),
        Output("trading-strategies-content", "children", allow_duplicate=True),
        Output("navigation-store", "data", allow_duplicate=True),
        Input({"type": "scan-perf-btn", "index": dash.ALL}, "n_clicks"),
        State("scan-results-store", "data"),
        prevent_initial_call=True,
    )
    def handle_performance_button(n_clicks_list, stored_data):
        """Handle Performance button clicks in the scanner dashboard"""
        if not any(n_clicks_list) or not stored_data:
            return [dash.no_update] * 3
            
        # Get the triggered button's index (symbol)
        ctx = dash.callback_context
        if not ctx.triggered:
            return [dash.no_update] * 3
            
        # Get the button ID that was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            # Parse the JSON string to get the button index (symbol)
            button_data = json.loads(button_id)
            symbol = button_data['index']
            
            # Check if we have data for this symbol
            if symbol != stored_data.get('symbol'):
                # We need to load data for this symbol
                logger.info(f"Performance button clicked for {symbol}, loading new data...")
                try:
                    # Create scan parameters - avoid thread issues
                    params = ScanParameters(
                        symbol=symbol,
                        exchange=stored_data.get('exchange', 'NSE'),
                        interval=stored_data.get('interval', 'daily'),
                        lookback=1000,
                        num_cycles=3,
                        price_source='close',
                        generate_chart=False  # Important: don't generate chart to avoid thread issues
                    )
                    
                    # Use the scanner to analyze this symbol with the application's config
                    scanner_instance = FibCycleScanner(load_config(APP_CONFIG_PATH))
                    result = scanner_instance.analyze_symbol(params)
                    
                    # Make sure data is attached to the result
                    if getattr(result, 'data', None) is None:
                        from data.data_management import DataFetcher
                        data_fetcher = DataFetcher(load_config(APP_CONFIG_PATH))
                        result.data = data_fetcher.get_data(
                            symbol=symbol,
                            exchange=stored_data.get('exchange', 'NSE'),
                            interval=stored_data.get('interval', 'daily'),
                            lookback=params.lookback
                        )
                    
                    # Set initial parameters for trading strategy view
                    initial_params = {
                        'symbol': symbol,
                        'exchange': stored_data.get('exchange', 'NSE'),
                        'interval': stored_data.get('interval', 'daily'),
                        'lookback': params.lookback if hasattr(params, 'lookback') else 1000,  # Pass lookback from params or default
                        'strategy': 'advanced_fibonacci',
                        'action': 'performance',  # Indicate we want to show performance view
                        'num_cycles': 3  # Add number of cycles for better analysis
                    }
                    
                    # Create strategy dashboard with initial parameters
                    strategy_dashboard = create_strategy_dashboard(result, initial_params)
                    
                    # Return content, switch to Strategies tab, and store navigation data
                    return "tab-strategies", strategy_dashboard, initial_params
                    
                except Exception as e:
                    logger.exception(f"Error loading data for symbol {symbol}: {str(e)}")
                    error_content = html.Div([
                        html.H3(f"Error loading data for {symbol}"),
                        html.P(f"An error occurred: {str(e)}"),
                    ])
                    return "tab-strategies", error_content, None
            else:
                # We already have data for this symbol
                result = ScanResult.from_dict(stored_data)
                
                # Set initial parameters for trading strategy view
                initial_params = {
                    'symbol': symbol,
                    'exchange': stored_data.get('exchange', 'NSE'),
                    'interval': stored_data.get('interval', 'daily'),
                    'lookback': stored_data.get('lookback', 1000),  # Get lookback from stored data or use default
                    'strategy': 'advanced_fibonacci',
                    'action': 'performance'  # Indicate we want to show performance view
                }
                
                # Create strategy dashboard with initial parameters
                strategy_dashboard = create_strategy_dashboard(result, initial_params)
                
                # Return content, switch to Strategies tab, and store navigation data
                return "tab-strategies", strategy_dashboard, initial_params
                
        except Exception as e:
            logger.exception(f"Error handling performance button: {str(e)}")
            error_content = html.Div([
                html.H3("Error Loading Performance View"),
                html.P(f"An error occurred: {str(e)}"),
            ])
            return "tab-strategies", error_content, None
    
    # Callback for batch advanced strategy buttons
    @app.callback(
        Output("batch-advanced-content", "children"),
        Output("main-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "batch-advanced-btn", "index": dash.ALL}, "n_clicks"),
        State("scan-results-store", "data"),
        prevent_initial_call=True,
    )
    def handle_batch_advanced_button(n_clicks_list, scan_results_data):
        """Handle clicks on the Advanced button in batch results."""
        if not any(n_clicks_list):
            return dash.no_update, dash.no_update
            
        # Process all results from batch scan
        try:
            # Get all batch results from storage
            from storage.results_repository import ResultsRepository
            repo = ResultsRepository(load_config(APP_CONFIG_PATH))
            
            # Try to get the latest batch results
            latest_batch = repo.get_latest_batch_results()
            
            if latest_batch:
                logger.info(f"Found {len(latest_batch)} results in latest batch")
                # Create the batch advanced signals dashboard with explicit app parameter
                logger.info(f"Creating batch advanced signals for {len(latest_batch)} results with app instance")
                batch_signals = create_batch_advanced_signals(latest_batch, app=app)
                return batch_signals, "tab-batch-advanced"
            else:
                # If no batch results found, show error
                error_content = html.Div([
                    html.H3("No Batch Results Found"),
                    html.P("Please run a batch scan first to see advanced strategy signals."),
                ])
                return error_content, "tab-batch-advanced"
        except Exception as e:
            logger.exception(f"Error processing batch advanced button: {e}")
            error_content = html.Div([
                html.H3("Error Processing Batch"),
                html.P(f"An error occurred: {str(e)}"),
            ])
            return error_content, "tab-batch-advanced"
    
    # Callback for batch advanced tab
    @app.callback(
        Output("batch-advanced-content", "children", allow_duplicate=True),
        Input("main-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def load_batch_advanced_content(active_tab):
        """Load batch advanced strategies content when tab is selected."""
        if active_tab != "tab-batch-advanced":
            return dash.no_update
            
        try:
            # Get all batch results from storage
            from storage.results_repository import ResultsRepository
            repo = ResultsRepository(load_config(APP_CONFIG_PATH))
            
            # Try to get the latest batch results
            latest_batch = repo.get_latest_batch_results()
            
            if latest_batch:
                logger.info(f"Found {len(latest_batch)} results in latest batch")
                # Create the batch advanced signals dashboard with explicit app parameter
                logger.info(f"Creating batch advanced signals for {len(latest_batch)} results with app instance")
                batch_signals = create_batch_advanced_signals(latest_batch, app=app)
                return batch_signals
            else:
                # If no batch results found, show placeholder
                return html.Div([
                    html.H3("No Batch Results Available"),
                    html.P("Please run a batch scan first to see advanced strategy signals."),
                    html.Hr(),
                    html.P("Then click the 'Advanced' button next to any symbol in the batch results table."),
                ])
        except Exception as e:
            logger.exception(f"Error loading batch advanced content: {e}")
            return html.Div([
                html.H3("Error Loading Batch Results"),
                html.P(f"An error occurred: {str(e)}"),
            ])
    
    # Callback for advanced strategies tab
    @app.callback(
        Output("advanced-strategies-content", "children", allow_duplicate=True),
        Input("main-tabs", "active_tab"),
        State("scan-results-store", "data"),
        prevent_initial_call=True,
    )
    def load_advanced_strategies_content(active_tab, scan_results_data):
        """Load advanced strategies module content"""
        if active_tab != "tab-advanced-strategies" or not scan_results_data:
            return dash.no_update
            
        try:
            # Remove any non-standard fields that might cause issues
            sanitized_data = {k: v for k, v in scan_results_data.items() 
                             if k not in ['has_data', 'has_chart']}
            
            # Convert dict back to ScanResult
            result = ScanResult.from_dict(sanitized_data)
            
            # Log advanced strategy generation
            logger.info(f"Generating advanced strategies for {result.symbol}")
            
            # Create advanced strategies dashboard with full result object and params
            params = {
                'symbol': result.symbol,
                'exchange': result.exchange,
                'interval': result.interval,
                'strategy': 'rapid_cycle_fld',
                'timestamp': datetime.now().isoformat()
            }
            advanced_dashboard = create_advanced_strategy_dashboard(result=result, initial_params=params)
            
            return advanced_dashboard
        except Exception as e:
            # Handle any errors
            logger.exception(f"Error loading advanced strategies: {str(e)}")
            
            # Create error content
            error_content = html.Div([
                html.H3("Advanced Strategies Error"),
                html.P(f"An error occurred loading advanced strategies: {str(e)}"),
                html.Hr(),
                html.P("Please try again or check the logs for more information.")
            ])
            
            return error_content
    
    # Callback for trading strategies module
    @app.callback(
        Output("trading-strategies-content", "children", allow_duplicate=True),
        Input("main-tabs", "active_tab"),
        State("scan-results-store", "data"),
        prevent_initial_call=True,
    )
    def load_trading_strategies(active_tab, scan_results_data):
        """Load trading strategies module content"""
        if active_tab != "tab-strategies" or not scan_results_data:
            return dash.no_update
        
        try:
            # Remove any non-standard fields that might cause issues
            sanitized_data = {k: v for k, v in scan_results_data.items() 
                             if k not in ['has_data', 'has_chart']}
            
            # Convert dict back to ScanResult
            result = ScanResult.from_dict(sanitized_data)
            
            # Log trading strategy generation
            logger.info(f"Generating trading strategies for {result.symbol}")
            
            # Create trading strategies dashboard
            strategy_dashboard = create_strategy_dashboard(result)
            
            return strategy_dashboard
        except Exception as e:
            # Handle any errors
            logger.exception(f"Error loading trading strategies: {str(e)}")
            
            # Create error content
            error_content = html.Div([
                html.H3("Trading Strategies Error"),
                html.P(f"An error occurred loading trading strategies: {str(e)}"),
                html.Hr(),
                html.P("Please try again or check the logs for more information.")
            ])
            
            return error_content


def create_analysis_content(result: ScanResult) -> html.Div:
    """
    Create analysis content for a single scan result.
    
    Args:
        result: ScanResult instance
        
    Returns:
        Dash Div component
    """
    if not result.success:
        return html.Div([
            html.H3(f"Error analyzing {result.symbol}"),
            html.P(result.error),
        ])
    
    # Build signal badge
    signal_color = "success" if "buy" in result.signal['signal'] else (
        "danger" if "sell" in result.signal['signal'] else "secondary"
    )
    
    # Confidence badge
    confidence_color = {
        "high": "success",
        "medium": "warning",
        "low": "danger"
    }.get(result.signal['confidence'], "secondary")
    
    # Format detected cycles
    cycles_text = ", ".join([f"{cycle} ({power:.2f})" 
                           for cycle, power in zip(result.detected_cycles, 
                                                  result.cycle_powers.values())])
    
    return html.Div([
        html.H3(f"Analysis Results: {result.symbol}"),
        
        # Price and date info with highlighted closing price
        html.Div([
            html.Strong("Price: "),
            html.Span(f"{result.price:.2f}"),
            html.Strong(" as of "),
            html.Span(result.timestamp.strftime("%Y-%m-%d %H:%M")),
            html.Div([
                html.Strong("Closing Price (from original scan): ", className="text-primary"),
                html.Span(f"{result.price:.2f}", className="fw-bold text-primary")
            ], className="mt-1"),
            html.Div([
                dbc.Badge("Original Analysis Price", 
                          color="secondary", className="mt-2 p-1 me-2"),
                dbc.Badge("Real-time Price Updates Available", 
                          color="info", className="mt-2 p-1")
            ], className="mt-1")
        ], className="mb-3"),
        
        # Signal information
        html.Div([
            html.H4("Signal"),
            dbc.Row([
                dbc.Col([
                    dbc.Badge(result.signal['signal'].replace("_", " ").upper(), 
                             color=signal_color, className="p-2"),
                ]),
                dbc.Col([
                    html.Strong("Confidence: "),
                    dbc.Badge(result.signal['confidence'].upper(), color=confidence_color),
                ]),
                dbc.Col([
                    html.Strong("Strength: "),
                    html.Span(f"{result.signal['strength']:.2f}"),
                ]),
                dbc.Col([
                    html.Strong("Alignment: "),
                    html.Span(f"{result.signal['alignment']:.2f}"),
                ]),
            ]),
        ], className="mb-3"),
        
        # Note about Enhanced Entry/Exit Strategy
        html.Div([
            dbc.Alert([
                html.Strong("Note: "), 
                "See the Enhanced Entry/Exit Strategy tab for detailed trade recommendations, position sizing, and cycle-optimized entry windows."
            ], color="info", className="mb-2")
        ], className="mb-3"),
        
        # Cycle information
        html.Div([
            html.H4("Detected Cycles"),
            html.P(cycles_text),
            
            html.H5("Cycle States"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Cycle"),
                    html.Th("State"),
                    html.Th("Days Since Cross"),
                    html.Th("Price/FLD Ratio"),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(f"{state['cycle_length']}"),
                        html.Td(
                            html.Span("Bullish", className="text-success") if state['is_bullish'] 
                            else html.Span("Bearish", className="text-danger")
                        ),
                        html.Td(f"{state['days_since_crossover']}" if state['days_since_crossover'] is not None else "N/A"),
                        html.Td(f"{state['price_to_fld_ratio']:.4f}"),
                    ])
                    for state in result.cycle_states
                ]),
            ], bordered=True, striped=True, hover=True, size="sm"),
        ], className="mb-3"),
        
        # Harmonic relationships
        html.Div([
            html.H4("Harmonic Relationships"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Cycles"),
                    html.Th("Ratio"),
                    html.Th("Harmonic"),
                    html.Th("Precision"),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(cycles_pair),
                        html.Td(f"{relation['ratio']:.3f}"),
                        html.Td(relation['harmonic']),
                        html.Td(f"{relation['precision']:.2f}%"),
                    ])
                    for cycles_pair, relation in result.harmonic_relationships.items()
                ]),
            ], bordered=True, striped=True, hover=True, size="sm"),
        ]),
    ])


def run_app(config_path: str = "config/config.json", debug: bool = True, port: int = 8050, host: str = "127.0.0.1"):
    """
    Run the Dash application.
    
    Args:
        config_path: Path to configuration file
        debug: Whether to run in debug mode
        port: Port to run the server on
        host: Host to run the server on
    """
    try:
        # Make sure config directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        app = create_app(config_path)
        print(f"Starting dashboard on http://{host}:{port}")
        
        # Handle different Dash versions
        if hasattr(app, 'run'):
            app.run(debug=debug, port=port, host=host)
        else:  # Older versions use run_server
            app.run_server(debug=debug, port=port, host=host)
    except Exception as e:
        logger.exception(f"Error starting application: {e}")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fibonacci Harmonic Trading System Dashboard")
    parser.add_argument("--config", type=str, default="config/config.json", 
                        help="Path to configuration file")
    parser.add_argument("--port", type=int, default=8050, 
                        help="Port to run the dashboard on (default: 8050)")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Host to run the dashboard on (default: 127.0.0.1)")
    
    args = parser.parse_args()
    
    # Check if config path exists, or use default
    config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}. Using default configuration.")
        config_path = "config/default_config.json"
        
        # If default config doesn't exist either, create it
        if not os.path.exists(config_path):
            print(f"Creating default config file at {config_path}")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump({
                    "general": {
                        "default_exchange": "NSE",
                        "default_source": "tradingview",
                        "symbols_file_path": "config/symbols.json"
                    },
                    "data": {
                        "cache_dir": "data/cache",
                        "cache_expiry": {
                            "1m": 1,
                            "5m": 1,
                            "15m": 1,
                            "30m": 1,
                            "1h": 7,
                            "4h": 7,
                            "daily": 30,
                            "weekly": 90,
                            "monthly": 90
                        }
                    },
                    "tradingview": {
                        "username": "",
                        "password": ""
                    },
                    "analysis": {
                        "min_period": 10,
                        "max_period": 250,
                        "fib_cycles": [21, 34, 55, 89, 144, 233],
                        "power_threshold": 0.2,
                        "cycle_tolerance": 0.15,
                        "detrend_method": "diff",
                        "window_function": "hanning",
                        "gap_threshold": 0.01,
                        "crossover_lookback": 5
                    },
                    "scanner": {
                        "default_symbols": ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"],
                        "default_exchange": "NSE",
                        "default_interval": "daily",
                        "default_lookback": 1000,
                        "price_source": "close",
                        "num_cycles": 3,
                        "filter_signal": null,
                        "min_confidence": null,
                        "min_alignment": 0.6,
                        "ranking_factor": "strength"
                    },
                    "web": {
                        "dashboard_host": args.host,
                        "dashboard_port": args.port,
                        "refresh_interval": 60
                    },
                    "performance": {
                        "max_workers": 5
                    }
                }, f, indent=2)
    
    # Run the app with the specified arguments
    run_app(config_path, debug=args.debug, port=args.port, host=args.host)