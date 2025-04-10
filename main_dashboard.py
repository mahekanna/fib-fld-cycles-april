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
from web.state_manager import state_manager

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
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        assets_folder=assets_path,
        prevent_initial_callbacks=True
    )
    
    # Add realtime price stylesheet and custom styles for state management indicators
    custom_styles = f"""
    {get_realtime_price_stylesheet()}
    
    /* State Management Indicators */
    .state-snapshot-info {{
        padding: 5px 10px;
        border-radius: 4px;
        background-color: rgba(0,128,0,0.1);
        border: 1px solid #006400;
        margin-top: 10px;
        font-size: 12px;
    }}
    
    .state-snapshot-warning {{
        padding: 5px 10px;
        border-radius: 4px;
        background-color: rgba(255,165,0,0.1);
        border: 1px solid #FF8C00;
        margin-top: 10px;
        font-size: 12px;
    }}
    """
    
    app.index_string = app.index_string.replace(
        '</head>',
        f'<style>{custom_styles}</style></head>'
    )
    
    # Set app title
    app.title = "Fibonacci Harmonic Trading System"
    
    # Create real-time price components
    realtime_components = create_realtime_price_components()
    
    # Create interval component for scan progress updates
    scan_progress_interval = dcc.Interval(
        id="scan-progress-interval",
        interval=1000,  # Update every second
        n_intervals=0,
        disabled=True
    )
    
    # Register real-time price update callbacks
    register_realtime_price_callbacks(app)
    
    # Define app layout
    app.layout = html.Div([
        # Store component for shared data
        dcc.Store(id='scan-results-store'),
        dcc.Store(id='trading-data-store'),
        dcc.Store(id='navigation-store'),  # For navigation between components
        dcc.Store(id='scan-progress-store'),  # For scan progress tracking
        
        # Real-time price components
        realtime_components["price_store"],
        realtime_components["activation_store"],
        realtime_components["interval"],
        
        # Scan progress update interval
        scan_progress_interval,
        
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
                    dbc.Col(html.Div([
                        html.H2("Fibonacci Cycles Trading System", className="text-white"),
                        html.Span("🔄 UI UPDATED", style={"color": "#FFA500", "fontSize": "12px", "marginLeft": "10px"})
                    ])),
                    dbc.Col(
                        html.Div([
                            html.Div([
                                html.Span("🔒", id="consistency-mode-icon", 
                                         style={"fontSize": "20px", "marginRight": "5px"}),
                                html.Span("Strict Consistency Mode", id="consistency-mode-label",
                                         style={"fontSize": "12px", "color": "#90EE90"})
                            ], style={"marginRight": "15px"}),
                            realtime_components["status"],
                            dbc.Button("Scan", id="scan-button", color="success", className="ms-auto"),
                        ], className="d-flex align-items-center"),
                        width={"size": 2},
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
                        
                        # Batch scan control buttons
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Batch Scan", id="batch-scan-button", color="warning", className="w-100"),
                                width=6
                            ),
                            dbc.Col(
                                dbc.Button("Pause", id="pause-scan-button", color="info", className="w-100", disabled=True),
                                width=3
                            ),
                            dbc.Col(
                                dbc.Button("Resume", id="resume-scan-button", color="success", className="w-100", disabled=True),
                                width=3
                            ),
                        ], className="mt-2"),
                        
                        # Scan progress indicators
                        html.Div([
                            html.Div([
                                html.Span("Scan Progress: ", className="fw-bold"),
                                html.Span("0%", id="scan-progress-percentage")
                            ], className="mt-2"),
                            dbc.Progress(
                                id="scan-progress-bar", 
                                value=0, 
                                className="mt-1",
                                style={"height": "10px"}
                            ),
                            html.Div([
                                html.Span("Status: ", className="fw-bold"),
                                html.Span("Not started", id="scan-status")
                            ], className="mt-1 small"),
                            html.Div([
                                html.Span("Symbols: ", className="fw-bold"),
                                html.Span("0/0", id="scan-symbols-count")
                            ], className="mt-1 small")
                        ], id="scan-progress-container", style={"display": "none"}),
                    ], className="p-3 bg-light rounded shadow")
                ], width=3),
                
                # Main content area - Improved navigation flow
                dbc.Col([
                    # Top-level tabs to separate Single Symbol and Batch modes
                    dbc.Tabs([
                        # =================== SINGLE SYMBOL ANALYSIS ===================
                        dbc.Tab(label="Single Symbol Analysis", tab_id="tab-single-analysis", children=[
                            html.Div([
                                # Welcome message when no scan has been performed
                                html.Div([
                                    html.H3("Welcome to Fibonacci Cycles Trading System", className="mb-3"),
                                    html.P("Please enter a symbol and click the 'Scan' button to begin analysis.", className="text-muted"),
                                    html.P("Single Symbol Analysis provides in-depth cycle detection and signal analysis for individual symbols.")
                                ], id="welcome-message", className="my-3"),
                                
                                # Results tabs - only show after scan is complete
                                html.Div([
                                    dbc.Tabs([
                                        dbc.Tab(label="Analysis Results", tab_id="single-tab-analysis", children=[
                                            html.Div(id="analysis-content", className="mt-3")
                                        ]),
                                        dbc.Tab(label="Advanced Strategies", tab_id="single-tab-advanced-strategies", children=[
                                            html.Div(id="advanced-strategies-content", className="mt-3")
                                        ]),
                                        dbc.Tab(label="Enhanced Entry/Exit", tab_id="single-tab-enhanced", children=[
                                            html.Div(id="enhanced-entry-exit-content", className="mt-3")
                                        ]),
                                        dbc.Tab(label="Cycle Visualization", tab_id="single-tab-cycles", children=[
                                            html.Div(id="cycle-visualization-content", className="mt-3")
                                        ]),
                                        dbc.Tab(label="FLD Analysis", tab_id="single-tab-fld", children=[
                                            html.Div(id="fld-visualization-content", className="mt-3")
                                        ]),
                                        dbc.Tab(label="Harmonic Patterns", tab_id="single-tab-harmonic", children=[
                                            html.Div(id="harmonic-visualization-content", className="mt-3")
                                        ]),
                                        dbc.Tab(label="Trading Strategies", tab_id="single-tab-strategies", children=[
                                            html.Div(id="trading-strategies-content", className="mt-3")
                                        ]),
                                    ], id="single-tabs", active_tab="single-tab-analysis"),
                                ], id="single-results-container", style={"display": "none"})
                            ])
                        ]),
                        
                        # =================== BATCH ANALYSIS ===================
                        dbc.Tab(label="Batch Analysis", tab_id="tab-batch-analysis", children=[
                            html.Div([
                                # Welcome message for batch mode
                                html.Div([
                                    html.H3("Batch Analysis Mode", className="mb-3"),
                                    html.P("Enter symbols in the text area and click 'Batch Scan' to analyze multiple symbols.", className="text-muted"),
                                    html.P("Batch Analysis allows you to scan multiple symbols at once and view consolidated results.")
                                ], id="batch-welcome-message", className="my-3"),
                                
                                # Batch results tabs - only show after batch scan
                                html.Div([
                                    dbc.Tabs([
                                        dbc.Tab(label="Batch Results", tab_id="batch-tab-results", children=[
                                            html.Div(id="batch-content", className="mt-3")
                                        ]),
                                        dbc.Tab(label="Batch Advanced Signals", tab_id="batch-tab-advanced", children=[
                                            html.Div(id="batch-advanced-content", className="mt-3")
                                        ]),
                                        dbc.Tab(label="Scanner Dashboard", tab_id="batch-tab-scanner", children=[
                                            html.Div(id="scanner-dashboard-content", className="mt-3")
                                        ]),
                                    ], id="batch-tabs", active_tab="batch-tab-results"),
                                ], id="batch-results-container", style={"display": "none"})
                            ])
                        ])
                    ], id="main-tabs", active_tab="tab-single-analysis"),
                ], width=9),
            ], className="mt-3"),
        ], fluid=True),
        
        # Detail view modal - made larger for comprehensive content
        dbc.Modal([
            dbc.ModalHeader(id="detail-modal-title", children="Detail Analysis"),
            dbc.ModalBody(id="detail-modal-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-detail-modal", className="ms-auto")
            )
        ], id="detail-view-modal", size="xl", style={"maxWidth": "90%", "margin": "10px auto"}),
        
        # UI Update notification modal - shown on first load
        dbc.Modal([
            dbc.ModalHeader("UI Update Notification"),
            dbc.ModalBody([
                html.H4("Dashboard UI Has Been Updated", className="text-success"),
                html.P("The dashboard now has an improved navigation structure:"),
                html.Ul([
                    html.Li("Separated Single Symbol and Batch Analysis modes"),
                    html.Li("Improved navigation flow between components"),
                    html.Li("Better organization of tabs and components"),
                    html.Li("Modal popups for detail views"),
                    html.Li("Price consistency with state management")
                ]),
                html.Hr(),
                html.P([
                    "The UI now matches the simplified demo."
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Got it!", id="close-notification-button", className="ms-auto")
            )
        ], id="notification-modal", is_open=True),
    ])
    
    # Register callbacks
    register_callbacks(app, scanner, repository)
    
    return app


# GLOBAL BATCH RESULT STORAGE - Will hold the actual result objects to ensure we use the EXACT SAME objects
# across dashboard components without re-fetching data or recreating objects
# This MUST be outside the callback registration function to persist across the entire application
_batch_result_objects = []

# Flag to track if we should use strict mode (completely disable fetching fresh data)
_strict_consistency_mode = True

def register_callbacks(app, scanner, repository):
    """
    Register all app callbacks.
    
    Args:
        app: Dash application instance
        scanner: FibCycleScanner instance
        repository: ResultsRepository instance
    """
    # Modal notification close callback
    @app.callback(
        Output("notification-modal", "is_open"),
        Input("close-notification-button", "n_clicks"),
        prevent_initial_call=True
    )
    def close_notification(n_clicks):
        return False

    # Add Detail Modal callback
    @app.callback(
        Output("detail-view-modal", "is_open"),
        Output("detail-modal-title", "children"),
        Output("detail-modal-body", "children"),
        Input({"type": "simple-detail-btn", "index": dash.ALL}, "n_clicks"),
        Input("close-detail-modal", "n_clicks"),
        State("detail-view-modal", "is_open"),
        State("batch-results-persistent-store", "data"),
        prevent_initial_call=True
    )
    def toggle_detail_modal(detail_clicks, close_clicks, is_open, batch_data):
        """Toggle modal and update content based on which symbol was clicked"""
        # Get callback context to determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open, "Detail Analysis", dash.no_update
            
        # Get the button ID that was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Check if close button was clicked
        if button_id == "close-detail-modal":
            return False, "Detail Analysis", dash.no_update
        
        # Handle detail button click - parse which symbol was selected
        try:
            button_data = json.loads(button_id)
            symbol = button_data['index']
            
            # Get batch data from state manager
            batch_id = batch_data.get("batch_id") if batch_data else None
            
            if batch_id:
                results = state_manager.get_batch_results(batch_id)
                # Find the selected symbol in results
                result = next((r for r in results if r.symbol == symbol), None)
                
                if result:
                    # Create comprehensive detail view with tabs for all analysis types
                    detail_content = html.Div([
                        # Tabs for all analysis types
                        dbc.Tabs([
                            dbc.Tab(label="Analysis Results", tab_id="detail-tab-analysis", children=[
                                html.Div(create_analysis_content(result), className="mt-3")
                            ]),
                            dbc.Tab(label="Advanced Strategies", tab_id="detail-tab-advanced", children=[
                                html.Div(create_advanced_strategy_dashboard(result=result), className="mt-3")
                            ]),
                            dbc.Tab(label="Enhanced Entry/Exit", tab_id="detail-tab-enhanced", children=[
                                html.Div(create_enhanced_entry_exit_ui(result), className="mt-3")
                            ]),
                            dbc.Tab(label="Cycle Visualization", tab_id="detail-tab-cycles", children=[
                                html.Div(create_cycle_visualization(result), className="mt-3")
                            ]),
                            dbc.Tab(label="FLD Analysis", tab_id="detail-tab-fld", children=[
                                html.Div(create_fld_visualization(result), className="mt-3")
                            ]),
                            dbc.Tab(label="Harmonic Patterns", tab_id="detail-tab-harmonic", children=[
                                html.Div(create_harmonic_visualization(result), className="mt-3")
                            ]),
                            dbc.Tab(label="Trading Strategies", tab_id="detail-tab-strategies", children=[
                                html.Div(create_strategy_dashboard(result), className="mt-3")
                            ]),
                        ], id="detail-tabs", active_tab="detail-tab-analysis"),
                    ])
                    return True, f"Comprehensive Analysis: {symbol}", detail_content
            
            # Fallback for if we don't find the result
            return True, f"Detail Analysis: {symbol}", html.Div([
                html.H3(f"Details for {symbol}"),
                html.P("Could not find detailed analysis for this symbol."),
            ])
            
        except Exception as e:
            logger.error(f"Error showing detail view: {e}")
            return True, "Error", html.Div([
                html.H3("Error"),
                html.P(f"An error occurred: {str(e)}")
            ])
    
    # Add Store component for batch results - directly add to layout
    if 'batch-results-persistent-store' not in [comp.id for comp in app.layout.children if hasattr(comp, 'id')]:
        app.layout.children.append(
            dcc.Store(id="batch-results-persistent-store", storage_type="memory")
        )

    # Callback for main scan functionality
    @app.callback(
        # Use outputs with allow_duplicate for shared outputs
        Output("scan-results-store", "data"),
        Output("analysis-content", "children", allow_duplicate=True),
        Output("enhanced-entry-exit-content", "children", allow_duplicate=True),
        Output("cycle-visualization-content", "children", allow_duplicate=True),
        Output("fld-visualization-content", "children", allow_duplicate=True),
        Output("harmonic-visualization-content", "children", allow_duplicate=True),
        Output("advanced-strategies-content", "children", allow_duplicate=True),
        Output("trading-strategies-content", "children", allow_duplicate=True),
        Output("welcome-message", "style"),                         # New: Hide welcome message
        Output("single-results-container", "style"),                # New: Show results container
        Output("single-tabs", "active_tab"),                        # New: Switch to analysis tab
        Output("main-tabs", "active_tab", allow_duplicate=True),    # New: Make sure we're on single analysis tab
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
            return [dash.no_update] * 12

        if not symbol:
            return [dash.no_update] * 12

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

            # Add trading strategies content
            trading_strategies = create_strategy_dashboard(result)

            # Return all components AND update the UI flow to show results and hide welcome message
            return (
                result_data, 
                analysis_content,
                enhanced_entry_exit,
                cycle_visualization, 
                fld_visualization, 
                harmonic_visualization, 
                advanced_strategies,
                trading_strategies,
                {"display": "none"},                  # Hide welcome message
                {"display": "block"},                 # Show results container
                "single-tab-analysis",                # Activate analysis tab
                "tab-single-analysis"                 # Make sure we're on single analysis tab
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
                {},             # Empty store
                error_content,  # Analysis tab
                error_content,  # Enhanced entry/exit tab
                error_content,  # Cycle visualization tab
                error_content,  # FLD visualization tab
                error_content,  # Harmonic visualization tab
                error_content,  # Advanced strategies tab
                error_content,  # Trading strategies tab
                {"display": "none"},    # Hide welcome message
                {"display": "block"},   # Show results container with error
                "single-tab-analysis",  # Activate analysis tab
                "tab-single-analysis"   # Make sure we're on single analysis tab
            )

    # Callback for batch scan 
    @app.callback(
        Output("batch-content", "children"),
        Output("scanner-dashboard-content", "children"),          # Add scanner dashboard content
        Output("symbols-textarea", "value", allow_duplicate=True),
        Output("batch-advanced-content", "children", allow_duplicate=True),  # Add batch advanced signals
        Output("batch-results-persistent-store", "data", allow_duplicate=True),  # Store results for persistence
        Output("batch-welcome-message", "style"),                 # New: Hide welcome message
        Output("batch-results-container", "style"),              # New: Show results container
        Output("batch-tabs", "active_tab"),                      # New: Switch to results tab
        Output("main-tabs", "active_tab", allow_duplicate=True), # New: Make sure we're on batch tab
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
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

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
                        ]), html.Div(), dash.no_update, dash.no_update, dash.no_update, {"display": "none"}, {"display": "block"}, "batch-tab-results", "tab-batch-analysis"
                else:
                    logger.error(f"Symbols CSV file not found: {symbols_csv_path}")
                    return html.Div([
                        html.H3("Batch Scan Error"),
                        html.P("Symbols CSV file not found. Please provide symbols in the text area."),
                        html.Hr(),
                        html.P(f"Expected path: {symbols_csv_path}")
                    ]), html.Div(), dash.no_update, dash.no_update, dash.no_update, {"display": "none"}, {"display": "block"}, "batch-tab-results", "tab-batch-analysis"

            if not symbols:
                logger.warning("No symbols to scan")
                return html.Div([
                    html.H3("Batch Scan Error"),
                    html.P("No symbols to scan. Please provide symbols in the text area or create a symbols.csv file."),
                ]), html.Div(), dash.no_update, dash.no_update, dash.no_update, {"display": "none"}, {"display": "block"}, "batch-tab-results", "tab-batch-analysis"

            # INCREMENTAL SCANNING SUPPORT
            # Initialize scan state in the state manager
            scan_id = state_manager.initialize_scan_state(symbols, exchange, interval)
            logger.info(f"🔄 Initialized incremental batch scan {scan_id} with {len(symbols)} symbols")

            # Process in batches for efficiency and to support very large symbol lists
            BATCH_SIZE = 10  # Process 10 symbols at a time - adjust based on system performance
            all_results = []

            # Get all remaining symbols (for new scan this is all symbols)
            remaining_symbols = state_manager.get_remaining_symbols()

            for i in range(0, len(remaining_symbols), BATCH_SIZE):
                # Get the current batch of symbols
                batch_symbols = remaining_symbols[i:i+BATCH_SIZE]

                # Create scan parameters for this batch
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
                    for symbol in batch_symbols
                ]

                # Log batch scan progress
                logger.info(f"🔄 Processing batch {i//BATCH_SIZE + 1}/{(len(remaining_symbols) + BATCH_SIZE - 1)//BATCH_SIZE}: {batch_symbols}")

                # Perform scan for this batch
                batch_results = scanner.scan_batch(params_list)

                # Update scan state for each symbol
                for result in batch_results:
                    state_manager.update_scan_progress(
                        result.symbol, 
                        success=result.success, 
                        error=result.error if hasattr(result, 'error') else None
                    )

                # Add to all results
                all_results.extend(batch_results)

            # Get the final scan state
            scan_state = state_manager.get_scan_state()

            # Log completion summary
            logger.info(f"🏁 Batch scan {scan_id} complete: {scan_state['completed_symbols']}/{scan_state['total_symbols']} symbols processed successfully")

            # Use the collected results
            results = all_results

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

            # Create batch results content - batch results table
            batch_content = html.Div([
                html.H4(f"Batch Scan Results ({len(results)} symbols)"),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Symbol"),
                        html.Th("Price"),
                        html.Th("Signal"),
                        html.Th("Strength"),
                        html.Th("Confidence"),
                        html.Th("Actions"),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(result.symbol),
                            html.Td(f"{result.price:.2f}"),
                            html.Td(
                                dbc.Badge(
                                    result.signal['signal'].replace("_", " ").upper(),
                                    color="success" if "buy" in result.signal['signal'] else (
                                        "danger" if "sell" in result.signal['signal'] else "secondary"
                                    ),
                                    className="p-2"
                                )
                            ),
                            html.Td(f"{result.signal['strength']:.2f}"),
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
                            html.Td(
                                dbc.Button(
                                    "Details", 
                                    color="primary", 
                                    size="sm",
                                    id={"type": "simple-detail-btn", "index": result.symbol},
                                    className="me-1"
                                )
                            ),
                        ])
                        for result in results if result.success
                    ]),
                ], bordered=True, hover=True, responsive=True)
            ])
                
            # Create scanner dashboard content separately
            scanner_dashboard_content = create_scanner_dashboard(results)

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
                        # CRITICAL FIX: Use force_download=False and use_cache=True to ensure consistency
                        result.data = data_fetcher.get_data(
                            symbol=result.symbol,
                            exchange=result.exchange,
                            interval=result.interval,
                            lookback=int(lookback),
                            force_download=False,  # Ensure consistent data with original analysis
                            use_cache=True         # Prioritize cached data
                        )
                        logger.info(f"Added data to result for {result.symbol} using cached data")
                    except Exception as e:
                        logger.error(f"Error adding data to result for {result.symbol}: {e}")

            # ARCHITECTURAL IMPROVEMENT: Register batch results with central state manager
            # This ensures price and signal consistency across all dashboard components
            batch_id = state_manager.register_batch_results(results)
            logger.warning(f"🔒 Registered batch {batch_id} with state manager containing {len(results)} results")

            # For backward compatibility, also store in global variable 
            global _batch_result_objects
            _batch_result_objects = results.copy()

            # Create batch advanced signals - batch will be retrieved from state manager
            logger.info(f"Creating batch advanced signals with app instance - USING STATE MANAGER (batch {batch_id})")
            batch_advanced_signals = create_batch_advanced_signals(results, app=app)

            # Create persistence marker with timestamp, count and batch ID
            persistence_marker = {
                "timestamp": datetime.now().isoformat(),
                "count": len(results),
                "symbols": [r.symbol for r in results if r.success],
                "batch_id": batch_id
            }

            # Return all components and update UI navigation flow
            return (
                batch_content,                  # Batch content
                scanner_dashboard_content,      # Scanner dashboard content
                symbols_display,                # Update symbols textarea 
                batch_advanced_signals,         # Batch advanced signals content
                persistence_marker,             # Store results for persistence
                {"display": "none"},            # Hide welcome message
                {"display": "block"},           # Show results container
                "batch-tab-results",            # Activate batch results tab
                "tab-batch-analysis"            # Make sure we're on batch analysis tab
            )

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

            return error_content, error_content, dash.no_update, dash.no_update, dash.no_update, {"display": "none"}, {"display": "block"}, "batch-tab-results", "tab-batch-analysis"

    # Scan progress update callbacks
    @app.callback(
        Output("scan-progress-interval", "disabled"),
        Output("batch-scan-button", "disabled"),
        Output("pause-scan-button", "disabled"),
        Output("resume-scan-button", "disabled"),
        Output("scan-progress-container", "style"),
        Input("batch-scan-button", "n_clicks"),
        Input("pause-scan-button", "n_clicks"),
        Input("resume-scan-button", "n_clicks"),
        prevent_initial_call=True
    )
    def handle_scan_control_buttons(start_clicks, pause_clicks, resume_clicks):
        """Handle the scan control buttons (start, pause, resume)"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return True, False, True, True, {"display": "none"}

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Get current scan state
        scan_state = state_manager.get_scan_state()
        current_status = scan_state["status"] if scan_state else None

        if button_id == "batch-scan-button" and start_clicks:
            # Starting a new scan
            return False, True, False, True, {"display": "block"}

        elif button_id == "pause-scan-button" and pause_clicks:
            # Pausing a scan
            state_manager.pause_scan()
            return True, False, True, False, {"display": "block"}

        elif button_id == "resume-scan-button" and resume_clicks:
            # Resuming a scan
            state_manager.resume_scan()
            return False, True, False, True, {"display": "block"}

        # Default - shouldn't reach here
        return True, False, True, True, {"display": "none"}

    # Update scan progress indicators
    @app.callback(
        Output("scan-progress-percentage", "children"),
        Output("scan-progress-bar", "value"),
        Output("scan-status", "children"),
        Output("scan-symbols-count", "children"),
        Output("scan-progress-store", "data"),
        Input("scan-progress-interval", "n_intervals"),
        prevent_initial_call=True
    )
    def update_scan_progress(n_intervals):
        """Update the scan progress indicators"""
        # Get current scan state
        scan_state = state_manager.get_scan_state()

        if not scan_state:
            return "0%", 0, "Not started", "0/0", {}

        # Calculate percentage
        percentage = scan_state.get("completion_percentage", 0)

        # Get status
        status = scan_state.get("status", "unknown").upper()

        # Get symbols counts
        completed = scan_state.get("completed_symbols", 0)
        total = scan_state.get("total_symbols", 0)
        symbols_text = f"{completed}/{total}"

        # Store scan info for other components
        scan_info = {
            "id": scan_state.get("id"),
            "status": status,
            "percentage": percentage,
            "completed": completed,
            "total": total,
            "updated_at": datetime.now().isoformat()
        }

        return f"{percentage}%", percentage, status, symbols_text, scan_info

    # Create interval component for state manager check
    app.layout.children.append(
        dcc.Interval(id="state-manager-check", interval=2000, n_intervals=0)
    )
    
    # Callback to display state manager status
    @app.callback(
        Output("consistency-mode-icon", "children"),
        Output("consistency-mode-label", "children"),
        Output("consistency-mode-label", "style"),
        Input("main-tabs", "active_tab"),
        Input("batch-results-persistent-store", "data"),
        Input("scan-progress-store", "data"),
        # Check frequently for changes
        Input("state-manager-check", "n_intervals"),
        prevent_initial_call=True
    )
    def update_consistency_status(active_tab, stored_data, scan_progress, n_intervals):
        """Update the consistency mode status indicator"""
        # Get info from state manager
        info = state_manager.get_info()

        # Check if a scan is in progress
        scan_state = state_manager.get_scan_state()
        if scan_state and scan_state.get("status") in ["initialized", "in_progress"]:
            # Show scan progress in the header
            percentage = scan_state.get("completion_percentage", 0)
            completed = scan_state.get("completed_symbols", 0)
            total = scan_state.get("total_symbols", 0)
            return (
                "⏳", 
                f"Scanning: {percentage}% complete ({completed}/{total} symbols)",
                {"fontSize": "12px", "color": "#1E90FF"}
            )
        elif scan_state and scan_state.get("status") == "paused":
            # Show paused scan in the header
            completed = scan_state.get("completed_symbols", 0)
            total = scan_state.get("total_symbols", 0)
            return (
                "⏸️", 
                f"Scan paused: {completed}/{total} symbols processed",
                {"fontSize": "12px", "color": "#FFD700"}
            )
        elif state_manager.strict_mode:
            # We're in strict consistency mode
            active_batch = state_manager.active_batch_id
            snapshot_count = len(state_manager.result_snapshots)

            if active_batch and snapshot_count > 0:
                # We have active snapshots
                return (
                    "🔒", 
                    f"Strict Mode: {snapshot_count} snapshots (Batch: {active_batch})",
                    {"fontSize": "12px", "color": "#90EE90"}
                )
            else:
                # Strict mode but no snapshots yet
                return (
                    "🔓", 
                    "Strict Mode: No snapshots yet",
                    {"fontSize": "12px", "color": "#FFD700"}
                )
        else:
            # Dynamic mode - real-time data enabled
            return (
                "🔄", 
                "Dynamic Mode: Real-time updates enabled",
                {"fontSize": "12px", "color": "#FF8C00"}
            )

    # Store batch results when they're first created
    @app.callback(
        Output("batch-results-persistent-store", "data", allow_duplicate=True),
        Input("batch-scan-button", "n_clicks"),
        Input("symbols-textarea", "value"),
        Input("exchange-input", "value"),
        Input("interval-dropdown", "value"),
        prevent_initial_call=True
    )
    def store_batch_scan_results(n_clicks, symbols_text, exchange, interval):
        """Store batch scan results for persistence across tabs"""
        # Only run when batch-scan-button is clicked
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'batch-scan-button.n_clicks' or not n_clicks:
            return dash.no_update

        try:
            # Get the latest batch results from repository
            from storage.results_repository import ResultsRepository
            repo = ResultsRepository(load_config(APP_CONFIG_PATH))
            latest_batch = repo.get_latest_batch_results()

            # Store results in the global variable for sharing across components
            global _batch_result_objects
            _batch_result_objects = latest_batch.copy()

            # Create a marker to indicate results are available
            return {"timestamp": datetime.now().isoformat(), "count": len(latest_batch) if latest_batch else 0}

        except Exception as e:
            logger.exception(f"Error storing batch results: {e}")
            return dash.no_update

    # Callback for batch advanced tab - USE THE STORED OBJECTS, DON'T RECREATE THEM
    @app.callback(
        Output("batch-advanced-content", "children", allow_duplicate=True),
        Input("main-tabs", "active_tab"),
        State("batch-results-persistent-store", "data"),
        prevent_initial_call=True,
    )
    def load_batch_advanced_content(active_tab, stored_marker):
        """Load batch advanced strategies content when tab is selected."""
        if active_tab != "tab-batch-advanced":
            return dash.no_update

        try:
            # ARCHITECTURAL IMPROVEMENT: Use the state manager to retrieve consistent snapshots
            # This ensures we use the EXACT SAME data across all components
            batch_id = None

            # If stored marker contains a batch ID, use it
            if stored_marker and "batch_id" in stored_marker:
                batch_id = stored_marker["batch_id"]
                logger.info(f"Using batch ID {batch_id} from persistent store")
            else:
                # Otherwise use the active batch from state manager
                batch_id = state_manager.active_batch_id
                logger.info(f"Using active batch ID {batch_id} from state manager")

            if batch_id:
                # Get batch results from state manager
                batch_results = state_manager.get_batch_results(batch_id)

                if batch_results:
                    logger.warning(f"🔒 STRICT CONSISTENCY MODE: Using {len(batch_results)} results from state manager batch {batch_id}")
                    for i, result in enumerate(batch_results[:3]):  # Log first 3 for debugging
                        if hasattr(result, 'price') and result.price:
                            logger.warning(f"STATE MANAGER RESULT [{i}]: {result.symbol} @ price {result.price}")

                    # Create batch advanced signals with state manager objects
                    batch_signals = create_batch_advanced_signals(batch_results, app=app)
                    return batch_signals
                else:
                    logger.warning(f"⚠️ No results found in state manager for batch {batch_id}")

            # Fallback to global variable for backward compatibility
            global _batch_result_objects
            if _batch_result_objects:
                logger.warning(f"⚠️ FALLBACK: Using {len(_batch_result_objects)} stored batch results from global variable")
                batch_signals = create_batch_advanced_signals(_batch_result_objects, app=app)
                return batch_signals

            # If all else fails, try repository
            elif stored_marker and stored_marker.get("count", 0) > 0:
                # Get all batch results from storage
                from storage.results_repository import ResultsRepository
                repo = ResultsRepository(load_config(APP_CONFIG_PATH))

                # Try to get the latest batch results
                latest_batch = repo.get_latest_batch_results()

                if latest_batch:
                    logger.info(f"⚠️ LAST RESORT: Found {len(latest_batch)} results in repository")
                    # Register with state manager and use
                    new_batch_id = state_manager.register_batch_results(latest_batch)
                    logger.warning(f"🔒 Registered repository results as new batch {new_batch_id}")

                    # Keep global variable updated
                    _batch_result_objects = latest_batch.copy()

                    # Create the batch advanced signals dashboard
                    batch_signals = create_batch_advanced_signals(latest_batch, app=app)
                    return batch_signals

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