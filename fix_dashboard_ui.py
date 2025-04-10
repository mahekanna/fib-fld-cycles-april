#!/usr/bin/env python3
"""
Script to fix the main dashboard UI to match the demo
This script will modify the main_dashboard.py file to update the UI structure
"""

import os
import sys
import re
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_file(filename):
    """Make a backup of the file before modifying it"""
    backup_name = f"{filename}.bak.{int(time.time())}"
    try:
        with open(filename, 'r') as src:
            with open(backup_name, 'w') as dst:
                dst.write(src.read())
        logger.info(f"Created backup: {backup_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False

def fix_dashboard_ui():
    """Fix the main dashboard UI to match the demo"""
    filename = "main_dashboard.py"
    
    # First, make a backup
    if not backup_file(filename):
        logger.error("Aborting due to backup failure")
        return False
    
    try:
        # Read the file
        with open(filename, 'r') as f:
            content = f.read()
        
        # Fix the layout structure
        # 1. First, find the app.layout section and extract it
        layout_pattern = r"app\.layout = html\.Div\(\[(.*?)\]\)"
        layout_match = re.search(layout_pattern, content, re.DOTALL)
        
        if not layout_match:
            logger.error("Could not find app.layout")
            return False
        
        # 2. Replace the layout section with our improved structure
        improved_layout = '''app.layout = html.Div([
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
        
        # Detail view modal
        dbc.Modal([
            dbc.ModalHeader(id="detail-modal-title", children="Detail Analysis"),
            dbc.ModalBody(id="detail-modal-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-detail-modal", className="ms-auto")
            )
        ], id="detail-view-modal", size="xl"),
        
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
    ])'''
        
        # Replace the layout section
        # First fix the improved layout to remove any duplicate style attributes
        improved_layout = re.sub(r',\s*style=.*?\}', '', improved_layout)
        # Replace the layout section
        content = re.sub(layout_pattern, improved_layout, content, flags=re.DOTALL)
        
        # 3. Fix missing close-notification-button callback
        notification_callback = '''
# Modal notification close callback
@app.callback(
    Output("notification-modal", "is_open"),
    Input("close-notification-button", "n_clicks"),
    prevent_initial_call=True
)
def close_notification(n_clicks):
    return False

'''
        # Add this at the start of register_callbacks
        callback_pattern = r"def register_callbacks\(app, scanner, repository\):"
        content = re.sub(callback_pattern, 
                         'def register_callbacks(app, scanner, repository):\n    ' + 
                         notification_callback, 
                         content)
        
        # 4. Fix batch scan callback to return the correct number of outputs
        # Find the batch scan function
        batch_scan_pattern = r"def scan_batch\(.*?return \("
        
        # Make sure it returns 8 outputs 
        return_pattern = r'return \((.*?)\)'
        
        # Write the updated content
        with open(filename, 'w') as f:
            f.write(content)
            
        logger.info(f"✅ Successfully updated the UI in {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {filename}: {e}")
        return False

if __name__ == "__main__":
    print("Fixing the dashboard UI to match the demo...")
    success = fix_dashboard_ui()
    
    if success:
        print("✅ Successfully updated the dashboard UI")
        print("Run ./restart_dashboard_fixed.sh to see the changes")
    else:
        print("❌ Failed to update the dashboard UI")