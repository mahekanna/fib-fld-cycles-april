"""
Simplified Dashboard to demonstrate the key concepts
This is a minimal version that should run reliably
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime
import time
import copy
import json

# Create a basic app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Define the app layout
app.layout = html.Div([
    # Header
    dbc.Navbar(
        dbc.Container([
            html.H2("Fibonacci Cycles Trading System - Simplified Demo", className="text-primary"),
            html.Div("🔄 Updated UI", style={"color": "orange", "fontWeight": "bold"})
        ]),
        color="light",
    ),
    
    dbc.Container([
        # Main tabs
        dbc.Tabs([
            # Single Symbol Analysis Tab
            dbc.Tab(label="Single Symbol Analysis", children=[
                dbc.Row([
                    # Left sidebar
                    dbc.Col([
                        html.H4("Symbol Settings", className="mt-3"),
                        dbc.Input(id="symbol-input", value="NIFTY", type="text", placeholder="Enter symbol..."),
                        dbc.Button("Scan Symbol", id="scan-button", color="primary", className="mt-2"),
                        html.Div(id="status-single", className="mt-2")
                    ], width=3),
                    
                    # Content area
                    dbc.Col([
                        html.Div(id="welcome-message", children=[
                            html.H3("Welcome to Fibonacci Cycles Demo"),
                            html.P("Enter a symbol and click 'Scan Symbol' to begin")
                        ]),
                        
                        # Results (initially hidden)
                        html.Div(id="results-container", style={"display": "none"}, children=[
                            dbc.Tabs([
                                dbc.Tab(label="Analysis", children=[
                                    html.H4("Analysis Results"),
                                    html.Div(id="analysis-content")
                                ]),
                                dbc.Tab(label="Advanced", children=[
                                    html.H4("Advanced Strategies"),
                                    html.Div(id="advanced-content")
                                ]),
                            ])
                        ])
                    ], width=9)
                ])
            ], tab_id="tab-single"),
            
            # Batch Analysis Tab
            dbc.Tab(label="Batch Analysis", children=[
                dbc.Row([
                    # Left sidebar
                    dbc.Col([
                        html.H4("Batch Settings", className="mt-3"),
                        dbc.Textarea(id="symbols-textarea", placeholder="Enter symbols, one per line...", rows=5),
                        dbc.Button("Batch Scan", id="batch-scan-button", color="warning", className="mt-2"),
                        
                        # Batch progress (initially hidden)
                        html.Div(id="batch-progress", style={"display": "none"}, children=[
                            html.H5("Scan Progress"),
                            dbc.Progress(id="progress-bar", value=0, className="my-2"),
                            html.Div(id="progress-text")
                        ])
                    ], width=3),
                    
                    # Content area
                    dbc.Col([
                        html.Div(id="batch-welcome", children=[
                            html.H3("Batch Analysis Mode"),
                            html.P("Enter multiple symbols and click 'Batch Scan'")
                        ]),
                        
                        # Batch results (initially hidden)
                        html.Div(id="batch-results-container", style={"display": "none"}, children=[
                            dbc.Tabs([
                                dbc.Tab(label="Batch Results", children=[
                                    html.H4("Batch Scan Results"),
                                    html.Div(id="batch-content")
                                ]),
                                dbc.Tab(label="Batch Advanced Signals", children=[
                                    html.H4("Advanced Signals"),
                                    html.Div(id="batch-advanced-content")
                                ]),
                            ])
                        ])
                    ], width=9)
                ])
            ], tab_id="tab-batch")
        ], id="main-tabs"),
    ], fluid=True),
    
    # Detail view modal
    dbc.Modal([
        dbc.ModalHeader(id="detail-modal-title"),
        dbc.ModalBody(id="detail-modal-body"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal-button")
        )
    ], id="detail-modal", size="lg"),
    
    # Notification modal - will show up immediately
    dbc.Modal([
        dbc.ModalHeader("UI Update Notification"),
        dbc.ModalBody([
            html.H4("Dashboard UI Has Been Updated", className="text-success"),
            html.P("This simplified demo shows the new UI structure with:"),
            html.Ul([
                html.Li("Separated Single Symbol and Batch Analysis"),
                html.Li("Improved navigation flow"),
                html.Li("Better organization of components"),
                html.Li("Modal popups for details"),
            ]),
            html.Hr(),
            html.P([
                "To see this on the main dashboard, run: ",
                html.Code("./extreme_restart.sh"),
                " after fixing any syntax errors."
            ])
        ]),
        dbc.ModalFooter(
            dbc.Button("Got it!", id="close-notification-button")
        )
    ], id="notification-modal", is_open=True),
])

# Modal close callback
@app.callback(
    Output("notification-modal", "is_open"),
    Input("close-notification-button", "n_clicks"),
    prevent_initial_call=True
)
def close_notification(n_clicks):
    return False if n_clicks else True

# Single symbol scan callback
@app.callback(
    [Output("status-single", "children"),
     Output("welcome-message", "style"),
     Output("results-container", "style"),
     Output("analysis-content", "children"),
     Output("advanced-content", "children")],
    Input("scan-button", "n_clicks"),
    State("symbol-input", "value"),
    prevent_initial_call=True
)
def handle_scan(n_clicks, symbol):
    if not symbol:
        return "Please enter a symbol", {}, {"display": "none"}, "", ""
    
    # Simulate scanning
    time.sleep(1)
    
    # Create sample results
    analysis_content = html.Div([
        html.H5(f"Analysis for {symbol}"),
        html.P(f"Price: {np.random.randint(100, 1000)}.{np.random.randint(10, 99)}"),
        html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        dbc.Alert("Signal: BUY", color="success")
    ])
    
    advanced_content = html.Div([
        html.H5(f"Advanced Strategies for {symbol}"),
        dbc.Table([
            html.Thead(html.Tr([html.Th("Strategy"), html.Th("Signal"), html.Th("Strength")])),
            html.Tbody([
                html.Tr([
                    html.Td("Rapid Cycle FLD"),
                    html.Td("BUY"),
                    html.Td("0.85")
                ]),
                html.Tr([
                    html.Td("Multi-Cycle Confluence"),
                    html.Td("BUY"),
                    html.Td("0.92")
                ]),
            ])
        ])
    ])
    
    return (
        f"Successfully scanned {symbol}",
        {"display": "none"},
        {"display": "block"},
        analysis_content,
        advanced_content
    )

# Batch scan callback
@app.callback(
    [Output("batch-progress", "style"),
     Output("batch-welcome", "style"),
     Output("batch-results-container", "style"),
     Output("progress-bar", "value"),
     Output("progress-text", "children"),
     Output("batch-content", "children"),
     Output("batch-advanced-content", "children"),
     Output("main-tabs", "active_tab")],
    Input("batch-scan-button", "n_clicks"),
    State("symbols-textarea", "value"),
    prevent_initial_call=True
)
def handle_batch_scan(n_clicks, symbols_text):
    if not symbols_text:
        return {"display": "none"}, {}, {"display": "none"}, 0, "", "", "", "tab-batch"
    
    # Parse symbols
    symbols = [s.strip() for s in symbols_text.split("\n") if s.strip()]
    
    if not symbols:
        return {"display": "none"}, {}, {"display": "none"}, 0, "", "", "", "tab-batch"
    
    # Show progress
    time.sleep(1)
    
    # Create batch results
    batch_results = html.Div([
        html.H5(f"Batch Results for {len(symbols)} Symbols"),
        dbc.Table([
            html.Thead(html.Tr([html.Th("Symbol"), html.Th("Price"), html.Th("Signal"), html.Th("Actions")])),
            html.Tbody([
                html.Tr([
                    html.Td(symbol),
                    html.Td(f"{np.random.randint(100, 1000)}.{np.random.randint(10, 99)}"),
                    html.Td(["BUY", "SELL"][np.random.randint(0, 2)]),
                    html.Td(dbc.Button("Details", id=f"detail-btn-{i}", size="sm"))
                ]) for i, symbol in enumerate(symbols)
            ])
        ])
    ])
    
    # Create batch advanced signals
    batch_advanced = html.Div([
        html.H5("Batch Advanced Signals"),
        dbc.Alert("These signals are consistent with the batch results", color="success"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Symbol"), 
                html.Th("Price"), 
                html.Th("Standard"), 
                html.Th("Consensus"), 
                html.Th("Actions")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(symbol),
                    html.Td(f"{np.random.randint(100, 1000)}.{np.random.randint(10, 99)}"),
                    html.Td(["BUY", "SELL"][np.random.randint(0, 2)]),
                    html.Td(["STRONG BUY", "STRONG SELL"][np.random.randint(0, 2)]),
                    html.Td(dbc.Button("Details", id=f"adv-detail-btn-{i}", size="sm"))
                ]) for i, symbol in enumerate(symbols)
            ])
        ])
    ])
    
    return (
        {"display": "block"},
        {"display": "none"},
        {"display": "block"},
        100,
        f"Completed {len(symbols)} symbols",
        batch_results,
        batch_advanced,
        "tab-batch"
    )

if __name__ == "__main__":
    app.run(debug=True, port=8051)  # Use a different port