"""
Extremely simple dashboard to demonstrate the UI concepts
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Initialize the app - use a light theme for better visibility
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Define the app layout with the new structure
app.layout = html.Div([
    # Header
    dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H2("Fibonacci Cycles System - New UI", className="text-primary")),
                dbc.Col(
                    html.Div("🔄 UI UPDATED", 
                             style={"color": "orange", "fontWeight": "bold", "textAlign": "right"}),
                    width=3
                )
            ])
        ]),
        color="light",
    ),
    
    # Main content
    dbc.Container([
        # Top-level tabs - Single vs Batch
        dbc.Tabs([
            # Single Symbol Tab
            dbc.Tab(label="Single Symbol Analysis", children=[
                dbc.Row([
                    # Sidebar
                    dbc.Col([
                        html.Div([
                            html.H4("Analysis Parameters", className="mt-3"),
                            html.Label("Symbol", className="mt-2"),
                            dbc.Input(id="symbol-input", value="NIFTY", type="text", className="mb-2"),
                            html.Label("Interval", className="mt-2"),
                            dbc.Select(
                                id="interval-dropdown",
                                options=[
                                    {"label": "Daily", "value": "daily"},
                                    {"label": "15 Minute", "value": "15m"},
                                ],
                                value="daily",
                                className="mb-2"
                            ),
                            dbc.Button("Scan", id="scan-button", color="primary", className="mt-3 w-100"),
                        ], className="p-3 bg-light rounded shadow")
                    ], width=3),
                    
                    # Main content
                    dbc.Col([
                        # Initial welcome message
                        html.Div([
                            html.H3("Welcome to the Fibonacci Cycles System", className="mb-3"),
                            html.P("Enter a symbol and click 'Scan' to begin analysis", className="text-muted"),
                            html.P("This demonstration shows the improved UI navigation flow", className="text-muted"),
                        ], id="welcome-message"),
                        
                        # Results content - initially hidden
                        html.Div([
                            # Sub-tabs for single symbol
                            dbc.Tabs([
                                dbc.Tab(label="Analysis Results", children=[
                                    html.H4("Analysis Results", className="mt-3"),
                                    html.P("Price and cycle analysis for the selected symbol", className="text-muted"),
                                ]),
                                dbc.Tab(label="Advanced Strategies", children=[
                                    html.H4("Advanced Strategies", className="mt-3"),
                                    html.P("Strategy signals and recommendations", className="text-muted"),
                                ]),
                                dbc.Tab(label="Visualization", children=[
                                    html.H4("Cycle Visualization", className="mt-3"),
                                    html.P("Visual representation of detected cycles", className="text-muted"),
                                ]),
                            ], id="single-tabs"),
                        ], id="results-container", style={"display": "none"}),
                    ], width=9),
                ]),
            ], id="tab-single"),
            
            # Batch Analysis Tab
            dbc.Tab(label="Batch Analysis", children=[
                dbc.Row([
                    # Sidebar
                    dbc.Col([
                        html.Div([
                            html.H4("Batch Parameters", className="mt-3"),
                            html.Label("Symbols (one per line)", className="mt-2"),
                            dbc.Textarea(
                                id="symbols-textarea",
                                placeholder="Enter symbols, one per line...\nNIFTY\nBANKNIFTY\nRELIANCE",
                                rows=5,
                                className="mb-2"
                            ),
                            dbc.Button("Batch Scan", id="batch-scan-button", color="warning", className="mt-3 w-100"),
                            
                            # Progress indicators (initially hidden)
                            html.Div([
                                html.H5("Scan Progress", className="mt-3"),
                                dbc.Progress(id="progress-bar", value=75, className="mb-2"),
                                html.Div("Processing symbols...", id="progress-text"),
                            ], id="progress-container", style={"display": "none"}),
                        ], className="p-3 bg-light rounded shadow")
                    ], width=3),
                    
                    # Main content
                    dbc.Col([
                        # Initial welcome message
                        html.Div([
                            html.H3("Batch Analysis Mode", className="mb-3"),
                            html.P("Enter multiple symbols and click 'Batch Scan'", className="text-muted"),
                            html.P("Process multiple symbols at once", className="text-muted"),
                        ], id="batch-welcome"),
                        
                        # Batch results content - initially hidden
                        html.Div([
                            # Sub-tabs for batch analysis
                            dbc.Tabs([
                                dbc.Tab(label="Batch Results", children=[
                                    html.H4("Batch Results", className="mt-3"),
                                    html.P("Consolidated results for all scanned symbols", className="text-muted"),
                                    
                                    # Sample batch results table
                                    dbc.Table([
                                        html.Thead(html.Tr([
                                            html.Th("Symbol"),
                                            html.Th("Price"),
                                            html.Th("Signal"),
                                            html.Th("Actions"),
                                        ])),
                                        html.Tbody([
                                            html.Tr([
                                                html.Td("NIFTY"),
                                                html.Td("19876.50"),
                                                html.Td("BUY"),
                                                html.Td(dbc.Button("Details", size="sm")),
                                            ]),
                                            html.Tr([
                                                html.Td("BANKNIFTY"),
                                                html.Td("42547.75"),
                                                html.Td("SELL"),
                                                html.Td(dbc.Button("Details", size="sm")),
                                            ]),
                                        ]),
                                    ], bordered=True, hover=True),
                                ]),
                                dbc.Tab(label="Batch Advanced Signals", children=[
                                    html.H4("Batch Advanced Signals", className="mt-3"),
                                    dbc.Alert([
                                        html.H5("🔒 Using Consistent Prices", className="alert-heading"),
                                        html.P("All signals computed with immutable price snapshots"),
                                    ], color="success", className="mt-3"),
                                    
                                    # Sample advanced signals table
                                    dbc.Table([
                                        html.Thead(html.Tr([
                                            html.Th("Symbol"),
                                            html.Th("Price"),
                                            html.Th("Standard"),
                                            html.Th("Consensus"),
                                            html.Th("Actions"),
                                        ])),
                                        html.Tbody([
                                            html.Tr([
                                                html.Td("NIFTY"),
                                                html.Td("19876.50"),
                                                html.Td("BUY"),
                                                html.Td("STRONG BUY"),
                                                html.Td(dbc.Button("Details", size="sm", id="detail-btn-demo")),
                                            ]),
                                            html.Tr([
                                                html.Td("BANKNIFTY"),
                                                html.Td("42547.75"),
                                                html.Td("SELL"),
                                                html.Td("STRONG SELL"),
                                                html.Td(dbc.Button("Details", size="sm")),
                                            ]),
                                        ]),
                                    ], bordered=True, hover=True),
                                ]),
                            ], id="batch-tabs"),
                        ], id="batch-results-container", style={"display": "none"}),
                    ], width=9),
                ]),
            ], id="tab-batch"),
        ], id="main-tabs"),
    ], fluid=True),
    
    # Modal for details - initially hidden
    dbc.Modal([
        dbc.ModalHeader("Detail View - NIFTY"),
        dbc.ModalBody([
            html.H5("Signal Analysis"),
            html.P("Price: 19876.50"),
            dbc.Alert("STRONG BUY Signal", color="success"),
            html.H5("Strategy Breakdown", className="mt-3"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Strategy"),
                    html.Th("Signal"),
                    html.Th("Strength"),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("Rapid Cycle FLD"),
                        html.Td("BUY"),
                        html.Td("0.87"),
                    ]),
                    html.Tr([
                        html.Td("Multi-Cycle Confluence"),
                        html.Td("STRONG BUY"),
                        html.Td("0.92"),
                    ]),
                ]),
            ], bordered=True, size="sm"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal-button", className="ms-auto")
        ),
    ], id="detail-modal", size="lg"),
    
    # Help notification - shows immediately
    dbc.Modal([
        dbc.ModalHeader("UI Demonstration"),
        dbc.ModalBody([
            html.H4("New UI Structure"),
            html.P("This simplified demo shows how the new UI structure works:"),
            html.Ul([
                html.Li("Separated Single Symbol and Batch Analysis modes"),
                html.Li("Scan buttons show relevant sub-tabs automatically"),
                html.Li("Detail view in modal overlay"),
                html.Li("Price consistency indicators"),
            ]),
            html.Hr(),
            html.H5("Try the following:"),
            html.Ol([
                html.Li("Click the 'Scan' button to see the Single Symbol results"),
                html.Li("Switch to 'Batch Analysis' tab"),
                html.Li("Click 'Batch Scan' to see batch results"),
                html.Li("Click any 'Details' button to see the modal view"),
            ]),
        ]),
        dbc.ModalFooter(
            dbc.Button("Got it!", id="close-help-button", className="ms-auto")
        ),
    ], id="help-modal", is_open=True),
])

# Callback for the scan button
@app.callback(
    [Output("welcome-message", "style"),
     Output("results-container", "style")],
    Input("scan-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_scan(n_clicks):
    # Hide welcome message, show results
    return {"display": "none"}, {"display": "block"}

# Callback for the batch scan button
@app.callback(
    [Output("batch-welcome", "style"),
     Output("batch-results-container", "style"),
     Output("progress-container", "style")],
    Input("batch-scan-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_batch_scan(n_clicks):
    # Hide welcome message, show results and progress
    return {"display": "none"}, {"display": "block"}, {"display": "block"}

# Modal toggle callbacks
@app.callback(
    Output("help-modal", "is_open"),
    Input("close-help-button", "n_clicks"),
    prevent_initial_call=True
)
def close_help(n_clicks):
    return False

@app.callback(
    Output("detail-modal", "is_open"),
    [Input("detail-btn-demo", "n_clicks"),
     Input("close-modal-button", "n_clicks")],
    [State("detail-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(open_clicks, close_clicks, is_open):
    # Toggle modal based on button clicks
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# Run the app
if __name__ == "__main__":
    try:
        app.run(debug=True, port=8051)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("If you see 'app.run_server has been replaced by app.run', try using app.run() instead.")