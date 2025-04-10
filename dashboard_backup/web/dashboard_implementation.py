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
from typing import Dict, List, Optional

from ..core.scanner import FibCycleScanner
from ..models.scan_parameters import ScanParameters
from ..models.scan_result import ScanResult
from ..utils.config import load_config
from ..storage.results_repository import ResultsRepository
from .advanced_strategies_ui import create_strategy_dashboard


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
    
    # Initialize scanner and repository
    scanner = FibCycleScanner(config)
    repository = ResultsRepository(config)
    
    # Create Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True
    )
    
    # Set app title
    app.title = "Fibonacci Harmonic Trading System"
    
    # Define app layout
    app.layout = html.Div([
        # Store component for shared data
        dcc.Store(id='scan-results-store'),
        
        # Header
        dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H2("Fibonacci Harmonic Trading System", className="text-white")),
                    dbc.Col(
                        dbc.Button("Scan", id="scan-button", color="success", className="ms-auto"),
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
                        html.H4("Analysis Parameters", className="mt-3"),
                        
                        html.Label("Symbol"),
                        dbc.Input(id="symbol-input", value="NIFTY", type="text"),
                        
                        html.Label("Exchange", className="mt-2"),
                        dbc.Input(id="exchange-input", value="NSE", type="text"),
                        
                        html.Label("Interval", className="mt-2"),
                        dcc.Dropdown(
                            id="interval-dropdown",
                            options=[
                                {"label": "Daily (Apr 5, 2024)", "value": "daily"},
                                {"label": "4 Hour (Mar 25, 2024)", "value": "4h"},
                                {"label": "1 Hour (Mar 25, 2024)", "value": "1h"},
                                {"label": "15 Minute (Mar 25, 2024)", "value": "15min"},
                            ],
                            value="daily",
                        ),
                        
                        html.Label("Lookback (bars)", className="mt-2"),
                        dbc.Input(id="lookback-input", value="1000", type="number"),
                        
                        html.Label("Number of Cycles", className="mt-2"),
                        dbc.Input(id="cycles-input", value="3", type="number", min=1, max=5),
                        
                        html.Label("Price Source", className="mt-2"),
                        dcc.Dropdown(
                            id="price-source-dropdown",
                            options=[
                                {"label": "Close", "value": "close"},
                                {"label": "HLC3", "value": "hlc3"},
                                {"label": "OHLC4", "value": "ohlc4"},
                                {"label": "HL2", "value": "hl2"},
                            ],
                            value="close",
                        ),
                        
                        html.Hr(),
                        
                        html.H4("Batch Scan"),
                        dbc.Textarea(
                            id="symbols-textarea",
                            placeholder="Enter symbols, one per line...",
                            rows=5,
                        ),
                        dbc.Button("Batch Scan", id="batch-scan-button", color="warning", className="mt-2"),
                    ], className="p-3 bg-light rounded")
                ], width=3),
                
                # Main content area - Tabs
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="Analysis Results", tab_id="tab-analysis", children=[
                            html.Div(id="analysis-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Batch Results", tab_id="tab-batch", children=[
                            html.Div(id="batch-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Cycle Chart", tab_id="tab-chart", children=[
                            html.Div(id="chart-content", className="mt-3")
                        ]),
                        dbc.Tab(label="Advanced Strategies", tab_id="tab-advanced-strategies", children=[
                            html.Div(id="advanced-strategies-content", className="mt-3")
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
    
    # Add callback for loading advanced strategies tab content
    @app.callback(
        Output("advanced-strategies-content", "children"),
        Input("main-tabs", "active_tab"),
        State("scan-results-store", "data"),
        prevent_initial_call=True,
    )
    def load_advanced_strategies_content(active_tab, results_data):
        """Load the advanced strategies content when tab is selected."""
        if active_tab != "tab-advanced-strategies":
            return dash.no_update
            
        # If we have scan results, pass them to the advanced strategies UI
        params = None
        if results_data:
            params = {
                "symbol": results_data.get("symbol"),
                "exchange": results_data.get("exchange"),
                "interval": results_data.get("interval"),
                "strategy": "rapid_cycle_fld",
                "timestamp": datetime.now().isoformat()
            }
            
        # Create and return the advanced strategies dashboard
        return create_strategy_dashboard(initial_params=params)
    
    @app.callback(
        Output("scan-results-store", "data"),
        Output("analysis-content", "children"),
        Output("chart-content", "children"),
        Input("scan-button", "n_clicks"),
        State("symbol-input", "value"),
        State("exchange-input", "value"),
        State("interval-dropdown", "value"),
        State("lookback-input", "value"),
        State("cycles-input", "value"),
        State("price-source-dropdown", "value"),
        prevent_initial_call=True,
    )
    def scan_symbol(n_clicks, symbol, exchange, interval, lookback, num_cycles, price_source):
        """Handle single symbol scan"""
        if not n_clicks or not symbol:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Log the interval we're using
        print(f"Scanning {symbol} with interval: {interval}")
        
        # Create scan parameters
        params = ScanParameters(
            symbol=symbol,
            exchange=exchange,
            interval=interval,  # The interval from dropdown
            lookback=int(lookback),
            num_cycles=int(num_cycles),
            price_source=price_source,
            generate_chart=True
        )
        
        # Clear cache for selected interval to force fresh data
        try:
            # Get cache_dir from config
            cache_dir = scanner.data_fetcher.cache_dir
            
            # Create a simple filename pattern to look for
            import os
            cache_pattern = f"{exchange}_{symbol}_{interval}"
            
            # Manually delete any matching cache files
            for filename in os.listdir(cache_dir):
                if cache_pattern in filename or (filename.endswith('.pkl') and symbol in filename):
                    try:
                        os.remove(os.path.join(cache_dir, filename))
                        print(f"Cleared cache file: {filename}")
                    except Exception as e:
                        print(f"Failed to clear cache file {filename}: {e}")
        except Exception as e:
            print(f"Cache clearing error: {e}")
        
        # Perform scan with force_download=True to ensure fresh data
        result = scanner.analyze_symbol(params)
        
        # Double-check that interval was correctly set in the result
        if hasattr(result, 'interval'):
            print(f"Result interval: {result.interval}")
        
        # Save result to repository
        repository.save_result(result)
        
        # Serialize for store
        result_data = result.to_dict()
        
        # Create analysis content
        analysis_content = create_analysis_content(result)
        
        # Create chart content
        chart_content = create_chart_content(result)
        
        return result_data, analysis_content, chart_content
    
    @app.callback(
        Output("batch-content", "children"),
        Input("batch-scan-button", "n_clicks"),
        State("symbols-textarea", "value"),
        State("exchange-input", "value"),
        State("interval-dropdown", "value"),
        State("lookback-input", "value"),
        State("cycles-input", "value"),
        State("price-source-dropdown", "value"),
        prevent_initial_call=True,
    )
    def scan_batch(n_clicks, symbols_text, exchange, interval, lookback, num_cycles, price_source):
        """Handle batch symbol scan"""
        if not n_clicks or not symbols_text:
            return dash.no_update
        
        # Log the interval we're using
        print(f"Batch scanning with interval: {interval}")
        
        # Parse symbols from text
        symbols = [s.strip() for s in symbols_text.split("\n") if s.strip()]
        
        # Clear cache for selected interval to force fresh data
        try:
            # Get cache_dir from config
            cache_dir = scanner.data_fetcher.cache_dir
            
            # Manually delete any matching cache files for interval
            import os
            for symbol in symbols:
                cache_pattern = f"{exchange}_{symbol}_{interval}"
                for filename in os.listdir(cache_dir):
                    if cache_pattern in filename or (filename.endswith('.pkl') and symbol in filename and interval in filename):
                        try:
                            os.remove(os.path.join(cache_dir, filename))
                            print(f"Cleared cache file: {filename}")
                        except Exception as e:
                            print(f"Failed to clear cache file {filename}: {e}")
        except Exception as e:
            print(f"Cache clearing error: {e}")
        
        # Create scan parameters for each symbol
        params_list = [
            ScanParameters(
                symbol=symbol,
                exchange=exchange,
                interval=interval,  # Use the selected interval
                lookback=int(lookback),
                num_cycles=int(num_cycles),
                price_source=price_source,
                generate_chart=False
            )
            for symbol in symbols
        ]
        
        # Perform batch scan
        results = scanner.scan_batch(params_list)
        
        # Save results to repository
        repository.save_batch_results(results)
        
        # Create batch results content
        batch_content = create_batch_results_table(results)
        
        return batch_content


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
        
        # Price and date info
        html.Div([
            html.Strong("Price: "),
            html.Span(f"{result.price:.2f}"),
            html.Strong(" as of "),
            html.Span(result.timestamp.strftime("%Y-%m-%d %H:%M")),
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
        
        # Position guidance
        html.Div([
            html.H4("Position Guidance"),
            dbc.Row([
                dbc.Col([
                    html.Strong("Entry: "),
                    html.Span(f"{result.position_guidance['entry_price']:.2f}"),
                ]),
                dbc.Col([
                    html.Strong("Stop Loss: "),
                    html.Span(f"{result.position_guidance['stop_loss']:.2f}"),
                ]),
                dbc.Col([
                    html.Strong("Target: "),
                    html.Span(f"{result.position_guidance['target_price']:.2f}"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Strong("Risk: "),
                    html.Span(f"{result.position_guidance['risk_percentage']:.2f}%"),
                ]),
                dbc.Col([
                    html.Strong("Reward: "),
                    html.Span(f"{result.position_guidance['target_percentage']:.2f}%"),
                ]),
                dbc.Col([
                    html.Strong("R/R Ratio: "),
                    html.Span(f"{result.position_guidance['risk_reward_ratio']:.2f}"),
                ]),
            ]),
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


def create_chart_content(result: ScanResult) -> html.Div:
    """
    Create chart visualization for a scan result.
    
    Args:
        result: ScanResult instance
        
    Returns:
        Dash Div component
    """
    if not result.success:
        return html.Div([
            html.H3("Chart not available"),
            html.P("The analysis did not complete successfully."),
        ])
    
    # Create an interactive Plotly chart
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{result.symbol} Price with FLDs", "Signal Strength")
    )
    
    # Create comprehensive chart with real data
    # Get the actual data from the scanner result
    data = result.data if hasattr(result, 'data') else None
    
    # If there's no data in the result, try to fetch it
    if data is None:
        try:
            from ..data.data_management import DataFetcher
            config = result.config if hasattr(result, 'config') else None
            
            # If no config in result, load from config path
            if config is None:
                from ..utils.config import load_config
                config = load_config()
                
            data_fetcher = DataFetcher(config)
            print(f"Fetching data for {result.symbol} with interval {result.interval}")
            
            # Use lookback from scan parameters or default to 1000 if not available
            lookback = getattr(result, 'lookback', 1000)
            print(f"Using lookback={lookback} bars for chart data")
            
            # Force download fresh data for the specified interval with proper lookback
            data = data_fetcher.get_data(
                symbol=result.symbol,
                exchange=result.exchange,
                interval=result.interval,
                lookback=lookback,
                force_download=True  # Force refresh to ensure we get data for the correct interval
            )
        except Exception as e:
            print(f"Error getting data for chart: {str(e)}")
    
    # If we have data, create a comprehensive visualization
    if data is not None and not data.empty:
        # Add price candlesticks - use all available data instead of limiting to last 250 bars
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add FLD lines for each detected cycle
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
                
        # Add cycle waves if available
        for cycle in result.detected_cycles:
            wave_col = f'cycle_wave_{cycle}'
            if wave_col in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[wave_col].dropna(),
                        mode='lines',
                        name=f"{cycle} Cycle",
                        line=dict(width=1.5, dash='dot')
                    ),
                    row=1, col=1
                )
                
        # Add signal strength indicator in bottom panel
        if hasattr(result, 'signal') and 'strength' in result.signal:
            # Create a simple time series of the signal strength
            # In a real implementation, you'd have historical signal strength
            strength = result.signal['strength']
            strength_color = 'green' if strength > 0 else 'red'
            
            fig.add_trace(
                go.Bar(
                    x=[data.index[-1]],
                    y=[abs(strength)],
                    name="Signal Strength",
                    marker_color=strength_color
                ),
                row=2, col=1
            )
            
            # Add a reference line for neutral (zero)
            fig.add_shape(
                type="line",
                x0=data.index[0], 
                x1=data.index[-1],
                y0=0, 
                y1=0,
                line=dict(color="white", width=1, dash="dash"),
                row=2, col=1
            )
    
    # If no data, add a message to the chart
    else:
        fig.add_annotation(
            x=0.5, 
            y=0.5,
            text="No data available for visualization",
            showarrow=False,
            font=dict(size=14, color="white"),
            row=1, col=1
        )
    
    # Add layout settings
    fig.update_layout(
        title=f"{result.symbol} Cycle Analysis",
        template="plotly_dark",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return html.Div([
        html.H3("Cycle Chart"),
        dcc.Graph(figure=fig, id="cycle-chart"),
    ])


def create_batch_results_table(results: List[ScanResult]) -> html.Div:
    """
    Create a table displaying batch scan results.
    
    Args:
        results: List of ScanResult instances
        
    Returns:
        Dash Div component
    """
    # Filter to successful results
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    # Create results tables
    return html.Div([
        html.H3(f"Batch Scan Results: {len(successful_results)} Successful, {len(failed_results)} Failed"),
        
        # Filter controls
        dbc.Row([
            dbc.Col([
                html.Label("Filter Signals:"),
                dcc.Dropdown(
                    id="signal-filter-dropdown",
                    options=[
                        {"label": "All Signals", "value": "all"},
                        {"label": "Buy Signals", "value": "buy"},
                        {"label": "Sell Signals", "value": "sell"},
                        {"label": "Strong Buy Signals", "value": "strong_buy"},
                        {"label": "Strong Sell Signals", "value": "strong_sell"},
                    ],
                    value="all",
                    clearable=False,
                ),
            ], width=4),
            dbc.Col([
                html.Label("Sort By:"),
                dcc.Dropdown(
                    id="sort-dropdown",
                    options=[
                        {"label": "Signal Strength", "value": "strength"},
                        {"label": "Cycle Alignment", "value": "alignment"},
                        {"label": "Risk/Reward Ratio", "value": "risk_reward"},
                    ],
                    value="strength",
                    clearable=False,
                ),
            ], width=4),
        ], className="mb-3"),
        
        # Results table
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Symbol"),
                html.Th("Price"),
                html.Th("Signal"),
                html.Th("Strength"),
                html.Th("Confidence"),
                html.Th("Alignment"),
                html.Th("R/R Ratio"),
                html.Th("Cycles"),
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
                            )
                        )
                    ),
                    html.Td(f"{result.signal['strength']:.2f}"),
                    html.Td(result.signal['confidence'].upper()),
                    html.Td(f"{result.signal['alignment']:.2f}"),
                    html.Td(f"{result.position_guidance['risk_reward_ratio']:.2f}"),
                    html.Td(", ".join(map(str, result.detected_cycles))),
                    html.Td([
                        dbc.Button(
                            "View", 
                            color="primary", 
                            size="sm", 
                            className="me-1",
                            id={"type": "batch-view-btn", "index": result.symbol}  # Different ID type for batch view
                        ),
                        dbc.Button(
                            "Advanced", 
                            color="success", 
                            size="sm",
                            id={"type": "batch-advanced-btn", "index": result.symbol}  # New button for advanced strategies
                        )
                    ]),
                ], className="buy-row" if "buy" in result.signal['signal'] else (
                    "sell-row" if "sell" in result.signal['signal'] else ""
                ))
                for result in successful_results
            ]),
        ], bordered=True, striped=True, hover=True),
        
        # Failed scans
        html.H4("Failed Scans", className="mt-4") if failed_results else None,
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Symbol"),
                html.Th("Error"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(result.symbol),
                    html.Td(result.error),
                ])
                for result in failed_results
            ]),
        ], bordered=True, striped=True, hover=True) if failed_results else None,
    ])


def run_app(config_path: str = "config/default_config.json", debug: bool = True, port: int = 8050):
    """
    Run the Dash application.
    
    Args:
        config_path: Path to configuration file
        debug: Whether to run in debug mode
        port: Port to run the server on
    """
    app = create_app(config_path)
    app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    run_app()
