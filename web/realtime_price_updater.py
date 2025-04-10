"""
Real-time Price Updater for Dashboard

This module provides components and callbacks for updating prices in real-time
in the dashboard interfaces.
"""

import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import logging
import time
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
import threading

# Import centralized logging system if available
try:
    from utils.logging_utils import get_component_logger
    logger = get_component_logger("web.realtime_price_updater")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import data refresher
from data.data_refresher import get_data_refresher


# Global state for real-time price updates
class RealTimePriceState:
    """Global state for real-time price updates"""
    def __init__(self):
        self.active_symbols = {}  # symbol -> {exchange, interval, last_price, last_update}
        self.update_callbacks = []  # List of callbacks to notify on updates
        self.subscribed_symbols = set()  # Set of "symbol_exchange_interval" being tracked
        self.refresh_interval = 5000  # Default refresh interval in ms
        self.started = False
        self.active_tab = None  # Track which tab is active

# Create singleton instance
price_state = RealTimePriceState()


def register_realtime_price_callbacks(app):
    """
    Register callbacks for real-time price updates.
    
    Args:
        app: Dash application instance
    """
    if not app:
        logger.error("No app provided, cannot register callbacks")
        return
    
    logger.info("Registering real-time price update callbacks")
    
    # Callback for the interval component to trigger price updates
    @app.callback(
        Output("realtime-price-store", "data"),
        Input("realtime-price-interval", "n_intervals"),
        State("realtime-price-store", "data"),
        prevent_initial_call=True
    )
    def update_prices(n_intervals, current_data):
        """Update prices for active symbols periodically"""
        if not price_state.active_symbols:
            return current_data or {}
        
        # Initialize data refresher
        try:
            from utils.config import load_config
            config_path = "config/config.json"
            config = load_config(config_path)
            refresher = get_data_refresher(config)
        except Exception as e:
            logger.error(f"Error initializing data refresher: {e}")
            return current_data or {}
        
        # Update each active symbol
        updated_data = current_data.copy() if current_data else {}
        
        for symbol, info in price_state.active_symbols.items():
            try:
                # Get latest data
                exchange = info.get('exchange', 'NSE')
                interval = info.get('interval', 'daily')
                
                # Only update if not updated recently (within 5 seconds)
                last_update = info.get('last_update', 0)
                if time.time() - last_update < 5:
                    continue
                
                # Get latest data
                data = refresher.get_latest_data(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    refresh_if_needed=True
                )
                
                if data is not None and not data.empty:
                    # Extract latest price
                    latest_price = data['close'].iloc[-1]
                    prev_price = info.get('last_price', 0)
                    
                    # Update state with real-time data, but mark it as a real-time price
                    # so it's clear this is different from the original analysis price
                    price_state.active_symbols[symbol]['last_price'] = latest_price
                    price_state.active_symbols[symbol]['last_update'] = time.time()
                    price_state.active_symbols[symbol]['timestamp'] = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    price_state.active_symbols[symbol]['change'] = latest_price - prev_price
                    price_state.active_symbols[symbol]['change_pct'] = (latest_price / prev_price - 1) * 100 if prev_price else 0
                    
                    # Update data with real-time flag
                    updated_data[symbol] = {
                        'price': latest_price,
                        'timestamp': data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                        'change': latest_price - prev_price,
                        'change_pct': (latest_price / prev_price - 1) * 100 if prev_price else 0,
                        'exchange': exchange,
                        'interval': interval,
                        'is_realtime': True  # Flag to indicate this is a real-time price, not the original
                    }
                    
                    logger.info(f"Updated price for {symbol}: {latest_price:.2f}")
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")
        
        return updated_data
    
    # Callback to handle symbol activation/deactivation
    @app.callback(
        Output("realtime-activation-store", "data"),
        Input("realtime-activate-btn", "n_clicks"),
        Input("realtime-deactivate-btn", "n_clicks"),
        State("symbol-input", "value"),
        State("exchange-input", "value"),
        State("interval-dropdown", "value"),
        State("realtime-activation-store", "data"),
        prevent_initial_call=True
    )
    def handle_symbol_activation(activate_clicks, deactivate_clicks, symbol, exchange, interval, current_activation):
        """Handle symbol activation/deactivation for real-time updates"""
        if not symbol:
            return current_activation or {}
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_activation or {}
        
        # Determine which button was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Initialize data refresher
        try:
            from utils.config import load_config
            config_path = "config/config.json"
            config = load_config(config_path)
            refresher = get_data_refresher(config)
        except Exception as e:
            logger.error(f"Error initializing data refresher: {e}")
            return current_activation or {}
        
        # Handle activation/deactivation
        current = current_activation.copy() if current_activation else {}
        
        if button_id == "realtime-activate-btn":
            # Activate symbol
            key = f"{symbol}_{exchange}_{interval}"
            
            # Subscribe to updates
            refresher.subscribe(symbol, exchange, interval)
            price_state.subscribed_symbols.add(key)
            
            # Add to active symbols
            price_state.active_symbols[symbol] = {
                'exchange': exchange,
                'interval': interval,
                'last_price': 0,
                'last_update': 0
            }
            
            # Add to activation store
            current[symbol] = {
                'active': True,
                'exchange': exchange,
                'interval': interval,
                'activated_at': datetime.now().isoformat()
            }
            
            # Make the symbol priority
            refresher.add_priority_symbol(symbol)
            
            # Force an immediate refresh
            refresher.refresh_symbol(symbol, exchange, interval)
            
            # Start the data refresher if not already running
            if not refresher.running:
                refresher.start_refresh_thread()
                
            logger.info(f"Activated real-time updates for {symbol} ({interval})")
            
        elif button_id == "realtime-deactivate-btn":
            # Deactivate symbol
            key = f"{symbol}_{exchange}_{interval}"
            
            # Unsubscribe from updates
            refresher.unsubscribe(symbol, exchange, interval)
            
            if key in price_state.subscribed_symbols:
                price_state.subscribed_symbols.remove(key)
            
            # Remove from active symbols
            if symbol in price_state.active_symbols:
                del price_state.active_symbols[symbol]
            
            # Remove from activation store
            if symbol in current:
                current[symbol] = {
                    'active': False,
                    'exchange': exchange,
                    'interval': interval,
                    'deactivated_at': datetime.now().isoformat()
                }
            
            # Remove priority
            refresher.remove_priority_symbol(symbol)
            
            logger.info(f"Deactivated real-time updates for {symbol} ({interval})")
        
        return current
    
    # Callback to update the price display component
    @app.callback(
        Output("realtime-price-display", "children"),
        Input("realtime-price-store", "data"),
        State("symbol-input", "value"),
        prevent_initial_call=True
    )
    def update_price_display(price_data, current_symbol):
        """Update the price display component with latest prices"""
        if not price_data or not current_symbol or current_symbol not in price_data:
            return html.Div("No real-time data available", className="text-muted")
        
        # Get price info for current symbol
        info = price_data[current_symbol]
        price = info.get('price', 0)
        timestamp = info.get('timestamp', 'N/A')
        change = info.get('change', 0)
        change_pct = info.get('change_pct', 0)
        
        # Determine color based on change
        color = "success" if change >= 0 else "danger"
        
        # Create price display - clearly indicating these are real-time prices separate from analysis
        return html.Div([
            html.H5([
                "Real-time Price: ",
                html.Span(f"₹{price:.2f}", className=f"text-{color}"),
                html.Small(" (separate from scan results)", className="ms-2 text-muted small")
            ]),
            html.Div([
                html.Span(f"{change:+.2f} ({change_pct:+.2f}%)", className=f"text-{color} me-2"),
                html.Small(f"Updated: {timestamp}", className="text-muted")
            ]),
            dbc.Row([
                dbc.Col(dbc.Badge("LIVE", color="success", className="mt-1"), width="auto"),
                dbc.Col(
                    html.Small("Note: This price is separate from the original scan results", 
                              className="text-muted fst-italic"), 
                    className="mt-1"
                )
            ])
        ], className="real-time-price-display")


def create_realtime_price_components():
    """
    Create components for real-time price updates.
    
    Returns:
        Dict of components for real-time price updates
    """
    components = {
        # Hidden components for state management
        "interval": dcc.Interval(
            id="realtime-price-interval",
            interval=price_state.refresh_interval,  # ms
            n_intervals=0,
            disabled=False
        ),
        "price_store": dcc.Store(
            id="realtime-price-store",
            storage_type="memory"
        ),
        "activation_store": dcc.Store(
            id="realtime-activation-store",
            storage_type="memory"
        ),
        
        # Visible components
        "controls": html.Div([
            dbc.ButtonGroup([
                dbc.Button(
                    "Activate Real-time",
                    id="realtime-activate-btn",
                    color="success",
                    size="sm",
                    className="me-1"
                ),
                dbc.Button(
                    "Deactivate",
                    id="realtime-deactivate-btn",
                    color="danger",
                    size="sm"
                )
            ], className="mb-2"),
            html.Div(
                id="realtime-price-display",
                className="mt-2"
            )
        ], className="realtime-price-controls"),
        
        # Status indicator
        "status": html.Div(
            id="realtime-status-indicator",
            className="realtime-status-indicator"
        )
    }
    
    return components


def get_realtime_price_stylesheet():
    """
    Get CSS styles for real-time price components.
    
    Returns:
        CSS styles as string
    """
    return """
    .real-time-price-display {
        padding: 8px;
        border-radius: 4px;
        background-color: rgba(0, 0, 0, 0.05);
        margin-top: 10px;
    }
    
    .realtime-status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #28a745;
        display: inline-block;
        margin-right: 5px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            opacity: 1;
        }
    }
    """