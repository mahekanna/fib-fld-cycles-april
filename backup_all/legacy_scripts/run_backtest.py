#!/usr/bin/env python3

import os
import sys
import argparse
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fibonacci Cycle Backtesting Tool")
    
    parser.add_argument("-c", "--config", 
                        help="Path to configuration file", 
                        default="config/config.json")
    
    parser.add_argument("-p", "--port", 
                        help="Port to run the dashboard on",
                        type=int,
                        default=8055)
    
    parser.add_argument("-d", "--debug", 
                        help="Run in debug mode",
                        action="store_true")
    
    parser.add_argument("-s", "--symbol", 
                        help="Initial symbol to load",
                        type=str,
                        default="")
    
    return parser.parse_args()


def create_app(config_path: str, symbol: str = ""):
    """
    Create a simple backtesting app with imports done inside the function
    to avoid package import issues.
    """
    # Import locally to avoid import errors
    import sys
    import os
    
    # Import dependencies  
    from utils.config import load_config
    from web.backtest_ui import create_backtest_ui, register_backtest_callbacks
    
    # Load configuration
    config = load_config(config_path)
    
    # Create Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True
    )
    
    # Set app title
    app.title = "Fibonacci Cycle Backtesting Tool"
    
    # Define app layout
    app.layout = html.Div([
        # Header
        dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H2("Fibonacci Cycle Backtesting Tool", className="text-white")),
                ]),
            ]),
            color="primary",
            dark=True,
        ),
        
        # Main content
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    create_backtest_ui()
                ]),
            ], className="mt-3"),
        ], fluid=True),
    ])
    
    # Register backtesting callbacks
    register_backtest_callbacks(app)
    
    return app


def main():
    """Main entry point for backtesting application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Add the current directory to Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Set PYTHONPATH environment variable
    os.environ['PYTHONPATH'] = os.path.abspath(os.path.dirname(__file__))
    
    try:
        # Create the app
        app = create_app(args.config, args.symbol)
        
        # Run the server
        logger.info(f"Starting backtesting dashboard on port {args.port}")
        
        if hasattr(app, 'run'):
            app.run(debug=args.debug, port=args.port, host="0.0.0.0")
        else:  # Older versions use run_server
            app.run_server(debug=args.debug, port=args.port, host="0.0.0.0")
    except Exception as e:
        logger.error(f"Error starting backtesting app: {str(e)}")
        logger.error("If you are having import issues, try running: export PYTHONPATH=$PYTHONPATH:$(pwd)")
        sys.exit(1)


if __name__ == "__main__":
    main()