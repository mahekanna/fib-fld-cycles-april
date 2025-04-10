#!/usr/bin/env python
"""Debugging wrapper for main_dashboard.py with enhanced backtesting support"""

import os
import sys
import logging
import traceback
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description='Run dashboard with backtesting debug mode')
parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
parser.add_argument('--log-level', default='DEBUG', help='Logging level')
args = parser.parse_args()

# Set up detailed logging
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dashboard_debug.log', mode='w')  # Overwrite previous log
    ]
)
logger = logging.getLogger("backtesting_debug")

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
logger.info(f"Added current directory to sys.path: {current_dir}")
logger.info(f"Python path: {sys.path}")

# Override standard exception hook to log unhandled exceptions
def exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_handler

try:
    # Check for backtesting modules first
    logger.info("Checking for backtesting modules...")
    try:
        import backtesting.backtesting_framework
        logger.info("Successfully imported backtesting framework")
    except ImportError as e:
        logger.error(f"Error importing backtesting framework: {e}")
        
    try:
        # Check for data fetcher - make sure it's from our bridge file first
        from data.fetcher import DataFetcher
        logger.info("Successfully imported DataFetcher from data.fetcher")
    except ImportError as e:
        logger.error(f"Error importing DataFetcher from fetcher: {e}")
        try:
            # Try importing from data_management as fallback
            from data.data_management import DataFetcher
            logger.info("Successfully imported DataFetcher from data_management")
        except ImportError as e2:
            logger.error(f"Error importing DataFetcher from data_management: {e2}")
    
    try:
        from web.backtest_ui import create_backtest_ui, create_backtest_results_ui
        logger.info("Successfully imported backtest_ui components")
    except ImportError as e:
        logger.error(f"Error importing backtest_ui: {e}")

    # Now import the dashboard
    logger.info("Importing main_dashboard...")
    import main_dashboard
    
    # Patch the dashboard to prioritize backtesting components if needed
    if hasattr(main_dashboard, 'app') and hasattr(main_dashboard, 'layout'):
        logger.info("Patching dashboard layout to ensure backtesting tab is visible")
        # Custom patching could go here if needed
    
    logger.info(f"Running dashboard on port {args.port}...")
    main_dashboard.run_app(debug=True, port=args.port)
    
except Exception as e:
    logger.error(f"Error in backtesting debug launcher: {e}")
    logger.error(traceback.format_exc())
