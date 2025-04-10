#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fibonacci Cycle Dashboard")
    
    parser.add_argument("-c", "--config", 
                        help="Path to configuration file", 
                        default="config/config.json")
    
    parser.add_argument("-p", "--port", 
                        help="Port to run the dashboard on",
                        type=int,
                        default=8050)
    
    parser.add_argument("-d", "--debug", 
                        help="Run in debug mode",
                        action="store_true")
    
    parser.add_argument("--host", 
                        help="Host to bind to",
                        default="127.0.0.1")
    
    return parser.parse_args()

def main():
    """Main function to run the dashboard."""
    args = parse_arguments()
    
    # Ensure path is set correctly
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Set environment variable to ensure imports work properly
    os.environ["PYTHONPATH"] = project_root
    
    # Print status
    logger.info(f"Starting dashboard on http://{args.host}:{args.port}")
    logger.info(f"Project root: {project_root}")
    
    # Import the main dashboard module at runtime to avoid import errors 
    try:
        from fib_cycles_system.web.scanner_dashboard import create_scanner_dashboard
        logger.info("Successfully imported create_scanner_dashboard")
        
        # Try to import main dashboard module
        main_dashboard = importlib.import_module("main_dashboard")
        logger.info("Successfully imported main_dashboard module")
        
        # Run the app from the imported module
        app = main_dashboard.create_app(args.config)
        app.run_server(debug=args.debug, host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()