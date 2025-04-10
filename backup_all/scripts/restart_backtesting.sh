#!/bin/bash

# Special script to restart the dashboard with a focus on backtesting functionality
# This script will clean all caches and force a fresh restart

# Set colored output for better readability
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

echo -e "${BLUE}=== Restarting Fibonacci Cycle Trading System Dashboard - Backtesting Debug Mode ===${NC}"

# Kill any running instances
echo -e "${BLUE}Killing any running dashboard instances...${NC}"
pkill -f "python.*main_dashboard.py" || true
sleep 2

# Make sure we're in the project directory
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

# Clean all caches thoroughly
echo -e "${BLUE}Performing cache cleaning...${NC}"

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

# Clean Dash cache
rm -f ./.dash_session* 2>/dev/null || true
rm -rf ~/.dash_jupyter_hooks 2>/dev/null || true
rm -rf ~/.dash_cache 2>/dev/null || true
rm -rf ~/.cache/dash 2>/dev/null || true
rm -rf ~/.cache/flask-session 2>/dev/null || true

# Ensure the cache directory exists
mkdir -p "${PROJECT_ROOT}/data/cache"

# Generate mock data for common symbols to ensure backtesting works
echo -e "${BLUE}Preparing mock data for backtesting...${NC}"
python - << EOF
import os
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
try:
    from data.mock_data_generator import generate_mock_price_data, save_mock_data_to_cache
    
    # Generate data for common symbols
    symbols = ["AAPL", "MSFT", "NIFTY", "RELIANCE", "TCS", "BANKNIFTY", "INFY"]
    exchanges = ["NYSE", "NYSE", "NSE", "NSE", "NSE", "NSE", "NSE"]
    intervals = ["daily", "4h", "1h", "15m"]
    
    for sym, ex in zip(symbols, exchanges):
        for interval in intervals:
            try:
                cache_path = os.path.join("${PROJECT_ROOT}", "data", "cache", f"{ex}_{sym}_{interval}.csv") 
                if not os.path.exists(cache_path):
                    print(f"Generating mock data for {sym} on {ex} ({interval})")
                    data = generate_mock_price_data(sym, lookback=500, interval=interval)
                    save_mock_data_to_cache(sym, ex, interval, data)
                else:
                    print(f"Mock data already exists for {sym} on {ex} ({interval})")
            except Exception as e:
                print(f"Error generating mock data for {sym}: {e}")
except Exception as e:
    print(f"Error importing mock data generator: {e}")
EOF

# Run the test_backtest script to verify backtesting works
echo -e "${BLUE}Verifying backtesting functionality...${NC}"
python test_backtest.py > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Backtesting verification successful!${NC}"
else
    echo -e "${YELLOW}Backtesting verification had issues, but continuing anyway...${NC}"
    echo -e "${YELLOW}Run ./run_backtest_test.sh separately for more details.${NC}"
fi

# Set environment variables for debugging
export PYTHONVERBOSE=1
export DASH_DEBUG=1
export DEBUG=1

# Set PYTHONPATH to the project root
export PYTHONPATH="${PROJECT_ROOT}"
echo -e "${BLUE}PYTHONPATH set to: $PYTHONPATH${NC}"

# Create debug script to run dashboard with specific backtesting focus
cat > run_debug.py << 'EOF'
#!/usr/bin/env python
"""Debugging wrapper for main_dashboard.py with enhanced backtesting support"""

import os
import sys
import logging
import traceback
import argparse
import importlib.util
from datetime import datetime

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

# Helper function to load modules from file path
def load_module_from_path(module_name, file_path):
    """Import a module from file path."""
    try:
        if os.path.exists(file_path):
            logger.info(f"Loading {module_name} from {file_path}")
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                logger.error(f"Could not create spec for {module_name} from {file_path}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            logger.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading {module_name} from {file_path}: {str(e)}")
        return None

# Override standard exception hook to log unhandled exceptions
def exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_handler

try:
    # Ensure we have mock data generated for testing
    logger.info("Checking for mock data generation module...")
    
    # Try to load mock data generator
    mock_data_module = load_module_from_path(
        "mock_data_generator", 
        os.path.join(current_dir, "data", "mock_data_generator.py")
    )
    
    if mock_data_module:
        # Generate mock data for a test symbol
        logger.info("Generating test mock data for NYSE_AAPL_daily...")
        try:
            mock_data = mock_data_module.generate_mock_price_data("AAPL", lookback=500, interval="daily")
            mock_data_module.save_mock_data_to_cache("AAPL", "NYSE", "daily", mock_data)
            logger.info("Successfully generated test mock data")
        except Exception as me:
            logger.error(f"Error generating mock data: {str(me)}")
    
    # Check for backtesting modules
    logger.info("Loading backtesting framework...")
    backtesting_module = load_module_from_path(
        "backtesting_framework", 
        os.path.join(current_dir, "backtesting", "backtesting_framework.py")
    )
    
    if backtesting_module:
        logger.info("Successfully loaded backtesting framework")
        BacktestEngine = backtesting_module.BacktestEngine
        BacktestParameters = backtesting_module.BacktestParameters
        
        # Test if we can create a basic BacktestParameters object
        params = BacktestParameters(
            symbol="AAPL",
            exchange="NYSE",
            interval="daily",
            start_date=datetime.now(),
            lookback=100
        )
        logger.info(f"Successfully created BacktestParameters: {params.symbol}")
    else:
        logger.warning("Failed to load backtesting framework")
        
    # Load data management
    logger.info("Loading data management...")
    data_module = None
    
    # Try different possible locations
    possible_paths = [
        os.path.join(current_dir, "data", "data_management.py"),
        os.path.join(current_dir, "data", "fetcher.py")
    ]
    
    for path in possible_paths:
        module_name = os.path.basename(path).split('.')[0]
        data_module = load_module_from_path(module_name, path)
        if data_module and hasattr(data_module, 'DataFetcher'):
            logger.info(f"Successfully loaded DataFetcher from {path}")
            break
    
    if not data_module or not hasattr(data_module, 'DataFetcher'):
        logger.warning("Failed to load DataFetcher")
    
    # Load backtest UI components
    logger.info("Loading backtest UI components...")
    backtest_ui_module = load_module_from_path(
        "backtest_ui", 
        os.path.join(current_dir, "web", "backtest_ui.py")
    )
    
    if backtest_ui_module:
        logger.info("Successfully loaded backtest_ui module")
    else:
        logger.warning("Failed to load backtest_ui module")

    # Now import the dashboard
    logger.info("Importing main_dashboard...")
    main_dashboard_module = load_module_from_path(
        "main_dashboard",
        os.path.join(current_dir, "main_dashboard.py")
    )
    
    if main_dashboard_module:
        logger.info("Successfully loaded main_dashboard")
        logger.info(f"Running dashboard on port {args.port}...")
        if hasattr(main_dashboard_module, 'run_app'):
            main_dashboard_module.run_app(debug=True, port=args.port)
        else:
            logger.error("main_dashboard module does not have run_app function")
    else:
        logger.error("Failed to load main_dashboard module")
        # Try to import directly as fallback
        import main_dashboard
        main_dashboard.run_app(debug=True, port=args.port)
    
except Exception as e:
    logger.error(f"Error in backtesting debug launcher: {e}")
    logger.error(traceback.format_exc())
EOF

chmod +x run_debug.py

# Create logs directory
mkdir -p logs

# Run the debug script with a custom port
PORT=8050
echo -e "${GREEN}Starting dashboard in backtesting debug mode on port ${PORT}...${NC}"
echo -e "${YELLOW}Logs will be written to dashboard_debug.log${NC}"
echo -e "${YELLOW}Access the dashboard at: http://127.0.0.1:${PORT}${NC}"
echo -e "${YELLOW}Navigate to the 'Backtesting' tab to test backtesting functionality${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}"

# Run the debug script - make sure it actually runs
echo -e "${BLUE}Starting dashboard on port ${PORT}...${NC}"
python run_debug.py --port ${PORT} --log-level=DEBUG