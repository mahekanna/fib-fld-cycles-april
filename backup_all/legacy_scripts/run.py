#!/usr/bin/env python
"""
Fibonacci Cycle Trading System - Main Entry Point

This script serves as the main entry point for running the Fibonacci Cycle Trading System.
It provides a command-line interface for launching different components of the system,
with various options for customization.

Usage:
    python run.py [options]

Options:
    --mode           Mode to run (dashboard, scanner, backtest) [default: dashboard]
    --port           Port for the dashboard [default: 8050]
    --host           Host for the dashboard [default: 127.0.0.1]
    --symbols        Comma-separated list of symbols to analyze
    --exchange       Exchange to use for data [default: NSE]
    --interval       Timeframe interval [default: daily]
    --debug          Run in debug mode [default: False]
    --clean          Clean caches before running [default: False]
    --kill           Kill existing processes before running [default: True]
    --data-source    Data source to use (tvdatafeed, yfinance, mock) [default: auto]
"""

import os
import sys
import signal
import argparse
import subprocess
import importlib
import time
import json
import psutil
from datetime import datetime
from typing import List, Dict, Optional, Any

# Make sure the current directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our centralized logging system
try:
    from utils.logging_utils import (
        configure_logging, 
        get_component_logger, 
        log_manager, 
        log_exception,
        configure_root_logger
    )
    # Configure the root logger first, then get a module-specific logger
    configure_root_logger(level="INFO")
    logger = get_component_logger("run")
    logger.info("Using centralized logging system")
except ImportError:
    # Fallback logging if the logging_utils module is not available
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.warning("Using fallback logging - utils.logging_utils not found")

# Set default values
DEFAULT_PORT = 8050
DEFAULT_HOST = "127.0.0.1"

def color_text(text: str, color_code: str) -> str:
    """Apply ANSI color code to text."""
    return f"{color_code}{text}\033[0m"

# ANSI color codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
MAGENTA = "\033[0;35m"
CYAN = "\033[0;36m"

def print_banner():
    """Print a stylish banner for the application."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   █▀▀ █ █▄▄ █▀█ █▄░█ ▄▀█ █▀▀ █▀▀ █   █▀▀ █▄█ █▀▀ █░░ █▀▀ █▀     ║
║   █▀░ █ █▄█ █▄█ █░▀█ █▀█ █▄▄ █▄▄ █   █▄▄ ░█░ █▄▄ █▄▄ ██▄ ▄█     ║
║                                                                   ║
║                 Trading System v1.0.0 (April 2025)                ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(color_text(banner, BLUE))

def find_processes_by_name(name: str) -> List[psutil.Process]:
    """Find all processes matching a name pattern."""
    processes = []
    current_pid = os.getpid()
    my_process = psutil.Process(current_pid)
    my_script_path = os.path.abspath(__file__)
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Skip our own process
            if proc.pid == current_pid:
                continue
            
            # Get command line as string
            cmdline = ""
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline']).lower()
                
            # Skip if this is another instance of run.py with the same path
            if my_script_path.lower() in cmdline:
                continue
                
            # Check if the process name matches
            if name in proc.info['name'].lower():
                processes.append(proc)
                continue
                
            # Check if the command line contains the name
            if name in cmdline:
                processes.append(proc)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def kill_existing_processes(process_names: List[str], ports: List[int] = None):
    """Kill existing processes to avoid conflicts."""
    logger.info("Checking for existing processes...")
    killed_any = False
    
    # Kill processes by name
    for name in process_names:
        processes = find_processes_by_name(name)
        if processes:
            logger.info(f"Found {len(processes)} processes matching '{name}'")
            for proc in processes:
                try:
                    pid = proc.pid
                    if pid == os.getpid():  # Don't kill ourselves
                        continue
                    proc.terminate()
                    logger.info(f"Terminated process {pid} ({' '.join(proc.cmdline()[:2])})")
                    killed_any = True
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"Failed to terminate process {proc.pid}: {e}")
    
    # Kill processes by port
    if ports:
        for port in ports:
            try:
                connections = [conn for conn in psutil.net_connections() if conn.laddr.port == port]
                for conn in connections:
                    if conn.pid and conn.pid != os.getpid():
                        try:
                            proc = psutil.Process(conn.pid)
                            proc.terminate()
                            logger.info(f"Terminated process {conn.pid} using port {port}")
                            killed_any = True
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            logger.warning(f"Failed to terminate process {conn.pid}: {e}")
            except psutil.AccessDenied:
                logger.warning(f"Cannot check port {port} usage: access denied")
    
    if killed_any:
        # Give processes time to terminate
        time.sleep(2)
        return True
    else:
        logger.info("No conflicting processes found")
        return False

def clean_caches():
    """Clean Python and application caches."""
    logger.info("Cleaning caches...")
    
    # Clean Python cache files
    try:
        # Remove __pycache__ directories
        subprocess.run(
            "find . -type d -name __pycache__ -exec rm -rf {} +",
            shell=True, check=False, stdout=subprocess.PIPE
        )
        
        # Remove .pyc files
        subprocess.run(
            "find . -name '*.pyc' -delete",
            shell=True, check=False, stdout=subprocess.PIPE
        )
        
        # Clean Dash cache files
        home_dir = os.path.expanduser("~")
        dash_dirs = [
            os.path.join(home_dir, ".dash_jupyter_hooks"),
            os.path.join(home_dir, ".dash_cache"),
            os.path.join(home_dir, ".cache", "dash"),
            os.path.join(home_dir, ".cache", "flask-session")
        ]
        
        for dash_dir in dash_dirs:
            if os.path.exists(dash_dir):
                logger.info(f"Removing Dash cache: {dash_dir}")
                try:
                    if os.path.isdir(dash_dir):
                        import shutil
                        shutil.rmtree(dash_dir)
                    else:
                        os.remove(dash_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean cache {dash_dir}: {e}")
        
        logger.info("Cache cleaning completed")
        return True
    except Exception as e:
        logger.warning(f"Error cleaning caches: {e}")
        return False

def ensure_cache_directories():
    """Ensure necessary cache directories exist."""
    logger.info("Ensuring cache directories exist...")
    
    try:
        # Create cache directories
        cache_dirs = [
            os.path.join("data", "cache"),
            "logs",
            "storage", 
            os.path.join("storage", "results"),
            os.path.join("storage", "results", "json"),
            os.path.join("storage", "results", "pickle")
        ]
        
        for directory in cache_dirs:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating cache directories: {e}")
        return False

def check_data_dependencies() -> Dict[str, bool]:
    """Check if required data dependencies are installed."""
    dependencies = {
        'yfinance': False,
        'tvdatafeed': False,
        'mock_generator': False
    }
    
    # Check for data sources
    try:
        import importlib.util
        
        # Check for yfinance
        try:
            importlib.import_module('yfinance')
            dependencies['yfinance'] = True
        except ImportError:
            pass
        
        # Check for tvdatafeed
        try:
            importlib.import_module('tvdatafeed')
            dependencies['tvdatafeed'] = True
        except ImportError:
            pass
        
        # Check for mock data generator
        mock_generator_path = os.path.join('data', 'mock_data_generator.py')
        if os.path.exists(mock_generator_path):
            dependencies['mock_generator'] = True
    except Exception as e:
        logger.warning(f"Error checking data dependencies: {e}")
    
    return dependencies

def generate_mock_data(symbols: List[str], exchanges: List[str], intervals: List[str]):
    """Generate mock data for testing if needed."""
    logger.info("Generating mock data for testing...")
    
    try:
        # Import mock data generator
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data.mock_data_generator import generate_mock_price_data, save_mock_data_to_cache
        
        # Generate data for each symbol and interval
        for symbol, exchange in zip(symbols, exchanges):
            for interval in intervals:
                try:
                    # Check if data already exists
                    cache_path = os.path.join("data", "cache", f"{exchange}_{symbol}_{interval}.csv")
                    if os.path.exists(cache_path):
                        logger.info(f"Mock data already exists for {symbol} on {exchange} ({interval})")
                        continue
                    
                    # Generate and save mock data
                    logger.info(f"Generating mock data for {symbol} on {exchange} ({interval})")
                    data = generate_mock_price_data(symbol, lookback=500, interval=interval)
                    save_mock_data_to_cache(symbol, exchange, interval, data)
                except Exception as e:
                    logger.warning(f"Error generating mock data for {symbol} ({interval}): {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        return False

def run_dashboard(args: argparse.Namespace):
    """Run the dashboard with the specified options."""
    logger.info("Starting dashboard mode...")
    
    try:
        # Try to import the dashboard module
        import main_dashboard
        
        # Prepare environment variables
        os.environ['DASH_DEBUG'] = "1" if args.debug else "0"
        
        # Set up configuration
        config_path = args.config if args.config else "config/config.json"
        
        # Inform user
        url = f"http://{args.host}:{args.port}"
        print(f"\n{color_text('Starting dashboard:', GREEN)} {color_text(url, BLUE)}")
        print(f"{color_text('Press Ctrl+C to stop the application', YELLOW)}\n")
        
        # Run the dashboard
        main_dashboard.run_app(config_path=config_path, debug=args.debug, 
                              port=args.port, host=args.host)
    except ImportError:
        logger.error("Failed to import main_dashboard module. Make sure it exists and is importable.")
        print(f"\n{color_text('Error:', RED)} Dashboard module could not be imported.")
        print(f"Make sure main_dashboard.py exists and PYTHONPATH is set correctly.")
        print(f"Try running: {color_text('export PYTHONPATH=$PYTHONPATH:$(pwd)', CYAN)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        print(f"\n{color_text('Error starting dashboard:', RED)} {str(e)}")
        sys.exit(1)

def run_scanner(args: argparse.Namespace):
    """Run the scanner with the specified options."""
    logger.info("Starting scanner mode...")
    
    try:
        # Import required modules
        import main
        from core.scanner_system import FibCycleScanner
        
        # Get symbols to scan - default to Indian indices for NSE
        if args.exchange == "NSE" and not args.symbols:
            # Default NSE symbols
            symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
            logger.info(f"Using default NSE symbols: {', '.join(symbols)}")
        else:
            symbols = args.symbols.split(",") if args.symbols else ["NIFTY", "BANKNIFTY"]
        
        # Run scanner
        print(f"\n{color_text('Running scanner for:', GREEN)} {', '.join(symbols)}")
        print(f"{color_text('Exchange:', BLUE)} {args.exchange}")
        print(f"{color_text('Interval:', BLUE)} {args.interval}")
        
        # Use main.py's functionality
        scanner_args = [
            "--mode", "scan",
            "--symbols", ",".join(symbols),
            "--config", args.config if args.config else "config/config.json",
            "--exchange", args.exchange
        ]
        
        if args.debug:
            scanner_args.extend(["--log-level", "DEBUG"])
            
        # Update sys.argv
        sys.argv = [sys.argv[0]] + scanner_args
        main.main()
        
    except ImportError:
        logger.error("Failed to import scanner modules. Make sure they exist and are importable.")
        print(f"\n{color_text('Error:', RED)} Scanner modules could not be imported.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running scanner: {e}")
        print(f"\n{color_text('Error running scanner:', RED)} {str(e)}")
        sys.exit(1)

def run_advanced_backtest(args: argparse.Namespace):
    """Run the advanced backtesting dashboard with the specified options."""
    logger.info("Starting advanced backtesting mode...")
    
    try:
        # Locate the advanced backtesting script
        script_path = os.path.join(project_root, "run_advanced_backtesting.sh")
        
        # Check if script exists
        if not os.path.exists(script_path):
            logger.error(f"Advanced backtesting script not found at {script_path}")
            print(f"\n{color_text('Error:', RED)} Advanced backtesting script not found.")
            print(f"Please make sure run_advanced_backtesting.sh exists and is executable.")
            sys.exit(1)
            
        # Make sure it's executable
        os.chmod(script_path, 0o755)
        
        # Print information
        print(f"\n{color_text('Starting Advanced Backtesting Dashboard:', GREEN)}")
        print(f"{color_text('URL:', BLUE)} http://{args.host}:{args.port}")
        print(f"{color_text('Press Ctrl+C to stop the application', YELLOW)}\n")
        
        # Run the script
        subprocess.run([script_path])
        
    except Exception as e:
        logger.error(f"Error running advanced backtesting: {e}")
        print(f"\n{color_text('Error running advanced backtesting:', RED)} {str(e)}")
        sys.exit(1)


def run_backtest(args: argparse.Namespace):
    """Run the backtesting with the specified options."""
    logger.info("Starting backtest mode...")
    
    try:
        # Import required modules
        import main
        
        # Get symbols to backtest - default to main Indian index for NSE
        if args.exchange == "NSE" and not args.symbols:
            # Default to NIFTY for backtest
            symbols = ["NIFTY"]
            logger.info(f"Using default NSE symbol for backtest: NIFTY")
        else:
            symbols = args.symbols.split(",") if args.symbols else ["NIFTY"]
        
        # Run backtest
        print(f"\n{color_text('Running backtest for:', GREEN)} {', '.join(symbols)}")
        print(f"{color_text('Exchange:', BLUE)} {args.exchange}")
        print(f"{color_text('Interval:', BLUE)} {args.interval}")
        print(f"{color_text('Data Source:', BLUE)} {args.data_source}")
        
        # Use main.py's functionality
        backtest_args = [
            "--mode", "backtest",
            "--symbols", ",".join(symbols),
            "--config", args.config if args.config else "config/config.json",
            "--exchange", args.exchange,
            "--interval", args.interval
        ]
        
        if args.debug:
            backtest_args.extend(["--log-level", "DEBUG"])
            
        # Update sys.argv
        sys.argv = [sys.argv[0]] + backtest_args
        main.main()
        
    except ImportError:
        logger.error("Failed to import backtest modules. Make sure they exist and are importable.")
        print(f"\n{color_text('Error:', RED)} Backtest modules could not be imported.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        print(f"\n{color_text('Error running backtest:', RED)} {str(e)}")
        sys.exit(1)

def setup_data_source(args: argparse.Namespace):
    """Set up the data source based on arguments or available sources."""
    data_dependencies = check_data_dependencies()
    
    # If data source is auto, determine best available
    if args.data_source == "auto":
        # For Indian markets (NSE), tvDatafeed is the primary choice
        if data_dependencies['tvdatafeed']:
            args.data_source = "tvdatafeed"
            logger.info("Using TradingView as data source (primary choice for NSE/Indian markets)")
        elif data_dependencies['yfinance']:
            args.data_source = "yfinance"
            logger.warning("Using Yahoo Finance as fallback - limited NSE data. Consider installing tvDatafeed.")
            if args.exchange == "NSE":
                print(f"{color_text('Warning:', YELLOW)} Yahoo Finance has limited NSE data.")
                print(f"For optimal results with Indian markets, install tvDatafeed:")
                print(f"{color_text('pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git', CYAN)}")
        elif data_dependencies['mock_generator']:
            args.data_source = "mock"
            logger.warning("No real data sources available. Using mock data as last resort.")
            print(f"{color_text('Warning:', RED)} No real data sources detected!")
            print(f"For Indian market data, please install tvDatafeed:")
            print(f"{color_text('pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git', CYAN)}")
        else:
            logger.warning("No data sources available. Defaulting to mock data.")
            args.data_source = "mock"
            print(f"{color_text('Warning:', RED)} No data sources available.")
            print(f"For Indian market data, please install tvDatafeed:")
            print(f"{color_text('pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git', CYAN)}")
    
    # If using mock data, generate it (as last resort only)
    if args.data_source == "mock":
        # Define symbols for mock data - prioritize NSE symbols
        default_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "AAPL", "MSFT"]
        default_exchanges = ["NSE", "NSE", "NSE", "NSE", "NSE", "NYSE", "NYSE"]
        default_intervals = ["daily", "4h", "1h", "15m"]
        
        # Include user-specified symbols
        if args.symbols:
            user_symbols = args.symbols.split(",")
            user_exchanges = [args.exchange] * len(user_symbols)
            
            # Combine with defaults
            symbols = user_symbols + default_symbols
            exchanges = user_exchanges + default_exchanges
        else:
            symbols = default_symbols
            exchanges = default_exchanges
        
        # Generate mock data
        generate_mock_data(symbols, exchanges, default_intervals)
    
    # Update config with data source
    if args.config:
        try:
            # Load existing config
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # Update data source
            if 'data' not in config:
                config['data'] = {}
            config['data']['source'] = args.data_source
            
            # Write updated config
            with open(args.config, 'w') as f:
                json.dump(config, f, indent=4)
                
            logger.info(f"Updated config with data source: {args.data_source}")
        except Exception as e:
            logger.warning(f"Error updating config with data source: {e}")

def ensure_python_path():
    """Ensure the current directory is in PYTHONPATH."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        logger.info(f"Added current directory to PYTHONPATH: {current_dir}")

def setup_logging(args: argparse.Namespace):
    """Set up the logging system based on arguments."""
    try:
        # Try to import our centralized logging system
        from utils.logging_utils import configure_root_logger, set_log_level, log_manager
        
        # Set log level based on debug flag
        log_level = "DEBUG" if args.debug else "INFO"
        
        # Configure root logger
        configure_root_logger(level=log_level)
        
        # Set up session logging for this run
        session_logger = log_manager.create_session_log(f"run_{args.mode}")
        session_logger.info(f"Starting application in {args.mode} mode")
        session_logger.info(f"Arguments: {args}")
        
        # Set log levels for all components
        log_manager.set_default_level(log_level)
        log_manager.set_all_levels(log_level)
        
        logger.info(f"Logging system initialized with level {log_level}")
        
        return True
    except ImportError:
        logger.warning("Centralized logging system not available - using basic logging")
        
        # Set up basic logging
        import logging
        level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"logs/run_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        
        return False

def initialize_environment(args: argparse.Namespace):
    """Initialize the environment for running the application."""
    # Print banner
    print_banner()
    
    # Ensure PYTHONPATH is set
    ensure_python_path()
    
    # Set up logging system
    setup_logging(args)
    
    # Kill existing processes if requested
    if args.kill:
        # Only kill other dashboard processes, not our script
        process_names = ["main_dashboard.py", "dash"]
        ports = [args.port]
        kill_existing_processes(process_names, ports)
    
    # Clean caches if requested
    if args.clean:
        clean_caches()
    
    # Ensure cache directories exist
    ensure_cache_directories()
    
    # Set up data source
    setup_data_source(args)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fibonacci Cycle Trading System - Indian Markets Focus")
    
    # Mode options
    parser.add_argument("--mode", default="dashboard", 
                      choices=["dashboard", "scanner", "backtest", "advanced-backtest"],
                      help="Mode to run (dashboard, scanner, backtest, advanced-backtest)")
    
    # Dashboard options
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                      help=f"Port for the dashboard (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default=DEFAULT_HOST,
                      help=f"Host for the dashboard (default: {DEFAULT_HOST})")
    
    # Scanner/Backtest options - set defaults for Indian markets
    parser.add_argument("--symbols", help="Comma-separated list of symbols to analyze (default: NIFTY,BANKNIFTY)")
    parser.add_argument("--exchange", default="NSE",
                      help="Exchange to use for data (default: NSE for Indian markets)")
    parser.add_argument("--interval", default="daily",
                      help="Timeframe interval (default: daily)")
    
    # General options
    parser.add_argument("--debug", action="store_true",
                      help="Run in debug mode")
    parser.add_argument("--clean", action="store_true",
                      help="Clean caches before running")
    parser.add_argument("--no-kill", dest="kill", action="store_false",
                      help="Don't kill existing processes before running")
    parser.add_argument("--config", help="Path to config file (default: config/config.json)")
    parser.add_argument("--data-source", default="auto", 
                      choices=["auto", "tvdatafeed", "yfinance", "mock"],
                      help="Data source to use (default: auto selects tvdatafeed for NSE data)")
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Initialize environment
        initialize_environment(args)
        
        # Run appropriate mode
        if args.mode == "dashboard":
            run_dashboard(args)
        elif args.mode == "scanner":
            run_scanner(args)
        elif args.mode == "backtest":
            run_backtest(args)
        elif args.mode == "advanced-backtest":
            run_advanced_backtest(args)
        else:
            print(f"{color_text('Error:', RED)} Invalid mode: {args.mode}")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{color_text('Application stopped by user', YELLOW)}")
        sys.exit(0)
    except Exception as e:
        # Use our log_exception function if available
        try:
            from utils.logging_utils import log_exception
            log_exception(logger, e, "Unhandled exception in main application")
        except ImportError:
            # Fallback if the utility isn't available
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            
        print(f"\n{color_text('Unhandled error:', RED)} {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()