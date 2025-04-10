#!/usr/bin/env python
"""
Fibonacci Cycles Trading System CLI

A comprehensive command-line interface for the Fibonacci Cycles Trading System.
This unified tool replaces multiple scripts with a single interface for all functions.

Usage:
    python fib_cycles_cli.py <command> [options]

Commands:
    dashboard   - Start the interactive dashboard UI
    analyze     - Analyze one or more symbols for cycle patterns
    backtest    - Run backtests on trading strategies
    scan        - Scan markets for trading signals based on cycle analysis
    trade       - Execute automated trading based on cycle signals
    data        - Data management functions (download, clean, import)
    setup       - System setup and configuration

Options vary by command. Use -h/--help with any command for details.
"""

import os
import sys
import argparse
import json
import logging
import subprocess
import time
import signal
import importlib
import datetime
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")

# Initialize console colors for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def color_text(text, color):
    """Color text for terminal output"""
    return f"{color}{text}{Colors.ENDC}"

# Setup logging
def setup_logging(log_level="INFO", component="cli", log_file=None):
    """Configure logging with custom format and level"""
    try:
        # Try to import our centralized logging system
        from utils.logging_utils import get_component_logger, configure_root_logger
        
        # Configure root logger
        configure_root_logger(level=log_level)
        
        # Get a logger for this component
        logger = get_component_logger(component)
        logger.info(f"Using centralized logging system")
        
        return logger
    except ImportError:
        # Fallback logging configuration
        log_format = '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
        
        # Create logs directory if needed
        os.makedirs("logs", exist_ok=True)
        
        # Set log file name
        if not log_file:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"logs/{component}_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(component)
        logger.warning("Using fallback logging (utils.logging_utils not found)")
        
        return logger

logger = setup_logging(log_level="INFO")

#############################
# Configuration Management #
#############################

def ensure_config_exists():
    """Make sure the config file exists, creating default if needed"""
    config_dir = os.path.join(project_root, "config")
    config_path = os.path.join(config_dir, "config.json")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
        
    if not os.path.exists(config_path):
        logger.info("Creating default configuration...")
        try:
            # Import model definitions to get default config
            from models.model_definitions import get_default_config
            default_config = get_default_config()
            
            # Write default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
                
            logger.info(f"Default configuration created at {config_path}")
        except ImportError:
            # Create minimal config if get_default_config is not available
            minimal_config = {
                "general": {
                    "default_exchange": "NSE",
                    "default_source": "tradingview"
                },
                "data": {
                    "cache_dir": "data/cache",
                },
                "analysis": {
                    "min_period": 10,
                    "max_period": 250,
                    "fib_cycles": [21, 34, 55, 89, 144, 233],
                    "power_threshold": 0.2
                },
                "scanner": {
                    "default_symbols": ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"],
                    "default_exchange": "NSE",
                    "default_interval": "daily",
                    "default_lookback": 1000,
                    "num_cycles": 3
                },
                "visualization": {
                    "theme": "dark",
                    "default_chart_height": 800,
                    "default_chart_width": 1200
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(minimal_config, f, indent=4)
                
            logger.info(f"Minimal configuration created at {config_path}")
    
    return config_path

def load_config(config_path=None):
    """Load configuration from file"""
    if not config_path:
        config_path = os.path.join(project_root, "config", "config.json")
    
    # Make sure config exists
    ensure_config_exists()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        print(color_text(f"Error loading configuration: {e}", Colors.RED))
        sys.exit(1)

def update_config(updates, config_path=None):
    """Update configuration with new values"""
    if not config_path:
        config_path = os.path.join(project_root, "config", "config.json")
    
    try:
        # Load current config
        config = load_config(config_path)
        
        # Apply updates (recursively)
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_nested_dict(d[k], v)
                else:
                    d[k] = v
            return d
        
        config = update_nested_dict(config, updates)
        
        # Write updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        logger.info(f"Configuration updated at {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        print(color_text(f"Error updating configuration: {e}", Colors.RED))
        return None

#############################
# System Preparation Utilities #
#############################

def ensure_directories():
    """Ensure all required directories exist"""
    required_dirs = [
        os.path.join(project_root, "logs"),
        os.path.join(project_root, "data", "cache"),
        os.path.join(project_root, "storage", "results", "json"),
        os.path.join(project_root, "storage", "results", "pickle"),
        os.path.join(project_root, "assets")
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

def kill_existing_processes(port=None):
    """Terminate any existing dashboard processes"""
    import psutil
    
    killed = False
    dashboard_procs = []
    
    # Find processes by name
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.cmdline()).lower() if proc.cmdline() else ""
            
            # Look for dashboard processes
            if "main_dashboard.py" in cmdline or "dashboard" in cmdline:
                if proc.pid != os.getpid():  # Don't kill ourselves
                    dashboard_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Find processes by port
    if port:
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.pid:
                    proc = psutil.Process(conn.pid)
                    if proc.pid != os.getpid():  # Don't kill ourselves
                        dashboard_procs.append(proc)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    
    # Kill the processes
    for proc in dashboard_procs:
        try:
            proc.terminate()
            logger.info(f"Terminated process {proc.pid}")
            killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not terminate process {proc.pid}: {e}")
    
    if killed:
        time.sleep(1)  # Give processes time to terminate
        logger.info("Terminated existing dashboard processes")
    
    return killed

def clean_cache():
    """Clean data cache and Python cache files"""
    import shutil
    
    # Clear Python cache
    for root, dirs, files in os.walk(project_root):
        # Remove __pycache__ directories
        for dir in dirs:
            if dir == "__pycache__":
                cache_dir = os.path.join(root, dir)
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Removed Python cache: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache directory {cache_dir}: {e}")
                    
        # Remove .pyc files
        for file in files:
            if file.endswith(".pyc"):
                cache_file = os.path.join(root, file)
                try:
                    os.remove(cache_file)
                    logger.debug(f"Removed Python cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    # Clear data cache
    data_cache = os.path.join(project_root, "data", "cache")
    if os.path.exists(data_cache):
        for file in os.listdir(data_cache):
            if file.endswith(".pkl") or file.endswith(".csv"):
                cache_file = os.path.join(data_cache, file)
                try:
                    os.remove(cache_file)
                    logger.info(f"Removed data cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove data cache file {cache_file}: {e}")
    
    logger.info("Cache cleaning completed")

#############################
# Command Implementations #
#############################

def print_banner():
    """Print application banner"""
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
    print(color_text(banner, Colors.BLUE))

def cmd_dashboard(args):
    """Launch the interactive dashboard"""
    logger.info("Starting dashboard mode")
    
    try:
        # Prepare environment
        ensure_directories()
        
        if args.kill:
            kill_existing_processes(port=args.port)
        
        if args.clean:
            clean_cache()
        
        # Set default config path if none provided
        if args.config is None:
            args.config = os.path.join(project_root, "config", "config.json")
            
        # Ensure config exists
        ensure_config_exists()
        
        # Set up dashboard environment variables
        os.environ['DASH_DEBUG'] = "1" if args.debug else "0"
        
        # Print launch information
        url = f"http://{args.host}:{args.port}"
        print(color_text("\nLaunching Fibonacci Cycles Dashboard", Colors.GREEN))
        print(f"URL: {color_text(url, Colors.CYAN)}")
        print(f"Config: {color_text(args.config, Colors.CYAN)}")
        print(f"Press Ctrl+C to stop the application\n")
        
        # Import main dashboard module
        try:
            import main_dashboard
            # Launch dashboard
            main_dashboard.run_app(
                config_path=args.config,
                debug=args.debug,
                port=args.port,
                host=args.host
            )
        except ImportError:
            logger.error("Failed to import dashboard module")
            print(color_text("\nError: Dashboard module could not be imported.", Colors.RED))
            print("Make sure main_dashboard.py exists and the project structure is intact.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(color_text("\nDashboard stopped by user", Colors.WARNING))
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}", exc_info=True)
        print(color_text(f"\nError starting dashboard: {e}", Colors.RED))
        sys.exit(1)

def cmd_analyze(args):
    """Analyze symbols for cycle patterns"""
    logger.info(f"Analyzing symbols: {args.symbols}")
    
    symbols = args.symbols.split(",") if args.symbols else ["NIFTY"]
    exchange = args.exchange
    interval = args.interval
    
    print(color_text(f"\nAnalyzing symbols: {', '.join(symbols)}", Colors.GREEN))
    print(f"Exchange: {color_text(exchange, Colors.CYAN)}")
    print(f"Interval: {color_text(interval, Colors.CYAN)}")
    
    try:
        # Import required modules
        from core.scanner_system import FibCycleScanner
        from models.scan_parameters import ScanParameters
        from utils.config import load_config
        
        # Load config
        config = load_config(args.config)
        
        # Initialize scanner
        scanner = FibCycleScanner(config)
        
        # Create scan parameters for each symbol
        scan_parameters = []
        for symbol in symbols:
            params = ScanParameters(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                lookback=args.lookback,
                num_cycles=args.cycles,
                price_source=args.price_source,
                generate_chart=True
            )
            scan_parameters.append(params)
        
        # Run analysis
        results = []
        for params in scan_parameters:
            print(f"\nAnalyzing {params.symbol}...")
            result = scanner.analyze_symbol(params)
            results.append(result)
            
            # Display results
            if result.success:
                print(color_text("Analysis successful:", Colors.GREEN))
                print(f"  Detected cycles: {', '.join(map(str, result.detected_cycles))}")
                print(f"  Signal: {result.signal} (Strength: {result.signal_strength:.2f})")
                
                # Show cycle projections
                if hasattr(result, 'cycle_projections'):
                    print(color_text("\nCycle projections:", Colors.BLUE))
                    for cycle, projections in result.cycle_projections.items():
                        print(f"  Cycle {cycle}:")
                        for i, turn in enumerate(projections[:3]):  # Show first 3
                            print(f"    {i+1}. {turn['type']} on {turn['date'].strftime('%Y-%m-%d')}")
            else:
                print(color_text(f"Analysis failed: {result.error}", Colors.RED))
        
        # Save results
        if args.output:
            from storage.results_repository import ResultsRepository
            repo = ResultsRepository()
            for result in results:
                if result.success:
                    repo.save_result(result)
            print(color_text(f"\nResults saved to storage", Colors.GREEN))
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(color_text(f"\nError importing required modules: {e}", Colors.RED))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(color_text(f"\nError during analysis: {e}", Colors.RED))
        sys.exit(1)

def cmd_backtest(args):
    """Run backtests on trading strategies"""
    logger.info(f"Running backtest on {args.symbols} ({args.interval})")
    
    symbols = args.symbols.split(",") if args.symbols else ["NIFTY"]
    
    print(color_text(f"\nRunning backtest for: {', '.join(symbols)}", Colors.GREEN))
    print(f"Exchange: {color_text(args.exchange, Colors.CYAN)}")
    print(f"Interval: {color_text(args.interval, Colors.CYAN)}")
    print(f"Strategy: {color_text(args.strategy, Colors.CYAN)}")
    
    try:
        # Call the appropriate backtest script based on type
        if args.type == "advanced":
            print(color_text("\nLaunching advanced backtesting dashboard...", Colors.BLUE))
            try:
                # Try to use the advanced backtest UI
                from web.advanced_backtest_ui import run_app as run_backtest_app
                run_backtest_app(
                    debug=args.debug,
                    port=args.port,
                    host=args.host
                )
            except ImportError:
                # Fallback to script if module not found
                backtest_script = os.path.join(project_root, "run_advanced_backtesting.sh")
                if os.path.exists(backtest_script):
                    subprocess.run(["bash", backtest_script])
                else:
                    raise ImportError("Advanced backtesting module not available")
        else:
            # Standard backtesting
            print(color_text("\nRunning standard backtest...", Colors.BLUE))
            
            # Import necessary modules
            from models.scan_parameters import ScanParameters
            
            # Create scan parameters for backtest
            params = []
            for symbol in symbols:
                param = ScanParameters(
                    symbol=symbol,
                    exchange=args.exchange,
                    interval=args.interval,
                    lookback=args.lookback,
                    num_cycles=args.cycles,
                    price_source=args.price_source,
                    generate_chart=True
                )
                params.append(param)
            
            # Set up backtest parameters
            backtest_params = {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "initial_capital": args.capital,
                "position_size": args.position_size,
                "strategy": args.strategy
            }
            
            # Run backtest
            try:
                import run_backtest
                results = run_backtest.run_backtest(params[0], backtest_params)
                
                # Print summary
                print(color_text("\nBacktest Results:", Colors.GREEN))
                print(f"Symbol: {results.get('symbol')}")
                print(f"Period: {results.get('start_date')} to {results.get('end_date')}")
                print(f"Total Return: {results.get('total_return', 0):.2f}%")
                print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
                print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
                print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
                
                # Save results if requested
                if args.output:
                    output_file = args.output
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=4, default=str)
                    print(color_text(f"\nResults saved to {output_file}", Colors.GREEN))
                    
            except ImportError:
                print(color_text("\nError: Backtesting module not found", Colors.RED))
                print("Standard backtesting is not available in this build.")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print(color_text("\nBacktest stopped by user", Colors.WARNING))
    except Exception as e:
        logger.error(f"Error during backtest: {e}", exc_info=True)
        print(color_text(f"\nError during backtest: {e}", Colors.RED))
        sys.exit(1)

def cmd_scan(args):
    """Scan markets for trading opportunities"""
    logger.info(f"Scanning markets with interval {args.interval}")
    
    # Determine symbols to scan
    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        # Use default symbols from config
        config = load_config(args.config)
        symbols = config.get("scanner", {}).get("default_symbols", ["NIFTY", "BANKNIFTY"])
    
    print(color_text(f"\nScanning {len(symbols)} symbols for trading signals", Colors.GREEN))
    print(f"Exchange: {color_text(args.exchange, Colors.CYAN)}")
    print(f"Interval: {color_text(args.interval, Colors.CYAN)}")
    
    try:
        # Import main scan functionality
        import main
        
        # Build arguments for main.py scan function
        scan_args = [
            "--mode", "scan",
            "--symbols", ",".join(symbols),
            "--config", args.config,
            "--exchange", args.exchange,
            "--interval", args.interval
        ]
        
        if args.debug:
            scan_args.extend(["--log-level", "DEBUG"])
            
        # Update sys.argv
        orig_argv = sys.argv
        sys.argv = [sys.argv[0]] + scan_args
        
        # Run scan
        main.main()
        
        # Restore original argv
        sys.argv = orig_argv
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(color_text(f"\nError importing scan module: {e}", Colors.RED))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during scan: {e}", exc_info=True)
        print(color_text(f"\nError during scan: {e}", Colors.RED))
        sys.exit(1)

def cmd_trade(args):
    """Execute automated trading"""
    logger.info(f"Starting automated trading with strategy {args.strategy}")
    
    symbols = args.symbols.split(",") if args.symbols else ["NIFTY"]
    
    print(color_text(f"\nStarting automated trading for: {', '.join(symbols)}", Colors.GREEN))
    print(f"Strategy: {color_text(args.strategy, Colors.CYAN)}")
    print(f"Mode: {color_text(args.mode, Colors.CYAN)}")
    
    # Check if trading module is available
    try:
        # Import trading integration if available
        import integration.broker_integration as broker
        
        # Check if we're in paper trading or live trading mode
        if args.mode == "paper":
            print(color_text("\nRunning in PAPER TRADING mode", Colors.WARNING))
            print("Real trading disabled. Trades will be simulated.")
        else:
            print(color_text("\nRunning in LIVE TRADING mode", Colors.RED))
            print(color_text("WARNING: Real money will be used for trades!", Colors.RED))
            
            # Confirm before proceeding with live trading
            confirmation = input(color_text("Type 'CONFIRM' to proceed with live trading: ", Colors.WARNING))
            if confirmation != "CONFIRM":
                print(color_text("Live trading aborted", Colors.WARNING))
                return
        
        # Initialize the trading system
        try:
            trading_system = broker.initialize_broker(
                broker=args.broker,
                api_key=args.api_key,
                api_secret=args.api_secret,
                mode=args.mode
            )
            
            print(color_text(f"\nConnected to {args.broker} successfully", Colors.GREEN))
            
            # Run the trading strategy
            for symbol in symbols:
                print(f"\nSetting up trading for {symbol}...")
                # Set up scanner parameters
                scan_params = {
                    "symbol": symbol,
                    "exchange": args.exchange,
                    "interval": args.interval,
                    "price_source": args.price_source,
                    "cycles": args.cycles
                }
                
                # Setup strategy
                strategy_params = {
                    "name": args.strategy,
                    "position_size": args.position_size,
                    "stop_loss": args.stop_loss,
                    "take_profit": args.take_profit
                }
                
                # Start trading session
                broker.start_trading_session(
                    trading_system=trading_system,
                    scan_params=scan_params,
                    strategy_params=strategy_params,
                    max_duration=args.duration
                )
        
        except Exception as e:
            logger.error(f"Error in trading system: {e}", exc_info=True)
            print(color_text(f"\nError in trading system: {e}", Colors.RED))
            
    except ImportError:
        print(color_text("\nError: Trading module not found", Colors.RED))
        print("The trading functionality is not available in this build.")
        print("Please install the broker integration module to use trading features.")
        sys.exit(1)

def cmd_data(args):
    """Data management functions"""
    logger.info(f"Running data command: {args.action}")
    
    if args.action == "download":
        # Download data
        symbols = args.symbols.split(",") if args.symbols else ["NIFTY"]
        
        print(color_text(f"\nDownloading data for: {', '.join(symbols)}", Colors.GREEN))
        print(f"Exchange: {color_text(args.exchange, Colors.CYAN)}")
        print(f"Interval: {color_text(args.interval, Colors.CYAN)}")
        
        try:
            # Import data fetcher
            from data.data_management import DataFetcher
            from utils.config import load_config
            
            # Load config
            config = load_config(args.config)
            
            # Create data fetcher
            data_fetcher = DataFetcher(config)
            
            # Download data for each symbol
            for symbol in symbols:
                print(f"\nDownloading {symbol} data...")
                data = data_fetcher.get_data(
                    symbol=symbol,
                    exchange=args.exchange,
                    interval=args.interval,
                    lookback=args.lookback,
                    force_download=True
                )
                
                if data is not None:
                    print(color_text(f"Downloaded {len(data)} bars for {symbol}", Colors.GREEN))
                else:
                    print(color_text(f"Failed to download data for {symbol}", Colors.RED))
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            print(color_text(f"\nError importing data modules: {e}", Colors.RED))
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error downloading data: {e}", exc_info=True)
            print(color_text(f"\nError downloading data: {e}", Colors.RED))
            sys.exit(1)
    
    elif args.action == "clean":
        # Clean data cache
        print(color_text("\nCleaning data cache...", Colors.GREEN))
        
        try:
            import os
            import shutil
            
            # Clean data cache directory
            cache_dir = os.path.join(project_root, "data", "cache")
            if os.path.exists(cache_dir):
                for file in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            print(f"Removed: {file}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            print(f"Removed directory: {file}")
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {e}")
                        print(color_text(f"Error removing {file}: {e}", Colors.RED))
            
            print(color_text("\nData cache cleaned successfully", Colors.GREEN))
            
        except Exception as e:
            logger.error(f"Error cleaning data cache: {e}", exc_info=True)
            print(color_text(f"\nError cleaning data cache: {e}", Colors.RED))
            sys.exit(1)
    
    elif args.action == "import":
        # Import external data
        print(color_text(f"\nImporting data from {args.file}", Colors.GREEN))
        
        try:
            import pandas as pd
            import os
            
            # Check if file exists
            if not os.path.exists(args.file):
                print(color_text(f"Error: File not found: {args.file}", Colors.RED))
                sys.exit(1)
            
            # Load the data file
            try:
                if args.file.endswith('.csv'):
                    df = pd.read_csv(args.file)
                elif args.file.endswith('.xlsx'):
                    df = pd.read_excel(args.file)
                elif args.file.endswith('.pkl'):
                    df = pd.read_pickle(args.file)
                else:
                    print(color_text(f"Unsupported file format: {args.file}", Colors.RED))
                    sys.exit(1)
            except Exception as e:
                print(color_text(f"Error reading file: {e}", Colors.RED))
                sys.exit(1)
            
            # Display data summary
            print(f"\nData summary:")
            print(f"Rows: {len(df)}")
            print(f"Columns: {', '.join(df.columns)}")
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(color_text(f"Warning: Missing required columns: {', '.join(missing_cols)}", Colors.WARNING))
                # Ask for column mapping if needed
                for col in missing_cols:
                    mapped_col = input(f"Enter column name for '{col}': ")
                    if mapped_col in df.columns:
                        df[col] = df[mapped_col]
                    else:
                        print(color_text(f"Error: Column '{mapped_col}' not found in data", Colors.RED))
                        sys.exit(1)
            
            # Check date column
            date_col = 'date'
            if date_col not in df.columns:
                print(color_text(f"Warning: Missing date column", Colors.WARNING))
                date_col = input("Enter column name for 'date': ")
                if date_col not in df.columns:
                    print(color_text(f"Error: Column '{date_col}' not found in data", Colors.RED))
                    sys.exit(1)
            
            # Convert date column to datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            except Exception as e:
                print(color_text(f"Error converting date column: {e}", Colors.RED))
                sys.exit(1)
            
            # Save to cache
            try:
                # Ensure cache directory exists
                cache_dir = os.path.join(project_root, "data", "cache")
                os.makedirs(cache_dir, exist_ok=True)
                
                # Generate cache filename
                symbol = args.symbol if args.symbol else "IMPORTED"
                exchange = args.exchange if args.exchange else "CUSTOM"
                interval = args.interval if args.interval else "daily"
                
                cache_file = os.path.join(cache_dir, f"{exchange}_{symbol}_{interval}.pkl")
                
                # Save to cache
                df.to_pickle(cache_file)
                print(color_text(f"\nData imported successfully and saved to {cache_file}", Colors.GREEN))
                
            except Exception as e:
                print(color_text(f"Error saving imported data: {e}", Colors.RED))
                sys.exit(1)
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            print(color_text(f"\nError importing required modules: {e}", Colors.RED))
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error importing data: {e}", exc_info=True)
            print(color_text(f"\nError importing data: {e}", Colors.RED))
            sys.exit(1)
    
    else:
        print(color_text(f"Unknown data action: {args.action}", Colors.RED))
        sys.exit(1)

def cmd_setup(args):
    """System setup and configuration"""
    logger.info(f"Running setup: {args.action}")
    
    if args.action == "config":
        # Configure the system
        print(color_text("\nConfiguring Fibonacci Cycles System", Colors.GREEN))
        
        # Make sure config exists
        config_path = ensure_config_exists()
        config = load_config(config_path)
        
        if args.list:
            # Display current configuration
            print(color_text("\nCurrent Configuration:", Colors.BLUE))
            print(json.dumps(config, indent=2))
            return
        
        # Check if we need to update specific settings
        if args.set:
            updates = {}
            for setting in args.set:
                if '=' not in setting:
                    print(color_text(f"Invalid setting format: {setting} (use key=value)", Colors.RED))
                    continue
                
                key, value = setting.split('=', 1)
                
                # Parse key path (e.g., "data.cache_dir")
                key_parts = key.split('.')
                
                # Convert value to appropriate type
                try:
                    # Try to convert to number or boolean if possible
                    if value.lower() == 'true':
                        typed_value = True
                    elif value.lower() == 'false':
                        typed_value = False
                    elif value.isdigit():
                        typed_value = int(value)
                    elif '.' in value and all(p.isdigit() for p in value.split('.', 1)):
                        typed_value = float(value)
                    else:
                        typed_value = value
                except:
                    typed_value = value
                
                # Build nested dictionary
                current = updates
                for i, part in enumerate(key_parts):
                    if i == len(key_parts) - 1:
                        current[part] = typed_value
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
            
            # Update config
            if updates:
                update_config(updates, config_path)
                print(color_text("\nConfiguration updated successfully", Colors.GREEN))
                
                # Display updated values
                print("\nUpdated settings:")
                for setting in args.set:
                    key, value = setting.split('=', 1)
                    print(f"{key} = {value}")
            else:
                print(color_text("\nNo valid settings provided", Colors.WARNING))
    
    elif args.action == "check":
        # Check system
        print(color_text("\nChecking Fibonacci Cycles System", Colors.GREEN))
        
        # Perform system checks
        all_ok = True
        
        # Check Python version
        import platform
        python_version = platform.python_version()
        python_ok = python_version >= "3.6"
        print(f"Python Version: {python_version}" + (" ✓" if python_ok else " ✗"))
        all_ok = all_ok and python_ok
        
        # Check directory structure
        dirs_to_check = [
            "logs",
            "data",
            "data/cache",
            "config",
            "storage",
            "storage/results"
        ]
        
        print("\nDirectory Structure:")
        dir_ok = True
        for directory in dirs_to_check:
            dir_path = os.path.join(project_root, directory)
            exists = os.path.exists(dir_path)
            print(f"  {directory}" + (" ✓" if exists else " ✗"))
            dir_ok = dir_ok and exists
        
        all_ok = all_ok and dir_ok
        
        # Check core modules
        print("\nCore Modules:")
        modules_to_check = [
            "core.cycle_detection",
            "core.scanner_system",
            "core.fld_signal_generator",
            "data.data_management",
            "web.cycle_visualization",
            "main_dashboard"
        ]
        
        module_ok = True
        for module_name in modules_to_check:
            try:
                importlib.import_module(module_name)
                print(f"  {module_name} ✓")
            except ImportError as e:
                print(f"  {module_name} ✗ ({e})")
                module_ok = False
        
        all_ok = all_ok and module_ok
        
        # Check data sources
        print("\nData Sources:")
        
        try:
            importlib.import_module("yfinance")
            print("  Yahoo Finance ✓")
            yfin_ok = True
        except ImportError:
            print("  Yahoo Finance ✗ (not installed)")
            yfin_ok = False
        
        try:
            importlib.import_module("tvdatafeed")
            print("  TradingView ✓")
            tv_ok = True
        except ImportError:
            print("  TradingView ✗ (not installed)")
            tv_ok = False
        
        data_ok = yfin_ok or tv_ok
        all_ok = all_ok and data_ok
        
        # Check configuration
        config_path = os.path.join(project_root, "config", "config.json")
        config_exists = os.path.exists(config_path)
        print("\nConfiguration:")
        print(f"  config.json" + (" ✓" if config_exists else " ✗"))
        
        all_ok = all_ok and config_exists
        
        # Summary
        print("\nSystem Check " + (color_text("PASSED", Colors.GREEN) if all_ok else color_text("FAILED", Colors.RED)))
        
        if not all_ok:
            print("\nRecommended actions:")
            if not python_ok:
                print("  - Upgrade Python to version 3.6 or later")
            if not dir_ok:
                print("  - Run 'fib_cycles_cli.py setup init' to create missing directories")
            if not module_ok:
                print("  - Reinstall the application to restore missing modules")
            if not data_ok:
                print("  - Install at least one data source:")
                if not yfin_ok:
                    print("    - pip install yfinance")
                if not tv_ok:
                    print("    - pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git")
            if not config_exists:
                print("  - Run 'fib_cycles_cli.py setup init' to create default configuration")
    
    elif args.action == "init":
        # Initialize system
        print(color_text("\nInitializing Fibonacci Cycles System", Colors.GREEN))
        
        # Create directories
        ensure_directories()
        print("Directories created ✓")
        
        # Create default configuration
        config_path = ensure_config_exists()
        print("Configuration created ✓")
        
        # Clear caches
        clean_cache()
        print("Caches cleaned ✓")
        
        print(color_text("\nSystem initialized successfully", Colors.GREEN))
        print("You can now run the dashboard with: python fib_cycles_cli.py dashboard")
    
    else:
        print(color_text(f"Unknown setup action: {args.action}", Colors.RED))
        sys.exit(1)

#############################
# Command Line Argument Parsing #
#############################

def create_parser():
    """Create the command-line argument parser with all subcommands"""
    parser = argparse.ArgumentParser(
        description='Fibonacci Cycles Trading System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the dashboard UI
  python fib_cycles_cli.py dashboard

  # Analyze a specific symbol
  python fib_cycles_cli.py analyze --symbols NIFTY,BANKNIFTY --interval daily

  # Run a backtest
  python fib_cycles_cli.py backtest --symbols NIFTY --interval daily

  # Scan the market for signals
  python fib_cycles_cli.py scan --interval daily

  # Download market data
  python fib_cycles_cli.py data download --symbols NIFTY --interval daily
"""
    )
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--config', help='Path to configuration file')
    common_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', parents=[common_parser], 
                                            help='Start the interactive dashboard UI')
    dashboard_parser.add_argument('--port', type=int, default=8050, help='Port for the dashboard')
    dashboard_parser.add_argument('--host', default='127.0.0.1', help='Host for the dashboard')
    dashboard_parser.add_argument('--clean', action='store_true', help='Clean caches before starting')
    dashboard_parser.add_argument('--no-kill', dest='kill', action='store_false', default=True,
                                help='Do not kill existing dashboard processes')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', parents=[common_parser],
                                        help='Analyze symbols for cycle patterns')
    analyze_parser.add_argument('--symbols', help='Comma-separated list of symbols to analyze')
    analyze_parser.add_argument('--exchange', default='NSE', help='Exchange to use')
    analyze_parser.add_argument('--interval', default='daily', help='Analysis interval')
    analyze_parser.add_argument('--lookback', type=int, default=1000, help='Number of bars to look back')
    analyze_parser.add_argument('--cycles', type=int, default=3, help='Number of cycles to detect')
    analyze_parser.add_argument('--price-source', default='close', help='Price data source (close, hl2, etc.)')
    analyze_parser.add_argument('--output', action='store_true', help='Save results to storage')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', parents=[common_parser],
                                           help='Run backtests on trading strategies')
    backtest_parser.add_argument('--symbols', help='Comma-separated list of symbols to backtest')
    backtest_parser.add_argument('--exchange', default='NSE', help='Exchange to use')
    backtest_parser.add_argument('--interval', default='daily', help='Backtest interval')
    backtest_parser.add_argument('--lookback', type=int, default=1000, help='Number of bars to look back')
    backtest_parser.add_argument('--cycles', type=int, default=3, help='Number of cycles to detect')
    backtest_parser.add_argument('--price-source', default='close', help='Price data source (close, hl2, etc.)')
    backtest_parser.add_argument('--strategy', default='fld_crossover', 
                                help='Trading strategy to backtest')
    backtest_parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    backtest_parser.add_argument('--position-size', type=float, default=0.1, 
                                help='Position size as fraction of capital')
    backtest_parser.add_argument('--output', help='Output file for results (JSON)')
    backtest_parser.add_argument('--type', choices=['standard', 'advanced'], default='standard',
                                help='Type of backtest to run')
    backtest_parser.add_argument('--port', type=int, default=8050, 
                                help='Port for advanced backtesting dashboard')
    backtest_parser.add_argument('--host', default='127.0.0.1', 
                                help='Host for advanced backtesting dashboard')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', parents=[common_parser],
                                      help='Scan markets for trading signals')
    scan_parser.add_argument('--symbols', help='Comma-separated list of symbols to scan')
    scan_parser.add_argument('--exchange', default='NSE', help='Exchange to use')
    scan_parser.add_argument('--interval', default='daily', help='Scan interval')
    scan_parser.add_argument('--lookback', type=int, default=1000, help='Number of bars to look back')
    scan_parser.add_argument('--min-strength', type=float, default=0.6,
                           help='Minimum signal strength to report')
    scan_parser.add_argument('--filter', choices=['buy', 'sell', 'any'], default='any',
                           help='Filter signals by direction')
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', parents=[common_parser],
                                       help='Execute automated trading')
    trade_parser.add_argument('--symbols', help='Comma-separated list of symbols to trade')
    trade_parser.add_argument('--exchange', default='NSE', help='Exchange to use')
    trade_parser.add_argument('--interval', default='daily', help='Trading interval')
    trade_parser.add_argument('--cycles', type=int, default=3, help='Number of cycles to detect')
    trade_parser.add_argument('--price-source', default='close', help='Price data source (close, hl2, etc.)')
    trade_parser.add_argument('--strategy', default='fld_crossover',
                            help='Trading strategy to use')
    trade_parser.add_argument('--broker', default='paper',
                            help='Broker to use for trading')
    trade_parser.add_argument('--api-key', help='API key for broker')
    trade_parser.add_argument('--api-secret', help='API secret for broker')
    trade_parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                            help='Trading mode (paper or live)')
    trade_parser.add_argument('--position-size', type=float, default=0.1,
                            help='Position size as fraction of capital')
    trade_parser.add_argument('--stop-loss', type=float, default=0.02,
                            help='Stop loss as fraction of entry price')
    trade_parser.add_argument('--take-profit', type=float, default=0.05,
                            help='Take profit as fraction of entry price')
    trade_parser.add_argument('--duration', type=int, default=480,
                            help='Maximum duration of trading session in minutes')
    
    # Data command
    data_parser = subparsers.add_parser('data', parents=[common_parser],
                                      help='Data management functions')
    data_subparsers = data_parser.add_subparsers(dest='action', help='Data action to perform')
    
    # Data download
    download_parser = data_subparsers.add_parser('download', help='Download market data')
    download_parser.add_argument('--symbols', help='Comma-separated list of symbols to download')
    download_parser.add_argument('--exchange', default='NSE', help='Exchange to use')
    download_parser.add_argument('--interval', default='daily', help='Data interval')
    download_parser.add_argument('--lookback', type=int, default=1000, help='Number of bars to download')
    
    # Data clean
    clean_parser = data_subparsers.add_parser('clean', help='Clean data cache')
    
    # Data import
    import_parser = data_subparsers.add_parser('import', help='Import external data')
    import_parser.add_argument('file', help='File to import (CSV, Excel, or pickle)')
    import_parser.add_argument('--symbol', help='Symbol name for imported data')
    import_parser.add_argument('--exchange', help='Exchange name for imported data')
    import_parser.add_argument('--interval', help='Interval for imported data')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', parents=[common_parser],
                                       help='System setup and configuration')
    setup_subparsers = setup_parser.add_subparsers(dest='action', help='Setup action to perform')
    
    # Setup config
    config_parser = setup_subparsers.add_parser('config', help='Configure the system')
    config_parser.add_argument('--list', action='store_true', help='List current configuration')
    config_parser.add_argument('--set', nargs='+', help='Set configuration values (key=value)')
    
    # Setup check
    check_parser = setup_subparsers.add_parser('check', help='Check system installation')
    
    # Setup init
    init_parser = setup_subparsers.add_parser('init', help='Initialize the system')
    
    return parser

def main():
    """Main entry point for the CLI"""
    # Print banner
    print_banner()
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # If no command provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Dispatch command
    try:
        if args.command == 'dashboard':
            cmd_dashboard(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'backtest':
            cmd_backtest(args)
        elif args.command == 'scan':
            cmd_scan(args)
        elif args.command == 'trade':
            cmd_trade(args)
        elif args.command == 'data':
            cmd_data(args)
        elif args.command == 'setup':
            cmd_setup(args)
        else:
            print(color_text(f"Unknown command: {args.command}", Colors.RED))
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print(color_text("\nOperation cancelled by user", Colors.WARNING))
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        print(color_text(f"\nError: {e}", Colors.RED))
        sys.exit(1)

if __name__ == "__main__":
    main()