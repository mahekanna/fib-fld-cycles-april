import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

# Import core components
from core.cycle_detection import CycleDetector
from core.fld_signal_generator import FLDCalculator, SignalGenerator
from core.scanner_system import FibCycleScanner
from core.market_regime_detector import MarketRegimeDetector

# Import data management
from data.data_management import DataFetcher, DataProcessor

# Import models
from models.model_definitions import get_default_config

# Import visualization
from visualization.visualization_system import create_dashboard

# Backtesting removed

# Import utilities
from utils.report_generation import ReportGenerator


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up logging
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = log_levels.get(log_level.upper(), logging.INFO)
    
    # Configure logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/fib_cycles.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("fib_cycles")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config dictionary
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Creating default config...")
        
        # Get default config
        config = get_default_config()
        
        # Save default config
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
        return config
    
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fibonacci Cycle Analysis System")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--mode", type=str, default="scan", help="Operation mode (scan, backtest, dashboard)")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to analyze")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Fibonacci Cycle Analysis System")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded successfully")
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(config, logger)
    logger.info("Data fetcher initialized")
    
    # Process based on mode
    if args.mode == "scan":
        # Run scanner mode
        scanner = FibCycleScanner(config, data_fetcher, logger)
        
        # Get symbols to scan
        if args.symbols:
            symbols = args.symbols.split(",")
        else:
            # Use default symbols from config
            symbols = config.get("scanner", {}).get("default_symbols", [])
        
        if not symbols:
            logger.error("No symbols specified for scanning")
            sys.exit(1)
        
        logger.info(f"Scanning {len(symbols)} symbols")
        
        # Create scan parameters
        from models.scan_parameters import ScanParameters
        
        scan_parameters = []
        for symbol in symbols:
            params = ScanParameters(
                symbol=symbol,
                exchange=config.get("scanner", {}).get("default_exchange", "NSE"),
                interval=config.get("scanner", {}).get("default_interval", "daily"),
                lookback=config.get("scanner", {}).get("default_lookback", 1000),
                price_source=config.get("scanner", {}).get("price_source", "close"),
                num_cycles=config.get("scanner", {}).get("num_cycles", 3),
                generate_chart=True
            )
            scan_parameters.append(params)
        
        # Run the scan
        results = scanner.scan_batch(scan_parameters)
        
        # Filter and rank results
        filtered_results = scanner.filter_signals(
            results,
            signal_type=config.get("scanner", {}).get("filter_signal"),
            min_confidence=config.get("scanner", {}).get("min_confidence"),
            min_alignment=config.get("scanner", {}).get("min_alignment")
        )
        
        ranked_results = scanner.rank_results(
            filtered_results,
            ranking_factor=config.get("scanner", {}).get("ranking_factor", "strength")
        )
        
        # Generate report
        report_generator = ReportGenerator(config, logger)
        report_generator.generate_scan_report(ranked_results, "scan_results.html")
        
        logger.info(f"Scan completed with {len(ranked_results)} filtered results")
        
    elif args.mode == "backtest":
        # Backtesting functionality removed
        logger.info("Backtesting mode has been removed from this system")
        sys.exit(1)
        
    elif args.mode == "dashboard":
        # Run dashboard mode
        import web.dashboard_implementation as dashboard
        
        logger.info("Starting dashboard...")
        dashboard.run_dashboard(config, data_fetcher)
        
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    logger.info("Execution completed successfully")


if __name__ == "__main__":
    main()