#!/usr/bin/env python3

import os
import sys
import inspect
import importlib
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("import_check.log")
    ]
)
logger = logging.getLogger("import_checker")

def check_import(module_name):
    """Try to import a module and report its status."""
    try:
        module = importlib.import_module(module_name)
        logger.info(f"✓ Successfully imported {module_name}")
        return True, module
    except ImportError as e:
        logger.error(f"✗ Failed to import {module_name}: {e}")
        logger.debug(traceback.format_exc())
        return False, None
    except Exception as e:
        logger.error(f"! Error importing {module_name}: {e}")
        logger.debug(traceback.format_exc())
        return False, None

def main():
    """Main diagnostic function."""
    # Print system information
    logger.info("=== Python Import Path Diagnostic Tool ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Get the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Project root: {project_root}")
    
    # Add project root to path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added {project_root} to sys.path")
    
    # Print path
    logger.info("Python path (sys.path):")
    for i, path in enumerate(sys.path):
        logger.info(f"  {i}: {path}")
    
    # Check if key directories exist
    web_dir = os.path.join(project_root, "web")
    if os.path.exists(web_dir):
        logger.info(f"✓ Web directory exists: {web_dir}")
    else:
        logger.error(f"✗ Web directory does not exist: {web_dir}")
    
    # Try importing key modules
    modules_to_check = [
        "web.cycle_visualization",
        "web.fld_visualization",
        "web.scanner_dashboard",
        "web.trading_strategies_ui",
        "web.enhanced_entry_exit_ui",
        "web.backtest_ui",
        "models.scan_result",
        "utils.config",
        "backtesting.backtesting_framework",
        "main_dashboard"
    ]
    
    logger.info("\nTesting imports:")
    for module_name in modules_to_check:
        success, module = check_import(module_name)
        if success and module_name == "web.backtest_ui":
            # Check if we can access key functions
            try:
                create_fn = getattr(module, "create_backtest_ui")
                register_fn = getattr(module, "register_backtest_callbacks")
                logger.info(f"  ✓ Function create_backtest_ui exists")
                logger.info(f"  ✓ Function register_backtest_callbacks exists")
            except AttributeError as e:
                logger.error(f"  ✗ Function error in {module_name}: {e}")

    logger.info("\nDiagnostic complete. Check import_check.log for full details.")

if __name__ == "__main__":
    main()