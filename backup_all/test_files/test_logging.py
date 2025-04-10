#!/usr/bin/env python
"""
Test script for the centralized logging system.
This script demonstrates the usage of the logging system and verifies its functionality.
"""

import os
import sys
import time
from datetime import datetime

# Ensure the current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the logging utilities
try:
    from utils.logging_utils import (
        get_component_logger,
        log_manager,
        configure_root_logger,
        log_exception,
        set_log_level
    )
except ImportError:
    print("Error: Could not import logging_utils. Make sure the module exists.")
    sys.exit(1)

def test_basic_logging():
    """Test basic logging functionality."""
    logger = get_component_logger("test")
    
    print("\nTesting basic logging...")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    return True

def test_component_loggers():
    """Test component-specific loggers."""
    print("\nTesting component loggers...")
    
    # Create loggers for different components
    scanner_logger = get_component_logger("core.scanner")
    data_logger = get_component_logger("data.fetcher")
    web_logger = get_component_logger("web.dashboard")
    
    # Log messages from each component
    scanner_logger.info("Scanner initialized")
    data_logger.info("Fetching data for NIFTY")
    web_logger.info("Dashboard started")
    
    # Get a list of all configured loggers
    all_loggers = log_manager.loggers
    print(f"Created {len(all_loggers)} component loggers")
    
    return True

def test_exception_logging():
    """Test exception logging functionality."""
    print("\nTesting exception logging...")
    logger = get_component_logger("test.exceptions")
    
    try:
        # Intentionally cause an exception
        result = 1 / 0
    except Exception as e:
        # Log the exception
        log_exception(logger, e, "Error during test calculation")
        print("Exception logged successfully")
    
    return True

def test_log_levels():
    """Test changing log levels."""
    print("\nTesting log level changes...")
    logger = get_component_logger("test.levels")
    
    # Default level should be INFO
    logger.debug("This debug message should NOT appear")
    logger.info("This info message should appear")
    
    # Change to DEBUG level
    set_log_level("test.levels", "DEBUG")
    logger.debug("This debug message should now appear")
    
    # Change all loggers to WARNING
    log_manager.set_all_levels("WARNING")
    logger.debug("This debug message should NOT appear")
    logger.info("This info message should NOT appear")
    logger.warning("This warning message should appear")
    
    # Reset to INFO
    log_manager.set_all_levels("INFO")
    
    return True

def test_session_logging():
    """Test session logging functionality."""
    print("\nTesting session logging...")
    
    # Create a session logger
    session_name = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_logger = log_manager.create_session_log(session_name)
    
    # Log some session data
    session_logger.info(f"Starting test session: {session_name}")
    session_logger.info("Session parameter 1: value1")
    session_logger.info("Session parameter 2: value2")
    session_logger.info(f"Session completed successfully at {datetime.now()}")
    
    print(f"Session log created: session_{session_name}_*.log")
    
    return True

def test_log_files():
    """Test log file creation and content."""
    print("\nChecking log files...")
    
    # Get a list of all log files
    log_files = log_manager.get_log_files()
    
    if not log_files:
        print("No log files found in the logs directory")
        return False
    
    print(f"Found {len(log_files)} log files:")
    for log_file in log_files:
        print(f"  - {log_file}")
    
    return True

def main():
    """Run the logging tests."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Initialize root logger
    configure_root_logger(level="INFO")
    logger = get_component_logger("test_logging")
    
    print("=" * 60)
    print("LOGGING SYSTEM TEST")
    print("=" * 60)
    
    # Create a list of tests to run
    tests = [
        test_basic_logging,
        test_component_loggers,
        test_exception_logging,
        test_log_levels,
        test_session_logging,
        test_log_files
    ]
    
    # Run all tests
    results = {}
    for test in tests:
        test_name = test.__name__
        logger.info(f"Running test: {test_name}")
        
        try:
            result = test()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"Test {test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"Test {test_name} failed with exception: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        if not result:
            all_passed = False
        print(f"{test_name}: {status}")
    
    print("\nAll tests passed:", "Yes" if all_passed else "No")
    print("\nLog files are available in the 'logs' directory")
    
    logger.info("Logging system test completed")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())