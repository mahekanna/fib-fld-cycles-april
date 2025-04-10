# Centralized Logging System

This document describes the centralized logging system implemented for the Fibonacci Cycle Trading System.

## Overview

The logging system provides a unified approach to logging across all components of the application. It ensures consistent log formatting, log file management, and severity levels throughout the application.

## Key Features

- **Centralized Configuration**: All logging is configured through a single module
- **Component-specific Loggers**: Each component gets its own logger with appropriate namespace
- **Log Rotation**: Log files are automatically rotated to prevent excessive growth
- **Session Logging**: Special session logs can be created for each application run
- **Exception Handling**: Built-in utilities for logging exceptions with full tracebacks
- **Fallback Mechanism**: Components gracefully fall back to standard logging if the centralized system is unavailable

## Usage

### Basic Usage

```python
from utils.logging_utils import get_component_logger

# Get a logger for your component
logger = get_component_logger("component_name")

# Use the logger
logger.info("This is an informational message")
logger.warning("This is a warning")
logger.error("An error occurred")
```

### Logging Exceptions

```python
from utils.logging_utils import get_component_logger, log_exception

logger = get_component_logger("component_name")

try:
    # Some code that might raise an exception
    result = 1 / 0
except Exception as e:
    log_exception(logger, e, "Error during calculation")
```

### Creating Session Logs

```python
from utils.logging_utils import log_manager

# Create a session-specific logger
session_logger = log_manager.create_session_log("backtest_run")
session_logger.info("Starting backtest session")
```

## Log Files

All logs are stored in the `logs/` directory:

- `fib_cycles.log`: Main application log 
- `debug.log`: Debug-level messages
- `error.log`: Error and critical messages
- `component_name.log`: Component-specific logs
- `session_*.log`: Session-specific logs

## Integration with run.py

The `run.py` script automatically initializes the logging system based on command-line arguments. The `--debug` flag will set more verbose logging.

## Advanced Usage

### LogManager Class

For advanced logging needs, you can use the `LogManager` class directly:

```python
from utils.logging_utils import log_manager

# Get recent logs from a component
recent_logs = log_manager.get_recent_logs("scanner", lines=50)

# Set the log level for all components
log_manager.set_all_levels("DEBUG")

# Get a list of all log files
log_files = log_manager.get_log_files()
```

## Best Practices

1. **Use Component Loggers**: Always use `get_component_logger()` to get a logger specific to your component
2. **Appropriate Log Levels**: Use the right severity level for each message:
   - `DEBUG`: Detailed information for diagnosing problems
   - `INFO`: Confirmation that things are working as expected
   - `WARNING`: An indication that something unexpected happened
   - `ERROR`: A serious problem that prevented an operation from completing
   - `CRITICAL`: A very serious error that might prevent the program from continuing
3. **Structured Messages**: Include relevant context in log messages (e.g., symbol names, timeframes)
4. **Exception Logging**: Use `log_exception()` for logging exceptions with full tracebacks