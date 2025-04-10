"""
Centralized logging configuration for the Fibonacci Harmonic Trading System.
This provides a consistent logging setup across all project components.
"""

import os
import sys
import logging
import logging.handlers
import traceback
from datetime import datetime
from typing import Dict, Optional, List, Union, Tuple

# Define log levels dictionary for easy reference
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Default format string for log messages
DEFAULT_LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"

# Global configuration
logs_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
default_log_file = os.path.join(logs_directory, 'fib_cycles.log')
debug_log_file = os.path.join(logs_directory, 'debug.log')
error_log_file = os.path.join(logs_directory, 'error.log')

# Make sure logs directory exists
os.makedirs(logs_directory, exist_ok=True)

# Global log handler registry to avoid duplicates
_log_handlers = {}


def configure_logging(
    level: str = 'INFO',
    component: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    log_format: str = DEFAULT_LOG_FORMAT
) -> logging.Logger:
    """
    Configure logging for a specific component.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        component: Component name (used as logger name)
        log_file: Path to log file (defaults to logs/component_name.log)
        console: Whether to log to console
        file_logging: Whether to log to file
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        log_format: Format string for log messages
    
    Returns:
        Configured logger instance
    """
    # Determine logger name
    logger_name = component or "fib_cycles"
    
    # Get or create logger
    logger = logging.getLogger(logger_name)
    
    # Skip if already configured with handlers
    if logger.handlers:
        return logger
    
    # Set level
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # Determine log file path
    if log_file is None and component is not None:
        log_file = os.path.join(logs_directory, f"{component.lower().replace('.', '_')}.log")
    elif log_file is None:
        log_file = default_log_file
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Configure file handler if requested
    if file_logging:
        # Use RotatingFileHandler for log rotation
        try:
            # Check if we already have a handler for this file
            if log_file in _log_handlers:
                file_handler = _log_handlers[log_file]
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                file_handler.setFormatter(formatter)
                _log_handlers[log_file] = file_handler
            
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, log to console as fallback
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.error(f"Failed to set up file logging: {e}")
    
    return logger


def configure_root_logger(level: str = 'INFO'):
    """
    Configure the root logger for the application.
    
    Args:
        level: Logging level for the root logger
    """
    # Configure the root logger
    root_logger = configure_logging(
        level=level,
        component=None,  # Root logger
        log_file=default_log_file,
        console=True,
        file_logging=True
    )
    
    # Install exception hook to log unhandled exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        root_logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = exception_handler


def get_logger(component: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    If the logger doesn't exist, it will be created with default settings.
    
    Args:
        component: Component name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(component)
    
    # If logger has no handlers, configure it with defaults
    if not logger.handlers:
        return configure_logging(component=component)
        
    return logger


def get_component_loggers() -> Dict[str, logging.Logger]:
    """
    Get a dictionary of all configured loggers in the project.
    
    Returns:
        Dictionary mapping logger names to logger instances
    """
    return {name: logging.getLogger(name) for name in logging.root.manager.loggerDict}


def log_exception(logger: logging.Logger, exception: Exception, message: str = "An error occurred:"):
    """
    Log an exception with full traceback information.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        message: Message to prefix the exception
    """
    logger.error(f"{message} {str(exception)}")
    logger.debug(f"Exception traceback:", exc_info=True)


def set_log_level(component: Optional[str] = None, level: str = 'INFO'):
    """
    Set the log level for a specific component or the root logger.
    
    Args:
        component: Component name (None for root logger)
        level: Logging level
    """
    logger = logging.getLogger(component) if component else logging.getLogger()
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))


def create_component_logger(component_name: str, level: str = 'INFO') -> logging.Logger:
    """
    Create a logger for a specific component with standard configuration.
    
    Args:
        component_name: Name of the component
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    return configure_logging(
        level=level,
        component=component_name
    )


class LogManager:
    """Manager class for handling multiple loggers across the application."""
    
    def __init__(self):
        """Initialize the log manager."""
        self.loggers = {}
        self.default_level = 'INFO'
        
        # Ensure logs directory exists
        os.makedirs(logs_directory, exist_ok=True)
        
        # Configure root logger
        configure_root_logger(self.default_level)
    
    def get_logger(self, component: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get a logger for a specific component.
        
        Args:
            component: Component name
            level: Logging level (uses default if None)
            
        Returns:
            Logger instance
        """
        if component not in self.loggers:
            self.loggers[component] = configure_logging(
                level=level or self.default_level,
                component=component
            )
        
        return self.loggers[component]
    
    def set_default_level(self, level: str):
        """
        Set the default log level for new loggers.
        
        Args:
            level: New default log level
        """
        if level.upper() in LOG_LEVELS:
            self.default_level = level.upper()
    
    def set_all_levels(self, level: str):
        """
        Set the log level for all existing loggers.
        
        Args:
            level: New log level
        """
        level_value = LOG_LEVELS.get(level.upper(), logging.INFO)
        
        # Update all managed loggers
        for logger in self.loggers.values():
            logger.setLevel(level_value)
            
        # Also update root logger
        logging.getLogger().setLevel(level_value)
    
    def get_log_files(self) -> List[str]:
        """
        Get a list of all log files.
        
        Returns:
            List of log file paths
        """
        try:
            return [f for f in os.listdir(logs_directory) if f.endswith('.log')]
        except Exception:
            return []
    
    def get_recent_logs(self, component: Optional[str] = None, lines: int = 100) -> List[str]:
        """
        Get recent log entries from a component log file.
        
        Args:
            component: Component name (uses root logger if None)
            lines: Number of lines to retrieve
            
        Returns:
            List of recent log lines
        """
        log_file = os.path.join(logs_directory, f"{component.lower()}.log") if component else default_log_file
        
        try:
            if not os.path.exists(log_file):
                return [f"Log file not found: {log_file}"]
                
            with open(log_file, 'r') as f:
                # Read all lines and get the last 'lines' entries
                all_lines = f.readlines()
                return all_lines[-lines:] if lines < len(all_lines) else all_lines
        except Exception as e:
            return [f"Error reading log file: {str(e)}"]
    
    def create_session_log(self, session_name: str) -> logging.Logger:
        """
        Create a dedicated logger for a specific session.
        
        Args:
            session_name: Session name or identifier
            
        Returns:
            Logger instance
        """
        # Format session name for filename (remove special characters)
        safe_name = ''.join(c if c.isalnum() else '_' for c in session_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_file = os.path.join(logs_directory, f"session_{safe_name}_{timestamp}.log")
        
        logger = configure_logging(
            level=self.default_level,
            component=f"session.{safe_name}",
            log_file=session_log_file
        )
        
        return logger


# Create a global log manager instance
log_manager = LogManager()

# Convenience function to get a logger
def get_component_logger(component: str, level: Optional[str] = None) -> logging.Logger:
    """Get a logger for a specific component using the global log manager."""
    return log_manager.get_logger(component, level)