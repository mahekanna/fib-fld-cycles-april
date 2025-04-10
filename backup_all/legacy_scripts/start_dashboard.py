#!/usr/bin/env python
"""
Simple starter script for the Fibonacci Cycle Trading System Dashboard.
This is a simplified version that bypasses process management.
"""

import os
import sys
import logging
from datetime import datetime

# Make sure the current directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up basic logging
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("start_dashboard")

# Import required modules for the dashboard
try:
    import main_dashboard
    logger.info("Successfully imported main_dashboard module")
except ImportError as e:
    logger.error(f"Failed to import main_dashboard module: {e}")
    sys.exit(1)

def print_banner():
    """Print a banner for the application."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   █▀▀ █ █▄▄ █▀█ █▄░█ ▄▀█ █▀▀ █▀▀ █   █▀▀ █▄█ █▀▀ █░░ █▀▀ █▀     ║
║   █▀░ █ █▄█ █▄█ █░▀█ █▀█ █▄▄ █▄▄ █   █▄▄ ░█░ █▄▄ █▄▄ ██▄ ▄█     ║
║                                                                   ║
║                 Trading System Dashboard                          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Run the dashboard application."""
    print_banner()
    
    # Print info for the user
    host = "127.0.0.1"
    port = 8050
    print(f"\nStarting dashboard on http://{host}:{port}")
    print("Press Ctrl+C to stop the application\n")
    
    # Run the dashboard
    try:
        main_dashboard.run_app(
            config_path="config/config.json",
            debug=True,
            port=port,
            host=host
        )
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        print(f"\nError running dashboard: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()