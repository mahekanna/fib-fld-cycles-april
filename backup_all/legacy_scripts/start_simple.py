#!/usr/bin/env python
"""
Simplified starter for the Fibonacci Cycle Trading System Dashboard.
This version avoids process management entirely and simply starts the dashboard.
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
logger = logging.getLogger("start_simple")

def main():
    """Run the dashboard application directly."""
    # Print a banner
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   Fibonacci Cycles Trading System - Simple Dashboard Starter      ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    host = "127.0.0.1"
    port = 8050
    config_path = "config/config.json"
    
    print(f"\nStarting dashboard on http://{host}:{port}")
    print("Press Ctrl+C to stop the application\n")
    
    try:
        # Now just directly import main_dashboard and run
        import main_dashboard
        logger.info("Successfully imported main_dashboard module")
        
        # Set any environment variables needed
        os.environ['DASH_DEBUG'] = "1"  # Enable debug mode
        
        # Run the dashboard directly (no process management)
        main_dashboard.run_app(
            config_path=config_path,
            debug=True,
            port=port,
            host=host
        )
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}", exc_info=True)
        print(f"\nError running dashboard: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()