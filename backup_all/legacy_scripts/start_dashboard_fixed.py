#!/usr/bin/env python
"""
Fixed starter script for the Fibonacci Cycle Trading System Dashboard.
This version addresses the duplicate callbacks issue and other problems.
"""

import os
import sys
import time
import logging
import subprocess
import signal
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

def check_ports_in_use(port):
    """Check if a port is in use."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('127.0.0.1', port))
        s.close()
        return False  # Port is available
    except socket.error:
        return True  # Port is in use

def kill_processes_by_port(port):
    """Kill processes using the given port."""
    my_pid = os.getpid()
    try:
        # Try using lsof command
        try:
            logger.info(f"Checking for processes using port {port}...")
            output = subprocess.check_output(f"lsof -i :{port} -t", shell=True).decode().strip()
            if output:
                pids = output.split('\n')
                for pid in pids:
                    try:
                        pid = int(pid.strip())
                        # Skip our own process
                        if pid == my_pid:
                            logger.info(f"Skipping our own process {pid}")
                            continue
                            
                        # Also check if it's a parent process
                        parent = os.getppid()
                        if pid == parent:
                            logger.info(f"Skipping parent process {pid}")
                            continue
                            
                        logger.info(f"Killing process {pid} using port {port}")
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(1)  # Give it time to terminate
                    except Exception as e:
                        logger.warning(f"Error killing process {pid}: {e}")
        except subprocess.SubprocessError:
            logger.warning("lsof command failed, trying alternative approach")
            
        # Alternative: check for dash, python processes - but avoid killing ourselves
        try:
            my_pid = os.getpid()
            parent_pid = os.getppid()
            # Get all python processes in my process group to avoid killing
            my_process_group = []
            try:
                # Try to get process group
                my_process_group = [my_pid, parent_pid]
                pgid = os.getpgid(my_pid)
                logger.info(f"My process group ID: {pgid}")
            except Exception as e:
                logger.warning(f"Error getting process group: {e}")
            
            logger.info("Checking for dash/python processes...")
            for proc_name in ["dash", "python main_dashboard.py", "python run.py"]:
                try:
                    output = subprocess.check_output(f"ps aux | grep '{proc_name}' | grep -v grep", 
                                                   shell=True, stderr=subprocess.DEVNULL).decode()
                    for line in output.split('\n'):
                        if line.strip():
                            try:
                                parts = line.split()
                                pid = int(parts[1])
                                
                                # Comprehensive check to avoid killing our own process tree
                                if pid == my_pid or pid == parent_pid or pid in my_process_group:
                                    logger.info(f"Skipping our process tree: {pid}")
                                    continue
                                    
                                # Extra safety: check if this is python start_dashboard_fixed.py
                                cmdline = ' '.join(parts[10:]).lower() if len(parts) > 10 else ''
                                if 'start_dashboard_fixed.py' in cmdline:
                                    logger.info(f"Skipping our script: {pid} ({cmdline})")
                                    continue
                                
                                logger.info(f"Killing process {pid} ({proc_name})")
                                os.kill(pid, signal.SIGTERM)
                                time.sleep(1)  # Give it time to terminate
                            except Exception as e:
                                logger.warning(f"Error processing line '{line}': {e}")
                except subprocess.SubprocessError as e:
                    logger.warning(f"Error finding {proc_name} processes: {e}")
        except Exception as e:
            logger.warning(f"Alternative process kill approach failed: {e}")
            
    except Exception as e:
        logger.warning(f"Error in kill_processes_by_port: {e}")

def clean_dash_cache():
    """Clean Dash cache to prevent callback issues."""
    try:
        logger.info("Cleaning Dash cache...")
        
        # Clear Python cache files
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
    
    # Configuration
    host = "127.0.0.1"
    port = 8050
    config_path = "config/config.json"
    
    # Pre-clean, just to be sure
    clean_dash_cache()
    
    # Check if port is in use
    if check_ports_in_use(port):
        logger.warning(f"Port {port} is already in use. Killing processes...")
        kill_processes_by_port(port)
        time.sleep(2)  # Give processes time to die
        
        # Check again
        if check_ports_in_use(port):
            logger.error(f"Port {port} is still in use after attempting to kill processes.")
            print(f"\nError: Port {port} is already in use and could not be freed.")
            print("Please close the application using that port and try again.")
            sys.exit(1)
    
    print(f"\nStarting dashboard on http://{host}:{port}")
    print("Press Ctrl+C to stop the application\n")
    
    try:
        # Import here to avoid early importing potentially causing issues
        import main_dashboard
        logger.info("Successfully imported main_dashboard module")
        
        # Now explicitly configure the dashboard environment
        os.environ['DASH_DEBUG'] = "1"  # Enable debug mode
        
        # Run the dashboard
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