#!/bin/bash

# Completely kill all Dash caches and restart the dashboard
# This is a more thorough solution when you encounter duplicate output errors

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Fibonacci Harmonic Trading System ===${NC}"
echo -e "${BLUE}Performing complete cache clear and restart...${NC}"

# First, kill any existing dashboard processes
echo -e "${YELLOW}Stopping any running dashboards...${NC}"
./kill_dashboard.sh

# Wait a moment to ensure ports are freed
sleep 3
echo ""

# Deep system cache cleaning - much more aggressive
echo -e "${RED}Performing deep cache cleanup...${NC}"

# 1. Remove all __pycache__ directories
echo -e "${YELLOW}Removing Python cache files...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete || true
find . -name "*.pyo" -delete || true

# 2. Remove all Dash-related cache directories
echo -e "${YELLOW}Removing Dash cache files...${NC}"
if [ -d "$HOME/.dash_jupyter_hooks" ]; then
    rm -rf "$HOME/.dash_jupyter_hooks"
fi

if [ -d "$HOME/.dash_cache" ]; then
    rm -rf "$HOME/.dash_cache"
fi

if [ -d "$HOME/.cache/dash" ]; then
    rm -rf "$HOME/.cache/dash"
fi

# 3. Remove Flask session files
echo -e "${YELLOW}Removing Flask session files...${NC}"
if [ -d "$HOME/.cache/flask-session" ]; then
    rm -rf "$HOME/.cache/flask-session"
fi

# 4. Remove all assets - dashboards create these at runtime
echo -e "${YELLOW}Cleaning assets directory...${NC}"
if [ -d "./assets" ]; then
    rm -rf ./assets
fi
mkdir -p ./assets

# 5. Remove temporary files in the current directory
echo -e "${YELLOW}Removing temporary files...${NC}"
rm -f ./*.tmp
rm -f ./*.lock
rm -f ./*.bak

# 6. If dot_dash_cache exists in current directory, remove it
if [ -d "./.dash_cache" ]; then
    rm -rf ./.dash_cache
fi

# 7. Check for any stray Python processes and kill them
echo -e "${YELLOW}Checking for stray Python processes...${NC}"
python_processes=$(ps aux | grep -i "[p]ython.*main_dashboard" | awk '{print $2}')
if [ ! -z "$python_processes" ]; then
    echo -e "${YELLOW}Killing stray Python processes: $python_processes${NC}"
    kill -9 $python_processes 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}Cache cleanup complete!${NC}"
echo ""

# Get port and host from arguments or use defaults
PORT=${1:-8050}  # Default to 8050 if not specified
HOST=${2:-"127.0.0.1"}  # Default to 127.0.0.1 if not specified

echo -e "${GREEN}Starting dashboard with clean environment on http://$HOST:$PORT...${NC}"
echo -e "${YELLOW}-------------------------------------${NC}"

# Add project path to PYTHONPATH to ensure imports work correctly
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Run the dashboard in debug mode to force reloading of all components
python main_dashboard.py --port $PORT --host $HOST --debug

# Note: The script will wait here while the dashboard runs
# When the dashboard is stopped (e.g., with Ctrl+C), the script will continue

echo -e "${BLUE}Dashboard has been stopped.${NC}"
echo -e "If you encounter more issues, use this script again to restart with a clean cache."