#!/bin/bash

# restart_dashboard.sh - Kills any running instances and starts a fresh dashboard

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Fibonacci Harmonic Trading System ===${NC}"
echo -e "${BLUE}Restarting dashboard...${NC}"

# First, kill any existing dashboard processes
echo -e "${YELLOW}Stopping any running dashboards...${NC}"
./kill_dashboard.sh

# Wait a moment to ensure ports are freed
sleep 2
echo ""

# Clear all caches to ensure a fresh start
echo -e "${BLUE}Clearing all caches for fresh start...${NC}"

# Find and remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# More aggressive cache cleaning for Dash - similar to run_dashboard.sh
if [ -d "$HOME/.dash_jupyter_hooks" ]; then
    echo -e "${BLUE}Removing Dash Jupyter hooks...${NC}"
    rm -rf "$HOME/.dash_jupyter_hooks"
fi

if [ -d "$HOME/.dash_cache" ]; then
    echo -e "${BLUE}Removing Dash cache...${NC}"
    rm -rf "$HOME/.dash_cache"
fi

if [ -d "$HOME/.cache/dash" ]; then
    echo -e "${BLUE}Removing browser dash cache...${NC}"
    rm -rf "$HOME/.cache/dash"
fi

if [ -d "$HOME/.cache/flask-session" ]; then
    echo -e "${BLUE}Removing flask session cache...${NC}"
    rm -rf "$HOME/.cache/flask-session"
fi

# Remove any assets that might be cached
if [ -d "./assets" ]; then
    echo -e "${BLUE}Recreating assets directory...${NC}"
    rm -rf ./assets
    mkdir -p ./assets
fi

echo ""

# Get port and host from arguments
PORT=${1:-8050}  # Default to 8050 if not specified
HOST=${2:-"127.0.0.1"}  # Default to 127.0.0.1 if not specified

echo -e "${GREEN}Starting new dashboard on http://$HOST:$PORT...${NC}"
echo -e "${YELLOW}-------------------------------------${NC}"

# Run the dashboard with the specified port and host, with debug mode enabled
# Debug mode forces Dash to recreate all callbacks
PYTHONPATH=$PYTHONPATH:$(pwd) python main_dashboard.py --port $PORT --host $HOST --debug

# Note: The script will wait here while the dashboard runs
# When the dashboard is stopped (e.g., with Ctrl+C), the script will continue

echo -e "${BLUE}Dashboard has been stopped.${NC}"
echo -e "You can restart it with: ./restart_dashboard.sh [port] [host]"