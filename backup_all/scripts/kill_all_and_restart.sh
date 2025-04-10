#!/bin/bash

# Super aggressive script to kill everything, recreate the dashboard from scratch

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== EXTREME CLEANUP: Fibonacci Harmonic Trading System ===${NC}"
echo -e "${RED}Performing extreme cleanup - removing ALL caches and temporary files...${NC}"

# First, kill any existing dashboard processes
echo -e "${YELLOW}Stopping any running dashboards...${NC}"
./kill_dashboard.sh

# Wait to ensure ports are freed
sleep 3
echo ""

# ======= NUCLEAR OPTION - REMOVE ABSOLUTELY EVERYTHING CACHE RELATED =========
echo -e "${RED}NUCLEAR CLEANUP: Removing ALL Python caches, temp files, and assets...${NC}"

# 1. Remove all __pycache__ directories and Python cache files
echo -e "${YELLOW}Removing ALL Python cache files...${NC}"
find / -name "__pycache__" -type d -user $(whoami) -exec rm -rf {} + 2>/dev/null || true
find / -name "*.pyc" -user $(whoami) -delete 2>/dev/null || true
find / -name "*.pyo" -user $(whoami) -delete 2>/dev/null || true
find / -name ".pytest_cache" -type d -user $(whoami) -exec rm -rf {} + 2>/dev/null || true

# 2. Systematically remove all Dash-related caches
echo -e "${YELLOW}Removing ALL Dash cache files...${NC}"
find ~/.dash* -user $(whoami) -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/dash -user $(whoami) -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/flask* -user $(whoami) -exec rm -rf {} + 2>/dev/null || true
find ~ -name ".dash_*" -user $(whoami) -exec rm -rf {} + 2>/dev/null || true
find ~ -name "dash_*" -user $(whoami) -exec rm -rf {} + 2>/dev/null || true

# 3. Remove project-specific caches and temp files
echo -e "${YELLOW}Removing project-specific temporary files...${NC}"
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.lock" -delete 2>/dev/null || true
find . -name "*.bak" -delete 2>/dev/null || true
find . -name ".dash_*" -exec rm -rf {} + 2>/dev/null || true
find . -name ".jupyter*" -exec rm -rf {} + 2>/dev/null || true

# 4. Remove and recreate assets directory
if [ -d "./assets" ]; then
    rm -rf ./assets
fi
mkdir -p ./assets

# 5. Kill ALL Python processes (this is extreme, be careful!)
echo -e "${RED}Killing ALL Python processes belonging to this user...${NC}"
killall -9 python python3 2>/dev/null || true
sleep 2

# 6. Clear any ipython/jupyter caches
if [ -d "~/.ipython" ]; then
    rm -rf ~/.ipython/profile_default/history.sqlite 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}Extreme cleanup complete! ALL caches should be gone.${NC}"
echo ""

# Get port and host from arguments or use defaults
PORT=${1:-8050}  # Default to 8050 if not specified
HOST=${2:-"127.0.0.1"}  # Default to 127.0.0.1 if not specified

echo -e "${GREEN}Starting dashboard with CLEAN environment on http://$HOST:$PORT...${NC}"
echo -e "${YELLOW}-------------------------------------${NC}"

# Add project path to PYTHONPATH to ensure imports work correctly
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Force Python to not use cached imports
export PYTHONDONTWRITEBYTECODE=1

# Run the dashboard with debug mode forced
python -B main_dashboard.py --port $PORT --host $HOST --debug

# Note: The script will wait here while the dashboard runs
# When the dashboard is stopped (e.g., with Ctrl+C), the script will continue

echo -e "${BLUE}Dashboard has been stopped.${NC}"
echo -e "If you still have issues, you might need to restart your terminal session completely."