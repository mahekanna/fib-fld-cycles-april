#!/bin/bash

# Run script to start the Fibonacci Harmonic Trading System Dashboard
# This script first tests imports, then runs the dashboard if successful

# Set colored output for better readability
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

echo -e "${BLUE}=== Fibonacci Harmonic Trading System ===${NC}"
echo -e "${BLUE}Testing imports before launching dashboard...${NC}"

# Make sure we're in the project directory
cd "$(dirname "$0")"

# Set PYTHONPATH cleanly to current directory only
export PYTHONPATH="$(pwd)"
echo -e "${BLUE}PYTHONPATH set to: $PYTHONPATH${NC}"

# Clear any Dash cache files to prevent callback conflicts
echo -e "${BLUE}Cleaning cache and temporary files...${NC}"
# Find and remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Clear market data cache to ensure correct lookback behavior
echo -e "${BLUE}Clearing market data cache for fresh data fetching...${NC}"
if [ -f "data/cache/market_data.db" ]; then
    echo "Removing market_data.db to ensure lookback parameter is honored..."
    rm -f data/cache/market_data.db
fi

# More aggressive cache cleaning for Dash
# Remove dash_jupyter_hook which can store old callback references
if [ -d "$HOME/.dash_jupyter_hooks" ]; then
    echo -e "${BLUE}Removing Dash Jupyter hooks...${NC}"
    rm -rf "$HOME/.dash_jupyter_hooks"
fi

# Remove any temporary Dash files in the .dash_cache directory
if [ -d "$HOME/.dash_cache" ]; then
    echo -e "${BLUE}Removing Dash cache...${NC}"
    rm -rf "$HOME/.dash_cache"
fi

# Remove any browser cache files from Chrome/Firefox that might be storing Dash assets
echo -e "${BLUE}Removing browser cache files if present...${NC}"
if [ -d "$HOME/.cache/dash" ]; then
    rm -rf "$HOME/.cache/dash"
fi

# Remove any Flask session files that might be causing problems
if [ -d "$HOME/.cache/flask-session" ]; then
    rm -rf "$HOME/.cache/flask-session"
fi

# Remove any assets that might be cached
if [ -d "./assets" ]; then
    echo -e "${BLUE}Recreating assets directory...${NC}"
    rm -rf ./assets
    mkdir -p ./assets
fi

# Run the import test
python test_imports.py
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}=== Starting Dashboard ===${NC}"
    # Run dashboard with the same environment and any command line arguments
    python main_dashboard.py "$@"
else
    echo -e "${RED}=== Dashboard startup aborted due to import errors ===${NC}"
    echo -e "${RED}Please check the error message above and resolve the issues.${NC}"
    echo ""
    echo -e "${YELLOW}Common solutions:${NC}"
    echo "1. Make sure you have activated the conda environment: conda activate fib_cycles"
    echo "2. Install required packages: pip install -r requirements.txt"
    echo "3. Verify that all modules are in the correct directories"
    exit 1
fi