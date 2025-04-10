#!/bin/bash

# Emergency dashboard restart script to fix Dash callback errors
echo "EMERGENCY DASHBOARD RESTART SCRIPT"
echo "=================================="

# Kill any running dashboard processes
echo "Stopping any running dashboard processes..."
pkill -f main_dashboard.py || true

# Clear browser cache directory to prevent Dash callback conflicts 
if [ -d "$HOME/.dash_jupyter_hooks" ]; then
    echo "Removing Dash Jupyter hooks..."
    rm -rf "$HOME/.dash_jupyter_hooks"
fi

# Remove any temporary Dash files in the .dash_cache directory
if [ -d "$HOME/.dash_cache" ]; then
    echo "Removing Dash cache..."
    rm -rf "$HOME/.dash_cache"
fi

# Clear market data cache
echo "Clearing market data cache..."
rm -f data/cache/market_data.db

# Clear all __pycache__ directories to ensure clean Python imports
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Recreate assets directory to clear any cached assets
if [ -d "./assets" ]; then
    echo "Recreating assets directory..."
    rm -rf ./assets
    mkdir -p ./assets
fi

# Wait a moment for all processes to fully terminate
sleep 2

# Set PYTHONPATH to the project directory
export PYTHONPATH="$(pwd)"

# Start the dashboard with full debug logging
echo "Starting dashboard with debug logging..."
python main_dashboard.py --debug

echo "If the dashboard fails to start, run ./fix_data_cache.sh and try again."