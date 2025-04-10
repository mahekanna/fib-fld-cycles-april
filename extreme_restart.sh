#!/bin/bash
# EXTREME CACHE CLEARING AND RESTART
echo "=== EXTREME CACHE CLEARING AND RESTART ==="
echo "This will completely clear all caches and restart the dashboard"

# Kill any running dashboard processes
echo "Stopping all running processes..."
pkill -f "python.*main_dashboard.py" || true

# Clear browser cache directory
echo "Removing Dash browser hooks..."
rm -rf "$HOME/.dash_jupyter_hooks" 2>/dev/null || true
rm -rf "$HOME/.dash_cache" 2>/dev/null || true

# Remove all __pycache__ directories
echo "Clearing all Python cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Clear all .so files
echo "Removing compiled extensions..."
find . -name "*.so" -delete 2>/dev/null || true

# Clear assets directory
echo "Recreating assets directory..."
rm -rf ./assets 2>/dev/null || true
mkdir -p ./assets

# Clear data cache
echo "Clearing data cache..."
rm -f data/cache/market_data.db 2>/dev/null || true

# Clear state file
echo "Clearing state files..."
rm -f data/state_snapshot.pkl 2>/dev/null || true 

# Wait to ensure everything is stopped
echo "Waiting for processes to terminate..."
sleep 2

# Run the force update script
echo "Applying forced UI updates..."
python force_ui_update.py

# Set the PYTHONPATH
export PYTHONPATH="$(pwd)"

# Start with a completely fresh Python interpreter
echo "Starting dashboard with fresh Python interpreter..."
python -W ignore main_dashboard.py

echo "If you still don't see UI changes, try closing and reopening your browser"
