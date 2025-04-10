#!/bin/bash
# Clean everything and start the dashboard fresh

echo "Cleaning processes and starting fresh..."

# Kill existing dashboard processes
pkill -f "python.*dash"
pkill -f "python.*main_dashboard.py"
pkill -f "python.*run.py"

# Sleep to let processes die
sleep 2

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name '*.pyc' -delete 

# Clear Dash cache
rm -rf ~/.dash_jupyter_hooks 2>/dev/null
rm -rf ~/.dash_cache 2>/dev/null
rm -rf ~/.cache/dash 2>/dev/null
rm -rf ~/.cache/flask-session 2>/dev/null

# Run the fixed dashboard
python start_dashboard_fixed.py