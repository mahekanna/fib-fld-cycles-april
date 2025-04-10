#!/bin/bash

# Run the dashboard with the advanced backtesting component
# This script starts the main dashboard with improved backtesting capabilities

# Navigate to project directory
cd "$(dirname "$0")"

# Ensure environment is set up
if [ -f "fib_cycles.pth" ]; then
    echo "Using project Python path file for imports"
    export PYTHONPATH="$(pwd):$PYTHONPATH"
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Create assets directory if it doesn't exist
mkdir -p assets

# Start the dashboard
echo "Starting Fibonacci Cycles Trading System with Advanced Backtesting..."
python3 main_dashboard.py --debug --host 127.0.0.1 --port 8050