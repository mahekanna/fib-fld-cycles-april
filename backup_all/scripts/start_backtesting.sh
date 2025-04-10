#!/bin/bash

# Start the Fibonacci Cycles Trading System with Advanced Backtesting
# This script provides a convenient way to launch the new backtesting system

# Navigate to project directory
cd "$(dirname "$0")"

# Print welcome message
echo -e "\e[1;34m"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║   Fibonacci Cycles Advanced Backtesting System                    ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "\e[0m"

# Make sure the advanced backtesting script is executable
chmod +x run_advanced_backtesting.sh

# Run the advanced backtesting dashboard
./run_advanced_backtesting.sh

# Exit with the same status as the last command
exit $?