#!/bin/bash
# Run the simplified demo dashboard

echo "====================================================="
echo "Starting Simplified UI Demo on port 8051"
echo "====================================================="
echo "This demo shows the key UI changes in a minimal version"
echo "that should run without complex dependencies."
echo ""
echo "Open your browser to: http://localhost:8051"
echo ""
echo "CTRL+C to stop the demo"
echo "====================================================="

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Run the simplified dashboard
python simple_dashboard.py