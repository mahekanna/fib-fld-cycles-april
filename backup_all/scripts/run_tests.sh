#!/bin/bash

# Display banner
echo "=================================="
echo "Fibonacci Harmonic Trading System"
echo "Test Runner"
echo "=================================="

# Run the tests
echo "Running tests for web UI modules..."
python -m pytest -xvs tests/web/

echo ""
echo "Test summary:"
echo "=================================="
python -m pytest -v tests/web/ --no-header --no-summary -q