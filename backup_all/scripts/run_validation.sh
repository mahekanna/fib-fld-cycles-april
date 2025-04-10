#!/bin/bash

# Display banner
echo "=================================="
echo "Fibonacci Harmonic Trading System"
echo "Validation Script"
echo "=================================="

# Run the test checker
echo "Validating test modules..."
python check_tests.py

# Check for web module files
echo ""
echo "Validating web UI modules..."
for module in "cycle_visualization" "fld_visualization" "harmonic_visualization" "scanner_dashboard" "trading_strategies_ui"
do
    if [ -f "web/${module}.py" ]; then
        echo "✅ Found web/${module}.py"
    else
        echo "❌ Missing web/${module}.py"
    fi
done

# Check for main dashboard
if [ -f "main_dashboard.py" ]; then
    echo "✅ Found main dashboard at main_dashboard.py"
else
    echo "❌ Missing main dashboard at main_dashboard.py"
fi

# Validate documentation
echo ""
echo "Validating documentation..."
for doc in "DASHBOARD_GUIDE.md" "Technical-Documentation.md" "TEST_RESULTS.md"
do
    if [ -f "docs/${doc}" ]; then
        echo "✅ Found docs/${doc}"
    else
        echo "❌ Missing docs/${doc}"
    fi
done

echo ""
echo "Validation complete!"