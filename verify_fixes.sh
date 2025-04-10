#!/bin/bash
# Script to verify fixes in the Fibonacci Cycles Trading System

# Display header
echo "============================================"
echo "Fibonacci Cycles Trading System - Fix Verification"
echo "============================================"
echo

# Create a timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Test script
echo "Running test_fixes.py to verify all fixes..."
python test_fixes.py | tee "$LOG_DIR/verify_fixes_$TIMESTAMP.log"
TEST_RESULT=$?

echo
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ All tests completed successfully!"
else
    echo "❌ Some tests failed. Check the log file for details."
    echo "   Log file: $LOG_DIR/verify_fixes_$TIMESTAMP.log"
fi

echo
echo "Checking for available dashboard script..."
if [ -f "main_dashboard.py" ]; then
    echo "✅ Dashboard script found: main_dashboard.py"
    echo "   To run the dashboard: python main_dashboard.py"
else
    echo "❌ Dashboard script not found!"
fi

echo
echo "Checking for available backtesting functionality..."
if [ -d "backtesting" ] && [ -f "backtesting/backtesting_framework.py" ]; then
    echo "✅ Backtesting functionality available"
else
    echo "❌ Backtesting functionality not available or incomplete!"
fi

echo
echo "======================="
echo "System files summary:"
echo "======================="
echo "- Main files:"
ls -la main.py main_dashboard.py 2>/dev/null || echo "  No main files found!"
echo
echo "- Core components:"
ls -la core/ 2>/dev/null || echo "  No core components found!"
echo
echo "- Web components:"
ls -la web/ 2>/dev/null || echo "  No web components found!"
echo

echo "- State Management:"
if [ -f "web/state_manager.py" ]; then
    echo "  ✅ State Manager found: web/state_manager.py"
else
    echo "  ❌ State Manager not found!"
fi
echo
echo "- Backtesting components:"
ls -la backtesting/ 2>/dev/null || echo "  No backtesting components found!"
echo
echo "- Data components:"
ls -la data/ 2>/dev/null || echo "  No data components found!"

echo
echo "======================="
echo "Next steps:"
echo "======================="
echo "1. Run the dashboard: python main_dashboard.py"
echo "2. Verify interval selection works correctly"
echo "3. Verify lookback parameter is respected"
echo "4. Verify state management system by checking price consistency between tabs:"
echo "   - Run a batch scan"
echo "   - Verify that prices and signals are consistent between Batch Results and Batch Advanced Signals tabs"
echo "   - Check for the 🔒 icon in the header indicating Strict Consistency Mode"
echo "5. Read FIXES_SUMMARY.md and STATE_MANAGEMENT_SYSTEM.md for a detailed overview of the fixes"
echo

# Display completion
echo "Verification completed at $(date)"