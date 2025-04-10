#!/bin/bash

# Simple script to run the backtesting test with the right Python path

# Set colored output for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

echo -e "${BLUE}=== Running Fibonacci Cycle Backtesting Test ===${NC}"

# Make sure we're in the project directory
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

# Clean up any previous test runs
rm -f *_test.png 2>/dev/null

# Set clean paths
export PYTHONPATH="${PROJECT_ROOT}"
echo -e "${BLUE}Project root: ${PROJECT_ROOT}${NC}"

# Create a logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Make sure cache directory exists
mkdir -p "${PROJECT_ROOT}/data/cache"

# Generate mock data for testing if needed
echo -e "${BLUE}Generating mock data for testing...${NC}"
python - << EOF
import os
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
try:
    from data.mock_data_generator import generate_mock_price_data, save_mock_data_to_cache
    
    # Generate data for common symbols
    symbols = ["AAPL", "MSFT", "NIFTY", "RELIANCE", "TCS"]
    exchanges = ["NYSE", "NYSE", "NSE", "NSE", "NSE"]
    intervals = ["daily"]
    
    for sym, ex in zip(symbols, exchanges):
        for interval in intervals:
            try:
                cache_path = os.path.join("${PROJECT_ROOT}", "data", "cache", f"{ex}_{sym}_{interval}.csv") 
                if not os.path.exists(cache_path):
                    print(f"Generating mock data for {sym} on {ex} ({interval})")
                    data = generate_mock_price_data(sym, lookback=500, interval=interval)
                    save_mock_data_to_cache(sym, ex, interval, data)
                else:
                    print(f"Mock data already exists for {sym} on {ex} ({interval})")
            except Exception as e:
                print(f"Error generating mock data for {sym}: {e}")
except Exception as e:
    print(f"Error importing mock data generator: {e}")
EOF

# Verify data sources
echo -e "${BLUE}Checking available data sources...${NC}"
python - << EOF
import os
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
try:
    from data.data_sources_validation import print_data_source_status
    print_data_source_status()
except Exception as e:
    print(f"Error checking data sources: {e}")
EOF

# Ask the user if they want to continue
echo -e "${YELLOW}Do you want to continue with the backtesting test? (y/n)${NC}"
read -r response
if [[ $response =~ ^[Nn]$ ]]; then
    echo -e "${BLUE}Skipping backtesting test. Run ./install_data_dependencies.sh to install required data sources.${NC}"
    exit 0
fi

# Run the test script with logging
echo -e "${BLUE}Running backtesting test script...${NC}"
python test_backtest.py 2>&1 | tee "logs/backtest_test_$(date +%Y%m%d_%H%M%S).log"

# Check if the test completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}Backtesting test completed successfully!${NC}"
    echo -e "${YELLOW}Now try running the dashboard with: ./restart_backtesting.sh${NC}"
else
    echo -e "${RED}Backtesting test failed. Check the logs for details.${NC}"
    echo -e "${YELLOW}You may need to run ./install_data_dependencies.sh first.${NC}"
    
    # Try to help diagnose the issue
    echo -e "${BLUE}Checking for common backtesting issues...${NC}"
    
    # Check if mock data was generated properly
    MOCK_DATA_COUNT=$(find "${PROJECT_ROOT}/data/cache" -name "*.csv" | wc -l)
    if [ "$MOCK_DATA_COUNT" -eq 0 ]; then
        echo -e "${RED}No data files found in cache directory. This is likely the issue.${NC}"
        echo -e "${YELLOW}Attempting to generate mock data...${NC}"
        
        # Generate mock data
        python - << EOF
import os
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
try:
    from data.mock_data_generator import generate_mock_price_data, save_mock_data_to_cache
    
    # Generate data for common symbols
    symbols = ["AAPL", "MSFT", "NIFTY", "RELIANCE", "TCS"]
    exchanges = ["NYSE", "NYSE", "NSE", "NSE", "NSE"]
    intervals = ["daily"]
    
    for sym, ex in zip(symbols, exchanges):
        for interval in intervals:
            try:
                print(f"Generating mock data for {sym} on {ex} ({interval})")
                data = generate_mock_price_data(sym, lookback=500, interval=interval)
                save_mock_data_to_cache(sym, ex, interval, data)
            except Exception as e:
                print(f"Error generating mock data for {sym}: {e}")
except Exception as e:
    print(f"Error importing mock data generator: {e}")
EOF
        
        echo -e "${YELLOW}Try running the test again with: ./run_backtest_test.sh${NC}"
    else
        echo -e "${GREEN}Found ${MOCK_DATA_COUNT} data files in cache, data availability should be OK.${NC}"
        echo -e "${YELLOW}The issue might be with the backtesting engine itself.${NC}"
        echo -e "${YELLOW}Check the logs for more details.${NC}"
    fi
fi