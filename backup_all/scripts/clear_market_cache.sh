#!/bin/bash

# Script to clear market data cache to refresh data and honor lookback parameter
echo "Clearing market data cache for fresh data fetching..."

# Make sure we're in the project directory
cd "$(dirname "$0")"

# Remove market_data.db database
if [ -f "data/cache/market_data.db" ]; then
    echo "Removing market_data.db to ensure lookback parameter is honored..."
    rm -f data/cache/market_data.db
    echo "Cache cleared successfully!"
else
    echo "No cache file found at data/cache/market_data.db"
fi

# Optionally remove CSV files if needed
read -p "Do you want to also remove CSV cache files? (y/n): " answer
if [[ $answer == "y" || $answer == "Y" ]]; then
    echo "Removing CSV cache files..."
    rm -f data/cache/*.csv
    echo "All cache files removed!"
fi

echo "Done! The next scan will fetch fresh data and honor the lookback parameter."