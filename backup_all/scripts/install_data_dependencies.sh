#!/bin/bash

# Install required data packages for backtesting
# This script installs the necessary packages for fetching market data

# Set colored output for better readability
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

echo -e "${BLUE}=== Installing Data Dependencies for Fibonacci Cycle Trading System ===${NC}"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed or not in PATH${NC}"
    exit 1
fi

# Install yfinance
echo -e "${BLUE}Installing yfinance...${NC}"
pip install yfinance

# Install tvDatafeed
echo -e "${BLUE}Installing tvDatafeed...${NC}"
pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

# Install alpha_vantage
echo -e "${BLUE}Installing alpha_vantage...${NC}"
pip install alpha_vantage

echo -e "${GREEN}All data dependencies have been installed successfully!${NC}"
echo -e "${YELLOW}Note: For TradingView data, you may need to set up credentials in the config file.${NC}"
echo -e "${YELLOW}For Alpha Vantage, you need to get an API key and add it to your config.${NC}"

# Create or update a documentation file with instructions
cat > data_sources_setup.md << 'EOF'
# Setting Up Data Sources for Backtesting

## Available Data Sources

The backtesting system can use the following data sources:

1. **TradingView** - Provides comprehensive market data for many exchanges
2. **Yahoo Finance** - Free data source for most US stocks and some international symbols
3. **Alpha Vantage** - API-based data source with both free and paid tiers
4. **Local CSV Files** - Can use previously downloaded data stored locally

## Configuration

### TradingView Setup

If you have a TradingView account:

1. Edit the config file at `config/config.json` 
2. Add your TradingView credentials:
   ```json
   "tradingview": {
     "username": "your_username",
     "password": "your_password"
   }
   ```

### Alpha Vantage Setup

1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Add it to your config file:
   ```json
   "alpha_vantage": {
     "api_key": "your_api_key"
   }
   ```

### Using Local CSV Files

For offline usage or to avoid rate limits:

1. Place CSV files in the `data/cache/` directory
2. Use the following naming format: `EXCHANGE_SYMBOL_INTERVAL.csv`
   - Example: `NSE_RELIANCE_daily.csv` or `NYSE_AAPL_weekly.csv`
3. CSV files should have columns: date/timestamp, open, high, low, close, volume
4. Date/timestamp should be in a standard format like YYYY-MM-DD

## Troubleshooting

If you're having issues with data sources:

1. Check the logs for specific error messages
2. Verify your internet connection
3. Make sure your API keys and credentials are correct
4. Some exchanges may restrict data access based on your location

For Indian markets (NSE/BSE), TradingView tends to be the most reliable source.
EOF

echo -e "${GREEN}Created documentation file: data_sources_setup.md${NC}"