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
