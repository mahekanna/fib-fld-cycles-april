# Fibonacci Harmonic Trading System - Dashboard Guide

## Environment Setup

### Creating a Conda Environment

1. Install Miniconda or Anaconda if you haven't already.

2. Create a new conda environment for the project:
   ```bash
   conda create -n fib_cycles python=3.9
   ```

3. Activate the environment:
   ```bash
   conda activate fib_cycles
   ```

### Installing Dependencies

1. Install base requirements:
   ```bash
   conda install -c conda-forge numpy pandas scipy matplotlib plotly
   conda install -c conda-forge dash dash-bootstrap-components jupyter
   ```

2. Install TradingView data feed:
   ```bash
   pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
   ```

3. Install additional dependencies:
   ```bash
   pip install scikit-learn statsmodels ta-lib
   pip install pytest pytest-cov
   pip install pytz requests
   ```

4. Add the project to Python path:
   ```bash
   pip install -e .
   ```

## Running the Dashboard

### Starting the Dashboard

1. Ensure your conda environment is activated:
   ```bash
   conda activate fib_cycles
   ```

2. Navigate to the project root directory:
   ```bash
   cd /home/vijji/advanced_fld_cycles/fib_cycles_system/
   ```

3. Run the dashboard application:
   ```bash
   python main_dashboard.py
   ```

4. Access the dashboard in your web browser at `http://127.0.0.1:8050`

## Features

The dashboard integrates five specialized visualization modules:

1. **Cycle Visualization Module**: View detected market cycles, power spectrum, and cycle alignment metrics
2. **FLD Analysis Module**: Analyze FLD signals, crossovers, and strength indicators
3. **Harmonic Pattern Visualization**: Identify and analyze harmonic patterns with Fibonacci ratios
4. **Scanner Results Dashboard**: Review scan results across multiple symbols with filtering and sorting
5. **Trading Strategies Module**: Configure, test, and monitor various trading strategies

## Using the Dashboard

### Analysis Parameters

Configure your analysis using the sidebar controls:

- **Symbol**: Enter the trading symbol (e.g., NIFTY, RELIANCE)
- **Exchange**: Specify the exchange (e.g., NSE, BSE)
- **Interval**: Select the timeframe for analysis (daily, 4h, 1h, 15m)
- **Lookback**: Number of historical bars to include
- **Number of Cycles**: How many dominant cycles to detect (1-5)
- **Price Source**: Select which price data to use (close, HLC3, etc.)

Click the **Scan** button to analyze the current symbol.

### Batch Analysis

For analyzing multiple symbols at once:

1. Enter symbols in the text area (one per line)
2. Configure general parameters on the left sidebar
3. Click **Batch Scan** to start the analysis
4. View results in the **Batch Results** and **Scanner Dashboard** tabs

### Navigation

Use the tabs at the top of the main panel to navigate between different visualization modules:

- **Analysis Results**: Summary of scan results and main metrics
- **Cycle Visualization**: Interactive charts of detected cycles and FFT analysis
- **FLD Analysis**: View FLD lines and crossover signals
- **Harmonic Patterns**: Visualize detected harmonic patterns and their measurements
- **Scanner Dashboard**: Interactive dashboard for reviewing scan results
- **Trading Strategies**: Configure and monitor trading strategies
- **Batch Results**: Results table from batch scanning operations

## Troubleshooting

If you encounter issues:

1. Check that all dependencies are correctly installed:
   ```bash
   conda list
   ```

2. If TA-Lib installation fails, you might need to install the C dependencies first:
   
   For Ubuntu/Debian:
   ```bash
   sudo apt-get install build-essential
   sudo apt-get install libta-lib-dev
   ```
   
   For macOS:
   ```bash
   brew install ta-lib
   ```
   
   Then try reinstalling:
   ```bash
   pip install --no-binary ta-lib ta-lib
   ```

3. Ensure your configuration file exists and is properly formatted
4. Verify that data sources are accessible
5. Check browser console for JavaScript errors

## Data Setup

### TradingView Data Feed Configuration

The system uses TradingView data for market analysis. Set up your credentials:

1. Create a file named `tv_credentials.json` in your config directory:
   ```bash
   mkdir -p /home/vijji/advanced_fld_cycles/fib_cycles_system/config
   nano /home/vijji/advanced_fld_cycles/fib_cycles_system/config/tv_credentials.json
   ```

2. Add your TradingView login details:
   ```json
   {
     "username": "your_tradingview_username",
     "password": "your_tradingview_password"
   }
   ```

3. Ensure this file is referenced correctly in your main configuration file.

## Further Development

To extend the dashboard:

1. Add new visualization modules in the `fib_cycles_system/web/` directory
2. Register new modules in `main_dashboard.py`
3. Add corresponding tabs and callbacks

## Support

For questions or issues, please refer to the technical documentation or contact the development team.