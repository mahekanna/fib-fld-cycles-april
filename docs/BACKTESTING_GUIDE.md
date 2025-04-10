# Advanced Backtesting System Guide

This document provides a comprehensive guide to using the advanced backtesting system in the Fibonacci Cycles Trading System.

## Overview

The Advanced Backtesting System allows you to test trading strategies against historical market data to evaluate their performance. It provides detailed performance metrics, trade logs, equity curves, and other analytical tools to help you optimize your trading strategies.

## Features

- **Comprehensive Backtesting Framework**: Test strategies with detailed metrics and visualizations
- **Multiple Strategy Support**: Fibonacci Cycle, Harmonic Patterns, FLD Crossover, and more
- **Advanced Position Sizing**: Control your risk with percentage-based or fixed position sizing
- **Risk Management Tools**: Set stop-loss, take-profit, and trailing stop parameters
- **Detailed Analytics**: Get key metrics like Sharpe ratio, drawdown, and win rate
- **Trade Logging**: Review every trade with entry/exit prices, profit/loss, and reason
- **Performance Reporting**: Monthly returns, drawdown analysis, and more
- **Seamless Integration**: Access backtesting directly from the scanner dashboard

## Accessing the Backtesting System

You can access the backtesting system in two ways:

1. **From the main dashboard**: Click on the "Backtesting" tab
2. **From the scanner dashboard**: Click the "Backtest" button on any symbol in the scan results

## Configuration Parameters

The backtesting system offers a wide range of configuration options:

### Market & Symbol

- **Exchange**: Select the market (NSE, BSE, NYSE, NASDAQ)
- **Symbol**: Enter the trading symbol (e.g., NIFTY, RELIANCE)
- **Timeframe**: Select the data timeframe (1m, 5m, 15m, 30m, 1h, 4h, daily, weekly, monthly)

### Backtest Period

- **Start Date**: The beginning date for the backtest
- **End Date**: The ending date for the backtest

### Strategy Configuration

- **Trading Strategy**: Choose from Fibonacci Cycle, Harmonic Patterns, FLD Crossover, Multi-Timeframe, or Enhanced Entry/Exit

### Capital & Position Sizing

- **Initial Capital**: Starting capital for the backtest
- **Position Size (%)**: Percentage of capital to allocate per trade
- **Max Open Positions**: Maximum number of concurrent positions
- **Pyramiding**: Number of additional entries allowed for the same signal

### Entry Conditions

- **Signal Strength Threshold**: Minimum strength required for entry
- **Require Cycle Alignment**: Whether to require cycles to be aligned

### Risk Management

- **Take Profit Multiple**: Risk-to-reward ratio for take-profit
- **Use Trailing Stop**: Enable/disable trailing stop
- **Trailing Stop (%)**: Percentage for trailing stop

## Backtest Results

After running a backtest, you'll see detailed results:

### Summary Metrics

- **Total Return**: Overall percentage return
- **CAGR**: Compound Annual Growth Rate
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profits divided by gross losses
- **Max Drawdown**: Maximum percentage decline from peak
- **Sharpe Ratio**: Risk-adjusted return

### Performance Charts

- **Equity Curve**: Visual representation of account growth over time
- **Drawdown Chart**: Visual representation of drawdowns over time
- **Monthly Returns Heatmap**: Returns by month and year
- **Trade Distribution**: Statistical distribution of trade outcomes

### Trade History

A table showing all trades with the following information:
- Entry and exit dates
- Entry and exit prices
- Profit/loss amount and percentage
- Exit reason

### Detailed Metrics

- Statistical analysis of performance
- Risk metrics
- Return metrics
- Comparison to benchmarks

## Exporting Results

You can export backtest results in various formats:
- CSV
- Excel
- JSON
- PDF Report

## Advanced Usage

### Comparing Strategies

Use the "Compare Strategies" button to run multiple strategy configurations against the same symbol and compare their performance.

### Saving Configurations

Save your backtest configurations for later use with the "Save Configuration" button.

## Running from Command Line

To run the dashboard with advanced backtesting:

```bash
./run_advanced_backtesting.sh
```

## Troubleshooting

If you encounter issues:

1. Check that you have all required data for the selected symbol and timeframe
2. Verify that the date range has sufficient data
3. Check the logs for detailed error messages

## Integration with Scanner Dashboard

The "Backtest" button in the scanner dashboard allows you to quickly run backtests on signals identified by the scanner. When clicked, it will:

1. Open the advanced backtesting UI
2. Pre-populate the configuration with the symbol and analysis parameters
3. Allow you to customize the backtest and run it

This integration provides a seamless workflow from market scanning to strategy validation.