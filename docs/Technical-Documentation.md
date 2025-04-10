# Fibonacci Harmonic Trading System - Technical Documentation

## System Overview

The Fibonacci Harmonic Trading System is a comprehensive market analysis and trading platform that combines cycle detection using Fast Fourier Transform (FFT) with Fibonacci relationships to identify potential market turning points. The system analyzes price data across multiple timeframes, detects dominant cycles, calculates Future Lines of Demarcation (FLD), and generates trading signals based on cycle alignment and harmonic patterns.

## Project Architecture

The project follows a modular architecture with clean separation of concerns:

### Core Components

1. **Cycle Detection (`core/cycle_detection.py`)**:
   - Implements FFT-based cycle detection algorithms
   - Identifies dominant cycles in price data
   - Analyzes harmonic relationships between cycles
   - Includes detrending and filtering for better signal quality

2. **FLD Signal Generator (`core/fld_signal_generator.py`)**:
   - Calculates Future Lines of Demarcation based on detected cycles
   - Implements crossover detection and signal generation
   - Includes gap detection and state tracking for better signal quality

3. **Scanner System (`core/scanner_system.py`)**:
   - Orchestrates cycle detection and signal generation
   - Manages the scanning of multiple symbols
   - Filters and ranks results based on configurable criteria

4. **Market Regime Detector (`core/market_regime_detector.py`)**:
   - Identifies different market regimes (trending, ranging, volatile)
   - Adjusts signal generation based on detected regime
   - Provides regime-specific trading parameters

### Data Management

1. **Data Management (`data/data_management.py`)**:
   - Fetches data from multiple sources (TradingView, Yahoo Finance)
   - Implements data processing and normalization
   - Manages data caching to improve performance
   - Supports multiple timeframes and data sources

### Models

1. **Model Definitions (`models/model_definitions.py`)**:
   - Defines data structures used throughout the system
   - Provides typed models for parameters, results, and configurations
   - Includes default configuration generation

### Trading Strategies

1. **Base Trading Strategies (`trading/trading_strategies.py`)**:
   - Implements base `TradingStrategy` class with common functionality
   - Provides specialized strategy classes:
     - `FibonacciCycleStrategy`: Basic cycle-based strategy
     - `MultiTimeframeHarmonicStrategy`: Combines signals from multiple timeframes
     - `MLEnhancedCycleStrategy`: Adds ML filtering for signal improvement

2. **Practical Trading Strategies (`trading/practical_strategies.py`)**:
   - Contains concrete strategy implementations:
     - `AdvancedFibonacciStrategy`: Complex cycle and FLD based strategy
     - `SwingTradingStrategy`: Optimized for multi-day holding periods
     - `DayTradingStrategy`: Optimized for intraday trades
     - `HarmonicPatternStrategy`: Focuses on harmonic price patterns

3. **Harmonic Pattern Utility (`trading/harmonic_pattern_utils.py`)**:
   - Implements the `HarmonicPatternDetector` class
   - Detects and analyzes harmonic price patterns:
     - Gartley patterns
     - Butterfly patterns
     - Bat patterns
     - Crab patterns
   - Calculates pattern quality and completion metrics
   - Generates entry, stop loss, and target levels

### Backtesting

1. **Backtesting Framework (`backtesting/backtesting_framework.py`)**:
   - Provides infrastructure for strategy testing
   - Simulates trades based on historical data
   - Calculates performance metrics
   - Generates performance reports

### Visualization

1. **Visualization System (`visualization/visualization_system.py`)**:
   - Generates comprehensive price charts with cycle overlays
   - Visualizes detected cycles, FLDs, and signals
   - Provides multiple chart types and visualization options

### Web Interface

1. **Dashboard Implementation (`web/dashboard_implementation.py`)**:
   - Provides interactive web dashboard for system monitoring
   - Displays real-time signals and analysis
   - Includes configurable views and filters

### Integration

1. **Broker Integration (`integration/broker_integration.py`)**:
   - Connects to brokerage APIs
   - Implements order placement and management
   - Handles position tracking and updates

2. **Telegram Bot (`telegram/telegram_bot.py`)**:
   - Provides alert notifications via Telegram
   - Implements command handling for remote monitoring
   - Delivers signal and performance updates

### ML Enhancements

1. **ML Enhancements (`ml/ml_enhancements.py`)**:
   - Implements machine learning models for signal enhancement
   - Adds anomaly detection to identify abnormal market behavior
   - Includes regime classification for adaptive trading
   - Uses ensemble methods for improved signal quality

### Utilities

1. **Report Generation (`utils/report_generation.py`)**:
   - Creates HTML and PDF reports of analysis results
   - Generates performance metrics and visualizations
   - Provides custom reporting templates

## System Entry Points

The main entry point is `main.py`, which supports multiple operation modes:

1. **Scan Mode**:
   - Analyzes specified symbols to generate trading signals
   - Filters and ranks results based on configuration
   - Generates reports with signals and visualizations

2. **Backtest Mode**:
   - Tests strategies against historical data
   - Calculates performance metrics
   - Generates detailed backtest reports

3. **Dashboard Mode**:
   - Launches interactive web dashboard
   - Provides real-time monitoring and analysis
   - Supports user-configurable views and filters

## Configuration

The system uses a comprehensive configuration system (`config/config.json`) that includes:

1. **General Settings**:
   - Default exchange and data source
   - Symbol configuration

2. **Data Settings**:
   - Cache directory and expiry timeframes
   - Database configuration

3. **Analysis Parameters**:
   - Cycle detection parameters
   - Fibonacci cycle values
   - Processing options

4. **Scanner Settings**:
   - Default symbols and exchanges
   - Filtering and ranking criteria

5. **Backtest Settings**:
   - Testing date ranges
   - Initial capital and position sizing
   - Transaction costs

6. **Visualization Settings**:
   - Theme and color palette
   - Chart dimensions and styling

7. **Telegram Settings**:
   - Bot configuration
   - Notification preferences

8. **Web Dashboard Settings**:
   - Host and port configuration
   - Refresh intervals

## Key Functionalities

### Cycle Detection & Analysis

- Identifies dominant market cycles using Fast Fourier Transform
- Detects harmonic relationships between cycles
- Analyzes cycle strength and alignment for signal generation

### FLD Calculation & Signal Generation

- Calculates Future Lines of Demarcation based on cycle lengths
- Detects meaningful crossovers of price and FLDs
- Generates trading signals with confidence metrics

### Harmonic Pattern Recognition

- Identifies common harmonic patterns: Gartley, Butterfly, Bat, and Crab
- Calculates pattern quality and completion metrics
- Generates entry, stop, and target levels based on patterns

### Multiple Trading Strategies

- Base strategies for different trading approaches (swing, day, harmonic)
- Multi-timeframe strategies for comprehensive analysis
- ML-enhanced strategies for improved signal quality

### Backtesting & Performance Analysis

- Comprehensive backtesting framework for strategy validation
- Position sizing and risk management
- Performance metrics calculation and reporting

### Visualization & Reporting

- Advanced charting with cycle and FLD overlays
- Interactive dashboard for real-time monitoring
- Detailed reporting for analysis and backtesting

### External Integrations

- Broker connections for live trading
- Telegram notifications for alerts and monitoring
- Data sources: TradingView, Yahoo Finance

## Technical Implementation Details

1. **FFT Analysis**:
   - Uses SciPy's FFT implementation for cycle detection
   - Applies detrending and windowing for improved results
   - Filters results based on power thresholds

2. **Fibonacci Relationships**:
   - Focuses on key Fibonacci cycles: 21, 34, 55, 89, 144, 233
   - Analyzes harmonic relationships between cycles
   - Uses Fibonacci ratios for pattern recognition

3. **Data Processing**:
   - Implements caching to improve performance
   - Supports multiple data sources with adapter pattern
   - Includes normalization and feature calculation

4. **Signal Generation**:
   - Combines cycle alignment, strength, and confidence
   - Uses multi-factor approach for improved quality
   - Includes regime-specific adjustments

5. **Pattern Recognition**:
   - Implements Fibonacci retracement analysis
   - Calculates pattern quality based on ratio matching
   - Provides entry, stop, and target recommendations

## Conclusion

The Fibonacci Harmonic Trading System provides a comprehensive framework for market analysis, signal generation, and trading automation. Its modular architecture allows for easy extension and customization, while its advanced technical analysis capabilities provide deep market insights based on cyclic analysis and Fibonacci relationships.