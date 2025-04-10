'''
Project Structure: Fibonacci Harmonic Trading System
'''

# Core
fib_harmonic/
|
|-- core/
|   |-- __init__.py
|   |-- cycle_detection.py        # Enhanced FFT-based cycle detection
|   |-- fld_calculation.py        # FLD computation and crossover analysis
|   |-- signal_generation.py      # Multi-cycle signal generation
|   |-- harmonic_analysis.py      # Cycle harmonic relationship analysis
|   |-- cycle_projection.py       # Forward projection of cycle patterns
|   |-- position_guidance.py      # Entry, stop, and target recommendations
|   |-- scanner.py                # Main analysis orchestration
|
|-- data/
|   |-- __init__.py
|   |-- fetcher.py                # Multi-source data acquisition
|   |-- processor.py              # Data normalization and preprocessing
|   |-- cache.py                  # Intelligent data caching
|   |-- symbol_manager.py         # Symbol metadata management
|
|-- visualization/
|   |-- __init__.py
|   |-- price_charts.py           # Base price visualization
|   |-- fld_overlays.py           # FLD visualization components
|   |-- cycle_overlays.py         # Cycle pattern visualization
|   |-- signal_markers.py         # Signal and trade visualization
|   |-- projection_charts.py      # Forward projection visualization
|   |-- harmonic_charts.py        # Harmonic relationship visualization
|
|-- web/
|   |-- __init__.py
|   |-- app.py                    # Main Dash application
|   |-- layouts.py                # Dashboard layouts
|   |-- callbacks.py              # Interactive functionality
|   |-- components/               # Reusable UI components
|       |-- __init__.py
|       |-- charts.py
|       |-- sidebar.py
|       |-- scanner_controls.py
|       |-- results_table.py
|
|-- telegram/
|   |-- __init__.py
|   |-- bot.py                    # Telegram bot implementation
|   |-- commands.py               # Command handlers
|   |-- notifications.py          # Alert and notification system
|   |-- image_generator.py        # Chart image generation
|   |-- scheduler.py              # Scheduled reporting
|
|-- models/
|   |-- __init__.py
|   |-- scan_parameters.py        # Analysis parameter models
|   |-- scan_result.py            # Analysis result models
|   |-- cycle_state.py            # Cycle state tracking models
|   |-- signal.py                 # Signal models
|   |-- backtest_trade.py         # Backtesting models
|
|-- storage/
|   |-- __init__.py
|   |-- database.py               # Database connection management
|   |-- results_repository.py     # Results storage and retrieval
|   |-- backtest_repository.py    # Backtest results storage
|   |-- settings_repository.py    # User settings storage
|
|-- backtesting/
|   |-- __init__.py
|   |-- engine.py                 # Backtesting simulation engine
|   |-- performance_metrics.py    # Trading performance calculation
|   |-- optimization.py           # Parameter optimization
|   |-- results_analyzer.py       # Backtest results analysis
|
|-- utils/
|   |-- __init__.py
|   |-- logging.py                # Enhanced logging system
|   |-- config.py                 # Configuration management
|   |-- validators.py             # Input validation utilities
|   |-- math_utils.py             # Mathematical utilities
|
|-- integration/
|   |-- __init__.py
|   |-- tradingview_api.py        # TradingView integration
|   |-- broker_api.py             # Broker API integration
|   |-- export.py                 # Data export functionality
|
|-- config/
|   |-- default_config.json       # Default configuration
|
|-- run.py                        # Main application entry point
|-- run_telegram_bot.py           # Telegram bot entry point
|-- setup.py                      # Package setup file
|-- requirements.txt              # Project dependencies
