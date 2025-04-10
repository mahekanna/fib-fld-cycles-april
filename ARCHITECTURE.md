# Fibonacci Harmonic Trading System Architecture

This document maps the system's conceptual components to their actual implementation locations and explains the overall architecture.

## System Architecture Overview

The Fibonacci Harmonic Trading System is built with a modular architecture that follows a domain-driven design approach. The system is organized into the following main directories:

```
fib_cycles_system/
├── core/               # Core analysis engines and algorithms
├── models/             # Data models and structures
├── data/               # Data management and fetching
├── trading/            # Trading strategies and position management
├── visualization/      # Visualization components
├── web/                # Dashboard and web interface
├── storage/            # Results persistence and retrieval
├── utils/              # Utility functions and configuration
└── backtesting/        # Backtesting framework
```

## Conceptual Components to Implementation Mapping

The following table maps conceptual components to their actual implementation locations:

| Conceptual Component | Implementation Location | Description |
|----------------------|-------------------------|-------------|
| Cycle Detection | `core/cycle_detection.py` | FFT-based cycle detection, extremes detection, and harmonic relationship analysis |
| FLD Calculation | `core/fld_signal_generator.py` | Future Line of Demarcation calculation and crossover detection |
| Signal Generation | `core/fld_signal_generator.py` | Multi-cycle signal generation based on FLD crossovers |
| Harmonic Analysis | `trading/harmonic_pattern_utils.py` | Harmonic price pattern detection (Gartley, Butterfly, etc.) |
| Cycle Projection | `core/cycle_detection.py` | Forward projection of detected cycles |
| Position Guidance | `core/fld_signal_generator.py` & `trading/enhanced_entry_exit.py` | Entry, stop, and target recommendations |
| Market Scanning | `core/scanner_system.py` | Symbol scanning and batch analysis |

## Core Components Explained

### Cycle Detection (`core/cycle_detection.py`)
- **Functionality:** Detects dominant market cycles using FFT analysis
- **Key Classes:** `CycleDetector`
- **Key Methods:** 
  - `detect_cycles()` - Identifies dominant cycles in price data
  - `detect_cycle_extremes()` - Finds cycle highs and lows
  - `analyze_harmonic_relationships()` - Analyzes relationships between cycles
  - `generate_cycle_wave()` - Creates synthetic cycle waves (includes projection functionality)

### FLD Calculation and Signal Generation (`core/fld_signal_generator.py`)
- **Functionality:** Calculates FLDs, detects crossovers, generates signals
- **Key Classes:** `FLDCalculator`, `SignalGenerator`
- **Key Methods:**
  - `calculate_fld()` - Computes the Future Line of Demarcation
  - `detect_crossovers()` - Identifies FLD crossovers
  - `generate_signal()` - Creates trading signals based on multiple cycles
  - `generate_position_guidance()` - Provides entry, stop, and target recommendations

### Market Scanner (`core/scanner_system.py`)
- **Functionality:** Orchestrates full market analysis
- **Key Classes:** `FibCycleScanner`
- **Key Methods:**
  - `analyze_symbol()` - Performs complete analysis on a single symbol
  - `scan_batch()` - Analyzes multiple symbols in batch mode

### Market Regime Detection (`core/market_regime_detector.py`)
- **Functionality:** Identifies market regimes (trending, ranging, volatile)
- **Key Classes:** `MarketRegimeDetector`
- **Key Methods:**
  - `detect_regime()` - Detects current market regime
  - `calculate_volatility()` - Analyzes market volatility

### Harmonic Pattern Analysis (`trading/harmonic_pattern_utils.py`)
- **Functionality:** Detects harmonic price patterns
- **Key Classes:** `HarmonicPatternDetector`
- **Key Methods:**
  - `detect_patterns()` - Identifies harmonic patterns
  - `validate_pattern()` - Validates pattern completeness and quality

### Enhanced Entry/Exit Strategy (`trading/enhanced_entry_exit.py`)
- **Functionality:** Advanced entry/exit strategy based on cycle maturity
- **Key Classes:** `EnhancedEntryExitStrategy`
- **Key Methods:**
  - `analyze()` - Performs comprehensive entry/exit analysis
  - `_calculate_cycle_maturity()` - Analyzes cycle completion percentage
  - `_calculate_entry_windows()` - Determines optimal entry windows
  - `_generate_enhanced_guidance()` - Creates enhanced position guidance

## Data Flow

1. **Data Acquisition**: Market data is fetched via `data/data_management.py`
2. **Cycle Analysis**: `CycleDetector` identifies dominant cycles and their harmonics
3. **FLD Analysis**: `FLDCalculator` computes FLDs for each detected cycle
4. **Signal Generation**: `SignalGenerator` combines cycle and FLD data to generate signals
5. **Enhanced Entry/Exit**: `EnhancedEntryExitStrategy` provides optimal entry/exit recommendations
6. **Visualization**: Results are visualized via components in `visualization/` and `web/`
7. **Storage**: Results are stored via `storage/results_repository.py`

## Web Dashboard Components

The dashboard is built with Dash and is structured as follows:

| Dashboard Tab | Implementation | Description |
|---------------|----------------|-------------|
| Analysis Results | `main_dashboard.py` | Summary of analysis results |
| Enhanced Entry/Exit | `web/enhanced_entry_exit_ui.py` | Advanced entry/exit recommendations |
| Cycle Visualization | `web/cycle_visualization.py` | Visualization of detected cycles |
| FLD Analysis | `web/fld_visualization.py` | FLD crossover visualization |
| Harmonic Patterns | `web/harmonic_visualization.py` | Harmonic pattern visualization |
| Scanner Dashboard | `web/scanner_dashboard.py` | Batch scan results dashboard |
| Trading Strategies | `web/trading_strategies_ui.py` | Strategy implementation and backtesting |

## Integration Points

The system integrates various components through:

1. **Models**: `ScanParameters` and `ScanResult` classes in `models/` directory
2. **Configuration**: Configuration management in `utils/config.py`
3. **Results Repository**: Results persistence in `storage/results_repository.py`
4. **Dashboard**: Web interface in `main_dashboard.py` and components in `web/`

## Development Considerations

When extending the system:

1. Follow the existing architecture to maintain consistency
2. Use the models for data exchange between components
3. Add new visualization modules in the `web/` directory
4. Implement new strategies in the `trading/` directory
5. Add docstrings to document new functionality