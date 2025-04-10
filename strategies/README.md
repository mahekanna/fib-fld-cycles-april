# Advanced Trading Strategies for Fibonacci Cycles System

This directory contains implementations of the advanced trading strategies documented in `Advanced-Trading-Strategies-Documentation-for-Fibonacci-Cycles-System.txt`. These strategies leverage cycle detection, FLD (Future Line of Demarcation) analysis, and harmonic relationships to generate trading signals.

## Strategy Implementations

The following strategies have been implemented:

1. **Rapid Cycle FLD Strategy** (`rapid_cycle_fld_strategy.py`)
   - Fast-moving strategy focusing on the shortest detected cycle
   - Uses FLD crossovers with cycle alignment confirmation
   - Designed for intraday trading on 15-minute and 1-hour timeframes
   - Implements tight stop-losses at 0.3 x cycle amplitude
   - Targets 1:2 risk-reward minimum

2. **Multi-Cycle Confluence Strategy** (`multi_cycle_confluence_strategy.py`) 
   - Identifies when multiple cycle FLDs align in the same direction
   - Enters on retracements to the primary FLD
   - Places stops beyond recent cycle extremes
   - Targets next projected cycle turn
   - Optimal for range-bound markets with clear cyclical behavior

3. **Turning Point Anticipation Strategy** (`turning_point_anticipation_strategy.py`)
   - Leverages projected cycle turns to anticipate market reversals
   - Monitors approaching projected cycle turns from multiple timeframes
   - Confirms reversals with price action patterns
   - Holds through cycle duration with trailing stops
   - Designed for swing trading

4. **Cycle Phase Trading Strategy** (`cycle_phase_trading_strategy.py`)
   - Sophisticated approach to trading different phases of identified cycles
   - Accumulation Phase: Enter after trough confirmation in longest cycle
   - Distribution Phase: Begin scaling out as projected peak approaches
   - Multiple entries on shorter cycle retracements
   - Complete exit when longest cycle peaks

## Usage

### Basic Usage

Each strategy inherits from the `BaseStrategy` class and implements the required methods for signal generation, position sizing, and risk management. Here's how to use a strategy:

```python
from strategies.strategy_factory import get_strategy

# Create configuration
config = {
    'risk_per_trade': 1.0,
    'max_positions': 5,
    'use_trailing_stop': True,
    'min_alignment_threshold': 0.7
}

# Get a strategy instance
strategy = get_strategy('rapid_cycle_fld', config)

# Use the strategy to generate signals
signal = strategy.generate_signal(data, cycles, fld_crossovers, cycle_states)

# Calculate position size, stops, and targets
if signal['signal'] in ['buy', 'sell']:
    direction = 'long' if signal['signal'] == 'buy' else 'short'
    stop_price = strategy.set_stop_loss(data, signal, current_price, direction)
    target_price = strategy.set_take_profit(data, signal, current_price, stop_price, direction)
    position_size = strategy.calculate_position_size(account_value, signal, current_price, stop_price)
```

### Backtesting

A backtesting engine is included to test the strategies against historical data:

```python
from strategies.backtest_engine import run_strategy_backtest
from strategies.strategy_factory import get_strategy_by_name

# Run a backtest
result = run_strategy_backtest(
    data=historical_data,
    strategy_class=get_strategy_by_name('multi_cycle_confluence'),
    cycle_detector=your_cycle_detector,
    config=config,
    symbol="AAPL",
    timeframe="daily"
)

# Analyze results
print(f"Total return: {result.total_return_pct:.2f}%")
print(f"Win rate: {result.win_rate*100:.1f}%")

# Plot results
result.plot_results(save_path="backtest_chart.png")
```

### Example Script

An example script (`run_strategy_example.py`) is included in the main directory to demonstrate how to use the strategies:

```bash
# Run with default settings
python run_strategy_example.py

# Specify strategy and symbol
python run_strategy_example.py --strategy multi_cycle_confluence --symbol AAPL

# Skip backtesting
python run_strategy_example.py --no-backtest
```

## Integration with Core System

These strategies are designed to be integrated with the core Fibonacci Cycles system. They expect:

1. **Cycle Detection**: A component that can detect dominant cycles in the data
2. **FLD Analysis**: Methods to calculate FLD crossovers
3. **Historical Data**: OHLCV price data for analysis

The strategies can be used independently if you provide mock implementations of these dependencies, as shown in the example script.

## Extending the System

To create your own strategy:

1. Extend the `BaseStrategy` class
2. Implement the required methods:
   - `generate_signal`: Generate trading signals
   - `calculate_position_size`: Determine position size
   - `set_stop_loss`: Calculate stop loss level
   - `set_take_profit`: Calculate take profit level
3. Register it with the factory:
   ```python
   from strategies.strategy_factory import strategy_factory
   strategy_factory.register_strategy("my_strategy", MyStrategyClass)
   ```

## Requirements

- Python 3.7+
- Pandas
- NumPy
- Matplotlib (for plotting backtest results)