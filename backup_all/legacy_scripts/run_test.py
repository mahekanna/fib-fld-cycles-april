
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Run the test module
from fib_cycles_system import test_backtest
print('Running test as module...')
test_backtest.run_simple_backtest()
test_backtest.test_data_fetcher()

