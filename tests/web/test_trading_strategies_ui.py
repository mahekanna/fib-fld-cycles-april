import pytest
import dash
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the module to test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from web.trading_strategies_ui import create_strategy_dashboard
from tests.conftest import ScanResult


def test_create_strategy_dashboard(mock_scan_result):
    """Test that the trading strategies dashboard is created successfully"""
    # Use the fixture from conftest.py
    result = mock_scan_result
    
    # Call the function
    dashboard = create_strategy_dashboard(result)
    
    # Assert the dashboard contains the expected elements
    assert isinstance(dashboard, html.Div)
    
    # Check for strategy parameters card
    assert any(isinstance(child, dbc.Row) for child in dashboard.children)
    
    # Test with failed scan result
    failed_result = mock_scan_result
    failed_result.success = False
    failed_result.error = "Test error"
    
    # Call the function with failed result
    failed_dashboard = create_strategy_dashboard(failed_result)
    
    # Assert that an error message is displayed
    assert "Error" in str(failed_dashboard.children)
    
    # Test without backtest data
    no_backtest_result = mock_scan_result
    no_backtest_result.backtest_results = None
    
    # Call the function without backtest data
    no_backtest_dashboard = create_strategy_dashboard(no_backtest_result)
    
    # Dashboard should still render
    assert isinstance(no_backtest_dashboard, html.Div)


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])