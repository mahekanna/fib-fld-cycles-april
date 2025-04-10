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
from web.scanner_dashboard import create_scanner_dashboard
from tests.conftest import ScanResult


def test_create_scanner_dashboard(mock_scan_results):
    """Test that the scanner dashboard is created successfully"""
    # Use the fixture from conftest.py
    results = mock_scan_results
    
    # Call the function
    dashboard = create_scanner_dashboard(results)
    
    # Assert the dashboard contains the expected elements
    assert isinstance(dashboard, html.Div)
    
    # Check for main components
    assert any(isinstance(child, dbc.Row) for child in dashboard.children)
    
    # Test with empty results list
    empty_dashboard = create_scanner_dashboard([])
    
    # Assert that an "no results" message is displayed
    assert "No scan results" in str(empty_dashboard.children)
    
    # Test with mixed success/failure results
    mixed_results = mock_scan_results[:3]
    mixed_results[1].success = False
    mixed_results[1].error = "Test error"
    
    # Call the function with mixed results
    mixed_dashboard = create_scanner_dashboard(mixed_results)
    
    # Assert the dashboard still renders
    assert isinstance(mixed_dashboard, html.Div)


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])