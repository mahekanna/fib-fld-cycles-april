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
from web.harmonic_visualization import create_harmonic_visualization
from tests.conftest import ScanResult


def test_create_harmonic_visualization(mock_scan_result):
    """Test that the harmonic pattern visualization is created successfully"""
    # Use fixture from conftest.py
    result = mock_scan_result
    
    # Call the function
    visualization = create_harmonic_visualization(result)
    
    # Assert the visualization contains the expected elements
    assert isinstance(visualization, html.Div)
    
    # Check for main chart
    assert any(isinstance(child, dbc.Card) for child in visualization.children)
    
    # Test with failed scan result
    failed_result = mock_scan_result
    failed_result.success = False
    failed_result.error = "Test error"
    
    # Call the function with failed result
    failed_viz = create_harmonic_visualization(failed_result)
    
    # Assert that an error message is displayed
    assert "Error" in str(failed_viz.children[0].children)
    
    # Test with no harmonic patterns
    no_patterns_result = mock_scan_result
    no_patterns_result.harmonic_relationships = {}
    
    # Call the function with no patterns
    no_patterns_viz = create_harmonic_visualization(no_patterns_result)
    
    # Assert that a "no patterns" message is displayed - updated to match actual implementation
    assert "No harmonic patterns detected" in str(no_patterns_viz.children[0].children)


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])