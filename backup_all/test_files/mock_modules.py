"""
Mock modules for testing without installing dependencies.
"""
import sys
from unittest.mock import MagicMock

# Create mock classes for Dash components
class html:
    Div = MagicMock
    H3 = MagicMock
    H4 = MagicMock
    P = MagicMock
    Label = MagicMock
    Strong = MagicMock
    Span = MagicMock

class dcc:
    Graph = MagicMock
    Dropdown = MagicMock

class dbc:
    Card = MagicMock
    CardHeader = MagicMock
    CardBody = MagicMock
    Table = MagicMock
    Row = MagicMock
    Col = MagicMock
    Button = MagicMock

# Create mock for dash
dash = MagicMock()

# Create mock for pytest
pytest = MagicMock()
pytest.main = MagicMock(return_value=0)

# Add mocks to sys.modules
sys.modules['dash'] = MagicMock()
sys.modules['dash.html'] = html
sys.modules['dash.dcc'] = dcc
sys.modules['dash_bootstrap_components'] = dbc
sys.modules['pytest'] = pytest
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()