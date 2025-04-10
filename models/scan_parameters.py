from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ScanParameters:
    """
    Parameters for cycle scanning and analysis.
    """
    symbol: str
    exchange: str
    interval: str = "daily"
    lookback: int = 1000
    num_cycles: int = 3
    price_source: str = "close"
    generate_chart: bool = False
    custom_data: Optional[Any] = None