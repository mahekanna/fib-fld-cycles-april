from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


@dataclass
class ScanResult:
    """
    Complete result of a cycle analysis.
    """
    # Basic information
    symbol: str
    exchange: str
    interval: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0
    lookback: int = 1000  # Store the requested lookback period
    
    # Price information
    price: float = 0.0
    
    # Cycle information
    detected_cycles: List[int] = field(default_factory=list)
    cycle_powers: Dict[int, float] = field(default_factory=dict)
    cycle_states: List[Dict] = field(default_factory=list)
    harmonic_relationships: Dict[str, Dict] = field(default_factory=dict)
    
    # Signal information
    signal: Dict = field(default_factory=dict)
    position_guidance: Dict = field(default_factory=dict)
    
    # Original data DataFrame - needed for visualizations
    data: Optional[Any] = None
    
    # Chart image (if generated)
    chart_image: Optional[Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        # Helper function to convert numpy types to Python native types
        def convert_numpy_types(obj):
            import numpy as np
            import pandas as pd
            
            # Direct instance checks for common NumPy types
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, bool):
                # Ensure boolean values are Python booleans
                return bool(obj)
            
            # Handle cases where direct instance check doesn't work
            # but dtype attribute is available
            if hasattr(obj, 'dtype'):
                if np.issubdtype(obj.dtype, np.bool_):
                    return bool(obj)
                elif np.issubdtype(obj.dtype, np.integer):
                    return int(obj)
                elif np.issubdtype(obj.dtype, np.floating):
                    return float(obj)
            
            # Handle pandas objects
            if isinstance(obj, pd.Series):
                return obj.to_list()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            
            # Recursively handle containers
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert cycle powers from numpy types
        cycle_powers = {}
        for k, v in self.cycle_powers.items():
            # Convert key to int if it's a numpy type
            key = int(k) if hasattr(k, 'dtype') else k
            # Convert value to float if it's a numpy type
            value = float(v) if hasattr(v, 'dtype') else v
            cycle_powers[key] = value
            
        # Convert cycle states
        cycle_states = []
        for state in self.cycle_states:
            # Explicitly handle common NumPy bool_ issue in cycle states
            state_copy = state.copy()
            if 'is_bullish' in state_copy and hasattr(state_copy['is_bullish'], 'dtype'):
                state_copy['is_bullish'] = bool(state_copy['is_bullish'])
            cycle_states.append(convert_numpy_types(state_copy))
            
        # Process signal and position guidance
        signal = convert_numpy_types(self.signal)
        
        # Make sure all position guidance values are proper Python types, not NumPy types
        position_guidance_copy = self.position_guidance.copy()
        for key in position_guidance_copy:
            if hasattr(position_guidance_copy[key], 'dtype'):
                if np.issubdtype(position_guidance_copy[key].dtype, np.bool_):
                    position_guidance_copy[key] = bool(position_guidance_copy[key])
                elif np.issubdtype(position_guidance_copy[key].dtype, np.integer):
                    position_guidance_copy[key] = int(position_guidance_copy[key])
                elif np.issubdtype(position_guidance_copy[key].dtype, np.floating):
                    position_guidance_copy[key] = float(position_guidance_copy[key])
        
        position_guidance = convert_numpy_types(position_guidance_copy)
        
        # Build result dictionary with converted values
        # Apply the conversion function to all values to ensure no NumPy types remain
        raw_result = {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'interval': self.interval,
            'timestamp': self.timestamp.isoformat(),
            'success': bool(self.success),  # Convert numpy bool to Python bool
            'error': self.error,
            'execution_time': float(self.execution_time),
            'lookback': int(self.lookback),  # Include lookback parameter for UI
            'price': float(self.price),
            'detected_cycles': [int(c) for c in self.detected_cycles],
            'cycle_powers': cycle_powers,
            'cycle_states': cycle_states,
            'harmonic_relationships': convert_numpy_types(self.harmonic_relationships),
            'signal': signal,
            'position_guidance': position_guidance,
            # Don't serialize the data property - it's a pandas DataFrame which can't be serialized to JSON
            # and it's not needed in the UI store since it's passed directly to the visualization components
            'has_data': bool(self.data is not None and not self.data.empty)
        }
        
        # Final conversion to ensure everything is serializable
        result = convert_numpy_types(raw_result)
        
        # Chart image can't be serialized directly
        if self.chart_image is not None:
            result['has_chart'] = True
        else:
            result['has_chart'] = False
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScanResult':
        """Create from dictionary."""
        # Make a copy to avoid modifying the input
        data_copy = data.copy()
        
        # Convert timestamp string to datetime
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'].replace('Z', '+00:00'))
            
        # Handle chart image and data properties
        data_copy['chart_image'] = None
        data_copy['data'] = None
        
        # Remove non-class properties
        for key in ['has_data', 'has_chart']:
            if key in data_copy:
                del data_copy[key]
            
        return cls(**data_copy)