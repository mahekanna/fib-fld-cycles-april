import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import sys
import numpy as np
import pandas as pd

# Add project root to path to enable absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use absolute imports
from models.scan_result import ScanResult


class ResultsRepository:
    """
    Repository for storing and retrieving scan results.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize the repository with configuration.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Create storage directory if it doesn't exist
        self.results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "storage", "results"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create cache directories
        self.json_dir = os.path.join(self.results_dir, "json")
        self.pickle_dir = os.path.join(self.results_dir, "pickle")
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.pickle_dir, exist_ok=True)
        
    def save_result(self, result: ScanResult, force_update: bool = False) -> str:
        """
        Save a scan result to storage with intelligent caching.
        
        Args:
            result: ScanResult instance
            force_update: If True, ignore cache and create a new result file
            
        Returns:
            Path to the saved result file
        """
        if not result:
            self.logger.warning("Attempted to save an empty result")
            return ""
            
        # Check if we already have a recent result for this symbol/interval/lookback
        if not force_update:
            existing_path = self.find_existing_result(
                result.symbol, 
                result.exchange, 
                result.interval, 
                getattr(result, 'lookback', 1000)  # Default to 1000 if not specified
            )
            
            if existing_path:
                self.logger.info(f"Using existing result for {result.symbol}/{result.interval}: {existing_path}")
                return existing_path
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{result.symbol}_{result.interval}_{timestamp}"
        
        try:
            # Save as JSON (without chart image)
            json_path = os.path.join(self.json_dir, f"{base_filename}.json")
            with open(json_path, 'w') as f:
                # Convert to dict for JSON serialization
                result_dict = result.to_dict()
                json.dump(result_dict, f, indent=4)
            
            # Save as pickle (with chart image)
            pickle_path = os.path.join(self.pickle_dir, f"{base_filename}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(result, f)
            
            self.logger.info(f"Saved scan result for {result.symbol} to {json_path}")
            return json_path
            
        except Exception as e:
            self.logger.error(f"Error saving result for {result.symbol}: {str(e)}")
            
            # Create an emergency simplified version if serialization fails
            try:
                # More comprehensive NumPy type conversion function
                def convert_to_native(obj):
                    # Handle direct NumPy types
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    
                    # Handle objects with dtype attribute
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
                    
                    # Handle containers recursively
                    if isinstance(obj, dict):
                        return {k: convert_to_native(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_native(item) for item in obj]
                    
                    # Return other types unchanged
                    return obj
                
                # Create raw dictionary with basic types
                raw_simplified = {
                    "symbol": result.symbol,
                    "exchange": result.exchange,
                    "interval": result.interval,
                    "timestamp": datetime.now().isoformat(),
                    "success": bool(result.success),
                    "error": str(result.error) if result.error else None,
                    "price": float(result.price) if hasattr(result.price, "__float__") else 0.0,
                    "detected_cycles": [int(x) for x in result.detected_cycles],
                    "signal": {
                        "signal": result.signal.get("signal", "neutral"),
                        "strength": float(convert_to_native(result.signal.get("strength", 0.0))) if result.signal else 0.0,
                        "confidence": result.signal.get("confidence", "low") if result.signal else "low"
                    } 
                }
                
                # Apply full type conversion to ensure all nested structures are Python native types
                simplified = convert_to_native(raw_simplified)
                
                # Save simplified version
                emergency_path = os.path.join(self.json_dir, f"{base_filename}_simplified.json")
                with open(emergency_path, 'w') as f:
                    json.dump(simplified, f, indent=4)
                    
                self.logger.info(f"Saved simplified result for {result.symbol} to {emergency_path}")
                return emergency_path
                
            except Exception as e2:
                self.logger.error(f"Could not save even simplified result: {str(e2)}")
                return ""
    
    def save_batch_results(self, results: List[ScanResult], force_update: bool = False) -> List[str]:
        """
        Save multiple scan results with intelligent caching.
        
        Args:
            results: List of ScanResult instances
            force_update: If True, ignore cache and create new result files
            
        Returns:
            List of paths to the saved result files
        """
        paths = []
        for result in results:
            path = self.save_result(result, force_update)
            if path:
                paths.append(path)
        
        self.logger.info(f"Saved {len(paths)} results (out of {len(results)} total)")
        return paths
    
    def load_result(self, filepath: str) -> Optional[ScanResult]:
        """
        Load a scan result from storage.
        
        Args:
            filepath: Path to the result file
            
        Returns:
            ScanResult instance or None if loading failed
        """
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    return ScanResult.from_dict(data)
            elif filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                self.logger.error(f"Unknown file format for {filepath}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading result from {filepath}: {str(e)}")
            return None
    
    def get_latest_results(self, symbol: Optional[str] = None, 
                          interval: Optional[str] = None,
                          limit: int = 10) -> List[ScanResult]:
        """
        Get the latest scan results, optionally filtered by symbol and interval.
        
        Args:
            symbol: Optional symbol to filter by
            interval: Optional interval to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of ScanResult instances
        """
        # List JSON files (they're easier to filter by metadata)
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        
        # Filter by symbol and interval if provided
        if symbol:
            json_files = [f for f in json_files if f.startswith(f"{symbol}_")]
        if interval:
            json_files = [f for f in json_files if f"{interval}_" in f]
        
        # Sort by timestamp (descending)
        json_files.sort(reverse=True)
        
        # Load results up to the limit
        results = []
        for filename in json_files[:limit]:
            filepath = os.path.join(self.json_dir, filename)
            result = self.load_result(filepath)
            if result:
                results.append(result)
        
        return results
        
    def get_latest_batch_results(self, max_age_hours: int = 24) -> List[ScanResult]:
        """
        Get the latest batch scan results.
        
        This tries to identify results from the most recent batch scan by looking at
        file creation timestamps and grouping them together.
        
        Args:
            max_age_hours: Maximum age of results in hours
            
        Returns:
            List of ScanResult instances from the latest batch scan
        """
        try:
            # Get all JSON files in the directory
            json_files = []
            
            for filename in os.listdir(self.json_dir):
                if filename.endswith(".json") and not filename.endswith("_simplified.json"):
                    file_path = os.path.join(self.json_dir, filename)
                    
                    # Use file modification time to group by batch
                    file_timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Check if the file is recent enough
                    age_hours = (datetime.now() - file_timestamp).total_seconds() / 3600
                    
                    if age_hours <= max_age_hours:
                        json_files.append((file_path, file_timestamp))
            
            if not json_files:
                self.logger.info("No recent batch results found")
                return []
                
            # Sort by timestamp (newest first)
            json_files.sort(key=lambda x: x[1], reverse=True)
            
            # Group files by batch - files created within 2 minutes of each other
            batch_groups = []
            current_batch = []
            
            for file_path, timestamp in json_files:
                if not current_batch:
                    # Start a new batch
                    current_batch = [(file_path, timestamp)]
                else:
                    # Check if this file is part of the current batch
                    prev_timestamp = current_batch[-1][1]
                    time_diff = abs((timestamp - prev_timestamp).total_seconds())
                    
                    if time_diff <= 120:  # 2 minutes threshold
                        current_batch.append((file_path, timestamp))
                    else:
                        # Start a new batch
                        batch_groups.append(current_batch)
                        current_batch = [(file_path, timestamp)]
            
            # Add the last batch
            if current_batch:
                batch_groups.append(current_batch)
            
            # Get the most recent batch with more than 1 result (likely a batch scan)
            latest_batch = None
            for batch in batch_groups:
                if len(batch) > 1:
                    latest_batch = batch
                    break
            
            # If no multi-result batch found, use the most recent single result
            if not latest_batch and batch_groups:
                latest_batch = batch_groups[0]
            
            if not latest_batch:
                self.logger.info("No batch scan results found")
                return []
            
            # Load all results in the batch
            results = []
            for file_path, _ in latest_batch:
                result = self.load_result(file_path)
                if result:
                    results.append(result)
            
            self.logger.info(f"Loaded {len(results)} results from latest batch scan")
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting latest batch results: {str(e)}")
            return []
        
    def find_existing_result(self, symbol: str, exchange: str, interval: str, lookback: int) -> Optional[str]:
        """
        Find an existing result file for the given symbol and parameters.
        Only returns results that are less than 24 hours old for daily/weekly intervals,
        or less than 2 hours old for intraday intervals.
        
        Args:
            symbol: Symbol name
            exchange: Exchange name
            interval: Time interval
            lookback: Number of bars to look back
            
        Returns:
            Path to existing result file or None if not found
        """
        try:
            # Get all JSON files for this symbol and interval
            json_files = []
            
            for filename in os.listdir(self.json_dir):
                if (filename.startswith(f"{symbol}_") and 
                    f"_{interval}_" in filename and 
                    filename.endswith(".json") and
                    not filename.endswith("_simplified.json")):
                    
                    file_path = os.path.join(self.json_dir, filename)
                    
                    # Extract timestamp from filename or use file modification time
                    try:
                        # Format: symbol_interval_YYYYMMDD_HHMMSS.json
                        timestamp_str = filename.replace(f"{symbol}_{interval}_", "").replace(".json", "")
                        file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    except:
                        # Fallback to file modification time
                        file_timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Check if the file is recent enough
                    age_hours = (datetime.now() - file_timestamp).total_seconds() / 3600
                    max_age = 24 if interval in ['daily', 'weekly', 'monthly'] else 2  # Daily/weekly: 24h, Intraday: 2h
                    
                    if age_hours <= max_age:
                        json_files.append((file_path, file_timestamp))
            
            # Sort by timestamp (newest first)
            json_files.sort(key=lambda x: x[1], reverse=True)
            
            # Return the newest file if any
            if json_files:
                self.logger.info(f"Found existing result for {symbol}/{interval}: {json_files[0][0]}")
                return json_files[0][0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding existing result: {str(e)}")
            return None