# Incremental Scanning Strategy for Large Symbol Lists

## Problem Statement

The current system requires a full batch scan for all symbols every time, which is inefficient for large symbol lists:

1. Takes too long to process thousands of stocks
2. Wastes network bandwidth by re-downloading data that hasn't changed
3. Duplicates analysis work that has already been done
4. No ability to resume interrupted scans

## Solution: Incremental Scanning System

The incremental scanning system allows for more efficient processing of large symbol lists by:

1. Only downloading new data for existing symbols
2. Only performing full scans for new symbols or those with updated data
3. Caching and reusing analysis results
4. Providing resume capability for interrupted scans

## Implementation Strategy

### 1. Scan State Management

Add a scan state management component to the existing state manager:

```python
class StateManager:
    # ... existing code ...
    
    def initialize_scan_state(self, symbols, exchange, interval):
        """Initialize a new scan state for a list of symbols"""
        scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.scan_state = {
            "id": scan_id,
            "total_symbols": len(symbols),
            "completed_symbols": 0,
            "pending_symbols": symbols.copy(),
            "completed_symbols_list": [],
            "failed_symbols": [],
            "exchange": exchange,
            "interval": interval,
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "status": "initialized"
        }
        
        return scan_id
        
    def update_scan_progress(self, symbol, success=True, error=None):
        """Update scan progress for a symbol"""
        if not hasattr(self, 'scan_state') or not self.scan_state:
            return False
            
        if symbol in self.scan_state["pending_symbols"]:
            self.scan_state["pending_symbols"].remove(symbol)
            
            if success:
                self.scan_state["completed_symbols"] += 1
                self.scan_state["completed_symbols_list"].append(symbol)
            else:
                self.scan_state["failed_symbols"].append({"symbol": symbol, "error": str(error)})
                
        self.scan_state["last_updated"] = datetime.now().isoformat()
        self.scan_state["status"] = "in_progress"
        
        # Check if scan is complete
        if not self.scan_state["pending_symbols"]:
            self.scan_state["status"] = "completed"
            self.scan_state["end_time"] = datetime.now().isoformat()
            
        return True
        
    def get_scan_state(self):
        """Get the current scan state"""
        if not hasattr(self, 'scan_state') or not self.scan_state:
            return None
            
        return self.scan_state
        
    def get_remaining_symbols(self):
        """Get list of symbols remaining to be scanned"""
        if not hasattr(self, 'scan_state') or not self.scan_state:
            return []
            
        return self.scan_state["pending_symbols"]
```

### 2. Modification to Batch Scan Process

Modify the batch scanning process in main_dashboard.py:

```python
def scan_batch(n_clicks, symbols_text, exchange, interval, lookback, num_cycles, price_source):
    # ... existing code ...
    
    # Initialize scan state
    scan_id = state_manager.initialize_scan_state(symbols, exchange, interval)
    
    # Process symbol batches instead of all at once
    batch_size = 10  # Process 10 symbols at a time
    all_results = []
    
    # Get all remaining symbols (for new scan this is all symbols, for resumed scan it's the remaining ones)
    remaining_symbols = state_manager.get_remaining_symbols()
    
    for i in range(0, len(remaining_symbols), batch_size):
        batch_symbols = remaining_symbols[i:i+batch_size]
        
        # Create scan parameters for batch
        params_list = [
            ScanParameters(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                lookback=int(lookback),
                num_cycles=int(num_cycles),
                price_source=price_source,
                generate_chart=False
            )
            for symbol in batch_symbols
        ]
        
        # Scan the batch
        batch_results = scanner.scan_batch(params_list)
        
        # Update scan state for each symbol
        for result in batch_results:
            state_manager.update_scan_progress(result.symbol, success=result.success, 
                                             error=result.error if hasattr(result, 'error') else None)
        
        # Add to results
        all_results.extend(batch_results)
        
        # Update UI with progress (this would require additional callbacks)
        
    # Register results with state manager
    batch_id = state_manager.register_batch_results(all_results)
    
    # ... existing code ...
```

### 3. Add Resume Capability

Add UI components to allow resuming interrupted scans:

```python
# Add to the dashboard layout
html.Div([
    html.H4("Scan Control", className="text-dark"),
    html.Div([
        html.Div(id="scan-progress-indicator", className="mb-2"),
        dbc.Button("Pause Scan", id="pause-scan-button", color="warning", className="me-2"),
        dbc.Button("Resume Scan", id="resume-scan-button", color="info"),
    ], id="scan-control-container", style={"display": "none"})
], className="mt-3")
```

### 4. Data Cache Validation

Modify the data fetcher to check if cached data needs refreshing:

```python
def needs_refresh(self, symbol, exchange, interval):
    """Check if data needs refreshing based on last update time"""
    cache_key = self._get_cache_key(symbol, exchange, interval)
    cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    if not os.path.exists(cache_file):
        return True
        
    # Check when file was last modified
    last_modified = os.path.getmtime(cache_file)
    last_modified_time = datetime.fromtimestamp(last_modified)
    
    # Get the appropriate refresh interval based on timeframe
    if interval in ["1m", "5m", "15m", "30m"]:
        # Refresh intraday data more frequently
        refresh_after = timedelta(hours=1)
    elif interval in ["1h", "2h", "4h"]:
        # Refresh hourly data daily
        refresh_after = timedelta(days=1)
    else:
        # Refresh daily/weekly/monthly less frequently
        refresh_after = timedelta(days=7)
        
    return datetime.now() - last_modified_time > refresh_after
```

## Benefits

1. **Efficiency**: Dramatically reduces processing time for large symbol lists
2. **Bandwidth Optimization**: Only downloads new data when needed
3. **Resilience**: Ability to pause/resume long-running scans
4. **Progress Tracking**: Shows completion percentage and estimated time remaining
5. **Cache Optimization**: Intelligently refreshes data based on timeframe

## Implementation Plan

1. Add scan state management to StateManager
2. Modify batch scan process to use batching
3. Add resume capability and progress tracking
4. Implement smart data cache validation
5. Add UI components for scan control

This incremental scanning approach is essential for scaling the system to handle large symbol lists efficiently.