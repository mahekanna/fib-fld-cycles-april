# Lookback Parameter Fix

## Issue
The system was incorrectly hardcoding data display to use a fixed number of bars (250) regardless of the lookback parameter set by the user. This caused:

1. Charts to always show only 250 bars even when the lookback was much larger
2. FLD indicators being calculated on potentially longer data but displayed with truncated data
3. Cycle waves being generated for only 250 bars

## Root Cause
1. In `dashboard_implementation.py`, the chart rendering code used fixed slicing with `-250:` everywhere
2. In `scanner_system.py`, cycle waves were generated using 250 points regardless of actual lookback
3. The lookback parameter from the scan was not passed to the UI, so the chart couldn't use it

## Solution Implemented

### 1. Updated Dashboard Chart Display
- Removed hardcoded `-250:` slicing in all data visualizations
- Now using the full data range requested by the lookback parameter

### 2. Fixed Cycle Wave Generation
- Updated cycle wave generation to use actual lookback parameter:
```python
# Use proper lookback instead of hardcoded 250
sample_size = min(len(price_series), parameters.lookback)
```

### 3. Improved Data Parameter Passing
- Added lookback parameter to the ScanResult class
- Modified scanner_system.py to store the lookback in the result
- Updated chart creation to use the correct lookback parameter

### 4. Preserved TvDatafeed Data
- Ensured real market data from TvDatafeed is used properly
- No more artificial limitation of data points

## How to Test
1. Change the lookback parameter in the UI (e.g., try 500, 1000, 2000)
2. Verify that charts show the full requested number of bars
3. Check that FLD lines and cycle waves span the entire requested range
4. Confirm that different timeframes (daily, 4h, 1h, 15min) properly respect lookback

## Benefits
- Proper respect for user-specified lookback parameter
- Consistent data display across all timeframes
- Full use of available data for analysis and visualization
- More accurate cycle analysis with complete data