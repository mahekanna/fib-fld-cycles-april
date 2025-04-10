# Price Handling Fix for Batch Advanced Signals

## Issue

There are inconsistencies in price handling between the batch advanced signals and advanced strategies components. This causes several problems:

1. The price shown in the batch advanced signals table may not match the price used for analysis in detailed strategy views
2. Real-time price updates are logged but not consistently used, creating confusion between cached analysis prices and real-time display
3. The `create_batch_advanced_signals` function in `web/advanced_strategies_ui.py` has multiple methods for determining prices:
   - Using the original price from the scan result's price attribute
   - Falling back to the data's 'close' column when price attribute is missing
   - Getting real-time price updates via data_refresher but not overwriting the original price

4. The `create_detailed_trading_plan` function also has multiple price sources but lacks a clear hierarchy

## Root Cause

The system is designed to perform analysis on a snapshot of data, but also provides real-time price monitoring. When navigating between different UI components, the price source can change, causing inconsistencies. Specifically:

1. The original scan result contains a snapshot price in the `result.price` attribute, set in `scanner_system.py`
2. The `create_batch_advanced_signals` function provides a real-time price monitoring capability but doesn't use these prices for strategy calculations
3. When viewing detailed trading plans, the function tries multiple approaches to get a price, potentially using a different source than the original analysis

## Fix Implementation

### 1. Enhanced Price Source Logging in Batch Advanced Signals

In `web/advanced_strategies_ui.py`, the `create_batch_advanced_signals` function was modified with:

```python
# CRITICAL: ALWAYS use the exact same price from the scan result
# This guarantees consistency with Analysis Results tab
current_price = 0.0

# First priority: Get price directly from result.price attribute
if hasattr(result, 'price'):
    current_price = result.price
    logger.warning(f"PRICE SOURCE [1]: Using exact price from scan result for {result.symbol}: {current_price}")

# Only use data as fallback if price attribute is missing or zero
if current_price == 0 and hasattr(result, 'data') and result.data is not None and 'close' in result.data and len(result.data) > 0:
    current_price = result.data['close'].iloc[-1]
    logger.warning(f"PRICE SOURCE [2]: Fallback to data['close'] for {result.symbol}: {current_price}")
```

The function now prioritizes using the exact price from the scan result for consistency.

### 2. Real-Time Price Difference Logging Without Overwriting

The function now logs real-time price differences but doesn't overwrite the original price:

```python
# IMPORTANT: We will NOT override the original price from the scan result
# This ensures consistency between batch scanning and advanced strategies
# Instead, we'll store the real-time price separately if needed
real_time_price = 0.0
try:
    # Get latest data but don't overwrite the original price
    latest_data = refresher.get_latest_data(
        symbol=result.symbol, 
        exchange=result.exchange, 
        interval=result.interval,
        refresh_if_needed=True
    )
    
    if latest_data is not None and not latest_data.empty:
        real_time_price = latest_data['close'].iloc[-1]
        logger.warning(f"PRICE SOURCE [REALTIME]: Got real-time price for {result.symbol}: {real_time_price:.2f} (original: {current_price:.2f})")
        
        # Log the difference but don't overwrite the original price
        if abs(real_time_price - current_price) > 0.01:
            logger.warning(f"PRICE DIFFERENCE: {result.symbol} original {current_price:.2f} vs real-time {real_time_price:.2f}")
```

### 3. Clear Price Source Hierarchy in Detailed Strategy View

The `create_detailed_trading_plan` function was updated with a clear price source hierarchy:

```python
# IMPORTANT: For proper price consistency, we need to use the exact same price
# from the original scan result across all UI components

# Get the price from symbol_data (which should be the ScanResult)
price = 0.0
primary_source = None

# First priority: Use price from the provided symbol_data
if symbol_data and hasattr(symbol_data, 'price'):
    price = symbol_data.price
    primary_source = "symbol_data parameter"
    logger.info(f"1️⃣ Using price from symbol_data: {price} for {symbol}")

# Second priority: Look for the exact symbol in batch results
if price == 0.0:
    for result in _batch_results:
        if hasattr(result, 'symbol') and result.symbol == symbol:
            if hasattr(result, 'price'):
                price = result.price
                primary_source = "batch_results global list"
                logger.info(f"2️⃣ Found price in batch results: {price} for {symbol}")
                break

# Third priority: Find the symbol in global _current_scan_result
if price == 0.0:
    if _current_scan_result and hasattr(_current_scan_result, 'symbol') and _current_scan_result.symbol == symbol:
        if hasattr(_current_scan_result, 'price'):
            price = _current_scan_result.price
            primary_source = "current_scan_result global variable"
            logger.info(f"3️⃣ Found price in current scan result: {price} for {symbol}")

# Last resort: Get fresh data (should rarely be needed, only if price is actually 0)
if price == 0.0:
    # Try fetching fresh data as last resort
```

### 4. Price Source Documentation in the UI

Added UI elements to clearly indicate the price source to users:

```python
# Add price source information for debugging
html.Div([
    html.P([
        "Price: ",
        html.Strong(f"₹{symbol_info.get('price', 0):.2f}")
    ], className="small text-muted")
], className="mb-2"),
```

In the real-time price display, a note was added:

```python
html.Small("Note: This price is separate from the original scan results", 
          className="text-muted fst-italic")
```

## Testing

The fix has been tested with:

1. Multiple batch advanced signal runs to verify price consistency
2. Navigation between different views to ensure the same price is used
3. Extended waiting periods to allow real-time prices to update, confirming that the original analysis prices remain consistent

## Future Improvements

1. Add a configuration option to allow users to choose between consistent cached prices vs. real-time prices
2. Implement a unified price management system that manages both cached analysis prices and real-time updates
3. Add UI toggle to allow switching between analysis price view and real-time price view