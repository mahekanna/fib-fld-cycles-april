# Incremental Scanning Usage Guide

The Fibonacci Cycles Trading System now supports incremental scanning of large symbol lists. This guide explains how to use this feature effectively.

## Benefits of Incremental Scanning

1. **Efficiency**: Process thousands of symbols in manageable batches
2. **Progress Tracking**: Monitor scan progress and completion percentage
3. **Pause & Resume**: Pause and resume long-running scans as needed
4. **Resilience**: Recover from interrupted scans without losing progress
5. **Data Optimization**: Only download and process new data when needed

## Using Incremental Scanning

### Starting a Batch Scan

1. Enter your symbols in the text area (one symbol per line) or the system will use `symbols.csv`
2. Click the "Batch Scan" button to begin
3. The scan progress will appear in two places:
   - In the header (showing overall progress)
   - In the left sidebar (showing detailed progress)

### Monitoring Progress

The scan progress indicators show:
- Completion percentage
- Current status (INITIALIZED, IN_PROGRESS, PAUSED, or COMPLETED)
- Symbol count (completed/total)
- A visual progress bar

### Pausing a Scan

If you need to pause a running scan:
1. Click the "Pause" button in the batch scan section
2. The status will change to "PAUSED"
3. The header indicator will show a pause icon

### Resuming a Scan

To resume a paused scan:
1. Click the "Resume" button in the batch scan section
2. The scan will continue from where it left off
3. The status will change back to "IN_PROGRESS"

## Technical Details

When a batch scan is running:
1. Symbols are processed in batches of 10 (configurable)
2. Each batch is scanned and results are registered with the state manager
3. Progress is updated in real-time
4. Both successful and failed symbols are tracked

## Tips for Large Symbol Lists

When scanning thousands of symbols:

1. **Start with a smaller test**: First scan 10-20 symbols to check everything works
2. **Use incremental batches**: Process your watchlist in segments (e.g., by sector or market cap)
3. **Monitor resources**: Keep an eye on memory usage during large scans
4. **Use the pause button**: For very large lists, pause occasionally to view results so far
5. **Save batch scan results**: Important results can be saved for offline analysis

## Troubleshooting

If you encounter issues:

1. **Scan appears stuck**: Check the logs for errors; try refreshing the page
2. **Results incomplete**: Verify the symbols exist and data is available for the selected interval
3. **Performance issues**: Reduce batch size for slower systems

For any persistent issues, check the logs or restart the dashboard with `./restart_dashboard_fixed.sh`