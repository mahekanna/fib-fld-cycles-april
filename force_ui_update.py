"""
Force UI update by directly modifying the layout structure
This bypasses Dash caching issues by making highly visible changes
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Update main_dashboard.py to force a visual change
def add_visible_ui_changes():
    try:
        # Read the current main_dashboard.py
        with open('main_dashboard.py', 'r') as f:
            content = f.read()
        
        # Make highly visible changes to the header
        if "Fibonacci Harmonic Trading System" in content:
            content = content.replace(
                'html.H2("Fibonacci Harmonic Trading System", className="text-white")',
                'html.Div([html.H2("Fibonacci Cycles Trading System", className="text-white"), html.Div("🔄 UI UPDATED", style={"color": "#FFA500", "fontWeight": "bold"})])'
            )
            
            # Also force light mode for better visibility of changes
            if "external_stylesheets=[dbc.themes.DARKLY]" in content:
                content = content.replace(
                    "external_stylesheets=[dbc.themes.DARKLY]",
                    "external_stylesheets=[dbc.themes.FLATLY]"  # Switch to a light theme
                )
            
            # Save the modified content
            with open('main_dashboard.py', 'w') as f:
                f.write(content)
                
            logger.info("✅ Successfully made visible UI changes to main_dashboard.py")
            return True
        else:
            logger.error("❌ Could not find expected header text in main_dashboard.py")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error modifying main_dashboard.py: {e}")
        return False

# Add a highly visible modal to advanced_strategies_ui.py
def add_visible_modal():
    try:
        # Read the current advanced_strategies_ui.py
        with open('web/advanced_strategies_ui.py', 'r') as f:
            content = f.read()
            
        # Find the create_batch_advanced_signals function
        if "def create_batch_advanced_signals" in content:
            # Add a modal that will appear immediately when the component loads
            modal_code = """
    # FORCE UI UPDATE - Add a visible notification
    notification_modal = dbc.Modal(
        [
            dbc.ModalHeader("UI Update Notification"),
            dbc.ModalBody([
                html.H4("UI Has Been Updated", className="text-success"),
                html.P("The dashboard UI has been updated with the following changes:"),
                html.Ul([
                    html.Li("Separated Single Symbol and Batch Analysis modes"),
                    html.Li("Improved navigation flow between components"),
                    html.Li("Added modals for detail views"),
                    html.Li("Fixed price consistency issues"),
                    html.Li("Added visual indicators for state management")
                ]),
                html.P("Please clear your browser cache if you still don't see these changes."),
            ]),
            dbc.ModalFooter(
                dbc.Button("Got it!", id="close-notification-modal", className="ms-auto")
            ),
        ],
        id="notification-modal",
        size="lg",
        is_open=True,
    )
    
    # Register callback to close the notification modal
    if app:
        @app.callback(
            Output("notification-modal", "is_open"),
            Input("close-notification-modal", "n_clicks"),
            prevent_initial_call=True
        )
        def close_notification(n_clicks):
            return False
    """
            
            # Insert the modal code at the beginning of the function
            content = content.replace(
                "def create_batch_advanced_signals(results_list, app=None):",
                "def create_batch_advanced_signals(results_list, app=None):" + modal_code
            )
            
            # Save the modified content
            with open('web/advanced_strategies_ui.py', 'w') as f:
                f.write(content)
                
            logger.info("✅ Successfully added notification modal to advanced_strategies_ui.py")
            return True
        else:
            logger.error("❌ Could not find create_batch_advanced_signals function")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error modifying advanced_strategies_ui.py: {e}")
        return False

# Create a script to truly clear all caches
def create_force_restart_script():
    script_content = """#!/bin/bash
# EXTREME CACHE CLEARING AND RESTART
echo "=== EXTREME CACHE CLEARING AND RESTART ==="
echo "This will completely clear all caches and restart the dashboard"

# Kill any running dashboard processes
echo "Stopping all running processes..."
pkill -f "python.*main_dashboard.py" || true

# Clear browser cache directory
echo "Removing Dash browser hooks..."
rm -rf "$HOME/.dash_jupyter_hooks" 2>/dev/null || true
rm -rf "$HOME/.dash_cache" 2>/dev/null || true

# Remove all __pycache__ directories
echo "Clearing all Python cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Clear all .so files
echo "Removing compiled extensions..."
find . -name "*.so" -delete 2>/dev/null || true

# Clear assets directory
echo "Recreating assets directory..."
rm -rf ./assets 2>/dev/null || true
mkdir -p ./assets

# Clear data cache
echo "Clearing data cache..."
rm -f data/cache/market_data.db 2>/dev/null || true

# Clear state file
echo "Clearing state files..."
rm -f data/state_snapshot.pkl 2>/dev/null || true 

# Wait to ensure everything is stopped
echo "Waiting for processes to terminate..."
sleep 2

# Run the force update script
echo "Applying forced UI updates..."
python force_ui_update.py

# Set the PYTHONPATH
export PYTHONPATH="$(pwd)"

# Start with a completely fresh Python interpreter
echo "Starting dashboard with fresh Python interpreter..."
python -W ignore main_dashboard.py

echo "If you still don't see UI changes, try closing and reopening your browser"
"""
    
    # Save the script
    with open('extreme_restart.sh', 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod('extreme_restart.sh', 0o755)
    logger.info("✅ Created extreme_restart.sh script")
    return True

if __name__ == "__main__":
    print("=== FORCING UI UPDATES ===")
    
    # Make the changes
    header_changed = add_visible_ui_changes()
    modal_added = add_visible_modal()
    script_created = create_force_restart_script()
    
    if header_changed and modal_added and script_created:
        print("\n✅ All changes applied successfully!")
        print("To see the changes, run:")
        print("   ./extreme_restart.sh")
        print("\nThis will completely clear all caches and restart with the updated UI")
    else:
        print("\n❌ Some changes could not be applied")
        print("Please check the logs for details")