#!/bin/bash
# Fibonacci Cycles System Universal Runner Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display usage
display_usage() {
    echo -e "${BLUE}Fibonacci Cycles Trading System${NC}"
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  dashboard    - Start the interactive dashboard"
    echo "  clean        - Clean caches and restart dashboard"
    echo "  help         - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh dashboard    - Start the dashboard"
    echo "  ./run.sh clean        - Clean caches and restart"
}

# Set PYTHONPATH to the project directory
export PYTHONPATH="$(pwd)"
echo -e "${BLUE}PYTHONPATH set to: $(pwd)${NC}"

# Parse command line arguments
COMMAND="$1"

case "$COMMAND" in
    dashboard)
        echo -e "${GREEN}Starting Dashboard...${NC}"
        # Uses the fixed restart script
        bash restart_dashboard_fixed.sh
        ;;
    clean)
        echo -e "${YELLOW}Cleaning and restarting dashboard...${NC}"
        # Clear caches and restart
        echo "Removing browser cache files if present..."
        rm -rf "$HOME/.dash_jupyter_hooks" 2>/dev/null || true
        rm -rf "$HOME/.dash_cache" 2>/dev/null || true
        
        echo "Clearing Python cache..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete 2>/dev/null || true
        
        echo "Recreating assets directory..."
        rm -rf ./assets 2>/dev/null || true
        mkdir -p ./assets
        
        # Run the dashboard
        bash restart_dashboard_fixed.sh
        ;;
    help|--help|-h)
        display_usage
        ;;
    *)
        if [ -z "$COMMAND" ]; then
            display_usage
        else
            echo -e "${RED}Unknown command: $COMMAND${NC}"
            display_usage
            exit 1
        fi
        ;;
esac

exit 0