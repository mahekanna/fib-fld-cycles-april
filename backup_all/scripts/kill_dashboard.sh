#!/bin/bash

# kill_dashboard.sh - Kills any running instances of the dashboard

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Fibonacci Harmonic Trading System ===${NC}"
echo -e "${YELLOW}Finding and killing dashboard processes...${NC}"

# Find processes using port 8050 (the dashboard port)
PORT_USED=$(lsof -i:8050 -t)

if [ -z "$PORT_USED" ]; then
    echo -e "${GREEN}No dashboard processes found running on port 8050.${NC}"
else
    # Count how many processes were found
    NUM_PROCESSES=$(echo "$PORT_USED" | wc -l)
    echo -e "${YELLOW}Found $NUM_PROCESSES process(es) using port 8050${NC}"
    
    # Kill each process
    for PID in $PORT_USED; do
        echo -e "Killing process $PID..."
        kill -9 $PID
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully killed process $PID${NC}"
        else
            echo -e "${RED}Failed to kill process $PID${NC}"
        fi
    done
    
    # Verify processes are killed
    sleep 1
    PORT_USED_AFTER=$(lsof -i:8050 -t)
    if [ -z "$PORT_USED_AFTER" ]; then
        echo -e "${GREEN}All dashboard processes successfully terminated.${NC}"
    else
        echo -e "${RED}Some processes are still running on port 8050.${NC}"
        echo -e "${RED}You may need to manually kill them:${NC}"
        echo -e "${YELLOW}sudo kill -9 $PORT_USED_AFTER${NC}"
    fi
fi

# Also find any Python processes matching 'main_dashboard.py'
PYTHON_PROCESSES=$(ps aux | grep "[m]ain_dashboard.py" | awk '{print $2}')

if [ -n "$PYTHON_PROCESSES" ]; then
    echo -e "${YELLOW}Found additional dashboard processes:${NC}"
    for PID in $PYTHON_PROCESSES; do
        echo -e "Killing Python process $PID..."
        kill -9 $PID
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully killed process $PID${NC}"
        else
            echo -e "${RED}Failed to kill process $PID${NC}"
        fi
    done
fi

echo -e "${GREEN}Dashboard shutdown complete.${NC}"
echo -e "You can now start a fresh dashboard with: ./run_dashboard.sh"