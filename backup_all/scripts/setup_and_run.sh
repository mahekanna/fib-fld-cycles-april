#!/bin/bash

# Display setup banner
echo "=================================="
echo "Fibonacci Harmonic Trading System"
echo "Setup and Run Script"
echo "=================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create and activate conda environment
read -p "Create new conda environment 'fib_cycles'? (y/n): " create_env
if [[ $create_env == "y" || $create_env == "Y" ]]; then
    echo "Creating conda environment 'fib_cycles'..."
    conda create -y -n fib_cycles python=3.9
    
    echo "Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate fib_cycles
    
    echo "Installing dependencies from conda-forge..."
    conda install -y -c conda-forge numpy pandas scipy matplotlib plotly
    conda install -y -c conda-forge dash dash-bootstrap-components jupyter
    
    echo "Installing Python packages..."
    pip install -r requirements.txt
    
    echo "Installing TradingView data feed..."
    pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
    
    echo "Installing package in development mode..."
    pip install -e .
else
    echo "Activating existing environment..."
    eval "$(conda shell.bash hook)"
    conda activate fib_cycles
fi

# Check for TradingView credentials
if [ ! -f "./config/tv_credentials.json" ]; then
    echo "TradingView credentials not found."
    read -p "Would you like to create a credentials file now? (y/n): " create_creds
    
    if [[ $create_creds == "y" || $create_creds == "Y" ]]; then
        mkdir -p ./config
        
        echo "Enter your TradingView username:"
        read tv_username
        
        echo "Enter your TradingView password:"
        read -s tv_password
        
        # Create credentials file
        cat > ./config/tv_credentials.json << EOF
{
  "username": "$tv_username",
  "password": "$tv_password"
}
EOF
        echo "Credentials file created at ./config/tv_credentials.json"
    else
        echo "Please create a credentials file manually before running the application."
    fi
fi

# Setup PYTHONPATH for proper imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Run the import test script first
echo "Testing imports..."
python test_imports.py
TEST_RESULT=$?

# Run the dashboard
read -p "Run the dashboard now? (y/n): " run_dash
if [[ $run_dash == "y" || $run_dash == "Y" ]]; then
    if [ $TEST_RESULT -eq 0 ]; then
        echo "Starting dashboard..."
        python main_dashboard.py
    else
        echo "Import tests failed. Please fix the issues before running the dashboard."
        echo "You can try reinstalling the package with: pip install -e ."
    fi
else
    echo "To run the dashboard later, use: conda activate fib_cycles && python main_dashboard.py"
fi