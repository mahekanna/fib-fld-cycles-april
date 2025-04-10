#!/usr/bin/env python
"""
Validation utility for data sources in the Fibonacci Cycle Trading System.
This module helps identify which data sources are available and provides
feedback on missing dependencies.
"""

import importlib
import logging
import os
import sys
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSourceValidator:
    """Validator for checking and reporting on data source availability."""
    
    def __init__(self):
        """Initialize the validator."""
        self.data_sources = {
            'yfinance': {
                'name': 'Yahoo Finance',
                'import_name': 'yfinance',
                'install_cmd': 'pip install yfinance',
                'available': False
            },
            'tvdatafeed': {
                'name': 'TradingView',
                'import_name': 'tvdatafeed',
                'install_cmd': 'pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git',
                'available': False
            },
            'alpha_vantage': {
                'name': 'Alpha Vantage',
                'import_name': 'alpha_vantage',
                'install_cmd': 'pip install alpha_vantage',
                'available': False
            },
            'pandas': {
                'name': 'Pandas',
                'import_name': 'pandas',
                'install_cmd': 'pip install pandas',
                'available': False
            },
            'numpy': {
                'name': 'NumPy',
                'import_name': 'numpy',
                'install_cmd': 'pip install numpy',
                'available': False
            }
        }
        
        # Check cache directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = os.path.join(self.project_root, 'data', 'cache')
        self.cache_available = os.path.exists(self.cache_dir)
        self.cached_files = []
        
        if self.cache_available:
            try:
                self.cached_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.csv')]
            except Exception as e:
                logger.error(f"Error reading cache directory: {e}")
    
    def check_dependencies(self) -> Dict:
        """
        Check which data source dependencies are available.
        
        Returns:
            Dictionary with validation results
        """
        # Check each data source
        for source_key, source_info in self.data_sources.items():
            try:
                # Try to import the module
                importlib.import_module(source_info['import_name'])
                self.data_sources[source_key]['available'] = True
                logger.info(f"✓ {source_info['name']} is available")
            except ImportError:
                logger.info(f"✗ {source_info['name']} is not available")
        
        # Check additional dependencies for data processing
        self._check_additional_dependencies()
        
        return {
            'data_sources': self.data_sources,
            'cache_available': self.cache_available,
            'cached_files': self.cached_files,
            'mock_generator_available': self._is_mock_generator_available()
        }
    
    def _check_additional_dependencies(self):
        """Check additional dependencies that might affect data handling."""
        # Check for matplotlib (for plotting)
        try:
            importlib.import_module('matplotlib')
            self.data_sources['matplotlib'] = {
                'name': 'Matplotlib',
                'import_name': 'matplotlib',
                'install_cmd': 'pip install matplotlib',
                'available': True
            }
        except ImportError:
            self.data_sources['matplotlib'] = {
                'name': 'Matplotlib',
                'import_name': 'matplotlib',
                'install_cmd': 'pip install matplotlib',
                'available': False
            }
            
        # Check for scipy (for FFT)
        try:
            importlib.import_module('scipy')
            self.data_sources['scipy'] = {
                'name': 'SciPy',
                'import_name': 'scipy',
                'install_cmd': 'pip install scipy',
                'available': True
            }
        except ImportError:
            self.data_sources['scipy'] = {
                'name': 'SciPy',
                'import_name': 'scipy',
                'install_cmd': 'pip install scipy',
                'available': False
            }
            
        # Check for talib
        try:
            importlib.import_module('talib')
            self.data_sources['talib'] = {
                'name': 'TA-Lib',
                'import_name': 'talib',
                'install_cmd': 'pip install ta-lib',
                'available': True
            }
        except ImportError:
            self.data_sources['talib'] = {
                'name': 'TA-Lib',
                'import_name': 'talib',
                'install_cmd': 'pip install ta-lib',
                'available': False
            }
    
    def _is_mock_generator_available(self) -> bool:
        """Check if the mock data generator is available."""
        mock_generator_path = os.path.join(self.project_root, 'data', 'mock_data_generator.py')
        return os.path.exists(mock_generator_path)
    
    def get_missing_dependencies(self) -> List[Dict]:
        """
        Get a list of missing but important dependencies.
        
        Returns:
            List of dictionaries with missing dependency information
        """
        missing = []
        
        for source_key, source_info in self.data_sources.items():
            if not source_info['available']:
                missing.append(source_info)
        
        return missing
    
    def get_install_commands(self) -> str:
        """
        Generate commands to install missing dependencies.
        
        Returns:
            String with installation commands
        """
        commands = []
        
        for source_key, source_info in self.data_sources.items():
            if not source_info['available']:
                commands.append(source_info['install_cmd'])
        
        if not commands:
            return "# All dependencies are installed!"
        
        return "\n".join(commands)
    
    def get_data_options(self) -> List[Tuple[str, str]]:
        """
        Get a list of available data sources with descriptions.
        
        Returns:
            List of tuples (name, description)
        """
        options = []
        
        # Check for real data sources
        if self.data_sources['yfinance']['available']:
            options.append(('yfinance', 'Yahoo Finance (good for US stocks)'))
        
        if self.data_sources['tvdatafeed']['available']:
            options.append(('tvdatafeed', 'TradingView (best for global markets)'))
        
        if self.data_sources['alpha_vantage']['available']:
            options.append(('alpha_vantage', 'Alpha Vantage API'))
        
        # Check for cache/mock options
        if self.cache_available and self.cached_files:
            options.append(('cache', f'Local cache ({len(self.cached_files)} files available)'))
        
        if self._is_mock_generator_available():
            options.append(('mock', 'Generate mock data with embedded cycles'))
        
        return options
    
    def generate_recommendation(self) -> str:
        """
        Generate a recommendation based on available data sources.
        
        Returns:
            String with recommendation
        """
        options = self.get_data_options()
        missing = self.get_missing_dependencies()
        
        if not options:
            return ("No data sources available. Please install at least one data source:\n" +
                   self.get_install_commands())
        
        recommendation = "Available data sources:\n"
        for option, desc in options:
            recommendation += f"- {option}: {desc}\n"
        
        if missing:
            recommendation += "\nMissing dependencies:\n"
            for dep in missing:
                recommendation += f"- {dep['name']}: {dep['install_cmd']}\n"
        
        recommendation += "\nRecommendation: "
        
        if 'tvdatafeed' in [o[0] for o in options]:
            recommendation += "Use TradingView as your primary data source for best results."
        elif 'yfinance' in [o[0] for o in options]:
            recommendation += "Use Yahoo Finance for US stocks and major indices."
        elif 'cache' in [o[0] for o in options]:
            recommendation += "Use cached data for testing. Install a real data source for production use."
        elif 'mock' in [o[0] for o in options]:
            recommendation += "Only mock data is available. Install a real data source for actual trading."
        
        return recommendation


def verify_data_sources() -> Dict:
    """
    Verify available data sources and return results.
    
    Returns:
        Dictionary with verification results
    """
    validator = DataSourceValidator()
    return validator.check_dependencies()


def print_data_source_status():
    """Print a summary of data source availability."""
    validator = DataSourceValidator()
    validator.check_dependencies()
    
    print("\n=== Fibonacci Cycle Trading System Data Source Status ===\n")
    
    # Print available data sources
    options = validator.get_data_options()
    if options:
        print("Available data sources:")
        for option, desc in options:
            print(f"- {option}: {desc}")
    else:
        print("No data sources available.")
    
    # Print cache status
    if validator.cache_available:
        print(f"\nCache directory: {validator.cache_dir}")
        if validator.cached_files:
            print(f"Cached files: {len(validator.cached_files)}")
            # Print a sample of cached files
            if len(validator.cached_files) > 5:
                sample = validator.cached_files[:5]
                print(f"Sample files: {', '.join(sample)}, ...")
            else:
                print(f"Files: {', '.join(validator.cached_files)}")
        else:
            print("No cached files found.")
    else:
        print("\nCache directory not found.")
    
    # Print mock data generator status
    if validator._is_mock_generator_available():
        print("\nMock data generator is available.")
    else:
        print("\nMock data generator is not available.")
    
    # Print recommendation
    print("\n" + validator.generate_recommendation())
    
    # Print install commands for missing dependencies
    missing = validator.get_missing_dependencies()
    if missing:
        print("\nTo install missing dependencies, run:")
        print(validator.get_install_commands())
    
    print("\nFor more information, see data_sources_setup.md")


if __name__ == "__main__":
    print_data_source_status()