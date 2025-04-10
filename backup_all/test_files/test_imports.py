"""
Test script to verify that all imports for the dashboard are working correctly.
Run this script to test if the system can import all required modules without errors.
"""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all necessary imports for the dashboard"""
    logger.info("Testing imports for Fibonacci Harmonic Trading System...")
    
    try:
        # Add project root to path explicitly
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        logger.info(f"Project root added to path: {project_root}")
        
        # Print current Python path for debugging
        logger.info(f"Python path: {sys.path}")
        
        # Test basic Python libraries
        import numpy
        import pandas
        import matplotlib
        logger.info("✓ Basic Python libraries imported successfully")
        
        # Test dash libraries
        import dash
        import dash_bootstrap_components
        logger.info("✓ Dash libraries imported successfully")
        
        # Test core modules
        from core.scanner import FibCycleScanner
        logger.info("✓ FibCycleScanner imported successfully")
        
        from core.cycle_detection import CycleDetector
        logger.info("✓ CycleDetector imported successfully")
        
        from core.fld_signal_generator import FLDCalculator, SignalGenerator
        logger.info("✓ FLD components imported successfully")
        
        # Test model modules
        from models.scan_parameters import ScanParameters
        logger.info("✓ ScanParameters imported successfully")
        
        from models.scan_result import ScanResult
        logger.info("✓ ScanResult imported successfully")
        
        # Test utility modules
        from utils.config import load_config
        logger.info("✓ Utility modules imported successfully")
        
        # Test storage modules
        from storage.results_repository import ResultsRepository
        logger.info("✓ Storage modules imported successfully")
        
        # Test web UI modules
        from web.cycle_visualization import create_cycle_visualization
        from web.fld_visualization import create_fld_visualization
        from web.harmonic_visualization import create_harmonic_visualization
        from web.scanner_dashboard import create_scanner_dashboard
        from web.trading_strategies_ui import create_strategy_dashboard
        logger.info("✓ Web UI modules imported successfully")
        
        # Test visualization modules
        from visualization.price_charts import generate_plot_image
        logger.info("✓ Visualization modules imported successfully")
        
        # Test main dashboard
        import main_dashboard
        logger.info("✓ Main dashboard module imported successfully")
        
        logger.info("All imports successful! The system should be ready to run.")
        return True
    
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ All imports successful. The dashboard should be ready to run.")
        print("To start the dashboard, run: python main_dashboard.py")
    else:
        print("\n❌ Some imports failed. Please check the logs above for details.")
        print("Resolve the issues before running the dashboard application.")