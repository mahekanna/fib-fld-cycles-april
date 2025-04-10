"""
Test script for verifying the state management system
"""

import os
import sys
import unittest
import logging
from datetime import datetime
import copy

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_state_management")

# Import the state manager
from web.state_manager import state_manager, StateManager

# Import models for testing
from models.scan_result import ScanResult

class TestStateManagement(unittest.TestCase):
    """Test state management system for consistent price handling"""
    
    def setUp(self):
        """Set up test environment"""
        # Clear state manager for tests
        state_manager.clear_state()
        logger.info("Cleared state manager for testing")
    
    def test_singleton_pattern(self):
        """Test that StateManager follows singleton pattern"""
        # Getting StateManager instance again should return the same object
        new_manager = StateManager()
        self.assertIs(state_manager, new_manager, "StateManager should be a singleton")
        
    def test_batch_registration(self):
        """Test registering batch results"""
        # Create mock results
        results = []
        for i in range(5):
            result = ScanResult()
            result.symbol = f"TEST{i}"
            result.price = 100.0 + i
            result.success = True
            results.append(result)
        
        # Register with state manager
        batch_id = state_manager.register_batch_results(results)
        
        # Verify batch was registered
        self.assertIsNotNone(batch_id, "Batch ID should not be None")
        self.assertEqual(state_manager.active_batch_id, batch_id, "Batch should be set as active")
        self.assertEqual(len(state_manager.result_snapshots), 5, "Should have 5 snapshots")
        
    def test_immutability(self):
        """Test immutability of snapshots"""
        # Create a mock result
        result = ScanResult()
        result.symbol = "TEST"
        result.price = 100.0
        result.success = True
        
        # Register with state manager
        batch_id = state_manager.register_batch_results([result])
        
        # Get snapshot back
        snapshots = state_manager.get_current_snapshots()
        snapshot = snapshots["TEST"]
        
        # Modify original result
        result.price = 200.0
        
        # Snapshot should be unchanged
        self.assertEqual(snapshot.price, 100.0, "Snapshot should be immutable")
        
    def test_batch_retrieval(self):
        """Test retrieving batch results"""
        # Create mock results
        results = []
        for i in range(3):
            result = ScanResult()
            result.symbol = f"TEST{i}"
            result.price = 100.0 + i
            result.success = True
            results.append(result)
        
        # Register with state manager
        batch_id = state_manager.register_batch_results(results)
        
        # Retrieve batch results
        batch_results = state_manager.get_batch_results(batch_id)
        
        # Should get same number of results
        self.assertEqual(len(batch_results), 3, "Should get 3 results back")
        
        # Verify values
        for i, result in enumerate(batch_results):
            self.assertEqual(result.symbol, f"TEST{i}", f"Symbol should be TEST{i}")
            self.assertEqual(result.price, 100.0 + i, f"Price should be {100.0 + i}")
        
    def test_persistence(self):
        """Test state persistence"""
        # Create a mock result
        result = ScanResult()
        result.symbol = "TEST"
        result.price = 100.0
        result.success = True
        
        # Register and persist
        batch_id = state_manager.register_batch_results([result])
        state_manager.persist_state()
        
        # Create a new state manager instance (bypassing singleton for test)
        # This simulates reloading from disk
        test_instance = StateManager.__new__(StateManager)
        test_instance._initialize()
        test_instance._load_persisted_state()
        
        # Verify state was loaded
        self.assertEqual(len(test_instance.result_snapshots), 1, "Should have 1 snapshot after loading")
        self.assertEqual(test_instance.result_snapshots["TEST"].price, 100.0, "Price should be preserved")

if __name__ == "__main__":
    logger.info("Running state management tests...")
    unittest.main()