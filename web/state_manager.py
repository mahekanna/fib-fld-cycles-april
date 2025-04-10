"""
State Manager for Fibonacci Cycles System

This module implements a centralized state management system to ensure data consistency
across all components of the dashboard, particularly between tabs.

It uses a singleton pattern to provide global access to consistent state and implements
proper data snapshot mechanisms to prevent data refreshing issues.
"""

import copy
import logging
import threading
import time
from typing import Dict, List, Any, Optional
import pickle
import os
import json
from datetime import datetime

# Configure logging
try:
    from utils.logging_utils import get_component_logger
    logger = get_component_logger("web.state_manager")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("web.state_manager")

# STATE PERSISTENCE FILE - Used to save state between application restarts if needed
STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "state_snapshot.pkl")

class StateManager:
    """
    Singleton State Manager to ensure consistent state across components.
    
    This class maintains immutable snapshots of data and provides a single source of truth
    for all components in the application.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StateManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the state manager with empty state containers"""
        # Master data snapshots - immutable complete copies of results
        self.result_snapshots = {}
        
        # Tracks which batch is currently active
        self.active_batch_id = None
        
        # Collection of all batches by timestamp
        self.batch_collection = {}
        
        # Track last modification time to detect changes
        self.last_modified = time.time()
        
        # Flag for strict mode (completely disable data refreshing)
        self.strict_mode = True
        
        # Metadata for debugging
        self.debug_info = {
            "created_at": datetime.now().isoformat(),
            "snapshot_count": 0,
            "last_batch_id": None
        }
        
        logger.info("🔐 State Manager initialized in strict consistency mode")
        
        # Try to load persisted state if it exists
        self._load_persisted_state()
    
    def _load_persisted_state(self):
        """Attempt to load previously persisted state"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'rb') as f:
                    state = pickle.load(f)
                    
                self.result_snapshots = state.get('result_snapshots', {})
                self.active_batch_id = state.get('active_batch_id', None)
                self.batch_collection = state.get('batch_collection', {})
                
                logger.info(f"🔄 Loaded persisted state with {len(self.result_snapshots)} snapshots")
                self.debug_info["loaded_from_disk"] = True
                self.debug_info["snapshot_count"] = len(self.result_snapshots)
                
            except Exception as e:
                logger.error(f"❌ Failed to load persisted state: {e}")
                # Reset to empty state
                self.result_snapshots = {}
                self.active_batch_id = None
                self.batch_collection = {}
    
    def persist_state(self):
        """Persist the current state to disk for potential recovery"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            
            with open(STATE_FILE, 'wb') as f:
                state = {
                    'result_snapshots': self.result_snapshots,
                    'active_batch_id': self.active_batch_id,
                    'batch_collection': self.batch_collection,
                    'persisted_at': datetime.now().isoformat()
                }
                pickle.dump(state, f)
                
            logger.info(f"💾 Persisted state with {len(self.result_snapshots)} snapshots")
            
        except Exception as e:
            logger.error(f"❌ Failed to persist state: {e}")
    
    def register_batch_results(self, results_list, batch_id=None):
        """
        Register a new batch of results and create immutable snapshots.
        
        Args:
            results_list: List of ScanResult objects
            batch_id: Optional batch identifier. If None, a timestamp will be created.
            
        Returns:
            batch_id: The identifier for this batch
        """
        if not results_list:
            logger.warning("⚠️ Attempted to register empty results list")
            return None
            
        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create deep copies (immutable snapshots) of all results
        batch_snapshots = {}
        for result in results_list:
            if not hasattr(result, 'symbol') or not result.success:
                continue
                
            # Create complete deep copy
            try:
                batch_snapshots[result.symbol] = copy.deepcopy(result)
                logger.info(f"📸 Created snapshot for {result.symbol} @ price {getattr(result, 'price', 'N/A')}")
            except Exception as e:
                logger.error(f"❌ Failed to create snapshot for {getattr(result, 'symbol', 'unknown')}: {e}")
        
        # Store batch in collection
        self.batch_collection[batch_id] = {
            'created_at': datetime.now().isoformat(),
            'snapshots': batch_snapshots,
            'symbol_count': len(batch_snapshots)
        }
        
        # Update result snapshots with this batch
        self.result_snapshots.update(batch_snapshots)
        
        # Set as active batch
        self.active_batch_id = batch_id
        
        # Update metadata
        self.last_modified = time.time()
        self.debug_info["snapshot_count"] = len(self.result_snapshots)
        self.debug_info["last_batch_id"] = batch_id
        
        logger.warning(f"🔒 Registered batch {batch_id} with {len(batch_snapshots)} results")
        
        # Persist to disk for recovery
        self.persist_state()
        
        return batch_id
    
    def get_batch_results(self, batch_id=None):
        """
        Get immutable snapshots of batch results.
        
        Args:
            batch_id: Batch identifier. If None, uses the active batch.
            
        Returns:
            List of ScanResult objects (immutable snapshots)
        """
        # Use active batch if none specified
        if batch_id is None:
            batch_id = self.active_batch_id
        
        if batch_id is None or batch_id not in self.batch_collection:
            logger.warning(f"⚠️ Requested unknown batch ID: {batch_id}")
            return []
        
        # Get the snapshots for this batch
        batch_data = self.batch_collection[batch_id]
        snapshots = batch_data['snapshots']
        
        logger.info(f"🔍 Retrieved {len(snapshots)} snapshots for batch {batch_id}")
        
        # Return list of snapshot objects
        return list(snapshots.values())
    
    def get_current_snapshots(self):
        """
        Get all current result snapshots.
        
        Returns:
            Dict mapping symbol to result snapshot
        """
        return self.result_snapshots
    
    def get_snapshot_for_symbol(self, symbol):
        """
        Get snapshot for a specific symbol.
        
        Args:
            symbol: Symbol to retrieve
            
        Returns:
            ScanResult object (immutable snapshot) or None if not found
        """
        return self.result_snapshots.get(symbol)
    
    def get_info(self):
        """Get debug information about the state manager"""
        info = copy.copy(self.debug_info)
        info.update({
            "current_time": datetime.now().isoformat(),
            "snapshot_count": len(self.result_snapshots),
            "batch_count": len(self.batch_collection),
            "active_batch": self.active_batch_id,
            "symbols": list(self.result_snapshots.keys())[:5] + ["..."] if len(self.result_snapshots) > 5 else list(self.result_snapshots.keys()),
            "scan_state": self.scan_state if hasattr(self, 'scan_state') else None
        })
        return info
    
    def clear_state(self):
        """Clear all state (for testing or reset)"""
        self.result_snapshots = {}
        self.active_batch_id = None
        self.batch_collection = {}
        self.last_modified = time.time()
        self.debug_info["snapshot_count"] = 0
        self.debug_info["last_batch_id"] = None
        if hasattr(self, 'scan_state'):
            self.scan_state = None
        logger.warning("🧹 State manager cleared")
        
    # INCREMENTAL SCANNING SUPPORT
    def initialize_scan_state(self, symbols, exchange, interval):
        """Initialize a new scan state for a list of symbols"""
        scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.scan_state = {
            "id": scan_id,
            "total_symbols": len(symbols),
            "completed_symbols": 0,
            "pending_symbols": symbols.copy(),
            "completed_symbols_list": [],
            "failed_symbols": [],
            "exchange": exchange,
            "interval": interval,
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "status": "initialized"
        }
        
        logger.info(f"🔄 Initialized incremental scan {scan_id} with {len(symbols)} symbols")
        return scan_id
        
    def update_scan_progress(self, symbol, success=True, error=None):
        """Update scan progress for a symbol"""
        if not hasattr(self, 'scan_state') or not self.scan_state:
            logger.warning("⚠️ Attempted to update scan progress but no scan is in progress")
            return False
            
        if symbol in self.scan_state["pending_symbols"]:
            self.scan_state["pending_symbols"].remove(symbol)
            
            if success:
                self.scan_state["completed_symbols"] += 1
                self.scan_state["completed_symbols_list"].append(symbol)
                logger.info(f"✅ Symbol {symbol} scan completed successfully")
            else:
                self.scan_state["failed_symbols"].append({"symbol": symbol, "error": str(error)})
                logger.warning(f"❌ Symbol {symbol} scan failed: {error}")
                
        self.scan_state["last_updated"] = datetime.now().isoformat()
        self.scan_state["status"] = "in_progress"
        
        # Calculate completion percentage
        total = self.scan_state["total_symbols"]
        completed = self.scan_state["completed_symbols"]
        percentage = (completed / total) * 100 if total > 0 else 0
        self.scan_state["completion_percentage"] = round(percentage, 2)
        
        # Check if scan is complete
        if not self.scan_state["pending_symbols"]:
            self.scan_state["status"] = "completed"
            self.scan_state["end_time"] = datetime.now().isoformat()
            logger.info(f"🏁 Scan {self.scan_state['id']} completed. Success: {completed}, Failed: {len(self.scan_state['failed_symbols'])}")
            
        return True
        
    def get_scan_state(self):
        """Get the current scan state"""
        if not hasattr(self, 'scan_state') or not self.scan_state:
            return None
            
        return self.scan_state
        
    def get_remaining_symbols(self):
        """Get list of symbols remaining to be scanned"""
        if not hasattr(self, 'scan_state') or not self.scan_state:
            return []
            
        return self.scan_state["pending_symbols"]
        
    def pause_scan(self):
        """Pause the current scan"""
        if not hasattr(self, 'scan_state') or not self.scan_state:
            return False
            
        self.scan_state["status"] = "paused"
        self.scan_state["paused_at"] = datetime.now().isoformat()
        logger.info(f"⏸️ Scan {self.scan_state['id']} paused")
        return True
        
    def resume_scan(self):
        """Resume a paused scan"""
        if not hasattr(self, 'scan_state') or not self.scan_state:
            return False
            
        if self.scan_state["status"] != "paused":
            return False
            
        self.scan_state["status"] = "in_progress"
        self.scan_state["resumed_at"] = datetime.now().isoformat()
        logger.info(f"▶️ Scan {self.scan_state['id']} resumed with {len(self.scan_state['pending_symbols'])} symbols remaining")
        return True

# Create singleton instance
state_manager = StateManager()