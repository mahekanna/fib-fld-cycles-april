import logging
import time
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import threading
import queue

class BrokerBase(ABC):
    """
    Abstract base class for broker API integration.
    Provides standard interface for different broker implementations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the broker API client.
        
        Args:
            config: Configuration dictionary for broker
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.account_info = {}
        self.positions = {}
        self.orders = {}
        
        # Event handling
        self.callbacks = {
            'order_update': [],
            'position_update': [],
            'trade_update': [],
            'account_update': [],
            'connection_status': []
        }
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker API.
        
        Returns:
            Boolean indicating connection success
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the broker API.
        
        Returns:
            Boolean indicating disconnection success
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Dictionary containing account information
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict:
        """
        Get current positions.
        
        Returns:
            Dictionary mapping symbols to position information
        """
        pass
    
    @abstractmethod
    def get_orders(self, status: Optional[str] = None) -> Dict:
        """
        Get current orders.
        
        Args:
            status: Optional status filter
            
        Returns:
            Dictionary mapping order IDs to order information
        """
        pass
    
    @abstractmethod
    def place_order(self, 
                   symbol: str, 
                   side: str, 
                   order_type: str, 
                   quantity: float,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = 'DAY',
                   **kwargs) -> Dict:
        """
        Place a new order.
        
        Args:
            symbol: Symbol to trade
            side: Order side ("BUY" or "SELL")
            order_type: Order type ("MARKET", "LIMIT", "STOP", "STOP_LIMIT")
            quantity: Order quantity
            price: Optional limit price
            stop_price: Optional stop price
            time_in_force: Time in force ("DAY", "GTC", "IOC", "FOK")
            kwargs: Additional broker-specific parameters
            
        Returns:
            Dictionary containing order information
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Boolean indicating cancellation success
        """
        pass
    
    @abstractmethod
    def modify_order(self, 
                    order_id: str, 
                    quantity: Optional[float] = None,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    **kwargs) -> Dict:
        """
        Modify an existing order.
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            stop_price: New stop price
            kwargs: Additional broker-specific parameters
            
        Returns:
            Dictionary containing updated order information
        """
        pass
    
    @abstractmethod
    def get_market_data(self, 
                       symbol: str, 
                       data_type: str = 'QUOTE') -> Dict:
        """
        Get real-time market data.
        
        Args:
            symbol: Symbol to get data for
            data_type: Type of data ("QUOTE", "TRADE", "BOOK")
            
        Returns:
            Dictionary containing market data
        """
        pass
    
    def register_callback(self, event_type: str, callback: callable) -> None:
        """
        Register a callback function for a specific event.
        
        Args:
            event_type: Type of event ("order_update", "position_update", etc.)
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            self.logger.info(f"Registered callback for {event_type}")
        else:
            self.logger.error(f"Unknown event type: {event_type}")
    
    def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """
        Notify registered callbacks for an event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in {event_type} callback: {e}")


class InteractiveBrokersBridge(BrokerBase):
    """
    Implementation of the BrokerBase for Interactive Brokers.
    Uses ib_insync library for communication.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Interactive Brokers API client.
        
        Args:
            config: Configuration dictionary for IB connection
        """
        super().__init__(config)
        
        try:
            from ib_insync import IB, Contract, Order, util
            self.IB = IB
            self.Contract = Contract
            self.Order = Order
            self.util = util
        except ImportError:
            self.logger.error("ib_insync library not installed. Run: pip install ib_insync")
            raise ImportError("ib_insync library required")
        
        self.ib = self.IB()
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7497)  # 7497 for TWS Paper, 7496 for TWS Live, 4002 for Gateway
        self.client_id = config.get('client_id', 1)
        self.timeout = config.get('timeout', 30)
        
        # Contract cache for faster lookups
        self.contract_cache = {}
        
        # Background thread for event processing
        self.event_thread = None
        self.event_queue = queue.Queue()
        self.running = False
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers.
        
        Returns:
            Boolean indicating connection success
        """
        try:
            self.ib.connect(
                self.host, 
                self.port, 
                clientId=self.client_id, 
                timeout=self.timeout,
                readonly=False
            )
            
            self.connected = self.ib.isConnected()
            
            if self.connected:
                self.logger.info(f"Connected to Interactive Brokers at {self.host}:{self.port}")
                
                # Register callbacks
                self.ib.orderStatusEvent += self._handle_order_status
                self.ib.execDetailsEvent += self._handle_execution
                self.ib.positionEvent += self._handle_position
                self.ib.accountSummaryEvent += self._handle_account_summary
                self.ib.error += self._handle_error
                
                # Start event processing thread
                self.running = True
                self.event_thread = threading.Thread(target=self._process_events)
                self.event_thread.daemon = True
                self.event_thread.start()
                
                # Fetch initial account and position information
                self.account_info = self.get_account_info()
                self.positions = self.get_positions()
                
                return True
            else:
                self.logger.error("Failed to connect to Interactive Brokers")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to Interactive Brokers: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Interactive Brokers.
        
        Returns:
            Boolean indicating disconnection success
        """
        try:
            self.running = False
            if self.event_thread and self.event_thread.is_alive():
                self.event_thread.join(timeout=2.0)
            
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from Interactive Brokers")
            
            # Notify callbacks
            self._notify_callbacks('connection_status', {'connected': False})
            
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Interactive Brokers: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """
        Get account information from Interactive Brokers.
        
        Returns:
            Dictionary containing account information
        """
        if not self.connected:
            self.logger.error("Not connected to Interactive Brokers")
            return {}
        
        try:
            account = self.ib.accountSummary()
            
            # Convert to dictionary structure
            result = {}
            for item in account:
                if item.tag not in result:
                    result[item.tag] = {}
                result[item.tag] = item.value
            
            # Notify callbacks
            self._notify_callbacks('account_update', result)
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting account information: {e}")
            return {}
    
    def get_positions(self) -> Dict:
        """
        Get current positions from Interactive Brokers.
        
        Returns:
            Dictionary mapping symbols to position information
        """
        if not self.connected:
            self.logger.error("Not connected to Interactive Brokers")
            return {}
        
        try:
            positions = self.ib.positions()
            
            # Convert to dictionary structure
            result = {}
            for pos in positions:
                symbol = self._get_symbol_from_contract(pos.contract)
                result[symbol] = {
                    'symbol': symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.marketValue if hasattr(pos, 'marketValue') else None,
                    'market_price': pos.marketPrice if hasattr(pos, 'marketPrice') else None,
                    'contract': pos.contract
                }
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_orders(self, status: Optional[str] = None) -> Dict:
        """
        Get current orders from Interactive Brokers.
        
        Args:
            status: Optional status filter
            
        Returns:
            Dictionary mapping order IDs to order information
        """
        if not self.connected:
            self.logger.error("Not connected to Interactive Brokers")
            return {}
        
        try:
            orders = self.ib.openOrders()
            
            # Convert to dictionary structure
            result = {}
            for order in orders:
                # Filter by status if specified
                if status and order.orderStatus.status != status:
                    continue
                
                order_id = str(order.orderId)
                symbol = self._get_symbol_from_contract(order.contract)
                
                result[order_id] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'action': order.order.action,  # 'BUY' or 'SELL'
                    'order_type': order.order.orderType,
                    'quantity': order.order.totalQuantity,
                    'filled_quantity': order.orderStatus.filled,
                    'remaining_quantity': order.orderStatus.remaining,
                    'price': order.order.lmtPrice if hasattr(order.order, 'lmtPrice') else None,
                    'stop_price': order.order.auxPrice if hasattr(order.order, 'auxPrice') else None,
                    'status': order.orderStatus.status,
                    'order': order
                }
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return {}
    
    def place_order(self, 
                   symbol: str, 
                   side: str, 
                   order_type: str, 
                   quantity: float,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = 'DAY',
                   **kwargs) -> Dict:
        """
        Place a new order with Interactive Brokers.
        
        Args:
            symbol: Symbol to trade
            side: Order side ("BUY" or "SELL")
            order_type: Order type ("MARKET", "LIMIT", "STOP", "STOP_LIMIT")
            quantity: Order quantity
            price: Optional limit price
            stop_price: Optional stop price
            time_in_force: Time in force ("DAY", "GTC", "IOC", "FOK")
            kwargs: Additional IB-specific parameters
            
        Returns:
            Dictionary containing order information
        """
        if not self.connected:
            self.logger.error("Not connected to Interactive Brokers")
            return {}
        
        try:
            # Get or create contract
            contract = self._get_contract(symbol)
            
            # Map order type to IB order type
            ib_order_type_map = {
                'MARKET': 'MKT',
                'LIMIT': 'LMT',
                'STOP': 'STP',
                'STOP_LIMIT': 'STP LMT'
            }
            ib_order_type = ib_order_type_map.get(order_type, 'MKT')
            
            # Map time in force to IB time in force
            ib_tif_map = {
                'DAY': 'DAY',
                'GTC': 'GTC',
                'IOC': 'IOC',
                'FOK': 'FOK'
            }
            ib_tif = ib_tif_map.get(time_in_force, 'DAY')
            
            # Create order object
            order = self.Order()
            order.action = side
            order.totalQuantity = quantity
            order.orderType = ib_order_type
            order.tif = ib_tif
            
            # Set price for limit orders
            if order_type in ['LIMIT', 'STOP_LIMIT'] and price is not None:
                order.lmtPrice = price
            
            # Set stop price for stop orders
            if order_type in ['STOP', 'STOP_LIMIT'] and stop_price is not None:
                order.auxPrice = stop_price
            
            # Add additional parameters
            for key, value in kwargs.items():
                setattr(order, key, value)
            
            # Submit order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order to be acknowledged
            for _ in range(10):  # Wait up to 1 second
                if trade.orderStatus.status != 'PendingSubmit':
                    break
                time.sleep(0.1)
                self.ib.sleep(0)
            
            # Return order information
            return {
                'order_id': str(trade.order.orderId),
                'symbol': symbol,
                'action': side,
                'order_type': order_type,
                'quantity': quantity,
                'price': price,
                'stop_price': stop_price,
                'status': trade.orderStatus.status,
                'message': trade.orderStatus.lastMessage,
                'filled_quantity': trade.orderStatus.filled,
                'remaining_quantity': trade.orderStatus.remaining,
                'trade': trade
            }
        
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order with Interactive Brokers.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Boolean indicating cancellation success
        """
        if not self.connected:
            self.logger.error("Not connected to Interactive Brokers")
            return False
        
        try:
            # Convert order_id to integer
            ib_order_id = int(order_id)
            
            # Find the order
            orders = self.ib.openOrders()
            order_to_cancel = None
            
            for order in orders:
                if order.orderId == ib_order_id:
                    order_to_cancel = order
                    break
            
            if not order_to_cancel:
                self.logger.error(f"Order {order_id} not found")
                return False
            
            # Cancel the order
            self.ib.cancelOrder(order_to_cancel.order)
            
            # Wait for cancellation to be acknowledged
            for _ in range(10):  # Wait up to 1 second
                orders = self.ib.openOrders()
                found = False
                for order in orders:
                    if order.orderId == ib_order_id:
                        found = True
                        if order.orderStatus.status == 'Cancelled':
                            return True
                if not found:
                    return True
                time.sleep(0.1)
                self.ib.sleep(0)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def modify_order(self, 
                    order_id: str, 
                    quantity: Optional[float] = None,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    **kwargs) -> Dict:
        """
        Modify an existing order with Interactive Brokers.
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            stop_price: New stop price
            kwargs: Additional IB-specific parameters
            
        Returns:
            Dictionary containing updated order information
        """
        if not self.connected:
            self.logger.error("Not connected to Interactive Brokers")
            return {}
        
        try:
            # Convert order_id to integer
            ib_order_id = int(order_id)
            
            # Find the order
            orders = self.ib.openOrders()
            order_to_modify = None
            
            for order in orders:
                if order.orderId == ib_order_id:
                    order_to_modify = order
                    break
            
            if not order_to_modify:
                self.logger.error(f"Order {order_id} not found")
                return {'error': f"Order {order_id} not found"}
            
            # Create a new order based on the existing one
            new_order = self.Order()
            for prop in vars(order_to_modify.order):
                if prop.startswith('__'):
                    continue
                setattr(new_order, prop, getattr(order_to_modify.order, prop))
            
            # Update order parameters
            if quantity is not None:
                new_order.totalQuantity = quantity
            
            if price is not None and hasattr(new_order, 'lmtPrice'):
                new_order.lmtPrice = price
            
            if stop_price is not None and hasattr(new_order, 'auxPrice'):
                new_order.auxPrice = stop_price
            
            # Add additional parameters
            for key, value in kwargs.items():
                setattr(new_order, key, value)
            
            # Cancel original order
            self.ib.cancelOrder(order_to_modify.order)
            
            # Submit new order
            trade = self.ib.placeOrder(order_to_modify.contract, new_order)
            
            # Wait for order to be acknowledged
            for _ in range(10):  # Wait up to 1 second
                if trade.orderStatus.status != 'PendingSubmit':
                    break
                time.sleep(0.1)
                self.ib.sleep(0)
            
            # Return order information
            return {
                'order_id': str(trade.order.orderId),
                'symbol': self._get_symbol_from_contract(order_to_modify.contract),
                'action': trade.order.action,
                'order_type': trade.order.orderType,
                'quantity': trade.order.totalQuantity,
                'price': trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
                'stop_price': trade.order.auxPrice if hasattr(trade.order, 'auxPrice') else None,
                'status': trade.orderStatus.status,
                'message': trade.orderStatus.lastMessage,
                'filled_quantity': trade.orderStatus.filled,
                'remaining_quantity': trade.orderStatus.remaining,
                'trade': trade
            }
        
        except Exception as e:
            self.logger.error(f"Error modifying order: {e}")
            return {'error': str(e)}
    
    def get_market_data(self, 
                       symbol: str, 
                       data_type: str = 'QUOTE') -> Dict:
        """
        Get real-time market data from Interactive Brokers.
        
        Args:
            symbol: Symbol to get data for
            data_type: Type of data ("QUOTE", "TRADE", "BOOK")
            
        Returns:
            Dictionary containing market data
        """
        if not self.connected:
            self.logger.error("Not connected to Interactive Brokers")
            return {}
        
        try:
            # Get contract
            contract = self._get_contract(symbol)
            
            if data_type == 'QUOTE':
                # Request market data
                self.ib.reqMktData(contract)
                
                # Wait for data to arrive
                for _ in range(10):  # Wait up to 1 second
                    ticker = self.ib.ticker(contract)
                    if ticker.bid or ticker.ask or ticker.last:
                        break
                    time.sleep(0.1)
                    self.ib.sleep(0)
                
                # Get ticker
                ticker = self.ib.ticker(contract)
                
                # Return quote data
                return {
                    'symbol': symbol,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'last': ticker.last,
                    'volume': ticker.volume,
                    'open': ticker.open,
                    'high': ticker.high,
                    'low': ticker.low,
                    'close': ticker.close,
                    'time': datetime.now().isoformat()
                }
            
            elif data_type == 'BOOK':
                # Request market depth
                self.ib.reqMktDepth(contract)
                
                # Wait for data to arrive
                for _ in range(10):  # Wait up to 1 second
                    depth = self.ib.depthByContract().get(contract, None)
                    if depth and (depth.bids or depth.asks):
                        break
                    time.sleep(0.1)
                    self.ib.sleep(0)
                
                # Get depth data
                depth = self.ib.depthByContract().get(contract, None)
                
                if not depth:
                    return {
                        'symbol': symbol,
                        'bids': [],
                        'asks': []
                    }
                
                # Return order book data
                return {
                    'symbol': symbol,
                    'bids': [{'price': bid.price, 'size': bid.size} for bid in depth.bids],
                    'asks': [{'price': ask.price, 'size': ask.size} for ask in depth.asks],
                    'time': datetime.now().isoformat()
                }
            
            else:
                self.logger.error(f"Unsupported data type: {data_type}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}
    
    def _get_contract(self, symbol: str) -> Any:
        """
        Get or create a contract for a symbol.
        
        Args:
            symbol: Symbol to get contract for
            
        Returns:
            IB Contract object
        """
        # Check cache first
        if symbol in self.contract_cache:
            return self.contract_cache[symbol]
        
        # Parse symbol to extract exchange and type information
        parts = symbol.split(':')
        
        if len(parts) == 1:
            # Default to stock on SMART exchange
            contract = self.Contract(symbol=symbol, secType='STK', exchange='SMART', currency='USD')
        
        elif len(parts) == 2:
            # Format: SYMBOL:EXCHANGE
            sym, exchange = parts
            contract = self.Contract(symbol=sym, secType='STK', exchange=exchange, currency='USD')
        
        elif len(parts) == 3:
            # Format: SYMBOL:EXCHANGE:TYPE
            sym, exchange, sec_type = parts
            contract = self.Contract(symbol=sym, secType=sec_type, exchange=exchange, currency='USD')
        
        else:
            # Handle more complex cases or validation
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        # Resolve contract details
        details = self.ib.reqContractDetails(contract)
        
        if not details:
            raise ValueError(f"Could not resolve contract details for {symbol}")
        
        # Store in cache
        resolved_contract = details[0].contract
        self.contract_cache[symbol] = resolved_contract
        
        return resolved_contract
    
    def _get_symbol_from_contract(self, contract: Any) -> str:
        """
        Get a standardized symbol from a contract.
        
        Args:
            contract: IB Contract object
            
        Returns:
            Standardized symbol string
        """
        if hasattr(contract, 'secType') and contract.secType != 'STK':
            return f"{contract.symbol}:{contract.exchange}:{contract.secType}"
        else:
            return f"{contract.symbol}:{contract.exchange}"
    
    def _handle_order_status(self, trade: Any) -> None:
        """
        Handle order status updates from IB.
        
        Args:
            trade: IB Trade object
        """
        try:
            # Extract order information
            order_id = str(trade.order.orderId)
            symbol = self._get_symbol_from_contract(trade.contract)
            
            order_info = {
                'order_id': order_id,
                'symbol': symbol,
                'action': trade.order.action,
                'order_type': trade.order.orderType,
                'quantity': trade.order.totalQuantity,
                'filled_quantity': trade.orderStatus.filled,
                'remaining_quantity': trade.orderStatus.remaining,
                'price': trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
                'stop_price': trade.order.auxPrice if hasattr(trade.order, 'auxPrice') else None,
                'status': trade.orderStatus.status,
                'message': trade.orderStatus.lastMessage,
                'time': datetime.now().isoformat()
            }
            
            # Add to event queue
            self.event_queue.put(('order_update', order_info))
            
        except Exception as e:
            self.logger.error(f"Error handling order status: {e}")
    
    def _handle_execution(self, trade: Any, fill: Any) -> None:
        """
        Handle execution updates from IB.
        
        Args:
            trade: IB Trade object
            fill: IB Fill object
        """
        try:
            # Extract execution information
            execution_id = fill.execution.execId
            order_id = str(trade.order.orderId)
            symbol = self._get_symbol_from_contract(trade.contract)
            
            execution_info = {
                'execution_id': execution_id,
                'order_id': order_id,
                'symbol': symbol,
                'action': fill.execution.side,
                'quantity': fill.execution.shares,
                'price': fill.execution.price,
                'time': fill.execution.time.isoformat(),
                'commission': fill.commissionReport.commission if hasattr(fill, 'commissionReport') else None
            }
            
            # Add to event queue
            self.event_queue.put(('trade_update', execution_info))
            
        except Exception as e:
            self.logger.error(f"Error handling execution: {e}")
    
    def _handle_position(self, position: Any) -> None:
        """
        Handle position updates from IB.
        
        Args:
            position: IB Position object
        """
        try:
            # Extract position information
            symbol = self._get_symbol_from_contract(position.contract)
            
            position_info = {
                'symbol': symbol,
                'position': position.position,
                'avg_cost': position.avgCost,
                'market_value': position.marketValue if hasattr(position, 'marketValue') else None,
                'market_price': position.marketPrice if hasattr(position, 'marketPrice') else None,
                'time': datetime.now().isoformat()
            }
            
            # Update positions dictionary
            self.positions[symbol] = position_info
            
            # Add to event queue
            self.event_queue.put(('position_update', position_info))
            
        except Exception as e:
            self.logger.error(f"Error handling position update: {e}")
    
    def _handle_account_summary(self, account: Any) -> None:
        """
        Handle account summary updates from IB.
        
        Args:
            account: IB AccountValue object
        """
        try:
            # Extract account information
            account_info = {
                'tag': account.tag,
                'value': account.value,
                'currency': account.currency,
                'account': account.account,
                'time': datetime.now().isoformat()
            }
            
            # Update account info dictionary
            if account.tag not in self.account_info:
                self.account_info[account.tag] = {}
            self.account_info[account.tag] = account.value
            
            # Add to event queue
            self.event_queue.put(('account_update', account_info))
            
        except Exception as e:
            self.logger.error(f"Error handling account summary: {e}")
    
    def _handle_error(self, req_id: int, error_code: int, error_string: str, contract: Any) -> None:
        """
        Handle errors from IB.
        
        Args:
            req_id: Request ID
            error_code: Error code
            error_string: Error message
            contract: Related contract
        """
        # Skip certain informational messages
        if error_code in [2104, 2106, 2158]:  # Market data farm connection messages
            return
        
        symbol = None
        if contract:
            try:
                symbol = self._get_symbol_from_contract(contract)
            except:
                symbol = None
        
        error_info = {
            'req_id': req_id,
            'error_code': error_code,
            'error_message': error_string,
            'symbol': symbol,
            'time': datetime.now().isoformat()
        }
        
        self.logger.error(f"IB API Error {error_code}: {error_string}")
        
        # Add to event queue
        self.event_queue.put(('error', error_info))
    
    def _process_events(self) -> None:
        """
        Process events from the event queue.
        """
        while self.running:
            try:
                # Check for IB messages
                self.ib.sleep(0.1)
                
                # Process any events in queue
                while not self.event_queue.empty():
                    event_type, event_data = self.event_queue.get_nowait()
                    self._notify_callbacks(event_type, event_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in event processing thread: {e}")
                time.sleep(1)  # Sleep to avoid tight error loop


class AlpacaBrokerBridge(BrokerBase):
    """
    Implementation of the BrokerBase for Alpaca Markets.
    Uses alpaca-trade-api library for communication.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Alpaca API client.
        
        Args:
            config: Configuration dictionary for Alpaca connection
        """
        super().__init__(config)
        
        try:
            import alpaca_trade_api as tradeapi
            self.tradeapi = tradeapi
        except ImportError:
            self.logger.error("alpaca-trade-api library not installed. Run: pip install alpaca-trade-api")
            raise ImportError("alpaca-trade-api library required")
        
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        self.api = None
        self.ws = None
        
        # Background thread for WebSocket connection
        self.ws_thread = None
        self.running = False
    
    def connect(self) -> bool:
        """
        Connect to Alpaca Markets.
        
        Returns:
            Boolean indicating connection success
        """
        try:
            # Create REST API connection
            self.api = self.tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Verify connection by getting account info
            account = self.api.get_account()
            
            # Connect to WebSocket for live updates
            stream_url = self.base_url.replace('https://', 'wss://').replace('paper-api.', 'paper-api.data.')
            stream_url = f"{stream_url}/stream"
            
            self.ws = self.tradeapi.Stream(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=stream_url,
                data_stream='alpacadatav2'
            )
            
            # Register WebSocket callbacks
            self.ws.on('trade_updates', self._handle_trade_updates)
            self.ws.on('account_updates', self._handle_account_updates)
            
            # Start WebSocket thread
            self.running = True
            self.ws_thread = threading.Thread(target=self._run_websocket)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            self.connected = True
            self.logger.info(f"Connected to Alpaca Markets API")
            
            # Fetch initial account and position information
            self.account_info = self.get_account_info()
            self.positions = self.get_positions()
            
            # Notify callbacks
            self._notify_callbacks('connection_status', {'connected': True})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to Alpaca Markets: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Alpaca Markets.
        
        Returns:
            Boolean indicating disconnection success
        """
        try:
            self.running = False
            
            if self.ws:
                self.ws.stop()
            
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2.0)
            
            self.connected = False
            self.logger.info("Disconnected from Alpaca Markets")
            
            # Notify callbacks
            self._notify_callbacks('connection_status', {'connected': False})
            
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Alpaca Markets: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """
        Get account information from Alpaca Markets.
        
        Returns:
            Dictionary containing account information
        """
        if not self.connected:
            self.logger.error("Not connected to Alpaca Markets")
            return {}
        
        try:
            account = self.api.get_account()
            
            # Convert to dictionary
            result = {
                'id': account.id,
                'account_number': account.account_number,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': int(account.daytrade_count),
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at.isoformat()
            }
            
            # Notify callbacks
            self._notify_callbacks('account_update', result)
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting account information: {e}")
            return {}
    
    def get_positions(self) -> Dict:
        """
        Get current positions from Alpaca Markets.
        
        Returns:
            Dictionary mapping symbols to position information
        """
        if not self.connected:
            self.logger.error("Not connected to Alpaca Markets")
            return {}
        
        try:
            positions = self.api.list_positions()
            
            # Convert to dictionary structure
            result = {}
            for pos in positions:
                symbol = pos.symbol
                result[symbol] = {
                    'symbol': symbol,
                    'position': float(pos.qty),
                    'avg_cost': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'market_price': float(pos.current_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                }
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_orders(self, status: Optional[str] = None) -> Dict:
        """
        Get current orders from Alpaca Markets.
        
        Args:
            status: Optional status filter
            
        Returns:
            Dictionary mapping order IDs to order information
        """
        if not self.connected:
            self.logger.error("Not connected to Alpaca Markets")
            return {}
        
        try:
            if status:
                orders = self.api.list_orders(status=status)
            else:
                orders = self.api.list_orders()
            
            # Convert to dictionary structure
            result = {}
            for order in orders:
                order_id = order.id
                
                result[order_id] = {
                    'order_id': order_id,
                    'symbol': order.symbol,
                    'action': order.side,
                    'order_type': order.type,
                    'quantity': float(order.qty),
                    'filled_quantity': float(order.filled_qty) if hasattr(order, 'filled_qty') else 0,
                    'remaining_quantity': float(order.qty) - float(order.filled_qty) if hasattr(order, 'filled_qty') else float(order.qty),
                    'price': float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                    'stop_price': float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
                    'status': order.status,
                    'time_in_force': order.time_in_force,
                    'created_at': order.created_at.isoformat(),
                    'raw_order': order
                }
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return {}
    
    def place_order(self, 
                   symbol: str, 
                   side: str, 
                   order_type: str, 
                   quantity: float,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = 'DAY',
                   **kwargs) -> Dict:
        """
        Place a new order with Alpaca Markets.
        
        Args:
            symbol: Symbol to trade
            side: Order side ("BUY" or "SELL")
            order_type: Order type ("MARKET", "LIMIT", "STOP", "STOP_LIMIT")
            quantity: Order quantity
            price: Optional limit price
            stop_price: Optional stop price
            time_in_force: Time in force ("DAY", "GTC", "IOC", "FOK")
            kwargs: Additional Alpaca-specific parameters
            
        Returns:
            Dictionary containing order information
        """
        if not self.connected:
            self.logger.error("Not connected to Alpaca Markets")
            return {}
        
        try:
            # Map order type to Alpaca order type
            alpaca_order_type_map = {
                'MARKET': 'market',
                'LIMIT': 'limit',
                'STOP': 'stop',
                'STOP_LIMIT': 'stop_limit'
            }
            alpaca_order_type = alpaca_order_type_map.get(order_type, 'market')
            
            # Map time in force to Alpaca time in force
            alpaca_tif_map = {
                'DAY': 'day',
                'GTC': 'gtc',
                'IOC': 'ioc',
                'FOK': 'fok'
            }
            alpaca_tif = alpaca_tif_map.get(time_in_force, 'day')
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'qty': quantity,
                'side': side.lower(),
                'type': alpaca_order_type,
                'time_in_force': alpaca_tif
            }
            
            # Add limit price if applicable
            if order_type in ['LIMIT', 'STOP_LIMIT'] and price is not None:
                order_params['limit_price'] = price
            
            # Add stop price if applicable
            if order_type in ['STOP', 'STOP_LIMIT'] and stop_price is not None:
                order_params['stop_price'] = stop_price
            
            # Add additional parameters
            for key, value in kwargs.items():
                order_params[key] = value
            
            # Submit order
            order = self.api.submit_order(**order_params)
            
            # Return order information
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'action': order.side,
                'order_type': order.type,
                'quantity': float(order.qty),
                'price': float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                'stop_price': float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
                'status': order.status,
                'time_in_force': order.time_in_force,
                'created_at': order.created_at.isoformat(),
                'raw_order': order
            }
        
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order with Alpaca Markets.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Boolean indicating cancellation success
        """
        if not self.connected:
            self.logger.error("Not connected to Alpaca Markets")
            return False
        
        try:
            # Cancel the order
            self.api.cancel_order(order_id)
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def modify_order(self, 
                    order_id: str, 
                    quantity: Optional[float] = None,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    **kwargs) -> Dict:
        """
        Modify an existing order with Alpaca Markets.
        Note: Alpaca does not support direct order modification, so we cancel and replace.
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            stop_price: New stop price
            kwargs: Additional Alpaca-specific parameters
            
        Returns:
            Dictionary containing updated order information
        """
        if not self.connected:
            self.logger.error("Not connected to Alpaca Markets")
            return {}
        
        try:
            # Get the existing order
            order = self.api.get_order(order_id)
            
            # Create parameters for new order
            order_params = {
                'symbol': order.symbol,
                'qty': quantity if quantity is not None else float(order.qty),
                'side': order.side,
                'type': order.type,
                'time_in_force': order.time_in_force
            }
            
            # Add limit price if applicable
            if hasattr(order, 'limit_price') and order.limit_price:
                order_params['limit_price'] = price if price is not None else float(order.limit_price)
            
            # Add stop price if applicable
            if hasattr(order, 'stop_price') and order.stop_price:
                order_params['stop_price'] = stop_price if stop_price is not None else float(order.stop_price)
            
            # Add additional parameters
            for key, value in kwargs.items():
                order_params[key] = value
            
            # Cancel the old order
            self.api.cancel_order(order_id)
            
            # Submit new order
            new_order = self.api.submit_order(**order_params)
            
            # Return order information
            return {
                'order_id': new_order.id,
                'symbol': new_order.symbol,
                'action': new_order.side,
                'order_type': new_order.type,
                'quantity': float(new_order.qty),
                'price': float(new_order.limit_price) if hasattr(new_order, 'limit_price') and new_order.limit_price else None,
                'stop_price': float(new_order.stop_price) if hasattr(new_order, 'stop_price') and new_order.stop_price else None,
                'status': new_order.status,
                'time_in_force': new_order.time_in_force,
                'created_at': new_order.created_at.isoformat(),
                'replaced_order_id': order_id,
                'raw_order': new_order
            }
        
        except Exception as e:
            self.logger.error(f"Error modifying order: {e}")
            return {'error': str(e)}
    
    def get_market_data(self, 
                       symbol: str, 
                       data_type: str = 'QUOTE') -> Dict:
        """
        Get real-time market data from Alpaca Markets.
        
        Args:
            symbol: Symbol to get data for
            data_type: Type of data ("QUOTE", "TRADE", "BOOK")
            
        Returns:
            Dictionary containing market data
        """
        if not self.connected:
            self.logger.error("Not connected to Alpaca Markets")
            return {}
        
        try:
            if data_type == 'QUOTE':
                # Get last quote
                quote = self.api.get_latest_quote(symbol)
                
                return {
                    'symbol': symbol,
                    'bid': float(quote.bp),
                    'ask': float(quote.ap),
                    'bid_size': int(quote.bs),
                    'ask_size': int(quote.as_),
                    'time': quote.t.isoformat()
                }
            
            elif data_type == 'TRADE':
                # Get last trade
                trade = self.api.get_latest_trade(symbol)
                
                return {
                    'symbol': symbol,
                    'price': float(trade.p),
                    'size': int(trade.s),
                    'time': trade.t.isoformat()
                }
            
            elif data_type == 'BOOK':
                # This requires a different API tier - fallback to basic quote
                self.logger.warning("Order book data not available in basic Alpaca API")
                return self.get_market_data(symbol, 'QUOTE')
            
            else:
                self.logger.error(f"Unsupported data type: {data_type}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}
    
    def _run_websocket(self) -> None:
        """
        Run the WebSocket connection in a background thread.
        """
        try:
            # Subscribe to trade updates
            self.ws.subscribe(['trade_updates', 'account_updates'])
            
            # Start WebSocket
            self.ws.run()
            
        except Exception as e:
            self.logger.error(f"Error in WebSocket thread: {e}")
            
        finally:
            self.logger.info("WebSocket thread exiting")
    
    def _handle_trade_updates(self, data: Dict) -> None:
        """
        Handle trade updates from WebSocket.
        
        Args:
            data: Trade update data
        """
        try:
            event = data.get('data', {})
            event_type = event.get('event')
            order = event.get('order', {})
            
            # Process based on event type
            if event_type == 'fill' or event_type == 'partial_fill':
                # Handle execution
                execution_info = {
                    'event_type': event_type,
                    'order_id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'filled_quantity': float(order.get('filled_qty', 0)),
                    'filled_price': float(event.get('price', 0)),
                    'timestamp': event.get('timestamp')
                }
                
                self._notify_callbacks('trade_update', execution_info)
            
            # Handle order status updates
            order_info = {
                'event_type': event_type,
                'order_id': order.get('id'),
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'order_type': order.get('type'),
                'quantity': float(order.get('qty', 0)),
                'filled_quantity': float(order.get('filled_qty', 0)),
                'status': order.get('status'),
                'timestamp': event.get('timestamp')
            }
            
            self._notify_callbacks('order_update', order_info)
            
        except Exception as e:
            self.logger.error(f"Error handling trade update: {e}")
    
    def _handle_account_updates(self, data: Dict) -> None:
        """
        Handle account updates from WebSocket.
        
        Args:
            data: Account update data
        """
        try:
            update = data.get('data', {})
            
            account_info = {
                'id': update.get('id'),
                'cash': float(update.get('cash', 0)),
                'portfolio_value': float(update.get('portfolio_value', 0)),
                'timestamp': update.get('timestamp')
            }
            
            self._notify_callbacks('account_update', account_info)
            
        except Exception as e:
            self.logger.error(f"Error handling account update: {e}")


class BrokerManager:
    """
    Manager for broker connections and automated trading execution.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the broker manager.
        
        Args:
            config: Configuration dictionary for broker manager
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.brokers = {}
        self.default_broker = None
        
        # Initialize configured brokers
        self._init_brokers()
    
    def _init_brokers(self) -> None:
        """
        Initialize configured broker connections.
        """
        broker_configs = self.config.get('brokers', {})
        default_broker_name = self.config.get('default_broker')
        
        for name, broker_config in broker_configs.items():
            broker_type = broker_config.get('type')
            
            if not broker_config.get('enabled', False):
                self.logger.info(f"Broker {name} is disabled, skipping initialization")
                continue
            
            try:
                if broker_type == 'interactive_brokers':
                    broker = InteractiveBrokersBridge(broker_config)
                elif broker_type == 'alpaca':
                    broker = AlpacaBrokerBridge(broker_config)
                else:
                    self.logger.error(f"Unsupported broker type: {broker_type}")
                    continue
                
                self.brokers[name] = broker
                
                # Set as default if configured
                if name == default_broker_name:
                    self.default_broker = broker
                
                self.logger.info(f"Initialized broker: {name}")
                
            except Exception as e:
                self.logger.error(f"Error initializing broker {name}: {e}")
    
    def get_broker(self, name: Optional[str] = None) -> Optional[BrokerBase]:
        """
        Get a broker by name or the default broker.
        
        Args:
            name: Optional broker name
            
        Returns:
            BrokerBase instance or None if not found
        """
        if name:
            return self.brokers.get(name)
        else:
            return self.default_broker
    
    def connect_broker(self, name: Optional[str] = None) -> bool:
        """
        Connect to a broker by name or the default broker.
        
        Args:
            name: Optional broker name
            
        Returns:
            Boolean indicating connection success
        """
        broker = self.get_broker(name)
        
        if not broker:
            self.logger.error(f"Broker {name or 'default'} not found")
            return False
        
        return broker.connect()
    
    def disconnect_broker(self, name: Optional[str] = None) -> bool:
        """
        Disconnect from a broker by name or the default broker.
        
        Args:
            name: Optional broker name
            
        Returns:
            Boolean indicating disconnection success
        """
        broker = self.get_broker(name)
        
        if not broker:
            self.logger.error(f"Broker {name or 'default'} not found")
            return False
        
        return broker.disconnect()
    
    def disconnect_all(self) -> bool:
        """
        Disconnect from all brokers.
        
        Returns:
            Boolean indicating overall success
        """
        success = True
        
        for name, broker in self.brokers.items():
            try:
                if broker.connected:
                    if not broker.disconnect():
                        self.logger.warning(f"Failed to disconnect from broker {name}")
                        success = False
            except Exception as e:
                self.logger.error(f"Error disconnecting from broker {name}: {e}")
                success = False
        
        return success
    
    def execute_trade(self,
                     trade_params: Dict,
                     broker_name: Optional[str] = None) -> Dict:
        """
        Execute a trade with the specified broker.
        
        Args:
            trade_params: Trade parameters dictionary
            broker_name: Optional broker name
            
        Returns:
            Dictionary containing trade result
        """
        broker = self.get_broker(broker_name)
        
        if not broker:
            self.logger.error(f"Broker {broker_name or 'default'} not found")
            return {'success': False, 'error': f"Broker {broker_name or 'default'} not found"}
        
        if not broker.connected:
            self.logger.error(f"Broker {broker_name or 'default'} not connected")
            return {'success': False, 'error': f"Broker {broker_name or 'default'} not connected"}
        
        try:
            # Extract trade parameters
            symbol = trade_params.get('symbol')
            side = trade_params.get('side')
            order_type = trade_params.get('order_type', 'MARKET')
            quantity = trade_params.get('quantity')
            price = trade_params.get('price')
            stop_price = trade_params.get('stop_price')
            time_in_force = trade_params.get('time_in_force', 'DAY')
            
            # Validate required parameters
            if not symbol or not side or not quantity:
                return {'success': False, 'error': "Missing required trade parameters"}
            
            # Place order
            order_result = broker.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            # Check for error
            if 'error' in order_result:
                return {'success': False, 'error': order_result['error']}
            
            # Return success
            return {
                'success': True,
                'order': order_result
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_signal(self,
                      scan_result: Dict,
                      risk_pct: float = 1.0,
                      broker_name: Optional[str] = None) -> Dict:
        """
        Execute a trade based on a signal from scan result.
        
        Args:
            scan_result: Scan result dictionary
            risk_pct: Risk percentage for position sizing
            broker_name: Optional broker name
            
        Returns:
            Dictionary containing trade result
        """
        broker = self.get_broker(broker_name)
        
        if not broker:
            self.logger.error(f"Broker {broker_name or 'default'} not found")
            return {'success': False, 'error': f"Broker {broker_name or 'default'} not found"}
        
        if not broker.connected:
            self.logger.error(f"Broker {broker_name or 'default'} not connected")
            return {'success': False, 'error': f"Broker {broker_name or 'default'} not connected"}
        
        try:
            # Extract signal information
            symbol = scan_result.get('symbol')
            signal = scan_result.get('signal', {})
            position_guidance = scan_result.get('position_guidance', {})
            
            signal_type = signal.get('signal', 'neutral')
            
            # Skip neutral signals
            if 'neutral' in signal_type:
                return {'success': False, 'error': "Neutral signal, no trade executed"}
            
            # Determine trade direction
            side = 'BUY' if 'buy' in signal_type else 'SELL'
            
            # Get account information for position sizing
            account_info = broker.get_account_info()
            buying_power = float(account_info.get('buying_power', 0))
            
            if buying_power <= 0:
                return {'success': False, 'error': "Insufficient buying power"}
            
            # Get entry and stop prices
            entry_price = position_guidance.get('entry_price', 0)
            stop_loss = position_guidance.get('stop_loss', 0)
            
            if entry_price <= 0 or stop_loss <= 0:
                return {'success': False, 'error': "Invalid entry or stop price"}
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                return {'success': False, 'error': "Zero risk per share"}
            
            # Calculate position size based on risk percentage
            # Risk amount = account_value * risk_percentage / 100
            account_value = float(account_info.get('portfolio_value', buying_power))
            risk_amount = account_value * risk_pct / 100
            
            # Shares to trade = risk_amount / risk_per_share
            shares = int(risk_amount / risk_per_share)
            
            # Ensure minimum shares
            shares = max(shares, 1)
            
            # Cap to maximum position size
            max_position_pct = self.config.get('max_position_pct', 5.0)
            max_shares = int((account_value * max_position_pct / 100) / entry_price)
            shares = min(shares, max_shares)
            
            # Create trade parameters
            trade_params = {
                'symbol': symbol,
                'side': side,
                'order_type': 'LIMIT',
                'quantity': shares,
                'price': entry_price,
                'time_in_force': 'DAY'
            }
            
            # Execute the entry order
            entry_result = self.execute_trade(trade_params, broker_name)
            
            if not entry_result.get('success', False):
                return entry_result
            
            # Place stop loss order
            stop_params = {
                'symbol': symbol,
                'side': 'SELL' if side == 'BUY' else 'BUY',
                'order_type': 'STOP',
                'quantity': shares,
                'stop_price': stop_loss,
                'time_in_force': 'GTC'
            }
            
            stop_result = self.execute_trade(stop_params, broker_name)
            
            # Return combined result
            return {
                'success': True,
                'entry_order': entry_result.get('order', {}),
                'stop_order': stop_result.get('order', {}) if stop_result.get('success', False) else None,
                'signal': signal,
                'position_guidance': position_guidance,
                'shares': shares,
                'risk_amount': risk_amount,
                'risk_per_share': risk_per_share
            }
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return {'success': False, 'error': str(e)}
