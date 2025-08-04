# ⚡ Stage 6: Trading Execution Module

## 📋 CHECKLIST: Trading Execution (16-20 horas)

### ✅ Prerrequisitos
- [ ] Risk Management funcionando y aprobando trades
- [ ] Risk decisions fluyendo en tiempo real
- [ ] Binance API keys configuradas y validadas
- [ ] Database con esquemas de trades y posiciones

### ✅ Objetivos de la Etapa
Implementar la ejecución de trades en tiempo real:
- **Order Management**: Gestión completa de órdenes
- **Binance Integration**: Integración con Binance REST API
- **Position Tracking**: Seguimiento de posiciones abiertas
- **Trade Execution**: Ejecución de trades con validación
- **Order Types**: Soporte para market, limit, y OCO orders

## 🏗️ Arquitectura del Módulo

```
modules/trading-execution/
├── __init__.py
├── main.py                    # Entry point del servicio
├── brokers/
│   ├── __init__.py
│   ├── binance_client.py     # Cliente Binance REST API
│   ├── order_manager.py      # Gestión de órdenes
│   └── position_tracker.py   # Tracking de posiciones
├── execution/
│   ├── __init__.py
│   ├── trade_executor.py     # Ejecutor principal
│   ├── order_validator.py    # Validación de órdenes
│   └── execution_monitor.py  # Monitor de ejecución
├── orders/
│   ├── __init__.py
│   ├── market_order.py       # Órdenes de mercado
│   ├── limit_order.py        # Órdenes limitadas
│   └── oco_order.py          # Órdenes OCO
├── portfolio/
│   ├── __init__.py
│   ├── portfolio_manager.py  # Gestión de portfolio
│   ├── pnl_calculator.py     # Cálculo de PnL
│   └── balance_tracker.py    # Tracking de balance
└── config/
    └── execution_config.py   # Configuración de ejecución
```

## 🚀 Implementación Detallada

### Binance Client Integration
```python
# modules/trading-execution/brokers/binance_client.py
import asyncio
import hmac
import hashlib
import time
from typing import Dict, List, Any, Optional
import aiohttp
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class BinanceRestClient:
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api"
            
        self.session = None
        
    async def start(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'X-MBX-APIKEY': self.api_key}
        )
        
        # Test connectivity
        await self.ping()
        logger.info("Binance REST client initialized")
        
    async def ping(self) -> bool:
        """Test connectivity to Binance"""
        try:
            async with self.session.get(f"{self.base_url}/v3/ping") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Binance ping failed: {e}")
            return False
            
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            params = {'timestamp': int(time.time() * 1000)}
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            async with self.session.get(
                f"{self.base_url}/v3/account",
                params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get account info: {error_text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
            
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """Place market order"""
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': 'MARKET',
                'quantity': f"{quantity:.6f}",
                'timestamp': int(time.time() * 1000)
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            async with self.session.post(
                f"{self.base_url}/v3/order",
                data=params
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    logger.info(f"Market order placed: {symbol} {side} {quantity}")
                    return result
                else:
                    logger.error(f"Failed to place market order: {result}")
                    return {'error': result}
                    
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return {'error': str(e)}
            
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """Place limit order"""
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': f"{quantity:.6f}",
                'price': f"{price:.2f}",
                'timestamp': int(time.time() * 1000)
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            async with self.session.post(
                f"{self.base_url}/v3/order",
                data=params
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    logger.info(f"Limit order placed: {symbol} {side} {quantity} @ {price}")
                    return result
                else:
                    logger.error(f"Failed to place limit order: {result}")
                    return {'error': result}
                    
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return {'error': str(e)}
            
    async def place_oco_order(self, symbol: str, side: str, quantity: float, 
                             price: float, stop_price: float, stop_limit_price: float) -> Dict[str, Any]:
        """Place OCO (One-Cancels-Other) order"""
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'quantity': f"{quantity:.6f}",
                'price': f"{price:.2f}",
                'stopPrice': f"{stop_price:.2f}",
                'stopLimitPrice': f"{stop_limit_price:.2f}",
                'stopLimitTimeInForce': 'GTC',
                'timestamp': int(time.time() * 1000)
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            async with self.session.post(
                f"{self.base_url}/v3/order/oco",
                data=params
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    logger.info(f"OCO order placed: {symbol} {side} {quantity}")
                    return result
                else:
                    logger.error(f"Failed to place OCO order: {result}")
                    return {'error': result}
                    
        except Exception as e:
            logger.error(f"Error placing OCO order: {e}")
            return {'error': str(e)}
            
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel order"""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id,
                'timestamp': int(time.time() * 1000)
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            async with self.session.delete(
                f"{self.base_url}/v3/order",
                params=params
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    logger.info(f"Order cancelled: {symbol} {order_id}")
                    return result
                else:
                    logger.error(f"Failed to cancel order: {result}")
                    return {'error': result}
                    
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'error': str(e)}
            
    async def get_order_status(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Get order status"""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id,
                'timestamp': int(time.time() * 1000)
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            async with self.session.get(
                f"{self.base_url}/v3/order",
                params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get order status: {error_text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {}
            
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        try:
            params = {'timestamp': int(time.time() * 1000)}
            if symbol:
                params['symbol'] = symbol
                
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            async with self.session.get(
                f"{self.base_url}/v3/openOrders",
                params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get open orders: {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
            
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate signature for authenticated requests"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    async def stop(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
        logger.info("Binance REST client stopped")
```

### Trade Executor
```python
# modules/trading-execution/execution/trade_executor.py
import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from shared.messaging.message_broker import HybridMessageBroker
from shared.database.connection import DatabaseManager
from shared.logging.structured_logger import get_logger
from brokers.binance_client import BinanceRestClient
from execution.order_validator import OrderValidator
from portfolio.portfolio_manager import PortfolioManager

logger = get_logger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

@dataclass
class TradeOrder:
    id: str
    timestamp: int
    symbol: str
    side: str  # BUY/SELL
    order_type: str  # MARKET/LIMIT/OCO
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    error_message: Optional[str] = None

class TradeExecutor:
    def __init__(self, message_broker: HybridMessageBroker, 
                 db_manager: DatabaseManager, 
                 binance_client: BinanceRestClient):
        self.message_broker = message_broker
        self.db_manager = db_manager
        self.binance_client = binance_client
        
        self.order_validator = OrderValidator()
        self.portfolio_manager = PortfolioManager(db_manager)
        
        # Order tracking
        self.active_orders = {}
        self.execution_metrics = {
            'orders_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_volume': 0.0,
            'avg_execution_time': 0.0
        }
        
    async def start(self):
        """Start trade execution service"""
        try:
            # Subscribe to risk decisions
            await self.message_broker.subscribe_fast(
                ['risk_decisions'],
                'trading-execution',
                'executor-1',
                self._process_risk_decision
            )
            
            # Start order monitoring loop
            asyncio.create_task(self._order_monitoring_loop())
            
            logger.info("Trade executor started")
            
        except Exception as e:
            logger.error(f"Failed to start trade executor: {e}")
            raise
            
    async def _process_risk_decision(self, stream: str, message_id: str, decision_data: Dict[str, Any]):
        """Process risk management decision"""
        try:
            decision = decision_data['decision']
            
            if decision == 'approve':
                await self._execute_approved_trade(decision_data)
            elif decision == 'modify':
                await self._execute_modified_trade(decision_data)
            else:
                logger.info(f"Trade rejected by risk management: {decision_data.get('reasons', [])}")
                
        except Exception as e:
            logger.error(f"Error processing risk decision: {e}")
            
    async def _execute_approved_trade(self, decision_data: Dict[str, Any]):
        """Execute approved trade"""
        try:
            # Create trade order
            order = TradeOrder(
                id=f"order_{int(time.time() * 1000)}",
                timestamp=int(time.time() * 1000),
                symbol=decision_data['symbol'],
                side='BUY' if decision_data['original_signal'] == 2 else 'SELL',
                order_type='MARKET',  # Start with market orders
                quantity=decision_data['approved_size']
            )
            
            # Validate order
            validation_result = await self.order_validator.validate_order(order)
            if not validation_result['valid']:
                logger.error(f"Order validation failed: {validation_result['reasons']}")
                return
                
            # Execute order
            execution_result = await self._execute_order(order)
            
            # Update portfolio
            if execution_result['success']:
                await self.portfolio_manager.update_position(order, execution_result)
                
            # Publish execution result
            await self._publish_execution_result(order, execution_result)
            
        except Exception as e:
            logger.error(f"Error executing approved trade: {e}")
            
    async def _execute_order(self, order: TradeOrder) -> Dict[str, Any]:
        """Execute order with broker"""
        start_time = time.time()
        
        try:
            order.status = OrderStatus.SUBMITTED
            self.active_orders[order.id] = order
            
            # Store order in database
            await self._store_order(order)
            
            # Execute based on order type
            if order.order_type == 'MARKET':
                result = await self.binance_client.place_market_order(
                    order.symbol, order.side, order.quantity
                )
            elif order.order_type == 'LIMIT':
                result = await self.binance_client.place_limit_order(
                    order.symbol, order.side, order.quantity, order.price
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
                
            # Process result
            if 'error' in result:
                order.status = OrderStatus.FAILED
                order.error_message = str(result['error'])
                execution_time = time.time() - start_time
                
                self.execution_metrics['failed_executions'] += 1
                
                return {
                    'success': False,
                    'error': result['error'],
                    'execution_time': execution_time
                }
            else:
                # Order submitted successfully
                order.broker_order_id = str(result.get('orderId'))
                order.status = OrderStatus.FILLED if result.get('status') == 'FILLED' else OrderStatus.SUBMITTED
                
                if order.status == OrderStatus.FILLED:
                    order.filled_quantity = float(result.get('executedQty', 0))
                    order.avg_fill_price = float(result.get('price', 0))
                    order.commission = float(result.get('commission', 0))
                    
                execution_time = time.time() - start_time
                
                # Update metrics
                self.execution_metrics['orders_executed'] += 1
                self.execution_metrics['successful_executions'] += 1
                self.execution_metrics['total_volume'] += order.filled_quantity
                
                # Update average execution time
                current_avg = self.execution_metrics['avg_execution_time']
                total_orders = self.execution_metrics['orders_executed']
                self.execution_metrics['avg_execution_time'] = (
                    (current_avg * (total_orders - 1) + execution_time) / total_orders
                )
                
                logger.info(f"Order executed successfully: {order.id} in {execution_time:.3f}s")
                
                return {
                    'success': True,
                    'broker_order_id': order.broker_order_id,
                    'filled_quantity': order.filled_quantity,
                    'avg_fill_price': order.avg_fill_price,
                    'execution_time': execution_time,
                    'commission': order.commission
                }
                
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            execution_time = time.time() - start_time
            
            self.execution_metrics['failed_executions'] += 1
            
            logger.error(f"Order execution failed: {order.id} - {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
        finally:
            # Update order in database
            await self._update_order(order)
            
    async def _order_monitoring_loop(self):
        """Monitor active orders"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check status of submitted orders
                orders_to_check = [
                    order for order in self.active_orders.values()
                    if order.status == OrderStatus.SUBMITTED and order.broker_order_id
                ]
                
                for order in orders_to_check:
                    await self._check_order_status(order)
                    
            except Exception as e:
                logger.error(f"Error in order monitoring loop: {e}")
                
    async def _check_order_status(self, order: TradeOrder):
        """Check order status with broker"""
        try:
            if not order.broker_order_id:
                return
                
            status_result = await self.binance_client.get_order_status(
                order.symbol, int(order.broker_order_id)
            )
            
            if status_result:
                broker_status = status_result.get('status')
                
                if broker_status == 'FILLED':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = float(status_result.get('executedQty', 0))
                    order.avg_fill_price = float(status_result.get('price', 0))
                    
                    # Update portfolio
                    await self.portfolio_manager.update_position(order, {
                        'success': True,
                        'filled_quantity': order.filled_quantity,
                        'avg_fill_price': order.avg_fill_price
                    })
                    
                    # Remove from active orders
                    if order.id in self.active_orders:
                        del self.active_orders[order.id]
                        
                    logger.info(f"Order filled: {order.id} - {order.filled_quantity} @ {order.avg_fill_price}")
                    
                elif broker_status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    order.status = OrderStatus.CANCELLED
                    
                    if order.id in self.active_orders:
                        del self.active_orders[order.id]
                        
                    logger.warning(f"Order cancelled/rejected: {order.id} - {broker_status}")
                    
                # Update order in database
                await self._update_order(order)
                
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            
    async def _store_order(self, order: TradeOrder):
        """Store order in database"""
        try:
            query = """
                INSERT INTO trading.orders (
                    id, timestamp, symbol, side, order_type, quantity, price,
                    stop_price, status, broker_order_id, filled_quantity,
                    avg_fill_price, commission, error_message
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """
            
            await self.db_manager.execute(
                query,
                order.id, order.timestamp, order.symbol, order.side,
                order.order_type, order.quantity, order.price,
                order.stop_price, order.status.value, order.broker_order_id,
                order.filled_quantity, order.avg_fill_price, order.commission,
                order.error_message
            )
            
        except Exception as e:
            logger.error(f"Error storing order: {e}")
            
    async def _update_order(self, order: TradeOrder):
        """Update order in database"""
        try:
            query = """
                UPDATE trading.orders SET
                    status = $1, broker_order_id = $2, filled_quantity = $3,
                    avg_fill_price = $4, commission = $5, error_message = $6
                WHERE id = $7
            """
            
            await self.db_manager.execute(
                query,
                order.status.value, order.broker_order_id, order.filled_quantity,
                order.avg_fill_price, order.commission, order.error_message,
                order.id
            )
            
        except Exception as e:
            logger.error(f"Error updating order: {e}")
            
    async def _publish_execution_result(self, order: TradeOrder, result: Dict[str, Any]):
        """Publish execution result"""
        try:
            execution_data = {
                'timestamp': int(time.time() * 1000),
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'status': order.status.value,
                'success': result['success'],
                'filled_quantity': order.filled_quantity,
                'avg_fill_price': order.avg_fill_price,
                'execution_time': result.get('execution_time', 0),
                'commission': order.commission,
                'error': result.get('error')
            }
            
            # Publish to monitoring
            await self.message_broker.publish_fast('trade_executions', execution_data)
            
            # Store in Kafka for audit
            await self.message_broker.publish_reliable('trade_executions', execution_data)
            
        except Exception as e:
            logger.error(f"Error publishing execution result: {e}")
            
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        return {
            **self.execution_metrics,
            'active_orders': len(self.active_orders),
            'success_rate': (
                self.execution_metrics['successful_executions'] / 
                max(self.execution_metrics['orders_executed'], 1)
            ) * 100
        }
```

## ✅ Testing y Validación

### Execution Tests
```python
# tests/unit/test_trading_execution.py
import pytest
from unittest.mock import Mock, AsyncMock
from modules.trading_execution.execution.trade_executor import TradeExecutor, TradeOrder, OrderStatus

@pytest.mark.asyncio
async def test_trade_execution():
    """Test trade execution flow"""
    # Mock dependencies
    message_broker = Mock()
    db_manager = Mock()
    db_manager.execute = AsyncMock()
    
    binance_client = Mock()
    binance_client.place_market_order = AsyncMock(return_value={
        'orderId': '12345',
        'status': 'FILLED',
        'executedQty': '0.001',
        'price': '50000.0'
    })
    
    executor = TradeExecutor(message_broker, db_manager, binance_client)
    
    # Test order
    order = TradeOrder(
        id='test_order_1',
        timestamp=1234567890,
        symbol='BTCUSDT',
        side='BUY',
        order_type='MARKET',
        quantity=0.001
    )
    
    result = await executor._execute_order(order)
    
    assert result['success'] is True
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 0.001
```

## ✅ Checklist de Completitud

### Broker Integration
- [ ] Binance REST API client
- [ ] Order placement (market, limit, OCO)
- [ ] Order cancellation
- [ ] Order status monitoring
- [ ] Account information retrieval
- [ ] Error handling y retry logic

### Order Management
- [ ] Order validation
- [ ] Order tracking
- [ ] Status monitoring
- [ ] Execution reporting
- [ ] Commission tracking
- [ ] Partial fill handling

### Portfolio Management
- [ ] Position tracking
- [ ] PnL calculation
- [ ] Balance monitoring
- [ ] Risk exposure tracking
- [ ] Performance metrics
- [ ] Portfolio reporting

### Execution Monitoring
- [ ] Real-time execution metrics
- [ ] Latency monitoring
- [ ] Success rate tracking
- [ ] Error analysis
- [ ] Performance benchmarking
- [ ] Alert system

**Tiempo estimado**: 16-20 horas  
**Responsable**: Trading Systems Developer  
**Dependencias**: Risk Management funcionando

---

**Next Step**: [Stage 7: Monitoring & Dashboard](./stage-7-monitoring-dashboard.md)
