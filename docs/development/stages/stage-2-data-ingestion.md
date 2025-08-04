# 📊 Stage 2: Data Ingestion Module

## 📋 CHECKLIST: Data Ingestion (12-16 horas)

### ✅ Prerrequisitos
- [ ] Shared libraries completadas y testeadas
- [ ] Docker infrastructure funcionando
- [ ] APIs keys configuradas (Binance, Twitter)
- [ ] Message brokers operativos

### ✅ Objetivos de la Etapa
Implementar la ingesta de datos en tiempo real desde:
- **Binance WebSocket**: Tick data y order book de BTCUSD
- **Twitter API**: Posts relacionados con Bitcoin para sentiment
- **Data Storage**: Almacenamiento en MinIO y TimescaleDB
- **Real-time Streaming**: Distribución via Redis Streams

## 🏗️ Arquitectura del Módulo

```
modules/data-ingestion/
├── __init__.py
├── main.py                 # Entry point del servicio
├── binance/
│   ├── __init__.py
│   ├── websocket_client.py # Cliente WebSocket Binance
│   ├── rest_client.py      # Cliente REST Binance
│   └── data_processor.py   # Procesamiento de datos
├── twitter/
│   ├── __init__.py
│   ├── stream_client.py    # Cliente Twitter Streaming
│   └── sentiment_processor.py # Procesamiento básico
├── storage/
│   ├── __init__.py
│   ├── minio_client.py     # Cliente MinIO
│   └── timescale_writer.py # Escritor TimescaleDB
└── config/
    └── settings.py         # Configuración del módulo
```

## 🚀 Implementación

### Binance WebSocket Client
```python
# modules/data-ingestion/binance/websocket_client.py
import asyncio
import json
import websockets
from typing import Callable, Dict, Any
from shared.messaging.message_broker import HybridMessageBroker
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class BinanceWebSocketClient:
    def __init__(self, message_broker: HybridMessageBroker):
        self.message_broker = message_broker
        self.websocket = None
        self.running = False
        
    async def start(self):
        """Start WebSocket connection"""
        self.running = True
        await self._connect_and_stream()
        
    async def _connect_and_stream(self):
        """Connect to Binance WebSocket and stream data"""
        uri = "wss://stream.binance.com:9443/ws/btcusdt@ticker/btcusdt@depth20@100ms"
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websocket = websocket
                    logger.info("Connected to Binance WebSocket")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        data = json.loads(message)
                        await self._process_message(data)
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Reconnect delay
                    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        try:
            if 'e' in data:  # Event type
                if data['e'] == '24hrTicker':
                    await self._handle_ticker(data)
                elif data['e'] == 'depthUpdate':
                    await self._handle_depth(data)
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    async def _handle_ticker(self, data: Dict[str, Any]):
        """Handle ticker data"""
        ticker_data = {
            'symbol': data['s'],
            'price': float(data['c']),
            'volume': float(data['v']),
            'timestamp': data['E'],
            'change_24h': float(data['P']),
            'high_24h': float(data['h']),
            'low_24h': float(data['l'])
        }
        
        # Send to Redis Streams for real-time processing
        await self.message_broker.publish_fast('market_data', ticker_data)
        
        # Send to Kafka for historical storage
        await self.message_broker.publish_reliable('market_data', ticker_data)
        
    async def _handle_depth(self, data: Dict[str, Any]):
        """Handle order book depth data"""
        depth_data = {
            'symbol': data['s'],
            'timestamp': data['E'],
            'bids': [[float(bid[0]), float(bid[1])] for bid in data['b'][:10]],
            'asks': [[float(ask[0]), float(ask[1])] for ask in data['a'][:10]]
        }
        
        # Send to Redis Streams for real-time processing
        await self.message_broker.publish_fast('order_book', depth_data)
        
    async def stop(self):
        """Stop WebSocket client"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
        logger.info("Binance WebSocket client stopped")
```

### Twitter Stream Client
```python
# modules/data-ingestion/twitter/stream_client.py
import asyncio
import tweepy
from typing import Dict, Any
from shared.messaging.message_broker import HybridMessageBroker
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class TwitterStreamClient:
    def __init__(self, bearer_token: str, message_broker: HybridMessageBroker):
        self.bearer_token = bearer_token
        self.message_broker = message_broker
        self.stream = None
        
    async def start(self):
        """Start Twitter streaming"""
        try:
            # Create stream listener
            listener = TwitterStreamListener(self.message_broker)
            
            # Initialize streaming client
            self.stream = tweepy.StreamingClient(
                bearer_token=self.bearer_token,
                wait_on_rate_limit=True
            )
            
            # Add rules for Bitcoin-related tweets
            rules = [
                tweepy.StreamRule("bitcoin OR btc OR $BTC lang:en -is:retweet"),
                tweepy.StreamRule("cryptocurrency OR crypto lang:en -is:retweet")
            ]
            
            # Delete existing rules and add new ones
            existing_rules = self.stream.get_rules()
            if existing_rules.data:
                rule_ids = [rule.id for rule in existing_rules.data]
                self.stream.delete_rules(rule_ids)
                
            self.stream.add_rules(rules)
            
            # Start streaming
            self.stream.filter(
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                threaded=True
            )
            
            logger.info("Twitter streaming started")
            
        except Exception as e:
            logger.error(f"Failed to start Twitter streaming: {e}")
            raise
            
    async def stop(self):
        """Stop Twitter streaming"""
        if self.stream:
            self.stream.disconnect()
        logger.info("Twitter streaming stopped")

class TwitterStreamListener(tweepy.StreamingClient):
    def __init__(self, message_broker: HybridMessageBroker):
        self.message_broker = message_broker
        
    def on_tweet(self, tweet):
        """Handle incoming tweet"""
        try:
            tweet_data = {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at.isoformat(),
                'author_id': tweet.author_id,
                'public_metrics': tweet.public_metrics,
                'timestamp': int(tweet.created_at.timestamp() * 1000)
            }
            
            # Send to message broker
            asyncio.create_task(
                self.message_broker.publish_fast('social_data', tweet_data)
            )
            
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            
    def on_error(self, status_code):
        logger.error(f"Twitter streaming error: {status_code}")
        return True  # Continue streaming
```

### Data Storage Components
```python
# modules/data-ingestion/storage/timescale_writer.py
import asyncio
from typing import Dict, Any, List
from shared.database.connection import DatabaseManager
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class TimescaleWriter:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.batch_size = 1000
        self.batch_timeout = 5  # seconds
        self.batches = {
            'market_data': [],
            'order_book': [],
            'social_data': []
        }
        
    async def start(self):
        """Start batch writer"""
        asyncio.create_task(self._batch_writer())
        logger.info("TimescaleDB writer started")
        
    async def write_market_data(self, data: Dict[str, Any]):
        """Add market data to batch"""
        self.batches['market_data'].append(data)
        
        if len(self.batches['market_data']) >= self.batch_size:
            await self._flush_market_data()
            
    async def write_order_book(self, data: Dict[str, Any]):
        """Add order book data to batch"""
        self.batches['order_book'].append(data)
        
        if len(self.batches['order_book']) >= self.batch_size:
            await self._flush_order_book()
            
    async def _batch_writer(self):
        """Periodic batch flusher"""
        while True:
            await asyncio.sleep(self.batch_timeout)
            
            if self.batches['market_data']:
                await self._flush_market_data()
                
            if self.batches['order_book']:
                await self._flush_order_book()
                
    async def _flush_market_data(self):
        """Flush market data batch to database"""
        if not self.batches['market_data']:
            return
            
        try:
            batch = self.batches['market_data'].copy()
            self.batches['market_data'].clear()
            
            # Prepare bulk insert
            values = [
                (
                    item['timestamp'],
                    item['symbol'],
                    item['price'],
                    item['volume'],
                    item.get('change_24h'),
                    item.get('high_24h'),
                    item.get('low_24h')
                )
                for item in batch
            ]
            
            query = """
                INSERT INTO trading.market_data 
                (timestamp, symbol, price, volume, change_24h, high_24h, low_24h)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            async with self.db_manager.get_connection() as conn:
                await conn.executemany(query, values)
                
            logger.debug(f"Flushed {len(batch)} market data records")
            
        except Exception as e:
            logger.error(f"Error flushing market data: {e}")
```

### Service Main Entry Point
```python
# modules/data-ingestion/main.py
import asyncio
import signal
from shared.messaging.message_broker import HybridMessageBroker
from shared.messaging.kafka_client import AsyncKafkaClient
from shared.messaging.redis_client import AsyncRedisStreamsClient
from shared.database.connection import DatabaseManager
from shared.config.config_manager import ConfigManager
from shared.logging.structured_logger import get_logger
from binance.websocket_client import BinanceWebSocketClient
from twitter.stream_client import TwitterStreamClient
from storage.timescale_writer import TimescaleWriter

logger = get_logger(__name__)

class DataIngestionService:
    def __init__(self):
        self.config_manager = None
        self.message_broker = None
        self.db_manager = None
        self.binance_client = None
        self.twitter_client = None
        self.timescale_writer = None
        self.running = False
        
    async def start(self):
        """Start data ingestion service"""
        try:
            # Initialize configuration
            self.config_manager = ConfigManager()
            await self.config_manager.start()
            config = await self.config_manager.get_config('data-ingestion')
            
            # Initialize database
            self.db_manager = DatabaseManager(config['database']['dsn'])
            await self.db_manager.start()
            
            # Initialize message broker
            kafka_client = AsyncKafkaClient(config['kafka']['bootstrap_servers'])
            redis_client = AsyncRedisStreamsClient(
                config['redis']['host'],
                config['redis']['port'],
                config['redis']['password']
            )
            
            self.message_broker = HybridMessageBroker(kafka_client, redis_client)
            await self.message_broker.start()
            
            # Initialize storage writer
            self.timescale_writer = TimescaleWriter(self.db_manager)
            await self.timescale_writer.start()
            
            # Initialize data clients
            self.binance_client = BinanceWebSocketClient(self.message_broker)
            self.twitter_client = TwitterStreamClient(
                config['twitter']['bearer_token'],
                self.message_broker
            )
            
            # Start data ingestion
            await asyncio.gather(
                self.binance_client.start(),
                self.twitter_client.start()
            )
            
            self.running = True
            logger.info("Data ingestion service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start data ingestion service: {e}")
            raise
            
    async def stop(self):
        """Stop data ingestion service"""
        self.running = False
        
        # Stop all components
        if self.binance_client:
            await self.binance_client.stop()
            
        if self.twitter_client:
            await self.twitter_client.stop()
            
        if self.message_broker:
            await self.message_broker.stop()
            
        if self.db_manager:
            await self.db_manager.stop()
            
        logger.info("Data ingestion service stopped")

async def main():
    service = DataIngestionService()
    
    # Setup signal handlers
    def signal_handler():
        asyncio.create_task(service.stop())
        
    signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
    
    try:
        await service.start()
        
        # Keep service running
        while service.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## ✅ Testing y Validación

### Unit Tests
```python
# tests/unit/test_binance_client.py
import pytest
from unittest.mock import Mock, AsyncMock
from modules.data_ingestion.binance.websocket_client import BinanceWebSocketClient

@pytest.mark.asyncio
async def test_binance_websocket_client():
    # Mock message broker
    message_broker = Mock()
    message_broker.publish_fast = AsyncMock()
    message_broker.publish_reliable = AsyncMock()
    
    client = BinanceWebSocketClient(message_broker)
    
    # Test message processing
    ticker_data = {
        'e': '24hrTicker',
        's': 'BTCUSDT',
        'c': '50000.00',
        'v': '1000.00',
        'E': 1234567890,
        'P': '2.5',
        'h': '51000.00',
        'l': '49000.00'
    }
    
    await client._process_message(ticker_data)
    
    # Verify message was published
    message_broker.publish_fast.assert_called_once()
    message_broker.publish_reliable.assert_called_once()
```

## ✅ Checklist de Completitud

### Binance Integration
- [ ] WebSocket client implementado
- [ ] Ticker data processing
- [ ] Order book processing
- [ ] Error handling y reconnection
- [ ] Rate limiting respetado
- [ ] Unit tests pasando

### Twitter Integration
- [ ] Streaming client implementado
- [ ] Tweet filtering configurado
- [ ] Basic sentiment processing
- [ ] Rate limiting manejado
- [ ] Error handling robusto
- [ ] Unit tests pasando

### Data Storage
- [ ] TimescaleDB writer funcionando
- [ ] MinIO storage implementado
- [ ] Batch processing optimizado
- [ ] Data validation implementada
- [ ] Performance optimizada
- [ ] Integration tests pasando

### Service Integration
- [ ] Main service funcionando
- [ ] Configuration management
- [ ] Health checks implementados
- [ ] Graceful shutdown
- [ ] Docker container funcionando
- [ ] End-to-end tests pasando

**Tiempo estimado**: 12-16 horas  
**Responsable**: Backend Developer + Data Scientist  
**Dependencias**: Shared libraries, Docker infrastructure

---

**Next Step**: [Stage 3: Feature Engineering](./stage-3-feature-engineering.md)
