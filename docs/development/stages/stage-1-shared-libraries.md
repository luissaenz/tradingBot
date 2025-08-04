# 📚 Stage 1: Shared Libraries Development

## 📋 CHECKLIST: Shared Libraries (8-12 horas)

### ✅ Prerrequisitos
- [ ] Docker infrastructure funcionando
- [ ] Todos los servicios UP y healthy
- [ ] Environment variables configuradas
- [ ] Python virtual environment activado

### ✅ Objetivos de la Etapa
Crear las librerías compartidas que serán utilizadas por todos los módulos del sistema:
- **Messaging**: Abstracción para Kafka y Redis Streams
- **Database**: Conexiones y operaciones con PostgreSQL/TimescaleDB
- **Configuration**: Gestión centralizada de configuración con Consul
- **Logging**: Sistema de logging estructurado
- **Metrics**: Recolección de métricas para Prometheus

## 🏗️ Arquitectura de Shared Libraries

```
shared/
├── messaging/
│   ├── __init__.py
│   ├── kafka_client.py      # Cliente Kafka
│   ├── redis_client.py      # Cliente Redis Streams
│   ├── message_broker.py    # Abstracción unificada
│   └── schemas.py           # Esquemas de mensajes
├── database/
│   ├── __init__.py
│   ├── connection.py        # Pool de conexiones
│   ├── models.py           # Modelos SQLAlchemy
│   ├── repositories.py     # Repositorios base
│   └── migrations/         # Migraciones Alembic
├── config/
│   ├── __init__.py
│   ├── consul_client.py    # Cliente Consul
│   ├── config_manager.py   # Gestor de configuración
│   └── schemas.py          # Esquemas Pydantic
├── logging/
│   ├── __init__.py
│   ├── structured_logger.py # Logger estructurado
│   ├── formatters.py       # Formateadores custom
│   └── handlers.py         # Handlers custom
├── metrics/
│   ├── __init__.py
│   ├── prometheus_client.py # Cliente Prometheus
│   ├── collectors.py       # Collectors custom
│   └── decorators.py       # Decoradores para métricas
└── utils/
    ├── __init__.py
    ├── datetime_utils.py   # Utilidades de fecha/hora
    ├── validation.py       # Validaciones comunes
    └── exceptions.py       # Excepciones custom
```

## 🚀 Implementación Detallada

### Paso 1: Messaging Library (3-4 horas)

#### Kafka Client
```python
# shared/messaging/kafka_client.py
import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import logging

logger = logging.getLogger(__name__)

class AsyncKafkaClient:
    def __init__(self, bootstrap_servers: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumers: Dict[str, KafkaConsumer] = {}
        
    async def start(self):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
            
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        """Publish message to Kafka topic"""
        if not self.producer:
            raise RuntimeError("Kafka producer not initialized")
            
        try:
            future = self.producer.send(topic, value=message, key=key)
            record_metadata = future.get(timeout=10)
            logger.debug(f"Message sent to {topic}: partition {record_metadata.partition}, offset {record_metadata.offset}")
            return record_metadata
        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            raise
            
    async def subscribe(self, topics: List[str], group_id: str, callback: Callable):
        """Subscribe to Kafka topics"""
        try:
            consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            self.consumers[group_id] = consumer
            
            # Start consuming in background task
            asyncio.create_task(self._consume_messages(consumer, callback))
            logger.info(f"Subscribed to topics {topics} with group {group_id}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topics {topics}: {e}")
            raise
            
    async def _consume_messages(self, consumer: KafkaConsumer, callback: Callable):
        """Consume messages from Kafka"""
        try:
            for message in consumer:
                try:
                    await callback(message.topic, message.key, message.value)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except Exception as e:
            logger.error(f"Error in message consumption: {e}")
            
    async def stop(self):
        """Stop Kafka client"""
        if self.producer:
            self.producer.close()
            
        for consumer in self.consumers.values():
            consumer.close()
            
        logger.info("Kafka client stopped")
```

#### Redis Streams Client
```python
# shared/messaging/redis_client.py
import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

class AsyncRedisStreamsClient:
    def __init__(self, host: str, port: int, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        self.client = None
        self.consumers: Dict[str, asyncio.Task] = {}
        
    async def start(self):
        """Initialize Redis connection"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            logger.info("Redis Streams client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
            
    async def publish(self, stream: str, message: Dict[str, Any], max_len: int = 10000):
        """Publish message to Redis Stream"""
        if not self.client:
            raise RuntimeError("Redis client not initialized")
            
        try:
            # Serialize complex objects
            serialized_message = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in message.items()
            }
            
            message_id = await self.client.xadd(
                stream, 
                serialized_message, 
                maxlen=max_len,
                approximate=True
            )
            
            logger.debug(f"Message sent to stream {stream}: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send message to stream {stream}: {e}")
            raise
            
    async def subscribe(self, streams: List[str], group_id: str, consumer_id: str, callback: Callable):
        """Subscribe to Redis Streams"""
        if not self.client:
            raise RuntimeError("Redis client not initialized")
            
        try:
            # Create consumer group if it doesn't exist
            for stream in streams:
                try:
                    await self.client.xgroup_create(stream, group_id, id='0', mkstream=True)
                except redis.ResponseError as e:
                    if "BUSYGROUP" not in str(e):
                        raise
                        
            # Start consuming task
            task = asyncio.create_task(
                self._consume_messages(streams, group_id, consumer_id, callback)
            )
            self.consumers[f"{group_id}:{consumer_id}"] = task
            
            logger.info(f"Subscribed to streams {streams} with group {group_id}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to streams {streams}: {e}")
            raise
            
    async def _consume_messages(self, streams: List[str], group_id: str, consumer_id: str, callback: Callable):
        """Consume messages from Redis Streams"""
        stream_dict = {stream: '>' for stream in streams}
        
        while True:
            try:
                messages = await self.client.xreadgroup(
                    group_id,
                    consumer_id,
                    stream_dict,
                    count=10,
                    block=1000
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            # Deserialize message
                            deserialized_fields = {}
                            for k, v in fields.items():
                                try:
                                    deserialized_fields[k] = json.loads(v)
                                except (json.JSONDecodeError, TypeError):
                                    deserialized_fields[k] = v
                                    
                            await callback(stream, msg_id, deserialized_fields)
                            
                            # Acknowledge message
                            await self.client.xack(stream, group_id, msg_id)
                            
                        except Exception as e:
                            logger.error(f"Error processing message {msg_id}: {e}")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Redis Streams consumption: {e}")
                await asyncio.sleep(1)
                
    async def stop(self):
        """Stop Redis client"""
        # Cancel all consumer tasks
        for task in self.consumers.values():
            task.cancel()
            
        # Wait for tasks to complete
        if self.consumers:
            await asyncio.gather(*self.consumers.values(), return_exceptions=True)
            
        if self.client:
            await self.client.close()
            
        logger.info("Redis Streams client stopped")
```

#### Message Broker Abstraction
```python
# shared/messaging/message_broker.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
import asyncio

class MessageBrokerType(Enum):
    KAFKA = "kafka"
    REDIS_STREAMS = "redis_streams"

class MessageBroker(ABC):
    @abstractmethod
    async def start(self):
        pass
        
    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        pass
        
    @abstractmethod
    async def subscribe(self, topics: List[str], group_id: str, callback: Callable):
        pass
        
    @abstractmethod
    async def stop(self):
        pass

class HybridMessageBroker:
    """Hybrid message broker using both Kafka and Redis Streams"""
    
    def __init__(self, kafka_client: AsyncKafkaClient, redis_client: AsyncRedisStreamsClient):
        self.kafka_client = kafka_client
        self.redis_client = redis_client
        
    async def start(self):
        """Start both clients"""
        await asyncio.gather(
            self.kafka_client.start(),
            self.redis_client.start()
        )
        
    async def publish_fast(self, stream: str, message: Dict[str, Any]):
        """Publish to Redis Streams for ultra-low latency"""
        return await self.redis_client.publish(stream, message)
        
    async def publish_reliable(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        """Publish to Kafka for reliability and history"""
        return await self.kafka_client.publish(topic, message, key)
        
    async def publish_both(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        """Publish to both brokers"""
        await asyncio.gather(
            self.kafka_client.publish(topic, message, key),
            self.redis_client.publish(topic, message)
        )
        
    async def subscribe_fast(self, streams: List[str], group_id: str, consumer_id: str, callback: Callable):
        """Subscribe to Redis Streams for real-time processing"""
        await self.redis_client.subscribe(streams, group_id, consumer_id, callback)
        
    async def subscribe_reliable(self, topics: List[str], group_id: str, callback: Callable):
        """Subscribe to Kafka for reliable processing"""
        await self.kafka_client.subscribe(topics, group_id, callback)
        
    async def stop(self):
        """Stop both clients"""
        await asyncio.gather(
            self.kafka_client.stop(),
            self.redis_client.stop()
        )
```

### Paso 2: Database Library (2-3 horas)

#### Connection Manager
```python
# shared/database/connection.py
import asyncio
from typing import Optional, Dict, Any
import asyncpg
from asyncpg import Pool
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, dsn: str, min_connections: int = 5, max_connections: int = 20):
        self.dsn = dsn
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool: Optional[Pool] = None
        
    async def start(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=60,
                server_settings={
                    'jit': 'off',
                    'application_name': 'trading_agent'
                }
            )
            logger.info(f"Database pool initialized with {self.min_connections}-{self.max_connections} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
            
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
            
        async with self.pool.acquire() as connection:
            yield connection
            
    async def execute(self, query: str, *args) -> str:
        """Execute query and return status"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
            
    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
            
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch single row"""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
            
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch single value"""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
            
    async def stop(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
```

### Paso 3: Configuration Library (2-3 horas)

#### Consul Client
```python
# shared/config/consul_client.py
import asyncio
import json
from typing import Dict, Any, Optional, Callable
import aiohttp
import logging

logger = logging.getLogger(__name__)

class ConsulClient:
    def __init__(self, host: str = "localhost", port: int = 8500):
        self.base_url = f"http://{host}:{port}/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.watchers: Dict[str, asyncio.Task] = {}
        
    async def start(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        logger.info("Consul client initialized")
        
    async def get_config(self, key: str) -> Optional[Dict[str, Any]]:
        """Get configuration from Consul KV store"""
        if not self.session:
            raise RuntimeError("Consul client not initialized")
            
        try:
            async with self.session.get(f"{self.base_url}/kv/{key}") as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        # Decode base64 value
                        import base64
                        value = base64.b64decode(data[0]['Value']).decode('utf-8')
                        return json.loads(value)
                elif response.status == 404:
                    return None
                else:
                    response.raise_for_status()
                    
        except Exception as e:
            logger.error(f"Failed to get config {key}: {e}")
            raise
            
    async def set_config(self, key: str, value: Dict[str, Any]) -> bool:
        """Set configuration in Consul KV store"""
        if not self.session:
            raise RuntimeError("Consul client not initialized")
            
        try:
            json_value = json.dumps(value)
            async with self.session.put(
                f"{self.base_url}/kv/{key}",
                data=json_value
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            raise
            
    async def watch_config(self, key: str, callback: Callable[[Dict[str, Any]], None]):
        """Watch for configuration changes"""
        if key in self.watchers:
            return
            
        task = asyncio.create_task(self._watch_key(key, callback))
        self.watchers[key] = task
        logger.info(f"Started watching config key: {key}")
        
    async def _watch_key(self, key: str, callback: Callable):
        """Watch a specific key for changes"""
        last_modify_index = 0
        
        while True:
            try:
                params = {'index': last_modify_index, 'wait': '30s'}
                async with self.session.get(
                    f"{self.base_url}/kv/{key}",
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            modify_index = data[0]['ModifyIndex']
                            if modify_index > last_modify_index:
                                # Configuration changed
                                import base64
                                value = base64.b64decode(data[0]['Value']).decode('utf-8')
                                config = json.loads(value)
                                await callback(config)
                                last_modify_index = modify_index
                                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching key {key}: {e}")
                await asyncio.sleep(5)
                
    async def stop(self):
        """Stop Consul client"""
        # Cancel all watchers
        for task in self.watchers.values():
            task.cancel()
            
        if self.watchers:
            await asyncio.gather(*self.watchers.values(), return_exceptions=True)
            
        if self.session:
            await self.session.close()
            
        logger.info("Consul client stopped")
```

### ✅ Testing y Validación

#### Unit Tests
```python
# tests/unit/test_messaging.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from shared.messaging.message_broker import HybridMessageBroker

@pytest.mark.asyncio
async def test_hybrid_message_broker():
    # Mock clients
    kafka_client = Mock()
    kafka_client.start = AsyncMock()
    kafka_client.publish = AsyncMock(return_value="kafka_result")
    
    redis_client = Mock()
    redis_client.start = AsyncMock()
    redis_client.publish = AsyncMock(return_value="redis_result")
    
    # Test broker
    broker = HybridMessageBroker(kafka_client, redis_client)
    await broker.start()
    
    # Test publishing
    result = await broker.publish_fast("test_stream", {"key": "value"})
    assert result == "redis_result"
    
    result = await broker.publish_reliable("test_topic", {"key": "value"})
    assert result == "kafka_result"
```

#### Integration Tests
```python
# tests/integration/test_database.py
import pytest
import asyncio
from shared.database.connection import DatabaseManager

@pytest.mark.asyncio
async def test_database_connection():
    dsn = "postgresql://trader:password@localhost:5432/trading_agent"
    db = DatabaseManager(dsn, min_connections=1, max_connections=2)
    
    await db.start()
    
    # Test basic query
    result = await db.fetchval("SELECT 1")
    assert result == 1
    
    # Test with parameters
    result = await db.fetchval("SELECT $1::text", "test")
    assert result == "test"
    
    await db.stop()
```

## ✅ Checklist de Completitud

### Messaging Library
- [ ] Kafka client implementado y testeado
- [ ] Redis Streams client implementado y testeado
- [ ] Hybrid message broker funcionando
- [ ] Message schemas definidos
- [ ] Error handling implementado
- [ ] Logging configurado
- [ ] Unit tests pasando (>80% coverage)
- [ ] Integration tests pasando

### Database Library
- [ ] Connection pool configurado
- [ ] Context managers implementados
- [ ] Query helpers funcionando
- [ ] Transaction support implementado
- [ ] Error handling robusto
- [ ] Connection health checks
- [ ] Unit tests pasando
- [ ] Integration tests con PostgreSQL

### Configuration Library
- [ ] Consul client funcionando
- [ ] Configuration watching implementado
- [ ] Hot-reload funcionando
- [ ] Error handling y fallbacks
- [ ] Configuration validation
- [ ] Unit tests pasando
- [ ] Integration tests con Consul

### Logging Library
- [ ] Structured logging configurado
- [ ] Custom formatters implementados
- [ ] Log levels configurables
- [ ] Performance optimizado
- [ ] Integration con monitoring
- [ ] Unit tests pasando

### Metrics Library
- [ ] Prometheus client configurado
- [ ] Custom collectors implementados
- [ ] Decorators para métricas automáticas
- [ ] Performance counters
- [ ] Health metrics
- [ ] Unit tests pasando

## 📊 Métricas de Éxito

- **Code Coverage**: >80% en todas las librerías
- **Performance**: <10ms latencia promedio para operaciones básicas
- **Reliability**: >99.9% uptime en tests de stress
- **Memory Usage**: <100MB por proceso
- **Documentation**: 100% de APIs documentadas

## 🚨 Troubleshooting

### Problemas Comunes

#### Conexiones de base de datos
```python
# Verificar pool de conexiones
async def check_db_pool():
    print(f"Pool size: {db.pool.get_size()}")
    print(f"Pool idle: {db.pool.get_idle_size()}")
```

#### Mensajería
```bash
# Verificar topics Kafka
docker exec trading-kafka kafka-topics --list --bootstrap-server localhost:9092

# Verificar streams Redis
docker exec trading-redis-master redis-cli XINFO GROUPS test-stream
```

**Tiempo estimado**: 8-12 horas  
**Responsable**: Backend Developer  
**Dependencias**: Docker infrastructure funcionando

---

**Next Step**: Una vez completadas las shared libraries, proceder con [Stage 2: Data Ingestion](./stage-2-data-ingestion.md)
