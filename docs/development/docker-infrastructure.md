# 🐳 Docker Infrastructure Setup

## 📋 CHECKLIST: Docker Infrastructure

### ✅ Prerrequisitos
- [ ] Environment setup completado
- [ ] Docker Desktop funcionando
- [ ] Variables de entorno configuradas (.env)
- [ ] Puertos disponibles verificados

### ✅ Servicios de Infraestructura

#### Base de Datos Principal
- [ ] **PostgreSQL + TimescaleDB** (Puerto 5432)
  - Almacenamiento de trades, posiciones, señales
  - Extensión TimescaleDB para series temporales
  - Configuración de replicación
  - Backup automático configurado

#### Cache y Mensajería
- [ ] **Redis Cluster** (Puertos 6379-6384)
  - Cache de datos en tiempo real
  - Redis Streams para mensajería ultra-rápida
  - Configuración de persistencia
  - Clustering para alta disponibilidad

- [ ] **Apache Kafka** (Puerto 9092)
  - Mensajería asíncrona entre módulos
  - Retención de mensajes para auditoría
  - Múltiples particiones para escalabilidad
  - Zookeeper para coordinación (Puerto 2181)

#### Almacenamiento de Objetos
- [ ] **MinIO** (Puertos 9000, 9001)
  - Almacenamiento S3-compatible
  - Datos raw de mercado
  - Modelos ML y backups
  - Interface web para gestión

#### Monitoreo y Métricas
- [ ] **InfluxDB** (Puerto 8086)
  - Métricas de performance en tiempo real
  - Datos de latencia y throughput
  - Retención automática de datos

- [ ] **Prometheus** (Puerto 9090)
  - Recolección de métricas de aplicación
  - Alerting rules configuradas
  - Service discovery automático

- [ ] **Grafana** (Puerto 3000)
  - Dashboards de trading
  - Alertas visuales
  - Reportes automáticos

#### Configuración Centralizada
- [ ] **Consul** (Puerto 8500)
  - Service discovery
  - Configuración centralizada
  - Health checks automáticos
  - Hot-reload de configuraciones

### ✅ Arquitectura de Red

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network                           │
│                     trading-network                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ PostgreSQL  │  │    Redis    │  │   Kafka     │             │
│  │ TimescaleDB │  │   Cluster   │  │ Zookeeper   │             │
│  │   :5432     │  │ :6379-6384  │  │ :9092,:2181 │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   MinIO     │  │  InfluxDB   │  │   Consul    │             │
│  │ :9000,:9001 │  │    :8086    │  │    :8500    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │ Prometheus  │  │   Grafana   │                              │
│  │    :9090    │  │    :3000    │                              │
│  └─────────────┘  └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Implementación

### Paso 1: Docker Compose Principal

```yaml
# docker-compose.infrastructure.yml
version: '3.8'

networks:
  trading-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres_data:
  redis_data:
  kafka_data:
  zookeeper_data:
  minio_data:
  influxdb_data:
  consul_data:
  prometheus_data:
  grafana_data:

services:
  # PostgreSQL + TimescaleDB
  postgres:
    image: timescale/timescaledb:latest-pg15
    container_name: trading-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: trading_agent
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      TIMESCALEDB_TELEMETRY: off
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/postgres/init:/docker-entrypoint-initdb.d
    networks:
      trading-network:
        ipv4_address: 172.20.0.10
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trader -d trading_agent"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cluster
  redis-master:
    image: redis:7-alpine
    container_name: trading-redis-master
    restart: unless-stopped
    command: >
      redis-server 
      --requirepass ${REDIS_PASSWORD}
      --appendonly yes
      --appendfsync everysec
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      trading-network:
        ipv4_address: 172.20.0.11
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Zookeeper
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    container_name: trading-zookeeper
    restart: unless-stopped
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
    networks:
      trading-network:
        ipv4_address: 172.20.0.12

  # Kafka
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    container_name: trading-kafka
    restart: unless-stopped
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: true
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
    ports:
      - "9092:9092"
    volumes:
      - kafka_data:/var/lib/kafka/data
    networks:
      trading-network:
        ipv4_address: 172.20.0.13
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9092"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MinIO
  minio:
    image: minio/minio:latest
    container_name: trading-minio
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    networks:
      trading-network:
        ipv4_address: 172.20.0.14
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  # InfluxDB
  influxdb:
    image: influxdb:2.7-alpine
    container_name: trading-influxdb
    restart: unless-stopped
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: ${INFLUX_PASSWORD}
      DOCKER_INFLUXDB_INIT_ORG: trading-org
      DOCKER_INFLUXDB_INIT_BUCKET: trading-metrics
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2
    networks:
      trading-network:
        ipv4_address: 172.20.0.15

  # Consul
  consul:
    image: consul:1.16
    container_name: trading-consul
    restart: unless-stopped
    command: >
      consul agent 
      -server 
      -bootstrap-expect=1 
      -ui 
      -client=0.0.0.0 
      -bind=0.0.0.0
    ports:
      - "8500:8500"
    volumes:
      - consul_data:/consul/data
      - ./infrastructure/consul/config:/consul/config
    networks:
      trading-network:
        ipv4_address: 172.20.0.16

  # Prometheus
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: trading-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    volumes:
      - prometheus_data:/prometheus
      - ./infrastructure/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./infrastructure/prometheus/rules:/etc/prometheus/rules
    networks:
      trading-network:
        ipv4_address: 172.20.0.17

  # Grafana
  grafana:
    image: grafana/grafana:10.2.0
    container_name: trading-grafana
    restart: unless-stopped
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
      GF_USERS_ALLOW_SIGN_UP: false
      GF_INSTALL_PLUGINS: grafana-clock-panel,grafana-simple-json-datasource
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infrastructure/grafana/provisioning:/etc/grafana/provisioning
      - ./infrastructure/grafana/dashboards:/var/lib/grafana/dashboards
    networks:
      trading-network:
        ipv4_address: 172.20.0.18
    depends_on:
      - prometheus
      - influxdb
```

### Paso 2: Configuraciones de Servicios

#### PostgreSQL Initialization
```sql
-- infrastructure/postgres/init/01-init.sql
-- Create extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create users
CREATE USER data_ingestion WITH PASSWORD 'data_pass_2024';
CREATE USER signal_generator WITH PASSWORD 'signal_pass_2024';
CREATE USER risk_manager WITH PASSWORD 'risk_pass_2024';
CREATE USER trading_executor WITH PASSWORD 'executor_pass_2024';

-- Grant permissions
GRANT USAGE ON SCHEMA trading TO data_ingestion, signal_generator, risk_manager, trading_executor;
GRANT USAGE ON SCHEMA analytics TO data_ingestion, signal_generator;
GRANT USAGE ON SCHEMA monitoring TO ALL;

-- Create basic tables
CREATE TABLE IF NOT EXISTS trading.market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    bid DECIMAL(20,8),
    ask DECIMAL(20,8),
    spread DECIMAL(10,8),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('trading.market_data', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_timestamp ON trading.trades (timestamp);
CREATE INDEX idx_trades_symbol ON trading.trades (symbol);
CREATE INDEX idx_trades_strategy ON trading.trades (strategy_id);
```

#### Prometheus Configuration
```yaml
# infrastructure/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'trading-services'
    consul_sd_configs:
      - server: 'consul:8500'
        services: ['data-ingestion', 'signal-generation', 'risk-manager', 'trading-execution']
    relabel_configs:
      - source_labels: [__meta_consul_service]
        target_label: service
      - source_labels: [__meta_consul_node]
        target_label: node

  - job_name: 'infrastructure'
    static_configs:
      - targets: 
        - 'postgres:5432'
        - 'redis-master:6379'
        - 'kafka:9092'
        - 'minio:9000'
        - 'influxdb:8086'
        - 'consul:8500'
```

#### Grafana Provisioning
```yaml
# infrastructure/grafana/provisioning/datasources/datasources.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    database: trading-metrics
    user: admin
    secureJsonData:
      password: ${INFLUX_PASSWORD}

  - name: PostgreSQL
    type: postgres
    access: proxy
    url: postgres:5432
    database: trading_agent
    user: trader
    secureJsonData:
      password: ${POSTGRES_PASSWORD}
```

### Paso 3: Scripts de Gestión

#### Startup Script
```bash
#!/bin/bash
# scripts/start-infrastructure.sh

echo "🚀 Starting BTC Trading Agent Infrastructure..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Load environment variables
source .env

# Create necessary directories
mkdir -p {infrastructure/{postgres/init,prometheus/rules,grafana/{provisioning,dashboards}},logs}

# Start infrastructure services
echo "📦 Starting infrastructure services..."
docker-compose -f docker-compose.infrastructure.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🏥 Running health checks..."
./scripts/health-check-infrastructure.sh

echo "✅ Infrastructure startup complete!"
echo "🌐 Access points:"
echo "  - Grafana: http://localhost:3000 (admin/${GRAFANA_ADMIN_PASSWORD})"
echo "  - Prometheus: http://localhost:9090"
echo "  - MinIO Console: http://localhost:9001 (${MINIO_ROOT_USER}/${MINIO_ROOT_PASSWORD})"
echo "  - Consul UI: http://localhost:8500"
```

#### Health Check Script
```bash
#!/bin/bash
# scripts/health-check-infrastructure.sh

echo "🏥 Infrastructure Health Check"
echo "=============================="

services=(
    "postgres:5432"
    "redis-master:6379"
    "kafka:9092"
    "minio:9000"
    "influxdb:8086"
    "consul:8500"
    "prometheus:9090"
    "grafana:3000"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    echo -n "Checking $name:$port... "
    
    if docker exec trading-$name sh -c "nc -z localhost $port" 2>/dev/null; then
        echo "✅"
    else
        echo "❌"
    fi
done

# Check Docker containers
echo ""
echo "📦 Container Status:"
docker-compose -f docker-compose.infrastructure.yml ps

# Check disk usage
echo ""
echo "💾 Disk Usage:"
docker system df

echo ""
echo "🎯 Infrastructure health check complete!"
```

#### Cleanup Script
```bash
#!/bin/bash
# scripts/cleanup-infrastructure.sh

echo "🧹 Cleaning up BTC Trading Agent Infrastructure..."

# Stop all containers
docker-compose -f docker-compose.infrastructure.yml down

# Remove volumes (optional - uncomment if needed)
# echo "⚠️  Removing all data volumes..."
# docker-compose -f docker-compose.infrastructure.yml down -v

# Clean up unused Docker resources
echo "🗑️  Cleaning up unused Docker resources..."
docker system prune -f

# Remove dangling images
docker image prune -f

echo "✅ Cleanup complete!"
```

### ✅ Validación y Testing

#### Test de Conectividad
```python
# scripts/test-infrastructure.py
#!/usr/bin/env python3
import asyncio
import asyncpg
import redis
import boto3
from kafka import KafkaProducer, KafkaConsumer
import requests
from dotenv import load_dotenv
import os

load_dotenv()

async def test_postgres():
    try:
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='trader',
            password=os.getenv('POSTGRES_PASSWORD'),
            database='trading_agent'
        )
        result = await conn.fetchval('SELECT version()')
        await conn.close()
        print("✅ PostgreSQL: Connected successfully")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: {e}")
        return False

def test_redis():
    try:
        r = redis.Redis(
            host='localhost',
            port=6379,
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )
        r.ping()
        print("✅ Redis: Connected successfully")
        return True
    except Exception as e:
        print(f"❌ Redis: {e}")
        return False

def test_kafka():
    try:
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: x.encode('utf-8')
        )
        producer.send('test-topic', 'test-message')
        producer.flush()
        producer.close()
        print("✅ Kafka: Connected successfully")
        return True
    except Exception as e:
        print(f"❌ Kafka: {e}")
        return False

def test_minio():
    try:
        client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id=os.getenv('MINIO_ROOT_USER'),
            aws_secret_access_key=os.getenv('MINIO_ROOT_PASSWORD')
        )
        client.list_buckets()
        print("✅ MinIO: Connected successfully")
        return True
    except Exception as e:
        print(f"❌ MinIO: {e}")
        return False

def test_monitoring():
    services = {
        'Prometheus': 'http://localhost:9090/-/healthy',
        'Grafana': 'http://localhost:3000/api/health',
        'Consul': 'http://localhost:8500/v1/status/leader',
        'InfluxDB': 'http://localhost:8086/health'
    }
    
    results = {}
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: Healthy")
                results[name] = True
            else:
                print(f"❌ {name}: Status {response.status_code}")
                results[name] = False
        except Exception as e:
            print(f"❌ {name}: {e}")
            results[name] = False
    
    return all(results.values())

async def main():
    print("🔍 Testing infrastructure connectivity...")
    
    postgres_ok = await test_postgres()
    redis_ok = test_redis()
    kafka_ok = test_kafka()
    minio_ok = test_minio()
    monitoring_ok = test_monitoring()
    
    all_ok = all([postgres_ok, redis_ok, kafka_ok, minio_ok, monitoring_ok])
    
    if all_ok:
        print("\n🎉 All infrastructure services are healthy!")
    else:
        print("\n⚠️  Some services failed health checks.")
    
    return all_ok

if __name__ == "__main__":
    asyncio.run(main())
```

## 📊 Monitoreo y Métricas

### Métricas Clave a Monitorear

#### Infraestructura
- **CPU Usage**: <70% promedio
- **Memory Usage**: <80% promedio
- **Disk I/O**: <80% utilización
- **Network Latency**: <10ms interno

#### Bases de Datos
- **PostgreSQL Connections**: <80% del máximo
- **Query Response Time**: <100ms promedio
- **Redis Memory Usage**: <90% del límite
- **Cache Hit Rate**: >95%

#### Mensajería
- **Kafka Lag**: <1000 mensajes
- **Message Throughput**: >1000 msg/s
- **Redis Streams Backlog**: <100 mensajes

## 🚨 Troubleshooting

### Problemas Comunes

#### Servicios no inician
```bash
# Verificar logs
docker-compose -f docker-compose.infrastructure.yml logs <service-name>

# Reiniciar servicio específico
docker-compose -f docker-compose.infrastructure.yml restart <service-name>
```

#### Problemas de conectividad
```bash
# Verificar red Docker
docker network ls
docker network inspect trading-network

# Test de conectividad entre contenedores
docker exec trading-postgres ping trading-redis-master
```

#### Problemas de performance
```bash
# Verificar recursos
docker stats

# Verificar espacio en disco
docker system df
```

## ✅ Checklist Final

- [ ] Todos los servicios iniciando correctamente
- [ ] Health checks pasando
- [ ] Conectividad entre servicios verificada
- [ ] Monitoreo funcionando
- [ ] Dashboards accesibles
- [ ] Backup y recovery procedures documentados
- [ ] Performance benchmarks establecidos

**Tiempo estimado**: 4-6 horas  
**Responsable**: DevOps Engineer

---

**Next Step**: Una vez completada la infraestructura, proceder con [Shared Libraries Development](./stages/stage-1-shared-libraries.md)
