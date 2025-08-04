# 🚀 Production Deployment Guide

## 📋 CHECKLIST: Production Deployment

### ✅ Pre-Deployment
- [ ] All development stages completed and tested
- [ ] Paper trading validation successful (>1 week)
- [ ] Performance benchmarks met
- [ ] Security audit completed
- [ ] Backup procedures tested
- [ ] Monitoring dashboards configured
- [ ] Alert systems tested
- [ ] Documentation updated

### ✅ Infrastructure Setup
- [ ] Production environment provisioned
- [ ] SSL certificates configured
- [ ] Domain names configured
- [ ] Load balancers configured
- [ ] Database clusters setup
- [ ] Storage buckets created
- [ ] Monitoring infrastructure deployed

## 🏗️ Deployment Architecture

### **Production Environment**
```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ENVIRONMENT                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Load Balancer │  │   API Gateway   │  │   Web Dashboard │ │
│  │   (nginx/HAProxy│  │   (Kong/Traefik)│  │   (React/Vue)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                       │                       │     │
│           ▼                       ▼                       ▼     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              KUBERNETES CLUSTER                             │ │
│  │                                                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │ │
│  │  │Data Ingestion│ │Feature Eng. │ │Signal Gen.  │          │ │
│  │  │   (3 pods)   │ │  (2 pods)   │ │  (2 pods)   │          │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘          │ │
│  │                                                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │ │
│  │  │Risk Manager │ │Trade Exec.  │ │Monitoring   │          │ │
│  │  │  (2 pods)   │ │  (2 pods)   │ │  (1 pod)    │          │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│           │                       │                       │     │
│           ▼                       ▼                       ▼     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Databases     │  │   Message Queue │  │    Storage      │ │
│  │                 │  │                 │  │                 │ │
│  │ • PostgreSQL    │  │ • Kafka Cluster │  │ • MinIO Cluster │ │
│  │ • TimescaleDB   │  │ • Redis Cluster │  │ • Backup Storage│ │
│  │ • InfluxDB      │  │ • Schema Reg.   │  │ • Log Storage   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🐳 Docker Production Setup

### **Production Docker Compose**
```yaml
# docker-compose.production.yml
version: '3.8'

networks:
  trading-prod:
    driver: bridge
    ipam:
      config:
        - subnet: 172.30.0.0/16

volumes:
  postgres_prod_data:
  redis_prod_data:
  kafka_prod_data:
  minio_prod_data:
  influxdb_prod_data:
  prometheus_prod_data:
  grafana_prod_data:

services:
  # Production PostgreSQL with TimescaleDB
  postgres-prod:
    image: timescale/timescaledb:latest-pg15
    container_name: trading-postgres-prod
    restart: always
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      TIMESCALEDB_TELEMETRY: off
    ports:
      - "5432:5432"
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
      - ./infrastructure/postgres/prod-init:/docker-entrypoint-initdb.d
    networks:
      trading-prod:
        ipv4_address: 172.30.0.10
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Production Redis Cluster
  redis-prod-master:
    image: redis:7-alpine
    container_name: trading-redis-prod-master
    restart: always
    command: >
      redis-server 
      --requirepass ${REDIS_PASSWORD}
      --appendonly yes
      --appendfsync everysec
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
    ports:
      - "6379:6379"
    volumes:
      - redis_prod_data:/data
    networks:
      trading-prod:
        ipv4_address: 172.30.0.11
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  # Production Kafka
  kafka-prod:
    image: confluentinc/cp-kafka:7.4.0
    container_name: trading-kafka-prod
    restart: always
    depends_on:
      - zookeeper-prod
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper-prod:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-prod:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: false
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_NUM_PARTITIONS: 3
      KAFKA_DEFAULT_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"
    volumes:
      - kafka_prod_data:/var/lib/kafka/data
    networks:
      trading-prod:
        ipv4_address: 172.30.0.13
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  # Trading Services
  data-ingestion-prod:
    build:
      context: .
      dockerfile: modules/data-ingestion/Dockerfile.prod
    container_name: trading-data-ingestion-prod
    restart: always
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - POSTGRES_DSN=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres-prod:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis-prod-master:6379
      - KAFKA_BOOTSTRAP_SERVERS=kafka-prod:29092
    depends_on:
      - postgres-prod
      - redis-prod-master
      - kafka-prod
    networks:
      - trading-prod
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  signal-generation-prod:
    build:
      context: .
      dockerfile: modules/signal-generation/Dockerfile.prod
    container_name: trading-signal-generation-prod
    restart: always
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - MODEL_PATH=/app/models/production_model.joblib
    volumes:
      - ./data/models:/app/models:ro
    depends_on:
      - data-ingestion-prod
    networks:
      - trading-prod
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  risk-manager-prod:
    build:
      context: .
      dockerfile: modules/risk-manager/Dockerfile.prod
    container_name: trading-risk-manager-prod
    restart: always
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - MAX_DAILY_DRAWDOWN=0.02
      - MAX_POSITION_SIZE=0.10
    depends_on:
      - signal-generation-prod
    networks:
      - trading-prod
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  trading-execution-prod:
    build:
      context: .
      dockerfile: modules/trading-execution/Dockerfile.prod
    container_name: trading-execution-prod
    restart: always
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_SECRET_KEY=${BINANCE_SECRET_KEY}
      - BINANCE_TESTNET=false
    depends_on:
      - risk-manager-prod
    networks:
      - trading-prod
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
```

## 🔧 Production Configuration

### **Environment Variables**
```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database
POSTGRES_DB=trading_agent_prod
POSTGRES_USER=trader_prod
POSTGRES_PASSWORD=<secure_password>

# Redis
REDIS_PASSWORD=<secure_password>

# Trading APIs
BINANCE_API_KEY=<production_api_key>
BINANCE_SECRET_KEY=<production_secret_key>
BINANCE_TESTNET=false

# Social Data
TWITTER_BEARER_TOKEN=<production_bearer_token>

# Object Storage
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=<secure_password>

# Monitoring
GRAFANA_ADMIN_PASSWORD=<secure_password>
INFLUX_PASSWORD=<secure_password>

# Security
JWT_SECRET=<secure_jwt_secret>
ENCRYPTION_KEY=<secure_encryption_key>

# Performance
MAX_WORKERS=4
BATCH_SIZE=1000
CACHE_TTL=300
```

### **Production Dockerfile Example**
```dockerfile
# modules/signal-generation/Dockerfile.prod
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY modules/signal-generation/requirements.txt ./signal-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r signal-requirements.txt

# Copy shared libraries
COPY shared/ ./shared/

# Copy module code
COPY modules/signal-generation/ ./modules/signal-generation/

# Create non-root user
RUN useradd --create-home --shell /bin/bash trader
RUN chown -R trader:trader /app
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "modules.signal_generation.main"]
```

## 🚀 Deployment Scripts

### **Production Deployment Script**
```bash
#!/bin/bash
# scripts/deploy-production.sh

set -e

echo "🚀 Starting BTC Trading Agent Production Deployment..."

# Configuration
ENVIRONMENT="production"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
HEALTH_CHECK_TIMEOUT=300

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check if all required environment variables are set
required_vars=(
    "POSTGRES_PASSWORD"
    "REDIS_PASSWORD" 
    "BINANCE_API_KEY"
    "BINANCE_SECRET_KEY"
    "TWITTER_BEARER_TOKEN"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    fi
done

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Create backup
echo "💾 Creating backup..."
mkdir -p $BACKUP_DIR
./scripts/backup-production.sh $BACKUP_DIR

# Build production images
echo "🏗️ Building production images..."
docker-compose -f docker-compose.production.yml build --no-cache

# Stop existing services gracefully
echo "⏹️ Stopping existing services..."
docker-compose -f docker-compose.production.yml down --timeout 60

# Start infrastructure services first
echo "🏗️ Starting infrastructure services..."
docker-compose -f docker-compose.production.yml up -d \
    postgres-prod redis-prod-master kafka-prod zookeeper-prod \
    minio-prod influxdb-prod consul-prod

# Wait for infrastructure to be ready
echo "⏳ Waiting for infrastructure services..."
sleep 60

# Health check infrastructure
./scripts/health-check-infrastructure.sh

# Start trading services
echo "📈 Starting trading services..."
docker-compose -f docker-compose.production.yml up -d \
    data-ingestion-prod feature-engineering-prod signal-generation-prod \
    risk-manager-prod trading-execution-prod

# Start monitoring services
echo "📊 Starting monitoring services..."
docker-compose -f docker-compose.production.yml up -d \
    prometheus-prod grafana-prod

# Wait for all services to be ready
echo "⏳ Waiting for all services to start..."
sleep 120

# Health check all services
echo "🏥 Running health checks..."
if ./scripts/health-check-production.sh; then
    echo "✅ All services are healthy"
else
    echo "❌ Health check failed, rolling back..."
    ./scripts/rollback-production.sh $BACKUP_DIR
    exit 1
fi

# Run smoke tests
echo "🧪 Running smoke tests..."
if ./scripts/smoke-tests.sh; then
    echo "✅ Smoke tests passed"
else
    echo "❌ Smoke tests failed, rolling back..."
    ./scripts/rollback-production.sh $BACKUP_DIR
    exit 1
fi

# Enable monitoring alerts
echo "🚨 Enabling monitoring alerts..."
./scripts/enable-alerts.sh

echo "🎉 Production deployment completed successfully!"
echo "📊 Dashboard: https://dashboard.btc-trading-agent.com"
echo "📈 Grafana: https://monitoring.btc-trading-agent.com"
echo "📋 Logs: docker-compose -f docker-compose.production.yml logs -f"
```

### **Health Check Script**
```bash
#!/bin/bash
# scripts/health-check-production.sh

echo "🏥 Production Health Check"
echo "========================="

# Services to check
services=(
    "postgres-prod:5432"
    "redis-prod-master:6379"
    "kafka-prod:9092"
    "data-ingestion-prod:8000"
    "signal-generation-prod:8001"
    "risk-manager-prod:8002"
    "trading-execution-prod:8003"
)

all_healthy=true

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    echo -n "Checking $name:$port... "
    
    if docker exec $name sh -c "nc -z localhost $port" 2>/dev/null; then
        echo "✅"
    else
        echo "❌"
        all_healthy=false
    fi
done

# Check API endpoints
echo ""
echo "Checking API endpoints..."

endpoints=(
    "http://localhost:8000/health"
    "http://localhost:8001/health"
    "http://localhost:8002/health"
    "http://localhost:8003/health"
)

for endpoint in "${endpoints[@]}"; do
    echo -n "Checking $endpoint... "
    
    if curl -f -s $endpoint > /dev/null; then
        echo "✅"
    else
        echo "❌"
        all_healthy=false
    fi
done

# Check trading system status
echo ""
echo "Checking trading system status..."

# Check if data is flowing
echo -n "Data ingestion active... "
if docker logs data-ingestion-prod --tail 10 | grep -q "Data received"; then
    echo "✅"
else
    echo "❌"
    all_healthy=false
fi

# Check if signals are being generated
echo -n "Signal generation active... "
if docker logs signal-generation-prod --tail 10 | grep -q "Signal generated"; then
    echo "✅"
else
    echo "❌"
    all_healthy=false
fi

if $all_healthy; then
    echo ""
    echo "🎉 All systems healthy!"
    exit 0
else
    echo ""
    echo "⚠️ Some systems are unhealthy!"
    exit 1
fi
```

## 🔒 Security Hardening

### **SSL/TLS Configuration**
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.btc-trading-agent.com;
    
    ssl_certificate /etc/ssl/certs/btc-trading-agent.crt;
    ssl_certificate_key /etc/ssl/private/btc-trading-agent.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://trading-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### **Firewall Rules**
```bash
# firewall-rules.sh
#!/bin/bash

# Allow SSH
ufw allow 22/tcp

# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow monitoring
ufw allow 3000/tcp  # Grafana
ufw allow 9090/tcp  # Prometheus

# Deny all other incoming
ufw default deny incoming
ufw default allow outgoing

# Enable firewall
ufw --force enable
```

## 📊 Monitoring Setup

### **Production Alerts**
```yaml
# infrastructure/prometheus/prod-alerts.yml
groups:
  - name: trading-system
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, execution_latency_seconds) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High execution latency detected"
          
      - alert: LowWinRate
        expr: win_rate < 50
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Win rate below 50%"
          
      - alert: HighDrawdown
        expr: current_drawdown < -0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Drawdown exceeds 5%"
          
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"
```

## 🔄 Backup & Recovery

### **Automated Backup Script**
```bash
#!/bin/bash
# scripts/backup-production.sh

BACKUP_DIR=${1:-"./backups/$(date +%Y%m%d_%H%M%S)"}
mkdir -p $BACKUP_DIR

echo "💾 Starting production backup to $BACKUP_DIR..."

# Database backup
echo "Backing up PostgreSQL..."
docker exec postgres-prod pg_dump -U trader_prod trading_agent_prod > $BACKUP_DIR/postgres_backup.sql

# Redis backup
echo "Backing up Redis..."
docker exec redis-prod-master redis-cli --rdb $BACKUP_DIR/redis_backup.rdb

# Configuration backup
echo "Backing up configurations..."
cp -r infrastructure/ $BACKUP_DIR/
cp .env.production $BACKUP_DIR/
cp docker-compose.production.yml $BACKUP_DIR/

# Model backup
echo "Backing up models..."
cp -r data/models/ $BACKUP_DIR/

# Compress backup
echo "Compressing backup..."
tar -czf $BACKUP_DIR.tar.gz -C $(dirname $BACKUP_DIR) $(basename $BACKUP_DIR)
rm -rf $BACKUP_DIR

echo "✅ Backup completed: $BACKUP_DIR.tar.gz"
```

## ✅ Deployment Checklist

### **Pre-Production**
- [ ] All tests passing (unit, integration, e2e)
- [ ] Performance benchmarks met
- [ ] Security audit completed
- [ ] Load testing completed
- [ ] Disaster recovery tested
- [ ] Documentation updated
- [ ] Team training completed

### **Production Deployment**
- [ ] Backup created
- [ ] Infrastructure provisioned
- [ ] SSL certificates installed
- [ ] DNS configured
- [ ] Services deployed
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Alerts enabled
- [ ] Smoke tests passing

### **Post-Deployment**
- [ ] Performance monitoring active
- [ ] Error tracking configured
- [ ] Log aggregation working
- [ ] Backup schedule verified
- [ ] Alert escalation tested
- [ ] Documentation updated
- [ ] Team notified

**Tiempo estimado**: 8-12 horas para deployment inicial  
**Responsable**: DevOps Engineer + Team Lead  
**Dependencias**: Todas las etapas de desarrollo completadas

---

**Siguiente**: [Monitoring Setup](./monitoring.md) para configuración detallada de monitoreo
