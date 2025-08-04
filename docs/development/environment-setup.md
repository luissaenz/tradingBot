# 🛠️ Environment Setup Guide

## 📋 CHECKLIST: Environment Setup

### ✅ Prerrequisitos del Sistema

#### Sistema Operativo
- [ ] **Windows 10/11** con WSL2 habilitado
- [ ] **macOS 12+** con Homebrew instalado
- [ ] **Ubuntu 20.04+** o distribución compatible

#### Software Base
- [ ] **Docker Desktop** v20.10+ instalado y funcionando
- [ ] **Docker Compose** v2.0+ instalado
- [ ] **Git** v2.30+ configurado con SSH keys
- [ ] **Python 3.11+** instalado localmente
- [ ] **Node.js 18+** (para herramientas de desarrollo)

#### Editor de Código
- [ ] **Visual Studio Code** instalado
- [ ] Extensión **Docker** instalada
- [ ] Extensión **Python** instalada
- [ ] Extensión **Remote-Containers** instalada

#### Herramientas de Testing
- [ ] **Postman** o **Insomnia** para testing de APIs
- [ ] **DBeaver** o similar para gestión de bases de datos

### ✅ Configuración de Cuentas y APIs

#### Binance (Trading)
- [ ] Crear cuenta en [Binance](https://www.binance.com)
- [ ] Habilitar **Binance Testnet** en [testnet.binance.vision](https://testnet.binance.vision)
- [ ] Generar **API Keys** en testnet:
  - API Key
  - Secret Key
  - Habilitar **Spot Trading**
  - Habilitar **Futures Trading** (opcional)
- [ ] Verificar conectividad con API

#### Twitter (Sentiment Data)
- [ ] Crear cuenta **Twitter Developer** en [developer.twitter.com](https://developer.twitter.com)
- [ ] Aplicar para **Academic Research** access (gratuito)
- [ ] Generar **Bearer Token**
- [ ] Verificar límites de rate (300 requests/15min)

#### GitHub (Código)
- [ ] Repositorio creado (privado recomendado)
- [ ] SSH keys configuradas
- [ ] Colaboradores agregados al repositorio

### ✅ Configuración del Proyecto

#### Clonar Repositorio
```bash
# Clonar el repositorio
git clone git@github.com:your-username/btc-trading-agent.git
cd btc-trading-agent

# Verificar estructura
ls -la
```

#### Variables de Entorno
```bash
# Copiar template de environment
cp .env.example .env

# Editar variables (usar tu editor preferido)
nano .env
```

**Contenido de .env:**
```bash
# Trading APIs
BINANCE_API_KEY=your_binance_testnet_api_key
BINANCE_SECRET_KEY=your_binance_testnet_secret_key
BINANCE_TESTNET=true

# Social Data
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Database Passwords
POSTGRES_PASSWORD=secure_postgres_password
REDIS_PASSWORD=secure_redis_password
INFLUX_PASSWORD=secure_influx_password

# Object Storage
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=secure_minio_password

# Monitoring
GRAFANA_ADMIN_PASSWORD=secure_grafana_password

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

#### Verificar Docker
```bash
# Verificar Docker está funcionando
docker --version
docker-compose --version

# Test básico
docker run hello-world

# Verificar recursos disponibles
docker system df
```

### ✅ Estructura de Proyecto

#### Crear Directorios Base
```bash
# Crear estructura de directorios
mkdir -p {docs,infrastructure,modules,shared,data,tests,scripts}
mkdir -p docs/{development,architecture,operations,trading,testing}
mkdir -p infrastructure/{docker,configs,scripts}
mkdir -p modules/{data-ingestion,feature-engineering,signal-generation,risk-manager,trading-execution,monitoring}
mkdir -p shared/{messaging,database,config,logging,metrics}
mkdir -p data/{raw,processed,models,backtest}
mkdir -p tests/{unit,integration,e2e}

# Verificar estructura
tree -L 3
```

#### Archivos de Configuración Base
```bash
# Crear .gitignore
cat > .gitignore << 'EOF'
# Environment
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
data/models/*
!data/models/.gitkeep

# Logs
logs/
*.log

# Docker
.docker/

# IDE
.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db
EOF

# Crear archivos .gitkeep
touch data/{raw,processed,models,backtest}/.gitkeep
touch logs/.gitkeep
```

### ✅ Validación del Entorno

#### Test de Conectividad APIs
```bash
# Crear script de validación
cat > scripts/validate-apis.py << 'EOF'
#!/usr/bin/env python3
import os
import requests
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

def test_binance():
    try:
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        client = Client(api_key, secret_key, testnet=True)
        info = client.get_account()
        print("✅ Binance API: Connected successfully")
        return True
    except Exception as e:
        print(f"❌ Binance API: {e}")
        return False

def test_twitter():
    try:
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        headers = {'Authorization': f'Bearer {bearer_token}'}
        
        response = requests.get(
            'https://api.twitter.com/2/tweets/search/recent?query=bitcoin',
            headers=headers
        )
        
        if response.status_code == 200:
            print("✅ Twitter API: Connected successfully")
            return True
        else:
            print(f"❌ Twitter API: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Twitter API: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Validating API connections...")
    binance_ok = test_binance()
    twitter_ok = test_twitter()
    
    if binance_ok and twitter_ok:
        print("\n🎉 All APIs validated successfully!")
    else:
        print("\n⚠️  Some APIs failed validation. Check your credentials.")
EOF

# Ejecutar validación
python scripts/validate-apis.py
```

#### Test de Puertos
```bash
# Verificar puertos disponibles
cat > scripts/check-ports.sh << 'EOF'
#!/bin/bash

PORTS=(5432 6379 8086 9000 9001 9092 2181 8500 9090 3000 8000 8001 8002 8003 8004 8005)

echo "🔍 Checking required ports..."

for port in "${PORTS[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "❌ Port $port is already in use"
    else
        echo "✅ Port $port is available"
    fi
done
EOF

chmod +x scripts/check-ports.sh
./scripts/check-ports.sh
```

### ✅ Instalación de Dependencias

#### Python Dependencies
```bash
# Crear requirements.txt base
cat > requirements.txt << 'EOF'
# Core
python-dotenv==1.0.0
pydantic==2.5.2
fastapi==0.104.1
uvicorn==0.24.0

# Trading
python-binance==1.0.19
ccxt==4.1.74

# Data Processing
pandas==2.1.4
numpy==1.24.3
redis==5.0.1

# ML
lightgbm==4.1.0
scikit-learn==1.3.2

# Social Data
tweepy==4.14.0
transformers==4.36.2

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.25

# Monitoring
prometheus-client==0.19.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# Development
black==23.11.0
flake8==6.1.0
mypy==1.7.1
EOF

# Crear virtual environment
python -m venv venv

# Activar virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

#### Docker Dependencies
```bash
# Crear docker-compose.yml básico para testing
cat > docker-compose.test.yml << 'EOF'
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    
  postgres:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: trading_agent
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      
networks:
  default:
    name: trading-test-network
EOF

# Test Docker setup
docker-compose -f docker-compose.test.yml up -d
sleep 10
docker-compose -f docker-compose.test.yml ps
docker-compose -f docker-compose.test.yml down
```

### ✅ Configuración de IDE

#### Visual Studio Code
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    },
    "docker.showStartPage": false
}
```

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Docker: Attach to Container",
            "type": "docker",
            "request": "attach",
            "platform": "python",
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/app"
                }
            ]
        }
    ]
}
```

### ✅ Validación Final

#### Health Check Script
```bash
# Crear script de health check completo
cat > scripts/health-check.sh << 'EOF'
#!/bin/bash

echo "🏥 BTC Trading Agent - Health Check"
echo "=================================="

# Check Python
echo -n "Python 3.11+: "
python --version | grep -q "3.1[1-9]" && echo "✅" || echo "❌"

# Check Docker
echo -n "Docker: "
docker --version >/dev/null 2>&1 && echo "✅" || echo "❌"

# Check Git
echo -n "Git: "
git --version >/dev/null 2>&1 && echo "✅" || echo "❌"

# Check Environment File
echo -n ".env file: "
[ -f .env ] && echo "✅" || echo "❌"

# Check API Keys
echo -n "Binance API Key: "
grep -q "BINANCE_API_KEY=" .env && echo "✅" || echo "❌"

echo -n "Twitter Bearer Token: "
grep -q "TWITTER_BEARER_TOKEN=" .env && echo "✅" || echo "❌"

# Check Directory Structure
echo -n "Project Structure: "
[ -d "modules" ] && [ -d "shared" ] && [ -d "docs" ] && echo "✅" || echo "❌"

# Check Python Dependencies
echo -n "Python Dependencies: "
pip list | grep -q "pandas\|lightgbm\|fastapi" && echo "✅" || echo "❌"

echo ""
echo "🎯 Environment setup validation complete!"
echo "Next step: Review docker-infrastructure.md"
EOF

chmod +x scripts/health-check.sh
./scripts/health-check.sh
```

## 🚨 Troubleshooting

### Problemas Comunes

#### Docker no inicia
```bash
# Windows: Reiniciar Docker Desktop
# macOS: Reiniciar Docker Desktop
# Linux: Reiniciar servicio
sudo systemctl restart docker
```

#### Puertos ocupados
```bash
# Encontrar proceso usando puerto
lsof -i :6379
# Matar proceso si es necesario
kill -9 <PID>
```

#### APIs no conectan
- Verificar API keys en .env
- Verificar conectividad a internet
- Verificar límites de rate en APIs

#### Python dependencies fallan
```bash
# Actualizar pip
pip install --upgrade pip
# Instalar dependencias del sistema (Ubuntu)
sudo apt-get install python3-dev libpq-dev
```

## ✅ Checklist Final

- [ ] Todos los prerrequisitos instalados
- [ ] APIs configuradas y validadas
- [ ] Estructura de proyecto creada
- [ ] Variables de entorno configuradas
- [ ] Docker funcionando correctamente
- [ ] Python environment configurado
- [ ] IDE configurado
- [ ] Health check pasando

**Tiempo estimado**: 2-4 horas  
**Responsable**: DevOps Engineer + Team Lead

---

**Next Step**: Una vez completado este setup, proceder con [Docker Infrastructure Setup](./docker-infrastructure.md)
