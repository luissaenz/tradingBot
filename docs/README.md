# 🏆 BTC Trading Agent - Documentación del Proyecto

## 📋 Índice de Documentación

### 🚀 Guías de Desarrollo
- [**Development Roadmap**](./development/roadmap.md) - Roadmap completo del proyecto
- [**Environment Setup**](./development/environment-setup.md) - Configuración inicial del entorno
- [**Docker Infrastructure**](./development/docker-infrastructure.md) - Setup de infraestructura Docker
- [**Development Stages**](./development/stages/) - Documentación detallada por etapas

### 🏗️ Arquitectura
- [**System Architecture**](./architecture/system-overview.md) - Visión general del sistema
- [**Enhanced Modules Specification**](./architecture/enhanced-modules-specification.md) - Módulos institucionales actualizados
- [**Module Specifications**](./architecture/modules/) - Especificaciones por módulo
- [**Data Flow**](./architecture/data-flow.md) - Flujo de datos del sistema
- [**API Specifications**](./architecture/api-specs.md) - Especificaciones de APIs

### 🛠️ Operaciones
- [**Deployment Guide**](./operations/deployment.md) - Guía de deployment
- [**Monitoring Setup**](./operations/monitoring.md) - Configuración de monitoreo
- [**Troubleshooting**](./operations/troubleshooting.md) - Guía de resolución de problemas
- [**Backup & Recovery**](./operations/backup-recovery.md) - Procedimientos de backup

### 📊 Trading
- [**Strategy Documentation**](./trading/strategies/) - Documentación de estrategias
- [**Risk Management**](./trading/risk-management.md) - Gestión de riesgo
- [**Backtesting Guide**](./trading/backtesting.md) - Guía de backtesting
- [**Performance Metrics**](./trading/performance-metrics.md) - Métricas de performance

### 🧪 Testing
- [**Testing Strategy**](./testing/testing-strategy.md) - Estrategia de testing
- [**Test Data**](./testing/test-data.md) - Datos de prueba
- [**Performance Testing**](./testing/performance-testing.md) - Testing de performance

## 🎯 Objetivo del Proyecto

**BTC Trading Agent** es un sistema de trading algorítmico institucional diseñado para operar en BTCUSD utilizando:

- **Microstructure Analysis**: Order book imbalances y delta volume
- **Sentiment Analysis**: Análisis de sentiment en tiempo real de Twitter
- **Machine Learning**: LightGBM para generación de señales
- **Risk Management**: Gestión de riesgo avanzada con circuit breakers
- **Auto-Optimization**: Optimización automática basada en performance

## 🏗️ Arquitectura del Sistema - Grado Institucional

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │───▶│ Enhanced        │───▶│ Multi-Timeframe │
│  (Binance + X)  │    │ Microstructure  │    │ Signal Generator│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Institutional   │    │ Key Levels &    │    │ Dynamic Risk    │
│ Positioning     │    │ Volatility      │    │ Management      │
│ Analysis        │    │ Surface         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┬───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │     Storage     │
                    │ (MinIO + TSDB)  │
                    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Risk Manager    │    │ Trade Execution │    │ Compliance &    │
│ (Circuit Break) │    │   (Binance)     │    │ Advanced        │
└─────────────────┘    └─────────────────┘    │ Analytics       │
                                 └─────────────────┘
```

## 🚀 Quick Start

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd btc-trading-agent

# 2. Configurar environment
cp .env.example .env
# Editar .env con tus API keys

# 3. Iniciar infraestructura
docker-compose up -d

# 4. Verificar servicios
./scripts/health-check.sh

# 5. Iniciar trading (paper mode)
docker-compose exec signal-generation python -m src.main --paper-trading
```

## 📈 Métricas de Performance Objetivo

| Métrica | Target | Descripción |
|---------|--------|-------------|
| Win Rate | >65% | Porcentaje de trades ganadores |
| Sharpe Ratio | >1.5 | Ratio riesgo-ajustado |
| Max Drawdown | <5% | Máxima pérdida mensual |
| Latency | <100ms | Tiempo señal-a-ejecución |
| Uptime | >99.5% | Disponibilidad del sistema |

## 🛡️ Risk Management

- **Daily Drawdown Limit**: 2% del capital
- **Weekly Drawdown Limit**: 5% del capital
- **Position Size**: Máximo 10% por trade
- **Stop Loss**: 2x ATR dinámico
- **Kill Switch**: Parada automática en condiciones críticas

## 👥 Team Roles

- **DevOps Engineer**: Infraestructura y deployment
- **ML Engineer**: Modelos y optimización
- **Data Scientist**: Feature engineering y análisis
- **Backend Developer**: APIs y servicios
- **Trading Systems Developer**: Ejecución y risk management

## 📞 Soporte

Para preguntas o issues:
1. Revisar [Troubleshooting Guide](./operations/troubleshooting.md)
2. Consultar [FAQ](./faq.md)
3. Crear issue en GitHub
4. Contactar al team lead

---

**⚠️ Disclaimer**: Este sistema es para propósitos educativos y de investigación. El trading algorítmico involucra riesgos significativos. Usar bajo tu propia responsabilidad.
