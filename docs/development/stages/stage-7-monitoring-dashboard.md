# 📊 Stage 7: Monitoring & Dashboard

## 📋 CHECKLIST: Monitoring & Dashboard (12-16 horas)

### ✅ Prerrequisitos
- [ ] Trading Execution funcionando
- [ ] Prometheus y Grafana operativos
- [ ] Datos fluyendo a través del sistema
- [ ] InfluxDB recibiendo métricas

### ✅ Objetivos de la Etapa
Implementar monitoreo completo del sistema:
- **Real-time Dashboard**: Dashboard en tiempo real
- **Performance Metrics**: Métricas de performance
- **Alert System**: Sistema de alertas
- **Health Monitoring**: Monitoreo de salud del sistema
- **Trading Analytics**: Analytics de trading

## 🏗️ Arquitectura del Módulo

```
modules/monitoring/
├── __init__.py
├── main.py                    # Entry point del servicio
├── dashboards/
│   ├── __init__.py
│   ├── trading_dashboard.py  # Dashboard principal
│   ├── risk_dashboard.py     # Dashboard de riesgo
│   └── system_dashboard.py   # Dashboard del sistema
├── metrics/
│   ├── __init__.py
│   ├── metrics_collector.py  # Recolector de métricas
│   ├── performance_tracker.py # Tracker de performance
│   └── health_checker.py     # Health checker
├── alerts/
│   ├── __init__.py
│   ├── alert_manager.py      # Gestor de alertas
│   ├── notification_sender.py # Envío de notificaciones
│   └── alert_rules.py        # Reglas de alertas
└── config/
    └── monitoring_config.py  # Configuración
```

## 🚀 Implementación

### Metrics Collector
```python
# modules/monitoring/metrics/metrics_collector.py
import asyncio
import time
from typing import Dict, Any, List
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from shared.messaging.message_broker import HybridMessageBroker
from shared.database.connection import DatabaseManager
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    def __init__(self, message_broker: HybridMessageBroker, db_manager: DatabaseManager):
        self.message_broker = message_broker
        self.db_manager = db_manager
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        
        # Trading metrics
        self.signals_generated = Counter('signals_generated_total', 'Total signals generated', registry=self.registry)
        self.trades_executed = Counter('trades_executed_total', 'Total trades executed', registry=self.registry)
        self.execution_latency = Histogram('execution_latency_seconds', 'Trade execution latency', registry=self.registry)
        
        # Performance metrics
        self.current_pnl = Gauge('current_pnl', 'Current PnL', registry=self.registry)
        self.win_rate = Gauge('win_rate', 'Win rate percentage', registry=self.registry)
        self.sharpe_ratio = Gauge('sharpe_ratio', 'Sharpe ratio', registry=self.registry)
        
        # System metrics
        self.system_health = Gauge('system_health', 'System health score', registry=self.registry)
        self.data_latency = Histogram('data_latency_seconds', 'Data processing latency', registry=self.registry)
        
    async def start(self):
        """Start metrics collection"""
        try:
            # Subscribe to all relevant streams
            await self.message_broker.subscribe_fast(
                ['trading_signals', 'trade_executions', 'risk_decisions'],
                'monitoring',
                'collector-1',
                self._process_metric_event
            )
            
            # Start periodic metrics update
            asyncio.create_task(self._periodic_metrics_update())
            
            logger.info("Metrics collector started")
            
        except Exception as e:
            logger.error(f"Failed to start metrics collector: {e}")
            raise
            
    async def _process_metric_event(self, stream: str, message_id: str, data: Dict[str, Any]):
        """Process metric events"""
        try:
            if stream == 'trading_signals':
                self.signals_generated.inc()
                
            elif stream == 'trade_executions':
                self.trades_executed.inc()
                if 'execution_time' in data:
                    self.execution_latency.observe(data['execution_time'])
                    
            elif stream == 'risk_decisions':
                # Track risk decisions
                pass
                
        except Exception as e:
            logger.error(f"Error processing metric event: {e}")
            
    async def _periodic_metrics_update(self):
        """Update metrics periodically"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Update system health
                await self._update_system_health()
                
            except Exception as e:
                logger.error(f"Error in periodic metrics update: {e}")
                
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate current PnL
            pnl = await self._calculate_current_pnl()
            self.current_pnl.set(pnl)
            
            # Calculate win rate
            win_rate = await self._calculate_win_rate()
            self.win_rate.set(win_rate)
            
            # Calculate Sharpe ratio
            sharpe = await self._calculate_sharpe_ratio()
            self.sharpe_ratio.set(sharpe)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            
    async def _calculate_current_pnl(self) -> float:
        """Calculate current PnL"""
        try:
            query = """
                SELECT COALESCE(SUM(pnl), 0) as total_pnl
                FROM trading.trades 
                WHERE status = 'FILLED'
            """
            result = await self.db_manager.fetchrow(query)
            return float(result['total_pnl']) if result else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating PnL: {e}")
            return 0.0
```

### Trading Dashboard
```python
# modules/monitoring/dashboards/trading_dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from shared.database.connection import DatabaseManager

class TradingDashboard:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def render(self):
        """Render main trading dashboard"""
        st.set_page_config(
            page_title="BTC Trading Agent Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("🚀 BTC Trading Agent Dashboard")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_pnl_metric()
        with col2:
            self._render_win_rate_metric()
        with col3:
            self._render_trades_metric()
        with col4:
            self._render_signals_metric()
            
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_pnl_chart()
        with col2:
            self._render_signals_chart()
            
        # Recent trades table
        self._render_recent_trades()
        
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("Controls")
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["1 Hour", "6 Hours", "24 Hours", "7 Days"]
        )
        
        # Refresh button
        if st.sidebar.button("Refresh Data"):
            st.experimental_rerun()
            
        # System status
        st.sidebar.header("System Status")
        st.sidebar.success("✅ Data Ingestion")
        st.sidebar.success("✅ Signal Generation")
        st.sidebar.success("✅ Risk Management")
        st.sidebar.success("✅ Trade Execution")
        
    def _render_pnl_metric(self):
        """Render PnL metric"""
        pnl = asyncio.run(self._get_current_pnl())
        
        delta_color = "normal" if pnl >= 0 else "inverse"
        st.metric(
            label="Current PnL",
            value=f"${pnl:.2f}",
            delta=f"{pnl:.2f}",
            delta_color=delta_color
        )
        
    def _render_win_rate_metric(self):
        """Render win rate metric"""
        win_rate = asyncio.run(self._get_win_rate())
        
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"{win_rate - 50:.1f}%"
        )
        
    def _render_trades_metric(self):
        """Render trades count metric"""
        trades_count = asyncio.run(self._get_trades_count())
        
        st.metric(
            label="Total Trades",
            value=trades_count,
            delta="Today"
        )
        
    def _render_signals_metric(self):
        """Render signals count metric"""
        signals_count = asyncio.run(self._get_signals_count())
        
        st.metric(
            label="Signals Generated",
            value=signals_count,
            delta="Today"
        )
        
    async def _get_current_pnl(self) -> float:
        """Get current PnL"""
        try:
            query = "SELECT COALESCE(SUM(pnl), 0) as pnl FROM trading.trades WHERE status = 'FILLED'"
            result = await self.db_manager.fetchrow(query)
            return float(result['pnl']) if result else 0.0
        except:
            return 0.0
            
    async def _get_win_rate(self) -> float:
        """Get win rate"""
        try:
            query = """
                SELECT 
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate
                FROM trading.trades 
                WHERE status = 'FILLED' AND pnl IS NOT NULL
            """
            result = await self.db_manager.fetchrow(query)
            return float(result['win_rate']) if result else 0.0
        except:
            return 0.0
```

### Alert Manager
```python
# modules/monitoring/alerts/alert_manager.py
import asyncio
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from shared.messaging.message_broker import HybridMessageBroker
from shared.logging.structured_logger import get_logger
from alerts.notification_sender import NotificationSender

logger = get_logger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    timestamp: int
    severity: AlertSeverity
    title: str
    message: str
    source: str
    metrics: Dict[str, Any]

class AlertManager:
    def __init__(self, message_broker: HybridMessageBroker, config: Dict[str, Any]):
        self.message_broker = message_broker
        self.config = config
        self.notification_sender = NotificationSender(config.get('notifications', {}))
        
        # Alert rules
        self.alert_rules = {
            'high_drawdown': {'threshold': -0.05, 'severity': AlertSeverity.CRITICAL},
            'low_win_rate': {'threshold': 0.4, 'severity': AlertSeverity.WARNING},
            'execution_latency': {'threshold': 5.0, 'severity': AlertSeverity.WARNING},
            'system_error': {'threshold': 1, 'severity': AlertSeverity.CRITICAL}
        }
        
        # Active alerts
        self.active_alerts = {}
        
    async def start(self):
        """Start alert manager"""
        try:
            # Subscribe to system events
            await self.message_broker.subscribe_fast(
                ['system_metrics', 'error_events', 'performance_metrics'],
                'monitoring',
                'alerts-1',
                self._process_alert_event
            )
            
            logger.info("Alert manager started")
            
        except Exception as e:
            logger.error(f"Failed to start alert manager: {e}")
            raise
            
    async def _process_alert_event(self, stream: str, message_id: str, data: Dict[str, Any]):
        """Process potential alert events"""
        try:
            if stream == 'system_metrics':
                await self._check_system_alerts(data)
            elif stream == 'performance_metrics':
                await self._check_performance_alerts(data)
            elif stream == 'error_events':
                await self._handle_error_alert(data)
                
        except Exception as e:
            logger.error(f"Error processing alert event: {e}")
            
    async def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check performance-based alerts"""
        try:
            # Check drawdown
            if 'current_drawdown' in metrics:
                drawdown = metrics['current_drawdown']
                if drawdown < self.alert_rules['high_drawdown']['threshold']:
                    await self._trigger_alert(
                        'high_drawdown',
                        AlertSeverity.CRITICAL,
                        f"High Drawdown Alert",
                        f"Current drawdown: {drawdown:.2%}",
                        metrics
                    )
                    
            # Check win rate
            if 'win_rate' in metrics:
                win_rate = metrics['win_rate'] / 100
                if win_rate < self.alert_rules['low_win_rate']['threshold']:
                    await self._trigger_alert(
                        'low_win_rate',
                        AlertSeverity.WARNING,
                        f"Low Win Rate Alert",
                        f"Current win rate: {win_rate:.1%}",
                        metrics
                    )
                    
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
            
    async def _trigger_alert(self, alert_id: str, severity: AlertSeverity, 
                           title: str, message: str, metrics: Dict[str, Any]):
        """Trigger an alert"""
        try:
            import time
            
            alert = Alert(
                id=alert_id,
                timestamp=int(time.time() * 1000),
                severity=severity,
                title=title,
                message=message,
                source='trading_system',
                metrics=metrics
            )
            
            # Store active alert
            self.active_alerts[alert_id] = alert
            
            # Send notification
            await self.notification_sender.send_alert(alert)
            
            # Publish alert
            await self.message_broker.publish_fast('alerts', {
                'id': alert.id,
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'metrics': alert.metrics
            })
            
            logger.warning(f"Alert triggered: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
```

## ✅ Checklist de Completitud

### Dashboard Development
- [ ] Real-time trading dashboard
- [ ] Performance metrics display
- [ ] System health monitoring
- [ ] Interactive charts
- [ ] Mobile responsive design
- [ ] Auto-refresh functionality

### Metrics Collection
- [ ] Prometheus integration
- [ ] Custom metrics collectors
- [ ] Performance tracking
- [ ] System health metrics
- [ ] Real-time data streaming
- [ ] Historical data analysis

### Alert System
- [ ] Alert rule engine
- [ ] Notification system
- [ ] Email/Slack integration
- [ ] Alert escalation
- [ ] Alert acknowledgment
- [ ] Alert history

### Monitoring Infrastructure
- [ ] Grafana dashboards
- [ ] InfluxDB integration
- [ ] Log aggregation
- [ ] Error tracking
- [ ] Performance monitoring
- [ ] Uptime monitoring

**Tiempo estimado**: 12-16 horas  
**Responsable**: DevOps Engineer  
**Dependencias**: Trading Execution funcionando

---

**Next Step**: [Stage 8: Auto-Optimization](./stage-8-auto-optimization.md)
