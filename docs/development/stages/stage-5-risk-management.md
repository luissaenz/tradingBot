# 🛡️ Stage 5: Risk Management Module

## 📋 CHECKLIST: Risk Management (16-20 horas)

### ✅ Prerrequisitos
- [ ] Signal Generation funcionando y emitiendo señales
- [ ] Trading signals fluyendo en tiempo real
- [ ] Database con posiciones y trades
- [ ] Configuración de límites de riesgo definida

### ✅ Objetivos de la Etapa
Implementar el sistema de gestión de riesgo:
- **Position Sizing**: Cálculo dinámico de tamaño de posición
- **Drawdown Control**: Monitoreo y límites de drawdown
- **Stop Loss Management**: Stop loss dinámico basado en volatilidad
- **Circuit Breakers**: Paradas automáticas en condiciones críticas
- **Risk Metrics**: Cálculo de métricas de riesgo en tiempo real

## 🏗️ Arquitectura del Módulo

```
modules/risk-manager/
├── __init__.py
├── main.py                    # Entry point del servicio
├── core/
│   ├── __init__.py
│   ├── risk_engine.py        # Motor principal de riesgo
│   ├── position_sizer.py     # Cálculo de tamaño de posición
│   ├── drawdown_monitor.py   # Monitor de drawdown
│   └── circuit_breaker.py    # Circuit breakers
├── stops/
│   ├── __init__.py
│   ├── stop_loss_manager.py  # Gestión de stop loss
│   ├── trailing_stop.py      # Trailing stops
│   └── volatility_stops.py   # Stops basados en volatilidad
├── metrics/
│   ├── __init__.py
│   ├── risk_calculator.py    # Cálculo de métricas
│   ├── var_calculator.py     # Value at Risk
│   └── performance_tracker.py # Tracking de performance
├── validators/
│   ├── __init__.py
│   ├── signal_validator.py   # Validación de señales
│   ├── position_validator.py # Validación de posiciones
│   └── trade_validator.py    # Validación de trades
└── config/
    └── risk_config.py        # Configuración de riesgo
```

## 🚀 Implementación Detallada

### Risk Engine Core
```python
# modules/risk-manager/core/risk_engine.py
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from shared.messaging.message_broker import HybridMessageBroker
from shared.database.connection import DatabaseManager
from shared.logging.structured_logger import get_logger
from core.position_sizer import PositionSizer
from core.drawdown_monitor import DrawdownMonitor
from core.circuit_breaker import CircuitBreaker
from stops.stop_loss_manager import StopLossManager

logger = get_logger(__name__)

class RiskDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class RiskAssessment:
    decision: RiskDecision
    approved_size: float
    risk_score: float
    stop_loss_price: Optional[float]
    reasons: List[str]
    metrics: Dict[str, float]

@dataclass
class TradingSignal:
    timestamp: int
    symbol: str
    signal: int  # 0: sell, 1: hold, 2: buy
    confidence: float
    signal_strength: str

class RiskEngine:
    def __init__(self, message_broker: HybridMessageBroker, db_manager: DatabaseManager, config: Dict[str, Any]):
        self.message_broker = message_broker
        self.db_manager = db_manager
        self.config = config
        
        # Initialize components
        self.position_sizer = PositionSizer(config.get('position_sizing', {}))
        self.drawdown_monitor = DrawdownMonitor(config.get('drawdown_limits', {}))
        self.circuit_breaker = CircuitBreaker(config.get('circuit_breaker', {}))
        self.stop_loss_manager = StopLossManager(config.get('stop_loss', {}))
        
        # Risk state
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.max_drawdown = 0.0
        self.risk_metrics = {}
        
        # Emergency state
        self.emergency_stop_active = False
        self.last_risk_check = time.time()
        
    async def start(self):
        """Start risk management service"""
        try:
            # Subscribe to trading signals
            await self.message_broker.subscribe_fast(
                ['trading_signals'],
                'risk-manager',
                'engine-1',
                self._process_trading_signal
            )
            
            # Subscribe to position updates
            await self.message_broker.subscribe_fast(
                ['position_updates', 'trade_executions'],
                'risk-manager',
                'engine-1',
                self._process_position_update
            )
            
            # Start risk monitoring loop
            asyncio.create_task(self._risk_monitoring_loop())
            
            # Load current positions
            await self._load_current_positions()
            
            logger.info("Risk engine started")
            
        except Exception as e:
            logger.error(f"Failed to start risk engine: {e}")
            raise
            
    async def _process_trading_signal(self, stream: str, message_id: str, signal_data: Dict[str, Any]):
        """Process incoming trading signal"""
        try:
            signal = TradingSignal(
                timestamp=signal_data['timestamp'],
                symbol=signal_data['symbol'],
                signal=signal_data['signal'],
                confidence=signal_data['confidence'],
                signal_strength=signal_data['signal_strength']
            )
            
            # Assess risk for this signal
            risk_assessment = await self._assess_signal_risk(signal)
            
            # Publish risk decision
            await self._publish_risk_decision(signal, risk_assessment)
            
            logger.info(f"Risk assessment for {signal.symbol}: {risk_assessment.decision.value} "
                       f"(size: {risk_assessment.approved_size}, score: {risk_assessment.risk_score:.3f})")
            
        except Exception as e:
            logger.error(f"Error processing trading signal: {e}")
            
    async def _assess_signal_risk(self, signal: TradingSignal) -> RiskAssessment:
        """Comprehensive risk assessment for trading signal"""
        try:
            reasons = []
            risk_score = 0.0
            
            # Check emergency stop
            if self.emergency_stop_active:
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    approved_size=0.0,
                    risk_score=1.0,
                    stop_loss_price=None,
                    reasons=["Emergency stop active"],
                    metrics={}
                )
                
            # Check circuit breaker
            if self.circuit_breaker.is_triggered():
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    approved_size=0.0,
                    risk_score=1.0,
                    stop_loss_price=None,
                    reasons=["Circuit breaker triggered"],
                    metrics={}
                )
                
            # Check drawdown limits
            drawdown_check = await self.drawdown_monitor.check_limits(
                self.daily_pnl, self.weekly_pnl, self.max_drawdown
            )
            
            if not drawdown_check['allowed']:
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    approved_size=0.0,
                    risk_score=0.9,
                    stop_loss_price=None,
                    reasons=drawdown_check['reasons'],
                    metrics=drawdown_check['metrics']
                )
                
            # Calculate position size
            current_price = await self._get_current_price(signal.symbol)
            if current_price is None:
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    approved_size=0.0,
                    risk_score=0.8,
                    stop_loss_price=None,
                    reasons=["Cannot determine current price"],
                    metrics={}
                )
                
            position_size = await self.position_sizer.calculate_size(
                signal=signal,
                current_price=current_price,
                account_balance=await self._get_account_balance(),
                current_positions=self.current_positions
            )
            
            if position_size <= 0:
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    approved_size=0.0,
                    risk_score=0.7,
                    stop_loss_price=None,
                    reasons=["Position size too small or risk too high"],
                    metrics={}
                )
                
            # Calculate stop loss
            stop_loss_price = await self.stop_loss_manager.calculate_stop_loss(
                signal=signal,
                entry_price=current_price,
                position_size=position_size
            )
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(signal, position_size, current_price)
            
            # Make final decision
            if risk_score > 0.8:
                decision = RiskDecision.REJECT
                reasons.append("Risk score too high")
            elif risk_score > 0.6:
                decision = RiskDecision.MODIFY
                position_size *= 0.5  # Reduce position size
                reasons.append("Risk score elevated - reducing position size")
            else:
                decision = RiskDecision.APPROVE
                reasons.append("Risk assessment passed")
                
            return RiskAssessment(
                decision=decision,
                approved_size=position_size,
                risk_score=risk_score,
                stop_loss_price=stop_loss_price,
                reasons=reasons,
                metrics={
                    'current_price': current_price,
                    'position_value': position_size * current_price,
                    'daily_pnl': self.daily_pnl,
                    'max_drawdown': self.max_drawdown
                }
            )
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return RiskAssessment(
                decision=RiskDecision.REJECT,
                approved_size=0.0,
                risk_score=1.0,
                stop_loss_price=None,
                reasons=[f"Risk assessment error: {str(e)}"],
                metrics={}
            )
            
    def _calculate_risk_score(self, signal: TradingSignal, position_size: float, current_price: float) -> float:
        """Calculate overall risk score (0-1, higher = riskier)"""
        try:
            risk_factors = []
            
            # Signal confidence factor (lower confidence = higher risk)
            confidence_risk = 1.0 - signal.confidence
            risk_factors.append(confidence_risk * 0.3)
            
            # Position size factor
            position_value = position_size * current_price
            account_balance = asyncio.run(self._get_account_balance())
            
            if account_balance > 0:
                position_ratio = position_value / account_balance
                size_risk = min(position_ratio * 2, 1.0)  # Cap at 1.0
                risk_factors.append(size_risk * 0.25)
                
            # Drawdown factor
            if self.max_drawdown > 0:
                drawdown_risk = min(abs(self.max_drawdown) / 0.05, 1.0)  # 5% max drawdown
                risk_factors.append(drawdown_risk * 0.25)
                
            # Market volatility factor (placeholder - would use actual volatility)
            volatility_risk = 0.2  # Default moderate volatility
            risk_factors.append(volatility_risk * 0.2)
            
            return sum(risk_factors)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.8  # Conservative default
            
    async def _publish_risk_decision(self, signal: TradingSignal, assessment: RiskAssessment):
        """Publish risk decision"""
        try:
            decision_data = {
                'timestamp': int(time.time() * 1000),
                'signal_timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'original_signal': signal.signal,
                'decision': assessment.decision.value,
                'approved_size': assessment.approved_size,
                'risk_score': assessment.risk_score,
                'stop_loss_price': assessment.stop_loss_price,
                'reasons': assessment.reasons,
                'metrics': assessment.metrics
            }
            
            # Publish to execution module
            await self.message_broker.publish_fast('risk_decisions', decision_data)
            
            # Also store in Kafka for audit
            await self.message_broker.publish_reliable('risk_decisions', decision_data)
            
        except Exception as e:
            logger.error(f"Error publishing risk decision: {e}")
            
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Check for emergency conditions
                await self._check_emergency_conditions()
                
                # Update circuit breaker
                self.circuit_breaker.update_metrics(self.risk_metrics)
                
                self.last_risk_check = time.time()
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                
    async def _update_risk_metrics(self):
        """Update current risk metrics"""
        try:
            # Calculate current PnL
            self.daily_pnl = await self._calculate_daily_pnl()
            self.weekly_pnl = await self._calculate_weekly_pnl()
            
            # Update max drawdown
            current_drawdown = await self._calculate_current_drawdown()
            self.max_drawdown = min(self.max_drawdown, current_drawdown)
            
            # Update risk metrics
            self.risk_metrics = {
                'daily_pnl': self.daily_pnl,
                'weekly_pnl': self.weekly_pnl,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': current_drawdown,
                'position_count': len(self.current_positions),
                'total_exposure': sum(pos.get('value', 0) for pos in self.current_positions.values()),
                'last_update': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            
    async def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        try:
            emergency_triggers = []
            
            # Check daily drawdown
            if self.daily_pnl < -0.02 * await self._get_account_balance():  # -2% daily
                emergency_triggers.append("Daily drawdown exceeded -2%")
                
            # Check max drawdown
            if self.max_drawdown < -0.05 * await self._get_account_balance():  # -5% max
                emergency_triggers.append("Maximum drawdown exceeded -5%")
                
            # Check system health
            if time.time() - self.last_risk_check > 60:  # No updates for 1 minute
                emergency_triggers.append("Risk monitoring system unresponsive")
                
            if emergency_triggers and not self.emergency_stop_active:
                await self._trigger_emergency_stop(emergency_triggers)
                
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
            
    async def _trigger_emergency_stop(self, reasons: List[str]):
        """Trigger emergency stop"""
        try:
            self.emergency_stop_active = True
            
            emergency_data = {
                'timestamp': int(time.time() * 1000),
                'reasons': reasons,
                'current_metrics': self.risk_metrics
            }
            
            # Publish emergency stop
            await self.message_broker.publish_fast('emergency_stop', emergency_data)
            await self.message_broker.publish_reliable('emergency_stop', emergency_data)
            
            logger.critical(f"EMERGENCY STOP TRIGGERED: {', '.join(reasons)}")
            
        except Exception as e:
            logger.error(f"Error triggering emergency stop: {e}")
            
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            query = """
                SELECT price FROM trading.market_data 
                WHERE symbol = $1 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            result = await self.db_manager.fetchrow(query, symbol)
            return float(result['price']) if result else None
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
            
    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            # This would integrate with actual broker API
            # For now, return a placeholder
            return 10000.0  # $10,000 default
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0
            
    async def _calculate_daily_pnl(self) -> float:
        """Calculate daily PnL"""
        try:
            from datetime import datetime, timedelta
            
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            timestamp_start = int(today_start.timestamp() * 1000)
            
            query = """
                SELECT COALESCE(SUM(pnl), 0) as daily_pnl
                FROM trading.trades 
                WHERE timestamp >= $1 AND status = 'FILLED'
            """
            
            result = await self.db_manager.fetchrow(query, timestamp_start)
            return float(result['daily_pnl']) if result else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
            
    async def _calculate_weekly_pnl(self) -> float:
        """Calculate weekly PnL"""
        try:
            from datetime import datetime, timedelta
            
            week_start = datetime.now() - timedelta(days=7)
            timestamp_start = int(week_start.timestamp() * 1000)
            
            query = """
                SELECT COALESCE(SUM(pnl), 0) as weekly_pnl
                FROM trading.trades 
                WHERE timestamp >= $1 AND status = 'FILLED'
            """
            
            result = await self.db_manager.fetchrow(query, timestamp_start)
            return float(result['weekly_pnl']) if result else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating weekly PnL: {e}")
            return 0.0
            
    async def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        try:
            # Get account balance history and calculate drawdown
            # This is a simplified version
            current_balance = await self._get_account_balance()
            peak_balance = current_balance  # Would track actual peak
            
            if peak_balance > 0:
                return (current_balance - peak_balance) / peak_balance
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
            
    async def _load_current_positions(self):
        """Load current open positions"""
        try:
            query = """
                SELECT symbol, side, quantity, entry_price, timestamp
                FROM trading.positions 
                WHERE status = 'OPEN'
            """
            
            positions = await self.db_manager.fetch(query)
            
            for pos in positions:
                self.current_positions[pos['symbol']] = {
                    'side': pos['side'],
                    'quantity': float(pos['quantity']),
                    'entry_price': float(pos['entry_price']),
                    'timestamp': pos['timestamp'],
                    'value': float(pos['quantity']) * float(pos['entry_price'])
                }
                
            logger.info(f"Loaded {len(self.current_positions)} open positions")
            
        except Exception as e:
            logger.error(f"Error loading current positions: {e}")
```

### Position Sizer
```python
# modules/risk-manager/core/position_sizer.py
import math
from typing import Dict, Any, Optional
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class PositionSizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Position sizing parameters
        self.max_position_pct = config.get('max_position_pct', 0.10)  # 10% max per position
        self.max_total_exposure = config.get('max_total_exposure', 0.80)  # 80% max total
        self.min_position_value = config.get('min_position_value', 100)  # $100 minimum
        self.risk_per_trade = config.get('risk_per_trade', 0.01)  # 1% risk per trade
        
    async def calculate_size(self, 
                           signal: Any,
                           current_price: float,
                           account_balance: float,
                           current_positions: Dict[str, Any]) -> float:
        """Calculate optimal position size"""
        try:
            if account_balance <= 0 or current_price <= 0:
                return 0.0
                
            # Calculate base position size based on account percentage
            base_size = (account_balance * self.max_position_pct) / current_price
            
            # Adjust based on signal confidence
            confidence_multiplier = self._get_confidence_multiplier(signal.confidence)
            adjusted_size = base_size * confidence_multiplier
            
            # Check total exposure limit
            current_exposure = sum(pos.get('value', 0) for pos in current_positions.values())
            max_new_exposure = account_balance * self.max_total_exposure - current_exposure
            
            if max_new_exposure <= 0:
                logger.warning("Maximum total exposure reached")
                return 0.0
                
            max_size_by_exposure = max_new_exposure / current_price
            adjusted_size = min(adjusted_size, max_size_by_exposure)
            
            # Apply risk-based sizing (Kelly Criterion simplified)
            risk_adjusted_size = self._apply_risk_sizing(
                adjusted_size, signal, current_price, account_balance
            )
            
            # Ensure minimum position value
            min_size = self.min_position_value / current_price
            if risk_adjusted_size < min_size:
                return 0.0
                
            # Round to reasonable precision
            return round(risk_adjusted_size, 6)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
            
    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Get position size multiplier based on signal confidence"""
        if confidence >= 0.9:
            return 1.0
        elif confidence >= 0.8:
            return 0.8
        elif confidence >= 0.7:
            return 0.6
        elif confidence >= 0.6:
            return 0.4
        else:
            return 0.2
            
    def _apply_risk_sizing(self, base_size: float, signal: Any, 
                          current_price: float, account_balance: float) -> float:
        """Apply risk-based position sizing"""
        try:
            # Calculate risk amount (1% of account)
            risk_amount = account_balance * self.risk_per_trade
            
            # Estimate stop loss distance (2% for now, would be dynamic)
            stop_loss_pct = 0.02
            stop_loss_distance = current_price * stop_loss_pct
            
            # Calculate position size based on risk
            if stop_loss_distance > 0:
                risk_based_size = risk_amount / stop_loss_distance
                return min(base_size, risk_based_size)
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error applying risk sizing: {e}")
            return base_size * 0.5  # Conservative fallback
```

## ✅ Testing y Validación

### Risk Management Tests
```python
# tests/unit/test_risk_management.py
import pytest
from unittest.mock import Mock, AsyncMock
from modules.risk_manager.core.risk_engine import RiskEngine, TradingSignal, RiskDecision

@pytest.mark.asyncio
async def test_risk_assessment():
    """Test risk assessment logic"""
    # Mock dependencies
    message_broker = Mock()
    db_manager = Mock()
    db_manager.fetchrow = AsyncMock(return_value={'price': 50000.0})
    
    config = {
        'position_sizing': {'max_position_pct': 0.1},
        'drawdown_limits': {'daily_max': 0.02},
        'circuit_breaker': {'enabled': True},
        'stop_loss': {'default_pct': 0.02}
    }
    
    risk_engine = RiskEngine(message_broker, db_manager, config)
    
    # Test signal
    signal = TradingSignal(
        timestamp=1234567890,
        symbol='BTCUSDT',
        signal=2,  # Buy
        confidence=0.75,
        signal_strength='medium'
    )
    
    # Mock account balance
    risk_engine._get_account_balance = AsyncMock(return_value=10000.0)
    
    assessment = await risk_engine._assess_signal_risk(signal)
    
    assert assessment.decision in [RiskDecision.APPROVE, RiskDecision.MODIFY]
    assert assessment.approved_size > 0
    assert 0 <= assessment.risk_score <= 1
```

## ✅ Checklist de Completitud

### Core Risk Engine
- [ ] Risk assessment pipeline
- [ ] Signal validation
- [ ] Position size calculation
- [ ] Stop loss management
- [ ] Emergency stop system
- [ ] Real-time monitoring

### Risk Controls
- [ ] Drawdown monitoring
- [ ] Circuit breakers
- [ ] Position limits
- [ ] Exposure limits
- [ ] Volatility-based stops
- [ ] Time-based limits

### Metrics & Monitoring
- [ ] Real-time risk metrics
- [ ] Performance tracking
- [ ] Risk reporting
- [ ] Alert system
- [ ] Audit logging
- [ ] Dashboard integration

**Tiempo estimado**: 16-20 horas  
**Responsable**: Trading Systems Developer  
**Dependencias**: Signal Generation funcionando

---

**Next Step**: [Stage 6: Trading Execution](./stage-6-trading-execution.md)
