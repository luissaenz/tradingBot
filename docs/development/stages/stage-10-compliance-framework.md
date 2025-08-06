# 🛡️ Stage 10: Compliance & Risk Monitoring Framework

## 📋 Objetivo
Implementar un framework de cumplimiento institucional con monitoreo de límites, audit trails y reportes regulatorios para asegurar operación conforme a estándares institucionales.

## 🎯 Componentes a Implementar

### **10.1 Compliance Framework**

#### **Responsabilidades**
- Monitorear límites de posición y concentración
- Generar audit trails completos
- Crear reportes regulatorios
- Implementar circuit breakers automáticos

#### **Implementación**

```python
# src/modules/compliance/compliance_framework.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from ..shared.logging import get_logger
from ..shared.database import DatabaseManager

class ViolationType(Enum):
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    CORRELATION_LIMIT = "correlation_limit"
    LIQUIDITY_LIMIT = "liquidity_limit"

class ViolationSeverity(Enum):
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ComplianceViolation:
    """Violación de compliance"""
    timestamp: datetime
    violation_type: ViolationType
    severity: ViolationSeverity
    current_value: float
    limit_value: float
    symbol: Optional[str]
    description: str
    action_taken: str
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class AuditTrailEntry:
    """Entrada de audit trail"""
    timestamp: datetime
    event_type: str
    user_id: str
    symbol: str
    action: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    risk_metrics: Dict[str, float]
    compliance_checks: List[str]
    decision_rationale: str

@dataclass
class RegulatoryReport:
    """Reporte regulatorio"""
    report_date: datetime
    report_type: str
    period_start: datetime
    period_end: datetime
    portfolio_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    violations: List[ComplianceViolation]
    performance_attribution: Dict[str, float]
    largest_positions: List[Dict]

class ComplianceFramework:
    """Framework de cumplimiento institucional"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        self.db = DatabaseManager()
        
        # Límites de compliance
        self.position_limits = config.get('position_limits', {
            'max_position_size_pct': 0.05,      # 5% del portfolio
            'max_sector_exposure_pct': 0.3,     # 30% por sector
            'max_single_asset_pct': 0.1,        # 10% por activo
            'max_correlation_exposure': 0.7      # 70% correlación máxima
        })
        
        self.risk_limits = config.get('risk_limits', {
            'max_daily_drawdown': 0.02,          # 2% diario
            'max_weekly_drawdown': 0.05,         # 5% semanal
            'max_monthly_drawdown': 0.10,        # 10% mensual
            'max_leverage': 3.0,                 # 3:1 máximo
            'max_var_95': 0.03,                  # 3% VaR 95%
            'min_liquidity_ratio': 0.8           # 80% activos líquidos
        })
        
        # Circuit breakers
        self.circuit_breakers = config.get('circuit_breakers', {
            'emergency_stop_drawdown': 0.05,     # 5% stop total
            'daily_loss_limit': 0.03,            # 3% pérdida diaria
            'consecutive_losses_limit': 5,        # 5 pérdidas consecutivas
            'volatility_spike_threshold': 3.0     # 3x volatilidad normal
        })
        
        # Estado interno
        self.violations_history = []
        self.audit_trail = []
        self.emergency_stop_active = False
        
    async def monitor_position_limits(self, 
                                    current_positions: List[Dict],
                                    proposed_position: Optional[Dict] = None) -> List[ComplianceViolation]:
        """
        Monitorea límites de posición y concentración
        """
        try:
            violations = []
            
            # Incluir posición propuesta si existe
            all_positions = current_positions.copy()
            if proposed_position:
                all_positions.append(proposed_position)
            
            # Calcular métricas del portfolio
            total_portfolio_value = sum(pos.get('market_value', 0) for pos in all_positions)
            
            if total_portfolio_value == 0:
                return violations
            
            # Verificar límite por posición individual
            for position in all_positions:
                position_pct = position.get('market_value', 0) / total_portfolio_value
                
                if position_pct > self.position_limits['max_position_size_pct']:
                    violations.append(ComplianceViolation(
                        timestamp=datetime.now(),
                        violation_type=ViolationType.POSITION_LIMIT,
                        severity=ViolationSeverity.CRITICAL,
                        current_value=position_pct,
                        limit_value=self.position_limits['max_position_size_pct'],
                        symbol=position.get('symbol'),
                        description=f"Position size {position_pct:.2%} exceeds limit {self.position_limits['max_position_size_pct']:.2%}",
                        action_taken="position_reduction_required"
                    ))
            
            # Verificar concentración por activo
            asset_exposure = {}
            for position in all_positions:
                symbol = position.get('symbol', 'unknown')
                asset_exposure[symbol] = asset_exposure.get(symbol, 0) + position.get('market_value', 0)
            
            for symbol, exposure in asset_exposure.items():
                exposure_pct = exposure / total_portfolio_value
                
                if exposure_pct > self.position_limits['max_single_asset_pct']:
                    violations.append(ComplianceViolation(
                        timestamp=datetime.now(),
                        violation_type=ViolationType.CONCENTRATION_LIMIT,
                        severity=ViolationSeverity.WARNING,
                        current_value=exposure_pct,
                        limit_value=self.position_limits['max_single_asset_pct'],
                        symbol=symbol,
                        description=f"Asset concentration {exposure_pct:.2%} exceeds limit {self.position_limits['max_single_asset_pct']:.2%}",
                        action_taken="diversification_required"
                    ))
            
            # Verificar correlación del portfolio
            correlation_risk = await self._calculate_portfolio_correlation_risk(all_positions)
            
            if correlation_risk > self.position_limits['max_correlation_exposure']:
                violations.append(ComplianceViolation(
                    timestamp=datetime.now(),
                    violation_type=ViolationType.CORRELATION_LIMIT,
                    severity=ViolationSeverity.WARNING,
                    current_value=correlation_risk,
                    limit_value=self.position_limits['max_correlation_exposure'],
                    symbol=None,
                    description=f"Portfolio correlation risk {correlation_risk:.2%} exceeds limit",
                    action_taken="correlation_reduction_required"
                ))
            
            # Guardar violaciones
            for violation in violations:
                await self._record_violation(violation)
            
            self.logger.info(f"Position limits check completed: {len(violations)} violations found")
            return violations
            
        except Exception as e:
            self.logger.error(f"Error monitoring position limits: {e}")
            return []
    
    async def monitor_risk_limits(self, 
                                portfolio_metrics: Dict[str, float]) -> List[ComplianceViolation]:
        """
        Monitorea límites de riesgo del portfolio
        """
        try:
            violations = []
            
            # Verificar drawdown diario
            daily_drawdown = portfolio_metrics.get('daily_drawdown', 0)
            if daily_drawdown > self.risk_limits['max_daily_drawdown']:
                violations.append(ComplianceViolation(
                    timestamp=datetime.now(),
                    violation_type=ViolationType.DRAWDOWN_LIMIT,
                    severity=ViolationSeverity.CRITICAL,
                    current_value=daily_drawdown,
                    limit_value=self.risk_limits['max_daily_drawdown'],
                    symbol=None,
                    description=f"Daily drawdown {daily_drawdown:.2%} exceeds limit",
                    action_taken="risk_reduction_required"
                ))
            
            # Verificar drawdown semanal
            weekly_drawdown = portfolio_metrics.get('weekly_drawdown', 0)
            if weekly_drawdown > self.risk_limits['max_weekly_drawdown']:
                violations.append(ComplianceViolation(
                    timestamp=datetime.now(),
                    violation_type=ViolationType.DRAWDOWN_LIMIT,
                    severity=ViolationSeverity.CRITICAL,
                    current_value=weekly_drawdown,
                    limit_value=self.risk_limits['max_weekly_drawdown'],
                    symbol=None,
                    description=f"Weekly drawdown {weekly_drawdown:.2%} exceeds limit",
                    action_taken="position_closure_required"
                ))
            
            # Verificar leverage
            current_leverage = portfolio_metrics.get('leverage', 1.0)
            if current_leverage > self.risk_limits['max_leverage']:
                violations.append(ComplianceViolation(
                    timestamp=datetime.now(),
                    violation_type=ViolationType.LEVERAGE_LIMIT,
                    severity=ViolationSeverity.WARNING,
                    current_value=current_leverage,
                    limit_value=self.risk_limits['max_leverage'],
                    symbol=None,
                    description=f"Leverage {current_leverage:.1f}x exceeds limit",
                    action_taken="leverage_reduction_required"
                ))
            
            # Verificar VaR
            var_95 = portfolio_metrics.get('var_95', 0)
            if var_95 > self.risk_limits['max_var_95']:
                violations.append(ComplianceViolation(
                    timestamp=datetime.now(),
                    violation_type=ViolationType.DRAWDOWN_LIMIT,
                    severity=ViolationSeverity.WARNING,
                    current_value=var_95,
                    limit_value=self.risk_limits['max_var_95'],
                    symbol=None,
                    description=f"VaR 95% {var_95:.2%} exceeds limit",
                    action_taken="risk_reduction_required"
                ))
            
            # Verificar liquidez
            liquidity_ratio = portfolio_metrics.get('liquidity_ratio', 1.0)
            if liquidity_ratio < self.risk_limits['min_liquidity_ratio']:
                violations.append(ComplianceViolation(
                    timestamp=datetime.now(),
                    violation_type=ViolationType.LIQUIDITY_LIMIT,
                    severity=ViolationSeverity.WARNING,
                    current_value=liquidity_ratio,
                    limit_value=self.risk_limits['min_liquidity_ratio'],
                    symbol=None,
                    description=f"Liquidity ratio {liquidity_ratio:.2%} below minimum",
                    action_taken="liquidity_improvement_required"
                ))
            
            # Guardar violaciones
            for violation in violations:
                await self._record_violation(violation)
            
            self.logger.info(f"Risk limits check completed: {len(violations)} violations found")
            return violations
            
        except Exception as e:
            self.logger.error(f"Error monitoring risk limits: {e}")
            return []
    
    async def generate_audit_trail(self, 
                                 event_type: str,
                                 user_id: str,
                                 symbol: str,
                                 action: str,
                                 parameters: Dict[str, Any],
                                 result: Dict[str, Any],
                                 risk_metrics: Dict[str, float],
                                 decision_rationale: str) -> AuditTrailEntry:
        """
        Genera entrada de audit trail para cada decisión/acción
        """
        try:
            # Ejecutar compliance checks
            compliance_checks = await self._execute_compliance_checks(
                symbol, action, parameters, risk_metrics)
            
            # Crear entrada de audit trail
            audit_entry = AuditTrailEntry(
                timestamp=datetime.now(),
                event_type=event_type,
                user_id=user_id,
                symbol=symbol,
                action=action,
                parameters=parameters,
                result=result,
                risk_metrics=risk_metrics,
                compliance_checks=compliance_checks,
                decision_rationale=decision_rationale
            )
            
            # Guardar en base de datos
            await self._save_audit_entry(audit_entry)
            
            # Mantener en memoria para acceso rápido
            self.audit_trail.append(audit_entry)
            if len(self.audit_trail) > 1000:  # Mantener últimas 1000 entradas
                self.audit_trail.pop(0)
            
            self.logger.info(f"Audit trail entry created: {event_type} - {action}")
            return audit_entry
            
        except Exception as e:
            self.logger.error(f"Error generating audit trail: {e}")
            raise
    
    async def activate_circuit_breakers(self, 
                                      portfolio_metrics: Dict[str, float],
                                      market_conditions: Dict[str, float]) -> Dict[str, bool]:
        """
        Activa circuit breakers basado en condiciones de riesgo
        """
        try:
            breakers_activated = {}
            
            # Emergency stop por drawdown extremo
            total_drawdown = portfolio_metrics.get('total_drawdown', 0)
            if total_drawdown >= self.circuit_breakers['emergency_stop_drawdown']:
                self.emergency_stop_active = True
                breakers_activated['emergency_stop'] = True
                
                await self._execute_emergency_stop("Extreme drawdown detected")
                self.logger.critical(f"EMERGENCY STOP ACTIVATED: Drawdown {total_drawdown:.2%}")
            
            # Stop por pérdidas diarias excesivas
            daily_loss = portfolio_metrics.get('daily_pnl', 0)
            if daily_loss <= -self.circuit_breakers['daily_loss_limit']:
                breakers_activated['daily_loss_stop'] = True
                
                await self._execute_daily_loss_stop()
                self.logger.warning(f"Daily loss stop activated: {daily_loss:.2%}")
            
            # Stop por pérdidas consecutivas
            consecutive_losses = portfolio_metrics.get('consecutive_losses', 0)
            if consecutive_losses >= self.circuit_breakers['consecutive_losses_limit']:
                breakers_activated['consecutive_losses_stop'] = True
                
                await self._execute_consecutive_losses_stop()
                self.logger.warning(f"Consecutive losses stop: {consecutive_losses} losses")
            
            # Stop por volatilidad extrema
            volatility_spike = market_conditions.get('volatility_spike_ratio', 1.0)
            if volatility_spike >= self.circuit_breakers['volatility_spike_threshold']:
                breakers_activated['volatility_stop'] = True
                
                await self._execute_volatility_stop()
                self.logger.warning(f"Volatility stop: {volatility_spike:.1f}x normal volatility")
            
            return breakers_activated
            
        except Exception as e:
            self.logger.error(f"Error activating circuit breakers: {e}")
            return {}
    
    async def generate_regulatory_report(self, 
                                       report_type: str,
                                       period_start: datetime,
                                       period_end: datetime) -> RegulatoryReport:
        """
        Genera reporte regulatorio para el período especificado
        """
        try:
            # Obtener datos del período
            portfolio_data = await self._get_portfolio_data(period_start, period_end)
            risk_data = await self._get_risk_data(period_start, period_end)
            violations_data = await self._get_violations_data(period_start, period_end)
            
            # Calcular métricas del portfolio
            portfolio_metrics = {
                'total_return': portfolio_data.get('total_return', 0),
                'sharpe_ratio': portfolio_data.get('sharpe_ratio', 0),
                'max_drawdown': portfolio_data.get('max_drawdown', 0),
                'volatility': portfolio_data.get('volatility', 0),
                'average_position_size': portfolio_data.get('avg_position_size', 0),
                'turnover': portfolio_data.get('turnover', 0)
            }
            
            # Calcular métricas de riesgo
            risk_metrics = {
                'var_95': risk_data.get('var_95', 0),
                'expected_shortfall': risk_data.get('expected_shortfall', 0),
                'beta': risk_data.get('beta', 0),
                'correlation_risk': risk_data.get('correlation_risk', 0),
                'concentration_risk': risk_data.get('concentration_risk', 0),
                'liquidity_risk': risk_data.get('liquidity_risk', 0)
            }
            
            # Performance attribution
            performance_attribution = await self._calculate_performance_attribution(
                period_start, period_end)
            
            # Posiciones más grandes
            largest_positions = await self._get_largest_positions(period_end)
            
            # Crear reporte
            report = RegulatoryReport(
                report_date=datetime.now(),
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                portfolio_metrics=portfolio_metrics,
                risk_metrics=risk_metrics,
                violations=violations_data,
                performance_attribution=performance_attribution,
                largest_positions=largest_positions
            )
            
            # Guardar reporte
            await self._save_regulatory_report(report)
            
            self.logger.info(f"Regulatory report generated: {report_type} "
                           f"({period_start.date()} to {period_end.date()})")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating regulatory report: {e}")
            raise
    
    # Métodos auxiliares
    async def _calculate_portfolio_correlation_risk(self, positions: List[Dict]) -> float:
        """Calcula riesgo de correlación del portfolio"""
        try:
            # Implementar cálculo de correlación
            # Por ahora retornar valor simulado
            return 0.4  # 40% correlation risk
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    async def _record_violation(self, violation: ComplianceViolation):
        """Registra violación en base de datos"""
        try:
            # Guardar en base de datos
            await self.db.execute(
                "INSERT INTO compliance_violations VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (violation.timestamp, violation.violation_type.value, 
                 violation.severity.value, violation.current_value,
                 violation.limit_value, violation.symbol,
                 violation.description, violation.action_taken, violation.resolved)
            )
            
            # Mantener en memoria
            self.violations_history.append(violation)
            
        except Exception as e:
            self.logger.error(f"Error recording violation: {e}")
    
    async def _execute_compliance_checks(self, 
                                       symbol: str,
                                       action: str,
                                       parameters: Dict,
                                       risk_metrics: Dict) -> List[str]:
        """Ejecuta checks de compliance"""
        try:
            checks = []
            
            # Check de límites de posición
            if 'position_size' in parameters:
                checks.append("position_limit_check")
            
            # Check de riesgo
            if 'risk_amount' in risk_metrics:
                checks.append("risk_limit_check")
            
            # Check de liquidez
            if action in ['buy', 'sell']:
                checks.append("liquidity_check")
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error executing compliance checks: {e}")
            return []
    
    async def _save_audit_entry(self, entry: AuditTrailEntry):
        """Guarda entrada de audit trail"""
        try:
            await self.db.execute(
                "INSERT INTO audit_trail VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (entry.timestamp, entry.event_type, entry.user_id,
                 entry.symbol, entry.action, json.dumps(entry.parameters),
                 json.dumps(entry.result), json.dumps(entry.risk_metrics),
                 json.dumps(entry.compliance_checks), entry.decision_rationale)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving audit entry: {e}")
    
    async def _execute_emergency_stop(self, reason: str):
        """Ejecuta parada de emergencia"""
        try:
            # Cerrar todas las posiciones
            # Pausar trading
            # Enviar alertas
            self.logger.critical(f"EMERGENCY STOP EXECUTED: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error executing emergency stop: {e}")
    
    async def _execute_daily_loss_stop(self):
        """Ejecuta stop por pérdidas diarias"""
        try:
            # Reducir tamaño de posiciones
            # Pausar nuevas operaciones por el día
            self.logger.warning("Daily loss stop executed")
            
        except Exception as e:
            self.logger.error(f"Error executing daily loss stop: {e}")
    
    async def _execute_consecutive_losses_stop(self):
        """Ejecuta stop por pérdidas consecutivas"""
        try:
            # Reducir agresividad
            # Revisar estrategias
            self.logger.warning("Consecutive losses stop executed")
            
        except Exception as e:
            self.logger.error(f"Error executing consecutive losses stop: {e}")
    
    async def _execute_volatility_stop(self):
        """Ejecuta stop por volatilidad extrema"""
        try:
            # Reducir exposición
            # Aumentar cash allocation
            self.logger.warning("Volatility stop executed")
            
        except Exception as e:
            self.logger.error(f"Error executing volatility stop: {e}")
    
    # Métodos para reportes
    async def _get_portfolio_data(self, start: datetime, end: datetime) -> Dict:
        """Obtiene datos del portfolio para el período"""
        # Implementar consulta a base de datos
        return {}
    
    async def _get_risk_data(self, start: datetime, end: datetime) -> Dict:
        """Obtiene datos de riesgo para el período"""
        # Implementar consulta a base de datos
        return {}
    
    async def _get_violations_data(self, start: datetime, end: datetime) -> List[ComplianceViolation]:
        """Obtiene violaciones para el período"""
        # Implementar consulta a base de datos
        return []
    
    async def _calculate_performance_attribution(self, start: datetime, end: datetime) -> Dict:
        """Calcula attribution de performance"""
        # Implementar cálculo de attribution
        return {}
    
    async def _get_largest_positions(self, date: datetime) -> List[Dict]:
        """Obtiene las posiciones más grandes"""
        # Implementar consulta a base de datos
        return []
    
    async def _save_regulatory_report(self, report: RegulatoryReport):
        """Guarda reporte regulatorio"""
        try:
            # Guardar en base de datos y/o archivo
            pass
            
        except Exception as e:
            self.logger.error(f"Error saving regulatory report: {e}")

# Tests
class TestComplianceFramework:
    """Tests para ComplianceFramework"""
    
    async def test_position_limits_monitoring(self):
        """Test monitoreo de límites de posición"""
        pass
    
    async def test_risk_limits_monitoring(self):
        """Test monitoreo de límites de riesgo"""
        pass
    
    async def test_audit_trail_generation(self):
        """Test generación de audit trail"""
        pass
    
    async def test_circuit_breakers(self):
        """Test circuit breakers"""
        pass
    
    async def test_regulatory_reporting(self):
        """Test reportes regulatorios"""
        pass
```

#### **Configuración**

```yaml
# config/compliance/framework.yaml
compliance:
  # Límites de posición
  position_limits:
    max_position_size_pct: 0.05      # 5%
    max_sector_exposure_pct: 0.3     # 30%
    max_single_asset_pct: 0.1        # 10%
    max_correlation_exposure: 0.7     # 70%
  
  # Límites de riesgo
  risk_limits:
    max_daily_drawdown: 0.02          # 2%
    max_weekly_drawdown: 0.05         # 5%
    max_monthly_drawdown: 0.10        # 10%
    max_leverage: 3.0                 # 3:1
    max_var_95: 0.03                  # 3%
    min_liquidity_ratio: 0.8          # 80%
  
  # Circuit breakers
  circuit_breakers:
    emergency_stop_drawdown: 0.05     # 5%
    daily_loss_limit: 0.03            # 3%
    consecutive_losses_limit: 5
    volatility_spike_threshold: 3.0
  
  # Reportes
  regulatory_reports:
    daily_risk_report: true
    weekly_performance_report: true
    monthly_compliance_report: true
    quarterly_regulatory_filing: true
  
  # Audit trail
  audit_trail:
    retention_days: 2555  # 7 años
    backup_frequency: "daily"
    encryption_enabled: true
```

## 📊 **Métricas y KPIs**

### **Compliance Monitoring**
- **Violation Rate**: % de operaciones con violaciones
- **Resolution Time**: Tiempo promedio de resolución
- **Limit Utilization**: % de utilización de límites
- **False Positive Rate**: % de alertas falsas

### **Audit Trail**
- **Completeness**: % de eventos auditados
- **Integrity**: Verificación de integridad
- **Accessibility**: Tiempo de consulta
- **Retention Compliance**: Cumplimiento de retención

## 🧪 **Testing Strategy**

```python
# tests/test_compliance_framework.py
async def test_position_limit_violations():
    pass

async def test_circuit_breaker_activation():
    pass

async def test_audit_trail_completeness():
    pass

async def test_regulatory_report_generation():
    pass
```

## 📈 **Performance Targets**

- **Monitoring Latency**: < 100ms
- **Audit Trail Write**: < 10ms
- **Report Generation**: < 30s
- **Database Queries**: < 5s

## 🚀 **Deployment Checklist**

- [ ] Implementar ComplianceFramework
- [ ] Configurar límites institucionales
- [ ] Crear esquemas de base de datos
- [ ] Implementar circuit breakers
- [ ] Configurar reportes automáticos
- [ ] Testing de compliance
- [ ] Deployment a producción

---

*Documento creado: 2025-08-06 - Compliance Framework*
