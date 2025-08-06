# ⚖️ Stage 5: Dynamic Risk Management

## 📋 Objetivo
Implementar un sistema de gestión de riesgo institucional avanzado con ratios dinámicos, position sizing óptimo y monitoreo de calidad de ejecución.

## 🎯 Componentes a Implementar

### **5.1 Dynamic Risk Manager**

#### **Responsabilidades**
- Calcular ratios risk/reward dinámicos basados en condiciones de mercado
- Implementar Kelly Criterion para position sizing óptimo
- Gestionar heat del portfolio y límites de correlación
- Monitorear calidad de ejecución y slippage

#### **Implementación**

```python
# src/modules/trading/dynamic_risk_manager.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from ..shared.logging import get_logger

class MarketCondition(Enum):
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    NEWS_EVENT = "news_event"
    EXTREME_STRESS = "extreme_stress"

@dataclass
class RiskParameters:
    base_risk_reward_ratio: float
    adjusted_risk_reward_ratio: float
    position_size_pct: float
    max_portfolio_heat: float
    stop_loss_distance: float
    take_profit_distance: float
    market_condition: MarketCondition
    confidence_level: float
    expected_slippage: float

@dataclass
class PortfolioRisk:
    total_portfolio_value: float
    total_risk_amount: float
    portfolio_heat: float
    max_drawdown: float
    var_95: float
    correlation_risk: float
    concentration_risk: float

class DynamicRiskManager:
    """Gestor de riesgo dinámico institucional"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Parámetros base
        self.base_risk_reward_ratio = config.get('base_risk_reward_ratio', 2.0)
        self.max_portfolio_heat = config.get('max_portfolio_heat', 0.02)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)
        
        # Ajustes por condición de mercado
        self.market_condition_adjustments = {
            MarketCondition.NORMAL: {'ratio_multiplier': 1.0, 'size_multiplier': 1.0},
            MarketCondition.HIGH_VOLATILITY: {'ratio_multiplier': 1.5, 'size_multiplier': 0.7},
            MarketCondition.LOW_LIQUIDITY: {'ratio_multiplier': 2.0, 'size_multiplier': 0.5},
            MarketCondition.NEWS_EVENT: {'ratio_multiplier': 2.5, 'size_multiplier': 0.3},
            MarketCondition.EXTREME_STRESS: {'ratio_multiplier': 3.0, 'size_multiplier': 0.2}
        }
        
        self.trade_history = []
    
    async def calculate_dynamic_risk_parameters(self, 
                                              symbol: str,
                                              entry_price: float,
                                              market_data: Dict,
                                              portfolio_data: Dict) -> RiskParameters:
        """Calcula parámetros de riesgo dinámicos"""
        try:
            # Detectar condición de mercado
            market_condition = await self._detect_market_condition(market_data)
            
            # Calcular ratio ajustado
            base_ratio = self.base_risk_reward_ratio
            condition_adjustment = self.market_condition_adjustments[market_condition]
            adjusted_ratio = base_ratio * condition_adjustment['ratio_multiplier']
            
            # Ajustes adicionales
            volatility_adj = await self._calculate_volatility_adjustment(symbol, market_data)
            liquidity_adj = await self._calculate_liquidity_adjustment(symbol, market_data)
            time_adj = await self._calculate_time_adjustment()
            
            final_ratio = adjusted_ratio * volatility_adj * liquidity_adj * time_adj
            
            # Position sizing
            optimal_size = await self._calculate_optimal_position_size(
                symbol, final_ratio, portfolio_data)
            final_size = optimal_size * condition_adjustment['size_multiplier']
            
            # Stop distances
            stop_distance, tp_distance = await self._calculate_stop_distances(
                symbol, entry_price, final_ratio, market_data)
            
            # Slippage estimation
            expected_slippage = await self._estimate_slippage(symbol, final_size, market_data)
            
            return RiskParameters(
                base_risk_reward_ratio=base_ratio,
                adjusted_risk_reward_ratio=final_ratio,
                position_size_pct=final_size,
                max_portfolio_heat=self.max_portfolio_heat,
                stop_loss_distance=stop_distance,
                take_profit_distance=tp_distance,
                market_condition=market_condition,
                confidence_level=0.95,
                expected_slippage=expected_slippage
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {e}")
            return self._get_conservative_parameters()
    
    async def position_sizing_kelly(self, 
                                  symbol: str,
                                  win_rate: float,
                                  avg_win: float,
                                  avg_loss: float) -> float:
        """Kelly Criterion position sizing"""
        try:
            if avg_loss == 0 or win_rate == 0:
                return 0.01
            
            # Kelly Criterion: f = (bp - q) / b
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            optimal_fraction = max(0, kelly_fraction * self.kelly_fraction)
            
            return min(optimal_fraction, 0.05)  # Max 5%
            
        except Exception as e:
            self.logger.error(f"Error in Kelly sizing: {e}")
            return 0.01
    
    async def portfolio_heat_management(self, 
                                      current_positions: List,
                                      proposed_position: Dict) -> Dict:
        """Gestiona heat del portfolio"""
        try:
            current_heat = sum(pos.get('heat_contribution', 0) for pos in current_positions)
            new_heat = proposed_position.get('heat_contribution', 0)
            projected_heat = current_heat + new_heat
            
            can_add = projected_heat <= self.max_portfolio_heat
            
            # Calcular riesgo de correlación
            correlation_risk = await self._calculate_correlation_risk(
                current_positions, proposed_position)
            
            return {
                'can_add_position': can_add,
                'current_heat': current_heat,
                'projected_heat': projected_heat,
                'heat_limit': self.max_portfolio_heat,
                'correlation_risk': correlation_risk
            }
            
        except Exception as e:
            self.logger.error(f"Error in heat management: {e}")
            return {'can_add_position': False}
    
    async def execution_quality_monitoring(self, 
                                         executed_trades: List[Dict]) -> Dict:
        """Monitorea calidad de ejecución"""
        try:
            if not executed_trades:
                return {}
            
            # Calcular métricas de slippage
            slippage_data = []
            fill_rates = []
            latency_data = []
            
            for trade in executed_trades:
                expected_price = trade.get('expected_price', 0)
                executed_price = trade.get('executed_price', 0)
                
                if expected_price > 0:
                    slippage = (executed_price - expected_price) / expected_price
                    slippage_data.append(slippage)
                
                # Fill rate
                requested_qty = trade.get('requested_quantity', 0)
                filled_qty = trade.get('filled_quantity', 0)
                
                if requested_qty > 0:
                    fill_rate = filled_qty / requested_qty
                    fill_rates.append(fill_rate)
                
                # Latencia
                if 'execution_latency_ms' in trade:
                    latency_data.append(trade['execution_latency_ms'])
            
            # Calcular estadísticas
            metrics = {}
            
            if slippage_data:
                metrics['slippage'] = {
                    'mean': np.mean(slippage_data),
                    'median': np.median(slippage_data),
                    'std': np.std(slippage_data),
                    'percentile_95': np.percentile(slippage_data, 95)
                }
            
            if fill_rates:
                metrics['fill_rate'] = {
                    'mean': np.mean(fill_rates),
                    'median': np.median(fill_rates),
                    'min': min(fill_rates)
                }
            
            if latency_data:
                metrics['latency'] = {
                    'mean': np.mean(latency_data),
                    'median': np.median(latency_data),
                    'percentile_95': np.percentile(latency_data, 95)
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring execution quality: {e}")
            return {}
    
    # Métodos auxiliares
    async def _detect_market_condition(self, market_data: Dict) -> MarketCondition:
        """Detecta condición de mercado"""
        try:
            volatility = market_data.get('volatility', 0)
            spread = market_data.get('spread', 0)
            volume = market_data.get('volume', 0)
            news_impact = market_data.get('news_impact', 0)
            
            if news_impact > 0.8:
                return MarketCondition.NEWS_EVENT
            elif volatility > 0.1:
                return MarketCondition.EXTREME_STRESS
            elif volatility > 0.05:
                return MarketCondition.HIGH_VOLATILITY
            elif spread > 0.002 or volume < 1000:
                return MarketCondition.LOW_LIQUIDITY
            else:
                return MarketCondition.NORMAL
                
        except Exception as e:
            self.logger.error(f"Error detecting market condition: {e}")
            return MarketCondition.NORMAL
    
    async def _calculate_volatility_adjustment(self, symbol: str, market_data: Dict) -> float:
        """Ajuste por volatilidad"""
        try:
            current_vol = market_data.get('volatility', 0.02)
            historical_vol = market_data.get('historical_volatility', 0.02)
            
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            adjustment = 1.0 + (vol_ratio - 1.0) * 0.5
            
            return max(0.5, min(2.0, adjustment))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    async def _calculate_liquidity_adjustment(self, symbol: str, market_data: Dict) -> float:
        """Ajuste por liquidez"""
        try:
            spread = market_data.get('spread', 0.001)
            volume = market_data.get('volume', 1000)
            
            spread_adjustment = 1.0 + spread * 100
            volume_adjustment = 1.0 + max(0, (1000 - volume) / 1000 * 0.5)
            
            total_adjustment = spread_adjustment * volume_adjustment
            return max(1.0, min(2.0, total_adjustment))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity adjustment: {e}")
            return 1.0
    
    async def _calculate_time_adjustment(self) -> float:
        """Ajuste por horario"""
        try:
            import datetime
            current_hour = datetime.datetime.now().hour
            
            if 13 <= current_hour <= 16:  # London-NY overlap
                return 1.0
            elif 8 <= current_hour <= 17:  # Londres
                return 1.1
            elif 14 <= current_hour <= 21:  # NY
                return 1.1
            else:  # Fuera de horarios
                return 1.3
                
        except Exception as e:
            self.logger.error(f"Error calculating time adjustment: {e}")
            return 1.0
    
    async def _calculate_optimal_position_size(self, 
                                             symbol: str,
                                             risk_reward_ratio: float,
                                             portfolio_data: Dict) -> float:
        """Calcula position size óptimo"""
        try:
            # Usar Kelly como base
            kelly_size = await self.position_sizing_kelly(symbol, 0.6, 0.02, -0.01)
            
            # Ajustar por R/R ratio
            rr_adjustment = min(1.5, risk_reward_ratio / 2.0)
            adjusted_size = kelly_size * rr_adjustment
            
            # Limitar por restricciones
            max_allowed = min(0.05, self.max_portfolio_heat / 2)
            final_size = min(adjusted_size, max_allowed)
            
            return max(0.001, final_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size: {e}")
            return 0.01
    
    async def _calculate_stop_distances(self, 
                                      symbol: str,
                                      entry_price: float,
                                      risk_reward_ratio: float,
                                      market_data: Dict) -> Tuple[float, float]:
        """Calcula distancias de stop"""
        try:
            atr = market_data.get('atr', entry_price * 0.02)
            
            stop_distance = atr * 1.5
            take_profit_distance = stop_distance * risk_reward_ratio
            
            return stop_distance, take_profit_distance
            
        except Exception as e:
            self.logger.error(f"Error calculating stop distances: {e}")
            return entry_price * 0.02, entry_price * 0.04
    
    async def _estimate_slippage(self, 
                               symbol: str,
                               position_size: float,
                               market_data: Dict) -> float:
        """Estima slippage esperado"""
        try:
            spread = market_data.get('spread', 0.001)
            volume = market_data.get('volume', 1000)
            
            base_slippage = spread / 2
            size_impact = position_size * 10000 / volume
            impact_slippage = size_impact * spread
            
            total_slippage = base_slippage + impact_slippage
            return min(0.005, total_slippage)
            
        except Exception as e:
            self.logger.error(f"Error estimating slippage: {e}")
            return 0.001
    
    async def _calculate_correlation_risk(self, 
                                        current_positions: List,
                                        proposed_position: Dict) -> float:
        """Calcula riesgo de correlación"""
        try:
            # Implementar cálculo de correlación
            return 0.3  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.5
    
    def _get_conservative_parameters(self) -> RiskParameters:
        """Parámetros conservadores por defecto"""
        return RiskParameters(
            base_risk_reward_ratio=2.0,
            adjusted_risk_reward_ratio=3.0,
            position_size_pct=0.01,
            max_portfolio_heat=0.02,
            stop_loss_distance=0.02,
            take_profit_distance=0.06,
            market_condition=MarketCondition.EXTREME_STRESS,
            confidence_level=0.95,
            expected_slippage=0.002
        )
```

#### **Configuración**

```yaml
# config/trading/dynamic_risk.yaml
dynamic_risk:
  base_risk_reward_ratio: 2.0
  max_portfolio_heat: 0.02
  max_position_size: 0.05
  kelly_fraction: 0.25
  
  market_adjustments:
    normal:
      ratio_multiplier: 1.0
      size_multiplier: 1.0
    high_volatility:
      ratio_multiplier: 1.5
      size_multiplier: 0.7
    low_liquidity:
      ratio_multiplier: 2.0
      size_multiplier: 0.5
    news_event:
      ratio_multiplier: 2.5
      size_multiplier: 0.3
    extreme_stress:
      ratio_multiplier: 3.0
      size_multiplier: 0.2
  
  execution_quality:
    max_slippage_threshold: 0.001
    min_fill_rate: 0.95
    max_latency_ms: 1000
    
  portfolio_limits:
    max_correlation: 0.7
    max_concentration: 0.3
    
  circuit_breakers:
    daily_drawdown_limit: 0.02
    weekly_drawdown_limit: 0.05
    monthly_drawdown_limit: 0.10
```

## 📊 **Métricas y KPIs**

### **Risk Management**
- **Risk-Adjusted Returns**: Sharpe, Sortino ratios
- **Portfolio Heat**: Heat actual vs límites
- **Position Sizing Accuracy**: vs Kelly óptimo
- **Drawdown Control**: Máximo drawdown histórico

### **Execution Quality**
- **Average Slippage**: Slippage promedio
- **Fill Rate**: % órdenes completadas
- **Execution Latency**: Latencia promedio
- **Market Impact**: Impacto en precio

## 🧪 **Testing Strategy**

```python
# tests/test_dynamic_risk_manager.py
async def test_market_condition_detection():
    pass

async def test_kelly_criterion_calculation():
    pass

async def test_portfolio_heat_calculation():
    pass
```

## 📈 **Performance Targets**

- **Risk Calculation Latency**: < 10ms
- **Portfolio Analysis**: < 100ms
- **Memory Usage**: < 200MB
- **Accuracy**: > 90% en estimación slippage

## 🚀 **Deployment Checklist**

- [ ] Implementar DynamicRiskManager
- [ ] Crear configuraciones YAML
- [ ] Escribir unit tests
- [ ] Configurar métricas
- [ ] Testing en desarrollo
- [ ] Deployment a producción

---

*Documento actualizado: 2025-08-06 - Dynamic Risk Management*
