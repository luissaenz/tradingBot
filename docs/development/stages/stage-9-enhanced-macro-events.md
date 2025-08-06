# 🌍 Stage 9: Enhanced Macro Events Management

## 📋 Objetivo
Extender el módulo de macro events con análisis de posicionamiento institucional, flujos de exchange y datos derivados para mejorar la toma de decisiones.

## 🎯 Componentes a Implementar

### **9.1 Institutional Positioning Analyzer**

#### **Responsabilidades**
- Analizar datos COT de futuros de Bitcoin
- Monitorear flujos de exchanges y wallets institucionales
- Analizar posicionamiento en derivados
- Detectar cambios en sentiment institucional

#### **Implementación**

```python
# src/modules/macro/institutional_positioning_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import aiohttp
from ..shared.logging import get_logger

@dataclass
class COTData:
    """Datos del Commitment of Traders"""
    date: datetime
    asset_manager_long: int
    asset_manager_short: int
    leveraged_funds_long: int
    leveraged_funds_short: int
    other_reportables_long: int
    other_reportables_short: int
    total_long: int
    total_short: int
    open_interest: int

@dataclass
class ExchangeFlow:
    """Flujos de exchanges"""
    timestamp: datetime
    exchange: str
    inflow: float
    outflow: float
    net_flow: float
    whale_transactions: int
    large_transaction_threshold: float

@dataclass
class InstitutionalSignal:
    """Señal institucional generada"""
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    strength: float   # 0-1
    source: str       # 'cot', 'exchange_flows', 'derivatives'
    confidence: float # 0-1
    timeframe: str    # 'short', 'medium', 'long'
    description: str

class InstitutionalPositioningAnalyzer:
    """Analizador de posicionamiento institucional"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        
        # APIs y endpoints
        self.cot_api_url = config.get('cot_api_url', 'https://api.cftc.gov/cot')
        self.whale_alert_api = config.get('whale_alert_api', '')
        self.glassnode_api = config.get('glassnode_api', '')
        
        # Thresholds
        self.whale_threshold = config.get('whale_threshold', 100)  # BTC
        self.extreme_positioning_percentile = config.get('extreme_positioning', 0.9)
        
        # Cache para datos históricos
        self.cot_history = []
        self.exchange_flows_history = []
        
    async def analyze_cot_bitcoin_futures(self) -> List[InstitutionalSignal]:
        """
        Analiza datos COT de futuros de Bitcoin (CME)
        """
        try:
            # Obtener datos COT más recientes
            cot_data = await self._fetch_cot_data()
            if not cot_data:
                return []
            
            signals = []
            
            # Analizar posicionamiento de Asset Managers
            am_signal = await self._analyze_asset_manager_positioning(cot_data)
            if am_signal:
                signals.append(am_signal)
            
            # Analizar posicionamiento de Leveraged Funds
            lf_signal = await self._analyze_leveraged_funds_positioning(cot_data)
            if lf_signal:
                signals.append(lf_signal)
            
            # Analizar cambios en Open Interest
            oi_signal = await self._analyze_open_interest_changes(cot_data)
            if oi_signal:
                signals.append(oi_signal)
            
            # Detectar posicionamiento extremo (contrarian signals)
            extreme_signals = await self._detect_extreme_positioning(cot_data)
            signals.extend(extreme_signals)
            
            self.logger.info(f"Generated {len(signals)} COT-based signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing COT data: {e}")
            return []
    
    async def analyze_exchange_flows(self) -> List[InstitutionalSignal]:
        """
        Analiza flujos de exchanges para detectar actividad institucional
        """
        try:
            # Obtener datos de flujos de exchanges
            exchange_flows = await self._fetch_exchange_flows()
            if not exchange_flows:
                return []
            
            signals = []
            
            # Analizar inflows/outflows netos
            net_flow_signal = await self._analyze_net_flows(exchange_flows)
            if net_flow_signal:
                signals.append(net_flow_signal)
            
            # Detectar acumulación institucional
            accumulation_signal = await self._detect_institutional_accumulation(exchange_flows)
            if accumulation_signal:
                signals.append(accumulation_signal)
            
            # Analizar flujos de stablecoins (preparación institucional)
            stablecoin_signal = await self._analyze_stablecoin_flows()
            if stablecoin_signal:
                signals.append(stablecoin_signal)
            
            # Detectar selling pressure de miners
            miner_signal = await self._analyze_miner_selling_pressure()
            if miner_signal:
                signals.append(miner_signal)
            
            self.logger.info(f"Generated {len(signals)} exchange flow signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing exchange flows: {e}")
            return []
    
    async def analyze_derivatives_positioning(self) -> List[InstitutionalSignal]:
        """
        Analiza posicionamiento en derivados
        """
        try:
            signals = []
            
            # Analizar options flow
            options_signal = await self._analyze_options_flow()
            if options_signal:
                signals.append(options_signal)
            
            # Analizar futures open interest
            futures_signal = await self._analyze_futures_open_interest()
            if futures_signal:
                signals.append(futures_signal)
            
            # Analizar funding rates
            funding_signal = await self._analyze_funding_rates()
            if funding_signal:
                signals.append(funding_signal)
            
            # Analizar premium perpetual vs spot
            premium_signal = await self._analyze_perpetual_premium()
            if premium_signal:
                signals.append(premium_signal)
            
            self.logger.info(f"Generated {len(signals)} derivatives signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing derivatives: {e}")
            return []
    
    async def generate_composite_institutional_signal(self) -> Optional[InstitutionalSignal]:
        """
        Genera señal compuesta basada en todos los análisis institucionales
        """
        try:
            # Obtener todas las señales
            cot_signals = await self.analyze_cot_bitcoin_futures()
            flow_signals = await self.analyze_exchange_flows()
            derivatives_signals = await self.analyze_derivatives_positioning()
            
            all_signals = cot_signals + flow_signals + derivatives_signals
            
            if not all_signals:
                return None
            
            # Calcular señal compuesta
            bullish_signals = [s for s in all_signals if s.signal_type == 'bullish']
            bearish_signals = [s for s in all_signals if s.signal_type == 'bearish']
            
            # Weighted scoring
            bullish_score = sum(s.strength * s.confidence for s in bullish_signals)
            bearish_score = sum(s.strength * s.confidence for s in bearish_signals)
            
            total_weight = sum(s.confidence for s in all_signals)
            
            if total_weight == 0:
                return None
            
            # Determinar señal final
            net_score = (bullish_score - bearish_score) / total_weight
            
            if net_score > 0.3:
                signal_type = 'bullish'
                strength = min(1.0, net_score)
            elif net_score < -0.3:
                signal_type = 'bearish'
                strength = min(1.0, abs(net_score))
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Calcular confianza promedio
            avg_confidence = sum(s.confidence for s in all_signals) / len(all_signals)
            
            composite_signal = InstitutionalSignal(
                signal_type=signal_type,
                strength=strength,
                source='composite',
                confidence=avg_confidence,
                timeframe='medium',
                description=f"Composite institutional signal from {len(all_signals)} sources"
            )
            
            self.logger.info(f"Generated composite signal: {signal_type} "
                           f"(strength: {strength:.2f}, confidence: {avg_confidence:.2f})")
            
            return composite_signal
            
        except Exception as e:
            self.logger.error(f"Error generating composite signal: {e}")
            return None
    
    # Métodos auxiliares para COT analysis
    async def _fetch_cot_data(self) -> Optional[COTData]:
        """Obtiene datos COT más recientes"""
        try:
            # Implementar llamada a API CFTC
            # Por ahora retornar datos simulados
            return COTData(
                date=datetime.now(),
                asset_manager_long=15000,
                asset_manager_short=5000,
                leveraged_funds_long=8000,
                leveraged_funds_short=12000,
                other_reportables_long=3000,
                other_reportables_short=2000,
                total_long=26000,
                total_short=19000,
                open_interest=45000
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching COT data: {e}")
            return None
    
    async def _analyze_asset_manager_positioning(self, cot_data: COTData) -> Optional[InstitutionalSignal]:
        """Analiza posicionamiento de Asset Managers"""
        try:
            net_position = cot_data.asset_manager_long - cot_data.asset_manager_short
            total_position = cot_data.asset_manager_long + cot_data.asset_manager_short
            
            if total_position == 0:
                return None
            
            net_ratio = net_position / total_position
            
            # Determinar señal
            if net_ratio > 0.6:  # 60% net long
                signal_type = 'bullish'
                strength = min(1.0, net_ratio)
            elif net_ratio < -0.6:  # 60% net short
                signal_type = 'bearish'
                strength = min(1.0, abs(net_ratio))
            else:
                return None  # Neutral, no generar señal
            
            return InstitutionalSignal(
                signal_type=signal_type,
                strength=strength,
                source='cot_asset_managers',
                confidence=0.8,
                timeframe='medium',
                description=f"Asset Managers net {signal_type} (ratio: {net_ratio:.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing asset manager positioning: {e}")
            return None
    
    async def _analyze_leveraged_funds_positioning(self, cot_data: COTData) -> Optional[InstitutionalSignal]:
        """Analiza posicionamiento de Leveraged Funds (contrarian)"""
        try:
            net_position = cot_data.leveraged_funds_long - cot_data.leveraged_funds_short
            total_position = cot_data.leveraged_funds_long + cot_data.leveraged_funds_short
            
            if total_position == 0:
                return None
            
            net_ratio = net_position / total_position
            
            # Señal contrarian para leveraged funds
            if net_ratio > 0.7:  # Extremely long -> bearish signal
                signal_type = 'bearish'
                strength = min(1.0, net_ratio - 0.5)
            elif net_ratio < -0.7:  # Extremely short -> bullish signal
                signal_type = 'bullish'
                strength = min(1.0, abs(net_ratio) - 0.5)
            else:
                return None
            
            return InstitutionalSignal(
                signal_type=signal_type,
                strength=strength,
                source='cot_leveraged_funds',
                confidence=0.7,
                timeframe='short',
                description=f"Leveraged Funds contrarian signal: {signal_type}"
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing leveraged funds: {e}")
            return None
    
    async def _analyze_open_interest_changes(self, cot_data: COTData) -> Optional[InstitutionalSignal]:
        """Analiza cambios en Open Interest"""
        try:
            if len(self.cot_history) < 2:
                self.cot_history.append(cot_data)
                return None
            
            previous_oi = self.cot_history[-1].open_interest
            current_oi = cot_data.open_interest
            
            oi_change = (current_oi - previous_oi) / previous_oi
            
            # Actualizar historial
            self.cot_history.append(cot_data)
            if len(self.cot_history) > 10:  # Mantener últimas 10 semanas
                self.cot_history.pop(0)
            
            # Generar señal basada en cambio de OI
            if oi_change > 0.1:  # 10% increase
                return InstitutionalSignal(
                    signal_type='bullish',
                    strength=min(1.0, oi_change * 5),
                    source='cot_open_interest',
                    confidence=0.6,
                    timeframe='medium',
                    description=f"Open Interest increased by {oi_change:.1%}"
                )
            elif oi_change < -0.1:  # 10% decrease
                return InstitutionalSignal(
                    signal_type='bearish',
                    strength=min(1.0, abs(oi_change) * 5),
                    source='cot_open_interest',
                    confidence=0.6,
                    timeframe='medium',
                    description=f"Open Interest decreased by {oi_change:.1%}"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing OI changes: {e}")
            return None
    
    async def _detect_extreme_positioning(self, cot_data: COTData) -> List[InstitutionalSignal]:
        """Detecta posicionamiento extremo para señales contrarian"""
        try:
            signals = []
            
            # Calcular percentiles históricos (requiere más datos históricos)
            # Por ahora usar thresholds fijos
            
            am_net = cot_data.asset_manager_long - cot_data.asset_manager_short
            lf_net = cot_data.leveraged_funds_long - cot_data.leveraged_funds_short
            
            # Asset Managers extremely bullish -> potential top
            if am_net > 15000:  # Threshold ejemplo
                signals.append(InstitutionalSignal(
                    signal_type='bearish',
                    strength=0.7,
                    source='cot_extreme_positioning',
                    confidence=0.6,
                    timeframe='long',
                    description="Asset Managers extremely bullish - potential top"
                ))
            
            # Leveraged Funds extremely bearish -> potential bottom
            if lf_net < -8000:  # Threshold ejemplo
                signals.append(InstitutionalSignal(
                    signal_type='bullish',
                    strength=0.7,
                    source='cot_extreme_positioning',
                    confidence=0.6,
                    timeframe='medium',
                    description="Leveraged Funds extremely bearish - potential bottom"
                ))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error detecting extreme positioning: {e}")
            return []
    
    # Métodos para Exchange Flows
    async def _fetch_exchange_flows(self) -> List[ExchangeFlow]:
        """Obtiene datos de flujos de exchanges"""
        try:
            # Implementar llamadas a APIs de exchanges
            # Por ahora retornar datos simulados
            return [
                ExchangeFlow(
                    timestamp=datetime.now(),
                    exchange='binance',
                    inflow=1500.0,
                    outflow=1200.0,
                    net_flow=300.0,
                    whale_transactions=5,
                    large_transaction_threshold=100.0
                )
            ]
            
        except Exception as e:
            self.logger.error(f"Error fetching exchange flows: {e}")
            return []
    
    async def _analyze_net_flows(self, flows: List[ExchangeFlow]) -> Optional[InstitutionalSignal]:
        """Analiza flujos netos de exchanges"""
        try:
            total_net_flow = sum(flow.net_flow for flow in flows)
            
            # Normalizar por volumen típico
            normalized_flow = total_net_flow / 10000  # 10k BTC como base
            
            if normalized_flow > 0.1:  # Significant outflow
                return InstitutionalSignal(
                    signal_type='bullish',
                    strength=min(1.0, normalized_flow * 5),
                    source='exchange_flows',
                    confidence=0.7,
                    timeframe='short',
                    description=f"Net outflow from exchanges: {total_net_flow:.0f} BTC"
                )
            elif normalized_flow < -0.1:  # Significant inflow
                return InstitutionalSignal(
                    signal_type='bearish',
                    strength=min(1.0, abs(normalized_flow) * 5),
                    source='exchange_flows',
                    confidence=0.7,
                    timeframe='short',
                    description=f"Net inflow to exchanges: {abs(total_net_flow):.0f} BTC"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing net flows: {e}")
            return None
    
    async def _detect_institutional_accumulation(self, flows: List[ExchangeFlow]) -> Optional[InstitutionalSignal]:
        """Detecta acumulación institucional"""
        try:
            # Buscar patrones de acumulación (outflows consistentes)
            recent_flows = flows[-7:]  # Últimos 7 días
            
            if len(recent_flows) < 5:
                return None
            
            positive_days = sum(1 for flow in recent_flows if flow.net_flow > 0)
            
            if positive_days >= 5:  # 5+ días de outflows
                avg_outflow = sum(flow.net_flow for flow in recent_flows) / len(recent_flows)
                
                return InstitutionalSignal(
                    signal_type='bullish',
                    strength=0.8,
                    source='institutional_accumulation',
                    confidence=0.8,
                    timeframe='medium',
                    description=f"Consistent accumulation pattern: {avg_outflow:.0f} BTC/day"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting accumulation: {e}")
            return None
    
    async def _analyze_stablecoin_flows(self) -> Optional[InstitutionalSignal]:
        """Analiza flujos de stablecoins"""
        try:
            # Implementar análisis de stablecoin flows
            # Placeholder por ahora
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing stablecoin flows: {e}")
            return None
    
    async def _analyze_miner_selling_pressure(self) -> Optional[InstitutionalSignal]:
        """Analiza presión de venta de miners"""
        try:
            # Implementar análisis de miner flows
            # Placeholder por ahora
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing miner pressure: {e}")
            return None
    
    # Métodos para Derivatives
    async def _analyze_options_flow(self) -> Optional[InstitutionalSignal]:
        """Analiza flujo de opciones"""
        try:
            # Implementar análisis de options flow
            # Placeholder por ahora
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing options flow: {e}")
            return None
    
    async def _analyze_futures_open_interest(self) -> Optional[InstitutionalSignal]:
        """Analiza open interest de futuros"""
        try:
            # Implementar análisis de futures OI
            # Placeholder por ahora
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing futures OI: {e}")
            return None
    
    async def _analyze_funding_rates(self) -> Optional[InstitutionalSignal]:
        """Analiza funding rates"""
        try:
            # Implementar análisis de funding rates
            # Placeholder por ahora
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing funding rates: {e}")
            return None
    
    async def _analyze_perpetual_premium(self) -> Optional[InstitutionalSignal]:
        """Analiza premium perpetual vs spot"""
        try:
            # Implementar análisis de premium
            # Placeholder por ahora
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing perpetual premium: {e}")
            return None

# Tests
class TestInstitutionalPositioningAnalyzer:
    """Tests para InstitutionalPositioningAnalyzer"""
    
    async def test_cot_analysis(self):
        """Test análisis COT"""
        pass
    
    async def test_exchange_flows_analysis(self):
        """Test análisis de flujos"""
        pass
    
    async def test_composite_signal_generation(self):
        """Test generación de señal compuesta"""
        pass
```

#### **Configuración**

```yaml
# config/macro/institutional_positioning.yaml
institutional_positioning:
  # APIs
  cot_api_url: "https://api.cftc.gov/cot"
  whale_alert_api: ""
  glassnode_api: ""
  
  # Thresholds
  whale_threshold: 100  # BTC
  extreme_positioning_percentile: 0.9
  
  # COT Analysis
  cot:
    asset_manager_bullish_threshold: 0.6
    leveraged_funds_extreme_threshold: 0.7
    open_interest_change_threshold: 0.1
    
  # Exchange Flows
  exchange_flows:
    significant_flow_threshold: 0.1  # Normalized
    accumulation_days_required: 5
    whale_transaction_threshold: 100  # BTC
    
  # Signal Weights
  signal_weights:
    cot_asset_managers: 0.8
    cot_leveraged_funds: 0.7
    cot_open_interest: 0.6
    exchange_flows: 0.7
    institutional_accumulation: 0.8
    derivatives: 0.6
```

## 📊 **Métricas y KPIs**

### **COT Analysis**
- **Positioning Accuracy**: % de señales COT correctas
- **Signal Timing**: Tiempo promedio hasta materialización
- **Extreme Detection**: Accuracy en detección de extremos

### **Exchange Flows**
- **Flow Prediction**: Accuracy en predicción de movimientos
- **Whale Detection**: % de transacciones whale detectadas
- **Accumulation Patterns**: Accuracy en detección de patrones

## 🧪 **Testing Strategy**

```python
# tests/test_institutional_positioning.py
async def test_cot_data_processing():
    pass

async def test_exchange_flow_analysis():
    pass

async def test_signal_generation():
    pass
```

## 📈 **Performance Targets**

- **Data Processing**: < 5s para análisis completo
- **Signal Generation**: < 1s por señal
- **Memory Usage**: < 100MB
- **API Response**: < 2s por llamada

## 🚀 **Deployment Checklist**

- [ ] Implementar InstitutionalPositioningAnalyzer
- [ ] Configurar APIs externas
- [ ] Crear configuraciones YAML
- [ ] Escribir unit tests
- [ ] Configurar logging y métricas
- [ ] Testing con datos reales
- [ ] Deployment a producción

---

*Documento creado: 2025-08-06 - Enhanced Macro Events*
