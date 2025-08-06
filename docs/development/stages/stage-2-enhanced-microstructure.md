# 🔬 Stage 2: Enhanced Microstructure Processing

## 📋 Objetivo
Extender el módulo de microstructure processing con capacidades institucionales avanzadas para detectar niveles clave, analizar volatilidad y mejorar la calidad de las señales.

## 🎯 Componentes a Implementar

### **2.1 Key Levels Detection System**

#### **Responsabilidades**
- Detectar niveles de soporte/resistencia institucionales
- Identificar zonas de alta liquidez
- Calcular niveles VWAP multi-timeframe
- Detectar niveles psicológicos y fibonacci

#### **Implementación**

```python
# src/modules/microstructure/key_levels_detector.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from ..shared.logging import get_logger

@dataclass
class KeyLevel:
    """Representa un nivel clave identificado"""
    price: float
    strength: float  # 0-1, donde 1 es máxima fuerza
    level_type: str  # 'support', 'resistance', 'poc', 'vwap'
    timeframe: str   # '1m', '5m', '15m', '1h', '4h', '1d'
    volume: float    # Volumen asociado al nivel
    touches: int     # Número de veces que el precio tocó este nivel
    last_touch: pd.Timestamp
    confidence: float # Confianza estadística del nivel

class KeyLevelsDetector:
    """Detector de niveles clave institucionales"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        self.min_strength = config.get('min_strength', 0.3)
        self.lookback_periods = config.get('lookback_periods', {
            '1m': 1440,   # 1 día
            '5m': 2016,   # 1 semana
            '15m': 2688,  # 4 semanas
            '1h': 2160,   # 3 meses
            '4h': 2160,   # 1 año
            '1d': 365     # 1 año
        })
    
    def detect_volume_profile_levels(self, 
                                   price_data: pd.DataFrame, 
                                   volume_data: pd.DataFrame) -> List[KeyLevel]:
        """
        Detecta niveles basados en perfil de volumen
        
        Returns:
            List[KeyLevel]: Niveles POC, VAH, VAL
        """
        levels = []
        
        try:
            # Calcular Value Area (70% del volumen)
            price_bins = np.linspace(price_data['low'].min(), 
                                   price_data['high'].max(), 100)
            volume_profile = self._calculate_volume_profile(
                price_data, volume_data, price_bins)
            
            # Point of Control (POC) - máximo volumen
            poc_idx = volume_profile.argmax()
            poc_price = price_bins[poc_idx]
            poc_volume = volume_profile[poc_idx]
            
            levels.append(KeyLevel(
                price=poc_price,
                strength=1.0,
                level_type='poc',
                timeframe='current',
                volume=poc_volume,
                touches=self._count_touches(price_data, poc_price),
                last_touch=price_data.index[-1],
                confidence=0.95
            ))
            
            # Value Area High/Low (70% del volumen)
            vah, val = self._calculate_value_area(volume_profile, price_bins)
            
            levels.extend([
                KeyLevel(
                    price=vah,
                    strength=0.8,
                    level_type='resistance',
                    timeframe='current',
                    volume=volume_profile[np.argmin(np.abs(price_bins - vah))],
                    touches=self._count_touches(price_data, vah),
                    last_touch=price_data.index[-1],
                    confidence=0.85
                ),
                KeyLevel(
                    price=val,
                    strength=0.8,
                    level_type='support',
                    timeframe='current',
                    volume=volume_profile[np.argmin(np.abs(price_bins - val))],
                    touches=self._count_touches(price_data, val),
                    last_touch=price_data.index[-1],
                    confidence=0.85
                )
            ])
            
            self.logger.info(f"Detected {len(levels)} volume profile levels")
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting volume profile levels: {e}")
            return []
    
    def detect_algorithmic_levels(self, price_data: pd.DataFrame) -> List[KeyLevel]:
        """
        Detecta niveles algorítmicos y psicológicos
        """
        levels = []
        
        try:
            current_price = price_data['close'].iloc[-1]
            
            # Niveles psicológicos (números redondos)
            psychological_levels = self._get_psychological_levels(current_price)
            for level in psychological_levels:
                levels.append(KeyLevel(
                    price=level,
                    strength=0.6,
                    level_type='psychological',
                    timeframe='static',
                    volume=0,
                    touches=self._count_touches(price_data, level),
                    last_touch=pd.Timestamp.now(),
                    confidence=0.7
                ))
            
            # Fibonacci retracements
            fib_levels = self._calculate_fibonacci_levels(price_data)
            levels.extend(fib_levels)
            
            # Previous day/week/month high/low
            pivot_levels = self._calculate_pivot_levels(price_data)
            levels.extend(pivot_levels)
            
            # VWAP levels multi-timeframe
            vwap_levels = self._calculate_vwap_levels(price_data)
            levels.extend(vwap_levels)
            
            self.logger.info(f"Detected {len(levels)} algorithmic levels")
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting algorithmic levels: {e}")
            return []
    
    def detect_liquidity_zones(self, orderbook_data: pd.DataFrame) -> List[KeyLevel]:
        """
        Detecta zonas de alta liquidez y órdenes iceberg
        """
        levels = []
        
        try:
            # Analizar order book depth
            bid_levels = self._analyze_orderbook_side(orderbook_data, 'bid')
            ask_levels = self._analyze_orderbook_side(orderbook_data, 'ask')
            
            levels.extend(bid_levels)
            levels.extend(ask_levels)
            
            # Detectar órdenes iceberg (patrones de reposición)
            iceberg_levels = self._detect_iceberg_orders(orderbook_data)
            levels.extend(iceberg_levels)
            
            self.logger.info(f"Detected {len(levels)} liquidity zones")
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity zones: {e}")
            return []
    
    def _calculate_volume_profile(self, price_data: pd.DataFrame, 
                                volume_data: pd.DataFrame, 
                                price_bins: np.ndarray) -> np.ndarray:
        """Calcula el perfil de volumen por precio"""
        volume_profile = np.zeros(len(price_bins) - 1)
        
        for i in range(len(price_data)):
            high = price_data['high'].iloc[i]
            low = price_data['low'].iloc[i]
            volume = volume_data['volume'].iloc[i]
            
            # Distribuir volumen proporcionalmente en el rango high-low
            bin_indices = np.where((price_bins >= low) & (price_bins <= high))[0]
            if len(bin_indices) > 0:
                volume_per_bin = volume / len(bin_indices)
                for idx in bin_indices[:-1]:  # Excluir último bin
                    volume_profile[idx] += volume_per_bin
        
        return volume_profile
    
    def _calculate_value_area(self, volume_profile: np.ndarray, 
                            price_bins: np.ndarray) -> Tuple[float, float]:
        """Calcula Value Area High y Low (70% del volumen)"""
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.7
        
        # Encontrar POC
        poc_idx = volume_profile.argmax()
        
        # Expandir desde POC hasta alcanzar 70% del volumen
        accumulated_volume = volume_profile[poc_idx]
        left_idx = poc_idx
        right_idx = poc_idx
        
        while accumulated_volume < target_volume:
            # Decidir si expandir hacia la izquierda o derecha
            left_volume = volume_profile[left_idx - 1] if left_idx > 0 else 0
            right_volume = volume_profile[right_idx + 1] if right_idx < len(volume_profile) - 1 else 0
            
            if left_volume >= right_volume and left_idx > 0:
                left_idx -= 1
                accumulated_volume += left_volume
            elif right_idx < len(volume_profile) - 1:
                right_idx += 1
                accumulated_volume += right_volume
            else:
                break
        
        vah = price_bins[right_idx]
        val = price_bins[left_idx]
        
        return vah, val
    
    def _count_touches(self, price_data: pd.DataFrame, level: float, 
                      tolerance: float = 0.001) -> int:
        """Cuenta cuántas veces el precio tocó un nivel específico"""
        touches = 0
        for i in range(len(price_data)):
            high = price_data['high'].iloc[i]
            low = price_data['low'].iloc[i]
            
            if low <= level * (1 + tolerance) and high >= level * (1 - tolerance):
                touches += 1
        
        return touches
    
    def _get_psychological_levels(self, current_price: float) -> List[float]:
        """Genera niveles psicológicos (números redondos)"""
        levels = []
        
        # Determinar el orden de magnitud
        magnitude = 10 ** int(np.log10(current_price))
        
        # Niveles cada 1000, 500, 100, 50 dependiendo del precio
        if current_price > 10000:
            step = 1000
        elif current_price > 1000:
            step = 500
        elif current_price > 100:
            step = 100
        else:
            step = 50
        
        # Generar niveles arriba y abajo del precio actual
        base = int(current_price / step) * step
        for i in range(-5, 6):
            level = base + (i * step)
            if level > 0:
                levels.append(float(level))
        
        return levels
    
    def _calculate_fibonacci_levels(self, price_data: pd.DataFrame) -> List[KeyLevel]:
        """Calcula niveles de retroceso de Fibonacci"""
        levels = []
        
        # Encontrar swing high y low recientes
        swing_high = price_data['high'].rolling(20).max().iloc[-1]
        swing_low = price_data['low'].rolling(20).min().iloc[-1]
        
        # Niveles de Fibonacci
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            # Retroceso desde high a low
            fib_level = swing_high - (swing_high - swing_low) * ratio
            
            levels.append(KeyLevel(
                price=fib_level,
                strength=0.7,
                level_type='fibonacci',
                timeframe='swing',
                volume=0,
                touches=self._count_touches(price_data, fib_level),
                last_touch=pd.Timestamp.now(),
                confidence=0.75
            ))
        
        return levels
    
    def _calculate_pivot_levels(self, price_data: pd.DataFrame) -> List[KeyLevel]:
        """Calcula niveles pivot (previous high/low)"""
        levels = []
        
        # Previous day high/low
        prev_high = price_data['high'].iloc[-2] if len(price_data) > 1 else price_data['high'].iloc[-1]
        prev_low = price_data['low'].iloc[-2] if len(price_data) > 1 else price_data['low'].iloc[-1]
        
        levels.extend([
            KeyLevel(
                price=prev_high,
                strength=0.8,
                level_type='resistance',
                timeframe='daily',
                volume=0,
                touches=self._count_touches(price_data, prev_high),
                last_touch=pd.Timestamp.now(),
                confidence=0.8
            ),
            KeyLevel(
                price=prev_low,
                strength=0.8,
                level_type='support',
                timeframe='daily',
                volume=0,
                touches=self._count_touches(price_data, prev_low),
                last_touch=pd.Timestamp.now(),
                confidence=0.8
            )
        ])
        
        return levels
    
    def _calculate_vwap_levels(self, price_data: pd.DataFrame) -> List[KeyLevel]:
        """Calcula VWAP para múltiples timeframes"""
        levels = []
        
        # VWAP diario
        daily_vwap = (price_data['close'] * price_data['volume']).sum() / price_data['volume'].sum()
        
        levels.append(KeyLevel(
            price=daily_vwap,
            strength=0.75,
            level_type='vwap',
            timeframe='daily',
            volume=price_data['volume'].sum(),
            touches=self._count_touches(price_data, daily_vwap),
            last_touch=pd.Timestamp.now(),
            confidence=0.85
        ))
        
        return levels
    
    def _analyze_orderbook_side(self, orderbook_data: pd.DataFrame, 
                              side: str) -> List[KeyLevel]:
        """Analiza un lado del order book para detectar liquidez"""
        levels = []
        
        # Implementar análisis de order book depth
        # Esta función requiere datos de order book en tiempo real
        
        return levels
    
    def _detect_iceberg_orders(self, orderbook_data: pd.DataFrame) -> List[KeyLevel]:
        """Detecta órdenes iceberg por patrones de reposición"""
        levels = []
        
        # Implementar detección de patrones iceberg
        # Requiere análisis temporal del order book
        
        return levels

# Tests
class TestKeyLevelsDetector:
    """Tests para KeyLevelsDetector"""
    
    def test_volume_profile_detection(self):
        """Test detección de perfil de volumen"""
        # Implementar tests
        pass
    
    def test_algorithmic_levels_detection(self):
        """Test detección de niveles algorítmicos"""
        # Implementar tests
        pass
```

#### **Configuración**

```yaml
# config/microstructure/key_levels.yaml
key_levels:
  min_strength: 0.3
  lookback_periods:
    1m: 1440
    5m: 2016
    15m: 2688
    1h: 2160
    4h: 2160
    1d: 365
  
  volume_profile:
    price_bins: 100
    value_area_percentage: 0.7
    min_volume_threshold: 1000
  
  psychological_levels:
    enabled: true
    step_sizes:
      high_price: 1000  # > 10000
      medium_price: 500 # 1000-10000
      low_price: 100    # 100-1000
      very_low_price: 50 # < 100
  
  fibonacci:
    enabled: true
    ratios: [0.236, 0.382, 0.5, 0.618, 0.786]
    swing_period: 20
  
  vwap:
    timeframes: ['1d', '1w', '1m']
    enabled: true
```

### **2.2 Volatility Surface Analyzer**

#### **Implementación**

```python
# src/modules/microstructure/volatility_surface.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
from ..shared.logging import get_logger

@dataclass
class VolatilityRegime:
    """Representa un régimen de volatilidad"""
    regime_type: str  # 'low', 'medium', 'high', 'extreme'
    current_vol: float
    percentile: float  # Percentil histórico
    duration: int     # Duración en períodos
    mean_reversion_prob: float  # Probabilidad de reversión
    breakout_prob: float       # Probabilidad de breakout

class VolatilitySurfaceAnalyzer:
    """Analizador de superficie de volatilidad"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        self.lookback_window = config.get('lookback_window', 252)
        self.regime_thresholds = config.get('regime_thresholds', {
            'low': 0.25,
            'medium': 0.75,
            'high': 0.9,
            'extreme': 0.95
        })
    
    def calculate_realized_volatility_regimes(self, 
                                            price_data: pd.DataFrame) -> VolatilityRegime:
        """
        Calcula regímenes de volatilidad realizada usando múltiples estimadores
        """
        try:
            # Calcular diferentes estimadores de volatilidad
            parkinson_vol = self._calculate_parkinson_volatility(price_data)
            garman_klass_vol = self._calculate_garman_klass_volatility(price_data)
            rogers_satchell_vol = self._calculate_rogers_satchell_volatility(price_data)
            
            # Volatilidad combinada (promedio ponderado)
            combined_vol = (
                0.4 * parkinson_vol +
                0.4 * garman_klass_vol +
                0.2 * rogers_satchell_vol
            )
            
            current_vol = combined_vol.iloc[-1]
            
            # Calcular percentil histórico
            percentile = stats.percentileofscore(combined_vol.dropna(), current_vol) / 100
            
            # Determinar régimen
            regime_type = self._classify_volatility_regime(percentile)
            
            # Calcular duración del régimen actual
            duration = self._calculate_regime_duration(combined_vol, regime_type)
            
            # Calcular probabilidades de transición
            mean_reversion_prob = self._calculate_mean_reversion_probability(
                combined_vol, current_vol)
            breakout_prob = self._calculate_breakout_probability(
                combined_vol, current_vol)
            
            regime = VolatilityRegime(
                regime_type=regime_type,
                current_vol=current_vol,
                percentile=percentile,
                duration=duration,
                mean_reversion_prob=mean_reversion_prob,
                breakout_prob=breakout_prob
            )
            
            self.logger.info(f"Volatility regime: {regime_type} "
                           f"(percentile: {percentile:.2f})")
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility regimes: {e}")
            return None
    
    def volatility_term_structure(self, 
                                 price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analiza la estructura temporal de volatilidad
        """
        try:
            term_structure = {}
            
            # Calcular volatilidad para diferentes horizontes
            horizons = [5, 10, 20, 60, 120, 252]  # días
            
            for horizon in horizons:
                if len(price_data) >= horizon:
                    recent_data = price_data.tail(horizon)
                    vol = self._calculate_parkinson_volatility(recent_data).iloc[-1]
                    term_structure[f'{horizon}d'] = vol
            
            # Calcular pendiente de la estructura temporal
            if len(term_structure) >= 2:
                horizons_list = list(term_structure.keys())
                vols_list = list(term_structure.values())
                
                # Regresión lineal para obtener pendiente
                x = np.arange(len(vols_list))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, vols_list)
                
                term_structure['slope'] = slope
                term_structure['r_squared'] = r_value ** 2
            
            self.logger.info(f"Volatility term structure calculated: {len(term_structure)} points")
            return term_structure
            
        except Exception as e:
            self.logger.error(f"Error calculating term structure: {e}")
            return {}
    
    def volatility_breakout_detection(self, 
                                    price_data: pd.DataFrame) -> Dict[str, any]:
        """
        Detecta breakouts de volatilidad
        """
        try:
            # Calcular volatilidad rolling
            vol_window = 20
            current_vol = self._calculate_parkinson_volatility(
                price_data.tail(vol_window)).iloc[-1]
            
            # Volatilidad promedio histórica
            historical_vol = self._calculate_parkinson_volatility(price_data)
            avg_vol = historical_vol.mean()
            std_vol = historical_vol.std()
            
            # Z-score de volatilidad actual
            vol_zscore = (current_vol - avg_vol) / std_vol
            
            # Detectar compresión/expansión
            compression_threshold = -1.5
            expansion_threshold = 2.0
            
            signal = 'neutral'
            if vol_zscore < compression_threshold:
                signal = 'compression'
            elif vol_zscore > expansion_threshold:
                signal = 'expansion'
            
            # Calcular probabilidad de breakout
            breakout_prob = self._calculate_breakout_probability_advanced(
                historical_vol, current_vol)
            
            result = {
                'signal': signal,
                'current_vol': current_vol,
                'avg_vol': avg_vol,
                'vol_zscore': vol_zscore,
                'breakout_probability': breakout_prob,
                'compression_detected': vol_zscore < compression_threshold,
                'expansion_detected': vol_zscore > expansion_threshold
            }
            
            self.logger.info(f"Volatility breakout analysis: {signal} "
                           f"(z-score: {vol_zscore:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in breakout detection: {e}")
            return {}
    
    def _calculate_parkinson_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calcula volatilidad de Parkinson (usa high/low)"""
        log_hl = np.log(price_data['high'] / price_data['low'])
        parkinson_vol = np.sqrt(log_hl ** 2 / (4 * np.log(2)))
        return parkinson_vol.rolling(window=20).mean() * np.sqrt(252)
    
    def _calculate_garman_klass_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calcula volatilidad de Garman-Klass"""
        log_hl = np.log(price_data['high'] / price_data['low'])
        log_co = np.log(price_data['close'] / price_data['open'])
        
        gk_vol = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        return np.sqrt(gk_vol.rolling(window=20).mean() * 252)
    
    def _calculate_rogers_satchell_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calcula volatilidad de Rogers-Satchell"""
        log_ho = np.log(price_data['high'] / price_data['open'])
        log_hc = np.log(price_data['high'] / price_data['close'])
        log_lo = np.log(price_data['low'] / price_data['open'])
        log_lc = np.log(price_data['low'] / price_data['close'])
        
        rs_vol = log_ho * log_hc + log_lo * log_lc
        return np.sqrt(rs_vol.rolling(window=20).mean() * 252)
    
    def _classify_volatility_regime(self, percentile: float) -> str:
        """Clasifica el régimen de volatilidad basado en percentil"""
        if percentile >= self.regime_thresholds['extreme']:
            return 'extreme'
        elif percentile >= self.regime_thresholds['high']:
            return 'high'
        elif percentile >= self.regime_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_regime_duration(self, vol_series: pd.Series, 
                                 current_regime: str) -> int:
        """Calcula la duración del régimen actual"""
        # Implementar lógica para calcular duración
        return 1  # Placeholder
    
    def _calculate_mean_reversion_probability(self, vol_series: pd.Series, 
                                            current_vol: float) -> float:
        """Calcula probabilidad de reversión a la media"""
        mean_vol = vol_series.mean()
        distance_from_mean = abs(current_vol - mean_vol) / vol_series.std()
        
        # Probabilidad aumenta con la distancia de la media
        prob = min(0.95, distance_from_mean * 0.3)
        return prob
    
    def _calculate_breakout_probability(self, vol_series: pd.Series, 
                                      current_vol: float) -> float:
        """Calcula probabilidad de breakout de volatilidad"""
        percentile = stats.percentileofscore(vol_series.dropna(), current_vol) / 100
        
        # Probabilidad de breakout baja en regímenes extremos
        if percentile > 0.9 or percentile < 0.1:
            return 0.2
        else:
            return 0.8
    
    def _calculate_breakout_probability_advanced(self, vol_series: pd.Series, 
                                               current_vol: float) -> float:
        """Versión avanzada del cálculo de probabilidad de breakout"""
        # Implementar modelo más sofisticado
        return self._calculate_breakout_probability(vol_series, current_vol)
```

## 📊 **Métricas y KPIs**

### **Key Levels Detection**
- **Accuracy**: % de niveles que efectivamente actúan como S/R
- **Response Rate**: % de veces que el precio reacciona al tocar un nivel
- **False Positive Rate**: % de niveles identificados incorrectamente
- **Level Strength Distribution**: Distribución de fuerza de niveles

### **Volatility Analysis**
- **Regime Classification Accuracy**: % de clasificación correcta de regímenes
- **Mean Reversion Prediction**: Accuracy de predicciones de reversión
- **Breakout Prediction**: Accuracy de predicciones de breakout
- **Volatility Forecast Error**: Error en forecasting de volatilidad

## 🧪 **Testing Strategy**

### **Unit Tests**
```python
# tests/test_key_levels_detector.py
def test_volume_profile_calculation():
    # Test cálculo de perfil de volumen
    pass

def test_psychological_levels_generation():
    # Test generación de niveles psicológicos
    pass

def test_fibonacci_levels_calculation():
    # Test cálculo de niveles Fibonacci
    pass
```

### **Integration Tests**
```python
# tests/integration/test_enhanced_microstructure.py
def test_key_levels_integration():
    # Test integración con pipeline principal
    pass

def test_volatility_regime_integration():
    # Test integración de regímenes de volatilidad
    pass
```

## 📈 **Performance Targets**

- **Latency**: < 50ms para detección de niveles
- **Memory Usage**: < 500MB para datos históricos
- **CPU Usage**: < 20% durante operación normal
- **Accuracy**: > 75% en identificación de niveles efectivos

## 🚀 **Deployment Checklist**

- [ ] Implementar KeyLevelsDetector
- [ ] Implementar VolatilitySurfaceAnalyzer
- [ ] Crear configuraciones YAML
- [ ] Escribir unit tests
- [ ] Escribir integration tests
- [ ] Documentar APIs
- [ ] Configurar logging
- [ ] Configurar métricas
- [ ] Testing en ambiente de desarrollo
- [ ] Deployment a producción

---

*Documento creado: 2025-08-06 - Stage 2 Enhanced Microstructure*
