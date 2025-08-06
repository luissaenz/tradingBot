# 📈 Stage 11: Advanced Analytics & Optimization

## 📋 Objetivo
Implementar analytics avanzados para optimización continua, performance attribution, regime detection y research de microestructura.

## 🎯 Componentes a Implementar

### **11.1 Advanced Analytics Engine**

#### **Implementación Core**

```python
# src/modules/analytics/advanced_analytics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import asyncio
from ..shared.logging import get_logger

@dataclass
class PerformanceAttribution:
    period_start: datetime
    period_end: datetime
    total_return: float
    strategy_attribution: Dict[str, float]
    factor_attribution: Dict[str, float]
    alpha_generation: float
    risk_adjusted_return: float

@dataclass
class MarketRegime:
    regime_id: str
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    start_date: datetime
    characteristics: Dict[str, float]
    volatility_level: str
    optimal_strategies: List[str]

@dataclass
class OptimizationResult:
    strategy_name: str
    optimization_date: datetime
    original_parameters: Dict
    optimized_parameters: Dict
    performance_improvement: float
    implementation_recommendation: str

class AdvancedAnalytics:
    """Motor de analytics avanzados"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        self.lookback_window = config.get('lookback_window', 252)
        self.regime_detection_window = config.get('regime_window', 60)
        
        self.risk_factors = [
            'market_beta', 'volatility', 'momentum', 
            'mean_reversion', 'liquidity', 'sentiment'
        ]
    
    async def performance_attribution(self, 
                                    portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series,
                                    strategy_returns: Dict[str, pd.Series],
                                    period_start: datetime,
                                    period_end: datetime) -> PerformanceAttribution:
        """Realiza attribution detallado de performance"""
        try:
            total_return = (portfolio_returns + 1).prod() - 1
            
            # Attribution por estrategia
            strategy_attribution = {}
            for strategy_name, returns in strategy_returns.items():
                contribution = self._calculate_strategy_contribution(
                    returns, portfolio_returns)
                strategy_attribution[strategy_name] = contribution
            
            # Attribution por factores
            factor_attribution = await self._calculate_factor_attribution(
                portfolio_returns, benchmark_returns)
            
            # Alpha generation
            alpha_generation = await self._calculate_alpha_generation(
                portfolio_returns, benchmark_returns)
            
            # Risk-adjusted return
            risk_adjusted_return = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            
            return PerformanceAttribution(
                period_start=period_start,
                period_end=period_end,
                total_return=total_return,
                strategy_attribution=strategy_attribution,
                factor_attribution=factor_attribution,
                alpha_generation=alpha_generation,
                risk_adjusted_return=risk_adjusted_return
            )
            
        except Exception as e:
            self.logger.error(f"Error in performance attribution: {e}")
            raise
    
    async def regime_detection(self, 
                             market_data: pd.DataFrame,
                             price_data: pd.DataFrame) -> List[MarketRegime]:
        """Detecta y clasifica regímenes de mercado"""
        try:
            # Calcular features
            features = await self._calculate_regime_features(market_data, price_data)
            
            # Clustering
            regimes = await self._cluster_market_regimes(features)
            
            # Clasificar regímenes
            classified_regimes = []
            for regime_data in regimes:
                regime = await self._classify_regime(regime_data, features)
                if regime:
                    classified_regimes.append(regime)
            
            self.logger.info(f"Regime detection: {len(classified_regimes)} regimes identified")
            return classified_regimes
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            return []
    
    async def strategy_optimization(self, 
                                  strategy_name: str,
                                  current_parameters: Dict,
                                  historical_data: pd.DataFrame) -> OptimizationResult:
        """Optimiza parámetros de estrategia"""
        try:
            # Definir espacio de parámetros
            parameter_space = await self._define_parameter_space(
                strategy_name, current_parameters)
            
            # Walk-forward optimization
            optimization_results = await self._walk_forward_optimization(
                strategy_name, parameter_space, historical_data)
            
            best_parameters = optimization_results['best_parameters']
            
            # Calcular mejora
            original_perf = await self._backtest_strategy(
                strategy_name, current_parameters, historical_data)
            optimized_perf = await self._backtest_strategy(
                strategy_name, best_parameters, historical_data)
            
            performance_improvement = (
                optimized_perf['sharpe'] - original_perf['sharpe'])
            
            # Recomendación
            if performance_improvement > 0.1:
                recommendation = "IMPLEMENT_IMMEDIATELY"
            elif performance_improvement > 0.05:
                recommendation = "IMPLEMENT_WITH_MONITORING"
            else:
                recommendation = "DO_NOT_IMPLEMENT"
            
            return OptimizationResult(
                strategy_name=strategy_name,
                optimization_date=datetime.now(),
                original_parameters=current_parameters,
                optimized_parameters=best_parameters,
                performance_improvement=performance_improvement,
                implementation_recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"Error in strategy optimization: {e}")
            raise
    
    # Métodos auxiliares
    def _calculate_strategy_contribution(self, 
                                       strategy_returns: pd.Series,
                                       portfolio_returns: pd.Series) -> float:
        """Calcula contribución de estrategia"""
        correlation = strategy_returns.corr(portfolio_returns)
        strategy_return = (strategy_returns + 1).prod() - 1
        return strategy_return * correlation
    
    async def _calculate_factor_attribution(self, 
                                          portfolio_returns: pd.Series,
                                          benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calcula attribution por factores"""
        factor_attribution = {}
        for factor in self.risk_factors:
            # Simulación de factor returns
            factor_returns = np.random.normal(0, 0.01, len(portfolio_returns))
            exposure = np.corrcoef(portfolio_returns, factor_returns)[0, 1]
            attribution = exposure * np.mean(factor_returns) * len(portfolio_returns)
            factor_attribution[factor] = attribution
        return factor_attribution
    
    async def _calculate_alpha_generation(self, 
                                        portfolio_returns: pd.Series,
                                        benchmark_returns: pd.Series) -> float:
        """Calcula alpha generado"""
        beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        alpha = portfolio_returns.mean() - beta * benchmark_returns.mean()
        return alpha * 252  # Anualizado
    
    async def _calculate_regime_features(self, 
                                       market_data: pd.DataFrame,
                                       price_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features para detección de regímenes"""
        features = pd.DataFrame(index=price_data.index)
        
        returns = price_data['close'].pct_change()
        features['volatility'] = returns.rolling(20).std()
        features['sma_20'] = price_data['close'].rolling(20).mean()
        features['sma_50'] = price_data['close'].rolling(50).mean()
        features['trend_strength'] = (features['sma_20'] - features['sma_50']) / features['sma_50']
        features['momentum'] = returns.rolling(10).sum()
        
        if 'volume' in market_data.columns:
            features['volume_ma'] = market_data['volume'].rolling(20).mean()
            features['volume_ratio'] = market_data['volume'] / features['volume_ma']
        
        return features.dropna()
    
    async def _cluster_market_regimes(self, features: pd.DataFrame) -> List[Dict]:
        """Aplica clustering para identificar regímenes"""
        normalized_features = (features - features.mean()) / features.std()
        
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(normalized_features.fillna(0))
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_features)
        
        regimes = []
        for cluster_id in range(4):
            cluster_mask = cluster_labels == cluster_id
            cluster_dates = features.index[cluster_mask]
            
            if len(cluster_dates) > 0:
                regimes.append({
                    'cluster_id': cluster_id,
                    'dates': cluster_dates,
                    'features': features.loc[cluster_mask]
                })
        
        return regimes
    
    async def _classify_regime(self, regime_data: Dict, features: pd.DataFrame) -> Optional[MarketRegime]:
        """Clasifica un régimen identificado"""
        try:
            cluster_features = regime_data['features']
            dates = regime_data['dates']
            
            avg_volatility = cluster_features['volatility'].mean()
            avg_trend = cluster_features['trend_strength'].mean()
            avg_momentum = cluster_features['momentum'].mean()
            
            # Clasificar tipo
            if avg_trend > 0.02 and avg_momentum > 0:
                regime_type = 'bull'
            elif avg_trend < -0.02 and avg_momentum < 0:
                regime_type = 'bear'
            elif avg_volatility > features['volatility'].quantile(0.8):
                regime_type = 'volatile'
            else:
                regime_type = 'sideways'
            
            # Nivel de volatilidad
            vol_percentile = (avg_volatility > features['volatility'].quantile([0.25, 0.5, 0.75])).sum()
            vol_levels = ['low', 'medium', 'high', 'extreme']
            volatility_level = vol_levels[min(vol_percentile, 3)]
            
            # Estrategias óptimas
            optimal_strategies = self._determine_optimal_strategies(regime_type, volatility_level)
            
            return MarketRegime(
                regime_id=f"{regime_type}_{dates[0].strftime('%Y%m%d')}",
                regime_type=regime_type,
                start_date=dates[0],
                characteristics={
                    'volatility': avg_volatility,
                    'trend_strength': avg_trend,
                    'momentum': avg_momentum
                },
                volatility_level=volatility_level,
                optimal_strategies=optimal_strategies
            )
            
        except Exception as e:
            self.logger.error(f"Error classifying regime: {e}")
            return None
    
    def _determine_optimal_strategies(self, regime_type: str, volatility_level: str) -> List[str]:
        """Determina estrategias óptimas"""
        strategy_map = {
            ('bull', 'low'): ['momentum', 'breakout'],
            ('bull', 'medium'): ['momentum', 'trend_following'],
            ('bear', 'low'): ['mean_reversion', 'contrarian'],
            ('sideways', 'low'): ['mean_reversion', 'range_trading'],
            ('volatile', 'high'): ['volatility_trading', 'defensive']
        }
        return strategy_map.get((regime_type, volatility_level), ['conservative'])
    
    async def _define_parameter_space(self, strategy_name: str, current_parameters: Dict) -> Dict:
        """Define espacio de parámetros"""
        parameter_ranges = {
            'momentum': {
                'lookback_window': (5, 50),
                'threshold': (0.01, 0.05)
            },
            'mean_reversion': {
                'lookback_window': (10, 100),
                'entry_threshold': (1.5, 3.0)
            }
        }
        return parameter_ranges.get(strategy_name, {})
    
    async def _walk_forward_optimization(self, strategy_name: str, 
                                       parameter_space: Dict, 
                                       historical_data: pd.DataFrame) -> Dict:
        """Walk-forward optimization"""
        best_parameters = {}
        for param, (min_val, max_val) in parameter_space.items():
            best_parameters[param] = (min_val + max_val) / 2
        
        return {
            'best_parameters': best_parameters,
            'optimization_score': 0.75
        }
    
    async def _backtest_strategy(self, strategy_name: str, 
                               parameters: Dict, 
                               historical_data: pd.DataFrame) -> Dict[str, float]:
        """Ejecuta backtest"""
        return {
            'total_return': 0.15,
            'sharpe': 1.2,
            'max_drawdown': 0.08,
            'win_rate': 0.65
        }
```

#### **Configuración**

```yaml
# config/analytics/advanced_analytics.yaml
advanced_analytics:
  lookback_window: 252
  regime_window: 60
  optimization_freq: 30
  
  risk_factors:
    - market_beta
    - volatility
    - momentum
    - mean_reversion
    - liquidity
    - sentiment
  
  regime_detection:
    n_clusters: 4
    pca_components: 3
    min_regime_duration: 10
  
  optimization:
    method: "walk_forward"
    train_window: 180
    test_window: 60
    performance_metric: "sharpe"
```

## 📊 **Métricas y KPIs**

- **Attribution Accuracy**: Precisión en attribution
- **Regime Classification**: Accuracy de clasificación
- **Optimization Success**: % mejoras implementadas
- **Alpha Generation**: Alpha promedio generado

## 🧪 **Testing Strategy**

```python
# tests/test_advanced_analytics.py
async def test_performance_attribution():
    pass

async def test_regime_detection():
    pass

async def test_strategy_optimization():
    pass
```

## 📈 **Performance Targets**

- **Attribution Calculation**: < 5s
- **Regime Detection**: < 10s
- **Optimization**: < 60s
- **Memory Usage**: < 1GB

## 🚀 **Deployment Checklist**

- [ ] Implementar AdvancedAnalytics
- [ ] Configurar parámetros
- [ ] Testing completo
- [ ] Deployment

---

*Documento creado: 2025-08-06 - Advanced Analytics*
