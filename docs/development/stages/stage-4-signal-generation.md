# 🎯 Stage 4: Signal Generation Module

## 📋 CHECKLIST: Signal Generation (20-24 horas)

### ✅ Prerrequisitos
- [ ] Feature Engineering funcionando y generando features
- [ ] Features normalizadas fluyendo en tiempo real
- [ ] Datos históricos suficientes para entrenamiento (>1 semana)
- [ ] LightGBM environment configurado

### ✅ Objetivos de la Etapa
Implementar el core ML del sistema de trading:
- **LightGBM Model**: Modelo principal para generación de señales
- **Signal Logic**: Lógica de decisión buy/sell/hold
- **Model Training**: Pipeline de entrenamiento automático
- **Confidence Scoring**: Scoring de confianza de señales
- **Performance Tracking**: Métricas de performance del modelo

## 🏗️ Arquitectura del Módulo

```
modules/signal-generation/
├── __init__.py
├── main.py                    # Entry point del servicio
├── models/
│   ├── __init__.py
│   ├── lightgbm_model.py     # Modelo LightGBM principal
│   ├── model_trainer.py      # Pipeline de entrenamiento
│   ├── model_validator.py    # Validación de modelos
│   └── ensemble_model.py     # Ensemble de modelos (futuro)
├── signals/
│   ├── __init__.py
│   ├── signal_generator.py   # Generador de señales
│   ├── signal_validator.py   # Validación de señales
│   └── confidence_scorer.py  # Scoring de confianza
├── training/
│   ├── __init__.py
│   ├── data_loader.py        # Carga de datos para training
│   ├── feature_selector.py   # Selección de features
│   ├── hyperparameter_tuner.py # Tuning de hiperparámetros
│   └── cross_validator.py    # Cross-validation
├── evaluation/
│   ├── __init__.py
│   ├── performance_metrics.py # Métricas de performance
│   ├── backtester.py         # Backtesting básico
│   └── model_monitor.py      # Monitoreo de modelo
└── config/
    └── model_config.py       # Configuración del modelo
```

## 🚀 Implementación Detallada

### LightGBM Model Core
```python
# modules/signal-generation/models/lightgbm_model.py
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
from pathlib import Path
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class LightGBMTradingModel:
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = {}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train LightGBM model"""
        try:
            # Prepare datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Model parameters
            params = {
                'objective': 'multiclass',
                'num_class': 3,  # 0: sell, 1: hold, 2: buy
                'metric': ['multi_logloss', 'multi_error'],
                'boosting_type': 'gbdt',
                'num_leaves': self.config.get('num_leaves', 31),
                'learning_rate': self.config.get('learning_rate', 0.1),
                'feature_fraction': self.config.get('feature_fraction', 0.8),
                'bagging_fraction': self.config.get('bagging_fraction', 0.8),
                'bagging_freq': self.config.get('bagging_freq', 5),
                'min_child_samples': self.config.get('min_child_samples', 20),
                'lambda_l1': self.config.get('lambda_l1', 0.1),
                'lambda_l2': self.config.get('lambda_l2', 0.1),
                'verbosity': -1,
                'seed': 42
            }
            
            # Train model
            callbacks = [
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
            
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                num_boost_round=self.config.get('num_boost_round', 1000),
                callbacks=callbacks
            )
            
            # Store feature information
            self.feature_names = X_train.columns.tolist()
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importance(importance_type='gain')
            ))
            
            # Calculate training metrics
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            self.training_metrics = {
                'train_accuracy': self._calculate_accuracy(y_train, train_pred),
                'val_accuracy': self._calculate_accuracy(y_val, val_pred),
                'train_precision': self._calculate_precision(y_train, train_pred),
                'val_precision': self._calculate_precision(y_val, val_pred),
                'best_iteration': self.model.best_iteration,
                'feature_count': len(self.feature_names)
            }
            
            logger.info(f"Model trained successfully. Val accuracy: {self.training_metrics['val_accuracy']:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores"""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
            
        try:
            # Get predictions (probabilities)
            probabilities = self.model.predict(X)
            
            # Convert to class predictions
            predictions = np.argmax(probabilities, axis=1)
            
            # Calculate confidence scores (max probability)
            confidence_scores = np.max(probabilities, axis=1)
            
            return predictions, confidence_scores
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
            
    def predict_single(self, features: Dict[str, float]) -> Tuple[int, float, Dict[str, float]]:
        """Make single prediction with detailed output"""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
            
        try:
            # Convert to DataFrame
            X = pd.DataFrame([features])
            
            # Ensure feature order matches training
            if self.feature_names:
                X = X.reindex(columns=self.feature_names, fill_value=0)
                
            # Get prediction
            probabilities = self.model.predict(X)[0]
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # Create probability dict
            prob_dict = {
                'sell_prob': float(probabilities[0]),
                'hold_prob': float(probabilities[1]),
                'buy_prob': float(probabilities[2])
            }
            
            return prediction, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Error making single prediction: {e}")
            return 1, 0.0, {'sell_prob': 0.33, 'hold_prob': 0.34, 'buy_prob': 0.33}
            
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features"""
        if not self.feature_importance:
            return {}
            
        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
        
    def save_model(self, filepath: str):
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'training_metrics': self.training_metrics,
                'config': self.config
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, filepath: str):
        """Load model from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data['feature_importance']
            self.training_metrics = model_data['training_metrics']
            self.config = model_data['config']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def _calculate_accuracy(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate accuracy"""
        y_pred = np.argmax(y_pred_proba, axis=1)
        return np.mean(y_true == y_pred)
        
    def _calculate_precision(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate weighted precision"""
        from sklearn.metrics import precision_score
        y_pred = np.argmax(y_pred_proba, axis=1)
        return precision_score(y_true, y_pred, average='weighted', zero_division=0)
```

### Signal Generator
```python
# modules/signal-generation/signals/signal_generator.py
import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from shared.messaging.message_broker import HybridMessageBroker
from shared.logging.structured_logger import get_logger
from models.lightgbm_model import LightGBMTradingModel
from signals.confidence_scorer import ConfidenceScorer

logger = get_logger(__name__)

@dataclass
class TradingSignal:
    timestamp: int
    symbol: str
    signal: int  # 0: sell, 1: hold, 2: buy
    confidence: float
    probabilities: Dict[str, float]
    features_used: Dict[str, float]
    model_version: str
    signal_strength: str  # 'weak', 'medium', 'strong'

class SignalGenerator:
    def __init__(self, message_broker: HybridMessageBroker, model: LightGBMTradingModel):
        self.message_broker = message_broker
        self.model = model
        self.confidence_scorer = ConfidenceScorer()
        
        # Signal thresholds
        self.confidence_threshold = 0.65  # Minimum confidence for signal
        self.strong_threshold = 0.80      # Strong signal threshold
        self.medium_threshold = 0.70      # Medium signal threshold
        
        # Signal tracking
        self.last_signal = None
        self.signal_count = 0
        self.performance_metrics = {
            'signals_generated': 0,
            'strong_signals': 0,
            'medium_signals': 0,
            'weak_signals': 0
        }
        
    async def start(self):
        """Start signal generation service"""
        try:
            # Subscribe to features stream
            await self.message_broker.subscribe_fast(
                ['features'],
                'signal-generation',
                'generator-1',
                self._process_features
            )
            
            logger.info("Signal generator started")
            
        except Exception as e:
            logger.error(f"Failed to start signal generator: {e}")
            raise
            
    async def _process_features(self, stream: str, message_id: str, features: Dict[str, Any]):
        """Process incoming features and generate signals"""
        try:
            # Generate signal
            signal = await self._generate_signal(features)
            
            if signal and self._should_emit_signal(signal):
                # Publish signal
                await self.message_broker.publish_fast('trading_signals', {
                    'timestamp': signal.timestamp,
                    'symbol': signal.symbol,
                    'signal': signal.signal,
                    'confidence': signal.confidence,
                    'probabilities': signal.probabilities,
                    'signal_strength': signal.signal_strength,
                    'model_version': signal.model_version
                })
                
                # Also publish to Kafka for history
                await self.message_broker.publish_reliable('trading_signals', {
                    'timestamp': signal.timestamp,
                    'symbol': signal.symbol,
                    'signal': signal.signal,
                    'confidence': signal.confidence,
                    'probabilities': signal.probabilities,
                    'features_used': signal.features_used,
                    'signal_strength': signal.signal_strength,
                    'model_version': signal.model_version
                })
                
                # Update metrics
                self._update_metrics(signal)
                
                logger.info(f"Signal generated: {self._signal_to_string(signal.signal)} "
                           f"(confidence: {signal.confidence:.3f}, strength: {signal.signal_strength})")
                
        except Exception as e:
            logger.error(f"Error processing features for signal generation: {e}")
            
    async def _generate_signal(self, features: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal from features"""
        try:
            # Extract feature values (remove metadata)
            feature_dict = {k: v for k, v in features.items() 
                          if k not in ['timestamp', 'symbol']}
            
            # Make prediction
            prediction, confidence, probabilities = self.model.predict_single(feature_dict)
            
            # Calculate additional confidence metrics
            enhanced_confidence = self.confidence_scorer.calculate_confidence(
                base_confidence=confidence,
                probabilities=probabilities,
                features=feature_dict
            )
            
            # Determine signal strength
            signal_strength = self._determine_signal_strength(enhanced_confidence)
            
            signal = TradingSignal(
                timestamp=features.get('timestamp', int(time.time() * 1000)),
                symbol=features.get('symbol', 'BTCUSDT'),
                signal=prediction,
                confidence=enhanced_confidence,
                probabilities=probabilities,
                features_used=feature_dict,
                model_version='lightgbm_v1.0',
                signal_strength=signal_strength
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
            
    def _should_emit_signal(self, signal: TradingSignal) -> bool:
        """Determine if signal should be emitted"""
        # Check confidence threshold
        if signal.confidence < self.confidence_threshold:
            return False
            
        # Don't emit hold signals unless confidence is very high
        if signal.signal == 1 and signal.confidence < 0.85:
            return False
            
        # Check for signal change (avoid spam)
        if self.last_signal:
            # If same signal and recent, don't emit unless confidence significantly higher
            time_diff = signal.timestamp - self.last_signal.timestamp
            if (signal.signal == self.last_signal.signal and 
                time_diff < 30000 and  # 30 seconds
                signal.confidence < self.last_signal.confidence + 0.1):
                return False
                
        return True
        
    def _determine_signal_strength(self, confidence: float) -> str:
        """Determine signal strength based on confidence"""
        if confidence >= self.strong_threshold:
            return 'strong'
        elif confidence >= self.medium_threshold:
            return 'medium'
        else:
            return 'weak'
            
    def _update_metrics(self, signal: TradingSignal):
        """Update performance metrics"""
        self.performance_metrics['signals_generated'] += 1
        
        if signal.signal_strength == 'strong':
            self.performance_metrics['strong_signals'] += 1
        elif signal.signal_strength == 'medium':
            self.performance_metrics['medium_signals'] += 1
        else:
            self.performance_metrics['weak_signals'] += 1
            
        self.last_signal = signal
        
    def _signal_to_string(self, signal: int) -> str:
        """Convert signal to string"""
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return signal_map.get(signal, 'UNKNOWN')
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            'last_signal_time': self.last_signal.timestamp if self.last_signal else None,
            'last_signal_type': self.last_signal.signal if self.last_signal else None,
            'last_signal_confidence': self.last_signal.confidence if self.last_signal else None
        }
```

### Model Training Pipeline
```python
# modules/signal-generation/training/model_trainer.py
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from shared.database.connection import DatabaseManager
from shared.logging.structured_logger import get_logger
from models.lightgbm_model import LightGBMTradingModel
from training.feature_selector import FeatureSelector
from training.hyperparameter_tuner import HyperparameterTuner

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.feature_selector = FeatureSelector()
        self.hyperparameter_tuner = HyperparameterTuner()
        
    async def train_model(self, 
                         training_days: int = 7,
                         validation_days: int = 1,
                         optimize_hyperparams: bool = True) -> LightGBMTradingModel:
        """Train new model with recent data"""
        try:
            logger.info(f"Starting model training with {training_days} days of data")
            
            # Load training data
            X_train, y_train, X_val, y_val = await self._load_training_data(
                training_days, validation_days
            )
            
            logger.info(f"Loaded training data: {len(X_train)} train samples, {len(X_val)} val samples")
            
            # Feature selection
            selected_features = self.feature_selector.select_features(X_train, y_train)
            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features]
            
            logger.info(f"Selected {len(selected_features)} features for training")
            
            # Hyperparameter optimization
            if optimize_hyperparams:
                best_params = await self.hyperparameter_tuner.optimize(
                    X_train_selected, y_train, X_val_selected, y_val
                )
            else:
                best_params = self._get_default_params()
                
            # Train final model
            model = LightGBMTradingModel(best_params)
            training_metrics = model.train(X_train_selected, y_train, X_val_selected, y_val)
            
            # Validate model performance
            if training_metrics['val_accuracy'] < 0.55:  # Should be better than random
                logger.warning(f"Model validation accuracy is low: {training_metrics['val_accuracy']:.4f}")
                
            # Save model
            model_path = f"data/models/lightgbm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            model.save_model(model_path)
            
            logger.info(f"Model training completed. Validation accuracy: {training_metrics['val_accuracy']:.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
            
    async def _load_training_data(self, training_days: int, validation_days: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and prepare training data"""
        try:
            # Calculate date ranges
            end_date = datetime.now()
            val_start_date = end_date - timedelta(days=validation_days)
            train_start_date = val_start_date - timedelta(days=training_days)
            
            # Load features
            features_query = """
                SELECT * FROM trading.features 
                WHERE timestamp >= $1 AND timestamp < $2
                ORDER BY timestamp
            """
            
            train_features = await self.db_manager.fetch(
                features_query,
                int(train_start_date.timestamp() * 1000),
                int(val_start_date.timestamp() * 1000)
            )
            
            val_features = await self.db_manager.fetch(
                features_query,
                int(val_start_date.timestamp() * 1000),
                int(end_date.timestamp() * 1000)
            )
            
            # Convert to DataFrames
            train_df = pd.DataFrame(train_features)
            val_df = pd.DataFrame(val_features)
            
            if train_df.empty or val_df.empty:
                raise ValueError("Insufficient training data")
                
            # Generate labels
            train_labels = await self._generate_labels(train_df)
            val_labels = await self._generate_labels(val_df)
            
            # Prepare feature matrices
            feature_columns = [col for col in train_df.columns 
                             if col not in ['timestamp', 'symbol']]
            
            X_train = train_df[feature_columns]
            X_val = val_df[feature_columns]
            
            return X_train, train_labels, X_val, val_labels
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
            
    async def _generate_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate trading labels from price data"""
        try:
            # Get price data for the same time period
            price_query = """
                SELECT timestamp, price FROM trading.market_data 
                WHERE timestamp >= $1 AND timestamp <= $2 
                AND symbol = 'BTCUSDT'
                ORDER BY timestamp
            """
            
            min_timestamp = features_df['timestamp'].min()
            max_timestamp = features_df['timestamp'].max()
            
            price_data = await self.db_manager.fetch(price_query, min_timestamp, max_timestamp)
            price_df = pd.DataFrame(price_data)
            
            if price_df.empty:
                raise ValueError("No price data available for labeling")
                
            # Calculate future returns (5-minute forward looking)
            price_df['future_return'] = price_df['price'].pct_change(periods=5).shift(-5)
            
            # Create labels based on return thresholds
            labels = []
            for _, feature_row in features_df.iterrows():
                timestamp = feature_row['timestamp']
                
                # Find closest price data
                price_row = price_df.iloc[(price_df['timestamp'] - timestamp).abs().argsort()[:1]]
                
                if not price_row.empty:
                    future_return = price_row['future_return'].iloc[0]
                    
                    if pd.isna(future_return):
                        label = 1  # Hold if no future data
                    elif future_return > 0.002:  # 0.2% threshold for buy
                        label = 2  # Buy
                    elif future_return < -0.002:  # -0.2% threshold for sell
                        label = 0  # Sell
                    else:
                        label = 1  # Hold
                else:
                    label = 1  # Hold if no price data
                    
                labels.append(label)
                
            return pd.Series(labels)
            
        except Exception as e:
            logger.error(f"Error generating labels: {e}")
            raise
            
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default model parameters"""
        return {
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'num_boost_round': 500
        }
```

## ✅ Testing y Validación

### Model Performance Tests
```python
# tests/unit/test_signal_generation.py
import pytest
import pandas as pd
import numpy as np
from modules.signal_generation.models.lightgbm_model import LightGBMTradingModel

@pytest.mark.asyncio
async def test_model_training():
    """Test model training pipeline"""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
    y_train = pd.Series(np.random.randint(0, 3, n_samples))
    
    X_val = pd.DataFrame(np.random.randn(200, n_features))
    y_val = pd.Series(np.random.randint(0, 3, 200))
    
    # Train model
    config = {'num_boost_round': 10, 'num_leaves': 10}
    model = LightGBMTradingModel(config)
    
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    # Check metrics
    assert 'val_accuracy' in metrics
    assert metrics['val_accuracy'] > 0.2  # Should be better than random
    
def test_signal_generation():
    """Test signal generation logic"""
    from modules.signal_generation.signals.signal_generator import SignalGenerator
    from unittest.mock import Mock
    
    # Mock dependencies
    message_broker = Mock()
    model = Mock()
    model.predict_single.return_value = (2, 0.75, {'sell_prob': 0.1, 'hold_prob': 0.15, 'buy_prob': 0.75})
    
    generator = SignalGenerator(message_broker, model)
    
    # Test signal generation
    features = {
        'timestamp': 1234567890,
        'symbol': 'BTCUSDT',
        'imbalance_ratio': 0.5,
        'sentiment_compound': 0.3
    }
    
    signal = asyncio.run(generator._generate_signal(features))
    
    assert signal is not None
    assert signal.signal == 2  # Buy signal
    assert signal.confidence > 0.6
```

## ✅ Checklist de Completitud

### Model Development
- [ ] LightGBM model implementado
- [ ] Training pipeline funcionando
- [ ] Feature selection automática
- [ ] Hyperparameter tuning
- [ ] Model validation
- [ ] Model persistence (save/load)

### Signal Generation
- [ ] Real-time signal generation
- [ ] Confidence scoring
- [ ] Signal strength classification
- [ ] Signal filtering logic
- [ ] Performance tracking
- [ ] Signal validation

### Training Pipeline
- [ ] Automated data loading
- [ ] Label generation from price data
- [ ] Cross-validation
- [ ] Model evaluation metrics
- [ ] Automated retraining
- [ ] Model versioning

### Performance Monitoring
- [ ] Real-time metrics tracking
- [ ] Model drift detection
- [ ] Signal quality monitoring
- [ ] Performance benchmarking
- [ ] Alerting on poor performance

**Tiempo estimado**: 20-24 horas  
**Responsable**: ML Engineer  
**Dependencias**: Feature Engineering funcionando

---

**Next Step**: [Stage 5: Risk Management](./stage-5-risk-management.md)
