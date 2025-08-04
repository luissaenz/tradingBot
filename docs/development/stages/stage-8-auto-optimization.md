# 🔄 Stage 8: Auto-Optimization Module

## 📋 CHECKLIST: Auto-Optimization (20-24 horas)

### ✅ Prerrequisitos
- [ ] Sistema completo funcionando end-to-end
- [ ] Datos históricos suficientes (>2 semanas)
- [ ] Monitoring y métricas operativas
- [ ] Performance baseline establecido

### ✅ Objetivos de la Etapa
Implementar optimización automática del sistema:
- **Model Retraining**: Reentrenamiento automático de modelos
- **Hyperparameter Tuning**: Optimización de hiperparámetros
- **Feature Selection**: Selección automática de features
- **A/B Testing**: Testing de estrategias
- **Performance Optimization**: Optimización basada en performance

## 🏗️ Arquitectura del Módulo

```
modules/auto-optimization/
├── __init__.py
├── main.py                    # Entry point del servicio
├── model_optimization/
│   ├── __init__.py
│   ├── auto_trainer.py       # Reentrenamiento automático
│   ├── hyperparameter_optimizer.py # Optimización hiperparámetros
│   ├── feature_optimizer.py  # Optimización de features
│   └── model_validator.py    # Validación de modelos
├── strategy_optimization/
│   ├── __init__.py
│   ├── strategy_tuner.py     # Tuning de estrategias
│   ├── ab_tester.py          # A/B testing
│   └── performance_analyzer.py # Análisis de performance
├── risk_optimization/
│   ├── __init__.py
│   ├── risk_tuner.py         # Optimización de riesgo
│   └── parameter_optimizer.py # Optimización de parámetros
└── config/
    └── optimization_config.py # Configuración
```

## 🚀 Implementación Detallada

### Auto Trainer
```python
# modules/auto-optimization/model_optimization/auto_trainer.py
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from shared.database.connection import DatabaseManager
from shared.messaging.message_broker import HybridMessageBroker
from shared.logging.structured_logger import get_logger
from model_optimization.hyperparameter_optimizer import HyperparameterOptimizer
from model_optimization.feature_optimizer import FeatureOptimizer
from model_optimization.model_validator import ModelValidator

logger = get_logger(__name__)

class AutoTrainer:
    def __init__(self, message_broker: HybridMessageBroker, 
                 db_manager: DatabaseManager, config: Dict[str, Any]):
        self.message_broker = message_broker
        self.db_manager = db_manager
        self.config = config
        
        # Optimization components
        self.hyperparameter_optimizer = HyperparameterOptimizer(config)
        self.feature_optimizer = FeatureOptimizer(config)
        self.model_validator = ModelValidator(config)
        
        # Training schedule
        self.retrain_interval = config.get('retrain_interval_hours', 24)  # 24 hours
        self.performance_threshold = config.get('performance_threshold', 0.02)  # 2% degradation
        
        # State tracking
        self.last_training_time = 0
        self.current_model_performance = {}
        self.optimization_history = []
        
    async def start(self):
        """Start auto-optimization service"""
        try:
            # Subscribe to performance metrics
            await self.message_broker.subscribe_fast(
                ['performance_metrics'],
                'auto-optimization',
                'trainer-1',
                self._process_performance_metrics
            )
            
            # Start optimization loop
            asyncio.create_task(self._optimization_loop())
            
            logger.info("Auto-trainer started")
            
        except Exception as e:
            logger.error(f"Failed to start auto-trainer: {e}")
            raise
            
    async def _optimization_loop(self):
        """Main optimization loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check if retraining is needed
                if await self._should_retrain():
                    logger.info("Starting automatic model retraining...")
                    await self._perform_optimization()
                    
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                
    async def _should_retrain(self) -> bool:
        """Determine if model retraining is needed"""
        try:
            current_time = time.time()
            
            # Time-based retraining
            if current_time - self.last_training_time > (self.retrain_interval * 3600):
                logger.info("Time-based retraining triggered")
                return True
                
            # Performance-based retraining
            if await self._performance_degraded():
                logger.info("Performance-based retraining triggered")
                return True
                
            # Data drift detection
            if await self._data_drift_detected():
                logger.info("Data drift retraining triggered")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
            return False
            
    async def _performance_degraded(self) -> bool:
        """Check if model performance has degraded"""
        try:
            # Get recent performance metrics
            recent_performance = await self._get_recent_performance()
            
            if not recent_performance or not self.current_model_performance:
                return False
                
            # Compare key metrics
            metrics_to_check = ['accuracy', 'precision', 'sharpe_ratio']
            
            for metric in metrics_to_check:
                if metric in recent_performance and metric in self.current_model_performance:
                    current = recent_performance[metric]
                    baseline = self.current_model_performance[metric]
                    
                    if baseline > 0:
                        degradation = (baseline - current) / baseline
                        if degradation > self.performance_threshold:
                            logger.warning(f"Performance degradation detected in {metric}: {degradation:.3f}")
                            return True
                            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return False
            
    async def _data_drift_detected(self) -> bool:
        """Detect data drift in features"""
        try:
            # Get recent feature statistics
            recent_stats = await self._get_recent_feature_stats()
            baseline_stats = await self._get_baseline_feature_stats()
            
            if not recent_stats or not baseline_stats:
                return False
                
            # Calculate statistical distance (simplified KL divergence)
            drift_threshold = 0.1
            
            for feature in recent_stats:
                if feature in baseline_stats:
                    recent_mean = recent_stats[feature]['mean']
                    recent_std = recent_stats[feature]['std']
                    baseline_mean = baseline_stats[feature]['mean']
                    baseline_std = baseline_stats[feature]['std']
                    
                    # Simple drift detection using mean shift
                    if baseline_std > 0:
                        mean_shift = abs(recent_mean - baseline_mean) / baseline_std
                        if mean_shift > drift_threshold:
                            logger.warning(f"Data drift detected in feature {feature}: {mean_shift:.3f}")
                            return True
                            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            return False
            
    async def _perform_optimization(self):
        """Perform complete model optimization"""
        try:
            optimization_start = time.time()
            
            # Load training data
            training_data = await self._load_training_data()
            if training_data is None:
                logger.error("Failed to load training data for optimization")
                return
                
            X_train, y_train, X_val, y_val = training_data
            
            # Feature optimization
            logger.info("Starting feature optimization...")
            optimal_features = await self.feature_optimizer.optimize_features(X_train, y_train, X_val, y_val)
            
            # Apply feature selection
            X_train_opt = X_train[optimal_features]
            X_val_opt = X_val[optimal_features]
            
            # Hyperparameter optimization
            logger.info("Starting hyperparameter optimization...")
            optimal_params = await self.hyperparameter_optimizer.optimize(X_train_opt, y_train, X_val_opt, y_val)
            
            # Train optimized model
            logger.info("Training optimized model...")
            from modules.signal_generation.models.lightgbm_model import LightGBMTradingModel
            
            optimized_model = LightGBMTradingModel(optimal_params)
            training_metrics = optimized_model.train(X_train_opt, y_train, X_val_opt, y_val)
            
            # Validate new model
            validation_result = await self.model_validator.validate_model(
                optimized_model, X_val_opt, y_val, self.current_model_performance
            )
            
            if validation_result['approved']:
                # Deploy new model
                await self._deploy_optimized_model(optimized_model, optimal_features, optimal_params)
                
                # Update performance baseline
                self.current_model_performance = training_metrics
                self.last_training_time = time.time()
                
                # Record optimization
                optimization_record = {
                    'timestamp': int(time.time() * 1000),
                    'duration': time.time() - optimization_start,
                    'features_selected': len(optimal_features),
                    'optimal_params': optimal_params,
                    'validation_metrics': training_metrics,
                    'improvement': validation_result['improvement'],
                    'deployed': True
                }
                
                self.optimization_history.append(optimization_record)
                
                # Publish optimization result
                await self.message_broker.publish_reliable('model_optimization', optimization_record)
                
                logger.info(f"Model optimization completed successfully in {optimization_record['duration']:.1f}s")
                logger.info(f"Performance improvement: {validation_result['improvement']:.3f}")
                
            else:
                logger.warning(f"Optimized model failed validation: {validation_result['reasons']}")
                
        except Exception as e:
            logger.error(f"Error in model optimization: {e}")
            
    async def _load_training_data(self) -> Optional[tuple]:
        """Load training data for optimization"""
        try:
            # Get data from last 14 days for training, last 2 days for validation
            end_time = datetime.now()
            val_start_time = end_time - timedelta(days=2)
            train_start_time = end_time - timedelta(days=14)
            
            # Load features
            features_query = """
                SELECT * FROM trading.features 
                WHERE timestamp >= $1 AND timestamp < $2
                ORDER BY timestamp
            """
            
            train_features = await self.db_manager.fetch(
                features_query,
                int(train_start_time.timestamp() * 1000),
                int(val_start_time.timestamp() * 1000)
            )
            
            val_features = await self.db_manager.fetch(
                features_query,
                int(val_start_time.timestamp() * 1000),
                int(end_time.timestamp() * 1000)
            )
            
            if not train_features or not val_features:
                logger.error("Insufficient training data")
                return None
                
            # Convert to DataFrames
            train_df = pd.DataFrame(train_features)
            val_df = pd.DataFrame(val_features)
            
            # Generate labels (simplified - would use actual labeling logic)
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
            return None
            
    async def _deploy_optimized_model(self, model, features: List[str], params: Dict[str, Any]):
        """Deploy optimized model"""
        try:
            # Save model
            model_path = f"data/models/optimized_model_{int(time.time())}.joblib"
            model.save_model(model_path)
            
            # Publish model update
            model_update = {
                'timestamp': int(time.time() * 1000),
                'model_path': model_path,
                'selected_features': features,
                'model_params': params,
                'version': f"auto_optimized_{int(time.time())}"
            }
            
            await self.message_broker.publish_fast('model_update', model_update)
            await self.message_broker.publish_reliable('model_update', model_update)
            
            logger.info(f"Optimized model deployed: {model_path}")
            
        except Exception as e:
            logger.error(f"Error deploying optimized model: {e}")
```

### A/B Testing Framework
```python
# modules/auto-optimization/strategy_optimization/ab_tester.py
import asyncio
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from shared.messaging.message_broker import HybridMessageBroker
from shared.database.connection import DatabaseManager
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class TestStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"

@dataclass
class ABTest:
    id: str
    name: str
    description: str
    start_time: int
    end_time: Optional[int]
    status: TestStatus
    control_config: Dict[str, Any]
    treatment_config: Dict[str, Any]
    traffic_split: float  # 0.5 = 50/50 split
    metrics: Dict[str, Any]

class ABTester:
    def __init__(self, message_broker: HybridMessageBroker, db_manager: DatabaseManager):
        self.message_broker = message_broker
        self.db_manager = db_manager
        
        # Active tests
        self.active_tests = {}
        self.test_assignments = {}  # User/session -> test variant
        
    async def start(self):
        """Start A/B testing service"""
        try:
            # Subscribe to trading signals for test routing
            await self.message_broker.subscribe_fast(
                ['trading_signals'],
                'ab-testing',
                'tester-1',
                self._route_signal_for_testing
            )
            
            # Load active tests
            await self._load_active_tests()
            
            logger.info("A/B tester started")
            
        except Exception as e:
            logger.error(f"Failed to start A/B tester: {e}")
            raise
            
    async def create_test(self, test_config: Dict[str, Any]) -> str:
        """Create new A/B test"""
        try:
            import time
            import uuid
            
            test_id = str(uuid.uuid4())
            
            test = ABTest(
                id=test_id,
                name=test_config['name'],
                description=test_config['description'],
                start_time=int(time.time() * 1000),
                end_time=test_config.get('end_time'),
                status=TestStatus.RUNNING,
                control_config=test_config['control_config'],
                treatment_config=test_config['treatment_config'],
                traffic_split=test_config.get('traffic_split', 0.5),
                metrics={}
            )
            
            self.active_tests[test_id] = test
            
            # Store in database
            await self._store_test(test)
            
            logger.info(f"A/B test created: {test.name} ({test_id})")
            
            return test_id
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            raise
            
    async def _route_signal_for_testing(self, stream: str, message_id: str, signal_data: Dict[str, Any]):
        """Route signals through A/B tests"""
        try:
            # Check if any active tests apply to this signal
            for test_id, test in self.active_tests.items():
                if test.status == TestStatus.RUNNING:
                    # Determine test variant
                    variant = self._assign_variant(signal_data, test)
                    
                    # Apply test configuration
                    modified_signal = self._apply_test_config(signal_data, test, variant)
                    
                    # Publish modified signal
                    await self.message_broker.publish_fast(
                        f'testing_signals_{test_id}',
                        {
                            **modified_signal,
                            'test_id': test_id,
                            'variant': variant
                        }
                    )
                    
                    # Track test metrics
                    await self._track_test_metrics(test_id, variant, signal_data)
                    
        except Exception as e:
            logger.error(f"Error routing signal for testing: {e}")
            
    def _assign_variant(self, signal_data: Dict[str, Any], test: ABTest) -> str:
        """Assign signal to test variant"""
        # Use signal timestamp for consistent assignment
        signal_hash = hash(str(signal_data.get('timestamp', 0)))
        
        # Normalize hash to 0-1 range
        normalized_hash = (signal_hash % 1000000) / 1000000
        
        if normalized_hash < test.traffic_split:
            return 'treatment'
        else:
            return 'control'
            
    def _apply_test_config(self, signal_data: Dict[str, Any], test: ABTest, variant: str) -> Dict[str, Any]:
        """Apply test configuration to signal"""
        modified_signal = signal_data.copy()
        
        if variant == 'treatment':
            config = test.treatment_config
        else:
            config = test.control_config
            
        # Apply configuration modifications
        for key, value in config.items():
            if key == 'confidence_threshold':
                # Modify confidence threshold
                modified_signal['confidence_threshold'] = value
            elif key == 'position_size_multiplier':
                # Modify position sizing
                modified_signal['position_size_multiplier'] = value
            # Add more configuration options as needed
            
        return modified_signal
        
    async def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} not found")
                
            test = self.active_tests[test_id]
            
            # Get test metrics from database
            control_metrics = await self._get_variant_metrics(test_id, 'control')
            treatment_metrics = await self._get_variant_metrics(test_id, 'treatment')
            
            # Calculate statistical significance
            significance_result = self._calculate_statistical_significance(
                control_metrics, treatment_metrics
            )
            
            # Determine winner
            winner = self._determine_winner(control_metrics, treatment_metrics, significance_result)
            
            results = {
                'test_id': test_id,
                'test_name': test.name,
                'control_metrics': control_metrics,
                'treatment_metrics': treatment_metrics,
                'statistical_significance': significance_result,
                'winner': winner,
                'confidence_level': significance_result.get('confidence_level', 0),
                'recommendation': self._generate_recommendation(winner, significance_result)
            }
            
            logger.info(f"A/B test analysis completed for {test.name}: Winner = {winner}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing test results: {e}")
            return {}
```

## ✅ Testing y Validación

### Optimization Tests
```python
# tests/unit/test_auto_optimization.py
import pytest
from unittest.mock import Mock, AsyncMock
from modules.auto_optimization.model_optimization.auto_trainer import AutoTrainer

@pytest.mark.asyncio
async def test_performance_degradation_detection():
    """Test performance degradation detection"""
    message_broker = Mock()
    db_manager = Mock()
    
    config = {
        'performance_threshold': 0.05,  # 5% degradation threshold
        'retrain_interval_hours': 24
    }
    
    trainer = AutoTrainer(message_broker, db_manager, config)
    
    # Set baseline performance
    trainer.current_model_performance = {
        'accuracy': 0.70,
        'precision': 0.68,
        'sharpe_ratio': 1.5
    }
    
    # Mock recent performance (degraded)
    trainer._get_recent_performance = AsyncMock(return_value={
        'accuracy': 0.65,  # 7.1% degradation
        'precision': 0.64,  # 5.9% degradation
        'sharpe_ratio': 1.4  # 6.7% degradation
    })
    
    # Should detect degradation
    degraded = await trainer._performance_degraded()
    assert degraded is True
```

## ✅ Checklist de Completitud

### Model Optimization
- [ ] Automatic retraining pipeline
- [ ] Hyperparameter optimization
- [ ] Feature selection automation
- [ ] Model validation framework
- [ ] Performance monitoring
- [ ] Model deployment automation

### Strategy Optimization
- [ ] A/B testing framework
- [ ] Strategy parameter tuning
- [ ] Performance comparison
- [ ] Statistical significance testing
- [ ] Winner determination
- [ ] Automated rollout

### Risk Optimization
- [ ] Risk parameter tuning
- [ ] Drawdown optimization
- [ ] Position sizing optimization
- [ ] Stop loss optimization
- [ ] Risk-adjusted returns
- [ ] Portfolio optimization

### System Integration
- [ ] End-to-end optimization
- [ ] Performance tracking
- [ ] Optimization history
- [ ] Rollback capabilities
- [ ] Monitoring integration
- [ ] Alert system

**Tiempo estimado**: 20-24 horas  
**Responsable**: ML Engineer  
**Dependencias**: Sistema completo funcionando

---

**Final Step**: [Production Deployment](../operations/deployment.md)
