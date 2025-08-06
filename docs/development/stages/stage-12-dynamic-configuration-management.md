# ⚙️ Stage 12: Dynamic Configuration Management System

## 📋 Objetivo
Implementar un sistema de gestión de configuración dinámica que permita actualizar parámetros en tiempo real sin reiniciar el sistema, con optimización automática y audit trail completo.

## 🎯 Componentes a Implementar

### **12.1 Dynamic Configuration Manager**

#### **Database Schema**

```sql
-- Tabla principal de parámetros
CREATE TABLE trading_parameters (
    id SERIAL PRIMARY KEY,
    module_name VARCHAR(50) NOT NULL,
    parameter_name VARCHAR(100) NOT NULL,
    parameter_value JSONB NOT NULL,
    parameter_type VARCHAR(20) NOT NULL,
    min_value DECIMAL(15,8),
    max_value DECIMAL(15,8),
    valid_from TIMESTAMP NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMP,
    created_by VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    description TEXT,
    optimization_enabled BOOLEAN DEFAULT false,
    UNIQUE(module_name, parameter_name, valid_from)
);

-- Historial de cambios
CREATE TABLE parameter_changes (
    id SERIAL PRIMARY KEY,
    parameter_id INTEGER REFERENCES trading_parameters(id),
    old_value JSONB,
    new_value JSONB,
    change_reason VARCHAR(200),
    changed_by VARCHAR(50),
    changed_at TIMESTAMP DEFAULT NOW(),
    performance_before DECIMAL(10,6),
    performance_after DECIMAL(10,6)
);

-- A/B Testing
CREATE TABLE ab_test_configs (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    module_name VARCHAR(50) NOT NULL,
    parameter_name VARCHAR(100) NOT NULL,
    control_value JSONB NOT NULL,
    test_value JSONB NOT NULL,
    traffic_split DECIMAL(3,2) DEFAULT 0.5,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    created_by VARCHAR(50)
);
```

#### **Implementación Core**

```python
# src/modules/config/dynamic_config_manager.py
import asyncio
import json
import redis
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from ..shared.logging import get_logger
from ..shared.database import DatabaseManager

class ChangeReason(Enum):
    MANUAL = "manual"
    AUTO_OPTIMIZATION = "auto_optimization"
    AB_TEST = "ab_test"
    ROLLBACK = "rollback"

@dataclass
class ParameterDefinition:
    module_name: str
    parameter_name: str
    current_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    optimization_enabled: bool = False

class DynamicConfigManager:
    """Gestor de configuración dinámica"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        self.db = DatabaseManager()
        
        # Redis para cache
        self.redis = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Cache local
        self.local_cache = {}
        self.cache_ttl = 60
        
        # Suscriptores
        self.change_subscribers = {}
        
        # A/B tests activos
        self.ab_tests = {}
    
    async def initialize(self):
        """Inicializa el sistema"""
        await self._load_active_configurations()
        await self._load_active_ab_tests()
        self.logger.info("Dynamic Configuration Manager initialized")
    
    async def get_parameter(self, module_name: str, parameter_name: str, 
                          user_id: Optional[str] = None) -> Any:
        """Obtiene valor de parámetro con soporte A/B testing"""
        cache_key = f"{module_name}:{parameter_name}"
        
        # Verificar A/B test
        if user_id and cache_key in self.ab_tests:
            ab_config = self.ab_tests[cache_key]
            if self._should_use_test_value(user_id, ab_config):
                return ab_config['test_value']
            return ab_config['control_value']
        
        # Cache local
        if cache_key in self.local_cache:
            cached = self.local_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(seconds=self.cache_ttl):
                return cached['value']
        
        # Redis cache
        redis_value = self.redis.get(cache_key)
        if redis_value:
            value = json.loads(redis_value)
            self.local_cache[cache_key] = {
                'value': value,
                'timestamp': datetime.now()
            }
            return value
        
        # Base de datos
        value = await self._load_parameter_from_db(module_name, parameter_name)
        await self._update_caches(cache_key, value)
        return value
    
    async def update_parameter(self, 
                             module_name: str,
                             parameter_name: str,
                             new_value: Any,
                             changed_by: str,
                             change_reason: ChangeReason = ChangeReason.MANUAL) -> bool:
        """Actualiza parámetro con hot-reload"""
        try:
            # Validar valor
            if not await self._validate_parameter_value(module_name, parameter_name, new_value):
                return False
            
            # Obtener valor actual
            current_value = await self.get_parameter(module_name, parameter_name)
            
            # Medir performance antes
            performance_before = await self._measure_performance(module_name)
            
            # Actualizar en DB
            await self._update_parameter_in_db(
                module_name, parameter_name, current_value, new_value,
                changed_by, change_reason, performance_before)
            
            # Invalidar caches
            cache_key = f"{module_name}:{parameter_name}"
            await self._invalidate_caches(cache_key)
            
            # Notificar cambio
            await self._notify_parameter_change(
                module_name, parameter_name, current_value, new_value)
            
            # Programar medición post-cambio
            asyncio.create_task(self._schedule_performance_measurement(
                module_name, parameter_name, delay_seconds=300))
            
            self.logger.info(f"Parameter updated: {module_name}.{parameter_name} = {new_value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating parameter: {e}")
            return False
    
    async def create_ab_test(self, 
                           test_name: str,
                           module_name: str,
                           parameter_name: str,
                           control_value: Any,
                           test_value: Any,
                           traffic_split: float = 0.5,
                           duration_hours: int = 24) -> bool:
        """Crea A/B test"""
        try:
            # Guardar en DB
            await self._save_ab_test_config(
                test_name, module_name, parameter_name,
                control_value, test_value, traffic_split, duration_hours)
            
            # Activar en memoria
            cache_key = f"{module_name}:{parameter_name}"
            self.ab_tests[cache_key] = {
                'test_name': test_name,
                'control_value': control_value,
                'test_value': test_value,
                'traffic_split': traffic_split,
                'start_date': datetime.now(),
                'end_date': datetime.now() + timedelta(hours=duration_hours)
            }
            
            self.logger.info(f"A/B test created: {test_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating A/B test: {e}")
            return False
    
    async def optimize_parameters(self, module_name: str) -> Dict[str, Any]:
        """Optimiza parámetros automáticamente"""
        try:
            # Obtener parámetros optimizables
            optimizable_params = await self._get_optimizable_parameters(module_name)
            
            optimization_results = {}
            
            for param in optimizable_params:
                # Obtener datos de performance
                performance_data = await self._get_performance_data(module_name, param['name'])
                
                if len(performance_data) < 50:  # Mínimo 50 samples
                    continue
                
                # Ejecutar optimización
                optimal_value = await self._optimize_single_parameter(param, performance_data)
                
                if optimal_value is not None:
                    expected_improvement = await self._estimate_improvement(
                        param, optimal_value, performance_data)
                    
                    if expected_improvement > 0.02:  # 2% mejora mínima
                        optimization_results[param['name']] = {
                            'current_value': param['current_value'],
                            'optimal_value': optimal_value,
                            'expected_improvement': expected_improvement
                        }
            
            # Aplicar optimizaciones
            applied = []
            for param_name, result in optimization_results.items():
                success = await self.update_parameter(
                    module_name, param_name, result['optimal_value'],
                    "auto_optimizer", ChangeReason.AUTO_OPTIMIZATION)
                
                if success:
                    applied.append(param_name)
            
            return {
                "status": "completed",
                "optimizations_found": len(optimization_results),
                "optimizations_applied": len(applied),
                "results": optimization_results
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return {"status": "error", "message": str(e)}
    
    async def subscribe_to_changes(self, module_name: str, callback):
        """Suscribe a cambios de parámetros"""
        if module_name not in self.change_subscribers:
            self.change_subscribers[module_name] = []
        self.change_subscribers[module_name].append(callback)
    
    async def rollback_parameter(self, module_name: str, parameter_name: str,
                               target_timestamp: datetime, rolled_back_by: str) -> bool:
        """Rollback a valor anterior"""
        try:
            target_value = await self._get_parameter_value_at_timestamp(
                module_name, parameter_name, target_timestamp)
            
            if target_value is None:
                return False
            
            return await self.update_parameter(
                module_name, parameter_name, target_value,
                rolled_back_by, ChangeReason.ROLLBACK)
            
        except Exception as e:
            self.logger.error(f"Error rolling back parameter: {e}")
            return False
    
    # Métodos auxiliares
    def _should_use_test_value(self, user_id: str, ab_config: Dict) -> bool:
        """Determina si usar valor de test"""
        import hashlib
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (hash_value % 100) / 100.0
        return bucket < ab_config['traffic_split']
    
    async def _load_active_configurations(self):
        """Carga configuraciones activas"""
        query = """
        SELECT module_name, parameter_name, parameter_value
        FROM trading_parameters
        WHERE is_active = true AND valid_from <= NOW()
        AND (valid_until IS NULL OR valid_until > NOW())
        """
        rows = await self.db.fetch(query)
        
        for row in rows:
            cache_key = f"{row['module_name']}:{row['parameter_name']}"
            await self._update_caches(cache_key, row['parameter_value'])
    
    async def _load_active_ab_tests(self):
        """Carga A/B tests activos"""
        query = """
        SELECT * FROM ab_test_configs
        WHERE status = 'active' AND start_date <= NOW()
        AND (end_date IS NULL OR end_date > NOW())
        """
        rows = await self.db.fetch(query)
        
        for row in rows:
            cache_key = f"{row['module_name']}:{row['parameter_name']}"
            self.ab_tests[cache_key] = {
                'test_name': row['test_name'],
                'control_value': row['control_value'],
                'test_value': row['test_value'],
                'traffic_split': float(row['traffic_split'])
            }
    
    async def _update_caches(self, cache_key: str, value: Any):
        """Actualiza caches"""
        self.local_cache[cache_key] = {
            'value': value,
            'timestamp': datetime.now()
        }
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(value))
    
    async def _invalidate_caches(self, cache_key: str):
        """Invalida caches"""
        if cache_key in self.local_cache:
            del self.local_cache[cache_key]
        self.redis.delete(cache_key)
    
    async def _notify_parameter_change(self, module_name: str, parameter_name: str,
                                     old_value: Any, new_value: Any):
        """Notifica cambio a suscriptores"""
        if module_name in self.change_subscribers:
            for callback in self.change_subscribers[module_name]:
                try:
                    await callback(parameter_name, old_value, new_value)
                except Exception as e:
                    self.logger.error(f"Error notifying subscriber: {e}")
    
    # Métodos de DB (implementar según necesidad)
    async def _load_parameter_from_db(self, module_name: str, parameter_name: str):
        """Carga parámetro desde DB"""
        pass
    
    async def _validate_parameter_value(self, module_name: str, parameter_name: str, value: Any) -> bool:
        """Valida valor de parámetro"""
        return True  # Implementar validación
    
    async def _update_parameter_in_db(self, module_name: str, parameter_name: str,
                                    old_value: Any, new_value: Any, changed_by: str,
                                    change_reason: ChangeReason, performance_before: float):
        """Actualiza parámetro en DB"""
        pass
    
    async def _measure_performance(self, module_name: str) -> float:
        """Mide performance actual"""
        return 0.0  # Implementar medición
    
    async def _optimize_single_parameter(self, param: Dict, performance_data: list):
        """Optimiza parámetro individual"""
        return None  # Implementar optimización
```

#### **Configuración**

```yaml
# config/dynamic_config.yaml
dynamic_config:
  redis_host: "localhost"
  redis_port: 6379
  cache_ttl: 60
  optimization_interval: 3600
  min_samples_for_optimization: 100
  performance_improvement_threshold: 0.02
  
  # Parámetros por módulo
  modules:
    dynamic_risk:
      parameters:
        - name: "base_risk_reward_ratio"
          type: "float"
          min_value: 1.5
          max_value: 3.0
          optimization_enabled: true
        - name: "kelly_fraction"
          type: "float"
          min_value: 0.1
          max_value: 0.5
          optimization_enabled: true
    
    key_levels:
      parameters:
        - name: "min_strength_threshold"
          type: "float"
          min_value: 0.2
          max_value: 0.8
          optimization_enabled: true
```

## 📊 **Métricas y KPIs**

- **Parameter Update Latency**: < 100ms
- **Cache Hit Rate**: > 95%
- **Optimization Success Rate**: > 70%
- **Rollback Time**: < 30s

## 🧪 **Testing Strategy**

```python
# tests/test_dynamic_config_manager.py
async def test_parameter_hot_reload():
    pass

async def test_ab_testing():
    pass

async def test_auto_optimization():
    pass
```

## 🚀 **Deployment Checklist**

- [ ] Implementar DynamicConfigManager
- [ ] Crear schema de base de datos
- [ ] Configurar Redis cache
- [ ] Implementar API REST
- [ ] Testing completo
- [ ] Deployment

---

*Documento creado: 2025-08-06 - Dynamic Configuration Management*
