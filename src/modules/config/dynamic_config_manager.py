# src/modules/config/dynamic_config_manager.py
import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncpg
import redis.asyncio as redis
from scipy import stats
import numpy as np
import hashlib

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    JSON = "json"
    LIST = "list"

class ChangeReason(Enum):
    MANUAL = "manual"
    OPTIMIZATION = "optimization"
    AB_TEST = "ab_test"
    ROLLBACK = "rollback"
    SYSTEM = "system"

@dataclass
class Parameter:
    module_name: str
    parameter_name: str
    current_value: Any
    parameter_type: ParameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    optimization_enabled: bool = False
    last_updated: datetime = None
    updated_by: str = "system"

@dataclass
class ABTestConfig:
    test_name: str
    module_name: str
    parameter_name: str
    control_value: Any
    test_value: Any
    traffic_split: float
    start_date: datetime
    end_date: Optional[datetime] = None
    status: str = "active"
    winner: Optional[str] = None
    confidence_level: Optional[float] = None

class DynamicConfigManager:
    """Gestor de configuración dinámica con optimización automática"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.parameter_cache: Dict[str, Parameter] = {}
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.optimization_tasks: Dict[str, asyncio.Task] = {}
        self.notification_callbacks: List[callable] = []
        
    async def initialize(self):
        """Inicializa conexiones y carga configuración inicial"""
        try:
            # Conexión a PostgreSQL
            self.db_pool = await asyncpg.create_pool(
                host=self.config.get('postgres_host', 'localhost'),
                port=self.config.get('postgres_port', 5432),
                user=self.config.get('postgres_user', 'trading_user'),
                password=self.config.get('postgres_password', 'trading_pass'),
                database=self.config.get('postgres_db', 'trading_bot'),
                min_size=2,
                max_size=10
            )
            
            # Conexión a Redis
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            
            # Crear tablas si no existen
            await self._create_tables()
            
            # Cargar parámetros en cache
            await self._load_parameters_cache()
            
            # Cargar A/B tests activos
            await self._load_active_ab_tests()
            
            logger.info("DynamicConfigManager inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando DynamicConfigManager: {e}")
            raise
    
    async def _create_tables(self):
        """Crea las tablas necesarias en PostgreSQL"""
        async with self.db_pool.acquire() as conn:
            # Tabla de parámetros
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_parameters (
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
            """)
            
            # Tabla de historial de cambios
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_changes (
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
            """)
            
            # Tabla de A/B tests
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_configs (
                    id SERIAL PRIMARY KEY,
                    test_name VARCHAR(100) NOT NULL UNIQUE,
                    module_name VARCHAR(50) NOT NULL,
                    parameter_name VARCHAR(100) NOT NULL,
                    control_value JSONB NOT NULL,
                    test_value JSONB NOT NULL,
                    traffic_split DECIMAL(3,2) DEFAULT 0.5,
                    start_date TIMESTAMP DEFAULT NOW(),
                    end_date TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active',
                    winner VARCHAR(20),
                    confidence_level DECIMAL(5,4),
                    created_by VARCHAR(50) NOT NULL
                );
            """)
            
            # Tabla de métricas de A/B tests
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_metrics (
                    id SERIAL PRIMARY KEY,
                    test_name VARCHAR(100) REFERENCES ab_test_configs(test_name),
                    variant VARCHAR(20) NOT NULL,
                    user_id VARCHAR(100),
                    metric_value DECIMAL(15,8),
                    recorded_at TIMESTAMP DEFAULT NOW()
                );
            """)
    
    async def get_parameter(self, module_name: str, parameter_name: str, 
                          user_id: Optional[str] = None) -> Any:
        """Obtiene el valor de un parámetro, considerando A/B tests"""
        cache_key = f"{module_name}.{parameter_name}"
        
        # Verificar A/B tests activos
        if user_id:
            for test_name, ab_config in self.ab_tests.items():
                if (ab_config.module_name == module_name and 
                    ab_config.parameter_name == parameter_name):
                    
                    if self._should_use_test_value(user_id, ab_config):
                        return ab_config.test_value
                    else:
                        return ab_config.control_value
        
        # Valor normal del cache
        if cache_key in self.parameter_cache:
            return self.parameter_cache[cache_key].current_value
        
        raise ValueError(f"Parámetro {module_name}.{parameter_name} no encontrado")
    
    async def update_parameter(self, module_name: str, parameter_name: str, 
                             new_value: Any, changed_by: str = "system",
                             reason: ChangeReason = ChangeReason.MANUAL) -> bool:
        """Actualiza un parámetro y registra el cambio"""
        try:
            cache_key = f"{module_name}.{parameter_name}"
            old_value = None
            
            # Obtener valor anterior
            if cache_key in self.parameter_cache:
                old_value = self.parameter_cache[cache_key].current_value
            
            # Validar el nuevo valor
            if not await self._validate_parameter_value(module_name, parameter_name, new_value):
                return False
            
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Desactivar parámetro anterior
                    await conn.execute("""
                        UPDATE trading_parameters 
                        SET valid_until = NOW() 
                        WHERE module_name = $1 AND parameter_name = $2 AND is_active = true
                    """, module_name, parameter_name)
                    
                    # Insertar nuevo parámetro
                    param_id = await conn.fetchval("""
                        INSERT INTO trading_parameters 
                        (module_name, parameter_name, parameter_value, parameter_type, 
                         created_by, description, optimization_enabled)
                        VALUES ($1, $2, $3, 'string', $4, '', false)
                        RETURNING id
                    """, module_name, parameter_name, json.dumps(new_value), changed_by)
                    
                    # Registrar cambio
                    await conn.execute("""
                        INSERT INTO parameter_changes 
                        (parameter_id, old_value, new_value, change_reason, changed_by)
                        VALUES ($1, $2, $3, $4, $5)
                    """, param_id, json.dumps(old_value), json.dumps(new_value), 
                        reason.value, changed_by)
            
            # Actualizar cache
            if cache_key in self.parameter_cache:
                self.parameter_cache[cache_key].current_value = new_value
                self.parameter_cache[cache_key].last_updated = datetime.now()
                self.parameter_cache[cache_key].updated_by = changed_by
            
            # Notificar cambio
            await self._notify_parameter_change(module_name, parameter_name, old_value, new_value)
            
            logger.info(f"Parámetro {cache_key} actualizado: {old_value} -> {new_value}")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando parámetro {module_name}.{parameter_name}: {e}")
            return False
    
    async def create_ab_test(self, test_name: str, module_name: str, parameter_name: str,
                           control_value: Any, test_value: Any, traffic_split: float = 0.5,
                           duration_hours: int = 24, created_by: str = "system") -> bool:
        """Crea un nuevo A/B test"""
        try:
            end_date = datetime.now() + timedelta(hours=duration_hours)
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ab_test_configs 
                    (test_name, module_name, parameter_name, control_value, test_value,
                     traffic_split, end_date, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, test_name, module_name, parameter_name, 
                    json.dumps(control_value), json.dumps(test_value),
                    traffic_split, end_date, created_by)
            
            # Agregar al cache
            ab_test = ABTestConfig(
                test_name=test_name,
                module_name=module_name,
                parameter_name=parameter_name,
                control_value=control_value,
                test_value=test_value,
                traffic_split=traffic_split,
                start_date=datetime.now(),
                end_date=end_date
            )
            
            self.ab_tests[test_name] = ab_test
            
            logger.info(f"A/B test '{test_name}' creado para {module_name}.{parameter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creando A/B test {test_name}: {e}")
            return False
    
    def _should_use_test_value(self, user_id: str, ab_config: ABTestConfig) -> bool:
        """Determina si usar valor de test basado en hash consistente"""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (hash_value % 100) / 100.0
        return bucket < ab_config.traffic_split
    
    async def _validate_parameter_value(self, module_name: str, parameter_name: str, value: Any) -> bool:
        """Valida que un valor de parámetro sea válido"""
        return True  # Implementación simplificada
    
    async def _notify_parameter_change(self, module_name: str, parameter_name: str, 
                                     old_value: Any, new_value: Any):
        """Notifica cambio de parámetro"""
        change_data = {
            "module_name": module_name,
            "parameter_name": parameter_name,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publicar en Redis para notificaciones en tiempo real
        if self.redis_client:
            await self.redis_client.publish(
                "config_changes", 
                json.dumps(change_data)
            )
    
    async def _load_parameters_cache(self):
        """Carga parámetros en cache desde base de datos"""
        # Implementación simplificada para inicialización
        pass
    
    async def _load_active_ab_tests(self):
        """Carga A/B tests activos"""
        # Implementación simplificada para inicialización
        pass
