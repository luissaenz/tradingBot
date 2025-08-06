# tests/test_dynamic_config_manager.py
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.modules.config.dynamic_config_manager import (
    DynamicConfigManager, 
    Parameter, 
    ParameterType, 
    ChangeReason,
    ABTestConfig
)

@pytest.fixture
async def config_manager():
    """Fixture para DynamicConfigManager con mocks"""
    config = {
        'postgres_host': 'localhost',
        'postgres_port': 5432,
        'postgres_user': 'test_user',
        'postgres_password': 'test_pass',
        'postgres_db': 'test_db',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_db': 0
    }
    
    manager = DynamicConfigManager(config)
    
    # Mock de conexiones
    manager.db_pool = AsyncMock()
    manager.redis_client = AsyncMock()
    
    # Mock de métodos de inicialización
    manager._create_tables = AsyncMock()
    manager._load_parameters_cache = AsyncMock()
    manager._load_active_ab_tests = AsyncMock()
    
    await manager.initialize()
    
    return manager

@pytest.fixture
def sample_parameter():
    """Parámetro de ejemplo para tests"""
    return Parameter(
        module_name="trading_engine",
        parameter_name="max_position_size",
        current_value=1000.0,
        parameter_type=ParameterType.FLOAT,
        min_value=100.0,
        max_value=10000.0,
        description="Tamaño máximo de posición",
        optimization_enabled=True,
        last_updated=datetime.now(),
        updated_by="test_user"
    )

@pytest.fixture
def sample_ab_test():
    """A/B test de ejemplo para tests"""
    return ABTestConfig(
        test_name="position_size_test",
        module_name="trading_engine",
        parameter_name="max_position_size",
        control_value=1000.0,
        test_value=1500.0,
        traffic_split=0.5,
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(hours=24)
    )

class TestDynamicConfigManager:
    """Tests para DynamicConfigManager"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, config_manager):
        """Test de inicialización correcta"""
        assert config_manager.db_pool is not None
        assert config_manager.redis_client is not None
        assert isinstance(config_manager.parameter_cache, dict)
        assert isinstance(config_manager.ab_tests, dict)
    
    @pytest.mark.asyncio
    async def test_get_parameter_from_cache(self, config_manager, sample_parameter):
        """Test obtener parámetro desde cache"""
        # Agregar parámetro al cache
        cache_key = f"{sample_parameter.module_name}.{sample_parameter.parameter_name}"
        config_manager.parameter_cache[cache_key] = sample_parameter
        
        # Obtener parámetro
        value = await config_manager.get_parameter(
            sample_parameter.module_name,
            sample_parameter.parameter_name
        )
        
        assert value == sample_parameter.current_value
    
    @pytest.mark.asyncio
    async def test_get_parameter_not_found(self, config_manager):
        """Test parámetro no encontrado"""
        with pytest.raises(ValueError, match="no encontrado"):
            await config_manager.get_parameter("nonexistent", "parameter")
    
    @pytest.mark.asyncio
    async def test_get_parameter_with_ab_test(self, config_manager, sample_ab_test):
        """Test obtener parámetro con A/B test activo"""
        # Agregar A/B test
        config_manager.ab_tests[sample_ab_test.test_name] = sample_ab_test
        
        # Mock del método _should_use_test_value
        config_manager._should_use_test_value = MagicMock(return_value=True)
        
        # Obtener parámetro con user_id
        value = await config_manager.get_parameter(
            sample_ab_test.module_name,
            sample_ab_test.parameter_name,
            user_id="test_user_123"
        )
        
        assert value == sample_ab_test.test_value
        config_manager._should_use_test_value.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_parameter_success(self, config_manager, sample_parameter):
        """Test actualización exitosa de parámetro"""
        # Setup cache
        cache_key = f"{sample_parameter.module_name}.{sample_parameter.parameter_name}"
        config_manager.parameter_cache[cache_key] = sample_parameter
        
        # Mock de validación
        config_manager._validate_parameter_value = AsyncMock(return_value=True)
        config_manager._notify_parameter_change = AsyncMock()
        
        # Mock de base de datos
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()
        mock_conn.transaction.return_value = mock_transaction
        mock_conn.fetchval.return_value = 123  # parameter_id
        config_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Actualizar parámetro
        new_value = 2000.0
        result = await config_manager.update_parameter(
            sample_parameter.module_name,
            sample_parameter.parameter_name,
            new_value,
            "test_user",
            ChangeReason.MANUAL
        )
        
        assert result is True
        assert config_manager.parameter_cache[cache_key].current_value == new_value
        config_manager._notify_parameter_change.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_parameter_validation_failed(self, config_manager, sample_parameter):
        """Test actualización con validación fallida"""
        # Mock de validación que falla
        config_manager._validate_parameter_value = AsyncMock(return_value=False)
        
        # Intentar actualizar parámetro
        result = await config_manager.update_parameter(
            sample_parameter.module_name,
            sample_parameter.parameter_name,
            -500.0,  # Valor inválido
            "test_user"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_ab_test_success(self, config_manager):
        """Test creación exitosa de A/B test"""
        # Mock de base de datos
        mock_conn = AsyncMock()
        config_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Crear A/B test
        result = await config_manager.create_ab_test(
            test_name="new_test",
            module_name="trading_engine",
            parameter_name="stop_loss_pct",
            control_value=0.02,
            test_value=0.03,
            traffic_split=0.5,
            duration_hours=48,
            created_by="test_user"
        )
        
        assert result is True
        assert "new_test" in config_manager.ab_tests
        
        # Verificar configuración del A/B test
        ab_test = config_manager.ab_tests["new_test"]
        assert ab_test.control_value == 0.02
        assert ab_test.test_value == 0.03
        assert ab_test.traffic_split == 0.5
    
    @pytest.mark.asyncio
    async def test_create_ab_test_database_error(self, config_manager):
        """Test creación de A/B test con error de base de datos"""
        # Mock que simula error de base de datos
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("Database error")
        config_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Intentar crear A/B test
        result = await config_manager.create_ab_test(
            test_name="failing_test",
            module_name="trading_engine",
            parameter_name="stop_loss_pct",
            control_value=0.02,
            test_value=0.03
        )
        
        assert result is False
        assert "failing_test" not in config_manager.ab_tests
    
    def test_should_use_test_value_consistent_hashing(self, config_manager, sample_ab_test):
        """Test que el hash sea consistente para el mismo user_id"""
        user_id = "test_user_123"
        
        # Llamar múltiples veces con el mismo user_id
        result1 = config_manager._should_use_test_value(user_id, sample_ab_test)
        result2 = config_manager._should_use_test_value(user_id, sample_ab_test)
        result3 = config_manager._should_use_test_value(user_id, sample_ab_test)
        
        # Debe ser consistente
        assert result1 == result2 == result3
    
    def test_should_use_test_value_distribution(self, config_manager, sample_ab_test):
        """Test que la distribución de A/B test sea aproximadamente correcta"""
        test_users = [f"user_{i}" for i in range(1000)]
        test_assignments = []
        
        for user_id in test_users:
            is_test = config_manager._should_use_test_value(user_id, sample_ab_test)
            test_assignments.append(is_test)
        
        # Con traffic_split=0.5, esperamos aproximadamente 50% en test
        test_percentage = sum(test_assignments) / len(test_assignments)
        
        # Permitir cierta variación (±5%)
        assert 0.45 <= test_percentage <= 0.55
    
    @pytest.mark.asyncio
    async def test_notify_parameter_change(self, config_manager):
        """Test notificación de cambio de parámetro"""
        # Llamar método de notificación
        await config_manager._notify_parameter_change(
            "trading_engine",
            "max_position_size",
            1000.0,
            2000.0
        )
        
        # Verificar que se publicó en Redis
        config_manager.redis_client.publish.assert_called_once()
        
        # Verificar contenido del mensaje
        call_args = config_manager.redis_client.publish.call_args
        channel, message = call_args[0]
        
        assert channel == "config_changes"
        
        message_data = json.loads(message)
        assert message_data["module_name"] == "trading_engine"
        assert message_data["parameter_name"] == "max_position_size"
        assert message_data["old_value"] == 1000.0
        assert message_data["new_value"] == 2000.0
        assert "timestamp" in message_data

class TestParameterValidation:
    """Tests para validación de parámetros"""
    
    @pytest.mark.asyncio
    async def test_validate_parameter_value_success(self, config_manager, sample_parameter):
        """Test validación exitosa de parámetro"""
        cache_key = f"{sample_parameter.module_name}.{sample_parameter.parameter_name}"
        config_manager.parameter_cache[cache_key] = sample_parameter
        
        # Valor válido dentro del rango
        result = await config_manager._validate_parameter_value(
            sample_parameter.module_name,
            sample_parameter.parameter_name,
            500.0
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_parameter_value_new_parameter(self, config_manager):
        """Test validación de parámetro nuevo (no en cache)"""
        # Parámetro que no existe en cache
        result = await config_manager._validate_parameter_value(
            "new_module",
            "new_parameter",
            "any_value"
        )
        
        # Debe asumir válido para parámetros nuevos
        assert result is True

@pytest.mark.asyncio
async def test_integration_parameter_lifecycle(config_manager):
    """Test de integración: ciclo completo de parámetro"""
    # 1. Crear parámetro inicial
    param = Parameter(
        module_name="test_module",
        parameter_name="test_param",
        current_value=100,
        parameter_type=ParameterType.INTEGER,
        min_value=1,
        max_value=1000
    )
    
    cache_key = f"{param.module_name}.{param.parameter_name}"
    config_manager.parameter_cache[cache_key] = param
    
    # Mock necesarios
    config_manager._validate_parameter_value = AsyncMock(return_value=True)
    config_manager._notify_parameter_change = AsyncMock()
    
    mock_conn = AsyncMock()
    mock_transaction = AsyncMock()
    mock_conn.transaction.return_value = mock_transaction
    mock_conn.fetchval.return_value = 123
    config_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    # 2. Obtener valor inicial
    initial_value = await config_manager.get_parameter(param.module_name, param.parameter_name)
    assert initial_value == 100
    
    # 3. Actualizar parámetro
    update_result = await config_manager.update_parameter(
        param.module_name,
        param.parameter_name,
        200,
        "integration_test"
    )
    assert update_result is True
    
    # 4. Verificar nuevo valor
    new_value = await config_manager.get_parameter(param.module_name, param.parameter_name)
    assert new_value == 200
    
    # 5. Crear A/B test
    ab_result = await config_manager.create_ab_test(
        "integration_test",
        param.module_name,
        param.parameter_name,
        200,  # control
        300,  # test
        0.5
    )
    assert ab_result is True
    
    # 6. Obtener valor con A/B test (simulando user en grupo test)
    config_manager._should_use_test_value = MagicMock(return_value=True)
    ab_value = await config_manager.get_parameter(
        param.module_name,
        param.parameter_name,
        user_id="test_user"
    )
    assert ab_value == 300

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
