# src/api/config_management_api.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from ..modules.config.dynamic_config_manager import DynamicConfigManager, ChangeReason

router = APIRouter(prefix="/api/config", tags=["configuration"])

# Modelos Pydantic
class ParameterUpdateRequest(BaseModel):
    value: Any
    changed_by: str = "api_user"
    reason: str = "manual"

class ABTestRequest(BaseModel):
    test_name: str
    module_name: str
    parameter_name: str
    control_value: Any
    test_value: Any
    traffic_split: float = 0.5
    duration_hours: int = 24
    created_by: str = "api_user"

class RollbackRequest(BaseModel):
    target_timestamp: str
    rolled_back_by: str = "api_user"

class ParameterDefinitionRequest(BaseModel):
    module_name: str
    parameter_name: str
    parameter_type: str
    current_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    optimization_enabled: bool = False

# Dependency injection
async def get_config_manager() -> DynamicConfigManager:
    # En producción, esto vendría del container de DI
    from ..shared.config import get_config
    config = get_config()
    manager = DynamicConfigManager(config.get('dynamic_config', {}))
    await manager.initialize()
    return manager

@router.get("/health")
async def health_check():
    """Health check del sistema de configuración"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/modules")
async def get_modules(config_manager: DynamicConfigManager = Depends(get_config_manager)):
    """Obtiene lista de módulos configurables"""
    try:
        modules = await config_manager.get_available_modules()
        return {"status": "success", "modules": modules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parameters/{module_name}")
async def get_module_parameters(
    module_name: str,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Obtiene todos los parámetros de un módulo"""
    try:
        parameters = await config_manager.get_module_parameters(module_name)
        return {"status": "success", "parameters": parameters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parameters/{module_name}/{parameter_name}")
async def get_parameter(
    module_name: str,
    parameter_name: str,
    user_id: Optional[str] = None,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Obtiene un parámetro específico"""
    try:
        value = await config_manager.get_parameter(module_name, parameter_name, user_id)
        return {
            "status": "success",
            "module_name": module_name,
            "parameter_name": parameter_name,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/parameters/{module_name}/{parameter_name}")
async def update_parameter(
    module_name: str,
    parameter_name: str,
    request: ParameterUpdateRequest,
    background_tasks: BackgroundTasks,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Actualiza un parámetro"""
    try:
        success = await config_manager.update_parameter(
            module_name=module_name,
            parameter_name=parameter_name,
            new_value=request.value,
            changed_by=request.changed_by,
            change_reason=ChangeReason(request.reason)
        )
        
        if success:
            # Programar validación post-cambio en background
            background_tasks.add_task(
                validate_parameter_change,
                config_manager, module_name, parameter_name
            )
            
            return {
                "status": "success",
                "message": "Parameter updated successfully",
                "module_name": module_name,
                "parameter_name": parameter_name,
                "new_value": request.value,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update parameter")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parameters")
async def create_parameter_definition(
    request: ParameterDefinitionRequest,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Crea una nueva definición de parámetro"""
    try:
        success = await config_manager.create_parameter_definition(
            module_name=request.module_name,
            parameter_name=request.parameter_name,
            parameter_type=request.parameter_type,
            current_value=request.current_value,
            min_value=request.min_value,
            max_value=request.max_value,
            description=request.description,
            optimization_enabled=request.optimization_enabled
        )
        
        if success:
            return {"status": "success", "message": "Parameter definition created"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create parameter definition")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests")
async def create_ab_test(
    request: ABTestRequest,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Crea un A/B test"""
    try:
        success = await config_manager.create_ab_test(
            test_name=request.test_name,
            module_name=request.module_name,
            parameter_name=request.parameter_name,
            control_value=request.control_value,
            test_value=request.test_value,
            traffic_split=request.traffic_split,
            duration_hours=request.duration_hours
        )
        
        if success:
            return {
                "status": "success",
                "message": "A/B test created successfully",
                "test_name": request.test_name,
                "start_time": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create A/B test")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-tests")
async def get_active_ab_tests(
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Obtiene A/B tests activos"""
    try:
        tests = await config_manager.get_active_ab_tests()
        return {"status": "success", "ab_tests": tests}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-tests/{test_name}/results")
async def get_ab_test_results(
    test_name: str,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Obtiene resultados de un A/B test"""
    try:
        results = await config_manager.get_ab_test_results(test_name)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests/{test_name}/stop")
async def stop_ab_test(
    test_name: str,
    winner: Optional[str] = None,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Detiene un A/B test"""
    try:
        success = await config_manager.stop_ab_test(test_name, winner)
        if success:
            return {"status": "success", "message": f"A/B test {test_name} stopped"}
        else:
            raise HTTPException(status_code=400, detail="Failed to stop A/B test")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize/{module_name}")
async def optimize_module_parameters(
    module_name: str,
    background_tasks: BackgroundTasks,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Optimiza parámetros de un módulo"""
    try:
        # Ejecutar optimización en background para no bloquear
        background_tasks.add_task(
            run_parameter_optimization,
            config_manager, module_name
        )
        
        return {
            "status": "success",
            "message": f"Parameter optimization started for {module_name}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimize/{module_name}/status")
async def get_optimization_status(
    module_name: str,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Obtiene estado de optimización"""
    try:
        status = await config_manager.get_optimization_status(module_name)
        return {"status": "success", "optimization_status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{module_name}/{parameter_name}")
async def get_parameter_history(
    module_name: str,
    parameter_name: str,
    days: int = 30,
    limit: int = 100,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Obtiene historial de cambios de un parámetro"""
    try:
        history = await config_manager.get_parameter_history(
            module_name, parameter_name, days, limit)
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback/{module_name}/{parameter_name}")
async def rollback_parameter(
    module_name: str,
    parameter_name: str,
    request: RollbackRequest,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Rollback de un parámetro a un valor anterior"""
    try:
        target_timestamp = datetime.fromisoformat(request.target_timestamp)
        
        success = await config_manager.rollback_parameter(
            module_name=module_name,
            parameter_name=parameter_name,
            target_timestamp=target_timestamp,
            rolled_back_by=request.rolled_back_by
        )
        
        if success:
            return {
                "status": "success",
                "message": "Parameter rolled back successfully",
                "target_timestamp": request.target_timestamp,
                "rollback_time": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to rollback parameter")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/{module_name}")
async def get_module_performance(
    module_name: str,
    hours: int = 24,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Obtiene métricas de performance de un módulo"""
    try:
        performance = await config_manager.get_module_performance(module_name, hours)
        return {"status": "success", "performance": performance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit-trail")
async def get_audit_trail(
    module_name: Optional[str] = None,
    parameter_name: Optional[str] = None,
    changed_by: Optional[str] = None,
    days: int = 7,
    limit: int = 100,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Obtiene audit trail de cambios"""
    try:
        audit_trail = await config_manager.get_audit_trail(
            module_name=module_name,
            parameter_name=parameter_name,
            changed_by=changed_by,
            days=days,
            limit=limit
        )
        return {"status": "success", "audit_trail": audit_trail}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate/{module_name}")
async def validate_module_configuration(
    module_name: str,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Valida la configuración actual de un módulo"""
    try:
        validation_result = await config_manager.validate_module_configuration(module_name)
        return {"status": "success", "validation": validation_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backup")
async def backup_configuration(
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Crea backup de toda la configuración"""
    try:
        backup_id = await config_manager.create_configuration_backup()
        return {
            "status": "success",
            "message": "Configuration backup created",
            "backup_id": backup_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restore/{backup_id}")
async def restore_configuration(
    backup_id: str,
    config_manager: DynamicConfigManager = Depends(get_config_manager)
):
    """Restaura configuración desde backup"""
    try:
        success = await config_manager.restore_configuration_backup(backup_id)
        if success:
            return {
                "status": "success",
                "message": "Configuration restored successfully",
                "backup_id": backup_id,
                "restore_time": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to restore configuration")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def validate_parameter_change(config_manager: DynamicConfigManager,
                                  module_name: str, parameter_name: str):
    """Valida cambio de parámetro en background"""
    try:
        # Esperar un poco para que el cambio tome efecto
        await asyncio.sleep(10)
        
        # Validar que el módulo sigue funcionando correctamente
        validation_result = await config_manager.validate_parameter_change(
            module_name, parameter_name)
        
        if not validation_result.get('valid', True):
            # Si la validación falla, considerar rollback automático
            await config_manager.handle_validation_failure(
                module_name, parameter_name, validation_result)
                
    except Exception as e:
        config_manager.logger.error(f"Error validating parameter change: {e}")

async def run_parameter_optimization(config_manager: DynamicConfigManager,
                                   module_name: str):
    """Ejecuta optimización de parámetros en background"""
    try:
        result = await config_manager.optimize_parameters(module_name)
        
        # Log resultado
        config_manager.logger.info(f"Parameter optimization completed for {module_name}: {result}")
        
        # Notificar resultado via WebSocket si está configurado
        await config_manager.notify_optimization_result(module_name, result)
        
    except Exception as e:
        config_manager.logger.error(f"Error in parameter optimization: {e}")

# WebSocket endpoint para notificaciones en tiempo real
@router.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket,
                                config_manager: DynamicConfigManager = Depends(get_config_manager)):
    """WebSocket para notificaciones de cambios en tiempo real"""
    await websocket.accept()
    
    try:
        # Registrar cliente para notificaciones
        client_id = str(uuid.uuid4())
        await config_manager.register_websocket_client(client_id, websocket)
        
        while True:
            # Mantener conexión viva
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        await config_manager.unregister_websocket_client(client_id)
    except Exception as e:
        config_manager.logger.error(f"WebSocket error: {e}")
        await config_manager.unregister_websocket_client(client_id)
