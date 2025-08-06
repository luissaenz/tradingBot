-- sql/init/01_init_config_management.sql
-- Inicialización de base de datos para gestión de configuración dinámica

-- Crear extensión TimescaleDB si no existe
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Tabla principal de parámetros
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

-- Historial de cambios
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

-- A/B Testing
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

-- Métricas de A/B tests
CREATE TABLE IF NOT EXISTS ab_test_metrics (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) REFERENCES ab_test_configs(test_name),
    variant VARCHAR(20) NOT NULL,
    user_id VARCHAR(100),
    metric_value DECIMAL(15,8),
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- Convertir tabla de métricas a hypertable para TimescaleDB
SELECT create_hypertable('ab_test_metrics', 'recorded_at', if_not_exists => TRUE);
SELECT create_hypertable('parameter_changes', 'changed_at', if_not_exists => TRUE);

-- Índices para optimización
CREATE INDEX IF NOT EXISTS idx_trading_parameters_module_param 
ON trading_parameters(module_name, parameter_name);

CREATE INDEX IF NOT EXISTS idx_trading_parameters_active 
ON trading_parameters(is_active, valid_from) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_parameter_changes_param_id 
ON parameter_changes(parameter_id);

CREATE INDEX IF NOT EXISTS idx_parameter_changes_changed_at 
ON parameter_changes(changed_at DESC);

CREATE INDEX IF NOT EXISTS idx_ab_test_configs_status 
ON ab_test_configs(status) WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_ab_test_metrics_test_name 
ON ab_test_metrics(test_name, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_ab_test_metrics_variant 
ON ab_test_metrics(test_name, variant, recorded_at DESC);

-- Insertar parámetros de configuración inicial
INSERT INTO trading_parameters (module_name, parameter_name, parameter_value, parameter_type, created_by, description, optimization_enabled) VALUES
-- Data Ingestion Module
('data_ingestion', 'websocket_reconnect_delay', '5', 'integer', 'system', 'Delay in seconds before reconnecting WebSocket', false),
('data_ingestion', 'max_reconnect_attempts', '10', 'integer', 'system', 'Maximum number of reconnection attempts', false),
('data_ingestion', 'buffer_size', '1000', 'integer', 'system', 'Size of data buffer', true),
('data_ingestion', 'batch_size', '100', 'integer', 'system', 'Batch size for processing', true),

-- Signal Generation Module
('signal_generation', 'lookback_period', '20', 'integer', 'system', 'Number of periods to look back for signals', true),
('signal_generation', 'confidence_threshold', '0.65', 'float', 'system', 'Minimum confidence for signal generation', true),
('signal_generation', 'retraining_interval_hours', '168', 'integer', 'system', 'Hours between model retraining (weekly)', false),
('signal_generation', 'feature_importance_threshold', '0.01', 'float', 'system', 'Minimum feature importance to keep', true),

-- Trading Execution Module
('trading_execution', 'max_position_size', '1000.0', 'float', 'system', 'Maximum position size in USD', true),
('trading_execution', 'risk_per_trade', '0.01', 'float', 'system', 'Risk percentage per trade', true),
('trading_execution', 'stop_loss_pct', '0.02', 'float', 'system', 'Stop loss percentage', true),
('trading_execution', 'take_profit_pct', '0.04', 'float', 'system', 'Take profit percentage', true),
('trading_execution', 'max_daily_trades', '50', 'integer', 'system', 'Maximum trades per day', false),

-- Risk Management Module
('risk_management', 'max_daily_drawdown', '0.02', 'float', 'system', 'Maximum daily drawdown allowed', false),
('risk_management', 'max_monthly_drawdown', '0.05', 'float', 'system', 'Maximum monthly drawdown allowed', false),
('risk_management', 'position_sizing_method', '"kelly"', 'string', 'system', 'Position sizing method', false),
('risk_management', 'correlation_threshold', '0.7', 'float', 'system', 'Maximum correlation between positions', true),

-- Microstructure Analysis Module
('microstructure', 'order_book_depth', '10', 'integer', 'system', 'Order book depth to analyze', true),
('microstructure', 'imbalance_threshold', '0.6', 'float', 'system', 'Order book imbalance threshold', true),
('microstructure', 'volume_spike_threshold', '2.0', 'float', 'system', 'Volume spike detection threshold', true),
('microstructure', 'price_impact_window', '5', 'integer', 'system', 'Window for price impact calculation', true)

ON CONFLICT (module_name, parameter_name, valid_from) DO NOTHING;

-- Crear vista para parámetros activos
CREATE OR REPLACE VIEW active_parameters AS
SELECT 
    module_name,
    parameter_name,
    parameter_value,
    parameter_type,
    min_value,
    max_value,
    description,
    optimization_enabled,
    valid_from,
    created_by
FROM trading_parameters 
WHERE is_active = true 
AND (valid_until IS NULL OR valid_until > NOW())
ORDER BY module_name, parameter_name;

-- Crear vista para historial de cambios con performance
CREATE OR REPLACE VIEW parameter_change_history AS
SELECT 
    tp.module_name,
    tp.parameter_name,
    pc.old_value,
    pc.new_value,
    pc.change_reason,
    pc.changed_by,
    pc.changed_at,
    pc.performance_before,
    pc.performance_after,
    CASE 
        WHEN pc.performance_after IS NOT NULL AND pc.performance_before IS NOT NULL 
        THEN ((pc.performance_after - pc.performance_before) / pc.performance_before * 100)
        ELSE NULL 
    END as performance_improvement_pct
FROM parameter_changes pc
JOIN trading_parameters tp ON pc.parameter_id = tp.id
ORDER BY pc.changed_at DESC;

-- Crear función para obtener parámetro actual
CREATE OR REPLACE FUNCTION get_current_parameter(
    p_module_name VARCHAR(50),
    p_parameter_name VARCHAR(100)
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT parameter_value INTO result
    FROM trading_parameters
    WHERE module_name = p_module_name 
    AND parameter_name = p_parameter_name
    AND is_active = true
    AND (valid_until IS NULL OR valid_until > NOW())
    ORDER BY valid_from DESC
    LIMIT 1;
    
    RETURN COALESCE(result, 'null'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Crear función para audit trail
CREATE OR REPLACE FUNCTION log_parameter_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Log cuando se desactiva un parámetro (UPDATE)
    IF TG_OP = 'UPDATE' AND OLD.is_active = true AND NEW.is_active = false THEN
        INSERT INTO parameter_changes (
            parameter_id, 
            old_value, 
            new_value, 
            change_reason, 
            changed_by
        ) VALUES (
            OLD.id,
            OLD.parameter_value,
            NEW.parameter_value,
            'parameter_deactivated',
            NEW.created_by
        );
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Crear trigger para audit trail
CREATE TRIGGER trigger_log_parameter_change
    AFTER UPDATE ON trading_parameters
    FOR EACH ROW
    EXECUTE FUNCTION log_parameter_change();

-- Insertar configuración inicial para A/B tests de ejemplo (comentado)
/*
INSERT INTO ab_test_configs (test_name, module_name, parameter_name, control_value, test_value, traffic_split, created_by) VALUES
('stop_loss_optimization', 'trading_execution', 'stop_loss_pct', '0.02', '0.015', 0.5, 'system'),
('confidence_threshold_test', 'signal_generation', 'confidence_threshold', '0.65', '0.70', 0.3, 'system');
*/

-- Crear políticas de retención (opcional)
-- SELECT add_retention_policy('parameter_changes', INTERVAL '1 year');
-- SELECT add_retention_policy('ab_test_metrics', INTERVAL '6 months');

COMMIT;
