import os
import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de conexión
DB_HOST = os.getenv("TIMESCALEDB_HOST", "localhost")  # Usa localhost por defecto para ejecución local
DB_PORT = os.getenv("TIMESCALEDB_PORT", "5432")
DB_NAME = os.getenv("TIMESCALEDB_DB", "postgres")
DB_USER = os.getenv("TIMESCALEDB_USER", "admin")
DB_PASSWORD = os.getenv("TIMESCALEDB_PASSWORD", "password")

def create_connection():
    """Crea una conexión a TimescaleDB."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Conexión exitosa a TimescaleDB")
        return conn
    except OperationalError as e:
        print(f"Error de conexión: {e}")
        return None

def create_tables(conn):
    """Crea las tablas necesarias en TimescaleDB."""
    try:
        cursor = conn.cursor()

        # Tabla para datos de microestructura
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS microstructure_data (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                bid_price DOUBLE PRECISION,
                ask_price DOUBLE PRECISION,
                bid_volume DOUBLE PRECISION,
                ask_volume DOUBLE PRECISION,
                imbalance DOUBLE PRECISION,
                delta_volume DOUBLE PRECISION
            );
        """)
        cursor.execute("SELECT create_hypertable('microstructure_data', 'timestamp', if_not_exists => TRUE);")

        # Tabla para datos de sentimiento
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                timestamp TIMESTAMPTZ NOT NULL,
                source TEXT NOT NULL,
                sentiment_score DOUBLE PRECISION,
                text TEXT
            );
        """)
        cursor.execute("SELECT create_hypertable('sentiment_data', 'timestamp', if_not_exists => TRUE);")

        # Tabla para trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price DOUBLE PRECISION,
                quantity DOUBLE PRECISION,
                order_id TEXT
            );
        """)

        # Tabla para logs de errores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS errors_log (
                timestamp TIMESTAMPTZ NOT NULL,
                module TEXT NOT NULL,
                error_message TEXT
            );
        """)

        # Tabla para resultados de backtesting
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                backtest_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                strategy TEXT NOT NULL,
                profit_loss DOUBLE PRECISION,
                trades_count INTEGER,
                win_rate DOUBLE PRECISION
            );
        """)

        conn.commit()
        print("Tablas creadas exitosamente")
        cursor.close()
    except Exception as e:
        print(f"Error al crear tablas: {e}")
        conn.rollback()

def main():
    """Función principal para inicializar la base de datos."""
    conn = create_connection()
    if conn is None:
        print("No se pudo conectar a la base de datos. Terminando.")
        return

    try:
        create_tables(conn)
    finally:
        conn.close()
        print("Conexión cerrada")

if __name__ == "__main__":
    main()