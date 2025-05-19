import os
import pandas as pd
from sqlalchemy import create_engine
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from dotenv import load_dotenv
import logging
import psycopg2
from datetime import datetime, timedelta
import numpy as np
import signal

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


def fetch_data():
    """Extrae datos de TimescaleDB."""
    try:
        engine = create_engine(
            f"postgresql://admin:password@{os.getenv('TIMESCALEDB_HOST', 'timescaledb')}:{os.getenv('TIMESCALEDB_PORT', '5432')}/{os.getenv('TIMESCALEDB_DB', 'postgres')}"
        )
        query = """
        SELECT timestamp, bid_price, ask_price, bid_volume, ask_volume, imbalance, delta_volume
        FROM microstructure_data
        WHERE timestamp > %s
        AND bid_price IS NOT NULL AND ask_price IS NOT NULL
        ORDER BY timestamp
        """
        df = pd.read_sql(query, engine, params=(
            datetime.now() - timedelta(days=7),))
        engine.dispose()
        logger.info(f"Datos extraídos: {len(df)} filas")
        return df
    except Exception as e:
        logger.error(f"Error al extraer datos: {e}")
        return None


def prepare_features(df):
    """Prepara features para el modelo."""
    try:
        df['price'] = (df['bid_price'] + df['ask_price']) / 2
        df['spread'] = df['ask_price'] - df['bid_price']
        df['imbalance_lag1'] = df['imbalance'].shift(1)
        df['delta_volume_lag1'] = df['delta_volume'].shift(1)
        df['price_ma5'] = df['price'].rolling(window=5).mean()
        df['price_ema10'] = df['price'].ewm(span=10, adjust=False).mean()
        df['total_volume'] = df['bid_volume'] + df['ask_volume']
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        # Volatilidad
        df['volatility'] = df['price'].rolling(window=20).std()

        features = [
            'imbalance', 'delta_volume', 'spread', 'imbalance_lag1', 'delta_volume_lag1',
            'price_ma5', 'price_ema10', 'total_volume', 'rsi', 'volatility'
        ]
        X = df[features].dropna()

        # Normalizar features, mantener nombres
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(
            X), columns=features, index=X.index)
        logger.info(f"Datos tras preparar features: {len(X)} filas")
        return X_scaled, df.loc[X.index]
    except Exception as e:
        logger.error(f"Error al preparar features: {e}")
        return None, None


def save_signal(signal, timestamp):
    """Guarda la señal en TimescaleDB."""
    try:
        conn = psycopg2.connect(
            host=os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
            port=os.getenv('TIMESCALEDB_PORT', '5432'),
            database=os.getenv('TIMESCALEDB_DB', 'postgres'),
            user=os.getenv('TIMESCALEDB_USER', 'admin'),
            password=os.getenv('TIMESCALEDB_PASSWORD', 'password')
        )
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (order_id, timestamp, symbol, side, price, quantity, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                f"signal_{timestamp.strftime('%Y%m%d%H%M%S')}",
                timestamp,
                "BTC/USDT",
                signal,
                0.0,
                0.0,
                'pending'
            )
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Señal guardada en TimescaleDB")
    except Exception as e:
        logger.error(f"Error al guardar señal: {e}")


def save_metrics(cv_score, cv_std, timestamp):
    """Guarda métricas de validación en TimescaleDB."""
    try:
        conn = psycopg2.connect(
            host=os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
            port=os.getenv('TIMESCALEDB_PORT', '5432'),
            database=os.getenv('TIMESCALEDB_DB', 'postgres'),
            user=os.getenv('TIMESCALEDB_USER', 'admin'),
            password=os.getenv('TIMESCALEDB_PASSWORD', 'password')
        )
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_metrics (timestamp, cv_score, cv_std)
            VALUES (%s, %s, %s)
            """,
            (timestamp, float(cv_score), float(cv_std))
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Métricas guardadas en TimescaleDB")
    except Exception as e:
        logger.error(f"Error al guardar métricas: {e}")


def train_model(X, y):
    """Entrena el modelo LightGBM con hiperparámetros fijos."""
    try:
        if len(X) != len(y):
            logger.error(f"Desalineación: X={len(X)}, y={len(y)}")
            return None
        model = lgb.LGBMClassifier(
            class_weight='balanced',
            min_child_samples=10,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            force_col_wise=True
        )
        # Timeout para evitar entrenamientos largos

        def timeout_handler(signum, frame):
            raise TimeoutError("Entrenamiento excedió el tiempo límite")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minutos
        model.fit(X, y)
        signal.alarm(0)  # Desactivar timeout
        # Validación cruzada
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_score = cv_scores.mean()
        cv_std = cv_scores.std()
        logger.info(f"Cross-validation score: {cv_score:.4f} ± {cv_std:.4f}")
        save_metrics(cv_score, cv_std, datetime.now())
        logger.info("Entrenamiento completado")
        return model
    except TimeoutError:
        logger.error("Entrenamiento cancelado por timeout")
        return None
    except Exception as e:
        logger.error(f"Error al entrenar modelo: {e}")
        return None


def generate_signal(model, X):
    """Genera una señal de trading."""
    try:
        if model is None or X.shape[0] == 0:
            logger.error("Modelo o datos inválidos")
            return None
        pred = model.predict(X.iloc[-1:])
        signal = 'buy' if pred[0] == 1 else 'sell'
        logger.info(f"Señal generada: {signal} at {datetime.now()}")
        return signal
    except Exception as e:
        logger.error(f"Error al generar señal: {e}")
        return None


def main():
    """Función principal para generar señales."""
    logger.info("Iniciando signals.py")
    df = fetch_data()
    if df is None or len(df) < 50:
        logger.error("Datos insuficientes")
        return

    df['target'] = (df['bid_price'].shift(-1) > df['bid_price']).astype(int)
    X, df_filtered = prepare_features(df)
    if X is None or df_filtered is None:
        logger.error("Fallo en preparación de features")
        return
    y = df_filtered['target'].iloc[:len(X)]

    logger.info("Iniciando entrenamiento")
    model = train_model(X[:-1], y[:-1])
    if model:
        logger.info("Generando señal")
        signal = generate_signal(model, X)
        if signal:
            save_signal(signal, datetime.now())
            logger.info(f"Señal final: {signal}")
    else:
        logger.error("Fallo en entrenamiento del modelo")


if __name__ == "__main__":
    main()
