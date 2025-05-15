import os
import pandas as pd
from sqlalchemy import create_engine
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def fetch_data():
    try:
        engine = create_engine(
            f"postgresql://admin:password@{os.getenv('TIMESCALEDB_HOST', 'timescaledb')}:{os.getenv('TIMESCALEDB_PORT', '5432')}/{os.getenv('TIMESCALEDB_DB', 'postgres')}")
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


def save_signal(signal, timestamp, price):
    try:
        engine = create_engine(
            f"postgresql://admin:password@{os.getenv('TIMESCALEDB_HOST', 'timescaledb')}:{os.getenv('TIMESCALEDB_PORT', '5432')}/{os.getenv('TIMESCALEDB_DB', 'postgres')}")
        with engine.connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (order_id, timestamp, symbol, side, price, quantity)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (f"signal_only_{timestamp.strftime('%Y%m%d%H%M%S')}",
                 timestamp, "BTC/USDT", signal, price, 0.0)
            )
            logger.info(f"Señal guardada en trades: {signal}")
    except Exception as e:
        logger.error(f"Error al guardar señal: {e}")


def save_model_metrics(cv_score, cv_std, timestamp):
    try:
        engine = create_engine(
            f"postgresql://admin:password@{os.getenv('TIMESCALEDB_HOST', 'timescaledb')}:{os.getenv('TIMESCALEDB_PORT', '5432')}/{os.getenv('TIMESCALEDB_DB', 'postgres')}")
        with engine.connect() as conn:
            conn.execute(
                """
                INSERT INTO model_metrics (timestamp, cv_score, cv_std)
                VALUES (%s, %s, %s)
                """,
                (timestamp, cv_score, cv_std)
            )
            logger.info(
                f"Métricas guardadas: CV Score={cv_score:.4f}, CV Std={cv_std:.4f}")
    except Exception as e:
        logger.error(f"Error al guardar métricas: {e}")


def prepare_features(df):
    df['price'] = (df['bid_price'] + df['ask_price']) / 2
    df['spread'] = df['ask_price'] - df['bid_price']
    df['imbalance_lag1'] = df['imbalance'].shift(1)
    df['delta_volume_lag1'] = df['delta_volume'].shift(1)
    df['price_ma5'] = df['price'].rolling(window=5).mean()
    df['price_ema10'] = df['price'].ewm(span=10, adjust=False).mean()
    df['total_volume'] = df['bid_volume'] + df['ask_volume']
    df['rsi_14'] = calculate_rsi(df['price'], periods=14)
    df['volatility_20'] = df['price'].rolling(window=20).std()
    features = ['imbalance', 'delta_volume', 'spread', 'imbalance_lag1', 'delta_volume_lag1',
                'price_ma5', 'price_ema10', 'total_volume', 'rsi_14', 'volatility_20']
    df_features = df[features].dropna()
    logger.info(f"Datos tras preparar features: {len(df_features)} filas")
    return df_features, df['price'].iloc[df_features.index]


def train_model(X, y):
    try:
        model = lgb.LGBMClassifier(
            class_weight='balanced', min_child_samples=10)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [20, 31]
        }
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        logger.info(
            f"Cross-validation score: {grid_search.best_score_:.4f} (+/- {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f})")
        return best_model, grid_search.best_score_, grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    except Exception as e:
        logger.error(f"Error al entrenar modelo: {e}")
        return None, None, None


def generate_signal(model, X, scaler, price):
    try:
        X_scaled = scaler.transform(X[-1:])
        pred = model.predict(X_scaled)
        signal = 'buy' if pred[0] == 1 else 'sell'
        logger.info(
            f"Señal generada: {signal} at {datetime.now()}, price: {price.iloc[-1]}")
        return signal
    except Exception as e:
        logger.error(f"Error al generar señal: {e}")
        return None


def main():
    df = fetch_data()
    if df is None or len(df) < 50:
        logger.error("Datos insuficientes")
        return

    df['target'] = (df['bid_price'].shift(-1) > df['bid_price']).astype(int)
    X, prices = prepare_features(df)
    y = df['target'].iloc[X.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model, cv_score, cv_std = train_model(X_scaled[:-1], y[:-1])
    if model:
        signal = generate_signal(model, X, scaler, prices)
        if signal:
            timestamp = datetime.now()
            save_signal(signal, timestamp, prices.iloc[-1])
            if cv_score is not None and cv_std is not None:
                save_model_metrics(cv_score, cv_std, timestamp)
            logger.info(f"Señal final: {signal}")


if __name__ == "__main__":
    main()
