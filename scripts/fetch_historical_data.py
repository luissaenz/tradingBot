import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_binance_klines(symbol, interval, start_time, limit=1000):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "limit": limit
    }
    logger.info(f"Obteniendo datos desde {start_time}")
    for attempt in range(3):  # Reintentos para rate-limiting
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
        except requests.exceptions.RequestException as e:
            logger.warning(f"Intento {attempt+1} fallido: {e}")
            time.sleep(2 ** attempt)
    logger.error("Fallo tras reintentos")
    return pd.DataFrame()


def fetch_historical_data(symbol="BTCUSDT", interval="5m",
                          start_date="2023-01-01", end_date="2025-05-16"):
    start_time = pd.to_datetime(start_date)
    end_time = pd.to_datetime(end_date)
    all_data = []
    limit = 1000
    while start_time < end_time:
        df = fetch_binance_klines(symbol, interval, start_time, limit)
        if df.empty:
            logger.warning("Datos vacíos, deteniendo.")
            break
        all_data.append(df)
        start_time = pd.to_datetime(
            df['timestamp'].iloc[-1], unit='ms') + timedelta(minutes=5)
        logger.info(
            f"Avanzando a {start_time}, filas totales: {sum(len(d) for d in all_data)}")
        time.sleep(0.2)  # Evitar rate-limiting
    if not all_data:
        logger.error("No se obtuvieron datos.")
        return pd.DataFrame()
    df = pd.concat(all_data, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low',
             'close', 'volume', 'taker_buy_volume']]
    df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']] = df[[
        'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']].astype(float)
    df['symbol'] = symbol.replace("USDT", "/USDT")
    # Calcular imbalance y delta_volume
    df['imbalance'] = (df['taker_buy_volume'] -
                       (df['volume'] - df['taker_buy_volume'])) / df['volume']
    df['delta_volume'] = df['taker_buy_volume'] - \
        (df['volume'] - df['taker_buy_volume'])
    df = df[['timestamp', 'open', 'high', 'low', 'close',
             'volume', 'symbol', 'imbalance', 'delta_volume']]
    # Verificar continuidad temporal
    df = df.sort_values('timestamp')
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    gaps = df[df['time_diff'] > 300]  # 5 minutos = 300 segundos
    if not gaps.empty:
        logger.warning(f"Detectados {len(gaps)} huecos temporales")
    df = df.drop(columns=['time_diff'])
    # Eliminar duplicados
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    return df


if __name__ == "__main__":
    logger.info("Iniciando descarga de datos históricos")
    df = fetch_historical_data()
    os.makedirs('/app/data', exist_ok=True)
    df.to_parquet('/app/data/ohlcv_historical.parquet')
    logger.info(
        f"Guardadas {len(df)} filas en /app/data/ohlcv_historical.parquet")
