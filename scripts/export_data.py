import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
db_url = f"postgresql://{os.getenv('TIMESCALEDB_USER')}:{os.getenv('TIMESCALEDB_PASSWORD')}@{os.getenv('TIMESCALEDB_HOST')}:{os.getenv('TIMESCALEDB_PORT')}/{os.getenv('TIMESCALEDB_DB')}"
engine = create_engine(db_url)

# Exportar trades
query_trades = "SELECT timestamp, symbol, price, quantity, side FROM trades WHERE timestamp >= NOW() - INTERVAL '2 years'"
trades_df = pd.read_sql(query_trades, engine)
trades_df.to_parquet('/app/data/trades.parquet')

# Exportar microstructure_data
query_micro = "SELECT timestamp, symbol, bid_price AS price, bid_volume AS volume, imbalance, delta_volume FROM microstructure_data WHERE timestamp >= NOW() - INTERVAL '2 years'"
micro_df = pd.read_sql(query_micro, engine)
# Crear OHLCV
ohlcv_df = micro_df.groupby('symbol').resample('5min', on='timestamp').agg({
    'price': ['first', 'max', 'min', 'last'],
    'volume': 'sum',
    'imbalance': 'mean',
    'delta_volume': 'sum'
}).reset_index()
ohlcv_df.columns = ['symbol', 'timestamp', 'open', 'high',
                    'low', 'close', 'volume', 'imbalance', 'delta_volume']
# Rellenar NaN
ohlcv_df[['open', 'high', 'low', 'close']] = ohlcv_df[[
    'open', 'high', 'low', 'close']].ffill()
ohlcv_df[['imbalance', 'delta_volume']] = ohlcv_df[[
    'imbalance', 'delta_volume']].fillna(0)
ohlcv_df.to_parquet('/app/data/ohlcv.parquet')
