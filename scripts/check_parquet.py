import pandas as pd
import numpy as np


def check_parquet():
    try:
        df = pd.read_parquet('/app/data/ohlcv_historical.parquet')

        print("\n=== Resumen de datos ===")
        print(f"Total filas: {len(df)}")

        # Convertir timestamp a datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"\nRango temporal:")
        print(f"Desde: {df['datetime'].min()}")
        print(f"Hasta: {df['datetime'].max()}")

        print("\n=== Valores nulos ===")
        print(df.isna().sum())

        print("\n=== Estadísticas básicas ===")
        print(df[['open', 'high', 'low', 'close', 'volume']].describe())

        return True
    except Exception as e:
        print(f"\nError al verificar archivo: {str(e)}")
        return False


if __name__ == "__main__":
    check_parquet()
