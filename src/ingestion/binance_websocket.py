import os
import asyncio
import json
import logging
import psycopg2
from dotenv import load_dotenv
import websockets
from prometheus_client import Counter, Histogram
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Métricas de Prometheus
order_book_requests = Counter(
    'order_book_requests_total', 'Total de solicitudes de order book')
order_book_latency = Histogram(
    'order_book_latency_seconds', 'Latencia de solicitud de order book')


def calculate_metrics(order_book):
    """Calcula el desequilibrio y el volumen delta."""
    bids = [(float(price), float(volume))
            for price, volume in order_book['bids'][:5]]
    asks = [(float(price), float(volume))
            for price, volume in order_book['asks'][:5]]

    bid_volume = sum(volume for _, volume in bids)
    ask_volume = sum(volume for _, volume in asks)

    imbalance = (bid_volume - ask_volume) / (bid_volume +
                                             ask_volume) if (bid_volume + ask_volume) > 0 else 0
    delta_volume = bid_volume - ask_volume

    return imbalance, delta_volume


def save_to_timescaledb(order_book, imbalance, delta_volume):
    """Guarda los datos en TimescaleDB."""
    try:
        conn = psycopg2.connect(
            host=os.getenv('TIMESCALEDB_HOST', 'localhost'),
            port=os.getenv('TIMESCALEDB_PORT', '5432'),
            database=os.getenv('TIMESCALEDB_DB', 'postgres'),
            user=os.getenv('TIMESCALEDB_USER', 'admin'),
            password=os.getenv('TIMESCALEDB_PASSWORD', 'password')
        )
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO microstructure_data (
                timestamp, symbol, bid_price, ask_price, bid_volume, ask_volume, imbalance, delta_volume
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                datetime.fromtimestamp(order_book['timestamp'] / 1000.0),
                "BTC/USDT",
                float(order_book['bids'][0][0]
                      ) if order_book['bids'] else None,
                float(order_book['asks'][0][0]
                      ) if order_book['asks'] else None,
                float(order_book['bids'][0][1]
                      ) if order_book['bids'] else None,
                float(order_book['asks'][0][1]
                      ) if order_book['asks'] else None,
                imbalance,
                delta_volume
            )
        )
        conn.commit()
        logger.info("Datos guardados en TimescaleDB")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error al guardar en TimescaleDB: {e}")


async def fetch_order_book(symbol="btcusdt"):
    """Conecta al WebSocket de Binance y procesa el order book."""
    logger.info("Iniciando WebSocket para Binance...")

    is_testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    if is_testnet:
        ws_url = f"wss://stream.testnet.binance.vision/stream?streams={symbol.lower()}@depth@100ms"
    else:
        ws_url = f"wss://fstream.binance.com/stream?streams={symbol.lower()}@depth@100ms"

    async with websockets.connect(ws_url) as websocket:
        logger.info(f"Conectado al stream: {symbol.lower()}@depth@100ms")

        while True:
            try:
                with order_book_latency.time():
                    message = await websocket.recv()
                    data = json.loads(message)
                    order_book_requests.inc()

                # Verificar si es un mensaje de suscripción exitoso
                if 'result' in data and data['result'] is None:
                    logger.info("Suscripción exitosa al stream.")
                    continue

                # Procesar el mensaje del stream combinado
                if 'data' in data:
                    payload = data['data']
                    if 'b' in payload and 'a' in payload and 'E' in payload:
                        order_book = {
                            'bids': payload['b'],
                            'asks': payload['a'],
                            'timestamp': payload['E']
                        }
                        imbalance, delta_volume = calculate_metrics(order_book)
                        logger.info(
                            f"Order book: {order_book}, Imbalance: {imbalance:.4f}, Delta Volume: {delta_volume:.2f}")
                        save_to_timescaledb(
                            order_book, imbalance, delta_volume)
                    else:
                        logger.warning(
                            f"Estructura de mensaje inesperada: {payload}")
                else:
                    logger.warning(f"Mensaje inesperado: {data}")
            except Exception as e:
                logger.error(f"Error en la conexión: {e}")
                break

if __name__ == "__main__":
    asyncio.run(fetch_order_book())
