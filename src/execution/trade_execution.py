import psycopg2
import os
import requests
import hmac
import hashlib
import time
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import psycopg2

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


class TradeExecutor:
    def __init__(self):
        """Inicializa el ejecutor de trades."""
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.base_url = 'https://testnet.binancefuture.com'
        self.symbol = 'BTCUSDT'
        self.leverage = 5
        self.risk_per_trade = 0.02  # 2% del capital
        self.sl_percentage = 0.002  # 0.2% stop-loss inicial
        self.max_retries = 3  # Máximo de reintentos para confirmar llenado
        self.retry_delay = 5  # Segundos entre reintentos
        self.engine = create_engine(
            f"postgresql://admin:password@{os.getenv('TIMESCALEDB_HOST', 'timescaledb')}:{os.getenv('TIMESCALEDB_PORT', '5432')}/{os.getenv('TIMESCALEDB_DB', 'postgres')}"
        )

    def sign_request(self, params):
        """Genera la firma HMAC-SHA256 para autenticación."""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params

    def initialize_exchange(self):
        """Configura el exchange (apalancamiento)."""
        try:
            endpoint = '/fapi/v1/leverage'
            params = {
                'symbol': self.symbol,
                'leverage': self.leverage,
                'timestamp': int(time.time() * 1000)
            }
            params = self.sign_request(params)
            headers = {'X-MBX-APIKEY': self.api_key}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            logger.info(
                f"Exchange inicializado: {self.symbol}, leverage {self.leverage}x")
        except Exception as e:
            logger.error(f"Error al inicializar exchange: {e}")

    def get_account_balance(self):
        """Obtiene el balance de USDT."""
        try:
            endpoint = '/fapi/v2/balance'
            params = {'timestamp': int(time.time() * 1000)}
            params = self.sign_request(params)
            headers = {'X-MBX-APIKEY': self.api_key}
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            balances = response.json()
            for asset in balances:
                if asset['asset'] == 'USDT':
                    balance = float(asset['availableBalance'])
                    logger.info(f"Balance USDT: {balance}")
                    return balance
            logger.error("USDT no encontrado en balance")
            return 0.0
        except Exception as e:
            logger.error(f"Error al obtener balance: {e}")
            return 0.0

    def check_open_positions(self, side):
        """Verifica posiciones abiertas para evitar acumulación de riesgo."""
        try:
            endpoint = '/fapi/v2/positionRisk'
            params = {'symbol': self.symbol,
                      'timestamp': int(time.time() * 1000)}
            params = self.sign_request(params)
            headers = {'X-MBX-APIKEY': self.api_key}
            response = requests.get(
                f"{self.base_url}{endpoint}", headers=headers, params=params)
            response.raise_for_status()
            positions = response.json()
            for pos in positions:
                if pos['symbol'] == self.symbol and float(pos['positionAmt']) != 0:
                    position_side = 'BUY' if float(
                        pos['positionAmt']) > 0 else 'SELL'
                    if position_side == side.upper():
                        logger.warning(
                            f"Posición abierta existente: {position_side}, cantidad={pos['positionAmt']}")
                        return False
            logger.info("No hay posiciones abiertas en la misma dirección")
            return True
        except Exception as e:
            logger.error(f"Error al verificar posiciones: {e}")
            return False

    def fetch_pending_signals(self):
        """Extrae señales pendientes de TimescaleDB."""
        try:
            query = """
            SELECT order_id, timestamp, symbol, side
            FROM trades
            WHERE status = 'pending' AND price = 0 AND quantity = 0
            ORDER BY timestamp DESC LIMIT 1
            """
            df = pd.read_sql(query, self.engine)
            logger.info(f"Señales pendientes extraídas: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Error al extraer señales: {e}")
            return pd.DataFrame()

    def get_current_price(self):
        """Obtiene el precio actual del símbolo."""
        try:
            endpoint = '/fapi/v1/ticker/price'
            params = {'symbol': self.symbol}
            response = requests.get(
                f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            price = float(response.json()['price'])
            logger.info(f"Precio actual: {price}")
            return price
        except Exception as e:
            logger.error(f"Error al obtener precio: {e}")
            return 0.0

    def calculate_quantity(self, price, balance):
        """Calcula la cantidad a operar."""
        try:
            capital = balance * self.risk_per_trade
            quantity = (capital * self.leverage) / price
            quantity = round(quantity, 3)
            logger.info(
                f"Cantidad calculada: {quantity} (capital={capital}, leverage={self.leverage}, price={price})")
            return quantity
        except Exception as e:
            logger.error(f"Error al calcular cantidad: {e}")
            return 0.0

    def set_stop_loss(self, side, entry_price):
        """Calcula el precio de stop-loss."""
        try:
            if side == 'buy':
                sl_price = entry_price * (1 - self.sl_percentage)
            else:
                sl_price = entry_price * (1 + self.sl_percentage)
            sl_price = round(sl_price, 2)
            logger.info(
                f"Stop-loss calculado: {sl_price} (side={side}, entry_price={entry_price})")
            return sl_price
        except Exception as e:
            logger.error(f"Error al calcular stop-loss: {e}")
            return 0.0

    def get_trade_details(self, order_id):
        """Consulta detalles del llenado de la orden."""
        try:
            endpoint = '/fapi/v1/userTrades'
            params = {
                'symbol': self.symbol,
                'orderId': order_id,
                'timestamp': int(time.time() * 1000)
            }
            params = self.sign_request(params)
            headers = {'X-MBX-APIKEY': self.api_key}
            response = requests.get(
                f"{self.base_url}{endpoint}", headers=headers, params=params)
            response.raise_for_status()
            trades = response.json()
            if trades:
                avg_price = sum(float(t['price']) * float(t['qty'])
                                for t in trades) / sum(float(t['qty']) for t in trades)
                total_qty = sum(float(t['qty']) for t in trades)
                logger.info(
                    f"Trade details: orderId={order_id}, avgPrice={avg_price}, totalQty={total_qty}")
                return avg_price, total_qty
            logger.warning(f"No se encontraron trades para orderId={order_id}")
            return 0.0, 0.0
        except Exception as e:
            logger.error(f"Error al obtener detalles del trade: {e}")
            return 0.0, 0.0

    def archive_old_trades(self):
        """Archiva trades antiguos (>30 días) y los elimina de trades."""
        try:
            conn = psycopg2.connect(
                host=os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
                port=os.getenv('TIMESCALEDB_PORT', '5432'),
                database=os.getenv('TIMESCALEDB_DB', 'postgres'),
                user=os.getenv('TIMESCALEDB_USER', 'admin'),
                password=os.getenv('TIMESCALEDB_PASSWORD', 'password')
            )
            cursor = conn.cursor()
            # Crear tabla trades_archive si no existe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades_archive (
                    order_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price FLOAT NOT NULL,
                    quantity FLOAT NOT NULL,
                    status TEXT NOT NULL
                );
            """)
            # Archivar trades antiguos
            cursor.execute("""
                INSERT INTO trades_archive
                SELECT * FROM trades
                WHERE timestamp < %s
                ON CONFLICT (order_id) DO NOTHING;
            """, (datetime.now() - timedelta(days=30),))
            archived_count = cursor.rowcount
            # Eliminar trades archivados
            cursor.execute("""
                DELETE FROM trades
                WHERE timestamp < %s;
            """, (datetime.now() - timedelta(days=30),))
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(
                f"Trades archivados: {archived_count}, eliminados: {deleted_count}")
        except Exception as e:
            logger.error(f"Error al archivar trades: {e}")

    def execute_trade(self, signal):
        """Ejecuta un trade basado en la señal."""
        try:
            side = signal['side'].upper()
            # Verificar posiciones abiertas
            if not self.check_open_positions(side):
                logger.error(
                    f"No se puede ejecutar trade: posición abierta en {side}")
                return

            price = self.get_current_price()
            if price <= 0:
                logger.error("Precio inválido")
                return

            balance = self.get_account_balance()
            if balance <= 0:
                logger.error("Balance insuficiente")
                return

            quantity = self.calculate_quantity(price, balance)
            if quantity <= 0:
                logger.error("Cantidad inválida")
                return

            sl_price = self.set_stop_loss(side.lower(), price)
            if sl_price <= 0:
                logger.error("Stop-loss inválido")
                return

            # Crear  Crear orden de mercado
            endpoint = '/fapi/v1/order'
            params = {
                'symbol': self.symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': quantity,
                'timestamp': int(time.time() * 1000)
            }
            params = self.sign_request(params)
            headers = {'X-MBX-APIKEY': self.api_key}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            order = response.json()
            logger.info(f"Respuesta de orden: {json.dumps(order, indent=2)}")

            # Confirmar llenado con reintentos
            final_price = float(order.get('avgPrice', 0.0))
            final_quantity = float(
                order.get('executedQty', order.get('origQty', 0.0)))
            if order['status'] == 'NEW':
                for attempt in range(self.max_retries):
                    time.sleep(self.retry_delay)
                    avg_price, total_qty = self.get_trade_details(
                        order['orderId'])
                    if avg_price > 0 and total_qty > 0:
                        final_price = avg_price
                        final_quantity = total_qty
                        break
                    logger.warning(
                        f"Intento {attempt + 1}/{self.max_retries}: No se encontraron detalles del trade")
                else:
                    logger.warning(
                        f"Reintentos agotados, usando market_price={price}, calculated_quantity={quantity}")
                    final_price = price
                    final_quantity = quantity

            # Guardar trade
            self.save_trade(
                signal['order_id'], signal['timestamp'], final_price, final_quantity)

            # Crear orden stop-loss
            sl_side = 'SELL' if side == 'BUY' else 'BUY'
            sl_params = {
                'symbol': self.symbol,
                'side': sl_side,
                'type': 'STOP_MARKET',
                'quantity': quantity,
                'stopPrice': sl_price,
                'timestamp': int(time.time() * 1000)
            }
            sl_params = self.sign_request(sl_params)
            sl_response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=sl_params
            )
            sl_response.raise_for_status()
            logger.info(f"Stop-loss creado: {sl_side}, stopPrice={sl_price}")

            logger.info(
                f"Trade ejecutado: {side.lower()}, price={final_price}, quantity={final_quantity}, sl_price={sl_price}")
        except Exception as e:
            logger.error(f"Error al ejecutar trade: {e}")

    def save_trade(self, order_id, timestamp, price, quantity):
        """Guarda los detalles del trade en TimescaleDB."""
        try:
            conn = psycopg2.connect(
                host=os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
                port=os.getenv('TIMESCALEDB_PORT', '5432'),
                database=os.getenv('TIMESCALEDB_DB', 'postgres'),
                user=os.getenv('TIMESCALEDB_USER', 'admin'),
                password=os.getenv('TIMESCALEDB_PASSWORD', 'password')
            )
            cursor = conn.cursor()
            logger.info(
                f"Guardando trade: order_id={order_id}, price={price}, quantity={quantity}")
            cursor.execute(
                """
                UPDATE trades
                SET price = %s, quantity = %s, status = %s
                WHERE order_id = %s AND timestamp = %s
                """,
                (
                    price,
                    quantity,
                    'executed',
                    order_id,
                    timestamp
                )
            )
            if cursor.rowcount == 0:
                logger.error(
                    f"No se actualizó ningún trade para order_id={order_id}, timestamp={timestamp}")
            else:
                logger.info(
                    f"Trade actualizado: order_id={order_id}, price={price}, quantity={quantity}")
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error al guardar trade: {e}")

    def run(self):
        """Bucle principal para procesar señales."""
        self.initialize_exchange()
        while True:
            self.archive_old_trades()  # Archivar trades antiguos
            signals = self.fetch_pending_signals()
            if not signals.empty:
                for _, signal in signals.iterrows():
                    self.execute_trade(signal)
            else:
                logger.info("No hay señales pendientes")
            time.sleep(60)

    def close(self):
        """Cierra la conexión a la base de datos."""
        try:
            self.engine.dispose()
            logger.info("Conexión a base de datos cerrada")
        except Exception as e:
            logger.error(f"Error al cerrar conexiones: {e}")


def main():
    """Función principal."""
    executor = TradeExecutor()
    try:
        executor.run()
    except KeyboardInterrupt:
        logger.info("Cerrando TradeExecutor")
    finally:
        executor.close()


if __name__ == "__main__":
    main()
