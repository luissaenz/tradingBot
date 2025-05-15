import os
import asyncio
import ccxt.async_support as ccxt
import psycopg2
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class TradeExecutor:
    def __init__(self):
        """Initialize the trade executor."""
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'testnet': True  # Using testnet for safety
        })
        self.symbol = 'BTC/USDT'
        self.leverage = 5
        self.risk_per_trade = 0.02  # 2% of capital per trade
        self.sl_percentage = 0.002  # 0.2% initial stop-loss
        self.db_params = {
            'host': os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
            'port': os.getenv('TIMESCALEDB_PORT', '5432'),
            'database': os.getenv('TIMESCALEDB_DB', 'postgres'),
            'user': os.getenv('TIMESCALEDB_USER', 'admin'),
            'password': os.getenv('TIMESCALEDB_PASSWORD', 'password')
        }

    async def initialize_exchange(self):
        """Configure the exchange."""
        try:
            await self.exchange.load_markets()
            # Set leverage for futures trading
            await self.exchange.fapiPrivate_post_leverage({
                'symbol': self.symbol.replace('/', ''),
                'leverage': self.leverage
            })
            logger.info(
                f"Exchange initialized: {self.symbol}, leverage {self.leverage}x")
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise

    async def get_account_balance(self):
        """Get the account balance."""
        try:
            balance = await self.exchange.fapiPrivate_get_balance()
            for asset in balance:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
            logger.error("USDT not found in balance")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0

    def fetch_pending_signals(self):
        """Extract pending signals from TimescaleDB."""
        try:
            conn = psycopg2.connect(**self.db_params)
            query = """
            SELECT order_id, timestamp, symbol, side
            FROM trades
            WHERE price = 0 AND quantity = 0
            ORDER BY timestamp DESC LIMIT 1
            """
            df = pd.read_sql(query, conn)
            conn.close()
            logger.info(f"Pending signals extracted: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Error extracting signals: {e}")
            return pd.DataFrame()

    async def calculate_quantity(self, price, balance):
        """Calculate the trading quantity."""
        try:
            capital = balance * self.risk_per_trade
            quantity = (capital * self.leverage) / price
            # Adjust precision according to Binance requirements
            return round(quantity, 3)
        except Exception as e:
            logger.error(f"Error calculating quantity: {e}")
            return 0.0

    async def set_stop_loss(self, side, entry_price):
        """Calculate stop-loss price."""
        try:
            if side == 'buy':
                sl_price = entry_price * (1 - self.sl_percentage)
            else:
                sl_price = entry_price * (1 + self.sl_percentage)
            return round(sl_price, 2)
        except Exception as e:
            logger.error(f"Error calculating stop-loss: {e}")
            return 0.0

    async def execute_trade(self, signal):
        """Execute a trade based on the signal."""
        try:
            side = signal['side']
            await self.exchange.load_markets()
            ticker = await self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']

            balance = await self.get_account_balance()
            if balance <= 0:
                logger.error("Insufficient balance")
                return

            quantity = await self.calculate_quantity(price, balance)
            if quantity <= 0:
                logger.error("Invalid quantity")
                return

            sl_price = await self.set_stop_loss(side, price)

            # Execute market order
            if side == 'buy':
                order = await self.exchange.create_market_buy_order(self.symbol, quantity)
            else:
                order = await self.exchange.create_market_sell_order(self.symbol, quantity)

            # Record trade
            self.save_trade(signal['order_id'], signal['timestamp'], order)

            # Set stop-loss order
            if sl_price > 0:
                sl_side = 'sell' if side == 'buy' else 'buy'
                await self.exchange.create_stop_loss_order(self.symbol, sl_side, quantity, sl_price)

            logger.info(
                f"Trade executed: {side}, price={price}, quantity={quantity}, sl_price={sl_price}")
        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def save_trade(self, order_id, timestamp, order):
        """Save trade details to TimescaleDB."""
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE trades
                SET price = %s, quantity = %s
                WHERE order_id = %s AND timestamp = %s
                """,
                (
                    float(order['price']) if 'price' in order else 0.0,
                    float(order['amount']) if 'amount' in order else 0.0,
                    order_id,
                    timestamp
                )
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Trade updated in TimescaleDB")
        except Exception as e:
            logger.error(f"Error saving trade: {e}")

    async def run(self):
        """Main loop to process signals."""
        await self.initialize_exchange()
        try:
            while True:
                signals = self.fetch_pending_signals()
                if not signals.empty:
                    for _, signal in signals.iterrows():
                        await self.execute_trade(signal)
                else:
                    logger.info("No pending signals")
                # Check for new signals every minute
                await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise

    async def close(self):
        """Close the exchange connection."""
        try:
            await self.exchange.close()
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing exchange: {e}")


async def main():
    """Main function."""
    executor = TradeExecutor()
    try:
        await executor.run()
    except KeyboardInterrupt:
        logger.info("Closing TradeExecutor due to keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        await executor.close()

if __name__ == "__main__":
    asyncio.run(main())
