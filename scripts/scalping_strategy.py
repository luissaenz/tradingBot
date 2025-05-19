import backtrader as bt
import pandas as pd
from datetime import datetime
import pytz


class ScalpingStrategy(bt.Strategy):
    params = (
        ('ma_period', 5),
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
        ('bb_period', 20),
        ('bb_dev', 1.0),
        ('stop_loss', 0.002),
        ('take_profit', 0.007),
        ('volume_threshold', 20.0),
    )

    def __init__(self):
        self.ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.ma_period)
        self.rsi = bt.indicators.RSI(
            self.data.close, period=self.params.rsi_period)
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.params.bb_period, devfactor=self.params.bb_dev)
        self.order = None

    def next(self):
        if self.order or self.data.volume[0] < self.params.volume_threshold:
            return
        price = self.data.close[0]
        size = 0.001
        if (self.data.close[0] > self.ma[0] and self.rsi[0] < self.params.rsi_high and
                self.data.close[0] < self.bb.lines.top[0]):
            self.order = self.buy(size=size)
            self.sell(exectype=bt.Order.Stop, price=price *
                      (1 - self.params.stop_loss), size=size)
            self.sell(exectype=bt.Order.Limit, price=price *
                      (1 + self.params.take_profit), size=size)
        elif (self.data.close[0] < self.ma[0] and self.rsi[0] > self.params.rsi_low and
              self.data.close[0] > self.bb.lines.bot[0]):
            self.order = self.sell(size=size)
            self.buy(exectype=bt.Order.Stop, price=price *
                     (1 + self.params.stop_loss), size=size)
            self.buy(exectype=bt.Order.Limit, price=price *
                     (1 - self.params.take_profit), size=size)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


class RelaxedScalpingStrategy(bt.Strategy):
    params = (
        ('ma_period', 5),
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
        ('stop_loss', 0.002),
        ('take_profit', 0.003),
        ('volume_threshold', 0.05),
    )

    def __init__(self):
        # Simple moving average
        self.ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.ma_period)
        # RSI indicator
        self.rsi = bt.indicators.RSI(
            self.data.close, period=self.params.rsi_period)
        # Track open orders
        self.order = None
        # Volume average for threshold comparison
        self.volume_avg = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=20)

    def next(self):
        # Don't enter if we have an open order
        if self.order:
            return

        # Check if current volume is above our volume threshold compared to average
        if self.data.volume[0] < self.volume_avg[0] * self.params.volume_threshold:
            return

        price = self.data.close[0]

        # Long signal: price above MA, RSI not overbought
        if self.data.close[0] > self.ma[0] and self.rsi[0] < self.params.rsi_high:
            self.order = self.buy()
            # Set stop loss and take profit orders
            self.sell(exectype=bt.Order.Stop, price=price *
                      (1 - self.params.stop_loss))
            self.sell(exectype=bt.Order.Limit, price=price *
                      (1 + self.params.take_profit))

        # Short signal: price below MA, RSI not oversold
        elif self.data.close[0] < self.ma[0] and self.rsi[0] > self.params.rsi_low:
            self.order = self.sell()
            # Set stop loss and take profit orders
            self.buy(exectype=bt.Order.Stop, price=price *
                     (1 + self.params.stop_loss))
            self.buy(exectype=bt.Order.Limit, price=price *
                     (1 - self.params.take_profit))

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


class StrictScalpingStrategy(bt.Strategy):
    params = (
        ('ma_period', 5),
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
        ('stop_loss', 0.002),
        ('take_profit', 0.003),
        ('volume_threshold', 2.0),
    )

    def __init__(self):
        # Moving averages for trend detection
        self.ma_short = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.ma_period)
        self.ma_long = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.ma_period * 3)
        # RSI for overbought/oversold detection
        self.rsi = bt.indicators.RSI(
            self.data.close, period=self.params.rsi_period)
        # Bollinger Bands for volatility
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=20, devfactor=2.0)
        # Track open orders
        self.order = None
        # Volume average for threshold comparison
        self.volume_avg = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=20)

    def next(self):
        # Don't enter if we have an open order
        if self.order:
            return

        # Strict volume filter
        if self.data.volume[0] < self.volume_avg[0] * self.params.volume_threshold:
            return

        price = self.data.close[0]

        # Long signal: strong trend, oversold RSI, price near lower BB
        if (self.ma_short[0] > self.ma_long[0] and
            self.rsi[0] < self.params.rsi_low and
                self.data.close[0] < self.bb.lines.bot[0] * 1.01):
            self.order = self.buy()
            # Set stop loss and take profit orders
            self.sell(exectype=bt.Order.Stop, price=price *
                      (1 - self.params.stop_loss))
            self.sell(exectype=bt.Order.Limit, price=price *
                      (1 + self.params.take_profit))

        # Short signal: downtrend, overbought RSI, price near upper BB
        elif (self.ma_short[0] < self.ma_long[0] and
              self.rsi[0] > self.params.rsi_high and
              self.data.close[0] > self.bb.lines.top[0] * 0.99):
            self.order = self.sell()
            # Set stop loss and take profit orders
            self.buy(exectype=bt.Order.Stop, price=price *
                     (1 + self.params.stop_loss))
            self.buy(exectype=bt.Order.Limit, price=price *
                     (1 - self.params.take_profit))

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


class VolumeBreakoutStrategy(bt.Strategy):
    params = (
        ('ma_period', 10),
        ('volume_threshold', 20.0),
        ('stop_loss', 0.002),
        ('take_profit', 0.003),
    )

    def __init__(self):
        # Price moving average
        self.ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.ma_period)
        # Volume indicators
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.params.ma_period)
        # Track open orders
        self.order = None
        # High and low prices
        self.highest = bt.indicators.Highest(
            self.data.high, period=self.params.ma_period)
        self.lowest = bt.indicators.Lowest(
            self.data.low, period=self.params.ma_period)

    def next(self):
        # Don't enter if we have an open order
        if self.order:
            return

        # Check for volume breakout
        if self.data.volume[0] <= self.volume_ma[0] * self.params.volume_threshold:
            return

        price = self.data.close[0]

        # Long signal: price breakout with volume spike
        if (self.data.close[0] > self.highest[-1] and
                self.data.close[0] > self.ma[0]):
            self.order = self.buy()
            # Set stop loss and take profit orders
            self.sell(exectype=bt.Order.Stop, price=price *
                      (1 - self.params.stop_loss))
            self.sell(exectype=bt.Order.Limit, price=price *
                      (1 + self.params.take_profit))

        # Short signal: price breakdown with volume spike
        elif (self.data.close[0] < self.lowest[-1] and
              self.data.close[0] < self.ma[0]):
            self.order = self.sell()
            # Set stop loss and take profit orders
            self.buy(exectype=bt.Order.Stop, price=price *
                     (1 + self.params.stop_loss))
            self.buy(exectype=bt.Order.Limit, price=price *
                     (1 - self.params.take_profit))

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


class PriceMomentumStrategy(bt.Strategy):
    params = (
        ('momentum_period', 3),
        ('stop_loss', 0.002),
        ('take_profit', 0.003),
    )

    def __init__(self):
        # Momentum indicators
        self.momentum = bt.indicators.Momentum(
            self.data.close, period=self.params.momentum_period)
        # Track open orders
        self.order = None
        # ATR for volatility measurement
        self.atr = bt.indicators.ATR(self.data, period=14)

    def next(self):
        # Don't enter if we have an open order
        if self.order:
            return

        price = self.data.close[0]

        # Long signal: strong positive momentum
        if self.momentum[0] > self.atr[0] * 2:
            self.order = self.buy()
            # Set stop loss and take profit orders
            self.sell(exectype=bt.Order.Stop, price=price *
                      (1 - self.params.stop_loss))
            self.sell(exectype=bt.Order.Limit, price=price *
                      (1 + self.params.take_profit))

        # Short signal: strong negative momentum
        elif self.momentum[0] < -self.atr[0] * 2:
            self.order = self.sell()
            # Set stop loss and take profit orders
            self.buy(exectype=bt.Order.Stop, price=price *
                     (1 + self.params.stop_loss))
            self.buy(exectype=bt.Order.Limit, price=price *
                     (1 - self.params.take_profit))

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


def load_data():
    df = pd.read_parquet('/app/data/ohlcv_historical.parquet')
    print(f"Filas iniciales: {len(df)}")
    df['datetime'] = pd.to_datetime(
        df['timestamp'], unit='ms').dt.tz_localize('UTC')
    print(f"Valores nulos en datetime: {df['datetime'].isna().sum()}")
    print(f"Rango inicial: {df['datetime'].min()} a {df['datetime'].max()}")

    # Limpieza y validación de datos
    print("\nEstadísticas de valores nulos ANTES de limpieza:")
    print(df[['open', 'high', 'low', 'close', 'volume']].isna().sum())

    # Rellenar valores faltantes solo si hay suficientes datos válidos
    valid_cols = ['open', 'high', 'low', 'close', 'volume']
    if df[valid_cols].count().min() > len(df) * 0.9:  # Al menos 90% de datos válidos
        df[valid_cols] = df[valid_cols].ffill().bfill()
    else:
        print("Advertencia: Demasiados valores nulos, aplicando limpieza conservadora")
        df = df.dropna(subset=valid_cols)

    print("\nEstadísticas de valores nulos DESPUÉS de limpieza:")
    print(df[valid_cols].isna().sum())

    if df.empty:
        print("\n=== Datos crudos ===")
        print(df.head())
        print("\n=== Tipos de datos ===")
        print(df.dtypes)
        print("\n=== Estadísticas descriptivas ===")
        print(df.describe())
        raise ValueError(
            f"Error inesperado: DataFrame vacío tras limpieza. Filas iniciales: {len(df)}. Valores nulos:\n{df.isna().sum()}\nColumnas: {df.columns.tolist()}")

    df = df.sort_values('datetime')
    df.set_index('datetime', inplace=True)
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume'
    )
    print(f"Datos cargados en PandasData, filas estimadas: {len(df)}")
    return data
