import backtrader as bt
import pandas as pd
from sqlalchemy import create_engine


class SignalStrategy(bt.Strategy):
    def __init__(self):
        self.signal = self.datas[0].signal

    def next(self):
        if self.signal[0] == 1:  # Buy
            self.buy()
        elif self.signal[0] == -1:  # Sell
            self.sell()


# Cargar datos
engine = create_engine("postgresql://admin:password@timescaledb:5432/postgres")
df = pd.read_sql(
    "SELECT timestamp, bid_price as close, side FROM trades WHERE status='executed'", engine)
df['signal'] = df['side'].map({'buy': 1, 'sell': -1, '': 0})

# Configurar Backtrader
cerebro = bt.Cerebro()
data = bt.feeds.PandasData(
    dataname=df, datetime='timestamp', close='close', signal='signal')
cerebro.adddata(data)
cerebro.addstrategy(SignalStrategy)
cerebro.run()
cerebro.plot()
