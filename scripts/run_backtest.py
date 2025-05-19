import backtrader as bt
from scalping_strategy import RelaxedScalpingStrategy, StrictScalpingStrategy, VolumeBreakoutStrategy, PriceMomentumStrategy, load_data
from datetime import datetime
import logging
import time
import signal
import psutil
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def timeout_handler(signum, frame):
    raise TimeoutError("Backtest excedió el tiempo máximo de 800 segundos")


def log_resources():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    cpu = psutil.cpu_percent(interval=0.1)
    logger.info(f"Recursos: Memoria={mem:.2f}MB, CPU={cpu:.2f}%")


class ProgressObserver(bt.Observer):
    lines = ('datetime',)
    plotlines = dict(datetime=dict(_plotskip=True))

    def __init__(self):
        self.last_date = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        if self.last_date != current_date:
            logger.info(f"Procesando fecha: {current_date}")
            self.last_date = current_date
            log_resources()


cerebro = bt.Cerebro(optreturn=False)
cerebro.addobserver(ProgressObserver)
try:
    data = load_data()
    if data.buflen() == 0:
        logger.error(
            "Datos cargados pero buffer vacío - revisar formato de datos")
        with open('/app/data/backtest.log', 'a') as f:
            f.write("\nError: Datos cargados pero buffer vacío\n")
        exit(1)
except Exception as e:
    logger.error(f"Error al cargar datos: {str(e)}")
    with open('/app/data/backtest.log', 'a') as f:
        f.write(f"\nError al cargar datos: {str(e)}\n")
    exit(1)
    with open('/app/data/backtest.log', 'w') as f:
        f.write("Error: No hay datos válidos en el rango especificado\n")
    exit(1)
logger.info(
    f"Rango de datos: {data.datetime.date(0)} a {data.datetime.date(-1)}, Filas: {data.buflen()}")
cerebro.adddata(data)
cerebro.broker.setcash(13188.0)
cerebro.broker.setcommission(commission=0.0004)
cerebro.addsizer(bt.sizers.PercentSizer, percents=2)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
# Fechas ya están filtradas en fetch_historical_data.py
cerebro.datas[0].fromdate = datetime(2023, 1, 1, tzinfo=pytz.UTC)
cerebro.datas[0].todate = datetime(2025, 5, 16, tzinfo=pytz.UTC)

cerebro.optstrategy(
    RelaxedScalpingStrategy,
    ma_period=[3, 5],
    rsi_low=[3, 5],
    volume_threshold=[0.05],
    take_profit=[0.002, 0.003]
)
cerebro.optstrategy(
    StrictScalpingStrategy,
    ma_period=[3],
    rsi_low=[10],
    volume_threshold=[2.0],
    take_profit=[0.003]
)
cerebro.optstrategy(
    VolumeBreakoutStrategy,
    ma_period=[5, 10],
    volume_threshold=[20.0, 50.0],
    take_profit=[0.002, 0.003]
)
cerebro.optstrategy(
    PriceMomentumStrategy,
    momentum_period=[2, 3],
    take_profit=[0.002, 0.003]
)

logger.info("Iniciando backtest")
start_time = time.time()

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(800)

try:
    results = cerebro.run()
except TimeoutError:
    logger.error("Backtest interrumpido por timeout")
    results = []
except Exception as e:
    logger.error(f"Error en backtest: {e}")
    results = []
finally:
    signal.alarm(0)

elapsed = time.time() - start_time
logger.info(f"Backtest completado en {elapsed:.2f} segundos")


def save_metrics():
    if not results:
        logger.error("No se encontraron resultados válidos")
        with open('/app/data/backtest.log', 'w') as f:
            f.write("No se encontraron resultados válidos\n")
        return
    best_sharpe = -float('inf')
    best_result = None
    for strat_run in results:
        for strat in strat_run:
            try:
                sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = strat
            except Exception as e:
                logger.error(f"Error en análisis de estrategia: {e}")
    if best_result is None:
        logger.error("No se encontraron resultados válidos")
        with open('/app/data/backtest.log', 'w') as f:
            f.write("No se encontraron resultados válidos\n")
        return
    try:
        sharpe = best_result.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
        drawdown = best_result.analyzers.drawdown.get_analysis()[
            'max']['drawdown']
        trades = best_result.analyzers.trades.get_analysis()
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        pnl_net = trades.get('pnl', {}).get('net', {}).get('total', 0)
        params = best_result.params
        with open('/app/data/backtest.log', 'w') as f:
            f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
            f.write(f"Max Drawdown: {drawdown:.2f}%\n")
            f.write(f"Win Rate: {win_rate:.2f}%\n")
            f.write(f"Total Trades: {total_trades}\n")
            f.write(f"Net PNL: {pnl_net:.2f}\n")
            f.write(f"Best Parameters: {params.__dict__}\n")
        logger.info(
            f"Métricas: Sharpe={sharpe:.2f}, Drawdown={drawdown:.2f}%, WinRate={win_rate:.2f}%, Trades={total_trades}, PNL={pnl_net:.2f}, Params={params.__dict__}")
    except Exception as e:
        logger.error(f"Error al guardar métricas: {e}")
        with open('/app/data/backtest.log', 'w') as f:
            f.write("Error al calcular métricas\n")


try:
    save_metrics()
except KeyboardInterrupt:
    logger.info("Backtest interrumpido, guardando métricas parciales")
    save_metrics()
