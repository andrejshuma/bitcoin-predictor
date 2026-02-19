import ccxt
import time
import pandas as pd
from datetime import datetime, timedelta, timezone


def fetch_binance_btc(timeframe: str, periodInDays: float):
    binance = ccxt.binance()

    symbol = 'BTC/USDT'
    limit = 1000

    since_dt = datetime.now(timezone.utc) - timedelta(days=periodInDays)
    since = int(since_dt.timestamp() * 1000)                        # second in milliseconds

    all_candles = []

    while True:
        candles = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break

        all_candles += candles

        since = candles[-1][0] + 1      # last candle timestamp + 1 ms
        time.sleep(0.5)                 # avoid rate limits with api

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    df.set_index('timestamp', inplace=True)

    return df
