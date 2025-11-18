import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange


def create_target_column(df,
                         lookahead_candles=4,
                         buy_threshold=0.003,
                         sell_threshold=-0.003):
    """
    Create classification target: BUY (1), HOLD (0), SELL (-1)

    lookahead_candles: How many candles ahead to predict
    buy_threshold: Min % gain to classify as BUY
    sell_threshold: Max % loss to classify as SELL
    """

    # Calculate future price at lookahead_candles ahead
    future_close = df['close'].shift(-lookahead_candles)

    # Calculate percentage return
    future_return = (future_close - df['close']) / df['close']

    # Classify based on thresholds
    target = pd.cut(future_return,
                    bins=[float('-inf'), sell_threshold, buy_threshold, float('inf')],
                    labels=[-1, 0, 1])  # -1=SELL, 0=HOLD, 1=BUY

    df['target'] = target.fillna(0).astype(int)
    return df


def get_technical_indicators(df):
    """
    1. Base Indicators (raw calculations)
    2. Momentum Indicators (RSI engineering)
    3. Trend Indicators (MACD engineering)
    4. Moving Average Engineering
    5. Volatility Indicators (Bollinger Bands & ATR)
    6. Volume Analysis
    7. Confluence & Signal Strength
    8. Price Action & Pattern Recognition
    9. Temporal & Momentum Features
    """

    # ==================== 1. BASE INDICATORS ====================
    # RSI - Relative Strength Index (0-100 scale)
    df['rsi_6'] = RSIIndicator(df['close'], window=6).rsi()
    df['rsi_12'] = RSIIndicator(df['close'], window=12).rsi()

    # MACD - Moving Average Convergence Divergence
    macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    # Moving Averages - Trend direction
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Bollinger Bands - Volatility & mean reversion
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['bollinger_bandwidth'] = bollinger.bollinger_wband()

    # ATR - Average True Range (volatility)
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()

    # Remove NaN rows from indicator calculation
    df.dropna(inplace=True)

    # ==================== 2. MOMENTUM INDICATORS (RSI Engineering) ====================
    # RSI momentum: Is RSI accelerating up or down?
    df['rsi_6_momentum'] = df['rsi_6'].diff(1)
    df['rsi_12_momentum'] = df['rsi_12'].diff(1)

    # RSI extreme zones: Overbought (>70) and oversold (<30)
    df['rsi_oversold'] = (df['rsi_6'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_6'] > 70).astype(int)

    # RSI reversals: Strong momentum when turning in extreme zones
    df['rsi_turning_up'] = (df['rsi_6'] > df['rsi_6'].shift(1)).astype(int) & df['rsi_oversold']
    df['rsi_turning_down'] = (df['rsi_6'] < df['rsi_6'].shift(1)).astype(int) & df['rsi_overbought']

    # ==================== 3. TREND INDICATORS (MACD Engineering) ====================
    # MACD crossover: Most important MACD signal (zero-cross)
    df['macd_crossover_up'] = (df['macd'] > df['macd_signal']) & \
                              (df['macd'].shift(1) <= df['macd_signal'].shift(1))

    # MACD histogram momentum: Is MACD getting stronger or weaker?
    df['macd_hist_momentum'] = df['macd_hist'].diff(1)

    # MACD strength: Normalized histogram (handles different price scales)
    df['macd_strength'] = df['macd_hist'] / (df['macd_hist'].rolling(20).std() + 1e-6)

    # ==================== 4. MOVING AVERAGE ENGINEERING ====================
    # MA alignment: All moving averages bullish (bullish signal)
    df['ma_aligned_bullish'] = (df['close'] > df['ema_21']) & (df['ema_21'] > df['sma_50'])

    # Price distance from EMA: How extended is price from short-term trend?
    df['price_above_ema_pct'] = ((df['close'] - df['ema_21']) / df['ema_21']) * 100

    # EMA slope: Is the trend accelerating?
    df['ema_21_slope'] = (df['ema_21'] - df['ema_21'].shift(5)) / 5

    # Distance between EMAs: Trend strength indicator
    df['ma_distance'] = ((df['ema_21'] - df['sma_50']) / df['sma_50']) * 100

    # ==================== 5. VOLATILITY INDICATORS (Bollinger Bands & ATR) ====================
    # Bollinger position: Where is price in the band? (0=lower band, 1=upper band)
    df['bb_position'] = (df['close'] - df['bollinger_lband']) / \
                        (df['bollinger_hband'] - df['bollinger_lband'])

    # Bollinger squeeze: Low volatility = potential breakout setup
    df['bb_squeeze'] = df['bollinger_bandwidth'] < df['bollinger_bandwidth'].rolling(20).quantile(0.25)

    # Bollinger band extremes: Mean reversion signals
    df['price_at_upper_band'] = (df['close'] >= df['bollinger_hband']).astype(int)
    df['price_at_lower_band'] = (df['close'] <= df['bollinger_lband']).astype(int)

    # ATR as percentage: Volatility relative to price
    df['atr_percent'] = (df['atr'] / df['close']) * 100

    # ATR momentum: Is volatility increasing or decreasing?
    df['atr_momentum'] = df['atr'].diff(5)

    # ==================== 6. VOLUME ANALYSIS ====================
    # Volume relative to moving average: Is this volume unusual?
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-6)

    # Volume spike: Sudden increase in volume (strong confirmation)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)

    # Volume confirmation: Is volume supporting price movement?
    df['volume_supports_up'] = (df['close'] > df['close'].shift(1)) & (df['volume_ratio'] > 1.0)
    df['volume_supports_down'] = (df['close'] < df['close'].shift(1)) & (df['volume_ratio'] > 1.0)

    # ==================== 7. CONFLUENCE & SIGNAL STRENGTH ====================
    # Bullish confluence: Count how many indicators agree (0-4 points)
    # Higher = more confirmations = higher confidence buy signal
    df['bullish_confluence'] = (
            (df['close'] > df['ema_21']).astype(int) +  # Price above short-term trend
            (df['ema_21'] > df['sma_50']).astype(int) +  # Short trend above long trend
            (df['macd'] > df['macd_signal']).astype(int) +  # MACD bullish
            (df['rsi_6'] > 50).astype(int)  # RSI in bullish zone
    )

    # Bearish confluence: Count how many indicators agree (0-4 points)
    # Higher = more confirmations = higher confidence sell signal
    df['bearish_confluence'] = (
            (df['close'] < df['ema_21']).astype(int) +  # Price below short-term trend
            (df['ema_21'] < df['sma_50']).astype(int) +  # Short trend below long trend
            (df['macd'] < df['macd_signal']).astype(int) +  # MACD bearish
            (df['rsi_6'] < 50).astype(int)  # RSI in bearish zone
    )

    # Strong bullish signal: Confluence + volume (0-5 confidence score)
    # Use this to filter: only trade BUY when score >= 4
    df['strong_bullish_signal'] = (
            (df['macd'] > df['macd_signal']).astype(int) +
            (df['rsi_6'] > 50).astype(int) +
            (df['close'] > df['ema_21']).astype(int) +
            (df['ema_21'] > df['sma_50']).astype(int) +
            df['volume_supports_up'].astype(int)
    )

    # Strong bearish signal: Confluence + volume (0-5 confidence score)
    # Use this to filter: only trade SELL when score >= 4
    df['strong_bearish_signal'] = (
            (df['macd'] < df['macd_signal']).astype(int) +
            (df['rsi_6'] < 50).astype(int) +
            (df['close'] < df['ema_21']).astype(int) +
            (df['ema_21'] < df['sma_50']).astype(int) +
            df['volume_supports_down'].astype(int)
    )

    # ==================== 8. PRICE ACTION & PATTERN RECOGNITION ====================
    # Higher lows (3 candles): Bullish pattern - price making higher lows
    df['higher_lows_3'] = (df['low'] > df['low'].shift(1)) & \
                          (df['low'].shift(1) > df['low'].shift(2)) & \
                          (df['low'].shift(2) > df['low'].shift(3))

    # Lower highs (3 candles): Bearish pattern - price making lower highs
    df['lower_highs_3'] = (df['high'] < df['high'].shift(1)) & \
                          (df['high'].shift(1) < df['high'].shift(2)) & \
                          (df['high'].shift(2) < df['high'].shift(3))

    # Trend strength bullish: How strong is the bullish move? (0-1 scale)
    # Closer to 1 = price at upper band = strong uptrend
    df['trend_strength_bullish'] = (df['close'] - df['bollinger_lband']) / \
                                   (df['bollinger_hband'] - df['bollinger_lband'])

    # Trend strength bearish: How strong is the bearish move? (0-1 scale)
    # Closer to 1 = price at lower band = strong downtrend
    df['trend_strength_bearish'] = (df['bollinger_hband'] - df['close']) / \
                                   (df['bollinger_hband'] - df['bollinger_lband'])

    # Distance from support/resistance (SMA_50): How far from equilibrium?
    df['distance_from_sma_pct'] = ((df['close'] - df['sma_50']) / df['sma_50']) * 100

    # Candle quality: Is the candle strong (large body) or weak (large wicks)?
    # Closer to 1 = strong decisive candle, Closer to 0 = weak/indecisive candle
    df['candle_body_pct'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)

    # ==================== 9. TEMPORAL & MOMENTUM FEATURES ====================
    # Short-term momentum: Rate of change over 5 candles
    df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)

    # Medium-term momentum: Rate of change over 10 candles
    df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)

    # Ensure no NaN values remain after all calculations
    df.dropna(inplace=True)

    return df
