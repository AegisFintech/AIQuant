"""
aiquant/features/technical.py
==============================
Comprehensive technical indicator feature engineering.
Covers trend, momentum, volatility, volume, and cycle indicators.
Optimised for 1m BTCUSD data with RAM-heavy vectorised operations.
"""

import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Moving averages, MACD, ADX, Ichimoku, Supertrend."""
    df = df.copy()

    # Simple and Exponential Moving Averages
    for p in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{p}'] = ta.trend.sma_indicator(df['close'], window=p)
        df[f'ema_{p}'] = ta.trend.ema_indicator(df['close'], window=p)

    # Price relative to moving averages (normalised distance)
    for p in [21, 50, 200]:
        df[f'close_vs_sma{p}'] = (df['close'] - df[f'sma_{p}']) / df[f'sma_{p}']

    # MACD (standard + fast variants for 1m)
    for fast, slow, sig in [(12, 26, 9), (5, 13, 4), (3, 10, 16)]:
        macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=sig)
        tag = f"{fast}_{slow}"
        df[f'macd_{tag}'] = macd.macd()
        df[f'macd_signal_{tag}'] = macd.macd_signal()
        df[f'macd_diff_{tag}'] = macd.macd_diff()

    # ADX — trend strength
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
    df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])

    # Ichimoku Cloud
    ichi = ta.trend.IchimokuIndicator(df['high'], df['low'])
    df['ichi_a'] = ichi.ichimoku_a()
    df['ichi_b'] = ichi.ichimoku_b()
    df['ichi_base'] = ichi.ichimoku_base_line()
    df['ichi_conv'] = ichi.ichimoku_conversion_line()

    # Parabolic SAR
    df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()

    # Aroon
    aroon = ta.trend.AroonIndicator(df['high'], df['low'])
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()
    df['aroon_ind'] = aroon.aroon_indicator()

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, Stochastic, Williams %R, CCI, ROC, TRIX, Ultimate Oscillator."""
    df = df.copy()

    # RSI — multiple periods
    for p in [7, 14, 21]:
        df[f'rsi_{p}'] = ta.momentum.rsi(df['close'], window=p)

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Williams %R
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])

    # CCI
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])

    # Rate of Change
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = ta.momentum.roc(df['close'], window=p)

    # TRIX
    df['trix'] = ta.trend.trix(df['close'])

    # Ultimate Oscillator
    df['uo'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])

    # TSI — True Strength Index
    df['tsi'] = ta.momentum.tsi(df['close'])

    # Percentage Price Oscillator
    df['ppo'] = ta.momentum.ppo(df['close'])

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Bands, ATR, Keltner Channels, Donchian, realised vol."""
    df = df.copy()

    # Returns and log returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Realised volatility (annualised) over multiple windows
    for p in [20, 60, 120, 240]:
        df[f'realvol_{p}'] = df['returns'].rolling(p).std() * np.sqrt(525_600)  # 1m bars in a year

    # Bollinger Bands — standard and wide
    for std in [1.5, 2.0, 2.5]:
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=std)
        tag = str(std).replace('.', '')
        df[f'bb_high_{tag}'] = bb.bollinger_hband()
        df[f'bb_low_{tag}'] = bb.bollinger_lband()
        df[f'bb_width_{tag}'] = bb.bollinger_wband()
        df[f'bb_pct_{tag}'] = bb.bollinger_pband()

    # ATR — multiple periods
    for p in [7, 14, 21]:
        df[f'atr_{p}'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=p)

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
    df['kc_high'] = kc.keltner_channel_hband()
    df['kc_low'] = kc.keltner_channel_lband()
    df['kc_pct'] = kc.keltner_channel_pband()

    # Donchian Channel
    dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
    df['dc_high'] = dc.donchian_channel_hband()
    df['dc_low'] = dc.donchian_channel_lband()
    df['dc_width'] = dc.donchian_channel_wband()

    # Volatility regime: current vol vs rolling average
    df['vol_regime'] = df['realvol_20'] / df['realvol_240'].replace(0, np.nan)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """OBV, VWAP, CMF, MFI, Force Index, volume z-score."""
    df = df.copy()

    # On-Balance Volume
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    # VWAP
    df['vwap'] = ta.volume.volume_weighted_average_price(
        df['high'], df['low'], df['close'], df['volume']
    )
    df['close_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']

    # Chaikin Money Flow
    df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])

    # Money Flow Index
    df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])

    # Force Index
    df['fi'] = ta.volume.force_index(df['close'], df['volume'])

    # Volume moving averages and z-score
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_sma60'] = df['volume'].rolling(60).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
    vol_std = df['volume'].rolling(60).std()
    df['vol_zscore'] = (df['volume'] - df['vol_sma60']) / vol_std.replace(0, np.nan)

    # Ease of Movement
    df['eom'] = ta.volume.ease_of_movement(df['high'], df['low'], df['volume'])

    # Volume Price Trend
    df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])

    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candlestick-derived features: body, wicks, gap, range."""
    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['body_pct'] = df['body'] / df['open'].replace(0, np.nan)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['candle_range'] = df['high'] - df['low']
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1).replace(0, np.nan)
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_bearish'] = (df['close'] < df['open']).astype(int)
    # Consecutive candle direction
    df['consec_bull'] = df['is_bullish'].groupby(
        (df['is_bullish'] != df['is_bullish'].shift()).cumsum()
    ).cumcount() + 1
    df['consec_bear'] = df['is_bearish'].groupby(
        (df['is_bearish'] != df['is_bearish'].shift()).cumsum()
    ).cumcount() + 1
    return df


def generate_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: apply all technical feature groups."""
    logger.info("Generating all technical features...")
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_candle_features(df)
    logger.info(f"Technical features complete. Shape: {df.shape}")
    return df
