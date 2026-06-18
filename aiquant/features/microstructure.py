"""
aiquant/features/microstructure.py
====================================
Market microstructure features for HFT and statistical arbitrage.

Memory-efficient: all functions mutate the DataFrame in-place (no df.copy()).
The caller owns the DataFrame and is responsible for copying if needed.

Implements:
  - Order Flow Imbalance (OFI)
  - Trade Imbalance (buy vs sell volume proxy)
  - VPIN (Volume-Synchronized Probability of Informed Trading)
  - Amihud Illiquidity Ratio
  - Kyle's Lambda (price impact)
  - Roll's Spread Estimator
  - Bid-Ask Spread Proxy (from OHLCV)
  - Realized Variance and Bipower Variation
  - Autocorrelation of Returns (short-term mean reversion signal)
  - Intraday Seasonality (time-of-day effects)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def order_flow_imbalance(df: pd.DataFrame, windows: list = [5, 15, 30, 60]) -> pd.DataFrame:
    """
    Proxy for Order Flow Imbalance using price and volume.
    When price rises with volume, buying pressure dominates (OFI > 0).
    When price falls with volume, selling pressure dominates (OFI < 0).
    """
    # In-place — no df.copy()
    df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
    df['signed_volume'] = df['tick_direction'] * df['volume']

    for w in windows:
        df[f'ofi_{w}'] = df['signed_volume'].rolling(w).sum()
        df[f'ofi_norm_{w}'] = df[f'ofi_{w}'] / df['volume'].rolling(w).sum().replace(0, np.nan)

    return df


def trade_imbalance_proxy(df: pd.DataFrame, windows: list = [10, 30, 60]) -> pd.DataFrame:
    """
    Estimate buy/sell volume split using the Lee-Ready algorithm proxy.
    Buys: volume when close > open; Sells: volume when close < open.
    """
    # In-place — no df.copy()
    hl_range = (df['high'] - df['low']).replace(0, np.nan)
    df['buy_vol_proxy']  = df['volume'] * ((df['close'] - df['low']) / hl_range)
    df['sell_vol_proxy'] = df['volume'] * ((df['high'] - df['close']) / hl_range)

    for w in windows:
        total = df['volume'].rolling(w).sum().replace(0, np.nan)
        df[f'buy_ratio_{w}']       = df['buy_vol_proxy'].rolling(w).sum() / total
        df[f'sell_ratio_{w}']      = df['sell_vol_proxy'].rolling(w).sum() / total
        df[f'trade_imbalance_{w}'] = df[f'buy_ratio_{w}'] - df[f'sell_ratio_{w}']

    return df


def vpin(df: pd.DataFrame, bucket_size: int = 50, n_buckets: int = 50) -> pd.DataFrame:
    """
    Simplified VPIN (Volume-Synchronized Probability of Informed Trading).
    High VPIN signals toxic order flow and potential adverse selection.
    """
    # In-place — no df.copy()
    hl_range = (df['high'] - df['low']).replace(0, np.nan)
    buy_frac = ((df['close'] - df['low']) / hl_range).fillna(0.5).clip(0, 1)
    df['buy_frac']      = buy_frac
    df['buy_vol']       = buy_frac * df['volume']
    df['sell_vol']      = (1 - buy_frac) * df['volume']
    df['abs_imbalance'] = (df['buy_vol'] - df['sell_vol']).abs()

    window    = bucket_size * n_buckets
    total_vol = df['volume'].rolling(window).sum().replace(0, np.nan)
    df['vpin'] = df['abs_imbalance'].rolling(window).sum() / total_vol

    return df


def amihud_illiquidity(df: pd.DataFrame, windows: list = [20, 60, 120]) -> pd.DataFrame:
    """
    Amihud (2002) illiquidity ratio: |return| / dollar_volume.
    Higher values indicate lower liquidity and higher price impact.
    """
    # In-place — no df.copy()
    df['dollar_volume'] = df['close'] * df['volume']
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()
    df['abs_return'] = df['returns'].abs()

    for w in windows:
        df[f'amihud_{w}'] = (
            df['abs_return'] / df['dollar_volume'].replace(0, np.nan)
        ).rolling(w).mean() * 1e6

    return df


def kyles_lambda(df: pd.DataFrame, windows: list = [20, 60]) -> pd.DataFrame:
    """
    Kyle's Lambda: price impact per unit of signed order flow.
    Estimated via OLS regression of price change on signed volume.
    """
    # In-place — no df.copy()
    if 'signed_volume' not in df.columns:
        df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
        df['signed_volume']  = df['tick_direction'] * df['volume']

    price_change = df['close'].diff()

    for w in windows:
        cov = price_change.rolling(w).cov(df['signed_volume'])
        var = df['signed_volume'].rolling(w).var().replace(0, np.nan)
        df[f'kyle_lambda_{w}'] = cov / var

    return df


def roll_spread(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Roll (1984) spread estimator — Numba JIT compiled.
    """
    from ..utils.fast_math import rolling_roll_spread_nb
    close    = df['close'].to_numpy(dtype=np.float64)
    roll_arr = rolling_roll_spread_nb(close, window=window)
    # In-place — no df.copy()
    df['roll_spread'] = roll_arr
    return df


def bid_ask_spread_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corwin-Schultz (2012) high-low spread estimator.
    """
    # In-place — no df.copy()
    log_hl  = np.log(df['high'] / df['low'].replace(0, np.nan))
    beta    = (log_hl ** 2).rolling(2).sum()
    h_max   = df['high'].rolling(2).max()
    l_min   = df['low'].rolling(2).min()
    gamma   = (np.log(h_max / l_min.replace(0, np.nan))) ** 2
    alpha   = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) \
              - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    spread  = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    df['cs_spread'] = spread.clip(lower=0)
    return df


def realized_variance(df: pd.DataFrame, windows: list = [20, 60, 120]) -> pd.DataFrame:
    """
    Realized variance and bipower variation for jump detection.
    """
    # In-place — no df.copy()
    if 'log_returns' not in df.columns:
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    for w in windows:
        df[f'rv_{w}']   = (df['log_returns'] ** 2).rolling(w).sum()
        abs_ret         = df['log_returns'].abs()
        df[f'bpv_{w}']  = (abs_ret * abs_ret.shift(1)).rolling(w).sum() * (np.pi / 2)
        df[f'jump_{w}'] = np.maximum(0, df[f'rv_{w}'] - df[f'bpv_{w}'])

    return df


def return_autocorrelation(df: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Rolling autocorrelation of returns at various lags.
    Negative autocorrelation → mean reversion; Positive → momentum.
    """
    # In-place — no df.copy()
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()

    from ..utils.fast_math import rolling_autocorr_nb
    returns_arr = df['returns'].to_numpy(dtype=np.float64)
    for lag in lags:
        df[f'autocorr_{lag}'] = rolling_autocorr_nb(returns_arr, window=60, lag=lag)

    return df


def intraday_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-of-day and day-of-week features for intraday seasonality.
    """
    # In-place — no df.copy()
    idx = df.index
    df['hour']        = idx.hour
    df['minute']      = idx.minute
    df['day_of_week'] = idx.dayofweek
    df['is_weekend']  = (idx.dayofweek >= 5).astype(np.int8)

    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']    = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']    = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    return df


def generate_all_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: apply all microstructure feature groups in-place."""
    logger.info("Generating microstructure features...")
    df = order_flow_imbalance(df)
    df = trade_imbalance_proxy(df)
    df = vpin(df)
    df = amihud_illiquidity(df)
    df = kyles_lambda(df)
    df = roll_spread(df)
    df = bid_ask_spread_proxy(df)
    df = realized_variance(df)
    df = return_autocorrelation(df)
    df = intraday_seasonality(df)
    logger.info(f"Microstructure features complete. Shape: {df.shape}")
    return df
