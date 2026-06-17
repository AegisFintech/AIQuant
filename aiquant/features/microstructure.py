"""
aiquant/features/microstructure.py
====================================
Market microstructure features for HFT and statistical arbitrage.

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
    df = df.copy()
    # Tick rule: classify each bar as buy or sell based on price direction
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
    df = df.copy()
    df['buy_vol_proxy'] = df['volume'] * ((df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan))
    df['sell_vol_proxy'] = df['volume'] * ((df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan))

    for w in windows:
        total = df['volume'].rolling(w).sum().replace(0, np.nan)
        df[f'buy_ratio_{w}'] = df['buy_vol_proxy'].rolling(w).sum() / total
        df[f'sell_ratio_{w}'] = df['sell_vol_proxy'].rolling(w).sum() / total
        df[f'trade_imbalance_{w}'] = df[f'buy_ratio_{w}'] - df[f'sell_ratio_{w}']

    return df


def vpin(df: pd.DataFrame, bucket_size: int = 50, n_buckets: int = 50) -> pd.DataFrame:
    """
    Simplified VPIN (Volume-Synchronized Probability of Informed Trading).
    High VPIN signals toxic order flow and potential adverse selection.
    """
    df = df.copy()
    # Estimate buy volume fraction using bulk volume classification
    df['buy_frac'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['buy_frac'] = df['buy_frac'].fillna(0.5).clip(0, 1)
    df['buy_vol'] = df['buy_frac'] * df['volume']
    df['sell_vol'] = (1 - df['buy_frac']) * df['volume']
    df['abs_imbalance'] = (df['buy_vol'] - df['sell_vol']).abs()

    window = bucket_size * n_buckets
    total_vol = df['volume'].rolling(window).sum().replace(0, np.nan)
    df['vpin'] = df['abs_imbalance'].rolling(window).sum() / total_vol

    return df


def amihud_illiquidity(df: pd.DataFrame, windows: list = [20, 60, 120]) -> pd.DataFrame:
    """
    Amihud (2002) illiquidity ratio: |return| / dollar_volume.
    Higher values indicate lower liquidity and higher price impact.
    """
    df = df.copy()
    df['dollar_volume'] = df['close'] * df['volume']
    df['abs_return'] = df['returns'].abs() if 'returns' in df.columns else df['close'].pct_change().abs()

    for w in windows:
        df[f'amihud_{w}'] = (
            df['abs_return'] / df['dollar_volume'].replace(0, np.nan)
        ).rolling(w).mean() * 1e6  # Scale for readability

    return df


def kyles_lambda(df: pd.DataFrame, windows: list = [20, 60]) -> pd.DataFrame:
    """
    Kyle's Lambda: price impact per unit of signed order flow.
    Estimated via OLS regression of price change on signed volume.
    """
    df = df.copy()
    if 'signed_volume' not in df.columns:
        df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
        df['signed_volume'] = df['tick_direction'] * df['volume']

    price_change = df['close'].diff()

    for w in windows:
        # Rolling covariance / variance estimate
        cov = price_change.rolling(w).cov(df['signed_volume'])
        var = df['signed_volume'].rolling(w).var().replace(0, np.nan)
        df[f'kyle_lambda_{w}'] = cov / var

    return df


def roll_spread(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Roll (1984) spread estimator — Numba JIT compiled.
    Replaces pandas .rolling().apply(autocorr * var) — 100x+ speedup.
    """
    from ..utils.fast_math import rolling_roll_spread_nb
    close = df['close'].to_numpy(dtype=np.float64)
    roll_arr = rolling_roll_spread_nb(close, window=window)
    df = df.copy()
    df['roll_spread'] = roll_arr
    return df


def bid_ask_spread_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corwin-Schultz (2012) high-low spread estimator.
    Uses daily high-low range as a proxy for bid-ask spread.
    """
    df = df.copy()
    beta = (np.log(df['high'] / df['low']) ** 2).rolling(2).sum()
    gamma = (np.log(df[['high', 'high']].max(axis=1) / df[['low', 'low']].min(axis=1))) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    df['cs_spread'] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    df['cs_spread'] = df['cs_spread'].clip(lower=0)
    return df


def realized_variance(df: pd.DataFrame, windows: list = [20, 60, 120]) -> pd.DataFrame:
    """
    Realized variance and bipower variation for jump detection.
    """
    df = df.copy()
    if 'log_returns' not in df.columns:
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    for w in windows:
        df[f'rv_{w}'] = (df['log_returns'] ** 2).rolling(w).sum()
        # Bipower variation (robust to jumps)
        abs_ret = df['log_returns'].abs()
        df[f'bpv_{w}'] = (abs_ret * abs_ret.shift(1)).rolling(w).sum() * (np.pi / 2)
        # Jump component
        df[f'jump_{w}'] = np.maximum(0, df[f'rv_{w}'] - df[f'bpv_{w}'])

    return df


def return_autocorrelation(df: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Rolling autocorrelation of returns at various lags.
    Negative autocorrelation → mean reversion; Positive → momentum.
    """
    df = df.copy()
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()

    from ..utils.fast_math import rolling_autocorr_nb
    returns_arr = df['returns'].to_numpy(dtype=np.float64)
    for lag in lags:
        # Numba JIT compiled autocorrelation — replaces pandas .rolling().apply()
        df[f'autocorr_{lag}'] = rolling_autocorr_nb(returns_arr, window=60, lag=lag)

    return df


def intraday_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-of-day and day-of-week features for intraday seasonality.
    Bitcoin exhibits strong intraday patterns (e.g., US open/close effects).
    """
    df = df.copy()
    idx = df.index
    df['hour'] = idx.hour
    df['minute'] = idx.minute
    df['day_of_week'] = idx.dayofweek
    df['is_weekend'] = (idx.dayofweek >= 5).astype(int)

    # Cyclical encoding (preserves periodicity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    return df


def generate_all_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: apply all microstructure feature groups."""
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
