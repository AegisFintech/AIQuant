"""
aiquant/features/statarb.py
============================
Statistical arbitrage and regime detection features.

Memory-efficient: all functions mutate the DataFrame in-place (no df.copy()).
The caller owns the DataFrame and is responsible for copying if needed.

Implements:
  - Z-score of price relative to rolling mean/std
  - Half-life of mean reversion (Ornstein-Uhlenbeck)
  - Hurst Exponent (trend vs mean-reversion regime)
  - ADF stationarity test (rolling, Numba-accelerated)
  - Kalman Filter dynamic spread
  - Structural break detection (CUSUM)
"""

import pandas as pd
import numpy as np
import logging

from ..utils.fast_math import (
    rolling_hurst_nb,
    rolling_half_life_nb,
    rolling_adf_fast_nb,
    kalman_filter_nb,
    rolling_roll_spread_nb,
)

logger = logging.getLogger(__name__)


def zscore_features(df: pd.DataFrame, windows: list = [20, 60, 120, 240]) -> pd.DataFrame:
    """
    Rolling z-score of close price and returns.
    Core signal for mean-reversion strategies.
    """
    # In-place — no df.copy()
    close = df['close']
    for w in windows:
        mu    = close.rolling(w).mean()
        sigma = close.rolling(w).std().replace(0, np.nan)
        df[f'zscore_price_{w}'] = (close - mu) / sigma

        if 'returns' in df.columns:
            r     = df['returns']
            mu_r  = r.rolling(w).mean()
            sig_r = r.rolling(w).std().replace(0, np.nan)
            df[f'zscore_returns_{w}'] = (r - mu_r) / sig_r

    return df


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Compute the Hurst Exponent of a time series.
      H < 0.5 → mean-reverting
      H = 0.5 → random walk
      H > 0.5 → trending
    """
    lags = range(2, max_lag)
    tau  = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def rolling_hurst(df: pd.DataFrame, window: int = 240, max_lag: int = 20) -> pd.DataFrame:
    """
    Rolling Hurst Exponent — Numba JIT compiled.
    """
    close     = df['close'].to_numpy(dtype=np.float64)
    hurst_arr = rolling_hurst_nb(close, window=window, max_lag=max_lag)
    # In-place — no df.copy()
    df['hurst'] = hurst_arr
    bins   = np.array([0.0, 0.45, 0.55, 1.0])
    labels = np.array(['mean_reverting', 'random_walk', 'trending'])
    idx    = np.clip(np.digitize(hurst_arr, bins) - 1, 0, len(labels) - 1)
    df['regime'] = labels[idx]
    return df


def rolling_half_life(df: pd.DataFrame, window: int = 240) -> pd.DataFrame:
    """
    Rolling OU half-life — Numba JIT compiled.
    """
    close  = df['close'].to_numpy(dtype=np.float64)
    hl_arr = rolling_half_life_nb(close, window=window)
    # In-place — no df.copy()
    df['half_life'] = hl_arr
    return df


def adf_test_rolling(df: pd.DataFrame, window: int = 240) -> pd.DataFrame:
    """
    Rolling ADF p-value — Numba parallel approximation (~500x faster than
    statsmodels adfuller loop).
    """
    close   = df['close'].to_numpy(dtype=np.float64)
    adf_arr = rolling_adf_fast_nb(close, window=window)
    # In-place — no df.copy()
    df['adf_pvalue']    = adf_arr
    df['is_stationary'] = (adf_arr < 0.05).astype(np.int8)
    return df


def kalman_filter_spread(df: pd.DataFrame, delta: float = 1e-4) -> pd.DataFrame:
    """
    Kalman Filter for dynamic mean estimation.
    """
    close = df['close'].to_numpy(dtype=np.float64)
    kalman_mean, kalman_residual, kalman_zscore = kalman_filter_nb(close, delta=delta)
    # In-place — no df.copy()
    df['kalman_mean']     = kalman_mean
    df['kalman_residual'] = kalman_residual
    df['kalman_zscore']   = kalman_zscore
    return df


def cusum_structural_break(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    CUSUM statistic for detecting structural breaks / regime changes.
    """
    # In-place — no df.copy()
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()

    mu           = df['returns'].expanding().mean()
    sigma        = df['returns'].expanding().std().replace(0, np.nan)
    standardised = (df['returns'] - mu) / sigma
    df['cusum_pos']    = standardised.clip(lower=0).cumsum()
    df['cusum_neg']    = (-standardised).clip(lower=0).cumsum()
    df['regime_break'] = (
        (df['cusum_pos'] > threshold) | (df['cusum_neg'] > threshold)
    ).astype(np.int8)

    return df


def generate_all_statarb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: apply all statistical arbitrage feature groups in-place."""
    logger.info("Generating statistical arbitrage features...")
    df = zscore_features(df)
    df = rolling_hurst(df)
    df = rolling_half_life(df)
    df = adf_test_rolling(df)
    df = kalman_filter_spread(df)
    df = cusum_structural_break(df)
    logger.info(f"StatArb features complete. Shape: {df.shape}")
    return df
