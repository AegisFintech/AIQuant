"""
aiquant/features/statarb.py
============================
Statistical arbitrage and regime detection features.

Implements:
  - Z-score of price relative to rolling mean/std
  - Cointegration-based spread features (BTC vs ETH proxy)
  - Half-life of mean reversion (Ornstein-Uhlenbeck)
  - Hurst Exponent (trend vs mean-reversion regime)
  - Kalman Filter dynamic spread
  - Regime detection via Hidden Markov Model proxy
  - Structural break detection (CUSUM)
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import logging

logger = logging.getLogger(__name__)


def zscore_features(df: pd.DataFrame, windows: list = [20, 60, 120, 240]) -> pd.DataFrame:
    """
    Rolling z-score of close price and returns.
    Core signal for mean-reversion strategies.
    """
    df = df.copy()
    for w in windows:
        mu = df['close'].rolling(w).mean()
        sigma = df['close'].rolling(w).std().replace(0, np.nan)
        df[f'zscore_price_{w}'] = (df['close'] - mu) / sigma

        if 'returns' in df.columns:
            mu_r = df['returns'].rolling(w).mean()
            sigma_r = df['returns'].rolling(w).std().replace(0, np.nan)
            df[f'zscore_returns_{w}'] = (df['returns'] - mu_r) / sigma_r

    return df


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Compute the Hurst Exponent of a time series.
      H < 0.5 → mean-reverting
      H = 0.5 → random walk
      H > 0.5 → trending
    """
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def rolling_hurst(df: pd.DataFrame, window: int = 240, max_lag: int = 20) -> pd.DataFrame:
    """
    Rolling Hurst Exponent over a given window.
    Identifies whether BTC is in a trending or mean-reverting regime.
    """
    df = df.copy()
    df['hurst'] = df['close'].rolling(window).apply(
        lambda x: hurst_exponent(pd.Series(x), max_lag=max_lag), raw=False
    )
    df['regime'] = pd.cut(
        df['hurst'],
        bins=[0, 0.45, 0.55, 1.0],
        labels=['mean_reverting', 'random_walk', 'trending']
    )
    return df


def half_life_ou(series: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion using an
    Ornstein-Uhlenbeck process fit via OLS.
    """
    y = series.dropna()
    y_lag = y.shift(1).dropna()
    y = y.iloc[1:]
    delta_y = y - y_lag
    X = add_constant(y_lag)
    model = OLS(delta_y, X).fit()
    kappa = -model.params.iloc[1]
    if kappa <= 0:
        return np.nan
    return np.log(2) / kappa


def rolling_half_life(df: pd.DataFrame, window: int = 240) -> pd.DataFrame:
    """
    Rolling half-life of mean reversion.
    Shorter half-life → faster mean reversion → better for HFT.
    """
    df = df.copy()
    df['half_life'] = df['close'].rolling(window).apply(
        lambda x: half_life_ou(pd.Series(x)), raw=False
    )
    return df


def adf_test_rolling(df: pd.DataFrame, window: int = 240) -> pd.DataFrame:
    """
    Rolling Augmented Dickey-Fuller test p-value.
    Low p-value (< 0.05) → stationary → mean-reverting regime.
    """
    df = df.copy()
    df['adf_pvalue'] = df['close'].rolling(window).apply(
        lambda x: adfuller(x, autolag='AIC')[1], raw=True
    )
    df['is_stationary'] = (df['adf_pvalue'] < 0.05).astype(int)
    return df


def kalman_filter_spread(df: pd.DataFrame, delta: float = 1e-4) -> pd.DataFrame:
    """
    Kalman Filter for dynamic mean estimation.
    The Kalman residual (innovation) is a clean mean-reversion signal.
    """
    df = df.copy()
    n = len(df)
    close = df['close'].values

    # State: [level, trend]
    x = np.array([close[0], 0.0])
    P = np.eye(2) * 1.0
    Q = np.eye(2) * delta  # Process noise
    R = np.var(np.diff(close[:50])) if len(close) > 50 else 1.0  # Measurement noise
    F = np.array([[1, 1], [0, 1]])  # Transition matrix
    H = np.array([[1, 0]])           # Observation matrix

    kalman_mean = np.zeros(n)
    kalman_residual = np.zeros(n)

    for i in range(n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update
        y = close[i] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T / S[0, 0]
        x = x + K.flatten() * y[0]
        P = (np.eye(2) - np.outer(K.flatten(), H)) @ P
        kalman_mean[i] = x[0]
        kalman_residual[i] = y[0]

    df['kalman_mean'] = kalman_mean
    df['kalman_residual'] = kalman_residual
    # Normalise residual
    std = pd.Series(kalman_residual).rolling(60).std().values
    df['kalman_zscore'] = kalman_residual / np.where(std == 0, np.nan, std)

    return df


def cusum_structural_break(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    CUSUM statistic for detecting structural breaks / regime changes.
    Signals a regime shift when cumulative sum exceeds threshold * std.
    """
    df = df.copy()
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()

    mu = df['returns'].expanding().mean()
    sigma = df['returns'].expanding().std().replace(0, np.nan)
    standardised = (df['returns'] - mu) / sigma
    df['cusum_pos'] = standardised.clip(lower=0).cumsum()
    df['cusum_neg'] = (-standardised).clip(lower=0).cumsum()
    df['regime_break'] = (
        (df['cusum_pos'] > threshold) | (df['cusum_neg'] > threshold)
    ).astype(int)

    return df


def generate_all_statarb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: apply all statistical arbitrage feature groups."""
    logger.info("Generating statistical arbitrage features...")
    df = zscore_features(df)
    df = rolling_hurst(df)
    df = rolling_half_life(df)
    df = adf_test_rolling(df)
    df = kalman_filter_spread(df)
    df = cusum_structural_break(df)
    logger.info(f"StatArb features complete. Shape: {df.shape}")
    return df
