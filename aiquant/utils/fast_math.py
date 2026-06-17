"""
aiquant/utils/fast_math.py
===========================
Numba JIT-compiled implementations of computationally expensive
rolling window functions used in feature engineering.

Why Numba?
----------
pandas .rolling().apply(func) is the slowest operation in the codebase.
For a 100-bar window over 130,000 bars, it calls the Python function
130,000 times — once per row. Each call has Python interpreter overhead,
GIL acquisition, and no SIMD vectorisation.

Numba @njit compiles the function to native machine code on first call
(~0.5s warm-up), then executes at C speed with SIMD on subsequent calls.
Typical speedup: 50–200x over pandas .rolling().apply().

Usage
-----
These functions accept raw NumPy float64 arrays and return float64 arrays.
They are called from the feature engineering modules via a thin wrapper
that extracts .values from the DataFrame, calls the JIT function, and
assigns the result back.
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Hurst Exponent (R/S Analysis)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _hurst_rs(x: np.ndarray, max_lag: int = 20) -> float:
    """
    Compute the Hurst Exponent for a 1-D array using R/S analysis.
    Compiled to native code via Numba @njit.

    Returns 0.5 (random walk) if computation fails.
    """
    n = len(x)
    if n < max_lag * 2:
        return 0.5

    lags = np.arange(2, max_lag + 1)
    rs_vals = np.empty(len(lags), dtype=np.float64)

    for idx in range(len(lags)):
        lag = lags[idx]
        rs_list = np.empty(n // lag, dtype=np.float64)
        count = 0
        for start in range(0, n - lag + 1, lag):
            sub = x[start:start + lag]
            mean_sub = 0.0
            for v in sub:
                mean_sub += v
            mean_sub /= lag

            # Cumulative deviation
            cum_dev = np.empty(lag, dtype=np.float64)
            running = 0.0
            for j in range(lag):
                running += sub[j] - mean_sub
                cum_dev[j] = running

            r = cum_dev.max() - cum_dev.min()

            # Standard deviation
            var = 0.0
            for v in sub:
                var += (v - mean_sub) ** 2
            s = (var / lag) ** 0.5

            if s > 0:
                rs_list[count] = r / s
                count += 1

        if count == 0:
            rs_vals[idx] = 1.0
        else:
            rs_vals[idx] = rs_list[:count].mean()

    # OLS regression of log(RS) on log(lag)
    log_lags = np.log(lags.astype(np.float64))
    log_rs   = np.log(rs_vals + 1e-10)

    n_pts = len(log_lags)
    sum_x  = log_lags.sum()
    sum_y  = log_rs.sum()
    sum_xx = (log_lags * log_lags).sum()
    sum_xy = (log_lags * log_rs).sum()

    denom = n_pts * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.5

    slope = (n_pts * sum_xy - sum_x * sum_y) / denom
    return max(0.0, min(1.0, slope))


@njit(cache=True, parallel=False)
def rolling_hurst_nb(close: np.ndarray, window: int = 240, max_lag: int = 20) -> np.ndarray:
    """
    Rolling Hurst Exponent over a price array.
    Returns float64 array of same length; first (window-1) values are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        out[i] = _hurst_rs(close[i - window + 1:i + 1], max_lag)
    return out


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck Half-Life
# ---------------------------------------------------------------------------

@njit(cache=True)
def _ou_half_life(x: np.ndarray) -> float:
    """
    Estimate OU half-life via OLS on lagged differences.
    Returns NaN if mean reversion is not detected (kappa <= 0).
    """
    n = len(x)
    if n < 10:
        return np.nan

    # y = x[1:], y_lag = x[:-1], delta_y = y - y_lag
    y_lag = x[:-1]
    delta_y = x[1:] - x[:-1]
    m = n - 1

    # OLS: delta_y = alpha + kappa * y_lag
    # Using closed-form OLS with constant
    sum_yl  = 0.0
    sum_dy  = 0.0
    sum_yl2 = 0.0
    sum_ydy = 0.0
    for j in range(m):
        sum_yl  += y_lag[j]
        sum_dy  += delta_y[j]
        sum_yl2 += y_lag[j] * y_lag[j]
        sum_ydy += y_lag[j] * delta_y[j]

    denom = m * sum_yl2 - sum_yl * sum_yl
    if abs(denom) < 1e-12:
        return np.nan

    kappa = -(m * sum_ydy - sum_yl * sum_dy) / denom
    if kappa <= 0:
        return np.nan

    return 0.6931471805599453 / kappa   # ln(2) / kappa


@njit(cache=True)
def rolling_half_life_nb(close: np.ndarray, window: int = 240) -> np.ndarray:
    """
    Rolling OU half-life over a price array.
    Returns float64 array; first (window-1) values are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        out[i] = _ou_half_life(close[i - window + 1:i + 1])
    return out


# ---------------------------------------------------------------------------
# Rolling Autocorrelation
# ---------------------------------------------------------------------------

@njit(cache=True)
def _autocorr(x: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at a given lag for a 1-D array."""
    n = len(x)
    if n <= lag:
        return np.nan

    x1 = x[:n - lag]
    x2 = x[lag:]
    m = n - lag

    mean1 = x1.sum() / m
    mean2 = x2.sum() / m

    num = 0.0
    var1 = 0.0
    var2 = 0.0
    for j in range(m):
        d1 = x1[j] - mean1
        d2 = x2[j] - mean2
        num  += d1 * d2
        var1 += d1 * d1
        var2 += d2 * d2

    denom = (var1 * var2) ** 0.5
    if denom < 1e-12:
        return 0.0
    return num / denom


@njit(cache=True)
def rolling_autocorr_nb(returns: np.ndarray, window: int = 60, lag: int = 1) -> np.ndarray:
    """
    Rolling autocorrelation of returns at a given lag.
    Returns float64 array; first (window-1) values are NaN.
    """
    n = len(returns)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        out[i] = _autocorr(returns[i - window + 1:i + 1], lag)
    return out


# ---------------------------------------------------------------------------
# Roll Spread (serial covariance of price changes)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _serial_cov(x: np.ndarray) -> float:
    """Serial covariance Cov(x_t, x_{t-1}) for a 1-D array of differences."""
    n = len(x)
    if n < 3:
        return 0.0
    x1 = x[:-1]
    x2 = x[1:]
    m  = n - 1
    mean1 = x1.sum() / m
    mean2 = x2.sum() / m
    cov = 0.0
    for j in range(m):
        cov += (x1[j] - mean1) * (x2[j] - mean2)
    return cov / m


@njit(cache=True)
def rolling_roll_spread_nb(close: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Rolling Roll (1984) spread estimator.
    Spread = 2 * sqrt(max(0, -Cov(delta_p_t, delta_p_{t-1})))
    """
    n = len(close)
    delta_p = np.empty(n, dtype=np.float64)
    delta_p[0] = 0.0
    for i in range(1, n):
        delta_p[i] = close[i] - close[i - 1]

    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        cov = _serial_cov(delta_p[i - window:i + 1])
        out[i] = 2.0 * (max(0.0, -cov) ** 0.5)
    return out


# ---------------------------------------------------------------------------
# Kalman Filter (vectorised NumPy — already fast, kept here for reference)
# ---------------------------------------------------------------------------

def kalman_filter_nb(close: np.ndarray, delta: float = 1e-4):
    """
    Kalman Filter for dynamic mean estimation.
    Pure NumPy implementation — already vectorised, no Numba needed.
    Returns (kalman_mean, kalman_residual, kalman_zscore) arrays.
    """
    n = len(close)
    x = np.array([close[0], 0.0], dtype=np.float64)
    P = np.eye(2, dtype=np.float64)
    Q = np.eye(2, dtype=np.float64) * delta
    R = float(np.var(np.diff(close[:50]))) if n > 50 else 1.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)

    kalman_mean     = np.empty(n, dtype=np.float64)
    kalman_residual = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update
        innov = close[i] - (H @ x)[0]
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T).flatten() / S
        x = x + K * innov
        P = P - np.outer(K, H @ P)

        kalman_mean[i]     = x[0]
        kalman_residual[i] = innov

    # Z-score of residuals
    std = np.std(kalman_residual)
    kalman_zscore = kalman_residual / std if std > 1e-10 else np.zeros(n)

    return kalman_mean, kalman_residual, kalman_zscore


# ---------------------------------------------------------------------------
# ADF p-value rolling (thin wrapper — statsmodels is already C-optimised)
# ---------------------------------------------------------------------------

def rolling_adf_nb(close: np.ndarray, window: int = 240) -> np.ndarray:
    """
    Rolling ADF p-value. Uses statsmodels adfuller (Fortran LAPACK).
    Parallelised via numpy pre-extraction to avoid repeated pandas overhead.
    """
    from statsmodels.tsa.stattools import adfuller
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        try:
            out[i] = adfuller(close[i - window + 1:i + 1], autolag='AIC')[1]
        except Exception:
            out[i] = 1.0
    return out


# ---------------------------------------------------------------------------
# Warm up Numba JIT (call once at import time to avoid first-call latency)
# ---------------------------------------------------------------------------

def warmup():
    """
    Pre-compile all Numba JIT functions with a small dummy array.
    Call this once at startup to avoid ~0.5s warm-up on first real use.
    """
    dummy = np.random.randn(300).cumsum() + 50000.0
    _ = rolling_hurst_nb(dummy, window=100, max_lag=10)
    _ = rolling_half_life_nb(dummy, window=100)
    _ = rolling_autocorr_nb(np.diff(dummy), window=60, lag=1)
    _ = rolling_roll_spread_nb(dummy, window=20)
