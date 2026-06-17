"""
aiquant/strategies/trend_following.py
=======================================
BACKUP STRATEGY 2: Multi-Timeframe Trend Following on 1m BTCUSD.

Uses EMA crossover + ADX trend strength + MACD confirmation
+ Volume trend filter. Designed to capture momentum regimes
when the Hurst exponent indicates trending (H > 0.55).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    """
    Multi-timeframe trend following strategy:
      - Fast/slow EMA crossover (primary signal)
      - ADX > threshold (trend strength confirmation)
      - MACD histogram direction (momentum confirmation)
      - Hurst > 0.55 (trending regime filter)
      - Volume above average (volume confirmation)
    """

    def __init__(
        self,
        fast_ema: int = 14,
        slow_ema: int = 50,
        adx_threshold: float = 25.0,
        hurst_min: float = 0.55,
        vol_ratio_min: float = 1.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.adx_threshold = adx_threshold
        self.hurst_min = hurst_min
        self.vol_ratio_min = vol_ratio_min
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trend-following signals using fully vectorised NumPy operations.

        Crossover detection uses np.diff on boolean arrays (O(n) single pass)
        instead of pandas .shift() which allocates a new Series.
        """
        # Guard: empty DataFrame returns flat signal
        if len(df) == 0:
            return pd.Series(np.zeros(0, dtype=np.int8), index=df.index, name='signal')

        fast_col = f'ema_{self.fast_ema}'
        slow_col = f'ema_{self.slow_ema}'
        macd_col = f'macd_diff_{self.macd_fast}_{self.macd_slow}'

        for col in [fast_col, slow_col, 'adx', 'hurst', 'vol_ratio']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Extract to contiguous NumPy float64 arrays
        ema_fast  = df[fast_col].to_numpy(dtype=np.float64)
        ema_slow  = df[slow_col].to_numpy(dtype=np.float64)
        adx       = df['adx'].to_numpy(dtype=np.float64)
        hurst     = df['hurst'].to_numpy(dtype=np.float64)
        vol_ratio = df['vol_ratio'].to_numpy(dtype=np.float64)

        n = len(df)

        # Replace NaN with neutral values before boolean ops
        ema_fast  = np.nan_to_num(ema_fast,  nan=0.0)
        ema_slow  = np.nan_to_num(ema_slow,  nan=0.0)
        adx       = np.nan_to_num(adx,       nan=0.0)
        hurst     = np.nan_to_num(hurst,     nan=0.5)
        vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)

        # Regime, strength, and volume filters — all produce shape (n,) bool
        trending     = (hurst     > self.hurst_min).reshape(n)
        strong_trend = (adx       > self.adx_threshold).reshape(n)
        vol_ok       = (vol_ratio > self.vol_ratio_min).reshape(n)

        # EMA crossover state (bool arrays)
        ema_bull = (ema_fast > ema_slow).reshape(n)
        ema_bear = (ema_fast < ema_slow).reshape(n)

        # MACD confirmation — always produce shape (n,)
        if macd_col in df.columns:
            macd = np.nan_to_num(df[macd_col].to_numpy(dtype=np.float64), nan=0.0).reshape(n)
            macd_bull = (macd > 0).reshape(n)
            macd_bear = (macd < 0).reshape(n)
        else:
            macd_bull = np.ones(n, dtype=bool)
            macd_bear = np.ones(n, dtype=bool)

        long_state  = trending & strong_trend & vol_ok & ema_bull & macd_bull
        short_state = trending & strong_trend & vol_ok & ema_bear & macd_bear

        # Crossover detection — safe: explicitly ensure length n
        long_diff  = np.diff(long_state.astype(np.int8))
        short_diff = np.diff(short_state.astype(np.int8))
        long_cross  = np.concatenate([[False], long_diff  > 0])[:n]
        short_cross = np.concatenate([[False], short_diff > 0])[:n]

        # Build int8 signal array
        signals_arr = np.zeros(n, dtype=np.int8)
        signals_arr[long_cross]  =  1
        signals_arr[short_cross] = -1

        logger.info(
            f"[TrendFollowing] Long crossovers={int(np.sum(long_cross))}, "
            f"Short crossovers={int(np.sum(short_cross))}, Total bars={len(df)}"
        )
        return pd.Series(signals_arr, index=df.index, name='signal')
