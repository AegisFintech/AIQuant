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

        # Regime, strength, and volume filters
        trending     = hurst     > self.hurst_min
        strong_trend = adx       > self.adx_threshold
        vol_ok       = vol_ratio > self.vol_ratio_min

        # EMA crossover state (bool arrays)
        ema_bull = ema_fast > ema_slow   # fast above slow = bullish
        ema_bear = ema_fast < ema_slow   # fast below slow = bearish

        # MACD confirmation
        if macd_col in df.columns:
            macd = df[macd_col].to_numpy(dtype=np.float64)
            macd_bull = macd > 0
            macd_bear = macd < 0
        else:
            macd_bull = np.ones(len(df), dtype=bool)
            macd_bear = np.ones(len(df), dtype=bool)

        long_state  = trending & strong_trend & vol_ok & ema_bull & macd_bull
        short_state = trending & strong_trend & vol_ok & ema_bear & macd_bear

        # Crossover detection via np.diff (transition 0->1 only, not persistent state)
        # np.diff returns array of length n-1; prepend False to preserve alignment
        long_cross  = np.concatenate([[False], np.diff(long_state.astype(np.int8))  > 0])
        short_cross = np.concatenate([[False], np.diff(short_state.astype(np.int8)) > 0])

        # Build int8 signal array
        signals_arr = np.zeros(len(df), dtype=np.int8)
        signals_arr[long_cross]  =  1
        signals_arr[short_cross] = -1

        logger.info(
            f"[TrendFollowing] Long crossovers={int(np.sum(long_cross))}, "
            f"Short crossovers={int(np.sum(short_cross))}, Total bars={len(df)}"
        )
        return pd.Series(signals_arr, index=df.index, name='signal')
