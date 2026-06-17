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
        fast_col = f'ema_{self.fast_ema}'
        slow_col = f'ema_{self.slow_ema}'
        macd_col = f'macd_diff_{self.macd_fast}_{self.macd_slow}'

        for col in [fast_col, slow_col, 'adx', 'hurst', 'vol_ratio']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Trend regime filter
        trending = df['hurst'] > self.hurst_min

        # ADX filter: strong trend
        strong_trend = df['adx'] > self.adx_threshold

        # Volume confirmation
        vol_ok = df['vol_ratio'] > self.vol_ratio_min

        # EMA crossover
        ema_bull = df[fast_col] > df[slow_col]
        ema_bear = df[fast_col] < df[slow_col]

        # MACD confirmation (if available)
        if macd_col in df.columns:
            macd_bull = df[macd_col] > 0
            macd_bear = df[macd_col] < 0
        else:
            macd_bull = pd.Series(True, index=df.index)
            macd_bear = pd.Series(True, index=df.index)

        long_entry = trending & strong_trend & vol_ok & ema_bull & macd_bull
        short_entry = trending & strong_trend & vol_ok & ema_bear & macd_bear

        # Only signal on crossover (transition), not persistent state
        long_cross = long_entry & ~long_entry.shift(1).fillna(False)
        short_cross = short_entry & ~short_entry.shift(1).fillna(False)

        signals = pd.Series(0, index=df.index, name='signal')
        signals[long_cross] = 1
        signals[short_cross] = -1

        logger.info(
            f"[TrendFollowing] Long crossovers={long_cross.sum()}, Short crossovers={short_cross.sum()}"
        )
        return signals
