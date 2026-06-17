"""
aiquant/strategies/mean_reversion.py
======================================
BACKUP STRATEGY 1: Multi-Signal Mean Reversion on 1m BTCUSD.

Uses Bollinger Bands + RSI + Volume confirmation + Volatility filter.
Designed for high-frequency 1m data with tight risk controls.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
    Multi-signal mean reversion strategy combining:
      - Bollinger Band position (price overextension)
      - RSI (momentum exhaustion)
      - Volume confirmation (capitulation/euphoria volume)
      - Volatility spike filter (avoid during extreme moves)
      - Returns z-score (standardised deviation)
    """

    def __init__(
        self,
        bb_pct_long: float = 0.05,
        bb_pct_short: float = 0.95,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        vol_ratio_min: float = 1.2,
        zscore_long: float = -2.0,
        zscore_short: float = 2.0,
        vol_spike_max: float = 3.0,
        rsi_period: int = 14,
    ):
        self.bb_pct_long = bb_pct_long
        self.bb_pct_short = bb_pct_short
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.vol_ratio_min = vol_ratio_min
        self.zscore_long = zscore_long
        self.zscore_short = zscore_short
        self.vol_spike_max = vol_spike_max
        self.rsi_period = rsi_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        bb_col = 'bb_pct_20'
        rsi_col = f'rsi_{self.rsi_period}'
        zscore_col = 'zscore_returns_60'

        for col in [bb_col, rsi_col, 'vol_ratio', zscore_col, 'vol_regime']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Volatility spike filter: avoid extreme vol environments
        vol_ok = df['vol_regime'] < self.vol_spike_max

        long_entry = (
            vol_ok
            & (df[bb_col] < self.bb_pct_long)
            & (df[rsi_col] < self.rsi_oversold)
            & (df['vol_ratio'] > self.vol_ratio_min)
            & (df[zscore_col] < self.zscore_long)
        )

        short_entry = (
            vol_ok
            & (df[bb_col] > self.bb_pct_short)
            & (df[rsi_col] > self.rsi_overbought)
            & (df['vol_ratio'] > self.vol_ratio_min)
            & (df[zscore_col] > self.zscore_short)
        )

        signals = pd.Series(0, index=df.index, name='signal')
        signals[long_entry] = 1
        signals[short_entry] = -1

        logger.info(
            f"[MeanReversion] Long={long_entry.sum()}, Short={short_entry.sum()}"
        )
        return signals
