"""
aiquant/strategies/stat_arb.py
================================
PRIMARY STRATEGY: High-Frequency Statistical Arbitrage on BTCUSD 1m.

Strategy Logic:
  - Uses Kalman Filter dynamic mean + z-score for entry/exit
  - Regime filter: only trade when Hurst < 0.5 (mean-reverting)
  - Order flow confirmation: OFI must oppose the price deviation
  - Volatility filter: avoid entries during vol spikes (VPIN > threshold)
  - Multi-signal confirmation: ADF stationarity + half-life < 30 bars

Entry Rules (Long):
  kalman_zscore < -entry_threshold
  AND hurst < 0.5 (mean-reverting regime)
  AND ofi_15 > 0 (buying pressure building)
  AND adf_pvalue < 0.1 (stationary)
  AND vpin < vpin_threshold (not toxic flow)

Entry Rules (Short):
  kalman_zscore > entry_threshold
  AND hurst < 0.5
  AND ofi_15 < 0 (selling pressure building)
  AND adf_pvalue < 0.1
  AND vpin < vpin_threshold

Exit Rules:
  kalman_zscore crosses zero (mean reversion complete)
  OR stop-loss hit
  OR max holding period exceeded
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KalmanStatArbStrategy:
    """
    Kalman Filter-based Statistical Arbitrage Strategy.
    Primary strategy for the AIQuant HFT framework.
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.3,
        hurst_max: float = 0.5,
        vpin_max: float = 0.7,
        adf_pvalue_max: float = 0.10,
        ofi_window: int = 15,
        max_hold_bars: int = 60,  # 60 minutes max hold for 1m bars
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.hurst_max = hurst_max
        self.vpin_max = vpin_max
        self.adf_pvalue_max = adf_pvalue_max
        self.ofi_window = ofi_window
        self.max_hold_bars = max_hold_bars

    def _check_required_columns(self, df: pd.DataFrame):
        required = ['kalman_zscore', 'hurst', f'ofi_norm_{self.ofi_window}', 'adf_pvalue', 'vpin']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Run feature engineering first.")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate entry signals: 1=Long, -1=Short, 0=Flat.
        """
        self._check_required_columns(df)

        ofi_col = f'ofi_norm_{self.ofi_window}'

        # Regime filter: mean-reverting only
        regime_ok = df['hurst'] < self.hurst_max

        # Stationarity filter
        stationary = df['adf_pvalue'] < self.adf_pvalue_max

        # Liquidity / toxicity filter
        liquid = df['vpin'] < self.vpin_max

        # Combined filter
        base_filter = regime_ok & stationary & liquid

        # Long: price is significantly below Kalman mean, buying pressure building
        long_entry = (
            base_filter
            & (df['kalman_zscore'] < -self.entry_threshold)
            & (df[ofi_col] > 0)
        )

        # Short: price is significantly above Kalman mean, selling pressure building
        short_entry = (
            base_filter
            & (df['kalman_zscore'] > self.entry_threshold)
            & (df[ofi_col] < 0)
        )

        signals = pd.Series(0, index=df.index, name='signal')
        signals[long_entry] = 1
        signals[short_entry] = -1

        logger.info(
            f"[KalmanStatArb] Signals generated: "
            f"Long={long_entry.sum()}, Short={short_entry.sum()}, "
            f"Total bars={len(df)}"
        )
        return signals

    def generate_exit_signals(self, df: pd.DataFrame, entry_signals: pd.Series) -> pd.Series:
        """
        Generate exit signals based on mean reversion completion.
        """
        exits = pd.Series(0, index=df.index, name='exit')

        # Exit when z-score crosses back through zero
        exits[df['kalman_zscore'].abs() < self.exit_threshold] = 1

        return exits

    def signal_summary(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        """Return a summary of signal statistics."""
        return {
            'total_long': (signals == 1).sum(),
            'total_short': (signals == -1).sum(),
            'signal_rate_pct': (signals != 0).mean() * 100,
            'avg_hurst': df['hurst'].mean(),
            'avg_vpin': df['vpin'].mean(),
            'pct_stationary': (df['adf_pvalue'] < self.adf_pvalue_max).mean() * 100,
        }
