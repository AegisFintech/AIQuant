"""
aiquant/strategies/ensemble.py
================================
Strategy Ensemble: combines all strategy signals with regime-adaptive weighting.

Priority:
  1. KalmanStatArb   (primary — HFT mean reversion)
  2. MeanReversion   (backup 1 — BB + RSI)
  3. TrendFollowing  (backup 2 — EMA crossover)
  4. MLSignal        (backup 3 — ensemble ML)

Regime routing:
  - Hurst < 0.45  → StatArb + MeanReversion dominant
  - Hurst 0.45–0.55 → ML signal dominant
  - Hurst > 0.55  → TrendFollowing dominant
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

from .stat_arb import KalmanStatArbStrategy
from .mean_reversion import MeanReversionStrategy
from .trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class StrategyEnsemble:
    """
    Regime-adaptive strategy ensemble.
    """

    def __init__(
        self,
        statarb_weight: float = 0.5,
        mr_weight: float = 0.25,
        trend_weight: float = 0.15,
        ml_weight: float = 0.10,
        min_agreement: int = 1,  # Minimum strategies agreeing for a signal
    ):
        self.statarb_weight = statarb_weight
        self.mr_weight = mr_weight
        self.trend_weight = trend_weight
        self.ml_weight = ml_weight
        self.min_agreement = min_agreement

        self.statarb = KalmanStatArbStrategy()
        self.mr = MeanReversionStrategy()
        self.trend = TrendFollowingStrategy()

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_signals: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate weighted ensemble signals.
        Returns a DataFrame with individual and combined signals.
        """
        result = pd.DataFrame(index=df.index)

        # Generate individual signals (with fallback on missing features)
        try:
            result['statarb'] = self.statarb.generate_signals(df)
        except (ValueError, KeyError) as e:
            logger.warning(f"StatArb signal failed: {e}. Setting to 0.")
            result['statarb'] = 0

        try:
            result['mr'] = self.mr.generate_signals(df)
        except (ValueError, KeyError) as e:
            logger.warning(f"MeanReversion signal failed: {e}. Setting to 0.")
            result['mr'] = 0

        try:
            result['trend'] = self.trend.generate_signals(df)
        except (ValueError, KeyError) as e:
            logger.warning(f"TrendFollowing signal failed: {e}. Setting to 0.")
            result['trend'] = 0

        if ml_signals is not None:
            result['ml'] = ml_signals.reindex(df.index).fillna(0)
        else:
            result['ml'] = 0

        # Weighted score
        result['score'] = (
            result['statarb'] * self.statarb_weight
            + result['mr'] * self.mr_weight
            + result['trend'] * self.trend_weight
            + result['ml'] * self.ml_weight
        )

        # Regime-adaptive routing
        if 'hurst' in df.columns:
            hurst = df['hurst'].fillna(0.5)
            # Mean-reverting regime: boost StatArb + MR
            mr_regime = hurst < 0.45
            trend_regime = hurst > 0.55

            result.loc[mr_regime, 'score'] = (
                result.loc[mr_regime, 'statarb'] * 0.60
                + result.loc[mr_regime, 'mr'] * 0.30
                + result.loc[mr_regime, 'ml'] * 0.10
            )
            result.loc[trend_regime, 'score'] = (
                result.loc[trend_regime, 'trend'] * 0.60
                + result.loc[trend_regime, 'ml'] * 0.25
                + result.loc[trend_regime, 'mr'] * 0.15
            )

        # Final signal: threshold on weighted score
        result['final_signal'] = 0
        result.loc[result['score'] >= 0.25, 'final_signal'] = 1
        result.loc[result['score'] <= -0.25, 'final_signal'] = -1

        long_count = (result['final_signal'] == 1).sum()
        short_count = (result['final_signal'] == -1).sum()
        logger.info(
            f"[Ensemble] Final signals — Long: {long_count}, Short: {short_count}, "
            f"Total bars: {len(df)}"
        )

        return result
