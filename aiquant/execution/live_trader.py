"""
aiquant/execution/live_trader.py
==================================
Live / Paper Trading Orchestrator.

Connects the full pipeline:
  DataFetcher → FeatureEngineering → StrategyEnsemble → RiskManager → HyperliquidTrader

Designed for 1-minute polling with real-time signal generation.
"""

import time
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from ..data.fetcher import BTCDataFetcher
from ..features import build_full_feature_set
from ..strategies.ensemble import StrategyEnsemble
from ..risk.position_sizing import RiskManager
from .hyperliquid_trader import HyperliquidPaperTrader

logger = logging.getLogger(__name__)


class LiveTradingOrchestrator:
    """
    Full live/paper trading orchestrator for BTCUSD on Hyperliquid.

    Workflow per tick (1 minute):
      1. Fetch latest 1m OHLCV bars (rolling window)
      2. Compute all features
      3. Generate ensemble signal
      4. Apply risk management (Kelly sizing, drawdown limits)
      5. Execute on Hyperliquid testnet
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        use_testnet: bool = True,
        lookback_bars: int = 500,   # Bars of history needed for features
        symbol: str = 'BTC/USDT',
        coin: str = 'BTC',
        data_dir: str = 'data',
        log_dir: str = 'logs',
        initial_capital: float = 10_000.0,
        kelly_fraction: float = 0.5,
    ):
        self.symbol = symbol
        self.coin = coin
        self.lookback_bars = lookback_bars
        self.initial_capital = initial_capital

        # Components
        self.fetcher = BTCDataFetcher(data_dir=data_dir)
        self.strategy = StrategyEnsemble()
        self.risk = RiskManager(kelly_fraction=kelly_fraction)
        self.trader = HyperliquidPaperTrader(
            private_key=private_key,
            use_testnet=use_testnet,
            log_dir=log_dir,
        )

        self.is_running = False
        self.current_signal = 0
        self.bar_count = 0

    def start(self, poll_interval_sec: float = 60.0, max_bars: Optional[int] = None):
        """
        Start the live trading loop.
        Polls every poll_interval_sec seconds (default: 60s for 1m bars).
        """
        logger.info("=" * 60)
        logger.info("  AIQuant Live Trading Orchestrator Starting")
        logger.info(f"  Symbol: {self.symbol} | Mode: {'TESTNET' if self.trader.use_testnet else 'MAINNET'}")
        logger.info("=" * 60)

        # Connect to Hyperliquid
        if not self.trader.connect():
            logger.error("Failed to connect to Hyperliquid. Aborting.")
            return

        self.trader.print_account_summary()
        self.is_running = True

        while self.is_running:
            try:
                self._tick()
                self.bar_count += 1

                if max_bars and self.bar_count >= max_bars:
                    logger.info(f"Reached max_bars={max_bars}. Stopping.")
                    break

                logger.info(f"Sleeping {poll_interval_sec}s until next bar...")
                time.sleep(poll_interval_sec)

            except KeyboardInterrupt:
                logger.info("Live trading stopped by user.")
                self._shutdown()
                break
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)
                time.sleep(poll_interval_sec)

    def _tick(self):
        """Execute one trading tick."""
        ts = datetime.utcnow().isoformat()
        logger.info(f"\n--- Tick @ {ts} ---")

        # 1. Fetch recent data
        df = self.fetcher.fetch_ohlcv(
            symbol=self.symbol,
            timeframe='1m',
            since_date=(pd.Timestamp.utcnow() - pd.Timedelta(hours=self.lookback_bars // 60 + 2)).strftime('%Y-%m-%d'),
            save=False,
        )
        if df is None or len(df) < 300:
            logger.warning("Insufficient data. Skipping tick.")
            return

        # Use only the most recent lookback_bars
        df = df.tail(self.lookback_bars)

        # 2. Feature engineering
        try:
            df_feat = build_full_feature_set(df)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return

        # 3. Generate signals
        try:
            signals_df = self.strategy.generate_signals(df_feat)
            signal = int(signals_df['final_signal'].iloc[-1])
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            signal = 0

        logger.info(f"Signal: {signal} ({'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'FLAT'})")

        # 4. Risk management
        account = self.trader.get_account_state()
        portfolio_value = account.get('account_value', self.initial_capital)
        self.risk.update(portfolio_value)

        price = df['close'].iloc[-1]
        current_vol = df_feat['realvol_20'].iloc[-1] if 'realvol_20' in df_feat.columns else None
        baseline_vol = df_feat['realvol_240'].iloc[-1] if 'realvol_240' in df_feat.columns else None

        trade_log_df = self.trader.get_trade_log_df()
        sizing = self.risk.get_position_size(
            portfolio_value=portfolio_value,
            price=price,
            trade_log=trade_log_df if not trade_log_df.empty else None,
            current_vol=current_vol,
            baseline_vol=baseline_vol,
        )

        if sizing.get('blocked'):
            logger.warning(f"Trade blocked by risk manager: {sizing.get('reason')}")
            return

        size_usd = sizing['position_usd']
        kelly_used = sizing['kelly_fraction_used']
        logger.info(
            f"Kelly fraction: {kelly_used:.3f} | "
            f"Position size: ${size_usd:,.2f} | "
            f"Portfolio: ${portfolio_value:,.2f}"
        )

        # 5. Execute
        if signal != self.current_signal:
            if signal == 1:
                self.trader.close_position(self.coin)
                result = self.trader.place_market_order(self.coin, 'long', size_usd)
            elif signal == -1:
                self.trader.close_position(self.coin)
                result = self.trader.place_market_order(self.coin, 'short', size_usd)
            else:
                result = self.trader.close_position(self.coin)

            self.current_signal = signal
            logger.info(f"Order result: {result.get('success', False)}")
        else:
            logger.info("Signal unchanged. No order placed.")

    def _shutdown(self):
        """Graceful shutdown: close all positions."""
        logger.info("Shutting down. Closing all positions...")
        self.trader.close_position(self.coin)
        self.trader.print_account_summary()
        self.is_running = False
