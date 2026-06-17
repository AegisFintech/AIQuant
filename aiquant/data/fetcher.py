"""
aiquant/data/fetcher.py
=======================
Professional-grade data fetcher for BTCUSD using CCXT.
Supports 1m OHLCV, multi-timeframe, order book snapshots, and funding rate data.
Designed for high-RAM environments with large in-memory datasets.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BTCDataFetcher:
    """
    Multi-source BTCUSD data fetcher.
    Primary source: Binance (via CCXT) — free, no API key required for public endpoints.
    Handles full pagination, rate limiting, deduplication, and parquet storage.
    """

    TIMEFRAME_MS = {
        '1m': 60_000,
        '5m': 300_000,
        '15m': 900_000,
        '1h': 3_600_000,
        '4h': 14_400_000,
        '1d': 86_400_000,
    }

    def __init__(
        self,
        exchange_id: str = 'binance',
        data_dir: str = 'data',
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.exchange_id = exchange_id
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        exchange_config: Dict = {'enableRateLimit': True, 'timeout': 30000}
        if api_key and api_secret:
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = api_secret

        self.exchange = getattr(ccxt, exchange_id)(exchange_config)
        logger.info(f"Initialized {exchange_id} exchange connector")

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1m',
        since_date: str = '2024-01-01',
        until_date: Optional[str] = None,
        limit: int = 1000,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch full OHLCV history with automatic pagination.
        Stores result as parquet for fast re-loading.
        """
        logger.info(f"Fetching {timeframe} OHLCV for {symbol} from {since_date}")
        since_ms = self.exchange.parse8601(f"{since_date}T00:00:00Z")
        until_ms = (
            self.exchange.parse8601(f"{until_date}T23:59:59Z")
            if until_date
            else self.exchange.milliseconds()
        )

        all_ohlcv: List = []
        tf_ms = self.TIMEFRAME_MS.get(timeframe, 60_000)

        while since_ms < until_ms:
            try:
                batch = self.exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit)
                if not batch:
                    break
                all_ohlcv.extend(batch)
                last_ts = batch[-1][0]
                since_ms = last_ts + tf_ms
                dt_str = datetime.utcfromtimestamp(last_ts / 1000).strftime('%Y-%m-%d %H:%M')
                logger.info(f"  → {dt_str} | total bars: {len(all_ohlcv):,}")
                time.sleep(self.exchange.rateLimit / 1000)
            except ccxt.NetworkError as e:
                logger.warning(f"Network error: {e}. Retrying in 15s...")
                time.sleep(15)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                break

        df = self._ohlcv_to_df(all_ohlcv)
        if save:
            path = self._save_ohlcv(df, symbol, timeframe)
            logger.info(f"Saved {len(df):,} bars to {path}")
        return df

    def fetch_multi_timeframe(
        self,
        symbol: str = 'BTC/USDT',
        timeframes: Optional[List[str]] = None,
        since_date: str = '2024-01-01',
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple timeframes simultaneously for multi-timeframe analysis.
        Returns a dict keyed by timeframe string.
        """
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        result = {}
        for tf in timeframes:
            logger.info(f"Fetching {tf} data...")
            result[tf] = self.fetch_ohlcv(symbol=symbol, timeframe=tf, since_date=since_date)
        return result

    # ------------------------------------------------------------------
    # Order Book Snapshots
    # ------------------------------------------------------------------

    def fetch_orderbook_snapshot(
        self,
        symbol: str = 'BTC/USDT',
        depth: int = 50,
    ) -> Dict:
        """
        Fetch a live order book snapshot.
        Returns bids/asks with depth levels.
        """
        ob = self.exchange.fetch_order_book(symbol, depth)

        # Convert directly to NumPy arrays — avoids DataFrame overhead for
        # scalar aggregations (sum, mean) on small order book snapshots
        bids_arr = np.array(ob['bids'], dtype=np.float64)  # shape (depth, 2): [price, size]
        asks_arr = np.array(ob['asks'], dtype=np.float64)

        bid_prices, bid_sizes = bids_arr[:, 0], bids_arr[:, 1]
        ask_prices, ask_sizes = asks_arr[:, 0], asks_arr[:, 1]

        bid_depth   = np.sum(bid_sizes)
        ask_depth   = np.sum(ask_sizes)
        total_depth = bid_depth + ask_depth
        imbalance   = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        # Weighted mid price (volume-weighted best bid/ask)
        mid_price = (bid_prices[0] + ask_prices[0]) / 2.0
        spread    = ask_prices[0] - bid_prices[0]

        # Keep DataFrames for downstream use but build from arrays
        bids = pd.DataFrame({'price': bid_prices, 'size': bid_sizes})
        asks = pd.DataFrame({'price': ask_prices, 'size': ask_sizes})

        ts = pd.Timestamp.utcnow()
        return {
            'timestamp':  ts,
            'bids':       bids,
            'asks':       asks,
            'mid_price':  float(mid_price),
            'spread':     float(spread),
            'bid_depth':  float(bid_depth),
            'ask_depth':  float(ask_depth),
            'imbalance':  float(imbalance),
        }

    def stream_orderbook_snapshots(
        self,
        symbol: str = 'BTC/USDT',
        n_snapshots: int = 100,
        interval_sec: float = 1.0,
        depth: int = 20,
    ) -> pd.DataFrame:
        """
        Collect N order book snapshots at a given interval.
        Useful for building microstructure features.
        """
        records = []
        logger.info(f"Collecting {n_snapshots} order book snapshots for {symbol}...")
        for i in range(n_snapshots):
            snap = self.fetch_orderbook_snapshot(symbol, depth)
            records.append({
                'timestamp': snap['timestamp'],
                'mid_price': snap['mid_price'],
                'spread': snap['spread'],
                'bid_depth': snap['bid_depth'],
                'ask_depth': snap['ask_depth'],
                'ofi': snap['imbalance'],  # Order Flow Imbalance proxy
            })
            if i % 10 == 0:
                logger.info(f"  Snapshot {i+1}/{n_snapshots} | mid={snap['mid_price']:.2f}")
            time.sleep(interval_sec)
        df = pd.DataFrame(records).set_index('timestamp')
        return df

    # ------------------------------------------------------------------
    # Funding Rate & Open Interest (Binance Futures)
    # ------------------------------------------------------------------

    def fetch_funding_rates(
        self,
        symbol: str = 'BTC/USDT',
        since_date: str = '2024-01-01',
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates from Binance perpetual futures.
        Funding rate is a key signal for mean-reversion and sentiment.
        """
        logger.info(f"Fetching funding rates for {symbol}...")
        try:
            # Use Binance futures endpoint via CCXT
            exchange_futures = ccxt.binanceusdm({'enableRateLimit': True})
            since_ms = exchange_futures.parse8601(f"{since_date}T00:00:00Z")
            all_rates = []
            while True:
                rates = exchange_futures.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
                if not rates:
                    break
                all_rates.extend(rates)
                last_ts = rates[-1]['timestamp']
                if last_ts >= exchange_futures.milliseconds() - 8 * 3600_000:
                    break
                since_ms = last_ts + 1
                time.sleep(exchange_futures.rateLimit / 1000)

            if not all_rates:
                logger.warning("No funding rate data returned.")
                return pd.DataFrame()

            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(r['timestamp'], unit='ms'),
                'funding_rate': r['fundingRate'],
            } for r in all_rates])
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            logger.info(f"Fetched {len(df):,} funding rate records")
            return df
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Recent Trades (Tick Data Proxy)
    # ------------------------------------------------------------------

    def fetch_recent_trades(
        self,
        symbol: str = 'BTC/USDT',
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch the most recent trades for tick-level analysis.
        """
        trades = self.exchange.fetch_trades(symbol, limit=limit)
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime(t['timestamp'], unit='ms'),
            'price': t['price'],
            'amount': t['amount'],
            'side': t['side'],
            'cost': t['cost'],
        } for t in trades])
        df.set_index('timestamp', inplace=True)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def load_ohlcv(self, symbol: str = 'BTC/USDT', timeframe: str = '1m') -> Optional[pd.DataFrame]:
        """Load previously saved OHLCV data from parquet."""
        path = self._ohlcv_path(symbol, timeframe)
        if path.exists():
            logger.info(f"Loading cached data from {path}")
            return pd.read_parquet(path)
        logger.warning(f"No cached data found at {path}")
        return None

    def _ohlcv_to_df(self, raw: List) -> pd.DataFrame:
        """
        Convert raw CCXT OHLCV list to a typed DataFrame.

        Uses np.array for the numeric columns so pandas receives a pre-typed
        float64 block — avoids per-column dtype inference overhead on large
        batches (100k+ bars).
        """
        if not raw:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        arr = np.array(raw, dtype=np.float64)          # shape (n, 6)
        timestamps = pd.to_datetime(arr[:, 0].astype(np.int64), unit='ms', utc=True)
        df = pd.DataFrame(
            {
                'open':   arr[:, 1],
                'high':   arr[:, 2],
                'low':    arr[:, 3],
                'close':  arr[:, 4],
                'volume': arr[:, 5],
            },
            index=timestamps,
        )
        df.index.name = 'timestamp'
        df = df[~df.index.duplicated(keep='first')].sort_index()
        return df

    def _ohlcv_path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace('/', '_')
        return self.raw_dir / f"{safe}_{timeframe}.parquet"

    def _save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        path = self._ohlcv_path(symbol, timeframe)
        df.to_parquet(path, compression='snappy')
        return path
