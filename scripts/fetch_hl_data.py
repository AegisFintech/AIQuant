"""
scripts/fetch_hl_data.py
========================
Fetch N days of BTCUSD 1m data from Hyperliquid public API.
Batches in 3-day windows (4,320 bars each) to stay within API limits.
No auth required. Saves to data/raw/BTC_1m_hl.parquet
"""
import sys, time, requests
sys.path.insert(0, '/home/ubuntu/AIQuant')
import pandas as pd
import numpy as np
from pathlib import Path

CACHE = Path('/home/ubuntu/AIQuant/data/raw/BTC_1m_hl.parquet')
BASE  = "https://api.hyperliquid.xyz/info"
DAYS  = 90

# Hyperliquid returns max ~4321 bars per request (3 days of 1m)
BATCH_DAYS = 3
BATCH_MS   = BATCH_DAYS * 24 * 60 * 60 * 1000


def fetch_hl_candles(days: int = 90, force: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not force:
        age_h = (time.time() - CACHE.stat().st_mtime) / 3600
        if age_h < 6:
            df = pd.read_parquet(CACHE)
            print(f"  Loaded cache ({df.shape[0]:,} bars, {age_h:.1f}h old)")
            return df

    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    cursor   = start_ms
    all_bars = []
    n_batches = (days + BATCH_DAYS - 1) // BATCH_DAYS

    print(f"  Fetching {days}d of BTC 1m from Hyperliquid ({n_batches} batches)...")

    for i in range(n_batches + 5):
        end_ms = min(cursor + BATCH_MS, now_ms)
        if cursor >= now_ms:
            break

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": "BTC",
                "interval": "1m",
                "startTime": cursor,
                "endTime":   end_ms
            }
        }

        for attempt in range(3):
            try:
                r = requests.post(BASE, json=payload, timeout=20)
                data = r.json()
                break
            except Exception as e:
                print(f"\n  Batch {i+1} attempt {attempt+1} failed: {e}")
                time.sleep(2)
                data = []

        if not data:
            cursor = end_ms + 60_000
            continue

        all_bars.extend(data)
        last_t = data[-1]['t']
        cursor = last_t + 60_000

        pct = min((i + 1) / n_batches * 100, 100)
        print(f"  [{pct:5.1f}%] {len(all_bars):>7,} bars  "
              f"up to {pd.Timestamp(last_t, unit='ms').strftime('%Y-%m-%d %H:%M')}",
              end='\r')

        time.sleep(0.15)  # polite rate limit

    print(f"\n  Total fetched: {len(all_bars):,} bars")

    if not all_bars:
        raise RuntimeError("No data returned from Hyperliquid. Check API availability.")

    # Build DataFrame
    df = pd.DataFrame({
        'open':   np.array([float(b['o']) for b in all_bars], dtype=np.float64),
        'high':   np.array([float(b['h']) for b in all_bars], dtype=np.float64),
        'low':    np.array([float(b['l']) for b in all_bars], dtype=np.float64),
        'close':  np.array([float(b['c']) for b in all_bars], dtype=np.float64),
        'volume': np.array([float(b['v']) for b in all_bars], dtype=np.float64),
    }, index=pd.to_datetime([b['t'] for b in all_bars], unit='ms', utc=True))
    df.index.name = 'open_time'

    # Deduplicate and sort
    df = df[~df.index.duplicated(keep='last')].sort_index()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    print(f"  Saved → {CACHE}")
    print(f"  {df.shape[0]:,} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


if __name__ == '__main__':
    import sys
    force = '--force' in sys.argv
    df = fetch_hl_candles(DAYS, force=force)
    print(df.tail(3))
