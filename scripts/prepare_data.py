"""
scripts/prepare_data.py
========================
Combine Binance Vision monthly CSVs + Hyperliquid recent data
into a single clean parquet file for backtesting.

Binance Vision timestamps are in microseconds.
Hyperliquid timestamps are in milliseconds.
"""
import sys
sys.path.insert(0, '/home/ubuntu/AIQuant')
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path('/home/ubuntu/AIQuant/data/raw/BTCUSDT_1m_full.parquet')

cols = ['open_time','open','high','low','close','volume',
        'close_time','quote_vol','n_trades','taker_buy_base','taker_buy_quote','ignore']

dfs = []

# ── Binance Vision monthly CSVs (microsecond timestamps) ─────────────────────
binance_files = sorted(Path('/home/ubuntu/AIQuant/data/raw').glob('BTCUSDT-1m-2026-*.csv'))
for f in binance_files:
    df = pd.read_csv(f, header=None, names=cols)
    # Timestamps are in microseconds — divide by 1000 to get ms, then parse
    df['open_time'] = pd.to_datetime(df['open_time'] // 1000, unit='ms', utc=True)
    df = df.set_index('open_time')[['open','high','low','close','volume']].astype(np.float64)
    print(f"  {f.name}: {len(df):,} bars  "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    dfs.append(df)

# ── Hyperliquid recent data (millisecond timestamps, already parsed) ──────────
hl_path = Path('/home/ubuntu/AIQuant/data/raw/BTC_1m_hl.parquet')
if hl_path.exists():
    hl = pd.read_parquet(hl_path)
    print(f"  Hyperliquid:    {len(hl):,} bars  "
          f"{hl.index[0].strftime('%Y-%m-%d')} → {hl.index[-1].strftime('%Y-%m-%d')}")
    dfs.append(hl)

# ── Combine ───────────────────────────────────────────────────────────────────
combined = pd.concat(dfs).sort_index()
combined = combined[~combined.index.duplicated(keep='last')]

print(f"\n  Combined: {len(combined):,} bars  "
      f"{combined.index[0].strftime('%Y-%m-%d')} → {combined.index[-1].strftime('%Y-%m-%d')}")

OUT.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(OUT)
print(f"  Saved → {OUT}")
print(combined.describe().round(2))
