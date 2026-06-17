"""
scripts/prepare_data.py
========================
Combine Binance Vision monthly CSVs (Jun 2025 - May 2026) + Hyperliquid
recent data into a single clean 365-day parquet file for backtesting.

Binance Vision timestamps may be microseconds (16 digits) or milliseconds (13 digits).
"""
import sys
sys.path.insert(0, '/home/ubuntu/AIQuant')
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path('/home/ubuntu/AIQuant/data/raw/BTCUSDT_1m_full.parquet')

COLS = ['open_time','open','high','low','close','volume',
        'close_time','quote_vol','n_trades','taker_buy_base','taker_buy_quote','ignore']

RAW = Path('/home/ubuntu/AIQuant/data/raw')

dfs = []

# ── Binance Vision monthly CSVs ───────────────────────────────────────────────
# Include all months from Jun 2025 through May 2026
months = [
    '2025-06','2025-07','2025-08','2025-09','2025-10','2025-11',
    '2025-12','2026-01','2026-02','2026-03','2026-04','2026-05'
]

for ym in months:
    f = RAW / f'BTCUSDT-1m-{ym}.csv'
    if not f.exists():
        print(f"  ✗ Missing: {f.name} (skipped)")
        continue
    df = pd.read_csv(f, header=None, names=COLS, dtype=str)
    ts = df['open_time'].astype(np.int64)
    # Auto-detect microseconds vs milliseconds
    if ts.iloc[0] > 1_000_000_000_000_000:   # 16-digit = microseconds
        ts = ts // 1000
    df.index = pd.to_datetime(ts, unit='ms', utc=True)
    df.index.name = 'open_time'
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['open','high','low','close','volume']].copy()
    print(f"  {f.name}: {len(df):,} bars  "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    dfs.append(df)

# ── Hyperliquid recent data ───────────────────────────────────────────────────
hl_path = RAW / 'BTC_1m_hl.parquet'
if hl_path.exists():
    hl = pd.read_parquet(hl_path)
    print(f"  Hyperliquid:    {len(hl):,} bars  "
          f"{hl.index[0].strftime('%Y-%m-%d')} → {hl.index[-1].strftime('%Y-%m-%d')}")
    dfs.append(hl)

if not dfs:
    print("ERROR: No data files found. Run fetch_hl_data.py first.")
    sys.exit(1)

# ── Combine ───────────────────────────────────────────────────────────────────
combined = pd.concat(dfs).sort_index()
combined = combined[~combined.index.duplicated(keep='last')]
combined = combined.dropna()
combined = combined[(combined['close'] > 0) & (combined['volume'] >= 0)]

print(f"\n  Combined: {len(combined):,} bars  "
      f"{combined.index[0].strftime('%Y-%m-%d')} → {combined.index[-1].strftime('%Y-%m-%d')}")
print(f"  Price range: ${combined['close'].min():,.0f} → ${combined['close'].max():,.0f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(OUT)
print(f"  Saved → {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
print(combined.describe().round(2))
