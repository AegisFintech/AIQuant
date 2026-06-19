"""
scripts/prepare_data.py
========================
Prepare BTCUSDT 1m OHLCV data for backtesting.

Priority order:
  1. If a zip file exists in data/ or data/raw/ matching BTCUSDT*.zip
     → extract it and use the CSVs inside (no internet needed)
  2. Otherwise, download required monthly files from Binance Vision
     (free, no API key, https://data.binance.vision)

Then fetches the last 7 days from Hyperliquid to bridge the gap to today,
and combines everything into data/raw/BTCUSDT_1m_full.parquet.

Usage:
  python3 scripts/prepare_data.py [--days N]

  --days N   How many days of history to include (default: 1825 = 5 years)
             Only affects which months are downloaded; local zips are always
             fully extracted regardless of --days.
"""

import sys
import os
import argparse
import zipfile
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent
DATA    = BASE / 'data'
RAW_DIR = DATA / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT = RAW_DIR / 'BTCUSDT_1m_full.parquet'

COLS = ['open_time','open','high','low','close','volume',
        'close_time','quote_vol','n_trades','taker_buy_base','taker_buy_quote','ignore']

BINANCE_BASE = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_csv(path: Path) -> pd.DataFrame:
    """Load a Binance Vision monthly CSV into a clean OHLCV DataFrame."""
    df = pd.read_csv(path, header=None, names=COLS, dtype=str)
    ts = df['open_time'].astype(np.int64)
    if ts.iloc[0] > 1_000_000_000_000_000:   # microseconds → milliseconds
        ts = ts // 1000
    df.index = pd.to_datetime(ts, unit='ms', utc=True)
    df.index.name = 'open_time'
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['open', 'high', 'low', 'close', 'volume']]


import re as _re
_MONTHLY_PAT = _re.compile(r'^BTCUSDT-1m-\d{4}-\d{2}\.zip$')

def _find_local_zips() -> list:
    """Search data/ and data/raw/ for bundle zip files that contain BTCUSDT CSVs.
    Accepts any filename (21.zip, 22.zip, BTCUSDT_1m_2021.zip, etc.).
    Skips per-month Binance Vision zips (BTCUSDT-1m-YYYY-MM.zip) since those
    are intermediate download artifacts, not user-supplied bundles.
    """
    zips = []
    for d in [DATA, RAW_DIR]:
        for z in sorted(d.glob('*.zip')):
            # Skip per-month Binance Vision zips (already extracted)
            if _MONTHLY_PAT.match(z.name):
                continue
            # Skip files too small to be a real bundle (< 100 KB)
            if z.stat().st_size < 100_000:
                continue
            # Verify it is actually a valid zip before adding
            if not zipfile.is_zipfile(z):
                print(f"  ⚠  Skipping {z.name} — not a valid zip file")
                continue
            zips.append(z)
    # Deduplicate by resolved path
    seen = set()
    result = []
    for z in zips:
        rp = z.resolve()
        if rp not in seen:
            seen.add(rp)
            result.append(z)
    return result


def _extract_zip(zip_path: Path) -> int:
    """
    Extract a zip into RAW_DIR. Handles two formats:
      - Flat zip: contains BTCUSDT-1m-YYYY-MM.csv directly
      - Nested zip: contains BTCUSDT-1m-YYYY-MM.zip, each containing a CSV
    Returns number of CSV files extracted.
    """
    print(f"  Extracting {zip_path.name}  ({zip_path.stat().st_size/1e6:.1f} MB)...")
    extracted = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            # Skip macOS metadata
            if '__MACOSX' in member or member.startswith('.'):
                continue
            name = Path(member).name
            if name.endswith('.csv') and 'BTCUSDT-1m-' in name:
                dest = RAW_DIR / name
                if not dest.exists():
                    with z.open(member) as src, open(dest, 'wb') as dst:
                        dst.write(src.read())
                    extracted += 1
                else:
                    print(f"    (cached) {name}")
                    extracted += 1
            elif name.endswith('.zip') and 'BTCUSDT-1m-' in name:
                # Inner zip — extract to temp then extract inner CSV
                inner_dest = RAW_DIR / name
                with z.open(member) as src, open(inner_dest, 'wb') as dst:
                    dst.write(src.read())
                with zipfile.ZipFile(inner_dest) as iz:
                    for inner_member in iz.namelist():
                        inner_name = Path(inner_member).name
                        if inner_name.endswith('.csv'):
                            csv_dest = RAW_DIR / inner_name
                            if not csv_dest.exists():
                                with iz.open(inner_member) as isrc, open(csv_dest, 'wb') as idst:
                                    idst.write(isrc.read())
                                extracted += 1
                            else:
                                extracted += 1
                inner_dest.unlink()
    return extracted


def _download_month(ym: str) -> bool:
    """Download BTCUSDT-1m-YYYY-MM.zip from Binance Vision and extract CSV."""
    csv_path = RAW_DIR / f'BTCUSDT-1m-{ym}.csv'
    if csv_path.exists():
        print(f"  ✓ {ym}  (cached)")
        return True
    zip_url  = BINANCE_BASE + f'BTCUSDT-1m-{ym}.zip'
    zip_path = RAW_DIR / f'BTCUSDT-1m-{ym}.zip'
    try:
        r = requests.get(zip_url, timeout=120, stream=True)
        if r.status_code == 404:
            print(f"  ✗ {ym}  not yet on Binance Vision — skipped")
            return False
        if r.status_code != 200:
            print(f"  ✗ {ym}  HTTP {r.status_code}")
            return False
        downloaded = 0
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                downloaded += len(chunk)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW_DIR)
        zip_path.unlink()
        print(f"  ✓ {ym}  ({downloaded/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ {ym}  {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def _fetch_hyperliquid() -> pd.DataFrame | None:
    """Fetch last 7 days from Hyperliquid to bridge gap to today."""
    try:
        url     = 'https://api.hyperliquid.xyz/info'
        now_ms  = int(time.time() * 1000)
        start_ms = now_ms - 7 * 24 * 60 * 60 * 1000
        r = requests.post(url, json={
            'type': 'candleSnapshot',
            'req': {'coin': 'BTC', 'interval': '1m',
                    'startTime': start_ms, 'endTime': now_ms}
        }, timeout=20)
        bars = r.json()
        if not bars:
            return None
        rows = []
        for b in bars:
            ts_ms = int(b.get('t', b.get('time', b.get('T', 0))))
            rows.append({'open': float(b.get('o', 0)), 'high': float(b.get('h', 0)),
                         'low':  float(b.get('l', 0)), 'close': float(b.get('c', 0)),
                         'volume': float(b.get('v', 0)), 'open_time': ts_ms})
        hl = pd.DataFrame(rows)
        hl.index = pd.to_datetime(hl['open_time'], unit='ms', utc=True)
        hl.index.name = 'open_time'
        return hl[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"  ⚠  Hyperliquid unavailable ({e}) — using CSV data only")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Prepare BTCUSDT 1m data for AIQuant')
    parser.add_argument('--days', type=int, default=1825,
                        help='Days of history to include (default: 1825 = 5 years)')
    args = parser.parse_args()

    print(f"\n  AIQuant — Data Preparation  (target: {args.days} days)")
    print(f"  {'─' * 58}")

    dfs = []

    # ── Step 1: Check for local zip ───────────────────────────────────────────
    local_zips = _find_local_zips()
    if local_zips:
        print(f"\n  ✓ Found {len(local_zips)} local zip(s) in data/ — extracting...")
        total_extracted = 0
        for zp in local_zips:
            n = _extract_zip(zp)
            total_extracted += n
            print(f"    → {n} CSV files from {zp.name}")
        print(f"  ✓ Extracted {total_extracted} CSV files total")
        print(f"  ℹ  Skipping Binance Vision download (local data found)")
    else:
        # ── Step 2: Download from Binance Vision ──────────────────────────────
        print(f"\n  No local zip found — downloading from Binance Vision...")
        try:
            from dateutil.relativedelta import relativedelta as _rd
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'python-dateutil'],
                           check=True)
            from dateutil.relativedelta import relativedelta as _rd

        now_utc    = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0)
        start_date = now_utc - _rd(days=args.days)
        months = []
        cursor = start_date.replace(day=1)
        while cursor < now_utc:
            months.append(cursor.strftime('%Y-%m'))
            cursor += _rd(months=1)

        print(f"  Downloading {len(months)} monthly files ({months[0]} → {months[-1]})...")
        for ym in months:
            _download_month(ym)

    # ── Step 3: Load all available CSVs ──────────────────────────────────────
    csv_files = sorted(RAW_DIR.glob('BTCUSDT-1m-????-??.csv'))
    print(f"\n  Loading {len(csv_files)} CSV files...")
    for f in csv_files:
        try:
            df = _parse_csv(f)
            print(f"  ✓ {f.name}: {len(df):,} bars  "
                  f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")

    if not dfs:
        print("\n  ERROR: No data loaded. Place a BTCUSDT*.zip in the data/ folder and retry.")
        sys.exit(1)

    # ── Step 4: Hyperliquid bridge ────────────────────────────────────────────
    print(f"\n  Fetching Hyperliquid bridge (last 7 days)...")
    hl = _fetch_hyperliquid()
    if hl is not None:
        hl.to_parquet(RAW_DIR / 'BTC_1m_hl.parquet')
        print(f"  ✓ Hyperliquid: {len(hl):,} bars  "
              f"{hl.index[0].strftime('%Y-%m-%d')} → {hl.index[-1].strftime('%Y-%m-%d')}")
        dfs.append(hl)

    # ── Step 5: Combine and save ──────────────────────────────────────────────
    print(f"\n  Combining {len(dfs)} sources...")
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.dropna()
    combined = combined[(combined['close'] > 0) & (combined['volume'] >= 0)]

    # Trim to requested days
    if args.days:
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=args.days)
        combined = combined[combined.index >= cutoff]

    combined.to_parquet(OUT)

    print(f"\n  {'─' * 58}")
    print(f"  ✓ Dataset saved → {OUT.relative_to(BASE)}")
    print(f"    Bars  : {len(combined):,}")
    print(f"    Range : {combined.index[0].strftime('%Y-%m-%d')} → "
          f"{combined.index[-1].strftime('%Y-%m-%d')}")
    print(f"    Price : ${combined['close'].min():,.0f} → ${combined['close'].max():,.0f}")
    print(f"    Size  : {OUT.stat().st_size/1e6:.1f} MB\n")


if __name__ == '__main__':
    main()
