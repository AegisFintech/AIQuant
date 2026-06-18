"""
scripts/build_colab.py
=======================
Rebuilds AIQuant_Colab.ipynb with the unified ML ensemble pipeline.
Run: python3 scripts/build_colab.py
"""

import json
from pathlib import Path

ROOT    = Path(__file__).parent.parent
NB_PATH = ROOT / 'AIQuant_Colab.ipynb'


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split('\n')}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source.split('\n')],
    }


cells = []

# ── Header ────────────────────────────────────────────────────────────────────
cells.append(md("""# AIQuant — HFT Statistical Arbitrage
**AegisFintech · Apache 2.0 · [github.com/AegisFintech/AIQuant](https://github.com/AegisFintech/AIQuant)**

> **GPU Runtime Recommended** — Go to `Runtime → Change runtime type → GPU (T4 or A100)` for faster ML training.

| Step | Description |
|------|-------------|
| 1 | Clone repository |
| 2 | Install dependencies + verify GPU |
| 3 | Configuration |
| 4 | Download Binance Vision dataset (auto-scales with DAYS setting) |
| 5 | GPU Feature Engineering (CuPy + Numba parallel) |
| 6 | ML Ensemble Backtest (XGBoost + LightGBM + LSTM, walk-forward CV) |
| 7 | View results chart |
| 8 | Live Trading (Hyperliquid mainnet) |
| 9 | Download results |

**Default: 5 years of BTCUSDT 1m data (Jan 2021 – Jun 2026 · ~2.63M bars)**

Validated on 365-day window (Jun 2025 – Jun 2026):

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **+0.625** |
| Total Return | +6.26% |
| Max Drawdown | -7.87% |
| Win Rate | 63.5% |
| Total Trades | 19,302 |
| Dataset | 530,628 bars · $60k → $126k |"""))

# ── Step 1: Clone ─────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Clone Repository"))
cells.append(code("""import os, sys, shutil
from pathlib import Path

# Purge any stale .pyc cache from previous runs
os.chdir('/content')
!rm -rf AIQuant
!git clone https://github.com/AegisFintech/AIQuant.git
os.chdir('/content/AIQuant')

# Purge pycache to avoid stale bytecode issues
for p in Path('/content/AIQuant').rglob('__pycache__'):
    shutil.rmtree(p, ignore_errors=True)

!git log --oneline -3
print('\\n✓ Repository ready')"""))

# ── Step 2: Install + GPU check ───────────────────────────────────────────────
cells.append(md("""## Step 2 — Install Dependencies & Verify GPU

Installs all required packages and verifies GPU availability.
CuPy enables GPU-accelerated feature engineering (~10x faster than CPU)."""))
cells.append(code("""# Install all dependencies
!pip install -q -r requirements.txt

# Explicit installs for GPU-critical packages
!pip install -q cupy-cuda12x xgboost lightgbm torch torchvision
!pip install -q ta numba pyarrow statsmodels scikit-learn python-dotenv

print('\\n' + '='*60)
print('  GPU & Library Verification')
print('='*60)

# PyTorch CUDA
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  ✓ PyTorch CUDA  : {name} ({mem:.1f} GB)')
        print(f'    CUDA version  : {torch.version.cuda}')
    else:
        print('  ✗ PyTorch CUDA  : NOT available — switch to GPU runtime!')
except Exception as e:
    print(f'  ✗ PyTorch        : {e}')

# CuPy
try:
    import cupy as cp
    cp.array([1.0])
    mem = cp.cuda.Device(0).mem_info
    print(f'  ✓ CuPy CUDA     : {cp.__version__}  ({mem[0]/1e9:.1f}/{mem[1]/1e9:.1f} GB free)')
except Exception as e:
    print(f'  ✗ CuPy           : {e}')
    print('    → Feature engineering will fall back to CPU (Numba parallel)')

# XGBoost
try:
    import xgboost as xgb
    print(f'  ✓ XGBoost        : {xgb.__version__}')
except Exception as e:
    print(f'  ✗ XGBoost        : {e}')

# LightGBM
try:
    import lightgbm as lgb
    print(f'  ✓ LightGBM       : {lgb.__version__}')
except Exception as e:
    print(f'  ✗ LightGBM       : {e}')

# Numba
try:
    import numba
    print(f'  ✓ Numba          : {numba.__version__}')
except Exception as e:
    print(f'  ✗ Numba          : {e}')

print('='*60)"""))

# ── Step 3: Config ────────────────────────────────────────────────────────────
cells.append(md("""## Step 3 — Configuration

Edit the variables below. Only `HYPERLIQUID_PRIVATE_KEY` is required for live trading.
Everything else works without any API keys."""))
cells.append(code("""import os, sys
sys.path.insert(0, '/content/AIQuant')
os.chdir('/content/AIQuant')

# ── Settings ──────────────────────────────────────────────────────────────────
PAIR                    = 'BTCUSDT'   # Trading pair
DAYS                    = 1825        # Days of backtest data (1825 = 5 years, max ~3285 = 9 years)
INITIAL_CAPITAL         = 100_000     # Starting capital (USD)
HYPERLIQUID_PRIVATE_KEY = ''          # Your Hyperliquid private key (for live trading)
FAST_MODE               = False       # True = skip LSTM (~3x faster, slightly lower Sharpe)

# ── Write .env ────────────────────────────────────────────────────────────────
env_content = f\"\"\"HYPERLIQUID_PRIVATE_KEY={HYPERLIQUID_PRIVATE_KEY}
HYPERLIQUID_TESTNET=false
TRADING_PAIR={PAIR}
INITIAL_CAPITAL={INITIAL_CAPITAL}
\"\"\"
with open('.env', 'w') as f:
    f.write(env_content)

print(f'✓ Config set: {PAIR}  |  {DAYS} days  |  ${INITIAL_CAPITAL:,} capital')
print(f'  Fast mode (skip LSTM): {FAST_MODE}')
print(f'  Hyperliquid live trading: {\"ENABLED\" if HYPERLIQUID_PRIVATE_KEY else \"DISABLED (no key)\"}'
)"""))

# ── Step 4: Download Binance Vision data ──────────────────────────────────────
cells.append(md("""## Step 4 — Download Binance Vision Dataset

Downloads the required monthly BTCUSDT 1-minute OHLCV files from Binance Vision (free, no API key).
The number of files is **automatically calculated from the `DAYS` setting in Step 3** —
change `DAYS = 1825` and this step downloads 5 years of data automatically.

| DAYS | Files | Approx size | Approx bars |
|------|-------|-------------|-------------|
| 365 (1 year) | 12 | ~40 MB | 530k |
| 730 (2 years) | 24 | ~80 MB | 1.05M |
| 1095 (3 years) | 36 | ~120 MB | 1.58M |
| 1825 (5 years) | 60 | ~200 MB | 2.63M |

> Binance Vision has BTC 1m data going back to **January 2017**. Files are cached so re-runs are instant."""))
cells.append(code("""import os, sys, requests, zipfile, time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

try:
    from dateutil.relativedelta import relativedelta as _rd
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'python-dateutil'], check=True)
    from dateutil.relativedelta import relativedelta as _rd

RAW_DIR = Path('/content/AIQuant/data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_VISION_BASE = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/'
COLS = ['open_time','open','high','low','close','volume',
        'close_time','quote_vol','n_trades','taker_buy_base','taker_buy_quote','ignore']

# Dynamically compute required months from DAYS (set in Step 3)
now_utc    = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
start_date = now_utc - _rd(days=DAYS)

months = []
cursor = start_date.replace(day=1)
while cursor < now_utc:
    months.append(cursor.strftime('%Y-%m'))
    cursor += _rd(months=1)

print(f'DAYS={DAYS}  →  {len(months)} monthly files required')
print(f'Coverage: {months[0]} → {months[-1]}')
print()
t0 = time.time()

for ym in months:
    csv_path = RAW_DIR / f'BTCUSDT-1m-{ym}.csv'
    if csv_path.exists():
        print(f'  ✓ {ym}  (cached)')
        continue
    zip_url  = BINANCE_VISION_BASE + f'BTCUSDT-1m-{ym}.zip'
    zip_path = RAW_DIR / f'BTCUSDT-1m-{ym}.zip'
    try:
        r = requests.get(zip_url, timeout=120, stream=True)
        if r.status_code != 200:
            print(f'  ✗ {ym}  HTTP {r.status_code} (not yet on Binance Vision — skipped)')
            continue
        downloaded = 0
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                downloaded += len(chunk)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW_DIR)
        zip_path.unlink()
        print(f'  ✓ {ym}  downloaded  ({downloaded/1e6:.1f} MB)')
    except Exception as e:
        print(f'  ✗ {ym}  {e}')

# Fetch recent Hyperliquid bars to bridge gap to today
print()
print('Fetching recent bars from Hyperliquid (last 7 days)...')
try:
    import time as _time
    url = 'https://api.hyperliquid.xyz/info'
    now_ms   = int(_time.time() * 1000)
    start_ms = now_ms - 7 * 24 * 60 * 60 * 1000
    r = requests.post(url, json={
        'type': 'candleSnapshot',
        'req': {'coin': 'BTC', 'interval': '1m', 'startTime': start_ms, 'endTime': now_ms}
    }, timeout=30)
    bars = r.json()
    if bars:
        rows = []
        for b in bars:
            ts_ms = int(b.get('t', b.get('time', b.get('T', 0))))
            rows.append({'open': float(b.get('o', 0)), 'high': float(b.get('h', 0)),
                         'low': float(b.get('l', 0)), 'close': float(b.get('c', 0)),
                         'volume': float(b.get('v', 0)), 'open_time': ts_ms})
        hl_df = pd.DataFrame(rows)
        hl_df.index = pd.to_datetime(hl_df['open_time'], unit='ms', utc=True)
        hl_df.index.name = 'open_time'
        hl_df = hl_df[['open','high','low','close','volume']]
        hl_df.to_parquet(RAW_DIR / 'BTC_1m_hl.parquet')
        print(f'  ✓ Hyperliquid: {len(hl_df):,} bars')
except Exception as e:
    print(f'  ✗ Hyperliquid: {e}')

# Build combined parquet
print()
print('Building combined dataset...')
dfs = []
for ym in months:
    f = RAW_DIR / f'BTCUSDT-1m-{ym}.csv'
    if not f.exists(): continue
    df = pd.read_csv(f, header=None, names=COLS, dtype=str)
    ts = df['open_time'].astype(np.int64)
    if ts.iloc[0] > 1_000_000_000_000_000:
        ts = ts // 1000
    df.index = pd.to_datetime(ts, unit='ms', utc=True)
    df.index.name = 'open_time'
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    dfs.append(df[['open','high','low','close','volume']])

hl_path = RAW_DIR / 'BTC_1m_hl.parquet'
if hl_path.exists():
    dfs.append(pd.read_parquet(hl_path))

combined = pd.concat(dfs).sort_index()
combined = combined[~combined.index.duplicated(keep='last')].dropna()
combined = combined[(combined['close'] > 0)]
combined.to_parquet(RAW_DIR / 'BTCUSDT_1m_full.parquet')

elapsed = time.time() - t0
print(f'\\n✓ Dataset ready in {elapsed:.1f}s')
print(f'  Months : {len(months)} files')
print(f'  Bars   : {len(combined):,}')
print(f'  Range  : {combined.index[0].date()} → {combined.index[-1].date()}')
print(f'  Price  : ${combined["close"].min():,.0f} → ${combined["close"].max():,.0f}')
print(f'  Size   : {(RAW_DIR / "BTCUSDT_1m_full.parquet").stat().st_size/1e6:.1f} MB')
combined.tail(3)"""))

# ── Step 5: GPU Feature Engineering ──────────────────────────────────────────
cells.append(md("""## Step 5 — GPU Feature Engineering

Builds 171+ features using **CuPy CUDA** for all rolling/element-wise operations.
Falls back to CPU (NumPy + Numba parallel) if no GPU is available.

| Stage | Features | Operations |
|-------|----------|-----------|
| Technical | +92 | SMA, EMA, MACD, RSI, Bollinger, ATR, Stochastic, OBV, VWAP |
| Microstructure | +55 | OFI, VPIN, Amihud, Kyle's λ, Corwin-Schultz, Realised Var |
| StatArb | +19 | Z-score, CUSUM, Hurst, OU half-life, ADF, Kalman |"""))
cells.append(code("""import sys, os, shutil, time
sys.path.insert(0, '/content/AIQuant')
os.chdir('/content/AIQuant')

# Purge pycache before feature engineering to avoid stale bytecode
for p in Path('/content/AIQuant').rglob('__pycache__'):
    shutil.rmtree(p, ignore_errors=True)

import pandas as pd
from aiquant.utils.fast_math import warmup
from aiquant.features import build_full_feature_set

# Pre-compile Numba JIT functions
print('Pre-compiling Numba JIT functions...')
warmup()
print('✓ Numba JIT ready\\n')

# Load raw data
df_raw = pd.read_parquet('/content/AIQuant/data/raw/BTCUSDT_1m_full.parquet')
print(f'Loaded: {len(df_raw):,} bars')

# Build all features
t0 = time.time()
df_feat = build_full_feature_set(df_raw.copy(), verbose=True)
df_feat = df_feat.dropna()
print(f'\\n✓ Features ready: {df_feat.shape[0]:,} bars × {df_feat.shape[1]} features')
print(f'  Total time: {time.time()-t0:.1f}s')
df_feat.tail(3)"""))

# ── Step 6: ML Ensemble Backtest ─────────────────────────────────────────────
cells.append(md("""## Step 6 — ML Ensemble Backtest (XGBoost + LightGBM + LSTM)

Runs the full ML ensemble backtest using **walk-forward cross-validation** (no lookahead bias):
- **XGBoost** — gradient boosted trees, 200 estimators, CUDA if available
- **LightGBM** — leaf-wise gradient boosting, 200 estimators, GPU if available
- **PyTorch LSTM + Attention** — 30-bar sequence model (skip with `FAST_MODE=True`)

Ensemble score = XGB 40% + LGBM 40% + LSTM 20%.
Signals are generated **only on out-of-sample data** (46 walk-forward folds × 7 days each).

**Expected runtime:** ~6 min (fast mode) · ~25 min (with LSTM on CPU) · ~8 min (LSTM on T4 GPU)"""))
cells.append(code("""import subprocess, sys, time, os

fast_flag = ['--fast'] if FAST_MODE else []
cmd = [sys.executable, '-u', 'run.py', 'backtest',
       '--pair', PAIR, '--days', str(DAYS),
       '--capital', str(INITIAL_CAPITAL)] + fast_flag

print(f'Running: {" ".join(cmd)}')
print(f'Fast mode: {FAST_MODE}  (LSTM: {"SKIP" if FAST_MODE else "ENABLED"})')
print()

t0 = time.time()
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'

# Stream output line-by-line so progress appears immediately in Colab
with subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    cwd='/content/AIQuant',
    env=env,
) as proc:
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

elapsed = time.time() - t0
print(f'\\n✓ Backtest completed in {elapsed/60:.1f} min')"""))

# ── Step 7: View chart ────────────────────────────────────────────────────────
cells.append(md("## Step 7 — View Results Chart"))
cells.append(code("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

chart_path = Path('/content/AIQuant/results/backtest_results.png')
if chart_path.exists():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.imshow(mpimg.imread(str(chart_path)))
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    print(f'Chart: {chart_path}')
    print(f'Size : {chart_path.stat().st_size/1e3:.0f} KB')
else:
    print('No chart found — run Step 6 first')"""))

# ── Step 8: Live trading ──────────────────────────────────────────────────────
cells.append(md("""## Step 8 — Live Trading on Hyperliquid Mainnet (ML Mode)

Requires `HYPERLIQUID_PRIVATE_KEY` set in Step 3 **and** Step 6 completed (backtest must have run to save the model bundle).

The ML live trading loop:
1. Loads the trained model bundle (`models/ml_live_bundle.pkl`) saved by Step 6
2. Every 60 seconds: fetches the latest 600 bars from Hyperliquid
3. Builds all 183 features on the live data
4. Runs XGBoost + LightGBM + LSTM → ensemble confidence score
5. Applies the optimised thresholds → LONG / SHORT / FLAT signal
6. Sizes position with half-Kelly (max 25% of portfolio per trade)
7. Executes on Hyperliquid mainnet

**To generate a new wallet:**
```python
from eth_account import Account
a = Account.create()
print(a.key.hex())   # private key — fund this address on Hyperliquid
```

> Press the **stop button (■)** in Colab to halt the trading loop gracefully."""))
cells.append(code("""import subprocess, sys, os, time

if not HYPERLIQUID_PRIVATE_KEY:
    print('⚠  No HYPERLIQUID_PRIVATE_KEY set in Step 3.')
    print('   Set your key and re-run Step 3, then run this cell.')
    print()
    print('   To generate a new wallet:')
    print('   from eth_account import Account; a = Account.create(); print(a.key.hex())')
else:
    from pathlib import Path
    bundle = Path('/content/AIQuant/models/ml_live_bundle.pkl')
    if not bundle.exists():
        print('⚠  Model bundle not found. Run Step 6 (backtest) first to train and save the model.')
    else:
        import joblib
        b = joblib.load(bundle)
        print(f'Model bundle loaded:')
        print(f'  Trained at    : {b.get("trained_at", "unknown")}')
        print(f'  Pair          : {b.get("pair", "unknown")}')
        print(f'  Long threshold: {b.get("long_thresh")}')
        print(f'  Short threshold:{b.get("short_thresh")}')
        print(f'  LSTM available: {"yes" if b.get("lstm_state") is not None else "no"}')
        print()
        print('Starting ML live trading on Hyperliquid mainnet...')
        print('Press the stop button (■) in Colab to halt the trading loop.')
        print()

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['HYPERLIQUID_PRIVATE_KEY'] = HYPERLIQUID_PRIVATE_KEY

        # Stream output line-by-line so every tick prints immediately
        with subprocess.Popen(
            [sys.executable, '-u', 'run.py', 'live', '--ml',
             '--pair', PAIR, '--capital', str(INITIAL_CAPITAL)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd='/content/AIQuant',
            env=env,
        ) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
            proc.wait()"""))

# ── Step 9: Download results ──────────────────────────────────────────────────
cells.append(md("## Step 9 — Download Results"))
cells.append(code("""from google.colab import files
from pathlib import Path
import glob

# Download chart
chart = Path('/content/AIQuant/results/backtest_results.png')
if chart.exists():
    files.download(str(chart))
    print(f'✓ Downloaded: {chart.name}')

# Download best params JSON
params = Path('/content/AIQuant/config/ml_best_params.json')
if params.exists():
    files.download(str(params))
    print(f'✓ Downloaded: {params.name}')

# Download model bundle (for use outside Colab)
bundle = Path('/content/AIQuant/models/ml_live_bundle.pkl')
if bundle.exists():
    files.download(str(bundle))
    print(f'✓ Downloaded: {bundle.name}')

# Download all trade logs (backtest + live)
for log_file in glob.glob('/content/AIQuant/logs/**/*.json', recursive=True):
    files.download(log_file)
    print(f'✓ Downloaded: {Path(log_file).name}')

print('\\n✓ Download complete')"""))

# ── Build notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "AIQuant_Colab.ipynb"
        }
    },
    "cells": cells
}

NB_PATH.write_text(json.dumps(nb, indent=2))
print(f"✓ Notebook written: {NB_PATH}")
print(f"  Cells: {len(cells)}")
print(f"  Size : {NB_PATH.stat().st_size/1e3:.0f} KB")
print(f"\n  Open in Colab:")
print(f"  https://colab.research.google.com/github/AegisFintech/AIQuant/blob/main/AIQuant_Colab.ipynb")
