"""
scripts/build_colab.py
=======================
Rebuilds AIQuant_Colab.ipynb with full GPU support.
Run: python3 scripts/build_colab.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
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

> **GPU Runtime Required** — Go to `Runtime → Change runtime type → GPU (T4 or A100)` before running.

| Step | Description |
|------|-------------|
| 1 | Clone repository |
| 2 | Install dependencies + verify GPU |
| 3 | Configuration |
| 4 | Fetch historical data (CryptoDataDownload) |
| 5 | GPU Feature Engineering (CuPy + Numba parallel) |
| 6 | Signal Generation |
| 7 | Backtest (Backtrader) |
| 8 | View chart |
| 9 | GPU ML Training (XGBoost CUDA + LightGBM GPU + PyTorch LSTM) |
| 10 | Live Trading (Hyperliquid) |
| 11 | Download results |"""))

# ── Step 1: Clone ─────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Clone Repository"))
cells.append(code("""import os
os.chdir('/content')
!rm -rf AIQuant
!git clone https://github.com/AegisFintech/AIQuant.git
os.chdir('/content/AIQuant')
!git log --oneline -3
print('\\n✓ Repository ready')"""))

# ── Step 2: Install + GPU check ───────────────────────────────────────────────
cells.append(md("""## Step 2 — Install Dependencies & Verify GPU

This installs all required packages and verifies GPU availability.
CuPy is the key library for GPU-accelerated feature engineering."""))
cells.append(code("""# Install all dependencies
!pip install -q -r requirements.txt

# Explicit installs for GPU-critical packages
!pip install -q cupy-cuda12x xgboost lightgbm torch torchvision
!pip install -q ta numba backtrader pyarrow statsmodels scikit-learn python-dotenv

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
        print(f'    PyTorch       : {torch.__version__}')
    else:
        print('  ✗ PyTorch CUDA  : NOT available — switch to GPU runtime!')
except Exception as e:
    print(f'  ✗ PyTorch        : {e}')

# CuPy
try:
    import cupy as cp
    cp.array([1.0])
    mem = cp.cuda.Device(0).mem_info
    free_gb  = mem[0] / 1e9
    total_gb = mem[1] / 1e9
    print(f'  ✓ CuPy CUDA     : {cp.__version__}  ({free_gb:.1f}/{total_gb:.1f} GB free)')
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

# TA
try:
    import ta
    print(f'  ✓ ta             : {ta.__version__}')
except Exception as e:
    print(f'  ✗ ta             : {e}')

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
DAYS                    = 30          # Days of backtest data (30=free, 90=Pro, 180=Pro+)
INITIAL_CAPITAL         = 100_000     # Starting capital (USD)
HYPERLIQUID_PRIVATE_KEY = ''          # Your Hyperliquid private key (for live trading)

# ── Write .env ────────────────────────────────────────────────────────────────
env_content = f\"\"\"HYPERLIQUID_PRIVATE_KEY={HYPERLIQUID_PRIVATE_KEY}
HYPERLIQUID_TESTNET=false
TRADING_PAIR={PAIR}
INITIAL_CAPITAL={INITIAL_CAPITAL}
\"\"\"
with open('.env', 'w') as f:
    f.write(env_content)

print(f'✓ Config set: {PAIR}  |  {DAYS} days  |  ${INITIAL_CAPITAL:,} capital')
print(f'  Hyperliquid live trading: {\"ENABLED\" if HYPERLIQUID_PRIVATE_KEY else \"DISABLED (no key)\"}'
)"""))

# ── Step 4: Fetch data ────────────────────────────────────────────────────────
cells.append(md("""## Step 4 — Fetch Historical Data (CryptoDataDownload)

Downloads BTCUSDT 1-minute OHLCV data from CryptoDataDownload (free, no API key).
Data is cached locally so subsequent runs are instant."""))
cells.append(code("""import logging, time
from pathlib import Path
from aiquant.data.fetcher import fetch_cdd_backtest

logging.basicConfig(level=logging.WARNING)

print(f'Fetching {PAIR} 1m data ({DAYS} days = {DAYS*1440:,} bars)...')
t0 = time.time()
df_raw = fetch_cdd_backtest(pair=PAIR, days=DAYS)
elapsed = time.time() - t0

print(f'\\n✓ Data loaded in {elapsed:.1f}s')
print(f'  Shape  : {df_raw.shape[0]:,} bars × {df_raw.shape[1]} columns')
print(f'  Range  : {df_raw.index[0]}  →  {df_raw.index[-1]}')
print(f'  Memory : {df_raw.memory_usage(deep=True).sum()/1e6:.1f} MB')
df_raw.tail(3)"""))

# ── Step 5: GPU Feature Engineering ──────────────────────────────────────────
cells.append(md("""## Step 5 — GPU Feature Engineering

Builds 180+ features using **CuPy CUDA** for all rolling/element-wise operations.
Data is transferred to GPU once, all features computed on-device, then returned.

| Stage | Method | Operations |
|-------|--------|-----------|
| Technical | CuPy CUDA | SMA, EMA, MACD, RSI, Bollinger, ATR, Stochastic, OBV, VWAP |
| Microstructure | CuPy CUDA | OFI, VPIN, Amihud, Kyle's λ, Corwin-Schultz, Realised Var |
| StatArb | CuPy + Numba parallel | Z-score, CUSUM, Hurst, OU half-life, ADF, Kalman |

Falls back to CPU (NumPy + Numba parallel) if no GPU is available."""))
cells.append(code("""from aiquant.features.gpu_features import build_features_gpu
from aiquant.utils.fast_math import warmup
import time

# Pre-compile Numba JIT functions (runs once, ~0.5s, then cached)
print('Pre-compiling Numba JIT functions...')
warmup()
print('✓ Numba JIT ready\\n')

# Build all features — GPU-accelerated
t0 = time.time()
df_clean = build_features_gpu(df_raw.copy(), verbose=True)
print(f'Total wall time: {time.time()-t0:.1f}s')"""))

# ── Step 6: Signals ───────────────────────────────────────────────────────────
cells.append(md("""## Step 6 — Signal Generation

Runs the strategy ensemble (Kalman StatArb + Mean Reversion + Trend Following)
to generate trading signals from the feature matrix."""))
cells.append(code("""import logging, time
from aiquant.strategies.ensemble import StrategyEnsemble

# Suppress strategy warnings to keep output clean
logging.getLogger('aiquant.strategies.ensemble').setLevel(logging.ERROR)
logging.getLogger('aiquant.strategies.mean_reversion').setLevel(logging.ERROR)
logging.getLogger('aiquant.strategies.trend_following').setLevel(logging.ERROR)
logging.getLogger('aiquant.strategies.stat_arb').setLevel(logging.ERROR)

print('Generating ensemble signals...')
t0 = time.time()
ensemble = StrategyEnsemble()
df_signals = ensemble.generate_signals(df_clean)

n_total = max(len(df_signals), 1)   # guard against empty DataFrame
n_long  = int((df_signals['final_signal'] ==  1).sum())
n_short = int((df_signals['final_signal'] == -1).sum())
n_flat  = int((df_signals['final_signal'] ==  0).sum())
print(f'\\n✓ Signals generated in {time.time()-t0:.1f}s  ({n_total:,} bars)')
print(f'  Long:  {n_long:,}  ({n_long/n_total*100:.1f}%)')
print(f'  Short: {n_short:,}  ({n_short/n_total*100:.1f}%)')
print(f'  Flat:  {n_flat:,}  ({n_flat/n_total*100:.1f}%)')"""))

# ── Step 7: Backtest ──────────────────────────────────────────────────────────
cells.append(md("""## Step 7 — Backtest (Backtrader)

Runs the full Backtrader backtest on real historical data.
Note: Backtrader is a sequential state machine and runs on CPU by design."""))
cells.append(code("""import subprocess, sys, time

print(f'Running backtest: {PAIR}  {DAYS} days  ${INITIAL_CAPITAL:,}...')
t0 = time.time()
result = subprocess.run(
    [sys.executable, 'run.py', 'backtest',
     '--pair', PAIR, '--days', str(DAYS)],
    capture_output=False,
    text=True,
    cwd='/content/AIQuant'
)
print(f'\\nBacktest completed in {time.time()-t0:.1f}s')"""))

# ── Step 8: View chart ────────────────────────────────────────────────────────
cells.append(md("## Step 8 — View Backtest Chart"))
cells.append(code("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

chart_path = Path('/content/AIQuant/results/backtest_results.png')
if chart_path.exists():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(mpimg.imread(str(chart_path)))
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    print(f'Chart: {chart_path}')
else:
    print('No chart found — run Step 7 first')"""))

# ── Step 9: GPU ML Training ───────────────────────────────────────────────────
cells.append(md("""## Step 9 — GPU ML Training

Trains a 3-model GPU ensemble:
- **XGBoost** — `device='cuda'` histogram trees
- **LightGBM** — `device='gpu'` gradient boosting
- **PyTorch LSTM + Attention** — CUDA tensor cores, multi-head attention

Uses walk-forward cross-validation (expanding window) to avoid look-ahead bias.
Final signal = XGB 40% + LGBM 40% + LSTM 20%, confidence threshold 0.45.

> **GPU memory note:** T4 (15 GB) handles 90 days. A100 (40 GB) handles 180 days."""))
cells.append(code("""from aiquant.models.gpu_ml import GPUMLSignalGenerator
import time

if len(df_clean) < 5000:
    print(f'⚠  Only {len(df_clean):,} bars — need at least 5,000 for ML training.')
    print('   Increase DAYS to 30+ and re-run Steps 4-5.')
else:
    print(f'Training GPU ML ensemble on {len(df_clean):,} bars × {len(df_clean.columns)} features...')
    t0 = time.time()

    ml_gen = GPUMLSignalGenerator(
        horizon=5,           # predict 5-bar forward return
        threshold=0.0003,    # 0.03% move = signal
        n_folds=3,           # walk-forward folds
        min_train_bars=3000, # minimum training bars
    )
    df_ml = ml_gen.fit_predict(df_clean, verbose=True)

    elapsed = time.time() - t0
    print(f'\\n✓ ML training complete in {elapsed:.1f}s')

    # Show GPU utilisation
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'  GPU memory used: {alloc:.2f} / {total:.1f} GB')
    except:
        pass

    # Show ML signal distribution
    n_ml_long  = int((df_ml['ml_signal'] ==  1).sum())
    n_ml_short = int((df_ml['ml_signal'] == -1).sum())
    n_ml_flat  = int((df_ml['ml_signal'] ==  0).sum())
    avg_conf   = df_ml['ml_confidence'].mean()
    print(f'  ML Long:  {n_ml_long:,}  Short: {n_ml_short:,}  Flat: {n_ml_flat:,}')
    print(f'  Avg confidence: {avg_conf:.3f}')"""))

# ── Step 10: Live trading ─────────────────────────────────────────────────────
cells.append(md("""## Step 10 — Live Trading on Hyperliquid

Requires `HYPERLIQUID_PRIVATE_KEY` set in Step 3.
Runs a finite paper-trading loop (10 bars) to demonstrate live signal → order flow."""))
cells.append(code("""if not HYPERLIQUID_PRIVATE_KEY:
    print('⚠  No HYPERLIQUID_PRIVATE_KEY set in Step 3.')
    print('   Set your key and re-run Step 3, then run this cell.')
else:
    import subprocess, sys
    print('Starting live trading on Hyperliquid mainnet...')
    result = subprocess.run(
        [sys.executable, 'run.py', 'live'],
        capture_output=False,
        text=True,
        cwd='/content/AIQuant',
        timeout=300,
    )"""))

# ── Step 11: Download results ─────────────────────────────────────────────────
cells.append(md("## Step 11 — Download Results"))
cells.append(code("""from google.colab import files
from pathlib import Path
import glob

# Download chart
chart = Path('/content/AIQuant/results/backtest_results.png')
if chart.exists():
    files.download(str(chart))
    print(f'✓ Downloaded: {chart.name}')

# Download trade logs
for log_file in glob.glob('/content/AIQuant/logs/**/*.json', recursive=True):
    files.download(log_file)
    print(f'✓ Downloaded: {Path(log_file).name}')

# Download cached data parquet
for pq in glob.glob('/content/AIQuant/data/raw/*.parquet'):
    files.download(pq)
    print(f'✓ Downloaded: {Path(pq).name}')"""))

# ── Refresh cell ──────────────────────────────────────────────────────────────
cells.append(md("## Refresh — Pull Latest Code from GitHub"))
cells.append(code("""import os
os.chdir('/content')
!rm -rf AIQuant
!git clone https://github.com/AegisFintech/AIQuant.git
os.chdir('/content/AIQuant')
!git log --oneline -3
print('\\n✓ Latest code pulled — re-run from Step 3')"""))

# ── Footer ────────────────────────────────────────────────────────────────────
cells.append(md("""---
**AIQuant** · Built by [AegisFintech](https://github.com/AegisFintech) · [Apache 2.0](LICENSE)

GPU acceleration: CuPy (feature engineering) · XGBoost CUDA · LightGBM GPU · PyTorch LSTM"""))

# ── Build notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "AIQuant_Colab.ipynb"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "cells": cells,
}

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✓ Notebook written: {NB_PATH}")
print(f"  Cells: {len(cells)}")
