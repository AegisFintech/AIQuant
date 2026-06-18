"""
scripts/smoke_test.py
=====================
Fast end-to-end smoke test for the AIQuant pipeline.
Run this before every commit/push to catch regressions.

Usage:
    python3 scripts/smoke_test.py

Exits with code 0 on success, 1 on any failure.
"""
import sys, os, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m·\033[0m"

errors = []

def check(name, fn):
    t0 = time.time()
    try:
        fn()
        print(f"  {PASS}  {name}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  {FAIL}  {name}  ({time.time()-t0:.1f}s)")
        print(f"       {e}")
        errors.append((name, str(e)))

print("\n" + "="*60)
print("  AIQuant Smoke Test")
print("="*60 + "\n")

# ── Test 1: Imports ──────────────────────────────────────────
def test_imports():
    import aiquant.features
    import aiquant.features.technical
    import aiquant.features.microstructure
    import aiquant.features.statarb
    import run
check("Imports", test_imports)

# ── Test 2: Feature engineering — no duplicate columns ───────
def test_no_duplicate_cols():
    import pandas as pd, numpy as np
    from aiquant.features import build_full_feature_set
    parquet = 'data/raw/BTCUSDT_1m_full.parquet'
    if not os.path.exists(parquet):
        raise FileNotFoundError(f"Missing {parquet} — run prepare_data.py first")
    df_raw = pd.read_parquet(parquet).tail(3000)
    df = build_full_feature_set(df_raw, verbose=False)
    dups = df.columns[df.columns.duplicated()].tolist()
    if dups:
        raise ValueError(f"Duplicate columns: {dups}")
check("Feature engineering — no duplicate columns", test_no_duplicate_cols)

# ── Test 3: float32 cast works ───────────────────────────────
def test_float32_cast():
    import pandas as pd, numpy as np
    from aiquant.features import build_full_feature_set
    parquet = 'data/raw/BTCUSDT_1m_full.parquet'
    df_raw = pd.read_parquet(parquet).tail(3000)
    df = build_full_feature_set(df_raw, verbose=False)
    float_cols = df.select_dtypes(include=[np.float64]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
check("float32 cast on feature DataFrame", test_float32_cast)

# ── Test 4: Full backtest — 60-day fast mode ─────────────────
def test_backtest_60d():
    result = subprocess.run(
        [sys.executable, 'run.py', 'backtest', '--days', '60', '--fast'],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        # Show last 30 lines of stderr/stdout for diagnosis
        out = (result.stdout + result.stderr).strip().split('\n')
        raise RuntimeError('\n'.join(out[-30:]))
    # Check key success markers in combined output
    combined = result.stdout + result.stderr
    if 'sharpe' not in combined.lower():
        out = combined.strip().split('\n')
        raise RuntimeError("Backtest completed but no Sharpe ratio in output.\n" + '\n'.join(out[-20:]))
check("Full backtest — 60-day fast mode (no LSTM)", test_backtest_60d)

# ── Test 5: Syntax check all Python files ────────────────────
def test_syntax():
    import glob
    files = glob.glob('**/*.py', recursive=True)
    files = [f for f in files if '.git' not in f and 'node_modules' not in f]
    for f in files:
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', f],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise SyntaxError(f"{f}: {result.stderr.strip()}")
check(f"Syntax check all .py files", test_syntax)

# ── Test 6: ml_live_trader imports cleanly ───────────────────
def test_live_trader_import():
    from aiquant.execution.ml_live_trader import MLLiveTrader
check("MLLiveTrader import", test_live_trader_import)

# ── Summary ──────────────────────────────────────────────────
print()
if errors:
    print(f"  {FAIL}  {len(errors)} test(s) FAILED:\n")
    for name, msg in errors:
        print(f"    - {name}")
        print(f"      {msg[:200]}")
    print()
    sys.exit(1)
else:
    print(f"  {PASS}  All tests passed — safe to push.\n")
    sys.exit(0)
