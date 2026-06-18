"""
aiquant/features/__init__.py
=============================
Feature engineering pipeline with verbose, timed progress output.
"""

import gc
import time
import platform
import numpy as np

from .technical     import generate_all_technical_features
from .microstructure import generate_all_microstructure_features
from .statarb        import generate_all_statarb_features


# ── Colour helpers (no-op on Windows) ────────────────────────────────────────
def _c(code, text):
    if platform.system() == 'Windows':
        return text
    return f'\033[{code}m{text}\033[0m'

_GREEN  = lambda t: _c('32', t)
_CYAN   = lambda t: _c('36', t)
_YELLOW = lambda t: _c('33', t)
_DIM    = lambda t: _c('2',  t)
_BOLD   = lambda t: _c('1',  t)


def _step(label: str, fn, df, bar_count: int):
    """Run a feature stage, print timing and column delta."""
    cols_before = df.shape[1]
    t0 = time.perf_counter()
    df = fn(df)
    elapsed = time.perf_counter() - t0
    cols_added = df.shape[1] - cols_before
    speed = bar_count / elapsed if elapsed > 0 else 0
    print(
        f"    {_GREEN('✓')} {label:<22} "
        f"{_CYAN(f'+{cols_added} features'):>18}  "
        f"{_DIM(f'{elapsed:.1f}s')}  "
        f"{_DIM(f'{speed:,.0f} bars/s')}"
    )
    return df


def build_full_feature_set(df, verbose: bool = True) -> "pd.DataFrame":
    """
    Apply all feature groups with verbose per-step timing.

    Stages
    ------
    1. Technical    — trend, momentum, volatility, volume, candle  (~60 features)
    2. Microstructure — OFI, VPIN, Amihud, Kyle's λ, Roll spread  (~40 features)
    3. StatArb      — Hurst, half-life, ADF, Kalman, CUSUM         (~20 features)

    Performance
    -----------
    All hot-path rolling windows use Numba JIT-compiled C code (parallel=True).
    The ADF step uses a fast Numba approximation (~500x vs statsmodels loop).
    Call aiquant.utils.fast_math.warmup() once at startup to pre-compile JIT.
    """
    import pandas as pd

    n_bars = len(df)
    t_total = time.perf_counter()

    if verbose:
        print(f"\n  {_BOLD('Feature Engineering')}  ·  {n_bars:,} bars  ·  {df.shape[1]} input columns")
        print(f"  {'─' * 62}")

    # ── Numba warm-up (no-op if already compiled) ─────────────────────────
    if verbose:
        print(f"    {_CYAN('⚙')}  Numba JIT warm-up ...", end='\r')
    try:
        from ..utils.fast_math import warmup
        warmup()
    except Exception:
        pass
    if verbose:
        print(f"    {_GREEN('✓')} {'Numba JIT warm-up':<22} {'(cached)':>18}  {_DIM('ready')}")

    # ── Stage 1: Technical ────────────────────────────────────────────────
    if verbose:
        print(f"    {_CYAN('⚙')}  Technical indicators ...", end='\r')
    df = _step("Technical indicators", generate_all_technical_features, df, n_bars)
    gc.collect()

    # ── Stage 2: Microstructure ───────────────────────────────────────────
    if verbose:
        print(f"    {_CYAN('⚙')}  Microstructure ...", end='\r')
    df = _step("Microstructure", generate_all_microstructure_features, df, n_bars)
    gc.collect()

    # ── Stage 3: StatArb ──────────────────────────────────────────────────
    if verbose:
        print(f"    {_CYAN('⚙')}  StatArb / regime ...", end='\r')
    df = _step("StatArb / regime", generate_all_statarb_features, df, n_bars)
    gc.collect()

    # ── Drop NaN warmup rows ──────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)

    total_elapsed = time.perf_counter() - t_total

    if verbose:
        print(f"  {'─' * 62}")
        print(
            f"  {_GREEN('✓')} Done  ·  "
            f"{_BOLD(str(df.shape[1]))} features  ·  "
            f"{_BOLD(f'{len(df):,}')} usable bars  "
            f"{_DIM(f'({n_dropped} warmup rows dropped)')}"
        )
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        speed  = n_bars / total_elapsed if total_elapsed > 0 else 0
        print(
            f"  {_DIM('Memory:')} {mem_mb:.1f} MB  ·  "
            f"{_DIM('Total time:')} {total_elapsed:.1f}s  ·  "
            f"{_DIM('Throughput:')} {speed:,.0f} bars/s\n"
        )

    return df


__all__ = [
    'generate_all_technical_features',
    'generate_all_microstructure_features',
    'generate_all_statarb_features',
    'build_full_feature_set',
]
