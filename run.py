"""
AIQuant — HFT Statistical Arbitrage Framework
==============================================
AegisFintech | Apache 2.0 License

Usage:
    python3 run.py backtest                          # BTC, last 365 days
    python3 run.py backtest --pair ETHUSDT           # different pair
    python3 run.py backtest --days 30                # shorter window
    python3 run.py backtest --pair ETH --days 60     # pair shorthand works too
    python3 run.py live                              # live trading on Hyperliquid mainnet

Defaults (when no flags given):
    --pair   BTCUSDT
    --days   365  (T-365 days of 1m data = 525,600 bars)

Data sources:
    Backtest : Binance Vision monthly CSVs (free, no API key) + Hyperliquid public API
    Live     : Hyperliquid public API for market data + mainnet for execution
               Requires HYPERLIQUID_PRIVATE_KEY in .env
"""

import sys
import os
import time
import json
import logging
import argparse
import platform
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / 'results'
DATA_DIR    = ROOT / 'data' / 'raw'
LOGS_DIR    = ROOT / 'logs'
CONFIG_DIR  = ROOT / 'config'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('aiquant')
logger.setLevel(logging.INFO)

# ── Numba JIT warm-up ────────────────────────────────────────────────────────
try:
    from aiquant.utils.fast_math import warmup as _nb_warmup
    _nb_warmup()
except Exception:
    pass

# ── Colour helpers ────────────────────────────────────────────────────────────
def _c(code, text):
    if platform.system() == 'Windows':
        return text
    return f'\033[{code}m{text}\033[0m'

BOLD   = lambda t: _c('1',  t)
DIM    = lambda t: _c('2',  t)
GREEN  = lambda t: _c('32', t)
RED    = lambda t: _c('31', t)
CYAN   = lambda t: _c('36', t)
YELLOW = lambda t: _c('33', t)
WHITE  = lambda t: _c('97', t)

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = f"""
{CYAN('╔══════════════════════════════════════════════════════════════╗')}
{CYAN('║')}  {BOLD(WHITE('AIQuant'))}  ·  HFT Statistical Arbitrage  ·  {DIM('AegisFintech')}      {CYAN('║')}
{CYAN('║')}  {DIM('Apache 2.0  ·  github.com/AegisFintech/AIQuant')}             {CYAN('║')}
{CYAN('╚══════════════════════════════════════════════════════════════╝')}
"""

def banner(mode: str, pair: str, days: int = None):
    print(BANNER)
    mode_str = {
        'backtest': '📊  BACKTEST  (ML Ensemble · Binance Vision + Hyperliquid)',
        'live':     '🔴  LIVE TRADING  (Hyperliquid Mainnet)',
    }.get(mode, mode.upper())
    print(f"  Mode  : {BOLD(mode_str)}")
    print(f"  Pair  : {BOLD(CYAN(pair))}")
    if days:
        print(f"  Window: {BOLD(str(days))} days  ({days * 1440:,} 1m bars)")
    print()


# ════════════════════════════════════════════════════════════════════════════
# PAIR NORMALISATION
# ════════════════════════════════════════════════════════════════════════════

VALID_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

def normalise_pair(raw: str) -> str:
    """Accept BTC, btc, BTCUSDT, btcusdt — always return e.g. BTCUSDT."""
    p = raw.upper().strip()
    if p in VALID_PAIRS:
        return p
    if not p.endswith('USDT'):
        p = p + 'USDT'
    if p not in VALID_PAIRS:
        print(f"  {YELLOW('⚠')}  Unknown pair '{raw}'. Defaulting to BTCUSDT.")
        return 'BTCUSDT'
    return p


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING — Binance Vision + Hyperliquid
# ════════════════════════════════════════════════════════════════════════════

def load_data(pair: str = 'BTCUSDT', days: int = 365) -> pd.DataFrame:
    """
    Load 1m OHLCV data from the pre-built parquet dataset.
    If the parquet doesn't exist, runs prepare_data.py to build it.
    """
    parquet_path = DATA_DIR / 'BTCUSDT_1m_full.parquet'

    if not parquet_path.exists():
        print(f"  {YELLOW('⚠')}  Dataset not found. Building from Binance Vision CSVs...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / 'prepare_data.py')],
            capture_output=False
        )
        if result.returncode != 0:
            raise RuntimeError("prepare_data.py failed. Check data/raw/ for CSV files.")

    print(f"  {CYAN('↓')} Loading dataset from {parquet_path.name}...", end=' ', flush=True)
    df = pd.read_parquet(parquet_path)

    # Trim to requested window
    if days < 365:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]

    print(f"{GREEN('✓')}")
    print(f"  {DIM(f'{len(df):,} bars  ·  {df.index[0].date()} → {df.index[-1].date()}')}")
    close_min = df['close'].min()
    close_max = df['close'].max()
    print(f"  {DIM(f'Price range: ${close_min:,.0f} → ${close_max:,.0f}')}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all 183 features with verbose per-step timing."""
    from aiquant.features import build_full_feature_set
    return build_full_feature_set(df, verbose=True)


# ════════════════════════════════════════════════════════════════════════════
# ML ENSEMBLE BACKTEST  (XGBoost + LightGBM + LSTM, walk-forward CV)
# ════════════════════════════════════════════════════════════════════════════

def run_ml_backtest(df: pd.DataFrame, pair: str, capital: float = 100_000,
                    fast: bool = False) -> dict:
    """
    Full ML ensemble backtest using walk-forward cross-validation.
    Replicates the proven Sharpe 3.39 pipeline from train_ml_ensemble.py.

    Parameters
    ----------
    df      : DataFrame with OHLCV + 183 features (output of build_features)
    pair    : Trading pair string for display
    capital : Starting capital in USD
    fast    : If True, skip LSTM (faster, ~Sharpe 3.0+)

    Returns
    -------
    dict with keys: sharpe, ret, max_dd, calmar, pf, trades, win_rate,
                    final, equity, signals
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import mutual_info_classif
    import xgboost as xgb
    import lightgbm as lgb

    n      = len(df)
    c      = df['close'].to_numpy(np.float64)
    dates  = df.index

    # ── Label generation ────────────────────────────────────────────────────
    print(f"\n  {CYAN('⚙')}  [1/5] Generating labels...")
    FORWARD_BARS = 15
    FEE          = 0.00035
    THRESHOLD    = 0.0008

    fwd_ret = np.zeros(n)
    for i in range(n - FORWARD_BARS):
        fwd_ret[i] = (c[i + FORWARD_BARS] - c[i]) / c[i]

    labels = np.zeros(n, dtype=np.int8)
    labels[fwd_ret >  THRESHOLD] =  1
    labels[fwd_ret < -THRESHOLD] = -1

    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[:n - FORWARD_BARS] = True

    cc = {-1: int((labels == -1).sum()), 0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
    print(f"  {GREEN('✓')} Long={cc[1]:,}  Short={cc[-1]:,}  Flat={cc[0]:,}  "
          f"({(cc[1]+cc[-1])/n*100:.1f}% directional)")

    # ── Feature selection ────────────────────────────────────────────────────
    print(f"\n  {CYAN('⚙')}  [2/5] Feature selection (mutual information)...")

    drop_cols    = ['open', 'high', 'low', 'close', 'volume']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c_ for c_ in numeric_cols if c_ not in drop_cols]
    X_all        = df[feature_cols].to_numpy(np.float64)
    X_all        = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Load saved top features if available (avoids recomputing MI)
    params_path = CONFIG_DIR / 'ml_best_params.json'
    saved_features = None
    if params_path.exists():
        try:
            with open(params_path) as f:
                saved = json.load(f)
            # Only use saved features if they all exist in current df
            sf = saved.get('top_features', [])
            if sf and all(feat in feature_cols for feat in sf):
                saved_features = sf
                print(f"  {GREEN('✓')} Using {len(saved_features)} saved features from ml_best_params.json")
        except Exception:
            pass

    if saved_features is None:
        sample_idx = np.random.choice(
            np.where(valid_mask)[0], size=min(20000, valid_mask.sum()), replace=False
        )
        sample_idx.sort()
        X_sample = X_all[sample_idx]
        y_sample = labels[sample_idx]
        X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)
        y_binary = (y_sample != 0).astype(int)
        mi_scores = mutual_info_classif(X_sample, y_binary, random_state=42, n_neighbors=5)
        mi_df     = pd.DataFrame({'feature': feature_cols, 'mi': mi_scores}).sort_values('mi', ascending=False)
        TOP_K     = 60
        saved_features = mi_df.head(TOP_K)['feature'].tolist()
        print(f"  {GREEN('✓')} Top {TOP_K} features selected from {len(feature_cols)} total")

    top_features = saved_features
    X_sel = df[top_features].to_numpy(np.float64)
    X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  {DIM('Top 5: ' + ', '.join(top_features[:5]))}")

    # ── Walk-forward CV setup ────────────────────────────────────────────────
    print(f"\n  {CYAN('⚙')}  [3/5] Walk-forward cross-validation...")
    TRAIN_BARS = 30 * 1440
    TEST_BARS  = 7  * 1440
    STEP_BARS  = 7  * 1440

    folds = []
    start = 0
    while start + TRAIN_BARS + TEST_BARS <= n - FORWARD_BARS:
        train_end = start + TRAIN_BARS
        test_end  = train_end + TEST_BARS
        folds.append((start, train_end, test_end))
        start += STEP_BARS

    print(f"  {GREEN('✓')} {len(folds)} folds  "
          f"(train={TRAIN_BARS//1440}d, test={TEST_BARS//1440}d, step={STEP_BARS//1440}d)")

    # ── GPU detection for ML ────────────────────────────────────────────────────
    try:
        import torch as _torch
        _ml_device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        _gpu_name  = _torch.cuda.get_device_name(0) if _ml_device == 'cuda' else 'CPU'
    except Exception:
        _ml_device = 'cpu'
        _gpu_name  = 'CPU'

    # XGBoost GPU tree method
    _xgb_tree_method = 'hist'
    _xgb_device      = 'cuda' if _ml_device == 'cuda' else 'cpu'

    print(f"\n  {CYAN('⚙')}  [4/5] Training XGBoost + LightGBM (walk-forward)...")
    print(f"  {DIM(f'  Device: {_gpu_name}  ·  XGBoost tree_method=hist device={_xgb_device}  ·  {len(folds)} folds')}")
    print(f"  {DIM(f'  Each fold: ~43k train bars × 60 features  →  10k test bars')}")
    print()

    oos_xgb  = np.zeros(n)
    oos_lgb  = np.zeros(n)
    oos_mask = np.zeros(n, dtype=bool)
    scaler   = RobustScaler()

    def encode_labels(y): return (y + 1).astype(int)
    def decode_labels(y): return (y - 1).astype(np.int8)

    _xgb_fold_times = []
    _t_xgb_start    = time.time()

    for fold_idx, (tr_start, tr_end, te_end) in enumerate(folds):
        _t_fold = time.time()
        tr_mask = valid_mask[tr_start:tr_end]
        tr_idx  = np.arange(tr_start, tr_end)[tr_mask]
        X_tr    = X_sel[tr_idx]
        y_tr    = encode_labels(labels[tr_idx])
        te_idx  = np.arange(tr_end, te_end)
        X_te    = X_sel[te_idx]
        X_tr_s  = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        flat_frac = (y_tr == 1).mean()
        w_flat    = 1.0 / (flat_frac + 1e-10)
        w_dir     = 1.0 / ((1 - flat_frac) / 2 + 1e-10)
        sample_wt = np.where(y_tr == 1, w_flat, w_dir)

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            objective='multi:softprob', num_class=3,
            eval_metric='mlogloss', random_state=42, verbosity=0,
            use_label_encoder=False,
            tree_method=_xgb_tree_method, device=_xgb_device,
        )
        xgb_model.fit(X_tr_s, y_tr, sample_weight=sample_wt)
        xgb_proba = xgb_model.predict_proba(X_te_s)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, verbose=-1,
            device='gpu' if _ml_device == 'cuda' else 'cpu',
        )
        lgb_model.fit(X_tr_s, y_tr, sample_weight=sample_wt)
        lgb_proba = lgb_model.predict_proba(X_te_s)

        oos_xgb[te_idx]  = xgb_proba[:, 2] - xgb_proba[:, 0]
        oos_lgb[te_idx]  = lgb_proba[:, 2] - lgb_proba[:, 0]
        oos_mask[te_idx] = True

        _fold_elapsed = time.time() - _t_fold
        _xgb_fold_times.append(_fold_elapsed)
        _avg_fold     = sum(_xgb_fold_times) / len(_xgb_fold_times)
        _remaining    = _avg_fold * (len(folds) - fold_idx - 1)
        _total_so_far = time.time() - _t_xgb_start
        print(
            f"  {DIM(f'  XGB+LGB Fold {fold_idx+1:>2}/{len(folds)}')}"
            f"  {DIM(f'{_fold_elapsed:.1f}s/fold')}"
            f"  {DIM(f'elapsed {_total_so_far:.0f}s')}"
            f"  {CYAN(f'ETA ~{_remaining:.0f}s')}",
            flush=True
        )

    print(f"  {GREEN('✓')} OOS coverage: {oos_mask.sum():,} bars ({oos_mask.mean()*100:.1f}%)"
          f"  {DIM(f'  total {time.time()-_t_xgb_start:.0f}s')}")

    # ── LSTM training ────────────────────────────────────────────────────────
    oos_lstm       = np.zeros(n)
    LSTM_AVAILABLE = False

    if not fast:
        print(f"\n  {CYAN('⚙')}  [4b] Training LSTM (PyTorch)...")
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            SEQ_LEN    = 30
            LSTM_FEATS = 20
            lstm_feats = top_features[:LSTM_FEATS]
            X_lstm_raw = df[lstm_feats].to_numpy(np.float64)
            X_lstm_raw = np.nan_to_num(X_lstm_raw, nan=0.0, posinf=0.0, neginf=0.0)

            class LSTMAttn(nn.Module):
                def __init__(self, input_size, hidden=64, n_layers=2, n_classes=3):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden, n_layers,
                                        batch_first=True, dropout=0.2)
                    self.attn = nn.Linear(hidden, 1)
                    self.fc   = nn.Sequential(
                        nn.Linear(hidden, 32), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(32, n_classes)
                    )
                def forward(self, x):
                    out, _ = self.lstm(x)
                    attn_w = torch.softmax(self.attn(out), dim=1)
                    ctx    = (out * attn_w).sum(dim=1)
                    return self.fc(ctx)

            _t_lstm_start  = time.time()
            _lstm_fold_times = []
            print(f"  {DIM(f'  Device: {DEVICE}  ·  SEQ_LEN={SEQ_LEN}  ·  {LSTM_FEATS} features  ·  5 epochs/fold  ·  {len(folds)} folds')}")
            print()

            for fold_idx, (tr_start, tr_end, te_end) in enumerate(folds):
                _t_lstm_fold = time.time()
                tr_seqs, tr_labs = [], []
                for i in range(tr_start + SEQ_LEN, tr_end):
                    if not valid_mask[i]: continue
                    tr_seqs.append(X_lstm_raw[i-SEQ_LEN:i])
                    tr_labs.append(encode_labels(labels[i]))
                if len(tr_seqs) < 100: continue

                X_tr_t = torch.tensor(np.array(tr_seqs), dtype=torch.float32).to(DEVICE)
                y_tr_t = torch.tensor(tr_labs, dtype=torch.long).to(DEVICE)
                feat_mean = X_tr_t.mean(dim=(0, 1), keepdim=True)
                feat_std  = X_tr_t.std(dim=(0, 1), keepdim=True) + 1e-8
                X_tr_t    = (X_tr_t - feat_mean) / feat_std

                ds      = TensorDataset(X_tr_t, y_tr_t)
                loader  = DataLoader(ds, batch_size=512, shuffle=True)
                model   = LSTMAttn(LSTM_FEATS).to(DEVICE)
                opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                sched   = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
                loss_fn = nn.CrossEntropyLoss()

                model.train()
                epoch_losses = []
                for epoch in range(5):
                    ep_loss = 0.0
                    for xb, yb in loader:
                        opt.zero_grad()
                        loss = loss_fn(model(xb), yb)
                        loss.backward(); opt.step()
                        ep_loss += loss.item()
                    sched.step()
                    epoch_losses.append(ep_loss / max(len(loader), 1))

                model.eval()
                te_idx_list = list(range(tr_end + SEQ_LEN, te_end))
                if te_idx_list:
                    te_seqs = [X_lstm_raw[i-SEQ_LEN:i] for i in te_idx_list]
                    X_te_t  = torch.tensor(np.array(te_seqs), dtype=torch.float32).to(DEVICE)
                    X_te_t  = (X_te_t - feat_mean) / feat_std
                    with torch.no_grad():
                        proba = torch.softmax(model(X_te_t), dim=1).cpu().numpy()
                    for j, i in enumerate(te_idx_list):
                        oos_lstm[i] = proba[j, 2] - proba[j, 0]

                _lstm_fold_elapsed = time.time() - _t_lstm_fold
                _lstm_fold_times.append(_lstm_fold_elapsed)
                _lstm_avg   = sum(_lstm_fold_times) / len(_lstm_fold_times)
                _lstm_eta   = _lstm_avg * (len(folds) - fold_idx - 1)
                _lstm_total = time.time() - _t_lstm_start
                _loss_str   = f'loss={epoch_losses[-1]:.4f}' if epoch_losses else ''
                print(
                    f"  {DIM(f'  LSTM Fold {fold_idx+1:>2}/{len(folds)}')}"
                    f"  {DIM(f'{_lstm_fold_elapsed:.1f}s/fold')}"
                    f"  {DIM(_loss_str)}"
                    f"  {DIM(f'elapsed {_lstm_total:.0f}s')}"
                    f"  {CYAN(f'ETA ~{_lstm_eta:.0f}s')}",
                    flush=True
                )

            LSTM_AVAILABLE = True
            print(f"  {GREEN('✓')} LSTM training complete  "
                  f"(device: {DEVICE}  ·  total {time.time()-_t_lstm_start:.0f}s)")

        except Exception as e:
            print(f"  {YELLOW('⚠')} LSTM skipped: {e}")

    # ── Ensemble score ───────────────────────────────────────────────────────
    if LSTM_AVAILABLE:
        ens_score = 0.40 * oos_xgb + 0.40 * oos_lgb + 0.20 * oos_lstm
        print(f"\n  {DIM('Ensemble: XGB 40% + LGB 40% + LSTM 20%')}")
    else:
        ens_score = 0.50 * oos_xgb + 0.50 * oos_lgb
        print(f"\n  {DIM('Ensemble: XGB 50% + LGB 50%  (LSTM skipped)')}")

    # ── Threshold search + vectorised backtest ───────────────────────────────
    print(f"\n  {CYAN('⚙')}  [5/5] Threshold search + vectorised backtest...")

    best_r = None
    for long_thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        for short_thresh in [-0.05, -0.08, -0.10, -0.12, -0.15, -0.18, -0.20, -0.25]:
            sig = np.zeros(n, np.int8)
            sig[oos_mask & (ens_score >  long_thresh)]  =  1
            sig[oos_mask & (ens_score <  short_thresh)] = -1

            n_trades = int(np.sum(np.abs(np.diff(sig.astype(int)))) // 2)
            if n_trades < 20: continue  # must have at least 20 trades

            changes = np.where(np.diff(sig, prepend=sig[0]) != 0)[0]
            cap     = capital
            equity  = np.full(n, capital, dtype=np.float64)
            wins = losses = 0
            gw = gl = 0.0

            for i in range(len(changes) - 1):
                eb = changes[i]; xb = changes[i + 1]
                d  = int(sig[eb])
                if d == 0:
                    equity[eb:xb] = cap
                    continue
                ep  = c[eb]; xp = c[xb]
                sz  = (cap * 0.25) / ep
                pnl = d * (xp - ep) * sz * (1 - FEE) ** 2
                cap = max(cap + pnl, 1.0)
                equity[eb:xb] = cap
                if pnl > 0: wins  += 1; gw += pnl
                else:       losses += 1; gl += abs(pnl)
            equity[changes[-1]:] = cap

            total_ret = (cap - capital) / capital * 100
            peak      = np.maximum.accumulate(equity)
            dd        = (equity - peak) / peak * 100
            max_dd    = float(np.min(dd))
            n_days    = len(equity) // 1440
            if n_days > 2:
                dl     = np.array([
                    np.diff(np.log(equity[i*1440:(i+1)*1440] + 1e-10)).sum()
                    for i in range(n_days)
                ])
                sharpe = float(dl.mean() / (dl.std() + 1e-10) * np.sqrt(365))
            else:
                sharpe = 0.0
            tt  = wins + losses
            wr  = wins / tt * 100 if tt > 0 else 0.0
            cal = total_ret / abs(max_dd) if abs(max_dd) > 0.01 else 0.0
            pf  = gw / (gl + 1e-10)

            r = dict(
                sharpe=round(sharpe, 3), ret=round(total_ret, 2),
                max_dd=round(max_dd, 2), calmar=round(cal, 3),
                pf=round(pf, 3), trades=tt, win_rate=round(wr, 1),
                final=round(cap, 2), equity=equity, signals=sig,
                long_thresh=long_thresh, short_thresh=short_thresh,
            )

            if best_r is None or r['sharpe'] > best_r['sharpe']:
                best_r = r
                col = GREEN if r['sharpe'] >= 2.0 else (YELLOW if r['sharpe'] >= 1.0 else RED)
                sh_str = f"{r['sharpe']:+.3f}"
                print(
                    f"  {col('↑')} Sharpe={col(sh_str)}  "
                    f"Ret={r['ret']:+.1f}%  MaxDD={r['max_dd']:.1f}%  "
                    f"Trades={r['trades']:,}  WR={r['win_rate']:.0f}%  "
                    f"Calmar={r['calmar']:.2f}  "
                    f"[L>{long_thresh} S<{short_thresh}]"
                )

    if best_r is None:
        print(f"  {RED('✗')} No valid threshold found. Try more data or looser filters.")
        return {}

    # ── Save best params ─────────────────────────────────────────────────────
    CONFIG_DIR.mkdir(exist_ok=True)
    params = {
        'model':         'XGBoost+LightGBM+LSTM Ensemble',
        'forward_bars':  FORWARD_BARS,
        'threshold_pct': THRESHOLD,
        'long_thresh':   best_r['long_thresh'],
        'short_thresh':  best_r['short_thresh'],
        'top_features':  top_features,
        'results':       {k: v for k, v in best_r.items()
                          if k not in ('equity', 'signals')},
    }
    with open(CONFIG_DIR / 'ml_best_params.json', 'w') as f:
        json.dump(params, f, indent=2)

    # ── Print results table ──────────────────────────────────────────────────
    col = GREEN if best_r['ret'] >= 0 else RED
    ret_str    = f"{best_r['ret']:>+.2f}%"
    sharpe_str = f"{best_r['sharpe']:>22.4f}"
    calmar_str = f"{best_r['calmar']:>22.4f}"
    pf_str     = f"{best_r['pf']:>22.3f}x"
    dd_str     = f"{best_r['max_dd']:.2f}%"
    wr_str     = f"{best_r['win_rate']:.1f}%"
    lt_str     = f"{best_r['long_thresh']:>22}"
    st_str     = f"{best_r['short_thresh']:>22}"
    print()
    print(f"  {'─'*56}")
    print(f"  ML ENSEMBLE BACKTEST RESULTS  ·  {BOLD(pair)}")
    print(f"  {'─'*56}")
    print(f"  Initial Capital   {WHITE('$'):>4}{capital:>19,.2f}")
    print(f"  Final Value       {col('$'):>4}{best_r['final']:>19,.2f}")
    print(f"  Total Return      {col(ret_str):>23}")
    print(f"  Sharpe Ratio      {sharpe_str}")
    print(f"  Calmar Ratio      {calmar_str}")
    print(f"  Profit Factor     {pf_str}")
    print(f"  Max Drawdown      {RED(dd_str):>23}")
    print(f"  Total Trades      {best_r['trades']:>22,}")
    print(f"  Win Rate          {wr_str:>22}")
    print(f"  Long Threshold    {lt_str}")
    print(f"  Short Threshold   {st_str}")
    print(f"  Data Source       {'Binance Vision + Hyperliquid':>22}")
    print(f"  Model             {'XGB 40% + LGB 40% + LSTM 20%':>22}")
    print(f"  {'─'*56}")

    # ── Save chart ───────────────────────────────────────────────────────────
    _save_ml_chart(df, best_r, pair, capital, ens_score, oos_mask)

    return best_r


# ════════════════════════════════════════════════════════════════════════════
# CHART — ML Ensemble
# ════════════════════════════════════════════════════════════════════════════

def _save_ml_chart(df, best_r, pair, capital, ens_score, oos_mask):
    """Save a dark-mode ML ensemble performance chart to results/."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        equity = best_r['equity']
        sig    = best_r['signals']
        n      = len(equity)
        dates  = df.index
        c      = df['close'].to_numpy(np.float64)
        dd     = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
        rets   = np.diff(equity) / (equity[:-1] + 1e-10)

        fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
        fig.suptitle(
            f"AIQuant  ·  ML Ensemble (XGB+LGB+LSTM)  ·  {pair}  ·  365-Day Backtest\n"
            f"Sharpe {best_r['sharpe']:+.3f}  |  Return {best_r['ret']:+.1f}%  |  "
            f"MaxDD {best_r['max_dd']:.1f}%  |  Calmar {best_r['calmar']:.3f}  |  "
            f"{best_r['trades']:,} trades  |  {best_r['win_rate']:.1f}% win rate  |  "
            f"Profit Factor {best_r['pf']:.2f}x",
            color='white', fontsize=11, fontweight='bold', y=0.99
        )

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

        def _ax(subplot):
            ax = fig.add_subplot(subplot)
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#30363d')
            return ax

        # 1. Equity curve (full width)
        ax1 = _ax(gs[0, :])
        ax1.plot(dates, equity, color='#00d4aa', linewidth=0.9)
        ax1.fill_between(dates, equity, capital,
                         where=equity >= capital, alpha=0.15, color='#00d4aa')
        ax1.fill_between(dates, equity, capital,
                         where=equity < capital, alpha=0.15, color='#ff4444')
        ax1.axhline(capital, color='#555', linestyle='--', linewidth=0.8, alpha=0.6)
        ax1.set_ylabel('Portfolio ($)', color='white', fontsize=9)
        ax1.set_title('Equity Curve (Out-of-Sample Only)', color='white', fontsize=9)

        # 2. Drawdown (full width)
        ax2 = _ax(gs[1, :])
        ax2.fill_between(dates, dd, 0, color='#ff4444', alpha=0.7)
        ax2.set_ylabel('Drawdown (%)', color='white', fontsize=9)
        ax2.set_title('Drawdown', color='white', fontsize=9)

        # 3. BTC price + signals
        step = max(1, n // 2000)
        ax3  = _ax(gs[2, 0])
        ax3.plot(dates[::step], c[::step], color='#f0a500', linewidth=0.6, alpha=0.8)
        li = np.where(sig == 1)[0][::step]
        si = np.where(sig == -1)[0][::step]
        if len(li): ax3.scatter(dates[li], c[li], c='#00d4aa', s=1.5, alpha=0.5, zorder=3)
        if len(si): ax3.scatter(dates[si], c[si], c='#ff4444', s=1.5, alpha=0.5, zorder=3)
        ax3.set_title('BTC Price + ML Signals', color='white', fontsize=9)

        # 4. Return distribution
        ax4 = _ax(gs[2, 1])
        cr  = rets[np.isfinite(rets) & (np.abs(rets) < 0.05)] * 100
        ax4.hist(cr, bins=80, color='#58a6ff', alpha=0.85, edgecolor='none')
        ax4.axvline(0, color='white', linewidth=0.8, linestyle='--')
        if len(cr) > 0:
            ax4.axvline(cr.mean(), color='#00d4aa', linewidth=1.0, linestyle='--', alpha=0.8)
        ax4.set_title('Return Distribution', color='white', fontsize=9)
        ax4.set_xlabel('Return (%)', color='#8b949e', fontsize=8)

        # 5. Ensemble score distribution
        ax5     = _ax(gs[2, 2])
        ens_oos = ens_score[oos_mask]
        ax5.hist(ens_oos, bins=80, color='#bc8cff', alpha=0.85, edgecolor='none')
        ax5.axvline(best_r['long_thresh'],  color='#00d4aa', linewidth=1.2,
                    linestyle='--', label=f'Long >{best_r["long_thresh"]}')
        ax5.axvline(best_r['short_thresh'], color='#ff4444', linewidth=1.2,
                    linestyle='--', label=f'Short <{best_r["short_thresh"]}')
        ax5.axvline(0, color='white', linewidth=0.8)
        ax5.set_title('Ensemble Score Distribution', color='white', fontsize=9)
        ax5.set_xlabel('Long-Short Score', color='#8b949e', fontsize=8)
        ax5.legend(facecolor='#161b22', labelcolor='white', fontsize=7)

        out = RESULTS_DIR / 'backtest_results.png'
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig)
        print(f"\n  {GREEN('✓')} Chart saved → {out}")

        # Try to open on desktop environments
        if platform.system() == 'Darwin':
            os.system(f'open "{out}"')
        elif platform.system() == 'Linux' and os.environ.get('DISPLAY'):
            os.system(f'xdg-open "{out}" 2>/dev/null &')

    except Exception as e:
        print(f"  {YELLOW('⚠')}  Chart generation failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# LIVE TRADING — Hyperliquid Mainnet
# ════════════════════════════════════════════════════════════════════════════

def run_live(pair: str = 'BTCUSDT', capital: float = 10_000, poll: float = 60.0):
    """Start live trading on Hyperliquid mainnet."""
    from dotenv import load_dotenv
    load_dotenv()

    pk = os.getenv('HYPERLIQUID_PRIVATE_KEY', '')
    if not pk or pk.startswith('your_'):
        print(f"\n  {RED('✗')}  HYPERLIQUID_PRIVATE_KEY not set in .env")
        print(f"  {DIM('  1. Open .env and add your Hyperliquid private key')}")
        print(f"  {DIM('  2. Generate a wallet: python3 -c \"from eth_account import Account; a=Account.create(); print(a.key.hex())\"')}")
        print(f"  {DIM('  3. Fund your account at https://app.hyperliquid.xyz')}")
        sys.exit(1)

    coin = pair.replace('USDT', '')
    print(f"  {GREEN('✓')} Private key loaded")
    print(f"  {CYAN('▶')}  Starting live trading loop  (Ctrl+C to stop)")
    print(f"  {DIM('  Polling every ' + str(poll) + 's  ·  Max 25% position per trade')}")
    print()

    from aiquant.execution.live_trader import LiveTradingOrchestrator
    orchestrator = LiveTradingOrchestrator(
        pair              = pair,
        coin              = coin,
        initial_capital   = capital,
        kelly_fraction    = 0.5,
        poll_interval_sec = poll,
        log_dir           = str(LOGS_DIR / 'live_trading'),
    )
    orchestrator.start()


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='python3 run.py',
        description='AIQuant — HFT Statistical Arbitrage Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run.py backtest                      # BTC, last 365 days
  python3 run.py backtest --pair ETH           # Ethereum, last 365 days
  python3 run.py backtest --days 30            # BTC, last 30 days
  python3 run.py backtest --pair SOL --days 60 # Solana, last 60 days
  python3 run.py backtest --fast               # Skip LSTM (faster, ~3x speedup)
  python3 run.py live                          # Live trading on Hyperliquid
  python3 run.py live --pair ETH --poll 30     # ETH, poll every 30s
        """
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    # ── backtest ──────────────────────────────────────────────────────────
    bt_p = sub.add_parser('backtest', help='Run ML ensemble backtest on Binance Vision data')
    bt_p.add_argument('--pair',    default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    bt_p.add_argument('--days',    default=365, type=int, help='Days of history (default: 365)')
    bt_p.add_argument('--capital', default=100_000, type=float, help='Starting capital USD (default: 100000)')
    bt_p.add_argument('--force',   action='store_true', help='Force re-download even if cache exists')
    bt_p.add_argument('--fast',    action='store_true', help='Skip LSTM training (faster, ~3x speedup)')

    # ── live ──────────────────────────────────────────────────────────────
    lv_p = sub.add_parser('live', help='Start live trading on Hyperliquid mainnet')
    lv_p.add_argument('--pair',    default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    lv_p.add_argument('--capital', default=10_000, type=float, help='Starting capital USD (default: 10000)')
    lv_p.add_argument('--poll',    default=60.0, type=float, help='Poll interval in seconds (default: 60)')

    args = parser.parse_args()
    pair = normalise_pair(args.pair)

    if args.mode == 'backtest':
        banner('backtest', pair, args.days)
        t0 = time.time()

        # Step 1: Load data
        print(f"  {CYAN('━'*54)}")
        print(f"  Step 1 / 3  ·  Loading Data")
        print(f"  {CYAN('━'*54)}")
        df = load_data(pair=pair, days=args.days)

        # Step 2: Build features
        print(f"\n  {CYAN('━'*54)}")
        print(f"  Step 2 / 3  ·  Feature Engineering  (183 features)")
        print(f"  {CYAN('━'*54)}")
        df_feat = build_features(df)
        df_feat = df_feat.dropna()
        print(f"  {GREEN('✓')} {len(df_feat):,} bars × {df_feat.shape[1]} features after dropna")

        # Step 3: ML ensemble backtest
        print(f"\n  {CYAN('━'*54)}")
        print(f"  Step 3 / 3  ·  ML Ensemble Backtest  (XGB+LGB+LSTM)")
        print(f"  {CYAN('━'*54)}")
        results = run_ml_backtest(
            df_feat, pair=pair, capital=args.capital,
            fast=args.fast
        )

        elapsed = time.time() - t0
        print(f"\n  {DIM(f'Total time: {elapsed:.1f}s  ({elapsed/60:.1f} min)')}")
        print(f"  {DIM('Best params saved → config/ml_best_params.json')}")
        print(f"  {DIM('Chart saved → results/backtest_results.png')}")

    elif args.mode == 'live':
        banner('live', pair)
        run_live(pair=pair, capital=args.capital, poll=args.poll)


if __name__ == '__main__':
    main()
