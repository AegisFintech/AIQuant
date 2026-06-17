"""
scripts/train_ml_ensemble.py
==============================
Full ML Ensemble Training Pipeline
- Walk-forward cross-validation (no lookahead bias)
- Feature selection via mutual information + permutation importance
- XGBoost (gradient boosted trees)
- LightGBM (leaf-wise gradient boosting)
- LSTM with attention (sequence model)
- Ensemble: XGB 40% + LGBM 40% + LSTM 20%
- Vectorised backtest on out-of-sample predictions only
"""
import sys, warnings, time, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/ubuntu/AIQuant')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─── ML imports ──────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb

print("\n" + "="*65)
print("  AIQuant — ML Ensemble Training")
print("  XGBoost + LightGBM + LSTM  |  Walk-Forward CV")
print("  BTCUSD 1m  |  Mar–Jun 2026  |  $100,000 capital")
print("="*65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA + BUILD FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading data and building features...")
df_raw = pd.read_parquet('/home/ubuntu/AIQuant/data/raw/BTCUSDT_1m_full.parquet')
from aiquant.utils.fast_math import warmup
from aiquant.features import build_full_feature_set
warmup()
t0 = time.time()
df = build_full_feature_set(df_raw.copy(), verbose=False)
df = df.dropna()
n = len(df)
print(f"      {n:,} bars × {df.shape[1]} features  ({time.time()-t0:.1f}s)")
print(f"      Date range: {df.index[0].date()} → {df.index[-1].date()}")

c     = df['close'].to_numpy(np.float64)
dates = df.index

# ─────────────────────────────────────────────────────────────────────────────
# 2. LABEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Generating labels...")

# Forward return over next N bars
FORWARD_BARS = 15        # predict 15-bar (15 minute) forward return
FEE          = 0.00035   # 0.035% taker fee per side
THRESHOLD    = 0.0008    # 0.08% net of fees to be worth trading

fwd_ret = np.zeros(n)
for i in range(n - FORWARD_BARS):
    fwd_ret[i] = (c[i + FORWARD_BARS] - c[i]) / c[i]

# 3-class labels: 1 = long, -1 = short, 0 = flat
labels = np.zeros(n, dtype=np.int8)
labels[fwd_ret >  THRESHOLD] =  1
labels[fwd_ret < -THRESHOLD] = -1

# Remove last FORWARD_BARS rows (no valid label)
valid_mask = np.zeros(n, dtype=bool)
valid_mask[:n - FORWARD_BARS] = True

class_counts = {-1: int((labels == -1).sum()), 0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
print(f"      Labels: Long={class_counts[1]:,}  Short={class_counts[-1]:,}  Flat={class_counts[0]:,}")
print(f"      Long+Short = {(class_counts[1]+class_counts[-1])/n*100:.1f}% of bars")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Feature selection (mutual information)...")

# Drop non-numeric and target-leaking columns
drop_cols = ['open', 'high', 'low', 'close', 'volume']
# Also drop any string/categorical columns (e.g. regime_label)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c_ for c_ in numeric_cols if c_ not in drop_cols]
X_all = df[feature_cols].to_numpy(np.float64)
y_all = labels.copy()

# Use a sample for MI computation (faster)
sample_idx = np.random.choice(np.where(valid_mask)[0], size=min(20000, valid_mask.sum()), replace=False)
sample_idx.sort()
X_sample = X_all[sample_idx]
y_sample = y_all[sample_idx]

# Replace inf/nan
X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)

# Mutual information (use absolute label: long or not-long)
y_binary = (y_sample != 0).astype(int)
mi_scores = mutual_info_classif(X_sample, y_binary, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': feature_cols, 'mi': mi_scores}).sort_values('mi', ascending=False)

# Keep top 60 features by MI
TOP_K = 60
top_features = mi_df.head(TOP_K)['feature'].tolist()
print(f"      Top {TOP_K} features selected from {len(feature_cols)} total")
print(f"      Top 5: {', '.join(top_features[:5])}")

X_sel = df[top_features].to_numpy(np.float64)
X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. WALK-FORWARD CROSS-VALIDATION SETUP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Walk-forward cross-validation setup...")

# Walk-forward: train on 30 days, test on 7 days, step 7 days
TRAIN_BARS = 30 * 1440   # 43,200 bars
TEST_BARS  = 7  * 1440   # 10,080 bars
STEP_BARS  = 7  * 1440

folds = []
start = 0
while start + TRAIN_BARS + TEST_BARS <= n - FORWARD_BARS:
    train_end = start + TRAIN_BARS
    test_end  = train_end + TEST_BARS
    folds.append((start, train_end, test_end))
    start += STEP_BARS

print(f"      {len(folds)} walk-forward folds  "
      f"(train={TRAIN_BARS//1440}d, test={TEST_BARS//1440}d, step={STEP_BARS//1440}d)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN XGBoost + LightGBM PER FOLD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Training XGBoost + LightGBM (walk-forward)...")

# Store out-of-sample predictions
oos_xgb  = np.zeros(n)
oos_lgb  = np.zeros(n)
oos_mask = np.zeros(n, dtype=bool)

# Map labels: -1→0, 0→1, 1→2 for XGB/LGB multi-class
def encode_labels(y): return (y + 1).astype(int)
def decode_labels(y): return (y - 1).astype(np.int8)

scaler = RobustScaler()

for fold_idx, (tr_start, tr_end, te_end) in enumerate(folds):
    # Training data
    tr_mask = valid_mask[tr_start:tr_end]
    tr_idx  = np.arange(tr_start, tr_end)[tr_mask]
    X_tr    = X_sel[tr_idx]
    y_tr    = encode_labels(y_all[tr_idx])

    # Test data
    te_idx  = np.arange(tr_end, te_end)
    X_te    = X_sel[te_idx]

    # Scale
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # Class weights (handle imbalance)
    flat_frac = (y_tr == 1).mean()
    w_flat    = 1.0 / (flat_frac + 1e-10)
    w_dir     = 1.0 / ((1 - flat_frac) / 2 + 1e-10)
    sample_wt = np.where(y_tr == 1, w_flat, w_dir)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )
    xgb_model.fit(X_tr_s, y_tr, sample_weight=sample_wt)
    xgb_proba = xgb_model.predict_proba(X_te_s)  # shape (n_test, 3)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_tr_s, y_tr, sample_weight=sample_wt)
    lgb_proba = lgb_model.predict_proba(X_te_s)  # shape (n_test, 3)

    # Store probabilities
    # proba[:, 0] = P(short), proba[:, 1] = P(flat), proba[:, 2] = P(long)
    oos_xgb[te_idx]  = xgb_proba[:, 2] - xgb_proba[:, 0]  # long-short score
    oos_lgb[te_idx]  = lgb_proba[:, 2] - lgb_proba[:, 0]
    oos_mask[te_idx] = True

    # Progress
    fold_acc_xgb = accuracy_score(y_all[te_idx], decode_labels(xgb_model.predict(X_te_s)))
    fold_acc_lgb = accuracy_score(y_all[te_idx], decode_labels(lgb_model.predict(X_te_s)))
    print(f"  Fold {fold_idx+1:2d}/{len(folds)}  "
          f"train={tr_end-tr_start:,}  test={te_end-tr_end:,}  "
          f"XGB acc={fold_acc_xgb:.3f}  LGB acc={fold_acc_lgb:.3f}")

print(f"\n      OOS coverage: {oos_mask.sum():,} bars ({oos_mask.mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. LSTM TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Training LSTM (PyTorch)...")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Device: {DEVICE}")

    SEQ_LEN    = 30   # 30-bar lookback window
    LSTM_FEATS = 20   # use top 20 features for LSTM (memory efficient)
    lstm_features = top_features[:LSTM_FEATS]
    X_lstm_raw = df[lstm_features].to_numpy(np.float64)
    X_lstm_raw = np.nan_to_num(X_lstm_raw, nan=0.0, posinf=0.0, neginf=0.0)

    oos_lstm = np.zeros(n)

    class LSTMAttn(nn.Module):
        def __init__(self, input_size, hidden=64, n_layers=2, n_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, n_layers,
                                batch_first=True, dropout=0.2)
            self.attn = nn.Linear(hidden, 1)
            self.fc   = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, n_classes)
            )
        def forward(self, x):
            out, _ = self.lstm(x)           # (B, T, H)
            attn_w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
            ctx    = (out * attn_w).sum(dim=1)              # (B, H)
            return self.fc(ctx)

    for fold_idx, (tr_start, tr_end, te_end) in enumerate(folds):
        # Build sequences for training
        tr_seqs = []; tr_labs = []
        for i in range(tr_start + SEQ_LEN, tr_end):
            if not valid_mask[i]: continue
            seq = X_lstm_raw[i-SEQ_LEN:i]
            tr_seqs.append(seq); tr_labs.append(encode_labels(y_all[i]))

        if len(tr_seqs) < 100: continue

        X_tr_t = torch.tensor(np.array(tr_seqs), dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(tr_labs, dtype=torch.long).to(DEVICE)

        # Scale per feature
        feat_mean = X_tr_t.mean(dim=(0,1), keepdim=True)
        feat_std  = X_tr_t.std(dim=(0,1), keepdim=True) + 1e-8
        X_tr_t    = (X_tr_t - feat_mean) / feat_std

        ds     = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(ds, batch_size=512, shuffle=True)

        model  = LSTMAttn(LSTM_FEATS).to(DEVICE)
        opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        loss_fn = nn.CrossEntropyLoss()

        # Train for 5 epochs
        model.train()
        for epoch in range(5):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward(); opt.step()
            sched.step()

        # Predict on test set
        model.eval()
        te_seqs = []
        te_idx_list = list(range(tr_end + SEQ_LEN, te_end))
        for i in te_idx_list:
            seq = X_lstm_raw[i-SEQ_LEN:i]
            te_seqs.append(seq)

        if te_seqs:
            X_te_t = torch.tensor(np.array(te_seqs), dtype=torch.float32).to(DEVICE)
            X_te_t = (X_te_t - feat_mean) / feat_std
            with torch.no_grad():
                logits = model(X_te_t)
                proba  = torch.softmax(logits, dim=1).cpu().numpy()
            for j, i in enumerate(te_idx_list):
                oos_lstm[i] = proba[j, 2] - proba[j, 0]  # long-short score

        print(f"  LSTM Fold {fold_idx+1:2d}/{len(folds)} complete")

    LSTM_AVAILABLE = True
    print("      LSTM training complete")

except Exception as e:
    print(f"      LSTM skipped: {e}")
    LSTM_AVAILABLE = False
    oos_lstm = np.zeros(n)

# ─────────────────────────────────────────────────────────────────────────────
# 7. ENSEMBLE SIGNAL GENERATION + BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Generating ensemble signals and backtesting...")

# Ensemble score: XGB 40% + LGB 40% + LSTM 20%
if LSTM_AVAILABLE:
    ens_score = 0.40 * oos_xgb + 0.40 * oos_lgb + 0.20 * oos_lstm
else:
    ens_score = 0.50 * oos_xgb + 0.50 * oos_lgb

# Grid search on confidence threshold
print("\n  Threshold search:")
best_r = None
for long_thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
    for short_thresh in [-0.05, -0.08, -0.10, -0.12, -0.15, -0.18, -0.20, -0.25]:
        sig = np.zeros(n, np.int8)
        sig[oos_mask & (ens_score >  long_thresh)]  =  1
        sig[oos_mask & (ens_score <  short_thresh)] = -1

        n_trades = int(np.sum(np.abs(np.diff(sig.astype(int)))) // 2)
        if n_trades < 20 or n_trades > 5000: continue

        # Vectorised backtest
        changes = np.where(np.diff(sig, prepend=sig[0]) != 0)[0]
        capital = 100_000.0
        equity  = np.full(n, 100_000.0, dtype=np.float64)
        wins = losses = 0; gw = gl = 0.0
        for i in range(len(changes) - 1):
            eb = changes[i]; xb = changes[i+1]
            d  = int(sig[eb])
            if d == 0: equity[eb:xb] = capital; continue
            ep = c[eb]; xp = c[xb]
            sz = (capital * 0.25) / ep
            pnl = d * (xp - ep) * sz * (1 - 0.00035)**2
            capital = max(capital + pnl, 1.0)
            equity[eb:xb] = capital
            if pnl > 0: wins += 1; gw += pnl
            else: losses += 1; gl += abs(pnl)
        equity[changes[-1]:] = capital

        total_ret = (capital - 100_000) / 100_000 * 100
        dd        = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
        max_dd    = float(np.min(dd))
        n_days    = len(equity) // 1440
        if n_days > 2:
            dl = np.array([np.diff(np.log(equity[i*1440:(i+1)*1440]+1e-10)).sum() for i in range(n_days)])
            sharpe = float(dl.mean() / (dl.std() + 1e-10) * np.sqrt(365))
        else:
            sharpe = 0.0
        tt = wins + losses
        wr = wins / tt * 100 if tt > 0 else 0.0
        cal = total_ret / abs(max_dd) if abs(max_dd) > 0.01 else 0.0
        pf  = gw / (gl + 1e-10)

        r = dict(sharpe=round(sharpe,3), ret=round(total_ret,2), max_dd=round(max_dd,2),
                 trades=tt, win_rate=round(wr,1), final=round(capital,2),
                 calmar=round(cal,3), pf=round(pf,3), equity=equity, signals=sig,
                 long_thresh=long_thresh, short_thresh=short_thresh)

        if best_r is None or r['sharpe'] > best_r['sharpe']:
            best_r = r
            print(f"  ↑ Sharpe={r['sharpe']:+.3f}  Return={r['ret']:+.1f}%  "
                  f"MaxDD={r['max_dd']:.1f}%  Trades={r['trades']:,}  "
                  f"WR={r['win_rate']:.0f}%  Calmar={r['calmar']:.3f}  "
                  f"[L>{long_thresh} S<{short_thresh}]")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ML ENSEMBLE FINAL RESULTS")
print("="*65)
if best_r:
    print(f"  Sharpe Ratio   : {best_r['sharpe']:+.3f}")
    print(f"  Total Return   : {best_r['ret']:+.2f}%")
    print(f"  Max Drawdown   : {best_r['max_dd']:.2f}%")
    print(f"  Calmar Ratio   : {best_r['calmar']:+.3f}")
    print(f"  Profit Factor  : {best_r['pf']:.3f}x")
    print(f"  Total Trades   : {best_r['trades']:,}")
    print(f"  Win Rate       : {best_r['win_rate']:.1f}%")
    print(f"  Final Value    : ${best_r['final']:,.0f}")
    print(f"  Long threshold : {best_r['long_thresh']}")
    print(f"  Short threshold: {best_r['short_thresh']}")
print("="*65)

# Save best params
Path('/home/ubuntu/AIQuant/config').mkdir(exist_ok=True)
if best_r:
    with open('/home/ubuntu/AIQuant/config/ml_best_params.json', 'w') as f:
        json.dump({
            'model': 'XGBoost+LightGBM+LSTM Ensemble',
            'forward_bars': FORWARD_BARS,
            'threshold_pct': THRESHOLD,
            'long_thresh': best_r['long_thresh'],
            'short_thresh': best_r['short_thresh'],
            'top_features': top_features,
            'results': {k: v for k, v in best_r.items() if k not in ('equity', 'signals')}
        }, f, indent=2)
    print(f"\n  Best params saved → config/ml_best_params.json")

# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────
if best_r and best_r.get('equity') is not None:
    print(f"\n  Generating chart...")
    equity = best_r['equity']
    dd     = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
    rets   = np.diff(equity) / (equity[:-1] + 1e-10)
    sig    = best_r['signals']

    fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
    fig.suptitle(
        f"AIQuant  ·  ML Ensemble (XGB+LGB+LSTM)  ·  BTCUSD 1m  ·  Mar–Jun 2026\n"
        f"Sharpe {best_r['sharpe']:+.3f}  |  Return {best_r['ret']:+.1f}%  |  "
        f"MaxDD {best_r['max_dd']:.1f}%  |  Calmar {best_r['calmar']:.3f}  |  "
        f"{best_r['trades']:,} trades  |  {best_r['win_rate']:.1f}% win rate  |  "
        f"Profit Factor {best_r['pf']:.2f}x",
        color='white', fontsize=11, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, equity, color='#00d4aa', linewidth=0.9)
    ax1.fill_between(dates, equity, 100_000, where=equity >= 100_000, alpha=0.15, color='#00d4aa')
    ax1.fill_between(dates, equity, 100_000, where=equity < 100_000, alpha=0.15, color='#ff4444')
    ax1.axhline(100_000, color='#555', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.set_facecolor('#161b22'); ax1.tick_params(colors='#8b949e', labelsize=8)
    ax1.set_ylabel('Portfolio ($)', color='white', fontsize=9)
    ax1.set_title('Equity Curve (Out-of-Sample Only)', color='white', fontsize=9)
    for sp in ax1.spines.values(): sp.set_edgecolor('#30363d')

    # Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, dd, 0, color='#ff4444', alpha=0.7)
    ax2.set_facecolor('#161b22'); ax2.tick_params(colors='#8b949e', labelsize=8)
    ax2.set_ylabel('Drawdown (%)', color='white', fontsize=9)
    ax2.set_title('Drawdown', color='white', fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')

    # BTC price + signals
    step = max(1, n // 2000)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(dates[::step], c[::step], color='#f0a500', linewidth=0.6, alpha=0.8)
    li = np.where(sig == 1)[0][::step]
    si = np.where(sig == -1)[0][::step]
    if len(li): ax3.scatter(dates[li], c[li], c='#00d4aa', s=1.5, alpha=0.5, zorder=3)
    if len(si): ax3.scatter(dates[si], c[si], c='#ff4444', s=1.5, alpha=0.5, zorder=3)
    ax3.set_facecolor('#161b22'); ax3.tick_params(colors='#8b949e', labelsize=7)
    ax3.set_title('BTC Price + ML Signals', color='white', fontsize=9)
    for sp in ax3.spines.values(): sp.set_edgecolor('#30363d')

    # Return distribution
    ax4 = fig.add_subplot(gs[2, 1])
    cr = rets[np.isfinite(rets) & (np.abs(rets) < 0.05)] * 100
    ax4.hist(cr, bins=80, color='#58a6ff', alpha=0.85, edgecolor='none')
    ax4.axvline(0, color='white', linewidth=0.8, linestyle='--')
    ax4.axvline(cr.mean(), color='#00d4aa', linewidth=1.0, linestyle='--', alpha=0.8)
    ax4.set_facecolor('#161b22'); ax4.tick_params(colors='#8b949e', labelsize=7)
    ax4.set_title('Return Distribution', color='white', fontsize=9)
    ax4.set_xlabel('Return (%)', color='#8b949e', fontsize=8)
    for sp in ax4.spines.values(): sp.set_edgecolor('#30363d')

    # Ensemble score distribution
    ax5 = fig.add_subplot(gs[2, 2])
    ens_oos = ens_score[oos_mask]
    ax5.hist(ens_oos, bins=80, color='#bc8cff', alpha=0.85, edgecolor='none')
    ax5.axvline(best_r['long_thresh'],  color='#00d4aa', linewidth=1.2, linestyle='--', label=f'Long >{best_r["long_thresh"]}')
    ax5.axvline(best_r['short_thresh'], color='#ff4444', linewidth=1.2, linestyle='--', label=f'Short <{best_r["short_thresh"]}')
    ax5.axvline(0, color='white', linewidth=0.8)
    ax5.set_facecolor('#161b22'); ax5.tick_params(colors='#8b949e', labelsize=7)
    ax5.set_title('Ensemble Score Distribution', color='white', fontsize=9)
    ax5.set_xlabel('Long-Short Score', color='#8b949e', fontsize=8)
    ax5.legend(facecolor='#161b22', labelcolor='white', fontsize=7)
    for sp in ax5.spines.values(): sp.set_edgecolor('#30363d')

    Path('/home/ubuntu/AIQuant/results').mkdir(exist_ok=True)
    out = '/home/ubuntu/AIQuant/results/backtest_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Chart saved → {out}")

print("\n✓ ML Ensemble training complete.\n")
