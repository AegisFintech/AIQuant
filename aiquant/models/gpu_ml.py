"""
aiquant/models/gpu_ml.py
=========================
GPU-accelerated ML signal generation.

Models
------
1. XGBoost  — tree_method='hist' + device='cuda'  (GPU histogram trees)
2. LightGBM — device='gpu'                         (GPU-accelerated GBDT)
3. PyTorch  — LSTM + Attention on CUDA             (sequence model)
4. Ensemble — weighted average of all three

GPU utilisation
---------------
- XGBoost GPU: ~10-50x faster than CPU for large feature matrices
- LightGBM GPU: ~5-20x faster than CPU
- PyTorch LSTM: runs entirely on CUDA — matrix multiplications on GPU tensor cores

Walk-forward cross-validation
------------------------------
All models use walk-forward CV (expanding window) to avoid look-ahead bias.
Folds are trained sequentially (required for time series), but each fold's
GPU training is fully parallelised on the device.

Usage
-----
    from aiquant.models.gpu_ml import GPUMLSignalGenerator
    gen = GPUMLSignalGenerator()
    signals = gen.fit_predict(df_features)
"""

import numpy as np
import pandas as pd
import time
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)


# ── Device detection ──────────────────────────────────────────────────────────

def get_device():
    """Return best available device string for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"PyTorch CUDA: {name} ({mem:.1f} GB)")
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("PyTorch MPS (Apple Silicon)")
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def get_xgb_device():
    """Return XGBoost device string."""
    try:
        import xgboost as xgb
        import cupy as cp
        cp.array([1.0])  # test CUDA
        return 'cuda'
    except Exception:
        return 'cpu'


def get_lgbm_device():
    """Return LightGBM device string."""
    try:
        import cupy as cp
        cp.array([1.0])
        return 'gpu'
    except Exception:
        return 'cpu'


# ── Feature selection ─────────────────────────────────────────────────────────

# Core feature groups for ML — exclude raw price levels (non-stationary)
FEATURE_GROUPS = {
    'returns':        ['returns', 'log_returns'],
    'momentum':       [c for c in [f'rsi_{p}' for p in [7, 14, 21]] +
                       [f'roc_{p}' for p in [5, 10, 20]] +
                       [f'macd_diff_{f}_{s}' for f, s in [(12, 26), (5, 13)]]
                       if True],
    'volatility':     [f'realvol_{p}' for p in [20, 60, 120, 240]] +
                      [f'atr_{p}' for p in [7, 14, 21]] +
                      [f'bb_width_{m}' for m in ['15', '20', '25']],
    'microstructure': [f'ofi_norm_{w}' for w in [5, 15, 30, 60]] +
                      [f'trade_imbalance_{w}' for w in [10, 30, 60]] +
                      ['vpin', 'kyle_lambda_20', 'kyle_lambda_60',
                       'amihud_20', 'amihud_60', 'cs_spread'],
    'statarb':        [f'zscore_price_{w}' for w in [20, 60, 120, 240]] +
                      [f'zscore_returns_{w}' for w in [20, 60, 120, 240]] +
                      ['hurst', 'half_life', 'adf_pvalue',
                       'kalman_zscore', 'is_stationary',
                       'cusum_pos', 'cusum_neg', 'regime_break'],
    'seasonality':    ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                       'minute_sin', 'minute_cos', 'is_weekend'],
    'volume':         ['vol_ratio', 'vol_zscore', 'buy_ratio_30',
                       'sell_ratio_30', 'trade_imbalance_30'],
    'candle':         ['body_pct', 'upper_wick', 'lower_wick',
                       'is_bullish', 'is_bearish'],
    'autocorr':       [f'autocorr_{l}' for l in [1, 2, 3, 5, 10]],
}


def get_available_features(df: pd.DataFrame) -> list:
    """Return all feature columns that exist in df."""
    all_feats = []
    for group in FEATURE_GROUPS.values():
        all_feats.extend(group)
    return [f for f in all_feats if f in df.columns]


def make_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0003) -> np.ndarray:
    """
    Forward-looking labels for classification.
    1 = price rises > threshold in next `horizon` bars
    -1 = price falls > threshold
    0 = flat

    Uses future returns — only valid for training, not live prediction.
    """
    close = df['close'].to_numpy(dtype=np.float64)
    n     = len(close)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - horizon):
        fwd_ret = (close[i + horizon] - close[i]) / close[i]
        if fwd_ret > threshold:
            labels[i] = 1
        elif fwd_ret < -threshold:
            labels[i] = -1
    labels[-horizon:] = 0   # last horizon bars: no label
    return labels


# ── XGBoost GPU ───────────────────────────────────────────────────────────────

class XGBoostGPUModel:
    """XGBoost with GPU histogram method."""

    def __init__(self, device: str = 'auto'):
        self.device = get_xgb_device() if device == 'auto' else device
        self.models = {}   # one per class in OvR
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        import xgboost as xgb

        self.feature_names = feature_names
        params = {
            'objective':        'multi:softprob',
            'num_class':        3,
            'tree_method':      'hist',
            'device':           self.device,
            'n_estimators':     300,
            'max_depth':        6,
            'learning_rate':    0.05,
            'subsample':        0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma':            0.1,
            'reg_alpha':        0.1,
            'reg_lambda':       1.0,
            'eval_metric':      'mlogloss',
            'verbosity':        0,
            'nthread':          -1,
        }
        # Remap labels: -1 → 0, 0 → 1, 1 → 2
        y_mapped = (y + 1).astype(int)
        dtrain = xgb.DMatrix(X, label=y_mapped,
                             feature_names=feature_names)
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            verbose_eval=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        proba = self.model.predict(dtest)   # shape (n, 3)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        mapped = np.argmax(proba, axis=1)
        return mapped - 1   # remap: 0→-1, 1→0, 2→1


# ── LightGBM GPU ─────────────────────────────────────────────────────────────

class LightGBMGPUModel:
    """LightGBM with GPU acceleration."""

    def __init__(self, device: str = 'auto'):
        self.device = get_lgbm_device() if device == 'auto' else device
        self.model  = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        import lightgbm as lgb

        params = {
            'objective':        'multiclass',
            'num_class':        3,
            'device':           self.device,
            'n_estimators':     300,
            'num_leaves':       63,
            'learning_rate':    0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq':     5,
            'min_child_samples': 20,
            'reg_alpha':        0.1,
            'reg_lambda':       1.0,
            'verbose':          -1,
            'n_jobs':           -1,
        }
        y_mapped = (y + 1).astype(int)
        ds = lgb.Dataset(X, label=y_mapped, feature_name=feature_names or 'auto')
        callbacks = [lgb.log_evaluation(period=-1)]
        self.model = lgb.train(params, ds,
                               num_boost_round=params['n_estimators'],
                               callbacks=callbacks)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)   # shape (n, 3)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) - 1


# ── PyTorch LSTM + Attention ──────────────────────────────────────────────────

class LSTMAttentionModel:
    """LSTM with multi-head attention — runs on CUDA tensor cores."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, seq_len: int = 60,
                 device: str = 'auto'):
        self.device  = get_device() if device == 'auto' else device
        self.seq_len = seq_len
        self.net     = None
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._build_model()

    def _build_model(self):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, seq_len):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=0.2)
                self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4,
                                                   batch_first=True, dropout=0.1)
                self.norm = nn.LayerNorm(hidden_dim)
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 3),   # 3 classes: short, flat, long
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                attn_out, _ = self.attn(out, out, out)
                out = self.norm(out + attn_out)
                return self.head(out[:, -1, :])   # last timestep

        self.net = _Net(self.input_dim, self.hidden_dim,
                        self.num_layers, self.seq_len).to(self.device)

    def _make_sequences(self, X: np.ndarray) -> "torch.Tensor":
        import torch
        n = len(X)
        seqs = np.zeros((n, self.seq_len, X.shape[1]), dtype=np.float32)
        for i in range(self.seq_len, n):
            seqs[i] = X[i - self.seq_len:i]
        return torch.tensor(seqs, dtype=torch.float32).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20,
            batch_size: int = 512, lr: float = 1e-3):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        X_t = self._make_sequences(X)
        y_t = torch.tensor((y + 1).astype(np.int64)).to(self.device)

        ds     = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            pin_memory=(self.device == 'cuda'))

        opt       = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        self.net.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.net(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                avg = total_loss / len(loader)
                logger.info(f"  LSTM epoch {epoch+1}/{epochs}  loss={avg:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.net.eval()
        X_t = self._make_sequences(X)
        with torch.no_grad():
            logits = self.net(X_t)
            proba  = F.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) - 1


# ── GPU ML Ensemble ───────────────────────────────────────────────────────────

class GPUMLSignalGenerator:
    """
    Ensemble of XGBoost GPU + LightGBM GPU + PyTorch LSTM.

    Walk-forward cross-validation with expanding window.
    Final signal is a weighted vote: XGB 40% + LGBM 40% + LSTM 20%.
    """

    def __init__(self, horizon: int = 5, threshold: float = 0.0003,
                 n_folds: int = 5, min_train_bars: int = 5000):
        self.horizon       = horizon
        self.threshold     = threshold
        self.n_folds       = n_folds
        self.min_train     = min_train_bars
        self.xgb_model     = None
        self.lgbm_model    = None
        self.lstm_model    = None
        self.feature_cols  = None
        self.scaler_mean   = None
        self.scaler_std    = None

    def _normalise(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Z-score normalise features."""
        if fit:
            self.scaler_mean = np.nanmean(X, axis=0)
            self.scaler_std  = np.nanstd(X, axis=0) + 1e-8
        X = (X - self.scaler_mean) / self.scaler_std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit_predict(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Walk-forward fit and predict.
        Returns df with new columns: ml_signal, ml_confidence, ml_long_prob,
        ml_short_prob, ml_flat_prob.
        """
        import platform
        _c = lambda code, t: f'\033[{code}m{t}\033[0m' if platform.system() != 'Windows' else t
        G = lambda t: _c('32', t)
        C = lambda t: _c('36', t)
        D = lambda t: _c('2',  t)
        B = lambda t: _c('1',  t)

        self.feature_cols = get_available_features(df)
        labels = make_labels(df, self.horizon, self.threshold)

        X_all = df[self.feature_cols].to_numpy(dtype=np.float32)
        y_all = labels

        n = len(df)
        fold_size = (n - self.min_train) // self.n_folds

        if fold_size < 500:
            if verbose:
                print(f"  {C('!')} Insufficient data for walk-forward CV — using single train/test split")
            self.n_folds = 1
            fold_size = n - self.min_train

        if verbose:
            xgb_dev  = get_xgb_device()
            lgbm_dev = get_lgbm_device()
            pt_dev   = get_device()
            print(f"\n  {B('GPU ML Training')}")
            print(f"  {'─' * 60}")
            print(f"    {D('XGBoost device:')}  {C(xgb_dev)}")
            print(f"    {D('LightGBM device:')} {C(lgbm_dev)}")
            print(f"    {D('PyTorch device:')}  {C(pt_dev)}")
            print(f"    {D('Features:')}        {len(self.feature_cols)}")
            print(f"    {D('Samples:')}         {n:,}")
            print(f"    {D('Walk-fwd folds:')}  {self.n_folds}")
            print(f"  {'─' * 60}")

        # Accumulate OOF predictions
        oof_xgb  = np.zeros((n, 3), dtype=np.float32)
        oof_lgbm = np.zeros((n, 3), dtype=np.float32)
        oof_lstm = np.zeros((n, 3), dtype=np.float32)

        for fold in range(self.n_folds):
            train_end = self.min_train + fold * fold_size
            test_end  = min(train_end + fold_size, n - self.horizon)

            X_train = X_all[:train_end]
            y_train = y_all[:train_end]
            X_test  = X_all[train_end:test_end]

            X_train_n = self._normalise(X_train, fit=True)
            X_test_n  = self._normalise(X_test,  fit=False)

            t0 = time.perf_counter()

            # XGBoost GPU
            self.xgb_model = XGBoostGPUModel()
            self.xgb_model.fit(X_train_n, y_train, self.feature_cols)
            oof_xgb[train_end:test_end] = self.xgb_model.predict_proba(X_test_n)

            # LightGBM GPU
            self.lgbm_model = LightGBMGPUModel()
            self.lgbm_model.fit(X_train_n, y_train, self.feature_cols)
            oof_lgbm[train_end:test_end] = self.lgbm_model.predict_proba(X_test_n)

            # PyTorch LSTM
            try:
                self.lstm_model = LSTMAttentionModel(
                    input_dim=X_train_n.shape[1],
                    hidden_dim=128, num_layers=2, seq_len=60
                )
                self.lstm_model.fit(X_train_n, y_train, epochs=15, batch_size=512)
                oof_lstm[train_end:test_end] = self.lstm_model.predict_proba(X_test_n)
            except Exception as e:
                logger.warning(f"LSTM fold {fold+1} failed: {e} — using XGB proba")
                oof_lstm[train_end:test_end] = oof_xgb[train_end:test_end]

            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"    {G('✓')} Fold {fold+1}/{self.n_folds}  "
                      f"train={train_end:,}  test={fold_size:,}  "
                      f"{D(f'{elapsed:.1f}s')}")

        # ── Weighted ensemble ─────────────────────────────────────────────
        ensemble_proba = (0.40 * oof_xgb +
                          0.40 * oof_lgbm +
                          0.20 * oof_lstm)

        ml_signal     = np.argmax(ensemble_proba, axis=1) - 1   # -1, 0, 1
        ml_confidence = np.max(ensemble_proba, axis=1)

        # Only trade when confidence > 0.45
        ml_signal = np.where(ml_confidence > 0.45, ml_signal, 0).astype(np.int8)

        df = df.copy()
        df['ml_signal']     = ml_signal
        df['ml_confidence'] = ml_confidence
        df['ml_long_prob']  = ensemble_proba[:, 2]
        df['ml_flat_prob']  = ensemble_proba[:, 1]
        df['ml_short_prob'] = ensemble_proba[:, 0]

        if verbose:
            n_long  = int((ml_signal ==  1).sum())
            n_short = int((ml_signal == -1).sum())
            n_flat  = int((ml_signal ==  0).sum())
            print(f"  {'─' * 60}")
            print(f"  {G('✓')} ML signals — {C('▲')} {n_long} long  "
                  f"{C('▼')} {n_short} short  {D(str(n_flat))} flat  "
                  f"(confidence threshold: 0.45)")

        return df
