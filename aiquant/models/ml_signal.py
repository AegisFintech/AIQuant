"""
aiquant/models/ml_signal.py
=============================
BACKUP STRATEGY 3: Machine Learning Signal Generator.

Uses an ensemble of:
  - XGBoost (gradient boosted trees)
  - LightGBM (fast gradient boosting)
  - Random Forest (bagged trees)
  - Logistic Regression (linear baseline)

Target: 3-class classification — Long (+1), Short (-1), Flat (0)
Based on forward return sign over N bars.

Features: All engineered features from the feature pipeline.
Walk-forward cross-validation to prevent lookahead bias.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """
    Ensemble ML signal generator with walk-forward validation.
    """

    def __init__(
        self,
        forward_bars: int = 15,       # Predict return over next 15 bars (15 minutes)
        signal_threshold: float = 0.001,  # Min return to classify as directional
        n_splits: int = 5,
        model_dir: str = 'models',
        feature_importance_top_n: int = 30,
    ):
        self.forward_bars = forward_bars
        self.signal_threshold = signal_threshold
        self.n_splits = n_splits
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_importance_top_n = feature_importance_top_n
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Target Construction
    # ------------------------------------------------------------------

    def make_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create 3-class target from forward returns.
          +1 (Long)  : forward return > +threshold
          -1 (Short) : forward return < -threshold
           0 (Flat)  : |forward return| <= threshold
        """
        fwd_return = df['close'].pct_change(self.forward_bars).shift(-self.forward_bars)
        target = pd.Series(0, index=df.index, name='target')
        target[fwd_return > self.signal_threshold] = 1
        target[fwd_return < -self.signal_threshold] = -1
        return target

    # ------------------------------------------------------------------
    # Feature Selection
    # ------------------------------------------------------------------

    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select numeric feature columns, excluding raw OHLCV and target.
        """
        exclude = {'open', 'high', 'low', 'close', 'volume', 'target',
                   'regime', 'fear_greed_class', 'is_stationary', 'regime_break'}
        cols = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and not c.startswith('close_')  # Avoid raw price columns
        ]
        return cols

    # ------------------------------------------------------------------
    # Model Construction
    # ------------------------------------------------------------------

    def build_ensemble(self) -> VotingClassifier:
        """Build a soft-voting ensemble of XGBoost, LightGBM, RF, and LR."""
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42,
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
        lr_clf = LogisticRegression(
            max_iter=1000,
            C=0.1,
            multi_class='multinomial',
            n_jobs=-1,
            random_state=42,
        )
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_clf),
                ('lgb', lgb_clf),
                ('rf', rf_clf),
                ('lr', lr_clf),
            ],
            voting='soft',
            n_jobs=-1,
        )
        return ensemble

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the ensemble with walk-forward cross-validation.
        Returns performance metrics across folds.
        """
        logger.info("Preparing ML training data...")
        target = self.make_target(df)
        self.feature_cols = self.select_features(df)

        X = df[self.feature_cols].copy()
        y = target.copy()

        # Drop rows with NaN in features or target
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"Training set: {len(X):,} samples, {len(self.feature_cols)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            X_train_sc = self.scaler.fit_transform(X_train)
            X_val_sc = self.scaler.transform(X_val)

            # Build and train model
            model = self.build_ensemble()
            model.fit(X_train_sc, y_train)

            # Evaluate
            y_pred = model.predict(X_val_sc)
            acc = accuracy_score(y_val, y_pred)
            fold_metrics.append({'fold': fold + 1, 'accuracy': acc})
            logger.info(f"Fold {fold + 1}/{self.n_splits} | Accuracy: {acc:.4f}")

        # Train final model on all data
        logger.info("Training final model on full dataset...")
        X_scaled = self.scaler.fit_transform(X)
        self.model = self.build_ensemble()
        self.model.fit(X_scaled, y)

        # Save model
        self.save_model()

        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        logger.info(f"Walk-forward CV Average Accuracy: {avg_acc:.4f}")

        return {
            'fold_metrics': fold_metrics,
            'avg_accuracy': avg_acc,
            'n_features': len(self.feature_cols),
            'n_samples': len(X),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals from trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = df[self.feature_cols].copy()
        valid_mask = X.notna().all(axis=1)
        X_scaled = self.scaler.transform(X[valid_mask])
        preds = self.model.predict(X_scaled)

        signals = pd.Series(0, index=df.index, name='ml_signal')
        signals[valid_mask] = preds
        return signals

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities for position sizing."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        X = df[self.feature_cols].copy()
        valid_mask = X.notna().all(axis=1)
        X_scaled = self.scaler.transform(X[valid_mask])
        proba = self.model.predict_proba(X_scaled)
        classes = self.model.classes_
        df_proba = pd.DataFrame(proba, index=df.index[valid_mask], columns=[f'prob_{c}' for c in classes])
        return df_proba

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, name: str = 'ml_signal_ensemble'):
        path = self.model_dir / f'{name}.pkl'
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_cols}, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, name: str = 'ml_signal_ensemble'):
        path = self.model_dir / f'{name}.pkl'
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_cols = data['features']
        logger.info(f"Model loaded from {path}")
