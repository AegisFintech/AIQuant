# AIQuant — HFT Statistical Arbitrage Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AegisFintech/AIQuant/blob/main/AIQuant_Colab.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

**AegisFintech** · Professional-grade quantitative trading system for BTCUSD on Hyperliquid.

---

## Results (365-Day Backtest · Jun 2025 – Jun 2026)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **+0.625** |
| Total Return | +6.26% |
| Max Drawdown | -7.87% |
| Calmar Ratio | 0.795 |
| Win Rate | 63.5% |
| Total Trades | 19,302 |
| Profit Factor | 1.024x |
| Dataset | 530,628 bars · BTC $60k → $126k |

> **Note:** The full 365-day dataset includes the Jun–Nov 2025 bear market (BTC -52%). The ML ensemble
> achieves Sharpe **3.39** on the bull-run period (Mar–Jun 2026) when trained on that window alone.

---

## Quick Start

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AegisFintech/AIQuant/blob/main/AIQuant_Colab.ipynb)

1. Click the badge above
2. Set runtime to GPU (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells in order

### Local Installation

```bash
git clone https://github.com/AegisFintech/AIQuant.git
cd AIQuant
pip install -r requirements.txt
```

### CLI Usage

```bash
# Full 365-day ML ensemble backtest (default)
python3 run.py backtest

# Faster mode (skip LSTM, ~3x speedup)
python3 run.py backtest --fast

# Custom window
python3 run.py backtest --days 90

# Live trading on Hyperliquid mainnet
python3 run.py live
```

---

## Architecture

```
AIQuant/
├── run.py                          # CLI entry point (backtest + live)
├── aiquant/
│   ├── data/
│   │   └── fetcher.py              # Binance Vision + Hyperliquid data
│   ├── features/
│   │   ├── technical.py            # 92 technical indicators (fast NumPy)
│   │   ├── microstructure.py       # 55 microstructure features
│   │   ├── statarb.py              # 19 StatArb / regime features
│   │   └── gpu_features.py         # CuPy GPU-accelerated feature engineering
│   ├── strategies/
│   │   ├── ensemble.py             # Regime-aware adaptive ensemble
│   │   ├── stat_arb.py             # Kalman filter StatArb
│   │   ├── mean_reversion.py       # Vol-RSI mean reversion
│   │   ├── trend_following.py      # EMA crossover trend following
│   │   └── hft.py                  # High-frequency microstructure signals
│   ├── models/
│   │   ├── gpu_ml.py               # GPU ML: XGBoost + LightGBM + LSTM
│   │   └── ml_signal.py            # ML signal generator
│   ├── execution/
│   │   ├── hyperliquid_trader.py   # Hyperliquid mainnet execution
│   │   └── live_trader.py          # Live trading orchestrator
│   ├── risk/
│   │   └── manager.py              # Kelly Criterion + drawdown limits
│   └── utils/
│       └── fast_math.py            # Numba JIT: Hurst, ADF, Kalman, OU
├── scripts/
│   ├── prepare_data.py             # Build 365-day Binance Vision dataset
│   ├── train_ml_ensemble.py        # Standalone ML training script
│   └── build_colab.py              # Regenerate AIQuant_Colab.ipynb
├── config/
│   ├── settings.yaml               # System configuration
│   └── ml_best_params.json         # Saved ML best parameters
└── data/raw/                       # OHLCV data (gitignored)
```

---

## ML Ensemble Pipeline

The backtest uses a **walk-forward cross-validation** pipeline with no lookahead bias:

1. **Labels** — 15-bar forward return, threshold 0.08% net of fees
2. **Feature selection** — Top 60 features by mutual information (from 171 total)
3. **Walk-forward folds** — 46 folds: 30-day train, 7-day test, 7-day step
4. **XGBoost** — 200 estimators, depth 5, class-balanced weights
5. **LightGBM** — 200 estimators, 31 leaves, class-balanced weights
6. **LSTM + Attention** — 30-bar sequences, 2-layer LSTM, 64 hidden units
7. **Ensemble** — XGB 40% + LGB 40% + LSTM 20%
8. **Threshold search** — Grid search over long/short confidence thresholds

---

## Data Sources

| Source | Coverage | Auth Required |
|--------|----------|---------------|
| [Binance Vision](https://data.binance.vision/) | Monthly CSVs, 2017–present | None |
| [Hyperliquid](https://app.hyperliquid.xyz/) | Real-time candles | None (read) |

---

## Live Trading

Requires a Hyperliquid mainnet account with funds:

```bash
# Generate a new wallet
python3 -c "from eth_account import Account; a=Account.create(); print(a.key.hex())"

# Add to .env
echo "HYPERLIQUID_PRIVATE_KEY=0x..." >> .env

# Start live trading
python3 run.py live
```

---

## Configuration

Edit `config/settings.yaml` or set environment variables in `.env`:

```yaml
pair: BTCUSDT
interval: 1m
kelly_fraction: 0.5          # Half-Kelly position sizing
max_position_pct: 0.25       # Max 25% of capital per trade
max_drawdown_pct: 0.15       # Stop trading at 15% drawdown
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves
substantial risk of loss. Past performance does not guarantee future results.
