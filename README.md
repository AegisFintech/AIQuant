# AIQuant

**HFT Statistical Arbitrage Framework for BTCUSD**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Pair](https://img.shields.io/badge/default%20pair-BTCUSDT-orange)](https://www.binance.com)
[![Data](https://img.shields.io/badge/data-Binance%20Public%20API-green)](https://api.binance.com)

A professional-grade quantitative trading system built for high-RAM machines. Designed around a **Kalman Filter Statistical Arbitrage** primary strategy with trend-following, mean-reversion, and ML ensemble backups. Includes a full Backtrader backtesting engine, a self-contained paper trading simulator, and Kelly Criterion position sizing — all runnable from a single command with zero account registration required.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AegisFintech/AIQuant.git
cd AIQuant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment
cp .env.example .env
# (Optional) Edit .env to add API keys for on-chain data enrichment

# 4. Run a backtest — defaults to BTC, last 90 days, no config needed
python3 run.py backtest
```

That is it. No account, no API key, no registration needed for backtesting and paper trading.

---

## Commands

All commands follow the pattern `python3 run.py <mode> [options]`.

| Command | Description |
|---|---|
| `python3 run.py backtest` | Run full backtest on BTC, last 90 days |
| `python3 run.py paper` | Paper trade BTC (30-bar replay demo) |
| `python3 run.py paper --bars 0` | Live paper trade (polls Binance every 60s) |
| `python3 run.py fetch` | Pre-fetch and cache BTC data |
| `python3 run.py serve` | Open chart viewer in browser |

### Options

| Flag | Default | Description |
|---|---|---|
| `--pair` | `BTCUSDT` | Trading pair (e.g. `ETHUSDT`, `SOLUSDT`) |
| `--days` | `90` | Days of history to use (T-90 from today) |
| `--bars` | `30` | Bars to replay in paper trading mode (`0` = live) |
| `--capital` | `10000` | Starting capital for paper trading |
| `--force` | off | Force re-fetch data even if cache is fresh |
| `--port` | `8765` | Port for the chart viewer server |

### Examples

```bash
# BTC backtest, last 90 days (default — no flags needed)
python3 run.py backtest

# ETH backtest, last 60 days
python3 run.py backtest --pair ETHUSDT --days 60

# BTC backtest, last 30 days
python3 run.py backtest --days 30

# Paper trade BTC, replay 50 bars at 2s each
python3 run.py paper --bars 50

# Live paper trading — polls Binance every 60 seconds indefinitely
python3 run.py paper --bars 0

# Pre-fetch 180 days of BTC data (recommended for ML training)
python3 run.py fetch --days 180

# Open chart viewer in browser (for headless server users)
python3 run.py serve
```

---

## Viewing Charts

The backtest chart is automatically saved to `results/backtest_results.png` after every run.

| Environment | How to View |
|---|---|
| **Desktop (Mac / Linux / Windows)** | Chart opens automatically in your default image viewer |
| **Headless server** | Run `python3 run.py serve` then open `http://localhost:8765/viewer.html` |
| **Remote server** | `scp user@yourserver:~/AIQuant/results/backtest_results.png .` |

---

## About T-90 Days

The default window of **90 days = 129,600 one-minute bars**. This is more than sufficient for all strategy components:

| Requirement | Bars Needed | T-90 Coverage |
|---|---|---|
| EMA 200 warmup | 200 bars | Yes |
| Hurst Exponent (100-bar window) | 100 bars | Yes |
| Kalman Filter convergence | ~500 bars | Yes |
| Rolling Kelly (50-trade history) | ~1,000+ bars | Yes |
| ML model training (recommended) | 30,000+ bars | Yes (129,600) |

For ML ensemble training, use `--days 180` (259,200 bars). The minimum usable window is `--days 7` for quick iteration.

---

## Architecture

```
AIQuant/
├── run.py                          <- Single entry point (start here)
├── aiquant/
│   ├── data/
│   │   ├── fetcher.py              <- Binance 1m OHLCV, order book, funding rates
│   │   ├── onchain.py              <- Glassnode, CoinGecko, Fear & Greed Index
│   │   └── pipeline.py             <- Data orchestrator
│   ├── features/
│   │   ├── technical.py            <- EMA, RSI, MACD, Bollinger, ATR, VWAP (50+ indicators)
│   │   ├── microstructure.py       <- OFI, VPIN, Amihud, Kyle's Lambda, Roll Spread
│   │   └── statarb.py              <- Kalman Filter, Hurst, OU half-life, ADF, CUSUM
│   ├── strategies/
│   │   ├── stat_arb.py             <- Primary: Kalman StatArb (mean-reversion regime)
│   │   ├── mean_reversion.py       <- Backup: Bollinger + RSI reversion
│   │   ├── trend_following.py      <- Backup: EMA crossover + MACD momentum
│   │   └── ensemble.py             <- Regime-adaptive signal combiner
│   ├── models/
│   │   └── ml_signal.py            <- XGBoost + LightGBM + RF ensemble (walk-forward CV)
│   ├── backtest/
│   │   ├── engine.py               <- Backtrader engine (1m data, realistic fees + slippage)
│   │   └── analytics.py            <- Sharpe, Sortino, Calmar, VaR, CVaR, SQN
│   ├── risk/
│   │   └── position_sizing.py      <- Full Kelly, Half-Kelly, Vol-Adjusted Kelly, drawdown halts
│   └── execution/
│       ├── paper_trader.py         <- Self-contained paper trading (no account needed)
│       └── live_trader.py          <- Live trading orchestrator
├── config/
│   └── settings.yaml               <- All strategy parameters
├── data/raw/                       <- Cached parquet files (gitignored)
├── results/                        <- Charts and reports
├── logs/                           <- Trade logs and equity curves
├── .env.example                    <- Environment variable template
└── requirements.txt
```

---

## Strategy Stack

### Primary — Kalman Filter Statistical Arbitrage

Exploits mean-reverting price regimes identified by a **Hurst Exponent < 0.45**. The Kalman Filter tracks a dynamic fair-value mean; trades are entered when the price deviates beyond ±2.0 standard deviations, confirmed by RSI and Order Flow Imbalance.

- **Entry:** Kalman z-score > ±2.0, RSI confirmation (< 42 long / > 58 short), OFI alignment
- **Exit:** Z-score reverts to ±0.5, stop-loss 1.2%, take-profit 2.5%, max hold 60 bars
- **Regime filter:** Only active when Hurst < 0.45 (confirmed mean-reverting)

### Backup — Trend Following

Activates when Hurst > 0.55 (confirmed trending regime). Uses EMA crossover (5/20) with MACD histogram confirmation and volume surge filter (volume ratio > 1.2x average).

### Backup — Mean Reversion (Bollinger)

Bollinger Band %B reversion with RSI divergence. Operates in neutral regimes (0.45 <= Hurst <= 0.55).

### ML Ensemble

XGBoost + LightGBM + Random Forest trained on 100+ features with walk-forward cross-validation. Used as a signal quality filter on top of the rule-based strategies.

---

## Risk Management

All position sizing uses the **Kelly Criterion** with a 0.5 fraction (Half-Kelly) by default.

| Control | Value |
|---|---|
| Kelly fraction | 0.5 (Half-Kelly) |
| Max position size | 25% of portfolio |
| Stop-loss | 1.2% per trade |
| Take-profit | 2.5% per trade |
| Max drawdown halt | 15% |
| Daily loss limit | 5% |
| Taker fee (simulated) | 0.035% |
| Slippage (simulated) | 0.01% |

---

## Data Sources

All data sources used are **free and require no API key** for basic operation.

| Source | Data | Key Required |
|---|---|---|
| Binance Public API | 1m OHLCV, order book | No |
| CoinGecko | BTC market data, dominance | No |
| Alternative.me | Fear & Greed Index | No |
| Glassnode | On-chain metrics (SOPR, MVRV, hash rate) | Yes (free tier) |
| CryptoCompare | Social sentiment, news | Yes (free tier) |

---

## Configuration

Edit `config/settings.yaml` to tune strategy parameters, or set environment variables in `.env`.

Key parameters to tune for improved performance:

```yaml
strategy:
  stat_arb:
    kalman_zscore_entry: 2.0     # Widen to 2.5 for fewer, higher-quality trades
    stop_loss: 0.012             # 1.2% stop
    take_profit: 0.025           # 2.5% target
  regime:
    hurst_mean_rev_threshold: 0.45
    hurst_trend_threshold: 0.55

risk:
  kelly_fraction: 0.5            # Half-Kelly (conservative)
  max_drawdown: 0.15             # Halt trading at 15% drawdown
```

---

## Installation

**Requirements:** Python 3.10+, 8 GB+ RAM recommended (16 GB+ for ML training on 180-day datasets)

```bash
pip install -r requirements.txt
```

Core dependencies: `backtrader`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `requests`, `pyarrow`

---

## License

Copyright 2026 AegisFintech. Licensed under the [Apache License 2.0](LICENSE).

---

## Disclaimer

This software is for research and educational purposes only. It does not constitute financial advice. Past backtest performance does not guarantee future results. Trading cryptocurrencies involves significant risk of loss.
