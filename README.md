# AIQuant

**HFT Statistical Arbitrage Framework for BTCUSD and major crypto pairs.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Data](https://img.shields.io/badge/Backtest_Data-CryptoDataDownload-green)](https://www.cryptodatadownload.com)
[![Execution](https://img.shields.io/badge/Live_Trading-Hyperliquid-purple)](https://hyperliquid.xyz)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AegisFintech/AIQuant/blob/main/AIQuant_Colab.ipynb)

Built by [AegisFintech](https://github.com/AegisFintech) · Apache 2.0 License

---

## Overview

AIQuant is a professional-grade quantitative trading system built around **Kalman Filter Statistical Arbitrage** as the primary strategy, with Mean Reversion, Trend Following, and an ML Ensemble (XGBoost + LightGBM) as backups. It uses **Kelly Criterion** position sizing, a **Backtrader** backtesting engine, and **Hyperliquid** for live execution.

| Mode | Data Source | Purpose |
|---|---|---|
| `backtest` | [CryptoDataDownload](https://www.cryptodatadownload.com/) | Historical 1m OHLCV, full history since 2017, no API key needed |
| `live` | Hyperliquid public API + mainnet | Live 1m candles + real execution, requires private key in `.env` |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/AegisFintech/AIQuant.git
cd AIQuant
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — only HYPERLIQUID_PRIVATE_KEY is required for live trading

# 3. Run a backtest (BTC, last 90 days — no API key needed)
python3 run.py backtest
```

---

## Commands

```
python3 run.py <mode> [options]
```

### `backtest` — Historical Backtesting

```bash
python3 run.py backtest                        # BTC, last 90 days (default)
python3 run.py backtest --pair ETH             # Ethereum (shorthand accepted)
python3 run.py backtest --days 30              # shorter window
python3 run.py backtest --pair SOL --days 60   # Solana, 60 days
python3 run.py backtest --capital 50000        # custom starting capital
python3 run.py backtest --force                # force re-download data
```

| Option | Default | Description |
|---|---|---|
| `--pair` | `BTCUSDT` | Trading pair. Accepts `BTC`, `btc`, `BTCUSDT` — all normalised automatically |
| `--days` | `90` | Days of 1m history (90 days = 129,600 bars) |
| `--capital` | `100000` | Starting capital in USD |
| `--force` | off | Force re-download even if cached data exists |

### `live` — Live Trading on Hyperliquid

```bash
python3 run.py live                            # BTC, poll every 60s
python3 run.py live --pair ETH --poll 30       # ETH, poll every 30s
python3 run.py live --capital 5000             # custom capital
```

| Option | Default | Description |
|---|---|---|
| `--pair` | `BTCUSDT` | Trading pair |
| `--capital` | `10000` | Starting capital in USD |
| `--poll` | `60` | Seconds between each trading loop tick |

---

## Data Sources

### Backtest — CryptoDataDownload

- **No API key required.** Completely free.
- Full 1m OHLCV history from 2017 to present for BTC, ETH, SOL, BNB, XRP.
- AIQuant uses a **streaming tail-read** — only the last N days are downloaded, not the full 177MB file. A 90-day backtest downloads ~5MB.
- Data is cached to `data/raw/<pair>_1m_cdd.parquet` after first fetch. Subsequent runs load from cache instantly.

### Live Trading — Hyperliquid

- **Market data:** Hyperliquid public `candleSnapshot` API — no auth needed, returns live 1m candles.
- **Execution:** Hyperliquid mainnet perpetuals DEX — requires `HYPERLIQUID_PRIVATE_KEY` in `.env`.
- Non-custodial, no KYC, 0.035% taker fee, deep liquidity.

#### Setting up Hyperliquid

```bash
# 1. Generate a wallet
python3 -c "from eth_account import Account; a=Account.create(); print('Key:', a.key.hex(), '\nAddress:', a.address)"

# 2. Add to .env
HYPERLIQUID_PRIVATE_KEY=<your_key_here>

# 3. Fund your account at https://app.hyperliquid.xyz (deposit USDC via Arbitrum)

# 4. Start live trading
python3 run.py live
```

---

## Supported Pairs

| Pair | Backtest (CDD) | Live (Hyperliquid) |
|---|---|---|
| `BTCUSDT` | Full history since 2017 | ✅ |
| `ETHUSDT` | Full history since 2017 | ✅ |
| `SOLUSDT` | Full history | ✅ |
| `BNBUSDT` | Full history | ✅ |
| `XRPUSDT` | Full history | ✅ |

---

## Strategy Architecture

### Primary — Kalman Filter Statistical Arbitrage

Exploits mean-reverting price regimes using a Kalman Filter dynamic mean. Only trades when the market is statistically confirmed to be mean-reverting (Hurst exponent < 0.45). Entry triggered when Kalman z-score exceeds ±1.8 standard deviations, confirmed by Order Flow Imbalance.

### Backup Strategies

| Strategy | Trigger | Regime |
|---|---|---|
| **Mean Reversion** | Bollinger Band + RSI extremes | Mean-reverting (Hurst < 0.5) |
| **Trend Following** | EMA crossover + ADX > 25 | Trending (Hurst > 0.55) |
| **ML Ensemble** | XGBoost + LightGBM + RF | Any regime |

---

## Feature Engineering (100+ features)

| Category | Features |
|---|---|
| **Technical** | EMA, SMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, Williams %R, CCI, OBV, VWAP |
| **Microstructure** | OFI, VPIN, Amihud Illiquidity, Kyle's Lambda, Roll Spread, Corwin-Schultz spread |
| **Statistical Arbitrage** | Kalman Filter mean/z-score, rolling Hurst Exponent, OU half-life, ADF p-value, CUSUM |
| **Intraday Seasonality** | Cyclically encoded hour, minute, day-of-week |

---

## Risk Management

| Parameter | Default | Description |
|---|---|---|
| Kelly Fraction | 0.5 (Half-Kelly) | Reduces full Kelly by 50% to account for estimation error |
| Max Position | 25% of portfolio | Hard cap per trade |
| Max Daily Loss | 3% | Trading halts for the day if breached |
| Max Drawdown | 15% | System halts if portfolio drawdown exceeds this |

---

## Viewing Backtest Charts

| Environment | How to View |
|---|---|
| **Desktop (Mac / Linux / Windows)** | Chart opens automatically after backtest |
| **Headless Linux server** | Saved to `results/backtest_results.png` — `scp` it down |
| **Google Colab** | Displayed inline in the notebook |

---

## About T-90 Days

The default 90-day window = **129,600 one-minute bars**. Sufficient for all strategy components:

| Requirement | Bars Needed | T-90 Coverage |
|---|---|---|
| EMA 200 warmup | 200 bars | ✅ |
| Hurst Exponent (100-bar window) | 100 bars | ✅ |
| Kalman Filter convergence | ~500 bars | ✅ |
| Rolling Kelly (50-trade history) | ~1,000 bars | ✅ |
| ML model training | 30,000+ bars | ✅ (129,600) |

For ML ensemble training, use `--days 180` (259,200 bars).

---

## Project Structure

```
AIQuant/
├── run.py                          <- Main CLI entry point
├── .env.example                    <- Environment variable template
├── requirements.txt
├── LICENSE                         <- Apache 2.0
├── AIQuant_Colab.ipynb             <- Google Colab notebook
├── QUANT_RESEARCH_REPORT.md        <- Mathematical foundations
├── aiquant/
│   ├── data/
│   │   ├── fetcher.py              <- CDD (backtest) + Hyperliquid (live) data
│   │   ├── onchain.py              <- On-chain metrics (CoinGecko, Fear & Greed)
│   │   └── pipeline.py            <- Data pipeline orchestrator
│   ├── features/
│   │   ├── technical.py            <- 50+ technical indicators
│   │   ├── microstructure.py       <- Market microstructure features
│   │   └── statarb.py              <- Statistical arbitrage features
│   ├── strategies/
│   │   ├── stat_arb.py             <- Primary: Kalman Filter StatArb
│   │   ├── mean_reversion.py       <- Backup: Bollinger + RSI reversion
│   │   ├── trend_following.py      <- Backup: EMA crossover + MACD
│   │   └── ensemble.py             <- Regime-adaptive signal combiner
│   ├── models/
│   │   └── ml_signal.py            <- XGBoost + LightGBM + RF ensemble
│   ├── backtest/
│   │   ├── engine.py               <- Backtrader engine
│   │   └── analytics.py            <- Sharpe, Sortino, Calmar, VaR, CVaR
│   ├── risk/
│   │   └── position_sizing.py      <- Kelly criterion + drawdown controls
│   ├── execution/
│   │   ├── hyperliquid_trader.py   <- Hyperliquid mainnet execution
│   │   └── live_trader.py          <- Live trading orchestrator
│   └── utils/
│       └── fast_math.py            <- Numba JIT-compiled math functions
├── data/raw/                       <- Cached parquet files (gitignored)
├── results/                        <- Backtest charts
└── logs/live_trading/              <- Trade logs (JSON)
```

---

## Google Colab

Run AIQuant in the cloud with zero local setup:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AegisFintech/AIQuant/blob/main/AIQuant_Colab.ipynb)

| Colab Tier | RAM | Recommended Days |
|---|---|---|
| Free | 12 GB | 30 days |
| Pro | 25 GB | 90 days |
| Pro+ | 52 GB | 180 days |

---

## Performance Optimisations

| Technique | Where Used | Speedup |
|---|---|---|
| **Numba JIT** | Hurst, OU half-life, autocorrelation, Kalman filter | 100–200x vs pandas rolling.apply |
| **NumPy vectorisation** | All signal generation and feature computation | 10–50x vs Python loops |
| **Parquet caching** | CDD data cached after first fetch | Instant re-load |
| **Streaming tail-read** | CDD download | Only fetches required bars, not full 177MB |
| **int8 signal arrays** | Signal generation | 8x less RAM than int64 |

---

## Requirements

- Python 3.9+
- 8 GB RAM minimum (16 GB recommended for 90-day backtests)
- Internet connection

```bash
pip install -r requirements.txt
```

---

## License

Apache License 2.0 — Copyright 2026 AegisFintech. See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for research and educational purposes only. It does not constitute financial advice. Past backtest performance does not guarantee future results. Trading cryptocurrencies involves significant risk of loss.
