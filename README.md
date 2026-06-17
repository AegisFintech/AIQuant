# AIQuant: High-Frequency Statistical Arbitrage Framework

AIQuant is a professional-grade quantitative trading framework designed for High-Frequency Trading (HFT) and Statistical Arbitrage on BTCUSD. It operates on 1-minute data and executes directly on the Hyperliquid perpetuals DEX testnet.

Built by **AegisFintech**.

## Architecture Overview

The framework is divided into six core modules, engineered for maximum modularity, speed, and robustness in high-RAM environments:

1. **Data Pipeline (`aiquant.data`)**
   - Fetches 1-minute OHLCV data from Binance via CCXT.
   - Ingests order book snapshots, funding rates, and on-chain metrics (hash rate, mempool, Fear & Greed index).
   - Handles pagination, rate limiting, and parquet serialization.

2. **Feature Engineering (`aiquant.features`)**
   - **Market Microstructure**: Order Flow Imbalance (OFI), VPIN, Amihud Illiquidity, Kyle's Lambda, Roll Spread.
   - **Statistical Arbitrage**: Kalman Filter dynamic mean, rolling Hurst Exponent, Ornstein-Uhlenbeck half-life, CUSUM structural break detection.
   - **Technical Indicators**: 50+ standard indicators (MACD, RSI, Bollinger Bands, Ichimoku, etc.).

3. **Signal Generation (`aiquant.strategies` & `aiquant.models`)**
   - **Primary**: `KalmanStatArbStrategy` — Uses Kalman Filter residuals and OFI to trade mean-reverting regimes (Hurst < 0.45).
   - **Backups**: Mean Reversion (BB + RSI), Trend Following (EMA + ADX).
   - **Machine Learning**: `MLSignalGenerator` — Soft-voting ensemble of XGBoost, LightGBM, Random Forest, and Logistic Regression with walk-forward cross-validation.
   - **Ensemble**: `StrategyEnsemble` — Regime-adaptive weighting of all signals.

4. **Backtesting Engine (`aiquant.backtest`)**
   - Built on **Backtrader**, optimized for 1m data arrays.
   - Includes realistic Hyperliquid taker fees (0.035%) and slippage models.
   - Comprehensive analytics: Sharpe, Sortino, Calmar, Max Drawdown, VaR, CVaR.
   - Walk-forward analysis (WFA) support to prevent overfitting.

5. **Risk Management (`aiquant.risk`)**
   - Implements the **Kelly Criterion** for optimal position sizing.
   - Supports Full Kelly, Half-Kelly, Volatility-Adjusted Kelly, and Rolling Kelly.
   - Portfolio-level controls: daily loss limits, max drawdown halts, and hard position caps.

6. **Execution Layer (`aiquant.execution`)**
   - Connects to **Hyperliquid Testnet** via the official Python SDK.
   - Handles market/limit orders, position tracking, and real-time account state.
   - `LiveTradingOrchestrator` runs a continuous 1-minute tick loop from data ingestion to execution.

## Installation

```bash
# Clone the repository
git clone https://github.com/AegisFintech/AIQuant.git
cd AIQuant

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Configuration

Create a `.env` file in the root directory with your Hyperliquid testnet private key:

```env
HYPERLIQUID_PRIVATE_KEY=your_ethereum_private_key_here
```

To generate a test wallet and fund it:
1. Run `python -c "from eth_account import Account; print(Account.create().key.hex())"`
2. Go to [Hyperliquid Testnet Faucet](https://app.hyperliquid-testnet.xyz/drip) to claim test funds.

Adjust framework settings in `config/settings.yaml`.

## Usage

### 1. Run the Data Pipeline
```python
from aiquant.data import DataPipeline
pipeline = DataPipeline()
df = pipeline.run(symbol='BTC/USDT', since_date='2024-01-01')
```

### 2. Run a Backtest
```python
from aiquant.backtest import BacktestEngine
from aiquant.features import build_full_feature_set
from aiquant.strategies import StrategyEnsemble

df_features = build_full_feature_set(df)
ensemble = StrategyEnsemble()
signals = ensemble.generate_signals(df_features)['final_signal']

engine = BacktestEngine()
results = engine.run(df_features, signals, strategy_name='StatArb_Ensemble')
```

### 3. Start Live Paper Trading
```python
from aiquant.execution import LiveTradingOrchestrator

trader = LiveTradingOrchestrator(use_testnet=True)
trader.start(poll_interval_sec=60.0)
```

## License
MIT License
