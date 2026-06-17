# Quantitative Research Report: High-Frequency Statistical Arbitrage on BTCUSD

**Author:** Manus AI for AegisFintech  
**Date:** June 2026  
**Target Asset:** BTCUSD (Perpetual Futures)  
**Frequency:** 1-Minute Intraday  

---

## 1. Executive Summary

This report details the theoretical foundation and mathematical architecture of the AIQuant high-frequency trading (HFT) framework. The system is designed to exploit short-term microstructural inefficiencies and statistical mispricings in the BTCUSD perpetual futures market. 

By combining market microstructure signals (Order Flow Imbalance, VPIN) with advanced statistical filtering (Kalman Filters, Hurst Exponent) and Kelly Criterion position sizing, the framework aims to generate consistent alpha in both mean-reverting and trending market regimes.

## 2. Market Microstructure and Order Flow

Traditional technical analysis often fails in high-frequency domains due to noise. AIQuant relies on microstructure features that proxy the underlying limit order book dynamics [1].

### 2.1 Order Flow Imbalance (OFI)
OFI measures the net buying or selling pressure by tracking changes in the best bid and ask queues. In the absence of full Level-2 tick data, we approximate OFI using the tick rule on 1-minute volume:
$$ OFI_t = \sum_{i=1}^{n} \text{sign}(\Delta P_i) \times V_i $$
Where $\Delta P_i$ is the price change and $V_i$ is the volume. A positive OFI strongly predicts short-term upward price movement due to inventory depletion on the ask side.

### 2.2 Volume-Synchronized Probability of Informed Trading (VPIN)
VPIN estimates the toxicity of order flow [2]. High VPIN indicates that informed traders are aggressively taking liquidity, which often precedes volatility spikes or structural breaks. The framework uses VPIN as a **negative filter**—halting mean-reversion trades when VPIN exceeds a critical threshold to avoid adverse selection.

## 3. Statistical Arbitrage and Regime Detection

Statistical arbitrage relies on the assumption that prices will revert to a historical mean. However, blindly applying mean-reversion is dangerous if the market undergoes a structural regime shift.

### 3.1 Regime Detection: The Hurst Exponent
The Hurst Exponent ($H$) characterizes the autocorrelation of a time series:
- $H < 0.5$: Mean-reverting (anti-persistent)
- $H \approx 0.5$: Random walk
- $H > 0.5$: Trending (persistent)

AIQuant dynamically routes signals based on a rolling Hurst calculation. The primary Statistical Arbitrage strategy is only active when $H < 0.45$.

### 3.2 Dynamic Mean Estimation: Kalman Filter
Instead of using static moving averages (which suffer from lag), the framework uses a 1D Kalman Filter to estimate the "true" underlying price level. The Kalman residual (innovation) represents the instantaneous mispricing:
$$ y_t = P_t - \hat{P}_{t|t-1} $$
When the z-score of $y_t$ exceeds a threshold (e.g., $\pm 2.0$), a trade is initiated, betting on the residual reverting to zero.

## 4. Machine Learning Ensemble

To capture non-linear relationships between the 100+ engineered features, AIQuant includes a machine learning ensemble.

- **Architecture:** Soft-voting ensemble comprising XGBoost, LightGBM, Random Forest, and Logistic Regression.
- **Target:** 3-class classification (Long, Short, Flat) based on forward 15-minute returns exceeding a noise threshold.
- **Validation:** Walk-Forward Cross-Validation (WFA) is strictly enforced. The model is trained on an expanding window and tested on the immediate out-of-sample period to prevent lookahead bias and adapt to concept drift.

## 5. Risk Management: The Kelly Criterion

Position sizing is mathematically optimized using the Kelly Criterion, which maximizes the geometric growth rate of the portfolio [3].

### 5.1 Formula
$$ f^* = \frac{p \cdot b - q}{b} $$
Where:
- $p$ = Probability of a winning trade
- $q$ = Probability of a losing trade ($1 - p$)
- $b$ = Ratio of average win to average loss

### 5.2 Implementation
Because the full Kelly fraction can lead to extreme volatility and assumes perfect knowledge of $p$ and $b$, AIQuant implements a **Half-Kelly** approach ($f^* / 2$). Furthermore, the fraction is dynamically scaled inversely to current market volatility (Realized Variance), ensuring smaller positions during turbulent periods.

## 6. Execution on Hyperliquid

Hyperliquid is utilized as the execution venue due to its on-chain matching engine, zero gas fees for trading, and sub-second latency. The `LiveTradingOrchestrator` runs a continuous loop that:
1. Ingests the latest 1m bar.
2. Computes the feature vector.
3. Queries the `StrategyEnsemble` for the final signal.
4. Sizes the position via the `RiskManager`.
5. Executes an Immediate-Or-Cancel (IOC) market order via the Hyperliquid SDK.

## References

[1] Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223-236.  
[2] Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *The Review of Financial Studies*, 25(5), 1457-1493.  
[3] Thorp, E. O. (2006). The Kelly criterion in blackjack, sports betting, and the stock market. *Handbook of Asset and Liability Management*, 385-428.
