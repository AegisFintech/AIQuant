"""
scripts/demo_run.py
====================
Live demo runner — shows both backtesting and paper trading in action.

Run with:
    python3 scripts/demo_run.py --mode backtest
    python3 scripts/demo_run.py --mode paper
    python3 scripts/demo_run.py --mode both
"""

import sys
import os
import logging
import argparse
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: Fetch real BTCUSD 1m data from Binance
# ============================================================

def fetch_btc_1m(days: int = 7, verbose: bool = True) -> pd.DataFrame:
    """
    Fetch BTCUSDT 1-minute bars from Binance public API.
    No API key required.
    """
    print("\n" + "═" * 65)
    print("  STEP 1: FETCHING REAL BTCUSD 1m DATA FROM BINANCE")
    print("═" * 65)

    url = "https://api.binance.com/api/v3/klines"
    all_bars = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    limit = 1000
    fetched = 0

    current_start = start_ms
    while current_start < end_ms:
        try:
            resp = requests.get(url, params={
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'startTime': current_start,
                'endTime': end_ms,
                'limit': limit,
            }, timeout=15)
            resp.raise_for_status()
            bars = resp.json()
            if not bars:
                break
            all_bars.extend(bars)
            current_start = bars[-1][0] + 60_000  # next minute
            fetched += len(bars)
            if verbose:
                ts = datetime.fromtimestamp(bars[-1][0] / 1000, tz=timezone.utc)
                print(f"  Fetched {fetched:>6} bars | Latest: {ts.strftime('%Y-%m-%d %H:%M')} UTC", end='\r')
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            break

    print(f"\n  ✓ Total bars fetched: {len(all_bars):,}")

    df = pd.DataFrame(all_bars, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore',
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df.set_index('open_time', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Save raw data
    raw_path = Path(__file__).parent.parent / 'data' / 'raw' / 'BTCUSDT_1m.parquet'
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path)
    print(f"  ✓ Saved to {raw_path}")
    print(f"  ✓ Date range: {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  ✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ============================================================
# STEP 2: Feature Engineering (inline, fast)
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact but powerful feature set for demo purposes.
    """
    print("\n" + "═" * 65)
    print("  STEP 2: FEATURE ENGINEERING")
    print("═" * 65)

    d = df.copy()
    c = d['close']
    h = d['high']
    lo = d['low']
    v = d['volume']

    # --- Returns ---
    d['ret_1'] = c.pct_change(1)
    d['ret_5'] = c.pct_change(5)
    d['ret_15'] = c.pct_change(15)

    # --- Moving Averages ---
    for w in [5, 10, 20, 50, 100, 200]:
        d[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()
    d['ema_cross_5_20'] = d['ema_5'] - d['ema_20']
    d['ema_cross_20_50'] = d['ema_20'] - d['ema_50']

    # --- Bollinger Bands ---
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d['bb_upper'] = bb_mid + 2 * bb_std
    d['bb_lower'] = bb_mid - 2 * bb_std
    d['bb_pct'] = (c - d['bb_lower']) / (d['bb_upper'] - d['bb_lower'] + 1e-9)
    d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / bb_mid

    # --- RSI ---
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d['rsi_14'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d['macd'] = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']

    # --- ATR ---
    tr = pd.concat([
        h - lo,
        (h - c.shift()).abs(),
        (lo - c.shift()).abs(),
    ], axis=1).max(axis=1)
    d['atr_14'] = tr.rolling(14).mean()
    d['atr_pct'] = d['atr_14'] / c

    # --- Volatility ---
    d['realvol_20'] = d['ret_1'].rolling(20).std() * np.sqrt(1440)
    d['realvol_60'] = d['ret_1'].rolling(60).std() * np.sqrt(1440)

    # --- Volume ---
    d['vol_ma_20'] = v.rolling(20).mean()
    d['vol_ratio'] = v / (d['vol_ma_20'] + 1e-9)

    # --- Order Flow Imbalance (approximation) ---
    price_change = c.diff()
    d['ofi'] = np.where(price_change > 0, v, np.where(price_change < 0, -v, 0))
    d['ofi_ma_10'] = pd.Series(d['ofi']).rolling(10).mean()

    # --- Kalman Filter (dynamic mean) ---
    print("  Computing Kalman Filter dynamic mean...")
    kf_mean = np.zeros(len(c))
    kf_var = np.ones(len(c)) * 1000
    Q = 0.01   # Process noise
    R = 1.0    # Observation noise
    prices = c.values
    for i in range(1, len(prices)):
        kf_var[i] = kf_var[i-1] + Q
        K = kf_var[i] / (kf_var[i] + R)
        kf_mean[i] = kf_mean[i-1] + K * (prices[i] - kf_mean[i-1])
        kf_var[i] = (1 - K) * kf_var[i]
    d['kalman_mean'] = kf_mean
    d['kalman_residual'] = c - d['kalman_mean']
    d['kalman_zscore'] = (
        d['kalman_residual']
        / (d['kalman_residual'].rolling(60).std() + 1e-9)
    )

    # --- Hurst Exponent (rolling, simplified R/S) ---
    print("  Computing rolling Hurst Exponent...")
    def hurst_rs(series):
        n = len(series)
        if n < 20:
            return 0.5
        mean = series.mean()
        deviation = (series - mean).cumsum()
        r = deviation.max() - deviation.min()
        s = series.std()
        if s == 0:
            return 0.5
        return np.log(r / s + 1e-9) / np.log(n)

    d['hurst'] = d['ret_1'].rolling(100).apply(hurst_rs, raw=True)

    d.dropna(inplace=True)
    print(f"  ✓ Features built: {d.shape[1]} columns | {d.shape[0]:,} rows")
    return d


# ============================================================
# STEP 3: Signal Generation
# ============================================================

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Regime-adaptive signal generation:
    - Mean-reverting regime (Hurst < 0.45): Kalman Z-score + RSI + OFI
    - Trending regime (Hurst > 0.55): EMA crossover + MACD
    - Neutral: no trade
    """
    print("\n" + "═" * 65)
    print("  STEP 3: SIGNAL GENERATION (Regime-Adaptive)")
    print("═" * 65)

    signals = pd.Series(0, index=df.index)

    # --- Regime classification ---
    hurst = df['hurst'].fillna(0.5)
    mr_regime = hurst < 0.45
    trend_regime = hurst > 0.55

    # --- StatArb signals (mean-reverting regime) ---
    z = df['kalman_zscore']
    rsi = df['rsi_14']
    ofi = df['ofi_ma_10']

    # Long: price too far below Kalman mean, RSI oversold, positive OFI
    statarb_long = (z < -1.8) & (rsi < 40) & (ofi > 0)
    # Short: price too far above Kalman mean, RSI overbought, negative OFI
    statarb_short = (z > 1.8) & (rsi > 60) & (ofi < 0)

    signals[mr_regime & statarb_long] = 1
    signals[mr_regime & statarb_short] = -1

    # --- Trend signals (trending regime) ---
    ema_cross = df['ema_cross_5_20']
    macd_hist = df['macd_hist']

    trend_long = (ema_cross > 0) & (macd_hist > 0) & (df['vol_ratio'] > 1.2)
    trend_short = (ema_cross < 0) & (macd_hist < 0) & (df['vol_ratio'] > 1.2)

    signals[trend_regime & trend_long] = 1
    signals[trend_regime & trend_short] = -1

    long_count = (signals == 1).sum()
    short_count = (signals == -1).sum()
    flat_count = (signals == 0).sum()
    total = len(signals)

    print(f"  ✓ Total bars:   {total:,}")
    print(f"  ✓ Long signals: {long_count:,}  ({long_count/total*100:.1f}%)")
    print(f"  ✓ Short signals:{short_count:,}  ({short_count/total*100:.1f}%)")
    print(f"  ✓ Flat:         {flat_count:,}  ({flat_count/total*100:.1f}%)")
    print(f"  ✓ Mean-rev bars:{mr_regime.sum():,}  ({mr_regime.mean()*100:.1f}%)")
    print(f"  ✓ Trend bars:   {trend_regime.sum():,}  ({trend_regime.mean()*100:.1f}%)")
    return signals


# ============================================================
# STEP 4: Backtesting with Backtrader
# ============================================================

def run_backtest(df: pd.DataFrame, signals: pd.Series):
    """Run the full Backtrader backtest."""
    print("\n" + "═" * 65)
    print("  STEP 4: BACKTESTING WITH BACKTRADER")
    print("═" * 65)

    try:
        import backtrader as bt
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # --- Custom Data Feed ---
        class BTCData(bt.feeds.PandasData):
            lines = ('signal',)
            params = (('signal', -1), ('openinterest', -1))

        # --- Strategy ---
        class SignalStrategy(bt.Strategy):
            params = (
                ('stop_loss_pct', 0.012),
                ('take_profit_pct', 0.025),
                ('max_hold_bars', 60),
                ('position_pct', 0.20),
            )

            def __init__(self):
                self.signal = self.data.signal
                self.order = None
                self.entry_price = None
                self.entry_bar = None
                self.trade_count = 0
                self.wins = 0
                self.losses = 0

            def notify_order(self, order):
                if order.status in [order.Submitted, order.Accepted]:
                    return
                self.order = None

            def notify_trade(self, trade):
                if trade.isclosed:
                    self.trade_count += 1
                    if trade.pnlcomm > 0:
                        self.wins += 1
                    else:
                        self.losses += 1
                    if self.trade_count % 10 == 0:
                        win_rate = self.wins / self.trade_count * 100
                        val = self.broker.getvalue()
                        ret = (val - 100_000) / 100_000 * 100
                        print(
                            f"  [Trade {self.trade_count:>4}] "
                            f"PnL: ${trade.pnlcomm:>+8.2f}  |  "
                            f"Win Rate: {win_rate:.1f}%  |  "
                            f"Portfolio: ${val:>12,.2f}  |  "
                            f"Return: {ret:>+.2f}%"
                        )

            def next(self):
                if self.order:
                    return

                price = self.data.close[0]
                sig = self.signal[0]
                val = self.broker.getvalue()
                size = (val * self.params.position_pct) / price

                if self.position:
                    bars_held = len(self) - (self.entry_bar or 0)
                    entry = self.entry_price or price
                    pnl_pct = (price - entry) / entry

                    if self.position.size > 0:
                        if (pnl_pct <= -self.params.stop_loss_pct
                                or pnl_pct >= self.params.take_profit_pct
                                or bars_held >= self.params.max_hold_bars):
                            self.order = self.close()
                            return
                    else:
                        if (-pnl_pct <= -self.params.stop_loss_pct
                                or -pnl_pct >= self.params.take_profit_pct
                                or bars_held >= self.params.max_hold_bars):
                            self.order = self.close()
                            return

                if not self.position:
                    if sig == 1:
                        self.order = self.buy(size=size)
                        self.entry_price = price
                        self.entry_bar = len(self)
                    elif sig == -1:
                        self.order = self.sell(size=size)
                        self.entry_price = price
                        self.entry_bar = len(self)
                else:
                    if self.position.size > 0 and sig == -1:
                        self.order = self.close()
                    elif self.position.size < 0 and sig == 1:
                        self.order = self.close()

        # Prepare data
        df_bt = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df_bt['signal'] = signals.reindex(df_bt.index).fillna(0)
        df_bt.index = pd.to_datetime(df_bt.index)

        print(f"  Running backtest on {len(df_bt):,} bars...")
        print(f"  Date range: {df_bt.index[0].strftime('%Y-%m-%d')} → {df_bt.index[-1].strftime('%Y-%m-%d')}")

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.broker.setcommission(commission=0.00035)
        cerebro.broker.set_slippage_perc(0.0001)

        feed = BTCData(
            dataname=df_bt,
            datetime=None,
            open='open', high='high', low='low',
            close='close', volume='volume', signal='signal',
        )
        cerebro.adddata(feed)
        cerebro.addstrategy(SignalStrategy)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.04, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return', timeframe=bt.TimeFrame.Days)

        start_val = cerebro.broker.getvalue()
        results = cerebro.run()
        end_val = cerebro.broker.getvalue()
        strat = results[0]

        # Extract metrics
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio') or 0
        dd = strat.analyzers.drawdown.get_analysis()
        ta = strat.analyzers.trades.get_analysis()
        sqn = strat.analyzers.sqn.get_analysis().get('sqn') or 0
        time_ret = strat.analyzers.time_return.get_analysis()

        total_return = (end_val - start_val) / start_val * 100
        max_dd = dd.get('max', {}).get('drawdown', 0)
        total_trades = ta.get('total', {}).get('closed', 0)
        won = ta.get('won', {}).get('total', 0)
        win_rate = won / total_trades * 100 if total_trades > 0 else 0
        avg_win = ta.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = ta.get('lost', {}).get('pnl', {}).get('average', 0)

        # Equity curve
        if time_ret:
            eq = pd.Series(time_ret).sort_index()
            equity = (1 + eq).cumprod() * 100_000
        else:
            equity = pd.Series([start_val, end_val])

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('AIQuant — BTCUSD 1m Backtest Results', fontsize=15, fontweight='bold')

        axes[0].plot(equity.values, color='#00b4d8', linewidth=1.5)
        axes[0].axhline(y=100_000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_title('Equity Curve', fontweight='bold')
        axes[0].set_ylabel('Portfolio Value (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        eq_arr = equity.values
        roll_max = np.maximum.accumulate(eq_arr)
        drawdown_arr = (eq_arr - roll_max) / roll_max * 100
        axes[1].fill_between(range(len(drawdown_arr)), drawdown_arr, 0, color='#e63946', alpha=0.7)
        axes[1].set_title('Drawdown (%)', fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)

        # Signal distribution
        sig_counts = signals.value_counts().sort_index()
        colors = {-1: '#e63946', 0: '#adb5bd', 1: '#2dc653'}
        bars = axes[2].bar(
            [str(k) for k in sig_counts.index],
            sig_counts.values,
            color=[colors.get(k, 'gray') for k in sig_counts.index]
        )
        axes[2].set_title('Signal Distribution', fontweight='bold')
        axes[2].set_xlabel('Signal (-1=Short, 0=Flat, 1=Long)')
        axes[2].set_ylabel('Count')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = RESULTS_DIR / 'backtest_results.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Print results
        print("\n" + "═" * 65)
        print("  BACKTEST RESULTS")
        print("═" * 65)
        print(f"  Initial Capital:  ${'100,000.00':>15}")
        print(f"  Final Value:      ${end_val:>15,.2f}")
        print(f"  Total Return:     {'+' if total_return >= 0 else ''}{total_return:>14.2f}%")
        print(f"  Sharpe Ratio:     {sharpe:>15.4f}")
        print(f"  Max Drawdown:     {max_dd:>14.2f}%")
        print(f"  SQN:              {sqn:>15.4f}")
        print(f"  Total Trades:     {total_trades:>15,}")
        print(f"  Win Rate:         {win_rate:>14.1f}%")
        print(f"  Avg Win:          ${avg_win:>+14.2f}")
        print(f"  Avg Loss:         ${avg_loss:>+14.2f}")
        print("═" * 65)
        print(f"  ✓ Chart saved to: {plot_path}")

        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'equity': equity,
            'plot_path': str(plot_path),
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        return None


# ============================================================
# STEP 5: Paper Trading Demo (5 bars, 5s intervals for demo)
# ============================================================

def run_paper_trading_demo(df: pd.DataFrame, n_bars: int = 10):
    """
    Run a paper trading demo using the last N bars of historical data
    (replayed at 5s intervals to simulate live trading visually).
    """
    print("\n" + "═" * 65)
    print("  STEP 5: PAPER TRADING DEMO (Replaying Live Bars)")
    print("═" * 65)

    from aiquant.execution.paper_trader import PaperTradingEngine

    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        kelly_fraction=0.5,
        log_dir='logs/paper_trading',
    )

    # Use last n_bars + 300 for feature warmup
    demo_df = df.tail(n_bars + 300).copy()

    def signal_gen(current_df: pd.DataFrame) -> int:
        """Inline signal generator for demo."""
        c = current_df['close']
        # Kalman filter
        prices = c.values
        kf_m = prices[0]
        kf_v = 1000.0
        for p in prices[1:]:
            kf_v += 0.01
            K = kf_v / (kf_v + 1.0)
            kf_m = kf_m + K * (p - kf_m)
            kf_v = (1 - K) * kf_v
        residual = prices[-1] - kf_m
        std = c.pct_change().rolling(60).std().iloc[-1] * c.iloc[-1]
        z = residual / (std + 1e-9)

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))

        # Hurst (simplified)
        rets = c.pct_change().dropna().tail(100)
        if len(rets) >= 20:
            mean = rets.mean()
            dev = (rets - mean).cumsum()
            r = dev.max() - dev.min()
            s = rets.std()
            hurst = np.log(r / (s + 1e-9) + 1e-9) / np.log(len(rets))
        else:
            hurst = 0.5

        if hurst < 0.45:
            if z < -1.8 and rsi < 42:
                return 1
            elif z > 1.8 and rsi > 58:
                return -1
        elif hurst > 0.55:
            ema5 = c.ewm(span=5).mean().iloc[-1]
            ema20 = c.ewm(span=20).mean().iloc[-1]
            if ema5 > ema20:
                return 1
            elif ema5 < ema20:
                return -1
        return 0

    print(f"  Replaying {n_bars} bars at 2s intervals (simulating live 1m feed)...")
    print(f"  Initial Capital: ${engine.initial_capital:,.2f}\n")

    bar_count = 0
    for i in range(300, len(demo_df)):
        window = demo_df.iloc[:i+1]
        ts = str(window.index[-1])
        price = window['close'].iloc[-1]

        signal = signal_gen(window)

        # Update PnL
        if engine.position:
            engine.position.update_pnl(price)
            engine.portfolio_value = engine.cash + engine.position.unrealised_pnl

        can_trade, reason = engine._can_trade()

        if not can_trade:
            if engine.position:
                engine._close_position(price, ts, reason=f"RISK: {reason}")
        else:
            if signal == 1 and (engine.position is None or engine.position.side == 'short'):
                if engine.position:
                    engine._close_position(price, ts, reason='Signal Reversal')
                engine._open_position('long', price, ts)
            elif signal == -1 and (engine.position is None or engine.position.side == 'long'):
                if engine.position:
                    engine._close_position(price, ts, reason='Signal Reversal')
                engine._open_position('short', price, ts)
            elif signal == 0 and engine.position:
                engine._close_position(price, ts, reason='Signal Flat')

        engine.equity_curve.append({
            'timestamp': ts,
            'price': price,
            'portfolio_value': engine.portfolio_value,
            'signal': signal,
            'position': engine.position.side if engine.position else 'flat',
        })

        engine._print_dashboard(ts, price, signal, bar_count)
        engine._save_state()

        bar_count += 1
        if bar_count >= n_bars:
            break
        time.sleep(2)  # 2s per bar for visual demo

    engine._print_final_summary()
    return engine


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIQuant Demo Runner')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'both'], default='both')
    parser.add_argument('--days', type=int, default=7, help='Days of 1m data to fetch')
    parser.add_argument('--paper-bars', type=int, default=20, help='Bars for paper trading demo')
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  AIQuant — HFT Statistical Arbitrage Framework")
    print("  AegisFintech | BTCUSD 1m | Live Demo")
    print("═" * 65)

    # Fetch data
    df_raw = fetch_btc_1m(days=args.days)

    # Build features
    df_feat = build_features(df_raw)

    # Generate signals
    signals = generate_signals(df_feat)

    if args.mode in ('backtest', 'both'):
        run_backtest(df_feat, signals)

    if args.mode in ('paper', 'both'):
        run_paper_trading_demo(df_feat, n_bars=args.paper_bars)

    print("\n  ✓ Demo complete. Results saved to results/ and logs/")
