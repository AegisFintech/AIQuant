#!/usr/bin/env python3
"""
AIQuant — HFT Statistical Arbitrage Framework
==============================================
AegisFintech | Apache 2.0 License

Usage:
    python3 run.py backtest
    python3 run.py backtest --pair ETHUSDT --days 60
    python3 run.py paper
    python3 run.py paper --pair BTCUSDT --bars 50
    python3 run.py fetch --pair BTCUSDT --days 90

Defaults (when no flags given):
    --pair   BTCUSDT
    --days   90  (T-90 days of 1m data = 129,600 bars)
"""

import sys
import os
import time
import logging
import argparse
import subprocess
import platform
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / 'results'
DATA_DIR    = ROOT / 'data' / 'raw'
LOGS_DIR    = ROOT / 'logs' / 'paper_trading'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # Suppress library noise
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('aiquant')
logger.setLevel(logging.INFO)

# ── Colour helpers (works on Linux/Mac; gracefully degrades on Windows) ─────
def _c(code, text):
    if platform.system() == 'Windows':
        return text
    return f'\033[{code}m{text}\033[0m'

BOLD  = lambda t: _c('1',    t)
DIM   = lambda t: _c('2',    t)
GREEN = lambda t: _c('32',   t)
RED   = lambda t: _c('31',   t)
CYAN  = lambda t: _c('36',   t)
YELLOW= lambda t: _c('33',   t)
WHITE = lambda t: _c('97',   t)

# ── Banner ───────────────────────────────────────────────────────────────────
BANNER = f"""
{CYAN('╔══════════════════════════════════════════════════════════════╗')}
{CYAN('║')}  {BOLD(WHITE('AIQuant'))}  ·  HFT Statistical Arbitrage  ·  {DIM('AegisFintech')}      {CYAN('║')}
{CYAN('║')}  {DIM('Apache 2.0  ·  github.com/AegisFintech/AIQuant')}             {CYAN('║')}
{CYAN('╚══════════════════════════════════════════════════════════════╝')}
"""

def banner(mode: str, pair: str, days: int = None):
    print(BANNER)
    mode_str = {'backtest': '📊  BACKTEST', 'paper': '🔴  PAPER TRADING', 'fetch': '📥  DATA FETCH'}.get(mode, mode.upper())
    print(f"  Mode  : {BOLD(mode_str)}")
    print(f"  Pair  : {BOLD(CYAN(pair))}")
    if days:
        print(f"  Window: {BOLD(str(days))} days  ({days * 1440:,} 1m bars)")
    print()


# ════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ════════════════════════════════════════════════════════════════════════════

def fetch_data(pair: str = 'BTCUSDT', days: int = 90, force: bool = False) -> pd.DataFrame:
    """
    Fetch 1m OHLCV bars from Binance public API.
    Caches to parquet — skips re-fetch if fresh data exists (< 1h old).
    """
    cache_path = DATA_DIR / f'{pair}_1m.parquet'

    # Use cache if fresh enough
    if not force and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 1:
            print(f"  {GREEN('✓')} Using cached data  {DIM(f'({cache_path.name}, {age_hours:.1f}h old)')}")
            df = pd.read_parquet(cache_path)
            # Trim to requested window
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
            df = df[df.index >= cutoff]
            d0 = df.index[0].strftime('%Y-%m-%d')
            d1 = df.index[-1].strftime('%Y-%m-%d')
            print(f"  {GREEN('✓')} Loaded {len(df):,} bars  {DIM(d0 + ' → ' + d1)}")
            return df

    print(f"  {CYAN('↓')} Fetching {days} days of {pair} 1m data from Binance...")
    url = "https://api.binance.com/api/v3/klines"
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_bars = []
    current  = start_ms
    total_expected = days * 1440

    while current < end_ms:
        try:
            resp = requests.get(url, params={
                'symbol': pair, 'interval': '1m',
                'startTime': current, 'endTime': end_ms, 'limit': 1000,
            }, timeout=15)
            resp.raise_for_status()
            bars = resp.json()
            if not bars:
                break
            all_bars.extend(bars)
            current = bars[-1][0] + 60_000
            pct = len(all_bars) / total_expected * 100
            bar_fill = int(pct / 2)
            bar_str  = '█' * bar_fill + '░' * (50 - bar_fill)
            ts = datetime.fromtimestamp(bars[-1][0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            print(f"\r  [{bar_str}] {pct:5.1f}%  {len(all_bars):>7,} bars  up to {ts} UTC", end='', flush=True)
            time.sleep(0.08)
        except Exception as e:
            print(f"\n  {RED('✗')} Fetch error: {e}")
            break

    print(f"\r  {GREEN('✓')} Fetched {len(all_bars):,} bars{' ' * 60}")

    cols = ['open_time','open','high','low','close','volume',
            'close_time','quote_vol','trades','taker_base','taker_quote','ignore']
    df = pd.DataFrame(all_bars, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df.set_index('open_time', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df = df[['open','high','low','close','volume']].copy()
    df.to_parquet(cache_path)
    print(f"  {GREEN('✓')} Saved to {DIM(str(cache_path))}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n  {CYAN('⚙')}  Building features...")
    d = df.copy()
    c, h, lo, v = d['close'], d['high'], d['low'], d['volume']

    # Returns
    d['ret_1']  = c.pct_change(1)
    d['ret_5']  = c.pct_change(5)
    d['ret_15'] = c.pct_change(15)

    # EMAs
    for w in [5, 10, 20, 50, 100, 200]:
        d[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()
    d['ema_cross_5_20']  = d['ema_5']  - d['ema_20']
    d['ema_cross_20_50'] = d['ema_20'] - d['ema_50']

    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d['bb_upper'] = bb_mid + 2 * bb_std
    d['bb_lower'] = bb_mid - 2 * bb_std
    d['bb_pct']   = (c - d['bb_lower']) / (d['bb_upper'] - d['bb_lower'] + 1e-9)
    d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / (bb_mid + 1e-9)

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d['macd']        = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist']   = d['macd'] - d['macd_signal']

    # ATR
    tr = pd.concat([(h - lo), (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    d['atr_14']  = tr.rolling(14).mean()
    d['atr_pct'] = d['atr_14'] / (c + 1e-9)

    # Volatility
    d['realvol_20'] = d['ret_1'].rolling(20).std()  * np.sqrt(1440)
    d['realvol_60'] = d['ret_1'].rolling(60).std()  * np.sqrt(1440)
    d['realvol_240']= d['ret_1'].rolling(240).std() * np.sqrt(1440)

    # Volume
    d['vol_ma_20'] = v.rolling(20).mean()
    d['vol_ratio'] = v / (d['vol_ma_20'] + 1e-9)

    # Order Flow Imbalance (approximation)
    price_chg = c.diff()
    d['ofi']      = np.where(price_chg > 0, v, np.where(price_chg < 0, -v, 0))
    d['ofi_ma_10']= pd.Series(d['ofi'], index=d.index).rolling(10).mean()

    # Kalman Filter
    prices   = c.values
    kf_m     = np.zeros(len(prices))
    kf_v     = np.ones(len(prices)) * 1000.0
    kf_m[0]  = prices[0]
    for i in range(1, len(prices)):
        kf_v[i] = kf_v[i-1] + 0.01
        K        = kf_v[i] / (kf_v[i] + 1.0)
        kf_m[i]  = kf_m[i-1] + K * (prices[i] - kf_m[i-1])
        kf_v[i]  = (1 - K) * kf_v[i]
    d['kalman_mean']     = kf_m
    d['kalman_residual'] = c - d['kalman_mean']
    d['kalman_zscore']   = (
        d['kalman_residual']
        / (d['kalman_residual'].rolling(60).std() + 1e-9)
    )

    # Hurst Exponent (rolling R/S)
    def hurst_rs(x):
        if len(x) < 20: return 0.5
        mean = x.mean(); dev = (x - mean).cumsum()
        r = dev.max() - dev.min(); s = x.std()
        return np.log(r / (s + 1e-9) + 1e-9) / np.log(len(x))

    d['hurst'] = d['ret_1'].rolling(100).apply(hurst_rs, raw=True)

    # Intraday seasonality
    d['hour_sin'] = np.sin(2 * np.pi * d.index.hour / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d.index.hour / 24)

    d.dropna(inplace=True)
    print(f"  {GREEN('✓')} {d.shape[1]} features  ·  {d.shape[0]:,} usable bars")
    return d


# ════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame) -> pd.Series:
    print(f"\n  {CYAN('⚙')}  Generating signals...")
    signals = pd.Series(0, index=df.index)
    hurst   = df['hurst'].fillna(0.5)
    mr      = hurst < 0.45
    trend   = hurst > 0.55

    # StatArb (mean-reverting)
    z   = df['kalman_zscore']
    rsi = df['rsi_14']
    ofi = df['ofi_ma_10']
    signals[mr & (z < -2.0) & (rsi < 40) & (ofi > 0)] =  1
    signals[mr & (z >  2.0) & (rsi > 60) & (ofi < 0)] = -1

    # Trend following
    ema_x = df['ema_cross_5_20']
    mh    = df['macd_hist']
    vr    = df['vol_ratio']
    signals[trend & (ema_x > 0) & (mh > 0) & (vr > 1.2)] =  1
    signals[trend & (ema_x < 0) & (mh < 0) & (vr > 1.2)] = -1

    n     = len(signals)
    longs = (signals ==  1).sum()
    shts  = (signals == -1).sum()
    mr_pct    = mr.mean()   * 100
    trend_pct = trend.mean()* 100

    print(f"  {GREEN('✓')} {n:,} bars  ·  "
          f"{GREEN(f'▲ {longs:,} long')}  ·  "
          f"{RED(f'▼ {shts:,} short')}  ·  "
          f"{DIM(f'mean-rev {mr_pct:.0f}%  trend {trend_pct:.0f}%')}")
    return signals


# ════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, signals: pd.Series, pair: str):
    import backtrader as bt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n  {CYAN('▶')}  Running Backtrader backtest...")
    d0b = df.index[0].strftime("%Y-%m-%d")
    d1b = df.index[-1].strftime("%Y-%m-%d")
    print(f"  {DIM(str(len(df)) + ' bars  ·  ' + d0b + ' → ' + d1b)}")

    class _Data(bt.feeds.PandasData):
        lines = ('signal',)
        params = (('signal', -1), ('openinterest', -1))

    class _Strat(bt.Strategy):
        params = (
            ('stop_loss',   0.012),
            ('take_profit', 0.025),
            ('max_hold',    60),
            ('pos_pct',     0.20),
        )
        def __init__(self):
            self.sig = self.data.signal
            self.order = None
            self.entry_px = None
            self.entry_bar = None
            self._trade_count = 0; self._wins = 0

        def notify_order(self, o):
            if o.status not in [o.Submitted, o.Accepted]:
                self.order = None

        def notify_trade(self, t):
            if t.isclosed:
                self._trade_count += 1
                if t.pnlcomm > 0: self._wins += 1
                if self._trade_count % 20 == 0:
                    val = self.broker.getvalue()
                    ret = (val - 100_000) / 100_000 * 100
                    wr  = self._wins / self._trade_count * 100
                    sign = GREEN(f'+{ret:.2f}%') if ret >= 0 else RED(f'{ret:.2f}%')
                    print(f"    Trade {self._trade_count:>4}  ·  "
                          f"PnL {'+' if t.pnlcomm>=0 else ''}{t.pnlcomm:>8.2f}  ·  "
                          f"WR {wr:.0f}%  ·  "
                          f"Portfolio ${val:>12,.2f}  ·  Return {sign}")

        def next(self):
            if self.order: return
            px  = self.data.close[0]
            sig = self.sig[0]
            sz  = (self.broker.getvalue() * self.params.pos_pct) / px

            if self.position:
                held = len(self) - (self.entry_bar or 0)
                pnl  = (px - (self.entry_px or px)) / (self.entry_px or px)
                if self.position.size > 0:
                    if pnl <= -self.params.stop_loss or pnl >= self.params.take_profit or held >= self.params.max_hold:
                        self.order = self.close(); return
                else:
                    if -pnl <= -self.params.stop_loss or -pnl >= self.params.take_profit or held >= self.params.max_hold:
                        self.order = self.close(); return

            if not self.position:
                if sig == 1:
                    self.order = self.buy(size=sz); self.entry_px = px; self.entry_bar = len(self)
                elif sig == -1:
                    self.order = self.sell(size=sz); self.entry_px = px; self.entry_bar = len(self)
            else:
                if self.position.size > 0 and sig == -1: self.order = self.close()
                elif self.position.size < 0 and sig == 1: self.order = self.close()

    df_bt = df[['open','high','low','close','volume']].copy()
    df_bt['signal'] = signals.reindex(df_bt.index).fillna(0)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.00035)
    cerebro.broker.set_slippage_perc(0.0001)
    cerebro.adddata(_Data(dataname=df_bt, datetime=None,
                          open='open', high='high', low='low',
                          close='close', volume='volume', signal='signal'))
    cerebro.addstrategy(_Strat)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name='sharpe', riskfreerate=0.04, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cerebro.addanalyzer(bt.analyzers.SQN,           _name='sqn')
    cerebro.addanalyzer(bt.analyzers.TimeReturn,    _name='tr', timeframe=bt.TimeFrame.Days)

    start_val = cerebro.broker.getvalue()
    results   = cerebro.run()
    end_val   = cerebro.broker.getvalue()
    strat     = results[0]

    sharpe  = strat.analyzers.sharpe.get_analysis().get('sharperatio') or 0
    dd      = strat.analyzers.dd.get_analysis()
    ta      = strat.analyzers.ta.get_analysis()
    sqn     = strat.analyzers.sqn.get_analysis().get('sqn') or 0
    tr_data = strat.analyzers.tr.get_analysis()

    total_ret   = (end_val - start_val) / start_val * 100
    max_dd      = dd.get('max', {}).get('drawdown', 0)
    total_trades= ta.get('total', {}).get('closed', 0)
    won         = ta.get('won',   {}).get('total', 0)
    win_rate    = won / total_trades * 100 if total_trades else 0
    avg_win     = ta.get('won',  {}).get('pnl', {}).get('average', 0)
    avg_loss    = ta.get('lost', {}).get('pnl', {}).get('average', 0)

    # Equity curve
    if tr_data:
        eq = pd.Series(tr_data).sort_index()
        equity = (1 + eq).cumprod() * 100_000
    else:
        equity = pd.Series([start_val, end_val])

    # ── Print results ────────────────────────────────────────────────────
    ret_str = GREEN(f'+{total_ret:.2f}%') if total_ret >= 0 else RED(f'{total_ret:.2f}%')
    wr_str  = GREEN(f'{win_rate:.1f}%')   if win_rate >= 50  else YELLOW(f'{win_rate:.1f}%')

    print(f"\n  {'─'*58}")
    print(f"  {BOLD('BACKTEST RESULTS')}  ·  {pair}")
    print(f"  {'─'*58}")
    print(f"  {'Initial Capital':<24} ${'100,000.00':>14}")
    print(f"  {'Final Value':<24} ${end_val:>14,.2f}")
    print(f"  {'Total Return':<24} {ret_str:>20}")
    print(f"  {'Sharpe Ratio':<24} {sharpe:>15.4f}")
    print(f"  {'Max Drawdown':<24} {RED(f'{max_dd:.2f}%'):>20}")
    print(f"  {'SQN':<24} {sqn:>15.4f}")
    print(f"  {'Total Trades':<24} {total_trades:>15,}")
    print(f"  {'Win Rate':<24} {wr_str:>20}")
    print(f"  {'Avg Win':<24} ${avg_win:>+14.2f}")
    print(f"  {'Avg Loss':<24} ${avg_loss:>+14.2f}")
    print(f"  {'─'*58}")

    # ── Plot ─────────────────────────────────────────────────────────────
    _plot_backtest(equity, signals, pair, total_ret, sharpe, max_dd, win_rate, total_trades)

    return {'total_return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd,
            'total_trades': total_trades, 'win_rate': win_rate, 'equity': equity}


def _plot_backtest(equity, signals, pair, ret, sharpe, max_dd, wr, trades):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#21262d')
        ax.yaxis.label.set_color('#8b949e')
        ax.xaxis.label.set_color('#8b949e')

    title = (f"AIQuant  ·  {pair} 1m  ·  "
             f"Return: {'+' if ret>=0 else ''}{ret:.2f}%  ·  "
             f"Sharpe: {sharpe:.2f}  ·  "
             f"Max DD: {max_dd:.2f}%  ·  "
             f"Win Rate: {wr:.1f}%  ·  "
             f"Trades: {trades}")
    fig.suptitle(title, fontsize=11, color='#c9d1d9', y=0.99)

    # Equity
    color = '#3fb950' if equity.iloc[-1] >= equity.iloc[0] else '#f85149'
    axes[0].plot(equity.values, color=color, linewidth=1.5)
    axes[0].axhline(y=100_000, color='#484f58', linestyle='--', alpha=0.6, label='Initial Capital')
    axes[0].set_title('Equity Curve', color='#c9d1d9', fontweight='bold', pad=8)
    axes[0].set_ylabel('Portfolio Value (USD)', color='#8b949e')
    axes[0].legend(facecolor='#161b22', edgecolor='#21262d', labelcolor='#8b949e')
    axes[0].grid(True, alpha=0.15, color='#30363d')

    # Drawdown
    eq_arr   = equity.values
    roll_max = np.maximum.accumulate(eq_arr)
    dd_arr   = (eq_arr - roll_max) / roll_max * 100
    axes[1].fill_between(range(len(dd_arr)), dd_arr, 0, color='#f85149', alpha=0.6)
    axes[1].set_title('Drawdown (%)', color='#c9d1d9', fontweight='bold', pad=8)
    axes[1].set_ylabel('Drawdown (%)', color='#8b949e')
    axes[1].grid(True, alpha=0.15, color='#30363d')

    # Signal distribution
    vc = signals.value_counts().sort_index()
    clrs = {-1: '#f85149', 0: '#484f58', 1: '#3fb950'}
    axes[2].bar([str(k) for k in vc.index], vc.values,
                color=[clrs.get(k, '#8b949e') for k in vc.index], width=0.5)
    axes[2].set_title('Signal Distribution', color='#c9d1d9', fontweight='bold', pad=8)
    axes[2].set_xlabel('Signal  (−1 Short  ·  0 Flat  ·  1 Long)', color='#8b949e')
    axes[2].set_ylabel('Count', color='#8b949e')
    axes[2].grid(True, alpha=0.15, color='#30363d', axis='y')

    plt.tight_layout()
    out = RESULTS_DIR / 'backtest_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\n  {GREEN('✓')} Chart saved → {DIM(str(out))}")
    _open_chart(out)


def _open_chart(path: Path):
    """Auto-open chart if running on a desktop. Print path if headless."""
    system = platform.system()
    try:
        if system == 'Darwin':
            subprocess.Popen(['open', str(path)])
            print(f"  {GREEN('✓')} Chart opened in Preview")
        elif system == 'Linux':
            # Try common viewers; silently skip if none available
            for viewer in ['xdg-open', 'eog', 'feh', 'display']:
                if subprocess.run(['which', viewer], capture_output=True).returncode == 0:
                    subprocess.Popen([viewer, str(path)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"  {GREEN('✓')} Chart opened with {viewer}")
                    return
            # Headless server — just print the path
            print(f"  {YELLOW('ℹ')}  Headless server detected.")
            print(f"      To view: {BOLD('scp')} or copy {BOLD(str(path))} to your local machine.")
            print(f"      Or run:  {BOLD('python3 run.py serve')} to start a local viewer.")
        elif system == 'Windows':
            os.startfile(str(path))
            print(f"  {GREEN('✓')} Chart opened")
    except Exception:
        print(f"  {YELLOW('ℹ')}  Chart saved to: {BOLD(str(path))}")


# ════════════════════════════════════════════════════════════════════════════
# PAPER TRADING
# ════════════════════════════════════════════════════════════════════════════

def run_paper(pair: str = 'BTCUSDT', bars: int = 30, capital: float = 10_000.0):
    """
    Paper trading demo — replays recent bars at 2s intervals.
    For continuous live trading (60s poll), pass bars=0.
    """
    from aiquant.execution.paper_trader import PaperTradingEngine

    engine = PaperTradingEngine(
        initial_capital=capital,
        kelly_fraction=0.5,
        log_dir=str(LOGS_DIR),
    )

    def _signal(df: pd.DataFrame) -> int:
        c = df['close']
        prices = c.values
        kf_m, kf_v = prices[0], 1000.0
        for p in prices[1:]:
            kf_v += 0.01; K = kf_v / (kf_v + 1.0)
            kf_m = kf_m + K * (p - kf_m); kf_v = (1 - K) * kf_v
        res = prices[-1] - kf_m
        std = c.pct_change().rolling(60).std().iloc[-1] * c.iloc[-1]
        z   = res / (std + 1e-9)
        delta = c.diff()
        rsi = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean().iloc[-1] /
                           ((-delta.clip(upper=0)).rolling(14).mean().iloc[-1] + 1e-9))
        rets = c.pct_change().dropna().tail(100)
        hurst = 0.5
        if len(rets) >= 20:
            mean = rets.mean(); dev = (rets - mean).cumsum()
            r = dev.max() - dev.min(); s = rets.std()
            hurst = np.log(r / (s + 1e-9) + 1e-9) / np.log(len(rets))
        if hurst < 0.45:
            if z < -2.0 and rsi < 42: return 1
            if z >  2.0 and rsi > 58: return -1
        elif hurst > 0.55:
            ema5  = c.ewm(span=5).mean().iloc[-1]
            ema20 = c.ewm(span=20).mean().iloc[-1]
            if ema5 > ema20: return 1
            if ema5 < ema20: return -1
        return 0

    if bars > 0:
        # Replay mode — use last N bars from cached data
        df_raw  = fetch_data(pair=pair, days=3)
        df_feat = df_raw.tail(bars + 300)
        print(f"\n  {CYAN('▶')}  Replaying {bars} bars at 2s intervals...\n")

        bar_count = 0
        for i in range(300, len(df_feat)):
            window = df_feat.iloc[:i+1]
            ts     = str(window.index[-1])
            price  = window['close'].iloc[-1]
            signal = _signal(window)

            if engine.position:
                engine.position.update_pnl(price)
                engine.portfolio_value = engine.cash + engine.position.unrealised_pnl

            can_trade, reason = engine._can_trade()
            if not can_trade:
                if engine.position: engine._close_position(price, ts, f"RISK: {reason}")
            else:
                if signal == 1 and (engine.position is None or engine.position.side == 'short'):
                    if engine.position: engine._close_position(price, ts, 'Reversal')
                    engine._open_position('long', price, ts)
                elif signal == -1 and (engine.position is None or engine.position.side == 'long'):
                    if engine.position: engine._close_position(price, ts, 'Reversal')
                    engine._open_position('short', price, ts)
                elif signal == 0 and engine.position:
                    engine._close_position(price, ts, 'Flat')

            engine.equity_curve.append({'timestamp': ts, 'price': price,
                                        'portfolio_value': engine.portfolio_value,
                                        'signal': signal,
                                        'position': engine.position.side if engine.position else 'flat'})
            engine._print_dashboard(ts, price, signal, bar_count)
            engine._save_state()
            bar_count += 1
            if bar_count >= bars: break
            time.sleep(2)
    else:
        # Live mode — poll Binance every 60s
        engine.run(_signal, poll_interval_sec=60.0, lookback=500)

    engine._print_final_summary()
    print(f"\n  {GREEN('✓')} Trade log → {DIM(str(LOGS_DIR / 'paper_trades.csv'))}")


# ════════════════════════════════════════════════════════════════════════════
# SERVE (optional chart viewer)
# ════════════════════════════════════════════════════════════════════════════

def run_serve(port: int = 8765):
    import http.server, socketserver, webbrowser, threading
    os.chdir(RESULTS_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # Suppress access logs
    with socketserver.TCPServer(('', port), handler) as httpd:
        url = f'http://localhost:{port}/viewer.html'
        print(f"\n  {GREEN('✓')} Chart viewer running at {BOLD(CYAN(url))}")
        print(f"  {DIM('Press Ctrl+C to stop.')}\n")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n  {YELLOW('ℹ')}  Server stopped.")


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
  python3 run.py backtest                          # BTC, last 90 days
  python3 run.py backtest --pair ETHUSDT           # ETH, last 90 days
  python3 run.py backtest --pair BTCUSDT --days 30 # BTC, last 30 days
  python3 run.py paper                             # BTC paper trade, 30 bars replay
  python3 run.py paper --bars 0                    # BTC live paper trade (60s poll)
  python3 run.py fetch --days 180                  # Pre-fetch 180 days of BTC data
  python3 run.py serve                             # Open chart viewer in browser
        """,
    )
    parser.add_argument('mode', choices=['backtest', 'paper', 'fetch', 'serve'],
                        help='What to run')
    parser.add_argument('--pair',    default='BTCUSDT',
                        help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--days',    type=int, default=90,
                        help='Days of history to use (default: 90)')
    parser.add_argument('--bars',    type=int, default=30,
                        help='Bars for paper trading replay (0 = live, default: 30)')
    parser.add_argument('--capital', type=float, default=10_000.0,
                        help='Starting capital for paper trading (default: 10000)')
    parser.add_argument('--force',   action='store_true',
                        help='Force re-fetch data even if cache exists')
    parser.add_argument('--port',    type=int, default=8765,
                        help='Port for serve mode (default: 8765)')
    args = parser.parse_args()

    # Normalise pair
    pair = args.pair.upper().replace('/', '')
    if not pair.endswith('USDT') and not pair.endswith('USDC') and not pair.endswith('BTC'):
        pair = pair + 'USDT'

    banner(args.mode, pair, args.days if args.mode != 'serve' else None)

    if args.mode == 'fetch':
        fetch_data(pair=pair, days=args.days, force=args.force)
        print(f"\n  {GREEN('✓')} Done.\n")

    elif args.mode == 'backtest':
        df_raw  = fetch_data(pair=pair, days=args.days, force=args.force)
        df_feat = build_features(df_raw)
        signals = generate_signals(df_feat)
        run_backtest(df_feat, signals, pair)
        print()

    elif args.mode == 'paper':
        run_paper(pair=pair, bars=args.bars, capital=args.capital)
        print()

    elif args.mode == 'serve':
        run_serve(port=args.port)


if __name__ == '__main__':
    main()
