"""
AIQuant — HFT Statistical Arbitrage Framework
==============================================
AegisFintech | Apache 2.0 License

Usage:
    python3 run.py backtest                          # BTC, last 90 days
    python3 run.py backtest --pair ETHUSDT           # different pair
    python3 run.py backtest --days 30                # shorter window
    python3 run.py backtest --pair ETH --days 60     # pair shorthand works too
    python3 run.py live                              # live trading on Hyperliquid mainnet

Defaults (when no flags given):
    --pair   BTCUSDT
    --days   90  (T-90 days of 1m data = 129,600 bars)

Data sources:
    Backtest : CryptoDataDownload (free, no API key, full history since 2017)
    Live     : Hyperliquid public API for market data + mainnet for execution
               Requires HYPERLIQUID_PRIVATE_KEY in .env
"""

import sys
import os
import time
import logging
import argparse
import platform
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / 'results'
DATA_DIR    = ROOT / 'data' / 'raw'
LOGS_DIR    = ROOT / 'logs'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('aiquant')
logger.setLevel(logging.INFO)

# ── Numba JIT warm-up ────────────────────────────────────────────────────────
try:
    from aiquant.utils.fast_math import warmup as _nb_warmup
    _nb_warmup()
except Exception:
    pass

# ── Colour helpers ────────────────────────────────────────────────────────────
def _c(code, text):
    if platform.system() == 'Windows':
        return text
    return f'\033[{code}m{text}\033[0m'

BOLD   = lambda t: _c('1',  t)
DIM    = lambda t: _c('2',  t)
GREEN  = lambda t: _c('32', t)
RED    = lambda t: _c('31', t)
CYAN   = lambda t: _c('36', t)
YELLOW = lambda t: _c('33', t)
WHITE  = lambda t: _c('97', t)

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = f"""
{CYAN('╔══════════════════════════════════════════════════════════════╗')}
{CYAN('║')}  {BOLD(WHITE('AIQuant'))}  ·  HFT Statistical Arbitrage  ·  {DIM('AegisFintech')}      {CYAN('║')}
{CYAN('║')}  {DIM('Apache 2.0  ·  github.com/AegisFintech/AIQuant')}             {CYAN('║')}
{CYAN('╚══════════════════════════════════════════════════════════════╝')}
"""

def banner(mode: str, pair: str, days: int = None):
    print(BANNER)
    mode_str = {
        'backtest': '📊  BACKTEST  (CryptoDataDownload)',
        'live':     '🔴  LIVE TRADING  (Hyperliquid Mainnet)',
    }.get(mode, mode.upper())
    print(f"  Mode  : {BOLD(mode_str)}")
    print(f"  Pair  : {BOLD(CYAN(pair))}")
    if days:
        print(f"  Window: {BOLD(str(days))} days  ({days * 1440:,} 1m bars)")
    print()


# ════════════════════════════════════════════════════════════════════════════
# PAIR NORMALISATION
# ════════════════════════════════════════════════════════════════════════════

VALID_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

def normalise_pair(raw: str) -> str:
    """Accept BTC, btc, BTCUSDT, btcusdt — always return e.g. BTCUSDT."""
    p = raw.upper().strip()
    if p in VALID_PAIRS:
        return p
    if not p.endswith('USDT'):
        p = p + 'USDT'
    if p not in VALID_PAIRS:
        print(f"  {YELLOW('⚠')}  Unknown pair '{raw}'. Defaulting to BTCUSDT.")
        return 'BTCUSDT'
    return p


# ════════════════════════════════════════════════════════════════════════════
# DATA FETCHING — CryptoDataDownload
# ════════════════════════════════════════════════════════════════════════════

def fetch_data(pair: str = 'BTCUSDT', days: int = 90, force: bool = False) -> pd.DataFrame:
    """
    Fetch 1m OHLCV bars from CryptoDataDownload.
    Streams only the required number of bars (tail-read) — no full 177MB download.
    Caches to parquet — skips re-fetch if fresh data exists (< 1h old).
    """
    from aiquant.data.fetcher import fetch_cdd_backtest

    cache_path = DATA_DIR / f'{pair}_1m_cdd.parquet'

    # Use cache if fresh (CDD is static — cache for 24h)
    if not force and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"  {GREEN('✓')} Using cached data  {DIM(f'({cache_path.name}, {age_hours:.1f}h old)')}")
            df = pd.read_parquet(cache_path)
            # CDD data may be historical (not live) — use tail slice, not date filter
            return df.iloc[-days * 1440:].copy()

    print(f"  {CYAN('↓')} Streaming {pair} 1m data from CryptoDataDownload...")
    print(f"    {DIM('(streaming last ' + str(days) + ' days — no full 177MB download)')}")

    df = fetch_cdd_backtest(pair=pair, days=days, cache_dir=DATA_DIR, force=force)

    print(f"  {GREEN('✓')} Loaded {len(df):,} bars  "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all 100+ features."""
    print(f"  {CYAN('⚙')}  Building features...")
    from aiquant.features.technical import generate_all_technical_features
    from aiquant.features.microstructure import generate_all_microstructure_features
    from aiquant.features.statarb import generate_all_statarb_features

    df = generate_all_technical_features(df)
    df = generate_all_microstructure_features(df)
    df = generate_all_statarb_features(df)
    df = df.dropna()
    print(f"  {GREEN('✓')} {len(df.columns)} features  ·  {len(df):,} usable bars")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate ensemble signals."""
    print(f"  {CYAN('⚙')}  Generating signals...")
    from aiquant.strategies.ensemble import StrategyEnsemble
    ensemble = StrategyEnsemble()
    signals  = ensemble.generate_signals(df)
    n_long   = int((signals['final_signal'] ==  1).sum())
    n_short  = int((signals['final_signal'] == -1).sum())
    regime   = signals.get('regime', pd.Series(['unknown'] * len(signals)))
    n_mr     = int((regime == 'mean_reverting').sum()) if 'regime' in signals.columns else 0
    n_tr     = int((regime == 'trending').sum())       if 'regime' in signals.columns else 0
    print(
        f"  {GREEN('✓')} {len(signals):,} bars  ·  "
        f"{CYAN('▲')} {n_long} long  ·  {RED('▼')} {n_short} short  ·  "
        f"mean-rev {n_mr/len(signals)*100:.0f}%  trend {n_tr/len(signals)*100:.0f}%"
    )
    return signals


# ════════════════════════════════════════════════════════════════════════════
# BACKTEST — Backtrader
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, pair: str, capital: float = 100_000):
    """Run Backtrader backtest and display results."""
    import backtrader as bt
    import warnings
    warnings.filterwarnings('ignore')

    class _Strat(bt.Strategy):
        def __init__(self):
            self._trade_count = 0
            self._wins  = 0
            self._pnls  = []
            self._curve = []

        def next(self):
            idx = len(self.data) - 1
            if idx >= len(signals):
                return
            sig = int(signals['final_signal'].iloc[idx])
            pos = self.position.size
            if sig == 1 and pos <= 0:
                self.close()
                self.buy(size=(self.broker.getcash() * 0.95) / self.data.close[0])
            elif sig == -1 and pos >= 0:
                self.close()
                self.sell(size=(self.broker.getcash() * 0.95) / self.data.close[0])
            elif sig == 0 and pos != 0:
                self.close()
            self._curve.append(self.broker.getvalue())

        def notify_trade(self, trade):
            if trade.isclosed:
                self._trade_count += 1
                self._pnls.append(trade.pnlcomm)
                if trade.pnlcomm > 0:
                    self._wins += 1
                n = self._trade_count
                if n % 20 == 0:
                    wr  = self._wins / n * 100
                    ret = (self.broker.getvalue() - capital) / capital * 100
                    pnl = self._pnls[-1]
                    sign = GREEN('+') if pnl >= 0 else RED('-')
                    print(
                        f"    Trade {n:>4}  ·  "
                        f"PnL {sign}{abs(pnl):>8.2f}  ·  "
                        f"WR {wr:.0f}%  ·  "
                        f"Portfolio {GREEN('$') if ret >= 0 else RED('$')} "
                        f"{self.broker.getvalue():>12,.2f}  ·  "
                        f"Return {GREEN('+') if ret >= 0 else RED('')}{ret:.2f}%"
                    )

    # Prepare data feed
    df_bt = df[['open', 'high', 'low', 'close', 'volume']].dropna().copy()
    if hasattr(df_bt.index, 'tz') and df_bt.index.tz is not None:
        df_bt.index = df_bt.index.tz_localize(None)

    feed    = bt.feeds.PandasData(dataname=df_bt)
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(feed)
    cerebro.addstrategy(_Strat)
    cerebro.broker.setcash(capital)
    cerebro.broker.setcommission(commission=0.00035)

    print(f"  {CYAN('▶')}  Running Backtrader backtest...")
    print(f"  {DIM(str(len(df_bt)) + ' bars  ·  ' + df_bt.index[0].strftime('%Y-%m-%d') + ' → ' + df_bt.index[-1].strftime('%Y-%m-%d'))}")

    results  = cerebro.run()
    strat    = results[0]
    final    = cerebro.broker.getvalue()
    ret_pct  = (final - capital) / capital * 100
    n_trades = strat._trade_count
    win_rate = strat._wins / n_trades * 100 if n_trades > 0 else 0
    pnl_arr  = np.array(strat._pnls)
    avg_win  = float(pnl_arr[pnl_arr > 0].mean()) if (pnl_arr > 0).any() else 0
    avg_loss = float(pnl_arr[pnl_arr < 0].mean()) if (pnl_arr < 0).any() else 0

    # Drawdown
    curve    = np.array(strat._curve) if strat._curve else np.array([capital, final])
    peak     = np.maximum.accumulate(curve)
    dd       = (curve - peak) / peak * 100
    max_dd   = float(dd.min())

    # Sharpe (annualised, 1m bars)
    if len(curve) > 1:
        r_arr  = np.diff(curve) / curve[:-1]
        sharpe = (np.mean(r_arr) / (np.std(r_arr) + 1e-10)) * np.sqrt(1440 * 365)
    else:
        sharpe = 0.0

    col = GREEN if ret_pct >= 0 else RED
    print()
    print(f"  {'─'*52}")
    print(f"  BACKTEST RESULTS  ·  {BOLD(pair)}")
    print(f"  {'─'*52}")
    print(f"  Initial Capital   {WHITE('$'):>4}{capital:>15,.2f}")
    print(f"  Final Value       {col('$'):>4}{final:>15,.2f}")
    print(f"  Total Return      {col(f'{ret_pct:>+.2f}%'):>19}")
    print(f"  Sharpe Ratio      {f'{sharpe:>18.4f}'}")
    print(f"  Max Drawdown      {RED(f'{max_dd:.2f}%'):>19}")
    print(f"  Total Trades      {n_trades:>18,}")
    print(f"  Win Rate          {f'{win_rate:.1f}%':>18}")
    print(f"  Avg Win           {GREEN('$'):>4}{avg_win:>+14.2f}")
    print(f"  Avg Loss          {RED('$'):>4}{avg_loss:>+14.2f}")
    print(f"  {'─'*52}")

    # Save chart
    _save_chart(df, signals, strat._curve, pair, ret_pct, n_trades, win_rate, max_dd, sharpe)

    return {
        'final_value': final, 'total_return_pct': ret_pct,
        'sharpe': sharpe, 'max_drawdown_pct': max_dd,
        'n_trades': n_trades, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss,
    }


# ════════════════════════════════════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════════════════════════════════════

def _save_chart(df, signals, equity_curve, pair, ret_pct, n_trades, win_rate, max_dd, sharpe):
    """Save a dark-mode performance chart to results/."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')
        gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

        def _ax(r, c, rs=1, cs=1):
            ax = fig.add_subplot(gs[r:r+rs, c:c+cs])
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#c9d1d9', labelsize=8)
            for sp in ax.spines.values():
                sp.set_color('#30363d')
            return ax

        close_arr = df['close'].to_numpy(dtype=np.float64)
        sig_arr   = signals['final_signal'].to_numpy(dtype=np.int8)

        # 1. Price + signals
        ax1 = _ax(0, 0, 1, 2)
        ax1.plot(close_arr, color='#58a6ff', linewidth=0.5, label=pair)
        li = np.where(sig_arr ==  1)[0]
        si = np.where(sig_arr == -1)[0]
        ax1.scatter(li, close_arr[li], marker='^', color='#3fb950', s=6, alpha=0.5, label='Long')
        ax1.scatter(si, close_arr[si], marker='v', color='#f85149', s=6, alpha=0.5, label='Short')
        ax1.set_title(f'{pair} Price + Signals', color='#c9d1d9', fontweight='bold', fontsize=9)
        ax1.legend(facecolor='#161b22', labelcolor='#c9d1d9', fontsize=7)
        ax1.grid(True, alpha=0.15, color='#30363d')

        # 2. Equity curve
        ax2 = _ax(0, 2)
        if equity_curve:
            eq = np.array(equity_curve)
            color = '#3fb950' if eq[-1] >= eq[0] else '#f85149'
            ax2.plot(eq, color=color, linewidth=1)
            ax2.fill_between(range(len(eq)), eq, eq[0], alpha=0.15, color=color)
        ax2.set_title('Equity Curve', color='#c9d1d9', fontweight='bold', fontsize=9)
        ax2.grid(True, alpha=0.15, color='#30363d')

        # 3. Drawdown
        ax3 = _ax(1, 0, 1, 2)
        if equity_curve:
            eq   = np.array(equity_curve)
            peak = np.maximum.accumulate(eq)
            dd   = (eq - peak) / peak * 100
            ax3.fill_between(range(len(dd)), dd, 0, color='#f85149', alpha=0.6)
            ax3.plot(dd, color='#f85149', linewidth=0.5)
        ax3.set_title('Drawdown (%)', color='#c9d1d9', fontweight='bold', fontsize=9)
        ax3.grid(True, alpha=0.15, color='#30363d')

        # 4. Hurst Exponent
        ax4 = _ax(1, 2)
        if 'hurst' in df.columns:
            h = df['hurst'].to_numpy()
            ax4.plot(h, color='#d2a8ff', linewidth=0.5)
            ax4.axhline(0.45, color='#3fb950', linestyle='--', alpha=0.6, linewidth=0.8)
            ax4.axhline(0.55, color='#f85149', linestyle='--', alpha=0.6, linewidth=0.8)
            ax4.set_ylim(0, 1)
        ax4.set_title('Hurst Exponent', color='#c9d1d9', fontweight='bold', fontsize=9)
        ax4.grid(True, alpha=0.15, color='#30363d')

        # 5. Kalman Z-Score
        ax5 = _ax(2, 0, 1, 2)
        if 'kalman_zscore' in df.columns:
            kz = df['kalman_zscore'].to_numpy()
            ax5.plot(kz, color='#ffa657', linewidth=0.4, alpha=0.8)
            ax5.axhline( 1.8, color='#3fb950', linestyle='--', alpha=0.6, linewidth=0.8)
            ax5.axhline(-1.8, color='#f85149', linestyle='--', alpha=0.6, linewidth=0.8)
            ax5.axhline(0, color='white', linestyle=':', alpha=0.3, linewidth=0.6)
            ax5.set_ylim(-6, 6)
        ax5.set_title('Kalman Z-Score (Entry Signal)', color='#c9d1d9', fontweight='bold', fontsize=9)
        ax5.grid(True, alpha=0.15, color='#30363d')

        # 6. Metrics table
        ax6 = _ax(2, 2)
        ax6.axis('off')
        col_ret = '#3fb950' if ret_pct >= 0 else '#f85149'
        rows = [
            ('Total Return',  f'{ret_pct:+.2f}%',  col_ret),
            ('Sharpe Ratio',  f'{sharpe:.3f}',      '#c9d1d9'),
            ('Max Drawdown',  f'{max_dd:.2f}%',     '#f85149'),
            ('Total Trades',  f'{n_trades:,}',       '#c9d1d9'),
            ('Win Rate',      f'{win_rate:.1f}%',    '#c9d1d9'),
            ('Data Source',   'CryptoDataDownload',  '#8b949e'),
            ('Execution',     'Hyperliquid',         '#8b949e'),
        ]
        for i, (label, val, vc) in enumerate(rows):
            y = 0.92 - i * 0.13
            ax6.text(0.05, y, label, color='#8b949e', fontsize=8, transform=ax6.transAxes)
            ax6.text(0.95, y, val,   color=vc,        fontsize=8, transform=ax6.transAxes, ha='right', fontweight='bold')

        fig.suptitle(
            f'AIQuant  ·  {pair}  ·  Backtest Results',
            color='#c9d1d9', fontsize=13, fontweight='bold', y=1.01
        )

        out = RESULTS_DIR / 'backtest_results.png'
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig)
        print(f"\n  {GREEN('✓')} Chart saved → {out}")

        # Try to open on desktop environments
        if platform.system() == 'Darwin':
            os.system(f'open "{out}"')
        elif platform.system() == 'Linux' and os.environ.get('DISPLAY'):
            os.system(f'xdg-open "{out}" 2>/dev/null &')
        elif platform.system() == 'Windows':
            os.startfile(str(out))
        else:
            print(f"  {DIM('  (headless server — copy chart from: ' + str(out) + ')')}")

    except Exception as e:
        print(f"  {YELLOW('⚠')}  Chart generation failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# LIVE TRADING — Hyperliquid Mainnet
# ════════════════════════════════════════════════════════════════════════════

def run_live(pair: str = 'BTCUSDT', capital: float = 10_000, poll: float = 60.0):
    """Start live trading on Hyperliquid mainnet."""
    from dotenv import load_dotenv
    load_dotenv()

    pk = os.getenv('HYPERLIQUID_PRIVATE_KEY', '')
    if not pk or pk.startswith('your_'):
        print(f"\n  {RED('✗')}  HYPERLIQUID_PRIVATE_KEY not set in .env")
        print(f"  {DIM('  1. Open .env and add your Hyperliquid private key')}")
        print(f"  {DIM('  2. Generate a wallet: python3 -c \"from eth_account import Account; a=Account.create(); print(a.key.hex())\"')}")
        print(f"  {DIM('  3. Fund your account at https://app.hyperliquid.xyz')}")
        sys.exit(1)

    coin = pair.replace('USDT', '')
    print(f"  {GREEN('✓')} Private key loaded")
    print(f"  {CYAN('▶')}  Starting live trading loop  (Ctrl+C to stop)")
    print(f"  {DIM('  Polling every ' + str(poll) + 's  ·  Max 25% position per trade')}")
    print()

    from aiquant.execution.live_trader import LiveTradingOrchestrator
    orchestrator = LiveTradingOrchestrator(
        pair              = pair,
        coin              = coin,
        initial_capital   = capital,
        kelly_fraction    = 0.5,
        poll_interval_sec = poll,
        log_dir           = str(LOGS_DIR / 'live_trading'),
    )
    orchestrator.start()


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
  python3 run.py backtest                      # BTC, last 90 days
  python3 run.py backtest --pair ETH           # Ethereum, last 90 days
  python3 run.py backtest --days 30            # BTC, last 30 days
  python3 run.py backtest --pair SOL --days 60 # Solana, last 60 days
  python3 run.py live                          # Live trading on Hyperliquid
  python3 run.py live --pair ETH --poll 30     # ETH, poll every 30s
        """
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    # ── backtest ──────────────────────────────────────────────────────────
    bt_p = sub.add_parser('backtest', help='Run historical backtest using CryptoDataDownload data')
    bt_p.add_argument('--pair',    default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    bt_p.add_argument('--days',    default=90, type=int, help='Days of history (default: 90)')
    bt_p.add_argument('--capital', default=100_000, type=float, help='Starting capital USD (default: 100000)')
    bt_p.add_argument('--force',   action='store_true', help='Force re-download even if cache exists')

    # ── live ──────────────────────────────────────────────────────────────
    lv_p = sub.add_parser('live', help='Start live trading on Hyperliquid mainnet')
    lv_p.add_argument('--pair',    default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    lv_p.add_argument('--capital', default=10_000, type=float, help='Starting capital USD (default: 10000)')
    lv_p.add_argument('--poll',    default=60.0, type=float, help='Poll interval in seconds (default: 60)')

    args = parser.parse_args()

    # Normalise pair for both modes
    pair = normalise_pair(args.pair)

    if args.mode == 'backtest':
        banner('backtest', pair, args.days)
        t0 = time.time()
        df      = fetch_data(pair=pair, days=args.days, force=args.force)
        df_feat = build_features(df)
        signals = generate_signals(df_feat)
        run_backtest(df_feat, signals, pair=pair, capital=args.capital)
        print(f"\n  {DIM('Total time: ' + f'{time.time()-t0:.1f}s')}")

    elif args.mode == 'live':
        banner('live', pair)
        run_live(pair=pair, capital=args.capital, poll=args.poll)


if __name__ == '__main__':
    main()
