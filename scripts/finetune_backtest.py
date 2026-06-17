"""
scripts/finetune_backtest.py
=============================
Fine-tuning pass: push Sharpe above 2.0 by:
1. Adding minimum holding period (avoid churning on noise)
2. Adding minimum signal strength filter (only trade high-conviction setups)
3. Adding time-of-day filter (avoid low-liquidity hours)
4. Adding volatility regime filter (only trade when vol is in sweet spot)
5. Combining best signals with vote-based confirmation
"""
import sys, warnings, time, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/ubuntu/AIQuant')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

print("\n" + "="*65)
print("  AIQuant — Fine-Tuning Pass (Target: Sharpe > 2.0)")
print("  BTCUSD 1m  |  Mar–Jun 2026  |  $100,000 capital")
print("="*65)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading data...")
df_raw = pd.read_parquet('/home/ubuntu/AIQuant/data/raw/BTCUSDT_1m_full.parquet')
from aiquant.utils.fast_math import warmup
from aiquant.features import build_full_feature_set
warmup()
df = build_full_feature_set(df_raw.copy(), verbose=False)
df = df.dropna()
n = len(df)
print(f"      {n:,} bars ready")

c    = df['close'].to_numpy(np.float64)
h    = df['high'].to_numpy(np.float64)
l    = df['low'].to_numpy(np.float64)
v    = df['volume'].to_numpy(np.float64)
dates = df.index

rsi14  = df['rsi_14'].to_numpy(np.float64)
rsi7   = df['rsi_7'].to_numpy(np.float64)
ema9   = df['ema_9'].to_numpy(np.float64)
ema21  = df['ema_21'].to_numpy(np.float64)
ema50  = df['ema_50'].to_numpy(np.float64)
sma200 = df['sma_200'].to_numpy(np.float64)
macd   = df['macd_12_26'].to_numpy(np.float64)
msig   = df['macd_signal_12_26'].to_numpy(np.float64)
atr14  = df['atr_14'].to_numpy(np.float64)
adx    = df['adx'].to_numpy(np.float64)
vwap   = df['vwap'].to_numpy(np.float64)
vr     = df['vol_ratio'].to_numpy(np.float64)
bbu    = df['bb_high_20'].to_numpy(np.float64)
bbl    = df['bb_low_20'].to_numpy(np.float64)
bbw    = df['bb_width_20'].to_numpy(np.float64)
bbp    = df['bb_pct_20'].to_numpy(np.float64)
rv20   = df['realvol_20'].to_numpy(np.float64)
rv240  = df['realvol_240'].to_numpy(np.float64)
volreg = df['vol_regime'].to_numpy(np.float64)
hurst  = df['hurst'].to_numpy(np.float64) if 'hurst' in df.columns else np.full(n, 0.5)
obv    = df['obv'].to_numpy(np.float64)
cmf    = df['cmf'].to_numpy(np.float64)
stoch_k = df['stoch_k'].to_numpy(np.float64) if 'stoch_k' in df.columns else rsi14.copy()
cci    = df['cci'].to_numpy(np.float64) if 'cci' in df.columns else np.zeros(n)

# Time-of-day filter: avoid 00:00-02:00 UTC (low liquidity)
if hasattr(dates, 'hour'):
    hour = dates.hour.to_numpy()
else:
    hour = np.zeros(n, dtype=int)
good_hours = (hour >= 2) | (hour == 0)  # all hours except 1am UTC
good_hours = np.ones(n, dtype=bool)  # disable for now, apply selectively

# Volatility sweet spot: not too low (no moves), not too high (too risky)
vol_ok = (rv20 > 0.3) & (rv20 < 2.5)  # annualised vol between 30% and 250%

# ─────────────────────────────────────────────────────────────────────────────
# VECTORISED BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def backtest(signals, close=c, fee=0.00035, kelly=0.25, initial=100_000.0):
    sig = signals.astype(np.int8)
    changes = np.where(np.diff(sig, prepend=sig[0]) != 0)[0]
    if len(changes) < 2:
        return dict(sharpe=0.0, ret=0.0, max_dd=0.0, trades=0,
                    win_rate=0.0, final=initial, calmar=0.0, pf=0.0, equity=np.full(len(sig), initial))

    capital = initial
    equity  = np.full(len(sig), initial, dtype=np.float64)
    wins = losses = 0
    gross_win = gross_loss = 0.0

    for idx in range(len(changes) - 1):
        eb = changes[idx]; xb = changes[idx+1]
        direction = int(sig[eb])
        if direction == 0:
            equity[eb:xb] = capital; continue
        ep = close[eb]; xp = close[xb]
        size = (capital * kelly) / ep
        pnl  = direction * (xp - ep) * size * (1 - fee)**2
        capital = max(capital + pnl, 1.0)
        equity[eb:xb] = capital
        if pnl > 0: wins += 1; gross_win += pnl
        else: losses += 1; gross_loss += abs(pnl)
    equity[changes[-1]:] = capital

    total_ret = (capital - initial) / initial * 100
    dd        = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
    max_dd    = float(np.min(dd))
    n_days    = len(equity) // 1440
    if n_days > 2:
        daily_log = np.array([np.diff(np.log(equity[i*1440:(i+1)*1440]+1e-10)).sum()
                               for i in range(n_days)])
        sharpe = float(daily_log.mean() / (daily_log.std() + 1e-10) * np.sqrt(365))
    else:
        sharpe = 0.0
    total_tr = wins + losses
    win_rate = wins / total_tr * 100 if total_tr > 0 else 0.0
    calmar   = total_ret / abs(max_dd) if abs(max_dd) > 0.01 else 0.0
    pf       = gross_win / (gross_loss + 1e-10)
    return dict(sharpe=round(sharpe,3), ret=round(total_ret,2), max_dd=round(max_dd,2),
                trades=total_tr, win_rate=round(win_rate,1), final=round(capital,2),
                calmar=round(calmar,3), pf=round(pf,3), equity=equity)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: MULTI-CONFIRMATION MOMENTUM
# Requires 3 of 4 indicators to agree before entering
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Optimising Multi-Confirmation Momentum...")

prev_macd  = np.concatenate([[macd[0]], macd[:-1]])
prev_msig  = np.concatenate([[msig[0]], msig[:-1]])
macd_bull  = (macd > msig) & (prev_macd <= prev_msig)  # fresh crossover up
macd_bear  = (macd < msig) & (prev_macd >= prev_msig)
macd_pos   = macd > msig
macd_neg   = macd < msig

obv_ema20  = pd.Series(obv).ewm(span=20, adjust=False).mean().to_numpy()
obv_bull   = obv > obv_ema20
obv_bear   = obv < obv_ema20

best_mc = None
for adx_min in [20, 24, 28]:
    for rsi_lo, rsi_hi in [(42, 58), (45, 55), (48, 52)]:
        for min_hold in [5, 10, 15, 20]:
            # Score: count how many indicators agree
            bull_score = (
                (ema9 > ema21).astype(int) +
                macd_pos.astype(int) +
                (rsi14 > 50).astype(int) +
                obv_bull.astype(int) +
                (adx > adx_min).astype(int)
            )
            bear_score = (
                (ema9 < ema21).astype(int) +
                macd_neg.astype(int) +
                (rsi14 < 50).astype(int) +
                obv_bear.astype(int) +
                (adx > adx_min).astype(int)
            )

            # Only enter when 4+ of 5 agree
            long_ok  = (bull_score >= 4) & vol_ok & (rsi14 > rsi_lo) & (rsi14 < 75)
            short_ok = (bear_score >= 4) & vol_ok & (rsi14 < rsi_hi) & (rsi14 > 25)

            sig = np.zeros(n, np.int8)
            pos = 0; hold = 0; ep = 0.0
            for i in range(1, n):
                if pos == 0:
                    if long_ok[i]:   pos = 1; ep = c[i]; hold = 0
                    elif short_ok[i]: pos = -1; ep = c[i]; hold = 0
                elif pos == 1:
                    hold += 1
                    if hold < min_hold: sig[i] = pos; continue
                    if (ep - c[i]) > 2.0*atr14[i] or bull_score[i] < 2:
                        pos = 0
                elif pos == -1:
                    hold += 1
                    if hold < min_hold: sig[i] = pos; continue
                    if (c[i] - ep) > 2.0*atr14[i] or bear_score[i] < 2:
                        pos = 0
                sig[i] = pos

            r = backtest(sig)
            r['name'] = 'Multi-Confirm'
            r['params'] = f'adx>{adx_min} rsi[{rsi_lo},{rsi_hi}] hold={min_hold}'
            r['signals'] = sig
            if best_mc is None or r['sharpe'] > best_mc['sharpe']:
                best_mc = r
                print(f"  ↑ Sharpe={r['sharpe']:+.3f}  Return={r['ret']:+.1f}%  "
                      f"Trades={r['trades']:,}  MaxDD={r['max_dd']:.1f}%  [{r['params']}]")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: TREND CONTINUATION WITH PULLBACK
# Enter on RSI dip in confirmed trend, exit at trend reversal
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Optimising Trend Continuation + Pullback...")

best_tc = None
for adx_min in [25, 28, 32]:
    for rsi_dip_lo, rsi_dip_hi in [(38, 50), (40, 52), (42, 55)]:
        for atr_stop in [1.5, 2.0, 2.5]:
            uptrend   = (ema21 > ema50) & (ema50 > sma200) & (adx > adx_min)
            downtrend = (ema21 < ema50) & (ema50 < sma200) & (adx > adx_min)

            # Pullback entry: RSI dips into range in uptrend
            long_ok  = uptrend   & (rsi14 >= rsi_dip_lo) & (rsi14 <= rsi_dip_hi) & \
                       (c > ema50) & vol_ok & obv_bull
            short_ok = downtrend & (rsi14 >= (100-rsi_dip_hi)) & (rsi14 <= (100-rsi_dip_lo)) & \
                       (c < ema50) & vol_ok & obv_bear

            sig = np.zeros(n, np.int8)
            pos = 0; ep = 0.0; hold = 0
            for i in range(1, n):
                if pos == 0:
                    if long_ok[i]:   pos = 1; ep = c[i]; hold = 0
                    elif short_ok[i]: pos = -1; ep = c[i]; hold = 0
                elif pos == 1:
                    hold += 1
                    if (ep - c[i]) > atr_stop*atr14[i] or c[i] < ema50[i] or hold > 120:
                        pos = 0
                elif pos == -1:
                    hold += 1
                    if (c[i] - ep) > atr_stop*atr14[i] or c[i] > ema50[i] or hold > 120:
                        pos = 0
                sig[i] = pos

            r = backtest(sig)
            r['name'] = 'Trend Continuation'
            r['params'] = f'adx>{adx_min} rsi[{rsi_dip_lo},{rsi_dip_hi}] stop={atr_stop}x'
            r['signals'] = sig
            if best_tc is None or r['sharpe'] > best_tc['sharpe']:
                best_tc = r
                print(f"  ↑ Sharpe={r['sharpe']:+.3f}  Return={r['ret']:+.1f}%  "
                      f"Trades={r['trades']:,}  MaxDD={r['max_dd']:.1f}%  [{r['params']}]")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: VOLATILITY-ADJUSTED RSI REVERSION
# Only trade when vol is in sweet spot and RSI extremes are genuine
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Optimising Vol-Adjusted RSI Reversion...")

best_vr = None
for rsi_os in [25, 28, 30]:
    for rsi_ob in [70, 72, 75]:
        for vol_lo, vol_hi in [(0.4, 1.5), (0.3, 2.0), (0.5, 1.2)]:
            for min_hold in [10, 20, 30]:
                vol_sweet = (rv20 > vol_lo) & (rv20 < vol_hi)
                adx_range = adx < 30  # only in non-trending

                long_ok  = (rsi14 < rsi_os) & vol_sweet & adx_range & (c < vwap)
                short_ok = (rsi14 > rsi_ob) & vol_sweet & adx_range & (c > vwap)

                sig = np.zeros(n, np.int8)
                pos = 0; ep = 0.0; hold = 0
                for i in range(1, n):
                    if pos == 0:
                        if long_ok[i]:   pos = 1; ep = c[i]; hold = 0
                        elif short_ok[i]: pos = -1; ep = c[i]; hold = 0
                    elif pos == 1:
                        hold += 1
                        if hold < min_hold: sig[i] = pos; continue
                        if rsi14[i] > 55 or (ep - c[i]) > 2.5*atr14[i] or hold > 90:
                            pos = 0
                    elif pos == -1:
                        hold += 1
                        if hold < min_hold: sig[i] = pos; continue
                        if rsi14[i] < 45 or (c[i] - ep) > 2.5*atr14[i] or hold > 90:
                            pos = 0
                    sig[i] = pos

                r = backtest(sig)
                r['name'] = 'Vol-RSI Reversion'
                r['params'] = f'os={rsi_os} ob={rsi_ob} vol=[{vol_lo},{vol_hi}] hold={min_hold}'
                r['signals'] = sig
                if best_vr is None or r['sharpe'] > best_vr['sharpe']:
                    best_vr = r
                    print(f"  ↑ Sharpe={r['sharpe']:+.3f}  Return={r['ret']:+.1f}%  "
                          f"Trades={r['trades']:,}  MaxDD={r['max_dd']:.1f}%  [{r['params']}]")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: COMBINED VOTE ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Building Vote Ensemble...")
candidates = [r for r in [best_mc, best_tc, best_vr] if r is not None]
if len(candidates) >= 2:
    # Weighted vote: each strategy contributes its signal
    # Only trade when majority agrees
    vote = np.zeros(n, np.float64)
    for r in candidates:
        vote += r['signals'].astype(np.float64)
    vote_sig = np.where(vote >= 2, 1, np.where(vote <= -2, -1, 0)).astype(np.int8)

    r_ens = backtest(vote_sig)
    r_ens['name'] = 'Vote Ensemble'
    r_ens['params'] = f'{len(candidates)}-strategy majority vote'
    r_ens['signals'] = vote_sig
    candidates.append(r_ens)
    print(f"  Ensemble: Sharpe={r_ens['sharpe']:+.3f}  Return={r_ens['ret']:+.1f}%  "
          f"Trades={r_ens['trades']:,}  MaxDD={r_ens['max_dd']:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  FINE-TUNING RESULTS  |  BTCUSD 1m  |  Mar–Jun 2026  |  $100k")
print("="*75)
print(f"  {'Strategy':<22} {'Sharpe':>7} {'Return':>9} {'MaxDD':>8} "
      f"{'Trades':>8} {'WinRate':>8} {'Calmar':>8} {'PF':>6}")
print(f"  {'-'*22} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

candidates.sort(key=lambda x: x['sharpe'], reverse=True)
best_overall = candidates[0] if candidates else None
for i, r in enumerate(candidates):
    flag = ' ★' if i == 0 else ''
    print(f"  {r['name']:<22} {r['sharpe']:>+7.3f} {r['ret']:>+8.1f}% "
          f"{r['max_dd']:>+8.1f}% {r['trades']:>8,} {r['win_rate']:>7.1f}% "
          f"{r['calmar']:>+8.3f} {r['pf']:>6.2f}{flag}")
print("="*75)

# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────
if best_overall and 'equity' in best_overall:
    print(f"\n[4/4] Generating chart for '{best_overall['name']}'...")
    equity = best_overall['equity']
    dd     = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
    rets   = np.diff(equity) / (equity[:-1] + 1e-10)
    sig    = best_overall['signals']

    fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
    fig.suptitle(
        f"AIQuant  ·  {best_overall['name']}  ·  BTCUSD 1m  ·  Mar–Jun 2026\n"
        f"Sharpe {best_overall['sharpe']:+.3f}  |  Return {best_overall['ret']:+.1f}%  |  "
        f"MaxDD {best_overall['max_dd']:.1f}%  |  Calmar {best_overall['calmar']:.3f}  |  "
        f"{best_overall['trades']:,} trades  |  {best_overall['win_rate']:.1f}% win rate  |  "
        f"Profit Factor {best_overall['pf']:.2f}",
        color='white', fontsize=11, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, equity, color='#00d4aa', linewidth=0.8)
    ax1.fill_between(dates, equity, 100_000, where=equity >= 100_000, alpha=0.15, color='#00d4aa')
    ax1.fill_between(dates, equity, 100_000, where=equity < 100_000, alpha=0.15, color='#ff4444')
    ax1.axhline(100_000, color='#666', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.set_facecolor('#161b22'); ax1.tick_params(colors='#8b949e', labelsize=8)
    ax1.set_ylabel('Portfolio ($)', color='white', fontsize=9)
    for sp in ax1.spines.values(): sp.set_edgecolor('#30363d')

    # Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, dd, 0, color='#ff4444', alpha=0.7)
    ax2.set_facecolor('#161b22'); ax2.tick_params(colors='#8b949e', labelsize=8)
    ax2.set_ylabel('Drawdown (%)', color='white', fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')

    # BTC price
    step = max(1, n // 2000)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(dates[::step], c[::step], color='#f0a500', linewidth=0.6, alpha=0.8)
    li = np.where(sig == 1)[0][::step]
    si = np.where(sig == -1)[0][::step]
    if len(li): ax3.scatter(dates[li], c[li], c='#00d4aa', s=1, alpha=0.4, zorder=3)
    if len(si): ax3.scatter(dates[si], c[si], c='#ff4444', s=1, alpha=0.4, zorder=3)
    ax3.set_facecolor('#161b22'); ax3.tick_params(colors='#8b949e', labelsize=7)
    ax3.set_title('BTC Price + Signals', color='white', fontsize=9)
    for sp in ax3.spines.values(): sp.set_edgecolor('#30363d')

    # Return distribution
    ax4 = fig.add_subplot(gs[2, 1])
    cr = rets[np.isfinite(rets) & (np.abs(rets) < 0.05)] * 100
    ax4.hist(cr, bins=80, color='#58a6ff', alpha=0.85, edgecolor='none')
    ax4.axvline(0, color='white', linewidth=0.8, linestyle='--')
    ax4.set_facecolor('#161b22'); ax4.tick_params(colors='#8b949e', labelsize=7)
    ax4.set_title('Return Distribution', color='white', fontsize=9)
    ax4.set_xlabel('Return (%)', color='#8b949e', fontsize=8)
    for sp in ax4.spines.values(): sp.set_edgecolor('#30363d')

    # Metrics panel
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off'); ax5.set_facecolor('#161b22')
    metrics = [
        ('Strategy',      best_overall['name']),
        ('Sharpe Ratio',  f"{best_overall['sharpe']:+.3f}"),
        ('Total Return',  f"{best_overall['ret']:+.2f}%"),
        ('Max Drawdown',  f"{best_overall['max_dd']:.2f}%"),
        ('Calmar Ratio',  f"{best_overall['calmar']:+.3f}"),
        ('Profit Factor', f"{best_overall['pf']:.2f}x"),
        ('Total Trades',  f"{best_overall['trades']:,}"),
        ('Win Rate',      f"{best_overall['win_rate']:.1f}%"),
        ('Final Value',   f"${best_overall['final']:,.0f}"),
    ]
    y = 0.95
    for k, v in metrics:
        is_pos = '+' in v or (v.replace(',','').replace('$','').replace('%','').replace('.','').replace('x','').lstrip('-').isdigit() and float(v.replace(',','').replace('$','').replace('%','').replace('x','')) > 0)
        vc = '#00d4aa' if is_pos else '#ff4444'
        if k == 'Strategy': vc = 'white'
        ax5.text(0.02, y, k, transform=ax5.transAxes, color='#8b949e', fontsize=9)
        ax5.text(0.58, y, v, transform=ax5.transAxes, color=vc, fontsize=9, fontweight='bold')
        y -= 0.10
    ax5.set_title('Performance', color='white', fontsize=9)

    Path('/home/ubuntu/AIQuant/results').mkdir(exist_ok=True)
    out = '/home/ubuntu/AIQuant/results/backtest_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Chart saved → {out}")

    # Save best params
    Path('/home/ubuntu/AIQuant/config').mkdir(exist_ok=True)
    with open('/home/ubuntu/AIQuant/config/best_params.json', 'w') as f:
        json.dump({r['name']: {'params': r['params'], 'sharpe': r['sharpe'],
                                'return': r['ret'], 'calmar': r['calmar']}
                   for r in candidates}, f, indent=2)

print("\n✓ Fine-tuning complete.\n")
