"""
scripts/deep_optimize.py
=========================
Deep optimisation pass targeting Sharpe > 2.0.
Key insight: 2026 BTC is choppy with equal bull/bear time.
Strategy: use RSI extremes + VWAP deviation + volume spike confirmation.
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
print("  AIQuant — Deep Optimisation (Target: Sharpe > 2.0)")
print("  BTCUSD 1m  |  Mar–Jun 2026  |  $100,000 capital")
print("="*65)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading data...")
df_raw = pd.read_parquet('/home/ubuntu/AIQuant/data/raw/BTCUSDT_1m_full.parquet')
from aiquant.utils.fast_math import warmup
from aiquant.features import build_full_feature_set
warmup()
df = build_full_feature_set(df_raw.copy(), verbose=False)
df = df.dropna()
n = len(df)
print(f"      {n:,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

c      = df['close'].to_numpy(np.float64)
h      = df['high'].to_numpy(np.float64)
l      = df['low'].to_numpy(np.float64)
v      = df['volume'].to_numpy(np.float64)
dates  = df.index

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
rv20   = df['realvol_20'].to_numpy(np.float64)
rv240  = df['realvol_240'].to_numpy(np.float64)
obv    = df['obv'].to_numpy(np.float64)
cmf    = df['cmf'].to_numpy(np.float64)

# Derived
vwap_dev = (c - vwap) / (atr14 + 1e-10)  # VWAP deviation in ATR units
vol_spike = vr > 1.5                       # volume above 1.5x average
vol_ok    = (rv20 > 0.3) & (rv20 < 2.5)
obv_ema   = pd.Series(obv).ewm(span=20, adjust=False).mean().to_numpy()
obv_bull  = obv > obv_ema
obv_bear  = obv < obv_ema

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def backtest(signals, fee=0.00035, kelly=0.25, initial=100_000.0):
    sig = signals.astype(np.int8)
    changes = np.where(np.diff(sig, prepend=sig[0]) != 0)[0]
    if len(changes) < 4:
        return dict(sharpe=0.0, ret=0.0, max_dd=0.0, trades=0,
                    win_rate=0.0, final=initial, calmar=0.0, pf=0.0,
                    equity=np.full(len(sig), initial))
    capital = initial
    equity  = np.full(len(sig), initial, dtype=np.float64)
    wins = losses = 0; gw = gl = 0.0
    for i in range(len(changes) - 1):
        eb = changes[i]; xb = changes[i+1]
        d  = int(sig[eb])
        if d == 0: equity[eb:xb] = capital; continue
        ep = c[eb]; xp = c[xb]
        sz = (capital * kelly) / ep
        pnl = d * (xp - ep) * sz * (1 - fee)**2
        capital = max(capital + pnl, 1.0)
        equity[eb:xb] = capital
        if pnl > 0: wins += 1; gw += pnl
        else: losses += 1; gl += abs(pnl)
    equity[changes[-1]:] = capital
    total_ret = (capital - initial) / initial * 100
    dd        = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
    max_dd    = float(np.min(dd))
    n_days    = len(equity) // 1440
    if n_days > 2:
        dl = np.array([np.diff(np.log(equity[i*1440:(i+1)*1440]+1e-10)).sum() for i in range(n_days)])
        sharpe = float(dl.mean() / (dl.std() + 1e-10) * np.sqrt(365))
    else:
        sharpe = 0.0
    tt = wins + losses
    wr = wins / tt * 100 if tt > 0 else 0.0
    cal = total_ret / abs(max_dd) if abs(max_dd) > 0.01 else 0.0
    pf  = gw / (gl + 1e-10)
    return dict(sharpe=round(sharpe,3), ret=round(total_ret,2), max_dd=round(max_dd,2),
                trades=tt, win_rate=round(wr,1), final=round(capital,2),
                calmar=round(cal,3), pf=round(pf,3), equity=equity)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: VWAP MEAN REVERSION + VOLUME CONFIRMATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] VWAP Mean Reversion + Volume...")
best_vwap = None
for vwap_lo, vwap_hi in [(-2.5, 2.5), (-2.0, 2.0), (-1.8, 1.8), (-3.0, 3.0)]:
    for rsi_lo, rsi_hi in [(28, 72), (30, 70), (32, 68)]:
        for adx_max in [25, 30, 35]:
            for min_hold in [5, 10, 15, 20]:
                long_ok  = (vwap_dev < vwap_lo) & (rsi14 < rsi_lo) & \
                           (adx < adx_max) & vol_ok
                short_ok = (vwap_dev > vwap_hi) & (rsi14 > rsi_hi) & \
                           (adx < adx_max) & vol_ok

                sig = np.zeros(n, np.int8)
                pos = 0; ep = 0.0; hold = 0
                for i in range(1, n):
                    if pos == 0:
                        if long_ok[i]:   pos = 1; ep = c[i]; hold = 0
                        elif short_ok[i]: pos = -1; ep = c[i]; hold = 0
                    elif pos == 1:
                        hold += 1
                        if hold < min_hold: sig[i] = pos; continue
                        if rsi14[i] > 55 or vwap_dev[i] > 0.5 or \
                           (ep - c[i]) > 2.5*atr14[i] or hold > 120:
                            pos = 0
                    elif pos == -1:
                        hold += 1
                        if hold < min_hold: sig[i] = pos; continue
                        if rsi14[i] < 45 or vwap_dev[i] < -0.5 or \
                           (c[i] - ep) > 2.5*atr14[i] or hold > 120:
                            pos = 0
                    sig[i] = pos

                r = backtest(sig)
                r['name'] = 'VWAP Reversion'
                r['params'] = f'vwap[{vwap_lo},{vwap_hi}] rsi[{rsi_lo},{rsi_hi}] adx<{adx_max} hold={min_hold}'
                r['signals'] = sig
                if best_vwap is None or r['sharpe'] > best_vwap['sharpe']:
                    best_vwap = r
                    print(f"  ↑ Sharpe={r['sharpe']:+.3f}  Return={r['ret']:+.1f}%  "
                          f"Trades={r['trades']:,}  MaxDD={r['max_dd']:.1f}%  WR={r['win_rate']:.0f}%  [{r['params']}]")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: RSI DIVERGENCE + BB BAND TOUCH
# Enter when RSI is extreme AND price touches BB band
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] RSI Extreme + BB Band Touch...")
best_bb = None
for rsi_os in [25, 28, 30]:
    for rsi_ob in [70, 72, 75]:
        for bb_touch in [0.02, 0.03, 0.05]:  # within X% of band
            for min_hold in [10, 15, 20, 30]:
                near_lower = (c - bbl) / (c + 1e-10) < bb_touch
                near_upper = (bbu - c) / (c + 1e-10) < bb_touch

                long_ok  = (rsi14 < rsi_os) & near_lower & (adx < 32) & vol_ok
                short_ok = (rsi14 > rsi_ob) & near_upper & (adx < 32) & vol_ok

                sig = np.zeros(n, np.int8)
                pos = 0; ep = 0.0; hold = 0
                for i in range(1, n):
                    if pos == 0:
                        if long_ok[i]:   pos = 1; ep = c[i]; hold = 0
                        elif short_ok[i]: pos = -1; ep = c[i]; hold = 0
                    elif pos == 1:
                        hold += 1
                        if hold < min_hold: sig[i] = pos; continue
                        if rsi14[i] > 58 or (ep - c[i]) > 2.5*atr14[i] or hold > 90:
                            pos = 0
                    elif pos == -1:
                        hold += 1
                        if hold < min_hold: sig[i] = pos; continue
                        if rsi14[i] < 42 or (c[i] - ep) > 2.5*atr14[i] or hold > 90:
                            pos = 0
                    sig[i] = pos

                r = backtest(sig)
                r['name'] = 'RSI+BB Touch'
                r['params'] = f'os={rsi_os} ob={rsi_ob} touch={bb_touch} hold={min_hold}'
                r['signals'] = sig
                if best_bb is None or r['sharpe'] > best_bb['sharpe']:
                    best_bb = r
                    print(f"  ↑ Sharpe={r['sharpe']:+.3f}  Return={r['ret']:+.1f}%  "
                          f"Trades={r['trades']:,}  MaxDD={r['max_dd']:.1f}%  WR={r['win_rate']:.0f}%  [{r['params']}]")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3: TREND MOMENTUM (EMA + MACD + ADX triple confirmation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Triple-Confirm Trend Momentum...")
prev_macd = np.concatenate([[macd[0]], macd[:-1]])
prev_msig = np.concatenate([[msig[0]], msig[:-1]])
macd_cross_up   = (macd > msig) & (prev_macd <= prev_msig)
macd_cross_down = (macd < msig) & (prev_macd >= prev_msig)

best_trend = None
for adx_min in [22, 26, 30]:
    for rsi_lo, rsi_hi in [(45, 55), (42, 58), (48, 52)]:
        for atr_stop in [1.5, 2.0, 2.5]:
            uptrend   = (ema9 > ema21) & (ema21 > ema50) & (adx > adx_min)
            downtrend = (ema9 < ema21) & (ema21 < ema50) & (adx > adx_min)

            long_ok  = uptrend   & macd_cross_up   & (rsi14 > rsi_lo) & (rsi14 < 72) & vol_ok
            short_ok = downtrend & macd_cross_down & (rsi14 < rsi_hi) & (rsi14 > 28) & vol_ok

            sig = np.zeros(n, np.int8)
            pos = 0; ep = 0.0; hold = 0
            for i in range(1, n):
                if pos == 0:
                    if long_ok[i]:   pos = 1; ep = c[i]; hold = 0
                    elif short_ok[i]: pos = -1; ep = c[i]; hold = 0
                elif pos == 1:
                    hold += 1
                    if (ep - c[i]) > atr_stop*atr14[i] or \
                       (ema9[i] < ema21[i] and hold > 5) or hold > 180:
                        pos = 0
                elif pos == -1:
                    hold += 1
                    if (c[i] - ep) > atr_stop*atr14[i] or \
                       (ema9[i] > ema21[i] and hold > 5) or hold > 180:
                        pos = 0
                sig[i] = pos

            r = backtest(sig)
            r['name'] = 'Triple Trend'
            r['params'] = f'adx>{adx_min} rsi[{rsi_lo},{rsi_hi}] stop={atr_stop}x'
            r['signals'] = sig
            if best_trend is None or r['sharpe'] > best_trend['sharpe']:
                best_trend = r
                print(f"  ↑ Sharpe={r['sharpe']:+.3f}  Return={r['ret']:+.1f}%  "
                      f"Trades={r['trades']:,}  MaxDD={r['max_dd']:.1f}%  WR={r['win_rate']:.0f}%  [{r['params']}]")

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE: STACK ALL BEST STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Building Stacked Ensemble...")
candidates = [r for r in [best_vwap, best_bb, best_trend] if r is not None and r['sharpe'] > 0]

best_ensemble = None
if len(candidates) >= 2:
    # Try different vote thresholds
    for threshold in [1, 2]:
        vote = np.zeros(n, np.float64)
        for r in candidates:
            vote += r['signals'].astype(np.float64)
        ens_sig = np.where(vote >= threshold, 1, np.where(vote <= -threshold, -1, 0)).astype(np.int8)
        r_ens = backtest(ens_sig)
        r_ens['name'] = f'Ensemble (vote≥{threshold})'
        r_ens['params'] = f'{len(candidates)} strategies, threshold={threshold}'
        r_ens['signals'] = ens_sig
        candidates.append(r_ens)
        print(f"  Ensemble vote≥{threshold}: Sharpe={r_ens['sharpe']:+.3f}  "
              f"Return={r_ens['ret']:+.1f}%  Trades={r_ens['trades']:,}  "
              f"MaxDD={r_ens['max_dd']:.1f}%  WR={r_ens['win_rate']:.0f}%")

# Add all previous bests
from_prev = [
    dict(name='Vol-RSI (prev)', sharpe=1.721, ret=1.6, max_dd=-1.6, trades=274, win_rate=60.6, calmar=0.973, pf=1.15, final=101600, equity=None),
]

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
all_res = [r for r in candidates if r.get('equity') is not None]
all_res.sort(key=lambda x: x['sharpe'], reverse=True)
best_overall = all_res[0] if all_res else None

print("\n" + "="*80)
print("  DEEP OPTIMISATION RESULTS  |  BTCUSD 1m  |  Mar–Jun 2026  |  $100k")
print("="*80)
print(f"  {'Strategy':<26} {'Sharpe':>7} {'Return':>9} {'MaxDD':>8} "
      f"{'Trades':>8} {'WinRate':>8} {'Calmar':>8} {'PF':>6}")
print(f"  {'-'*26} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
for i, r in enumerate(all_res[:10]):
    flag = ' ★' if i == 0 else ''
    print(f"  {r['name']:<26} {r['sharpe']:>+7.3f} {r['ret']:>+8.1f}% "
          f"{r['max_dd']:>+8.1f}% {r['trades']:>8,} {r['win_rate']:>7.1f}% "
          f"{r['calmar']:>+8.3f} {r['pf']:>6.2f}{flag}")
print("="*80)

# Save best params
if all_res:
    Path('/home/ubuntu/AIQuant/config').mkdir(exist_ok=True)
    with open('/home/ubuntu/AIQuant/config/best_params.json', 'w') as f:
        json.dump({r['name']: {'params': r.get('params',''), 'sharpe': r['sharpe'],
                                'return': r['ret'], 'calmar': r['calmar']}
                   for r in all_res[:5]}, f, indent=2)
    print(f"\n  Best params saved → config/best_params.json")

# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────
if best_overall and best_overall.get('equity') is not None:
    print(f"\n  Generating chart for '{best_overall['name']}'...")
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

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, equity, color='#00d4aa', linewidth=0.8)
    ax1.fill_between(dates, equity, 100_000, where=equity >= 100_000, alpha=0.15, color='#00d4aa')
    ax1.fill_between(dates, equity, 100_000, where=equity < 100_000, alpha=0.15, color='#ff4444')
    ax1.axhline(100_000, color='#666', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.set_facecolor('#161b22'); ax1.tick_params(colors='#8b949e', labelsize=8)
    ax1.set_ylabel('Portfolio ($)', color='white', fontsize=9)
    ax1.set_title('Equity Curve', color='white', fontsize=9)
    for sp in ax1.spines.values(): sp.set_edgecolor('#30363d')

    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, dd, 0, color='#ff4444', alpha=0.7)
    ax2.set_facecolor('#161b22'); ax2.tick_params(colors='#8b949e', labelsize=8)
    ax2.set_ylabel('Drawdown (%)', color='white', fontsize=9)
    ax2.set_title('Drawdown', color='white', fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')

    step = max(1, n // 2000)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(dates[::step], c[::step], color='#f0a500', linewidth=0.6, alpha=0.8)
    li = np.where(sig == 1)[0][::step]
    si = np.where(sig == -1)[0][::step]
    if len(li): ax3.scatter(dates[li], c[li], c='#00d4aa', s=1, alpha=0.5, zorder=3)
    if len(si): ax3.scatter(dates[si], c[si], c='#ff4444', s=1, alpha=0.5, zorder=3)
    ax3.set_facecolor('#161b22'); ax3.tick_params(colors='#8b949e', labelsize=7)
    ax3.set_title('BTC Price + Signals', color='white', fontsize=9)
    for sp in ax3.spines.values(): sp.set_edgecolor('#30363d')

    ax4 = fig.add_subplot(gs[2, 1])
    cr = rets[np.isfinite(rets) & (np.abs(rets) < 0.05)] * 100
    ax4.hist(cr, bins=80, color='#58a6ff', alpha=0.85, edgecolor='none')
    ax4.axvline(0, color='white', linewidth=0.8, linestyle='--')
    mean_r = cr.mean(); std_r = cr.std()
    ax4.axvline(mean_r, color='#00d4aa', linewidth=1.0, linestyle='--', alpha=0.8)
    ax4.set_facecolor('#161b22'); ax4.tick_params(colors='#8b949e', labelsize=7)
    ax4.set_title('Return Distribution', color='white', fontsize=9)
    ax4.set_xlabel('Return (%)', color='#8b949e', fontsize=8)
    for sp in ax4.spines.values(): sp.set_edgecolor('#30363d')

    ax5 = fig.add_subplot(gs[2, 2])
    names_s   = [r['name'][:14] for r in all_res[:6]]
    sharpes_s = [r['sharpe'] for r in all_res[:6]]
    colors_s  = ['#00d4aa' if s > 0 else '#ff4444' for s in sharpes_s]
    ax5.barh(names_s, sharpes_s, color=colors_s, alpha=0.85)
    ax5.axvline(0, color='white', linewidth=0.8)
    ax5.set_facecolor('#161b22'); ax5.tick_params(colors='#8b949e', labelsize=7)
    ax5.set_title('Sharpe Comparison', color='white', fontsize=9)
    ax5.set_xlabel('Sharpe Ratio', color='#8b949e', fontsize=8)
    for sp in ax5.spines.values(): sp.set_edgecolor('#30363d')

    Path('/home/ubuntu/AIQuant/results').mkdir(exist_ok=True)
    out = '/home/ubuntu/AIQuant/results/backtest_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Chart saved → {out}")

print("\n✓ Deep optimisation complete.\n")
