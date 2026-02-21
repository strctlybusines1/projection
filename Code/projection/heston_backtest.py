#!/usr/bin/env python3
"""
heston_backtest.py — Heston Stochastic Volatility Backtest
============================================================
Jim Simons Approach: "We don't start with models. We start with data.
We look for things that can be replicated thousands of times."

The Heston model couples TWO stochastic processes:
  1. Player performance:  dX(t) = θ(μ - X(t))dt + √V(t) dW₁(t)
  2. Volatility itself:   dV(t) = κ(θ_v - V(t))dt + ξ√V(t) dW₂(t)
  
  Correlation: dW₁·dW₂ = ρ dt

Key insight: Some players go through QUIET periods (low V(t), scoring
near their mean) and CHAOTIC periods (high V(t), booming or busting).
Knowing which regime a player is in RIGHT NOW is the edge:

  - Low V(t) + below mean  = HIGH CONFIDENCE bounce → Cash + GPP
  - High V(t) + below mean = UNCERTAIN bounce → GPP only (high ceiling)
  - Low V(t) + above mean  = STABLE producer → Cash lock
  - High V(t) + above mean = REGRESSION RISK → Fade in cash

This script:
1. Fits Heston parameters per player (κ, θ_v, ξ, ρ) on 2021-2025
2. Validates volatility regime classification on 2025-2026
3. Tests GPP/cash play identification accuracy
4. Measures if volatility regime predicts next-game outcome spread
5. Combines with O-U for stacked signals
6. Exports features for LSTM-CNN consumption

Requires: ou_backtest.py must have been run first (reads ou_parameters.csv)

Run: python heston_backtest.py
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

DB_PATH = 'data/nhl_dfs_history.db'

TEST_START = '2025-10-07'
MIN_GAMES_FIT = 30
MIN_GAMES_TEST = 10

# Volatility estimation windows
VOL_WINDOW_SHORT = 5     # recent 5-game vol
VOL_WINDOW_MED = 10      # medium-term vol
VOL_WINDOW_LONG = 20     # long-term vol (baseline)

# Regime thresholds (calibrated during backtest)
VOL_REGIME_QUANTILES = [0.33, 0.67]  # low / mid / high

TIER_BINS = {
    'ELITE': (7.0, 999),
    'GOOD': (5.0, 7.0),
    'MID': (3.0, 5.0),
    'DEPTH': (0, 3.0),
}


# ═══════════════════════════════════════════════════════════════
#  HESTON PARAMETER ESTIMATION
# ═══════════════════════════════════════════════════════════════

def estimate_realized_volatility(fpts_series, window=5):
    """
    Estimate realized volatility series from FPTS game log.
    Uses rolling standard deviation of residuals from O-U fit.
    
    Returns array of V(t) estimates, one per game.
    """
    x = np.array(fpts_series, dtype=float)
    n = len(x)
    
    if n < window + 5:
        return None
    
    # O-U residuals: ε_t = X_{t+1} - X_t - θ(μ - X_t)
    # Simplified: just use rolling variance of raw returns
    returns = np.diff(x)
    
    # Rolling squared returns as variance proxy
    sq_returns = returns ** 2
    v_t = np.full(n, np.nan)
    
    for i in range(window, n):
        v_t[i] = np.mean(sq_returns[max(0, i-window-1):i-1])
    
    return v_t


def fit_heston_full(fpts_series, vol_window=5):
    """
    Fit full Heston model parameters.
    
    Performance process (O-U with stochastic vol):
      dX(t) = θ(μ - X(t))dt + √V(t) dW₁(t)
    
    Volatility process (CIR):
      dV(t) = κ(θ_v - V(t))dt + ξ√V(t) dW₂(t)
    
    Correlation:
      Corr(dW₁, dW₂) = ρ
    
    Estimates via method of moments on discrete observations.
    """
    x = np.array(fpts_series, dtype=float)
    n = len(x)
    
    if n < 40:
        return None
    
    # ── Step 1: Fit O-U on the level ──
    x_prev = x[:-1]
    x_next = x[1:]
    
    ss_xx = np.sum((x_prev - x_prev.mean()) ** 2)
    ss_xy = np.sum((x_prev - x_prev.mean()) * (x_next - x_next.mean()))
    
    if ss_xx == 0:
        return None
    
    b = ss_xy / ss_xx
    a = x_next.mean() - b * x_prev.mean()
    
    theta = 1.0 - b
    if theta <= 0 or theta > 2:
        return None
    
    mu = a / theta
    if mu < 0:
        return None
    
    # O-U residuals
    ou_pred = x_prev + theta * (mu - x_prev)
    residuals = x_next - ou_pred
    
    # ── Step 2: Estimate V(t) from rolling squared residuals ──
    sq_resid = residuals ** 2
    
    v_t = np.full(len(residuals), np.nan)
    for i in range(vol_window, len(residuals)):
        v_t[i] = np.mean(sq_resid[i-vol_window:i])
    
    valid = ~np.isnan(v_t)
    v_clean = v_t[valid]
    
    if len(v_clean) < 20:
        return None
    
    # ── Step 3: Fit CIR to volatility process ──
    # V_{t+1} = V_t + κ(θ_v - V_t) + ξ√V_t · Z
    v_prev = v_clean[:-1]
    v_next = v_clean[1:]
    
    # Prevent division by zero
    v_prev_safe = np.maximum(v_prev, 1e-6)
    
    ss_vv = np.sum((v_prev - v_prev.mean()) ** 2)
    ss_vvy = np.sum((v_prev - v_prev.mean()) * (v_next - v_next.mean()))
    
    if ss_vv == 0:
        return None
    
    b_v = ss_vvy / ss_vv
    a_v = v_next.mean() - b_v * v_prev.mean()
    
    kappa = 1.0 - b_v  # vol mean reversion speed
    
    if kappa <= 0 or kappa > 2:
        kappa = np.clip(kappa, 0.01, 1.99)
    
    theta_v = a_v / kappa if kappa > 0 else v_clean.mean()  # long-run variance
    theta_v = max(theta_v, 1e-6)
    
    # Vol-of-vol: residual std of CIR, normalized by √V
    v_resid = v_next - (v_prev + kappa * (theta_v - v_prev))
    xi_raw = np.std(v_resid)
    xi = xi_raw / np.mean(np.sqrt(v_prev_safe))  # normalize by √V
    
    # ── Step 4: Estimate correlation ρ between dW₁ and dW₂ ──
    # Match residuals in time using valid indices
    valid_idx = np.where(valid)[0]
    # v_resid[i] corresponds to transition valid_idx[i] -> valid_idx[i+1]
    # Match with performance residuals at valid_idx[1:] (the "next" timestep)
    perf_idx = valid_idx[1:]  # indices into residuals for v_resid alignment
    min_len = min(len(perf_idx), len(v_resid))
    if min_len > 10:
        perf_resid = residuals[perf_idx[:min_len]]
        vol_resid_matched = v_resid[:min_len]
        rho = np.corrcoef(perf_resid, vol_resid_matched)[0, 1]
        if np.isnan(rho):
            rho = 0.0
    else:
        rho = 0.0
    
    # ── Step 5: Compute diagnostics ──
    vol_half_life = np.log(2) / kappa if kappa > 0 else np.inf
    
    # Feller condition: 2κθ_v > ξ² (ensures V(t) stays positive)
    feller = 2 * kappa * theta_v > xi ** 2
    
    return {
        # O-U parameters (performance)
        'theta': theta,
        'mu': mu,
        'ou_sigma': np.std(residuals),
        
        # CIR parameters (volatility)
        'kappa': kappa,
        'theta_v': theta_v,
        'xi': xi,
        'rho': rho,
        
        # Diagnostics
        'vol_half_life': vol_half_life,
        'feller_satisfied': feller,
        'mean_vol': np.mean(v_clean),
        'vol_of_vol': np.std(v_clean),
        
        # Raw series
        'v_series': v_t,
        'residuals': residuals,
        'n_games': n,
    }


def classify_vol_regime(v_current, v_history, quantiles=[0.33, 0.67]):
    """
    Classify current volatility into regime: LOW, MID, HIGH.
    Based on historical distribution of V(t) for this player.
    """
    if v_current is None or np.isnan(v_current):
        return 'UNKNOWN', 0.5
    
    clean = v_history[~np.isnan(v_history)]
    if len(clean) < 10:
        return 'UNKNOWN', 0.5
    
    q_low, q_high = np.quantile(clean, quantiles)
    
    if v_current <= q_low:
        regime = 'LOW'
        percentile = stats.percentileofscore(clean, v_current) / 100
    elif v_current >= q_high:
        regime = 'HIGH'
        percentile = stats.percentileofscore(clean, v_current) / 100
    else:
        regime = 'MID'
        percentile = stats.percentileofscore(clean, v_current) / 100
    
    return regime, percentile


def classify_dfs_play(vol_regime, z_score, theta):
    """
    Classify player as GPP target, cash lock, fade, or neutral.
    
    Decision matrix:
                        Below Mean (z<-0.5)    Near Mean    Above Mean (z>0.5)
    Low Vol (stable)     CASH_BOUNCE           CASH_LOCK     CASH_LOCK
    Mid Vol              GPP_BOUNCE            NEUTRAL       NEUTRAL
    High Vol (chaotic)   GPP_CEILING           GPP_PLAY      FADE_CASH
    """
    if vol_regime == 'LOW':
        if z_score < -0.5:
            return 'CASH_BOUNCE'    # reliable player, temporarily cold → high confidence bounce
        elif z_score > 0.5:
            return 'CASH_LOCK'      # stable producer, above mean → safe play
        else:
            return 'CASH_LOCK'      # near mean, low vol → predictable
    
    elif vol_regime == 'HIGH':
        if z_score < -0.5:
            return 'GPP_CEILING'    # volatile + cold → could explode, high ceiling play
        elif z_score > 0.5:
            return 'FADE_CASH'      # volatile + hot → regression risk, avoid in cash
        else:
            return 'GPP_PLAY'       # volatile near mean → unpredictable, GPP only
    
    else:  # MID
        if z_score < -0.5:
            return 'GPP_BOUNCE'     # moderate confidence bounce
        elif z_score > 0.5:
            return 'NEUTRAL'        # could go either way
        else:
            return 'NEUTRAL'


# ═══════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load historical + current season data."""
    conn = sqlite3.connect(DB_PATH)
    
    hist = pd.read_sql_query("""
        SELECT player_id, player_name, team, position, game_date,
               opponent, home_road, goals, assists, points, shots,
               hits, blocked_shots, toi_seconds, pp_goals, pp_points,
               dk_fpts, season
        FROM historical_skaters
        WHERE dk_fpts IS NOT NULL
        ORDER BY player_id, game_date
    """, conn)
    
    curr = pd.read_sql_query("""
        SELECT player_id, player_name, team, position, game_date,
               opponent, home_road, goals, assists, points, shots,
               hits, blocked_shots, toi_seconds, pp_goals,
               dk_fpts
        FROM boxscore_skaters
        WHERE dk_fpts IS NOT NULL AND game_date >= ?
        ORDER BY player_id, game_date
    """, conn, params=[TEST_START])
    curr['season'] = '20252026'
    
    conn.close()
    return hist, curr


# ═══════════════════════════════════════════════════════════════
#  MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_heston_backtest():
    print("=" * 70)
    print("  HESTON STOCHASTIC VOLATILITY BACKTEST")
    print("  Training: 2021-2025 | Testing: 2025-2026")
    print("  'Volatility itself is volatile' — the DFS edge")
    print("=" * 70)
    
    hist, curr = load_data()
    
    print(f"\n  Historical: {len(hist):,} rows | {hist['game_date'].min()} → {hist['game_date'].max()}")
    print(f"  Current:    {len(curr):,} rows | {curr['game_date'].min()} → {curr['game_date'].max()}")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 1: Fit Heston per player on historical data
    # ══════════════════════════════════════════════════════════
    
    print("\n[1] Fitting Heston parameters per player (historical)...")
    
    heston_params = {}
    n_fit = 0
    n_skip = 0
    
    for pid, group in hist.groupby('player_id'):
        fpts = group.sort_values('game_date')['dk_fpts'].values
        
        if len(fpts) < MIN_GAMES_FIT:
            n_skip += 1
            continue
        
        params = fit_heston_full(fpts)
        if params is not None:
            params['player_name'] = group['player_name'].iloc[0]
            params['position'] = group['position'].iloc[0]
            params['mean_fpts'] = np.mean(fpts)
            heston_params[pid] = params
            n_fit += 1
        else:
            n_skip += 1
    
    print(f"  Fitted: {n_fit} players | Skipped: {n_skip}")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 2: Heston parameter distributions by tier
    # ══════════════════════════════════════════════════════════
    
    print(f"\n[2] Heston Parameters by Player Tier:")
    print(f"  {'Tier':<8} {'N':>4} {'κ (vol revert)':>14} {'θ_v (LR vol)':>13} "
          f"{'ξ (vol-of-vol)':>14} {'ρ (corr)':>10} {'Feller%':>8}")
    print("  " + "-" * 75)
    
    for tier, (lo, hi) in TIER_BINS.items():
        tier_p = {pid: p for pid, p in heston_params.items()
                 if lo <= p['mean_fpts'] < hi}
        if not tier_p:
            continue
        
        kappas = [p['kappa'] for p in tier_p.values()]
        theta_vs = [p['theta_v'] for p in tier_p.values()]
        xis = [p['xi'] for p in tier_p.values()]
        rhos = [p['rho'] for p in tier_p.values()]
        feller_pct = sum(1 for p in tier_p.values() if p['feller_satisfied']) / len(tier_p) * 100
        
        print(f"  {tier:<8} {len(tier_p):>4} "
              f"{np.mean(kappas):>10.4f}±{np.std(kappas):.3f} "
              f"{np.mean(theta_vs):>10.3f}±{np.std(theta_vs):.2f} "
              f"{np.mean(xis):>10.4f}±{np.std(xis):.3f} "
              f"{np.mean(rhos):>8.3f} "
              f"{feller_pct:>7.0f}%")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 3: Out-of-sample volatility regime classification
    # ══════════════════════════════════════════════════════════
    
    print("\n[3] Out-of-Sample Regime Classification (2025-2026)...")
    
    predictions = []
    
    for pid, group in curr.groupby('player_id'):
        if pid not in heston_params:
            continue
        
        params = heston_params[pid]
        theta = params['theta']
        mu = params['mu']
        kappa = params['kappa']
        theta_v = params['theta_v']
        
        games = group.sort_values('game_date')
        fpts = games['dk_fpts'].values
        
        if len(fpts) < MIN_GAMES_TEST:
            continue
        
        # Compute V(t) for current season using multiple windows
        v_short = estimate_realized_volatility(fpts, window=VOL_WINDOW_SHORT)
        v_med = estimate_realized_volatility(fpts, window=VOL_WINDOW_MED)
        
        if v_short is None:
            continue
        
        # Get historical vol distribution for regime classification
        hist_fpts = hist[hist['player_id'] == pid].sort_values('game_date')['dk_fpts'].values
        v_hist = estimate_realized_volatility(hist_fpts, window=VOL_WINDOW_SHORT)
        
        if v_hist is None:
            continue
        
        # Running season mean
        for i in range(max(VOL_WINDOW_SHORT + 2, 8), len(fpts)):
            v_now = v_short[i]
            
            if np.isnan(v_now):
                continue
            
            x_current = fpts[i-1]
            x_actual = fpts[i]
            running_mean = np.mean(fpts[:i])
            
            # Blend historical and current season mean
            weight_hist = max(0.3, MIN_GAMES_FIT / (MIN_GAMES_FIT + i))
            mu_blend = weight_hist * mu + (1 - weight_hist) * running_mean
            
            # O-U z-score
            sigma_est = np.sqrt(max(v_now, 1e-6))
            z_score = (x_current - mu_blend) / sigma_est if sigma_est > 0 else 0
            
            # Classify volatility regime
            vol_regime, vol_pctl = classify_vol_regime(v_now, v_hist)
            
            # Classify DFS play type
            dfs_class = classify_dfs_play(vol_regime, z_score, theta)
            
            # O-U prediction (with stochastic vol)
            ou_pred = x_current + theta * (mu_blend - x_current)
            baseline_pred = running_mean
            
            # Actual outcome metrics
            actual_deviation = x_actual - mu_blend  # how far from mean they actually scored
            actual_abs_dev = abs(actual_deviation)   # magnitude of deviation
            beat_mean = 1 if x_actual > mu_blend else 0
            
            # Ceiling hit (top 25% outcome for this player)
            ceiling_threshold = mu_blend + sigma_est
            hit_ceiling = 1 if x_actual > ceiling_threshold else 0
            
            # Floor hit (bottom 25% outcome)
            floor_threshold = mu_blend - sigma_est
            hit_floor = 1 if x_actual < floor_threshold else 0
            
            tier = next((t for t, (tlo, thi) in TIER_BINS.items()
                        if tlo <= params['mean_fpts'] < thi), 'UNKNOWN')
            
            predictions.append({
                'player_id': pid,
                'player_name': params['player_name'],
                'position': params['position'],
                'game_date': games.iloc[i]['game_date'],
                'tier': tier,
                
                # Current state
                'x_current': x_current,
                'mu_blend': mu_blend,
                'z_score': z_score,
                'v_now': v_now,
                'vol_regime': vol_regime,
                'vol_percentile': vol_pctl,
                'dfs_class': dfs_class,
                
                # Heston params
                'theta': theta,
                'kappa': kappa,
                'theta_v': theta_v,
                'rho': params['rho'],
                
                # Predictions
                'ou_pred': ou_pred,
                'baseline_pred': baseline_pred,
                
                # Actuals
                'actual': x_actual,
                'actual_deviation': actual_deviation,
                'beat_mean': beat_mean,
                'hit_ceiling': hit_ceiling,
                'hit_floor': hit_floor,
                
                # Errors
                'ou_error': abs(ou_pred - x_actual),
                'baseline_error': abs(baseline_pred - x_actual),
            })
    
    df = pd.DataFrame(predictions)
    print(f"  Total predictions: {len(df):,}")
    print(f"  Players tested: {df['player_id'].nunique()}")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 4: Volatility regime predicts outcome spread
    # ══════════════════════════════════════════════════════════
    
    print(f"\n[4] DOES VOLATILITY REGIME PREDICT OUTCOME SPREAD?")
    print(f"  (Higher V(t) should → wider actual outcomes)")
    print(f"  {'Vol Regime':<12} {'N':>6} {'Avg |Dev|':>10} {'Std Dev':>9} {'Ceiling%':>9} {'Floor%':>8}")
    print("  " + "-" * 60)
    
    for regime in ['LOW', 'MID', 'HIGH']:
        r_df = df[df['vol_regime'] == regime]
        if len(r_df) < 50:
            continue
        
        avg_abs_dev = r_df['actual_deviation'].abs().mean()
        std_dev = r_df['actual'].std()
        ceiling_pct = r_df['hit_ceiling'].mean() * 100
        floor_pct = r_df['hit_floor'].mean() * 100
        
        print(f"  {regime:<12} {len(r_df):>6} {avg_abs_dev:>10.3f} {std_dev:>9.3f} "
              f"{ceiling_pct:>8.1f}% {floor_pct:>7.1f}%")
    
    # Statistical test: HIGH vol games have wider spreads?
    if len(df[df['vol_regime'] == 'HIGH']) > 50 and len(df[df['vol_regime'] == 'LOW']) > 50:
        high_devs = df[df['vol_regime'] == 'HIGH']['actual_deviation'].abs()
        low_devs = df[df['vol_regime'] == 'LOW']['actual_deviation'].abs()
        t_stat, p_val = stats.ttest_ind(high_devs, low_devs)
        print(f"\n  High vs Low volatility spread: t={t_stat:.3f}, p={p_val:.6f}")
        
        levene_stat, levene_p = stats.levene(
            df[df['vol_regime'] == 'HIGH']['actual'],
            df[df['vol_regime'] == 'LOW']['actual']
        )
        print(f"  Levene's test (variance equality): F={levene_stat:.3f}, p={levene_p:.6f}")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 5: DFS Classification Accuracy
    # ══════════════════════════════════════════════════════════
    
    print(f"\n[5] DFS PLAY CLASSIFICATION RESULTS:")
    print(f"  {'Classification':<16} {'N':>6} {'Avg FPTS':>10} {'vs Mean':>9} {'Beat%':>7} "
          f"{'Ceiling%':>9} {'Floor%':>8}")
    print("  " + "-" * 70)
    
    for dfs_class in ['CASH_BOUNCE', 'CASH_LOCK', 'GPP_BOUNCE', 'GPP_CEILING',
                       'GPP_PLAY', 'FADE_CASH', 'NEUTRAL']:
        c_df = df[df['dfs_class'] == dfs_class]
        if len(c_df) < 30:
            continue
        
        avg_fpts = c_df['actual'].mean()
        vs_mean = c_df['actual_deviation'].mean()
        beat_pct = c_df['beat_mean'].mean() * 100
        ceil_pct = c_df['hit_ceiling'].mean() * 100
        floor_pct = c_df['hit_floor'].mean() * 100
        
        print(f"  {dfs_class:<16} {len(c_df):>6} {avg_fpts:>10.3f} {vs_mean:>+8.3f} "
              f"{beat_pct:>6.1f}% {ceil_pct:>8.1f}% {floor_pct:>7.1f}%")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 6: CASH_BOUNCE vs FADE_CASH — the money test
    # ══════════════════════════════════════════════════════════
    
    print(f"\n[6] THE MONEY TEST — CASH_BOUNCE vs FADE_CASH:")
    
    bounce = df[df['dfs_class'] == 'CASH_BOUNCE']
    fade = df[df['dfs_class'] == 'FADE_CASH']
    
    if len(bounce) > 20 and len(fade) > 20:
        print(f"  CASH_BOUNCE: N={len(bounce):,} | Avg FPTS={bounce['actual'].mean():.3f} | "
              f"vs Mean={bounce['actual_deviation'].mean():+.3f} | "
              f"Beat Mean={bounce['beat_mean'].mean()*100:.1f}%")
        print(f"  FADE_CASH:   N={len(fade):,} | Avg FPTS={fade['actual'].mean():.3f} | "
              f"vs Mean={fade['actual_deviation'].mean():+.3f} | "
              f"Beat Mean={fade['beat_mean'].mean()*100:.1f}%")
        
        # Are CASH_BOUNCE plays actually better relative to expectations?
        t_stat, p_val = stats.ttest_ind(
            bounce['actual_deviation'], fade['actual_deviation']
        )
        print(f"\n  CASH_BOUNCE vs FADE_CASH (deviation from mean): t={t_stat:.3f}, p={p_val:.6f}")
        
        diff = bounce['actual_deviation'].mean() - fade['actual_deviation'].mean()
        print(f"  Edge per play: {diff:+.3f} FPTS")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 7: GPP Ceiling identification
    # ══════════════════════════════════════════════════════════
    
    print(f"\n[7] GPP CEILING IDENTIFICATION:")
    print(f"  Do HIGH volatility players hit ceiling more often?")
    
    for tier in ['ELITE', 'GOOD', 'MID', 'DEPTH']:
        tier_df = df[df['tier'] == tier]
        if len(tier_df) < 100:
            continue
        
        high_vol = tier_df[tier_df['vol_regime'] == 'HIGH']
        low_vol = tier_df[tier_df['vol_regime'] == 'LOW']
        
        if len(high_vol) < 20 or len(low_vol) < 20:
            continue
        
        h_ceil = high_vol['hit_ceiling'].mean() * 100
        l_ceil = low_vol['hit_ceiling'].mean() * 100
        h_floor = high_vol['hit_floor'].mean() * 100
        l_floor = low_vol['hit_floor'].mean() * 100
        
        print(f"  {tier:<8} | HIGH vol: ceiling={h_ceil:.1f}%, floor={h_floor:.1f}% | "
              f"LOW vol: ceiling={l_ceil:.1f}%, floor={l_floor:.1f}% | "
              f"Ceiling Δ={h_ceil - l_ceil:+.1f}pp")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 8: Combined O-U + Heston stacked signal
    # ══════════════════════════════════════════════════════════
    
    print(f"\n[8] STACKED SIGNAL: O-U Distance × Vol Regime × Tier:")
    print(f"  {'Signal':<45} {'N':>5} {'Avg FPTS':>9} {'vs Mean':>8} {'Beat%':>7}")
    print("  " + "-" * 78)
    
    stacked_signals = [
        ('ELITE + cold (z<-1) + LOW vol', 
         (df['tier'] == 'ELITE') & (df['z_score'] < -1) & (df['vol_regime'] == 'LOW')),
        ('ELITE + cold (z<-1) + HIGH vol',
         (df['tier'] == 'ELITE') & (df['z_score'] < -1) & (df['vol_regime'] == 'HIGH')),
        ('ELITE + hot (z>1) + HIGH vol (FADE)',
         (df['tier'] == 'ELITE') & (df['z_score'] > 1) & (df['vol_regime'] == 'HIGH')),
        ('GOOD + cold (z<-1) + LOW vol',
         (df['tier'] == 'GOOD') & (df['z_score'] < -1) & (df['vol_regime'] == 'LOW')),
        ('DEPTH + cold (z<-1) + LOW vol',
         (df['tier'] == 'DEPTH') & (df['z_score'] < -1) & (df['vol_regime'] == 'LOW')),
        ('DEPTH + cold (z<-1) + HIGH vol (GPP ceiling)',
         (df['tier'] == 'DEPTH') & (df['z_score'] < -1) & (df['vol_regime'] == 'HIGH')),
        ('Any tier + hot (z>1) + HIGH vol (FADE ALL)',
         (df['z_score'] > 1) & (df['vol_regime'] == 'HIGH')),
        ('Any tier + cold (z<-1) + LOW vol (BEST BOUNCE)',
         (df['z_score'] < -1) & (df['vol_regime'] == 'LOW')),
    ]
    
    for label, mask in stacked_signals:
        subset = df[mask]
        if len(subset) < 15:
            print(f"  {label:<45} {'<15':>5} {'—':>9} {'—':>8} {'—':>7}")
            continue
        
        avg = subset['actual'].mean()
        vs_mean = subset['actual_deviation'].mean()
        beat = subset['beat_mean'].mean() * 100
        
        print(f"  {label:<45} {len(subset):>5} {avg:>9.2f} {vs_mean:>+7.2f} {beat:>6.1f}%")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 9: ρ (correlation) analysis
    # ══════════════════════════════════════════════════════════
    
    print(f"\n[9] LEVERAGE EFFECT (ρ) — Does poor performance increase volatility?")
    
    rhos = [(pid, p['rho'], p['player_name'], p['mean_fpts']) 
            for pid, p in heston_params.items() if p['n_games'] >= 50]
    
    if rhos:
        rho_vals = [r[1] for r in rhos]
        print(f"  Mean ρ across all players: {np.mean(rho_vals):.4f}")
        print(f"  Players with ρ < -0.1 (poor perf → higher vol): {sum(1 for r in rho_vals if r < -0.1)}")
        print(f"  Players with ρ > 0.1 (good perf → higher vol):  {sum(1 for r in rho_vals if r > 0.1)}")
        
        # By tier
        for tier, (lo, hi) in TIER_BINS.items():
            tier_rhos = [r[1] for r in rhos if lo <= r[3] < hi]
            if tier_rhos:
                print(f"  {tier:<8}: mean ρ = {np.mean(tier_rhos):+.4f} (N={len(tier_rhos)})")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 10: Export everything
    # ══════════════════════════════════════════════════════════
    
    print(f"\n[10] Exporting results...")
    
    # Heston parameters
    params_export = pd.DataFrame([
        {
            'player_id': pid,
            'player_name': p['player_name'],
            'position': p['position'],
            'mean_fpts': p['mean_fpts'],
            'theta': p['theta'],
            'mu': p['mu'],
            'kappa': p['kappa'],
            'theta_v': p['theta_v'],
            'xi': p['xi'],
            'rho': p['rho'],
            'vol_half_life': p['vol_half_life'],
            'feller_satisfied': p['feller_satisfied'],
            'n_games': p['n_games'],
        }
        for pid, p in heston_params.items()
    ])
    params_export.to_csv('data/heston_parameters_full.csv', index=False)
    print(f"  Saved: data/heston_parameters_full.csv ({len(params_export)} players)")
    
    # Full predictions with classifications
    df.to_csv('data/heston_backtest_results.csv', index=False)
    print(f"  Saved: data/heston_backtest_results.csv ({len(df):,} predictions)")
    
    # DFS classification summary (for quick lookup on game day)
    # Get most recent classification per player
    latest = df.sort_values('game_date').groupby('player_id').last().reset_index()
    latest_export = latest[['player_id', 'player_name', 'position', 'tier',
                            'vol_regime', 'vol_percentile', 'z_score', 'dfs_class',
                            'mu_blend', 'theta', 'kappa', 'v_now']].copy()
    latest_export.to_csv('data/heston_current_classifications.csv', index=False)
    print(f"  Saved: data/heston_current_classifications.csv ({len(latest_export)} players)")
    
    # SDE features for LSTM-CNN (enhanced with Heston)
    sde_features = df[['player_id', 'player_name', 'game_date', 'x_current',
                        'mu_blend', 'z_score', 'v_now', 'vol_regime', 'vol_percentile',
                        'dfs_class', 'theta', 'kappa', 'theta_v', 'rho',
                        'ou_pred', 'actual']].copy()
    sde_features.to_csv('data/heston_sde_features.csv', index=False)
    print(f"  Saved: data/heston_sde_features.csv ({len(sde_features):,} rows)")
    
    # ══════════════════════════════════════════════════════════
    #  VERDICT
    # ══════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    
    # Check if vol regime classifies outcomes
    if len(df[df['vol_regime'] == 'HIGH']) > 50 and len(df[df['vol_regime'] == 'LOW']) > 50:
        high_std = df[df['vol_regime'] == 'HIGH']['actual'].std()
        low_std = df[df['vol_regime'] == 'LOW']['actual'].std()
        
        levene_stat, levene_p = stats.levene(
            df[df['vol_regime'] == 'HIGH']['actual'],
            df[df['vol_regime'] == 'LOW']['actual']
        )
        
        if levene_p < 0.05:
            print(f"  ✓ VOLATILITY REGIMES ARE REAL (Levene p={levene_p:.6f})")
            print(f"    HIGH vol games: σ={high_std:.2f} FPTS")
            print(f"    LOW vol games:  σ={low_std:.2f} FPTS")
            print(f"    → Use for GPP/cash classification")
        else:
            print(f"  △ Volatility regimes weakly differentiated (p={levene_p:.4f})")
    
    # Check DFS classifications
    bounce_df = df[df['dfs_class'] == 'CASH_BOUNCE']
    fade_df = df[df['dfs_class'] == 'FADE_CASH']
    
    if len(bounce_df) > 20 and len(fade_df) > 20:
        edge = bounce_df['actual_deviation'].mean() - fade_df['actual_deviation'].mean()
        _, p = stats.ttest_ind(bounce_df['actual_deviation'], fade_df['actual_deviation'])
        
        if p < 0.05:
            print(f"  ✓ DFS CLASSIFICATION WORKS: CASH_BOUNCE beats FADE_CASH by {edge:+.2f} FPTS (p={p:.4f})")
        else:
            print(f"  △ DFS classification shows {edge:+.2f} FPTS edge but p={p:.4f}")
    
    print(f"\n  KEY OUTPUTS:")
    print(f"  • Heston params:           data/heston_parameters_full.csv")
    print(f"  • Backtest results:        data/heston_backtest_results.csv")
    print(f"  • Current classifications: data/heston_current_classifications.csv")
    print(f"  • SDE features for LSTM:   data/heston_sde_features.csv")
    print(f"\n  NEXT: Feed heston_sde_features.csv into LSTM-CNN as additional input columns")
    
    return df, heston_params


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results, heston_params = run_heston_backtest()
