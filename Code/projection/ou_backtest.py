#!/usr/bin/env python3
"""
ou_backtest.py — Ornstein-Uhlenbeck Mean Reversion Backtest
=============================================================
Jim Simons Approach: "Find patterns that repeat. Test ruthlessly.
Deploy only what survives."

This script:
1. Fits O-U parameters (θ, μ, σ) per player from 2021-2025
2. Validates on 2025-2026 current season (out-of-sample)
3. Tests if O-U bounce signals predict next-game FPTS
4. Compares O-U predictions vs season average baseline
5. Generates SDE features for LSTM-CNN consumption

The O-U process:  dX(t) = θ(μ - X(t))dt + σ dW(t)
  θ = speed of mean reversion (how fast they bounce back)
  μ = long-run mean FPTS (their "true" level)
  σ = volatility of performance

Run: python ou_backtest.py
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

DB_PATH = 'data/nhl_dfs_history.db'

# Train on historical seasons, test on current
TRAIN_SEASONS = ['20202021', '20212022', '20222023', '20232024', '20242025']
TEST_START = '2025-10-07'  # Current season

# Minimum games to fit O-U (need enough data for reliable params)
MIN_GAMES_FIT = 30
MIN_GAMES_TEST = 10

# Player tiers by avg FPTS
TIER_BINS = {
    'ELITE': (7.0, 999),
    'GOOD': (5.0, 7.0),
    'MID': (3.0, 5.0),
    'DEPTH': (0, 3.0),
}


# ═══════════════════════════════════════════════════════════════
#  O-U PARAMETER ESTIMATION (Maximum Likelihood)
# ═══════════════════════════════════════════════════════════════

def fit_ou_mle(fpts_series):
    """
    Fit Ornstein-Uhlenbeck parameters via Maximum Likelihood Estimation.
    
    For discrete observations X_0, X_1, ..., X_n with Δt = 1 (game-to-game):
    X_{n+1} = X_n + θ(μ - X_n) + σ * Z,  Z ~ N(0,1)
    
    This is equivalent to AR(1): X_{n+1} = (1-θ)X_n + θμ + σZ
    
    MLE estimates:
    φ = (1-θ) = regression coefficient of X_n on X_{n+1}
    θ = 1 - φ
    μ = intercept / θ
    σ = residual standard deviation
    """
    x = np.array(fpts_series, dtype=float)
    
    if len(x) < 10:
        return None
    
    # AR(1) regression: X_{n+1} = a + b*X_n + ε
    x_prev = x[:-1]
    x_next = x[1:]
    
    # OLS
    n = len(x_prev)
    x_bar = x_prev.mean()
    y_bar = x_next.mean()
    
    ss_xx = np.sum((x_prev - x_bar) ** 2)
    ss_xy = np.sum((x_prev - x_bar) * (x_next - y_bar))
    
    if ss_xx == 0:
        return None
    
    b = ss_xy / ss_xx  # AR(1) coefficient = (1 - θ)
    a = y_bar - b * x_bar  # intercept = θμ
    
    # O-U parameters
    theta = 1.0 - b  # mean reversion speed
    
    if theta <= 0 or theta > 2:
        # No mean reversion or unstable — skip
        return None
    
    mu = a / theta  # long-run mean
    
    # Residual volatility
    residuals = x_next - (a + b * x_prev)
    sigma = np.std(residuals, ddof=2)
    
    if sigma <= 0 or mu < 0:
        return None
    
    # Half-life of mean reversion (in games)
    half_life = np.log(2) / theta if theta > 0 else np.inf
    
    # R² of the AR(1) fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((x_next - y_bar) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life': half_life,
        'r_squared': r_squared,
        'n_games': len(x),
        'mean_fpts': np.mean(x),
        'std_fpts': np.std(x),
    }


# ═══════════════════════════════════════════════════════════════
#  HESTON STOCHASTIC VOLATILITY ESTIMATION
# ═══════════════════════════════════════════════════════════════

def fit_heston(fpts_series, window=5):
    """
    Estimate time-varying volatility (Heston-like) from game log.
    
    Uses rolling window squared residuals from O-U as proxy for V(t).
    Then fits O-U to the volatility series itself:
    
    dV(t) = κ(θ_v - V(t))dt + ξ√V(t) dW₂(t)
    
    Returns per-game volatility estimates and Heston parameters.
    """
    x = np.array(fpts_series, dtype=float)
    
    if len(x) < window + 10:
        return None
    
    # First fit O-U to get residuals
    ou_params = fit_ou_mle(x)
    if ou_params is None:
        return None
    
    # Compute squared residuals (proxy for instantaneous variance)
    theta, mu = ou_params['theta'], ou_params['mu']
    x_prev = x[:-1]
    x_next = x[1:]
    predicted = x_prev + theta * (mu - x_prev)
    residuals = x_next - predicted
    
    # Rolling variance (proxy for V(t))
    sq_resid = residuals ** 2
    v_t = pd.Series(sq_resid).rolling(window=window, min_periods=3).mean().values
    
    # Remove NaNs
    valid = ~np.isnan(v_t)
    v_t = v_t[valid]
    
    if len(v_t) < 15:
        return None
    
    # Fit O-U to the volatility series (V(t))
    vol_params = fit_ou_mle(v_t)
    
    if vol_params is None:
        return None
    
    return {
        'kappa': vol_params['theta'],        # vol mean reversion speed
        'theta_v': vol_params['mu'],          # long-run volatility
        'xi': vol_params['sigma'],            # vol-of-vol
        'v_series': v_t,                      # time-varying volatility
        'vol_half_life': vol_params['half_life'],
        'ou_params': ou_params,               # underlying O-U params
    }


# ═══════════════════════════════════════════════════════════════
#  GENERATE SDE FEATURES FOR LSTM-CNN
# ═══════════════════════════════════════════════════════════════

def generate_sde_features(fpts_series, ou_params, heston_params=None, window=5):
    """
    Generate per-game SDE features to feed into LSTM-CNN.
    
    Returns DataFrame with columns:
    - distance_from_mean: μ - X(t), how far below/above true level
    - expected_bounce: θ(μ - X(t)), predicted reversion magnitude
    - z_score: (X(t) - μ) / σ, standardized deviation
    - games_below_mean: consecutive games below μ
    - rolling_vol: V(t), current volatility estimate
    - vol_regime: high/low volatility classification
    - half_life: player's reversion speed
    """
    x = np.array(fpts_series, dtype=float)
    n = len(x)
    
    theta = ou_params['theta']
    mu = ou_params['mu']
    sigma = ou_params['sigma']
    
    features = pd.DataFrame({
        'fpts': x,
        'distance_from_mean': mu - x,
        'expected_bounce': theta * (mu - x),
        'z_score': (x - mu) / sigma if sigma > 0 else 0,
        'theta': theta,
        'mu': mu,
        'half_life': ou_params['half_life'],
    })
    
    # Consecutive games below mean
    below = (x < mu).astype(int)
    consec_below = np.zeros(n)
    for i in range(1, n):
        if below[i-1]:
            consec_below[i] = consec_below[i-1] + 1
        else:
            consec_below[i] = 0
    features['games_below_mean'] = consec_below
    
    # Consecutive games above mean
    above = (x > mu).astype(int)
    consec_above = np.zeros(n)
    for i in range(1, n):
        if above[i-1]:
            consec_above[i] = consec_above[i-1] + 1
        else:
            consec_above[i] = 0
    features['games_above_mean'] = consec_above
    
    # Rolling volatility (Heston V(t))
    sq_dev = (x - mu) ** 2
    rolling_vol = pd.Series(sq_dev).rolling(window=window, min_periods=3).mean().values
    features['rolling_vol'] = rolling_vol
    
    # Volatility regime (above/below long-run vol)
    long_run_vol = sigma ** 2
    features['vol_regime'] = (rolling_vol > long_run_vol).astype(int)
    
    # Heston-specific features
    if heston_params is not None:
        features['vol_distance'] = heston_params['theta_v'] - rolling_vol
        features['vol_z_score'] = (rolling_vol - heston_params['theta_v']) / heston_params['xi'] if heston_params['xi'] > 0 else 0
    
    return features


# ═══════════════════════════════════════════════════════════════
#  MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load historical + current season data."""
    conn = sqlite3.connect(DB_PATH)
    
    # Historical data (training)
    hist = pd.read_sql_query("""
        SELECT player_id, player_name, team, position, game_date, 
               opponent, home_road, goals, assists, points, shots,
               hits, blocked_shots, toi_seconds, pp_goals, pp_points,
               dk_fpts, season
        FROM historical_skaters
        WHERE dk_fpts IS NOT NULL
        ORDER BY player_id, game_date
    """, conn)
    
    # Current season (testing)
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
    
    print(f"  Historical: {len(hist):,} rows | {hist['game_date'].min()} → {hist['game_date'].max()}")
    print(f"  Current:    {len(curr):,} rows | {curr['game_date'].min()} → {curr['game_date'].max()}")
    print(f"  Unique players (hist): {hist['player_id'].nunique()}")
    print(f"  Unique players (curr): {curr['player_id'].nunique()}")
    
    return hist, curr


def run_ou_backtest():
    """
    PHASE 1: Fit O-U on historical data, validate on current season.
    
    Test: For players in a "downgrade" state (below mean), does O-U
    predicted bounce correlate with actual next-game performance?
    """
    print("=" * 70)
    print("  ORNSTEIN-UHLENBECK MEAN REVERSION BACKTEST")
    print("  Training: 2021-2025 | Testing: 2025-2026")
    print("=" * 70)
    
    hist, curr = load_data()
    
    # ── Step 1: Fit O-U per player on historical data ──
    print("\n[1] Fitting O-U parameters per player (historical)...")
    
    ou_params = {}
    heston_params = {}
    
    players = hist.groupby('player_id')
    n_fit = 0
    n_skip = 0
    
    for pid, group in players:
        fpts = group.sort_values('game_date')['dk_fpts'].values
        
        if len(fpts) < MIN_GAMES_FIT:
            n_skip += 1
            continue
        
        params = fit_ou_mle(fpts)
        if params is not None:
            params['player_name'] = group['player_name'].iloc[0]
            params['position'] = group['position'].iloc[0]
            ou_params[pid] = params
            
            # Also fit Heston
            h_params = fit_heston(fpts)
            if h_params is not None:
                heston_params[pid] = h_params
            
            n_fit += 1
        else:
            n_skip += 1
    
    print(f"  Fitted: {n_fit} players | Skipped: {n_skip}")
    print(f"  Heston fitted: {len(heston_params)} players")
    
    # ── Step 2: Analyze O-U parameters by tier ──
    print("\n[2] O-U Parameters by Player Tier:")
    print(f"  {'Tier':<8} {'N':>4} {'θ (reversion)':>14} {'μ (mean)':>10} {'σ (vol)':>9} {'Half-Life':>10}")
    print("  " + "-" * 60)
    
    for tier, (lo, hi) in TIER_BINS.items():
        tier_players = {pid: p for pid, p in ou_params.items() 
                       if lo <= p['mean_fpts'] < hi}
        if not tier_players:
            continue
        
        thetas = [p['theta'] for p in tier_players.values()]
        mus = [p['mu'] for p in tier_players.values()]
        sigmas = [p['sigma'] for p in tier_players.values()]
        half_lives = [p['half_life'] for p in tier_players.values()]
        
        print(f"  {tier:<8} {len(tier_players):>4} "
              f"{np.mean(thetas):>10.4f}±{np.std(thetas):.3f} "
              f"{np.mean(mus):>8.2f} "
              f"{np.mean(sigmas):>7.2f} "
              f"{np.mean(half_lives):>8.1f}g")
    
    # ── Step 3: Out-of-sample backtest on current season ──
    print("\n[3] Out-of-Sample Backtest (2025-2026 season)...")
    
    # For each player with historical O-U params, test predictions
    predictions = []
    baseline_errors = []
    ou_errors = []
    
    curr_players = curr.groupby('player_id')
    
    for pid, group in curr_players:
        if pid not in ou_params:
            continue
        
        params = ou_params[pid]
        theta = params['theta']
        mu_hist = params['mu']  # historical mean
        sigma = params['sigma']
        
        games = group.sort_values('game_date')
        fpts_vals = games['dk_fpts'].values
        
        if len(fpts_vals) < MIN_GAMES_TEST:
            continue
        
        # Also compute current-season running mean for adaptive μ
        # Use blend: 70% historical μ + 30% running mean (Bayesian shrinkage)
        
        for i in range(5, len(fpts_vals)):  # need 5 games of context
            x_current = fpts_vals[i-1]  # last game FPTS
            x_actual = fpts_vals[i]     # what they actually scored
            
            # Running mean (current season)
            running_mean = np.mean(fpts_vals[:i])
            
            # Blended μ (shrinkage toward historical)
            weight_hist = max(0.3, MIN_GAMES_FIT / (MIN_GAMES_FIT + i))
            mu_blend = weight_hist * mu_hist + (1 - weight_hist) * running_mean
            
            # O-U prediction: E[X_{n+1}] = X_n + θ(μ - X_n)
            ou_pred = x_current + theta * (mu_blend - x_current)
            
            # Baseline: season average so far
            baseline_pred = running_mean
            
            # Distance from mean (signal strength)
            distance = mu_blend - x_current
            z_score = distance / sigma if sigma > 0 else 0
            
            # Volatility regime (from Heston if available)
            vol_regime = 'unknown'
            if pid in heston_params:
                recent_var = np.var(fpts_vals[max(0,i-5):i])
                long_run_var = sigma ** 2
                vol_regime = 'high' if recent_var > long_run_var else 'low'
            
            predictions.append({
                'player_id': pid,
                'player_name': params['player_name'],
                'game_date': games.iloc[i]['game_date'],
                'x_current': x_current,
                'mu_blend': mu_blend,
                'theta': theta,
                'ou_pred': ou_pred,
                'baseline_pred': baseline_pred,
                'actual': x_actual,
                'distance': distance,
                'z_score': z_score,
                'vol_regime': vol_regime,
                'ou_error': abs(ou_pred - x_actual),
                'baseline_error': abs(baseline_pred - x_actual),
                'tier': next((t for t, (lo, hi) in TIER_BINS.items() 
                            if lo <= params['mean_fpts'] < hi), 'UNKNOWN'),
            })
    
    df = pd.DataFrame(predictions)
    print(f"  Total predictions: {len(df):,}")
    print(f"  Unique players tested: {df['player_id'].nunique()}")
    
    # ── Step 4: Overall Results ──
    print("\n[4] OVERALL RESULTS:")
    print(f"  {'Metric':<20} {'Season Avg':>12} {'O-U':>12} {'Δ':>10} {'Better?':>8}")
    print("  " + "-" * 65)
    
    baseline_mae = df['baseline_error'].mean()
    ou_mae = df['ou_error'].mean()
    delta_mae = (baseline_mae - ou_mae) / baseline_mae * 100
    
    baseline_rmse = np.sqrt((df['baseline_error']**2).mean())
    ou_rmse = np.sqrt((df['ou_error']**2).mean())
    delta_rmse = (baseline_rmse - ou_rmse) / baseline_rmse * 100
    
    # Correlation with actual
    baseline_corr = np.corrcoef(df['baseline_pred'], df['actual'])[0, 1]
    ou_corr = np.corrcoef(df['ou_pred'], df['actual'])[0, 1]
    
    print(f"  {'MAE':<20} {baseline_mae:>12.4f} {ou_mae:>12.4f} {delta_mae:>9.2f}% {'✓' if delta_mae > 0 else '✗':>7}")
    print(f"  {'RMSE':<20} {baseline_rmse:>12.4f} {ou_rmse:>12.4f} {delta_rmse:>9.2f}% {'✓' if delta_rmse > 0 else '✗':>7}")
    print(f"  {'Correlation':<20} {baseline_corr:>12.4f} {ou_corr:>12.4f} {ou_corr - baseline_corr:>9.4f} {'✓' if ou_corr > baseline_corr else '✗':>7}")
    
    # Paired t-test
    diffs = df['baseline_error'] - df['ou_error']
    t_stat, p_val = stats.ttest_rel(df['baseline_error'], df['ou_error'])
    print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_val:.6f}")
    print(f"  Mean improvement per prediction: {diffs.mean():.4f} FPTS")
    
    # ── Step 5: Results by Tier ──
    print("\n[5] RESULTS BY TIER:")
    print(f"  {'Tier':<8} {'N':>6} {'Baseline MAE':>13} {'O-U MAE':>10} {'Δ%':>8} {'p-value':>10}")
    print("  " + "-" * 60)
    
    for tier in ['ELITE', 'GOOD', 'MID', 'DEPTH']:
        tier_df = df[df['tier'] == tier]
        if len(tier_df) < 50:
            continue
        
        b_mae = tier_df['baseline_error'].mean()
        o_mae = tier_df['ou_error'].mean()
        delta = (b_mae - o_mae) / b_mae * 100
        _, p = stats.ttest_rel(tier_df['baseline_error'], tier_df['ou_error'])
        
        print(f"  {tier:<8} {len(tier_df):>6} {b_mae:>13.4f} {o_mae:>10.4f} {delta:>7.2f}% {p:>10.6f}")
    
    # ── Step 6: SIGNAL TEST — Do players below mean bounce back? ──
    print("\n[6] BOUNCE-BACK SIGNAL TEST:")
    print("  (Does being below mean predict above-average next game?)")
    
    # Group by z-score bins
    bins = [
        ('z < -2.0 (deep cold)', df['z_score'] < -2.0),
        ('-2.0 ≤ z < -1.0 (cold)', (df['z_score'] >= -2.0) & (df['z_score'] < -1.0)),
        ('-1.0 ≤ z < 0 (below avg)', (df['z_score'] >= -1.0) & (df['z_score'] < 0)),
        ('0 ≤ z < 1.0 (above avg)', (df['z_score'] >= 0) & (df['z_score'] < 1.0)),
        ('z ≥ 1.0 (hot)', df['z_score'] >= 1.0),
    ]
    
    print(f"  {'Z-Score Bin':<30} {'N':>6} {'Avg Next FPTS':>14} {'vs Mean':>10} {'O-U Pred':>10}")
    print("  " + "-" * 75)
    
    for label, mask in bins:
        subset = df[mask]
        if len(subset) < 30:
            continue
        
        avg_actual = subset['actual'].mean()
        avg_mu = subset['mu_blend'].mean()
        avg_ou = subset['ou_pred'].mean()
        
        diff = avg_actual - avg_mu
        print(f"  {label:<30} {len(subset):>6} {avg_actual:>14.3f} {diff:>+9.3f} {avg_ou:>10.3f}")
    
    # ── Step 7: COMBINED SIGNAL — O-U + Opponent + Volatility ──
    print("\n[7] STACKED SIGNAL TEST (O-U distance × volatility regime):")
    
    # Low vol + deep below mean = highest confidence bounce
    for vol in ['low', 'high']:
        for z_label, z_lo, z_hi in [('cold (z<-1)', -999, -1), ('warm (z>1)', 1, 999)]:
            mask = (df['vol_regime'] == vol) & (df['z_score'] >= z_lo) & (df['z_score'] < z_hi)
            subset = df[mask]
            if len(subset) < 20:
                continue
            
            avg_actual = subset['actual'].mean()
            avg_mu = subset['mu_blend'].mean()
            avg_baseline = subset['baseline_pred'].mean()
            
            print(f"  Vol={vol:<4} + {z_label:<15}: N={len(subset):>5} | "
                  f"Actual={avg_actual:.2f} | Mean={avg_mu:.2f} | "
                  f"Diff={avg_actual - avg_mu:+.2f}")
    
    # ── Step 8: Top O-U Players (fastest reversion) ──
    print("\n[8] TOP 20 FASTEST MEAN-REVERTERS (highest θ, 50+ games):")
    print(f"  {'Player':<25} {'Pos':>3} {'θ':>6} {'μ':>6} {'σ':>6} {'Half-Life':>10} {'R²':>6}")
    print("  " + "-" * 65)
    
    sorted_players = sorted(
        [(pid, p) for pid, p in ou_params.items() if p['n_games'] >= 50],
        key=lambda x: x[1]['theta'],
        reverse=True
    )[:20]
    
    for pid, p in sorted_players:
        print(f"  {p['player_name']:<25} {p['position']:>3} "
              f"{p['theta']:>6.3f} {p['mu']:>6.2f} {p['sigma']:>6.2f} "
              f"{p['half_life']:>8.1f}g {p['r_squared']:>6.3f}")
    
    # ── Step 9: Export SDE features for LSTM-CNN ──
    print("\n[9] Exporting SDE features...")
    
    sde_features_all = []
    
    for pid, group in curr_players:
        if pid not in ou_params:
            continue
        
        games = group.sort_values('game_date')
        fpts_vals = games['dk_fpts'].values
        
        if len(fpts_vals) < 10:
            continue
        
        h_params = heston_params.get(pid)
        features = generate_sde_features(fpts_vals, ou_params[pid], h_params)
        features['player_id'] = pid
        features['player_name'] = ou_params[pid]['player_name']
        features['game_date'] = games['game_date'].values[:len(features)]
        
        sde_features_all.append(features)
    
    if sde_features_all:
        sde_df = pd.concat(sde_features_all, ignore_index=True)
        sde_df.to_csv('data/sde_features.csv', index=False)
        print(f"  Saved: data/sde_features.csv ({len(sde_df):,} rows)")
    
    # ── Step 10: Export O-U parameters ──
    params_df = pd.DataFrame([
        {
            'player_id': pid,
            'player_name': p['player_name'],
            'position': p['position'],
            'theta': p['theta'],
            'mu': p['mu'],
            'sigma': p['sigma'],
            'half_life': p['half_life'],
            'r_squared': p['r_squared'],
            'n_games': p['n_games'],
            'mean_fpts': p['mean_fpts'],
        }
        for pid, p in ou_params.items()
    ])
    params_df.to_csv('data/ou_parameters.csv', index=False)
    print(f"  Saved: data/ou_parameters.csv ({len(params_df)} players)")
    
    # Heston params
    if heston_params:
        heston_df = pd.DataFrame([
            {
                'player_id': pid,
                'kappa': p['kappa'],
                'theta_v': p['theta_v'],
                'xi': p['xi'],
                'vol_half_life': p['vol_half_life'],
            }
            for pid, p in heston_params.items()
        ])
        heston_df.to_csv('data/heston_parameters.csv', index=False)
        print(f"  Saved: data/heston_parameters.csv ({len(heston_df)} players)")
    
    # ── Final Verdict ──
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    
    if delta_mae > 0 and p_val < 0.05:
        print(f"  ✓ O-U BEATS SEASON AVERAGE by {delta_mae:.2f}% MAE (p={p_val:.6f})")
        print(f"  → Integrate into pipeline")
    elif delta_mae > 0 and p_val >= 0.05:
        print(f"  △ O-U slightly better (+{delta_mae:.2f}%) but NOT significant (p={p_val:.4f})")
        print(f"  → Use as SELECTION signal, not projection adjustment")
    else:
        print(f"  ✗ O-U does NOT beat season average ({delta_mae:.2f}%)")
        print(f"  → Use θ and z-score as FEATURES for MDN/LSTM-CNN, not standalone")
    
    print(f"\n  KEY OUTPUTS:")
    print(f"  • O-U parameters per player: data/ou_parameters.csv")
    print(f"  • Heston parameters: data/heston_parameters.csv")
    print(f"  • SDE features for LSTM-CNN: data/sde_features.csv")
    print(f"  • All three files feed directly into your existing models")
    
    return df, ou_params, heston_params


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results, ou_params, heston_params = run_ou_backtest()
