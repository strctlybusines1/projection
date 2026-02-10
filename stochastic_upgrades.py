#!/usr/bin/env python3
"""
Stochastic Process Upgrades for NHL DFS Pipeline
==================================================

Implements the full audit recommendations:
1. Gamma distribution for player FPTS simulation (replaces Normal)
2. Negative Binomial for event counts (replaces Poisson)
3. Poisson-tail bonus probabilities (replaces Normal approximation)
4. Lineup-level variance with correlation for Contest EV
5. Beta distribution for ownership uncertainty
6. Backtest-calibrated variance parameters

All parameters derived from 2,137 player-game observations across 7 dates.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════
#  Backtest-Calibrated Parameters
# ═══════════════════════════════════════════════════════════════════

VARIANCE_PATH = Path(__file__).parent / "data" / "backtest_variance.json"

def load_variance_params() -> Dict:
    """Load position × salary tier variance from backtest."""
    try:
        with open(VARIANCE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

_VARIANCE_CACHE = None

def get_variance_params() -> Dict:
    global _VARIANCE_CACHE
    if _VARIANCE_CACHE is None:
        _VARIANCE_CACHE = load_variance_params()
    return _VARIANCE_CACHE


def get_player_variance(position: str, salary: float) -> Dict[str, float]:
    """
    Get backtest-calibrated variance parameters for a player.
    
    Returns:
        dict with 'actual_std', 'mae', 'rmse', 'overdispersion_r', 'bias'
    """
    params = get_variance_params()
    
    if salary < 3500: tier = 'min'
    elif salary < 5000: tier = 'mid'
    elif salary < 7000: tier = 'high'
    else: tier = 'elite'
    
    # Normalize position
    pos = position.upper()
    if pos in ('LW', 'RW'): pos = 'W'
    
    key = f"{pos}_{tier}"
    if key in params:
        p = params[key]
        return {
            'actual_std': p['actual_std'],
            'mae': p['mae'],
            'rmse': p['rmse'],
            'overdispersion_r': p.get('overdispersion_r'),
            'bias': p['bias'],
            'residual_std': p['residual_std'],
        }
    
    # Fallback: use position average
    pos_keys = [k for k in params if k.startswith(pos + '_')]
    if pos_keys:
        avg_std = np.mean([params[k]['actual_std'] for k in pos_keys])
        avg_mae = np.mean([params[k]['mae'] for k in pos_keys])
        return {
            'actual_std': avg_std, 'mae': avg_mae, 'rmse': avg_std,
            'overdispersion_r': None, 'bias': 0, 'residual_std': avg_std,
        }
    
    # Ultimate fallback
    return {'actual_std': 6.0, 'mae': 5.0, 'rmse': 6.5,
            'overdispersion_r': None, 'bias': 0, 'residual_std': 5.5}


# ═══════════════════════════════════════════════════════════════════
#  1. Gamma Distribution for Player FPTS (replaces Normal)
# ═══════════════════════════════════════════════════════════════════

def sample_fpts_gamma(projected: float, position: str, salary: float,
                      n: int = 1, variance_override: float = None) -> np.ndarray:
    """
    Sample FPTS from Gamma distribution calibrated to backtest variance.
    
    Gamma is better than Normal because:
    - Support on [0, ∞) (FPTS can't be negative)
    - Right-skewed (matches real FPTS distribution)
    - Mean and variance independently controllable
    
    Gamma parameterization: shape=k, scale=θ
    Mean = k*θ, Var = k*θ²
    So: k = mean²/var, θ = var/mean
    """
    if projected <= 0:
        return np.zeros(n)
    
    if variance_override is not None:
        var = variance_override
    else:
        vp = get_player_variance(position, salary)
        var = vp['actual_std'] ** 2
    
    # Ensure variance is reasonable
    var = max(var, 1.0)
    
    # Gamma parameters
    k = projected ** 2 / var  # shape
    theta = var / projected    # scale
    
    return np.random.gamma(k, theta, n)


def sample_fpts_gamma_batch(projected: np.ndarray, positions: np.ndarray,
                            salaries: np.ndarray) -> np.ndarray:
    """Batch sample FPTS from Gamma for an entire lineup/pool."""
    result = np.zeros(len(projected))
    for i in range(len(projected)):
        if projected[i] > 0:
            result[i] = sample_fpts_gamma(projected[i], positions[i], salaries[i], n=1)[0]
    return result


# ═══════════════════════════════════════════════════════════════════
#  2. Negative Binomial for Event Counts (replaces Poisson)
# ═══════════════════════════════════════════════════════════════════

# Overdispersion parameters calibrated via MLE from game logs + actuals
# (4,244 skater-games, 110 goalie-games)
# Higher r = closer to Poisson; lower r = more overdispersed
DEFAULT_NB_R = {
    'goals': 1.98,          # Var/μ=1.09, moderately overdispersed
    'assists': 2.82,        # Var/μ=1.11, mild overdispersion
    'shots': 3.74,          # Var/μ=1.45, notable overdispersion
    'blocks': 1.90,         # Var/μ=1.39, bursty — most overdispersed skater event
    'saves': 16.52,         # Var/μ=2.12, but high mean makes r large
    'goals_against': 116.10, # Var/μ=1.03, essentially Poisson
}


def sample_negative_binomial(rate: float, r: float = None,
                             event_type: str = 'goals', n: int = 1) -> np.ndarray:
    """
    Sample from Negative Binomial distribution.
    
    NB(r, p) where:
    - r = dispersion parameter (higher = less overdispersed)
    - p = r / (r + rate)
    - Mean = rate, Var = rate + rate²/r
    
    When r → ∞, NB → Poisson.
    """
    if rate <= 0:
        return np.zeros(n, dtype=int)
    
    if r is None:
        r = DEFAULT_NB_R.get(event_type, 2.0)
    
    # NB parameterization: n=r, p=r/(r+mu)
    p = r / (r + rate)
    return np.random.negative_binomial(r, p, n)


def poisson_or_nb(rate: float, event_type: str = 'goals',
                  use_nb: bool = True, n: int = 1) -> np.ndarray:
    """
    Draw event counts using Poisson or Negative Binomial.
    Drop-in replacement for np.random.poisson(rate, n).
    """
    if not use_nb or rate <= 0:
        return np.random.poisson(max(rate, 0), n)
    
    r = DEFAULT_NB_R.get(event_type, 2.0)
    return sample_negative_binomial(rate, r, event_type, n)


# ═══════════════════════════════════════════════════════════════════
#  3. Poisson-Tail Bonus Probabilities (replaces Normal approximation)
# ═══════════════════════════════════════════════════════════════════

def bonus_prob_poisson(rate: float, threshold: int) -> float:
    """
    P(X >= threshold) using Poisson CDF.
    
    Replaces the logistic/normal approximation in features.py:
        z = (threshold - mean) / std
        prob = 1 / (1 + exp(1.7 * z))
    
    With exact:
        prob = 1 - poisson.cdf(threshold - 1, rate)
    """
    if rate <= 0:
        return 0.0
    return float(1 - stats.poisson.cdf(threshold - 1, rate))


def bonus_prob_nb(rate: float, threshold: int, r: float = 2.0) -> float:
    """
    P(X >= threshold) using Negative Binomial CDF.
    Heavier tails than Poisson — more realistic for boom games.
    """
    if rate <= 0:
        return 0.0
    p = r / (r + rate)
    return float(1 - stats.nbinom.cdf(threshold - 1, r, p))


def compute_all_bonus_probs(shot_rate: float, goal_rate: float,
                            assist_rate: float, block_rate: float,
                            save_rate: float = 0, use_nb: bool = True) -> Dict[str, float]:
    """
    Compute all bonus probabilities from rates.
    
    Replaces features.py _estimate_bonus_prob with rate-based CDF.
    """
    fn = bonus_prob_nb if use_nb else bonus_prob_poisson
    
    probs = {}
    probs['p_five_sog'] = fn(shot_rate, 5, DEFAULT_NB_R['shots']) if use_nb else bonus_prob_poisson(shot_rate, 5)
    probs['p_hat_trick'] = fn(goal_rate, 3, DEFAULT_NB_R['goals']) if use_nb else bonus_prob_poisson(goal_rate, 3)
    probs['p_three_blocks'] = fn(block_rate, 3, DEFAULT_NB_R['blocks']) if use_nb else bonus_prob_poisson(block_rate, 3)
    probs['p_three_points'] = fn(goal_rate + assist_rate, 3, 2.0) if use_nb else bonus_prob_poisson(goal_rate + assist_rate, 3)
    
    if save_rate > 0:
        probs['p_35_saves'] = fn(save_rate, 35, DEFAULT_NB_R['saves']) if use_nb else bonus_prob_poisson(save_rate, 35)
    
    return probs


# ═══════════════════════════════════════════════════════════════════
#  4. Lineup-Level Variance with Correlation (Contest EV)
# ═══════════════════════════════════════════════════════════════════

# Correlation constants from game log analysis
CORR_LINEMATE = 0.30      # Same line
CORR_SAME_TEAM = 0.15     # Same team, different lines
CORR_GOALIE_OPP = -0.10   # Goalie vs opposing skaters
CORR_OPP_SKATERS = 0.05   # Opposing skaters (game-level)


def lineup_variance(lineup_df: pd.DataFrame,
                    linemate_pairs: set = None) -> Tuple[float, float]:
    """
    Compute lineup total mean and std with correlation structure.
    
    Returns (mean, std) where std accounts for:
    - Individual player variance (from backtest)
    - Intra-team correlation (stacking boost)
    - Linemate correlation (same-line boost)
    
    This replaces the deterministic sigmoid in contest_roi._bucket_probs
    with a proper distribution for lineup totals.
    """
    mean = lineup_df['projected_fpts'].sum()
    
    # Per-player variance
    player_vars = []
    player_stds = []
    for _, p in lineup_df.iterrows():
        vp = get_player_variance(p['position'], p['salary'])
        player_vars.append(vp['actual_std'] ** 2)
        player_stds.append(vp['actual_std'])
    
    # Base variance (independent sum)
    base_var = sum(player_vars)
    
    # Correlation boost
    corr_boost = 0.0
    teams = lineup_df.groupby('team')
    
    for team_name, team_players in teams:
        skaters = team_players[team_players['position'] != 'G']
        n_sk = len(skaters)
        if n_sk < 2:
            continue
        
        # Get stds for team's skaters
        team_stds = []
        for _, p in skaters.iterrows():
            vp = get_player_variance(p['position'], p['salary'])
            team_stds.append(vp['actual_std'])
        
        avg_std = np.mean(team_stds)
        n_pairs = n_sk * (n_sk - 1) // 2
        
        # Check linemate pairs
        if linemate_pairs:
            names = set(skaters['name'].str.lower())
            n_linemate = sum(1 for a, b in linemate_pairs 
                           if a.lower() in names and b.lower() in names)
            n_other = n_pairs - n_linemate
        else:
            n_linemate = 0
            n_other = n_pairs
        
        corr_boost += n_linemate * 2 * CORR_LINEMATE * avg_std * avg_std
        corr_boost += n_other * 2 * CORR_SAME_TEAM * avg_std * avg_std
    
    total_var = base_var + corr_boost
    total_std = np.sqrt(max(total_var, 1.0))
    
    return mean, total_std


def contest_ev_from_distribution(lineup_df: pd.DataFrame,
                                  payout_curve: Dict = None,
                                  entry_fee: float = 121.0,
                                  linemate_pairs: set = None) -> Dict[str, float]:
    """
    Compute contest EV using proper distribution for lineup total.
    
    Replaces contest_roi._bucket_probs sigmoid with:
    lineup_total ~ Normal(mean, std) with correlation
    P(place) = P(lineup_total > threshold) from CDF
    
    Uses the payout curve from tournament_equity.py.
    """
    mean, std = lineup_variance(lineup_df, linemate_pairs)
    
    # Load payout curve if not provided
    if payout_curve is None:
        from tournament_equity import PAYOUT_CURVE
        payout_curve = PAYOUT_CURVE
    
    # Compute probabilities
    p_cash = 1 - stats.norm.cdf(110, mean, std)
    p_top10 = 1 - stats.norm.cdf(130, mean, std)
    p_top5 = 1 - stats.norm.cdf(140, mean, std)
    p_win = 1 - stats.norm.cdf(150, mean, std)
    
    # Expected payout via numerical integration
    te = 0.0
    bins = sorted(payout_curve.keys())
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mid_payout = (payout_curve[lo] + payout_curve[hi]) / 2
        p = stats.norm.cdf(hi, mean, std) - stats.norm.cdf(lo, mean, std)
        te += p * mid_payout
    
    return {
        'te': te,
        'ev': te - entry_fee,
        'mean': mean,
        'std': std,
        'p_cash': p_cash,
        'p_top10': p_top10,
        'p_top5': p_top5,
        'p_win': p_win,
    }


# ═══════════════════════════════════════════════════════════════════
#  5. Beta Distribution for Ownership Uncertainty
# ═══════════════════════════════════════════════════════════════════

# From ownership backtest residuals (typical MAE ~ 3-5% points)
OWNERSHIP_RESIDUAL_STD = 0.04  # 4 percentage points


def ownership_beta_params(predicted_pct: float,
                          residual_std: float = OWNERSHIP_RESIDUAL_STD) -> Tuple[float, float]:
    """
    Convert predicted ownership % to Beta(alpha, beta) parameters.
    
    Beta has support [0, 1], mean = alpha/(alpha+beta), 
    var = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
    
    Given mean=p and desired var=v:
    concentration = p*(1-p)/v - 1
    alpha = p * concentration
    beta = (1-p) * concentration
    """
    p = max(0.01, min(0.99, predicted_pct / 100.0))
    v = residual_std ** 2
    
    # Ensure variance is feasible: v < p*(1-p)
    max_v = p * (1 - p) * 0.9
    v = min(v, max_v)
    
    if v <= 0:
        return 100.0, 100.0 * (1 - p) / p  # Very tight
    
    concentration = p * (1 - p) / v - 1
    concentration = max(concentration, 2.0)  # Minimum concentration
    
    alpha = p * concentration
    beta = (1 - p) * concentration
    
    return float(alpha), float(beta)


def sample_ownership(predicted_pct: float, n: int = 1,
                     residual_std: float = OWNERSHIP_RESIDUAL_STD) -> np.ndarray:
    """Sample ownership from Beta distribution."""
    alpha, beta = ownership_beta_params(predicted_pct, residual_std)
    return np.random.beta(alpha, beta, n) * 100  # Return as percentage


# ═══════════════════════════════════════════════════════════════════
#  Integration Helpers
# ═══════════════════════════════════════════════════════════════════

def upgrade_simulator_sampling(pool: pd.DataFrame) -> np.ndarray:
    """
    Drop-in replacement for simulator._sample_projections.
    Uses Gamma instead of Normal.
    """
    projected = pool['projected_fpts'].values
    positions = pool['position'].values
    salaries = pool['salary'].values
    
    sampled = np.zeros(len(projected))
    for i in range(len(projected)):
        if projected[i] > 0:
            sampled[i] = sample_fpts_gamma(projected[i], positions[i], salaries[i], n=1)[0]
    
    return sampled


def upgrade_bayesian_simulate_skater(goal_r, assist_r, shot_r, block_r, sh_r, n,
                                      use_nb=True):
    """
    Drop-in replacement for SkaterProjector._simulate.
    Uses Negative Binomial instead of Poisson.
    """
    from config import (GOAL_PTS, ASSIST_PTS, SOG_PTS, BLOCK_PTS, SH_BONUS,
                        HAT_TRICK_BONUS, FIVE_SOG_BONUS, THREE_BLOCK_BONUS, THREE_POINT_BONUS)
    
    goals = poisson_or_nb(goal_r, 'goals', use_nb, n)
    assists = poisson_or_nb(assist_r, 'assists', use_nb, n)
    shots = poisson_or_nb(shot_r, 'shots', use_nb, n)
    blocks = poisson_or_nb(block_r, 'blocks', use_nb, n)
    sh = poisson_or_nb(sh_r, 'goals', use_nb, n)  # Short-handed similar to goals
    points = goals + assists

    fpts = (
        goals * GOAL_PTS +
        assists * ASSIST_PTS +
        shots * SOG_PTS +
        blocks * BLOCK_PTS +
        sh * SH_BONUS +
        (goals >= 3) * HAT_TRICK_BONUS +
        (shots >= 5) * FIVE_SOG_BONUS +
        (blocks >= 3) * THREE_BLOCK_BONUS +
        (points >= 3) * THREE_POINT_BONUS
    )
    return fpts


def upgrade_bayesian_simulate_goalie(win_r, e_saves, e_ga, so_r, otl_r, n,
                                      use_nb=True):
    """
    Drop-in replacement for GoalieProjector._simulate.
    Uses NB for saves/GA where overdispersed.
    """
    from config import (WIN_PTS, SAVE_PTS, GA_PTS, SHUTOUT_BONUS, 
                        OTL_PTS, SAVE_35_BONUS)
    
    wins = np.random.binomial(1, min(win_r, 0.99), n)
    saves = poisson_or_nb(e_saves, 'saves', use_nb, n)
    ga = poisson_or_nb(e_ga, 'goals_against', use_nb, n)
    shutouts = (ga == 0).astype(float)
    otl = np.random.binomial(1, min(otl_r, 0.99), n) * (1 - wins)

    fpts = (
        wins * WIN_PTS +
        saves * SAVE_PTS +
        ga * GA_PTS +
        shutouts * SHUTOUT_BONUS +
        otl * OTL_PTS +
        (saves >= 35) * SAVE_35_BONUS
    )
    return fpts
