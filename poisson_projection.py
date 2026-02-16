#!/usr/bin/env python3
"""
poisson_projection.py — Probability-Based NHL DFS Projection Model
===================================================================
Projects player DK fantasy points using Poisson-distributed stat simulations.

Instead of producing a single point estimate, this model estimates the RATE
(λ) for each DK-scoring stat, then simulates thousands of games to produce
a full probability distribution. This gives us:

  1. Expected FPTS (mean of simulated distribution)
  2. Ceiling probability (P(FPTS >= threshold))
  3. Bonus probabilities (hat trick, 3+ pts, 5+ SOG, 3+ blocks)
  4. Floor risk (P(FPTS < 2.0))
  5. Distribution shape (skew, kurtosis, percentiles)

The key insight is that per-60 rates from NST convert to per-game λ values
via: λ = rate_per_60 × (expected_TOI_minutes / 60)

Since TOI varies by situation (5v5, PP, PK), each situation contributes
independently to the overall stat projection.

DraftKings Classic Scoring:
  Goals:   8.5     Assists:   5.0
  SOG:     1.5     Blocks:    1.3
  +/-:     0.5
  Bonuses: Hat trick (3G) +3.0, 3+ Points +3.0,
           5+ SOG +1.0, 3+ Blocks +1.0

Usage:
    # Walk-forward backtest
    python poisson_projection.py --backtest

    # Single-date projection
    python poisson_projection.py --date 2026-02-25

    # Show model diagnostics
    python poisson_projection.py --diagnostics
"""

import argparse
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"

# DraftKings scoring weights
DK_SCORING = {
    'goals': 8.5,
    'assists': 5.0,
    'shots': 1.5,       # SOG
    'blocked_shots': 1.3,
    'plus_minus': 0.5,
}

# DraftKings bonuses
DK_BONUSES = {
    'hat_trick': {'stat': 'goals', 'threshold': 3, 'value': 3.0},
    'multi_point': {'stat': 'points', 'threshold': 3, 'value': 3.0},
    'sog_5': {'stat': 'shots', 'threshold': 5, 'value': 1.0},
    'blocks_3': {'stat': 'blocked_shots', 'threshold': 3, 'value': 1.0},
}

# Number of Monte Carlo simulations per player
N_SIMS = 10_000

# Minimum games required to use a player's own rates
MIN_GAMES_PERSONAL = 5

# Bayesian shrinkage: how much to weight population prior vs personal data
# At MIN_GAMES, weight is ~50/50. At 30+ games, personal data dominates.
SHRINKAGE_STRENGTH = 5  # lower = trust personal data more quickly


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_boxscore_through(conn: sqlite3.Connection, as_of_date: str) -> pd.DataFrame:
    """Load all boxscore data through a given date (inclusive)."""
    return pd.read_sql_query("""
        SELECT player_name, player_id, team, position, game_date, opponent,
               goals, assists, shots, hits, blocked_shots, plus_minus,
               pp_goals, toi_seconds, dk_fpts, game_id
        FROM boxscore_skaters
        WHERE toi_seconds > 0 AND game_date <= ?
        ORDER BY player_name, game_date
    """, conn, params=(as_of_date,))


def load_nst_snapshot(conn: sqlite3.Connection, as_of_date: str,
                      entity: str = 'skaters') -> pd.DataFrame:
    """
    Load the most recent NST snapshot that covers data through as_of_date.

    For backtesting, we need the snapshot where to_date <= as_of_date
    (i.e., only uses data available at that time).
    """
    table = f"nst_{entity}"
    # Find the best matching snapshot (largest to_date that doesn't exceed as_of_date)
    result = conn.execute(f"""
        SELECT DISTINCT to_date FROM {table}
        WHERE to_date <= ? AND from_date = '2025-10-07'
        ORDER BY to_date DESC LIMIT 1
    """, (as_of_date,)).fetchone()

    if not result:
        return pd.DataFrame()

    snap_date = result[0]
    return pd.read_sql_query(f"""
        SELECT * FROM {table}
        WHERE to_date = ? AND from_date = '2025-10-07'
    """, conn, params=(snap_date,))


def parse_nst_toi(toi_str) -> float:
    """Parse NST TOI value to total minutes. NST stores as decimal minutes."""
    if pd.isna(toi_str) or toi_str == '':
        return 0.0
    try:
        return float(str(toi_str).replace(',', ''))
    except (ValueError, TypeError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════
#  RATE ESTIMATION (Per-60 rates from historical data)
# ═══════════════════════════════════════════════════════════════════

def compute_per60_rates(player_games: pd.DataFrame,
                        recency_halflife: int = 15) -> Dict[str, float]:
    """
    Compute per-game stat rates using exponential recency weighting.

    Two modes controlled by USE_PER_GAME_RATES:
    - per-60: rate per 60 min of ice time (traditional)
    - per-game: weighted average stats per game (simpler, less noise)

    halflife=15 means a game 15 games ago has half the weight of the most recent.
    """
    if len(player_games) == 0:
        return {}

    # Sort by date (most recent last)
    pg = player_games.sort_values('game_date').reset_index(drop=True)
    n = len(pg)

    # Exponential decay weights: most recent game = highest weight
    decay = np.log(2) / recency_halflife
    weights = np.exp(decay * np.arange(n))  # increasing weights
    weights /= weights.sum()

    toi_min = pg['toi_seconds'].values / 60.0

    rates = {}

    # PER-GAME rates (used as lambda directly — no TOI conversion needed)
    for stat in ['goals', 'assists', 'shots', 'blocked_shots']:
        rates[f'{stat}_per_game'] = (pg[stat].values * weights).sum()

    # Also compute per-60 rates for NST integration
    weighted_toi = (toi_min * weights).sum()
    if weighted_toi > 0:
        for stat in ['goals', 'assists', 'shots', 'blocked_shots']:
            weighted_total = (pg[stat].values * weights).sum()
            rates[f'{stat}_per60'] = (weighted_total / weighted_toi) * 60.0
    else:
        for stat in ['goals', 'assists', 'shots', 'blocked_shots']:
            rates[f'{stat}_per60'] = 0.0

    # Plus/minus: weighted mean
    rates['plus_minus_mean'] = (pg['plus_minus'].values * weights).sum()
    rates['plus_minus_std'] = max(pg['plus_minus'].std(), 0.5)

    # TOI: use last 10 games for stable recent estimate
    recent = pg.tail(min(10, n))
    toi_recent = recent['toi_seconds'] / 60.0
    rates['toi_mean'] = toi_recent.mean()
    rates['toi_std'] = max(toi_recent.std(), 0.5)

    # FPTS anchor: exponentially weighted average of actual dk_fpts
    # This gives us a better point estimate than decomposed stat rates
    if 'dk_fpts' in pg.columns:
        rates['fpts_anchor'] = (pg['dk_fpts'].values * weights).sum()

    # Games played
    rates['gp'] = len(player_games)

    # PP goals rate (for more accurate goal projection)
    rates['pp_goal_rate'] = player_games['pp_goals'].sum() / max(player_games['goals'].sum(), 1)

    return rates


def compute_population_priors(all_games: pd.DataFrame, position: str) -> Dict[str, float]:
    """
    Compute population-level rates for a given position.
    Used as Bayesian priors for players with limited data.
    """
    pos_games = all_games[all_games['position'] == position]
    if len(pos_games) == 0:
        pos_games = all_games  # fallback

    total_toi_min = pos_games['toi_seconds'].sum() / 60.0

    priors = {}
    for stat in ['goals', 'assists', 'shots', 'blocked_shots']:
        # Per-game rates (primary)
        priors[f'{stat}_per_game'] = pos_games[stat].mean()
        # Per-60 rates (for NST integration)
        if total_toi_min > 0:
            priors[f'{stat}_per60'] = (pos_games[stat].sum() / total_toi_min) * 60.0
        else:
            priors[f'{stat}_per60'] = 0.0

    priors['plus_minus_mean'] = pos_games['plus_minus'].mean()
    priors['plus_minus_std'] = max(pos_games['plus_minus'].std(), 0.5)
    priors['toi_mean'] = (pos_games['toi_seconds'] / 60.0).mean()
    priors['toi_std'] = max((pos_games['toi_seconds'] / 60.0).std(), 1.0)

    return priors


def bayesian_shrink(personal_rate: float, prior_rate: float,
                    n_games: int) -> float:
    """
    Shrink a personal rate toward the population prior based on sample size.

    Uses empirical Bayes: weight = n / (n + k)
    At n=k, it's 50/50. At n >> k, personal dominates.
    """
    weight = n_games / (n_games + SHRINKAGE_STRENGTH)
    return weight * personal_rate + (1 - weight) * prior_rate


# ═══════════════════════════════════════════════════════════════════
#  NST SITUATION ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════

def get_nst_situation_adjustments(nst_skaters: pd.DataFrame,
                                  nst_teams: pd.DataFrame,
                                  player_name: str,
                                  opponent: str = None) -> Dict[str, float]:
    """
    Use NST situation-specific data to adjust projections.

    Returns multipliers for how much better/worse a player performs
    in different situations relative to their overall rate.

    Key adjustments:
    1. PP boost: How much does this player's production increase on PP?
    2. Opponent quality: How does opponent's xGF%/SV% affect scoring?
    """
    adjustments = {'pp_boost': 1.0, 'opp_factor': 1.0}

    if nst_skaters.empty:
        return adjustments

    # PP boost from individual stats
    player_std = nst_skaters[
        (nst_skaters['stat_type'] == 'std') &
        (nst_skaters['player'] == player_name)
    ]

    if len(player_std) > 0:
        pp_row = player_std[player_std['situation'] == 'pp']
        ev_row = player_std[player_std['situation'] == '5v5']

        if len(pp_row) > 0 and len(ev_row) > 0:
            pp_toi = parse_nst_toi(pp_row.iloc[0].get('toi', 0))
            ev_toi = parse_nst_toi(ev_row.iloc[0].get('toi', 0))

            if pp_toi > 0 and ev_toi > 0:
                pp_goals_p60 = (pp_row.iloc[0].get('goals', 0) or 0) / pp_toi * 60
                ev_goals_p60 = (ev_row.iloc[0].get('goals', 0) or 0) / ev_toi * 60

                if ev_goals_p60 > 0:
                    adjustments['pp_boost'] = max(1.0, pp_goals_p60 / ev_goals_p60)

    # Opponent quality adjustment
    if opponent and not nst_teams.empty:
        opp_5v5 = nst_teams[
            (nst_teams['situation'] == '5v5') &
            (nst_teams['team'] == opponent)
        ]
        if len(opp_5v5) > 0:
            opp_xgf = opp_5v5.iloc[0].get('xgf_pct')
            opp_sv = opp_5v5.iloc[0].get('sv_pct')

            if opp_xgf is not None:
                # Lower opponent xGF% = weaker team = easier opponent
                # Normalize: 50% = neutral, <50% = easy, >50% = hard
                adjustments['opp_factor'] = 1.0 + (50.0 - float(opp_xgf)) * 0.005

            if opp_sv is not None:
                # Lower opponent SV% = leakier goaltending = more goals
                # Normalize around league avg ~90.5%
                sv_adj = (90.5 - float(opp_sv)) * 0.03
                adjustments['opp_factor'] *= (1.0 + sv_adj)

    return adjustments


# ═══════════════════════════════════════════════════════════════════
#  POISSON SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def simulate_player_game(rates: Dict[str, float],
                         adjustments: Dict[str, float] = None,
                         n_sims: int = N_SIMS) -> Dict:
    """
    Simulate n_sims games for a player using Poisson distributions.

    Each stat is drawn from Poisson(λ) where λ = per_60_rate × (toi / 60).
    Plus/minus is drawn from Normal(mean, std).
    FPTS is computed including bonuses.

    Returns dict with full distribution statistics.
    """
    adj = adjustments or {'pp_boost': 1.0, 'opp_factor': 1.0}
    rng = np.random.default_rng()

    results = {}
    toi_samples = np.full(n_sims, rates.get('toi_mean', 15.0))

    # ── Step 1: Compute raw Poisson lambdas from per-game rates ──
    lambda_goals = rates.get('goals_per_game',
                             rates.get('goals_per60', 0) * rates.get('toi_mean', 15) / 60)
    lambda_assists = rates.get('assists_per_game',
                               rates.get('assists_per60', 0) * rates.get('toi_mean', 15) / 60)
    lambda_shots = rates.get('shots_per_game',
                             rates.get('shots_per60', 0) * rates.get('toi_mean', 15) / 60)
    lambda_blocks = rates.get('blocked_shots_per_game',
                              rates.get('blocked_shots_per60', 0) * rates.get('toi_mean', 15) / 60)

    # ── Step 2: Calibrate to FPTS anchor ──
    # If we have a direct FPTS estimate (from Kalman or weighted avg),
    # scale lambdas so the expected FPTS matches it. This preserves
    # the distribution SHAPE while anchoring to a better point estimate.
    if 'fpts_anchor' in rates and rates['fpts_anchor'] > 0:
        # Compute what the raw lambdas would give us
        raw_expected = (lambda_goals * DK_SCORING['goals'] +
                       lambda_assists * DK_SCORING['assists'] +
                       lambda_shots * DK_SCORING['shots'] +
                       lambda_blocks * DK_SCORING['blocked_shots'] +
                       rates.get('plus_minus_mean', 0) * DK_SCORING['plus_minus'])
        if raw_expected > 0:
            scale = rates['fpts_anchor'] / raw_expected
            # Don't scale too aggressively — clamp to [0.5, 2.0]
            scale = max(0.5, min(2.0, scale))
            lambda_goals *= scale
            lambda_assists *= scale
            lambda_shots *= scale
            lambda_blocks *= scale

    # Apply opponent quality adjustment to offensive stats
    lambda_goals *= adj['opp_factor']
    lambda_assists *= adj['opp_factor']

    # Simulate from Poisson
    goals = rng.poisson(np.maximum(lambda_goals, 0.001), n_sims)
    results['goals'] = goals

    assists = rng.poisson(np.maximum(lambda_assists, 0.001), n_sims)
    results['assists'] = assists

    shots = rng.poisson(np.maximum(lambda_shots, 0.001), n_sims)
    shots = np.maximum(shots, goals)  # can't score without shooting
    results['shots'] = shots

    blocks = rng.poisson(np.maximum(lambda_blocks, 0.001), n_sims)
    results['blocked_shots'] = blocks

    # Plus/minus: Normal (not Poisson — can be negative)
    pm = np.round(rng.normal(rates['plus_minus_mean'], rates['plus_minus_std'], n_sims))
    results['plus_minus'] = pm.astype(int)

    # Points
    points = goals + assists
    results['points'] = points

    # ── Compute FPTS ────────────────────────────────────────────
    fpts = (
        goals * DK_SCORING['goals'] +
        assists * DK_SCORING['assists'] +
        shots * DK_SCORING['shots'] +
        blocks * DK_SCORING['blocked_shots'] +
        pm * DK_SCORING['plus_minus']
    )

    # Add bonuses
    fpts += (goals >= 3) * DK_BONUSES['hat_trick']['value']       # Hat trick
    fpts += (points >= 3) * DK_BONUSES['multi_point']['value']    # 3+ points
    fpts += (shots >= 5) * DK_BONUSES['sog_5']['value']           # 5+ SOG
    fpts += (blocks >= 3) * DK_BONUSES['blocks_3']['value']       # 3+ blocks

    results['fpts'] = fpts

    # ── Distribution statistics ─────────────────────────────────
    output = {
        'expected_fpts': float(np.mean(fpts)),
        'median_fpts': float(np.median(fpts)),
        'std_fpts': float(np.std(fpts)),
        'floor_fpts': float(np.percentile(fpts, 10)),
        'ceiling_fpts': float(np.percentile(fpts, 90)),

        # Probability thresholds
        'p_above_10': float(np.mean(fpts >= 10)),
        'p_above_15': float(np.mean(fpts >= 15)),
        'p_above_20': float(np.mean(fpts >= 20)),
        'p_above_25': float(np.mean(fpts >= 25)),
        'p_below_2': float(np.mean(fpts < 2)),

        # Bonus probabilities
        'p_hat_trick': float(np.mean(goals >= 3)),
        'p_3plus_points': float(np.mean(points >= 3)),
        'p_5plus_sog': float(np.mean(shots >= 5)),
        'p_3plus_blocks': float(np.mean(blocks >= 3)),

        # Stat expectations
        'exp_goals': float(np.mean(goals)),
        'exp_assists': float(np.mean(assists)),
        'exp_shots': float(np.mean(shots)),
        'exp_blocks': float(np.mean(blocks)),
        'exp_pm': float(np.mean(pm)),
        'exp_toi': float(np.mean(toi_samples)),

        # Distribution shape
        'skewness': float(scipy_stats.skew(fpts)),
        'kurtosis': float(scipy_stats.kurtosis(fpts)),

        # Raw lambda values (for diagnostics)
        'lambda_goals': float(lambda_goals) if np.isscalar(lambda_goals) else float(np.mean(lambda_goals)),
        'lambda_assists': float(lambda_assists) if np.isscalar(lambda_assists) else float(np.mean(lambda_assists)),
        'lambda_shots': float(lambda_shots) if np.isscalar(lambda_shots) else float(np.mean(lambda_shots)),
        'lambda_blocks': float(lambda_blocks) if np.isscalar(lambda_blocks) else float(np.mean(lambda_blocks)),
    }

    return output


# ═══════════════════════════════════════════════════════════════════
#  PROJECTION ENGINE
# ═══════════════════════════════════════════════════════════════════

class PoissonProjectionModel:
    """
    Full Poisson projection model with Bayesian rate estimation
    and NST situation adjustments.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)

    def project_slate(self, as_of_date: str,
                      players: List[Dict] = None) -> pd.DataFrame:
        """
        Generate projections for a slate of players.

        Args:
            as_of_date: Project games ON this date using data through
                        the day before.
            players: Optional list of player dicts with 'name', 'team',
                     'opponent'. If None, projects all players who have
                     games on as_of_date.

        Returns:
            DataFrame with full projection distribution for each player.
        """
        conn = sqlite3.connect(self.db_path)

        # Load historical data through the day before projection date
        prev_date = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        boxscore = load_boxscore_through(conn, prev_date)

        if boxscore.empty:
            conn.close()
            return pd.DataFrame()

        # Load NST snapshots (most recent available)
        nst_skaters = load_nst_snapshot(conn, prev_date, 'skaters')
        nst_teams = load_nst_snapshot(conn, prev_date, 'teams')

        # Compute population priors by position
        priors = {}
        for pos in boxscore['position'].unique():
            priors[pos] = compute_population_priors(boxscore, pos)

        # If no player list provided, use players who played on as_of_date
        if players is None:
            game_day = pd.read_sql_query("""
                SELECT DISTINCT player_name, team, position, opponent
                FROM boxscore_skaters
                WHERE game_date = ? AND toi_seconds > 0
            """, conn, params=(as_of_date,))
            players = game_day.to_dict('records')

        conn.close()

        if not players:
            return pd.DataFrame()

        # Project each player
        projections = []
        for p in players:
            name = p.get('player_name', p.get('name', ''))
            team = p.get('team', '')
            position = p.get('position', 'C')
            opponent = p.get('opponent', '')

            # Get player's game history
            player_games = boxscore[boxscore['player_name'] == name]

            if len(player_games) < MIN_GAMES_PERSONAL:
                # Not enough data — use population priors
                prior = priors.get(position, priors.get('C', {}))
                if not prior:
                    continue
                rates = prior.copy()
                rates['gp'] = len(player_games)
            else:
                # Compute personal rates
                personal = compute_per60_rates(player_games)
                prior = priors.get(position, {})

                # Bayesian shrinkage
                rates = {}
                gp = personal.get('gp', 0)
                for key in ['goals_per_game', 'assists_per_game', 'shots_per_game',
                            'blocked_shots_per_game',
                            'goals_per60', 'assists_per60', 'shots_per60',
                            'blocked_shots_per60']:
                    if key in personal and key in prior:
                        rates[key] = bayesian_shrink(personal[key], prior[key], gp)
                    elif key in personal:
                        rates[key] = personal[key]
                    elif key in prior:
                        rates[key] = prior[key]
                    else:
                        rates[key] = 0.0

                rates['plus_minus_mean'] = personal.get('plus_minus_mean',
                                                         prior.get('plus_minus_mean', 0))
                rates['plus_minus_std'] = personal.get('plus_minus_std',
                                                       prior.get('plus_minus_std', 1.0))
                rates['toi_mean'] = personal.get('toi_mean', prior.get('toi_mean', 15.0))
                rates['toi_std'] = personal.get('toi_std', prior.get('toi_std', 3.0))
                rates['gp'] = gp

            # Get NST situation adjustments
            adjustments = get_nst_situation_adjustments(
                nst_skaters, nst_teams, name, opponent
            )

            # Run simulation
            sim = simulate_player_game(rates, adjustments)

            # Build result row
            result = {
                'player_name': name,
                'team': team,
                'position': position,
                'opponent': opponent,
                'games_used': rates.get('gp', 0),
                **sim,
                'pp_boost': adjustments['pp_boost'],
                'opp_factor': adjustments['opp_factor'],
            }
            projections.append(result)

        df = pd.DataFrame(projections)
        if not df.empty:
            df = df.sort_values('expected_fpts', ascending=False)
        return df


    def backtest(self, start_date: str = "2025-11-07",
                 end_date: str = "2026-02-05",
                 verbose: bool = True) -> Dict:
        """
        Walk-forward backtest: for each game date, project all players
        using only data available before that date, then compare to actuals.

        Returns comprehensive accuracy metrics.
        """
        conn = sqlite3.connect(self.db_path)

        # Get all game dates in range
        game_dates = pd.read_sql_query("""
            SELECT DISTINCT game_date FROM boxscore_skaters
            WHERE game_date >= ? AND game_date <= ?
            ORDER BY game_date
        """, conn, params=(start_date, end_date))['game_date'].tolist()

        if verbose:
            print(f"\n{'='*70}")
            print(f"  POISSON MODEL WALK-FORWARD BACKTEST")
            print(f"  Date range: {start_date} to {end_date}")
            print(f"  Game dates: {len(game_dates)}")
            print(f"  Simulations per player: {N_SIMS:,}")
            print(f"{'='*70}\n")

        all_results = []
        t0 = time.time()

        for i, gd in enumerate(game_dates):
            # Get actual results for this date
            actuals = pd.read_sql_query("""
                SELECT player_name, team, position, opponent,
                       dk_fpts, goals, assists, shots, blocked_shots,
                       toi_seconds
                FROM boxscore_skaters
                WHERE game_date = ? AND toi_seconds > 0
            """, conn, params=(gd,))

            if actuals.empty:
                continue

            # Project using the model
            proj = self.project_slate(gd)

            if proj.empty:
                continue

            # Merge projections with actuals
            merged = actuals.merge(
                proj[['player_name', 'expected_fpts', 'median_fpts',
                      'ceiling_fpts', 'floor_fpts', 'std_fpts',
                      'p_above_10', 'p_above_15', 'p_above_20',
                      'p_hat_trick', 'p_3plus_points', 'p_5plus_sog',
                      'p_3plus_blocks', 'games_used',
                      'exp_goals', 'exp_assists', 'exp_shots', 'exp_blocks',
                      'pp_boost', 'opp_factor']],
                on='player_name', how='inner'
            )

            if merged.empty:
                continue

            merged['game_date'] = gd
            merged['error'] = merged['expected_fpts'] - merged['dk_fpts']
            merged['abs_error'] = merged['error'].abs()

            all_results.append(merged)

            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                mae_so_far = pd.concat(all_results)['abs_error'].mean()
                print(f"  [{i+1}/{len(game_dates)}] {gd} | "
                      f"Players: {len(merged)} | "
                      f"Running MAE: {mae_so_far:.3f} | "
                      f"Elapsed: {elapsed:.0f}s")

        conn.close()

        if not all_results:
            print("No results generated!")
            return {}

        results_df = pd.concat(all_results, ignore_index=True)
        elapsed = time.time() - t0

        # ── Compute metrics ─────────────────────────────────────
        metrics = self._compute_backtest_metrics(results_df)

        if verbose:
            self._print_backtest_report(metrics, results_df, elapsed)

        return {
            'metrics': metrics,
            'results': results_df,
            'elapsed_seconds': elapsed,
        }

    def _compute_backtest_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute comprehensive backtest accuracy metrics."""
        metrics = {}

        # Overall accuracy
        metrics['n_predictions'] = len(df)
        metrics['n_players'] = df['player_name'].nunique()
        metrics['n_dates'] = df['game_date'].nunique()
        metrics['mae'] = df['abs_error'].mean()
        metrics['median_ae'] = df['abs_error'].median()
        metrics['rmse'] = np.sqrt((df['error'] ** 2).mean())
        metrics['mean_bias'] = df['error'].mean()
        metrics['correlation'] = df['expected_fpts'].corr(df['dk_fpts'])

        # Baseline comparisons
        # Season average baseline (NOTE: uses all data = slight look-ahead bias,
        # but provides standard comparison point)
        player_avgs = df.groupby('player_name')['dk_fpts'].transform('mean')
        metrics['baseline_season_avg_mae'] = (player_avgs - df['dk_fpts']).abs().mean()

        # Walk-forward baseline: expanding mean up to each game
        df_sorted = df.sort_values(['player_name', 'game_date'])
        wf_avg = df_sorted.groupby('player_name')['dk_fpts'].expanding().mean().shift(1)
        wf_avg = wf_avg.droplevel(0)
        valid_wf = wf_avg.dropna()
        if len(valid_wf) > 0:
            wf_mae = (wf_avg.loc[valid_wf.index] - df_sorted.loc[valid_wf.index, 'dk_fpts']).abs().mean()
            metrics['baseline_walkforward_avg_mae'] = wf_mae

        # By position
        metrics['by_position'] = {}
        for pos in sorted(df['position'].unique()):
            pos_df = df[df['position'] == pos]
            if len(pos_df) > 50:
                metrics['by_position'][pos] = {
                    'mae': pos_df['abs_error'].mean(),
                    'n': len(pos_df),
                    'corr': pos_df['expected_fpts'].corr(pos_df['dk_fpts']),
                }

        # Ceiling accuracy: when we predict high ceiling, do they deliver?
        high_ceil = df[df['p_above_15'] >= 0.20]  # predicted 20%+ chance of 15+ FPTS
        if len(high_ceil) > 0:
            metrics['ceiling_hit_rate'] = (high_ceil['dk_fpts'] >= 15).mean()
            metrics['ceiling_n'] = len(high_ceil)

        # Floor accuracy: when we predict low floor risk, are they safe?
        safe = df[df['floor_fpts'] >= 3.0]  # predicted 10th percentile >= 3
        if len(safe) > 0:
            metrics['floor_miss_rate'] = (safe['dk_fpts'] < 2).mean()
            metrics['floor_n'] = len(safe)

        # Bonus prediction accuracy
        for bonus_name, bonus_info in DK_BONUSES.items():
            stat_col = bonus_info['stat']
            thresh = bonus_info['threshold']
            p_col = f'p_{bonus_name}'

            if stat_col == 'points':
                actual_hit = (df['goals'] + df['assists']) >= thresh
            elif stat_col in df.columns:
                actual_hit = df[stat_col] >= thresh
            else:
                continue

            if p_col in df.columns:
                # Brier score for probability calibration
                predicted_p = df[p_col]
                brier = ((predicted_p - actual_hit.astype(float)) ** 2).mean()
                actual_rate = actual_hit.mean()
                predicted_rate = predicted_p.mean()
                metrics[f'bonus_{bonus_name}_brier'] = brier
                metrics[f'bonus_{bonus_name}_actual_rate'] = actual_rate
                metrics[f'bonus_{bonus_name}_predicted_rate'] = predicted_rate

        # Stat-level accuracy
        for stat in ['goals', 'assists', 'shots', 'blocked_shots']:
            exp_col = f'exp_{stat.replace("blocked_shots", "blocks")}'
            if exp_col in df.columns and stat in df.columns:
                mae_stat = (df[exp_col] - df[stat]).abs().mean()
                corr_stat = df[exp_col].corr(df[stat])
                metrics[f'stat_{stat}_mae'] = mae_stat
                metrics[f'stat_{stat}_corr'] = corr_stat

        return metrics

    def _print_backtest_report(self, metrics: Dict, df: pd.DataFrame,
                               elapsed: float):
        """Print formatted backtest report."""
        print(f"\n{'='*70}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"  Predictions:    {metrics['n_predictions']:,}")
        print(f"  Players:        {metrics['n_players']}")
        print(f"  Game dates:     {metrics['n_dates']}")
        print(f"  Elapsed:        {elapsed:.1f}s")
        print(f"\n  ── FPTS Accuracy ──")
        print(f"  MAE:            {metrics['mae']:.4f}")
        print(f"  Median AE:      {metrics['median_ae']:.4f}")
        print(f"  RMSE:           {metrics['rmse']:.4f}")
        print(f"  Bias:           {metrics['mean_bias']:+.4f}")
        print(f"  Correlation:    {metrics['correlation']:.4f}")
        print(f"\n  ── Baseline Comparison ──")
        print(f"  Season Avg MAE: {metrics['baseline_season_avg_mae']:.4f}")
        improvement = metrics['baseline_season_avg_mae'] - metrics['mae']
        pct = improvement / metrics['baseline_season_avg_mae'] * 100
        print(f"  Improvement:    {improvement:+.4f} ({pct:+.1f}%)")
        print(f"  Kalman ref MAE: 4.318 (from previous backtest)")

        print(f"\n  ── By Position ──")
        for pos, pos_metrics in sorted(metrics.get('by_position', {}).items()):
            print(f"  {pos:>2s}: MAE={pos_metrics['mae']:.3f}, "
                  f"corr={pos_metrics['corr']:.3f}, n={pos_metrics['n']}")

        if 'ceiling_hit_rate' in metrics:
            print(f"\n  ── Ceiling Accuracy ──")
            print(f"  Predicted 20%+ chance of 15+ FPTS: {metrics['ceiling_n']} players")
            print(f"  Actually hit 15+ FPTS: {metrics['ceiling_hit_rate']*100:.1f}%")

        if 'floor_miss_rate' in metrics:
            print(f"\n  ── Floor Accuracy ──")
            print(f"  Predicted safe floor (10p >= 3): {metrics['floor_n']} players")
            print(f"  Busted below 2 FPTS: {metrics['floor_miss_rate']*100:.1f}%")

        print(f"\n  ── Bonus Calibration ──")
        for bonus in ['hat_trick', '3plus_points', '5plus_sog', '3plus_blocks']:
            brier_key = f'bonus_{bonus}_brier'
            if brier_key in metrics:
                print(f"  {bonus:>14s}: predicted={metrics[f'bonus_{bonus}_predicted_rate']*100:.1f}%, "
                      f"actual={metrics[f'bonus_{bonus}_actual_rate']*100:.1f}%, "
                      f"Brier={metrics[brier_key]:.4f}")

        print(f"\n  ── Stat-Level Accuracy ──")
        for stat in ['goals', 'assists', 'shots', 'blocked_shots']:
            mae_key = f'stat_{stat}_mae'
            if mae_key in metrics:
                print(f"  {stat:>14s}: MAE={metrics[mae_key]:.3f}, "
                      f"corr={metrics.get(f'stat_{stat}_corr', 0):.3f}")

        print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisson DFS Projection Model")
    parser.add_argument("--backtest", action="store_true",
                        help="Run walk-forward backtest")
    parser.add_argument("--start-date", type=str, default="2025-11-07",
                        help="Backtest start date")
    parser.add_argument("--end-date", type=str, default="2026-02-05",
                        help="Backtest end date")
    parser.add_argument("--date", type=str, default=None,
                        help="Project for a specific date")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Show model diagnostics")
    args = parser.parse_args()

    model = PoissonProjectionModel()

    if args.backtest:
        result = model.backtest(
            start_date=args.start_date,
            end_date=args.end_date,
        )
    elif args.date:
        proj = model.project_slate(args.date)
        if not proj.empty:
            print(f"\nProjections for {args.date}:")
            print(proj[['player_name', 'team', 'position', 'expected_fpts',
                        'ceiling_fpts', 'p_above_15', 'p_hat_trick',
                        'games_used']].head(30).to_string())
        else:
            print(f"No projections generated for {args.date}")
    elif args.diagnostics:
        print("Diagnostics mode — checking data availability...")
        conn = sqlite3.connect(str(DB_PATH))

        # NST snapshot availability
        for table in ['nst_teams', 'nst_skaters', 'nst_goalies']:
            dates = conn.execute(f"""
                SELECT DISTINCT to_date, COUNT(*) as n
                FROM {table} WHERE from_date = '2025-10-07'
                GROUP BY to_date ORDER BY to_date
            """).fetchall()
            print(f"\n{table} snapshots:")
            for d, n in dates:
                print(f"  through {d}: {n} rows")

        conn.close()
    else:
        parser.print_help()
