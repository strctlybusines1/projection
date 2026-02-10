#!/usr/bin/env python3
"""
Bayesian Player-Event Probability Model for NHL DFS
====================================================

Instead of projecting a single FPTS number, models each scoring event
independently using Bayesian inference:

Skaters:
    P(goals=0), P(goals=1), P(goals=2+)
    P(assists=0), P(assists=1), P(assists=2+)
    P(shots>=5)  → 3pt bonus
    P(blocks>=3) → 3pt bonus
    P(points>=3) → 3pt bonus
    P(SH point)

Goalies:
    P(win)
    E[saves]
    E[goals_against]
    P(shutout)
    P(OT loss)
    P(saves>=35) → 3pt bonus

Uses Beta-Binomial conjugate priors:
    - Prior: league-average rate per game for each event
    - Update: player's game log (weighted by recency)
    - Posterior: updated probability of each event

Then computes:
    - Expected FPTS from event probabilities (comparable to current model)
    - Full distribution via Monte Carlo simulation
    - True floor (10th percentile) and ceiling (90th percentile)

Usage:
    python bayesian_projections.py --date 2026-02-25
    python bayesian_projections.py --compare    # Compare to current model

    from bayesian_projections import BayesianProjector
    bp = BayesianProjector()
    projections = bp.project_slate(player_pool, vegas_data)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# DK Scoring
GOAL_PTS = 8.5
ASSIST_PTS = 5.0
SOG_PTS = 1.5
BLOCK_PTS = 1.3
SH_BONUS = 2.0
HAT_TRICK_BONUS = 3.0
FIVE_SOG_BONUS = 3.0
THREE_BLOCK_BONUS = 3.0
THREE_POINT_BONUS = 3.0
# Goalie
WIN_PTS = 6.0
SAVE_PTS = 0.7
GA_PTS = -3.5
SHUTOUT_BONUS = 4.0
OTL_PTS = 2.0
SAVE_35_BONUS = 3.0

# Number of Monte Carlo simulations per player
N_SIMS = 5000


# ================================================================
#  League Priors (2025-26 season averages)
# ================================================================

# These are the "uninformed" priors — league average rates per game.
# They anchor the model when we have little data on a player.
# Source: approximate from NHL stats, tuned to match observed distributions.

LEAGUE_PRIORS = {
    # Skater priors (per game rates)
    'goals_rate': 0.135,        # ~0.135 goals/game for avg skater
    'assists_rate': 0.22,       # ~0.22 assists/game
    'shots_rate': 2.3,          # ~2.3 SOG/game
    'blocks_rate': 0.85,        # ~0.85 blocks/game
    'sh_points_rate': 0.008,    # ~0.8% chance of SH point per game

    # Prior strength (number of pseudo-observations)
    # Lower = player data dominates faster
    # Higher = more conservative, slower to update
    'goals_prior_n': 15,
    'assists_prior_n': 15,
    'shots_prior_n': 10,
    'blocks_prior_n': 10,
    'sh_prior_n': 30,           # Rare event, need more data

    # Goalie priors
    'win_rate': 0.50,
    'save_pct': 0.905,
    'ga_rate': 2.8,             # Goals against per game
    'shutout_rate': 0.06,       # ~6% of starts
    'shots_against_rate': 29.0, # Average shots faced
    'goalie_prior_n': 12,
}

# Position-specific prior adjustments
POSITION_GOAL_PRIORS = {
    'C': 0.16,   # Centers score more
    'W': 0.14,   # Wingers slightly less
    'D': 0.06,   # Defensemen much less
}

POSITION_ASSIST_PRIORS = {
    'C': 0.28,
    'W': 0.20,
    'D': 0.22,
}

POSITION_SHOT_PRIORS = {
    'C': 2.6,
    'W': 2.4,
    'D': 1.8,
}


# ================================================================
#  Bayesian Updater
# ================================================================

class BayesianUpdater:
    """
    Beta-Binomial conjugate prior updater for event rates.

    For rate-based events (goals per game, etc.):
        prior:     Beta(alpha, beta) where alpha = rate * n, beta = (1-rate) * n
        update:    observe k successes in m trials
        posterior: Beta(alpha + k, beta + m - k)

    The posterior mean is a weighted blend of prior and observed rate,
    with the weight determined by how much data we have.
    """

    @staticmethod
    def update_rate(prior_rate: float, prior_n: float,
                    observed_count: float, games_played: int,
                    recency_weight: float = 1.0) -> Tuple[float, float, float]:
        """
        Update a per-game event rate with observed data.

        Args:
            prior_rate: League average rate (e.g., 0.135 goals/game)
            prior_n: Prior strength (pseudo-observations)
            observed_count: Total events observed (e.g., 5 goals)
            games_played: Number of games in sample
            recency_weight: Weight for recent games (1.0 = all equal)

        Returns:
            (posterior_rate, lower_95, upper_95)
        """
        # Effective games (discounted by recency if needed)
        eff_games = games_played * recency_weight

        # Beta prior parameters
        alpha_prior = prior_rate * prior_n
        beta_prior = (1 - prior_rate) * prior_n

        # For Poisson-like events (goals, assists), convert to rate
        # Treat as: in each "trial" (game), event happened at rate r
        # Approximate with binomial: k successes in m trials
        # where success prob = prior_rate
        alpha_post = alpha_prior + observed_count
        beta_post = beta_prior + eff_games - observed_count

        # Clamp to valid range
        alpha_post = max(alpha_post, 0.01)
        beta_post = max(beta_post, 0.01)

        posterior_rate = alpha_post / (alpha_post + beta_post)

        # 95% credible interval
        lower = stats.beta.ppf(0.025, alpha_post, beta_post)
        upper = stats.beta.ppf(0.975, alpha_post, beta_post)

        return posterior_rate, lower, upper

    @staticmethod
    def poisson_probs(rate: float, max_k: int = 5) -> Dict[int, float]:
        """
        Compute P(X=k) for a Poisson distribution.

        Returns dict: {0: P(0), 1: P(1), ..., max_k: P(>=max_k)}
        """
        probs = {}
        for k in range(max_k):
            probs[k] = stats.poisson.pmf(k, rate)
        # P(>= max_k) is the tail
        probs[max_k] = 1 - stats.poisson.cdf(max_k - 1, rate)
        return probs


# ================================================================
#  Skater Projection
# ================================================================

class SkaterProjector:
    """Project a single skater's DK FPTS using Bayesian event probabilities."""

    def __init__(self):
        self.updater = BayesianUpdater()

    def project(self, player: dict, matchup_factor: float = 1.0) -> Dict:
        """
        Project a skater's expected FPTS and distribution.

        Args:
            player: Dict with keys:
                - goals_per_game, assists_per_game, shots_per_game
                - games_played (or gp)
                - position (C, W, D)
                - blocks_per_game (optional)
                - sh_points (optional)
                - pp_points_per_game (optional)
            matchup_factor: Multiplier for matchup quality (1.0 = neutral)

        Returns:
            Dict with expected_fpts, floor, ceiling, event probabilities
        """
        pos = player.get('position', 'C')
        gp = player.get('games_played', player.get('gp', 20))
        gp = max(gp, 1)

        # --- Goals ---
        goals_prior = POSITION_GOAL_PRIORS.get(pos, LEAGUE_PRIORS['goals_rate'])
        goals_observed = player.get('goals_per_game', goals_prior) * gp
        goal_rate, goal_lo, goal_hi = self.updater.update_rate(
            goals_prior, LEAGUE_PRIORS['goals_prior_n'],
            goals_observed, gp
        )
        goal_rate *= matchup_factor

        # --- Assists ---
        assists_prior = POSITION_ASSIST_PRIORS.get(pos, LEAGUE_PRIORS['assists_rate'])
        assists_observed = player.get('assists_per_game', assists_prior) * gp
        assist_rate, _, _ = self.updater.update_rate(
            assists_prior, LEAGUE_PRIORS['assists_prior_n'],
            assists_observed, gp
        )
        assist_rate *= matchup_factor

        # --- Shots ---
        shots_prior = POSITION_SHOT_PRIORS.get(pos, LEAGUE_PRIORS['shots_rate'])
        shots_observed = player.get('shots_per_game', shots_prior) * gp
        shot_rate_raw, _, _ = self.updater.update_rate(
            shots_prior / 10, LEAGUE_PRIORS['shots_prior_n'],
            shots_observed / 10, gp  # Normalize to [0,1] range
        )
        shot_rate = shot_rate_raw * 10  # Scale back

        # --- Blocks ---
        blocks_prior = LEAGUE_PRIORS['blocks_rate']
        if pos == 'D':
            blocks_prior = 1.4  # D-men block more
        blocks_observed = player.get('blocks_per_game', blocks_prior) * gp
        block_rate_raw, _, _ = self.updater.update_rate(
            blocks_prior / 5, LEAGUE_PRIORS['blocks_prior_n'],
            blocks_observed / 5, gp
        )
        block_rate = block_rate_raw * 5

        # --- SH points ---
        sh_rate = LEAGUE_PRIORS['sh_points_rate']
        if player.get('sh_points', 0) > 0:
            sh_observed = player['sh_points']
            sh_rate, _, _ = self.updater.update_rate(
                LEAGUE_PRIORS['sh_points_rate'],
                LEAGUE_PRIORS['sh_prior_n'],
                sh_observed, gp
            )

        # --- Event probabilities ---
        goal_probs = self.updater.poisson_probs(goal_rate)
        assist_probs = self.updater.poisson_probs(assist_rate)

        p_hat_trick = 1 - stats.poisson.cdf(2, goal_rate)  # P(goals >= 3)
        p_five_sog = 1 - stats.poisson.cdf(4, shot_rate)    # P(SOG >= 5)
        p_three_blocks = 1 - stats.poisson.cdf(2, block_rate)  # P(blocks >= 3)
        p_three_points = 1 - stats.poisson.cdf(2, goal_rate + assist_rate)  # P(G+A >= 3)

        # --- Expected FPTS (analytical) ---
        e_goals = goal_rate
        e_assists = assist_rate
        e_shots = shot_rate
        e_blocks = block_rate
        e_sh = sh_rate

        expected_fpts = (
            e_goals * GOAL_PTS +
            e_assists * ASSIST_PTS +
            e_shots * SOG_PTS +
            e_blocks * BLOCK_PTS +
            e_sh * SH_BONUS +
            p_hat_trick * HAT_TRICK_BONUS +
            p_five_sog * FIVE_SOG_BONUS +
            p_three_blocks * THREE_BLOCK_BONUS +
            p_three_points * THREE_POINT_BONUS
        )

        # --- Monte Carlo for distribution ---
        sims = self._simulate(goal_rate, assist_rate, shot_rate,
                              block_rate, sh_rate, N_SIMS)

        return {
            'expected_fpts': round(expected_fpts, 2),
            'median_fpts': round(np.median(sims), 2),
            'floor': round(np.percentile(sims, 10), 2),
            'ceiling': round(np.percentile(sims, 90), 2),
            'p5': round(np.percentile(sims, 5), 2),
            'p95': round(np.percentile(sims, 95), 2),
            'std': round(np.std(sims), 2),
            # Event probabilities
            'p_goal': round(1 - goal_probs[0], 3),
            'p_multi_goal': round(1 - goal_probs[0] - goal_probs[1], 3),
            'p_assist': round(1 - assist_probs[0], 3),
            'p_multi_assist': round(1 - assist_probs[0] - assist_probs[1], 3),
            'p_five_sog': round(p_five_sog, 3),
            'p_three_blocks': round(p_three_blocks, 3),
            'p_hat_trick': round(p_hat_trick, 4),
            'p_three_points': round(p_three_points, 3),
            # Rates
            'goal_rate': round(goal_rate, 4),
            'assist_rate': round(assist_rate, 4),
            'shot_rate': round(shot_rate, 2),
            'block_rate': round(block_rate, 2),
        }

    def _simulate(self, goal_r, assist_r, shot_r, block_r, sh_r, n: int) -> np.ndarray:
        """Monte Carlo simulation of FPTS outcomes using Negative Binomial.
        
        NB provides heavier tails than Poisson (Var = mu + mu²/r > mu),
        better modeling boom/bust games. Falls back to Poisson if unavailable.
        """
        try:
            from stochastic_upgrades import poisson_or_nb
            goals = poisson_or_nb(goal_r, 'goals', True, n)
            assists = poisson_or_nb(assist_r, 'assists', True, n)
            shots = poisson_or_nb(shot_r, 'shots', True, n)
            blocks = poisson_or_nb(block_r, 'blocks', True, n)
            sh = poisson_or_nb(sh_r, 'goals', True, n)
        except ImportError:
            goals = np.random.poisson(goal_r, n)
            assists = np.random.poisson(assist_r, n)
            shots = np.random.poisson(shot_r, n)
            blocks = np.random.poisson(block_r, n)
            sh = np.random.poisson(sh_r, n)
        
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


# ================================================================
#  Goalie Projection
# ================================================================

class GoalieProjector:
    """Project a goalie's DK FPTS using Bayesian event probabilities."""

    def __init__(self):
        self.updater = BayesianUpdater()

    def project(self, goalie: dict, opp_implied_total: float = None) -> Dict:
        """
        Project a goalie's expected FPTS and distribution.

        Args:
            goalie: Dict with save_pct, games_started, wins, shutouts, etc.
            opp_implied_total: Vegas implied goals for opponent (optional)
        """
        gp = goalie.get('games_started', goalie.get('gp', 15))
        gp = max(gp, 1)

        # --- Win probability ---
        wins = goalie.get('wins', goalie.get('win_rate', 0.5) * gp)
        win_rate, _, _ = self.updater.update_rate(
            LEAGUE_PRIORS['win_rate'],
            LEAGUE_PRIORS['goalie_prior_n'],
            wins, gp
        )

        # --- Expected goals against ---
        if opp_implied_total:
            e_ga = opp_implied_total
        else:
            ga_total = goalie.get('goals_against', LEAGUE_PRIORS['ga_rate'] * gp)
            ga_rate_raw, _, _ = self.updater.update_rate(
                LEAGUE_PRIORS['ga_rate'] / 10,
                LEAGUE_PRIORS['goalie_prior_n'],
                ga_total / 10, gp
            )
            e_ga = ga_rate_raw * 10

        # --- Expected saves ---
        save_pct = goalie.get('save_pct', LEAGUE_PRIORS['save_pct'])
        shots_against = goalie.get('shots_against_per_game',
                                   LEAGUE_PRIORS['shots_against_rate'])
        e_saves = shots_against * save_pct

        # --- Shutout probability ---
        shutouts = goalie.get('shutouts', LEAGUE_PRIORS['shutout_rate'] * gp)
        so_rate, _, _ = self.updater.update_rate(
            LEAGUE_PRIORS['shutout_rate'],
            LEAGUE_PRIORS['goalie_prior_n'] * 2,  # More conservative for rare event
            shutouts, gp
        )

        # --- OT loss ---
        p_otl = 0.10  # ~10% of games go to OT and goalie loses

        # --- Expected FPTS ---
        p_35_saves = 1 - stats.poisson.cdf(34, e_saves)  # P(saves >= 35)

        expected_fpts = (
            win_rate * WIN_PTS +
            e_saves * SAVE_PTS +
            e_ga * GA_PTS +
            so_rate * SHUTOUT_BONUS +
            p_otl * (1 - win_rate) * OTL_PTS +
            p_35_saves * SAVE_35_BONUS
        )

        # --- Monte Carlo ---
        sims = self._simulate(win_rate, e_saves, e_ga, so_rate, p_otl, N_SIMS)

        return {
            'expected_fpts': round(expected_fpts, 2),
            'median_fpts': round(np.median(sims), 2),
            'floor': round(np.percentile(sims, 10), 2),
            'ceiling': round(np.percentile(sims, 90), 2),
            'std': round(np.std(sims), 2),
            'p_win': round(win_rate, 3),
            'p_shutout': round(so_rate, 4),
            'p_35_saves': round(p_35_saves, 3),
            'e_saves': round(e_saves, 1),
            'e_ga': round(e_ga, 2),
        }

    def _simulate(self, win_r, e_saves, e_ga, so_r, otl_r, n: int) -> np.ndarray:
        """Monte Carlo simulation of goalie FPTS outcomes using NB for saves/GA."""
        try:
            from stochastic_upgrades import poisson_or_nb
            wins = np.random.binomial(1, min(win_r, 0.99), n)
            saves = poisson_or_nb(e_saves, 'saves', True, n)
            ga = poisson_or_nb(e_ga, 'goals_against', True, n)
        except ImportError:
            wins = np.random.binomial(1, min(win_r, 0.99), n)
            saves = np.random.poisson(e_saves, n)
            ga = np.random.poisson(e_ga, n)
        
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


# ================================================================
#  Full Projector
# ================================================================

class BayesianProjector:
    """
    Project an entire slate using Bayesian event probabilities.

    Can work alongside the current model — produces a parallel set of
    projections that can be blended or compared.
    """

    def __init__(self):
        self.skater_proj = SkaterProjector()
        self.goalie_proj = GoalieProjector()

    def project_player_pool(self, pool: pd.DataFrame,
                            vegas: pd.DataFrame = None,
                            date_str: str = None) -> pd.DataFrame:
        """
        Generate Bayesian projections for an entire player pool.

        Args:
            pool: DataFrame with player stats (from your pipeline)
            vegas: Optional Vegas data for matchup adjustments
            date_str: Slate date for Vegas lookup

        Returns:
            pool with added bayesian_fpts, bayes_floor, bayes_ceiling columns
        """
        df = pool.copy()

        # Build Vegas team total map if available
        team_totals = {}
        if vegas is not None and not vegas.empty and date_str:
            dv = vegas[vegas['date'] == date_str] if 'date' in vegas.columns else vegas
            for _, row in dv.iterrows():
                team_totals[row['Team']] = row['TeamGoal']

        avg_total = np.mean(list(team_totals.values())) if team_totals else 3.0

        results = []
        for _, player in df.iterrows():
            pos = player.get('position', 'C')

            if pos == 'G':
                # Goalie projection
                opp_total = None
                if team_totals:
                    # Find opponent's implied total
                    # (would need opponent info — use average if not available)
                    opp_total = avg_total

                gdata = {
                    'save_pct': player.get('save_pct', 0.905),
                    'games_started': player.get('games_played', 20),
                    'wins': player.get('wins', 10),
                    'shutouts': player.get('shutouts', 1),
                    'goals_against': player.get('goals_against', 56),
                    'shots_against_per_game': player.get('shots_against_per_game', 29),
                }
                result = self.goalie_proj.project(gdata, opp_total)
            else:
                # Skater projection
                matchup = 1.0
                if team_totals and player.get('team') in team_totals:
                    matchup = team_totals[player['team']] / avg_total

                pdata = {
                    'position': pos,
                    'games_played': player.get('games_played', player.get('gp', 25)),
                    'goals_per_game': player.get('goals_per_game', LEAGUE_PRIORS['goals_rate']),
                    'assists_per_game': player.get('assists_per_game', LEAGUE_PRIORS['assists_rate']),
                    'shots_per_game': player.get('shots_per_game', LEAGUE_PRIORS['shots_rate']),
                    'blocks_per_game': player.get('blocks_per_game', LEAGUE_PRIORS['blocks_rate']),
                    'sh_points': player.get('sh_points', 0),
                    'pp_points_per_game': player.get('pp_points_per_game', 0),
                }
                result = self.skater_proj.project(pdata, matchup)

            results.append(result)

        # Attach results to dataframe
        result_df = pd.DataFrame(results)
        for col in result_df.columns:
            df[f'bayes_{col}'] = result_df[col].values

        return df


# ================================================================
#  Comparison Tool
# ================================================================

def compare_models(date_str: str):
    """Compare Bayesian projections to current model and actuals."""
    from pathlib import Path
    import glob

    PROJECT_ROOT = Path(__file__).resolve().parent

    # Find projection CSV with per-game stats
    proj_dir = PROJECT_ROOT / 'daily_projections'
    from datetime import datetime
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    prefixes = [f"{dt.month:02d}_{dt.day:02d}_{dt.strftime('%y')}"]

    proj_file = None
    for f in sorted(proj_dir.glob('*NHLprojections_*.csv')):
        if '_lineups' in f.name:
            continue
        for prefix in prefixes:
            if f.name.startswith(prefix):
                proj_file = f

    if not proj_file:
        print(f"  No projection CSV for {date_str}")
        return

    pool = pd.read_csv(proj_file)
    print(f"\n  Loaded {len(pool)} players from {proj_file.name}")

    # Load Vegas if available
    vegas = None
    vegas_path = PROJECT_ROOT / 'Vegas_Historical.csv'
    if vegas_path.exists():
        vdf = pd.read_csv(vegas_path, encoding='utf-8-sig')
        vdf['date'] = vdf['Date'].apply(
            lambda d: f"20{d.split('.')[2]}-{int(d.split('.')[0]):02d}-{int(d.split('.')[1]):02d}"
        )
        vdf['win_pct'] = vdf['Win %'].str.rstrip('%').astype(float) / 100
        vegas = vdf

    # Run Bayesian projector
    bp = BayesianProjector()
    result = bp.project_player_pool(pool, vegas, date_str)

    # Show comparison
    skaters = result[result['position'] != 'G'].sort_values('projected_fpts', ascending=False)

    print(f"\n{'=' * 90}")
    print(f"  BAYESIAN vs CURRENT MODEL — {date_str}")
    print(f"{'=' * 90}")

    print(f"\n  {'Name':<22} {'Pos':>3} {'Current':>8} {'Bayes':>8} {'Diff':>6} "
          f"{'P(Goal)':>8} {'P(Ast)':>7} {'Floor':>6} {'Ceil':>6}")
    print(f"  {'─' * 82}")

    for _, p in skaters.head(25).iterrows():
        current = p.get('projected_fpts', 0)
        bayes = p.get('bayes_expected_fpts', 0)
        diff = bayes - current
        p_goal = p.get('bayes_p_goal', 0)
        p_assist = p.get('bayes_p_assist', 0)
        floor_b = p.get('bayes_floor', 0)
        ceil_b = p.get('bayes_ceiling', 0)

        print(f"  {p['name']:<22} {p['position']:>3} {current:>8.1f} {bayes:>8.1f} {diff:>+6.1f} "
              f"{p_goal:>7.1%} {p_assist:>6.1%} {floor_b:>6.1f} {ceil_b:>6.1f}")

    # Aggregate comparison
    has_both = result[result['bayes_expected_fpts'].notna()]
    if not has_both.empty:
        print(f"\n  AGGREGATE:")
        print(f"    Current model mean: {has_both['projected_fpts'].mean():.2f}")
        print(f"    Bayesian mean:      {has_both['bayes_expected_fpts'].mean():.2f}")
        print(f"    Correlation:        {has_both[['projected_fpts', 'bayes_expected_fpts']].corr().iloc[0,1]:.3f}")

    # Load actuals if available
    actuals_path = PROJECT_ROOT / 'backtests' / 'batch_backtest_details.csv'
    if actuals_path.exists():
        actuals = pd.read_csv(actuals_path)
        act_date = actuals[actuals['date'] == date_str]
        if not act_date.empty:
            # Match on last name + team
            def ln(n):
                return n.strip().split()[-1].lower()
            act_date = act_date.copy()
            act_date['_key'] = act_date['name'].apply(ln) + '_' + act_date['team'].str.lower()
            result['_key'] = result['name'].apply(ln) + '_' + result['team'].str.lower()

            merged = act_date.merge(
                result[['_key', 'bayes_expected_fpts', 'projected_fpts']].drop_duplicates('_key'),
                on='_key', how='inner', suffixes=('_actual', '_proj')
            )

            if not merged.empty:
                merged['bayes_error'] = (merged['bayes_expected_fpts'] - merged['actual_fpts']).abs()
                merged['current_error'] = (merged['projected_fpts_proj'] - merged['actual_fpts']).abs()

                bayes_mae = merged['bayes_error'].mean()
                current_mae = merged['current_error'].mean()

                print(f"\n  vs ACTUALS ({len(merged)} matched):")
                print(f"    Current model MAE: {current_mae:.3f}")
                print(f"    Bayesian MAE:      {bayes_mae:.3f}")
                print(f"    Difference:        {current_mae - bayes_mae:+.3f} "
                      f"({'Bayes BETTER' if bayes_mae < current_mae else 'Current BETTER'})")


# ================================================================
#  CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian NHL DFS Projections')
    parser.add_argument('--date', type=str, default='2026-01-23',
                       help='Slate date (YYYY-MM-DD)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare to current model and actuals')
    args = parser.parse_args()

    if args.compare:
        compare_models(args.date)
    else:
        compare_models(args.date)
