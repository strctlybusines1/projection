#!/usr/bin/env python3
"""
Correlated Monte Carlo Simulation Engine
==========================================

Renaissance-style lineup evaluation using correlated player simulations.

Instead of point estimates, we model each player as a zero-inflated lognormal
distribution, simulate 10,000 correlated outcomes per lineup, and select
lineups that maximize P(exceed target).

Correlation structure (from 29,339 skater-games, 113 slates):
  - Same team, same line:   r = 0.124
  - Same team, diff line:   r = 0.034
  - Goalie ↔ own team:      r = 0.191
  - Skater ↔ opponent team: r = -0.016 (negligible)
  - Goalie ↔ opponent team: r = -0.340

Usage:
    from simulation_engine import SimulationEngine
    engine = SimulationEngine()
    engine.fit_player_distributions(pool, historical_scores)
    result = engine.simulate_lineup(lineup, n_sims=10000)
    best = engine.select_best_lineup(candidates, target=111)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"

# Default correlation structure
DEFAULT_CORRELATIONS = {
    'same_team_same_line': 0.124,
    'same_team_diff_line': 0.034,
    'goalie_own_team': 0.191,
    'opponent_skaters': -0.016,
    'goalie_opp_team': -0.340,
}


class PlayerDistribution:
    """Zero-inflated lognormal distribution for a single player."""

    __slots__ = ['name', 'team', 'position', 'salary', 'line', 'pp_unit',
                 'p_floor', 'mu', 'sigma', 'env_mult', 'n_games', 'mean', 'std']

    def __init__(self, name: str, team: str, position: str, salary: float,
                 scores: np.ndarray, line: float = None, pp_unit: float = None,
                 env_mult: float = 1.0):
        self.name = name
        self.team = team
        self.position = position
        self.salary = salary
        self.line = line
        self.pp_unit = pp_unit
        self.env_mult = env_mult
        self.n_games = len(scores)

        if len(scores) < 3:
            # Insufficient data — use wide prior
            self.p_floor = 0.30
            self.mu = np.log(4.0)
            self.sigma = 0.80
            self.mean = 4.0
            self.std = 4.0
            return

        # Floor probability
        self.p_floor = float((scores <= 1.5).mean())

        # Fit lognormal to positive scores
        pos_scores = scores[scores > 1.5]
        if len(pos_scores) < 2:
            self.mu = np.log(max(scores.mean(), 1.0))
            self.sigma = 0.80
        else:
            log_scores = np.log(pos_scores)
            self.mu = float(log_scores.mean())
            self.sigma = float(max(log_scores.std(), 0.20))

        self.mean = float(scores.mean())
        self.std = float(scores.std()) if len(scores) > 1 else self.mean * 0.8

    def sample_independent(self, n: int = 10000) -> np.ndarray:
        """Generate independent samples (no correlation)."""
        is_floor = np.random.random(n) < self.p_floor
        log_samples = np.random.normal(
            self.mu + np.log(max(self.env_mult, 0.5)),
            self.sigma, n)
        pos_samples = np.exp(log_samples)
        floor_samples = np.random.uniform(0, 1.5, n)
        return np.where(is_floor, floor_samples, pos_samples)

    def sample_from_z(self, z: np.ndarray) -> np.ndarray:
        """
        Convert correlated standard normal z-scores to FPTS samples.
        
        This is the key to correlated simulation:
        1. Generate correlated z-scores via Cholesky decomposition
        2. Convert each player's z to their personal distribution
        """
        n = len(z)
        # Use z to determine floor vs non-floor via uniform transform
        u = _norm_cdf(z)  # Convert to uniform [0,1]

        # Floor if u < p_floor
        is_floor = u < self.p_floor

        # For non-floor: map remaining u to lognormal
        # Rescale u from [p_floor, 1] to [0, 1] for the lognormal portion
        u_rescaled = np.where(is_floor, 0.5,
                              (u - self.p_floor) / max(1.0 - self.p_floor, 0.001))
        u_rescaled = np.clip(u_rescaled, 0.001, 0.999)

        # Inverse lognormal CDF
        z_ln = _norm_ppf(u_rescaled)
        log_samples = self.mu + np.log(max(self.env_mult, 0.5)) + self.sigma * z_ln
        pos_samples = np.exp(log_samples)

        # Floor samples
        floor_samples = u * 1.5 / max(self.p_floor, 0.001)
        floor_samples = np.clip(floor_samples, 0, 1.5)

        return np.where(is_floor, floor_samples, pos_samples)


# Fast vectorized norm CDF/PPF using scipy (hot path — must be fast)
try:
    from scipy.special import erf as _scipy_erf, erfinv as _scipy_erfinv
    from scipy.stats import norm as _scipy_norm

    def _norm_cdf(x):
        """Standard normal CDF — vectorized via scipy."""
        return 0.5 * (1.0 + _scipy_erf(x / np.sqrt(2.0)))

    def _norm_ppf(u):
        """Inverse normal CDF — vectorized via scipy."""
        u = np.clip(u, 1e-8, 1 - 1e-8)
        return _scipy_norm.ppf(u)

except ImportError:
    # Fallback: pure numpy (still vectorized, no scalar loop)
    def _norm_cdf(x):
        """Standard normal CDF using numpy's built-in erf."""
        # numpy doesn't have erf, but we can use the tanh approximation
        # Accurate to ~1e-4, which is plenty for simulation
        a = 0.3480242
        b = 0.0958798
        c = 0.7478556
        t = 1.0 / (1.0 + 0.47047 * np.abs(x))
        val = 1.0 - t * (a + t * (-b + t * c)) * np.exp(-0.5 * x * x)
        return np.where(x >= 0, val, 1.0 - val)

    def _norm_ppf(u):
        """Approximate inverse normal CDF (Acklam's rational approximation, vectorized)."""
        u = np.asarray(u, dtype=np.float64)
        u = np.clip(u, 1e-8, 1 - 1e-8)

        a = np.array([-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
                       1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0])
        b = np.array([-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
                       6.680131188771972e1, -1.328068155288572e1])
        c = np.array([-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
                       -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0])
        d = np.array([7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
                       3.754408661907416e0])

        p_low = 0.02425
        p_high = 1.0 - p_low
        result = np.zeros_like(u)

        # Lower region
        mask_low = u < p_low
        if mask_low.any():
            q = np.sqrt(-2.0 * np.log(u[mask_low]))
            result[mask_low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

        # Central region
        mask_mid = (~mask_low) & (u <= p_high)
        if mask_mid.any():
            q = u[mask_mid] - 0.5
            r = q * q
            result[mask_mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)

        # Upper region
        mask_high = u > p_high
        if mask_high.any():
            q = np.sqrt(-2.0 * np.log(1.0 - u[mask_high]))
            result[mask_high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

        return result


class SimulationEngine:
    """
    Correlated Monte Carlo simulation for DFS lineup evaluation.
    
    Flow:
      1. fit_player_distributions() — build per-player distributions from history
      2. simulate_lineup() — run N correlated simulations of a lineup
      3. select_best_lineup() — pick lineup with highest P(exceed target)
    """

    def __init__(self, n_sims: int = 10000):
        self.n_sims = n_sims
        self.player_dists: Dict[str, PlayerDistribution] = {}
        self.correlations = DEFAULT_CORRELATIONS
        self._load_correlations()
        self._cholesky_cache: Dict[str, np.ndarray] = {}  # Cache Cholesky decompositions

    def _load_correlations(self):
        try:
            with open(DATA_DIR / 'correlation_structure.json') as f:
                self.correlations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def fit_player_distributions(self, pool: pd.DataFrame,
                                  historical: pd.DataFrame = None,
                                  date_str: str = None):
        """
        Fit distributions for all players in the pool.
        
        pool: today's slate DataFrame
        historical: DataFrame with columns [Player, Team, Score, slate_date]
        date_str: today's date (to prevent leakage)
        """
        self.player_dists = {}

        for _, row in pool.iterrows():
            name = row.get('name', row.get('Player', ''))
            team = row.get('team', row.get('Team', ''))
            pos = row.get('position', row.get('Pos', 'C'))
            salary = row.get('salary', row.get('Salary', 3000))
            line = row.get('line', row.get('Start/Line', None))
            pp_unit = row.get('pp_unit', row.get('PP Unit', None))

            try:
                line = float(line) if pd.notna(line) else None
            except (ValueError, TypeError):
                line = None
            try:
                pp_unit = float(pp_unit) if pd.notna(pp_unit) else None
            except (ValueError, TypeError):
                pp_unit = None

            # Get historical scores
            if historical is not None:
                mask = (historical['Player'] == name) & (historical['Team'] == team)
                if date_str:
                    mask = mask & (historical['slate_date'] < date_str)
                scores = historical.loc[mask, 'Score'].dropna().values
            else:
                scores = np.array([])

            # Game environment multiplier
            team_goal = row.get('TeamGoal', row.get('team_goal', 3.0))
            try:
                team_goal = float(team_goal) if pd.notna(team_goal) else 3.0
            except (ValueError, TypeError):
                team_goal = 3.0
            env_mult = team_goal / 3.0

            key = f"{name}_{team}"
            self.player_dists[key] = PlayerDistribution(
                name=name, team=team, position=pos, salary=salary,
                scores=scores, line=line, pp_unit=pp_unit,
                env_mult=env_mult)

    def _build_correlation_matrix(self, players: List[PlayerDistribution]) -> np.ndarray:
        """
        Build NxN correlation matrix for a lineup of N players.
        Uses measured correlations based on team/line/position relationships.
        """
        n = len(players)
        corr = np.eye(n)

        c = self.correlations
        r_same_line = c.get('same_team_same_line', 0.124)
        r_diff_line = c.get('same_team_diff_line', 0.034)
        r_g_own = c.get('goalie_own_team', 0.191)
        r_opp = c.get('opponent_skaters', -0.016)
        r_g_opp = c.get('goalie_opp_team', -0.340)

        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = players[i], players[j]

                if pi.position == 'G' or pj.position == 'G':
                    # Goalie correlation
                    goalie = pi if pi.position == 'G' else pj
                    skater = pj if pi.position == 'G' else pi
                    if goalie.team == skater.team:
                        rho = r_g_own
                    else:
                        rho = r_g_opp
                elif pi.team == pj.team:
                    # Same-team skaters
                    if (pi.line is not None and pj.line is not None
                            and pi.line == pj.line):
                        rho = r_same_line
                    else:
                        rho = r_diff_line
                else:
                    # Different teams (opponent effect)
                    rho = r_opp

                corr[i, j] = rho
                corr[j, i] = rho

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < -1e-8:
            # Nearest PSD matrix via eigenvalue clipping
            eigvals_full, eigvecs = np.linalg.eigh(corr)
            eigvals_full = np.maximum(eigvals_full, 1e-8)
            corr = eigvecs @ np.diag(eigvals_full) @ eigvecs.T
            # Renormalize diagonal
            d = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d, d)

        return corr

    def simulate_lineup(self, lineup: pd.DataFrame,
                        n_sims: int = None) -> Dict:
        """
        Simulate a lineup N times with correlated player outcomes.
        
        Returns dict with:
          - simulated_totals: array of N lineup total FPTS
          - mean, std, p_cash, p_gpp, percentiles
          - player_sims: dict of player -> simulated array
        """
        if n_sims is None:
            n_sims = self.n_sims

        # Get distributions for lineup players
        players = []
        player_keys = []
        for _, row in lineup.iterrows():
            name = row.get('name', row.get('Player', ''))
            team = row.get('team', row.get('Team', ''))
            key = f"{name}_{team}"

            if key in self.player_dists:
                players.append(self.player_dists[key])
                player_keys.append(key)
            else:
                # Fallback: create distribution from projected_fpts
                proj = row.get('projected_fpts', row.get('Avg', 5.0))
                try:
                    proj = float(proj) if pd.notna(proj) else 5.0
                except (ValueError, TypeError):
                    proj = 5.0
                pos = row.get('position', row.get('Pos', 'C'))
                salary = row.get('salary', row.get('Salary', 3000))
                scores = np.array([proj] * 5)  # Minimal prior
                dist = PlayerDistribution(
                    name=name, team=team, position=pos, salary=salary,
                    scores=scores)
                players.append(dist)
                player_keys.append(key)

        n_players = len(players)

        # Build and decompose correlation matrix (with caching)
        cache_key = '_'.join(f"{p.team}:{p.position}:{p.line}" for p in players)
        if cache_key in self._cholesky_cache:
            L = self._cholesky_cache[cache_key]
        else:
            corr_matrix = self._build_correlation_matrix(players)
            try:
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                corr_matrix += np.eye(n_players) * 0.01
                L = np.linalg.cholesky(corr_matrix)
            self._cholesky_cache[cache_key] = L

        # Generate correlated standard normals
        z_independent = np.random.standard_normal((n_players, n_sims))
        z_correlated = L @ z_independent  # Shape: (n_players, n_sims)

        # Convert each player's z-scores to their FPTS distribution
        player_sims = {}
        lineup_sims = np.zeros(n_sims)

        for i, (player, key) in enumerate(zip(players, player_keys)):
            fpts_samples = player.sample_from_z(z_correlated[i])
            player_sims[key] = fpts_samples
            lineup_sims += fpts_samples

        # Compute statistics
        result = {
            'simulated_totals': lineup_sims,
            'mean': float(np.mean(lineup_sims)),
            'std': float(np.std(lineup_sims)),
            'median': float(np.median(lineup_sims)),
            'p5': float(np.percentile(lineup_sims, 5)),
            'p25': float(np.percentile(lineup_sims, 25)),
            'p75': float(np.percentile(lineup_sims, 75)),
            'p95': float(np.percentile(lineup_sims, 95)),
            'p_cash': float((lineup_sims >= 111).mean()),
            'p_gpp': float((lineup_sims >= 140).mean()),
            'p_120': float((lineup_sims >= 120).mean()),
            'max': float(np.max(lineup_sims)),
            'player_sims': player_sims,
            'n_sims': n_sims,
        }

        return result

    def select_best_lineup(self, candidates: List[pd.DataFrame],
                           target: float = 111.0,
                           mode: str = 'cash',
                           n_sims: int = None,
                           verbose: bool = True) -> Tuple[int, pd.DataFrame, Dict]:
        """
        Evaluate all candidate lineups via simulation and pick the best.
        
        mode='cash': maximize P(total >= target)
        mode='gpp': maximize P(total >= gpp_threshold)
        mode='ceiling': maximize P95
        
        Returns: (best_index, best_lineup, all_results)
        """
        if n_sims is None:
            n_sims = self.n_sims

        results = []
        for i, lineup in enumerate(candidates):
            sim = self.simulate_lineup(lineup, n_sims=n_sims)
            results.append({
                'idx': i,
                'mean': sim['mean'],
                'std': sim['std'],
                'p_cash': sim['p_cash'],
                'p_gpp': sim['p_gpp'],
                'p_120': sim['p_120'],
                'p95': sim['p95'],
                'median': sim['median'],
                'max': sim['max'],
            })

        rdf = pd.DataFrame(results)

        # Select based on mode
        if mode == 'cash':
            best_idx = int(rdf.loc[rdf['p_cash'].idxmax(), 'idx'])
            sort_col = 'p_cash'
        elif mode == 'gpp':
            best_idx = int(rdf.loc[rdf['p_gpp'].idxmax(), 'idx'])
            sort_col = 'p_gpp'
        elif mode == 'ceiling':
            best_idx = int(rdf.loc[rdf['p95'].idxmax(), 'idx'])
            sort_col = 'p95'
        else:
            best_idx = int(rdf.loc[rdf['p_cash'].idxmax(), 'idx'])
            sort_col = 'p_cash'

        if verbose:
            print(f"\n  Simulation Results ({len(candidates)} lineups × {n_sims:,} sims):")
            print(f"  {'Rank':>4} {'Mean':>6} {'Std':>5} {'P(111+)':>8} {'P(120+)':>8} "
                  f"{'P(140+)':>8} {'P95':>6} {'Max':>6}")
            top = rdf.nlargest(5, sort_col)
            for rank, (_, r) in enumerate(top.iterrows(), 1):
                marker = ' ◄' if int(r['idx']) == best_idx else ''
                print(f"  {rank:>4} {r['mean']:>6.1f} {r['std']:>5.1f} "
                      f"{r['p_cash']:>7.1%} {r['p_120']:>7.1%} "
                      f"{r['p_gpp']:>7.1%} {r['p95']:>6.1f} {r['max']:>6.0f}{marker}")

        return best_idx, candidates[best_idx], rdf


def quick_simulate(lineup: pd.DataFrame, pool: pd.DataFrame,
                   historical: pd.DataFrame, date_str: str = None,
                   n_sims: int = 10000, target: float = 111.0) -> Dict:
    """Convenience function: fit + simulate a single lineup."""
    engine = SimulationEngine(n_sims=n_sims)
    engine.fit_player_distributions(pool, historical, date_str)
    return engine.simulate_lineup(lineup)
