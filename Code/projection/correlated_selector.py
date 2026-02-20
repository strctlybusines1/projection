#!/usr/bin/env python3
"""
Correlated Simulation Lineup Selector
=======================================

Uses Iman-Conover-style correlated Monte Carlo to re-rank candidate lineups.

Instead of picking one lineup per strategy based on point estimates, this module:
  1. Generates EXPANDED candidates (top-3 per strategy = ~48 lineups)
  2. Fits per-player zero-inflated lognormal distributions from historical game logs
  3. Builds a correlation matrix for each candidate lineup using empirical coefficients
  4. Runs N correlated simulations via Cholesky decomposition
  5. Selects the single best lineup by P(exceed target) or P(exceed ceiling)

Empirical correlations (from 23K skater-games, 113 slates 2025-26):
  - PP1 unit members:          r = 0.184
  - Same team, same line:      r = 0.180
  - Same team, diff line:      r = 0.022
  - Goalie <-> own team:       r = 0.045
  - Goalie <-> opponent team:  r = -0.181
  - Opponent skaters (same game): r = 0.023

Key insight: The correlation between linemates matters because a goal event
generates ~8-12 DK FPTS distributed among 1-3 players who all share the same
line. When we simulate correlated outcomes, stacked lineups show higher
P(ceiling) than their independent-simulation equivalents would suggest.

Usage:
    from correlated_selector import CorrelatedSelector
    selector = CorrelatedSelector(conn, n_sims=5000)
    selector.fit_distributions(dk_pool, date_str)
    best_idx, best_lineup, sim_results = selector.select_best(
        candidate_lineups, mode='gpp', target=140)
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "data" / "nhl_dfs_history.db"

# Empirical correlations from 2025-26 season game logs
EMPIRICAL_CORRELATIONS = {
    'pp1_linemates': 0.184,       # PP1 unit members
    'same_team_same_line': 0.180,  # Same team, same line (L1 or L2)
    'same_team_diff_line': 0.022,  # Same team, different lines
    'goalie_own_team': 0.045,      # Goalie vs own-team skaters
    'goalie_opp_team': -0.181,     # Goalie vs opponent skaters
    'opponent_skaters': 0.023,     # Opponent skaters in same game
    'diff_game': 0.0,             # Players from different games
}

TEAM_NORM = {'nj': 'njd', 'la': 'lak', 'sj': 'sjs', 'tb': 'tbl'}

def norm(t):
    return TEAM_NORM.get(str(t).lower().strip(), str(t).lower().strip())


class PlayerDist:
    """Lightweight zero-inflated lognormal distribution for simulation."""
    __slots__ = ['name', 'team', 'position', 'line', 'pp_unit', 'salary',
                 'p_floor', 'mu', 'sigma', 'env_mult', 'proj']

    def __init__(self, name, team, position, salary, scores,
                 line=None, pp_unit=None, env_mult=1.0, proj=5.0):
        self.name = name
        self.team = norm(team)
        self.position = position
        self.line = line
        self.pp_unit = pp_unit
        self.salary = salary
        self.env_mult = env_mult
        self.proj = proj

        if len(scores) < 3:
            self.p_floor = 0.30
            self.mu = np.log(max(proj, 2.0))
            self.sigma = 0.80
            return

        self.p_floor = float((scores <= 1.5).mean())
        pos_scores = scores[scores > 1.5]
        if len(pos_scores) < 2:
            self.mu = np.log(max(scores.mean(), 1.0))
            self.sigma = 0.80
        else:
            log_scores = np.log(pos_scores)
            self.mu = float(log_scores.mean())
            self.sigma = float(max(log_scores.std(), 0.20))


class CorrelatedSelector:
    """
    Selects best lineup from candidates using correlated Monte Carlo simulation.

    Two modes:
      - 'cash': maximize P(total >= cash_line)
      - 'gpp':  maximize P(total >= ceiling_target) for WTA/GPP contests
    """

    def __init__(self, conn=None, n_sims=5000):
        self.conn = conn
        self.n_sims = n_sims
        self.player_dists: Dict[str, PlayerDist] = {}
        self.corr = EMPIRICAL_CORRELATIONS
        self._historical_scores = None  # Cache

    def _load_historical_scores(self, date_str):
        """Load historical DK FPTS for all players before this date."""
        if self._historical_scores is not None:
            return self._historical_scores

        if self.conn is None:
            return {}

        df = pd.read_sql("""
            SELECT player_name, team, game_date, dk_fpts
            FROM game_logs_skaters
            WHERE game_date < ?
            ORDER BY game_date
        """, self.conn, params=(date_str,))

        # Also load goalie scores
        gdf = pd.read_sql("""
            SELECT player_name, team, game_date, dk_fpts
            FROM game_logs_goalies
            WHERE game_date < ?
            ORDER BY game_date
        """, self.conn, params=(date_str,))

        # Index by player_team key
        scores = {}
        for _, row in df.iterrows():
            key = f"{row['player_name'].lower().strip()}_{norm(row['team'])}"
            if key not in scores:
                scores[key] = []
            scores[key].append(row['dk_fpts'])

        for _, row in gdf.iterrows():
            key = f"{row['player_name'].lower().strip()}_{norm(row['team'])}"
            if key not in scores:
                scores[key] = []
            scores[key].append(row['dk_fpts'])

        self._historical_scores = scores
        return scores

    def fit_distributions(self, dk_pool, date_str):
        """Fit zero-inflated lognormal distributions for all players on the slate."""
        hist = self._load_historical_scores(date_str)
        self.player_dists = {}

        for _, row in dk_pool.iterrows():
            name = row.get('player_name', '')
            team = row.get('team', '')
            key = f"{name.lower().strip()}_{norm(team)}"
            pos = row.get('position', 'C')
            salary = row.get('salary', 3000)
            line = row.get('start_line', None)
            pp_unit = row.get('pp_unit', None)

            try:
                line = str(line).strip() if pd.notna(line) else None
            except:
                line = None
            try:
                pp_unit = str(pp_unit).strip() if pd.notna(pp_unit) else None
            except:
                pp_unit = None

            # Historical scores
            player_scores = np.array(hist.get(key, []))

            # Environment multiplier from implied total
            impl = row.get('team_implied_total', 3.0)
            try:
                impl = float(impl) if pd.notna(impl) else 3.0
            except:
                impl = 3.0
            env_mult = impl / 3.0

            # Projection from DK or our model
            proj = row.get('my_proj', row.get('dk_avg_fpts', 5.0))
            try:
                proj = float(proj) if pd.notna(proj) else 5.0
            except:
                proj = 5.0

            self.player_dists[key] = PlayerDist(
                name=name, team=team, position=pos, salary=salary,
                scores=player_scores, line=line, pp_unit=pp_unit,
                env_mult=env_mult, proj=proj)

    def _get_correlation(self, p1: PlayerDist, p2: PlayerDist) -> float:
        """Determine pairwise correlation between two players."""
        c = self.corr

        # Goalie involved
        if p1.position == 'G' or p2.position == 'G':
            goalie = p1 if p1.position == 'G' else p2
            other = p2 if p1.position == 'G' else p1
            if goalie.team == other.team:
                return c['goalie_own_team']
            else:
                return c['goalie_opp_team']

        # Both skaters
        if p1.team == p2.team:
            # Check if same line
            if (p1.line is not None and p2.line is not None
                    and p1.line == p2.line):
                # Check if both PP1 — slightly higher correlation
                if (p1.pp_unit in ['1', '1.0'] and p2.pp_unit in ['1', '1.0']):
                    return c['pp1_linemates']
                return c['same_team_same_line']
            return c['same_team_diff_line']
        else:
            return c['opponent_skaters']

    def _build_corr_matrix(self, players: List[PlayerDist]) -> np.ndarray:
        """Build NxN correlation matrix for a lineup."""
        n = len(players)
        corr = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                rho = self._get_correlation(players[i], players[j])
                corr[i, j] = rho
                corr[j, i] = rho

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < -1e-8:
            eigvals_full, eigvecs = np.linalg.eigh(corr)
            eigvals_full = np.maximum(eigvals_full, 1e-8)
            corr = eigvecs @ np.diag(eigvals_full) @ eigvecs.T
            d = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d, d)

        return corr

    def _simulate_lineup(self, lineup_players: List[PlayerDist],
                         n_sims: int = None) -> np.ndarray:
        """
        Run correlated simulation for a lineup. Returns array of simulated totals.

        Uses Cholesky decomposition to generate correlated standard normals,
        then maps through each player's zero-inflated lognormal distribution.
        """
        if n_sims is None:
            n_sims = self.n_sims

        n = len(lineup_players)

        # Build and decompose correlation matrix
        corr = self._build_corr_matrix(lineup_players)
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            corr += np.eye(n) * 0.01
            L = np.linalg.cholesky(corr)

        # Generate correlated standard normals
        z_ind = np.random.standard_normal((n, n_sims))
        z_corr = L @ z_ind  # (n, n_sims)

        # Convert to FPTS via inverse CDF transform
        totals = np.zeros(n_sims)
        for i, player in enumerate(lineup_players):
            # Standard normal CDF -> uniform
            u = _norm_cdf(z_corr[i])

            # Zero-inflated lognormal inverse CDF
            is_floor = u < player.p_floor

            # Rescale non-floor portion
            u_rescaled = np.where(
                is_floor, 0.5,
                (u - player.p_floor) / max(1.0 - player.p_floor, 0.001))
            u_rescaled = np.clip(u_rescaled, 0.001, 0.999)

            # Inverse lognormal
            z_ln = _norm_ppf(u_rescaled)
            log_samples = player.mu + np.log(max(player.env_mult, 0.5)) + player.sigma * z_ln
            pos_samples = np.exp(log_samples)

            # Floor samples
            floor_samples = u * 1.5 / max(player.p_floor, 0.001)
            floor_samples = np.clip(floor_samples, 0, 1.5)

            fpts = np.where(is_floor, floor_samples, pos_samples)
            totals += fpts

        return totals

    def _lineup_to_dists(self, lineup: list) -> List[PlayerDist]:
        """Convert lineup (list of dicts) to list of PlayerDist objects."""
        players = []
        for p in lineup:
            key = f"{p['name'].lower().strip()}_{norm(p['team'])}"
            if key in self.player_dists:
                players.append(self.player_dists[key])
            else:
                # Fallback: create minimal distribution
                players.append(PlayerDist(
                    name=p['name'], team=p['team'],
                    position=p.get('position', 'C'),
                    salary=p.get('salary', 3000),
                    scores=np.array([5.0] * 3),
                    proj=5.0))
        return players

    def select_best(self, candidate_lineups: Dict[str, dict],
                    mode='gpp', target=None,
                    cash_line=111.0, gpp_target=140.0,
                    verbose=False) -> Tuple[str, list, pd.DataFrame]:
        """
        Re-rank all candidate lineups using correlated simulation.

        candidate_lineups: dict of strategy_name -> {'lineup': [...], 'stack': {...}}
        mode: 'cash' (maximize P >= cash_line), 'gpp' (maximize P >= gpp_target),
              'ceiling' (maximize P95), 'ev' (maximize mean * P(cash))

        Returns: (best_strategy, best_lineup, results_df)
        """
        if target is not None:
            if mode == 'cash':
                cash_line = target
            else:
                gpp_target = target

        results = []

        for strat_name, data in candidate_lineups.items():
            lineup = data['lineup']
            if lineup is None:
                continue

            players = self._lineup_to_dists(lineup)
            if len(players) < 9:
                continue

            # Run correlated simulation
            totals = self._simulate_lineup(players)

            # Compute metrics
            mean_fpts = float(np.mean(totals))
            std_fpts = float(np.std(totals))
            p_cash = float((totals >= cash_line).mean())
            p_gpp = float((totals >= gpp_target).mean())
            p95 = float(np.percentile(totals, 95))
            p99 = float(np.percentile(totals, 99))
            p50 = float(np.median(totals))
            sim_max = float(np.max(totals))

            # Compute stack correlation bonus — how much does correlation help this lineup?
            # Compare correlated mean to sum of individual means
            indep_mean = sum(p.proj for p in players)
            corr_lift = mean_fpts - indep_mean  # Usually small but can reveal stacking benefit

            results.append({
                'strategy': strat_name,
                'mean': mean_fpts,
                'std': std_fpts,
                'p_cash': p_cash,
                'p_gpp': p_gpp,
                'p95': p95,
                'p99': p99,
                'median': p50,
                'sim_max': sim_max,
                'corr_lift': corr_lift,
                'salary': sum(p.salary for p in players),
                'n_teams': len(set(p.team for p in players)),
            })

        if not results:
            return None, None, pd.DataFrame()

        rdf = pd.DataFrame(results)

        # Select best based on mode
        if mode == 'cash':
            best_idx = rdf['p_cash'].idxmax()
        elif mode == 'gpp':
            best_idx = rdf['p_gpp'].idxmax()
        elif mode == 'ceiling':
            best_idx = rdf['p95'].idxmax()
        elif mode == 'ev':
            # Expected value weighting: mean * sqrt(P(cash)) — balances floor + ceiling
            rdf['ev_score'] = rdf['mean'] * np.sqrt(rdf['p_cash'].clip(0.001))
            best_idx = rdf['ev_score'].idxmax()
        else:
            best_idx = rdf['p_cash'].idxmax()

        best_strat = rdf.loc[best_idx, 'strategy']

        if verbose:
            print(f"\n  Correlated Sim Results ({len(results)} lineups × {self.n_sims:,} sims, mode={mode}):")
            print(f"  {'Strategy':20s} {'Mean':>6} {'Std':>5} {'P(cash)':>8} {'P(GPP)':>8} "
                  f"{'P95':>6} {'P99':>6} {'Max':>6}")
            for _, r in rdf.nlargest(5, 'p_cash' if mode == 'cash' else 'p_gpp').iterrows():
                marker = ' ◄' if r['strategy'] == best_strat else ''
                print(f"  {r['strategy']:20s} {r['mean']:>6.1f} {r['std']:>5.1f} "
                      f"{r['p_cash']:>7.1%} {r['p_gpp']:>7.1%} "
                      f"{r['p95']:>6.1f} {r['p99']:>6.1f} {r['sim_max']:>6.0f}{marker}")

        best_lineup = candidate_lineups[best_strat]['lineup']
        return best_strat, best_lineup, rdf

    def reset_cache(self):
        """Reset historical scores cache for new date."""
        self._historical_scores = None


def generate_expanded_candidates(stacks, dk_pool, actuals, has_ml=False):
    """
    Generate EXPANDED candidates: top-3 per strategy instead of top-1.
    Returns dict of strategy_variant -> {'lineup': [...], 'stack': {...}}

    This gives the correlated selector ~48 candidates to evaluate
    instead of the usual 16.
    """
    from line_multi_stack import (
        HEURISTIC_STRATEGIES, ML_STRATEGIES, ALL_STRATEGIES,
        fill_lineup, norm as lms_norm, _score_d_ceiling, _score_g_ceiling
    )

    strategies = ALL_STRATEGIES if has_ml else HEURISTIC_STRATEGIES
    results = {}

    for strategy in strategies:
        if strategy.startswith('ml_') and not has_ml:
            continue
        if strategy.startswith('ml_') and all(s['ml_proj'] is None for s in stacks):
            continue

        single_stacks = [s for s in stacks if s['stack_type'] == 'single']
        dual_stacks = [s for s in stacks if s['stack_type'] == 'dual']

        # Same sorting logic as select_lineups
        if strategy == 'chalk':
            candidates = sorted(single_stacks,
                                key=lambda s: (s['combo_proj'], s.get('pp1_score', 0)),
                                reverse=True)
        elif strategy == 'contrarian_1':
            candidates = [s for s in single_stacks if 3 <= s['impl_rank'] <= 5]
            candidates.sort(key=lambda s: s['combo_proj'], reverse=True)
        elif strategy == 'contrarian_2':
            candidates = [s for s in single_stacks if 5 <= s['impl_rank'] <= 8]
            candidates.sort(key=lambda s: s['combo_proj'], reverse=True)
        elif strategy == 'value':
            candidates = [s for s in single_stacks if s['combo_proj'] > 35]
            candidates.sort(key=lambda s: s['fwd_salary'])
        elif strategy == 'ceiling':
            def ceiling_score(s):
                max_sal = max(f['salary'] for f in s['forwards']) if s['forwards'] else 0
                pp1_bonus = s.get('pp1_score', 0) * 500
                return max_sal + s['combo_proj'] * 0.1 + pp1_bonus
            candidates = sorted(single_stacks, key=ceiling_score, reverse=True)
        elif strategy == 'game_stack':
            candidates = sorted(single_stacks, key=lambda s: (s['game_total'], s['combo_proj']), reverse=True)
        elif strategy == 'pp1_stack':
            candidates = sorted(single_stacks,
                                key=lambda s: (s.get('pp1_score', 0), s['combo_proj']),
                                reverse=True)
        elif strategy == 'dual_chalk':
            candidates = sorted(dual_stacks, key=lambda s: s['combo_proj'], reverse=True)
        elif strategy == 'dual_ceiling':
            def dual_ceil_score(s):
                max_sal = max(f['salary'] for f in s['forwards']) if s['forwards'] else 0
                pp1_bonus = s.get('pp1_score', 0) * 500
                return max_sal + s['combo_proj'] * 0.1 + pp1_bonus
            candidates = sorted(dual_stacks, key=dual_ceil_score, reverse=True)
        elif strategy == 'dual_game':
            candidates = sorted(dual_stacks, key=lambda s: (s['game_total'], s['combo_proj']), reverse=True)
        elif strategy == 'ml_chalk':
            candidates = sorted([s for s in single_stacks if s['ml_proj'] is not None],
                                key=lambda s: s['ml_proj'], reverse=True)
        elif strategy == 'ml_ceiling':
            candidates = sorted([s for s in single_stacks if s['ml_ceiling'] is not None],
                                key=lambda s: s['ml_ceiling'], reverse=True)
        elif strategy == 'ml_contrarian':
            candidates = [s for s in single_stacks if 3 <= s['impl_rank'] <= 6 and s['ml_proj'] is not None]
            candidates.sort(key=lambda s: s['ml_proj'], reverse=True)
        elif strategy == 'ml_value':
            candidates = [s for s in single_stacks if s['ml_proj'] is not None and s['ml_proj'] > 30]
            candidates.sort(key=lambda s: s['fwd_salary'])
        elif strategy == 'ml_dual_chalk':
            candidates = sorted([s for s in dual_stacks if s['ml_proj'] is not None],
                                key=lambda s: s['ml_proj'], reverse=True)
        elif strategy == 'ml_dual_ceiling':
            candidates = sorted([s for s in dual_stacks if s['ml_ceiling'] is not None],
                                key=lambda s: s['ml_ceiling'], reverse=True)
        else:
            continue

        # Fill mode
        if strategy in ['chalk', 'value', 'ml_chalk', 'ml_value']:
            fill_mode = 'salary'
        elif strategy in ['game_stack', 'dual_game']:
            fill_mode = 'game_corr'
        else:
            fill_mode = 'ceiling'

        # Generate top-3 instead of top-1
        filled_count = 0
        for rank, stack in enumerate(candidates[:8]):  # Try up to 8 to get 3
            lineup = fill_lineup(stack, dk_pool, actuals, fill_mode=fill_mode)
            if lineup is not None:
                variant_key = f"{strategy}_{filled_count}" if filled_count > 0 else strategy
                results[variant_key] = {
                    'lineup': lineup,
                    'stack': stack,
                }
                filled_count += 1
                if filled_count >= 3:
                    break

    return results


# ============================================================================
# BACKTEST: Compare sim-selected vs point-estimate-selected
# ============================================================================

def run_correlated_backtest(start_date='2025-11-07', n_sims=5000,
                            modes=('cash', 'gpp', 'ceiling', 'ev')):
    """
    Full walk-forward backtest comparing:
      1. Original 16-strategy point-estimate selection
      2. Sim-selected from original 16 candidates
      3. Sim-selected from expanded ~48 candidates
      4. Oracle (best possible from all candidates)
    """
    from line_multi_stack import (
        build_all_stacks, select_lineups, fill_lineup, load_actuals,
        score_players, build_line_history, train_ml_models,
        ALL_STRATEGIES, MIN_ML_TRAIN, norm as lms_norm, DB_PATH
    )

    conn = sqlite3.connect(str(DB_PATH))

    print("Loading game logs...")
    all_logs = pd.read_sql("""
        SELECT player_name, team, game_date, dk_fpts, position
        FROM game_logs_skaters ORDER BY game_date
    """, conn)
    all_logs['game_date'] = pd.to_datetime(all_logs['game_date'])
    all_logs['_key'] = all_logs['player_name'].str.lower().str.strip() + '_' + all_logs['team'].apply(norm)

    print("Building ML feature history...")
    line_history = build_line_history(conn)

    dates = pd.read_sql(f"""
        SELECT DISTINCT d.slate_date FROM dk_salaries d
        WHERE d.slate_date >= '{start_date}'
        AND EXISTS (SELECT 1 FROM game_logs_skaters g WHERE g.game_date = d.slate_date)
        ORDER BY d.slate_date
    """, conn)['slate_date'].tolist()

    print(f"\n{'='*90}")
    print(f"  CORRELATED SIMULATION BACKTEST | {len(dates)} dates | {n_sims:,} sims/lineup")
    print(f"  Modes: {', '.join(modes)}")
    print(f"{'='*90}")

    selector = CorrelatedSelector(conn, n_sims=n_sims)
    all_results = []

    for di, date_str in enumerate(dates, 1):
        dk_pool = pd.read_sql("SELECT * FROM dk_salaries WHERE slate_date = ?",
                               conn, params=(date_str,))
        actuals = load_actuals(date_str, conn)

        if dk_pool.empty or actuals.empty:
            continue

        # Contest info
        contest = conn.execute("""
            SELECT MIN(CASE WHEN n_cashed > 0 THEN score END),
                   MAX(CASE WHEN place = 1 THEN score END),
                   total_entries,
                   MAX(CASE WHEN place = 1 THEN profit END)
            FROM contest_results WHERE slate_date = ?
        """, (date_str,)).fetchone()
        cash_line = contest[0] if contest and contest[0] else 0
        first_score = contest[1] if contest and contest[1] else 0
        total_entries = contest[2] if contest and contest[2] else 0

        n_teams = dk_pool['team'].nunique()

        # Train ML models
        ml_models = train_ml_models(line_history, date_str)
        has_ml = ml_models is not None

        # Build stacks
        stacks = build_all_stacks(dk_pool, all_logs, date_str, ml_models)
        if not stacks:
            continue

        # 1. Original 16 lineups (point-estimate selected)
        orig_lineups = select_lineups(stacks, dk_pool, actuals, has_ml=has_ml)
        if not orig_lineups:
            continue

        # 2. Expanded candidates (top-3 per strategy)
        expanded = generate_expanded_candidates(stacks, dk_pool, actuals, has_ml=has_ml)

        # 3. Fit player distributions for simulation
        selector.reset_cache()
        selector.fit_distributions(dk_pool, date_str)

        # Score original lineups with actuals
        orig_actuals = {}
        for strat, data in orig_lineups.items():
            actual_total, n_matched, n_scratched = score_players(data['lineup'], actuals)
            orig_actuals[strat] = actual_total

        # Score expanded lineups with actuals
        expanded_actuals = {}
        for strat, data in expanded.items():
            actual_total, n_matched, n_scratched = score_players(data['lineup'], actuals)
            expanded_actuals[strat] = actual_total

        # Oracle: best possible from all candidates
        all_candidates = {**orig_lineups, **{k: v for k, v in expanded.items() if k not in orig_lineups}}
        all_actuals = {**orig_actuals, **{k: v for k, v in expanded_actuals.items() if k not in orig_actuals}}

        oracle_strat = max(all_actuals, key=all_actuals.get)
        oracle_actual = all_actuals[oracle_strat]

        # Best of original 16 (point-estimate oracle)
        orig_oracle_strat = max(orig_actuals, key=orig_actuals.get)
        orig_oracle_actual = orig_actuals[orig_oracle_strat]

        # 4. Run sim selection for each mode
        for mode in modes:
            target = cash_line if mode == 'cash' and cash_line > 0 else None

            # Sim-select from original 16
            sim_strat_orig, _, sim_results_orig = selector.select_best(
                orig_lineups, mode=mode,
                cash_line=cash_line if cash_line > 0 else 111,
                gpp_target=first_score if first_score > 0 else 140)

            sim_actual_orig = orig_actuals.get(sim_strat_orig, 0) if sim_strat_orig else 0

            # Sim-select from expanded candidates
            sim_strat_exp, _, sim_results_exp = selector.select_best(
                expanded, mode=mode,
                cash_line=cash_line if cash_line > 0 else 111,
                gpp_target=first_score if first_score > 0 else 140)

            sim_actual_exp = expanded_actuals.get(sim_strat_exp, 0) if sim_strat_exp else 0

            is_cash_sim_orig = sim_actual_orig >= cash_line if cash_line > 0 else None
            is_cash_sim_exp = sim_actual_exp >= cash_line if cash_line > 0 else None
            is_first_sim_orig = sim_actual_orig >= first_score if first_score > 0 else None
            is_first_sim_exp = sim_actual_exp >= first_score if first_score > 0 else None

            all_results.append({
                'date': date_str,
                'mode': mode,
                'n_teams': n_teams,
                'cash_line': cash_line,
                'first_score': first_score,
                'total_entries': total_entries,
                # Original oracle (best-of-16)
                'orig_oracle_strat': orig_oracle_strat,
                'orig_oracle_actual': orig_oracle_actual,
                'orig_oracle_cash': orig_oracle_actual >= cash_line if cash_line > 0 else None,
                'orig_oracle_first': orig_oracle_actual >= first_score if first_score > 0 else None,
                # Expanded oracle (best-of-all)
                'exp_oracle_strat': oracle_strat,
                'exp_oracle_actual': oracle_actual,
                'exp_oracle_cash': oracle_actual >= cash_line if cash_line > 0 else None,
                # Sim-selected from original 16
                'sim_orig_strat': sim_strat_orig,
                'sim_orig_actual': sim_actual_orig,
                'sim_orig_cash': is_cash_sim_orig,
                'sim_orig_first': is_first_sim_orig,
                # Sim-selected from expanded
                'sim_exp_strat': sim_strat_exp,
                'sim_exp_actual': sim_actual_exp,
                'sim_exp_cash': is_cash_sim_exp,
                'sim_exp_first': is_first_sim_exp,
                # Number of candidates
                'n_orig': len(orig_lineups),
                'n_expanded': len(expanded),
            })

        # Print progress
        best_sim = max(
            (r for r in all_results if r['date'] == date_str),
            key=lambda r: r['sim_exp_actual'], default=None)
        if best_sim:
            status = 'CASH' if best_sim['sim_exp_cash'] else ('miss' if best_sim['sim_exp_cash'] is not None else '  - ')
            print(f"  [{di:3d}] {date_str} ({n_teams:2d}t) | "
                  f"SimExp: {best_sim['sim_exp_strat']:18s} {best_sim['sim_exp_actual']:6.1f} | "
                  f"Oracle: {best_sim['orig_oracle_actual']:6.1f} | "
                  f"Cash:{cash_line:5.1f} | {status} | "
                  f"({best_sim['n_orig']}→{best_sim['n_expanded']} candidates)")

    conn.close()

    # ============================================================================
    # ANALYSIS
    # ============================================================================
    r = pd.DataFrame(all_results)
    if r.empty:
        print("No results!")
        return r

    print(f"\n{'='*90}")
    print(f"  CORRELATED SIMULATION RESULTS")
    print(f"{'='*90}")

    for mode in modes:
        mr = r[r['mode'] == mode]
        if mr.empty:
            continue

        with_cash = mr[mr['cash_line'] > 0]
        n_dates = len(with_cash)

        print(f"\n  ── MODE: {mode.upper()} ({n_dates} dates with cash line) ──")

        # 1. Original Oracle (best-of-16 with perfect hindsight)
        orig_cash = int((with_cash['orig_oracle_cash'] == True).sum())
        orig_first = int((with_cash['orig_oracle_first'] == True).sum())
        print(f"  {'Orig Oracle (best-of-16)':35s}: avg={with_cash['orig_oracle_actual'].mean():6.1f}  "
              f"cash={orig_cash}/{n_dates} ({orig_cash/max(1,n_dates)*100:.1f}%)  "
              f"1st={orig_first}/{n_dates} ({orig_first/max(1,n_dates)*100:.1f}%)")

        # 2. Expanded Oracle
        exp_cash = int((with_cash['exp_oracle_cash'] == True).sum())
        print(f"  {'Exp Oracle (best-of-all)':35s}: avg={with_cash['exp_oracle_actual'].mean():6.1f}  "
              f"cash={exp_cash}/{n_dates} ({exp_cash/max(1,n_dates)*100:.1f}%)")

        # 3. Sim-selected from original 16
        sim_orig_cash = int((with_cash['sim_orig_cash'] == True).sum())
        sim_orig_first = int((with_cash['sim_orig_first'] == True).sum())
        print(f"  {'Sim-Selected (from 16)':35s}: avg={with_cash['sim_orig_actual'].mean():6.1f}  "
              f"cash={sim_orig_cash}/{n_dates} ({sim_orig_cash/max(1,n_dates)*100:.1f}%)  "
              f"1st={sim_orig_first}/{n_dates} ({sim_orig_first/max(1,n_dates)*100:.1f}%)")

        # 4. Sim-selected from expanded
        sim_exp_cash = int((with_cash['sim_exp_cash'] == True).sum())
        sim_exp_first = int((with_cash['sim_exp_first'] == True).sum())
        print(f"  {'Sim-Selected (expanded)':35s}: avg={with_cash['sim_exp_actual'].mean():6.1f}  "
              f"cash={sim_exp_cash}/{n_dates} ({sim_exp_cash/max(1,n_dates)*100:.1f}%)  "
              f"1st={sim_exp_first}/{n_dates} ({sim_exp_first/max(1,n_dates)*100:.1f}%)")

        # Gap analysis
        orig_oracle_avg = with_cash['orig_oracle_actual'].mean()
        sim_orig_avg = with_cash['sim_orig_actual'].mean()
        sim_exp_avg = with_cash['sim_exp_actual'].mean()
        print(f"\n  Gap vs Oracle:  Sim16={orig_oracle_avg - sim_orig_avg:+.1f}  "
              f"SimExp={orig_oracle_avg - sim_exp_avg:+.1f}")

        # Which strategies does the sim pick most?
        print(f"  Sim picks (from 16):")
        for strat, cnt in with_cash['sim_orig_strat'].value_counts().head(5).items():
            print(f"    {strat:20s}: {cnt:2d} ({cnt/n_dates*100:.0f}%)")

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = Path(DB_PATH).parent.parent / 'backtests' / f"correlated_sim_backtest_{ts}.csv"
    r.to_csv(str(output), index=False)
    print(f"\n  Saved: {output}")

    return r


# ============================================================================
# MATH UTILITIES (vectorized)
# ============================================================================

try:
    from scipy.special import erf as _scipy_erf
    from scipy.stats import norm as _scipy_norm

    def _norm_cdf(x):
        return 0.5 * (1.0 + _scipy_erf(np.asarray(x) / np.sqrt(2.0)))

    def _norm_ppf(u):
        u = np.clip(np.asarray(u, dtype=np.float64), 1e-8, 1 - 1e-8)
        return _scipy_norm.ppf(u)

except ImportError:
    def _norm_cdf(x):
        a, b, c = 0.3480242, 0.0958798, 0.7478556
        t = 1.0 / (1.0 + 0.47047 * np.abs(x))
        val = 1.0 - t * (a + t * (-b + t * c)) * np.exp(-0.5 * x * x)
        return np.where(x >= 0, val, 1.0 - val)

    def _norm_ppf(u):
        u = np.clip(np.asarray(u, dtype=np.float64), 1e-8, 1 - 1e-8)
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
        mask_low = u < p_low
        if mask_low.any():
            q = np.sqrt(-2.0 * np.log(u[mask_low]))
            result[mask_low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        mask_mid = (~mask_low) & (u <= p_high)
        if mask_mid.any():
            q = u[mask_mid] - 0.5
            r = q * q
            result[mask_mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        mask_high = u > p_high
        if mask_high.any():
            q = np.sqrt(-2.0 * np.log(1.0 - u[mask_high]))
            result[mask_high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        return result


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    n_sims = 5000
    for arg in sys.argv[1:]:
        if arg.startswith('--sims='):
            n_sims = int(arg.split('=')[1])

    print(f"Running correlated simulation backtest with {n_sims:,} sims/lineup...")
    run_correlated_backtest(n_sims=n_sims)
