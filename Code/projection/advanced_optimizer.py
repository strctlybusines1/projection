#!/usr/bin/env python3
"""
Advanced Optimizer — Phase 1+2: Thompson Sampling Bandit + Genetic Algorithm
=============================================================================

Phase 1: Contextual Bandit (Thompson Sampling)
  - Learns which of 16 strategies to play given slate context
  - Context features: n_teams, avg_game_total, max_impl, impl_spread, etc.
  - Bayesian posterior updated after each date's results
  - Predicts best strategy BEFORE games (not oracle best-of-16 after)

Phase 2: Genetic Algorithm Optimizer
  - Population-based search over full 9-player lineups
  - Crossover: combine two lineups' player selections
  - Mutation: swap random players for alternatives
  - Fitness: weighted sum of projection + ceiling + correlation
  - Escapes local optima that greedy fill_lineup misses

Both integrate with line_multi_stack.py backtest infrastructure.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import from line_multi_stack
from line_multi_stack import (
    norm, SALARY_CAP, DB_PATH, BACKTESTS_DIR,
    build_line_history, train_ml_models, build_all_stacks,
    select_lineups, fill_lineup, score_players, load_actuals,
    ALL_STRATEGIES, HEURISTIC_STRATEGIES, ML_STRATEGIES,
    _get_team_lines, _find_game_matchups, _score_d_ceiling, _score_g_ceiling,
)


# ==============================================================================
# PHASE 1: THOMPSON SAMPLING CONTEXTUAL BANDIT
# ==============================================================================

class ThompsonSamplingBandit:
    """Contextual bandit that learns which strategy to play per slate.

    Uses Bayesian Linear Regression (Thompson Sampling):
    - For each strategy, maintains posterior over weight vector w
    - Context x = slate features, reward = actual FPTS
    - At decision time: sample w ~ posterior, pick strategy with highest x @ w
    - After observing reward: update posterior via Bayesian update

    This is the LinTS (Linear Thompson Sampling) algorithm from
    Agrawal & Goyal (2013).
    """

    def __init__(self, n_features, strategies, lambda_prior=1.0, noise_var=100.0):
        self.strategies = strategies
        self.n_strategies = len(strategies)
        self.n_features = n_features
        self.noise_var = noise_var  # Observation noise variance

        # Per-strategy Bayesian linear regression parameters
        # Posterior: w_i ~ N(mu_i, Sigma_i)
        self.B = {}  # B = X^T X + lambda * I (precision matrix)
        self.f = {}  # f = X^T y (sufficient statistic)
        self.mu = {}  # mu = B^{-1} f (posterior mean)

        for s in strategies:
            self.B[s] = lambda_prior * np.eye(n_features)
            self.f[s] = np.zeros(n_features)
            self.mu[s] = np.zeros(n_features)

        self.history = []

    def select_strategy(self, context, available_strategies=None):
        """Select strategy via Thompson Sampling.

        Args:
            context: feature vector (n_features,)
            available_strategies: subset of strategies to choose from

        Returns:
            (strategy_name, expected_reward)
        """
        if available_strategies is None:
            available_strategies = self.strategies

        best_strategy = None
        best_sample = -np.inf

        for s in available_strategies:
            # Sample weight vector from posterior
            try:
                B_inv = np.linalg.inv(self.B[s])
                w_sample = np.random.multivariate_normal(self.mu[s], self.noise_var * B_inv)
            except np.linalg.LinAlgError:
                # Fallback: use posterior mean if matrix is singular
                w_sample = self.mu[s]

            # Predicted reward = context @ sampled_weights
            reward_sample = context @ w_sample

            if reward_sample > best_sample:
                best_sample = reward_sample
                best_strategy = s

        return best_strategy, best_sample

    def update(self, strategy, context, reward):
        """Update posterior after observing reward.

        Bayesian linear regression update:
          B_new = B_old + x x^T
          f_new = f_old + x * y
          mu_new = B_new^{-1} f_new
        """
        x = context.reshape(-1, 1)
        self.B[strategy] += x @ x.T
        self.f[strategy] += context * reward

        try:
            self.mu[strategy] = np.linalg.solve(self.B[strategy], self.f[strategy])
        except np.linalg.LinAlgError:
            pass

        self.history.append({
            'strategy': strategy,
            'reward': reward,
        })

    def get_strategy_stats(self):
        """Get posterior mean reward for each strategy at mean context."""
        stats = {}
        for s in self.strategies:
            stats[s] = {
                'posterior_mean': self.mu[s].tolist(),
                'n_updates': sum(1 for h in self.history if h['strategy'] == s),
            }
        return stats


def build_slate_context(dk_pool, date_str):
    """Build context feature vector for a slate.

    Features capture slate structure that determines which strategy works best:
    - Slate size (small slates → more concentrated, favor chalk)
    - Game total spread (high spread → target specific game, favor game_stack)
    - Implied total spread (high spread → clear favorites, favor chalk)
    - Average/max implied total (high scoring environment)
    """
    n_teams = dk_pool['team'].nunique()
    n_players = len(dk_pool)

    impl = dk_pool['team_implied_total'].dropna()
    gt = dk_pool['game_total'].dropna()

    features = np.array([
        n_teams,
        n_teams <= 10,  # Small slate indicator
        n_teams >= 20,  # Large slate indicator
        impl.mean() if len(impl) > 0 else 3.0,
        impl.max() if len(impl) > 0 else 3.5,
        impl.min() if len(impl) > 0 else 2.5,
        (impl.max() - impl.min()) if len(impl) > 0 else 1.0,
        gt.mean() if len(gt) > 0 else 6.0,
        gt.max() if len(gt) > 0 else 6.5,
        (gt.max() - gt.min()) if len(gt) > 0 else 1.0,
        dk_pool['salary'].mean() / 1000,
        1.0,  # Bias term
    ], dtype=np.float64)

    return features


N_CONTEXT_FEATURES = 12  # Must match build_slate_context output


# ==============================================================================
# PHASE 2: GENETIC ALGORITHM LINEUP OPTIMIZER
# ==============================================================================

class GeneticAlgorithmOptimizer:
    """Evolves lineups via crossover + mutation to escape greedy local optima.

    Population: set of complete 9-player lineups (2C, 3W, 2D, 1G, 1UTIL)
    Fitness: projection + ceiling + correlation bonuses
    Crossover: take subset of players from parent A, fill rest from parent B
    Mutation: swap 1-2 players with alternatives at similar salary

    Key advantage over greedy: explores combinations like "slightly worse D1
    but enables much better G" that sequential fill can't find.
    """

    def __init__(self, pop_size=100, generations=150, mutation_rate=0.15,
                 elite_pct=0.2, tournament_size=5):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_pct = elite_pct
        self.tournament_size = tournament_size

    def optimize(self, dk_pool, stack_forwards, stack_teams,
                 actuals=None, seed_lineups=None):
        """Run GA to find optimal D/G fill for a given forward stack.

        Args:
            dk_pool: full DK player pool DataFrame
            stack_forwards: list of forward dicts (the 6-player stack core)
            stack_teams: list of team abbreviations in stack
            actuals: if backtesting, filter to confirmed players
            seed_lineups: optional list of seed lineups from heuristic fill

        Returns:
            best lineup (list of player dicts) or None
        """
        fwd_names = set(f['player_name'] for f in stack_forwards)
        fwd_salary = sum(f['salary'] for f in stack_forwards)
        remaining = SALARY_CAP - fwd_salary

        # Build candidate pools
        pool_d = dk_pool[(dk_pool['position'] == 'D') &
                         (~dk_pool['player_name'].isin(fwd_names))].copy()
        pool_g = dk_pool[dk_pool['position'] == 'G'].copy()

        if actuals is not None:
            actual_keys = set(actuals['_key'].tolist())
            pool_d['_k'] = pool_d['player_name'].str.lower().str.strip() + '_' + pool_d['team'].apply(norm)
            pool_g['_k'] = pool_g['player_name'].str.lower().str.strip() + '_' + pool_g['team'].apply(norm)
            pool_d = pool_d[pool_d['_k'].isin(actual_keys)]
            pool_g = pool_g[pool_g['_k'].isin(actual_keys)]

        d_list = pool_d.to_dict('records')
        g_list = pool_g.to_dict('records')

        if len(d_list) < 2 or len(g_list) < 1:
            return None

        # Pre-compute scores for fitness function
        stack_game_teams = set()
        for f in stack_forwards:
            stack_game_teams.add(norm(str(f.get('team', ''))))
        stack_gt = stack_forwards[0].get('game_total', 0) if stack_forwards else 0
        if stack_gt > 0:
            for d in d_list:
                d_gt = d.get('game_total', 0)
                if pd.notna(d_gt) and abs(d_gt - stack_gt) < 0.1:
                    stack_game_teams.add(norm(str(d.get('team', ''))))

        # Identify opponent teams for goalie penalty
        opponent_teams = set()
        for st in stack_teams:
            st_rows = dk_pool[dk_pool['team'].apply(norm) == norm(st)]
            if not st_rows.empty:
                s_gt = st_rows.iloc[0].get('game_total', 0)
                if s_gt > 0:
                    for d in d_list + g_list:
                        d_team = norm(str(d.get('team', '')))
                        if d_team not in [norm(t) for t in stack_teams]:
                            d_gt = d.get('game_total', 0)
                            if pd.notna(d_gt) and abs(d_gt - s_gt) < 0.1:
                                opponent_teams.add(d_team)

        norm_stack_teams = [norm(str(t)) for t in stack_teams]

        def fitness(d1_idx, d2_idx, g_idx):
            """Score a D1+D2+G combination."""
            d1, d2, g = d_list[d1_idx], d_list[d2_idx], g_list[g_idx]
            total_salary = d1['salary'] + d2['salary'] + g['salary']
            if total_salary > remaining:
                return -1000  # Invalid

            # D scoring (ceiling-weighted with stack correlation)
            d1_score = _score_d_ceiling(d1, norm_stack_teams[0] if norm_stack_teams else None,
                                         stack_game_teams, norm_stack_teams)
            d2_score = _score_d_ceiling(d2, norm_stack_teams[0] if norm_stack_teams else None,
                                         stack_game_teams, norm_stack_teams)

            # G scoring (with stack team preference + opponent penalty)
            g_score = _score_g_ceiling(g, norm_stack_teams[0] if norm_stack_teams else None,
                                        norm_stack_teams, opponent_teams)

            # Salary efficiency bonus (use more cap = more talent)
            salary_used = fwd_salary + total_salary
            salary_pct = salary_used / SALARY_CAP
            salary_bonus = max(0, (salary_pct - 0.95)) * 200  # Bonus for >95% usage

            # Diversity bonus: D from different teams (if not stack-correlated)
            d1_team = norm(str(d1.get('team', '')))
            d2_team = norm(str(d2.get('team', '')))
            diversity = 5 if d1_team != d2_team else 0

            return d1_score + d2_score + g_score + salary_bonus + diversity

        # Initialize population
        population = []
        n_d = len(d_list)
        n_g = len(g_list)

        # Seed with heuristic solutions if available
        if seed_lineups:
            for sl in seed_lineups[:5]:
                d_indices = []
                g_idx = 0
                for p in sl:
                    if p.get('position') == 'D' and p.get('role') == 'fill':
                        for i, d in enumerate(d_list):
                            if d['player_name'] == p['name']:
                                d_indices.append(i)
                                break
                    elif p.get('position') == 'G':
                        for i, g in enumerate(g_list):
                            if g['player_name'] == p['name']:
                                g_idx = i
                                break
                if len(d_indices) == 2:
                    population.append((d_indices[0], d_indices[1], g_idx))

        # Fill rest with random valid lineups
        attempts = 0
        while len(population) < self.pop_size and attempts < self.pop_size * 20:
            d1 = np.random.randint(n_d)
            d2 = np.random.randint(n_d)
            g = np.random.randint(n_g)
            if d1 != d2:
                sal = d_list[d1]['salary'] + d_list[d2]['salary'] + g_list[g]['salary']
                if sal <= remaining:
                    population.append((d1, d2, g))
            attempts += 1

        if len(population) < 10:
            return None  # Not enough valid combinations

        # Evolution loop
        for gen in range(self.generations):
            # Score population
            scored = [(ind, fitness(*ind)) for ind in population]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Elite selection
            n_elite = max(2, int(len(scored) * self.elite_pct))
            new_pop = [s[0] for s in scored[:n_elite]]

            # Fill rest via tournament selection + crossover + mutation
            while len(new_pop) < self.pop_size:
                # Tournament selection
                parent_a = self._tournament_select(scored)
                parent_b = self._tournament_select(scored)

                # Crossover: take D from one parent, G from other
                if np.random.random() < 0.5:
                    child = (parent_a[0], parent_a[1], parent_b[2])
                else:
                    child = (parent_b[0], parent_b[1], parent_a[2])

                # Mutation
                child = self._mutate(child, n_d, n_g, d_list, g_list, remaining)

                # Validate
                if child[0] != child[1]:
                    sal = d_list[child[0]]['salary'] + d_list[child[1]]['salary'] + g_list[child[2]]['salary']
                    if sal <= remaining:
                        new_pop.append(child)

            population = new_pop[:self.pop_size]

        # Return best lineup
        best_scored = [(ind, fitness(*ind)) for ind in population]
        best_scored.sort(key=lambda x: x[1], reverse=True)
        best = best_scored[0][0]

        d1, d2, g = d_list[best[0]], d_list[best[1]], g_list[best[2]]

        players = []
        for f in stack_forwards:
            players.append({'name': f['player_name'], 'team': f['team'],
                           'salary': f['salary'], 'position': f['position'], 'role': 'stack'})
        for d in [d1, d2]:
            players.append({'name': d['player_name'], 'team': d['team'],
                           'salary': d['salary'], 'position': 'D', 'role': 'fill_ga'})
        players.append({'name': g['player_name'], 'team': g['team'],
                       'salary': g['salary'], 'position': 'G', 'role': 'fill_ga'})

        return players

    def _tournament_select(self, scored):
        """Tournament selection: pick best from random subset."""
        indices = np.random.choice(len(scored), size=min(self.tournament_size, len(scored)), replace=False)
        best_idx = max(indices, key=lambda i: scored[i][1])
        return scored[best_idx][0]

    def _mutate(self, individual, n_d, n_g, d_list, g_list, remaining):
        """Mutate individual by swapping D or G."""
        d1, d2, g = individual
        if np.random.random() < self.mutation_rate:
            # Mutate D1
            d1 = np.random.randint(n_d)
        if np.random.random() < self.mutation_rate:
            # Mutate D2
            d2 = np.random.randint(n_d)
        if np.random.random() < self.mutation_rate:
            # Mutate G
            g = np.random.randint(n_g)
        return (d1, d2, g)


# ==============================================================================
# GA-ENHANCED FILL (replaces fill_lineup for selected strategies)
# ==============================================================================

def fill_lineup_ga(stack, dk_pool, actuals=None, ga_params=None):
    """Fill a stack's D/G slots using Genetic Algorithm.

    Replaces the greedy fill_lineup for strategies that benefit from
    broader search (ceiling, contrarian, pp1_stack, dual variants).
    """
    if ga_params is None:
        ga_params = {'pop_size': 80, 'generations': 100,
                     'mutation_rate': 0.15, 'elite_pct': 0.2}

    ga = GeneticAlgorithmOptimizer(**ga_params)
    stack_teams = [str(t) for t in stack.get('teams', [stack.get('team', '')])]

    # Get seed from heuristic fill for warm-start
    seed = fill_lineup(stack, dk_pool, actuals, fill_mode='ceiling')
    seeds = [seed] if seed else []

    result = ga.optimize(
        dk_pool=dk_pool,
        stack_forwards=stack['forwards'],
        stack_teams=stack_teams,
        actuals=actuals,
        seed_lineups=seeds,
    )

    return result


# ==============================================================================
# INTEGRATED BACKTEST: Bandit + GA + Original
# ==============================================================================

def run_advanced_backtest(start_date='2025-11-07', confirmed_only=True):
    """Run backtest comparing:
      1. Original 16 strategies (line_multi_stack.py)
      2. GA-enhanced fill for ceiling/contrarian strategies
      3. Thompson Sampling bandit for strategy selection

    The bandit trains walk-forward: only sees past dates' results.
    """
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
    print(f"  Line records: {len(line_history)}, Features: 29")

    dates = pd.read_sql(f"""
        SELECT DISTINCT d.slate_date FROM dk_salaries d
        WHERE d.slate_date >= '{start_date}'
        AND EXISTS (SELECT 1 FROM game_logs_skaters g WHERE g.game_date = d.slate_date)
        ORDER BY d.slate_date
    """, conn)['slate_date'].tolist()

    # Initialize Thompson Sampling bandit
    bandit = ThompsonSamplingBandit(
        n_features=N_CONTEXT_FEATURES,
        strategies=ALL_STRATEGIES,
        lambda_prior=1.0,
        noise_var=400.0,  # High variance for FPTS outcomes
    )

    # GA optimizer (shared across dates)
    ga = GeneticAlgorithmOptimizer(
        pop_size=80,
        generations=100,
        mutation_rate=0.15,
        elite_pct=0.2,
    )

    # Strategies that get GA fill (ceiling-oriented, benefit from broader search)
    GA_STRATEGIES = ['ceiling', 'pp1_stack', 'contrarian_1', 'contrarian_2',
                     'dual_ceiling', 'dual_chalk', 'ml_ceiling']

    print(f"\n{'='*100}")
    print(f"  ADVANCED BACKTEST: Bandit + GA + Original | {len(dates)} dates")
    print(f"  Strategies: {len(ALL_STRATEGIES)} original + GA variants + bandit selection")
    print(f"  GA-enhanced strategies: {', '.join(GA_STRATEGIES)}")
    print(f"{'='*100}")

    all_results = []
    bandit_picks = []
    ml_active_count = 0

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
        first_profit = contest[3] if contest and contest[3] else 0

        n_teams = dk_pool['team'].nunique()

        # Train ML models (walk-forward)
        ml_models = train_ml_models(line_history, date_str)
        has_ml = ml_models is not None
        if has_ml:
            ml_active_count += 1

        # Build stacks
        stacks = build_all_stacks(dk_pool, all_logs, date_str, ml_models)
        if not stacks:
            continue

        # ─── ORIGINAL STRATEGIES (baseline) ──────────────────────────────
        lineups_orig = select_lineups(
            stacks, dk_pool,
            actuals if confirmed_only else None,
            has_ml=has_ml
        )

        # ─── GA-ENHANCED STRATEGIES ──────────────────────────────────────
        lineups_ga = {}
        for strat, data in lineups_orig.items():
            if strat in GA_STRATEGIES:
                ga_lineup = fill_lineup_ga(
                    data['stack'], dk_pool,
                    actuals if confirmed_only else None,
                )
                if ga_lineup:
                    lineups_ga[f"ga_{strat}"] = {
                        'lineup': ga_lineup,
                        'stack': data['stack'],
                    }

        # ─── BANDIT SELECTION ────────────────────────────────────────────
        context = build_slate_context(dk_pool, date_str)
        available = [s for s in ALL_STRATEGIES if s in lineups_orig]
        bandit_strategy, bandit_expected = bandit.select_strategy(context, available)

        # Score all lineups
        best_strat = None
        best_actual = 0
        date_results = {}

        for source, lineups in [('orig', lineups_orig), ('ga', lineups_ga)]:
            for strat, data in lineups.items():
                lineup = data['lineup']
                stack = data['stack']
                actual_total, n_matched, n_scratched = score_players(lineup, actuals)

                is_cash = actual_total >= cash_line if cash_line > 0 else None
                is_first = actual_total >= first_score if first_score > 0 else None

                if strat.startswith('ml_'):
                    projected = stack.get('ml_proj', 0) or 0
                elif strat.startswith('ga_'):
                    base_strat = strat[3:]
                    projected = stack.get('ml_proj' if base_strat.startswith('ml_') else 'combo_proj', 0) or 0
                else:
                    projected = stack.get('combo_proj', 0) or 0

                lineup_teams = set(norm(str(p['team'])) for p in lineup)
                fwd_teams = set(norm(str(p['team'])) for p in lineup if p['role'] == 'stack')

                result = {
                    'date': date_str,
                    'strategy': strat,
                    'source': source,
                    'scoring': 'ML' if strat.startswith('ml_') else ('GA' if strat.startswith('ga_') else 'heuristic'),
                    'stack_type': stack.get('stack_type', 'single'),
                    'stack_team': stack['team'],
                    'impl_rank': stack['impl_rank'],
                    'projected': projected,
                    'actual': actual_total,
                    'matched': n_matched,
                    'scratched': n_scratched,
                    'salary': sum(p['salary'] for p in lineup),
                    'cash_line': cash_line,
                    'first_score': first_score,
                    'total_entries': total_entries,
                    'n_teams': n_teams,
                    'n_lineup_teams': len(lineup_teams),
                    'n_fwd_teams': len(fwd_teams),
                    'is_cash': is_cash,
                    'is_first': is_first,
                }

                all_results.append(result)
                date_results[strat] = actual_total

                if actual_total > best_actual:
                    best_actual = actual_total
                    best_strat = strat

        # ─── UPDATE BANDIT ───────────────────────────────────────────────
        # Update bandit with actual results for each original strategy
        for strat in available:
            reward = date_results.get(strat, 0)
            bandit.update(strat, context, reward)

        # Record bandit's pick
        bandit_actual = date_results.get(bandit_strategy, 0)
        bandit_cash = bandit_actual >= cash_line if cash_line > 0 else None
        bandit_first = bandit_actual >= first_score if first_score > 0 else None
        bandit_picks.append({
            'date': date_str,
            'bandit_pick': bandit_strategy,
            'bandit_actual': bandit_actual,
            'bandit_cash': bandit_cash,
            'bandit_first': bandit_first,
            'oracle_best': best_strat,
            'oracle_actual': best_actual,
            'n_teams': n_teams,
            'cash_line': cash_line,
            'first_score': first_score,
        })

        # Print progress
        status = 'CASH' if bandit_cash else ('miss' if bandit_cash is not None else '  - ')
        b_flag = ' 1ST!' if bandit_first else ''
        best_status = 'CASH' if (best_actual >= cash_line if cash_line > 0 else False) else ''
        print(f"  [{di:3d}] {date_str} ({n_teams:2d}t) | "
              f"Bandit:{bandit_strategy:16s} {bandit_actual:6.1f} {status}{b_flag} | "
              f"Oracle:{best_strat:16s} {best_actual:6.1f} {best_status} | "
              f"GA best: {max((v for k,v in date_results.items() if k.startswith('ga_')), default=0):.1f}")

    conn.close()

    # ==============================================================================
    # ANALYSIS
    # ==============================================================================
    r = pd.DataFrame(all_results)
    bp = pd.DataFrame(bandit_picks)

    if r.empty:
        print("No results!")
        return r, bp

    print(f"\n{'='*100}")
    print(f"  ADVANCED RESULTS SUMMARY")
    print(f"{'='*100}")

    # ── BANDIT PERFORMANCE ──
    print(f"\n  ── THOMPSON SAMPLING BANDIT ──")
    bp_cash = bp[bp['cash_line'] > 0]
    if not bp_cash.empty:
        n_cash_b = int(bp_cash['bandit_cash'].sum())
        n_first_b = int(bp_cash['bandit_first'].sum()) if 'bandit_first' in bp_cash else 0
        n_cash_o = int((bp_cash['oracle_actual'] >= bp_cash['cash_line']).sum())
        n_first_o = int((bp_cash['oracle_actual'] >= bp_cash['first_score']).sum())
        n_dates = len(bp_cash)

        print(f"  Bandit avg FPTS:   {bp_cash['bandit_actual'].mean():.1f}")
        print(f"  Oracle avg FPTS:   {bp_cash['oracle_actual'].mean():.1f}")
        print(f"  Bandit cash rate:  {n_cash_b}/{n_dates} ({n_cash_b/n_dates*100:.1f}%)")
        print(f"  Oracle cash rate:  {n_cash_o}/{n_dates} ({n_cash_o/n_dates*100:.1f}%)")
        print(f"  Bandit 1st rate:   {n_first_b}/{n_dates} ({n_first_b/n_dates*100:.1f}%)")
        print(f"  Oracle 1st rate:   {n_first_o}/{n_dates} ({n_first_o/n_dates*100:.1f}%)")
        print(f"  Bandit vs Oracle gap: {(bp_cash['oracle_actual'] - bp_cash['bandit_actual']).mean():.1f} FPTS")

    # Bandit strategy distribution
    print(f"\n  Bandit picks distribution:")
    for strat, count in bp['bandit_pick'].value_counts().items():
        avg = bp[bp['bandit_pick'] == strat]['bandit_actual'].mean()
        print(f"    {strat:20s}: {count:3d} times, avg {avg:.1f} FPTS")

    # ── GA vs ORIGINAL ──
    print(f"\n  ── GA vs ORIGINAL (same stacks, different D/G fill) ──")
    for ga_strat in sorted(set(s for s in r['strategy'] if s.startswith('ga_'))):
        orig_strat = ga_strat[3:]  # Remove 'ga_' prefix
        ga_rows = r[r['strategy'] == ga_strat]
        orig_rows = r[r['strategy'] == orig_strat]

        common_dates = set(ga_rows['date']) & set(orig_rows['date'])
        if not common_dates:
            continue

        ga_common = ga_rows[ga_rows['date'].isin(common_dates)]
        orig_common = orig_rows[orig_rows['date'].isin(common_dates)]

        ga_avg = ga_common['actual'].mean()
        orig_avg = orig_common['actual'].mean()
        diff = ga_avg - orig_avg

        # Per-date comparison
        ga_by_date = ga_common.set_index('date')['actual']
        orig_by_date = orig_common.set_index('date')['actual']
        common_idx = ga_by_date.index.intersection(orig_by_date.index)
        ga_wins = int((ga_by_date[common_idx] > orig_by_date[common_idx]).sum())
        orig_wins = int((orig_by_date[common_idx] > ga_by_date[common_idx]).sum())
        ties = len(common_idx) - ga_wins - orig_wins

        mark = '↑' if diff > 0.5 else ('↓' if diff < -0.5 else '→')
        print(f"  {mark} {ga_strat:20s} vs {orig_strat:20s}: "
              f"GA={ga_avg:.1f} orig={orig_avg:.1f} diff={diff:+.1f} | "
              f"GA wins {ga_wins}, orig wins {orig_wins}, ties {ties}")

    # ── BEST-OF-ALL (including GA variants) ──
    print(f"\n  ── BEST-OF-ALL COMPARISON ──")
    for label, filt in [
        ('Original 16', r[r['source'] == 'orig']),
        ('GA variants only', r[r['source'] == 'ga']),
        ('All (orig + GA)', r),
    ]:
        if filt.empty:
            continue
        best = filt.groupby('date').apply(
            lambda g: g.nlargest(1, 'actual').iloc[0], include_groups=False)
        with_cash = best[best['cash_line'] > 0]
        n_cash = int((with_cash['is_cash'] == True).sum())
        n_first = int((with_cash['is_first'] == True).sum())
        n_dates = len(with_cash)
        print(f"  {label:25s}: avg={best['actual'].mean():.1f}, "
              f"cash={n_cash}/{n_dates} ({n_cash/n_dates*100:.1f}%), "
              f"1st={n_first}/{n_dates} ({n_first/n_dates*100:.1f}%)")

    # ── Bandit vs Best-of-All ──
    if not bp_cash.empty:
        print(f"\n  ── STRATEGY SELECTION COMPARISON ──")
        print(f"  Bandit (1 pick/night):  avg={bp_cash['bandit_actual'].mean():.1f} FPTS, "
              f"cash={n_cash_b}/{n_dates} ({n_cash_b/n_dates*100:.1f}%)")
        # Random strategy baseline
        orig_only = r[r['source'] == 'orig']
        random_avg = orig_only.groupby('date')['actual'].mean().mean()
        print(f"  Random strategy:        avg={random_avg:.1f} FPTS")
        print(f"  Oracle best-of-all:     avg={bp_cash['oracle_actual'].mean():.1f} FPTS, "
              f"cash={n_cash_o}/{n_dates}")

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = BACKTESTS_DIR / f"advanced_backtest_{ts}.csv"
    r.to_csv(str(output), index=False)
    bp_output = BACKTESTS_DIR / f"bandit_picks_{ts}.csv"
    bp.to_csv(str(bp_output), index=False)
    print(f"\n  Saved: {output}")
    print(f"  Saved: {bp_output}")

    return r, bp


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2025-11-07')
    parser.add_argument('--all-players', action='store_true')
    args = parser.parse_args()

    run_advanced_backtest(
        start_date=args.start,
        confirmed_only=not args.all_players,
    )
