#!/usr/bin/env python3
"""
Comprehensive NHL DFS Backtest Engine (2025-10-07 to 2026-02-19)
==================================================================

Tests the NHL DFS projection system against ALL 113 historical dates.

Key features:
  1. Team normalization (NJ→NJD, LA→LAK, SJ→SJS, TB→TBL)
  2. Scratch player detection (don't penalize for unplayed selections)
  3. Salary-rank baseline strategy (projected_fpts = salary / 1000)
  4. SE optimizer strategy (uses daily_projections CSVs)
  5. Cash line detection from contest_results
  6. Wilson confidence intervals on cash rates
  7. Per-date detail CSV output

Usage:
    python backtest_full.py                  # Full backtest (all 113 dates)
    python backtest_full.py --method se      # SE optimizer only
    python backtest_full.py --method salary  # Salary-rank baseline only
"""

import sqlite3
import pandas as pd
import numpy as np
import glob
import warnings
import time
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "data" / "nhl_dfs_history.db"
BACKTESTS_DIR = PROJECT_DIR / "backtests"
MASTER_SEED = 2026

# Ensure output directory exists
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# TEAM NORMALIZATION (using team_normalize.py)
# ==============================================================================

def get_normalize_team():
    """Import and return the normalize_team function."""
    try:
        from team_normalize import normalize_team_lower
        return normalize_team_lower
    except ImportError:
        # Fallback
        TEAM_NORM = {
            'nj': 'njd', 'la': 'lak', 'sj': 'sjs', 'tb': 'tbl',
            'njd': 'njd', 'lak': 'lak', 'sjs': 'sjs', 'tbl': 'tbl',
        }
        def normalize_team_lower(team: str) -> str:
            t = team.lower().strip()
            return TEAM_NORM.get(t, t)
        return normalize_team_lower

normalize_team = get_normalize_team()


# ==============================================================================
# SCORING ENGINE
# ==============================================================================

def load_actuals(date_str: str, conn) -> pd.DataFrame:
    """Load actual DK FPTS from game logs. Single source of truth."""
    skaters = pd.read_sql_query(
        "SELECT player_name as name, team, dk_fpts as actual_fpts FROM game_logs_skaters WHERE game_date = ?",
        conn, params=(date_str,))
    goalies = pd.read_sql_query(
        "SELECT player_name as name, team, 'G' as position, dk_fpts as actual_fpts FROM game_logs_goalies WHERE game_date = ?",
        conn, params=(date_str,))
    actuals = pd.concat([skaters, goalies], ignore_index=True)

    # Normalize team abbreviations
    actuals['team_norm'] = actuals['team'].apply(normalize_team)
    # Build composite key: name_team (lowercase, stripped, normalized)
    actuals['_key'] = actuals['name'].str.lower().str.strip() + '_' + actuals['team_norm']

    return actuals


def score_lineup(players: List[Dict], actuals: pd.DataFrame) -> Tuple[float, int, int]:
    """
    Score a lineup against actuals using fuzzy name+team matching.

    Returns:
        (total_actual_fpts, n_matched, n_scratched)

    Scratched players (in DK pool but not in actuals) don't penalize the score.
    """
    total = 0.0
    matched = 0
    scratched = 0
    actuals_keys = actuals['_key'].tolist()

    for p in players:
        name = str(p.get('name', '')).lower().strip()
        team = normalize_team(str(p.get('team', '')))
        key = f"{name}_{team}"

        # Exact match first
        exact = actuals[actuals['_key'] == key]
        if not exact.empty:
            total += exact.iloc[0]['actual_fpts']
            matched += 1
            continue

        # Fuzzy match (same team, best name similarity)
        team_players = actuals[actuals['team_norm'] == team]
        if not team_players.empty:
            best_score = 0
            best_fpts = 0
            for _, row in team_players.iterrows():
                score = SequenceMatcher(None, name, row['name'].lower().strip()).ratio()
                if score > best_score:
                    best_score = score
                    best_fpts = row['actual_fpts']
            if best_score >= 0.70:
                total += best_fpts
                matched += 1
                continue

        # Not matched: check if this is a scratch player
        # A player is scratched if they're not in actuals at all for their team
        team_players_all = actuals[actuals['team_norm'] == team]
        if team_players_all.empty:
            # Team didn't play — scratch
            scratched += 1
        else:
            # Team played but player not found — likely scratch
            scratched += 1

    return total, matched, scratched


def load_contest_data(date_str: str, conn) -> Dict:
    """Load cash line for a date. Returns minimum score where n_cashed > 0."""
    row = conn.execute("""
        SELECT MIN(CASE WHEN n_cashed > 0 THEN score END) as cash_line,
               MAX(score) as winning_score
        FROM contest_results WHERE slate_date = ?
    """, (date_str,)).fetchone()
    return {
        'cash_line': row[0] if row and row[0] else 0,
        'winning_score': row[1] if row and row[1] else 999,
    }


def load_dk_pool(date_str: str, conn) -> pd.DataFrame:
    """Load DK salary pool for a date."""
    df = pd.read_sql_query(
        "SELECT player_name as name, team, position, salary FROM dk_salaries WHERE slate_date = ?",
        conn, params=(date_str,))
    return df


# ==============================================================================
# METHOD A: Salary-Rank Baseline
# ==============================================================================

def run_salary_rank_optimizer(date_str: str, conn, seed: int = MASTER_SEED) -> Optional[List[Dict]]:
    """
    Build a baseline lineup using salary as a proxy for value.

    Strategy: projected_fpts = salary / 1000, then optimize.
    This is a simple "follow the money" approach.
    """
    np.random.seed(seed)

    pool = load_dk_pool(date_str, conn)
    if pool.empty:
        return None

    # Set projected_fpts = salary / 1000
    pool['projected_fpts'] = pool['salary'] / 1000.0

    try:
        from optimizer import NHLLineupOptimizer
        optimizer = NHLLineupOptimizer()
        lineups = optimizer.optimize_lineup(pool, n_lineups=1, randomness=0)
        if not lineups:
            return None

        # Convert DataFrame lineup to list of dicts
        best = lineups[0]
        players = []
        for _, row in best.iterrows():
            players.append({
                'name': row['name'],
                'team': row['team'],
                'salary': row['salary'],
                'projected_fpts': row.get('projected_fpts', row['salary'] / 1000.0),
            })
        return players
    except Exception as e:
        print(f"    Salary-rank optimizer error: {e}")
        return None


# ==============================================================================
# METHOD A2: Salary-Rank with Confirmed Players Only
# ==============================================================================

def run_salary_confirmed(date_str: str, conn, actuals: pd.DataFrame, seed: int = MASTER_SEED) -> Optional[List[Dict]]:
    """
    Salary-rank optimizer but ONLY using confirmed players (those in game logs).
    Simulates pre-lock lineup confirmation via DailyFaceoff.
    """
    np.random.seed(seed)

    pool = load_dk_pool(date_str, conn)
    if pool.empty:
        return None

    # Build keys for pool and actuals
    pool['key'] = pool['name'].str.lower().str.strip() + '_' + pool['team'].apply(lambda t: normalize_team(str(t)))
    actual_keys = set(actuals['_key'].tolist())

    # Filter to confirmed players only
    pool_confirmed = pool[pool['key'].isin(actual_keys)].copy()

    n_g = (pool_confirmed['position'] == 'G').sum()
    n_sk = (pool_confirmed['position'] != 'G').sum()
    if n_g < 1 or n_sk < 8:
        return None

    pool_confirmed['projected_fpts'] = pool_confirmed['salary'] / 1000.0
    pool_confirmed = pool_confirmed.drop(columns=['key'], errors='ignore')

    try:
        from optimizer import NHLLineupOptimizer
        optimizer = NHLLineupOptimizer()
        lineups = optimizer.optimize_lineup(pool_confirmed, n_lineups=1, randomness=0)
        if not lineups:
            return None

        best = lineups[0]
        players = []
        for _, row in best.iterrows():
            players.append({
                'name': row['name'],
                'team': row['team'],
                'salary': row['salary'],
                'projected_fpts': row.get('projected_fpts', row['salary'] / 1000.0),
            })
        return players
    except Exception as e:
        print(f"    Salary-confirmed optimizer error: {e}")
        return None


# ==============================================================================
# METHOD B: SE Optimizer (Daily Projections CSVs)
# ==============================================================================

def get_best_csv(date_str: str) -> Optional[Path]:
    """Find best daily projection CSV for a date."""
    parts = date_str.split('-')
    prefix = f"{parts[1]}_{parts[2]}_{parts[0][2:]}"
    pattern = str(PROJECT_DIR / "daily_projections" / f"{prefix}NHLprojections_*.csv")
    files = sorted(glob.glob(pattern))
    candidates = []
    for f in files:
        if '_lineups' in f or '_simulator' in f:
            continue
        try:
            df = pd.read_csv(f, nrows=2)
            has_dk = 'dk_id' in df.columns or 'player_id' in df.columns
            n_rows = len(pd.read_csv(f))
            candidates.append((f, has_dk, n_rows))
        except Exception:
            continue
    candidates.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    return Path(candidates[0][0]) if candidates else None


def run_se_optimizer(date_str: str, seed: int = MASTER_SEED) -> Optional[List[Dict]]:
    """Run SE optimizer from daily projection CSV."""
    np.random.seed(seed)
    csv_path = get_best_csv(date_str)
    if not csv_path:
        return None

    try:
        pool = pd.read_csv(csv_path, encoding='utf-8-sig')
        pool = pool.dropna(subset=['salary', 'projected_fpts'])
        pool['salary'] = pd.to_numeric(pool['salary'], errors='coerce')
        pool['projected_fpts'] = pd.to_numeric(pool['projected_fpts'], errors='coerce')
        pool = pool[pool['salary'] > 0].copy()

        if (pool['position'] == 'G').sum() < 1 or (pool['position'] != 'G').sum() < 8:
            return None

        from optimizer import NHLLineupOptimizer
        optimizer = NHLLineupOptimizer()
        lineups = optimizer.optimize_lineup(pool, n_lineups=1, randomness=0.08)
        if not lineups:
            return None

        # Convert DataFrame lineup to list of dicts
        best = lineups[0]
        players = []
        for _, row in best.iterrows():
            players.append({
                'name': row['name'],
                'team': row['team'],
                'salary': row['salary'],
                'projected_fpts': row.get('projected_fpts', 0),
            })
        return players
    except Exception as e:
        print(f"    SE optimizer error: {e}")
        return None


# ==============================================================================
# MAIN BACKTEST ENGINE
# ==============================================================================

def run_full_backtest(methods: List[str] = None, verbose: bool = True) -> Dict:
    """Run comprehensive backtest over all 113 dates."""
    if methods is None:
        methods = ['salary', 'confirmed', 'se']

    conn = sqlite3.connect(str(DB_PATH))

    # Find all dates with both DK salaries AND game logs
    all_dates = pd.read_sql("""
        SELECT DISTINCT d.slate_date
        FROM dk_salaries d
        WHERE EXISTS (
            SELECT 1 FROM game_logs_skaters g WHERE g.game_date = d.slate_date
        )
        OR EXISTS (
            SELECT 1 FROM game_logs_goalies g WHERE g.game_date = d.slate_date
        )
        ORDER BY d.slate_date
    """, conn)['slate_date'].tolist()

    if verbose:
        print(f"\n{'='*80}")
        print(f"  COMPREHENSIVE NHL DFS BACKTEST ENGINE")
        print(f"  Seed: {MASTER_SEED} | Methods: {', '.join(methods)} | Dates: {len(all_dates)}")
        print(f"  Date Range: {all_dates[0] if all_dates else 'N/A'} to {all_dates[-1] if all_dates else 'N/A'}")
        print(f"{'='*80}")

    # Results storage
    results = {m: [] for m in methods}
    summary_stats = {m: [] for m in methods}

    # Process each date
    for date_idx, date_str in enumerate(all_dates, 1):
        contest = load_contest_data(date_str, conn)
        actuals = load_actuals(date_str, conn)

        if actuals.empty:
            continue

        # Count slate composition
        pool = load_dk_pool(date_str, conn)
        n_dk_players = len(pool)
        n_actual_players = len(actuals)
        n_teams_on_slate = pool['team'].nunique()
        n_teams_with_actuals = actuals['team_norm'].nunique()

        if verbose:
            print(f"\n[{date_idx:3d}/{len(all_dates)}] {date_str} | "
                  f"DK: {n_dk_players:3d} players, {n_teams_on_slate} teams | "
                  f"Actuals: {n_actual_players:3d} players, {n_teams_with_actuals} teams | "
                  f"Cash: {contest['cash_line']:>6.1f}")

        # Run each method
        for method in methods:
            t0 = time.time()

            if method == 'salary':
                players = run_salary_rank_optimizer(date_str, conn, seed=MASTER_SEED)
            elif method == 'confirmed':
                players = run_salary_confirmed(date_str, conn, actuals, seed=MASTER_SEED)
            elif method == 'se':
                players = run_se_optimizer(date_str, seed=MASTER_SEED)
            else:
                continue

            elapsed = time.time() - t0

            if players is None:
                if verbose:
                    print(f"    {method.upper():<6}: NO LINEUP")
                results[method].append({
                    'date': date_str,
                    'actual': 0,
                    'projected': 0,
                    'matched': 0,
                    'scratched': 0,
                    'is_cash': False,
                    'is_win': False,
                    'salary': 0,
                    'valid': False,
                    'n_dk_players': n_dk_players,
                    'n_actual_players': n_actual_players,
                    'n_teams_on_slate': n_teams_on_slate,
                    'n_teams_with_actuals': n_teams_with_actuals,
                    'cash_line': contest['cash_line'],
                })
                continue

            # Score the lineup
            actual_total, n_matched, n_scratched = score_lineup(players, actuals)
            projected_total = sum(p.get('projected_fpts', 0) for p in players)
            salary_total = sum(p.get('salary', 0) for p in players)
            is_cash = actual_total >= contest['cash_line'] if contest['cash_line'] > 0 else None
            is_win = actual_total >= contest['winning_score']

            status = 'CASH' if is_cash else 'miss'
            win_str = ' WIN!' if is_win else ''
            if verbose:
                print(f"    {method.upper():<6}: {actual_total:>6.1f} actual "
                      f"({n_matched}/9 matched, {n_scratched} scratched) "
                      f"proj={projected_total:.1f} → {status}{win_str} [{elapsed:.2f}s]")

            results[method].append({
                'date': date_str,
                'actual': actual_total,
                'projected': projected_total,
                'matched': n_matched,
                'scratched': n_scratched,
                'is_cash': is_cash,
                'is_win': is_win,
                'salary': salary_total,
                'valid': True,
                'n_dk_players': n_dk_players,
                'n_actual_players': n_actual_players,
                'n_teams_on_slate': n_teams_on_slate,
                'n_teams_with_actuals': n_teams_with_actuals,
                'cash_line': contest['cash_line'],
                'winning_score': contest['winning_score'],
            })

    conn.close()

    # ==============================================================================
    # SUMMARY STATISTICS
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"  BACKTEST RESULTS SUMMARY")
        print(f"{'='*80}")

    all_results = {}
    for method in methods:
        r = pd.DataFrame(results[method])
        valid = r[r['valid']]

        if valid.empty:
            if verbose:
                print(f"\n  {method.upper()}: No valid results")
            all_results[method] = None
            continue

        has_cash = valid[valid['is_cash'].notna()]
        cash_rate = has_cash['is_cash'].mean() * 100 if not has_cash.empty else 0
        win_rate = valid['is_win'].mean() * 100
        avg_fpts = valid['actual'].mean()
        avg_proj = valid['projected'].mean()
        avg_salary = valid['salary'].mean()
        avg_matched = valid['matched'].mean()
        avg_scratched = valid['scratched'].mean()

        # Wilson confidence interval (95%)
        n = len(has_cash)
        p = cash_rate / 100
        if n > 0:
            z = 1.96
            denom = 1 + z**2/n
            center = (p + z**2/(2*n)) / denom
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            ci_low = max(0, center - margin) * 100
            ci_high = min(1, center + margin) * 100
        else:
            ci_low, ci_high = 0, 0

        summary = {
            'method': method,
            'n_valid': len(valid),
            'n_cash': int(has_cash['is_cash'].sum()) if not has_cash.empty else 0,
            'cash_rate': cash_rate,
            'cash_rate_ci_low': ci_low,
            'cash_rate_ci_high': ci_high,
            'win_rate': win_rate,
            'n_wins': int(valid['is_win'].sum()),
            'avg_fpts': avg_fpts,
            'avg_proj': avg_proj,
            'proj_error': avg_proj - avg_fpts,
            'avg_salary': avg_salary,
            'avg_matched': avg_matched,
            'avg_scratched': avg_scratched,
        }

        all_results[method] = summary

        if verbose:
            print(f"\n  {method.upper()} ({len(valid)} dates)")
            print(f"  {'─'*50}")
            print(f"  Cash rate:    {cash_rate:6.1f}% ({summary['n_cash']}/{n})")
            print(f"    95% CI:     [{ci_low:6.1f}% — {ci_high:6.1f}%]")
            print(f"  Win rate:     {win_rate:6.1f}% ({summary['n_wins']}/{len(valid)})")
            print(f"  Avg FPTS:     {avg_fpts:6.1f}")
            print(f"  Avg projected:{avg_proj:6.1f}")
            print(f"  Proj error:   {avg_proj - avg_fpts:+6.1f} FPTS")
            print(f"  Avg salary:   ${avg_salary:>9,.0f}")
            print(f"  Avg matched:  {avg_matched:6.1f}/9 players")
            print(f"  Avg scratched:{avg_scratched:6.1f} players")

        # Per-date detail table
        if verbose:
            print(f"\n  {'Date':<12} {'Actual':>7} {'Proj':>7} {'Match':>6} {'Scratchd':>8} {'Cash':>7}")
            print(f"  {'─'*60}")
            for _, row in valid.iterrows():
                print(f"  {row['date']:<12} {row['actual']:>7.1f} {row['projected']:>7.1f} "
                      f"{row['matched']:>6.0f} {row['scratched']:>8.0f} {row.get('cash_line', 0):>7.1f}")

    # ==============================================================================
    # HEAD-TO-HEAD COMPARISON
    # ==============================================================================
    if len(methods) > 1 and verbose:
        method_results_dfs = []
        valid_methods = []
        for m in methods:
            if all_results[m] is not None:
                r = pd.DataFrame(results[m])
                valid = r[r['valid']]
                if not valid.empty:
                    method_results_dfs.append(valid[['date', 'actual']])
                    valid_methods.append(m)

        if len(method_results_dfs) > 1:
            comparison = method_results_dfs[0].copy()
            comparison.columns = ['date', 'actual_' + valid_methods[0]]
            for i, valid_m in enumerate(valid_methods[1:], 1):
                comparison = comparison.merge(
                    method_results_dfs[i].rename(columns={'actual': 'actual_' + valid_m}),
                    on='date',
                    how='inner'
                )

            if not comparison.empty:
                print(f"\n  HEAD-TO-HEAD ({len(comparison)} dates)")
                print(f"  {'─'*70}")

                col_names = valid_methods
                col_actuals = [f'actual_{m}' for m in col_names]

                # Pairwise wins
                for i, m1 in enumerate(col_names):
                    for m2 in col_names[i+1:]:
                        wins_m1 = (comparison[f'actual_{m1}'] > comparison[f'actual_{m2}']).sum()
                        wins_m2 = (comparison[f'actual_{m2}'] > comparison[f'actual_{m1}']).sum()
                        ties = (comparison[f'actual_{m1}'] == comparison[f'actual_{m2}']).sum()
                        avg_m1 = comparison[f'actual_{m1}'].mean()
                        avg_m2 = comparison[f'actual_{m2}'].mean()
                        edge = avg_m1 - avg_m2
                        print(f"  {m1.upper()} vs {m2.upper()}: {wins_m1}W-{wins_m2}L-{ties}T | "
                              f"Avg: {avg_m1:.1f} vs {avg_m2:.1f} ({edge:+.1f} FPTS/slate)")

    # ==============================================================================
    # SAVE PER-DATE CSV
    # ==============================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for method in methods:
        if all_results[method] is None:
            continue

        r = pd.DataFrame(results[method])
        output_csv = BACKTESTS_DIR / f"backtest_{method}_{timestamp}.csv"
        r.to_csv(str(output_csv), index=False)

        if verbose:
            print(f"\n  Per-date details saved to: {output_csv}")

    if verbose:
        print(f"\n{'='*80}\n")

    return {
        'summary': all_results,
        'detailed': results,
        'timestamp': timestamp,
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive NHL DFS Backtest Engine')
    parser.add_argument('--method', choices=['salary', 'confirmed', 'se', 'all', 'both'], default='all',
                        help='Which strategy to test')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()

    if args.method == 'all':
        methods = ['salary', 'confirmed', 'se']
    elif args.method == 'both':
        methods = ['salary', 'se']
    else:
        methods = [args.method]

    results = run_full_backtest(methods=methods, verbose=not args.quiet)
