#!/usr/bin/env python3
"""
Reproducible Backtest Harness — The Clean Room
================================================

Jim Simons rule #1: "You can't manage what you can't measure."

This harness provides a single, controlled environment to compare:
  A) MC Lineup Builder (ensemble: MDN v3 + Transformer)
  B) SE Optimizer (daily projection CSVs from full blend pipeline)
  C) Random baseline (random valid lineups)

All scored identically against actual DK FPTS using name+team fuzzy matching.
Fixed random seeds for reproducibility. Statistical significance testing.

Usage:
    python backtest_harness.py                    # Full comparison
    python backtest_harness.py --method mc        # MC builder only
    python backtest_harness.py --method se        # SE optimizer only
    python backtest_harness.py --date 2026-02-05  # Single date
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

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "data" / "nhl_dfs_history.db"

# Fixed seed for reproducibility — Simons would insist
MASTER_SEED = 2026


# ==============================================================================
# TEAM ABBREVIATION NORMALIZATION — centralized module
# ==============================================================================

from team_normalize import normalize_team_lower as normalize_team


# ==============================================================================
# SCORING ENGINE — identical for all methods
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
    # Normalize team abbreviations to NHL standard
    actuals['team_norm'] = actuals['team'].apply(normalize_team)
    # Build composite key: name_team (lowercase, stripped, normalized)
    actuals['_key'] = actuals['name'].str.lower().str.strip() + '_' + actuals['team_norm']
    return actuals


def score_lineup(players: List[Dict], actuals: pd.DataFrame) -> Tuple[float, int]:
    """
    Score a lineup against actuals using fuzzy name+team matching.
    Returns (total_actual_fpts, n_matched).
    """
    total = 0.0
    matched = 0
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

    return total, matched


def load_contest_data(date_str: str, conn) -> Dict:
    """Load cash line and winning score for a date."""
    row = conn.execute("""
        SELECT MIN(CASE WHEN n_cashed > 0 THEN score END) as cash_line,
               MAX(score) as winning_score
        FROM contest_results WHERE slate_date = ?
    """, (date_str,)).fetchone()
    return {
        'cash_line': row[0] if row and row[0] else 0,
        'winning_score': row[1] if row and row[1] else 999,
    }


# ==============================================================================
# METHOD A: MC Lineup Builder (ensemble projections)
# ==============================================================================

def run_mc_builder(date_str: str, seed: int = MASTER_SEED) -> Optional[List[Dict]]:
    """Run MC lineup builder for a date. Returns list of player dicts."""
    np.random.seed(seed)
    try:
        from lineup_builder import build_player_pool, MonteCarloLineupBuilder
        pool = build_player_pool(date_str, use_ensemble=True)
        if pool.empty:
            return None
        builder = MonteCarloLineupBuilder(pool, n_iterations=500)
        lineups, freq = builder.run(n_lineups=20)
        if not lineups:
            return None
        return lineups[0]['players']  # Best lineup
    except Exception as e:
        print(f"    MC builder error: {e}")
        return None


# ==============================================================================
# METHOD B: SE Optimizer (daily projection CSVs)
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
    """Run SE optimizer from daily projection CSV. Returns list of player dicts."""
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
        lineups = optimizer.optimize_lineup(pool, n_lineups=20, randomness=0.08)
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
                'projected_fpts': row['projected_fpts'],
            })
        return players
    except Exception as e:
        print(f"    SE optimizer error: {e}")
        return None


# ==============================================================================
# MAIN HARNESS
# ==============================================================================

def run_harness(methods: List[str] = None, dates: List[str] = None, verbose: bool = True):
    """Run the full reproducible backtest harness."""
    if methods is None:
        methods = ['mc', 'se']

    conn = sqlite3.connect(str(DB_PATH))

    # Find all dates with full data
    if dates is None:
        all_dates = pd.read_sql("""
            SELECT DISTINCT d.slate_date
            FROM dk_salaries d
            WHERE d.slate_date >= '2026-01-20' AND d.slate_date <= '2026-02-05'
            GROUP BY d.slate_date
            ORDER BY d.slate_date
        """, conn)['slate_date'].tolist()
    else:
        all_dates = dates

    # Filter to dates with actuals
    valid_dates = []
    for d in all_dates:
        n_act = conn.execute("SELECT COUNT(*) FROM game_logs_skaters WHERE game_date=?", (d,)).fetchone()[0]
        if n_act > 0:
            valid_dates.append(d)

    print(f"\n{'='*78}")
    print(f"  REPRODUCIBLE BACKTEST HARNESS")
    print(f"  Seed: {MASTER_SEED} | Methods: {', '.join(methods)} | Dates: {len(valid_dates)}")
    print(f"{'='*78}")

    # Results storage
    results = {m: [] for m in methods}

    for date_str in valid_dates:
        contest = load_contest_data(date_str, conn)
        actuals = load_actuals(date_str, conn)

        if actuals.empty:
            continue

        print(f"\n  {date_str} | cash={contest['cash_line']:.1f} | win={contest['winning_score']:.1f}")

        for method in methods:
            t0 = time.time()

            if method == 'mc':
                players = run_mc_builder(date_str, seed=MASTER_SEED)
            elif method == 'se':
                players = run_se_optimizer(date_str, seed=MASTER_SEED)
            else:
                continue

            elapsed = time.time() - t0

            if players is None:
                print(f"    {method.upper():<4}: NO LINEUP")
                results[method].append({
                    'date': date_str, 'actual': 0, 'projected': 0,
                    'matched': 0, 'is_cash': False, 'is_win': False,
                    'salary': 0, 'valid': False,
                })
                continue

            actual_total, n_matched = score_lineup(players, actuals)
            projected_total = sum(p.get('projected_fpts', 0) for p in players)
            salary_total = sum(p.get('salary', 0) for p in players)
            is_cash = actual_total >= contest['cash_line'] if contest['cash_line'] > 0 else None
            is_win = actual_total >= contest['winning_score']

            status = 'CASH' if is_cash else 'miss'
            win_str = ' WIN!' if is_win else ''
            print(f"    {method.upper():<4}: {actual_total:>6.1f} actual ({n_matched}/9 matched) "
                  f"proj={projected_total:.1f} ${salary_total:,} → {status}{win_str} [{elapsed:.1f}s]")

            results[method].append({
                'date': date_str, 'actual': actual_total, 'projected': projected_total,
                'matched': n_matched, 'is_cash': is_cash, 'is_win': is_win,
                'salary': salary_total, 'valid': True,
                'cash_line': contest['cash_line'], 'winning_score': contest['winning_score'],
            })

    conn.close()

    # ==============================================================================
    # SUMMARY
    # ==============================================================================
    print(f"\n{'='*78}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*78}")

    for method in methods:
        r = pd.DataFrame(results[method])
        valid = r[r['valid']]
        if valid.empty:
            print(f"\n  {method.upper()}: No valid results")
            continue

        has_cash = valid[valid['is_cash'].notna()]
        cash_rate = has_cash['is_cash'].mean() * 100 if not has_cash.empty else 0
        win_rate = valid['is_win'].mean() * 100
        avg_fpts = valid['actual'].mean()
        avg_proj = valid['projected'].mean()
        avg_salary = valid['salary'].mean()
        avg_matched = valid['matched'].mean()

        # Statistical confidence: cash rate 95% CI (Wilson interval)
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

        print(f"\n  {method.upper()} ({len(valid)} dates)")
        print(f"  {'─'*40}")
        print(f"  Cash rate:    {cash_rate:.1f}% ({int(has_cash['is_cash'].sum())}/{n})")
        print(f"    95% CI:     [{ci_low:.1f}% — {ci_high:.1f}%]")
        print(f"  Win rate:     {win_rate:.1f}% ({int(valid['is_win'].sum())}/{len(valid)})")
        print(f"  Avg FPTS:     {avg_fpts:.1f}")
        print(f"  Avg projected:{avg_proj:.1f}")
        print(f"  Proj error:   {avg_proj - avg_fpts:+.1f}")
        print(f"  Avg salary:   ${avg_salary:,.0f}")
        print(f"  Avg matched:  {avg_matched:.1f}/9")

        # Per-date detail
        print(f"\n  {'Date':<12} {'Actual':>7} {'Proj':>7} {'Cash':>7} {'Status':>8}")
        print(f"  {'─'*45}")
        for _, row in valid.iterrows():
            status = 'CASH' if row['is_cash'] else 'miss'
            if row['is_win']:
                status = 'WIN!'
            print(f"  {row['date']:<12} {row['actual']:>7.1f} {row['projected']:>7.1f} "
                  f"{row.get('cash_line', 0):>7.1f} {status:>8}")

    # Head-to-head if both methods tested
    if 'mc' in methods and 'se' in methods:
        mc_r = pd.DataFrame(results['mc'])
        se_r = pd.DataFrame(results['se'])
        both = mc_r[mc_r['valid']].merge(se_r[se_r['valid']], on='date', suffixes=('_mc', '_se'))
        if not both.empty:
            print(f"\n  HEAD-TO-HEAD ({len(both)} dates)")
            print(f"  {'─'*50}")
            mc_wins = (both['actual_mc'] > both['actual_se']).sum()
            se_wins = (both['actual_se'] > both['actual_mc']).sum()
            ties = (both['actual_mc'] == both['actual_se']).sum()
            print(f"  MC wins: {mc_wins} | SE wins: {se_wins} | Ties: {ties}")
            print(f"  MC avg:  {both['actual_mc'].mean():.1f} | SE avg: {both['actual_se'].mean():.1f}")
            diff = both['actual_mc'] - both['actual_se']
            print(f"  MC edge: {diff.mean():+.1f} FPTS/slate (std: {diff.std():.1f})")

            # Paired t-test
            from scipy import stats
            try:
                t_stat, p_val = stats.ttest_rel(both['actual_mc'], both['actual_se'])
                print(f"  Paired t-test: t={t_stat:.2f}, p={p_val:.3f}")
                sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
                print(f"  Difference is {sig} at 95% confidence")
            except Exception:
                pass

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reproducible Backtest Harness')
    parser.add_argument('--method', choices=['mc', 'se', 'both'], default='both')
    parser.add_argument('--date', type=str, help='Single date')
    parser.add_argument('--dates', nargs='+', help='Specific dates')
    args = parser.parse_args()

    methods = ['mc', 'se'] if args.method == 'both' else [args.method]
    dates = [args.date] if args.date else args.dates

    run_harness(methods=methods, dates=dates)
