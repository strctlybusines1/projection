#!/usr/bin/env python3
"""
SE Lineup Backtest
===================

For each historical slate:
1. Load the projection CSV (last/best run for that date)
2. Run the full pipeline (blend + all models)
3. Generate 40 SE candidate lineups via optimizer
4. Score each lineup against actual DK FPTS
5. Report: best lineup, worst lineup, SE-selected lineup, random baseline

Usage:
    python backtest_se_lineups.py                    # All available dates
    python backtest_se_lineups.py --date 2026-02-05  # Single date
    python backtest_se_lineups.py --candidates 60    # More candidates
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import glob
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "data" / "nhl_dfs_history.db"


def get_best_projection_csv(date_str: str) -> Path:
    """
    Find the best (latest, with dk_id) projection CSV for a date.
    Date format: 2026-01-29 -> 01_29_26
    """
    parts = date_str.split('-')
    date_prefix = f"{parts[1]}_{parts[2]}_{parts[0][2:]}"  # MM_DD_YY

    pattern = str(PROJECT_DIR / "daily_projections" / f"{date_prefix}NHLprojections_*.csv")
    files = sorted(glob.glob(pattern))

    # Filter out lineup files and find ones with dk_id
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

    # Prefer files with dk_id, then latest (most rows usually = most complete)
    candidates.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)

    if candidates:
        return Path(candidates[0][0])
    return None


def get_actuals_from_db(date_str: str) -> pd.DataFrame:
    """Pull actual DK FPTS from game logs for a date."""
    conn = sqlite3.connect(str(DB_PATH))

    skaters = pd.read_sql_query("""
        SELECT player_name as name, team, position, dk_fpts as actual_fpts
        FROM game_logs_skaters WHERE game_date = ?
    """, conn, params=(date_str,))

    goalies = pd.read_sql_query("""
        SELECT player_name as name, team, 'G' as position, dk_fpts as actual_fpts
        FROM game_logs_goalies WHERE game_date = ?
    """, conn, params=(date_str,))

    conn.close()

    actuals = pd.concat([skaters, goalies], ignore_index=True)
    if not actuals.empty:
        actuals['_key'] = actuals['name'].str.lower().str.strip() + '_' + actuals['team'].str.lower().str.strip()
    return actuals


def load_and_prepare_pool(csv_path: Path) -> pd.DataFrame:
    """Load projection CSV and prepare for optimizer."""
    pool = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Ensure required columns
    required = ['name', 'team', 'position', 'salary', 'projected_fpts']
    missing = [c for c in required if c not in pool.columns]
    if missing:
        print(f"    Missing columns: {missing}")
        return pd.DataFrame()

    # Clean up
    pool = pool.dropna(subset=['salary', 'projected_fpts'])
    pool['salary'] = pd.to_numeric(pool['salary'], errors='coerce')
    pool['projected_fpts'] = pd.to_numeric(pool['projected_fpts'], errors='coerce')
    pool = pool[pool['salary'] > 0].copy()

    # Ensure we have goalies and skaters
    n_g = (pool['position'] == 'G').sum()
    n_sk = (pool['position'] != 'G').sum()
    if n_g < 1 or n_sk < 8:
        print(f"    Insufficient pool: {n_g} goalies, {n_sk} skaters")
        return pd.DataFrame()

    return pool


def generate_se_candidates(pool: pd.DataFrame, n_candidates: int = 40) -> List[pd.DataFrame]:
    """Generate N candidate lineups with randomness."""
    from optimizer import NHLLineupOptimizer

    optimizer = NHLLineupOptimizer()
    candidates = optimizer.optimize_lineup(
        pool,
        n_lineups=n_candidates,
        randomness=0.08,
    )
    return candidates if candidates else []


def score_lineup_actual(lineup: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """Score a lineup against actual results. Returns total actual FPTS."""
    lineup = lineup.copy()
    lineup['_key'] = lineup['name'].str.lower().str.strip() + '_' + lineup['team'].str.lower().str.strip()

    merged = lineup.merge(actuals[['_key', 'actual_fpts']], on='_key', how='left')
    total_actual = merged['actual_fpts'].sum()
    matched = merged['actual_fpts'].notna().sum()

    return total_actual, matched, len(lineup)


def run_se_backtest_date(date_str: str, n_candidates: int = 40, verbose: bool = True):
    """Run full SE backtest for a single date."""
    if verbose:
        print(f"\n  ── SE Lineup Backtest: {date_str} ──")

    # 1. Find projection CSV
    csv_path = get_best_projection_csv(date_str)
    if not csv_path:
        if verbose:
            print(f"    No projection CSV found for {date_str}")
        return None

    if verbose:
        print(f"    Projection: {csv_path.name}")

    # 2. Load pool
    pool = load_and_prepare_pool(csv_path)
    if pool.empty:
        return None

    if verbose:
        n_sk = (pool['position'] != 'G').sum()
        n_g = (pool['position'] == 'G').sum()
        print(f"    Pool: {n_sk} skaters, {n_g} goalies")

    # 3. Get actuals
    actuals = get_actuals_from_db(date_str)
    if actuals.empty:
        if verbose:
            print(f"    No actuals found for {date_str}")
        return None

    if verbose:
        print(f"    Actuals: {len(actuals)} players")

        # Coverage check
        pool_key = pool['name'].str.lower().str.strip() + '_' + pool['team'].str.lower().str.strip()
        actuals_keys = set(actuals['_key']) if '_key' in actuals.columns else set()
        matched_count = pool_key.isin(actuals_keys).sum()
        print(f"    Coverage: {matched_count}/{len(pool)} pool players have actuals ({matched_count/len(pool)*100:.0f}%)")

        if matched_count < 20:
            print(f"    ⚠ Low coverage — results may not be reliable")
            print(f"      (Need 32-team game logs for full accuracy)")

    # 4. Generate candidates
    try:
        candidates = generate_se_candidates(pool, n_candidates)
    except Exception as e:
        if verbose:
            print(f"    Optimizer failed: {e}")
        return None

    if not candidates:
        if verbose:
            print(f"    No candidates generated")
        return None

    if verbose:
        print(f"    Generated: {len(candidates)} candidate lineups")

    # 5. Score each candidate against actuals
    results = []
    for i, lineup in enumerate(candidates):
        actual_pts, matched, total = score_lineup_actual(lineup, actuals)
        projected_pts = lineup['projected_fpts'].sum()
        results.append({
            'lineup_idx': i,
            'projected_fpts': projected_pts,
            'actual_fpts': actual_pts,
            'matched': matched,
            'total_players': total,
            'salary': lineup['salary'].sum(),
        })

    results_df = pd.DataFrame(results)

    # 6. SE selector scoring
    try:
        from single_entry import SingleEntrySelector
        # Build team totals from Vegas if available
        team_totals = {}
        try:
            from historical_odds import get_odds_for_date
            odds = get_odds_for_date(date_str)
            if not odds.empty:
                for _, r in odds.iterrows():
                    team_totals[r['Team']] = r['TeamGoal']
        except Exception:
            pass

        selector = SingleEntrySelector(pool, team_totals=team_totals)
        se_scores = []
        for i, lineup in enumerate(candidates):
            try:
                score = selector.score_lineup(lineup)
                se_scores.append({'lineup_idx': i, 'se_score': score.get('total', 0), **score})
            except Exception:
                se_scores.append({'lineup_idx': i, 'se_score': 0})

        se_df = pd.DataFrame(se_scores)
        results_df = results_df.merge(se_df[['lineup_idx', 'se_score']], on='lineup_idx', how='left')
    except Exception as e:
        if verbose:
            print(f"    SE scoring failed: {e}")
        results_df['se_score'] = results_df['projected_fpts']

    # 7. Analysis
    best_actual = results_df.loc[results_df['actual_fpts'].idxmax()]
    worst_actual = results_df.loc[results_df['actual_fpts'].idxmin()]
    se_pick = results_df.loc[results_df['se_score'].idxmax()]
    proj_pick = results_df.loc[results_df['projected_fpts'].idxmax()]
    avg_actual = results_df['actual_fpts'].mean()
    median_actual = results_df['actual_fpts'].median()

    # SE rank: where does the SE-selected lineup rank by actual FPTS?
    results_df['actual_rank'] = results_df['actual_fpts'].rank(ascending=False)
    se_actual_rank = results_df.loc[results_df['se_score'].idxmax(), 'actual_rank']
    proj_actual_rank = results_df.loc[results_df['projected_fpts'].idxmax(), 'actual_rank']

    if verbose:
        print(f"\n    {'Lineup':<18} {'Projected':>9} {'Actual':>8} {'Salary':>8} {'Rank':>6}")
        print(f"    {'-'*52}")
        print(f"    {'Best (actual)':<18} {best_actual['projected_fpts']:>9.1f} {best_actual['actual_fpts']:>8.1f} "
              f"${best_actual['salary']:>7,.0f} {'1':>6}")
        print(f"    {'SE Selected':<18} {se_pick['projected_fpts']:>9.1f} {se_pick['actual_fpts']:>8.1f} "
              f"${se_pick['salary']:>7,.0f} {se_actual_rank:>6.0f}")
        print(f"    {'Max Projection':<18} {proj_pick['projected_fpts']:>9.1f} {proj_pick['actual_fpts']:>8.1f} "
              f"${proj_pick['salary']:>7,.0f} {proj_actual_rank:>6.0f}")
        print(f"    {'Average':<18} {'':>9} {avg_actual:>8.1f}")
        print(f"    {'Worst':<18} {worst_actual['projected_fpts']:>9.1f} {worst_actual['actual_fpts']:>8.1f} "
              f"${worst_actual['salary']:>7,.0f} {len(candidates):>6}")

        # Show the SE-selected lineup
        se_lineup = candidates[int(se_pick['lineup_idx'])]
        se_lineup_scored = se_lineup.merge(
            actuals[['_key', 'actual_fpts']],
            left_on=se_lineup['name'].str.lower().str.strip() + '_' + se_lineup['team'].str.lower().str.strip(),
            right_on='_key', how='left'
        )
        print(f"\n    SE Lineup:")
        for _, p in se_lineup_scored.iterrows():
            act = p.get('actual_fpts', 0)
            act_str = f"{act:.1f}" if pd.notna(act) else "N/A"
            print(f"      {p['position']:<3} {p['name']:<22} {p['team']:<4} "
                  f"${p['salary']:>6,} proj={p['projected_fpts']:>5.1f} act={act_str:>5}")

    return {
        'date': date_str,
        'n_candidates': len(candidates),
        'best_actual': best_actual['actual_fpts'],
        'worst_actual': worst_actual['actual_fpts'],
        'avg_actual': avg_actual,
        'median_actual': median_actual,
        'se_actual': se_pick['actual_fpts'],
        'se_rank': se_actual_rank,
        'se_projected': se_pick['projected_fpts'],
        'proj_pick_actual': proj_pick['actual_fpts'],
        'proj_rank': proj_actual_rank,
        'se_vs_avg': se_pick['actual_fpts'] - avg_actual,
        'se_vs_proj': se_pick['actual_fpts'] - proj_pick['actual_fpts'],
        'pool_size': len(pool),
        'actuals_size': len(actuals),
        'results_df': results_df,
    }


def run_full_backtest(dates: List[str] = None, n_candidates: int = 40):
    """Run SE backtest across multiple dates."""

    if dates is None:
        # All dates with dk_id projections and game log actuals
        dates = ['2026-01-29', '2026-01-31', '2026-02-01', '2026-02-02',
                 '2026-02-03', '2026-02-04', '2026-02-05']

    print(f"\n{'='*60}")
    print(f"  SE LINEUP BACKTEST — {len(dates)} slates, {n_candidates} candidates each")
    print(f"{'='*60}")

    all_results = []
    for date_str in dates:
        result = run_se_backtest_date(date_str, n_candidates=n_candidates)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n  No valid results.")
        return

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {len(all_results)} slates")
    print(f"{'='*60}")
    print(f"\n  {'Date':<12} {'Best':>6} {'SE':>6} {'Proj':>6} {'Avg':>6} {'SE Rank':>7} {'SE vs Avg':>9}")
    print(f"  {'-'*58}")

    total_se = 0
    total_avg = 0
    total_best = 0
    se_ranks = []

    for r in all_results:
        print(f"  {r['date']:<12} {r['best_actual']:>6.1f} {r['se_actual']:>6.1f} "
              f"{r['proj_pick_actual']:>6.1f} {r['avg_actual']:>6.1f} "
              f"{r['se_rank']:>5.0f}/{r['n_candidates']:<2} {r['se_vs_avg']:>+8.1f}")
        total_se += r['se_actual']
        total_avg += r['avg_actual']
        total_best += r['best_actual']
        se_ranks.append(r['se_rank'] / r['n_candidates'])

    print(f"  {'-'*58}")
    n = len(all_results)
    print(f"  {'Average':<12} {total_best/n:>6.1f} {total_se/n:>6.1f} "
          f"{'':>6} {total_avg/n:>6.1f} "
          f"{'':>7} {(total_se-total_avg)/n:>+8.1f}")

    avg_pctile = 1.0 - np.mean(se_ranks)
    print(f"\n  SE selection percentile: {avg_pctile:.1%} (1.0 = always picks best)")
    print(f"  SE vs random advantage: {(total_se-total_avg)/n:+.1f} FPTS/slate")
    print(f"  SE captures {(total_se/total_best)*100:.1f}% of best possible")

    # Export results
    summary_rows = [{k: v for k, v in r.items() if k != 'results_df'} for r in all_results]
    out_path = PROJECT_DIR / 'backtests' / 'se_lineup_backtest.csv'
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    # Export all candidate lineups to JSON
    export_path = PROJECT_DIR / 'backtests' / 'se_lineup_candidates.json'
    export_data = {}
    for r in all_results:
        rdf = r['results_df']
        export_data[r['date']] = rdf.drop(columns=['actual_rank'], errors='ignore').to_dict('records')
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    print(f"  Candidate details saved to {export_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SE Lineup Backtest')
    parser.add_argument('--date', type=str, help='Single date to backtest')
    parser.add_argument('--candidates', type=int, default=40, help='Number of candidate lineups')
    parser.add_argument('--dates', nargs='+', help='Specific dates to backtest')
    args = parser.parse_args()

    if args.date:
        run_se_backtest_date(args.date, n_candidates=args.candidates)
    elif args.dates:
        run_full_backtest(dates=args.dates, n_candidates=args.candidates)
    else:
        run_full_backtest(n_candidates=args.candidates)
