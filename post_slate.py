#!/usr/bin/env python3
"""
Auto Post-Slate Backtest — Run after games finish to grade your projections.

Ties together: actual_scores.py → compare to projections → log to history_db.

Usage:
    python post_slate.py                      # Auto-detect last slate date
    python post_slate.py --date 2026-02-25    # Specific date
    python post_slate.py --date 2026-02-25 --no-db   # Skip DB logging
    python post_slate.py --backfill           # Backfill all dates with projection CSVs

What it does:
    1. Finds your latest projection CSV for the date
    2. Fetches actual box scores from NHL API
    3. Matches players (name + team fuzzy matching)
    4. Prints accuracy report (MAE, bias, correlation, biggest misses)
    5. Saves details CSV to backtests/
    6. Logs results to historical SQLite database
"""

import argparse
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np

# Project imports
from actual_scores import fetch_actual_scores
from history_db import get_connection, ingest_backtest_csv
from config import DAILY_PROJECTIONS_DIR, BACKTESTS_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
PROJ_DIR = PROJECT_ROOT / DAILY_PROJECTIONS_DIR
BACKTEST_DIR = PROJECT_ROOT / BACKTESTS_DIR


# ================================================================
#  Find Projection File
# ================================================================

def find_projection_csv(date: str) -> Optional[Path]:
    """Find the latest projection CSV for a given date."""
    if not PROJ_DIR.exists():
        return None

    dt = datetime.strptime(date, '%Y-%m-%d')
    prefixes = [
        f"{dt.month:02d}_{dt.day:02d}_{dt.strftime('%y')}",
        f"{dt.month}_{dt.day}_{dt.strftime('%y')}",
    ]

    matches = []
    for f in sorted(PROJ_DIR.glob('*NHLprojections_*.csv')):
        if '_lineups' in f.name:
            continue
        for prefix in prefixes:
            if f.name.startswith(prefix):
                matches.append(f)
                break

    if not matches:
        return None

    # Return the latest one (by filename timestamp)
    return sorted(matches)[-1]


def discover_dates_with_projections() -> List[str]:
    """Find all dates that have projection CSVs."""
    if not PROJ_DIR.exists():
        return []

    dates = set()
    for f in PROJ_DIR.glob('*NHLprojections_*.csv'):
        if '_lineups' in f.name:
            continue
        match = re.match(r'(\d{2})_(\d{2})_(\d{2})NHL', f.name)
        if match:
            m, d, y = match.groups()
            dates.add(f"20{y}-{m}-{d}")

    return sorted(dates)


def guess_last_slate_date() -> Optional[str]:
    """Guess the most recent slate date (yesterday or today if games are done)."""
    dates = discover_dates_with_projections()
    if not dates:
        return None
    return dates[-1]


# ================================================================
#  Name Matching
# ================================================================

def _clean_for_match(name: str) -> str:
    """Normalize name for matching."""
    import unicodedata
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    s = s.lower().strip()
    s = re.sub(r'[^a-z\s]', '', s)
    return s


def _last_name(name: str) -> str:
    parts = name.strip().split()
    return parts[-1].lower() if parts else ''


def match_players(actuals: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    """
    Match actual scores to projections using name + team.

    Strategy:
        1. Exact name match
        2. Last name + team match
        3. Cleaned name match (accent removal)
    """
    proj = projections.copy()
    act = actuals.copy()

    # Ensure consistent column names
    if 'projected_fpts' not in proj.columns:
        return pd.DataFrame()

    # Build lookup dicts
    proj['_clean'] = proj['name'].apply(_clean_for_match)
    proj['_last'] = proj['name'].apply(_last_name)
    act['_clean'] = act['name'].apply(_clean_for_match)
    act['_last'] = act['name'].apply(_last_name)

    # Method 1: exact name merge
    merged = act.merge(
        proj[['name', 'team', 'projected_fpts', 'salary', 'position']].drop_duplicates('name'),
        on='name', how='inner', suffixes=('', '_proj')
    )

    matched_names = set(merged['name'].tolist())

    # Method 2: last name + team for unmatched
    unmatched_act = act[~act['name'].isin(matched_names)]
    if not unmatched_act.empty:
        # For each unmatched actual, find projection with same last name + team
        for _, arow in unmatched_act.iterrows():
            candidates = proj[
                (proj['_last'] == arow['_last']) &
                (proj['team'] == arow.get('team', ''))
            ]
            if len(candidates) == 1:
                prow = candidates.iloc[0]
                row_dict = arow.to_dict()
                row_dict['projected_fpts'] = prow['projected_fpts']
                row_dict['salary'] = prow.get('salary')
                merged = pd.concat([merged, pd.DataFrame([row_dict])], ignore_index=True)
                matched_names.add(arow['name'])

    # Method 3: cleaned name for remaining
    still_unmatched = act[~act['name'].isin(matched_names)]
    if not still_unmatched.empty:
        proj_clean_map = {row['_clean']: row for _, row in proj.iterrows()}
        for _, arow in still_unmatched.iterrows():
            if arow['_clean'] in proj_clean_map:
                prow = proj_clean_map[arow['_clean']]
                row_dict = arow.to_dict()
                row_dict['projected_fpts'] = prow['projected_fpts']
                row_dict['salary'] = prow.get('salary')
                merged = pd.concat([merged, pd.DataFrame([row_dict])], ignore_index=True)

    # Calculate errors
    if not merged.empty and 'projected_fpts' in merged.columns:
        merged['error'] = merged['projected_fpts'] - merged['actual_fpts']
        merged['abs_error'] = merged['error'].abs()

    # Drop helper columns
    for col in ['_clean', '_last']:
        if col in merged.columns:
            merged = merged.drop(columns=[col])

    return merged


# ================================================================
#  Report
# ================================================================

def print_accuracy_report(merged: pd.DataFrame, date: str):
    """Print detailed accuracy report."""
    if merged.empty:
        print("  No matched players to report.")
        return

    skaters = merged[merged['player_type'] == 'skater'] if 'player_type' in merged.columns else merged[merged['position'] != 'G']
    goalies = merged[merged['player_type'] == 'goalie'] if 'player_type' in merged.columns else merged[merged['position'] == 'G']

    print(f"\n{'=' * 70}")
    print(f"  POST-SLATE BACKTEST — {date}")
    print(f"{'=' * 70}")

    for label, subset in [('SKATERS', skaters), ('GOALIES', goalies)]:
        if subset.empty:
            continue
        mae = subset['abs_error'].mean()
        bias = subset['error'].mean()
        corr = subset[['projected_fpts', 'actual_fpts']].corr().iloc[0, 1] if len(subset) > 2 else float('nan')
        print(f"\n  {label} ({len(subset)} matched):")
        print(f"    MAE:         {mae:.2f}")
        print(f"    Bias:        {bias:+.2f}")
        print(f"    Correlation: {corr:.3f}" if not np.isnan(corr) else "    Correlation: N/A")
        print(f"    Avg Proj:    {subset['projected_fpts'].mean():.2f}")
        print(f"    Avg Actual:  {subset['actual_fpts'].mean():.2f}")

    # Overall
    mae = merged['abs_error'].mean()
    bias = merged['error'].mean()
    print(f"\n  OVERALL ({len(merged)} players):")
    print(f"    MAE:  {mae:.2f}")
    print(f"    Bias: {bias:+.2f}")

    # Top misses
    print(f"\n  TOP 5 OVER-PROJECTIONS:")
    for _, r in merged.nlargest(5, 'error').iterrows():
        print(f"    {r['name']:<25} Proj: {r['projected_fpts']:5.1f}  "
              f"Actual: {r['actual_fpts']:5.1f}  Error: {r['error']:+.1f}")

    print(f"\n  TOP 5 UNDER-PROJECTIONS:")
    for _, r in merged.nsmallest(5, 'error').iterrows():
        print(f"    {r['name']:<25} Proj: {r['projected_fpts']:5.1f}  "
              f"Actual: {r['actual_fpts']:5.1f}  Error: {r['error']:+.1f}")

    # Value hits (high actual FPTS on low salary)
    if 'salary' in merged.columns and merged['salary'].notna().any():
        merged_sal = merged[merged['salary'].notna()].copy()
        if not merged_sal.empty:
            merged_sal['actual_value'] = merged_sal['actual_fpts'] / (merged_sal['salary'] / 1000)
            print(f"\n  TOP 5 VALUE PLAYS (actual FPTS/1k salary):")
            for _, r in merged_sal.nlargest(5, 'actual_value').iterrows():
                print(f"    {r['name']:<25} ${int(r['salary']):>6,}  "
                      f"Actual: {r['actual_fpts']:5.1f}  Value: {r['actual_value']:.2f}x")

    print()


# ================================================================
#  Save Results
# ================================================================

def save_backtest_csv(merged: pd.DataFrame, date: str) -> Path:
    """Save backtest details CSV."""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    dt = datetime.strptime(date, '%Y-%m-%d')
    filename = f"{dt.month}.{dt.day}.{dt.strftime('%y')}_post_slate_backtest.csv"
    filepath = BACKTEST_DIR / filename

    # Select columns to save
    cols = ['name', 'team', 'position', 'actual_fpts', 'projected_fpts',
            'error', 'abs_error', 'salary']
    save_cols = [c for c in cols if c in merged.columns]
    merged['date'] = date

    out = merged[save_cols + ['date']].copy()
    out.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    return filepath


# ================================================================
#  Main Pipeline
# ================================================================

def run_post_slate(date: str, log_to_db: bool = True, verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Full post-slate pipeline:
        1. Find projection CSV
        2. Fetch actuals from NHL API
        3. Match and compute errors
        4. Print report
        5. Save CSV
        6. Log to history DB
    """
    if verbose:
        print(f"\n  Post-slate backtest for {date}")
        print(f"  {'─' * 40}")

    # Step 1: Find projections
    proj_file = find_projection_csv(date)
    if not proj_file:
        print(f"  ✗ No projection CSV found for {date}")
        return None
    if verbose:
        print(f"  ✓ Projections: {proj_file.name}")

    projections = pd.read_csv(proj_file)

    # Step 2: Fetch actuals
    actuals = fetch_actual_scores(date, verbose=verbose)
    if actuals.empty:
        print(f"  ✗ No actual scores available for {date} (games not finished?)")
        return None
    if verbose:
        print(f"  ✓ Actuals: {len(actuals)} players scored")

    # Step 3: Match
    merged = match_players(actuals, projections)
    if merged.empty:
        print(f"  ✗ Could not match any players between projections and actuals")
        return None
    if verbose:
        print(f"  ✓ Matched: {len(merged)} players")

    # Step 4: Report
    print_accuracy_report(merged, date)

    # Step 5: Save CSV
    csv_path = save_backtest_csv(merged, date)

    # Step 6: Log to DB
    if log_to_db:
        try:
            conn = get_connection()
            n = ingest_backtest_csv(conn, str(csv_path), slate_date=date)
            conn.close()
            if verbose:
                print(f"  ✓ Logged {n} rows to history database")
        except Exception as e:
            print(f"  ⚠ DB logging failed: {e}")

    return merged


def backfill_all_dates(log_to_db: bool = True):
    """Run post-slate backtest for every date with projection CSVs."""
    dates = discover_dates_with_projections()
    if not dates:
        print("  No projection CSVs found to backfill.")
        return

    print(f"\n  Backfilling {len(dates)} dates: {dates[0]} → {dates[-1]}")
    print(f"  {'=' * 60}")

    results = {}
    for date in dates:
        try:
            merged = run_post_slate(date, log_to_db=log_to_db, verbose=False)
            if merged is not None and not merged.empty:
                mae = merged['abs_error'].mean()
                n = len(merged)
                results[date] = {'mae': mae, 'n': n}
                print(f"  ✓ {date}: MAE={mae:.2f} ({n} players)")
            else:
                print(f"  ✗ {date}: no data")
        except Exception as e:
            print(f"  ✗ {date}: error — {e}")

    if results:
        avg_mae = np.mean([r['mae'] for r in results.values()])
        print(f"\n  {'─' * 40}")
        print(f"  Average MAE across {len(results)} slates: {avg_mae:.2f}")


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Auto post-slate backtest — grade your projections after games finish'
    )
    parser.add_argument('--date', type=str, default=None,
                       help='Slate date (YYYY-MM-DD). Auto-detects if not specified.')
    parser.add_argument('--no-db', action='store_true',
                       help='Skip logging to history database')
    parser.add_argument('--backfill', action='store_true',
                       help='Backfill all dates with projection CSVs')
    args = parser.parse_args()

    if args.backfill:
        backfill_all_dates(log_to_db=not args.no_db)
        return

    date = args.date
    if not date:
        date = guess_last_slate_date()
        if not date:
            print("  No projection CSVs found. Specify --date YYYY-MM-DD")
            sys.exit(1)
        print(f"  Auto-detected last slate: {date}")

    result = run_post_slate(date, log_to_db=not args.no_db)
    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
