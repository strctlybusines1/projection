#!/usr/bin/env python3
"""
Rebuild Actuals & Run Full Pipeline Backtest
=============================================

Step 1: Re-runs batch backtest to regenerate batch_backtest_details.csv
        with ALL available dates (including Feb 2-5).
Step 2: Runs the full 5-signal pipeline backtest against those actuals.

Usage:
    python rebuild_and_backtest.py                   # Rebuild actuals then backtest
    python rebuild_and_backtest.py --backtest-only    # Skip rebuild, just backtest
    python rebuild_and_backtest.py --goalie-detail    # Show every goalie game
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


def rebuild_actuals():
    """Re-run batch backtest to regenerate actuals CSV with all dates."""
    print("=" * 60)
    print("  STEP 1: Rebuilding batch_backtest_details.csv")
    print("          (fetching actuals for ALL available dates)")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "backtest.py", "--batch-backtest"],
        cwd=str(PROJECT_DIR),
        capture_output=False,
    )

    csv_path = PROJECT_DIR / "backtests" / "batch_backtest_details.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        dates = sorted(df['date'].unique())
        print(f"\n  ✓ Actuals rebuilt: {len(df)} records across {len(dates)} dates")
        print(f"    Dates: {', '.join(dates)}")
    else:
        print("  ⚠ batch_backtest_details.csv not found after rebuild")
        sys.exit(1)

    return result.returncode


def run_pipeline_backtest(goalie_detail=False):
    """Run the full pipeline backtest."""
    print("\n" + "=" * 60)
    print("  STEP 2: Full Pipeline Backtest")
    print("=" * 60)

    args = [sys.executable, "backtest_full_pipeline.py"]
    if goalie_detail:
        args.append("--goalie-detail")

    subprocess.run(args, cwd=str(PROJECT_DIR))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest-only', action='store_true',
                       help='Skip rebuilding actuals, just run backtest')
    parser.add_argument('--goalie-detail', action='store_true',
                       help='Show every goalie game with adjustments')
    args = parser.parse_args()

    if not args.backtest_only:
        rebuild_actuals()

    run_pipeline_backtest(goalie_detail=args.goalie_detail)
