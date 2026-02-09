#!/usr/bin/env python3
"""
Projection Blender — Combines Current Model + Bayesian Event Model
===================================================================

Optimal parameters (backtested across 7 slates, 1451 player-games):
    - 45% current model + 55% Bayesian model
    - Skater bias correction: -3.0 FPTS
    - Result: 4.105 MAE (25.8% improvement over current 5.533)

Integration:
    Called from main.py after generate_projections() returns.
    Adds 'blended_fpts' column and optionally replaces 'projected_fpts'.

Usage:
    from projection_blender import blend_projections
    player_pool = blend_projections(player_pool, vegas_data, date_str)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ================================================================
#  Blend Parameters (backtested optimal)
# ================================================================

# Weight for current model (1 - this = Bayesian weight)
CURRENT_WEIGHT = 0.50
BAYESIAN_WEIGHT = 1 - CURRENT_WEIGHT

# Post-blend bias corrections
SKATER_BIAS_SHIFT = -2.50    # Compromise: -4.0 optimal on backtest but -3.0 for newer slates
GOALIE_BIAS_SHIFT = 0.0     # Goalies: no shift needed

# Floor: don't let blended projection go below this
MIN_PROJECTION = 0.5


# ================================================================
#  Blender
# ================================================================

def blend_projections(player_pool: pd.DataFrame,
                      vegas: pd.DataFrame = None,
                      date_str: str = None,
                      replace: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Blend current projections with Bayesian event-probability projections.

    Args:
        player_pool: DataFrame with 'projected_fpts' and per-game stat columns
        vegas: Optional Vegas data with Team, TeamGoal columns
        date_str: Slate date for Vegas lookup
        replace: If True, overwrite 'projected_fpts' with blended value
        verbose: Print summary

    Returns:
        player_pool with 'blended_fpts' column (and updated 'projected_fpts' if replace=True)
    """
    try:
        from bayesian_projections import BayesianProjector
    except ImportError:
        if verbose:
            print("  ⚠ bayesian_projections.py not found — skipping blend")
        return player_pool

    df = player_pool.copy()
    bp = BayesianProjector()

    # Generate Bayesian projections
    result = bp.project_player_pool(df, vegas, date_str)

    if 'bayes_expected_fpts' not in result.columns:
        if verbose:
            print("  ⚠ Bayesian projections failed — using current model only")
        return player_pool

    # Store originals
    df['current_fpts'] = result['projected_fpts']
    df['bayes_fpts'] = result['bayes_expected_fpts']

    # Additional Bayesian outputs
    for col in ['bayes_floor', 'bayes_ceiling', 'bayes_std',
                'bayes_p_goal', 'bayes_p_assist', 'bayes_p_five_sog',
                'bayes_p_three_blocks', 'bayes_p_shutout', 'bayes_p_win',
                'bayes_median_fpts', 'bayes_p5', 'bayes_p95']:
        if col in result.columns:
            df[col] = result[col]

    # Blend
    blended = CURRENT_WEIGHT * df['current_fpts'] + BAYESIAN_WEIGHT * df['bayes_fpts']

    # Apply position-specific bias correction
    skater_mask = df['position'] != 'G'
    goalie_mask = df['position'] == 'G'

    blended[skater_mask] += SKATER_BIAS_SHIFT
    blended[goalie_mask] += GOALIE_BIAS_SHIFT

    # Floor
    blended = blended.clip(lower=MIN_PROJECTION)

    df['blended_fpts'] = blended.round(2)

    if replace:
        df['projected_fpts'] = df['blended_fpts']
        # Update derived columns
        if 'salary' in df.columns:
            df['value'] = (df['projected_fpts'] / (df['salary'] / 1000)).round(3)
        if 'dk_avg_fpts' in df.columns:
            df['edge'] = (df['projected_fpts'] - df['dk_avg_fpts']).round(3)
        # Update floor/ceiling with Bayesian values if available
        if 'bayes_floor' in df.columns:
            df['floor'] = df['bayes_floor']
        if 'bayes_ceiling' in df.columns:
            df['ceiling'] = df['bayes_ceiling']

    if verbose:
        # Summary stats
        n_skaters = skater_mask.sum()
        n_goalies = goalie_mask.sum()

        sk_current = df.loc[skater_mask, 'current_fpts'].mean()
        sk_bayes = df.loc[skater_mask, 'bayes_fpts'].mean()
        sk_blended = df.loc[skater_mask, 'blended_fpts'].mean()

        print(f"\n  ── Projection Blend ──────────────────────────────")
        print(f"  Weights: {CURRENT_WEIGHT:.0%} current / {BAYESIAN_WEIGHT:.0%} Bayesian")
        print(f"  Skater bias correction: {SKATER_BIAS_SHIFT:+.1f} FPTS")
        print(f"  Players: {n_skaters} skaters, {n_goalies} goalies")
        print(f"  Skater avg — Current: {sk_current:.1f}  Bayes: {sk_bayes:.1f}  Blended: {sk_blended:.1f}")

        if n_goalies > 0:
            g_current = df.loc[goalie_mask, 'current_fpts'].mean()
            g_bayes = df.loc[goalie_mask, 'bayes_fpts'].mean()
            g_blended = df.loc[goalie_mask, 'blended_fpts'].mean()
            print(f"  Goalie avg — Current: {g_current:.1f}  Bayes: {g_bayes:.1f}  Blended: {g_blended:.1f}")

        # Show top players comparison
        top = df.nlargest(10, 'blended_fpts')
        print(f"\n  {'Name':<22} {'Cur':>6} {'Bay':>6} {'Bld':>6} {'P(G)':>6} {'P(A)':>6} {'Floor':>6} {'Ceil':>6}")
        print(f"  {'─' * 68}")
        for _, p in top.iterrows():
            pg = p.get('bayes_p_goal', 0) or 0
            pa = p.get('bayes_p_assist', 0) or 0
            fl = p.get('bayes_floor', 0) or 0
            cl = p.get('bayes_ceiling', 0) or 0
            print(f"  {p['name']:<22} {p['current_fpts']:>6.1f} {p['bayes_fpts']:>6.1f} "
                  f"{p['blended_fpts']:>6.1f} {pg:>5.0%} {pa:>5.0%} {fl:>6.1f} {cl:>6.1f}")

        print()

    return df


# ================================================================
#  Calibration Update (run after each slate)
# ================================================================

def recalibrate(actuals_csv: str = 'backtests/batch_backtest_details.csv',
                vegas_csv: str = 'Vegas_Historical.csv',
                n_recent: int = 10):
    """
    Re-run the blend weight optimization using latest backtests.

    Call this periodically (weekly or after accumulating more slates)
    to update CURRENT_WEIGHT and SKATER_BIAS_SHIFT.

    Prints the new optimal parameters — manually update this file.
    """
    from datetime import datetime
    from bayesian_projections import BayesianProjector

    PROJECT_ROOT = Path(__file__).resolve().parent
    proj_dir = PROJECT_ROOT / 'daily_projections'

    actuals = pd.read_csv(PROJECT_ROOT / actuals_csv)
    vegas = None
    vpath = PROJECT_ROOT / vegas_csv
    if vpath.exists():
        vdf = pd.read_csv(vpath, encoding='utf-8-sig')
        vdf['date'] = vdf['Date'].apply(
            lambda d: f"20{d.split('.')[2]}-{int(d.split('.')[0]):02d}-{int(d.split('.')[1]):02d}"
        )
        vegas = vdf

    bp = BayesianProjector()
    def ln(n): return n.strip().split()[-1].lower()

    dates = sorted(actuals['date'].unique())[-n_recent:]
    all_records = []

    for date_str in dates:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        prefix = f'{dt.month:02d}_{dt.day:02d}_{dt.strftime("%y")}'
        proj_file = None
        for f in sorted(proj_dir.glob('*NHLprojections_*.csv')):
            if '_lineups' in f.name: continue
            if f.name.startswith(prefix): proj_file = f
        if not proj_file: continue

        pool = pd.read_csv(proj_file)
        result = bp.project_player_pool(pool, vegas, date_str)
        act_date = actuals[actuals['date'] == date_str].copy()
        if act_date.empty: continue

        act_date['_key'] = act_date['name'].apply(ln) + '_' + act_date['team'].str.lower()
        result['_key'] = result['name'].apply(ln) + '_' + result['team'].str.lower()

        merged = act_date.merge(
            result[['_key', 'bayes_expected_fpts', 'projected_fpts']].drop_duplicates('_key'),
            on='_key', how='inner', suffixes=('_actual', '_proj')
        )
        if not merged.empty:
            merged['date'] = date_str
            all_records.append(merged)

    if not all_records:
        print("No matching data found for recalibration")
        return

    combined = pd.concat(all_records, ignore_index=True)
    print(f"Recalibrating on {len(combined)} observations across {len(all_records)} slates")

    best_mae = 999
    best_params = {}

    for w in np.arange(0, 1.01, 0.05):
        for shift in np.arange(-5.0, 1.0, 0.25):
            blended = w * combined['projected_fpts_proj'] + (1-w) * combined['bayes_expected_fpts']
            sk = combined['position'] != 'G'
            adj = blended.copy()
            adj[sk] += shift
            adj = adj.clip(lower=0.5)
            mae = (adj - combined['actual_fpts']).abs().mean()
            if mae < best_mae:
                best_mae = mae
                best_params = {'w': w, 'shift': shift}

    p = best_params
    print(f"\n  OPTIMAL BLEND PARAMETERS:")
    print(f"    CURRENT_WEIGHT = {p['w']:.2f}")
    print(f"    BAYESIAN_WEIGHT = {1-p['w']:.2f}")
    print(f"    SKATER_BIAS_SHIFT = {p['shift']:.2f}")
    print(f"    MAE: {best_mae:.3f}")
    print(f"\n  Update projection_blender.py with these values.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--recalibrate', action='store_true',
                       help='Find optimal blend weights from latest backtests')
    parser.add_argument('--recent', type=int, default=10,
                       help='Number of recent slates to use for recalibration')
    args = parser.parse_args()

    if args.recalibrate:
        recalibrate(n_recent=args.recent)
    else:
        print("Usage:")
        print("  python projection_blender.py --recalibrate")
        print("  python projection_blender.py --recalibrate --recent 5")
