#!/usr/bin/env python3
"""
Full Pipeline Backtest — 5-Signal Blend + Ceiling + Goalie Context
===================================================================

Run this on your local machine where you have the full 32-team game log DB.

Usage:
    python backtest_full_pipeline.py
    python backtest_full_pipeline.py --goalie-detail
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent
proj_dir = PROJECT_DIR / 'daily_projections'
actuals_path = PROJECT_DIR / 'backtests' / 'batch_backtest_details.csv'
vegas_path = PROJECT_DIR / 'Vegas_Historical.csv'


def ln(n):
    return str(n).strip().split()[-1].lower()


def run_backtest(goalie_detail=False):
    from projection_blender import blend_projections
    from ceiling_clustering import predict_ceiling_probability

    actuals = pd.read_csv(actuals_path)

    # Load Vegas data: CSV as base, DB odds overlay per-date
    vdf = None
    if vegas_path.exists():
        vdf = pd.read_csv(vegas_path, encoding='utf-8-sig')
        vdf['date'] = vdf['Date'].apply(
            lambda d: f"20{d.split('.')[2]}-{int(d.split('.')[0]):02d}-{int(d.split('.')[1]):02d}"
        )

    # Try DB odds (richer data with moneylines)
    try:
        from historical_odds import get_odds_for_date as _get_odds_db
        _has_db_odds = True
    except ImportError:
        _has_db_odds = False

    dates = sorted(actuals['date'].unique())

    print("=" * 78)
    print("  FULL PIPELINE BACKTEST")
    print("  5-Signal Blend (Cur + Bayes + HMM + ESN + Goalie Context)")
    print("  + Ceiling Probability + PP/PK Penalty Matchup")
    print("=" * 78)

    # Per-slate tracking
    sk_cur_t, sk_v2_t, sk_n = 0, 0, 0
    g_cur_t, g_v2_t, g_n = 0, 0, 0
    all_goalie_records = []

    print(f"\n  {'Date':<12} {'N':>5} {'Cur':>7} {'Blend':>7} {'Imp':>7}"
          f"  {'G_Cur':>6} {'G_Bld':>6} {'G_Imp':>7} {'G_Adj':>5}")
    print(f"  {'-' * 72}")

    for date_str in dates:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        prefix = f'{dt.month:02d}_{dt.day:02d}_{dt.strftime("%y")}'

        proj_file = None
        for f in sorted(proj_dir.glob('*NHLprojections_*.csv')):
            if '_lineups' in f.name:
                continue
            if f.name.startswith(prefix):
                proj_file = f
        if not proj_file:
            continue

        pool = pd.read_csv(proj_file)

        # Step 1: 5-signal blend (use DB odds if available, else CSV)
        date_vegas = vdf  # CSV fallback
        if _has_db_odds:
            db_odds = _get_odds_db(date_str)
            if not db_odds.empty:
                db_odds['date'] = date_str
                date_vegas = db_odds

        blended = blend_projections(pool, vegas=date_vegas, date_str=date_str,
                                     replace=True, verbose=False)

        # Step 2: Ceiling probability
        try:
            blended = predict_ceiling_probability(blended)
            if 'p_ceiling' in blended.columns and 'ceiling' in blended.columns:
                base_rate = 0.065
                p = blended['p_ceiling'].clip(0.01, 0.50)
                ceiling_scale = (1.0 + 0.3 * np.log(p / base_rate)).clip(0.85, 1.4)
                base_ceiling = blended['projected_fpts'] * 2.5 + 5
                goalie_mask_c = blended['position'] == 'G'
                base_ceiling[goalie_mask_c] = blended.loc[goalie_mask_c, 'projected_fpts'] * 2.0 + 10
                blended['ceiling'] = (base_ceiling * ceiling_scale).round(1)
        except Exception:
            pass

        # Step 3: Game environment (Vegas + pace + recency)
        try:
            from game_environment import GameEnvironmentModel
            env = GameEnvironmentModel()
            env.fit()
            if env.fitted:
                blended = env.adjust_projections(
                    blended, vegas=date_vegas, date_str=date_str, verbose=False
                )
        except Exception:
            pass

        # Match to actuals
        act_date = actuals[actuals['date'] == date_str].copy()
        act_date['_key'] = act_date['name'].apply(ln) + '_' + act_date['team'].str.lower()
        blended['_key'] = blended['name'].apply(ln) + '_' + blended['team'].str.lower()

        goalie_cols = ['_key', 'blended_fpts', 'current_fpts']
        if 'goalie_context_adj' in blended.columns:
            goalie_cols.append('goalie_context_adj')

        merged = act_date.merge(
            blended[goalie_cols].drop_duplicates('_key'),
            on='_key', how='inner'
        )
        if merged.empty:
            continue

        sk = merged[merged['position'] != 'G']
        g = merged[merged['position'] == 'G']

        # Per-slate MAE
        all_cur_mae = (merged['current_fpts'] - merged['actual_fpts']).abs().mean()
        all_bld_mae = (merged['blended_fpts'] - merged['actual_fpts']).abs().mean()
        all_imp = (all_cur_mae - all_bld_mae) / all_cur_mae * 100

        g_cur_mae = (g['current_fpts'] - g['actual_fpts']).abs().mean() if len(g) > 0 else 0
        g_bld_mae = (g['blended_fpts'] - g['actual_fpts']).abs().mean() if len(g) > 0 else 0
        g_imp = (g_cur_mae - g_bld_mae) / g_cur_mae * 100 if g_cur_mae > 0 else 0

        g_adj_n = 0
        if 'goalie_context_adj' in g.columns:
            g_adj_n = (g['goalie_context_adj'].abs() > 0.05).sum()

        print(f"  {date_str:<12} {len(merged):>5} {all_cur_mae:>7.3f} {all_bld_mae:>7.3f} {all_imp:>+6.1f}%"
              f"  {g_cur_mae:>6.2f} {g_bld_mae:>6.2f} {g_imp:>+6.1f}% {g_adj_n:>4}/{len(g)}")

        # Accumulate
        sk_cur_t += (sk['current_fpts'] - sk['actual_fpts']).abs().sum()
        sk_v2_t += (sk['blended_fpts'] - sk['actual_fpts']).abs().sum()
        sk_n += len(sk)
        g_cur_t += (g['current_fpts'] - g['actual_fpts']).abs().sum()
        g_v2_t += (g['blended_fpts'] - g['actual_fpts']).abs().sum()
        g_n += len(g)

        if len(g) > 0:
            g_copy = g.copy()
            g_copy['date'] = date_str
            all_goalie_records.append(g_copy)

    # ── Summary ──
    sc = sk_cur_t / sk_n
    sv = sk_v2_t / sk_n
    gc = g_cur_t / g_n
    gv = g_v2_t / g_n
    oc = (sk_cur_t + g_cur_t) / (sk_n + g_n)
    ov = (sk_v2_t + g_v2_t) / (sk_n + g_n)

    print(f"  {'-' * 72}")
    print(f"\n  RESULTS:")
    print(f"  Skaters ({sk_n:>4}):  {sc:.3f} → {sv:.3f}  ({(sc-sv)/sc*100:+.1f}%)")
    print(f"  Goalies ({g_n:>4}):   {gc:.3f} → {gv:.3f}  ({(gc-gv)/gc*100:+.1f}%)")
    print(f"  Combined ({sk_n+g_n:>4}): {oc:.3f} → {ov:.3f}  ({(oc-ov)/oc*100:+.1f}%)")

    # ── Goalie Detail ──
    if all_goalie_records:
        all_g = pd.concat(all_goalie_records, ignore_index=True)

        g_bias_cur = (all_g['current_fpts'] - all_g['actual_fpts']).mean()
        g_bias_bld = (all_g['blended_fpts'] - all_g['actual_fpts']).mean()

        adj_count = 0
        avg_adj = 0
        if 'goalie_context_adj' in all_g.columns:
            adj_mask = all_g['goalie_context_adj'].abs() > 0.05
            adj_count = adj_mask.sum()
            if adj_count > 0:
                avg_adj = all_g.loc[adj_mask, 'goalie_context_adj'].mean()

                # MAE for adjusted vs non-adjusted goalies
                adj_g = all_g[adj_mask]
                non_g = all_g[~adj_mask]
                adj_cur_mae = (adj_g['current_fpts'] - adj_g['actual_fpts']).abs().mean()
                adj_bld_mae = (adj_g['blended_fpts'] - adj_g['actual_fpts']).abs().mean()
                non_cur_mae = (non_g['current_fpts'] - non_g['actual_fpts']).abs().mean() if len(non_g) > 0 else 0
                non_bld_mae = (non_g['blended_fpts'] - non_g['actual_fpts']).abs().mean() if len(non_g) > 0 else 0

        print(f"\n  GOALIE BREAKDOWN:")
        print(f"    Bias: Current {g_bias_cur:+.2f} → Blend {g_bias_bld:+.2f}")
        print(f"    Context adjusted: {adj_count}/{len(all_g)} goalies (avg adj: {avg_adj:+.2f})")

        if adj_count > 0 and len(non_g) > 0:
            adj_imp = (adj_cur_mae - adj_bld_mae) / adj_cur_mae * 100 if adj_cur_mae > 0 else 0
            non_imp = (non_cur_mae - non_bld_mae) / non_cur_mae * 100 if non_cur_mae > 0 else 0
            print(f"    Adjusted goalies MAE:     {adj_cur_mae:.3f} → {adj_bld_mae:.3f} ({adj_imp:+.1f}%)")
            print(f"    Non-adjusted goalies MAE: {non_cur_mae:.3f} → {non_bld_mae:.3f} ({non_imp:+.1f}%)")

        if goalie_detail:
            print(f"\n  GOALIE-BY-GOALIE:")
            print(f"  {'Name':<22} {'Date':<12} {'Actual':>7} {'Cur':>6} {'Blend':>6} {'Adj':>5} {'Δ':>6}")
            print(f"  {'-' * 65}")
            all_g = all_g.sort_values('date')
            for _, row in all_g.iterrows():
                adj = row.get('goalie_context_adj', 0) or 0
                cur_err = abs(row['current_fpts'] - row['actual_fpts'])
                bld_err = abs(row['blended_fpts'] - row['actual_fpts'])
                delta = cur_err - bld_err
                print(f"  {row['name']:<22} {row['date']:<12} {row['actual_fpts']:>7.1f} "
                      f"{row['current_fpts']:>6.1f} {row['blended_fpts']:>6.1f} "
                      f"{adj:>+5.1f} {delta:>+5.1f}")

    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--goalie-detail', action='store_true',
                       help='Show every goalie game with adjustments')
    args = parser.parse_args()
    run_backtest(goalie_detail=args.goalie_detail)
