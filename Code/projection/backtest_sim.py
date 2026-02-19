#!/usr/bin/env python3
"""
Simons Backtest — Full Simulation Engine Validation
=====================================================

Runs the complete pipeline on 7 historical slates:
1. Load DK season file as player pool (with actual scores)
2. Use ONLY games BEFORE that date for distribution fitting (no leakage)
3. Generate 100+ triple-mix candidates via optimizer
4. Run SimSelector with all selection modes (m3s, gpp, ev, cash)
5. Compare every selector against actual outcomes
6. Diagnose goalie picks, stack quality, team overlap with best possible

Usage:
    python backtest_sim.py                    # All 7 slates
    python backtest_sim.py --date 2026-02-05  # Single date
    python backtest_sim.py --candidates 150   # More candidates
    python backtest_sim.py --sims 15000       # More Monte Carlo sims

Backtest dates: 01-29, 01-31, 02-01, 02-02, 02-03, 02-04, 02-05
"""

import argparse
import glob
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Add projection dir to path
PROJ_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR))

from simulation_engine import SimulationEngine
from sim_selector import SimSelector, _load_payout_curve
from optimizer import NHLLineupOptimizer

DK_DIR = Path.home() / "Desktop" / "DKSalaries_NHL_season"

# 7 backtest dates: target_date → slate_date suffix in the DataFrame
BACKTEST_DATES = {
    '2026-01-29': '2026-01-29_players (1)',
    '2026-01-31': '2026-01-31_players (2)',
    '2026-02-01': '2026-02-01_players (1)',
    '2026-02-02': '2026-02-02_players (1)',
    '2026-02-03': '2026-02-03_players (1)',
    '2026-02-04': '2026-02-04_players (1)',
    '2026-02-05': '2026-02-05_players (1)',
}


def normalize_pos(p):
    p = str(p).upper()
    if p in ('L', 'R', 'LW', 'RW'): return 'W'
    if p in ('LD', 'RD'): return 'D'
    return p


def load_all_dk_history():
    """Load all DK season files into one DataFrame."""
    search_paths = [
        DK_DIR,
        Path("/home/claude/dk_salaries_season/DKSalaries_NHL_season"),
        PROJ_DIR.parent.parent / "dk_salaries_season" / "DKSalaries_NHL_season",
    ]
    
    files = []
    for sp in search_paths:
        files = sorted(glob.glob(str(sp / "draftkings_NHL_*.csv")))
        if files:
            break

    all_data = []
    for f in files:
        # Extract date portion: draftkings_NHL_YYYY-MM-DD_players*.csv
        basename = os.path.basename(f)
        # slate_date = full suffix after "draftkings_NHL_" minus ".csv"
        suffix = basename.replace('draftkings_NHL_', '').replace('.csv', '')
        try:
            df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
            for col in ['Score', 'Salary', 'Avg', 'Ceiling', 'TeamGoal', 'OppGoal']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['slate_date'] = suffix  # e.g. "2026-01-29_players (1)"
            # Also store clean date for temporal ordering
            df['clean_date'] = suffix[:10]  # "2026-01-29"
            all_data.append(df)
        except Exception as e:
            print(f"  ⚠ Could not load {basename}: {e}")

    if not all_data:
        print("ERROR: No DK season files found.")
        print(f"  Searched: {[str(sp) for sp in search_paths]}")
        sys.exit(1)

    full = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(full)} rows from {len(all_data)} DK files")
    return full


def generate_candidates(pool: pd.DataFrame, n_candidates: int = 100):
    """Generate triple-mix candidates from pool."""
    optimizer = NHLLineupOptimizer()
    candidates = []

    capped = pool[pool['salary'] <= 7500].copy()
    pos_counts = capped['position'].value_counts()
    has_capped = (pos_counts.get('C', 0) >= 3 and pos_counts.get('W', 0) >= 4
                  and pos_counts.get('D', 0) >= 3 and pos_counts.get('G', 0) >= 2)

    for rand in [0.05, 0.10, 0.15, 0.20, 0.25]:
        n_std = max(n_candidates // 10, 5)
        n_cap = max(n_candidates // 20, 3)
        n_33 = max(n_candidates // 20, 3)

        try:
            batch = optimizer.optimize_lineup(pool, n_lineups=n_std, randomness=rand)
            if batch: candidates.extend(batch)
        except Exception:
            pass

        if has_capped:
            try:
                batch = optimizer.optimize_lineup(capped, n_lineups=n_cap, randomness=rand)
                if batch: candidates.extend(batch)
            except Exception:
                pass

        try:
            batch = optimizer.optimize_lineup(pool, n_lineups=n_33, randomness=rand, max_from_team=3)
            if batch: candidates.extend(batch)
        except Exception:
            pass

    return candidates


def compute_ev(sim_result, payout_curve):
    """Compute E[payout] from simulated totals."""
    totals = sim_result['simulated_totals']
    p_bins = sorted(payout_curve.keys())
    p_vals = [payout_curve[b] for b in p_bins]
    ev = 0.0
    n = len(totals)
    for i in range(len(p_bins) - 1):
        lo, hi = p_bins[i], p_bins[i + 1]
        mid_pay = (p_vals[i] + p_vals[i + 1]) / 2
        ev += (np.sum((totals >= lo) & (totals < hi)) / n) * mid_pay
    ev += (np.sum(totals >= p_bins[-1]) / n) * p_vals[-1]
    return ev


def stack_desc(lu):
    tc = lu['team'].value_counts()
    return '-'.join(str(v) for v in sorted(tc.values, reverse=True) if v >= 2)


def run_single_date(target_date, dk_date, full_df, n_candidates=100, n_sims=8000, verbose=True):
    """Run full backtest for one date. Returns result dict."""
    t0 = time.time()

    # Get today's pool
    pool = full_df[full_df['slate_date'] == dk_date].copy()
    pool = pool[pool['Score'].notna() & pool['Salary'].notna()].copy()

    # Build history: ONLY dates BEFORE this one (use clean_date for ordering)
    hist = full_df[full_df['Score'].notna() & (full_df['clean_date'] < target_date)].copy()
    hist_for_sim = hist[['Player', 'Team', 'Score', 'Pos', 'slate_date']].copy()

    if len(pool) < 30:
        print(f"  {target_date}: insufficient pool ({len(pool)}), skip")
        return None

    # Prepare pool for optimizer
    opt_pool = pool.rename(columns={
        'Player': 'name', 'Team': 'team', 'Pos': 'position',
        'Salary': 'salary', 'Avg': 'projected_fpts',
    }).copy()
    opt_pool['position'] = opt_pool['position'].apply(normalize_pos)
    opt_pool['projected_fpts'] = opt_pool['projected_fpts'].fillna(3.0)

    # Generate candidates
    candidates = generate_candidates(opt_pool, n_candidates)
    if len(candidates) < 20:
        print(f"  {target_date}: only {len(candidates)} candidates, skip")
        return None

    # Score actuals
    score_map = dict(zip(pool['Player'] + '_' + pool['Team'], pool['Score']))
    actual_arr = np.array([
        sum(score_map.get(f"{r['name']}_{r['team']}", 0) for _, r in lu.iterrows())
        for lu in candidates
    ])
    proj_arr = np.array([lu['projected_fpts'].sum() for lu in candidates])

    # Fit simulation engine
    engine = SimulationEngine(n_sims=n_sims)
    engine.fit_player_distributions(opt_pool, hist_for_sim, date_str=dk_date)

    # Simulate all candidates
    sim_results = []
    for lu in candidates:
        sim = engine.simulate_lineup(lu, n_sims=n_sims)
        sim_results.append(sim)

    sim_means = np.array([s['mean'] for s in sim_results])
    sim_stds = np.array([s['std'] for s in sim_results])
    sim_pcash = np.array([s['p_cash'] for s in sim_results])
    sim_pgpp = np.array([s['p_gpp'] for s in sim_results])
    sim_p95 = np.array([s['p95'] for s in sim_results])

    # M+3σ (sim-calibrated)
    m3s = sim_means + 3.0 * sim_stds

    # E[payout]
    payout = _load_payout_curve()
    sim_evs = np.array([compute_ev(sr, payout) for sr in sim_results])

    # All selectors
    picks = {
        'Best Possible':  int(np.argmax(actual_arr)),
        'Sim M+3σ':       int(np.argmax(m3s)),
        'Sim P(≥140)':    int(np.argmax(sim_pgpp)),
        'Sim E[payout]':  int(np.argmax(sim_evs)),
        'Sim P(≥111)':    int(np.argmax(sim_pcash)),
        'Sim P95':        int(np.argmax(sim_p95)),
        'Max Projection':  int(np.argmax(proj_arr)),
        'Sim Mean':        int(np.argmax(sim_means)),
        'Max Std':         int(np.argmax(sim_stds)),
    }

    avg_actual = actual_arr.mean()
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  ── {target_date} ({len(candidates)} candidates, {len(hist_for_sim)} hist, {elapsed:.1f}s) ──")
        print(f"    {'Selector':<18} {'Actual':>7} {'vs Avg':>7} {'Pctile':>7} {'SimM':>6} {'SimS':>5} {'M+3σ':>6}")
        print(f"    {'-' * 56}")

    result = {
        'date': target_date, 'n_candidates': len(candidates),
        'n_hist': len(hist_for_sim), 'avg': avg_actual, 'elapsed': elapsed,
    }

    for label, idx in picks.items():
        act = actual_arr[idx]
        pctile = (actual_arr < act).sum() / len(actual_arr) * 100
        vs_avg = act - avg_actual
        sm = sim_means[idx]
        ss = sim_stds[idx]
        m3 = m3s[idx]
        marker = ' ◄' if label == 'Sim M+3σ' else ''

        if verbose:
            print(f"    {label:<18} {act:>7.1f} {vs_avg:>+6.1f} {pctile:>6.0f}% "
                  f"{sm:>6.1f} {ss:>5.1f} {m3:>6.0f}{marker}")

        result[label] = act
        result[f'{label}_pctile'] = pctile

    # Goalie diagnosis
    sim_pick_idx = picks['Sim M+3σ']
    best_idx = picks['Best Possible']
    sim_lu = candidates[sim_pick_idx]
    best_lu = candidates[best_idx]

    sim_goalie = sim_lu[sim_lu['position'] == 'G']
    best_goalie = best_lu[best_lu['position'] == 'G']

    if verbose and len(sim_goalie) > 0 and len(best_goalie) > 0:
        sg = sim_goalie.iloc[0]
        bg = best_goalie.iloc[0]
        sg_act = score_map.get(f"{sg['name']}_{sg['team']}", 0)
        bg_act = score_map.get(f"{bg['name']}_{bg['team']}", 0)
        print(f"\n    Goalie: Sim picked {sg['name']} ({sg['team']}) → {sg_act:.1f} FPTS")
        print(f"    Goalie: Best was   {bg['name']} ({bg['team']}) → {bg_act:.1f} FPTS")
        result['sim_goalie'] = sg['name']
        result['sim_goalie_fpts'] = sg_act
        result['best_goalie'] = bg['name']
        result['best_goalie_fpts'] = bg_act

        # Stack analysis
        print(f"    Stacks: Sim={stack_desc(sim_lu)}, Best={stack_desc(best_lu)}")
        overlap = set(sim_lu['team']) & set(best_lu['team'])
        print(f"    Team overlap: {len(overlap)}/{len(set(best_lu['team']))} ({', '.join(sorted(overlap))})")

    return result


def main():
    parser = argparse.ArgumentParser(description='Simons Backtest — Simulation Engine Validation')
    parser.add_argument('--date', type=str, default=None,
                        help='Single date to backtest (YYYY-MM-DD)')
    parser.add_argument('--candidates', type=int, default=100,
                        help='Number of candidates per slate (default: 100)')
    parser.add_argument('--sims', type=int, default=8000,
                        help='Monte Carlo simulations per lineup (default: 8000)')
    args = parser.parse_args()

    print(f"{'=' * 80}")
    print(f"  SIMONS BACKTEST — Simulation Engine Validation")
    print(f"  Candidates: {args.candidates} | Sims: {args.sims:,}")
    print(f"{'=' * 80}")

    full = load_all_dk_history()

    # Select dates
    if args.date:
        if args.date not in BACKTEST_DATES:
            print(f"Date {args.date} not in backtest set. Available: {list(BACKTEST_DATES.keys())}")
            sys.exit(1)
        dates = {args.date: BACKTEST_DATES[args.date]}
    else:
        dates = BACKTEST_DATES

    all_results = []
    for target_date, dk_date in dates.items():
        result = run_single_date(target_date, dk_date, full,
                                 n_candidates=args.candidates,
                                 n_sims=args.sims)
        if result:
            all_results.append(result)

    if len(all_results) < 2:
        print("\nNot enough slates for summary.")
        return

    rdf = pd.DataFrame(all_results)

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    selectors = ['Sim M+3σ', 'Sim P(≥140)', 'Sim E[payout]', 'Sim P(≥111)',
                 'Max Projection', 'Sim Mean', 'Max Std']

    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY — {len(rdf)} slates")
    print(f"{'=' * 80}")

    avg_avg = rdf['avg'].mean()
    avg_best = rdf['Best Possible'].mean()
    best_edge = avg_best - avg_avg

    print(f"\n  {'Selector':<18} {'Avg FPTS':>9} {'Edge':>7} {'%Best':>6} {'AvgPctile':>10} {'Top25%':>7}")
    print(f"  {'-' * 60}")

    for s in selectors:
        if s not in rdf.columns:
            continue
        avg_val = rdf[s].mean()
        edge = avg_val - avg_avg
        pct = edge / best_edge * 100 if best_edge > 0 else 0
        avg_pctile = rdf[f'{s}_pctile'].mean()
        top25 = (rdf[f'{s}_pctile'] >= 75).sum()
        n = len(rdf)
        print(f"  {s:<18} {avg_val:>9.1f} {edge:>+6.1f} {pct:>5.0f}% {avg_pctile:>9.0f}% "
              f"{top25}/{n} ({top25/n:.0%})")

    print(f"\n  Avg candidate: {avg_avg:.0f} | Best possible: {avg_best:.0f} | Max edge: {best_edge:.0f}")

    # Goalie analysis
    if 'sim_goalie_fpts' in rdf.columns and 'best_goalie_fpts' in rdf.columns:
        goalie_delta = rdf['sim_goalie_fpts'].mean() - rdf['best_goalie_fpts'].mean()
        goalie_match = (rdf['sim_goalie'] == rdf['best_goalie']).sum()
        print(f"\n  GOALIE ANALYSIS:")
        print(f"    Sim goalie avg: {rdf['sim_goalie_fpts'].mean():.1f} vs Best goalie avg: {rdf['best_goalie_fpts'].mean():.1f}")
        print(f"    Goalie match rate: {goalie_match}/{len(rdf)} ({goalie_match/len(rdf):.0%})")
        print(f"    Goalie delta: {goalie_delta:+.1f} FPTS/slate")

    # Per-slate detail table
    print(f"\n  {'Date':<12} {'Best':>6} {'SimM3σ':>7} {'P140':>6} {'E[$]':>6} {'MaxPr':>6} {'Avg':>6}")
    print(f"  {'-' * 50}")
    for _, r in rdf.iterrows():
        print(f"  {r['date']:<12} {r['Best Possible']:>6.0f} {r.get('Sim M+3σ', 0):>7.0f} "
              f"{r.get('Sim P(≥140)', 0):>6.0f} {r.get('Sim E[payout]', 0):>6.0f} "
              f"{r.get('Max Projection', 0):>6.0f} {r['avg']:>6.0f}")

    total_elapsed = rdf['elapsed'].sum()
    print(f"\n  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Avg per slate: {total_elapsed/len(rdf):.1f}s")


if __name__ == '__main__':
    main()
