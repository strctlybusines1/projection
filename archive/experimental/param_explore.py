#!/usr/bin/env python3
"""
Parallel Parameter Exploration for NHL DFS
=============================================

Tests different parameter combinations across the sim selector backtest
to find optimal settings for lineup generation.

Tunable parameters (from config.py):
  - DK_AVG_BLEND_WEIGHT: How much we trust our projection vs DK average
  - PRIMARY_STACK_BOOST: Boost given to primary stack players in optimizer
  - GOALIE_CORRELATION_BOOST: Boost for skaters correlated with our goalie
  - PREFERRED_PRIMARY_STACK_SIZE: Target stack size (3, 4, or 5)

Runs 5 parallel sub-tasks, each with different parameter combos.
"""

import json
import os
import sys
import time
import subprocess
import warnings
from pathlib import Path
from itertools import product

warnings.filterwarnings('ignore')

PROJ_DIR = Path(__file__).parent
RESULTS_DIR = PROJ_DIR / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# PARAMETER GRID
# ═══════════════════════════════════════════════════════════════

PARAM_GRID = {
    'dk_avg_blend': [0.70, 0.80, 0.90],         # Current: 0.80
    'primary_stack_boost': [0.10, 0.20, 0.30],   # Current: 0.20
    'goalie_corr_boost': [0.05, 0.10, 0.20],     # Current: 0.10
}

# Generate 5 strategic combos (not full grid of 27)
PARAM_COMBOS = [
    # Run 0: Current defaults (baseline)
    {'dk_avg_blend': 0.80, 'primary_stack_boost': 0.20, 'goalie_corr_boost': 0.10,
     'label': 'BASELINE (current defaults)'},
    # Run 1: More trust in our model
    {'dk_avg_blend': 0.70, 'primary_stack_boost': 0.20, 'goalie_corr_boost': 0.10,
     'label': 'More model trust (70/30 blend)'},
    # Run 2: More trust in DK average
    {'dk_avg_blend': 0.90, 'primary_stack_boost': 0.20, 'goalie_corr_boost': 0.10,
     'label': 'More DK trust (90/10 blend)'},
    # Run 3: Aggressive stacking
    {'dk_avg_blend': 0.80, 'primary_stack_boost': 0.30, 'goalie_corr_boost': 0.20,
     'label': 'Aggressive stacking (+30% stack, +20% goalie)'},
    # Run 4: Conservative stacking
    {'dk_avg_blend': 0.80, 'primary_stack_boost': 0.10, 'goalie_corr_boost': 0.05,
     'label': 'Conservative stacking (+10% stack, +5% goalie)'},
]


def write_worker_script(run_id, params):
    """Write a standalone Python script that runs one parameter combo."""
    script = f'''#!/usr/bin/env python3
"""Worker for parameter exploration run {run_id}"""
import sys, os, time, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')
sys.path.insert(0, '{PROJ_DIR}')

# Override config params BEFORE importing anything that reads them
import config
config.DK_AVG_BLEND_WEIGHT = {params['dk_avg_blend']}
config.PRIMARY_STACK_BOOST = {params['primary_stack_boost']}
config.GOALIE_CORRELATION_BOOST = {params['goalie_corr_boost']}

from simulation_engine import SimulationEngine
from optimizer import NHLLineupOptimizer
import glob

DK_DIR = "{PROJ_DIR.parent.parent / 'dk_salaries_season' / 'DKSalaries_NHL_season'}"
# Try alternate paths
for p in [DK_DIR, "/home/claude/dk_salaries_season/DKSalaries_NHL_season",
          os.path.expanduser("~/Desktop/DKSalaries_NHL_season")]:
    if glob.glob(os.path.join(p, "draftkings_NHL_*.csv")):
        DK_DIR = p
        break

def normalize_pos(p):
    p = str(p).upper()
    if p in ('L','R','LW','RW'): return 'W'
    if p in ('LD','RD'): return 'D'
    return p

# Load data
files = sorted(glob.glob(os.path.join(DK_DIR, "draftkings_NHL_*.csv")))
all_data = []
for f in files:
    basename = os.path.basename(f)
    suffix = basename.replace('draftkings_NHL_','').replace('.csv','')
    df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
    for col in ['Score','Salary','Avg','Ceiling','TeamGoal','OppGoal']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['slate_date'] = suffix
    df['clean_date'] = suffix[:10]
    all_data.append(df)
full = pd.concat(all_data, ignore_index=True)

# Select 25 mid/late-season slates for robust testing
slate_info = []
for f in files:
    basename = os.path.basename(f)
    suffix = basename.replace('draftkings_NHL_','').replace('.csv','')
    clean_date = suffix[:10]
    if clean_date < '2025-12-01': continue
    pool = full[(full['slate_date']==suffix) & full['Score'].notna() & full['Salary'].notna()]
    goalies = pool[pool['Pos']=='G']
    hist = full[full['Score'].notna() & (full['clean_date'] < clean_date)]
    if len(pool) >= 30 and len(goalies) >= 2 and len(hist) >= 15000:
        slate_info.append((clean_date, suffix))

optimizer = NHLLineupOptimizer()
N_SIMS = 5000
N_CANDIDATES = 100
results_per_slate = []
t_total = time.time()

for clean_date, suffix in slate_info:
    pool_raw = full[(full['slate_date']==suffix) & full['Score'].notna() & full['Salary'].notna()].copy()
    hist = full[full['Score'].notna() & (full['clean_date'] < clean_date)].copy()
    hist_for_sim = hist[['Player','Team','Score','Pos','slate_date']].copy()
    
    opt_pool = pool_raw.rename(columns={{
        'Player':'name','Team':'team','Pos':'position',
        'Salary':'salary','Avg':'projected_fpts'
    }}).copy()
    opt_pool['position'] = opt_pool['position'].apply(normalize_pos)
    opt_pool['projected_fpts'] = opt_pool['projected_fpts'].fillna(3.0)
    
    # Apply blend weight to projections
    if 'Avg' in pool_raw.columns:
        dk_avg = pd.to_numeric(pool_raw['Avg'], errors='coerce').fillna(3.0)
        blend = config.DK_AVG_BLEND_WEIGHT
        # Our model projection is projected_fpts; blend with DK avg
        opt_pool['projected_fpts'] = blend * opt_pool['projected_fpts'] + (1-blend) * dk_avg.values[:len(opt_pool)]
    
    # Generate candidates
    candidates = []
    capped = opt_pool[opt_pool['salary']<=7500].copy()
    pc = capped['position'].value_counts()
    has_capped = (pc.get('C',0)>=3 and pc.get('W',0)>=4 and pc.get('D',0)>=3 and pc.get('G',0)>=2)
    
    for rand in [0.05, 0.12, 0.20]:
        try:
            batch = optimizer.optimize_lineup(opt_pool, n_lineups=max(N_CANDIDATES//6,5), randomness=rand)
            if batch: candidates.extend(batch)
        except: pass
        if has_capped:
            try:
                batch = optimizer.optimize_lineup(capped, n_lineups=max(N_CANDIDATES//12,3), randomness=rand)
                if batch: candidates.extend(batch)
            except: pass
        try:
            batch = optimizer.optimize_lineup(opt_pool, n_lineups=max(N_CANDIDATES//12,3), randomness=rand, max_from_team=3)
            if batch: candidates.extend(batch)
        except: pass
    
    if len(candidates) < 15: continue
    
    score_map = dict(zip(pool_raw['Player']+'_'+pool_raw['Team'], pool_raw['Score']))
    actual_arr = np.array([
        sum(score_map.get(f"{{r['name']}}_{{r['team']}}",0) for _,r in lu.iterrows())
        for lu in candidates
    ])
    
    engine = SimulationEngine(n_sims=N_SIMS)
    engine.fit_player_distributions(opt_pool, hist_for_sim, date_str=suffix)
    
    sim_results = [engine.simulate_lineup(lu, n_sims=N_SIMS) for lu in candidates]
    sim_means = np.array([s['mean'] for s in sim_results])
    sim_stds = np.array([s['std'] for s in sim_results])
    m3s = sim_means + 3.0 * sim_stds
    
    m3s_idx = int(np.argmax(m3s))
    avg_actual = actual_arr.mean()
    
    results_per_slate.append({{
        'date': clean_date,
        'n_cands': len(candidates),
        'avg': avg_actual,
        'best': float(actual_arr.max()),
        'm3s_fpts': float(actual_arr[m3s_idx]),
        'm3s_edge': float(actual_arr[m3s_idx] - avg_actual),
        'm3s_pctile': float((actual_arr < actual_arr[m3s_idx]).sum() / len(actual_arr) * 100),
    }})

elapsed = time.time() - t_total
rdf = pd.DataFrame(results_per_slate)

# Compute summary
summary = {{
    'run_id': {run_id},
    'label': '{params["label"]}',
    'params': {{
        'dk_avg_blend': {params['dk_avg_blend']},
        'primary_stack_boost': {params['primary_stack_boost']},
        'goalie_corr_boost': {params['goalie_corr_boost']},
    }},
    'n_slates': len(rdf),
    'avg_m3s_edge': float(rdf['m3s_edge'].mean()) if len(rdf) > 0 else 0,
    'avg_m3s_pctile': float(rdf['m3s_pctile'].mean()) if len(rdf) > 0 else 0,
    'avg_m3s_fpts': float(rdf['m3s_fpts'].mean()) if len(rdf) > 0 else 0,
    'avg_best_possible': float(rdf['best'].mean()) if len(rdf) > 0 else 0,
    'avg_candidate_fpts': float(rdf['avg'].mean()) if len(rdf) > 0 else 0,
    'top25_rate': float((rdf['m3s_pctile']>=75).sum() / len(rdf)) if len(rdf) > 0 else 0,
    'bot25_rate': float((rdf['m3s_pctile']<25).sum() / len(rdf)) if len(rdf) > 0 else 0,
    'win_rate': float((rdf['m3s_edge']>0).sum() / len(rdf)) if len(rdf) > 0 else 0,
    'elapsed_seconds': elapsed,
    'per_slate': results_per_slate,
}}

out_path = '{RESULTS_DIR}/run_{run_id}.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Run {run_id} done: {{len(rdf)}} slates, M3σ edge={{rdf['m3s_edge'].mean():+.1f}}, "
      f"pctile={{rdf['m3s_pctile'].mean():.0f}}%, {{elapsed:.0f}}s")
'''
    
    script_path = RESULTS_DIR / f"worker_{run_id}.py"
    with open(script_path, 'w') as f:
        f.write(script)
    return script_path


def main():
    print(f"{'='*70}")
    print(f"  PARALLEL PARAMETER EXPLORATION")
    print(f"  {len(PARAM_COMBOS)} parameter combinations")
    print(f"{'='*70}")
    
    for i, combo in enumerate(PARAM_COMBOS):
        print(f"\n  Run {i}: {combo['label']}")
        print(f"    blend={combo['dk_avg_blend']} stack_boost={combo['primary_stack_boost']} "
              f"goalie_boost={combo['goalie_corr_boost']}")
    
    # Write worker scripts
    scripts = []
    for i, combo in enumerate(PARAM_COMBOS):
        script_path = write_worker_script(i, combo)
        scripts.append(script_path)
    
    # Launch all 5 in parallel
    print(f"\n  Launching {len(scripts)} parallel workers...")
    t0 = time.time()
    
    processes = []
    for i, script in enumerate(scripts):
        log_path = RESULTS_DIR / f"run_{i}.log"
        log_file = open(log_path, 'w')
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJ_DIR),
        )
        processes.append((proc, log_file, i))
        print(f"    Worker {i} launched (PID {proc.pid})")
    
    # Wait for all to complete
    print(f"\n  Waiting for workers to complete...")
    for proc, log_file, run_id in processes:
        proc.wait()
        log_file.close()
        status = "✓" if proc.returncode == 0 else f"✗ (exit {proc.returncode})"
        print(f"    Worker {run_id}: {status}")
    
    total_time = time.time() - t0
    print(f"\n  All workers done in {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # ═══════════════════════════════════════════════════════════════
    # COLLECT & COMPARE RESULTS
    # ═══════════════════════════════════════════════════════════════
    all_results = []
    for i in range(len(PARAM_COMBOS)):
        result_path = RESULTS_DIR / f"run_{i}.json"
        if result_path.exists():
            with open(result_path) as f:
                all_results.append(json.load(f))
        else:
            print(f"  ⚠ Missing results for run {i}")
            # Check log for errors
            log_path = RESULTS_DIR / f"run_{i}.log"
            if log_path.exists():
                with open(log_path) as f:
                    print(f"    Log: {f.read()[-500:]}")
    
    if not all_results:
        print("ERROR: No results collected!")
        return
    
    # Sort by M+3σ edge (primary metric)
    all_results.sort(key=lambda x: x['avg_m3s_edge'], reverse=True)
    
    print(f"\n\n{'='*90}")
    print(f"  PARAMETER EXPLORATION RESULTS — {all_results[0]['n_slates']} slates per run")
    print(f"{'='*90}")
    
    print(f"\n  {'Run':>3} {'Label':<40} {'Blend':>5} {'Stack':>5} {'Goalie':>6} "
          f"{'Edge':>7} {'Pctile':>7} {'Top25%':>6} {'Win%':>5}")
    print(f"  {'-'*90}")
    
    for r in all_results:
        p = r['params']
        marker = ' ◄ BEST' if r == all_results[0] else ''
        baseline = ' (baseline)' if r['run_id'] == 0 else ''
        print(f"  {r['run_id']:>3} {r['label']:<40} "
              f"{p['dk_avg_blend']:>5.2f} {p['primary_stack_boost']:>5.2f} {p['goalie_corr_boost']:>6.2f} "
              f"{r['avg_m3s_edge']:>+6.1f} {r['avg_m3s_pctile']:>6.0f}% "
              f"{r['top25_rate']:>5.0%} {r['win_rate']:>5.0%}{marker}{baseline}")
    
    # ═══════════════════════════════════════════════════════════════
    # BEST vs BASELINE COMPARISON
    # ═══════════════════════════════════════════════════════════════
    best = all_results[0]
    baseline = [r for r in all_results if r['run_id'] == 0][0]
    
    print(f"\n  {'─'*50}")
    print(f"  BEST: Run {best['run_id']} — {best['label']}")
    print(f"  {'─'*50}")
    print(f"    M+3σ edge:    {best['avg_m3s_edge']:>+6.1f} FPTS/slate "
          f"(baseline: {baseline['avg_m3s_edge']:>+6.1f}, delta: {best['avg_m3s_edge']-baseline['avg_m3s_edge']:>+.1f})")
    print(f"    Avg pctile:   {best['avg_m3s_pctile']:>6.0f}% "
          f"(baseline: {baseline['avg_m3s_pctile']:>6.0f}%)")
    print(f"    Top-25% rate: {best['top25_rate']:>6.0%} "
          f"(baseline: {baseline['top25_rate']:>6.0%})")
    print(f"    Win rate:     {best['win_rate']:>6.0%} "
          f"(baseline: {baseline['win_rate']:>6.0%})")
    
    bp = best['params']
    print(f"\n    Winning parameters:")
    print(f"      DK_AVG_BLEND_WEIGHT:        {bp['dk_avg_blend']} (was 0.80)")
    print(f"      PRIMARY_STACK_BOOST:         {bp['primary_stack_boost']} (was 0.20)")
    print(f"      GOALIE_CORRELATION_BOOST:    {bp['goalie_corr_boost']} (was 0.10)")
    
    # Save combined results
    combined_path = RESULTS_DIR / "combined_results.json"
    with open(combined_path, 'w') as f:
        json.dump({
            'best_run': best['run_id'],
            'best_params': best['params'],
            'best_edge': best['avg_m3s_edge'],
            'baseline_edge': baseline['avg_m3s_edge'],
            'all_runs': [{k: v for k, v in r.items() if k != 'per_slate'} for r in all_results],
        }, f, indent=2)
    
    print(f"\n  Results saved to {RESULTS_DIR}/")
    
    # ═══════════════════════════════════════════════════════════════
    # RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════
    delta = best['avg_m3s_edge'] - baseline['avg_m3s_edge']
    if delta > 1.0:
        print(f"\n  ✅ RECOMMENDATION: Update config.py with winning parameters ({delta:+.1f} edge improvement)")
    elif delta > 0:
        print(f"\n  ⚠ MARGINAL: Best is slightly better ({delta:+.1f}) — need more slates to confirm")
    else:
        print(f"\n  ✅ CURRENT DEFAULTS ARE OPTIMAL — no changes needed")


if __name__ == '__main__':
    main()
