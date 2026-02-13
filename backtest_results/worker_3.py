#!/usr/bin/env python3
"""Worker for parameter exploration run 3"""
import sys, os, time, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/brendanhorlbeck/Desktop/Code/projection')

# Override config params BEFORE importing anything that reads them
import config
config.DK_AVG_BLEND_WEIGHT = 0.8
config.PRIMARY_STACK_BOOST = 0.3
config.GOALIE_CORRELATION_BOOST = 0.2

from simulation_engine import SimulationEngine
from optimizer import NHLLineupOptimizer
import glob

DK_DIR = "/Users/brendanhorlbeck/Desktop/dk_salaries_season/DKSalaries_NHL_season"
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
    
    opt_pool = pool_raw.rename(columns={
        'Player':'name','Team':'team','Pos':'position',
        'Salary':'salary','Avg':'projected_fpts'
    }).copy()
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
        sum(score_map.get(f"{r['name']}_{r['team']}",0) for _,r in lu.iterrows())
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
    
    results_per_slate.append({
        'date': clean_date,
        'n_cands': len(candidates),
        'avg': avg_actual,
        'best': float(actual_arr.max()),
        'm3s_fpts': float(actual_arr[m3s_idx]),
        'm3s_edge': float(actual_arr[m3s_idx] - avg_actual),
        'm3s_pctile': float((actual_arr < actual_arr[m3s_idx]).sum() / len(actual_arr) * 100),
    })

elapsed = time.time() - t_total
rdf = pd.DataFrame(results_per_slate)

# Compute summary
summary = {
    'run_id': 3,
    'label': 'Aggressive stacking (+30% stack, +20% goalie)',
    'params': {
        'dk_avg_blend': 0.8,
        'primary_stack_boost': 0.3,
        'goalie_corr_boost': 0.2,
    },
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
}

out_path = '/Users/brendanhorlbeck/Desktop/Code/projection/backtest_results/run_3.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Run 3 done: {len(rdf)} slates, M3σ edge={rdf['m3s_edge'].mean():+.1f}, "
      f"pctile={rdf['m3s_pctile'].mean():.0f}%, {elapsed:.0f}s")
