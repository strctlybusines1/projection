#!/usr/bin/env python3
"""
Parameter Exploration v2 — with Diverse Candidate Generator
Tests old triple-mix vs diverse generator across best param combos.
"""
import json, os, sys, time, subprocess, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
PROJ_DIR = Path(__file__).parent
RESULTS_DIR = PROJ_DIR / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)

PARAM_COMBOS = [
    {'dk_avg_blend': 0.80, 'primary_stack_boost': 0.20, 'goalie_corr_boost': 0.10,
     'use_diverse': False, 'label': 'Triple-mix + 80/20 blend (BASELINE)'},
    {'dk_avg_blend': 0.90, 'primary_stack_boost': 0.20, 'goalie_corr_boost': 0.10,
     'use_diverse': False, 'label': 'Triple-mix + 90/10 blend'},
    {'dk_avg_blend': 0.80, 'primary_stack_boost': 0.30, 'goalie_corr_boost': 0.20,
     'use_diverse': False, 'label': 'Triple-mix + aggressive stack'},
    {'dk_avg_blend': 0.80, 'primary_stack_boost': 0.20, 'goalie_corr_boost': 0.10,
     'use_diverse': True, 'label': 'DIVERSE gen + 80/20 blend'},
    {'dk_avg_blend': 0.90, 'primary_stack_boost': 0.20, 'goalie_corr_boost': 0.10,
     'use_diverse': True, 'label': 'DIVERSE gen + 90/10 blend'},
    {'dk_avg_blend': 0.80, 'primary_stack_boost': 0.30, 'goalie_corr_boost': 0.20,
     'use_diverse': True, 'label': 'DIVERSE gen + aggressive stack'},
]

WORKER_TEMPLATE = '''#!/usr/bin/env python3
import sys, os, time, json, warnings, glob
import numpy as np, pandas as pd
from collections import Counter
warnings.filterwarnings('ignore')
sys.path.insert(0, '{proj_dir}')

import config
config.DK_AVG_BLEND_WEIGHT = {dk_avg_blend}
config.PRIMARY_STACK_BOOST = {primary_stack_boost}
config.GOALIE_CORRELATION_BOOST = {goalie_corr_boost}

from simulation_engine import SimulationEngine
from optimizer import NHLLineupOptimizer
{diverse_import}

USE_DIVERSE = {use_diverse}

def normalize_pos(p):
    p = str(p).upper()
    if p in ('L','R','LW','RW'): return 'W'
    if p in ('LD','RD'): return 'D'
    return p

DK_DIR = os.path.expanduser("~/Desktop/DKSalaries_NHL_season")
if not glob.glob(os.path.join(DK_DIR, "draftkings_NHL_*.csv")):
    DK_DIR = os.path.expanduser("~/dk_salaries_season/DKSalaries_NHL_season")

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

    opt_pool = pool_raw.rename(columns=dict(
        Player='name', Team='team', Pos='position', Salary='salary', Avg='projected_fpts'
    )).copy()
    opt_pool['position'] = opt_pool['position'].apply(normalize_pos)
    opt_pool['projected_fpts'] = opt_pool['projected_fpts'].fillna(3.0)

    if 'Avg' in pool_raw.columns:
        dk_avg = pd.to_numeric(pool_raw['Avg'], errors='coerce').fillna(3.0)
        blend = config.DK_AVG_BLEND_WEIGHT
        opt_pool['projected_fpts'] = blend * opt_pool['projected_fpts'] + (1 - blend) * dk_avg.values[:len(opt_pool)]

    if USE_DIVERSE:
        candidates = generate_diverse_candidates(opt_pool, optimizer, n_total=N_CANDIDATES, verbose=False)
    else:
        candidates = []
        capped = opt_pool[opt_pool['salary'] <= 7500].copy()
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
        sum(score_map.get(f"{{r['name']}}_{{r['team']}}", 0) for _, r in lu.iterrows())
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
    results_per_slate.append(dict(
        date=clean_date, n_cands=len(candidates), avg=avg_actual,
        best=float(actual_arr.max()),
        m3s_fpts=float(actual_arr[m3s_idx]),
        m3s_edge=float(actual_arr[m3s_idx] - avg_actual),
        m3s_pctile=float((actual_arr < actual_arr[m3s_idx]).sum() / len(actual_arr) * 100),
    ))

elapsed = time.time() - t_total
rdf = pd.DataFrame(results_per_slate)

summary = dict(
    run_id={run_id}, label='{label}',
    params=dict(dk_avg_blend={dk_avg_blend}, primary_stack_boost={primary_stack_boost},
                goalie_corr_boost={goalie_corr_boost}, use_diverse={use_diverse}),
    n_slates=len(rdf),
    avg_m3s_edge=float(rdf['m3s_edge'].mean()) if len(rdf) else 0,
    avg_m3s_pctile=float(rdf['m3s_pctile'].mean()) if len(rdf) else 0,
    avg_m3s_fpts=float(rdf['m3s_fpts'].mean()) if len(rdf) else 0,
    top25_rate=float((rdf['m3s_pctile']>=75).sum()/len(rdf)) if len(rdf) else 0,
    win_rate=float((rdf['m3s_edge']>0).sum()/len(rdf)) if len(rdf) else 0,
    elapsed_seconds=elapsed,
)
with open('{results_dir}/run_{run_id}.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Run {run_id}: {{len(rdf)}} slates, edge={{rdf['m3s_edge'].mean():+.1f}}, {{elapsed:.0f}}s")
'''

def main():
    print(f"{'='*70}")
    print(f"  PARAMETER EXPLORATION v2 — with Diverse Generator")
    print(f"  {len(PARAM_COMBOS)} runs across 61 slates each")
    print(f"{'='*70}")
    for i, c in enumerate(PARAM_COMBOS):
        gen = "DIVERSE" if c['use_diverse'] else "triple-mix"
        print(f"  Run {i}: [{gen}] blend={c['dk_avg_blend']} stack={c['primary_stack_boost']} goalie={c['goalie_corr_boost']}")

    # Write & launch workers
    processes = []
    for i, combo in enumerate(PARAM_COMBOS):
        diverse_import = "from candidate_generator import generate_diverse_candidates" if combo['use_diverse'] else ""
        script = WORKER_TEMPLATE.format(
            proj_dir=PROJ_DIR, run_id=i, label=combo['label'],
            dk_avg_blend=combo['dk_avg_blend'],
            primary_stack_boost=combo['primary_stack_boost'],
            goalie_corr_boost=combo['goalie_corr_boost'],
            use_diverse=combo['use_diverse'],
            diverse_import=diverse_import,
            results_dir=RESULTS_DIR,
        )
        script_path = RESULTS_DIR / f"worker_{i}.py"
        with open(script_path, 'w') as f:
            f.write(script)

        log = open(RESULTS_DIR / f"run_{i}.log", 'w')
        proc = subprocess.Popen([sys.executable, str(script_path)],
                                stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJ_DIR))
        processes.append((proc, log, i))
        print(f"  Launched worker {i} (PID {proc.pid})")

    print(f"\n  Waiting for {len(processes)} workers...")
    t0 = time.time()
    for proc, log, rid in processes:
        proc.wait()
        log.close()
        print(f"  Worker {rid}: {'✓' if proc.returncode==0 else f'✗ exit {proc.returncode}'}")
    print(f"  Done in {time.time()-t0:.0f}s")

    # Collect results
    all_results = []
    for i in range(len(PARAM_COMBOS)):
        path = RESULTS_DIR / f"run_{i}.json"
        if path.exists():
            with open(path) as f:
                all_results.append(json.load(f))
        else:
            log_path = RESULTS_DIR / f"run_{i}.log"
            print(f"  ⚠ Missing run {i}", open(log_path).read()[-300:] if log_path.exists() else "")

    all_results.sort(key=lambda x: x['avg_m3s_edge'], reverse=True)

    print(f"\n{'='*95}")
    print(f"  RESULTS — sorted by M+3σ edge")
    print(f"{'='*95}")
    print(f"  {'Run':>3} {'Label':<42} {'Edge':>7} {'Pctile':>7} {'Top25%':>6} {'Win%':>5}")
    print(f"  {'-'*75}")
    for r in all_results:
        best = ' ◄' if r == all_results[0] else ''
        print(f"  {r['run_id']:>3} {r['label']:<42} {r['avg_m3s_edge']:>+6.1f} "
              f"{r['avg_m3s_pctile']:>6.0f}% {r['top25_rate']:>5.0%} {r['win_rate']:>5.0%}{best}")

    # Diverse vs Triple-mix summary
    diverse_runs = [r for r in all_results if r['params'].get('use_diverse')]
    triple_runs = [r for r in all_results if not r['params'].get('use_diverse')]
    if diverse_runs and triple_runs:
        d_avg = sum(r['avg_m3s_edge'] for r in diverse_runs) / len(diverse_runs)
        t_avg = sum(r['avg_m3s_edge'] for r in triple_runs) / len(triple_runs)
        print(f"\n  GENERATOR COMPARISON:")
        print(f"    Triple-mix avg edge: {t_avg:+.1f}")
        print(f"    Diverse gen avg edge: {d_avg:+.1f}")
        print(f"    Diverse delta: {d_avg - t_avg:+.1f} FPTS/slate")

    with open(RESULTS_DIR / "combined_v2.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    main()
