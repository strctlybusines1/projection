#!/usr/bin/env python3
"""
Multi-Agent Blend Experiment
============================

Tests 3 specialist projection strategies against historical actuals,
then finds the optimal blend weights to minimize MAE.

Agents (philosophies):
    1. BASELINE — Current production settings (control group)
    2. MATCHUP_HEAVY — Cranks up matchup sensitivity, widens swing caps
    3. RECENCY — Weights recent form heavily, reduces season-avg anchoring
    4. VEGAS_DERIVED — Lets implied team totals drive individual projections

Each agent re-projects all players for each historical date using different
config overrides, then we:
    a) Score each agent's MAE per slate
    b) Find optimal blend weights via least-squares on holdout slates
    c) Show where each agent excels and where it fails

Usage:
    python blend_experiment.py                          # Run full experiment
    python blend_experiment.py --dates 2026-01-23       # Single date
    python blend_experiment.py --agent matchup           # Test one agent
    python blend_experiment.py --optimize                # Find blend weights

Requires: backtests/batch_backtest_details.csv (actuals)
          Vegas_Historical.csv (historical odds)
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent

# ================================================================
#  Agent Configurations
# ================================================================

# These are the config overrides each "agent" applies.
# The baseline agent uses production defaults (no overrides).

AGENT_CONFIGS = {
    'baseline': {
        'description': 'Production settings (control)',
        'overrides': {},
    },

    'matchup_heavy': {
        'description': 'Amplified matchup sensitivity — lets opponent quality swing projections more',
        'overrides': {
            # projections.py
            'MAX_MULTIPLICATIVE_SWING': 0.25,       # Was 0.15 — allow ±25% swing
            'GLOBAL_BIAS_CORRECTION': 0.75,          # Was 0.80 — slightly less dampening
            # config.py
            'SIGNAL_COMPOSITE_SENSITIVITY': 0.50,    # Was 0.30 — matchups matter 67% more
            'SIGNAL_COMPOSITE_CLIP_LOW': 0.85,       # Was 0.92 — wider downside
            'SIGNAL_COMPOSITE_CLIP_HIGH': 1.15,      # Was 1.08 — wider upside
        },
    },

    'recency': {
        'description': 'Recent form dominates — less anchoring to season averages',
        'overrides': {
            # projections.py
            'GLOBAL_BIAS_CORRECTION': 0.88,          # Was 0.80 — trust raw projection more
            'SKATER_HIGH_PROJ_BLEND': 0.92,          # Was 0.80 — less mean reversion
            'GOALIE_HIGH_PROJ_BLEND': 0.92,          # Was 0.80
            # config.py
            'DK_AVG_BLEND_WEIGHT': 0.65,             # Was 0.80 — season avg gets less say
        },
    },

    'vegas_derived': {
        'description': 'Vegas-anchored — scale projections by team implied total deviation',
        'overrides': {
            # projections.py
            'GLOBAL_BIAS_CORRECTION': 0.70,          # Was 0.80 — more aggressive dampening
            'MAX_MULTIPLICATIVE_SWING': 0.20,        # Was 0.15 — allow Vegas to push more
            # config.py
            'DK_AVG_BLEND_WEIGHT': 0.75,             # Was 0.80
            # Special flag — agent applies Vegas team total scaling in post-processing
            '_VEGAS_SCALE': True,
        },
    },
}


# ================================================================
#  Data Loading
# ================================================================

def load_actuals() -> pd.DataFrame:
    """Load actual FPTS from batch backtest."""
    path = PROJECT_ROOT / 'backtests' / 'batch_backtest_details.csv'
    if not path.exists():
        print(f"  ✗ Missing: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    # Standardize
    df = df.rename(columns={'error': 'orig_error'})
    return df


def load_vegas() -> pd.DataFrame:
    """Load historical Vegas data."""
    # Check multiple locations
    for loc in [PROJECT_ROOT / 'Vegas_Historical.csv',
                PROJECT_ROOT / 'vegas' / 'Vegas_Historical.csv',
                Path('/mnt/user-data/uploads/Vegas_Historical.csv')]:
        if loc.exists():
            df = pd.read_csv(loc, encoding='utf-8-sig')
            # Parse date: "1.23.26" → "2026-01-23"
            def parse_date(d):
                parts = d.strip().split('.')
                if len(parts) == 3:
                    m, d, y = parts
                    return f"20{y}-{int(m):02d}-{int(d):02d}"
                return d
            df['date'] = df['Date'].apply(parse_date)
            df['win_pct'] = df['Win %'].str.rstrip('%').astype(float) / 100
            return df
    print("  ✗ Vegas_Historical.csv not found")
    return pd.DataFrame()


def load_projection_csv(date_str: str) -> Optional[pd.DataFrame]:
    """Load the latest projection CSV for a given date."""
    proj_dir = PROJECT_ROOT / 'daily_projections'
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    prefixes = [
        f"{dt.month:02d}_{dt.day:02d}_{dt.strftime('%y')}",
        f"{dt.month}_{dt.day}_{dt.strftime('%y')}",
    ]

    matches = []
    for f in sorted(proj_dir.glob('*NHLprojections_*.csv')):
        if '_lineups' in f.name:
            continue
        for prefix in prefixes:
            if f.name.startswith(prefix):
                matches.append(f)
                break

    if not matches:
        return None
    return pd.read_csv(matches[-1])


# ================================================================
#  Agent Projection Modifier
# ================================================================

def apply_agent_overrides(projections: pd.DataFrame, agent_name: str,
                          vegas: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Apply an agent's philosophy to modify base projections.

    Each agent applies DIRECT adjustments to the baseline projected_fpts.
    This avoids reverse-engineering the pipeline math.
    """
    cfg = AGENT_CONFIGS[agent_name]
    overrides = cfg['overrides']
    df = projections.copy()

    if agent_name == 'baseline':
        return df

    skater_mask = df['position'] != 'G'
    goalie_mask = df['position'] == 'G'

    if agent_name == 'matchup_heavy':
        # Amplify edge signal — players with positive edge get boosted more,
        # negative edge get penalized more
        if 'edge' in df.columns:
            edge = df['edge'].fillna(0)
            # Baseline edge is already baked in. Apply ADDITIONAL matchup effect.
            # Scale: 0.67x more matchup impact (sensitivity 0.50 vs 0.30 baseline)
            additional_factor = 0.67
            mean_fpts = df.loc[skater_mask, 'projected_fpts'].mean()
            edge_pct = edge / df['projected_fpts'].clip(1)
            boost = edge_pct * additional_factor
            # Wider clip (±25% vs ±15%)
            boost = boost.clip(-0.10, 0.10)
            df.loc[skater_mask, 'projected_fpts'] *= (1 + boost[skater_mask])

    elif agent_name == 'recency':
        # Trust raw projection more, reduce mean-reversion anchoring
        # Effect: high projections go higher, low projections go lower
        if 'dk_avg_fpts' in df.columns:
            dk_avg = df['dk_avg_fpts'].fillna(df['projected_fpts'])
            # Current pipeline blends 80% proj + 20% dk_avg
            # Recency agent: shift toward 92% proj + 8% dk_avg
            # Net effect: proj += 0.12 * (proj - dk_avg)
            shift = 0.12 * (df['projected_fpts'] - dk_avg)
            df.loc[skater_mask, 'projected_fpts'] += shift[skater_mask]

        # Also reduce global bias correction (let projections run hotter)
        # Baseline has 0.80 correction; recency uses 0.88
        # Net effect: multiply by 0.88/0.80 = 1.10
        df.loc[skater_mask, 'projected_fpts'] *= 1.10
        # But cap to avoid runaway — goalie correction stays
        df.loc[goalie_mask, 'projected_fpts'] *= 1.05

    elif agent_name == 'vegas_derived':
        # Scale projections by team implied total vs league average
        if not vegas.empty:
            date_vegas = vegas[vegas['date'] == date_str]
            if not date_vegas.empty:
                team_totals = dict(zip(date_vegas['Team'], date_vegas['TeamGoal']))
                if team_totals:
                    avg_total = np.mean(list(team_totals.values()))
                    for team, total in team_totals.items():
                        team_skater_mask = (df['team'] == team) & skater_mask
                        if team_skater_mask.any():
                            # Scale factor: team implied / league avg for this slate
                            raw_scale = total / avg_total
                            # Damped: don't let Vegas swing more than ±12%
                            damped = 1.0 + (raw_scale - 1.0) * 0.6
                            damped = max(0.88, min(1.12, damped))
                            df.loc[team_skater_mask, 'projected_fpts'] *= damped

                        # Goalies: opposing team's total affects goalie negatively
                        team_goalie_mask = (df['team'] == team) & goalie_mask
                        if team_goalie_mask.any():
                            # Find opponent's implied total
                            opp_row = date_vegas[
                                (date_vegas['Opp'].str.contains(team, na=False)) &
                                (date_vegas['Team'] != team)
                            ]
                            if not opp_row.empty:
                                opp_total = opp_row.iloc[0]['TeamGoal']
                                opp_scale = avg_total / opp_total  # Inverted: fewer opponent goals = better for goalie
                                g_damped = 1.0 + (opp_scale - 1.0) * 0.4
                                g_damped = max(0.90, min(1.10, g_damped))
                                df.loc[team_goalie_mask, 'projected_fpts'] *= g_damped

        # Also slightly more aggressive dampening
        df.loc[skater_mask, 'projected_fpts'] *= 0.875  # 0.70/0.80 ratio

    return df


# ================================================================
#  Scoring
# ================================================================

def score_agent(agent_proj: pd.DataFrame, actuals: pd.DataFrame,
                date_str: str) -> Dict:
    """Match agent's projections to actuals and compute MAE, bias, correlation."""
    act = actuals[actuals['date'] == date_str].copy()
    if act.empty:
        return {'mae': None, 'bias': None, 'corr': None, 'n': 0}

    # Actuals use "J. Eichel", projections use "Jack Eichel"
    # Match on last name + team
    def last_name(n):
        parts = n.strip().split()
        return parts[-1].lower() if parts else ''

    agent_proj = agent_proj.copy()
    agent_proj['_last'] = agent_proj['name'].apply(last_name)
    agent_proj['_key'] = agent_proj['_last'] + '_' + agent_proj['team'].str.lower()

    act['_last'] = act['name'].apply(last_name)
    act['_key'] = act['_last'] + '_' + act['team'].str.lower()

    # Merge on last name + team (handles dupes by taking first)
    proj_deduped = agent_proj.drop_duplicates('_key')[['_key', 'projected_fpts']].rename(
        columns={'projected_fpts': 'agent_proj'}
    )
    merged = act.merge(proj_deduped, on='_key', how='inner')

    if merged.empty:
        return {'mae': None, 'bias': None, 'corr': None, 'n': 0}

    # Use agent projection, not the original one baked into actuals
    merged['projected_fpts'] = merged['agent_proj']
    merged['error'] = merged['agent_proj'] - merged['actual_fpts']
    merged['abs_error'] = merged['error'].abs()

    skaters = merged[merged['position'] != 'G']
    goalies = merged[merged['position'] == 'G']

    result = {
        'mae': merged['abs_error'].mean(),
        'bias': merged['error'].mean(),
        'n': len(merged),
    }

    if len(merged) > 3:
        result['corr'] = merged[['projected_fpts', 'actual_fpts']].corr().iloc[0, 1]
    else:
        result['corr'] = None

    if not skaters.empty:
        result['sk_mae'] = skaters['abs_error'].mean()
        result['sk_bias'] = skaters['error'].mean()
    if not goalies.empty:
        result['g_mae'] = goalies['abs_error'].mean()
        result['g_bias'] = goalies['error'].mean()

    result['details'] = merged
    return result


# ================================================================
#  Blend Optimization
# ================================================================

def find_optimal_blend(agent_projections: Dict[str, Dict[str, pd.DataFrame]],
                       actuals: pd.DataFrame,
                       dates: List[str]) -> Dict[str, float]:
    """
    Find optimal blend weights across agents using grid search.

    For each player on each date, the blended projection is:
        blend_proj = w1*agent1 + w2*agent2 + w3*agent3 + w4*agent4

    We search weight combinations that minimize overall MAE.
    """
    # Build player-level projection matrix
    records = []
    agent_names = list(agent_projections.keys())

    def last_name(n):
        parts = n.strip().split()
        return parts[-1].lower() if parts else ''

    for date_str in dates:
        act = actuals[actuals['date'] == date_str]
        if act.empty:
            continue

        act_clean = act.copy()
        act_clean['_key'] = act_clean['name'].apply(last_name) + '_' + act_clean['team'].str.lower()

        for _, arow in act_clean.iterrows():
            rec = {
                'date': date_str,
                'name': arow['name'],
                'actual': arow['actual_fpts'],
                'position': arow['position'],
            }

            all_found = True
            for agent_name in agent_names:
                agent_df = agent_projections[agent_name].get(date_str)
                if agent_df is None:
                    all_found = False
                    break
                agent_df = agent_df.copy()
                agent_df['_key'] = agent_df['name'].apply(last_name) + '_' + agent_df['team'].str.lower()
                match = agent_df[agent_df['_key'] == arow['_key']]
                if match.empty:
                    all_found = False
                    break
                rec[f'proj_{agent_name}'] = match.iloc[0]['projected_fpts']

            if all_found:
                records.append(rec)

    if not records:
        print("  ✗ Could not build blend matrix — no overlapping players")
        return {name: 1.0 / len(agent_names) for name in agent_names}

    blend_df = pd.DataFrame(records)
    print(f"\n  Blend matrix: {len(blend_df)} player-date observations")

    # Grid search over weight combinations
    best_mae = float('inf')
    best_weights = None
    n_agents = len(agent_names)

    # Generate weight grid (step=0.05, sum to 1.0)
    step = 0.05
    from itertools import product
    weight_range = np.arange(0, 1.01, step)

    if n_agents == 4:
        # For 4 agents, use coarser grid to stay tractable
        weight_range = np.arange(0, 1.01, 0.1)

    for combo in product(weight_range, repeat=n_agents - 1):
        remaining = 1.0 - sum(combo)
        if remaining < -0.001 or remaining > 1.001:
            continue
        weights = list(combo) + [max(0, remaining)]

        # Compute blended projection
        blended = np.zeros(len(blend_df))
        for i, agent_name in enumerate(agent_names):
            blended += weights[i] * blend_df[f'proj_{agent_name}'].values

        mae = np.mean(np.abs(blended - blend_df['actual'].values))
        if mae < best_mae:
            best_mae = mae
            best_weights = dict(zip(agent_names, weights))

    return best_weights


# ================================================================
#  Main Experiment
# ================================================================

def run_experiment(target_dates: List[str] = None, target_agent: str = None):
    """Run the full blend experiment."""
    print("\n" + "=" * 72)
    print("  MULTI-AGENT BLEND EXPERIMENT")
    print("=" * 72)

    # Load data
    actuals = load_actuals()
    vegas = load_vegas()
    available_dates = sorted(actuals['date'].unique())

    if target_dates:
        available_dates = [d for d in available_dates if d in target_dates]

    print(f"\n  Dates with actuals: {available_dates}")
    print(f"  Total actual observations: {len(actuals)}")
    if not vegas.empty:
        print(f"  Vegas data: {len(vegas)} team-date rows")

    # Determine which agents to test
    if target_agent:
        agents_to_test = ['baseline', target_agent]
    else:
        agents_to_test = list(AGENT_CONFIGS.keys())

    print(f"  Agents: {agents_to_test}")

    # Run each agent on each date
    all_results = {}
    all_projections = {name: {} for name in agents_to_test}

    for date_str in available_dates:
        base_proj = load_projection_csv(date_str)
        if base_proj is None:
            print(f"\n  ⚠ No projection CSV for {date_str} — skipping")
            continue

        print(f"\n  {'─' * 50}")
        print(f"  DATE: {date_str} ({len(base_proj)} players projected)")

        for agent_name in agents_to_test:
            cfg = AGENT_CONFIGS[agent_name]
            agent_proj = apply_agent_overrides(base_proj, agent_name, vegas, date_str)
            all_projections[agent_name][date_str] = agent_proj

            result = score_agent(agent_proj, actuals, date_str)
            if result['mae'] is not None:
                all_results.setdefault(agent_name, []).append({
                    'date': date_str,
                    **result,
                })

                mae_str = f"{result['mae']:.2f}"
                bias_str = f"{result['bias']:+.2f}"
                corr_str = f"{result['corr']:.3f}" if result['corr'] else "N/A"
                n_str = f"{result['n']}"
                print(f"    {agent_name:<16} MAE: {mae_str:>6}  "
                      f"Bias: {bias_str:>6}  Corr: {corr_str:>6}  "
                      f"({n_str} matched)")

    # ── Summary Table ──
    print(f"\n\n{'=' * 72}")
    print("  AGENT PERFORMANCE SUMMARY")
    print(f"{'=' * 72}")
    print(f"\n  {'Agent':<16} {'Avg MAE':>8} {'Avg Bias':>9} {'Avg Corr':>9} {'Slates':>7} {'Best On':>8}")
    print(f"  {'─' * 60}")

    agent_avg_mae = {}
    for agent_name in agents_to_test:
        results = all_results.get(agent_name, [])
        if not results:
            continue
        maes = [r['mae'] for r in results if r['mae'] is not None]
        biases = [r['bias'] for r in results if r['bias'] is not None]
        corrs = [r['corr'] for r in results if r['corr'] is not None]

        avg_mae = np.mean(maes) if maes else float('inf')
        avg_bias = np.mean(biases) if biases else 0
        avg_corr = np.mean(corrs) if corrs else 0
        agent_avg_mae[agent_name] = avg_mae

        # Count dates where this agent had the best MAE
        best_count = 0
        for r in results:
            date = r['date']
            date_maes = {a: next((x['mae'] for x in all_results.get(a, []) if x['date'] == date), float('inf'))
                        for a in agents_to_test}
            if date_maes.get(agent_name, float('inf')) == min(date_maes.values()):
                best_count += 1

        print(f"  {agent_name:<16} {avg_mae:>8.3f} {avg_bias:>+9.3f} {avg_corr:>9.3f} "
              f"{len(maes):>7} {best_count:>8}")

    # ── Per-Date Winner ──
    print(f"\n  {'─' * 50}")
    print(f"  PER-DATE WINNERS:")
    for date_str in available_dates:
        date_maes = {}
        for agent_name in agents_to_test:
            for r in all_results.get(agent_name, []):
                if r['date'] == date_str and r['mae'] is not None:
                    date_maes[agent_name] = r['mae']

        if date_maes:
            winner = min(date_maes, key=date_maes.get)
            runner_up = sorted(date_maes.items(), key=lambda x: x[1])
            mae_spread = runner_up[-1][1] - runner_up[0][1] if len(runner_up) > 1 else 0
            print(f"    {date_str}: {winner:<16} (MAE {date_maes[winner]:.3f}, "
                  f"spread: {mae_spread:.3f})")

    # ── Blend Optimization ──
    if len(agents_to_test) >= 3 and len(available_dates) >= 3:
        print(f"\n\n{'=' * 72}")
        print("  OPTIMAL BLEND WEIGHTS")
        print(f"{'=' * 72}")

        best_weights = find_optimal_blend(all_projections, actuals, available_dates)

        print(f"\n  Optimal weights (minimize MAE across all slates):")
        for agent_name, weight in sorted(best_weights.items(), key=lambda x: -x[1]):
            bar = '█' * int(weight * 40) + '░' * (40 - int(weight * 40))
            print(f"    {agent_name:<16} {bar} {weight:.2f}")

        # Compute blended MAE
        blend_records = []
        for date_str in available_dates:
            act = actuals[actuals['date'] == date_str]
            act_clean = act.copy()

            def _ln(n):
                parts = n.strip().split()
                return parts[-1].lower() if parts else ''

            act_clean['_key'] = act_clean['name'].apply(_ln) + '_' + act_clean['team'].str.lower()

            for _, arow in act_clean.iterrows():
                blended = 0
                all_found = True
                for agent_name, weight in best_weights.items():
                    agent_df = all_projections[agent_name].get(date_str)
                    if agent_df is None:
                        all_found = False
                        break
                    agent_df = agent_df.copy()
                    agent_df['_key'] = agent_df['name'].apply(_ln) + '_' + agent_df['team'].str.lower()
                    match = agent_df[agent_df['_key'] == arow['_key']]
                    if match.empty:
                        all_found = False
                        break
                    blended += weight * match.iloc[0]['projected_fpts']

                if all_found:
                    blend_records.append({
                        'actual': arow['actual_fpts'],
                        'blended': blended,
                    })

        if blend_records:
            br = pd.DataFrame(blend_records)
            blend_mae = (br['blended'] - br['actual']).abs().mean()
            blend_bias = (br['blended'] - br['actual']).mean()
            blend_corr = br[['blended', 'actual']].corr().iloc[0, 1]

            baseline_mae = agent_avg_mae.get('baseline', float('inf'))
            improvement = baseline_mae - blend_mae

            print(f"\n  BLENDED RESULT:")
            print(f"    MAE:         {blend_mae:.3f}")
            print(f"    Bias:        {blend_bias:+.3f}")
            print(f"    Correlation: {blend_corr:.3f}")
            print(f"    vs Baseline: {improvement:+.3f} MAE ({'BETTER' if improvement > 0 else 'WORSE'})")

            if improvement > 0:
                pct = (improvement / baseline_mae) * 100
                print(f"    Improvement: {pct:.1f}%")
            print()

    # ── Agent Philosophy Insights ──
    print(f"\n{'=' * 72}")
    print("  INSIGHTS")
    print(f"{'=' * 72}")

    if agent_avg_mae:
        best_agent = min(agent_avg_mae, key=agent_avg_mae.get)
        worst_agent = max(agent_avg_mae, key=agent_avg_mae.get)
        print(f"\n  Best single agent: {best_agent} (MAE {agent_avg_mae[best_agent]:.3f})")
        print(f"  Worst single agent: {worst_agent} (MAE {agent_avg_mae[worst_agent]:.3f})")
        spread = agent_avg_mae[worst_agent] - agent_avg_mae[best_agent]
        print(f"  Agent spread: {spread:.3f} MAE points")

        if spread < 0.1:
            print(f"\n  → Small agent spread suggests your baseline is already well-tuned.")
            print(f"    Blending may help at the margins but won't be transformative.")
        elif spread > 0.3:
            print(f"\n  → Large agent spread — significant room for improvement.")
            print(f"    The winning philosophy should be studied and integrated.")
        else:
            print(f"\n  → Moderate spread — blending likely captures meaningful gains.")

    print()


# ================================================================
#  CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Agent Blend Experiment')
    parser.add_argument('--dates', nargs='+', help='Specific dates to test (YYYY-MM-DD)')
    parser.add_argument('--agent', type=str, choices=list(AGENT_CONFIGS.keys()),
                       help='Test a specific agent vs baseline')
    parser.add_argument('--optimize', action='store_true',
                       help='Only run blend optimization (skip individual reports)')
    args = parser.parse_args()

    run_experiment(target_dates=args.dates, target_agent=args.agent)
