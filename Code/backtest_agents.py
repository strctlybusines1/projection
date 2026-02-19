#!/usr/bin/env python3
"""
Simons Agent Panel Backtest
==============================

Runs the 3-agent panel on historical slates to measure whether
agents add value over pure Sim M+3σ selection.

For each slate:
1. Generate candidates + run sim engine (same as backtest_sim.py)
2. Build SlateContext from historical DK data (Vegas, goalies, B2B, Own%)
3. Run 3 agents (via Claude API or offline heuristic fallback)
4. Compare: sim-only pick vs agent-consensus pick vs best possible

Key metric: agent_delta = agent_pick_fpts - sim_pick_fpts
After 20 slates, if mean(agent_delta) < 0, agents get cut.

Usage:
    # With Claude API (automated agents)
    ANTHROPIC_API_KEY=sk-... python backtest_agents.py

    # Without API (heuristic agents — rule-based, no LLM)
    python backtest_agents.py

    # Single date
    python backtest_agents.py --date 2026-02-05

    # More candidates/sims
    python backtest_agents.py --candidates 150 --sims 8000
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

PROJ_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR))

from simulation_engine import SimulationEngine
from sim_selector import SimSelector, _load_payout_curve
from optimizer import NHLLineupOptimizer
from agent_panel import AgentPanel, SlateContext, log_slate_result, print_agent_tracker_summary

# DK season files — update this path to match your machine
DK_DIR = Path.home() / "Desktop" / "DKSalaries_NHL_season"

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
    """Load all DK season files."""
    search_paths = [
        DK_DIR,
        Path.home() / "dk_salaries_season" / "DKSalaries_NHL_season",
        Path("/home/claude/dk_salaries_season/DKSalaries_NHL_season"),
        PROJ_DIR.parent.parent / "dk_salaries_season" / "DKSalaries_NHL_season",
    ]
    files = []
    for sp in search_paths:
        files = sorted(glob.glob(str(sp / "draftkings_NHL_*.csv")))
        if files:
            print(f"  Found DK files at: {sp}")
            break

    all_data = []
    for f in files:
        basename = os.path.basename(f)
        suffix = basename.replace('draftkings_NHL_', '').replace('.csv', '')
        try:
            df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
            for col in ['Score', 'Salary', 'Avg', 'Ceiling', 'TeamGoal', 'OppGoal']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['slate_date'] = suffix
            df['clean_date'] = suffix[:10]
            all_data.append(df)
        except Exception:
            pass

    if not all_data:
        print("ERROR: No DK season files found.")
        print(f"  Searched: {[str(sp) for sp in search_paths]}")
        sys.exit(1)

    full = pd.concat(all_data, ignore_index=True)
    print(f"  Loaded {len(full)} rows from {len(all_data)} files")
    return full


def build_slate_context(pool_raw, full_df, target_date, dk_date):
    """
    Build SlateContext from historical DK data.
    This replicates what you'd know BEFORE lock on that date.
    """
    ctx = SlateContext()

    # Vegas lines from DK file
    games_seen = set()
    for _, row in pool_raw.iterrows():
        team = row.get('Team', '')
        opp = str(row.get('Opp', ''))
        # Clean up Opp field (sometimes has "@ " or "vs " prefix)
        opp_clean = opp.replace('@ ', '').replace('vs ', '').strip()
        game_key = tuple(sorted([team, opp_clean]))
        if game_key not in games_seen and row.get('Total') and pd.notna(row['Total']):
            total = float(row['Total'])
            # Determine home/away from Opp prefix
            if opp.startswith('@'):
                ctx.add_vegas(home=opp_clean, away=team, total=total)
            else:
                ctx.add_vegas(home=team, away=opp_clean, total=total)
            games_seen.add(game_key)

    # Goalies — from DK pool (all goalies listed, whether starting or not)
    goalies = pool_raw[pool_raw['Pos'] == 'G'].copy()
    for _, g in goalies.iterrows():
        opp = str(g.get('Opp', '?')).replace('@ ', '').replace('vs ', '').strip()
        ctx.add_goalie(g['Player'], g['Team'], opp)

    # Back-to-back detection: find teams that played the day before
    prev_date = pd.to_datetime(target_date) - pd.Timedelta(days=1)
    prev_str = prev_date.strftime('%Y-%m-%d')
    prev_games = full_df[(full_df['clean_date'] == prev_str) & full_df['Score'].notna()]
    if len(prev_games) > 0:
        prev_teams = set(prev_games['Team'].unique())
        curr_teams = set(pool_raw[pool_raw['Score'].notna()]['Team'].unique())
        b2b_teams = prev_teams & curr_teams
        for t in sorted(b2b_teams):
            ctx.add_b2b(t)

    # Ownership from DK file
    own_col = None
    for c in ['Own%', 'Ownership', 'Own']:
        if c in pool_raw.columns:
            own_col = c
            break
    if own_col:
        own_map = {}
        for _, row in pool_raw.iterrows():
            val = row[own_col]
            if pd.notna(val):
                try:
                    own_map[row['Player']] = float(str(val).replace('%', ''))
                except (ValueError, TypeError):
                    pass
        ctx.set_ownership(own_map)

    # Injuries from DK 'Inj' column
    if 'Inj' in pool_raw.columns:
        for _, row in pool_raw.iterrows():
            inj = row.get('Inj', '')
            if pd.notna(inj) and str(inj).strip():
                ctx.add_injury(row['Player'], row['Team'], str(inj).strip())

    ctx.contest_type = 'se_gpp'
    return ctx


# ═══════════════════════════════════════════════════════════════════
#  Heuristic Agents (no LLM required)
# ═══════════════════════════════════════════════════════════════════

def heuristic_contrarian(lineup, ctx, sim_result):
    """
    Rule-based Contrarian: flag high-chalk lineups with no differentiation.
    """
    name_col = 'name' if 'name' in lineup.columns else 'Player'
    high_own = 0
    low_own = 0
    for _, row in lineup.iterrows():
        own = ctx.ownership.get(row[name_col], 0)
        if own > 25:
            high_own += 1
        if own < 5 and own > 0:
            low_own += 1

    if high_own >= 3 and low_own == 0:
        return 'FLAG', f'{high_own} players >25% owned, no contrarian plays', 0.7
    if high_own >= 4:
        return 'FLAG', f'{high_own} players >25% owned — very chalky', 0.8
    return 'APPROVE', '', 0.5


def heuristic_narrative(lineup, ctx, sim_result):
    """
    Rule-based Narrative: flag B2B stacks, injured goalies.
    CONSERVATIVE: only flag when it's a structural problem, not just presence.
    """
    name_col = 'name' if 'name' in lineup.columns else 'Player'
    team_col = 'team' if 'team' in lineup.columns else 'Team'
    pos_col = 'position' if 'position' in lineup.columns else 'Pos'

    # Only flag B2B if we're STACKING that B2B team (4+ players)
    team_counts = lineup[team_col].value_counts()
    for team, count in team_counts.items():
        if team in ctx.back_to_back and count >= 4:
            return 'FLAG', f'{count}-man stack on {team} which is on B2B — high bust risk', 0.7

    # Flag B2B GOALIE specifically (goalie is highest-leverage single player)
    goalie = lineup[lineup[pos_col] == 'G']
    if len(goalie) > 0:
        g_team = goalie.iloc[0][team_col]
        if g_team in ctx.back_to_back:
            return 'FLAG', f'Goalie {goalie.iloc[0][name_col]} on B2B team {g_team} — backup risk', 0.8

    # Flag injured goalie (critical — goalie is 15-20% of lineup outcome)
    if len(goalie) > 0:
        g_name = goalie.iloc[0][name_col]
        for inj in ctx.injuries:
            if inj['name'] == g_name:
                return 'FLAG', f'Goalie {g_name} listed as injured: {inj["status"]}', 0.9

    return 'APPROVE', '', 0.5


def heuristic_structure(lineup, ctx, sim_result):
    """
    Rule-based Structure: flag goalie opposing primary stack, low-total stacks.
    """
    name_col = 'name' if 'name' in lineup.columns else 'Player'
    team_col = 'team' if 'team' in lineup.columns else 'Team'
    pos_col = 'position' if 'position' in lineup.columns else 'Pos'

    # Find primary stack team
    team_counts = lineup[team_col].value_counts()
    primary_stack = team_counts.index[0] if len(team_counts) > 0 else None
    stack_size = team_counts.iloc[0] if len(team_counts) > 0 else 0

    # Find goalie's team
    goalie = lineup[lineup[pos_col] == 'G']
    if len(goalie) > 0:
        g_team = goalie.iloc[0][team_col]

        # CRITICAL: goalie opposing our primary stack = bad (r=-0.34)
        # Check if goalie's opponent IS our primary stack
        for g_info in ctx.confirmed_goalies:
            if g_info['name'] == goalie.iloc[0][name_col]:
                if g_info.get('opponent', '') == primary_stack:
                    return 'VETO', (f"Goalie {g_info['name']} ({g_team}) opposes primary "
                                    f"stack {primary_stack} — r=-0.34 correlation hurts"), 0.95

    # Check if primary stack is in a low-total game
    for vg in ctx.vegas_lines:
        if primary_stack in (vg['home'], vg['away']):
            if vg['total'] < 5.5 and stack_size >= 4:
                return 'FLAG', (f'{stack_size}-man {primary_stack} stack in low-total '
                                f'game ({vg["total"]})'), 0.7

    # Small stack = less correlation value
    if stack_size < 3:
        return 'FLAG', f'No real stack — largest team group is {stack_size}', 0.6

    return 'APPROVE', '', 0.5


def run_heuristic_agents(lineups, results, ctx, actual_scores):
    """
    Run all 3 heuristic agents on top lineups.
    Returns: list of (lineup, result, consensus_score, actual_fpts)
    """
    ADJUSTMENTS = {'APPROVE': 0, 'FLAG': -5, 'VETO': -15, 'BOOST': +3}
    n = len(lineups)

    scored = []
    for i in range(n):
        lu = lineups[i]
        r = results[i]

        # Base score from sim rank
        base = n - i

        # Run each agent
        agents_output = []
        for agent_fn, agent_name in [
            (heuristic_contrarian, 'Contrarian'),
            (heuristic_narrative, 'Narrative'),
            (heuristic_structure, 'Structure'),
        ]:
            verdict, reason, conf = agent_fn(lu, ctx, r)
            adj = ADJUSTMENTS.get(verdict, 0)
            base += adj
            agents_output.append((agent_name, verdict, reason, conf))

        # Get actual score
        name_col = 'name' if 'name' in lu.columns else 'Player'
        team_col = 'team' if 'team' in lu.columns else 'Team'
        act = sum(actual_scores.get(f"{row[name_col]}_{row[team_col]}", 0)
                  for _, row in lu.iterrows())

        scored.append((lu, r, base, act, agents_output))

    # Sort by consensus score
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════════════
#  API Agents (Claude-powered)
# ═══════════════════════════════════════════════════════════════════

def run_api_agents(lineups, results, ctx, actual_scores, api_key):
    """
    Run agent panel via Claude API.
    Falls back to heuristic if API fails.
    """
    panel = AgentPanel(ctx, api_key=api_key)

    try:
        ranked = panel.review(lineups, results, verbose=False)
    except Exception as e:
        print(f"    ⚠ API agent failed ({e}), falling back to heuristic")
        return run_heuristic_agents(lineups, results, ctx, actual_scores)

    # Attach actual scores
    name_col = 'name' if 'name' in lineups[0].columns else 'Player'
    team_col = 'team' if 'team' in lineups[0].columns else 'Team'

    scored = []
    for lu, r, consensus in ranked:
        act = sum(actual_scores.get(f"{row[name_col]}_{row[team_col]}", 0)
                  for _, row in lu.iterrows())
        scored.append((lu, r, consensus, act, []))

    return scored


# ═══════════════════════════════════════════════════════════════════
#  Candidate Generation (same as backtest_sim.py)
# ═══════════════════════════════════════════════════════════════════

def generate_candidates(pool, n_candidates=100):
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
        except Exception: pass
        if has_capped:
            try:
                batch = optimizer.optimize_lineup(capped, n_lineups=n_cap, randomness=rand)
                if batch: candidates.extend(batch)
            except Exception: pass
        try:
            batch = optimizer.optimize_lineup(pool, n_lineups=n_33, randomness=rand, max_from_team=3)
            if batch: candidates.extend(batch)
        except Exception: pass
    return candidates


def stack_desc(lu):
    col = 'team' if 'team' in lu.columns else 'Team'
    tc = lu[col].value_counts()
    return '-'.join(str(v) for v in sorted(tc.values, reverse=True) if v >= 2)


# ═══════════════════════════════════════════════════════════════════
#  Main Backtest
# ═══════════════════════════════════════════════════════════════════

def run_date(target_date, dk_date, full_df, n_candidates, n_sims, n_review, api_key):
    """Run full agent backtest for one date."""
    t0 = time.time()

    # Pool + history
    pool_raw = full_df[full_df['slate_date'] == dk_date].copy()
    pool = pool_raw[pool_raw['Score'].notna() & pool_raw['Salary'].notna()].copy()
    hist = full_df[full_df['Score'].notna() & (full_df['clean_date'] < target_date)].copy()
    hist_for_sim = hist[['Player', 'Team', 'Score', 'Pos', 'slate_date']].copy()

    if len(pool) < 30:
        print(f"  {target_date}: insufficient pool, skip")
        return None

    # Actual scores lookup
    score_map = dict(zip(pool['Player'] + '_' + pool['Team'], pool['Score']))

    # Build optimizer pool
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

    # Actual totals for all candidates
    actual_arr = np.array([
        sum(score_map.get(f"{r['name']}_{r['team']}", 0) for _, r in lu.iterrows())
        for lu in candidates
    ])

    # Sim engine
    engine = SimulationEngine(n_sims=n_sims)
    engine.fit_player_distributions(opt_pool, hist_for_sim, date_str=dk_date)

    sim_results = []
    for lu in candidates:
        sim = engine.simulate_lineup(lu, n_sims=n_sims)
        sim_results.append(sim)

    sim_means = np.array([s['mean'] for s in sim_results])
    sim_stds = np.array([s['std'] for s in sim_results])
    m3s = sim_means + 3.0 * sim_stds

    # Sim M+3σ pick (pure math, no agents)
    sim_pick_idx = int(np.argmax(m3s))
    sim_pick_fpts = actual_arr[sim_pick_idx]

    # Best possible
    best_idx = int(np.argmax(actual_arr))
    best_fpts = actual_arr[best_idx]
    avg_fpts = actual_arr.mean()

    # Sort candidates by M+3σ for agent review
    sorted_indices = np.argsort(-m3s)
    top_n = min(n_review, len(candidates))
    top_lineups = [candidates[i] for i in sorted_indices[:top_n]]
    top_results = [sim_results[i] for i in sorted_indices[:top_n]]
    # Add m3s to results
    for i, idx in enumerate(sorted_indices[:top_n]):
        top_results[i]['m3s'] = m3s[idx]

    # Build slate context
    ctx = build_slate_context(pool_raw, full_df, target_date, dk_date)

    # Run agents
    if api_key:
        agent_scored = run_api_agents(top_lineups, top_results, ctx, score_map, api_key)
    else:
        agent_scored = run_heuristic_agents(top_lineups, top_results, ctx, score_map)

    # Agent's top pick
    agent_pick_fpts = agent_scored[0][3]  # actual FPTS of consensus #1
    agent_delta = agent_pick_fpts - sim_pick_fpts

    # Agent's top pick sim rank (what was its original M+3σ rank?)
    agent_lu = agent_scored[0][0]
    try:
        agent_orig_idx = top_lineups.index(agent_lu)
        agent_sim_rank = agent_orig_idx + 1
    except ValueError:
        agent_sim_rank = -1

    elapsed = time.time() - t0

    # Percentiles
    sim_pctile = (actual_arr < sim_pick_fpts).sum() / len(actual_arr) * 100
    agent_pctile = (actual_arr < agent_pick_fpts).sum() / len(actual_arr) * 100

    # Print results
    print(f"\n  ── {target_date} ({len(candidates)} cands, {elapsed:.1f}s) ──")
    print(f"    {'Pick':<20} {'FPTS':>7} {'Pctile':>7} {'vs Avg':>7}")
    print(f"    {'-' * 45}")
    print(f"    {'Best Possible':<20} {best_fpts:>7.1f} {'99%':>7} {best_fpts - avg_fpts:>+6.1f}")
    print(f"    {'Sim M+3σ (#1)':<20} {sim_pick_fpts:>7.1f} {sim_pctile:>6.0f}% {sim_pick_fpts - avg_fpts:>+6.1f}")
    print(f"    {'Agent Consensus':<20} {agent_pick_fpts:>7.1f} {agent_pctile:>6.0f}% {agent_pick_fpts - avg_fpts:>+6.1f}")
    print(f"    {'Agent Delta':<20} {agent_delta:>+6.1f} FPTS  (agent was sim rank #{agent_sim_rank})")

    # Show agent actions on top 5
    print(f"\n    Agent Actions (top 5):")
    for rank, (lu, r, score, act, actions) in enumerate(agent_scored[:5], 1):
        m3 = r.get('m3s', r['mean'] + 3 * r['std'])
        action_str = ''
        for aname, verdict, reason, conf in actions:
            if verdict != 'APPROVE':
                action_str += f' [{aname[0]}:{verdict}]'
        if not action_str:
            action_str = ' [all APPROVE]'
        print(f"      #{rank} actual={act:>6.1f} M+3σ={m3:>5.0f} score={score:>3.0f}{action_str}")

    # Goalie comparison
    goalie_col = 'position' if 'position' in candidates[sim_pick_idx].columns else 'Pos'
    name_col = 'name' if 'name' in candidates[sim_pick_idx].columns else 'Player'
    team_col = 'team' if 'team' in candidates[sim_pick_idx].columns else 'Team'

    sim_lu = candidates[sim_pick_idx]
    sim_g = sim_lu[sim_lu[goalie_col] == 'G']
    agent_lu_final = agent_scored[0][0]
    agent_g = agent_lu_final[agent_lu_final[goalie_col] == 'G']

    if len(sim_g) > 0 and len(agent_g) > 0:
        sg = sim_g.iloc[0]
        ag = agent_g.iloc[0]
        sg_act = score_map.get(f"{sg[name_col]}_{sg[team_col]}", 0)
        ag_act = score_map.get(f"{ag[name_col]}_{ag[team_col]}", 0)
        if sg[name_col] != ag[name_col]:
            print(f"\n    Goalie change! Sim: {sg[name_col]} ({sg_act:.1f}) → Agent: {ag[name_col]} ({ag_act:.1f})")
        else:
            print(f"\n    Same goalie: {sg[name_col]} ({sg_act:.1f})")

    return {
        'date': target_date,
        'n_candidates': len(candidates),
        'best_fpts': best_fpts,
        'avg_fpts': avg_fpts,
        'sim_fpts': sim_pick_fpts,
        'sim_pctile': sim_pctile,
        'agent_fpts': agent_pick_fpts,
        'agent_pctile': agent_pctile,
        'agent_delta': agent_delta,
        'agent_sim_rank': agent_sim_rank,
        'elapsed': elapsed,
        'b2b_teams': list(ctx.back_to_back),
    }


def main():
    parser = argparse.ArgumentParser(description='Simons Agent Panel Backtest')
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--candidates', type=int, default=100)
    parser.add_argument('--sims', type=int, default=8000)
    parser.add_argument('--review', type=int, default=20,
                        help='Top N lineups for agents to review')
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    mode = 'API (Claude)' if api_key else 'Heuristic (rule-based)'

    print(f"{'=' * 80}")
    print(f"  SIMONS AGENT PANEL BACKTEST")
    print(f"  Mode: {mode}")
    print(f"  Candidates: {args.candidates} | Sims: {args.sims:,} | Review top: {args.review}")
    print(f"{'=' * 80}")

    full = load_all_dk_history()

    dates = {args.date: BACKTEST_DATES[args.date]} if args.date else BACKTEST_DATES
    all_results = []

    for target_date, dk_date in dates.items():
        result = run_date(target_date, dk_date, full,
                          args.candidates, args.sims, args.review, api_key)
        if result:
            all_results.append(result)

    if len(all_results) < 2:
        print("\nNot enough slates for summary.")
        return

    rdf = pd.DataFrame(all_results)

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 80}")
    print(f"  AGENT PANEL RESULTS — {len(rdf)} slates ({mode})")
    print(f"{'=' * 80}")

    print(f"\n  {'Date':<12} {'Best':>6} {'Sim M3σ':>8} {'Agent':>7} {'Delta':>7} {'AgentRk':>8}")
    print(f"  {'-' * 52}")
    for _, r in rdf.iterrows():
        marker = ' ✓' if r['agent_delta'] > 0 else ' ✗' if r['agent_delta'] < 0 else ' ='
        print(f"  {r['date']:<12} {r['best_fpts']:>6.0f} {r['sim_fpts']:>8.1f} "
              f"{r['agent_fpts']:>7.1f} {r['agent_delta']:>+6.1f}{marker} "
              f"#{r['agent_sim_rank']:>2}")

    print(f"  {'-' * 52}")

    # Key metrics
    avg_delta = rdf['agent_delta'].mean()
    win_rate = (rdf['agent_delta'] > 0).sum() / len(rdf)
    sim_avg = rdf['sim_fpts'].mean()
    agent_avg = rdf['agent_fpts'].mean()
    sim_avg_pctile = rdf['sim_pctile'].mean()
    agent_avg_pctile = rdf['agent_pctile'].mean()

    print(f"\n  SCORECARD:")
    print(f"    Sim M+3σ avg:       {sim_avg:>7.1f} FPTS  ({sim_avg_pctile:.0f}th pctile)")
    print(f"    Agent consensus avg: {agent_avg:>7.1f} FPTS  ({agent_avg_pctile:.0f}th pctile)")
    print(f"    Agent delta:         {avg_delta:>+6.1f} FPTS/slate")
    print(f"    Agent win rate:      {win_rate:.0%} ({(rdf['agent_delta'] > 0).sum()}/{len(rdf)})")
    print(f"    Total edge:          {rdf['agent_delta'].sum():>+6.0f} FPTS across {len(rdf)} slates")

    if avg_delta > 2.0:
        print(f"\n  VERDICT: ✅ AGENTS ADD VALUE ({avg_delta:+.1f}/slate)")
    elif avg_delta < 0:
        print(f"\n  VERDICT: ❌ AGENTS HURT ({avg_delta:+.1f}/slate) — consider removing")
    else:
        print(f"\n  VERDICT: ⚠ MARGINAL ({avg_delta:+.1f}/slate) — need more data")

    remaining = max(0, 20 - len(rdf))
    if remaining > 0:
        print(f"    ({remaining} more slates needed for full evaluation)")

    # B2B impact
    b2b_slates = rdf[rdf['b2b_teams'].apply(len) > 0]
    if len(b2b_slates) > 0:
        print(f"\n  B2B IMPACT: {len(b2b_slates)}/{len(rdf)} slates had B2B teams")
        b2b_delta = b2b_slates['agent_delta'].mean()
        non_b2b_delta = rdf[rdf['b2b_teams'].apply(len) == 0]['agent_delta'].mean()
        print(f"    Agent delta on B2B slates:     {b2b_delta:+.1f}")
        if len(rdf[rdf['b2b_teams'].apply(len) == 0]) > 0:
            print(f"    Agent delta on non-B2B slates: {non_b2b_delta:+.1f}")

    total_time = rdf['elapsed'].sum()
    print(f"\n  Runtime: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
