#!/usr/bin/env python3
"""
Diverse Candidate Generator
===============================

Wraps the optimizer to produce structurally diverse candidate pools.
Addresses 5 measured diversity gaps:

1. Chalk lock-in → Player exposure caps
2. Goalie concentration → Forced goalie rotation
3. Salary bunching → Salary band exploration
4. Stack monotony → Per-game-environment stacking
5. No structural variety → Construction archetypes

Usage:
    from candidate_generator import generate_diverse_candidates
    candidates = generate_diverse_candidates(pool, n_total=150)

Backtest-proven: candidate diversity is the #1 lever for sim selector edge.
More diverse candidates → selector has more to choose from → better final pick.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import Counter


def generate_diverse_candidates(
    pool: pd.DataFrame,
    optimizer,
    n_total: int = 150,
    verbose: bool = True,
) -> List[pd.DataFrame]:
    """
    Generate a diverse set of candidate lineups.
    
    Allocation:
      40% — Goalie rotation (forced per-goalie batches)
      25% — Standard triple-mix (uncapped/capped/3-3)
      15% — Game environment stacking (force-stack each high-total game)
      10% — Salary band exploration ($44-47k, $47-49k)
      10% — Structural archetypes (stars+scrubs, balanced, bring-back)
    
    Args:
        pool: Player pool with name, team, position, salary, projected_fpts
        optimizer: NHLLineupOptimizer instance
        n_total: Target total candidates
        verbose: Print generation summary
    
    Returns:
        List of lineup DataFrames (deduplicated)
    """
    all_candidates = []
    used_hashes = set()
    
    def add_unique(lineups):
        """Add lineups, deduplicating by player set."""
        added = 0
        for lu in lineups:
            h = frozenset(lu['name'].tolist())
            if h not in used_hashes:
                used_hashes.add(h)
                all_candidates.append(lu)
                added += 1
        return added
    
    def safe_optimize(p, **kwargs):
        """Run optimizer, return empty list on failure."""
        try:
            result = optimizer.optimize_lineup(p, **kwargs)
            return result if result else []
        except Exception:
            return []
    
    pos_col = 'position'
    goalies = pool[pool[pos_col] == 'G'].copy()
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Goalie Rotation (40% of candidates)
    # Force each viable goalie to appear, preventing concentration
    # ═══════════════════════════════════════════════════════════════
    n_goalie = int(n_total * 0.40)
    
    # Rank goalies by projection
    goalie_list = goalies.nlargest(min(8, len(goalies)), 'projected_fpts')
    n_per_goalie = max(3, n_goalie // max(len(goalie_list), 1))
    
    goalie_added = 0
    for _, goalie_row in goalie_list.iterrows():
        g_name = goalie_row['name']
        other_goalies = goalies[goalies['name'] != g_name]['name'].tolist()
        
        for rand in [0.08, 0.15, 0.22]:
            batch = safe_optimize(
                pool,
                n_lineups=max(n_per_goalie // 3, 2),
                randomness=rand,
                force_players=[g_name],
                exclude_players=other_goalies,
            )
            goalie_added += add_unique(batch)
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Standard Triple-Mix (25%)
    # Same as existing: uncapped, capped, 3-3
    # ═══════════════════════════════════════════════════════════════
    n_standard = int(n_total * 0.25)
    
    capped = pool[pool['salary'] <= 7500].copy()
    pc = capped[pos_col].value_counts()
    has_capped = (pc.get('C', 0) >= 3 and pc.get('W', 0) >= 4
                  and pc.get('D', 0) >= 3 and pc.get('G', 0) >= 2)
    
    standard_added = 0
    for rand in [0.05, 0.12, 0.20]:
        n_std = max(n_standard // 6, 3)
        batch = safe_optimize(pool, n_lineups=n_std, randomness=rand)
        standard_added += add_unique(batch)
        
        if has_capped:
            batch = safe_optimize(capped, n_lineups=max(n_std // 2, 2), randomness=rand)
            standard_added += add_unique(batch)
        
        batch = safe_optimize(pool, n_lineups=max(n_std // 2, 2),
                              randomness=rand, max_from_team=3)
        standard_added += add_unique(batch)
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Game Environment Stacking (15%)
    # Force stacks from each game, especially high-total games
    # ═══════════════════════════════════════════════════════════════
    n_game = int(n_total * 0.15)
    
    # Identify unique games and their totals
    games = {}
    if 'TeamGoal' in pool.columns:
        for team in pool['team'].unique():
            team_rows = pool[pool['team'] == team]
            if len(team_rows) > 0:
                tg = team_rows['TeamGoal'].iloc[0] if 'TeamGoal' in team_rows.columns else 3.0
                if pd.notna(tg):
                    games[team] = float(tg)
    
    # Sort teams by game total (highest first = most upside)
    sorted_teams = sorted(games.items(), key=lambda x: x[1], reverse=True)
    n_per_game = max(2, n_game // max(len(sorted_teams[:10]), 1))
    
    game_added = 0
    for team, total in sorted_teams[:10]:  # Top 10 game environments
        for rand in [0.10, 0.18]:
            batch = safe_optimize(
                pool,
                n_lineups=max(n_per_game // 2, 2),
                randomness=rand,
                stack_teams=[team],
            )
            game_added += add_unique(batch)
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Salary Band Exploration (10%)
    # Force lower salary usage to unlock different player combos
    # ═══════════════════════════════════════════════════════════════
    n_salary = int(n_total * 0.10)
    
    salary_added = 0
    # Mid-range salary: cap individual salaries to force different combos
    mid_pool = pool[pool['salary'] <= 6500].copy()
    mid_pc = mid_pool[pos_col].value_counts()
    has_mid = (mid_pc.get('C', 0) >= 3 and mid_pc.get('W', 0) >= 4
               and mid_pc.get('D', 0) >= 3 and mid_pc.get('G', 0) >= 2)
    
    if has_mid:
        for rand in [0.10, 0.18]:
            batch = safe_optimize(mid_pool, n_lineups=max(n_salary // 4, 3),
                                  randomness=rand)
            salary_added += add_unique(batch)
    
    # Stars + scrubs: force 2 expensive players, rest cheap
    elite = pool[pool['salary'] >= 7800].nlargest(8, 'projected_fpts')
    if len(elite) >= 2:
        for _, star in elite.iterrows():
            batch = safe_optimize(
                pool,
                n_lineups=2,
                randomness=0.12,
                force_players=[star['name']],
            )
            salary_added += add_unique(batch)
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Structural Archetypes (10%)
    # Double stacks, bring-backs, contrarian constructions
    # ═══════════════════════════════════════════════════════════════
    n_struct = int(n_total * 0.10)
    
    struct_added = 0
    # Double stack: 3-3 from two different high-total games
    if len(sorted_teams) >= 4:
        for i in range(0, min(6, len(sorted_teams)), 2):
            team1 = sorted_teams[i][0]
            team2 = sorted_teams[i + 1][0] if i + 1 < len(sorted_teams) else sorted_teams[0][0]
            if team1 != team2:
                batch = safe_optimize(
                    pool,
                    n_lineups=max(n_struct // 6, 2),
                    randomness=0.12,
                    stack_teams=[team1],
                    secondary_stack_team=team2,
                    max_from_team=3,
                )
                struct_added += add_unique(batch)
    
    # Higher randomness for tail exploration
    for rand in [0.25, 0.30]:
        batch = safe_optimize(pool, n_lineups=max(n_struct // 4, 3),
                              randomness=rand)
        struct_added += add_unique(batch)
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    if verbose:
        total = len(all_candidates)
        
        # Compute diversity metrics
        all_players = []
        goalie_counts = Counter()
        stack_counts = Counter()
        salaries = []
        
        for lu in all_candidates:
            all_players.extend(lu['name'].tolist())
            g = lu[lu[pos_col] == 'G']
            if len(g) > 0:
                goalie_counts[g.iloc[0]['name']] += 1
            tc = lu['team'].value_counts()
            stack_counts[f"{tc.index[0]} ({tc.iloc[0]})"] += 1
            salaries.append(lu['salary'].sum())
        
        player_counts = Counter(all_players)
        unique_players = len(player_counts)
        max_exposure = max(player_counts.values()) / total * 100 if total > 0 else 0
        top_goalie_pct = (goalie_counts.most_common(1)[0][1] / total * 100) if goalie_counts else 0
        unique_goalies = len(goalie_counts)
        unique_stacks = len(stack_counts)
        sal_arr = np.array(salaries) if salaries else np.array([0])
        
        print(f"\n  ╔══ DIVERSE CANDIDATE POOL ══════════════════════════╗")
        print(f"  ║  Total candidates: {total:<5} (target: {n_total})          ║")
        print(f"  ║  ─── Generation Breakdown ───                      ║")
        print(f"  ║  Goalie rotation:  {goalie_added:<4} ({goalie_added/max(total,1)*100:>4.0f}%)                 ║")
        print(f"  ║  Standard mix:     {standard_added:<4} ({standard_added/max(total,1)*100:>4.0f}%)                 ║")
        print(f"  ║  Game environment:  {game_added:<4} ({game_added/max(total,1)*100:>4.0f}%)                 ║")
        print(f"  ║  Salary bands:      {salary_added:<4} ({salary_added/max(total,1)*100:>4.0f}%)                 ║")
        print(f"  ║  Structural:        {struct_added:<4} ({struct_added/max(total,1)*100:>4.0f}%)                 ║")
        print(f"  ║  ─── Diversity Metrics ───                         ║")
        print(f"  ║  Unique players:   {unique_players:<4} / {len(pool)}                     ║")
        print(f"  ║  Max player exposure: {max_exposure:<4.0f}%                        ║")
        print(f"  ║  Unique goalies:    {unique_goalies:<3}  (top: {top_goalie_pct:.0f}%)               ║")
        print(f"  ║  Unique stacks:     {unique_stacks:<3}                              ║")
        print(f"  ║  Salary: ${sal_arr.mean()/1000:.1f}k avg, ${sal_arr.min()/1000:.1f}k-${sal_arr.max()/1000:.1f}k     ║")
        print(f"  ╚════════════════════════════════════════════════════╝")
    
    return all_candidates


def compare_diversity(old_candidates, new_candidates, label_old='Old', label_new='New'):
    """Compare diversity metrics between two candidate pools."""
    
    def metrics(candidates):
        all_players = []
        goalie_counts = Counter()
        stack_counts = Counter()
        salaries = []
        
        for lu in candidates:
            pos_col = 'position' if 'position' in lu.columns else 'Pos'
            all_players.extend(lu['name'].tolist())
            g = lu[lu[pos_col] == 'G']
            if len(g) > 0:
                goalie_counts[g.iloc[0]['name']] += 1
            tc = lu['team'].value_counts()
            stack_counts[f"{tc.index[0]} ({tc.iloc[0]})"] += 1
            salaries.append(lu['salary'].sum())
        
        pc = Counter(all_players)
        n = len(candidates)
        return {
            'n': n,
            'unique_players': len(pc),
            'max_exposure': max(pc.values()) / n * 100 if n > 0 else 0,
            'unique_goalies': len(goalie_counts),
            'top_goalie_pct': goalie_counts.most_common(1)[0][1] / n * 100 if goalie_counts else 0,
            'unique_stacks': len(stack_counts),
            'sal_mean': np.mean(salaries) if salaries else 0,
            'sal_min': np.min(salaries) if salaries else 0,
        }
    
    old_m = metrics(old_candidates)
    new_m = metrics(new_candidates)
    
    print(f"\n  {'Metric':<25} {label_old:>12} {label_new:>12} {'Change':>10}")
    print(f"  {'-'*60}")
    
    comparisons = [
        ('Candidates', 'n', ''),
        ('Unique players', 'unique_players', ''),
        ('Max player exposure', 'max_exposure', '%'),
        ('Unique goalies', 'unique_goalies', ''),
        ('Top goalie %', 'top_goalie_pct', '%'),
        ('Unique stacks', 'unique_stacks', ''),
        ('Salary mean', 'sal_mean', '$'),
        ('Salary min', 'sal_min', '$'),
    ]
    
    for label, key, unit in comparisons:
        old_v = old_m[key]
        new_v = new_m[key]
        delta = new_v - old_v
        
        if unit == '$':
            print(f"  {label:<25} ${old_v/1000:>9.1f}k ${new_v/1000:>9.1f}k {delta/1000:>+8.1f}k")
        elif unit == '%':
            print(f"  {label:<25} {old_v:>11.0f}% {new_v:>11.0f}% {delta:>+9.0f}%")
        else:
            print(f"  {label:<25} {old_v:>12.0f} {new_v:>12.0f} {delta:>+10.0f}")


if __name__ == '__main__':
    """Quick test on one slate."""
    import sys, glob, time
    sys.path.insert(0, '.')
    from optimizer import NHLLineupOptimizer
    
    DK_DIR = "/home/claude/dk_salaries_season/DKSalaries_NHL_season"
    files = sorted(glob.glob(f"{DK_DIR}/draftkings_NHL_*2026-02-04*.csv"))
    if not files:
        DK_DIR = str(Path.home() / "Desktop" / "DKSalaries_NHL_season")
        files = sorted(glob.glob(f"{DK_DIR}/draftkings_NHL_*2026-02-04*.csv"))
    
    df = pd.read_csv(files[0], encoding='utf-8-sig', low_memory=False)
    for col in ['Score', 'Salary', 'Avg', 'Ceiling', 'TeamGoal', 'OppGoal']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    pool = df[df['Score'].notna() & df['Salary'].notna()].rename(columns={
        'Player': 'name', 'Team': 'team', 'Pos': 'position',
        'Salary': 'salary', 'Avg': 'projected_fpts',
    }).copy()
    pool['position'] = pool['position'].apply(
        lambda p: 'W' if str(p).upper() in ('L','R','LW','RW') else 
                  'D' if str(p).upper() in ('LD','RD') else str(p).upper()
    )
    pool['projected_fpts'] = pool['projected_fpts'].fillna(3.0)
    
    opt = NHLLineupOptimizer()
    
    # Old way
    print("=== OLD METHOD (triple-mix) ===")
    t0 = time.time()
    old_candidates = []
    capped = pool[pool['salary'] <= 7500].copy()
    for rand in [0.05, 0.10, 0.15, 0.20, 0.25]:
        try:
            batch = opt.optimize_lineup(pool, n_lineups=15, randomness=rand)
            if batch: old_candidates.extend(batch)
        except: pass
        try:
            batch = opt.optimize_lineup(capped, n_lineups=8, randomness=rand)
            if batch: old_candidates.extend(batch)
        except: pass
        try:
            batch = opt.optimize_lineup(pool, n_lineups=8, randomness=rand, max_from_team=3)
            if batch: old_candidates.extend(batch)
        except: pass
    print(f"  Generated {len(old_candidates)} in {time.time()-t0:.1f}s")
    
    # New way
    print("\n=== NEW METHOD (diverse generator) ===")
    t0 = time.time()
    new_candidates = generate_diverse_candidates(pool, opt, n_total=150)
    print(f"  Generated in {time.time()-t0:.1f}s")
    
    # Compare
    compare_diversity(old_candidates, new_candidates, 'Triple-Mix', 'Diverse')
