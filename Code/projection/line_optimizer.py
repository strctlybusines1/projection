#!/usr/bin/env python3
"""
Line-Based NHL DFS Optimizer
==============================

Jim Simons approach: Don't project 250 players. Project LINES.
Pick the right L1 + L2 combination, fill 2D + 1G around them.

Contest: ~87 entries, 24% cash (top 17), 1st place avg 154 FPTS
Strategy: Same-team L1+L2 stack = correlated upside

Key insight: Implied total only picks the best L1 21% of the time.
The field all stacks top-implied. Edge comes from better line selection.

DK Roster: 2C + 3W + 2D + 1G + 1UTIL (C/W only)
A standard L1+L2 same-team stack fills: 2C + 3-4W = 5-6 of 9 spots.
Remaining: 2D + 1G + 0-1 UTIL flex.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "data" / "nhl_dfs_history.db"
BACKTESTS_DIR = PROJECT_DIR / "backtests"
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

SALARY_CAP = 50000
TEAM_NORM = {'nj': 'njd', 'la': 'lak', 'sj': 'sjs', 'tb': 'tbl'}

def norm(t):
    return TEAM_NORM.get(str(t).lower().strip(), str(t).lower().strip())


def load_actuals(date_str, conn):
    sk = pd.read_sql("SELECT player_name as name, team, dk_fpts FROM game_logs_skaters WHERE game_date = ?",
                      conn, params=(date_str,))
    gl = pd.read_sql("SELECT player_name as name, team, dk_fpts FROM game_logs_goalies WHERE game_date = ?",
                      conn, params=(date_str,))
    actuals = pd.concat([sk, gl], ignore_index=True)
    actuals['_key'] = actuals['name'].str.lower().str.strip() + '_' + actuals['team'].apply(norm)
    return actuals


def score_players(players, actuals):
    """Score a list of player dicts against actuals. Returns (total, matched, scratched)."""
    total = 0.0
    matched = 0
    scratched = 0
    for p in players:
        name = str(p.get('name', '')).lower().strip()
        team = norm(str(p.get('team', '')))
        key = f"{name}_{team}"
        exact = actuals[actuals['_key'] == key]
        if not exact.empty:
            total += exact.iloc[0]['dk_fpts']
            matched += 1
            continue
        team_players = actuals[actuals['_key'].str.endswith(f"_{team}")]
        best_score = 0
        best_fpts = 0
        for _, row in team_players.iterrows():
            score = SequenceMatcher(None, name, row['name'].lower().strip()).ratio()
            if score > best_score:
                best_score = score
                best_fpts = row['dk_fpts']
        if best_score >= 0.70:
            total += best_fpts
            matched += 1
        else:
            scratched += 1
    return total, matched, scratched


# ==============================================================================
# LINE BUILDER: Extract lines from DK salary data
# ==============================================================================

def build_team_lines(dk_pool: pd.DataFrame) -> Dict:
    """
    Build line groupings from DK pool with start_line data.
    Returns dict: team -> {1: [players], 2: [players], ...}
    """
    teams = {}
    has_lines = dk_pool[dk_pool['start_line'].isin(['1', '2', '3', '4'])].copy()

    for team, team_df in has_lines.groupby('team'):
        teams[team] = {}
        for line_num in ['1', '2', '3', '4']:
            line_players = team_df[team_df['start_line'] == line_num]
            forwards = line_players[line_players['position'].isin(['C', 'W', 'LW', 'RW', 'L', 'R'])]
            defensemen = line_players[line_players['position'] == 'D']
            teams[team][int(line_num)] = {
                'forwards': forwards.to_dict('records'),
                'defensemen': defensemen.to_dict('records'),
                'all': line_players.to_dict('records'),
                'fwd_salary': forwards['salary'].sum(),
                'fwd_names': list(forwards['player_name'].values),
                'n_fwd': len(forwards),
                'team_impl': team_df['team_implied_total'].iloc[0] if pd.notna(team_df['team_implied_total'].iloc[0]) else 0,
                'game_total': team_df['game_total'].iloc[0] if pd.notna(team_df['game_total'].iloc[0]) else 0,
            }
    return teams


# ==============================================================================
# LINE SCORING MODEL
# ==============================================================================

def score_line(line_data: Dict, line_num: int, all_game_logs: pd.DataFrame,
               date_str: str) -> float:
    """
    Score a line based on available features.
    Returns projected total FPTS for the line's forwards.

    Features used:
      - Team implied total (weak but real: rho=0.12)
      - Line salary (stronger: rho=0.22-0.26)
      - Recent form of individual players (L5 avg FPTS)
      - PP unit presence
      - Game total (pace proxy)
    """
    if line_data['n_fwd'] < 2:
        return 0.0

    # Base: historical average by line number
    LINE_BASELINES = {1: 25.8, 2: 20.5, 3: 13.6, 4: 7.2}
    base = LINE_BASELINES.get(line_num, 10.0)

    # --- Implied total adjustment ---
    impl = line_data.get('team_impl', 0)
    if impl > 0:
        # Average implied total is ~3.0; scale around it
        impl_factor = impl / 3.0
        impl_factor = np.clip(impl_factor, 0.70, 1.40)
    else:
        impl_factor = 1.0

    # --- Salary adjustment ---
    # Higher salary lines = better players = more production
    fwd_sal = line_data.get('fwd_salary', 0)
    if line_num == 1:
        avg_l1_sal = 16764  # from our analysis
        sal_factor = fwd_sal / avg_l1_sal if avg_l1_sal > 0 else 1.0
    elif line_num == 2:
        avg_l2_sal = 13172
        sal_factor = fwd_sal / avg_l2_sal if avg_l2_sal > 0 else 1.0
    else:
        sal_factor = 1.0
    sal_factor = np.clip(sal_factor, 0.70, 1.40)

    # --- Recent form of line players ---
    form_factor = 1.0
    if all_game_logs is not None and not all_game_logs.empty:
        target_date = pd.Timestamp(date_str)
        prior = all_game_logs[all_game_logs['game_date'] < target_date]

        player_forms = []
        for fwd in line_data['forwards']:
            pname = str(fwd.get('player_name', '')).lower().strip()
            pteam = norm(str(fwd.get('team', '')))
            pkey = f"{pname}_{pteam}"

            player_logs = prior[prior['_key'] == pkey].sort_values('game_date', ascending=False).head(5)
            if len(player_logs) >= 3:
                player_forms.append(player_logs['dk_fpts'].mean())

        if player_forms:
            avg_form = np.mean(player_forms)
            # Compare to expected per-player average for this line
            expected_per_player = base / max(line_data['n_fwd'], 1)
            if expected_per_player > 0:
                form_factor = avg_form / expected_per_player
                form_factor = np.clip(form_factor, 0.60, 1.60)

    # --- PP boost ---
    pp_boost = 1.0
    for fwd in line_data['forwards']:
        pp = fwd.get('pp_unit')
        if pd.notna(pp) and str(pp) in ['1', '1.0']:
            pp_boost = 1.08  # PP1 players produce ~10% more
            break
        elif pd.notna(pp) and str(pp) in ['2', '2.0']:
            pp_boost = max(pp_boost, 1.03)

    # --- Game total (pace) ---
    game_total = line_data.get('game_total', 0)
    if game_total > 0:
        pace_factor = game_total / 6.0  # avg game total ~6.0
        pace_factor = np.clip(pace_factor, 0.85, 1.20)
    else:
        pace_factor = 1.0

    # Composite: weight the factors
    # Salary is strongest signal (rho=0.25), form next, impl/pace weaker
    projected = base * (
        0.35 * sal_factor +
        0.30 * form_factor +
        0.15 * impl_factor +
        0.10 * pace_factor +
        0.10 * 1.0  # constant baseline
    ) * pp_boost

    return projected


# ==============================================================================
# COMBINATORIC LINE-PAIR OPTIMIZER
# ==============================================================================

def find_best_line_combos(team_lines: Dict, dk_pool: pd.DataFrame,
                          all_game_logs: pd.DataFrame, date_str: str,
                          n_combos: int = 10) -> List[Dict]:
    """
    Find the best L1+L2 combinations (same-team stacks) and build full lineups.

    For each team with L1+L2 data:
      1. Score L1 and L2
      2. Check if L1+L2 forwards fit DK roster (2C+3W+UTIL)
      3. Calculate remaining salary budget for 2D + 1G
      4. Fill with best available D and G under budget
      5. Return ranked lineup candidates

    Returns list of lineup dicts sorted by projected total.
    """
    candidates = []

    # Score all lines
    for team, lines in team_lines.items():
        if 1 not in lines or 2 not in lines:
            continue
        l1 = lines[1]
        l2 = lines[2]

        if l1['n_fwd'] < 2 or l2['n_fwd'] < 2:
            continue

        l1_proj = score_line(l1, 1, all_game_logs, date_str)
        l2_proj = score_line(l2, 2, all_game_logs, date_str)

        # Combine forwards from L1 + L2
        all_fwds = l1['forwards'] + l2['forwards']
        fwd_salary = sum(f['salary'] for f in all_fwds)
        n_fwd = len(all_fwds)

        # Check position feasibility: need 2C + 3W + 1UTIL(C/W)
        n_c = sum(1 for f in all_fwds if f['position'] == 'C')
        n_w = sum(1 for f in all_fwds if f['position'] in ['W', 'LW', 'RW', 'L', 'R'])

        # We need exactly 2C and 3W for main slots, UTIL can be C or W
        # Total forwards should ideally be 5-6 (2C+3W or 2C+3W+1UTIL)
        if n_c < 1 or n_w < 2:
            continue  # Can't fill minimum positions

        remaining_budget = SALARY_CAP - fwd_salary
        remaining_slots = 9 - n_fwd  # Need to fill with D, G, and maybe flex

        # We need 2D + 1G minimum
        n_d_needed = 2
        n_g_needed = 1
        n_flex_needed = remaining_slots - n_d_needed - n_g_needed

        candidates.append({
            'team': team,
            'l1_proj': l1_proj,
            'l2_proj': l2_proj,
            'combo_proj': l1_proj + l2_proj,
            'fwd_salary': fwd_salary,
            'remaining_budget': remaining_budget,
            'n_fwd': n_fwd,
            'n_c': n_c,
            'n_w': n_w,
            'n_d_needed': n_d_needed,
            'n_g_needed': n_g_needed,
            'n_flex_needed': max(0, n_flex_needed),
            'forwards': all_fwds,
            'team_impl': l1.get('team_impl', 0),
            'l1_names': l1['fwd_names'],
            'l2_names': l2['fwd_names'],
        })

    # Sort by projected combo FPTS
    candidates.sort(key=lambda x: x['combo_proj'], reverse=True)

    return candidates[:n_combos]


def fill_lineup(candidate: Dict, dk_pool: pd.DataFrame, actuals: pd.DataFrame = None) -> Optional[List[Dict]]:
    """
    Fill a lineup candidate with 2D + 1G + flex from the available pool.

    If actuals provided (backtest mode), filter to confirmed players.
    """
    team = candidate['team']
    forwards = candidate['forwards']
    remaining = candidate['remaining_budget']

    # Get available D (not on same team for diversity, or same team for correlation)
    used_names = set(f['player_name'] for f in forwards)

    pool_d = dk_pool[
        (dk_pool['position'] == 'D') &
        (~dk_pool['player_name'].isin(used_names))
    ].copy()

    pool_g = dk_pool[dk_pool['position'] == 'G'].copy()

    # If backtest mode, filter to confirmed
    if actuals is not None:
        actual_keys = set(actuals['_key'].tolist())
        pool_d['_key2'] = pool_d['player_name'].str.lower().str.strip() + '_' + pool_d['team'].apply(norm)
        pool_g['_key2'] = pool_g['player_name'].str.lower().str.strip() + '_' + pool_g['team'].apply(norm)
        pool_d = pool_d[pool_d['_key2'].isin(actual_keys)]
        pool_g = pool_g[pool_g['_key2'].isin(actual_keys)]

    if len(pool_d) < 2 or len(pool_g) < 1:
        return None

    # Score D candidates: use salary as proxy (rho=0.37 for D)
    pool_d = pool_d.sort_values('salary', ascending=False)

    # Score G candidates: use salary as proxy (best we have)
    pool_g = pool_g.sort_values('salary', ascending=False)

    # Try to fit 2D + 1G + any flex under remaining budget
    best_lineup = None
    best_d_score = 0

    # Try top goalies
    for g_idx, g_row in pool_g.head(5).iterrows():
        g_sal = g_row['salary']
        budget_after_g = remaining - g_sal

        if budget_after_g < 5000:  # Need at least $2.5k * 2 for D
            continue

        # Pick best 2 D under budget
        affordable_d = pool_d[pool_d['salary'] <= budget_after_g]
        if len(affordable_d) < 2:
            continue

        # Greedy: pick highest salary D, then next under remaining
        d1 = affordable_d.iloc[0]
        d1_sal = d1['salary']
        d2_pool = affordable_d[(affordable_d.index != d1.name) &
                                (affordable_d['salary'] <= budget_after_g - d1_sal)]
        if d2_pool.empty:
            continue
        d2 = d2_pool.iloc[0]

        total_fill_sal = g_sal + d1_sal + d2['salary']
        if total_fill_sal > remaining:
            continue

        d_score = d1_sal + d2['salary']  # proxy for quality
        if d_score > best_d_score:
            best_d_score = d_score
            best_lineup = {
                'goalie': g_row.to_dict(),
                'd1': d1.to_dict(),
                'd2': d2.to_dict(),
                'total_salary': sum(f['salary'] for f in forwards) + total_fill_sal,
            }

    if best_lineup is None:
        return None

    # Build final player list
    players = []
    for f in forwards:
        players.append({
            'name': f['player_name'],
            'team': f['team'],
            'salary': f['salary'],
            'position': f['position'],
            'role': 'stack',
        })
    for d_key in ['d1', 'd2']:
        d = best_lineup[d_key]
        players.append({
            'name': d['player_name'],
            'team': d['team'],
            'salary': d['salary'],
            'position': 'D',
            'role': 'fill',
        })
    g = best_lineup['goalie']
    players.append({
        'name': g['player_name'],
        'team': g['team'],
        'salary': g['salary'],
        'position': 'G',
        'role': 'fill',
    })

    return players


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================

def run_line_backtest(start_date='2025-11-07', n_top_stacks=3, confirmed_only=True):
    """
    Backtest line-based strategy.

    For each date:
      1. Build team lines from DK data
      2. Score and rank L1+L2 combos
      3. Fill with 2D+1G
      4. Score against actuals
      5. Compare to contest results
    """
    conn = sqlite3.connect(str(DB_PATH))

    # Pre-load game logs for rolling features
    print("Loading game logs...")
    all_logs = pd.read_sql("""
        SELECT player_name, team, game_date, dk_fpts, position
        FROM game_logs_skaters ORDER BY game_date
    """, conn)
    all_logs['game_date'] = pd.to_datetime(all_logs['game_date'])
    all_logs['_key'] = all_logs['player_name'].str.lower().str.strip() + '_' + all_logs['team'].apply(norm)
    print(f"  {len(all_logs)} game log records")

    dates = pd.read_sql(f"""
        SELECT DISTINCT d.slate_date FROM dk_salaries d
        WHERE d.slate_date >= '{start_date}'
        AND d.start_line IS NOT NULL AND d.start_line != ''
        AND EXISTS (SELECT 1 FROM game_logs_skaters g WHERE g.game_date = d.slate_date)
        ORDER BY d.slate_date
    """, conn)['slate_date'].tolist()

    print(f"\n{'='*70}")
    print(f"  LINE-BASED BACKTEST | {len(dates)} dates | Top {n_top_stacks} stacks per date")
    print(f"  Confirmed only: {confirmed_only}")
    print(f"{'='*70}")

    results = []

    for di, date_str in enumerate(dates, 1):
        dk_pool = pd.read_sql("SELECT * FROM dk_salaries WHERE slate_date = ?",
                               conn, params=(date_str,))
        dk_pool = dk_pool.rename(columns={'player_name': 'player_name'})  # ensure consistency

        actuals = load_actuals(date_str, conn)

        if dk_pool.empty or actuals.empty:
            continue

        # Get contest info
        contest = conn.execute("""
            SELECT MIN(CASE WHEN n_cashed > 0 THEN score END),
                   MAX(CASE WHEN place = 1 THEN score END)
            FROM contest_results WHERE slate_date = ?
        """, (date_str,)).fetchone()
        cash_line = contest[0] if contest and contest[0] else 0
        first_place = contest[1] if contest and contest[1] else 0

        # Build lines
        team_lines = build_team_lines(dk_pool)

        # Find best combos
        combos = find_best_line_combos(team_lines, dk_pool, all_logs, date_str, n_combos=10)

        if not combos:
            print(f"  [{di:3d}] {date_str}: No valid line combos")
            continue

        # Fill and score top N stacks
        best_actual = 0
        best_lineup_info = None

        for ci, combo in enumerate(combos[:n_top_stacks]):
            lineup = fill_lineup(
                combo, dk_pool,
                actuals if confirmed_only else None
            )
            if lineup is None:
                continue

            actual_total, n_matched, n_scratched = score_players(lineup, actuals)

            if actual_total > best_actual:
                best_actual = actual_total
                best_lineup_info = {
                    'stack_team': combo['team'],
                    'combo_proj': combo['combo_proj'],
                    'actual': actual_total,
                    'matched': n_matched,
                    'scratched': n_scratched,
                    'salary': sum(p['salary'] for p in lineup),
                    'n_fwd': combo['n_fwd'],
                    'lineup': lineup,
                }

        if best_lineup_info is None:
            print(f"  [{di:3d}] {date_str}: Could not fill any lineup")
            continue

        is_cash = best_actual >= cash_line if cash_line > 0 else None
        is_first = best_actual >= first_place if first_place > 0 else None
        status = 'CASH' if is_cash else ('miss' if is_cash is not None else '  - ')

        n_teams = dk_pool['team'].nunique()
        print(f"  [{di:3d}] {date_str} ({n_teams:2d}t) | "
              f"Stack: {best_lineup_info['stack_team']:4s} | "
              f"Proj: {best_lineup_info['combo_proj']:5.1f} | "
              f"Actual: {best_actual:6.1f} ({best_lineup_info['matched']}/9) | "
              f"Cash: {cash_line:5.1f} | {status}"
              f"{' | 1ST!' if is_first else ''}")

        results.append({
            'date': date_str,
            'n_teams': n_teams,
            'stack_team': best_lineup_info['stack_team'],
            'combo_proj': best_lineup_info['combo_proj'],
            'actual': best_actual,
            'matched': best_lineup_info['matched'],
            'scratched': best_lineup_info['scratched'],
            'salary': best_lineup_info['salary'],
            'cash_line': cash_line,
            'first_place': first_place,
            'is_cash': is_cash,
            'is_first': is_first,
        })

    conn.close()

    # Summary
    r = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"  Dates: {len(r)}")
    print(f"  Avg actual FPTS: {r['actual'].mean():.1f}")
    print(f"  Median actual FPTS: {r['actual'].median():.1f}")
    print(f"  Avg matched: {r['matched'].mean():.1f}/9")

    with_cash = r[r['cash_line'] > 0]
    if len(with_cash) > 0:
        n_cash = with_cash['is_cash'].sum()
        cash_rate = n_cash / len(with_cash) * 100
        print(f"  Cash rate: {cash_rate:.1f}% ({n_cash}/{len(with_cash)})")

        n_first = with_cash['is_first'].sum() if 'is_first' in with_cash else 0
        first_rate = n_first / len(with_cash) * 100
        print(f"  1st place rate: {first_rate:.1f}% ({n_first}/{len(with_cash)})")

        avg_gap = (with_cash['cash_line'] - with_cash['actual']).mean()
        print(f"  Avg gap to cash: {avg_gap:+.1f}")

    # By slate size
    def slate_type(n):
        if n >= 20: return 'MAIN'
        if n >= 12: return 'LARGE'
        if n >= 6: return 'MED'
        return 'SHORT'

    r['stype'] = r['n_teams'].apply(slate_type)
    for st in ['MAIN', 'LARGE', 'MED']:
        sub = with_cash[with_cash['date'].isin(r[r['stype'] == st]['date'])]
        if len(sub) > 0:
            cr = sub['is_cash'].mean() * 100
            avg = sub['actual'].mean()
            print(f"    {st:6s}: {cr:5.1f}% cash, {avg:5.1f} avg ({len(sub)} dates)")

    # Save
    output = BACKTESTS_DIR / f"line_backtest_{'conf' if confirmed_only else 'all'}.csv"
    r.to_csv(str(output), index=False)
    print(f"\n  Saved: {output}")

    return r


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2025-11-07')
    parser.add_argument('--top', type=int, default=3, help='Top N stacks to try per date')
    parser.add_argument('--all-players', action='store_true', help='Include unconfirmed players')
    args = parser.parse_args()

    run_line_backtest(
        start_date=args.start,
        n_top_stacks=args.top,
        confirmed_only=not args.all_players,
    )
