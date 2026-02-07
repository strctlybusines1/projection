"""
Fetch actual DraftKings NHL fantasy scores from NHL API boxscores.

Uses the same DK scoring formulas from config.py to calculate actual FPTS
from official NHL box score stats. This gives you the ground truth to compare
against your projections.

Usage:
    # From command line:
    python actual_scores.py 2026-01-29                    # Fetch scores for a date
    python actual_scores.py 2026-01-29 --save             # Fetch + save to daily_projections/
    python actual_scores.py 2026-01-29 --compare          # Fetch + compare to your projections

    # From other modules:
    from actual_scores import fetch_actual_scores
    actuals = fetch_actual_scores('2026-01-29')  # Returns DataFrame
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import argparse
import sys
import time

from nhl_api import NHLAPIClient
from config import (
    calculate_skater_fantasy_points,
    calculate_goalie_fantasy_points,
    DAILY_PROJECTIONS_DIR,
    TEAM_FULL_NAME_TO_ABBREV,
)


def _get_game_ids_for_date(api: NHLAPIClient, date: str) -> List[dict]:
    """
    Get all game IDs for a given date.

    Args:
        api: NHLAPIClient instance
        date: Date string YYYY-MM-DD

    Returns:
        List of dicts with game_id, away_abbrev, home_abbrev, game_state
    """
    sched = api.get_schedule(date)
    games = []

    for day_data in sched.get('gameWeek', []):
        if day_data.get('date') == date:
            for game in day_data.get('games', []):
                games.append({
                    'game_id': game['id'],
                    'away_abbrev': game.get('awayTeam', {}).get('abbrev', ''),
                    'home_abbrev': game.get('homeTeam', {}).get('abbrev', ''),
                    'game_state': game.get('gameState', ''),
                })
            break

    return games


def _parse_boxscore_skaters(box: dict, team_abbrev: str, side: str) -> List[dict]:
    """
    Extract skater stats from a boxscore for DK scoring.

    Args:
        box: Full boxscore dict from NHL API
        team_abbrev: Team abbreviation (e.g. 'PIT')
        side: 'awayTeam' or 'homeTeam'

    Returns:
        List of player stat dicts
    """
    pbgs = box.get('playerByGameStats', {}).get(side, {})
    landing_scoring = box.get('summary', {}).get('scoring', [])

    # Build lookup of shorthanded goals/assists from scoring summary
    sh_goals = {}   # player_id -> count
    sh_assists = {}  # player_id -> count
    for period in landing_scoring:
        for goal in period.get('goals', []):
            if goal.get('strength') == 'sh':
                scorer_id = goal.get('playerId')
                if scorer_id:
                    sh_goals[scorer_id] = sh_goals.get(scorer_id, 0) + 1
                for assist in goal.get('assists', []):
                    a_id = assist.get('playerId')
                    if a_id:
                        sh_assists[a_id] = sh_assists.get(a_id, 0) + 1

    players = []
    for pos_group in ['forwards', 'defense']:
        for p in pbgs.get(pos_group, []):
            pid = p.get('playerId')
            toi = p.get('toi', '0:00')

            # Skip players with 0 TOI (scratches listed in boxscore)
            if toi == '0:00':
                continue

            name = p.get('name', {})
            if isinstance(name, dict):
                name = name.get('default', '')

            players.append({
                'player_id': pid,
                'name': name,
                'team': team_abbrev,
                'position': p.get('position', 'F'),
                'player_type': 'skater',
                'goals': p.get('goals', 0),
                'assists': p.get('assists', 0),
                'shots': p.get('sog', 0),
                'blocks': p.get('blockedShots', 0),
                'sh_goals': sh_goals.get(pid, 0),
                'sh_assists': sh_assists.get(pid, 0),
                'shootout_goals': 0,  # Updated later if game went to SO
                'hits': p.get('hits', 0),
                'pim': p.get('pim', 0),
                'pp_goals': p.get('powerPlayGoals', 0),
                'toi': toi,
            })

    return players


def _parse_boxscore_goalies(box: dict, team_abbrev: str, side: str,
                             game_outcome: dict) -> List[dict]:
    """
    Extract goalie stats from a boxscore for DK scoring.

    Args:
        box: Full boxscore dict from NHL API
        team_abbrev: Team abbreviation
        side: 'awayTeam' or 'homeTeam'
        game_outcome: gameOutcome dict with lastPeriodType

    Returns:
        List of goalie stat dicts
    """
    pbgs = box.get('playerByGameStats', {}).get(side, {})

    goalies = []
    for g in pbgs.get('goalies', []):
        toi = g.get('toi', '0:00')
        if toi == '0:00':
            continue

        name = g.get('name', {})
        if isinstance(name, dict):
            name = name.get('default', '')

        saves = g.get('saves', 0)
        goals_against = g.get('goalsAgainst', 0)
        decision = g.get('decision', '')

        win = decision == 'W'
        ot_loss = decision == 'O'  # NHL API: 'O' = OT/SO loss, 'W' = win, 'L' = reg loss
        shutout = goals_against == 0 and g.get('starter', False)

        # Check for shootout — if game went to SO, don't count SO goals against goalie
        # NHL API goalsAgainst already excludes shootout goals

        goalies.append({
            'player_id': g.get('playerId'),
            'name': name,
            'team': team_abbrev,
            'position': 'G',
            'player_type': 'goalie',
            'saves': saves,
            'goals_against': goals_against,
            'win': win,
            'ot_loss': ot_loss,
            'shutout': shutout,
            'decision': decision,
            'save_pct': g.get('savePctg', 0),
            'shots_against': g.get('shotsAgainst', 0),
            'starter': g.get('starter', False),
            'toi': toi,
        })

    return goalies


def fetch_actual_scores(date: str, verbose: bool = True) -> pd.DataFrame:
    """
    Fetch actual DK fantasy scores for all players on a given date.

    Pulls boxscores from NHL API, extracts individual stats, and calculates
    DraftKings fantasy points using the same formulas as config.py.

    Args:
        date: Date string in YYYY-MM-DD format
        verbose: Print progress

    Returns:
        DataFrame with columns:
            name, team, position, player_type, actual_fpts,
            goals, assists, shots, blocks, sh_goals, sh_assists (skaters)
            saves, goals_against, win, shutout (goalies)
    """
    api = NHLAPIClient()

    if verbose:
        print(f"\nFetching actual scores for {date}...")

    # Get all games for the date
    games = _get_game_ids_for_date(api, date)

    if not games:
        if verbose:
            print(f"  No games found for {date}")
        return pd.DataFrame()

    # Check if games are finished
    unfinished = [g for g in games if g['game_state'] not in ('OFF', 'FINAL')]
    if unfinished:
        if verbose:
            print(f"  Warning: {len(unfinished)} game(s) not yet final")

    if verbose:
        print(f"  Found {len(games)} games")

    all_players = []

    for game in games:
        gid = game['game_id']
        state = game['game_state']

        if state not in ('OFF', 'FINAL'):
            if verbose:
                print(f"  Skipping {game['away_abbrev']}@{game['home_abbrev']} (state: {state})")
            continue

        if verbose:
            print(f"  {game['away_abbrev']} @ {game['home_abbrev']}...", end=' ')

        box = api.get_boxscore(gid)

        # Also get landing page for shorthanded goal detail
        try:
            landing = api.get_game_landing(gid)
            # Inject scoring summary into boxscore for SH parsing
            if 'summary' in landing:
                box['summary'] = landing['summary']
        except Exception:
            pass

        game_outcome = box.get('gameOutcome', {})

        away_abbrev = box.get('awayTeam', {}).get('abbrev', game['away_abbrev'])
        home_abbrev = box.get('homeTeam', {}).get('abbrev', game['home_abbrev'])

        away_score = box.get('awayTeam', {}).get('score', 0)
        home_score = box.get('homeTeam', {}).get('score', 0)

        # Parse skaters
        away_skaters = _parse_boxscore_skaters(box, away_abbrev, 'awayTeam')
        home_skaters = _parse_boxscore_skaters(box, home_abbrev, 'homeTeam')

        # Parse goalies
        away_goalies = _parse_boxscore_goalies(box, away_abbrev, 'awayTeam', game_outcome)
        home_goalies = _parse_boxscore_goalies(box, home_abbrev, 'homeTeam', game_outcome)

        # Parse shootout goals (DK awards 1.5 pts per SO goal to skaters)
        if game_outcome.get('lastPeriodType') == 'SO':
            shootout_events = box.get('summary', {}).get('shootout', {}).get('events', [])
            so_goals = {}  # player_id -> count
            for event in shootout_events:
                if event.get('result') == 'goal':
                    pid = event.get('playerId')
                    if pid:
                        so_goals[pid] = so_goals.get(pid, 0) + 1

            # Apply SO goals to skaters
            for player_list in [away_skaters, home_skaters]:
                for p in player_list:
                    pid = p.get('player_id')
                    if pid in so_goals:
                        p['shootout_goals'] = so_goals[pid]

        all_players.extend(away_skaters)
        all_players.extend(home_skaters)
        all_players.extend(away_goalies)
        all_players.extend(home_goalies)

        if verbose:
            print(f"{away_abbrev} {away_score} - {home_abbrev} {home_score} "
                  f"({len(away_skaters)+len(home_skaters)} skaters, "
                  f"{len(away_goalies)+len(home_goalies)} goalies)")

        time.sleep(0.3)  # Rate limiting

    if not all_players:
        return pd.DataFrame()

    df = pd.DataFrame(all_players)

    # Calculate DK fantasy points
    skater_mask = df['player_type'] == 'skater'
    goalie_mask = df['player_type'] == 'goalie'

    df.loc[skater_mask, 'actual_fpts'] = df[skater_mask].apply(
        lambda r: calculate_skater_fantasy_points(
            goals=r['goals'],
            assists=r['assists'],
            shots=r['shots'],
            blocks=r['blocks'],
            sh_goals=r.get('sh_goals', 0),
            sh_assists=r.get('sh_assists', 0),
            shootout_goals=r.get('shootout_goals', 0),
        ), axis=1
    )

    df.loc[goalie_mask, 'actual_fpts'] = df[goalie_mask].apply(
        lambda r: calculate_goalie_fantasy_points(
            win=r['win'],
            saves=r['saves'],
            goals_against=r['goals_against'],
            shutout=r['shutout'],
            ot_loss=r.get('ot_loss', False),
        ), axis=1
    )

    df['date'] = date
    df = df.sort_values('actual_fpts', ascending=False)

    if verbose:
        print(f"\n  Total: {len(df)} players scored")
        print(f"  Top skater:  {df[skater_mask].iloc[0]['name']} "
              f"({df[skater_mask].iloc[0]['team']}) — "
              f"{df[skater_mask].iloc[0]['actual_fpts']:.1f} FPTS")
        if goalie_mask.any():
            top_g = df[goalie_mask].iloc[0]
            print(f"  Top goalie:  {top_g['name']} ({top_g['team']}) — "
                  f"{top_g['actual_fpts']:.1f} FPTS")

    return df


def save_actual_scores(df: pd.DataFrame, date: str, output_dir: Path = None):
    """Save actual scores CSV to daily_projections folder."""
    if output_dir is None:
        output_dir = Path(__file__).parent / DAILY_PROJECTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    dt = datetime.strptime(date, '%Y-%m-%d')
    filename = f"{dt.strftime('%m_%d_%y')}NHL_actuals.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    print(f"\n  Actual scores saved to: {filepath}")
    return filepath


def _match_player_name(api_name: str, dk_names: List[str]) -> Optional[str]:
    """
    Match NHL API abbreviated name (e.g. "A. DeBrincat") to DK full name
    (e.g. "Alex DeBrincat").

    Strategy:
    1. Exact match
    2. Last name match (API names almost always have correct last name)
    3. Last name match + first initial check
    4. Fall back to fuzzy_match from lines.py
    """
    api_lower = api_name.lower().strip()
    api_parts = api_lower.split()

    if len(api_parts) < 2:
        return None

    api_first = api_parts[0].rstrip('.')
    api_last = ' '.join(api_parts[1:])  # Handle multi-word last names

    # 1. Exact match
    for dk in dk_names:
        if dk.lower().strip() == api_lower:
            return dk

    # 2. Last name match + first initial
    # This is the most common case: "A. DeBrincat" -> "Alex DeBrincat"
    candidates = []
    for dk in dk_names:
        dk_lower = dk.lower().strip()
        dk_parts = dk_lower.split()
        if len(dk_parts) < 2:
            continue
        dk_last = ' '.join(dk_parts[1:])

        if dk_last == api_last:
            # Last names match — check first initial
            dk_first = dk_parts[0]
            if dk_first.startswith(api_first[0]) if api_first else False:
                candidates.append(dk)
            else:
                # Same last name, different initial — still track as fallback
                candidates.append(dk)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Multiple players with same last name — use initial to disambiguate
        for dk in candidates:
            dk_first = dk.lower().strip().split()[0]
            if dk_first.startswith(api_first[0]) if api_first else False:
                return dk
        # If still ambiguous, return first match
        return candidates[0]

    # 3. Fuzzy fallback
    try:
        from lines import find_player_match
        match = find_player_match(api_name, dk_names, threshold=0.75)
        if match:
            return match
    except ImportError:
        pass

    return None


def compare_to_projections(actuals: pd.DataFrame, date: str,
                           backtests_dir: Path = None) -> Optional[pd.DataFrame]:
    """
    Compare actual scores (from NHL API) to DraftKings actual scores
    from your backtest xlsx files, to verify scoring accuracy.

    Also compares to your projections if available in the same file.

    Looks for: backtests/M.DD.YY_nhl_backtest.xlsx
    """
    if backtests_dir is None:
        backtests_dir = Path(__file__).parent / 'backtests'

    if not backtests_dir.exists():
        print("  No backtests/ directory found")
        return None

    # Find backtest file for this date
    dt = datetime.strptime(date, '%Y-%m-%d')
    date_variants = [
        f"{dt.month}.{dt.day:02d}.{dt.strftime('%y')}",   # 1.29.26
        f"{dt.month}.{dt.day}.{dt.strftime('%y')}",        # 1.29.26 (no leading zero)
        f"{dt.month:02d}.{dt.day:02d}.{dt.strftime('%y')}", # 01.29.26
    ]

    bt_file = None
    for f in sorted(backtests_dir.glob('*_nhl_backtest.xlsx')):
        for variant in date_variants:
            if f.name.startswith(variant):
                bt_file = f
                break

    if not bt_file:
        print(f"  No backtest file found for {date}")

        # Fall back: try daily_projections CSV
        proj_dir = Path(__file__).parent / DAILY_PROJECTIONS_DIR
        if proj_dir.exists():
            for f in sorted(proj_dir.glob('*NHLprojections*.csv')):
                for variant in [f"{dt.month}_{dt.day}_{dt.strftime('%y')}",
                                f"{dt.month:02d}_{dt.day:02d}_{dt.strftime('%y')}"]:
                    if variant in f.name:
                        print(f"  Found projection CSV: {f.name}")
                        proj = pd.read_csv(f)
                        if 'projected_fpts' in proj.columns and 'name' in proj.columns:
                            merged = actuals.merge(
                                proj[['name', 'projected_fpts']].drop_duplicates('name'),
                                on='name', how='inner'
                            )
                            if not merged.empty:
                                merged['error'] = merged['projected_fpts'] - merged['actual_fpts']
                                merged['abs_error'] = merged['error'].abs()
                                _print_accuracy_report(merged, date)
                                return merged
        return None

    print(f"  Comparing to: {bt_file.name}")

    # Read DK actual scores from the Actual sheet
    try:
        dk_actuals = pd.read_excel(bt_file, 'Actual')
    except Exception as e:
        print(f"  Error reading Actual sheet: {e}")
        return None

    # Normalize column names
    dk_col_map = {}
    for col in dk_actuals.columns:
        if col.lower() in ('player', 'name'):
            dk_col_map[col] = 'dk_name'
        elif col.lower() in ('fpts', 'actual_fpts', 'fantasy points'):
            dk_col_map[col] = 'dk_fpts'
    dk_actuals = dk_actuals.rename(columns=dk_col_map)

    if 'dk_name' not in dk_actuals.columns or 'dk_fpts' not in dk_actuals.columns:
        print(f"  Could not find Player/FPTS columns in Actual sheet")
        return None

    # Merge NHL API actuals with DK actuals on player name
    # API uses abbreviated names (e.g. "A. DeBrincat"), DK uses full names
    # Challenge: DK Actual sheet has no team column, so when last names collide
    # (e.g. T. Raddysh vs D. Raddysh) we use FPTS proximity to disambiguate
    dk_names = dk_actuals['dk_name'].tolist()

    # Build a dk_name -> dk_fpts lookup for disambiguation
    dk_fpts_lookup = dict(zip(dk_actuals['dk_name'], dk_actuals['dk_fpts']))

    # First pass: find all potential matches per DK name
    dk_to_api_candidates = {}  # dk_name -> list of (api_row, fpts_diff)
    for _, row in actuals.iterrows():
        match = _match_player_name(row['name'], dk_names)
        if match:
            dk_fpts = dk_fpts_lookup.get(match, 0)
            diff = abs(row['actual_fpts'] - dk_fpts)
            if match not in dk_to_api_candidates:
                dk_to_api_candidates[match] = []
            dk_to_api_candidates[match].append((row, diff))

    # Second pass: for each DK name, pick the API player with closest FPTS
    matches = []
    used_api_names = set()
    for dk_name, candidates in dk_to_api_candidates.items():
        # Sort by FPTS difference — closest match wins
        candidates.sort(key=lambda x: x[1])
        for row, diff in candidates:
            if row['name'] not in used_api_names:
                dk_fpts = dk_fpts_lookup.get(dk_name, 0)
                matches.append({
                    'name': dk_name,
                    'api_name': row['name'],
                    'team': row['team'],
                    'position': row['position'],
                    'player_type': row['player_type'],
                    'api_fpts': row['actual_fpts'],
                    'dk_fpts': dk_fpts,
                    'fpts_diff': row['actual_fpts'] - dk_fpts,
                })
                used_api_names.add(row['name'])
                break
    if not matches:
        print("  Could not match any players between API actuals and DK actuals")
        return None

    compare_df = pd.DataFrame(matches)

    # Report: how accurate is the NHL API scoring vs DK
    print(f"\n  {'=' * 70}")
    print(f"  NHL API vs DRAFTKINGS SCORING VERIFICATION — {date}")
    print(f"  {'=' * 70}")
    print(f"\n  Matched: {len(compare_df)} players")

    exact_matches = (compare_df['fpts_diff'].abs() < 0.01).sum()
    close_matches = (compare_df['fpts_diff'].abs() <= 0.5).sum()
    print(f"  Exact matches (±0):    {exact_matches} ({exact_matches/len(compare_df)*100:.0f}%)")
    print(f"  Close matches (±0.5):  {close_matches} ({close_matches/len(compare_df)*100:.0f}%)")
    print(f"  Mean abs difference:   {compare_df['fpts_diff'].abs().mean():.2f}")

    # Show mismatches
    mismatches = compare_df[compare_df['fpts_diff'].abs() > 0.01].sort_values(
        'fpts_diff', key=abs, ascending=False
    )
    if not mismatches.empty:
        print(f"\n  MISMATCHES ({len(mismatches)} players):")
        print(f"  {'Name':<25} {'Team':<5} {'API':>6} {'DK':>6} {'Diff':>6}")
        print(f"  {'-'*50}")
        for _, r in mismatches.head(10).iterrows():
            print(f"  {r['name']:<25} {r['team']:<5} {r['api_fpts']:6.1f} "
                  f"{r['dk_fpts']:6.1f} {r['fpts_diff']:+6.1f}")
    else:
        print(f"\n  ✓ All scores match DraftKings exactly!")

    # Also show projection accuracy if backtest has it
    try:
        proj_sheet = pd.read_excel(bt_file, 'Projection')
        if 'projected_fpts' in proj_sheet.columns and 'actual' in proj_sheet.columns:
            print(f"\n  (Projection accuracy already in backtest file — use your backtest module for full analysis)")
    except Exception:
        pass

    return compare_df


def _print_accuracy_report(merged: pd.DataFrame, date: str):
    """Print projection accuracy report from merged data."""
    skaters = merged[merged['player_type'] == 'skater']
    goalies = merged[merged['player_type'] == 'goalie']

    print(f"\n  {'=' * 70}")
    print(f"  PROJECTION ACCURACY — {date}")
    print(f"  {'=' * 70}")

    for label, subset in [('SKATERS', skaters), ('GOALIES', goalies)]:
        if subset.empty:
            continue
        mae = subset['abs_error'].mean()
        bias = subset['error'].mean()
        corr = subset[['projected_fpts', 'actual_fpts']].corr().iloc[0, 1]
        print(f"\n  {label} ({len(subset)} matched):")
        print(f"    MAE:         {mae:.2f}")
        print(f"    Bias:        {bias:+.2f}")
        print(f"    Correlation: {corr:.3f}")
        print(f"    Avg Proj:    {subset['projected_fpts'].mean():.2f}")
        print(f"    Avg Actual:  {subset['actual_fpts'].mean():.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch actual DK NHL scores from NHL API')
    parser.add_argument('date', type=str, help='Date to fetch (YYYY-MM-DD)')
    parser.add_argument('--save', action='store_true', help='Save to daily_projections/')
    parser.add_argument('--compare', action='store_true', help='Compare to your projections')
    args = parser.parse_args()

    actuals = fetch_actual_scores(args.date)

    if actuals.empty:
        print("No scores found.")
        sys.exit(1)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"ACTUAL DK SCORES — {args.date}")
    print(f"{'=' * 70}")

    skaters = actuals[actuals['player_type'] == 'skater'].head(15)
    print(f"\nTop 15 Skaters:")
    print(f"  {'Name':<25} {'Team':<5} {'FPTS':>6} {'G':>3} {'A':>3} {'SOG':>4} {'BLK':>4}")
    print(f"  {'-'*55}")
    for _, r in skaters.iterrows():
        print(f"  {r['name']:<25} {r['team']:<5} {r['actual_fpts']:6.1f} "
              f"{int(r['goals']):3d} {int(r['assists']):3d} {int(r['shots']):4d} {int(r['blocks']):4d}")

    goalies = actuals[actuals['player_type'] == 'goalie']
    if not goalies.empty:
        print(f"\nGoalies:")
        print(f"  {'Name':<25} {'Team':<5} {'FPTS':>6} {'SV':>4} {'GA':>3} {'W':>3}")
        print(f"  {'-'*50}")
        for _, r in goalies.iterrows():
            w = 'W' if r.get('win') else ('OTL' if r.get('ot_loss') else 'L')
            print(f"  {r['name']:<25} {r['team']:<5} {r['actual_fpts']:6.1f} "
                  f"{int(r['saves']):4d} {int(r['goals_against']):3d} {w:>3}")

    if args.save:
        save_actual_scores(actuals, args.date)

    if args.compare:
        compare_to_projections(actuals, args.date)
