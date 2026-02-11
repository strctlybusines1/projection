#!/usr/bin/env python3
"""
NHL DFS Projection Tool - Main CLI

Usage:
    python main.py                      # Generate projections for today's slate
    python main.py --date 2026-01-22    # Generate for specific date
    python main.py --salaries file.csv  # Use specific salary file
    python main.py --stacks             # Show stacking recommendations
    python main.py --show-injuries      # Display injury report
    python main.py --include-dtd        # Include Day-to-Day players
    python main.py --no-injuries        # Disable injury filtering
"""

import argparse
import os
import pandas as pd
import numpy as np
import re
import requests
from datetime import datetime
from pathlib import Path
import sys

from data_pipeline import NHLDataPipeline
from projections import NHLProjectionModel
from optimizer import NHLLineupOptimizer
from config import (
    calculate_skater_fantasy_points,
    calculate_goalie_fantasy_points,
    INJURY_STATUSES_EXCLUDE,
    DAILY_SALARIES_DIR,
    VEGAS_DIR,
    DAILY_PROJECTIONS_DIR,
    TEAM_FULL_NAME_TO_ABBREV,
)
from lines import LinesScraper, StackBuilder, print_team_lines, find_player_match
from ownership import OwnershipModel, print_ownership_report
# Edge stats now handled in data_pipeline.py
from contest_roi import (
    ContestProfile,
    recommend_leverage,
    score_lineups as contest_score_lineups,
    print_leverage_recommendation,
    PAYOUT_PRESETS,
)
from single_entry import SingleEntrySelector, print_se_lineup, ContestProfile as SEContestProfile, prompt_contest_profile
from tournament_equity import TournamentEquitySelector, print_te_lineup
from validate import run_validation


def normalize_position(pos: str) -> str:
    """
    Normalize NHL positions to DraftKings format.

    L, LW, R, RW -> W (Wing)
    C -> C (Center)
    D -> D (Defense)
    G -> G (Goalie)
    """
    if pd.isna(pos):
        return 'W'
    pos = str(pos).upper().strip()
    if pos in ('L', 'LW', 'R', 'RW'):
        return 'W'
    return pos


def normalize_positions_column(df: pd.DataFrame, col: str = 'position') -> pd.DataFrame:
    """Apply position normalization to a DataFrame column."""
    if col in df.columns:
        df[col] = df[col].apply(normalize_position)
    return df


def load_dk_salaries(csv_path: str) -> pd.DataFrame:
    """Load and parse DraftKings salary CSV."""
    print(f"Loading DK salaries from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize columns
    df = df.rename(columns={
        'Name': 'dk_name',
        'Roster Position': 'dk_position',
        'Salary': 'salary',
        'TeamAbbrev': 'team',
        'AvgPointsPerGame': 'dk_avg_fpts',
        'Position': 'position',
        'ID': 'dk_id',
    })

    # Parse position (first position listed) and normalize L/R to W
    if 'position' in df.columns:
        df['base_position'] = df['position'].str.upper()
        df = normalize_positions_column(df, 'position')
        df = normalize_positions_column(df, 'base_position')

    # Normalize team codes
    if 'team' in df.columns:
        df['team'] = df['team'].str.upper()

    print(f"  Loaded {len(df)} players")
    print(f"  Salary range: ${df['salary'].min():,} - ${df['salary'].max():,}")

    return df


def merge_projections_with_salaries(projections: pd.DataFrame,
                                     salaries: pd.DataFrame,
                                     player_type: str = 'skater') -> pd.DataFrame:
    """Merge our projections with DK salaries."""

    # Create matching keys
    proj = projections.copy()
    sal = salaries.copy()

    # Clean names for matching (transliterate unicode accents first)
    import unicodedata
    _UMLAUT_MAP = {'ü': 'ue', 'ö': 'oe', 'ä': 'ae', 'ß': 'ss'}
    def _clean_name(s):
        for orig, repl in _UMLAUT_MAP.items():
            s = s.replace(orig, repl).replace(orig.upper(), repl.capitalize())
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
        s = s.lower().strip()
        s = re.sub(r'[^a-z\s]', '', s)
        return s

    proj['name_clean'] = proj['name'].apply(_clean_name)
    sal['name_clean'] = sal['dk_name'].apply(_clean_name)

    # Rename DK position to avoid collision with projection position
    if 'position' in sal.columns:
        sal = sal.rename(columns={'position': 'dk_pos'})

    # Merge - include dk_pos (actual position) for roster eligibility
    merge_cols = ['name_clean', 'salary', 'dk_position', 'dk_id', 'dk_avg_fpts']
    if 'dk_pos' in sal.columns:
        merge_cols.append('dk_pos')
    if 'Game Info' in sal.columns:
        sal = sal.rename(columns={'Game Info': 'game_info'})
        merge_cols.append('game_info')

    merged = proj.merge(
        sal[merge_cols],
        on='name_clean',
        how='inner'
    )

    if len(merged) == 0:
        print(f"  Warning: No matches found for {player_type}s!")
        print(f"  Sample projection names: {proj['name'].head(5).tolist()}")
        print(f"  Sample salary names: {sal['dk_name'].head(5).tolist()}")
        return merged

    # ==================== DK Season Average Blending ====================
    # Blend our projection with DK's season average to anchor toward market consensus
    if 'dk_avg_fpts' in merged.columns:
        from config import DK_AVG_BLEND_WEIGHT
        valid_dk = merged['dk_avg_fpts'].notna() & (merged['dk_avg_fpts'] > 0)
        merged.loc[valid_dk, 'projected_fpts'] = (
            DK_AVG_BLEND_WEIGHT * merged.loc[valid_dk, 'projected_fpts'] +
            (1 - DK_AVG_BLEND_WEIGHT) * merged.loc[valid_dk, 'dk_avg_fpts']
        )

    # Recalculate floor/ceiling after blending
    from projections import FLOOR_MULTIPLIER
    merged['floor'] = merged['projected_fpts'] * FLOOR_MULTIPLIER
    if player_type == 'goalie':
        merged['ceiling'] = merged['projected_fpts'] * 2.0 + 10
    else:
        merged['ceiling'] = merged['projected_fpts'] * 2.5 + 5

    # Calculate value metrics
    merged['value'] = merged['projected_fpts'] / (merged['salary'] / 1000)
    merged['edge'] = merged['projected_fpts'] - merged['dk_avg_fpts']

    # Drop temp columns
    merged = merged.drop(columns=['name_clean'], errors='ignore')

    return merged


def print_projections_table(df: pd.DataFrame, title: str, n: int = 25):
    """Print formatted projections table."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    cols = ['name', 'team', 'position', 'salary', 'projected_fpts', 'dk_avg_fpts', 'edge', 'value']
    cols = [c for c in cols if c in df.columns]

    display_df = df.nlargest(n, 'projected_fpts')[cols].copy()

    # Normalize positions for display (L/R -> W)
    if 'position' in display_df.columns:
        display_df['position'] = display_df['position'].apply(normalize_position)

    # Format columns
    if 'salary' in display_df.columns:
        display_df['salary'] = display_df['salary'].apply(lambda x: f"${x:,}")
    if 'projected_fpts' in display_df.columns:
        display_df['projected_fpts'] = display_df['projected_fpts'].round(1)
    if 'dk_avg_fpts' in display_df.columns:
        display_df['dk_avg_fpts'] = display_df['dk_avg_fpts'].round(1)
    if 'edge' in display_df.columns:
        display_df['edge'] = display_df['edge'].round(1)
    if 'value' in display_df.columns:
        display_df['value'] = display_df['value'].round(2)

    # Rename for display
    display_df.columns = ['Name', 'Team', 'Pos', 'Salary', 'Proj', 'DK Avg', 'Edge', 'Value'][:len(cols)]

    print(display_df.to_string(index=False))


def print_value_plays(df: pd.DataFrame, title: str, n: int = 15):
    """Print top value plays."""
    print(f"\n{'=' * 80}")
    print(f" {title} (Sorted by Value)")
    print(f"{'=' * 80}")

    cols = ['name', 'team', 'position', 'salary', 'projected_fpts', 'value']
    cols = [c for c in cols if c in df.columns]

    # Filter to reasonable salary range for value plays
    value_df = df[df['salary'] <= 6500].copy()
    display_df = value_df.nlargest(n, 'value')[cols].copy()

    # Normalize positions for display (L/R -> W)
    if 'position' in display_df.columns:
        display_df['position'] = display_df['position'].apply(normalize_position)

    if 'salary' in display_df.columns:
        display_df['salary'] = display_df['salary'].apply(lambda x: f"${x:,}")
    if 'projected_fpts' in display_df.columns:
        display_df['projected_fpts'] = display_df['projected_fpts'].round(1)
    if 'value' in display_df.columns:
        display_df['value'] = display_df['value'].round(2)

    display_df.columns = ['Name', 'Team', 'Pos', 'Salary', 'Proj', 'Value'][:len(cols)]

    print(display_df.to_string(index=False))


def print_stacking_recommendations(stack_builder: StackBuilder, projections_df: pd.DataFrame, teams: list):
    """Print stacking recommendations for all teams on the slate."""
    print(f"\n{'=' * 80}")
    print(" STACKING RECOMMENDATIONS")
    print(f"{'=' * 80}")

    all_stacks = []

    for team in teams:
        stacks = stack_builder.get_best_stacks(team, projections_df)
        for stack in stacks:
            if stack.get('type') in ['PP1', 'Line1']:
                all_stacks.append(stack)

    # Sort by projected total
    all_stacks.sort(key=lambda x: x.get('projected_total', 0), reverse=True)

    for stack in all_stacks[:10]:
        stack_type = stack.get('type', '')
        team = stack.get('team', '')
        players = stack.get('players', [])
        proj = stack.get('projected_total', 0)
        corr = stack.get('correlation', 0)

        print(f"\n{team} {stack_type} (corr: {corr:.0%}, proj: {proj:.1f} pts)")
        for p in players:
            # Find matching player in projections
            match = find_player_match(p, projections_df['name'].tolist())
            if match:
                player_data = projections_df[projections_df['name'] == match].iloc[0]
                print(f"  - {match:<25} ${player_data['salary']:,}  {player_data['projected_fpts']:.1f} pts")
            else:
                print(f"  - {p:<25} (not in DK pool)")


def print_line_update_status(stack_builder: StackBuilder):
    """Print when line combinations were last updated."""
    from datetime import datetime as dt, timezone

    print(f"\n{'=' * 80}")
    print(" LINE COMBINATION UPDATE STATUS")
    print(f"{'=' * 80}")

    timestamps = stack_builder.get_update_timestamps()

    if not timestamps:
        print("  No update timestamps available")
        return

    # Parse and display timestamps
    now = dt.now(timezone.utc)
    stale_teams = []

    for team, ts in sorted(timestamps.items()):
        if not ts:
            print(f"  {team:<5} No update time available")
            continue

        try:
            # Parse ISO format timestamp (UTC)
            update_time = dt.fromisoformat(ts.replace('Z', '+00:00'))
            age = now - update_time
            hours_ago = age.total_seconds() / 3600

            # Convert to local time for display
            local_time = update_time.astimezone()

            if hours_ago < 0:
                age_str = "just now"
            elif hours_ago < 1:
                age_str = f"{int(age.total_seconds() / 60)} min ago"
            elif hours_ago < 24:
                age_str = f"{hours_ago:.1f} hrs ago"
            else:
                age_str = f"{int(hours_ago / 24)} days ago"
                stale_teams.append(team)

            print(f"  {team:<5} Updated {age_str:<15} ({local_time.strftime('%m/%d %I:%M %p')})")
        except Exception as e:
            print(f"  {team:<5} {ts}")

    if stale_teams:
        print(f"\n  WARNING: {', '.join(stale_teams)} have stale data (>24 hrs old)")


def print_injury_report(injuries: pd.DataFrame, teams: list = None):
    """
    Print injury report for slate teams.

    Args:
        injuries: DataFrame with injury data from MoneyPuck
        teams: Optional list of teams to filter (slate teams)
    """
    print(f"\n{'=' * 80}")
    print(" INJURY REPORT")
    print(f"{'=' * 80}")

    if injuries.empty:
        print("  No injury data available")
        return

    # Filter to slate teams if provided
    if teams and 'team' in injuries.columns:
        injuries = injuries[injuries['team'].isin(teams)]

    if injuries.empty:
        print("  No injured players on today's slate")
        return

    # Group by injury status
    for status in ['IR', 'IR-LT', 'IR-NR', 'O', 'DTD']:
        status_df = injuries[injuries.get('injury_status', '') == status]

        if status_df.empty:
            continue

        status_label = {
            'IR': 'Injured Reserve',
            'IR-LT': 'IR Long-Term',
            'IR-NR': 'IR Non-Roster',
            'O': 'Out',
            'DTD': 'Day-to-Day'
        }.get(status, status)

        print(f"\n  {status_label} ({len(status_df)} players):")
        print(f"  {'-' * 60}")

        # Sort by team
        status_df = status_df.sort_values('team')

        for _, row in status_df.iterrows():
            team = row.get('team', '???')
            name = row.get('player_name', 'Unknown')
            position = row.get('position', '?')
            games_to_miss = row.get('games_to_miss', '')
            return_date = row.get('return_date', '')

            games_str = f", ~{int(games_to_miss)} games" if pd.notna(games_to_miss) and games_to_miss else ""
            date_str = f", ETA: {return_date.strftime('%m/%d')}" if pd.notna(return_date) else ""

            print(f"    {team:<5} {name:<25} ({position}){games_str}{date_str}")

    # Summary
    total = len(injuries)
    severe = len(injuries[injuries['injury_status'].isin(INJURY_STATUSES_EXCLUDE)])
    dtd = len(injuries[injuries['injury_status'] == 'DTD'])

    print(f"\n  Summary: {total} total injured ({severe} out, {dtd} day-to-day)")


def print_confirmed_goalies(stack_builder: StackBuilder, projections_df: pd.DataFrame):
    """Print confirmed starting goalies with quality tier labels."""
    print(f"\n{'=' * 80}")
    print(" CONFIRMED STARTING GOALIES")
    print(f"{'=' * 80}")

    goalies = stack_builder.get_all_starting_goalies()

    if not goalies:
        print("  No confirmed starters found")
        return

    for team, goalie_name in goalies.items():
        match = find_player_match(goalie_name, projections_df['name'].tolist())
        if match:
            player_data = projections_df[projections_df['name'] == match].iloc[0]
            tier = player_data.get('goalie_tier', '?')
            tier_str = f"[{tier}]"
            warning = " ⚠ WARNING: BACKUP-TIER GOALIE" if tier == 'BACKUP' else ""
            print(f"  {team:<5} {match:<25} {tier_str:<10} ${player_data['salary']:,}  "
                  f"{player_data['projected_fpts']:.1f} pts{warning}")
        else:
            print(f"  {team:<5} {goalie_name:<25} (not in DK pool)")


def _ml_to_implied_team_total(home_ml, away_ml, game_total):
    """Derive per-team implied totals from game total + moneylines.

    Convert American moneylines to implied win probabilities, normalize
    to remove the vig, then multiply each side's probability by the
    game total.  Returns (away_tt, home_tt) rounded to 1 decimal, or
    (None, None) when any input is missing.
    """
    if home_ml is None or away_ml is None or game_total is None:
        return None, None

    def _ml_to_prob(ml):
        ml = float(ml)
        if ml > 0:
            return 100.0 / (ml + 100.0)
        else:
            return (-ml) / (-ml + 100.0)

    home_prob = _ml_to_prob(home_ml)
    away_prob = _ml_to_prob(away_ml)
    total_prob = home_prob + away_prob  # > 1 due to vig
    home_fair = home_prob / total_prob
    away_fair = away_prob / total_prob
    home_tt = round(home_fair * game_total, 1)
    away_tt = round(away_fair * game_total, 1)
    return away_tt, home_tt


def _fetch_odds_api():
    """Fetch NHL odds from The Odds API.  Returns (games_list, error_string)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass

    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        return None, "ODDS_API_KEY not set in .env"
    url = (
        "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
        "?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey=" + key
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        return None, str(e)

    games = []
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        game_total = None
        home_ml = away_ml = None
        spread_home = spread_away = None
        for bm in ev.get("bookmakers", [])[:1]:
            for m in bm.get("markets", []):
                if m.get("key") == "totals" and m.get("outcomes"):
                    for o in m["outcomes"]:
                        if o.get("name") == "Over":
                            game_total = o.get("point")
                            break
                elif m.get("key") == "h2h" and m.get("outcomes"):
                    for o in m["outcomes"]:
                        if o.get("name") == home:
                            home_ml = o.get("price")
                        elif o.get("name") == away:
                            away_ml = o.get("price")
                elif m.get("key") == "spreads" and m.get("outcomes"):
                    for o in m["outcomes"]:
                        if o.get("name") == home:
                            spread_home = o.get("point")
                        elif o.get("name") == away:
                            spread_away = o.get("point")
        away_tt, home_tt = _ml_to_implied_team_total(home_ml, away_ml, game_total)
        games.append({
            "matchup": f"{away} @ {home}",
            "game_total": game_total,
            "away_team_total": away_tt,
            "home_team_total": home_tt,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "spread_home": spread_home,
            "spread_away": spread_away,
        })
    games.sort(key=lambda x: (x["game_total"] is None, -(x["game_total"] or 0)))
    return games, None


def print_vegas_ranking(vegas_path: str = None):
    """
    Print Vegas game total ranking with derived team totals.

    Fetches live odds from The Odds API (if ODDS_API_KEY is set).
    Falls back to a CSV path if the API is unavailable and vegas_path
    is provided.

    Ranks games by game_total descending and labels:
    - PRIMARY TARGET for highest total
    - SECONDARY for 2nd highest
    - TERTIARY for 3rd highest

    Returns:
        List of game dicts (same as get_vegas_games) for reuse by caller.
    """
    print(f"\n{'=' * 80}")
    print(" VEGAS GAME RANKING (by Game Total)")
    print(f"{'=' * 80}")

    games, api_err = _fetch_odds_api()

    if games is None and vegas_path:
        # Fallback to CSV
        print(f"  (API unavailable: {api_err} — using CSV fallback)")
        games = _load_vegas_csv(vegas_path)

    if not games:
        msg = api_err or "No Vegas data available"
        print(f"  {msg}")
        return []

    if api_err is None:
        print("  Source: The Odds API (live)")
    print()

    labels = ['PRIMARY TARGET', 'SECONDARY', 'TERTIARY']

    for i, g in enumerate(games):
        label = labels[i] if i < len(labels) else ''
        matchup = g.get("matchup", f"Game {i+1}")
        total = g.get("game_total", "—")
        away_tt = g.get("away_team_total")
        home_tt = g.get("home_team_total")

        spread_val = g.get("spread_away")
        spread = f"  Spread: {spread_val}" if spread_val is not None else ""
        ml_val = g.get("away_ml")
        ml = f"  ML: {ml_val}" if ml_val is not None else ""

        tt_str = ""
        if away_tt is not None and home_tt is not None:
            # Extract team abbreviations from "Away Team @ Home Team"
            parts = matchup.split(" @ ")
            away_label = parts[0].strip().split()[-1] if parts else "Away"
            home_label = parts[1].strip().split()[-1] if len(parts) > 1 else "Home"
            tt_str = f"  ({away_label} {away_tt} / {home_label} {home_tt})"

        if label:
            print(f"\n  >>> {label} <<<")
        print(f"  {matchup:<40} Total: {total}{tt_str}{spread}{ml}")

    print()
    return games or []


def _load_vegas_csv(vegas_path: str):
    """Load a Vegas CSV as fallback and return the same list-of-dicts shape."""
    try:
        df = pd.read_csv(vegas_path)
    except Exception:
        return []
    if df.empty or "team" not in df.columns or "opp" not in df.columns:
        return []
    total_col = None
    for c in ["game_total", "total", "Total"]:
        if c in df.columns:
            total_col = c
            break
    if not total_col:
        return []

    grouped = {}
    for _, row in df.iterrows():
        t, o = str(row["team"]).strip(), str(row["opp"]).strip()
        key = tuple(sorted([t, o]))
        if key not in grouped:
            grouped[key] = {"away": key[0], "home": key[1], "rows": []}
        grouped[key]["rows"].append(row)

    games = []
    for key, g in grouped.items():
        rows = g["rows"]
        away, home = g["away"], g["home"]
        away_row = next((r for r in rows if r["team"] == away), None)
        home_row = next((r for r in rows if r["team"] == home), None)
        game_total = float(away_row[total_col]) if away_row is not None and pd.notna(away_row.get(total_col)) else None
        away_ml = int(away_row["moneyline"]) if away_row is not None and pd.notna(away_row.get("moneyline")) else None
        home_ml = int(home_row["moneyline"]) if home_row is not None and pd.notna(home_row.get("moneyline")) else None
        spread_away = float(away_row["spread"]) if away_row is not None and pd.notna(away_row.get("spread")) else None
        away_tt, home_tt = _ml_to_implied_team_total(home_ml, away_ml, game_total)
        games.append({
            "matchup": f"{away} @ {home}",
            "game_total": game_total,
            "away_team_total": away_tt,
            "home_team_total": home_tt,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "spread_away": spread_away,
        })
    games.sort(key=lambda x: (x["game_total"] is None, -(x["game_total"] or 0)))
    return games


def get_vegas_games(vegas_csv_path: str = None):
    """Fetch Vegas game data (API first, CSV fallback). Returns list of game dicts."""
    games, _ = _fetch_odds_api()
    if games is None and vegas_csv_path:
        games = _load_vegas_csv(vegas_csv_path)
    return games or []


def build_team_total_map(games):
    """Build team_abbrev -> implied_total and team_abbrev -> game_total dicts.

    The Odds API returns full team names in 'matchup' (e.g. 'Boston Bruins @ ...').
    We parse these and map to standard NHL abbreviations.

    Returns:
        (team_totals, team_game_totals) tuple of dicts
    """
    # Build case-insensitive full-name lookup
    name_to_abbrev = {k.upper(): v for k, v in TEAM_FULL_NAME_TO_ABBREV.items()}

    team_totals = {}       # team_abbrev -> implied team total
    team_game_totals = {}  # team_abbrev -> game total

    for game in games:
        matchup = game.get('matchup', '')
        game_total = game.get('game_total')
        away_tt = game.get('away_team_total')
        home_tt = game.get('home_team_total')

        # Parse "Away Team @ Home Team"
        parts = matchup.split(' @ ')
        if len(parts) != 2:
            continue
        away_name = parts[0].strip().upper()
        home_name = parts[1].strip().upper()

        away_abbrev = name_to_abbrev.get(away_name)
        home_abbrev = name_to_abbrev.get(home_name)

        if away_abbrev and away_tt is not None:
            team_totals[away_abbrev] = away_tt
        if home_abbrev and home_tt is not None:
            team_totals[home_abbrev] = home_tt

        # Store game total for both teams
        if game_total is not None:
            if away_abbrev:
                team_game_totals[away_abbrev] = game_total
            if home_abbrev:
                team_game_totals[home_abbrev] = game_total

    return team_totals, team_game_totals


def print_lineup(lineup: pd.DataFrame):
    """Print optimized lineup."""
    print(f"\n{'=' * 80}")
    print(" OPTIMIZED LINEUP")
    print(f"{'=' * 80}")

    total_salary = lineup['salary'].sum()
    total_proj = lineup['projected_fpts'].sum()

    print(f"Total Salary: ${total_salary:,} / $50,000 (${50000 - total_salary:,} remaining)")
    print(f"Total Projected: {total_proj:.1f} pts")

    # Show stack info if available
    if 'stack_info' in lineup.columns and lineup['stack_info'].iloc[0]:
        print(f"Stacks: {lineup['stack_info'].iloc[0]}")

    print()

    print(f"{'Slot':<6} {'Name':<28} {'Team':<5} {'Salary':<9} {'Proj':<7} {'Value':<6}")
    print("-" * 70)

    # Sort by roster slot for display
    slot_order = {'G': 0, 'C1': 1, 'C2': 2, 'W1': 3, 'W2': 4, 'W3': 5, 'D1': 6, 'D2': 7, 'UTIL': 8}
    lineup = lineup.copy()
    if 'roster_slot' in lineup.columns:
        lineup['_slot_order'] = lineup['roster_slot'].map(lambda x: slot_order.get(x, 9))
        lineup = lineup.sort_values('_slot_order')
        slot_col = 'roster_slot'
    else:
        pos_order = {'C': 0, 'LW': 1, 'RW': 2, 'W': 2, 'L': 2, 'R': 2, 'D': 3, 'G': 4}
        lineup['_slot_order'] = lineup['position'].map(lambda x: pos_order.get(str(x).upper(), 5))
        lineup = lineup.sort_values('_slot_order')
        slot_col = 'position'

    for _, row in lineup.iterrows():
        print(f"{row[slot_col]:<6} {row['name']:<28} {row['team']:<5} "
              f"${row['salary']:<8,} {row['projected_fpts']:<7.1f} {row.get('value', 0):<6.2f}")


def export_projections(skaters: pd.DataFrame, goalies: pd.DataFrame, output_path: str):
    """Export projections to CSV."""
    # Combine and export
    skaters = skaters.copy()
    goalies = goalies.copy()
    skaters['player_type'] = 'skater'
    goalies['player_type'] = 'goalie'

    combined = pd.concat([skaters, goalies], ignore_index=True)

    export_cols = ['name', 'team', 'position', 'salary', 'projected_fpts',
                   'dk_avg_fpts', 'edge', 'value', 'floor', 'ceiling', 'player_type']
    export_cols = [c for c in export_cols if c in combined.columns]

    combined[export_cols].to_csv(output_path, index=False)
    print(f"\nProjections exported to: {output_path}")


def export_lineup_for_dk(lineup: pd.DataFrame, output_path: str):
    """
    Export lineup in DraftKings CSV upload format.

    DK format requires columns in order: C, C, W, W, W, D, D, G, UTIL
    with player IDs or names.
    """
    # DK column order and corresponding roster_slot names from optimizer
    dk_slots = ['C1', 'C2', 'W1', 'W2', 'W3', 'D1', 'D2', 'G', 'UTIL']
    slot_to_player = {}
    for _, player in lineup.iterrows():
        slot = player.get('roster_slot', '')
        if slot:
            slot_to_player[slot] = player

    # Build one row in exact DK order: C, C, W, W, W, D, D, G, UTIL
    row = []
    for slot in dk_slots:
        player = slot_to_player.get(slot)
        val = player.get('dk_id', player['name']) if player is not None else ''
        row.append(val)

    dk_columns = ['C', 'C', 'W', 'W', 'W', 'D', 'D', 'G', 'UTIL']
    dk_df = pd.DataFrame([row], columns=dk_columns)
    dk_df.to_csv(output_path, index=False)
    print(f"\nLineup exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='NHL DFS Projection Tool')
    parser.add_argument('--date', type=str, default=None,
                        help='Date to project for (YYYY-MM-DD)')
    parser.add_argument('--salaries', type=str, default=None,
                        help='Path to DraftKings salary CSV')
    parser.add_argument('--export', type=str, default=None,
                        help='Export projections to CSV')
    parser.add_argument('--lineups', type=int, default=1,
                        help='Number of lineups to generate')
    parser.add_argument('--stacks', action='store_true',
                        help='Show stacking recommendations from line combinations')
    parser.add_argument('--no-stacks', action='store_true',
                        help='Disable stacking boost in optimizer')
    parser.add_argument('--force-stack', type=str, default=None,
                        help='Force a specific stack type (PP1, Line1)')

    # Injury filtering options
    parser.add_argument('--show-injuries', action='store_true',
                        help='Display injury report for slate teams')
    parser.add_argument('--include-dtd', action='store_true',
                        help='Include Day-to-Day players in projections (normally excluded)')
    parser.add_argument('--no-injuries', action='store_true',
                        help='Disable injury filtering (include all players)')

    # Vegas lines
    parser.add_argument('--vegas', type=str, default=None,
                        help='Path to Vegas lines CSV file for game ranking')

    # Contest ROI: leverage recommendation + lineup ranking by expected payout
    parser.add_argument('--contest-entry-fee', type=float, default=None,
                        help='Contest entry fee ($) for leverage/EV ranking')
    parser.add_argument('--contest-max-entries', type=int, default=None,
                        help='Max entries per user for this contest')
    parser.add_argument('--contest-field-size', type=int, default=None,
                        help='Total contest entries (field size)')
    parser.add_argument('--contest-payout', type=str, default=None,
                        choices=list(PAYOUT_PRESETS.keys()),
                        help='Payout preset: top_heavy_gpp, flat, high_dollar_single, small_se_gpp')
    parser.add_argument('--contest-prize-pool', type=float, default=None,
                        help='Actual prize pool ($) — overrides entry_fee * field_size (accounts for rake)')
    parser.add_argument('--contest-min-cash-entries', type=int, default=None,
                        help='Total number of paid entries (min-cash and above)')

    # Simulator
    parser.add_argument('--simulate', action='store_true',
                        help='Run optimal lineup simulator (team-pair frequency analysis)')
    parser.add_argument('--sim-iterations', type=int, default=0,
                        help='Monte Carlo iterations for simulator (0=deterministic, recommended=100)')
    parser.add_argument('--sim-lift', type=float, nargs='?', const=0.15, default=None,
                        help='Run two-pass lift-adjusted simulation (blend factor, default 0.15)')

    # Single-entry mode
    parser.add_argument('--single-entry', action='store_true',
                        help='Single-entry mode: generate N candidates then select best via '
                             'SE scoring (goalie quality, stack correlation, salary efficiency). '
                             'Use with --lineups 40-60 for best results.')
    parser.add_argument('--contest', type=str, default='se_gpp',
                        choices=['satellite', 'se_gpp', 'custom', 'prompt'],
                        help='Contest type for SE mode: satellite ($14 WTA), '
                             'se_gpp ($121 SE, default), custom, or prompt (interactive)')
    parser.add_argument('--contest-entries', type=int, default=80,
                        help='Expected entries for SE GPP (default 80)')

    # Bayesian blend
    parser.add_argument('--blend', action='store_true',
                        help='Blend projections with Bayesian event model (45/55 split, ~25%% MAE improvement)')

    # Advanced stats options
    parser.add_argument('--no-advanced', action='store_true',
                        help='Skip fetching Natural Stat Trick advanced stats')

    # Recent game scoring
    parser.add_argument('--no-recent-scores', action='store_true',
                        help='Skip fetching individual recent game scores (faster)')
    parser.add_argument('--refresh-recent-scores', action='store_true',
                        help='Force refresh recent scores from API (ignore cache)')

    # NHL Edge stats
    parser.add_argument('--edge', action='store_true',
                        help='Apply NHL Edge tracking boosts (speed, OZ time, bursts)')
    parser.add_argument('--no-edge', action='store_true',
                        help='Skip NHL Edge stats (faster)')
    parser.add_argument('--refresh-edge', action='store_true',
                        help='Force refresh Edge stats from API (ignore cache). '
                             'Use for first run of day or to get latest data.')

    # Linemate correlation boosts
    parser.add_argument('--validate', action='store_true',
                        help='Run pre-flight validation checks before generating lineups')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run pre-flight validation and exit (no projections)')

    parser.add_argument('--linemates', action='store_true',
                        help='Apply linemate chemistry boosts from play-by-play correlation')
    parser.add_argument('--linemate-games', type=int, default=10,
                        help='Number of recent games to analyze per team (default: 10)')
    parser.add_argument('--linemate-report', action='store_true',
                        help='Print full linemate chemistry report for slate teams')

    args = parser.parse_args()

    # Determine date
    target_date = args.date or datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'#' * 80}")
    print(f"  NHL DFS PROJECTIONS - {target_date}")
    print(f"{'#' * 80}")

    # Find salary file
    project_dir = Path(__file__).parent
    if args.salaries:
        salary_path = args.salaries
    else:
        # Look in daily_salaries/ first, then project root
        salaries_dir = project_dir / DAILY_SALARIES_DIR
        salary_files = list(salaries_dir.glob('DKSalaries*.csv')) if salaries_dir.exists() else []
        if not salary_files:
            salary_files = list(project_dir.glob('DKSalaries*.csv'))
        if salary_files:
            salary_path = str(sorted(salary_files)[-1])  # Most recent
            print(f"\nAuto-detected salary file: {salary_path}")
        else:
            print("\nNo DraftKings salary file found. Run with --salaries <path> or add DKSalaries*.csv to daily_salaries/")
            sys.exit(1)

    # Load DK salaries
    dk_salaries = load_dk_salaries(salary_path)

    # Get slate teams for filtering
    slate_teams = list(dk_salaries['team'].unique())

    # Show Vegas game ranking — prefer Odds API, fall back to CSV
    csv_fallback = args.vegas
    if not csv_fallback:
        vegas_dir = project_dir / VEGAS_DIR
        if vegas_dir.exists():
            vegas_files = sorted(vegas_dir.glob('Vegas*.csv')) + sorted(vegas_dir.glob('VegasNHL*.csv'))
            vegas_files = list(dict.fromkeys(vegas_files))  # dedupe
            if vegas_files:
                csv_fallback = str(vegas_files[-1])
    vegas_games = print_vegas_ranking(csv_fallback)

    # Build Vegas team total map for ownership model (Feature 1)
    team_totals, team_game_totals = build_team_total_map(vegas_games)
    if team_totals:
        print(f"  Vegas team totals mapped for {len(team_totals)} teams")

    # --- Pre-flight: validate-only quick check (exit before data fetch) ---
    if args.validate_only:
        report = run_validation(
            salary_path=salary_path,
            target_date=target_date,
            vegas_games=vegas_games,
            slate_teams=slate_teams,
            quick=True
        )
        report.print_report()
        sys.exit(0 if report.is_go else 1)

    # Fetch NHL data
    print("\nFetching NHL data...")
    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(
        include_game_logs=False,
        include_injuries=not args.no_injuries,
        include_advanced_stats=not args.no_advanced,
        include_edge_stats=args.edge and not args.no_edge,
        force_refresh_edge=args.refresh_edge
    )

    # Show injury report if requested
    if args.show_injuries and 'injuries' in data and not data['injuries'].empty:
        print_injury_report(data['injuries'], slate_teams)

    # Generate projections
    print("\nGenerating projections...")
    # Pass Vegas data through for goalie danger-zone model
    data['team_totals'] = team_totals
    data['team_game_totals'] = team_game_totals
    model = NHLProjectionModel()
    projections = model.generate_projections(
        data,
        target_date=target_date,
        filter_injuries=not args.no_injuries,
        include_dtd=not args.include_dtd  # If --include-dtd, don't filter DTD
    )

    # Separate goalies from skaters in DK salaries
    dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'W', 'D'])]
    dk_goalies = dk_salaries[dk_salaries['position'] == 'G']

    # Add position column to goalies if missing
    if 'position' not in projections['goalies'].columns:
        projections['goalies']['position'] = 'G'

    # Merge projections with salaries
    print("\nMerging projections with DK salaries...")
    skaters_merged = merge_projections_with_salaries(
        projections['skaters'], dk_skaters, 'skater'
    )
    goalies_merged = merge_projections_with_salaries(
        projections['goalies'], dk_goalies, 'goalie'
    )

    print(f"  Matched {len(skaters_merged)} skaters")
    print(f"  Matched {len(goalies_merged)} goalies")

    # Apply Goalie Edge boosts if enabled
    if args.edge and not args.no_edge:
        try:
            from edge_stats import apply_goalie_edge_boosts
            print("\nApplying goalie Edge stats...")
            goalies_merged = apply_goalie_edge_boosts(
                goalies_merged,
                force_refresh=args.refresh_edge
            )
        except ImportError:
            print("  Warning: apply_goalie_edge_boosts not available")
        except Exception as e:
            print(f"  Warning: Goalie Edge boosts failed: {e}")

    # Apply linemate chemistry boosts if enabled
    if args.linemates and len(skaters_merged) > 0:
        try:
            from linemate_corr import get_linemate_boosts, print_team_report
            print("\nApplying linemate chemistry boosts...")
            n_lg = args.linemate_games

            # Build player name + team lists from merged skaters
            lm_names = skaters_merged['name'].tolist()
            lm_teams = skaters_merged['team'].tolist()

            boosts = get_linemate_boosts(lm_names, lm_teams, n_games=n_lg)

            # Apply boosts to projected_fpts
            boosted_count = 0
            for idx, row in skaters_merged.iterrows():
                mult = boosts.get(row['name'], 1.0)
                if mult > 1.0:
                    skaters_merged.at[idx, 'projected_fpts'] *= mult
                    boosted_count += 1

            # Recalculate value after boost
            if 'salary' in skaters_merged.columns:
                skaters_merged['value'] = skaters_merged['projected_fpts'] / (skaters_merged['salary'] / 1000)
            if 'dk_avg_fpts' in skaters_merged.columns:
                skaters_merged['edge'] = skaters_merged['projected_fpts'] - skaters_merged['dk_avg_fpts']

            print(f"  Linemate boosts applied to {boosted_count} skaters (from {n_lg}-game window)")

            # Print full report if requested
            if args.linemate_report:
                for team in sorted(set(lm_teams)):
                    print_team_report(team, n_lg)

        except ImportError:
            print("  Warning: linemate_corr module not found — skipping linemate boosts")
            print("  Place linemate_corr.py in projection/ directory")
        except Exception as e:
            print(f"  Warning: Linemate chemistry boosts failed: {e}")

    # Apply line context adjustments (PP unit × game environment)
    if len(skaters_merged) > 0:
        try:
            from line_model import apply_line_adjustments
            # Build opp_totals from team_totals
            opp_totals = {}
            if team_totals:
                # For each team, get their opponent's implied total
                for _, row in skaters_merged.drop_duplicates('team').iterrows():
                    opp = row.get('opponent', row.get('opp', ''))
                    if opp and isinstance(opp, str):
                        opp_clean = opp.replace('@', '').replace('vs', '').strip()
                        opp_totals[row['team']] = team_totals.get(opp_clean, 3.0)

            skaters_merged = apply_line_adjustments(
                skaters_merged,
                team_totals=team_totals if team_totals else {},
                opp_totals=opp_totals,
                verbose=True,
            )
        except ImportError:
            print("  Line model not available — skipping line context adjustments")
        except Exception as e:
            print(f"  Warning: Line context adjustments failed: {e}")

    # Combine pools
    player_pool = pd.concat([skaters_merged, goalies_merged], ignore_index=True)

    # Fetch line combinations if stacking enabled
    stack_builder = None

    if not args.no_stacks or args.stacks:
        print("\nFetching line combinations from DailyFaceoff...")
        scraper = LinesScraper(rate_limit=0.5)
        lines_data = scraper.get_multiple_teams(slate_teams)
        stack_builder = StackBuilder(lines_data)

        # Persist lines for dashboard (daily_projections/lines_{date}.json)
        out_dir = project_dir / DAILY_PROJECTIONS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        lines_path = out_dir / f"lines_{target_date.replace('-', '_')}.json"
        try:
            import json
            with open(lines_path, "w", encoding="utf-8") as f:
                json.dump(lines_data, f, indent=0)
        except Exception as e:
            print(f"  Note: could not write lines to {lines_path}: {e}")

        # Show line update timestamps
        print_line_update_status(stack_builder)

        # Show confirmed goalies
        print_confirmed_goalies(stack_builder, player_pool)

        # Filter player pool and goalie projections to confirmed starters only
        confirmed = stack_builder.get_all_starting_goalies()
        if confirmed:
            from lines import fuzzy_match as _fm
            confirmed_names = list(confirmed.values())
            def _is_confirmed(name):
                return any(_fm(name, cn) for cn in confirmed_names)
            before = len(goalies_merged)
            goalies_merged = goalies_merged[goalies_merged['name'].apply(_is_confirmed)]
            player_pool = pd.concat([skaters_merged, goalies_merged], ignore_index=True)
            filtered_count = before - len(goalies_merged)
            if filtered_count > 0:
                print(f"  Filtered {filtered_count} non-confirmed goalies from pool")

        # Show stacking recommendations if requested
        if args.stacks:
            print_stacking_recommendations(stack_builder, player_pool, slate_teams)

    # --- Pre-flight: full validation (after all data loaded) ---
    if args.validate or args.validate_only:
        report = run_validation(
            salary_path=salary_path,
            target_date=target_date,
            vegas_games=vegas_games,
            data=data,
            skaters_merged=skaters_merged,
            goalies_merged=goalies_merged,
            player_pool=player_pool,
            stack_builder=stack_builder,
            slate_teams=slate_teams,
            quick=False
        )
        report.print_report()
        if not report.is_go:
            print("  ⚠️  Fix failures above before trusting these projections.\n")

    # Print projections
    if len(skaters_merged) > 0:
        print_projections_table(skaters_merged, "TOP SKATER PROJECTIONS", n=25)
        print_value_plays(skaters_merged, "VALUE SKATER PLAYS", n=15)

    if len(goalies_merged) > 0:
        print_projections_table(goalies_merged, "TOP GOALIE PROJECTIONS", n=10)

    # Contest profile for leverage + EV ranking (optional)
    contest_profile = None
    if any(x is not None for x in [args.contest_entry_fee, args.contest_max_entries, args.contest_field_size, args.contest_payout]):
        contest_profile = ContestProfile(
            entry_fee=args.contest_entry_fee if args.contest_entry_fee is not None else 5.0,
            max_entries=args.contest_max_entries if args.contest_max_entries is not None else 1,
            field_size=args.contest_field_size if args.contest_field_size is not None else 10000,
            payout_preset=args.contest_payout or "top_heavy_gpp",
            prize_pool_override=args.contest_prize_pool,
            min_cash_entries=args.contest_min_cash_entries,
        )
        rec = recommend_leverage(contest_profile)
        print_leverage_recommendation(contest_profile, rec)

    # --- Bayesian Projection Blend (--blend flag) ---
    if args.blend:
        try:
            from projection_blender import blend_projections

            # ── Auto-capture today's live odds ──
            try:
                from historical_odds import capture_daily, get_odds_for_date
                capture_daily()
            except Exception as e:
                print(f"  ⚠ Odds capture: {e}")

            # ── Load Vegas data: DB first, CSV fallback ──
            vegas_blend = None
            try:
                from historical_odds import get_odds_for_date
                db_odds = get_odds_for_date(target_date)
                if not db_odds.empty:
                    db_odds['date'] = target_date
                    vegas_blend = db_odds
                    print(f"  Vegas: {len(db_odds)} team-games from DB")
            except Exception:
                pass

            if vegas_blend is None:
                vegas_paths = [
                    project_dir / 'Vegas_Historical.csv',
                    project_dir / 'vegas' / 'Vegas_Historical.csv',
                ]
                for vp in vegas_paths:
                    if vp.exists():
                        import pandas as _pd
                        vdf = _pd.read_csv(vp, encoding='utf-8-sig')
                        vdf['date'] = vdf['Date'].apply(
                            lambda d: f"20{d.split('.')[2]}-{int(d.split('.')[0]):02d}-{int(d.split('.')[1]):02d}"
                        )
                        vegas_blend = vdf
                        print(f"  Vegas: loaded from {vp.name} (CSV fallback)")
                        break

            player_pool = blend_projections(
                player_pool, vegas=vegas_blend, date_str=target_date,
                replace=True, verbose=True
            )
        except Exception as e:
            print(f"  ⚠ Blend failed: {e} — using current projections")

    # --- Ceiling Probability (from game log clustering) ---
    try:
        from ceiling_clustering import predict_ceiling_probability
        player_pool = predict_ceiling_probability(player_pool)
        if 'p_ceiling' in player_pool.columns:
            n_high = (player_pool['p_ceiling'] > 0.15).sum()
            avg_p = player_pool['p_ceiling'].mean()
            print(f"\n  Ceiling probability: {n_high} high-ceiling players (avg P={avg_p:.1%})")

            # Adjust ceiling column using p_ceiling
            # Scale: p_ceiling relative to base rate → ceiling multiplier
            # p=0.065 (base) → 1.0x, p=0.20 → 1.25x, p=0.40 → 1.4x, p=0.01 → 0.85x
            if 'ceiling' in player_pool.columns and 'projected_fpts' in player_pool.columns:
                base_rate = 0.065  # ~6.5% baseline ceiling rate
                p = player_pool['p_ceiling'].clip(0.01, 0.50)
                # Log scale: smoother, prevents extreme multipliers
                ceiling_scale = (1.0 + 0.3 * np.log(p / base_rate)).clip(0.85, 1.4)
                base_ceiling = player_pool['projected_fpts'] * 2.5 + 5
                goalie_mask = player_pool['position'] == 'G'
                base_ceiling[goalie_mask] = player_pool.loc[goalie_mask, 'projected_fpts'] * 2.0 + 10
                player_pool['ceiling'] = (base_ceiling * ceiling_scale).round(1)
    except Exception as e:
        player_pool['p_ceiling'] = 0.05
        print(f"  ⚠ Ceiling model: {e}")

    # --- Game Environment Model (Vegas implied + pace + recency) ---
    try:
        from game_environment import GameEnvironmentModel
        env_model = GameEnvironmentModel()
        env_model.fit()
        if env_model.fitted:
            _vb = vegas_blend if 'vegas_blend' in dir() else None
            player_pool = env_model.adjust_projections(
                player_pool, vegas=_vb, date_str=target_date, verbose=True
            )
    except Exception as e:
        print(f"  ⚠ Game environment: {e}")

    # --- Linemate Correlation Boosts ---
    try:
        from linemate_corr import get_linemate_boosts
        skater_mask = player_pool['position'] != 'G'
        if skater_mask.sum() > 0:
            lm_names = player_pool.loc[skater_mask, 'name'].tolist()
            lm_teams = player_pool.loc[skater_mask, 'team'].tolist()
            boosts = get_linemate_boosts(lm_names, lm_teams)
            lm_count = 0
            for idx, row in player_pool[skater_mask].iterrows():
                mult = boosts.get(row['name'], 1.0)
                if mult > 1.0:
                    player_pool.loc[idx, 'projected_fpts'] *= mult
                    lm_count += 1
            if lm_count:
                print(f"\n  Linemate boosts: {lm_count} skaters adjusted")
    except Exception as e:
        print(f"  ⚠ Linemate corr: {e}")

    # --- Fetch recent game scores for ownership model (Feature 5) ---
    recent_scores = {}
    if not args.no_recent_scores and 'player_id' in player_pool.columns:
        player_ids = player_pool['player_id'].dropna().unique().tolist()
        # Convert to int (player_id may be float after merge)
        player_ids = [int(pid) for pid in player_ids if pd.notna(pid)]
        if player_ids:
            print("\nFetching recent game scores for ownership model...")
            try:
                from recent_scores_cache import get_cached_recent_scores
                recent_scores = get_cached_recent_scores(
                    player_ids,
                    pipeline,
                    force_refresh=args.refresh_recent_scores
                )
            except ImportError:
                # Fallback to direct fetch if caching module not available
                recent_scores = pipeline.fetch_recent_game_scores(player_ids)

    # --- Run ownership model on player pool (before lineups when contest EV is used) ---
    print("\nGenerating ownership projections...")

    if args.single_entry:
        # Use SE-specific ownership model (trained on actual small-field SE data)
        try:
            from se_ownership import SEOwnershipModel
            se_model = SEOwnershipModel()
            se_model.fit_from_contests()

            # Pass confirmed goalies if available
            if stack_builder:
                confirmed = stack_builder.get_all_starting_goalies()
                if confirmed:
                    se_model.set_confirmed_goalies(confirmed)

            player_pool = se_model.predict(player_pool, verbose=True)
            print("  Using SE ownership model (trained on small-field contest data)")
        except Exception as e:
            print(f"  ⚠ SE ownership model failed: {e} — falling back to GPP model")
            args._se_ownership_failed = True

    if not args.single_entry or getattr(args, '_se_ownership_failed', False):
        # Standard GPP ownership model
        ownership_model = OwnershipModel()
        if stack_builder:
            confirmed = stack_builder.get_all_starting_goalies()
            ownership_model.set_lines_data(stack_builder.lines_data, confirmed)

        # Feature 1: Vegas implied team totals
        if team_totals:
            ownership_model.set_vegas_data(team_totals, team_game_totals)

        # Feature 4: Return-from-injury buzz
        if 'injuries' in data and not data['injuries'].empty:
            ownership_model.set_injury_data(data['injuries'], target_date)

        # Feature 5: Individual recent game scoring
        if recent_scores:
            ownership_model.set_recent_scores(recent_scores)

        # Feature 6: TOI surge map (player name -> delta in minutes)
        toi_surge_map = {}
        if recent_scores and 'toi_per_game' in player_pool.columns:
            for _, row in player_pool.iterrows():
                pid = row.get('player_id')
                if pid and pid in recent_scores:
                    recent_toi = recent_scores[pid].get('last_3_avg_toi_min')
                    season_toi = row.get('toi_per_game')
                    if recent_toi and season_toi and season_toi > 0:
                        # season toi may be in seconds (>100) or minutes
                        season_min = season_toi / 60.0 if season_toi > 100 else season_toi
                        toi_surge_map[row['name']] = recent_toi - season_min

        if toi_surge_map:
            ownership_model.set_toi_surge_data(toi_surge_map)
            print(f"  TOI surge data set for {len(toi_surge_map)} players")

        player_pool = ownership_model.predict_ownership(player_pool)
        print_ownership_report(player_pool)

    # --- Run simulator if requested ---
    if args.simulate:
        from simulator import OptimalLineupSimulator
        sim_iterations = args.sim_iterations

        std_dev_data = None
        if sim_iterations > 0 and 'player_id' in player_pool.columns:
            # Build player_type map from pool
            player_type_map = {}
            for _, row in player_pool.iterrows():
                pid = row.get('player_id')
                if pid and pd.notna(pid):
                    ptype = row.get('player_type', 'skater')
                    player_type_map[int(pid)] = ptype

            player_ids = [int(pid) for pid in player_pool['player_id'].dropna().unique()]
            print(f"\nFetching game logs for Monte Carlo std dev ({len(player_ids)} players)...")
            std_dev_data = pipeline.compute_player_fpts_std_dev(
                player_ids, player_type_map
            )

        # First-pass simulation
        simulator = OptimalLineupSimulator(
            player_pool, dk_salaries,
            std_dev_data=std_dev_data,
            n_iterations=sim_iterations,
        )
        sim_results = simulator.run()
        simulator.print_results(sim_results)

        # Auto-export first-pass
        sim_out_dir = project_dir / DAILY_PROJECTIONS_DIR
        sim_out_dir.mkdir(parents=True, exist_ok=True)
        sim_date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%m_%d_%y')
        sim_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_tag = f"mc{sim_iterations}" if sim_iterations > 0 else "det"
        sim_path = str(sim_out_dir / f"{sim_date_str}NHLsimulator_{mode_tag}_{sim_timestamp}.csv")
        simulator.export_results(sim_results, sim_path)

        # Second-pass: lift-adjusted re-simulation
        if args.sim_lift is not None and not sim_results.empty:
            blend = args.sim_lift
            print(f"\n{'=' * 110}")
            print(f" LIFT-ADJUSTED RE-SIMULATION (blend={blend:.2f})")
            print(f"{'=' * 110}")

            lifted_pool = OptimalLineupSimulator.apply_lift_adjustments(
                player_pool, sim_results, blend=blend
            )

            lift_simulator = OptimalLineupSimulator(
                lifted_pool, dk_salaries,
                std_dev_data=std_dev_data,
                n_iterations=sim_iterations,
            )
            lift_results = lift_simulator.run()
            lift_simulator.print_results(lift_results)

            # Auto-export lift-adjusted
            lift_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            lift_path = str(sim_out_dir / f"{sim_date_str}NHLsimulator_{mode_tag}_lift{blend}_{lift_timestamp}.csv")
            lift_simulator.export_results(lift_results, lift_path)

    # Generate optimized lineup
    lineups = []
    if len(skaters_merged) > 0 and len(goalies_merged) > 0:

        # ── Single-Entry Mode ──────────────────────────────────────────────
        # Generate many candidates with randomness, then pick the best one
        # using Tournament Equity (v4) — scores lineups in DOLLARS not abstract 0-1
        if args.single_entry:
            n_candidates = max(args.lineups, 100)  # At least 100 candidates (mixed randomness)

            # Contest profile for entry fee
            try:
                if args.contest == 'satellite':
                    entry_fee = 14.0
                    se_contest = SEContestProfile.satellite()
                elif args.contest == 'se_gpp':
                    entry_fee = 121.0
                    se_contest = SEContestProfile.se_gpp(getattr(args, 'contest_entries', 80))
                elif args.contest in ('custom', 'prompt'):
                    se_contest = prompt_contest_profile()
                    entry_fee = se_contest.entry_fee
                else:
                    entry_fee = 121.0
                    se_contest = SEContestProfile.se_gpp()
            except Exception:
                entry_fee = 121.0
                se_contest = SEContestProfile.se_gpp()

            print(f"\nSingle-Entry Mode: Tournament Equity v4 | {se_contest.name}")
            print(f"  Entry fee: ${entry_fee:.0f} | Generating {n_candidates} candidates (mixed randomness)...")

            optimizer = NHLLineupOptimizer(stack_builder=stack_builder if not args.no_stacks else None)

            # Mixed randomness pool: diverse candidates give M+3σ selector
            # more to differentiate. Backtest: 100 mixed = 94.0 avg vs 83.9 for 60 uniform.
            candidates = []
            n_per_tier = max(n_candidates // 5, 10)
            for rand_level in [0.05, 0.10, 0.15, 0.20, 0.25]:
                batch = optimizer.optimize_lineup(
                    player_pool,
                    n_lineups=n_per_tier,
                    randomness=rand_level,
                    stack_teams=[args.force_stack] if args.force_stack else None,
                )
                if batch:
                    candidates.extend(batch)

            print(f"  Generated {len(candidates)} candidates across 5 randomness tiers")

            if candidates:
                # Primary: Tournament Equity selector (v4)
                te_selector = TournamentEquitySelector(
                    player_pool,
                    entry_fee=entry_fee,
                    stack_builder=stack_builder,
                )
                best_lineup, te_result = te_selector.select(candidates, verbose=True)
                print_te_lineup(best_lineup, te_result)
                lineups = [best_lineup]

                # Export all ranked candidates to JSON for website
                try:
                    from lineup_export import export_se_candidates
                    all_scored = []
                    for lu in candidates:
                        te = te_selector.compute_te_analytical(lu)
                        # Wrap TE result with 'total' key for compatibility
                        score_dict = {**te, 'total': te['te']}
                        all_scored.append((lu, score_dict))
                    all_scored.sort(key=lambda x: x[1]['total'], reverse=True)
                    export_se_candidates(all_scored, target_date, str(out_dir))
                except Exception as e:
                    print(f"  Note: Could not export SE lineup details: {e}")
            else:
                print("  Warning: No candidate lineups generated")

        # ── Standard Mode (multi-entry or default) ────────────────────────
        else:
            print("\nOptimizing lineup...")

            # Pass stack builder to optimizer if available
            optimizer = NHLLineupOptimizer(stack_builder=stack_builder if not args.no_stacks else None)

            lineups = optimizer.optimize_lineup(
                player_pool,
                n_lineups=args.lineups,
                randomness=0.05 if args.lineups > 1 else 0,
                stack_teams=[args.force_stack] if args.force_stack else None
            )

            # Contest EV: score and re-rank lineups by expected payout
            if contest_profile and lineups:
                scored = contest_score_lineups(lineups, contest_profile, player_pool)
                lineups = [lu for lu, _ in scored]
                # Attach EV to each lineup for display (store as list of (lineup, ev))
                lineup_ev_pairs = [(lu, ev) for lu, ev in scored]
            else:
                lineup_ev_pairs = [(lu, None) for lu in lineups]

            for i, (lineup, contest_ev) in enumerate(lineup_ev_pairs):
                if args.lineups > 1:
                    ev_str = f" (Contest EV: ${contest_ev:.2f})" if contest_ev is not None else ""
                    print(f"\n--- Lineup {i+1}{ev_str} ---")
                else:
                    if contest_ev is not None:
                        print(f"\nContest EV: ${contest_ev:.2f}")
                print_lineup(lineup)

    # --- Auto-export projections + ownership with timestamp ---
    date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%m_%d_%y')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = project_dir / DAILY_PROJECTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    auto_export_path = str(out_dir / f"{date_str}NHLprojections_{timestamp}.csv")

    export_cols = ['name', 'team', 'position', 'salary', 'projected_fpts',
                   'dk_avg_fpts', 'edge', 'value', 'floor', 'ceiling', 'p_ceiling',
                   'player_type', 'predicted_ownership', 'ownership_tier', 'leverage_score', 'dk_id']
    export_cols = [c for c in export_cols if c in player_pool.columns]
    player_pool.sort_values('projected_fpts', ascending=False)[export_cols].to_csv(auto_export_path, index=False)
    print(f"\nProjections + ownership exported to: {auto_export_path}")

    # Also export lineups CSV if generated
    if lineups:
        lineup_path = auto_export_path.replace('.csv', '_lineups.csv')
        export_lineup_for_dk(lineups[0], lineup_path)

    # Legacy --export flag still works
    if args.export:
        export_path = args.export
        if Path(export_path).name == export_path or '/' not in export_path and '\\' not in export_path:
            export_path = str(out_dir / export_path)
        export_projections(skaters_merged, goalies_merged, export_path)

    print(f"\n{'#' * 80}")
    print("  PROJECTION COMPLETE")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
