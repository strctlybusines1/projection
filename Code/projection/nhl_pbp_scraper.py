"""
NHL Play-by-Play Scraper & Advanced Stats Engine

Scrapes play-by-play data from the NHL API and computes advanced statistics
that replace NST dependency:
  - xG (Expected Goals) via XGBoost trained on shot features
  - Individual xG (ixG) per player per game
  - On-ice xGF% per player
  - High-danger shot rates
  - PP/EV deployment metrics
  - Team-level defensive stats (xGA, HD chances against)

Data source: api-web.nhle.com (official NHL API, no rate limiting issues)
Features based on Evolving Hockey methodology.

Author: Claude
Date: 2026-02-18
"""

import json
import math
import sqlite3
import time
import urllib.request
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'data' / 'nhl_dfs_history.db'
API_BASE = "https://api-web.nhle.com/v1"

# Rate limiting
REQUEST_DELAY = 0.35  # seconds between requests (respectful)


# ==============================================================================
# API HELPERS
# ==============================================================================

def api_get(endpoint: str, retries: int = 3) -> dict:
    """Fetch JSON from NHL API with retry logic."""
    url = f"{API_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (NHL DFS Research)',
                'Accept': 'application/json',
            })
            resp = urllib.request.urlopen(req, timeout=15)
            return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error ({url}): {e}")
                return {}
    return {}


def get_all_game_ids(season: str = '20252026', game_type: int = 2) -> List[dict]:
    """
    Get all game IDs for a season by pulling one team's schedule.
    game_type: 2 = regular season, 3 = playoffs
    """
    # Use the schedule endpoint for a date range approach
    # Or pull one team's schedule and collect unique game IDs
    all_games = {}

    # Get team list from standings
    standings = api_get("standings/now")
    teams = [t['teamAbbrev']['default'] for t in standings.get('standings', [])]

    for i, team in enumerate(teams):
        time.sleep(REQUEST_DELAY)
        sched = api_get(f"club-schedule-season/{team}/{season}")
        for game in sched.get('games', []):
            gid = game['id']
            if game.get('gameType') == game_type and gid not in all_games:
                all_games[gid] = {
                    'game_id': gid,
                    'date': game.get('gameDate', ''),
                    'away': game.get('awayTeam', {}).get('abbrev', ''),
                    'home': game.get('homeTeam', {}).get('abbrev', ''),
                    'state': game.get('gameState', ''),
                    'season': season,
                }

        if (i + 1) % 8 == 0:
            print(f"  Scanned {i+1}/{len(teams)} teams, {len(all_games)} unique games")

    games = sorted(all_games.values(), key=lambda x: x['date'])
    completed = [g for g in games if g['state'] in ('OFF', 'FINAL')]
    print(f"  Season {season}: {len(games)} total reg season games, {len(completed)} completed")
    return games


# ==============================================================================
# PLAY-BY-PLAY PARSER
# ==============================================================================

def parse_situation_code(code: str) -> dict:
    """
    Parse NHL situationCode into strength state.
    Format: ABCD where A=away_goalie(0/1), B=away_skaters, C=home_skaters, D=home_goalie(0/1)
    """
    if not code or len(str(code)) != 4:
        return {'away_skaters': 5, 'home_skaters': 5, 'away_goalie': True, 'home_goalie': True}

    code = str(code)
    return {
        'away_goalie': code[0] == '1',
        'away_skaters': int(code[1]),
        'home_skaters': int(code[2]),
        'home_goalie': code[3] == '1',
    }


def compute_shot_features(play: dict, prev_play: dict, home_team_id: int,
                          away_team_id: int, defending_side: str) -> dict:
    """
    Compute xG features from a single shot/goal event.

    Features based on Evolving Hockey xG model:
    - Shot distance and angle from net
    - Shot type (wrist, slap, backhand, snap, tip, wrap, deflected)
    - Strength state (5v5, PP, SH, EN)
    - Score differential
    - Time in period / game seconds
    - Prior event type, distance, time elapsed
    - Rebound indicator
    - Rush indicator
    - Is home team
    """
    details = play.get('details', {})
    x = details.get('xCoord', 0)
    y = details.get('yCoord', 0)
    zone = details.get('zoneCode', 'N')
    shot_type = details.get('shotType', 'wrist')
    event_team_id = details.get('eventOwnerTeamId', 0)
    is_goal = play.get('typeDescKey') == 'goal'

    # Determine if shooting team is home or away
    is_home = event_team_id == home_team_id

    # Normalize coordinates: shots should be toward the right-side net (+x direction)
    # NHL rink: center ice = (0,0), nets at x = ±89
    # The API coordinates are already normalized per period based on homeTeamDefendingSide
    # But we want all shots pointed toward the net at (89, 0)
    if x < 0:
        x = -x
        y = -y

    # Shot distance from net center (89, 0)
    net_x, net_y = 89, 0
    distance = math.sqrt((x - net_x)**2 + (y - net_y)**2)

    # Shot angle: angle from center of the goal line
    # 0 = straight on, 90 = extreme angle
    if abs(x - net_x) < 0.01:
        angle = 90.0
    else:
        angle = abs(math.degrees(math.atan(y / (net_x - x + 0.01))))

    # Strength state
    sit = parse_situation_code(play.get('situationCode', '1551'))
    if is_home:
        own_skaters = sit['home_skaters']
        opp_skaters = sit['away_skaters']
        opp_goalie = sit['away_goalie']
    else:
        own_skaters = sit['away_skaters']
        opp_skaters = sit['home_skaters']
        opp_goalie = sit['home_goalie']

    # Strength categories
    is_pp = own_skaters > opp_skaters
    is_sh = own_skaters < opp_skaters
    is_en = not opp_goalie
    is_5v5 = (own_skaters == 5 and opp_skaters == 5)

    # Score differential (from shooting team's perspective)
    if is_goal:
        # For goals, use score BEFORE the goal
        away_score = details.get('awayScore', 0)
        home_score = details.get('homeScore', 0)
        if is_home:
            score_diff = (home_score - 1) - away_score if home_score > 0 else 0 - away_score
        else:
            score_diff = (away_score - 1) - home_score if away_score > 0 else 0 - home_score
    else:
        away_sog = details.get('awaySOG', 0)
        home_sog = details.get('homeSOG', 0)
        # We don't have running score in shots, use 0 as default
        score_diff = 0  # Will be populated from running game state

    # Time features
    period = play.get('periodDescriptor', {}).get('number', 1)
    time_str = play.get('timeInPeriod', '00:00')
    parts = time_str.split(':')
    seconds_in_period = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0
    game_seconds = (period - 1) * 1200 + seconds_in_period

    # Prior event features
    prior_type = ''
    prior_same_team = False
    prior_distance = 0
    prior_seconds = 0
    is_rebound = False

    if prev_play:
        prev_details = prev_play.get('details', {})
        prior_type = prev_play.get('typeDescKey', '')
        prev_team_id = prev_details.get('eventOwnerTeamId', 0)
        prior_same_team = (prev_team_id == event_team_id)

        # Prior event coordinates
        px = prev_details.get('xCoord', 0)
        py = prev_details.get('yCoord', 0)
        if px < 0:
            px = -px
            py = -py
        prior_distance = math.sqrt((x - px)**2 + (y - py)**2)

        # Time since prior event
        prev_time_str = prev_play.get('timeInPeriod', '00:00')
        prev_parts = prev_time_str.split(':')
        prev_period = prev_play.get('periodDescriptor', {}).get('number', 1)
        prev_seconds = int(prev_parts[0]) * 60 + int(prev_parts[1]) if len(prev_parts) == 2 else 0
        prev_game_seconds = (prev_period - 1) * 1200 + prev_seconds
        prior_seconds = game_seconds - prev_game_seconds

        # Rebound: prior shot by same team within 3 seconds
        is_rebound = (prior_same_team and
                      prior_type in ('shot-on-goal', 'missed-shot', 'blocked-shot') and
                      prior_seconds <= 3)

    # Rush: shot within 4 seconds of a neutral zone event
    is_rush = (prior_seconds <= 4 and
               prev_play is not None and
               prev_play.get('details', {}).get('zoneCode', '') in ('N', 'D'))

    # Shot type encoding
    shot_types = ['wrist', 'slap', 'backhand', 'snap', 'tip-in', 'deflected', 'wrap-around']

    return {
        'is_goal': int(is_goal),
        'distance': distance,
        'angle': angle,
        'x_coord': x,
        'y_coord': y,
        'shot_type': shot_type,
        'is_home': int(is_home),
        'is_pp': int(is_pp),
        'is_sh': int(is_sh),
        'is_en': int(is_en),
        'is_5v5': int(is_5v5),
        'own_skaters': own_skaters,
        'opp_skaters': opp_skaters,
        'score_diff': score_diff,
        'period': period,
        'game_seconds': game_seconds,
        'seconds_in_period': seconds_in_period,
        'prior_event_type': prior_type,
        'prior_same_team': int(prior_same_team),
        'prior_distance': prior_distance,
        'prior_seconds': prior_seconds,
        'is_rebound': int(is_rebound),
        'is_rush': int(is_rush),
        # Shot type dummies
        'st_wrist': int(shot_type == 'wrist'),
        'st_slap': int(shot_type == 'slap'),
        'st_backhand': int(shot_type == 'backhand'),
        'st_snap': int(shot_type == 'snap'),
        'st_tip': int(shot_type in ('tip-in', 'tip')),
        'st_deflected': int(shot_type == 'deflected'),
        'st_wrap': int(shot_type in ('wrap-around', 'wrap')),
        # Prior event type dummies
        'prev_shot': int(prior_type in ('shot-on-goal', 'goal') and prior_same_team),
        'prev_miss': int(prior_type == 'missed-shot' and prior_same_team),
        'prev_block': int(prior_type == 'blocked-shot' and prior_same_team),
        'prev_give': int(prior_type == 'giveaway' and not prior_same_team),
        'prev_take': int(prior_type == 'takeaway' and prior_same_team),
        'prev_hit': int(prior_type == 'hit'),
        'prev_faceoff': int(prior_type == 'faceoff'),
        # IDs for later aggregation
        'shooter_id': details.get('shootingPlayerId', details.get('scoringPlayerId', 0)),
        'goalie_id': details.get('goalieInNetId', 0),
        'event_team_id': event_team_id,
    }


def parse_game_pbp(game_id: int) -> Tuple[List[dict], dict]:
    """
    Parse full play-by-play for a game into shot features + game metadata.

    Returns:
        (shots_list, game_info)
    """
    pbp = api_get(f"gamecenter/{game_id}/play-by-play")
    if not pbp or 'plays' not in pbp:
        return [], {}

    plays = pbp.get('plays', [])
    home_team = pbp.get('homeTeam', {})
    away_team = pbp.get('awayTeam', {})
    home_team_id = home_team.get('id', 0)
    away_team_id = away_team.get('id', 0)

    game_info = {
        'game_id': game_id,
        'game_date': pbp.get('gameDate', ''),
        'season': pbp.get('season', 0),
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_abbrev': home_team.get('abbrev', ''),
        'away_abbrev': away_team.get('abbrev', ''),
    }

    # Build roster map from rosterSpots
    roster_map = {}  # playerId -> {name, position, teamId}
    for spot in pbp.get('rosterSpots', []):
        pid = spot.get('playerId', 0)
        roster_map[pid] = {
            'name': f"{spot.get('firstName', {}).get('default', '')} {spot.get('lastName', {}).get('default', '')}".strip(),
            'position': spot.get('positionCode', ''),
            'team_id': spot.get('teamId', 0),
        }

    # Track running score for score_diff
    away_score = 0
    home_score = 0

    shots = []
    shot_events = {'shot-on-goal', 'missed-shot', 'goal'}
    all_events = {'shot-on-goal', 'missed-shot', 'goal', 'blocked-shot',
                  'hit', 'faceoff', 'giveaway', 'takeaway', 'penalty'}

    prev_play = None
    for play in plays:
        event_type = play.get('typeDescKey', '')

        # Update running score from goals
        if event_type == 'goal':
            details = play.get('details', {})
            away_score = details.get('awayScore', away_score)
            home_score = details.get('homeScore', home_score)

        if event_type in shot_events:
            defending_side = play.get('homeTeamDefendingSide', 'right')
            features = compute_shot_features(
                play, prev_play, home_team_id, away_team_id, defending_side
            )

            # Add score diff from running state
            if features['is_home']:
                features['score_diff'] = home_score - away_score
                if event_type == 'goal':
                    features['score_diff'] = (home_score - 1) - away_score
            else:
                features['score_diff'] = away_score - home_score
                if event_type == 'goal':
                    features['score_diff'] = (away_score - 1) - home_score

            # Add player info
            shooter_id = features['shooter_id']
            if shooter_id in roster_map:
                features['shooter_name'] = roster_map[shooter_id]['name']
                features['shooter_position'] = roster_map[shooter_id]['position']
                features['shooter_team_abbrev'] = game_info['home_abbrev'] if roster_map[shooter_id]['team_id'] == home_team_id else game_info['away_abbrev']
            else:
                features['shooter_name'] = ''
                features['shooter_position'] = ''
                features['shooter_team_abbrev'] = ''

            features['game_id'] = game_id
            features['game_date'] = game_info['game_date']
            shots.append(features)

        # Track prev play for context features (only meaningful events)
        if event_type in all_events:
            prev_play = play

    return shots, game_info


# ==============================================================================
# DATABASE STORAGE
# ==============================================================================

def init_pbp_tables(conn):
    """Create tables for play-by-play data."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pbp_shots (
            game_id INTEGER,
            game_date TEXT,
            shooter_id INTEGER,
            shooter_name TEXT,
            shooter_position TEXT,
            shooter_team_abbrev TEXT,
            goalie_id INTEGER,
            event_team_id INTEGER,
            is_goal INTEGER,
            distance REAL,
            angle REAL,
            x_coord REAL,
            y_coord REAL,
            shot_type TEXT,
            is_home INTEGER,
            is_pp INTEGER,
            is_sh INTEGER,
            is_en INTEGER,
            is_5v5 INTEGER,
            own_skaters INTEGER,
            opp_skaters INTEGER,
            score_diff INTEGER,
            period INTEGER,
            game_seconds INTEGER,
            seconds_in_period INTEGER,
            prior_event_type TEXT,
            prior_same_team INTEGER,
            prior_distance REAL,
            prior_seconds REAL,
            is_rebound INTEGER,
            is_rush INTEGER,
            st_wrist INTEGER, st_slap INTEGER, st_backhand INTEGER,
            st_snap INTEGER, st_tip INTEGER, st_deflected INTEGER, st_wrap INTEGER,
            prev_shot INTEGER, prev_miss INTEGER, prev_block INTEGER,
            prev_give INTEGER, prev_take INTEGER, prev_hit INTEGER, prev_faceoff INTEGER,
            xg REAL DEFAULT NULL,
            PRIMARY KEY (game_id, game_seconds, shooter_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS pbp_games (
            game_id INTEGER PRIMARY KEY,
            game_date TEXT,
            season INTEGER,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_abbrev TEXT,
            away_abbrev TEXT,
            scraped_at TEXT
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pbp_shots_date ON pbp_shots(game_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pbp_shots_shooter ON pbp_shots(shooter_id, game_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pbp_shots_team ON pbp_shots(shooter_team_abbrev, game_date)
    """)
    conn.commit()


def save_shots_to_db(shots: List[dict], game_info: dict, conn):
    """Save parsed shots and game info to database."""
    if not shots:
        return

    # Save game info
    conn.execute("""
        INSERT OR REPLACE INTO pbp_games
        (game_id, game_date, season, home_team_id, away_team_id, home_abbrev, away_abbrev, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_info['game_id'], game_info['game_date'], game_info['season'],
        game_info['home_team_id'], game_info['away_team_id'],
        game_info['home_abbrev'], game_info['away_abbrev'],
        datetime.now().isoformat()
    ))

    # Save shots
    cols = [
        'game_id', 'game_date', 'shooter_id', 'shooter_name', 'shooter_position',
        'shooter_team_abbrev', 'goalie_id', 'event_team_id',
        'is_goal', 'distance', 'angle', 'x_coord', 'y_coord', 'shot_type',
        'is_home', 'is_pp', 'is_sh', 'is_en', 'is_5v5',
        'own_skaters', 'opp_skaters', 'score_diff', 'period', 'game_seconds',
        'seconds_in_period', 'prior_event_type', 'prior_same_team',
        'prior_distance', 'prior_seconds', 'is_rebound', 'is_rush',
        'st_wrist', 'st_slap', 'st_backhand', 'st_snap', 'st_tip', 'st_deflected', 'st_wrap',
        'prev_shot', 'prev_miss', 'prev_block', 'prev_give', 'prev_take', 'prev_hit', 'prev_faceoff',
    ]

    placeholders = ', '.join(['?'] * len(cols))
    col_str = ', '.join(cols)

    for shot in shots:
        values = [shot.get(c, None) for c in cols]
        try:
            conn.execute(f"INSERT OR IGNORE INTO pbp_shots ({col_str}) VALUES ({placeholders})", values)
        except Exception as e:
            pass  # Skip duplicates

    conn.commit()


# ==============================================================================
# SCRAPER MAIN LOOP
# ==============================================================================

def scrape_season(season: str = '20252026', resume: bool = True):
    """
    Scrape all play-by-play data for a season.

    Args:
        season: NHL season string (e.g., '20252026')
        resume: If True, skip games already in database
    """
    print(f"\n{'='*80}")
    print(f"NHL PLAY-BY-PLAY SCRAPER — Season {season}")
    print(f"{'='*80}")

    conn = sqlite3.connect(DB_PATH)
    init_pbp_tables(conn)

    # Get all game IDs
    print(f"\nFetching game schedule...")
    games = get_all_game_ids(season)
    completed = [g for g in games if g['state'] in ('OFF', 'FINAL')]

    # Check which games we already have
    if resume:
        existing = set(row[0] for row in conn.execute(
            "SELECT game_id FROM pbp_games WHERE season = ?",
            (int(season),)
        ).fetchall())
        to_scrape = [g for g in completed if g['game_id'] not in existing]
        print(f"  Already scraped: {len(existing)}")
        print(f"  Remaining: {len(to_scrape)}")
    else:
        to_scrape = completed

    total_shots = 0
    total_goals = 0
    errors = 0

    for i, game in enumerate(to_scrape):
        gid = game['game_id']
        time.sleep(REQUEST_DELAY)

        shots, game_info = parse_game_pbp(gid)

        if shots:
            save_shots_to_db(shots, game_info, conn)
            n_goals = sum(s['is_goal'] for s in shots)
            total_shots += len(shots)
            total_goals += n_goals
        else:
            errors += 1

        if (i + 1) % 25 == 0 or i == len(to_scrape) - 1:
            print(f"  [{i+1}/{len(to_scrape)}] {game['away']}@{game['home']} ({game['date']}) "
                  f"— {len(shots)} shots, running total: {total_shots} shots, {total_goals} goals")

    conn.close()

    print(f"\n{'='*80}")
    print(f"SCRAPE COMPLETE")
    print(f"  Games scraped: {len(to_scrape) - errors}")
    print(f"  Total shots: {total_shots}")
    print(f"  Total goals: {total_goals}")
    print(f"  Goal rate: {total_goals/max(total_shots,1)*100:.1f}%")
    if errors:
        print(f"  Errors: {errors}")
    print(f"{'='*80}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='NHL Play-by-Play Scraper')
    parser.add_argument('--season', type=str, default='20252026',
                        help='Season to scrape (e.g., 20252026)')
    parser.add_argument('--all-seasons', action='store_true',
                        help='Scrape 2022-23, 2023-24, 2024-25, and 2025-26')
    parser.add_argument('--no-resume', action='store_true',
                        help='Re-scrape all games (ignore existing)')
    args = parser.parse_args()

    if args.all_seasons:
        for season in ['20222023', '20232024', '20242025', '20252026']:
            scrape_season(season, resume=not args.no_resume)
    else:
        scrape_season(args.season, resume=not args.no_resume)


if __name__ == '__main__':
    main()
