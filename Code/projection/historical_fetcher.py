#!/usr/bin/env python3
"""
historical_fetcher.py — Fetch multi-season NHL boxscore data
=============================================================
Fetches boxscore data from NHL API for seasons 2020-21 through 2024-25.
This gives us ~5 full seasons plus the current partial season for:
  - Regression coefficient estimation (year-over-year correlations)
  - Multi-season signal validation
  - Larger HMM/Kalman training sets
  - Proper sample size for statistical significance

NHL API endpoints:
  Score/schedule: api-web.nhle.com/v1/score/{date}
  Boxscore:       api-web.nhle.com/v1/gamecenter/{gameId}/boxscore

Game IDs: {season_start_year}02{game_number:04d}
  - Regular season prefix: 02
  - ~1,312 games per full season (32 teams × 82 games / 2)

Usage:
    # Fetch all historical seasons
    python historical_fetcher.py --fetch-all

    # Fetch specific season
    python historical_fetcher.py --season 2023

    # Show status
    python historical_fetcher.py --status
"""

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "nhl_dfs_history.db"
CACHE_DIR = Path(__file__).parent / ".linemate_cache"

# Season definitions
SEASONS = {
    2020: {"start": "2021-01-13", "end": "2021-05-19", "id_prefix": 2020,
           "note": "COVID shortened, 56 games per team"},
    2021: {"start": "2021-10-12", "end": "2022-04-29", "id_prefix": 2021,
           "note": "Full 82-game season"},
    2022: {"start": "2022-10-07", "end": "2023-04-14", "id_prefix": 2022,
           "note": "Full 82-game season"},
    2023: {"start": "2023-10-10", "end": "2024-04-18", "id_prefix": 2023,
           "note": "Full 82-game season"},
    2024: {"start": "2024-10-04", "end": "2025-04-17", "id_prefix": 2024,
           "note": "Full 82-game season"},
}

REQUEST_DELAY = 0.3  # seconds between API requests
MAX_RETRIES = 3


def get_db():
    """Get database connection."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # Create historical boxscore table (separate from current season)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_skaters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            player_id INTEGER,
            player_name TEXT NOT NULL,
            team TEXT NOT NULL,
            position TEXT,
            game_id INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            opponent TEXT,
            home_road TEXT,
            goals INTEGER DEFAULT 0,
            assists INTEGER DEFAULT 0,
            points INTEGER DEFAULT 0,
            plus_minus INTEGER DEFAULT 0,
            pim INTEGER DEFAULT 0,
            hits INTEGER DEFAULT 0,
            shots INTEGER DEFAULT 0,
            blocked_shots INTEGER DEFAULT 0,
            shifts INTEGER DEFAULT 0,
            toi TEXT,
            toi_seconds INTEGER DEFAULT 0,
            pp_goals INTEGER DEFAULT 0,
            pp_points INTEGER DEFAULT 0,
            sh_goals INTEGER DEFAULT 0,
            sh_points INTEGER DEFAULT 0,
            takeaways INTEGER DEFAULT 0,
            giveaways INTEGER DEFAULT 0,
            faceoff_pct REAL DEFAULT 0,
            dk_fpts REAL DEFAULT 0,
            UNIQUE(season, game_id, player_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_goalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            player_id INTEGER,
            player_name TEXT NOT NULL,
            team TEXT NOT NULL,
            game_id INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            opponent TEXT,
            home_road TEXT,
            decision TEXT,
            shots_against INTEGER DEFAULT 0,
            saves INTEGER DEFAULT 0,
            goals_against INTEGER DEFAULT 0,
            sv_pct REAL DEFAULT 0,
            toi TEXT,
            toi_seconds INTEGER DEFAULT 0,
            UNIQUE(season, game_id, player_id)
        )
    """)

    # Index for fast queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_sk_player ON historical_skaters(player_name, season)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_sk_date ON historical_skaters(game_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_sk_season ON historical_skaters(season)")

    conn.commit()
    return conn


def fetch_boxscore(game_id: int) -> Optional[dict]:
    """Fetch a single game boxscore, using cache if available."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = CACHE_DIR / f"boxscore_{game_id}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                return data
            elif resp.status_code == 404:
                return None  # game doesn't exist
        except requests.RequestException:
            time.sleep(1)

    return None


def toi_to_seconds(toi_str: str) -> int:
    """Convert TOI string like '18:45' to seconds."""
    if not toi_str or toi_str == '--':
        return 0
    try:
        parts = str(toi_str).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 0


def compute_dk_fpts(stats: dict) -> float:
    """Compute DraftKings fantasy points from stat dict."""
    g = stats.get('goals', 0) or 0
    a = stats.get('assists', 0) or 0
    sog = stats.get('shots', 0) or 0
    blk = stats.get('blockedShots', 0) or 0
    pm = stats.get('plusMinus', 0) or 0

    fpts = g * 8.5 + a * 5.0 + sog * 1.5 + blk * 1.3 + pm * 0.5

    # Bonuses
    pts = g + a
    if g >= 3:
        fpts += 3.0  # hat trick
    if pts >= 3:
        fpts += 3.0  # 3+ points
    if sog >= 5:
        fpts += 1.0  # 5+ SOG
    if blk >= 3:
        fpts += 1.0  # 3+ blocks

    return round(fpts, 1)


def parse_boxscore(game_data: dict, season: int) -> Tuple[List[dict], List[dict]]:
    """Parse a boxscore JSON into skater and goalie rows."""
    skaters = []
    goalies = []

    game_id = game_data.get('id', 0)
    game_date = game_data.get('gameDate', '')
    game_type = game_data.get('gameType', 0)

    # Only regular season games (type 2)
    if game_type != 2:
        return [], []

    away_team = game_data.get('awayTeam', {})
    home_team = game_data.get('homeTeam', {})
    away_abbr = away_team.get('abbrev', '')
    home_abbr = home_team.get('abbrev', '')

    player_by_team = game_data.get('playerByGameStats', {})
    if not player_by_team:
        # Try alternate structure
        boxscore = game_data.get('boxscore', {})
        if boxscore:
            player_by_team = boxscore.get('playerByGameStats', {})

    if not player_by_team:
        return [], []

    for side, team_abbr, opp_abbr, hr in [
        ('awayTeam', away_abbr, home_abbr, 'R'),
        ('homeTeam', home_abbr, away_abbr, 'H')
    ]:
        team_data = player_by_team.get(side, {})
        if not team_data:
            continue

        # Forwards and defense
        for section in ['forwards', 'defense']:
            for player in team_data.get(section, []):
                pid = player.get('playerId', 0)
                name_data = player.get('name', {})
                if isinstance(name_data, dict):
                    first = name_data.get('default', name_data.get('first', ''))
                    # Actually the API uses firstName/lastName
                    pname = player.get('firstName', {}).get('default', '') + ' ' + player.get('lastName', {}).get('default', '')
                    if pname.strip() == '':
                        pname = str(name_data.get('default', f'Player {pid}'))
                else:
                    pname = str(name_data)

                # Abbreviated name format: "N. MacKinnon"
                parts = pname.strip().split()
                if len(parts) >= 2:
                    abbr_name = parts[0][0] + '. ' + ' '.join(parts[1:])
                else:
                    abbr_name = pname

                toi = player.get('toi', '0:00')
                toi_sec = toi_to_seconds(toi)

                pos = player.get('position', section[0].upper())
                if pos in ('LW', 'RW'):
                    pos = pos[0]

                stats = {
                    'goals': player.get('goals', 0),
                    'assists': player.get('assists', 0),
                    'shots': player.get('sog', player.get('shots', 0)),
                    'blockedShots': player.get('blockedShots', player.get('blocked_shots', 0)),
                    'plusMinus': player.get('plusMinus', player.get('plus_minus', 0)),
                }

                dk = compute_dk_fpts(stats)

                skaters.append({
                    'season': season,
                    'player_id': pid,
                    'player_name': abbr_name,
                    'team': team_abbr,
                    'position': pos,
                    'game_id': game_id,
                    'game_date': game_date,
                    'opponent': opp_abbr,
                    'home_road': hr,
                    'goals': stats['goals'],
                    'assists': stats['assists'],
                    'points': (stats['goals'] or 0) + (stats['assists'] or 0),
                    'plus_minus': stats['plusMinus'],
                    'pim': player.get('pim', 0),
                    'hits': player.get('hits', 0),
                    'shots': stats['shots'],
                    'blocked_shots': stats['blockedShots'],
                    'shifts': player.get('shifts', 0),
                    'toi': toi,
                    'toi_seconds': toi_sec,
                    'pp_goals': player.get('powerPlayGoals', 0),
                    'pp_points': player.get('powerPlayPoints', 0),
                    'sh_goals': player.get('shorthandedGoals', 0),
                    'sh_points': player.get('shorthandedPoints', 0),
                    'takeaways': player.get('takeaways', 0),
                    'giveaways': player.get('giveaways', 0),
                    'faceoff_pct': player.get('faceoffWinningPctg', 0),
                    'dk_fpts': dk,
                })

        # Goalies
        for player in team_data.get('goalies', []):
            pid = player.get('playerId', 0)
            pname = player.get('firstName', {}).get('default', '') + ' ' + player.get('lastName', {}).get('default', '')
            parts = pname.strip().split()
            abbr_name = parts[0][0] + '. ' + ' '.join(parts[1:]) if len(parts) >= 2 else pname

            toi = player.get('toi', '0:00')
            toi_sec = toi_to_seconds(toi)

            goalies.append({
                'season': season,
                'player_id': pid,
                'player_name': abbr_name,
                'team': team_abbr,
                'game_id': game_id,
                'game_date': game_date,
                'opponent': opp_abbr,
                'home_road': hr,
                'decision': player.get('decision', ''),
                'shots_against': player.get('shotsAgainst', player.get('saveShotsAgainst', '').split('/')[1] if '/' in str(player.get('saveShotsAgainst', '')) else 0),
                'saves': player.get('saves', player.get('saveShotsAgainst', '').split('/')[0] if '/' in str(player.get('saveShotsAgainst', '')) else 0),
                'goals_against': player.get('goalsAgainst', 0),
                'sv_pct': player.get('savePctg', 0),
                'toi': toi,
                'toi_seconds': toi_sec,
            })

    return skaters, goalies


def store_skaters(conn: sqlite3.Connection, skaters: List[dict]) -> int:
    """Store skater rows into historical_skaters table."""
    count = 0
    for s in skaters:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO historical_skaters
                (season, player_id, player_name, team, position, game_id, game_date,
                 opponent, home_road, goals, assists, points, plus_minus, pim,
                 hits, shots, blocked_shots, shifts, toi, toi_seconds,
                 pp_goals, pp_points, sh_goals, sh_points,
                 takeaways, giveaways, faceoff_pct, dk_fpts)
                VALUES (?,?,?,?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?, ?,?,?,?)
            """, (
                s['season'], s['player_id'], s['player_name'], s['team'],
                s['position'], s['game_id'], s['game_date'],
                s['opponent'], s['home_road'],
                s.get('goals', 0), s.get('assists', 0), s.get('points', 0),
                s.get('plus_minus', 0), s.get('pim', 0),
                s.get('hits', 0), s.get('shots', 0), s.get('blocked_shots', 0),
                s.get('shifts', 0), s.get('toi', ''), s.get('toi_seconds', 0),
                s.get('pp_goals', 0), s.get('pp_points', 0),
                s.get('sh_goals', 0), s.get('sh_points', 0),
                s.get('takeaways', 0), s.get('giveaways', 0),
                s.get('faceoff_pct', 0), s.get('dk_fpts', 0),
            ))
            count += 1
        except Exception as e:
            pass  # skip duplicates silently
    conn.commit()
    return count


def fetch_season(season_year: int, conn: sqlite3.Connection = None) -> Dict:
    """Fetch all regular season boxscores for a given season."""
    info = SEASONS.get(season_year)
    if not info:
        print(f"Unknown season: {season_year}")
        return {}

    own_conn = conn is None
    if own_conn:
        conn = get_db()

    prefix = info['id_prefix']
    start_date = info['start']
    end_date = info['end']

    # Check what we already have
    existing = conn.execute(
        "SELECT COUNT(DISTINCT game_id) FROM historical_skaters WHERE season = ?",
        (season_year,)
    ).fetchone()[0]

    print(f"\n{'='*60}")
    print(f"  SEASON {season_year}-{season_year+1}")
    print(f"  {info['note']}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Already have: {existing} games")
    print(f"{'='*60}")

    if existing >= 1200:  # ~1312 max regular season games
        print(f"  Season appears complete, skipping.")
        if own_conn:
            conn.close()
        return {'games': existing, 'skaters': 0, 'new': 0}

    # Fetch game-by-game using score endpoint (day by day)
    total_skaters = 0
    total_games = 0
    new_games = 0

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://api-web.nhle.com/v1/score/{date_str}"

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                games = data.get('games', [])

                for game in games:
                    gid = game.get('id', 0)
                    gtype = game.get('gameType', 0)

                    if gtype != 2:  # regular season only
                        continue

                    total_games += 1

                    # Check if already stored
                    exists = conn.execute(
                        "SELECT COUNT(*) FROM historical_skaters WHERE game_id = ? AND season = ?",
                        (gid, season_year)
                    ).fetchone()[0]

                    if exists > 0:
                        continue

                    # Fetch and parse boxscore
                    box = fetch_boxscore(gid)
                    if box:
                        skaters, goalies = parse_boxscore(box, season_year)
                        if skaters:
                            n = store_skaters(conn, skaters)
                            total_skaters += n
                            new_games += 1

                    time.sleep(REQUEST_DELAY)

        except Exception as e:
            pass  # skip bad dates

        current += timedelta(days=1)

        # Progress every 30 days
        if (current - datetime.strptime(start_date, "%Y-%m-%d")).days % 30 == 0:
            print(f"  {date_str}: {total_games} games found, {new_games} new, {total_skaters} skaters")

    print(f"\n  Season {season_year} complete: {total_games} total games, "
          f"{new_games} new, {total_skaters} new skater rows")

    if own_conn:
        conn.close()

    return {'games': total_games, 'skaters': total_skaters, 'new': new_games}


def show_status():
    """Show historical data status."""
    conn = get_db()
    print(f"\n{'='*60}")
    print(f"  HISTORICAL DATA STATUS")
    print(f"{'='*60}")

    for season in sorted(SEASONS.keys()):
        try:
            r = conn.execute("""
                SELECT COUNT(*) as rows, COUNT(DISTINCT game_id) as games,
                       COUNT(DISTINCT player_name) as players,
                       MIN(game_date) as first, MAX(game_date) as last
                FROM historical_skaters WHERE season = ?
            """, (season,)).fetchone()
            print(f"\n  {season}-{season+1}: {r[0]:,} rows, {r[1]} games, "
                  f"{r[2]} players ({r[3]} to {r[4]})")
        except:
            print(f"\n  {season}-{season+1}: no data")

    # Also check current season
    try:
        r = conn.execute("""
            SELECT COUNT(*) as rows, COUNT(DISTINCT game_id) as games,
                   COUNT(DISTINCT player_name) as players
            FROM boxscore_skaters
        """).fetchone()
        print(f"\n  2025-26 (current): {r[0]:,} rows, {r[1]} games, {r[2]} players")
    except:
        pass

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical NHL Data Fetcher")
    parser.add_argument("--fetch-all", action="store_true",
                        help="Fetch all historical seasons (2020-2024)")
    parser.add_argument("--season", type=int, default=None,
                        help="Fetch specific season (e.g., 2023)")
    parser.add_argument("--status", action="store_true",
                        help="Show data status")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.fetch_all:
        conn = get_db()
        for season in sorted(SEASONS.keys()):
            fetch_season(season, conn)
        conn.close()
        show_status()
    elif args.season:
        fetch_season(args.season)
        show_status()
    else:
        parser.print_help()
