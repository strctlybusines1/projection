#!/usr/bin/env python3
"""
Backfill goalie data from cached boxscore JSON files into historical_goalies table.

This script:
1. Adds missing columns to historical_goalies (dk_fpts, goals, assists)
2. Reads all cached boxscore JSON files from .linemate_cache/
3. Extracts goalie data from both home and away teams
4. Computes DK fantasy points using the scoring formula
5. Inserts/updates goalie records in the database
6. Prints summary statistics by season
"""

import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Database and cache paths
DB_PATH = "/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db"
CACHE_DIR = "/sessions/youthful-funny-faraday/mnt/Code/projection/.linemate_cache"

# DK Goalie Scoring
DK_SCORING = {
    "win": 6.0,
    "save": 0.7,
    "goal_against": -3.5,
    "shutout_bonus": 2.0,
    "goal": 8.5,
    "assist": 5.0,
}


def toi_to_seconds(toi_str):
    """Convert 'MM:SS' format to seconds, handling empty/invalid values."""
    if not toi_str or toi_str == "00:00":
        return 0
    try:
        parts = toi_str.split(":")
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0


def parse_save_shots_format(save_shots_str):
    """
    Parse format like '25/28' or '31/34' into (saves, shots_against).
    Returns (saves, shots_against) tuple, or (0, 0) if parsing fails.
    """
    if not save_shots_str or "/" not in save_shots_str:
        return 0, 0
    try:
        parts = save_shots_str.split("/")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return 0, 0


def calculate_dk_fpts(goalie_data):
    """
    Calculate DK fantasy points for a goalie based on their stats.

    Scoring:
    - Win: +6.0
    - Save: +0.7 each
    - Goal Against: -3.5 each
    - Shutout Bonus: +2.0 (if goals_against == 0)
    - Goal: +8.5 each (rare)
    - Assist: +5.0 each
    """
    fpts = 0.0

    # Win bonus
    if goalie_data.get("decision") == "W":
        fpts += DK_SCORING["win"]

    # Saves
    saves = goalie_data.get("saves", 0) or 0
    fpts += saves * DK_SCORING["save"]

    # Goals against
    goals_against = goalie_data.get("goals_against", 0) or 0
    fpts += goals_against * DK_SCORING["goal_against"]

    # Shutout bonus (if played significant time and allowed no goals)
    if goals_against == 0 and goalie_data.get("toi_seconds", 0) > 0:
        fpts += DK_SCORING["shutout_bonus"]

    # Goalie goals (very rare)
    goals = goalie_data.get("goals", 0) or 0
    fpts += goals * DK_SCORING["goal"]

    # Goalie assists
    assists = goalie_data.get("assists", 0) or 0
    fpts += assists * DK_SCORING["assist"]

    return fpts


def extract_goalie_data(game_data, game_id, game_date, season, team_side="homeTeam"):
    """
    Extract goalie data from a game JSON for either homeTeam or awayTeam.
    Returns list of goalie records.
    """
    goalies = []

    # Get team abbreviation
    team_data = game_data.get(team_side, {})
    team_abbrev = team_data.get("abbrev", "")

    # Get goalies from playerByGameStats
    player_stats = game_data.get("playerByGameStats", {})
    team_goalies = player_stats.get(team_side, {}).get("goalies", [])

    # Determine opponent abbreviation
    if team_side == "homeTeam":
        opponent_abbrev = game_data.get("awayTeam", {}).get("abbrev", "")
        home_road = "H"
    else:
        opponent_abbrev = game_data.get("homeTeam", {}).get("abbrev", "")
        home_road = "R"

    for goalie in team_goalies:
        # Parse goalie name - try 'name' field first, fall back to firstName/lastName
        player_name = goalie.get("name", {})
        if isinstance(player_name, dict):
            player_name = player_name.get("default", "")

        if not player_name:
            # Fall back to firstName/lastName format
            first_name = goalie.get("firstName", {})
            if isinstance(first_name, dict):
                first_name = first_name.get("default", "")

            last_name = goalie.get("lastName", {})
            if isinstance(last_name, dict):
                last_name = last_name.get("default", "")

            player_name = f"{first_name} {last_name}".strip()

        player_id = goalie.get("playerId")

        # Skip if no player name or ID
        if not player_name or not player_id:
            continue

        # Parse TOI
        toi = goalie.get("toi", "00:00")
        toi_seconds = toi_to_seconds(toi)

        # Skip if goalie didn't play (0 TOI)
        if toi_seconds == 0:
            continue

        # Parse saves and shots against
        # Try new format first (separate fields)
        saves = goalie.get("saves")
        shots_against = goalie.get("shotsAgainst")

        # Fall back to combined format if needed
        if saves is None or shots_against is None:
            save_shots_str = goalie.get("saveShotsAgainst", "0/0")
            saves, shots_against = parse_save_shots_format(save_shots_str)

        saves = saves or 0
        shots_against = shots_against or 0

        # Parse other stats
        goals_against = goalie.get("goalsAgainst", 0) or 0
        sv_pct = goalie.get("savePctg", 0.0) or 0.0
        decision = goalie.get("decision", "")
        goals = goalie.get("goals", 0) or 0
        assists = goalie.get("assists", 0) or 0

        # Prepare goalie record
        goalie_record = {
            "season": season,
            "player_id": player_id,
            "player_name": player_name,
            "team": team_abbrev,
            "game_id": game_id,
            "game_date": game_date,
            "opponent": opponent_abbrev,
            "home_road": home_road,
            "decision": decision,
            "shots_against": shots_against,
            "saves": saves,
            "goals_against": goals_against,
            "sv_pct": sv_pct,
            "toi": toi,
            "toi_seconds": toi_seconds,
            "goals": goals,
            "assists": assists,
        }

        # Calculate DK fantasy points
        goalie_record["dk_fpts"] = calculate_dk_fpts(goalie_record)

        goalies.append(goalie_record)

    return goalies


def extract_season_from_game_id(game_id):
    """Extract season from game_id. First 4 digits are the season year (2020, 2021, etc.)"""
    return int(str(game_id)[:4])


def setup_database(db_path):
    """Add missing columns to historical_goalies table if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check which columns exist
    cursor.execute("PRAGMA table_info(historical_goalies)")
    columns = {row[1] for row in cursor.fetchall()}

    # Add missing columns
    if "dk_fpts" not in columns:
        print("Adding dk_fpts column...")
        cursor.execute("""
            ALTER TABLE historical_goalies
            ADD COLUMN dk_fpts REAL DEFAULT 0.0
        """)

    if "goals" not in columns:
        print("Adding goals column...")
        cursor.execute("""
            ALTER TABLE historical_goalies
            ADD COLUMN goals INTEGER DEFAULT 0
        """)

    if "assists" not in columns:
        print("Adding assists column...")
        cursor.execute("""
            ALTER TABLE historical_goalies
            ADD COLUMN assists INTEGER DEFAULT 0
        """)

    conn.commit()
    conn.close()


def backfill_goalies(db_path, cache_dir):
    """
    Backfill goalie data from cached boxscore JSON files.
    """
    print(f"Starting goalie backfill from cache: {cache_dir}")
    print(f"Database: {db_path}")
    print()

    # Setup database schema
    setup_database(db_path)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Track statistics by season
    stats_by_season = defaultdict(lambda: {"inserted": 0, "updated": 0, "errors": 0})

    # Find all boxscore JSON files
    cache_path = Path(cache_dir)
    boxscore_files = sorted(cache_path.glob("boxscore_*.json"))

    print(f"Found {len(boxscore_files)} boxscore files to process")
    print()

    for idx, file_path in enumerate(boxscore_files, 1):
        if idx % 500 == 0:
            print(f"Processing file {idx}/{len(boxscore_files)}...")

        try:
            # Load JSON
            with open(file_path, "r") as f:
                game_data = json.load(f)

            game_id = game_data.get("id")
            game_date = game_data.get("gameDate", "")

            if not game_id or not game_date:
                continue

            # Extract season from game_id
            season = extract_season_from_game_id(game_id)

            # Extract goalies from both teams
            all_goalies = []
            all_goalies.extend(extract_goalie_data(game_data, game_id, game_date, season, "homeTeam"))
            all_goalies.extend(extract_goalie_data(game_data, game_id, game_date, season, "awayTeam"))

            # Insert/update goalies in database
            for goalie in all_goalies:
                try:
                    cursor.execute("""
                        INSERT INTO historical_goalies (
                            season, player_id, player_name, team, game_id, game_date,
                            opponent, home_road, decision, shots_against, saves,
                            goals_against, sv_pct, toi, toi_seconds, goals, assists, dk_fpts
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(season, game_id, player_id) DO UPDATE SET
                            decision = excluded.decision,
                            shots_against = excluded.shots_against,
                            saves = excluded.saves,
                            goals_against = excluded.goals_against,
                            sv_pct = excluded.sv_pct,
                            toi = excluded.toi,
                            toi_seconds = excluded.toi_seconds,
                            goals = excluded.goals,
                            assists = excluded.assists,
                            dk_fpts = excluded.dk_fpts
                    """, (
                        goalie["season"],
                        goalie["player_id"],
                        goalie["player_name"],
                        goalie["team"],
                        goalie["game_id"],
                        goalie["game_date"],
                        goalie["opponent"],
                        goalie["home_road"],
                        goalie["decision"],
                        goalie["shots_against"],
                        goalie["saves"],
                        goalie["goals_against"],
                        goalie["sv_pct"],
                        goalie["toi"],
                        goalie["toi_seconds"],
                        goalie["goals"],
                        goalie["assists"],
                        goalie["dk_fpts"],
                    ))

                    stats_by_season[season]["inserted"] += 1

                except sqlite3.IntegrityError:
                    stats_by_season[season]["updated"] += 1
                except Exception as e:
                    print(f"Error inserting goalie: {e}")
                    stats_by_season[season]["errors"] += 1

        except json.JSONDecodeError:
            stats_by_season["unknown"]["errors"] += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Commit all changes
    conn.commit()
    conn.close()

    # Print summary
    print()
    print("=" * 70)
    print("GOALIE BACKFILL SUMMARY")
    print("=" * 70)
    print()

    for season in sorted(stats_by_season.keys()):
        stats = stats_by_season[season]
        total = stats["inserted"] + stats["updated"]
        print(f"Season {season}:")
        print(f"  Inserted: {stats['inserted']:,}")
        print(f"  Updated:  {stats['updated']:,}")
        print(f"  Total:    {total:,}")
        if stats["errors"]:
            print(f"  Errors:   {stats['errors']}")
        print()

    # Print overall totals
    total_inserted = sum(s["inserted"] for s in stats_by_season.values())
    total_updated = sum(s["updated"] for s in stats_by_season.values())
    total_errors = sum(s["errors"] for s in stats_by_season.values())

    print(f"GRAND TOTAL:")
    print(f"  Total Inserted: {total_inserted:,}")
    print(f"  Total Updated:  {total_updated:,}")
    print(f"  Total Errors:   {total_errors:,}")
    print(f"  Grand Total:    {total_inserted + total_updated:,}")
    print()
    print("Goalie backfill completed successfully!")


if __name__ == "__main__":
    backfill_goalies(DB_PATH, CACHE_DIR)
