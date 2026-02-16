"""
linemate_corr.py — Linemate Correlation Analysis for NHL DFS Projections

Pulls play-by-play and boxscore data from the NHL API to identify linemate
pairings and calculate chemistry metrics. Produces boost factors that integrate
into the projection pipeline alongside EDGE boosts.

Usage:
    # Standalone — print correlation report for tonight's slate
    python linemate_corr.py

    # Import into projections pipeline
    from linemate_corr import get_linemate_boosts
    boosts = get_linemate_boosts(player_names, team_abbrevs, n_games=10)

Integration with main.py:
    Add --linemates flag to apply linemate chemistry boosts after EDGE boosts.
"""

import requests
import time
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = "https://api-web.nhle.com/v1/"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".linemate_cache")

# How many recent games to analyze per team (more = stable, fewer = reactive)
DEFAULT_GAME_WINDOW = 10

# Minimum shared events to consider a pairing meaningful
MIN_SHARED_EVENTS = 5

# Boost thresholds (calibrated conservatively to start)
BOOST_TIERS = {
    "elite":      {"threshold": 0.75, "boost": 0.08},   # top-tier chemistry
    "above_avg":  {"threshold": 0.50, "boost": 0.04},   # solid pairing
    "average":    {"threshold": 0.25, "boost": 0.00},   # neutral
}

# Rate limiting — be polite to the NHL API
REQUEST_DELAY = 0.4  # seconds between requests


# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, use_cache: bool = True) -> dict:
    """GET from NHL API with optional file cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = endpoint.replace("/", "_").replace("?", "_")
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # Check cache (valid for 2 hours)
    if use_cache and os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < 7200:
            with open(cache_path, "r") as f:
                return json.load(f)

    time.sleep(REQUEST_DELAY)
    resp = requests.get(API_URL + endpoint, params={"Content-Type": "application/json"})
    resp.raise_for_status()
    data = resp.json()

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def get_team_recent_game_ids(team_abbrev: str, n_games: int = DEFAULT_GAME_WINDOW) -> list[int]:
    """Get the last N completed game IDs for a team via club schedule."""
    today = datetime.now()
    game_ids = []

    # Walk backwards week by week until we have enough games
    for weeks_back in range(0, 12):
        date = today - timedelta(weeks=weeks_back)
        date_str = date.strftime("%Y-%m-%d")
        try:
            data = _get(f"club-schedule/{team_abbrev}/week/{date_str}")
        except Exception:
            continue

        for game in data.get("games", []):
            if game.get("gameState") in ("OFF", "FINAL") and game["id"] not in game_ids:
                game_ids.append(game["id"])

        if len(game_ids) >= n_games:
            break

    # Sort descending (most recent first) and trim
    game_ids = sorted(set(game_ids), reverse=True)[:n_games]
    return game_ids


# ---------------------------------------------------------------------------
# Play-by-Play Parsing
# ---------------------------------------------------------------------------

def parse_game_events(game_id: int) -> dict:
    """
    Parse play-by-play for a single game. Returns:
    {
        "game_id": int,
        "roster": {player_id: {"name": str, "team": str, "pos": str}},
        "goal_events": [{"scorer_id", "assist1_id", "assist2_id", "period", ...}],
        "shot_events": [{"shooter_id", "period", ...}],
    }
    """
    pbp = _get(f"gamecenter/{game_id}/play-by-play")

    # Build roster lookup
    roster = {}
    for spot in pbp.get("rosterSpots", []):
        pid = spot.get("playerId")
        if pid:
            first = spot.get("firstName", {}).get("default", "")
            last = spot.get("lastName", {}).get("default", "")
            roster[pid] = {
                "name": f"{first} {last}".strip(),
                "team": spot.get("teamId"),
                "pos": spot.get("positionCode", ""),
                "sweater": spot.get("sweaterNumber", 0),
            }

    # Map teamId -> team abbreviation from game data
    home_team_id = pbp.get("homeTeam", {}).get("id")
    away_team_id = pbp.get("awayTeam", {}).get("id")
    home_abbrev = pbp.get("homeTeam", {}).get("abbrev", "")
    away_abbrev = pbp.get("awayTeam", {}).get("abbrev", "")
    team_id_to_abbrev = {home_team_id: home_abbrev, away_team_id: away_abbrev}

    # Update roster with abbreviations
    for pid in roster:
        tid = roster[pid]["team"]
        roster[pid]["team_abbrev"] = team_id_to_abbrev.get(tid, "")

    goal_events = []
    shot_events = []

    for play in pbp.get("plays", []):
        event_type = play.get("typeDescKey", "")
        details = play.get("details", {})
        period = play.get("periodDescriptor", {}).get("number", 0)
        time_in_period = play.get("timeInPeriod", "00:00")

        if event_type == "goal":
            event = {
                "game_id": game_id,
                "period": period,
                "time": time_in_period,
                "scorer_id": details.get("scoringPlayerId"),
                "assist1_id": details.get("assist1PlayerId"),
                "assist2_id": details.get("assist2PlayerId"),
                "home_score": details.get("homeScore", 0),
                "away_score": details.get("awayScore", 0),
                "event_owner_team": details.get("eventOwnerTeamId"),
            }
            goal_events.append(event)

        elif event_type == "shot-on-goal":
            event = {
                "game_id": game_id,
                "period": period,
                "time": time_in_period,
                "shooter_id": details.get("shootingPlayerId"),
                "event_owner_team": details.get("eventOwnerTeamId"),
            }
            shot_events.append(event)

    return {
        "game_id": game_id,
        "roster": roster,
        "goal_events": goal_events,
        "shot_events": shot_events,
        "team_id_to_abbrev": team_id_to_abbrev,
    }


def get_boxscore_toi(game_id: int) -> dict[int, float]:
    """Get TOI per player from boxscore. Returns {player_id: toi_seconds}."""
    box = _get(f"gamecenter/{game_id}/boxscore")
    toi_map = {}

    for side in ("homeTeam", "awayTeam"):
        team_data = box.get("playerByGameStats", {}).get(side, {})
        for pos_group in ("forwards", "defense", "goalies"):
            for player in team_data.get(pos_group, []):
                pid = player.get("playerId")
                toi_str = player.get("toi", "0:00")
                parts = toi_str.split(":")
                if len(parts) == 2:
                    toi_map[pid] = int(parts[0]) * 60 + int(parts[1])
    return toi_map


# ---------------------------------------------------------------------------
# Linemate Pair Correlation
# ---------------------------------------------------------------------------

def build_pair_stats(game_ids: list[int], team_abbrev: str = None) -> dict:
    """
    Across multiple games, build pair-level stats:
    - shared_goals: times both players were involved in same goal (scorer + assists)
    - shared_goal_chains: specific (playerA_role, playerB_role) combos
    - games_together: number of games both appeared in roster
    - individual stats per player for context

    Returns:
        {
            "pairs": {(pidA, pidB): {stats}},
            "players": {pid: {aggregate stats}},
            "roster_lookup": {pid: {name, team, pos}},
        }
    """
    pairs = defaultdict(lambda: {
        "shared_goals": 0,
        "shared_goal_chains": [],
        "games_together": 0,
        "combined_goals": 0,
        "combined_assists": 0,
    })
    players = defaultdict(lambda: {
        "goals": 0,
        "primary_assists": 0,
        "secondary_assists": 0,
        "shots": 0,
        "games": 0,
        "toi_total": 0,
    })
    roster_lookup = {}

    for gid in game_ids:
        try:
            parsed = parse_game_events(gid)
            toi_map = get_boxscore_toi(gid)
        except Exception as e:
            print(f"  ⚠ Skipping game {gid}: {e}")
            continue

        roster = parsed["roster"]
        roster_lookup.update(roster)

        # Filter to team if specified
        if team_abbrev:
            team_pids = {pid for pid, info in roster.items()
                         if info.get("team_abbrev") == team_abbrev}
        else:
            team_pids = set(roster.keys())

        # Track who played this game
        game_players = set()
        for pid in team_pids:
            if pid in toi_map and toi_map[pid] > 0:
                game_players.add(pid)
                players[pid]["games"] += 1
                players[pid]["toi_total"] += toi_map.get(pid, 0)

        # Track pairs who both played
        for a, b in combinations(sorted(game_players), 2):
            pairs[(a, b)]["games_together"] += 1

        # Process goals — find shared involvement
        for goal in parsed["goal_events"]:
            scorer = goal["scorer_id"]
            a1 = goal["assist1_id"]
            a2 = goal["assist2_id"]

            involved = [p for p in [scorer, a1, a2] if p and p in team_pids]

            # Update individual stats
            if scorer and scorer in team_pids:
                players[scorer]["goals"] += 1
            if a1 and a1 in team_pids:
                players[a1]["primary_assists"] += 1
            if a2 and a2 in team_pids:
                players[a2]["secondary_assists"] += 1

            # Update pair stats for all combos of involved players
            for pa, pb in combinations(sorted(involved), 2):
                pairs[(pa, pb)]["shared_goals"] += 1

                # Track the chain (who did what)
                roles = {}
                if scorer in (pa, pb):
                    roles[scorer] = "G"
                if a1 in (pa, pb):
                    roles[a1] = "A1"
                if a2 in (pa, pb):
                    roles[a2] = "A2"
                pairs[(pa, pb)]["shared_goal_chains"].append(roles)

        # Process shots — individual only
        for shot in parsed["shot_events"]:
            shooter = shot["shooter_id"]
            if shooter and shooter in team_pids:
                players[shooter]["shots"] += 1

    return {
        "pairs": dict(pairs),
        "players": dict(players),
        "roster_lookup": roster_lookup,
    }


def calculate_chemistry_scores(pair_stats: dict) -> list[dict]:
    """
    Calculate a chemistry score for each pair based on:
    - shared_goal_rate: shared goals / games together
    - involvement_ratio: what % of their combined goals were shared
    - consistency: did they produce across multiple games or just one blowup?

    Returns sorted list of pair dicts with chemistry_score.
    """
    pairs = pair_stats["pairs"]
    players = pair_stats["players"]
    roster = pair_stats["roster_lookup"]

    results = []

    for (pid_a, pid_b), pdata in pairs.items():
        games_tog = pdata["games_together"]
        if games_tog < 2:
            continue

        shared_goals = pdata["shared_goals"]
        if shared_goals < 1:
            continue

        # Individual production
        a_goals = players.get(pid_a, {}).get("goals", 0)
        b_goals = players.get(pid_b, {}).get("goals", 0)
        a_pts = a_goals + players.get(pid_a, {}).get("primary_assists", 0) + players.get(pid_a, {}).get("secondary_assists", 0)
        b_pts = b_goals + players.get(pid_b, {}).get("primary_assists", 0) + players.get(pid_b, {}).get("secondary_assists", 0)
        combined_pts = a_pts + b_pts

        # Shared goal rate per game together
        shared_goal_rate = shared_goals / games_tog

        # What fraction of their combined points came from shared goals?
        # Each shared goal contributes 2+ points (scorer + assister(s))
        if combined_pts > 0:
            involvement_ratio = (shared_goals * 2) / combined_pts
        else:
            involvement_ratio = 0

        # Consistency: how many distinct games had shared goals?
        games_with_shared = len(set(
            chain.get("game_id") for chain in pdata.get("shared_goal_chains", [])
            if isinstance(chain, dict) and "game_id" in chain
        )) if False else min(shared_goals, games_tog)  # approximation

        consistency = games_with_shared / games_tog if games_tog > 0 else 0

        # Composite chemistry score (0-1 scale)
        chemistry_score = (
            0.45 * min(shared_goal_rate / 0.5, 1.0) +    # rate component (cap at 0.5/game)
            0.35 * min(involvement_ratio / 0.6, 1.0) +    # involvement component
            0.20 * consistency                              # consistency component
        )

        name_a = roster.get(pid_a, {}).get("name", str(pid_a))
        name_b = roster.get(pid_b, {}).get("name", str(pid_b))
        team = roster.get(pid_a, {}).get("team_abbrev", "")

        results.append({
            "player_a": name_a,
            "player_b": name_b,
            "player_a_id": pid_a,
            "player_b_id": pid_b,
            "team": team,
            "games_together": games_tog,
            "shared_goals": shared_goals,
            "shared_goal_rate": round(shared_goal_rate, 3),
            "involvement_ratio": round(involvement_ratio, 3),
            "consistency": round(consistency, 3),
            "chemistry_score": round(chemistry_score, 3),
        })

    results.sort(key=lambda x: x["chemistry_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Boost Generation (integrates with projections.py)
# ---------------------------------------------------------------------------

def get_linemate_boosts(
    player_names: list[str],
    team_abbrevs: list[str],
    n_games: int = DEFAULT_GAME_WINDOW,
) -> dict[str, float]:
    """
    Main integration point for projections pipeline.

    Given a list of player names and their teams from tonight's slate,
    returns a dict of {player_name: boost_multiplier} based on linemate
    chemistry with other players on the slate.

    Example return: {"Connor McDavid": 1.08, "Leon Draisaitl": 1.06, ...}
    Players with no significant chemistry signal return 1.0 (no boost).

    Args:
        player_names: list of player names from DK salary file
        team_abbrevs: corresponding team abbreviations
        n_games: how many recent games to analyze per team

    Returns:
        dict mapping player_name -> boost multiplier (1.0 = no change)
    """
    # Group players by team
    team_players = defaultdict(list)
    for name, team in zip(player_names, team_abbrevs):
        team_players[team].append(name)

    boosts = {name: 1.0 for name in player_names}

    # Only analyze teams with 2+ players on slate (need pairs)
    for team, names in team_players.items():
        if len(names) < 2:
            continue

        print(f"  📊 Analyzing linemate chemistry for {team} ({len(names)} players)...")

        # Get recent games
        game_ids = get_team_recent_game_ids(team, n_games)
        if not game_ids:
            print(f"    ⚠ No recent games found for {team}")
            continue

        print(f"    Found {len(game_ids)} recent games")

        # Build pair stats
        pair_stats = build_pair_stats(game_ids, team_abbrev=team)
        chemistry = calculate_chemistry_scores(pair_stats)

        if not chemistry:
            continue

        # Build name -> best chemistry score mapping
        # A player's boost is based on their best pairing with another
        # player on the DFS slate
        name_lower_set = {n.lower() for n in names}
        roster = pair_stats["roster_lookup"]

        # Map roster names to DK names (fuzzy match by last name)
        dk_name_map = {}  # roster_name -> dk_name
        for pid, info in roster.items():
            r_name = info.get("name", "")
            r_last = r_name.split()[-1].lower() if r_name else ""
            for dk_name in names:
                dk_last = dk_name.split()[-1].lower() if dk_name else ""
                if r_last and dk_last and r_last == dk_last:
                    # Check first initial too to reduce false matches
                    r_first = r_name.split()[0][0].lower() if r_name else ""
                    dk_first = dk_name.split()[0][0].lower() if dk_name else ""
                    if r_first == dk_first:
                        dk_name_map[r_name] = dk_name

        # Apply boosts from chemistry scores
        for pair in chemistry:
            a_dk = dk_name_map.get(pair["player_a"])
            b_dk = dk_name_map.get(pair["player_b"])

            # Both players must be on the slate to apply boost
            if not a_dk or not b_dk:
                continue

            score = pair["chemistry_score"]

            # Determine boost tier
            boost_val = 0.0
            for tier_name, tier in sorted(BOOST_TIERS.items(),
                                           key=lambda x: x[1]["threshold"],
                                           reverse=True):
                if score >= tier["threshold"]:
                    boost_val = tier["boost"]
                    break

            if boost_val > 0:
                # Apply the best boost each player has (don't stack)
                for dk_name in (a_dk, b_dk):
                    current = boosts[dk_name] - 1.0
                    if boost_val > current:
                        boosts[dk_name] = 1.0 + boost_val
                        print(f"    ✅ {dk_name}: +{boost_val*100:.0f}% "
                              f"(chemistry={score:.3f} with "
                              f"{b_dk if dk_name == a_dk else a_dk})")

    return boosts


# ---------------------------------------------------------------------------
# Standalone Report
# ---------------------------------------------------------------------------

def print_team_report(team_abbrev: str, n_games: int = DEFAULT_GAME_WINDOW):
    """Print a full linemate correlation report for a team."""
    print(f"\n{'='*60}")
    print(f"  LINEMATE CHEMISTRY REPORT: {team_abbrev}")
    print(f"  Last {n_games} games")
    print(f"{'='*60}\n")

    game_ids = get_team_recent_game_ids(team_abbrev, n_games)
    print(f"Games analyzed: {len(game_ids)}")
    print(f"Game IDs: {game_ids[:5]}{'...' if len(game_ids) > 5 else ''}\n")

    pair_stats = build_pair_stats(game_ids, team_abbrev=team_abbrev)
    chemistry = calculate_chemistry_scores(pair_stats)

    if not chemistry:
        print("No significant pairings found.")
        return

    print(f"{'Pair':<40} {'Games':>5} {'Shared':>6} {'Rate':>6} "
          f"{'Invlv':>6} {'Chem':>6}")
    print("-" * 75)

    for pair in chemistry[:20]:  # top 20 pairs
        name = f"{pair['player_a']} + {pair['player_b']}"
        if len(name) > 38:
            name = name[:38] + ".."
        print(f"{name:<40} {pair['games_together']:>5} "
              f"{pair['shared_goals']:>6} {pair['shared_goal_rate']:>6.3f} "
              f"{pair['involvement_ratio']:>6.3f} {pair['chemistry_score']:>6.3f}")

    # Top pairings summary
    elite = [p for p in chemistry if p["chemistry_score"] >= BOOST_TIERS["elite"]["threshold"]]
    above = [p for p in chemistry if BOOST_TIERS["above_avg"]["threshold"] <= p["chemistry_score"] < BOOST_TIERS["elite"]["threshold"]]

    print(f"\n🔥 Elite chemistry pairs: {len(elite)}")
    for p in elite:
        print(f"   {p['player_a']} + {p['player_b']} ({p['chemistry_score']:.3f})")

    print(f"✅ Above-average pairs: {len(above)}")
    for p in above[:5]:
        print(f"   {p['player_a']} + {p['player_b']} ({p['chemistry_score']:.3f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        teams = sys.argv[1:]
    else:
        # Default: a few popular teams for demo
        teams = ["EDM", "FLA", "COL"]

    n_games = DEFAULT_GAME_WINDOW

    for team in teams:
        try:
            print_team_report(team.upper(), n_games)
        except Exception as e:
            print(f"Error analyzing {team}: {e}")

    print(f"\n{'='*60}")
    print("  Integration: from linemate_corr import get_linemate_boosts")
    print("  Then apply boosts to FPTS projections in projections.py")
    print(f"{'='*60}")
