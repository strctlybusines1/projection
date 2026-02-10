#!/usr/bin/env python3
"""
linemate_corr.py — Linemate Correlation Analysis for NHL DFS
=============================================================

Two approaches:
1. DB-BASED (fast, large sample): Calculate FPTS correlations from game logs
   already in the database. Uses full season data (30-50 games per team).
2. API-BASED (original): Parse play-by-play for shared goal involvement.
   Falls back to this when DB data is insufficient.

The DB approach is preferred because:
- 30-50 game samples vs 10 game API window
- No API calls needed (instant)
- Correlates actual DK FPTS (what we care about) not just goals
- Captures shots, blocks, assists — everything that moves FPTS together

Usage:
    python linemate_corr.py COL EDM TBL     # Report for specific teams
    python linemate_corr.py --all             # All teams with data
    
    from linemate_corr import get_linemate_boosts
    boosts = get_linemate_boosts(player_names, team_abbrevs)
"""

import sqlite3
import requests
import time
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"
API_URL = "https://api-web.nhle.com/v1/"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".linemate_cache")

DEFAULT_GAME_WINDOW = 10
MIN_SHARED_GAMES = 10
MIN_CORRELATION = 0.15

BOOST_TIERS = {
    "elite":     {"threshold": 0.45, "boost": 0.06},
    "above_avg": {"threshold": 0.30, "boost": 0.03},
    "average":   {"threshold": 0.15, "boost": 0.00},
}

REQUEST_DELAY = 0.4


class LinemateCorrelationDB:
    """Calculate linemate FPTS correlations from game log database."""

    def __init__(self, db_path=None):
        self.db_path = db_path or str(DB_PATH)
        self.team_correlations = {}
        self.fitted = False

    def fit(self, teams=None):
        if not Path(self.db_path).exists():
            return self
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT player_id, player_name, team, position, game_id, game_date,
                   dk_fpts, goals, assists, shots, pp_points
            FROM game_logs_skaters
        """
        if teams:
            placeholders = ','.join('?' for _ in teams)
            query += f" WHERE team IN ({placeholders})"
            sk = pd.read_sql_query(query, conn, params=teams)
        else:
            sk = pd.read_sql_query(query, conn)
        conn.close()

        if sk.empty:
            return self
        for team, group in sk.groupby('team'):
            self._build_team_correlations(team, group)
        self.fitted = True
        return self

    def _build_team_correlations(self, team, team_logs):
        pivot = team_logs.pivot_table(index='game_id', columns='player_id', values='dk_fpts')
        if pivot.shape[1] < 2:
            return

        player_info = {}
        for _, row in team_logs.drop_duplicates('player_id').iterrows():
            player_info[row['player_id']] = {
                'name': row['player_name'], 'position': row['position'], 'team': row['team'],
            }

        pairs = []
        player_ids = list(pivot.columns)
        for i, pid_a in enumerate(player_ids):
            for pid_b in player_ids[i+1:]:
                shared = pivot[[pid_a, pid_b]].dropna()
                n_shared = len(shared)
                if n_shared < MIN_SHARED_GAMES:
                    continue
                corr = shared[pid_a].corr(shared[pid_b])
                if np.isnan(corr):
                    continue

                a_mean, b_mean = shared[pid_a].mean(), shared[pid_b].mean()
                both_above = ((shared[pid_a] > a_mean) & (shared[pid_b] > b_mean)).mean()

                above_mask = (shared[pid_a] > a_mean) | (shared[pid_b] > b_mean)
                upside_corr = shared.loc[above_mask, pid_a].corr(shared.loc[above_mask, pid_b]) if above_mask.sum() > 5 else corr

                info_a = player_info.get(pid_a, {})
                info_b = player_info.get(pid_b, {})
                pairs.append({
                    'player_a': info_a.get('name', str(pid_a)),
                    'player_b': info_b.get('name', str(pid_b)),
                    'player_a_id': pid_a, 'player_b_id': pid_b,
                    'pos_a': info_a.get('position', '?'),
                    'pos_b': info_b.get('position', '?'),
                    'team': team,
                    'correlation': round(corr, 3),
                    'upside_corr': round(upside_corr, 3) if not np.isnan(upside_corr) else round(corr, 3),
                    'games_together': n_shared,
                    'avg_fpts_a': round(a_mean, 1), 'avg_fpts_b': round(b_mean, 1),
                    'both_above_pct': round(both_above, 3),
                    'combined_avg': round(a_mean + b_mean, 1),
                })

        pairs.sort(key=lambda x: x['correlation'], reverse=True)
        self.team_correlations[team] = pairs

    def get_top_pairs(self, team, n=15):
        return self.team_correlations.get(team, [])[:n]

    def get_pair_correlation(self, player_a_name, player_b_name, team=None):
        teams = [team] if team else list(self.team_correlations.keys())
        for t in teams:
            for pair in self.team_correlations.get(t, []):
                a_last = player_a_name.split()[-1].lower()
                b_last = player_b_name.split()[-1].lower()
                pa_last = pair['player_a'].split()[-1].lower()
                pb_last = pair['player_b'].split()[-1].lower()
                if (a_last == pa_last and b_last == pb_last) or \
                   (a_last == pb_last and b_last == pa_last):
                    return pair
        return None

    def print_report(self, team):
        pairs = self.team_correlations.get(team, [])
        if not pairs:
            print(f"  No correlation data for {team}")
            return
        n_games = pairs[0]['games_together'] if pairs else 0
        print(f"\n  -- Linemate Correlations: {team} ({n_games}+ shared games) --")
        print(f"  {'Pair':<40} {'Corr':>5} {'Up':>5} {'GP':>4} {'AvgA':>5} {'AvgB':>5} {'Both':>5}")
        print(f"  {'-' * 70}")
        for p in pairs[:20]:
            name = f"{p['player_a'][:18]} + {p['player_b'][:18]}"
            tier = ''
            if p['correlation'] >= BOOST_TIERS['elite']['threshold']:
                tier = ' ELITE'
            elif p['correlation'] >= BOOST_TIERS['above_avg']['threshold']:
                tier = ' GOOD'
            print(f"  {name:<40} {p['correlation']:>5.3f} {p['upside_corr']:>5.3f} "
                  f"{p['games_together']:>4} {p['avg_fpts_a']:>5.1f} {p['avg_fpts_b']:>5.1f} "
                  f"{p['both_above_pct']:>5.1%}{tier}")


# ================================================================
#  API Fallback
# ================================================================

def _get(endpoint, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = endpoint.replace("/", "_").replace("?", "_")
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if use_cache and os.path.exists(cache_path):
        if time.time() - os.path.getmtime(cache_path) < 7200:
            with open(cache_path, "r") as f:
                return json.load(f)
    time.sleep(REQUEST_DELAY)
    resp = requests.get(API_URL + endpoint)
    resp.raise_for_status()
    data = resp.json()
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def get_team_recent_game_ids(team_abbrev, n_games=DEFAULT_GAME_WINDOW):
    today = datetime.now()
    game_ids = []
    for weeks_back in range(0, 12):
        date = today - timedelta(weeks=weeks_back)
        try:
            data = _get(f"club-schedule/{team_abbrev}/week/{date.strftime('%Y-%m-%d')}")
        except Exception:
            continue
        for game in data.get("games", []):
            if game.get("gameState") in ("OFF", "FINAL") and game["id"] not in game_ids:
                game_ids.append(game["id"])
        if len(game_ids) >= n_games:
            break
    return sorted(set(game_ids), reverse=True)[:n_games]


def parse_game_events(game_id):
    pbp = _get(f"gamecenter/{game_id}/play-by-play")
    roster = {}
    for spot in pbp.get("rosterSpots", []):
        pid = spot.get("playerId")
        if pid:
            first = spot.get("firstName", {}).get("default", "")
            last = spot.get("lastName", {}).get("default", "")
            roster[pid] = {"name": f"{first} {last}".strip(),
                          "team": spot.get("teamId"), "pos": spot.get("positionCode", "")}
    home_id = pbp.get("homeTeam", {}).get("id")
    away_id = pbp.get("awayTeam", {}).get("id")
    team_map = {home_id: pbp.get("homeTeam", {}).get("abbrev", ""),
                away_id: pbp.get("awayTeam", {}).get("abbrev", "")}
    for pid in roster:
        roster[pid]["team_abbrev"] = team_map.get(roster[pid]["team"], "")
    goal_events = []
    for play in pbp.get("plays", []):
        if play.get("typeDescKey") == "goal":
            d = play.get("details", {})
            goal_events.append({"scorer_id": d.get("scoringPlayerId"),
                               "assist1_id": d.get("assist1PlayerId"),
                               "assist2_id": d.get("assist2PlayerId")})
    return {"game_id": game_id, "roster": roster, "goal_events": goal_events}


# ================================================================
#  Boost Generation
# ================================================================

def get_linemate_boosts(player_names, team_abbrevs, n_games=DEFAULT_GAME_WINDOW):
    """Main integration point. Returns {player_name: boost_multiplier}."""
    team_players = defaultdict(list)
    for name, team in zip(player_names, team_abbrevs):
        team_players[team].append(name)

    boosts = {name: 1.0 for name in player_names}
    unique_teams = list(team_players.keys())

    db_corr = LinemateCorrelationDB()
    db_corr.fit(teams=unique_teams)

    for team, names in team_players.items():
        if len(names) < 2:
            continue

        pairs = db_corr.team_correlations.get(team, [])
        if not pairs:
            print(f"  No DB data for {team}, using API fallback...")
            _apply_api_boosts(team, names, boosts, n_games)
            continue

        name_map = {}
        for pair in pairs:
            for key in ('player_a', 'player_b'):
                db_name = pair[key]
                db_last = db_name.split()[-1].lower()
                db_first = db_name.split()[0][0].lower() if db_name else ''
                for dk_name in names:
                    dk_last = dk_name.split()[-1].lower()
                    dk_first = dk_name.split()[0][0].lower() if dk_name else ''
                    if db_last == dk_last and db_first == dk_first:
                        name_map[db_name] = dk_name

        for pair in pairs:
            a_dk = name_map.get(pair['player_a'])
            b_dk = name_map.get(pair['player_b'])
            if not a_dk or not b_dk:
                continue
            corr = pair['correlation']
            boost_val = 0.0
            for _, tier in sorted(BOOST_TIERS.items(), key=lambda x: x[1]["threshold"], reverse=True):
                if corr >= tier["threshold"]:
                    boost_val = tier["boost"]
                    break
            if boost_val > 0:
                for dk_name in (a_dk, b_dk):
                    if boost_val > boosts[dk_name] - 1.0:
                        boosts[dk_name] = 1.0 + boost_val
                        partner = b_dk if dk_name == a_dk else a_dk
                        print(f"    {dk_name}: +{boost_val*100:.0f}% (corr={corr:.3f} w/ {partner}, {pair['games_together']}gp)")

    return boosts


def _apply_api_boosts(team, names, boosts, n_games):
    try:
        game_ids = get_team_recent_game_ids(team, n_games)
        if not game_ids:
            return
        goal_pairs = defaultdict(int)
        roster_names = {}
        for gid in game_ids:
            try:
                parsed = parse_game_events(gid)
            except Exception:
                continue
            for pid, info in parsed['roster'].items():
                if info.get('team_abbrev') == team:
                    roster_names[pid] = info['name']
            for goal in parsed['goal_events']:
                involved = [p for p in [goal['scorer_id'], goal['assist1_id'], goal['assist2_id']]
                           if p and p in roster_names]
                for a, b in combinations(sorted(involved), 2):
                    goal_pairs[(a, b)] += 1

        for (a, b), shared in sorted(goal_pairs.items(), key=lambda x: x[1], reverse=True):
            if shared < 2:
                continue
            rate = shared / len(game_ids)
            if rate >= 0.3:
                boost_val = BOOST_TIERS['elite']['boost']
            elif rate >= 0.15:
                boost_val = BOOST_TIERS['above_avg']['boost']
            else:
                continue
            for pid_name in (roster_names.get(a, ''), roster_names.get(b, '')):
                for dk_name in names:
                    if pid_name.split()[-1].lower() == dk_name.split()[-1].lower():
                        if boost_val > boosts[dk_name] - 1.0:
                            boosts[dk_name] = 1.0 + boost_val
    except Exception as e:
        print(f"    API fallback failed for {team}: {e}")


if __name__ == "__main__":
    import sys
    teams = []
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            db = LinemateCorrelationDB()
            db.fit()
            teams = sorted(db.team_correlations.keys())
        else:
            teams = [t.upper() for t in sys.argv[1:]]
            db = LinemateCorrelationDB()
            db.fit(teams=teams)
    else:
        teams = ["EDM", "COL", "TBL"]
        db = LinemateCorrelationDB()
        db.fit(teams=teams)

    for team in teams:
        db.print_report(team)

    print(f"\n  -- Summary --")
    for team in teams:
        pairs = db.team_correlations.get(team, [])
        elite = len([p for p in pairs if p['correlation'] >= BOOST_TIERS['elite']['threshold']])
        above = len([p for p in pairs if BOOST_TIERS['above_avg']['threshold'] <= p['correlation'] < BOOST_TIERS['elite']['threshold']])
        if elite or above:
            print(f"  {team}: {elite} elite, {above} above-avg pairs")
