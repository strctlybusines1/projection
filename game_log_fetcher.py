#!/usr/bin/env python3
"""
NHL Game Log Fetcher & Analyzer
=================================

Fetches per-player game logs from the NHL API, stores them in SQLite,
and provides analysis hooks for Regime Switching, Change Point Detection,
PELT, and Binary Segmentation models.

Data volume:
    ~32 active teams × ~23 players/roster = ~740 players
    ~740 API calls at 0.3s rate limit = ~4 minutes total
    Can batch by team count if needed (--teams-per-batch 8)

Storage:
    SQLite database: data/nhl_dfs_history.db (same as history_db.py)
    Tables: game_logs_skaters, game_logs_goalies, roster_cache

Usage:
    # Fetch ALL game logs for every rostered player (full season)
    python game_log_fetcher.py fetch --all

    # Fetch specific teams only
    python game_log_fetcher.py fetch --teams COL TBL NYR

    # Fetch in batches (e.g., 8 teams per run, for rate limit safety)
    python game_log_fetcher.py fetch --batch 1 --teams-per-batch 8

    # Update only (skip players already fetched today)
    python game_log_fetcher.py fetch --all --update-only

    # Run analysis (regime switching + change point)
    python game_log_fetcher.py analyze

    # Export game logs to CSV
    python game_log_fetcher.py export --output game_logs_export.csv

    # Show status
    python game_log_fetcher.py status
"""

import os
import sys
import time
import json
import sqlite3
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Project imports
from nhl_api import NHLAPIClient
from config import CURRENT_SEASON, NHL_TEAMS

# ================================================================
#  Database Setup
# ================================================================

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "nhl_dfs_history.db"


def get_db() -> sqlite3.Connection:
    """Get database connection and ensure game log tables exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    _create_game_log_tables(conn)
    return conn


def _create_game_log_tables(conn: sqlite3.Connection):
    """Create game log tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS game_logs_skaters (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id       INTEGER NOT NULL,
            player_name     TEXT NOT NULL,
            team            TEXT NOT NULL,
            position        TEXT,
            game_id         INTEGER NOT NULL,
            game_date       TEXT NOT NULL,
            opponent        TEXT,
            home_road       TEXT,
            goals           INTEGER DEFAULT 0,
            assists         INTEGER DEFAULT 0,
            points          INTEGER DEFAULT 0,
            plus_minus      INTEGER DEFAULT 0,
            shots           INTEGER DEFAULT 0,
            pim             INTEGER DEFAULT 0,
            pp_goals        INTEGER DEFAULT 0,
            pp_points       INTEGER DEFAULT 0,
            sh_goals        INTEGER DEFAULT 0,
            sh_points       INTEGER DEFAULT 0,
            gw_goals        INTEGER DEFAULT 0,
            ot_goals        INTEGER DEFAULT 0,
            shifts          INTEGER DEFAULT 0,
            toi             TEXT,
            toi_seconds     INTEGER DEFAULT 0,
            dk_fpts         REAL DEFAULT 0,
            fetched_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(player_id, game_id)
        );

        CREATE TABLE IF NOT EXISTS game_logs_goalies (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id       INTEGER NOT NULL,
            player_name     TEXT NOT NULL,
            team            TEXT NOT NULL,
            game_id         INTEGER NOT NULL,
            game_date       TEXT NOT NULL,
            opponent        TEXT,
            home_road       TEXT,
            games_started   INTEGER DEFAULT 0,
            decision        TEXT,
            shots_against   INTEGER DEFAULT 0,
            goals_against   INTEGER DEFAULT 0,
            saves           INTEGER DEFAULT 0,
            save_pct        REAL DEFAULT 0,
            shutouts        INTEGER DEFAULT 0,
            goals           INTEGER DEFAULT 0,
            assists         INTEGER DEFAULT 0,
            pim             INTEGER DEFAULT 0,
            toi             TEXT,
            toi_seconds     INTEGER DEFAULT 0,
            dk_fpts         REAL DEFAULT 0,
            fetched_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(player_id, game_id)
        );

        CREATE TABLE IF NOT EXISTS roster_cache (
            player_id       INTEGER NOT NULL,
            player_name     TEXT NOT NULL,
            team            TEXT NOT NULL,
            position        TEXT,
            position_code   TEXT,
            fetched_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(player_id, team)
        );

        CREATE INDEX IF NOT EXISTS idx_skater_gl_player
            ON game_logs_skaters(player_id, game_date);
        CREATE INDEX IF NOT EXISTS idx_skater_gl_date
            ON game_logs_skaters(game_date);
        CREATE INDEX IF NOT EXISTS idx_skater_gl_team
            ON game_logs_skaters(team, game_date);

        CREATE INDEX IF NOT EXISTS idx_goalie_gl_player
            ON game_logs_goalies(player_id, game_date);
        CREATE INDEX IF NOT EXISTS idx_goalie_gl_date
            ON game_logs_goalies(game_date);
    """)
    conn.commit()


# ================================================================
#  DraftKings FPTS Calculation
# ================================================================

def calc_skater_dk_fpts(g: Dict) -> float:
    """Calculate DraftKings fantasy points for a skater game."""
    goals = g.get('goals', 0)
    assists = g.get('assists', 0)
    shots = g.get('shots', 0)
    pim = g.get('pim', 0)  # not used in DK but useful
    pp_points = g.get('pp_points', 0) or g.get('powerPlayPoints', 0)
    sh_goals = g.get('sh_goals', 0) or g.get('shorthandedGoals', 0)
    sh_points = g.get('sh_points', 0) or g.get('shorthandedPoints', 0)
    gw_goals = g.get('gw_goals', 0) or g.get('gameWinningGoals', 0)

    # Blocks not in game log API — estimated as 0
    blocks = g.get('blocks', 0)

    fpts = (
        goals * 8.5 +
        assists * 5.0 +
        shots * 1.5 +
        blocks * 1.3 +
        sh_goals * 2.0 +       # SH goal bonus (+2 on top of goal pts)
        (sh_points - sh_goals) * 2.0 +  # SH assist bonus
        gw_goals * 0.0          # No DK bonus for GWG currently
    )

    # Hat trick bonus
    if goals >= 3:
        fpts += 3.0

    # 3+ points bonus
    if (goals + assists) >= 3:
        fpts += 3.0

    # 5+ SOG bonus
    if shots >= 5:
        fpts += 3.0

    # 3+ blocks bonus
    if blocks >= 3:
        fpts += 3.0

    return round(fpts, 1)


def calc_goalie_dk_fpts(g: Dict) -> float:
    """Calculate DraftKings fantasy points for a goalie game."""
    decision = g.get('decision', '')
    saves = g.get('saves', 0)
    goals_against = g.get('goals_against', 0) or g.get('goalsAgainst', 0)
    shutouts = g.get('shutouts', 0)

    win = 1 if decision == 'W' else 0
    otl = 1 if decision == 'O' else 0

    fpts = (
        win * 6.0 +
        saves * 0.7 +
        goals_against * -3.5 +
        shutouts * 4.0 +
        otl * 2.0
    )

    # 35+ saves bonus
    if saves >= 35:
        fpts += 3.0

    return round(fpts, 1)


def parse_toi_seconds(toi_str: str) -> int:
    """Parse '23:01' format to total seconds."""
    if not toi_str or ':' not in str(toi_str):
        return 0
    parts = str(toi_str).split(':')
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 0


# ================================================================
#  Fetcher
# ================================================================

class GameLogFetcher:
    """Fetch and store game logs for all NHL players."""

    def __init__(self, rate_limit: float = 0.35):
        self.api = NHLAPIClient(rate_limit_delay=rate_limit)
        self.conn = get_db()
        self.stats = {'rosters_fetched': 0, 'game_logs_fetched': 0,
                      'skater_games': 0, 'goalie_games': 0,
                      'skipped': 0, 'errors': 0}

    def fetch_teams(self, teams: List[str], update_only: bool = False):
        """
        Fetch game logs for all players on the given teams.

        Args:
            teams: List of team abbreviations (e.g., ['COL', 'TBL'])
            update_only: If True, skip players fetched in the last 24 hours
        """
        print(f"\n{'=' * 60}")
        print(f"  NHL Game Log Fetcher")
        print(f"  Teams: {len(teams)} | Season: {CURRENT_SEASON}")
        print(f"  Update only: {update_only}")
        print(f"{'=' * 60}\n")

        for i, team in enumerate(teams):
            print(f"  [{i+1}/{len(teams)}] {team}...", end='', flush=True)

            try:
                roster = self.api.get_roster(team)
                self.stats['rosters_fetched'] += 1
            except Exception as e:
                print(f" ⚠ roster error: {e}")
                self.stats['errors'] += 1
                continue

            players = []
            for group, pos_type in [('forwards', 'F'), ('defensemen', 'D'), ('goalies', 'G')]:
                for p in roster.get(group, []):
                    pid = p.get('id')
                    first = p.get('firstName', {}).get('default', '')
                    last = p.get('lastName', {}).get('default', '')
                    name = f"{first} {last}".strip()
                    pos_code = p.get('positionCode', pos_type)
                    players.append({
                        'id': pid, 'name': name, 'team': team,
                        'pos_type': pos_type, 'pos_code': pos_code,
                    })

            # Cache roster
            for pl in players:
                self.conn.execute("""
                    INSERT OR REPLACE INTO roster_cache
                    (player_id, player_name, team, position, position_code, fetched_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                """, (pl['id'], pl['name'], pl['team'], pl['pos_type'], pl['pos_code']))
            self.conn.commit()

            # Fetch game logs
            skaters_done = 0
            goalies_done = 0

            for pl in players:
                if update_only and self._recently_fetched(pl['id']):
                    self.stats['skipped'] += 1
                    continue

                try:
                    gl = self.api.get_player_game_log(pl['id'])
                    games = gl.get('gameLog', [])
                    self.stats['game_logs_fetched'] += 1

                    if pl['pos_type'] == 'G':
                        self._store_goalie_games(pl, games)
                        goalies_done += 1
                        self.stats['goalie_games'] += len(games)
                    else:
                        self._store_skater_games(pl, games)
                        skaters_done += 1
                        self.stats['skater_games'] += len(games)

                except Exception as e:
                    self.stats['errors'] += 1
                    # Don't print individual errors to keep output clean
                    continue

            print(f" {skaters_done} skaters, {goalies_done} goalies")

        self.conn.commit()
        self._print_summary()

    def _store_skater_games(self, player: Dict, games: List[Dict]):
        """Store skater game log entries."""
        for g in games:
            toi_str = g.get('toi', '0:00')
            toi_sec = parse_toi_seconds(toi_str)

            row = {
                'goals': g.get('goals', 0),
                'assists': g.get('assists', 0),
                'points': g.get('points', 0),
                'plus_minus': g.get('plusMinus', 0),
                'shots': g.get('shots', 0),
                'pim': g.get('pim', 0),
                'pp_goals': g.get('powerPlayGoals', 0),
                'pp_points': g.get('powerPlayPoints', 0),
                'sh_goals': g.get('shorthandedGoals', 0),
                'sh_points': g.get('shorthandedPoints', 0),
                'gw_goals': g.get('gameWinningGoals', 0),
                'ot_goals': g.get('otGoals', 0),
                'shifts': g.get('shifts', 0),
                'blocks': 0,  # Not in game log API
            }

            dk_fpts = calc_skater_dk_fpts(row)

            self.conn.execute("""
                INSERT OR REPLACE INTO game_logs_skaters
                (player_id, player_name, team, position, game_id, game_date,
                 opponent, home_road, goals, assists, points, plus_minus,
                 shots, pim, pp_goals, pp_points, sh_goals, sh_points,
                 gw_goals, ot_goals, shifts, toi, toi_seconds, dk_fpts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player['id'], player['name'], player['team'], player['pos_code'],
                g.get('gameId', 0), g.get('gameDate', ''),
                g.get('opponentAbbrev', ''), g.get('homeRoadFlag', ''),
                row['goals'], row['assists'], row['points'], row['plus_minus'],
                row['shots'], row['pim'], row['pp_goals'], row['pp_points'],
                row['sh_goals'], row['sh_points'], row['gw_goals'], row['ot_goals'],
                row['shifts'], toi_str, toi_sec, dk_fpts,
            ))

    def _store_goalie_games(self, player: Dict, games: List[Dict]):
        """Store goalie game log entries."""
        for g in games:
            toi_str = g.get('toi', '0:00')
            toi_sec = parse_toi_seconds(toi_str)
            shots_against = g.get('shotsAgainst', 0)
            goals_against = g.get('goalsAgainst', 0)
            saves = shots_against - goals_against

            row = {
                'decision': g.get('decision', ''),
                'saves': saves,
                'goals_against': goals_against,
                'shutouts': g.get('shutouts', 0),
            }

            dk_fpts = calc_goalie_dk_fpts(row)

            self.conn.execute("""
                INSERT OR REPLACE INTO game_logs_goalies
                (player_id, player_name, team, game_id, game_date,
                 opponent, home_road, games_started, decision,
                 shots_against, goals_against, saves, save_pct,
                 shutouts, goals, assists, pim, toi, toi_seconds, dk_fpts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player['id'], player['name'], player['team'],
                g.get('gameId', 0), g.get('gameDate', ''),
                g.get('opponentAbbrev', ''), g.get('homeRoadFlag', ''),
                g.get('gamesStarted', 0), g.get('decision', ''),
                shots_against, goals_against, saves,
                g.get('savePctg', 0), g.get('shutouts', 0),
                g.get('goals', 0), g.get('assists', 0), g.get('pim', 0),
                toi_str, toi_sec, dk_fpts,
            ))

    def _recently_fetched(self, player_id: int, hours: int = 24) -> bool:
        """Check if this player was fetched recently."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        row = self.conn.execute("""
            SELECT 1 FROM game_logs_skaters
            WHERE player_id = ? AND fetched_at > ?
            UNION
            SELECT 1 FROM game_logs_goalies
            WHERE player_id = ? AND fetched_at > ?
            LIMIT 1
        """, (player_id, cutoff, player_id, cutoff)).fetchone()
        return row is not None

    def _print_summary(self):
        """Print fetch summary."""
        s = self.stats
        print(f"\n  ── Fetch Summary ──────────────────")
        print(f"  Rosters fetched:  {s['rosters_fetched']}")
        print(f"  Game logs fetched: {s['game_logs_fetched']}")
        print(f"  Skater games:     {s['skater_games']}")
        print(f"  Goalie games:     {s['goalie_games']}")
        print(f"  Skipped (recent): {s['skipped']}")
        print(f"  Errors:           {s['errors']}")

        # DB totals
        sk_count = self.conn.execute("SELECT COUNT(*) FROM game_logs_skaters").fetchone()[0]
        g_count = self.conn.execute("SELECT COUNT(*) FROM game_logs_goalies").fetchone()[0]
        sk_players = self.conn.execute("SELECT COUNT(DISTINCT player_id) FROM game_logs_skaters").fetchone()[0]
        g_players = self.conn.execute("SELECT COUNT(DISTINCT player_id) FROM game_logs_goalies").fetchone()[0]
        print(f"\n  Database totals:")
        print(f"    Skater game logs: {sk_count:,} ({sk_players} players)")
        print(f"    Goalie game logs: {g_count:,} ({g_players} players)")


# ================================================================
#  Analysis (Regime Switching + Change Point)
# ================================================================

def analyze_game_logs(min_games: int = 10, top_n: int = 20):
    """
    Run Regime Switching and Change Point Detection on stored game logs.
    Uses PELT and Binary Segmentation from ruptures library.
    """
    conn = get_db()

    # Load all skater FPTS sequences
    sk_df = pd.read_sql_query("""
        SELECT player_id, player_name, team, game_date, dk_fpts,
               goals, assists, shots, toi_seconds
        FROM game_logs_skaters
        ORDER BY player_id, game_date
    """, conn)

    g_df = pd.read_sql_query("""
        SELECT player_id, player_name, team, game_date, dk_fpts,
               saves, goals_against, decision, toi_seconds
        FROM game_logs_goalies
        WHERE games_started = 1 OR toi_seconds > 1800
        ORDER BY player_id, game_date
    """, conn)

    print(f"{'=' * 72}")
    print(f"  GAME LOG ANALYSIS")
    print(f"{'=' * 72}")
    print(f"  Skaters: {sk_df['player_id'].nunique()} players, {len(sk_df)} game logs")
    print(f"  Goalies: {g_df['player_id'].nunique()} players, {len(g_df)} game logs")

    # ── Regime Switching (Markov) ──
    print(f"\n{'─' * 50}")
    print(f"  REGIME SWITCHING MODEL")

    try:
        from advanced_models import MarkovSwitchingModel

        for label, df in [("SKATERS", sk_df), ("GOALIES", g_df)]:
            sequences = []
            player_info = []
            for pid, group in df.groupby('player_id'):
                fpts = group.sort_values('game_date')['dk_fpts'].values
                if len(fpts) >= min_games:
                    sequences.append(fpts)
                    player_info.append({
                        'id': pid,
                        'name': group.iloc[0]['player_name'],
                        'team': group.iloc[0]['team'],
                        'games': len(fpts),
                    })

            if not sequences:
                print(f"\n  {label}: No players with {min_games}+ games")
                continue

            msm = MarkovSwitchingModel(n_regimes=2)
            msm.fit(sequences)

            print(f"\n  {label} ({len(sequences)} players, {min_games}+ games):")
            for i, rp in enumerate(msm.regime_params):
                print(f"    Regime {i}: mean={rp['mean']:.1f} FPTS, std={rp['std']:.1f}")
            print(f"    Transition matrix:")
            for i, row in enumerate(msm.transition_probs):
                print(f"      {['Cold','Hot'][i]}: stay={row[i]:.2f}, switch={row[1-i]:.2f}")

            # Find players currently in each regime
            hot_players = []
            cold_players = []
            for seq, info in zip(sequences, player_info):
                result = msm.predict_regime(seq)
                if result['regime'] == 'hot':
                    hot_players.append((info, result, seq[-5:].mean()))
                else:
                    cold_players.append((info, result, seq[-5:].mean()))

            hot_players.sort(key=lambda x: x[2], reverse=True)
            cold_players.sort(key=lambda x: x[2])

            print(f"\n    HOT regime ({len(hot_players)} players):")
            for info, result, avg in hot_players[:top_n]:
                print(f"      {info['name']:<25} ({info['team']}) "
                      f"last5={avg:.1f}  persist={result['persistence']:.2f}")

            print(f"\n    COLD regime ({len(cold_players)} players):")
            for info, result, avg in cold_players[:top_n]:
                print(f"      {info['name']:<25} ({info['team']}) "
                      f"last5={avg:.1f}  persist={result['persistence']:.2f}")

    except ImportError as e:
        print(f"  ⚠ {e}")

    # ── Change Point Detection (PELT + BinSeg) ──
    print(f"\n{'─' * 50}")
    print(f"  CHANGE POINT DETECTION (PELT + Binary Segmentation)")

    try:
        import ruptures as rpt
    except ImportError:
        print("  ⚠ Install: pip install ruptures")
        rpt = None

    if rpt:
        for label, df in [("SKATERS", sk_df), ("GOALIES", g_df)]:
            breakouts = []
            breakdowns = []

            for pid, group in df.groupby('player_id'):
                fpts = group.sort_values('game_date')['dk_fpts'].values
                if len(fpts) < min_games:
                    continue

                info = {
                    'id': pid,
                    'name': group.iloc[0]['player_name'],
                    'team': group.iloc[0]['team'],
                    'games': len(fpts),
                }

                signal = fpts.reshape(-1, 1)

                # PELT (Pruned Exact Linear Time)
                try:
                    pelt = rpt.Pelt(model='rbf', min_size=3).fit(signal)
                    pelt_bkps = pelt.predict(pen=3.0)
                except Exception:
                    pelt_bkps = []

                # Binary Segmentation
                try:
                    binseg = rpt.Binseg(model='l2', min_size=3).fit(signal)
                    binseg_bkps = binseg.predict(n_bkps=min(3, len(fpts) // 5))
                except Exception:
                    binseg_bkps = []

                # Combine change points (consensus)
                all_bkps = set()
                for bp in pelt_bkps:
                    if bp < len(fpts):
                        all_bkps.add(bp)
                for bp in binseg_bkps:
                    if bp < len(fpts):
                        all_bkps.add(bp)

                if not all_bkps:
                    continue

                # Analyze segments
                sorted_bkps = sorted(all_bkps)
                segments = []
                prev = 0
                for bp in sorted_bkps:
                    seg = fpts[prev:bp]
                    if len(seg) > 0:
                        segments.append({'start': prev, 'end': bp,
                                        'mean': float(np.mean(seg)),
                                        'n': len(seg)})
                    prev = bp
                if prev < len(fpts):
                    seg = fpts[prev:]
                    if len(seg) > 0:
                        segments.append({'start': prev, 'end': len(fpts),
                                        'mean': float(np.mean(seg)),
                                        'n': len(seg)})

                if len(segments) >= 2:
                    last = segments[-1]['mean']
                    prev_mean = segments[-2]['mean']
                    delta = last - prev_mean

                    entry = {
                        **info,
                        'n_changes': len(all_bkps),
                        'pelt_changes': len([b for b in pelt_bkps if b < len(fpts)]),
                        'binseg_changes': len([b for b in binseg_bkps if b < len(fpts)]),
                        'prev_mean': prev_mean,
                        'current_mean': last,
                        'delta': delta,
                        'segments': segments,
                    }

                    if delta > 2.0:
                        breakouts.append(entry)
                    elif delta < -2.0:
                        breakdowns.append(entry)

            breakouts.sort(key=lambda x: x['delta'], reverse=True)
            breakdowns.sort(key=lambda x: x['delta'])

            print(f"\n  {label}:")
            print(f"    Breakouts ({len(breakouts)} players trending UP):")
            for p in breakouts[:top_n]:
                print(f"      {p['name']:<25} ({p['team']}) "
                      f"{p['prev_mean']:.1f} → {p['current_mean']:.1f} "
                      f"(+{p['delta']:.1f}) "
                      f"[PELT:{p['pelt_changes']}, BinSeg:{p['binseg_changes']}]")

            print(f"    Breakdowns ({len(breakdowns)} players trending DOWN):")
            for p in breakdowns[:top_n]:
                print(f"      {p['name']:<25} ({p['team']}) "
                      f"{p['prev_mean']:.1f} → {p['current_mean']:.1f} "
                      f"({p['delta']:.1f}) "
                      f"[PELT:{p['pelt_changes']}, BinSeg:{p['binseg_changes']}]")

    conn.close()


# ================================================================
#  Utility Functions
# ================================================================

def show_status():
    """Show database status."""
    conn = get_db()

    sk_count = conn.execute("SELECT COUNT(*) FROM game_logs_skaters").fetchone()[0]
    g_count = conn.execute("SELECT COUNT(*) FROM game_logs_goalies").fetchone()[0]
    sk_players = conn.execute("SELECT COUNT(DISTINCT player_id) FROM game_logs_skaters").fetchone()[0]
    g_players = conn.execute("SELECT COUNT(DISTINCT player_id) FROM game_logs_goalies").fetchone()[0]
    roster_count = conn.execute("SELECT COUNT(*) FROM roster_cache").fetchone()[0]

    print(f"\n  Game Log Database Status")
    print(f"  {'─' * 40}")
    print(f"  Skater game logs: {sk_count:,} ({sk_players} players)")
    print(f"  Goalie game logs: {g_count:,} ({g_players} players)")
    print(f"  Roster cache:     {roster_count} players")

    if sk_count > 0:
        date_range = conn.execute("""
            SELECT MIN(game_date), MAX(game_date)
            FROM game_logs_skaters
        """).fetchone()
        print(f"  Date range:       {date_range[0]} to {date_range[1]}")

        # Teams covered
        teams = conn.execute("""
            SELECT DISTINCT team FROM game_logs_skaters ORDER BY team
        """).fetchall()
        print(f"  Teams:            {len(teams)} ({', '.join(t[0] for t in teams)})")

    # Last fetch
    last_fetch = conn.execute("""
        SELECT MAX(fetched_at) FROM game_logs_skaters
    """).fetchone()[0]
    if last_fetch:
        print(f"  Last fetch:       {last_fetch}")

    conn.close()


def export_game_logs(output: str = 'game_logs_export.csv'):
    """Export all game logs to CSV."""
    conn = get_db()

    sk = pd.read_sql_query("SELECT * FROM game_logs_skaters ORDER BY game_date", conn)
    g = pd.read_sql_query("SELECT * FROM game_logs_goalies ORDER BY game_date", conn)

    sk['player_type'] = 'skater'
    g['player_type'] = 'goalie'

    print(f"  Exported {len(sk)} skater + {len(g)} goalie game logs to {output}")
    sk.to_csv(output, index=False)

    goalie_output = output.replace('.csv', '_goalies.csv')
    g.to_csv(goalie_output, index=False)
    print(f"  Goalie logs: {goalie_output}")

    conn.close()


def get_player_sequence(player_name: str = None, player_id: int = None,
                        player_type: str = 'skater') -> Optional[np.ndarray]:
    """
    Get a player's FPTS sequence from the database.
    Useful for ad-hoc analysis.
    """
    conn = get_db()
    table = 'game_logs_goalies' if player_type == 'goalie' else 'game_logs_skaters'

    if player_id:
        df = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE player_id = ? ORDER BY game_date",
            conn, params=(player_id,)
        )
    elif player_name:
        df = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE player_name LIKE ? ORDER BY game_date",
            conn, params=(f"%{player_name}%",)
        )
    else:
        return None

    conn.close()

    if df.empty:
        return None

    return df['dk_fpts'].values


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='NHL Game Log Fetcher & Analyzer')
    sub = parser.add_subparsers(dest='command', help='Command')

    # Fetch
    fetch_p = sub.add_parser('fetch', help='Fetch game logs from NHL API')
    fetch_p.add_argument('--all', action='store_true', help='Fetch all 32 teams')
    fetch_p.add_argument('--teams', nargs='+', help='Specific teams (e.g., COL TBL)')
    fetch_p.add_argument('--batch', type=int, default=0,
                        help='Batch number (1-indexed, use with --teams-per-batch)')
    fetch_p.add_argument('--teams-per-batch', type=int, default=8,
                        help='Teams per batch (default: 8)')
    fetch_p.add_argument('--update-only', action='store_true',
                        help='Skip players fetched in last 24h')
    fetch_p.add_argument('--rate-limit', type=float, default=0.35,
                        help='Seconds between API calls (default: 0.35)')

    # Analyze
    analyze_p = sub.add_parser('analyze', help='Run regime switching + change point analysis')
    analyze_p.add_argument('--min-games', type=int, default=10,
                          help='Minimum games for analysis (default: 10)')
    analyze_p.add_argument('--top', type=int, default=15,
                          help='Top N players to show (default: 15)')

    # Status
    sub.add_parser('status', help='Show database status')

    # Export
    export_p = sub.add_parser('export', help='Export game logs to CSV')
    export_p.add_argument('--output', default='game_logs_export.csv')

    args = parser.parse_args()

    if args.command == 'fetch':
        # Determine teams to fetch
        if args.teams:
            teams = [t.upper() for t in args.teams]
        elif args.all:
            # Remove ARI (relocated to UTA)
            teams = [t for t in sorted(NHL_TEAMS) if t != 'ARI']
        else:
            parser.error("Specify --all or --teams")
            return

        # Apply batching
        if args.batch > 0:
            n = args.teams_per_batch
            start = (args.batch - 1) * n
            teams = teams[start:start + n]
            total_batches = -(-len(NHL_TEAMS) // n)  # ceil division
            print(f"  Batch {args.batch}/{total_batches}: {teams}")

        fetcher = GameLogFetcher(rate_limit=args.rate_limit)
        fetcher.fetch_teams(teams, update_only=args.update_only)

    elif args.command == 'analyze':
        analyze_game_logs(min_games=args.min_games, top_n=args.top)

    elif args.command == 'status':
        show_status()

    elif args.command == 'export':
        export_game_logs(args.output)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
