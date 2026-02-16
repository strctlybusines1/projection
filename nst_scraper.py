#!/usr/bin/env python3
"""
Natural Stat Trick (NST) Scraper
=================================
Fetches team, skater, and goalie stats from naturalstattrick.com across
multiple game situations (5v5, PP, PK, All) and stores them in SQLite.

The date parameters on NST work as a cumulative window: fd=start, td=end
gives you totals from that range. For backtesting, we pull the full season
as a single snapshot (Oct 7 through the last game date before Olympic break).

For daily snapshots (to track regime changes over time), run with --daily
which pulls the same data but tags each fetch with the current date.

Usage:
    # Full season pull (all situations, all entity types)
    python nst_scraper.py --fetch-all --from-date 2025-10-07 --to-date 2026-02-04

    # Specific situation
    python nst_scraper.py --fetch-all --sit 5v5

    # Teams only
    python nst_scraper.py --teams --from-date 2025-10-07 --to-date 2026-02-04

    # Show database status
    python nst_scraper.py --status

NST URL patterns:
    Teams:           teamtable.php?sit=5v5&...
    Skaters on-ice:  playerteams.php?stdoi=oi&pos=S&sit=5v5&...
    Skaters indiv:   playerteams.php?stdoi=std&pos=S&sit=5v5&...
    Goalies:         playerteams.php?stdoi=g&pos=G&sit=5v5&...

Situations:  5v5, pp, pk, all
"""

import argparse
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ================================================================
#  Configuration
# ================================================================

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "nhl_dfs_history.db"
NST_DIR = Path(__file__).parent / "nst_data"

BASE_URL = "https://www.naturalstattrick.com"

# Situations to scrape
SITUATIONS = ["5v5", "pp", "pk", "all"]

# Rate limiting
REQUEST_DELAY = 2.0  # seconds between requests (be respectful)

# Common URL params
COMMON_PARAMS = {
    "fromseason": "20252026",
    "thruseason": "20252026",
    "stype": "2",          # regular season
    "score": "all",
    "rate": "n",           # raw counts, not per-60
    "loc": "B",            # both home and away
    "gpf": "410",
    "tgp": "410",
}

# NST team abbreviations -> our standard
NST_TEAM_MAP = {
    # Abbreviation forms (used in skater/goalie tables)
    "ANA": "ANA", "BOS": "BOS", "BUF": "BUF", "CGY": "CGY",
    "CAR": "CAR", "CHI": "CHI", "COL": "COL", "CBJ": "CBJ",
    "DAL": "DAL", "DET": "DET", "EDM": "EDM", "FLA": "FLA",
    "L.A": "LAK", "MIN": "MIN", "MTL": "MTL", "NSH": "NSH",
    "N.J": "NJD", "NYI": "NYI", "NYR": "NYR", "OTT": "OTT",
    "PHI": "PHI", "PIT": "PIT", "S.J": "SJS", "SEA": "SEA",
    "STL": "STL", "T.B": "TBL", "TOR": "TOR", "UTA": "UTA",
    "VAN": "VAN", "VGK": "VGK", "WSH": "WSH", "WPG": "WPG",
    # Full name forms (used in team table)
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL", "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
    "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA", "St. Louis Blues": "STL", "St Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}


# ================================================================
#  Database Setup
# ================================================================

def get_db() -> sqlite3.Connection:
    """Get database connection and ensure NST tables exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    _create_nst_tables(conn)
    return conn


def _create_nst_tables(conn: sqlite3.Connection):
    """Create NST data tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nst_teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_date TEXT NOT NULL,
            from_date TEXT NOT NULL,
            to_date TEXT NOT NULL,
            situation TEXT NOT NULL,
            team TEXT NOT NULL,
            gp INTEGER,
            toi TEXT,
            w INTEGER, l INTEGER, otl INTEGER,
            cf INTEGER, ca INTEGER, cf_pct REAL,
            ff INTEGER, fa INTEGER, ff_pct REAL,
            sf INTEGER, sa INTEGER, sf_pct REAL,
            gf INTEGER, ga INTEGER, gf_pct REAL,
            xgf REAL, xga REAL, xgf_pct REAL,
            scf INTEGER, sca INTEGER, scf_pct REAL,
            hdcf INTEGER, hdca INTEGER, hdcf_pct REAL,
            hdgf INTEGER, hdga INTEGER, hdgf_pct REAL,
            mdcf INTEGER, mdca INTEGER, mdcf_pct REAL,
            ldcf INTEGER, ldca INTEGER, ldcf_pct REAL,
            sh_pct REAL, sv_pct REAL, pdo REAL,
            raw_data TEXT,
            UNIQUE(fetch_date, from_date, to_date, situation, team)
        );

        CREATE TABLE IF NOT EXISTS nst_skaters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_date TEXT NOT NULL,
            from_date TEXT NOT NULL,
            to_date TEXT NOT NULL,
            situation TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            player TEXT NOT NULL,
            team TEXT NOT NULL,
            position TEXT,
            gp INTEGER,
            toi TEXT,
            -- On-ice columns (stdoi=oi)
            cf INTEGER, ca INTEGER, cf_pct REAL,
            ff INTEGER, fa INTEGER, ff_pct REAL,
            sf INTEGER, sa INTEGER, sf_pct REAL,
            gf INTEGER, ga INTEGER, gf_pct REAL,
            xgf REAL, xga REAL, xgf_pct REAL,
            scf INTEGER, sca INTEGER, scf_pct REAL,
            hdcf INTEGER, hdca INTEGER, hdcf_pct REAL,
            on_ice_sh_pct REAL, on_ice_sv_pct REAL, pdo REAL,
            oz_start_pct REAL,
            -- Individual columns (stdoi=std)
            goals INTEGER, total_assists INTEGER,
            first_assists INTEGER, second_assists INTEGER,
            total_points INTEGER, ipp REAL,
            shots INTEGER, sh_pct REAL,
            ixg REAL, icf INTEGER, iff INTEGER,
            iscf INTEGER, ihdcf INTEGER,
            rush_attempts INTEGER, rebounds_created INTEGER,
            pim INTEGER, total_penalties INTEGER,
            penalties_drawn INTEGER,
            giveaways INTEGER, takeaways INTEGER,
            hits INTEGER, hits_taken INTEGER,
            shots_blocked INTEGER,
            raw_data TEXT,
            UNIQUE(fetch_date, from_date, to_date, situation, stat_type, player, team)
        );

        CREATE TABLE IF NOT EXISTS nst_goalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_date TEXT NOT NULL,
            from_date TEXT NOT NULL,
            to_date TEXT NOT NULL,
            situation TEXT NOT NULL,
            player TEXT NOT NULL,
            team TEXT NOT NULL,
            gp INTEGER,
            toi TEXT,
            shots_against INTEGER, saves INTEGER,
            goals_against INTEGER, sv_pct REAL,
            gaa REAL, gsaa REAL, xga REAL,
            hd_shots_against INTEGER, hd_saves INTEGER,
            hd_goals_against INTEGER, hd_sv_pct REAL,
            hd_gaa REAL, hd_gsaa REAL,
            md_shots_against INTEGER, md_saves INTEGER,
            md_goals_against INTEGER, md_sv_pct REAL,
            ld_shots_against INTEGER, ld_saves INTEGER,
            ld_goals_against INTEGER, ld_sv_pct REAL,
            rush_attempts_against INTEGER,
            rebound_attempts_against INTEGER,
            avg_shot_distance REAL,
            raw_data TEXT,
            UNIQUE(fetch_date, from_date, to_date, situation, player, team)
        );
    """)
    conn.commit()


# ================================================================
#  HTML Table Parser
# ================================================================

def fetch_page(url: str, params: dict, max_retries: int = 3) -> Optional[str]:
    """Fetch a page with retry logic."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.text
            print(f"    HTTP {resp.status_code} on attempt {attempt+1}")
        except requests.RequestException as e:
            print(f"    Request error on attempt {attempt+1}: {e}")
        time.sleep(REQUEST_DELAY * (attempt + 1))
    return None


def parse_html_table(html: str, table_id: str = None) -> Optional[pd.DataFrame]:
    """
    Parse the main data table from an NST HTML page.
    NST uses DataTables with the data in a <table> element.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find the data table - NST typically uses id="teams" or similar
    if table_id:
        table = soup.find("table", {"id": table_id})
    else:
        # Find the largest table on the page
        tables = soup.find_all("table")
        table = max(tables, key=lambda t: len(t.find_all("tr"))) if tables else None

    if not table:
        return None

    # Parse headers
    headers = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all("th"):
            text = th.get_text(strip=True)
            headers.append(text)

    # Parse rows
    rows = []
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            cells = []
            for td in tr.find_all("td"):
                text = td.get_text(strip=True)
                cells.append(text)
            if cells:
                rows.append(cells)

    if not headers or not rows:
        return None

    # Ensure header count matches data
    max_cols = max(len(r) for r in rows) if rows else 0
    while len(headers) < max_cols:
        headers.append(f"col_{len(headers)}")

    # Trim rows to header length
    rows = [r[:len(headers)] for r in rows]

    df = pd.DataFrame(rows, columns=headers)
    return df


def clean_numeric(val):
    """Convert NST string values to numeric, handling percentages and dashes."""
    if pd.isna(val) or val in ("", "-", "--", "N/A"):
        return None
    s = str(val).strip().replace(",", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


# ================================================================
#  Fetch Functions
# ================================================================

def fetch_team_stats(situation: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
    """Fetch team stats from NST for a given situation."""
    params = {
        **COMMON_PARAMS,
        "sit": situation,
        "team": "all",
        "fd": from_date,
        "td": to_date,
    }
    url = f"{BASE_URL}/teamtable.php"
    print(f"  Fetching teams [{situation}] {from_date} to {to_date}...")

    html = fetch_page(url, params)
    if not html:
        print("    FAILED to fetch page")
        return None

    df = parse_html_table(html)
    if df is None or df.empty:
        print("    No data found in table")
        return None

    # Standardize team names
    if "Team" in df.columns:
        df["team_std"] = df["Team"].map(NST_TEAM_MAP).fillna(df["Team"])

    df["situation"] = situation
    df["from_date"] = from_date
    df["to_date"] = to_date
    df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")

    print(f"    Got {len(df)} teams")
    return df


def fetch_skater_stats(stdoi: str, situation: str, from_date: str,
                       to_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch skater stats from NST.
    stdoi: 'oi' for on-ice, 'std' for individual
    """
    params = {
        **COMMON_PARAMS,
        "sit": situation,
        "stdoi": stdoi,
        "team": "ALL",
        "pos": "S",
        "toi": "0",
        "gpfilt": "none",
        "fd": from_date,
        "td": to_date,
        "lines": "single",
        "datea": "",
    }
    url = f"{BASE_URL}/playerteams.php"
    label = "on-ice" if stdoi == "oi" else "individual"
    print(f"  Fetching skaters {label} [{situation}] {from_date} to {to_date}...")

    html = fetch_page(url, params)
    if not html:
        print("    FAILED to fetch page")
        return None

    df = parse_html_table(html)
    if df is None or df.empty:
        print("    No data found in table")
        return None

    # Standardize team names
    if "Team" in df.columns:
        df["team_std"] = df["Team"].map(NST_TEAM_MAP).fillna(df["Team"])

    df["situation"] = situation
    df["stat_type"] = stdoi
    df["from_date"] = from_date
    df["to_date"] = to_date
    df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")

    print(f"    Got {len(df)} skaters")
    return df


def fetch_goalie_stats(situation: str, from_date: str,
                       to_date: str) -> Optional[pd.DataFrame]:
    """Fetch goalie stats from NST (stdoi=g)."""
    params = {
        **COMMON_PARAMS,
        "sit": situation,
        "stdoi": "g",
        "team": "ALL",
        "pos": "G",
        "toi": "0",
        "gpfilt": "none",
        "fd": from_date,
        "td": to_date,
        "lines": "single",
        "datea": "",
    }
    url = f"{BASE_URL}/playerteams.php"
    print(f"  Fetching goalies [{situation}] {from_date} to {to_date}...")

    html = fetch_page(url, params)
    if not html:
        print("    FAILED to fetch page")
        return None

    df = parse_html_table(html)
    if df is None or df.empty:
        print("    No data found in table")
        return None

    if "Team" in df.columns:
        df["team_std"] = df["Team"].map(NST_TEAM_MAP).fillna(df["Team"])

    df["situation"] = situation
    df["from_date"] = from_date
    df["to_date"] = to_date
    df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")

    print(f"    Got {len(df)} goalies")
    return df


# ================================================================
#  Database Storage
# ================================================================

def _safe_float(row, col):
    """Safely extract a float from a pandas Series row."""
    try:
        val = row.get(col) if hasattr(row, 'get') else row[col]
        return clean_numeric(val)
    except (KeyError, IndexError, TypeError):
        return None


def _safe_int(row, col):
    """Safely extract an int from a pandas Series row."""
    v = _safe_float(row, col)
    return int(v) if v is not None else None


def store_team_stats(conn: sqlite3.Connection, df: pd.DataFrame):
    """Store team stats DataFrame into nst_teams table."""
    if df is None or df.empty:
        return 0

    count = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO nst_teams
                (fetch_date, from_date, to_date, situation, team,
                 gp, toi, w, l, otl,
                 cf, ca, cf_pct, ff, fa, ff_pct,
                 sf, sa, sf_pct, gf, ga, gf_pct,
                 xgf, xga, xgf_pct,
                 scf, sca, scf_pct,
                 hdcf, hdca, hdcf_pct, hdgf, hdga, hdgf_pct,
                 mdcf, mdca, mdcf_pct,
                 ldcf, ldca, ldcf_pct,
                 sh_pct, sv_pct, pdo,
                 raw_data)
                VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?)
            """, (
                row.get("fetch_date", ""),
                row.get("from_date", ""),
                row.get("to_date", ""),
                row.get("situation", ""),
                row.get("team_std", row.get("Team", "")),
                _safe_int(row, "GP"),
                row.get("TOI", ""),
                _safe_int(row, "W"), _safe_int(row, "L"), _safe_int(row, "OTL"),
                _safe_int(row, "CF"), _safe_int(row, "CA"), _safe_float(row, "CF%"),
                _safe_int(row, "FF"), _safe_int(row, "FA"), _safe_float(row, "FF%"),
                _safe_int(row, "SF"), _safe_int(row, "SA"), _safe_float(row, "SF%"),
                _safe_int(row, "GF"), _safe_int(row, "GA"), _safe_float(row, "GF%"),
                _safe_float(row, "xGF"), _safe_float(row, "xGA"), _safe_float(row, "xGF%"),
                _safe_int(row, "SCF"), _safe_int(row, "SCA"), _safe_float(row, "SCF%"),
                _safe_int(row, "HDCF"), _safe_int(row, "HDCA"), _safe_float(row, "HDCF%"),
                _safe_int(row, "HDGF"), _safe_int(row, "HDGA"), _safe_float(row, "HDGF%"),
                _safe_int(row, "MDCF"), _safe_int(row, "MDCA"), _safe_float(row, "MDCF%"),
                _safe_int(row, "LDCF"), _safe_int(row, "LDCA"), _safe_float(row, "LDCF%"),
                _safe_float(row, "SH%"), _safe_float(row, "SV%"), _safe_float(row, "PDO"),
                str(row.to_dict()),
            ))
            count += 1
        except Exception as e:
            print(f"    Error storing team {row.get('Team', '?')}: {e}")

    conn.commit()
    return count


def store_skater_stats(conn: sqlite3.Connection, df: pd.DataFrame, stat_type: str):
    """Store skater stats into nst_skaters table."""
    if df is None or df.empty:
        return 0

    count = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO nst_skaters
                (fetch_date, from_date, to_date, situation, stat_type,
                 player, team, position, gp, toi,
                 cf, ca, cf_pct, ff, fa, ff_pct,
                 sf, sa, sf_pct, gf, ga, gf_pct,
                 xgf, xga, xgf_pct,
                 scf, sca, scf_pct,
                 hdcf, hdca, hdcf_pct,
                 on_ice_sh_pct, on_ice_sv_pct, pdo, oz_start_pct,
                 goals, total_assists, first_assists, second_assists,
                 total_points, ipp, shots, sh_pct,
                 ixg, icf, iff, iscf, ihdcf,
                 rush_attempts, rebounds_created,
                 pim, total_penalties, penalties_drawn,
                 giveaways, takeaways, hits, hits_taken, shots_blocked,
                 raw_data)
                VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?, ?,?,?, ?,?,?,?,?, ?)
            """, (
                row.get("fetch_date", ""),
                row.get("from_date", ""),
                row.get("to_date", ""),
                row.get("situation", ""),
                stat_type,
                row.get("Player", ""),
                row.get("team_std", row.get("Team", "")),
                row.get("Position", ""),
                _safe_int(row, "GP"),
                row.get("TOI", ""),
                # On-ice fields
                _safe_int(row, "CF"), _safe_int(row, "CA"), _safe_float(row, "CF%"),
                _safe_int(row, "FF"), _safe_int(row, "FA"), _safe_float(row, "FF%"),
                _safe_int(row, "SF"), _safe_int(row, "SA"), _safe_float(row, "SF%"),
                _safe_int(row, "GF"), _safe_int(row, "GA"), _safe_float(row, "GF%"),
                _safe_float(row, "xGF"), _safe_float(row, "xGA"), _safe_float(row, "xGF%"),
                _safe_int(row, "SCF"), _safe_int(row, "SCA"), _safe_float(row, "SCF%"),
                _safe_int(row, "HDCF"), _safe_int(row, "HDCA"), _safe_float(row, "HDCF%"),
                _safe_float(row, "On-Ice SH%") if stat_type == "oi" else _safe_float(row, "SH%"),
                _safe_float(row, "On-Ice SV%"),
                _safe_float(row, "PDO"),
                _safe_float(row, "Off. Zone Start %"),
                # Individual fields
                _safe_int(row, "Goals"), _safe_int(row, "Total Assists"),
                _safe_int(row, "First Assists"), _safe_int(row, "Second Assists"),
                _safe_int(row, "Total Points"), _safe_float(row, "IPP"),
                _safe_int(row, "Shots"), _safe_float(row, "SH%") if stat_type == "std" else None,
                _safe_float(row, "ixG"), _safe_int(row, "iCF"), _safe_int(row, "iFF"),
                _safe_int(row, "iSCF"), _safe_int(row, "iHDCF"),
                _safe_int(row, "Rush Attempts"), _safe_int(row, "Rebounds Created"),
                _safe_int(row, "PIM"), _safe_int(row, "Total Penalties"),
                _safe_int(row, "Penalties Drawn"),
                _safe_int(row, "Giveaways"), _safe_int(row, "Takeaways"),
                _safe_int(row, "Hits"), _safe_int(row, "Hits Taken"),
                _safe_int(row, "Shots Blocked"),
                str(row.to_dict()),
            ))
            count += 1
        except Exception as e:
            print(f"    Error storing skater {row.get('Player', '?')}: {e}")

    conn.commit()
    return count


def store_goalie_stats(conn: sqlite3.Connection, df: pd.DataFrame):
    """Store goalie stats into nst_goalies table."""
    if df is None or df.empty:
        return 0

    count = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO nst_goalies
                (fetch_date, from_date, to_date, situation,
                 player, team, gp, toi,
                 shots_against, saves, goals_against, sv_pct, gaa, gsaa, xga,
                 hd_shots_against, hd_saves, hd_goals_against, hd_sv_pct, hd_gaa, hd_gsaa,
                 md_shots_against, md_saves, md_goals_against, md_sv_pct,
                 ld_shots_against, ld_saves, ld_goals_against, ld_sv_pct,
                 rush_attempts_against, rebound_attempts_against, avg_shot_distance,
                 raw_data)
                VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?, ?)
            """, (
                row.get("fetch_date", ""),
                row.get("from_date", ""),
                row.get("to_date", ""),
                row.get("situation", ""),
                row.get("Player", ""),
                row.get("team_std", row.get("Team", "")),
                _safe_int(row, "GP"),
                row.get("TOI", ""),
                _safe_int(row, "Shots Against"), _safe_int(row, "Saves"),
                _safe_int(row, "Goals Against"), _safe_float(row, "SV%"),
                _safe_float(row, "GAA"), _safe_float(row, "GSAA"),
                _safe_float(row, "xG Against"),
                _safe_int(row, "HD Shots Against"), _safe_int(row, "HD Saves"),
                _safe_int(row, "HD Goals Against"), _safe_float(row, "HDSV%"),
                _safe_float(row, "HDGAA"), _safe_float(row, "HDGSAA"),
                _safe_int(row, "MD Shots Against"), _safe_int(row, "MD Saves"),
                _safe_int(row, "MD Goals Against"), _safe_float(row, "MDSV%"),
                _safe_int(row, "LD Shots Against"), _safe_int(row, "LD Saves"),
                _safe_int(row, "LD Goals Against"), _safe_float(row, "LDSV%"),
                _safe_int(row, "Rush Attempts Against"),
                _safe_int(row, "Rebound Attempts Against"),
                _safe_float(row, "Avg. Shot Distance"),
                str(row.to_dict()),
            ))
            count += 1
        except Exception as e:
            print(f"    Error storing goalie {row.get('Player', '?')}: {e}")

    conn.commit()
    return count


# ================================================================
#  CSV File Storage (backup + HMM compatibility)
# ================================================================

def save_csv_backup(df: pd.DataFrame, entity_type: str, situation: str,
                    from_date: str, to_date: str):
    """Save a CSV backup to nst_data/ directory."""
    os.makedirs(NST_DIR, exist_ok=True)
    fname = f"{to_date}_{entity_type}_{situation}.csv"
    path = NST_DIR / fname
    df.to_csv(path, index=False)
    return path


# ================================================================
#  Main Fetch Orchestrator
# ================================================================

def fetch_all(from_date: str, to_date: str,
              situations: List[str] = None,
              skip_teams: bool = False,
              skip_skaters: bool = False,
              skip_goalies: bool = False):
    """
    Fetch all data from NST for the given date range and situations.
    """
    sits = situations or SITUATIONS
    conn = get_db()

    total = {"teams": 0, "skaters_oi": 0, "skaters_std": 0, "goalies": 0}

    print(f"\n{'='*60}")
    print(f"  NST SCRAPER — {from_date} to {to_date}")
    print(f"  Situations: {', '.join(sits)}")
    print(f"{'='*60}\n")

    for sit in sits:
        print(f"\n--- Situation: {sit.upper()} ---")

        # Teams
        if not skip_teams:
            df = fetch_team_stats(sit, from_date, to_date)
            if df is not None:
                n = store_team_stats(conn, df)
                total["teams"] += n
                save_csv_backup(df, "team", sit, from_date, to_date)
            time.sleep(REQUEST_DELAY)

        # Skaters on-ice
        if not skip_skaters:
            df = fetch_skater_stats("oi", sit, from_date, to_date)
            if df is not None:
                n = store_skater_stats(conn, df, "oi")
                total["skaters_oi"] += n
                save_csv_backup(df, "skater_oi", sit, from_date, to_date)
            time.sleep(REQUEST_DELAY)

            # Skaters individual
            df = fetch_skater_stats("std", sit, from_date, to_date)
            if df is not None:
                n = store_skater_stats(conn, df, "std")
                total["skaters_std"] += n
                save_csv_backup(df, "skater_std", sit, from_date, to_date)
            time.sleep(REQUEST_DELAY)

        # Goalies
        if not skip_goalies:
            df = fetch_goalie_stats(sit, from_date, to_date)
            if df is not None:
                n = store_goalie_stats(conn, df)
                total["goalies"] += n
                save_csv_backup(df, "goalie", sit, from_date, to_date)
            time.sleep(REQUEST_DELAY)

    conn.close()

    print(f"\n{'='*60}")
    print(f"  FETCH COMPLETE")
    print(f"  Teams:           {total['teams']}")
    print(f"  Skaters (on-ice): {total['skaters_oi']}")
    print(f"  Skaters (indiv):  {total['skaters_std']}")
    print(f"  Goalies:          {total['goalies']}")
    print(f"{'='*60}\n")

    return total


def fetch_backtest_snapshots(from_date: str = "2025-10-07",
                              final_date: str = "2026-02-04",
                              interval_days: int = 7,
                              situations: List[str] = None,
                              rate: str = "n"):
    """
    Fetch cumulative NST snapshots for walk-forward backtesting.

    For each snapshot, pulls data from `from_date` through `snapshot_date`.
    This gives us "as-of" data that only includes information available
    at that point in time — no future leakage.

    Args:
        from_date: Season start date (always Oct 7 for 2025-26)
        final_date: Last date to pull through (Feb 4 = last pre-Olympic game)
        interval_days: Days between snapshots (7 = weekly)
        situations: List of situations to fetch (default: all 4)
        rate: "n" for raw counts, "y" for per-60 rates
    """
    sits = situations or SITUATIONS
    conn = get_db()

    # Build list of snapshot end-dates
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(final_date, "%Y-%m-%d")

    # First snapshot should give ~1 month of data (for stable rates)
    # So first to_date = from_date + 30 days
    snapshot_dates = []
    current = start + timedelta(days=30)  # first snapshot after ~1 month
    while current <= end:
        snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=interval_days)
    # Always include the final date
    if snapshot_dates[-1] != final_date:
        snapshot_dates.append(final_date)

    total_snapshots = len(snapshot_dates)
    total_requests = total_snapshots * len(sits) * 4  # 4 entity types per situation
    est_minutes = (total_requests * REQUEST_DELAY) / 60

    print(f"\n{'='*60}")
    print(f"  NST BACKTEST SNAPSHOT FETCHER")
    print(f"{'='*60}")
    print(f"  Season start:    {from_date}")
    print(f"  Final date:      {final_date}")
    print(f"  Interval:        {interval_days} days")
    print(f"  Snapshots:       {total_snapshots}")
    print(f"  Situations:      {', '.join(sits)}")
    print(f"  Rate mode:       {'per-60' if rate == 'y' else 'raw counts'}")
    print(f"  Total requests:  ~{total_requests}")
    print(f"  Est. time:       ~{est_minutes:.0f} minutes")
    print(f"{'='*60}\n")

    # Check which snapshots already exist in DB
    existing = set()
    for table in ['nst_teams', 'nst_skaters', 'nst_goalies']:
        rows = conn.execute(
            f"SELECT DISTINCT to_date FROM {table} WHERE from_date = ?",
            (from_date,)
        ).fetchall()
        for r in rows:
            existing.add(r[0])

    if existing:
        print(f"  Already have snapshots for: {sorted(existing)}")

    # Override COMMON_PARAMS rate setting
    params_override = {"rate": rate}

    grand_total = {"teams": 0, "skaters_oi": 0, "skaters_std": 0, "goalies": 0}

    for i, snap_date in enumerate(snapshot_dates):
        # Skip if we already have this snapshot
        if snap_date in existing:
            print(f"\n[{i+1}/{total_snapshots}] {snap_date} — SKIPPING (already exists)")
            continue

        print(f"\n{'='*60}")
        print(f"  [{i+1}/{total_snapshots}] Snapshot: {from_date} → {snap_date}")
        print(f"{'='*60}")

        for sit in sits:
            print(f"\n  --- {sit.upper()} ---")

            # Teams
            df = fetch_team_stats(sit, from_date, snap_date)
            if df is not None:
                n = store_team_stats(conn, df)
                grand_total["teams"] += n
                save_csv_backup(df, "team", sit, from_date, snap_date)
            time.sleep(REQUEST_DELAY)

            # Skaters on-ice
            df = fetch_skater_stats("oi", sit, from_date, snap_date)
            if df is not None:
                n = store_skater_stats(conn, df, "oi")
                grand_total["skaters_oi"] += n
                save_csv_backup(df, "skater_oi", sit, from_date, snap_date)
            time.sleep(REQUEST_DELAY)

            # Skaters individual
            df = fetch_skater_stats("std", sit, from_date, snap_date)
            if df is not None:
                n = store_skater_stats(conn, df, "std")
                grand_total["skaters_std"] += n
                save_csv_backup(df, "skater_std", sit, from_date, snap_date)
            time.sleep(REQUEST_DELAY)

            # Goalies
            df = fetch_goalie_stats(sit, from_date, snap_date)
            if df is not None:
                n = store_goalie_stats(conn, df)
                grand_total["goalies"] += n
                save_csv_backup(df, "goalie", sit, from_date, snap_date)
            time.sleep(REQUEST_DELAY)

    conn.close()

    print(f"\n{'='*60}")
    print(f"  BACKTEST SNAPSHOTS COMPLETE")
    print(f"  Snapshots fetched: {total_snapshots - len(existing)} new")
    print(f"  Teams:             {grand_total['teams']}")
    print(f"  Skaters (on-ice):  {grand_total['skaters_oi']}")
    print(f"  Skaters (indiv):   {grand_total['skaters_std']}")
    print(f"  Goalies:           {grand_total['goalies']}")
    print(f"{'='*60}\n")

    return grand_total


def show_status():
    """Show current NST data status."""
    conn = get_db()

    print(f"\n{'='*60}")
    print(f"  NST DATA STATUS")
    print(f"{'='*60}")

    for table, label in [("nst_teams", "Teams"), ("nst_skaters", "Skaters"),
                         ("nst_goalies", "Goalies")]:
        try:
            r = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            total = r[0]

            if total > 0:
                sits = conn.execute(
                    f"SELECT situation, COUNT(*) FROM {table} GROUP BY situation"
                ).fetchall()
                dates = conn.execute(
                    f"SELECT MIN(from_date), MAX(to_date), MAX(fetch_date) FROM {table}"
                ).fetchone()

                print(f"\n  {label}: {total:,} rows")
                print(f"    Date range: {dates[0]} to {dates[1]}")
                print(f"    Last fetch: {dates[2]}")
                for sit, n in sits:
                    print(f"    {sit:>5s}: {n:,} rows")
            else:
                print(f"\n  {label}: empty")
        except Exception as e:
            print(f"\n  {label}: error — {e}")

    conn.close()


# ================================================================
#  CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natural Stat Trick Scraper")
    parser.add_argument("--fetch-all", action="store_true",
                        help="Fetch all data (teams, skaters, goalies)")
    parser.add_argument("--teams", action="store_true", help="Fetch teams only")
    parser.add_argument("--skaters", action="store_true", help="Fetch skaters only")
    parser.add_argument("--goalies", action="store_true", help="Fetch goalies only")
    parser.add_argument("--from-date", type=str, default="2025-10-07",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, default="2026-02-04",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--sit", type=str, default=None,
                        help="Specific situation (5v5, pp, pk, all)")
    parser.add_argument("--backtest-snapshots", action="store_true",
                        help="Fetch weekly cumulative snapshots for backtesting")
    parser.add_argument("--interval", type=int, default=7,
                        help="Days between backtest snapshots (default: 7)")
    parser.add_argument("--status", action="store_true", help="Show data status")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.backtest_snapshots:
        situations = [args.sit] if args.sit else SITUATIONS
        fetch_backtest_snapshots(
            from_date=args.from_date,
            final_date=args.to_date,
            interval_days=args.interval,
            situations=situations,
        )
    elif args.fetch_all or args.teams or args.skaters or args.goalies:
        situations = [args.sit] if args.sit else SITUATIONS
        fetch_all(
            from_date=args.from_date,
            to_date=args.to_date,
            situations=situations,
            skip_teams=not (args.fetch_all or args.teams),
            skip_skaters=not (args.fetch_all or args.skaters),
            skip_goalies=not (args.fetch_all or args.goalies),
        )
    else:
        parser.print_help()
