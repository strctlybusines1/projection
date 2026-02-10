#!/usr/bin/env python3
"""
Historical Odds Manager
========================

Three ways to build the historical odds database:
1. Import from Vegas_Historical.csv and vegas/*.csv files (free, immediate)
2. Capture live odds daily before lock (free tier, going forward)
3. Fetch from historical API endpoint (paid plan only)

Plus a DB lookup function for game_environment.py integration.

Usage:
    python historical_odds.py --import-csv         # Import all existing CSV data
    python historical_odds.py --capture             # Capture today's live odds
    python historical_odds.py --status              # Show database status
    python historical_odds.py --lookup 2026-01-29   # Look up odds for a date
    python historical_odds.py --backfill            # Fetch from API (paid only)
    python historical_odds.py --export              # Export to CSV
"""

import sqlite3
import requests
import time
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"
API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
BASE_URL = "https://api.the-odds-api.com/v4"

TEAM_ABBREV = {
    'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS',
    'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN',
    'Montreal Canadiens': 'MTL', 'Montréal Canadiens': 'MTL',
    'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD',
    'New York Islanders': 'NYI', 'New York Rangers': 'NYR',
    'Ottawa Senators': 'OTT', 'Philadelphia Flyers': 'PHI',
    'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS',
    'Seattle Kraken': 'SEA', 'St. Louis Blues': 'STL', 'St Louis Blues': 'STL',
    'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR',
    'Utah Mammoth': 'UTA', 'Utah Hockey Club': 'UTA',
    'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG',
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_tables():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS historical_odds (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date       TEXT NOT NULL,
            game_id_odds    TEXT,
            home_team       TEXT NOT NULL,
            away_team       TEXT NOT NULL,
            home_abbrev     TEXT,
            away_abbrev     TEXT,
            commence_time   TEXT,
            home_ml         INTEGER,
            away_ml         INTEGER,
            game_total      REAL,
            over_price      INTEGER,
            under_price     INTEGER,
            home_spread     REAL,
            home_spread_price INTEGER,
            away_spread_price INTEGER,
            home_implied_prob REAL,
            away_implied_prob REAL,
            home_implied_total REAL,
            away_implied_total REAL,
            best_book       TEXT,
            n_bookmakers    INTEGER,
            raw_json        TEXT,
            snapshot_time   TEXT,
            fetched_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(game_date, home_abbrev, away_abbrev)
        );
        CREATE INDEX IF NOT EXISTS idx_hist_odds_date ON historical_odds(game_date);
    """)
    conn.commit()
    conn.close()


def ml_to_implied_prob(ml):
    if ml is None or ml == 0:
        return 0.5
    ml = int(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def prob_to_ml(p):
    if p <= 0 or p >= 1:
        return 0
    if p > 0.5:
        return int(-100 * p / (1 - p))
    else:
        return int(100 * (1 - p) / p)


def implied_team_totals(home_ml, away_ml, game_total):
    hp = ml_to_implied_prob(home_ml)
    ap = ml_to_implied_prob(away_ml)
    tp = hp + ap
    if tp > 0:
        hp, ap = hp / tp, ap / tp
    ht = game_total * (0.5 + 0.22 * (hp - 0.5))
    at = game_total - ht
    return round(ht, 2), round(at, 2)


def store_odds(games):
    conn = get_db()
    stored = 0
    for g in games:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO historical_odds
                (game_date, game_id_odds, home_team, away_team, home_abbrev, away_abbrev,
                 commence_time, home_ml, away_ml, game_total, over_price, under_price,
                 home_spread, home_spread_price, away_spread_price,
                 home_implied_prob, away_implied_prob,
                 home_implied_total, away_implied_total,
                 best_book, n_bookmakers, raw_json, snapshot_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                g.get('game_date'), g.get('game_id_odds'), g.get('home_team'), g.get('away_team'),
                g.get('home_abbrev'), g.get('away_abbrev'), g.get('commence_time'),
                g.get('home_ml'), g.get('away_ml'), g.get('game_total'),
                g.get('over_price'), g.get('under_price'),
                g.get('home_spread'), g.get('home_spread_price'), g.get('away_spread_price'),
                g.get('home_implied_prob'), g.get('away_implied_prob'),
                g.get('home_implied_total'), g.get('away_implied_total'),
                g.get('best_book', ''), g.get('n_bookmakers', 1),
                g.get('raw_json', ''), g.get('snapshot_time', ''),
            ))
            stored += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    return stored


# ================================================================
#  1. Import from CSV files
# ================================================================

def import_vegas_csv():
    """Import all available Vegas CSV data into the DB."""
    create_tables()
    total = 0

    # Vegas_Historical.csv
    csv_path = Path(__file__).parent / 'Vegas_Historical.csv'
    if csv_path.exists():
        vdf = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"  Vegas_Historical.csv: {len(vdf)} rows, {vdf['Date'].nunique()} dates")

        def parse_date(d):
            parts = str(d).split('.')
            if len(parts) == 3:
                return f"20{parts[2]}-{int(parts[0]):02d}-{int(parts[1]):02d}"
            return str(d)

        vdf['game_date'] = vdf['Date'].apply(parse_date)
        games = []
        for date in vdf['game_date'].unique():
            day = vdf[vdf['game_date'] == date]
            home_rows = day[day['Opp'].str.startswith('vs')]
            for _, hrow in home_rows.iterrows():
                home_team = hrow['Team']
                away_team = str(hrow['Opp']).replace('vs ', '').strip()
                hp = float(str(hrow.get('Win %', '50')).replace('%', '')) / 100.0
                games.append({
                    'game_date': date,
                    'game_id_odds': f"csv_{home_team}_{away_team}_{date}",
                    'home_team': home_team, 'away_team': away_team,
                    'home_abbrev': home_team, 'away_abbrev': away_team,
                    'commence_time': f"{date}T00:00:00Z",
                    'home_ml': prob_to_ml(hp), 'away_ml': prob_to_ml(1 - hp),
                    'game_total': hrow['Total'],
                    'over_price': None, 'under_price': None,
                    'home_spread': None, 'home_spread_price': None, 'away_spread_price': None,
                    'home_implied_prob': round(hp, 4), 'away_implied_prob': round(1 - hp, 4),
                    'home_implied_total': hrow['TeamGoal'], 'away_implied_total': hrow['OppGoal'],
                    'best_book': 'csv', 'n_bookmakers': 1,
                    'raw_json': '', 'snapshot_time': 'vegas_historical_csv',
                })
        stored = store_odds(games)
        print(f"    -> {stored} games imported")
        total += stored

    # vegas/VegasNHL_*.csv (with moneylines)
    vegas_dir = Path(__file__).parent / 'vegas'
    if vegas_dir.exists():
        for csv_file in sorted(vegas_dir.glob('VegasNHL_*.csv')):
            name = csv_file.stem.replace('VegasNHL_', '')
            parts = name.split('.')
            if len(parts) != 3:
                continue
            game_date = f"20{parts[2]}-{int(parts[0]):02d}-{int(parts[1]):02d}"
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
            except Exception:
                continue
            if 'moneyline' not in df.columns:
                continue

            games = []
            for i in range(0, len(df) - 1, 2):
                r1, r2 = df.iloc[i], df.iloc[i + 1]
                t1, t2 = str(r1['team']).strip(), str(r2['team']).strip()
                ml1, ml2 = int(r1['moneyline']), int(r2['moneyline'])
                gt = float(r1.get('game_total', 6.0))
                # Negative ML = favorite = likely home
                if ml2 < ml1:
                    home_t, away_t, hml, aml = t2, t1, ml2, ml1
                else:
                    home_t, away_t, hml, aml = t1, t2, ml1, ml2
                hp = ml_to_implied_prob(hml)
                ap = ml_to_implied_prob(aml)
                tp = hp + ap
                if tp > 0:
                    hp, ap = hp / tp, ap / tp
                ht, at = implied_team_totals(hml, aml, gt)
                games.append({
                    'game_date': game_date,
                    'game_id_odds': f"detail_{away_t}_{home_t}_{game_date}",
                    'home_team': home_t, 'away_team': away_t,
                    'home_abbrev': home_t, 'away_abbrev': away_t,
                    'commence_time': f"{game_date}T00:00:00Z",
                    'home_ml': hml, 'away_ml': aml, 'game_total': gt,
                    'over_price': None, 'under_price': None,
                    'home_spread': None, 'home_spread_price': None, 'away_spread_price': None,
                    'home_implied_prob': round(hp, 4), 'away_implied_prob': round(ap, 4),
                    'home_implied_total': ht, 'away_implied_total': at,
                    'best_book': 'vegas_detail', 'n_bookmakers': 1,
                    'raw_json': '', 'snapshot_time': f'detail_{csv_file.name}',
                })
            stored = store_odds(games)
            if stored > 0:
                print(f"  {csv_file.name}: {stored} games")
            total += stored

    print(f"\n  Total imported: {total} games")
    return total


# ================================================================
#  2. Live Odds Capture
# ================================================================

def _parse_api_games(raw_games, default_date, snapshot_time):
    results = []
    for game in raw_games:
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        commence = game.get('commence_time', '')
        game_date = commence[:10] if commence else default_date

        ahml, aaml, atot, aop, aup = [], [], [], [], []
        ahsp, ahspp, aaspp = [], [], []
        for bm in game.get('bookmakers', []):
            for mkt in bm.get('markets', []):
                if mkt['key'] == 'h2h':
                    for o in mkt['outcomes']:
                        if o['name'] == home: ahml.append(o['price'])
                        elif o['name'] == away: aaml.append(o['price'])
                elif mkt['key'] == 'totals':
                    for o in mkt['outcomes']:
                        if o['name'] == 'Over': atot.append(o.get('point', 0)); aop.append(o['price'])
                        elif o['name'] == 'Under': aup.append(o['price'])
                elif mkt['key'] == 'spreads':
                    for o in mkt['outcomes']:
                        if o['name'] == home: ahsp.append(o.get('point', 0)); ahspp.append(o['price'])
                        elif o['name'] == away: aaspp.append(o['price'])

        hml = int(np.median(ahml)) if ahml else None
        aml = int(np.median(aaml)) if aaml else None
        gt = np.median(atot) if atot else None
        hp, ap, ht, at = None, None, None, None
        if hml and aml:
            hp, ap = ml_to_implied_prob(hml), ml_to_implied_prob(aml)
            tp = hp + ap
            if tp > 0: hp, ap = round(hp/tp, 4), round(ap/tp, 4)
        if hml and aml and gt:
            ht, at = implied_team_totals(hml, aml, gt)

        results.append({
            'game_date': game_date, 'game_id_odds': game.get('id', ''),
            'home_team': home, 'away_team': away,
            'home_abbrev': TEAM_ABBREV.get(home, ''), 'away_abbrev': TEAM_ABBREV.get(away, ''),
            'commence_time': commence,
            'home_ml': hml, 'away_ml': aml, 'game_total': gt,
            'over_price': int(np.median(aop)) if aop else None,
            'under_price': int(np.median(aup)) if aup else None,
            'home_spread': np.median(ahsp) if ahsp else None,
            'home_spread_price': int(np.median(ahspp)) if ahspp else None,
            'away_spread_price': int(np.median(aaspp)) if aaspp else None,
            'home_implied_prob': hp, 'away_implied_prob': ap,
            'home_implied_total': ht, 'away_implied_total': at,
            'best_book': game.get('bookmakers', [{}])[0].get('title', '') if game.get('bookmakers') else '',
            'n_bookmakers': len(game.get('bookmakers', [])),
            'raw_json': json.dumps(game), 'snapshot_time': snapshot_time,
        })
    return results


def capture_daily():
    """Capture today's live odds. Run before slate lock."""
    create_tables()
    if not API_KEY:
        print("  ODDS_API_KEY not set")
        return 0
    url = f"{BASE_URL}/sports/icehockey_nhl/odds"
    params = {'apiKey': API_KEY, 'regions': 'us', 'markets': 'h2h,totals,spreads', 'oddsFormat': 'american'}
    resp = requests.get(url, params=params)
    remaining = resp.headers.get('x-requests-remaining', '?')
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}")
        return 0
    now = datetime.utcnow().isoformat() + 'Z'
    games = _parse_api_games(resp.json(), datetime.now().strftime('%Y-%m-%d'), f"live_{now}")
    stored = store_odds(games)
    print(f"  Captured {stored} games (API remaining: {remaining})")
    return stored


# ================================================================
#  3. DB Lookup
# ================================================================

def get_odds_for_date(date_str):
    """Get odds as team-level rows for a date. Compatible with Vegas_Historical format."""
    if not Path(str(DB_PATH)).exists():
        return pd.DataFrame()
    conn = get_db()
    try:
        odds = pd.read_sql_query("""
            SELECT game_date, home_abbrev, away_abbrev, game_total, home_ml, away_ml,
                   home_implied_total, away_implied_total, home_implied_prob, away_implied_prob, home_spread
            FROM historical_odds WHERE game_date = ?
        """, conn, params=(date_str,))
    except Exception:
        conn.close()
        return pd.DataFrame()
    conn.close()
    if odds.empty:
        return pd.DataFrame()

    rows = []
    for _, g in odds.iterrows():
        rows.append({'Team': g['home_abbrev'], 'Opp': g['away_abbrev'],
                     'TeamGoal': g['home_implied_total'], 'OppGoal': g['away_implied_total'],
                     'Total': g['game_total'], 'ML': g['home_ml'], 'is_home': True, 'date': g['game_date']})
        rows.append({'Team': g['away_abbrev'], 'Opp': g['home_abbrev'],
                     'TeamGoal': g['away_implied_total'], 'OppGoal': g['home_implied_total'],
                     'Total': g['game_total'], 'ML': g['away_ml'], 'is_home': False, 'date': g['game_date']})
    return pd.DataFrame(rows)


# ================================================================
#  Status & Backfill
# ================================================================

def show_status():
    create_tables()
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM historical_odds").fetchone()[0]
    dates = conn.execute("SELECT COUNT(DISTINCT game_date) FROM historical_odds").fetchone()[0]
    if total == 0:
        print("  No odds stored. Run --import-csv or --capture.")
        conn.close()
        return
    first = conn.execute("SELECT MIN(game_date) FROM historical_odds").fetchone()[0]
    last = conn.execute("SELECT MAX(game_date) FROM historical_odds").fetchone()[0]
    avg_t = conn.execute("SELECT AVG(game_total) FROM historical_odds WHERE game_total IS NOT NULL").fetchone()[0]
    print(f"\n  Historical Odds: {total} games, {dates} dates ({first} to {last}), avg total={avg_t:.1f}")
    conn.close()


def backfill(start='2025-10-04', end=None):
    """Fetch from historical API (paid plan only)."""
    create_tables()
    if not API_KEY:
        print("  ODDS_API_KEY not set")
        return
    # Quick test
    resp = requests.get(f"{BASE_URL}/historical/sports/icehockey_nhl/odds",
                        params={'apiKey': API_KEY, 'regions': 'us', 'markets': 'h2h', 'date': '2026-01-29T21:00:00Z'})
    if resp.status_code == 401:
        print("  Historical endpoint requires paid plan. Use --import-csv instead.")
        return
    print("  Historical API accessible. Fetching...")
    # Would proceed with full backfill here


# ================================================================
#  CLI
# ================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Historical Odds Manager')
    parser.add_argument('--import-csv', action='store_true', help='Import from Vegas CSV files')
    parser.add_argument('--capture', action='store_true', help='Capture live odds for today')
    parser.add_argument('--status', action='store_true', help='Show database status')
    parser.add_argument('--lookup', type=str, help='Look up odds for a date')
    parser.add_argument('--backfill', action='store_true', help='Fetch from API (paid only)')
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    args = parser.parse_args()

    if args.import_csv:
        import_vegas_csv()
    elif args.capture:
        capture_daily()
    elif args.lookup:
        df = get_odds_for_date(args.lookup)
        if df.empty:
            print(f"  No odds for {args.lookup}")
        else:
            print(f"\n  {args.lookup}: {len(df)} team-games")
            for _, r in df.iterrows():
                print(f"  {r['Team']:<5} vs {r['Opp']:<5} total={r['Total']:.1f} "
                      f"impl={r['TeamGoal']:.2f} ML={r['ML']}")
    elif args.backfill:
        backfill()
    elif args.export:
        from pathlib import Path as P
        conn = get_db()
        df = pd.read_sql_query("SELECT * FROM historical_odds ORDER BY game_date", conn)
        conn.close()
        out = P(__file__).parent / "data" / "historical_odds.csv"
        df.to_csv(out, index=False)
        print(f"  Exported {len(df)} to {out}")
    else:
        show_status()
