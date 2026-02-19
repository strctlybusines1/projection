#!/usr/bin/env python3
"""
NHL DFS ROI Dashboard — Track contest entries, profit/loss, and cumulative ROI.

Parses DraftKings contest export CSVs to find your entries and compute P&L.

Usage:
    # Set your DK username first (one time)
    python roi_dashboard.py --set-username "YourDKName"

    # Log a contest result manually
    python roi_dashboard.py log --date 2026-02-25 --contest "$5 SE" --fee 5 --winnings 12.50

    # Scan contest CSVs for your entries automatically
    python roi_dashboard.py scan
    python roi_dashboard.py scan --username "YourDKName"

    # Show dashboard
    python roi_dashboard.py report
    python roi_dashboard.py report --last 10

    # Export to CSV
    python roi_dashboard.py export --output roi_history.csv

Schema stored in data/nhl_dfs_history.db (same DB as history_db.py):
    contest_entries: One row per contest entry
"""

import argparse
import re
import os
import sys
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

from history_db import get_connection

PROJECT_ROOT = Path(__file__).resolve().parent
CONTESTS_DIR = PROJECT_ROOT / "contests"
CONFIG_FILE = PROJECT_ROOT / "data" / "roi_config.json"


# ================================================================
#  Config (DK username)
# ================================================================

def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(cfg: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def get_username() -> Optional[str]:
    return load_config().get('dk_username')


def set_username(username: str):
    cfg = load_config()
    cfg['dk_username'] = username
    save_config(cfg)
    print(f"  DK username set to: {username}")


# ================================================================
#  Database Setup
# ================================================================

def _ensure_roi_table(conn):
    """Create contest_entries table if needed (extends history_db schema)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS contest_entries (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_date     TEXT NOT NULL,
            contest_name   TEXT,
            entry_fee      REAL,
            field_size     INTEGER,
            our_score      REAL,
            our_rank       INTEGER,
            winning_score  REAL,
            cash_line      REAL,
            winnings       REAL DEFAULT 0,
            profit         REAL,
            lineup_hash    TEXT,
            entry_name     TEXT,
            contest_type   TEXT,
            created_at     TEXT DEFAULT (datetime('now')),
            UNIQUE(slate_date, contest_name, entry_name)
        );
        CREATE INDEX IF NOT EXISTS idx_ce_date ON contest_entries(slate_date);
    """)
    conn.commit()


# ================================================================
#  Parse Contest CSV
# ================================================================

def _parse_contest_filename(filename: str) -> Dict:
    """Extract contest metadata from DK export filename.

    Examples:
        $333NHL_1.22.26.csv → fee=3.33, date=2026-01-22, type=GPP
        $5SE_NHL1.28.26.csv → fee=5, date=2026-01-28, type=SE
        $121SE_NHL_2.5.26.csv → fee=121, date=2026-02-05, type=SE
        $10main_NHL1.22.26.csv → fee=10, date=2026-01-22, type=GPP
        $360NHLSpin_1.27.26.csv → fee=3.60, date=2026-01-27, type=GPP
    """
    result = {'filename': filename}

    # Extract entry fee
    fee_match = re.match(r'\$(\d+)', filename)
    if fee_match:
        raw_fee = int(fee_match.group(1))
        # Heuristic: $333 = $3.33, $360 = $3.60, but $5 = $5, $10 = $10, $121 = $121
        if raw_fee in (333,):
            result['entry_fee'] = 3.33
        elif raw_fee in (360,):
            result['entry_fee'] = 3.60
        elif raw_fee >= 100 and raw_fee not in (121,):
            result['entry_fee'] = raw_fee / 100
        else:
            result['entry_fee'] = float(raw_fee)

    # Detect SE vs GPP
    if 'SE' in filename.upper():
        result['contest_type'] = 'SE'
    elif 'main' in filename.lower():
        result['contest_type'] = 'GPP'
    elif 'Spin' in filename:
        result['contest_type'] = 'GPP'
    else:
        result['contest_type'] = 'GPP'

    # Extract date
    date_match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{2})', filename)
    if date_match:
        m, d, y = date_match.groups()
        result['slate_date'] = f"20{y}-{int(m):02d}-{int(d):02d}"

    result['contest_name'] = filename.replace('.csv', '')

    return result


def _parse_dk_payout(contest_df: pd.DataFrame, rank: int, entry_fee: float) -> float:
    """
    Estimate winnings from DK contest structure.

    DK doesn't include payouts in the export CSV, so we estimate:
    - For now, return 0 (user should log manually or we add payout parsing later)
    """
    # TODO: If DK adds payout column, parse it here
    return 0.0


def scan_contest_file(filepath: str, username: str) -> List[Dict]:
    """
    Scan a DK contest export CSV for entries matching the username.

    Returns list of entry dicts with score, rank, etc.
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return []

    if 'EntryName' not in df.columns:
        return []

    meta = _parse_contest_filename(os.path.basename(filepath))

    # Find user's entries (case-insensitive, partial match)
    # DK format: "username (1/5)" for multi-entry
    user_lower = username.lower()
    mask = df['EntryName'].str.lower().str.contains(user_lower, na=False)
    user_entries = df[mask]

    if user_entries.empty:
        return []

    field_size = df['EntryName'].nunique()

    # Get contest-level stats
    winning_score = df['Points'].max() if 'Points' in df.columns else None
    # Cash line estimate: ~20th percentile for GPP, ~50th for SE
    if meta.get('contest_type') == 'SE':
        cash_pct = 0.50
    else:
        cash_pct = 0.20
    cash_line = df['Points'].quantile(1 - cash_pct) if 'Points' in df.columns else None

    results = []
    for _, row in user_entries.iterrows():
        rank = int(row.get('Rank', 0))
        score = float(row.get('Points', 0))

        results.append({
            'slate_date': meta.get('slate_date', ''),
            'contest_name': meta.get('contest_name', ''),
            'entry_fee': meta.get('entry_fee', 0),
            'field_size': field_size,
            'our_score': score,
            'our_rank': rank,
            'winning_score': winning_score,
            'cash_line': cash_line,
            'winnings': 0.0,  # User must log manually
            'profit': -meta.get('entry_fee', 0),  # Default to loss until winnings logged
            'entry_name': row.get('EntryName', ''),
            'contest_type': meta.get('contest_type', 'GPP'),
        })

    return results


def scan_all_contests(username: str, log_to_db: bool = True) -> pd.DataFrame:
    """Scan all contest CSVs for user entries."""
    if not CONTESTS_DIR.exists():
        print("  No contests/ directory found.")
        return pd.DataFrame()

    files = sorted(glob.glob(str(CONTESTS_DIR / "*.csv")))
    all_entries = []

    for f in files:
        entries = scan_contest_file(f, username)
        all_entries.extend(entries)
        if entries:
            print(f"  Found {len(entries)} entries in {os.path.basename(f)}")

    if not all_entries:
        print(f"  No entries found for username '{username}'")
        return pd.DataFrame()

    df = pd.DataFrame(all_entries)
    print(f"\n  Total: {len(df)} entries across {df['slate_date'].nunique()} slates")

    if log_to_db:
        conn = get_connection()
        _ensure_roi_table(conn)
        inserted = 0
        for _, row in df.iterrows():
            try:
                conn.execute("""
                    INSERT INTO contest_entries
                        (slate_date, contest_name, entry_fee, field_size,
                         our_score, our_rank, winning_score, cash_line,
                         winnings, profit, entry_name, contest_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(slate_date, contest_name, entry_name) DO UPDATE SET
                        our_score=excluded.our_score, our_rank=excluded.our_rank,
                        field_size=excluded.field_size
                """, (
                    row['slate_date'], row['contest_name'], row['entry_fee'],
                    row['field_size'], row['our_score'], row['our_rank'],
                    row['winning_score'], row['cash_line'],
                    row['winnings'], row['profit'],
                    row['entry_name'], row['contest_type'],
                ))
                inserted += 1
            except Exception as e:
                pass
        conn.commit()
        conn.close()
        print(f"  Logged {inserted} entries to database")

    return df


# ================================================================
#  Manual Entry Logging
# ================================================================

def log_entry(date: str, contest: str, fee: float, winnings: float,
              score: float = None, rank: int = None, contest_type: str = 'GPP'):
    """Manually log a contest entry with known winnings."""
    conn = get_connection()
    _ensure_roi_table(conn)

    profit = winnings - fee
    entry_name = get_username() or 'manual'

    conn.execute("""
        INSERT INTO contest_entries
            (slate_date, contest_name, entry_fee, our_score, our_rank,
             winnings, profit, entry_name, contest_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(slate_date, contest_name, entry_name) DO UPDATE SET
            winnings=excluded.winnings, profit=excluded.profit,
            our_score=excluded.our_score, our_rank=excluded.our_rank
    """, (date, contest, fee, score, rank, winnings, profit, entry_name, contest_type))
    conn.commit()
    conn.close()

    icon = "💰" if profit > 0 else "📉"
    print(f"  {icon} Logged: {contest} on {date} — Fee: ${fee:.2f}, "
          f"Won: ${winnings:.2f}, Profit: ${profit:+.2f}")


def update_winnings(date: str, contest: str, winnings: float):
    """Update winnings for a previously scanned entry."""
    conn = get_connection()
    _ensure_roi_table(conn)

    # Get the entry fee to compute profit
    row = conn.execute(
        "SELECT entry_fee FROM contest_entries WHERE slate_date=? AND contest_name LIKE ?",
        (date, f"%{contest}%")
    ).fetchone()

    if not row:
        print(f"  Entry not found for {date} / {contest}")
        conn.close()
        return

    fee = row[0]
    profit = winnings - fee

    conn.execute("""
        UPDATE contest_entries SET winnings=?, profit=?
        WHERE slate_date=? AND contest_name LIKE ?
    """, (winnings, profit, date, f"%{contest}%"))
    conn.commit()
    conn.close()
    print(f"  Updated: {contest} on {date} — Won: ${winnings:.2f}, Profit: ${profit:+.2f}")


# ================================================================
#  Reports
# ================================================================

def print_roi_report(last_n: int = None):
    """Print ROI summary dashboard."""
    conn = get_connection()
    _ensure_roi_table(conn)

    query = """
        SELECT slate_date, contest_name, entry_fee, our_score, our_rank,
               field_size, winnings, profit, contest_type, entry_name
        FROM contest_entries
        ORDER BY slate_date DESC, contest_name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("  No contest entries in database.")
        print("  Use 'python roi_dashboard.py scan' to import from contest CSVs")
        print("  Or 'python roi_dashboard.py log' to add manually")
        return

    if last_n:
        dates = df['slate_date'].unique()[:last_n]
        df = df[df['slate_date'].isin(dates)]

    # Summary stats
    total_entries = len(df)
    total_fees = df['entry_fee'].sum()
    total_winnings = df['winnings'].sum()
    total_profit = df['profit'].sum()
    roi = (total_profit / total_fees * 100) if total_fees > 0 else 0
    n_slates = df['slate_date'].nunique()

    print(f"\n{'=' * 72}")
    print(f"  NHL DFS ROI DASHBOARD")
    print(f"{'=' * 72}")
    print(f"\n  Period: {df['slate_date'].min()} → {df['slate_date'].max()} ({n_slates} slates)")
    print(f"  Total Entries:  {total_entries}")
    print(f"  Total Fees:     ${total_fees:,.2f}")
    print(f"  Total Winnings: ${total_winnings:,.2f}")
    print(f"  Net Profit:     ${total_profit:+,.2f}")
    print(f"  ROI:            {roi:+.1f}%")

    # By contest type
    print(f"\n  {'─' * 50}")
    print(f"  BY CONTEST TYPE:")
    for ctype in ['GPP', 'SE']:
        subset = df[df['contest_type'] == ctype]
        if subset.empty:
            continue
        fees = subset['entry_fee'].sum()
        wins = subset['winnings'].sum()
        profit = subset['profit'].sum()
        ct_roi = (profit / fees * 100) if fees > 0 else 0
        print(f"    {ctype:<5} {len(subset):>4} entries  "
              f"Fees: ${fees:>8,.2f}  Won: ${wins:>8,.2f}  "
              f"P/L: ${profit:>+8,.2f}  ROI: {ct_roi:>+6.1f}%")

    # By date
    print(f"\n  {'─' * 50}")
    print(f"  BY DATE:")
    print(f"  {'Date':<12} {'Entries':>8} {'Fees':>10} {'Won':>10} {'P/L':>10} {'ROI':>8}")
    print(f"  {'─' * 62}")

    daily = df.groupby('slate_date').agg(
        entries=('entry_fee', 'count'),
        fees=('entry_fee', 'sum'),
        winnings=('winnings', 'sum'),
        profit=('profit', 'sum'),
    ).sort_index()

    cumulative_profit = 0
    for date, row in daily.iterrows():
        d_roi = (row['profit'] / row['fees'] * 100) if row['fees'] > 0 else 0
        cumulative_profit += row['profit']
        icon = "🟢" if row['profit'] > 0 else ("🔴" if row['profit'] < 0 else "⚪")
        print(f"  {date:<12} {row['entries']:>8} ${row['fees']:>9,.2f} "
              f"${row['winnings']:>9,.2f} ${row['profit']:>+9,.2f} {d_roi:>+7.1f}%  {icon}")

    print(f"  {'─' * 62}")
    print(f"  {'CUMULATIVE':<12} {total_entries:>8} ${total_fees:>9,.2f} "
          f"${total_winnings:>9,.2f} ${total_profit:>+9,.2f} {roi:>+7.1f}%")

    # Note about missing winnings
    zero_winnings = (df['winnings'] == 0).sum()
    if zero_winnings > 0:
        print(f"\n  ⚠ {zero_winnings} entries have $0 winnings (not yet updated).")
        print(f"    Use: python roi_dashboard.py log --date YYYY-MM-DD --contest \"name\" --fee X --winnings Y")

    print()


def export_roi(output: str):
    """Export all contest entries to CSV."""
    conn = get_connection()
    _ensure_roi_table(conn)
    df = pd.read_sql_query("SELECT * FROM contest_entries ORDER BY slate_date, contest_name", conn)
    conn.close()
    df.to_csv(output, index=False)
    print(f"  Exported {len(df)} entries to {output}")


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='NHL DFS ROI Dashboard')
    sub = parser.add_subparsers(dest='command')

    # set-username
    p_user = sub.add_parser('set-username', help='Set your DraftKings username')
    p_user.add_argument('username', type=str)

    # scan
    p_scan = sub.add_parser('scan', help='Scan contest CSVs for your entries')
    p_scan.add_argument('--username', type=str, default=None)

    # log
    p_log = sub.add_parser('log', help='Manually log a contest result')
    p_log.add_argument('--date', type=str, required=True)
    p_log.add_argument('--contest', type=str, required=True)
    p_log.add_argument('--fee', type=float, required=True)
    p_log.add_argument('--winnings', type=float, required=True)
    p_log.add_argument('--score', type=float, default=None)
    p_log.add_argument('--rank', type=int, default=None)
    p_log.add_argument('--type', type=str, default='GPP', choices=['GPP', 'SE', 'CASH'])

    # update (update winnings for a scanned entry)
    p_upd = sub.add_parser('update', help='Update winnings for a scanned entry')
    p_upd.add_argument('--date', type=str, required=True)
    p_upd.add_argument('--contest', type=str, required=True)
    p_upd.add_argument('--winnings', type=float, required=True)

    # report
    p_report = sub.add_parser('report', help='Show ROI dashboard')
    p_report.add_argument('--last', type=int, default=None)

    # export
    p_export = sub.add_parser('export', help='Export to CSV')
    p_export.add_argument('--output', type=str, default='roi_export.csv')

    args = parser.parse_args()

    if not args.command:
        # Default: show report
        print_roi_report()
        return

    if args.command == 'set-username':
        set_username(args.username)

    elif args.command == 'scan':
        username = args.username or get_username()
        if not username:
            print("  No username set. Use: python roi_dashboard.py set-username YourDKName")
            sys.exit(1)
        scan_all_contests(username)

    elif args.command == 'log':
        log_entry(args.date, args.contest, args.fee, args.winnings,
                 args.score, args.rank, args.type)

    elif args.command == 'update':
        update_winnings(args.date, args.contest, args.winnings)

    elif args.command == 'report':
        print_roi_report(last_n=args.last)

    elif args.command == 'export':
        export_roi(args.output)


if __name__ == '__main__':
    main()
