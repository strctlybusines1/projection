"""
Historical SQLite Database for NHL DFS Pipeline.

Stores projections, actual results, backtest metrics, and contest entries
for long-term tracking and analysis.

Usage:
    # Auto-ingest after a slate:
    python history_db.py ingest --date 2026-02-25

    # Ingest all existing backtest CSVs:
    python history_db.py backfill

    # Query historical accuracy:
    python history_db.py report
    python history_db.py report --last 10

    # Export to CSV:
    python history_db.py export --output history_export.csv

Schema:
    projections: One row per player per slate (our projection vs actual)
    slates:      One row per slate date (aggregate metrics)
    contests:    One row per contest entry (optional - contest results)
"""

import sqlite3
import os
import re
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

# ================================================================
#  Database Path
# ================================================================

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "nhl_dfs_history.db"


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Get database connection, creating DB and tables if needed."""
    path = db_path or str(DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS slates (
            slate_date     TEXT PRIMARY KEY,
            n_games        INTEGER,
            n_players      INTEGER,
            n_goalies      INTEGER,
            skater_mae     REAL,
            goalie_mae     REAL,
            overall_mae    REAL,
            skater_corr    REAL,
            goalie_corr    REAL,
            skater_bias    REAL,
            goalie_bias    REAL,
            model_version  TEXT,
            notes          TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS projections (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_date     TEXT NOT NULL,
            name           TEXT NOT NULL,
            team           TEXT,
            position       TEXT,
            salary         INTEGER,
            projected_fpts REAL,
            actual_fpts    REAL,
            error          REAL,
            abs_error      REAL,
            dk_avg_fpts    REAL,
            edge           REAL,
            value          REAL,
            predicted_own  REAL,
            player_type    TEXT,
            FOREIGN KEY (slate_date) REFERENCES slates(slate_date),
            UNIQUE(slate_date, name)
        );

        CREATE TABLE IF NOT EXISTS contests (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_date     TEXT NOT NULL,
            contest_name   TEXT,
            entry_fee      REAL,
            field_size     INTEGER,
            our_score      REAL,
            our_rank       INTEGER,
            winning_score  REAL,
            cash_line      REAL,
            profit         REAL,
            lineup_hash    TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_proj_date ON projections(slate_date);
        CREATE INDEX IF NOT EXISTS idx_proj_name ON projections(name);
        CREATE INDEX IF NOT EXISTS idx_proj_team ON projections(team);
        CREATE INDEX IF NOT EXISTS idx_proj_pos  ON projections(position);
        CREATE INDEX IF NOT EXISTS idx_contest_date ON contests(slate_date);
    """)
    conn.commit()


# ================================================================
#  Ingest Functions
# ================================================================

def ingest_backtest_csv(conn: sqlite3.Connection, csv_path: str,
                        slate_date: str = None, model_version: str = None):
    """
    Ingest a batch_backtest_details.csv or individual backtest CSV.

    Expected columns: name, team, actual_fpts, projected_fpts, error, position, date
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Determine date column
    if 'date' in df.columns:
        date_col = 'date'
    elif 'slate_date' in df.columns:
        date_col = 'slate_date'
    else:
        if slate_date:
            df['date'] = slate_date
            date_col = 'date'
        else:
            print(f"  Warning: No date column in {csv_path}, skipping")
            return 0

    dates = df[date_col].unique()
    total_inserted = 0

    for date in dates:
        date_df = df[df[date_col] == date].copy()
        _slate_date = str(date)

        # Calculate metrics
        has_actual = 'actual_fpts' in date_df.columns and date_df['actual_fpts'].notna().any()

        if has_actual:
            date_df['error'] = date_df['projected_fpts'] - date_df['actual_fpts']
            date_df['abs_error'] = date_df['error'].abs()

            skaters = date_df[date_df['position'].str.upper() != 'G']
            goalies = date_df[date_df['position'].str.upper() == 'G']

            skater_mae = skaters['abs_error'].mean() if len(skaters) > 0 else None
            goalie_mae = goalies['abs_error'].mean() if len(goalies) > 0 else None
            overall_mae = date_df['abs_error'].mean()

            skater_corr = skaters[['projected_fpts', 'actual_fpts']].corr().iloc[0, 1] \
                if len(skaters) > 5 else None
            goalie_corr = goalies[['projected_fpts', 'actual_fpts']].corr().iloc[0, 1] \
                if len(goalies) > 2 else None

            skater_bias = skaters['error'].mean() if len(skaters) > 0 else None
            goalie_bias = goalies['error'].mean() if len(goalies) > 0 else None
        else:
            skater_mae = goalie_mae = overall_mae = None
            skater_corr = goalie_corr = None
            skater_bias = goalie_bias = None

        n_skaters = len(date_df[date_df['position'].str.upper() != 'G']) if 'position' in date_df.columns else len(date_df)
        n_goalies = len(date_df[date_df['position'].str.upper() == 'G']) if 'position' in date_df.columns else 0

        # Upsert slate
        conn.execute("""
            INSERT INTO slates (slate_date, n_players, n_goalies, skater_mae, goalie_mae,
                                overall_mae, skater_corr, goalie_corr, skater_bias, goalie_bias,
                                model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slate_date) DO UPDATE SET
                n_players=excluded.n_players, n_goalies=excluded.n_goalies,
                skater_mae=excluded.skater_mae, goalie_mae=excluded.goalie_mae,
                overall_mae=excluded.overall_mae, skater_corr=excluded.skater_corr,
                goalie_corr=excluded.goalie_corr, skater_bias=excluded.skater_bias,
                goalie_bias=excluded.goalie_bias, model_version=excluded.model_version
        """, (_slate_date, n_skaters, n_goalies, skater_mae, goalie_mae,
              overall_mae, skater_corr, goalie_corr, skater_bias, goalie_bias,
              model_version))

        # Insert player projections
        for _, row in date_df.iterrows():
            try:
                conn.execute("""
                    INSERT INTO projections (slate_date, name, team, position,
                                            projected_fpts, actual_fpts, error, abs_error,
                                            player_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(slate_date, name) DO UPDATE SET
                        projected_fpts=excluded.projected_fpts,
                        actual_fpts=excluded.actual_fpts,
                        error=excluded.error, abs_error=excluded.abs_error
                """, (
                    _slate_date,
                    row.get('name', ''),
                    row.get('team', ''),
                    row.get('position', ''),
                    row.get('projected_fpts'),
                    row.get('actual_fpts'),
                    row.get('error'),
                    row.get('abs_error') if 'abs_error' in row else (abs(row['error']) if pd.notna(row.get('error')) else None),
                    'goalie' if str(row.get('position', '')).upper() == 'G' else 'skater',
                ))
                total_inserted += 1
            except Exception as e:
                print(f"  Warning: Failed to insert {row.get('name', '?')}: {e}")

    conn.commit()
    return total_inserted


def ingest_projection_csv(conn: sqlite3.Connection, csv_path: str, slate_date: str):
    """
    Ingest a daily projection CSV (pre-game, no actuals yet).

    Expected columns: name, team, position, salary, projected_fpts, dk_avg_fpts,
                      edge, value, predicted_ownership, dk_id
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT INTO projections (slate_date, name, team, position, salary,
                                        projected_fpts, dk_avg_fpts, edge, value,
                                        predicted_own, player_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slate_date, name) DO UPDATE SET
                    salary=excluded.salary, projected_fpts=excluded.projected_fpts,
                    dk_avg_fpts=excluded.dk_avg_fpts, edge=excluded.edge,
                    value=excluded.value, predicted_own=excluded.predicted_own
            """, (
                slate_date,
                row.get('name', ''),
                row.get('team', ''),
                row.get('position', ''),
                int(row['salary']) if pd.notna(row.get('salary')) else None,
                row.get('projected_fpts'),
                row.get('dk_avg_fpts'),
                row.get('edge'),
                row.get('value'),
                row.get('predicted_ownership'),
                'goalie' if str(row.get('position', '')).upper() == 'G' else 'skater',
            ))
            inserted += 1
        except Exception as e:
            pass

    conn.commit()
    return inserted


# ================================================================
#  Backfill From Existing Files
# ================================================================

def backfill_all(conn: sqlite3.Connection, project_dir: str = None):
    """Ingest all existing backtest and projection files."""
    project_dir = project_dir or str(Path(__file__).parent)
    backtest_dir = os.path.join(project_dir, "backtests")
    proj_dir = os.path.join(project_dir, "daily_projections")

    total = 0

    # 1. batch_backtest_details.csv (multi-date backtest results)
    batch_csv = os.path.join(backtest_dir, "batch_backtest_details.csv")
    if os.path.exists(batch_csv):
        n = ingest_backtest_csv(conn, batch_csv)
        print(f"  Ingested {n} rows from batch_backtest_details.csv")
        total += n

    # 2. Individual backtest xlsx files (try to find matching CSVs or skip)
    # The xlsx files have weird formatting — rely on batch CSV instead

    # 3. Daily projection CSVs
    if os.path.exists(proj_dir):
        proj_files = sorted(glob.glob(os.path.join(proj_dir, "*NHLprojections_*.csv")))
        for pf in proj_files:
            fname = os.path.basename(pf)
            # Parse date from filename: 02_05_26NHLprojections_20260205_193900.csv
            date_match = re.match(r'(\d{2})_(\d{2})_(\d{2})NHL', fname)
            if date_match:
                m, d, y = date_match.groups()
                slate_date = f"20{y}-{m}-{d}"
                n = ingest_projection_csv(conn, pf, slate_date)
                if n > 0:
                    print(f"  Ingested {n} projections for {slate_date}")
                    total += n

    print(f"\n  Total rows ingested: {total}")
    return total


# ================================================================
#  Reports
# ================================================================

def print_slate_report(conn: sqlite3.Connection, last_n: int = None):
    """Print accuracy report across slates."""
    query = "SELECT * FROM slates ORDER BY slate_date DESC"
    if last_n:
        query += f" LIMIT {last_n}"

    df = pd.read_sql_query(query, conn)
    if df.empty:
        print("  No slate data in database.")
        return

    print(f"\n{'=' * 85}")
    print("  HISTORICAL SLATE ACCURACY")
    print(f"{'=' * 85}")
    print(f"  {'Date':<12} {'Players':>8} {'Goalies':>8} {'SK MAE':>8} {'G MAE':>8} "
          f"{'Overall':>8} {'SK Corr':>8} {'SK Bias':>8}")
    print(f"  {'-' * 80}")

    for _, row in df.iterrows():
        sk_mae = f"{row['skater_mae']:.2f}" if pd.notna(row['skater_mae']) else "—"
        g_mae = f"{row['goalie_mae']:.2f}" if pd.notna(row['goalie_mae']) else "—"
        overall = f"{row['overall_mae']:.2f}" if pd.notna(row['overall_mae']) else "—"
        sk_corr = f"{row['skater_corr']:.3f}" if pd.notna(row['skater_corr']) else "—"
        sk_bias = f"{row['skater_bias']:+.2f}" if pd.notna(row['skater_bias']) else "—"
        print(f"  {row['slate_date']:<12} {row['n_players'] or 0:>8} {row['n_goalies'] or 0:>8} "
              f"{sk_mae:>8} {g_mae:>8} {overall:>8} {sk_corr:>8} {sk_bias:>8}")

    # Aggregate
    if len(df) > 1:
        print(f"  {'-' * 80}")
        agg_mae = df['overall_mae'].dropna().mean()
        agg_sk_corr = df['skater_corr'].dropna().mean()
        agg_sk_bias = df['skater_bias'].dropna().mean()
        n_slates = len(df)
        print(f"  {'AVERAGE':<12} {'':<8} {'':<8} {'':<8} {'':<8} "
              f"{agg_mae:>8.2f} {agg_sk_corr:>8.3f} {agg_sk_bias:>+8.2f}  ({n_slates} slates)")

    print()


def print_player_history(conn: sqlite3.Connection, player_name: str):
    """Print projection history for a specific player."""
    df = pd.read_sql_query(
        "SELECT * FROM projections WHERE name LIKE ? ORDER BY slate_date",
        conn, params=(f"%{player_name}%",)
    )
    if df.empty:
        print(f"  No data found for '{player_name}'")
        return

    print(f"\n  History for: {df.iloc[0]['name']}")
    print(f"  {'Date':<12} {'Proj':>6} {'Actual':>7} {'Error':>7} {'Salary':>7}")
    print(f"  {'-' * 45}")
    for _, row in df.iterrows():
        actual = f"{row['actual_fpts']:.1f}" if pd.notna(row['actual_fpts']) else "—"
        error = f"{row['error']:+.1f}" if pd.notna(row['error']) else "—"
        salary = f"${row['salary']:,}" if pd.notna(row.get('salary')) else "—"
        print(f"  {row['slate_date']:<12} {row['projected_fpts']:>6.1f} {actual:>7} "
              f"{error:>7} {salary:>7}")


def print_position_breakdown(conn: sqlite3.Connection):
    """Show accuracy by position."""
    df = pd.read_sql_query("""
        SELECT position, 
               COUNT(*) as n,
               AVG(abs_error) as mae,
               AVG(error) as bias,
               AVG(projected_fpts) as avg_proj,
               AVG(actual_fpts) as avg_actual
        FROM projections
        WHERE actual_fpts IS NOT NULL
        GROUP BY position
        ORDER BY position
    """, conn)

    if df.empty:
        print("  No actual data to analyze.")
        return

    print(f"\n{'=' * 60}")
    print("  ACCURACY BY POSITION")
    print(f"{'=' * 60}")
    print(f"  {'Pos':<6} {'N':>6} {'MAE':>7} {'Bias':>7} {'Avg Proj':>9} {'Avg Act':>8}")
    print(f"  {'-' * 50}")
    for _, row in df.iterrows():
        print(f"  {row['position']:<6} {row['n']:>6} {row['mae']:>7.2f} {row['bias']:>+7.2f} "
              f"{row['avg_proj']:>9.2f} {row['avg_actual']:>8.2f}")
    print()


def export_history(conn: sqlite3.Connection, output_path: str):
    """Export full projection history to CSV."""
    df = pd.read_sql_query("""
        SELECT p.*, s.overall_mae as slate_mae, s.skater_corr as slate_corr
        FROM projections p
        LEFT JOIN slates s ON p.slate_date = s.slate_date
        ORDER BY p.slate_date, p.name
    """, conn)
    df.to_csv(output_path, index=False)
    print(f"  Exported {len(df)} rows to {output_path}")


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='NHL DFS Historical Database')
    sub = parser.add_subparsers(dest='command')

    # ingest
    p_ingest = sub.add_parser('ingest', help='Ingest a backtest CSV')
    p_ingest.add_argument('--csv', type=str, help='Path to CSV file')
    p_ingest.add_argument('--date', type=str, help='Slate date (YYYY-MM-DD)')
    p_ingest.add_argument('--projection', type=str, help='Path to projection CSV (pre-game)')

    # backfill
    p_backfill = sub.add_parser('backfill', help='Ingest all existing backtest/projection files')

    # report
    p_report = sub.add_parser('report', help='Print accuracy report')
    p_report.add_argument('--last', type=int, default=None, help='Show last N slates')
    p_report.add_argument('--player', type=str, default=None, help='Show history for player')
    p_report.add_argument('--positions', action='store_true', help='Show position breakdown')

    # export
    p_export = sub.add_parser('export', help='Export history to CSV')
    p_export.add_argument('--output', type=str, default='history_export.csv')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    conn = get_connection()

    if args.command == 'ingest':
        if args.csv:
            n = ingest_backtest_csv(conn, args.csv, slate_date=args.date)
            print(f"  Ingested {n} rows from {args.csv}")
        elif args.projection and args.date:
            n = ingest_projection_csv(conn, args.projection, args.date)
            print(f"  Ingested {n} projections for {args.date}")
        else:
            print("  Provide --csv or --projection + --date")

    elif args.command == 'backfill':
        backfill_all(conn)

    elif args.command == 'report':
        if args.player:
            print_player_history(conn, args.player)
        elif args.positions:
            print_position_breakdown(conn)
        else:
            print_slate_report(conn, last_n=args.last)

    elif args.command == 'export':
        export_history(conn, args.output)

    conn.close()


if __name__ == "__main__":
    main()
