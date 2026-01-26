#!/usr/bin/env python3
"""
NHL DFS Projection Tool - Main CLI

Usage:
    python main.py                      # Generate projections for today's slate
    python main.py --date 2026-01-22    # Generate for specific date
    python main.py --salaries file.csv  # Use specific salary file
    python main.py --stacks             # Show stacking recommendations
    python main.py --show-injuries      # Display injury report
    python main.py --include-dtd        # Include Day-to-Day players
    python main.py --no-injuries        # Disable injury filtering
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

from data_pipeline import NHLDataPipeline
from projections import NHLProjectionModel
from optimizer import NHLLineupOptimizer
from config import calculate_skater_fantasy_points, calculate_goalie_fantasy_points, INJURY_STATUSES_EXCLUDE
from lines import LinesScraper, StackBuilder, print_team_lines, find_player_match


def normalize_position(pos: str) -> str:
    """
    Normalize NHL positions to DraftKings format.

    L, LW, R, RW -> W (Wing)
    C -> C (Center)
    D -> D (Defense)
    G -> G (Goalie)
    """
    if pd.isna(pos):
        return 'W'
    pos = str(pos).upper().strip()
    if pos in ('L', 'LW', 'R', 'RW'):
        return 'W'
    return pos


def normalize_positions_column(df: pd.DataFrame, col: str = 'position') -> pd.DataFrame:
    """Apply position normalization to a DataFrame column."""
    if col in df.columns:
        df[col] = df[col].apply(normalize_position)
    return df


def load_dk_salaries(csv_path: str) -> pd.DataFrame:
    """Load and parse DraftKings salary CSV."""
    print(f"Loading DK salaries from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize columns
    df = df.rename(columns={
        'Name': 'dk_name',
        'Roster Position': 'dk_position',
        'Salary': 'salary',
        'TeamAbbrev': 'team',
        'AvgPointsPerGame': 'dk_avg_fpts',
        'Position': 'position',
        'ID': 'dk_id',
    })

    # Parse position (first position listed) and normalize L/R to W
    if 'position' in df.columns:
        df['base_position'] = df['position'].str.upper()
        df = normalize_positions_column(df, 'position')
        df = normalize_positions_column(df, 'base_position')

    # Normalize team codes
    if 'team' in df.columns:
        df['team'] = df['team'].str.upper()

    print(f"  Loaded {len(df)} players")
    print(f"  Salary range: ${df['salary'].min():,} - ${df['salary'].max():,}")

    return df


def merge_projections_with_salaries(projections: pd.DataFrame,
                                     salaries: pd.DataFrame,
                                     player_type: str = 'skater') -> pd.DataFrame:
    """Merge our projections with DK salaries."""

    # Create matching keys
    proj = projections.copy()
    sal = salaries.copy()

    # Clean names for matching
    proj['name_clean'] = proj['name'].str.lower().str.strip()
    proj['name_clean'] = proj['name_clean'].str.replace(r'[^a-z\s]', '', regex=True)

    sal['name_clean'] = sal['dk_name'].str.lower().str.strip()
    sal['name_clean'] = sal['name_clean'].str.replace(r'[^a-z\s]', '', regex=True)

    # Rename DK position to avoid collision with projection position
    if 'position' in sal.columns:
        sal = sal.rename(columns={'position': 'dk_pos'})

    # Merge - include dk_pos (actual position) for roster eligibility
    merge_cols = ['name_clean', 'salary', 'dk_position', 'dk_id', 'dk_avg_fpts']
    if 'dk_pos' in sal.columns:
        merge_cols.append('dk_pos')

    merged = proj.merge(
        sal[merge_cols],
        on='name_clean',
        how='inner'
    )

    if len(merged) == 0:
        print(f"  Warning: No matches found for {player_type}s!")
        print(f"  Sample projection names: {proj['name'].head(5).tolist()}")
        print(f"  Sample salary names: {sal['dk_name'].head(5).tolist()}")
        return merged

    # Calculate value metrics
    merged['value'] = merged['projected_fpts'] / (merged['salary'] / 1000)
    merged['edge'] = merged['projected_fpts'] - merged['dk_avg_fpts']

    # Drop temp columns
    merged = merged.drop(columns=['name_clean'], errors='ignore')

    return merged


def print_projections_table(df: pd.DataFrame, title: str, n: int = 25):
    """Print formatted projections table."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    cols = ['name', 'team', 'position', 'salary', 'projected_fpts', 'dk_avg_fpts', 'edge', 'value']
    cols = [c for c in cols if c in df.columns]

    display_df = df.nlargest(n, 'projected_fpts')[cols].copy()

    # Normalize positions for display (L/R -> W)
    if 'position' in display_df.columns:
        display_df['position'] = display_df['position'].apply(normalize_position)

    # Format columns
    if 'salary' in display_df.columns:
        display_df['salary'] = display_df['salary'].apply(lambda x: f"${x:,}")
    if 'projected_fpts' in display_df.columns:
        display_df['projected_fpts'] = display_df['projected_fpts'].round(1)
    if 'dk_avg_fpts' in display_df.columns:
        display_df['dk_avg_fpts'] = display_df['dk_avg_fpts'].round(1)
    if 'edge' in display_df.columns:
        display_df['edge'] = display_df['edge'].round(1)
    if 'value' in display_df.columns:
        display_df['value'] = display_df['value'].round(2)

    # Rename for display
    display_df.columns = ['Name', 'Team', 'Pos', 'Salary', 'Proj', 'DK Avg', 'Edge', 'Value'][:len(cols)]

    print(display_df.to_string(index=False))


def print_value_plays(df: pd.DataFrame, title: str, n: int = 15):
    """Print top value plays."""
    print(f"\n{'=' * 80}")
    print(f" {title} (Sorted by Value)")
    print(f"{'=' * 80}")

    cols = ['name', 'team', 'position', 'salary', 'projected_fpts', 'value']
    cols = [c for c in cols if c in df.columns]

    # Filter to reasonable salary range for value plays
    value_df = df[df['salary'] <= 6500].copy()
    display_df = value_df.nlargest(n, 'value')[cols].copy()

    # Normalize positions for display (L/R -> W)
    if 'position' in display_df.columns:
        display_df['position'] = display_df['position'].apply(normalize_position)

    if 'salary' in display_df.columns:
        display_df['salary'] = display_df['salary'].apply(lambda x: f"${x:,}")
    if 'projected_fpts' in display_df.columns:
        display_df['projected_fpts'] = display_df['projected_fpts'].round(1)
    if 'value' in display_df.columns:
        display_df['value'] = display_df['value'].round(2)

    display_df.columns = ['Name', 'Team', 'Pos', 'Salary', 'Proj', 'Value'][:len(cols)]

    print(display_df.to_string(index=False))


def print_stacking_recommendations(stack_builder: StackBuilder, projections_df: pd.DataFrame, teams: list):
    """Print stacking recommendations for all teams on the slate."""
    print(f"\n{'=' * 80}")
    print(" STACKING RECOMMENDATIONS")
    print(f"{'=' * 80}")

    all_stacks = []

    for team in teams:
        stacks = stack_builder.get_best_stacks(team, projections_df)
        for stack in stacks:
            if stack.get('type') in ['PP1', 'Line1']:
                all_stacks.append(stack)

    # Sort by projected total
    all_stacks.sort(key=lambda x: x.get('projected_total', 0), reverse=True)

    for stack in all_stacks[:10]:
        stack_type = stack.get('type', '')
        team = stack.get('team', '')
        players = stack.get('players', [])
        proj = stack.get('projected_total', 0)
        corr = stack.get('correlation', 0)

        print(f"\n{team} {stack_type} (corr: {corr:.0%}, proj: {proj:.1f} pts)")
        for p in players:
            # Find matching player in projections
            match = find_player_match(p, projections_df['name'].tolist())
            if match:
                player_data = projections_df[projections_df['name'] == match].iloc[0]
                print(f"  - {match:<25} ${player_data['salary']:,}  {player_data['projected_fpts']:.1f} pts")
            else:
                print(f"  - {p:<25} (not in DK pool)")


def print_line_update_status(stack_builder: StackBuilder):
    """Print when line combinations were last updated."""
    from datetime import datetime as dt, timezone

    print(f"\n{'=' * 80}")
    print(" LINE COMBINATION UPDATE STATUS")
    print(f"{'=' * 80}")

    timestamps = stack_builder.get_update_timestamps()

    if not timestamps:
        print("  No update timestamps available")
        return

    # Parse and display timestamps
    now = dt.now(timezone.utc)
    stale_teams = []

    for team, ts in sorted(timestamps.items()):
        if not ts:
            print(f"  {team:<5} No update time available")
            continue

        try:
            # Parse ISO format timestamp (UTC)
            update_time = dt.fromisoformat(ts.replace('Z', '+00:00'))
            age = now - update_time
            hours_ago = age.total_seconds() / 3600

            # Convert to local time for display
            local_time = update_time.astimezone()

            if hours_ago < 0:
                age_str = "just now"
            elif hours_ago < 1:
                age_str = f"{int(age.total_seconds() / 60)} min ago"
            elif hours_ago < 24:
                age_str = f"{hours_ago:.1f} hrs ago"
            else:
                age_str = f"{int(hours_ago / 24)} days ago"
                stale_teams.append(team)

            print(f"  {team:<5} Updated {age_str:<15} ({local_time.strftime('%m/%d %I:%M %p')})")
        except Exception as e:
            print(f"  {team:<5} {ts}")

    if stale_teams:
        print(f"\n  WARNING: {', '.join(stale_teams)} have stale data (>24 hrs old)")


def print_injury_report(injuries: pd.DataFrame, teams: list = None):
    """
    Print injury report for slate teams.

    Args:
        injuries: DataFrame with injury data from MoneyPuck
        teams: Optional list of teams to filter (slate teams)
    """
    print(f"\n{'=' * 80}")
    print(" INJURY REPORT")
    print(f"{'=' * 80}")

    if injuries.empty:
        print("  No injury data available")
        return

    # Filter to slate teams if provided
    if teams and 'team' in injuries.columns:
        injuries = injuries[injuries['team'].isin(teams)]

    if injuries.empty:
        print("  No injured players on today's slate")
        return

    # Group by injury status
    for status in ['IR', 'IR-LT', 'IR-NR', 'O', 'DTD']:
        status_df = injuries[injuries.get('injury_status', '') == status]

        if status_df.empty:
            continue

        status_label = {
            'IR': 'Injured Reserve',
            'IR-LT': 'IR Long-Term',
            'IR-NR': 'IR Non-Roster',
            'O': 'Out',
            'DTD': 'Day-to-Day'
        }.get(status, status)

        print(f"\n  {status_label} ({len(status_df)} players):")
        print(f"  {'-' * 60}")

        # Sort by team
        status_df = status_df.sort_values('team')

        for _, row in status_df.iterrows():
            team = row.get('team', '???')
            name = row.get('player_name', 'Unknown')
            position = row.get('position', '?')
            games_to_miss = row.get('games_to_miss', '')
            return_date = row.get('return_date', '')

            games_str = f", ~{int(games_to_miss)} games" if pd.notna(games_to_miss) and games_to_miss else ""
            date_str = f", ETA: {return_date.strftime('%m/%d')}" if pd.notna(return_date) else ""

            print(f"    {team:<5} {name:<25} ({position}){games_str}{date_str}")

    # Summary
    total = len(injuries)
    severe = len(injuries[injuries['injury_status'].isin(INJURY_STATUSES_EXCLUDE)])
    dtd = len(injuries[injuries['injury_status'] == 'DTD'])

    print(f"\n  Summary: {total} total injured ({severe} out, {dtd} day-to-day)")


def print_confirmed_goalies(stack_builder: StackBuilder, projections_df: pd.DataFrame):
    """Print confirmed starting goalies."""
    print(f"\n{'=' * 80}")
    print(" CONFIRMED STARTING GOALIES")
    print(f"{'=' * 80}")

    goalies = stack_builder.get_all_starting_goalies()

    if not goalies:
        print("  No confirmed starters found")
        return

    for team, goalie_name in goalies.items():
        match = find_player_match(goalie_name, projections_df['name'].tolist())
        if match:
            player_data = projections_df[projections_df['name'] == match].iloc[0]
            print(f"  {team:<5} {match:<25} ${player_data['salary']:,}  {player_data['projected_fpts']:.1f} pts")
        else:
            print(f"  {team:<5} {goalie_name:<25} (not in DK pool)")


def print_lineup(lineup: pd.DataFrame):
    """Print optimized lineup."""
    print(f"\n{'=' * 80}")
    print(" OPTIMIZED LINEUP")
    print(f"{'=' * 80}")

    total_salary = lineup['salary'].sum()
    total_proj = lineup['projected_fpts'].sum()

    print(f"Total Salary: ${total_salary:,} / $50,000 (${50000 - total_salary:,} remaining)")
    print(f"Total Projected: {total_proj:.1f} pts")

    # Show stack info if available
    if 'stack_info' in lineup.columns and lineup['stack_info'].iloc[0]:
        print(f"Stacks: {lineup['stack_info'].iloc[0]}")

    print()

    print(f"{'Slot':<6} {'Name':<28} {'Team':<5} {'Salary':<9} {'Proj':<7} {'Value':<6}")
    print("-" * 70)

    # Sort by roster slot for display
    slot_order = {'G': 0, 'C1': 1, 'C2': 2, 'W1': 3, 'W2': 4, 'W3': 5, 'D1': 6, 'D2': 7, 'UTIL': 8}
    lineup = lineup.copy()
    if 'roster_slot' in lineup.columns:
        lineup['_slot_order'] = lineup['roster_slot'].map(lambda x: slot_order.get(x, 9))
        lineup = lineup.sort_values('_slot_order')
        slot_col = 'roster_slot'
    else:
        pos_order = {'C': 0, 'LW': 1, 'RW': 2, 'W': 2, 'L': 2, 'R': 2, 'D': 3, 'G': 4}
        lineup['_slot_order'] = lineup['position'].map(lambda x: pos_order.get(str(x).upper(), 5))
        lineup = lineup.sort_values('_slot_order')
        slot_col = 'position'

    for _, row in lineup.iterrows():
        print(f"{row[slot_col]:<6} {row['name']:<28} {row['team']:<5} "
              f"${row['salary']:<8,} {row['projected_fpts']:<7.1f} {row.get('value', 0):<6.2f}")


def export_projections(skaters: pd.DataFrame, goalies: pd.DataFrame, output_path: str):
    """Export projections to CSV."""
    # Combine and export
    skaters = skaters.copy()
    goalies = goalies.copy()
    skaters['player_type'] = 'skater'
    goalies['player_type'] = 'goalie'

    combined = pd.concat([skaters, goalies], ignore_index=True)

    export_cols = ['name', 'team', 'position', 'salary', 'projected_fpts',
                   'dk_avg_fpts', 'edge', 'value', 'floor', 'ceiling', 'player_type']
    export_cols = [c for c in export_cols if c in combined.columns]

    combined[export_cols].to_csv(output_path, index=False)
    print(f"\nProjections exported to: {output_path}")


def export_lineup_for_dk(lineup: pd.DataFrame, output_path: str):
    """
    Export lineup in DraftKings CSV upload format.

    DK format requires columns: C, C, W, W, W, D, D, G, UTIL
    with player IDs or names.
    """
    # Map roster slots to DK positions
    slot_to_dk = {
        'C1': 'C', 'C2': 'C',
        'W1': 'W', 'W2': 'W', 'W3': 'W',
        'D1': 'D', 'D2': 'D',
        'G': 'G',
        'UTIL': 'UTIL'
    }

    # Build DK format row
    dk_row = {}
    for _, player in lineup.iterrows():
        slot = player.get('roster_slot', '')
        dk_pos = slot_to_dk.get(slot, slot)

        # Handle duplicate positions (C, C, W, W, W)
        pos_key = dk_pos
        counter = 1
        while pos_key in dk_row:
            counter += 1
            pos_key = f"{dk_pos}_{counter}"

        # Use DK ID if available, otherwise name
        player_id = player.get('dk_id', player['name'])
        dk_row[pos_key] = player_id

    # Create DataFrame in DK order
    dk_order = ['C', 'C_2', 'W', 'W_2', 'W_3', 'D', 'D_2', 'G', 'UTIL']
    dk_df = pd.DataFrame([dk_row])

    # Rename columns to simple format
    dk_df.columns = ['C', 'C', 'W', 'W', 'W', 'D', 'D', 'G', 'UTIL'][:len(dk_df.columns)]

    dk_df.to_csv(output_path, index=False)
    print(f"\nLineup exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='NHL DFS Projection Tool')
    parser.add_argument('--date', type=str, default=None,
                        help='Date to project for (YYYY-MM-DD)')
    parser.add_argument('--salaries', type=str, default=None,
                        help='Path to DraftKings salary CSV')
    parser.add_argument('--export', type=str, default=None,
                        help='Export projections to CSV')
    parser.add_argument('--lineups', type=int, default=1,
                        help='Number of lineups to generate')
    parser.add_argument('--stacks', action='store_true',
                        help='Show stacking recommendations from line combinations')
    parser.add_argument('--no-stacks', action='store_true',
                        help='Disable stacking boost in optimizer')
    parser.add_argument('--force-stack', type=str, default=None,
                        help='Force a specific stack type (PP1, Line1)')

    # Injury filtering options
    parser.add_argument('--show-injuries', action='store_true',
                        help='Display injury report for slate teams')
    parser.add_argument('--include-dtd', action='store_true',
                        help='Include Day-to-Day players in projections (normally excluded)')
    parser.add_argument('--no-injuries', action='store_true',
                        help='Disable injury filtering (include all players)')

    # Advanced stats options
    parser.add_argument('--no-advanced', action='store_true',
                        help='Skip fetching Natural Stat Trick advanced stats')

    args = parser.parse_args()

    # Determine date
    target_date = args.date or datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'#' * 80}")
    print(f"  NHL DFS PROJECTIONS - {target_date}")
    print(f"{'#' * 80}")

    # Find salary file
    if args.salaries:
        salary_path = args.salaries
    else:
        # Look for most recent DK salary file in current directory
        project_dir = Path(__file__).parent
        salary_files = list(project_dir.glob('DKSalaries*.csv'))
        if salary_files:
            salary_path = str(sorted(salary_files)[-1])  # Most recent
            print(f"\nAuto-detected salary file: {salary_path}")
        else:
            print("\nNo DraftKings salary file found. Run with --salaries <path>")
            sys.exit(1)

    # Load DK salaries
    dk_salaries = load_dk_salaries(salary_path)

    # Get slate teams for filtering
    slate_teams = list(dk_salaries['team'].unique())

    # Fetch NHL data
    print("\nFetching NHL data...")
    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(
        include_game_logs=False,
        include_injuries=not args.no_injuries,
        include_advanced_stats=not args.no_advanced
    )

    # Show injury report if requested
    if args.show_injuries and 'injuries' in data and not data['injuries'].empty:
        print_injury_report(data['injuries'], slate_teams)

    # Generate projections
    print("\nGenerating projections...")
    model = NHLProjectionModel()
    projections = model.generate_projections(
        data,
        target_date=target_date,
        filter_injuries=not args.no_injuries,
        include_dtd=not args.include_dtd  # If --include-dtd, don't filter DTD
    )

    # Separate goalies from skaters in DK salaries
    dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'W', 'D'])]
    dk_goalies = dk_salaries[dk_salaries['position'] == 'G']

    # Add position column to goalies if missing
    if 'position' not in projections['goalies'].columns:
        projections['goalies']['position'] = 'G'

    # Merge projections with salaries
    print("\nMerging projections with DK salaries...")
    skaters_merged = merge_projections_with_salaries(
        projections['skaters'], dk_skaters, 'skater'
    )
    goalies_merged = merge_projections_with_salaries(
        projections['goalies'], dk_goalies, 'goalie'
    )

    print(f"  Matched {len(skaters_merged)} skaters")
    print(f"  Matched {len(goalies_merged)} goalies")

    # Combine pools
    player_pool = pd.concat([skaters_merged, goalies_merged], ignore_index=True)

    # Fetch line combinations if stacking enabled
    stack_builder = None

    if not args.no_stacks or args.stacks:
        print("\nFetching line combinations from DailyFaceoff...")
        scraper = LinesScraper(rate_limit=0.5)
        lines_data = scraper.get_multiple_teams(slate_teams)
        stack_builder = StackBuilder(lines_data)

        # Show line update timestamps
        print_line_update_status(stack_builder)

        # Show confirmed goalies
        print_confirmed_goalies(stack_builder, player_pool)

        # Show stacking recommendations if requested
        if args.stacks:
            print_stacking_recommendations(stack_builder, player_pool, slate_teams)

    # Print projections
    if len(skaters_merged) > 0:
        print_projections_table(skaters_merged, "TOP SKATER PROJECTIONS", n=25)
        print_value_plays(skaters_merged, "VALUE SKATER PLAYS", n=15)

    if len(goalies_merged) > 0:
        print_projections_table(goalies_merged, "TOP GOALIE PROJECTIONS", n=10)

    # Generate optimized lineup
    lineups = []
    if len(skaters_merged) > 0 and len(goalies_merged) > 0:
        print("\nOptimizing lineup...")

        # Pass stack builder to optimizer if available
        optimizer = NHLLineupOptimizer(stack_builder=stack_builder if not args.no_stacks else None)

        lineups = optimizer.optimize_lineup(
            player_pool,
            n_lineups=args.lineups,
            randomness=0.05 if args.lineups > 1 else 0,
            stack_teams=[args.force_stack] if args.force_stack else None
        )

        for i, lineup in enumerate(lineups):
            if args.lineups > 1:
                print(f"\n--- Lineup {i+1} ---")
            print_lineup(lineup)

    # Export if requested
    if args.export:
        export_projections(skaters_merged, goalies_merged, args.export)

        # Also export lineups if generated
        if lineups:
            lineup_path = args.export.replace('.csv', '_lineups.csv')
            export_lineup_for_dk(lineups[0], lineup_path)

    print(f"\n{'#' * 80}")
    print("  PROJECTION COMPLETE")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
