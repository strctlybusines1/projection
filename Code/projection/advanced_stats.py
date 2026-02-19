"""
Advanced Stats Engine — Compute per-player and per-team daily stats from PBP data.

Replaces NST dependency with our own metrics computed from NHL API play-by-play:
  - ixG (individual expected goals) per game
  - On-ice xGF%
  - High-danger shot rates (shots from <20 ft)
  - PP/EV deployment (% of PP shots, PP xG)
  - Shot generation rates (shots/60, xG/60)
  - Team defensive quality (xGA/game, HD chances against/game)

All stats are computed with rolling windows (5g, 10g, season) matching
the existing feature structure in our projection models.

Author: Claude
Date: 2026-02-18
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'data' / 'nhl_dfs_history.db'

# High-danger zone: shots within ~20 feet of net (distance < 20)
HD_DISTANCE_THRESHOLD = 25  # feet (conservative, ~inner slot)
MED_DISTANCE_THRESHOLD = 40  # medium danger zone


# ==============================================================================
# CORE: PER-PLAYER PER-GAME STATS FROM PBP
# ==============================================================================

def compute_player_game_stats(min_date: str = None, max_date: str = None) -> pd.DataFrame:
    """
    Aggregate PBP shot data into per-player per-game stat lines.

    Returns DataFrame with columns:
        player_id, player_name, team, position, game_date, game_id,
        shots, goals, xg, hd_shots, hd_goals, hd_xg,
        pp_shots, pp_goals, pp_xg, ev_shots, ev_goals, ev_xg,
        is_goal (for each shot — no, we aggregate),
        etc.
    """
    conn = sqlite3.connect(DB_PATH)

    query = "SELECT * FROM pbp_shots WHERE xg IS NOT NULL OR xg_v2 IS NOT NULL"
    if min_date:
        query += f" AND game_date >= '{min_date}'"
    if max_date:
        query += f" AND game_date <= '{max_date}'"

    shots = pd.read_sql(query, conn)
    conn.close()

    if shots.empty:
        print("No PBP shots found with xG values")
        return pd.DataFrame()

    # Prefer xg_v2 (stratified, 602K training) over xg_v1
    if 'xg_v2' in shots.columns:
        shots['xg'] = shots['xg_v2'].fillna(shots['xg'])

    print(f"Computing player game stats from {len(shots)} shots across "
          f"{shots['game_id'].nunique()} games (using xG v2)...")

    # Classify shot danger zones
    shots['is_hd'] = (shots['distance'] < HD_DISTANCE_THRESHOLD).astype(int)
    shots['is_md'] = ((shots['distance'] >= HD_DISTANCE_THRESHOLD) &
                       (shots['distance'] < MED_DISTANCE_THRESHOLD)).astype(int)

    # Aggregate per player per game
    agg = shots.groupby(['shooter_id', 'shooter_name', 'shooter_team_abbrev',
                          'shooter_position', 'game_date', 'game_id']).agg(
        # Total shooting
        shots=('is_goal', 'count'),
        goals=('is_goal', 'sum'),
        xg=('xg', 'sum'),

        # High-danger
        hd_shots=('is_hd', 'sum'),
        hd_xg=('xg', lambda x: x[shots.loc[x.index, 'is_hd'] == 1].sum()),

        # Medium-danger
        md_shots=('is_md', 'sum'),

        # Power play
        pp_shots=('is_pp', 'sum'),
        pp_xg=('xg', lambda x: x[shots.loc[x.index, 'is_pp'] == 1].sum()),
        pp_goals=('is_goal', lambda x: x[shots.loc[x.index, 'is_pp'] == 1].sum()),

        # Even strength (5v5)
        ev_shots=('is_5v5', 'sum'),
        ev_xg=('xg', lambda x: x[shots.loc[x.index, 'is_5v5'] == 1].sum()),
        ev_goals=('is_goal', lambda x: x[shots.loc[x.index, 'is_5v5'] == 1].sum()),

        # Rebound/rush xG
        rebound_xg=('xg', lambda x: x[shots.loc[x.index, 'is_rebound'] == 1].sum()),
        rush_xg=('xg', lambda x: x[shots.loc[x.index, 'is_rush'] == 1].sum()),

        # Avg shot distance
        avg_distance=('distance', 'mean'),
    ).reset_index()

    # Rename columns
    agg = agg.rename(columns={
        'shooter_id': 'player_id',
        'shooter_name': 'player_name',
        'shooter_team_abbrev': 'team',
        'shooter_position': 'position',
    })

    agg['game_date'] = pd.to_datetime(agg['game_date'])
    agg = agg.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    print(f"  {len(agg)} player-game records for {agg['player_id'].nunique()} players")
    return agg


def compute_team_game_stats(min_date: str = None, max_date: str = None) -> pd.DataFrame:
    """
    Aggregate PBP shot data into per-team per-game stats (both for and against).
    """
    conn = sqlite3.connect(DB_PATH)

    query = "SELECT * FROM pbp_shots WHERE xg IS NOT NULL OR xg_v2 IS NOT NULL"
    if min_date:
        query += f" AND game_date >= '{min_date}'"
    if max_date:
        query += f" AND game_date <= '{max_date}'"

    shots = pd.read_sql(query, conn)

    # Get game info for opponent mapping
    games = pd.read_sql("SELECT * FROM pbp_games", conn)
    conn.close()

    if shots.empty:
        return pd.DataFrame()

    # Prefer xg_v2 (stratified, 602K training) over xg_v1
    if 'xg_v2' in shots.columns:
        shots['xg'] = shots['xg_v2'].fillna(shots['xg'])

    shots['is_hd'] = (shots['distance'] < HD_DISTANCE_THRESHOLD).astype(int)

    # Build game-team-opponent map
    game_teams = {}
    for _, g in games.iterrows():
        gid = g['game_id']
        game_teams[(gid, g['home_abbrev'])] = g['away_abbrev']
        game_teams[(gid, g['away_abbrev'])] = g['home_abbrev']

    # Shots FOR (offensive stats)
    shots_for = shots.groupby(['shooter_team_abbrev', 'game_date', 'game_id']).agg(
        sf=('is_goal', 'count'),
        gf=('is_goal', 'sum'),
        xgf=('xg', 'sum'),
        hd_sf=('is_hd', 'sum'),
        pp_sf=('is_pp', 'sum'),
        pp_xgf=('xg', lambda x: x[shots.loc[x.index, 'is_pp'] == 1].sum()),
    ).reset_index().rename(columns={'shooter_team_abbrev': 'team'})

    # Shots AGAINST (defensive stats) — shots by opponent team
    # We need to figure out which team was defending
    shots['defending_team'] = shots.apply(
        lambda row: game_teams.get((row['game_id'], row['shooter_team_abbrev']), ''),
        axis=1
    )

    shots_against = shots.groupby(['defending_team', 'game_date', 'game_id']).agg(
        sa=('is_goal', 'count'),
        ga=('is_goal', 'sum'),
        xga=('xg', 'sum'),
        hd_sa=('is_hd', 'sum'),
    ).reset_index().rename(columns={'defending_team': 'team'})

    # Merge for and against
    team_stats = shots_for.merge(shots_against, on=['team', 'game_date', 'game_id'], how='outer')
    team_stats = team_stats.fillna(0)

    # Compute ratios
    team_stats['xgf_pct'] = team_stats['xgf'] / (team_stats['xgf'] + team_stats['xga'] + 1e-6) * 100
    team_stats['sf_pct'] = team_stats['sf'] / (team_stats['sf'] + team_stats['sa'] + 1e-6) * 100
    team_stats['hd_sf_pct'] = team_stats['hd_sf'] / (team_stats['hd_sf'] + team_stats['hd_sa'] + 1e-6) * 100

    team_stats['game_date'] = pd.to_datetime(team_stats['game_date'])
    team_stats = team_stats.sort_values(['team', 'game_date']).reset_index(drop=True)

    print(f"Team game stats: {len(team_stats)} team-games for {team_stats['team'].nunique()} teams")
    return team_stats


# ==============================================================================
# ROLLING FEATURES (matches existing model feature structure)
# ==============================================================================

def compute_rolling_advanced_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling window advanced stats per player.

    Creates features matching our projection model structure:
      - rolling_ixg_5g, rolling_ixg_10g: Individual xG rolling avg
      - rolling_hd_shots_5g: High-danger shot rate
      - rolling_pp_xg_5g: PP xG rate
      - ixg_ewm: Exponentially weighted ixG
    """
    df = player_stats.copy()
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    # Rolling windows per player
    for window in [5, 10]:
        for col in ['xg', 'hd_shots', 'md_shots', 'pp_xg', 'ev_xg', 'shots', 'goals',
                    'rebound_xg', 'rush_xg']:
            df[f'rolling_{col}_{window}g'] = (
                df.groupby('player_id')[col]
                .rolling(window=window, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )

    # EWM for ixG
    df['ixg_ewm'] = (
        df.groupby('player_id')['xg']
        .ewm(halflife=15, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    # Season averages
    for col in ['xg', 'hd_shots', 'md_shots', 'pp_xg', 'ev_xg', 'shots', 'goals',
                'rebound_xg', 'rush_xg']:
        cumsum = df.groupby('player_id')[col].cumsum()
        gp = df.groupby('player_id').cumcount() + 1
        df[f'season_avg_{col}'] = cumsum / gp

    # PP share: what % of player's xG comes from PP
    df['pp_xg_share'] = df['pp_xg'] / (df['xg'] + 1e-6)
    df['rolling_pp_xg_share_5g'] = (
        df.groupby('player_id')['pp_xg_share']
        .rolling(window=5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    return df


def compute_rolling_team_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling team-level stats for opponent quality features."""
    df = team_stats.copy()
    df = df.sort_values(['team', 'game_date']).reset_index(drop=True)

    for window in [5, 10]:
        for col in ['xgf', 'xga', 'xgf_pct', 'sa', 'hd_sa', 'ga']:
            df[f'rolling_{col}_{window}g'] = (
                df.groupby('team')[col]
                .rolling(window=window, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )

    # Season averages
    for col in ['xgf', 'xga', 'xgf_pct', 'sa', 'hd_sa']:
        cumsum = df.groupby('team')[col].cumsum()
        gp = df.groupby('team').cumcount() + 1
        df[f'season_avg_{col}'] = cumsum / gp

    return df


# ==============================================================================
# FEATURE EXPORT (for model integration)
# ==============================================================================

def get_player_features_for_date(date_str: str, player_stats_rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Get the most recent advanced stats features for each player as of a date.

    These features can be merged into the projection model player pool.
    """
    target_date = pd.to_datetime(date_str)

    # Get most recent game for each player before target date
    prior = player_stats_rolling[player_stats_rolling['game_date'] < target_date]
    if prior.empty:
        return pd.DataFrame()

    latest = prior.groupby('player_id').tail(1).copy()

    # Select features for model
    feature_cols = [
        'player_id', 'player_name', 'team', 'position',
        'rolling_xg_5g', 'rolling_xg_10g',
        'rolling_hd_shots_5g', 'rolling_hd_shots_10g',
        'rolling_md_shots_5g', 'rolling_md_shots_10g',
        'rolling_rebound_xg_5g', 'rolling_rebound_xg_10g',
        'rolling_rush_xg_5g', 'rolling_rush_xg_10g',
        'rolling_pp_xg_5g', 'rolling_pp_xg_10g',
        'rolling_ev_xg_5g', 'rolling_ev_xg_10g',
        'rolling_shots_5g', 'rolling_shots_10g',
        'ixg_ewm',
        'season_avg_xg', 'season_avg_hd_shots', 'season_avg_md_shots',
        'season_avg_rebound_xg', 'season_avg_rush_xg', 'season_avg_pp_xg',
        'rolling_pp_xg_share_5g',
    ]

    available_cols = [c for c in feature_cols if c in latest.columns]
    return latest[available_cols].reset_index(drop=True)


def get_team_features_for_date(date_str: str, team_stats_rolling: pd.DataFrame,
                                opponent: str) -> dict:
    """
    Get team defensive quality features for a specific opponent as of a date.
    """
    target_date = pd.to_datetime(date_str)

    prior = team_stats_rolling[
        (team_stats_rolling['team'] == opponent) &
        (team_stats_rolling['game_date'] < target_date)
    ]

    if prior.empty:
        return {}

    latest = prior.iloc[-1]

    return {
        'opp_rolling_xga_10g': latest.get('rolling_xga_10g', 0),
        'opp_rolling_hd_sa_10g': latest.get('rolling_hd_sa_10g', 0),
        'opp_rolling_xgf_pct_10g': latest.get('rolling_xgf_pct_10g', 50),
        'opp_season_xga': latest.get('season_avg_xga', 0),
        'opp_season_hd_sa': latest.get('season_avg_hd_sa', 0),
    }


# ==============================================================================
# SAVE TO DATABASE
# ==============================================================================

def save_advanced_stats(player_stats: pd.DataFrame, team_stats: pd.DataFrame):
    """Save computed stats to database tables."""
    conn = sqlite3.connect(DB_PATH)

    # Save player game stats
    player_stats.to_sql('adv_player_games', conn, if_exists='replace', index=False)
    team_stats.to_sql('adv_team_games', conn, if_exists='replace', index=False)

    conn.close()
    print(f"Saved {len(player_stats)} player-games and {len(team_stats)} team-games to database")


# ==============================================================================
# MAIN
# ==============================================================================

def build_all_stats():
    """Full pipeline: PBP → player stats → team stats → rolling features → save."""
    print(f"\n{'='*80}")
    print(f"ADVANCED STATS ENGINE")
    print(f"{'='*80}")

    # 1. Compute per-player per-game stats
    player_stats = compute_player_game_stats()
    if player_stats.empty:
        print("No data available. Run xG model first.")
        return None, None

    # 2. Compute per-team per-game stats
    team_stats = compute_team_game_stats()

    # 3. Compute rolling features
    print(f"\nComputing rolling features...")
    player_rolling = compute_rolling_advanced_stats(player_stats)
    team_rolling = compute_rolling_team_stats(team_stats)

    # 4. Save
    save_advanced_stats(player_rolling, team_rolling)

    # 5. Summary
    print(f"\n{'='*80}")
    print(f"ADVANCED STATS SUMMARY")
    print(f"{'='*80}")
    print(f"  Players: {player_rolling['player_id'].nunique()}")
    print(f"  Player-games: {len(player_rolling)}")
    print(f"  Teams: {team_rolling['team'].nunique()}")
    print(f"  Team-games: {len(team_rolling)}")
    print(f"  Date range: {player_rolling['game_date'].min().date()} to {player_rolling['game_date'].max().date()}")

    # Top xG players
    top = player_rolling.groupby(['player_name', 'team', 'position']).agg(
        games=('xg', 'count'),
        total_xg=('xg', 'sum'),
        total_goals=('goals', 'sum'),
        avg_xg=('xg', 'mean'),
    ).sort_values('total_xg', ascending=False).head(15)

    print(f"\n  Top 15 players by total xG:")
    print(f"  {'Name':<25} {'Team':<5} {'Pos':<4} {'GP':>4} {'xG':>7} {'Goals':>6} {'xG/G':>6}")
    for (name, team, pos), row in top.iterrows():
        print(f"  {name:<25} {team:<5} {pos:<4} {row['games']:>4.0f} {row['total_xg']:>7.2f} "
              f"{row['total_goals']:>6.0f} {row['avg_xg']:>6.3f}")

    return player_rolling, team_rolling


if __name__ == '__main__':
    build_all_stats()
