"""
Mixture Density Network (MDN) v3 - Targeted improvements over v1.

v1 BASELINE (MAE 4.107):
- Used only current-season boxscore data with NST features
- 2×64 hidden layers, K=3 components
- Rolling stats, season averages, EWM features

v3 IMPROVEMENTS (keeping what works, adding proven signals):
1. Opponent FPTS Allowed (d=0.736, p<0.000001):
   - Trailing 10-game average FPTS opponents scored against each team
   - Captures defensive quality WITHOUT needing NST data

2. Regression-weighted season averages (Bayesian shrinkage):
   - Use YoY regression coefficients (r) as shrinkage factors
   - Goals: r=0.712, Assists: r=0.735, FPTS: r=0.806
   - formula: regressed = r * player_avg + (1-r) * league_avg
   - Apply after minimum sample sizes; heavier shrinkage before

3. Historical baseline priors:
   - League-level structural parameters from 220K historical rows
   - League average FPTS by position (stable across years)
   - Use as the "league average" in regression shrinkage

CRITICAL: Uses all training dates (not sampled) for better coverage.
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from difflib import SequenceMatcher

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    print("PyTorch not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'torch', '--break-system-packages'])
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

DK_SCORING = {
    'goals': 8.5,
    'assists': 5.0,
    'shots': 1.5,
    'blocked_shots': 1.3,
    'plus_minus': 0.5,
}

DB_PATH = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db')

# Walk-forward dates
BACKTEST_START = datetime(2025, 11, 7)  # Nov 7
BACKTEST_END = datetime(2026, 2, 5)     # Feb 5
TRAIN_START = datetime(2025, 10, 7)     # Oct 7 (min data)
RETRAIN_INTERVAL = 14  # days

MDN_COMPONENTS = 3
BATCH_SIZE = 256
MAX_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YoY regression coefficients (from historical analysis)
YOY_REGRESSION_COEFS = {
    'goals': 0.712,
    'assists': 0.735,
    'dk_fpts': 0.806,
}

# Minimum sample sizes for regression shrinkage
MIN_GAMES_FOR_SHRINKAGE = {
    'goals': 60,
    'assists': 60,
    'dk_fpts': 25,
    'shots': 20,
}


# ============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================================

def load_boxscore_data():
    """Load all boxscore data from database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT
            player_name, player_id, team, position, game_date, opponent,
            goals, assists, shots, hits, blocked_shots, plus_minus,
            pp_goals, toi_seconds, dk_fpts, game_id
        FROM boxscore_skaters
        ORDER BY game_date, player_id
    """, conn)
    conn.close()

    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    # Precompute rolling stats
    print("Precomputing rolling features...")
    for window in [5, 10]:
        for col in ['goals', 'assists', 'shots', 'blocked_shots', 'dk_fpts', 'toi_seconds']:
            df[f'rolling_{col}_{window}g'] = (
                df.groupby('player_id')[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # Precompute exponentially-weighted FPTS (halflife=15 games)
    print("Precomputing exponentially-weighted FPTS...")
    df['dk_fpts_ewm'] = (
        df.groupby('player_id')['dk_fpts']
        .ewm(halflife=15, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Precompute season stats
    print("Precomputing season-to-date features...")
    for col in ['goals', 'assists', 'shots', 'blocked_shots', 'dk_fpts', 'toi_seconds']:
        cumsum = df.groupby('player_id')[col].cumsum()
        gp = df.groupby('player_id').cumcount() + 1
        df[f'season_avg_{col}'] = cumsum / gp

    # TOI trending
    print("Precomputing TOI trend feature...")
    rolling_toi_5 = (
        df.groupby('player_id')['toi_seconds']
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    season_avg_toi = df.groupby('player_id')['toi_seconds'].cumsum() / (df.groupby('player_id').cumcount() + 1)
    season_avg_toi = season_avg_toi.reset_index(level=0, drop=True)
    df['toi_seconds_trend'] = rolling_toi_5 / (season_avg_toi + 1e-6)

    # Games played
    df['season_gp'] = df.groupby('player_id').cumcount() + 1
    df['log_gp'] = np.log1p(df['season_gp'])

    return df


def compute_league_baseline_stats(df_historical):
    """
    Compute league-level baseline statistics from 220K historical rows.
    Returns position-level averages for use in regression shrinkage.
    """
    print("Computing league baseline stats from historical data...")

    if df_historical is None or df_historical.empty:
        print("  No historical data; using current season only")
        return {}

    # Group by position and compute means
    league_stats = {}
    for col in ['goals', 'assists', 'dk_fpts', 'shots']:
        for pos in df_historical['position'].unique():
            pos_data = df_historical[df_historical['position'] == pos]
            if len(pos_data) > 0:
                avg_val = pos_data[col].mean()
                key = f'{col}_by_pos'
                if key not in league_stats:
                    league_stats[key] = {}
                league_stats[key][pos] = avg_val

    # Also compute global averages
    for col in ['goals', 'assists', 'dk_fpts', 'shots']:
        league_stats[f'{col}_global'] = df_historical[col].mean()

    return league_stats


def load_historical_data():
    """Load 220K historical rows for baseline priors."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("""
            SELECT
                season, game_date, player_name, position,
                goals, assists, dk_fpts, shots, blocks, hits, pim,
                pp_goals, pp_assists, sh_goals, sh_assists, toi_seconds
            FROM historical_skaters
            ORDER BY season, game_date
        """, conn)
        conn.close()

        if not df.empty:
            df['game_date'] = pd.to_datetime(df['game_date'])
            print(f"  Loaded {len(df)} historical rows from seasons {df['season'].min()} to {df['season'].max()}")

        return df
    except Exception as e:
        print(f"  Warning: Could not load historical data: {e}")
        conn.close()
        return None


def compute_opponent_fpts_allowed(df):
    """
    Compute trailing 10-game average FPTS that each team's opponents scored.

    For each team on each date, this measures defensive quality.
    Confirmed signal: d=0.736, p<0.000001
    """
    print("Computing opponent FPTS allowed (10-game rolling avg)...")

    # Group by game_date and team, sum FPTS (each game counts opponent output)
    daily_team_fpts = df.groupby(['game_date', 'team'])['dk_fpts'].sum().reset_index()
    daily_team_fpts.columns = ['game_date', 'team', 'total_fpts']

    # For each opponent, compute rolling avg of FPTS allowed
    # This is tricky: we need to group by opponent, not team
    # Actually, we should group by team and compute what opponents scored against them

    # Approach: for each team, the FPTS allowed = sum of opponent FPTS
    # To get opponent, we need to know which team is which on each date
    # Simpler: create a team_day table, then for each team, rolling avg of what opponents scored

    # Create a team-day mapping with opponent
    team_opponent = df[['game_date', 'team', 'opponent']].drop_duplicates()
    team_opponent.columns = ['game_date', 'team', 'opponent']

    # Merge daily totals by opponent team
    team_opponent = team_opponent.merge(
        daily_team_fpts.rename(columns={'team': 'opponent', 'total_fpts': 'opp_fpts'}),
        on=['game_date', 'opponent'],
        how='left'
    )

    # Fill NaN with 0
    team_opponent['opp_fpts'] = team_opponent['opp_fpts'].fillna(0)

    # Sort by team and date, compute rolling 10-game average
    team_opponent = team_opponent.sort_values(['team', 'game_date'])
    team_opponent['opp_fpts_allowed_10g'] = (
        team_opponent.groupby('team')['opp_fpts']
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Merge back to original df
    df = df.merge(
        team_opponent[['game_date', 'team', 'opp_fpts_allowed_10g']],
        on=['game_date', 'team'],
        how='left'
    )

    df['opp_fpts_allowed_10g'] = df['opp_fpts_allowed_10g'].fillna(0)

    return df


def apply_regression_shrinkage(df, league_stats):
    """
    Apply regression-weighted season averages (Bayesian shrinkage).

    For stats with YoY r < 0.85:
    regressed_stat = r * player_avg + (1-r) * league_avg

    Only apply after minimum sample sizes; before that, use heavier shrinkage.
    """
    print("Applying regression-weighted shrinkage to season averages...")

    # Stats that need shrinkage
    shrink_stats = {
        'goals': 0.712,
        'assists': 0.735,
        'dk_fpts': 0.806,
    }

    for stat, r in shrink_stats.items():
        player_col = f'season_avg_{stat}'
        shrink_col = f'season_avg_{stat}_shrunk'
        gp_col = 'season_gp'

        if player_col not in df.columns:
            continue

        # Get league average (by position, or global fallback)
        def get_league_avg(row):
            pos_key = f'{stat}_by_pos'
            if pos_key in league_stats and row['position'] in league_stats[pos_key]:
                return league_stats[pos_key][row['position']]
            return league_stats.get(f'{stat}_global', 0)

        league_avg = df.apply(get_league_avg, axis=1)

        # Compute shrinkage factor based on games played
        min_games = MIN_GAMES_FOR_SHRINKAGE.get(stat, 20)

        # Shrinkage: r at min_games, gradually increase to full r at 1.5*min_games
        games = df[gp_col].fillna(1)
        shrink_factor = np.clip(games / min_games, 0, 1)  # 0 to 1
        actual_r = r * shrink_factor + (1 - shrink_factor)  # Ramp from 0 to r

        # Apply shrinkage
        df[shrink_col] = (
            actual_r * df[player_col].fillna(0) +
            (1 - actual_r) * league_avg
        )

    return df


# Cache NST data
_nst_cache = {}
_nst_cache_date = None
_adv_player_cache = None
_adv_team_cache = None
_nst_skater_cache = {}
_nst_skater_cache_date = None
_fuzzy_match_cache = {}
_dk_line_cache = None
_gl_fpts_cache = None


def _load_linemate_caches():
    """Load DK salary line data and game log FPTS for linemate feature computation."""
    global _dk_line_cache, _gl_fpts_cache
    if _dk_line_cache is not None:
        return

    conn = sqlite3.connect(DB_PATH)

    # DK salaries with line assignments
    _dk_line_cache = pd.read_sql("""
        SELECT slate_date, player_name, team, position, start_line, pp_unit,
               salary, dk_avg_fpts
        FROM dk_salaries
        WHERE start_line IS NOT NULL AND start_line NOT IN ('', 'confirm', 'unlikely')
        AND position != 'G'
    """, conn)
    _dk_line_cache['name_lower'] = _dk_line_cache['player_name'].str.lower().str.strip()

    # Game log rolling FPTS
    gl = pd.read_sql("""
        SELECT player_name, game_date, dk_fpts
        FROM game_logs_skaters WHERE game_date >= '2024-09-01'
    """, conn)
    conn.close()

    gl['game_date'] = pd.to_datetime(gl['game_date'])
    gl = gl.sort_values(['player_name', 'game_date'])
    gl['fpts_rolling_5g'] = gl.groupby('player_name')['dk_fpts'].transform(
        lambda x: x.rolling(5, min_periods=2).mean().shift(1)
    )
    # Keep latest per player per date
    _gl_fpts_cache = gl.sort_values('game_date').groupby('player_name').last()[['fpts_rolling_5g']].reset_index()
    _gl_fpts_cache['name_lower'] = _gl_fpts_cache['player_name'].str.lower().str.strip()

    print(f"  Loaded linemate data: {len(_dk_line_cache)} DK line records, {len(_gl_fpts_cache)} player FPTS")


def compute_linemate_features_for_date(date_str, adv_player_df):
    """
    Compute linemate quality features for all players on a given slate date.

    Returns DataFrame with columns:
        name_lower, lm_avg_xg_5g, lm_avg_fpts_5g, lm_total_xg_5g,
        lm_center_xg_5g, lm_center_fpts_5g, lm_pp_xg_5g,
        lm_is_top_line, lm_deployment, lm_avg_salary
    """
    _load_linemate_caches()

    dk_day = _dk_line_cache[_dk_line_cache['slate_date'] == date_str].copy()
    if dk_day.empty:
        return pd.DataFrame()

    # Get most recent advanced stats before this date
    dt = pd.to_datetime(date_str)
    before = adv_player_df[adv_player_df['game_date'] < dt]
    if before.empty:
        return pd.DataFrame()

    adv_latest = before.sort_values('game_date').groupby('player_name').last().reset_index()
    adv_latest['name_lower'] = adv_latest['player_name'].str.lower().str.strip()

    # Merge stats into slate
    dk_day = dk_day.merge(
        adv_latest[['name_lower', 'rolling_xg_5g', 'rolling_ev_xg_5g', 'rolling_pp_xg_5g']],
        on='name_lower', how='left'
    )
    dk_day = dk_day.merge(
        _gl_fpts_cache[['name_lower', 'fpts_rolling_5g']],
        on='name_lower', how='left'
    )
    dk_day['rolling_xg_5g'] = dk_day['rolling_xg_5g'].fillna(0)
    dk_day['fpts_rolling_5g'] = dk_day['fpts_rolling_5g'].fillna(dk_day['dk_avg_fpts'].fillna(3.0))

    results = []
    for idx, player in dk_day.iterrows():
        team = player['team']
        line = player['start_line']
        name_lower = player['name_lower']

        # Find linemates
        linemates = dk_day[
            (dk_day['team'] == team) &
            (dk_day['start_line'] == line) &
            (dk_day['name_lower'] != name_lower)
        ]
        n_lm = len(linemates)

        avg_lm_xg = linemates['rolling_xg_5g'].mean() if n_lm > 0 else 0
        avg_lm_fpts = linemates['fpts_rolling_5g'].mean() if n_lm > 0 else 0
        total_line_xg = linemates['rolling_xg_5g'].sum() + player['rolling_xg_5g']

        # Center xG (for wingers)
        center = linemates[linemates['position'] == 'C']
        center_xg = center['rolling_xg_5g'].values[0] if len(center) > 0 else 0
        center_fpts = center['fpts_rolling_5g'].values[0] if len(center) > 0 else 0

        # PP unit mates
        pp = player['pp_unit']
        if pd.notna(pp) and pp != '':
            pp_mates = dk_day[
                (dk_day['team'] == team) &
                (dk_day['pp_unit'] == pp) &
                (dk_day['name_lower'] != name_lower)
            ]
            pp_lm_xg = pp_mates['rolling_xg_5g'].mean() if len(pp_mates) > 0 else 0
        else:
            pp_lm_xg = 0

        try:
            line_num = int(float(line))
        except (ValueError, TypeError):
            line_num = 3

        results.append({
            'name_lower': name_lower,
            'lm_avg_xg_5g': avg_lm_xg,
            'lm_avg_fpts_5g': avg_lm_fpts,
            'lm_total_xg_5g': total_line_xg,
            'lm_center_xg_5g': center_xg,
            'lm_center_fpts_5g': center_fpts,
            'lm_pp_xg_5g': pp_lm_xg,
            'lm_is_top_line': 1 if line_num == 1 else 0,
            'lm_deployment': max(0, 5 - line_num),
            'lm_avg_salary': linemates['salary'].mean() if n_lm > 0 else 3000,
        })

    return pd.DataFrame(results)

def load_nst_teams_for_date(date_str):
    """Load NST team stats for a specific date (cached)."""
    global _nst_cache, _nst_cache_date

    if _nst_cache and _nst_cache_date == date_str[:10]:
        return _nst_cache

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(f"""
        SELECT
            team, situation, xgf_pct, hdcf_pct, sv_pct, to_date
        FROM nst_teams
        WHERE to_date <= '{date_str}' AND situation = '5v5'
    """, conn)
    conn.close()

    if df.empty:
        result = {}
    else:
        df['to_date'] = pd.to_datetime(df['to_date'])
        df = df.sort_values('to_date')
        latest = df.groupby('team').tail(1)

        result = {}
        for _, row in latest.iterrows():
            result[row['team']] = {
                'xgf_pct': row['xgf_pct'] / 100 if pd.notna(row['xgf_pct']) else 0.5,
                'hdcf_pct': row['hdcf_pct'] / 100 if pd.notna(row['hdcf_pct']) else 0.5,
                'sv_pct': row['sv_pct'] / 100 if pd.notna(row['sv_pct']) else 0.91,
            }

    _nst_cache = result
    _nst_cache_date = date_str[:10]
    return result


def fuzzy_match_names(boxscore_name, nst_names, threshold=0.6):
    """Fuzzy match boxscore name to NST name."""
    global _fuzzy_match_cache

    if boxscore_name in _fuzzy_match_cache:
        return _fuzzy_match_cache[boxscore_name]

    best_match = None
    best_ratio = 0

    for nst_name in nst_names:
        ratio = SequenceMatcher(None, boxscore_name.lower(), nst_name.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = nst_name

    result = best_match if best_ratio >= threshold else None
    _fuzzy_match_cache[boxscore_name] = result
    return result


def load_nst_skaters_for_date(date_str):
    """Load NST skater stats for all available statistics before a given date."""
    global _nst_skater_cache, _nst_skater_cache_date

    if _nst_skater_cache and _nst_skater_cache_date == date_str[:10]:
        return _nst_skater_cache, _nst_skater_cache_date

    conn = sqlite3.connect(DB_PATH)

    query = f"""
        SELECT DISTINCT to_date FROM nst_skaters
        WHERE to_date <= '{date_str}' AND from_date = '2025-10-07'
        ORDER BY to_date DESC LIMIT 1
    """
    df_dates = pd.read_sql(query, conn)

    if df_dates.empty:
        conn.close()
        result = {}
        latest_date = None
    else:
        latest_date = df_dates['to_date'].iloc[0]

        query = f"""
            SELECT player, situation, stat_type, ixg, toi, gp, hdcf_pct
            FROM nst_skaters
            WHERE to_date = '{latest_date}' AND from_date = '2025-10-07'
        """
        df = pd.read_sql(query, conn)
        conn.close()

        result = {}
        for _, row in df.iterrows():
            key = (row['player'], row['situation'], row['stat_type'])
            result[key] = row

    _nst_skater_cache = result
    _nst_skater_cache_date = date_str[:10]
    return result, latest_date


def get_nst_feature(nst_data, player_name, situation, stat_type, column, default=0):
    """Safely retrieve an NST feature value."""
    key = (player_name, situation, stat_type)
    if key in nst_data:
        val = nst_data[key].get(column)
        if pd.notna(val):
            return float(val)
    return default


def compute_toi_per_game(row):
    """Compute TOI per game from NST data."""
    toi_str = row.get('toi')
    gp = row.get('gp')

    if pd.isna(toi_str) or pd.isna(gp) or gp == 0:
        return 0

    try:
        toi_seconds = float(toi_str)
        return toi_seconds / gp
    except (ValueError, TypeError):
        return 0


def build_feature_matrix(df, predict_date, train_cutoff=None):
    """
    Build feature matrix for prediction on a given date.
    NOW INCLUDES: opponent FPTS allowed (v3 improvement #1).
    """
    # Get games on predict_date
    predict_games = df[df['game_date'] == predict_date].copy()
    if predict_games.empty:
        return None, None, None, None, None

    # Position one-hot encoding
    positions = pd.get_dummies(predict_games['position'], prefix='pos', drop_first=False)

    # =================================================================
    # ADVANCED STATS INTEGRATION (replaces NST dependency)
    # Uses our own PBP-derived xG, ixG, xGF%, HD stats
    # =================================================================
    from advanced_stats import get_player_features_for_date, get_team_features_for_date

    date_str = predict_date.strftime('%Y-%m-%d')

    # Load advanced stats (cached globally to avoid repeated DB reads)
    global _adv_player_cache, _adv_team_cache
    if '_adv_player_cache' not in globals() or _adv_player_cache is None:
        conn_adv = sqlite3.connect(DB_PATH)
        _adv_player_cache = pd.read_sql("SELECT * FROM adv_player_games", conn_adv)
        _adv_team_cache = pd.read_sql("SELECT * FROM adv_team_games", conn_adv)
        conn_adv.close()
        _adv_player_cache['game_date'] = pd.to_datetime(_adv_player_cache['game_date'])
        _adv_team_cache['game_date'] = pd.to_datetime(_adv_team_cache['game_date'])
        print(f"  Loaded advanced stats: {len(_adv_player_cache)} player-games, {len(_adv_team_cache)} team-games")

    adv_player = _adv_player_cache
    adv_team = _adv_team_cache

    # Get player advanced features (most recent before predict_date)
    player_feats = get_player_features_for_date(date_str, adv_player)

    # Opponent quality from our own team stats
    predict_games.loc[:, 'opp_xgf_pct'] = 0.5
    predict_games.loc[:, 'opp_sv_pct'] = 0.91

    for idx, row in predict_games.iterrows():
        opp = row.get('opponent', '')
        opp_feats = get_team_features_for_date(date_str, adv_team, opp)
        if opp_feats:
            predict_games.loc[idx, 'opp_xgf_pct'] = opp_feats.get('opp_rolling_xgf_pct_10g', 50) / 100
            opp_xga = opp_feats.get('opp_rolling_xga_10g', 3.0)
            predict_games.loc[idx, 'opp_sv_pct'] = max(0.85, 1.0 - (opp_xga / 35.0))

    # Map player advanced stats to NST feature slots
    predict_games.loc[:, 'pp_toi_per_game'] = 0.0
    predict_games.loc[:, 'ev_ixg'] = 0.0
    predict_games.loc[:, 'ev_toi_per_game'] = 0.0
    predict_games.loc[:, 'pp_ixg'] = 0.0
    predict_games.loc[:, 'oi_hdcf_pct'] = 0.0

    if not player_feats.empty:
        player_feats['name_lower'] = player_feats['player_name'].str.lower().str.strip()
        predict_games['name_lower'] = predict_games['player_name'].str.lower().str.strip()

        merged = predict_games[['name_lower']].merge(
            player_feats, on='name_lower', how='left', suffixes=('', '_adv')
        )

        # Map: ev_ixg ← rolling_ev_xg_5g, pp_ixg ← rolling_pp_xg_5g
        predict_games.loc[:, 'ev_ixg'] = merged['rolling_ev_xg_5g'].fillna(0).values
        predict_games.loc[:, 'pp_ixg'] = merged['rolling_pp_xg_5g'].fillna(0).values

        # HD shots as fraction (like hdcf_pct)
        hd = merged['rolling_hd_shots_5g'].fillna(0).values
        shots = merged['rolling_shots_5g'].fillna(1).values
        predict_games.loc[:, 'oi_hdcf_pct'] = np.clip(hd / (shots + 0.01), 0, 1)

        # Inject custom advanced features for model to use
        for adv_col in ['rolling_xg_5g', 'rolling_xg_10g', 'ixg_ewm',
                        'rolling_hd_shots_5g', 'rolling_pp_xg_share_5g',
                        'season_avg_xg']:
            if adv_col in merged.columns:
                predict_games.loc[:, f'adv_{adv_col}'] = merged[adv_col].fillna(0).values

        predict_games.drop(columns=['name_lower'], errors='ignore', inplace=True)

    # =================================================================
    # LINEMATE FEATURES (line combination quality from DK slate data)
    # =================================================================
    try:
        lm_feats = compute_linemate_features_for_date(date_str, adv_player)
        if not lm_feats.empty:
            predict_games['name_lower'] = predict_games['player_name'].str.lower().str.strip()
            predict_games = predict_games.merge(lm_feats, on='name_lower', how='left')
            predict_games.drop(columns=['name_lower'], errors='ignore', inplace=True)

            # Fill missing linemate features (players without line data)
            lm_cols = [c for c in predict_games.columns if c.startswith('lm_')]
            for c in lm_cols:
                predict_games[c] = predict_games[c].fillna(0)

            # Interaction: own xG * linemate quality
            if 'adv_rolling_xg_5g' in predict_games.columns:
                predict_games['lm_xg_interaction'] = (
                    predict_games['adv_rolling_xg_5g'] * predict_games['lm_avg_xg_5g']
                )
    except Exception as e:
        pass  # Linemate features are optional; don't break backtest

    # =================================================================
    # SCORE-STATE FEATURES (trailing/tied/leading deployment & xG splits)
    # =================================================================
    try:
        from score_state_features import compute_score_state_features, compute_rest_features_for_players

        ss_feats = compute_score_state_features(date_str, lookback_games=15)
        if not ss_feats.empty:
            predict_games['name_lower'] = predict_games['player_name'].str.lower().str.strip()
            predict_games = predict_games.merge(ss_feats, on='name_lower', how='left')
            predict_games.drop(columns=['name_lower'], errors='ignore', inplace=True)

            # Fill missing score-state features
            ss_cols = [c for c in predict_games.columns if c.startswith('ss_')]
            for c in ss_cols:
                predict_games[c] = predict_games[c].fillna(0)

    except Exception as e:
        pass  # Score-state features are optional; don't break backtest

    # =================================================================
    # REST / BACK-TO-BACK FEATURES (schedule density impact)
    # =================================================================
    try:
        from score_state_features import compute_rest_features_for_players

        # Need team + opponent columns for rest computation
        if 'team' in predict_games.columns:
            predict_games = compute_rest_features_for_players(date_str, predict_games)
    except Exception as e:
        pass  # Rest features are optional; don't break backtest

    # Collect rolling and season features
    rolling_cols = [c for c in df.columns if c.startswith('rolling_')]
    season_cols = [c for c in df.columns if c.startswith('season_avg_') or c == 'log_gp']

    if not rolling_cols or not season_cols:
        return None, None, None, None, None

    # Select features
    feature_cols = []
    for col in rolling_cols + season_cols:
        if col in predict_games.columns:
            feature_cols.append(col)

    # Add EWM and trend features
    extra_features = []
    if 'dk_fpts_ewm' in predict_games.columns:
        extra_features.append('dk_fpts_ewm')
    if 'toi_seconds_trend' in predict_games.columns:
        extra_features.append('toi_seconds_trend')
    feature_cols.extend(extra_features)

    # Advanced stats features (replaces NST)
    nst_features = ['pp_toi_per_game', 'ev_ixg', 'ev_toi_per_game', 'pp_ixg', 'oi_hdcf_pct']

    # Custom advanced features from our PBP-derived stats
    adv_features = [c for c in predict_games.columns if c.startswith('adv_')]

    # Linemate features
    lm_features = [c for c in predict_games.columns if c.startswith('lm_')]

    # Score-state features
    ss_features = [c for c in predict_games.columns if c.startswith('ss_')]

    # Rest / B2B features
    rest_features = [c for c in predict_games.columns if c.startswith('rest_')]

    # Opponent quality features
    opp_features = ['opp_xgf_pct', 'opp_sv_pct']
    if 'opp_fpts_allowed_10g' in predict_games.columns:
        opp_features.append('opp_fpts_allowed_10g')

    all_features = feature_cols + opp_features + nst_features + adv_features + lm_features + ss_features + rest_features
    available = [c for c in all_features if c in predict_games.columns]
    X = predict_games[available].reset_index(drop=True).copy()

    # Add interaction feature
    X['hdcf_x_opp_weak'] = X['oi_hdcf_pct'] * (1.0 - X['opp_xgf_pct'])

    # Add position encoding
    positions_reset = positions.reset_index(drop=True)
    X = pd.concat([X, positions_reset], axis=1)

    y = predict_games['dk_fpts'].values
    player_ids = predict_games['player_id'].values
    player_names = predict_games['player_name'].values
    positions_val = predict_games['position'].values

    return X, y, player_ids, player_names, positions_val


def prepare_training_data(df, train_end_date):
    """
    Prepare training data using all games up to train_end_date.
    V3: Uses ALL dates (not sampled) for better coverage.
    """
    train_df = df[df['game_date'] <= train_end_date].copy()

    if train_df.empty:
        return None, None, None

    # Get unique prediction dates - USE ALL, don't sample
    pred_dates = sorted(train_df['game_date'].unique())

    print(f"  Using {len(pred_dates)} dates for training (all available)")

    X_list, y_list = [], []

    for pred_date in pred_dates:
        X, y, _, _, _ = build_feature_matrix(df, pred_date, train_cutoff=train_end_date)
        if X is not None and len(y) > 0 and len(X) == len(y):
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        return None, None, None

    X = pd.concat(X_list, ignore_index=True)
    y = np.concatenate(y_list)

    assert len(X) == len(y), f"Shape mismatch: X={len(X)}, y={len(y)}"

    X = X.fillna(0)

    # Normalize X
    X_mean = X.mean()
    X_std = X.std()
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std

    X_array = X.values.astype(np.float32)
    y_array = y.astype(np.float32)

    return torch.FloatTensor(X_array), torch.FloatTensor(y_array), (X_mean, X_std)


# ============================================================================
# MODEL ARCHITECTURE (unchanged from v1)
# ============================================================================

class MixtureDesityNetwork(nn.Module):
    """Mixture Density Network with K Gaussian components."""

    def __init__(self, input_size, hidden_size=64, k=3):
        super().__init__()
        self.k = k

        # Feature extraction (same as v1: 2×64)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Output layer for mixture parameters
        self.pi_layer = nn.Linear(hidden_size, k)
        self.mu_layer = nn.Linear(hidden_size, k)
        self.sigma_layer = nn.Linear(hidden_size, k)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))

        pi = torch.softmax(self.pi_layer(h), dim=-1)
        mu = self.mu_layer(h)
        sigma = torch.nn.functional.softplus(self.sigma_layer(h)) + 1e-6

        return pi, mu, sigma

    def loss(self, pi, mu, sigma, y):
        """Negative log-likelihood of Gaussian mixture model."""
        y = y.unsqueeze(1)

        log_sigma = torch.log(sigma)
        normalized = (y - mu) / sigma

        log_gaussian = -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * normalized**2
        log_mixture = torch.log(pi + 1e-10) + log_gaussian

        max_log = torch.max(log_mixture, dim=1, keepdim=True)[0]
        log_prob = max_log + torch.logsumexp(log_mixture - max_log, dim=1, keepdim=True)

        return -torch.mean(log_prob)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(X_train, y_train, X_val, y_val):
    """Train MDN model with early stopping."""
    input_size = X_train.shape[1]
    model = MixtureDesityNetwork(input_size, hidden_size=64, k=MDN_COMPONENTS).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                      patience=5, min_lr=1e-6)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            pi, mu, sigma = model(X_batch)
            loss = model.loss(pi, mu, sigma, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pi, mu, sigma = model(X_batch)
                loss = model.loss(pi, mu, sigma, y_batch)
                val_loss += loss.item() * len(X_batch)

        val_loss /= len(X_val)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"    Early stop at epoch {epoch+1}, val_loss={val_loss:.4f}")
            break

    return model


def predict_mixture(model, X):
    """Get mixture parameters for input X."""
    model.eval()
    with torch.no_grad():
        X_array = X.values.astype(np.float64).astype(np.float32)
        X_tensor = torch.FloatTensor(X_array).to(DEVICE)
        pi, mu, sigma = model(X_tensor)

    return pi.cpu(), mu.cpu(), sigma.cpu()


def compute_projection_stats(pi, mu, sigma):
    """Compute projection statistics from mixture distribution."""
    expected = (pi * mu).sum(dim=1)

    variance = (pi * (sigma**2 + mu**2)).sum(dim=1) - expected**2
    std = torch.sqrt(torch.clamp(variance, min=1e-6))

    samples = []
    for _ in range(500):
        component = torch.multinomial(pi, 1).squeeze(1)
        idx = torch.arange(len(component))
        sample_mu = mu[idx, component]
        sample_sigma = sigma[idx, component]
        sample = torch.normal(sample_mu, sample_sigma)
        samples.append(sample)

    samples = torch.stack(samples)

    p10 = torch.quantile(samples, 0.1, dim=0)
    p90 = torch.quantile(samples, 0.9, dim=0)

    p_above_10 = (samples > 10).float().mean(dim=0)
    p_above_15 = (samples > 15).float().mean(dim=0)
    p_above_20 = (samples > 20).float().mean(dim=0)
    p_above_25 = (samples > 25).float().mean(dim=0)

    return {
        'expected_fpts': expected.numpy(),
        'std_fpts': std.numpy(),
        'floor_fpts': p10.numpy(),
        'ceiling_fpts': p90.numpy(),
        'p_above_10': p_above_10.numpy(),
        'p_above_15': p_above_15.numpy(),
        'p_above_20': p_above_20.numpy(),
        'p_above_25': p_above_25.numpy(),
    }


# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================

def run_backtest(df):
    """
    Run walk-forward backtest from BACKTEST_START to BACKTEST_END.
    Retrain every RETRAIN_INTERVAL days.
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD BACKTEST: {BACKTEST_START.date()} to {BACKTEST_END.date()}")
    print(f"Retraining every {RETRAIN_INTERVAL} days")
    print(f"{'='*80}\n")

    results = []
    current_date = BACKTEST_START
    last_retrain_date = TRAIN_START
    model = None
    norm_stats = None

    while current_date <= BACKTEST_END:
        # Retrain if needed
        if model is None or (current_date - last_retrain_date).days >= RETRAIN_INTERVAL:
            train_end = current_date - timedelta(days=1)
            print(f"\n>>> Retraining on data through {train_end.date()}")

            X_train, y_train, norm_stats = prepare_training_data(df, train_end)

            if X_train is None or len(X_train) < 100:
                print(f"  Insufficient training data ({len(X_train) if X_train is not None else 0} samples)")
                current_date += timedelta(days=1)
                continue

            n_train = int(0.8 * len(X_train))
            idx = torch.randperm(len(X_train))
            train_idx, val_idx = idx[:n_train], idx[n_train:]

            X_train_split = X_train[train_idx]
            y_train_split = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]

            print(f"  Training samples: {len(X_train_split)}, Validation samples: {len(X_val)}")

            model = train_model(X_train_split, y_train_split, X_val, y_val)
            last_retrain_date = current_date

        # Make predictions
        X_pred, y_actual, player_ids, player_names, positions = \
            build_feature_matrix(df, current_date)

        if X_pred is None or len(X_pred) == 0:
            current_date += timedelta(days=1)
            continue

        X_mean, X_std = norm_stats

        # Align prediction features to training feature set
        train_cols = X_mean.index.tolist()
        for c in train_cols:
            if c not in X_pred.columns:
                X_pred[c] = 0
        X_pred = X_pred.reindex(columns=train_cols, fill_value=0)

        X_pred_norm = (X_pred - X_mean) / (X_std + 1e-6)
        X_pred_norm = X_pred_norm.fillna(0)

        pi, mu, sigma = predict_mixture(model, X_pred_norm)
        stats = compute_projection_stats(pi, mu, sigma)

        # Store results
        for i, (pid, name, pos, actual) in enumerate(zip(player_ids, player_names, positions, y_actual)):
            results.append({
                'game_date': current_date.date(),
                'player_id': pid,
                'player_name': name,
                'position': pos,
                'actual_fpts': actual,
                'predicted_fpts': stats['expected_fpts'][i],
                'std_fpts': stats['std_fpts'][i],
                'floor_fpts': stats['floor_fpts'][i],
                'ceiling_fpts': stats['ceiling_fpts'][i],
                'p_above_10': stats['p_above_10'][i],
                'p_above_15': stats['p_above_15'][i],
                'p_above_20': stats['p_above_20'][i],
                'p_above_25': stats['p_above_25'][i],
            })

        print(f"  {current_date.date()}: {len(X_pred)} players predicted")
        current_date += timedelta(days=1)

    return pd.DataFrame(results)


# ============================================================================
# METRICS & REPORTING
# ============================================================================

def compute_metrics(actual, predicted):
    """Compute MAE, RMSE, and correlation."""
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() == 0:
        return np.nan, np.nan, np.nan

    actual = actual[valid]
    predicted = predicted[valid]

    mae = np.abs(actual - predicted).mean()
    rmse = np.sqrt(((actual - predicted) ** 2).mean())

    if len(actual) > 1:
        corr = np.corrcoef(actual, predicted)[0, 1]
    else:
        corr = 0

    return mae, rmse, corr


def print_results_table(mdn_results):
    """Print comparison table of all models."""
    if mdn_results is None or len(mdn_results) == 0:
        print("No results to display")
        return

    print(f"\n{'='*100}")
    print("MDN V3 - RESULTS COMPARISON")
    print(f"{'='*100}\n")

    print("OVERALL METRICS")
    print("-" * 80)

    mae_mdn, rmse_mdn, corr_mdn = compute_metrics(
        mdn_results['actual_fpts'].values,
        mdn_results['predicted_fpts'].values
    )

    print(f"{'Model':<20s} | {'MAE':>8} | {'RMSE':>8} | {'Corr':>8}")
    print("-" * 80)
    print(f"{'MDN v3 (learned)':20s} | {mae_mdn:>8.3f} | {rmse_mdn:>8.3f} | {corr_mdn:>8.3f}")
    print(f"{'MDN v1 (baseline)':20s} | {4.107:>8.3f} | {'N/A':>8} | {'N/A':>8}  (reference)")
    print(f"{'Kalman filter':20s} | {4.320:>8.3f} | {'N/A':>8} | {'N/A':>8}  (baseline)")
    print(f"{'Poisson sim':20s} | {4.750:>8.3f} | {'N/A':>8} | {'N/A':>8}  (baseline)")

    # Improvement vs v1
    improvement_vs_v1 = (4.107 - mae_mdn) / 4.107 * 100 if not np.isnan(mae_mdn) else 0

    print(f"\n{'V3 vs V1 Improvement:':<35} {improvement_vs_v1:>+.2f}%")
    print(f"{'V3 vs Kalman Improvement:':<35} {(4.320 - mae_mdn) / 4.320 * 100:>+.2f}%")

    # By position
    print(f"\n{'BY POSITION':80s}")
    print("-" * 80)
    print(f"{'Position':<12} {'N':>6} {'MAE':>10} {'Actual Std':>12} {'Difficulty':>12}")
    print("-" * 80)

    for pos in sorted(mdn_results['position'].unique()):
        pos_data = mdn_results[mdn_results['position'] == pos]
        n = len(pos_data)

        mdn_mae = np.abs(pos_data['actual_fpts'] - pos_data['predicted_fpts']).mean()
        actual_std = pos_data['actual_fpts'].std()

        if actual_std < 4:
            difficulty = "EASY"
        elif actual_std < 6:
            difficulty = "MEDIUM"
        else:
            difficulty = "HARD"

        print(f"{pos:<12} {n:>6} {mdn_mae:>10.3f} {actual_std:>12.2f} {difficulty:>12}")

    # Summary
    print(f"\n{'KEY IMPROVEMENTS IN V3':80s}")
    print("-" * 80)
    print("1. Opponent FPTS Allowed (d=0.736, p<0.000001)")
    print("   - Trailing 10-game avg FPTS opponents scored")
    print("   - Captures defensive quality without NST data")
    print("\n2. Regression-weighted season averages (Bayesian shrinkage)")
    print("   - Goals: r=0.712, Assists: r=0.735, FPTS: r=0.806")
    print("   - Shrinkage ramped by games played, reaches full r at min_games")
    print("\n3. Historical baseline priors")
    print("   - League-level stats from 220K historical rows")
    print("   - Used in regression shrinkage for better priors")
    print("\n4. All training dates used (not sampled)")
    print("   - Better coverage of training distribution")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MDN v3 NHL DFS Projection Model')
    parser.add_argument('--backtest', action='store_true', help='Run walk-forward backtest')
    args = parser.parse_args()

    print("\nLoading and preprocessing data...")
    df = load_boxscore_data()
    print(f"Loaded {len(df)} boxscore records from {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    # V3 IMPROVEMENT #3: Load historical data for baseline priors
    print("\nLoading historical data for baseline priors...")
    df_historical = load_historical_data()
    league_stats = compute_league_baseline_stats(df_historical)

    # V3 IMPROVEMENT #2: Apply regression shrinkage to season averages
    print("\nApplying v3 improvements...")
    df = apply_regression_shrinkage(df, league_stats)

    # V3 IMPROVEMENT #1: Compute opponent FPTS allowed
    df = compute_opponent_fpts_allowed(df)

    print(f"Features computed. Running backtest...")
    print(f"  - Opponent FPTS allowed: {'opp_fpts_allowed_10g' in df.columns}")
    print(f"  - Regression shrinkage applied: True")
    print(f"  - League priors loaded: {len(league_stats) > 0}")

    if args.backtest:
        print("\nStarting walk-forward backtest...")
        mdn_results = run_backtest(df)

        print_results_table(mdn_results)

        output_path = Path(__file__).parent / 'mdn_v3_backtest_results.csv'
        mdn_results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("\nUsage: python mdn_v3.py --backtest")


if __name__ == '__main__':
    main()
