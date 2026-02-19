"""
Score-State & Rest/B2B Feature Engineering
==========================================

Computes per-player features based on:
1. Score-state deployment: xG, shot rates, PP usage when trailing/tied/leading
2. Back-to-back / rest days: performance impact from schedule density

Data sources:
    - pbp_shots: 600K+ shots with score_diff for score-state splits
    - game_logs_skaters / adv_player_games: game dates for B2B detection

Features are prefixed 'ss_' (score-state) and 'rest_' so MDN v3's
auto-discovery picks them up automatically.
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / 'data' / 'nhl_dfs_history.db'

# Module-level caches (loaded once, reused across dates in backtest)
_pbp_cache = None
_game_dates_cache = None


def _load_pbp_cache():
    """Load and cache pbp_shots for score-state computation."""
    global _pbp_cache
    if _pbp_cache is not None:
        return _pbp_cache

    conn = sqlite3.connect(DB_PATH)
    _pbp_cache = pd.read_sql("""
        SELECT shooter_id, shooter_name, shooter_team_abbrev as team,
               game_date, game_id, is_goal, xg, score_diff,
               is_pp, is_5v5, period, game_seconds
        FROM pbp_shots
    """, conn)
    conn.close()

    _pbp_cache['game_date'] = pd.to_datetime(_pbp_cache['game_date'])
    print(f"  Score-state: loaded {len(_pbp_cache)} shots for {_pbp_cache['shooter_id'].nunique()} shooters")
    return _pbp_cache


def _load_game_dates_cache():
    """Load and cache game dates per team for B2B/rest computation."""
    global _game_dates_cache
    if _game_dates_cache is not None:
        return _game_dates_cache

    conn = sqlite3.connect(DB_PATH)
    _game_dates_cache = pd.read_sql("""
        SELECT DISTINCT team, game_date
        FROM adv_player_games
        ORDER BY team, game_date
    """, conn)
    conn.close()

    _game_dates_cache['game_date'] = pd.to_datetime(_game_dates_cache['game_date'])
    return _game_dates_cache


def compute_score_state_features(date_str: str, lookback_games: int = 15) -> pd.DataFrame:
    """
    Compute score-state deployment features for all players with data before date_str.

    For each player, computes over their last `lookback_games` games:
    - ss_trailing_xg_rate: xG per 60 when trailing (score_diff < 0)
    - ss_leading_xg_rate: xG per 60 when leading (score_diff > 0)
    - ss_tied_xg_rate: xG per 60 when tied (score_diff == 0)
    - ss_trailing_shot_share: fraction of shots taken when trailing
    - ss_trailing_pp_share: fraction of PP shots when trailing (deployment signal)
    - ss_close_game_xg: xG rate in close games (|score_diff| <= 1)
    - ss_blowout_xg: xG rate in non-close games (|score_diff| > 2)
    - ss_deployment_skew: trailing_shot_share - leading_shot_share (>0 = more usage when trailing)

    Returns:
        DataFrame with columns [name_lower, ss_*] indexed by name_lower
    """
    pbp = _load_pbp_cache()
    cutoff = pd.Timestamp(date_str)

    # Filter to shots before prediction date
    recent = pbp[pbp['game_date'] < cutoff].copy()
    if recent.empty:
        return pd.DataFrame()

    # Keep only last N games per player (vectorized)
    # Rank game_ids per player (higher = more recent)
    game_ranks = (
        recent[['shooter_id', 'game_id']].drop_duplicates()
        .assign(rank=lambda d: d.groupby('shooter_id')['game_id'].rank(method='dense', ascending=False))
    )
    valid_pairs = game_ranks[game_ranks['rank'] <= lookback_games][['shooter_id', 'game_id']]
    recent = recent.merge(valid_pairs, on=['shooter_id', 'game_id'], how='inner')

    if recent.empty:
        return pd.DataFrame()

    # Classify score state
    recent['state'] = np.where(
        recent['score_diff'] < 0, 'trailing',
        np.where(recent['score_diff'] > 0, 'leading', 'tied')
    )
    recent['is_close'] = (recent['score_diff'].abs() <= 1).astype(int)
    recent['is_blowout'] = (recent['score_diff'].abs() > 2).astype(int)

    # Aggregate per player
    results = []
    for pid, grp in recent.groupby('shooter_id'):
        total_shots = len(grp)
        if total_shots < 10:  # Need minimum sample
            continue

        name = grp['shooter_name'].iloc[0]
        n_games = grp['game_id'].nunique()

        # Score-state splits
        trailing = grp[grp['state'] == 'trailing']
        leading = grp[grp['state'] == 'leading']
        tied = grp[grp['state'] == 'tied']
        close = grp[grp['is_close'] == 1]
        blowout = grp[grp['is_blowout'] == 1]

        # xG rates (per game to normalize)
        trailing_xg = trailing['xg'].sum() / max(n_games, 1)
        leading_xg = leading['xg'].sum() / max(n_games, 1)
        tied_xg = tied['xg'].sum() / max(n_games, 1)
        close_xg = close['xg'].sum() / max(n_games, 1)
        blowout_xg = blowout['xg'].sum() / max(n_games, 1)

        # Shot distribution across states
        trailing_shot_share = len(trailing) / total_shots
        leading_shot_share = len(leading) / total_shots

        # PP deployment when trailing (coaches lean on top players)
        trailing_pp = trailing['is_pp'].sum() if len(trailing) > 0 else 0
        total_pp = grp['is_pp'].sum()
        trailing_pp_share = trailing_pp / max(total_pp, 1)

        # Deployment skew: positive = more usage when trailing
        deployment_skew = trailing_shot_share - leading_shot_share

        results.append({
            'name_lower': name.lower().strip(),
            'ss_trailing_xg': trailing_xg,
            'ss_leading_xg': leading_xg,
            'ss_tied_xg': tied_xg,
            'ss_close_game_xg': close_xg,
            'ss_blowout_xg': blowout_xg,
            'ss_trailing_shot_share': trailing_shot_share,
            'ss_trailing_pp_share': trailing_pp_share,
            'ss_deployment_skew': deployment_skew,
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Add interaction: trailing xG advantage (how much better when trailing)
    avg_xg = (result_df['ss_trailing_xg'] + result_df['ss_leading_xg'] + result_df['ss_tied_xg']) / 3
    result_df['ss_trailing_advantage'] = result_df['ss_trailing_xg'] - avg_xg

    return result_df


def compute_rest_features(date_str: str, teams_on_slate: list) -> pd.DataFrame:
    """
    Compute back-to-back and rest features for teams playing on date_str.

    Features:
    - rest_is_b2b: 1 if team played yesterday
    - rest_days: days since last game (capped at 7)
    - rest_is_rested: 1 if 3+ days rest (extra rest)
    - rest_opp_is_b2b: 1 if opponent is on B2B (advantage)
    - rest_b2b_disadvantage: rest_is_b2b - rest_opp_is_b2b (negative = we're the tired team)

    Returns:
        DataFrame with columns [team, rest_*]
    """
    game_dates = _load_game_dates_cache()
    target = pd.Timestamp(date_str)

    results = []
    team_rest = {}

    for team in teams_on_slate:
        team_games = game_dates[game_dates['team'] == team]['game_date'].sort_values()
        # Games before target date
        prior = team_games[team_games < target]

        if len(prior) == 0:
            days_rest = 3  # default
        else:
            last_game = prior.iloc[-1]
            days_rest = (target - last_game).days

        is_b2b = 1 if days_rest == 1 else 0
        is_rested = 1 if days_rest >= 3 else 0

        team_rest[team] = {
            'rest_is_b2b': is_b2b,
            'rest_days': min(days_rest, 7),
            'rest_is_rested': is_rested,
        }

    # Now compute opponent rest (need to know matchups)
    # We'll return team-level and let the caller join with opponent
    result_df = pd.DataFrame([
        {'team': team, **vals} for team, vals in team_rest.items()
    ])

    return result_df


def compute_rest_features_for_players(date_str: str, player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest features for a player DataFrame that has 'team' and 'opponent' columns.

    Adds rest_* columns directly to the returned DataFrame.
    """
    if player_df.empty:
        return player_df

    teams = list(set(
        player_df['team'].unique().tolist() +
        (player_df['opponent'].unique().tolist() if 'opponent' in player_df.columns else [])
    ))

    rest_df = compute_rest_features(date_str, teams)
    if rest_df.empty:
        return player_df

    result = player_df.copy()

    # Merge team rest
    rest_team = rest_df.rename(columns={
        'rest_is_b2b': 'rest_is_b2b',
        'rest_days': 'rest_days',
        'rest_is_rested': 'rest_is_rested',
    })
    result = result.merge(rest_team, on='team', how='left')

    # Merge opponent rest
    if 'opponent' in result.columns:
        rest_opp = rest_df.rename(columns={
            'team': 'opponent',
            'rest_is_b2b': 'rest_opp_is_b2b',
            'rest_days': 'rest_opp_days',
            'rest_is_rested': 'rest_opp_is_rested',
        })
        result = result.merge(rest_opp, on='opponent', how='left')

        # B2B disadvantage: positive means we're more rested
        result['rest_b2b_advantage'] = (
            result['rest_opp_is_b2b'].fillna(0) - result['rest_is_b2b'].fillna(0)
        )
    else:
        result['rest_opp_is_b2b'] = 0
        result['rest_b2b_advantage'] = 0

    # Fill NaN rest features
    for c in [col for col in result.columns if col.startswith('rest_')]:
        result[c] = result[c].fillna(0)

    return result


# ==============================================================================
# QUICK VALIDATION
# ==============================================================================

if __name__ == '__main__':
    import time

    print("=" * 70)
    print("SCORE-STATE & REST FEATURE VALIDATION")
    print("=" * 70)

    # Test score-state features
    print("\n[1] Score-state features for 2026-02-01...")
    t0 = time.time()
    ss_df = compute_score_state_features('2026-02-01', lookback_games=15)
    print(f"  Computed in {time.time()-t0:.1f}s: {len(ss_df)} players")

    if not ss_df.empty:
        # Top players by trailing xG
        top = ss_df.nlargest(10, 'ss_trailing_xg')
        print("\n  Top 10 by trailing xG per game:")
        for _, r in top.iterrows():
            print(f"    {r['name_lower']:<25} trail={r['ss_trailing_xg']:.3f}  "
                  f"lead={r['ss_leading_xg']:.3f}  tied={r['ss_tied_xg']:.3f}  "
                  f"skew={r['ss_deployment_skew']:+.3f}")

        # Biggest deployment skew (trailing usage)
        print("\n  Top 10 deployment skew (more shots when trailing):")
        skew = ss_df.nlargest(10, 'ss_deployment_skew')
        for _, r in skew.iterrows():
            print(f"    {r['name_lower']:<25} skew={r['ss_deployment_skew']:+.3f}  "
                  f"trail_pp={r['ss_trailing_pp_share']:.2f}")

        print(f"\n  Feature stats:")
        for col in [c for c in ss_df.columns if c.startswith('ss_')]:
            print(f"    {col:<30} mean={ss_df[col].mean():.4f}  std={ss_df[col].std():.4f}")

    # Test rest features
    print("\n[2] Rest features for 2026-02-01...")
    rest_df = compute_rest_features('2026-02-01', ['EDM', 'TOR', 'BOS', 'TB', 'DET', 'COL'])
    print(rest_df.to_string())

    # Test B2B detection across backtest dates
    print("\n[3] B2B detection across multiple dates...")
    conn = sqlite3.connect(DB_PATH)
    dates = pd.read_sql(
        "SELECT DISTINCT slate_date FROM dk_salaries WHERE slate_date >= '2026-01-15' ORDER BY slate_date",
        conn
    )
    for _, row in dates.iterrows():
        d = row['slate_date']
        teams = pd.read_sql(f"SELECT DISTINCT team FROM dk_salaries WHERE slate_date='{d}'", conn)
        rest = compute_rest_features(d, teams['team'].tolist())
        b2b_teams = rest[rest['rest_is_b2b'] == 1]['team'].tolist()
        if b2b_teams:
            print(f"  {d}: B2B teams: {', '.join(b2b_teams)}")
    conn.close()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
