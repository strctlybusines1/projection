"""
Transformer-based NHL DFS Projection Model v1

Architecture: Self-attention over player feature groups, allowing the model
to learn WHICH features matter for WHICH players in WHICH matchups.

Feature Groups (each gets its own embedding):
  1. Rolling Performance: 5g/10g rolling stats from boxscores (what MDN v3 uses)
  2. Season Profile: MoneyPuck season-level talent metrics (xG, gameScore, PP usage)
  3. Matchup Context: Opponent team quality from MoneyPuck teams
  4. Structural: Position, games played, TOI trend

The attention mechanism learns cross-group interactions:
  - "This player has high PP usage AND faces a weak PK" → boost
  - "This player has declining rolling stats BUT strong season profile" → context

Walk-forward backtest: Nov 7 → Feb 5, retrain every 14 days (same as MDN v3).
Target: Beat MDN v3 MAE of 4.091.

Author: Claude
Date: 2026-02-17
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==============================================================================
# CONFIG
# ==============================================================================

DB_PATH = Path(__file__).parent / 'data' / 'nhl_dfs_history.db'

BACKTEST_START = datetime(2025, 11, 7)
BACKTEST_END = datetime(2026, 2, 5)
RETRAIN_INTERVAL = 14

BATCH_SIZE = 256
MAX_EPOCHS = 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature group definitions
ROLLING_FEATURES = [
    'rolling_dk_fpts_5g', 'rolling_dk_fpts_10g',
    'rolling_goals_5g', 'rolling_goals_10g',
    'rolling_assists_5g', 'rolling_assists_10g',
    'rolling_shots_5g', 'rolling_shots_10g',
    'rolling_blocked_shots_5g', 'rolling_blocked_shots_10g',
    'rolling_toi_seconds_5g', 'rolling_toi_seconds_10g',
    'dk_fpts_ewm',
]

SEASON_FEATURES = [
    'season_avg_dk_fpts', 'season_avg_goals', 'season_avg_assists',
    'season_avg_shots', 'season_avg_blocked_shots', 'season_avg_toi_seconds',
    'log_gp', 'toi_seconds_trend',
]

# MoneyPuck player talent features (season-level, per-game rates)
MP_PLAYER_FEATURES = [
    'mp_gameScore_pg', 'mp_xG_pg_5v5', 'mp_shots_pg_5v5',
    'mp_pp_ice_pg', 'mp_pp_xG_pg', 'mp_pp_points_pg',
    'mp_onIce_xGF_pct', 'mp_ozone_pct',
]

# MoneyPuck opponent quality features
MP_OPPONENT_FEATURES = [
    'mp_opp_xGA_pg', 'mp_opp_hdxGA_pg', 'mp_opp_xGF_pct',
    'opp_fpts_allowed_10g',  # From v3 (daily rolling, not season-level)
]

# Existing v3 features
NST_FEATURES = [
    'opp_xgf_pct', 'opp_sv_pct',
    'pp_toi_per_game', 'ev_ixg', 'ev_toi_per_game', 'pp_ixg', 'oi_hdcf_pct',
]

POSITION_FEATURES = ['pos_C', 'pos_D', 'pos_L', 'pos_R']


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_boxscore_data():
    """Load boxscore data with precomputed rolling features."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT player_name, player_id, team, position, game_date, opponent,
               goals, assists, shots, hits, blocked_shots, plus_minus,
               pp_goals, toi_seconds, dk_fpts, game_id
        FROM boxscore_skaters
        ORDER BY game_date, player_id
    """, conn)
    conn.close()

    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    # Rolling stats
    for window in [5, 10]:
        for col in ['goals', 'assists', 'shots', 'blocked_shots', 'dk_fpts', 'toi_seconds']:
            df[f'rolling_{col}_{window}g'] = (
                df.groupby('player_id')[col]
                .rolling(window=window, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )

    # EWM
    df['dk_fpts_ewm'] = (
        df.groupby('player_id')['dk_fpts']
        .ewm(halflife=15, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    # Season averages
    for col in ['goals', 'assists', 'shots', 'blocked_shots', 'dk_fpts', 'toi_seconds']:
        cumsum = df.groupby('player_id')[col].cumsum()
        gp = df.groupby('player_id').cumcount() + 1
        df[f'season_avg_{col}'] = cumsum / gp

    # TOI trend
    rolling_toi_5 = (
        df.groupby('player_id')['toi_seconds']
        .rolling(window=5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    season_avg_toi = df.groupby('player_id')['toi_seconds'].cumsum() / (df.groupby('player_id').cumcount() + 1)
    df['toi_seconds_trend'] = rolling_toi_5 / (season_avg_toi + 1e-6)

    df['season_gp'] = df.groupby('player_id').cumcount() + 1
    df['log_gp'] = np.log1p(df['season_gp'])

    return df


def load_moneypuck_data():
    """Load MoneyPuck season-level data for player talent and opponent quality."""
    conn = sqlite3.connect(DB_PATH)

    # Skater data (5v5 + PP for current season)
    mp_sk = pd.read_sql("""
        SELECT * FROM mp_skaters
        WHERE season = 2025 AND situation IN ('5on5', '5on4')
    """, conn)

    # Team data (5v5 for opponent quality)
    mp_tm = pd.read_sql("""
        SELECT * FROM mp_teams
        WHERE season = 2025 AND situation = '5on5'
    """, conn)

    conn.close()

    # Process skater 5v5 features
    sk5v5 = mp_sk[mp_sk['situation'] == '5on5'].copy()
    gp = sk5v5['games_played'].clip(lower=1)
    sk5v5['mp_xG_pg_5v5'] = sk5v5['I_F_xGoals'] / gp
    sk5v5['mp_hd_xG_pg_5v5'] = sk5v5['I_F_highDangerxGoals'] / gp
    sk5v5['mp_shots_pg_5v5'] = sk5v5['I_F_shotsOnGoal'] / gp
    sk5v5['mp_gameScore_pg'] = sk5v5['gameScore'] / gp
    sk5v5['mp_onIce_xGF_pct'] = sk5v5['onIce_xGoalsPercentage']
    sk5v5['mp_ozone_pct'] = sk5v5['I_F_oZoneShiftStarts'] / (
        sk5v5['I_F_oZoneShiftStarts'] + sk5v5['I_F_dZoneShiftStarts'] + 1
    )

    # Process skater PP features
    skpp = mp_sk[mp_sk['situation'] == '5on4'].copy()
    gp_pp = skpp['games_played'].clip(lower=1)
    skpp['mp_pp_xG_pg'] = skpp['I_F_xGoals'] / gp_pp
    skpp['mp_pp_points_pg'] = skpp['I_F_points'] / gp_pp
    skpp['mp_pp_ice_pg'] = skpp['icetime'] / gp_pp

    # Process team opponent quality
    gp_tm = mp_tm['games_played'].clip(lower=1)
    mp_tm['mp_opp_xGA_pg'] = mp_tm['xGoalsAgainst'] / gp_tm
    mp_tm['mp_opp_hdxGA_pg'] = mp_tm['highDangerxGoalsAgainst'] / gp_tm
    mp_tm['mp_opp_xGF_pct'] = mp_tm['xGoalsPercentage']

    return sk5v5, skpp, mp_tm


def strip_accents(s):
    """Strip diacritics from a string."""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s))
                   if unicodedata.category(c) != 'Mn')


def name_match_key(name):
    """Create matching key: last_firstinit, accent-stripped."""
    if pd.isna(name):
        return ''
    name = strip_accents(str(name).strip())
    name_lower = name.lower()
    for umlaut, repl in [('ue', 'u'), ('ae', 'a'), ('oe', 'o')]:
        name_lower = name_lower.replace(umlaut, repl)
    parts = name_lower.split()
    if len(parts) < 2:
        return name_lower
    return f"{parts[-1]}_{parts[0][0]}"


def merge_moneypuck_features(df, sk5v5, skpp, mp_tm):
    """Merge MoneyPuck features onto boxscore data."""
    print("Merging MoneyPuck features...")

    # Build player lookup by ID first, then name fallback
    sk5v5_cols = ['playerId', 'mp_xG_pg_5v5', 'mp_hd_xG_pg_5v5', 'mp_shots_pg_5v5',
                  'mp_gameScore_pg', 'mp_onIce_xGF_pct', 'mp_ozone_pct']
    skpp_cols = ['playerId', 'mp_pp_xG_pg', 'mp_pp_points_pg', 'mp_pp_ice_pg']

    # Merge by player_id
    df = df.merge(sk5v5[sk5v5_cols], left_on='player_id', right_on='playerId', how='left')
    df = df.merge(skpp[skpp_cols], left_on='player_id', right_on='playerId', how='left',
                  suffixes=('', '_pp'))

    # For unmatched, try name matching
    unmatched_mask = df['mp_gameScore_pg'].isna()
    if unmatched_mask.sum() > 0:
        df['_match_key'] = df['player_name'].apply(name_match_key)
        sk5v5['_match_key'] = sk5v5['name'].apply(name_match_key)
        skpp['_match_key'] = skpp['name'].apply(name_match_key)

        sk5v5_by_key = sk5v5.drop_duplicates('_match_key').set_index('_match_key')
        skpp_by_key = skpp.drop_duplicates('_match_key').set_index('_match_key')

        for idx in df[unmatched_mask].index:
            key = df.loc[idx, '_match_key']
            if key in sk5v5_by_key.index:
                for col in ['mp_xG_pg_5v5', 'mp_hd_xG_pg_5v5', 'mp_shots_pg_5v5',
                            'mp_gameScore_pg', 'mp_onIce_xGF_pct', 'mp_ozone_pct']:
                    df.loc[idx, col] = sk5v5_by_key.loc[key, col]
            if key in skpp_by_key.index:
                for col in ['mp_pp_xG_pg', 'mp_pp_points_pg', 'mp_pp_ice_pg']:
                    df.loc[idx, col] = skpp_by_key.loc[key, col]

    # Merge opponent team quality
    opp_cols = ['team', 'mp_opp_xGA_pg', 'mp_opp_hdxGA_pg', 'mp_opp_xGF_pct']
    df = df.merge(mp_tm[opp_cols], left_on='opponent', right_on='team',
                  how='left', suffixes=('', '_opp_team'))

    # Fill NaN with medians
    mp_all_cols = [c for c in df.columns if c.startswith('mp_')]
    for col in mp_all_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median if not pd.isna(median) else 0)

    matched = (df['mp_gameScore_pg'] != 0).sum()
    print(f"  MoneyPuck match rate: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    return df


def compute_opponent_fpts_allowed(df):
    """Compute trailing 10-game average FPTS allowed per team (from MDN v3)."""
    daily_team_fpts = df.groupby(['game_date', 'team'])['dk_fpts'].sum().reset_index()
    daily_team_fpts.columns = ['game_date', 'team', 'total_fpts']

    team_opponent = df[['game_date', 'team', 'opponent']].drop_duplicates()
    team_opponent = team_opponent.merge(
        daily_team_fpts.rename(columns={'team': 'opponent', 'total_fpts': 'opp_fpts'}),
        on=['game_date', 'opponent'], how='left'
    )
    team_opponent['opp_fpts'] = team_opponent['opp_fpts'].fillna(0)
    team_opponent = team_opponent.sort_values(['team', 'game_date'])
    team_opponent['opp_fpts_allowed_10g'] = (
        team_opponent.groupby('team')['opp_fpts']
        .rolling(window=10, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df = df.merge(
        team_opponent[['game_date', 'team', 'opp_fpts_allowed_10g']],
        on=['game_date', 'team'], how='left'
    )
    df['opp_fpts_allowed_10g'] = df['opp_fpts_allowed_10g'].fillna(0)
    return df


# ==============================================================================
# NST LOADING (from MDN v3 — keep for compatibility)
# ==============================================================================

_nst_cache = {}
_nst_cache_date = None

def load_nst_teams_for_date(date_str):
    global _nst_cache, _nst_cache_date
    if _nst_cache and _nst_cache_date == date_str[:10]:
        return _nst_cache
    conn = sqlite3.connect(DB_PATH)
    nst_df = pd.read_sql(f"""
        SELECT team, situation, xgf_pct, hdcf_pct, sv_pct, to_date
        FROM nst_teams WHERE to_date <= '{date_str}' AND situation = '5v5'
    """, conn)
    conn.close()
    if nst_df.empty:
        result = {}
    else:
        nst_df['to_date'] = pd.to_datetime(nst_df['to_date'])
        nst_df = nst_df.sort_values('to_date')
        latest = nst_df.groupby('team').tail(1)
        result = {}
        for _, row in latest.iterrows():
            result[row['team']] = {
                'xgf_pct': row['xgf_pct'] / 100 if pd.notna(row['xgf_pct']) else 0.5,
                'sv_pct': row['sv_pct'] / 100 if pd.notna(row['sv_pct']) else 0.91,
            }
    _nst_cache = result
    _nst_cache_date = date_str[:10]
    return result


_nst_skater_cache = {}
_nst_skater_cache_date = None
_fuzzy_match_cache = {}

def load_nst_skaters_for_date(date_str):
    global _nst_skater_cache, _nst_skater_cache_date
    if _nst_skater_cache and _nst_skater_cache_date == date_str[:10]:
        return _nst_skater_cache, _nst_skater_cache_date
    conn = sqlite3.connect(DB_PATH)
    df_dates = pd.read_sql(f"""
        SELECT DISTINCT to_date FROM nst_skaters
        WHERE to_date <= '{date_str}' AND from_date = '2025-10-07'
        ORDER BY to_date DESC LIMIT 1
    """, conn)
    if df_dates.empty:
        conn.close()
        _nst_skater_cache = {}
        _nst_skater_cache_date = date_str[:10]
        return {}, None
    latest_date = df_dates['to_date'].iloc[0]
    df = pd.read_sql(f"""
        SELECT player, situation, stat_type, ixg, toi, gp, hdcf_pct
        FROM nst_skaters WHERE to_date = '{latest_date}' AND from_date = '2025-10-07'
    """, conn)
    conn.close()
    result = {}
    for _, row in df.iterrows():
        key = (row['player'], row['situation'], row['stat_type'])
        result[key] = row
    _nst_skater_cache = result
    _nst_skater_cache_date = date_str[:10]
    return result, latest_date


def fuzzy_match_names(boxscore_name, nst_names, threshold=0.6):
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


# ==============================================================================
# FEATURE MATRIX BUILDER
# ==============================================================================

def build_feature_matrix(df, predict_date, train_cutoff=None):
    """Build feature matrix with all feature groups for a prediction date."""
    predict_games = df[df['game_date'] == predict_date].copy()
    if predict_games.empty:
        return None, None, None, None, None

    # Position one-hot
    positions = pd.get_dummies(predict_games['position'], prefix='pos', drop_first=False)

    # NST opponent features (from v3)
    nst_teams = load_nst_teams_for_date(predict_date.strftime('%Y-%m-%d'))
    predict_games['opp_xgf_pct'] = predict_games['opponent'].map(
        lambda x: nst_teams.get(x, {}).get('xgf_pct', 0.5))
    predict_games['opp_sv_pct'] = predict_games['opponent'].map(
        lambda x: nst_teams.get(x, {}).get('sv_pct', 0.91))

    # NST skater features
    nst_data, nst_date = load_nst_skaters_for_date(predict_date.strftime('%Y-%m-%d'))
    for col in ['pp_toi_per_game', 'ev_ixg', 'ev_toi_per_game', 'pp_ixg', 'oi_hdcf_pct']:
        predict_games[col] = 0.0

    if nst_data:
        nst_player_names = list(set(key[0] for key in nst_data.keys()))
        for idx, row in predict_games.iterrows():
            matched = fuzzy_match_names(row['player_name'], nst_player_names)
            if matched:
                pp_row = nst_data.get((matched, 'pp', 'std'))
                if pp_row is not None:
                    toi = pp_row.get('toi')
                    gp = pp_row.get('gp')
                    if pd.notna(toi) and pd.notna(gp) and gp > 0:
                        predict_games.loc[idx, 'pp_toi_per_game'] = float(toi) / gp
                    ixg = pp_row.get('ixg')
                    if pd.notna(ixg):
                        predict_games.loc[idx, 'pp_ixg'] = float(ixg)

                ev_row = nst_data.get((matched, '5v5', 'std'))
                if ev_row is not None:
                    ixg = ev_row.get('ixg')
                    if pd.notna(ixg):
                        predict_games.loc[idx, 'ev_ixg'] = float(ixg)
                    toi = ev_row.get('toi')
                    gp = ev_row.get('gp')
                    if pd.notna(toi) and pd.notna(gp) and gp > 0:
                        predict_games.loc[idx, 'ev_toi_per_game'] = float(toi) / gp

                oi_row = nst_data.get((matched, '5v5', 'oi'))
                if oi_row is not None:
                    hdcf = oi_row.get('hdcf_pct')
                    if pd.notna(hdcf):
                        predict_games.loc[idx, 'oi_hdcf_pct'] = float(hdcf) / 100

    # Interaction features
    predict_games['hdcf_x_opp_weak'] = predict_games['oi_hdcf_pct'] * (1.0 - predict_games['opp_xgf_pct'])

    # Collect all feature columns
    all_features = []

    # Rolling features
    for col in ROLLING_FEATURES:
        if col in predict_games.columns:
            all_features.append(col)

    # Season features
    for col in SEASON_FEATURES:
        if col in predict_games.columns:
            all_features.append(col)

    # MoneyPuck player features
    for col in MP_PLAYER_FEATURES:
        if col in predict_games.columns:
            all_features.append(col)

    # MoneyPuck opponent features
    for col in MP_OPPONENT_FEATURES:
        if col in predict_games.columns:
            all_features.append(col)

    # NST features
    for col in NST_FEATURES:
        if col in predict_games.columns:
            all_features.append(col)

    # Interaction
    all_features.append('hdcf_x_opp_weak')

    X = predict_games[all_features].reset_index(drop=True).copy()

    # Add positions
    positions_reset = positions.reset_index(drop=True)
    X = pd.concat([X, positions_reset], axis=1)

    y = predict_games['dk_fpts'].values
    player_ids = predict_games['player_id'].values
    player_names = predict_games['player_name'].values
    positions_val = predict_games['position'].values

    return X, y, player_ids, player_names, positions_val


def prepare_training_data(df, train_end_date):
    """Prepare training data using all games up to train_end_date."""
    train_df = df[df['game_date'] <= train_end_date].copy()
    if train_df.empty:
        return None, None, None

    pred_dates = sorted(train_df['game_date'].unique())

    # Sample dates for efficiency (use every other date for training)
    if len(pred_dates) > 40:
        pred_dates = pred_dates[::2]  # Every other date
    print(f"  Using {len(pred_dates)} training dates")

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
    X = X.fillna(0)

    X_mean = X.mean()
    X_std = X.std()
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std

    return (
        torch.FloatTensor(X.values.astype(np.float32)),
        torch.FloatTensor(y.astype(np.float32)),
        (X_mean, X_std)
    )


# ==============================================================================
# TRANSFORMER MODEL
# ==============================================================================

class GroupedFeatureTokenizer(nn.Module):
    """
    Tokenize feature GROUPS into embeddings (more efficient than per-feature).
    Each group of features gets a single linear projection to d_model.
    This creates a shorter sequence for the transformer (5-6 tokens vs 30+).
    """
    def __init__(self, group_sizes, d_model):
        super().__init__()
        self.group_sizes = group_sizes  # list of (start_idx, end_idx) for each group
        self.projections = nn.ModuleList([
            nn.Linear(end - start, d_model)
            for start, end in group_sizes
        ])

    def forward(self, x):
        tokens = []
        for i, (start, end) in enumerate(self.group_sizes):
            group_features = x[:, start:end]  # (batch, group_size)
            token = self.projections[i](group_features)  # (batch, d_model)
            tokens.append(token)
        return torch.stack(tokens, dim=1)  # (batch, n_groups, d_model)


class TransformerProjectionModel(nn.Module):
    """
    Transformer-based DFS projection model.

    Architecture:
    1. Feature Tokenizer: Each input feature → d_model embedding
    2. Positional encoding for feature groups
    3. Multi-head self-attention (2 layers)
    4. [CLS] token aggregation → prediction head
    5. Output: point estimate + uncertainty (mean, std)
    """

    def __init__(self, n_features, d_model=64, nhead=4, n_layers=2, dropout=0.15,
                 feature_group_sizes=None):
        super().__init__()

        self.d_model = d_model
        self.n_features = n_features

        # Feature group tokenizer (efficient: groups → tokens)
        if feature_group_sizes is None:
            # Default: split evenly into ~6 groups
            chunk = max(1, n_features // 6)
            feature_group_sizes = []
            for i in range(0, n_features, chunk):
                feature_group_sizes.append((i, min(i + chunk, n_features)))
        self.n_groups = len(feature_group_sizes)
        self.tokenizer = GroupedFeatureTokenizer(feature_group_sizes, d_model)

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learnable, one per group + CLS)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_groups + 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Prediction head: from [CLS] token → (mean, log_std)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),  # mean, log_std
        )

    def forward(self, x):
        # x: (batch, n_features)
        batch_size = x.shape[0]

        # Tokenize features
        tokens = self.tokenizer(x)  # (batch, n_features, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_features+1, d_model)

        # Add positional encoding
        tokens = tokens + self.pos_encoding[:, :tokens.shape[1], :]

        # Transformer
        tokens = self.transformer(tokens)  # (batch, n_features+1, d_model)

        # Extract [CLS] representation
        cls_out = self.norm(tokens[:, 0, :])  # (batch, d_model)

        # Predict mean and std
        output = self.head(cls_out)  # (batch, 2)
        mean = output[:, 0]
        log_std = output[:, 1]
        std = torch.nn.functional.softplus(log_std) + 0.5  # Minimum std of 0.5

        return mean, std

    def loss(self, mean, std, y):
        """Gaussian negative log-likelihood loss."""
        nll = 0.5 * torch.log(2 * np.pi * std**2) + 0.5 * ((y - mean) / std)**2
        return nll.mean()


# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(X_train, y_train, X_val, y_val):
    """Train transformer with early stopping."""
    input_size = X_train.shape[1]

    # Define feature groups matching our feature categories
    # This lets the transformer attend across conceptual groups
    n_rolling = len(ROLLING_FEATURES)
    n_season = len(SEASON_FEATURES)
    n_mp_player = len(MP_PLAYER_FEATURES)
    n_mp_opp = len(MP_OPPONENT_FEATURES)
    # Remaining: NST + interaction + position features
    n_remaining = input_size - n_rolling - n_season - n_mp_player - n_mp_opp

    idx = 0
    groups = []
    for size in [n_rolling, n_season, n_mp_player, n_mp_opp, n_remaining]:
        if size > 0:
            groups.append((idx, idx + size))
            idx += size

    model = TransformerProjectionModel(
        n_features=input_size,
        d_model=64,
        nhead=4,
        n_layers=2,
        dropout=0.15,
        feature_group_sizes=groups,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    max_patience = 12

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            mean, std = model(X_batch)
            loss = model.loss(mean, std, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train)
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                mean, std = model(X_batch)
                loss = model.loss(mean, std, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(X_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"    Early stop at epoch {epoch+1}, val_loss={val_loss:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, X):
    """Get predictions from transformer."""
    model.eval()
    with torch.no_grad():
        X_array = X.values.astype(np.float64).astype(np.float32)
        X_tensor = torch.FloatTensor(X_array).to(DEVICE)
        mean, std = model(X_tensor)
    return mean.cpu().numpy(), std.cpu().numpy()


# ==============================================================================
# WALK-FORWARD BACKTEST
# ==============================================================================

def run_backtest(df):
    """Walk-forward backtest: Nov 7 → Feb 5, retrain every 14 days."""
    print(f"\n{'='*80}")
    print(f"TRANSFORMER v1 WALK-FORWARD BACKTEST")
    print(f"{'='*80}")
    print(f"  Period: {BACKTEST_START.date()} to {BACKTEST_END.date()}")
    print(f"  Retrain every {RETRAIN_INTERVAL} days")
    print(f"  Architecture: d_model=64, nhead=4, layers=2")
    print()

    results = []
    current_date = BACKTEST_START
    last_retrain_date = datetime(2025, 10, 7)
    model = None
    norm_stats = None

    while current_date <= BACKTEST_END:
        # Retrain if needed
        if model is None or (current_date - last_retrain_date).days >= RETRAIN_INTERVAL:
            train_end = current_date - timedelta(days=1)
            print(f"\n>>> Retraining on data through {train_end.date()}")

            X_train, y_train, norm_stats = prepare_training_data(df, train_end)
            if X_train is None or len(X_train) < 100:
                current_date += timedelta(days=1)
                continue

            n_train = int(0.8 * len(X_train))
            idx = torch.randperm(len(X_train))
            X_tr = X_train[idx[:n_train]]
            y_tr = y_train[idx[:n_train]]
            X_va = X_train[idx[n_train:]]
            y_va = y_train[idx[n_train:]]

            print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Features: {X_train.shape[1]}")
            model = train_model(X_tr, y_tr, X_va, y_va)
            last_retrain_date = current_date

        # Predict
        X_pred, y_actual, player_ids, player_names, positions = \
            build_feature_matrix(df, current_date)

        if X_pred is None or len(X_pred) == 0:
            current_date += timedelta(days=1)
            continue

        X_mean, X_std = norm_stats
        X_pred_norm = (X_pred.fillna(0) - X_mean) / (X_std + 1e-6)

        predicted_fpts, predicted_std = predict(model, X_pred_norm)

        for i in range(len(y_actual)):
            results.append({
                'game_date': current_date.date(),
                'player_id': player_ids[i],
                'player_name': player_names[i],
                'position': positions[i],
                'actual_fpts': y_actual[i],
                'predicted_fpts': predicted_fpts[i],
                'std_fpts': predicted_std[i],
            })

        print(f"  {current_date.date()}: {len(X_pred)} players")
        current_date += timedelta(days=1)

    return pd.DataFrame(results)


# ==============================================================================
# METRICS
# ==============================================================================

def print_results(results_df):
    """Print comprehensive results table."""
    if results_df.empty:
        print("No results")
        return

    actual = results_df['actual_fpts'].values
    predicted = results_df['predicted_fpts'].values

    mae = np.abs(actual - predicted).mean()
    rmse = np.sqrt(((actual - predicted)**2).mean())
    corr = np.corrcoef(actual, predicted)[0, 1]

    print(f"\n{'='*80}")
    print(f"TRANSFORMER v1 — RESULTS")
    print(f"{'='*80}")
    print(f"\n  Overall MAE:  {mae:.3f}")
    print(f"  Overall RMSE: {rmse:.3f}")
    print(f"  Correlation:  {corr:.3f}")
    print(f"  Predictions:  {len(results_df)}")
    print()

    # vs benchmarks
    print(f"  {'Model':<25s} {'MAE':>8}")
    print(f"  {'-'*35}")
    print(f"  {'Transformer v1':<25s} {mae:>8.3f}")
    print(f"  {'MDN v3 (baseline)':<25s} {'4.091':>8}")
    print(f"  {'MDN v1':<25s} {'4.107':>8}")
    print(f"  {'Kalman filter':<25s} {'4.318':>8}")
    print(f"  {'Poisson sim':<25s} {'4.749':>8}")
    improvement = (4.091 - mae) / 4.091 * 100
    print(f"\n  vs MDN v3: {improvement:+.2f}%")

    # By position
    print(f"\n  BY POSITION:")
    print(f"  {'Pos':<6} {'N':>6} {'MAE':>8} {'Corr':>8}")
    print(f"  {'-'*30}")
    for pos in sorted(results_df['position'].unique()):
        pos_data = results_df[results_df['position'] == pos]
        pos_mae = np.abs(pos_data['actual_fpts'] - pos_data['predicted_fpts']).mean()
        pos_corr = np.corrcoef(pos_data['actual_fpts'], pos_data['predicted_fpts'])[0, 1]
        print(f"  {pos:<6} {len(pos_data):>6} {pos_mae:>8.3f} {pos_corr:>8.3f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transformer NHL DFS Projection')
    parser.add_argument('--backtest', action='store_true')
    args = parser.parse_args()

    print("Loading boxscore data...")
    df = load_boxscore_data()
    print(f"  {len(df)} records, {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    print("Loading MoneyPuck data...")
    sk5v5, skpp, mp_tm = load_moneypuck_data()
    print(f"  Skaters 5v5: {len(sk5v5)}, PP: {len(skpp)}, Teams: {len(mp_tm)}")

    print("Merging features...")
    df = merge_moneypuck_features(df, sk5v5, skpp, mp_tm)
    df = compute_opponent_fpts_allowed(df)

    if args.backtest:
        results = run_backtest(df)
        print_results(results)

        out = Path(__file__).parent / 'data' / 'transformer_v1_backtest_results.csv'
        results.to_csv(out, index=False)
        print(f"\nSaved to {out}")
    else:
        print("\nUsage: python transformer_v1.py --backtest")


if __name__ == '__main__':
    main()
