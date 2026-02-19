"""
Enhanced Mixture Density Network (MDN v2) for NHL DFS Projections.

Improvements over v1:
- Multi-season pre-training on historical data (2020-2024: 128K+ rows)
- Opponent quality feature (avg FPTS allowed by team)
- Regression-weighted features (shrinkage toward league average for unstable stats)
- Fine-tuning on current season with walk-forward backtest
- Larger hidden layers (128 vs 64) for better representation learning

Architecture:
- Pre-train on historical_skaters (all 4 seasons)
- Fine-tune on boxscore_skaters (2024-25 season)
- Opponent quality: rolling 10-game average FPTS allowed by opponent
- Regression weights: blend unstable stats toward league average
- Hidden: 2 layers of 128 units with 0.1 dropout
- Output: K=3 Gaussian mixture components (π, μ, σ)

Walk-forward backtest:
- Train from Oct 7 through the day before prediction
- Retrain every 14 days
- Evaluate Nov 7 through Feb 5
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

# Regression weights from YoY analysis (shrinkage factors)
REGRESSION_WEIGHTS = {
    'dk_fpts_pg': {'r': 0.806, 'shrinkage': 0.194},
    'shots_pg': {'r': 0.823, 'shrinkage': 0.177},
    'blocks_pg': {'r': 0.869, 'shrinkage': 0.131},
    'hits_pg': {'r': 0.829, 'shrinkage': 0.171},
    'toi_per_game': {'r': 0.846, 'shrinkage': 0.154},
    'goals_pg': {'r': 0.712, 'shrinkage': 0.261},
    'assists_pg': {'r': 0.735, 'shrinkage': 0.259},
}

# Minimum sample sizes for reliability (split-half >0.70)
MIN_SAMPLES = {
    'hits': 15,
    'blocks': 20,
    'shots': 20,
    'pp_assists': 20,
    'dk_fpts': 25,
    'goals': 60,
}


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_historical_data():
    """Load historical skaters data (2020-2024)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT
            season, player_name, player_id, team, position, game_date, opponent,
            goals, assists, shots, blocked_shots, dk_fpts, toi_seconds,
            pp_goals, hits, plus_minus
        FROM historical_skaters
        WHERE season IN (2020, 2021, 2022, 2023)
        ORDER BY season, game_date, player_name
    """, conn)
    conn.close()

    df['game_date'] = pd.to_datetime(df['game_date'])
    # Note: historical_skaters has 'blocked_shots', which we use as 'blocks'
    df = df.sort_values(['season', 'player_name', 'game_date']).reset_index(drop=True)

    return df


def load_boxscore_data():
    """Load current season boxscore data (2024-25)."""
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
    df['season'] = 2024  # Current season identifier
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    return df


def compute_opponent_quality(df, window=10):
    """
    Compute opponent defensive quality as rolling average of FPTS allowed.
    For each opponent on each date, compute the average dk_fpts scored against them
    over the last `window` games.

    Returns dict: {(opponent, date): avg_fpts_allowed}
    """
    opp_quality = {}

    # Group by opponent and date, compute average FPTS allowed
    for opponent in df['opponent'].unique():
        opp_games = df[df['opponent'] == opponent].copy()
        opp_games = opp_games.sort_values('game_date').reset_index(drop=True)

        # Rolling average of FPTS scored against this opponent
        rolling_fpts = opp_games['dk_fpts'].rolling(window=window, min_periods=1).mean()

        for idx, (date, fpts) in enumerate(zip(opp_games['game_date'], rolling_fpts)):
            # Use this value for predictions AFTER this date
            next_date = date + timedelta(days=1)
            key = (opponent, next_date)
            opp_quality[key] = fpts

    return opp_quality


def build_rolling_features(df, window=None):
    """
    Precompute rolling features for efficiency.
    Handles both current season (grouped by player_id) and historical data (grouped by season+player).
    """
    print("Precomputing rolling features...")

    if window is None:
        windows = [5, 10]
    else:
        windows = [window]

    # Determine groupby key based on whether 'season' column exists
    if 'season' in df.columns:
        # Historical data: group by (season, player_name)
        groupby_key = ['season', 'player_name']
    else:
        # Current data: group by player_id
        groupby_key = 'player_id'

    for col in ['goals', 'assists', 'shots', 'blocked_shots', 'dk_fpts', 'toi_seconds']:
        for w in windows:
            feat_name = f'rolling_{col}_{w}g'
            rolling_vals = (
                df.groupby(groupby_key, sort=False)[col]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            rolling_vals.index = df.index
            df[feat_name] = rolling_vals

    # EWM FPTS (halflife=15 games)
    ewm_vals = (
        df.groupby(groupby_key, sort=False)['dk_fpts']
        .ewm(halflife=15, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    ewm_vals.index = df.index
    df['dk_fpts_ewm'] = ewm_vals

    # Season-to-date stats
    for col in ['goals', 'assists', 'shots', 'blocked_shots', 'dk_fpts', 'toi_seconds']:
        cumsum = df.groupby(groupby_key, sort=False)[col].cumsum()
        gp = df.groupby(groupby_key, sort=False).cumcount() + 1
        cumsum.index = df.index
        gp.index = df.index
        df[f'season_avg_{col}'] = cumsum / gp

    # TOI trend: last 5 games avg / season avg
    rolling_toi_5 = (
        df.groupby(groupby_key, sort=False)['toi_seconds']
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_toi_5.index = df.index

    cumsum_toi = df.groupby(groupby_key, sort=False)['toi_seconds'].cumsum()
    gp = df.groupby(groupby_key, sort=False).cumcount() + 1
    cumsum_toi.index = df.index
    gp.index = df.index
    season_avg_toi = cumsum_toi / gp

    df['toi_seconds_trend'] = rolling_toi_5 / (season_avg_toi + 1e-6)

    # Games played
    gp = df.groupby(groupby_key, sort=False).cumcount() + 1
    gp.index = df.index
    df['season_gp'] = gp
    df['log_gp'] = np.log1p(df['season_gp'])

    return df


def apply_regression_weights(df, target_stats=None):
    """
    Apply shrinkage to unstable statistics by blending toward league average.

    For stats with YoY correlation < 0.80, blend the season stat toward league average
    proportional to shrinkage factor.

    new_stat = (r * observed_stat) + (shrinkage * league_avg_stat)
    """
    if target_stats is None:
        target_stats = REGRESSION_WEIGHTS.keys()

    league_stats = {}

    # Compute league averages
    for stat_type in target_stats:
        if 'goals' in stat_type:
            col = 'goals'
        elif 'assists' in stat_type:
            col = 'assists'
        elif 'blocks' in stat_type:
            col = 'blocked_shots'
        elif 'hits' in stat_type:
            col = 'hits'
        elif 'shots' in stat_type:
            col = 'shots'
        elif 'dk_fpts' in stat_type:
            col = 'dk_fpts'
        elif 'toi' in stat_type:
            col = 'toi_seconds'
        else:
            continue

        if col in df.columns and f'season_avg_{col}' in df.columns:
            league_stats[stat_type] = df[f'season_avg_{col}'].mean()

    # Apply shrinkage for stats with low YoY correlation
    for stat_type, weights in REGRESSION_WEIGHTS.items():
        if weights['r'] < 0.80:
            r = weights['r']
            shrinkage = weights['shrinkage']

            # Map stat_type to season_avg column
            if 'goals' in stat_type:
                col = 'season_avg_goals'
            elif 'assists' in stat_type:
                col = 'season_avg_assists'
            elif 'blocks' in stat_type:
                col = 'season_avg_blocked_shots'
            elif 'hits' in stat_type:
                col = 'season_avg_hits'
            elif 'shots' in stat_type:
                col = 'season_avg_shots'
            elif 'dk_fpts' in stat_type:
                col = 'season_avg_dk_fpts'
            elif 'toi' in stat_type:
                col = 'season_avg_toi_seconds'
            else:
                continue

            if col in df.columns and stat_type in league_stats:
                league_avg = league_stats[stat_type]
                # Blend: higher shrinkage = more regress toward mean
                df[f'{col}_shrunk'] = r * df[col] + shrinkage * league_avg
            else:
                # Fallback: no shrinkage
                df[f'{col}_shrunk'] = df.get(col, 0)

    return df


def get_opponent_quality_for_date(df, opp_quality_dict, pred_date, opponent):
    """Safely retrieve opponent quality for a specific date and opponent."""
    # Look for the most recent quality assessment <= pred_date
    key = (opponent, pred_date)
    if key in opp_quality_dict:
        return opp_quality_dict[key]

    # Fallback: compute from data up to pred_date
    mask = (df['opponent'] == opponent) & (df['game_date'] < pred_date)
    if mask.sum() > 0:
        return df[mask]['dk_fpts'].mean()

    # Final fallback: league average
    return df[df['game_date'] < pred_date]['dk_fpts'].mean()


# Cache for NST data
_nst_cache = {}
_nst_cache_date = None
_fuzzy_match_cache = {}


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
    """Fuzzy match player names (cached)."""
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


def build_feature_matrix(df, predict_date, opp_quality_dict=None, all_positions=None):
    """
    Build feature matrix for prediction on a given date.

    Args:
        df: DataFrame with rolling features precomputed
        predict_date: datetime to predict for
        opp_quality_dict: dict of opponent quality (optional)
        all_positions: list of all possible positions to ensure consistent columns
    """
    # Get games on predict_date
    predict_games = df[df['game_date'] == predict_date].copy()
    if predict_games.empty:
        return None, None, None, None, None

    # Position one-hot encoding with consistent columns
    if all_positions is not None:
        # Use pre-specified positions to ensure consistent columns
        positions = pd.get_dummies(predict_games['position'], prefix='pos', drop_first=False)
        for pos in all_positions:
            col_name = f'pos_{pos}'
            if col_name not in positions.columns:
                positions[col_name] = 0
        # Reorder to match all_positions
        positions = positions[[f'pos_{p}' for p in all_positions]]
    else:
        # Default: just get dummies
        positions = pd.get_dummies(predict_games['position'], prefix='pos', drop_first=False)

    # Load opponent quality
    nst_teams = load_nst_teams_for_date(predict_date.strftime('%Y-%m-%d'))
    predict_games.loc[:, 'opp_xgf_pct'] = predict_games['opponent'].map(
        lambda x: nst_teams.get(x, {}).get('xgf_pct', 0.5)
    )
    predict_games.loc[:, 'opp_sv_pct'] = predict_games['opponent'].map(
        lambda x: nst_teams.get(x, {}).get('sv_pct', 0.91)
    )

    # NEW: Opponent defensive quality (FPTS allowed)
    if opp_quality_dict is not None:
        predict_games.loc[:, 'opp_fpts_allowed'] = predict_games['opponent'].map(
            lambda x: opp_quality_dict.get((x, predict_date),
                                          df[df['game_date'] < predict_date]['dk_fpts'].mean())
        )
    else:
        predict_games.loc[:, 'opp_fpts_allowed'] = df[df['game_date'] < predict_date]['dk_fpts'].mean()

    # Collect features
    rolling_cols = [c for c in df.columns if c.startswith('rolling_')]
    season_cols = [c for c in df.columns if (c.startswith('season_avg_') or c == 'log_gp')]

    if not rolling_cols or not season_cols:
        return None, None, None, None, None

    feature_cols = []
    for col in rolling_cols + season_cols:
        if col in predict_games.columns:
            feature_cols.append(col)

    # Add shrunk features (if available)
    for col in predict_games.columns:
        if col.endswith('_shrunk'):
            feature_cols.append(col)

    # Add EWM and trend
    extra_features = []
    if 'dk_fpts_ewm' in predict_games.columns:
        extra_features.append('dk_fpts_ewm')
    if 'toi_seconds_trend' in predict_games.columns:
        extra_features.append('toi_seconds_trend')
    feature_cols.extend(extra_features)

    # Build feature matrix
    X = predict_games[feature_cols + ['opp_xgf_pct', 'opp_sv_pct', 'opp_fpts_allowed']].reset_index(drop=True).copy()

    # Interaction feature
    X['hdcf_x_opp_weak'] = 0.0  # Placeholder since we don't have oi_hdcf_pct without NST

    # Add position encoding
    positions_reset = positions.reset_index(drop=True)
    X = pd.concat([X, positions_reset], axis=1)

    y = predict_games['dk_fpts'].values
    player_ids = predict_games['player_id'].values
    player_names = predict_games['player_name'].values
    positions_val = predict_games['position'].values

    return X, y, player_ids, player_names, positions_val


def prepare_training_data(df, train_end_date, opp_quality_dict=None):
    """Prepare training data with normalization."""
    train_df = df[df['game_date'] <= train_end_date].copy()

    if train_df.empty:
        return None, None, None

    # Get all possible positions in the training data
    all_positions = sorted(train_df['position'].unique())
    print(f"  Positions found: {all_positions}")

    # Get unique prediction dates (sample to speed up)
    pred_dates = sorted(train_df['game_date'].unique())
    sample_interval = max(1, len(pred_dates) // 12)
    pred_dates = pred_dates[::sample_interval]

    print(f"  Using {len(pred_dates)} dates for training (sampled from {len(train_df['game_date'].unique())})")

    X_list, y_list = [], []

    for pred_date in pred_dates:
        X, y, _, _, _ = build_feature_matrix(train_df, pred_date, opp_quality_dict, all_positions=all_positions)
        if X is not None and len(y) > 0 and len(X) == len(y):
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        return None, None, None

    X = pd.concat(X_list, ignore_index=True)
    y = np.concatenate(y_list)

    assert len(X) == len(y), f"Shape mismatch: X={len(X)}, y={len(y)}"

    X = X.fillna(0)

    # Normalize
    X_mean = X.mean()
    X_std = X.std()
    # Replace zero or very small std with 1.0 to avoid division issues
    X_std[X_std < 0.1] = 1.0
    X = (X - X_mean) / X_std

    X_array = X.values.astype(np.float32)
    y_array = y.astype(np.float32)

    return torch.FloatTensor(X_array), torch.FloatTensor(y_array), (X_mean, X_std, all_positions)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MixtureDesityNetwork(nn.Module):
    """Mixture Density Network with K Gaussian components and dropout."""

    def __init__(self, input_size, hidden_size=128, k=3, dropout_rate=0.1):
        super().__init__()
        self.k = k

        # Feature extraction with dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layers for mixture parameters
        self.pi_layer = nn.Linear(hidden_size, k)
        self.mu_layer = nn.Linear(hidden_size, k)
        self.sigma_layer = nn.Linear(hidden_size, k)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.dropout1(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout2(h)

        # Mixing coefficients
        pi = torch.softmax(self.pi_layer(h), dim=-1)

        # Means
        mu = self.mu_layer(h)

        # Standard deviations (ensure positive)
        sigma = torch.nn.functional.softplus(self.sigma_layer(h)) + 1e-6

        return pi, mu, sigma

    def loss(self, pi, mu, sigma, y):
        """Negative log-likelihood of Gaussian mixture model."""
        y = y.unsqueeze(1)  # (batch_size, 1)

        log_sigma = torch.log(sigma)
        normalized = (y - mu) / sigma

        log_gaussian = -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * normalized**2
        log_mixture = torch.log(pi + 1e-10) + log_gaussian

        max_log = torch.max(log_mixture, dim=1, keepdim=True)[0]
        log_prob = max_log + torch.logsumexp(log_mixture - max_log, dim=1, keepdim=True)

        return -torch.mean(log_prob)


# ============================================================================
# TRAINING WITH PRE-TRAINING AND FINE-TUNING
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, pretrained_model=None, fine_tune_lr=1e-4):
    """
    Train MDN model with optional pre-trained weights and lower learning rate for fine-tuning.
    """
    input_size = X_train.shape[1]

    if pretrained_model is not None:
        # Start from pre-trained model
        model = pretrained_model
        lr = fine_tune_lr  # Lower LR for fine-tuning
        print(f"  Fine-tuning from pre-trained model (lr={lr})")
    else:
        # Train from scratch
        model = MixtureDesityNetwork(input_size, hidden_size=128, k=MDN_COMPONENTS, dropout_rate=0.1).to(DEVICE)
        lr = 1e-3
        print(f"  Training from scratch (lr={lr})")

    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        # Training
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

        # Validation
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

        # Early stopping
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
    # Expected value
    expected = (pi * mu).sum(dim=1)

    # Variance: E[σ²] + E[(μ - E[Y])²]
    variance = (pi * (sigma**2 + mu**2)).sum(dim=1) - expected**2
    std = torch.sqrt(torch.clamp(variance, min=1e-6))

    # Percentiles via sampling
    samples = []
    for _ in range(500):
        # Sample component for each sample
        component = torch.multinomial(pi, 1).squeeze(1)

        # Sample from selected component
        idx = torch.arange(len(component))
        sample_mu = mu[idx, component]
        sample_sigma = sigma[idx, component]
        sample = torch.normal(sample_mu, sample_sigma)
        samples.append(sample)

    samples = torch.stack(samples)

    p10 = torch.quantile(samples, 0.1, dim=0)
    p90 = torch.quantile(samples, 0.9, dim=0)

    # Probability of exceeding thresholds
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
# WALK-FORWARD BACKTEST WITH MULTI-SEASON PRE-TRAINING
# ============================================================================

def run_backtest(df_current, df_historical=None, opp_quality_dict=None):
    """
    Run walk-forward backtest with pre-training on historical data.

    Args:
        df_current: Current season (2024-25) data with features precomputed
        df_historical: Historical data (2020-2024) with features precomputed (optional)
        opp_quality_dict: Pre-computed opponent quality dictionary
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD BACKTEST WITH MULTI-SEASON PRE-TRAINING")
    print(f"Evaluation: {BACKTEST_START.date()} to {BACKTEST_END.date()}")
    print(f"Retraining every {RETRAIN_INTERVAL} days")
    print(f"{'='*80}\n")

    results = []
    current_date = BACKTEST_START
    last_retrain_date = TRAIN_START
    model = None
    norm_stats = None
    all_positions = None

    # Pre-train on historical data if available
    if df_historical is not None and len(df_historical) > 1000:
        print(">>> PRE-TRAINING on historical data (2020-2024)...")
        X_hist_train, y_hist_train, hist_norm_stats = prepare_training_data(
            df_historical,
            df_historical['game_date'].max(),
            opp_quality_dict
        )

        if X_hist_train is not None and len(X_hist_train) > 100:
            # Unpack hist_norm_stats
            X_hist_mean, X_hist_std, hist_positions = hist_norm_stats
            all_positions = hist_positions  # Use historical positions as base

            # Train-val split
            n_hist = int(0.8 * len(X_hist_train))
            idx_hist = torch.randperm(len(X_hist_train))
            train_idx, val_idx = idx_hist[:n_hist], idx_hist[n_hist:]

            X_hist_split = X_hist_train[train_idx]
            y_hist_split = y_hist_train[train_idx]
            X_hist_val = X_hist_train[val_idx]
            y_hist_val = y_hist_train[val_idx]

            print(f"  Pre-training samples: {len(X_hist_split)}, Validation: {len(X_hist_val)}")
            model = train_model(X_hist_split, y_hist_split, X_hist_val, y_hist_val, pretrained_model=None)
            print(">>> Pre-training complete. Model ready for fine-tuning.\n")
        else:
            print("  Insufficient historical data for pre-training.\n")
            model = None

    # Walk-forward backtest
    while current_date <= BACKTEST_END:
        # Retrain if needed
        if model is None or (current_date - last_retrain_date).days >= RETRAIN_INTERVAL:
            train_end = current_date - timedelta(days=1)
            print(f"\n>>> Retraining on data through {train_end.date()}")

            # Prepare training data from current season only
            train_result = prepare_training_data(
                df_current,
                train_end,
                opp_quality_dict
            )
            X_train, y_train, norm_stats = train_result
            X_mean, X_std, current_positions = norm_stats

            # Merge positions from historical and current
            if all_positions is not None:
                all_positions = sorted(set(all_positions) | set(current_positions))
            else:
                all_positions = current_positions

            if X_train is None or len(X_train) < 100:
                print(f"  Insufficient training data ({len(X_train) if X_train is not None else 0} samples)")
                current_date += timedelta(days=1)
                continue

            # Train-validation split (80-20)
            n_train = int(0.8 * len(X_train))
            idx = torch.randperm(len(X_train))
            train_idx, val_idx = idx[:n_train], idx[n_train:]

            X_train_split = X_train[train_idx]
            y_train_split = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]

            print(f"  Training samples: {len(X_train_split)}, Validation samples: {len(X_val)}")

            # Fine-tune from pre-trained model or train from scratch
            if model is not None:
                model = train_model(X_train_split, y_train_split, X_val, y_val,
                                  pretrained_model=model, fine_tune_lr=1e-4)
            else:
                model = train_model(X_train_split, y_train_split, X_val, y_val,
                                  pretrained_model=None)

            last_retrain_date = current_date

        # Make predictions for current_date
        X_pred, y_actual, player_ids, player_names, positions = \
            build_feature_matrix(df_current, current_date, opp_quality_dict, all_positions=all_positions)

        if X_pred is None or len(X_pred) == 0:
            current_date += timedelta(days=1)
            continue

        # Normalize X_pred using training statistics
        X_pred_norm = (X_pred - X_mean) / (X_std + 1e-6)

        # Get predictions
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


def print_results_table(mdn_v2_results, mdn_v1_mae=4.107):
    """Print comparison of MDN v1 vs v2."""
    if mdn_v2_results is None or len(mdn_v2_results) == 0:
        print("No results to display")
        return

    print(f"\n{'='*100}")
    print("MODEL COMPARISON: MDN v1 vs MDN v2")
    print(f"{'='*100}\n")

    # Overall metrics
    print("OVERALL METRICS")
    print("-" * 100)

    mae_v2, rmse_v2, corr_v2 = compute_metrics(
        mdn_v2_results['actual_fpts'].values,
        mdn_v2_results['predicted_fpts'].values
    )

    print(f"{'Model':<30s} | {'MAE':>8} | {'RMSE':>8} | {'Corr':>8} | {'vs v1':>8}")
    print("-" * 100)
    print(f"{'MDN v1 (baseline)':30s} | {mdn_v1_mae:>8.3f} | {'N/A':>8} | {'N/A':>8} | {'--':>8}")
    improvement = ((mdn_v1_mae - mae_v2) / mdn_v1_mae * 100) if not np.isnan(mae_v2) else 0
    print(f"{'MDN v2 (w/ pre-training)':30s} | {mae_v2:>8.3f} | {rmse_v2:>8.3f} | {corr_v2:>8.3f} | {improvement:+8.1f}%")

    # By position
    print(f"\n{'BY POSITION':100s}")
    print("-" * 100)
    print(f"{'Position':<12} {'N':>6} {'MAE v2':>10} {'vs v1':>10} {'Actual Std':>12} {'Difficulty':>12}")
    print("-" * 100)

    for pos in sorted(mdn_v2_results['position'].unique()):
        pos_data = mdn_v2_results[mdn_v2_results['position'] == pos]
        n = len(pos_data)

        mae_v2_pos = np.abs(pos_data['actual_fpts'] - pos_data['predicted_fpts']).mean()
        actual_std = pos_data['actual_fpts'].std()

        if actual_std < 4:
            difficulty = "EASY"
        elif actual_std < 6:
            difficulty = "MEDIUM"
        else:
            difficulty = "HARD"

        print(f"{pos:<12} {n:>6} {mae_v2_pos:>10.2f} {'N/A':>10} {actual_std:>12.2f} {difficulty:>12}")

    # Summary statistics
    print(f"\n{'KEY FINDINGS':100s}")
    print("-" * 100)

    print(f"MDN v2 MAE: {mae_v2:.3f}")
    print(f"  vs MDN v1 (4.107): {improvement:+.1f}%")
    print(f"\nEnhancements:")
    print(f"  + Multi-season pre-training on 128K+ historical records")
    print(f"  + Opponent defensive quality (FPTS allowed)")
    print(f"  + Regression-weighted features (shrinkage toward league average)")
    print(f"  + Larger hidden layers (128 units) with dropout regularization")
    print(f"  + Fine-tuning from pre-trained weights")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MDN v2: Enhanced NHL DFS Projection Model')
    parser.add_argument('--backtest', action='store_true', help='Run walk-forward backtest')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("MDN v2: ENHANCED MIXTURE DENSITY NETWORK")
    print("="*80)

    # Load and preprocess historical data
    print("\nLoading historical data (2020-2024)...")
    df_historical = load_historical_data()
    print(f"  Loaded {len(df_historical)} historical records from {df_historical['game_date'].min().date()} to {df_historical['game_date'].max().date()}")

    print("Precomputing rolling features on historical data...")
    df_historical = build_rolling_features(df_historical)
    print("Applying regression weights to historical data...")
    df_historical = apply_regression_weights(df_historical)

    # Load and preprocess current season data
    print("\nLoading current season data (2024-25)...")
    df_current = load_boxscore_data()
    print(f"  Loaded {len(df_current)} boxscore records from {df_current['game_date'].min().date()} to {df_current['game_date'].max().date()}")

    print("Precomputing rolling features on current season data...")
    df_current = build_rolling_features(df_current)
    print("Applying regression weights to current season data...")
    df_current = apply_regression_weights(df_current)

    # Compute opponent quality from combined data
    print("\nComputing opponent defensive quality...")
    combined_df = pd.concat([df_historical, df_current], ignore_index=True)
    opp_quality_dict = compute_opponent_quality(combined_df, window=10)
    print(f"  Computed quality metrics for {len(set(t[0] for t in opp_quality_dict.keys()))} opponent teams")

    if args.backtest:
        print("\nStarting walk-forward backtest...")
        mdn_v2_results = run_backtest(df_current, df_historical, opp_quality_dict)

        print_results_table(mdn_v2_results, mdn_v1_mae=4.107)

        # Save results
        output_path = Path(__file__).parent / 'mdn_v2_backtest_results.csv'
        mdn_v2_results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("\nUsage: python3 mdn_v2.py --backtest")


if __name__ == '__main__':
    main()
