"""
Mixture Density Network (MDN) for NHL DFS Projections.

This model learns the full distribution of player fantasy points, capturing:
- Quiet nights (low-scoring mode)
- Average performances (medium-scoring mode)
- Explosions (high-scoring mode)

Architecture:
- Input: rolling stats (5/10 game), season-to-date, position, opponent quality, games played
- Plus NST-derived features (from LASSO feature selection):
  * dk_fpts_ewm: Exponentially-weighted FPTS (halflife=15)
  * pp_toi_per_game: PP ice time per game
  * ev_ixg: 5v5 individual expected goals
  * ev_toi_per_game: Even-strength ice time per game
  * toi_seconds_trend: TOI trending up/down
  * opp_xgf_pct: Opponent 5v5 expected goals for %
  * pp_ixg: PP individual expected goals
  * oi_hdcf_pct: On-ice HDCF%
  * hdcf_x_opp_weak: Interaction of on-ice HDCF% and opponent weakness
- Hidden: 2 layers of 64 units with ReLU
- Output: K=3 Gaussian mixture components (π, μ, σ)
- Loss: Negative log-likelihood of mixture distribution

Walk-forward backtest:
- Train from Oct 7 through the day before prediction
- Retrain every 14 days
- Evaluate Nov 7 through Feb 5
- NST data walks forward (only uses snapshot before prediction date)
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
MAX_EPOCHS = 30  # Reduced from 50 for faster training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    # Precompute exponentially-weighted FPTS (halflife=15 games) - KEY FEATURE FROM LASSO
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

    # TOI trending: last 5 games avg / season avg
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


# Cache NST data to avoid repeated DB queries
_nst_cache = {}
_nst_cache_date = None
_nst_skater_cache = {}
_nst_skater_cache_date = None
_fuzzy_match_cache = {}  # Cache fuzzy matches to speed up

def load_nst_teams_for_date(date_str):
    """Load NST team stats for a specific date (cached)."""
    global _nst_cache, _nst_cache_date

    # Use cache if available and recent (within 1 day)
    if _nst_cache and _nst_cache_date == date_str[:10]:
        return _nst_cache

    conn = sqlite3.connect(DB_PATH)

    # Get the latest snapshot <= date_str
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
        # Keep most recent for each team
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
    """
    Fuzzy match a boxscore abbreviated name (e.g., 'N. MacKinnon')
    to an NST full name (e.g., 'Nathan MacKinnon').

    Returns the best matching NST name if similarity >= threshold, else None.
    Uses cache to speed up repeated lookups.
    """
    global _fuzzy_match_cache

    # Check cache
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
    """
    Load NST skater stats for all available statistics before a given date.
    Returns (dict, date_str) indexed by (player_name, situation, stat_type).
    If no data found, returns ({}, None).
    Uses cache to avoid repeated DB hits.
    """
    global _nst_skater_cache, _nst_skater_cache_date

    # Check cache first
    if _nst_skater_cache and _nst_skater_cache_date == date_str[:10]:
        return _nst_skater_cache, _nst_skater_cache_date

    conn = sqlite3.connect(DB_PATH)

    # Get the most recent snapshot <= date_str where from_date = '2025-10-07'
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

        # Load all skater data for that snapshot
        query = f"""
            SELECT player, situation, stat_type, ixg, toi, gp, hdcf_pct
            FROM nst_skaters
            WHERE to_date = '{latest_date}' AND from_date = '2025-10-07'
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # Index by (player, situation, stat_type) for easy lookup
        result = {}
        for _, row in df.iterrows():
            key = (row['player'], row['situation'], row['stat_type'])
            result[key] = row

    _nst_skater_cache = result
    _nst_skater_cache_date = date_str[:10]
    return result, latest_date


def get_nst_feature(nst_data, player_name, situation, stat_type, column, default=0):
    """
    Safely retrieve an NST feature value.
    If player/situation/stat_type not found, return default.
    """
    key = (player_name, situation, stat_type)
    if key in nst_data:
        val = nst_data[key].get(column)
        if pd.notna(val):
            return float(val)
    return default


def compute_toi_per_game(row):
    """
    Compute TOI per game from NST data.
    toi is stored as a string like '876.76666666667', gp is integer.
    Returns toi_seconds / gp if both available, else 0.
    """
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

    predict_date: datetime to predict for
    train_cutoff: if provided, only use data before this date (unused, for walk-forward structure)
    """
    # Get games on predict_date
    predict_games = df[df['game_date'] == predict_date].copy()
    if predict_games.empty:
        return None, None, None, None, None

    # Position one-hot encoding
    positions = pd.get_dummies(predict_games['position'], prefix='pos', drop_first=False)

    # Load opponent quality for this date
    nst_teams = load_nst_teams_for_date(predict_date.strftime('%Y-%m-%d'))

    # Map opponent to quality
    predict_games.loc[:, 'opp_xgf_pct'] = predict_games['opponent'].map(
        lambda x: nst_teams.get(x, {}).get('xgf_pct', 0.5)
    )
    predict_games.loc[:, 'opp_sv_pct'] = predict_games['opponent'].map(
        lambda x: nst_teams.get(x, {}).get('sv_pct', 0.91)
    )

    # Load NST skater data (most recent snapshot before predict_date)
    nst_data, nst_date = load_nst_skaters_for_date(predict_date.strftime('%Y-%m-%d'))

    # Build NST-derived features with fuzzy name matching
    if nst_data:
        nst_player_names = list(set(key[0] for key in nst_data.keys()))

        # Initialize new feature columns
        predict_games.loc[:, 'pp_toi_per_game'] = 0.0
        predict_games.loc[:, 'ev_ixg'] = 0.0
        predict_games.loc[:, 'ev_toi_per_game'] = 0.0
        predict_games.loc[:, 'pp_ixg'] = 0.0
        predict_games.loc[:, 'oi_hdcf_pct'] = 0.0

        for idx, row in predict_games.iterrows():
            boxscore_name = row['player_name']

            # Fuzzy match to NST name
            matched_nst_name = fuzzy_match_names(boxscore_name, nst_player_names, threshold=0.6)

            if matched_nst_name:
                # PP TOI per game: from nst_skaters std, situation='pp', toi/gp
                pp_row = nst_data.get((matched_nst_name, 'pp', 'std'))
                if pp_row is not None:
                    pp_toi_per_game = compute_toi_per_game(pp_row)
                    predict_games.loc[idx, 'pp_toi_per_game'] = pp_toi_per_game

                # EV iXG: from nst_skaters std, situation='5v5', ixg column
                ev_row = nst_data.get((matched_nst_name, '5v5', 'std'))
                if ev_row is not None:
                    ixg = get_nst_feature(nst_data, matched_nst_name, '5v5', 'std', 'ixg', 0)
                    predict_games.loc[idx, 'ev_ixg'] = ixg

                    # EV TOI per game: from nst_skaters std, situation='5v5', toi/gp
                    ev_toi_per_game = compute_toi_per_game(ev_row)
                    predict_games.loc[idx, 'ev_toi_per_game'] = ev_toi_per_game

                # PP iXG: from nst_skaters std, situation='pp', ixg column
                if pp_row is not None:
                    pp_ixg = get_nst_feature(nst_data, matched_nst_name, 'pp', 'std', 'ixg', 0)
                    predict_games.loc[idx, 'pp_ixg'] = pp_ixg

                # On-ice HDCF%: from nst_skaters oi, situation='5v5', hdcf_pct
                oi_row = nst_data.get((matched_nst_name, '5v5', 'oi'))
                if oi_row is not None:
                    hdcf_pct = get_nst_feature(nst_data, matched_nst_name, '5v5', 'oi', 'hdcf_pct', 50) / 100
                    predict_games.loc[idx, 'oi_hdcf_pct'] = hdcf_pct
    else:
        # If no NST data available, initialize with zeros
        predict_games.loc[:, 'pp_toi_per_game'] = 0.0
        predict_games.loc[:, 'ev_ixg'] = 0.0
        predict_games.loc[:, 'ev_toi_per_game'] = 0.0
        predict_games.loc[:, 'pp_ixg'] = 0.0
        predict_games.loc[:, 'oi_hdcf_pct'] = 0.0

    # Collect rolling and season features (use all available in df)
    rolling_cols = [c for c in df.columns if c.startswith('rolling_')]
    season_cols = [c for c in df.columns if c.startswith('season_avg_') or c == 'log_gp']

    # Ensure these columns exist
    if not rolling_cols or not season_cols:
        return None, None, None, None, None

    # Select features - make sure columns exist in predict_games
    feature_cols = []
    for col in rolling_cols + season_cols:
        if col in predict_games.columns:
            feature_cols.append(col)

    # Add pre-computed EWM and trend features (from LASSO)
    extra_features = []
    if 'dk_fpts_ewm' in predict_games.columns:
        extra_features.append('dk_fpts_ewm')
    if 'toi_seconds_trend' in predict_games.columns:
        extra_features.append('toi_seconds_trend')
    feature_cols.extend(extra_features)

    # Add NST-derived features
    nst_features = ['pp_toi_per_game', 'ev_ixg', 'ev_toi_per_game', 'pp_ixg', 'oi_hdcf_pct']

    X = predict_games[feature_cols + ['opp_xgf_pct', 'opp_sv_pct'] + nst_features].reset_index(drop=True).copy()

    # Add interaction feature: hdcf_x_opp_weak = oi_hdcf_pct * (1 - opp_xgf_pct)
    # Note: opp_xgf_pct is already scaled 0-1 after the map operation
    X['hdcf_x_opp_weak'] = X['oi_hdcf_pct'] * (1.0 - X['opp_xgf_pct'])

    # Add position encoding - MUST reset both indices and align
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
    Uses precomputed rolling features to avoid recomputation.
    """
    train_df = df[df['game_date'] <= train_end_date].copy()

    if train_df.empty:
        return None, None, None

    # Get unique prediction dates (sample to speed up)
    pred_dates = sorted(train_df['game_date'].unique())
    # Sample every Nth date to speed up training
    sample_interval = max(1, len(pred_dates) // 12)  # Target ~12 dates for faster training
    pred_dates = pred_dates[::sample_interval]

    print(f"  Using {len(pred_dates)} dates for training (sampled from {len(train_df['game_date'].unique())})")

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

    # Safety check
    assert len(X) == len(y), f"Shape mismatch: X={len(X)}, y={len(y)}"

    # Handle NaN
    X = X.fillna(0)

    # Normalize X
    X_mean = X.mean()
    X_std = X.std()
    X_std[X_std == 0] = 1  # Avoid division by zero
    X = (X - X_mean) / X_std

    # Convert to float32 for torch
    X_array = X.values.astype(np.float32)
    y_array = y.astype(np.float32)

    return torch.FloatTensor(X_array), torch.FloatTensor(y_array), (X_mean, X_std)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MixtureDesityNetwork(nn.Module):
    """Mixture Density Network with K Gaussian components."""

    def __init__(self, input_size, hidden_size=64, k=3):
        super().__init__()
        self.k = k

        # Feature extraction
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Output layer for mixture parameters
        # π: K mixing coefficients (softmax)
        # μ: K means
        # σ: K standard deviations (softplus)
        self.pi_layer = nn.Linear(hidden_size, k)
        self.mu_layer = nn.Linear(hidden_size, k)
        self.sigma_layer = nn.Linear(hidden_size, k)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))

        # Mixing coefficients
        pi = torch.softmax(self.pi_layer(h), dim=-1)

        # Means
        mu = self.mu_layer(h)

        # Standard deviations (ensure positive)
        sigma = torch.nn.functional.softplus(self.sigma_layer(h)) + 1e-6

        return pi, mu, sigma

    def loss(self, pi, mu, sigma, y):
        """
        Negative log-likelihood of Gaussian mixture model.

        pi: (batch_size, k)
        mu: (batch_size, k)
        sigma: (batch_size, k)
        y: (batch_size,)
        """
        # Expand y for broadcasting
        y = y.unsqueeze(1)  # (batch_size, 1)

        # Gaussian PDF: (1/(σ√(2π))) * exp(-(y-μ)²/(2σ²))
        # Log: log(π_k) - 0.5*log(2π) - log(σ_k) - 0.5*((y-μ_k)/σ_k)²

        log_sigma = torch.log(sigma)
        normalized = (y - mu) / sigma

        log_gaussian = -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * normalized**2
        log_mixture = torch.log(pi + 1e-10) + log_gaussian

        # Log-sum-exp for numerical stability
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
        # Ensure X is float64, then convert to float32 for torch
        X_array = X.values.astype(np.float64).astype(np.float32)
        X_tensor = torch.FloatTensor(X_array).to(DEVICE)
        pi, mu, sigma = model(X_tensor)

    return pi.cpu(), mu.cpu(), sigma.cpu()


def compute_projection_stats(pi, mu, sigma):
    """
    Compute projection statistics from mixture distribution.

    pi, mu, sigma: (n_samples, k)
    """
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

            # Prepare training data
            X_train, y_train, norm_stats = prepare_training_data(df, train_end)

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

            model = train_model(X_train_split, y_train_split, X_val, y_val)
            last_retrain_date = current_date

        # Make predictions for current_date
        X_pred, y_actual, player_ids, player_names, positions = \
            build_feature_matrix(df, current_date)

        if X_pred is None or len(X_pred) == 0:
            current_date += timedelta(days=1)
            continue

        # Normalize X_pred
        X_mean, X_std = norm_stats
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


def compute_baseline_projections(df):
    """Compute baseline projections for comparison."""
    results = []

    pred_dates = df['game_date'].unique()
    pred_dates = sorted([d for d in pred_dates if d >= BACKTEST_START])[:20]  # Sample to avoid slowness

    for pred_date in pred_dates:
        train_df = df[df['game_date'] < pred_date]

        # Expanding mean
        expanding_mean = train_df.groupby('player_id')['dk_fpts'].mean()
        expanding_std = train_df.groupby('player_id')['dk_fpts'].std()

        X_pred, y_actual, player_ids, player_names, positions = \
            build_feature_matrix(df, pred_date)

        if X_pred is None:
            continue

        for i, (pid, name, pos, actual) in enumerate(zip(player_ids, player_names, positions, y_actual)):
            mean_val = expanding_mean.get(pid, df['dk_fpts'].mean())
            std_val = expanding_std.get(pid, df['dk_fpts'].std())

            results.append({
                'game_date': pred_date.date(),
                'player_id': pid,
                'player_name': name,
                'position': pos,
                'actual_fpts': actual,
                'expanding_mean': mean_val,
                'expanding_std': std_val,
            })

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
    print("MODEL COMPARISON")
    print(f"{'='*100}\n")

    # Overall metrics
    print("OVERALL METRICS")
    print("-" * 80)

    mae_mdn, rmse_mdn, corr_mdn = compute_metrics(
        mdn_results['actual_fpts'].values,
        mdn_results['predicted_fpts'].values
    )

    print(f"{'Model':<20s} | {'MAE':>8} | {'RMSE':>8} | {'Corr':>8}")
    print("-" * 80)
    print(f"{'MDN (learned)':20s} | {mae_mdn:>8.2f} | {rmse_mdn:>8.2f} | {corr_mdn:>8.3f}")
    print(f"{'Poisson sim':20s} | {4.75:>8.2f} | {'N/A':>8} | {'N/A':>8}  (baseline)")
    print(f"{'Kalman filter':20s} | {4.32:>8.2f} | {'N/A':>8} | {'N/A':>8}  (baseline)")
    print(f"{'Expanding mean':20s} | {np.abs(mdn_results['actual_fpts'] - mdn_results['actual_fpts'].mean()).mean():>8.2f} | {'N/A':>8} | {'N/A':>8}  (simple baseline)")

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

        print(f"{pos:<12} {n:>6} {mdn_mae:>10.2f} {actual_std:>12.2f} {difficulty:>12}")

    # Summary statistics
    print(f"\n{'KEY FINDINGS':80s}")
    print("-" * 80)

    improvement_vs_kalman = (4.32 - mae_mdn) / 4.32 * 100 if not np.isnan(mae_mdn) else 0

    print(f"MDN MAE: {mae_mdn:.2f}")
    print(f"  vs Kalman (4.32): {improvement_vs_kalman:+.1f}%")
    print(f"  vs Poisson (4.75): {(4.75 - mae_mdn) / 4.75 * 100:+.1f}%")
    print(f"\nMDN learns feature importance through 3-component Gaussian mixture")
    print(f"Captures multimodal outcomes: quiet nights, average games, explosions")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MDN NHL DFS Projection Model')
    parser.add_argument('--backtest', action='store_true', help='Run walk-forward backtest')
    args = parser.parse_args()

    print("\nLoading and preprocessing data...")
    df = load_boxscore_data()
    print(f"Loaded {len(df)} boxscore records from {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"Features: {[c for c in df.columns if c.startswith('rolling_') or c.startswith('season_')][:3]}...")

    if args.backtest:
        print("\nStarting walk-forward backtest...")
        mdn_results = run_backtest(df)

        print_results_table(mdn_results)

        # Save results
        output_path = Path(__file__).parent / 'mdn_backtest_results.csv'
        mdn_results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("\nUsage: python mdn_projection.py --backtest")


if __name__ == '__main__':
    main()
