"""
LSTM-CNN Sequence Model for NHL DFS Projections

Combines temporal sequence learning with convolutional feature extraction:
- CNN branch: extracts local patterns from game sequences (kernel_size=3)
- LSTM branch: captures longer-term dependencies and trends
- Fusion: combines CNN + LSTM + static features for final prediction

Advantages over MDN v3 (which treats each game independently):
- Learns temporal patterns (streaks, trends, momentum shifts)
- Models regression to mean and performance cycles
- Captures player-specific seasonality

Architecture:
- Input: last 10 games (8 temporal features) + 6 static features
- CNN: Conv1d(8, 32, k=3) x2 → Global avg pool → 32-dim
- LSTM: LSTM(input_size=8, hidden=64) → final hidden state → 64-dim
- Fusion: Dense(102, 64) + ReLU → Dense(64, 1)
- Loss: MSE
- Training: walk-forward Nov 7 2025 - Feb 5 2026, retrain every 14 days
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import json
from collections import defaultdict

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

# Model config
SEQUENCE_LENGTH = 10
TEMPORAL_FEATURES = 8  # goals, assists, shots, blocked_shots, hits, pp_goals, toi_seconds, dk_fpts
STATIC_FEATURES = 6    # position_C, position_D, position_L, position_R, opp_fpts_allowed_10g, home_road
BATCH_SIZE = 256
MAX_EPOCHS = 20
PATIENCE = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LSTMCNNModel(nn.Module):
    """
    Fusion model combining CNN and LSTM branches.

    CNN branch: Local pattern extraction from game sequences
    LSTM branch: Temporal dependency modeling
    Fusion: Combined dense layers for final prediction
    """
    def __init__(self, temporal_features=8, static_features=6):
        super(LSTMCNNModel, self).__init__()

        # CNN Branch: Extract local patterns from sequences
        self.conv1 = nn.Conv1d(temporal_features, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pool
        cnn_out = 32

        # LSTM Branch: Capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=temporal_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )
        lstm_out = 64

        # Fusion layers
        fusion_input = cnn_out + lstm_out + static_features  # 32 + 64 + 6 = 102
        self.fc1 = nn.Linear(fusion_input, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, temporal_seq, static_feats):
        """
        Args:
            temporal_seq: (batch, 10, 8) - sequences of temporal features
            static_feats: (batch, 6) - static features

        Returns:
            predictions: (batch, 1) - predicted dk_fpts
        """
        # CNN branch
        # temporal_seq: (batch, 10, 8) → need (batch, 8, 10) for Conv1d
        x_cnn = temporal_seq.transpose(1, 2)  # (batch, 8, 10)
        x_cnn = self.relu1(self.conv1(x_cnn))  # (batch, 32, 10)
        x_cnn = self.relu2(self.conv2(x_cnn))  # (batch, 32, 10)
        x_cnn = self.pool(x_cnn).squeeze(-1)  # (batch, 32)

        # LSTM branch
        _, (h_n, _) = self.lstm(temporal_seq)  # h_n: (1, batch, 64)
        x_lstm = h_n.squeeze(0)  # (batch, 64)

        # Fusion
        x_fused = torch.cat([x_cnn, x_lstm, static_feats], dim=1)  # (batch, 102)
        x_fused = self.relu3(self.fc1(x_fused))  # (batch, 64)
        x_fused = self.dropout(x_fused)
        predictions = self.fc2(x_fused)  # (batch, 1)

        return predictions


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_boxscore_data():
    """Load current season boxscore data."""
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

    return df


def load_historical_data():
    """Load 220K historical rows for pre-training."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("""
            SELECT
                season, game_date, player_name, player_id, team, position,
                goals, assists, dk_fpts, shots, blocked_shots, hits, pp_goals, toi_seconds,
                opponent, home_road
            FROM historical_skaters
            ORDER BY season, player_name, game_date
        """, conn)
        conn.close()

        if not df.empty:
            df['game_date'] = pd.to_datetime(df['game_date'])
            print(f"Loaded {len(df)} historical rows from seasons {df['season'].min()} to {df['season'].max()}")

        return df
    except Exception as e:
        print(f"Warning: Could not load historical data: {e}")
        conn.close()
        return None


def compute_opponent_fpts_allowed(df):
    """
    Compute trailing 10-game average FPTS allowed by each team.
    Signal: d=0.736, p<0.000001
    """
    # Group by game_date and team, sum FPTS
    daily_team_fpts = df.groupby(['game_date', 'team'])['dk_fpts'].sum().reset_index()
    daily_team_fpts.columns = ['game_date', 'team', 'total_fpts']

    # Create team-opponent mapping
    team_opponent = df[['game_date', 'team', 'opponent']].drop_duplicates()

    # Merge opponent FPTS
    team_opponent = team_opponent.merge(
        daily_team_fpts.rename(columns={'team': 'opponent', 'total_fpts': 'opp_fpts'}),
        on=['game_date', 'opponent'],
        how='left'
    )

    team_opponent['opp_fpts'] = team_opponent['opp_fpts'].fillna(0)
    team_opponent = team_opponent.sort_values(['team', 'game_date'])

    # Rolling 10-game average
    team_opponent['opp_fpts_allowed_10g'] = (
        team_opponent.groupby('team')['opp_fpts']
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Merge back
    df = df.merge(
        team_opponent[['game_date', 'team', 'opp_fpts_allowed_10g']],
        on=['game_date', 'team'],
        how='left'
    )
    df['opp_fpts_allowed_10g'] = df['opp_fpts_allowed_10g'].fillna(0)

    return df


def normalize_features(df, feature_cols, mean_dict=None, std_dict=None):
    """
    Normalize features using z-score.
    If mean/std not provided, compute from data.
    """
    if mean_dict is None:
        mean_dict = {}
        std_dict = {}
        for col in feature_cols:
            if col in df.columns:
                mean_dict[col] = df[col].mean()
                std_dict[col] = df[col].std() + 1e-6

    df_norm = df.copy()
    for col in feature_cols:
        if col in df_norm.columns:
            df_norm[col] = (df_norm[col] - mean_dict[col]) / std_dict[col]

    return df_norm, mean_dict, std_dict


def create_sequences(df_player, sequence_length=10):
    """
    Create sequences for a single player.

    Returns list of dicts:
    {
        'temporal': array (10, 8),
        'static': array (6,),
        'target': float,
        'game_date': datetime,
        'player_name': str,
        'position': str
    }
    """
    temporal_cols = ['goals', 'assists', 'shots', 'blocked_shots', 'hits', 'pp_goals', 'toi_seconds', 'dk_fpts']

    sequences = []

    for i in range(sequence_length, len(df_player)):
        # Get sequence
        seq_data = df_player.iloc[i-sequence_length:i][temporal_cols].values.astype(np.float32)

        # Get static features for the prediction game
        pred_row = df_player.iloc[i]
        pos = pred_row['position']

        # One-hot encode position
        pos_onehot = np.zeros(4, dtype=np.float32)  # C, D, L, R
        pos_map = {'C': 0, 'D': 1, 'L': 2, 'R': 3}
        if pos in pos_map:
            pos_onehot[pos_map[pos]] = 1.0

        # Home/road encoding
        hr_val = pred_row.get('home_road', '')
        if pd.isna(hr_val):
            hr_val = ''
        home_road = 1.0 if str(hr_val).upper() == 'H' else 0.0

        # Opponent FPTS allowed
        opp_fpts = pred_row.get('opp_fpts_allowed_10g', 0.0)

        # Combine static features
        static = np.concatenate([pos_onehot, [opp_fpts, home_road]]).astype(np.float32)

        target = pred_row['dk_fpts']
        game_date = pred_row['game_date']
        player_name = pred_row['player_name']

        sequences.append({
            'temporal': seq_data,
            'static': static,
            'target': target,
            'game_date': game_date,
            'player_name': player_name,
            'position': pos
        })

    return sequences


def create_all_sequences(df, min_games=10, max_sequences_per_player=50):
    """
    Create sequences for all players with sufficient history.
    Limits sequences per player to avoid excessive memory usage.
    """
    all_sequences = []

    # Use player_name for grouping (player_id may be missing in historical data)
    grouped = df.groupby('player_name')

    for player_name, df_player in grouped:
        if len(df_player) < min_games:
            continue

        # Sort by game_date
        df_player = df_player.sort_values('game_date').reset_index(drop=True)

        player_sequences = create_sequences(df_player, sequence_length=SEQUENCE_LENGTH)

        # Limit sequences per player to avoid memory overload
        if len(player_sequences) > max_sequences_per_player:
            # Take last N sequences (most recent games)
            player_sequences = player_sequences[-max_sequences_per_player:]

        all_sequences.extend(player_sequences)

    return all_sequences


def sequences_to_tensors(sequences):
    """Convert sequences to PyTorch tensors."""
    temporals = np.array([s['temporal'] for s in sequences])
    statics = np.array([s['static'] for s in sequences])
    targets = np.array([s['target'] for s in sequences])

    temporal_tensor = torch.from_numpy(temporals).float()
    static_tensor = torch.from_numpy(statics).float()
    target_tensor = torch.from_numpy(targets).float().unsqueeze(-1)

    return temporal_tensor, static_tensor, target_tensor


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for temporal_batch, static_batch, target_batch in train_loader:
        temporal_batch = temporal_batch.to(device)
        static_batch = static_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        predictions = model(temporal_batch, static_batch)
        loss = loss_fn(predictions, target_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * temporal_batch.size(0)

    return total_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for temporal_batch, static_batch, target_batch in val_loader:
            temporal_batch = temporal_batch.to(device)
            static_batch = static_batch.to(device)
            target_batch = target_batch.to(device)

            predictions = model(temporal_batch, static_batch)
            loss = loss_fn(predictions, target_batch)

            total_loss += loss.item() * temporal_batch.size(0)

    return total_loss / len(val_loader.dataset)


def train_model(model, train_loader, val_loader, device, max_epochs=50, patience=5, learning_rate=0.001):
    """Train model with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return model


# ============================================================================
# PREDICTIONS & EVALUATION
# ============================================================================

def predict_batch(model, temporal_batch, static_batch, device):
    """Make predictions on a batch."""
    model.eval()
    with torch.no_grad():
        temporal_batch = temporal_batch.to(device)
        static_batch = static_batch.to(device)
        predictions = model(temporal_batch, static_batch)

    return predictions.cpu().numpy().squeeze()


def compute_mae(actuals, predictions):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(actuals - predictions))


def compute_rmse(actuals, predictions):
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((actuals - predictions) ** 2))


# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================

def get_retrain_dates(start_date, end_date, interval_days=14):
    """Get list of retraining dates."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=interval_days)
    return dates


def run_walk_forward_backtest(df_current, df_historical=None):
    """
    Run walk-forward backtest from BACKTEST_START to BACKTEST_END.
    Retrain every RETRAIN_INTERVAL days.
    """
    print("\n" + "="*80)
    print("WALK-FORWARD BACKTEST: LSTM-CNN vs MDN v3 vs Kalman")
    print("="*80)

    # Prepare data
    df_all = df_current.copy()
    print("Computing opponent FPTS allowed...")
    df_all = compute_opponent_fpts_allowed(df_all)

    # Normalize temporal features (compute stats from training data)
    temporal_cols = ['goals', 'assists', 'shots', 'blocked_shots', 'hits', 'pp_goals', 'toi_seconds', 'dk_fpts']
    static_cols = ['opp_fpts_allowed_10g']

    # Get all training data before backtest start
    df_train_full = df_all[df_all['game_date'] < BACKTEST_START].copy()

    if df_train_full.empty:
        print("ERROR: No training data before backtest start date")
        return None

    print(f"Training data size: {len(df_train_full)} rows")

    # Compute normalization stats from training data
    norm_stats = {}
    for col in temporal_cols + static_cols:
        if col in df_train_full.columns:
            norm_stats[col] = {
                'mean': df_train_full[col].mean(),
                'std': df_train_full[col].std() + 1e-6
            }

    # Get backtest dates
    retrain_dates = get_retrain_dates(BACKTEST_START, BACKTEST_END, RETRAIN_INTERVAL)

    all_results = []

    for retrain_idx, retrain_date in enumerate(retrain_dates):
        print(f"\nRetrain cycle {retrain_idx+1}: Training date = {retrain_date.date()}")

        # Get training cutoff: use data before retrain date
        df_train = df_all[df_all['game_date'] < retrain_date].copy()

        # Include historical data for sequence learning (optional, can be slow)
        if df_historical is not None and retrain_idx == 0:
            # Only use historical on first cycle to save time
            print("  Merging historical data...")
            df_hist_all = compute_opponent_fpts_allowed(df_historical.copy())
            # Limit to recent seasons to save memory
            df_hist_all = df_hist_all[df_hist_all['season'] >= 2023]
            df_train = pd.concat([df_hist_all, df_train], ignore_index=True)
            print(f"  Total training data: {len(df_train)} rows")

        # Create sequences
        print("  Creating sequences...")
        sequences = create_all_sequences(df_train, min_games=10)

        if len(sequences) < 100:
            print(f"  Warning: Only {len(sequences)} sequences; skipping this cycle")
            continue

        # Split into train/val
        split_idx = int(0.8 * len(sequences))
        train_seqs = sequences[:split_idx]
        val_seqs = sequences[split_idx:]

        # Normalize sequences
        for seq in train_seqs + val_seqs:
            for i, col in enumerate(temporal_cols):
                if col in norm_stats:
                    seq['temporal'][:, i] = (seq['temporal'][:, i] - norm_stats[col]['mean']) / norm_stats[col]['std']

            # Normalize static features
            seq['static'][-2] = (seq['static'][-2] - norm_stats['opp_fpts_allowed_10g']['mean']) / norm_stats['opp_fpts_allowed_10g']['std']

        # Convert to tensors
        train_temporal, train_static, train_target = sequences_to_tensors(train_seqs)
        val_temporal, val_static, val_target = sequences_to_tensors(val_seqs)

        # Create dataloaders
        train_dataset = TensorDataset(train_temporal, train_static, train_target)
        val_dataset = TensorDataset(val_temporal, val_static, val_target)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Train model
        print("  Training model...")
        model = LSTMCNNModel(temporal_features=TEMPORAL_FEATURES, static_features=STATIC_FEATURES)
        model.to(DEVICE)
        model = train_model(model, train_loader, val_loader, DEVICE, max_epochs=MAX_EPOCHS, patience=PATIENCE)

        # Get test period: next RETRAIN_INTERVAL days
        if retrain_idx < len(retrain_dates) - 1:
            next_retrain = retrain_dates[retrain_idx + 1]
        else:
            next_retrain = BACKTEST_END + timedelta(days=1)

        df_test_period = df_all[(df_all['game_date'] >= retrain_date) & (df_all['game_date'] < next_retrain)].copy()

        if df_test_period.empty:
            print(f"  No test data for this period")
            continue

        # CRITICAL FIX: For test sequences, we need the FULL history up to each test game
        # so the model can use the last 10 games as context.
        # Build sequences from ALL data up through the test period,
        # then filter to only keep predictions for test-period games.
        df_through_test = df_all[df_all['game_date'] < next_retrain].copy()
        all_test_sequences = create_all_sequences(df_through_test, min_games=10)

        # Filter to only sequences whose target game is in the test period
        test_dates = set(df_test_period['game_date'].dt.strftime('%Y-%m-%d').unique())
        test_sequences = [s for s in all_test_sequences
                         if str(s['game_date'])[:10] in test_dates
                         or (hasattr(s['game_date'], 'strftime') and s['game_date'].strftime('%Y-%m-%d') in test_dates)]

        print(f"  Test sequences: {len(test_sequences)} (from {len(df_test_period)} test games)")

        # Normalize test sequences
        for seq in test_sequences:
            for i, col in enumerate(temporal_cols):
                if col in norm_stats:
                    seq['temporal'][:, i] = (seq['temporal'][:, i] - norm_stats[col]['mean']) / norm_stats[col]['std']
            seq['static'][-2] = (seq['static'][-2] - norm_stats['opp_fpts_allowed_10g']['mean']) / norm_stats['opp_fpts_allowed_10g']['std']

        if len(test_sequences) == 0:
            print(f"  No test sequences after filtering")
            continue

        # Make predictions
        test_temporal, test_static, test_target = sequences_to_tensors(test_sequences)
        test_dataset = TensorDataset(test_temporal, test_static, test_target)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        predictions = []
        with torch.no_grad():
            for temporal_batch, static_batch, _ in test_loader:
                preds = predict_batch(model, temporal_batch, static_batch, DEVICE)
                if isinstance(preds, np.ndarray):
                    if preds.ndim == 0:
                        predictions.append(preds.item())
                    else:
                        predictions.extend(preds.tolist())
                else:
                    predictions.append(float(preds))

        predictions = np.array(predictions).flatten()
        actuals = test_target.numpy().flatten()

        # Verify shapes match
        if len(predictions) != len(actuals):
            print(f"  Warning: prediction count mismatch. {len(predictions)} vs {len(actuals)}")
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]
            test_sequences = test_sequences[:min_len]

        # Compute metrics
        mae = compute_mae(actuals, predictions)
        rmse = compute_rmse(actuals, predictions)

        print(f"  Test Period MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # Store results
        for i, seq in enumerate(test_sequences):
            if i < len(actuals):
                all_results.append({
                    'game_date': seq['game_date'],
                    'player_name': seq['player_name'],
                    'position': seq['position'],
                    'actual_fpts': float(actuals[i]),
                    'predicted_fpts': float(predictions[i]),
                    'error': float(abs(actuals[i] - predictions[i]))
                })

    return all_results


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

def analyze_results(results):
    """Analyze backtest results and compare to baselines."""
    if not results:
        print("No results to analyze")
        return

    results_df = pd.DataFrame(results)

    # Overall metrics
    mae = compute_mae(results_df['actual_fpts'].values, results_df['predicted_fpts'].values)
    rmse = compute_rmse(results_df['actual_fpts'].values, results_df['predicted_fpts'].values)

    print("\n" + "="*80)
    print("LSTM-CNN MODEL RESULTS")
    print("="*80)
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Predictions: {len(results_df)}")

    # Position breakdown
    print("\nBy Position:")
    for pos in ['C', 'D', 'L', 'R']:
        pos_df = results_df[results_df['position'] == pos]
        if len(pos_df) > 0:
            pos_mae = compute_mae(pos_df['actual_fpts'].values, pos_df['predicted_fpts'].values)
            print(f"  {pos}: MAE={pos_mae:.4f} (n={len(pos_df)})")

    # Monthly breakdown
    print("\nBy Month:")
    results_df['month'] = pd.to_datetime(results_df['game_date']).dt.to_period('M')
    for month, month_df in results_df.groupby('month'):
        month_mae = compute_mae(month_df['actual_fpts'].values, month_df['predicted_fpts'].values)
        print(f"  {month}: MAE={month_mae:.4f} (n={len(month_df)})")

    # Comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    comparison = {
        'Model': ['LSTM-CNN', 'MDN v3', 'Kalman'],
        'MAE': [f"{mae:.4f}", "4.091", "4.318"],
        'Data Type': ['Sequential', 'Independent Games', 'Kalman Filter'],
        'Learning': ['Temporal Patterns', 'Per-Game Distribution', 'Trend Filtering']
    }

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    # Save detailed results
    output_path = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/lstm_cnn_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTM-CNN model for NHL DFS projections')
    parser.add_argument('--backtest', action='store_true', help='Run walk-forward backtest')
    parser.add_argument('--quick-test', action='store_true', help='Quick test on small data')

    args = parser.parse_args()

    if args.quick_test:
        print("Running quick test...")
        df_current = load_boxscore_data()
        print(f"Loaded {len(df_current)} current season rows")

        # Quick test: first 100 rows
        df_test = df_current.head(100).copy()
        df_test = compute_opponent_fpts_allowed(df_test)

        sequences = create_all_sequences(df_test, min_games=5)
        print(f"Created {len(sequences)} sequences from {len(df_test)} rows")

        if len(sequences) > 0:
            print("Sample sequence temporal shape:", sequences[0]['temporal'].shape)
            print("Sample sequence static shape:", sequences[0]['static'].shape)
            print("Sample target:", sequences[0]['target'])

        print("Quick test passed!")
        return

    if args.backtest:
        print("Loading data...")
        df_current = load_boxscore_data()
        df_historical = load_historical_data()

        print(f"Loaded {len(df_current)} current season rows")
        if df_historical is not None:
            print(f"Loaded {len(df_historical)} historical rows")

        results = run_walk_forward_backtest(df_current, df_historical)

        if results:
            analyze_results(results)
        else:
            print("Backtest failed or produced no results")
    else:
        print("Usage: python3 lstm_cnn.py --backtest")
        print("       python3 lstm_cnn.py --quick-test")


if __name__ == '__main__':
    main()
