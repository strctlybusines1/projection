"""
Quick demo of LSTM-CNN model - runs just first retrain cycle
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import functions from main module
import sys
sys.path.insert(0, '/sessions/youthful-funny-faraday/mnt/Code/projection')

from lstm_cnn import (
    LSTMCNNModel,
    load_boxscore_data,
    load_historical_data,
    compute_opponent_fpts_allowed,
    create_all_sequences,
    sequences_to_tensors,
    train_model,
    compute_mae,
    compute_rmse,
    DEVICE,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    TEMPORAL_FEATURES,
    STATIC_FEATURES,
    BACKTEST_START,
    BACKTEST_END
)

def demo_single_cycle():
    print("\n" + "="*80)
    print("LSTM-CNN MODEL DEMO - Single Retrain Cycle")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df_current = load_boxscore_data()
    df_historical = load_historical_data()
    print(f"Current season: {len(df_current)} rows")
    print(f"Historical: {len(df_historical)} rows")

    # Prepare
    df_all = df_current.copy()
    print("\nComputing opponent FPTS allowed...")
    df_all = compute_opponent_fpts_allowed(df_all)

    # First retrain date
    retrain_date = BACKTEST_START
    print(f"\nRetrain cycle: {retrain_date.date()}")

    # Get training data
    df_train = df_all[df_all['game_date'] < retrain_date].copy()

    # Add historical from 2023+
    if df_historical is not None:
        print("Merging historical data...")
        df_hist_all = compute_opponent_fpts_allowed(df_historical.copy())
        df_hist_all = df_hist_all[df_hist_all['season'] >= 2023]
        df_train = pd.concat([df_hist_all, df_train], ignore_index=True)

    print(f"Training data: {len(df_train)} rows")

    # Compute normalization stats
    temporal_cols = ['goals', 'assists', 'shots', 'blocked_shots', 'hits', 'pp_goals', 'toi_seconds', 'dk_fpts']
    static_cols = ['opp_fpts_allowed_10g']

    norm_stats = {}
    for col in temporal_cols + static_cols:
        if col in df_train.columns:
            norm_stats[col] = {
                'mean': df_train[col].mean(),
                'std': df_train[col].std() + 1e-6
            }

    # Create sequences
    print("Creating sequences...")
    sequences = create_all_sequences(df_train, min_games=10)
    print(f"Created {len(sequences)} sequences")

    if len(sequences) == 0:
        print("No sequences created!")
        return

    # Split and normalize
    split_idx = int(0.8 * len(sequences))
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]

    for seq in train_seqs + val_seqs:
        for i, col in enumerate(temporal_cols):
            if col in norm_stats:
                seq['temporal'][:, i] = (seq['temporal'][:, i] - norm_stats[col]['mean']) / norm_stats[col]['std']
        seq['static'][-2] = (seq['static'][-2] - norm_stats['opp_fpts_allowed_10g']['mean']) / norm_stats['opp_fpts_allowed_10g']['std']

    # Create tensors
    train_temporal, train_static, train_target = sequences_to_tensors(train_seqs)
    val_temporal, val_static, val_target = sequences_to_tensors(val_seqs)

    print(f"Train set: {len(train_seqs)} sequences")
    print(f"Val set: {len(val_seqs)} sequences")

    # Create dataloaders
    train_dataset = TensorDataset(train_temporal, train_static, train_target)
    val_dataset = TensorDataset(val_temporal, val_static, val_target)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    print("\nTraining model...")
    model = LSTMCNNModel(temporal_features=TEMPORAL_FEATURES, static_features=STATIC_FEATURES)
    model.to(DEVICE)
    model = train_model(model, train_loader, val_loader, DEVICE, max_epochs=20, patience=2)

    # Test period
    next_retrain = retrain_date + timedelta(days=14)
    df_test = df_all[(df_all['game_date'] >= retrain_date) & (df_all['game_date'] < next_retrain)].copy()
    print(f"\nTest period: {retrain_date.date()} to {next_retrain.date()}")
    print(f"Test data: {len(df_test)} rows")

    # Create test sequences
    test_sequences = create_all_sequences(df_test, min_games=10)
    print(f"Test sequences: {len(test_sequences)}")

    if len(test_sequences) > 0:
        # Normalize test sequences
        for seq in test_sequences:
            for i, col in enumerate(temporal_cols):
                if col in norm_stats:
                    seq['temporal'][:, i] = (seq['temporal'][:, i] - norm_stats[col]['mean']) / norm_stats[col]['std']
            seq['static'][-2] = (seq['static'][-2] - norm_stats['opp_fpts_allowed_10g']['mean']) / norm_stats['opp_fpts_allowed_10g']['std']

        # Make predictions
        test_temporal, test_static, test_target = sequences_to_tensors(test_sequences)
        test_dataset = TensorDataset(test_temporal, test_static, test_target)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        predictions = []
        model.eval()
        with torch.no_grad():
            for temporal_batch, static_batch, _ in test_loader:
                temporal_batch = temporal_batch.to(DEVICE)
                static_batch = static_batch.to(DEVICE)
                preds = model(temporal_batch, static_batch).cpu().numpy()
                if preds.ndim == 0:
                    predictions.append(preds.item())
                else:
                    predictions.extend(preds.flatten().tolist())

        predictions = np.array(predictions).flatten()
        actuals = test_target.numpy().flatten()

        # Metrics
        mae = compute_mae(actuals, predictions)
        rmse = compute_rmse(actuals, predictions)

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # Comparison
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"{'Model':<20} {'MAE':<10} {'Learning Approach':<30}")
        print("-"*60)
        print(f"{'LSTM-CNN':<20} {mae:>8.4f} {'Temporal Sequence':<30}")
        print(f"{'MDN v3':<20} {'4.091':>8} {'Per-Game Distribution':<30}")
        print(f"{'Kalman':<20} {'4.318':>8} {'Trend Filtering':<30}")

        # Sample predictions
        print("\n" + "="*80)
        print("SAMPLE PREDICTIONS")
        print("="*80)
        for i in range(min(10, len(test_sequences))):
            seq = test_sequences[i]
            error = abs(actuals[i] - predictions[i])
            print(f"{seq['player_name']:<20} {seq['position']:<5} Actual: {actuals[i]:>6.2f}  Pred: {predictions[i]:>6.2f}  Error: {error:>5.2f}")

        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nTo run full walk-forward backtest:")
        print("  python3 lstm_cnn.py --backtest")
        print("\nModel file location:")
        print("  /sessions/youthful-funny-faraday/mnt/Code/projection/lstm_cnn.py")

if __name__ == '__main__':
    demo_single_cycle()
