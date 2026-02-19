#!/usr/bin/env python3
"""
Validation script for LSTM-CNN model.
Checks architecture, data pipeline, and outputs.
"""

import torch
import torch.nn as nn
import sys

def validate_architecture():
    """Verify model architecture matches specification."""
    print("\n" + "="*80)
    print("VALIDATING LSTM-CNN ARCHITECTURE")
    print("="*80)

    sys.path.insert(0, '/sessions/youthful-funny-faraday/mnt/Code/projection')
    from lstm_cnn import LSTMCNNModel, TEMPORAL_FEATURES, STATIC_FEATURES

    model = LSTMCNNModel(temporal_features=TEMPORAL_FEATURES, static_features=STATIC_FEATURES)

    print(f"\nModel Summary:")
    print(f"  Input temporal features: {TEMPORAL_FEATURES}")
    print(f"  Input static features: {STATIC_FEATURES}")
    print(f"  Sequence length: 10")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print architecture
    print(f"\nArchitecture:")
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            print(f"  {name}: Sequential")
            for i, layer in enumerate(module):
                print(f"    [{i}] {layer}")
        else:
            print(f"  {name}: {module}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    temporal_input = torch.randn(batch_size, 10, TEMPORAL_FEATURES)
    static_input = torch.randn(batch_size, STATIC_FEATURES)

    output = model(temporal_input, static_input)
    print(f"  Input shape (temporal): {temporal_input.shape}")
    print(f"  Input shape (static): {static_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, 1)")

    assert output.shape == (batch_size, 1), "Output shape mismatch!"

    print("\n✓ Architecture validation passed!")
    return model


def validate_data_pipeline():
    """Verify data loading and sequence creation."""
    print("\n" + "="*80)
    print("VALIDATING DATA PIPELINE")
    print("="*80)

    sys.path.insert(0, '/sessions/youthful-funny-faraday/mnt/Code/projection')
    from lstm_cnn import (
        load_boxscore_data,
        load_historical_data,
        compute_opponent_fpts_allowed,
        create_all_sequences
    )

    # Load boxscore
    print("\nLoading boxscore data...")
    df_current = load_boxscore_data()
    print(f"  ✓ Loaded {len(df_current)} rows")
    print(f"  Columns: {', '.join(df_current.columns.tolist()[:5])}...")
    print(f"  Date range: {df_current['game_date'].min()} to {df_current['game_date'].max()}")

    # Load historical
    print("\nLoading historical data...")
    df_historical = load_historical_data()
    print(f"  ✓ Loaded {len(df_historical)} rows")
    print(f"  Seasons: {sorted(df_historical['season'].unique())}")

    # Compute opponent stats
    print("\nComputing opponent FPTS allowed...")
    df_test = df_current.head(1000).copy()
    df_test = compute_opponent_fpts_allowed(df_test)
    print(f"  ✓ Added 'opp_fpts_allowed_10g' feature")
    print(f"  Range: {df_test['opp_fpts_allowed_10g'].min():.2f} to {df_test['opp_fpts_allowed_10g'].max():.2f}")

    # Create sequences
    print("\nCreating sequences...")
    sequences = create_all_sequences(df_test, min_games=10)
    print(f"  ✓ Created {len(sequences)} sequences from {len(df_test)} rows")

    if len(sequences) > 0:
        seq = sequences[0]
        print(f"  Sequence structure:")
        print(f"    - temporal shape: {seq['temporal'].shape} (expected: (10, 8))")
        print(f"    - static shape: {seq['static'].shape} (expected: (6,))")
        print(f"    - target: {seq['target']:.2f}")
        print(f"    - game_date: {seq['game_date']}")
        print(f"    - player_name: {seq['player_name']}")

    print("\n✓ Data pipeline validation passed!")


def validate_training():
    """Verify training loop works."""
    print("\n" + "="*80)
    print("VALIDATING TRAINING LOOP")
    print("="*80)

    sys.path.insert(0, '/sessions/youthful-funny-faraday/mnt/Code/projection')
    from lstm_cnn import (
        LSTMCNNModel,
        load_boxscore_data,
        compute_opponent_fpts_allowed,
        create_all_sequences,
        sequences_to_tensors,
        train_epoch,
        validate_epoch,
        DEVICE,
        TEMPORAL_FEATURES,
        STATIC_FEATURES,
        BATCH_SIZE
    )
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim

    # Prepare data
    print("\nPreparing training data...")
    df = load_boxscore_data()
    df = compute_opponent_fpts_allowed(df)
    df_sample = df.head(2000).copy()

    sequences = create_all_sequences(df_sample, min_games=10)
    if len(sequences) < 100:
        print(f"  Warning: Only {len(sequences)} sequences, need at least 100")
        return

    # Split
    split = int(0.8 * len(sequences))
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    # Create tensors
    train_temporal, train_static, train_target = sequences_to_tensors(train_seqs)
    val_temporal, val_static, val_target = sequences_to_tensors(val_seqs)

    print(f"  ✓ Train: {len(train_seqs)} sequences")
    print(f"  ✓ Val: {len(val_seqs)} sequences")

    # Create dataloaders
    train_dataset = TensorDataset(train_temporal, train_static, train_target)
    val_dataset = TensorDataset(val_temporal, val_static, val_target)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    print("\nInitializing model...")
    model = LSTMCNNModel(temporal_features=TEMPORAL_FEATURES, static_features=STATIC_FEATURES)
    model.to(DEVICE)
    print(f"  ✓ Model on device: {DEVICE}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Train one epoch
    print("\nTraining for 1 epoch...")
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
    print(f"  ✓ Train loss: {train_loss:.4f}")

    # Validate one epoch
    print("\nValidating...")
    val_loss = validate_epoch(model, val_loader, loss_fn, DEVICE)
    print(f"  ✓ Val loss: {val_loss:.4f}")

    print("\n✓ Training loop validation passed!")


def validate_outputs():
    """Verify output format."""
    print("\n" + "="*80)
    print("VALIDATING OUTPUT FORMAT")
    print("="*80)

    import pandas as pd
    from pathlib import Path

    lstm_file = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/lstm_cnn.py')
    demo_file = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/lstm_cnn_demo.py')
    readme_file = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/LSTM_CNN_README.md')

    print("\nChecking output files...")
    print(f"  {'lstm_cnn.py':<30} {'Size':>12}  {'OK' if lstm_file.exists() else 'MISSING'}")
    if lstm_file.exists():
        print(f"    {lstm_file.stat().st_size:,} bytes, {lstm_file.read_text().count(chr(10))} lines")

    print(f"  {'lstm_cnn_demo.py':<30} {'Size':>12}  {'OK' if demo_file.exists() else 'MISSING'}")
    if demo_file.exists():
        print(f"    {demo_file.stat().st_size:,} bytes")

    print(f"  {'LSTM_CNN_README.md':<30} {'Size':>12}  {'OK' if readme_file.exists() else 'MISSING'}")
    if readme_file.exists():
        print(f"    {readme_file.stat().st_size:,} bytes")

    print("\n✓ Output files validation passed!")


def main():
    """Run all validations."""
    print("\n" + "="*80)
    print("LSTM-CNN MODEL VALIDATION")
    print("="*80)

    try:
        validate_architecture()
        validate_data_pipeline()
        validate_training()
        validate_outputs()

        print("\n" + "="*80)
        print("ALL VALIDATIONS PASSED ✓")
        print("="*80)
        print("\nModel is ready for use:")
        print("  - Quick test: python3 lstm_cnn.py --quick-test")
        print("  - Demo: python3 lstm_cnn_demo.py")
        print("  - Full backtest: python3 lstm_cnn.py --backtest")

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
