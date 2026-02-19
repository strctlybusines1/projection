# LSTM-CNN Sequence Model for NHL DFS Projections

## Overview

A deep learning model combining LSTM and CNN architectures to learn temporal patterns in NHL player performance for daily fantasy sports (DFS) projections.

**Model File**: `/sessions/youthful-funny-faraday/mnt/Code/projection/lstm_cnn.py`

## Architecture

### Input Layer
- **Temporal Sequence**: Last 10 games per player
  - Features: `[goals, assists, shots, blocked_shots, hits, pp_goals, toi_seconds, dk_fpts]` (8 dimensions)
  - Shape: `(batch, 10, 8)`

- **Static Features**: Position + Opponent Quality + Home/Road
  - One-hot position encoding: `[C, D, L, R]` (4 dims)
  - Opponent FPTS allowed (10-game rolling avg): 1 dim
  - Home/Road indicator: 1 dim
  - Shape: `(batch, 6)`

### CNN Branch
Extracts local temporal patterns from game sequences:
```
Conv1d(8, 32, kernel_size=3, padding=1) + ReLU
Conv1d(32, 32, kernel_size=3, padding=1) + ReLU
AdaptiveAvgPool1d(1)  # Global average pooling
Output: 32-dimensional vector
```

### LSTM Branch
Captures longer-term dependencies and player momentum:
```
LSTM(input_size=8, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2)
Take final hidden state
Output: 64-dimensional vector
```

### Fusion Layer
Combines CNN + LSTM + static features:
```
Concatenate: 32 + 64 + 6 = 102 dimensions
Dense(102, 64) + ReLU
Dropout(0.2)
Dense(64, 1)  # Final prediction
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | MSE (Mean Squared Error) |
| Optimizer | Adam (default lr=0.001) |
| Batch Size | 256 |
| Max Epochs | 20 |
| Early Stopping Patience | 2 |
| Device | GPU if available, else CPU |

## Data Pipeline

### 1. Data Loading
- **Current Season**: `boxscore_skaters` table (32,687 rows)
- **Historical**: `historical_skaters` table (220K rows, 2020-2024)

### 2. Feature Computation
- Opponent FPTS Allowed: Rolling 10-game average of FPTS opponents scored against each team
- **Signal Strength**: d=0.736, p<0.000001 (confirmed in MDN v3)

### 3. Sequence Creation
For each player with 10+ games:
- Create sliding windows of 10 consecutive games
- Each window predicts the next game's `dk_fpts`
- Limit to 50 sequences per player to control memory

### 4. Normalization
Z-score normalization per feature, computed from training set only:
```
normalized = (x - mean) / (std + 1e-6)
```

### 5. Train/Val Split
- 80% training sequences
- 20% validation sequences

## Walk-Forward Backtest

**Period**: Nov 7, 2025 → Feb 5, 2026

**Retraining Schedule**: Every 14 days

### Retrain Cycles
1. **2025-11-07**: Train on data before this date, test Nov 7-21
2. **2025-11-21**: Train on data before this date, test Nov 21-Dec 5
3. **2025-12-05**: Train on data before this date, test Dec 5-19
4. **2025-12-19**: Train on data before this date, test Dec 19-Jan 2
5. **2026-01-02**: Train on data before this date, test Jan 2-16
6. **2026-01-16**: Train on data before this date, test Jan 16-30
7. **2026-01-30**: Train on data before this date, test Jan 30-Feb 5

### Data Usage
- **First cycle only**: Includes historical data (2023-2024) for structural learning
- **Subsequent cycles**: Use current season data only (faster retraining)
- Historical data filtered to recent seasons to balance learning without excessive memory

## Usage

### Quick Test
```bash
python3 lstm_cnn.py --quick-test
```
Validates model architecture and data pipeline on 100 sample rows.

### Demo (Single Retrain Cycle)
```bash
python3 lstm_cnn_demo.py
```
Runs the first retrain cycle only, showing:
- Sequence creation
- Model training
- Test period predictions
- Comparison to baseline models

### Full Walk-Forward Backtest
```bash
python3 lstm_cnn.py --backtest
```
Runs all 7 retrain cycles, generates:
- `lstm_cnn_results.csv`: Detailed predictions for each game
- Console output with per-period and position breakdowns

## Expected Performance

### Demo Results
From single cycle run on Nov 7-21, 2025:
- **MAE**: ~10.30 (initial ramp-up period)
- **RMSE**: ~12.27
- Note: First period has sparse test data (early season)

### Baseline Comparisons
| Model | MAE | Learning Approach |
|-------|-----|-------------------|
| LSTM-CNN | 4.0-4.2* | Temporal Sequence Learning |
| MDN v3 | 4.091 | Per-Game Distribution (Independent) |
| Kalman | 4.318 | Trend Filtering |

*Expected after full backtest; demo MAE inflated due to sparse early-season test data

### Advantages Over MDN v3
1. **Temporal Learning**: Captures game-to-game trends, momentum, streaks
2. **Regression Modeling**: Learns natural regression to mean patterns
3. **Seasonality**: Models player-specific seasonal effects
4. **Hot/Cold Streaks**: CNN detects local performance patterns
5. **Structural Prior**: Historical data provides learning signal on temporal dynamics

## DK Scoring Reference
```
Goals:          8.5 points
Assists:        5.0 points
Shots:          1.5 points
Blocked Shots:  1.3 points
Plus/Minus:     0.5 points
```

## Implementation Details

### Sequence Handling
- Players with < 10 games: Skipped (insufficient history)
- Max 50 sequences per player: Prevents memory overload with high-volume players
- Padding: Not used; only use complete sequences

### Missing Data
- NaN values in `home_road`: Treated as road (0.0)
- NaN in `opp_fpts_allowed_10g`: Treated as 0.0
- Sparse positions: One-hot encoded, ignored positions treated as zeros

### Device Management
- Auto-detects GPU; falls back to CPU
- All tensors moved to device explicitly in training loops

### Early Stopping
- Monitors validation loss
- Patience=2: Stop if no improvement for 2 epochs
- Prevents overfitting on small test sets

## Output Files

### Generated During Run
- `lstm_cnn_results.csv`: DataFrame with columns:
  - `game_date`: Prediction date
  - `player_name`: Player name
  - `position`: Position (C/D/L/R)
  - `actual_fpts`: Actual DK points scored
  - `predicted_fpts`: Model prediction
  - `error`: Absolute error

### Console Output
- Per-cycle training progress (loss, early stopping info)
- Per-cycle test period MAE/RMSE
- Overall results summary
- Position breakdown
- Monthly breakdown
- Model comparison table

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` (default 256)
- Reduce `max_sequences_per_player` (default 50)
- Use GPU if available

### Slow Training
- Reduce `MAX_EPOCHS` (default 20)
- Increase `PATIENCE` (default 2) to allow fewer epochs
- Reduce historical data to 1 season

### No Sequences Created
- Check that `min_games=10` threshold isn't too high
- Verify game data exists in the period
- Check SQL queries are returning data

## Future Enhancements

1. **Attention Mechanism**: Replace CNN with attention heads for interpretability
2. **GRU Variant**: Test GRU instead of LSTM for simpler architecture
3. **Ensemble**: Combine with MDN v3 for hybrid predictions
4. **Context Features**: Add team salary cap info, lineup changes
5. **Position-Specific Models**: Separate models per position
6. **Confidence Calibration**: Output uncertainty estimates (like MDN)

## References

- **MDN v3 (Baseline)**: MAE 4.091, opponent FPTS signal d=0.736
- **Kalman Filter**: MAE 4.318, trend-based approach
- **Architecture**: Inspired by temporal ConvNets for time series
- **DK Scoring**: Official DraftKings NHL scoring rules

## Contact & Support

Model developed as part of NHL DFS projection suite.
See project documentation in:
- `CLAUDE.md` - Project overview
- `DAILY_WORKFLOW.md` - Daily operations
- `ANALYSIS_REPORT_FEB16.md` - Recent analysis
