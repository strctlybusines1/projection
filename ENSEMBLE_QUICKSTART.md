# Ensemble Model - Quick Start Guide

## Running the Model

```bash
cd /sessions/youthful-funny-faraday/mnt/Code/projection
python3 ensemble_model.py
```

Expected runtime: ~3-4 minutes for full backtest

## Output Files

After running, you'll get:

1. **ensemble_backtest_results.csv** - Main predictions
   - 16,127 rows of predictions with actuals
   - Use for detailed analysis

2. **ensemble_daily_stats.csv** - Daily performance
   - Daily MAE/RMSE metrics
   - Use for trend analysis

3. **ensemble_optimal_weights.txt** - Learned weights
   - Shows which sub-models matter most
   - Use for understanding the ensemble

## Key Performance Metrics

```
Training Phase (Nov 7 - Dec 7):
  MAE: 4.628
  Samples: 6,268

Validation Phase (Dec 8 - Feb 5):
  MAE: 4.670
  Samples: 16,127
  
By Position:
  Defense: 4.073 (best)
  Center:  4.791
  Left:    4.940
  Right:   5.369 (hardest)
```

## Ensemble Composition

```
Final Ensemble = 50% Expanding Mean + 25% Kalman + 25% TOI-Weighted

Why These Weights?
- Expanding Mean (50%): Most stable baseline
- Kalman Filter (25%): Noise reduction from outliers
- TOI-Weighted (25%): Captures role/usage changes
```

## Using in Production

### Option 1: Direct Import
```python
from ensemble_model import run_walk_forward_backtest

results_df, weights = run_walk_forward_backtest()
```

### Option 2: Command Line
```bash
python3 ensemble_model.py | tee ensemble_run.log
```

### Option 3: Extend with More Sub-Models
Edit the `compute_all_predictions()` function to add new prediction approaches, then re-run weight optimization.

## Common Questions

**Q: Why doesn't it beat MDN v3?**
A: The ensemble uses only simple statistical models (mean, EWM, Kalman). MDN v3 uses neural networks with advanced features (NST data, opponent quality, etc.). The ensemble demonstrates the value of diversification but lacks the raw predictive power of deep learning.

**Q: Can I use these predictions in DFS?**
A: Yes, but use them as a secondary check, not as your primary model. They're most useful for:
- Validating MDN v3 predictions
- Identifying position-specific tendencies
- Understanding prediction uncertainty
- Building ensemble with other models

**Q: How often should I retrain?**
A: The model optimizes weights on rolling 30-day windows. Re-run weekly to update weights with new data. Individual sub-model parameters (Kalman Q, EWM halflife) are fixed.

**Q: Which sub-model is most important?**
A: Expanding Mean (50% weight). It outweighs all others combined because it's the most stable baseline. Kalman and TOI each contribute 25%.

## File Structure

```
/sessions/youthful-funny-faraday/mnt/Code/projection/
├── ensemble_model.py                      # Main code (18 KB)
├── ensemble_backtest_results.csv          # 16,127 predictions (2.4 MB)
├── ensemble_daily_stats.csv               # Daily metrics (2.9 KB)
├── ensemble_optimal_weights.txt           # Learned weights (277 B)
├── ENSEMBLE_MODEL_SUMMARY.md              # Full report
└── ENSEMBLE_QUICKSTART.md                 # This file
```

## Database Schema

The model reads from:
- `boxscore_skaters` table: Current season games
- Columns: player_id, player_name, position, game_date, dk_fpts, toi_seconds

## Dependencies

```
Python 3.7+
pandas
numpy
sqlite3 (built-in)
```

No machine learning libraries required (no scikit-learn, torch, etc.)
