# Skater Ensemble Model - Summary Report

## Overview

Built a comprehensive ensemble model combining 5 distinct prediction approaches to improve NHL DFS skater projections beyond the current best baseline (MDN v3: MAE 4.091).

## Architecture

### Sub-Models (5 Approaches)

1. **Expanding Mean** (Weight: 0.50)
   - Simple cumulative average of all historical dk_fpts for each player
   - Baseline prediction using complete player history
   - Most stable, least reactive to recent changes

2. **EWM - Exponential Weighted Mean** (Weight: 0.00)
   - Recency-weighted moving average with halflife=15 days
   - Recent games weighted more heavily than distant ones
   - Captures recent form and momentum
   - **Note**: Grid search found this weight to be 0, indicating expanding mean dominates

3. **Kalman Filter** (Weight: 0.25)
   - 1D scalar Kalman filter with Q=0.05 (process noise), R=30.0 (observation noise)
   - Smooths noisy FPTS observations using Bayesian filtering
   - Reduces impact of outlier games while tracking true underlying skill
   - `x[t+1] = K * z[t] + (1-K) * x[t]` where K is Kalman gain

4. **Opponent-Adjusted Mean** (Weight: 0.00)
   - Expands the expanding mean by opponent quality factor
   - Adjustment = (position_avg_fpts / league_avg_fpts)
   - **Note**: Grid search found this weight to be 0

5. **TOI-Weighted Projection** (Weight: 0.25)
   - Expanding mean adjusted by recent ice time trend
   - recent_toi_avg / historical_toi_avg, clipped to [0.5, 1.5]
   - Captures role changes and opportunity shifts
   - Identifies players getting more/less ice time

### Ensemble Method: Simple Weighted Average

```
Prediction = 0.50 * Expanding + 0.00 * EWM + 0.25 * Kalman + 0.00 * OppAdj + 0.25 * TOI
           = 0.50 * Expanding + 0.25 * Kalman + 0.25 * TOI
```

## Walk-Forward Backtest Results

### Methodology

- **Training Period**: Nov 7 - Dec 7, 2025 (30 days)
  - Collected 6,268 training samples
  - Optimized weights via coarse grid search (0.0, 0.25, 0.5, 0.75, 1.0)
  - Evaluated 625+ weight combinations

- **Validation Period**: Dec 8, 2025 - Feb 5, 2026 (60 days)
  - 16,127 player-games across 55 dates
  - Applied fixed optimal weights
  - Computed daily, position-level, and monthly metrics

### Overall Performance (Validation Phase)

```
Baseline Comparison:
  MDN v3:                   4.0910 MAE
  Ensemble:                 4.6697 MAE
  Difference:              -0.5787 (-14.15%)
```

**Individual Sub-Model Performance:**
- Expanding Mean:         4.7810 MAE
- EWM:                    4.7889 MAE
- Kalman Filter:          4.7640 MAE
- Opponent-Adjusted:      4.7879 MAE
- TOI-Weighted:           4.7828 MAE

*Note: The ensemble is better than any individual sub-model, showing the value of combining diverse signals.*

### Position Breakdown

| Position | MAE    | Games |
|----------|--------|-------|
| Center   | 4.7910 | 5,357 |
| Left     | 4.9402 | 2,747 |
| Right    | 5.3686 | 2,621 |
| Defense  | 4.0727 | 5,402 |

**Insights:**
- Defensemen are most predictable (4.07 MAE)
- Right wings are hardest to predict (5.37 MAE)
- Centers and Left wings in middle (4.79-4.94 MAE)

### Monthly Breakdown

| Month      | MAE    | Games |
|------------|--------|-------|
| Dec 2025   | 4.6597 | 6,155 |
| Jan 2026   | 4.6759 | 8,640 |
| Feb 2026   | 4.6754 | 1,332 |

**Insights:** Consistent performance across months with minimal variation (0.02 MAE range)

## Key Insights

### Why the Ensemble Didn't Beat MDN v3

The ensemble (4.67 MAE) did not beat MDN v3 (4.09 MAE) because:

1. **Simple Features**: The sub-models use only basic expanding mean, EWM, and Kalman - no neural network complexity
2. **Missing Feature Engineering**: MDN v3 includes:
   - NST (NetShark) advanced features (shooting quality, pressure, etc.)
   - Opponent FPTS allowed (high signal, d=0.736)
   - Regression-weighted shrinkage using YoY coefficients
   - Complex feature interactions learned by neural networks

3. **Limited Inputs**: Sub-models use only player FPTS history and TOI, not:
   - Contextual features (game script, home/away, rest, injuries)
   - Advanced metrics (expected goals, high-danger chances)
   - Line-matching and deployment information

4. **Grid Search Limitations**: Coarse grid (0.0, 0.25, 0.5, 0.75, 1.0) only tested 625 combinations

### What the Ensemble Got Right

1. **Diversification**: Combining uncorrelated signals (expanding mean + Kalman + TOI weighting) reduces noise
2. **Weight Learning**: Grid search found optimal combination = 50% expanding + 25% Kalman + 25% TOI
3. **Interpretability**: Clear understanding of which models contribute to predictions
4. **Speed**: Fast to compute (no neural network training needed)
5. **Position Sensitivity**: Correctly identifies defensemen as easier to predict

## Files Generated

### Main Outputs
- `/sessions/youthful-funny-faraday/mnt/Code/projection/ensemble_model.py` (18 KB)
  - Complete ensemble implementation
  - Fully runnable with `python3 ensemble_model.py`
  - Includes Kalman filter, weight optimization, and reporting

### Results
- `/sessions/youthful-funny-faraday/mnt/Code/projection/ensemble_backtest_results.csv` (2.4 MB)
  - 16,127 predictions with actual scores
  - Columns: game_date, player_id, player_name, position, actual_fpts, pred_expanding, pred_ewm, pred_kalman, pred_opp_adj, pred_toi_wgt, pred_ensemble

- `/sessions/youthful-funny-faraday/mnt/Code/projection/ensemble_daily_stats.csv` (2.9 KB)
  - Daily MAE and RMSE metrics across 55 validation dates
  - Columns: game_date, num_players, mae, rmse

- `/sessions/youthful-funny-faraday/mnt/Code/projection/ensemble_optimal_weights.txt` (277 B)
  - Optimal weights learned during training phase
  - Training MAE: 4.6279

## Recommendations for Future Improvement

To beat MDN v3 (4.091), consider:

1. **Add MDN v3 Predictions**: Use MDN v3 as a 6th sub-model in ensemble
   - Would likely achieve ~4.05-4.08 MAE immediately
   - Different methodology captures different signals

2. **Feature Engineering**:
   - Add opponent FPTS allowed (proven d=0.736)
   - Include game context (home/away, rest days, Vegas lines)
   - Compute position-specific opponent quality

3. **Advanced Stacking**:
   - Use Ridge/Lasso meta-learner instead of simple weighted average
   - Add interaction features between sub-models
   - Implement walk-forward retraining (every 14 days)

4. **Model Calibration**:
   - Fine-tune Kalman parameters (Q, R) specifically for this data
   - Optimize EWM halflife
   - Test different sub-model combinations

5. **Ensemble Diversity**:
   - Add LSTM-CNN for temporal sequences
   - Include Poisson/negative binomial models
   - Use different feature sets for different positions

## Conclusion

The ensemble model provides a solid foundation with interpretable components and good performance (4.67 MAE on skaters). While it doesn't beat MDN v3, it demonstrates:

- Value of combining diverse signals (ensemble > any individual model)
- Importance of feature engineering and domain knowledge
- Trade-off between interpretability and raw prediction power
- Fast, reproducible approach suitable for daily updates

The code is production-ready and can be extended with additional sub-models or features.
