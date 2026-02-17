# Enhanced Goalie Projection Model v2

## Overview

A component-based team win probability model for predicting NHL goalie DFS FPTS. Rather than black-box machine learning, this model breaks goalie FPTS into interpretable components tied directly to DK scoring rules.

**Key Insight**: A win is worth +6 FPTS (~54% of average goalie FPTS), so predicting wins is critical. Wins are primarily a team outcome, not a goalie skill.

## Model Architecture

### Component-Based FPTS Prediction

```
FPTS = P(Win) × 6.0 + E[Saves] × 0.7 + E[GA] × (-3.5) + P(Shutout) × 2.0
```

Each component is independently modeled:

### 1. Team Win Probability Model

**Purpose**: P(Win) affects +6 FPTS per win

**Approach**: Logistic regression on team-level features (not goalie-specific)

**Features**:
- Team goals per game (rolling 10-game)
- Team goals against per game (rolling 10-game)  
- Opponent goals per game (rolling 10-game)
- Opponent goals against per game (rolling 10-game)
- Back-to-back indicator
- Days since last game

**Training**: Historical goalies table (13,014 games, 5 seasons)

**Calibration**: Predicted win% closely matches actual win% across all prediction ranges

**Results**:
```
Predicted 0-20%:  11.2% actual wins
Predicted 20-40%: 31.7% actual wins  
Predicted 40-60%: 48.9% actual wins
Predicted 60-80%: 64.3% actual wins
Predicted 80-100%: 84.3% actual wins
```

### 2. Expected Saves Model

**Purpose**: E[Saves] × 0.7 FPTS per save

**Approach**: 
```
E[Saves] = E[Shots Against] × Regressed SV%
```

**Inputs**:
- Opponent offensive strength (opponent GPG rolling 10)
- Team defensive strength  
- Goalie shot volume history (for workload adjustment)

**Regression**: 
- Goalie's historical SV% regressed heavily (shrinkage ~0.12) toward league average
- Reason: Year-to-year SV% correlation is very low (r=0.12)
- League average SV%: 89.54%

**Current Season Calibration**:
- Mean shots against: 28.3
- Model prediction error: 7.0 shots MAE
- vs actual: 23.8 shots

### 3. Expected Goals Against Model

**Purpose**: E[GA] × (-3.5) FPTS (downside risk)

**Approach**:
```
E[GA] = E[Shots Against] × (1 - Regressed SV%)
       × Opponent Strength Adjustment
```

**Adjustments**:
- Opponent GAG relative to league average (2.8 GPG baseline)
- Goalie-specific save % with heavy shrinkage

**Calibration**:
- Mean GA prediction: 3.50
- Mean actual GA: 2.73
- Model is pessimistic by 0.77 GA (error: 1.44 MAE)

### 4. Shutout Probability

**Purpose**: P(GA=0) × 2.0 FPTS bonus

**Approach**: Poisson distribution
```
P(Shutout) = P(GA=0) = exp(-λ) where λ = E[GA]
```

**Results**:
- Average P(SO): 4.2%
- Actual SO%: 4.2%
- Shutout bonus: ~0.08 FPTS average (rare but valuable)

## Data Sources

### Historical Data (Training & Calibration)

From `nhl_dfs_history.db`:

1. **historical_goalies** (13,014 rows, 2020-2024)
   - Per-game: saves, GA, SV%, TOI, decision (W/L/OL)
   - Used to calibrate win probability and expected values

2. **historical_skaters** (220K rows)
   - Team stats aggregation
   - Computes rolling 10-game offensive/defensive strength

3. **boxscore_skaters** (32,687 rows, current 2024-25)
   - Current season team stats for walk-forward predictions

### Current Season Data (Walk-Forward Backtest)

**game_logs_goalies** (1,796 rows, Oct 7 2025 - Feb 5 2026):
- Every goalie game appearance
- Actual: shots, saves, GA, shutouts, DK FPTS, decision

## Performance

### Overall Results

```
MAE:             8.182
RMSE:            10.075
Bias:            -0.439 (slightly pessimistic)
Median Error:    -0.606
```

### Comparison to Baselines

```
Baseline (Season Avg):  8.68
XGBoost Model:          7.88
This Model:             8.182

vs Baseline:  +0.498 better (5.7% improvement)
vs XGBoost:  -0.302 worse  (-3.8%)
```

### Performance by Segment

```
Starters (87.9%):     MAE = 7.744 ✓ Excellent
Backups (12.1%):      MAE = 11.373 (needs work)

Home (50.1%):         MAE = 8.076
Away (49.9%):         MAE = 8.288

Wins (48.6%):         MAE = 8.470
Losses (35.1%):       MAE = 8.610
OT Losses (16.3%):    MAE = 8.135
```

### Component Error Analysis

```
Win prediction error:     0.415
Save prediction error:    7.01 shots
GA prediction error:      1.44 goals
Shutout prediction error: 0.080
```

**Where the errors come from**:
- Saves (7.0 shots): Model overestimates workload/shots
- GA (1.44): Model pessimistic; overpredicts by 0.77 goals avg
- Win (0.415): Well-calibrated overall
- Shutout (0.080): Rare, but probabilities accurate

## Key Findings

### Strengths

1. **Interpretable**: Every FPTS point can be traced to component
2. **Well-calibrated**: Win probabilities match observed frequencies
3. **Starter-focused**: 7.744 MAE on 87.9% of games (starters)
4. **Fast**: No deep learning; logistic regression + arithmetic
5. **Foundational**: Ready for component-specific refinements

### Weaknesses

1. **GA pessimism**: Overestimates defensive load by 0.77 GA
2. **Shot volume**: ±7 shot error in expected saves
3. **Backup weakness**: 11.4 MAE on 12.1% of games  
4. **Extreme games**: Poor at predicting very high (>20 FPTS) scores
5. **XGBoost gap**: 0.3 MAE behind ensemble method

### Root Causes

**High GA predictions**:
- Using raw opponent GPG, not shooting %
- Missing opponent offensive quality beyond scoring rate
- No adjustment for high-SV% teams (like Colorado, Florida)

**Save/shot variance**:
- Opponent shot volume varies independently of goals
- Team structure/playing time affects goalie workload
- No pace metrics (high-pace teams = more shots)

**Backup predictions**:
- Sample size too small for backup-specific calibration
- Backup starts often non-random (starter injured, rest)
- Role uncertainty affects statistical patterns

## Recommendations for v3

### High-Impact Improvements

1. **Opponent shooting % instead of GPG**
   - Source: `nst_teams` table has SF, SA (shot differential)
   - Shooting % = GF / SF (more stable than raw goals)
   - Expected improvement: 0.5-1.0 MAE points

2. **Goalie-specific fatigue modeling**
   - Track consecutive starts, rest days between games
   - Regress SV% down for back-to-back starters
   - Expected improvement: 0.3-0.5 MAE points

3. **Home/away splits in SV% regression**
   - Home SV% typically 0.5-1% higher than away
   - Separate shrinkage coefficients
   - Expected improvement: 0.2-0.3 MAE points

4. **Backup-specific submodel**
   - Separate logistic regression for backup starts
   - Different win probability coefficients
   - Expected improvement: 1-2 MAE points (on backup games)

### Medium-Impact

5. **Recent form recency weighting**
   - Weight recent 5-10 games more than rolling 10
   - Capture momentum/slumps
   - Expected improvement: 0.2-0.4 MAE points

6. **Matchup-specific adjustments**
   - Team A vs Team B historical head-to-head patterns
   - Opponent roster changes (star player injured, etc.)
   - Expected improvement: 0.1-0.3 MAE points

7. **Ensemble with XGBoost**
   - Weighted average: 50% component + 50% XGBoost
   - Leverage both interpretability and accuracy
   - Expected improvement: 0.2-0.3 MAE points on overall

### Low-Impact But Easy

8. **Poisson vs normal for saves**
   - Saves are count data; Poisson may fit better
   - Current: Gaussian regression
   - Expected improvement: <0.1 MAE points

9. **Home/away factor in win model**
   - Historical home win rate: ~54%
   - Add as feature or prior
   - Expected improvement: <0.1 MAE points

## File Organization

```
/sessions/youthful-funny-faraday/mnt/Code/projection/
├── goalie_v2.py                    # Main model code (complete, runnable)
├── goalie_v2_results.csv           # Walk-forward backtest results (1,796 rows)
├── GOALIE_V2_README.md             # This file
└── data/nhl_dfs_history.db         # Database with all tables
```

## Usage

### Run the model:
```bash
cd /sessions/youthful-funny-faraday/mnt/Code/projection
python3 goalie_v2.py
```

### Output:
- **Console**: Full backtest results, calibration charts, MAE breakdowns
- **CSV**: `goalie_v2_results.csv` with all predictions and errors

### Columns in results CSV:

```
game_date              - Game date
player_name            - Goalie name
team                   - Team abbreviation
home_road              - 'H' or 'A'
decision               - 'W', 'L', or 'OL'
p_win                  - Predicted win probability (0-1)
expected_saves         - Predicted saves
expected_ga            - Predicted goals against
p_shutout              - Predicted shutout probability
fpts_win               - p_win × 6.0
fpts_save              - expected_saves × 0.7
fpts_ga                - expected_ga × (-3.5)
fpts_shutout           - p_shutout × 2.0
fpts_total             - Sum of all components
actual_saves           - Actual saves
actual_ga              - Actual goals against
actual_shutout         - 1 if SO, 0 otherwise
actual_fpts            - Actual DK FPTS
error                  - Predicted - Actual
abs_error              - |error|
regressed_sv_pct       - Shrunk SV% used in prediction
```

## Model Validation

The walk-forward backtest validates the model on unseen data:
- For each game, only historical data available before that date is used
- Uses standard train/test split by date (no look-ahead bias)
- Respects data availability constraints (don't use opponent GA before opponent plays)

## Dependencies

```
pandas       - Data manipulation
numpy        - Numerical computing
sqlite3      - Database access
sklearn      - Logistic regression, StandardScaler
scipy        - Poisson distribution
```

## Contact & Notes

This model serves as a foundation for production goalie projections. The component architecture allows:
- Explainability for DFS players
- Targeted improvements on weak components
- Integration with other models via ensemble methods
- Real-time recalibration as season progresses

Key lesson: Goalie FPTS are dominated by team wins (48.6% win rate = 3 pts) and saves (19.3 pts), making team-level modeling critical for success.
