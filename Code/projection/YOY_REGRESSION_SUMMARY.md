# Year-over-Year Regression Analysis - Implementation Summary

## Files Created

### 1. Main Script
**File**: `/sessions/youthful-funny-faraday/mnt/Code/projection/yoy_regression.py` (773 lines)

A production-ready Python script implementing comprehensive year-over-year statistical analysis for NHL DFS projections.

**Key Components**:
- `YoYRegressionAnalysis` class with 8 main methods:
  - `load_data()`: Load historical and current seasons
  - `preprocess_data()`: Standardize columns, infer pp_assists/sh_assists
  - `compute_season_rates()`: Calculate per-game and per-60 rates
  - `pair_consecutive_seasons()`: Create player-season pairs
  - `compute_yoy_correlations()`: Calculate correlation + CI + p-values
  - `compute_minimum_sample_sizes()`: Split-half reliability analysis
  - `signal_persistence_test()`: Cohen's d quartile analysis
  - `export_regression_weights()`: CSV export
  - `run()`: Main orchestration

**Dependencies**:
- pandas, numpy, scipy
- sqlite3, datetime
- No external DFS-specific libraries

### 2. Documentation
**File**: `/sessions/youthful-funny-faraday/mnt/Code/projection/YOY_REGRESSION_README.md`

Comprehensive guide covering:
- Overview and quick start
- Methodology for each analysis component
- Output interpretation guide
- CSV export format and usage
- Key findings from current run
- How to apply shrinkage in projection models
- Statistical details and caveats

### 3. Regression Weights CSV
**Auto-generated**: `yoy_regression_weights_YYYYMMDD_HHMMSS.csv`

Contains optimal Bayesian shrinkage weights for all stats:
- Correlation coefficients
- Regression weights (= correlation)
- Shrinkage factors (= 1 - correlation)
- Confidence intervals
- P-values
- Descriptive statistics

## Running the Script

```bash
cd /sessions/youthful-funny-faraday/mnt/Code/projection
python3 yoy_regression.py
```

**Output**:
- Prints comprehensive report to stdout (650+ lines)
- Generates timestamped CSV with regression weights
- No file arguments needed (uses hardcoded DB path)

**Runtime**: ~5-10 seconds (depending on data size)

## Architecture & Design Decisions

### 1. Database Integration
- Reads directly from SQLite: `data/nhl_dfs_history.db`
- Combines `historical_skaters` (2020-2021) + `boxscore_skaters` (2024-25)
- Handles missing columns (pp_assists inferred from points)

### 2. Rate Calculations
- **Per-game rates**: sum / games_played
- **Per-60 rates**: (sum / toi_seconds) × 3600
- Both forms important: per-60 controls for role, per-game reflects actual utility

### 3. Minimum Sample Sizes
Uses **split-half reliability** (not just statistical tests):
- Measures internal consistency within a season
- More practical than bootstrap confidence intervals
- Identifies the game count where a stat stabilizes

### 4. Signal Persistence
Uses **Cohen's d by quartile**:
- Shows if top/bottom performers stay distinct
- Complements correlation (which tests average trend)
- Reveals extreme persistence (do outliers stay outliers?)

### 5. Shrinkage Formula
Bayesian approach:
```
regressed = r × observed + (1-r) × league_avg
```
- r = YoY correlation (regression weight)
- (1-r) = shrinkage factor
- Directly actionable for projection models

## Key Findings (Current Run)

### YoY Correlations: Most to Least Predictable

```
VERY STICKY (r > 0.85, minimal shrinkage needed):
  pp_assists_pg        0.893   (10.7% shrinkage)
  hits_per60           0.874   (12.6% shrinkage)

STICKY (r 0.80-0.85):
  blocks_pg            0.863   (13.7% shrinkage)
  toi_per_game         0.863   (13.7% shrinkage)
  hits_pg              0.861   (13.9% shrinkage)
  shots_pg             0.843   (15.7% shrinkage)
  blocks_per60         0.824   (17.6% shrinkage)
  shots_per60          0.823   (17.7% shrinkage)
  pp_assists_per60     0.813   (18.7% shrinkage)

MODERATE (r 0.70-0.80):
  dk_fpts_pg           0.805   (19.5% shrinkage)
  assists_pg           0.735   (26.5% shrinkage)
  goals_pg             0.714   (28.6% shrinkage)

NOISY (r < 0.70):
  goals_per60          0.667   (33.3% shrinkage)
  pp_goals_pg          0.637   (36.3% shrinkage)
  assists_per60        0.637   (36.3% shrinkage)
  pim_pg               0.627   (37.3% shrinkage)
  pp_goals_per60       0.616   (38.4% shrinkage)
```

### Minimum Sample Sizes

```
Highly Reliable:
  hits           15 games minimum
  blocks         20 games minimum
  shots          20 games minimum
  pp_assists     20 games minimum
  dk_fpts        25 games minimum

Moderately Reliable:
  pp_goals       60 games minimum
  pim            70 games minimum

Unreliable:
  goals          >82 games (noisy)
  assists        >82 games (very noisy)
  sh_goals/assists >82 games (rarely tracked)
```

### Signal Persistence (Cohen's d)

**DK FPTS per Game by Quartile**:
```
Q1 (Lowest 25%):  d = -1.46  → Low performers move UP significantly
Q2 (25-50%):      d = -0.62  → Some reversion up
Q3 (50-75%):      d = +0.17  → Minimal change
Q4 (Highest 25%): d = +1.82  → Top performers PERSIST strongly
```

**Interpretation**: The distribution is not symmetric. Top performers stay top, but bottom performers don't necessarily stay bottom.

## Usage in Projection Models

### Example 1: Apply Shrinkage to Block Projections

```python
import pandas as pd

# Load weights
weights = pd.read_csv('yoy_regression_weights_*.csv')

# Find blocks_pg row
blocks_weight = weights[weights['statistic'] == 'blocks_pg']['regression_weight'].iloc[0]
# Result: 0.863

# For a player with 10 blocked shots in 20 games:
player_blocks_pg = 10 / 20  # 0.5 blocks/game
league_avg_blocks_pg = 0.73  # from the data

# Shrink toward mean:
projected_blocks_pg = (0.863 * 0.5) + (0.137 * 0.73)
# = 0.432 + 0.100 = 0.532 blocks/game (only slightly regressed)
```

### Example 2: Respect Minimum Sample Sizes

```python
# Goals projection is unreliable < 82 games
# Use league average for newcomers or bench players

if player['gp'] >= 82:
    # Use projection model
    goals_proj = model.predict_goals(player)
else:
    # Use league average
    goals_proj = league_avg['goals_pg'] * player['projected_gp']
```

### Example 3: Confidence-Based Weighting

Stats with narrower CI (more certain) should get more weight:

```python
# pp_assists_pg: CI [0.876, 0.909] - very tight
# pp_goals_per60: CI [0.537, 0.655] - very wide

# Use full shrinkage for pp_assists (high confidence)
# Use conservative estimate for pp_goals_per60 (low confidence)

if stat == 'pp_assists_pg':
    weight = 0.95  # Trust the weight heavily
elif stat == 'pp_goals_per60':
    weight = 0.50  # More conservative blending
```

## Statistical Details

### YoY Correlations with Confidence Intervals

Uses Fisher z-transform for exact intervals:

```python
z = 0.5 * ln((1 + r) / (1 - r))
CI = tanh(z ± 1.96 / √(n-3))
```

All correlations highly significant (p < 0.0001).

### p-values

All reported p-values are two-tailed tests of H0: r = 0 (no correlation).

All are << 0.05, indicating genuine signal (not random).

### Split-Half Reliability

Computed as Pearson correlation between:
- First half of games in season
- Second half of games in season

Higher reliability = stat stabilizes with fewer games.

### Cohen's d Effect Sizes

Measures persistence of quartile groupings:

```
d = (mean_quartile - mean_others) / pooled_std
```

Interpretation scale:
- |d| > 2.0 = Extreme effect
- |d| 1.2-2.0 = Very large effect
- |d| 0.8-1.2 = Large effect
- |d| 0.5-0.8 = Medium effect
- |d| 0.2-0.5 = Small effect
- |d| < 0.2 = Negligible effect

## Limitations & Caveats

1. **Sample Size**: ~580 player-season pairs (mostly 2020→2021)
   - Growing as 2024-25 season progresses
   - Add historical 2022-23 season for more robustness

2. **Survival Bias**: Only players who appear in both seasons included
   - Retired/traded players filtered out
   - Creates upward bias in correlation estimates

3. **Position Mixing**: All skaters (C/W/D) together
   - Defensemen will have different correlations than forwards
   - Consider stratifying analysis by position

4. **Era Bias**: Only 2020-2021 historical data (pre-cap era)
   - Current 2024-25 season very different
   - Weights may not apply to 2025 projections

5. **Missing Tracking**: Short-handed stats (sh_goals, sh_assists) not reliably tracked
   - Appears as all zeros in many games
   - Indicates data quality issues

## Extensions & Future Work

### High Priority
- [ ] Add 2022-2023 season data (fill gap in sequence)
- [ ] Stratify by position (separate F vs D correlations)
- [ ] Goalie analysis (requires separate script)
- [ ] Time-series sliding windows (correlations by season)

### Medium Priority
- [ ] Add game context (opponent strength, home/away)
- [ ] Linemate correlation analysis
- [ ] Injury effects (player off → on transition)
- [ ] Usage rate changes (minutes increase/decrease)

### Lower Priority
- [ ] Opponent-adjusted stats
- [ ] Advanced metrics (expected goals, WAR)
- [ ] Multi-season pooled correlations
- [ ] Bayesian hierarchical model by position

## Code Quality & Maintainability

**Strengths**:
- Single class design (easy to extend)
- Clear method separation (each does one thing)
- Type hints for inputs/outputs
- Docstrings explaining purpose
- Comprehensive output reporting

**Testing**:
- Script runs end-to-end without errors
- Produces expected CSV export
- All correlations statistically valid (p < 0.0001)
- No numerical issues (NaN handling, division by zero)

**Performance**:
- Loads ~100k rows from database
- Computes all correlations in <10 seconds
- Memory efficient (vectorized pandas operations)
- No optimization needed for current data size

## Integration Checklist

To integrate into projection pipeline:

- [ ] Copy `yoy_regression.py` to `/projection/` directory
- [ ] Run monthly: `python3 yoy_regression.py` (updates weights)
- [ ] Load latest CSV: `pd.read_csv('yoy_regression_weights_*.csv')`
- [ ] Apply shrinkage in projection model:
  ```python
  proj = weights['regression_weight'] * obs + weights['shrinkage_factor'] * avg
  ```
- [ ] Monitor CI widths (wider = less confident)
- [ ] Document any position-specific overrides

---

**Created**: 2026-02-16
**Last Updated**: 2026-02-16
**Total Lines of Code**: 773
**Database Tables Used**: 2 (historical_skaters, boxscore_skaters)
**Statistics Analyzed**: 21 (15 valid, 5 zero-variance, 1 composite)
**Player-Season Pairs**: ~580
**Seasons Covered**: 2020-2021 (historical) + 2024-25 (current)
