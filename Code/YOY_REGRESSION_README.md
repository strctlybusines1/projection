# Year-over-Year Regression Analysis for NHL DFS

## Overview

`yoy_regression.py` is a comprehensive statistical analysis tool that computes year-over-year correlation coefficients for NHL player statistics to determine true regression rates. This Jim Simons-style analysis reveals which player stats are "sticky" (predictable) vs. "noisy" (regress to mean).

**Key Insight**: The YoY correlation for each stat tells you exactly how much to shrink projections toward league average in your model.

## Quick Start

```bash
cd /sessions/youthful-funny-faraday/mnt/Code/projection
python3 yoy_regression.py
```

Output includes:
- YoY correlation table for all stats
- Minimum sample sizes per stat
- Signal persistence tests (Cohen's d by quartile)
- Regression weights CSV export

## What It Does

### 1. **Per-Season Rate Computation**

For each player-season combination with 20+ games (15+ in COVID-shortened 2020-21):

**Per-game rates:**
- goals/gp, assists/gp, shots/gp, blocks/gp, hits/gp, pim/gp
- pp_goals/gp, pp_assists/gp, sh_goals/gp, sh_assists/gp
- dk_fpts/gp

**Per-60 rates (using toi_seconds):**
- goals_per60, assists_per60, shots_per60, blocks_per60, hits_per60
- pp_goals_per60, pp_assists_per60, sh_goals_per60, sh_assists_per60

**Other:**
- toi_per_game (minutes per game)

### 2. **Year-over-Year Correlation Analysis**

For each player who appears in consecutive seasons:
- Pairs Y1 and Y2 data
- Computes Pearson correlation for each stat
- Calculates 95% confidence intervals using Fisher transform
- Computes p-values
- Derives shrinkage factors

**Result**: 580 player-season pairs (mostly 2020→2021, some recent seasons in current 2024-25)

### 3. **Minimum Sample Size Determination**

For each stat, tests split-half reliability at different game counts:

```
Game counts tested: 10, 15, 20, 25, 30, 40, 50, 60, 70, 82
```

**Method**:
1. Group players by season
2. For each game count threshold:
   - Take first N games of each player-season
   - Split into first half vs second half
   - Compute correlation between halves
3. Find game count where reliability reaches 0.70+

**Example Output**:
```
blocks: 20 games minimum (reliability trajectory: 0.59 → 0.68 → 0.74 → 0.77 → 0.80 → ...)
```

This tells you: "Player blocking stats are reliable (0.70+ internal consistency) after 20 games"

### 4. **Signal Persistence Testing**

Tests whether player groupings persist across seasons using effect sizes.

**Method**:
1. Divide players into quartiles based on Y1 rate
2. Compute Cohen's d for each quartile comparing Y2 values
3. |d| measures how much outliers stay outliers

**Interpretation**:
- |d| > 1.2 = Strong persistence (top players stay top)
- |d| 0.5-1.2 = Moderate (some reversion to mean)
- |d| < 0.5 = Weak (heavy regression to mean)

## Key Output: The Correlation Table

```
Statistic                      Corr     N      95% CI               P-val      Shrink
---
pp_assists_pg                    0.893    580  [0.876, 0.909]       0.0000***    0.107
hits_per60                       0.874    580  [0.853, 0.892]       0.0000***    0.128
blocks_pg                        0.867    580  [0.845, 0.885]       0.0000***    0.134
dk_fpts_pg                       0.804    580  [0.773, 0.832]       0.0000***    0.196
goals_pg                         0.705    580  [0.662, 0.744]       0.0000***    0.295
pp_goals_pg                      0.625    580  [0.572, 0.672]       0.0000***    0.375
```

### Interpretation by Correlation Level

| Range | Meaning | Projection Action |
|-------|---------|-------------------|
| r > 0.80 | Very sticky | Use mostly observed (85%+ weight) |
| r 0.60-0.80 | Sticky | Use mostly observed (65-80% weight) |
| r 0.40-0.60 | Moderate | Blend equally with league avg (50/50) |
| r < 0.40 | Noisy | Heavy shrinkage toward league avg (30% weight) |

## The Regression Weight Formula

The script exports optimal Bayesian shrinkage weights:

```
regressed_value = regression_weight × observed_value + shrinkage_factor × league_average
```

**Examples**:
- pp_assists_pg (r=0.893): `0.893 × observed + 0.107 × avg`
  - Use 89.3% of observed, only 10.7% league average shrinkage
- goals_pg (r=0.705): `0.705 × observed + 0.295 × avg`
  - Use 70.5% observed, 29.5% league average shrinkage

## CSV Export: Regression Weights

Each run creates a timestamped CSV:
```
/sessions/youthful-funny-faraday/mnt/Code/projection/yoy_regression_weights_YYYYMMDD_HHMMSS.csv
```

**Columns**:
- `statistic`: Stat name (e.g., "goals_pg")
- `yoy_correlation`: Raw YoY correlation coefficient
- `regression_weight`: Optimal Bayesian weight (= correlation)
- `shrinkage_factor`: Weight toward league average (= 1 - correlation)
- `n_pairs`: Number of player-season pairs
- `p_value`: Statistical significance
- `ci_lower`, `ci_upper`: 95% confidence interval
- `y1_mean`, `y2_mean`: Mean values in Y1 vs Y2
- `y1_std`, `y2_std`: Standard deviations

Use this CSV in your projection pipeline to apply optimal shrinkage.

## Data Source

### Database: `/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db`

**Tables used**:
- `historical_skaters`: Seasons 2020-2021 (~60k rows)
- `boxscore_skaters`: Current 2024-25 season (~33k rows)

**Columns mapped**:
```
historical_skaters:
  - season (int): 2020, 2021
  - player_name, team, position
  - goals, assists, shots, blocked_shots (→ blocks)
  - hits, pim
  - pp_goals, pp_points (→ pp_assists)
  - sh_goals, sh_points (→ sh_assists)
  - toi_seconds
  - dk_fpts

boxscore_skaters:
  - season inferred as 2024
  - pp_assists calculated from available data
  - sh_goals, sh_assists: set to 0 (not tracked in boxscores)
```

## Key Findings (Current Run)

```
Average YoY correlation: 0.770
Median YoY correlation: 0.813
Range: [0.606, 0.893]
```

**Interpretation**: On average, 23% of a player's stats regress toward league average year-over-year. The other 77% remain stable.

### Most Predictable Stats (r > 0.80)
```
pp_assists_pg        r=0.893  (89.3% weight)
hits_per60           r=0.874  (87.4% weight)
blocks_pg            r=0.867  (86.7% weight)
toi_per_game         r=0.867  (86.7% weight)
hits_pg              r=0.860  (86.0% weight)
shots_pg             r=0.838  (83.8% weight)
blocks_per60         r=0.828  (82.8% weight)
shots_per60          r=0.824  (82.4% weight)
```

### Least Predictable Stats (r < 0.65)
```
pp_goals_per60       r=0.606  (60.6% weight)
assists_per60        r=0.637  (63.7% weight)
pp_goals_pg          r=0.625  (62.5% weight)
pim_pg               r=0.627  (62.7% weight)
```

### Signal Persistence Highlights

**DK FPTS per Game** (Cohen's d by quartile):
- Q1 (lowest): d = -1.45 (strongly regress upward)
- Q2: d = -0.62 (moderate reversion)
- Q3: d = 0.17 (weak persistence)
- Q4 (highest): d = 1.82 (strongly persist)

**Interpretation**: Top performers stay top (d=1.82), bottom performers move up (d=-1.45).

## Minimum Sample Sizes

```
Statistic           Min Games for 0.70+ Reliability
---
hits                15 games
blocks              20 games
shots               20 games
pp_assists          20 games
dk_fpts             25 games
goals               60 games (unreliable below 82)
pp_goals            60 games
assists             >82 games (noisy stat)
pim                 70 games
sh_goals, sh_assists  >82 games (never tracked well)
```

**Usage**: Only trust blocked shots projections after a player has 20+ games.

## How to Use in Your Model

### 1. **Apply Shrinkage to Observed Rates**

```python
import pandas as pd

# Load regression weights
weights = pd.read_csv('yoy_regression_weights_20260216_174716.csv')

# For each player stat:
for _, row in weights.iterrows():
    stat = row['statistic']
    reg_weight = row['regression_weight']

    # Projected value
    projected = (
        reg_weight * player[f'{stat}_observed'] +
        (1 - reg_weight) * league_avg[stat]
    )
```

### 2. **Respect Minimum Sample Sizes**

```python
# Only use goals projection if player has 60+ games
if player['games_played'] >= 60:
    goals_proj = calculate_goals_projection(...)
else:
    goals_proj = league_avg['goals_pg'] * player['expected_gp']
```

### 3. **Interpret Confidence Intervals**

Higher p-value + wider CI = less reliable correlation. Adjust weights accordingly.

```python
# Stat with CI [0.55, 0.65] (wide) is less certain than
# Stat with CI [0.82, 0.87] (narrow)
```

## Statistical Details

### YoY Correlation Computation

1. **Pearson Correlation**: Standard correlation coefficient
2. **95% CI**: Fisher z-transform with normal approximation
   - z = 0.5 × ln((1+r)/(1-r))
   - CI = tanh(z ± 1.96/√(n-3))
3. **p-value**: Two-tailed test against H0: r=0
4. **Shrinkage Factor**: 1 - r (Bayesian interpretation)

### Split-Half Reliability

Internal consistency measured within-season:
- First half of games vs second half
- Indicates stability of the measurement
- Higher reliability = stat is measured precisely

### Cohen's d

Effect size for signal persistence:
```
d = (mean_q - mean_others) / pooled_std
```

- Measures how distinct high/low performers remain year-over-year
- |d| > 1.2 indicates strong persistence

## Caveats & Limitations

1. **Sample Size**: Only ~580 player-season pairs (2020-2021 era, growing)
2. **League Changes**: NHL rules, pace, and talent distribution change
3. **Survival Bias**: Players who appear in Y2 are non-random (no retirements/injuries)
4. **Position Mixing**: Stats mixed across positions (C/W/D differences not separated)
5. **Team Effects**: No adjustment for team strength or usage changes

## Future Enhancements

- [ ] Stratify by position (F vs D)
- [ ] Stratify by experience level (rookie vs veteran)
- [ ] Time-series: use rolling correlations for recent seasons
- [ ] Goalie analysis (separate script needed)
- [ ] Add 2022-2023 season data
- [ ] Context adjustments (strength of schedule, line changes)

## Contact / Questions

The analysis is fully self-contained in `yoy_regression.py`. Modify the `YoYRegressionAnalysis` class to extend functionality.

---

**Last Updated**: 2026-02-16
**Data Seasons**: 2020, 2021, 2024-25 (current)
**Player-Season Pairs**: ~580
