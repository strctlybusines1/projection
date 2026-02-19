# Year-over-Year Regression Analysis - File Index

## Files Created (4 total)

### 1. Main Script (Production)
**File**: `yoy_regression.py` (773 lines)

The complete implementation. Run with:
```bash
python3 yoy_regression.py
```

**What it does**:
- Loads data from SQLite database
- Computes per-season player rates (per-game, per-60)
- Pairs consecutive seasons for same players
- Calculates YoY correlations with confidence intervals
- Determines minimum sample sizes via split-half reliability
- Tests signal persistence with Cohen's d by quartile
- Exports regression weights to CSV

**Key Methods**:
- `load_data()` - Load historical + current seasons
- `compute_season_rates()` - Calculate rates with min GP filter
- `pair_consecutive_seasons()` - Match Y1 → Y2 players
- `compute_yoy_correlations()` - Main correlation analysis
- `compute_minimum_sample_sizes()` - Split-half reliability
- `signal_persistence_test()` - Quartile-based effect sizes
- `export_regression_weights()` - CSV export

---

### 2. Documentation (Comprehensive)
**File**: `YOY_REGRESSION_README.md` (10 KB)

Full explanation of methodology, interpretation, and usage.

**Sections**:
1. Overview and Quick Start
2. What It Does (detailed breakdown)
3. Key Output: The Correlation Table
4. The Regression Weight Formula
5. CSV Export Format
6. Data Source (database schema)
7. Key Findings (current run)
8. How to Use in Your Model
9. Statistical Details
10. Caveats & Limitations
11. Future Enhancements

**Use this file**: To understand the analysis methodology and interpretation.

---

### 3. Implementation Summary
**File**: `YOY_REGRESSION_SUMMARY.md` (10 KB)

Technical implementation details and findings.

**Sections**:
1. Files Created (this list)
2. Running the Script
3. Architecture & Design Decisions
4. Key Findings (detailed results)
5. Usage in Projection Models (examples)
6. Statistical Details
7. Limitations & Caveats
8. Extensions & Future Work
9. Code Quality & Maintainability
10. Integration Checklist

**Use this file**: To understand the implementation, integrate into your pipeline, and see technical details.

---

### 4. Sample Output
**File**: `YOY_REGRESSION_SAMPLE_OUTPUT.txt` (3 KB)

Actual output from a complete script run (truncated for display).

**Shows**:
- Data loading summary
- YoY correlation table (all 21 statistics)
- Minimum sample sizes (game counts for reliability)
- Signal persistence tests (Cohen's d by quartile)
- Regression summary with key insights
- Regression weights and shrinkage formulas
- Interpretation guide
- Export confirmation

**Use this file**: To see what the script output looks like without running it.

---

## Quick Reference: What to Read

### I want to...

**Run the analysis**
→ Execute `python3 yoy_regression.py`

**Understand the methodology**
→ Read `YOY_REGRESSION_README.md` sections 1-2

**Learn interpretation**
→ Read `YOY_REGRESSION_README.md` section 3

**See sample output**
→ Read `YOY_REGRESSION_SAMPLE_OUTPUT.txt`

**Integrate into my model**
→ Read `YOY_REGRESSION_README.md` section 8, then `YOY_REGRESSION_SUMMARY.md` section 9

**Understand statistical methods**
→ Read `YOY_REGRESSION_README.md` section 9, then `YOY_REGRESSION_SUMMARY.md` section 6

**Extend the script**
→ Read `YOY_REGRESSION_SUMMARY.md` sections 3 and 8

**Know the limitations**
→ Read `YOY_REGRESSION_README.md` section 10, then `YOY_REGRESSION_SUMMARY.md` section 7

---

## Key Outputs Explained

### CSV Export
Generated file: `yoy_regression_weights_YYYYMMDD_HHMMSS.csv`

Contains optimal Bayesian shrinkage weights for all statistics:

| Column | Meaning | Usage |
|--------|---------|-------|
| statistic | Stat name (e.g., "blocks_pg") | Join key |
| yoy_correlation | YoY correlation (-1 to 1) | Diagnostic |
| regression_weight | Bayesian weight (= correlation) | Use in formula |
| shrinkage_factor | Weight toward mean (= 1-r) | Use in formula |
| n_pairs | Sample size (player-season pairs) | Confidence indicator |
| p_value | Statistical significance | Validity check |
| ci_lower, ci_upper | 95% confidence interval | Uncertainty range |
| y1_mean, y2_mean | Mean in Y1 and Y2 | Drift detection |
| y1_std, y2_std | Standard deviations | Volatility |

**Formula to use**:
```
projected_value = regression_weight × observed + shrinkage_factor × league_avg
```

### Printed Report
Contains 5 main tables:

1. **YoY Correlation Table**
   - All statistics ranked by correlation
   - p-values show significance
   - CI shows precision

2. **Minimum Sample Sizes**
   - Game count needed for 0.70+ reliability
   - Trajectory shows how reliability grows

3. **Signal Persistence Tests**
   - Cohen's d for each quartile
   - Shows if outliers persist

4. **Regression Summary**
   - Average/median/range of correlations
   - Classification of sticky vs noisy stats

5. **Regression Weights**
   - Top predictable statistics
   - Exact shrinkage formulas

---

## Data Flow

```
SQLite Database (nhl_dfs_history.db)
├── historical_skaters (2020-2021)
└── boxscore_skaters (2024-25)
    ↓
yoy_regression.py
├── Load & preprocess
├── Compute season rates
├── Pair consecutive seasons (580 pairs)
├── Calculate YoY correlations
├── Determine min sample sizes
├── Test signal persistence
└── Export results
    ├── Print full report (stdout)
    └── Generate yoy_regression_weights_*.csv
```

---

## Statistics Summary

### What's Measured

**21 Statistics**:
- 10 per-game rates (goals, assists, shots, blocks, hits, pim, pp_goals, pp_assists, sh_goals, sh_assists, dk_fpts)
- 10 per-60 rates (same stats, normalized to 60 minutes)
- 1 composite (toi_per_game)

### Valid Results

**15 statistics** with valid correlations:
- Ranges from r=0.616 (pp_goals_per60) to r=0.893 (pp_assists_pg)
- All highly significant (p < 0.0001)
- Sorted by predictability

### Zero-Variance Stats

**5 statistics** with no variance:
- sh_goals_pg, sh_assists_pg, sh_goals_per60, sh_assists_per60
- Reason: Almost all zeros (not tracked in short-handed situations)
- Action: Exclude from models

### Sample

**580 player-season pairs**:
- Mostly 2020 → 2021 transitions
- Growing with 2024-25 season
- No retirements/trades (survivorship bias)

---

## How the Script Works (30-second version)

1. Load ~100k rows from database (2020-2021 history + 2024-25 current)
2. Calculate per-game and per-60 rates for each player-season (20+ games)
3. Find 580 players appearing in consecutive seasons
4. For each statistic, compute correlation of rates Y1 vs Y2
5. Calculate 95% confidence intervals using Fisher transform
6. Test split-half reliability at 10, 15, 20... 82 games
7. Group players by quartiles, test persistence with Cohen's d
8. Export weights as regression_weight = correlation
9. Print comprehensive report explaining results

**Total time**: ~5-10 seconds

---

## Most Important Take-Away

**The regression weight IS the correlation**.

For any player stat, optimal projection uses:
```
projection = r × observed + (1-r) × league_average
```

Where r = year-over-year correlation from this analysis.

**Example**: If pp_assists_pg has r=0.893, then:
```
proj_pp_assists = 0.893 × (player's assists pg) + 0.107 × (league avg)
```

High correlation (r > 0.80) means use mostly observed values.
Low correlation (r < 0.50) means use mostly league average.

This automatically accounts for regression to the mean.

---

## Maintenance & Updates

### Run Monthly
```bash
cd /sessions/youthful-funny-faraday/mnt/Code/projection
python3 yoy_regression.py
```

Generates new CSV with current weights as more 2024-25 data accumulates.

### Add New Historical Seasons
Edit `load_data()` method to include 2022-2023 season when available.

### Stratify by Position
Modify `compute_season_rates()` to split by position (C/W vs D).

### Monitor Convergence
Watch how correlations stabilize as sample size grows from 580 → 1000+ pairs.

---

**Last Updated**: February 16, 2026
**Current Sample Size**: ~580 player-season pairs
**Statistics Analyzed**: 15 valid (5 zero-variance)
**Seasons Covered**: 2020-2021 (historical) + 2024-25 (current)
