# NHL DFS Projection System - Daily Workflow

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                       │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│   NHL API       │   MoneyPuck     │ Natural Stat    │   DailyFaceoff        │
│   (nhl_api.py)  │   (scrapers.py) │ Trick           │   (lines.py)          │
│                 │                 │ (scrapers.py)   │                       │
│ • Player stats  │ • Injuries      │ • xG/60         │ • Line combos         │
│ • Schedule      │ • Return dates  │ • Corsi/Fenwick │ • PP units            │
│ • Team stats    │                 │ • PDO           │ • Confirmed goalies   │
│ • Game logs     │                 │ • Recent form   │                       │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                     │
         └─────────────────┴─────────────────┴─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE (data_pipeline.py)                      │
│  • Fetches all data sources                                                  │
│  • Builds unified dataset: skaters, goalies, teams, schedule, injuries       │
│  • Filters injured players                                                   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING (features.py)                       │
│  • Per-game rates (goals_pg, shots_pg, etc.)                                │
│  • Bonus probabilities (5+ shots, 3+ points, hat trick)                     │
│  • Opponent adjustments (opp_softness, opp_xga_60)                          │
│  • xG matchup boost, PDO regression, streak adjustments                     │
│  • Injury opportunity boost                                                 │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROJECTIONS (projections.py)                            │
│  • Expected fantasy points calculation                                       │
│  • Position-specific bias corrections (backtest-derived)                     │
│  • Floor/ceiling estimates                                                   │
│  • Optional TabPFN ML model                                                  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────┐     ┌───────────────────────────────────────┐
│   OWNERSHIP (ownership.py)    │     │         OPTIMIZER (optimizer.py)       │
│  • Salary-based curve         │     │  • GPP mode (stacking)                 │
│  • PP1/Line1 boosts           │     │  • Cash mode (consistency)             │
│  • Goalie confirmation        │     │  • Salary cap: $50,000                 │
│  • Value adjustments          │     │  • Roster: 2C, 3W, 2D, 1G, 1UTIL       │
│  • 900% total normalization   │     │  • Team correlation boosts             │
└───────────────────────────────┘     └───────────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN CLI (main.py)                                 │
│  • Command-line interface                                                    │
│  • Loads DK salaries, merges with projections                               │
│  • Formats output, exports CSV                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKTEST (backtest.py)                                │
│  • Compare projections to actual results                                     │
│  • Calculate MAE, RMSE, correlation                                         │
│  • Model comparison (rolling avg vs TabPFN)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Reference

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `config.py` | Constants & settings | DK scoring rules, API URLs, thresholds |
| `nhl_api.py` | NHL API client | `NHLAPIClient` - fetches stats, schedule |
| `scrapers.py` | External scrapers | `MoneyPuckClient` (injuries), `NaturalStatTrickScraper` (xG) |
| `data_pipeline.py` | Data orchestration | `NHLDataPipeline.build_projection_dataset()` |
| `features.py` | Feature engineering | `FeatureEngineer.engineer_skater_features()` |
| `projections.py` | Projection model | `NHLProjectionModel.generate_projections()` |
| `lines.py` | Line combinations | `LinesScraper`, `StackBuilder` |
| `ownership.py` | Ownership prediction | `OwnershipModel.predict_ownership()` |
| `optimizer.py` | Lineup optimizer | `NHLLineupOptimizer.optimize_lineup()` |
| `main.py` | CLI entry point | `main()` - orchestrates everything |
| `backtest.py` | Backtesting | `NHLBacktester.run_backtest()` |

---

## Daily Workflow

### Phase 1: Pre-Slate Preparation (2-3 hours before lock)

#### Step 1: Download DraftKings Salary File
1. Go to DraftKings → NHL → Select contest
2. Export CSV → Save as `DKSalaries_MMDDYY.csv` in project folder

#### Step 2: Generate Projections
```bash
cd /Users/brendanhorlbeck/Desktop/Code/projection

# Basic projection run
python main.py

# Or with specific options
python main.py --stacks --show-injuries

# Full options:
#   --date YYYY-MM-DD    : Specific date (default: today)
#   --salaries FILE      : Specific salary file
#   --stacks             : Show stacking recommendations
#   --show-injuries      : Display injury report
#   --include-dtd        : Include day-to-day players
#   --no-injuries        : Disable injury filtering
#   --lineups N          : Generate N lineups
#   --export FILE        : Export to CSV
```

#### Step 3: Review Output
The system will output:
1. **Injury Report** - Who's out, who's DTD
2. **Line Combinations** - Current lines from DailyFaceoff
3. **Confirmed Goalies** - Which goalies are starting
4. **Top Projections** - Highest projected players
5. **Value Plays** - Best pts/$ plays
6. **Stacking Recommendations** - Best correlated groups
7. **Optimized Lineup** - Ready-to-enter lineup

#### Step 4: Manual Adjustments
Before finalizing:
1. Check Twitter for late news (scratches, line changes)
2. Verify goalie confirmations (usually ~5pm ET)
3. Consider Vegas lines (high totals = more scoring)
4. Adjust for narrative plays the model may miss

---

### Phase 2: Contest Entry

#### Build Final Lineups
```bash
# Generate multiple lineups for GPP
python main.py --lineups 5 --stacks

# Export for DraftKings upload
python main.py --lineups 20 --export projections.csv
```

#### Enter Contests
1. Use exported CSV or manually enter
2. Verify lineup validity on DraftKings
3. Submit before lock

---

### Phase 3: Post-Slate Analysis (Next Morning)

#### Step 1: Download Contest Results
1. DraftKings → My Contests → Click on finished contest
2. Download standings/results CSV
3. Save as `$5main_NHL_M.DD.YY.csv`

#### Step 2: Update Backtest Spreadsheet
1. Open `X.XX.XX_nhl_backtest.xlsx`
2. Add actual FPTS and ownership to "Actual" sheet
3. Compare to "Projection" sheet

#### Step 3: Run Validation
```bash
# Quick validation script
python3 << 'EOF'
import pandas as pd
from ownership import OwnershipModel

# Load projections and actual results
proj_df = pd.read_csv('01_23_26NHLprojections_TIMESTAMP.csv')
contest_df = pd.read_csv('$5main_NHL1.23.26.csv')

# Calculate MAE, correlation, bias
# ... (see backtest code)
EOF
```

#### Step 4: Update Plans
Based on backtest results, update:
- `DAILY_PROJECTION_IMPROVEMENT_PLAN.md` - FPTS model tuning
- `OWNERSHIP_PLAN.md` - Ownership model tuning

---

## Quick Commands Reference

### Daily Run (Most Common)
```bash
# Full run with stacks and injury report
python main.py --stacks --show-injuries

# Generate 3 GPP lineups
python main.py --lineups 3 --stacks
```

### Testing Individual Components
```bash
# Test data pipeline only
python data_pipeline.py

# Test projections only
python projections.py

# Test lines scraper only
python lines.py

# Test optimizer only
python optimizer.py

# Test ownership model
python ownership.py
```

### Backtest Previous Slate
```bash
python backtest.py --players 75
```

---

## Key Configuration (config.py)

### DraftKings Scoring
| Stat | Points |
|------|--------|
| Goal | 8.5 |
| Assist | 5.0 |
| Shot on Goal | 1.5 |
| Blocked Shot | 1.3 |
| SH Point Bonus | +2.0 |

| Bonus | Points | Trigger |
|-------|--------|---------|
| Hat Trick | +3.0 | 3+ goals |
| 5+ Shots | +3.0 | 5+ SOG |
| 3+ Blocks | +3.0 | 3+ blocks |
| 3+ Points | +3.0 | 3+ points |

### Goalie Scoring
| Stat | Points |
|------|--------|
| Win | 6.0 |
| Save | 0.7 |
| Goal Against | -3.5 |
| Shutout Bonus | +4.0 |
| OT Loss | +2.0 |
| 35+ Saves Bonus | +3.0 |

### Bias Corrections (projections.py)
| Position | Correction |
|----------|------------|
| C | 0.97 (-3%) |
| L/LW | 0.96 (-4%) |
| R/RW | 1.01 (+1%) |
| D | 0.95 (-5%) |
| G | 1.05 (+5%) |

---

## Troubleshooting

### "No DraftKings salary file found"
- Download from DK and save as `DKSalaries*.csv` in project folder

### "No games scheduled for today"
- Check the date with `--date YYYY-MM-DD`
- NHL API may not have games loaded yet

### "No confirmed goalies matched"
- DailyFaceoff may not have updated yet
- Manually check goalie starters closer to game time

### Low MAE but high bias
- Model is systematically over/under projecting
- Update bias corrections in `projections.py`

### Ownership predictions way off
- Check if lines data was fetched (confirmed goalies)
- Update multipliers in `ownership.py`

---

## File Dependencies

```
config.py (no dependencies)
    │
    ├──▶ nhl_api.py
    ├──▶ scrapers.py
    ├──▶ data_pipeline.py ──▶ nhl_api.py, scrapers.py
    ├──▶ features.py ──▶ config.py
    ├──▶ projections.py ──▶ config.py, features.py
    ├──▶ lines.py ──▶ config.py
    ├──▶ ownership.py
    ├──▶ optimizer.py ──▶ config.py, lines.py
    ├──▶ main.py ──▶ ALL FILES
    └──▶ backtest.py ──▶ nhl_api.py, config.py
```

---

## Current Model Performance

### Projection Accuracy (as of 1/24/26)
| Metric | Skaters | Goalies | Overall |
|--------|---------|---------|---------|
| MAE | 4.78 | 5.28 | 4.81 |
| Correlation | 0.354 | 0.310 | 0.372 |
| Bias | +1.71 | -1.78 | +1.55 |

### Ownership Accuracy (as of 1/24/26)
| Metric | Value |
|--------|-------|
| MAE | 2.16% |
| Correlation | 0.607 |
| Bias | -0.27% |

---

## Related Documentation
- `DAILY_PROJECTION_IMPROVEMENT_PLAN.md` - FPTS model tuning
- `OWNERSHIP_PLAN.md` - Ownership model details
- `README.md` - Project setup (if exists)

---

*Last Updated: January 24, 2026*
