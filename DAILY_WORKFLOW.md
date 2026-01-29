
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
│  • Results feed into projection bias and improvement plans                   │
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

### Data folders

All paths are relative to the `projection/` folder. The code looks in these directories first (with fallback to project root where applicable).

| Folder | Purpose |
|--------|---------|
| **daily_salaries/** | DK salary CSVs per slate (e.g. `DKSalaries_M.DD.YY.csv`) |
| **vegas/** | Vegas lines per slate (e.g. `VegasNHL_M.DD.YY.csv`) |
| **backtests/** | xlsx backtest workbooks (e.g. `1.26.26_nhl_backtest.xlsx`) |
| **contests/** | DK contest result CSVs (e.g. `$5main_NHL_M.DD.YY.csv`) |
| **daily_projections/** | Generated projection and lineup CSVs (dates in filenames) |

---

## Daily Workflow

### Phase 0: Review Latest Backtest (optional but recommended)

Before building today's slate, know where your model stands:

- **Option A:** Open the most recent backtest spreadsheet (**backtests/X.XX.XX_nhl_backtest.xlsx**) and skim MAE, correlation, and bias by position.
- **Option B:** Run `python backtest.py --players 75` to refresh metrics.

Use this to spot weak spots (e.g. goalie bias, position over/under) before locking in stacks.

---

### Phase 1: Data Collection & Verification

#### Step 1: Check Data Sources First
Run all APIs and verify data before anything else.

```bash
cd /Users/brendanhorlbeck/Desktop/Code/projection

# 1. Fetch line combinations and confirmed goalies
python lines.py

# 2. Check for postponements/schedule changes
# Review output for any games marked as postponed
```

**Data Source Checklist:**
- [ ] Line combinations loaded for all games
- [ ] Confirmed goalies identified (usually ~5pm ET)
- [ ] No postponed games (remove affected players)
- [ ] PP1 unit assignments noted

#### Step 2: Download Vegas Lines
Save as `VegasNHL_M.DD.YY.csv` in the **vegas/** folder.

**Key Vegas Metrics:**
- Team implied totals (3.5+ = high scoring environment)
- Game totals (6.5+ = shootout potential)
- Line movement (sharp money indicators)

#### Step 3: Download DraftKings Salary File
1. Go to DraftKings → NHL → Select contest
2. Export CSV → Save as `DKSalaries_M.DD.YY.csv` in the **daily_salaries/** folder
3. Cross-reference - remove players from postponed games

---

### Phase 2: Slate Analysis (THE CRITICAL STEP)

#### Step 4: Analyze Slate Characteristics

**Slate Size Strategy:**
| Slate Size | Games | Stack Strategy |
|------------|-------|----------------|
| Small | 2-4 | Concentrate in ONE game, max line + PP overlap |
| Medium | 5-7 | 1 primary line stack + secondary correlation |
| Large | 8+ | 2 line stacks from different games |

**Ceiling Game Identification:**
The #1 question: Which team/line will EXCEED its Vegas expectation tonight?

Check these signals for each game:
1. **Goalie Vulnerability**
   - [ ] Backup goalie starting?
   - [ ] Starter on back-to-back?
   - [ ] Recent save % struggles?

2. **Team Mean Regression**
   - [ ] High-skill team with recent cold streak?
   - [ ] Star player due for breakout (5-game avg vs season avg)?

3. **Special Teams Edge**
   - [ ] Elite PP vs weak PK matchup?
   - [ ] PP hot streak (unsustainable but exploitable)?
   - [ ] **Penalties volume**: Opponent takes a lot of penalties (boost your PP1); your team takes a lot (more PK, tougher for goalie). Matchups matter partly because of penalties taken/allowed.

4. **Rest & Schedule**
   - [ ] Team off rest vs back-to-back opponent?
   - [ ] Travel/timezone advantage?

**Chalk Trap Detection:**
Historical pattern shows top Vegas teams often BUST:
- Jan 22: CAR (Vegas #1 → Actual #11), EDM (Vegas #2 → Actual #14)
- Jan 23: COL (Vegas #1 → Actual #7), NYR (Vegas #3 → Actual #16)

Don't blindly stack the highest Vegas total. Look for WHY a team will exceed expectations.

#### Step 5: Build Stack Thesis
Before running projections, document your conviction:

```
PRIMARY STACK: [TEAM] - [Why they'll exceed expectations]
- Line: [Players]
- Catalyst: [Goalie vulnerability / regression / matchup edge]
- Ownership read: [Chalk / moderate / contrarian]

SECONDARY STACK: [TEAM or bring-back]
- Line: [Players]
- Correlation path: [Same line / PP overlap / opposing goalie]
```

**Stack Types Reference:**
| Stack Type | Description | When to Use |
|------------|-------------|-------------|
| Line Stack (3-man) | C + LW + RW same line | High conviction game |
| Line Stack (2-man) | Any 2 forwards same line | Moderate conviction |
| PP1 Stack | Players on first PP unit | PP-heavy matchup |
| Team Stack (4+) | 4+ players same team | Small slate, max correlation |
| Bring-back | Stack + 1 opposing player | Hedge correlation |

---

### Phase 3: Projection Generation

#### Step 6: Generate Projections
```bash
# Full run with stacks and injury report
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

#### Step 7: Review Output Against Your Thesis
The system outputs:
1. **Injury Report** - Who's out, who's DTD
2. **Line Combinations** - Current lines from DailyFaceoff
3. **Confirmed Goalies** - Which goalies are starting
4. **Top Projections** - Highest projected players
5. **Value Plays** - Best pts/$ plays
6. **Stacking Recommendations** - Best correlated groups

**Cross-check with your thesis:**
- Does the optimizer's stack align with your conviction?
- If not, why? (Salary constraints? Missing data?)
- Override if your thesis has strong catalyst support

---

### Phase 4: Lineup Construction & Export

#### Step 8: Build & Export Final Lineups

**Lineup Construction Framework:**
1. **Lock in core stack** (2-4 players from primary game)
2. **Add secondary correlation** (bring-back or second stack)
3. **Fill with ceiling pieces** (high-variance individuals)
4. **Select goalie** (win probability + saves upside)

**Ownership Leverage Guide:**
| Ownership | Strategy | When to Use |
|-----------|----------|-------------|
| 0-5% | Max leverage, need strong thesis | Large fields, top-heavy payouts |
| 5-15% | Balanced EV | Standard GPPs |
| 15-25% | Only if ceiling probability is elite | Small fields |
| 25%+ | Avoid in GPPs | Cash games only |

**DraftKings Constraints:**
- $50,000 salary cap
- Roster: 2C, 3W, 2D, 1G, 1UTIL
- **Minimum 3 teams** (enforced by DK)

**Generate and export lineups:**
```bash
# Generate multiple lineups for GPP
python main.py --lineups 5 --stacks

# Export for DraftKings upload (writes to daily_projections/)
python main.py --lineups 20 --export projections.csv
```
Projections and lineup exports are written to **daily_projections/** (filenames include date).

#### Step 9: Final Verification
Before submitting:
- [ ] Check Twitter for late scratches/line changes
- [ ] Re-verify goalie confirmations if close to lock
- [ ] Confirm 3-team minimum met
- [ ] Salary under $50,000
- [ ] Stack thesis documented for post-slate review

---

### Phase 5: Contest Entry

Use the lineups you built and exported in Phase 4.

1. Use exported CSV or manually enter on DraftKings
2. Verify lineup validity on DraftKings
3. Submit before lock

---

### Phase 6: Post-Slate Analysis (Next Morning)

#### Step 1: Download Contest Results
1. DraftKings → My Contests → Click on finished contest
2. Download standings/results CSV
3. Save as `$5main_NHL_M.DD.YY.csv` in the **contests/** folder

#### Step 2: Update Backtest Spreadsheet
1. Open **backtests/X.XX.XX_nhl_backtest.xlsx**
2. Add actual FPTS and ownership to "Actual" sheet
3. Compare to "Projection" sheet

#### Step 3: Run Backtest
```bash
python backtest.py --players 75
```
Review output (MAE, RMSE, correlation) and compare to your backtest spreadsheet. Use this to tune the model.

#### Step 4: Update Plans
Based on backtest results, update:
- `DAILY_PROJECTION_IMPROVEMENT_PLAN.md` - FPTS model tuning
- `OWNERSHIP_PLAN.md` - Ownership model tuning

---

## Quick Commands Reference

### Before Slate (optional)
```bash
# Review latest backtest metrics
python backtest.py --players 75
```

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

### Post-slate: run backtest
```bash
python backtest.py --players 75
```
(See Phase 6, Step 3 for full post-slate validation.)

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

| Document | Purpose |
|----------|---------|
| `DAILY_PROJECTION_IMPROVEMENT_PLAN.md` | **Strategic framework** - Stack theory, ceiling identification, leverage analysis, backtest case studies |
| `OWNERSHIP_PLAN.md` | Ownership model details and calibration |
| `README.md` | Project setup (if exists) |

**Quick Reference:**
- This doc (`DAILY_WORKFLOW.md`) = **WHAT to do, WHEN**
- Improvement Plan = **WHY (strategy, theory, historical analysis)**

---

*Last Updated: January 26, 2026*
