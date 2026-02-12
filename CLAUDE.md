# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## NHL DFS Project
- Project location: ~/Desktop/Code/projection/
- To run the full pipeline: `python main.py`
- Key dependencies: tabpfn, and any ML libraries requiring OpenMP
- If OpenMP duplicate library errors occur, set `export KMP_DUPLICATE_LIB_OK=TRUE` before running
- New data files are often downloaded as zips in ~/Downloads — unzip before copying

## Git & GitHub
- Use SSH for git push (not HTTPS) to avoid auth token issues
- Always verify the remote URL with `git remote -v` before pushing
- If push fails with auth errors, switch to SSH: `git remote set-url origin git@github.com:USER/REPO.git`

## Project Overview

NHL DFS (Daily Fantasy Sports) projection and lineup optimization system for DraftKings. Generates player fantasy point projections from multiple data sources, predicts ownership percentages, and builds salary-cap-constrained GPP lineups with correlated stacking.

## Daily Workflow (Order of Operations)

### Pre-Slate Workflow (2-3 hours before lock)

```bash
# Step 1: Download DK salary CSV from DraftKings
# Save to: projection/daily_salaries/DKSalaries_M.DD.YY.csv

# Step 2: First run of day with Edge (fetches + caches Edge data)
python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge

# Step 3: Subsequent runs (uses cached Edge data - much faster)
python main.py --stacks --show-injuries --lineups 5 --edge

# Step 4: Single-entry mode — generate 40 candidates, auto-select best
python main.py --stacks --show-injuries --lineups 40 --edge --single-entry

# Step 5: Review output files in daily_projections/
#   - {date}_projections_{timestamp}.csv  (player projections)
#   - {date}_lineups_{timestamp}.csv      (optimized lineups)
#   - {date}_lines.json                   (line combos/stacks)
```

### Pre-Lock Checklist (30 mins before lock)

```bash
# Confirm starting goalies (critical!)
python lines.py  # Check goalie confirmations

# Check for late scratches
# - Compare lineup players to DailyFaceoff
# - Any player with uncertain status → have pivot ready

# Verify lineup positions against DK
# - DK uses LW/RW/C/D, not just W/C
# - Confirm UTIL slot eligibility
```

### Post-Slate Workflow

```bash
# Step 1: Download contest results from DraftKings
# Save to: projection/contests/

# Step 2: Create actuals file with: name, actual, own, TOI

# Step 3: Run backtest to measure accuracy
python backtest.py --players 75

# Step 4: Update bias corrections if needed (see Calibration section)
```

## Common Commands

```bash
# === PRIMARY WORKFLOW ===

# Generate projections WITHOUT Edge (faster, use for initial build)
python main.py --stacks --show-injuries --lineups 5

# First run of day with Edge (fetches from API + caches)
python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge

# Subsequent runs with Edge (uses cache - seconds vs minutes)
python main.py --stacks --show-injuries --lineups 5 --edge

# Generate projections with NO Edge (explicit skip)
python main.py --stacks --show-injuries --lineups 5 --no-edge

# === SINGLE-ENTRY MODE ===

# Generate 40 candidates, score via SE engine, select best lineup
python main.py --stacks --show-injuries --lineups 40 --edge --single-entry

# SE with forced stack (e.g., force a MIN stack)
python main.py --stacks --show-injuries --lineups 40 --edge --single-entry --force-stack PP1

# === CONTEST-AWARE MODE ===

# For GPP optimization with EV scoring
python main.py --stacks --show-injuries --lineups 20 \
  --contest-entry-fee 5 --contest-field-size 10000 --contest-payout top_heavy_gpp

# === BACKTESTING ===

# Standard backtest against actual results
python backtest.py --players 75

# Edge stats backtest (calibrate boost values)
python backtest.py --edge-backtest

# Ownership model backtest
python backtest.py --ownership-backtest

# Batch backtest across multiple dates
python backtest.py --batch-backtest

# === UTILITIES ===

# Test line scraper + stack builder
python lines.py

# Test optimizer standalone
python optimizer.py

# Launch local dashboard
python dashboard/server.py

# === SIMULATION MODE ===

# Run lineup simulation (deterministic)
python main.py --simulate

# Run Monte Carlo simulation
python main.py --simulate --sim-iterations 100

# Two-pass lift-adjusted simulation
python main.py --simulate --sim-lift 0.15
```

All commands run from the `projection/` directory. There is no build step, linter, or test suite — validation is done via backtesting (`backtest.py`).

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MAIN.PY PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. LOAD DATA                                                           │
│     ├── DK Salaries (daily_salaries/*.csv)                              │
│     ├── Vegas Lines (The Odds API or vegas/*.csv fallback)              │
│     └── Line Combos (DailyFaceoff scrape via lines.py)                  │
│                              ↓                                          │
│  2. FETCH STATS                                                         │
│     └── data_pipeline.py → NHL API, MoneyPuck, Natural Stat Trick       │
│                              ↓                                          │
│  3. ENGINEER FEATURES                                                   │
│     └── features.py → rates, bonuses, matchup adjustments               │
│                              ↓                                          │
│  4. GENERATE PROJECTIONS                                                │
│     └── projections.py → base FPTS calculation + bias corrections       │
│                              ↓                                          │
│  5. APPLY EDGE BOOSTS (if --edge flag)                                  │
│     └── edge_stats.py → speed/OZ/bursts percentile boosts               │
│                              ↓                                          │
│  6. MERGE WITH SALARIES                                                 │
│     └── Fuzzy name matching, position normalization                     │
│                              ↓                                          │
│  7. PREDICT OWNERSHIP                                                   │
│     └── ownership.py → Ridge model or heuristic fallback                │
│                              ↓                                          │
│  8. OPTIMIZE LINEUPS                                                    │
│     └── optimizer.py → salary cap, stacks, correlation                  │
│                              ↓                                          │
│  8b. SINGLE-ENTRY SELECTION (if --single-entry)                         │
│     └── single_entry.py → score candidates on 6 dimensions, pick best   │
│                              ↓                                          │
│  9. EXPORT                                                              │
│     └── daily_projections/*.csv                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Purpose |
|--------|---------|
| **main.py** | CLI entry point. Orchestrates full pipeline. |
| **data_pipeline.py** | Fetches NHL API, MoneyPuck injuries, Natural Stat Trick xG/Corsi. |
| **features.py** | Computes per-game rates, bonus probabilities, opponent adjustments. |
| **projections.py** | Calculates expected FPTS with bias corrections. |
| **edge_stats.py** | Fetches NHL Edge tracking data, applies projection boosts (skaters + goalies). |
| **edge_cache.py** | Caches Edge stats daily to avoid redundant API calls. |
| **recent_scores_cache.py** | Caches recent game scores daily to avoid redundant API calls. |
| **lines.py** | Scrapes DailyFaceoff for lines/PP/goalies. Builds stack correlations. |
| **ownership.py** | Predicts ownership via Ridge/XGBoost regression or heuristic. |
| **optimizer.py** | Builds DK-legal lineups under salary cap with stacking. |
| **single_entry.py** | Scores candidate lineups for SE contests (goalie quality, stack correlation, salary efficiency, leverage). |
| **backtest.py** | Compares projections to actuals. Outputs MAE/RMSE/correlation. |
| **config.py** | Central configuration: DK scoring, API URLs, weights. |

### DraftKings NHL Roster

2 Centers (C), 3 Wings (W — covers LW/RW), 2 Defensemen (D), 1 Goalie (G), 1 UTIL (C or W only, D excluded). Salary cap: $50,000.

### Data Directories

All relative to `projection/`:

| Folder | Contents |
|--------|----------|
| `daily_salaries/` | DK salary CSVs (`DKSalaries_M.DD.YY.csv`) |
| `vegas/` | Vegas lines CSV fallback (`VegasNHL_M.DD.YY.csv`) |
| `daily_projections/` | Output: projection CSVs, lineup CSVs, lines JSON |
| `backtests/` | Backtest xlsx workbooks + `latest_mae.json` |
| `contests/` | DK contest result CSVs (for post-slate analysis) |
| `cache/` | Daily caches: Edge stats, goalie Edge, recent scores |

## NHL Edge Stats Integration

### Overview

NHL Edge provides player tracking data since 2021-22: skating speed, shot speed, zone time, and distance metrics. The `edge_stats.py` module fetches this data via `nhl-api-py` and applies projection boosts for players with elite underlying metrics.

### Edge & Data Caching Workflow

Edge stats, goalie stats, and recent scores are **cumulative season data** that update **once daily** (overnight after games). Caching avoids redundant API calls.

```bash
# First run of day: fetch fresh data and cache it
python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge

# Subsequent runs: use cached data (seconds vs minutes)
python main.py --stacks --show-injuries --lineups 5 --edge

# Skip Edge entirely (fastest)
python main.py --stacks --show-injuries --lineups 5 --no-edge
```

**Performance with caching:**
| Scenario | Old Time | New Time | Savings |
|----------|----------|----------|---------|
| First run of day | 16 min | 5-6 min | 62% |
| 2nd-5th runs | 16 min each | 1-2 min each | 90% |
| Full day (5 runs) | 80 min | 12 min | 85% |

**Cache locations:**
- `projection/cache/edge_stats_{date}.json` - Skater Edge (speed, OZ%, bursts)
- `projection/cache/goalie_edge_stats_{date}.json` - Goalie Edge (EV SV%, QS%)
- `projection/cache/recent_scores_{date}.json` - Recent game scores (last 1/3/5 games)

**When to refresh:**
- Morning (first run): `--refresh-edge` to get overnight updates
- During the day: use cache (data doesn't change)
- Pre-lock: optional `--refresh-edge` if paranoid

### Skater Metrics Tracked

| Metric | Description | DFS Value |
|--------|-------------|-----------|
| **Max Skating Speed** | Top speed in mph with league percentile | Breakaway/rush potential |
| **Bursts Over 20mph** | Count of explosive skating bursts | Transition game indicator |
| **Offensive Zone Time %** | Share of ice time in OZ | More scoring chances |
| **Zone Starts %** | OZ vs DZ faceoff starts | Usage/deployment indicator |
| **Shot Speed** | Hardest shot in mph | Scoring threat level |

### Goalie Metrics Tracked (NEW)

| Metric | Description | DFS Value |
|--------|-------------|-----------|
| **EV Save %** | Even-strength save percentage | Core goalie skill |
| **Quality Starts %** | % of games with QS | Consistency indicator |
| **PP/SH Save %** | Special teams save % | Situational performance |

### Skater Boost Thresholds (Calibrated 2/3/26)

Calibrated from 1,180-observation backtest across Jan 22 - Feb 2, 2026:

| Metric | Elite (≥90th) | Above-Avg (≥65th) | Backtest Correlation |
|--------|---------------|-------------------|---------------------|
| OZ Time | +10% | +4% | r=+0.18 (strongest) |
| Bursts | +5% | - | r=+0.15 |
| Speed | +2% | +1% | r=+0.07 (weakest) |

Maximum combined boost: ~17% for players elite in all metrics.

### Goalie Boost Thresholds (Calibrated 2/4/26)

| Metric | Elite Threshold | Boost | Penalty Threshold | Penalty |
|--------|-----------------|-------|-------------------|---------|
| EV Save % | ≥92.0% | +8% | <89.0% | -6% |
| EV Save % | ≥90.5% | +4% | - | - |
| Quality Starts % | ≥60% | +6% | <40% | -4% |
| Quality Starts % | ≥50% | +3% | - | - |

Maximum combined boost: ~14.5% for elite goalies. Maximum penalty: ~10% for struggling goalies.

### Backtest Validation

```bash
# Validate Edge boost calibration
python backtest.py --edge-backtest

# Expected output: correlation values for each metric
# If correlations drop below thresholds, recalibrate boosts
```

## Projection Calibration

### Current Bias Correction Values

| Parameter | Value | Notes |
|-----------|-------|-------|
| `GLOBAL_BIAS_CORRECTION` | 0.80 | Skater multiplier |
| Centers (`C`) | 1.01 | Near neutral |
| Wings (`W`/`L`/`R`/`LW`/`RW`) | 0.99 | Near neutral |
| Defensemen (`D`) | 1.00 | Near neutral |
| `GOALIE_BIAS_CORRECTION` | 0.40 | Goalie multiplier |

### Calibration Process

When recalibrating, always use projection CSVs generated with the **current** correction values:

```bash
# 1. Generate projections with current model
python main.py --stacks --show-injuries --lineups 5

# 2. After games complete, run backtest against that day
python backtest.py --skater-slate-date YYYY-MM-DD

# 3. If bias ≠ 0, calculate new correction:
#    new_correction = old_correction × (actual/projected ratio)
```

**Warning**: Never derive corrections from CSVs generated with different correction values — this causes calibration drift.

### Backtest TOI Filtering

`backtest.py` filters out players with TOI=0 (scratches/DNPs) before computing error metrics. Always include TOI in actuals data.

## Ownership Regression Model

### Approach

The ownership model has three paths:

1. **Ridge regression (default)**: Trained on ~1,200 historical contest observations. Loaded from `backtests/ownership_model.pkl`.
2. **TabPFN (alternative)**: TabPFN v6.3.1 regressor for comparison.
3. **Heuristic (fallback)**: Original 12-factor multiplicative model when no pickle exists.

### Retraining

```bash
# Run LODOCV for both Ridge and TabPFN
python backtest.py --ownership-backtest

# Train Ridge model and save pickle
python backtest.py --train-ownership

# Train TabPFN model instead
python backtest.py --train-ownership --ownership-tabpfn
```

### Performance (LODOCV)

| Metric | Value |
|--------|-------|
| Mean MAE | 4.16 |
| Mean RMSE | 5.92 |
| Mean Spearman | 0.725 |

## Key Patterns

- **Name matching**: Fuzzy matching at 0.85 threshold via `difflib.SequenceMatcher`.
- **Position normalization**: LW/RW/R/L → `W`. C/W → `C`. LD/RD → `D`.
- **Goalie opponent exclusion**: Optimizer removes skaters from goalie's opponent team.
- **Stack correlation**: PP1 (0.95), Line1 (0.85), Line1+D1 (0.75), Line2 (0.70).
- **Ownership normalization**: Targets ~900% total (9 roster spots × ~100% each).

## Known Gotchas

### Critical Issues

1. **Salary merge column whitelist**: `merge_projections_with_salaries()` uses explicit `merge_cols` list. Any column from DK salary data needed downstream **must be manually added** or it gets silently dropped.

2. **Stale salary files**: Optimizer loads latest file from `daily_salaries/`. If the file is old, few players match — silently produces zero lineups. **Always verify salary file date matches target slate.**

3. **Late scratches**: Players with TOI=0 are common. **Always check for late scratches 30 mins before lock** and have pivots ready.

4. **DK position mismatches**: Your projection file may show different positions than DK (e.g., Guentzel as C when DK has him as W). **Always verify positions against actual DK salary file before finalizing lineup.**

### Edge-Specific Issues

5. **Edge API timing**: Running `--edge` can fail if NHL API is slow. If this happens:
   - Run base projections first (no `--edge`)
   - Run again with `--edge` flag
   - Or skip Edge for that slate (`--no-edge`)

6. **Goalie Edge data**: Now available! Uses EV Save % and Quality Starts % from NHL Stats API (high-danger save % not exposed by NHL API).

### Name Matching Issues

7. **Fuzzy match false positives**: 0.85 threshold can match wrong players with similar names. Stack-building code should verify team membership after matching.

8. **Special characters**: Names like "Stützle" may not match "Stutzle". Check for encoding issues.

## External APIs & Rate Limits

| API | Purpose | Rate Limit |
|-----|---------|------------|
| NHL API (api-web.nhle.com) | Player stats, schedules | 0.3s delay |
| MoneyPuck (moneypuck.com) | Injury CSV | No auth |
| Natural Stat Trick | xG, Corsi, PDO | 2.0s delay |
| DailyFaceoff | Line combos, goalies | 0.5s delay |
| The Odds API | Vegas lines | Requires API key in `.env` |
| NHL Edge (via nhl-api-py) | Tracking data | 0.3s delay |

## Environment Setup

- Always activate the correct conda/venv environment before running scripts
- Required dependency: `pip install tabpfn` (not installed by default)
- If encountering duplicate library conflicts (e.g., OpenMP), export KMP_DUPLICATE_LIB_OK=TRUE

Requires `.env` file in `projection/` with:
```
ODDS_API_KEY=<the-odds-api-key>
```

Key Python dependencies: `pandas`, `numpy`, `requests`, `nhl-api-py`, `tabpfn`, `scikit-learn`, `flask`, `python-dotenv`, `tqdm`, `scipy`.

## Quick Reference: Contest Strategy

### Single-Entry GPP (use --single-entry flag)
- Generate 40+ candidates, let SE selector pick best construction
- Goalie is highest-leverage decision — SE scorer weights floor + matchup heavily
- 3-4 man primary stack with LINE-MATE correlation (not just same team)
- Moderate leverage: 1-2 contrarian plays, rest solid mid-owned
- D should be in a stack OR cheap — no expensive one-off D
- Target $49,400-$50,000 salary usage
- Scoring: projection (35%), ceiling (15%), goalie (15%), stacks (15%), salary (10%), leverage (10%)

### WTA (Winner-Take-All) - 10 person
- Maximum differentiation required
- Fade chalk aggressively
- Stack 4-5 players from highest O/U game
- Contrarian goalie in low-total game

### GPP (Guaranteed Prize Pool) - 100+ entries
- Balance ceiling + floor
- 3-4 man primary stack
- Some chalk OK if highest ceiling
- Target top 10-20% for cash

### Cash Games (50/50, Double-Up)
- Prioritize floor over ceiling
- Play chalk/safe plays
- Confirmed goalies only
- Avoid volatile players

## Troubleshooting

### "No lineups generated"
- Check salary file date matches slate
- Verify `min_teams` requirement met
- Check for position eligibility issues

### "Edge boosts not applied"
- Run two-step process (base then edge)
- Check `nhl-api-py` is installed
- Verify NHL Edge API is responding

### "Projection file has wrong positions"
- Positions come from DK salary file
- Re-download salary file if stale
- Manually verify against DK before lock

### "Name not found in merge"
- Check for special characters in names
- Try manual fuzzy match lookup
- Add player to manual mapping if needed
