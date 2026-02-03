# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NHL DFS (Daily Fantasy Sports) projection and lineup optimization system for DraftKings. Generates player fantasy point projections from multiple data sources, predicts ownership percentages, and builds salary-cap-constrained GPP lineups with correlated stacking.

## Common Commands

```bash
# Primary workflow — generate projections, ownership, and lineups
python main.py --stacks --show-injuries --lineups 5

# With contest-aware EV scoring
python main.py --stacks --show-injuries --lineups 20 \
  --contest-entry-fee 5 --contest-field-size 10000 --contest-payout top_heavy_gpp

# Run backtest against actual results
python backtest.py --players 75

# Test line scraper + stack builder
python lines.py

# Test optimizer standalone
python optimizer.py

# Launch local dashboard
python dashboard/server.py
```

All commands run from the `projection/` directory. There is no build step, linter, or test suite — validation is done via backtesting (`backtest.py`).

## Architecture

### Data Flow

```
External APIs → data_pipeline.py → features.py → projections.py
                                                       ↓
                                          ┌────────────┴────────────┐
                                          ↓                        ↓
                                   ownership.py              optimizer.py
                                          ↓                        ↓
                                          └────────────┬───────────┘
                                                       ↓
                                                    main.py → CSV export
```

### Key Modules

- **main.py** — CLI entry point. Orchestrates the full pipeline: load DK salaries, fetch data, generate projections, predict ownership, build lineups, export CSVs. Also fetches Vegas odds via The Odds API (key in `.env`).
- **data_pipeline.py** — `NHLDataPipeline.build_projection_dataset()` fetches from NHL API, MoneyPuck (injuries), and Natural Stat Trick (xG/Corsi/PDO). Returns unified skater/goalie/team DataFrames.
- **features.py** — `FeatureEngineer` computes per-game rates, bonus probabilities, opponent adjustments, xG matchup boosts, and injury opportunity boosts.
- **projections.py** — `NHLProjectionModel.generate_projections()` calculates expected fantasy points with position-specific bias corrections, multiplicative adjustment capping (±15%), high-projection mean regression, and goalie projection cap. Constants are defined at module top. Optional TabPFN ML model.
- **lines.py** — `LinesScraper` scrapes DailyFaceoff for line combos/PP units/confirmed goalies. `StackBuilder` builds correlation data and `get_best_stacks()` returns PP1/Line1/Line1+D1/Line2 groupings with projection matching.
- **ownership.py** — `OwnershipModel.predict_ownership()` uses salary curve, PP1/Line1 role boosts, Vegas totals, goalie confirmation, recent scoring, and TOI surge signals. Normalizes to ~900% total.
- **optimizer.py** — `NHLLineupOptimizer` builds lineups under DK constraints ($50k cap, 2C/3W/2D/1G/1UTIL). GPP mode uses `_get_correlated_stack_players()` to select from actual line/PP combos (PP1 > Line1+D1 > Line1 for primary, Line1 > Line2 > PP1 for secondary) with fallback to top-N-by-projection.
- **simulator.py** — `OptimalLineupSimulator` iterates all valid (team_A, team_B) ordered pairs, builds the best 4-3-1-1 lineup for each, and counts player appearance frequency. Supports deterministic mode (fixed projections) and Monte Carlo mode (samples from `N(projected, std)` each iteration). Computes per-position baseline probability accounting for UTIL slot asymmetry (C/W eligible, D excluded) and a `lift` column (actual_pct / baseline_pct) for context. Run via `python main.py --simulate` or `--simulate --sim-iterations N`. Supports two-pass lift-adjusted re-simulation via `--sim-lift [blend]` (see below).
- **contest_roi.py** — Leverage recommendations and contest EV scoring based on payout structure.
- **backtest.py** — Compares projections to actual scores. Outputs MAE/RMSE/correlation by position. Filters TOI=0 scratches/DNPs from error metrics. Results feed bias corrections in `projections.py`.
- **config.py** — Central configuration: DK scoring rules, API URLs, bias corrections, GPP optimizer settings, signal weights.

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

### External APIs & Rate Limits

- **NHL API** (api-web.nhle.com) — player stats, schedules, game logs. 0.3s delay.
- **MoneyPuck** (moneypuck.com) — injury CSV. No auth.
- **Natural Stat Trick** (naturalstattrick.com) — xG, Corsi, PDO. 2.0s delay between requests.
- **DailyFaceoff** (dailyfaceoff.com) — line combos, confirmed goalies. Web scrape, 0.5s delay.
- **The Odds API** (the-odds-api.com) — moneylines, spreads, totals. Requires `ODDS_API_KEY` in `.env`.

## Key Patterns

- **Name matching** between data sources uses fuzzy matching (`difflib.SequenceMatcher` at 0.85 threshold) throughout `lines.py` and `main.py`. The `find_player_match()` and `fuzzy_match()` functions in `lines.py` are the canonical implementations.
- **Bias corrections** in `projections.py` are derived from backtest results. Constants live in `projections.py` itself (`GLOBAL_BIAS_CORRECTION`, `POSITION_BIAS_CORRECTION`, `GOALIE_BIAS_CORRECTION`). Update these when backtest metrics shift. Current values are from a 7-date batch backtest (1,410 skater + 77 goalie observations, including defensemen after API key bug fix). Run `python backtest.py --batch-backtest` to re-derive.
- **Position normalization**: LW/RW/R/L all map to `W`. C/W and W/C map to `C`. LD/RD map to `D`. This happens in `optimizer.py._normalize_position()` and `simulator.py._normalise_pos()`.
- **Goalie opponent exclusion**: The optimizer removes all skaters from the goalie's opponent team (negative correlation — if opponent scores, goalie loses points).
- **Stack correlation flow**: `StackBuilder` stores correlation values (PP1: 0.95, Line1: 0.85, Line1+D1: 0.75, Line2: 0.70). The optimizer's `_get_correlated_stack_players()` picks from these actual line groupings rather than arbitrary top-N.
- **Ownership normalization** targets ~900% total (9 roster spots × ~100% each). The `_normalize_ownership()` method in `ownership.py` scales raw predictions to hit this target.

## Known Gotchas

- **Salary merge column whitelist**: `merge_projections_with_salaries()` in `main.py` uses an explicit `merge_cols` list. Any column from DK salary data that downstream code needs (e.g., `game_info` for goalie-opponent exclusion) **must be manually added** to this list or it gets silently dropped during the pandas merge. This caused a bug where `_get_opponent_team()` always returned `None` because `Game Info` wasn't preserved.
- **Weight normalization for `np.random.choice`**: When building a probability array for team selection, the weights array must be normalized (`weights / weights.sum()`) because the pool may have fewer teams than the hardcoded weight list expects. Without this, `np.random.choice` raises `ValueError: probabilities do not sum to 1`.
- **Stale salary files**: The optimizer's `__main__` test block loads the latest file from `daily_salaries/`. If the most recent salary file is old, few players will match today's slate — the optimizer silently produces zero lineups when `min_teams` isn't met. Always verify the salary file date matches the target slate.
- **Fuzzy match false positives**: Name matching at 0.85 threshold can occasionally match wrong players (e.g., two players with similar surnames on different teams). Stack-building code should verify team membership after fuzzy matching when possible.

## Simulator Baseline & Lift

The simulator's `_compute_baselines()` calculates per-position baseline probability — the chance a random player at that position would appear in a lineup if selection were uniform. This accounts for the UTIL slot asymmetry (C/W only, D excluded) by splitting the UTIL share proportionally by pool size:

```
effective_C_slots = 2 + N_C / (N_C + N_W)
effective_W_slots = 3 + N_W / (N_C + N_W)
effective_D_slots = 2
effective_G_slots = 1
baseline_pct(pos) = effective_slots / pool_size * 100
lift = actual_pct / baseline_pct
```

A lift of 1.0x means random-level selection. Values well below 1.0x indicate a player is only appearing due to salary/position constraints rather than projection strength. The baseline header and lift column appear in both terminal output and CSV exports.

## Lift-Adjusted Re-Simulation (`--sim-lift`)

The `--sim-lift [blend]` flag runs a two-pass simulation:

1. **First pass**: Normal simulation (deterministic or MC per `--sim-iterations`). Produces lift values for each player.
2. **Second pass**: Re-runs the simulator with lift-adjusted projections, amplifying structural advantages (salary efficiency + position fit + projection strength).

**Formula**: `adjusted_fpts = projected_fpts * (1 + blend * (lift - 1.0))`

- `blend` defaults to **0.15** (configurable: `--sim-lift 0.25`)
- A player with 3.0x lift gets a 30% projection boost (at 0.15 blend)
- A player with 0.5x lift gets a 7.5% penalty
- A player with 1.0x lift is unchanged
- Players absent from first-pass results get lift = 0.0 (penalized — they never appeared)

The second-pass output shows both original and adjusted projection columns with a `[LIFT-ADJUSTED]` header tag. Both first-pass and lift-adjusted CSVs are exported to `daily_projections/`.

```bash
# Deterministic two-pass with default 0.15 blend
python main.py --simulate --sim-lift

# MC mode with custom blend
python main.py --simulate --sim-iterations 100 --sim-lift 0.25
```

## Projection Calibration (updated 2/3/26)

### Calibration Drift Bug Fix (2/3/26)

**Root Cause**: The previous `GLOBAL_BIAS_CORRECTION = 0.45` was derived from projection CSVs (Jan 23 - Feb 1) that were generated with **older, weaker** correction values (~0.92). When 0.45 was applied to the raw projection calculation, it caused **double-correction** — skater projections were reduced by ~78% too much.

**Evidence**: Feb 2 backtest showed skaters under-projected by 78% (mean_proj=4.70, mean_act=8.38). Date-by-date analysis revealed the progression:
- Jan 23-31: Mean proj ~7.8, ratio ~0.55 (old corrections)
- Feb 2: Mean proj ~4.2, ratio ~1.11 (new 0.45 applied → over-corrected)

**Fix**: Recalibrated using Feb 2 data (which has current corrections applied). New values: `GLOBAL = 0.45 × 1.78 = 0.80`.

### Current Bias Correction Values

| Parameter | Value | Notes |
|-----------|-------|-------|
| `GLOBAL_BIAS_CORRECTION` | 0.80 | Was 0.45 (over-corrected) |
| Centers (`C`) | 1.01 | Near neutral |
| Wings (`W`/`L`/`R`/`LW`/`RW`) | 0.99 | Near neutral |
| Defensemen (`D`) | 1.00 | Near neutral |
| `GOALIE_BIAS_CORRECTION` | 0.40 | Was 0.76 (under-corrected) |

### Post-Fix Verification (Feb 2 simulated)

| Position | Old Bias | New Bias |
|----------|----------|----------|
| C | -3.72 | +0.00 |
| W | -4.36 | -0.01 |
| D | -3.04 | +0.00 |
| G | +4.08 | -0.02 |
| **Skaters** | **-3.68** | **-0.00** |

### Projection Controls

The model has several layers to manage projection accuracy:

1. **Multiplicative adjustment cap** (`MAX_MULTIPLICATIVE_SWING = 0.15`): Seven adjustments (signal matchup, xG matchup, streak, PDO, opportunity, role, home ice) clamped to ±15%.

2. **High-projection mean regression**: Skater projections >14.0 FPTS blended 80/20 toward league mean (6.0). Goalie projections >12.0 blended toward 9.0.

3. **Goalie projection cap** (`GOALIE_PROJECTION_CAP = 16.0`): Hard ceiling.

4. **DK season average blending** (`DK_AVG_BLEND_WEIGHT = 0.80`): Projections blended 80% model / 20% DK's `AvgPointsPerGame` after salary merge.

5. **Symmetric signal clips**: `SIGNAL_COMPOSITE_CLIP_HIGH/LOW` at ±8%.

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

`backtest.py` filters out players with TOI=0 (scratches/DNPs) before computing error metrics. The `_parse_toi_minutes()` utility handles both `"MM:SS"` strings and numeric values.

## Environment Setup

Requires `.env` file in `projection/` with:
```
ODDS_API_KEY=<the-odds-api-key>
```

Key Python dependencies: `pandas`, `numpy`, `requests`, `tabpfn`, `scikit-learn`, `flask`, `python-dotenv`, `tqdm`, `scipy`.

## Ownership Regression Model

### Approach

The ownership model has three paths:

1. **Ridge regression (default)**: Ridge regression trained on ~1,200 historical contest observations from 6 matchable dates. Uses `StandardScaler` + `Ridge` with alpha tuned via leave-one-date-out cross-validation. Loaded from `backtests/ownership_model.pkl`.
2. **TabPFN (alternative)**: TabPFN v6.3.1 regressor (`TabPFNRegressor(ignore_pretraining_limits=True)`) trained on the same data. No hyperparameters to tune — just 6 LODOCV folds vs Ridge's 48 (8 alphas × 6 folds). Uses the same `StandardScaler` + feature pipeline as Ridge. Fully pickleable (inherits sklearn `BaseEstimator`).
3. **Heuristic (fallback)**: Original 12-factor multiplicative model using salary curve, PP1/Line1 boosts, goalie confirmation, value/projection ratios, Vegas totals, scarcity, recency, TOI surge. Used when no trained model pickle exists.

### Features (26 total)

| Category | Features |
|----------|----------|
| Core (from projection CSVs) | `salary`, `projected_fpts`, `dk_avg_fpts`, `floor`, `ceiling`, `edge`, `value` |
| Derived ranks | `salary_rank_in_pos`, `proj_rank_in_pos`, `value_rank_in_pos`, `salary_pctile`, `proj_pctile` |
| Derived ratios | `dk_value_ratio`, `salary_bin` |
| Position | `pos_C`, `pos_W`, `pos_D`, `pos_G`, `is_goalie` |
| Slate context | `slate_size`, `n_players_at_pos` |
| Lines (conditional) | `is_pp1`, `is_pp2`, `is_line1`, `is_d1`, `is_confirmed_goalie` |

Lines features are 0 when no lines JSON is available (3 of 6 training dates have lines data).

### Training Data

6 dates (~1,200 player-observations). SE (Single Entry) contests preferred for representative ownership:

| Date | Contest | Projection | Lines? |
|------|---------|-----------|--------|
| Jan 23 | `$5main_NHL1.23.26.csv` | `01_23_26...190750.csv` | No |
| Jan 26 | `$5SE_NHL1.26.26.csv` | `01_26_26...184134.csv` | No |
| Jan 28 | `$5SE_NHL1.28.26.csv` | `01_28_26...191024.csv` | No |
| Jan 29 | `$1SE_NHL_1.29.26.csv` | `01_29_26...184650.csv` | Yes |
| Jan 31 | `$5SE_NHL1.31.26.csv` | `01_31_26...190255.csv` | Yes |
| Feb 1 | `$5SE_NHL2.1.26.csv` | `02_01_26...140426.csv` | Yes |

### Retraining

```bash
# Run LODOCV for both Ridge and TabPFN, print side-by-side comparison
python backtest.py --ownership-backtest

# Train Ridge model (default) on all data and save pickle
python backtest.py --train-ownership

# Train TabPFN model instead and save pickle
python backtest.py --train-ownership --ownership-tabpfn
```

| Command | Effect |
|---------|--------|
| `--ownership-backtest` | Run LODOCV for both Ridge and TabPFN, print comparison |
| `--train-ownership` | Train Ridge (default), save pickle |
| `--train-ownership --ownership-tabpfn` | Train TabPFN, save pickle |

The trained model is saved to `backtests/ownership_model.pkl`. The pickle stores its `model_type` (`'ridge'` or `'tabpfn'`), so `predict_ownership()` loads whichever was last trained. Older pickles without `model_type` default to `'ridge'`. Delete the pickle to force the heuristic fallback path.

### Performance (LODOCV)

| Date | N | MAE | RMSE | Spearman |
|------|---|-----|------|----------|
| jan23 | 283 | 3.52 | 4.47 | 0.650 |
| jan26 | 133 | 4.22 | 6.43 | 0.651 |
| jan28 | 101 | 4.25 | 6.90 | 0.836 |
| jan29 | 428 | 4.83 | 5.45 | 0.653 |
| jan31 | 228 | 3.51 | 4.62 | 0.744 |
| feb01 | 99 | 4.62 | 7.64 | 0.818 |
| **Mean** | **1272** | **4.16** | **5.92** | **0.725** |

Best alpha: 100.0. Top features by coefficient magnitude: `slate_size` (-2.0), `proj_rank_in_pos` (+1.4), `proj_pctile` (+1.3), `n_players_at_pos` (-1.2), `salary_rank_in_pos` (+1.1), `dk_avg_fpts` (+1.1).

### Fallback Behavior

`predict_ownership()` in `ownership.py`:
1. Attempts to load `backtests/ownership_model.pkl`
2. If loaded: builds feature matrix → Ridge predict → clip to [0.1, 50.0]
3. If not loaded: runs `_heuristic_predict()` (original 12-factor model)
4. Then normalizes ownership to ~900% total, computes leverage scores and tiers
