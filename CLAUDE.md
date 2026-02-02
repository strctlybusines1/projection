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

## Projection Calibration (updated 2/2/26)

### Over-Projection Controls

The model has several layers to combat systematic over-projection, added based on 7-date batch backtest analysis:

1. **Multiplicative adjustment cap** (`MAX_MULTIPLICATIVE_SWING = 0.15` in `projections.py`): Seven multiplicative adjustments (signal matchup, xG matchup, streak, PDO, opportunity, role, home ice) are collected into a single combined multiplier and clamped to ±15% before applying. Without this, compounding inflated high projections by ~25%.

2. **High-projection mean regression** (`projections.py`): Skater projections above 14.0 FPTS are blended 80/20 toward league mean (6.0). Goalie projections above 12.0 are blended 80/20 toward goalie mean (9.0). This targets the worst-calibrated bucket (15+ projected had the highest bias).

3. **Goalie projection cap** (`GOALIE_PROJECTION_CAP = 16.0`): Hard ceiling on goalie projections. Goalies projected 12+ had +3.79 average bias.

4. **DK season average blending** (`DK_AVG_BLEND_WEIGHT = 0.80` in `config.py`): After salary merge in `main.py`, projections are blended 80% model / 20% DK's `AvgPointsPerGame`. This anchors toward market consensus and reduces outlier projections. Floor/ceiling/edge/value are recalculated after blending.

5. **Symmetric signal clips** (`config.py`): `SIGNAL_COMPOSITE_CLIP_HIGH` reduced from 1.10 to 1.08, matching the low clip at 0.92 for symmetric ±8%.

### Current Bias Correction Values

| Parameter | Value | Notes |
|-----------|-------|-------|
| `GLOBAL_BIAS_CORRECTION` | 0.45 | Applied to all skaters (was 0.92) |
| Centers (`C`) | 1.03 | actual/proj ratio 0.557 |
| Wings (`W`/`L`/`R`/`LW`/`RW`) | 0.93 | Highest bias position (+4.08) |
| Defensemen (`D`) | 1.04 | Now validated with actual D data |
| `GOALIE_BIAS_CORRECTION` | 0.76 | Was 0.88 |

### Backtest Results (7-date batch, pre-recalibration)

| Date | N Skaters | N Goalies |
|------|-----------|-----------|
| Jan 23 | 269 | 15 |
| Jan 26 | 135 | 7 |
| Jan 28 | 100 | 6 |
| Jan 29 | 507 | 30 |
| Jan 30 | 34 | 2 |
| Jan 31 | 263 | 11 |
| Feb 1 | 102 | 6 |
| **Total** | **1,410** | **77** |

Aggregate (pre-recalibration): Skater MAE=5.47, bias=+3.63, RMSE=6.69. Goalie MAE=6.70, bias=+1.52.

Per-position (pre-recalibration):
| Position | N | MAE | Bias | Mean Proj | Mean Actual | Ratio |
|----------|---|-----|------|-----------|-------------|-------|
| C | 475 | 5.71 | +3.56 | 8.03 | 4.48 | 0.557 |
| W | 469 | 6.03 | +4.08 | 8.43 | 4.34 | 0.515 |
| D | 466 | 4.67 | +3.25 | 7.49 | 4.25 | 0.567 |
| G | 77 | 6.70 | +1.52 | 11.43 | 9.91 | 0.867 |

Key observations:
- Defensemen now included in backtests (466 observations) after fixing `"defense"` vs `"defensemen"` API key bug in `fetch_skaters_actuals_for_date()`
- D have the lowest MAE (4.67) and lowest bias (+3.25) of skater positions
- Wings have the highest bias (+4.08, ratio 0.515)
- Goalie bias reduced from prior +2.30 to +1.52 with 77 observations
- Batch backtest available via `python backtest.py --batch-backtest`

### Post-Recalibration Spot Check (Feb 1)

After applying the updated bias corrections, `python backtest.py --skater-slate-date 2026-02-01` produced:
- **Skater MAE: 3.78** (down from 5.47 aggregate pre-recalibration)
- **Skater bias: -1.76** (flipped from +3.63 over-projection to slight under-projection)
- **104 skaters matched** (35 D, 39 W, 30 C — defensemen fully represented)

The corrections shifted from systematic over-projection to slight under-projection, which is preferable for DFS (conservative projections avoid chasing inflated ceilings). The position column is now included in `run_slate_skater_backtest()` details output for per-position analysis on any single-date backtest.

### Backtest TOI Filtering

`backtest.py` now filters out players with TOI=0 (scratches/DNPs) before computing error metrics. The `_parse_toi_minutes()` utility handles both `"MM:SS"` strings and numeric values from the NHL API boxscore data. Both `fetch_skaters_actuals_for_date()` and `fetch_goalies_actuals_for_date()` now include a `toi_minutes` column.

## Environment Setup

Requires `.env` file in `projection/` with:
```
ODDS_API_KEY=<the-odds-api-key>
```

Key Python dependencies: `pandas`, `numpy`, `requests`, `tabpfn`, `scikit-learn`, `flask`, `python-dotenv`, `tqdm`.
