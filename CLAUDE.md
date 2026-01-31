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
- **projections.py** — `NHLProjectionModel.generate_projections()` calculates expected fantasy points with position-specific bias corrections (from backtest). Optional TabPFN ML model.
- **lines.py** — `LinesScraper` scrapes DailyFaceoff for line combos/PP units/confirmed goalies. `StackBuilder` builds correlation data and `get_best_stacks()` returns PP1/Line1/Line1+D1/Line2 groupings with projection matching.
- **ownership.py** — `OwnershipModel.predict_ownership()` uses salary curve, PP1/Line1 role boosts, Vegas totals, goalie confirmation, recent scoring, and TOI surge signals. Normalizes to ~900% total.
- **optimizer.py** — `NHLLineupOptimizer` builds lineups under DK constraints ($50k cap, 2C/3W/2D/1G/1UTIL). GPP mode uses `_get_correlated_stack_players()` to select from actual line/PP combos (PP1 > Line1+D1 > Line1 for primary, Line1 > Line2 > PP1 for secondary) with fallback to top-N-by-projection.
- **contest_roi.py** — Leverage recommendations and contest EV scoring based on payout structure.
- **backtest.py** — Compares projections to actual scores. Outputs MAE/RMSE/correlation by position. Results feed bias corrections in `config.py`.
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
- **Bias corrections** in `projections.py` are derived from backtest results and stored in `config.py` (`POSITION_BIAS_CORRECTION`, `GOALIE_BIAS_CORRECTION`, `GLOBAL_BIAS_CORRECTION`). Update these when backtest metrics shift.
- **Position normalization**: LW/RW/R/L all map to `W`. C/W and W/C map to `C`. LD/RD map to `D`. This happens in `optimizer.py._normalize_position()`.
- **Goalie opponent exclusion**: The optimizer removes all skaters from the goalie's opponent team (negative correlation — if opponent scores, goalie loses points).
- **Stack correlation flow**: `StackBuilder` stores correlation values (PP1: 0.95, Line1: 0.85, Line1+D1: 0.75, Line2: 0.70). The optimizer's `_get_correlated_stack_players()` picks from these actual line groupings rather than arbitrary top-N.
- **Ownership normalization** targets ~900% total (9 roster spots × ~100% each). The `_normalize_ownership()` method in `ownership.py` scales raw predictions to hit this target.

## Environment Setup

Requires `.env` file in `projection/` with:
```
ODDS_API_KEY=<the-odds-api-key>
```

Key Python dependencies: `pandas`, `numpy`, `requests`, `tabpfn`, `scikit-learn`, `flask`, `python-dotenv`, `tqdm`.
