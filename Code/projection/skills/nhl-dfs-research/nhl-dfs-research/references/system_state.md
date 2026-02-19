# NHL DFS Projection System — Current State

Last updated: 2026-02-18

## Architecture Overview

```
Daily Workflow:
  DK Salaries CSV + NHL API + MoneyPuck + Vegas Lines
    → Feature Engineering (features.py)
    → Projection Models (MDN v3, Transformer v1, Ensemble)
    → Goalie Model (GBM + FC blend)
    → Ownership Prediction (Ridge v2)
    → Lineup Optimization (MILP + Stacking)
    → Monte Carlo Simulation
    → Final Lineups (5-20 for DraftKings)
```

## Models & Performance

### Skater Projection (core model)
| Model | MAE (FPTS) | Notes |
|-------|-----------|-------|
| MDN v3 | 4.091 | Mixture Density Network, best single model |
| Transformer v1 | 4.096 | Custom transformer architecture |
| Ensemble (60/40 MDN/Trans) | 4.066 | Current default, best overall |
| FC baseline (features.py) | ~4.2 | Simple feature-to-FPTS linear model |

The ensemble helps most on high-usage players (star players). MDN is stronger in recent windows (Jan-Feb), transformer adds value over full season.

### Expected Goals (xG)
| Metric | Value |
|--------|-------|
| Model | GradientBoostingClassifier (sklearn) |
| AUC (test) | 0.7513 |
| Log Loss | 0.2441 |
| Training data | ~15K shots (175 games, 2025-26 partial season) |
| Top features | distance (20.4%), angle (10%), prior_distance (9.8%) |

Built from NHL API play-by-play, following Evolving Hockey methodology. Features: distance, angle, shot type (7 types), strength state, score differential, prior event type/distance/time, is_rebound, is_rush, period, game_seconds, is_home.

**Known limitation**: Only trained on partial 2025-26 season. Full 6-season dataset (~600K+ shots) is being scraped and will allow retraining.

### Goalie Projection
| Metric | Value |
|--------|-------|
| Model | GBM (75%) + DK avg baseline (25%) blend |
| MAE | ~3.7 FPTS (no FC dependency) |
| Features | MoneyPuck season stats, DK salary, opponent quality |

### Ownership Prediction
| Metric | Value |
|--------|-------|
| Model | Ridge Regression (ownership_v2.py) |
| MAE | 1.92% |
| Correlation | 0.905 |
| Features | Salary, projected FPTS, value, positional ranks |

### Lineup Optimization
| Metric | Value |
|--------|-------|
| Optimizer | MILP (PuLP) |
| Stacking | Line stacks (2-3 players), goalie-skater correlation |
| Simulation | Monte Carlo (10K iterations) |
| Backtest cash rate | ~90% (19/21 slates) |
| Backtest "win" rate | ~38% (top lineup scores > median GPP payout) |

## Data Sources

### Available & Active
- **NHL API** (`api-web.nhle.com`): Play-by-play (shots, goals, penalties, faceoffs with x/y coords), boxscores (per-player stats), schedules, rosters
- **MoneyPuck**: Season-level skater stats (xG, CF%, HDCF%, etc.), team stats, goalie stats — loaded into database tables `mp_skaters`, `mp_teams`, `mp_goalies`
- **DraftKings**: Daily salary CSVs with player names, positions, salaries, game info
- **DailyFaceoff**: Line combinations and starting goalies (scraped via lines.py)
- **Vegas Lines**: Game totals and spreads (manual/cached)
- **NHL EDGE Tracking**: Speed, acceleration, OZ time — accessed via edge_stats.py with calibrated boosts

### Blocked / Unavailable
- **Natural Stat Trick (NST)**: IP blocked (207.5.25.109). Was primary source for on-ice stats. Replaced by custom PBP-derived stats.

## Database Schema (SQLite: data/nhl_dfs_history.db)

### Core Tables
- `boxscore_skaters` — Per-game boxscore stats from NHL API
- `historical_skaters` — Multi-season historical data
- `dk_salaries` — DraftKings salary history
- `actuals` — Actual DFS points scored per slate
- `contest_results` — DraftKings contest results for ROI tracking

### MoneyPuck Tables
- `mp_skaters` — Season-level skater advanced stats
- `mp_teams` — Team-level stats
- `mp_goalies` — Goalie season stats

### PBP-Derived Tables
- `pbp_shots` — Individual shot events with xG, coordinates, features (~207K+ rows, growing)
- `pbp_games` — Game metadata (teams, date, season)
- `adv_player_games` — Per-player per-game advanced stats with rolling features
- `adv_team_games` — Per-team per-game stats with rolling features

## Key Files

### Models
- `mdn_v3.py` — Mixture Density Network (best single model)
- `transformer_v1.py` — Transformer architecture
- `lineup_builder.py` — Main pipeline: ensemble projections → goalie model → optimizer → simulation
- `ownership_v2.py` — Ownership prediction model
- `goalie_model.py` / `goalie_v2.py` — Goalie projections

### Data & Features
- `data_pipeline.py` — NHL API data fetching, MoneyPuck integration
- `features.py` — Feature engineering for projection models
- `nhl_pbp_scraper.py` — Play-by-play scraper (NHL API)
- `advanced_stats.py` — Custom advanced stats from PBP (ixG, xGF%, HD rates)
- `edge_stats.py` — NHL EDGE tracking data integration

### Optimization & Simulation
- `optimizer.py` — MILP lineup optimizer
- `simulator.py` — Monte Carlo simulation engine
- `lines.py` — DailyFaceoff line scraper + stack building

### Analysis
- `backtest.py` — Historical accuracy validation
- `contest_analysis.py` — Post-slate ROI analysis

## DraftKings Constraints
- Salary cap: $50,000
- Roster: 2C, 3W, 2D, 1G, 1UTIL (C/W only for UTIL)
- Position mapping: LW/RW → W, LD/RD → D

## Known Gaps & Opportunities

1. **xG model undertrained** — Only 15K shots from partial season. Full 6-season retrain expected to improve AUC significantly.
2. **No line combination features** — We don't model who plays with whom, which is arguably the biggest factor in hockey DFS.
3. **No pre-shot passing data** — xG model uses only the shot itself, not the passing sequence leading to it.
4. **Simple ensemble** — Just weighted average of MDN + Transformer. No stacking, no learned combination.
5. **Ownership model is linear** — Ridge regression works well but may miss non-linear patterns.
6. **No score-state modeling** — Don't model how player usage changes based on game script.
7. **No travel/rest features** — Back-to-back games, travel distance not used.
8. **Correlation estimation is crude** — Line stacking uses heuristic correlations, not estimated from data.
9. **FC projections removed** — All FantasyCruncher projection dependencies have been eliminated. System uses own ensemble (MDN+Transformer) projections throughout. Historical data in own.csv/dk_salaries still contains FC Proj column but it's only used as fallback when our own projections aren't available for older dates.
