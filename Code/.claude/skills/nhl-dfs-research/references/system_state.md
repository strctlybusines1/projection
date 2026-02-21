# NHL DFS Projection System — Current State

Last updated: 2026-02-21

## Architecture Overview

### Daily Pipeline (Production)

```
DK Salaries CSV + NHL API + MoneyPuck + Vegas Lines + DailyFaceoff
  → data_pipeline.py (fetch stats from NHL API, MoneyPuck, NST)
  → features.py (per-game rates, bonuses, matchup adjustments)
  → projections.py (base FPTS calculation + bias corrections)
  → edge_stats.py (NHL Edge tracking boosts, if --edge flag)
  → merge with DK salaries (fuzzy name matching, position normalization)
  → ownership.py (Ridge regression or heuristic fallback)
  → optimizer.py (greedy heuristic) or optimizer_ilp.py (ILP/PuLP)
  → single_entry.py (if --single-entry, scores candidates on 6 dimensions)
  → Export to daily_projections/*.csv
```

Entry point: `main.py` with CLI flags (`--stacks`, `--edge`, `--lineups N`, `--single-entry`, etc.)

### Experimental Models (Not in Daily Pipeline)

| Model | MAE (FPTS) | Notes |
|-------|-----------|-------|
| MDN v3 | 4.091 | Mixture Density Network, best single model |
| Transformer v1 | 4.096 | Custom transformer architecture |
| Ensemble (60/40 MDN/Trans) | 4.066 | Best overall, not yet integrated |

These live in `archive/experimental/`. The daily pipeline uses `projections.py` with bias corrections instead.

## Daily Pipeline Performance

### Skater Projections
| Parameter | Value |
|-----------|-------|
| Method | Feature-based with bias corrections |
| GLOBAL_BIAS_CORRECTION | 0.80 |
| Position corrections | C: 1.01, W: 0.99, D: 1.00 |
| GOALIE_BIAS_CORRECTION | 0.40 |
| Backtest MAE | ~4.07 FPTS |
| Tuning | All constants in `tuning_params.json` |

### Edge Stats Boosts (Calibrated Feb 2026)

**Skater boosts** (from 1,180-observation backtest, Jan 22 - Feb 2, 2026):

| Metric | Elite (≥90th) | Above-Avg (≥65th) | Backtest Correlation |
|--------|---------------|-------------------|---------------------|
| OZ Time | +10% | +4% | r=+0.18 (strongest) |
| Bursts | +5% | - | r=+0.15 |
| Speed | +2% | +1% | r=+0.07 (weakest) |

Max combined boost: ~17%.

**Goalie boosts** (calibrated Feb 4, 2026):

| Metric | Elite Threshold | Boost | Penalty Threshold | Penalty |
|--------|-----------------|-------|-------------------|---------|
| EV Save % | ≥92.0% | +8% | <89.0% | -6% |
| EV Save % | ≥90.5% | +4% | - | - |
| Quality Starts % | ≥60% | +6% | <40% | -4% |
| Quality Starts % | ≥50% | +3% | - | - |

Max combined: +14.5% boost, -10% penalty.

### Expected Goals (xG)
| Metric | Value |
|--------|-------|
| Model | GradientBoostingClassifier (sklearn) |
| AUC (test) | 0.7513 |
| Log Loss | 0.2441 |
| Training data | ~15K shots (175 games, 2025-26 partial season) |
| Top features | distance (20.4%), angle (10%), prior_distance (9.8%) |

Built from NHL API play-by-play, following Evolving Hockey methodology. Features: distance, angle, shot type, strength state, score differential, prior event type/distance/time, is_rebound, is_rush, period, game_seconds, is_home.

**Known limitation**: Only trained on partial 2025-26 season. Full 6-season dataset would improve AUC significantly.

### Ownership Prediction
| Metric | Value |
|--------|-------|
| Model | Ridge Regression (loaded from `backtests/ownership_model.pkl`) |
| Alternative | TabPFN v6.3.1 regressor |
| Fallback | 12-factor heuristic |
| LODOCV MAE | 4.16 |
| LODOCV RMSE | 5.92 |
| LODOCV Spearman | 0.725 |
| Training data | ~1,200 historical contest observations |

### Lineup Optimization
| Metric | Value |
|--------|-------|
| Primary optimizer | Greedy heuristic (optimizer.py) |
| Alternative | ILP/PuLP (optimizer_ilp.py, via `--optimizer ilp`) |
| Stacking | PP1 (0.95), Line1 (0.85), Line1+D1 (0.75), Line2 (0.70) |
| Simulation | Basic MC (simulator.py) or correlated MC (simulation_engine.py) |
| SE scoring | projection 35%, ceiling 15%, goalie 15%, stacks 15%, salary 10%, leverage 10% |

## Key Modules

| Module | Purpose |
|--------|---------|
| **main.py** | CLI entry point, orchestrates full pipeline |
| **utils.py** | Shared utilities: position normalization, fuzzy matching, DK scoring |
| **components.py** | Component factory: swappable optimizer/simulator/ownership |
| **tuning.py** | Tuning parameter loader from `tuning_params.json` |
| **config.py** | Static configuration: DK scoring rules, API URLs, team mappings |
| **data_pipeline.py** | NHL API, MoneyPuck, Natural Stat Trick data fetching |
| **features.py** | Per-game rates, bonus probabilities, opponent adjustments |
| **projections.py** | Expected FPTS with bias corrections |
| **edge_stats.py** | NHL Edge tracking data + projection boosts |
| **edge_cache.py** | Daily Edge stats caching |
| **lines.py** | DailyFaceoff scraper + stack correlation builder |
| **ownership.py** | Ridge/TabPFN/heuristic ownership prediction |
| **optimizer.py** | Greedy heuristic lineup optimizer |
| **optimizer_ilp.py** | ILP (PuLP) optimizer |
| **simulator.py** | Deterministic + independent MC simulator |
| **simulation_engine.py** | Correlated MC with zero-inflated lognormal |
| **single_entry.py** | SE contest candidate scorer |
| **backtest.py** | Projection accuracy validation (MAE/RMSE/correlation) |

## Data Sources

### Available & Active
- **NHL API** (`api-web.nhle.com`): Play-by-play, boxscores, schedules, rosters (0.3s rate limit)
- **MoneyPuck** (`moneypuck.com`): Season-level skater/team/goalie advanced stats
- **DraftKings**: Daily salary CSVs with names, positions, salaries, game info
- **DailyFaceoff**: Line combinations and starting goalies (scraped, 0.5s delay)
- **The Odds API**: Vegas lines — game totals and spreads (requires API key in `.env`)
- **NHL Edge** (via `nhl-api-py`): Tracking data — speed, acceleration, OZ time (0.3s delay)

### Blocked / Unavailable
- **Natural Stat Trick (NST)**: IP blocked. Was primary source for on-ice stats. Replaced by custom PBP-derived stats.

## DraftKings Constraints
- Salary cap: $50,000
- Roster: 2C, 3W, 2D, 1G, 1UTIL (C/W only for UTIL, D excluded)
- Position mapping: LW/RW → W, LD/RD → D

## Known Gaps & Opportunities

### High Priority (from Feb 18 research report)
1. **No line combination features** — Linemate quality is the biggest predictor variance in hockey DFS. DailyFaceoff data already scraped but not used as features.
2. **No score-state deployment modeling** — Player usage changes with game script. Vegas spread available as proxy.
3. **Simple ensemble** — Experimental models use fixed 60/40 weighted average. Stacking with meta-learner would improve.
4. **Heuristic correlations** — Optimizer uses hand-coded stack correlations. Ledoit-Wolf shrinkage on historical actuals would be data-driven.
5. **Linear ownership model** — Ridge regression may miss non-linear patterns. XGBoost recommended.

### Medium Priority
6. **No rest/travel features** — Back-to-back games reduce output 3-8%. Not currently modeled.
7. **xG model undertrained** — Only 15K shots from partial season. Full 6-season retrain needed.
8. **No conformal prediction** — Point estimates only. Calibrated intervals would improve simulation.
9. **No Kelly criterion** — Lineup portfolio equally weighted. Kelly would optimize allocation.

### Long-Term
10. **Graph neural networks** — Model player interactions as graph. Needs full spatiotemporal PBP data.
11. **Skill-adjusted xG** — Account for shooter/goalie skill in xG. Needs 6-season scrape.
12. **Late-swap optimization** — Re-optimize 15 mins before lock with updated info.
13. **Multi-slate optimization** — Joint lineup optimization across Main/Showdown/Secondary.

## Research Backlog

Recommendations from the Feb 18 research report, organized by implementation status:

### Not Yet Started
- [ ] Line combination features in features.py (Priority 1)
- [ ] Score-state deployment features (Priority 2)
- [ ] Back-to-back / rest features (Priority 3)
- [ ] Stacking ensemble with MLP meta-learner
- [ ] Ledoit-Wolf correlation matrix for optimizer
- [ ] XGBoost ownership model to replace Ridge
- [ ] Conformal prediction intervals for simulation
- [ ] Kelly criterion lineup portfolio sizing
- [ ] Temporal weighting for ensemble (recency bias)
- [ ] Extract full MDN mixture components

### Completed / In Progress
- [x] Edge stats integration (skater + goalie boosts, calibrated Feb 2026)
- [x] ILP optimizer alternative (optimizer_ilp.py)
- [x] Correlated MC simulation (simulation_engine.py)
- [x] Single-entry scoring engine (single_entry.py)
- [x] Component factory for swappable implementations (components.py)
- [x] Tuning parameter centralization (tuning_params.json)
- [x] Edge caching for performance (edge_cache.py)

### Key References from Feb 18 Report
- Conformal prediction: arxiv.org/abs/2107.07511 + github.com/aangelopoulos/conformal-prediction
- GNN for sports: arxiv.org/html/2207.14124
- Skill-adjusted xG: arxiv.org/html/2511.07703
- Stacking ensembles: Nature Scientific Reports 2025 NBA study
- Evolving Hockey xG: github.com/evolvingwild/hockey-all
- PyTorch Forecasting (Temporal Fusion Transformer): github.com/sktime/pytorch-forecasting
