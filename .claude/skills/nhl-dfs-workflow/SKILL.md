---
name: nhl-dfs-workflow
description: NHL Daily Fantasy Sports projection and lineup optimization workflow. Use this skill when working on the NHL DFS projection system including generating player projections, applying EDGE stats boosts, predicting ownership, building optimized lineups, running backtests, organizing project structure, or cleaning up files. Triggers on NHL DFS, fantasy hockey, DraftKings NHL, projection model, lineup optimization, or project cleanup tasks.
---

# NHL DFS Workflow

## Project Structure

```
projection/
├── main.py                 # CLI entry - orchestrates pipeline
├── data_pipeline.py        # NHL API, MoneyPuck, Natural Stat Trick
├── features.py             # Feature engineering
├── projections.py          # FPTS calculation + bias corrections
├── edge_stats.py           # NHL Edge tracking + boosts
├── lines.py                # DailyFaceoff scraper + stacks
├── ownership.py            # Ownership prediction (Ridge)
├── optimizer.py            # Lineup optimization
├── simulator.py            # Monte Carlo simulation
├── backtest.py             # Accuracy validation
├── config.py               # Configuration
│
├── daily_salaries/         # DK salary CSVs
├── daily_projections/      # Output CSVs
├── backtests/              # Results + pickles
├── contests/               # DK contest results
└── vegas/                  # Vegas lines fallback
```

## Daily Workflow

### Pre-Slate (2-3 hours before lock)

```bash
# 1. Download DK salary CSV → daily_salaries/

# 2. Generate projections (filter injured)
python main.py --stacks --show-injuries --filter-injuries --lineups 5

# 3. Apply EDGE boosts (separate step)
python main.py --stacks --show-injuries --filter-injuries --lineups 5 --edge
```

### Pre-Lock (30 mins before)

```bash
python lines.py  # Confirm goalies

# Manual: Verify positions vs DK, check scratches, prepare pivots
```

### Post-Slate

```bash
# 1. Create actuals CSV (name, actual, own, TOI)
# 2. Run full backtest
python backtest.py
```

## Commands Reference

| Command | Purpose |
|---------|---------|
| `python main.py --stacks --show-injuries --lineups 5` | Base projections |
| `python main.py ... --edge` | Add EDGE boosts |
| `python main.py ... --filter-injuries` | Remove injured |
| `python backtest.py` | Full backtest |
| `python backtest.py --edge-backtest` | Validate EDGE |
| `python lines.py` | Check lines/goalies |

## EDGE Boosts (Calibrated)

| Metric | Elite (≥90th) | Above-Avg (≥65th) | Correlation |
|--------|---------------|-------------------|-------------|
| OZ Time | +10% | +4% | r=0.18 |
| Bursts | +5% | - | r=0.15 |
| Speed | +2% | +1% | r=0.07 |

**Two-step workflow if API issues:**
```bash
python main.py --stacks --show-injuries --lineups 5          # Step 1
python main.py --stacks --show-injuries --lineups 5 --edge   # Step 2
```

## DraftKings Constraints

- Cap: $50,000
- Roster: 2C, 3W, 2D, 1G, 1UTIL (C/W only)
- Positions: LW/RW → W, LD/RD → D

## File Naming

| Type | Pattern |
|------|---------|
| Salaries | `DKSalaries_M_D_YY.csv` |
| Projections | `{date}_projections_{timestamp}.csv` |
| Actuals | `{date}_actual.csv` |

## Project Cleanup

To reorganize the project, see `references/cleanup_checklist.md`.

Standard cleanup tasks:
1. Archive old salary files (>7 days)
2. Archive old projection files (>7 days)
3. Consolidate backtest results
4. Remove duplicate/temp files
5. Verify .env and dependencies
