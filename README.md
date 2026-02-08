# NHL DFS Projection System

DraftKings NHL daily fantasy projection and lineup optimization pipeline.

## Quick Start

```bash
cd ~/Desktop/Code/projection

# Standard run (projections + 5 GPP lineups)
python main.py --stacks --show-injuries --lineups 5 --edge

# Single-entry mode (40 candidates → best pick)
python main.py --stacks --show-injuries --lineups 40 --edge --single-entry

# First run of day (refresh Edge cache)
python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge

# Pre-flight validation only (no projections)
python main.py --validate-only

# Full run with validation report
python main.py --validate --stacks --show-injuries --lineups 5 --edge
```

## Pre-Slate Checklist

1. Download DK salary CSV → `daily_salaries/DKSalaries_M.DD.YY.csv`
2. `python main.py --validate-only` — verify files, schedule, Vegas, NST
3. `python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge`
4. Review projections, adjust lineups, enter on DraftKings
5. After lock: `python main.py --stacks --lineups 5 --edge` (re-run with late scratches)

## Architecture

```
Data Sources                Pipeline              Output
─────────────              ────────              ──────
NHL API (stats/schedule)
MoneyPuck (injuries)        data_pipeline.py
Natural Stat Trick (xG)  →  features.py    →  projections.py → optimizer.py → Lineups
DailyFaceoff (lines)        goalie_model.py    ownership.py     single_entry.py
The Odds API (Vegas)
Edge Stats (NHL tracking)
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | CLI entry point, orchestrates full pipeline |
| `data_pipeline.py` | Fetches and merges all data sources |
| `features.py` | Feature engineering (per-game rates, bonuses, matchups) |
| `projections.py` | Fantasy point projections with bias correction |
| `goalie_model.py` | Danger-zone goalie model (XGBoost + shot quality) |
| `optimizer.py` | Salary-cap lineup optimization with stacking |
| `ownership.py` | Ownership % prediction (XGBoost regression) |
| `single_entry.py` | Single-entry contest lineup selection |
| `lines.py` | Line combo and PP unit scraping (DailyFaceoff) |
| `edge_stats.py` | NHL tracking data (Edge stats) integration |
| `edge_cache.py` | Daily caching layer for Edge API calls |
| `config.py` | DK scoring rules, constants, signal weights |
| `validate.py` | Pre-flight validation (9 checks, GO/NO-GO) |
| `history_db.py` | Historical SQLite database for accuracy tracking |
| `contest_roi.py` | Contest EV and leverage recommendations |
| `contest_analysis.py` | Post-contest strategy analysis |

### Supporting Tools

| Tool | Purpose |
|------|---------|
| `run_daily.sh` | Shell shortcuts for common commands |
| `backtest.py` | Full pipeline backtesting |
| `goalie_backtest.py` | Danger-zone goalie model backtest |
| `blend_backtest.py` | L10/L25 rolling window blend test |
| `season_signal_backtest.py` | Signal persistence and noise analysis |

## Key Directories

```
daily_salaries/     DraftKings salary CSVs (per slate)
daily_projections/  Output projections and lineup CSVs
vegas/              Vegas lines CSV fallback files
cache/              Edge stats daily cache (auto-managed)
backtests/          Backtest results, model artifacts
contests/           DraftKings contest export CSVs
data/               SQLite historical database
tests/              Pytest suite
```

## Testing

```bash
pytest tests/ -v                        # Full suite
pytest tests/ -v -k "test_scoring"      # Just scoring math
pytest tests/ -v -k "TestOptimizer"     # Just optimizer
pytest tests/ -v -k "TestValidation"    # Just validation checks
```

## Historical Database

```bash
python history_db.py backfill                     # Ingest all existing data
python history_db.py report                       # Slate accuracy table
python history_db.py report --last 5              # Last 5 slates
python history_db.py report --player "McDavid"    # Player history
python history_db.py report --positions            # Accuracy by position
python history_db.py export --output history.csv  # Full CSV export
```

## DraftKings Scoring

**Skaters:** Goal 8.5 | Assist 5.0 | SOG 1.5 | Block 1.3 | SH Point +2.0
**Bonuses:** Hat trick 3.0 | 5+ SOG 3.0 | 3+ Blocks 3.0 | 3+ Points 3.0
**Goalies:** Win 6.0 | Save 0.7 | GA -3.5 | Shutout 4.0 | OTL 2.0 | 35+ Saves 3.0

## Model Performance (as of Feb 2026)

| Metric | Value |
|--------|-------|
| Skater MAE | ~4.3 FPTS |
| Goalie MAE | ~3.7 FPTS |
| Skater Correlation | ~0.54 |
| Ownership MAE | ~2.2% |
| Improvement over naive (L5 avg) | ~7-10% |

## Git Backup

```bash
# Quick commit and push
./git_backup.sh

# Or manually
git add -A && git commit -m "slate update $(date +%m/%d)" && git push
```
