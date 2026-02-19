# Backtest Engine Documentation

## Overview

`backtest_full.py` is a comprehensive NHL DFS backtest engine that evaluates lineup strategies against all historical dates (Oct 7, 2025 - Feb 5, 2026) with complete DK salary + game log data.

## What It Does

1. **Loads Data:** Queries SQLite database for DK salaries and actual game logs
2. **Normalizes Teams:** Converts DK abbreviations (NJ, LA, SJ, TB) to NHL standard (NJD, LAK, SJS, TBL)
3. **Detects Scratches:** Identifies players listed in DK pool but not in game logs
4. **Scores Lineups:** Matches player names using exact + fuzzy (70% threshold) matching
5. **Runs Strategies:** Compares two strategies:
   - **Salary-Rank:** Uses salary as proxy for value (projected_fpts = salary / 1000)
   - **SE Optimizer:** Uses daily projection CSVs with actual player projections
6. **Computes Stats:** Cash rates, win rates, confidence intervals, per-date details
7. **Exports Results:** Saves CSV files to backtests/ directory

## Usage

### Basic Commands

```bash
# Run both strategies on all 113 dates
python backtest_full.py --method both

# Test salary-rank baseline only
python backtest_full.py --method salary

# Test SE optimizer only
python backtest_full.py --method se

# Suppress verbose output
python backtest_full.py --method both --quiet
```

### Example Output

```
================================================================================
  COMPREHENSIVE NHL DFS BACKTEST ENGINE
  Seed: 2026 | Methods: salary, se | Dates: 113
  Date Range: 2025-10-07 to 2026-02-05
================================================================================

[  1/113] 2025-10-07 | DK: 212 players, 6 teams | Actuals: 108 players, 8 teams | Cash:   81.8
    SALARY:   27.4 actual (7/9 matched, 2 scratched) proj=49.9 → miss [0.02s]
    SE    : NO LINEUP

[  2/113] 2025-10-08 | DK: 199 players, 8 teams | Actuals: 148 players, 9 teams | Cash:  134.3
    SALARY:   97.4 actual (8/9 matched, 1 scratched) proj=49.9 → miss [0.16s]
    SE    : NO LINEUP
```

## Output Files

### Per-Date CSVs

Located in `/sessions/youthful-funny-faraday/mnt/Code/projection/backtests/`

```
backtest_salary_20260219_124307.csv    # 106 valid lineups
backtest_se_20260219_124307.csv         # 10 valid lineups
```

### CSV Columns

| Column | Description |
|--------|-------------|
| `date` | Slate date (YYYY-MM-DD) |
| `actual` | Total DK FPTS scored by lineup |
| `projected` | Sum of projected_fpts for lineup |
| `matched` | Number of players matched to actuals (0-9) |
| `scratched` | Number of players not found in actuals |
| `is_cash` | True if actual >= cash_line |
| `is_win` | True if actual >= winning_score |
| `salary` | Total salary of lineup |
| `valid` | True if lineup was successfully built |
| `n_dk_players` | Size of DK player pool for date |
| `n_actual_players` | Size of actual game logs for date |
| `n_teams_on_slate` | Number of teams on DK slate |
| `n_teams_with_actuals` | Number of teams with game log data |
| `cash_line` | Minimum score to cash |
| `winning_score` | Highest score in contest (if available) |

### Example CSV Output

```csv
date,actual,projected,matched,scratched,is_cash,is_win,salary,valid,n_dk_players,n_actual_players,n_teams_on_slate,n_teams_with_actuals,cash_line,winning_score
2025-10-07,27.4,49.9,7,2,False,False,49900,True,212,108,6,8,81.8,120.6
2025-10-08,97.4,49.9,8,1,False,False,49900,True,199,148,8,9,134.3,166.8
2025-10-09,86.5,47.5,9,0,False,False,47500,True,724,508,28,30,107.7,177.2
```

## Data Requirements

### Database: `/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db`

**Required Tables:**

1. `dk_salaries` - DK salary pool
   - `slate_date` (TEXT)
   - `player_name` (TEXT)
   - `team` (TEXT)
   - `position` (TEXT)
   - `salary` (INTEGER)

2. `game_logs_skaters` - Actual skater performance
   - `game_date` (TEXT)
   - `player_name` (TEXT)
   - `team` (TEXT)
   - `dk_fpts` (REAL)

3. `game_logs_goalies` - Actual goalie performance
   - `game_date` (TEXT)
   - `player_name` (TEXT)
   - `team` (TEXT)
   - `dk_fpts` (REAL)

4. `contest_results` - Contest metadata
   - `slate_date` (TEXT)
   - `score` (REAL)
   - `n_cashed` (INTEGER)

### Supporting Files:

- `/sessions/youthful-funny-faraday/mnt/Code/projection/team_normalize.py` - Team normalization functions
- `/sessions/youthful-funny-faraday/mnt/Code/projection/optimizer.py` - NHLLineupOptimizer class
- `/sessions/youthful-funny-faraday/mnt/Code/projection/daily_projections/` - Daily projection CSVs (for SE strategy)

## Implementation Details

### Team Normalization

The script imports `normalize_team_lower` from `team_normalize.py` for robust handling of:
- DK format: `NJ`, `LA`, `SJ`, `TB`
- NHL format: `NJD`, `LAK`, `SJS`, `TBL`

All team abbreviations are normalized to lowercase NHL standard before matching.

### Scoring Logic

1. **Load Actuals:** Combine game_logs_skaters + game_logs_goalies for game_date
2. **Exact Match:** Find players with exact (name, team) match
3. **Fuzzy Match:** Use SequenceMatcher (threshold 0.70) for name variants
4. **Scratch Detection:** Count unmatched players as "scratched" (no score penalty)
5. **Total Score:** Sum all matched player FPTS

### Cash Line Detection

```sql
SELECT MIN(CASE WHEN n_cashed > 0 THEN score END) as cash_line
FROM contest_results WHERE slate_date = ?
```

This finds the minimum score where at least one entry cashed.

### Confidence Intervals

Wilson score interval (95%) for cash rates:
- Handles edge cases (0% or 100% cash rates)
- Works well with small sample sizes
- Formula: `CI = [(p + z²/2n) ± z√(p(1-p)/n + z²/4n²)] / (1 + z²/n)`

## Key Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Cash Rate** | (lineups >= cash_line) / total | % of lineups that cashed |
| **Win Rate** | (lineups >= winning_score) / total | % of lineups that won |
| **Avg FPTS** | sum(actual) / count | Average actual score |
| **Matched %** | avg(matched) / 9 | Quality of player matching |
| **Scratched %** | avg(scratched) / 9 | % of unplayed selections |
| **Proj Error** | avg(projected - actual) | Bias in projections |

## Troubleshooting

### Error: `no such column: name`

The `dk_salaries` table uses `player_name`, not `name`. This is automatically handled in the code.

### No lineups generated for date

**Salary-rank:** Usually indicates malformed optimizer call
**SE Optimizer:** Usually means no projection CSV exists for that date

Check:
```bash
ls daily_projections/ | grep "^MM_DD_YY"
```

### Mismatched player counts

If `matched` is too low (<6/9), likely causes:
1. Name spelling differences (nicknames, accents)
2. Team abbreviation mismatch (caught by normalize_team)
3. Player traded mid-season

Increase fuzzy matching threshold or add manual name mappings.

## Performance

### Runtime

- **Full backtest (both strategies):** ~5 minutes
- **Single strategy:** ~2-3 minutes
- **Per-date:** ~0.01-0.2s per lineup

### Memory Usage

- **Database:** 50-100 MB loaded
- **Peak:** ~200 MB during full backtest
- **Output CSV:** ~100 KB per strategy

## Extending the Engine

### Add New Strategy

```python
def run_my_strategy(date_str: str, seed: int = MASTER_SEED) -> Optional[List[Dict]]:
    """Build lineup using custom strategy."""
    np.random.seed(seed)
    try:
        # Load data
        pool = load_dk_pool(date_str, conn)
        
        # Build lineup
        # ... your logic ...
        
        # Return list of dicts: [{name, team, salary, projected_fpts}, ...]
        return players
    except Exception as e:
        print(f"Strategy error: {e}")
        return None
```

Then in `run_full_backtest()`:
```python
if method == 'my_strategy':
    players = run_my_strategy(date_str, seed=MASTER_SEED)
```

### Modify Scoring

To change matching logic, edit `score_lineup()` function:
```python
def score_lineup(players: List[Dict], actuals: pd.DataFrame) -> Tuple[float, int, int]:
    # Custom matching logic here
    # ... your code ...
    return total, matched, scratched
```

## References

- **Team Normalization:** `/sessions/youthful-funny-faraday/mnt/Code/projection/team_normalize.py`
- **Optimizer:** `/sessions/youthful-funny-faraday/mnt/Code/projection/optimizer.py`
- **Results:** `/sessions/youthful-funny-faraday/mnt/Code/projection/BACKTEST_RESULTS.md`
