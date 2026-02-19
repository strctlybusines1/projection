# Project Cleanup Checklist

## Directory Organization

### daily_salaries/
- [ ] Archive files older than 7 days to `daily_salaries/archive/`
- [ ] Verify naming convention: `DKSalaries_M_D_YY.csv`
- [ ] Remove any duplicate files

### daily_projections/
- [ ] Archive files older than 7 days to `daily_projections/archive/`
- [ ] Keep only latest projection + lineup for each date
- [ ] Remove intermediate/test outputs

### backtests/
- [ ] Consolidate xlsx files by month
- [ ] Keep `latest_mae.json` updated
- [ ] Archive ownership_model.pkl versions with dates

### contests/
- [ ] Organize by contest type (WTA, GPP, SE)
- [ ] Archive processed results

### vegas/
- [ ] Archive old Vegas fallback files
- [ ] Keep only last 7 days active

## Code Cleanup

### Remove Dead Code
- [ ] Check for unused imports in each .py file
- [ ] Remove commented-out code blocks
- [ ] Delete unused utility functions

### Consolidate Configuration
- [ ] All constants in config.py
- [ ] Bias corrections in projections.py (documented)
- [ ] EDGE thresholds in edge_stats.py (documented)

### Fix Known Issues
- [ ] EDGE API timing (two-step workflow documented)
- [ ] Injury filter flag (--filter-injuries)
- [ ] Position normalization consistency

## Documentation

### CLAUDE.md
- [ ] Update workflow section
- [ ] Document all CLI flags
- [ ] Add troubleshooting section

### Code Comments
- [ ] Each module has docstring header
- [ ] Complex functions documented
- [ ] Magic numbers explained

## Dependencies

### .env
- [ ] ODDS_API_KEY present
- [ ] No hardcoded API keys in code

### requirements.txt
- [ ] All dependencies listed
- [ ] Version pins for critical packages:
  - pandas
  - numpy
  - nhl-api-py
  - scikit-learn

## Archive Script

```bash
#!/bin/bash
# Run from projection/ directory

# Create archive directories
mkdir -p daily_salaries/archive
mkdir -p daily_projections/archive
mkdir -p vegas/archive

# Archive files older than 7 days
find daily_salaries -maxdepth 1 -name "*.csv" -mtime +7 -exec mv {} daily_salaries/archive/ \;
find daily_projections -maxdepth 1 -name "*.csv" -mtime +7 -exec mv {} daily_projections/archive/ \;
find vegas -maxdepth 1 -name "*.csv" -mtime +7 -exec mv {} vegas/archive/ \;

echo "Archived old files"
```

## Validation Checks

After cleanup, verify:

```bash
# 1. Main pipeline still works
python main.py --stacks --show-injuries --lineups 1

# 2. Backtest still works  
python backtest.py --players 10

# 3. Lines scraper works
python lines.py

# 4. No import errors
python -c "import main, data_pipeline, features, projections, edge_stats, lines, ownership, optimizer, backtest"
```
