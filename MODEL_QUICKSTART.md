# Ownership Prediction Model v2 - Quick Start Guide

## Files Overview

| File | Size | Purpose |
|------|------|---------|
| `ownership_v2.py` | 28 KB | Main model training pipeline |
| `ownership_v2_results.csv` | 1.1 MB | Model predictions on all 13,705 records |
| `OWNERSHIP_MODEL_REPORT.txt` | 15 KB | Comprehensive technical report |
| `MODEL_QUICKSTART.md` | This file | Quick reference guide |

## Model Performance at a Glance

```
Validation Set Metrics:
- Mean Absolute Error: 3.3560%
- Spearman Correlation: 0.6466
- R²: 0.4954
- Position Best: Centers (r=0.69), Defensemen (MAE=2.80%)
```

## Key Features (43 total)

**Top 5 Most Important:**
1. Number of games on slate (slate diversity)
2. Value score (projection per 1k salary)
3. DK average fantasy points (historical performance)
4. Projection × top line (interaction)
5. Center position indicator

## Game Theory Layer

### Leverage Score = Projection / (Ownership + 1)

**High-Leverage Plays** (299 players):
- Avg projection: 13.76 pts
- Avg predicted ownership: 1.44% (very low!)
- Avg leverage: 5.74x
- Best for: Contrarian GPP lineups

**Chalk Plays** (3,425 players):
- Avg projection: 12.76 pts
- Avg predicted ownership: 14.65%
- Best for: Dominant cores, cash games

### Stack Analysis
High-leverage team stacks enable portfolio optimization:
- Identify teams with high projection + low ownership
- Build contrarian lineups with multiple team combinations
- Backtest vs historical results

## How to Use the Results

### 1. Building Contrarian GPP Lineups
```python
import pandas as pd

results = pd.read_csv('ownership_v2_results.csv')

# Get high-leverage plays
high_leverage = results[results['is_high_leverage'] == 1]
print(high_leverage[['Player', 'Pos', 'fc_proj', 'own_pred', 'leverage']])

# Sort by leverage (best risk/reward)
high_leverage_sorted = high_leverage.sort_values('leverage', ascending=False)
```

### 2. Identifying Fades
```python
# Chalk players (high ownership, mediocre projection)
fades = results[
    (results['own_pred'] > results['own_pred'].quantile(0.75)) &
    (results['fc_proj'] < results['fc_proj'].median())
]
```

### 3. Team Stack Analysis
```python
# Group by date and team
stacks = results.groupby(['date', 'Team']).agg({
    'leverage': 'sum',
    'fc_proj': 'sum',
    'own_pred': 'mean'
}).sort_values('leverage', ascending=False)
```

## Model Calibration Details

### Position-Based Normalization
After raw XGBoost prediction:
1. Isotonic regression applied for probability calibration
2. Position-specific ownership targets enforced:
   - Centers: ~200% total
   - Wings: ~300% total
   - Defensemen: ~200% total
   - Goalies: ~100% total
   - **Total per slate: ~800% (9-10x stacking depth)**

## Training Approach

**Walk-Forward Validation:**
- Train: Oct 7 - Dec 29, 2025 (70 dates, 9,460 rows)
- Test: Dec 30, 2025 - Feb 4, 2026 (31 dates, 4,245 rows)
- No data leakage: Historical ownership features lagged

**Feature Engineering:**
- Slate-level: Contest type, field size, number of games
- Player-level: Salary, projections, value scoring
- Performance: Recent 5-game averages (rolling)
- Historical: Lagged past ownership (no future info)
- Interactions: 5 cross-feature interactions
- Vegas: Odds, spreads, implied totals

## Position-Specific Performance

| Position | Count | MAE | Correlation | Best For |
|----------|-------|-----|-------------|----------|
| G | 398 | 4.57% | 0.578 | Prediction challenging |
| W | 1,691 | 3.32% | 0.656 | Good prediction signal |
| C | 902 | 3.66% | 0.689 | Strongest signal |
| D | 1,254 | 2.80% | 0.607 | Most predictable |

## Interpretation Examples

### Example 1: High-Leverage Play
```
Player: Connor McDavid
Position: C
Projection: 27.36 pts
Predicted Own: 5.7%
Leverage: 4.80x
Insight: Elite projection, very low predicted ownership
Action: Candidate for contrarian GPP lineups
```

### Example 2: Chalk/Fade
```
Player: Nathan MacKinnon
Position: C
Projection: 31.96 pts
Predicted Own: 33.6%
Leverage: 0.92x
Insight: Elite player but very high ownership
Action: Consider fading in contrarian lineups
```

## Limitations & Considerations

**When Model Works Well:**
- Building contrarian GPP lineups (primary use case)
- Ranking plays by relative value
- Identifying ownership disparities
- Team stack optimization

**When to be Cautious:**
- Cash games (different ownership distribution, needs higher accuracy)
- Last-minute news/injuries (model can't react in real-time)
- Very small or very large field contests (assumptions may not hold)
- Game lines heavily influenced by public news after model trained

## Future Improvements

1. **Contest-Type Models**: Separate SE vs multi-entry models
2. **Real-Time Recalibration**: Update ownership for late-breaking news
3. **Time-Series Features**: Model ownership momentum
4. **Ensemble Approach**: Combine with projection models
5. **Automated Lineup Generation**: Direct connection to optimizer

## Quick Stats

```
Dataset Size: 13,705 rows (101 unique dates, 764 unique players)
Slate Coverage: Oct 7, 2025 - Feb 4, 2026
Ownership Range: 0.7% to 82.4%
Mean Ownership: 6.63% (std: 7.96%)
Training Time: ~5-10 minutes
Model Complexity: XGBoost with 500 estimators, max_depth=7
```

## Contact & Debugging

If model predictions seem off:
1. Check contest type (SE vs multi-entry behaves differently)
2. Verify player name matching (name normalization in code)
3. Look for recent injuries not in database yet
4. Check if overlay is active (changes ownership concentrations)

## References

- Main Report: `OWNERSHIP_MODEL_REPORT.txt`
- Source Code: `ownership_v2.py`
- Result Data: `ownership_v2_results.csv`
- Database: `data/nhl_dfs_history.db`

---
Model Version: 2.0
Last Updated: 2026-02-16
Status: Production Ready
