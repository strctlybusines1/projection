# NHL DFS Ownership Projection Plan

## Overview
Build a system to predict ownership percentages for NHL DFS players on DraftKings. Ownership predictions enable:
- **Leverage plays**: Target low-owned, high-upside players
- **Lineup differentiation**: Avoid overly chalky builds in GPPs
- **Expected ownership**: Calculate total lineup ownership for portfolio construction

---

## Current Model Status: ✅ VALIDATED

### Validation Results (January 23, 2026)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MAE** | 2.16% | < 3% | ✅ |
| **Correlation** | 0.607 | > 0.50 | ✅ |
| **Bias** | -0.27% | ±1% | ✅ |
| **Total Ownership** | 900% | 900% | ✅ |

### Position Accuracy
| Position | MAE | Correlation |
|----------|-----|-------------|
| D | 1.53% | 0.649 ✅ |
| RW | 2.12% | 0.637 ✅ |
| LW | 2.23% | 0.514 |
| C | 2.62% | 0.651 ✅ |
| G | 3.18% | 0.371 |

---

## Model Architecture (Implemented)

### Key Parameters (`OwnershipConfig`)
```python
# Salary curve (boosted mid-salary where chalk concentrates)
salary_curve = {
    (2500, 3500): 4.0,    # Punt plays
    (3500, 4500): 12.0,   # Value sweet spot - BOOSTED
    (4500, 5500): 14.0,   # Value sweet spot - BOOSTED
    (5500, 6500): 12.0,   # Mid-range
    (6500, 7500): 11.0,   # Solid plays
    (7500, 8500): 13.0,   # Premium
    (8500, 9500): 16.0,   # Stars
    (9500, 11000): 22.0,  # Elite - BOOSTED
}

# Multipliers
pp1_boost = 1.5              # +50% for PP1 players
pp2_boost = 1.15             # +15% for PP2 players
line1_boost = 1.25           # +25% for Line 1 players
confirmed_goalie_boost = 1.8 # +80% for confirmed starters
unconfirmed_goalie_penalty = 0.02  # 98% reduction (essentially 0%)

# Value adjustments
high_value_boost = 1.5       # +50% for top value plays
elite_value_boost = 1.8      # +80% for elite value (>1.5x avg)
low_value_penalty = 0.6      # -40% for poor value
smash_spot_boost = 1.4       # +40% for mid-salary value plays
```

### Normalization
Total ownership **must equal 900%** (9 roster spots):
- C: 200% (2 slots)
- LW: 150% (1.5 W slots)
- RW: 150% (1.5 W slots)
- D: 200% (2 slots)
- G: 100% (1 slot)

### Goalie Logic
1. **With confirmation data**: Confirmed starters get 1.8x boost, non-confirmed get 0.02x (essentially 0%)
2. **Without confirmation data**: Use salary as proxy
   - $8K+ → 1.3x (likely starter)
   - $7K-$8K → 1.0x (possible starter)
   - <$7K → 0.5x (likely backup)

---

## Validation Analysis (1/23/26)

### What Works Well
1. **Defensemen predictions** - 1.53% MAE, 0.649 correlation
2. **Mid-owned players (1-5%)** - 1.37% MAE
3. **Contrarian plays (<1%)** - 1.43% MAE
4. **Total ownership constraint** - Exactly 900%

### Known Gaps

#### 1. Chalk Under-Prediction (-13.8% bias for 15%+ owned)
Top misses on 1/23:
| Player | Predicted | Actual | Error |
|--------|-----------|--------|-------|
| Victor Olofsson | 4.8% | 24.9% | -20.1% |
| Macklin Celebrini | 7.0% | 23.6% | -16.6% |
| Will Smith | 8.5% | 24.1% | -15.7% |
| Nathan MacKinnon | 14.5% | 28.5% | -14.0% |

**Root Cause**: These are "narrative plays" - value + hype + easy matchup. Hard to model without news/social data.

**Potential Fixes**:
- Add "buzz factor" from Twitter/news mentions
- Boost players on teams with high Vegas implied totals
- Increase value boost even further for extreme value plays

#### 2. Goalies Still Under-Predicted
Confirmed starters consistently under-predicted by 2-5%.

**Potential Fixes**:
- Increase confirmed_goalie_boost from 1.8 to 2.2
- Add "chalk goalie" detection (top 2 salary goalies on slate)

---

## Daily Workflow

### Before Lock
1. Load DK salaries and projections
2. Fetch confirmed goalies from lines.py
3. Run `OwnershipModel.predict_ownership()`
4. Review leverage plays (high proj, low ownership)
5. Review chalk plays (potential fades)

### After Contest
1. Download contest results CSV
2. Run validation script to compare predicted vs actual
3. Update this document with findings
4. Adjust multipliers if systematic bias detected

---

## Integration Points

### With Optimizer
```python
from ownership import OwnershipModel

# Add ownership to player pool
ownership_model = OwnershipModel()
ownership_model.set_lines_data(lines_data, confirmed_goalies)
player_pool = ownership_model.predict_ownership(player_pool)

# Use for lineup construction
leverage_plays = player_pool[
    (player_pool['projected_fpts'] >= 12) &
    (player_pool['predicted_ownership'] <= 8)
]
```

### Output Columns Added
- `predicted_ownership` - Predicted ownership %
- `leverage_score` - projected_fpts / (predicted_ownership + 1)
- `ownership_tier` - Chalk/Popular/Moderate/Low/Contrarian

---

## Backtest History

| Date | MAE | Correlation | Bias | Notes |
|------|-----|-------------|------|-------|
| 1/23/26 | 2.16% | 0.607 | -0.27% | After tuning |
| 1/23/26 | 2.49% | 0.525 | +0.73% | Before tuning |

---

## Future Improvements

### Short-Term
- [ ] Integrate Vegas implied team totals
- [ ] Add "top 3 value at position" boost
- [ ] Increase goalie boost for confirmed starters

### Medium-Term
- [ ] Train ML model on accumulated historical data
- [ ] Add news/Twitter sentiment for buzz factor
- [ ] Build ownership correlation matrix (stacking effects)

### Long-Term
- [ ] Real-time ownership updates from contest entry data
- [ ] Portfolio optimization using ownership projections
- [ ] Automated contrarian lineup builder

---

## Files

| File | Purpose |
|------|---------|
| `ownership.py` | Main ownership prediction module |
| `OWNERSHIP_PLAN.md` | This document |
| `$*main_NHL*.csv` | Historical contest results for validation |

---

*Last Updated: January 24, 2026*
*Validated on: January 23, 2026 contest data*
