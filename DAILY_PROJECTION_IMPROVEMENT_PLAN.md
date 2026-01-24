# NHL DFS Projection Improvement Plan

## Daily Reference Guide for Reducing MAE

**Current Baseline (2-day sample: 1/22-1/23/26)**
- Overall MAE: **4.88 pts**
- Skater MAE: **4.84 pts**
- Goalie MAE: **5.75 pts**
- Correlation: **0.370**

---

## 1. DAILY BACKTEST CHECKLIST

Run this every morning after the previous night's games:

```bash
# After updating the backtest Excel with actual results
python3 -c "
# Quick backtest script - paste actual results and run
# See backtest.py for full implementation
"
```

### Key Metrics to Track Daily:
| Metric | Target | Current |
|--------|--------|---------|
| Overall MAE | < 4.5 | 4.88 |
| Skater MAE | < 4.5 | 4.84 |
| Goalie MAE | < 5.0 | 5.75 |
| Correlation | > 0.45 | 0.370 |
| Within 5 pts | > 70% | 62.5% |
| Floor accuracy | > 80% | 69.5% |

---

## 2. IDENTIFIED ISSUES & FIXES

### Issue #1: Skater Over-Projection Bias (+1.04 pts)

**Root Cause:** The projection formula in `projections.py` consistently overestimates skaters.

**Current Code (lines 41-110):**
- Multiple multiplicative boosts compound: opp_softness, xg_matchup_boost, streak adjustment, PDO adjustment, opportunity_boost, home ice
- Each 2-3% boost compounds to significant over-projection

**FIX: Add position-specific dampening factors**

```python
# In projections.py, add after line 103 (before home ice):

# Position-specific bias correction (based on backtest data)
POSITION_BIAS_CORRECTION = {
    'C': 0.96,   # Centers over-projected by ~1.28 pts
    'L': 0.95,   # Left wings over-projected by ~1.65 pts
    'R': 1.01,   # Right wings slightly under-projected
    'D': 0.94,   # Defensemen over-projected by ~1.56 pts
}

position = row.get('position', 'C')
bias_correction = POSITION_BIAS_CORRECTION.get(position, 0.97)
expected_pts *= bias_correction
```

### Issue #2: Floor Calculation Too High (30.5% below floor)

**Current Code (line 165):**
```python
df['floor'] = df['projected_fpts'] * 0.4  # Bad game
```

**Problem:** 0.4x is too high. Many players score 0-2 pts.

**FIX: Use 0.25x multiplier with position adjustment**

```python
# Replace line 165 with:
floor_mult = {'C': 0.25, 'L': 0.25, 'R': 0.25, 'D': 0.30, 'G': 0.20}
df['floor'] = df.apply(
    lambda r: r['projected_fpts'] * floor_mult.get(r['position'], 0.25),
    axis=1
)
```

### Issue #3: Goalie Under-Projection (-0.86 pts bias)

**Current Code (lines 112-151):**
- Win rate weight may be too conservative
- Save projection doesn't account for high-event games

**FIX: Increase goalie baseline and save expectations**

```python
# In calculate_expected_fantasy_points_goalie, adjust:

# Increase saves expectation for facing high-shot teams
if row.get('opp_shots_per_game', 30) > 32:
    saves = saves * 1.08  # High-volume opponent

# Increase win bonus weight
expected_pts += win_rate * GOALIE_SCORING['win'] * 1.15  # Was 1.0
```

### Issue #4: Ceiling Calculation (only 1.9% exceeded)

**Current Code (line 166):**
```python
df['ceiling'] = df['projected_fpts'] * 2.5 + 5  # Great game with bonuses
```

**Analysis:** This is working well (low exceed rate is good for ceiling).
**Keep as-is** - ceiling formula is appropriate.

---

## 3. ROLLING BIAS TRACKER

Update this daily to track systematic bias trends:

| Date | Skater Bias | Goalie Bias | Overall MAE | Notes |
|------|-------------|-------------|-------------|-------|
| 1/22 | +0.34 | +0.13 | 4.98 | Small sample |
| 1/23 | +1.71 | -1.78 | 4.81 | Stars underperformed |
| 1/24 | TBD | TBD | TBD | |

**7-Day Rolling Bias Target:** Within +/- 0.5 pts

---

## 4. CONFIG.PY ADJUSTMENTS

### Current Settings That May Need Tuning:

```python
# Streak adjustment (currently 10% - may be too high)
STREAK_ADJUSTMENT_FACTOR = 0.10  # Consider reducing to 0.05

# PDO regression (currently 5% - appropriate)
PDO_REGRESSION_FACTOR = 0.05  # Keep as-is

# Home ice advantage (in projections.py)
# Currently: 1.02 for skaters, 1.03 for goalies
# Consider: 1.01 for skaters (smaller effect)
```

---

## 5. FEATURE IMPORTANCE ANALYSIS

Based on backtest correlation with actual results:

### High-Impact Features (keep/enhance):
1. `goals_pg` - Strong predictor
2. `shots_pg` - Strong predictor
3. `toi_minutes` - Correlates with opportunity
4. `opp_softness` - Matchup matters

### Low-Impact Features (consider removing/reducing weight):
1. `team_hot_streak` / `team_cold_streak` - May add noise
2. `pdo_adj_factor` - Small effect, may add complexity
3. `opportunity_boost` - Hard to measure accurately

---

## 6. DAILY WORKFLOW

### Morning (Before Lock):
1. Run backtest on previous night's results
2. Update Rolling Bias Tracker
3. If bias > 1.0 pts, apply temporary correction factor
4. Review biggest misses - look for patterns

### Pre-Projection:
1. Check for late scratches/injuries
2. Verify goalie confirmations
3. Apply any same-day bias corrections

### Post-Slate:
1. Record actual results in backtest spreadsheet
2. Note any systematic issues (e.g., "All goalies outperformed")
3. Update this document with learnings

---

## 7. SPECIFIC CODE CHANGES TO IMPLEMENT

### Priority 1: Bias Correction (Immediate)

In `projections.py`, add after line 103:

```python
# Backtest-derived bias correction
BIAS_CORRECTION = 0.97  # Reduce all projections by 3%
expected_pts *= BIAS_CORRECTION
```

### Priority 2: Position-Specific Adjustments (This Week)

In `projections.py`, replace home ice section with:

```python
# Position-specific adjustments (backtest-derived)
position = row.get('position', 'C')
if position == 'D':
    expected_pts *= 0.95  # Defensemen consistently over-projected
elif position in ['L', 'LW']:
    expected_pts *= 0.96  # Left wings over-projected
elif position == 'C':
    expected_pts *= 0.97  # Centers slightly over-projected

# Home ice advantage (reduced from 1.02)
if row.get('is_home') == True:
    expected_pts *= 1.01
```

### Priority 3: Goalie Model Update (This Week)

In `projections.py`, update goalie function:

```python
# Increase baseline for goalies (they're under-projected)
expected_pts *= 1.05  # 5% boost to goalie projections
```

### Priority 4: Floor Formula (This Week)

In `projections.py`, line 165:

```python
# More conservative floor
df['floor'] = df['projected_fpts'] * 0.25
```

---

## 8. EXPECTED IMPACT

If all changes implemented:

| Metric | Current | Target | Expected |
|--------|---------|--------|----------|
| Overall MAE | 4.88 | 4.50 | 4.55 |
| Skater Bias | +1.04 | 0.0 | +0.3 |
| Goalie Bias | -0.86 | 0.0 | -0.2 |
| Floor Accuracy | 69.5% | 80% | 78% |

---

## 9. TRACKING TEMPLATE

Copy this for each day's backtest:

```
Date: ____
Games: ____
Skaters Projected: ____
Goalies Projected: ____

Results:
- Overall MAE: ____
- Skater MAE: ____
- Goalie MAE: ____
- Skater Bias: ____
- Goalie Bias: ____
- Within 5 pts: ____%
- Below Floor: ____%

Biggest Misses:
1. ____ (Proj: ___, Actual: ___)
2. ____ (Proj: ___, Actual: ___)
3. ____ (Proj: ___, Actual: ___)

Notes/Adjustments Needed:
_________________________________
```

---

## 10. LONG-TERM IMPROVEMENTS

### Data Enhancements:
- [ ] Add line combination data (who plays with whom)
- [ ] Add power play unit assignments
- [ ] Track back-to-back games
- [ ] Add Vegas implied team totals

### Model Enhancements:
- [ ] Train TabPFN on historical backtest data
- [ ] Add player-specific variance estimates
- [ ] Implement Bayesian updating from recent games

### Process Improvements:
- [ ] Automate daily backtest pipeline
- [ ] Build dashboard for tracking metrics
- [ ] Alert system for large bias shifts

---

*Last Updated: January 24, 2026*
*Based on backtest data from January 22-23, 2026*
