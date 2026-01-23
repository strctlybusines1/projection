# NHL DFS Ownership Projection Plan

## Overview
Build a system to predict ownership percentages for NHL DFS players on DraftKings. Ownership predictions enable:
- **Leverage plays**: Target low-owned, high-upside players
- **Lineup differentiation**: Avoid overly chalky builds in GPPs
- **Expected ownership**: Calculate total lineup ownership for portfolio construction

---

## Phase 1: Data Collection & Analysis

### Available Data Sources
1. **DraftKings Salary File**
   - Salary (strong ownership driver)
   - AvgPointsPerGame (popularity proxy)
   - Position, Team, Game Info

2. **Our Projections**
   - projected_fpts, floor, ceiling
   - value (pts/$1000)
   - edge vs DK average

3. **Lines Data (from lines.py)**
   - PP1 status (high ownership driver)
   - Line combinations
   - Confirmed goalie starters

4. **Historical Contest Results**
   - Actual ownership % from past contests (e.g., `$10main_NHL1.22.26.csv`)
   - Can be used to train/validate model

### Key Ownership Drivers (based on DFS research)
| Factor | Impact | Notes |
|--------|--------|-------|
| Salary | High | Mid-range ($4-6K) often highest owned |
| Value (pts/$) | High | High value = high ownership |
| PP1 status | High | Power play players get boosted |
| Confirmed starter (G) | Very High | Non-confirmed goalies get ~0% |
| Recent performance | Medium | Hot streaks drive ownership |
| Game total (Vegas) | Medium | High totals = more interest |
| Star power | Medium | Name recognition matters |
| News/Injuries | High | Backup goalies spike when starter out |

---

## Phase 2: Model Architecture

### Option A: Heuristic Model (Start Here)
Simple formula-based approach that can be tuned:

```python
base_ownership = f(salary)  # Salary curve (mid-range peaks)
ownership_multipliers = [
    value_factor,        # High value = higher ownership
    pp1_boost,           # +50% if on PP1
    confirmed_goalie,    # Goalies: 0% if not confirmed
    projection_factor,   # High projection = higher ownership
    recency_boost,       # Recent high scores
]
predicted_ownership = base_ownership * product(multipliers)
normalize_to_100()  # Scale so total ownership makes sense
```

### Option B: Machine Learning Model (Future)
Train on historical ownership data:
- **Features**: salary, value, projection, PP1, confirmed, team implied total
- **Target**: actual ownership %
- **Models**: Ridge Regression, Random Forest, or XGBoost
- **Validation**: Backtest on held-out contest data

---

## Phase 3: Implementation Plan

### Step 1: Create `ownership.py`
```
ownership.py
├── OwnershipModel class
│   ├── __init__(player_pool, lines_data, vegas_data=None)
│   ├── calculate_base_ownership()     # Salary-based curve
│   ├── apply_value_adjustment()       # Value multiplier
│   ├── apply_pp1_boost()              # Power play boost
│   ├── apply_goalie_filter()          # Confirmed starters only
│   ├── apply_projection_factor()      # Higher proj = higher own
│   ├── normalize_ownership()          # Scale to realistic totals
│   └── predict_ownership()            # Main method
├── analyze_historical_ownership()     # Analyze past contests
└── validate_predictions()             # Compare to actuals
```

### Step 2: Salary-Based Ownership Curve
Based on DFS research, ownership follows a curve:
- $2,500-3,500: Low (punt plays) ~3-8%
- $4,000-6,000: Highest (value sweet spot) ~10-20%
- $6,500-8,000: Medium (solid plays) ~8-15%
- $8,500+: Medium-High (stars/chalk) ~12-25%

### Step 3: Key Adjustments
```python
# PP1 Boost
if player in pp1_unit:
    ownership *= 1.5  # +50%

# Confirmed Goalie (critical)
if position == 'G' and not confirmed_starter:
    ownership = 0.5  # Near zero

# Value Multiplier
value_ratio = player_value / avg_value
ownership *= (0.5 + 0.5 * value_ratio)  # Scale based on value

# Projection Factor
proj_ratio = projected_fpts / position_avg_projection
ownership *= (0.7 + 0.3 * proj_ratio)
```

### Step 4: Output Format
```
name, team, position, salary, projected_fpts, predicted_ownership
Nathan MacKinnon, COL, C, 10200, 29.0, 22.5%
Connor McDavid, EDM, C, 10100, 28.5, 25.1%
...
```

---

## Phase 4: Integration with Optimizer

Once validated, add to optimizer:
1. **Display ownership** in lineup output
2. **Lineup ownership score**: Sum of player ownerships
3. **Leverage score**: High projection / low ownership ratio
4. **Contrarian mode**: Prefer low-owned players

---

## Phase 5: Validation & Tuning

### Metrics
- **MAE**: Mean Absolute Error vs actual ownership
- **Correlation**: Spearman rank correlation
- **Calibration**: Are 20% predicted players actually ~20% owned?

### Tuning Process
1. Run model on historical slate
2. Compare to actual ownership from contest results
3. Adjust multipliers/curve
4. Repeat until MAE < 5%

---

## Files to Create
1. `ownership.py` - Main ownership prediction module
2. Update `optimizer.py` - Add ownership display and leverage calculations

---

## Timeline
1. **Tonight**: Build v1 heuristic model in `ownership.py`
2. **Tomorrow**: Validate against tonight's contest results
3. **This week**: Tune model based on multiple slates
4. **Future**: Train ML model on accumulated data

---

## Questions to Resolve
1. Do we have Vegas lines data? (game totals, team implied totals)
2. How many historical contest files do we have for training?
3. What contest types to target? (Main slate $10 GPP vs single game)
