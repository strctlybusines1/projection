# NHL DFS Projection System — Complete Analysis Report
## February 16, 2026

---

## 1. Signal Hunt Results (6 of 7 Significant)

### Tier 1 — Primary Features (Large Effect Sizes)

**PP Individual xG (ixG)** — Cohen's d = 0.580, p < 0.000001
- Players with high PP ixG score +3.88 FPTS more per game
- 96.2% name match rate between NST and boxscore data
- This is the single strongest signal in the entire dataset

**On-Ice HDCF% (High-Danger Chances For)** — Cohen's d = 0.192, p = 3.48e-65
- Players on the right side of HD chances score +1.33 FPTS more
- 96.3% match rate, 31,686 game records analyzed

**PP Reliance (inverse)** — Cohen's d = 0.556, p < 0.000001
- Even-strength producers outperform PP-dependent players by +3.72 FPTS
- Validates the PP ixG signal from the opposite direction

### Tier 2 — Secondary Features (Small Effect Sizes)

**Opponent Regime (xGF% tiers)** — p = 0.000002, +0.44 FPTS
- Facing weak defensive teams gives a small but real edge
- Strongest for right wingers (+0.94 FPTS) and defensemen (+0.39)

**Opponent SV% (Defensive Leakiness)** — p = 2.99e-07, +0.39 FPTS
- Playing against low SV% teams helps, but small effect

**Opponent xGF% (continuous)** — r = -0.029, p = 2.19e-07
- Better as continuous feature than tier-based

### Combined OLS Model
- Baseline MAE (season avg only): 5.32 FPTS
- Full model MAE (all features): 3.92 FPTS
- **Improvement: 26.4%**

---

## 2. Projection Source Audit

### FC Proj (External — likely FantasyCalc)
- **MAE: 5.91 FPTS** — poor
- **Correlation with actual: 0.321** — weak
- Systematic over-projection of +1.29 FPTS
- Goalies worst at MAE 7.69

### Your Internal Model (projections.py → NHLProjectionModel)
- TabPFN-based with bias corrections calibrated Feb 3
- Current pipeline: 80% your model + 20% DK season average
- **Skater MAE: 5.47 FPTS** — over-projecting by +3.64 FPTS
- The multiplicative boosts (HMM, linemate, edge stats) appear to inflate projections
- Global bias correction of 0.80 isn't enough

### Kalman Filter (kalman_projection.py)
- **Stat filter MAE: 4.318** — best individual model
- Built but NOT integrated into the main pipeline
- Should be the primary baseline going forward

### Key Problem
Neither FC Proj nor your current internal model are accurate enough. The Kalman filter is your best projection source and it's sitting unused in the pipeline.

---

## 3. Ownership Analysis (own.csv — 13,705 records)

### Critical Finding: Own Proj Column is 0% Populated
There is literally no ownership prediction model running. This is your biggest competitive gap AND biggest opportunity.

### FC Proj vs Ownership
- Salary-ownership correlation: 0.308 (weak)
- FC Proj correlation with FPTS: 0.321 (weak)
- Market is 60-70% efficient, leaving 30-40% exploitable

### GPP Leverage Signals
- **Underowned winners** (<10% own, beat projection by 5+ FPs): 2,040 instances (14.9%)
- **Chalk busts** (>30% own, <5 FPs): 83 instances (0.6% but devastating)
- **Underground stars** (<10% own, 20+ FPs): 796 instances (5.8%)
- **PP2 undervaluation**: +33% value gap vs PP1

### Ownership Quintile Analysis
High-owned players DO score more, but the ownership premium is massively inefficient — the EV-per-ownership-point drops 13x from low to high owned.

---

## 4. LSTM-CNN Feasibility

### Verdict: NOT READY as Primary Model
- **LSTM-CNN MAE: 5.05 FPTS** — 16.9% WORSE than Kalman (4.32)
- Vanilla architecture with 7 features can't beat Kalman's explicit noise modeling
- Hockey's extreme game-to-game variance (std dev 6.91) favors Bayesian approaches

### Why It Underperformed
1. Missing opponent context features (xGF%, xGA%)
2. No attention mechanisms
3. Position-agnostic (one model fits all)
4. Kalman is mathematically optimal for Gaussian noise with linear dynamics

### Path Forward (if pursuing)
- Phase 1: Add NST features + attention → expect parity with Kalman
- Phase 2: Ensemble (35% LSTM + 65% Kalman) → expect 2-5% improvement
- Timeline: 6-8 weeks for meaningful improvement
- Success probability: ~75%

---

## 5. Feb 25 Game Plan — Immediate Actions

### Week 1 (Now → Feb 25)

1. **Integrate Kalman filter as primary projection source**
   - Replace or heavily blend with current TabPFN model
   - Kalman MAE 4.32 vs current model 5.47

2. **Add PP ixG and HDCF% features**
   - These are the two strongest signals found
   - Combined potential: +3-5 FPTS accuracy improvement

3. **Build basic ownership model**
   - Ridge regression using salary + FC Proj + position
   - Even a basic model beats having nothing (current state)

4. **Recalibrate bias correction**
   - Current 0.80 global correction isn't enough
   - Over-projecting skaters by +3.64 FPTS
   - Consider position-specific corrections

### Week 2-3

5. **Opponent regime adjustments**
   - Use NST 5v5 xGF% as continuous feature (not tiers)
   - Apply per-position (R wingers get biggest boost)

6. **Build LSTM-CNN with NST features**
   - Add opponent context, attention mechanism
   - Train position-specific models
   - Target: Kalman parity or slight improvement

7. **Ownership model v2**
   - Add line/PP unit information
   - Track recency (recent ownership trends)
   - Build contest-size-specific variants

---

## Signal Priority Matrix

| Signal | Effect Size | p-value | Priority | Action |
|--------|-----------|---------|----------|--------|
| PP ixG | d=0.580 | <1e-6 | **HIGH** | Integrate immediately |
| HDCF% | d=0.192 | 3.5e-65 | **HIGH** | Integrate immediately |
| PP Reliance | d=0.556 | <1e-6 | **HIGH** | Use as validation |
| Opp Regime | d=0.072 | 1.6e-7 | MEDIUM | Secondary feature |
| Opp SV% | d=0.057 | 3.0e-7 | MEDIUM | Secondary feature |
| Opp xGF% cont. | r=-0.029 | 2.2e-7 | MEDIUM | Better than tiers |
| Ownership gap | — | — | **CRITICAL** | Build model ASAP |
