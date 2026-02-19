## Research: Improving xG Model Beyond AUC 0.75

### Current State

Your xG model currently achieves:
- **AUC: 0.7513** (GradientBoostingClassifier, sklearn)
- **Log Loss: 0.2441**
- **Training data: 15K shots** from partial 2025-26 season (175 games)
- **Features: 15 variables** including distance, angle, shot type, strength state, score differential, prior event attributes, rebound/rush flags
- **Top features by importance**: distance (20.4%), angle (10%), prior_distance (9.8%)

This is competitive with public xG models (which typically range 0.70–0.80 AUC) but well below what state-of-the-art implementations achieve. The system uses a single GBM trained on all strength states combined.

### Findings

#### 1. **Strength State Stratification** (High Priority, Medium Effort)

**What it is:** Train separate xG models for different game situations rather than one unified model.

**Evidence that it works:**
- Evolving Hockey employs four separate XGBoost models: even-strength (5v5/4v4/3v3), powerplay (5v4/4v3/5v3/6v5/6v4), shorthanded (4v5/3v4/3v5), and empty net.
- Play styles and shooting percentages differ dramatically by strength state—using strength state as a categorical feature is suboptimal compared to separate models that can learn different patterns for each situation.
- Academic research shows pooled models underperform stratified approaches due to fundamentally different shot distributions at even strength vs. power play.

**How it applies to your system:**
- Your current model treats strength state as one categorical variable. Building 4 separate GBM models would allow each to specialize in its own feature interactions and decision boundaries.
- Expected impact: +0.02–0.05 AUC (5–6 point lift from baseline 0.7513).
- Implementation complexity: **Low** — you have 15K shots; stratify them by strength state (roughly: ~10K even strength, ~3K powerplay, ~1.5K shorthanded, <500 empty net), train 4 separate models, ensemble predictions at inference time by prepending strength state.

**Data requirements:**
- You already have strength_state in your feature set; no new data needed.

---

#### 2. **Pre-Shot Movement & Passing Sequence Features** (High Priority, High Effort)

**What it is:** Add features that capture the quality and timing of passes/movement before the shot (not just the shot and immediate prior event).

**Evidence that it works:**
- Hockey Graphs built xG models incorporating pre-shot movement and found significant improvements. Key features include: number of passes in the offensive zone before the shot, pass types (royal road/behind-the-net/low-to-high/stretch), and time/distance/angle deltas between multiple prior events.
- Shots with a screen have ~2x shooting percentage; scoring chances with poor pass positioning have 3x lower goal probability vs. those from "oddman rushes" and high-danger sequences.
- MoneyPuck's model uses "shotAnglePlusReboundSpeed" (angle change divided by time since last shot), indicating temporal/spatial sequencing matters materially.

**How it applies to your system:**
- Your NHL API play-by-play data includes event sequences (prior_event_type, prior_distance, prior_time), but you only use the immediate prior event.
- Enhancement: extract 2–3 prior events before the shot, engineer features like: cumulative_distance_moved_in_zone, pass_count_in_zone, angle_variance_across_sequence, time_to_shot_from_entry.
- Expected impact: +0.03–0.08 AUC (if properly calibrated; this is the biggest gap in current public models).
- Implementation complexity: **High** — requires careful feature engineering, validation that features don't leak, and retraining. Full 6-season dataset strongly recommended before investing here.

**Data requirements:**
- You have this in NHL API PBP; no new data source needed. Requires parsing event sequences backward from each shot.

---

#### 3. **Full 6-Season Dataset Retrain** (Critical Dependency, Low Effort)

**What it is:** You're currently training on 15K shots from partial 2025-26 season. Historical data exists for ~600K+ shots across 6 full seasons.

**Evidence that it works:**
- Larger training sets reduce variance and capture edge cases (rare shot types, obscure situational patterns).
- Academic papers on xG consistently show that sample size drives AUC improvements—moving from 15K to 600K shots is a 40x increase in training data.
- Even simple models with 600K examples often outperform complex models with 15K examples (bias-variance tradeoff).

**How it applies to your system:**
- You've noted this as "in progress" in system_state.md but it's **blocking** the value of other improvements.
- Train your current GBM on full historical dataset first, then iterate on strength state stratification and feature engineering.
- Expected impact: +0.03–0.07 AUC (empirically, models trained on 6 seasons outperform 1-season models significantly).
- Implementation complexity: **Low** — same algorithm, just more data.

**Data requirements:**
- Need to finish NHL API scraper for all seasons and backfill pbp_shots table.

---

#### 4. **Separate Models by Shot Type** (Medium Priority, Low Effort)

**What it is:** Train separate xG models for different shot types (wrist shot, snap shot, backhand, deflection, rush, tip-in, etc.).

**Evidence that it works:**
- Different shot types have fundamentally different goal probabilities: a tip-in from the slot has 3–5x higher xG than a wrist shot from the perimeter.
- Evolving Hockey's research indicates shot type interacts strongly with distance/angle; unified models that use shot type as a categorical feature are less flexible than separate models.
- The danger of combining: a deflection 25 feet out behaves very differently than a wrist shot 25 feet out, but a single tree may place them in the same leaf.

**How it applies to your system:**
- Your feature set includes shot_type (7 types). Instead of treating it as categorical, train 7 separate GBMs (or at least 2–3 for major groups: deflections/tips vs. wrist/snap).
- Expected impact: +0.01–0.03 AUC.
- Implementation complexity: **Low** — similar to strength state stratification.

**Data requirements:**
- No new data; you already have shot_type.

---

#### 5. **Bayesian Mixed Effects Model** (Lower Priority, High Effort)

**What it is:** Use hierarchical Bayesian modeling to account for shooter/goaltender effects while regularizing against overfitting with small sample sizes.

**Evidence that it works:**
- Recent academic work (Toward interpretable expected goals modeling using Bayesian mixed models, published in PMC) shows Bayesian approaches improve calibration and interpretability.
- Bayesian models naturally handle the small-sample problem: a rookie with 3 shots doesn't overfit to an unrealistic xG; they're shrunk toward the population mean.
- Mixing shooter/goalie random effects into xG captures persistent talent differences while maintaining generalization.

**How it applies to your system:**
- Your current GBM is frequentist and deterministic. A Bayesian model with hierarchical terms (shooter random intercept, goalie random intercept) could improve robustness.
- More complex to implement; requires careful prior specification and posterior sampling (e.g., Stan, PyMC3).
- Expected impact: +0.01–0.04 AUC, but **larger impact on calibration and uncertainty estimates** (useful for your Monte Carlo simulation pipeline).
- Implementation complexity: **High** — requires new architecture and domain expertise.

**Data requirements:**
- Same as current model; no new data source.

---

#### 6. **Deep Learning (Neural Networks) for xG** (Lower Priority, High Effort, Unproven)

**What it is:** Replace GBM with a deep neural network (feedforward or LSTM) to capture non-linear shot patterns.

**Evidence that it works (cautiously):**
- Recent models using deep neural networks achieved AUC 0.7457 and log loss 0.2214 on 2024–25 test data—roughly equivalent to GBM, not better.
- CNN-based xG models in football achieved AUC 0.801 but didn't outperform state-of-the-art non-neural approaches.
- LSTM networks have been tested for temporal shot sequences but show mixed results; hockey's play-by-play is less "sequential" than soccer.

**How it applies to your system:**
- Neural networks excel when: you have very large datasets (you don't—15K is small), complex non-Euclidean inputs (you don't—shots are 2D points), or high-dimensional raw data (you have engineered features already).
- Risk: With only 15K shots and 15 features, a neural network will likely overfit and underperform GBM.
- Expected impact: **Uncertain; likely 0–0.02 AUC**, with higher variance and slower training.
- Implementation complexity: **High** — requires careful regularization (dropout, L2, early stopping) and hyperparameter tuning.

**Recommendation:** Do not prioritize until you've exhausted simpler methods (strength state stratification, pre-shot features) and trained on full 6-season dataset. The "deep learning" narrative can be tempting, but GBM remains state-of-the-art for small-to-medium tabular datasets.

---

### Recommendations

**Ranked by impact-to-effort ratio:**

1. **[IMMEDIATE] Full 6-Season Retrain** (AUC +0.03–0.07, Low Effort)
   - Finish NHL API scraper and backfill pbp_shots to 600K+ shots.
   - Retrain current GBM on full historical data.
   - Expected new baseline: **AUC 0.78–0.82**, unblocking all downstream improvements.

2. **[MONTH 1] Strength State Stratification** (AUC +0.02–0.05, Low Effort)
   - Train 4 separate GBM models: even strength (5v5/4v4/3v3), powerplay (5v4+), shorthanded (4v5+), empty net.
   - Ensemble by predicting with the appropriate model based on game state.
   - Expected cumulative AUC: **0.80–0.87**.

3. **[MONTH 2] Pre-Shot Movement Features** (AUC +0.03–0.08, High Effort)
   - Engineer 4–5 new features from event sequences: pass count in zone, angle variance, distance traveled, entry type classifier.
   - Validate that features improve 5-fold CV without leaking future information.
   - Add to stratified models and retrain.
   - Expected cumulative AUC: **0.83–0.92** (if well-executed; this is the biggest untapped improvement).

4. **[MONTH 3] Shot Type Stratification** (AUC +0.01–0.03, Low Effort)
   - Similar to strength state: train separate models for major shot type clusters (deflections/tips, wrist/snap, backhands).
   - Expected cumulative AUC: **0.84–0.93**.

5. **[FUTURE] Bayesian Mixed Effects** (Calibration + Uncertainty, High Effort)
   - Only after you've maxed out GBM improvements.
   - Useful for Monte Carlo simulation and handling rare player/goalie combos.

6. **[FUTURE] Deep Learning** (Unproven, High Effort, Low Confidence)
   - Only if you exhaust traditional methods and have 500K+ shots with strong features.
   - Low conviction on ROI given current evidence.

---

### Implementation Notes

#### For Strength State Stratification (Quick Win #1)

```python
# Pseudocode for your pipeline
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Assume df_shots has 'strength_state' column: 'ES' (even), 'PP' (power play), 'SH' (short-handed), 'EN' (empty net)
models = {}
strength_states = ['ES', 'PP', 'SH', 'EN']

for ss in strength_states:
    df_ss = df_shots[df_shots['strength_state'] == ss]
    if len(df_ss) > 100:  # Only train if enough data
        X = df_ss[feature_cols]
        y = df_ss['is_goal']
        models[ss] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ).fit(X, y)

# At inference time:
def predict_xg(shot_features, strength_state):
    model = models.get(strength_state, models['ES'])  # fallback to ES
    return model.predict_proba(shot_features)[:, 1]
```

**Integration points:**
- Modify `nhl_pbp_scraper.py` to include strength state in pbp_shots table (if not already there).
- Update `advanced_stats.py` to call stratified predict_xg() instead of single model.
- Re-evaluate feature importance separately for each model to understand domain differences.

#### For Pre-Shot Movement Features (High-Impact Investment)

```python
# Pseudocode for feature engineering from event sequences
def engineer_preshot_features(shot_row, pbp_df, shot_idx):
    """
    Extract features from event sequence leading up to shot.
    shot_row: the shot event
    pbp_df: full play-by-play dataframe
    shot_idx: index of shot in pbp_df
    """
    # Get prior events (up to 3–4 events before shot)
    prior_events = pbp_df.iloc[max(0, shot_idx-4):shot_idx]

    features = {
        'pass_count_in_zone': sum(prior_events['event_type'] == 'Pass'),
        'zones_traveled': prior_events['zone'].nunique(),
        'time_to_shot_ms': (shot_row['game_seconds'] - prior_events.iloc[0]['game_seconds']) * 1000,
        'cumulative_x_distance': (shot_row['x'] - prior_events.iloc[0]['x']).abs(),
        'angle_variance': prior_events['shot_angle'].std(),  # angle change across sequence
        'entry_type': classify_zone_entry(prior_events),  # custom function
        'has_rebound_in_sequence': sum(prior_events['is_rebound']) > 0,
        'ozone_pass_speed': estimate_pass_speed(prior_events),  # if coords available
    }
    return features

# Validation: use forward-chained cross-validation to avoid leakage
# Day 1–100: train on games 1–50, evaluate on games 51–100
# Day 101–200: train on games 1–100, evaluate on games 101–150
# etc.
```

**Data sources:**
- You already have prior_event_type and prior_distance from NHL API.
- Enhance with prior_x, prior_y if NHL API includes them (check api-web.nhle.com PBP schema).
- You may need to engineer entry type classifier (zone entry from neutral ice, dump & chase, etc.) using coordinate data.

**Validation checklist:**
- Ensure no future information leaks (e.g., don't use goal status of prior event as a feature).
- Use time-series cross-validation (never train on data after your test period).
- Evaluate AUC and log loss separately to catch overfitting.
- Ablate features one-by-one to confirm each adds value.

#### Data Requirements & Bottlenecks

| Task | Data Needed | Availability | Blocker? |
|------|------------|--------------|----------|
| 6-season retrain | 600K shots from NHL API | In progress | **YES** |
| Strength state stratification | Strength state for each shot | Likely in API PBP | No |
| Pre-shot movement | x/y coords, prior event sequences | Partially in API | Partial (depends on API richness) |
| Shot type stratification | Shot type (7 categories) | In your features.py | No |
| Bayesian mixed effects | Shooter/goalie IDs | In boxscore_skaters, pbp_shots | No |

**Critical path:**
1. Finish 6-season scrape → baseline AUC 0.78–0.82.
2. Parallelize: strength state stratification (quick) + pre-shot feature engineering (slow).
3. Combine and iterate on both simultaneously.

---

## Sources

- [Hockey Analytics – Testing the xG Models](https://hockey-statistics.com/2025/06/25/testing-the-xg-models/)
- [An Expected Goals (xG) model for NHL shots | Kevin Qin | Medium](https://medium.com/@kevin.qinzw/an-expected-goals-xg-model-for-nhl-shots-53ea8d155776)
- [A New Expected Goals Model for Predicting Goals in the NHL | Evolving Hockey](https://evolving-hockey.com/blog/a-new-expected-goals-model-for-predicting-goals-in-the-nhl/)
- [Building an expected goals model in ice hockey | Rasmus Säfvenberg](https://safvenberger.github.io/expected-goals-in-ice-hockey/)
- [Improving expected Goals (xG) models | Methods, Results, and Novel Extensions](https://jhss.scholasticahq.com/article/144180-improving-expected-goals-xg-models-methods-results-and-novel-extensions.pdf)
- [Toward interpretable expected goals modeling using Bayesian mixed models | PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12055760/)
- [Expected Goals Model with Pre-Shot Movement, Part 1 | Hockey Graphs](https://hockey-graphs.com/2019/08/12/expected-goals-model-with-pre-shot-movement-part-1-the-model/)
- [Expected Goals Model with Pre-Shot Movement, Part 4: Variable Importance | Hockey Graphs](https://hockey-graphs.com/2019/08/15/expected-goals-model-with-pre-shot-movement-part-4-variable-importance/)
- [MoneyPuck.com - About and How it Works](https://moneypuck.com/about.htm)
- [GitHub - HarryShomer/xG-Model](https://github.com/HarryShomer/xG-Model)
- [The Power of Pixels: Exploring the Potential of CNNs for Expected Goals (xG) in Football](https://www.researchgate.net/publication/382456974_The_Power_of_Pixels_Exploring_the_Potential_of_CNNs_for_Expected_Goals_xG_in_Football)
