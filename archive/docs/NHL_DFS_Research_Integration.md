# NHL DFS Research Integration Guide

## Executive Summary

This document consolidates insights from 7 academic papers on NHL analytics into actionable features and model improvements for our DFS projection system. The research spans 2012-2025 and covers expected goals (xG), goalie evaluation, player skill adjustment, and explainable AI.

**Current System Performance:**
- Overall MAE: 4.5 (skaters), 7.15 (goalies)
- Ownership MAE: 4.14% (Ridge), 3.92% (XGBoost)
- Boom Model AUC: 0.638

**Target Improvements:**
- Goalie MAE: <6.0 (from 7.15)
- Skater MAE: <4.0 (from 4.5)
- Better differentiation of elite vs average players

---

## Research Papers Reviewed

| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 1 | Barinberg - xG Modeling (Ramapo) | 2023 | XGBoost for shot quality, feature engineering |
| 2 | Macdonald - Weighted Shots (West Point) | 2012 | Adjusted save %, shooter fatigue, logistic regression |
| 3 | Macdonald - Expected Goals (Sloan) | 2012 | xG model, Corsi/Fenwick alternatives, adjusted +/- |
| 4 | Naples et al. - Goalie Analytics (SMU) | 2018 | Fenwick save %, shot type control, reliability testing |
| 5 | Noel - Skill-Adjusted xG (arXiv) | 2025 | Shooter/goalie skill decomposition, LightGBM |
| 6 | Evolving Hockey - xG Model | 2018 | XGBoost, 4 separate strength-state models, prior events |
| 7 | Pitassi & Cohen - Explainable AI (Waterloo) | 2024 | Three-phase trust framework, stakeholder explanations |

---

## Part 1: Feature Engineering Insights

### 1.1 Shot Quality Features (From Barinberg, Macdonald, Evolving Hockey)

These features predict whether a shot becomes a goal (xG). For DFS, players who generate high-xG shots score more points.

```python
SHOT_QUALITY_FEATURES = {
    # Primary xG predictors (from Barinberg)
    'shot_distance': {
        'source': 'NHL API play-by-play',
        'importance': 'CRITICAL - #1 predictor',
        'notes': 'Closer shots = higher xG. Log transform recommended.'
    },
    'shot_angle': {
        'source': 'Calculated from x,y coordinates',
        'importance': 'CRITICAL - #2 predictor',
        'notes': '0° = straight on (high xG), 90° = side (low xG)'
    },
    'shot_type': {
        'source': 'NHL API',
        'importance': 'HIGH',
        'values': ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected', 'wrap-around'],
        'xG_ranking': 'tip-in > deflected > wrist > snap > slap > backhand > wrap-around'
    },
    
    # Rebound/sequence features (from Macdonald)
    'is_rebound': {
        'source': 'Calculated: shot within 2 seconds of prior shot',
        'importance': 'VERY HIGH - 1.73x odds ratio',
        'notes': 'Rebounds are prime scoring chances'
    },
    'is_own_rebound': {
        'source': 'Same shooter as prior shot',
        'importance': 'MEDIUM - 0.59x odds ratio',
        'notes': 'Own rebounds slightly less dangerous than teammate rebounds'
    },
    'angle_change': {
        'source': 'Calculated from prior shot angle',
        'importance': 'HIGH for rebounds',
        'notes': 'Large angle change = goalie out of position'
    },
    
    # Prior event features (from Evolving Hockey)
    'prior_event_type': {
        'source': 'NHL API play-by-play',
        'importance': 'MEDIUM-HIGH',
        'values': ['shot', 'miss', 'block', 'faceoff', 'hit', 'giveaway', 'takeaway'],
        'notes': 'What happened before the shot matters'
    },
    'seconds_since_last_event': {
        'source': 'Calculated',
        'importance': 'MEDIUM',
        'notes': 'Quick sequences = higher xG'
    },
    'distance_from_last_event': {
        'source': 'Calculated from x,y coordinates',
        'importance': 'MEDIUM',
        'notes': 'Spatial context of play development'
    },
    
    # Fatigue features (from Macdonald)
    'shooter_time_on_ice': {
        'source': 'NHL API shift data',
        'importance': 'MEDIUM',
        'notes': 'Longer shifts = more tired = worse shots. 0.97x odds per second.'
    },
    'offensive_team_avg_toi': {
        'source': 'Calculated from shift data',
        'importance': 'LOW-MEDIUM',
        'notes': 'Team fatigue affects shot quality'
    },
    'defensive_team_avg_toi': {
        'source': 'Calculated from shift data',
        'importance': 'LOW',
        'notes': 'Tired defenders = better scoring chances'
    }
}
```

### 1.2 Game State Features (From Macdonald, Evolving Hockey)

```python
GAME_STATE_FEATURES = {
    # Score effects
    'score_differential': {
        'source': 'Live game data',
        'importance': 'MEDIUM-HIGH',
        'notes': 'Trailing teams shoot more/better. +1.03x odds when trailing.'
    },
    'score_state_bins': {
        'values': ['down_4+', 'down_3', 'down_2', 'down_1', 'tied', 'up_1', 'up_2', 'up_3', 'up_4+'],
        'notes': 'One-hot encode for non-linear effects'
    },
    
    # Strength state (CRITICAL from Evolving Hockey)
    'strength_state': {
        'importance': 'CRITICAL',
        'recommendation': 'Build SEPARATE models for each state',
        'states': {
            'EV': '5v5, 4v4, 3v3 - most data, baseline model',
            'PP': '5v4, 5v3, 4v3 - higher scoring rate',
            'SH': '4v5, 3v5, 3v4 - lower scoring rate',
            'EN': 'Empty net - very different dynamics'
        }
    },
    
    # Game context
    'game_period': {
        'importance': 'MEDIUM',
        'notes': '3rd period = more desperation, different strategies'
    },
    'game_seconds': {
        'importance': 'LOW-MEDIUM',
        'notes': 'End of period effects'
    },
    'is_home': {
        'importance': 'LOW',
        'notes': 'Small home ice advantage. -0.98x odds ratio (away slightly worse).'
    }
}
```

### 1.3 Skill Adjustment Features (From Noel 2025)

**Key Innovation:** Decompose player skill into THREE components:

```python
SKILL_DECOMPOSITION_FEATURES = {
    # Overall skill
    'shooter_goals_above_expected': {
        'formula': 'sum(weighted_goals) - sum(weighted_xG)',
        'notes': 'Career over/underperformance vs xG. Weight recent games more.'
    },
    'shooter_talent_ratio': {
        'formula': 'sum(weighted_goals) / sum(weighted_xG)',
        'notes': '>1.0 = elite finisher, <1.0 = poor finisher'
    },
    'goalie_saves_above_expected': {
        'formula': 'sum(weighted_xG_against) - sum(weighted_goals_against)',
        'notes': 'Positive = better than average goalie'
    },
    'goalie_talent_ratio': {
        'formula': 'sum(weighted_xG_against) / sum(weighted_goals_against)',
        'notes': '>1.0 = elite goalie, <1.0 = below average'
    },
    
    # Locational skill (zone-specific)
    'shooter_locational_skill': {
        'method': 'Bin ice into 9 zones, calculate talent ratio per zone',
        'notes': 'Some players excel from specific areas'
    },
    'goalie_locational_skill': {
        'method': 'Same binning, track save % by zone',
        'notes': 'Goalies may have weak spots (glove high, five hole, etc.)'
    },
    
    # Situational skill (Gower distance)
    'shooter_situational_skill': {
        'method': 'Find similar historical shots using Gower distance on: shot type, distance, angle, rebound, strength state',
        'notes': 'How does player perform in similar situations?'
    },
    'goalie_situational_skill': {
        'method': 'Same approach for saves',
        'notes': 'How does goalie perform against similar shots?'
    },
    
    # Combined "True" skill
    'true_shooter_skill': {
        'formula': 'overall_skill + locational_skill + situational_skill',
        'notes': 'Comprehensive skill measure'
    },
    'true_goalie_skill': {
        'formula': 'overall_skill + locational_skill + situational_skill',
        'notes': 'Comprehensive skill measure'
    }
}

# Weighting scheme for historical shots
SHOT_WEIGHTING = {
    'method': 'Linear decay by recency',
    'example': 'Shot 1 (oldest): weight 0.2, Shot 5 (newest): weight 1.0',
    'rationale': 'Recent performance more predictive than distant past'
}
```

**Key Finding from Noel:** Skill adjustment improves model by 1-5%, with **HIGH-SKILL players seeing 5% improvement**. This means skill adjustment matters most for predicting elite players.

### 1.4 Goalie-Specific Features (From Naples, Macdonald)

```python
GOALIE_FEATURES = {
    # Standard metrics (from NHL API)
    'save_pct': {
        'importance': 'MEDIUM',
        'notes': 'Weakly repeatable, high variance'
    },
    'goals_against_average': {
        'importance': 'MEDIUM',
        'notes': 'Affected by team defense'
    },
    
    # Adjusted metrics (from Macdonald)
    'adjusted_save_pct': {
        'formula': 'league_avg_save_pct + (actual_save_pct - expected_save_pct)',
        'importance': 'HIGHER than raw save %',
        'notes': 'Adjusts for quality of shots faced'
    },
    'expected_save_pct': {
        'formula': '1 - (weighted_shots_against / shots_against)',
        'notes': 'Based on shot quality faced'
    },
    'goals_saved_above_expected': {
        'formula': 'expected_goals_against - actual_goals_against',
        'importance': 'HIGH',
        'notes': 'Defense-independent goalie rating'
    },
    
    # Alternative metrics (from Naples - MORE STABLE)
    'fenwick_save_pct': {
        'formula': 'saves / (shots_on_goal + missed_shots)',
        'importance': 'HIGHER reliability than standard save %',
        'notes': 'Including misses increases sample size by ~30%, improves stability by 15%'
    },
    'save_pct_by_shot_type': {
        'notes': 'Track separately for wrist, slap, backhand, etc.',
        'importance': 'MEDIUM-HIGH'
    },
    
    # Opponent quality
    'opponent_xG_per_game': {
        'source': 'Natural Stat Trick or MoneyPuck',
        'importance': 'HIGH',
        'notes': 'Facing high-xG team = more goals against expected'
    },
    'opponent_shots_per_game': {
        'source': 'NHL API',
        'importance': 'MEDIUM',
        'notes': 'More shots = more save opportunities (DFS positive) but more GA risk'
    },
    'opponent_shooting_pct': {
        'source': 'NHL API',
        'importance': 'MEDIUM',
        'notes': 'High shooting % teams more dangerous'
    }
}
```

**Key Finding from Naples:** 
> "We find save percentage to be both a weakly repeatable skill and predictor of future performance... Fenwick save percentage improves reliability by ~15%."

### 1.5 Power Play Features (From Pitassi, Macdonald)

```python
PP_FEATURES = {
    # Player attributes (from Pitassi - EA Sports ratings as proxy)
    'offensive_awareness': {
        'importance': '#1 PP predictor',
        'notes': 'Players who find open ice on PP'
    },
    'puck_control': {
        'importance': '#2 PP predictor',
        'notes': 'Maintaining possession on PP'
    },
    'faceoff_ability': {
        'importance': '#3 PP predictor',
        'notes': 'Winning draws starts PP possessions'
    },
    'passing': {
        'importance': 'HIGH',
        'notes': 'PP success requires crisp passing'
    },
    'wristshot_accuracy': {
        'importance': 'MEDIUM-HIGH',
        'notes': 'Most PP goals are wrist shots'
    },
    
    # PP deployment
    'pp1_flag': {
        'importance': 'CRITICAL',
        'notes': 'PP1 gets ~70% of PP time, vast majority of PP points'
    },
    'pp_toi_per_game': {
        'importance': 'HIGH',
        'notes': 'More PP time = more opportunities'
    },
    'pp_points_per_60': {
        'importance': 'HIGH',
        'notes': 'Historical PP production rate'
    },
    
    # Team PP context
    'team_pp_pct': {
        'importance': 'MEDIUM-HIGH',
        'notes': 'Good PP units convert more often'
    },
    'opponent_pk_pct': {
        'importance': 'MEDIUM',
        'notes': 'Weak PK = more PP opportunities'
    }
}
```

---

## Part 2: Model Architecture Recommendations

### 2.1 Strength-State Specific Models (From Evolving Hockey)

**Key Recommendation:** Build FOUR separate xG/projection models:

```python
MODEL_ARCHITECTURE = {
    'even_strength': {
        'training_data': '5v5, 4v4, 3v3 shots',
        'seasons': '7 seasons (most recent)',
        'features': 'Full feature set',
        'notes': 'Largest dataset, most robust'
    },
    'power_play': {
        'training_data': '5v4, 5v3, 4v3 shots',
        'seasons': '7 seasons',
        'features': 'Full set + PP-specific features',
        'notes': 'Higher base xG, different dynamics'
    },
    'shorthanded': {
        'training_data': '4v5, 3v5, 3v4 shots',
        'seasons': '10 seasons (less data)',
        'features': 'Simplified set',
        'notes': 'Less data, focus on core features'
    },
    'empty_net': {
        'training_data': 'EN shots only',
        'seasons': '10 seasons (rare events)',
        'features': 'Distance, angle, time remaining',
        'notes': 'Very different from normal play'
    }
}
```

### 2.2 XGBoost Hyperparameters (From Evolving Hockey, Barinberg)

```python
XGBOOST_PARAMS = {
    'even_strength': {
        'max_depth': 6,
        'eta': 0.068,
        'gamma': 0.12,
        'subsample': 0.78,
        'colsample_bytree': 0.76,
        'min_child_weight': 5,
        'max_delta_step': 5,
        'n_rounds': 189,
        'early_stopping': 25
    },
    'notes': [
        'Use 5-fold cross-validation for tuning',
        'AUC is better metric than log loss for classification',
        'Early stopping prevents overfitting'
    ]
}
```

### 2.3 Model Stacking for Skill Adjustment (From Noel)

```python
MODEL_STACKING_APPROACH = {
    'layer_1': {
        'name': 'Base xG Model',
        'input': 'Shot features (distance, angle, type, etc.)',
        'output': 'Base xG probability',
        'notes': 'Standard xG model without skill adjustment'
    },
    'layer_2': {
        'name': 'Skill-Adjusted xG Model',
        'input': 'Base xG + shooter skill features + goalie skill features',
        'output': 'Skill-adjusted xG probability',
        'notes': 'Second model learns skill effects'
    },
    'rationale': 'Separating base xG from skill allows cleaner feature engineering'
}
```

---

## Part 3: Explainability Framework (From Pitassi)

### 3.1 Three-Phase Trust Framework

```python
EXPLAINABILITY_FRAMEWORK = {
    'phase_1': {
        'name': 'Stakeholder-Specific Explanations',
        'actions': [
            'Identify who uses the model (you, for DFS decisions)',
            'Determine explanation needs (quick decisions vs deep analysis)',
            'Create appropriate visualizations'
        ],
        'outputs': {
            'daily_decisions': 'Simple dashboard with top plays',
            'model_tuning': 'SHAP plots, feature importance',
            'bankroll_tracking': 'ROI charts, win rate graphs'
        }
    },
    'phase_2': {
        'name': 'Model-Appropriate Explainability',
        'actions': [
            'Identify model type (XGBoost = black box)',
            'Select appropriate XAI methods',
            'Generate explanations'
        ],
        'methods_for_xgboost': [
            'SHAP values (feature contribution to prediction)',
            'Partial Dependence Plots (feature effects)',
            'Feature Importance (overall importance)'
        ]
    },
    'phase_3': {
        'name': 'Model Reliability Assessment',
        'metrics': {
            'stability': 'Cross-validation variance (should be low)',
            'discrimination': 'Prediction range vs actual range (should be similar)',
            'independence': 'Does model add value vs baseline?'
        }
    }
}
```

### 3.2 SHAP Integration (Already in our models)

```python
# Example SHAP analysis for XGBoost ownership model
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary plot - shows feature importance and direction
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Dependence plot - shows single feature effect
shap.dependence_plot('Proj_pctile', shap_values, X_test)

# Force plot - explains single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

---

## Part 4: Implementation Roadmap

### 4.1 Phase 1: Quick Wins (1-2 days)

These can be implemented immediately with existing data:

| Feature | Source | Expected Impact |
|---------|--------|-----------------|
| Score differential | DK salary file + Vegas | Medium |
| PP1 flag (already have) | DailyFaceoff | Already implemented ✅ |
| Opponent implied total | Vegas | Already implemented ✅ |
| Goalie win probability | Calculated from spread | Medium |
| Days rest (goalies) | Schedule data | Medium |
| Back-to-back flag | Schedule data | Medium |

### 4.2 Phase 2: Goalie Model Improvements (1 week)

```python
GOALIE_MODEL_V2_FEATURES = [
    # Already have
    'Salary', 'FC Proj', 'TeamTotal', 'GameTotal', 'Ownership',
    'is_chalk_heavy', 'is_balanced', 'is_contrarian',
    
    # Add from NHL API
    'goalie_save_pct',
    'goalie_gaa', 
    'goalie_wins',
    'goalie_gp',
    'goalie_win_pct',
    'goalie_shutout_pct',
    
    # Add opponent quality
    'opp_goals_per_game',
    'opp_shots_per_game',
    'opp_shooting_pct',
    'opp_pp_pct',
    
    # Add derived
    'expected_GA',
    'expected_shots',
    'expected_saves',
    'win_prob_proxy',
    
    # Add context
    'is_home',
    'is_confirmed_starter',
]
```

### 4.3 Phase 3: Skater xG Integration (2-3 weeks)

This requires play-by-play data processing:

```python
SKATER_XG_FEATURES = [
    # Individual shot quality (aggregate to player level)
    'avg_shot_distance',
    'avg_shot_angle',
    'shot_type_distribution',
    'rebound_shot_pct',
    
    # Skill metrics
    'goals_above_expected',
    'shooting_talent_ratio',
    'xG_per_60',
    'actual_goals_per_60',
    
    # PP specific
    'pp_xG_per_60',
    'pp_shot_volume',
    
    # Opponent adjustment
    'opp_xGA_per_60',
    'opp_save_pct',
]
```

### 4.4 Phase 4: Full Skill-Adjusted Model (1 month)

Implement Noel's three-component skill decomposition:

```python
FULL_SKILL_MODEL = {
    'step_1': 'Build base xG model on all historical shots',
    'step_2': 'Calculate xG for all shots in dataset',
    'step_3': 'Compute overall skill (goals above expected)',
    'step_4': 'Compute locational skill (9-zone breakdown)',
    'step_5': 'Compute situational skill (Gower distance)',
    'step_6': 'Build skill-adjusted model using all components',
    'step_7': 'Validate on held-out season'
}
```

---

## Part 5: Key Metrics and Thresholds

### 5.1 xG Model Performance Benchmarks

| Metric | Target | Source |
|--------|--------|--------|
| ROC AUC | ≥0.76 | Macdonald (0.764) |
| Log Loss | ≤0.185 | Evolving Hockey |
| Brier Score | ≤0.085 | Noel |

### 5.2 Projection Model Performance Benchmarks

| Position | Current MAE | Target MAE | Improvement |
|----------|-------------|------------|-------------|
| Skaters | 4.5 | <4.0 | 11% |
| Goalies | 7.15 | <6.0 | 16% |
| All | 4.8 | <4.2 | 12% |

### 5.3 Ownership Model Performance Benchmarks

| Metric | Current | Target |
|--------|---------|--------|
| MAE | 3.92% (XGB) | <3.5% |
| AUC | 0.82 | >0.85 |
| Spearman | 0.725 | >0.75 |

---

## Part 6: Data Sources Required

### 6.1 Already Available

| Data | Source | Status |
|------|--------|--------|
| Player salaries | DraftKings CSV | ✅ |
| Vegas lines | The Odds API | ✅ |
| Line combinations | DailyFaceoff | ✅ |
| Basic stats | NHL API | ✅ |
| Edge tracking | NHL Edge API | ✅ |
| Injuries | MoneyPuck | ✅ |
| xG (team level) | Natural Stat Trick | ✅ |

### 6.2 Need to Integrate

| Data | Source | Priority |
|------|--------|----------|
| Play-by-play shots | NHL API | HIGH |
| Goalie season stats | NHL API (stats endpoint) | HIGH |
| Team shooting stats | NHL API | MEDIUM |
| Shot coordinates | NHL API play-by-play | MEDIUM |
| Shift data | NHL API | LOW |

### 6.3 API Endpoints

```python
NHL_API_ENDPOINTS = {
    'goalie_stats': 'https://api.nhle.com/stats/rest/en/goalie/summary?cayenneExp=seasonId={season}',
    'team_stats': 'https://api.nhle.com/stats/rest/en/team/summary?cayenneExp=seasonId={season}',
    'play_by_play': 'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play',
    'shifts': 'https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}',
}
```

---

## Part 7: Summary of Key Findings

### 7.1 Most Important Discoveries

1. **Shot Distance and Angle are #1 and #2 predictors of goals** (all papers agree)

2. **Rebounds have 1.73x higher goal probability** (Macdonald)

3. **Skill adjustment improves predictions by 1-5%**, with highest improvement for elite players (Noel)

4. **Fenwick Save % is 15% more stable than regular Save %** (Naples)

5. **Separate models by strength state** significantly improves accuracy (Evolving Hockey)

6. **Prior event type matters** - what happened before the shot affects xG (Evolving Hockey)

7. **Offensive Awareness is #1 PP attribute** (Pitassi)

8. **Trailing teams shoot better** - +1.03x odds when behind (Macdonald)

### 7.2 DFS-Specific Implications

| Finding | DFS Implication |
|---------|-----------------|
| Rebounds = high xG | Target players who crash the net |
| PP1 dominates | PP1 players are worth the chalk |
| Trailing teams shoot more | Target skaters on slight underdogs |
| Goalie save % is noisy | Don't overweight recent goalie performance |
| Skill matters for elites | Trust elite players' projections more |
| Low game totals favor goalies | Target goalies in <5.5 total games |

### 7.3 Model Improvement Priority

1. **Goalie Model** - Biggest MAE gap (7.15 vs target <6.0)
2. **Boom/Bust Model** - Low-owned booms are 6x more valuable
3. **Ownership Model** - XGBoost already beating Ridge
4. **Skater xG Integration** - Longer term, highest ceiling

---

## Appendix A: Feature Importance Rankings

### From XGBoost Ownership Model (Our Data)

1. Proj_pctile (0.15)
2. PP1_x_TeamTotal (0.075)
3. Slate_size (0.06)
4. is_balanced (0.06)
5. Proj_sq (0.045)

### From Macdonald xG Model

1. Distance (-0.054 per foot)
2. Shot type (slap 4.10x, wrist 2.60x)
3. Rebound (1.73x)
4. Angle (-0.017 per degree)
5. Strength state (PP53 2.53x)

### From Pitassi PP Model

1. Offensive Awareness
2. Puck Control
3. Faceoffs
4. Passing
5. Wristshot Accuracy

---

## Appendix B: Code Templates

### B.1 Skill Calculation Template

```python
def calculate_shooter_skill(player_shots_df, weight_recent=True):
    """
    Calculate shooter skill metrics from historical shot data.
    
    Parameters:
    -----------
    player_shots_df: DataFrame with columns [game_date, xG, is_goal]
    weight_recent: If True, weight recent shots more heavily
    
    Returns:
    --------
    dict with overall_skill, locational_skill, situational_skill
    """
    # Sort by date
    df = player_shots_df.sort_values('game_date')
    
    # Linear weights (most recent = 1.0, oldest = 0.2)
    n = len(df)
    if weight_recent and n > 1:
        weights = np.linspace(0.2, 1.0, n)
    else:
        weights = np.ones(n)
    
    # Weighted sums
    weighted_goals = (df['is_goal'] * weights).sum()
    weighted_xG = (df['xG'] * weights).sum()
    
    # Overall skill
    goals_above_expected = weighted_goals - weighted_xG
    talent_ratio = weighted_goals / weighted_xG if weighted_xG > 0 else 1.0
    
    return {
        'goals_above_expected': goals_above_expected,
        'talent_ratio': talent_ratio,
        'sample_size': n
    }
```

### B.2 Adjusted Save % Template

```python
def calculate_adjusted_save_pct(goalie_df, league_avg_save_pct=0.905):
    """
    Calculate adjusted save percentage per Macdonald methodology.
    
    Parameters:
    -----------
    goalie_df: DataFrame with columns [shots_against, saves, weighted_shots_against]
    league_avg_save_pct: League average save percentage
    
    Returns:
    --------
    float: Adjusted save percentage
    """
    actual_save_pct = goalie_df['saves'].sum() / goalie_df['shots_against'].sum()
    expected_save_pct = 1 - (goalie_df['weighted_shots_against'].sum() / 
                             goalie_df['shots_against'].sum())
    
    adjusted_save_pct = league_avg_save_pct + (actual_save_pct - expected_save_pct)
    
    return adjusted_save_pct
```

### B.3 Fenwick Save % Template

```python
def calculate_fenwick_save_pct(saves, shots_on_goal, missed_shots):
    """
    Calculate Fenwick save percentage (more stable than regular save %).
    
    Parameters:
    -----------
    saves: Number of saves
    shots_on_goal: Shots on goal
    missed_shots: Shots that missed the net
    
    Returns:
    --------
    float: Fenwick save percentage
    """
    total_attempts = shots_on_goal + missed_shots
    fenwick_save_pct = saves / total_attempts if total_attempts > 0 else 0.0
    return fenwick_save_pct
```

---

## Appendix C: References

1. Barinberg, M. (2023). "Shot Quality Modeling with XGBoost." Ramapo College MSDS Thesis.

2. Macdonald, B., Lennon, C., & Sturdivant, R. (2012). "Evaluating NHL Goalies, Skaters, and Teams Using Weighted Shots." arXiv:1205.1746.

3. Macdonald, B. (2012). "An Expected Goals Model for Evaluating NHL Teams and Players." MIT Sloan Sports Analytics Conference.

4. Naples, M., Gage, L., & Nussbaum, A. (2018). "Goalie Analytics: Statistical Evaluation of Context-Specific Goalie Performance Measures in the NHL." SMU Data Science Review, 1(2).

5. Noel, J.T.P. (2025). "Expected by Whom? A Skill-Adjusted Expected Goals Model for NHL Shooters and Goaltenders." arXiv:2511.07703.

6. Younggren, J. & Younggren, L. (2018). "A New Expected Goals Model for Predicting Goals in the NHL." Evolving Hockey.

7. Pitassi, M. & Cohen, R. (2024). "Advancing NHL Analytics through Explainable AI." Canadian Conference on Artificial Intelligence.

---

*Document Version: 1.0*
*Last Updated: February 4, 2026*
*Author: Claude (compiled from research)*
