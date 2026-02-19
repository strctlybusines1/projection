# NHL DFS GPP Optimization Plan

> **This document = Strategic Framework & Analysis**
> For operational steps, see `DAILY_WORKFLOW.md`

---

## Core Philosophy: Stacks > Individual Projections

**The fundamental insight:** In NHL DFS, goals create correlated fantasy points. When a player scores, 1-2 linemates typically get assists. This means:
- Individual player projections have limited value for GPPs
- **Stack correlation** is the primary driver of ceiling outcomes
- **Conditional probability** of a lineup hitting its ceiling determines GPP success
- **Ownership leverage** must be weighed against ceiling probability
- **Tournament structure** dictates optimal risk tolerance

---

## 1. STACK-BASED FRAMEWORK

### Why Stacks Dominate NHL GPPs

When analyzing winning lineups from recent contests:

**Jan 24, 2026 - Winner (236.3 pts):**
- EDM Stack: McDavid (52.8) + Bouchard (64.1)
- MTL/BOS pieces creating secondary correlation
- Key: Multiple linemates hitting simultaneously

**Jan 25, 2026 - Winner (147.6 pts):**
- FLA correlation: Reinhart + Verhaeghe + Sennecke
- Defensive correlation: Sanderson + Weegar
- Key: Stacking the right game environment

### Stack Types to Model

| Stack Type | Description | Correlation Strength |
|------------|-------------|---------------------|
| Line Stack (3-man) | C + LW + RW from same line | Very High |
| Line Stack (2-man) | Any 2 forwards from same line | High |
| PP1 Stack | Players on first power play unit | High (event-dependent) |
| Team Stack (4+) | 4+ players from same team | Ceiling maximizer |
| D + Goalie | Defenseman + opposing goalie | Negative correlation |
| Bring-back | Stack + 1 opposing player | Hedged correlation |

### Correlation Coefficients to Track

From historical game logs, calculate:
```
P(Teammate_Assist | Player_Goal) = ~0.85 (at least one)
P(Both_Linemates_Assist | Player_Goal) = ~0.40 (2 assists)
P(PP_Correlation | PP_Goal) = ~0.90 (power play events)
```

**Priority: Build correlation matrix from historical logs showing:**
- Which players appear on the same line
- Goal/assist co-occurrence rates
- PP unit assignments
- Shift overlap percentage

---

## 2. CONDITIONAL PROBABILITY OF CEILING

### The Key Question

For any given lineup, what is:
```
P(Lineup hits ceiling) = P(Stack1 hits) * P(Stack2 hits) * P(Goalie hits) * P(Correlation bonus)
```

### Ceiling Triggers (What Creates 200+ pt Lineups)

1. **Multi-goal game from stack** - When your stacked players score 2+ goals each
2. **PP dominance** - Power play units clicking (3+ PPG in a game)
3. **High-event game** - 7+ total goals in a game your stack is in
4. **Goalie ceiling** - 35+ saves + win + shutout potential
5. **Bonus stacking** - Multiple players hitting 3+ goal/assist games

### Modeling Ceiling Probability

For each player:
```python
# Current approach (wrong for GPPs)
expected_pts = projection  # Single point estimate

# New approach (ceiling-focused)
ceiling_probability = P(player scores 2+ goals OR 4+ points)
ceiling_value = expected_pts_given_ceiling_hit
downside_value = expected_pts_given_floor
```

For stacks:
```python
# Stack ceiling probability
stack_ceiling_prob = (
    P(primary_scorer_pops) *
    P(correlation_hits | primary_pops) *
    P(game_environment_favorable)
)
```

### Historical Log Analysis Required

Build database tracking:
- Game-by-game ceiling outcomes (which players hit 3x+ their baseline)
- Correlation events (how often linemates share in ceiling games)
- Game environment predictors (Vegas totals, pace, etc.)

---

## 3. OWNERSHIP LEVERAGE

### The Leverage Equation

```
EV(Lineup) = P(Ceiling) * (1 - Field_Exposure) * Payout_Multiple
```

A lineup with:
- 5% ceiling probability at 2% ownership is BETTER than
- 8% ceiling probability at 15% ownership

### Ownership Buckets

| Ownership | Strategy | When to Use |
|-----------|----------|-------------|
| 0-5% (Contrarian) | Max leverage, need strong thesis | Large fields, top-heavy payouts |
| 5-15% (Moderate) | Balanced EV | Standard GPPs |
| 15-25% (Chalk) | Only if ceiling probability is elite | Small fields, flatter payouts |
| 25%+ (Mega-chalk) | Avoid unless forced correlation | Cash games only |

### Ownership Projection

Build model to predict ownership based on:
- Recent performance (hot streaks drive ownership)
- Price changes (drops = ownership spikes)
- Narrative (back from injury, revenge game, etc.)
- Line combos announced (confirmed PP1 = ownership spike)
- Vegas lines (implied totals, spreads)

---

## 4. TOURNAMENT STRUCTURE CONSIDERATIONS

### Contest Types and Optimal Strategy

| Contest Type | Max Entries | Risk Profile | Strategy |
|--------------|-------------|--------------|----------|
| Single Entry | 1 | Lower variance | Balanced ceiling + floor, moderate leverage |
| 3-Max | 3 | Medium variance | 1 safe, 2 contrarian builds |
| 5-Max | 5 | Medium-high | Diversified correlations |
| 20-Max | 20 | High variance | Cover multiple game stacks |
| Mass Multi (150+) | 150+ | Max variance | Ceiling-only, max leverage on each |

### Payout Structure Analysis

**Key metrics to extract from each contest:**
- Total entries
- First place prize
- Min-cash line (typically top 20-25%)
- Top 10% payout
- Top 1% payout

**Strategy implications:**
```
Top-Heavy Payout (1st = 20%+ of pool):
  -> Max leverage, unique stacks, accept higher bust rate

Flat Payout (1st = 5-10% of pool):
  -> Balanced approach, don't over-leverage
  -> Multiple entries should diversify rather than differentiate
```

### Entry Allocation Framework

For a 20-max contest:
```
Core Stacks (8-10 entries): Your highest conviction game stacks
Secondary Stacks (6-8 entries): Alternative correlations, bring-backs
Leverage Plays (2-4 entries): Contrarian builds, anti-chalk
```

---

## 5. CEILING GAME IDENTIFICATION (THE CORE PROBLEM)

The key question: **Which team/line will over-deliver tonight?**

Both Jan 22 and Jan 23 winners (8-game slates each) had CONVICTION on specific teams that exploded:
- Jan 22: NSH (5 players = 92.2 pts) + MIN Line 1 (68 pts)
- Jan 23: VGK Line 1 (62.9 pts) + NJD Line 4 (57.9 pts)

### Potential Predictive Signals to Research

**1. Mean Regression**
- Players/teams due for breakout after cold streaks
- Track: rolling 5-game avg vs season avg
- Signal: High-skill player with recent underperformance = ceiling candidate

**2. Goalie Vulnerabilities**
- Backup goalies starting (often not priced in)
- Goalies on back-to-backs (fatigue)
- Goalies with poor recent sv% (regression or continued struggle?)
- Track: Goalie confirmed starter + recent performance

**3. Vegas Line Analysis**
- High implied team totals (3.5+ goals)
- Line movement (sharp money indicators)
- Total movement + team spread combination
- Track: Opening vs closing lines, where money is going

**CRITICAL INSIGHT FROM CROSS-SLATE ANALYSIS:**

```
JANUARY 22 - Vegas Rank vs Actual Stack Rank:
Team   Vegas   VegasRk  Actual   ActualRk  Diff
CAR    3.84    #1       41.8     #11       -10  ← CHALK TRAP
EDM    3.77    #2       34.6     #14       -12  ← CHALK TRAP
MIN    3.64    #3       117.1    #1        +2   ← Winner secondary
NSH    2.96    #13      101.1    #2        +11  ← WINNER'S PRIMARY (5 players!)
BOS    2.96    #14      85.6     #4        +10  ← Beat expectations

JANUARY 23 - Vegas Rank vs Actual Stack Rank:
Team   Vegas   VegasRk  Actual   ActualRk  Diff
COL    4.18    #1       77.9     #7        -6   ← CHALK TRAP
NYR    3.25    #3       43.5     #16       -13  ← MASSIVE BUST
SEA    3.14    #7       48.7     #15       -8   ← Optimizer picked this!
PHI    3.25    #5       128.5    #1        +4   ← Exploded
VGK    3.25    #4       94.9     #2        +2   ← Winner's primary
NJD    3.00    #9       84.3     #3        +6   ← Winner's secondary (leverage)
```

**THE PATTERN:** Top Vegas teams often BUST. But blindly fading chalk isn't the answer either.

**THE REAL QUESTION:** What signals predict a team will OUTSCORE its expectation?

This applies to ALL teams regardless of Vegas rank:
- MIN was Vegas #3 and delivered #1 (117.1 pts) - outscored expectation
- PHI was Vegas #5 and delivered #1 (128.5 pts) - outscored expectation
- NSH was Vegas #13 and delivered #2 (101.1 pts) - outscored expectation
- VGK was Vegas #4 and delivered #2 (94.9 pts) - outscored expectation

The edge is identifying WHY these teams exceeded expectations - not their Vegas rank.

**RESEARCH AGENDA - Factors to analyze for predicting outperformance:**
- Opponent goalie situation (backup, recent struggles, back-to-back)
- Team's recent scoring vs their season average (mean regression)
- Special teams matchup (PP% vs opponent PK%)
- **Penalties volume**: Penalties taken and penalties allowed per team—direct drivers of PP/PK opportunity and why matchups matter (see 5a below).
- Pace/style matchup (fast team vs slow defensive team)
- Home/away splits and rest advantages
- Line combination changes (new lines not priced in)
- Individual player mean regression (star player due to break out)
- Historical matchup data (team performance vs specific opponents)

**4. Pace & Style Matchups**
- Fast teams vs slow/bad defensive teams
- High event teams (lots of shots, hits, blocks)
- Track: Corsi, Fenwick, expected goals against

**5. Special Teams Mismatches**
- Elite PP vs bad PK (or vice versa)
- PP% hot streaks (unsustainable but exploitable short-term)
- Track: Recent PP/PK performance + matchup

**5a. Penalties: Why Matchups Matter (Volume Drivers)**
- **Penalties taken** (per team, per 60 or per game): Drives how much PK time your team sees and how much PP time the *opponent* gets. High penalties taken = more PK time for your goalie/skaters, fewer 5v5 minutes.
- **Penalties allowed / drawn** (opponent’s penalties taken): Drives how much PP time *your* team gets. When your opponent takes a lot of penalties, your PP1 gets more opportunities → matchup boost for your PP skaters and a tougher environment for the opposing goalie.
- **Use in projections**: Expected PP/PK TOI and matchup strength should incorporate both sides—e.g. PP1 skater vs team that takes many penalties (more expected PP time); goalie vs team that draws many penalties (more PK, harder slate). Track: team penalties taken/60, opponent penalties taken/60 (or PIM/60, minor penalties/60 from NST or box-score source).

**6. Rest & Schedule**
- Teams off rest vs back-to-back opponents
- Travel situations (cross-country, altitude)
- Track: Days rest, travel distance, time zone changes

**7. Recent Form & Momentum**
- Teams on winning streaks (confidence)
- Players with multi-point game streaks
- Track: Last 5/10 game trends vs season baseline

**8. Line Combination Changes**
- New line combos not yet reflected in ownership
- Call-ups skating with stars
- Track: Daily line changes vs previous game

**9. Revenge/Narrative Games**
- Players vs former teams
- Milestone games (1000th game, etc.)
- Track: Player history with opponent

**10. Historical Matchup Data**
- Some players dominate specific teams
- Track: Player splits by opponent

### Research Priority

Build a **Ceiling Game Score** for each game that combines:
```
Ceiling_Score = (
    w1 * vegas_implied_total +
    w2 * goalie_vulnerability +
    w3 * pace_mismatch +
    w4 * rest_advantage +
    w5 * recent_form_delta +
    w6 * special_teams_edge
)
```

Then rank games by Ceiling_Score to identify where to concentrate stacks.

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: Data Infrastructure (Immediate)

- [ ] Build line combination tracker (daily scrape of team lines)
- [ ] Create power play unit tracker
- [ ] Develop correlation matrix from historical logs
- [ ] **Team penalty stats**: Penalties taken and penalties allowed (or drawn) per team (per 60 or per game) for matchup/expected PP-PK TOI (see 5a; NST or box-score source).
- [ ] Add Vegas lines integration (implied totals, spreads)
  - *Manual entry for now until free/cheap API found*
  - Track: Team implied totals, game totals, spreads
  - **Key metric: Actual goals vs implied total (to find predictive signals)**

### Phase 2: Stack Scoring (Week 1)

- [ ] Create stack-based projection model
- [ ] Calculate stack ceiling probabilities
- [ ] Build correlation bonus system
- [ ] Implement game environment scoring

### Phase 3: Ownership Model (Week 2)

- [ ] Collect historical ownership data
- [ ] Build ownership prediction model
- [ ] Create leverage scoring system
- [ ] Integrate into lineup builder

### Phase 4: Tournament Optimizer (Week 3)

- [ ] Contest structure analyzer
- [ ] Entry allocation optimizer
- [ ] Correlation diversification logic
- [ ] Max entry portfolio builder

---

## 6. DAILY WORKFLOW

**See `DAILY_WORKFLOW.md` for complete operational steps.**

The workflow integrates these strategic principles:
1. **Data sources first** - Verify lines, goalies, Vegas before projections
2. **Slate analysis** - Identify ceiling games using signals below
3. **Stack thesis** - Document conviction before running optimizer
4. **Leverage check** - Cross-reference ownership vs ceiling probability

---

## 7. METRICS TO TRACK (REVISED)

### Old Metrics (Cash Game Focus - De-prioritize)
- Individual player MAE
- Projection accuracy %
- Floor hit rate

### New Metrics (GPP Focus - Prioritize)

| Metric | Description | Target |
|--------|-------------|--------|
| Stack Hit Rate | % of stacks that score 2+ goals | Track by stack type |
| Ceiling Capture | % of ceiling games correctly identified | > 30% |
| Leverage Accuracy | Ownership prediction error | < 5% MAE |
| Correlation Bonus | Extra points from lineup correlation | Maximize |
| ROI by Contest Type | Profitability segmented by format | Positive in each |

---

## 8. EXISTING CODE INTEGRATION

### Keep from Current System

The individual projection model is still useful for:
- Setting baseline expectations
- Identifying value (projection vs. salary)
- Goalie win probability
- Floor estimates for cash games

### Enhance with Stack Logic

```python
# Example enhancement to projections.py

def calculate_stack_ceiling(players, correlation_matrix, game_env):
    """
    Calculate the ceiling probability for a stack of players.

    Args:
        players: List of player objects in the stack
        correlation_matrix: Historical correlation data
        game_env: Game environment (Vegas total, pace, etc.)

    Returns:
        ceiling_prob: Probability stack hits ceiling
        ceiling_value: Expected points if ceiling hits
    """
    base_ceiling_probs = [p.ceiling_probability for p in players]

    # Correlation boost
    correlation_factor = get_correlation_factor(players, correlation_matrix)

    # Game environment modifier
    env_modifier = get_environment_modifier(game_env)

    # Combined probability
    stack_ceiling_prob = (
        np.mean(base_ceiling_probs) *
        correlation_factor *
        env_modifier
    )

    return stack_ceiling_prob
```

---

## 9. BACKTEST DATA REQUIREMENTS

### Historical Data to Collect

1. **Game logs** - Every player's game-by-game fantasy output
2. **Line combinations** - Who played with whom each game
3. **PP units** - Power play time by unit
4. **Ownership data** - Historical contest ownership percentages
5. **Contest results** - Winning lineup compositions
6. **Vegas lines** - Implied totals, spreads for each game

### Correlation Analysis

From game logs, calculate:
```sql
-- Example: Find goal/assist correlation between linemates
SELECT
    p1.player_name as scorer,
    p2.player_name as assister,
    COUNT(*) as co_occurrences,
    COUNT(*) / SUM(p1.goals) as correlation_rate
FROM game_events e1
JOIN game_events e2 ON e1.game_id = e2.game_id AND e1.event_id = e2.event_id
WHERE e1.event_type = 'goal' AND e2.event_type = 'assist'
GROUP BY p1.player_name, p2.player_name
```

---

## 10. SAMPLE CONTEST ANALYSIS

### Anatomy of a Winning Lineup (Jan 24, 236.3 pts)

```
Position | Player | FPTS | Notes
---------|--------|------|-------
C | Connor McDavid | 52.8 | Primary stack anchor
C | Elias Lindholm | 2.8 | Secondary piece (low score)
D | Charlie McAvoy | 20.6 | BOS D stack
D | Hampus Lindholm | 7.6 | BOS D stack correlation
G | John Gibson | 20.7 | Goalie ceiling hit
UTIL | Evan Bouchard | 64.1 | EDM correlation with McDavid
W | Cole Caufield | 43.5 | Ceiling game
W | Viktor Arvidsson | 17.1 | Secondary ceiling
W | Eetu Luostarinen | 7.1 | Punt play
```

**Key Insights:**
- 2-man EDM stack (McDavid + Bouchard) = 116.9 pts
- BOS D stack contributed defensive correlation
- Caufield ceiling game (43.5) was key differentiator
- Even with Lindholm (2.8) near-bust, ceiling elsewhere won

### Lessons for Model

1. Don't need every player to hit - correlation handles variance
2. 2-man stacks often sufficient (3+ can cap upside elsewhere)
3. D + Goalie can be uncorrelated or from different teams
4. One "punt" play (low salary, low floor) acceptable if upside exists

---

## 11. INDIVIDUAL PROJECTION CALIBRATION

*Kept from original plan but de-prioritized for GPP focus*

### Current Baseline (1/29/26 slate, 644 players)
- Overall MAE: 5.01 pts
- Skater MAE: 4.93 pts (C: 5.19, W: 5.38, D: 4.23)
- Goalie MAE: 6.48 pts
- Correlation: 0.406
- Bias: -1.57 (over-projection)
- Our model beats FantasyCruncher (MAE 5.01 vs 5.25, Corr 0.406 vs 0.345)

### Season-Long Backtest (top 75 skaters)
- TabPFN (pooled): MAE 6.97 — **best model** after retraining with cross-player pooling
- 10-game avg: MAE 7.11
- 5-game avg: MAE 7.44

### Bias Corrections (Applied 1/30/26)

```python
# Global: 0.97 applied to all skaters
POSITION_BIAS_CORRECTION = {
    'C': 0.95,   # Centers over-projected by ~1.58 pts
    'L': 0.94,   # Wings over-projected by ~1.85 pts
    'LW': 0.94,
    'R': 0.94,   # Corrected — was 1.01, now matches wing data
    'RW': 0.94,
    'W': 0.94,
    'D': 0.93,   # Defensemen over-projected by ~1.14 pts
}
GOALIE_BIAS_CORRECTION = 0.93  # Was 1.05 — flipped after 1/29 data showed -2.40 bias
```

### Floor/Ceiling Calibration

```python
# Floor (for cash games)
floor_mult = {'C': 0.25, 'L': 0.25, 'R': 0.25, 'D': 0.30, 'G': 0.20}

# Ceiling (for GPPs) - Keep current formula
df['ceiling'] = df['projected_fpts'] * 2.5 + 5
```

---

## 12. APPENDIX: CONTEST STRUCTURE REFERENCE

### Single Entry Strategy
- **Risk tolerance:** Medium
- **Stack depth:** 2-man stacks preferred
- **Leverage:** Moderate (5-15% ownership differential)
- **Goal:** Top 10% finish consistency

### 3-Max Strategy
- **Risk tolerance:** Medium-High
- **Entry 1:** Balanced, slight chalk lean
- **Entry 2:** Primary contrarian stack
- **Entry 3:** Secondary contrarian + different game correlation

### 20-Max Strategy
- **Risk tolerance:** High
- **Core (10 entries):** 2-3 primary game stacks, varied correlations
- **Pivot (6 entries):** Alternative game stacks
- **Leverage (4 entries):** Max contrarian, unique builds

### Mass Multi (150+) Strategy
- **Risk tolerance:** Maximum
- **Portfolio approach:** Cover all plausible ceiling outcomes
- **Leverage:** Max unique builds, minimize overlap with field
- **Correlation:** Every lineup should have clear stack thesis

---

## 13. BACKTEST CASE STUDY: January 23, 2026

### Optimizer vs Reality

**What the Optimizer Recommended:**
```
LINEUP 1 - SEA(5) + SJS(3) Stack
Projected: 126.4 pts | Actual: ~120.6 pts

LINEUP 2 - SJS(4) + SEA(2) Stack
Projected: 128.3 pts | Actual: ~101.7 pts

LINEUP 3 - SJS(4) + COL(2) + TBL(2) Stack
Projected: 134.6 pts | Actual: ~92.6 pts
```

**What Actually Won (162.6 pts):**
```
VGK LINE 1 (3 players): Eichel + Stone + Barbashev = 62.9 pts
NJD LINE 4 (3 players): Glass + Gritsyuk + Hameenaho = ~57.9 pts
CGY D PAIR (2 players): Weegar + Andersson = 27.4 pts
TBL Goalie: Vasilevskiy = 14.4 pts

STRUCTURE: Two complete line stacks + one D pair + goalie
           = Maximum correlation potential
```

### Ceiling Games Missed by Projections

| Player | Projected | Actual | Multiplier | Notes |
|--------|-----------|--------|------------|-------|
| Owen Tippett | 11.2 | 47.1 | **4.2x** | PHI explosion |
| Matvei Michkov | 9.0 | 35.5 | **3.9x** | PHI explosion |
| Cody Glass | 8.5 | 33.6 | **3.9x** | VGK Line 1 correlation |
| Keegan Kolesar | 4.4 | 19.0 | **4.3x** | VGK secondary |
| Denver Barkey | 5.7 | 16.5 | **2.9x** | PHI correlation |
| Linus Karlsson | 7.9 | 20.8 | **2.6x** | Unexpected ceiling |
| Arvid Soderblom | 8.5 | 19.5 | **2.3x** | Goalie ceiling |

### Optimizer Stack Performance (From Backtest)

**SEA Stack (Optimizer's Primary Recommendation):**
| Player | Projected | Actual | Diff |
|--------|-----------|--------|------|
| Philipp Grubauer | 17.5 | 8.4 | -9.1 |
| Jared McCann | 13.7 | 13.0 | -0.7 |
| Brandon Montour | 13.6 | 5.6 | -8.0 |
| Vince Dunn | 12.6 | 4.5 | -8.1 |
| Jordan Eberle | 11.8 | 6.5 | -5.3 |
| **SEA TOTAL** | **68.2** | **38.0** | **-30.2** |

**VGK Line 1 Stack (Winning Lineup):**
| Player | Projected | Actual | Diff | Line |
|--------|-----------|--------|------|------|
| Jack Eichel | 21.7 | 18.0 | -3.7 | Line 1 C |
| Mark Stone | 19.3 | 32.1 | +12.8 | Line 1 W |
| Ivan Barbashev | 9.6 | 12.8 | +3.2 | Line 1 W |
| **VGK TOTAL** | **50.6** | **62.9** | **+12.3** | **Full Line 1** |

*All 3 VGK players skate on the same line = maximum correlation potential*

**Key Individual: Cody Glass (NJD)**
- Projected: 8.5 | Actual: 33.6 | **+25.1 pts (3.9x)**
- Low projection = low ownership = MAX LEVERAGE

### Why the Optimizer Failed

1. **Ranked by projection sum, not ceiling probability**
   - SEA had higher projected team total
   - But SEA delivered 38.0 pts vs VGK's 62.9 pts
   - Stack choice alone cost ~25 pts

2. **Missed the leverage play**
   - Cody Glass at 8.5 projected = probably 3-5% owned
   - He scored 33.6 pts (4x ceiling)
   - This single player swing = tournament winner vs min-cash

3. **No game environment consideration**
   - PHI had a high-scoring game (Tippett 47.1, Michkov 35.5)
   - Vegas implied total would have flagged this

4. **Ownership blind**
   - Optimizer recommended highly-owned SEA stack
   - Winner had unique VGK stack + low-owned Glass

### The Winning Lineup Stack Breakdown

```
VGK LINE 1 STACK (3 players - Full Line Correlation):
Player          Projected   Actual    Notes
-----------------------------------------
Jack Eichel     21.7        18.0      VGK Line 1 C
Mark Stone      19.3        32.1      VGK Line 1 W (+12.8!)
Ivan Barbashev   9.6        12.8      VGK Line 1 W (+3.2!)
VGK TOTAL:      50.6        62.9      +12.3 pts

NJD LINE 4 STACK (3 players - Full Line Correlation):
Player          Projected   Actual    Notes
-----------------------------------------
Cody Glass       8.5        33.6      NJD Line 4 C (+25.1!!!)
Arseny Gritsyuk  9.9         5.0      NJD Line 4 W
Lenni Hameenaho  N/A       ~19.3      NJD Line 4 W
NJD TOTAL:      ~18.4      ~57.9      +39.5 pts !!!

CGY DEFENSE STACK (2 players - D Pairing Correlation):
MacKenzie Weegar 10.6       12.5      CGY D Pair
Rasmus Andersson 13.6       14.9      CGY D Pair
CGY D TOTAL:     24.2       27.4      +3.2 pts
```

**CRITICAL INSIGHT:** The winning lineup had **TWO COMPLETE LINE STACKS**:

1. **VGK Line 1** (premium): Eichel + Stone + Barbashev = 62.9 pts
2. **NJD Line 4** (value): Glass + Gritsyuk + Hameenaho = ~57.9 pts

Even a 4th line stack has correlation value! When Glass scored 33.6,
his linemates Gritsyuk and Hameenaho were in position for assists.

**IMPORTANT CONTEXT - SLATE SIZE MATTERS:**

This was a LARGE slate with many games. On large slates:
- Greater chance of 2 separate lines hitting ceiling
- Optimal to stack 2 different lines from different games
- More paths to ceiling = more diversification opportunities

On SMALL slates (2-4 games), strategy shifts:
- Fewer games = concentrate correlation in ONE game
- Stack Line 1 + PP1 teammates from same team
- Example: Eichel (Line 1 + PP1) + Stone (Line 1 + PP1) + Bouchard (PP1)
- This maximizes correlation within limited game environments

SLATE SIZE STACKING MATRIX:
| Slate Size | Games | Optimal Structure |
|------------|-------|-------------------|
| Large | 8+ | 2 line stacks from different games |
| Medium | 5-7 | 1 primary line + secondary stack |
| Small | 2-4 | 1 line + PP overlap (max correlation) |

### Corrective Actions

1. **Add line combination data to optimizer**
   - Track who plays on same line (including 4th lines!)
   - Boost correlation between linemates
   - **Target: Build lineups with 2 complete line stacks**

2. **Value 4th line stacks for leverage**
   - NJD Line 4 (Glass + Gritsyuk + Hameenaho) scored ~57.9 pts
   - Low projected lines = low ownership = max leverage
   - If one player pops, linemates get correlated points

3. **Calculate ceiling probability separately**
   - Low projection + linemates = high ceiling potential
   - Flag players like Glass as "leverage ceiling plays"

4. **Weight game environment**
   - Vegas implied totals
   - Pace of play data
   - Recent scoring trends

5. **Track historical ceiling rates**
   - Which players hit 3x+ their projection regularly?
   - Cody Glass: volatile player, high variance = ceiling candidate

### Actionable Metrics from This Backtest

| Metric | Current | Needed |
|--------|---------|--------|
| Identified winning stack | NO (recommended SEA/SJS) | Track line correlations |
| Flagged PHI ceiling | NO | Add game environment |
| Had Glass as ceiling play | NO (8.5 proj) | Add linemate correlation boost |
| Predicted 3x+ players | 0/7 | Build ceiling probability model |

---

## 14. BACKTEST CASE STUDY: January 22, 2026

### Vegas Totals vs Actual Performance

| Team | Implied Total | Winner Exposure | Stack Points |
|------|---------------|-----------------|--------------|
| CAR | 3.84 (highest) | 0 players | - |
| MIN | 3.64 | 2 players | 68.0 pts |
| NSH | 2.96 | **5 players** | **92.2 pts** |

### The NSH Over-Performance

NSH had the **13th highest** implied total (2.96) but the winner went heavy with 5 players:
- Ryan O'Reilly: 19.5
- Roman Josi: 12.3
- Juuse Saros: 11.6
- Steven Stamkos: 37.5 (6.5% owned!)
- Luke Evangelista: 11.3

**Why did NSH exceed expectations?** This is the signal to find.

### Winning Lineup Structure (182.0 pts)

```
NSH TEAM STACK (5 players): 92.2 pts
  - Concentrated in one game
  - Included goalie (Saros) for team correlation
  - Stamkos at 6.5% ownership = massive leverage

MIN LINE 1 STACK (2 players): 68.0 pts
  - Kaprizov (40.0) + Zuccarello (28.0)
  - Linemates = correlation boost
  - MIN had 3.64 implied (made sense)

OTHER: 21.8 pts
  - Noah Ostlund: 5.0
  - Mike Matheson: 16.8
```

### Key Differences: Jan 22 vs Jan 23

| Factor | Jan 22 | Jan 23 |
|--------|--------|--------|
| Slate Size | 8 games | 8 games |
| Primary Stack | 5-man team stack (NSH) | 3-man line stack (VGK) |
| Secondary Stack | 2-man line (MIN) | 3-man line stack (NJD) |
| Stack Approach | Team concentration | Line concentration |
| Key Ceiling | Kaprizov 40, Stamkos 37.5 | Glass 33.6, Stone 32.1 |
| Vegas Edge | NSH exceeded 2.96 implied | - |

### Takeaway

Same slate size, different optimal structures. The constant:
- **CONVICTION** on specific teams/lines that will exceed expectations
- **CORRELATION** through line stacks or team stacks
- **LEVERAGE** via low-owned ceiling players (Stamkos 6.5%, Glass ~3-5%)

---

## 15. FULL STACK ANALYSIS: Jan 22 & Jan 23

### January 22, 2026 - All Teams

| Rank | Team | Vegas | Top 5 Pts | Top Players |
|------|------|-------|-----------|-------------|
| 1 | MIN | 3.64 | 117.1 | Kaprizov(40), Zuccarello(28), Hartman(19) |
| 2 | NSH | 2.96 | 101.1 | Stamkos(38), O'Reilly(20), Marchessault(19) |
| 3 | VGK | 3.25 | 86.9 | Hertl(28), Eichel(20), Dorofeyev(16) |
| 4 | BOS | 2.96 | 85.6 | Pastrnak(34), McAvoy(21), Lindholm(19) |
| 5 | CBJ | 3.02 | 82.3 | Greaves(30), Werenski(20), Walman(15) |
| 6 | PIT | 3.25 | 65.6 | Matheson(17), Malkin(15), Rakell(13) |
| 7 | BUF | 3.25 | 64.2 | Luukkonen(21), Cozens(14), Tuch(13) |
| 8 | DET | 3.25 | 56.2 | Raymond(22), Seider(16), Larkin(8) |
| 9 | MTL | 3.43 | 51.0 | Suzuki(19), Caufield(12), Kapanen(10) |
| 10 | VAN | - | 42.5 | Hughes(22), Silovs(20) |
| 11 | CAR | 3.84 | 41.8 | Blake(13), Staal(10), Slavin(8) |
| 12 | WPG | 2.77 | 37.5 | Hellebuyck(12), Perfetti(12), Ehlers(6) |
| 13 | OTT | 3.00 | 35.9 | Sanderson(16), Greig(13), Chabot(4) |
| 14 | EDM | 3.77 | 34.6 | Bouchard(12), Arvidsson(7), Hyman(6) |
| 15 | DAL | 3.00 | 31.4 | Stankoven(10), Heiskanen(7), Johnston(6) |
| 16 | CHI | 3.00 | 24.6 | Hall(6), Bertuzzi(6), Bedard(6) |
| 17 | FLA | 2.75 | 13.4 | Tkachuk(6), Reinhart(4), Lundell(3) |

**Chalk Traps:** CAR (Vegas #1 → Actual #11), EDM (Vegas #2 → Actual #14)
**Exceeded Expectations:** NSH (Vegas #13 → Actual #2), BOS (Vegas #14 → Actual #4)

### January 23, 2026 - All Teams

| Rank | Team | Vegas | Top 5 Pts | Top Players |
|------|------|-------|-----------|-------------|
| 1 | PHI | 3.25 | 128.5 | Tippett(47), Michkov(36), Ersson(18) |
| 2 | VGK | 3.25 | 94.9 | Stone(32), Kolesar(19), Eichel(18) |
| 3 | NJD | 3.00 | 84.3 | Glass(34), Hischier(19), Brown(11) |
| 4 | WSH | 3.00 | 83.4 | Ovechkin(20), Thompson(20), Protas(16) |
| 5 | ANA | 3.00 | 82.6 | Gauthier(27), Mintyukov(21), Dostal(13) |
| 6 | SJS | 3.38 | 82.1 | Celebrini(23), Nedeljkovic(22), Smith(13) |
| 7 | COL | 4.18 | 77.9 | Necas(26), Olofsson(14), Makar(13) |
| 8 | DAL | 3.07 | 77.1 | Johnston(20), Robertson(19), Oettinger(13) |
| 9 | CGY | 2.80 | 75.3 | Cooley(20), Sharangovich(17), Andersson(15) |
| 10 | VAN | 2.77 | 65.8 | Karlsson(21), Pettersson(12), Boeser(12) |
| 11 | TOR | 3.20 | 61.1 | Knies(14), McMann(14), Tavares(13) |
| 12 | CHI | 2.24 | 55.4 | Soderblom(20), Greene(11), Vlasic(10) |
| 13 | TBL | 3.00 | 55.0 | Vasilevskiy(14), Raddysh(12), Kucherov(12) |
| 14 | STL | 2.75 | 50.9 | Buchnevich(16), Dvorsky(10), Kyrou(10) |
| 15 | SEA | 3.14 | 48.7 | McCann(13), Schwartz(10), Wright(9) |
| 16 | NYR | 3.25 | 43.5 | Borgen(12), Zibanejad(12), Carrick(10) |

**Chalk Traps:** COL (Vegas #1 → Actual #7), NYR (Vegas #3 → Actual #16), SEA (Vegas #7 → Actual #15)
**Exceeded Expectations:** NJD (Vegas #9 → Actual #3), WSH (Vegas #11 → Actual #4), ANA (Vegas #12 → Actual #5)

### The Core Challenge

These tables show WHAT happened. The model needs to predict WHICH teams will exceed expectations BEFORE the games.

Potential predictive factors to research:
1. **Goalie matchup quality** - Was the opponent starting a backup or struggling goalie?
2. **Recent team form** - Was the team cold and due for regression to mean?
3. **Special teams edge** - Did they have a PP/PK advantage?
4. **Rest/schedule** - Did they have a rest advantage?
5. **Individual player regression** - Were star players due for breakout games?
6. **Ownership leverage** - Low owned teams that hit provide more value

---

## 16. BACKTEST CASE STUDY: January 26, 2026

### The Slate
- 3-game slate: ANA@EDM, BOS@NYR, SJS@WPG
- Small slate = concentrated correlation opportunities

### Key Findings

**1. Vegas Total is the Top Signal**

| Game | Vegas Total | Top 5 Actual Pts | Pts Rank |
|------|-----------|-------------------|----------|
| ANA@EDM | 7.0 | 158.3 | #1 |
| BOS@NYR | 6.5 | 94.4 | #2 |
| SJS@WPG | 5.5 | ~70 | #3 |

The highest Vegas total game (ANA@EDM, 7.0) produced 158.3 pts from top 5 players — 67% more than the second-best game. **Vegas total should be the primary game selection signal.**

**2. Injuries = Opportunity (Granlund Case)**

ANA had multiple key injuries (Carlsson, Terry, McTavish), yet remaining ANA players exploded:
- **Mikael Granlund: 43.3 FPTS at 14.31% owned** — injury opportunity winner
- Remaining healthy ANA players absorbed ice time, PP time, and scoring chances
- Old model: flat +3% per injury → ~9% boost
- New model: quality-weighted +5% per key injury → ~15% boost

**3. Backup Goalie Trap (Korpisalo Case)**

System recommended Korpisalo as a starter. Result: **4.8 FPTS**.
- Korpisalo: .895 sv%, limited starts → BACKUP tier
- New goalie quality tier system now penalizes BACKUP goalies by 20%
- ELITE (sv% ≥ .915, 20+ GS): full projection
- STARTER (sv% ≥ .900, 15+ GS): 5% reduction
- BACKUP (everyone else): 20% reduction

**4. Decision Error: BOS@NYR over ANA@EDM**

The system prioritized BOS@NYR (6.5 total) over ANA@EDM (7.0 total) due to injury concerns for ANA. This was wrong — the injuries created opportunity, not reduced output.

### Corrective Actions Implemented

| Issue | Fix | File |
|-------|-----|------|
| Backup goalie recommended | Goalie Quality Tiers (ELITE/STARTER/BACKUP) | `config.py`, `features.py`, `projections.py`, `main.py` |
| Injuries treated as risk only | Quality-weighted injury boost (+5% key, +2% regular) | `config.py`, `features.py` |
| Vegas total not displayed | Vegas game ranking display with --vegas flag | `main.py` |
| No game prioritization | PRIMARY/SECONDARY/TERTIARY labels by Vegas total | `main.py` |

### Key Lessons

1. **Vegas total > all other signals** for game environment quality
2. **Injuries = opportunity** for remaining healthy players, not a reason to avoid the team
3. **Goalie quality matters** — backup-tier goalies are traps in DFS
4. **Small slates amplify these signals** — fewer games means getting the right game is critical

---

*Last Updated: January 30, 2026*
*Framework based on contest analysis from January 22-29, 2026*

---

> **Operational workflow:** See `DAILY_WORKFLOW.md` for daily steps
