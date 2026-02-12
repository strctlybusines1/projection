# CLAUDE.md — Jim Simons Operating System for NHL DFS

## Identity

You are operating as Jim Simons would if he ran a DFS fund. Every decision must be rooted in **measured signal**, not intuition. If a feature doesn't improve MAE or lineup selection in a backtest, it doesn't ship. Period.

The mantra: **Don't optimize the forecast. Optimize the decision.**

## Core Principles

1. **Signal must be proven in data.** No feature enters production without out-of-sample backtest improvement. "Makes theoretical sense" is not enough.
2. **The distribution matters more than the point estimate.** Player FPTS follows a zero-inflated lognormal. Model the full distribution, not the mean.
3. **Correlation is the hidden edge.** Goalie ↔ opponent skaters (r=-0.34), same-line (r=0.124), same-team (r=0.034). Simulate WITH correlations via Cholesky.
4. **Selection > Projection.** Improving projections by 0.7% MAE is noise. Picking the right lineup from candidates via P(exceed target) is worth +14.2 FPTS/slate.
5. **Variance is the product in GPPs.** Increasing lineup σ from 18→30 is worth more than increasing mean from 85→95. Seek correlated upside.
6. **Backtest every morning.** Before touching any parameter, run the numbers on yesterday's slate. Let the data tell you what broke.

## Current System Architecture

```
PIPELINE FLOW (single-entry mode):

  DK Salary CSV
       ↓
  data_pipeline.py  → NHL API + MoneyPuck injuries + NST data
       ↓
  features.py       → Feature engineering (TOI, rates, matchups)
       ↓
  projections.py    → Ridge regression (MAE=4.49 on 28,526 games)
                       TabPFN optional, DK Avg blend (80/20)
       ↓
  lines.py          → DailyFaceoff line/PP confirmation
       ↓
  ownership.py      → SE ownership model (trained on 2,249 obs)
       ↓
  optimizer.py      → Triple-mix candidate generation:
                       50% uncapped (4-3 stacks emerge)
                       25% salary-capped ($7.5k max)
                       25% forced 3-3 stacks
                       → 100+ candidates across 5 randomness levels
       ↓
  sim_selector.py   → Monte Carlo simulation selector (--sim-select)
                       Zero-inflated lognormal per player
                       Cholesky-correlated 8,000 sims per lineup
                       Select by E[payout] / P(cash) / P(gpp)
       ↓                  ── OR (legacy) ──
  tournament_equity.py → M+3σ selector (default without --sim-select)
       ↓
  FINAL LINEUP → DraftKings
```

### Key Files

| File | Purpose | Status |
|------|---------|--------|
| `main.py` | CLI orchestrator | Production |
| `projections.py` | Ridge + bias corrections (MAE=4.49) | Production |
| `optimizer.py` | Triple-mix candidate generation | Production |
| `simulation_engine.py` | Correlated Monte Carlo engine | Production |
| `sim_selector.py` | P(target) lineup selector | Production |
| `tournament_equity.py` | M+3σ selector (legacy fallback) | Backup |
| `ownership.py` | SE ownership prediction | Production |
| `se_ownership.py` | Small-field ownership model | Production |
| `config.py` | All tunable parameters | Production |
| `backtest.py` | Accuracy validation | Production |
| `pp_promotion.py` | PP unit change detector | Production |

### Data Assets

| Asset | Location | Size |
|-------|----------|------|
| DK season history | `~/dk_salaries_season/` | 113 slates, 46K rows |
| Historical DB | `data/nhl_dfs_history.db` | 35K actuals, 7.4K contest entries |
| Correlation structure | `data/correlation_structure.json` | Measured from 29K games |
| Score→payout curve | `data/score_payout_lookup.json` | 105 SE contests |
| PP promotion data | `data/pp_promotion_data.json` | All team PP histories |

## Proven Signals (Backtested)

These features have measured, out-of-sample predictive power. Do not remove without evidence of degradation.

| Signal | Effect Size | Evidence | Status |
|--------|------------|----------|--------|
| DK Season Avg | r=0.394, MAE=4.56 | 28K games | In production |
| Ridge projection | r=0.405, MAE=4.49 | 28K games, CV | In production |
| PP1 + high total | +2.1 FPTS vs no PP | 28K games | In production |
| Line assignment | 0.6 FPTS/line step | 28K games | In production |
| TeamGoal (Vegas) | 3.5+ = 1.147x rate mult | 28K games | In production |
| Goalie ↔ opp corr | r=-0.340 | 29K pairs | In sim engine |
| Same-line corr | r=0.124 | 51K pairs | In sim engine |
| Same-team corr | r=0.034 | 186K pairs | In sim engine |
| Floor rate (ZI) | 28% of games ≤1.5 FPTS | 29K games | In sim engine |
| Triple-mix candidates | +2.1 FPTS vs single-type | 7 slates | In production |
| M+3σ selector | +14.3 FPTS vs field | 7 slates | In production |
| Sim P(target) | +14.2 FPTS (26% of best) | 5 slates | In production |
| SE ownership | r=0.458 (dk_avg is #1) | 2,249 obs | In production |
| Goalie rolling model | +2.2 FPTS/lineup | 113 slates | In production |

## Rejected Signals (No Backtest Improvement)

Do NOT reintroduce without new evidence:

| Signal | Why Rejected | Evidence |
|--------|-------------|----------|
| Home/away splits | 0.3 FPTS diff, not significant | Welch t-test p>0.05, 28K games |
| 5v5 vs all-situations | 97% variance from rate, 3% TOI | CV analysis, 730 players |
| FPTS/60 rate model | r=0.303 vs raw FPTS r=0.445 | Dividing out TOI adds noise |
| Rate×TOI decomposition | +0.031 MAE (0.7%) over DK Avg | Full season, Ridge CV |
| PDO regression | r=0.107 | Reduced to 2% factor |
| Hot/cold streaks | Mean reversion dominates | Rolling window analysis |
| EWM FPTS (span=5) | r=0.331, worse than DK Avg | 28K games |
| Individual component rates | G/60 r=0.138, A/60 r=0.120 | Too noisy, 1,872 games |

## Daily Operating Procedure

### Morning Backtest (Before Any Changes)

```bash
# 1. Score yesterday's lineup vs actuals
python post_slate.py

# 2. Run backtest on latest data
python backtest.py

# 3. Answer these questions with DATA:
#    - Did our lineup cash? What percentile?
#    - Which players busted/boomed vs projection?
#    - Is there a SYSTEMATIC pattern or just variance?
```

**Simons Rule:** A single bad night is variance. Three consecutive misses on the same pattern demands investigation. One miss changes nothing.

### Pre-Slate Execution

```bash
# 1. Download DK salary CSV → daily_salaries/

# 2. SE GPP lineup (Monte Carlo selector)
python main.py --single-entry --sim-select --lineups 100 \
  --stacks --show-injuries --filter-injuries

# 3. WTA satellite (max ceiling mode)
python main.py --single-entry --sim-select --sim-mode gpp --lineups 100 \
  --stacks --show-injuries --filter-injuries

# 4. Pre-lock: confirm goalies
python lines.py
```

### When to Adjust vs When to Trust the Process

**DO adjust when:**
- Systematic bias >1.0 MAE shift sustained over 5+ slates
- New data source becomes available with proven signal
- Contest structure changes (field size, payout curve shifts)
- Correlation structure drifts >0.05 (re-measure monthly)

**DO NOT adjust when:**
- Single bad night (variance in a high-entropy sport)
- "Feel" that a player is mispriced
- A feature "should" help theoretically but hasn't been backtested
- Recency bias after one big miss

## Mathematical Foundation

### Player Distribution: Zero-Inflated Lognormal

```
P(FPTS = x) = π × δ(x ≤ 1.5) + (1-π) × LogNormal(μ, σ)

  π = floor probability (28% league-wide, per-player fitted)
  μ, σ = fitted from player's historical non-floor games
  Game environment: rate × (TeamGoal / 3.0)
```

### Lineup Simulation: Cholesky Decomposition

```
z_independent ~ N(0, I₉)       # 9 independent standard normals
z_correlated = L × z_ind       # L = cholesky(Σ)
FPTS_i = F_i⁻¹(Φ(z_i))        # Map to player's personal distribution
```

### Selection Criterion by Contest Type

```
SE GPP ($121):  argmax E[payout(lineup)]
WTA satellite:  argmax P(lineup ≥ 150)
Cash/double-up: argmax P(lineup ≥ 111)
```

## Performance Benchmarks

| Metric | Value | Source |
|--------|-------|--------|
| Skater projection MAE | 4.49 | 28,526 games |
| DK Avg baseline MAE | 4.57 | 28,526 games |
| Sim calibration error | <3.5% all percentiles | 4,946 held-out |
| Sim engine speed | 11ms/lineup @ 8K sims | 100 candidates |
| M+3σ selector edge | +14.3 FPTS vs field | 7 slates |
| Sim P(target) edge | +14.2 FPTS vs field | 5 slates |
| Hypothetical 7-slate ROI | +114.8% ($972 on $847) | M+3σ |

## Research Queue (Ranked by Expected Impact)

1. **Full-season sim selector backtest** — 113 slates, 150+ candidates. Definitive head-to-head vs M+3σ.
2. **Ownership leverage** — Weight P(target) by inverse ownership for contrarian upside.
3. **Multi-lineup portfolio optimization** — For 3-max or 20-max GPPs, optimize the portfolio.
4. **Game script modeling** — Trailing teams pull goalies → model late-game variance.
5. **Live odds movement** — Capture line movement pre-lock for correlation updates.
6. **Agent consensus swarm** — Multiple specialized AI agents vote on lineup selection.

---

*"The best thing about being a quantitative investor is that you don't need opinions. You need data."*
