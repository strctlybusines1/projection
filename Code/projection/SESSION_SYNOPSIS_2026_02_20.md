# Session Synopsis — February 20, 2026

## What We Did Today

### 1. Evaluated & Implemented 4for4 GPP Leverage Scores

**Source**: [4for4.com GPP Leverage Scores article](https://www.4for4.com/gpp-leverage-scores-balancing-value-ownership-dfs)

**Core concept**: Leverage = implied_ownership / predicted_ownership. Players with leverage > 1.0 are underowned relative to their ceiling probability (overweight them). Players < 1.0 are overowned (fade candidates).

**Three-step methodology implemented in `leverage.py`**:
1. **Win probability**: P(player exceeds position target) using normal CDF. Targets: C=16, W=15, D=11, G=22 FPTS. Std dev estimated from (ceiling - floor) / 4.
2. **Implied ownership**: Per-position probability shares scaled to total position ownership (e.g., 3 W slots × 100% = 300% total).
3. **Leverage score**: implied_ownership ÷ predicted_ownership, clipped [0.1, 10.0].

**Key article finding**: "88% of first-place lineups contained at least one player with 25%+ ownership" — don't blindly fade chalk, use leverage to find the RIGHT chalk.

### 2. Built 5 New Leverage Strategies in `line_multi_stack.py`

Added to the multi-strategy backtest framework:
- **lev_chalk**: Single stacks sorted by `lev_combo` (combo_proj × stack_leverage)
- **lev_ceiling**: Single stacks by (max_salary + PP1_bonus) × leverage + small proj tiebreaker
- **lev_contrarian**: Implied rank 3-8 stacks sorted by lev_combo
- **lev_dual**: Dual stacks sorted by lev_combo
- **lev_value**: Single stacks with proj > 35, sorted by leverage then cheapest salary

Also added leverage to D/G fill scoring: underowned D gets `(lev-1.0)*20` bonus, G gets `(lev-1.0)*15`.

### 3. Ran Full Backtest (23 Strategies × 85 Dates)

**Results**:
- Best-of-23 avg FPTS: 117.2 (vs 116.8 without leverage = +0.4)
- Cash rate: 63.6% (54/85)
- lev_chalk wins best-of-day 6/85 times (7%), 5th most frequent winner
- Only 1 truly unique leverage contribution: Nov 16 (DET stack, +35.8 FPTS swing)
- On 5/6 other winning days, ml_ceiling already found the same stack
- Leverage-Actual correlation: r=0.055, p=0.024 (small but statistically significant)

**Interpretation**: Leverage adds marginal diversification value. It doesn't transform the system but provides a safety net for edge cases where the ML model misses.

### 4. Wired Leverage into `main.py` Production Pipeline

**Changes made**:
- Added `from leverage import compute_leverage_scores, print_leverage_report` import
- After ownership prediction (line ~1157), computes leverage scores for full player pool
- Prints leverage report: top 15 leverage plays, top 15 fades, tier distribution
- Updated export columns: replaced old `leverage_score` with `gpp_leverage`, `implied_ownership`, `leverage_tier`, `win_probability`
- Wrapped in try/except with graceful fallback (defaults to 1.0 if fails)

**Pipeline flow is now**: data → projections → merge → ownership → **leverage** → simulator/optimizer → export

### 5. Removed Underperforming Strategies

Dropped from prior session's backtest analysis:
- `value13` — never won best-of-day
- `small_slate` — never won best-of-day
- `big_balanced` — never won best-of-day

## Current Strategy Roster (23 Total)

### Heuristic (17):
chalk, contrarian_1, contrarian_2, value, ceiling, game_stack, pp1_stack, dual_chalk, dual_ceiling, dual_game, mono_chalk, mono_ceiling, lev_chalk, lev_ceiling, lev_contrarian, lev_dual, lev_value

### ML (6):
ml_chalk, ml_ceiling, ml_contrarian, ml_value, ml_dual_chalk, ml_dual_ceiling

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `leverage.py` | **CREATED** | Full GPP leverage module (327 lines) |
| `line_multi_stack.py` | Modified | Added 5 leverage strategies, stack_leverage fields, leverage D/G fill |
| `main.py` | Modified | Wired leverage into production pipeline after ownership |

## Cumulative System Performance (Best-of-23)

| Metric | Value |
|--------|-------|
| Avg FPTS | 117.2 |
| Cash Rate | 63.6% (54/85) |
| Strategy Count | 23 |
| Top Strategies by Win Frequency | ml_ceiling (19), ceiling (10), chalk (8), mono_chalk (7), lev_chalk (6) |

## Pending / Next Steps

1. **Run live with leverage**: `python main.py --stacks --show-injuries --lineups 5 --edge` now includes leverage report automatically
2. **The line_multi_stack strategies are still only available in backtest mode** — they are NOT wired into main.py's optimizer (main.py still uses greedy/ILP optimizer). A `--multi-stack` CLI flag could connect them.
3. **Nested .git issue**: Need to run `rm -rf ~/Code/projection/.git` locally
4. **Git force push**: Need to run `git push --force origin main` locally
5. **Consider**: If leverage adds enough unique value over a few more live slates, could increase its weight in D/G fill scoring

## Contest Structure Reminder

- $14 WTA ticket → $121 GPP contest
- ~71-87 entries per contest, 24% cash
- 1st place avg: 153.7 FPTS
- Strategy: Maximize P(finishing 1st), not P(cashing)
