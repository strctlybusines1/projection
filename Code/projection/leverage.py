"""
GPP Leverage Scoring Module
============================

Implements the 4for4 GPP Leverage Score methodology for NHL DFS:

1. Calculate each player's probability of hitting a tournament-winning score
   using normal distribution (projected_fpts as mean, floor/ceiling as std dev)
2. Convert probabilities to "implied ownership" — what ownership SHOULD be
   based purely on winning potential
3. Divide implied ownership by predicted ownership = leverage score
   - Score > 1.0 = field undervalues this player → overweight
   - Score < 1.0 = field overvalues this player → fade

Integration points:
  - Stack selection: weight stacks by average leverage of forwards
  - D/G fill: add leverage bonus to ceiling scoring
  - Strategy routing: leverage-weighted strategies for differentiated lineups

Reference: 4for4.com/gpp-leverage-scores-balancing-value-ownership-dfs
Key finding: 88% of 1st-place lineups have at least one 25%+ owned player.
Don't auto-fade chalk — use leverage to find the RIGHT chalk.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple


# Position-specific target scores for tournament-winning lineups
# Derived from $121 contest 1st-place analysis (avg 153.7 total FPTS):
#   C slots (2): need ~16 each = 32 total
#   W slots (3): need ~16 each = 48 total
#   D slots (2): need ~12 each = 24 total
#   G slot (1): need ~22 = 22 total
#   UTIL (1): need ~16 = 16 total
#   Total: ~142 (conservative — real 1st needs variance above this)
#
# We use slightly above-average targets since we want P(contributing to a WIN),
# not P(being average). These are per-player contribution targets.
POSITION_TARGETS = {
    'C': 16.0,   # Centers: slightly above avg (elite C can hit 25+)
    'W': 15.0,   # Wings: slightly lower (more wing slots dilutes)
    'LW': 15.0,
    'RW': 15.0,
    'L': 15.0,
    'R': 15.0,
    'D': 11.0,   # Defensemen: lower ceiling position
    'G': 22.0,   # Goalies: need big game (win + saves + shutout potential)
}

# Position slot counts for DraftKings NHL
# Used to compute positional implied ownership targets
POSITION_SLOTS = {
    'C': 2,    # 2 Center slots
    'W': 3,    # 3 Wing slots (covers LW/RW)
    'D': 2,    # 2 Defenseman slots
    'G': 1,    # 1 Goalie slot
}

# Leverage score bounds (prevent extreme values from distorting rankings)
MIN_LEVERAGE = 0.1
MAX_LEVERAGE = 10.0

# Minimum projected FPTS to compute leverage (below this, player is noise)
MIN_PROJ_THRESHOLD = 2.0


def compute_win_probability(projected: float, floor: float, ceiling: float,
                            target: float) -> float:
    """Compute probability that a player exceeds their position's target score.

    Uses normal distribution with:
      - mean = projected_fpts
      - std = (ceiling - floor) / 4  (floor/ceiling ≈ ±2 std devs)

    Args:
        projected: Player's projected FPTS (mean of distribution)
        floor: Player's floor estimate
        ceiling: Player's ceiling estimate
        target: Position-specific target score for tournament-winning contribution

    Returns:
        P(player_score >= target), bounded [0.001, 0.999]
    """
    if projected <= MIN_PROJ_THRESHOLD:
        return 0.001

    # Estimate standard deviation from floor/ceiling range
    # Floor ≈ mean - 2σ, Ceiling ≈ mean + 2σ → range ≈ 4σ
    std = max((ceiling - floor) / 4.0, 1.0)  # Minimum std of 1.0

    # P(X >= target) = 1 - Φ((target - mean) / std)
    z = (target - projected) / std
    prob = 1.0 - norm.cdf(z)

    return np.clip(prob, 0.001, 0.999)


def compute_implied_ownership(probabilities: pd.Series,
                              positions: pd.Series,
                              position_total_own: Optional[Dict[str, float]] = None
                              ) -> pd.Series:
    """Convert win probabilities to implied ownership percentages.

    For each position:
      1. Sum all players' win probabilities
      2. Each player's share = their prob / total prob
      3. Scale to position's total ownership target

    Args:
        probabilities: Series of P(exceed target) per player
        positions: Series of normalized positions (C, W, D, G)
        position_total_own: Total ownership per position (default: slots * 100%)

    Returns:
        Series of implied ownership percentages
    """
    if position_total_own is None:
        position_total_own = {pos: slots * 100.0 for pos, slots in POSITION_SLOTS.items()}

    # Normalize positions
    pos_map = {'LW': 'W', 'RW': 'W', 'L': 'W', 'R': 'W'}
    norm_pos = positions.map(lambda p: pos_map.get(p, p))

    implied = pd.Series(0.0, index=probabilities.index)

    for pos in norm_pos.unique():
        mask = norm_pos == pos
        pos_probs = probabilities[mask]

        if pos_probs.sum() <= 0:
            continue

        total_own = position_total_own.get(pos, 100.0)

        # Each player's implied ownership = their share of position's win probability
        # × total position ownership
        implied[mask] = (pos_probs / pos_probs.sum()) * total_own

    return implied


def compute_leverage_scores(df: pd.DataFrame,
                            ownership_col: str = 'predicted_ownership',
                            proj_col: str = 'projected_fpts',
                            floor_col: str = 'floor',
                            ceiling_col: str = 'ceiling',
                            pos_col: str = 'position') -> pd.DataFrame:
    """Compute GPP leverage scores for all players in pool.

    This is the main entry point. Adds columns:
      - win_probability: P(exceed position target)
      - implied_ownership: What ownership SHOULD be based on ceiling probability
      - gpp_leverage: implied_ownership / predicted_ownership
      - leverage_tier: Categorical label (Strong Leverage, Moderate, Fair, Fade)

    Args:
        df: Player pool DataFrame (must have projected_fpts, floor, ceiling,
            predicted_ownership, and position columns)
        ownership_col: Column name for predicted ownership
        proj_col: Column name for projected FPTS
        floor_col: Column name for floor estimate
        ceiling_col: Column name for ceiling estimate
        pos_col: Column name for position

    Returns:
        DataFrame with leverage columns added
    """
    result = df.copy()

    # Get position column (handle dk_pos vs position)
    actual_pos_col = 'dk_pos' if 'dk_pos' in result.columns else pos_col

    # Ensure required columns exist
    for col in [proj_col, floor_col, ceiling_col, ownership_col, actual_pos_col]:
        if col not in result.columns:
            print(f"  [Leverage] Warning: missing column '{col}', skipping leverage computation")
            result['gpp_leverage'] = 1.0
            result['win_probability'] = 0.0
            result['implied_ownership'] = result.get(ownership_col, 5.0)
            result['leverage_tier'] = 'Unknown'
            return result

    # Step 1: Compute win probability for each player
    positions = result[actual_pos_col]
    targets = positions.map(lambda p: POSITION_TARGETS.get(p, 15.0))

    result['win_probability'] = [
        compute_win_probability(
            projected=row[proj_col],
            floor=row[floor_col],
            ceiling=row[ceiling_col],
            target=targets.iloc[i]
        )
        for i, (_, row) in enumerate(result.iterrows())
    ]

    # Step 2: Convert to implied ownership
    result['implied_ownership'] = compute_implied_ownership(
        result['win_probability'], positions
    )

    # Step 3: Compute leverage score
    pred_own = result[ownership_col].clip(lower=0.5)  # Prevent division by near-zero
    result['gpp_leverage'] = (result['implied_ownership'] / pred_own).clip(
        lower=MIN_LEVERAGE, upper=MAX_LEVERAGE
    )

    # Step 4: Assign leverage tiers
    result['leverage_tier'] = result['gpp_leverage'].apply(_get_leverage_tier)

    return result


def _get_leverage_tier(leverage: float) -> str:
    """Categorize leverage score into actionable tiers."""
    if leverage >= 2.0:
        return 'Strong Leverage'    # Field massively underweights this player
    elif leverage >= 1.3:
        return 'Moderate Leverage'  # Good value relative to ownership
    elif leverage >= 0.8:
        return 'Fair Value'         # Ownership roughly matches ceiling probability
    elif leverage >= 0.5:
        return 'Slight Fade'        # Slightly overowned for what they bring
    else:
        return 'Strong Fade'        # Field massively overweights this player


def compute_stack_leverage(forwards: List[Dict], leverage_map: Dict[str, float]) -> float:
    """Compute average leverage score for a stack's forwards.

    Args:
        forwards: List of forward dicts (must have 'player_name' key)
        leverage_map: Dict mapping player_name -> gpp_leverage score

    Returns:
        Average leverage score for the stack (1.0 = neutral)
    """
    if not forwards:
        return 1.0

    leverages = []
    for f in forwards:
        name = f.get('player_name', f.get('name', ''))
        lev = leverage_map.get(name, 1.0)
        leverages.append(lev)

    return np.mean(leverages) if leverages else 1.0


def compute_lineup_leverage(lineup: List[Dict], leverage_map: Dict[str, float]) -> Dict:
    """Compute leverage metrics for a full 9-player lineup.

    Returns dict with:
      - avg_leverage: Average across all 9 players
      - min_leverage: Lowest leveraged player
      - max_leverage: Highest leveraged player
      - n_strong_leverage: Count of players with leverage >= 2.0
      - n_fades: Count of players with leverage < 0.5
      - leverage_score: Composite score (higher = more differentiated from field)
    """
    leverages = []
    for p in lineup:
        name = p.get('name', p.get('player_name', ''))
        lev = leverage_map.get(name, 1.0)
        leverages.append(lev)

    if not leverages:
        return {'avg_leverage': 1.0, 'min_leverage': 1.0, 'max_leverage': 1.0,
                'n_strong_leverage': 0, 'n_fades': 0, 'leverage_score': 0.0}

    leverages = np.array(leverages)

    return {
        'avg_leverage': float(np.mean(leverages)),
        'min_leverage': float(np.min(leverages)),
        'max_leverage': float(np.max(leverages)),
        'n_strong_leverage': int(np.sum(leverages >= 2.0)),
        'n_fades': int(np.sum(leverages < 0.5)),
        'leverage_score': float(np.sum(np.maximum(leverages - 1.0, 0))),
    }


def print_leverage_report(df: pd.DataFrame, top_n: int = 15):
    """Print formatted leverage analysis report."""
    if 'gpp_leverage' not in df.columns:
        print("  [Leverage] No leverage data — run compute_leverage_scores first")
        return

    print("\n" + "=" * 80)
    print("  GPP LEVERAGE ANALYSIS")
    print("=" * 80)

    # Top leverage plays
    print(f"\n  TOP {top_n} LEVERAGE PLAYS (underowned relative to ceiling):")
    print("  " + "-" * 75)
    top_lev = df.nlargest(top_n, 'gpp_leverage')
    pos_col = 'dk_pos' if 'dk_pos' in df.columns else 'position'
    for _, row in top_lev.iterrows():
        pos = row.get(pos_col, '?')
        print(f"    {row['name']:<25} {row['team']:<4} {pos:<3} "
              f"${row['salary']:<6,} | Proj: {row['projected_fpts']:>5.1f} | "
              f"Own: {row['predicted_ownership']:>5.1f}% → Impl: {row['implied_ownership']:>5.1f}% | "
              f"Lev: {row['gpp_leverage']:.2f} ({row['leverage_tier']})")

    # Top fades
    print(f"\n  TOP {top_n} FADES (overowned relative to ceiling):")
    print("  " + "-" * 75)
    skaters = df[df[pos_col] != 'G']  # Don't fade goalies (too few options)
    top_fades = skaters[skaters['predicted_ownership'] >= 5.0].nsmallest(top_n, 'gpp_leverage')
    for _, row in top_fades.iterrows():
        pos = row.get(pos_col, '?')
        print(f"    {row['name']:<25} {row['team']:<4} {pos:<3} "
              f"${row['salary']:<6,} | Proj: {row['projected_fpts']:>5.1f} | "
              f"Own: {row['predicted_ownership']:>5.1f}% → Impl: {row['implied_ownership']:>5.1f}% | "
              f"Lev: {row['gpp_leverage']:.2f} ({row['leverage_tier']})")

    # Leverage distribution
    print(f"\n  LEVERAGE DISTRIBUTION:")
    print("  " + "-" * 40)
    for tier in ['Strong Leverage', 'Moderate Leverage', 'Fair Value', 'Slight Fade', 'Strong Fade']:
        count = (df['leverage_tier'] == tier).sum()
        pct = 100.0 * count / len(df) if len(df) > 0 else 0
        print(f"    {tier:<20} {count:>4} players ({pct:>5.1f}%)")
