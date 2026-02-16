"""
Slate-Aware Baseline Probability for Ownership Prediction

Computes each player's structural prior probability of being rostered based on
positional scarcity relative to DraftKings NHL roster slots.

DraftKings NHL Classic roster: 2C, 3W, 2D, 1G, 1UTIL (any skater)
Total: 9 players per lineup

Concept:
  On a 2-game slate with 6 centers and 2 C slots, each C has ~33% baseline.
  On a 10-game slate with 30 centers, each C has ~6.7% baseline.
  This structural prior anchors ownership prediction before any signal (projection,
  salary, lines) is applied.

Key rules:
  - Positions use DraftKings codes: C, W (includes LW/RW), D, G
  - UTIL can be filled by any skater (C, W, D) but NOT goalies
  - Goalie baseline comes only from the 1 G slot
"""

import pandas as pd
import numpy as np
from typing import Dict

from utils import normalize_position as _utils_normalize_position


def _normalize_position(pos) -> str:
    """Normalize position for baseline probability. Returns '' for missing positions."""
    if pd.isna(pos) or str(pos).strip() == '':
        return ''  # Missing position → excluded from baseline (0 probability)
    return _utils_normalize_position(pos)

# DraftKings NHL roster slot counts (pure position slots, excluding UTIL)
DK_POSITION_SLOTS = {
    'C': 2,
    'W': 3,
    'D': 2,
    'G': 1,
}

# UTIL slot: 1 slot, fillable by any skater (C, W, D), NOT goalies
UTIL_SLOTS = 1
UTIL_ELIGIBLE = {'C', 'W', 'D'}


def compute_baseline_probabilities(pool: pd.DataFrame) -> pd.DataFrame:
    """Compute slate-aware baseline probability for each player.

    Each player's baseline_prob represents their structural probability of
    being rostered based purely on positional scarcity (slots / players at pos).

    The UTIL slot is distributed proportionally among skater positions based
    on their relative pool sizes.

    Args:
        pool: Player pool DataFrame with columns: name, team, position, salary.
              May also have 'dk_pos' column (preferred over 'position').

    Returns:
        Copy of pool with added 'baseline_prob' column.
    """
    result = pool.copy()

    if len(result) == 0:
        result['baseline_prob'] = pd.Series(dtype=float)
        return result

    # Determine position column (prefer dk_pos if present)
    pos_col = 'dk_pos' if 'dk_pos' in result.columns else 'position'

    # Normalize positions for slot calculation
    norm_pos = result[pos_col].apply(_normalize_position)

    # Count players per normalized position
    pos_counts: Dict[str, int] = {}
    for pos in DK_POSITION_SLOTS:
        pos_counts[pos] = (norm_pos == pos).sum()

    # Compute base probability per position: slots / n_players
    # Then distribute UTIL slot proportionally among skaters
    total_skaters = sum(pos_counts.get(p, 0) for p in UTIL_ELIGIBLE)

    baseline_probs = np.zeros(len(result))

    for i, npos in enumerate(norm_pos):
        if npos not in DK_POSITION_SLOTS:
            # Unknown or missing position → 0
            baseline_probs[i] = 0.0
            continue

        n_at_pos = pos_counts.get(npos, 0)
        if n_at_pos == 0:
            baseline_probs[i] = 0.0
            continue

        # Pure position slot share
        pure_slots = DK_POSITION_SLOTS[npos]
        pure_prob = pure_slots / n_at_pos

        # UTIL slot share (skaters only)
        util_prob = 0.0
        if npos in UTIL_ELIGIBLE and total_skaters > 0:
            # UTIL slot is shared among all skaters proportionally
            util_prob = UTIL_SLOTS / total_skaters

        baseline_probs[i] = pure_prob + util_prob

    result['baseline_prob'] = baseline_probs
    return result
