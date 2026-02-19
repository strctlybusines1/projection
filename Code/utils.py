"""
Shared utilities for NHL DFS Projection System.

Centralizes duplicated logic that was previously scattered across:
- main.py, optimizer.py, optimizer_ilp.py, simulator.py, baseline_probability.py (position normalization)
- lines.py, edge_stats.py, linemate_corr.py, se_ownership.py, tournament_equity.py (fuzzy matching)
- config.py, backtest.py, backtest_agents.py, etc. (DK scoring)
- main.py (name cleaning)

Import from here instead of reimplementing.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional

import pandas as pd

from config import (
    SKATER_SCORING,
    SKATER_BONUSES,
    GOALIE_SCORING,
    GOALIE_BONUSES,
)


# ============================================================================
# Position Normalization
# ============================================================================

# Lookup table for fast normalization
_POS_NORMALIZE = {
    'L': 'W', 'LW': 'W', 'R': 'W', 'RW': 'W', 'W': 'W',
    'C': 'C', 'C/W': 'C', 'W/C': 'C',
    'D': 'D', 'LD': 'D', 'RD': 'D',
    'G': 'G',
}


def normalize_position(pos) -> str:
    """
    Normalize NHL positions to DraftKings format.

    L, LW, R, RW -> W (Wing)
    C, C/W, W/C -> C (Center)
    D, LD, RD   -> D (Defense)
    G            -> G (Goalie)
    """
    if pd.isna(pos):
        return 'W'
    return _POS_NORMALIZE.get(str(pos).upper().strip(), str(pos).upper().strip())


def normalize_positions_column(df: pd.DataFrame, col: str = 'position') -> pd.DataFrame:
    """Apply position normalization to a DataFrame column in-place."""
    if col in df.columns:
        df[col] = df[col].apply(normalize_position)
    return df


# ============================================================================
# Fuzzy Name Matching
# ============================================================================

def fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """
    Check if two player names are similar enough to be the same person.

    Uses SequenceMatcher ratio, substring containment, and last-name matching.
    """
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Exact match
    if n1 == n2:
        return True

    # Similarity ratio
    if SequenceMatcher(None, n1, n2).ratio() >= threshold:
        return True

    # Substring containment (for nicknames / abbreviated names)
    if n1 in n2 or n2 in n1:
        return True

    # Last name match (if last name is long enough to be distinctive)
    last1 = n1.split()[-1] if n1.split() else n1
    last2 = n2.split()[-1] if n2.split() else n2
    if last1 == last2 and len(last1) > 3:
        return True

    return False


def find_player_match(target_name: str, player_list: List[str],
                      threshold: float = 0.85) -> Optional[str]:
    """
    Find the best matching player name from a list.

    Returns the best match above the threshold, or None.
    """
    target_lower = target_name.lower().strip()

    # Exact match first
    for name in player_list:
        if name.lower().strip() == target_lower:
            return name

    # Fuzzy match — pick best ratio above threshold
    best_match = None
    best_ratio = 0.0

    for name in player_list:
        ratio = SequenceMatcher(None, target_lower, name.lower().strip()).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = name

    return best_match


# ============================================================================
# Name Cleaning (for DK salary merge)
# ============================================================================

_UMLAUT_MAP = {'ü': 'ue', 'ö': 'oe', 'ä': 'ae', 'ß': 'ss'}


def clean_name(name: str) -> str:
    """
    Normalize a player name for matching: transliterate accents, lowercase,
    strip non-alpha characters.
    """
    s = name
    for orig, repl in _UMLAUT_MAP.items():
        s = s.replace(orig, repl).replace(orig.upper(), repl.capitalize())
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.lower().strip()
    s = re.sub(r'[^a-z\s]', '', s)
    return s


# ============================================================================
# DraftKings Fantasy Scoring
# ============================================================================

def calculate_skater_fantasy_points(
    goals: int, assists: int, shots: int, blocks: int,
    sh_goals: int = 0, sh_assists: int = 0, shootout_goals: int = 0
) -> float:
    """Calculate DraftKings fantasy points for a skater."""
    points = 0.0

    # Base scoring
    points += goals * SKATER_SCORING["goals"]
    points += assists * SKATER_SCORING["assists"]
    points += shots * SKATER_SCORING["shots_on_goal"]
    points += blocks * SKATER_SCORING["blocked_shots"]
    points += (sh_goals + sh_assists) * SKATER_SCORING["shorthanded_points_bonus"]
    points += shootout_goals * SKATER_SCORING["shootout_goal"]

    # Bonuses
    if goals >= 3:
        points += SKATER_BONUSES["hat_trick"]
    if shots >= 5:
        points += SKATER_BONUSES["five_plus_shots"]
    if blocks >= 3:
        points += SKATER_BONUSES["three_plus_blocks"]
    if (goals + assists) >= 3:
        points += SKATER_BONUSES["three_plus_points"]

    return points


def calculate_goalie_fantasy_points(
    win: bool, saves: int, goals_against: int,
    shutout: bool = False, ot_loss: bool = False
) -> float:
    """Calculate DraftKings fantasy points for a goalie."""
    points = 0.0

    if win:
        points += GOALIE_SCORING["win"]
    if ot_loss:
        points += GOALIE_SCORING["overtime_loss"]

    points += saves * GOALIE_SCORING["save"]
    points += goals_against * GOALIE_SCORING["goal_against"]

    if shutout:
        points += GOALIE_SCORING["shutout_bonus"]

    # Bonuses
    if saves >= 35:
        points += GOALIE_BONUSES["thirty_five_plus_saves"]

    return points


# ============================================================================
# Game Info Parsing
# ============================================================================

def parse_opponent_from_game_info(player_team: str, game_info: str) -> Optional[str]:
    """
    Extract opponent team from DK game info string.

    Game info format: "ANA@EDM 01/26/2026 08:30PM ET"
    Returns the team abbreviation the player is playing AGAINST.
    """
    if not game_info or not player_team:
        return None

    try:
        matchup = game_info.split()[0] if game_info else ''
        if '@' in matchup:
            away, home = matchup.split('@')
            if player_team.upper() == away.upper():
                return home.upper()
            elif player_team.upper() == home.upper():
                return away.upper()
    except (IndexError, ValueError):
        pass

    return None
