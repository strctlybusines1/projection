"""
Centralized team abbreviation normalization for NHL DFS projection system.

Problem: DraftKings uses short abbreviations (NJ, LA, SJ, TB) while NHL game logs,
Natural Stat Trick, and MoneyPuck use standard NHL abbreviations (NJD, LAK, SJS, TBL).
This mismatch caused scoring bugs that silently dropped 1-2 players per lineup from
backtest actuals, inflating reported cash rates by ~30+ FPTS/slate.

This module provides a single source of truth for team normalization.

Author: Claude
Date: 2026-02-19
"""

# DraftKings → NHL standard mapping (lowercase)
# These are the ONLY teams that differ between DK and NHL standard
_DK_TO_NHL = {
    'nj': 'njd',
    'la': 'lak',
    'sj': 'sjs',
    'tb': 'tbl',
}

# NST dotted → NHL standard mapping
_NST_TO_NHL = {
    't.b': 'tbl',
    'n.j': 'njd',
    'l.a': 'lak',
    's.j': 'sjs',
}

# Reverse mapping: NHL standard → DK format
_NHL_TO_DK = {v: k for k, v in _DK_TO_NHL.items()}
_NHL_TO_DK.update({
    'njd': 'nj',
    'lak': 'la',
    'sjs': 'sj',
    'tbl': 'tb',
})

# Combined mapping: any format → NHL standard (lowercase)
_ALL_TO_NHL = {}
_ALL_TO_NHL.update(_DK_TO_NHL)
_ALL_TO_NHL.update(_NST_TO_NHL)
# Identity mappings for already-correct abbreviations
_ALL_TO_NHL.update({t.lower(): t.lower() for t in [
    'ANA', 'ARI', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL',
    'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR',
    'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'UTA', 'VAN',
    'VGK', 'WSH', 'WPG',
]})


def normalize_team(team: str) -> str:
    """
    Normalize any team abbreviation to NHL standard format (uppercase).

    Handles: DK format (NJ, LA, SJ, TB), NST format (T.B, N.J, L.A, S.J),
    and already-correct NHL format (NJD, LAK, SJS, TBL).

    Args:
        team: Team abbreviation in any format

    Returns:
        NHL standard abbreviation (uppercase), e.g. 'NJD', 'LAK', 'SJS', 'TBL'
    """
    t = str(team).lower().strip()
    normalized = _ALL_TO_NHL.get(t, t)
    return normalized.upper()


def normalize_team_lower(team: str) -> str:
    """Same as normalize_team but returns lowercase."""
    return normalize_team(team).lower()


def to_dk_format(team: str) -> str:
    """
    Convert NHL standard abbreviation to DraftKings format.
    Only NJD→NJ, LAK→LA, SJS→SJ, TBL→TB differ.

    Args:
        team: NHL standard abbreviation

    Returns:
        DK format abbreviation (uppercase)
    """
    t = team.lower().strip()
    result = _NHL_TO_DK.get(t, t)
    return result.upper()


def normalize_series(series, uppercase=True):
    """
    Normalize a pandas Series of team abbreviations.

    Args:
        series: pandas Series of team abbreviations
        uppercase: If True, return uppercase. If False, lowercase.

    Returns:
        pandas Series with normalized team abbreviations
    """
    import pandas as pd
    func = normalize_team if uppercase else normalize_team_lower
    return series.apply(func)
