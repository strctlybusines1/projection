"""
Contest ROI and leverage recommendation for NHL DFS.

Takes contest parameters (entry fee, max entries, payout structure),
recommends optimal leverage band, and scores lineups by contest-adjusted
expected value (EV) to maximize long-term ROI.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import argparse

import pandas as pd
import numpy as np

try:
    from config import CONTEST_PAYOUT_PRESETS as _CONFIG_PRESETS
    PAYOUT_PRESETS = _CONFIG_PRESETS
except (ImportError, AttributeError):
    # Fallback: share of prize pool to top 1%, top 10%, and min-cash tier
    PAYOUT_PRESETS = {
        "top_heavy_gpp": {
            "first_place_pct": 0.20,
            "top_10_pct_share": 0.50,
            "min_cash_pct": 0.20,
        },
        "flat": {
            "first_place_pct": 0.05,
            "top_10_pct_share": 0.15,
            "min_cash_pct": 0.25,
        },
        "high_dollar_single": {
            "first_place_pct": 0.15,
            "top_10_pct_share": 0.45,
            "min_cash_pct": 0.22,
        },
    }


@dataclass
class ContestProfile:
    """Contest parameters that drive leverage and EV."""
    entry_fee: float
    max_entries: int
    field_size: int
    payout_preset: str = "top_heavy_gpp"
    first_place_pct: Optional[float] = None
    top_10_pct_share: Optional[float] = None
    min_cash_pct: Optional[float] = None
    prize_pool_override: Optional[float] = None
    min_cash_entries: Optional[int] = None

    def get_payout_curve(self) -> Dict[str, float]:
        """Return first_place_pct, top_10_pct_share, min_cash_pct (from preset or overrides)."""
        preset = PAYOUT_PRESETS.get(self.payout_preset, PAYOUT_PRESETS["top_heavy_gpp"])
        return {
            "first_place_pct": self.first_place_pct if self.first_place_pct is not None else preset["first_place_pct"],
            "top_10_pct_share": self.top_10_pct_share if self.top_10_pct_share is not None else preset["top_10_pct_share"],
            "min_cash_pct": self.min_cash_pct if self.min_cash_pct is not None else preset["min_cash_pct"],
        }

    @property
    def prize_pool(self) -> float:
        if self.prize_pool_override is not None:
            return self.prize_pool_override
        return self.entry_fee * self.field_size

    def get_min_cash_count(self) -> int:
        """Number of entries in the min-cash tier (paid positions excluding top 10)."""
        if self.min_cash_entries is not None:
            return max(1, self.min_cash_entries - 10)
        return max(1, int(self.field_size * 0.2))


@dataclass
class LeverageRecommendation:
    """Recommended leverage band and entry allocation for a contest."""
    target_ownership_low: float
    target_ownership_high: float
    leverage_tier: str
    entry_allocation: Optional[Dict[str, int]] = None
    summary: str = ""


def recommend_leverage(profile: ContestProfile) -> LeverageRecommendation:
    """
    Map contest profile to recommended leverage band and optional entry allocation.

    Logic from DAILY_PROJECTION_IMPROVEMENT_PLAN:
    - Single entry -> moderate leverage
    - 3-5 max -> mix safe + contrarian
    - 20 max -> mix chalk and leverage
    - 150+ max -> max leverage
    - Top-heavy payout -> more leverage; flat -> balanced
    """
    curve = profile.get_payout_curve()
    first_pct = curve["first_place_pct"]
    max_ent = profile.max_entries
    field = profile.field_size

    # Payout type
    top_heavy = first_pct >= 0.15
    flat = first_pct <= 0.08

    # Field size band
    small_field = field < 10000
    large_field = field >= 50000

    # Default band (total lineup ownership as % sum of 9 players)
    if max_ent == 1:
        target_low, target_high = 40.0, 65.0
        tier = "Moderate"
        summary = "Single entry: balanced ceiling + floor, moderate leverage."
    elif max_ent <= 3:
        target_low, target_high = 35.0, 70.0
        tier = "Moderate"
        summary = "3-max: mix 1 safer, rest contrarian."
    elif max_ent <= 20:
        target_low, target_high = 30.0, 75.0
        tier = "Moderate to Aggressive"
        entry_allocation = {
            "chalk": max(1, max_ent // 3),
            "moderate": max(1, max_ent // 2),
            "leverage": max(1, max_ent - (max_ent // 3) - (max_ent // 2)),
        }
        summary = "20-max: cover multiple stacks; mix chalk and leverage."
    else:
        target_low, target_high = 20.0, 55.0
        tier = "Aggressive"
        entry_allocation = {"leverage": max_ent}
        summary = "150+ max: ceiling-only, max leverage on each."

    # Adjust for payout
    if top_heavy and not flat:
        target_high = min(70.0, target_high - 5.0)
        if tier == "Moderate":
            tier = "Moderate to Aggressive"
        summary += " Top-heavy payout: lean leverage, unique stacks."
    elif flat:
        target_low = min(50.0, target_low + 10.0)
        target_high = min(80.0, target_high + 5.0)
        if "Aggressive" in tier:
            tier = "Moderate"
        summary += " Flat payout: balanced approach, don't over-leverage."

    return LeverageRecommendation(
        target_ownership_low=target_low,
        target_ownership_high=target_high,
        leverage_tier=tier,
        entry_allocation=entry_allocation if max_ent > 1 else None,
        summary=summary.strip(),
    )


def _bucket_probs(total_proj: float, total_ownership_pct: float) -> Tuple[float, float, float]:
    """
    Heuristic P(top 1%), P(top 10%), P(min-cash or better) from projection and ownership.

    Higher projection -> more mass in top buckets.
    Lower total ownership -> more leverage -> higher variance (more mass in top 1% and in bust).
    Ensures p_min >= p_top10 >= p_top1.
    """
    proj_norm = (total_proj - 120.0) / 40.0
    p_top1 = 1.0 / (1.0 + np.exp(-proj_norm * 1.5))
    p_top10 = 1.0 / (1.0 + np.exp(-proj_norm * 1.0))
    p_min = 1.0 / (1.0 + np.exp(-proj_norm * 0.6))

    own_norm = (total_ownership_pct - 50.0) / 30.0
    p_top1 *= (1.0 - 0.3 * np.tanh(own_norm))
    p_top10 *= (1.0 - 0.15 * np.tanh(own_norm))
    p_min *= (1.0 + 0.2 * np.tanh(own_norm))

    p_top1 = max(0.001, min(0.12, p_top1))
    p_top10 = max(p_top1, min(0.35, p_top10))
    p_min = max(p_top10, min(0.50, p_min))
    return p_top1, p_top10, p_min


def contest_ev_score(
    lineup_df: pd.DataFrame,
    profile: ContestProfile,
    player_pool: pd.DataFrame,
    ownership_col: str = "predicted_ownership",
    proj_col: str = "projected_fpts",
    name_col: str = "name",
) -> float:
    """
    Expected payout ($) for one lineup given contest profile and ownership.

    Uses bucket model: P(top 1%), P(top 10%), P(min-cash) from lineup proj + ownership;
    multiplies by contest $ for each bucket; returns EV.
    """
    if lineup_df.empty or player_pool.empty:
        return 0.0
    total_proj = lineup_df[proj_col].sum()
    # Total lineup ownership = sum of 9 players' ownership %
    name_to_own = None
    if ownership_col in player_pool.columns and name_col in player_pool.columns:
        name_to_own = player_pool.set_index(name_col)[ownership_col].to_dict()
    if name_to_own is None:
        total_own = 50.0
    else:
        total_own = 0.0
        for _, row in lineup_df.iterrows():
            total_own += name_to_own.get(row.get(name_col, ""), 10.0)
        total_own = float(total_own)

    p_top1, p_top10, p_min = _bucket_probs(total_proj, total_own)
    curve = profile.get_payout_curve()
    pool = profile.prize_pool
    first_place = pool * curve["first_place_pct"]
    top10_avg = (pool * curve["top_10_pct_share"]) / 10.0
    min_cash_avg = (pool * curve["min_cash_pct"]) / profile.get_min_cash_count()
    ev = p_top1 * first_place + (p_top10 - p_top1) * top10_avg + (p_min - p_top10) * min_cash_avg
    return max(0.0, ev)


def score_lineups(
    lineups: List[pd.DataFrame],
    profile: ContestProfile,
    player_pool: pd.DataFrame,
    ownership_col: str = "predicted_ownership",
    proj_col: str = "projected_fpts",
    name_col: str = "name",
) -> List[Tuple[pd.DataFrame, float]]:
    """
    Score each lineup by contest EV; return list of (lineup_df, contest_ev) sorted by EV descending.
    """
    scored = []
    for lu in lineups:
        ev = contest_ev_score(lu, profile, player_pool, ownership_col, proj_col, name_col)
        scored.append((lu, ev))
    scored.sort(key=lambda x: -x[1])
    return scored


def print_leverage_recommendation(profile: ContestProfile, rec: LeverageRecommendation) -> None:
    """Print contest profile and leverage recommendation."""
    print("\n" + "=" * 60)
    print("CONTEST LEVERAGE RECOMMENDATION")
    print("=" * 60)
    print(f"Entry fee: ${profile.entry_fee:,.0f}  Max entries: {profile.max_entries}  Field: {profile.field_size:,}")
    print(f"Prize pool: ${profile.prize_pool:,.0f}")
    print(f"Leverage tier: {rec.leverage_tier}")
    print(f"Target total lineup ownership: {rec.target_ownership_low:.0f}% - {rec.target_ownership_high:.0f}%")
    if rec.entry_allocation:
        print("Entry allocation:", rec.entry_allocation)
    print()
    print(rec.summary)
    print("=" * 60)


def main_cli():
    parser = argparse.ArgumentParser(description="Contest leverage recommendation and ROI tool.")
    parser.add_argument("--entry-fee", type=float, default=5.0, help="Entry fee ($)")
    parser.add_argument("--max-entries", type=int, default=1, help="Max entries per user")
    parser.add_argument("--field-size", type=int, default=10000, help="Total contest entries")
    parser.add_argument("--payout", type=str, default="top_heavy_gpp",
                        choices=list(PAYOUT_PRESETS.keys()),
                        help="Payout preset: top_heavy_gpp, flat, high_dollar_single")
    args = parser.parse_args()
    profile = ContestProfile(
        entry_fee=args.entry_fee,
        max_entries=args.max_entries,
        field_size=args.field_size,
        payout_preset=args.payout,
    )
    rec = recommend_leverage(profile)
    print_leverage_recommendation(profile, rec)


if __name__ == "__main__":
    main_cli()
