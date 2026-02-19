#!/usr/bin/env python3
"""
Payout-Weighted Expected Value Calculator

Based on: Newell & Easton (2017) "Optimizing Daily Fantasy Sports Contests
Through Stochastic Integer Programming" — Kansas State University

Core idea: Instead of scoring lineups by M+3σ (a single upside metric),
compute the actual dollar expected value by integrating across all payout
tiers: E[payout] = Σ P(finishing in tier_j) × payout_j

This accounts for the SHAPE of the payout curve. A lineup with μ=130, σ=20
has different EV than μ=120, σ=30 depending on whether the payouts are
top-heavy (favors high σ) or flat (favors high μ).

Empirical calibration from 105 DraftKings $121 SE NHL contests (7,428 entries):
    1st place:  ~151 FPTS
    Cash line:  ~113 FPTS  (top ~22%)
    Field mean:  92.4 FPTS
    Field std:   27.8 FPTS
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════
# EMPIRICAL SCORE THRESHOLDS
# From 105 DK $121 SE NHL contests, 27-91 entries each
# These map: "to finish in place X, you need Y FPTS on average"
# ══════════════════════════════════════════════════════════════════

# (percentile_from_top, avg_score_needed)
# Percentile = place / n_entries
EMPIRICAL_THRESHOLDS = [
    (0.013, 151.2),  # 1st
    (0.026, 147.0),  # 2nd
    (0.038, 140.7),  # 3rd
    (0.051, 136.6),  # 4th
    (0.064, 133.9),  # 5th
    (0.077, 130.7),  # 6th
    (0.090, 128.5),  # 7th
    (0.103, 126.4),  # 8th
    (0.115, 125.0),  # 9th
    (0.128, 122.9),  # 10th
    (0.141, 121.3),  # 11th
    (0.154, 119.8),  # 12th
    (0.167, 118.3),  # 13th
    (0.179, 116.7),  # 14th
    (0.192, 115.1),  # 15th
    (0.205, 114.1),  # 16th
    (0.218, 112.5),  # 17th (typical cash line)
]


@dataclass
class PayoutTier:
    """A single payout tier in the contest."""
    place: int
    payout: float
    score_threshold: float  # FPTS needed to finish at this place


@dataclass
class ContestStructure:
    """Full contest payout structure."""
    entry_fee: float
    n_entries: int
    tiers: List[PayoutTier] = field(default_factory=list)
    total_prize: float = 0.0

    @classmethod
    def from_csv(cls, path: str) -> 'ContestStructure':
        """
        Parse a contest structure CSV.
        Expected columns: Place, Prize_Payout (plus header row with
        Total_Entries, Total_Prize$, Entry_Fee).
        """
        df = pd.read_csv(path, encoding='utf-8-sig')

        # Extract header info
        entry_fee = pd.to_numeric(df['Entry_Fee'].iloc[0], errors='coerce')
        n_entries = int(pd.to_numeric(df['Total_Entries'].iloc[0], errors='coerce'))
        total_prize = pd.to_numeric(df['Total_Prize$'].iloc[0], errors='coerce')

        # Build tiers
        tiers = []
        for _, row in df.iterrows():
            place = pd.to_numeric(row.get('Place'), errors='coerce')
            payout = pd.to_numeric(row.get('Prize_Payout'), errors='coerce')
            if pd.isna(place) or pd.isna(payout):
                continue

            place = int(place)
            pct = place / n_entries

            # Interpolate score threshold from empirical data
            threshold = _interpolate_threshold(pct)
            tiers.append(PayoutTier(place=place, payout=payout,
                                    score_threshold=threshold))

        return cls(entry_fee=entry_fee, n_entries=n_entries,
                   tiers=tiers, total_prize=total_prize)

    @classmethod
    def default_121_se(cls) -> 'ContestStructure':
        """
        Default $121 SE GPP structure based on typical DK NHL contests.
        78 entries, ~$8,500 prize pool.
        """
        payouts = [
            (1, 2000), (2, 1250), (3, 800), (4, 600), (5, 500),
            (6, 400), (7, 350), (8, 300), (9, 300), (10, 275),
            (11, 275), (12, 250), (13, 250), (14, 250), (15, 250),
            (16, 250), (17, 250),
        ]
        n_entries = 78
        tiers = []
        for place, payout in payouts:
            pct = place / n_entries
            threshold = _interpolate_threshold(pct)
            tiers.append(PayoutTier(place=place, payout=payout,
                                    score_threshold=threshold))

        return cls(entry_fee=121, n_entries=n_entries,
                   tiers=tiers, total_prize=8500)


def _interpolate_threshold(pct: float) -> float:
    """
    Interpolate the FPTS score threshold for a given percentile
    using empirical data from 105 contests.
    """
    for i in range(len(EMPIRICAL_THRESHOLDS) - 1):
        p1, s1 = EMPIRICAL_THRESHOLDS[i]
        p2, s2 = EMPIRICAL_THRESHOLDS[i + 1]
        if p1 <= pct <= p2:
            t = (pct - p1) / (p2 - p1)
            return s1 + t * (s2 - s1)

    # Extrapolate beyond bounds
    if pct < EMPIRICAL_THRESHOLDS[0][0]:
        # Above 1st place — extrapolate upward
        return EMPIRICAL_THRESHOLDS[0][1] + (EMPIRICAL_THRESHOLDS[0][0] - pct) * 200
    else:
        # Below cash line — extrapolate downward
        return EMPIRICAL_THRESHOLDS[-1][1] - (pct - EMPIRICAL_THRESHOLDS[-1][0]) * 100


@dataclass
class PayoutEVResult:
    """Result of a payout EV calculation for a single lineup."""
    expected_payout: float      # E[$] across all tiers
    expected_profit: float      # E[$] - entry_fee
    roi: float                  # (E[$] - entry_fee) / entry_fee
    p_cash: float               # P(finishing in any paid position)
    p_first: float              # P(finishing 1st)
    p_top5: float               # P(finishing top 5)
    tier_probs: Dict[int, float]  # {place: P(finishing at this place)}
    tier_ev: Dict[int, float]     # {place: P × payout for this tier}
    lineup_mean: float
    lineup_std: float


def compute_payout_ev(
    lineup_mean: float,
    lineup_std: float,
    contest: ContestStructure,
    leverage_mult: float = 1.0,
) -> PayoutEVResult:
    """
    Compute the payout-weighted expected value of a lineup.

    Uses the Newell & Easton (2017) framework:
    E[payout] = Σ P(score ∈ [threshold_j, threshold_{j-1})) × payout_j

    The lineup is modeled as Normal(μ, σ²) per Newell's assumption.
    While individual players aren't truly normal, the Central Limit
    Theorem makes the 9-player sum approximately normal, and our
    simulator already validates this.

    Args:
        lineup_mean: Expected total FPTS (μ)
        lineup_std: Standard deviation of total FPTS (σ)
        contest: Contest payout structure
        leverage_mult: Optional multiplier from LeverageScorer.
                       Applied as a tilt on the score distribution:
                       we shift the "effective" score by (mult-1) × σ,
                       which represents the ownership leverage edge.

    Returns:
        PayoutEVResult with full breakdown
    """
    if lineup_std <= 0:
        lineup_std = 0.01  # Prevent div by zero

    # Apply leverage as a mean shift
    # Intuition: leverage_mult > 1.0 means the lineup is under-owned,
    # so it has an "edge" in the contest field. We model this as a
    # small boost to the effective mean — the lineup performs as if
    # it were slightly better positioned in the field.
    # A mult of 1.10 shifts the mean up by 0.10 × σ (~2 FPTS).
    effective_mean = lineup_mean + (leverage_mult - 1.0) * lineup_std

    tier_probs = {}
    tier_ev = {}
    total_ev = 0.0

    # Sort tiers by place (1st, 2nd, ...)
    sorted_tiers = sorted(contest.tiers, key=lambda t: t.place)

    for i, tier in enumerate(sorted_tiers):
        # P(finishing at exactly this place) ≈ P(score ∈ [this_threshold, prev_threshold))
        upper = sorted_tiers[i - 1].score_threshold if i > 0 else float('inf')
        lower = tier.score_threshold

        p_above_lower = 1 - norm.cdf(lower, loc=effective_mean, scale=lineup_std)
        p_above_upper = 1 - norm.cdf(upper, loc=effective_mean, scale=lineup_std) if upper < float('inf') else 0.0

        p_tier = p_above_lower - p_above_upper
        p_tier = max(p_tier, 0.0)  # Numerical safety

        tier_probs[tier.place] = p_tier
        ev_contribution = p_tier * tier.payout
        tier_ev[tier.place] = ev_contribution
        total_ev += ev_contribution

    # Aggregate probabilities
    p_cash = sum(tier_probs.values())
    p_first = tier_probs.get(1, 0.0)
    p_top5 = sum(tier_probs.get(p, 0.0) for p in range(1, 6))

    return PayoutEVResult(
        expected_payout=total_ev,
        expected_profit=total_ev - contest.entry_fee,
        roi=(total_ev - contest.entry_fee) / contest.entry_fee if contest.entry_fee > 0 else 0,
        p_cash=p_cash,
        p_first=p_first,
        p_top5=p_top5,
        tier_probs=tier_probs,
        tier_ev=tier_ev,
        lineup_mean=lineup_mean,
        lineup_std=lineup_std,
    )


def score_lineups_ev(
    lineups: List[pd.DataFrame],
    sim_results: List[Dict],
    contest: ContestStructure,
    leverage_mults: Optional[List[float]] = None,
    verbose: bool = True,
) -> List[Tuple[pd.DataFrame, Dict, PayoutEVResult]]:
    """
    Score and rank lineups by payout-weighted EV.

    Args:
        lineups: Candidate lineups from optimizer
        sim_results: Matching sim results (need 'mean' and 'std')
        contest: Contest payout structure
        leverage_mults: Optional per-lineup leverage multipliers
        verbose: Print ranking table

    Returns:
        List of (lineup, sim_result, ev_result) sorted by EV descending
    """
    scored = []

    for i, (lu, res) in enumerate(zip(lineups, sim_results)):
        mu = res.get('mean', 100)
        sigma = res.get('std', 25)
        mult = leverage_mults[i] if leverage_mults else 1.0

        ev = compute_payout_ev(mu, sigma, contest, leverage_mult=mult)
        scored.append((lu, res, ev))

    scored.sort(key=lambda x: x[2].expected_payout, reverse=True)

    if verbose:
        _print_ev_table(scored, contest)

    return scored


def _print_ev_table(scored, contest, n_show=15):
    """Print the EV ranking table."""
    print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
    print(f"  ║  PAYOUT-WEIGHTED EV RANKING                                 ║")
    print(f"  ║  Contest: ${contest.entry_fee} SE | {contest.n_entries} entries "
          f"| ${contest.total_prize:,.0f} prize pool     ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════╝\n")

    print(f"  {'#':>3} {'Mean':>6} {'Std':>5} {'M+3σ':>6} {'E[$]':>7} {'E[Profit]':>9} "
          f"{'ROI':>7} {'P(Cash)':>8} {'P(1st)':>7} {'P(Top5)':>8}")
    print(f"  {'-' * 85}")

    for i, (lu, res, ev) in enumerate(scored[:n_show]):
        m3s = res.get('m3s', res['mean'] + 3 * res['std'])
        print(f"  {i+1:>3} {ev.lineup_mean:>6.0f} {ev.lineup_std:>5.0f} {m3s:>6.0f} "
              f"${ev.expected_payout:>6.2f} ${ev.expected_profit:>+8.2f} "
              f"{ev.roi:>+6.1%} {ev.p_cash:>7.1%} {ev.p_first:>6.2%} {ev.p_top5:>7.2%}")

    # Summary
    best = scored[0][2]
    print(f"\n  ✅ Top EV lineup: E[payout]=${best.expected_payout:.2f} "
          f"(profit=${best.expected_profit:+.2f}, ROI={best.roi:+.1%})")
    print(f"     μ={best.lineup_mean:.0f}, σ={best.lineup_std:.0f}, "
          f"P(cash)={best.p_cash:.1%}, P(1st)={best.p_first:.2%}")

    # Show EV breakdown for top pick
    print(f"\n  EV breakdown by tier:")
    top_tiers = sorted(best.tier_ev.items(), key=lambda x: x[1], reverse=True)[:5]
    for place, ev_contrib in top_tiers:
        prob = best.tier_probs[place]
        payout = [t.payout for t in sorted(scored[0][2].tier_probs.keys()) 
                  if t == place]
        # Find actual payout from contest structure
        tier_payout = next((t.payout for t in scored[0][2].__class__.__mro__), 0)
        print(f"    Place {place:>2}: P={prob:.2%} × payout → ${ev_contrib:.2f}")


def compare_m3s_vs_ev(
    lineups: List[pd.DataFrame],
    sim_results: List[Dict],
    contest: ContestStructure,
    leverage_mults: Optional[List[float]] = None,
) -> Dict:
    """
    Compare M+3σ ranking vs EV ranking to see if they diverge.
    Returns summary stats about agreement/disagreement.
    """
    # M+3σ ranking
    m3s_scored = []
    for i, (lu, res) in enumerate(zip(lineups, sim_results)):
        m3s = res.get('m3s', res['mean'] + 3 * res['std'])
        m3s_scored.append((i, m3s))
    m3s_scored.sort(key=lambda x: x[1], reverse=True)
    m3s_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(m3s_scored)}

    # EV ranking
    ev_scored = []
    for i, (lu, res) in enumerate(zip(lineups, sim_results)):
        mult = leverage_mults[i] if leverage_mults else 1.0
        ev = compute_payout_ev(res['mean'], res['std'], contest, mult)
        ev_scored.append((i, ev.expected_payout))
    ev_scored.sort(key=lambda x: x[1], reverse=True)
    ev_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(ev_scored)}

    # Compare
    same_top1 = m3s_scored[0][0] == ev_scored[0][0]
    same_top3 = set(x[0] for x in m3s_scored[:3]) == set(x[0] for x in ev_scored[:3])

    rank_diffs = [abs(m3s_rank[i] - ev_rank[i]) for i in range(len(lineups))]

    return {
        'same_top1': same_top1,
        'same_top3': same_top3,
        'avg_rank_diff': np.mean(rank_diffs),
        'max_rank_diff': max(rank_diffs),
        'm3s_top1_idx': m3s_scored[0][0],
        'ev_top1_idx': ev_scored[0][0],
    }


# ══════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    contest = ContestStructure.default_121_se()

    print("=" * 70)
    print("  PAYOUT EV CALCULATOR — Sensitivity Analysis")
    print("=" * 70)

    print(f"\n  Contest: ${contest.entry_fee} SE | {contest.n_entries} entries\n")
    print(f"  {'Lineup':>25} {'E[$]':>8} {'Profit':>9} {'ROI':>7} "
          f"{'P(Cash)':>8} {'P(1st)':>7} {'P(Top5)':>8}")
    print(f"  {'-' * 78}")

    # Test different mean/std combos
    test_cases = [
        ("Low μ, Low σ (chalk)",     100, 18),
        ("Low μ, High σ (contrarian)", 100, 30),
        ("Med μ, Low σ",             120, 18),
        ("Med μ, Med σ",             120, 25),
        ("Med μ, High σ",            120, 30),
        ("High μ, Low σ (safe)",     140, 18),
        ("High μ, Med σ (balanced)", 140, 25),
        ("High μ, High σ (GPP)",     140, 30),
        ("Elite μ, Med σ",           155, 25),
        ("Elite μ, High σ",          155, 30),
    ]

    for name, mu, sigma in test_cases:
        ev = compute_payout_ev(mu, sigma, contest)
        print(f"  {name:>25} ${ev.expected_payout:>7.2f} ${ev.expected_profit:>+8.2f} "
              f"{ev.roi:>+6.1%} {ev.p_cash:>7.1%} {ev.p_first:>6.2%} {ev.p_top5:>7.2%}")

    # Show how leverage tilts the EV
    print(f"\n\n  LEVERAGE IMPACT on μ=130, σ=25 lineup:")
    print(f"  {'Mult':>6} {'E[$]':>8} {'Profit':>9} {'ROI':>7} {'P(Cash)':>8}")
    print(f"  {'-' * 45}")
    for mult in [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]:
        ev = compute_payout_ev(130, 25, contest, leverage_mult=mult)
        print(f"  {mult:>5.2f}x ${ev.expected_payout:>7.2f} ${ev.expected_profit:>+8.2f} "
              f"{ev.roi:>+6.1%} {ev.p_cash:>7.1%}")

    # M+3σ vs EV: when do they disagree?
    print(f"\n\n  M+3σ vs EV: KEY COMPARISON")
    print(f"  Two lineups with SAME M+3σ but different μ/σ splits:\n")

    # Same M+3σ = 210, but different profiles
    pairs = [
        ("A: μ=135, σ=25 → M+3σ=210", 135, 25),
        ("B: μ=120, σ=30 → M+3σ=210", 120, 30),
        ("C: μ=150, σ=20 → M+3σ=210", 150, 20),
    ]
    for name, mu, sigma in pairs:
        ev = compute_payout_ev(mu, sigma, contest)
        print(f"  {name}")
        print(f"    E[$]=${ev.expected_payout:.2f}  P(cash)={ev.p_cash:.1%}  "
              f"P(1st)={ev.p_first:.2%}  P(top5)={ev.p_top5:.2%}")
