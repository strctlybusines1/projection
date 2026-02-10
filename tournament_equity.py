#!/usr/bin/env python3
"""
Tournament Equity (TE) Lineup Selector for NHL DFS
====================================================

v4 — Replaces heuristic scoring with DOLLAR VALUES.

Instead of scoring lineups 0-1 on abstract metrics, this calculates
the expected payout for each candidate lineup by:

1. Estimating each lineup's outcome distribution (mean + std)
2. Simulating outcomes across the score->payout curve
3. Picking the lineup with highest E[payout] - entry_fee

Built from 105 actual $121 SE contests (7,428 entries):
  Score ~110: EV = $-3  (breakeven)
  Score ~130: EV = $+259
  Score ~150: EV = $+688

KEY FINDING: Increasing lineup std from 18→30 is worth MORE than
increasing mean from 85→95. VARIANCE IS THE PRODUCT.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════
#  Score → Payout Lookup
# ═══════════════════════════════════════════════════════════════════

LOOKUP_PATH = Path(__file__).parent / "data" / "score_payout_lookup.json"

def load_payout_curve() -> Dict[int, float]:
    """Load the empirical score->payout lookup from 105 contests."""
    try:
        with open(LOOKUP_PATH) as f:
            raw = json.load(f)
        return {int(k): float(v) for k, v in raw.items()}
    except FileNotFoundError:
        # Hardcoded fallback from analysis
        return {
            0: 0, 5: 0, 10: 0, 15: 0, 20: 0, 25: 0, 30: 0, 35: 0,
            40: 0, 45: 0, 50: 0, 55: 0, 60: 0, 65: 0, 70: 0, 75: 0,
            80: 0, 85: 0, 90: 0, 95: 22, 100: 37, 105: 93,
            110: 145, 115: 231, 120: 275, 125: 392, 130: 365,
            135: 479, 140: 664, 145: 676, 150: 994, 155: 810,
            160: 1156, 165: 1073, 170: 969, 175: 1223, 180: 998,
            185: 1188, 190: 1484, 195: 1236, 200: 1836
        }

PAYOUT_CURVE = load_payout_curve()


# ═══════════════════════════════════════════════════════════════════
#  Intra-Team Correlation Parameters
# ═══════════════════════════════════════════════════════════════════

# From game log analysis: teammates have correlated outcomes.
# When a team scores, multiple players benefit.
# Estimated pairwise correlation between teammates: ~0.15-0.35
# Linemates (same line): ~0.30
# Same team, different lines: ~0.15

CORR_SAME_LINE = 0.30
CORR_SAME_TEAM = 0.15
CORR_GOALIE_OPP = -0.10  # goalie vs opposing skaters (negative)


# ═══════════════════════════════════════════════════════════════════
#  Tournament Equity Calculator
# ═══════════════════════════════════════════════════════════════════

class TournamentEquitySelector:
    """
    Score lineups by Tournament Equity (expected $ payout).
    
    For each candidate lineup:
    1. Compute lineup mean (sum of projected_fpts)
    2. Compute lineup std (from player stds + correlation boost)
    3. Integrate over score distribution × payout curve
    4. TE = E[payout]; EV = TE - entry_fee
    """

    def __init__(
        self,
        player_pool: pd.DataFrame,
        entry_fee: float = 121.0,
        n_simulations: int = 10000,
        payout_curve: Dict[int, float] = None,
        stack_builder=None,
    ):
        self.pool = player_pool.copy()
        self.entry_fee = entry_fee
        self.n_sim = n_simulations
        self.payout_curve = payout_curve or PAYOUT_CURVE
        self.stack_builder = stack_builder

        # Build linemate lookup for correlation estimation
        self._linemate_sets = {}
        if stack_builder and hasattr(stack_builder, 'lines_data'):
            for team, data in stack_builder.lines_data.items():
                self._linemate_sets[team] = []
                for key in ['line1', 'line2', 'line3', 'line4']:
                    line = data.get(key, {})
                    players = [line.get(p) for p in ['lw', 'c', 'rw'] if line.get(p)]
                    if len(players) >= 2:
                        self._linemate_sets[team].append(set(players))

        # Build the payout interpolation function
        self._payout_bins = sorted(self.payout_curve.keys())
        self._payout_vals = [self.payout_curve[b] for b in self._payout_bins]

    def _get_payout(self, score: float) -> float:
        """Get expected payout for a given score using linear interpolation."""
        if score <= self._payout_bins[0]:
            return 0.0
        if score >= self._payout_bins[-1]:
            return self._payout_vals[-1]

        # Find surrounding bins
        for i in range(len(self._payout_bins) - 1):
            if self._payout_bins[i] <= score < self._payout_bins[i + 1]:
                lo, hi = self._payout_bins[i], self._payout_bins[i + 1]
                lo_v, hi_v = self._payout_vals[i], self._payout_vals[i + 1]
                frac = (score - lo) / (hi - lo)
                return lo_v + frac * (hi_v - lo_v)
        return 0.0

    def _estimate_lineup_distribution(self, lineup: pd.DataFrame) -> Tuple[float, float]:
        """
        Estimate lineup outcome distribution (mean, std).
        
        Accounts for:
        - Individual player projection variance
        - Intra-team correlation (teammates boost variance)
        - Linemate correlation (same-line players boost more)
        """
        # Mean: sum of projected FPTS
        mean = lineup['projected_fpts'].sum()

        # Individual variance: use ceiling-based estimate if available
        player_vars = []
        for _, p in lineup.iterrows():
            proj = p['projected_fpts']
            
            if 'ceiling' in p.index and pd.notna(p.get('ceiling')) and p.get('ceiling', 0) > 0:
                # Estimate std from ceiling: ceiling ≈ mean + 2*std
                ceil = p['ceiling']
                player_std = max((ceil - proj) / 2, proj * 0.3)
            elif 'dk_stdv' in p.index and pd.notna(p.get('dk_stdv')) and p.get('dk_stdv', 0) > 0:
                player_std = p['dk_stdv']
            else:
                # Fallback: std ≈ 60% of projection (from DK data analysis)
                if p.get('position') == 'G':
                    player_std = max(proj * 0.80, 4.0)  # Goalies have highest variance
                else:
                    player_std = max(proj * 0.60, 2.0)
            
            player_vars.append(player_std ** 2)

        # Base variance (independent)
        base_var = sum(player_vars)

        # Correlation boost: teammates
        teams = lineup.groupby('team')
        corr_boost = 0.0

        for team_name, team_players in teams:
            if len(team_players) < 2:
                continue

            players_in_team = team_players[team_players['position'] != 'G']
            goalie_in_team = team_players[team_players['position'] == 'G']
            n_sk = len(players_in_team)
            
            if n_sk < 2:
                continue

            # Get individual stds for this team's players
            team_stds = []
            for _, p in players_in_team.iterrows():
                proj = p['projected_fpts']
                if 'ceiling' in p.index and pd.notna(p.get('ceiling')) and p.get('ceiling', 0) > 0:
                    team_stds.append(max((p['ceiling'] - proj) / 2, proj * 0.3))
                else:
                    team_stds.append(max(proj * 0.60, 2.0))

            # Check for linemate pairs (higher correlation)
            names = players_in_team['name'].tolist()
            n_linemate_pairs = self._count_linemate_pairs(names, team_name)
            n_other_pairs = n_sk * (n_sk - 1) // 2 - n_linemate_pairs

            # Correlation contribution: 2 * rho * std_i * std_j for each pair
            avg_std = np.mean(team_stds) if team_stds else 3.0
            corr_boost += n_linemate_pairs * 2 * CORR_SAME_LINE * avg_std * avg_std
            corr_boost += n_other_pairs * 2 * CORR_SAME_TEAM * avg_std * avg_std

        total_var = base_var + corr_boost
        total_std = np.sqrt(max(total_var, 1.0))

        return mean, total_std

    def _count_linemate_pairs(self, names: list, team: str) -> int:
        """Count how many player pairs are linemates."""
        pairs = 0
        for line_set in self._linemate_sets.get(team, []):
            overlap = sum(1 for n in names if self._fuzzy_match(n, line_set))
            if overlap >= 2:
                pairs += overlap * (overlap - 1) // 2
        return pairs

    @staticmethod
    def _fuzzy_match(name: str, name_set: set) -> bool:
        nl = name.lower()
        for s in name_set:
            sl = s.lower()
            if nl == sl:
                return True
            n_last = nl.split()[-1] if nl.split() else nl
            s_last = sl.split()[-1] if sl.split() else sl
            if n_last == s_last and len(n_last) > 3:
                return True
        return False

    def compute_te(self, lineup: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Tournament Equity for a lineup.
        
        Returns dict with:
          te: Expected payout ($)
          ev: Expected value (te - entry_fee)
          mean: Lineup mean projection
          std: Lineup std (with correlation)
          p_cash: P(score >= 110)
          p_top5: P(score >= 140)
          p_win:  P(score >= 150)
        """
        mean, std = self._estimate_lineup_distribution(lineup)

        # Monte Carlo: sample from lineup distribution, compute average payout
        np.random.seed(None)  # Random seed for each call
        samples = np.random.normal(mean, std, self.n_sim)
        samples = np.maximum(samples, 0)  # Floor at 0

        payouts = np.array([self._get_payout(s) for s in samples])
        te = float(np.mean(payouts))
        ev = te - self.entry_fee

        # Probability metrics
        p_cash = float(np.mean(samples >= 110))
        p_top5 = float(np.mean(samples >= 140))
        p_win = float(np.mean(samples >= 150))

        return {
            'te': te,
            'ev': ev,
            'mean': mean,
            'std': std,
            'p_cash': p_cash,
            'p_top5': p_top5,
            'p_win': p_win,
        }

    def compute_te_analytical(self, lineup: pd.DataFrame) -> Dict[str, float]:
        """
        Analytical TE computation (faster, no sampling noise).
        Integrates normal PDF × payout curve numerically.
        """
        from scipy import stats as sp_stats

        mean, std = self._estimate_lineup_distribution(lineup)

        te = 0.0
        for i in range(len(self._payout_bins) - 1):
            lo = self._payout_bins[i]
            hi = self._payout_bins[i + 1]
            mid = (lo + hi) / 2
            payout = (self._payout_vals[i] + self._payout_vals[i + 1]) / 2

            # P(score in [lo, hi])
            p = sp_stats.norm.cdf(hi, mean, std) - sp_stats.norm.cdf(lo, mean, std)
            te += p * payout

        ev = te - self.entry_fee
        p_cash = 1 - sp_stats.norm.cdf(110, mean, std)
        p_top5 = 1 - sp_stats.norm.cdf(140, mean, std)
        p_win = 1 - sp_stats.norm.cdf(150, mean, std)

        return {
            'te': te,
            'ev': ev,
            'mean': mean,
            'std': std,
            'p_cash': p_cash,
            'p_top5': p_top5,
            'p_win': p_win,
        }

    def select(
        self,
        candidate_lineups: List[pd.DataFrame],
        verbose: bool = True,
        use_analytical: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Score all candidates by TE and return the best."""
        if not candidate_lineups:
            raise ValueError("No candidate lineups provided")

        compute_fn = self.compute_te_analytical if use_analytical else self.compute_te

        scored = []
        for lu in candidate_lineups:
            try:
                te_result = compute_fn(lu)
                scored.append((lu, te_result))
            except Exception as e:
                scored.append((lu, {'te': 0, 'ev': -self.entry_fee, 'mean': 0, 'std': 0,
                                     'p_cash': 0, 'p_top5': 0, 'p_win': 0}))

        scored.sort(key=lambda x: x[1]['te'], reverse=True)

        if verbose:
            self._print_results(scored)

        return scored[0][0], scored[0][1]

    def _print_results(self, scored, n_show=10):
        """Print TE results for all candidates."""
        print(f"\n{'='*105}")
        print(f" TOURNAMENT EQUITY SELECTOR v4 | Entry: ${self.entry_fee:.0f}")
        print(f" {len(scored)} candidates scored | Payout curve from 105 contests")
        print(f"{'='*105}")

        header = (f"{'#':>3} {'TE':>7} {'EV':>7} {'Mean':>6} {'Std':>5} "
                  f"{'P(cash)':>7} {'P(top5)':>7} {'P(win)':>7} "
                  f"{'Sal':>7} {'Goalie':<14} {'Stack':<10}")
        print(f"\n{header}")
        print("-" * len(header))

        for i in range(min(len(scored), n_show)):
            lu, te = scored[i]
            sal = lu['salary'].sum()
            g = lu[lu['position'] == 'G']
            gn = g.iloc[0]['name'][:13] if not g.empty else "???"
            sk = lu[lu['position'] != 'G']
            stk = '+'.join(f"{t}{c}" for t, c in sk['team'].value_counts().items() if c >= 2)[:9]
            m = " <<<" if i == 0 else ""

            row = (f"{i+1:>3} ${te['te']:>6.0f} ${te['ev']:>+6.0f} {te['mean']:>6.1f} {te['std']:>5.1f} "
                   f"{te['p_cash']*100:>6.1f}% {te['p_top5']*100:>6.1f}% {te['p_win']*100:>6.2f}% "
                   f"${sal:>6,} {gn:<14} {stk:<10}{m}")
            print(row)

        # Detailed breakdown of #1
        best_lu, best_te = scored[0]
        worst_te = scored[-1][1]

        print(f"\n  Selected Lineup:")
        print(f"    Tournament Equity:  ${best_te['te']:.0f}")
        print(f"    Expected Value:     ${best_te['ev']:+.0f} per contest")
        print(f"    Lineup Mean:        {best_te['mean']:.1f} FPTS")
        print(f"    Lineup Std:         {best_te['std']:.1f} FPTS (with correlation)")
        print(f"    P(cash ≥110):       {best_te['p_cash']*100:.1f}%")
        print(f"    P(top-5 ≥140):      {best_te['p_top5']*100:.1f}%")
        print(f"    P(win ≥150):        {best_te['p_win']*100:.2f}%")
        print(f"    TE spread (1st-last): ${best_te['te'] - worst_te['te']:.0f}")


def print_te_lineup(lineup: pd.DataFrame, te_result: Dict[str, float]):
    """Print the TE-selected lineup with dollar values."""
    print(f"\n{'='*80}")
    print(f" TOURNAMENT EQUITY LINEUP (FINAL)")
    print(f"{'='*80}")

    sal = lineup['salary'].sum()
    print(f"  Salary: ${sal:,} / $50,000 (${50000 - sal:,} remaining)")
    print(f"  Projected Mean: {te_result['mean']:.1f} FPTS")
    print(f"  Lineup Std: {te_result['std']:.1f} FPTS")
    print(f"  Tournament Equity: ${te_result['te']:.0f}")
    print(f"  Expected Value: ${te_result['ev']:+.0f} per contest")
    print(f"  P(cash): {te_result['p_cash']*100:.1f}% | P(top-5): {te_result['p_top5']*100:.1f}% | P(win): {te_result['p_win']*100:.2f}%")

    sk = lineup[lineup['position'] != 'G']
    tc = sk['team'].value_counts()
    stacks = [f"{t} x{c}" for t, c in tc.items() if c >= 2]
    ones = [t for t, c in tc.items() if c == 1]
    if stacks: print(f"  Stacks: {', '.join(stacks)}")
    if ones: print(f"  One-offs: {', '.join(ones)}")
    print()

    display = lineup.copy()
    if 'roster_slot' in display.columns:
        so = {'G':0,'C1':1,'C2':2,'W1':3,'W2':4,'W3':5,'D1':6,'D2':7,'UTIL':8}
        display['_o'] = display['roster_slot'].map(lambda x: so.get(x, 9))
        display = display.sort_values('_o')
        sc = 'roster_slot'
    else:
        po = {'C':0,'W':2,'D':3,'G':4}
        display['_o'] = display['position'].map(lambda x: po.get(x, 5))
        display = display.sort_values('_o')
        sc = 'position'

    h = f"  {'Slot':<6} {'Name':<28} {'Team':<5} {'Salary':<9} {'Proj':>6} {'Ceil':>6}"
    print(h)
    print("  " + "-" * (len(h) - 2))
    for _, r in display.iterrows():
        cv = r.get('ceiling', 0) if pd.notna(r.get('ceiling', 0)) else 0
        print(f"  {r[sc]:<6} {r['name']:<28} {r['team']:<5} ${r['salary']:<8,} {r['projected_fpts']:>6.1f} {cv:>6.1f}")
    print()
