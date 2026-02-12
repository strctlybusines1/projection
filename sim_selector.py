#!/usr/bin/env python3
"""
Simulation-Based Lineup Selector for NHL DFS
==============================================

Replaces M+3σ (TournamentEquitySelector) with a correlated Monte Carlo
approach that models each player as a zero-inflated lognormal distribution
and simulates lineup outcomes with measured team correlations.

Backtest result (5 slates, 85 candidates each):
  Sim P(target):  +14.2 FPTS/slate  (26% of best possible edge)
  M+3σ:           +6.5  FPTS/slate  (12% of best possible edge)
  Max Projection:  +2.6 FPTS/slate  ( 5% of best possible edge)

Usage:
    from sim_selector import SimSelector, print_sim_lineup

    selector = SimSelector(player_pool)
    best_lineup, result = selector.select(candidates, verbose=True)
    print_sim_lineup(best_lineup, result)

Integration with main.py:
    Replace the TournamentEquitySelector block with SimSelector.
    Uses the same interface: .select(candidates) → (lineup, result_dict)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from simulation_engine import SimulationEngine

DATA_DIR = Path(__file__).parent / "data"
DK_SEASON_DIR = Path.home() / "dk_salaries_season" / "DKSalaries_NHL_season"

# Score → payout from 105 actual $121 SE contests
def _load_payout_curve():
    try:
        with open(DATA_DIR / "score_payout_lookup.json") as f:
            return {int(k): float(v) for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {
            0: 0, 50: 0, 80: 0, 90: 0, 95: 22, 100: 37, 105: 93,
            110: 145, 115: 231, 120: 275, 125: 392, 130: 365,
            135: 479, 140: 664, 145: 676, 150: 994, 155: 810,
            160: 1156, 170: 969, 180: 998, 190: 1484, 200: 1836,
        }


class SimSelector:
    """
    Monte Carlo simulation selector for DFS lineup selection.

    Instead of mean + 3σ (normal assumption), this:
    1. Fits zero-inflated lognormal distributions per player
    2. Simulates correlated outcomes via Cholesky decomposition
    3. Selects lineup with highest P(exceed target) or E[payout]

    Parameters:
        player_pool: DataFrame with columns [name, team, position, salary, projected_fpts]
        entry_fee: Contest entry fee for EV calculation
        n_sims: Number of Monte Carlo simulations per lineup
        history_df: Historical game scores for distribution fitting
                    If None, loads from DK season files automatically
        target_date: Current slate date (for excluding future data from history)
    """

    def __init__(
        self,
        player_pool: pd.DataFrame,
        entry_fee: float = 121.0,
        n_sims: int = 8000,
        history_df: pd.DataFrame = None,
        target_date: str = None,
        stack_builder=None,
    ):
        self.player_pool = player_pool
        self.entry_fee = entry_fee
        self.n_sims = n_sims
        self.stack_builder = stack_builder
        self.target_date = target_date

        # Load payout curve
        self.payout_curve = _load_payout_curve()
        self._payout_bins = sorted(self.payout_curve.keys())
        self._payout_vals = [self.payout_curve[b] for b in self._payout_bins]

        # Build simulation engine
        self.engine = SimulationEngine(n_sims=n_sims)

        # Load or use provided history
        if history_df is not None:
            hist = history_df
        else:
            hist = self._load_dk_history()

        # Fit player distributions
        self.engine.fit_player_distributions(player_pool, hist, date_str=target_date)

    def _load_dk_history(self) -> pd.DataFrame:
        """Load historical scores from DK season files."""
        import glob
        files = sorted(glob.glob(str(DK_SEASON_DIR / "draftkings_NHL_*.csv")))

        if not files:
            # Fallback: try relative path
            alt_dir = Path(__file__).parent.parent.parent / "dk_salaries_season" / "DKSalaries_NHL_season"
            files = sorted(glob.glob(str(alt_dir / "draftkings_NHL_*.csv")))

        if not files:
            print("  ⚠ No DK season history found — sim distributions will use pool projections only")
            return pd.DataFrame(columns=['Player', 'Team', 'Score', 'Pos', 'slate_date'])

        all_data = []
        for f in files:
            try:
                import os
                date = os.path.basename(f).replace('draftkings_NHL_', '').replace('_players.csv', '')
                df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
                df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
                df['slate_date'] = date
                all_data.append(df[df['Score'].notna()][['Player', 'Team', 'Score', 'Pos', 'slate_date']])
            except Exception:
                continue

        if all_data:
            hist = pd.concat(all_data, ignore_index=True)
            print(f"  Loaded {len(hist)} historical scores from {len(all_data)} DK files")
            return hist
        return pd.DataFrame(columns=['Player', 'Team', 'Score', 'Pos', 'slate_date'])

    def _compute_ev(self, sim_totals: np.ndarray) -> float:
        """Compute expected payout from simulated lineup totals using payout curve."""
        ev = 0.0
        n = len(sim_totals)
        for i in range(len(self._payout_bins) - 1):
            lo = self._payout_bins[i]
            hi = self._payout_bins[i + 1]
            payout = (self._payout_vals[i] + self._payout_vals[i + 1]) / 2
            count = np.sum((sim_totals >= lo) & (sim_totals < hi))
            ev += (count / n) * payout

        # Everything above max bin
        max_bin = self._payout_bins[-1]
        max_payout = self._payout_vals[-1]
        count = np.sum(sim_totals >= max_bin)
        ev += (count / n) * max_payout

        return ev

    def score_lineup(self, lineup: pd.DataFrame) -> Dict:
        """
        Score a single lineup via Monte Carlo simulation.

        Returns dict with simulation statistics and EV.
        """
        sim = self.engine.simulate_lineup(lineup, n_sims=self.n_sims)
        totals = sim['simulated_totals']

        ev = self._compute_ev(totals)

        return {
            'mean': sim['mean'],
            'std': sim['std'],
            'median': sim['median'],
            'p5': sim['p5'],
            'p25': sim['p25'],
            'p75': sim['p75'],
            'p95': sim['p95'],
            'p_cash': sim['p_cash'],      # P(≥111)
            'p_120': sim['p_120'],         # P(≥120)
            'p_top5': sim['p_gpp'],        # P(≥140)
            'p_win': float((totals >= 150).mean()),
            'max': sim['max'],
            'ev': ev,
            'te': ev,                      # Alias for TE compatibility
            'net_ev': ev - self.entry_fee,
            'upside_score': ev,            # For compatibility with TE selector sort
            'n_sims': self.n_sims,
        }

    def select(
        self,
        candidate_lineups: List[pd.DataFrame],
        verbose: bool = True,
        mode: str = 'ev',
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Score all candidates and return the best.

        mode:
            'ev':     Maximize E[payout] (default — best overall)
            'cash':   Maximize P(≥111)
            'gpp':    Maximize P(≥140)
            'ceiling': Maximize P95

        Returns: (best_lineup, result_dict)
        """
        if not candidate_lineups:
            raise ValueError("No candidate lineups provided")

        scored = []
        for i, lu in enumerate(candidate_lineups):
            try:
                result = self.score_lineup(lu)
                scored.append((lu, result))
            except Exception as e:
                scored.append((lu, {
                    'mean': 0, 'std': 0, 'median': 0,
                    'p5': 0, 'p25': 0, 'p75': 0, 'p95': 0,
                    'p_cash': 0, 'p_120': 0, 'p_top5': 0, 'p_win': 0,
                    'max': 0, 'ev': 0, 'te': 0, 'net_ev': -self.entry_fee,
                    'upside_score': 0, 'n_sims': 0,
                }))

        # Sort by selected criterion
        sort_keys = {
            'ev': lambda x: x[1]['ev'],
            'cash': lambda x: x[1]['p_cash'],
            'gpp': lambda x: x[1]['p_top5'],
            'ceiling': lambda x: x[1]['p95'],
        }
        sort_fn = sort_keys.get(mode, sort_keys['ev'])
        scored.sort(key=sort_fn, reverse=True)

        if verbose:
            self._print_results(scored, mode=mode)

        return scored[0][0], scored[0][1]

    def _print_results(self, scored, n_show=10, mode='ev'):
        """Print ranked lineup results."""
        n_show = min(n_show, len(scored))

        sort_label = {
            'ev': 'E[payout]', 'cash': 'P(cash)',
            'gpp': 'P(top5)', 'ceiling': 'P95',
        }.get(mode, 'E[payout]')

        print(f"\n{'═' * 90}")
        print(f"  SimSelector Results — {len(scored)} candidates × {self.n_sims:,} sims | sorted by {sort_label}")
        print(f"{'═' * 90}")
        print(f"  {'#':>3} {'Mean':>6} {'Std':>5} {'P(111)':>7} {'P(120)':>7} {'P(140)':>7} "
              f"{'P95':>5} {'E[$]':>6} {'Net EV':>7} {'Stacks':>8}")
        print(f"  {'-' * 82}")

        for rank, (lu, r) in enumerate(scored[:n_show], 1):
            # Detect stack structure
            stacks = _get_stack_desc(lu)

            marker = ' ◄' if rank == 1 else ''
            print(f"  {rank:>3} {r['mean']:>6.1f} {r['std']:>5.1f} "
                  f"{r['p_cash']:>6.1%} {r['p_120']:>6.1%} {r['p_top5']:>6.1%} "
                  f"{r['p95']:>5.0f} {r['ev']:>6.0f} {r['net_ev']:>+6.0f} "
                  f"{stacks:>8}{marker}")

        # Summary stats
        evs = [r['ev'] for _, r in scored]
        print(f"\n  Pool:  avg E[$]={np.mean(evs):.0f}  best={max(evs):.0f}  worst={min(evs):.0f}")
        print(f"  Selected: E[$]={scored[0][1]['ev']:.0f}  "
              f"P(cash)={scored[0][1]['p_cash']:.1%}  "
              f"P(top5)={scored[0][1]['p_top5']:.1%}")


def _get_stack_desc(lineup: pd.DataFrame) -> str:
    """Get a compact stack description like '4-3' or '3-3-2'."""
    team_col = 'team' if 'team' in lineup.columns else 'Team'
    if team_col not in lineup.columns:
        return '?'
    counts = lineup[team_col].value_counts().values
    counts = sorted(counts, reverse=True)
    return '-'.join(str(c) for c in counts if c >= 2)


def print_sim_lineup(lineup: pd.DataFrame, result: Dict):
    """Print the selected lineup with simulation details."""
    print(f"\n{'═' * 80}")
    print(f"  SIMULATION-SELECTED LINEUP")
    print(f"{'═' * 80}")

    # Determine column names
    name_col = 'name' if 'name' in lineup.columns else 'Player'
    team_col = 'team' if 'team' in lineup.columns else 'Team'
    pos_col = 'position' if 'position' in lineup.columns else 'Pos'
    sal_col = 'salary' if 'salary' in lineup.columns else 'Salary'
    proj_col = 'projected_fpts' if 'projected_fpts' in lineup.columns else 'Avg'

    print(f"  {'Pos':<4} {'Player':<25} {'Team':<5} {'Sal':>6} {'Proj':>5}")
    print(f"  {'-' * 48}")

    total_sal = 0
    total_proj = 0
    for _, row in lineup.iterrows():
        pos = row.get(pos_col, '?')
        name = row.get(name_col, '?')
        team = row.get(team_col, '?')
        sal = row.get(sal_col, 0)
        proj = row.get(proj_col, 0)
        total_sal += sal if pd.notna(sal) else 0
        total_proj += proj if pd.notna(proj) else 0
        print(f"  {pos:<4} {name:<25} {team:<5} ${sal:>5,.0f} {proj:>5.1f}")

    print(f"  {'-' * 48}")
    print(f"  {'':4} {'TOTAL':<25} {'':5} ${total_sal:>5,.0f} {total_proj:>5.1f}")

    stacks = _get_stack_desc(lineup)

    print(f"\n  Simulation ({result.get('n_sims', 'N/A'):,} iterations):")
    print(f"    Mean: {result['mean']:.1f}  |  Std: {result['std']:.1f}  |  Stacks: {stacks}")
    print(f"    P(cash ≥111): {result['p_cash']:.1%}  |  P(≥120): {result['p_120']:.1%}  "
          f"|  P(≥140): {result['p_top5']:.1%}")
    print(f"    Floor (P5): {result['p5']:.0f}  |  Ceiling (P95): {result['p95']:.0f}  "
          f"|  Max sim: {result['max']:.0f}")
    print(f"    E[payout]: ${result['ev']:.0f}  |  Net EV: ${result['net_ev']:+.0f}")
