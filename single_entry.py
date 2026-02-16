#!/usr/bin/env python3
"""
Single-Entry Lineup Selector for NHL DFS (DraftKings)

Purpose: Score and select the optimal lineup for single-entry GPP contests.
Sits between the optimizer (which generates N candidate lineups) and final output.

Design principles derived from backtesting (2/3-2/4/26 slates):
  1. Goalie is the highest-leverage binary decision (~26 FPTS swing)
  2. Line-mate correlation > same-team correlation (Boldy+Zuccarello > Boldy+Ek)
  3. One-off premium D without stack correlation is salary-inefficient
  4. Salary efficiency: target $49,600-$50,000 usage
  5. Moderate leverage in SE: not max chalk, not max contrarian

Usage:
    from single_entry import SingleEntrySelector
    selector = SingleEntrySelector(player_pool, stack_builder)
    best = selector.select(candidate_lineups)

    # Or integrated into main.py pipeline:
    python main.py --single-entry --lineups 50 --stacks --edge
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


# ─── Scoring Weights (tunable) ──────────────────────────────────────────────
# These weights determine how the selector ranks candidate lineups.
# Sum doesn't need to equal 1.0 — they're relative importance.

SE_WEIGHTS = {
    'projection':       0.35,   # Raw projected FPTS (trust the model)
    'ceiling':          0.15,   # Upside potential (need ceiling to win SE GPP)
    'goalie_quality':   0.15,   # Goalie floor/matchup/confirmation strength
    'stack_correlation': 0.15,  # Line-mate correlation within stacks
    'salary_efficiency': 0.10,  # How well salary cap is utilized
    'leverage':         0.10,   # Moderate ownership differentiation
}

# ─── Goalie Scoring Parameters ───────────────────────────────────────────────
# Goalie is the single highest-leverage decision. These params control how
# aggressively we score goalie quality.

GOALIE_PARAMS = {
    'min_floor_fpts': 5.0,          # Goalies below this floor get penalized hard
    'preferred_opp_goals_max': 3.2,  # Opponent implied goals — prefer facing weak offenses
    'favorite_bonus': 0.10,          # Bonus if goalie's team is favored
    'floor_penalty_weight': 2.0,     # How much to penalize low-floor goalies
    'confirmation_required': True,   # Only consider confirmed starters
}

# ─── Stack Parameters ────────────────────────────────────────────────────────
STACK_PARAMS = {
    'min_primary_stack': 3,     # Minimum players in primary stack
    'max_primary_stack': 5,     # Maximum players in primary stack
    'ideal_primary_stack': 4,   # Sweet spot for SE
    'linemate_bonus': 0.15,     # Extra credit for actual line-mates (vs just same team)
    'pp_unit_bonus': 0.10,      # Extra credit for players on same PP unit
    'secondary_stack_bonus': 0.05,  # Small bonus for having a 2-man secondary
    'lone_wolf_penalty': 0.15,  # Penalty per uncorrelated one-off player
    'goalie_stack_conflict': 0.20,  # Penalty if goalie faces your own stack's team
}

# ─── Salary Parameters ──────────────────────────────────────────────────────
SALARY_PARAMS = {
    'cap': 50000,
    'ideal_min': 49400,         # Don't leave more than $600 on the table
    'ideal_max': 50000,
    'waste_penalty_per_100': 0.01,  # Penalty per $100 unused below ideal_min
}


class SingleEntrySelector:
    """Score and select the best single-entry lineup from candidates."""

    def __init__(
        self,
        player_pool: pd.DataFrame,
        stack_builder=None,
        team_totals: Dict[str, float] = None,
        weights: Dict[str, float] = None,
    ):
        self.pool = player_pool.copy()
        self.stack_builder = stack_builder
        self.team_totals = team_totals or {}
        self.weights = weights or SE_WEIGHTS.copy()

        # Pre-compute pool-level stats for normalization
        skaters = self.pool[self.pool['position'] != 'G']
        self._proj_mean = skaters['projected_fpts'].mean() if len(skaters) > 0 else 8.0
        self._proj_std = skaters['projected_fpts'].std() if len(skaters) > 0 else 3.0
        self._ceil_mean = skaters['ceiling'].mean() if 'ceiling' in skaters.columns and len(skaters) > 0 else 25.0
        self._ceil_std = skaters['ceiling'].std() if 'ceiling' in skaters.columns and len(skaters) > 0 else 8.0

        # Build line-mate lookup from stack_builder
        self._linemate_sets = {}  # team -> list of sets of linemate names
        self._pp_sets = {}         # team -> list of sets of PP unit names
        if stack_builder:
            self._build_linemate_lookup()

    def _build_linemate_lookup(self):
        """Extract actual line combinations from the stack builder."""
        if not self.stack_builder or not hasattr(self.stack_builder, 'lines_data'):
            return

        from lines import find_player_match

        for team, data in self.stack_builder.lines_data.items():
            self._linemate_sets[team] = []
            self._pp_sets[team] = []

            # Even-strength lines
            for key in ['line1', 'line2', 'line3', 'line4']:
                line = data.get(key, {})
                players = []
                for pos in ['lw', 'c', 'rw']:
                    p = line.get(pos)
                    if p:
                        players.append(p)
                if len(players) >= 2:
                    self._linemate_sets[team].append(set(players))

            # Defense pairs
            for key in ['pair1', 'pair2', 'pair3']:
                pair = data.get(key, {})
                players = []
                for pos in ['ld', 'rd']:
                    p = pair.get(pos)
                    if p:
                        players.append(p)
                if len(players) >= 2:
                    self._linemate_sets[team].append(set(players))

            # Power play units
            for key in ['pp1', 'pp2']:
                pp = data.get(key, {})
                players = []
                for pos in ['lw', 'c', 'rw', 'ld', 'rd']:
                    p = pp.get(pos)
                    if p:
                        players.append(p)
                if len(players) >= 2:
                    self._pp_sets[team].append(set(players))

    def _find_linemate_overlap(self, names: List[str], team: str) -> int:
        """Count how many linemate pair connections exist among names on this team."""
        connections = 0
        team_lines = self._linemate_sets.get(team, [])
        for line_set in team_lines:
            # Count players from our lineup that are in this line
            overlap = sum(1 for n in names if self._fuzzy_in_set(n, line_set))
            if overlap >= 2:
                # Each pair of overlapping players is a connection
                connections += overlap * (overlap - 1) // 2
        return connections

    def _find_pp_overlap(self, names: List[str], team: str) -> int:
        """Count PP unit connections among names on this team."""
        connections = 0
        team_pp = self._pp_sets.get(team, [])
        for pp_set in team_pp:
            overlap = sum(1 for n in names if self._fuzzy_in_set(n, pp_set))
            if overlap >= 2:
                connections += overlap * (overlap - 1) // 2
        return connections

    @staticmethod
    def _fuzzy_in_set(name: str, name_set: set) -> bool:
        """Check if a player name fuzzy-matches any name in a set."""
        from difflib import SequenceMatcher
        name_lower = name.lower()
        for s in name_set:
            s_lower = s.lower()
            if name_lower == s_lower:
                return True
            # Check last-name match
            name_last = name_lower.split()[-1] if name_lower.split() else name_lower
            s_last = s_lower.split()[-1] if s_lower.split() else s_lower
            if name_last == s_last and len(name_last) > 3:
                return True
            if SequenceMatcher(None, name_lower, s_lower).ratio() >= 0.85:
                return True
        return False

    # ─── Component Scoring Functions ─────────────────────────────────────

    def score_projection(self, lineup: pd.DataFrame) -> float:
        """Score based on total projected FPTS (z-score normalized per player)."""
        total_proj = lineup['projected_fpts'].sum()
        # Normalize: a 9-player lineup with average players scores ~72 FPTS
        # Elite lineups score 95-110+. Normalize to 0-1 range.
        expected_total = self._proj_mean * 9
        expected_std = self._proj_std * 3  # lineup-level std (roughly sqrt(9) * player_std)
        if expected_std == 0:
            return 0.5
        z = (total_proj - expected_total) / expected_std
        return min(max(z / 4 + 0.5, 0), 1)  # Clamp to [0, 1]

    def score_ceiling(self, lineup: pd.DataFrame) -> float:
        """Score based on total ceiling potential."""
        if 'ceiling' not in lineup.columns:
            return 0.5
        total_ceil = lineup['ceiling'].sum()
        expected_ceil = self._ceil_mean * 9
        expected_std = self._ceil_std * 3
        if expected_std == 0:
            return 0.5
        z = (total_ceil - expected_ceil) / expected_std
        return min(max(z / 4 + 0.5, 0), 1)

    def score_goalie_quality(self, lineup: pd.DataFrame) -> float:
        """
        Score goalie selection quality.

        This is the HIGHEST LEVERAGE single-player decision.
        Factors: floor, opponent implied goals, favorite status, projection confidence.
        """
        goalie = lineup[lineup['position'] == 'G']
        if goalie.empty:
            return 0.0

        g = goalie.iloc[0]
        score = 0.5  # Baseline

        proj = g.get('projected_fpts', 0)
        floor = g.get('floor', 0)
        salary = g.get('salary', 7500)
        team = g.get('team', '')

        # 1. Floor quality — penalize goalies with low projected floors
        min_floor = GOALIE_PARAMS['min_floor_fpts']
        if floor >= min_floor:
            score += 0.15
        else:
            penalty = (min_floor - floor) / min_floor * GOALIE_PARAMS['floor_penalty_weight']
            score -= min(penalty, 0.3)

        # 2. Projection strength (z-score among all goalies in pool)
        all_goalies = self.pool[self.pool['position'] == 'G']
        if len(all_goalies) > 1:
            g_mean = all_goalies['projected_fpts'].mean()
            g_std = all_goalies['projected_fpts'].std()
            if g_std > 0:
                g_z = (proj - g_mean) / g_std
                score += g_z * 0.1  # Top goalie gets ~+0.2, bottom gets ~-0.2

        # 3. Opponent implied goals (lower = better for goalie)
        opp_team = self._get_goalie_opponent(g, lineup)
        if opp_team and opp_team in self.team_totals:
            opp_implied = self.team_totals[opp_team]
            max_opp = GOALIE_PARAMS['preferred_opp_goals_max']
            if opp_implied <= max_opp:
                score += 0.10  # Good matchup
            else:
                score -= (opp_implied - max_opp) * 0.05  # Penalty for facing high-scoring team

        # 4. Value — don't overpay for goalie in SE
        value = proj / (salary / 1000) if salary > 0 else 0
        if value >= 1.5:
            score += 0.05
        elif value < 1.0:
            score -= 0.05

        # 5. Goalie should NOT face your primary stack's team
        skater_teams = lineup[lineup['position'] != 'G']['team'].value_counts()
        if len(skater_teams) > 0:
            primary_stack_team = skater_teams.index[0]
            if opp_team and opp_team == primary_stack_team:
                score -= STACK_PARAMS['goalie_stack_conflict']

        return min(max(score, 0), 1)

    def _get_goalie_opponent(self, goalie_row, lineup: pd.DataFrame) -> Optional[str]:
        """Determine the goalie's opponent team."""
        goalie_team = goalie_row.get('team', '')
        if not goalie_team:
            return None

        # Try to find from game_info column
        game_info = goalie_row.get('game_info', '')
        if pd.notna(game_info) and isinstance(game_info, str):
            # Parse "TOR@BOS 07:00PM ET" format
            parts = str(game_info).split()
            if parts:
                matchup = parts[0]
                teams_in_game = matchup.replace('@', '/').split('/')
                for t in teams_in_game:
                    t = t.strip().upper()
                    if t != goalie_team.upper() and len(t) >= 2:
                        return t

        # Fallback: look at the pool for teams playing this goalie's team
        # (use the team_totals keys if available)
        return None

    def score_stack_correlation(self, lineup: pd.DataFrame) -> float:
        """
        Score stack quality based on actual line-mate correlation.

        Key insight from 2/4 backtest: Boldy+Zuccarello (same line) correlated.
        Eriksson Ek (different line) didn't participate in same scoring plays.
        Line-mate correlation > mere same-team correlation.
        """
        skaters = lineup[lineup['position'] != 'G']
        if skaters.empty:
            return 0.5

        score = 0.5  # Baseline

        # Count team stacking
        team_counts = skaters['team'].value_counts()
        primary_team = team_counts.index[0] if len(team_counts) > 0 else None
        primary_count = team_counts.iloc[0] if len(team_counts) > 0 else 0

        # 1. Primary stack size scoring
        ideal = STACK_PARAMS['ideal_primary_stack']
        if primary_count >= STACK_PARAMS['min_primary_stack']:
            # Reward being near ideal size
            dist_from_ideal = abs(primary_count - ideal)
            score += 0.15 - (dist_from_ideal * 0.03)
        else:
            score -= 0.10  # Penalty for no real stack

        # 2. LINE-MATE correlation bonus (the key insight)
        if primary_team and self._linemate_sets:
            primary_names = skaters[skaters['team'] == primary_team]['name'].tolist()
            linemate_connections = self._find_linemate_overlap(primary_names, primary_team)
            pp_connections = self._find_pp_overlap(primary_names, primary_team)

            # Each linemate connection is worth a bonus
            score += linemate_connections * STACK_PARAMS['linemate_bonus']
            score += pp_connections * STACK_PARAMS['pp_unit_bonus']

        # 3. Secondary stack bonus
        if len(team_counts) >= 2:
            second_count = team_counts.iloc[1]
            if second_count >= 2:
                score += STACK_PARAMS['secondary_stack_bonus']
                # Check linemate connections in secondary stack too
                second_team = team_counts.index[1]
                if second_team and self._linemate_sets:
                    second_names = skaters[skaters['team'] == second_team]['name'].tolist()
                    second_connections = self._find_linemate_overlap(second_names, second_team)
                    score += second_connections * STACK_PARAMS['linemate_bonus'] * 0.5

        # 4. Lone wolf penalty — one-off players with no team correlation
        one_off_teams = [t for t, c in team_counts.items() if c == 1]
        # One one-off is fine (UTIL flex). Two+ is inefficient for SE.
        if len(one_off_teams) > 1:
            score -= (len(one_off_teams) - 1) * STACK_PARAMS['lone_wolf_penalty']

        # 5. Premium one-off D penalty (the Quinn Hughes lesson)
        for _, player in skaters.iterrows():
            t = player['team']
            if team_counts.get(t, 0) == 1 and player['position'] == 'D':
                if player['salary'] >= 6000:
                    score -= 0.08  # Expensive D with no stack = bad SE construction

        return min(max(score, 0), 1)

    def score_salary_efficiency(self, lineup: pd.DataFrame) -> float:
        """Score how efficiently the salary cap is used."""
        total_sal = lineup['salary'].sum()
        cap = SALARY_PARAMS['cap']

        if total_sal > cap:
            return 0.0  # Invalid lineup

        remaining = cap - total_sal
        ideal_min = SALARY_PARAMS['ideal_min']

        if total_sal >= ideal_min:
            # Sweet spot — very little waste
            return 0.9 + (total_sal - ideal_min) / (cap - ideal_min) * 0.1
        else:
            # Penalty for wasted salary
            waste = ideal_min - total_sal
            penalty = (waste / 100) * SALARY_PARAMS['waste_penalty_per_100']
            return max(0.9 - penalty, 0.3)

    def score_leverage(self, lineup: pd.DataFrame) -> float:
        """
        Score ownership leverage — moderate differentiation for SE.

        In SE, we don't want max chalk (no edge) or max contrarian (too risky).
        We want the sweet spot: a few low-owned high-ceiling plays mixed with
        solid moderate-owned pieces.
        """
        if 'predicted_ownership' not in lineup.columns:
            return 0.5

        owns = lineup['predicted_ownership'].fillna(5.0)
        avg_own = owns.mean()

        # Ideal average ownership for SE: ~5-10%
        # Below 3% = too contrarian, above 15% = too chalky
        if 5 <= avg_own <= 10:
            score = 0.8
        elif 3 <= avg_own < 5 or 10 < avg_own <= 15:
            score = 0.6
        elif avg_own < 3:
            score = 0.4  # Too contrarian for SE
        else:
            score = 0.4  # Too chalky

        # Bonus for having 1-2 contrarian plays (< 3% own) with high ceiling
        contrarian_ceiling = lineup[
            (lineup['predicted_ownership'] < 3) &
            (lineup.get('ceiling', pd.Series(dtype=float)) > self._ceil_mean * 1.2 if 'ceiling' in lineup.columns else False)
        ]
        if 1 <= len(contrarian_ceiling) <= 3:
            score += 0.10

        # Penalty for ALL players being high-owned (no differentiation)
        high_own_count = (owns > 15).sum()
        if high_own_count >= 4:
            score -= 0.15

        return min(max(score, 0), 1)

    # ─── Main Scoring & Selection ────────────────────────────────────────

    def score_lineup(self, lineup: pd.DataFrame) -> Dict[str, float]:
        """Score a single lineup across all dimensions. Returns component scores + total."""
        components = {
            'projection':        self.score_projection(lineup),
            'ceiling':           self.score_ceiling(lineup),
            'goalie_quality':    self.score_goalie_quality(lineup),
            'stack_correlation': self.score_stack_correlation(lineup),
            'salary_efficiency': self.score_salary_efficiency(lineup),
            'leverage':          self.score_leverage(lineup),
        }

        # Weighted total
        total = sum(self.weights.get(k, 0) * v for k, v in components.items())
        weight_sum = sum(self.weights.get(k, 0) for k in components)
        components['total'] = total / weight_sum if weight_sum > 0 else 0

        return components

    def select(
        self,
        candidate_lineups: List[pd.DataFrame],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Score all candidate lineups and return the best one for single entry.

        Args:
            candidate_lineups: List of lineup DataFrames from optimizer
            verbose: Print scoring breakdown

        Returns:
            (best_lineup, scores_dict)
        """
        if not candidate_lineups:
            raise ValueError("No candidate lineups provided")

        if len(candidate_lineups) == 1:
            scores = self.score_lineup(candidate_lineups[0])
            if verbose:
                self._print_scores([(candidate_lineups[0], scores)], selected_idx=0)
            return candidate_lineups[0], scores

        # Score all candidates
        scored = []
        for lineup in candidate_lineups:
            scores = self.score_lineup(lineup)
            scored.append((lineup, scores))

        # Sort by total score descending
        scored.sort(key=lambda x: x[1]['total'], reverse=True)

        if verbose:
            self._print_scores(scored, selected_idx=0)

        return scored[0][0], scored[0][1]

    def _print_scores(self, scored: List[Tuple[pd.DataFrame, dict]], selected_idx: int = 0):
        """Print scoring breakdown for all candidates."""
        print(f"\n{'=' * 100}")
        print(" SINGLE-ENTRY LINEUP SELECTOR — CANDIDATE SCORING")
        print(f"{'=' * 100}")

        # Header
        components = ['projection', 'ceiling', 'goalie_quality', 'stack_correlation',
                       'salary_efficiency', 'leverage', 'total']
        header = f"{'#':>3} {'Proj':>6} {'Sal':>7} {'Goalie':<18} {'Stack':<12}"
        for c in components:
            short = c[:6].title()
            header += f" {short:>7}"
        print(f"\n{header}")
        print("-" * len(header))

        # Show top 10 (or all if fewer)
        n_show = min(len(scored), 10)
        for i in range(n_show):
            lineup, scores = scored[i]
            total_proj = lineup['projected_fpts'].sum()
            total_sal = lineup['salary'].sum()
            goalie = lineup[lineup['position'] == 'G']
            g_name = goalie.iloc[0]['name'][:16] if not goalie.empty else "???"

            # Stack summary
            skaters = lineup[lineup['position'] != 'G']
            team_counts = skaters['team'].value_counts()
            stack_str = '+'.join(f"{t}{c}" for t, c in team_counts.items() if c >= 2)
            if not stack_str:
                stack_str = "none"

            marker = " <<<" if i == selected_idx else ""
            row = f"{i+1:>3} {total_proj:>6.1f} ${total_sal:>6,} {g_name:<18} {stack_str:<12}"
            for c in components:
                val = scores.get(c, 0)
                row += f" {val:>7.3f}"
            row += marker
            print(row)

        if selected_idx == 0:
            print(f"\n  >>> LINEUP #1 SELECTED for single entry <<<")

        # Print detailed breakdown for selected lineup
        best_lineup, best_scores = scored[selected_idx]
        print(f"\n  Scoring Breakdown:")
        for comp in components[:-1]:
            weight = self.weights.get(comp, 0)
            raw = best_scores[comp]
            weighted = weight * raw
            bar = '█' * int(raw * 20) + '░' * (20 - int(raw * 20))
            print(f"    {comp:<22} {bar} {raw:.3f} × {weight:.2f} = {weighted:.3f}")
        print(f"    {'TOTAL':<22} {'':>20} {best_scores['total']:.3f}")


def print_se_lineup(lineup: pd.DataFrame, scores: Dict[str, float]):
    """Print the selected single-entry lineup with full context."""
    print(f"\n{'=' * 80}")
    print(" SINGLE-ENTRY LINEUP (FINAL)")
    print(f"{'=' * 80}")

    total_salary = lineup['salary'].sum()
    total_proj = lineup['projected_fpts'].sum()
    total_ceil = lineup['ceiling'].sum() if 'ceiling' in lineup.columns else 0

    print(f"  Salary: ${total_salary:,} / $50,000 (${50000 - total_salary:,} remaining)")
    print(f"  Projected: {total_proj:.1f} FPTS")
    if total_ceil > 0:
        print(f"  Ceiling: {total_ceil:.1f} FPTS")
    print(f"  SE Score: {scores['total']:.3f}")

    # Stack info
    skaters = lineup[lineup['position'] != 'G']
    team_counts = skaters['team'].value_counts()
    stacks = [f"{team} ×{count}" for team, count in team_counts.items() if count >= 2]
    one_offs = [f"{team}" for team, count in team_counts.items() if count == 1]
    if stacks:
        print(f"  Stacks: {', '.join(stacks)}")
    if one_offs:
        print(f"  One-offs: {', '.join(one_offs)}")

    print()
    slot_order = {'G': 0, 'C1': 1, 'C2': 2, 'W1': 3, 'W2': 4, 'W3': 5, 'D1': 6, 'D2': 7, 'UTIL': 8}
    display = lineup.copy()
    if 'roster_slot' in display.columns:
        display['_order'] = display['roster_slot'].map(lambda x: slot_order.get(x, 9))
        display = display.sort_values('_order')
        slot_col = 'roster_slot'
    else:
        pos_order = {'C': 0, 'W': 2, 'D': 3, 'G': 4}
        display['_order'] = display['position'].map(lambda x: pos_order.get(x, 5))
        display = display.sort_values('_order')
        slot_col = 'position'

    own_col = 'predicted_ownership' if 'predicted_ownership' in display.columns else None

    header = f"  {'Slot':<6} {'Name':<28} {'Team':<5} {'Salary':<9} {'Proj':>6} {'Ceil':>6} {'Value':>6}"
    if own_col:
        header += f" {'Own%':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for _, row in display.iterrows():
        ceil_val = row.get('ceiling', 0)
        val = row.get('value', 0)
        line = (f"  {row[slot_col]:<6} {row['name']:<28} {row['team']:<5} "
                f"${row['salary']:<8,} {row['projected_fpts']:>6.1f} {ceil_val:>6.1f} {val:>6.2f}")
        if own_col:
            own = row.get(own_col, 0)
            line += f" {own:>5.1f}%"
        print(line)

    print()
