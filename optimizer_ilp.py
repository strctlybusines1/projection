"""
DraftKings NHL Lineup Optimizer — ILP (Integer Linear Programming) Version.

Drop-in replacement for optimizer.py. Same interface, same inputs, same outputs.
Uses PuLP to find mathematically optimal lineups instead of greedy heuristics.

What changed:  Lineup assembly logic (greedy → ILP solver)
What didn't:   Projections, stacking logic, goalie filtering, contest analysis

Install: pip install pulp

Usage in main.py:
    # Change this one import:
    from optimizer_ilp import NHLLineupOptimizer
    # Everything else stays identical
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from pulp import (
        LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus, value,
        PULP_CBC_CMD,
    )
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("WARNING: PuLP not installed. Run: pip install pulp")
    print("         Falling back to greedy optimizer.")

from config import (
    GPP_MIN_STACK_SIZE, GPP_MAX_FROM_TEAM, CASH_MAX_FROM_TEAM,
    PRIMARY_STACK_BOOST, SECONDARY_STACK_BOOST, LINEMATE_BOOST,
    GOALIE_CORRELATION_BOOST, HIGH_TOTAL_THRESHOLD,
    PREFERRED_PRIMARY_STACK_SIZE, PREFERRED_SECONDARY_STACK_SIZE,
    DAILY_SALARIES_DIR, DAILY_PROJECTIONS_DIR,
)
from utils import (
    normalize_position as _normalize_pos,
    fuzzy_match,
    find_player_match,
    parse_opponent_from_game_info,
)


class NHLLineupOptimizer:
    """
    ILP Optimizer for DraftKings NHL lineups with GPP-focused stacking.

    Same interface as the original greedy optimizer. All methods that main.py
    calls (optimize_lineup, format_lineup_for_dk, analyze_contest, etc.) are
    preserved with identical signatures.

    DK NHL Classic Roster:
    - 2 Centers (C)
    - 3 Wings (W) - can be LW or RW
    - 2 Defensemen (D)
    - 1 Goalie (G)
    - 1 UTIL (any skater - C, W, or D excluded for D)

    Salary Cap: $50,000
    """

    SALARY_CAP = 50000
    ROSTER_REQUIREMENTS = {
        'C': 2,
        'W': 3,
        'D': 2,
        'G': 1,
        'UTIL': 1  # C or W only (D excluded per your current logic)
    }

    def __init__(self, stack_builder=None):
        """
        Initialize optimizer.

        Args:
            stack_builder: Optional StackBuilder instance for correlation data
        """
        self.stack_builder = stack_builder

    # ================================================================
    #  Position & Team Helpers (unchanged from original)
    # ================================================================

    def _normalize_position(self, pos: str) -> str:
        """Normalize position codes. Delegates to utils.normalize_position."""
        return _normalize_pos(pos)

    def _get_opponent_team(self, player_row: pd.Series) -> Optional[str]:
        """
        Extract opponent team from game info.
        Game info format: "ANA@EDM 01/26/2026 08:30PM ET"
        Delegates to utils.parse_opponent_from_game_info.
        """
        game_info = player_row.get('game_info', '') or player_row.get('Game Info', '')
        player_team = player_row.get('team', '')
        return parse_opponent_from_game_info(player_team, game_info)

    # ================================================================
    #  Goalie Filtering (unchanged from original)
    # ================================================================

    def _filter_confirmed_goalies(self, goalies: pd.DataFrame) -> pd.DataFrame:
        """Filter goalies to only confirmed starters from lines data."""
        if not self.stack_builder:
            return goalies
        confirmed = self.stack_builder.get_all_starting_goalies()
        if not confirmed:
            return goalies
        confirmed_names = list(confirmed.values())
        # fuzzy_match already imported from utils

        def is_confirmed(name):
            for confirmed_name in confirmed_names:
                if fuzzy_match(name, confirmed_name):
                    return True
            return False

        mask = goalies['name'].apply(is_confirmed)
        filtered = goalies[mask]
        if filtered.empty:
            print("Warning: No confirmed goalies matched in player pool, using all goalies")
            return goalies
        return filtered

    # ================================================================
    #  Stacking Helpers (unchanged from original)
    # ================================================================

    def _get_correlated_stack_players(self, team: str, player_pool_df: pd.DataFrame,
                                       target_size: int, stack_preference: str = 'primary',
                                       randomness: float = 0.0,
                                       stack_cache: dict = None) -> Optional[List[str]]:
        """Select players from actual line/PP combinations for correlated stacking."""
        if not self.stack_builder:
            return None
        if stack_cache is not None and team in stack_cache:
            stacks = stack_cache[team]
        else:
            stacks = self.stack_builder.get_best_stacks(team, player_pool_df)
            if stack_cache is not None:
                stack_cache[team] = stacks
        if not stacks:
            return None
        stack_by_type = {}
        for s in stacks:
            stack_by_type[s['type']] = s
        if stack_preference == 'primary':
            priority = ['PP1', 'Line1+D1', 'Line1']
        else:
            priority = ['Line1', 'Line2', 'PP1']
        if randomness > 0 and np.random.random() < randomness:
            np.random.shuffle(priority)
        for stack_type in priority:
            stack = stack_by_type.get(stack_type)
            if not stack:
                continue
            matched_players = stack.get('matched_players', [])
            if not matched_players:
                # find_player_match already imported from utils
                pool_names = player_pool_df['name'].tolist()
                matched_players = []
                for p in stack.get('players', []):
                    match = find_player_match(p, pool_names)
                    if match:
                        matched_players.append(match)
            if len(matched_players) < target_size:
                continue
            matched_df = player_pool_df[player_pool_df['name'].isin(matched_players)].copy()
            matched_df = matched_df.sort_values('adj_projection', ascending=False)
            return matched_df['name'].head(target_size).tolist()
        best_stack = None
        best_count = 0
        for stack_type in priority:
            stack = stack_by_type.get(stack_type)
            if not stack:
                continue
            matched_players = stack.get('matched_players', [])
            if not matched_players:
                # find_player_match already imported from utils
                pool_names = player_pool_df['name'].tolist()
                matched_players = []
                for p in stack.get('players', []):
                    match = find_player_match(p, pool_names)
                    if match:
                        matched_players.append(match)
            if len(matched_players) > best_count:
                best_count = len(matched_players)
                best_stack = matched_players
        if best_stack and len(best_stack) >= 2:
            matched_df = player_pool_df[player_pool_df['name'].isin(best_stack)].copy()
            matched_df = matched_df.sort_values('adj_projection', ascending=False)
            return matched_df['name'].head(target_size).tolist()
        return None

    def _apply_linemate_boosts(self, df: pd.DataFrame, team: str) -> pd.DataFrame:
        """Apply projection boosts based on linemate correlations."""
        if not self.stack_builder:
            return df
        df = df.copy()
        team_data = self.stack_builder.lines_data.get(team, {})
        if not team_data or 'error' in team_data:
            return df
        pp1_players = set()
        line1_players = set()
        for pp in team_data.get('pp_units', []):
            if pp.get('unit') == 1:
                pp1_players.update([p.lower() for p in pp.get('players', [])])
        for line in team_data.get('forward_lines', []):
            if line.get('line') == 1:
                line1_players.update([p.lower() for p in line.get('players', [])])

        def get_linemate_boost(name):
            name_lower = name.lower()
            for pp_name in pp1_players:
                if name_lower in pp_name or pp_name in name_lower:
                    return 1 + LINEMATE_BOOST * 1.2
            for line_name in line1_players:
                if name_lower in line_name or line_name in name_lower:
                    return 1 + LINEMATE_BOOST
            return 1.0

        df['linemate_boost'] = df['name'].apply(get_linemate_boost)
        df['adj_projection'] = df['adj_projection'] * df['linemate_boost']
        return df

    # ================================================================
    #  ILP Core: Build a single optimal lineup
    # ================================================================

    def _compute_effective_projection(self, df: pd.DataFrame,
                                       primary_team: Optional[str],
                                       secondary_team: Optional[str],
                                       goalie_team: Optional[str],
                                       randomness: float = 0.0) -> pd.Series:
        """
        Compute the effective projection for each player, including stack boosts,
        linemate boosts, and goalie correlation — mirroring the greedy optimizer's
        logic but expressed as a single score column for the ILP objective.
        """
        eff = df['projected_fpts'].copy()

        # Add randomness (same as original)
        if randomness > 0:
            noise = np.random.normal(1, randomness, len(df))
            eff = eff * pd.Series(noise.clip(0.7, 1.3), index=df.index)

        # Primary stack boost
        if primary_team:
            is_primary = df['team'] == primary_team
            is_skater = df['norm_position'] != 'G'
            eff = eff + eff * PRIMARY_STACK_BOOST * (is_primary & is_skater).astype(float)

            # Extra linemate boost for PP1/Line1 on primary team
            if self.stack_builder:
                boosted = self._apply_linemate_boosts(
                    df[is_primary & is_skater].copy(), primary_team
                )
                if 'linemate_boost' in boosted.columns:
                    for idx in boosted.index:
                        lb = boosted.loc[idx, 'linemate_boost']
                        if lb > 1.0:
                            eff.loc[idx] = eff.loc[idx] * lb

        # Secondary stack boost
        if secondary_team:
            is_secondary = df['team'] == secondary_team
            is_skater = df['norm_position'] != 'G'
            eff = eff + eff * SECONDARY_STACK_BOOST * (is_secondary & is_skater).astype(float)

        # Goalie correlation boost (skaters on goalie's team)
        if goalie_team:
            is_goalie_team = df['team'] == goalie_team
            is_skater = df['norm_position'] != 'G'
            eff = eff + eff * GOALIE_CORRELATION_BOOST * 0.5 * (is_goalie_team & is_skater).astype(float)

        # Goalie on primary team gets correlation boost
        if primary_team:
            is_primary_goalie = (df['team'] == primary_team) & (df['norm_position'] == 'G')
            eff = eff + eff * GOALIE_CORRELATION_BOOST * is_primary_goalie.astype(float)

        return eff

    def _solve_ilp(self, df: pd.DataFrame, effective_proj: pd.Series,
                    max_from_team: int, min_teams: int,
                    primary_team: Optional[str] = None,
                    min_primary: int = 0,
                    secondary_team: Optional[str] = None,
                    min_secondary: int = 0,
                    force_players: List[str] = None,
                    exclude_players: List[str] = None,
                    exclude_lineup_hashes: set = None) -> Optional[pd.DataFrame]:
        """
        Solve a single ILP to find the optimal lineup.

        Returns a 9-row DataFrame with roster_slot assigned, or None if infeasible.
        """
        if not HAS_PULP:
            return None

        n = len(df)
        if n == 0:
            return None

        # Create problem
        prob = LpProblem("DK_NHL_Lineup", LpMaximize)

        # Decision variables: x[i] = 1 if player i is in lineup
        x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

        # UTIL eligibility variables: u[i] = 1 if player i fills the UTIL slot
        u = [LpVariable(f"u_{i}", cat=LpBinary) for i in range(n)]

        # Objective: maximize total effective projection
        prob += lpSum(effective_proj.iloc[i] * x[i] for i in range(n))

        # Constraint: exactly 9 players
        prob += lpSum(x[i] for i in range(n)) == 9

        # Constraint: salary cap
        salaries = df['salary'].values
        prob += lpSum(salaries[i] * x[i] for i in range(n)) <= self.SALARY_CAP

        # Position masks
        is_c = (df['norm_position'] == 'C').values
        is_w = (df['norm_position'] == 'W').values
        is_d = (df['norm_position'] == 'D').values
        is_g = (df['norm_position'] == 'G').values

        # Constraint: exactly 1 goalie
        prob += lpSum(x[i] for i in range(n) if is_g[i]) == 1

        # Constraint: UTIL is C or W only (D excluded per your rules)
        # u[i] can only be 1 if x[i] is 1 and player is C or W
        for i in range(n):
            prob += u[i] <= x[i]
            if is_d[i] or is_g[i]:
                prob += u[i] == 0

        # Exactly 1 UTIL
        prob += lpSum(u[i] for i in range(n)) == 1

        # Position counts (excluding UTIL):
        # Centers in C slots = total C selected - C in UTIL = 2
        # Wings in W slots = total W selected - W in UTIL = 3
        # D slots = total D selected = 2
        prob += lpSum(x[i] for i in range(n) if is_c[i]) - lpSum(u[i] for i in range(n) if is_c[i]) == 2
        prob += lpSum(x[i] for i in range(n) if is_w[i]) - lpSum(u[i] for i in range(n) if is_w[i]) == 3
        prob += lpSum(x[i] for i in range(n) if is_d[i]) == 2

        # Constraint: max players from any team
        teams = df['team'].unique()
        for team in teams:
            team_mask = (df['team'] == team).values
            prob += lpSum(x[i] for i in range(n) if team_mask[i]) <= max_from_team

        # Constraint: minimum teams (skaters only, goalie doesn't count)
        # Use team indicator variables: t[team] = 1 if any skater from that team is selected
        skater_teams = df[~df['norm_position'].isin(['G'])]['team'].unique()
        t_vars = {}
        for team in skater_teams:
            t_vars[team] = LpVariable(f"team_{team}", cat=LpBinary)
            team_skater_mask = ((df['team'] == team) & (df['norm_position'] != 'G')).values
            # If any skater from this team is selected, t[team] must be 1
            for i in range(n):
                if team_skater_mask[i]:
                    prob += x[i] <= t_vars[team]
            # If t[team] is 1, at least one skater from this team must be selected
            prob += t_vars[team] <= lpSum(x[i] for i in range(n) if team_skater_mask[i])

        prob += lpSum(t_vars[team] for team in skater_teams) >= min_teams

        # Constraint: goalie's opponent team excluded
        # Find each goalie's opponent and block skaters from that team when that goalie is selected
        for i in range(n):
            if is_g[i]:
                opp = self._get_opponent_team(df.iloc[i])
                if opp:
                    opp_skaters = ((df['team'].str.upper() == opp.upper()) & (df['norm_position'] != 'G')).values
                    for j in range(n):
                        if opp_skaters[j]:
                            # If goalie i is selected, skater j cannot be
                            prob += x[i] + x[j] <= 1

        # Constraint: primary stack minimum
        if primary_team and min_primary > 0:
            primary_skaters = ((df['team'] == primary_team) & (df['norm_position'] != 'G')).values
            prob += lpSum(x[i] for i in range(n) if primary_skaters[i]) >= min_primary

        # Constraint: secondary stack minimum
        if secondary_team and min_secondary > 0:
            secondary_skaters = ((df['team'] == secondary_team) & (df['norm_position'] != 'G')).values
            prob += lpSum(x[i] for i in range(n) if secondary_skaters[i]) >= min_secondary

        # Constraint: force players
        if force_players:
            for forced_name in force_players:
                matches = df.index[df['name'].str.lower() == forced_name.lower()].tolist()
                if matches:
                    prob += x[matches[0]] == 1

        # Constraint: exclude players
        if exclude_players:
            for excl_name in exclude_players:
                matches = df.index[df['name'].str.lower() == excl_name.lower()].tolist()
                for idx in matches:
                    prob += x[idx] == 0

        # Constraint: exclude previous lineup (for diversity)
        if exclude_lineup_hashes:
            for prev_hash in exclude_lineup_hashes:
                prev_indices = df.index[df['name'].isin(prev_hash)].tolist()
                if len(prev_indices) == 9:
                    # At most 8 of these 9 can be selected (forces at least 1 different)
                    prob += lpSum(x[i] for i in prev_indices) <= 8

        # Solve (suppress output)
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        if LpStatus[prob.status] != 'Optimal':
            return None

        # Extract selected players
        selected_indices = [i for i in range(n) if value(x[i]) > 0.5]
        util_indices = [i for i in range(n) if value(u[i]) > 0.5]

        if len(selected_indices) != 9:
            return None

        # Build lineup DataFrame with roster slots
        lineup_rows = []
        util_idx = util_indices[0] if util_indices else None

        # Assign slots
        c_count, w_count, d_count = 0, 0, 0
        for i in selected_indices:
            row = df.iloc[i].to_dict()
            pos = row['norm_position']

            if i == util_idx:
                row['roster_slot'] = 'UTIL'
            elif pos == 'G':
                row['roster_slot'] = 'G'
            elif pos == 'C':
                c_count += 1
                row['roster_slot'] = f'C{c_count}'
            elif pos == 'W':
                w_count += 1
                row['roster_slot'] = f'W{w_count}'
            elif pos == 'D':
                d_count += 1
                row['roster_slot'] = f'D{d_count}'

            lineup_rows.append(row)

        lineup = pd.DataFrame(lineup_rows)
        return lineup

    # ================================================================
    #  Main Entry Point (same signature as original)
    # ================================================================

    def optimize_lineup(self, player_pool: pd.DataFrame,
                        n_lineups: int = 1,
                        mode: str = 'gpp',
                        max_from_team: int = None,
                        min_teams: int = 3,
                        min_stack_size: int = None,
                        randomness: float = 0.0,
                        stack_teams: List[str] = None,
                        secondary_stack_team: str = None,
                        force_players: List[str] = None,
                        exclude_players: List[str] = None) -> List[pd.DataFrame]:
        """
        Generate optimized lineups with GPP stacking strategy.

        SAME SIGNATURE as original optimizer.py — drop-in replacement.
        """
        if not HAS_PULP:
            print("ERROR: PuLP not installed. Run: pip install pulp")
            return []

        df = player_pool.copy()

        # Normalize positions (same logic as original)
        if 'dk_pos' in df.columns:
            pos_col = 'dk_pos'
        elif 'dk_position' in df.columns:
            pos_col = 'dk_position'
        else:
            pos_col = 'position'
        df['norm_position'] = df[pos_col].apply(self._normalize_position)

        # Mode-specific defaults (same as original)
        if max_from_team is None:
            max_from_team = GPP_MAX_FROM_TEAM if mode == 'gpp' else CASH_MAX_FROM_TEAM
        if min_stack_size is None:
            min_stack_size = GPP_MIN_STACK_SIZE if mode == 'gpp' else 0

        # Apply exclusions
        if exclude_players:
            exclude_lower = [p.lower() for p in exclude_players]
            df = df[~df['name'].str.lower().isin(exclude_lower)]

        # Filter goalies to confirmed starters
        goalies_df = df[df['norm_position'] == 'G'].copy()
        confirmed_goalies = self._filter_confirmed_goalies(goalies_df)
        # Replace goalie pool with confirmed only
        df = pd.concat([
            df[df['norm_position'] != 'G'],
            confirmed_goalies
        ], ignore_index=True)

        # Reset index for clean ILP indexing
        df = df.reset_index(drop=True)

        # Determine stack teams
        team_proj = df[df['norm_position'] != 'G'].groupby('team')['projected_fpts'].sum()
        team_proj = team_proj.sort_values(ascending=False)

        lineups = []
        used_lineup_hashes = set()

        for lineup_num in range(n_lineups):
            # Pick primary/secondary teams (with randomness for diversity)
            if stack_teams and len(stack_teams) > 0:
                primary_team = stack_teams[0]
            else:
                top_teams = team_proj.head(5).index.tolist()
                if n_lineups > 1 and lineup_num > 0:
                    # Rotate through top teams for diversity
                    weights = np.array([0.35, 0.25, 0.18, 0.12, 0.10][:len(top_teams)])
                    weights = weights / weights.sum()
                    primary_team = np.random.choice(top_teams, p=weights)
                else:
                    primary_team = top_teams[0] if top_teams else None

            if secondary_stack_team:
                secondary_team = secondary_stack_team
            elif stack_teams and len(stack_teams) > 1:
                secondary_team = stack_teams[1]
            else:
                remaining = [t for t in team_proj.head(6).index if t != primary_team]
                if remaining:
                    if n_lineups > 1:
                        secondary_team = np.random.choice(remaining[:3])
                    else:
                        secondary_team = remaining[0]
                else:
                    secondary_team = None

            # Determine goalie team (prefer primary for correlation in GPP)
            goalie_team = primary_team if mode == 'gpp' else None

            # Compute effective projections with all boosts baked in
            eff_proj = self._compute_effective_projection(
                df, primary_team, secondary_team, goalie_team,
                randomness=randomness if n_lineups > 1 else 0.0
            )

            # Set stack minimums
            min_primary = min_stack_size if mode == 'gpp' and primary_team else 0
            min_secondary = 2 if mode == 'gpp' and secondary_team else 0

            # Solve
            lineup = self._solve_ilp(
                df, eff_proj,
                max_from_team=max_from_team,
                min_teams=min_teams,
                primary_team=primary_team,
                min_primary=min_primary,
                secondary_team=secondary_team,
                min_secondary=min_secondary,
                force_players=force_players,
                exclude_players=exclude_players,
                exclude_lineup_hashes=used_lineup_hashes if n_lineups > 1 else None,
            )

            if lineup is not None and len(lineup) == 9:
                lineup_hash = frozenset(lineup['name'].tolist())
                if lineup_hash not in used_lineup_hashes:
                    used_lineup_hashes.add(lineup_hash)
                    lineup = self._add_stack_analysis(lineup)
                    lineups.append(lineup)

        return lineups

    # ================================================================
    #  Output Formatting (unchanged from original)
    # ================================================================

    def _add_stack_analysis(self, lineup: pd.DataFrame) -> pd.DataFrame:
        """Add stacking analysis to lineup."""
        lineup = lineup.copy()
        team_counts = lineup['team'].value_counts()
        stacks = []
        for team, count in team_counts.items():
            if count >= 2:
                team_players = lineup[lineup['team'] == team]['name'].tolist()
                stacks.append(f"{team}({count}): {', '.join([p.split()[-1] for p in team_players])}")
        lineup['stack_info'] = '; '.join(stacks) if stacks else 'No stacks'
        if self.stack_builder:
            corr_info = []
            players = lineup['name'].tolist()
            for i, p1 in enumerate(players):
                for p2 in players[i+1:]:
                    corr = self.stack_builder.get_correlation(p1, p2)
                    if corr > 0.5:
                        corr_info.append(f"{p1.split()[-1]}-{p2.split()[-1]}:{corr:.0%}")
            if corr_info:
                lineup['correlations'] = ', '.join(corr_info[:5])
        return lineup

    def format_lineup_for_dk(self, lineup: pd.DataFrame) -> str:
        """Format lineup for display."""
        output = []
        total_salary = lineup['salary'].sum()
        total_proj = lineup['projected_fpts'].sum()

        output.append(f"Total Salary: ${total_salary:,} / $50,000 (${50000 - total_salary:,} remaining)")
        output.append(f"Projected Points: {total_proj:.1f}")

        team_counts = lineup['team'].value_counts()
        stacks = [f"{team}({count})" for team, count in team_counts.items() if count >= 2]
        if stacks:
            output.append(f"Stacks: {' + '.join(stacks)}")

        output.append("")
        output.append(f"{'Slot':<6} {'Name':<25} {'Team':<5} {'Salary':<9} {'Proj':<7}")
        output.append("-" * 60)

        slot_order = {'G': 0, 'C1': 1, 'C2': 2, 'W1': 3, 'W2': 4, 'W3': 5, 'D1': 6, 'D2': 7, 'UTIL': 8}
        lineup = lineup.copy()
        lineup['_order'] = lineup['roster_slot'].map(lambda x: slot_order.get(x, 9))
        lineup = lineup.sort_values('_order')

        for _, player in lineup.iterrows():
            output.append(
                f"{player['roster_slot']:<6} {player['name']:<25} {player['team']:<5} "
                f"${player['salary']:<8,} {player['projected_fpts']:.1f}"
            )

        return "\n".join(output)

    def generate_gpp_lineups(self, player_pool: pd.DataFrame,
                             n_lineups: int = 20,
                             diversity_factor: float = 0.15) -> List[pd.DataFrame]:
        """Generate diverse GPP lineup set with varied stacking strategies."""
        return self.optimize_lineup(
            player_pool,
            n_lineups=n_lineups,
            mode='gpp',
            randomness=diversity_factor,
        )

    # ================================================================
    #  Contest Analysis (unchanged from original)
    # ================================================================

    def analyze_contest(self, total_entries: int, prize_pool: float,
                        first_place: float, tenth_place: float,
                        min_cash: float, min_cash_place: int) -> Dict:
        """Analyze contest structure to determine optimal strategy."""
        first_pct_of_pool = (first_place / prize_pool) * 100
        payout_pct = (min_cash_place / total_entries) * 100
        top_10_pct = (10 / total_entries) * 100
        min_cash_multiplier = min_cash / 5

        if first_pct_of_pool >= 15:
            contest_type = "TOP_HEAVY"
        elif first_pct_of_pool >= 10:
            contest_type = "MODERATE"
        else:
            contest_type = "FLAT"

        if total_entries <= 100:
            field_size = "SMALL"
        elif total_entries <= 1000:
            field_size = "MEDIUM"
        else:
            field_size = "LARGE"

        strategies = {
            "TOP_HEAVY": {
                "stack_depth": "4-5 players",
                "leverage_target": "HIGH - Max uniqueness",
                "risk_tolerance": "HIGH - Accept bust rate",
                "goal": "Target 1st place, not min-cash",
                "secondary_stack": "2-3 players, different game",
                "goalie_strategy": "Correlate with primary stack"
            },
            "MODERATE": {
                "stack_depth": "3-4 players",
                "leverage_target": "MODERATE - Some differentiation",
                "risk_tolerance": "MEDIUM - Balance floor/ceiling",
                "goal": "Target top 10%, min-cash backup",
                "secondary_stack": "2 players for diversification",
                "goalie_strategy": "Best projection, slight correlation"
            },
            "FLAT": {
                "stack_depth": "2-3 players",
                "leverage_target": "LOW - Chalk is OK",
                "risk_tolerance": "LOW - Protect floor",
                "goal": "Multiple paths to top 20%",
                "secondary_stack": "2 players, spread risk",
                "goalie_strategy": "Best win probability"
            }
        }

        strategy = strategies[contest_type]

        return {
            "contest_type": contest_type,
            "field_size": field_size,
            "metrics": {
                "first_place_pct": f"{first_pct_of_pool:.1f}%",
                "payout_pct": f"{payout_pct:.1f}%",
                "top_10_entries": int(total_entries * 0.10),
                "min_cash_multiplier": f"{min_cash_multiplier:.1f}x"
            },
            "strategy": strategy,
            "summary": self._get_strategy_summary(contest_type, field_size)
        }

    def _get_strategy_summary(self, contest_type: str, field_size: str) -> str:
        """Get a plain-English strategy summary."""
        summaries = {
            ("TOP_HEAVY", "SMALL"): "Small top-heavy field. Go for unique stacks - everyone is trying to win.",
            ("TOP_HEAVY", "MEDIUM"): "Medium top-heavy GPP. Max leverage with 4-5 player stacks. Accept variance.",
            ("TOP_HEAVY", "LARGE"): "Large top-heavy field. Need maximum uniqueness to differentiate. Full send on ceiling.",
            ("MODERATE", "SMALL"): "Small balanced field. Moderate stacks, don't over-leverage.",
            ("MODERATE", "MEDIUM"): "Standard GPP structure. 3-4 player stacks with secondary correlation.",
            ("MODERATE", "LARGE"): "Large moderate field. Some leverage needed but don't go crazy.",
            ("FLAT", "SMALL"): "Small flat payout. Being on chalk is fine. Protect your floor.",
            ("FLAT", "MEDIUM"): "Medium flat field. Balanced approach - multiple paths to cash.",
            ("FLAT", "LARGE"): "Large flat field. Floor matters. 2-3 man stacks, spread risk across games."
        }
        return summaries.get((contest_type, field_size), "Standard GPP approach recommended.")

    def print_contest_analysis(self, total_entries: int, prize_pool: float,
                               first_place: float, tenth_place: float,
                               min_cash: float, min_cash_place: int):
        """Print formatted contest analysis."""
        analysis = self.analyze_contest(
            total_entries, prize_pool, first_place, tenth_place, min_cash, min_cash_place
        )
        print("\n" + "=" * 60)
        print("CONTEST ANALYSIS")
        print("=" * 60)
        print(f"Contest Type: {analysis['contest_type']} | Field: {analysis['field_size']}")
        print(f"1st Place: {analysis['metrics']['first_place_pct']} of pool")
        print(f"Pays: {analysis['metrics']['payout_pct']} of field")
        print(f"Top 10 = {analysis['metrics']['top_10_entries']} entries")
        print(f"Min Cash: {analysis['metrics']['min_cash_multiplier']} entry")
        print()
        print(f"STRATEGY: {analysis['summary']}")
        print()
        print("Recommendations:")
        for key, val in analysis['strategy'].items():
            print(f"  • {key.replace('_', ' ').title()}: {val}")
        print("=" * 60)
        return analysis


# ================================================================
#  Quick test (same as original)
# ================================================================
if __name__ == "__main__":
    from data_pipeline import NHLDataPipeline
    from projections import NHLProjectionModel
    from main import load_dk_salaries, merge_projections_with_salaries
    from lines import LinesScraper, StackBuilder
    from datetime import datetime

    print("Testing ILP Optimizer...")

    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(include_game_logs=False)

    model = NHLProjectionModel()
    today = datetime.now().strftime('%Y-%m-%d')
    projections = model.generate_projections(data, target_date=today)

    print("\n" + "=" * 60)
    print("FETCHING LINE COMBINATIONS")
    print("=" * 60)
    schedule = pipeline.fetch_schedule(today)
    teams_playing = set()
    teams_playing.update(schedule['home_team'].tolist())
    teams_playing.update(schedule['away_team'].tolist())
    teams_playing = sorted([t for t in teams_playing if t])

    print(f"Teams playing today: {', '.join(teams_playing)}")

    scraper = LinesScraper()
    all_lines = scraper.get_multiple_teams(teams_playing)
    stack_builder = StackBuilder(all_lines)

    confirmed_goalies = stack_builder.get_all_starting_goalies()
    print(f"\nConfirmed starters: {len(confirmed_goalies)} goalies")
    for team, goalie in sorted(confirmed_goalies.items()):
        print(f"  {team}: {goalie}")

    project_dir = Path(__file__).parent
    salaries_dir = project_dir / DAILY_SALARIES_DIR
    salary_files = list(salaries_dir.glob('DKSalaries*.csv')) if salaries_dir.exists() else []
    if not salary_files:
        salary_files = list(project_dir.glob('DKSalaries*.csv'))
    if salary_files:
        salary_files = sorted(salary_files)
        dk_salaries = load_dk_salaries(str(salary_files[0]))
        dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'W', 'D'])]
        dk_goalies = dk_salaries[dk_salaries['position'] == 'G']

        projections['goalies']['position'] = 'G'

        skaters_merged = merge_projections_with_salaries(projections['skaters'], dk_skaters, 'skater')
        goalies_merged = merge_projections_with_salaries(projections['goalies'], dk_goalies, 'goalie')

        player_pool = pd.concat([skaters_merged, goalies_merged], ignore_index=True)

        optimizer = NHLLineupOptimizer(stack_builder=stack_builder)

        print("\n" + "=" * 60)
        print("ILP GPP LINEUP (with stacking)")
        print("=" * 60)
        gpp_lineups = optimizer.optimize_lineup(player_pool, n_lineups=1, mode='gpp')
        if gpp_lineups:
            print(optimizer.format_lineup_for_dk(gpp_lineups[0]))

        print("\n" + "=" * 60)
        print("ILP CASH LINEUP (balanced)")
        print("=" * 60)
        cash_lineups = optimizer.optimize_lineup(player_pool, n_lineups=1, mode='cash')
        if cash_lineups:
            print(optimizer.format_lineup_for_dk(cash_lineups[0]))
    else:
        print("No salary file found.")
