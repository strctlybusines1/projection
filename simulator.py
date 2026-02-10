"""
Optimal Lineup Simulator — exhaustive team-pair frequency analysis.

Supports two modes:
  - Deterministic (n_iterations=0): Uses fixed projected_fpts (original behavior).
  - Monte Carlo (n_iterations>=1): Samples from N(projected_fpts, std_fpts) each
    iteration, producing realistic exposure rates that account for game-to-game variance.

Iterates all valid (team_A, team_B) pairs on a slate, builds the best
4-3-1-1 lineup for each pair, and counts how often each player lands in
an optimal lineup.

Lineup structure (mirrors FireDog50's winning pattern):
    4 skaters from team A  (primary stack)
    3 skaters from team B  (secondary stack)
    1 fill skater from any other team (not A, not B)
    1 goalie — NOT from a team that opposes any of the 8 skaters

DraftKings roster: 2C / 3W / 2D / 1G / 1 UTIL (C or W only — D excluded).
Salary cap: $50,000.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from typing import Optional, List, Tuple, Dict


class OptimalLineupSimulator:

    SALARY_CAP = 50_000
    MIN_GOALIE_SALARY = 2_500
    MIN_FILL_SALARY = 2_500
    TOP_N_PER_TEAM = 12
    MC_TOP_N_PER_TEAM = 8  # Reduced pool in MC mode for speed

    # ------------------------------------------------------------------ #
    #  Init
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        player_pool: pd.DataFrame,
        dk_salaries: pd.DataFrame,
        std_dev_data: Optional[Dict[int, Dict]] = None,
        n_iterations: int = 0,
    ):
        pool = player_pool.copy()
        dk = dk_salaries.copy()
        self._n_iterations = n_iterations
        self._std_dev_data = std_dev_data or {}

        # Use smaller pool in MC mode for performance
        if n_iterations > 0:
            self.TOP_N_PER_TEAM = self.MC_TOP_N_PER_TEAM

        # --- position normalisation ----------------------------------- #
        pool['sim_pos'] = pool['position'].apply(self._normalise_pos)
        if 'dk_pos' in pool.columns:
            pool['sim_pos'] = pool['dk_pos'].apply(self._normalise_pos)

        # --- opponent map from DK Game Info column -------------------- #
        self.opponent_map: Dict[str, str] = {}  # team -> opponent
        gi_col = 'Game Info' if 'Game Info' in dk.columns else 'game_info'
        if gi_col in dk.columns:
            for _, row in dk.drop_duplicates('TeamAbbrev' if 'TeamAbbrev' in dk.columns else 'team').iterrows():
                team_col = 'TeamAbbrev' if 'TeamAbbrev' in dk.columns else 'team'
                team = str(row[team_col]).upper()
                gi = str(row.get(gi_col, ''))
                opp = self._parse_opponent(team, gi)
                if opp:
                    self.opponent_map[team] = opp

        # --- store full pool for MC re-sampling ----------------------- #
        self._pool = pool

        # --- build std_fpts lookup by player name --------------------- #
        self._std_by_name: Dict[str, float] = {}
        if std_dev_data and 'player_id' in pool.columns:
            for _, row in pool.iterrows():
                pid = row.get('player_id')
                if pid and pid in std_dev_data:
                    self._std_by_name[row['name']] = std_dev_data[pid]['std_fpts']

        # --- initial sort using projected_fpts (deterministic base) --- #
        self._build_pools('projected_fpts')

    def _build_pools(self, fpts_col: str):
        """Split pool into skaters/goalies sorted by fpts_col, index by team."""
        pool = self._pool

        self.skaters = (
            pool[pool['sim_pos'].isin(['C', 'W', 'D'])]
            .sort_values(fpts_col, ascending=False)
            .reset_index(drop=True)
        )
        self.goalies = (
            pool[pool['sim_pos'] == 'G']
            .sort_values(fpts_col, ascending=False)
            .reset_index(drop=True)
        )

        self.skaters_by_team: Dict[str, pd.DataFrame] = {}
        for team, grp in self.skaters.groupby('team'):
            self.skaters_by_team[team] = grp.head(self.TOP_N_PER_TEAM).reset_index(drop=True)

        self.all_teams = sorted(self.skaters['team'].unique())

    # ------------------------------------------------------------------ #
    #  Monte Carlo sampling
    # ------------------------------------------------------------------ #
    def _sample_projections(self):
        """Draw sampled_fpts from Gamma(mean, var) per player, rebuild pools.
        
        Gamma replaces N(projected, std) because:
        - Support on [0, ∞) — FPTS can't be negative
        - Right-skewed — matches real FPTS distribution  
        - Variance calibrated from backtest (2,137 obs across 7 dates)
        """
        pool = self._pool
        projected = pool['projected_fpts'].values.copy()
        positions = pool['position'].values if 'position' in pool.columns else np.array(['W'] * len(pool))
        salaries = pool['salary'].values if 'salary' in pool.columns else np.full(len(pool), 4000)
        
        try:
            from stochastic_upgrades import sample_fpts_gamma
            sampled = np.zeros(len(projected))
            for i in range(len(projected)):
                if projected[i] > 0:
                    sampled[i] = sample_fpts_gamma(projected[i], positions[i], salaries[i], n=1)[0]
                else:
                    sampled[i] = 0.0
        except ImportError:
            # Fallback to original Normal if stochastic_upgrades not available
            std = np.array([self._std_by_name.get(name, 5.5) for name in pool['name'].values])
            sampled = np.maximum(np.random.normal(projected, std), 0.0)
        
        pool = pool.copy()
        pool['sampled_fpts'] = sampled
        self._pool = pool
        self._build_pools('sampled_fpts')

    # ------------------------------------------------------------------ #
    #  Baseline probability per position
    # ------------------------------------------------------------------ #
    def _compute_baselines(self) -> Dict[str, float]:
        """
        Baseline probability that a random player at each position appears
        in an optimal lineup, accounting for UTIL slot (C/W only, D excluded).

        effective_C_slots = 2 + N_C / (N_C + N_W)
        effective_W_slots = 3 + N_W / (N_C + N_W)
        effective_D_slots = 2
        effective_G_slots = 1

        baseline_pct(pos) = effective_slots / pool_size * 100
        """
        n_c = len(self._pool[self._pool['sim_pos'] == 'C'])
        n_w = len(self._pool[self._pool['sim_pos'] == 'W'])
        n_d = len(self._pool[self._pool['sim_pos'] == 'D'])
        n_g = len(self._pool[self._pool['sim_pos'] == 'G'])

        self._pool_counts = {'C': n_c, 'W': n_w, 'D': n_d, 'G': n_g}

        cw_total = n_c + n_w
        util_c = n_c / cw_total if cw_total > 0 else 0.5
        util_w = n_w / cw_total if cw_total > 0 else 0.5

        baselines = {}
        baselines['C'] = round((2 + util_c) / n_c * 100, 1) if n_c > 0 else 0.0
        baselines['W'] = round((3 + util_w) / n_w * 100, 1) if n_w > 0 else 0.0
        baselines['D'] = round(2 / n_d * 100, 1) if n_d > 0 else 0.0
        baselines['G'] = round(1 / n_g * 100, 1) if n_g > 0 else 0.0

        return baselines

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_pos(pos) -> str:
        if pd.isna(pos):
            return 'W'
        pos = str(pos).upper().strip()
        if pos in ('L', 'LW', 'R', 'RW', 'W'):
            return 'W'
        if pos in ('C', 'C/W', 'W/C'):
            return 'C'
        if pos in ('D', 'LD', 'RD'):
            return 'D'
        if pos == 'G':
            return 'G'
        return pos

    @staticmethod
    def _parse_opponent(team: str, game_info: str) -> Optional[str]:
        """Extract opponent from 'MIN@EDM 01/31/2026 10:00PM ET'."""
        try:
            matchup = game_info.split()[0]
            if '@' in matchup:
                away, home = matchup.split('@')
                away, home = away.upper(), home.upper()
                if team == away:
                    return home
                elif team == home:
                    return away
        except (IndexError, ValueError, AttributeError):
            pass
        return None

    # ------------------------------------------------------------------ #
    #  Position feasibility
    # ------------------------------------------------------------------ #
    @staticmethod
    def _positions_feasible(positions: List[str]) -> bool:
        """
        Given exactly 8 skater positions, check whether they can fill
        2C + 3W + 2D + 1 UTIL (C/W only).

        Valid distributions: (2C 4W 2D) or (3C 3W 2D).
        D count must be exactly 2 (D cannot fill UTIL).
        """
        c = positions.count('C')
        w = positions.count('W')
        d = positions.count('D')
        if d != 2:
            return False
        if c + w != 6:
            return False
        # Need at least 2C and at least 3W
        # UTIL absorbs one extra C or W -> valid combos: 2C/4W or 3C/3W
        if c >= 2 and w >= 3:
            return True
        return False

    def _position_need(self, current_positions: List[str]) -> Optional[str]:
        """
        Given 7 skater positions, return which position the 8th must be
        to make the set feasible, or None if no single addition works.
        Returns 'CW' if either C or W would work.
        """
        c = current_positions.count('C')
        w = current_positions.count('W')
        d = current_positions.count('D')

        # We need exactly 2D total
        if d < 2:
            need_d = 2 - d
            # If we need a D as the 8th player
            if need_d == 1:
                # Check if C+W already works with remaining 5 spots being C/W
                cw_total = c + w
                if cw_total == 5 and c >= 2 and w >= 3:
                    return 'D'
                elif cw_total == 5 and (c >= 2 or w >= 3):
                    return 'D'
                elif cw_total == 5:
                    return 'D'
                return 'D'
            else:
                # Need 2 more D but only 1 spot -- impossible
                return None
        elif d > 2:
            return None  # Already too many D

        # d == 2, need C/W for the 8th slot
        cw_total = c + w  # should be 5 (7 skaters - 2 D)
        if cw_total != 5:
            return None

        # We need 6 C/W total -> adding one more
        # Check if adding C keeps feasible (c+1 >= 2 and w >= 3)
        c_ok = (c + 1 >= 2) and (w >= 3)
        # Check if adding W keeps feasible (c >= 2 and w+1 >= 3)
        w_ok = (c >= 2) and (w + 1 >= 3)

        if c_ok and w_ok:
            return 'CW'  # either works
        elif c_ok:
            return 'C'
        elif w_ok:
            return 'W'
        return None

    # ------------------------------------------------------------------ #
    #  Find best fill skater
    # ------------------------------------------------------------------ #
    def _find_best_fill(
        self,
        excluded_teams: set,
        used_names: set,
        current_positions: List[str],
        remaining_salary: int,
        fpts_col: str = 'projected_fpts',
    ) -> Optional[pd.Series]:
        """
        Scan pre-sorted skaters for the highest-projected fill player.
        Must not be from excluded teams, already used, or over budget.
        Must make the 8-skater position set feasible.
        Reserves min goalie salary from remaining budget.
        """
        need = self._position_need(current_positions)
        if need is None:
            return None

        budget = remaining_salary - self.MIN_GOALIE_SALARY

        for _, player in self.skaters.iterrows():
            if player['name'] in used_names:
                continue
            if player['team'] in excluded_teams:
                continue
            if player['salary'] > budget:
                continue
            pos = player['sim_pos']
            if need == 'D' and pos != 'D':
                continue
            if need == 'C' and pos != 'C':
                continue
            if need == 'W' and pos != 'W':
                continue
            if need == 'CW' and pos not in ('C', 'W'):
                continue
            return player
        return None

    # ------------------------------------------------------------------ #
    #  Find best goalie
    # ------------------------------------------------------------------ #
    def _find_best_goalie(
        self,
        excluded_opponent_teams: set,
        remaining_salary: int,
    ) -> Optional[pd.Series]:
        """
        Scan pre-sorted goalies for best projected goalie whose team
        is NOT in excluded_opponent_teams (i.e. doesn't oppose any skater).
        """
        for _, goalie in self.goalies.iterrows():
            if goalie['salary'] > remaining_salary:
                continue
            team = goalie['team']
            if team in excluded_opponent_teams:
                continue
            return goalie
        return None

    # ------------------------------------------------------------------ #
    #  Build optimal lineup for one (A, B) pair
    # ------------------------------------------------------------------ #
    def _build_optimal_lineup(
        self, team_a: str, team_b: str, fpts_col: str = 'projected_fpts'
    ) -> Optional[Tuple[float, List[pd.Series]]]:
        """
        Exhaustively search C(N,4) x C(N,3) combos from A x B,
        find the best fill + goalie, return best total fpts and player list.
        """
        skaters_a = self.skaters_by_team.get(team_a)
        skaters_b = self.skaters_by_team.get(team_b)
        if skaters_a is None or skaters_b is None:
            return None
        if len(skaters_a) < 4 or len(skaters_b) < 3:
            return None

        # Pre-compute arrays for speed
        a_names = skaters_a['name'].values
        a_salaries = skaters_a['salary'].values
        a_fpts = skaters_a[fpts_col].values
        a_pos = skaters_a['sim_pos'].values

        b_names = skaters_b['name'].values
        b_salaries = skaters_b['salary'].values
        b_fpts = skaters_b[fpts_col].values
        b_pos = skaters_b['sim_pos'].values

        n_a = len(skaters_a)
        n_b = len(skaters_b)

        # Min cost for fill + goalie
        min_remaining_cost = self.MIN_FILL_SALARY + self.MIN_GOALIE_SALARY

        best_total = -1.0
        best_lineup = None

        for combo_a in combinations(range(n_a), 4):
            sal_a = sum(a_salaries[i] for i in combo_a)
            fpts_a = sum(a_fpts[i] for i in combo_a)
            pos_a = [a_pos[i] for i in combo_a]

            # Early prune on salary
            if sal_a > self.SALARY_CAP - min_remaining_cost - self.MIN_FILL_SALARY * 3:
                continue

            for combo_b in combinations(range(n_b), 3):
                sal_b = sum(b_salaries[j] for j in combo_b)
                fpts_b = sum(b_fpts[j] for j in combo_b)
                sal_7 = sal_a + sal_b

                # Early prune: need room for fill + goalie
                if sal_7 > self.SALARY_CAP - min_remaining_cost:
                    continue

                # Early prune on projection ceiling (if we already have a
                # good lineup, skip combos that can't beat it even with
                # perfect fill + goalie)
                if best_total > 0:
                    # Generous upper bound for fill + goalie
                    max_fill_fpts = self.skaters[fpts_col].iloc[0] if len(self.skaters) > 0 else 0
                    max_goalie_fpts = self.goalies[fpts_col].iloc[0] if len(self.goalies) > 0 else 0
                    if fpts_a + fpts_b + max_fill_fpts + max_goalie_fpts <= best_total:
                        continue

                pos_b = [b_pos[j] for j in combo_b]
                positions_7 = pos_a + pos_b

                used_names = set(a_names[i] for i in combo_a) | set(b_names[j] for j in combo_b)
                excluded_teams = {team_a, team_b}
                remaining_salary = self.SALARY_CAP - sal_7

                # Find best fill
                fill = self._find_best_fill(
                    excluded_teams, used_names, positions_7, remaining_salary, fpts_col
                )
                if fill is None:
                    continue

                positions_8 = positions_7 + [fill['sim_pos']]
                if not self._positions_feasible(positions_8):
                    continue

                fill_team = fill['team']
                remaining_after_fill = remaining_salary - fill['salary']

                # Build excluded opponent teams for goalie
                # Goalie cannot be from a team that opposes any of our 3 skater teams
                skater_teams = {team_a, team_b, fill_team}
                excluded_opp_teams = set()
                for t in skater_teams:
                    opp = self.opponent_map.get(t)
                    if opp:
                        excluded_opp_teams.add(opp)

                goalie = self._find_best_goalie(excluded_opp_teams, remaining_after_fill)
                if goalie is None:
                    continue

                total_fpts = fpts_a + fpts_b + fill[fpts_col] + goalie[fpts_col]
                if total_fpts > best_total:
                    best_total = total_fpts
                    # Build player list
                    players_a = [skaters_a.iloc[i] for i in combo_a]
                    players_b = [skaters_b.iloc[j] for j in combo_b]
                    best_lineup = players_a + players_b + [fill, goalie]

        if best_lineup is None:
            return None
        return best_total, best_lineup

    # ------------------------------------------------------------------ #
    #  Main run loop
    # ------------------------------------------------------------------ #
    def run(self) -> pd.DataFrame:
        """
        Generate all valid (A, B) pairs, build optimal lineup for each,
        return frequency table of player appearances.

        In MC mode, repeats the process n_iterations times with sampled projections.
        """
        teams = self.all_teams
        pairs = []
        for a in teams:
            for b in teams:
                if a == b:
                    continue
                # Skip if A opposes B (goalie can't avoid opposing all skaters)
                if self.opponent_map.get(a) == b:
                    continue
                pairs.append((a, b))

        is_mc = self._n_iterations > 0
        n_iter = self._n_iterations if is_mc else 1
        fpts_col = 'sampled_fpts' if is_mc else 'projected_fpts'

        mode_str = f"Monte Carlo ({n_iter} iterations, TOP_N={self.TOP_N_PER_TEAM})" if is_mc else "Deterministic"
        print(f"\nSimulator: {len(teams)} teams, {len(pairs)} valid ordered pairs — {mode_str}")

        counts: Dict[str, int] = defaultdict(int)
        total_valid_lineups = 0

        for iteration in range(n_iter):
            if is_mc:
                self._sample_projections()

            valid_this_iter = 0
            for idx, (a, b) in enumerate(pairs):
                result = self._build_optimal_lineup(a, b, fpts_col)
                if result is not None:
                    valid_this_iter += 1
                    _, players = result
                    for p in players:
                        counts[p['name']] += 1

            total_valid_lineups += valid_this_iter

            if is_mc:
                if (iteration + 1) % 10 == 0 or iteration + 1 == n_iter:
                    print(f"  Iteration {iteration + 1}/{n_iter} — "
                          f"{valid_this_iter} valid lineups this iter, "
                          f"{total_valid_lineups} total")
            else:
                # Deterministic: show pair-level progress
                print(f"  Processed {len(pairs)} pairs ({valid_this_iter} valid lineups)")

        self._valid_lineups = total_valid_lineups
        self._total_pairs = len(pairs) * n_iter

        # Build results DataFrame
        if not counts:
            return pd.DataFrame()

        # Restore original pool for player info lookup
        if is_mc:
            self._build_pools('projected_fpts')

        rows = []
        all_players = pd.concat([self.skaters, self.goalies], ignore_index=True)
        player_info = {}
        for _, p in all_players.iterrows():
            if p['name'] not in player_info:
                player_info[p['name']] = p

        for name, count in counts.items():
            info = player_info.get(name)
            if info is None:
                continue
            row = {
                'name': name,
                'team': info.get('team', ''),
                'position': info.get('sim_pos', ''),
                'salary': info.get('salary', 0),
                'projected_fpts': info.get('projected_fpts', 0),
                'count': count,
                'pct': round(100.0 * count / total_valid_lineups, 1) if total_valid_lineups else 0,
            }
            # Add std_fpts column if available
            if name in self._std_by_name:
                row['std_fpts'] = round(self._std_by_name[name], 2)
            # Preserve pre-lift original projection if present
            if 'pre_lift_fpts' in info.index:
                row['pre_lift_fpts'] = info['pre_lift_fpts']
            rows.append(row)

        results = pd.DataFrame(rows).sort_values('count', ascending=False).reset_index(drop=True)

        # Baseline probability and lift
        self._baselines = self._compute_baselines()
        results['baseline_pct'] = results['position'].map(self._baselines)
        results['lift'] = np.where(
            results['baseline_pct'] > 0,
            results['pct'] / results['baseline_pct'],
            0.0,
        )
        results['lift'] = results['lift'].round(1)

        return results

    # ------------------------------------------------------------------ #
    #  Lift-adjusted re-simulation
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_lift_adjustments(
        player_pool: pd.DataFrame,
        lift_results: pd.DataFrame,
        blend: float = 0.15,
    ) -> pd.DataFrame:
        """Apply lift-based projection adjustments for a second-pass simulation.

        Formula: adjusted_fpts = projected_fpts * (1 + blend * (lift - 1.0))

        Players missing from lift_results (never appeared in first pass)
        get lift = 0.0, so they are penalized.

        Args:
            player_pool: Full player pool DataFrame with 'projected_fpts'.
            lift_results: First-pass results DataFrame with 'name' and 'lift'.
            blend: Blend factor (default 0.15). Higher = stronger adjustment.

        Returns:
            New player pool DataFrame with adjusted projected_fpts and
            original values stored in 'pre_lift_fpts'.
        """
        pool = player_pool.copy()

        # Build name -> lift mapping from first-pass results
        lift_map = dict(zip(lift_results['name'], lift_results['lift']))

        # Map lift to pool; players not in results get 0.0 (penalized)
        pool['_lift'] = pool['name'].map(lift_map).fillna(0.0)

        # Store original projections
        pool['pre_lift_fpts'] = pool['projected_fpts'].copy()

        # Apply adjustment
        pool['projected_fpts'] = pool['projected_fpts'] * (1.0 + blend * (pool['_lift'] - 1.0))

        pool = pool.drop(columns=['_lift'])
        return pool

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #
    def print_results(self, results: pd.DataFrame, top_n: int = 30):
        """Formatted frequency table."""
        if results.empty:
            print("\nNo valid lineups generated.")
            return

        is_mc = self._n_iterations > 0
        mode_str = f"MONTE CARLO ({self._n_iterations} iter)" if is_mc else "DETERMINISTIC"
        has_std = 'std_fpts' in results.columns
        has_lift_adj = 'pre_lift_fpts' in results.columns

        lift_tag = " [LIFT-ADJUSTED]" if has_lift_adj else ""
        print(f"\n{'=' * 110}")
        print(f" OPTIMAL LINEUP SIMULATOR — FREQUENCY RESULTS [{mode_str}]{lift_tag}")
        print(f"{'=' * 110}")
        if is_mc:
            print(f" Mode: Monte Carlo | Iterations: {self._n_iterations} | TOP_N: {self.TOP_N_PER_TEAM}")
        else:
            print(f" Mode: Deterministic | TOP_N: {self.TOP_N_PER_TEAM}")
        print(f" Total valid lineups: {self._valid_lineups} / {self._total_pairs} pair-iterations")

        # Baseline probability header
        if hasattr(self, '_baselines') and hasattr(self, '_pool_counts'):
            bl = self._baselines
            pc = self._pool_counts
            print(f"\n Baseline probability (random selection into 2C/3W/2D/1G/1UTIL):")
            print(f"   C: {bl['C']}% ({pc['C']} players, 2 slots + UTIL share)")
            print(f"   W: {bl['W']}% ({pc['W']} players, 3 slots + UTIL share)")
            print(f"   D: {bl['D']}% ({pc['D']} players, 2 slots)")
            print(f"   G: {bl['G']}% ({pc['G']} players, 1 slot)")
        print()

        def _print_section(section_df, section_title):
            print(f" {section_title}")
            # Build header
            if has_lift_adj:
                print(f" {'Name':<28} {'Team':<5} {'Pos':<4} {'Salary':>8} {'OrigProj':>8} {'AdjProj':>8} "
                      f"{'Count':>6} {'Pct':>6} {'Lift':>6}")
            elif has_std:
                print(f" {'Name':<28} {'Team':<5} {'Pos':<4} {'Salary':>8} {'Proj':>6} "
                      f"{'Std':>6} {'Count':>6} {'Pct':>6} {'Lift':>6}")
            else:
                print(f" {'Name':<28} {'Team':<5} {'Pos':<4} {'Salary':>8} {'Proj':>6} "
                      f"{'Count':>6} {'Pct':>6} {'Lift':>6}")
            print(f" {'-' * 102}")
            for _, row in section_df.iterrows():
                lift_str = f"{row['lift']:.1f}x" if pd.notna(row.get('lift')) else "    -"
                if has_lift_adj:
                    orig = row.get('pre_lift_fpts', row['projected_fpts'])
                    print(f" {row['name']:<28} {row['team']:<5} {row['position']:<4} "
                          f"${row['salary']:>7,} {orig:>8.1f} {row['projected_fpts']:>8.1f} "
                          f"{row['count']:>6} {row['pct']:>5.1f}% {lift_str:>6}")
                elif has_std:
                    std_str = f" {row['std_fpts']:>5.1f}" if pd.notna(row.get('std_fpts')) else "     -"
                    print(f" {row['name']:<28} {row['team']:<5} {row['position']:<4} "
                          f"${row['salary']:>7,} {row['projected_fpts']:>6.1f}{std_str} "
                          f"{row['count']:>6} {row['pct']:>5.1f}% {lift_str:>6}")
                else:
                    print(f" {row['name']:<28} {row['team']:<5} {row['position']:<4} "
                          f"${row['salary']:>7,} {row['projected_fpts']:>6.1f} "
                          f"{row['count']:>6} {row['pct']:>5.1f}% {lift_str:>6}")

        # Skaters
        skaters = results[results['position'] != 'G'].head(top_n)
        _print_section(skaters, f"TOP {min(top_n, len(skaters))} SKATERS BY FREQUENCY")

        # Goalies
        goalies = results[results['position'] == 'G']
        if not goalies.empty:
            print()
            _print_section(goalies, "GOALIES BY FREQUENCY")

    def export_results(self, results: pd.DataFrame, output_path: str):
        """Export frequency table to CSV."""
        if results.empty:
            return
        results.to_csv(output_path, index=False)
        print(f"\nSimulator results exported to: {output_path}")


# ---------------------------------------------------------------------- #
#  Standalone test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime

    project_dir = Path(__file__).parent

    # Auto-detect latest salary file
    sal_dir = project_dir / "daily_salaries"
    sal_files = sorted(sal_dir.glob("DKSalaries*.csv")) if sal_dir.exists() else []
    if not sal_files:
        print("No salary file found in daily_salaries/")
        raise SystemExit(1)
    sal_path = str(sal_files[-1])

    from main import load_dk_salaries, merge_projections_with_salaries, normalize_positions_column
    from data_pipeline import NHLDataPipeline
    from projections import NHLProjectionModel

    print("Loading data for simulator standalone test...")
    dk_salaries = load_dk_salaries(sal_path)
    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(include_game_logs=False, include_advanced_stats=False)
    model = NHLProjectionModel()
    today = datetime.now().strftime('%Y-%m-%d')
    projections = model.generate_projections(data, target_date=today)

    dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'W', 'D'])]
    dk_goalies = dk_salaries[dk_salaries['position'] == 'G']
    if 'position' not in projections['goalies'].columns:
        projections['goalies']['position'] = 'G'
    skaters_merged = merge_projections_with_salaries(projections['skaters'], dk_skaters, 'skater')
    goalies_merged = merge_projections_with_salaries(projections['goalies'], dk_goalies, 'goalie')
    import pandas as _pd
    player_pool = _pd.concat([skaters_merged, goalies_merged], ignore_index=True)

    sim = OptimalLineupSimulator(player_pool, dk_salaries)
    results = sim.run()
    sim.print_results(results)

    out_dir = project_dir / "daily_projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_str = datetime.now().strftime('%m_%d_%y')
    sim.export_results(results, str(out_dir / f"{date_str}NHLsimulator_{ts}.csv"))
