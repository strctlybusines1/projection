"""
Optimal Lineup Simulator — exhaustive team-pair frequency analysis.

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

    # ------------------------------------------------------------------ #
    #  Init
    # ------------------------------------------------------------------ #
    def __init__(self, player_pool: pd.DataFrame, dk_salaries: pd.DataFrame):
        pool = player_pool.copy()
        dk = dk_salaries.copy()

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

        # --- split skaters / goalies, pre-sort by projection ---------- #
        self.skaters = (
            pool[pool['sim_pos'].isin(['C', 'W', 'D'])]
            .sort_values('projected_fpts', ascending=False)
            .reset_index(drop=True)
        )
        self.goalies = (
            pool[pool['sim_pos'] == 'G']
            .sort_values('projected_fpts', ascending=False)
            .reset_index(drop=True)
        )

        # --- index skaters by team ------------------------------------ #
        self.skaters_by_team: Dict[str, pd.DataFrame] = {}
        for team, grp in self.skaters.groupby('team'):
            self.skaters_by_team[team] = grp.head(self.TOP_N_PER_TEAM).reset_index(drop=True)

        self.all_teams = sorted(self.skaters['team'].unique())

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
        # UTIL absorbs one extra C or W → valid combos: 2C/4W or 3C/3W
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
                # Need 2 more D but only 1 spot — impossible
                return None
        elif d > 2:
            return None  # Already too many D

        # d == 2, need C/W for the 8th slot
        cw_total = c + w  # should be 5 (7 skaters - 2 D)
        if cw_total != 5:
            return None

        # We need 6 C/W total → adding one more
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
        self, team_a: str, team_b: str
    ) -> Optional[Tuple[float, List[pd.Series]]]:
        """
        Exhaustively search C(12,4) × C(12,3) combos from A × B,
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
        a_fpts = skaters_a['projected_fpts'].values
        a_pos = skaters_a['sim_pos'].values

        b_names = skaters_b['name'].values
        b_salaries = skaters_b['salary'].values
        b_fpts = skaters_b['projected_fpts'].values
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
                    max_fill_fpts = self.skaters['projected_fpts'].iloc[0] if len(self.skaters) > 0 else 0
                    max_goalie_fpts = self.goalies['projected_fpts'].iloc[0] if len(self.goalies) > 0 else 0
                    if fpts_a + fpts_b + max_fill_fpts + max_goalie_fpts <= best_total:
                        continue

                pos_b = [b_pos[j] for j in combo_b]
                positions_7 = pos_a + pos_b

                used_names = set(a_names[i] for i in combo_a) | set(b_names[j] for j in combo_b)
                excluded_teams = {team_a, team_b}
                remaining_salary = self.SALARY_CAP - sal_7

                # Find best fill
                fill = self._find_best_fill(
                    excluded_teams, used_names, positions_7, remaining_salary
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

                total_fpts = fpts_a + fpts_b + fill['projected_fpts'] + goalie['projected_fpts']
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

        print(f"\nSimulator: {len(teams)} teams, {len(pairs)} valid ordered pairs")

        counts: Dict[str, int] = defaultdict(int)
        valid_lineups = 0

        for idx, (a, b) in enumerate(pairs):
            result = self._build_optimal_lineup(a, b)
            if result is not None:
                valid_lineups += 1
                _, players = result
                for p in players:
                    counts[p['name']] += 1

            if (idx + 1) % 50 == 0 or idx + 1 == len(pairs):
                print(f"  Processed {idx + 1}/{len(pairs)} pairs "
                      f"({valid_lineups} valid lineups so far)")

        self._valid_lineups = valid_lineups
        self._total_pairs = len(pairs)

        # Build results DataFrame
        if not counts:
            return pd.DataFrame()

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
            rows.append({
                'name': name,
                'team': info.get('team', ''),
                'position': info.get('sim_pos', ''),
                'salary': info.get('salary', 0),
                'projected_fpts': info.get('projected_fpts', 0),
                'count': count,
                'pct': round(100.0 * count / valid_lineups, 1) if valid_lineups else 0,
            })

        results = pd.DataFrame(rows).sort_values('count', ascending=False).reset_index(drop=True)
        return results

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #
    def print_results(self, results: pd.DataFrame, top_n: int = 30):
        """Formatted frequency table."""
        if results.empty:
            print("\nNo valid lineups generated.")
            return

        print(f"\n{'=' * 85}")
        print(f" OPTIMAL LINEUP SIMULATOR — FREQUENCY RESULTS")
        print(f"{'=' * 85}")
        print(f" Total valid lineups: {self._valid_lineups} / {self._total_pairs} pairs\n")

        # Skaters
        skaters = results[results['position'] != 'G'].head(top_n)
        print(f" TOP {min(top_n, len(skaters))} SKATERS BY FREQUENCY")
        print(f" {'Name':<28} {'Team':<5} {'Pos':<4} {'Salary':>8} {'Proj':>6} "
              f"{'Count':>6} {'Pct':>6}")
        print(f" {'-' * 75}")
        for _, row in skaters.iterrows():
            print(f" {row['name']:<28} {row['team']:<5} {row['position']:<4} "
                  f"${row['salary']:>7,} {row['projected_fpts']:>6.1f} "
                  f"{row['count']:>6} {row['pct']:>5.1f}%")

        # Goalies
        goalies = results[results['position'] == 'G']
        if not goalies.empty:
            print(f"\n GOALIES BY FREQUENCY")
            print(f" {'Name':<28} {'Team':<5} {'Pos':<4} {'Salary':>8} {'Proj':>6} "
                  f"{'Count':>6} {'Pct':>6}")
            print(f" {'-' * 75}")
            for _, row in goalies.iterrows():
                print(f" {row['name']:<28} {row['team']:<5} {row['position']:<4} "
                      f"${row['salary']:>7,} {row['projected_fpts']:>6.1f} "
                      f"{row['count']:>6} {row['pct']:>5.1f}%")

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
