"""
DraftKings NHL Lineup Optimizer with Stacking Support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class NHLLineupOptimizer:
    """
    Optimizer for DraftKings NHL lineups with correlation-based stacking.

    DK NHL Classic Roster:
    - 2 Centers (C)
    - 3 Wings (W) - can be LW or RW
    - 2 Defensemen (D)
    - 1 Goalie (G)
    - 1 UTIL (any skater - C, W, or D)

    Salary Cap: $50,000
    """

    SALARY_CAP = 50000
    ROSTER_REQUIREMENTS = {
        'C': 2,
        'W': 3,
        'D': 2,
        'G': 1,
        'UTIL': 1  # Any skater (C, W, or D)
    }

    def __init__(self, stack_builder=None):
        """
        Initialize optimizer.

        Args:
            stack_builder: Optional StackBuilder instance for correlation data
        """
        self.stack_builder = stack_builder

    def _normalize_position(self, pos: str) -> str:
        """Normalize position codes."""
        pos = str(pos).upper()
        if pos in ['L', 'R', 'LW', 'RW', 'W']:
            return 'W'
        if pos in ['C', 'C/W', 'W/C']:
            return 'C'
        if pos in ['D', 'LD', 'RD']:
            return 'D'
        if pos in ['G']:
            return 'G'
        return pos

    def optimize_lineup(self, player_pool: pd.DataFrame,
                        n_lineups: int = 1,
                        max_from_team: int = 4,
                        min_teams: int = 3,
                        randomness: float = 0.0,
                        stack_boost: float = 0.15,
                        force_stack: str = None) -> List[pd.DataFrame]:
        """
        Generate optimized lineups using value-based selection with stacking.

        Args:
            player_pool: DataFrame with projections and salaries
            n_lineups: Number of lineups to generate
            max_from_team: Maximum players from one team
            min_teams: Minimum number of teams represented
            randomness: Add randomness to projections (0-1 scale)
            stack_boost: Boost percentage for correlated players (default 15%)
            force_stack: Force a specific stack type ('PP1', 'Line1', etc.)

        Returns:
            List of DataFrames, each representing a lineup
        """
        df = player_pool.copy()
        df['norm_position'] = df['position'].apply(self._normalize_position)

        lineups = []
        used_lineup_hashes = set()

        for i in range(n_lineups * 3):  # Try more times to get unique lineups
            if len(lineups) >= n_lineups:
                break

            # Add randomness if specified
            if randomness > 0:
                noise = np.random.normal(1, randomness, len(df))
                df['adj_projection'] = df['projected_fpts'] * noise.clip(0.5, 1.5)
            else:
                df['adj_projection'] = df['projected_fpts']

            # Recalculate value with adjusted projections
            df['adj_value'] = df['adj_projection'] / (df['salary'] / 1000)

            lineup = self._build_lineup_with_stacking(
                df, max_from_team, stack_boost, force_stack
            )

            if lineup is not None:
                # Calculate stack bonus for the lineup
                if self.stack_builder:
                    lineup = self._add_stack_info(lineup)

                # Check for duplicate lineups
                lineup_hash = frozenset(lineup['name'].tolist())
                if lineup_hash not in used_lineup_hashes:
                    used_lineup_hashes.add(lineup_hash)
                    lineups.append(lineup)

        return lineups

    def _add_stack_info(self, lineup: pd.DataFrame) -> pd.DataFrame:
        """Add stacking information to lineup."""
        lineup = lineup.copy()
        stack_info = []
        players = lineup['name'].tolist()

        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                corr = self.stack_builder.get_correlation(p1, p2)
                if corr > 0:
                    stack_info.append(f"{p1.split()[-1]}-{p2.split()[-1]}:{corr:.0%}")

        lineup['stack_info'] = ', '.join(stack_info[:3]) if stack_info else ''
        return lineup

    def _build_lineup_with_stacking(self, df: pd.DataFrame,
                                     max_from_team: int,
                                     stack_boost: float,
                                     force_stack: str) -> Optional[pd.DataFrame]:
        """
        Build lineup with stacking preferences.

        Strategy:
        1. If stack_builder available, boost projections for correlated players
        2. Optionally force a specific stack (PP1, Line1)
        3. Use salary-aware selection
        """
        # If we have stacking data, boost correlated players
        if self.stack_builder and stack_boost > 0:
            df = self._apply_stack_boosts(df, stack_boost)

        # If forcing a specific stack, pre-select those players
        if force_stack and self.stack_builder:
            return self._build_forced_stack_lineup(df, max_from_team, force_stack)

        # Otherwise use salary-aware approach
        return self._build_lineup_with_salary_awareness(df, max_from_team)

    def _apply_stack_boosts(self, df: pd.DataFrame, boost_pct: float) -> pd.DataFrame:
        """Apply projection boosts based on correlation data."""
        df = df.copy()

        # Get PP1 players across all teams - they get the biggest boost
        pp1_players = set()
        line1_players = set()

        for team in df['team'].unique():
            stacks = self.stack_builder.get_best_stacks(team)
            for stack in stacks:
                if stack.get('type') == 'PP1':
                    for p in stack.get('players', []):
                        pp1_players.add(p.lower())
                elif stack.get('type') == 'Line1':
                    for p in stack.get('players', []):
                        line1_players.add(p.lower())

        # Apply boosts
        def get_boost(name):
            name_lower = name.lower()
            # Check PP1 (highest boost)
            for pp_name in pp1_players:
                if name_lower in pp_name or pp_name in name_lower:
                    return 1 + boost_pct
            # Check Line1 (moderate boost)
            for line_name in line1_players:
                if name_lower in line_name or line_name in name_lower:
                    return 1 + (boost_pct * 0.6)
            return 1.0

        df['stack_boost'] = df['name'].apply(get_boost)
        df['adj_projection'] = df['adj_projection'] * df['stack_boost']
        df['adj_value'] = df['adj_projection'] / (df['salary'] / 1000)

        return df

    def _build_forced_stack_lineup(self, df: pd.DataFrame,
                                    max_from_team: int,
                                    stack_type: str) -> Optional[pd.DataFrame]:
        """Build lineup forcing a specific stack."""
        from lines import find_player_match

        # Find the best stack of the requested type
        best_stack = None
        best_proj = 0

        for team in df['team'].unique():
            stacks = self.stack_builder.get_best_stacks(team, df)
            for stack in stacks:
                if stack.get('type') == stack_type:
                    proj = stack.get('projected_total', 0)
                    if proj > best_proj:
                        best_proj = proj
                        best_stack = stack

        if not best_stack:
            return self._build_lineup_with_salary_awareness(df, max_from_team)

        # Pre-select stack players
        forced_players = []
        for p in best_stack.get('players', []):
            match = find_player_match(p, df['name'].tolist())
            if match:
                forced_players.append(match)

        # Heavily boost forced players
        df = df.copy()
        df['adj_projection'] = df.apply(
            lambda r: r['adj_projection'] * 1.5 if r['name'] in forced_players else r['adj_projection'],
            axis=1
        )
        df['adj_value'] = df['adj_projection'] / (df['salary'] / 1000)

        return self._build_lineup_with_salary_awareness(df, max_from_team)

    def _build_lineup_with_salary_awareness(self, df: pd.DataFrame,
                                             max_from_team: int) -> Optional[pd.DataFrame]:
        """
        Build lineup with salary cap awareness.

        Strategy:
        1. Calculate target salary per position based on pool averages
        2. Select players balancing projection and value
        3. Fill required positions first, then UTIL
        """
        # Separate by position
        centers = df[df['norm_position'] == 'C'].copy()
        wings = df[df['norm_position'] == 'W'].copy()
        defense = df[df['norm_position'] == 'D'].copy()
        goalies = df[df['norm_position'] == 'G'].copy()
        skaters = df[df['norm_position'] != 'G'].copy()

        # Calculate average salary per position for targeting
        avg_salary = df['salary'].mean()

        # Target spending: leave ~$5500 avg per player = $49,500
        target_per_player = self.SALARY_CAP / 9

        lineup = []
        used_players = set()
        team_counts = {}
        remaining_salary = self.SALARY_CAP

        def can_add_player(player):
            if player['name'] in used_players:
                return False
            if player['salary'] > remaining_salary:
                return False
            team = player.get('team', 'UNK')
            if team_counts.get(team, 0) >= max_from_team:
                return False
            return True

        def add_player(player, slot):
            nonlocal remaining_salary
            player_dict = player.to_dict() if hasattr(player, 'to_dict') else dict(player)
            player_dict['roster_slot'] = slot
            lineup.append(player_dict)
            used_players.add(player['name'])
            team = player.get('team', 'UNK')
            team_counts[team] = team_counts.get(team, 0) + 1
            remaining_salary -= player['salary']

        def get_best_player(pool, remaining_spots, prefer_value=False):
            """Get best available player considering salary constraints."""
            available = pool[pool['name'].apply(lambda x: x not in used_players)]
            available = available[available['salary'] <= remaining_salary]
            available = available[available['team'].apply(
                lambda t: team_counts.get(t, 0) < max_from_team
            )]

            if available.empty:
                return None

            # Calculate how much salary we can afford per remaining spot
            salary_per_spot = remaining_salary / max(remaining_spots, 1)

            # If we're tight on salary, prioritize value players
            if salary_per_spot < target_per_player * 0.8 or prefer_value:
                # Prioritize value (projection per $1k salary)
                return available.nlargest(1, 'adj_value').iloc[0]
            else:
                # Prioritize raw projection
                return available.nlargest(1, 'adj_projection').iloc[0]

        # Fill positions strategically
        # Start with goalie (only 1 needed, less flexibility)
        remaining_spots = 9

        if not goalies.empty:
            best_g = get_best_player(goalies, remaining_spots)
            if best_g is not None:
                add_player(best_g, 'G')
                remaining_spots -= 1

        # Fill 2 Centers
        for i in range(2):
            best_c = get_best_player(centers, remaining_spots)
            if best_c is not None:
                add_player(best_c, f'C{i+1}')
                remaining_spots -= 1

        # Fill 3 Wings
        for i in range(3):
            best_w = get_best_player(wings, remaining_spots)
            if best_w is not None:
                add_player(best_w, f'W{i+1}')
                remaining_spots -= 1

        # Fill 2 Defense
        for i in range(2):
            best_d = get_best_player(defense, remaining_spots)
            if best_d is not None:
                add_player(best_d, f'D{i+1}')
                remaining_spots -= 1

        # Fill UTIL (best remaining skater)
        if remaining_spots > 0:
            best_util = get_best_player(skaters, remaining_spots, prefer_value=True)
            if best_util is not None:
                add_player(best_util, 'UTIL')
                remaining_spots -= 1

        if len(lineup) < 9:
            # Try again with more value-focused approach
            return self._build_value_lineup(df, max_from_team)

        lineup_df = pd.DataFrame(lineup)
        lineup_df['remaining_salary'] = remaining_salary

        return lineup_df

    def _build_value_lineup(self, df: pd.DataFrame, max_from_team: int) -> Optional[pd.DataFrame]:
        """
        Build lineup prioritizing value players to ensure salary compliance.
        """
        centers = df[df['norm_position'] == 'C'].sort_values('adj_value', ascending=False)
        wings = df[df['norm_position'] == 'W'].sort_values('adj_value', ascending=False)
        defense = df[df['norm_position'] == 'D'].sort_values('adj_value', ascending=False)
        goalies = df[df['norm_position'] == 'G'].sort_values('adj_value', ascending=False)
        skaters = df[df['norm_position'] != 'G'].sort_values('adj_value', ascending=False)

        lineup = []
        used_players = set()
        team_counts = {}
        remaining_salary = self.SALARY_CAP

        def can_add(player):
            if player['name'] in used_players:
                return False
            if player['salary'] > remaining_salary:
                return False
            team = player.get('team', 'UNK')
            if team_counts.get(team, 0) >= max_from_team:
                return False
            return True

        def add_player(player, slot):
            nonlocal remaining_salary
            player_dict = player.to_dict()
            player_dict['roster_slot'] = slot
            lineup.append(player_dict)
            used_players.add(player['name'])
            team = player.get('team', 'UNK')
            team_counts[team] = team_counts.get(team, 0) + 1
            remaining_salary -= player['salary']

        # Fill with value-first approach
        # Goalie
        for _, p in goalies.iterrows():
            if can_add(p):
                add_player(p, 'G')
                break

        # 2 Centers
        c_count = 0
        for _, p in centers.iterrows():
            if c_count >= 2:
                break
            if can_add(p):
                add_player(p, f'C{c_count+1}')
                c_count += 1

        # 3 Wings
        w_count = 0
        for _, p in wings.iterrows():
            if w_count >= 3:
                break
            if can_add(p):
                add_player(p, f'W{w_count+1}')
                w_count += 1

        # 2 Defense
        d_count = 0
        for _, p in defense.iterrows():
            if d_count >= 2:
                break
            if can_add(p):
                add_player(p, f'D{d_count+1}')
                d_count += 1

        # UTIL
        for _, p in skaters.iterrows():
            if len(lineup) >= 9:
                break
            if can_add(p):
                add_player(p, 'UTIL')
                break

        if len(lineup) < 9:
            return None

        lineup_df = pd.DataFrame(lineup)
        lineup_df['remaining_salary'] = remaining_salary

        return lineup_df

    def format_lineup_for_dk(self, lineup: pd.DataFrame) -> str:
        """Format lineup for display."""
        output = []
        total_salary = lineup['salary'].sum()
        total_proj = lineup['projected_fpts'].sum()

        output.append(f"Total Salary: ${total_salary:,} / $50,000")
        output.append(f"Projected Points: {total_proj:.1f}")
        output.append("")
        output.append(f"{'Slot':<6} {'Name':<25} {'Team':<5} {'Salary':<9} {'Proj':<7}")
        output.append("-" * 60)

        # Sort by roster slot
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


# Quick test
if __name__ == "__main__":
    from data_pipeline import NHLDataPipeline
    from projections import NHLProjectionModel
    from main import load_dk_salaries, merge_projections_with_salaries
    from datetime import datetime

    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(include_game_logs=False)

    model = NHLProjectionModel()
    today = datetime.now().strftime('%Y-%m-%d')
    projections = model.generate_projections(data, target_date=today)

    dk_salaries = load_dk_salaries('DKSalaries_1.22.26.csv')
    dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'LW', 'RW', 'D'])]
    dk_goalies = dk_salaries[dk_salaries['position'] == 'G']

    projections['goalies']['position'] = 'G'

    skaters_merged = merge_projections_with_salaries(projections['skaters'], dk_skaters, 'skater')
    goalies_merged = merge_projections_with_salaries(projections['goalies'], dk_goalies, 'goalie')

    player_pool = pd.concat([skaters_merged, goalies_merged], ignore_index=True)

    optimizer = NHLLineupOptimizer()
    lineups = optimizer.optimize_lineup(player_pool, n_lineups=1)

    if lineups:
        print("\n" + "=" * 60)
        print("OPTIMIZED LINEUP")
        print("=" * 60)
        print(optimizer.format_lineup_for_dk(lineups[0]))
    else:
        print("Failed to generate lineup")
