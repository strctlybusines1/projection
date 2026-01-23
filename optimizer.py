"""
DraftKings NHL Lineup Optimizer with GPP Stacking Support.

Based on analysis of winning GPP lineups:
- 68% of top 100 had team stacks (3-4 players)
- Winning lineup had 6-player stack + 2-player secondary stack
- Key pairs like Kaprizov+Zuccarello appeared in 59% of winners
- Goalie correlation with skaters is valuable in high-scoring games
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from config import (
    GPP_MIN_STACK_SIZE, GPP_MAX_FROM_TEAM, CASH_MAX_FROM_TEAM,
    PRIMARY_STACK_BOOST, SECONDARY_STACK_BOOST, LINEMATE_BOOST,
    GOALIE_CORRELATION_BOOST, HIGH_TOTAL_THRESHOLD,
    PREFERRED_PRIMARY_STACK_SIZE, PREFERRED_SECONDARY_STACK_SIZE
)


class NHLLineupOptimizer:
    """
    Optimizer for DraftKings NHL lineups with GPP-focused stacking.

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

        Args:
            player_pool: DataFrame with projections and salaries
            n_lineups: Number of lineups to generate
            mode: 'gpp' for tournaments, 'cash' for 50/50s and double-ups
            max_from_team: Maximum players from one team (auto-set by mode if None)
            min_teams: Minimum number of teams represented
            min_stack_size: Minimum stack size required (auto-set by mode if None)
            randomness: Add randomness to projections (0-1 scale)
            stack_teams: List of teams to prioritize for stacking
            secondary_stack_team: Specific team for secondary 2-3 player stack
            force_players: List of player names to force into lineup
            exclude_players: List of player names to exclude

        Returns:
            List of DataFrames, each representing a lineup
        """
        df = player_pool.copy()
        # Use DK position for roster eligibility if available, otherwise fall back to projection position
        # dk_pos is the actual position (C, LW, RW, D, G), dk_position is roster eligibility (W/UTIL, etc.)
        if 'dk_pos' in df.columns:
            pos_col = 'dk_pos'
        elif 'dk_position' in df.columns:
            pos_col = 'dk_position'
        else:
            pos_col = 'position'
        df['norm_position'] = df[pos_col].apply(self._normalize_position)

        # Set mode-specific defaults
        if max_from_team is None:
            max_from_team = GPP_MAX_FROM_TEAM if mode == 'gpp' else CASH_MAX_FROM_TEAM

        if min_stack_size is None:
            min_stack_size = GPP_MIN_STACK_SIZE if mode == 'gpp' else 0

        # Apply exclusions
        if exclude_players:
            exclude_lower = [p.lower() for p in exclude_players]
            df = df[~df['name'].str.lower().isin(exclude_lower)]

        lineups = []
        used_lineup_hashes = set()
        attempts = 0
        max_attempts = n_lineups * 10

        while len(lineups) < n_lineups and attempts < max_attempts:
            attempts += 1

            # Add randomness if specified
            if randomness > 0:
                noise = np.random.normal(1, randomness, len(df))
                df['adj_projection'] = df['projected_fpts'] * noise.clip(0.7, 1.3)
            else:
                df['adj_projection'] = df['projected_fpts']

            # Recalculate value with adjusted projections
            df['adj_value'] = df['adj_projection'] / (df['salary'] / 1000)

            # Build lineup based on mode
            if mode == 'gpp':
                lineup = self._build_gpp_lineup(
                    df, max_from_team, min_stack_size,
                    stack_teams, secondary_stack_team, force_players
                )
            else:
                lineup = self._build_cash_lineup(df, max_from_team, force_players)

            if lineup is not None and len(lineup) == 9:
                # Validate minimum stack requirement
                if min_stack_size > 0:
                    team_counts = lineup['team'].value_counts()
                    max_stack = team_counts.max() if len(team_counts) > 0 else 0
                    if max_stack < min_stack_size:
                        continue  # Reject lineup without sufficient stack

                # Validate minimum teams
                if lineup['team'].nunique() < min_teams:
                    continue

                # Check for duplicate lineups
                lineup_hash = frozenset(lineup['name'].tolist())
                if lineup_hash not in used_lineup_hashes:
                    used_lineup_hashes.add(lineup_hash)

                    # Add stack analysis info
                    lineup = self._add_stack_analysis(lineup)
                    lineups.append(lineup)

        return lineups

    def _build_gpp_lineup(self, df: pd.DataFrame,
                          max_from_team: int,
                          min_stack_size: int,
                          stack_teams: List[str] = None,
                          secondary_stack_team: str = None,
                          force_players: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Build GPP lineup with enforced stacking strategy.

        Strategy based on winning lineup analysis:
        1. Select primary stack team (highest projected or specified)
        2. Pick 4-5 correlated players from primary stack
        3. Add secondary stack of 2-3 players from another game
        4. Fill remaining spots with high-value plays
        5. Correlate goalie with skater stack when possible
        """
        # Identify best stack teams based on projections
        team_proj = df.groupby('team')['adj_projection'].sum().sort_values(ascending=False)

        # Select primary stack team
        if stack_teams and len(stack_teams) > 0:
            primary_team = stack_teams[0]
        else:
            # Choose from top 3 teams with some randomness
            top_teams = team_proj.head(5).index.tolist()
            weights = [0.4, 0.25, 0.15, 0.12, 0.08][:len(top_teams)]
            primary_team = np.random.choice(top_teams, p=weights)

        # Select secondary stack team
        if secondary_stack_team:
            secondary_team = secondary_stack_team
        elif stack_teams and len(stack_teams) > 1:
            secondary_team = stack_teams[1]
        else:
            # Choose different team from remaining top teams
            remaining_teams = [t for t in team_proj.head(6).index if t != primary_team]
            if remaining_teams:
                secondary_team = np.random.choice(remaining_teams[:3])
            else:
                secondary_team = None

        # Build the lineup with stacking
        lineup = []
        used_players = set()
        team_counts = defaultdict(int)
        remaining_salary = self.SALARY_CAP

        def can_add_player(player, check_team_limit=True):
            if player['name'] in used_players:
                return False
            if player['salary'] > remaining_salary:
                return False
            if check_team_limit:
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

        # Force players if specified
        if force_players:
            for forced_name in force_players:
                forced_match = df[df['name'].str.lower() == forced_name.lower()]
                if not forced_match.empty:
                    player = forced_match.iloc[0]
                    pos = player['norm_position']
                    if can_add_player(player, check_team_limit=False):
                        slot = self._get_available_slot(pos, lineup)
                        if slot:
                            add_player(player, slot)

        # Step 1: Select goalie (prefer from primary stack team for correlation)
        goalies = df[df['norm_position'] == 'G'].copy()

        # Filter to confirmed starters only
        goalies = self._filter_confirmed_goalies(goalies)

        # Boost primary team goalie for correlation
        goalies['stack_adj'] = goalies.apply(
            lambda r: r['adj_projection'] * (1 + GOALIE_CORRELATION_BOOST)
            if r['team'] == primary_team else r['adj_projection'],
            axis=1
        )
        goalies = goalies.sort_values('stack_adj', ascending=False)

        for _, g in goalies.iterrows():
            if can_add_player(g):
                add_player(g, 'G')
                break

        # Get the goalie's team for correlation
        goalie_team = lineup[0]['team'] if lineup else None

        # Step 2: Build primary stack (target 4-5 players)
        primary_players = df[df['team'] == primary_team].copy()
        primary_skaters = primary_players[primary_players['norm_position'] != 'G']

        # Apply linemate boosts if we have stack builder
        if self.stack_builder:
            primary_skaters = self._apply_linemate_boosts(primary_skaters, primary_team)

        primary_skaters = primary_skaters.sort_values('adj_projection', ascending=False)

        # Add primary stack players (target PREFERRED_PRIMARY_STACK_SIZE)
        primary_target = min(PREFERRED_PRIMARY_STACK_SIZE, max_from_team - team_counts.get(primary_team, 0))
        primary_added = 0

        for _, player in primary_skaters.iterrows():
            if primary_added >= primary_target:
                break
            if can_add_player(player):
                pos = player['norm_position']
                slot = self._get_available_slot(pos, lineup)
                if slot:
                    add_player(player, slot)
                    primary_added += 1

        # Step 3: Build secondary stack (target 2-3 players)
        if secondary_team:
            secondary_players = df[df['team'] == secondary_team].copy()
            secondary_skaters = secondary_players[secondary_players['norm_position'] != 'G']

            if self.stack_builder:
                secondary_skaters = self._apply_linemate_boosts(secondary_skaters, secondary_team)

            secondary_skaters = secondary_skaters.sort_values('adj_projection', ascending=False)

            secondary_target = PREFERRED_SECONDARY_STACK_SIZE
            secondary_added = 0

            for _, player in secondary_skaters.iterrows():
                if secondary_added >= secondary_target:
                    break
                if can_add_player(player):
                    pos = player['norm_position']
                    slot = self._get_available_slot(pos, lineup)
                    if slot:
                        add_player(player, slot)
                        secondary_added += 1

        # Step 4: Fill remaining positions with best available
        remaining_spots = 9 - len(lineup)

        if remaining_spots > 0:
            # Get all skaters not yet used, prioritize by value for remaining spots
            all_skaters = df[df['norm_position'] != 'G'].copy()
            all_skaters = all_skaters[~all_skaters['name'].isin(used_players)]
            all_skaters = all_skaters[all_skaters['salary'] <= remaining_salary]

            # Boost players from goalie's team for correlation
            if goalie_team:
                all_skaters['corr_adj'] = all_skaters.apply(
                    lambda r: r['adj_value'] * (1 + GOALIE_CORRELATION_BOOST * 0.5)
                    if r['team'] == goalie_team else r['adj_value'],
                    axis=1
                )
            else:
                all_skaters['corr_adj'] = all_skaters['adj_value']

            all_skaters = all_skaters.sort_values('corr_adj', ascending=False)

            for _, player in all_skaters.iterrows():
                if len(lineup) >= 9:
                    break
                if can_add_player(player):
                    pos = player['norm_position']
                    slot = self._get_available_slot(pos, lineup)
                    if slot:
                        add_player(player, slot)

        if len(lineup) < 9:
            return None

        return pd.DataFrame(lineup)

    def _build_cash_lineup(self, df: pd.DataFrame,
                           max_from_team: int,
                           force_players: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Build cash game lineup prioritizing floor and consistency.

        Cash games prioritize:
        - High floor players (consistent producers)
        - Value plays to enable studs
        - Less emphasis on correlation
        """
        lineup = []
        used_players = set()
        team_counts = defaultdict(int)
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

        # Force players if specified
        if force_players:
            for forced_name in force_players:
                forced_match = df[df['name'].str.lower() == forced_name.lower()]
                if not forced_match.empty:
                    player = forced_match.iloc[0]
                    pos = player['norm_position']
                    slot = self._get_available_slot(pos, lineup)
                    if slot and can_add_player(player):
                        add_player(player, slot)

        # Separate by position
        centers = df[df['norm_position'] == 'C'].sort_values('adj_value', ascending=False)
        wings = df[df['norm_position'] == 'W'].sort_values('adj_value', ascending=False)
        defense = df[df['norm_position'] == 'D'].sort_values('adj_value', ascending=False)
        goalies = df[df['norm_position'] == 'G'].sort_values('adj_projection', ascending=False)
        goalies = self._filter_confirmed_goalies(goalies)  # Filter to confirmed starters
        skaters = df[df['norm_position'] != 'G'].sort_values('adj_value', ascending=False)

        # Fill goalie first
        for _, g in goalies.iterrows():
            if 'G' not in [l.get('roster_slot') for l in lineup]:
                if can_add_player(g):
                    add_player(g, 'G')
                    break

        # Fill required positions
        positions_needed = self._get_positions_needed(lineup)

        for pos, count in positions_needed.items():
            if pos == 'G':
                continue

            pool = {'C': centers, 'W': wings, 'D': defense}.get(pos)
            if pool is None:
                continue

            filled = sum(1 for l in lineup if l.get('roster_slot', '').startswith(pos))
            for _, player in pool.iterrows():
                if filled >= count:
                    break
                if can_add_player(player):
                    slot = self._get_available_slot(pos, lineup)
                    if slot:
                        add_player(player, slot)
                        filled += 1

        # Fill UTIL
        if len(lineup) < 9:
            for _, player in skaters.iterrows():
                if len(lineup) >= 9:
                    break
                if can_add_player(player):
                    slot = self._get_available_slot(player['norm_position'], lineup)
                    if slot:
                        add_player(player, slot)

        if len(lineup) < 9:
            return None

        return pd.DataFrame(lineup)

    def _filter_confirmed_goalies(self, goalies: pd.DataFrame) -> pd.DataFrame:
        """Filter goalies to only confirmed starters from lines data."""
        if not self.stack_builder:
            return goalies

        confirmed = self.stack_builder.get_all_starting_goalies()
        if not confirmed:
            return goalies

        # Build list of confirmed goalie names
        confirmed_names = list(confirmed.values())

        # Filter to confirmed starters using fuzzy matching
        from lines import fuzzy_match

        def is_confirmed(name):
            for confirmed_name in confirmed_names:
                if fuzzy_match(name, confirmed_name):
                    return True
            return False

        mask = goalies['name'].apply(is_confirmed)
        filtered = goalies[mask]

        # If no matches found, return original (fallback)
        if filtered.empty:
            print("Warning: No confirmed goalies matched in player pool, using all goalies")
            return goalies

        return filtered

    def _apply_linemate_boosts(self, df: pd.DataFrame, team: str) -> pd.DataFrame:
        """Apply projection boosts based on linemate correlations."""
        if not self.stack_builder:
            return df

        df = df.copy()

        # Get team line data
        team_data = self.stack_builder.lines_data.get(team, {})
        if not team_data or 'error' in team_data:
            return df

        # Identify PP1 and Line1 players (highest correlation)
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
            # Check if player is on PP1 (highest boost)
            for pp_name in pp1_players:
                if name_lower in pp_name or pp_name in name_lower:
                    return 1 + LINEMATE_BOOST * 1.2  # Extra boost for PP1
            # Check if player is on Line1
            for line_name in line1_players:
                if name_lower in line_name or line_name in name_lower:
                    return 1 + LINEMATE_BOOST
            return 1.0

        df['linemate_boost'] = df['name'].apply(get_linemate_boost)
        df['adj_projection'] = df['adj_projection'] * df['linemate_boost']

        return df

    def _get_available_slot(self, position: str, lineup: List[Dict]) -> Optional[str]:
        """Get the next available roster slot for a position."""
        filled_slots = [p.get('roster_slot') for p in lineup]

        # Position requirements
        slots = {
            'C': ['C1', 'C2', 'UTIL'],
            'W': ['W1', 'W2', 'W3', 'UTIL'],
            'D': ['D1', 'D2', 'UTIL'],
            'G': ['G']
        }

        available = slots.get(position, [])
        for slot in available:
            if slot not in filled_slots:
                # For UTIL, check if we still need it
                if slot == 'UTIL':
                    # Count non-UTIL skater slots filled
                    c_filled = sum(1 for s in filled_slots if s.startswith('C') and s != 'UTIL')
                    w_filled = sum(1 for s in filled_slots if s.startswith('W'))
                    d_filled = sum(1 for s in filled_slots if s.startswith('D'))

                    # Only use UTIL if required positions are filled
                    if c_filled >= 2 and w_filled >= 3 and d_filled >= 2:
                        return slot
                    # Or if this position's slots are full
                    elif position == 'C' and c_filled >= 2:
                        return slot
                    elif position == 'W' and w_filled >= 3:
                        return slot
                    elif position == 'D' and d_filled >= 2:
                        return slot
                else:
                    return slot

        return None

    def _get_positions_needed(self, lineup: List[Dict]) -> Dict[str, int]:
        """Get remaining positions needed."""
        filled = defaultdict(int)
        for p in lineup:
            slot = p.get('roster_slot', '')
            if slot.startswith('C'):
                filled['C'] += 1
            elif slot.startswith('W'):
                filled['W'] += 1
            elif slot.startswith('D'):
                filled['D'] += 1
            elif slot == 'G':
                filled['G'] += 1

        return {
            'C': max(0, 2 - filled['C']),
            'W': max(0, 3 - filled['W']),
            'D': max(0, 2 - filled['D']),
            'G': max(0, 1 - filled['G']),
        }

    def _add_stack_analysis(self, lineup: pd.DataFrame) -> pd.DataFrame:
        """Add stacking analysis to lineup."""
        lineup = lineup.copy()

        # Count players per team
        team_counts = lineup['team'].value_counts()

        # Identify stacks
        stacks = []
        for team, count in team_counts.items():
            if count >= 2:
                team_players = lineup[lineup['team'] == team]['name'].tolist()
                stacks.append(f"{team}({count}): {', '.join([p.split()[-1] for p in team_players])}")

        lineup['stack_info'] = '; '.join(stacks) if stacks else 'No stacks'

        # Add correlation info if available
        if self.stack_builder:
            corr_info = []
            players = lineup['name'].tolist()
            for i, p1 in enumerate(players):
                for p2 in players[i+1:]:
                    corr = self.stack_builder.get_correlation(p1, p2)
                    if corr > 0.5:  # Only show strong correlations
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

        # Show stack info
        team_counts = lineup['team'].value_counts()
        stacks = [f"{team}({count})" for team, count in team_counts.items() if count >= 2]
        if stacks:
            output.append(f"Stacks: {' + '.join(stacks)}")

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

    def generate_gpp_lineups(self, player_pool: pd.DataFrame,
                             n_lineups: int = 20,
                             diversity_factor: float = 0.15) -> List[pd.DataFrame]:
        """
        Generate diverse GPP lineup set with varied stacking strategies.

        Args:
            player_pool: DataFrame with projections and salaries
            n_lineups: Number of lineups to generate
            diversity_factor: Randomness to ensure lineup diversity (0-1)

        Returns:
            List of unique lineups with different stack combinations
        """
        all_lineups = []
        teams = player_pool['team'].unique().tolist()

        # Generate lineups with different primary stacks
        lineups_per_strategy = max(1, n_lineups // len(teams))

        for primary_team in teams:
            other_teams = [t for t in teams if t != primary_team]

            for secondary_team in other_teams[:2]:  # Try 2 secondary stacks per primary
                lineups = self.optimize_lineup(
                    player_pool,
                    n_lineups=lineups_per_strategy,
                    mode='gpp',
                    randomness=diversity_factor,
                    stack_teams=[primary_team],
                    secondary_stack_team=secondary_team
                )
                all_lineups.extend(lineups)

                if len(all_lineups) >= n_lineups:
                    break

            if len(all_lineups) >= n_lineups:
                break

        # Deduplicate and return requested number
        seen = set()
        unique_lineups = []
        for lineup in all_lineups:
            lineup_hash = frozenset(lineup['name'].tolist())
            if lineup_hash not in seen:
                seen.add(lineup_hash)
                unique_lineups.append(lineup)

        return unique_lineups[:n_lineups]


# Quick test
if __name__ == "__main__":
    from data_pipeline import NHLDataPipeline
    from projections import NHLProjectionModel
    from main import load_dk_salaries, merge_projections_with_salaries
    from lines import LinesScraper, StackBuilder
    from datetime import datetime

    print("Testing GPP Optimizer...")

    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(include_game_logs=False)

    model = NHLProjectionModel()
    today = datetime.now().strftime('%Y-%m-%d')
    projections = model.generate_projections(data, target_date=today)

    # Fetch lines for all teams playing today
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

    # Show confirmed goalies
    confirmed_goalies = stack_builder.get_all_starting_goalies()
    print(f"\nConfirmed starters: {len(confirmed_goalies)} goalies")
    for team, goalie in sorted(confirmed_goalies.items()):
        print(f"  {team}: {goalie}")

    # Load salaries
    import glob
    salary_files = glob.glob('DKSalaries*.csv')
    if salary_files:
        dk_salaries = load_dk_salaries(salary_files[0])
        dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'LW', 'RW', 'D'])]
        dk_goalies = dk_salaries[dk_salaries['position'] == 'G']

        projections['goalies']['position'] = 'G'

        skaters_merged = merge_projections_with_salaries(projections['skaters'], dk_skaters, 'skater')
        goalies_merged = merge_projections_with_salaries(projections['goalies'], dk_goalies, 'goalie')

        player_pool = pd.concat([skaters_merged, goalies_merged], ignore_index=True)

        # Create optimizer with stack builder (includes confirmed goalie data)
        optimizer = NHLLineupOptimizer(stack_builder=stack_builder)

        print("\n" + "=" * 60)
        print("GPP LINEUP (with stacking)")
        print("=" * 60)
        gpp_lineups = optimizer.optimize_lineup(player_pool, n_lineups=1, mode='gpp')
        if gpp_lineups:
            print(optimizer.format_lineup_for_dk(gpp_lineups[0]))

        print("\n" + "=" * 60)
        print("CASH LINEUP (balanced)")
        print("=" * 60)
        cash_lineups = optimizer.optimize_lineup(player_pool, n_lineups=1, mode='cash')
        if cash_lineups:
            print(optimizer.format_lineup_for_dk(cash_lineups[0]))
    else:
        print("No salary file found. Please add DKSalaries*.csv file.")
