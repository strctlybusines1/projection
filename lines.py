"""
Line combinations scraper from DailyFaceoff.com

Fetches:
- Forward lines (1-4)
- Defense pairings (1-3)
- Power play units (PP1, PP2)
- Confirmed goalie starters
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
import re
import json
import time
from difflib import SequenceMatcher

from config import NHL_TEAMS


def fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """Check if two names are similar enough to be the same person."""
    # Normalize names
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Exact match
    if n1 == n2:
        return True

    # Check similarity ratio
    ratio = SequenceMatcher(None, n1, n2).ratio()
    if ratio >= threshold:
        return True

    # Check if one contains the other (for nicknames)
    if n1 in n2 or n2 in n1:
        return True

    # Check last name match
    last1 = n1.split()[-1] if n1.split() else n1
    last2 = n2.split()[-1] if n2.split() else n2
    if last1 == last2 and len(last1) > 3:
        return True

    return False


def find_player_match(target_name: str, player_list: List[str], threshold: float = 0.85) -> Optional[str]:
    """Find the best matching player name from a list."""
    target_lower = target_name.lower().strip()

    # First try exact match
    for name in player_list:
        if name.lower().strip() == target_lower:
            return name

    # Then try fuzzy match
    best_match = None
    best_ratio = 0

    for name in player_list:
        ratio = SequenceMatcher(None, target_lower, name.lower().strip()).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = name

    return best_match


class LinesScraper:
    """Scrape line combinations from DailyFaceoff using embedded JSON data."""

    BASE_URL = "https://www.dailyfaceoff.com/teams"

    # Map team abbreviations to DailyFaceoff URL slugs
    TEAM_SLUGS = {
        'ANA': 'anaheim-ducks',
        'BOS': 'boston-bruins',
        'BUF': 'buffalo-sabres',
        'CGY': 'calgary-flames',
        'CAR': 'carolina-hurricanes',
        'CHI': 'chicago-blackhawks',
        'COL': 'colorado-avalanche',
        'CBJ': 'columbus-blue-jackets',
        'DAL': 'dallas-stars',
        'DET': 'detroit-red-wings',
        'EDM': 'edmonton-oilers',
        'FLA': 'florida-panthers',
        'LAK': 'los-angeles-kings',
        'MIN': 'minnesota-wild',
        'MTL': 'montreal-canadiens',
        'NSH': 'nashville-predators',
        'NJD': 'new-jersey-devils',
        'NYI': 'new-york-islanders',
        'NYR': 'new-york-rangers',
        'OTT': 'ottawa-senators',
        'PHI': 'philadelphia-flyers',
        'PIT': 'pittsburgh-penguins',
        'SJS': 'san-jose-sharks',
        'SEA': 'seattle-kraken',
        'STL': 'st-louis-blues',
        'TBL': 'tampa-bay-lightning',
        'TOR': 'toronto-maple-leafs',
        'UTA': 'utah-hockey-club',
        'VAN': 'vancouver-canucks',
        'VGK': 'vegas-golden-knights',
        'WSH': 'washington-capitals',
        'WPG': 'winnipeg-jets',
        'ARI': 'utah-hockey-club',
    }

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def _fetch_json_data(self, team: str) -> Optional[Dict]:
        """Fetch and extract JSON data from DailyFaceoff page."""
        slug = self.TEAM_SLUGS.get(team.upper())
        if not slug:
            print(f"Unknown team: {team}")
            return None

        url = f"{self.BASE_URL}/{slug}/line-combinations"

        try:
            time.sleep(self.rate_limit)
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # Extract __NEXT_DATA__ JSON
            match = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                response.text,
                re.DOTALL
            )

            if match:
                data = json.loads(match.group(1))
                return data.get('props', {}).get('pageProps', {}).get('combinations', {})

        except Exception as e:
            print(f"Error fetching {team}: {e}")

        return None

    def get_team_lines(self, team: str) -> Dict:
        """
        Get line combinations for a team.

        Returns dict with:
        - forward_lines: List of lines with LW, C, RW
        - defense_pairs: List of pairs with LD, RD
        - pp_units: List of PP units with all players
        - pk_units: List of PK units
        - starting_goalie: Confirmed starter
        - players: Full player data for additional info
        """
        data = self._fetch_json_data(team)

        if not data:
            return {'team': team, 'error': 'Failed to fetch data'}

        players = data.get('players', [])

        result = {
            'team': team,
            'team_name': data.get('teamName', ''),
            'updated_at': data.get('updatedAt', ''),
            'forward_lines': [],
            'defense_pairs': [],
            'pp_units': [],
            'pk_units': [],
            'starting_goalie': None,
            'backup_goalie': None,
            'all_players': [],
        }

        # Group players by their line assignment
        groups = {}
        for player in players:
            group_id = player.get('groupIdentifier', '')
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(player)

            # Store all players for reference
            result['all_players'].append({
                'name': player.get('name'),
                'position': player.get('positionIdentifier'),
                'group': group_id,
                'injury': player.get('injuryStatus'),
                'season_stats': player.get('season', {}),
                'last5': player.get('last5', {}),
            })

        # Parse forward lines (f1, f2, f3, f4)
        for line_num in range(1, 5):
            group_id = f'f{line_num}'
            if group_id in groups:
                line_players = groups[group_id]
                line_data = {'line': line_num, 'LW': None, 'C': None, 'RW': None, 'players': []}

                for p in line_players:
                    pos = p.get('positionIdentifier', '').lower()
                    name = p.get('name')
                    line_data['players'].append(name)

                    if pos == 'lw':
                        line_data['LW'] = name
                    elif pos == 'c':
                        line_data['C'] = name
                    elif pos == 'rw':
                        line_data['RW'] = name

                result['forward_lines'].append(line_data)

        # Parse defense pairs (d1, d2, d3)
        for pair_num in range(1, 4):
            group_id = f'd{pair_num}'
            if group_id in groups:
                pair_players = groups[group_id]
                pair_data = {'pair': pair_num, 'LD': None, 'RD': None, 'players': []}

                for p in pair_players:
                    pos = p.get('positionIdentifier', '').lower()
                    name = p.get('name')
                    pair_data['players'].append(name)

                    if pos == 'ld':
                        pair_data['LD'] = name
                    elif pos == 'rd':
                        pair_data['RD'] = name

                result['defense_pairs'].append(pair_data)

        # Parse power play units (pp1, pp2)
        for pp_num in range(1, 3):
            group_id = f'pp{pp_num}'
            if group_id in groups:
                pp_players = groups[group_id]
                pp_data = {
                    'unit': pp_num,
                    'players': [p.get('name') for p in pp_players],
                }
                result['pp_units'].append(pp_data)

        # Parse penalty kill units (pk1, pk2)
        for pk_num in range(1, 3):
            group_id = f'pk{pk_num}'
            if group_id in groups:
                pk_players = groups[group_id]
                pk_data = {
                    'unit': pk_num,
                    'players': [p.get('name') for p in pk_players],
                }
                result['pk_units'].append(pk_data)

        # Parse goalies
        if 'g' in groups:
            goalies = groups['g']
            if len(goalies) >= 1:
                result['starting_goalie'] = goalies[0].get('name')
            if len(goalies) >= 2:
                result['backup_goalie'] = goalies[1].get('name')

        return result

    def get_multiple_teams(self, teams: List[str]) -> Dict[str, Dict]:
        """Get line combinations for multiple teams."""
        all_lines = {}
        for team in teams:
            print(f"Fetching lines for {team}...")
            all_lines[team] = self.get_team_lines(team)
        return all_lines

    def get_slate_lines(self, schedule_df: pd.DataFrame) -> Dict[str, Dict]:
        """Get line combinations for all teams on a slate."""
        teams = set()
        teams.update(schedule_df['home_team'].tolist())
        teams.update(schedule_df['away_team'].tolist())
        return self.get_multiple_teams(list(teams))


class StackBuilder:
    """Build stacking recommendations from line data."""

    def __init__(self, lines_data: Dict[str, Dict]):
        self.lines_data = lines_data
        self.correlations = self._build_correlation_matrix()

    def _build_correlation_matrix(self) -> List[Dict]:
        """Build player correlation data from line combinations."""
        correlations = []

        for team, data in self.lines_data.items():
            if not data or 'error' in data:
                continue

            # Forward line correlations
            for line in data.get('forward_lines', []):
                line_num = line.get('line', 1)
                players = line.get('players', [])
                corr_value = 0.85 if line_num == 1 else 0.70 if line_num == 2 else 0.55

                for i, p1 in enumerate(players):
                    for p2 in players[i + 1:]:
                        if p1 and p2:
                            correlations.append({
                                'player1': p1,
                                'player2': p2,
                                'team': team,
                                'correlation': corr_value,
                                'stack_type': f'line{line_num}',
                            })

            # Defense pair correlations
            for pair in data.get('defense_pairs', []):
                pair_num = pair.get('pair', 1)
                players = pair.get('players', [])
                corr_value = 0.50 if pair_num == 1 else 0.40

                for i, p1 in enumerate(players):
                    for p2 in players[i + 1:]:
                        if p1 and p2:
                            correlations.append({
                                'player1': p1,
                                'player2': p2,
                                'team': team,
                                'correlation': corr_value,
                                'stack_type': f'defense{pair_num}',
                            })

            # Power play correlations (highest)
            for pp in data.get('pp_units', []):
                unit_num = pp.get('unit', 1)
                players = pp.get('players', [])
                corr_value = 0.95 if unit_num == 1 else 0.75

                for i, p1 in enumerate(players):
                    for p2 in players[i + 1:]:
                        if p1 and p2:
                            correlations.append({
                                'player1': p1,
                                'player2': p2,
                                'team': team,
                                'correlation': corr_value,
                                'stack_type': f'pp{unit_num}',
                            })

        return correlations

    def get_correlation(self, player1: str, player2: str) -> float:
        """Get correlation between two players using fuzzy matching."""
        for corr in self.correlations:
            p1_match = fuzzy_match(player1, corr['player1']) or fuzzy_match(player1, corr['player2'])
            p2_match = fuzzy_match(player2, corr['player1']) or fuzzy_match(player2, corr['player2'])

            if p1_match and p2_match and corr['player1'] != corr['player2']:
                return corr['correlation']

        return 0.0

    def get_linemates(self, player: str) -> List[Tuple[str, float, str]]:
        """Get all linemates for a player with correlation and stack type."""
        linemates = []

        for corr in self.correlations:
            if fuzzy_match(player, corr['player1']):
                linemates.append((corr['player2'], corr['correlation'], corr['stack_type']))
            elif fuzzy_match(player, corr['player2']):
                linemates.append((corr['player1'], corr['correlation'], corr['stack_type']))

        return sorted(linemates, key=lambda x: x[1], reverse=True)

    def get_best_stacks(self, team: str, projections_df: pd.DataFrame = None) -> List[Dict]:
        """Get best stacking options for a team."""
        data = self.lines_data.get(team, {})
        if not data or 'error' in data:
            return []

        stacks = []

        # PP1 stack (highest correlation)
        for pp in data.get('pp_units', []):
            if pp.get('unit') == 1:
                players = pp.get('players', [])
                stack = {
                    'type': 'PP1',
                    'team': team,
                    'players': players,
                    'correlation': 0.95,
                }

                if projections_df is not None and not projections_df.empty:
                    # Find matching players in projections
                    matched = []
                    for p in players:
                        match = find_player_match(p, projections_df['name'].tolist())
                        if match:
                            matched.append(match)

                    if matched:
                        proj_sum = projections_df[projections_df['name'].isin(matched)]['projected_fpts'].sum()
                        stack['projected_total'] = proj_sum
                        stack['matched_players'] = matched

                stacks.append(stack)

        # Line 1 stack
        for line in data.get('forward_lines', []):
            if line.get('line') == 1:
                players = line.get('players', [])
                stack = {
                    'type': 'Line1',
                    'team': team,
                    'players': players,
                    'correlation': 0.85,
                }

                if projections_df is not None and not projections_df.empty:
                    matched = []
                    for p in players:
                        match = find_player_match(p, projections_df['name'].tolist())
                        if match:
                            matched.append(match)

                    if matched:
                        proj_sum = projections_df[projections_df['name'].isin(matched)]['projected_fpts'].sum()
                        stack['projected_total'] = proj_sum
                        stack['matched_players'] = matched

                stacks.append(stack)

        # Line1+D1 composite stack (3F from Line1 + 2D from D-pair 1 = 5 players)
        line1_data = None
        d1_data = None
        for line in data.get('forward_lines', []):
            if line.get('line') == 1:
                line1_data = line
                break
        for pair in data.get('defense_pairs', []):
            if pair.get('pair') == 1:
                d1_data = pair
                break

        if line1_data and d1_data:
            combo_players = line1_data.get('players', []) + d1_data.get('players', [])
            stack = {
                'type': 'Line1+D1',
                'team': team,
                'players': combo_players,
                'correlation': 0.75,  # Blended: Line1 0.85 + D1-Line1 cross ~0.50
            }

            if projections_df is not None and not projections_df.empty:
                matched = []
                for p in combo_players:
                    match = find_player_match(p, projections_df['name'].tolist())
                    if match:
                        matched.append(match)

                if matched:
                    proj_sum = projections_df[projections_df['name'].isin(matched)]['projected_fpts'].sum()
                    stack['projected_total'] = proj_sum
                    stack['matched_players'] = matched

            stacks.append(stack)

        # Line 2 stack
        for line in data.get('forward_lines', []):
            if line.get('line') == 2:
                players = line.get('players', [])
                stack = {
                    'type': 'Line2',
                    'team': team,
                    'players': players,
                    'correlation': 0.70,
                }

                if projections_df is not None and not projections_df.empty:
                    matched = []
                    for p in players:
                        match = find_player_match(p, projections_df['name'].tolist())
                        if match:
                            matched.append(match)

                    if matched:
                        proj_sum = projections_df[projections_df['name'].isin(matched)]['projected_fpts'].sum()
                        stack['projected_total'] = proj_sum
                        stack['matched_players'] = matched

                stacks.append(stack)

        return stacks

    def get_starting_goalie(self, team: str) -> Optional[str]:
        """Get confirmed starting goalie for a team."""
        data = self.lines_data.get(team, {})
        return data.get('starting_goalie')

    def get_all_starting_goalies(self) -> Dict[str, str]:
        """Get all confirmed starting goalies."""
        goalies = {}
        for team, data in self.lines_data.items():
            if data and 'starting_goalie' in data and data['starting_goalie']:
                goalies[team] = data['starting_goalie']
        return goalies

    def get_update_timestamps(self) -> Dict[str, str]:
        """Get last update timestamps for all teams."""
        timestamps = {}
        for team, data in self.lines_data.items():
            if data and 'updated_at' in data:
                timestamps[team] = data['updated_at']
        return timestamps

    def get_oldest_update(self) -> Tuple[str, str]:
        """Get the oldest update timestamp across all teams."""
        timestamps = self.get_update_timestamps()
        if not timestamps:
            return None, None

        oldest_team = min(timestamps, key=lambda t: timestamps[t] if timestamps[t] else '9999')
        return oldest_team, timestamps.get(oldest_team, '')


def print_team_lines(lines_data: Dict):
    """Pretty print team line combinations."""
    print(f"\n{'=' * 60}")
    print(f" {lines_data.get('team_name', lines_data.get('team', 'Unknown'))} Line Combinations")
    print(f"{'=' * 60}")

    print("\nFORWARD LINES:")
    for line in lines_data.get('forward_lines', []):
        print(f"  Line {line['line']}: {line.get('LW', '?')} - {line.get('C', '?')} - {line.get('RW', '?')}")

    print("\nDEFENSE PAIRS:")
    for pair in lines_data.get('defense_pairs', []):
        print(f"  Pair {pair['pair']}: {pair.get('LD', '?')} - {pair.get('RD', '?')}")

    print("\nPOWER PLAY UNITS:")
    for pp in lines_data.get('pp_units', []):
        print(f"  PP{pp['unit']}: {', '.join(pp.get('players', []))}")

    print("\nGOALIES:")
    print(f"  Starter: {lines_data.get('starting_goalie', 'TBD')}")
    print(f"  Backup: {lines_data.get('backup_goalie', 'TBD')}")


# Quick test
if __name__ == "__main__":
    from data_pipeline import NHLDataPipeline
    from datetime import datetime

    scraper = LinesScraper()
    pipeline = NHLDataPipeline()

    # Get today's schedule
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching schedule for {today}...")
    schedule = pipeline.fetch_schedule(today)

    if schedule.empty:
        print("No games scheduled for today.")
    else:
        # Get all teams playing today
        teams_playing = set()
        teams_playing.update(schedule['home_team'].tolist())
        teams_playing.update(schedule['away_team'].tolist())
        teams_playing = sorted([t for t in teams_playing if t])  # Remove None values

        print(f"Found {len(teams_playing)} teams playing today: {', '.join(teams_playing)}")
        print(f"Games: {len(schedule)}")
        for _, game in schedule.iterrows():
            print(f"  {game['away_team']} @ {game['home_team']}")

        # Fetch lines for all teams
        print(f"\n{'=' * 60}")
        print(f" FETCHING LINES FOR ALL {len(teams_playing)} TEAMS")
        print(f"{'=' * 60}")

        all_lines = scraper.get_multiple_teams(teams_playing)

        # Print each team's lines
        for team in teams_playing:
            if team in all_lines and 'error' not in all_lines[team]:
                print_team_lines(all_lines[team])
            else:
                print(f"\n[ERROR] Could not fetch lines for {team}")

        # Build stack recommendations for all teams
        print("\n" + "=" * 60)
        print(" STACK RECOMMENDATIONS (ALL TEAMS)")
        print("=" * 60)

        stack_builder = StackBuilder(all_lines)

        for team in teams_playing:
            stacks = stack_builder.get_best_stacks(team)
            if stacks:
                print(f"\n{team}:")
                for stack in stacks:
                    print(f"  {stack['type']} (corr: {stack['correlation']}): {', '.join(stack['players'][:4])}{'...' if len(stack['players']) > 4 else ''}")

        # Show all starting goalies
        print("\n" + "=" * 60)
        print(" CONFIRMED STARTING GOALIES")
        print("=" * 60)
        goalies = stack_builder.get_all_starting_goalies()
        for team in teams_playing:
            goalie = goalies.get(team, 'TBD')
            print(f"  {team}: {goalie}")
