"""
Data pipeline for fetching and processing NHL data for DFS projections.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from tqdm import tqdm
import time

from nhl_api import NHLAPIClient
from scrapers import MoneyPuckClient, NaturalStatTrickScraper
from config import (
    CURRENT_SEASON, NHL_TEAMS, calculate_skater_fantasy_points,
    calculate_goalie_fantasy_points, INJURY_STATUSES_EXCLUDE,
    NST_RECENT_FORM_GAMES, TEAM_DANGER_CSV_DIR, TEAM_DANGER_CSV,
)
from danger_stats import load_team_danger_stats


class NHLDataPipeline:
    """Pipeline for fetching and processing NHL data."""

    def __init__(self):
        self.client = NHLAPIClient(rate_limit_delay=0.3)
        self.injury_client = MoneyPuckClient()
        self.nst_scraper = NaturalStatTrickScraper()

    # ==================== Skater Data ====================

    def fetch_all_skater_stats(self, season: str = CURRENT_SEASON) -> pd.DataFrame:
        """Fetch season stats for all skaters including TOI breakdown."""
        print(f"Fetching skater stats for {season}...")

        # Get main summary stats
        summary = self.client.get_skater_stats(season=season, limit=-1)
        df = pd.DataFrame(summary.get('data', []))

        if df.empty:
            return df

        # Rename columns for clarity
        column_map = {
            'playerId': 'player_id',
            'skaterFullName': 'name',
            'teamAbbrevs': 'team',
            'positionCode': 'position',
            'gamesPlayed': 'games_played',
            'goals': 'goals',
            'assists': 'assists',
            'points': 'points',
            'plusMinus': 'plus_minus',
            'penaltyMinutes': 'pim',
            'ppGoals': 'pp_goals',
            'ppPoints': 'pp_points',
            'shGoals': 'sh_goals',
            'shPoints': 'sh_points',
            'gameWinningGoals': 'gw_goals',
            'otGoals': 'ot_goals',
            'shots': 'shots',
            'shootingPct': 'shooting_pct',
            'timeOnIcePerGame': 'toi_per_game',
            'faceoffWinPct': 'faceoff_pct',
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Fetch TOI breakdown (5v5, PP, PK)
        toi_data = self.fetch_skater_advanced_stats(season)
        if 'toi' in toi_data and not toi_data['toi'].empty:
            toi_df = toi_data['toi']

            # Rename TOI columns
            toi_column_map = {
                'playerId': 'player_id',
                'evTimeOnIcePerGame': 'ev_toi_per_game',  # 5v5 TOI
                'ppTimeOnIcePerGame': 'pp_toi_per_game',  # Power play TOI
                'shTimeOnIcePerGame': 'sh_toi_per_game',  # Shorthanded TOI
                'timeOnIcePerGame': 'total_toi_per_game',
            }
            toi_df = toi_df.rename(columns={k: v for k, v in toi_column_map.items() if k in toi_df.columns})

            # Merge TOI breakdown into main df
            toi_cols = ['player_id', 'ev_toi_per_game', 'pp_toi_per_game', 'sh_toi_per_game']
            toi_cols = [c for c in toi_cols if c in toi_df.columns]

            if 'player_id' in toi_df.columns and len(toi_cols) > 1:
                df = df.merge(toi_df[toi_cols], on='player_id', how='left')

        # Calculate per-game stats
        if 'games_played' in df.columns and df['games_played'].gt(0).any():
            for col in ['goals', 'assists', 'points', 'shots', 'pp_points']:
                if col in df.columns:
                    df[f'{col}_per_game'] = df[col] / df['games_played'].replace(0, np.nan)

        return df

    def fetch_skater_advanced_stats(self, season: str = CURRENT_SEASON) -> pd.DataFrame:
        """Fetch advanced stats including TOI breakdown, shot types, etc."""
        print("Fetching advanced skater stats...")

        # Time on ice breakdown
        toi_data = self.client.get_skater_advanced_stats(season, report="timeonice", limit=-1)
        toi_df = pd.DataFrame(toi_data.get('data', []))

        # Realtime stats (hits, blocks, etc.)
        realtime_data = self.client.get_skater_advanced_stats(season, report="realtime", limit=-1)
        realtime_df = pd.DataFrame(realtime_data.get('data', []))

        # Shooting stats
        shooting_data = self.client.get_skater_advanced_stats(season, report="summaryshooting", limit=-1)
        shooting_df = pd.DataFrame(shooting_data.get('data', []))

        return {
            'toi': toi_df,
            'realtime': realtime_df,
            'shooting': shooting_df
        }

    def fetch_player_game_logs(self, player_ids: List[int], season: str = CURRENT_SEASON,
                                max_players: Optional[int] = None) -> pd.DataFrame:
        """Fetch game-by-game logs for multiple players."""
        all_logs = []

        player_ids = player_ids[:max_players] if max_players else player_ids
        print(f"Fetching game logs for {len(player_ids)} players...")

        for pid in tqdm(player_ids, desc="Fetching game logs"):
            try:
                log = self.client.get_player_game_log(pid, season)
                games = log.get('gameLog', [])

                for game in games:
                    game['player_id'] = pid
                    all_logs.append(game)

            except Exception as e:
                print(f"Error fetching logs for player {pid}: {e}")
                continue

        df = pd.DataFrame(all_logs)
        return df

    # ==================== Recent Game Scoring ====================

    @staticmethod
    def _parse_toi_minutes(toi_str) -> Optional[float]:
        """Convert TOI string "MM:SS" to float minutes (e.g. "20:15" -> 20.25).

        Returns None if the string is missing or unparseable.
        """
        if not toi_str or not isinstance(toi_str, str):
            return None
        try:
            parts = toi_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
            return minutes + seconds / 60.0
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _calculate_game_dk_fpts(game: Dict) -> float:
        """Calculate DraftKings fantasy points from a single game log entry.

        Uses the NHL API game log fields (camelCase keys).
        """
        goals = game.get('goals', 0) or 0
        assists = game.get('assists', 0) or 0
        shots = game.get('shots', 0) or 0
        blocks = game.get('blockedShots', 0) or 0
        sh_goals = game.get('shorthandedGoals', 0) or 0
        sh_assists = game.get('shorthandedAssists', 0) or 0

        pts = 0.0
        pts += goals * 8.5
        pts += assists * 5.0
        pts += shots * 1.5
        pts += blocks * 1.3
        pts += (sh_goals + sh_assists) * 2.0

        # Bonuses
        if goals >= 3:
            pts += 3.0  # hat trick
        if shots >= 5:
            pts += 3.0  # 5+ shots
        if blocks >= 3:
            pts += 3.0  # 3+ blocks
        if (goals + assists) >= 3:
            pts += 3.0  # 3+ points

        return pts

    def fetch_recent_game_scores(self, player_ids: List[int]) -> Dict[int, Dict[str, float]]:
        """Fetch recent game DK fantasy scores for players.

        Args:
            player_ids: List of NHL player IDs

        Returns:
            Dict mapping player_id -> {
                'last_1_game_fpts': float,
                'last_3_avg_fpts': float,
                'last_5_avg_fpts': float,
                'last_3_avg_toi_min': float or None,  # avg TOI in minutes over last 3 games
            }
        """
        results = {}
        print(f"Fetching recent game scores for {len(player_ids)} players...")

        for pid in tqdm(player_ids, desc="Recent scores"):
            try:
                log = self.client.get_player_game_log_current(pid)
                games = log.get('gameLog', [])

                if not games:
                    continue

                # Games are returned most-recent first; compute DK FPTS for each
                fpts_list = [self._calculate_game_dk_fpts(g) for g in games[:5]]

                # Parse TOI from last 3 games (Feature 6)
                toi_values = []
                for g in games[:3]:
                    toi_min = self._parse_toi_minutes(g.get('toi'))
                    if toi_min is not None:
                        toi_values.append(toi_min)
                last_3_avg_toi = (
                    sum(toi_values) / len(toi_values) if toi_values else None
                )

                results[pid] = {
                    'last_1_game_fpts': fpts_list[0] if len(fpts_list) >= 1 else 0.0,
                    'last_3_avg_fpts': (
                        sum(fpts_list[:3]) / min(3, len(fpts_list))
                        if fpts_list else 0.0
                    ),
                    'last_5_avg_fpts': (
                        sum(fpts_list[:5]) / min(5, len(fpts_list))
                        if fpts_list else 0.0
                    ),
                    'last_3_avg_toi_min': last_3_avg_toi,
                }
            except Exception as e:
                # Skip individual failures silently to avoid spamming output
                continue

            # Rate limit to avoid overwhelming NHL API
            time.sleep(0.3)

        print(f"  Fetched recent scores for {len(results)} players")
        return results

    # ==================== Goalie Data ====================

    def fetch_all_goalie_stats(self, season: str = CURRENT_SEASON) -> pd.DataFrame:
        """Fetch season stats for all goalies."""
        print(f"Fetching goalie stats for {season}...")

        data = self.client.get_goalie_stats(season=season, limit=-1)
        df = pd.DataFrame(data.get('data', []))

        if df.empty:
            return df

        column_map = {
            'playerId': 'player_id',
            'goalieFullName': 'name',
            'teamAbbrevs': 'team',
            'gamesPlayed': 'games_played',
            'gamesStarted': 'games_started',
            'wins': 'wins',
            'losses': 'losses',
            'otLosses': 'ot_losses',
            'savePct': 'save_pct',
            'goalsAgainstAverage': 'gaa',
            'shotsAgainst': 'shots_against',
            'saves': 'saves',
            'goalsAgainst': 'goals_against',
            'shutouts': 'shutouts',
            'timeOnIce': 'toi',
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Calculate per-game stats
        if 'games_started' in df.columns:
            gp = df['games_started'].replace(0, np.nan)
            df['saves_per_start'] = df['saves'] / gp
            df['ga_per_start'] = df['goals_against'] / gp
            df['shots_against_per_start'] = df['shots_against'] / gp

        return df

    # ==================== Schedule & Games ====================

    def fetch_schedule(self, date: Optional[str] = None) -> pd.DataFrame:
        """Fetch schedule for a date (or today)."""
        schedule = self.client.get_schedule(date)

        games = []
        for day in schedule.get('gameWeek', []):
            for game in day.get('games', []):
                games.append({
                    'game_id': game.get('id'),
                    'date': day.get('date'),
                    'home_team': game.get('homeTeam', {}).get('abbrev'),
                    'away_team': game.get('awayTeam', {}).get('abbrev'),
                    'start_time': game.get('startTimeUTC'),
                    'game_state': game.get('gameState'),
                    'venue': game.get('venue', {}).get('default'),
                })

        return pd.DataFrame(games)

    def fetch_today_slate(self) -> pd.DataFrame:
        """Get today's games formatted for DFS slate."""
        schedule = self.fetch_schedule()
        today = datetime.now().strftime('%Y-%m-%d')
        return schedule[schedule['date'] == today]

    # ==================== Team Stats ====================

    def fetch_team_stats(self) -> pd.DataFrame:
        """Fetch stats for all teams."""
        print("Fetching team stats...")
        standings = self.client.get_standings()

        teams = []
        for team in standings.get('standings', []):
            teams.append({
                'team': team.get('teamAbbrev', {}).get('default'),
                'team_name': team.get('teamName', {}).get('default'),
                'games_played': team.get('gamesPlayed'),
                'wins': team.get('wins'),
                'losses': team.get('losses'),
                'ot_losses': team.get('otLosses'),
                'points': team.get('points'),
                'goals_for': team.get('goalFor'),
                'goals_against': team.get('goalAgainst'),
                'goal_diff': team.get('goalDifferential'),
                'streak': team.get('streakCode'),
            })

        df = pd.DataFrame(teams)

        if not df.empty and 'games_played' in df.columns:
            df['goals_for_per_game'] = df['goals_for'] / df['games_played'].replace(0, np.nan)
            df['goals_against_per_game'] = df['goals_against'] / df['games_played'].replace(0, np.nan)

        return df

    # ==================== Injuries ====================

    def fetch_injuries(self) -> pd.DataFrame:
        """
        Fetch current injuries from MoneyPuck.

        Returns:
            DataFrame with player_id, player_name, team, position, injury_status, etc.
        """
        return self.injury_client.fetch_injuries()

    def get_injured_player_ids(self, include_dtd: bool = True) -> List[int]:
        """
        Get list of injured player IDs.

        Args:
            include_dtd: If True, include Day-to-Day players in injury list

        Returns:
            List of player IDs who are injured
        """
        return self.injury_client.get_injured_player_ids(exclude_dtd=not include_dtd)

    def filter_injured_players(self, df: pd.DataFrame, include_dtd: bool = True,
                                player_id_col: str = 'player_id',
                                name_col: str = 'name') -> pd.DataFrame:
        """
        Filter out injured players from a DataFrame.

        Args:
            df: DataFrame with player data
            include_dtd: If True, also filter out Day-to-Day players
            player_id_col: Column name for player ID
            name_col: Column name for player name

        Returns:
            DataFrame with injured players removed
        """
        injuries = self.fetch_injuries()

        if injuries.empty:
            return df

        original_count = len(df)

        # Filter by injury status
        if include_dtd:
            # Exclude all injuries including DTD
            statuses_to_exclude = INJURY_STATUSES_EXCLUDE + ['DTD']
        else:
            # Only exclude severe injuries (IR, O, etc.)
            statuses_to_exclude = INJURY_STATUSES_EXCLUDE

        injured_df = injuries[injuries['injury_status'].isin(statuses_to_exclude)]

        # Filter by player ID if available
        if player_id_col in df.columns and 'player_id' in injured_df.columns:
            injured_ids = injured_df['player_id'].tolist()
            df = df[~df[player_id_col].isin(injured_ids)]
        # Otherwise filter by name
        elif name_col in df.columns and 'player_name' in injured_df.columns:
            injured_names = injured_df['player_name'].str.lower().str.strip().tolist()
            df = df[~df[name_col].str.lower().str.strip().isin(injured_names)]

        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(f"  Filtered {filtered_count} injured players")

        return df

    # ==================== Advanced Stats ====================

    def fetch_advanced_team_stats(self, season: str = CURRENT_SEASON,
                                   last_n_games: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch advanced team stats from Natural Stat Trick.

        Args:
            season: Season in YYYYYYYY format
            last_n_games: If set, fetch stats for last N games only

        Returns:
            Dict with keys: '5v5', 'pp', 'pk', 'recent_form' (if last_n_games)
        """
        stats = {}

        # Full season 5v5 stats
        stats['5v5'] = self.nst_scraper.fetch_team_stats(
            season=season, situation='5v5', rate=True
        )

        # Power play stats
        stats['pp'] = self.nst_scraper.fetch_team_stats(
            season=season, situation='pp', rate=True
        )

        # Penalty kill stats
        stats['pk'] = self.nst_scraper.fetch_team_stats(
            season=season, situation='pk', rate=True
        )

        # Recent form (last N games)
        if last_n_games:
            stats['recent_form'] = self.nst_scraper.fetch_team_stats(
                season=season, situation='5v5', rate=True, last_n_games=last_n_games
            )

        return stats

    def fetch_advanced_player_stats(self, season: str = CURRENT_SEASON,
                                     last_n_games: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch advanced player stats from Natural Stat Trick.

        Args:
            season: Season in YYYYYYYY format
            last_n_games: If set, fetch stats for last N games only

        Returns:
            Dict with keys: '5v5', 'pp', 'recent_form' (if last_n_games)
        """
        stats = {}

        # Full season 5v5 stats
        stats['5v5'] = self.nst_scraper.fetch_player_stats(
            season=season, situation='5v5', rate=True
        )

        # Power play stats
        stats['pp'] = self.nst_scraper.fetch_player_stats(
            season=season, situation='pp', rate=True
        )

        # Recent form
        if last_n_games:
            stats['recent_form'] = self.nst_scraper.fetch_player_stats(
                season=season, situation='5v5', rate=True, last_n_games=last_n_games
            )

        return stats

    # ==================== Combined Data ====================

    def build_projection_dataset(self, season: str = CURRENT_SEASON,
                                   include_game_logs: bool = False,
                                   include_injuries: bool = True,
                                   include_advanced_stats: bool = True,
                                   max_players_for_logs: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Build complete dataset for projections.

        Args:
            season: NHL season in YYYYYYYY format
            include_game_logs: Whether to fetch game-by-game logs (slow)
            include_injuries: Whether to fetch injury data from MoneyPuck
            include_advanced_stats: Whether to fetch xG/Corsi from Natural Stat Trick
            max_players_for_logs: Max players to fetch game logs for

        Returns dict with:
            - skaters: Season stats for all skaters
            - goalies: Season stats for all goalies
            - teams: Team-level stats
            - schedule: Upcoming games
            - injuries: (optional) Current injury data
            - advanced_team_stats: (optional) xG, Corsi, PDO from NST
            - advanced_player_stats: (optional) Player-level xG from NST
            - game_logs: (optional) Game-by-game data
        """
        print("=" * 50)
        print("Building NHL Projection Dataset")
        print("=" * 50)

        data = {}

        # Skater stats
        data['skaters'] = self.fetch_all_skater_stats(season)
        print(f"  Skaters: {len(data['skaters'])} players")

        # Goalie stats
        data['goalies'] = self.fetch_all_goalie_stats(season)
        print(f"  Goalies: {len(data['goalies'])} players")

        # Team stats
        data['teams'] = self.fetch_team_stats()
        print(f"  Teams: {len(data['teams'])} teams")

        # Schedule
        data['schedule'] = self.fetch_schedule()
        print(f"  Schedule: {len(data['schedule'])} games fetched")

        # Injuries from MoneyPuck
        if include_injuries:
            data['injuries'] = self.fetch_injuries()
            if not data['injuries'].empty:
                print(f"  Injuries: {len(data['injuries'])} injured players")
                # Show breakdown by status
                if 'injury_status' in data['injuries'].columns:
                    status_counts = data['injuries']['injury_status'].value_counts()
                    for status, count in status_counts.items():
                        print(f"    - {status}: {count}")

        # Advanced stats from Natural Stat Trick
        if include_advanced_stats:
            print("  Fetching advanced stats (this may take a moment)...")
            data['advanced_team_stats'] = self.fetch_advanced_team_stats(
                season=season, last_n_games=NST_RECENT_FORM_GAMES
            )
            # Count teams in 5v5 stats
            if '5v5' in data['advanced_team_stats'] and not data['advanced_team_stats']['5v5'].empty:
                n_teams = len(data['advanced_team_stats']['5v5'])
                print(f"  Advanced team stats: {n_teams} teams")

            data['advanced_player_stats'] = self.fetch_advanced_player_stats(
                season=season, last_n_games=NST_RECENT_FORM_GAMES
            )
            if '5v5' in data['advanced_player_stats'] and not data['advanced_player_stats']['5v5'].empty:
                n_players = len(data['advanced_player_stats']['5v5'])
                print(f"  Advanced player stats: {n_players} players")

        # Team danger (HD/MD/LD) from test-folder NST CSVs for goalie opponent shot quality
        if TEAM_DANGER_CSV_DIR:
            danger_df = load_team_danger_stats(csv_dir=TEAM_DANGER_CSV_DIR, csv_file=TEAM_DANGER_CSV)
            if danger_df is not None and not danger_df.empty:
                data['team_danger_stats'] = danger_df
                print(f"  Team danger stats: {len(danger_df)} teams (HD/MD/LD)")

        # Game logs (optional - takes longer)
        if include_game_logs and not data['skaters'].empty:
            top_players = data['skaters'].nlargest(max_players_for_logs, 'points')['player_id'].tolist()
            data['game_logs'] = self.fetch_player_game_logs(top_players, season)
            print(f"  Game logs: {len(data['game_logs'])} entries")

        print("=" * 50)
        print("Dataset build complete!")

        return data


# Quick test
if __name__ == "__main__":
    pipeline = NHLDataPipeline()

    # Build basic dataset (without game logs for speed)
    data = pipeline.build_projection_dataset(include_game_logs=False)

    print("\nSample skaters:")
    print(data['skaters'][['name', 'team', 'position', 'goals', 'assists', 'shots']].head(10))

    print("\nSample goalies:")
    print(data['goalies'][['name', 'team', 'wins', 'save_pct', 'gaa']].head(5))

    print("\nToday's schedule:")
    print(data['schedule'][['date', 'home_team', 'away_team']].head(10))
