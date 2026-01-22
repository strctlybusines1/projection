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
from config import (
    CURRENT_SEASON, NHL_TEAMS, calculate_skater_fantasy_points,
    calculate_goalie_fantasy_points
)


class NHLDataPipeline:
    """Pipeline for fetching and processing NHL data."""

    def __init__(self):
        self.client = NHLAPIClient(rate_limit_delay=0.3)

    # ==================== Skater Data ====================

    def fetch_all_skater_stats(self, season: str = CURRENT_SEASON) -> pd.DataFrame:
        """Fetch season stats for all skaters."""
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

    # ==================== Combined Data ====================

    def build_projection_dataset(self, season: str = CURRENT_SEASON,
                                   include_game_logs: bool = False,
                                   max_players_for_logs: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Build complete dataset for projections.

        Returns dict with:
            - skaters: Season stats for all skaters
            - goalies: Season stats for all goalies
            - teams: Team-level stats
            - schedule: Upcoming games
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
