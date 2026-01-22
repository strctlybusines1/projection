"""
NHL API Client for fetching player stats, schedules, and game data.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional
import time

from config import NHL_API_WEB_BASE, NHL_API_STATS_BASE, CURRENT_SEASON, GAME_TYPE_REGULAR


class NHLAPIClient:
    """Client for interacting with the NHL API."""

    def __init__(self, rate_limit_delay: float = 0.5):
        self.web_base = NHL_API_WEB_BASE
        self.stats_base = NHL_API_STATS_BASE
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def _get(self, url: str) -> dict:
        """Make a GET request with rate limiting."""
        time.sleep(self.rate_limit_delay)
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    # ==================== Schedule Endpoints ====================

    def get_schedule(self, date: Optional[str] = None) -> dict:
        """
        Get NHL schedule for a specific date.

        Args:
            date: Date in YYYY-MM-DD format. If None, gets today's schedule.
        """
        if date:
            url = f"{self.web_base}/v1/schedule/{date}"
        else:
            url = f"{self.web_base}/v1/schedule/now"
        return self._get(url)

    def get_team_schedule(self, team: str, season: str = CURRENT_SEASON) -> dict:
        """Get full season schedule for a team."""
        url = f"{self.web_base}/v1/club-schedule-season/{team}/{season}"
        return self._get(url)

    # ==================== Roster Endpoints ====================

    def get_roster(self, team: str) -> dict:
        """Get current roster for a team."""
        url = f"{self.web_base}/v1/roster/{team}/current"
        return self._get(url)

    def get_all_rosters(self, teams: list) -> dict:
        """Get rosters for multiple teams."""
        rosters = {}
        for team in teams:
            try:
                rosters[team] = self.get_roster(team)
            except Exception as e:
                print(f"Error fetching roster for {team}: {e}")
        return rosters

    # ==================== Player Stats Endpoints ====================

    def get_player_landing(self, player_id: int) -> dict:
        """Get comprehensive player information."""
        url = f"{self.web_base}/v1/player/{player_id}/landing"
        return self._get(url)

    def get_player_game_log(self, player_id: int, season: str = CURRENT_SEASON,
                           game_type: int = GAME_TYPE_REGULAR) -> dict:
        """
        Get player game log for a specific season.

        Args:
            player_id: NHL player ID
            season: Season in YYYYYYYY format (e.g., 20242025)
            game_type: 2 for regular season, 3 for playoffs
        """
        url = f"{self.web_base}/v1/player/{player_id}/game-log/{season}/{game_type}"
        return self._get(url)

    def get_player_game_log_current(self, player_id: int) -> dict:
        """Get player game log for current season through today."""
        url = f"{self.web_base}/v1/player/{player_id}/game-log/now"
        return self._get(url)

    # ==================== Bulk Stats Endpoints ====================

    def get_skater_stats(self, season: str = CURRENT_SEASON, limit: int = -1) -> dict:
        """
        Get bulk skater statistics.

        Args:
            season: Season ID (e.g., 20242025)
            limit: Number of results (-1 for all)
        """
        url = (f"{self.stats_base}/en/skater/summary?"
               f"limit={limit}&cayenneExp=seasonId={season} and gameTypeId=2")
        return self._get(url)

    def get_goalie_stats(self, season: str = CURRENT_SEASON, limit: int = -1) -> dict:
        """
        Get bulk goalie statistics.

        Args:
            season: Season ID (e.g., 20242025)
            limit: Number of results (-1 for all)
        """
        url = (f"{self.stats_base}/en/goalie/summary?"
               f"limit={limit}&cayenneExp=seasonId={season} and gameTypeId=2")
        return self._get(url)

    def get_skater_advanced_stats(self, season: str = CURRENT_SEASON,
                                   report: str = "realtime", limit: int = -1) -> dict:
        """
        Get advanced skater stats (realtime, faceoffs, penalties, etc.).

        Available reports: summary, bios, faceoffpercentages, faceoffwins,
        goalsForAgainst, realtime, penalties, penaltykill, powerplay,
        puckPossessions, summaryshooting, percentages, scoringRates,
        scoringpergame, shootout, shottype, timeonice
        """
        url = (f"{self.stats_base}/en/skater/{report}?"
               f"limit={limit}&cayenneExp=seasonId={season} and gameTypeId=2")
        return self._get(url)

    # ==================== Game Endpoints ====================

    def get_boxscore(self, game_id: int) -> dict:
        """Get boxscore for a specific game."""
        url = f"{self.web_base}/v1/gamecenter/{game_id}/boxscore"
        return self._get(url)

    def get_play_by_play(self, game_id: int) -> dict:
        """Get play-by-play data for a specific game."""
        url = f"{self.web_base}/v1/gamecenter/{game_id}/play-by-play"
        return self._get(url)

    def get_game_landing(self, game_id: int) -> dict:
        """Get game landing page data."""
        url = f"{self.web_base}/v1/gamecenter/{game_id}/landing"
        return self._get(url)

    # ==================== Standings & Team Stats ====================

    def get_standings(self, date: Optional[str] = None) -> dict:
        """Get league standings."""
        if date:
            url = f"{self.web_base}/v1/standings/{date}"
        else:
            url = f"{self.web_base}/v1/standings/now"
        return self._get(url)

    def get_team_stats(self, team: str) -> dict:
        """Get current team statistics."""
        url = f"{self.web_base}/v1/club-stats/{team}/now"
        return self._get(url)

    # ==================== Leaders ====================

    def get_skater_leaders(self, categories: str = "goals", limit: int = 20) -> dict:
        """Get skater stat leaders."""
        url = f"{self.web_base}/v1/skater-stats-leaders/current?categories={categories}&limit={limit}"
        return self._get(url)

    def get_goalie_leaders(self, categories: str = "wins", limit: int = 20) -> dict:
        """Get goalie stat leaders."""
        url = f"{self.web_base}/v1/goalie-stats-leaders/current?categories={categories}&limit={limit}"
        return self._get(url)


# Quick test
if __name__ == "__main__":
    client = NHLAPIClient()

    # Test getting today's schedule
    print("Fetching today's schedule...")
    schedule = client.get_schedule()
    print(f"Found {len(schedule.get('gameWeek', []))} days of games")

    # Test getting skater stats
    print("\nFetching skater stats...")
    skaters = client.get_skater_stats(limit=5)
    print(f"Sample skaters: {[p.get('skaterFullName') for p in skaters.get('data', [])[:5]]}")
