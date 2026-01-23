"""
Scrapers for MoneyPuck injuries and Natural Stat Trick advanced analytics.
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from io import StringIO
import time
from bs4 import BeautifulSoup


class MoneyPuckClient:
    """
    Client for fetching injury data from MoneyPuck.

    CSV Columns: playerId, playerName, teamCode, position, dateOfReturn,
                 gamesStillToMiss, playerInjuryStatus
    Status Codes: DTD (Day-to-Day), IR, IR-LT, IR-NR, O (Out)
    """

    INJURIES_URL = "https://moneypuck.com/moneypuck/playerData/playerNews/current_injuries.csv"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._cache = None
        self._cache_time = None

    def fetch_injuries(self, use_cache: bool = True, cache_minutes: int = 15) -> pd.DataFrame:
        """
        Fetch current injuries from MoneyPuck.

        Args:
            use_cache: Whether to use cached data if available
            cache_minutes: How long to cache data

        Returns:
            DataFrame with injury data
        """
        # Check cache
        if use_cache and self._cache is not None and self._cache_time is not None:
            age = (pd.Timestamp.now() - self._cache_time).total_seconds() / 60
            if age < cache_minutes:
                return self._cache.copy()

        try:
            print("Fetching injuries from MoneyPuck...")
            response = requests.get(self.INJURIES_URL, timeout=self.timeout)
            response.raise_for_status()

            # Parse CSV
            df = pd.read_csv(StringIO(response.text))

            # Standardize column names
            column_map = {
                'playerId': 'player_id',
                'playerName': 'player_name',
                'teamCode': 'team',
                'position': 'position',
                'dateOfReturn': 'return_date',
                'gamesStillToMiss': 'games_to_miss',
                'playerInjuryStatus': 'injury_status'
            }

            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

            # Parse return date
            if 'return_date' in df.columns:
                df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')

            # Cache the result
            self._cache = df
            self._cache_time = pd.Timestamp.now()

            print(f"  Found {len(df)} injured players")
            return df

        except Exception as e:
            print(f"Error fetching injuries: {e}")
            return pd.DataFrame()

    def get_injured_player_ids(self, exclude_dtd: bool = False) -> List[int]:
        """
        Get list of injured player IDs.

        Args:
            exclude_dtd: If True, exclude Day-to-Day players (they might play)

        Returns:
            List of player IDs who are injured
        """
        df = self.fetch_injuries()

        if df.empty:
            return []

        # Filter by status
        if exclude_dtd:
            # Exclude only DTD, keep IR/IR-LT/IR-NR/O
            df = df[df['injury_status'] != 'DTD']
        else:
            # Include all injuries
            pass

        if 'player_id' in df.columns:
            return df['player_id'].tolist()
        return []

    def get_injured_players_by_team(self, team: str) -> pd.DataFrame:
        """Get injured players for a specific team."""
        df = self.fetch_injuries()

        if df.empty or 'team' not in df.columns:
            return pd.DataFrame()

        return df[df['team'].str.upper() == team.upper()]

    def is_player_injured(self, player_name: str, team: Optional[str] = None) -> bool:
        """
        Check if a specific player is injured.

        Args:
            player_name: Player name to check
            team: Optional team code to narrow search

        Returns:
            True if player is found in injury list
        """
        df = self.fetch_injuries()

        if df.empty or 'player_name' not in df.columns:
            return False

        # Normalize name for comparison
        name_lower = player_name.lower().strip()
        df['name_lower'] = df['player_name'].str.lower().str.strip()

        matches = df[df['name_lower'].str.contains(name_lower, na=False)]

        if team and 'team' in df.columns:
            matches = matches[matches['team'].str.upper() == team.upper()]

        return len(matches) > 0

    def get_injury_status(self, player_name: str, team: Optional[str] = None) -> Optional[str]:
        """Get the injury status for a specific player."""
        df = self.fetch_injuries()

        if df.empty or 'player_name' not in df.columns:
            return None

        name_lower = player_name.lower().strip()
        df['name_lower'] = df['player_name'].str.lower().str.strip()

        matches = df[df['name_lower'].str.contains(name_lower, na=False)]

        if team and 'team' in df.columns:
            matches = matches[matches['team'].str.upper() == team.upper()]

        if len(matches) > 0 and 'injury_status' in matches.columns:
            return matches.iloc[0]['injury_status']

        return None


class NaturalStatTrickScraper:
    """
    Scraper for Natural Stat Trick advanced analytics.

    Fetches team and player stats including:
    - xGF/60, xGA/60 (expected goals)
    - CF%, FF% (Corsi/Fenwick)
    - HDCF/60 (high-danger chances)
    - SH%, SV%, PDO
    """

    BASE_URL = "https://www.naturalstattrick.com"

    def __init__(self, rate_limit_delay: float = 2.0, timeout: int = 30):
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _build_url(self, endpoint: str, params: dict) -> str:
        """Build URL with query parameters."""
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.BASE_URL}/{endpoint}?{param_str}"

    def fetch_team_stats(self, season: str = "20252026", situation: str = "5v5",
                          rate: bool = True, last_n_games: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch team-level advanced stats.

        Args:
            season: Season in format YYYYYYYY (e.g., "20252026")
            situation: Game situation ("5v5", "pp", "pk", "all")
            rate: If True, return per-60 rates
            last_n_games: If set, only include last N team games

        Returns:
            DataFrame with team advanced stats
        """
        self._rate_limit()

        # Map situation codes
        sit_map = {'5v5': '5v5', 'pp': 'pp', 'pk': 'pk', 'all': 'all', 'ev': 'ev'}
        sit = sit_map.get(situation.lower(), '5v5')

        params = {
            'fromseason': season,
            'thruseason': season,
            'stype': '2',  # Regular season
            'sit': sit,
            'score': 'all',
            'rate': 'y' if rate else 'n',
            'team': 'all',
            'loc': 'B',  # Both home and away
            'gpf': '410',  # All games
            'fd': '',
            'td': ''
        }

        # Add game filter if specified
        if last_n_games:
            params['gpfilt'] = 'gpteam'
            params['tgp'] = str(last_n_games)

        url = self._build_url("teamtable.php", params)

        try:
            print(f"Fetching team stats from Natural Stat Trick ({situation})...")
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML tables
            tables = pd.read_html(StringIO(response.text))

            if tables:
                df = tables[0]

                # Standardize column names
                df = self._standardize_team_columns(df)

                print(f"  Fetched stats for {len(df)} teams")
                return df
            else:
                print("  No tables found in response")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return pd.DataFrame()

    def fetch_player_stats(self, season: str = "20252026", situation: str = "5v5",
                           rate: bool = True, last_n_games: Optional[int] = None,
                           position: str = "all") -> pd.DataFrame:
        """
        Fetch player-level advanced stats.

        Args:
            season: Season in format YYYYYYYY
            situation: Game situation ("5v5", "pp", "pk", "all")
            rate: If True, return per-60 rates
            last_n_games: If set, only include last N team games
            position: Position filter ("all", "F", "D")

        Returns:
            DataFrame with player advanced stats
        """
        self._rate_limit()

        sit_map = {'5v5': '5v5', 'pp': 'pp', 'pk': 'pk', 'all': 'all', 'ev': 'ev'}
        sit = sit_map.get(situation.lower(), '5v5')

        pos_map = {'all': 'S', 'f': 'F', 'd': 'D', 's': 'S'}
        pos = pos_map.get(position.lower(), 'S')

        params = {
            'fromseason': season,
            'thruseason': season,
            'stype': '2',
            'sit': sit,
            'score': 'all',
            'stdoi': 'std',
            'rate': 'y' if rate else 'n',
            'team': 'ALL',
            'pos': pos,
            'loc': 'B',
            'toi': '0',  # Minimum TOI
            'gpfilt': 'none',
            'fd': '',
            'td': '',
            'tgp': '82',
            'lines': 'single',
            'dession': 'false'
        }

        if last_n_games:
            params['gpfilt'] = 'gpteam'
            params['tgp'] = str(last_n_games)

        url = self._build_url("playerteams.php", params)

        try:
            print(f"Fetching player stats from Natural Stat Trick ({situation})...")
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            tables = pd.read_html(StringIO(response.text))

            if tables:
                df = tables[0]
                df = self._standardize_player_columns(df)
                print(f"  Fetched stats for {len(df)} players")
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return pd.DataFrame()

    def fetch_recent_form(self, n_games: int = 10, season: str = "20252026") -> Dict[str, pd.DataFrame]:
        """
        Fetch recent form stats (last N games) for teams and players.

        Args:
            n_games: Number of recent games to consider
            season: Season in format YYYYYYYY

        Returns:
            Dict with 'teams' and 'players' DataFrames
        """
        return {
            'teams': self.fetch_team_stats(season=season, situation='5v5',
                                           rate=True, last_n_games=n_games),
            'players': self.fetch_player_stats(season=season, situation='5v5',
                                               rate=True, last_n_games=n_games)
        }

    def _standardize_team_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize team stats column names."""
        column_map = {
            'Team': 'team',
            'GP': 'games_played',
            'TOI': 'toi',
            'CF': 'cf',
            'CA': 'ca',
            'CF%': 'cf_pct',
            'FF': 'ff',
            'FA': 'fa',
            'FF%': 'ff_pct',
            'SF': 'sf',
            'SA': 'sa',
            'SF%': 'sf_pct',
            'GF': 'gf',
            'GA': 'ga',
            'GF%': 'gf_pct',
            'xGF': 'xgf',
            'xGA': 'xga',
            'xGF/60': 'xgf',  # NST uses /60 format
            'xGA/60': 'xga',  # NST uses /60 format
            'xGF%': 'xgf_pct',
            'SCF': 'scf',
            'SCA': 'sca',
            'SCF%': 'scf_pct',
            'HDCF': 'hdcf',
            'HDCA': 'hdca',
            'HDCF%': 'hdcf_pct',
            'HDGF': 'hdgf',
            'HDGA': 'hdga',
            'HDGF%': 'hdgf_pct',
            'SH%': 'sh_pct',
            'SV%': 'sv_pct',
            'PDO': 'pdo'
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Apply NST team code mappings
        if 'team' in df.columns:
            df['team'] = df['team'].apply(self._normalize_team_code)

        return df

    def _standardize_player_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize player stats column names."""
        column_map = {
            'Player': 'player_name',
            'Team': 'team',
            'Position': 'position',
            'GP': 'games_played',
            'TOI': 'toi',
            'CF': 'cf',
            'CA': 'ca',
            'CF%': 'cf_pct',
            'FF': 'ff',
            'FA': 'fa',
            'FF%': 'ff_pct',
            'GF': 'gf',
            'GA': 'ga',
            'xGF': 'xgf',
            'xGA': 'xga',
            'ixG': 'ixg',  # Individual expected goals
            'iCF': 'icf',
            'iFF': 'iff',
            'iSCF': 'iscf',
            'iHDCF': 'ihdcf',
            'SH%': 'sh_pct',
            'IPP': 'ipp',  # Individual points percentage
            'Goals': 'goals',
            'Total Assists': 'assists',
            'First Assists': 'first_assists',
            'Second Assists': 'second_assists',
            'Total Points': 'points'
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        if 'team' in df.columns:
            df['team'] = df['team'].apply(self._normalize_team_code)

        return df

    def _normalize_team_code(self, team: str) -> str:
        """Normalize NST team codes/names to standard NHL codes."""
        if pd.isna(team):
            return ''

        team_str = str(team).upper().strip()

        # Full team name mappings (NST often uses full names)
        full_name_to_code = {
            'ANAHEIM DUCKS': 'ANA',
            'ARIZONA COYOTES': 'ARI',
            'BOSTON BRUINS': 'BOS',
            'BUFFALO SABRES': 'BUF',
            'CALGARY FLAMES': 'CGY',
            'CAROLINA HURRICANES': 'CAR',
            'CHICAGO BLACKHAWKS': 'CHI',
            'COLORADO AVALANCHE': 'COL',
            'COLUMBUS BLUE JACKETS': 'CBJ',
            'DALLAS STARS': 'DAL',
            'DETROIT RED WINGS': 'DET',
            'EDMONTON OILERS': 'EDM',
            'FLORIDA PANTHERS': 'FLA',
            'LOS ANGELES KINGS': 'LAK',
            'MINNESOTA WILD': 'MIN',
            'MONTREAL CANADIENS': 'MTL',
            'NASHVILLE PREDATORS': 'NSH',
            'NEW JERSEY DEVILS': 'NJD',
            'NEW YORK ISLANDERS': 'NYI',
            'NEW YORK RANGERS': 'NYR',
            'OTTAWA SENATORS': 'OTT',
            'PHILADELPHIA FLYERS': 'PHI',
            'PITTSBURGH PENGUINS': 'PIT',
            'SAN JOSE SHARKS': 'SJS',
            'SEATTLE KRAKEN': 'SEA',
            'ST. LOUIS BLUES': 'STL',
            'ST LOUIS BLUES': 'STL',
            'TAMPA BAY LIGHTNING': 'TBL',
            'TORONTO MAPLE LEAFS': 'TOR',
            'UTAH HOCKEY CLUB': 'UTA',
            'VANCOUVER CANUCKS': 'VAN',
            'VEGAS GOLDEN KNIGHTS': 'VGK',
            'WASHINGTON CAPITALS': 'WSH',
            'WINNIPEG JETS': 'WPG',
        }

        # Check full name first
        if team_str in full_name_to_code:
            return full_name_to_code[team_str]

        # Short code mappings (for abbreviated codes)
        short_code_map = {
            'T.B': 'TBL',
            'N.J': 'NJD',
            'L.A': 'LAK',
            'S.J': 'SJS',
            'ANA': 'ANA',
            'ARI': 'ARI',
            'BOS': 'BOS',
            'BUF': 'BUF',
            'CGY': 'CGY',
            'CAR': 'CAR',
            'CHI': 'CHI',
            'COL': 'COL',
            'CBJ': 'CBJ',
            'DAL': 'DAL',
            'DET': 'DET',
            'EDM': 'EDM',
            'FLA': 'FLA',
            'LAK': 'LAK',
            'MIN': 'MIN',
            'MTL': 'MTL',
            'NSH': 'NSH',
            'NJD': 'NJD',
            'NYI': 'NYI',
            'NYR': 'NYR',
            'OTT': 'OTT',
            'PHI': 'PHI',
            'PIT': 'PIT',
            'SJS': 'SJS',
            'SEA': 'SEA',
            'STL': 'STL',
            'TBL': 'TBL',
            'TOR': 'TOR',
            'UTA': 'UTA',
            'VAN': 'VAN',
            'VGK': 'VGK',
            'WSH': 'WSH',
            'WPG': 'WPG'
        }

        return short_code_map.get(team_str, team_str)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing MoneyPuck Client")
    print("=" * 60)

    mp = MoneyPuckClient()
    injuries = mp.fetch_injuries()

    if not injuries.empty:
        print(f"\nSample injuries:")
        print(injuries.head(10))

        # Show by status
        if 'injury_status' in injuries.columns:
            print(f"\nInjury status breakdown:")
            print(injuries['injury_status'].value_counts())

    print("\n" + "=" * 60)
    print("Testing Natural Stat Trick Scraper")
    print("=" * 60)

    nst = NaturalStatTrickScraper()

    # Test team stats
    team_stats = nst.fetch_team_stats(situation='5v5', last_n_games=10)
    if not team_stats.empty:
        print(f"\nTeam stats (last 10 games):")
        cols_to_show = ['team', 'xgf', 'xga', 'cf_pct', 'pdo']
        cols_available = [c for c in cols_to_show if c in team_stats.columns]
        print(team_stats[cols_available].head(10))

    # Test player stats
    player_stats = nst.fetch_player_stats(situation='5v5', last_n_games=10)
    if not player_stats.empty:
        print(f"\nPlayer stats (last 10 games):")
        cols_to_show = ['player_name', 'team', 'xgf', 'ixg', 'cf_pct']
        cols_available = [c for c in cols_to_show if c in player_stats.columns]
        print(player_stats[cols_available].head(10))
