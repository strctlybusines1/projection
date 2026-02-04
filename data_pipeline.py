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
    GOALIE_SCORING, GOALIE_BONUSES,
    SIMULATION_DEFAULT_STD, SIMULATION_MIN_GAMES_FOR_STD,
)
from danger_stats import load_team_danger_stats

# Edge stats (optional - requires nhl-api-py)
try:
    from edge_stats import EdgeStatsClient
    EDGE_AVAILABLE = True
except ImportError:
    EDGE_AVAILABLE = False

# Edge caching (optional - much faster when available)
try:
    from edge_cache import EdgeStatsCache, get_cached_edge_stats
    EDGE_CACHE_AVAILABLE = True
except ImportError:
    EDGE_CACHE_AVAILABLE = False


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

    @staticmethod
    def _calculate_goalie_game_dk_fpts(game: Dict) -> float:
        """Calculate DraftKings fantasy points for a goalie from a single game log entry.

        Uses NHL API game log fields (camelCase keys).
        Mirrors backtest.py goalie scoring logic.
        """
        saves = game.get('saves', 0) or game.get('savesAgainst', 0) or 0
        goals_against = game.get('goalsAgainst', 0) or 0
        decision = game.get('decision', '')

        is_win = decision == 'W'
        is_otl = decision == 'O'
        is_shutout = goals_against == 0 and is_win

        pts = (
            saves * GOALIE_SCORING['save'] +
            goals_against * GOALIE_SCORING['goal_against']
        )

        if is_win:
            pts += GOALIE_SCORING['win']
        if is_otl:
            pts += GOALIE_SCORING['overtime_loss']
        if is_shutout:
            pts += GOALIE_SCORING['shutout_bonus']
        if saves >= 35:
            pts += GOALIE_BONUSES['thirty_five_plus_saves']

        return pts

    def compute_player_fpts_std_dev(
        self,
        player_ids: List[int],
        player_types: Dict[int, str],
        season: str = CURRENT_SEASON,
        min_games: int = SIMULATION_MIN_GAMES_FOR_STD,
    ) -> Dict[int, Dict]:
        """Compute per-player fantasy points standard deviation from game logs.

        Args:
            player_ids: List of NHL player IDs
            player_types: Dict mapping player_id -> 'skater' or 'goalie'
            season: Season in YYYYYYYY format
            min_games: Minimum games required to use player-specific std dev

        Returns:
            Dict mapping player_id -> {
                'mean_fpts': float,
                'std_fpts': float,
                'n_games': int,
                'used_default': bool,
            }
        """
        results = {}
        print(f"Computing per-player std dev for {len(player_ids)} players...")

        for pid in tqdm(player_ids, desc="Std dev"):
            try:
                log = self.client.get_player_game_log(pid, season)
                games = log.get('gameLog', [])

                if not games:
                    ptype = player_types.get(pid, 'skater')
                    results[pid] = {
                        'mean_fpts': 0.0,
                        'std_fpts': SIMULATION_DEFAULT_STD.get(ptype, 5.5),
                        'n_games': 0,
                        'used_default': True,
                    }
                    continue

                ptype = player_types.get(pid, 'skater')
                if ptype == 'goalie':
                    fpts_list = [self._calculate_goalie_game_dk_fpts(g) for g in games]
                else:
                    fpts_list = [self._calculate_game_dk_fpts(g) for g in games]

                n_games = len(fpts_list)
                mean_fpts = float(np.mean(fpts_list))

                if n_games >= min_games:
                    std_fpts = float(np.std(fpts_list, ddof=1))
                    used_default = False
                else:
                    std_fpts = SIMULATION_DEFAULT_STD.get(ptype, 5.5)
                    used_default = True

                results[pid] = {
                    'mean_fpts': mean_fpts,
                    'std_fpts': std_fpts,
                    'n_games': n_games,
                    'used_default': used_default,
                }
            except Exception:
                ptype = player_types.get(pid, 'skater')
                results[pid] = {
                    'mean_fpts': 0.0,
                    'std_fpts': SIMULATION_DEFAULT_STD.get(ptype, 5.5),
                    'n_games': 0,
                    'used_default': True,
                }
                continue

        print(f"  Computed std dev for {len(results)} players "
              f"({sum(1 for v in results.values() if not v['used_default'])} with actual data)")
        return results

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

    # ==================== Edge Stats ====================

    def fetch_edge_stats(self, player_ids: List[int], show_progress: bool = True,
                          force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch NHL Edge tracking stats for skaters.

        Uses caching when available (much faster - seconds vs minutes).
        Edge stats update once daily, so caching is safe.

        Args:
            player_ids: List of NHL player IDs to fetch Edge data for
            show_progress: Whether to show progress updates
            force_refresh: If True, fetch fresh data from API (ignore cache)

        Returns:
            DataFrame with player_id, edge_boost, edge_boost_reasons, and raw Edge metrics
        """
        if not EDGE_AVAILABLE:
            if show_progress:
                print("  Edge stats: skipped (nhl-api-py not installed)")
            return pd.DataFrame()

        # Try cached approach first (much faster on subsequent runs)
        if EDGE_CACHE_AVAILABLE:
            return self._fetch_edge_stats_cached(player_ids, force_refresh=force_refresh,
                                                   show_progress=show_progress)

        # Fall back to per-player API calls without caching
        return self._fetch_edge_stats_per_player(player_ids, show_progress)

    def _fetch_edge_stats_cached(self, player_ids: List[int],
                                   force_refresh: bool = False,
                                   show_progress: bool = True) -> pd.DataFrame:
        """
        Fetch Edge stats using cached data.

        This is the fast path - uses daily cache, completes in seconds if cached.
        """
        cache = EdgeStatsCache()

        if show_progress:
            if cache.is_cache_valid() and not force_refresh:
                age = cache.get_cache_age_hours()
                print(f"  Edge stats: using cache (age: {age:.1f} hours)")
            else:
                print("  Edge stats: fetching from API (will cache for today)...")

        # Get cached Edge stats (fetches if needed, passing player_ids)
        edge_df = get_cached_edge_stats(player_ids=player_ids, force_refresh=force_refresh)

        if edge_df.empty:
            if show_progress:
                print("  Edge stats: no data available")
            return pd.DataFrame()

        # Calculate boosts for each player
        edge_client = EdgeStatsClient(rate_limit_delay=0.3)
        results = []

        for _, row in edge_df.iterrows():
            # Build summary dict from cached data
            summary = {
                'max_speed_mph': row.get('max_speed_mph', 0),
                'speed_percentile': row.get('speed_percentile', 0),
                'bursts_over_20': row.get('bursts_over_20', 0),
                'bursts_percentile': row.get('bursts_percentile', 0),
                'oz_time_pct': row.get('oz_time_pct', 0),
                'oz_time_percentile': row.get('oz_time_percentile', 0),
            }

            # Normalize percentiles (API returns 0-100, boost calc expects 0-1)
            for key in ['speed_percentile', 'bursts_percentile', 'oz_time_percentile']:
                if summary.get(key, 0) > 1:
                    summary[key] = summary[key] / 100

            boost, reasons = edge_client.get_edge_projection_boost(summary)
            results.append({
                'player_id': row.get('player_id'),
                'player_name': row.get('player_name', ''),
                'edge_boost': boost,
                'edge_boost_reasons': '; '.join(reasons) if reasons else '',
                'max_speed_mph': summary.get('max_speed_mph', 0),
                'speed_percentile': summary.get('speed_percentile', 0),
                'bursts_over_20': summary.get('bursts_over_20', 0),
                'bursts_percentile': summary.get('bursts_percentile', 0),
                'oz_time_pct': summary.get('oz_time_pct', 0),
                'oz_time_percentile': summary.get('oz_time_percentile', 0),
            })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        if show_progress:
            boosted = len(df[df['edge_boost'] > 1.0])
            print(f"    Edge stats: {len(df)} players, {boosted} with boosts")

        return df

    def _fetch_edge_stats_per_player(self, player_ids: List[int],
                                       show_progress: bool = True) -> pd.DataFrame:
        """
        Fetch Edge stats via per-player API calls (legacy approach).

        This is slower but works when caching is unavailable.
        """
        edge_client = EdgeStatsClient(rate_limit_delay=0.3)
        results = []

        if show_progress:
            print(f"  Fetching Edge stats for {len(player_ids)} skaters (no cache)...")

        for i, pid in enumerate(player_ids):
            if show_progress and (i + 1) % 50 == 0:
                print(f"    Edge progress: {i + 1}/{len(player_ids)}")

            try:
                summary = edge_client.get_skater_edge_summary(pid)
                if summary:
                    boost, reasons = edge_client.get_edge_projection_boost(summary)
                    results.append({
                        'player_id': pid,
                        'edge_boost': boost,
                        'edge_boost_reasons': '; '.join(reasons) if reasons else '',
                        'max_speed_mph': summary.get('max_speed_mph', 0),
                        'speed_percentile': summary.get('speed_percentile', 0),
                        'bursts_over_20': summary.get('bursts_over_20', 0),
                        'bursts_percentile': summary.get('bursts_percentile', 0),
                        'oz_time_pct': summary.get('oz_time_pct', 0),
                        'oz_time_percentile': summary.get('oz_time_percentile', 0),
                    })
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        if show_progress:
            boosted = len(df[df['edge_boost'] > 1.0])
            print(f"    Edge stats: {len(df)} players, {boosted} with boosts")

        return df

    # ==================== Combined Data ====================

    def build_projection_dataset(self, season: str = CURRENT_SEASON,
                                   include_game_logs: bool = False,
                                   include_injuries: bool = True,
                                   include_advanced_stats: bool = True,
                                   include_edge_stats: bool = False,
                                   force_refresh_edge: bool = False,
                                   max_players_for_logs: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Build complete dataset for projections.

        Args:
            season: NHL season in YYYYYYYY format
            include_game_logs: Whether to fetch game-by-game logs (slow)
            include_injuries: Whether to fetch injury data from MoneyPuck
            include_advanced_stats: Whether to fetch xG/Corsi from Natural Stat Trick
            include_edge_stats: Whether to fetch NHL Edge tracking data
            force_refresh_edge: If True, fetch fresh Edge data (ignore cache)
            max_players_for_logs: Max players to fetch game logs for

        Returns dict with:
            - skaters: Season stats for all skaters (with edge_boost if include_edge_stats)
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

        # Edge stats (merge into skaters if enabled)
        if include_edge_stats and not data['skaters'].empty:
            player_ids = []
            if 'player_id' in data['skaters'].columns:
                player_ids = data['skaters']['player_id'].dropna().astype(int).unique().tolist()

            edge_df = self.fetch_edge_stats(player_ids, show_progress=True,
                                             force_refresh=force_refresh_edge)

            if not edge_df.empty:
                # Merge Edge data into skaters by player_id
                if 'player_id' in edge_df.columns and 'player_id' in data['skaters'].columns:
                    # Drop player_name from edge_df to avoid column collision
                    merge_cols = [c for c in edge_df.columns if c != 'player_name']
                    data['skaters'] = data['skaters'].merge(
                        edge_df[merge_cols], on='player_id', how='left'
                    )
                    data['skaters']['edge_boost'] = data['skaters']['edge_boost'].fillna(1.0)

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
