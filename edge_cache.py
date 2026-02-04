"""
NHL Edge Stats Caching Module (v2 - with Goalie Edge Stats)

Edge stats are cumulative season stats that update once daily (usually overnight).
This module caches Edge data to avoid redundant API calls during the same day.

NEW IN V2:
- Goalie Edge stats: High-danger SV%, midrange SV%, shot location details
- Unified lookup for both skaters and goalies
- Improved column mapping for boost calculation

Usage:
    from edge_cache import EdgeStatsCache, get_cached_edge_stats
    
    cache = EdgeStatsCache()
    edge_data = cache.get_edge_stats()  # Fetches if needed, else returns cached
    
    # Force refresh (e.g., if you know NHL updated)
    edge_data = cache.get_edge_stats(force_refresh=True)
    
    # Get goalie-specific Edge stats
    goalie_edge = cache.get_goalie_edge_stats()

Cache Location: cache/edge_stats_{date}.json, cache/goalie_edge_stats_{date}.json
"""

import os
import json
import requests
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import time

# Try to import nhl-api-py (used for Edge stats)
try:
    from nhlpy import NHLClient
    HAS_NHLPY = True
except ImportError:
    HAS_NHLPY = False
    print("Warning: nhl-api-py not installed. Edge stats will use direct API calls.")


# Direct API endpoints for goalie Edge stats (not in nhlpy yet)
NHL_API_WEB_BASE = "https://api-web.nhle.com"
NHL_API_STATS_BASE = "https://api.nhle.com/stats/rest/en"
CURRENT_SEASON = "20252026"


class EdgeStatsCache:
    """
    Caches NHL Edge tracking stats to avoid redundant API calls.
    
    Edge stats update once daily, typically overnight after games complete.
    This cache stores data per-date and reuses it for all projection runs
    on the same day.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cached data (relative to script location)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.today = date.today().strftime("%Y-%m-%d")
        self.skater_cache_file = self.cache_dir / f"edge_stats_{self.today}.json"
        self.goalie_cache_file = self.cache_dir / f"goalie_edge_stats_{self.today}.json"
        self.metadata_file = self.cache_dir / "edge_metadata.json"
        
        self.rate_limit_delay = 0.3  # seconds between API calls
        
    def _get_cache_key(self) -> str:
        """Generate a cache key for today's date."""
        return self.today
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata (last update times, etc.)."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def is_cache_valid(self, cache_type: str = 'skater') -> bool:
        """
        Check if today's cache exists and is valid.
        
        Args:
            cache_type: 'skater' or 'goalie'
            
        Returns:
            True if cache exists for today and appears valid
        """
        cache_file = self.skater_cache_file if cache_type == 'skater' else self.goalie_cache_file
        
        if not cache_file.exists():
            return False
        
        # Check file is not empty
        if cache_file.stat().st_size < 100:
            return False
        
        # Check metadata for fetch time
        metadata = self._load_metadata()
        last_fetch = metadata.get(f'last_fetch_date_{cache_type}')
        
        return last_fetch == self.today
    
    def get_cache_age_hours(self, cache_type: str = 'skater') -> Optional[float]:
        """Get how old the cache is in hours."""
        metadata = self._load_metadata()
        last_fetch_time = metadata.get(f'last_fetch_timestamp_{cache_type}')
        
        if not last_fetch_time:
            return None
        
        last_dt = datetime.fromisoformat(last_fetch_time)
        age = datetime.now() - last_dt
        return age.total_seconds() / 3600
    
    # ==================== SKATER EDGE STATS ====================
    
    def _fetch_skater_edge_from_api(self, player_ids: List[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch fresh skater Edge stats from NHL API using per-player endpoint.

        Args:
            player_ids: Optional list of player IDs. If None, fetches top skaters.

        Returns:
            Dict with 'skaters' DataFrame containing Edge metrics
        """
        if not HAS_NHLPY:
            print("⚠️ nhl-api-py not available, returning empty Edge data")
            return {}

        print("🔄 Fetching skater Edge stats from NHL API...")

        client = NHLClient()
        results = []

        # If no player_ids provided, get top skaters from stats API
        if not player_ids:
            try:
                url = f"{NHL_API_STATS_BASE}/skater/summary?limit=200&cayenneExp=seasonId={CURRENT_SEASON}"
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    player_ids = [p['playerId'] for p in data.get('data', [])[:150]]
                    print(f"  → Got {len(player_ids)} player IDs from stats API")
            except Exception as e:
                print(f"  ⚠️ Could not get player list: {e}")
                return {}

        print(f"  → Fetching Edge details for {len(player_ids)} skaters...")

        for i, pid in enumerate(player_ids):
            try:
                # Get player edge detail
                detail = client.edge.skater_detail(player_id=str(pid))
                time.sleep(self.rate_limit_delay)

                if not detail:
                    continue

                player_data = {
                    'player_id': pid,
                    'player_name': f"{detail.get('player', {}).get('firstName', {}).get('default', '')} {detail.get('player', {}).get('lastName', {}).get('default', '')}".strip(),
                    'team': detail.get('player', {}).get('team', {}).get('abbrev', ''),
                }

                # Skating speed
                skating = detail.get('skatingSpeed', {})
                if skating:
                    speed_max = skating.get('speedMax', {})
                    player_data['max_speed_mph'] = speed_max.get('imperial', 0)
                    player_data['speed_percentile'] = speed_max.get('percentile', 0)

                    bursts = skating.get('burstsOver20', {})
                    player_data['bursts_over_20'] = bursts.get('value', 0)
                    player_data['bursts_percentile'] = bursts.get('percentile', 0)

                # Get zone time (separate call)
                try:
                    zone = client.edge.skater_zone_time(player_id=str(pid))
                    time.sleep(self.rate_limit_delay)

                    if zone and 'zoneTimeDetails' in zone:
                        all_situations = next(
                            (z for z in zone['zoneTimeDetails'] if z.get('strengthCode') == 'all'),
                            None
                        )
                        if all_situations:
                            player_data['oz_time_pct'] = all_situations.get('offensiveZonePctg', 0)
                            player_data['oz_time_percentile'] = all_situations.get('offensiveZonePercentile', 0)
                except Exception:
                    pass

                results.append(player_data)

                if (i + 1) % 25 == 0:
                    print(f"    Processed {i + 1}/{len(player_ids)} skaters")

            except Exception:
                continue

        if not results:
            return {}

        print(f"    ✓ Got Edge data for {len(results)} skaters")
        return {'skaters': pd.DataFrame(results)}
    
    def _save_skater_cache(self, edge_data: Dict[str, pd.DataFrame]):
        """Save skater Edge data to cache file."""
        cache_content = {}
        for key, df in edge_data.items():
            if isinstance(df, pd.DataFrame):
                cache_content[key] = df.to_dict(orient='records')
            else:
                cache_content[key] = df
        
        with open(self.skater_cache_file, 'w') as f:
            json.dump(cache_content, f)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata['last_fetch_date_skater'] = self.today
        metadata['last_fetch_timestamp_skater'] = datetime.now().isoformat()
        metadata['skater_categories'] = list(edge_data.keys())
        metadata['skater_count'] = sum(len(df) for df in edge_data.values() if isinstance(df, pd.DataFrame))
        self._save_metadata(metadata)
        
        print(f"✅ Cached skater Edge stats to {self.skater_cache_file}")
    
    def _load_skater_cache(self) -> Dict[str, pd.DataFrame]:
        """Load skater Edge data from cache file."""
        with open(self.skater_cache_file, 'r') as f:
            cache_content = json.load(f)
        
        edge_data = {}
        for key, records in cache_content.items():
            edge_data[key] = pd.DataFrame(records)
        
        return edge_data
    
    def get_edge_stats(self, player_ids: List[int] = None,
                        force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get skater Edge stats, using cache if available.

        Args:
            player_ids: Optional list of player IDs to fetch
            force_refresh: If True, fetch fresh data even if cache exists

        Returns:
            Dict mapping stat category to DataFrame
        """
        # Check cache first
        if not force_refresh and self.is_cache_valid('skater'):
            age = self.get_cache_age_hours('skater')
            print(f"✅ Using cached skater Edge stats (age: {age:.1f} hours)")
            return self._load_skater_cache()

        # Fetch fresh data
        edge_data = self._fetch_skater_edge_from_api(player_ids)

        if edge_data:
            self._save_skater_cache(edge_data)

        return edge_data
    
    # ==================== GOALIE EDGE STATS ====================
    
    def _fetch_goalie_edge_from_api(self) -> pd.DataFrame:
        """
        Fetch goalie Edge stats directly from NHL API.
        
        Goalie Edge endpoints (new in 2024-25):
        - /v1/edge/goalie-rankings/{season}/{game-type}
        - /v1/edge/goalie-detail/{player-id}/{season}/{game-type}
        
        Returns:
            DataFrame with goalie Edge metrics
        """
        print("🔄 Fetching goalie Edge stats from NHL API...")
        
        all_goalies = []
        
        try:
            # Get goalie list from stats API
            url = f"{NHL_API_STATS_BASE}/goalie/summary?limit=100&cayenneExp=seasonId={CURRENT_SEASON}"
            print(f"  → Fetching goalie list...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            time.sleep(self.rate_limit_delay)

            # Extract goalie IDs from summary
            goalie_ids = []
            for g in data.get('data', []):
                pid = g.get('playerId')
                if pid:
                    goalie_ids.append({
                        'player_id': int(pid),
                        'name': g.get('goalieFullName', ''),
                        'team': g.get('teamAbbrevs', ''),
                    })
            
            print(f"    ✓ Found {len(goalie_ids)} goalies")
            
            # Fetch Edge detail for each goalie
            print(f"  → Fetching Edge details for goalies...")
            for i, goalie in enumerate(goalie_ids[:50]):  # Limit to top 50 for speed
                try:
                    pid = goalie['player_id']
                    edge_url = f"{NHL_API_WEB_BASE}/v1/edge/goalie-detail/{pid}/{CURRENT_SEASON}/2"
                    
                    resp = requests.get(edge_url, timeout=15)
                    time.sleep(self.rate_limit_delay)
                    
                    if resp.status_code == 200:
                        edge_data = resp.json()

                        # Extract key metrics
                        goalie_stats = {
                            'player_id': pid,
                            'name': goalie['name'],
                            'team': goalie['team'],
                        }

                        # Shot location save percentages (it's a list, not dict)
                        shot_locs = edge_data.get('shotLocationSummary', [])
                        if isinstance(shot_locs, list):
                            for loc in shot_locs:
                                loc_code = loc.get('locationCode', '')
                                if loc_code == 'high':
                                    goalie_stats['hd_save_pct'] = loc.get('savePctg')
                                    goalie_stats['hd_saves'] = loc.get('saves')
                                    goalie_stats['hd_goals_against'] = loc.get('goalsAgainst')
                                elif loc_code == 'mid':
                                    goalie_stats['mid_save_pct'] = loc.get('savePctg')
                                elif loc_code == 'low':
                                    goalie_stats['low_save_pct'] = loc.get('savePctg')
                                elif loc_code == 'all':
                                    goalie_stats['overall_save_pct'] = loc.get('savePctg')

                        # Stats section (contains 5v5 and consistency data)
                        stats = edge_data.get('stats', {})
                        if stats:
                            goalie_stats['five_v_five_save_pct'] = stats.get('fiveOnFiveSavePctg')
                            goalie_stats['games_played'] = stats.get('gamesPlayed')
                            # Look for consistency in gameByGame if available
                            game_by_game = edge_data.get('gameByGame', [])
                            if game_by_game:
                                games_above_900 = sum(1 for g in game_by_game
                                                       if g.get('savePctg', 0) >= 0.900)
                                total_games = len(game_by_game)
                                if total_games > 0:
                                    goalie_stats['games_above_900'] = games_above_900
                                    goalie_stats['pct_games_above_900'] = games_above_900 / total_games

                        all_goalies.append(goalie_stats)
                        
                        if (i + 1) % 10 == 0:
                            print(f"    Processed {i + 1}/{min(len(goalie_ids), 50)} goalies")
                    
                except Exception as e:
                    # Skip individual goalie errors
                    continue
            
            print(f"    ✓ Got Edge data for {len(all_goalies)} goalies")
            
        except Exception as e:
            print(f"⚠️ Error fetching goalie Edge stats: {e}")
        
        return pd.DataFrame(all_goalies) if all_goalies else pd.DataFrame()
    
    def _save_goalie_cache(self, goalie_df: pd.DataFrame):
        """Save goalie Edge data to cache file."""
        with open(self.goalie_cache_file, 'w') as f:
            json.dump(goalie_df.to_dict(orient='records'), f)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata['last_fetch_date_goalie'] = self.today
        metadata['last_fetch_timestamp_goalie'] = datetime.now().isoformat()
        metadata['goalie_count'] = len(goalie_df)
        self._save_metadata(metadata)
        
        print(f"✅ Cached goalie Edge stats to {self.goalie_cache_file}")
    
    def _load_goalie_cache(self) -> pd.DataFrame:
        """Load goalie Edge data from cache file."""
        with open(self.goalie_cache_file, 'r') as f:
            records = json.load(f)
        return pd.DataFrame(records)
    
    def get_goalie_edge_stats(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get goalie Edge stats, using cache if available.
        
        Args:
            force_refresh: If True, fetch fresh data even if cache exists
            
        Returns:
            DataFrame with goalie Edge metrics
        """
        # Check cache first
        if not force_refresh and self.is_cache_valid('goalie'):
            age = self.get_cache_age_hours('goalie')
            print(f"✅ Using cached goalie Edge stats (age: {age:.1f} hours)")
            return self._load_goalie_cache()
        
        # Fetch fresh data
        goalie_df = self._fetch_goalie_edge_from_api()
        
        if not goalie_df.empty:
            self._save_goalie_cache(goalie_df)
        
        return goalie_df
    
    # ==================== UNIFIED LOOKUP ====================
    
    def build_edge_lookup(self, edge_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Build a unified lookup DataFrame with all skater Edge metrics.

        Returns:
            DataFrame with one row per player, columns for each Edge metric
        """
        if edge_data is None:
            edge_data = self.get_edge_stats()

        if not edge_data:
            return pd.DataFrame()

        # Get skaters DataFrame (new format uses 'skaters' key)
        if 'skaters' in edge_data and not edge_data['skaters'].empty:
            base_df = edge_data['skaters'].copy()
        elif 'speed' in edge_data and not edge_data['speed'].empty:
            # Legacy format support
            base_df = edge_data['speed'].copy()
        else:
            # Use first available category
            for cat, df in edge_data.items():
                if df is not None and not df.empty:
                    base_df = df.copy()
                    break
            else:
                return pd.DataFrame()

        # Identify name column
        name_cols = ['player_name', 'skaterFullName', 'playerName', 'fullName', 'name']
        name_col = None
        for col in name_cols:
            if col in base_df.columns:
                name_col = col
                break

        if name_col and name_col != 'Player':
            base_df = base_df.rename(columns={name_col: 'Player'})

        return base_df
    
    def cleanup_old_caches(self, keep_days: int = 7):
        """Remove cache files older than keep_days."""
        cutoff = datetime.now() - pd.Timedelta(days=keep_days)
        
        for pattern in ["edge_stats_*.json", "goalie_edge_stats_*.json"]:
            for cache_file in self.cache_dir.glob(pattern):
                try:
                    # Extract date from filename
                    file_date_str = cache_file.stem.split('_')[-1]
                    if len(file_date_str) == 10:  # YYYY-MM-DD format
                        file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                        if file_date < cutoff:
                            cache_file.unlink()
                            print(f"🗑️ Removed old cache: {cache_file.name}")
                except (ValueError, IndexError):
                    continue


# ==================== CONVENIENCE FUNCTIONS ====================

def get_cached_edge_stats(player_ids: List[int] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
    """
    Get skater Edge stats as a single DataFrame, using cache when possible.

    This is the main function to call from edge_stats.py.

    Args:
        player_ids: Optional list of player IDs to fetch
        force_refresh: If True, fetch fresh data from API

    Returns:
        DataFrame with all Edge metrics per player
    """
    cache = EdgeStatsCache()
    edge_data = cache.get_edge_stats(player_ids=player_ids, force_refresh=force_refresh)
    return cache.build_edge_lookup(edge_data)


def get_cached_goalie_edge_stats(force_refresh: bool = False) -> pd.DataFrame:
    """
    Get goalie Edge stats as a DataFrame, using cache when possible.
    
    Args:
        force_refresh: If True, fetch fresh data from API
        
    Returns:
        DataFrame with goalie Edge metrics (HD save %, mid save %, etc.)
    """
    cache = EdgeStatsCache()
    return cache.get_goalie_edge_stats(force_refresh=force_refresh)


def check_edge_update_time() -> str:
    """
    Estimate when Edge stats were last updated by NHL.
    
    Returns:
        Estimated update time description
    """
    return "Estimated: ~6 AM ET daily (after all games complete)"


# ==================== TEST ====================

if __name__ == "__main__":
    print("=" * 60)
    print("EDGE STATS CACHE TEST (v2 with Goalie Edge)")
    print("=" * 60)
    
    cache = EdgeStatsCache()
    
    print(f"\nCache directory: {cache.cache_dir}")
    print(f"Today's date: {cache.today}")
    print(f"Skater cache valid: {cache.is_cache_valid('skater')}")
    print(f"Goalie cache valid: {cache.is_cache_valid('goalie')}")
    
    # Test skater Edge
    print("\n" + "=" * 60)
    print("SKATER EDGE STATS")
    print("=" * 60)
    skater_df = get_cached_edge_stats(force_refresh=False)
    if not skater_df.empty:
        print(f"✅ Got skater Edge data for {len(skater_df)} players")
        print(f"Columns: {list(skater_df.columns)[:8]}...")
    else:
        print("⚠️ No skater Edge data available")
    
    # Test goalie Edge
    print("\n" + "=" * 60)
    print("GOALIE EDGE STATS")
    print("=" * 60)
    goalie_df = get_cached_goalie_edge_stats(force_refresh=False)
    if not goalie_df.empty:
        print(f"✅ Got goalie Edge data for {len(goalie_df)} goalies")
        print(f"Columns: {list(goalie_df.columns)}")
        print("\nTop 5 by HD Save %:")
        if 'hd_save_pct' in goalie_df.columns:
            top_hd = goalie_df.nlargest(5, 'hd_save_pct')[['name', 'team', 'hd_save_pct', 'hd_saves']]
            print(top_hd.to_string(index=False))
    else:
        print("⚠️ No goalie Edge data available")
