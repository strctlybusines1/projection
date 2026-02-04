"""
NHL Edge Stats Caching Module

Edge stats are cumulative season stats that update once daily (usually overnight).
This module caches Edge data to avoid redundant API calls during the same day.

Usage:
    from edge_cache import EdgeStatsCache

    cache = EdgeStatsCache()
    edge_data = cache.get_edge_stats(player_ids)  # Fetches if needed, else returns cached

    # Force refresh (e.g., if you know NHL updated)
    edge_data = cache.get_edge_stats(player_ids, force_refresh=True)

Cache Location: projection/cache/edge_stats_{date}.json
"""

import os
import json
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
    print("Warning: nhl-api-py not installed. Edge stats will use fallback.")


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
        self.cache_file = self.cache_dir / f"edge_stats_{self.today}.json"
        self.metadata_file = self.cache_dir / "edge_metadata.json"

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

    def is_cache_valid(self) -> bool:
        """
        Check if today's cache exists and is valid.

        Returns:
            True if cache exists for today and appears valid
        """
        if not self.cache_file.exists():
            return False

        # Check file is not empty
        if self.cache_file.stat().st_size < 100:
            return False

        # Check metadata for fetch time
        metadata = self._load_metadata()
        last_fetch = metadata.get('last_fetch_date')

        return last_fetch == self.today

    def get_cache_age_hours(self) -> Optional[float]:
        """Get how old the cache is in hours."""
        metadata = self._load_metadata()
        last_fetch_time = metadata.get('last_fetch_timestamp')

        if not last_fetch_time:
            return None

        last_dt = datetime.fromisoformat(last_fetch_time)
        age = datetime.now() - last_dt
        return age.total_seconds() / 3600

    def _fetch_edge_stats_from_api(self, player_ids: List[int]) -> pd.DataFrame:
        """
        Fetch Edge stats from NHL API for specified players.

        Uses per-player API calls with rate limiting.

        Args:
            player_ids: List of NHL player IDs to fetch

        Returns:
            DataFrame with Edge stats for all players
        """
        if not HAS_NHLPY:
            print("Warning: nhl-api-py not available, returning empty Edge data")
            return pd.DataFrame()

        print(f"Fetching Edge stats from NHL API for {len(player_ids)} players...")
        print("  (This may take 2-3 minutes due to rate limiting)")

        client = NHLClient()
        results = []

        for i, pid in enumerate(player_ids):
            if (i + 1) % 50 == 0:
                print(f"  Edge progress: {i + 1}/{len(player_ids)}")

            try:
                # Get player edge detail
                detail = client.edge.skater_detail(player_id=str(pid))
                time.sleep(0.3)  # Rate limit

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
                    time.sleep(0.3)  # Rate limit

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

            except Exception as e:
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        print(f"  Fetched Edge data for {len(df)} players")

        return df

    def _save_to_cache(self, edge_df: pd.DataFrame):
        """Save Edge data to cache file."""
        cache_content = edge_df.to_dict(orient='records')

        with open(self.cache_file, 'w') as f:
            json.dump(cache_content, f)

        # Update metadata
        metadata = self._load_metadata()
        metadata['last_fetch_date'] = self.today
        metadata['last_fetch_timestamp'] = datetime.now().isoformat()
        metadata['player_count'] = len(edge_df)
        self._save_metadata(metadata)

        print(f"  Cached Edge stats to {self.cache_file}")

    def _load_from_cache(self) -> pd.DataFrame:
        """Load Edge data from cache file."""
        with open(self.cache_file, 'r') as f:
            cache_content = json.load(f)

        return pd.DataFrame(cache_content)

    def get_edge_stats(self, player_ids: List[int] = None,
                        force_refresh: bool = False) -> pd.DataFrame:
        """
        Get Edge stats, using cache if available.

        Args:
            player_ids: List of player IDs to fetch (required if cache miss)
            force_refresh: If True, fetch fresh data even if cache exists

        Returns:
            DataFrame with Edge stats
        """
        # Check cache first
        if not force_refresh and self.is_cache_valid():
            age = self.get_cache_age_hours()
            print(f"  Using cached Edge stats (age: {age:.1f} hours)")
            return self._load_from_cache()

        # Need to fetch - require player_ids
        if not player_ids:
            print("  Warning: No player IDs provided and cache invalid")
            return pd.DataFrame()

        # Fetch fresh data
        edge_df = self._fetch_edge_stats_from_api(player_ids)

        if not edge_df.empty:
            self._save_to_cache(edge_df)

        return edge_df

    def cleanup_old_caches(self, keep_days: int = 7):
        """Remove cache files older than keep_days."""
        cutoff = datetime.now() - pd.Timedelta(days=keep_days)

        for cache_file in self.cache_dir.glob("edge_stats_*.json"):
            try:
                file_date_str = cache_file.stem.replace("edge_stats_", "")
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")

                if file_date < cutoff:
                    cache_file.unlink()
                    print(f"  Removed old cache: {cache_file.name}")
            except ValueError:
                continue


def check_edge_update_time() -> Optional[str]:
    """
    Try to determine when Edge stats were last updated by NHL.

    Returns:
        Estimated update time or None if unknown
    """
    return "Estimated: ~6 AM ET daily (after all games complete)"


# Convenience function for integration with existing code
def get_cached_edge_stats(player_ids: List[int] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
    """
    Get Edge stats as a DataFrame, using cache when possible.

    This is the main function to call from data_pipeline.py.

    Args:
        player_ids: List of player IDs to fetch (required if cache miss)
        force_refresh: If True, fetch fresh data from API

    Returns:
        DataFrame with Edge stats per player
    """
    cache = EdgeStatsCache()
    return cache.get_edge_stats(player_ids=player_ids, force_refresh=force_refresh)


if __name__ == "__main__":
    # Test the cache
    print("=" * 60)
    print("EDGE STATS CACHE TEST")
    print("=" * 60)

    cache = EdgeStatsCache()

    print(f"\nCache directory: {cache.cache_dir}")
    print(f"Today's date: {cache.today}")
    print(f"Cache file: {cache.cache_file}")
    print(f"Cache valid: {cache.is_cache_valid()}")

    age = cache.get_cache_age_hours()
    if age:
        print(f"Cache age: {age:.1f} hours")

    print(f"\nEdge update info: {check_edge_update_time()}")

    # Test with a few player IDs if cache is invalid
    if not cache.is_cache_valid():
        print("\n" + "=" * 60)
        print("Testing with sample players (Connor McDavid, Leon Draisaitl)...")
        test_ids = [8478402, 8478483]  # McDavid, Draisaitl
        edge_df = cache.get_edge_stats(player_ids=test_ids)

        if not edge_df.empty:
            print(f"\nGot Edge data for {len(edge_df)} players")
            print(edge_df.to_string())
    else:
        print("\n" + "=" * 60)
        edge_df = cache.get_edge_stats()
        print(f"\nLoaded {len(edge_df)} players from cache")
