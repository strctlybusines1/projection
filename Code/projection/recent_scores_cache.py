"""
Recent Game Scores Caching Module

Recent scores (last 1/3/5 game averages) are player season data that updates once daily.
This module caches recent scores to avoid redundant API calls.

Usage:
    from recent_scores_cache import get_cached_recent_scores
    
    # First call fetches, subsequent calls use cache
    recent_scores = get_cached_recent_scores(player_ids, pipeline)
    
    # Force refresh
    recent_scores = get_cached_recent_scores(player_ids, pipeline, force_refresh=True)

Cache file: cache/recent_scores_{date}.json
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional


class RecentScoresCache:
    """Caches recent game scores to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.today = date.today().strftime("%Y-%m-%d")
        self.cache_file = self.cache_dir / f"recent_scores_{self.today}.json"
        self.metadata_file = self.cache_dir / "recent_scores_metadata.json"
    
    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def is_cache_valid(self) -> bool:
        """Check if today's cache exists and is valid."""
        if not self.cache_file.exists():
            return False
        if self.cache_file.stat().st_size < 100:
            return False
        metadata = self._load_metadata()
        return metadata.get('last_fetch_date') == self.today
    
    def get_cache_age_hours(self) -> Optional[float]:
        """Get cache age in hours."""
        metadata = self._load_metadata()
        ts = metadata.get('last_fetch_timestamp')
        if not ts:
            return None
        return (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 3600
    
    def get_cached_player_ids(self) -> List[int]:
        """Get list of player IDs already in cache."""
        if not self.cache_file.exists():
            return []
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            return [int(k) for k in data.keys()]
        except:
            return []
    
    def load_cache(self) -> Dict[int, Dict]:
        """Load cached recent scores."""
        if not self.cache_file.exists():
            return {}
        with open(self.cache_file, 'r') as f:
            data = json.load(f)
        # Convert string keys back to int
        return {int(k): v for k, v in data.items()}
    
    def save_cache(self, recent_scores: Dict[int, Dict]):
        """Save recent scores to cache."""
        # Convert int keys to string for JSON
        data = {str(k): v for k, v in recent_scores.items()}
        with open(self.cache_file, 'w') as f:
            json.dump(data, f)
        
        metadata = self._load_metadata()
        metadata['last_fetch_date'] = self.today
        metadata['last_fetch_timestamp'] = datetime.now().isoformat()
        metadata['player_count'] = len(recent_scores)
        self._save_metadata(metadata)
    
    def update_cache(self, new_scores: Dict[int, Dict]):
        """Add new scores to existing cache."""
        existing = self.load_cache()
        existing.update(new_scores)
        self.save_cache(existing)
    
    def cleanup_old_caches(self, keep_days: int = 7):
        """Remove old cache files."""
        import pandas as pd
        cutoff = datetime.now() - pd.Timedelta(days=keep_days)
        for f in self.cache_dir.glob("recent_scores_*.json"):
            try:
                date_str = f.stem.replace("recent_scores_", "")
                if len(date_str) == 10:
                    fdate = datetime.strptime(date_str, "%Y-%m-%d")
                    if fdate < cutoff:
                        f.unlink()
                        print(f"🗑️ Removed: {f.name}")
            except:
                pass


def get_cached_recent_scores(player_ids: List[int], 
                              pipeline, 
                              force_refresh: bool = False) -> Dict[int, Dict]:
    """
    Get recent game scores with caching.
    
    Args:
        player_ids: List of player IDs to fetch
        pipeline: NHLDataPipeline instance with fetch_recent_game_scores method
        force_refresh: If True, fetch fresh data even if cached
        
    Returns:
        Dict mapping player_id to recent score data
    """
    cache = RecentScoresCache()
    
    # If force refresh, fetch all
    if force_refresh:
        print(f"🔄 Fetching recent scores for {len(player_ids)} players (force refresh)...")
        recent_scores = pipeline.fetch_recent_game_scores(player_ids)
        cache.save_cache(recent_scores)
        print(f"✅ Cached recent scores for {len(recent_scores)} players")
        return recent_scores
    
    # Check cache
    if cache.is_cache_valid():
        cached_scores = cache.load_cache()
        cached_ids = set(cached_scores.keys())
        needed_ids = [pid for pid in player_ids if pid not in cached_ids]
        
        if not needed_ids:
            age = cache.get_cache_age_hours()
            print(f"✅ Using cached recent scores for {len(player_ids)} players (age: {age:.1f}h)")
            # Return only requested players
            return {pid: cached_scores[pid] for pid in player_ids if pid in cached_scores}
        
        # Fetch only missing players
        print(f"✅ Using cache for {len(cached_ids)} players, fetching {len(needed_ids)} new...")
        new_scores = pipeline.fetch_recent_game_scores(needed_ids)
        cache.update_cache(new_scores)
        
        # Combine cached + new
        all_scores = cached_scores.copy()
        all_scores.update(new_scores)
        return {pid: all_scores[pid] for pid in player_ids if pid in all_scores}
    
    # No valid cache, fetch all
    print(f"🔄 Fetching recent scores for {len(player_ids)} players...")
    recent_scores = pipeline.fetch_recent_game_scores(player_ids)
    cache.save_cache(recent_scores)
    print(f"✅ Cached recent scores for {len(recent_scores)} players")
    return recent_scores


# Test
if __name__ == "__main__":
    cache = RecentScoresCache()
    print(f"Cache dir: {cache.cache_dir}")
    print(f"Cache file: {cache.cache_file}")
    print(f"Cache valid: {cache.is_cache_valid()}")
    
    if cache.is_cache_valid():
        cached = cache.load_cache()
        print(f"Cached players: {len(cached)}")
