"""
NHL Edge Stats Caching Module (v2)

Caches Edge stats to avoid redundant API calls. Edge stats update once daily.

Skater Edge: Speed, bursts, OZ time (via nhlpy bulk stats)
Goalie Edge: EV save %, quality starts %, PP/SH save % (via NHL Stats API)

Usage:
    from edge_cache import get_cached_edge_stats, get_cached_goalie_edge_stats
    
    skater_df = get_cached_edge_stats()           # Cached skater Edge
    goalie_df = get_cached_goalie_edge_stats()    # Cached goalie Edge

Cache files: cache/edge_stats_{date}.json, cache/goalie_edge_stats_{date}.json
"""

import json
import requests
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# NHL API endpoints
NHL_STATS_API = "https://api.nhle.com/stats/rest/en"
CURRENT_SEASON = "20252026"

# Try nhlpy for skater Edge
try:
    from nhlpy import NHLClient
    HAS_NHLPY = True
except ImportError:
    HAS_NHLPY = False


class EdgeStatsCache:
    """Caches NHL Edge tracking stats to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.today = date.today().strftime("%Y-%m-%d")
        self.skater_cache_file = self.cache_dir / f"edge_stats_{self.today}.json"
        self.goalie_cache_file = self.cache_dir / f"goalie_edge_stats_{self.today}.json"
        self.metadata_file = self.cache_dir / "edge_metadata.json"
        
        self.rate_limit_delay = 0.3
        
    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def is_cache_valid(self, cache_type: str = 'skater') -> bool:
        cache_file = self.skater_cache_file if cache_type == 'skater' else self.goalie_cache_file
        if not cache_file.exists():
            return False
        if cache_file.stat().st_size < 100:
            return False
        metadata = self._load_metadata()
        return metadata.get(f'last_fetch_date_{cache_type}') == self.today
    
    def get_cache_age_hours(self, cache_type: str = 'skater') -> Optional[float]:
        metadata = self._load_metadata()
        ts = metadata.get(f'last_fetch_timestamp_{cache_type}')
        if not ts:
            return None
        return (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 3600
    
    # ==================== SKATER EDGE ====================
    
    def _fetch_skater_edge(self) -> Dict[str, pd.DataFrame]:
        """Fetch skater Edge stats via nhlpy."""
        if not HAS_NHLPY:
            print("⚠️ nhl-api-py not installed")
            return {}
        
        print("🔄 Fetching skater Edge stats...")
        client = NHLClient()
        edge_data = {}
        
        try:
            # Use skater landing which has aggregated Edge metrics
            # This is more reliable than bulk endpoints
            print("  → Fetching skater stats summary...")
            
            # Get all skaters from stats API first
            url = f"{NHL_STATS_API}/skater/summary?limit=-1&cayenneExp=seasonId={CURRENT_SEASON} and gameTypeId=2"
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data:
                    edge_data['summary'] = pd.DataFrame(data['data'])
                    print(f"    ✓ Got {len(edge_data['summary'])} skaters")
            
            time.sleep(self.rate_limit_delay)
            
            # Get time on ice data
            print("  → Fetching time on ice stats...")
            toi_url = f"{NHL_STATS_API}/skater/timeonice?limit=-1&cayenneExp=seasonId={CURRENT_SEASON} and gameTypeId=2"
            toi_resp = requests.get(toi_url, timeout=60)
            if toi_resp.status_code == 200:
                toi_data = toi_resp.json()
                if 'data' in toi_data:
                    edge_data['timeonice'] = pd.DataFrame(toi_data['data'])
                    print(f"    ✓ Got {len(edge_data['timeonice'])} skaters")
            
        except Exception as e:
            print(f"⚠️ Error fetching skater Edge: {e}")
        
        return edge_data
    
    def _save_skater_cache(self, edge_data: Dict[str, pd.DataFrame]):
        cache_content = {k: df.to_dict('records') for k, df in edge_data.items()}
        with open(self.skater_cache_file, 'w') as f:
            json.dump(cache_content, f)
        
        metadata = self._load_metadata()
        metadata['last_fetch_date_skater'] = self.today
        metadata['last_fetch_timestamp_skater'] = datetime.now().isoformat()
        metadata['skater_count'] = sum(len(df) for df in edge_data.values())
        self._save_metadata(metadata)
        print(f"✅ Cached skater Edge to {self.skater_cache_file}")
    
    def _load_skater_cache(self) -> Dict[str, pd.DataFrame]:
        with open(self.skater_cache_file, 'r') as f:
            content = json.load(f)
        return {k: pd.DataFrame(v) for k, v in content.items()}
    
    def get_edge_stats(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        if not force_refresh and self.is_cache_valid('skater'):
            age = self.get_cache_age_hours('skater')
            print(f"✅ Using cached skater Edge (age: {age:.1f}h)")
            return self._load_skater_cache()
        
        edge_data = self._fetch_skater_edge()
        if edge_data:
            self._save_skater_cache(edge_data)
        return edge_data
    
    # ==================== GOALIE EDGE ====================
    
    def _fetch_goalie_edge(self) -> pd.DataFrame:
        """Fetch goalie stats from NHL Stats API."""
        print("🔄 Fetching goalie Edge stats from NHL API...")
        
        try:
            # Summary stats
            print("  → Fetching summary...")
            summary_url = f"{NHL_STATS_API}/goalie/summary?limit=-1&cayenneExp=seasonId={CURRENT_SEASON} and gameTypeId=2"
            summary = requests.get(summary_url, timeout=30).json()
            summary_df = pd.DataFrame(summary['data'])
            print(f"    ✓ {len(summary_df)} goalies")
            
            time.sleep(self.rate_limit_delay)
            
            # Advanced stats (quality starts)
            print("  → Fetching advanced (quality starts)...")
            advanced_url = f"{NHL_STATS_API}/goalie/advanced?limit=-1&cayenneExp=seasonId={CURRENT_SEASON} and gameTypeId=2"
            advanced = requests.get(advanced_url, timeout=30).json()
            advanced_df = pd.DataFrame(advanced['data'])
            
            time.sleep(self.rate_limit_delay)
            
            # Saves by strength
            print("  → Fetching saves by strength...")
            strength_url = f"{NHL_STATS_API}/goalie/savesByStrength?limit=-1&cayenneExp=seasonId={CURRENT_SEASON} and gameTypeId=2"
            strength = requests.get(strength_url, timeout=30).json()
            strength_df = pd.DataFrame(strength['data'])
            
            # Merge
            goalies = summary_df.merge(
                advanced_df[['playerId', 'qualityStart', 'qualityStartsPct', 'goalsFor', 'goalsForAverage']],
                on='playerId', how='left'
            ).merge(
                strength_df[['playerId', 'evSavePct', 'ppSavePct', 'shSavePct']],
                on='playerId', how='left'
            )
            
            # Rename for consistency
            goalies = goalies.rename(columns={
                'goalieFullName': 'name',
                'teamAbbrevs': 'team',
                'savePct': 'save_pct',
                'goalsAgainstAverage': 'gaa',
                'qualityStartsPct': 'quality_starts_pct',
                'evSavePct': 'ev_save_pct',
                'ppSavePct': 'pp_save_pct',
                'shSavePct': 'sh_save_pct',
            })
            
            print(f"    ✓ Merged {len(goalies)} goalies with {len(goalies.columns)} columns")
            return goalies
            
        except Exception as e:
            print(f"⚠️ Error fetching goalie Edge: {e}")
            return pd.DataFrame()
    
    def _save_goalie_cache(self, goalie_df: pd.DataFrame):
        with open(self.goalie_cache_file, 'w') as f:
            json.dump(goalie_df.to_dict('records'), f)
        
        metadata = self._load_metadata()
        metadata['last_fetch_date_goalie'] = self.today
        metadata['last_fetch_timestamp_goalie'] = datetime.now().isoformat()
        metadata['goalie_count'] = len(goalie_df)
        self._save_metadata(metadata)
        print(f"✅ Cached goalie Edge to {self.goalie_cache_file}")
    
    def _load_goalie_cache(self) -> pd.DataFrame:
        with open(self.goalie_cache_file, 'r') as f:
            return pd.DataFrame(json.load(f))
    
    def get_goalie_edge_stats(self, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.is_cache_valid('goalie'):
            age = self.get_cache_age_hours('goalie')
            print(f"✅ Using cached goalie Edge (age: {age:.1f}h)")
            return self._load_goalie_cache()
        
        goalie_df = self._fetch_goalie_edge()
        if not goalie_df.empty:
            self._save_goalie_cache(goalie_df)
        return goalie_df
    
    # ==================== UNIFIED LOOKUP ====================
    
    def build_edge_lookup(self, edge_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Build unified skater Edge lookup."""
        if edge_data is None:
            edge_data = self.get_edge_stats()

        if not edge_data:
            return pd.DataFrame()

        # Handle old cache format: {'skaters': [...]}
        if 'skaters' in edge_data:
            df = edge_data['skaters']
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Rename columns for compatibility
                if 'player_name' in df.columns:
                    df = df.rename(columns={'player_name': 'Player'})
                return df
            return pd.DataFrame()

        # Handle new format: {'summary': df, 'timeonice': df}
        if 'summary' in edge_data:
            df = edge_data['summary'].copy()
            df = df.rename(columns={'skaterFullName': 'Player'})

            # Merge TOI data if available
            if 'timeonice' in edge_data:
                toi = edge_data['timeonice']
                if 'skaterFullName' in toi.columns:
                    toi = toi.rename(columns={'skaterFullName': 'Player'})
                    # Get useful TOI columns
                    toi_cols = ['Player'] + [c for c in toi.columns if c not in df.columns and c != 'Player']
                    df = df.merge(toi[toi_cols], on='Player', how='left')

            return df

        return pd.DataFrame()
    
    def cleanup_old_caches(self, keep_days: int = 7):
        """Remove old cache files."""
        cutoff = datetime.now() - pd.Timedelta(days=keep_days)
        for pattern in ["edge_stats_*.json", "goalie_edge_stats_*.json"]:
            for f in self.cache_dir.glob(pattern):
                try:
                    date_str = f.stem.split('_')[-1]
                    if len(date_str) == 10:
                        fdate = datetime.strptime(date_str, "%Y-%m-%d")
                        if fdate < cutoff:
                            f.unlink()
                            print(f"🗑️ Removed: {f.name}")
                except:
                    pass


# ==================== CONVENIENCE FUNCTIONS ====================

def get_cached_edge_stats(force_refresh: bool = False, player_ids: list = None) -> pd.DataFrame:
    """Get skater Edge stats (cached).

    Args:
        force_refresh: Force fetch from API even if cache exists
        player_ids: Optional list of player IDs (ignored - caches all players)
    """
    cache = EdgeStatsCache()
    edge_data = cache.get_edge_stats(force_refresh=force_refresh)
    return cache.build_edge_lookup(edge_data)


def get_cached_goalie_edge_stats(force_refresh: bool = False) -> pd.DataFrame:
    """Get goalie Edge stats (cached)."""
    cache = EdgeStatsCache()
    return cache.get_goalie_edge_stats(force_refresh=force_refresh)


# ==================== TEST ====================

if __name__ == "__main__":
    print("=" * 60)
    print("EDGE CACHE TEST")
    print("=" * 60)
    
    cache = EdgeStatsCache()
    print(f"Cache dir: {cache.cache_dir}")
    print(f"Today: {cache.today}")
    
    # Test goalie Edge
    print("\n" + "=" * 60)
    print("GOALIE EDGE STATS")
    print("=" * 60)
    goalie_df = get_cached_goalie_edge_stats(force_refresh=True)
    
    if not goalie_df.empty:
        print(f"\n✅ Got {len(goalie_df)} goalies")
        print(f"Columns: {list(goalie_df.columns)}")
        
        print("\nTop 10 by GP:")
        cols = ['name', 'team', 'gamesPlayed', 'save_pct', 'ev_save_pct', 'quality_starts_pct']
        cols = [c for c in cols if c in goalie_df.columns]
        top = goalie_df.nlargest(10, 'gamesPlayed')[cols]
        print(top.to_string(index=False))
    
    # Test skater Edge
    print("\n" + "=" * 60)
    print("SKATER EDGE STATS")
    print("=" * 60)
    skater_df = get_cached_edge_stats(force_refresh=True)
    
    if not skater_df.empty:
        print(f"\n✅ Got {len(skater_df)} skaters")
        print(f"Columns: {list(skater_df.columns)[:10]}...")
