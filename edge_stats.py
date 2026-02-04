"""
NHL Edge Stats Integration for DFS Projections (v2 - with Goalie Edge).

Fetches advanced player tracking data:
- Skating speed (max, bursts over 20/22 mph)
- Offensive zone time percentage
- Zone starts (OZ/DZ/NZ %)
- Distance skated per game
- Shot speed

NEW IN V2:
- Goalie Edge stats: High-danger SV%, midrange SV%, shot location details
- Daily caching for both skaters and goalies
- Goalie projection boosts based on HD save %

These metrics can boost projections for players with elite underlying metrics.

Supports caching to avoid redundant API calls (Edge stats update once daily).
Use force_refresh=True to fetch fresh data from the API.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from nhlpy import NHLClient
import time

# Import caching module
try:
    from edge_cache import (
        EdgeStatsCache, 
        get_cached_edge_stats, 
        get_cached_goalie_edge_stats
    )
    HAS_EDGE_CACHE = True
except ImportError:
    HAS_EDGE_CACHE = False
    print("Warning: edge_cache.py not found. Edge caching disabled.")


class EdgeStatsClient:
    """
    Client for fetching NHL Edge tracking data.

    Edge data available since 2021-22 season.
    """

    CURRENT_SEASON = "20252026"
    GAME_TYPE_REGULAR = 2

    # Percentile thresholds for projection boosts
    ELITE_PERCENTILE = 0.90      # Top 10%
    ABOVE_AVG_PERCENTILE = 0.65  # Top 35%

    # Projection boost factors (multiplicative)
    # Calibrated from 1,180-observation backtest (Jan 22 - Feb 2, 2026)
    # Raw data: elite OZ +55%, elite speed +9%, elite bursts +24%
    # Using ~20% of raw to stay conservative (avoid confounding with player quality)
    EDGE_BOOST_FACTORS = {
        'elite_oz_time': 1.10,      # +10% for elite OZ time (strongest predictor, r=0.18)
        'elite_speed': 1.02,        # +2% for elite skating speed (weak correlation, r=0.07)
        'elite_bursts': 1.05,       # +5% for elite burst count (solid predictor, r=0.15)
        'above_avg_oz_time': 1.04,  # +4% for above-avg OZ time
        'above_avg_speed': 1.01,    # +1% for above-avg speed
    }
    
    # Goalie Edge boost factors (NEW)
    GOALIE_EDGE_BOOST_FACTORS = {
        'elite_hd_save_pct': 1.08,     # +8% for elite high-danger SV% (top 10%)
        'above_avg_hd_save_pct': 1.04, # +4% for above-avg HD SV% (top 35%)
        'elite_consistency': 1.05,     # +5% for 80%+ games above .900
        'poor_hd_save_pct': 0.94,      # -6% for bottom 25% HD SV%
    }
    
    # Goalie Edge thresholds
    ELITE_HD_SAVE_PCT = 0.850       # Top goalies save 85%+ of HD shots
    ABOVE_AVG_HD_SAVE_PCT = 0.820   # Above average
    POOR_HD_SAVE_PCT = 0.780        # Below average
    ELITE_CONSISTENCY = 0.80        # 80%+ games above .900 SV%

    def __init__(self, rate_limit_delay: float = 0.5):
        self.client = NHLClient()
        self.rate_limit_delay = rate_limit_delay
        self._cache: Dict[str, dict] = {}

    def get_skater_edge_summary(self, player_id: int, season: str = None) -> Optional[dict]:
        """
        Get summary of key Edge metrics for a skater.

        Returns dict with:
            - max_speed_mph: Top skating speed
            - speed_percentile: League percentile for speed
            - bursts_over_20: Count of 20+ mph bursts
            - bursts_percentile: League percentile for bursts
            - oz_time_pct: Offensive zone time %
            - oz_time_percentile: League percentile for OZ time
            - oz_starts_pct: Offensive zone starts %
            - shot_speed_mph: Top shot speed
            - shot_speed_percentile: League percentile for shot speed
        """
        season = season or self.CURRENT_SEASON
        cache_key = f"{player_id}_{season}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Get full skater detail (contains most metrics)
            detail = self.client.edge.skater_detail(
                player_id=str(player_id),
                season=season
            )
            time.sleep(self.rate_limit_delay)

            if not detail:
                return None

            summary = {
                'player_id': player_id,
                'player_name': f"{detail.get('player', {}).get('firstName', {}).get('default', '')} {detail.get('player', {}).get('lastName', {}).get('default', '')}".strip(),
                'team': detail.get('player', {}).get('team', {}).get('abbrev', ''),
                'games_played': detail.get('player', {}).get('gamesPlayed', 0),
            }

            # Skating speed
            skating = detail.get('skatingSpeed', {})
            if skating:
                speed_max = skating.get('speedMax', {})
                summary['max_speed_mph'] = speed_max.get('imperial', 0)
                summary['speed_percentile'] = speed_max.get('percentile', 0)
                summary['speed_league_avg'] = speed_max.get('leagueAvg', {}).get('imperial', 22.1)

                bursts = skating.get('burstsOver20', {})
                summary['bursts_over_20'] = bursts.get('value', 0)
                summary['bursts_percentile'] = bursts.get('percentile', 0)

            # Shot speed
            shot = detail.get('topShotSpeed', {})
            if shot:
                summary['shot_speed_mph'] = shot.get('imperial', 0)
                summary['shot_speed_percentile'] = shot.get('percentile', 0)
                summary['shot_speed_league_avg'] = shot.get('leagueAvg', {}).get('imperial', 83.3)

            # Get zone time (requires separate call)
            try:
                zone = self.client.edge.skater_zone_time(
                    player_id=str(player_id),
                    season=season
                )
                time.sleep(self.rate_limit_delay)

                if zone and 'zoneTimeDetails' in zone:
                    # Get "all situations" zone time
                    all_situations = next(
                        (z for z in zone['zoneTimeDetails'] if z.get('strengthCode') == 'all'),
                        None
                    )
                    if all_situations:
                        summary['oz_time_pct'] = all_situations.get('offensiveZonePctg', 0)
                        summary['oz_time_percentile'] = all_situations.get('offensiveZonePercentile', 0)
                        summary['oz_time_league_avg'] = all_situations.get('offensiveZoneLeagueAvg', 0.424)
                        summary['dz_time_pct'] = all_situations.get('defensiveZonePctg', 0)

                # Zone starts
                zone_starts = zone.get('zoneStarts', {})
                if zone_starts:
                    summary['oz_starts_pct'] = zone_starts.get('offensiveZoneStartsPctg', 0)
                    summary['oz_starts_percentile'] = zone_starts.get('offensiveZoneStartsPctgPercentile', 0)

            except Exception as e:
                print(f"  Warning: Could not fetch zone time for {player_id}: {e}")

            self._cache[cache_key] = summary
            return summary

        except Exception as e:
            print(f"Error fetching Edge data for player {player_id}: {e}")
            return None

    def get_edge_projection_boost(self, edge_summary: dict) -> Tuple[float, List[str]]:
        """
        Calculate projection boost multiplier based on Edge metrics.

        Returns:
            (boost_multiplier, list of boost reasons)
        """
        if not edge_summary:
            return 1.0, []

        boost = 1.0
        reasons = []

        # Offensive zone time boost
        oz_pct = edge_summary.get('oz_time_percentile', 0)
        if oz_pct >= self.ELITE_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['elite_oz_time']
            reasons.append(f"Elite OZ time ({oz_pct:.0%} pctile)")
        elif oz_pct >= self.ABOVE_AVG_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['above_avg_oz_time']
            reasons.append(f"Above-avg OZ time ({oz_pct:.0%} pctile)")

        # Skating speed boost
        speed_pct = edge_summary.get('speed_percentile', 0)
        if speed_pct >= self.ELITE_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['elite_speed']
            reasons.append(f"Elite speed ({edge_summary.get('max_speed_mph', 0):.1f} mph)")
        elif speed_pct >= self.ABOVE_AVG_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['above_avg_speed']
            reasons.append(f"Above-avg speed ({edge_summary.get('max_speed_mph', 0):.1f} mph)")

        # Burst count boost
        bursts_pct = edge_summary.get('bursts_percentile', 0)
        if bursts_pct >= self.ELITE_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['elite_bursts']
            reasons.append(f"Elite bursts ({edge_summary.get('bursts_over_20', 0)} over 20mph)")

        return boost, reasons
    
    def get_goalie_edge_boost(self, goalie_edge: dict) -> Tuple[float, List[str]]:
        """
        Calculate projection boost for goalies based on Edge metrics.
        
        Key metrics:
        - High-danger save percentage (most predictive)
        - Games above .900 consistency
        - 5v5 save percentage
        
        Returns:
            (boost_multiplier, list of boost reasons)
        """
        if not goalie_edge:
            return 1.0, []
        
        boost = 1.0
        reasons = []
        
        # High-danger save percentage
        hd_sv_pct = goalie_edge.get('hd_save_pct')
        if hd_sv_pct is not None:
            if hd_sv_pct >= self.ELITE_HD_SAVE_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['elite_hd_save_pct']
                reasons.append(f"Elite HD SV% ({hd_sv_pct:.1%})")
            elif hd_sv_pct >= self.ABOVE_AVG_HD_SAVE_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['above_avg_hd_save_pct']
                reasons.append(f"Above-avg HD SV% ({hd_sv_pct:.1%})")
            elif hd_sv_pct < self.POOR_HD_SAVE_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['poor_hd_save_pct']
                reasons.append(f"Poor HD SV% ({hd_sv_pct:.1%})")
        
        # Consistency (games above .900)
        pct_above_900 = goalie_edge.get('pct_games_above_900')
        if pct_above_900 is not None and pct_above_900 >= self.ELITE_CONSISTENCY:
            boost *= self.GOALIE_EDGE_BOOST_FACTORS['elite_consistency']
            reasons.append(f"Elite consistency ({pct_above_900:.0%} games >.900)")
        
        return boost, reasons

    def fetch_edge_for_slate(self, player_ids: List[int],
                              show_progress: bool = True) -> pd.DataFrame:
        """
        Fetch Edge stats for all players on a slate.

        Args:
            player_ids: List of NHL player IDs
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with Edge stats and calculated boosts
        """
        results = []
        total = len(player_ids)

        if show_progress:
            print(f"Fetching NHL Edge stats for {total} players...")

        for i, pid in enumerate(player_ids):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Edge stats: {i + 1}/{total}")

            summary = self.get_skater_edge_summary(pid)
            if summary:
                boost, reasons = self.get_edge_projection_boost(summary)
                summary['edge_boost'] = boost
                summary['edge_boost_reasons'] = '; '.join(reasons) if reasons else ''
                results.append(summary)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        if show_progress:
            boosted = len(df[df['edge_boost'] > 1.0])
            print(f"  Edge stats fetched: {len(df)} players, {boosted} with boosts")

        return df

    def print_edge_leaders(self, df: pd.DataFrame, top_n: int = 10):
        """Print top players by various Edge metrics."""
        if df.empty:
            print("No Edge data available")
            return

        print("\n" + "=" * 70)
        print(" NHL EDGE STAT LEADERS")
        print("=" * 70)

        # Top skating speed
        if 'max_speed_mph' in df.columns:
            print("\nTop Skating Speed:")
            speed_df = df.nlargest(top_n, 'max_speed_mph')[
                ['player_name', 'team', 'max_speed_mph', 'speed_percentile']
            ]
            for _, row in speed_df.iterrows():
                print(f"  {row['player_name']:25} {row['team']:4} {row['max_speed_mph']:.1f} mph ({row['speed_percentile']:.0%} pctile)")

        # Top OZ time
        if 'oz_time_pct' in df.columns:
            print("\nTop Offensive Zone Time %:")
            oz_df = df.nlargest(top_n, 'oz_time_pct')[
                ['player_name', 'team', 'oz_time_pct', 'oz_time_percentile']
            ]
            for _, row in oz_df.iterrows():
                print(f"  {row['player_name']:25} {row['team']:4} {row['oz_time_pct']:.1%} ({row['oz_time_percentile']:.0%} pctile)")

        # Players with Edge boosts
        boosted = df[df['edge_boost'] > 1.0].sort_values('edge_boost', ascending=False)
        if not boosted.empty:
            print(f"\nPlayers with Edge Boosts ({len(boosted)} total):")
            for _, row in boosted.head(top_n).iterrows():
                boost_pct = (row['edge_boost'] - 1) * 100
                print(f"  {row['player_name']:25} {row['team']:4} +{boost_pct:.1f}% | {row['edge_boost_reasons']}")


# ==================== SKATER EDGE BOOST FUNCTIONS ====================

def apply_edge_boosts(projections_df: pd.DataFrame,
                       edge_client: EdgeStatsClient = None,
                       player_id_col: str = 'player_id',
                       force_refresh: bool = False,
                       use_cache: bool = True) -> pd.DataFrame:
    """
    Apply Edge stat boosts to skater projection DataFrame.

    Args:
        projections_df: DataFrame with projections (must have player_id column)
        edge_client: Optional pre-initialized EdgeStatsClient
        player_id_col: Name of player ID column
        force_refresh: If True, fetch fresh Edge data from API (ignore cache)
        use_cache: If True and cache available, use cached data (much faster)

    Returns:
        DataFrame with Edge boosts applied to 'projected_fpts' column
    """
    # Try cached approach first (much faster - seconds vs minutes)
    if use_cache and HAS_EDGE_CACHE:
        return _apply_edge_boosts_cached(projections_df, force_refresh=force_refresh)

    # Fall back to per-player API calls (slow but always works)
    return _apply_edge_boosts_per_player(projections_df, edge_client, player_id_col)


def _apply_edge_boosts_cached(projections_df: pd.DataFrame,
                               force_refresh: bool = False) -> pd.DataFrame:
    """
    Apply Edge boosts using cached bulk data.

    This is the fast path - fetches all Edge data once and caches it for the day.
    Subsequent runs reuse the cache (seconds instead of minutes).
    """
    if not HAS_EDGE_CACHE:
        print("Warning: Edge cache not available")
        return projections_df

    # Get cached Edge stats (fetches if needed or cache expired)
    edge_df = get_cached_edge_stats(force_refresh=force_refresh)

    if edge_df.empty:
        print("Warning: No Edge data available from cache")
        return projections_df

    # Build lookup by player name (cached data uses 'Player' column)
    edge_lookup = {}
    for _, row in edge_df.iterrows():
        player_name = row.get('Player', '')
        if player_name:
            edge_lookup[player_name.lower()] = row.to_dict()

    # Apply boosts to projections
    df = projections_df.copy()
    df['edge_boost'] = 1.0
    df['edge_boost_reasons'] = ''
    df['max_speed_mph'] = None
    df['oz_time_pct'] = None
    df['oz_time_percentile'] = None

    edge_client = EdgeStatsClient()  # For boost calculation thresholds
    boosted_count = 0

    for idx, row in df.iterrows():
        player_name = row.get('name', '')
        if not player_name:
            continue

        # Try exact match first, then fuzzy
        edge_data = edge_lookup.get(player_name.lower())
        if not edge_data:
            # Try fuzzy match
            for cached_name, cached_data in edge_lookup.items():
                if _fuzzy_match(player_name.lower(), cached_name):
                    edge_data = cached_data
                    break

        if not edge_data:
            continue

        # Map cached columns to expected format for boost calculation
        summary = _map_cached_to_summary(edge_data)

        if summary:
            boost, reasons = edge_client.get_edge_projection_boost(summary)
            df.at[idx, 'edge_boost'] = boost
            df.at[idx, 'edge_boost_reasons'] = '; '.join(reasons) if reasons else ''
            df.at[idx, 'max_speed_mph'] = summary.get('max_speed_mph')
            df.at[idx, 'oz_time_pct'] = summary.get('oz_time_pct')
            df.at[idx, 'oz_time_percentile'] = summary.get('oz_time_percentile')

            if boost > 1.0:
                boosted_count += 1

    # Apply boosts to projected points
    if 'projected_fpts' in df.columns:
        df['projected_fpts_pre_edge'] = df['projected_fpts']
        df['projected_fpts'] = df['projected_fpts'] * df['edge_boost']

    print(f"  Edge boosts applied: {boosted_count} players with boosts")

    return df


# ==================== GOALIE EDGE BOOST FUNCTIONS (NEW) ====================

def apply_goalie_edge_boosts(projections_df: pd.DataFrame,
                              force_refresh: bool = False) -> pd.DataFrame:
    """
    Apply Edge stat boosts to goalie projections.
    
    Uses high-danger save percentage and consistency metrics.
    
    Args:
        projections_df: DataFrame with goalie projections
        force_refresh: If True, fetch fresh Edge data from API
        
    Returns:
        DataFrame with Edge boosts applied
    """
    if not HAS_EDGE_CACHE:
        print("Warning: Edge cache not available for goalies")
        return projections_df
    
    # Get cached goalie Edge stats
    goalie_edge_df = get_cached_goalie_edge_stats(force_refresh=force_refresh)
    
    if goalie_edge_df.empty:
        print("Warning: No goalie Edge data available")
        return projections_df
    
    # Build lookup by name
    edge_lookup = {}
    for _, row in goalie_edge_df.iterrows():
        name = row.get('name', '')
        if name:
            edge_lookup[name.lower()] = row.to_dict()
    
    # Apply boosts
    df = projections_df.copy()
    df['goalie_edge_boost'] = 1.0
    df['goalie_edge_reasons'] = ''
    df['hd_save_pct'] = None
    df['pct_games_above_900'] = None
    
    edge_client = EdgeStatsClient()
    boosted_count = 0
    
    for idx, row in df.iterrows():
        goalie_name = row.get('name', '')
        if not goalie_name:
            continue
        
        # Try exact match first, then fuzzy
        edge_data = edge_lookup.get(goalie_name.lower())
        if not edge_data:
            for cached_name, cached_data in edge_lookup.items():
                if _fuzzy_match(goalie_name.lower(), cached_name):
                    edge_data = cached_data
                    break
        
        if not edge_data:
            continue
        
        # Calculate boost
        boost, reasons = edge_client.get_goalie_edge_boost(edge_data)
        df.at[idx, 'goalie_edge_boost'] = boost
        df.at[idx, 'goalie_edge_reasons'] = '; '.join(reasons) if reasons else ''
        df.at[idx, 'hd_save_pct'] = edge_data.get('hd_save_pct')
        df.at[idx, 'pct_games_above_900'] = edge_data.get('pct_games_above_900')
        
        if boost != 1.0:
            boosted_count += 1
    
    # Apply boosts to projected points
    if 'projected_fpts' in df.columns:
        df['projected_fpts_pre_goalie_edge'] = df['projected_fpts']
        df['projected_fpts'] = df['projected_fpts'] * df['goalie_edge_boost']
    
    print(f"  Goalie Edge boosts applied: {boosted_count} goalies with adjustments")
    
    return df


# ==================== HELPER FUNCTIONS ====================

def _map_cached_to_summary(edge_data: dict) -> dict:
    """
    Map cached Edge data columns to the summary format expected by boost calculation.

    Cached data comes from bulk API (skater_stats_with_options) which has different
    column names than the per-player Edge API.
    """
    summary = {}

    # Speed metrics (from 'speed' report)
    # Bulk API columns: skaterSpeedMax, skaterSpeedMaxPercentile
    if 'skaterSpeedMax' in edge_data:
        summary['max_speed_mph'] = edge_data.get('skaterSpeedMax', 0)
    elif 'speedMax' in edge_data:
        summary['max_speed_mph'] = edge_data.get('speedMax', 0)

    # Speed percentile
    if 'skaterSpeedMaxPctg' in edge_data:
        # Convert to 0-1 range if needed
        pctg = edge_data.get('skaterSpeedMaxPctg', 0)
        summary['speed_percentile'] = pctg / 100 if pctg > 1 else pctg
    elif 'speedPercentile' in edge_data:
        pctg = edge_data.get('speedPercentile', 0)
        summary['speed_percentile'] = pctg / 100 if pctg > 1 else pctg

    # Bursts (from 'skatingstats' report)
    # Bulk API columns: burstsOver20mph, burstsOver20mphPercentile
    if 'burstsOver20mph' in edge_data:
        summary['bursts_over_20'] = edge_data.get('burstsOver20mph', 0)
    elif 'bursts20' in edge_data:
        summary['bursts_over_20'] = edge_data.get('bursts20', 0)

    if 'burstsOver20mphPctg' in edge_data:
        pctg = edge_data.get('burstsOver20mphPctg', 0)
        summary['bursts_percentile'] = pctg / 100 if pctg > 1 else pctg
    elif 'burstsPercentile' in edge_data:
        pctg = edge_data.get('burstsPercentile', 0)
        summary['bursts_percentile'] = pctg / 100 if pctg > 1 else pctg

    # OZ time (from 'timeonice' report)
    # Bulk API columns: offensiveZonePctg, offensiveZonePercentile
    if 'offensiveZonePctg' in edge_data:
        oz_pct = edge_data.get('offensiveZonePctg', 0)
        summary['oz_time_pct'] = oz_pct / 100 if oz_pct > 1 else oz_pct
    elif 'ozPctg' in edge_data:
        oz_pct = edge_data.get('ozPctg', 0)
        summary['oz_time_pct'] = oz_pct / 100 if oz_pct > 1 else oz_pct

    if 'offensiveZonePercentile' in edge_data:
        pctg = edge_data.get('offensiveZonePercentile', 0)
        summary['oz_time_percentile'] = pctg / 100 if pctg > 1 else pctg
    elif 'ozPercentile' in edge_data:
        pctg = edge_data.get('ozPercentile', 0)
        summary['oz_time_percentile'] = pctg / 100 if pctg > 1 else pctg

    return summary


def _fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """Simple fuzzy match for player names."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, name1, name2).ratio() >= threshold


def _apply_edge_boosts_per_player(projections_df: pd.DataFrame,
                                    edge_client: EdgeStatsClient = None,
                                    player_id_col: str = 'player_id') -> pd.DataFrame:
    """
    Apply Edge boosts via per-player API calls (legacy approach).

    This is slower but more accurate since it uses the detailed Edge API endpoint.
    Use when cache is unavailable or for specific player lookups.
    """
    if player_id_col not in projections_df.columns:
        print("Warning: No player_id column found, cannot apply Edge boosts")
        return projections_df

    if edge_client is None:
        edge_client = EdgeStatsClient()

    # Get unique player IDs
    player_ids = projections_df[player_id_col].dropna().astype(int).unique().tolist()

    # Fetch Edge data
    edge_df = edge_client.fetch_edge_for_slate(player_ids)

    if edge_df.empty:
        return projections_df

    # Merge boosts
    df = projections_df.copy()
    edge_merge = edge_df[['player_id', 'edge_boost', 'edge_boost_reasons',
                          'max_speed_mph', 'oz_time_pct', 'oz_time_percentile']].copy()

    df = df.merge(edge_merge, left_on=player_id_col, right_on='player_id', how='left')

    # Apply boosts
    if 'projected_fpts' in df.columns:
        df['edge_boost'] = df['edge_boost'].fillna(1.0)
        df['projected_fpts_pre_edge'] = df['projected_fpts']
        df['projected_fpts'] = df['projected_fpts'] * df['edge_boost']

    return df


# ==================== TEST ====================

if __name__ == "__main__":
    # Test the Edge stats client
    client = EdgeStatsClient()

    # Test with a few star players
    test_players = [
        (8478402, "Connor McDavid"),
        (8478483, "Leon Draisaitl"),
        (8479318, "Nikita Kucherov"),
        (8479339, "Auston Matthews"),
    ]

    print("Testing NHL Edge Stats Integration\n")

    for pid, name in test_players:
        print(f"\n{'=' * 50}")
        print(f"{name} (ID: {pid})")
        print('=' * 50)

        summary = client.get_skater_edge_summary(pid)
        if summary:
            print(f"  Max Speed: {summary.get('max_speed_mph', 0):.1f} mph ({summary.get('speed_percentile', 0):.0%} pctile)")
            print(f"  Bursts 20+: {summary.get('bursts_over_20', 0)} ({summary.get('bursts_percentile', 0):.0%} pctile)")
            print(f"  OZ Time: {summary.get('oz_time_pct', 0):.1%} ({summary.get('oz_time_percentile', 0):.0%} pctile)")
            print(f"  Shot Speed: {summary.get('shot_speed_mph', 0):.1f} mph ({summary.get('shot_speed_percentile', 0):.0%} pctile)")

            boost, reasons = client.get_edge_projection_boost(summary)
            if boost > 1.0:
                print(f"  Edge Boost: +{(boost - 1) * 100:.1f}%")
                for r in reasons:
                    print(f"    - {r}")
            else:
                print("  Edge Boost: None (metrics below threshold)")
    
    # Test goalie Edge
    print("\n" + "=" * 60)
    print("GOALIE EDGE STATS TEST")
    print("=" * 60)
    
    if HAS_EDGE_CACHE:
        goalie_df = get_cached_goalie_edge_stats(force_refresh=False)
        if not goalie_df.empty:
            print(f"Got {len(goalie_df)} goalies with Edge data")
            print("\nTop 5 by HD Save %:")
            if 'hd_save_pct' in goalie_df.columns:
                top = goalie_df.nlargest(5, 'hd_save_pct')[['name', 'team', 'hd_save_pct']]
                for _, row in top.iterrows():
                    hd = row['hd_save_pct']
                    boost, reasons = client.get_goalie_edge_boost(row.to_dict())
                    boost_str = f"+{(boost-1)*100:.1f}%" if boost > 1 else f"{(boost-1)*100:.1f}%"
                    print(f"  {row['name']:<25} {row['team']:<4} {hd:.1%} | Boost: {boost_str}")
