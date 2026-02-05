"""
NHL Edge Stats Integration for DFS Projections (v2)

Skater Edge: Speed, OZ time, bursts - boosts for elite metrics
Goalie Edge: EV save %, quality starts %, consistency - boosts/penalties

Uses daily caching to avoid redundant API calls.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
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

# Try nhlpy for per-player Edge detail
try:
    from nhlpy import NHLClient
    HAS_NHLPY = True
except ImportError:
    HAS_NHLPY = False


class EdgeStatsClient:
    """Client for NHL Edge tracking data and boost calculations."""

    CURRENT_SEASON = "20252026"

    # Percentile thresholds
    ELITE_PERCENTILE = 0.90
    ABOVE_AVG_PERCENTILE = 0.65

    # Skater boost factors (calibrated from backtest)
    EDGE_BOOST_FACTORS = {
        'elite_oz_time': 1.10,
        'elite_speed': 1.02,
        'elite_bursts': 1.05,
        'above_avg_oz_time': 1.04,
        'above_avg_speed': 1.01,
    }
    
    # Goalie boost factors based on available NHL API metrics
    # EV save % and Quality Starts % are the most predictive
    GOALIE_EDGE_BOOST_FACTORS = {
        'elite_ev_save_pct': 1.08,      # +8% for top 10% EV save %
        'above_avg_ev_save_pct': 1.04,  # +4% for top 35% EV save %
        'elite_quality_starts': 1.06,   # +6% for 60%+ quality start rate
        'above_avg_quality_starts': 1.03,  # +3% for 50%+ QS rate
        'poor_ev_save_pct': 0.94,       # -6% for bottom 25% EV save %
        'poor_quality_starts': 0.96,    # -4% for <40% QS rate
    }
    
    # Goalie thresholds (based on 2025-26 data)
    ELITE_EV_SAVE_PCT = 0.920       # Top tier
    ABOVE_AVG_EV_SAVE_PCT = 0.905   # Above average
    POOR_EV_SAVE_PCT = 0.890        # Below average
    
    ELITE_QS_PCT = 0.60             # 60%+ quality starts
    ABOVE_AVG_QS_PCT = 0.50         # 50%+ quality starts
    POOR_QS_PCT = 0.40              # Below 40% quality starts

    def __init__(self, rate_limit_delay: float = 0.5):
        self.rate_limit_delay = rate_limit_delay
        self._cache: Dict[str, dict] = {}

    def get_edge_projection_boost(self, edge_summary: dict) -> Tuple[float, List[str]]:
        """Calculate skater projection boost from Edge metrics."""
        if not edge_summary:
            return 1.0, []

        boost = 1.0
        reasons = []

        # OZ time boost
        oz_pct = edge_summary.get('oz_time_percentile', 0)
        if oz_pct >= self.ELITE_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['elite_oz_time']
            reasons.append(f"Elite OZ time ({oz_pct:.0%})")
        elif oz_pct >= self.ABOVE_AVG_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['above_avg_oz_time']
            reasons.append(f"Above-avg OZ ({oz_pct:.0%})")

        # Speed boost
        speed_pct = edge_summary.get('speed_percentile', 0)
        if speed_pct >= self.ELITE_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['elite_speed']
            reasons.append(f"Elite speed ({edge_summary.get('max_speed_mph', 0):.1f} mph)")
        elif speed_pct >= self.ABOVE_AVG_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['above_avg_speed']

        # Bursts boost
        bursts_pct = edge_summary.get('bursts_percentile', 0)
        if bursts_pct >= self.ELITE_PERCENTILE:
            boost *= self.EDGE_BOOST_FACTORS['elite_bursts']
            reasons.append(f"Elite bursts ({edge_summary.get('bursts_over_20', 0)})")

        return boost, reasons
    
    def get_goalie_edge_boost(self, goalie_stats: dict) -> Tuple[float, List[str]]:
        """
        Calculate goalie projection boost from Edge metrics.
        
        Uses:
        - ev_save_pct: Even-strength save % (most predictive)
        - quality_starts_pct: % of games that are quality starts
        """
        if not goalie_stats:
            return 1.0, []
        
        boost = 1.0
        reasons = []
        
        # EV Save % boost/penalty
        ev_sv = goalie_stats.get('ev_save_pct')
        if ev_sv is not None and ev_sv > 0:
            if ev_sv >= self.ELITE_EV_SAVE_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['elite_ev_save_pct']
                reasons.append(f"Elite EV SV% ({ev_sv:.1%})")
            elif ev_sv >= self.ABOVE_AVG_EV_SAVE_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['above_avg_ev_save_pct']
                reasons.append(f"Above-avg EV SV% ({ev_sv:.1%})")
            elif ev_sv < self.POOR_EV_SAVE_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['poor_ev_save_pct']
                reasons.append(f"Poor EV SV% ({ev_sv:.1%})")
        
        # Quality Starts % boost/penalty
        qs_pct = goalie_stats.get('quality_starts_pct')
        if qs_pct is not None and qs_pct > 0:
            if qs_pct >= self.ELITE_QS_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['elite_quality_starts']
                reasons.append(f"Elite QS% ({qs_pct:.0%})")
            elif qs_pct >= self.ABOVE_AVG_QS_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['above_avg_quality_starts']
                reasons.append(f"Above-avg QS% ({qs_pct:.0%})")
            elif qs_pct < self.POOR_QS_PCT:
                boost *= self.GOALIE_EDGE_BOOST_FACTORS['poor_quality_starts']
                reasons.append(f"Poor QS% ({qs_pct:.0%})")
        
        return boost, reasons


# ==================== SKATER EDGE FUNCTIONS ====================

def apply_edge_boosts(projections_df: pd.DataFrame,
                       force_refresh: bool = False,
                       use_cache: bool = True) -> pd.DataFrame:
    """Apply Edge stat boosts to skater projections."""
    
    if use_cache and HAS_EDGE_CACHE:
        return _apply_edge_boosts_cached(projections_df, force_refresh)
    
    print("Warning: Edge cache not available, skipping boosts")
    return projections_df


def _apply_edge_boosts_cached(projections_df: pd.DataFrame,
                               force_refresh: bool = False) -> pd.DataFrame:
    """Apply skater Edge boosts using cached data."""
    if not HAS_EDGE_CACHE:
        return projections_df

    edge_df = get_cached_edge_stats(force_refresh=force_refresh)
    if edge_df.empty:
        print("Warning: No skater Edge data available")
        return projections_df

    # Build lookup
    edge_lookup = {}
    name_col = 'Player' if 'Player' in edge_df.columns else 'skaterFullName'
    if name_col not in edge_df.columns:
        for col in edge_df.columns:
            if 'name' in col.lower() or 'full' in col.lower():
                name_col = col
                break
    
    for _, row in edge_df.iterrows():
        name = row.get(name_col, '')
        if name:
            edge_lookup[name.lower()] = row.to_dict()

    df = projections_df.copy()
    df['edge_boost'] = 1.0
    df['edge_boost_reasons'] = ''

    edge_client = EdgeStatsClient()
    boosted_count = 0

    for idx, row in df.iterrows():
        player_name = row.get('name', '')
        if not player_name:
            continue

        edge_data = edge_lookup.get(player_name.lower())
        if not edge_data:
            for cached_name, cached_data in edge_lookup.items():
                if _fuzzy_match(player_name.lower(), cached_name):
                    edge_data = cached_data
                    break

        if not edge_data:
            continue

        summary = _map_cached_to_summary(edge_data)
        if summary:
            boost, reasons = edge_client.get_edge_projection_boost(summary)
            df.at[idx, 'edge_boost'] = boost
            df.at[idx, 'edge_boost_reasons'] = '; '.join(reasons) if reasons else ''
            if boost > 1.0:
                boosted_count += 1

    if 'projected_fpts' in df.columns:
        df['projected_fpts_pre_edge'] = df['projected_fpts']
        df['projected_fpts'] = df['projected_fpts'] * df['edge_boost']

    print(f"  Skater Edge boosts: {boosted_count} players boosted")
    return df


# ==================== GOALIE EDGE FUNCTIONS ====================

def apply_goalie_edge_boosts(projections_df: pd.DataFrame,
                              force_refresh: bool = False) -> pd.DataFrame:
    """Apply Edge stat boosts to goalie projections."""
    if not HAS_EDGE_CACHE:
        print("Warning: Edge cache not available for goalies")
        return projections_df
    
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
    
    df = projections_df.copy()
    df['goalie_edge_boost'] = 1.0
    df['goalie_edge_reasons'] = ''
    df['ev_save_pct'] = None
    df['quality_starts_pct'] = None
    
    edge_client = EdgeStatsClient()
    boosted_count = 0
    penalized_count = 0
    
    for idx, row in df.iterrows():
        goalie_name = row.get('name', '')
        if not goalie_name:
            continue
        
        edge_data = edge_lookup.get(goalie_name.lower())
        if not edge_data:
            for cached_name, cached_data in edge_lookup.items():
                if _fuzzy_match(goalie_name.lower(), cached_name):
                    edge_data = cached_data
                    break
        
        if not edge_data:
            continue
        
        boost, reasons = edge_client.get_goalie_edge_boost(edge_data)
        df.at[idx, 'goalie_edge_boost'] = boost
        df.at[idx, 'goalie_edge_reasons'] = '; '.join(reasons) if reasons else ''
        df.at[idx, 'ev_save_pct'] = edge_data.get('ev_save_pct')
        df.at[idx, 'quality_starts_pct'] = edge_data.get('quality_starts_pct')
        
        if boost > 1.0:
            boosted_count += 1
        elif boost < 1.0:
            penalized_count += 1
    
    if 'projected_fpts' in df.columns:
        df['projected_fpts_pre_goalie_edge'] = df['projected_fpts']
        df['projected_fpts'] = df['projected_fpts'] * df['goalie_edge_boost']
    
    print(f"  Goalie Edge: {boosted_count} boosted, {penalized_count} penalized")
    return df


# ==================== HELPER FUNCTIONS ====================

def _map_cached_to_summary(edge_data: dict) -> dict:
    """Map cached columns to boost calculation format."""
    summary = {}

    # Speed
    for col in ['skaterSpeedMax', 'speedMax', 'maxSpeed']:
        if col in edge_data:
            summary['max_speed_mph'] = edge_data[col]
            break

    # Speed percentile
    for col in ['skaterSpeedMaxPctg', 'speedPercentile']:
        if col in edge_data:
            pctg = edge_data[col]
            summary['speed_percentile'] = pctg / 100 if pctg > 1 else pctg
            break

    # Bursts
    for col in ['burstsOver20mph', 'bursts20']:
        if col in edge_data:
            summary['bursts_over_20'] = edge_data[col]
            break

    for col in ['burstsOver20mphPctg', 'burstsPercentile']:
        if col in edge_data:
            pctg = edge_data[col]
            summary['bursts_percentile'] = pctg / 100 if pctg > 1 else pctg
            break

    # OZ time
    for col in ['offensiveZonePctg', 'ozPctg', 'ozTimePct']:
        if col in edge_data:
            oz = edge_data[col]
            summary['oz_time_pct'] = oz / 100 if oz > 1 else oz
            break

    for col in ['offensiveZonePercentile', 'ozPercentile']:
        if col in edge_data:
            pctg = edge_data[col]
            summary['oz_time_percentile'] = pctg / 100 if pctg > 1 else pctg
            break

    return summary


def _fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """Simple fuzzy match for player names."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, name1, name2).ratio() >= threshold


# ==================== TEST ====================

if __name__ == "__main__":
    print("=" * 60)
    print("EDGE STATS TEST")
    print("=" * 60)
    
    # Test goalie boosts
    print("\nGoalie Edge Boost Examples:")
    client = EdgeStatsClient()
    
    test_goalies = [
        {'name': 'Elite Goalie', 'ev_save_pct': 0.925, 'quality_starts_pct': 0.65},
        {'name': 'Above Avg', 'ev_save_pct': 0.910, 'quality_starts_pct': 0.52},
        {'name': 'Average', 'ev_save_pct': 0.900, 'quality_starts_pct': 0.45},
        {'name': 'Struggling', 'ev_save_pct': 0.880, 'quality_starts_pct': 0.35},
    ]
    
    for g in test_goalies:
        boost, reasons = client.get_goalie_edge_boost(g)
        boost_str = f"+{(boost-1)*100:.1f}%" if boost > 1 else f"{(boost-1)*100:.1f}%"
        print(f"  {g['name']:<15} EV:{g['ev_save_pct']:.1%} QS:{g['quality_starts_pct']:.0%} → {boost_str}")
        if reasons:
            print(f"    Reasons: {', '.join(reasons)}")
    
    # Test with real data if cache available
    if HAS_EDGE_CACHE:
        print("\n" + "=" * 60)
        print("REAL GOALIE DATA TEST")
        print("=" * 60)
        
        goalie_df = get_cached_goalie_edge_stats(force_refresh=False)
        if not goalie_df.empty:
            # Show boosts for top 10 goalies
            print("\nTop 10 Goalies by GP with Edge Boosts:")
            top = goalie_df.nlargest(10, 'gamesPlayed')
            
            for _, row in top.iterrows():
                boost, reasons = client.get_goalie_edge_boost(row.to_dict())
                boost_str = f"+{(boost-1)*100:.1f}%" if boost >= 1 else f"{(boost-1)*100:.1f}%"
                ev = row.get('ev_save_pct', 0)
                qs = row.get('quality_starts_pct', 0)
                print(f"  {row['name']:<22} {row['team']:<4} EV:{ev:.1%} QS:{qs:.0%} → {boost_str}")
