"""
NHL Edge Stats Integration for DFS Projections.

Fetches advanced player tracking data:
- Skating speed (max, bursts over 20/22 mph)
- Offensive zone time percentage
- Zone starts (OZ/DZ/NZ %)
- Distance skated per game
- Shot speed

These metrics can boost projections for players with elite underlying metrics.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from nhlpy import NHLClient
import time


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
    EDGE_BOOST_FACTORS = {
        'elite_oz_time': 1.03,      # +3% for elite offensive zone time
        'elite_speed': 1.02,        # +2% for elite skating speed
        'elite_bursts': 1.02,       # +2% for elite burst count
        'above_avg_oz_time': 1.015, # +1.5% for above-avg OZ time
        'above_avg_speed': 1.01,    # +1% for above-avg speed
    }

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


# Convenience function for integration with main.py
def apply_edge_boosts(projections_df: pd.DataFrame,
                       edge_client: EdgeStatsClient = None,
                       player_id_col: str = 'player_id') -> pd.DataFrame:
    """
    Apply Edge stat boosts to projection DataFrame.

    Args:
        projections_df: DataFrame with projections (must have player_id column)
        edge_client: Optional pre-initialized EdgeStatsClient
        player_id_col: Name of player ID column

    Returns:
        DataFrame with Edge boosts applied to 'projected_fpts' column
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
