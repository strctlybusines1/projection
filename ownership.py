"""
NHL DFS Ownership Projection Model

Predicts ownership percentages for DraftKings NHL contests.
Optimized for top-heavy GPP structures ($5 entry, $5K to 1st).

Key insights from historical analysis:
- 47% of chalk (15%+ owned) busts (<10 FPTS)
- 20%+ owned players average only 7.1 FPTS (worst tier)
- Best leverage: 5-15% owned players with high ceilings
- Confirmed goalies critical (non-confirmed ~0% owned)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OwnershipConfig:
    """Configuration for ownership model parameters."""
    # Salary curve parameters (based on historical analysis)
    salary_curve = {
        (2500, 3500): 4.0,    # Punt plays
        (3500, 4500): 7.0,    # Low value
        (4500, 5500): 10.0,   # Value sweet spot
        (5500, 6500): 12.0,   # Mid-range
        (6500, 7500): 11.0,   # Solid plays
        (7500, 8500): 13.0,   # Premium
        (8500, 9500): 16.0,   # Stars
        (9500, 11000): 20.0,  # Elite
    }

    # Multipliers
    pp1_boost: float = 1.4           # +40% for PP1 players
    pp2_boost: float = 1.15          # +15% for PP2 players
    line1_boost: float = 1.2         # +20% for Line 1 players
    confirmed_goalie_boost: float = 1.5  # +50% for confirmed starters
    unconfirmed_goalie_penalty: float = 0.1  # 90% reduction if not confirmed

    # Value adjustments
    high_value_boost: float = 1.3    # +30% for top value plays
    low_value_penalty: float = 0.7   # -30% for poor value

    # Projection adjustments
    high_proj_boost: float = 1.25    # +25% for high projections

    # Recent performance
    hot_streak_boost: float = 1.2    # +20% for hot players


class OwnershipModel:
    """
    Predict ownership percentages for NHL DFS players.

    Designed for top-heavy GPP structures where differentiation matters.
    """

    def __init__(self, config: OwnershipConfig = None):
        self.config = config or OwnershipConfig()
        self.lines_data = None
        self.confirmed_goalies = None

    def set_lines_data(self, lines_data: Dict, confirmed_goalies: Dict = None):
        """Set line combination data for PP1/Line1 boosts."""
        self.lines_data = lines_data
        self.confirmed_goalies = confirmed_goalies or {}

    def _get_base_ownership(self, salary: float) -> float:
        """Get base ownership from salary curve."""
        for (low, high), base_own in self.config.salary_curve.items():
            if low <= salary < high:
                return base_own
        # Default for outliers
        if salary >= 11000:
            return 22.0
        return 3.0

    def _is_pp1(self, player_name: str, team: str) -> bool:
        """Check if player is on PP1."""
        if not self.lines_data or team not in self.lines_data:
            return False

        team_data = self.lines_data.get(team, {})
        pp_units = team_data.get('pp_units', [])

        for pp in pp_units:
            if pp.get('unit') == 1:
                pp1_players = [p.lower() for p in pp.get('players', [])]
                if any(player_name.lower() in p or p in player_name.lower()
                       for p in pp1_players):
                    return True
        return False

    def _is_pp2(self, player_name: str, team: str) -> bool:
        """Check if player is on PP2."""
        if not self.lines_data or team not in self.lines_data:
            return False

        team_data = self.lines_data.get(team, {})
        pp_units = team_data.get('pp_units', [])

        for pp in pp_units:
            if pp.get('unit') == 2:
                pp2_players = [p.lower() for p in pp.get('players', [])]
                if any(player_name.lower() in p or p in player_name.lower()
                       for p in pp2_players):
                    return True
        return False

    def _is_line1(self, player_name: str, team: str) -> bool:
        """Check if player is on Line 1."""
        if not self.lines_data or team not in self.lines_data:
            return False

        team_data = self.lines_data.get(team, {})
        forward_lines = team_data.get('forward_lines', [])

        for line in forward_lines:
            if line.get('line') == 1:
                line1_players = [p.lower() for p in line.get('players', [])]
                if any(player_name.lower() in p or p in player_name.lower()
                       for p in line1_players):
                    return True
        return False

    def _is_confirmed_goalie(self, player_name: str, team: str) -> bool:
        """Check if goalie is confirmed starter."""
        if not self.confirmed_goalies:
            return True  # Assume confirmed if no data

        confirmed = self.confirmed_goalies.get(team, '')
        if not confirmed:
            return False

        # Fuzzy match
        return (player_name.lower() in confirmed.lower() or
                confirmed.lower() in player_name.lower())

    def predict_ownership(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """
        Predict ownership for all players in the pool.

        Args:
            player_pool: DataFrame with columns: name, team, position, salary,
                        projected_fpts, value, dk_avg_fpts

        Returns:
            DataFrame with added 'predicted_ownership' and 'leverage_score' columns
        """
        df = player_pool.copy()

        # Calculate position averages for normalization
        pos_col = 'dk_pos' if 'dk_pos' in df.columns else 'position'
        pos_avg_proj = df.groupby(pos_col)['projected_fpts'].mean()
        pos_avg_value = df.groupby(pos_col)['value'].mean() if 'value' in df.columns else None

        ownership_predictions = []

        for idx, row in df.iterrows():
            name = row['name']
            team = row['team']
            salary = row['salary']
            position = row.get('dk_pos', row.get('position', 'C'))
            projected = row.get('projected_fpts', 10)
            value = row.get('value', 3)

            # 1. Base ownership from salary
            base_own = self._get_base_ownership(salary)

            # 2. Position/role multipliers
            multiplier = 1.0

            # PP1 boost (biggest ownership driver)
            if self._is_pp1(name, team):
                multiplier *= self.config.pp1_boost
            elif self._is_pp2(name, team):
                multiplier *= self.config.pp2_boost

            # Line 1 boost
            if self._is_line1(name, team):
                multiplier *= self.config.line1_boost

            # 3. Goalie confirmation (critical)
            if position == 'G':
                if self._is_confirmed_goalie(name, team):
                    multiplier *= self.config.confirmed_goalie_boost
                else:
                    multiplier *= self.config.unconfirmed_goalie_penalty

            # 4. Value adjustment
            if pos_avg_value is not None and position in pos_avg_value.index:
                avg_val = pos_avg_value[position]
                if avg_val > 0:
                    value_ratio = value / avg_val
                    if value_ratio > 1.2:
                        multiplier *= self.config.high_value_boost
                    elif value_ratio < 0.8:
                        multiplier *= self.config.low_value_penalty

            # 5. Projection adjustment
            if position in pos_avg_proj.index:
                avg_proj = pos_avg_proj[position]
                if avg_proj > 0:
                    proj_ratio = projected / avg_proj
                    if proj_ratio > 1.3:
                        multiplier *= self.config.high_proj_boost

            # Calculate final ownership
            predicted_own = base_own * multiplier

            # Cap at reasonable bounds
            predicted_own = max(0.5, min(45.0, predicted_own))

            ownership_predictions.append(predicted_own)

        df['predicted_ownership'] = ownership_predictions

        # Normalize so total ownership is reasonable
        # In a 9-player lineup, average ownership should be ~100/9 = 11.1% per slot
        # But with 300+ players, total ownership across all players = 900%
        df = self._normalize_ownership(df)

        # Calculate leverage score (high projection + low ownership = good leverage)
        df['leverage_score'] = df['projected_fpts'] / (df['predicted_ownership'] + 1)

        # Ownership tier labels
        df['ownership_tier'] = df['predicted_ownership'].apply(self._get_ownership_tier)

        return df

    def _normalize_ownership(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize ownership to realistic contest levels."""
        # Target: position-weighted ownership sums
        # Based on historical data, avg ownership ~3-4% across all players

        current_mean = df['predicted_ownership'].mean()
        target_mean = 3.5  # Based on historical analysis

        if current_mean > 0:
            scale_factor = target_mean / current_mean
            df['predicted_ownership'] = df['predicted_ownership'] * scale_factor

        # Ensure min/max bounds
        df['predicted_ownership'] = df['predicted_ownership'].clip(0.3, 40.0)

        return df

    def _get_ownership_tier(self, ownership: float) -> str:
        """Categorize ownership into tiers."""
        if ownership >= 20:
            return 'Chalk (20%+)'
        elif ownership >= 15:
            return 'Popular (15-20%)'
        elif ownership >= 10:
            return 'Moderate (10-15%)'
        elif ownership >= 5:
            return 'Low (5-10%)'
        else:
            return 'Contrarian (<5%)'

    def get_leverage_plays(self, df: pd.DataFrame,
                          min_projection: float = 12.0,
                          max_ownership: float = 10.0,
                          top_n: int = 20) -> pd.DataFrame:
        """
        Find best leverage plays (high upside, low ownership).

        These are the GPP-winning differentiators in top-heavy structures.
        """
        leverage = df[
            (df['projected_fpts'] >= min_projection) &
            (df['predicted_ownership'] <= max_ownership)
        ].copy()

        leverage = leverage.sort_values('leverage_score', ascending=False)

        return leverage.head(top_n)

    def get_chalk_plays(self, df: pd.DataFrame,
                        min_ownership: float = 15.0) -> pd.DataFrame:
        """Get high-ownership chalk plays (potential fades in GPPs)."""
        return df[df['predicted_ownership'] >= min_ownership].sort_values(
            'predicted_ownership', ascending=False
        )

    def calculate_lineup_ownership(self, lineup: pd.DataFrame) -> Dict:
        """Calculate total ownership metrics for a lineup."""
        if 'predicted_ownership' not in lineup.columns:
            return {'error': 'No ownership predictions in lineup'}

        total_own = lineup['predicted_ownership'].sum()
        avg_own = lineup['predicted_ownership'].mean()
        max_own = lineup['predicted_ownership'].max()
        min_own = lineup['predicted_ownership'].min()

        # Count players by tier
        tier_counts = lineup['ownership_tier'].value_counts().to_dict()

        return {
            'total_ownership': total_own,
            'avg_ownership': avg_own,
            'max_single_ownership': max_own,
            'min_single_ownership': min_own,
            'tier_breakdown': tier_counts,
            'leverage_rating': 'High' if avg_own < 8 else 'Medium' if avg_own < 12 else 'Low'
        }


def analyze_historical_ownership(contest_files: List[str]) -> pd.DataFrame:
    """
    Analyze historical ownership from contest result files.

    Args:
        contest_files: List of paths to DK contest CSV files

    Returns:
        DataFrame with aggregated ownership statistics
    """
    all_data = []

    for file in contest_files:
        try:
            df = pd.read_csv(file)
            if '%Drafted' in df.columns:
                ownership = df[['Player', 'Roster Position', '%Drafted', 'FPTS']].dropna()
                ownership = ownership.drop_duplicates(subset=['Player'])
                ownership['%Drafted'] = ownership['%Drafted'].str.replace('%', '').astype(float)
                ownership['contest'] = file
                all_data.append(ownership)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def print_ownership_report(df: pd.DataFrame, top_n: int = 20):
    """Print a formatted ownership report."""
    print("\n" + "=" * 70)
    print("OWNERSHIP PROJECTIONS REPORT")
    print("=" * 70)

    # Top projected ownership
    print(f"\nTop {top_n} Projected Ownership:")
    print("-" * 70)
    cols = ['name', 'team', 'dk_pos', 'salary', 'projected_fpts',
            'predicted_ownership', 'ownership_tier']
    cols = [c for c in cols if c in df.columns]

    top_owned = df.nlargest(top_n, 'predicted_ownership')[cols]
    for _, row in top_owned.iterrows():
        pos = row.get('dk_pos', row.get('position', '?'))
        print(f"  {row['name']:<25} {row['team']:<4} {pos:<3} "
              f"${row['salary']:<6,} {row['projected_fpts']:>5.1f} pts  "
              f"{row['predicted_ownership']:>5.1f}% ({row.get('ownership_tier', '')})")

    # Leverage plays
    print(f"\nTop {top_n} Leverage Plays (High Proj, Low Own):")
    print("-" * 70)
    leverage = df[df['projected_fpts'] >= 10].nlargest(top_n, 'leverage_score')[cols + ['leverage_score']]
    for _, row in leverage.iterrows():
        pos = row.get('dk_pos', row.get('position', '?'))
        print(f"  {row['name']:<25} {row['team']:<4} {pos:<3} "
              f"${row['salary']:<6,} {row['projected_fpts']:>5.1f} pts  "
              f"{row['predicted_ownership']:>5.1f}% | Lev: {row['leverage_score']:.2f}")

    # Ownership distribution
    print("\nOwnership Distribution:")
    print("-" * 70)
    tier_dist = df['ownership_tier'].value_counts()
    for tier, count in tier_dist.items():
        pct = 100 * count / len(df)
        print(f"  {tier:<20} {count:>4} players ({pct:>5.1f}%)")


# Quick test
if __name__ == "__main__":
    import glob
    from data_pipeline import NHLDataPipeline
    from projections import NHLProjectionModel
    from main import load_dk_salaries, merge_projections_with_salaries
    from lines import LinesScraper, StackBuilder
    from datetime import datetime

    print("Building Ownership Projections...")
    print("=" * 70)

    # Build projections
    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(include_game_logs=False)
    model = NHLProjectionModel()
    today = datetime.now().strftime('%Y-%m-%d')
    projections = model.generate_projections(data, target_date=today)

    # Fetch lines for confirmed goalies and PP units
    print("\nFetching line combinations...")
    schedule = pipeline.fetch_schedule(today)
    teams_playing = set()
    teams_playing.update(schedule['home_team'].tolist())
    teams_playing.update(schedule['away_team'].tolist())
    teams_playing = sorted([t for t in teams_playing if t])

    scraper = LinesScraper()
    all_lines = scraper.get_multiple_teams(teams_playing)
    stack_builder = StackBuilder(all_lines)
    confirmed_goalies = stack_builder.get_all_starting_goalies()

    # Load salaries and merge
    salary_files = glob.glob('DKSalaries*.csv')
    if salary_files:
        # Use most recent salary file
        salary_file = sorted(salary_files)[-1]
        dk_salaries = load_dk_salaries(salary_file)
        dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'LW', 'RW', 'D'])]
        dk_goalies = dk_salaries[dk_salaries['position'] == 'G']

        projections['goalies']['position'] = 'G'

        skaters_merged = merge_projections_with_salaries(projections['skaters'], dk_skaters, 'skater')
        goalies_merged = merge_projections_with_salaries(projections['goalies'], dk_goalies, 'goalie')

        player_pool = pd.concat([skaters_merged, goalies_merged], ignore_index=True)

        # Run ownership model
        ownership_model = OwnershipModel()
        ownership_model.set_lines_data(all_lines, confirmed_goalies)

        player_pool = ownership_model.predict_ownership(player_pool)

        # Print report
        print_ownership_report(player_pool)

        # Analyze historical for comparison
        print("\n" + "=" * 70)
        print("HISTORICAL OWNERSHIP COMPARISON")
        print("=" * 70)
        contest_files = glob.glob('$*main_NHL*.csv')
        if contest_files:
            historical = analyze_historical_ownership(contest_files)
            print(f"Analyzed {len(contest_files)} contests, {len(historical)} player observations")
            print(f"Historical avg ownership: {historical['%Drafted'].mean():.2f}%")
            print(f"Predicted avg ownership: {player_pool['predicted_ownership'].mean():.2f}%")
    else:
        print("No salary file found.")
