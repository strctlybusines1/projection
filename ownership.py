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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import DAILY_SALARIES_DIR, CONTESTS_DIR


@dataclass
class OwnershipConfig:
    """Configuration for ownership model parameters."""
    # Salary curve parameters (based on historical analysis)
    # Adjusted: boosted mid-salary range where chalk concentrates
    salary_curve = {
        (2500, 3500): 4.0,    # Punt plays
        (3500, 4500): 12.0,   # Value sweet spot - BOOSTED (was 7.0)
        (4500, 5500): 14.0,   # Value sweet spot - BOOSTED (was 10.0)
        (5500, 6500): 12.0,   # Mid-range
        (6500, 7500): 11.0,   # Solid plays
        (7500, 8500): 13.0,   # Premium
        (8500, 9500): 16.0,   # Stars
        (9500, 11000): 22.0,  # Elite - BOOSTED (was 20.0)
    }

    # Multipliers
    pp1_boost: float = 1.5           # +50% for PP1 players (was 1.4)
    pp2_boost: float = 1.15          # +15% for PP2 players
    line1_boost: float = 1.25        # +25% for Line 1 players (was 1.2)
    confirmed_goalie_boost: float = 1.8  # +80% for confirmed starters (was 1.5)
    unconfirmed_goalie_penalty: float = 0.02  # 98% reduction - essentially 0% (was 0.1)

    # Value adjustments - BOOSTED for mid-salary value plays
    high_value_boost: float = 1.5    # +50% for top value plays (was 1.3)
    elite_value_boost: float = 1.8   # +80% for elite value (NEW)
    low_value_penalty: float = 0.6   # -40% for poor value (was 0.7)

    # Projection adjustments
    high_proj_boost: float = 1.3     # +30% for high projections (was 1.25)

    # Smash spot boost (high-value player in soft matchup)
    smash_spot_boost: float = 1.4    # +40% for smash spots (NEW)

    # Vegas implied team total multipliers (Feature 1)
    vegas_high_team_total_boost: float = 1.3     # implied_total >= 3.5
    vegas_mid_team_total_boost: float = 1.15     # implied_total >= 3.0
    vegas_low_team_total_penalty: float = 0.8    # implied_total < 2.5
    vegas_high_game_total_boost: float = 1.15    # game_total >= 6.5

    # DK average FPTS perceived value multipliers (Feature 2)
    dk_value_elite_boost: float = 1.4   # dk_value_ratio > 4.0
    dk_value_high_boost: float = 1.2    # dk_value_ratio > 3.0
    dk_value_low_penalty: float = 0.8   # dk_value_ratio < 2.0

    # Salary tier scarcity multipliers (Feature 3)
    scarcity_only_option_boost: float = 1.3   # only high-value option in tier
    scarcity_crowded_penalty: float = 0.85    # 3+ equally good options in tier

    # Return-from-injury buzz multipliers (Feature 4)
    injury_return_boost: float = 1.3    # IR player returning today
    injury_dtd_boost: float = 1.1       # DTD player on the slate

    # Individual recent game scoring multipliers (Feature 5)
    recency_blowup_boost: float = 1.5    # last_1_game > 25 FPTS
    recency_hot_boost: float = 1.3       # last_3_avg > 15
    recency_warm_boost: float = 1.15     # last_3_avg > 10
    recency_cold_penalty: float = 0.7    # last_3_avg < 5

    # Composite multiplier safety cap
    composite_cap: float = 5.0    # max composite multiplier
    composite_floor: float = 0.1  # min composite multiplier


class OwnershipModel:
    """
    Predict ownership percentages for NHL DFS players.

    Designed for top-heavy GPP structures where differentiation matters.
    """

    def __init__(self, config: OwnershipConfig = None):
        self.config = config or OwnershipConfig()
        self.lines_data = None
        self.confirmed_goalies = None
        self.team_totals = None
        self.team_game_totals = None
        self.injury_data = None
        self.target_date = None
        self.recent_scores = None

    def set_lines_data(self, lines_data: Dict, confirmed_goalies: Dict = None):
        """Set line combination data for PP1/Line1 boosts."""
        self.lines_data = lines_data
        self.confirmed_goalies = confirmed_goalies or {}

    def set_vegas_data(self, team_totals: Dict[str, float],
                       team_game_totals: Dict[str, float]):
        """Set Vegas implied team totals and game totals for ownership multipliers.

        Args:
            team_totals: team_abbrev -> implied team total (e.g. {'BOS': 3.4})
            team_game_totals: team_abbrev -> game total (e.g. {'BOS': 6.5})
        """
        self.team_totals = team_totals or {}
        self.team_game_totals = team_game_totals or {}

    def set_injury_data(self, injuries: pd.DataFrame, target_date: str):
        """Set injury data for return-from-injury buzz detection."""
        self.injury_data = injuries
        self.target_date = target_date

    def set_recent_scores(self, recent_scores: Dict[int, Dict[str, float]]):
        """Set individual recent game scoring data.

        Args:
            recent_scores: dict mapping player_id -> {
                'last_1_game_fpts': float,
                'last_3_avg_fpts': float,
                'last_5_avg_fpts': float,
            }
        """
        self.recent_scores = recent_scores or {}

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
        """
        Check if goalie is confirmed starter.

        CRITICAL: Non-confirmed goalies get essentially 0% ownership.
        Default to NOT confirmed unless explicitly in confirmed_goalies dict.
        """
        if not self.confirmed_goalies:
            return False  # Default to NOT confirmed (was True - caused over-prediction)

        confirmed = self.confirmed_goalies.get(team, '')
        if not confirmed:
            return False

        # Fuzzy match - be generous with name matching
        player_lower = player_name.lower().strip()
        confirmed_lower = confirmed.lower().strip()

        # Check various matching patterns
        if player_lower in confirmed_lower or confirmed_lower in player_lower:
            return True

        # Check last name match
        player_last = player_lower.split()[-1] if player_lower else ''
        confirmed_last = confirmed_lower.split()[-1] if confirmed_lower else ''

        if player_last and confirmed_last and player_last == confirmed_last:
            return True

        return False

    def _build_scarcity_map(self, df: pd.DataFrame) -> Dict[str, float]:
        """Precompute salary tier scarcity multipliers.

        Groups players into $500 salary buckets, counts high-value players
        per bucket, and returns a dict mapping player name -> scarcity multiplier.
        """
        scarcity = {}
        if 'value' not in df.columns:
            return scarcity

        median_value = df['value'].median()
        df_tmp = df.copy()
        df_tmp['salary_bucket'] = (df_tmp['salary'] // 500) * 500

        for bucket, group in df_tmp.groupby('salary_bucket'):
            high_value_players = group[group['value'] > median_value]
            n_high = len(high_value_players)

            for _, row in group.iterrows():
                if row['value'] > median_value:
                    if n_high == 1:
                        scarcity[row['name']] = self.config.scarcity_only_option_boost
                    elif n_high >= 3:
                        scarcity[row['name']] = self.config.scarcity_crowded_penalty
                    else:
                        scarcity[row['name']] = 1.0
                else:
                    scarcity[row['name']] = 1.0

        return scarcity

    def _is_returning_from_injury(self, player_name: str, team: str) -> str:
        """Check if player is returning from injury today.

        Returns:
            'IR_RETURN' if returning from IR today,
            'DTD' if day-to-day on the slate,
            '' otherwise
        """
        if self.injury_data is None or self.injury_data.empty or not self.target_date:
            return ''

        try:
            target = pd.Timestamp(self.target_date)
        except Exception:
            return ''

        player_lower = player_name.lower().strip()
        for _, row in self.injury_data.iterrows():
            inj_name = str(row.get('player_name', '')).lower().strip()
            # Fuzzy name match: check last name or substring
            player_last = player_lower.split()[-1] if player_lower else ''
            inj_last = inj_name.split()[-1] if inj_name else ''
            if not (player_lower in inj_name or inj_name in player_lower or
                    (player_last and inj_last and player_last == inj_last)):
                continue

            status = row.get('injury_status', '')
            return_date = row.get('return_date')

            # IR player returning today
            if status in ('IR', 'IR-LT', 'IR-NR') and pd.notna(return_date):
                ret = pd.Timestamp(return_date)
                if ret.date() == target.date():
                    return 'IR_RETURN'

            # DTD player on the slate
            if status == 'DTD':
                return 'DTD'

        return ''

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

        # Precompute salary tier scarcity (Feature 3)
        scarcity_map = self._build_scarcity_map(df)

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
                elif self.confirmed_goalies:
                    # We have confirmation data but this goalie isn't confirmed
                    multiplier *= self.config.unconfirmed_goalie_penalty
                else:
                    # No confirmation data available - use salary as proxy
                    # Higher salary goalies are more likely starters
                    if salary >= 8000:
                        multiplier *= 1.3  # Likely starter
                    elif salary >= 7000:
                        multiplier *= 1.0  # Possible starter
                    else:
                        multiplier *= 0.5  # Likely backup

            # 4. Value adjustment (key driver for mid-salary chalk)
            if pos_avg_value is not None and position in pos_avg_value.index:
                avg_val = pos_avg_value[position]
                if avg_val > 0:
                    value_ratio = value / avg_val
                    if value_ratio > 1.5:
                        # Elite value plays get massive boost (Victor Olofsson type)
                        multiplier *= self.config.elite_value_boost
                    elif value_ratio > 1.2:
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

            # 6. Smash spot detection (high value + mid salary = chalk)
            # Players in $3.5K-$5.5K range with high value get extra boost
            if 3500 <= salary <= 5500 and value > 3.0:
                multiplier *= self.config.smash_spot_boost

            # 7. Vegas implied team total (Feature 1)
            if self.team_totals and team in self.team_totals:
                implied_total = self.team_totals[team]
                if implied_total >= 3.5:
                    multiplier *= self.config.vegas_high_team_total_boost
                elif implied_total >= 3.0:
                    multiplier *= self.config.vegas_mid_team_total_boost
                elif implied_total < 2.5:
                    multiplier *= self.config.vegas_low_team_total_penalty
            # Game total boost (all players in high-scoring games)
            if self.team_game_totals and team in self.team_game_totals:
                game_total = self.team_game_totals[team]
                if game_total is not None and game_total >= 6.5:
                    multiplier *= self.config.vegas_high_game_total_boost

            # 8. DK average FPTS perceived value (Feature 2)
            dk_avg = row.get('dk_avg_fpts', 0)
            if dk_avg and salary > 0:
                dk_value_ratio = dk_avg / (salary / 1000)
                if dk_value_ratio > 4.0:
                    multiplier *= self.config.dk_value_elite_boost
                elif dk_value_ratio > 3.0:
                    multiplier *= self.config.dk_value_high_boost
                elif dk_value_ratio < 2.0:
                    multiplier *= self.config.dk_value_low_penalty

            # 9. Salary tier scarcity (Feature 3)
            if name in scarcity_map:
                multiplier *= scarcity_map[name]

            # 10. Return-from-injury buzz (Feature 4)
            injury_status = self._is_returning_from_injury(name, team)
            if injury_status == 'IR_RETURN':
                multiplier *= self.config.injury_return_boost
            elif injury_status == 'DTD':
                multiplier *= self.config.injury_dtd_boost

            # 11. Individual recent game scoring (Feature 5)
            player_id = row.get('player_id')
            if self.recent_scores and player_id and player_id in self.recent_scores:
                scores = self.recent_scores[player_id]
                last_1 = scores.get('last_1_game_fpts', 0)
                last_3 = scores.get('last_3_avg_fpts', 0)
                # Use max of applicable boosts (not multiplicative within category)
                recency_mult = 1.0
                if last_1 > 25:
                    recency_mult = max(recency_mult, self.config.recency_blowup_boost)
                if last_3 > 15:
                    recency_mult = max(recency_mult, self.config.recency_hot_boost)
                elif last_3 > 10:
                    recency_mult = max(recency_mult, self.config.recency_warm_boost)
                elif last_3 < 5:
                    recency_mult = min(recency_mult, self.config.recency_cold_penalty)
                multiplier *= recency_mult

            # Apply composite multiplier cap (safety)
            multiplier = max(self.config.composite_floor,
                             min(self.config.composite_cap, multiplier))

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
        """
        Normalize ownership to realistic contest levels.

        Total ownership must sum to 900% (9 roster spots).
        Each position has a specific number of slots:
        - C: 2 slots = 200%
        - W (LW+RW): 3 slots = 300%
        - D: 2 slots = 200%
        - G: 1 slot = 100%
        - UTIL: 1 slot = 100% (can be any skater)

        We normalize within each position group to hit these targets.
        """
        df = df.copy()
        pos_col = 'dk_pos' if 'dk_pos' in df.columns else 'position'

        # Position slot allocations (how much total ownership per position)
        # UTIL is typically filled by best remaining skater, so we add it to skater pool
        position_targets = {
            'C': 200.0,      # 2 C slots
            'LW': 150.0,     # 1.5 W slots (3 W total, split)
            'RW': 150.0,     # 1.5 W slots
            'D': 200.0,      # 2 D slots
            'G': 100.0,      # 1 G slot
            'UTIL': 100.0,   # UTIL goes to highest value skater
        }

        # For positions we have, normalize to target
        for pos in df[pos_col].unique():
            pos_mask = df[pos_col] == pos
            pos_df = df[pos_mask]

            if len(pos_df) == 0:
                continue

            # Get target for this position
            target = position_targets.get(pos, 100.0)

            # Current sum for this position
            current_sum = pos_df['predicted_ownership'].sum()

            if current_sum > 0:
                scale_factor = target / current_sum
                df.loc[pos_mask, 'predicted_ownership'] = (
                    df.loc[pos_mask, 'predicted_ownership'] * scale_factor
                )

        # Ensure min/max bounds (after normalization)
        # Min 0.1% for non-confirmed goalies, max 35% for chalk
        df['predicted_ownership'] = df['predicted_ownership'].clip(0.1, 35.0)

        # Re-normalize slightly if clipping changed totals significantly
        total = df['predicted_ownership'].sum()
        if abs(total - 900) > 50:  # If off by more than 50%, adjust
            df['predicted_ownership'] = df['predicted_ownership'] * (900 / total)

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

    # Load salaries and merge (daily_salaries/ first, then project root)
    project_dir = Path(__file__).parent
    salaries_dir = project_dir / DAILY_SALARIES_DIR
    salary_files = list(salaries_dir.glob('DKSalaries*.csv')) if salaries_dir.exists() else []
    if not salary_files:
        salary_files = list(project_dir.glob('DKSalaries*.csv'))
    salary_files = sorted(salary_files)
    if salary_files:
        salary_file = str(salary_files[-1])
        dk_salaries = load_dk_salaries(salary_file)
        dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'LW', 'RW', 'D'])]
        dk_goalies = dk_salaries[dk_salaries['position'] == 'G']

        projections['goalies']['position'] = 'G'

        skaters_merged = merge_projections_with_salaries(projections['skaters'], dk_skaters, 'skater')
        goalies_merged = merge_projections_with_salaries(projections['goalies'], dk_goalies, 'goalie')

        # Filter to confirmed starting goalies only
        if confirmed_goalies:
            from lines import fuzzy_match as _fm
            confirmed_names = list(confirmed_goalies.values())
            def _is_confirmed(name):
                return any(_fm(name, cn) for cn in confirmed_names)
            before = len(goalies_merged)
            goalies_merged = goalies_merged[goalies_merged['name'].apply(_is_confirmed)]
            filtered_count = before - len(goalies_merged)
            if filtered_count > 0:
                print(f"  Filtered {filtered_count} non-confirmed goalies from pool")

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
        contests_dir = project_dir / CONTESTS_DIR
        contest_files = list(contests_dir.glob('$*main_NHL*.csv')) if contests_dir.exists() else []
        if not contest_files:
            contest_files = list(project_dir.glob('$*main_NHL*.csv'))
        contest_files = [str(p) for p in contest_files]
        if contest_files:
            historical = analyze_historical_ownership(contest_files)
            print(f"Analyzed {len(contest_files)} contests, {len(historical)} player observations")
            print(f"Historical avg ownership: {historical['%Drafted'].mean():.2f}%")
            print(f"Predicted avg ownership: {player_pool['predicted_ownership'].mean():.2f}%")
    else:
        print("No salary file found. Add DKSalaries*.csv to daily_salaries/ or project folder.")
