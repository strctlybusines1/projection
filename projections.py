"""
NHL DFS Projection Model using TabPFN.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from tabpfn import TabPFNRegressor

from config import (
    calculate_skater_fantasy_points, calculate_goalie_fantasy_points,
    SKATER_SCORING, SKATER_BONUSES, GOALIE_SCORING, GOALIE_BONUSES,
    INJURY_STATUSES_EXCLUDE, STREAK_ADJUSTMENT_FACTOR
)
from features import FeatureEngineer

# ==================== Backtest-Derived Bias Corrections ====================
# Based on backtest data from 1/22-1/23/26 (445 predictions)
# Overall skater bias: +1.04 pts (over-projection)
# Overall goalie bias: -0.86 pts (under-projection)

GLOBAL_BIAS_CORRECTION = 0.97  # 3% reduction to combat over-projection

# Position-specific bias corrections (skaters)
# Derived from MAE and bias analysis by position
POSITION_BIAS_CORRECTION = {
    'C': 0.97,   # Centers over-projected by ~1.28 pts
    'L': 0.96,   # Left wings over-projected by ~1.65 pts
    'LW': 0.96,  # Left wings (alternate code)
    'R': 1.01,   # Right wings slightly under-projected (-0.47 pts)
    'RW': 1.01,  # Right wings (alternate code)
    'W': 0.985,  # Generic wing (average of L and R)
    'D': 0.95,   # Defensemen over-projected by ~1.56 pts
}

# Goalie bias correction (they're under-projected by ~0.86 pts)
GOALIE_BIAS_CORRECTION = 1.05  # 5% boost

# Floor multiplier (reduced from 0.4 - 30.5% were below floor)
FLOOR_MULTIPLIER = 0.25


class NHLProjectionModel:
    """
    NHL DFS Projection Model using TabPFN for small-sample regression.
    """

    def __init__(self):
        self.skater_model = None
        self.goalie_model = None
        self.feature_engineer = FeatureEngineer()
        self._models_initialized = False

    def _init_models(self):
        """Initialize TabPFN models (lazy loading)."""
        if not self._models_initialized:
            print("Initializing TabPFN models...")
            self.skater_model = TabPFNRegressor()
            self.goalie_model = TabPFNRegressor()
            self._models_initialized = True
            print("Models initialized.")

    def calculate_expected_fantasy_points_skater(self, row: pd.Series) -> float:
        """
        Calculate expected DraftKings fantasy points for a skater based on per-game averages.
        Includes bonus expected value and advanced stat adjustments.
        """
        # Base expected points from per-game stats
        goals = row.get('goals_pg', 0)
        assists = row.get('assists_pg', 0)
        shots = row.get('shots_pg', 0)
        blocks = row.get('blocks_pg', 0)
        sh_points = row.get('sh_points_pg', 0)

        expected_pts = (
            goals * SKATER_SCORING['goals'] +
            assists * SKATER_SCORING['assists'] +
            shots * SKATER_SCORING['shots_on_goal'] +
            blocks * SKATER_SCORING['blocked_shots'] +
            sh_points * SKATER_SCORING['shorthanded_points_bonus']
        )

        # Add expected value of bonuses
        prob_5_shots = row.get('prob_5plus_shots', 0)
        prob_3_points = row.get('prob_3plus_points', 0)
        prob_hat_trick = row.get('prob_hat_trick', 0)
        prob_3_blocks = min(blocks / 3, 1) * 0.3  # Rough estimate

        expected_pts += prob_5_shots * SKATER_BONUSES['five_plus_shots']
        expected_pts += prob_3_points * SKATER_BONUSES['three_plus_points']
        expected_pts += prob_hat_trick * SKATER_BONUSES['hat_trick']
        expected_pts += prob_3_blocks * SKATER_BONUSES['three_plus_blocks']

        # Apply signal-weighted matchup adjustment (replaces basic opp_softness)
        # Based on backtest: share metrics (SFpct, xGFpct, SCFpct, FFpct, CFpct)
        # predict DFS output better than simple GA/game
        signal_boost = row.get('signal_matchup_boost', 1.0)
        if pd.notna(signal_boost):
            expected_pts *= signal_boost

        # ==================== Advanced Stat Adjustments ====================

        # Apply xG matchup boost (based on expected goals analysis)
        xg_matchup_boost = row.get('xg_matchup_boost', 1.0)
        if pd.notna(xg_matchup_boost):
            expected_pts *= xg_matchup_boost

        # Apply hot/cold streak adjustment
        if row.get('team_hot_streak', 0) == 1:
            # Hot team = boost projections
            expected_pts *= (1.0 + STREAK_ADJUSTMENT_FACTOR)
        elif row.get('team_cold_streak', 0) == 1:
            # Cold team = reduce projections
            expected_pts *= (1.0 - STREAK_ADJUSTMENT_FACTOR)

        # Apply PDO regression adjustment
        pdo_adj = row.get('pdo_adj_factor', 1.0)
        if pd.notna(pdo_adj):
            expected_pts *= pdo_adj

        # Apply opportunity boost from teammate injuries
        opp_boost = row.get('opportunity_boost', 1.0)
        if pd.notna(opp_boost):
            expected_pts *= opp_boost

        # ==================== Standard Adjustments ====================

        # Home ice advantage (reduced from 1.02 based on backtest)
        if row.get('is_home') == True:
            expected_pts *= 1.01

        # ==================== Backtest Bias Corrections ====================
        # Apply global bias correction
        expected_pts *= GLOBAL_BIAS_CORRECTION

        # Apply position-specific bias correction
        position = row.get('position', 'C')
        pos_correction = POSITION_BIAS_CORRECTION.get(position, 0.97)
        expected_pts *= pos_correction

        return expected_pts

    def calculate_expected_fantasy_points_goalie(self, row: pd.Series) -> float:
        """
        Calculate expected DraftKings fantasy points for a goalie.
        """
        saves = row.get('saves_pg', 25)
        ga = row.get('ga_pg', 2.5)
        win_rate = row.get('win_rate', 0.5)
        sa_pg = row.get('sa_pg', 28)

        # Base expected points
        expected_pts = (
            saves * GOALIE_SCORING['save'] +
            ga * GOALIE_SCORING['goal_against'] +
            win_rate * GOALIE_SCORING['win']
        )

        # OT loss expected value (rough: ~10% of non-wins are OTL)
        ot_loss_rate = (1 - win_rate) * 0.25
        expected_pts += ot_loss_rate * GOALIE_SCORING['overtime_loss']

        # Shutout expected value (very low probability)
        shutout_prob = max(0, 0.15 - ga * 0.04)  # Decreases with higher GA
        expected_pts += shutout_prob * GOALIE_SCORING['shutout_bonus']

        # 35+ saves bonus
        prob_35_saves = row.get('prob_35plus_saves', 0)
        expected_pts += prob_35_saves * GOALIE_BONUSES['thirty_five_plus_saves']

        # Opponent adjustment (weak offense = good for goalie)
        opp_weakness = row.get('opp_offense_weakness', 1.0)
        if pd.notna(opp_weakness):
            # Facing weak offense reduces GA
            ga_reduction = (opp_weakness - 1.0) * 0.1 + 1.0
            expected_pts *= ga_reduction

        # Home ice advantage
        if row.get('is_home') == True:
            expected_pts *= 1.03

        # ==================== Backtest Bias Correction ====================
        # Goalies are under-projected by ~0.86 pts on average
        expected_pts *= GOALIE_BIAS_CORRECTION

        # ==================== Goalie Quality Tier ====================
        # Penalizes BACKUP goalies by 20%, STARTER by 5%, ELITE unchanged
        tier_mult = row.get('tier_multiplier', 1.0)
        if pd.notna(tier_mult):
            expected_pts *= tier_mult

        return expected_pts

    def project_skaters_baseline(self, skater_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate baseline projections for skaters using calculated expected value.
        This doesn't require training data - uses per-game stats directly.
        """
        df = skater_features.copy()

        # Calculate expected fantasy points
        df['projected_fpts'] = df.apply(self.calculate_expected_fantasy_points_skater, axis=1)

        # Calculate floor/ceiling estimates
        # Floor reduced from 0.4 to 0.25 (30.5% were below floor in backtest)
        df['floor'] = df['projected_fpts'] * FLOOR_MULTIPLIER
        df['ceiling'] = df['projected_fpts'] * 2.5 + 5  # Great game with bonuses

        # Sort by projected points
        df = df.sort_values('projected_fpts', ascending=False)

        return df

    def project_goalies_baseline(self, goalie_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate baseline projections for goalies using calculated expected value.
        """
        df = goalie_features.copy()

        # Calculate expected fantasy points
        df['projected_fpts'] = df.apply(self.calculate_expected_fantasy_points_goalie, axis=1)

        # Floor/ceiling
        # Floor reduced from 0.3 to 0.20 for goalies (high variance position)
        df['floor'] = df['projected_fpts'] * 0.20  # Bad game (loss, high GA)
        df['ceiling'] = df['projected_fpts'] * 2.0 + 10  # Win + high saves + shutout

        df = df.sort_values('projected_fpts', ascending=False)

        return df

    def project_with_tabpfn(self, features_df: pd.DataFrame,
                            historical_fpts: pd.Series,
                            feature_cols: List[str],
                            player_type: str = 'skater') -> pd.DataFrame:
        """
        Use TabPFN to project fantasy points when historical data is available.

        Args:
            features_df: DataFrame with engineered features
            historical_fpts: Series of actual fantasy points (same index as features_df)
            feature_cols: List of feature column names to use
            player_type: 'skater' or 'goalie'
        """
        self._init_models()

        df = features_df.copy()
        model = self.skater_model if player_type == 'skater' else self.goalie_model

        # Prepare feature matrix
        available_cols = [c for c in feature_cols if c in df.columns]
        X = df[available_cols].copy()

        # Fill missing values
        X = X.fillna(X.median())

        # Handle any remaining issues
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # If we have historical data, fit and predict
        if historical_fpts is not None and len(historical_fpts) > 0:
            y = historical_fpts.values

            # TabPFN can do in-context learning without explicit train/test split
            # For now, we'll use it to predict on same data (in production, use proper CV)
            model.fit(X.values, y)
            predictions = model.predict(X.values)

            df['projected_fpts_tabpfn'] = predictions

        return df

    def generate_projections(self, data: Dict[str, pd.DataFrame],
                              target_date: Optional[str] = None,
                              use_tabpfn: bool = False,
                              filter_injuries: bool = True,
                              include_dtd: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate full projections for a slate.

        Args:
            data: Dict with 'skaters', 'goalies', 'teams', 'schedule' DataFrames
                  May also include 'injuries', 'advanced_team_stats', 'advanced_player_stats'
            target_date: Date to project for (YYYY-MM-DD)
            use_tabpfn: Whether to use TabPFN (requires historical fantasy point data)
            filter_injuries: If True, remove injured players from projections
            include_dtd: If True, also filter out Day-to-Day players

        Returns:
            Dict with 'skaters' and 'goalies' projection DataFrames
        """
        print(f"Generating projections for {target_date or 'upcoming games'}...")

        # Extract advanced stats if available
        advanced_stats = None
        if 'advanced_team_stats' in data or 'advanced_player_stats' in data:
            advanced_stats = {
                'team': data.get('advanced_team_stats', {}),
                'player': data.get('advanced_player_stats', {})
            }

        # Extract injuries if available
        injuries = data.get('injuries', pd.DataFrame())

        # Engineer features (now with advanced stats and injuries)
        skater_features = self.feature_engineer.engineer_skater_features(
            data['skaters'], data['teams'], data['schedule'], target_date,
            advanced_stats=advanced_stats, injuries=injuries
        )

        goalie_features = self.feature_engineer.engineer_goalie_features(
            data['goalies'], data['teams'], data['schedule'], target_date
        )

        # Generate baseline projections
        skater_projections = self.project_skaters_baseline(skater_features)
        goalie_projections = self.project_goalies_baseline(goalie_features)

        # Filter to only players on today's slate
        if target_date:
            games_today = data['schedule'][data['schedule']['date'] == target_date]
            teams_playing = set(games_today['home_team'].tolist() + games_today['away_team'].tolist())

            skater_projections = skater_projections[
                skater_projections['team'].isin(teams_playing)
            ]
            goalie_projections = goalie_projections[
                goalie_projections['team'].isin(teams_playing)
            ]

        # Filter out injured players
        if filter_injuries and not injuries.empty:
            skater_count_before = len(skater_projections)
            goalie_count_before = len(goalie_projections)

            skater_projections = self._filter_injured(
                skater_projections, injuries, include_dtd=include_dtd
            )
            goalie_projections = self._filter_injured(
                goalie_projections, injuries, include_dtd=include_dtd
            )

            skaters_filtered = skater_count_before - len(skater_projections)
            goalies_filtered = goalie_count_before - len(goalie_projections)

            if skaters_filtered > 0 or goalies_filtered > 0:
                print(f"  Filtered {skaters_filtered} injured skaters, {goalies_filtered} injured goalies")

        print(f"  Projected {len(skater_projections)} skaters")
        print(f"  Projected {len(goalie_projections)} goalies")

        # Normalize positions (L/LW/R/RW -> W) for DraftKings compatibility
        if 'position' in skater_projections.columns:
            skater_projections['position'] = skater_projections['position'].apply(
                lambda x: 'W' if str(x).upper() in ('L', 'LW', 'R', 'RW') else str(x).upper()
            )

        return {
            'skaters': skater_projections,
            'goalies': goalie_projections
        }

    def _filter_injured(self, df: pd.DataFrame, injuries: pd.DataFrame,
                        include_dtd: bool = True) -> pd.DataFrame:
        """
        Remove injured players from projections.

        Args:
            df: Player projections DataFrame
            injuries: Injuries DataFrame from MoneyPuck
            include_dtd: If True, also filter Day-to-Day players

        Returns:
            DataFrame with injured players removed
        """
        if injuries.empty or 'injury_status' not in injuries.columns:
            return df

        # Determine which statuses to exclude
        if include_dtd:
            statuses_to_exclude = INJURY_STATUSES_EXCLUDE + ['DTD']
        else:
            statuses_to_exclude = INJURY_STATUSES_EXCLUDE

        # Filter injuries to relevant statuses
        injured = injuries[injuries['injury_status'].isin(statuses_to_exclude)]

        if injured.empty:
            return df

        # Try to filter by player_id first
        if 'player_id' in df.columns and 'player_id' in injured.columns:
            injured_ids = set(injured['player_id'].tolist())
            return df[~df['player_id'].isin(injured_ids)]

        # Otherwise filter by name matching
        if 'name' in df.columns and 'player_name' in injured.columns:
            injured_names = set(injured['player_name'].str.lower().str.strip().tolist())
            df_lower = df['name'].str.lower().str.strip()
            return df[~df_lower.isin(injured_names)]

        return df

    def get_top_plays(self, projections: Dict[str, pd.DataFrame],
                       n_skaters: int = 20, n_goalies: int = 5) -> Dict[str, pd.DataFrame]:
        """Get top projected plays."""
        return {
            'skaters': projections['skaters'].nlargest(n_skaters, 'projected_fpts')[
                ['name', 'team', 'position', 'projected_fpts', 'floor', 'ceiling',
                 'goals_pg', 'shots_pg', 'opponent', 'opp_softness']
            ],
            'goalies': projections['goalies'].nlargest(n_goalies, 'projected_fpts')[
                ['name', 'team', 'projected_fpts', 'floor', 'ceiling',
                 'win_rate', 'saves_pg', 'opponent', 'opp_offense_weakness']
            ]
        }


# Quick test
if __name__ == "__main__":
    from data_pipeline import NHLDataPipeline
    from datetime import datetime

    # Fetch data
    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(include_game_logs=False)

    # Generate projections
    model = NHLProjectionModel()
    today = datetime.now().strftime('%Y-%m-%d')

    projections = model.generate_projections(data, target_date=today)

    # Get top plays
    top_plays = model.get_top_plays(projections)

    print("\n" + "=" * 60)
    print("TOP SKATER PROJECTIONS")
    print("=" * 60)
    print(top_plays['skaters'].to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP GOALIE PROJECTIONS")
    print("=" * 60)
    print(top_plays['goalies'].to_string(index=False))
