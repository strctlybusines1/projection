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
    SKATER_SCORING, SKATER_BONUSES, GOALIE_SCORING, GOALIE_BONUSES
)
from features import FeatureEngineer


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
        Includes bonus expected value.
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

        # Apply opponent adjustment
        opp_softness = row.get('opp_softness', 1.0)
        if pd.notna(opp_softness):
            # Soft opponent boosts scoring stats
            scoring_boost = (opp_softness - 1.0) * 0.15 + 1.0  # Dampen the effect
            expected_pts *= scoring_boost

        # Home ice advantage (small boost)
        if row.get('is_home') == True:
            expected_pts *= 1.02

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
        df['floor'] = df['projected_fpts'] * 0.4  # Bad game
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
        df['floor'] = df['projected_fpts'] * 0.3  # Bad game (loss, high GA)
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
                              use_tabpfn: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Generate full projections for a slate.

        Args:
            data: Dict with 'skaters', 'goalies', 'teams', 'schedule' DataFrames
            target_date: Date to project for (YYYY-MM-DD)
            use_tabpfn: Whether to use TabPFN (requires historical fantasy point data)

        Returns:
            Dict with 'skaters' and 'goalies' projection DataFrames
        """
        print(f"Generating projections for {target_date or 'upcoming games'}...")

        # Engineer features
        skater_features = self.feature_engineer.engineer_skater_features(
            data['skaters'], data['teams'], data['schedule'], target_date
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

        print(f"  Projected {len(skater_projections)} skaters")
        print(f"  Projected {len(goalie_projections)} goalies")

        return {
            'skaters': skater_projections,
            'goalies': goalie_projections
        }

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
