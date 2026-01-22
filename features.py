"""
Feature engineering for NHL DFS projections.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class FeatureEngineer:
    """Engineer features for NHL DFS projections."""

    def __init__(self):
        pass

    def engineer_skater_features(self, skaters: pd.DataFrame, teams: pd.DataFrame,
                                   schedule: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Engineer features for skater projections.

        Args:
            skaters: DataFrame with skater season stats
            teams: DataFrame with team stats
            schedule: DataFrame with upcoming games
            target_date: Date to project for (filters schedule)

        Returns:
            DataFrame with engineered features ready for modeling
        """
        df = skaters.copy()

        if df.empty:
            return df

        # Filter to players with minimum games
        df = df[df['games_played'] >= 5].copy()

        # ==================== Per-Game Averages ====================
        gp = df['games_played'].replace(0, np.nan)

        # Core fantasy stats per game
        df['goals_pg'] = df['goals'] / gp
        df['assists_pg'] = df['assists'] / gp
        df['points_pg'] = df['points'] / gp
        df['shots_pg'] = df['shots'] / gp

        # Blocks - extract from realtime if available, otherwise estimate
        if 'blockedShots' in df.columns:
            df['blocks_pg'] = df['blockedShots'] / gp
        else:
            # Estimate blocks based on position (D blocks more)
            df['blocks_pg'] = np.where(df['position'] == 'D', 1.5, 0.5)

        # ==================== Power Play Features ====================
        if 'pp_points' in df.columns:
            df['pp_points_pg'] = df['pp_points'] / gp
            df['pp_share'] = df['pp_points'] / df['points'].replace(0, np.nan)
            df['pp_share'] = df['pp_share'].fillna(0)

        if 'pp_goals' in df.columns:
            df['pp_goals_pg'] = df['pp_goals'] / gp

        # ==================== Shorthanded Features ====================
        if 'sh_goals' in df.columns:
            df['sh_goals_pg'] = df['sh_goals'] / gp
        if 'sh_points' in df.columns:
            df['sh_points_pg'] = df['sh_points'] / gp

        # ==================== Shooting Efficiency ====================
        if 'shooting_pct' in df.columns:
            df['shooting_pct_adj'] = df['shooting_pct'].fillna(df['shooting_pct'].median())
        else:
            df['shooting_pct_adj'] = (df['goals'] / df['shots'].replace(0, np.nan)).fillna(0.08)

        # Shot volume tiers (for 5+ shot bonus potential)
        df['high_shot_volume'] = (df['shots_pg'] >= 3.5).astype(int)
        df['elite_shot_volume'] = (df['shots_pg'] >= 4.5).astype(int)

        # ==================== Upside/Bonus Potential ====================
        # Estimate probability of hitting bonuses based on per-game rates

        # 5+ shots bonus probability (based on shot rate)
        df['prob_5plus_shots'] = self._estimate_bonus_prob(df['shots_pg'], threshold=5, std_factor=1.5)

        # 3+ points bonus probability
        df['prob_3plus_points'] = self._estimate_bonus_prob(df['points_pg'], threshold=3, std_factor=1.0)

        # Hat trick probability (very rare)
        df['prob_hat_trick'] = self._estimate_bonus_prob(df['goals_pg'], threshold=3, std_factor=0.8)

        # ==================== Position Encoding ====================
        df['is_center'] = (df['position'] == 'C').astype(int)
        df['is_wing'] = df['position'].isin(['L', 'R', 'LW', 'RW']).astype(int)
        df['is_defense'] = (df['position'] == 'D').astype(int)

        # ==================== Time on Ice ====================
        if 'toi_per_game' in df.columns:
            # Convert from seconds to minutes if needed
            toi = df['toi_per_game']
            if toi.median() > 100:  # Likely in seconds
                df['toi_minutes'] = toi / 60
            else:
                df['toi_minutes'] = toi

            # TOI relative to position average
            pos_avg_toi = df.groupby('position')['toi_minutes'].transform('mean')
            df['toi_vs_position'] = df['toi_minutes'] / pos_avg_toi

        # ==================== Opponent Adjustments ====================
        if not teams.empty and not schedule.empty:
            df = self._add_opponent_features(df, teams, schedule, target_date)

        # ==================== Consistency Score ====================
        # Players with higher per-game averages relative to GP are more consistent
        df['consistency'] = df['points_pg'] * np.log1p(df['games_played'])

        return df

    def engineer_goalie_features(self, goalies: pd.DataFrame, teams: pd.DataFrame,
                                   schedule: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
        """Engineer features for goalie projections."""
        df = goalies.copy()

        if df.empty:
            return df

        # Filter to goalies with minimum starts
        if 'games_started' in df.columns:
            df = df[df['games_started'] >= 3].copy()
            gs = df['games_started'].replace(0, np.nan)
        else:
            df = df[df['games_played'] >= 3].copy()
            gs = df['games_played'].replace(0, np.nan)

        # ==================== Core Goalie Metrics ====================
        if 'saves' in df.columns:
            df['saves_pg'] = df['saves'] / gs

        if 'goals_against' in df.columns:
            df['ga_pg'] = df['goals_against'] / gs

        if 'shots_against' in df.columns:
            df['sa_pg'] = df['shots_against'] / gs

        # ==================== Win Rate ====================
        if 'wins' in df.columns:
            df['win_rate'] = df['wins'] / gs

        # ==================== Save Percentage Quality ====================
        if 'save_pct' in df.columns:
            # Normalize save pct (league average ~0.905)
            df['save_pct_above_avg'] = df['save_pct'] - 0.905

        # ==================== Workload Features ====================
        # High workload = more saves but also more GA risk
        if 'sa_pg' in df.columns:
            df['high_workload'] = (df['sa_pg'] >= 30).astype(int)

            # 35+ saves bonus probability
            df['prob_35plus_saves'] = self._estimate_bonus_prob(df['sa_pg'], threshold=35, std_factor=5)

        # ==================== GAA Quality ====================
        if 'gaa' in df.columns:
            # Lower is better, normalize
            df['gaa_quality'] = 3.0 - df['gaa']  # Positive = better than 3.0 GAA

        # ==================== Team Quality ====================
        # Goalies on good teams win more
        if not teams.empty and 'team' in df.columns:
            team_stats = teams.set_index('team')[['goals_for_per_game', 'goals_against_per_game']].to_dict('index')

            df['team_gf_pg'] = df['team'].map(lambda x: team_stats.get(x, {}).get('goals_for_per_game', 3.0))
            df['team_ga_pg'] = df['team'].map(lambda x: team_stats.get(x, {}).get('goals_against_per_game', 3.0))

            # Good team = scores a lot, doesn't give up much
            df['team_quality'] = df['team_gf_pg'] - df['team_ga_pg']

        # ==================== Opponent Adjustments ====================
        if not teams.empty and not schedule.empty:
            df = self._add_goalie_opponent_features(df, teams, schedule, target_date)

        return df

    def _add_opponent_features(self, df: pd.DataFrame, teams: pd.DataFrame,
                                schedule: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
        """Add opponent-based features for skaters."""
        # Get team stats indexed by team code
        team_stats = teams.set_index('team').to_dict('index')

        # Get games for target date
        if target_date:
            games = schedule[schedule['date'] == target_date]
        else:
            games = schedule.head(20)  # Default to upcoming games

        # Build opponent mapping
        opponent_map = {}
        for _, game in games.iterrows():
            opponent_map[game['home_team']] = {
                'opponent': game['away_team'],
                'is_home': True
            }
            opponent_map[game['away_team']] = {
                'opponent': game['home_team'],
                'is_home': False
            }

        # Add opponent features
        def get_opponent_stats(row):
            team = row.get('team', '')
            if team in opponent_map:
                opp = opponent_map[team]['opponent']
                opp_stats = team_stats.get(opp, {})
                return pd.Series({
                    'opponent': opp,
                    'is_home': opponent_map[team]['is_home'],
                    'opp_ga_pg': opp_stats.get('goals_against_per_game', 3.0),
                    'opp_gf_pg': opp_stats.get('goals_for_per_game', 3.0),
                })
            return pd.Series({
                'opponent': None,
                'is_home': None,
                'opp_ga_pg': 3.0,
                'opp_gf_pg': 3.0,
            })

        opp_features = df.apply(get_opponent_stats, axis=1)
        df = pd.concat([df, opp_features], axis=1)

        # Opponent adjustment factor (higher = softer opponent)
        # Teams that give up more goals are easier to score against
        df['opp_softness'] = df['opp_ga_pg'] / 3.0  # Normalized to league average ~3.0

        return df

    def _add_goalie_opponent_features(self, df: pd.DataFrame, teams: pd.DataFrame,
                                        schedule: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
        """Add opponent-based features for goalies."""
        team_stats = teams.set_index('team').to_dict('index')

        if target_date:
            games = schedule[schedule['date'] == target_date]
        else:
            games = schedule.head(20)

        opponent_map = {}
        for _, game in games.iterrows():
            opponent_map[game['home_team']] = {
                'opponent': game['away_team'],
                'is_home': True
            }
            opponent_map[game['away_team']] = {
                'opponent': game['home_team'],
                'is_home': False
            }

        def get_opp_offense(row):
            team = row.get('team', '')
            if team in opponent_map:
                opp = opponent_map[team]['opponent']
                opp_stats = team_stats.get(opp, {})
                return pd.Series({
                    'opponent': opp,
                    'is_home': opponent_map[team]['is_home'],
                    'opp_gf_pg': opp_stats.get('goals_for_per_game', 3.0),
                })
            return pd.Series({
                'opponent': None,
                'is_home': None,
                'opp_gf_pg': 3.0,
            })

        opp_features = df.apply(get_opp_offense, axis=1)
        df = pd.concat([df, opp_features], axis=1)

        # For goalies, facing weak offense is good
        df['opp_offense_weakness'] = 3.0 / df['opp_gf_pg'].replace(0, 3.0)

        return df

    def _estimate_bonus_prob(self, per_game_avg: pd.Series, threshold: float, std_factor: float) -> pd.Series:
        """
        Estimate probability of hitting a bonus threshold based on per-game average.

        Uses a simple normal approximation.
        """
        # Assume standard deviation is proportional to mean
        std = per_game_avg * std_factor
        std = std.replace(0, 0.1)  # Avoid division by zero

        # Z-score for threshold
        z = (threshold - per_game_avg) / std

        # Convert to probability (using normal CDF approximation)
        # P(X >= threshold) = 1 - CDF(z)
        prob = 1 / (1 + np.exp(1.7 * z))  # Logistic approximation to normal CDF

        return prob.clip(0, 1)

    def get_feature_columns(self, player_type: str = 'skater') -> List[str]:
        """Get list of feature columns for modeling."""
        if player_type == 'skater':
            return [
                'goals_pg', 'assists_pg', 'points_pg', 'shots_pg', 'blocks_pg',
                'pp_points_pg', 'pp_share', 'shooting_pct_adj',
                'high_shot_volume', 'elite_shot_volume',
                'prob_5plus_shots', 'prob_3plus_points', 'prob_hat_trick',
                'is_center', 'is_wing', 'is_defense',
                'toi_minutes', 'toi_vs_position',
                'opp_softness', 'is_home', 'consistency'
            ]
        else:  # goalie
            return [
                'saves_pg', 'ga_pg', 'sa_pg', 'win_rate',
                'save_pct_above_avg', 'high_workload', 'prob_35plus_saves',
                'gaa_quality', 'team_quality', 'team_gf_pg', 'team_ga_pg',
                'opp_gf_pg', 'opp_offense_weakness', 'is_home'
            ]


# Quick test
if __name__ == "__main__":
    from data_pipeline import NHLDataPipeline

    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(include_game_logs=False)

    fe = FeatureEngineer()

    print("Engineering skater features...")
    skater_features = fe.engineer_skater_features(
        data['skaters'], data['teams'], data['schedule']
    )
    print(f"Skater features shape: {skater_features.shape}")
    print(f"Sample features:\n{skater_features[['name', 'goals_pg', 'shots_pg', 'prob_5plus_shots', 'opp_softness']].head(10)}")

    print("\nEngineering goalie features...")
    goalie_features = fe.engineer_goalie_features(
        data['goalies'], data['teams'], data['schedule']
    )
    print(f"Goalie features shape: {goalie_features.shape}")
    print(f"Sample features:\n{goalie_features[['name', 'saves_pg', 'win_rate', 'opp_offense_weakness']].head(5)}")
