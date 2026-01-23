"""
Feature engineering for NHL DFS projections.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List

from config import (
    LEAGUE_AVG_XGF_60, LEAGUE_AVG_XGA_60, LEAGUE_AVG_CF_PCT, LEAGUE_AVG_PDO,
    PDO_HIGH_THRESHOLD, PDO_LOW_THRESHOLD, PDO_REGRESSION_FACTOR,
    HOT_STREAK_THRESHOLD, COLD_STREAK_THRESHOLD, STREAK_ADJUSTMENT_FACTOR,
    INJURY_STATUSES_EXCLUDE
)


class FeatureEngineer:
    """Engineer features for NHL DFS projections."""

    def __init__(self):
        pass

    def engineer_skater_features(self, skaters: pd.DataFrame, teams: pd.DataFrame,
                                   schedule: pd.DataFrame, target_date: Optional[str] = None,
                                   advanced_stats: Optional[Dict[str, pd.DataFrame]] = None,
                                   injuries: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Engineer features for skater projections.

        Args:
            skaters: DataFrame with skater season stats
            teams: DataFrame with team stats
            schedule: DataFrame with upcoming games
            target_date: Date to project for (filters schedule)
            advanced_stats: Dict with 'team' and 'player' advanced stats from NST
            injuries: DataFrame with current injury data

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

        # ==================== Advanced Stats Features (xG, Form, PDO) ====================
        if advanced_stats is not None:
            # Extract team and player advanced stats
            team_adv = advanced_stats.get('team', {})
            player_adv = advanced_stats.get('player', {})

            # Add xG features
            df = self._add_xg_features(df, team_adv, schedule, target_date)

            # Add recent form features
            df = self._add_recent_form_features(df, team_adv)

            # Add PDO regression features
            df = self._add_pdo_regression_features(df, team_adv)

        # ==================== Injury Context Features ====================
        if injuries is not None and not injuries.empty:
            df = self._add_injury_context_features(df, injuries)

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

    # ==================== Advanced Analytics Features ====================

    def _add_xg_features(self, df: pd.DataFrame, team_adv_stats: Dict[str, pd.DataFrame],
                          schedule: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add expected goals (xG) features.

        Features added:
            - team_xgf_60: Team's xG for per 60 minutes
            - team_xga_60: Team's xG against per 60 minutes
            - opp_xga_60: Opponent's xG against per 60 (soft defense = good)
            - xg_matchup_boost: Combined xG matchup advantage
        """
        # Get 5v5 team stats
        team_5v5 = team_adv_stats.get('5v5', pd.DataFrame())

        if team_5v5.empty or 'team' not in df.columns:
            # Set defaults
            df['team_xgf_60'] = LEAGUE_AVG_XGF_60
            df['team_xga_60'] = LEAGUE_AVG_XGA_60
            df['opp_xga_60'] = LEAGUE_AVG_XGA_60
            df['xg_matchup_boost'] = 1.0
            return df

        # Build team xG lookup
        if 'team' in team_5v5.columns:
            team_5v5_indexed = team_5v5.set_index('team')
        else:
            df['team_xgf_60'] = LEAGUE_AVG_XGF_60
            df['team_xga_60'] = LEAGUE_AVG_XGA_60
            df['opp_xga_60'] = LEAGUE_AVG_XGA_60
            df['xg_matchup_boost'] = 1.0
            return df

        # Map team xG stats
        xgf_col = 'xgf' if 'xgf' in team_5v5_indexed.columns else None
        xga_col = 'xga' if 'xga' in team_5v5_indexed.columns else None

        if xgf_col:
            df['team_xgf_60'] = df['team'].map(
                lambda t: team_5v5_indexed.loc[t, xgf_col] if t in team_5v5_indexed.index else LEAGUE_AVG_XGF_60
            )
        else:
            df['team_xgf_60'] = LEAGUE_AVG_XGF_60

        if xga_col:
            df['team_xga_60'] = df['team'].map(
                lambda t: team_5v5_indexed.loc[t, xga_col] if t in team_5v5_indexed.index else LEAGUE_AVG_XGA_60
            )
        else:
            df['team_xga_60'] = LEAGUE_AVG_XGA_60

        # Get opponent xGA (opponent's defensive weakness)
        if 'opponent' in df.columns and xga_col:
            df['opp_xga_60'] = df['opponent'].map(
                lambda t: team_5v5_indexed.loc[t, xga_col] if pd.notna(t) and t in team_5v5_indexed.index else LEAGUE_AVG_XGA_60
            )
        else:
            df['opp_xga_60'] = LEAGUE_AVG_XGA_60

        # Calculate matchup boost
        # Good offense (high xGF) vs bad defense (high xGA) = boost
        team_offense_factor = df['team_xgf_60'] / LEAGUE_AVG_XGF_60
        opp_defense_weakness = df['opp_xga_60'] / LEAGUE_AVG_XGA_60

        df['xg_matchup_boost'] = (team_offense_factor * opp_defense_weakness).clip(0.8, 1.3)

        return df

    def _add_recent_form_features(self, df: pd.DataFrame,
                                   team_adv_stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add recent form features based on last N games.

        Features added:
            - team_form_xgf: Recent xGF vs season average
            - team_form_cf: Recent CF% vs season average
            - team_hot_streak: 1 if team is hot (>15% above avg)
            - team_cold_streak: 1 if team is cold (<15% below avg)
        """
        team_season = team_adv_stats.get('5v5', pd.DataFrame())
        team_recent = team_adv_stats.get('recent_form', pd.DataFrame())

        # Initialize defaults
        df['team_form_xgf'] = 1.0
        df['team_form_cf'] = 1.0
        df['team_hot_streak'] = 0
        df['team_cold_streak'] = 0

        if team_season.empty or team_recent.empty or 'team' not in df.columns:
            return df

        # Need team column in both DataFrames
        if 'team' not in team_season.columns or 'team' not in team_recent.columns:
            return df

        # Index by team
        season_idx = team_season.set_index('team')
        recent_idx = team_recent.set_index('team')

        def get_form_ratio(team, col):
            """Get recent/season ratio for a stat."""
            if team not in season_idx.index or team not in recent_idx.index:
                return 1.0
            if col not in season_idx.columns or col not in recent_idx.columns:
                return 1.0

            season_val = season_idx.loc[team, col]
            recent_val = recent_idx.loc[team, col]

            if pd.isna(season_val) or pd.isna(recent_val) or season_val == 0:
                return 1.0

            return recent_val / season_val

        # Calculate form ratios
        if 'xgf' in season_idx.columns:
            df['team_form_xgf'] = df['team'].apply(lambda t: get_form_ratio(t, 'xgf'))

        if 'cf_pct' in season_idx.columns:
            df['team_form_cf'] = df['team'].apply(lambda t: get_form_ratio(t, 'cf_pct'))

        # Flag hot/cold streaks
        df['team_hot_streak'] = (df['team_form_xgf'] >= HOT_STREAK_THRESHOLD).astype(int)
        df['team_cold_streak'] = (df['team_form_xgf'] <= COLD_STREAK_THRESHOLD).astype(int)

        return df

    def _add_pdo_regression_features(self, df: pd.DataFrame,
                                      team_adv_stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add PDO regression features.

        PDO = SH% + SV% (typically ~100, regresses to mean)
        High PDO teams are due for regression, low PDO teams should improve.

        Features added:
            - team_pdo: Team's current PDO
            - pdo_regression_flag: 1 if PDO suggests regression
            - pdo_adj_factor: Adjustment factor for projections
        """
        team_5v5 = team_adv_stats.get('5v5', pd.DataFrame())

        # Initialize defaults
        df['team_pdo'] = LEAGUE_AVG_PDO
        df['pdo_regression_flag'] = 0
        df['pdo_adj_factor'] = 1.0

        if team_5v5.empty or 'team' not in df.columns:
            return df

        if 'pdo' not in team_5v5.columns or 'team' not in team_5v5.columns:
            return df

        team_idx = team_5v5.set_index('team')

        # Map PDO
        df['team_pdo'] = df['team'].map(
            lambda t: team_idx.loc[t, 'pdo'] if t in team_idx.index else LEAGUE_AVG_PDO
        )

        # Flag regression candidates
        df['pdo_regression_flag'] = (
            (df['team_pdo'] >= PDO_HIGH_THRESHOLD) | (df['team_pdo'] <= PDO_LOW_THRESHOLD)
        ).astype(int)

        # Calculate adjustment factor
        # High PDO -> expect regression down -> reduce projections slightly
        # Low PDO -> expect regression up -> increase projections slightly
        def calc_pdo_adj(pdo):
            if pdo >= PDO_HIGH_THRESHOLD:
                # Reduce projections (team is over-performing)
                return 1.0 - PDO_REGRESSION_FACTOR * ((pdo - LEAGUE_AVG_PDO) / 10)
            elif pdo <= PDO_LOW_THRESHOLD:
                # Increase projections (team is under-performing)
                return 1.0 + PDO_REGRESSION_FACTOR * ((LEAGUE_AVG_PDO - pdo) / 10)
            return 1.0

        df['pdo_adj_factor'] = df['team_pdo'].apply(calc_pdo_adj).clip(0.9, 1.1)

        return df

    def _add_injury_context_features(self, df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
        """
        Add injury context features (opportunity boost from teammate injuries).

        When key players are injured, remaining players may see increased:
        - Ice time
        - Power play time
        - Scoring opportunities

        Features added:
            - team_injury_count: Number of injured players on team
            - opportunity_boost: Multiplier based on teammate injuries
        """
        df['team_injury_count'] = 0
        df['opportunity_boost'] = 1.0

        if injuries.empty or 'team' not in df.columns:
            return df

        if 'team' not in injuries.columns or 'injury_status' not in injuries.columns:
            return df

        # Count significant injuries per team (exclude DTD)
        severe_injuries = injuries[injuries['injury_status'].isin(INJURY_STATUSES_EXCLUDE)]

        if severe_injuries.empty:
            return df

        injury_counts = severe_injuries.groupby('team').size()

        # Map injury counts
        df['team_injury_count'] = df['team'].map(
            lambda t: injury_counts.get(t, 0)
        )

        # Calculate opportunity boost
        # More injuries = more opportunity for healthy players
        # Each injured teammate adds ~3% opportunity boost, capped at 15%
        df['opportunity_boost'] = (1.0 + df['team_injury_count'] * 0.03).clip(1.0, 1.15)

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

    def get_feature_columns(self, player_type: str = 'skater',
                             include_advanced: bool = True) -> List[str]:
        """
        Get list of feature columns for modeling.

        Args:
            player_type: 'skater' or 'goalie'
            include_advanced: Include xG, form, PDO features

        Returns:
            List of feature column names
        """
        if player_type == 'skater':
            base_features = [
                'goals_pg', 'assists_pg', 'points_pg', 'shots_pg', 'blocks_pg',
                'pp_points_pg', 'pp_share', 'shooting_pct_adj',
                'high_shot_volume', 'elite_shot_volume',
                'prob_5plus_shots', 'prob_3plus_points', 'prob_hat_trick',
                'is_center', 'is_wing', 'is_defense',
                'toi_minutes', 'toi_vs_position',
                'opp_softness', 'is_home', 'consistency'
            ]

            if include_advanced:
                advanced_features = [
                    # xG features
                    'team_xgf_60', 'team_xga_60', 'opp_xga_60', 'xg_matchup_boost',
                    # Form features
                    'team_form_xgf', 'team_form_cf', 'team_hot_streak', 'team_cold_streak',
                    # PDO features
                    'team_pdo', 'pdo_regression_flag', 'pdo_adj_factor',
                    # Injury context
                    'team_injury_count', 'opportunity_boost'
                ]
                return base_features + advanced_features

            return base_features

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
