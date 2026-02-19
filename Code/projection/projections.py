"""
NHL DFS Projection Model using TabPFN.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from tabpfn import TabPFNRegressor

import config
from config import (
    calculate_skater_fantasy_points, calculate_goalie_fantasy_points,
    SKATER_SCORING, SKATER_BONUSES, GOALIE_SCORING, GOALIE_BONUSES,
    INJURY_STATUSES_EXCLUDE, STREAK_ADJUSTMENT_FACTOR,
    LEAGUE_AVG_HD_SHARE, SHOT_QUALITY_ADJ_CAP,
)
from features import FeatureEngineer

# ==================== Backtest-Derived Bias Corrections ====================
# RECALIBRATED 2/19/26: Analysis of systematic biases across positions and salary tiers:
#   - Position biases: C: +1.17, W: +0.88, D: +2.43, G: +0.26
#   - Salary tier biases: <$3k: +1.30, $3-5k: +1.56, $5-7k: +1.51, $7k+: +1.85
#   - Projection tier biases: <5: +0.11, 5-8: +1.47, 8-12: +2.33, 12+: +3.30
# Previous GLOBAL_BIAS_CORRECTION = 0.80 was insufficient. New approach:
#   1. Data-derived position corrections (multiplicative)
#   2. Salary-tier subtractive corrections (fixed FPTS bias per tier)
#   3. More aggressive mean regression for high projections (10.0 threshold, 85% blend)

GLOBAL_BIAS_CORRECTION = 0.75  # Reduced from 0.80 to account for tier-specific corrections

# Position-specific bias corrections (skaters) — data-derived from systematic analysis
# Defensemen have the worst bias (+2.43), so D gets strongest correction (0.60 multiplier)
# Centers and Wings moderate bias; C gets 0.75, W gets 0.80
POSITION_BIAS_CORRECTION = {
    'C': 0.75,   # Centers — bias +1.17, multiplicative correction
    'L': 0.80,   # Left wings — bias ~+0.88
    'LW': 0.80,  # Left wings (alternate code)
    'R': 0.80,   # Right wings — bias ~+0.88
    'RW': 0.80,  # Right wings (alternate code)
    'W': 0.80,   # Wings — bias +0.88, multiplicative correction
    'D': 0.60,   # Defensemen — bias +2.43, strongest correction needed
}

# Goalie bias correction (actual bias was +0.26 FPTS per backtest 2/19/26)
# Changed from 0.40 (60% reduction) to 0.85 (15% mild correction)
GOALIE_BIAS_CORRECTION = 0.85

# Floor multiplier (reduced from 0.4 - 30.5% were below floor)
FLOOR_MULTIPLIER = 0.25

# ==================== Multiplicative Adjustment Cap ====================
# 7 multiplicative adjustments can compound (signal_matchup * xg_matchup * streak
# * pdo * opportunity * role * home) — this inflates high projections by ~25%
MAX_MULTIPLICATIVE_SWING = 0.15  # Cap total adjustment multiplier at ±15%

# ==================== High-Projection Mean Regression ====================
# Lowered threshold to 10.0 (was 14.0) since projections 12+ have +3.30 bias
# Increased blend to 85% (was 80%) for more aggressive regression
SKATER_HIGH_PROJ_THRESHOLD = 10.0
SKATER_HIGH_PROJ_BLEND = 0.85       # keep 85% of projection, blend 15% toward mean
SKATER_LEAGUE_MEAN_FPTS = 6.0       # approximate league average DK pts

GOALIE_HIGH_PROJ_THRESHOLD = 12.0
GOALIE_HIGH_PROJ_BLEND = 0.80
GOALIE_LEAGUE_MEAN_FPTS = 9.0

# ==================== Salary Tier Bias Corrections ====================
# Subtractive corrections for salary tier biases (applied AFTER multiplicative corrections)
# Measured biases: <$3k: +1.30, $3-5k: +1.56, $5-7k: +1.51, $7k+: +1.85
SALARY_TIER_BIAS_CORRECTION = {
    'under_3k': 1.30,      # Subtract 1.30 FPTS from projections <$3k salary
    '3k_to_5k': 1.56,      # Subtract 1.56 FPTS from projections $3-5k
    '5k_to_7k': 1.51,      # Subtract 1.51 FPTS from projections $5-7k
    '7k_plus': 1.85,       # Subtract 1.85 FPTS from projections $7k+
}

# ==================== Goalie Projection Cap ====================
GOALIE_PROJECTION_CAP = 16.0  # Raised from 14.0 — was overcorrecting on big goalie nights


# ==================== Salary Tier Helper Function ====================
def get_salary_tier_bias_correction(salary: float) -> float:
    """
    Determine salary tier and return subtractive bias correction.

    Args:
        salary: Player salary in dollars (e.g., 3500, 5500)

    Returns:
        Bias correction amount to subtract from projected FPTS
    """
    if pd.isna(salary) or salary <= 0:
        return 0.0  # No correction if salary unknown

    if salary < 3000:
        return SALARY_TIER_BIAS_CORRECTION['under_3k']
    elif salary < 5000:
        return SALARY_TIER_BIAS_CORRECTION['3k_to_5k']
    elif salary < 7000:
        return SALARY_TIER_BIAS_CORRECTION['5k_to_7k']
    else:
        return SALARY_TIER_BIAS_CORRECTION['7k_plus']


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
            self.skater_model = TabPFNRegressor(ignore_pretraining_limits=True)
            self.goalie_model = TabPFNRegressor(ignore_pretraining_limits=True)
            self._models_initialized = True
            print("Models initialized.")

    def calculate_expected_fantasy_points_skater(self, row: pd.Series) -> float:
        """
        Calculate expected DraftKings fantasy points for a skater based on per-game averages
        or rate-based (dk_pts_per_60 * expected_toi_minutes/60) when USE_DK_PER_TOI_PROJECTION.
        Includes bonus expected value and advanced stat adjustments.
        """
        use_rate = (
            config.USE_DK_PER_TOI_PROJECTION
            and pd.notna(row.get('dk_pts_per_60'))
            and pd.notna(row.get('expected_toi_minutes'))
            and row.get('expected_toi_minutes', 0) > 0
        )

        if use_rate:
            # Rate-based: base = dk_pts_per_60 * (expected_toi_minutes / 60)
            expected_pts = row['dk_pts_per_60'] * (row['expected_toi_minutes'] / 60.0)
            # When expected_toi_minutes == toi_minutes (no situation TOI in data), scale so comparison shows difference
            toi_min = row.get('toi_minutes')
            if (pd.notna(toi_min) and toi_min > 0 and
                    abs(float(row['expected_toi_minutes']) - float(toi_min)) < 0.01):
                expected_pts *= config.RATE_BASED_SAME_TOI_SCALE
            # Bonuses scale with TOI (same expected value of bonuses as per-game)
            goals = row.get('goals_pg', 0)
            assists = row.get('assists_pg', 0)
            shots = row.get('shots_pg', 0)
            blocks = row.get('blocks_pg', 0)
            prob_5_shots = row.get('prob_5plus_shots', 0)
            prob_3_points = row.get('prob_3plus_points', 0)
            prob_hat_trick = row.get('prob_hat_trick', 0)
            prob_3_blocks = min(blocks / 3, 1) * 0.3 if blocks else 0
            expected_pts += prob_5_shots * SKATER_BONUSES['five_plus_shots']
            expected_pts += prob_3_points * SKATER_BONUSES['three_plus_points']
            expected_pts += prob_hat_trick * SKATER_BONUSES['hat_trick']
            expected_pts += prob_3_blocks * SKATER_BONUSES['three_plus_blocks']
        else:
            # Per-game base
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

        # ==================== Collect Multiplicative Adjustments ====================
        # Collect all multipliers, then clamp to ±MAX_MULTIPLICATIVE_SWING before applying

        combined_mult = 1.0

        # Signal-weighted matchup adjustment
        signal_boost = row.get('signal_matchup_boost', 1.0)
        if pd.notna(signal_boost):
            combined_mult *= signal_boost

        # xG matchup boost
        xg_matchup_boost = row.get('xg_matchup_boost', 1.0)
        if pd.notna(xg_matchup_boost):
            combined_mult *= xg_matchup_boost

        # Hot/cold streak adjustment
        if row.get('team_hot_streak', 0) == 1:
            combined_mult *= (1.0 + STREAK_ADJUSTMENT_FACTOR)
        elif row.get('team_cold_streak', 0) == 1:
            combined_mult *= (1.0 - STREAK_ADJUSTMENT_FACTOR)

        # PDO regression adjustment
        pdo_adj = row.get('pdo_adj_factor', 1.0)
        if pd.notna(pdo_adj):
            combined_mult *= pdo_adj

        # Opportunity boost from teammate injuries
        opp_boost = row.get('opportunity_boost', 1.0)
        if pd.notna(opp_boost):
            combined_mult *= opp_boost

        # Line role multiplier (PP1/Line1 from lines_data)
        role_mult = row.get('role_multiplier', 1.0)
        if pd.notna(role_mult):
            combined_mult *= role_mult

        # Home ice advantage
        if row.get('is_home') == True:
            combined_mult *= 1.01

        # Clamp combined multiplier to prevent compounding inflation
        combined_mult = max(1.0 - MAX_MULTIPLICATIVE_SWING,
                           min(1.0 + MAX_MULTIPLICATIVE_SWING, combined_mult))

        expected_pts *= combined_mult

        # ==================== Backtest Bias Corrections ====================
        # Apply global bias correction
        expected_pts *= GLOBAL_BIAS_CORRECTION

        # Apply position-specific bias correction (multiplicative)
        position = row.get('position', 'C')
        pos_correction = POSITION_BIAS_CORRECTION.get(position, 0.75)
        expected_pts *= pos_correction

        # Apply salary-tier bias correction (subtractive)
        salary = row.get('salary')
        salary_correction = get_salary_tier_bias_correction(salary)
        expected_pts -= salary_correction

        # ==================== High-Projection Mean Regression ====================
        if expected_pts > SKATER_HIGH_PROJ_THRESHOLD:
            expected_pts = (SKATER_HIGH_PROJ_BLEND * expected_pts +
                           (1 - SKATER_HIGH_PROJ_BLEND) * SKATER_LEAGUE_MEAN_FPTS)

        return expected_pts

    def calculate_expected_fantasy_points_goalie(self, row: pd.Series) -> float:
        """
        Calculate expected DraftKings fantasy points for a goalie.
        When opponent shot mix and goalie (team) save % by type are available,
        uses danger-weighted expected saves and GA; otherwise uses raw saves_pg / ga_pg.

        IMPROVED (2/19/26):
        - Matchup-adjusted win probability (considers team offense vs opp defense)
        - Poisson-based shutout probability instead of linear formula
        - Opponent-aware 35+ saves bonus
        - Reduced GOALIE_BIAS_CORRECTION from 0.40 to 0.85
        """
        sa_pg = row.get('sa_pg', 28)

        # ==================== Matchup-Adjusted Win Probability ====================
        # Instead of just season-average win_rate, adjust for opponent strength
        base_win_rate = row.get('win_rate', 0.5)

        # If we have team offense and opponent defense data, use them
        team_goals_pg = row.get('team_goals_pg', 0)
        opp_ga_pg = row.get('opp_ga_pg', 0)
        league_avg_gf = row.get('league_avg_goals_for', 2.85)  # Approx NHL average
        league_avg_ga = row.get('league_avg_goals_against', 2.85)

        if pd.notna(team_goals_pg) and pd.notna(opp_ga_pg) and league_avg_gf > 0 and league_avg_ga > 0:
            # Simple log5 method: adjust win prob based on relative strength
            team_strength = team_goals_pg / league_avg_gf
            opp_strength = opp_ga_pg / league_avg_ga

            # Matchup adjustment: team is stronger than average vs weak defense = higher win prob
            matchup_adjustment = (team_strength / opp_strength) if opp_strength > 0 else 1.0
            matchup_adjustment = np.clip(matchup_adjustment, 0.5, 2.0)  # Bound to reasonable range

            adjusted_win_prob = base_win_rate * matchup_adjustment
            # Clip to [0.2, 0.8] to avoid extreme projections
            win_rate = np.clip(adjusted_win_prob, 0.2, 0.8)
        else:
            win_rate = base_win_rate

        # Opponent shot mix + goalie team save % by type → expected saves and GA
        opp_hd = row.get('opp_HD_share')
        opp_md = row.get('opp_MD_share')
        opp_ld = row.get('opp_LD_share')
        sv_hd = row.get('team_HDSV_pct')
        sv_md = row.get('team_MDSV_pct')
        sv_ld = row.get('team_LDSV_pct')
        use_danger_weighted = all(
            pd.notna(x) for x in [opp_hd, opp_md, opp_ld, sv_hd, sv_md, sv_ld]
        ) and (abs((opp_hd or 0) + (opp_md or 0) + (opp_ld or 0) - 1.0) < 0.01)

        if use_danger_weighted:
            # Expected save rate vs this opponent's shot mix
            expected_save_rate = opp_hd * sv_hd + opp_md * sv_md + opp_ld * sv_ld
            expected_save_rate = np.clip(expected_save_rate, 0.0, 1.0)
            saves = sa_pg * expected_save_rate
            ga = sa_pg * (1.0 - expected_save_rate)
        else:
            saves = row.get('saves_pg', 25)
            ga = row.get('ga_pg', 2.5)

        # Base expected points
        expected_pts = (
            saves * GOALIE_SCORING['save'] +
            ga * GOALIE_SCORING['goal_against'] +
            win_rate * GOALIE_SCORING['win']
        )

        # OT loss expected value (rough: ~25% of non-wins are OTL, per backtest data)
        ot_loss_rate = (1 - win_rate) * 0.25
        expected_pts += ot_loss_rate * GOALIE_SCORING['overtime_loss']

        # ==================== Poisson-Based Shutout Probability ====================
        # P(0 goals) = e^(-lambda), where lambda is expected GA
        # This is more principled than linear formula
        if ga > 0:
            shutout_prob = np.exp(-ga)
        else:
            shutout_prob = 0.5  # Conservative if ga==0

        shutout_prob = np.clip(shutout_prob, 0.0, 0.15)  # Shutouts are rare; cap at 15%
        expected_pts += shutout_prob * GOALIE_SCORING['shutout_bonus']

        # ==================== 35+ Saves Bonus (Opponent-Aware) ====================
        # If facing high-shot-volume team, 35+ saves bonus should be higher
        opp_shots_pg = row.get('opp_shots_pg', 28)  # Opponent's shot volume

        # Base probability from stats if available
        prob_35_saves = row.get('prob_35plus_saves', 0)

        # If we have shot volume data, adjust probability upward for high-shot teams
        if pd.notna(opp_shots_pg) and opp_shots_pg > 0 and prob_35_saves > 0:
            # Higher shot volume → more likely to get 35+ saves
            # Scale adjustment: if opp shoots 33 shots/game (above 28 avg), boost probability
            shot_volume_adj = (opp_shots_pg / 28.0) if opp_shots_pg > 0 else 1.0
            shot_volume_adj = np.clip(shot_volume_adj, 0.8, 1.5)  # Bound to reasonable range
            prob_35_saves = prob_35_saves * shot_volume_adj
            prob_35_saves = np.clip(prob_35_saves, 0.0, 0.30)  # 35+ saves is still rare; cap at 30%

        expected_pts += prob_35_saves * GOALIE_BONUSES['thirty_five_plus_saves']

        # Opponent adjustment (weak offense = good for goalie) – only when not using danger-weighted
        if not use_danger_weighted:
            opp_weakness = row.get('opp_offense_weakness', 1.0)
            if pd.notna(opp_weakness):
                ga_reduction = (opp_weakness - 1.0) * 0.1 + 1.0
                expected_pts *= ga_reduction
        # When using danger-weighted, opp mix is already in expected_saves/GA; optional small quality_adj
        else:
            opp_hd_share = row.get('opp_HD_share')
            if pd.notna(opp_hd_share):
                quality_adj = 1.0 + (LEAGUE_AVG_HD_SHARE - opp_hd_share) * 0.15  # lighter when danger-weighted
                quality_adj = np.clip(quality_adj, 1.0 - SHOT_QUALITY_ADJ_CAP, 1.0 + SHOT_QUALITY_ADJ_CAP)
                expected_pts *= quality_adj

        # Home ice advantage
        if row.get('is_home') == True:
            expected_pts *= 1.03

        # ==================== Backtest Bias Correction ====================
        # Goalies are over-projected by ~2.30 pts (6-backtest combined)
        expected_pts *= GOALIE_BIAS_CORRECTION

        # ==================== Goalie Quality Tier ====================
        # Penalizes BACKUP goalies by 20%, STARTER by 5%, ELITE unchanged
        tier_mult = row.get('tier_multiplier', 1.0)
        if pd.notna(tier_mult):
            expected_pts *= tier_mult

        # ==================== High-Projection Mean Regression ====================
        if expected_pts > GOALIE_HIGH_PROJ_THRESHOLD:
            expected_pts = (GOALIE_HIGH_PROJ_BLEND * expected_pts +
                           (1 - GOALIE_HIGH_PROJ_BLEND) * GOALIE_LEAGUE_MEAN_FPTS)

        # ==================== Goalie Projection Cap ====================
        # Goalies projected 12+ had +3.79 bias — cap prevents extreme projections
        expected_pts = min(expected_pts, GOALIE_PROJECTION_CAP)

        return expected_pts

    def project_skaters_baseline(self, skater_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate baseline projections for skaters using calculated expected value.
        This doesn't require training data - uses per-game stats directly.
        """
        df = skater_features.copy()

        # Calculate expected fantasy points
        df['projected_fpts'] = df.apply(self.calculate_expected_fantasy_points_skater, axis=1)

        # Apply Edge boost if available (from data pipeline)
        if 'edge_boost' in df.columns:
            df['projected_fpts_pre_edge'] = df['projected_fpts']
            df['projected_fpts'] = df['projected_fpts'] * df['edge_boost']

            # Report Edge boosts applied
            boosted = df[df['edge_boost'] > 1.0]
            if len(boosted) > 0:
                print(f"  Edge boosts applied to {len(boosted)} skaters:")
                for _, row in boosted.nlargest(5, 'edge_boost').iterrows():
                    boost_pct = (row['edge_boost'] - 1) * 100
                    reasons = row.get('edge_boost_reasons', '')
                    print(f"    {row['name']:25} +{boost_pct:.1f}% | {reasons}")
                if len(boosted) > 5:
                    print(f"    ... and {len(boosted) - 5} more")

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

        # Optional lines data (PP1/Line1 from DailyFaceoff) for role multiplier
        lines_data = data.get('lines_data')

        # Engineer features (now with advanced stats, injuries, and optional lines_data)
        skater_features = self.feature_engineer.engineer_skater_features(
            data['skaters'], data['teams'], data['schedule'], target_date,
            advanced_stats=advanced_stats, injuries=injuries, lines_data=lines_data
        )

        goalie_features = self.feature_engineer.engineer_goalie_features(
            data['goalies'], data['teams'], data['schedule'], target_date,
            team_danger_df=data.get('team_danger_stats'),
        )

        # Generate baseline projections
        skater_projections = self.project_skaters_baseline(skater_features)
        # OLD:


# NEW:
        from goalie_model import GoalieProjectionModel
        goalie_model = GoalieProjectionModel()
        goalie_projections = goalie_model.project_goalies(
            goalie_features, data['schedule'], target_date,
            team_totals=data.get('team_totals'),
            team_game_totals=data.get('team_game_totals'),
)

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
