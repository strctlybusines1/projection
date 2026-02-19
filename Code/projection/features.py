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
    INJURY_STATUSES_EXCLUDE, GOALIE_TIERS, INJURY_OPPORTUNITY,
    SIGNAL_WEIGHTS_NORMALIZED, SIGNAL_COMPOSITE_SENSITIVITY,
    SIGNAL_COMPOSITE_CLIP_LOW, SIGNAL_COMPOSITE_CLIP_HIGH,
    LEAGUE_AVG_SHARE_PCT, SKATER_SCORING,
    USE_EXPECTED_TOI_INJURY_BUMP, EXPECTED_TOI_BUMP_CAP,
)


class FeatureEngineer:
    """Engineer features for NHL DFS projections."""

    def __init__(self):
        pass

    def engineer_skater_features(self, skaters: pd.DataFrame, teams: pd.DataFrame,
                                   schedule: pd.DataFrame, target_date: Optional[str] = None,
                                   advanced_stats: Optional[Dict[str, pd.DataFrame]] = None,
                                   injuries: Optional[pd.DataFrame] = None,
                                   lines_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Engineer features for skater projections.

        Args:
            skaters: DataFrame with skater season stats
            teams: DataFrame with team stats
            schedule: DataFrame with upcoming games
            target_date: Date to project for (filters schedule)
            advanced_stats: Dict with 'team' and 'player' advanced stats from NST
            injuries: DataFrame with current injury data
            lines_data: Optional dict team -> {forward_lines, pp_units} for PP1/Line1 role tags

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

        # ==================== DK per 60 and Expected TOI (rate-based projection) ====================
        # expected_toi_minutes: from ev+pp+sh when available and positive, else toi_minutes (so rate-based can differ from per-game)
        if all(c in df.columns for c in ['ev_toi_per_game', 'pp_toi_per_game', 'sh_toi_per_game']):
            ev = df['ev_toi_per_game'].fillna(0)
            pp = df['pp_toi_per_game'].fillna(0)
            sh = df['sh_toi_per_game'].fillna(0)
            if (ev.median() > 100 or pp.median() > 100 or sh.median() > 100):
                raw_min = (ev + pp + sh) / 60.0  # situation TOI in minutes
            else:
                raw_min = ev + pp + sh
            # Per-row fallback: use toi_minutes when ev+pp+sh is 0 or tiny (no TOI breakdown for that player)
            if 'toi_minutes' in df.columns:
                df['expected_toi_minutes'] = np.where(raw_min > 0.5, raw_min, df['toi_minutes'].values)
            else:
                df['expected_toi_minutes'] = np.where(raw_min > 0.5, raw_min, np.nan)
        elif 'toi_minutes' in df.columns:
            df['expected_toi_minutes'] = df['toi_minutes'].copy()
        else:
            df['expected_toi_minutes'] = np.nan

        # dk_pts_per_60: base DK points per game (no bonuses) * 60 / toi_minutes
        base_dk_pg = (
            df['goals_pg'] * SKATER_SCORING['goals'] +
            df['assists_pg'] * SKATER_SCORING['assists'] +
            df['shots_pg'] * SKATER_SCORING['shots_on_goal'] +
            df['blocks_pg'] * SKATER_SCORING['blocked_shots'] +
            df.get('sh_points_pg', pd.Series(0, index=df.index)).fillna(0) * SKATER_SCORING['shorthanded_points_bonus']
        )
        toi_min = df['toi_minutes'] if 'toi_minutes' in df.columns else df.get('expected_toi_minutes')
        if toi_min is not None and (toi_min > 0).any():
            df['dk_pts_per_60'] = np.where(toi_min > 0, base_dk_pg * 60.0 / toi_min, np.nan)
        else:
            df['dk_pts_per_60'] = np.nan

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

            # Add signal-weighted composite matchup features
            df = self._add_signal_composite_features(df, team_adv, schedule, target_date)

        # ==================== Injury Context Features ====================
        if injuries is not None and not injuries.empty:
            df = self._add_injury_context_features(df, injuries)

        # Apply expected TOI bump (volume) when key teammates are out
        if USE_EXPECTED_TOI_INJURY_BUMP and 'expected_toi_bump' in df.columns and 'expected_toi_minutes' in df.columns:
            df['expected_toi_minutes'] = df['expected_toi_minutes'] * (1.0 + df['expected_toi_bump'])

        # Line role (PP1, Line1) from lines_data for expected TOI context and role multiplier
        if lines_data is not None and isinstance(lines_data, dict):
            df = self._add_line_role_features(df, lines_data)

        return df

    def engineer_goalie_features(self, goalies: pd.DataFrame, teams: pd.DataFrame,
                                   schedule: pd.DataFrame, target_date: Optional[str] = None,
                                   team_danger_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
            df = self._add_goalie_opponent_features(df, teams, schedule, target_date, team_danger_df)

        # ==================== Goalie Quality Tier ====================
        df = self.assign_goalie_tier(df)

        return df

    # ==================== Goalie Quality Tier ====================

    def assign_goalie_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign quality tier to goalies based on save_pct and games_started.

        Tiers (from config.GOALIE_TIERS):
            ELITE:   sv% >= 0.915 AND gs >= 20 → 1.0x multiplier
            STARTER: sv% >= 0.900 AND gs >= 15 → 0.95x multiplier
            BACKUP:  everyone else              → 0.80x multiplier

        Features added:
            - goalie_tier: 'ELITE', 'STARTER', or 'BACKUP'
            - tier_multiplier: projection multiplier for the tier
        """
        df['goalie_tier'] = 'BACKUP'
        df['tier_multiplier'] = GOALIE_TIERS['BACKUP']['projection_mult']

        save_pct_col = 'save_pct' if 'save_pct' in df.columns else None
        gs_col = 'games_started' if 'games_started' in df.columns else 'games_played'

        if save_pct_col is None or gs_col not in df.columns:
            return df

        # Assign tiers (STARTER first, then ELITE overwrites qualifying goalies)
        for tier_name in ['STARTER', 'ELITE']:
            tier = GOALIE_TIERS[tier_name]
            mask = (
                (df[save_pct_col] >= tier['min_save_pct']) &
                (df[gs_col] >= tier['min_games_started'])
            )
            df.loc[mask, 'goalie_tier'] = tier_name
            df.loc[mask, 'tier_multiplier'] = tier['projection_mult']

        return df

    # ==================== Advanced Analytics Features ====================

    def _add_xg_features(self, df: pd.DataFrame, team_adv_stats: Dict[str, pd.DataFrame],
                          schedule: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add expected goals (xG) features, scaled by player TOI.

        xGF/60 is a team-level per-60-minutes stat. Individual players don't play
        60 minutes, so the boost must be scaled by their actual TOI.

        Features added:
            - team_xgf_60: Team's xG for per 60 minutes
            - team_xga_60: Team's xG against per 60 minutes
            - opp_xga_60: Opponent's xG against per 60 (soft defense = good)
            - xg_matchup_boost: Combined xG matchup advantage (scaled by TOI)
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

        # Calculate raw matchup factor (before TOI scaling)
        # Good offense (high xGF) vs bad defense (high xGA) = advantage
        team_offense_factor = df['team_xgf_60'] / LEAGUE_AVG_XGF_60
        opp_defense_weakness = df['opp_xga_60'] / LEAGUE_AVG_XGA_60

        raw_matchup_advantage = (team_offense_factor * opp_defense_weakness) - 1.0  # Deviation from neutral

        # Scale by player's 5v5 TOI (not total TOI!)
        # xGF/60 is a 5v5 stat, so we only apply it to 5v5 ice time
        # PP/PK time has different dynamics
        if 'ev_toi_per_game' in df.columns:
            # ev_toi_per_game is 5v5 TOI in seconds
            ev_toi = df['ev_toi_per_game']
            if ev_toi.median() > 100:  # In seconds
                ev_toi_minutes = ev_toi / 60.0
            else:
                ev_toi_minutes = ev_toi
            toi_factor = ev_toi_minutes / 60.0  # Fraction of 60 min at 5v5
        elif 'toi_minutes' in df.columns:
            # Fallback: estimate 5v5 as ~80% of total TOI
            toi_factor = (df['toi_minutes'] * 0.80) / 60.0
        elif 'toi_per_game' in df.columns:
            # toi_per_game might be in seconds
            toi = df['toi_per_game']
            if toi.median() > 100:  # Likely in seconds
                total_toi_minutes = toi / 60.0
            else:
                total_toi_minutes = toi
            # Estimate 5v5 as ~80% of total TOI
            toi_factor = (total_toi_minutes * 0.80) / 60.0
        else:
            # Default assumption: average skater plays ~14 minutes at 5v5
            toi_factor = 14.0 / 60.0

        # Apply TOI-scaled boost
        # The boost is: 1 + (matchup_advantage * toi_factor * sensitivity)
        # sensitivity dampens the effect (0.4 = conservative estimate)
        sensitivity = 0.4
        df['xg_matchup_boost'] = (1.0 + raw_matchup_advantage * toi_factor * sensitivity).clip(0.97, 1.08)

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

    def _add_signal_composite_features(self, df: pd.DataFrame,
                                        team_adv_stats: Dict[str, pd.DataFrame],
                                        schedule: pd.DataFrame,
                                        target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add signal-weighted composite matchup feature based on backtest findings.

        The backtest (signal_noise_report.csv) found share/percentage metrics
        (SFpct, xGFpct, SCFpct, FFpct, CFpct) are the strongest predictors of
        DFS output with high persistence and predictive correlation.

        Creates a composite score: team offensive quality + opponent defensive weakness,
        weighted by each stat's backtest predictive correlation.

        Features added:
            - team_signal_composite: Team's weighted share deviation from 50%
            - opp_signal_composite: Opponent's defensive weakness (inverted share)
            - signal_matchup_boost: Combined matchup multiplier
        """
        team_5v5 = team_adv_stats.get('5v5', pd.DataFrame())

        # Initialize defaults
        df['team_signal_composite'] = 0.0
        df['opp_signal_composite'] = 0.0
        df['signal_matchup_boost'] = 1.0

        if team_5v5.empty or 'team' not in df.columns:
            return df

        if 'team' not in team_5v5.columns:
            return df

        team_5v5_idx = team_5v5.set_index('team')

        # Determine which signal columns are available
        available_signals = {
            stat: weight
            for stat, weight in SIGNAL_WEIGHTS_NORMALIZED.items()
            if stat in team_5v5_idx.columns
        }

        if not available_signals:
            return df

        # Re-normalize weights to available stats only
        total_avail = sum(available_signals.values())
        if total_avail == 0:
            return df
        weights = {k: v / total_avail for k, v in available_signals.items()}

        def calc_team_composite(team_code):
            """Weighted composite of team's offensive share quality."""
            if team_code not in team_5v5_idx.index:
                return 0.0
            row = team_5v5_idx.loc[team_code]
            composite = 0.0
            for stat, w in weights.items():
                val = row.get(stat, LEAGUE_AVG_SHARE_PCT)
                if pd.isna(val):
                    val = LEAGUE_AVG_SHARE_PCT
                composite += w * (val - LEAGUE_AVG_SHARE_PCT)
            return composite

        def calc_opp_weakness(opp_code):
            """Weighted composite of opponent's defensive weakness.
            If opponent has 47% SFpct, they give up 53% to opponents.
            Weakness = 50 - opp_value (positive = weaker defense)."""
            if pd.isna(opp_code) or opp_code not in team_5v5_idx.index:
                return 0.0
            row = team_5v5_idx.loc[opp_code]
            composite = 0.0
            for stat, w in weights.items():
                val = row.get(stat, LEAGUE_AVG_SHARE_PCT)
                if pd.isna(val):
                    val = LEAGUE_AVG_SHARE_PCT
                composite += w * (LEAGUE_AVG_SHARE_PCT - val)
            return composite

        df['team_signal_composite'] = df['team'].apply(calc_team_composite)

        if 'opponent' in df.columns:
            df['opp_signal_composite'] = df['opponent'].apply(calc_opp_weakness)

        # Net matchup: average of team quality + opponent weakness
        net_deviation = (df['team_signal_composite'] + df['opp_signal_composite']) / 2.0

        # Convert to multiplier (sensitivity=0.30 means +5 pct-pt deviation -> +1.5% boost)
        df['signal_matchup_boost'] = (
            1.0 + net_deviation * SIGNAL_COMPOSITE_SENSITIVITY / 100.0
        ).clip(SIGNAL_COMPOSITE_CLIP_LOW, SIGNAL_COMPOSITE_CLIP_HIGH)

        return df

    def _name_matches_line(self, full_name: str, line_names: List[str]) -> bool:
        """Return True if full_name matches any name in line_names (last-name or contains)."""
        if not full_name or not line_names:
            return False
        last = full_name.strip().split()[-1].lower() if full_name.strip().split() else ""
        if not last:
            return False
        for p in line_names:
            if not p:
                continue
            p_last = p.strip().split()[-1].lower() if p.strip().split() else ""
            if last == p_last or (len(last) > 2 and (last in p.lower() or p.lower().endswith(last))):
                return True
        return False

    def _add_line_role_features(self, df: pd.DataFrame, lines_data: Dict) -> pd.DataFrame:
        """
        Add line role features from lines_data (PP1, Line1) for expected TOI context and role multiplier.
        lines_data: team -> { pp_units: [{ unit: 1, players: [...] }], forward_lines: [{ line: 1, players: [...] }] }.
        Features: is_pp1, is_line1, role_multiplier (1.05 PP1, 1.02 Line1, else 1.0).
        """
        df = df.copy()
        df["is_pp1"] = 0
        df["is_line1"] = 0
        df["role_multiplier"] = 1.0
        if "team" not in df.columns or "name" not in df.columns:
            return df
        for team, data in lines_data.items():
            if not data or not isinstance(data, dict) or "error" in data:
                continue
            pp1_names = []
            for pp in data.get("pp_units", []):
                if pp.get("unit") == 1:
                    pp1_names.extend(pp.get("players", []))
            line1_names = []
            for line in data.get("forward_lines", []):
                if line.get("line") == 1:
                    line1_names.extend(line.get("players", []))
            mask = df["team"] == team
            for idx in df.index[mask]:
                name = df.at[idx, "name"]
                if isinstance(name, str):
                    if self._name_matches_line(name, pp1_names):
                        df.at[idx, "is_pp1"] = 1
                    if self._name_matches_line(name, line1_names):
                        df.at[idx, "is_line1"] = 1
            # role_multiplier: PP1 > Line1 > else
            df.loc[mask, "role_multiplier"] = np.where(
                df.loc[mask, "is_pp1"] == 1, 1.05, np.where(df.loc[mask, "is_line1"] == 1, 1.02, 1.0)
            )
        return df

    def _add_injury_context_features(self, df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
        """
        Add injury context features (opportunity boost from teammate injuries).

        Quality-weighted: key player injuries (top-6 F / top-4 D, ppg >= 0.5)
        give a larger boost than regular player injuries.

        Based on Jan 26 backtest: ANA losing Carlsson + Terry + McTavish (key players)
        gave remaining ANA players (Granlund 43.3 FPTS) a large opportunity boost.

        Features added:
            - team_injury_count: Number of injured players on team
            - team_key_injury_count: Number of key injured players on team
            - opportunity_boost: Multiplier based on teammate injuries (quality-weighted)
        """
        df['team_injury_count'] = 0
        df['team_key_injury_count'] = 0
        df['opportunity_boost'] = 1.0

        if injuries.empty or 'team' not in df.columns:
            return df

        if 'team' not in injuries.columns or 'injury_status' not in injuries.columns:
            return df

        # Count significant injuries per team (exclude DTD)
        severe_injuries = injuries[injuries['injury_status'].isin(INJURY_STATUSES_EXCLUDE)]

        if severe_injuries.empty:
            return df

        # Classify injured players as "key" or "regular"
        # Key player: ppg >= threshold (top-6 F / top-4 D caliber)
        ppg_threshold = INJURY_OPPORTUNITY['key_player_threshold_ppg']

        # Calculate ppg for injured players if stats available
        if 'points_per_game' in severe_injuries.columns:
            ppg_col = 'points_per_game'
        elif 'points' in severe_injuries.columns and 'games_played' in severe_injuries.columns:
            severe_injuries = severe_injuries.copy()
            gp = severe_injuries['games_played'].replace(0, np.nan)
            severe_injuries['_ppg'] = (severe_injuries['points'] / gp).fillna(0)
            ppg_col = '_ppg'
        else:
            ppg_col = None

        if ppg_col is not None:
            key_injuries = severe_injuries[severe_injuries[ppg_col] >= ppg_threshold]
            regular_injuries = severe_injuries[severe_injuries[ppg_col] < ppg_threshold]
        else:
            # Without stats, treat all as regular
            key_injuries = pd.DataFrame()
            regular_injuries = severe_injuries

        # Count by team
        injury_counts = severe_injuries.groupby('team').size()
        key_counts = key_injuries.groupby('team').size() if not key_injuries.empty else pd.Series(dtype=int)
        regular_counts = regular_injuries.groupby('team').size() if not regular_injuries.empty else pd.Series(dtype=int)

        # Map counts
        df['team_injury_count'] = df['team'].map(lambda t: injury_counts.get(t, 0))
        df['team_key_injury_count'] = df['team'].map(lambda t: key_counts.get(t, 0))

        # Calculate quality-weighted opportunity boost
        key_boost_rate = INJURY_OPPORTUNITY['key_player_boost']
        regular_boost_rate = INJURY_OPPORTUNITY['regular_player_boost']
        max_boost = INJURY_OPPORTUNITY['max_boost']

        key_boost = df['team'].map(lambda t: key_counts.get(t, 0)) * key_boost_rate
        reg_boost = df['team'].map(lambda t: regular_counts.get(t, 0)) * regular_boost_rate
        total_boost = (key_boost + reg_boost).clip(0, max_boost)

        df['opportunity_boost'] = (1.0 + total_boost).clip(1.0, 1.0 + max_boost)

        # Expected TOI bump (volume): when key/regular players are out, remaining players get more TOI
        if USE_EXPECTED_TOI_INJURY_BUMP:
            key_per_team = df['team'].map(lambda t: key_counts.get(t, 0))
            reg_per_team = df['team'].map(lambda t: regular_counts.get(t, 0))
            toi_bump = (key_per_team * 0.04 + reg_per_team * 0.01).clip(0, EXPECTED_TOI_BUMP_CAP)
            df['expected_toi_bump'] = toi_bump
        else:
            df['expected_toi_bump'] = 0.0

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

        # Home/away adjustment (NHL home teams score ~53% of goals on average)
        # Home: 1.06x multiplier (slightly above average)
        # Away: 0.94x multiplier (slightly below average)
        df['home_away_adjustment'] = np.where(
            df['is_home'] == True, 1.06, np.where(df['is_home'] == False, 0.94, 1.0)
        )

        return df

    def _add_goalie_opponent_features(self, df: pd.DataFrame, teams: pd.DataFrame,
                                        schedule: pd.DataFrame, target_date: Optional[str] = None,
                                        team_danger_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add opponent-based features for goalies (incl. opponent shot quality HD/MD/LD when available)."""
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

        # Opponent shot quality (HD/MD/LD from test-folder NST CSVs)
        if team_danger_df is not None and not team_danger_df.empty:
            danger_cols = ['hdsf_60', 'mdsf_60', 'ldsf_60', 'hd_share']
            if all(c in team_danger_df.columns for c in danger_cols):
                df['opp_HDSF_60'] = df['opponent'].map(
                    lambda opp: team_danger_df.loc[opp, 'hdsf_60'] if opp in team_danger_df.index else np.nan
                )
                df['opp_MDSF_60'] = df['opponent'].map(
                    lambda opp: team_danger_df.loc[opp, 'mdsf_60'] if opp in team_danger_df.index else np.nan
                )
                df['opp_LDSF_60'] = df['opponent'].map(
                    lambda opp: team_danger_df.loc[opp, 'ldsf_60'] if opp in team_danger_df.index else np.nan
                )
                df['opp_HD_share'] = df['opponent'].map(
                    lambda opp: team_danger_df.loc[opp, 'hd_share'] if opp in team_danger_df.index else np.nan
                )
                df['opp_MD_share'] = df['opponent'].map(
                    lambda opp: team_danger_df.loc[opp, 'md_share'] if opp in team_danger_df.index else np.nan
                )
                df['opp_LD_share'] = df['opponent'].map(
                    lambda opp: team_danger_df.loc[opp, 'ld_share'] if opp in team_danger_df.index else np.nan
                )
                # Display-only tier: High / Medium / Low by opp_HD_share percentiles on this slate
                valid = df['opp_HD_share'].notna()
                if valid.any():
                    p33 = df.loc[valid, 'opp_HD_share'].quantile(0.33)
                    p67 = df.loc[valid, 'opp_HD_share'].quantile(0.67)
                    def tier(h):
                        if pd.isna(h):
                            return None
                        if h <= p33:
                            return 'Low'
                        if h <= p67:
                            return 'Medium'
                        return 'High'
                    df['opp_shot_quality_tier'] = df['opp_HD_share'].apply(tier)
                else:
                    df['opp_shot_quality_tier'] = None

            # Goalie's team save % by shot type (HDSV%, MDSV%, LDSV%) – affects projection vs opponent mix
            sv_cols = ['hdsv_pct', 'mdsv_pct', 'ldsv_pct']
            if all(c in team_danger_df.columns for c in sv_cols):
                df['team_HDSV_pct'] = df['team'].map(
                    lambda t: team_danger_df.loc[t, 'hdsv_pct'] if t in team_danger_df.index else np.nan
                )
                df['team_MDSV_pct'] = df['team'].map(
                    lambda t: team_danger_df.loc[t, 'mdsv_pct'] if t in team_danger_df.index else np.nan
                )
                df['team_LDSV_pct'] = df['team'].map(
                    lambda t: team_danger_df.loc[t, 'ldsv_pct'] if t in team_danger_df.index else np.nan
                )

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
                'opp_softness', 'is_home', 'home_away_adjustment', 'consistency'
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
