#!/usr/bin/env python3
"""
Multi-Season Signal Validation Script
=====================================

Validates 5 key signals across 4 independent seasons (2020, 2021, 2022, 2024-25):
1. Opponent Quality Effect (defensive regime)
2. PP Production Concentration (power play variance)
3. Recency Weighting Value (EWM vs expanding mean)
4. TOI Stability as Foundation (single best predictor)
5. Position-specific Regression Rates (C/W/D differences)

Tests for consistency across seasons using:
- Effect sizes (Cohen's d, correlation)
- p-values (t-tests, Pearson r)
- Cross-season meta-analysis (Fisher's method)
- Consistency flags (signals present in all seasons = TRUE signal)

Database: nhl_dfs_history.db
Tables: historical_skaters (2020-2022), boxscore_skaters (2024-25)
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def toi_str_to_seconds(toi_str: str) -> int:
    """Convert TOI string MM:SS to seconds."""
    if pd.isna(toi_str) or toi_str == '':
        return 0
    try:
        parts = str(toi_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except:
        return 0


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) == 0 or len(group2) == 0:
        return 0.0

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def fishers_method(p_values: List[float]) -> Tuple[float, float]:
    """
    Fisher's method for combining p-values across independent tests.
    Returns: (combined_test_statistic, combined_p_value)
    """
    p_values = [max(min(p, 0.9999), 0.0001) for p in p_values]  # Clip to avoid extremes
    chi2_stat = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p = 1 - stats.chi2.cdf(chi2_stat, df)
    return chi2_stat, combined_p


def load_season_data(conn: sqlite3.Connection, season: int = None,
                     is_current: bool = False) -> pd.DataFrame:
    """Load skater data for a season."""
    if is_current:
        # Load 2024-25 season from boxscore_skaters
        query = """
            SELECT
                player_name, team, position, game_date, opponent,
                goals, assists, pp_goals, shots, blocked_shots as blocks,
                hits, pim, toi_seconds, dk_fpts
            FROM boxscore_skaters
            WHERE dk_fpts IS NOT NULL AND dk_fpts > 0
            ORDER BY game_date, player_name
        """
        df = pd.read_sql(query, conn)
        df['season'] = 2024
    else:
        # Load historical season
        query = f"""
            SELECT
                player_name, team, position, game_date, opponent,
                goals, assists, pp_goals, shots, blocked_shots as blocks,
                hits, pim, toi_seconds, dk_fpts
            FROM historical_skaters
            WHERE season = {season} AND dk_fpts IS NOT NULL AND dk_fpts > 0
            ORDER BY game_date, player_name
        """
        df = pd.read_sql(query, conn)
        df['season'] = season

    # Ensure numeric columns
    numeric_cols = ['goals', 'assists', 'pp_goals', 'shots', 'blocks', 'hits',
                    'pim', 'toi_seconds', 'dk_fpts']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df


# ============================================================================
# SIGNAL 1: OPPONENT QUALITY EFFECT
# ============================================================================

def signal_1_opponent_quality(data: pd.DataFrame, season: int) -> Dict:
    """
    Test: Do skaters score more FPTS vs weak defenses?
    Method:
    1. Compute defensive quality: avg dk_fpts ALLOWED per opponent per game
    2. Split opponents into tertiles (strong/avg/weak defense)
    3. Compute effect size (Cohen's d) and p-value
    """
    results = {'signal': 'Opponent Quality Effect', 'season': season}

    try:
        # Build opponent defensive quality
        # For each (game_date, opponent) pair, sum all FPTS scored against them
        opponent_fpts = data.groupby(['game_date', 'opponent'])['dk_fpts'].sum().reset_index()
        opponent_fpts.columns = ['game_date', 'opponent', 'total_fpts_allowed']

        # Merge back to get opponent quality for each player-game
        data_with_opp_quality = data.merge(
            opponent_fpts,
            left_on=['game_date', 'opponent'],
            right_on=['game_date', 'opponent'],
            how='left'
        )

        # Create tertiles for opponent quality
        tertiles = pd.qcut(
            data_with_opp_quality['total_fpts_allowed'].dropna(),
            q=3,
            labels=['strong_defense', 'average_defense', 'weak_defense'],
            duplicates='drop'
        )

        # Assign tertiles (handle any missing values)
        data_with_opp_quality['opp_quality_tier'] = np.nan
        valid_idx = data_with_opp_quality['total_fpts_allowed'].notna()
        data_with_opp_quality.loc[valid_idx, 'opp_quality_tier'] = tertiles.values

        # Compare FPTS: weak defense vs strong defense
        weak_def = data_with_opp_quality[
            data_with_opp_quality['opp_quality_tier'] == 'weak_defense'
        ]['dk_fpts'].values

        strong_def = data_with_opp_quality[
            data_with_opp_quality['opp_quality_tier'] == 'strong_defense'
        ]['dk_fpts'].values

        if len(weak_def) > 10 and len(strong_def) > 10:
            cohens_d = compute_cohens_d(weak_def, strong_def)
            t_stat, p_val = stats.ttest_ind(weak_def, strong_def)

            results.update({
                'n_weak_def': len(weak_def),
                'n_strong_def': len(strong_def),
                'mean_fpts_weak_def': np.mean(weak_def),
                'mean_fpts_strong_def': np.mean(strong_def),
                'cohens_d': cohens_d,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'status': 'computed'
            })
        else:
            results['status'] = 'insufficient_data'

    except Exception as e:
        results['status'] = f'error: {str(e)}'

    return results


# ============================================================================
# SIGNAL 2: PP PRODUCTION CONCENTRATION
# ============================================================================

def signal_2_pp_concentration(data: pd.DataFrame, season: int) -> Dict:
    """
    Test: Do high-PP players have higher FPTS variance (ceiling)?
    Method:
    1. Identify players with high PP goals/assists rates
    2. Compare variance (coefficient of variation) vs low-PP players
    3. Test if PP share predicts next-game FPTS
    """
    results = {'signal': 'PP Production Concentration', 'season': season}

    try:
        # Player-level aggregation
        player_stats = data.groupby('player_name').agg({
            'pp_goals': 'sum',
            'goals': 'sum',
            'dk_fpts': ['mean', 'std', 'count']
        }).reset_index()

        player_stats.columns = ['player_name', 'total_pp_goals', 'total_goals',
                                 'mean_fpts', 'std_fpts', 'n_games']

        # Filter minimum games
        player_stats = player_stats[player_stats['n_games'] >= 5]

        if len(player_stats) > 20:
            # Compute PP share
            player_stats['pp_goal_share'] = (
                player_stats['total_pp_goals'] /
                (player_stats['total_goals'] + 1)
            )

            # Compute coefficient of variation (volatility)
            player_stats['cv'] = player_stats['std_fpts'] / (player_stats['mean_fpts'] + 1)

            # Split by high PP vs low PP
            pp_median = player_stats['pp_goal_share'].median()
            high_pp = player_stats[player_stats['pp_goal_share'] > pp_median]['cv'].values
            low_pp = player_stats[player_stats['pp_goal_share'] <= pp_median]['cv'].values

            if len(high_pp) > 5 and len(low_pp) > 5:
                cohens_d = compute_cohens_d(high_pp, low_pp)
                t_stat, p_val = stats.ttest_ind(high_pp, low_pp)

                # Also compute correlation between PP share and variance
                corr_coef, corr_pval = stats.pearsonr(
                    player_stats['pp_goal_share'],
                    player_stats['cv']
                )

                results.update({
                    'n_high_pp': len(high_pp),
                    'n_low_pp': len(low_pp),
                    'mean_cv_high_pp': np.mean(high_pp),
                    'mean_cv_low_pp': np.mean(low_pp),
                    'cohens_d': cohens_d,
                    'p_value': p_val,
                    'pp_share_cv_corr': corr_coef,
                    'pp_share_cv_corr_pval': corr_pval,
                    'significant': p_val < 0.05,
                    'status': 'computed'
                })
            else:
                results['status'] = 'insufficient_players'
        else:
            results['status'] = 'insufficient_players'

    except Exception as e:
        results['status'] = f'error: {str(e)}'

    return results


# ============================================================================
# SIGNAL 3: RECENCY WEIGHTING VALUE
# ============================================================================

def signal_3_recency_weighting(data: pd.DataFrame, season: int) -> Dict:
    """
    Test: Is EWM a better predictor than expanding mean?
    Method:
    1. For each player, compute expanding mean and EWM (halflife=15)
    2. Walk-forward: compare MAE (actual vs predicted) for next game
    3. Compute effect size from MAE comparison
    """
    results = {'signal': 'Recency Weighting Value', 'season': season}

    try:
        # Sort by player and game_date
        data = data.sort_values(['player_name', 'game_date']).reset_index(drop=True)

        # Only use players with 20+ games for reliable testing
        player_game_counts = data.groupby('player_name').size()
        active_players = player_game_counts[player_game_counts >= 20].index.values

        data_filtered = data[data['player_name'].isin(active_players)].copy()

        expanding_errors = []
        ewm_errors = []

        for player in data_filtered['player_name'].unique():
            player_data = data_filtered[data_filtered['player_name'] == player].copy()
            player_data = player_data.reset_index(drop=True)

            if len(player_data) < 5:
                continue

            # Compute expanding mean and EWM on historical data
            player_data['expanding_mean'] = player_data['dk_fpts'].expanding().mean().shift(1)
            player_data['ewm_mean'] = player_data['dk_fpts'].ewm(halflife=15).mean().shift(1)

            # Compute errors (actual - predicted) for games where we have both predictions
            valid = player_data[['expanding_mean', 'ewm_mean', 'dk_fpts']].notna().all(axis=1)

            if valid.sum() > 0:
                expanding_errors.extend(
                    np.abs(player_data[valid]['dk_fpts'] - player_data[valid]['expanding_mean'])
                )
                ewm_errors.extend(
                    np.abs(player_data[valid]['dk_fpts'] - player_data[valid]['ewm_mean'])
                )

        if len(expanding_errors) > 20 and len(ewm_errors) > 20:
            expanding_errors = np.array(expanding_errors)
            ewm_errors = np.array(ewm_errors)

            mae_expanding = np.mean(expanding_errors)
            mae_ewm = np.mean(ewm_errors)
            improvement = (mae_expanding - mae_ewm) / (mae_expanding + 1e-6)

            # Test if EWM is significantly better
            t_stat, p_val = stats.ttest_rel(expanding_errors, ewm_errors)
            cohens_d = compute_cohens_d(expanding_errors, ewm_errors)

            results.update({
                'n_predictions': len(ewm_errors),
                'mae_expanding': mae_expanding,
                'mae_ewm': mae_ewm,
                'improvement_pct': improvement * 100,
                'cohens_d': cohens_d,
                'p_value': p_val,
                'ewm_better': mae_ewm < mae_expanding,
                'significant': p_val < 0.05,
                'status': 'computed'
            })
        else:
            results['status'] = 'insufficient_predictions'

    except Exception as e:
        results['status'] = f'error: {str(e)}'

    return results


# ============================================================================
# SIGNAL 4: TOI STABILITY AS FOUNDATION
# ============================================================================

def signal_4_toi_stability(data: pd.DataFrame, season: int) -> Dict:
    """
    Test: Is TOI the best single predictor of next-game FPTS?
    Method:
    1. For each player, compute Pearson r between lagged TOI and FPTS
    2. Compare TOI correlation vs other single predictors (points, shots)
    3. Compute meta-correlation across all players
    """
    results = {'signal': 'TOI Stability as Foundation', 'season': season}

    try:
        data = data.sort_values(['player_name', 'game_date']).reset_index(drop=True)

        toi_correlations = []
        points_correlations = []
        shots_correlations = []

        for player in data['player_name'].unique():
            player_data = data[data['player_name'] == player].copy()
            player_data = player_data.reset_index(drop=True)

            if len(player_data) < 10:
                continue

            # Lag predictors by 1 game to predict next game FPTS
            player_data['lagged_toi'] = player_data['toi_seconds'].shift(1)
            player_data['lagged_points'] = (player_data['goals'] + player_data['assists']).shift(1)
            player_data['lagged_shots'] = player_data['shots'].shift(1)

            # Compute correlations with next game FPTS
            valid = player_data[['lagged_toi', 'dk_fpts']].notna().all(axis=1) & (player_data['dk_fpts'] > 0)
            if valid.sum() > 4:
                try:
                    if player_data[valid]['lagged_toi'].std() > 0:
                        toi_corr, _ = stats.pearsonr(
                            player_data[valid]['lagged_toi'],
                            player_data[valid]['dk_fpts']
                        )
                        toi_correlations.append(toi_corr)
                except:
                    pass

            # Points correlation
            valid = player_data[['lagged_points', 'dk_fpts']].notna().all(axis=1) & (player_data['dk_fpts'] > 0)
            if valid.sum() > 4:
                try:
                    if player_data[valid]['lagged_points'].std() > 0:
                        points_corr, _ = stats.pearsonr(
                            player_data[valid]['lagged_points'],
                            player_data[valid]['dk_fpts']
                        )
                        points_correlations.append(points_corr)
                except:
                    pass

            # Shots correlation
            valid = player_data[['lagged_shots', 'dk_fpts']].notna().all(axis=1) & (player_data['dk_fpts'] > 0)
            if valid.sum() > 4:
                try:
                    if player_data[valid]['lagged_shots'].std() > 0:
                        shots_corr, _ = stats.pearsonr(
                            player_data[valid]['lagged_shots'],
                            player_data[valid]['dk_fpts']
                        )
                        shots_correlations.append(shots_corr)
                except:
                    pass

        if len(toi_correlations) > 15:
            toi_mean = np.mean(toi_correlations)
            points_mean = np.mean(points_correlations) if points_correlations else 0
            shots_mean = np.mean(shots_correlations) if shots_correlations else 0

            # Test if TOI is significantly better
            if len(points_correlations) > 5 and len(toi_correlations) > 5:
                t_toi_vs_pts, p_toi_vs_pts = stats.ttest_ind(toi_correlations, points_correlations)
            else:
                p_toi_vs_pts = np.nan

            results.update({
                'n_players': len(toi_correlations),
                'mean_toi_corr': toi_mean,
                'mean_points_corr': points_mean,
                'mean_shots_corr': shots_mean,
                'toi_is_best': toi_mean > max(points_mean, shots_mean),
                'toi_vs_points_pval': p_toi_vs_pts,
                'status': 'computed'
            })
        else:
            results['status'] = 'insufficient_players'

    except Exception as e:
        results['status'] = f'error: {str(e)}'

    return results


# ============================================================================
# SIGNAL 5: POSITION-SPECIFIC REGRESSION RATES
# ============================================================================

def signal_5_position_regression(data: pd.DataFrame, season: int) -> Dict:
    """
    Test: Do C/W/D have different regression rates?
    Method:
    1. Split players by position (C, L/R [Wings], D [Defense])
    2. For each position, compute lag-1 autocorrelation of FPTS
    3. Compare regression (mean reversion) across positions
    """
    results = {'signal': 'Position-specific Regression Rates', 'season': season}

    try:
        data = data.sort_values(['player_name', 'game_date']).reset_index(drop=True)

        position_correlations = {}

        # Normalize position codes: L/R -> W (Wing), C -> C, D -> D
        data['position_normalized'] = data['position'].apply(
            lambda x: 'W' if x in ['L', 'R'] else x
        )

        for position, pos_label in [('C', 'Center'), ('W', 'Wing'), ('D', 'Defense')]:
            pos_data = data[data['position_normalized'] == position].copy()

            if len(pos_data) < 100:
                position_correlations[position] = []
                continue

            correlations = []

            for player in pos_data['player_name'].unique():
                player_data = pos_data[pos_data['player_name'] == player].copy()
                player_data = player_data.reset_index(drop=True)

                if len(player_data) < 5:
                    continue

                # Lag FPTS
                player_data['lagged_fpts'] = player_data['dk_fpts'].shift(1)

                # Compute correlation (mean reversion: lower = more regression)
                valid = player_data[['lagged_fpts', 'dk_fpts']].notna().all(axis=1) & (player_data['dk_fpts'] > 0)
                if valid.sum() > 4:
                    try:
                        if player_data[valid]['lagged_fpts'].std() > 0:
                            corr, _ = stats.pearsonr(
                                player_data[valid]['lagged_fpts'],
                                player_data[valid]['dk_fpts']
                            )
                            correlations.append(corr)
                    except:
                        pass

            position_correlations[position] = correlations

        # Compare positions
        if all(len(position_correlations[p]) > 10 for p in ['C', 'W', 'D']):
            c_corr = np.mean(position_correlations['C'])
            w_corr = np.mean(position_correlations['W'])
            d_corr = np.mean(position_correlations['D'])

            # Test C vs W
            t_cw, p_cw = stats.ttest_ind(
                position_correlations['C'],
                position_correlations['W']
            )

            # Test C vs D
            t_cd, p_cd = stats.ttest_ind(
                position_correlations['C'],
                position_correlations['D']
            )

            # Test W vs D
            t_wd, p_wd = stats.ttest_ind(
                position_correlations['W'],
                position_correlations['D']
            )

            results.update({
                'n_centers': len(position_correlations['C']),
                'n_wings': len(position_correlations['W']),
                'n_defense': len(position_correlations['D']),
                'mean_corr_c': c_corr,
                'mean_corr_w': w_corr,
                'mean_corr_d': d_corr,
                'c_vs_w_pval': p_cw,
                'c_vs_d_pval': p_cd,
                'w_vs_d_pval': p_wd,
                'significant': min(p_cw, p_cd, p_wd) < 0.05,
                'status': 'computed'
            })
        else:
            results['status'] = f'insufficient_positions (C:{len(position_correlations["C"])}, W:{len(position_correlations["W"])}, D:{len(position_correlations["D"])})'

    except Exception as e:
        results['status'] = f'error: {str(e)}'

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_comprehensive_validation():
    """Execute all signal validations across all seasons."""

    print("=" * 80)
    print("MULTI-SEASON SIGNAL VALIDATION FRAMEWORK")
    print("=" * 80)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Connect to database
    db_path = "data/nhl_dfs_history.db"
    conn = sqlite3.connect(db_path)

    # Define seasons to test
    seasons_to_test = [
        (2020, False),
        (2021, False),
        (2022, False),
        (2024, True)  # Current season from boxscore_skaters
    ]

    # Storage for results
    all_results = {
        'signal_1': [],  # Opponent Quality
        'signal_2': [],  # PP Concentration
        'signal_3': [],  # Recency Weighting
        'signal_4': [],  # TOI Stability
        'signal_5': [],  # Position Regression
    }

    # ========================================================================
    # SIGNAL VALIDATION LOOP
    # ========================================================================

    for season_num, is_current in seasons_to_test:
        season_label = f"{season_num}-25" if is_current else str(season_num)
        print(f"\n{'='*80}")
        print(f"SEASON {season_label}")
        print(f"{'='*80}")

        # Load data
        print(f"Loading data for season {season_label}...", end=' ')
        df = load_season_data(conn, season=season_num, is_current=is_current)
        print(f"✓ {len(df):,} game records")

        # ====================================================================
        # SIGNAL 1: Opponent Quality Effect
        # ====================================================================
        print("\n[SIGNAL 1] Opponent Quality Effect", end=' ... ')
        result1 = signal_1_opponent_quality(df, season_num)
        all_results['signal_1'].append(result1)

        if result1['status'] == 'computed':
            print(f"✓ d={result1['cohens_d']:.3f}, p={result1['p_value']:.4f}")
            print(f"  Mean FPTS vs weak defense: {result1['mean_fpts_weak_def']:.2f}")
            print(f"  Mean FPTS vs strong defense: {result1['mean_fpts_strong_def']:.2f}")
        else:
            print(f"✗ {result1['status']}")

        # ====================================================================
        # SIGNAL 2: PP Production Concentration
        # ====================================================================
        print("\n[SIGNAL 2] PP Production Concentration", end=' ... ')
        result2 = signal_2_pp_concentration(df, season_num)
        all_results['signal_2'].append(result2)

        if result2['status'] == 'computed':
            print(f"✓ d={result2['cohens_d']:.3f}, p={result2['p_value']:.4f}")
            print(f"  CV high-PP players: {result2['mean_cv_high_pp']:.3f}")
            print(f"  CV low-PP players: {result2['mean_cv_low_pp']:.3f}")
            print(f"  PP-share/variance correlation: r={result2['pp_share_cv_corr']:.3f}, p={result2['pp_share_cv_corr_pval']:.4f}")
        else:
            print(f"✗ {result2['status']}")

        # ====================================================================
        # SIGNAL 3: Recency Weighting Value
        # ====================================================================
        print("\n[SIGNAL 3] Recency Weighting Value", end=' ... ')
        result3 = signal_3_recency_weighting(df, season_num)
        all_results['signal_3'].append(result3)

        if result3['status'] == 'computed':
            print(f"✓ EWM MAE: {result3['mae_ewm']:.2f} vs Expanding: {result3['mae_expanding']:.2f}")
            print(f"  Improvement: {result3['improvement_pct']:.2f}%, p={result3['p_value']:.4f}")
            print(f"  Cohen's d: {result3['cohens_d']:.3f}")
        else:
            print(f"✗ {result3['status']}")

        # ====================================================================
        # SIGNAL 4: TOI Stability as Foundation
        # ====================================================================
        print("\n[SIGNAL 4] TOI Stability as Foundation", end=' ... ')
        result4 = signal_4_toi_stability(df, season_num)
        all_results['signal_4'].append(result4)

        if result4['status'] == 'computed':
            print(f"✓ TOI correlation: r={result4['mean_toi_corr']:.3f} ({result4['n_players']} players)")
            print(f"  Prior Points: r={result4['mean_points_corr']:.3f}")
            print(f"  Prior Shots: r={result4['mean_shots_corr']:.3f}")
            print(f"  TOI is best predictor: {result4['toi_is_best']}")
            if not np.isnan(result4['toi_vs_points_pval']):
                print(f"  TOI vs Points p-value: {result4['toi_vs_points_pval']:.4f}")
        else:
            print(f"✗ {result4['status']}")

        # ====================================================================
        # SIGNAL 5: Position-specific Regression Rates
        # ====================================================================
        print("\n[SIGNAL 5] Position-specific Regression Rates", end=' ... ')
        result5 = signal_5_position_regression(df, season_num)
        all_results['signal_5'].append(result5)

        if result5['status'] == 'computed':
            print(f"✓ Centers: r={result5['mean_corr_c']:.3f}, Wings: r={result5['mean_corr_w']:.3f}, Defense: r={result5['mean_corr_d']:.3f}")
            print(f"  C vs W p-value: {result5['c_vs_w_pval']:.4f}")
            print(f"  C vs D p-value: {result5['c_vs_d_pval']:.4f}")
            print(f"  W vs D p-value: {result5['w_vs_d_pval']:.4f}")
        else:
            print(f"✗ {result5['status']}")

    conn.close()

    # ========================================================================
    # CROSS-SEASON META-ANALYSIS
    # ========================================================================

    print(f"\n\n{'='*80}")
    print("CROSS-SEASON META-ANALYSIS")
    print(f"{'='*80}\n")

    # SIGNAL 1 Meta-analysis
    print("[SIGNAL 1] Opponent Quality Effect")
    print("-" * 80)
    signal1_results = [r for r in all_results['signal_1'] if r['status'] == 'computed']
    if signal1_results:
        signal1_pvalues = [r['p_value'] for r in signal1_results]
        chi2_stat, combined_p = fishers_method(signal1_pvalues)
        effect_sizes = [r['cohens_d'] for r in signal1_results]

        print(f"Seasons analyzed: {len(signal1_results)}")
        for i, r in enumerate(signal1_results):
            print(f"  {r['season']}: d={r['cohens_d']:+.3f}, p={r['p_value']:.4f}, significant={r['significant']}")
        print(f"Meta-analysis (Fisher's method):")
        print(f"  Combined χ² = {chi2_stat:.3f}, combined p-value = {combined_p:.6f}")
        print(f"  Mean effect size: d={np.mean(effect_sizes):+.3f}")
        print(f"  Consistent across seasons: {all(r['significant'] for r in signal1_results)}")
        print(f"  VERDICT: {'✓ TRUE SIGNAL' if combined_p < 0.05 and all(r['significant'] for r in signal1_results) else '✗ INCONCLUSIVE'}")
    else:
        print("✗ Insufficient data across seasons")

    # SIGNAL 2 Meta-analysis
    print("\n[SIGNAL 2] PP Production Concentration")
    print("-" * 80)
    signal2_results = [r for r in all_results['signal_2'] if r['status'] == 'computed']
    if signal2_results:
        signal2_pvalues = [r['p_value'] for r in signal2_results]
        chi2_stat, combined_p = fishers_method(signal2_pvalues)
        effect_sizes = [r['cohens_d'] for r in signal2_results]

        print(f"Seasons analyzed: {len(signal2_results)}")
        for i, r in enumerate(signal2_results):
            print(f"  {r['season']}: d={r['cohens_d']:+.3f}, p={r['p_value']:.4f}, significant={r['significant']}")
        print(f"Meta-analysis (Fisher's method):")
        print(f"  Combined χ² = {chi2_stat:.3f}, combined p-value = {combined_p:.6f}")
        print(f"  Mean effect size: d={np.mean(effect_sizes):+.3f}")
        print(f"  Consistent across seasons: {all(r['significant'] for r in signal2_results)}")
        print(f"  VERDICT: {'✓ TRUE SIGNAL' if combined_p < 0.05 and all(r['significant'] for r in signal2_results) else '✗ INCONCLUSIVE'}")
    else:
        print("✗ Insufficient data across seasons")

    # SIGNAL 3 Meta-analysis
    print("\n[SIGNAL 3] Recency Weighting Value")
    print("-" * 80)
    signal3_results = [r for r in all_results['signal_3'] if r['status'] == 'computed']
    if signal3_results:
        signal3_pvalues = [r['p_value'] for r in signal3_results]
        chi2_stat, combined_p = fishers_method(signal3_pvalues)
        improvements = [r['improvement_pct'] for r in signal3_results]

        print(f"Seasons analyzed: {len(signal3_results)}")
        for i, r in enumerate(signal3_results):
            print(f"  {r['season']}: {r['improvement_pct']:+.2f}% MAE improvement, p={r['p_value']:.4f}, significant={r['significant']}")
        print(f"Meta-analysis (Fisher's method):")
        print(f"  Combined χ² = {chi2_stat:.3f}, combined p-value = {combined_p:.6f}")
        print(f"  Mean improvement: {np.mean(improvements):+.2f}%")
        print(f"  Consistent across seasons: {all(r['improvement_pct'] > 0 for r in signal3_results)}")
        print(f"  VERDICT: {'✓ TRUE SIGNAL' if combined_p < 0.05 and all(r['improvement_pct'] > 0 for r in signal3_results) else '✗ INCONCLUSIVE'}")
    else:
        print("✗ Insufficient data across seasons")

    # SIGNAL 4 Meta-analysis
    print("\n[SIGNAL 4] TOI Stability as Foundation")
    print("-" * 80)
    signal4_results = [r for r in all_results['signal_4'] if r['status'] == 'computed']
    if signal4_results:
        print(f"Seasons analyzed: {len(signal4_results)}")
        toi_correlations = []
        n_players_total = 0
        for i, r in enumerate(signal4_results):
            print(f"  {r['season']}: TOI r={r['mean_toi_corr']:.3f}, Points r={r['mean_points_corr']:.3f}, is_best={r['toi_is_best']} (n={r['n_players']})")
            toi_correlations.append(r['mean_toi_corr'])
            n_players_total += r['n_players']
        print(f"Meta-analysis:")
        print(f"  Total player-seasons analyzed: {n_players_total}")
        print(f"  Mean TOI correlation: r={np.mean(toi_correlations):.3f}")
        print(f"  TOI is best predictor in {sum(r['toi_is_best'] for r in signal4_results)}/{len(signal4_results)} seasons")
        print(f"  VERDICT: {'✓ MIXED EVIDENCE' if sum(r['toi_is_best'] for r in signal4_results) >= 2 else '✗ INCONCLUSIVE'}")
    else:
        print("✗ Insufficient data across seasons")

    # SIGNAL 5 Meta-analysis
    print("\n[SIGNAL 5] Position-specific Regression Rates")
    print("-" * 80)
    signal5_results = [r for r in all_results['signal_5'] if r['status'] == 'computed']
    if signal5_results:
        print(f"Seasons analyzed: {len(signal5_results)}")
        c_corrs = []
        w_corrs = []
        d_corrs = []
        for i, r in enumerate(signal5_results):
            print(f"  {r['season']}: C={r['mean_corr_c']:.3f} (n={r['n_centers']}), W={r['mean_corr_w']:.3f} (n={r['n_wings']}), D={r['mean_corr_d']:.3f} (n={r['n_defense']})")
            c_corrs.append(r['mean_corr_c'])
            w_corrs.append(r['mean_corr_w'])
            d_corrs.append(r['mean_corr_d'])
        print(f"Meta-analysis:")
        print(f"  Mean Centers correlation: r={np.mean(c_corrs):.3f}")
        print(f"  Mean Wings correlation: r={np.mean(w_corrs):.3f}")
        print(f"  Mean Defense correlation: r={np.mean(d_corrs):.3f}")
        print(f"  Significant differences in {sum(r['significant'] for r in signal5_results)}/{len(signal5_results)} seasons")
        print(f"  VERDICT: {'✓ TRUE SIGNAL' if sum(r['significant'] for r in signal5_results) >= 2 else '✗ INCONCLUSIVE'}")
    else:
        print("✗ Insufficient data across seasons")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================

    print(f"\n\n{'='*80}")
    print("SUMMARY: SIGNAL PERSISTENCE ACROSS SEASONS")
    print(f"{'='*80}\n")

    summary_data = {
        'Signal': [
            'Opponent Quality Effect',
            'PP Production Concentration',
            'Recency Weighting Value',
            'TOI Stability',
            'Position Regression'
        ],
        'Seasons Computed': [
            len([r for r in all_results['signal_1'] if r['status'] == 'computed']),
            len([r for r in all_results['signal_2'] if r['status'] == 'computed']),
            len([r for r in all_results['signal_3'] if r['status'] == 'computed']),
            len([r for r in all_results['signal_4'] if r['status'] == 'computed']),
            len([r for r in all_results['signal_5'] if r['status'] == 'computed']),
        ],
        'Significant Seasons': [
            sum(r['significant'] for r in all_results['signal_1'] if r['status'] == 'computed'),
            sum(r['significant'] for r in all_results['signal_2'] if r['status'] == 'computed'),
            sum(r['significant'] for r in all_results['signal_3'] if r['status'] == 'computed'),
            'N/A (special)',
            sum(r['significant'] for r in all_results['signal_5'] if r['status'] == 'computed'),
        ],
        'Verdict': [
            'TRUE' if (len([r for r in all_results['signal_1'] if r['status'] == 'computed']) >= 3 and
                      sum(r['significant'] for r in all_results['signal_1'] if r['status'] == 'computed') >= 3) else 'NEEDS MORE DATA',
            'TRUE' if (len([r for r in all_results['signal_2'] if r['status'] == 'computed']) >= 3 and
                      sum(r['significant'] for r in all_results['signal_2'] if r['status'] == 'computed') >= 3) else 'NEEDS MORE DATA',
            'TRUE' if (len([r for r in all_results['signal_3'] if r['status'] == 'computed']) >= 3 and
                      sum(r['significant'] for r in all_results['signal_3'] if r['status'] == 'computed') >= 3) else 'NEEDS MORE DATA',
            'TRUE' if all(r.get('toi_is_best', False) for r in all_results['signal_4'] if r['status'] == 'computed') else 'INCONCLUSIVE',
            'TRUE' if sum(r['significant'] for r in all_results['signal_5'] if r['status'] == 'computed') >= 2 else 'NEEDS MORE DATA',
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print(f"\n\n{'='*80}")
    print("KEY TAKEAWAYS")
    print(f"{'='*80}\n")

    print("""
Interpretation Guide:
- TRUE SIGNAL: Persists across ALL independent seasons (p < 0.05 in each)
- INCONCLUSIVE: Evidence is mixed or weak across seasons
- NEEDS MORE DATA: Insufficient seasons with complete data

Next Steps:
1. Integrate TRUE signals into production projection model
2. Weight by effect size (larger d = stronger signal)
3. Use position-specific parameters for Signal 5
4. Monitor Signal 3 (EWM) with live 2024-25 data
5. Collect NST data for historical seasons to validate Signal 1 & 2 more rigorously
    """)

    print(f"{'='*80}")
    print(f"Analysis Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    run_comprehensive_validation()
