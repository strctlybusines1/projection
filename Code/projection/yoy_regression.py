#!/usr/bin/env python3
"""
Year-over-Year Regression Analysis for NHL DFS Projections
==========================================================

Computes YoY correlation coefficients for player stats to determine true regression rates.
Based on Jim Simons-style analysis: correlation reveals signal persistence.

Usage:
    python3 yoy_regression.py

Output:
    - Comprehensive YoY correlation table
    - Optimal regression weights (shrinkage factors)
    - Minimum sample sizes per stat (split-half reliability)
    - Signal persistence tests (Cohen's d effect sizes)
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, t
from typing import Dict, Tuple, List, Optional
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')


class YoYRegressionAnalysis:
    """Compute year-over-year correlation and regression analysis."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load historical and current season data."""
        # Load historical data with season column
        historical = pd.read_sql_query(
            "SELECT * FROM historical_skaters",
            self.conn
        )

        # Load current season (boxscore_skaters) and add season column
        current = pd.read_sql_query(
            "SELECT * FROM boxscore_skaters",
            self.conn
        )
        current['season'] = 2024

        return historical, current

    def preprocess_data(self, historical: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        """
        Combine historical and current data, standardize column names.
        Infer pp_assists from pp_points - pp_goals.
        """
        # Standardize column names: use lowercase, handle variations
        historical_cols = historical.columns.str.lower()
        current_cols = current.columns.str.lower()

        historical.columns = historical_cols
        current.columns = current_cols

        # Rename blocked_shots to blocks for consistency
        if 'blocked_shots' in historical.columns:
            historical = historical.rename(columns={'blocked_shots': 'blocks'})
        if 'blocked_shots' in current.columns:
            current = current.rename(columns={'blocked_shots': 'blocks'})

        # Rename shot-related columns
        if 'shots' in historical.columns:
            # Already correct
            pass

        # For historical: infer pp_assists from pp_points
        if 'pp_points' in historical.columns and 'pp_goals' in historical.columns:
            historical['pp_assists'] = historical['pp_points'] - historical['pp_goals']
            historical['pp_assists'] = historical['pp_assists'].clip(lower=0)

        # For historical: infer sh_assists from sh_points
        if 'sh_points' in historical.columns and 'sh_goals' in historical.columns:
            historical['sh_assists'] = historical['sh_points'] - historical['sh_goals']
            historical['sh_assists'] = historical['sh_assists'].clip(lower=0)
        else:
            # If sh_goals/sh_points don't exist, create zeros
            historical['sh_goals'] = 0
            historical['sh_assists'] = 0

        # For current season, create sh_ columns as zero (not tracked in boxscore)
        if 'sh_goals' not in current.columns:
            current['sh_goals'] = 0
        if 'sh_assists' not in current.columns:
            current['sh_assists'] = 0
        if 'pp_assists' not in current.columns:
            # Infer from points if available, else zero
            if 'points' in current.columns and 'pp_goals' in current.columns:
                current['pp_assists'] = current['points'] - current['goals'] - current['pp_goals']
                current['pp_assists'] = current['pp_assists'].clip(lower=0)
            else:
                current['pp_assists'] = 0

        # Combine datasets
        combined = pd.concat([historical, current], ignore_index=True, sort=False)

        # Forward fill missing columns with zeros
        for col in ['pp_assists', 'sh_goals', 'sh_assists']:
            if col not in combined.columns:
                combined[col] = 0

        return combined

    def compute_season_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-season, per-player rates.

        Returns DataFrame with columns:
        - season, player_name, team, position
        - gp (games played), toi_total
        - Per-game rates: goals_pg, assists_pg, shots_pg, blocks_pg, hits_pg, pim_pg, etc.
        - Per-60 rates: goals_per60, assists_per60, shots_per60, etc.
        - toi_per_game
        """
        # Ensure required columns exist
        for col in ['goals', 'assists', 'shots', 'blocks', 'hits', 'pim',
                    'pp_goals', 'pp_assists', 'sh_goals', 'sh_assists', 'toi_seconds', 'dk_fpts']:
            if col not in df.columns:
                df[col] = 0

        # Fill NaN with 0
        stat_cols = ['goals', 'assists', 'shots', 'blocks', 'hits', 'pim',
                     'pp_goals', 'pp_assists', 'sh_goals', 'sh_assists', 'toi_seconds', 'dk_fpts']
        for col in stat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Group by season, player, team, position
        grouped = df.groupby(['season', 'player_name', 'team', 'position']).agg({
            'goals': 'sum',
            'assists': 'sum',
            'shots': 'sum',
            'blocks': 'sum',
            'hits': 'sum',
            'pim': 'sum',
            'pp_goals': 'sum',
            'pp_assists': 'sum',
            'sh_goals': 'sum',
            'sh_assists': 'sum',
            'toi_seconds': 'sum',
            'dk_fpts': 'sum',
            'game_id': 'count',  # games played
        }).reset_index()

        # Rename game_id to gp
        grouped = grouped.rename(columns={'game_id': 'gp'})

        # Apply minimum games filter (season-dependent)
        # 2020-21 was COVID-shortened (56 games) -> min 15 games
        # Others -> min 20 games
        min_games = grouped['season'].apply(lambda s: 15 if s == 2020 else 20)
        grouped = grouped[grouped['gp'] >= min_games].reset_index(drop=True)

        if len(grouped) == 0:
            print("Warning: No players meet minimum games threshold")
            return grouped

        # Compute per-game rates
        grouped['goals_pg'] = grouped['goals'] / grouped['gp']
        grouped['assists_pg'] = grouped['assists'] / grouped['gp']
        grouped['shots_pg'] = grouped['shots'] / grouped['gp']
        grouped['blocks_pg'] = grouped['blocks'] / grouped['gp']
        grouped['hits_pg'] = grouped['hits'] / grouped['gp']
        grouped['pim_pg'] = grouped['pim'] / grouped['gp']
        grouped['pp_goals_pg'] = grouped['pp_goals'] / grouped['gp']
        grouped['pp_assists_pg'] = grouped['pp_assists'] / grouped['gp']
        grouped['sh_goals_pg'] = grouped['sh_goals'] / grouped['gp']
        grouped['sh_assists_pg'] = grouped['sh_assists'] / grouped['gp']
        grouped['dk_fpts_pg'] = grouped['dk_fpts'] / grouped['gp']
        grouped['toi_per_game'] = grouped['toi_seconds'] / grouped['gp']

        # Compute per-60 rates (only if TOI > 0)
        for stat in ['goals', 'assists', 'shots', 'blocks', 'hits', 'pp_goals', 'pp_assists', 'sh_goals', 'sh_assists']:
            col_name = f'{stat}_per60'
            grouped[col_name] = np.where(
                grouped['toi_seconds'] > 0,
                (grouped[stat] / grouped['toi_seconds']) * 3600,
                0
            )

        return grouped

    def pair_consecutive_seasons(self, season_rates: pd.DataFrame) -> pd.DataFrame:
        """
        Pair consecutive seasons for same player.
        Returns DataFrame with columns:
        - player_name, y1_season, y2_season, position
        - [stat]_y1, [stat]_y2 for each rate stat
        """
        # Get all player-season combinations, sorted
        ps = season_rates.sort_values(['player_name', 'season']).reset_index(drop=True)

        # Self-join on player_name with season offset
        pairs = []
        for player in ps['player_name'].unique():
            player_seasons = ps[ps['player_name'] == player].sort_values('season')

            if len(player_seasons) < 2:
                continue

            # Pair consecutive seasons
            for i in range(len(player_seasons) - 1):
                y1 = player_seasons.iloc[i]
                y2 = player_seasons.iloc[i + 1]

                # Only pair if seasons are consecutive (or 1 apart, in case some years missing)
                if y2['season'] - y1['season'] > 2:
                    continue

                pair = {
                    'player_name': player,
                    'y1_season': int(y1['season']),
                    'y2_season': int(y2['season']),
                    'position': y1['position'],
                }

                # Add all rate stats
                rate_stats = [
                    'goals_pg', 'assists_pg', 'shots_pg', 'blocks_pg', 'hits_pg', 'pim_pg',
                    'pp_goals_pg', 'pp_assists_pg', 'sh_goals_pg', 'sh_assists_pg', 'dk_fpts_pg',
                    'goals_per60', 'assists_per60', 'shots_per60', 'blocks_per60', 'hits_per60',
                    'pp_goals_per60', 'pp_assists_per60', 'sh_goals_per60', 'sh_assists_per60',
                    'toi_per_game'
                ]

                for stat in rate_stats:
                    pair[f'{stat}_y1'] = y1[stat] if stat in y1.index else 0
                    pair[f'{stat}_y2'] = y2[stat] if stat in y2.index else 0

                pairs.append(pair)

        return pd.DataFrame(pairs)

    def compute_yoy_correlations(self, pairs: pd.DataFrame) -> Dict:
        """
        Compute YoY correlations for all rate stats.

        Returns dict with:
        - correlation
        - p_value
        - n_pairs
        - ci_lower, ci_upper (95% CI)
        - regression_weight (1 - correlation)
        """
        if len(pairs) == 0:
            print("Warning: No player-season pairs found")
            return {}

        rate_stats = [
            'goals_pg', 'assists_pg', 'shots_pg', 'blocks_pg', 'hits_pg', 'pim_pg',
            'pp_goals_pg', 'pp_assists_pg', 'sh_goals_pg', 'sh_assists_pg', 'dk_fpts_pg',
            'goals_per60', 'assists_per60', 'shots_per60', 'blocks_per60', 'hits_per60',
            'pp_goals_per60', 'pp_assists_per60', 'sh_goals_per60', 'sh_assists_per60',
            'toi_per_game'
        ]

        results = {}

        for stat in rate_stats:
            y1_col = f'{stat}_y1'
            y2_col = f'{stat}_y2'

            if y1_col not in pairs.columns or y2_col not in pairs.columns:
                continue

            # Filter out NaN/inf
            valid = pairs[[y1_col, y2_col]].dropna()
            valid = valid[(np.isfinite(valid[y1_col])) & (np.isfinite(valid[y2_col]))]

            if len(valid) < 5:  # Need at least 5 pairs
                continue

            y1_vals = valid[y1_col].values
            y2_vals = valid[y2_col].values

            # Correlation
            r, p_val = pearsonr(y1_vals, y2_vals)
            n = len(valid)

            # 95% CI using Fisher transform
            se = 1 / np.sqrt(n - 3) if n > 3 else np.inf
            z = 0.5 * np.log((1 + r) / (1 - r)) if -0.9999 < r < 0.9999 else 0
            ci_z_lower = z - 1.96 * se
            ci_z_upper = z + 1.96 * se
            ci_r_lower = np.tanh(ci_z_lower)
            ci_r_upper = np.tanh(ci_z_upper)

            # Regression weight (Bayesian shrinkage factor)
            regression_weight = r  # How much of Y1 predicts Y2
            shrinkage_factor = 1 - r  # How much to shrink toward mean

            results[stat] = {
                'correlation': r,
                'p_value': p_val,
                'n_pairs': n,
                'ci_lower': ci_r_lower,
                'ci_upper': ci_r_upper,
                'regression_weight': regression_weight,
                'shrinkage_factor': shrinkage_factor,
                'se': se,
                'y1_mean': y1_vals.mean(),
                'y2_mean': y2_vals.mean(),
                'y1_std': y1_vals.std(),
                'y2_std': y2_vals.std(),
            }

        return results

    def compute_minimum_sample_sizes(self, df: pd.DataFrame) -> Dict:
        """
        Compute minimum sample sizes using split-half reliability.

        For each stat and each game count (10, 15, 20, ..., 82):
        - Split each player-season into first half vs second half
        - Compute correlation between halves
        - Find game count where correlation stabilizes > 0.70
        """
        results = {}

        game_counts = [10, 15, 20, 25, 30, 40, 50, 60, 70, 82]

        # Ensure game_id column exists for sorting
        if 'game_id' not in df.columns:
            print("Warning: game_id not in dataframe, using observation order")
            df = df.reset_index(drop=True)
            df['game_id'] = df.groupby(['season', 'player_name']).cumcount()

        rate_stats = [
            'goals', 'assists', 'shots', 'blocks', 'hits', 'pim',
            'pp_goals', 'pp_assists', 'sh_goals', 'sh_assists', 'dk_fpts'
        ]

        for stat in rate_stats:
            if stat not in df.columns:
                continue

            split_correlations = {}

            for gp_threshold in game_counts:
                player_season_groups = df.groupby(['season', 'player_name'])

                half1_rates = []
                half2_rates = []

                for (season, player), group in player_season_groups:
                    if len(group) < gp_threshold:
                        continue

                    # Take first gp_threshold games
                    group_subset = group.iloc[:gp_threshold]

                    if len(group_subset) < 2:
                        continue

                    mid = len(group_subset) // 2
                    if mid == 0:
                        continue

                    half1 = group_subset.iloc[:mid]
                    half2 = group_subset.iloc[mid:]

                    # Per-game rates
                    if stat in ['toi_seconds']:
                        rate1 = half1[stat].sum() / len(half1)
                        rate2 = half2[stat].sum() / len(half2)
                    else:
                        rate1 = half1[stat].sum() / len(half1)
                        rate2 = half2[stat].sum() / len(half2)

                    half1_rates.append(rate1)
                    half2_rates.append(rate2)

                if len(half1_rates) < 5:
                    split_correlations[gp_threshold] = np.nan
                    continue

                r, _ = pearsonr(half1_rates, half2_rates)
                split_correlations[gp_threshold] = r

            results[stat] = split_correlations

        return results

    def signal_persistence_test(self, pairs: pd.DataFrame, season_rates: pd.DataFrame) -> Dict:
        """
        Test persistence of player groupings across seasons.

        For key signals:
        - Divide players by Y1 quartile/percentile
        - Compute effect sizes (Cohen's d) showing persistence in Y2
        """
        results = {}

        # Test key signals
        test_signals = [
            ('goals_pg', 'Goals per game'),
            ('shots_pg', 'Shots per game'),
            ('dk_fpts_pg', 'DK FPTS per game'),
            ('pp_goals_pg', 'PP goals per game'),
        ]

        for signal_stat, label in test_signals:
            if len(pairs) == 0:
                continue

            y1_col = f'{signal_stat}_y1'
            y2_col = f'{signal_stat}_y2'

            if y1_col not in pairs.columns or y2_col not in pairs.columns:
                continue

            # Valid pairs
            valid = pairs[[y1_col, y2_col, 'player_name']].dropna()
            valid = valid[(np.isfinite(valid[y1_col])) & (np.isfinite(valid[y2_col]))]

            if len(valid) < 20:
                continue

            # Split into quartiles based on Y1
            valid['y1_quartile'] = pd.qcut(valid[y1_col], q=4, duplicates='drop', labels=False)

            # Compute Cohen's d for each quartile
            quartile_effects = {}

            for q in sorted(valid['y1_quartile'].unique()):
                q_players = valid[valid['y1_quartile'] == q]
                q_y2 = q_players[y2_col]

                others = valid[valid['y1_quartile'] != q]
                others_y2 = others[y2_col]

                # Cohen's d
                mean_diff = q_y2.mean() - others_y2.mean()
                pooled_std = np.sqrt(
                    (q_y2.std()**2 + others_y2.std()**2) / 2
                )
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                quartile_effects[f'Q{q+1}'] = {
                    'mean_y1': q_players[y1_col].mean(),
                    'mean_y2': q_y2.mean(),
                    'cohens_d': cohens_d,
                    'n': len(q_players),
                }

            results[signal_stat] = {
                'label': label,
                'quartile_effects': quartile_effects,
            }

        return results

    def export_regression_weights(self, yoy_corr: Dict) -> str:
        """
        Export regression weights to CSV for use in models.

        Returns path to exported CSV file.
        """
        if not yoy_corr:
            return None

        # Build DataFrame
        rows = []
        for stat, metrics in yoy_corr.items():
            if np.isnan(metrics['correlation']):
                continue

            rows.append({
                'statistic': stat,
                'yoy_correlation': metrics['correlation'],
                'regression_weight': metrics['regression_weight'],
                'shrinkage_factor': metrics['shrinkage_factor'],
                'n_pairs': metrics['n_pairs'],
                'p_value': metrics['p_value'],
                'ci_lower': metrics['ci_lower'],
                'ci_upper': metrics['ci_upper'],
                'y1_mean': metrics['y1_mean'],
                'y2_mean': metrics['y2_mean'],
                'y1_std': metrics['y1_std'],
                'y2_std': metrics['y2_std'],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('yoy_correlation', ascending=False)

        # Export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'/sessions/youthful-funny-faraday/mnt/Code/projection/yoy_regression_weights_{timestamp}.csv'

        df.to_csv(output_path, index=False)
        return output_path

    def run(self):
        """Execute full analysis."""
        print("=" * 80)
        print("YEAR-OVER-YEAR NHL DFS REGRESSION ANALYSIS")
        print("=" * 80)
        print()

        # Load and preprocess
        print("Loading data from database...")
        historical, current = self.load_data()
        print(f"  Historical: {len(historical)} rows, seasons {historical['season'].unique()}")
        print(f"  Current (2024-25): {len(current)} rows")

        combined = self.preprocess_data(historical, current)
        print(f"  Combined: {len(combined)} rows, seasons {sorted(combined['season'].unique())}")
        print()

        # Compute season rates
        print("Computing per-season player rates...")
        season_rates = self.compute_season_rates(combined)
        print(f"  Player-seasons: {len(season_rates)}")
        print(f"  Seasons: {sorted(season_rates['season'].unique())}")
        print()

        # Pair consecutive seasons
        print("Pairing consecutive seasons...")
        pairs = self.pair_consecutive_seasons(season_rates)
        print(f"  Player-season pairs: {len(pairs)}")
        if len(pairs) > 0:
            print(f"  Season pairs: {pairs[['y1_season', 'y2_season']].drop_duplicates().values}")
        print()

        # Compute YoY correlations
        print("Computing year-over-year correlations...")
        yoy_corr = self.compute_yoy_correlations(pairs)
        print(f"  Statistics analyzed: {len(yoy_corr)}")
        print()

        # Print YoY correlation table
        print("=" * 80)
        print("YEAR-OVER-YEAR CORRELATION TABLE")
        print("=" * 80)
        print(f"{'Statistic':<30} {'Corr':<8} {'N':<6} {'95% CI':<20} {'P-val':<10} {'Shrink':<8}")
        print("-" * 80)

        sorted_stats = sorted(yoy_corr.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)

        for stat, metrics in sorted_stats:
            r = metrics['correlation']
            n = metrics['n_pairs']
            ci = f"[{metrics['ci_lower']:.3f}, {metrics['ci_upper']:.3f}]"
            p = metrics['p_value']
            shrink = metrics['shrinkage_factor']

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            p_str = f"{p:.4f}{sig}"

            print(f"{stat:<30} {r:>7.3f}  {n:>5d}  {ci:<20} {p_str:<10} {shrink:>7.3f}")

        print()

        # Compute minimum sample sizes
        print("Computing minimum sample sizes via split-half reliability...")
        min_samples = self.compute_minimum_sample_sizes(combined)
        print(f"  Stats analyzed: {len(min_samples)}")
        print()

        # Print minimum sample sizes
        print("=" * 80)
        print("MINIMUM SAMPLE SIZES (Game Count for 0.70+ Split-Half Reliability)")
        print("=" * 80)
        print(f"{'Statistic':<25} {'Min Games':<12} {'Reliability Trajectory':<50}")
        print("-" * 80)

        for stat in sorted(min_samples.keys()):
            reliabilities = min_samples[stat]

            # Find inflection point where reliability > 0.70
            min_games_for_70 = None
            for gp, r in sorted(reliabilities.items()):
                if not np.isnan(r) and r >= 0.70:
                    min_games_for_70 = gp
                    break

            if min_games_for_70 is None:
                min_games_for_70 = ">82"

            # Build trajectory string
            trajectory = []
            for gp in [10, 15, 20, 25, 30, 40, 50, 60, 70, 82]:
                if gp in reliabilities:
                    r = reliabilities[gp]
                    if np.isnan(r):
                        trajectory.append("--")
                    else:
                        trajectory.append(f"{r:.2f}")
            trajectory_str = " → ".join(trajectory)

            print(f"{stat:<25} {str(min_games_for_70):<12} {trajectory_str:<50}")

        print()

        # Signal persistence test
        print("Running signal persistence tests...")
        signal_persist = self.signal_persistence_test(pairs, season_rates)
        print(f"  Signals tested: {len(signal_persist)}")
        print()

        # Print signal persistence results
        print("=" * 80)
        print("SIGNAL PERSISTENCE TESTS (Cohen's d by Quartile)")
        print("=" * 80)

        for signal_stat in sorted(signal_persist.keys()):
            signal_data = signal_persist[signal_stat]
            label = signal_data['label']

            print(f"\n{label} ({signal_stat}):")
            cohens_label = "Cohen's d"
            print(f"  {'Quartile':<12} {'Mean Y1':<12} {'Mean Y2':<12} {cohens_label:<12} {'N':<6}")
            print(f"  " + "-" * 50)

            for quartile, metrics in sorted(signal_data['quartile_effects'].items()):
                d = metrics['cohens_d']
                print(f"  {quartile:<12} {metrics['mean_y1']:>11.3f}  {metrics['mean_y2']:>11.3f}  {d:>11.3f}  {metrics['n']:>5d}")

        print()

        # Summary statistics
        print("=" * 80)
        print("REGRESSION SUMMARY")
        print("=" * 80)

        if yoy_corr:
            corrs = [m['correlation'] for m in yoy_corr.values() if not np.isnan(m['correlation'])]
            if corrs:
                print(f"Average YoY correlation: {np.mean(corrs):.3f}")
                print(f"Median YoY correlation: {np.median(corrs):.3f}")
                print(f"Range: [{np.min(corrs):.3f}, {np.max(corrs):.3f}]")
            print()

            print("Key insights:")

            # High correlation (sticky) stats
            sticky = [s for s, m in yoy_corr.items()
                     if not np.isnan(m['correlation']) and m['correlation'] > 0.60 and m['n_pairs'] >= 10]
            if sticky:
                print(f"  - Sticky stats (r > 0.60, use minimal shrinkage):")
                for s in sorted(sticky, key=lambda x: yoy_corr[x]['correlation'], reverse=True)[:5]:
                    r = yoy_corr[s]['correlation']
                    shrink = yoy_corr[s]['shrinkage_factor']
                    print(f"      {s}: r={r:.3f}, shrinkage={shrink:.3f}")

            # Moderate correlation stats
            moderate = [s for s, m in yoy_corr.items()
                       if not np.isnan(m['correlation']) and 0.30 <= m['correlation'] <= 0.60 and m['n_pairs'] >= 10]
            if moderate:
                print(f"  - Moderate stats (0.30 < r < 0.60, moderate shrinkage):")
                for s in sorted(moderate, key=lambda x: yoy_corr[x]['correlation'], reverse=True)[:5]:
                    r = yoy_corr[s]['correlation']
                    shrink = yoy_corr[s]['shrinkage_factor']
                    print(f"      {s}: r={r:.3f}, shrinkage={shrink:.3f}")

            # Noisy stats
            noisy = [s for s, m in yoy_corr.items()
                    if not np.isnan(m['correlation']) and m['correlation'] < 0.30 and m['n_pairs'] >= 10]
            if noisy:
                print(f"  - Noisy stats (r < 0.30, heavy shrinkage toward mean):")
                for s in sorted(noisy, key=lambda x: yoy_corr[x]['correlation'])[:5]:
                    r = yoy_corr[s]['correlation']
                    shrink = yoy_corr[s]['shrinkage_factor']
                    print(f"      {s}: r={r:.3f}, shrinkage={shrink:.3f}")

        print()

        # Regression weight guide
        print("=" * 80)
        print("REGRESSION WEIGHTS & SHRINKAGE FORMULA")
        print("=" * 80)
        print()
        print("Bayesian Regression Adjustment Formula:")
        print("  regressed_value = r × observed_value + (1-r) × league_average")
        print()
        print("Where:")
        print("  r = YoY correlation (regression_weight)")
        print("  (1-r) = shrinkage_factor (toward mean)")
        print()
        print("Examples:")
        print("  - Stat with r=0.85: use 85% observed + 15% league avg")
        print("  - Stat with r=0.50: use 50% observed + 50% league avg")
        print("  - Stat with r=0.25: use 25% observed + 75% league avg")
        print()

        if yoy_corr:
            # Top candidates for shrinkage application
            sorted_by_r = sorted(
                [(s, m) for s, m in yoy_corr.items() if not np.isnan(m['correlation'])],
                key=lambda x: x[1]['correlation'],
                reverse=True
            )

            print("Top candidates for projection shrinkage (predictable stats):")
            print(f"  {'Stat':<30} {'Regress Weight':<15} {'Formula':<35}")
            print(f"  " + "-" * 80)
            for stat, metrics in sorted_by_r[:8]:
                r = metrics['regression_weight']
                shrink = metrics['shrinkage_factor']
                formula = f"{r:.1%} obs + {shrink:.1%} avg"
                print(f"  {stat:<30} {r:>14.3f}   {formula:<35}")

        print()
        print("=" * 80)
        print("INTERPRETATION GUIDE")
        print("=" * 80)
        print()
        print("1. YoY CORRELATIONS")
        print("   - r > 0.80: Very sticky (injury/role changes are exceptions)")
        print("   - r 0.60-0.80: Sticky (most regression models should use <20% shrinkage)")
        print("   - r 0.40-0.60: Moderate (50/50 blend recommended)")
        print("   - r < 0.40: Noisy (heavy shrinkage toward league average)")
        print()
        print("2. MINIMUM SAMPLE SIZES")
        print("   - Game count where split-half reliability reaches 0.70")
        print("   - Stats reaching 0.70+ reliability at lower game counts are more stable")
        print("   - Example: 'blocks: 20 games' means reliable blocking data at 20+ GP")
        print()
        print("3. SIGNAL PERSISTENCE (Cohen's d)")
        print("   - |d| > 2.0: Extreme persistence (outliers stay outliers)")
        print("   - |d| > 1.2: Large effect (clear groupings year-to-year)")
        print("   - |d| 0.5-1.2: Moderate effect (some mean reversion)")
        print("   - |d| < 0.5: Small effect (significant regression to mean)")
        print()
        print("4. KEY FINDING")
        if yoy_corr:
            corrs = [m['correlation'] for m in yoy_corr.values() if not np.isnan(m['correlation'])]
            if corrs:
                avg_r = np.mean(corrs)
                print(f"   Average YoY correlation across all stats: {avg_r:.3f}")
                print(f"   This means on average, knowing a player's {int((1-avg_r)*100)}% of their stats")
                print(f"   from last season regresses toward league average by year N+1.")
                print()
                print(f"   Most predictable stats (r > 0.80):")
                predictable = [s for s, m in sorted_by_r if m['correlation'] > 0.80]
                for s in predictable[:3]:
                    print(f"      - {s}")
        print()

        # Export regression weights
        print("=" * 80)
        print("EXPORTING REGRESSION WEIGHTS")
        print("=" * 80)

        csv_path = self.export_regression_weights(yoy_corr)
        if csv_path:
            print(f"Regression weights exported to: {csv_path}")
            print()
            print("Use these weights in your projection model with:")
            print("  projection = regression_weight * observed + shrinkage_factor * league_average")
            print()

        print("=" * 80)
        print("Analysis complete.")
        print("=" * 80)


if __name__ == '__main__':
    db_path = '/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db'

    analysis = YoYRegressionAnalysis(db_path)
    analysis.run()
