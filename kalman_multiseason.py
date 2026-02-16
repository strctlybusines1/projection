"""
Multi-season Kalman Filter Calibration for NHL DFS Projections.

OBJECTIVE:
Find globally optimal Q (process noise) and R (observation noise) parameters
using all historical data (2020-2024), then apply them to current season (2024-25).

APPROACH:
1. Reduced grid search using sampled players for speed
2. Find globally-optimal (Q, R) minimizing total MAE across all seasons
3. Run separate calibrations for Centers, Wings, Defense
4. Walk-forward backtest on 2024-25 season with globally-optimized params

Author: Claude Code
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Database path
DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"

# DK Scoring rules
DK_SCORING = {
    'goals': 8.5,
    'assists': 5.0,
    'shots': 1.5,
    'blocked_shots': 1.3,
    'plus_minus': 0.5,
}


# ============================================================================
# Core Kalman Filter (optimized)
# ============================================================================

@dataclass
class KalmanState:
    """State for a single Kalman filter."""
    x: float
    P: float
    n_observations: int = 0


class ScalarKalmanFilter:
    """1-D Kalman filter for tracking a player's true rate."""

    def __init__(self, process_noise: float, observation_noise: float,
                 initial_P: float = 50.0):
        self.Q = process_noise
        self.R = observation_noise
        self.initial_P = initial_P

    def initialize(self, initial_estimate: float) -> KalmanState:
        """Create initial state from a prior estimate."""
        return KalmanState(x=initial_estimate, P=self.initial_P)

    def predict_and_update(self, state: KalmanState, observation: float) -> KalmanState:
        """Combined predict + update (faster than separate steps)."""
        # Predict: P increases by Q
        P_pred = state.P + self.Q
        # Kalman gain
        K = P_pred / (P_pred + self.R)
        # Update estimate and uncertainty
        x_new = state.x + K * (observation - state.x)
        P_new = (1.0 - K) * P_pred
        return KalmanState(x=x_new, P=P_new, n_observations=state.n_observations + 1)


# ============================================================================
# Fast Multi-Season Calibration (Vectorized)
# ============================================================================

class FastMultiSeasonCalibrator:
    """
    Fast grid search using vectorized operations and player sampling.
    """

    def __init__(self):
        # Compact grid
        self.Q_grid = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
        self.R_grid = np.array([5, 10, 15, 20, 30, 40])
        self.seasons = [2020, 2021, 2022, 2023, 2024]
        self.burn_in = 5
        self.min_games = 10
        self.sample_players_per_season = 300  # Sample 300 players per season for speed

    def load_season_data(self, season: int) -> pd.DataFrame:
        """Load historical data for a specific season."""
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(f"""
            SELECT
                season, player_name, position, game_date, dk_fpts
            FROM historical_skaters
            WHERE season = {season}
              AND position IN ('C', 'LW', 'RW', 'W', 'D')
              AND dk_fpts IS NOT NULL
            ORDER BY player_name, game_date
        """, conn)
        conn.close()

        # Normalize positions
        df['position'] = df['position'].apply(lambda x: 'W' if x in ['LW', 'RW', 'L', 'R'] else x)
        df = df.sort_values(['player_name', 'game_date'])

        return df

    def filter_players_with_min_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only players with >= min_games."""
        counts = df.groupby('player_name').size()
        valid_players = counts[counts >= self.min_games].index.tolist()
        return df[df['player_name'].isin(valid_players)]

    def evaluate_params_fast(self, df: pd.DataFrame, Q: float, R: float) -> Tuple[float, int, Dict]:
        """
        Evaluate (Q, R) on a season's data (sampled).
        Returns (MAE, n_predictions, details_dict).
        """
        kf = ScalarKalmanFilter(process_noise=Q, observation_noise=R)

        predictions = []
        errors_by_position = defaultdict(list)

        for player_name, group in df.groupby('player_name'):
            games = group.reset_index(drop=True)

            # Initialize with first burn_in games
            init_fpts = games['dk_fpts'].iloc[:self.burn_in].mean()
            state = kf.initialize(init_fpts)

            # Process burn-in
            for idx in range(self.burn_in):
                obs = games.iloc[idx]['dk_fpts']
                state = kf.predict_and_update(state, obs)

            # Walk-forward predictions
            for idx in range(self.burn_in, len(games)):
                actual = games.iloc[idx]['dk_fpts']
                predicted = state.x
                position = games.iloc[idx]['position']

                error = abs(predicted - actual)
                predictions.append(error)
                errors_by_position[position].append(error)

                state = kf.predict_and_update(state, actual)

        if not predictions:
            return np.inf, 0, {}

        mae = np.mean(predictions)
        return mae, len(predictions), {
            'C': np.mean(errors_by_position['C']) if errors_by_position['C'] else np.nan,
            'W': np.mean(errors_by_position['W']) if errors_by_position['W'] else np.nan,
            'D': np.mean(errors_by_position['D']) if errors_by_position['D'] else np.nan,
        }

    def calibrate_season(self, season: int) -> Dict:
        """Grid search for optimal (Q, R) on a single season."""
        print(f"\nCalibrating season {season}...", end=' ', flush=True)

        df = self.load_season_data(season)
        df = self.filter_players_with_min_games(df)
        n_players = df['player_name'].nunique()
        n_rows = len(df)

        # Sample players for speed
        if n_players > self.sample_players_per_season:
            sampled_players = np.random.choice(df['player_name'].unique(),
                                               self.sample_players_per_season,
                                               replace=False)
            df = df[df['player_name'].isin(sampled_players)]
            n_players_sampled = len(sampled_players)
        else:
            n_players_sampled = n_players

        results = []
        best_mae = np.inf
        best_params = {}

        for Q in self.Q_grid:
            for R in self.R_grid:
                mae, n_pred, details = self.evaluate_params_fast(df, Q, R)

                results.append({
                    'Q': Q,
                    'R': R,
                    'MAE': mae,
                    'n_predictions': n_pred,
                    'MAE_C': details.get('C', np.nan),
                    'MAE_W': details.get('W', np.nan),
                    'MAE_D': details.get('D', np.nan),
                })

                if mae < best_mae and n_pred > 0:
                    best_mae = mae
                    best_params = {'Q': Q, 'R': R, 'MAE': mae}

        results_df = pd.DataFrame(results).sort_values('MAE')

        print(f"Best: Q={best_params['Q']:.2f}, R={best_params['R']:.1f}, MAE={best_params['MAE']:.4f}")

        return {
            'season': season,
            'best_params': best_params,
            'all_results': results_df,
            'n_rows': n_rows,
            'n_players': n_players,
            'n_players_sampled': n_players_sampled,
        }

    def calibrate_all_seasons(self) -> Dict:
        """Calibrate across all seasons and find globally optimal params."""
        print("\n" + "="*70)
        print("  MULTI-SEASON KALMAN FILTER CALIBRATION")
        print("="*70)

        season_results = {}
        global_results = []

        for season in self.seasons:
            result = self.calibrate_season(season)
            season_results[season] = result
            global_results.extend(result['all_results'].to_dict('records'))

        # Aggregate across seasons
        global_df = pd.DataFrame(global_results)
        grouped = global_df.groupby(['Q', 'R'])['MAE'].agg(['mean', 'std', 'count']).reset_index()
        grouped.columns = ['Q', 'R', 'MAE_mean', 'MAE_std', 'n_seasons']
        grouped = grouped.sort_values('MAE_mean')

        best_global = grouped.iloc[0]

        print("\n" + "="*70)
        print("  GLOBALLY OPTIMAL PARAMETERS (across all seasons)")
        print("="*70)
        print(f"Q = {best_global['Q']:.2f}")
        print(f"R = {best_global['R']:.1f}")
        print(f"Mean MAE across seasons: {best_global['MAE_mean']:.4f} (+/- {best_global['MAE_std']:.4f})")
        print(f"\nTop 10 (Q, R) by mean MAE:")
        print(grouped.head(10)[['Q', 'R', 'MAE_mean', 'MAE_std', 'n_seasons']].to_string(index=False))

        # Print per-season summary
        print("\n" + "="*70)
        print("  PER-SEASON SUMMARY")
        print("="*70)
        print(f"{'Season':<10} {'Best Q':<10} {'Best R':<10} {'Best MAE':<12} {'Players':<10} {'Sampled':<10}")
        print("-" * 62)
        for season in sorted(season_results.keys()):
            sr = season_results[season]
            best = sr['best_params']
            print(f"{season:<10} {best['Q']:<10.2f} {best['R']:<10.1f} {best['MAE']:<12.4f} {sr['n_players']:<10} {sr['n_players_sampled']:<10}")

        # Consistency check
        per_season_best = [season_results[s]['best_params'] for s in sorted(season_results.keys())]
        qs = [b['Q'] for b in per_season_best]
        rs = [b['R'] for b in per_season_best]
        print(f"\nConsistency check:")
        print(f"  Q range: {min(qs):.2f} - {max(qs):.2f} (std: {np.std(qs):.3f})")
        print(f"  R range: {min(rs):.1f} - {max(rs):.1f} (std: {np.std(rs):.3f})")

        return {
            'season_results': season_results,
            'global_results': grouped,
            'best_global': {
                'Q': float(best_global['Q']),
                'R': float(best_global['R']),
                'MAE': float(best_global['MAE_mean']),
                'std': float(best_global['MAE_std']),
            }
        }


# ============================================================================
# Position-Specific Calibration (Fast)
# ============================================================================

class FastPositionSpecificCalibrator:
    """Calibrate separate Q/R for Centers, Wings, Defense."""

    def __init__(self):
        self.Q_grid = np.array([0.1, 0.2, 0.5, 1.0])
        self.R_grid = np.array([10, 20, 30, 40])
        self.burn_in = 5
        self.min_games = 10

    def load_all_historical(self) -> pd.DataFrame:
        """Load all historical data."""
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT
                season, player_name, position, game_date, dk_fpts
            FROM historical_skaters
            WHERE position IN ('C', 'LW', 'RW', 'W', 'D')
              AND dk_fpts IS NOT NULL
            ORDER BY season, player_name, game_date
        """, conn)
        conn.close()

        df['position'] = df['position'].apply(lambda x: 'W' if x in ['LW', 'RW', 'L', 'R'] else x)
        return df

    def evaluate_position(self, df_pos: pd.DataFrame, Q: float, R: float) -> Tuple[float, int]:
        """Evaluate params for a single position."""
        kf = ScalarKalmanFilter(process_noise=Q, observation_noise=R)
        errors = []

        for player_name, group in df_pos.groupby('player_name'):
            games = group.reset_index(drop=True)

            if len(games) < self.min_games:
                continue

            init_fpts = games['dk_fpts'].iloc[:self.burn_in].mean()
            state = kf.initialize(init_fpts)

            for idx in range(self.burn_in):
                state = kf.predict_and_update(state, games.iloc[idx]['dk_fpts'])

            for idx in range(self.burn_in, len(games)):
                actual = games.iloc[idx]['dk_fpts']
                error = abs(state.x - actual)
                errors.append(error)
                state = kf.predict_and_update(state, actual)

        if not errors:
            return np.inf, 0
        return np.mean(errors), len(errors)

    def calibrate_positions(self) -> Dict:
        """Calibrate Q/R separately for C, W, D."""
        print("\n" + "="*70)
        print("  POSITION-SPECIFIC CALIBRATION")
        print("="*70)

        df = self.load_all_historical()
        results_by_position = {}

        for position in ['C', 'W', 'D']:
            print(f"\nCalibrating for {position}...", end=' ', flush=True)
            df_pos = df[df['position'] == position]
            n_players = df_pos['player_name'].nunique()
            n_rows = len(df_pos)

            results = []
            best_mae = np.inf
            best_params = {}

            for Q in self.Q_grid:
                for R in self.R_grid:
                    mae, n_pred = self.evaluate_position(df_pos, Q, R)
                    results.append({'Q': Q, 'R': R, 'MAE': mae, 'n_predictions': n_pred})

                    if mae < best_mae and n_pred > 0:
                        best_mae = mae
                        best_params = {'Q': Q, 'R': R, 'MAE': mae}

            results_df = pd.DataFrame(results).sort_values('MAE')
            results_by_position[position] = {
                'best_params': best_params,
                'all_results': results_df,
                'n_players': n_players,
            }

            print(f"Best: Q={best_params['Q']:.2f}, R={best_params['R']:.1f}, MAE={best_params['MAE']:.4f}")

        print("\n" + "="*70)
        print("  POSITION-SPECIFIC SUMMARY")
        print("="*70)
        print(f"{'Position':<10} {'Best Q':<10} {'Best R':<10} {'Best MAE':<12}")
        print("-" * 42)
        for pos in ['C', 'W', 'D']:
            best = results_by_position[pos]['best_params']
            print(f"{pos:<10} {best['Q']:<10.2f} {best['R']:<10.1f} {best['MAE']:<12.4f}")

        return results_by_position


# ============================================================================
# Walk-Forward Backtest on Current Season (2024-25)
# ============================================================================

class CurrentSeasonBacktest:
    """Walk-forward backtest on 2024-25 season using globally-optimized params."""

    def __init__(self, Q: float, R: float):
        self.Q = Q
        self.R = R
        self.kf = ScalarKalmanFilter(process_noise=Q, observation_noise=R)
        self.burn_in = 5
        self.min_games = 5

    def load_current_season(self) -> pd.DataFrame:
        """Load 2024-25 season data from boxscore_skaters."""
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT
                player_name, position, game_date, dk_fpts
            FROM boxscore_skaters
            WHERE position IN ('C', 'LW', 'RW', 'W', 'D')
              AND dk_fpts IS NOT NULL
            ORDER BY player_name, game_date
        """, conn)
        conn.close()

        df['position'] = df['position'].apply(lambda x: 'W' if x in ['LW', 'RW', 'L', 'R'] else x)
        return df.sort_values(['player_name', 'game_date'])

    def backtest(self) -> pd.DataFrame:
        """Walk-forward backtest: predict each game before observing it."""
        print("\n" + "="*70)
        print(f"  WALK-FORWARD BACKTEST: 2024-25 Season")
        print(f"  Q={self.Q:.2f}, R={self.R:.1f}")
        print("="*70)

        df = self.load_current_season()
        print(f"Loaded {len(df):,} game logs from {df['player_name'].nunique():,} players")

        predictions = []
        errors_by_position = defaultdict(list)

        for player_name, group in df.groupby('player_name'):
            games = group.reset_index(drop=True)

            if len(games) < self.min_games:
                continue

            # Initialize with first N games
            init_fpts = games['dk_fpts'].iloc[:self.burn_in].mean()
            state = self.kf.initialize(init_fpts)

            # Burn-in period
            for idx in range(self.burn_in):
                state = self.kf.predict_and_update(state, games.iloc[idx]['dk_fpts'])

            # Walk-forward predictions
            for idx in range(self.burn_in, len(games)):
                actual = games.iloc[idx]['dk_fpts']
                predicted = state.x
                position = games.iloc[idx]['position']
                game_date = games.iloc[idx]['game_date']
                error = abs(predicted - actual)

                predictions.append({
                    'player_name': player_name,
                    'position': position,
                    'game_date': game_date,
                    'game_number': idx + 1,
                    'actual_fpts': actual,
                    'predicted_fpts': predicted,
                    'error': error,
                })

                errors_by_position[position].append(error)
                state = self.kf.predict_and_update(state, actual)

        results_df = pd.DataFrame(predictions)

        # Compute metrics
        if not results_df.empty:
            mae = results_df['error'].mean()
            rmse = np.sqrt((results_df['error'] ** 2).mean())
            corr = results_df['actual_fpts'].corr(results_df['predicted_fpts'])

            print(f"\nResults:")
            print(f"  Predictions: {len(results_df):,}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Correlation: {corr:.4f}")

            print(f"\nBy position:")
            for pos in ['C', 'W', 'D']:
                if errors_by_position[pos]:
                    pos_mae = np.mean(errors_by_position[pos])
                    pos_n = len(errors_by_position[pos])
                    print(f"  {pos}: MAE={pos_mae:.4f} ({pos_n:,} games)")

            # Improvement over baseline
            improvement = 4.318 - mae
            print(f"\nImprovement vs single-season Kalman (MAE 4.318): {improvement:+.4f}")

        return results_df


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run full multi-season calibration analysis."""

    print("\n" + "="*70)
    print("  MULTI-SEASON KALMAN FILTER CALIBRATION FOR NHL DFS")
    print("="*70)
    print(f"Database: {DB_PATH}")
    print(f"Seasons: 2020-2024 (historical) + 2024-25 (current)")
    print(f"Previous single-season Kalman MAE: 4.318")
    print()

    # 1. Multi-season calibration (global optimization)
    print("\n" + "#" * 70)
    print("# STAGE 1: MULTI-SEASON GLOBAL OPTIMIZATION")
    print("#" * 70)
    multi_cal = FastMultiSeasonCalibrator()
    multi_results = multi_cal.calibrate_all_seasons()

    # 2. Position-specific calibration
    print("\n" + "#" * 70)
    print("# STAGE 2: POSITION-SPECIFIC CALIBRATION")
    print("#" * 70)
    pos_cal = FastPositionSpecificCalibrator()
    pos_results = pos_cal.calibrate_positions()

    # 3. Walk-forward backtest on current season
    print("\n" + "#" * 70)
    print("# STAGE 3: WALK-FORWARD BACKTEST ON 2024-25 SEASON")
    print("#" * 70)
    global_params = multi_results['best_global']
    current_backtest = CurrentSeasonBacktest(
        Q=global_params['Q'],
        R=global_params['R']
    )
    backtest_results = current_backtest.backtest()

    # 4. Summary report
    print("\n" + "="*70)
    print("  COMPREHENSIVE SUMMARY")
    print("="*70)

    print("\n1. GLOBALLY OPTIMAL PARAMETERS (all seasons)")
    print(f"   Q = {global_params['Q']:.2f}")
    print(f"   R = {global_params['R']:.1f}")
    print(f"   MAE (historical) = {global_params['MAE']:.4f} +/- {global_params['std']:.4f}")

    print("\n2. POSITION-SPECIFIC PARAMETERS")
    for pos in ['C', 'W', 'D']:
        best = pos_results[pos]['best_params']
        print(f"   {pos}: Q={best['Q']:.2f}, R={best['R']:.1f}, MAE={best['MAE']:.4f}")

    print("\n3. 2024-25 SEASON WALK-FORWARD RESULTS")
    if not backtest_results.empty:
        mae_2024 = backtest_results['error'].mean()
        print(f"   MAE = {mae_2024:.4f}")
        print(f"   RMSE = {np.sqrt((backtest_results['error']**2).mean()):.4f}")
        print(f"   Correlation = {backtest_results['actual_fpts'].corr(backtest_results['predicted_fpts']):.4f}")
        improvement = 4.318 - mae_2024
        print(f"   Improvement vs single-season (4.318): {improvement:+.4f}")

    print("\n4. CONSISTENCY CHECK (parameter stability across seasons)")
    season_results = multi_results['season_results']
    per_season_best = [season_results[s]['best_params'] for s in sorted(season_results.keys())]
    qs = [b['Q'] for b in per_season_best]
    rs = [b['R'] for b in per_season_best]
    print(f"   Q consistency: std={np.std(qs):.3f} (range {min(qs):.2f}-{max(qs):.2f})")
    print(f"   R consistency: std={np.std(rs):.3f} (range {min(rs):.1f}-{max(rs):.1f})")

    if np.std(qs) < 0.3 and np.std(rs) < 10:
        print("   VERDICT: Parameters are highly consistent across seasons!")
    else:
        print("   VERDICT: Parameters vary; consider position-specific tuning")

    print("\n5. RECOMMENDATIONS")
    if not backtest_results.empty:
        mae_2024 = backtest_results['error'].mean()
        if 4.318 - mae_2024 > 0.05:
            print(f"   Use globally-optimized params (Q={global_params['Q']:.2f}, R={global_params['R']:.1f})")
        else:
            print(f"   Global optimization shows minimal improvement; consider ensemble approaches")

    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70 + "\n")

    return {
        'multi_season': multi_results,
        'position_specific': pos_results,
        'backtest_current': backtest_results,
    }


if __name__ == "__main__":
    results = main()
