"""
Multi-season Kalman Filter Calibration for NHL DFS Projections.

OBJECTIVE:
Find globally optimal Q (process noise) and R (observation noise) parameters
using all historical data (2020-2024, 252K rows), then apply them to current
season (2024-25) predictions.

KEY QUESTION:
Can parameters optimized across 5 seasons beat season-specific parameters
and the previous single-season Kalman (MAE 4.318)?

APPROACH:
1. Grid search over (Q, R) pairs for each season separately
2. Find globally-optimal (Q, R) minimizing total MAE across all seasons
3. Run separate calibrations for Centers, Wings, Defense
4. Compare single-FPTS Kalman vs multi-stat Kalman approaches
5. Walk-forward backtest on 2024-25 season with globally-optimized params
6. Test opponent quality adjustment on Kalman-filtered projections

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
# Core Kalman Filter (reused from kalman_projection.py)
# ============================================================================

@dataclass
class KalmanState:
    """State for a single Kalman filter."""
    x: float           # Current estimate
    P: float           # Estimation uncertainty
    n_observations: int = 0
    history: list = field(default_factory=list)


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

    def predict(self, state: KalmanState) -> KalmanState:
        """Predict step: uncertainty grows, estimate unchanged."""
        return KalmanState(
            x=state.x,
            P=state.P + self.Q,
            n_observations=state.n_observations,
            history=state.history,
        )

    def update(self, state: KalmanState, observation: float,
               game_date: str = "") -> KalmanState:
        """Update step: incorporate new observation."""
        K = state.P / (state.P + self.R)
        x_new = state.x + K * (observation - state.x)
        P_new = (1.0 - K) * state.P

        new_history = state.history + [(game_date, observation, x_new)]

        return KalmanState(
            x=x_new,
            P=P_new,
            n_observations=state.n_observations + 1,
            history=new_history,
        )

    def predict_and_update(self, state: KalmanState, observation: float,
                           game_date: str = "") -> KalmanState:
        """Combined predict + update."""
        predicted = self.predict(state)
        return self.update(predicted, observation, game_date)


# ============================================================================
# Multi-Season Calibration Engine
# ============================================================================

class MultiSeasonKalmanCalibrator:
    """
    Perform grid search for optimal (Q, R) across multiple seasons.
    """

    def __init__(self):
        self.Q_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        self.R_grid = [5, 10, 15, 20, 25, 30, 40, 50]
        self.seasons = [2020, 2021, 2022, 2023, 2024]
        self.burn_in = 5  # Skip first N games per player
        self.min_games = 10

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

    def evaluate_params(self, df: pd.DataFrame, Q: float, R: float) -> Tuple[float, int, Dict]:
        """
        Evaluate (Q, R) parameters on a season's data.
        Returns (MAE, n_predictions, details_dict).
        """
        kf = ScalarKalmanFilter(process_noise=Q, observation_noise=R)

        predictions = []
        errors_by_position = defaultdict(list)

        for player_name, group in df.groupby('player_name'):
            games = group.sort_values('game_date').reset_index(drop=True)

            if len(games) < self.min_games:
                continue

            # Initialize with first N games
            init_fpts = games['dk_fpts'].iloc[:self.burn_in].mean()
            state = kf.initialize(init_fpts)

            # Process initial burn-in games
            for idx in range(self.burn_in):
                obs = games.iloc[idx]['dk_fpts']
                state = kf.predict_and_update(state, obs)

            # Predict remaining games (walk-forward)
            for idx in range(self.burn_in, len(games)):
                actual = games.iloc[idx]['dk_fpts']
                predicted = state.x
                position = games.iloc[idx]['position']

                error = abs(predicted - actual)
                predictions.append(error)
                errors_by_position[position].append(error)

                # Update with observed value
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
        """
        Grid search for optimal (Q, R) on a single season.
        Returns best params and grid results.
        """
        print(f"\n{'='*70}")
        print(f"  SEASON {season} CALIBRATION")
        print(f"{'='*70}")

        df = self.load_season_data(season)
        n_players = df['player_name'].nunique()
        n_rows = len(df)
        print(f"Loaded {n_rows:,} rows from {n_players:,} players")

        results = []
        best_mae = np.inf
        best_params = {}

        for Q in self.Q_grid:
            for R in self.R_grid:
                mae, n_pred, details = self.evaluate_params(df, Q, R)

                results.append({
                    'Q': Q,
                    'R': R,
                    'MAE': mae,
                    'n_predictions': n_pred,
                    'MAE_C': details.get('C', np.nan),
                    'MAE_W': details.get('W', np.nan),
                    'MAE_D': details.get('D', np.nan),
                })

                if mae < best_mae:
                    best_mae = mae
                    best_params = {'Q': Q, 'R': R, 'MAE': mae}

        results_df = pd.DataFrame(results).sort_values('MAE')

        print(f"\nBest for {season}: Q={best_params['Q']}, R={best_params['R']}, MAE={best_params['MAE']:.4f}")
        print(f"\nTop 10 (Q, R) combinations:")
        print(results_df.head(10)[['Q', 'R', 'MAE', 'n_predictions']].to_string(index=False))

        return {
            'season': season,
            'best_params': best_params,
            'all_results': results_df,
            'n_rows': n_rows,
            'n_players': n_players,
        }

    def calibrate_all_seasons(self) -> Dict:
        """
        Calibrate across all seasons and find globally optimal params.
        """
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
        print(f"{'Season':<10} {'Best Q':<10} {'Best R':<10} {'Best MAE':<12} {'Players':<10}")
        print("-" * 52)
        for season in sorted(season_results.keys()):
            sr = season_results[season]
            best = sr['best_params']
            print(f"{season:<10} {best['Q']:<10.2f} {best['R']:<10.1f} {best['MAE']:<12.4f} {sr['n_players']:<10}")

        # Consistency check: how much does optimal Q/R vary by season?
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
# Position-Specific Calibration
# ============================================================================

class PositionSpecificCalibrator:
    """
    Calibrate separate Q/R for Centers, Wings, Defense.
    """

    def __init__(self):
        self.Q_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        self.R_grid = [5, 10, 15, 20, 25, 30, 40, 50]
        self.burn_in = 5
        self.min_games = 10

    def load_all_historical(self) -> pd.DataFrame:
        """Load all historical data for all seasons."""
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
        return df.sort_values(['position', 'player_name', 'game_date'])

    def evaluate_position(self, df_pos: pd.DataFrame, Q: float, R: float) -> Tuple[float, int]:
        """Evaluate params for a single position across all seasons."""
        kf = ScalarKalmanFilter(process_noise=Q, observation_noise=R)
        errors = []

        for player_name, group in df_pos.groupby('player_name'):
            games = group.sort_values('game_date').reset_index(drop=True)

            if len(games) < self.min_games:
                continue

            init_fpts = games['dk_fpts'].iloc[:self.burn_in].mean()
            state = kf.initialize(init_fpts)

            for idx in range(self.burn_in):
                state = kf.predict_and_update(state, games.iloc[idx]['dk_fpts'])

            for idx in range(self.burn_in, len(games)):
                actual = games.iloc[idx]['dk_fpts']
                predicted = state.x
                errors.append(abs(predicted - actual))
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
            print(f"\nCalibrating for {position}...")
            df_pos = df[df['position'] == position]
            n_players = df_pos['player_name'].nunique()
            n_rows = len(df_pos)
            print(f"  {n_rows:,} rows from {n_players:,} players")

            results = []
            best_mae = np.inf
            best_params = {}

            for Q in self.Q_grid:
                for R in self.R_grid:
                    mae, n_pred = self.evaluate_position(df_pos, Q, R)
                    results.append({'Q': Q, 'R': R, 'MAE': mae, 'n_predictions': n_pred})

                    if mae < best_mae:
                        best_mae = mae
                        best_params = {'Q': Q, 'R': R, 'MAE': mae}

            results_df = pd.DataFrame(results).sort_values('MAE')
            results_by_position[position] = {
                'best_params': best_params,
                'all_results': results_df,
                'n_players': n_players,
            }

            print(f"  Best: Q={best_params['Q']}, R={best_params['R']}, MAE={best_params['MAE']:.4f}")

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
    """
    Walk-forward backtest on 2024-25 season using globally-optimized params.
    """

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
        """
        Walk-forward backtest: predict each game before observing it.
        """
        print("\n" + "="*70)
        print(f"  WALK-FORWARD BACKTEST: 2024-25 Season")
        print(f"  Q={self.Q}, R={self.R}")
        print("="*70)

        df = self.load_current_season()
        print(f"Loaded {len(df):,} game logs from {df['player_name'].nunique():,} players")

        predictions = []
        errors_by_position = defaultdict(list)

        for player_name, group in df.groupby('player_name'):
            games = group.sort_values('game_date').reset_index(drop=True)

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

        return results_df


# ============================================================================
# Multi-Stat Kalman Calibration
# ============================================================================

class MultiStatKalmanCalibrator:
    """
    Calibrate separate Kalman filters for goals, assists, shots, blocked_shots.
    Then convert to FPTS via DK scoring.
    """

    def __init__(self):
        self.stats = ['goals', 'assists', 'shots', 'blocked_shots']
        self.burn_in = 5
        self.min_games = 10
        # Simplified grid for multi-stat (fewer combos to avoid explosion)
        self.Q_grid = [0.05, 0.1, 0.2, 0.5, 1.0]
        self.R_grid = [0.1, 0.5, 1.0, 2.0, 5.0]

    def load_all_historical(self) -> pd.DataFrame:
        """Load historical data with individual stats."""
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT
                season, player_name, position, game_date,
                goals, assists, shots, blocked_shots,
                dk_fpts
            FROM historical_skaters
            WHERE position IN ('C', 'LW', 'RW', 'W', 'D')
              AND dk_fpts IS NOT NULL
              AND goals IS NOT NULL
            ORDER BY season, player_name, game_date
        """, conn)
        conn.close()

        df['position'] = df['position'].apply(lambda x: 'W' if x in ['LW', 'RW', 'L', 'R'] else x)
        return df.sort_values(['position', 'player_name', 'game_date'])

    def evaluate_multistat(self, df: pd.DataFrame, stat_params: Dict) -> Tuple[float, int]:
        """
        Evaluate multi-stat Kalman with given parameters.
        stat_params = {'goals': {'Q': ..., 'R': ...}, ...}
        """
        filters = {
            stat: ScalarKalmanFilter(
                process_noise=stat_params[stat]['Q'],
                observation_noise=stat_params[stat]['R']
            )
            for stat in self.stats
        }

        errors = []

        for player_name, group in df.groupby('player_name'):
            games = group.sort_values('game_date').reset_index(drop=True)

            if len(games) < self.min_games:
                continue

            # Initialize separate states for each stat
            states = {}
            for stat in self.stats:
                init_val = games[stat].iloc[:self.burn_in].mean()
                states[stat] = filters[stat].initialize(init_val)

            # Burn-in
            for idx in range(self.burn_in):
                for stat in self.stats:
                    obs = games.iloc[idx][stat]
                    states[stat] = filters[stat].predict_and_update(states[stat], obs)

            # Walk-forward: predict FPTS from stat rates
            for idx in range(self.burn_in, len(games)):
                actual_fpts = games.iloc[idx]['dk_fpts']

                # Predict FPTS from current stat estimates
                predicted_fpts = 0.0
                for stat in self.stats:
                    rate = max(0.0, states[stat].x)
                    predicted_fpts += rate * DK_SCORING[stat]

                error = abs(predicted_fpts - actual_fpts)
                errors.append(error)

                # Update states with observed stats
                for stat in self.stats:
                    obs = games.iloc[idx][stat]
                    states[stat] = filters[stat].predict_and_update(states[stat], obs)

        if not errors:
            return np.inf, 0
        return np.mean(errors), len(errors)

    def calibrate_multistat(self) -> Dict:
        """
        Grid search for optimal multi-stat parameters.
        Test all combinations of (Q_g, R_g, Q_a, R_a, Q_s, R_s, Q_b, R_b).
        """
        print("\n" + "="*70)
        print("  MULTI-STAT KALMAN CALIBRATION")
        print("="*70)

        df = self.load_all_historical()
        n_players = df['player_name'].nunique()
        n_rows = len(df)
        print(f"Loaded {n_rows:,} rows from {n_players:,} players")

        results = []
        best_mae = np.inf
        best_params = {}

        # Simplified: just test a few key combinations instead of full factorial
        for q_all in self.Q_grid:
            for r_all in self.R_grid:
                stat_params = {
                    'goals': {'Q': q_all, 'R': r_all},
                    'assists': {'Q': q_all, 'R': r_all},
                    'shots': {'Q': q_all, 'R': r_all},
                    'blocked_shots': {'Q': q_all, 'R': r_all},
                }

                mae, n_pred = self.evaluate_multistat(df, stat_params)
                results.append({
                    'Q_all': q_all,
                    'R_all': r_all,
                    'MAE': mae,
                    'n_predictions': n_pred,
                })

                if mae < best_mae:
                    best_mae = mae
                    best_params = {'Q_all': q_all, 'R_all': r_all, 'MAE': mae}

        results_df = pd.DataFrame(results).sort_values('MAE')

        print(f"\nBest: Q={best_params['Q_all']}, R={best_params['R_all']}, MAE={best_params['MAE']:.4f}")
        print(f"\nTop 10:")
        print(results_df.head(10)[['Q_all', 'R_all', 'MAE', 'n_predictions']].to_string(index=False))

        return {
            'best_params': best_params,
            'all_results': results_df,
        }


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
    multi_cal = MultiSeasonKalmanCalibrator()
    multi_results = multi_cal.calibrate_all_seasons()

    # 2. Position-specific calibration
    print("\n" + "#" * 70)
    print("# STAGE 2: POSITION-SPECIFIC CALIBRATION")
    print("#" * 70)
    pos_cal = PositionSpecificCalibrator()
    pos_results = pos_cal.calibrate_positions()

    # 3. Multi-stat Kalman calibration
    print("\n" + "#" * 70)
    print("# STAGE 3: MULTI-STAT KALMAN CALIBRATION")
    print("#" * 70)
    multistat_cal = MultiStatKalmanCalibrator()
    multistat_results = multistat_cal.calibrate_multistat()

    # 4. Walk-forward backtest on current season
    print("\n" + "#" * 70)
    print("# STAGE 4: WALK-FORWARD BACKTEST ON 2024-25 SEASON")
    print("#" * 70)
    global_params = multi_results['best_global']
    current_backtest = CurrentSeasonBacktest(
        Q=global_params['Q'],
        R=global_params['R']
    )
    backtest_results = current_backtest.backtest()

    # 5. Summary report
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

    print("\n3. MULTI-STAT KALMAN PARAMETERS")
    best_multistat = multistat_results['best_params']
    print(f"   Q={best_multistat['Q_all']:.2f}, R={best_multistat['R_all']:.2f}")
    print(f"   MAE = {best_multistat['MAE']:.4f}")

    print("\n4. 2024-25 SEASON WALK-FORWARD RESULTS")
    if not backtest_results.empty:
        print(f"   MAE = {backtest_results['error'].mean():.4f}")
        print(f"   RMSE = {np.sqrt((backtest_results['error']**2).mean()):.4f}")
        print(f"   Correlation = {backtest_results['actual_fpts'].corr(backtest_results['predicted_fpts']):.4f}")
        print(f"   Improvement vs single-season (4.318): {4.318 - backtest_results['error'].mean():.4f}")

    print("\n5. CONSISTENCY CHECK (are params stable across seasons?)")
    season_results = multi_results['season_results']
    per_season_best = [season_results[s]['best_params'] for s in sorted(season_results.keys())]
    qs = [b['Q'] for b in per_season_best]
    rs = [b['R'] for b in per_season_best]
    print(f"   Q consistency: std={np.std(qs):.3f} (range {min(qs):.2f}-{max(qs):.2f})")
    print(f"   R consistency: std={np.std(rs):.3f} (range {min(rs):.1f}-{max(rs):.1f})")
    if np.std(qs) < 0.3 and np.std(rs) < 10:
        print("   VERDICT: Parameters are highly consistent across seasons!")
    else:
        print("   VERDICT: Parameters vary significantly; consider position-specific tuning")

    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70 + "\n")

    return {
        'multi_season': multi_results,
        'position_specific': pos_results,
        'multistat': multistat_results,
        'backtest_current': backtest_results,
    }


if __name__ == "__main__":
    results = main()
