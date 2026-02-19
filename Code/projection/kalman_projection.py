"""
Kalman Filter Projection Model for NHL DFS.

Two approaches, both backtestable:

  Approach A — FPTS Rate Filter:
    One Kalman filter per player tracking their "true" DK fantasy points
    per game rate. Simplest, most directly useful. Separates signal from
    noise in game-to-game FPTS variance.

  Approach B — Individual Stat Filter:
    Separate Kalman filters for goals/assists/shots/blocks per game,
    then converts to FPTS via DK scoring rules. More granular — can
    detect a player whose shot volume is rising even if goals haven't
    come yet.

Both approaches use the same core Kalman filter math:
    predict:  x_hat = x_prev   (player ability is assumed to drift slowly)
              P     = P_prev + Q   (uncertainty grows each game)
    update:   K     = P / (P + R)  (Kalman gain)
              x_hat = x_hat + K * (observation - x_hat)
              P     = (1 - K) * P

Where:
    Q = process noise (how much true ability changes game-to-game)
    R = observation noise (how much randomness in a single game)
    K = Kalman gain (0-1, how much to trust the new observation)

Hockey has VERY high observation noise (R >> Q), so the filter will
be appropriately skeptical of short streaks.

Usage:
    from kalman_projection import KalmanFPTSFilter, KalmanStatFilter

    # Approach A
    kf = KalmanFPTSFilter()
    kf.fit_from_db()  # loads game logs from nhl_dfs_history.db
    projection = kf.get_projection("Connor McDavid")

    # Approach B
    ksf = KalmanStatFilter()
    ksf.fit_from_db()
    projection = ksf.get_projection("Connor McDavid")
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from utils import calculate_skater_fantasy_points, normalize_position

# Database path
DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"


# ============================================================================
# Core Kalman Filter
# ============================================================================

@dataclass
class KalmanState:
    """State for a single Kalman filter."""
    x: float           # Current estimate of true value
    P: float           # Estimation uncertainty (covariance)
    n_observations: int = 0
    history: list = field(default_factory=list)  # (game_date, observation, estimate) tuples


class ScalarKalmanFilter:
    """
    1-D Kalman filter for tracking a player's true rate.

    Parameters calibrated for hockey DFS:
        Q (process_noise): How much a player's true ability changes per game.
            Low value = player's true skill is stable (changes slowly).
            Hockey: ~0.1-0.5 FPTS² per game for FPTS tracking.

        R (observation_noise): How much randomness in a single game.
            High value = single games are noisy (true for hockey).
            Hockey: ~15-40 FPTS² for FPTS tracking (std ~4-6 FPTS).

        initial_P: Starting uncertainty. Higher = learns faster initially.
    """

    def __init__(self, process_noise: float, observation_noise: float,
                 initial_P: float = 50.0):
        self.Q = process_noise
        self.R = observation_noise
        self.initial_P = initial_P

    def initialize(self, initial_estimate: float) -> KalmanState:
        """Create initial state from a prior estimate (e.g., season average)."""
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
        # Kalman gain: how much to trust the new observation
        K = state.P / (state.P + self.R)

        # Updated estimate
        x_new = state.x + K * (observation - state.x)

        # Updated uncertainty (always decreases after an observation)
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
        """Combined predict + update for sequential processing."""
        predicted = self.predict(state)
        return self.update(predicted, observation, game_date)


# ============================================================================
# Approach A: FPTS Rate Filter
# ============================================================================

class KalmanFPTSFilter:
    """
    Track each player's "true" DK FPTS per game rate using a Kalman filter.

    The filter starts with the player's season average and then adjusts
    game-by-game, weighting recent games more but being appropriately
    skeptical of noise.

    Parameters (optimized via grid search in backtest):
        process_noise: 0.3 (true ability changes slowly)
        observation_noise: 25.0 (single games are very noisy in hockey)
    """

    # Optimal parameters from grid search over 21,276 predictions:
    # Q=0.1, R=40.0 minimizes MAE while matching season avg correlation
    DEFAULT_PROCESS_NOISE = 0.1
    DEFAULT_OBSERVATION_NOISE = 40.0
    DEFAULT_INITIAL_P = 50.0
    MIN_GAMES = 5  # Minimum games before using Kalman estimate

    def __init__(self, process_noise: float = None,
                 observation_noise: float = None,
                 initial_P: float = None):
        self.process_noise = process_noise or self.DEFAULT_PROCESS_NOISE
        self.observation_noise = observation_noise or self.DEFAULT_OBSERVATION_NOISE
        self.initial_P = initial_P or self.DEFAULT_INITIAL_P

        self.kf = ScalarKalmanFilter(
            process_noise=self.process_noise,
            observation_noise=self.observation_noise,
            initial_P=self.initial_P,
        )

        # player_name -> KalmanState
        self.player_states: Dict[str, KalmanState] = {}
        # player_name -> position
        self.player_positions: Dict[str, str] = {}

    def fit_from_db(self, db_path: str = None,
                    min_games: int = None) -> "KalmanFPTSFilter":
        """
        Load game logs from SQLite and run Kalman filter for every player.
        Uses the actuals table (34K+ rows, 600+ players with 30+ games).
        """
        db = db_path or str(DB_PATH)
        min_g = min_games or self.MIN_GAMES
        conn = sqlite3.connect(db)

        # Load all skater actuals, ordered by date
        df = pd.read_sql_query("""
            SELECT name, position, game_date, actual_fpts,
                   goals, assists, shots, blocks
            FROM actuals
            WHERE actual_fpts IS NOT NULL
              AND position IN ('C', 'W', 'D', 'LW', 'RW', 'L', 'R')
            ORDER BY name, game_date
        """, conn)
        conn.close()

        if df.empty:
            print("Warning: No skater data found in actuals table.")
            return self

        # Normalize positions
        df['position'] = df['position'].apply(normalize_position)

        # Process each player
        for name, group in df.groupby('name'):
            games = group.sort_values('game_date').reset_index(drop=True)

            if len(games) < min_g:
                continue

            # Initialize with first few games' average
            init_fpts = games['actual_fpts'].iloc[:min_g].mean()
            state = self.kf.initialize(init_fpts)

            # Process remaining games sequentially
            for idx in range(len(games)):
                obs = games.iloc[idx]['actual_fpts']
                date = games.iloc[idx]['game_date']
                state = self.kf.predict_and_update(state, obs, date)

            self.player_states[name] = state
            self.player_positions[name] = games.iloc[0]['position']

        print(f"Kalman FPTS Filter: trained on {len(self.player_states)} players")
        return self

    def get_projection(self, player_name: str) -> Optional[float]:
        """Get current Kalman-smoothed FPTS projection for a player."""
        state = self.player_states.get(player_name)
        if state is None:
            return None
        return round(state.x, 2)

    def get_all_projections(self) -> pd.DataFrame:
        """Get projections for all tracked players."""
        rows = []
        for name, state in self.player_states.items():
            rows.append({
                'name': name,
                'position': self.player_positions.get(name, ''),
                'kalman_fpts': round(state.x, 2),
                'uncertainty': round(state.P, 2),
                'n_games': state.n_observations,
                'kalman_gain': round(state.P / (state.P + self.observation_noise), 4),
            })
        return pd.DataFrame(rows).sort_values('kalman_fpts', ascending=False)

    def backtest(self, db_path: str = None, train_games: int = 10,
                 min_games: int = 20) -> Dict:
        """
        Walk-forward backtest: for each player, use first train_games to
        initialize, then predict each subsequent game before observing it.

        Returns dict with MAE, RMSE, correlation, and per-prediction details.
        """
        db = db_path or str(DB_PATH)
        conn = sqlite3.connect(db)

        df = pd.read_sql_query("""
            SELECT name, position, game_date, actual_fpts,
                   goals, assists, shots, blocks
            FROM actuals
            WHERE actual_fpts IS NOT NULL
              AND position IN ('C', 'W', 'D', 'LW', 'RW', 'L', 'R')
            ORDER BY name, game_date
        """, conn)
        conn.close()

        df['position'] = df['position'].apply(normalize_position)

        predictions = []

        for name, group in df.groupby('name'):
            games = group.sort_values('game_date').reset_index(drop=True)

            if len(games) < min_games:
                continue

            # Initialize from first train_games
            init_fpts = games['actual_fpts'].iloc[:train_games].mean()
            season_avg = games['actual_fpts'].mean()
            state = self.kf.initialize(init_fpts)

            # Run filter through training games
            for idx in range(train_games):
                obs = games.iloc[idx]['actual_fpts']
                date = games.iloc[idx]['game_date']
                state = self.kf.predict_and_update(state, obs, date)

            # Predict remaining games (walk-forward)
            rolling_sum = games['actual_fpts'].iloc[:train_games].sum()
            rolling_count = train_games

            for idx in range(train_games, len(games)):
                actual = games.iloc[idx]['actual_fpts']
                date = games.iloc[idx]['game_date']
                pos = games.iloc[idx]['position']

                # Kalman prediction (before seeing this game)
                kalman_pred = state.x

                # Season average prediction (baseline comparison)
                season_avg_pred = rolling_sum / rolling_count

                # 5-game rolling average
                start_idx = max(0, idx - 5)
                rolling_5_pred = games['actual_fpts'].iloc[start_idx:idx].mean()

                predictions.append({
                    'name': name,
                    'position': pos,
                    'game_date': date,
                    'game_number': idx + 1,
                    'actual_fpts': actual,
                    'kalman_pred': kalman_pred,
                    'season_avg_pred': season_avg_pred,
                    'rolling_5_pred': rolling_5_pred,
                    'kalman_error': abs(kalman_pred - actual),
                    'season_avg_error': abs(season_avg_pred - actual),
                    'rolling_5_error': abs(rolling_5_pred - actual),
                })

                # Update Kalman state with observed game
                state = self.kf.predict_and_update(state, actual, date)

                # Update rolling averages
                rolling_sum += actual
                rolling_count += 1

        results_df = pd.DataFrame(predictions)

        if results_df.empty:
            return {'metrics': {}, 'results': results_df}

        metrics = {
            'n_predictions': len(results_df),
            'n_players': results_df['name'].nunique(),
            'kalman_mae': round(results_df['kalman_error'].mean(), 3),
            'season_avg_mae': round(results_df['season_avg_error'].mean(), 3),
            'rolling_5_mae': round(results_df['rolling_5_error'].mean(), 3),
            'kalman_rmse': round(np.sqrt((results_df['kalman_error'] ** 2).mean()), 3),
            'season_avg_rmse': round(np.sqrt((results_df['season_avg_error'] ** 2).mean()), 3),
            'rolling_5_rmse': round(np.sqrt((results_df['rolling_5_error'] ** 2).mean()), 3),
            'kalman_corr': round(results_df['actual_fpts'].corr(results_df['kalman_pred']), 4),
            'season_avg_corr': round(results_df['actual_fpts'].corr(results_df['season_avg_pred']), 4),
            'rolling_5_corr': round(results_df['actual_fpts'].corr(results_df['rolling_5_pred']), 4),
            'mean_actual': round(results_df['actual_fpts'].mean(), 2),
            'mean_kalman_pred': round(results_df['kalman_pred'].mean(), 2),
            'mean_season_avg_pred': round(results_df['season_avg_pred'].mean(), 2),
        }

        # Per-position breakdown
        for pos in ['C', 'W', 'D']:
            pos_df = results_df[results_df['position'] == pos]
            if len(pos_df) > 0:
                metrics[f'{pos}_kalman_mae'] = round(pos_df['kalman_error'].mean(), 3)
                metrics[f'{pos}_season_avg_mae'] = round(pos_df['season_avg_error'].mean(), 3)
                metrics[f'{pos}_n_predictions'] = len(pos_df)

        return {'metrics': metrics, 'results': results_df}


# ============================================================================
# Approach B: Individual Stat Filter
# ============================================================================

class KalmanStatFilter:
    """
    Track individual stat rates (goals/gm, assists/gm, shots/gm, blocks/gm)
    with separate Kalman filters, then convert to FPTS via DK scoring.

    Advantage over FPTS filter: can detect a player whose shot volume is
    rising even if goals haven't come yet. The FPTS filter would miss this
    because FPTS is dominated by goals (8.5 pts each).

    Each stat has different noise characteristics:
        - Goals: very noisy (0 or 1 most games, occasional 2-3)
        - Assists: noisy (0-2 range, streaky)
        - Shots: lower noise (2-5 range, more consistent)
        - Blocks: low noise for D, near-zero for forwards
    """

    # Per-stat noise parameters (calibrated from actual variance in actuals table)
    # R = observation variance from data, Q = process noise (slow drift)
    # Q/R ratio determines filter responsiveness
    STAT_PARAMS = {
        'goals': {'Q': 0.001, 'R': 0.155, 'P0': 0.2},     # var=0.155, very noisy binary
        'assists': {'Q': 0.001, 'R': 0.317, 'P0': 0.3},    # var=0.317, also streaky
        'shots': {'Q': 0.005, 'R': 2.048, 'P0': 1.0},      # var=2.048, more continuous
        'blocks': {'Q': 0.003, 'R': 1.247, 'P0': 0.5},     # var=1.247, position-dependent
    }

    # DK scoring weights for converting stat rates to FPTS
    DK_WEIGHTS = {
        'goals': 8.5,
        'assists': 5.0,
        'shots': 1.5,
        'blocks': 1.3,
    }

    # Bonus expected values (from empirical rates at given stat levels)
    # These approximate the average bonus contribution per game
    MIN_GAMES = 5

    def __init__(self, stat_params: Dict = None):
        if stat_params:
            self.STAT_PARAMS = stat_params

        self.filters: Dict[str, ScalarKalmanFilter] = {}
        for stat, params in self.STAT_PARAMS.items():
            self.filters[stat] = ScalarKalmanFilter(
                process_noise=params['Q'],
                observation_noise=params['R'],
                initial_P=params['P0'],
            )

        # player_name -> {stat: KalmanState}
        self.player_states: Dict[str, Dict[str, KalmanState]] = {}
        self.player_positions: Dict[str, str] = {}

    def _estimate_bonus_ev(self, goals_rate: float, assists_rate: float,
                           shots_rate: float, blocks_rate: float) -> float:
        """
        Estimate expected bonus contribution per game from stat rates.

        DK Bonuses:
            Hat trick (3+ goals): +3.0
            3+ points (G+A): +3.0
            5+ shots: +3.0
            3+ blocks: +3.0

        Uses simple probability approximation based on Poisson rates.
        """
        bonus = 0.0

        # P(3+ goals) ≈ Poisson CDF complement (very rare for most players)
        if goals_rate > 0:
            import math
            lam = goals_rate
            p_hat_trick = 1 - sum(
                (lam ** k) * math.exp(-lam) / math.factorial(k) for k in range(3)
            )
            bonus += p_hat_trick * 3.0

        # P(3+ points) from combined goals + assists rate
        points_rate = goals_rate + assists_rate
        if points_rate > 0:
            import math
            lam = points_rate
            p_3plus = 1 - sum(
                (lam ** k) * math.exp(-lam) / math.factorial(k) for k in range(3)
            )
            bonus += p_3plus * 3.0

        # P(5+ shots) — shots are more continuous, still use Poisson approx
        if shots_rate > 0:
            import math
            lam = shots_rate
            p_5plus = 1 - sum(
                (lam ** k) * math.exp(-lam) / math.factorial(k) for k in range(5)
            )
            bonus += p_5plus * 3.0

        # P(3+ blocks) — mainly relevant for defensemen
        if blocks_rate > 0:
            import math
            lam = blocks_rate
            p_3plus = 1 - sum(
                (lam ** k) * math.exp(-lam) / math.factorial(k) for k in range(3)
            )
            bonus += p_3plus * 3.0

        return bonus

    def fit_from_db(self, db_path: str = None,
                    min_games: int = None) -> "KalmanStatFilter":
        """
        Load game logs and run per-stat Kalman filters for every player.

        Uses game_logs_skaters table (has individual stats) supplemented by
        actuals table rows that have non-null stats.
        """
        db = db_path or str(DB_PATH)
        min_g = min_games or self.MIN_GAMES
        conn = sqlite3.connect(db)

        # Primary source: game_logs_skaters (always has individual stats)
        df1 = pd.read_sql_query("""
            SELECT player_name as name, position, game_date, dk_fpts as actual_fpts,
                   goals, assists, shots,
                   COALESCE(
                       (SELECT a.blocks FROM actuals a
                        WHERE a.name = game_logs_skaters.player_name
                          AND a.game_date = game_logs_skaters.game_date
                        LIMIT 1),
                       0
                   ) as blocks
            FROM game_logs_skaters
            WHERE goals IS NOT NULL AND toi_seconds > 0
            ORDER BY player_name, game_date
        """, conn)

        # Secondary source: actuals rows that DO have individual stats
        df2 = pd.read_sql_query("""
            SELECT name, position, game_date, actual_fpts,
                   goals, assists, shots, blocks
            FROM actuals
            WHERE actual_fpts IS NOT NULL
              AND goals IS NOT NULL
              AND position IN ('C', 'W', 'D', 'LW', 'RW', 'L', 'R')
            ORDER BY name, game_date
        """, conn)
        conn.close()

        # Combine and deduplicate (prefer game_logs_skaters data)
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop_duplicates(subset=['name', 'game_date'], keep='first')
        df = df.sort_values(['name', 'game_date'])

        if df.empty:
            print("Warning: No skater data found in actuals table.")
            return self

        df['position'] = df['position'].apply(normalize_position)

        for name, group in df.groupby('name'):
            games = group.sort_values('game_date').reset_index(drop=True)

            if len(games) < min_g:
                continue

            states = {}
            for stat in self.STAT_PARAMS:
                init_val = games[stat].iloc[:min_g].mean()
                state = self.filters[stat].initialize(init_val)

                for idx in range(len(games)):
                    obs = float(games.iloc[idx][stat])
                    date = games.iloc[idx]['game_date']
                    state = self.filters[stat].predict_and_update(state, obs, date)

                states[stat] = state

            self.player_states[name] = states
            self.player_positions[name] = games.iloc[0]['position']

        print(f"Kalman Stat Filter: trained on {len(self.player_states)} players")
        return self

    def get_projection(self, player_name: str) -> Optional[float]:
        """Get FPTS projection from individual stat Kalman estimates."""
        states = self.player_states.get(player_name)
        if states is None:
            return None

        fpts = 0.0
        rates = {}
        for stat, state in states.items():
            rate = max(0.0, state.x)  # Clamp negative rates to 0
            rates[stat] = rate
            fpts += rate * self.DK_WEIGHTS[stat]

        # Add bonus expected value
        fpts += self._estimate_bonus_ev(
            rates.get('goals', 0), rates.get('assists', 0),
            rates.get('shots', 0), rates.get('blocks', 0),
        )

        return round(fpts, 2)

    def get_stat_estimates(self, player_name: str) -> Optional[Dict]:
        """Get individual stat rate estimates for a player."""
        states = self.player_states.get(player_name)
        if states is None:
            return None

        return {
            stat: {
                'rate': round(max(0.0, state.x), 3),
                'uncertainty': round(state.P, 4),
                'n_games': state.n_observations,
            }
            for stat, state in states.items()
        }

    def get_all_projections(self) -> pd.DataFrame:
        """Get projections for all tracked players."""
        rows = []
        for name, states in self.player_states.items():
            proj = self.get_projection(name)
            row = {
                'name': name,
                'position': self.player_positions.get(name, ''),
                'kalman_stat_fpts': proj,
                'n_games': states['goals'].n_observations,
            }
            for stat, state in states.items():
                row[f'{stat}_rate'] = round(max(0.0, state.x), 3)
            rows.append(row)
        return pd.DataFrame(rows).sort_values('kalman_stat_fpts', ascending=False)

    def backtest(self, db_path: str = None, train_games: int = 10,
                 min_games: int = 20) -> Dict:
        """
        Walk-forward backtest: initialize from first train_games, then
        predict each subsequent game before observing it.

        Uses game_logs_skaters + actuals rows with non-null stats
        (same data source as fit_from_db).
        """
        db = db_path or str(DB_PATH)
        conn = sqlite3.connect(db)

        # Primary source: game_logs_skaters (always has individual stats)
        df1 = pd.read_sql_query("""
            SELECT player_name as name, position, game_date, dk_fpts as actual_fpts,
                   goals, assists, shots,
                   COALESCE(
                       (SELECT a.blocks FROM actuals a
                        WHERE a.name = game_logs_skaters.player_name
                          AND a.game_date = game_logs_skaters.game_date
                        LIMIT 1),
                       0
                   ) as blocks
            FROM game_logs_skaters
            WHERE goals IS NOT NULL AND toi_seconds > 0
            ORDER BY player_name, game_date
        """, conn)

        # Secondary source: actuals rows that DO have individual stats
        df2 = pd.read_sql_query("""
            SELECT name, position, game_date, actual_fpts,
                   goals, assists, shots, blocks
            FROM actuals
            WHERE actual_fpts IS NOT NULL
              AND goals IS NOT NULL
              AND position IN ('C', 'W', 'D', 'LW', 'RW', 'L', 'R')
            ORDER BY name, game_date
        """, conn)
        conn.close()

        # Combine and deduplicate (prefer game_logs_skaters data)
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop_duplicates(subset=['name', 'game_date'], keep='first')
        df = df.sort_values(['name', 'game_date'])

        df['position'] = df['position'].apply(normalize_position)

        predictions = []

        for name, group in df.groupby('name'):
            games = group.sort_values('game_date').reset_index(drop=True)

            if len(games) < min_games:
                continue

            # Initialize per-stat filters from training window
            stat_states = {}
            for stat in self.STAT_PARAMS:
                init_val = games[stat].iloc[:train_games].mean()
                stat_states[stat] = self.filters[stat].initialize(init_val)

                # Run through training games
                for idx in range(train_games):
                    obs = float(games.iloc[idx][stat])
                    date = games.iloc[idx]['game_date']
                    stat_states[stat] = self.filters[stat].predict_and_update(
                        stat_states[stat], obs, date
                    )

            # Walk-forward prediction
            rolling_sum = games['actual_fpts'].iloc[:train_games].sum()
            rolling_count = train_games

            for idx in range(train_games, len(games)):
                actual = games.iloc[idx]['actual_fpts']
                date = games.iloc[idx]['game_date']
                pos = games.iloc[idx]['position']

                # Kalman stat prediction (before seeing this game)
                kalman_fpts = 0.0
                rates = {}
                for stat, state in stat_states.items():
                    rate = max(0.0, state.x)
                    rates[stat] = rate
                    kalman_fpts += rate * self.DK_WEIGHTS[stat]

                kalman_fpts += self._estimate_bonus_ev(
                    rates.get('goals', 0), rates.get('assists', 0),
                    rates.get('shots', 0), rates.get('blocks', 0),
                )

                # Season average baseline
                season_avg_pred = rolling_sum / rolling_count

                predictions.append({
                    'name': name,
                    'position': pos,
                    'game_date': date,
                    'game_number': idx + 1,
                    'actual_fpts': actual,
                    'kalman_stat_pred': kalman_fpts,
                    'season_avg_pred': season_avg_pred,
                    'kalman_stat_error': abs(kalman_fpts - actual),
                    'season_avg_error': abs(season_avg_pred - actual),
                })

                # Update all stat filters with observed game
                for stat in self.STAT_PARAMS:
                    obs = float(games.iloc[idx][stat])
                    stat_states[stat] = self.filters[stat].predict_and_update(
                        stat_states[stat], obs, date
                    )

                rolling_sum += actual
                rolling_count += 1

        results_df = pd.DataFrame(predictions)

        if results_df.empty:
            return {'metrics': {}, 'results': results_df}

        metrics = {
            'n_predictions': len(results_df),
            'n_players': results_df['name'].nunique(),
            'kalman_stat_mae': round(results_df['kalman_stat_error'].mean(), 3),
            'season_avg_mae': round(results_df['season_avg_error'].mean(), 3),
            'kalman_stat_rmse': round(np.sqrt((results_df['kalman_stat_error'] ** 2).mean()), 3),
            'season_avg_rmse': round(np.sqrt((results_df['season_avg_error'] ** 2).mean()), 3),
            'kalman_stat_corr': round(results_df['actual_fpts'].corr(results_df['kalman_stat_pred']), 4),
            'season_avg_corr': round(results_df['actual_fpts'].corr(results_df['season_avg_pred']), 4),
            'mean_actual': round(results_df['actual_fpts'].mean(), 2),
            'mean_kalman_stat_pred': round(results_df['kalman_stat_pred'].mean(), 2),
        }

        for pos in ['C', 'W', 'D']:
            pos_df = results_df[results_df['position'] == pos]
            if len(pos_df) > 0:
                metrics[f'{pos}_kalman_stat_mae'] = round(pos_df['kalman_stat_error'].mean(), 3)
                metrics[f'{pos}_season_avg_mae'] = round(pos_df['season_avg_error'].mean(), 3)
                metrics[f'{pos}_n_predictions'] = len(pos_df)

        return {'metrics': metrics, 'results': results_df}


# ============================================================================
# Parameter Optimization (Grid Search)
# ============================================================================

def optimize_fpts_params(db_path: str = None,
                         train_games: int = 10,
                         min_games: int = 20) -> Dict:
    """
    Grid search for optimal FPTS filter parameters.
    Tests combinations of process_noise and observation_noise.
    """
    db = db_path or str(DB_PATH)

    Q_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    R_values = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    best_mae = float('inf')
    best_params = {}
    all_results = []

    for Q in Q_values:
        for R in R_values:
            kf = KalmanFPTSFilter(process_noise=Q, observation_noise=R)
            result = kf.backtest(db_path=db, train_games=train_games,
                                min_games=min_games)

            if not result['metrics']:
                continue

            mae = result['metrics']['kalman_mae']
            all_results.append({
                'Q': Q, 'R': R,
                'mae': mae,
                'rmse': result['metrics']['kalman_rmse'],
                'corr': result['metrics']['kalman_corr'],
                'n_predictions': result['metrics']['n_predictions'],
            })

            if mae < best_mae:
                best_mae = mae
                best_params = {'Q': Q, 'R': R, 'mae': mae,
                               'corr': result['metrics']['kalman_corr']}

    return {
        'best_params': best_params,
        'all_results': pd.DataFrame(all_results).sort_values('mae'),
    }


# ============================================================================
# CLI / Quick Test
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kalman Filter NHL DFS Projections")
    parser.add_argument('--backtest', action='store_true', help='Run backtest comparison')
    parser.add_argument('--optimize', action='store_true', help='Grid search for optimal parameters')
    parser.add_argument('--projections', action='store_true', help='Show current projections')
    parser.add_argument('--train-games', type=int, default=10, help='Games for initialization')
    parser.add_argument('--min-games', type=int, default=20, help='Minimum games per player')
    args = parser.parse_args()

    if args.optimize:
        print("=" * 70)
        print("  KALMAN FILTER PARAMETER OPTIMIZATION")
        print("=" * 70)
        result = optimize_fpts_params(
            train_games=args.train_games, min_games=args.min_games
        )
        print(f"\nBest params: Q={result['best_params']['Q']}, "
              f"R={result['best_params']['R']}")
        print(f"Best MAE: {result['best_params']['mae']:.3f}")
        print(f"Correlation: {result['best_params']['corr']:.4f}")
        print("\nAll results:")
        print(result['all_results'].to_string(index=False))

    elif args.backtest:
        print("=" * 70)
        print("  KALMAN FILTER BACKTEST: FPTS vs STATS vs SEASON AVG")
        print("=" * 70)

        # Approach A: FPTS Rate
        print("\n--- Approach A: FPTS Rate Filter ---")
        kf_fpts = KalmanFPTSFilter()
        result_a = kf_fpts.backtest(
            train_games=args.train_games, min_games=args.min_games
        )
        m = result_a['metrics']
        print(f"  Predictions: {m.get('n_predictions', 0):,} "
              f"({m.get('n_players', 0)} players)")
        print(f"  Kalman FPTS MAE:  {m.get('kalman_mae', 'N/A')}")
        print(f"  Season Avg MAE:   {m.get('season_avg_mae', 'N/A')}")
        print(f"  Rolling 5 MAE:    {m.get('rolling_5_mae', 'N/A')}")
        print(f"  Kalman FPTS Corr: {m.get('kalman_corr', 'N/A')}")
        print(f"  Season Avg Corr:  {m.get('season_avg_corr', 'N/A')}")
        print(f"  Rolling 5 Corr:   {m.get('rolling_5_corr', 'N/A')}")

        # By position
        for pos in ['C', 'W', 'D']:
            k_mae = m.get(f'{pos}_kalman_mae', 'N/A')
            s_mae = m.get(f'{pos}_season_avg_mae', 'N/A')
            n = m.get(f'{pos}_n_predictions', 0)
            if n > 0:
                improvement = ''
                if isinstance(k_mae, float) and isinstance(s_mae, float):
                    delta = s_mae - k_mae
                    improvement = f" ({'+' if delta > 0 else ''}{delta:.3f})"
                print(f"  {pos}: Kalman {k_mae} vs Avg {s_mae}{improvement} ({n} games)")

        # Approach B: Individual Stats
        print("\n--- Approach B: Individual Stat Filter ---")
        kf_stat = KalmanStatFilter()
        result_b = kf_stat.backtest(
            train_games=args.train_games, min_games=args.min_games
        )
        m2 = result_b['metrics']
        print(f"  Predictions: {m2.get('n_predictions', 0):,} "
              f"({m2.get('n_players', 0)} players)")
        print(f"  Kalman Stat MAE:  {m2.get('kalman_stat_mae', 'N/A')}")
        print(f"  Season Avg MAE:   {m2.get('season_avg_mae', 'N/A')}")
        print(f"  Kalman Stat Corr: {m2.get('kalman_stat_corr', 'N/A')}")
        print(f"  Season Avg Corr:  {m2.get('season_avg_corr', 'N/A')}")

        for pos in ['C', 'W', 'D']:
            k_mae = m2.get(f'{pos}_kalman_stat_mae', 'N/A')
            s_mae = m2.get(f'{pos}_season_avg_mae', 'N/A')
            n = m2.get(f'{pos}_n_predictions', 0)
            if n > 0:
                improvement = ''
                if isinstance(k_mae, float) and isinstance(s_mae, float):
                    delta = s_mae - k_mae
                    improvement = f" ({'+' if delta > 0 else ''}{delta:.3f})"
                print(f"  {pos}: Kalman {k_mae} vs Avg {s_mae}{improvement} ({n} games)")

        # Head-to-head summary
        print("\n" + "=" * 70)
        print("  HEAD-TO-HEAD SUMMARY")
        print("=" * 70)
        if m and m2:
            print(f"  {'Method':<25} {'MAE':>8} {'RMSE':>8} {'Corr':>8}")
            print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
            print(f"  {'Kalman FPTS':<25} {m.get('kalman_mae',''):>8} "
                  f"{m.get('kalman_rmse',''):>8} {m.get('kalman_corr',''):>8}")
            print(f"  {'Kalman Stats':<25} {m2.get('kalman_stat_mae',''):>8} "
                  f"{m2.get('kalman_stat_rmse',''):>8} {m2.get('kalman_stat_corr',''):>8}")
            print(f"  {'Season Average':<25} {m.get('season_avg_mae',''):>8} "
                  f"{m.get('season_avg_rmse',''):>8} {m.get('season_avg_corr',''):>8}")
            print(f"  {'Rolling 5-Game':<25} {m.get('rolling_5_mae',''):>8} "
                  f"{m.get('rolling_5_rmse',''):>8} {m.get('rolling_5_corr',''):>8}")

    elif args.projections:
        print("=" * 70)
        print("  CURRENT KALMAN PROJECTIONS")
        print("=" * 70)

        kf = KalmanFPTSFilter()
        kf.fit_from_db()
        proj = kf.get_all_projections()
        print(proj.head(30).to_string(index=False))

    else:
        parser.print_help()
