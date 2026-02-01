"""
Backtesting module for NHL DFS projections.

Tests projection accuracy against historical game results.
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from nhl_api import NHLAPIClient
from config import (
    CURRENT_SEASON, SKATER_SCORING, SKATER_BONUSES,
    GOALIE_SCORING, GOALIE_BONUSES, BACKTESTS_DIR
)

# TabPFN import (optional - will gracefully handle if not installed)
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not installed. Install with: pip install tabpfn")

# Project root (projection/) for writing latest_mae.json
_BACKTEST_PROJECT_ROOT = Path(__file__).resolve().parent


def _latest_mae_path() -> Path:
    """Path to backtests/latest_mae.json for dashboard."""
    return _BACKTEST_PROJECT_ROOT / BACKTESTS_DIR / "latest_mae.json"


def _write_latest_mae(
    skater_mae: Optional[float] = None,
    goalie_mae: Optional[float] = None,
) -> None:
    """
    Update backtests/latest_mae.json with current MAE values.
    Merges with existing file so full backtest updates skater, slate goalie updates goalie.
    overall_mae = average of present skater_mae and goalie_mae, or the single value.
    """
    path = _latest_mae_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    if skater_mae is not None:
        existing["skater_mae"] = round(skater_mae, 2)
    if goalie_mae is not None:
        existing["goalie_mae"] = round(goalie_mae, 2)
    sk = existing.get("skater_mae")
    gk = existing.get("goalie_mae")
    if sk is not None and gk is not None:
        existing["overall_mae"] = round((sk + gk) / 2.0, 2)
    elif sk is not None:
        existing["overall_mae"] = sk
    elif gk is not None:
        existing["overall_mae"] = gk
    existing["updated"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


class NHLBacktester:
    """
    Backtest projection models against historical NHL data.
    """

    def __init__(self):
        self.client = NHLAPIClient(rate_limit_delay=0.25)

    def calculate_skater_dk_points(self, game: dict) -> float:
        """Calculate DraftKings fantasy points for a skater game."""
        goals = game.get('goals', 0)
        assists = game.get('assists', 0)
        shots = game.get('shots', 0)
        blocks = game.get('blockedShots', 0) or game.get('blocked', 0) or 0

        # Shorthanded points
        sh_goals = game.get('shorthandedGoals', 0) or game.get('shGoals', 0) or 0
        sh_assists = game.get('shorthandedAssists', 0) or 0
        sh_points = sh_goals + sh_assists

        # Base scoring
        pts = (
            goals * SKATER_SCORING['goals'] +
            assists * SKATER_SCORING['assists'] +
            shots * SKATER_SCORING['shots_on_goal'] +
            blocks * SKATER_SCORING['blocked_shots'] +
            sh_points * SKATER_SCORING['shorthanded_points_bonus']
        )

        # Bonuses
        if goals >= 3:
            pts += SKATER_BONUSES['hat_trick']
        if shots >= 5:
            pts += SKATER_BONUSES['five_plus_shots']
        if blocks >= 3:
            pts += SKATER_BONUSES['three_plus_blocks']
        if (goals + assists) >= 3:
            pts += SKATER_BONUSES['three_plus_points']

        return pts

    def calculate_goalie_dk_points(self, game: dict) -> float:
        """Calculate DraftKings fantasy points for a goalie game."""
        saves = game.get('saves', 0) or game.get('savesAgainst', 0) or 0
        goals_against = game.get('goalsAgainst', 0)
        decision = game.get('decision', '')

        # Win/Loss/OTL
        is_win = decision == 'W'
        is_otl = decision == 'O'
        is_shutout = goals_against == 0 and is_win

        pts = (
            saves * GOALIE_SCORING['save'] +
            goals_against * GOALIE_SCORING['goal_against']
        )

        if is_win:
            pts += GOALIE_SCORING['win']
        if is_otl:
            pts += GOALIE_SCORING['overtime_loss']
        if is_shutout:
            pts += GOALIE_SCORING['shutout_bonus']
        if saves >= 35:
            pts += GOALIE_BONUSES['thirty_five_plus_saves']

        return pts

    def fetch_player_game_logs(self, player_ids: List[int],
                                season: str = CURRENT_SEASON,
                                max_players: Optional[int] = None) -> pd.DataFrame:
        """Fetch game-by-game logs for multiple players."""
        all_logs = []

        player_ids = player_ids[:max_players] if max_players else player_ids
        print(f"Fetching game logs for {len(player_ids)} players...")

        for pid in tqdm(player_ids, desc="Fetching game logs"):
            try:
                log_data = self.client.get_player_game_log(pid, season)
                games = log_data.get('gameLog', [])

                for game in games:
                    game['player_id'] = pid
                    all_logs.append(game)

            except Exception as e:
                continue

        df = pd.DataFrame(all_logs)
        return df

    def prepare_backtest_data(self, game_logs: pd.DataFrame,
                               player_type: str = 'skater') -> pd.DataFrame:
        """
        Prepare game logs for backtesting.

        Calculates:
        - Actual DK fantasy points per game
        - Rolling averages for features
        """
        df = game_logs.copy()

        if df.empty:
            return df

        # Calculate actual fantasy points
        if player_type == 'skater':
            df['actual_fpts'] = df.apply(self.calculate_skater_dk_points, axis=1)
        else:
            df['actual_fpts'] = df.apply(self.calculate_goalie_dk_points, axis=1)

        # Sort by player and date
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df = df.sort_values(['player_id', 'gameDate'])

        # Calculate rolling features for each player
        rolling_windows = [3, 5, 10]

        for window in rolling_windows:
            # Rolling average fantasy points
            df[f'fpts_avg_{window}'] = df.groupby('player_id')['actual_fpts'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            if player_type == 'skater':
                # Rolling stats
                for col in ['goals', 'assists', 'shots']:
                    if col in df.columns:
                        df[f'{col}_avg_{window}'] = df.groupby('player_id')[col].transform(
                            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                        )

        # Games played (for sample size weighting)
        df['games_played'] = df.groupby('player_id').cumcount()

        return df

    def run_backtest(self, game_logs: pd.DataFrame,
                      min_games: int = 10,
                      train_games: int = 10,
                      player_type: str = 'skater') -> Dict:
        """
        Run backtest on historical data.

        For each player with enough games:
        - Use first `train_games` to establish baseline
        - Predict remaining games using rolling averages
        - Compare predictions to actuals

        Returns metrics and detailed results.
        """
        df = self.prepare_backtest_data(game_logs, player_type)

        if df.empty:
            return {'error': 'No data to backtest'}

        # Filter to players with enough games
        player_game_counts = df.groupby('player_id').size()
        valid_players = player_game_counts[player_game_counts >= min_games].index
        df = df[df['player_id'].isin(valid_players)]

        print(f"\nBacktesting {len(valid_players)} players with {min_games}+ games")

        results = []

        for player_id in tqdm(valid_players, desc="Backtesting"):
            player_df = df[df['player_id'] == player_id].copy()

            # Skip first `train_games` for training
            test_df = player_df.iloc[train_games:].copy()

            if len(test_df) == 0:
                continue

            # Simple prediction: use 5-game rolling average
            test_df['predicted_fpts'] = test_df['fpts_avg_5']

            # Calculate error
            test_df['error'] = test_df['predicted_fpts'] - test_df['actual_fpts']
            test_df['abs_error'] = test_df['error'].abs()
            test_df['squared_error'] = test_df['error'] ** 2

            results.append(test_df)

        if not results:
            return {'error': 'No valid test data'}

        all_results = pd.concat(results, ignore_index=True)

        # Calculate aggregate metrics
        metrics = {
            'n_predictions': len(all_results),
            'n_players': len(valid_players),
            'mae': all_results['abs_error'].mean(),
            'rmse': np.sqrt(all_results['squared_error'].mean()),
            'correlation': all_results[['predicted_fpts', 'actual_fpts']].corr().iloc[0, 1],
            'mean_actual': all_results['actual_fpts'].mean(),
            'mean_predicted': all_results['predicted_fpts'].mean(),
            'std_actual': all_results['actual_fpts'].std(),
            'std_predicted': all_results['predicted_fpts'].std(),
        }

        # Percentile accuracy (within X points)
        for threshold in [2, 5, 10]:
            pct = (all_results['abs_error'] <= threshold).mean() * 100
            metrics[f'within_{threshold}_pts_pct'] = pct

        return {
            'metrics': metrics,
            'results': all_results,
            'player_type': player_type
        }

    def run_model_comparison(self, game_logs: pd.DataFrame,
                              min_games: int = 15,
                              train_games: int = 10,
                              player_type: str = 'skater') -> Dict:
        """
        Compare different prediction models:
        1. 3-game rolling average
        2. 5-game rolling average
        3. 10-game rolling average
        4. Season average (all prior games)
        """
        df = self.prepare_backtest_data(game_logs, player_type)

        if df.empty:
            return {'error': 'No data to backtest'}

        # Filter to players with enough games
        player_game_counts = df.groupby('player_id').size()
        valid_players = player_game_counts[player_game_counts >= min_games].index
        df = df[df['player_id'].isin(valid_players)]

        print(f"\nComparing models on {len(valid_players)} players")

        models = {
            '3_game_avg': 'fpts_avg_3',
            '5_game_avg': 'fpts_avg_5',
            '10_game_avg': 'fpts_avg_10',
        }

        model_results = {}

        for model_name, feature_col in models.items():
            results = []

            for player_id in valid_players:
                player_df = df[df['player_id'] == player_id].copy()
                test_df = player_df.iloc[train_games:].copy()

                if len(test_df) == 0 or feature_col not in test_df.columns:
                    continue

                test_df['predicted_fpts'] = test_df[feature_col]
                test_df['error'] = test_df['predicted_fpts'] - test_df['actual_fpts']
                test_df['abs_error'] = test_df['error'].abs()
                test_df['squared_error'] = test_df['error'] ** 2

                results.append(test_df)

            if results:
                all_results = pd.concat(results, ignore_index=True)

                model_results[model_name] = {
                    'mae': all_results['abs_error'].mean(),
                    'rmse': np.sqrt(all_results['squared_error'].mean()),
                    'correlation': all_results[['predicted_fpts', 'actual_fpts']].corr().iloc[0, 1],
                    'n_predictions': len(all_results),
                }

        return model_results

    def print_backtest_report(self, backtest_results: Dict):
        """Print formatted backtest report."""
        if 'error' in backtest_results:
            print(f"Error: {backtest_results['error']}")
            return

        metrics = backtest_results['metrics']
        player_type = backtest_results.get('player_type', 'skater')

        print("\n" + "=" * 60)
        print(f" BACKTEST RESULTS - {player_type.upper()}S")
        print("=" * 60)

        print(f"\nSample Size:")
        print(f"  Predictions: {metrics['n_predictions']:,}")
        print(f"  Players: {metrics['n_players']}")

        print(f"\nAccuracy Metrics:")
        print(f"  MAE (Mean Absolute Error): {metrics['mae']:.2f} pts")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.2f} pts")
        print(f"  Correlation: {metrics['correlation']:.3f}")

        print(f"\nPrediction Accuracy:")
        print(f"  Within 2 pts: {metrics['within_2_pts_pct']:.1f}%")
        print(f"  Within 5 pts: {metrics['within_5_pts_pct']:.1f}%")
        print(f"  Within 10 pts: {metrics['within_10_pts_pct']:.1f}%")

        print(f"\nDistribution:")
        print(f"  Actual mean: {metrics['mean_actual']:.2f} pts (std: {metrics['std_actual']:.2f})")
        print(f"  Predicted mean: {metrics['mean_predicted']:.2f} pts (std: {metrics['std_predicted']:.2f})")

    def print_model_comparison(self, comparison_results: Dict):
        """Print model comparison table."""
        print("\n" + "=" * 60)
        print(" MODEL COMPARISON")
        print("=" * 60)

        print(f"\n{'Model':<15} {'MAE':<10} {'RMSE':<10} {'Corr':<10} {'N':<10}")
        print("-" * 55)

        for model_name, metrics in sorted(comparison_results.items(), key=lambda x: x[1]['mae']):
            print(f"{model_name:<15} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f} "
                  f"{metrics['correlation']:<10.3f} {metrics['n_predictions']:<10,}")

    def prepare_tabpfn_features(self, game_logs: pd.DataFrame,
                                  player_type: str = 'skater') -> pd.DataFrame:
        """
        Prepare rich feature matrix for TabPFN training.

        Features include:
        - Rolling averages (3, 5, 10 games)
        - Rolling std (variance indicator)
        - Games played (experience)
        - Home/away indicator
        - Days rest
        - Player identity: season average, position encoding
        """
        df = self.prepare_backtest_data(game_logs, player_type)

        if df.empty:
            return df

        # Additional features for TabPFN

        # Home/Away indicator
        if 'homeRoadFlag' in df.columns:
            df['is_home'] = (df['homeRoadFlag'] == 'H').astype(int)
        else:
            df['is_home'] = 0

        # Rolling standard deviation (measures consistency)
        for window in [5, 10]:
            df[f'fpts_std_{window}'] = df.groupby('player_id')['actual_fpts'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).std()
            )

        # Rolling max (ceiling indicator)
        df['fpts_max_10'] = df.groupby('player_id')['actual_fpts'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).max()
        )

        # Rolling min (floor indicator)
        df['fpts_min_10'] = df.groupby('player_id')['actual_fpts'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).min()
        )

        # Days since last game (rest)
        df['days_rest'] = df.groupby('player_id')['gameDate'].diff().dt.days.fillna(3)
        df['days_rest'] = df['days_rest'].clip(0, 10)  # Cap at 10 days

        # Trend (recent vs longer term)
        df['trend'] = df['fpts_avg_3'] - df['fpts_avg_10']

        # Player identity features for pooled training
        # Season expanding average (all prior games for this player)
        df['season_avg_fpts'] = df.groupby('player_id')['actual_fpts'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # Season expanding std
        df['season_std_fpts'] = df.groupby('player_id')['actual_fpts'].transform(
            lambda x: x.shift(1).expanding(min_periods=2).std()
        )

        if player_type == 'skater':
            # Shot trend
            if 'shots_avg_3' in df.columns and 'shots_avg_10' in df.columns:
                df['shots_trend'] = df['shots_avg_3'] - df['shots_avg_10']

            # Goals trend
            if 'goals_avg_3' in df.columns and 'goals_avg_10' in df.columns:
                df['goals_trend'] = df['goals_avg_3'] - df['goals_avg_10']

            # Season goal rate
            if 'goals' in df.columns:
                df['season_goals_pg'] = df.groupby('player_id')['goals'].transform(
                    lambda x: x.shift(1).expanding(min_periods=1).mean()
                )

            # Season shots rate
            if 'shots' in df.columns:
                df['season_shots_pg'] = df.groupby('player_id')['shots'].transform(
                    lambda x: x.shift(1).expanding(min_periods=1).mean()
                )

        return df

    def run_tabpfn_backtest(self, game_logs: pd.DataFrame,
                             min_games: int = 20,
                             train_games: int = 15,
                             player_type: str = 'skater') -> Dict:
        """
        Run backtest using TabPFN model with pooled cross-player training.

        Key improvement: trains ONE model on ALL players' data instead of
        per-player models with only 15 samples each. Player identity is
        captured via season averages and position features.

        Uses time-based split: first `train_games` per player form the
        training pool, remaining games are test predictions.
        """
        if not TABPFN_AVAILABLE:
            return {'error': 'TabPFN not installed'}

        df = self.prepare_tabpfn_features(game_logs, player_type)

        if df.empty:
            return {'error': 'No data to backtest'}

        # Define feature columns — includes player identity features
        feature_cols = [
            'fpts_avg_3', 'fpts_avg_5', 'fpts_avg_10',
            'fpts_std_5', 'fpts_std_10',
            'fpts_max_10', 'fpts_min_10',
            'season_avg_fpts', 'season_std_fpts',
            'games_played', 'is_home', 'days_rest', 'trend'
        ]

        if player_type == 'skater':
            feature_cols.extend([
                'goals_avg_3', 'goals_avg_5', 'goals_avg_10',
                'assists_avg_3', 'assists_avg_5', 'assists_avg_10',
                'shots_avg_3', 'shots_avg_5', 'shots_avg_10',
                'season_goals_pg', 'season_shots_pg',
            ])
            if 'shots_trend' in df.columns:
                feature_cols.append('shots_trend')
            if 'goals_trend' in df.columns:
                feature_cols.append('goals_trend')

        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Filter to players with enough games
        player_game_counts = df.groupby('player_id').size()
        valid_players = player_game_counts[player_game_counts >= min_games].index
        df = df[df['player_id'].isin(valid_players)]

        print(f"\nRunning TabPFN pooled backtest on {len(valid_players)} players")
        print(f"Using {len(feature_cols)} features (with player identity)")

        # Sort by date within each player for proper time-based split
        df = df.sort_values(['player_id', 'gameDate'])

        # Build pooled train/test sets from all players
        train_frames = []
        test_frames = []

        for player_id in valid_players:
            player_df = df[df['player_id'] == player_id].copy()

            if len(player_df) < min_games:
                continue

            train_frames.append(player_df.iloc[:train_games])
            test_frames.append(player_df.iloc[train_games:])

        if not train_frames or not test_frames:
            return {'error': 'No valid train/test data'}

        train_pool = pd.concat(train_frames, ignore_index=True)
        test_pool = pd.concat(test_frames, ignore_index=True)

        print(f"  Training pool: {len(train_pool)} games from {len(train_frames)} players")
        print(f"  Test pool: {len(test_pool)} games")

        # Prepare feature matrices
        X_train = train_pool[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_pool['actual_fpts'].values

        X_test = test_pool[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        try:
            model = TabPFNRegressor(ignore_pretraining_limits=True)
            print("  Training pooled TabPFN model...")
            model.fit(X_train.values, y_train)

            # Predict in batches if test set is large
            batch_size = 5000
            all_preds = []
            for i in range(0, len(X_test), batch_size):
                batch = X_test.iloc[i:i + batch_size]
                preds = model.predict(batch.values)
                all_preds.extend(preds)

            test_pool = test_pool.copy()
            test_pool['predicted_fpts'] = all_preds
            test_pool['error'] = test_pool['predicted_fpts'] - test_pool['actual_fpts']
            test_pool['abs_error'] = test_pool['error'].abs()
            test_pool['squared_error'] = test_pool['error'] ** 2

        except Exception as e:
            return {'error': f'TabPFN training failed: {e}'}

        all_results = test_pool

        # Calculate metrics
        metrics = {
            'n_predictions': len(all_results),
            'n_players': len(valid_players),
            'mae': all_results['abs_error'].mean(),
            'rmse': np.sqrt(all_results['squared_error'].mean()),
            'correlation': all_results[['predicted_fpts', 'actual_fpts']].corr().iloc[0, 1],
            'mean_actual': all_results['actual_fpts'].mean(),
            'mean_predicted': all_results['predicted_fpts'].mean(),
            'std_actual': all_results['actual_fpts'].std(),
            'std_predicted': all_results['predicted_fpts'].std(),
        }

        for threshold in [2, 5, 10]:
            pct = (all_results['abs_error'] <= threshold).mean() * 100
            metrics[f'within_{threshold}_pts_pct'] = pct

        return {
            'metrics': metrics,
            'results': all_results,
            'player_type': player_type,
            'model': 'TabPFN_pooled',
            'features': feature_cols
        }

    def run_full_model_comparison(self, game_logs: pd.DataFrame,
                                    min_games: int = 20,
                                    train_games: int = 15,
                                    player_type: str = 'skater') -> Dict:
        """
        Compare all models including TabPFN:
        1. 3-game rolling average
        2. 5-game rolling average
        3. 10-game rolling average
        4. TabPFN (ML model)
        """
        # First run standard comparison
        standard_results = self.run_model_comparison(
            game_logs, min_games=min_games, train_games=train_games, player_type=player_type
        )

        # Add TabPFN if available
        if TABPFN_AVAILABLE:
            print("\nTraining TabPFN model...")
            tabpfn_results = self.run_tabpfn_backtest(
                game_logs, min_games=min_games, train_games=train_games, player_type=player_type
            )

            if 'metrics' in tabpfn_results:
                standard_results['TabPFN'] = {
                    'mae': tabpfn_results['metrics']['mae'],
                    'rmse': tabpfn_results['metrics']['rmse'],
                    'correlation': tabpfn_results['metrics']['correlation'],
                    'n_predictions': tabpfn_results['metrics']['n_predictions'],
                }

        return standard_results

    def run_feature_comparison(self, game_logs: pd.DataFrame,
                                baseline_features: List[str],
                                enhanced_features: List[str],
                                min_games: int = 20,
                                train_games: int = 15,
                                player_type: str = 'skater') -> Dict:
        """
        Compare baseline features vs enhanced features (with xG, form, PDO).

        Args:
            game_logs: Game-by-game player data
            baseline_features: List of baseline feature columns
            enhanced_features: List of enhanced feature columns (including xG, etc.)
            min_games: Minimum games required
            train_games: Games to use for training
            player_type: 'skater' or 'goalie'

        Returns:
            Dict with comparison metrics for both feature sets
        """
        if not TABPFN_AVAILABLE:
            return {'error': 'TabPFN required for feature comparison'}

        df = self.prepare_tabpfn_features(game_logs, player_type)

        if df.empty:
            return {'error': 'No data to backtest'}

        # Filter to players with enough games
        player_game_counts = df.groupby('player_id').size()
        valid_players = player_game_counts[player_game_counts >= min_games].index
        df = df[df['player_id'].isin(valid_players)]

        print(f"\nRunning feature comparison on {len(valid_players)} players")

        results = {}

        for feature_set_name, feature_cols in [('baseline', baseline_features),
                                                ('enhanced', enhanced_features)]:
            # Filter to available columns
            available_cols = [c for c in feature_cols if c in df.columns]

            if len(available_cols) < 3:
                print(f"  Warning: {feature_set_name} has only {len(available_cols)} features")
                continue

            print(f"\n  Testing {feature_set_name} ({len(available_cols)} features)...")

            all_predictions = []

            for player_id in tqdm(valid_players, desc=f"{feature_set_name}"):
                player_df = df[df['player_id'] == player_id].copy()

                if len(player_df) < min_games:
                    continue

                train_df = player_df.iloc[:train_games]
                test_df = player_df.iloc[train_games:]

                if len(test_df) == 0:
                    continue

                X_train = train_df[available_cols].fillna(0).replace([np.inf, -np.inf], 0)
                y_train = train_df['actual_fpts'].values

                X_test = test_df[available_cols].fillna(0).replace([np.inf, -np.inf], 0)
                y_test = test_df['actual_fpts'].values

                try:
                    model = TabPFNRegressor(ignore_pretraining_limits=True)
                    model.fit(X_train.values, y_train)
                    predictions = model.predict(X_test.values)

                    test_df = test_df.copy()
                    test_df['predicted_fpts'] = predictions
                    test_df['error'] = test_df['predicted_fpts'] - test_df['actual_fpts']
                    test_df['abs_error'] = test_df['error'].abs()
                    test_df['squared_error'] = test_df['error'] ** 2

                    all_predictions.append(test_df)

                except Exception as e:
                    continue

            if all_predictions:
                all_results = pd.concat(all_predictions, ignore_index=True)

                results[feature_set_name] = {
                    'mae': all_results['abs_error'].mean(),
                    'rmse': np.sqrt(all_results['squared_error'].mean()),
                    'correlation': all_results[['predicted_fpts', 'actual_fpts']].corr().iloc[0, 1],
                    'n_predictions': len(all_results),
                    'n_features': len(available_cols)
                }

        return results

    def run_xg_ablation_study(self, game_logs: pd.DataFrame,
                               min_games: int = 20,
                               train_games: int = 15,
                               player_type: str = 'skater') -> Dict:
        """
        Ablation study: test incremental value of xG, form, and PDO features.

        Feature groups tested:
        1. Baseline (rolling averages only)
        2. + xG features
        3. + Recent form
        4. + PDO regression
        5. All combined

        Returns:
            Dict with metrics for each feature group
        """
        # Define feature groups
        baseline = [
            'fpts_avg_3', 'fpts_avg_5', 'fpts_avg_10',
            'goals_avg_3', 'goals_avg_5', 'goals_avg_10',
            'assists_avg_3', 'assists_avg_5', 'assists_avg_10',
            'shots_avg_3', 'shots_avg_5', 'shots_avg_10',
            'games_played', 'is_home', 'days_rest', 'trend'
        ]

        xg_features = [
            'team_xgf_60', 'team_xga_60', 'opp_xga_60', 'xg_matchup_boost'
        ]

        form_features = [
            'team_form_xgf', 'team_form_cf', 'team_hot_streak', 'team_cold_streak'
        ]

        pdo_features = [
            'team_pdo', 'pdo_regression_flag', 'pdo_adj_factor'
        ]

        injury_features = [
            'team_injury_count', 'opportunity_boost'
        ]

        feature_groups = {
            '1_baseline': baseline,
            '2_plus_xg': baseline + xg_features,
            '3_plus_form': baseline + xg_features + form_features,
            '4_plus_pdo': baseline + xg_features + form_features + pdo_features,
            '5_all': baseline + xg_features + form_features + pdo_features + injury_features
        }

        print("\n" + "=" * 60)
        print(" XG FEATURE ABLATION STUDY")
        print("=" * 60)

        results = {}

        for group_name, features in feature_groups.items():
            comparison = self.run_feature_comparison(
                game_logs,
                baseline_features=features,
                enhanced_features=features,  # Same for this test
                min_games=min_games,
                train_games=train_games
            )

            if 'baseline' in comparison:
                results[group_name] = comparison['baseline']
                print(f"  {group_name}: MAE={comparison['baseline']['mae']:.2f}")

        return results

    def print_feature_comparison_report(self, comparison_results: Dict):
        """Print formatted feature comparison report."""
        print("\n" + "=" * 60)
        print(" FEATURE COMPARISON REPORT")
        print("=" * 60)

        if not comparison_results:
            print("  No results to display")
            return

        print(f"\n{'Feature Set':<20} {'MAE':<10} {'RMSE':<10} {'Corr':<10} {'Features':<10} {'N':<10}")
        print("-" * 70)

        # Sort by MAE (lower is better)
        for name, metrics in sorted(comparison_results.items(), key=lambda x: x[1].get('mae', 999)):
            print(f"{name:<20} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f} "
                  f"{metrics['correlation']:<10.3f} {metrics.get('n_features', '-'):<10} "
                  f"{metrics['n_predictions']:<10,}")

        # Calculate improvement
        if 'baseline' in comparison_results and 'enhanced' in comparison_results:
            baseline_mae = comparison_results['baseline']['mae']
            enhanced_mae = comparison_results['enhanced']['mae']
            improvement = (baseline_mae - enhanced_mae) / baseline_mae * 100

            print(f"\n  MAE Improvement: {improvement:.1f}%")

            if improvement > 0:
                print(f"  Enhanced features reduced error by {baseline_mae - enhanced_mae:.2f} pts")
            else:
                print(f"  Enhanced features increased error by {enhanced_mae - baseline_mae:.2f} pts")


def _parse_toi_minutes(toi_val) -> float:
    """Parse TOI from MM:SS string or numeric to float minutes.

    Args:
        toi_val: TOI value — e.g. "18:30", "0:00", 18.5, or None

    Returns:
        Float minutes (e.g. 18.5). Returns 0.0 for missing/invalid values.
    """
    if toi_val is None or (isinstance(toi_val, float) and np.isnan(toi_val)):
        return 0.0
    if isinstance(toi_val, (int, float)):
        return float(toi_val)
    toi_str = str(toi_val).strip()
    if ':' in toi_str:
        parts = toi_str.split(':')
        try:
            return int(parts[0]) + int(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(toi_str)
    except ValueError:
        return 0.0


def fetch_skaters_actuals_for_date(client: NHLAPIClient, date: str) -> pd.DataFrame:
    """
    Fetch actual skater DK fantasy points for a slate date from NHL API boxscores.

    Args:
        client: NHLAPIClient instance
        date: YYYY-MM-DD (e.g. 2026-01-28)

    Returns:
        DataFrame with columns: name (from boxscore), team, actual_fpts, last_name (for matching)
    """
    sched = client.get_schedule(date=date)
    rows = []
    backtester = NHLBacktester()
    for week in sched.get("gameWeek", []):
        if week.get("date") != date:
            continue
        for game in week.get("games", []):
            if game.get("gameState") != "OFF":
                continue
            gid = game.get("id")
            away_abbrev = game.get("awayTeam", {}).get("abbrev", "")
            home_abbrev = game.get("homeTeam", {}).get("abbrev", "")
            try:
                box = client.get_boxscore(gid)
            except Exception:
                continue
            pbs = box.get("playerByGameStats", {})
            for side, abbrev in [("awayTeam", away_abbrev), ("homeTeam", home_abbrev)]:
                side_data = pbs.get(side, {})
                skater_list = side_data.get("skaters", [])
                if not skater_list:
                    fwd = side_data.get("forwards", [])
                    defs = side_data.get("defensemen", [])
                    skater_list = (fwd if isinstance(fwd, list) else []) + (defs if isinstance(defs, list) else [])
                for s in skater_list:
                    if not isinstance(s, dict):
                        continue
                    name_raw = s.get("name", {})
                    name = name_raw.get("default", "") if isinstance(name_raw, dict) else str(name_raw)
                    goals = s.get("goals", 0) or 0
                    assists = s.get("assists", 0) or 0
                    shots = s.get("shots", 0) or 0
                    blocks = s.get("blockedShots", 0) or s.get("blocked", 0) or 0
                    sh_goals = s.get("shorthandedGoals", 0) or s.get("shGoals", 0) or 0
                    sh_assists = s.get("shorthandedAssists", 0) or 0
                    toi_raw = s.get("toi", "0:00")
                    toi_minutes = _parse_toi_minutes(toi_raw)
                    game_dict = {
                        "goals": goals,
                        "assists": assists,
                        "shots": shots,
                        "blockedShots": blocks,
                        "shorthandedGoals": sh_goals,
                        "shorthandedAssists": sh_assists,
                    }
                    pts = backtester.calculate_skater_dk_points(game_dict)
                    last_name = name.split()[-1] if name else ""
                    rows.append({"name": name, "team": abbrev, "actual_fpts": pts,
                                 "last_name": last_name, "toi_minutes": toi_minutes})
    if not rows:
        return pd.DataFrame(columns=["name", "team", "actual_fpts", "last_name", "toi_minutes"])
    return pd.DataFrame(rows)


def fetch_goalies_actuals_for_date(client: NHLAPIClient, date: str) -> pd.DataFrame:
    """
    Fetch actual goalie DK fantasy points for a slate date from NHL API boxscores.

    Args:
        client: NHLAPIClient instance
        date: YYYY-MM-DD (e.g. 2026-01-28)

    Returns:
        DataFrame with columns: name (from boxscore, e.g. "S. Martin"), team, actual_fpts, last_name (for matching)
    """
    sched = client.get_schedule(date=date)
    rows = []
    for week in sched.get("gameWeek", []):
        if week.get("date") != date:
            continue
        for game in week.get("games", []):
            if game.get("gameState") != "OFF":
                continue
            gid = game.get("id")
            away_abbrev = game.get("awayTeam", {}).get("abbrev", "")
            home_abbrev = game.get("homeTeam", {}).get("abbrev", "")
            try:
                box = client.get_boxscore(gid)
            except Exception:
                continue
            pbs = box.get("playerByGameStats", {})
            backtester = NHLBacktester()
            for side, abbrev in [("awayTeam", away_abbrev), ("homeTeam", home_abbrev)]:
                for g in pbs.get(side, {}).get("goalies", []):
                    toi = g.get("toi", "0:00")
                    saves = g.get("saves", 0) or 0
                    if toi == "00:00" and saves == 0:
                        continue
                    name_raw = g.get("name", {})
                    name = name_raw.get("default", "") if isinstance(name_raw, dict) else str(name_raw)
                    toi_minutes = _parse_toi_minutes(toi)
                    game_dict = {
                        "saves": g.get("saves", 0),
                        "goalsAgainst": g.get("goalsAgainst", 0),
                        "decision": g.get("decision", ""),
                    }
                    pts = backtester.calculate_goalie_dk_points(game_dict)
                    last_name = name.split()[-1] if name else ""
                    rows.append({"name": name, "team": abbrev, "actual_fpts": pts,
                                 "last_name": last_name, "toi_minutes": toi_minutes})
    if not rows:
        return pd.DataFrame(columns=["name", "team", "actual_fpts", "last_name", "toi_minutes"])
    return pd.DataFrame(rows)


def run_slate_goalie_backtest(
    date: str = "2026-01-28",
    use_danger_stats: bool = True,
) -> Dict:
    """
    Run goalie projection vs actuals for a single slate date (e.g. last night).
    Uses opponent shot quality + save % by type when use_danger_stats=True.

    Returns:
        Dict with mae, bias, n_matched, n_actuals, n_projected, details DataFrame.
    """
    from data_pipeline import NHLDataPipeline
    from projections import NHLProjectionModel
    import config as cfg

    if use_danger_stats:
        cfg.TEAM_DANGER_CSV_DIR = "../test"
    backtester = NHLBacktester()
    client = backtester.client

    # Build dataset and generate projections for date
    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(
        include_game_logs=False,
        include_injuries=False,
        include_advanced_stats=False,
    )
    # Override schedule with the target date (default fetch uses today's date)
    data['schedule'] = pipeline.fetch_schedule(date)
    model = NHLProjectionModel()
    projections = model.generate_projections(data, target_date=date, filter_injuries=False)
    goalie_proj = projections.get("goalies")
    if goalie_proj is None or goalie_proj.empty:
        return {"error": "No goalie projections", "mae": None, "bias": None, "n_matched": 0}

    # Actuals from NHL API
    actuals_df = fetch_goalies_actuals_for_date(client, date)
    if actuals_df.empty:
        return {"error": "No actual goalie results for date", "mae": None, "bias": None, "n_matched": 0}

    # Match by last name + team (boxscore has "S. Martin", projection has "Scott Martin")
    def match_goalie(row_act):
        last = row_act["last_name"]
        team = row_act["team"]
        cand = goalie_proj[
            (goalie_proj["team"] == team)
            & (goalie_proj["name"].str.endswith(" " + last, na=False))
        ]
        if len(cand) == 1:
            return cand.index[0]
        if len(cand) > 1:
            # Prefer exact match on full name if possible
            for idx, r in cand.iterrows():
                if r["name"].split()[-1] == last:
                    return idx
            return cand.index[0]
        return None

    actuals_df["proj_idx"] = actuals_df.apply(match_goalie, axis=1)
    matched = actuals_df.dropna(subset=["proj_idx"])

    # Filter out goalies with TOI=0 before computing error metrics
    if "toi_minutes" in matched.columns:
        n_before = len(matched)
        matched = matched[matched["toi_minutes"] > 0].copy()
        n_filtered = n_before - len(matched)
        if n_filtered > 0:
            print(f"  Filtered {n_filtered} goalies with TOI=0")

    if matched.empty:
        return {
            "error": "No goalies matched between projections and actuals",
            "mae": None,
            "bias": None,
            "n_matched": 0,
            "n_actuals": len(actuals_df),
            "n_projected": len(goalie_proj),
        }

    proj_fpts = matched["proj_idx"].map(lambda i: goalie_proj.loc[i, "projected_fpts"])
    actual_fpts = matched["actual_fpts"].values
    err = proj_fpts.values - actual_fpts
    mae = float(np.abs(err).mean())
    bias = float(err.mean())

    details = matched[["name", "team", "actual_fpts"]].copy()
    details["projected_fpts"] = proj_fpts.values
    details["error"] = err

    return {
        "mae": mae,
        "bias": bias,
        "n_matched": len(matched),
        "n_actuals": len(actuals_df),
        "n_projected": len(goalie_proj),
        "details": details,
    }


def run_slate_skater_backtest(
    date: str = "2026-01-28",
    use_rate_based: bool = False,
) -> Dict:
    """
    Run skater projection vs actuals for a single slate date.
    When use_rate_based=True, sets USE_DK_PER_TOI_PROJECTION so projections use
    dk_pts_per_60 * (expected_toi_minutes/60).

    Returns:
        Dict with mae, bias, n_matched, n_actuals, n_projected, details DataFrame.
    """
    from data_pipeline import NHLDataPipeline
    from projections import NHLProjectionModel
    import config as cfg

    cfg.USE_DK_PER_TOI_PROJECTION = use_rate_based
    if use_rate_based:
        print("  (rate-based: USE_DK_PER_TOI_PROJECTION=True)")
    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(
        include_game_logs=False,
        include_injuries=False,
        include_advanced_stats=False,
    )
    data["schedule"] = pipeline.fetch_schedule(date)
    model = NHLProjectionModel()
    projections = model.generate_projections(data, target_date=date, filter_injuries=False)
    skater_proj = projections.get("skaters")
    if skater_proj is None or skater_proj.empty:
        return {"error": "No skater projections", "mae": None, "bias": None, "n_matched": 0}

    backtester = NHLBacktester()
    actuals_df = fetch_skaters_actuals_for_date(backtester.client, date)
    if actuals_df.empty:
        return {"error": "No actual skater results for date", "mae": None, "bias": None, "n_matched": 0}

    def match_skater(row_act):
        last = row_act["last_name"]
        team = row_act["team"]
        cand = skater_proj[
            (skater_proj["team"] == team)
            & (skater_proj["name"].str.endswith(" " + last, na=False))
        ]
        if len(cand) == 1:
            return cand.index[0]
        if len(cand) > 1:
            for idx, r in cand.iterrows():
                if r["name"].split()[-1] == last:
                    return idx
            return cand.index[0]
        return None

    actuals_df = actuals_df.copy()
    actuals_df["proj_idx"] = actuals_df.apply(match_skater, axis=1)
    matched = actuals_df.dropna(subset=["proj_idx"])

    # Filter out scratches/DNPs (TOI = 0) before computing error metrics
    if "toi_minutes" in matched.columns:
        n_before = len(matched)
        matched = matched[matched["toi_minutes"] > 0].copy()
        n_filtered = n_before - len(matched)
        if n_filtered > 0:
            print(f"  Filtered {n_filtered} skaters with TOI=0 (scratches/DNPs)")

    if matched.empty:
        return {
            "error": "No skaters matched between projections and actuals",
            "mae": None,
            "bias": None,
            "n_matched": 0,
            "n_actuals": len(actuals_df),
            "n_projected": len(skater_proj),
        }

    proj_fpts = matched["proj_idx"].map(lambda i: skater_proj.loc[i, "projected_fpts"])
    actual_fpts = matched["actual_fpts"].values
    err = proj_fpts.values - actual_fpts
    mae = float(np.abs(err).mean())
    bias = float(err.mean())
    details = matched[["name", "team", "actual_fpts"]].copy()
    details["projected_fpts"] = proj_fpts.values
    details["error"] = err

    return {
        "mae": mae,
        "bias": bias,
        "n_matched": len(matched),
        "n_actuals": len(actuals_df),
        "n_projected": len(skater_proj),
        "details": details,
    }


def run_slate_skater_comparison(date: str = "2026-01-28") -> Dict:
    """
    Compare per-game vs rate-based skater projections for a slate date.
    Runs run_slate_skater_backtest(date, use_rate_based=False) and
    run_slate_skater_backtest(date, use_rate_based=True), returns both results.
    """
    per_game = run_slate_skater_backtest(date, use_rate_based=False)
    rate_based = run_slate_skater_backtest(date, use_rate_based=True)
    return {"per_game": per_game, "rate_based": rate_based}


def run_full_backtest(max_players: int = 100, season: str = CURRENT_SEASON, include_tabpfn: bool = True):
    """
    Run full backtest pipeline.

    1. Fetch top players
    2. Get their game logs
    3. Run backtest
    4. Compare models (including TabPFN if requested)
    """
    from data_pipeline import NHLDataPipeline

    print("=" * 60)
    print(" NHL DFS PROJECTION BACKTEST")
    print("=" * 60)

    # Fetch current season stats to get top players
    pipeline = NHLDataPipeline()

    print("\nFetching player stats...")
    skaters = pipeline.fetch_all_skater_stats(season)

    # Get top players by points
    top_skaters = skaters.nlargest(max_players, 'points')
    player_ids = top_skaters['player_id'].tolist()

    print(f"Selected top {len(player_ids)} skaters by points")

    # Fetch game logs
    backtester = NHLBacktester()
    game_logs = backtester.fetch_player_game_logs(player_ids, season)

    print(f"\nFetched {len(game_logs)} total game entries")

    # Run baseline backtest
    print("\nRunning baseline backtest...")
    results = backtester.run_backtest(game_logs, min_games=15, train_games=10)
    backtester.print_backtest_report(results)

    # Compare models
    if include_tabpfn and TABPFN_AVAILABLE:
        print("\n" + "=" * 60)
        print(" COMPARING ALL MODELS (Including TabPFN)")
        print("=" * 60)
        comparison = backtester.run_full_model_comparison(
            game_logs, min_games=20, train_games=15
        )
    else:
        print("\nComparing baseline models...")
        comparison = backtester.run_model_comparison(game_logs, min_games=15, train_games=10)

    backtester.print_model_comparison(comparison)

    # Persist MAE for dashboard
    if "metrics" in results and results["metrics"].get("mae") is not None:
        _write_latest_mae(skater_mae=results["metrics"]["mae"], goalie_mae=None)

    return results, comparison, game_logs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NHL DFS Backtest')
    parser.add_argument('--players', type=int, default=75, help='Number of top players to test')
    parser.add_argument('--no-tabpfn', action='store_true', help='Skip TabPFN comparison')
    parser.add_argument('--feature-comparison', action='store_true',
                        help='Run feature comparison (baseline vs enhanced with xG)')
    parser.add_argument('--ablation', action='store_true',
                        help='Run xG feature ablation study')
    parser.add_argument('--slate-date', type=str, default=None,
                        help='Run slate goalie backtest for date (YYYY-MM-DD, e.g. 2026-01-28)')
    parser.add_argument('--no-danger', action='store_true',
                        help='With --slate-date: disable opponent shot quality / save%% by type')
    parser.add_argument('--skater-slate-date', type=str, default=None,
                        help='Run slate skater backtest for date (YYYY-MM-DD)')
    parser.add_argument('--skater-rate-based', action='store_true',
                        help='With --skater-slate-date: use rate-based (dk_pts_per_60) projections')
    parser.add_argument('--skater-compare', action='store_true',
                        help='With --skater-slate-date: compare per-game vs rate-based MAE')

    args = parser.parse_args()

    if args.skater_slate_date:
        date = args.skater_slate_date
        print("=" * 60)
        print(f" SLATE SKATER BACKTEST — {date}")
        print("=" * 60)
        if args.skater_compare:
            comp = run_slate_skater_comparison(date)
            pg, rb = comp["per_game"], comp["rate_based"]
            if pg.get("error"):
                print(f"  Per-game: {pg['error']}")
            else:
                print(f"  Per-game:   MAE={pg['mae']:.2f}  bias={pg['bias']:+.2f}  n={pg['n_matched']}")
            if rb.get("error"):
                print(f"  Rate-based: {rb['error']}")
            else:
                print(f"  Rate-based: MAE={rb['mae']:.2f}  bias={rb['bias']:+.2f}  n={rb['n_matched']}")
        else:
            result = run_slate_skater_backtest(date, use_rate_based=args.skater_rate_based)
            if result.get("error"):
                print(f"  Error: {result['error']}")
            else:
                if result.get("mae") is not None:
                    _write_latest_mae(skater_mae=result["mae"], goalie_mae=None)
                print(f"  Skater MAE:  {result['mae']:.2f} pts")
                print(f"  Skater bias: {result['bias']:+.2f} pts")
                print(f"  Matched:     {result['n_matched']} skaters")
                if result.get("details") is not None and not result["details"].empty:
                    print("\n  Sample (first 10):")
                    for _, r in result["details"].head(10).iterrows():
                        print(f"    {r['name']:<25} {r['team']}  proj={r['projected_fpts']:.1f}  actual={r['actual_fpts']:.1f}  err={r['error']:+.1f}")
        print("=" * 60)
        sys.exit(0)

    if args.slate_date:
        print("=" * 60)
        print(f" SLATE GOALIE BACKTEST — {args.slate_date}")
        print("=" * 60)
        result = run_slate_goalie_backtest(date=args.slate_date, use_danger_stats=not args.no_danger)
        if result.get("error"):
            print(f"  Error: {result['error']}")
        else:
            if result.get("mae") is not None:
                _write_latest_mae(skater_mae=None, goalie_mae=result["mae"])
            print(f"  Goalie MAE:  {result['mae']:.2f} pts")
            print(f"  Goalie bias: {result['bias']:+.2f} pts (positive = over-projected)")
            print(f"  Matched:     {result['n_matched']} goalies (of {result['n_actuals']} actuals, {result['n_projected']} projected)")
            if result.get("details") is not None and not result["details"].empty:
                print("\n  Per-goalie:")
                for _, r in result["details"].iterrows():
                    print(f"    {r['name']:<20} {r['team']}  proj={r['projected_fpts']:.1f}  actual={r['actual_fpts']:.1f}  err={r['error']:+.1f}")
        print("=" * 60)
        sys.exit(0)

    results, comparison, game_logs = run_full_backtest(
        max_players=args.players,
        include_tabpfn=not args.no_tabpfn
    )

    # Run additional tests if requested
    backtester = NHLBacktester()

    if args.feature_comparison:
        print("\n" + "=" * 60)
        print(" RUNNING FEATURE COMPARISON")
        print("=" * 60)

        # Define baseline and enhanced feature sets
        baseline_features = [
            'fpts_avg_3', 'fpts_avg_5', 'fpts_avg_10',
            'goals_avg_3', 'goals_avg_5', 'goals_avg_10',
            'assists_avg_3', 'assists_avg_5', 'assists_avg_10',
            'shots_avg_3', 'shots_avg_5', 'shots_avg_10',
            'games_played', 'is_home', 'days_rest', 'trend'
        ]

        enhanced_features = baseline_features + [
            'team_xgf_60', 'team_xga_60', 'opp_xga_60', 'xg_matchup_boost',
            'team_form_xgf', 'team_form_cf', 'team_hot_streak', 'team_cold_streak',
            'team_pdo', 'pdo_regression_flag', 'pdo_adj_factor',
            'team_injury_count', 'opportunity_boost'
        ]

        feature_comparison = backtester.run_feature_comparison(
            game_logs,
            baseline_features=baseline_features,
            enhanced_features=enhanced_features
        )

        backtester.print_feature_comparison_report(feature_comparison)

    if args.ablation:
        ablation_results = backtester.run_xg_ablation_study(game_logs)
        backtester.print_feature_comparison_report(ablation_results)
