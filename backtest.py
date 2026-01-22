"""
Backtesting module for NHL DFS projections.

Tests projection accuracy against historical game results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from nhl_api import NHLAPIClient
from config import (
    CURRENT_SEASON, SKATER_SCORING, SKATER_BONUSES,
    GOALIE_SCORING, GOALIE_BONUSES
)


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


def run_full_backtest(max_players: int = 100, season: str = CURRENT_SEASON):
    """
    Run full backtest pipeline.

    1. Fetch top players
    2. Get their game logs
    3. Run backtest
    4. Compare models
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

    # Run backtest
    print("\nRunning backtest...")
    results = backtester.run_backtest(game_logs, min_games=15, train_games=10)
    backtester.print_backtest_report(results)

    # Compare models
    print("\nComparing prediction models...")
    comparison = backtester.run_model_comparison(game_logs, min_games=15, train_games=10)
    backtester.print_model_comparison(comparison)

    return results, comparison


if __name__ == "__main__":
    results, comparison = run_full_backtest(max_players=75)
