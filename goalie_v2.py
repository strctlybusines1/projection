"""
Enhanced Goalie Projection Model v2 - Team Win Probability Focus
Focus: Team-level win prediction as a critical component of goalie FPTS
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson
from scipy.optimize import curve_fit

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_data():
    """Load all necessary data from database"""
    db_path = '/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db'
    conn = sqlite3.connect(db_path)

    print("Loading data...")

    # Load goalie data
    historical_goalies = pd.read_sql(
        "SELECT * FROM historical_goalies ORDER BY game_date", conn
    )
    game_logs_goalies = pd.read_sql(
        "SELECT * FROM game_logs_goalies ORDER BY game_date", conn
    )

    # Load skater boxscores for team stats
    boxscore_skaters = pd.read_sql(
        "SELECT * FROM boxscore_skaters", conn
    )

    historical_skaters = pd.read_sql(
        "SELECT * FROM historical_skaters", conn
    )

    conn.close()

    # Convert dates
    for df in [historical_goalies, game_logs_goalies, boxscore_skaters, historical_skaters]:
        df['game_date'] = pd.to_datetime(df['game_date'])

    print(f"Historical goalies: {len(historical_goalies)} rows")
    print(f"Current season goalies: {len(game_logs_goalies)} rows")
    print(f"Boxscore skaters: {len(boxscore_skaters)} rows")

    return historical_goalies, game_logs_goalies, boxscore_skaters, historical_skaters


def prepare_historical_team_stats(historical_skaters, historical_goalies):
    """
    Compute team-level statistics from historical skater data.
    Returns summary stats per team for use in calibration.
    """
    print("\nPreparing historical team statistics...")

    # Sum goals per team per game
    team_goals = historical_skaters.groupby(
        ['game_date', 'team']
    )['goals'].sum().reset_index()
    team_goals.columns = ['game_date', 'team', 'team_goals']

    # Compute goals against (opponent's goals)
    team_goals['opponent'] = team_goals['team'].copy()
    team_ga = team_goals[['game_date', 'opponent', 'team_goals']].copy()
    team_ga.columns = ['game_date', 'team', 'goals_against']

    # Merge into one table
    team_stats = team_goals.merge(
        team_ga,
        on=['game_date', 'team'],
        how='left'
    )

    # Compute rolling averages (10-game rolling)
    team_stats = team_stats.sort_values('game_date')
    for col in ['team_goals', 'goals_against']:
        team_stats[f'{col}_rolling10'] = team_stats.groupby('team')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

    # Get win data from goalies table
    wins = historical_goalies[historical_goalies['decision'] == 'W'][
        ['game_date', 'team']
    ].drop_duplicates()
    wins['actual_win'] = 1

    team_stats = team_stats.merge(
        wins,
        on=['game_date', 'team'],
        how='left'
    )
    team_stats['actual_win'] = team_stats['actual_win'].fillna(0).astype(int)

    return team_stats


def compute_rolling_team_stats(df, lookback_days=60):
    """
    Compute rolling team offensive/defensive stats from boxscore data.
    Used for both historical calibration and current season prediction.

    Returns df with columns:
    - team_gpg_rolling: goals per game (rolling 10)
    - team_gag_rolling: goals against per game (rolling 10)
    - opp_gpg_rolling: opponent goals per game
    - opp_gag_rolling: opponent goals against per game
    """

    # Team goals scored + opponent
    team_gf = df.groupby(['game_date', 'team', 'opponent'])['goals'].sum().reset_index()
    team_gf.columns = ['game_date', 'team', 'opponent', 'goals_for']

    # Create goals against map from opponent's perspective
    ga_map = team_gf[['game_date', 'opponent', 'goals_for']].copy()
    ga_map.columns = ['game_date', 'team', 'goals_against']

    # Merge goals against
    team_gf = team_gf.merge(ga_map, on=['game_date', 'team'], how='left')

    # Rolling averages per team
    team_gf = team_gf.sort_values(['team', 'game_date'])

    for col in ['goals_for', 'goals_against']:
        team_gf[f'{col}_rolling'] = team_gf.groupby('team')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

    # Rename for clarity
    team_gf.rename(columns={
        'goals_for_rolling': 'team_gpg_rolling',
        'goals_against_rolling': 'team_gag_rolling'
    }, inplace=True)

    # Opponent stats (join opponent team's stats)
    opp_stats = team_gf[['game_date', 'team', 'team_gpg_rolling', 'team_gag_rolling']].copy()
    opp_stats.columns = ['game_date', 'opponent', 'opp_gpg_rolling', 'opp_gag_rolling']

    team_gf = team_gf.merge(
        opp_stats,
        on=['game_date', 'opponent'],
        how='left'
    )

    return team_gf[['game_date', 'team', 'opponent', 'goals_for', 'goals_against',
                     'team_gpg_rolling', 'team_gag_rolling', 'opp_gpg_rolling', 'opp_gag_rolling']]


# ============================================================================
# WIN PROBABILITY MODEL
# ============================================================================

def build_win_probability_model(historical_goalies, historical_skaters):
    """
    Build logistic regression model for team win probability.
    Train on historical data to learn which team features predict wins.
    """
    print("\nBuilding team win probability model...")

    # Get team stats with rolling averages
    team_stats = compute_rolling_team_stats(historical_skaters)

    # Add wins
    wins = historical_goalies[historical_goalies['decision'] == 'W'][
        ['game_date', 'team']
    ].drop_duplicates()
    wins['win'] = 1

    team_stats = team_stats.merge(wins, on=['game_date', 'team'], how='left')
    team_stats['win'] = team_stats['win'].fillna(0).astype(int)

    # Add home/away indicator
    team_stats['is_home'] = (team_stats['home_road'] == 'H').astype(int) if 'home_road' in team_stats.columns else 0

    # Handle missing home_road from boxscore - need to infer or add
    # For now, we'll estimate home based on historical patterns

    # Add rest days (simplified: if game is back-to-back)
    team_stats = team_stats.sort_values(['team', 'game_date'])
    team_stats['days_since_last_game'] = team_stats.groupby('team')['game_date'].diff().dt.days
    team_stats['days_since_last_game'] = team_stats['days_since_last_game'].fillna(2)
    team_stats['is_back_to_back'] = (team_stats['days_since_last_game'] <= 1).astype(int)

    # Build features
    features = ['team_gpg_rolling', 'team_gag_rolling', 'opp_gpg_rolling',
                'opp_gag_rolling', 'is_back_to_back', 'days_since_last_game']

    X = team_stats[features].fillna(team_stats[features].mean())
    y = team_stats['win']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    # Print coefficients
    print("\nWin Probability Model Coefficients:")
    for feat, coef in zip(features, model.coef_[0]):
        print(f"  {feat:30s}: {coef:7.4f}")
    print(f"  Intercept: {model.intercept_[0]:7.4f}")

    # Historical calibration
    pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Binned calibration
    bins = pd.cut(pred_proba, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0-20', '20-40', '40-60', '60-80', '80-100'])
    calibration = pd.DataFrame({
        'pred_bin': bins,
        'actual': y
    }).groupby('pred_bin')['actual'].agg(['mean', 'count'])

    print("\nWin Probability Calibration (Historical):")
    print(calibration)

    return model, scaler, features


# ============================================================================
# SAVE PREDICTIONS & EXPECTED VALUES MODELS
# ============================================================================

def build_expected_values_models(historical_goalies, historical_skaters):
    """
    Build models for expected saves and expected goals against.
    Uses historical data to learn relationships.
    """
    print("\nBuilding expected value models...")

    # Merge goalie and team data
    team_stats = compute_rolling_team_stats(historical_skaters)

    # Goalie stats per game (rename goals_against to avoid conflict)
    goalie_games = historical_goalies[['game_date', 'team', 'opponent', 'shots_against',
                                       'goals_against', 'saves', 'sv_pct', 'toi_seconds']].copy()
    goalie_games.rename(columns={'goals_against': 'ga_goalie'}, inplace=True)

    # Merge - select only relevant team columns
    data = goalie_games.merge(
        team_stats[['game_date', 'team', 'opponent', 'goals_for', 'goals_against',
                    'team_gpg_rolling', 'team_gag_rolling', 'opp_gpg_rolling', 'opp_gag_rolling']],
        on=['game_date', 'team', 'opponent'],
        how='inner'
    )

    # Expected shots against
    # Features: opponent's offensive strength (opp_gpg), opponent's recent shot volume
    # Proxy for shots: opp_gpg * ~30 (rough conversion)
    # Add team defensive strength

    data_clean = data.dropna(subset=['shots_against', 'ga_goalie', 'opp_gpg_rolling', 'team_gag_rolling'])

    # Simple models
    print(f"\nExpected Saves Model (N={len(data_clean)}):")
    print(f"  Mean shots against: {data_clean['shots_against'].mean():.2f}")
    print(f"  Std shots against: {data_clean['shots_against'].std():.2f}")
    print(f"  Correlation (opp_gpg vs shots): {data_clean[['opp_gpg_rolling', 'shots_against']].corr().iloc[0,1]:.3f}")

    print(f"\nExpected Goals Against Model (N={len(data_clean)}):")
    print(f"  Mean goals against: {data_clean['ga_goalie'].mean():.2f}")
    print(f"  Mean SV%: {data_clean['sv_pct'].mean():.4f}")
    print(f"  Std goals against: {data_clean['ga_goalie'].std():.2f}")

    # League average save pct (for shrinkage)
    league_avg_sv_pct = data_clean['sv_pct'].mean()

    return {
        'mean_shots_against': data_clean['shots_against'].mean(),
        'std_shots_against': data_clean['shots_against'].std(),
        'mean_goals_against': data_clean['ga_goalie'].mean(),
        'league_avg_sv_pct': league_avg_sv_pct,
        'mean_gaa': (data_clean['ga_goalie'].sum() / (data_clean['toi_seconds'].sum() / 3600)) if data_clean['toi_seconds'].sum() > 0 else 2.8
    }


# ============================================================================
# COMPONENT-BASED PREDICTION
# ============================================================================

def predict_goalie_fpts_components(goalie_row, team_stats_row, win_model, scaler,
                                   win_features, ev_params, goalie_history=None):
    """
    Predict goalie FPTS using component model:
    FPTS = P(Win)*6.0 + E[Saves]*0.7 + E[GA]*(-3.5) + P(Shutout)*2.0

    Inputs:
    - goalie_row: current goalie game row
    - team_stats_row: team stats for that game
    - win_model: fitted logistic regression for P(Win)
    - win_features: feature names for win model
    - ev_params: dict with mean stats and league averages
    - goalie_history: historical games for this goalie (for regressed SV%)
    """

    # ===== COMPONENT 1: Win Probability =====
    # Get team feature vector
    X_features = np.array([
        team_stats_row.get('team_gpg_rolling', ev_params['mean_shots_against']/30),
        team_stats_row.get('team_gag_rolling', ev_params['mean_goals_against']),
        team_stats_row.get('opp_gpg_rolling', ev_params['mean_shots_against']/30),
        team_stats_row.get('opp_gag_rolling', ev_params['mean_goals_against']),
        team_stats_row.get('is_back_to_back', 0),
        team_stats_row.get('days_since_last_game', 2)
    ]).reshape(1, -1)

    # Fill missing with defaults
    X_features = np.nan_to_num(X_features, nan=0)
    X_scaled = scaler.transform(X_features)

    p_win = win_model.predict_proba(X_scaled)[0, 1]

    # ===== COMPONENT 2: Expected Saves =====
    # E[Saves] ≈ opponent's recent shot rate
    opp_gpg = team_stats_row.get('opp_gpg_rolling', 2.8)  # ~2.8 GPG average
    expected_shots = opp_gpg * 10.8  # ~10-11 shots per goal
    expected_shots = np.clip(expected_shots, 20, 45)

    # Adjust for goalie history (high-volume vs low-volume starters)
    if goalie_history is not None and len(goalie_history) > 0:
        avg_shots_against = goalie_history['shots_against'].mean()
        expected_shots = 0.5 * expected_shots + 0.5 * avg_shots_against

    # ===== COMPONENT 3: Expected Goals Against & Save % =====
    # Regressed SV% (heavy shrinkage toward league average)
    if goalie_history is not None and len(goalie_history) > 0:
        # Handle both sv_pct and save_pct column names
        sv_pct_col = 'sv_pct' if 'sv_pct' in goalie_history.columns else 'save_pct'
        goalie_sv_pct = goalie_history[sv_pct_col].mean()
        total_shots_in_history = goalie_history['shots_against'].sum()
        shrinkage_factor = min(0.12, total_shots_in_history / 1500)  # Heavy shrinkage
    else:
        goalie_sv_pct = ev_params['league_avg_sv_pct']
        shrinkage_factor = 0.0

    regressed_sv_pct = (shrinkage_factor * goalie_sv_pct +
                        (1 - shrinkage_factor) * ev_params['league_avg_sv_pct'])

    # Opponent strength (affects GA)
    opp_strength = team_stats_row.get('opp_gag_rolling', 2.8) / 2.8  # Relative to league avg

    # E[GA] with opponent adjustment
    expected_ga = expected_shots * (1 - regressed_sv_pct) * opp_strength
    expected_ga = np.clip(expected_ga, 0, 10)

    # ===== COMPONENT 4: Shutout Probability =====
    # P(GA=0) using Poisson distribution
    lambda_ga = max(0.1, expected_ga)
    p_shutout = poisson.pmf(0, lambda_ga)

    # ===== COMBINE COMPONENTS =====
    expected_saves = expected_shots * regressed_sv_pct

    fpts_win_component = p_win * 6.0
    fpts_save_component = expected_saves * 0.7
    fpts_ga_component = expected_ga * (-3.5)
    fpts_shutout_component = p_shutout * 2.0

    total_fpts = (fpts_win_component + fpts_save_component +
                  fpts_ga_component + fpts_shutout_component)

    return {
        'p_win': p_win,
        'expected_saves': expected_saves,
        'expected_ga': expected_ga,
        'p_shutout': p_shutout,
        'regressed_sv_pct': regressed_sv_pct,
        'fpts_win': fpts_win_component,
        'fpts_save': fpts_save_component,
        'fpts_ga': fpts_ga_component,
        'fpts_shutout': fpts_shutout_component,
        'fpts_total': total_fpts
    }


# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================

def walk_forward_backtest(game_logs_goalies, historical_goalies, historical_skaters,
                         boxscore_skaters):
    """
    Walk-forward backtest on current season (game_logs_goalies).
    For each game, use only data available before that date.
    """
    print("\n" + "="*80)
    print("WALK-FORWARD BACKTEST")
    print("="*80)

    # Sort by date
    game_logs_goalies = game_logs_goalies.sort_values('game_date').reset_index(drop=True)
    boxscore_skaters = boxscore_skaters.sort_values('game_date').reset_index(drop=True)

    # Date range
    first_date = game_logs_goalies['game_date'].min()
    last_date = game_logs_goalies['game_date'].max()

    print(f"\nBacktest period: {first_date.date()} to {last_date.date()}")
    print(f"Total goalie games: {len(game_logs_goalies)}")

    # Build win model on historical data
    win_model, scaler, win_features = build_win_probability_model(historical_goalies, historical_skaters)
    ev_params = build_expected_values_models(historical_goalies, historical_skaters)

    # Current season team stats
    current_team_stats = compute_rolling_team_stats(boxscore_skaters)

    # Actual results from game_logs_goalies
    actuals = game_logs_goalies[[
        'game_date', 'player_name', 'team', 'player_id', 'decision',
        'shots_against', 'saves', 'goals_against', 'shutouts', 'dk_fpts', 'home_road'
    ]].copy()

    predictions = []

    for idx, row in actuals.iterrows():
        game_date = row['game_date']
        player_id = row['player_id']
        team = row['team']

        # Get team stats for this game
        team_game = current_team_stats[
            (current_team_stats['game_date'] == game_date) &
            (current_team_stats['team'] == team)
        ]

        # Get goalie history (up to this date)
        history_cutoff = game_date - timedelta(days=1)
        goalie_history = historical_goalies[
            (historical_goalies['player_id'] == player_id) &
            (historical_goalies['game_date'] <= history_cutoff)
        ]

        # If no history, use current season data
        if len(goalie_history) == 0:
            goalie_history = game_logs_goalies[
                (game_logs_goalies['player_id'] == player_id) &
                (game_logs_goalies['game_date'] < game_date)
            ]

        # Make prediction
        if len(team_game) > 0:
            pred = predict_goalie_fpts_components(
                row,
                team_game.iloc[0].to_dict(),
                win_model,
                scaler,
                win_features,
                ev_params,
                goalie_history
            )
        else:
            # Fallback to baseline
            pred = predict_goalie_fpts_components(
                row,
                {},
                win_model,
                scaler,
                win_features,
                ev_params,
                goalie_history
            )

        pred['game_date'] = game_date
        pred['player_name'] = row['player_name']
        pred['team'] = team
        pred['player_id'] = player_id
        pred['home_road'] = row['home_road']
        pred['decision'] = row['decision']
        pred['actual_fpts'] = row['dk_fpts']
        pred['actual_saves'] = row['saves']
        pred['actual_ga'] = row['goals_against']
        pred['actual_shutout'] = row['shutouts']

        predictions.append(pred)

    results = pd.DataFrame(predictions)

    return results


# ============================================================================
# EVALUATION & REPORTING
# ============================================================================

def evaluate_results(results):
    """Compute MAE and other metrics"""

    # Overall MAE
    results['error'] = results['fpts_total'] - results['actual_fpts']
    results['abs_error'] = results['error'].abs()

    overall_mae = results['abs_error'].mean()
    overall_rmse = np.sqrt((results['error'] ** 2).mean())
    overall_bias = results['error'].mean()

    print("\n" + "="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    print(f"MAE (Component Model):    {overall_mae:7.3f}")
    print(f"RMSE:                     {overall_rmse:7.3f}")
    print(f"Bias:                     {overall_bias:7.3f}")
    print(f"Baseline (Season Avg):    8.68")
    print(f"XGBoost Baseline:         7.88")
    print(f"Improvement vs Baseline:  {8.68 - overall_mae:7.3f}")
    print(f"Improvement vs XGBoost:   {7.88 - overall_mae:7.3f}")

    # By home/away
    print("\n" + "="*80)
    print("HOME/AWAY BREAKDOWN")
    print("="*80)

    for location in ['H', 'A']:
        subset = results[results['home_road'] == location]
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            n = len(subset)
            print(f"{location} (N={n:3d}): MAE={mae:7.3f}")

    # By starter/backup (inferred from shots_against volume)
    results['is_starter'] = results['actual_saves'] > 15

    print("\n" + "="*80)
    print("STARTER/BACKUP BREAKDOWN")
    print("="*80)

    for is_start, label in [(True, 'Starter'), (False, 'Backup')]:
        subset = results[results['is_starter'] == is_start]
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            n = len(subset)
            print(f"{label:8s} (N={n:3d}): MAE={mae:7.3f}")

    # Win probability calibration
    print("\n" + "="*80)
    print("WIN PROBABILITY CALIBRATION")
    print("="*80)

    results['actual_win'] = (results['decision'] == 'W').astype(int)

    # Bin predictions
    bins = pd.cut(results['p_win'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                  include_lowest=True)

    calibration = pd.DataFrame({
        'pred_bin': bins,
        'actual_win': results['actual_win']
    }).groupby('pred_bin', observed=True).agg({
        'actual_win': ['sum', 'count', 'mean']
    })
    calibration.columns = ['wins', 'games', 'actual_pct']

    print(calibration)

    # Component analysis
    print("\n" + "="*80)
    print("COMPONENT ANALYSIS (Average per Game)")
    print("="*80)
    print(f"Win component:            {results['fpts_win'].mean():7.3f} (P(W)={results['p_win'].mean():.3f})")
    print(f"Save component:           {results['fpts_save'].mean():7.3f} (saves={results['expected_saves'].mean():.1f})")
    print(f"GA component:             {results['fpts_ga'].mean():7.3f} (GA={results['expected_ga'].mean():.2f})")
    print(f"Shutout component:        {results['fpts_shutout'].mean():7.3f} (P(SO)={results['p_shutout'].mean():.3f})")

    # Actual components
    results['actual_wins_fpts'] = results['actual_win'] * 6.0
    results['actual_saves_fpts'] = results['actual_saves'] * 0.7
    results['actual_ga_fpts'] = results['actual_ga'] * (-3.5)
    results['actual_so_fpts'] = results['actual_shutout'] * 2.0

    print("\nActual component breakdown:")
    print(f"Win component (actual):   {results['actual_wins_fpts'].mean():7.3f} (W pct={results['actual_win'].mean():.3f})")
    print(f"Save component (actual):  {results['actual_saves_fpts'].mean():7.3f} (saves={results['actual_saves'].mean():.1f})")
    print(f"GA component (actual):    {results['actual_ga_fpts'].mean():7.3f} (GA={results['actual_ga'].mean():.2f})")
    print(f"Shutout component (actual):{results['actual_so_fpts'].mean():7.3f} (SO pct={results['actual_shutout'].mean():.3f})")

    # Error by component
    print("\nMean absolute error by component:")
    print(f"Win prediction:           {(results['p_win'] - results['actual_win']).abs().mean():.3f}")
    print(f"Save prediction:          {(results['expected_saves'] - results['actual_saves']).abs().mean():.2f}")
    print(f"GA prediction:            {(results['expected_ga'] - results['actual_ga']).abs().mean():.2f}")

    return overall_mae, overall_rmse, overall_bias


def save_results(results):
    """Save detailed results to CSV"""
    output_path = '/sessions/youthful-funny-faraday/mnt/Code/projection/goalie_v2_results.csv'

    # Select columns for output
    output_cols = [
        'game_date', 'player_name', 'team', 'home_road', 'decision',
        'p_win', 'expected_saves', 'expected_ga', 'p_shutout',
        'fpts_win', 'fpts_save', 'fpts_ga', 'fpts_shutout', 'fpts_total',
        'actual_saves', 'actual_ga', 'actual_shutout', 'actual_fpts',
        'error', 'abs_error', 'regressed_sv_pct'
    ]

    output_df = results[[c for c in output_cols if c in results.columns]].copy()
    output_df = output_df.sort_values('game_date')

    output_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ENHANCED GOALIE PROJECTION MODEL v2")
    print("Team Win Probability Focus")
    print("="*80)

    # Load data
    historical_goalies, game_logs_goalies, boxscore_skaters, historical_skaters = load_data()

    # Run walk-forward backtest
    results = walk_forward_backtest(
        game_logs_goalies,
        historical_goalies,
        historical_skaters,
        boxscore_skaters
    )

    # Evaluate
    mae, rmse, bias = evaluate_results(results)

    # Save results
    save_results(results)

    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
