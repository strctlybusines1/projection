#!/usr/bin/env python3
"""
Goalie Projection Model for DFS
================================

Multi-component approach:
1. Win Probability Model (logistic regression)
2. Saves/Goals Against Model (linear regression from workload)
3. Combined FPTS prediction + XGBoost end-to-end model
4. Walk-forward backtest on 2024-25 season
5. Year-over-year regression analysis for goalie stat stability

DK Goalie Scoring: Win=+6.0, Save=+0.7, GA=-3.5, Shutout=+2.0, G/A=+8.5/+5.0
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Skipping XGBoost model.")


# ============================================================================
# CONFIG
# ============================================================================

DB_PATH = Path("/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db")
DK_SCORING = {
    'win': 6.0,
    'save': 0.7,
    'ga': -3.5,
    'shutout': 2.0,
    'goal': 8.5,
    'assist': 5.0,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_goalie_data():
    """Load and combine historical + current season goalie data."""
    conn = sqlite3.connect(DB_PATH)

    # Historical data
    hist = pd.read_sql("SELECT * FROM historical_goalies", conn)
    hist['source'] = 'historical'
    hist['season'] = hist['season'].astype(int)

    # Current season
    current = pd.read_sql("SELECT * FROM game_logs_goalies", conn)
    current['source'] = 'current'
    current['season'] = 2025

    # Standardize columns
    hist = hist[[
        'season', 'player_id', 'player_name', 'team', 'game_id', 'game_date',
        'opponent', 'home_road', 'decision', 'shots_against', 'saves',
        'goals_against', 'sv_pct', 'toi', 'toi_seconds', 'dk_fpts', 'goals',
        'assists', 'source'
    ]]

    current = current[[
        'season', 'player_id', 'player_name', 'team', 'game_id', 'game_date',
        'opponent', 'home_road', 'decision', 'shots_against', 'saves',
        'goals_against', 'save_pct', 'toi', 'toi_seconds', 'dk_fpts', 'goals',
        'assists', 'source'
    ]].copy()

    # Rename save_pct to sv_pct for consistency
    current.rename(columns={'save_pct': 'sv_pct'}, inplace=True)

    # Combine
    df = pd.concat([hist, current], ignore_index=True)

    # Clean dates
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Sort by date
    df = df.sort_values('game_date').reset_index(drop=True)

    conn.close()
    return df


def load_skater_data():
    """Load skater data for opponent feature aggregation."""
    conn = sqlite3.connect(DB_PATH)

    # Historical skaters
    hist = pd.read_sql("SELECT * FROM historical_skaters", conn)
    hist['source'] = 'historical'
    hist['season'] = hist['season'].astype(int)

    # Current season
    current = pd.read_sql("SELECT * FROM game_logs_skaters", conn)
    current['source'] = 'current'
    current['season'] = 2025

    # Standardize
    hist = hist[[
        'season', 'player_id', 'player_name', 'team', 'game_id', 'game_date',
        'opponent', 'home_road', 'goals', 'shots', 'dk_fpts', 'position'
    ]]

    current = current[[
        'season', 'player_id', 'player_name', 'team', 'game_id', 'game_date',
        'opponent', 'home_road', 'goals', 'shots', 'dk_fpts'
    ]].copy()
    current['position'] = 'UNK'  # Not available in current

    skaters = pd.concat([hist, current], ignore_index=True)
    skaters['game_date'] = pd.to_datetime(skaters['game_date'])
    skaters = skaters.sort_values('game_date').reset_index(drop=True)

    conn.close()
    return skaters


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_rolling_stats(df, group_col, value_col, windows=[5, 10], prefix=''):
    """Compute rolling averages."""
    result = df.copy()

    for window in windows:
        col_name = f"{prefix}{value_col}_last{window}"
        result[col_name] = (
            result.groupby(group_col)[value_col]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

    return result


def compute_win_rate(df, group_col, windows=[5, 10], prefix=''):
    """Compute rolling win rate (decision == 'W')."""
    result = df.copy()
    result['is_win'] = (result['decision'] == 'W').astype(int)

    for window in windows:
        col_name = f"{prefix}win_rate_last{window}"
        result[col_name] = (
            result.groupby(group_col)['is_win']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

    return result


def compute_ewm_stats(df, group_col, value_col, halflife=8, prefix=''):
    """Exponentially weighted moving average."""
    result = df.copy()
    col_name = f"{prefix}{value_col}_ewm"
    result[col_name] = (
        result.groupby(group_col)[value_col]
        .ewm(halflife=halflife, ignore_na=True)
        .mean()
        .reset_index(drop=True)
    )
    return result


def engineer_features(goalies_df, skaters_df):
    """Engineer all features for goalie prediction."""

    print("Engineering goalie features...")
    df = goalies_df.copy()

    # === GOALIE STATS ===
    df = compute_rolling_stats(df, 'player_id', 'sv_pct', windows=[5, 10], prefix='goalie_')
    df = compute_rolling_stats(df, 'player_id', 'goals_against', windows=[5, 10], prefix='goalie_')
    df = compute_rolling_stats(df, 'player_id', 'saves', windows=[5, 10], prefix='goalie_')
    df = compute_rolling_stats(df, 'player_id', 'shots_against', windows=[5, 10], prefix='goalie_')
    df = compute_win_rate(df, 'player_id', windows=[5, 10], prefix='goalie_')
    df = compute_ewm_stats(df, 'player_id', 'dk_fpts', halflife=8, prefix='goalie_')

    # Expanding mean (season-to-date average)
    df['goalie_fpts_season_avg'] = (
        df.groupby(['player_id', 'season'])['dk_fpts']
        .expanding(min_periods=1)
        .mean()
        .reset_index(drop=True)
    )

    # Starts count
    df['goalie_starts'] = (
        df.groupby(['player_id', 'season'])
        .cumcount() + 1
    )

    # Home/Away binary
    df['is_home'] = (df['home_road'] == 'H').astype(int)

    # === TEAM STATS (goalie's team) ===
    team_games = skaters_df.groupby(['team', 'game_date', 'opponent']).agg({
        'goals': 'sum',
        'shots': 'sum',
        'dk_fpts': 'sum',
    }).reset_index()

    team_games = compute_rolling_stats(
        team_games, 'team', 'goals', windows=[10], prefix='team_'
    )
    team_games = compute_rolling_stats(
        team_games, 'team', 'dk_fpts', windows=[10], prefix='team_'
    )

    team_games.rename(columns={
        'team_goals_last10': 'team_goals_per_game_last10',
        'team_dk_fpts_last10': 'team_fpts_per_game_last10',
    }, inplace=True)

    # Merge team stats onto goalie df
    df = df.merge(
        team_games[['team', 'game_date', 'team_goals_per_game_last10', 'team_fpts_per_game_last10']],
        on=['team', 'game_date'],
        how='left'
    )

    # === OPPONENT STATS ===
    opp_games = skaters_df.groupby(['team', 'game_date']).agg({
        'goals': 'sum',
        'shots': 'sum',
        'dk_fpts': 'sum',
    }).reset_index()

    opp_games.rename(columns={'team': 'opponent'}, inplace=True)

    opp_games = compute_rolling_stats(
        opp_games, 'opponent', 'goals', windows=[10], prefix='opp_'
    )
    opp_games = compute_rolling_stats(
        opp_games, 'opponent', 'shots', windows=[10], prefix='opp_'
    )
    opp_games = compute_rolling_stats(
        opp_games, 'opponent', 'dk_fpts', windows=[10], prefix='opp_'
    )

    opp_games.rename(columns={
        'opp_goals_last10': 'opp_goals_per_game_last10',
        'opp_shots_last10': 'opp_shots_per_game_last10',
        'opp_dk_fpts_last10': 'opp_fpts_per_game_last10',
    }, inplace=True)

    # Merge opponent stats
    df = df.merge(
        opp_games[['opponent', 'game_date', 'opp_goals_per_game_last10',
                   'opp_shots_per_game_last10', 'opp_fpts_per_game_last10']],
        on=['opponent', 'game_date'],
        how='left'
    )

    # === OUTCOME FEATURES ===
    df['is_win'] = (df['decision'] == 'W').astype(int)
    df['is_shutout'] = (df['goals_against'] == 0).astype(int)

    return df


# ============================================================================
# MODELING
# ============================================================================

def prepare_training_data(df, cutoff_date=None):
    """Prepare data, remove rows with NaN in key features."""

    feature_cols = [
        'goalie_sv_pct_last5', 'goalie_sv_pct_last10',
        'goalie_goals_against_last5', 'goalie_goals_against_last10',
        'goalie_saves_last5', 'goalie_saves_last10',
        'goalie_shots_against_last5', 'goalie_shots_against_last10',
        'goalie_win_rate_last5', 'goalie_win_rate_last10',
        'goalie_dk_fpts_ewm', 'goalie_fpts_season_avg',
        'goalie_starts',
        'is_home',
        'team_goals_per_game_last10', 'team_fpts_per_game_last10',
        'opp_goals_per_game_last10', 'opp_shots_per_game_last10', 'opp_fpts_per_game_last10',
    ]

    train_df = df.copy()

    if cutoff_date is not None:
        train_df = train_df[train_df['game_date'] < cutoff_date].copy()

    # Remove rows with NaN
    train_df = train_df.dropna(subset=feature_cols + ['dk_fpts', 'is_win', 'goals_against', 'saves'])

    return train_df, feature_cols


def train_win_model(train_df, feature_cols):
    """Train logistic regression for win probability."""
    X = train_df[feature_cols].values
    y = train_df['is_win'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


def train_saves_model(train_df, feature_cols):
    """Train linear regression for expected saves."""
    X = train_df[feature_cols].values
    y = train_df['saves'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler


def train_ga_model(train_df, feature_cols):
    """Train linear regression for expected goals against."""
    X = train_df[feature_cols].values
    y = train_df['goals_against'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler


def train_xgboost_model(train_df, feature_cols):
    """Train XGBoost end-to-end model."""
    if not HAS_XGBOOST:
        return None, None

    X = train_df[feature_cols].values
    y = train_df['dk_fpts'].values

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)

    params = {
        'max_depth': 4,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'random_state': 42,
    }

    model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)

    return model, feature_cols


def compute_component_prediction(test_df, feature_cols, win_model, win_scaler,
                                 saves_model, saves_scaler, ga_model, ga_scaler):
    """
    Compute FPTS as components:
    FPTS = P(win)*6 + E[saves]*0.7 + E[GA]*(-3.5) + P(shutout)*2
    """
    X = test_df[feature_cols].values

    # Win probability
    X_scaled_win = win_scaler.transform(X)
    p_win = win_model.predict_proba(X_scaled_win)[:, 1]

    # Expected saves
    X_scaled_saves = saves_scaler.transform(X)
    e_saves = saves_model.predict(X_scaled_saves)
    e_saves = np.maximum(e_saves, 0)

    # Expected GA
    X_scaled_ga = ga_scaler.transform(X)
    e_ga = ga_model.predict(X_scaled_ga)
    e_ga = np.maximum(e_ga, 0)

    # Shutout probability (approximation: if GA=0)
    p_shutout = np.exp(-e_ga) * 0.5
    p_shutout = np.clip(p_shutout, 0, 1)

    # Combine
    predictions = (
        p_win * DK_SCORING['win'] +
        e_saves * DK_SCORING['save'] +
        e_ga * DK_SCORING['ga'] +
        p_shutout * DK_SCORING['shutout']
    )

    return predictions


def compute_xgboost_prediction(test_df, feature_cols, model):
    """Get XGBoost predictions."""
    if model is None:
        return None

    X = test_df[feature_cols].values
    dtest = xgb.DMatrix(X, feature_names=feature_cols)
    predictions = model.predict(dtest)

    return predictions


# ============================================================================
# BACKTEST
# ============================================================================

def walk_forward_backtest(df, refit_days=14):
    """
    Walk-forward backtest on 2024-25 season.
    Retrain every N days using only data before that date.
    """

    # Filter to current season
    df_current = df[df['season'] == 2025].copy()

    if len(df_current) == 0:
        print("No data for 2024-25 season!")
        return None

    print(f"\nWalk-Forward Backtest (2024-25 season: {df_current['game_date'].min()} to {df_current['game_date'].max()})")
    print(f"Total goalie games: {len(df_current)}")
    print("=" * 80)

    # Get unique dates
    dates = sorted(df_current['game_date'].unique())

    # Backtest loop
    results = {
        'date': [],
        'actual_fpts': [],
        'baseline_fpts': [],
        'component_fpts': [],
        'xgb_fpts': [],
    }

    last_refit = dates[0]
    models = None

    for test_date in dates:
        # Refit if needed
        if (test_date - last_refit).days >= refit_days or models is None:
            print(f"\nRetraining on data before {test_date.date()}...")

            # Prepare training data
            full_train_df = df[df['game_date'] < test_date].copy()
            train_df, feature_cols = prepare_training_data(full_train_df, cutoff_date=test_date)

            if len(train_df) < 50:
                print(f"  Insufficient training data ({len(train_df)} rows). Skipping.")
                continue

            print(f"  Training on {len(train_df)} goalie games")

            # Train models
            win_model, win_scaler = train_win_model(train_df, feature_cols)
            saves_model, saves_scaler = train_saves_model(train_df, feature_cols)
            ga_model, ga_scaler = train_ga_model(train_df, feature_cols)
            xgb_model, xgb_cols = train_xgboost_model(train_df, feature_cols)

            models = {
                'win_model': win_model,
                'win_scaler': win_scaler,
                'saves_model': saves_model,
                'saves_scaler': saves_scaler,
                'ga_model': ga_model,
                'ga_scaler': ga_scaler,
                'xgb_model': xgb_model,
                'xgb_cols': xgb_cols,
                'feature_cols': feature_cols,
            }

            last_refit = test_date

        # Test on this date
        test_games = df_current[df_current['game_date'] == test_date].copy()
        test_games_clean = test_games.dropna(subset=models['feature_cols'] + ['dk_fpts'])

        if len(test_games_clean) == 0:
            continue

        # Predictions
        baseline = test_games_clean['goalie_fpts_season_avg'].fillna(test_games_clean['dk_fpts'].mean()).values

        component = compute_component_prediction(
            test_games_clean,
            models['feature_cols'],
            models['win_model'], models['win_scaler'],
            models['saves_model'], models['saves_scaler'],
            models['ga_model'], models['ga_scaler'],
        )

        xgb_pred = compute_xgboost_prediction(
            test_games_clean,
            models['feature_cols'],
            models['xgb_model']
        )
        if xgb_pred is None:
            xgb_pred = baseline

        # Store
        results['date'].extend([test_date] * len(test_games_clean))
        results['actual_fpts'].extend(test_games_clean['dk_fpts'].values)
        results['baseline_fpts'].extend(baseline)
        results['component_fpts'].extend(component)
        results['xgb_fpts'].extend(xgb_pred)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No results from backtest!")
        return None

    # Compute errors
    results_df['baseline_error'] = np.abs(results_df['actual_fpts'] - results_df['baseline_fpts'])
    results_df['component_error'] = np.abs(results_df['actual_fpts'] - results_df['component_fpts'])
    results_df['xgb_error'] = np.abs(results_df['actual_fpts'] - results_df['xgb_fpts'])

    # Print summary
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS (MAE = Mean Absolute Error)")
    print("=" * 80)

    baseline_mae = results_df['baseline_error'].mean()
    component_mae = results_df['component_error'].mean()
    xgb_mae = results_df['xgb_error'].mean()

    print(f"\nBaseline (Season Avg):")
    print(f"  MAE: {baseline_mae:.2f}")
    print(f"  Std: {results_df['baseline_error'].std():.2f}")

    print(f"\nComponent Model (Win + Saves + GA):")
    print(f"  MAE: {component_mae:.2f}")
    print(f"  Std: {results_df['component_error'].std():.2f}")
    if baseline_mae > 0:
        print(f"  Improvement: {(baseline_mae - component_mae) / baseline_mae * 100:.1f}%")

    print(f"\nXGBoost Model:")
    print(f"  MAE: {xgb_mae:.2f}")
    print(f"  Std: {results_df['xgb_error'].std():.2f}")
    if baseline_mae > 0:
        print(f"  Improvement: {(baseline_mae - xgb_mae) / baseline_mae * 100:.1f}%")

    # By position
    print("\n" + "-" * 80)
    print("BACKTEST BY STARTER/BACKUP")
    print("-" * 80)

    df_test = df_current[df_current['game_date'].isin(results_df['date'])].copy()
    df_test = df_test.merge(
        results_df[['date', 'actual_fpts', 'baseline_fpts', 'component_fpts', 'xgb_fpts']],
        left_on='game_date', right_on='date',
        how='inner'
    )

    # Categorize by starts
    df_test['position_type'] = pd.cut(df_test['goalie_starts'],
                                       bins=[0, 15, np.inf],
                                       labels=['Backup', 'Starter'])

    for pos in ['Backup', 'Starter']:
        subset = df_test[df_test['position_type'] == pos]
        if len(subset) > 0:
            print(f"\n{pos} ({len(subset)} games):")
            print(f"  Baseline MAE: {(subset['actual_fpts'] - subset['baseline_fpts']).abs().mean():.2f}")
            print(f"  Component MAE: {(subset['actual_fpts'] - subset['component_fpts']).abs().mean():.2f}")
            print(f"  XGBoost MAE: {(subset['actual_fpts'] - subset['xgb_fpts']).abs().mean():.2f}")

    # Home vs Away
    print("\n" + "-" * 80)
    print("BACKTEST BY HOME/AWAY")
    print("-" * 80)

    for ha in ['H', 'A']:
        subset = df_test[df_test['home_road'] == ha]
        if len(subset) > 0:
            location = 'Home' if ha == 'H' else 'Away'
            print(f"\n{location} ({len(subset)} games):")
            print(f"  Baseline MAE: {(subset['actual_fpts'] - subset['baseline_fpts']).abs().mean():.2f}")
            print(f"  Component MAE: {(subset['actual_fpts'] - subset['component_fpts']).abs().mean():.2f}")
            print(f"  XGBoost MAE: {(subset['actual_fpts'] - subset['xgb_fpts']).abs().mean():.2f}")

    # Win vs Loss prediction
    print("\n" + "-" * 80)
    print("BACKTEST BY ACTUAL RESULT")
    print("-" * 80)

    for decision in ['W', 'L']:
        subset = df_test[df_test['decision'] == decision]
        if len(subset) > 0:
            print(f"\n{decision} ({len(subset)} games):")
            print(f"  Baseline MAE: {(subset['actual_fpts'] - subset['baseline_fpts']).abs().mean():.2f}")
            print(f"  Component MAE: {(subset['actual_fpts'] - subset['component_fpts']).abs().mean():.2f}")
            print(f"  XGBoost MAE: {(subset['actual_fpts'] - subset['xgb_fpts']).abs().mean():.2f}")

    return results_df


# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================

def yoy_regression_analysis(df):
    """Analyze year-over-year correlation of goalie stats."""

    print("\n" + "=" * 80)
    print("YEAR-OVER-YEAR REGRESSION ANALYSIS")
    print("=" * 80)

    # Get season averages per goalie
    season_stats = df.groupby(['player_id', 'player_name', 'season']).agg({
        'sv_pct': 'mean',
        'goals_against': 'mean',
        'saves': 'mean',
        'dk_fpts': 'mean',
        'is_win': 'mean',
    }).reset_index()

    season_stats.columns = ['player_id', 'player_name', 'season', 'sv_pct', 'gaa', 'saves_per_game', 'fpts_per_start', 'win_rate']

    # Merge consecutive seasons
    pivot_sv_pct = season_stats.pivot(index='player_id', columns='season', values='sv_pct')
    pivot_gaa = season_stats.pivot(index='player_id', columns='season', values='gaa')
    pivot_fpts = season_stats.pivot(index='player_id', columns='season', values='fpts_per_start')
    pivot_win = season_stats.pivot(index='player_id', columns='season', values='win_rate')

    # Compute YoY correlations
    print("\nYear-over-Year Correlations (can goalies' stats from one season predict next season):\n")

    seasons = sorted(pivot_sv_pct.columns)
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]

        # SV%
        mask = pivot_sv_pct[[s1, s2]].notna().all(axis=1)
        corr_sv = pivot_sv_pct.loc[mask, [s1, s2]].corr().iloc[0, 1] if mask.sum() > 2 else np.nan

        # GAA
        mask = pivot_gaa[[s1, s2]].notna().all(axis=1)
        corr_gaa = pivot_gaa.loc[mask, [s1, s2]].corr().iloc[0, 1] if mask.sum() > 2 else np.nan

        # FPTS
        mask = pivot_fpts[[s1, s2]].notna().all(axis=1)
        corr_fpts = pivot_fpts.loc[mask, [s1, s2]].corr().iloc[0, 1] if mask.sum() > 2 else np.nan

        # Win Rate
        mask = pivot_win[[s1, s2]].notna().all(axis=1)
        corr_win = pivot_win.loc[mask, [s1, s2]].corr().iloc[0, 1] if mask.sum() > 2 else np.nan

        print(f"{s1}-{s2}:")
        print(f"  SV%:           r = {corr_sv:7.3f}  (indicates goalie skill carries over)")
        print(f"  GAA:           r = {corr_gaa:7.3f}  (defensive context matters)")
        print(f"  FPTS/Start:    r = {corr_fpts:7.3f}  (overall DFS performance)")
        print(f"  Win Rate:      r = {corr_win:7.3f}  (most team-dependent)")

    # Regression: can we predict next season from this season?
    print("\n" + "-" * 80)
    print("Simple Regression: Last Season → Current Season Projections")
    print("-" * 80)

    for metric in ['sv_pct', 'gaa', 'fpts_per_start', 'win_rate']:
        # Use 2024 to predict 2025
        s1_data = season_stats[season_stats['season'] == 2024][[f'{metric}', 'player_id']].copy()
        s2_data = season_stats[season_stats['season'] == 2025][[f'{metric}', 'player_id']].copy()

        s1_data.columns = ['last_year', 'player_id']
        s2_data.columns = ['this_year', 'player_id']

        combined = s1_data.merge(s2_data, on='player_id', how='inner')

        if len(combined) > 2:
            X = combined['last_year'].values.reshape(-1, 1)
            y = combined['this_year'].values

            reg = LinearRegression()
            reg.fit(X, y)

            r_sq = reg.score(X, y)
            coef = reg.coef_[0]
            intercept = reg.intercept_

            print(f"\n{metric.upper()}:")
            print(f"  R²: {r_sq:.3f}")
            print(f"  Regression: next_year = {intercept:.3f} + {coef:.3f} * last_year")
            if coef < 1.0:
                print(f"  → Regression to mean: {(1 - coef) * 100:.1f}% shrinkage")


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    """Main entry point."""

    print("Loading data...")
    goalies_df = load_goalie_data()
    skaters_df = load_skater_data()

    print(f"Loaded {len(goalies_df)} goalie games")
    print(f"Loaded {len(skaters_df)} skater games")
    print(f"\nGoalie data range: {goalies_df['game_date'].min()} to {goalies_df['game_date'].max()}")

    # Check data
    print(f"\nGoalie seasons: {sorted(goalies_df['season'].unique())}")
    print(f"Goalies in dataset: {goalies_df['player_id'].nunique()}")
    print(f"\nCurrent season (2024-25) games: {len(goalies_df[goalies_df['season'] == 2025])}")

    # Feature engineering
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    df_features = engineer_features(goalies_df, skaters_df)

    print(f"Final dataset: {len(df_features)} rows")

    # Missing data check
    print("\nMissing data summary:")
    feature_cols = [
        'goalie_sv_pct_last5', 'goalie_sv_pct_last10',
        'goalie_goals_against_last5', 'goalie_goals_against_last10',
        'goalie_saves_last5', 'goalie_saves_last10',
        'goalie_shots_against_last5', 'goalie_shots_against_last10',
        'goalie_win_rate_last5', 'goalie_win_rate_last10',
        'goalie_dk_fpts_ewm', 'goalie_fpts_season_avg',
        'goalie_starts',
        'is_home',
        'team_goals_per_game_last10', 'team_fpts_per_game_last10',
        'opp_goals_per_game_last10', 'opp_shots_per_game_last10', 'opp_fpts_per_game_last10',
    ]

    for col in feature_cols:
        if col in df_features.columns:
            missing = df_features[col].isna().sum()
            if missing > 0:
                pct = missing / len(df_features) * 100
                print(f"  {col}: {missing} ({pct:.1f}%)")
        else:
            print(f"  {col}: NOT FOUND")

    # Backtest if requested
    if args.backtest:
        results_df = walk_forward_backtest(df_features, refit_days=14)

    # YoY regression analysis
    print("\n")
    yoy_regression_analysis(df_features)

    # Save full dataset
    output_path = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/goalie_model_data.csv')
    df_features.to_csv(output_path, index=False)
    print(f"\nFeature dataset saved to {output_path}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Goalie Projection Model')
    parser.add_argument('--backtest', action='store_true', help='Run walk-forward backtest')
    args = parser.parse_args()

    main(args)
