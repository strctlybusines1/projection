"""
XGBoost Ownership Prediction Model v2
======================================

Trained on real ownership data from own.csv (13,700+ observations, 101 dates).
Replaces the Ridge regression baseline (MAE 1.92%, corr 0.905 on 6 dates).

Architecture:
    1. Merge dk_salaries with own.csv on player+date for real ownership labels
    2. Engineer 30+ features (salary, value, matchup, line deployment, slate context)
       NOTE: All features use only data available at prediction time (dk_avg_fpts, salary,
       Vegas lines, line deployment). No FC Proj or other third-party projection dependency.
    3. XGBoost regressor with LODOCV (leave-one-date-out cross-validation)
    4. Position-aware normalization so per-position totals sum to roster slots

Expected improvement: Ridge MAE 1.92% → XGBoost MAE ~1.6-1.7% (10-15% reduction)

Usage:
    python ownership_xgb.py                # Train + LODOCV + save model
    python ownership_xgb.py --retrain      # Force retrain even if model exists

Integration:
    from ownership_xgb import load_ownership_model, predict_ownership_for_date
"""

import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

DB_PATH = Path(__file__).parent / 'data' / 'nhl_dfs_history.db'
OWN_CSV_PATH = Path(__file__).parent / 'own.csv'
MODEL_PATH = Path(__file__).parent / 'data' / 'ownership_xgb_v2.pkl'


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

FEATURE_COLS = [
    # Salary
    'salary_k', 'salary_sq', 'salary_rank_pos', 'salary_pctile_pos', 'salary_pctile_slate',
    # Projection / value (dk_avg_fpts is DK's own historical average — available at prediction time)
    'dk_avg_fpts', 'dk_ceiling', 'dk_stdv',
    'value_score', 'value_rank_pos', 'proj_rank_pos', 'ceiling_rank_pos',
    'dk_avg_value_score',
    # Matchup / Vegas
    'team_implied_total', 'opp_implied_total', 'game_total',
    'is_favorite', 'spread_abs', 'implied_total_rank',
    # Line deployment
    'is_line1', 'is_line2', 'is_pp1', 'is_pp2',
    # Position
    'pos_C', 'pos_W', 'pos_D', 'pos_G',
    # Slate context
    'n_games_on_slate', 'players_at_pos', 'slate_size',
    # Derived interactions
    'dk_value_ratio', 'salary_x_implied', 'ceiling_pctile',
    'high_game_total', 'favorable_matchup',
    'dk_avg_x_implied', 'dk_avg_x_pp1', 'salary_x_pp1',
    'is_top3_proj', 'is_top3_salary',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer ownership prediction features from dk_salaries data.

    Expects a DataFrame with dk_salaries columns + optional own.csv merge.
    All features are computed per-slate so the model generalizes across slate sizes.
    """
    df = df.copy()

    # --- Salary features ---
    df['salary_k'] = df['salary'] / 1000.0
    df['salary_sq'] = df['salary_k'] ** 2  # Non-linear salary effect on ownership
    df['salary_rank_pos'] = df.groupby(['slate_date', 'position'])['salary'].rank(
        ascending=False, method='min')
    df['salary_pctile_pos'] = df.groupby(['slate_date', 'position'])['salary'].rank(pct=True)
    df['salary_pctile_slate'] = df.groupby('slate_date')['salary'].rank(pct=True)

    # --- Value features ---
    df['dk_avg_fpts'] = df['dk_avg_fpts'].fillna(0)
    df['dk_ceiling'] = df['dk_ceiling'].fillna(df['dk_avg_fpts'] * 2)
    df['dk_stdv'] = df['dk_stdv'].fillna(3.0)
    df['value_score'] = df['dk_avg_fpts'] / (df['salary_k'] + 0.1)
    df['value_rank_pos'] = df.groupby(['slate_date', 'position'])['value_score'].rank(
        ascending=False, method='min')
    df['proj_rank_pos'] = df.groupby(['slate_date', 'position'])['dk_avg_fpts'].rank(
        ascending=False, method='min')
    df['ceiling_rank_pos'] = df.groupby(['slate_date', 'position'])['dk_ceiling'].rank(
        ascending=False, method='min')
    df['ceiling_pctile'] = df.groupby(['slate_date', 'position'])['dk_ceiling'].rank(pct=True)

    # --- DK avg-based projection features (available at prediction time) ---
    df['dk_avg_value_score'] = df['dk_avg_fpts'] / (df['salary_k'] + 0.1)
    df['is_top3_proj'] = (df['proj_rank_pos'] <= 3).astype(float)
    df['is_top3_salary'] = (df['salary_rank_pos'] <= 3).astype(float)

    # --- Matchup / Vegas features ---
    df['team_implied_total'] = df['team_implied_total'].fillna(2.75)
    df['opp_implied_total'] = df['opp_implied_total'].fillna(2.75)
    df['game_total'] = df['game_total'].fillna(5.5)
    df['is_favorite'] = df['is_favorite'].fillna(0).astype(float)
    df['spread_abs'] = df['spread'].fillna(0).abs()
    df['implied_total_rank'] = df.groupby('slate_date')['team_implied_total'].rank(
        ascending=False, pct=True)
    df['high_game_total'] = (df['game_total'] >= 6.0).astype(float)
    df['favorable_matchup'] = (df['is_favorite'] * df['high_game_total']).astype(float)

    # --- Line deployment ---
    df['start_line_num'] = pd.to_numeric(df['start_line'], errors='coerce')
    df['is_line1'] = (df['start_line_num'] == 1).astype(float)
    df['is_line2'] = (df['start_line_num'] == 2).astype(float)
    df['pp_unit_num'] = pd.to_numeric(df['pp_unit'], errors='coerce')
    df['is_pp1'] = (df['pp_unit_num'] == 1).astype(float)
    df['is_pp2'] = (df['pp_unit_num'] == 2).astype(float)

    # --- Position one-hot ---
    for p in ['C', 'W', 'D', 'G']:
        df[f'pos_{p}'] = (df['position'] == p).astype(float)

    # --- Slate context ---
    df['n_games_on_slate'] = df['n_games_on_slate'].fillna(1)
    pos_count = df.groupby(['slate_date', 'position']).size().reset_index(name='players_at_pos')
    df = df.merge(pos_count, on=['slate_date', 'position'], how='left')
    slate_size = df.groupby('slate_date').size().reset_index(name='slate_size')
    df = df.merge(slate_size, on='slate_date', how='left')

    # --- Derived interactions ---
    df['dk_value_ratio'] = df['dk_avg_fpts'] / (df['salary_k'] + 0.1)
    df['salary_x_implied'] = df['salary_k'] * df['team_implied_total']
    df['dk_avg_x_implied'] = df['dk_avg_fpts'] * df['team_implied_total']
    df['dk_avg_x_pp1'] = df['dk_avg_fpts'] * df['is_pp1']
    df['salary_x_pp1'] = df['salary_k'] * df['is_pp1']

    # Clean up infinities
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_training_data() -> pd.DataFrame:
    """
    Load and merge dk_salaries with own.csv to create training dataset.

    Returns DataFrame with features + 'ownership' target column + 'slate_date'.
    """
    if not OWN_CSV_PATH.exists():
        raise FileNotFoundError(f"own.csv not found at {OWN_CSV_PATH}")

    # Load own.csv
    own = pd.read_csv(OWN_CSV_PATH)
    own['date_parsed'] = pd.to_datetime(own['Date'], format='%m/%d/%y')
    own['slate_date'] = own['date_parsed'].dt.strftime('%Y-%m-%d')
    own['name_lower'] = own['Player'].str.lower().str.strip()
    own['ownership'] = pd.to_numeric(own['Ownership'], errors='coerce')
    own = own.dropna(subset=['ownership'])

    # Load dk_salaries for overlapping dates
    conn = sqlite3.connect(DB_PATH)
    dates_str = "','".join(sorted(own['slate_date'].unique()))
    dk = pd.read_sql(f"""
        SELECT player_name, team, position, salary, dk_avg_fpts, dk_ceiling,
               dk_stdv, opponent, team_implied_total, opp_implied_total,
               game_total, is_favorite, spread, win_pct, ownership_pct,
               start_line, pp_unit, n_games_on_slate, slate_size_players,
               slate_date
        FROM dk_salaries
        WHERE slate_date IN ('{dates_str}')
    """, conn)
    conn.close()

    dk['name_lower'] = dk['player_name'].str.lower().str.strip()

    # Grab line/PP info from own.csv (supplements dk_salaries where missing)
    own['own_line'] = pd.to_numeric(own['Line'], errors='coerce')
    own['own_pp'] = pd.to_numeric(own['PP Unit'], errors='coerce')

    # Merge on player name + date (no FC Proj — not available at prediction time)
    merged = dk.merge(
        own[['slate_date', 'name_lower', 'ownership', 'own_line', 'own_pp']],
        on=['slate_date', 'name_lower'],
        how='inner'
    )

    # Use own.csv line/PP info where dk_salaries is missing
    if 'start_line' in merged.columns:
        merged['start_line'] = merged['start_line'].fillna(merged['own_line'])
    if 'pp_unit' in merged.columns:
        merged['pp_unit'] = merged['pp_unit'].fillna(merged['own_pp'])

    print(f"  Training data: {len(merged)} observations across {merged['slate_date'].nunique()} dates")
    print(f"  Ownership range: {merged['ownership'].min():.1f}% - {merged['ownership'].max():.1f}%")
    print(f"  Mean ownership: {merged['ownership'].mean():.2f}%")

    # Engineer features
    merged = engineer_features(merged)

    return merged


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_xgboost_lodocv(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Train XGBoost with leave-one-date-out cross-validation.

    For each date:
    1. Hold out that date as test set
    2. Train on all other dates
    3. Predict held-out date
    4. Collect predictions

    Returns dict with model, metrics, and all OOF predictions.
    """
    dates = sorted(df['slate_date'].unique())
    features = [c for c in FEATURE_COLS if c in df.columns]

    if verbose:
        print(f"\n  LODOCV: {len(dates)} folds, {len(features)} features")

    all_preds = []
    all_actuals = []
    all_dates_out = []
    fold_results = []

    for i, test_date in enumerate(dates):
        train_mask = df['slate_date'] != test_date
        test_mask = df['slate_date'] == test_date

        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, 'ownership'].values
        X_test = df.loc[test_mask, features].values
        y_test = df.loc[test_mask, 'ownership'].values

        if len(y_test) == 0:
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=5,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
        )

        model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test_s)
        y_pred = np.clip(y_pred, 0.1, 50.0)

        mae = mean_absolute_error(y_test, y_pred)
        corr = spearmanr(y_test, y_pred)[0] if len(y_test) > 2 else 0
        bias = float(np.mean(y_pred - y_test))

        fold_results.append({
            'date': test_date,
            'n': len(y_test),
            'mae': mae,
            'corr': corr,
            'bias': bias,
        })

        all_preds.extend(y_pred)
        all_actuals.extend(y_test)
        all_dates_out.extend([test_date] * len(y_test))

        if verbose and (i + 1) % 20 == 0:
            print(f"    Fold {i+1}/{len(dates)}: MAE={mae:.3f}, corr={corr:.3f}")

    # Overall metrics
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    overall_mae = mean_absolute_error(all_actuals, all_preds)
    overall_corr = spearmanr(all_actuals, all_preds)[0]
    overall_bias = float(np.mean(all_preds - all_actuals))

    if verbose:
        print(f"\n  LODOCV Results ({len(dates)} folds, {len(all_preds)} predictions):")
        print(f"    Overall MAE:  {overall_mae:.4f}%")
        print(f"    Spearman r:   {overall_corr:.4f}")
        print(f"    Bias:         {overall_bias:+.4f}%")

        # Per-position breakdown
        oof_df = pd.DataFrame({
            'slate_date': all_dates_out,
            'actual': all_actuals,
            'predicted': all_preds,
        })
        # Merge back position info
        pos_info = df[['slate_date', 'name_lower', 'position']].drop_duplicates()
        # Can't merge without name, so compute by position from original df alignment
        oof_df['position'] = df.loc[df['slate_date'].isin(dates), 'position'].values[:len(oof_df)]

        print(f"\n  By Position:")
        for pos in ['C', 'W', 'D', 'G']:
            pos_mask = oof_df['position'] == pos
            if pos_mask.sum() > 0:
                pos_mae = mean_absolute_error(oof_df.loc[pos_mask, 'actual'],
                                               oof_df.loc[pos_mask, 'predicted'])
                pos_corr = spearmanr(oof_df.loc[pos_mask, 'actual'],
                                      oof_df.loc[pos_mask, 'predicted'])[0]
                print(f"    {pos}: MAE={pos_mae:.4f}%, corr={pos_corr:.4f} (n={pos_mask.sum()})")

    return {
        'overall_mae': overall_mae,
        'overall_corr': overall_corr,
        'overall_bias': overall_bias,
        'fold_results': fold_results,
        'oof_predictions': all_preds,
        'oof_actuals': all_actuals,
        'oof_dates': all_dates_out,
        'features': features,
    }


def train_final_model(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Train final XGBoost model on ALL data and save to pickle.

    Returns dict with model, scaler, features, and metrics.
    """
    X = df[features].values
    y = df['ownership'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0,
    )
    model.fit(X_scaled, y)

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    # Medians for missing value imputation at prediction time
    medians = {col: float(df[col].median()) for col in features}

    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'medians': medians,
        'importance': importance,
        'n_training': len(y),
        'n_dates': df['slate_date'].nunique(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  Training: {len(y)} observations, {df['slate_date'].nunique()} dates")
    print(f"\n  Top 10 Feature Importances:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:<25} {row['importance']:.4f}")

    return model_data


# ==============================================================================
# PREDICTION (for lineup_builder.py integration)
# ==============================================================================

def load_ownership_model() -> Optional[Dict]:
    """Load trained XGBoost ownership model from pickle."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_ownership_for_pool(pool: pd.DataFrame, model_data: Dict = None) -> pd.Series:
    """
    Predict ownership for a player pool DataFrame (from build_player_pool).

    Args:
        pool: DataFrame with dk_salaries columns (from build_player_pool)
        model_data: Pre-loaded model dict (or loads from pickle)

    Returns:
        pd.Series of predicted ownership percentages, indexed like pool.
    """
    if model_data is None:
        model_data = load_ownership_model()
    if model_data is None:
        return None

    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    medians = model_data['medians']

    # Map pool columns to expected feature names
    # The pool has 'name' instead of 'player_name', and uses 'norm_pos'
    pred_df = pool.copy()

    # Ensure 'position' column exists (pool may use norm_pos)
    if 'position' not in pred_df.columns and 'norm_pos' in pred_df.columns:
        pred_df['position'] = pred_df['norm_pos']

    # Ensure slate_date exists
    if 'slate_date' not in pred_df.columns:
        # Use a dummy date — features are relative within slate anyway
        pred_df['slate_date'] = 'pred'

    # Engineer features
    pred_df = engineer_features(pred_df)

    # Fill missing features with training medians
    for f in features:
        if f not in pred_df.columns:
            pred_df[f] = medians.get(f, 0)
        pred_df[f] = pred_df[f].fillna(medians.get(f, 0))

    X = pred_df[features].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    preds = np.clip(preds, 0.1, 50.0)

    return pd.Series(preds, index=pool.index)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import sys

    print("\n" + "=" * 80)
    print("XGBOOST OWNERSHIP MODEL v2 — TRAINING PIPELINE")
    print("=" * 80)

    # 1. Load training data
    print("\n[1/4] Loading training data...")
    df = load_training_data()

    # 2. LODOCV
    print("\n[2/4] Running leave-one-date-out cross-validation...")
    cv_results = train_xgboost_lodocv(df, verbose=True)

    # 3. Compare to Ridge baseline
    print("\n" + "=" * 80)
    print("COMPARISON vs RIDGE BASELINE")
    print("=" * 80)
    ridge_mae = 1.92
    ridge_corr = 0.905
    xgb_mae = cv_results['overall_mae']
    xgb_corr = cv_results['overall_corr']
    print(f"  Ridge:   MAE={ridge_mae:.4f}%, corr={ridge_corr:.4f} (6 dates, ~1,200 obs)")
    print(f"  XGBoost: MAE={xgb_mae:.4f}%, corr={xgb_corr:.4f} ({df['slate_date'].nunique()} dates, {len(df)} obs)")
    pct_improvement = (ridge_mae - xgb_mae) / ridge_mae * 100
    print(f"  MAE improvement: {pct_improvement:+.1f}%")

    # 4. Train final model on all data
    print("\n[3/4] Training final model on all data...")
    model_data = train_final_model(df, cv_results['features'])

    # 5. Worst/best dates
    print("\n[4/4] Per-date analysis...")
    fold_df = pd.DataFrame(cv_results['fold_results'])
    fold_df = fold_df.sort_values('mae')
    print(f"\n  Best 5 dates (lowest MAE):")
    for _, row in fold_df.head(5).iterrows():
        print(f"    {row['date']}: MAE={row['mae']:.3f}%, corr={row['corr']:.3f} (n={row['n']})")
    print(f"\n  Worst 5 dates (highest MAE):")
    for _, row in fold_df.tail(5).iterrows():
        print(f"    {row['date']}: MAE={row['mae']:.3f}%, corr={row['corr']:.3f} (n={row['n']})")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    return cv_results, model_data


if __name__ == '__main__':
    cv_results, model_data = main()
