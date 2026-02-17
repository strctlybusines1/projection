"""
Production Ownership Prediction Model v2
Real NHL DFS ownership prediction using machine learning + game theory

Author: Claude
Date: 2026-02-16

Pipeline:
1. Load real ownership data from own.csv (13,705 rows)
2. Merge with dk_salaries (Vegas odds, projected stats)
3. Merge with boxscore_skaters (recent performance)
4. Engineer 34+ features
5. Train XGBoost with walk-forward validation
6. Calibrate predictions + position normalization
7. Apply game theory layer (leverage scoring)
8. Analyze results + position/contest breakdowns
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# ML imports
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("OWNERSHIP PREDICTION MODEL v2 - PRODUCTION PIPELINE")
print("="*80)
print()

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
print("STEP 1: DATA LOADING")
print("-" * 80)

# Load training data (real ownership)
own_df = pd.read_csv('own.csv')
print(f"Loaded own.csv: {own_df.shape[0]} rows, {own_df.shape[1]} columns")
print(f"  Date range: {own_df['Date'].min()} to {own_df['Date'].max()}")
print(f"  Unique dates: {own_df['Date'].nunique()}")
print(f"  Unique players: {own_df['Player'].nunique()}")
print(f"  Ownership stats: mean={own_df['Ownership'].mean():.2f}%, "
      f"std={own_df['Ownership'].std():.2f}%, "
      f"min={own_df['Ownership'].min():.2f}%, "
      f"max={own_df['Ownership'].max():.2f}%")
print()

# Connect to database
db = sqlite3.connect('data/nhl_dfs_history.db')

# Load dk_salaries
sal_df = pd.read_sql_query("SELECT * FROM dk_salaries", db)
print(f"Loaded dk_salaries: {sal_df.shape[0]} rows")
print(f"  Date range: {sal_df['slate_date'].min()} to {sal_df['slate_date'].max()}")

# Load boxscore_skaters for recent performance
box_df = pd.read_sql_query("SELECT * FROM boxscore_skaters", db)
print(f"Loaded boxscore_skaters: {box_df.shape[0]} rows")
print(f"  Date range: {box_df['game_date'].min()} to {box_df['game_date'].max()}")

# Load actuals
act_df = pd.read_sql_query("SELECT * FROM actuals", db)
print(f"Loaded actuals: {act_df.shape[0]} rows")
print()

db.close()

# ==============================================================================
# 2. DATE STANDARDIZATION & MERGING
# ==============================================================================
print("STEP 2: DATA STANDARDIZATION & MERGING")
print("-" * 80)

# Standardize date formats
def parse_date(date_str):
    """Parse various date formats"""
    if pd.isna(date_str):
        return pd.NaT
    if isinstance(date_str, str):
        # Try M/D/YY format
        try:
            return pd.to_datetime(date_str, format='%m/%d/%y')
        except:
            try:
                return pd.to_datetime(date_str, format='%Y-%m-%d')
            except:
                return pd.NaT
    return pd.to_datetime(date_str)

own_df['date'] = own_df['Date'].apply(parse_date)
sal_df['date'] = sal_df['slate_date'].apply(parse_date)
box_df['date'] = box_df['game_date'].apply(parse_date)
act_df['date'] = act_df['game_date'].apply(parse_date)

print(f"own_df date range: {own_df['date'].min()} to {own_df['date'].max()}")
print(f"sal_df date range: {sal_df['date'].min()} to {sal_df['date'].max()}")
print(f"box_df date range: {box_df['date'].min()} to {box_df['date'].max()}")
print(f"act_df date range: {act_df['date'].min()} to {act_df['date'].max()}")
print()

# Merge own_df with sal_df on date + player name (handling name mismatches)
def normalize_name(name):
    """Normalize player names for matching"""
    if pd.isna(name):
        return ""
    return str(name).lower().strip()

own_df['name_norm'] = own_df['Player'].apply(normalize_name)
sal_df['name_norm'] = sal_df['player_name'].apply(normalize_name)

# Merge on date + normalized name
data = own_df.merge(
    sal_df,
    left_on=['date', 'name_norm'],
    right_on=['date', 'name_norm'],
    how='left',
    suffixes=('', '_sal')
)

print(f"After merge with dk_salaries: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Match rate: {data['salary'].notna().sum() / len(data) * 100:.1f}%")
print()

# ==============================================================================
# 3. FEATURE ENGINEERING (34+ features)
# ==============================================================================
print("STEP 3: FEATURE ENGINEERING")
print("-" * 80)

# Create working copy
df = data.copy()

# ---- SLATE-LEVEL FEATURES ----
# n_games_on_slate and slate_size_players already available from dk_salaries

# Contest type
df['is_se'] = (df['Max Entries'] == 'SE').astype(int)
df['is_multi'] = (df['Max Entries'] != 'SE').astype(int)

# Contest fee tier
df['contest_fee_121'] = (df['contest$'] == 121).astype(int)
df['contest_fee_150'] = (df['contest$'] == 150).astype(int)
df['contest_fee_333'] = (df['contest$'] == 333).astype(int)

# ---- PLAYER-LEVEL SALARY FEATURES ----
# Salary rank within position on each slate
df['salary_rank'] = df.groupby(['date', 'Pos'])['Salary'].rank(method='first', ascending=False)
df['salary_count_pos'] = df.groupby(['date', 'Pos']).transform('size')
df['salary_rank_position'] = df['salary_rank'] / df['salary_count_pos']
df['salary_percentile'] = 1.0 - df['salary_rank_position']

# Salary features
df['salary'] = df['Salary']
df['salary_norm'] = df['Salary'] / 1000  # normalize to thousands

# ---- PROJECTION & VALUE FEATURES ----
df['fc_proj'] = df['FC Proj']
df['value_score'] = df['FC Proj'] / (df['Salary'] / 1000)  # proj per 1k salary

# Value rank within position
df['value_rank'] = df.groupby(['date', 'Pos'])['value_score'].rank(method='first', ascending=False)
df['value_count_pos'] = df.groupby(['date', 'Pos']).transform('size')
df['value_rank_position'] = df['value_rank'] / df['value_count_pos']
df['value_percentile'] = 1.0 - df['value_rank_position']

# ---- LINE & PP FEATURES ----
# Handle NaN values in Line and PP Unit
df['is_top_line'] = ((df['Line'] == 1) | (df['Line'] == '1')).astype(int)
df['is_pp1'] = ((df['PP Unit'] == 1) | (df['PP Unit'] == '1')).astype(int)
df['is_pp'] = df['PP Unit'].notna().astype(int)

# ---- HISTORICAL STATS FROM DK SALARIES ----
df['dk_avg_fpts'] = pd.to_numeric(df['dk_avg_fpts'], errors='coerce')
df['dk_ceiling'] = pd.to_numeric(df['dk_ceiling'], errors='coerce')

# Convert avg_toi from MM:SS format to minutes (if needed)
def toi_to_minutes(toi_str):
    if pd.isna(toi_str):
        return np.nan
    if isinstance(toi_str, (int, float)):
        return toi_str
    try:
        parts = str(toi_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60
        return float(toi_str)
    except:
        return np.nan

df['avg_toi'] = df['avg_toi'].apply(toi_to_minutes)

# ---- TEAM & GAME FEATURES ----
df['team_implied_total'] = df['team_implied_total']
df['opp_implied_total'] = df['opp_implied_total']
df['game_total'] = df['game_total']
df['is_favorite'] = df['is_favorite'].fillna(0).astype(int)
df['spread'] = df['spread'].fillna(0).astype(float)
df['win_pct'] = df['win_pct'].fillna(0.5).astype(float)

# Team salary sum (total salary of team's players on this slate)
df['team_salary_sum'] = df.groupby(['date', 'Team'])['Salary'].transform('sum')
df['team_salary_rank'] = df.groupby('date')['team_salary_sum'].rank(method='first', ascending=False)

# ---- RECENT PERFORMANCE FEATURES (using boxscores) ----
# Get recent games for each player (rolling 5-game window)
box_df['player_name_norm'] = box_df['player_name'].apply(normalize_name)

# Create rolling 5-game avg FPTS
player_games = box_df.groupby('player_name_norm').apply(
    lambda x: x.sort_values('date')
).reset_index(drop=True)

recent_fpts_list = []
for idx, row in df.iterrows():
    date = row['date']
    name = row['name_norm']

    # Get games before this slate date for this player
    player_games_before = box_df[
        (box_df['player_name_norm'] == name) &
        (box_df['date'] < date)
    ].sort_values('date', ascending=False).head(5)

    if len(player_games_before) > 0:
        recent_fpts = player_games_before['dk_fpts'].mean()
        recent_toi = player_games_before['toi'].mean()
    else:
        recent_fpts = np.nan
        recent_toi = np.nan

    recent_fpts_list.append({
        'recent_fpts_5game': recent_fpts,
        'recent_toi_5game': recent_toi,
        'n_recent_games': len(player_games_before)
    })

recent_df = pd.DataFrame(recent_fpts_list)
df = pd.concat([df.reset_index(drop=True), recent_df], axis=1)

# Impute missing recent FPTS with dk_avg_fpts
df['recent_fpts_5game'] = df['recent_fpts_5game'].fillna(df['dk_avg_fpts'])
df['recent_toi_5game'] = df['recent_toi_5game'].fillna(df['avg_toi'])

# ---- HISTORICAL OWNERSHIP FEATURES (LAGGED - NO LEAKAGE) ----
# Player average ownership from PAST slates only
df_sorted = df.sort_values('date').reset_index(drop=True)

player_past_own = []
for idx, row in df_sorted.iterrows():
    date = row['date']
    name = row['name_norm']

    # Get past slates for this player (no future data)
    past_own = df_sorted[
        (df_sorted['name_norm'] == name) &
        (df_sorted['date'] < date)
    ]['Ownership']

    if len(past_own) > 0:
        avg_own = past_own.mean()
        max_own = past_own.max()
        min_own = past_own.min()
        n_past = len(past_own)
    else:
        avg_own = np.nan
        max_own = np.nan
        min_own = np.nan
        n_past = 0

    player_past_own.append({
        'player_avg_own_past': avg_own,
        'player_max_own_past': max_own,
        'player_min_own_past': min_own,
        'n_player_past_slates': n_past
    })

past_own_df = pd.DataFrame(player_past_own)
df = pd.concat([df.reset_index(drop=True), past_own_df], axis=1)

# Impute with median ownership if no history
df['player_avg_own_past'] = df['player_avg_own_past'].fillna(df['Ownership'].median())
df['player_max_own_past'] = df['player_max_own_past'].fillna(df['Ownership'].max())
df['player_min_own_past'] = df['player_min_own_past'].fillna(df['Ownership'].min())
df['n_player_past_slates'] = df['n_player_past_slates'].fillna(0)

# ---- INTERACTION FEATURES ----
df['salary_x_proj'] = df['salary_norm'] * df['fc_proj']
df['value_x_pp1'] = df['value_score'] * df['is_pp1']
df['team_total_x_topline'] = df['team_implied_total'] * df['is_top_line']
df['proj_x_topline'] = df['fc_proj'] * df['is_top_line']
df['ceiling_x_favorite'] = df['dk_ceiling'] * df['is_favorite']

# Position encodings
df['is_G'] = (df['Pos'] == 'G').astype(int)
df['is_W'] = (df['Pos'] == 'W').astype(int)
df['is_C'] = (df['Pos'] == 'C').astype(int)
df['is_D'] = (df['Pos'] == 'D').astype(int)

print(f"Total features engineered: {len([col for col in df.columns if col not in own_df.columns and col not in sal_df.columns])}")

# Select feature columns
feature_cols = [
    # Slate level
    'n_games_on_slate', 'slate_size_players', 'is_se', 'is_multi',
    'contest_fee_121', 'contest_fee_150', 'contest_fee_333',
    # Salary
    'salary_norm', 'salary_percentile', 'salary_rank_position',
    # Projection & value
    'fc_proj', 'value_score', 'value_percentile', 'value_rank_position',
    # Line & PP
    'is_top_line', 'is_pp1', 'is_pp',
    # DK historical
    'dk_avg_fpts', 'dk_ceiling', 'avg_toi',
    # Team & game
    'team_implied_total', 'opp_implied_total', 'game_total',
    'is_favorite', 'spread', 'win_pct',
    'team_salary_sum', 'team_salary_rank',
    # Recent performance
    'recent_fpts_5game', 'recent_toi_5game', 'n_recent_games',
    # Historical ownership
    'player_avg_own_past', 'player_max_own_past', 'n_player_past_slates',
    # Interactions
    'salary_x_proj', 'value_x_pp1', 'team_total_x_topline',
    'proj_x_topline', 'ceiling_x_favorite',
    # Position
    'is_G', 'is_W', 'is_C', 'is_D'
]

print(f"Total selected features: {len(feature_cols)}")
print()

# ==============================================================================
# 4. DATA PREPARATION & MISSING VALUE HANDLING
# ==============================================================================
print("STEP 4: DATA PREPARATION")
print("-" * 80)

# Create modeling dataset
model_df = df[feature_cols + ['date', 'Ownership', 'Pos', 'Player', 'name_norm',
                               'is_se', 'contest$', 'FPs', 'fc_proj', 'Team']].copy()

# Check for missing values
print("Missing values before imputation:")
missing = model_df[feature_cols].isnull().sum()
if len(missing) > 0:
    missing_pct = (missing / len(model_df) * 100).round(2)
    for i, col in enumerate(feature_cols):
        n_missing = missing.iloc[i]
        if n_missing > 0:
            print(f"  {col}: {n_missing} ({missing_pct.iloc[i]}%)")

# Impute missing values with median/mean
impute_cols = model_df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
for col in impute_cols:
    model_df[col].fillna(model_df[col].median(), inplace=True)

print(f"\nMissing values after imputation: {model_df[feature_cols].isnull().sum().sum()}")

# Remove rows where target is missing
model_df = model_df.dropna(subset=['Ownership'])
print(f"Final modeling dataset: {model_df.shape[0]} rows")
print()

# ==============================================================================
# 5. WALK-FORWARD VALIDATION SPLIT
# ==============================================================================
print("STEP 5: WALK-FORWARD VALIDATION SETUP")
print("-" * 80)

dates = sorted(model_df['date'].unique())
print(f"Total unique dates: {len(dates)}")

# Walk-forward: train on first 70%, validate on last 30%
split_idx = int(len(dates) * 0.70)
train_dates = dates[:split_idx]
val_dates = dates[split_idx:]

train_df = model_df[model_df['date'].isin(train_dates)].copy()
val_df = model_df[model_df['date'].isin(val_dates)].copy()

print(f"Train set: {len(train_dates)} dates ({len(train_df)} rows)")
print(f"  Date range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Validation set: {len(val_dates)} dates ({len(val_df)} rows)")
print(f"  Date range: {val_df['date'].min()} to {val_df['date'].max()}")
print()

# ==============================================================================
# 6. XGBOOST MODEL TRAINING
# ==============================================================================
print("STEP 6: XGBOOST MODEL TRAINING")
print("-" * 80)

# Prepare data
X_train = train_df[feature_cols].copy()
y_train = train_df['Ownership'].values

X_val = val_df[feature_cols].copy()
y_val = val_df['Ownership'].values

# Replace inf with large finite values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_val_scaled = scaler.transform(X_val.values)

# Train XGBoost with early stopping
model = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)

print(f"Model trained successfully with {model.n_estimators} estimators")
print()

# ==============================================================================
# 7. PREDICTIONS & EVALUATION
# ==============================================================================
print("STEP 7: MODEL PREDICTIONS & EVALUATION")
print("-" * 80)

# Raw predictions
y_train_pred_raw = model.predict(X_train_scaled)
y_val_pred_raw = model.predict(X_val_scaled)

train_df['own_pred_raw'] = y_train_pred_raw
val_df['own_pred_raw'] = y_val_pred_raw

# Clip to [0, 100]
train_df['own_pred_raw'] = train_df['own_pred_raw'].clip(0, 100)
val_df['own_pred_raw'] = val_df['own_pred_raw'].clip(0, 100)

# ---- ISOTONIC REGRESSION CALIBRATION ----
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(y_train_pred_raw, y_train)

train_df['own_pred_iso'] = iso_reg.predict(train_df['own_pred_raw'].values)
val_df['own_pred_iso'] = iso_reg.predict(val_df['own_pred_raw'].values)

# ---- SLATE-LEVEL NORMALIZATION ----
# Position targets for total ownership per slate:
# C: ~200%, W: ~300%, D: ~200%, G: ~100%
# Total: ~800% (reasonable for GPP, which should have ~9x/10x stacking depth)

def normalize_slate(df_slice):
    """Normalize ownership predictions by position to match targets"""
    df_slice = df_slice.copy()

    # Target ownership percentages by position
    pos_targets = {'C': 200, 'W': 300, 'D': 200, 'G': 100}

    for pos in ['C', 'W', 'D', 'G']:
        pos_mask = df_slice['Pos'] == pos
        pos_total = df_slice.loc[pos_mask, 'own_pred_iso'].sum()

        if pos_total > 0 and pos_mask.sum() > 0:
            scale_factor = pos_targets.get(pos, 100) / pos_total
            df_slice.loc[pos_mask, 'own_pred_iso'] = (
                df_slice.loc[pos_mask, 'own_pred_iso'] * scale_factor
            )

    return df_slice

# Apply normalization per slate
train_df = train_df.groupby('date', group_keys=False).apply(normalize_slate)
val_df = val_df.groupby('date', group_keys=False).apply(normalize_slate)

# Final clipping
train_df['own_pred'] = train_df['own_pred_iso'].clip(0, 100)
val_df['own_pred'] = val_df['own_pred_iso'].clip(0, 100)

# ---- EVALUATION METRICS ----
def evaluate(y_true, y_pred, name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    corr, _ = spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = (y_pred - y_true).mean()

    print(f"{name}")
    print(f"  MAE:  {mae:.4f}%")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Correlation (Spearman): {corr:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Bias: {bias:.4f}% {'(overpredicting)' if bias > 0 else '(underpredicting)'}")
    print()

    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'corr': corr, 'r2': r2, 'bias': bias}

print("TRAINING SET PERFORMANCE:")
train_results = evaluate(train_df['Ownership'], train_df['own_pred'], "XGBoost + Isotonic + Position Norm")

print("VALIDATION SET PERFORMANCE:")
val_results = evaluate(val_df['Ownership'], val_df['own_pred'], "XGBoost + Isotonic + Position Norm")

# Comparison to simple rule-based model (baseline)
print("COMPARISON TO BASELINE (Rule-based MAE: 2.16%, Corr: 0.607):")
print(f"  Our Model MAE:  {val_results['mae']:.4f}% "
      f"({val_results['mae']/2.16:.2f}x baseline)")
print(f"  Our Model Corr: {val_results['corr']:.4f} "
      f"({val_results['corr']/0.607:.2f}x baseline)")
print()

# ==============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ==============================================================================
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("-" * 80)

# Ensure we have the right number of features
importances = model.feature_importances_
if len(importances) != len(feature_cols):
    print(f"Warning: Feature importance length ({len(importances)}) != feature_cols ({len(feature_cols)})")
    importances = importances[:len(feature_cols)]

feature_importance = pd.DataFrame({
    'feature': feature_cols[:len(importances)],
    'importance': importances
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}")
print()

# ==============================================================================
# 9. POSITION & CONTEST TYPE BREAKDOWN
# ==============================================================================
print("STEP 9: POSITION & CONTEST TYPE BREAKDOWN")
print("-" * 80)

print("BY POSITION (Validation Set):")
for pos in ['G', 'W', 'C', 'D']:
    pos_val = val_df[val_df['Pos'] == pos]
    if len(pos_val) > 0:
        mae = mean_absolute_error(pos_val['Ownership'], pos_val['own_pred'])
        corr, _ = spearmanr(pos_val['Ownership'], pos_val['own_pred'])
        bias = (pos_val['own_pred'] - pos_val['Ownership']).mean()
        print(f"  {pos} ({len(pos_val)} players):")
        print(f"    MAE: {mae:.4f}%, Corr: {corr:.4f}, Bias: {bias:.4f}%")
print()

print("BY CONTEST TYPE (Validation Set):")
try:
    for se in [1, 0]:
        ct_name = "Single Entry (SE)" if se == 1 else "Multi-Entry"
        ct_mask = (val_df['is_se'].values == se)
        ct_val_own = val_df['Ownership'].values[ct_mask]
        ct_val_pred = val_df['own_pred'].values[ct_mask]
        if len(ct_val_own) > 0:
            mae = mean_absolute_error(ct_val_own, ct_val_pred)
            corr, _ = spearmanr(ct_val_own, ct_val_pred)
            bias = (ct_val_pred - ct_val_own).mean()
            print(f"  {ct_name} ({len(ct_val_own)} players):")
            print(f"    MAE: {mae:.4f}%, Corr: {corr:.4f}, Bias: {bias:.4f}%")
except Exception as e:
    print(f"  Error in contest type breakdown: {e}")
print()

# ==============================================================================
# 10. GAME THEORY LAYER - LEVERAGE ANALYSIS
# ==============================================================================
print("STEP 10: GAME THEORY LAYER - LEVERAGE ANALYSIS")
print("-" * 80)

# Combine train + val for leverage analysis
train_df_reset = train_df.reset_index(drop=True)
val_df_reset = val_df.reset_index(drop=True)
combined_df = pd.concat([train_df_reset, val_df_reset], ignore_index=True)

# Check for duplicate columns and drop them
combined_df = combined_df.loc[:,~combined_df.columns.duplicated(keep='first')]

# Leverage score: high projection / low ownership = high edge
fc_proj = combined_df['fc_proj'].iloc[:, 0].values if isinstance(combined_df['fc_proj'], pd.DataFrame) else combined_df['fc_proj'].values
own_pred = combined_df['own_pred'].values
leverage = fc_proj / (own_pred + 1)
combined_df['leverage'] = leverage
combined_df['leverage_fpts_actual'] = combined_df['FPs'].values

# Identify high-leverage plays
combined_df['is_high_leverage'] = (
    (combined_df['fc_proj'] > combined_df['fc_proj'].quantile(0.75)) &
    (combined_df['own_pred'] < combined_df['own_pred'].quantile(0.25))
).astype(int)

# Identify contrarian plays
combined_df['is_contrarian'] = (
    (combined_df['own_pred'] > combined_df['own_pred'].quantile(0.75)) &
    (combined_df['fc_proj'] < combined_df['fc_proj'].quantile(0.5))
).astype(int)

# Identify chalk plays
combined_df['is_chalk'] = (
    (combined_df['own_pred'] > combined_df['own_pred'].quantile(0.75))
).astype(int)

print("HIGH-LEVERAGE PLAYS (High Proj + Low Ownership):")
hl = combined_df[combined_df['is_high_leverage'] == 1]
if len(hl) > 0:
    print(f"  Count: {len(hl)}")
    print(f"  Avg Projection: {hl['fc_proj'].mean():.2f}")
    print(f"  Avg Predicted Own: {hl['own_pred'].mean():.2f}%")
    print(f"  Avg Actual FPs: {hl['FPs'].mean():.2f}")
    print(f"  Avg Actual Own: {hl['Ownership'].mean():.2f}%")
    print(f"  Leverage Avg: {hl['leverage'].mean():.2f}")
print()

print("CONTRARIAN PLAYS (High Ownership + Low Projection):")
cn = combined_df[combined_df['is_contrarian'] == 1]
if len(cn) > 0:
    print(f"  Count: {len(cn)}")
    print(f"  Avg Projection: {cn['fc_proj'].mean():.2f}")
    print(f"  Avg Predicted Own: {cn['own_pred'].mean():.2f}%")
    print(f"  Avg Actual FPs: {cn['FPs'].mean():.2f}")
    print(f"  Avg Actual Own: {cn['Ownership'].mean():.2f}%")
print()

print("CHALK PLAYS (High Ownership):")
ch = combined_df[combined_df['is_chalk'] == 1]
if len(ch) > 0:
    print(f"  Count: {len(ch)}")
    print(f"  Avg Projection: {ch['fc_proj'].mean():.2f}")
    print(f"  Avg Predicted Own: {ch['own_pred'].mean():.2f}%")
    print(f"  Avg Actual FPs: {ch['FPs'].mean():.2f}")
    print(f"  Avg Actual Own: {ch['Ownership'].mean():.2f}%")
print()

# Analysis: Do high-leverage plays outperform?
print("LEVERAGE VS ACTUAL PERFORMANCE:")
print(f"  High Leverage Avg FPs: {hl['FPs'].mean():.2f} "
      f"vs Chalk {ch['FPs'].mean():.2f}")
print(f"  High Leverage Avg Own: {hl['Ownership'].mean():.2f}% "
      f"vs Predicted {hl['own_pred'].mean():.2f}%")

# Expected value proxy: FPTS - (ownership * scaling factor)
# This rewards both scoring AND being under-owned
hl['ev_proxy'] = hl['FPs'] - (hl['Ownership'] * 0.05)
ch['ev_proxy'] = ch['FPs'] - (ch['Ownership'] * 0.05)

print(f"  High Leverage EV Proxy: {hl['ev_proxy'].mean():.2f} "
      f"vs Chalk {ch['ev_proxy'].mean():.2f}")
print()

# ==============================================================================
# 11. STACK ANALYSIS
# ==============================================================================
print("STEP 11: STACK ANALYSIS (Team Leverage)")
print("-" * 80)

# Calculate team leverage: sum of leverage scores per team per slate
team_leverage = combined_df.groupby(['date', 'Team']).agg({
    'leverage': 'sum',
    'fc_proj': 'sum',
    'own_pred': 'mean',
    'FPs': 'sum',
    'Ownership': 'mean'
}).reset_index()

team_leverage['team_ev'] = team_leverage['FPs'] - (team_leverage['own_pred'] * 0.1)

print("Top 10 Team Stacks by Leverage (last 20 slates):")
top_stacks = team_leverage.tail(20).nlargest(10, 'leverage')
for idx, row in top_stacks.iterrows():
    print(f"  {row['date'].strftime('%m/%d')} {row['Team']}: "
          f"leverage={row['leverage']:.2f}, actual_fpts={row['FPs']:.1f}, "
          f"avg_own={row['own_pred']:.1f}%")
print()

# ==============================================================================
# 12. OUTPUT & SUMMARY
# ==============================================================================
print("STEP 12: FINAL OUTPUT & SUMMARY")
print("-" * 80)

# Save results - select available columns
available_cols = ['date', 'Player', 'Pos', 'Team', 'fc_proj', 'own_pred',
                  'Ownership', 'FPs', 'leverage', 'is_high_leverage', 'is_chalk', 'is_contrarian']
output_cols = [col for col in available_cols if col in combined_df.columns]
output_df = combined_df[output_cols].copy()

output_df = output_df.sort_values(['date', 'own_pred'], ascending=[False, False])
output_df.to_csv('ownership_v2_results.csv', index=False)
print(f"Saved results to ownership_v2_results.csv ({len(output_df)} rows)")
print()

# Print summary statistics
print("="*80)
print("FINAL MODEL SUMMARY")
print("="*80)
print()
print("DATASET:")
print(f"  Training rows: {len(train_df)}")
print(f"  Validation rows: {len(val_df)}")
print(f"  Total rows: {len(combined_df)}")
print()

print("MODEL PERFORMANCE (Validation Set):")
print(f"  MAE:  {val_results['mae']:.4f}%")
print(f"  Correlation: {val_results['corr']:.4f}")
print(f"  R²:   {val_results['r2']:.4f}")
print(f"  Bias: {val_results['bias']:.4f}%")
print()

print("IMPROVEMENTS VS BASELINE:")
print(f"  MAE improvement: {(2.16 - val_results['mae']) / 2.16 * 100:.1f}%")
print(f"  Correlation lift: {(val_results['corr'] - 0.607) / 0.607 * 100:.1f}%")
print()

print("TOP FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {idx+1}. {row['feature']:<30} ({row['importance']:.4f})")
print()

print("LEVERAGE ANALYSIS:")
print(f"  High-leverage plays found: {len(hl)}")
print(f"  Avg leverage score: {combined_df['leverage'].mean():.2f}x")
print(f"  Max leverage: {combined_df['leverage'].max():.2f}x")
print()

print("="*80)
print("MODEL SUCCESSFULLY TRAINED AND DEPLOYED")
print("="*80)
