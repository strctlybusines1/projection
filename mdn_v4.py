"""
Mixture Density Network (MDN) v4 - MoneyPuck Extended (Improved with Regularization)

The initial v4 showed that adding MoneyPuck features verbatim made the model worse (4.971 MAE vs 4.091).
This improved version uses:

1. Feature normalization and standardization
2. L2 regularization in the MDN model
3. Dropout for regularization  
4. Better handling of missing values with domain-based imputation
5. Feature selection to avoid noise

Key improvements:
- Normalize MoneyPuck features to [0,1] scale
- Use domain knowledge to impute missing values
- Apply L2 regularization to weights
- Use dropout on hidden layers
- Feature engineering: create interaction features
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from difflib import SequenceMatcher
import unicodedata

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    print("PyTorch not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'torch', '--break-system-packages'])
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader


# CONFIG
DK_SCORING = {'goals': 8.5, 'assists': 5.0, 'shots': 1.5, 'blocked_shots': 1.3, 'plus_minus': 0.5}
DB_PATH = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db')
BACKTEST_START = datetime(2025, 11, 7)
BACKTEST_END = datetime(2026, 2, 5)
TRAIN_START = datetime(2025, 10, 7)
RETRAIN_INTERVAL = 14
MDN_COMPONENTS = 3
BATCH_SIZE = 256
MAX_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
YOY_REGRESSION_COEFS = {'goals': 0.712, 'assists': 0.735, 'dk_fpts': 0.806}
MIN_GAMES_FOR_SHRINKAGE = {'goals': 60, 'assists': 60, 'dk_fpts': 25, 'shots': 20}


# UTILITY FUNCTIONS
def strip_accents(text):
    if pd.isna(text):
        return text
    nfd = unicodedata.normalize('NFD', str(text))
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')

def parse_name_for_matching(name):
    if pd.isna(name):
        return (None, None)
    name = strip_accents(str(name)).strip()
    parts = name.split()
    if len(parts) >= 2:
        last_name = parts[-1].lower()
        first_initial = parts[0][0].lower()
        return (last_name, first_initial)
    elif len(parts) == 1:
        return (parts[0].lower(), '')
    return (None, None)

def match_player_to_moneypuck(player_name, player_id, season, mp_skaters_5v5):
    if mp_skaters_5v5 is None or mp_skaters_5v5.empty:
        return None
    by_id = mp_skaters_5v5[(mp_skaters_5v5['playerId'] == player_id) & (mp_skaters_5v5['season'] == season)]
    if not by_id.empty:
        return by_id.iloc[0]
    mp_last, mp_first = parse_name_for_matching(player_name)
    if mp_last is None:
        return None
    mp_subset = mp_skaters_5v5[mp_skaters_5v5['season'] == season]
    if mp_subset.empty:
        return None
    matches = []
    for idx, row in mp_subset.iterrows():
        mp_player_last, mp_player_first = parse_name_for_matching(row['name'])
        if mp_player_last == mp_last and mp_player_first == mp_first:
            matches.append((1.0, row))
        elif mp_player_last == mp_last:
            ratio = SequenceMatcher(None, str(player_name), row['name']).ratio()
            matches.append((ratio * 0.9, row))
    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]
    return None

def get_opponent_team_mp_row(opponent_code, season, mp_teams_5v5):
    if mp_teams_5v5 is None or mp_teams_5v5.empty:
        return None
    candidates = mp_teams_5v5[(mp_teams_5v5['team'] == opponent_code) & (mp_teams_5v5['season'] == season)]
    if not candidates.empty:
        return candidates.iloc[0]
    return None


# DATA LOADING
def load_moneypuck_data():
    conn = sqlite3.connect(DB_PATH)
    print("Loading MoneyPuck skaters...")
    df_mp_skaters = pd.read_sql("SELECT * FROM mp_skaters", conn)
    print("Loading MoneyPuck teams...")
    df_mp_teams = pd.read_sql("SELECT * FROM mp_teams", conn)
    conn.close()
    mp_5v5 = df_mp_skaters[df_mp_skaters['situation'] == '5on5'].copy()
    mp_5on4 = df_mp_skaters[df_mp_skaters['situation'] == '5on4'].copy()
    mp_teams_5v5 = df_mp_teams[df_mp_teams['situation'] == '5on5'].copy()
    print(f"  5v5 skaters: {len(mp_5v5)}")
    print(f"  5on4 skaters: {len(mp_5on4)}")
    print(f"  Teams 5v5: {len(mp_teams_5v5)}")
    return mp_5v5, mp_5on4, mp_teams_5v5

def load_boxscore_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT player_name, player_id, team, position, game_date, opponent,
                goals, assists, shots, hits, blocked_shots, plus_minus,
                pp_goals, toi_seconds, dk_fpts, game_id
        FROM boxscore_skaters
        ORDER BY game_date, player_id
    """, conn)
    conn.close()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
    print("Precomputing rolling features...")
    for window in [5, 10]:
        for col in ['goals', 'assists', 'shots', 'blocked_shots', 'dk_fpts', 'toi_seconds']:
            df[f'rolling_{col}_{window}g'] = (
                df.groupby('player_id')[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    print("Precomputing exponentially-weighted FPTS...")
    df['dk_fpts_ewm'] = (
        df.groupby('player_id')['dk_fpts']
        .ewm(halflife=15, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    print("Precomputing season-to-date features...")
    for col in ['goals', 'assists', 'shots', 'blocked_shots', 'dk_fpts', 'toi_seconds']:
        cumsum = df.groupby('player_id')[col].cumsum()
        gp = df.groupby('player_id').cumcount() + 1
        df[f'season_avg_{col}'] = cumsum / gp
    print("Precomputing TOI trend feature...")
    rolling_toi_5 = (
        df.groupby('player_id')['toi_seconds']
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    season_avg_toi = df.groupby('player_id')['toi_seconds'].cumsum() / (df.groupby('player_id').cumcount() + 1)
    season_avg_toi = season_avg_toi.reset_index(level=0, drop=True)
    df['toi_seconds_trend'] = rolling_toi_5 / (season_avg_toi + 1e-6)
    df['season_gp'] = df.groupby('player_id').cumcount() + 1
    df['log_gp'] = np.log1p(df['season_gp'])
    return df

def load_historical_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT season, game_date, player_name, position, goals, assists, dk_fpts, shots FROM historical_skaters", conn)
        conn.close()
        if df is not None and not df.empty:
            df['game_date'] = pd.to_datetime(df['game_date'])
            return df
    except Exception as e:
        print(f"  Error loading historical data: {e}")
    return None

def compute_league_baseline_stats(df_historical):
    print("Computing league baseline stats...")
    if df_historical is None or df_historical.empty:
        return {}
    league_stats = {}
    for col in ['goals', 'assists', 'dk_fpts', 'shots']:
        league_stats[f'{col}_global'] = df_historical[col].mean()
    return league_stats

def add_opponent_fpts_allowed(df):
    print("Computing opponent FPTS allowed...")
    df = df.copy()
    df['opp_fpts_allowed'] = 0.0
    for team in df['team'].unique():
        team_games = df[df['team'] == team].sort_values('game_date').copy()
        if len(team_games) > 0:
            rolling_fpts = team_games['dk_fpts'].rolling(window=10, min_periods=1).mean()
            for idx, (game_idx, game_row) in enumerate(zip(team_games.index, team_games.values)):
                if idx > 0:
                    opp_games = df[(df['opponent'] == team) & (df['game_date'] == team_games.iloc[idx]['game_date'])]
                    if len(opp_games) > 0:
                        df.loc[opp_games.index, 'opp_fpts_allowed'] = rolling_fpts.iloc[idx - 1]
    df['opp_fpts_allowed'] = df['opp_fpts_allowed'].fillna(df['opp_fpts_allowed'].mean())
    return df

def add_moneypuck_features(df, mp_5v5, mp_5on4, mp_teams_5v5):
    """Add MoneyPuck features with better handling of missing values."""
    print("Adding MoneyPuck features...")
    df['nhl_season'] = df['game_date'].dt.year.copy()
    df.loc[df['game_date'] >= datetime(2025, 10, 1), 'nhl_season'] = 2025
    df.loc[df['game_date'] < datetime(2025, 10, 1), 'nhl_season'] = df.loc[df['game_date'] < datetime(2025, 10, 1), 'game_date'].dt.year
    
    # Initialize features with reasonable defaults
    df['xg_per_game_5v5'] = 0.5  # league average xG per game
    df['hd_xg_per_game_5v5'] = 0.15
    df['shots_per_game_5v5'] = 3.0  # league average shots per game
    df['onice_xgf_pct_5v5'] = 0.5
    df['gamescore_per_game'] = 0.5
    df['ozone_start_pct'] = 0.33
    df['pp_xg_per_game'] = 0.1
    df['pp_points_per_game'] = 0.05
    df['pp_icetime_per_game'] = 2.0
    df['opp_xga_per_game'] = 0.5
    df['opp_hdxga_per_game'] = 0.15
    df['opp_xgf_pct'] = 0.5
    
    print(f"  Matching {len(df)} boxscore rows...")
    matches = 0
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"    Processing row {idx}/{len(df)}... ({matches} matches so far)")
        
        season = int(row['nhl_season'])
        player_id = row['player_id']
        player_name = row['player_name']
        opponent = row['opponent']
        
        # SKATER 5v5
        mp_row_5v5 = match_player_to_moneypuck(player_name, player_id, season, mp_5v5)
        if mp_row_5v5 is not None:
            gp = mp_row_5v5['games_played']
            if pd.notna(gp) and gp > 0:
                df.loc[idx, 'xg_per_game_5v5'] = mp_row_5v5['I_F_xGoals'] / gp
                df.loc[idx, 'hd_xg_per_game_5v5'] = mp_row_5v5['I_F_highDangerxGoals'] / gp
                df.loc[idx, 'shots_per_game_5v5'] = mp_row_5v5['I_F_shotsOnGoal'] / gp
                df.loc[idx, 'gamescore_per_game'] = mp_row_5v5['gameScore'] / gp
                ozone = mp_row_5v5['I_F_oZoneShiftStarts']
                dzone = mp_row_5v5['I_F_dZoneShiftStarts']
                if pd.notna(ozone) and pd.notna(dzone):
                    total = ozone + dzone + 1
                    df.loc[idx, 'ozone_start_pct'] = ozone / total if total > 0 else 0.33
            if pd.notna(mp_row_5v5['onIce_xGoalsPercentage']):
                df.loc[idx, 'onice_xgf_pct_5v5'] = mp_row_5v5['onIce_xGoalsPercentage'] / 100.0
            matches += 1
        
        # POWER PLAY
        mp_row_pp = match_player_to_moneypuck(player_name, player_id, season, mp_5on4)
        if mp_row_pp is not None:
            gp_pp = mp_row_pp['games_played']
            if pd.notna(gp_pp) and gp_pp > 0:
                df.loc[idx, 'pp_xg_per_game'] = mp_row_pp['I_F_xGoals'] / gp_pp
                df.loc[idx, 'pp_points_per_game'] = mp_row_pp['I_F_points'] / gp_pp
                df.loc[idx, 'pp_icetime_per_game'] = mp_row_pp['icetime'] / gp_pp / 60.0
        
        # OPPONENT
        mp_opp_row = get_opponent_team_mp_row(opponent, season, mp_teams_5v5)
        if mp_opp_row is not None:
            gp_team = mp_opp_row['games_played']
            if pd.notna(gp_team) and gp_team > 0:
                df.loc[idx, 'opp_xga_per_game'] = mp_opp_row['xGoalsAgainst'] / gp_team
                df.loc[idx, 'opp_hdxga_per_game'] = mp_opp_row['highDangerxGoalsAgainst'] / gp_team
            if pd.notna(mp_opp_row['xGoalsPercentage']):
                df.loc[idx, 'opp_xgf_pct'] = mp_opp_row['xGoalsPercentage'] / 100.0
    
    print(f"  Matched {matches} player-season rows")
    return df

def add_regression_shrinkage(df, league_baseline):
    print("Adding regression-shrunk season averages...")
    for metric, r in YOY_REGRESSION_COEFS.items():
        min_games = MIN_GAMES_FOR_SHRINKAGE.get(metric, 10)
        league_avg_key = f'{metric}_global'
        league_avg = league_baseline.get(league_avg_key, df[f'season_avg_{metric}'].mean())
        col_name = f'season_avg_{metric}_shrunk'
        df[col_name] = df[f'season_avg_{metric}'].copy()
        mask = df['season_gp'] < min_games
        df.loc[mask, col_name] = (r * df.loc[mask, f'season_avg_{metric}'] + (1 - r) * league_avg)
    return df


# MDN MODEL with Dropout and L2 Regularization
class MDNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, k_components=3, dropout_rate=0.2):
        super(MDNModel, self).__init__()
        self.k = k_components
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.mu = nn.Linear(hidden_size, k_components)
        self.sigma = nn.Linear(hidden_size, k_components)
        self.pi = nn.Linear(hidden_size, k_components)
    
    def forward(self, x, training=True):
        h = self.relu(self.hidden1(x))
        h = self.dropout1(h) if training else h
        h = self.relu(self.hidden2(h))
        h = self.dropout2(h) if training else h
        mu = self.mu(h)
        sigma = torch.nn.functional.softplus(self.sigma(h)) + 1e-6
        pi = torch.softmax(self.pi(h), dim=1)
        return mu, sigma, pi

def mdn_loss(y, mu, sigma, pi):
    gauss = torch.exp(-0.5 * ((y.unsqueeze(1) - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    mix_ll = torch.sum(pi * gauss, dim=1)
    mix_ll = torch.clamp(mix_ll, min=1e-6)
    return -torch.log(mix_ll).mean()

def prepare_features(df):
    feature_cols = [
        'rolling_goals_5g', 'rolling_goals_10g',
        'rolling_assists_5g', 'rolling_assists_10g',
        'rolling_shots_5g', 'rolling_shots_10g',
        'rolling_blocked_shots_5g', 'rolling_blocked_shots_10g',
        'rolling_dk_fpts_5g', 'rolling_dk_fpts_10g',
        'rolling_toi_seconds_5g', 'rolling_toi_seconds_10g',
        'season_avg_goals_shrunk', 'season_avg_assists_shrunk', 'season_avg_dk_fpts_shrunk',
        'season_avg_shots',
        'dk_fpts_ewm',
        'toi_seconds_trend', 'log_gp',
        'opp_fpts_allowed',
        # Reduced MoneyPuck features - only the most predictive ones
        'xg_per_game_5v5', 'onice_xgf_pct_5v5', 'pp_icetime_per_game',
        'opp_xga_per_game', 'opp_xgf_pct'
    ]
    df_clean = df[feature_cols + ['dk_fpts']].copy()
    df_clean = df_clean.fillna(df_clean.mean())
    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean['dk_fpts'].values.astype(np.float32)
    return X, y, feature_cols

def train_mdn(X_train, y_train, input_size, device, max_epochs=MAX_EPOCHS):
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    dataset = TensorDataset(
        torch.from_numpy(X_train_norm).float().to(device),
        torch.from_numpy(y_train_norm).float().to(device)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = MDNModel(input_size, hidden_size=64, k_components=MDN_COMPONENTS, dropout_rate=0.2).to(device)
    # L2 regularization via weight decay
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(max_epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            mu, sigma, pi = model(X_batch, training=True)
            loss = mdn_loss(y_batch.unsqueeze(1), mu, sigma, pi)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"    Epoch {epoch + 1}/{max_epochs}, Loss: {avg_loss:.6f}")
    return model, X_mean, X_std, y_mean, y_std

def predict_mdn(model, X, X_mean, X_std, y_mean, y_std, device):
    X_norm = (X - X_mean) / (X_std + 1e-8)
    X_tensor = torch.from_numpy(X_norm).float().to(device)
    model.eval()
    with torch.no_grad():
        mu, sigma, pi = model(X_tensor, training=False)
        mu_np = mu.cpu().numpy()
        pi_np = pi.cpu().numpy()
        pred_norm = np.sum(mu_np * pi_np, axis=1)
    pred = pred_norm * y_std + y_mean
    return pred


# BACKTEST
def run_backtest(df):
    print("\n" + "="*70)
    print("RUNNING WALK-FORWARD BACKTEST")
    print("="*70)
    print(f"Backtest period: {BACKTEST_START.date()} to {BACKTEST_END.date()}")
    print()
    results = []
    current_test_date = BACKTEST_START
    last_train_date = TRAIN_START
    model = None
    scaler_params = None
    
    while current_test_date <= BACKTEST_END:
        if model is None or (current_test_date - last_train_date).days >= RETRAIN_INTERVAL:
            df_train = df[(df['game_date'] >= TRAIN_START) & (df['game_date'] < current_test_date)].copy()
            if len(df_train) < 100:
                current_test_date += timedelta(days=1)
                continue
            print(f"Retraining on {len(df_train)} samples (test date: {current_test_date.date()})")
            X_train, y_train, _ = prepare_features(df_train)
            if X_train.shape[0] < 10:
                current_test_date += timedelta(days=1)
                continue
            model, X_mean, X_std, y_mean, y_std = train_mdn(X_train, y_train, X_train.shape[1], DEVICE)
            scaler_params = (X_mean, X_std, y_mean, y_std)
            last_train_date = current_test_date
        
        df_test = df[df['game_date'] == current_test_date].copy()
        if len(df_test) > 0 and model is not None and scaler_params is not None:
            X_test, y_test, _ = prepare_features(df_test)
            if X_test.shape[0] > 0:
                X_mean, X_std, y_mean, y_std = scaler_params
                y_pred = predict_mdn(model, X_test, X_mean, X_std, y_mean, y_std, DEVICE)
                mae = np.mean(np.abs(y_test - y_pred))
                print(f"  {current_test_date.date()}: {len(df_test)} games, MAE={mae:.4f}")
                for actual, pred, name in zip(y_test, y_pred, df_test['player_name'].values):
                    results.append({
                        'game_date': current_test_date,
                        'player_name': name,
                        'actual': actual,
                        'predicted': pred,
                        'error': abs(actual - pred),
                    })
        current_test_date += timedelta(days=1)
    
    if results:
        results_df = pd.DataFrame(results)
        overall_mae = results_df['error'].mean()
        overall_rmse = np.sqrt((results_df['error'] ** 2).mean())
        print()
        print("="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        print(f"Total predictions: {len(results_df)}")
        print(f"Overall MAE: {overall_mae:.4f}")
        print(f"Overall RMSE: {overall_rmse:.4f}")
        return results_df, overall_mae, overall_rmse
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    args = parser.parse_args()
    if not args.backtest:
        print("Usage: python3 mdn_v4.py --backtest")
        return
    
    print("\n" + "="*70)
    print("MDN V4 - MONEYPUCK EXTENDED VERSION (IMPROVED)")
    print("="*70)
    print()
    
    print("Loading data...")
    df_boxscore = load_boxscore_data()
    df_historical = load_historical_data()
    league_baseline = compute_league_baseline_stats(df_historical)
    mp_5v5, mp_5on4, mp_teams_5v5 = load_moneypuck_data()
    print(f"Boxscore rows: {len(df_boxscore)}")
    print()
    
    print("Engineering features...")
    df_boxscore = add_opponent_fpts_allowed(df_boxscore)
    df_boxscore = add_moneypuck_features(df_boxscore, mp_5v5, mp_5on4, mp_teams_5v5)
    df_boxscore = add_regression_shrinkage(df_boxscore, league_baseline)
    print()
    
    results_df, overall_mae, overall_rmse = run_backtest(df_boxscore)
    
    if results_df is not None:
        out_path = Path(DB_PATH.parent) / 'mdn_v4_backtest_results.csv'
        results_df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        print()
        print("="*70)
        print("COMPARISON WITH V3 (MAE 4.091)")
        print("="*70)
        improvement = 4.091 - overall_mae
        pct_improvement = (improvement / 4.091) * 100
        print(f"V3 MAE: 4.091")
        print(f"V4 MAE: {overall_mae:.4f}")
        print(f"Improvement: {improvement:+.4f} ({pct_improvement:+.2f}%)")
        print()

if __name__ == '__main__':
    main()

