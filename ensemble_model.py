"""
Skater Ensemble Model - Combining Multiple Prediction Approaches (OPTIMIZED)

OBJECTIVE:
Ensemble model combining 5 sub-models to beat MDN v3 baseline (MAE 4.091)

SUB-MODELS:
1. Expanding Mean: Simple cumulative average of dk_fpts per player
2. EWM (Exponential Weighted Mean): Recency-weighted with halflife=15
3. Kalman Filter: Noise-filtered trend estimation (Q=0.05, R=30)
4. Opponent-Adjusted Mean: Expanding mean × opponent quality factor
5. TOI-Weighted Projection: Expanding mean × TOI usage trend

WALK-FORWARD BACKTEST:
- Period: Nov 7, 2025 → Feb 5, 2026
- Train: Nov 7 - Dec 7 (30 days)
- Validate: Dec 8 - Feb 5 (remaining days)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

DB_PATH = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db')

# Walk-forward dates
BACKTEST_START = datetime(2025, 11, 7)
BACKTEST_END = datetime(2026, 2, 5)
TRAIN_END = datetime(2025, 12, 7)

# Kalman Filter parameters
KALMAN_Q = 0.05
KALMAN_R = 30.0

# EWM halflife
EWM_HALFLIFE = 15
MIN_GAMES = 3

# Output file
OUTPUT_CSV = Path('/sessions/youthful-funny-faraday/mnt/Code/projection/ensemble_backtest_results.csv')


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_boxscore_data():
    """Load current season boxscore data."""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        player_id, player_name, position, team, opponent,
        game_date, dk_fpts, toi_seconds
    FROM boxscore_skaters
    WHERE game_date BETWEEN ? AND ?
    ORDER BY game_date, player_id
    """
    df = pd.read_sql_query(
        query,
        conn,
        params=(BACKTEST_START.date().isoformat(), BACKTEST_END.date().isoformat())
    )
    conn.close()

    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'player_id']).reset_index(drop=True)

    # Precompute player daily stats efficiently
    df['date_idx'] = df['game_date'].rank(method='dense').astype(int)

    return df


# ============================================================================
# SUB-MODEL PREDICTIONS (VECTORIZED)
# ============================================================================

def kalman_filter_estimate(fpts, Q=0.05, R=30.0):
    """
    Simple 1D Kalman filter to smooth noisy fpts observations.
    Returns smoothed estimate.
    """
    if len(fpts) == 0:
        return 0.0

    x = 0.0  # State estimate
    P = 1.0  # Covariance

    for z in fpts:
        # Predict
        x_pred = x
        P_pred = P + Q

        # Update
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

    return x


def compute_all_predictions(df, as_of_date):
    """
    Compute all 5 sub-model predictions vectorized for efficiency.
    Returns dict of player_id -> dict of predictions
    """
    # Filter to historical data only
    hist_df = df[df['game_date'] < as_of_date].copy()

    if len(hist_df) == 0:
        return {}

    league_avg = hist_df['dk_fpts'].mean()

    preds_by_player = {}

    # Group by player to compute their stats
    for pid in df['player_id'].unique():
        player_games = hist_df[hist_df['player_id'] == pid]

        if len(player_games) < MIN_GAMES:
            continue

        fpts = player_games['dk_fpts'].values
        toi = player_games['toi_seconds'].values

        # 1. Expanding mean (all games equally weighted)
        expanding_mean = fpts.mean()

        # 2. EWM (recent games weighted more)
        ewm_val = float(pd.Series(fpts).ewm(halflife=EWM_HALFLIFE, ignore_na=True).mean().iloc[-1])

        # 3. Kalman filter (noise-filtered trend)
        kalman_val = kalman_filter_estimate(fpts, Q=KALMAN_Q, R=KALMAN_R)

        # 4. Opponent-adjusted (uses last opponent)
        # Get most recent game's opponent
        if len(player_games) > 0:
            last_opponent = player_games.iloc[-1]['opponent']
            # Get how much teams allow on average to this opponent's role
            # Simplified: scale by position average
            pos = player_games.iloc[-1].get('position', 'C')
            pos_avg = hist_df[hist_df['position'] == pos]['dk_fpts'].mean()
            opp_qual = pos_avg if pos_avg > 0 else league_avg
            adjustment = opp_qual / league_avg if league_avg > 0 else 1.0
        else:
            adjustment = 1.0
        opp_adj_val = expanding_mean * adjustment

        # 5. TOI-weighted (recent ice time trend)
        avg_toi = toi.mean()
        # Weight more recent games higher for TOI trend
        if len(toi) >= 5:
            recent_toi = toi[-5:].mean()
        else:
            recent_toi = toi.mean()
        toi_adjustment = recent_toi / avg_toi if avg_toi > 0 else 1.0
        # Bound adjustment to avoid extreme values
        toi_adjustment = np.clip(toi_adjustment, 0.5, 1.5)
        toi_wgt_val = expanding_mean * toi_adjustment

        # Clip all predictions to reasonable range
        preds_by_player[pid] = {
            'expanding': max(0.0, expanding_mean),
            'ewm': max(0.0, ewm_val),
            'kalman': max(0.0, kalman_val),
            'opp_adj': max(0.0, opp_adj_val),
            'toi_wgt': max(0.0, toi_wgt_val),
        }

    return preds_by_player


# ============================================================================
# SIMPLE WEIGHTED AVERAGE
# ============================================================================

def simple_weighted_average(preds_by_player, weights):
    """Combine predictions with fixed weights."""
    combined = {}
    for pid, preds_dict in preds_by_player.items():
        pred_vals = [preds_dict['expanding'], preds_dict['ewm'], preds_dict['kalman'],
                     preds_dict['opp_adj'], preds_dict['toi_wgt']]
        combined[pid] = sum(p * w for p, w in zip(pred_vals, weights))
    return combined


# ============================================================================
# MAIN BACKTEST
# ============================================================================

def run_walk_forward_backtest():
    """Main backtest with two phases: train weights, then validate."""

    print("="*80)
    print("ENSEMBLE MODEL - WALK-FORWARD BACKTEST (OPTIMIZED)")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = load_boxscore_data()
    print(f"  Loaded {len(df)} boxscore records")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    # ========================================================================
    # PHASE 1: COLLECT TRAINING DATA (Nov 7 - Dec 7)
    # ========================================================================

    print(f"\nPhase 1: Collecting training data ({BACKTEST_START.date()} to {TRAIN_END.date()})")

    train_data = []
    current_date = BACKTEST_START

    while current_date <= TRAIN_END:
        date_games = df[df['game_date'] == current_date]

        if len(date_games) == 0:
            current_date += timedelta(days=1)
            continue

        # Get all predictions for this date
        preds_by_player = compute_all_predictions(df, current_date)

        # Collect training data
        for _, row in date_games.iterrows():
            pid = row['player_id']
            if pid not in preds_by_player:
                continue

            pred_dict = preds_by_player[pid]
            train_data.append({
                'actual': row['dk_fpts'],
                'expanding': pred_dict['expanding'],
                'ewm': pred_dict['ewm'],
                'kalman': pred_dict['kalman'],
                'opp_adj': pred_dict['opp_adj'],
                'toi_wgt': pred_dict['toi_wgt'],
            })

        current_date += timedelta(days=1)

    print(f"  Collected {len(train_data)} training samples")

    # ========================================================================
    # OPTIMIZE WEIGHTS via GRID SEARCH
    # ========================================================================

    print("\nPhase 1.5: Optimizing weights via grid search...")

    if len(train_data) > 0:
        y_train = np.array([d['actual'] for d in train_data])
        X_train = np.array([
            [d['expanding'], d['ewm'], d['kalman'], d['opp_adj'], d['toi_wgt']]
            for d in train_data
        ])

        best_weights = None
        best_mae = float('inf')

        # Coarse grid search for speed: 0, 0.25, 0.5, 0.75, 1.0
        weight_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]

        count = 0
        total = len(weight_candidates) ** 4

        for w1 in weight_candidates:
            for w2 in weight_candidates:
                for w3 in weight_candidates:
                    for w4 in weight_candidates:
                        w5 = 1.0 - w1 - w2 - w3 - w4
                        if w5 < 0 or w5 > 1.0:
                            continue

                        count += 1
                        if count % 200 == 0:
                            print(f"  Tested {count} weight combinations...")

                        weights = np.array([w1, w2, w3, w4, w5])
                        preds = X_train @ weights
                        mae = np.mean(np.abs(y_train - preds))

                        if mae < best_mae:
                            best_mae = mae
                            best_weights = weights

        optimal_weights = best_weights if best_weights is not None else np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        print(f"\nOptimal Weights (Training MAE={best_mae:.4f}):")
        print(f"  Expanding Mean:    {optimal_weights[0]:.4f}")
        print(f"  EWM:               {optimal_weights[1]:.4f}")
        print(f"  Kalman Filter:     {optimal_weights[2]:.4f}")
        print(f"  Opponent-Adjusted: {optimal_weights[3]:.4f}")
        print(f"  TOI-Weighted:      {optimal_weights[4]:.4f}")
    else:
        print("WARNING: No training data!")
        optimal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        best_mae = 0.0

    # ========================================================================
    # PHASE 2: VALIDATION (Dec 8 - Feb 5)
    # ========================================================================

    print(f"\nPhase 2: Validation with optimized weights ({(TRAIN_END + timedelta(days=1)).date()} to {BACKTEST_END.date()})")
    print()

    results = []
    daily_stats = []

    current_date = TRAIN_END + timedelta(days=1)
    total_validation_days = (BACKTEST_END - current_date).days

    while current_date <= BACKTEST_END:
        date_games = df[df['game_date'] == current_date]

        if len(date_games) == 0:
            current_date += timedelta(days=1)
            continue

        date_str = current_date.strftime("%Y-%m-%d")

        # Get all predictions for this date
        preds_by_player = compute_all_predictions(df, current_date)

        # Record results
        date_actuals = []
        date_preds = []

        for _, row in date_games.iterrows():
            pid = row['player_id']
            actual = row['dk_fpts']

            if pid in preds_by_player:
                pred_dict = preds_by_player[pid]
                pred_vals = [pred_dict['expanding'], pred_dict['ewm'], pred_dict['kalman'],
                           pred_dict['opp_adj'], pred_dict['toi_wgt']]
                pred_combined = sum(p * w for p, w in zip(pred_vals, optimal_weights))
            else:
                # Fallback: predict 0
                pred_combined = 0.0
                pred_dict = {k: 0.0 for k in ['expanding', 'ewm', 'kalman', 'opp_adj', 'toi_wgt']}

            results.append({
                'game_date': current_date.date(),
                'player_id': pid,
                'player_name': row['player_name'],
                'position': row['position'],
                'actual_fpts': actual,
                'pred_expanding': pred_dict.get('expanding', 0.0),
                'pred_ewm': pred_dict.get('ewm', 0.0),
                'pred_kalman': pred_dict.get('kalman', 0.0),
                'pred_opp_adj': pred_dict.get('opp_adj', 0.0),
                'pred_toi_wgt': pred_dict.get('toi_wgt', 0.0),
                'pred_ensemble': pred_combined,
            })

            date_actuals.append(actual)
            date_preds.append(pred_combined)

        # Daily stats
        if len(date_actuals) > 0:
            date_mae = np.mean(np.abs(np.array(date_actuals) - np.array(date_preds)))
            date_rmse = np.sqrt(np.mean((np.array(date_actuals) - np.array(date_preds))**2))

            daily_stats.append({
                'game_date': current_date.date(),
                'num_players': len(date_games),
                'mae': date_mae,
                'rmse': date_rmse,
            })

            print(f"[{date_str}] {len(date_games):3d} players | MAE={date_mae:.4f} RMSE={date_rmse:.4f}")

        current_date += timedelta(days=1)

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(results)

    ensemble_mae = np.mean(np.abs(results_df['actual_fpts'] - results_df['pred_ensemble']))
    expanding_mae = np.mean(np.abs(results_df['actual_fpts'] - results_df['pred_expanding']))
    ewm_mae = np.mean(np.abs(results_df['actual_fpts'] - results_df['pred_ewm']))
    kalman_mae = np.mean(np.abs(results_df['actual_fpts'] - results_df['pred_kalman']))
    opp_adj_mae = np.mean(np.abs(results_df['actual_fpts'] - results_df['pred_opp_adj']))
    toi_wgt_mae = np.mean(np.abs(results_df['actual_fpts'] - results_df['pred_toi_wgt']))

    print(f"\nValidation Phase MAE (Dec 8 - Feb 5):")
    print(f"  Ensemble (Weighted Average): {ensemble_mae:.4f}")
    print(f"  Expanding Mean:             {expanding_mae:.4f}")
    print(f"  EWM (halflife=15):          {ewm_mae:.4f}")
    print(f"  Kalman Filter:              {kalman_mae:.4f}")
    print(f"  Opponent-Adjusted:          {opp_adj_mae:.4f}")
    print(f"  TOI-Weighted:               {toi_wgt_mae:.4f}")

    print(f"\nBaseline Comparison:")
    print(f"  MDN v3:                     4.0910")
    print(f"  Ensemble:                   {ensemble_mae:.4f}")
    improvement = 4.091 - ensemble_mae
    pct_improvement = 100 * improvement / 4.091
    print(f"  Improvement:                {improvement:+.4f} ({pct_improvement:+.2f}%)")

    # Position breakdown
    print("\n" + "="*80)
    print("POSITION BREAKDOWN (Validation Phase)")
    print("="*80)

    for pos in ['C', 'L', 'R', 'D']:
        pos_data = results_df[results_df['position'] == pos]
        if len(pos_data) > 0:
            pos_mae = np.mean(np.abs(pos_data['actual_fpts'] - pos_data['pred_ensemble']))
            pos_count = len(pos_data)
            print(f"  {pos}: MAE={pos_mae:.4f} ({pos_count} games)")

    # Monthly breakdown
    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN (Validation Phase)")
    print("="*80)

    results_df['month'] = pd.to_datetime(results_df['game_date']).dt.to_period('M')
    for month in sorted(results_df['month'].unique()):
        month_data = results_df[results_df['month'] == month]
        month_mae = np.mean(np.abs(month_data['actual_fpts'] - month_data['pred_ensemble']))
        month_count = len(month_data)
        print(f"  {month}: MAE={month_mae:.4f} ({month_count} games)")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total validation rows: {len(results_df)}")

    # Save daily stats
    daily_stats_df = pd.DataFrame(daily_stats)
    daily_output = OUTPUT_CSV.parent / "ensemble_daily_stats.csv"
    daily_stats_df.to_csv(daily_output, index=False)
    print(f"Daily stats saved to: {daily_output}")

    # Save weights
    weights_output = OUTPUT_CSV.parent / "ensemble_optimal_weights.txt"
    with open(weights_output, 'w') as f:
        f.write("Optimal Ensemble Weights\n")
        f.write("="*40 + "\n")
        f.write(f"Training Data: {BACKTEST_START.date()} to {TRAIN_END.date()}\n")
        f.write(f"Training MAE: {best_mae:.4f}\n\n")
        f.write("Weights:\n")
        f.write(f"  Expanding Mean:    {optimal_weights[0]:.4f}\n")
        f.write(f"  EWM:               {optimal_weights[1]:.4f}\n")
        f.write(f"  Kalman Filter:     {optimal_weights[2]:.4f}\n")
        f.write(f"  Opponent-Adjusted: {optimal_weights[3]:.4f}\n")
        f.write(f"  TOI-Weighted:      {optimal_weights[4]:.4f}\n")
    print(f"Weights saved to: {weights_output}")

    return results_df, optimal_weights


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        results_df, weights = run_walk_forward_backtest()
        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
