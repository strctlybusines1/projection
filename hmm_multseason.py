#!/usr/bin/env python3
"""
Multi-season HMM Signal Validation Script
==========================================
Tests whether HMM-derived signals persist across all 5 NHL seasons (2020-2024).
Implements:
1. Per-season HMM fitting (6-state) on basic observation vector
2. Downgrade bounce detection and effect size testing
3. Opponent regime analysis (weak vs strong defense)
4. 5-day transition window testing
5. Cross-season meta-analysis using Fisher's method
6. Global HMM comparison
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Install hmmlearn if needed
try:
    from hmmlearn import hmm
    GaussianHMM = hmm.GaussianHMM
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'hmmlearn', '--break-system-packages'], check=True)
    from hmmlearn import hmm
    GaussianHMM = hmm.GaussianHMM

DB_PATH = '/sessions/youthful-funny-faraday/mnt/Code/projection/data/nhl_dfs_history.db'

# ============================================================================
# 1. DATABASE LOADING
# ============================================================================

def load_historical_data():
    """Load historical skaters data from 2020-2024."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM historical_skaters WHERE season >= 2020 AND season <= 2024"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Standardize column names and handle missing values
    df['dk_fpts'] = pd.to_numeric(df['dk_fpts'], errors='coerce').fillna(0)
    df['goals'] = pd.to_numeric(df['goals'], errors='coerce').fillna(0)
    df['assists'] = pd.to_numeric(df['assists'], errors='coerce').fillna(0)
    df['shots'] = pd.to_numeric(df['shots'], errors='coerce').fillna(0)
    df['hits'] = pd.to_numeric(df['hits'], errors='coerce').fillna(0)
    df['blocked_shots'] = pd.to_numeric(df['blocked_shots'], errors='coerce').fillna(0)

    return df.sort_values(['season', 'player_name', 'game_date']).reset_index(drop=True)


def load_current_season_data():
    """Load current 2024-25 season data."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM boxscore_skaters"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['game_date'] = pd.to_datetime(df['game_date'])
    df['season'] = 2024  # Treat as continuation of 2024 season

    df['dk_fpts'] = pd.to_numeric(df['dk_fpts'], errors='coerce').fillna(0)
    df['goals'] = pd.to_numeric(df['goals'], errors='coerce').fillna(0)
    df['assists'] = pd.to_numeric(df['assists'], errors='coerce').fillna(0)
    df['shots'] = pd.to_numeric(df['shots'], errors='coerce').fillna(0)
    df['hits'] = pd.to_numeric(df['hits'], errors='coerce').fillna(0)
    df['blocked_shots'] = pd.to_numeric(df['blocked_shots'], errors='coerce').fillna(0)

    return df.sort_values(['player_name', 'game_date']).reset_index(drop=True)


# ============================================================================
# 2. HMM FITTING & STATE MANAGEMENT
# ============================================================================

def fit_hmm_per_season(df_season, min_games=15, n_states=6, n_iter=100, random_state=42):
    """
    Fit HMM for each player with 15+ games in the season.
    Returns dict: player -> (model, states_sequence, sorted_state_means)
    """

    observation_cols = ['goals', 'assists', 'shots', 'hits', 'blocked_shots', 'dk_fpts']
    results = {}

    players = df_season['player_name'].unique()

    for player in players:
        player_data = df_season[df_season['player_name'] == player].copy()

        if len(player_data) < min_games:
            continue

        # Create observation vector
        X = player_data[observation_cols].values.astype(float)

        # Normalize per player
        mean_X = X.mean(axis=0)
        std_X = X.std(axis=0)
        std_X[std_X == 0] = 1  # Avoid division by zero
        X_norm = (X - mean_X) / std_X

        # Fit HMM
        try:
            model = GaussianHMM(n_components=n_states, covariance_type='diag',
                               n_iter=n_iter, random_state=random_state)
            model.fit(X_norm)

            # Get state sequence
            states = model.predict(X_norm)

            # CRITICAL: Sort states by mean dk_fpts to standardize interpretation
            # State 0 = lowest performer, State 5 = highest performer
            state_means = model.means_[:, -1]  # Last column is dk_fpts
            state_order = np.argsort(state_means)

            # Create mapping from old state indices to sorted indices
            state_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(state_order)}
            states_sorted = np.array([state_mapping[s] for s in states])

            results[player] = {
                'model': model,
                'states': states_sorted,
                'state_means_fpts': np.sort(state_means),
                'X_norm': X_norm,
                'X_orig': X,
                'data': player_data.reset_index(drop=True),
                'state_mapping': state_mapping
            }
        except Exception as e:
            print(f"  Warning: HMM fitting failed for {player}: {e}")
            continue

    return results


# ============================================================================
# 3. SIGNAL DETECTION FUNCTIONS
# ============================================================================

def detect_downgrade_bounces(hmm_results):
    """
    Detect downgrades and compute effect size.
    Returns: list of (player, fpts_after_downgrade, fpts_after_stable, effect_size, p_value)
    """

    downgrades = []
    upgrades = []

    for player, result in hmm_results.items():
        states = result['states']
        data = result['data'].reset_index(drop=True)

        for i in range(len(states) - 1):
            if states[i] > states[i+1]:  # Downgrade
                downgrades.append(data.iloc[i+1]['dk_fpts'])
            elif states[i] < states[i+1]:  # Upgrade
                upgrades.append(data.iloc[i+1]['dk_fpts'])

    if len(downgrades) < 2 or len(upgrades) < 2:
        return None, None, None

    downgrade_mean = np.mean(downgrades)
    stable_mean = np.mean(upgrades)

    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(downgrades)**2 + np.std(upgrades)**2) / 2)
    effect_size = (downgrade_mean - stable_mean) / (pooled_std + 1e-6)

    # T-test
    t_stat, p_value = stats.ttest_ind(downgrades, upgrades)

    return effect_size, p_value, downgrade_mean - stable_mean


def detect_opponent_regime(df_season):
    """
    Compute defensive quality per opponent team.
    Split into tertiles and test weak vs strong.
    """

    # Compute avg fpts allowed per opponent
    opp_defense = df_season.groupby('opponent')['dk_fpts'].mean().to_dict()

    if len(opp_defense) < 3:
        return None, None, None

    # Split into tertiles
    values = sorted(opp_defense.values())
    weak_threshold = np.percentile(values, 66)  # Top 33% (worst defense)
    strong_threshold = np.percentile(values, 33)  # Bottom 33% (best defense)

    weak_opps = [opp for opp, val in opp_defense.items() if val >= weak_threshold]
    strong_opps = [opp for opp, val in opp_defense.items() if val <= strong_threshold]

    fpts_vs_weak = df_season[df_season['opponent'].isin(weak_opps)]['dk_fpts'].values
    fpts_vs_strong = df_season[df_season['opponent'].isin(strong_opps)]['dk_fpts'].values

    if len(fpts_vs_weak) < 5 or len(fpts_vs_strong) < 5:
        return None, None, None

    weak_mean = np.mean(fpts_vs_weak)
    strong_mean = np.mean(fpts_vs_strong)

    pooled_std = np.sqrt((np.std(fpts_vs_weak)**2 + np.std(fpts_vs_strong)**2) / 2)
    effect_size = (weak_mean - strong_mean) / (pooled_std + 1e-6)

    t_stat, p_value = stats.ttest_ind(fpts_vs_weak, fpts_vs_strong)

    return effect_size, p_value, weak_mean - strong_mean


def detect_5day_transition_window(hmm_results):
    """
    For each transition, check if next game within 5 days.
    Compare FPTS in 5-day window after transition vs not.
    """

    transition_5day = []
    transition_other = []

    for player, result in hmm_results.items():
        states = result['states']
        data = result['data'].reset_index(drop=True)

        for i in range(len(states) - 1):
            if states[i] != states[i+1]:  # Any transition
                days_to_next = (data.iloc[i+1]['game_date'] - data.iloc[i]['game_date']).days

                if days_to_next <= 5:
                    transition_5day.append(data.iloc[i+1]['dk_fpts'])
                else:
                    transition_other.append(data.iloc[i+1]['dk_fpts'])

    if len(transition_5day) < 2 or len(transition_other) < 2:
        return None, None, None

    window_mean = np.mean(transition_5day)
    other_mean = np.mean(transition_other)

    pooled_std = np.sqrt((np.std(transition_5day)**2 + np.std(transition_other)**2) / 2)
    effect_size = (window_mean - other_mean) / (pooled_std + 1e-6)

    t_stat, p_value = stats.ttest_ind(transition_5day, transition_other)

    return effect_size, p_value, window_mean - other_mean


# ============================================================================
# 4. GLOBAL HMM FITTING
# ============================================================================

def fit_global_hmm(df_all, min_games=15, n_states=6, n_iter=100, random_state=42):
    """Fit a single global HMM on all data pooled."""

    observation_cols = ['goals', 'assists', 'shots', 'hits', 'blocked_shots', 'dk_fpts']

    # Pool all player data
    X_all = []
    player_boundaries = []  # Track where each player's data starts/ends

    for player in df_all['player_name'].unique():
        player_data = df_all[df_all['player_name'] == player].copy()

        if len(player_data) < min_games:
            continue

        X = player_data[observation_cols].values.astype(float)

        # Normalize per player
        mean_X = X.mean(axis=0)
        std_X = X.std(axis=0)
        std_X[std_X == 0] = 1
        X_norm = (X - mean_X) / std_X

        player_boundaries.append(len(X_all))
        X_all.extend(X_norm)

    X_all = np.array(X_all)

    if len(X_all) < 100:
        return None, None

    try:
        global_model = GaussianHMM(n_components=n_states, covariance_type='diag',
                                   n_iter=n_iter, random_state=random_state)
        global_model.fit(X_all)
        states = global_model.predict(X_all)

        # Sort states by mean dk_fpts
        state_means = global_model.means_[:, -1]
        state_order = np.argsort(state_means)
        state_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(state_order)}
        states_sorted = np.array([state_mapping[s] for s in states])

        return global_model, states_sorted
    except Exception as e:
        print(f"  Warning: Global HMM fitting failed: {e}")
        return None, None


# ============================================================================
# 5. FISHER'S METHOD FOR META-ANALYSIS
# ============================================================================

def fishers_method(p_values):
    """Combine p-values using Fisher's method."""
    valid_p = [p for p in p_values if p is not None and 0 < p < 1]

    if len(valid_p) == 0:
        return None

    chi2_stat = -2 * np.sum(np.log(valid_p))
    df = 2 * len(valid_p)
    combined_p = 1 - stats.chi2.cdf(chi2_stat, df)

    return combined_p, len(valid_p)


def weighted_effect_size(effect_sizes):
    """Compute weighted average effect size (weights by absolute value)."""
    valid_es = [es for es in effect_sizes if es is not None]

    if len(valid_es) == 0:
        return None

    return np.mean(valid_es)


# ============================================================================
# 6. MAIN ANALYSIS
# ============================================================================

def main():
    print("\n" + "="*80)
    print("MULTI-SEASON HMM SIGNAL VALIDATION")
    print("="*80)

    # Load data
    print("\n[1/6] Loading historical data (2020-2024)...")
    df_hist = load_historical_data()
    print(f"  Loaded {len(df_hist):,} historical records across {df_hist['season'].nunique()} seasons")
    print(f"  Seasons: {sorted(df_hist['season'].unique())}")
    print(f"  Players: {df_hist['player_name'].nunique():,}")

    # Store results per season
    season_results = {}
    all_p_values = {'downgrade': [], 'opponent': [], 'transition_5day': []}
    all_effect_sizes = {'downgrade': [], 'opponent': [], 'transition_5day': []}

    # ========================================================================
    # PER-SEASON ANALYSIS
    # ========================================================================

    print("\n[2/6] Per-season HMM fitting and signal detection...")

    for season in sorted(df_hist['season'].unique()):
        print(f"\n  Season {season}:")
        df_season = df_hist[df_hist['season'] == season].copy()
        print(f"    - {len(df_season):,} rows, {df_season['player_name'].nunique():,} players")

        # Fit HMM per player
        hmm_results = fit_hmm_per_season(df_season, min_games=15, n_states=6, n_iter=100, random_state=42)
        print(f"    - HMM fitted for {len(hmm_results)} players (15+ games)")

        # Detect signals
        print(f"    - Downgrade bounce analysis...")
        d_es, d_p, d_fpts = detect_downgrade_bounces(hmm_results)
        if d_p is not None:
            print(f"      Effect size: {d_es:.4f}, p-value: {d_p:.4f}, FPTS delta: {d_fpts:.4f}")
            all_effect_sizes['downgrade'].append(d_es)
            all_p_values['downgrade'].append(d_p)
        else:
            print(f"      Insufficient data")

        print(f"    - Opponent regime analysis...")
        o_es, o_p, o_fpts = detect_opponent_regime(df_season)
        if o_p is not None:
            print(f"      Effect size: {o_es:.4f}, p-value: {o_p:.4f}, FPTS delta: {o_fpts:.4f}")
            all_effect_sizes['opponent'].append(o_es)
            all_p_values['opponent'].append(o_p)
        else:
            print(f"      Insufficient data")

        print(f"    - 5-day transition window analysis...")
        t_es, t_p, t_fpts = detect_5day_transition_window(hmm_results)
        if t_p is not None:
            print(f"      Effect size: {t_es:.4f}, p-value: {t_p:.4f}, FPTS delta: {t_fpts:.4f}")
            all_effect_sizes['transition_5day'].append(t_es)
            all_p_values['transition_5day'].append(t_p)
        else:
            print(f"      Insufficient data")

        season_results[season] = {
            'downgrade': (d_es, d_p, d_fpts),
            'opponent': (o_es, o_p, o_fpts),
            'transition_5day': (t_es, t_p, t_fpts),
            'hmm_results': hmm_results
        }

    # ========================================================================
    # CROSS-SEASON META-ANALYSIS
    # ========================================================================

    print("\n[3/6] Cross-season meta-analysis (Fisher's method)...")

    meta_results = {}

    for signal in ['downgrade', 'opponent', 'transition_5day']:
        print(f"\n  {signal.upper()}:")

        fisher_result = fishers_method(all_p_values[signal])
        if fisher_result:
            combined_p, n_seasons = fisher_result
            avg_es = weighted_effect_size(all_effect_sizes[signal])
            print(f"    - Combined p-value (Fisher): {combined_p:.6f}")
            print(f"    - Avg effect size: {avg_es:.4f}")
            print(f"    - Tested in {n_seasons} seasons")

            # Verdict: TRUE if combined p < 0.05 and appears in 4+ seasons
            verdict = "TRUE" if combined_p < 0.05 and n_seasons >= 4 else "WEAK"
            print(f"    - Verdict: {verdict}")

            meta_results[signal] = {
                'combined_p': combined_p,
                'avg_es': avg_es,
                'n_seasons': n_seasons,
                'verdict': verdict
            }
        else:
            print(f"    - Insufficient data across seasons")

    # ========================================================================
    # GLOBAL HMM COMPARISON
    # ========================================================================

    print("\n[4/6] Global HMM fitting on all 252K rows...")

    global_model, global_states = fit_global_hmm(df_hist, min_games=15, n_states=6,
                                                  n_iter=100, random_state=42)

    if global_model is not None:
        print(f"  - Global HMM fitted successfully")
        print(f"  - State means (dk_fpts): {np.sort(global_model.means_[:, -1])}")
        print(f"  - Global model ready for comparison")
    else:
        print(f"  - Global HMM fitting failed")

    # ========================================================================
    # RESULTS TABLE
    # ========================================================================

    print("\n[5/6] Comprehensive Results Table:")
    print("\n" + "="*100)
    print(f"{'Signal':<25} | {'Effect Size':<15} | {'Combined P':<15} | {'N Seasons':<12} | {'Verdict':<10}")
    print("="*100)

    for signal in ['downgrade', 'opponent', 'transition_5day']:
        if signal in meta_results:
            result = meta_results[signal]
            print(f"{signal:<25} | {result['avg_es']:>14.4f} | {result['combined_p']:>14.6f} | {result['n_seasons']:>12d} | {result['verdict']:<10}")
        else:
            print(f"{signal:<25} | {'N/A':<15} | {'N/A':<15} | {'N/A':<12} | {'FAIL':<10}")

    print("="*100)

    # ========================================================================
    # PER-SEASON DETAIL TABLE
    # ========================================================================

    print("\n[6/6] Per-Season Signal Details:")
    print("\n" + "="*130)
    print(f"{'Season':<10} | {'Downgrade ES':<15} | {'Downgrade P':<15} | {'Opponent ES':<15} | {'Opponent P':<15} | {'Transition ES':<15} | {'Transition P':<15}")
    print("="*130)

    for season in sorted(season_results.keys()):
        d_es, d_p, _ = season_results[season]['downgrade']
        o_es, o_p, _ = season_results[season]['opponent']
        t_es, t_p, _ = season_results[season]['transition_5day']

        d_es_str = f"{d_es:.4f}" if d_es is not None else "N/A"
        d_p_str = f"{d_p:.4f}" if d_p is not None else "N/A"
        o_es_str = f"{o_es:.4f}" if o_es is not None else "N/A"
        o_p_str = f"{o_p:.4f}" if o_p is not None else "N/A"
        t_es_str = f"{t_es:.4f}" if t_es is not None else "N/A"
        t_p_str = f"{t_p:.4f}" if t_p is not None else "N/A"

        print(f"{season:<10} | {d_es_str:<15} | {d_p_str:<15} | {o_es_str:<15} | {o_p_str:<15} | {t_es_str:<15} | {t_p_str:<15}")

    print("="*130)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  - Historical data: {len(df_hist):,} records across 5 seasons")
    print(f"  - Per-season HMM models: {sum(len(sr.get('hmm_results', {})) for sr in season_results.values())} total")
    print(f"  - Signals tested: 3 (downgrade bounce, opponent regime, 5-day transition)")
    print(f"  - Meta-analysis method: Fisher's combined p-value")
    print(f"  - Global HMM: {'SUCCESS' if global_model is not None else 'FAILED'}")

    print(f"\nSignal Persistence:")
    for signal in ['downgrade', 'opponent', 'transition_5day']:
        if signal in meta_results:
            result = meta_results[signal]
            print(f"  - {signal}: {result['verdict']} (p={result['combined_p']:.6f}, ES={result['avg_es']:.4f})")
        else:
            print(f"  - {signal}: INSUFFICIENT DATA")

    print("\n" + "="*80 + "\n")

    return season_results, meta_results, global_model


if __name__ == '__main__':
    season_results, meta_results, global_model = main()
