#!/usr/bin/env python3
"""
Ceiling Game Clustering Analysis
==================================

Identifies patterns in what drives ceiling (boom) performances in NHL DFS.
Uses game log data from SQLite to cluster players and game contexts that
produce 20+ FPTS ceiling games, finding actionable signals for projection.

Key Questions:
    1. What player archetypes produce ceiling games?
    2. What game context features predict ceiling games?
    3. Can we identify "ceiling unlocked" conditions before a slate?
    4. Are there clusters of boom games with shared characteristics?

Usage:
    python ceiling_clustering.py                   # Full analysis
    python ceiling_clustering.py --threshold 15    # Lower ceiling threshold
    python ceiling_clustering.py --export          # Export cluster assignments
"""

import sqlite3
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"

# ================================================================
#  Data Loading
# ================================================================

def load_game_logs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load skater and goalie game logs from SQLite."""
    conn = sqlite3.connect(str(DB_PATH))

    sk = pd.read_sql_query("""
        SELECT s.player_id, s.player_name, s.team, s.position,
               s.game_id, s.game_date, s.opponent, s.home_road,
               s.goals, s.assists, s.points, s.plus_minus,
               s.shots, s.pim, s.pp_goals, s.pp_points,
               s.sh_goals, s.sh_points, s.gw_goals, s.ot_goals,
               s.shifts, s.toi_seconds, s.dk_fpts
        FROM game_logs_skaters s
        ORDER BY s.player_id, s.game_date
    """, conn)

    g = pd.read_sql_query("""
        SELECT g.player_id, g.player_name, g.team,
               g.game_id, g.game_date, g.opponent, g.home_road,
               g.games_started, g.decision, g.shots_against,
               g.goals_against, g.saves, g.save_pct, g.shutouts,
               g.toi_seconds, g.dk_fpts
        FROM game_logs_goalies g
        ORDER BY g.player_id, g.game_date
    """, conn)

    conn.close()
    return sk, g


def engineer_features(sk: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-game features including lagged/rolling stats
    that would be available BEFORE the game starts.
    """
    df = sk.copy()

    # Sort by player and date
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    # ── Per-player rolling features (available pre-game) ──
    for pid, group in df.groupby('player_id'):
        idx = group.index

        # Lagged FPTS
        df.loc[idx, 'prev1_fpts'] = group['dk_fpts'].shift(1)
        df.loc[idx, 'prev2_fpts'] = group['dk_fpts'].shift(2)

        # Rolling averages
        df.loc[idx, 'roll3_fpts'] = group['dk_fpts'].rolling(3, min_periods=1).mean().shift(1)
        df.loc[idx, 'roll5_fpts'] = group['dk_fpts'].rolling(5, min_periods=2).mean().shift(1)
        df.loc[idx, 'roll10_fpts'] = group['dk_fpts'].rolling(10, min_periods=3).mean().shift(1)

        # Rolling shots (volume indicator)
        df.loc[idx, 'roll5_shots'] = group['shots'].rolling(5, min_periods=2).mean().shift(1)

        # Rolling PP points (opportunity indicator)
        df.loc[idx, 'roll5_pp'] = group['pp_points'].rolling(5, min_periods=2).mean().shift(1)

        # Volatility (std of recent FPTS)
        df.loc[idx, 'roll5_std'] = group['dk_fpts'].rolling(5, min_periods=3).std().shift(1)
        df.loc[idx, 'roll10_std'] = group['dk_fpts'].rolling(10, min_periods=5).std().shift(1)

        # TOI trend
        df.loc[idx, 'roll5_toi'] = group['toi_seconds'].rolling(5, min_periods=2).mean().shift(1)

        # Streak features
        fpts = group['dk_fpts'].values
        streak = np.zeros(len(fpts))
        for i in range(1, len(fpts)):
            if fpts[i-1] >= 10:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 0
        df.loc[idx, 'hot_streak'] = streak

        # Games played (season progress)
        df.loc[idx, 'game_num'] = np.arange(len(group))

        # Season averages up to that point
        df.loc[idx, 'season_avg_fpts'] = group['dk_fpts'].expanding().mean().shift(1)
        df.loc[idx, 'season_avg_shots'] = group['shots'].expanding().mean().shift(1)

        # Ceiling rate (% of games ≥20 FPTS so far)
        df.loc[idx, 'ceiling_rate'] = (group['dk_fpts'] >= 20).expanding().mean().shift(1)

    # ── Game context features ──
    df['is_home'] = (df['home_road'] == 'H').astype(int) if 'home_road' in df.columns else 0
    df['toi_minutes'] = df['toi_seconds'] / 60 if 'toi_seconds' in df.columns else 0

    # Position encoding
    df['is_center'] = (df['position'] == 'C').astype(int)
    df['is_wing'] = df['position'].isin(['L', 'R', 'LW', 'RW']).astype(int)
    df['is_dman'] = (df['position'] == 'D').astype(int)

    # Binary: is this a ceiling game?
    df['is_ceiling'] = (df['dk_fpts'] >= 20).astype(int)
    df['is_boom'] = (df['dk_fpts'] >= 25).astype(int)

    return df


# ================================================================
#  Clustering: Player Archetypes
# ================================================================

def cluster_player_archetypes(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Cluster players by their ceiling game profile.

    Features per player:
        - ceiling_rate: % of games ≥ 20 FPTS
        - avg_fpts: season average
        - fpts_std: volatility
        - avg_shots: shot volume
        - pp_rate: PP involvement rate
        - max_fpts: highest ceiling game
        - boom_rate: % of games ≥ 25 FPTS
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Build per-player profiles
    profiles = []
    for pid, group in df.groupby('player_id'):
        if len(group) < 10:
            continue

        profile = {
            'player_id': pid,
            'player_name': group.iloc[-1]['player_name'],
            'team': group.iloc[-1]['team'],
            'position': group.iloc[-1]['position'],
            'games': len(group),
            'avg_fpts': group['dk_fpts'].mean(),
            'median_fpts': group['dk_fpts'].median(),
            'fpts_std': group['dk_fpts'].std(),
            'max_fpts': group['dk_fpts'].max(),
            'avg_shots': group['shots'].mean(),
            'avg_pp_pts': group['pp_points'].mean(),
            'ceiling_rate': (group['dk_fpts'] >= 20).mean(),
            'boom_rate': (group['dk_fpts'] >= 25).mean(),
            'bust_rate': (group['dk_fpts'] < 3).mean(),
            'avg_toi': group['toi_seconds'].mean() / 60,
            'consistency': group['dk_fpts'].mean() / max(group['dk_fpts'].std(), 0.1),
        }
        profiles.append(profile)

    prof_df = pd.DataFrame(profiles)

    if len(prof_df) < n_clusters:
        print(f"  Only {len(prof_df)} players with 10+ games — reducing clusters")
        n_clusters = min(n_clusters, max(2, len(prof_df) // 3))

    # Cluster features
    feature_cols = ['avg_fpts', 'fpts_std', 'avg_shots', 'avg_pp_pts',
                    'ceiling_rate', 'bust_rate', 'avg_toi', 'consistency']

    X = prof_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    prof_df['cluster'] = km.fit_predict(X_scaled)

    # Label clusters by characteristics
    cluster_labels = {}
    for c in range(n_clusters):
        cluster_data = prof_df[prof_df['cluster'] == c]
        avg = cluster_data['avg_fpts'].mean()
        ceil = cluster_data['ceiling_rate'].mean()
        bust = cluster_data['bust_rate'].mean()
        vol = cluster_data['fpts_std'].mean()

        if ceil > 0.15:
            cluster_labels[c] = 'ELITE_CEILING'
        elif ceil > 0.05 and avg > 8:
            cluster_labels[c] = 'CONSISTENT_PRODUCER'
        elif vol > 7 and ceil > 0.03:
            cluster_labels[c] = 'VOLATILE_UPSIDE'
        elif bust > 0.5:
            cluster_labels[c] = 'HIGH_BUST_RISK'
        elif avg > 5:
            cluster_labels[c] = 'SAFE_FLOOR'
        else:
            cluster_labels[c] = 'LOW_USAGE'

    prof_df['archetype'] = prof_df['cluster'].map(cluster_labels)

    return prof_df, cluster_labels


# ================================================================
#  Clustering: Ceiling Game Contexts
# ================================================================

def cluster_ceiling_contexts(df: pd.DataFrame, threshold: float = 20.0,
                             n_clusters: int = 4) -> pd.DataFrame:
    """
    Cluster ceiling games by their context features to find
    common conditions that produce boom performances.

    Features (all available pre-game):
        - Recent form (roll3, roll5, roll10)
        - Shot volume trend
        - PP involvement trend
        - Volatility (higher std = more boom potential)
        - Hot streak length
        - Season ceiling rate
        - Home/away
        - Position
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    ceiling = df[df['dk_fpts'] >= threshold].copy()
    ceiling = ceiling.dropna(subset=['roll3_fpts', 'roll5_shots'])

    if len(ceiling) < n_clusters * 2:
        print(f"  Only {len(ceiling)} ceiling games with features — reducing clusters")
        n_clusters = max(2, len(ceiling) // 5)

    feature_cols = [
        'roll3_fpts', 'roll5_fpts', 'roll10_fpts',
        'roll5_shots', 'roll5_pp', 'roll5_std',
        'hot_streak', 'season_avg_fpts', 'ceiling_rate',
        'is_home', 'is_center', 'is_wing', 'is_dman',
        'roll5_toi',
    ]

    available = [c for c in feature_cols if c in ceiling.columns]
    X = ceiling[available].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ceiling['context_cluster'] = km.fit_predict(X_scaled)

    return ceiling, available, km, scaler


# ================================================================
#  Ceiling Probability Model
# ================================================================

def build_ceiling_probability_model(df: pd.DataFrame, threshold: float = 20.0):
    """
    Build a model that predicts P(ceiling game) from pre-game features.
    This directly feeds into the projection pipeline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    # Target
    df = df.dropna(subset=['roll3_fpts', 'roll5_shots']).copy()
    y = (df['dk_fpts'] >= threshold).astype(int)

    feature_cols = [
        'roll3_fpts', 'roll5_fpts', 'roll10_fpts',
        'roll5_shots', 'roll5_pp', 'roll5_std',
        'hot_streak', 'season_avg_fpts', 'ceiling_rate',
        'is_home', 'is_center', 'is_wing', 'is_dman',
        'roll5_toi',
    ]

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic Regression (interpretable)
    lr = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
    lr_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='roc_auc')

    # Gradient Boosting (more powerful)
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=20, random_state=42,
    )
    gb_scores = cross_val_score(gb, X_scaled, y, cv=5, scoring='roc_auc')

    # Fit final models
    lr.fit(X_scaled, y)
    gb.fit(X_scaled, y)

    # Feature importance
    lr_importance = pd.Series(np.abs(lr.coef_[0]), index=available).sort_values(ascending=False)
    gb_importance = pd.Series(gb.feature_importances_, index=available).sort_values(ascending=False)

    return {
        'logistic': lr,
        'gradient_boosting': gb,
        'scaler': scaler,
        'features': available,
        'lr_auc': lr_scores.mean(),
        'gb_auc': gb_scores.mean(),
        'lr_importance': lr_importance,
        'gb_importance': gb_importance,
        'ceiling_rate': y.mean(),
        'n_samples': len(y),
    }


# ================================================================
#  Goalie Ceiling Analysis
# ================================================================

def analyze_goalie_ceilings(g: pd.DataFrame, threshold: float = 20.0):
    """Analyze what drives goalie ceiling games."""
    g = g.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    # Filter to starts only
    starts = g[(g['games_started'] == 1) | (g['toi_seconds'] > 1800)].copy()

    # Build lagged features
    for pid, group in starts.groupby('player_id'):
        idx = group.index
        starts.loc[idx, 'prev_fpts'] = group['dk_fpts'].shift(1)
        starts.loc[idx, 'roll3_fpts'] = group['dk_fpts'].rolling(3, min_periods=1).mean().shift(1)
        starts.loc[idx, 'roll5_fpts'] = group['dk_fpts'].rolling(5, min_periods=2).mean().shift(1)
        starts.loc[idx, 'roll3_saves'] = group['saves'].rolling(3, min_periods=1).mean().shift(1)
        starts.loc[idx, 'roll3_ga'] = group['goals_against'].rolling(3, min_periods=1).mean().shift(1)
        starts.loc[idx, 'season_avg'] = group['dk_fpts'].expanding().mean().shift(1)
        starts.loc[idx, 'game_num'] = np.arange(len(group))

    starts['is_home'] = (starts['home_road'] == 'H').astype(int)
    starts['is_ceiling'] = (starts['dk_fpts'] >= threshold).astype(int)
    starts['is_win'] = (starts['decision'] == 'W').astype(int)

    return starts


# ================================================================
#  Main Analysis
# ================================================================

def run_full_analysis(threshold: float = 20.0, export: bool = False):
    """Run the complete ceiling clustering analysis."""

    sk, g = load_game_logs()

    print("=" * 72)
    print("  CEILING GAME CLUSTERING ANALYSIS")
    print("=" * 72)
    print(f"\n  Data: {len(sk):,} skater games ({sk['player_id'].nunique()} players)")
    print(f"        {len(g):,} goalie games ({g['player_id'].nunique()} players)")
    print(f"  Ceiling threshold: {threshold}+ FPTS")

    n_ceiling = (sk['dk_fpts'] >= threshold).sum()
    print(f"  Ceiling games: {n_ceiling} ({n_ceiling/len(sk)*100:.1f}% of all games)")

    # ── Engineer features ──
    print(f"\n{'─' * 50}")
    print("  ENGINEERING FEATURES...")
    df = engineer_features(sk)

    # ── Player Archetype Clustering ──
    print(f"\n{'─' * 50}")
    print("  PLAYER ARCHETYPE CLUSTERING")

    prof_df, cluster_labels = cluster_player_archetypes(df)

    for archetype in sorted(prof_df['archetype'].unique()):
        cluster_data = prof_df[prof_df['archetype'] == archetype]
        print(f"\n  [{archetype}] ({len(cluster_data)} players)")
        print(f"    Avg FPTS: {cluster_data['avg_fpts'].mean():.1f} | "
              f"Ceiling rate: {cluster_data['ceiling_rate'].mean():.1%} | "
              f"Bust rate: {cluster_data['bust_rate'].mean():.1%} | "
              f"Volatility: {cluster_data['fpts_std'].mean():.1f}")
        # Top 5 players
        top = cluster_data.nlargest(5, 'avg_fpts')
        for _, p in top.iterrows():
            print(f"      {p['player_name']:<22} ({p['team']}) "
                  f"avg={p['avg_fpts']:.1f} ceil={p['ceiling_rate']:.0%} "
                  f"max={p['max_fpts']:.0f} bust={p['bust_rate']:.0%}")

    # ── Ceiling Context Clustering ──
    print(f"\n{'─' * 50}")
    print("  CEILING GAME CONTEXT CLUSTERING")

    ceiling_df, features_used, km, scaler = cluster_ceiling_contexts(df, threshold)

    for c in sorted(ceiling_df['context_cluster'].unique()):
        cluster = ceiling_df[ceiling_df['context_cluster'] == c]
        print(f"\n  Cluster {c} ({len(cluster)} ceiling games):")
        print(f"    Avg roll3_fpts: {cluster['roll3_fpts'].mean():.1f} | "
              f"Avg roll5_shots: {cluster['roll5_shots'].mean():.1f} | "
              f"Avg hot_streak: {cluster['hot_streak'].mean():.1f}")
        print(f"    PP involvement: {cluster['roll5_pp'].mean():.2f} | "
              f"Home: {cluster['is_home'].mean():.0%} | "
              f"Avg FPTS in game: {cluster['dk_fpts'].mean():.1f}")

        # Position breakdown
        pos_pct = cluster['position'].value_counts(normalize=True)
        pos_str = ' '.join(f"{p}:{v:.0%}" for p, v in pos_pct.items())
        print(f"    Positions: {pos_str}")

        # Representative games
        rep = cluster.nlargest(3, 'dk_fpts')
        for _, game in rep.iterrows():
            print(f"      {game['player_name']:<20} {game['dk_fpts']:.0f} FPTS "
                  f"({game['goals']:.0f}G {game['assists']:.0f}A {game['shots']:.0f}SOG) "
                  f"vs {game['opponent']} {game['game_date']}")

    # ── Ceiling Probability Model ──
    print(f"\n{'─' * 50}")
    print("  CEILING PROBABILITY MODEL (P(boom) from pre-game features)")

    model = build_ceiling_probability_model(df, threshold)

    print(f"\n  Logistic Regression AUC: {model['lr_auc']:.3f}")
    print(f"  Gradient Boosting AUC:  {model['gb_auc']:.3f}")
    print(f"  Base ceiling rate:      {model['ceiling_rate']:.1%}")
    print(f"  Training samples:       {model['n_samples']:,}")

    print(f"\n  Feature Importance (Gradient Boosting):")
    for feat, imp in model['gb_importance'].head(10).items():
        print(f"    {feat:<20} {imp:.3f}")

    print(f"\n  Feature Importance (Logistic Regression |coef|):")
    for feat, imp in model['lr_importance'].head(10).items():
        print(f"    {feat:<20} {imp:.3f}")

    # ── Ceiling "Unlock" Conditions ──
    print(f"\n{'─' * 50}")
    print("  CEILING UNLOCK CONDITIONS")
    print("  (When P(ceiling) is significantly above base rate)")

    valid = df.dropna(subset=['roll3_fpts', 'roll5_shots']).copy()
    X = valid[model['features']].fillna(0).values
    X_scaled = model['scaler'].transform(X)
    valid['p_ceiling'] = model['gradient_boosting'].predict_proba(X_scaled)[:, 1]

    # Find threshold conditions
    high_prob = valid[valid['p_ceiling'] > 0.15]
    low_prob = valid[valid['p_ceiling'] < 0.03]

    print(f"\n  HIGH ceiling probability (>15%): {len(high_prob)} game instances")
    print(f"    Actual ceiling rate: {high_prob['is_ceiling'].mean():.1%}")
    print(f"    Avg roll3_fpts: {high_prob['roll3_fpts'].mean():.1f}")
    print(f"    Avg roll5_shots: {high_prob['roll5_shots'].mean():.1f}")
    print(f"    Avg hot_streak: {high_prob['hot_streak'].mean():.1f}")

    print(f"\n  LOW ceiling probability (<3%): {len(low_prob)} game instances")
    print(f"    Actual ceiling rate: {low_prob['is_ceiling'].mean():.1%}")

    # Lift analysis
    base = model['ceiling_rate']
    if len(high_prob) > 0 and base > 0:
        lift = high_prob['is_ceiling'].mean() / base
        print(f"\n  LIFT: High-probability players hit ceiling at {lift:.1f}x the base rate")

    # ── Goalie Analysis ──
    print(f"\n{'─' * 50}")
    print("  GOALIE CEILING ANALYSIS")

    g_starts = analyze_goalie_ceilings(g, threshold)

    if len(g_starts) > 0:
        ceil_g = g_starts[g_starts['is_ceiling'] == 1]
        non_g = g_starts[g_starts['is_ceiling'] == 0]

        print(f"\n  Goalie starts: {len(g_starts)} | Ceiling games: {len(ceil_g)} ({len(ceil_g)/len(g_starts)*100:.1f}%)")

        if len(ceil_g) > 0 and 'roll3_fpts' in g_starts.columns:
            valid_ceil = ceil_g.dropna(subset=['roll3_fpts'])
            valid_non = non_g.dropna(subset=['roll3_fpts'])

            if len(valid_ceil) > 0 and len(valid_non) > 0:
                print(f"\n  Pre-game feature comparison (ceiling vs non-ceiling):")
                for feat in ['roll3_fpts', 'roll5_fpts', 'roll3_saves', 'roll3_ga', 'season_avg']:
                    if feat in valid_ceil.columns:
                        c = valid_ceil[feat].mean()
                        nc = valid_non[feat].mean()
                        print(f"    {feat:<20} Ceiling: {c:.1f}  Non-ceil: {nc:.1f}")

        # Top goalie ceiling games
        top_g = g_starts.nlargest(10, 'dk_fpts')
        print(f"\n  Top 10 goalie ceiling games:")
        for _, game in top_g.iterrows():
            print(f"    {game['player_name']:<22} {game['dk_fpts']:.1f} FPTS "
                  f"({game.get('saves',0):.0f} SV, {game.get('goals_against',0):.0f} GA) "
                  f"vs {game['opponent']} {game['game_date']}")

    # ── Summary & Actionable Insights ──
    print(f"\n{'=' * 72}")
    print("  ACTIONABLE INSIGHTS")
    print(f"{'=' * 72}")

    print(f"""
  1. CEILING PREDICTORS (by importance):
     {', '.join(f'{f}' for f in model['gb_importance'].head(5).index)}

  2. PLAYER ARCHETYPES: {len(prof_df['archetype'].unique())} types identified
     Target: ELITE_CEILING and VOLATILE_UPSIDE in GPP
     Target: CONSISTENT_PRODUCER and SAFE_FLOOR in cash

  3. CEILING UNLOCK: Players with >15% ceiling probability
     hit ceiling at {high_prob['is_ceiling'].mean()/base:.1f}x the base rate
     Key signals: recent form (roll3 > {high_prob['roll3_fpts'].mean():.0f}),
     shot volume (roll5 > {high_prob['roll5_shots'].mean():.0f}), PP involvement

  4. INTEGRATION: Add p_ceiling column to player pool
     Use as multiplier for ceiling projections in optimizer
""")

    # ── Export ──
    if export:
        prof_df.to_csv('ceiling_player_archetypes.csv', index=False)
        ceiling_df.to_csv('ceiling_game_contexts.csv', index=False)
        print("  Exported: ceiling_player_archetypes.csv, ceiling_game_contexts.csv")

    return {
        'player_profiles': prof_df,
        'ceiling_contexts': ceiling_df,
        'probability_model': model,
        'engineered_data': df,
    }


# ================================================================
#  Pipeline Integration Helper
# ================================================================

def predict_ceiling_probability(player_pool: pd.DataFrame,
                                actuals_db: str = None) -> pd.DataFrame:
    """
    Add p_ceiling column to a player pool using stored game logs.
    Call this from the projection pipeline.

    Args:
        player_pool: DataFrame with player projections
        actuals_db: Path to SQLite database (default: data/nhl_dfs_history.db)

    Returns:
        player_pool with 'p_ceiling' column added
    """
    db_path = actuals_db or str(DB_PATH)
    if not Path(db_path).exists():
        player_pool['p_ceiling'] = 0.05  # default
        return player_pool

    conn = sqlite3.connect(db_path)

    # Load recent game logs for feature building
    sk = pd.read_sql_query("""
        SELECT player_id, player_name, team, position, game_date,
               goals, assists, shots, pp_points, toi_seconds, dk_fpts,
               home_road, points, opponent
        FROM game_logs_skaters
        ORDER BY player_id, game_date
    """, conn)
    conn.close()

    if sk.empty:
        player_pool['p_ceiling'] = 0.05
        return player_pool

    # Build features for the latest state of each player
    df = engineer_features(sk)

    # Get latest feature row per player
    latest = df.sort_values('game_date').groupby('player_id').last().reset_index()

    # Build ceiling model
    try:
        model = build_ceiling_probability_model(df)
    except Exception:
        player_pool['p_ceiling'] = 0.05
        return player_pool

    # Score latest features
    feature_cols = model['features']
    X = latest[feature_cols].fillna(0).values
    X_scaled = model['scaler'].transform(X)
    latest['p_ceiling'] = model['gradient_boosting'].predict_proba(X_scaled)[:, 1]

    # Match to player pool
    def ln(n): return str(n).strip().split()[-1].lower()
    latest['_key'] = latest['player_name'].apply(ln) + '_' + latest['team'].str.lower()
    player_pool['_key'] = player_pool['name'].apply(ln) + '_' + player_pool['team'].str.lower()

    pool = player_pool.merge(
        latest[['_key', 'p_ceiling']].drop_duplicates('_key'),
        on='_key', how='left'
    )
    pool['p_ceiling'] = pool['p_ceiling'].fillna(0.05)
    pool.drop(columns=['_key'], inplace=True, errors='ignore')

    return pool


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ceiling Game Clustering Analysis')
    parser.add_argument('--threshold', type=float, default=20.0,
                       help='FPTS threshold for ceiling game (default: 20)')
    parser.add_argument('--export', action='store_true',
                       help='Export cluster assignments to CSV')
    args = parser.parse_args()

    run_full_analysis(threshold=args.threshold, export=args.export)
