#!/usr/bin/env python3
"""
Multi-Stack Line Optimizer — Production Build
==================================================

Combines heuristic + ML-powered line scoring into a unified optimizer.
Generates 16 strategy lineups per slate:
  HEURISTIC (10):
    - chalk: Highest projected stack (PP1 tiebreaker)
    - contrarian_1: Implied rank 3-5, sorted by projection
    - contrarian_2: Implied rank 5-8, sorted by projection
    - value: Cheapest stack with decent projection
    - ceiling: Highest star power + PP1 correlation
    - game_stack: Highest game total
    - pp1_stack: Highest PP1 overlap (L1=PP1 + PP1 D available)
    - dual_chalk: Best same-game dual stack by projection
    - dual_ceiling: Best same-game dual stack by ceiling + PP1
    - dual_game: Best same-game dual stack by game total

  ML-POWERED (6):
    - ml_chalk: Top ML projected stack (Huber L1/L2 split)
    - ml_ceiling: Top quantile-75 ceiling stack
    - ml_contrarian: Implied rank 3-6, sorted by ML projection
    - ml_value: ML proj > 30, cheapest salary
    - ml_dual_chalk: Top ML projected dual stack
    - ml_dual_ceiling: Top ML ceiling dual stack

Key changes (v2):
  - ALL fill modes now use ceiling scoring (salary-mode D fill killed)
  - PP1 D from primary stack team gets massive bonus (4-player correlated core)
  - Goalie from primary stack team preferred (positive correlation)
  - Goalie facing stack team blocked (negative correlation)
  - PP1 overlap scored as primary stack selection criterion

ML models trained walk-forward: all data BEFORE each date, no future leakage.
Models: Split Huber regressors for L1/L2, GradientBoosting quantile (75th) for ceiling.

Contest: ~71-87 entries, 24% cash, 1st place avg 154 FPTS
Strategy: Maximize P(finishing 1st) not P(cashing)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime
from scipy import stats
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "data" / "nhl_dfs_history.db"
BACKTESTS_DIR = PROJECT_DIR / "backtests"
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

SALARY_CAP = 50000
TEAM_NORM = {'nj': 'njd', 'la': 'lak', 'sj': 'sjs', 'tb': 'tbl'}
MIN_ML_TRAIN = 200  # Minimum historical line records to train ML models

def norm(t):
    return TEAM_NORM.get(str(t).lower().strip(), str(t).lower().strip())

# ==============================================================================
# ML FEATURE ENGINEERING
# ==============================================================================

FEATURES = ['line_salary', 'max_salary', 'impl_total', 'game_total', 'has_pp1',
            'fpts_avg_3', 'fpts_avg_5', 'fpts_avg_7', 'fpts_avg_10',
            'fpts_std_5', 'momentum', 'avg_dk_avg', 'avg_dk_ceiling', 'avg_fc_proj',
            'opp_impl_total', 'impl_diff', 'win_pct', 'n_players',
            'total_shots_avg5', 'total_pp_pts_avg5', 'avg_toi_avg5',
            'impl_x_salary', 'form_x_impl', 'ceiling_x_impl', 'pp_x_impl',
            'prev_fpts', 'sal_spread', 'n_pp', 'team_avg_fpts']


def build_line_history(conn):
    """Load and engineer all historical line-level features for ML training."""
    raw = pd.read_sql("""
        SELECT s.player_name, s.team, s.game_date, s.dk_fpts,
               s.goals, s.assists, s.shots, s.pim,
               s.pp_points, s.toi_seconds,
               d.start_line, d.salary, d.team_implied_total, d.opp_implied_total,
               d.game_total, d.pp_unit, d.position, d.dk_avg_fpts, d.dk_ceiling,
               d.fc_proj, d.win_pct
        FROM game_logs_skaters s
        JOIN dk_salaries d ON s.player_name = d.player_name
            AND s.game_date = d.slate_date AND s.team = d.team
        WHERE d.start_line IN ('1', '2')
        AND d.position IN ('C', 'W', 'LW', 'RW', 'L', 'R')
        ORDER BY s.game_date, s.team
    """, conn)
    raw['game_date'] = pd.to_datetime(raw['game_date'])

    def agg_line(grp):
        return pd.Series({
            'line_fpts': grp['dk_fpts'].sum(),
            'line_salary': grp['salary'].sum(),
            'n_players': len(grp),
            'max_salary': grp['salary'].max(),
            'min_salary': grp['salary'].min(),
            'sal_spread': grp['salary'].max() - grp['salary'].min(),
            'impl_total': grp['team_implied_total'].iloc[0],
            'opp_impl_total': grp['opp_implied_total'].iloc[0],
            'game_total': grp['game_total'].iloc[0],
            'win_pct': grp['win_pct'].iloc[0],
            'has_pp1': int(any(str(v) in ['1', '1.0'] for v in grp['pp_unit'])),
            'has_pp2': int(any(str(v) in ['2', '2.0'] for v in grp['pp_unit'])),
            'n_pp': sum(1 for v in grp['pp_unit'] if pd.notna(v) and str(v) in ['1','1.0','2','2.0']),
            'avg_dk_avg': grp['dk_avg_fpts'].mean(),
            'avg_dk_ceiling': grp['dk_ceiling'].mean(),
            'avg_fc_proj': grp['fc_proj'].mean(),
            'total_goals': grp['goals'].sum(),
            'total_assists': grp['assists'].sum(),
            'total_shots': grp['shots'].sum(),
            'total_pp_pts': grp['pp_points'].sum(),
            'avg_toi': grp['toi_seconds'].mean(),
        })

    lines = raw.groupby(['game_date', 'team', 'start_line']).apply(
        agg_line, include_groups=False).reset_index()
    lines = lines[lines['n_players'] >= 2].sort_values(
        ['team', 'start_line', 'game_date']).reset_index(drop=True)

    # Rolling features
    for w in [3, 5, 7, 10]:
        lines[f'fpts_avg_{w}'] = lines.groupby(['team', 'start_line'])['line_fpts'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        lines[f'fpts_std_{w}'] = lines.groupby(['team', 'start_line'])['line_fpts'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std())

    for col in ['total_goals', 'total_assists', 'total_shots', 'total_pp_pts', 'avg_toi']:
        lines[f'{col}_avg5'] = lines.groupby(['team', 'start_line'])[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    lines['momentum'] = lines['fpts_avg_3'] - lines['fpts_avg_10']
    lines['impl_diff'] = lines['impl_total'] - lines['opp_impl_total']
    lines['impl_x_salary'] = lines['impl_total'] * lines['line_salary'] / 100000
    lines['form_x_impl'] = lines['fpts_avg_5'] * lines['impl_total'] / 10
    lines['ceiling_x_impl'] = lines['avg_dk_ceiling'] * lines['impl_total'] / 10
    lines['pp_x_impl'] = lines['has_pp1'] * lines['impl_total']
    lines['prev_fpts'] = lines.groupby(['team', 'start_line'])['line_fpts'].shift(1)
    lines['is_l1'] = (lines['start_line'] == '1').astype(int)

    # Target encoding for team
    team_means = lines.groupby('team')['line_fpts'].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean())
    lines['team_avg_fpts'] = team_means

    return lines


def build_slate_features(team, line_num, fwd_list, train_lines):
    """Build ML feature vector for a team's line on current slate."""
    hist = train_lines[(train_lines['team'] == team) &
                       (train_lines['start_line'] == str(line_num))]

    fwd_salary = sum(f['salary'] for f in fwd_list)
    max_sal = max(f['salary'] for f in fwd_list) if fwd_list else 0
    min_sal = min(f['salary'] for f in fwd_list) if fwd_list else 0

    impl = fwd_list[0].get('team_implied_total', 0) if fwd_list else 0
    impl = impl if pd.notna(impl) else 0
    opp_impl = fwd_list[0].get('opp_implied_total', 0) if fwd_list else 0
    opp_impl = opp_impl if pd.notna(opp_impl) else 0
    gt = fwd_list[0].get('game_total', 0) if fwd_list else 0
    gt = gt if pd.notna(gt) else 0
    wp = fwd_list[0].get('win_pct', 0.5) if fwd_list else 0.5
    wp = wp if pd.notna(wp) else 0.5

    has_pp1 = int(any(str(f.get('pp_unit', '')).strip() in ['1', '1.0'] for f in fwd_list))
    n_pp = sum(1 for f in fwd_list if pd.notna(f.get('pp_unit')) and
               str(f.get('pp_unit', '')).strip() in ['1','1.0','2','2.0'])

    dk_avg = np.mean([f.get('dk_avg_fpts', 0) or 0 for f in fwd_list])
    dk_ceil = np.mean([f.get('dk_ceiling', 0) or 0 for f in fwd_list])
    fc = np.mean([f.get('fc_proj', 0) or 0 for f in fwd_list])

    if not hist.empty:
        last = hist.iloc[-1]
        fpts_avg_3 = last.get('fpts_avg_3', dk_avg) or dk_avg
        fpts_avg_5 = last.get('fpts_avg_5', dk_avg) or dk_avg
        fpts_avg_7 = last.get('fpts_avg_7', dk_avg) or dk_avg
        fpts_avg_10 = last.get('fpts_avg_10', dk_avg) or dk_avg
        fpts_std_5 = last.get('fpts_std_5', 5) or 5
        mom = last.get('momentum', 0) or 0
        shots_avg = last.get('total_shots_avg5', 8) or 8
        pp_pts_avg = last.get('total_pp_pts_avg5', 0.5) or 0.5
        toi_avg = last.get('avg_toi_avg5', 1000) or 1000
        prev = last.get('prev_fpts', dk_avg) or dk_avg
        team_avg = last.get('team_avg_fpts', dk_avg) or dk_avg
    else:
        fpts_avg_3 = fpts_avg_5 = fpts_avg_7 = fpts_avg_10 = dk_avg
        fpts_std_5 = 5
        mom = 0
        shots_avg, pp_pts_avg, toi_avg = 8, 0.5, 1000
        prev = dk_avg
        team_avg = dk_avg

    feats = [fwd_salary, max_sal, impl, gt, has_pp1,
             fpts_avg_3, fpts_avg_5, fpts_avg_7, fpts_avg_10,
             fpts_std_5, mom, dk_avg, dk_ceil, fc,
             opp_impl, impl - opp_impl, wp, len(fwd_list),
             shots_avg, pp_pts_avg, toi_avg,
             impl * fwd_salary / 100000, fpts_avg_5 * impl / 10,
             dk_ceil * impl / 10, has_pp1 * impl,
             prev, max_sal - min_sal, n_pp, team_avg]
    return np.array(feats, dtype=np.float64)


def train_ml_models(train_lines, target_date):
    """Train ML models on all line data before target_date. Returns (models, scaler) or None."""
    train_data = train_lines[train_lines['game_date'] < pd.Timestamp(target_date)].copy()
    train_valid = train_data.dropna(subset=FEATURES + ['line_fpts'])

    if len(train_valid) < MIN_ML_TRAIN:
        return None

    X_train = train_valid[FEATURES].values
    y_train = train_valid['line_fpts'].values

    l1_train = train_valid[train_valid['start_line'] == '1']
    l2_train = train_valid[train_valid['start_line'] == '2']

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)

    huber_l1 = HuberRegressor(max_iter=300)
    huber_l2 = HuberRegressor(max_iter=300)
    quantile_75 = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, loss='quantile', alpha=0.75, random_state=42)

    # Split L1/L2 training (key insight: L2 is 26% easier to predict)
    if len(l1_train.dropna(subset=FEATURES)) >= 50:
        X_l1 = scaler.transform(l1_train.dropna(subset=FEATURES)[FEATURES].values)
        y_l1 = l1_train.dropna(subset=FEATURES)['line_fpts'].values
        huber_l1.fit(X_l1, y_l1)
    else:
        huber_l1.fit(X_train_sc, y_train)

    if len(l2_train.dropna(subset=FEATURES)) >= 50:
        X_l2 = scaler.transform(l2_train.dropna(subset=FEATURES)[FEATURES].values)
        y_l2 = l2_train.dropna(subset=FEATURES)['line_fpts'].values
        huber_l2.fit(X_l2, y_l2)
    else:
        huber_l2.fit(X_train_sc, y_train)

    X_clean = X_train[~np.isnan(X_train).any(axis=1)]
    y_clean = y_train[~np.isnan(X_train).any(axis=1)]
    quantile_75.fit(X_clean, y_clean)

    return {
        'huber_l1': huber_l1,
        'huber_l2': huber_l2,
        'quantile_75': quantile_75,
        'scaler': scaler,
        'train_lines': train_data,  # Pass through for build_slate_features
    }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_actuals(date_str, conn):
    sk = pd.read_sql("SELECT player_name as name, team, dk_fpts FROM game_logs_skaters WHERE game_date = ?",
                      conn, params=(date_str,))
    gl = pd.read_sql("SELECT player_name as name, team, dk_fpts FROM game_logs_goalies WHERE game_date = ?",
                      conn, params=(date_str,))
    actuals = pd.concat([sk, gl], ignore_index=True)
    actuals['_key'] = actuals['name'].str.lower().str.strip() + '_' + actuals['team'].apply(norm)
    return actuals


def score_players(players, actuals):
    total = 0.0
    matched = 0
    scratched = 0
    for p in players:
        name = str(p.get('name', '')).lower().strip()
        team = norm(str(p.get('team', '')))
        key = f"{name}_{team}"
        exact = actuals[actuals['_key'] == key]
        if not exact.empty:
            total += exact.iloc[0]['dk_fpts']
            matched += 1
            continue
        team_players = actuals[actuals['_key'].str.endswith(f"_{team}")]
        best_score = 0
        best_fpts = 0
        for _, row in team_players.iterrows():
            score = SequenceMatcher(None, name, row['name'].lower().strip()).ratio()
            if score > best_score:
                best_score = score
                best_fpts = row['dk_fpts']
        if best_score >= 0.70:
            total += best_fpts
            matched += 1
        else:
            scratched += 1
    return total, matched, scratched


# ==============================================================================
# LINE SCORING — Heuristic (original method)
# ==============================================================================

LINE_BASELINES = {1: 25.8, 2: 20.5, 3: 13.6, 4: 7.2}

def score_line_heuristic(fwd_list, line_num, team_impl, game_total,
                         all_logs, date_str):
    """Score a forward line using heuristic method. Returns projected FPTS."""
    n_fwd = len(fwd_list)
    if n_fwd < 2:
        return 0.0

    base = LINE_BASELINES.get(line_num, 10.0)
    fwd_salary = sum(f['salary'] for f in fwd_list)

    avg_sal = {1: 16764, 2: 13172, 3: 9401, 4: 7643}.get(line_num, 10000)
    sal_factor = np.clip(fwd_salary / avg_sal, 0.65, 1.45) if avg_sal > 0 else 1.0
    impl_factor = np.clip(team_impl / 3.0, 0.70, 1.40) if team_impl > 0 else 1.0
    pace_factor = np.clip(game_total / 6.0, 0.85, 1.20) if game_total > 0 else 1.0

    form_factor = 1.0
    if all_logs is not None:
        target_date = pd.Timestamp(date_str)
        prior = all_logs[all_logs['game_date'] < target_date]
        player_forms = []
        for fwd in fwd_list:
            pkey = fwd.get('_key', '')
            if not pkey:
                pkey = str(fwd.get('player_name', '')).lower().strip() + '_' + norm(str(fwd.get('team', '')))
            plogs = prior[prior['_key'] == pkey].sort_values('game_date', ascending=False).head(5)
            if len(plogs) >= 3:
                player_forms.append(plogs['dk_fpts'].mean())
        if player_forms:
            avg_form = np.mean(player_forms)
            expected_pp = base / max(n_fwd, 1)
            if expected_pp > 0:
                form_factor = np.clip(avg_form / expected_pp, 0.55, 1.65)

    pp_boost = 1.0
    for fwd in fwd_list:
        pp = fwd.get('pp_unit')
        if pd.notna(pp) and str(pp) in ['1', '1.0']:
            pp_boost = 1.08
            break
        elif pd.notna(pp) and str(pp) in ['2', '2.0']:
            pp_boost = max(pp_boost, 1.03)

    projected = base * (
        0.35 * sal_factor +
        0.30 * form_factor +
        0.15 * impl_factor +
        0.10 * pace_factor +
        0.10 * 1.0
    ) * pp_boost

    return projected


# ==============================================================================
# STACK BUILDER — Unified (heuristic + ML)
# ==============================================================================

def _find_game_matchups(dk_pool, team_data):
    """Find which teams play each other based on matching game_total + implied totals.
    Returns list of (team_a, team_b, game_total) tuples.
    """
    # Group teams by game_total (teams in the same game share the same game_total)
    team_gt = {}
    for team in team_data:
        gt = team_data[team]['game_total']
        impl = team_data[team]['impl']
        if gt > 0:
            team_gt[team] = (gt, impl)

    # Also check from dk_pool for more precise matching (opp_implied_total)
    matchups = []
    seen = set()

    # Method: match teams where team_a's game_total == team_b's game_total
    # AND team_a's impl == team_b's opp_impl (and vice versa)
    team_info = {}
    for team in team_data:
        rows = dk_pool[dk_pool['team'] == team]
        if rows.empty:
            continue
        r = rows.iloc[0]
        team_info[team] = {
            'gt': team_data[team]['game_total'],
            'impl': team_data[team]['impl'],
            'opp_impl': r['opp_implied_total'] if pd.notna(r.get('opp_implied_total', None)) else 0,
        }

    teams = list(team_info.keys())
    for i, ta in enumerate(teams):
        for tb in teams[i+1:]:
            info_a, info_b = team_info[ta], team_info[tb]
            # Same game if game_total matches and implied totals cross-match
            gt_match = abs(info_a['gt'] - info_b['gt']) < 0.1 and info_a['gt'] > 0
            impl_match = (abs(info_a['impl'] - info_b['opp_impl']) < 0.3 and
                          abs(info_b['impl'] - info_a['opp_impl']) < 0.3)
            if gt_match and impl_match:
                key = tuple(sorted([ta, tb]))
                if key not in seen:
                    seen.add(key)
                    matchups.append((ta, tb, info_a['gt']))

    return matchups


def _get_team_lines(dk_pool):
    """Extract L1 and L2 forward groups per team from DK pool.
    Also tracks PP1 overlap and available PP1 D for stack correlation scoring.
    Returns dict: team -> {'l1': [records], 'l2': [records], 'impl': float,
                           'game_total': float, 'l1_pp1_count': int,
                           'has_pp1_d': bool, 'pp1_d': record or None}
    """
    has_lines = dk_pool[dk_pool['start_line'].isin(['1', '2', '3', '4'])].copy()
    has_lines['_key'] = has_lines['player_name'].str.lower().str.strip() + '_' + has_lines['team'].apply(norm)
    fwd_pos = ['C', 'W', 'LW', 'RW', 'L', 'R']

    team_data = {}
    for team, team_df in has_lines.groupby('team'):
        l1 = team_df[(team_df['start_line'] == '1') & (team_df['position'].isin(fwd_pos))].to_dict('records')
        l2 = team_df[(team_df['start_line'] == '2') & (team_df['position'].isin(fwd_pos))].to_dict('records')
        if len(l1) < 2 or len(l2) < 2:
            continue
        impl = team_df['team_implied_total'].iloc[0] if pd.notna(team_df['team_implied_total'].iloc[0]) else 0
        gt = team_df['game_total'].iloc[0] if pd.notna(team_df['game_total'].iloc[0]) else 0

        # PP1 overlap: how many L1 forwards are also PP1?
        l1_pp1_count = sum(1 for f in l1
                           if pd.notna(f.get('pp_unit')) and str(f['pp_unit']).strip() in ['1', '1.0'])

        # Check for PP1 D on this team (from full dk_pool, not just forwards)
        team_pool = dk_pool[(dk_pool['team'] == team) & (dk_pool['position'] == 'D')]
        pp1_d_rows = team_pool[team_pool['pp_unit'].apply(
            lambda x: pd.notna(x) and str(x).strip() in ['1', '1.0'])]
        has_pp1_d = len(pp1_d_rows) > 0
        pp1_d = pp1_d_rows.iloc[0].to_dict() if has_pp1_d else None

        team_data[team] = {
            'l1': l1, 'l2': l2, 'impl': impl, 'game_total': gt,
            'l1_pp1_count': l1_pp1_count,
            'has_pp1_d': has_pp1_d,
            'pp1_d': pp1_d,
        }
    return team_data


def _check_position_feasibility(forwards):
    """Check that a set of forwards can fill 2C + 3W + 1UTIL (C/W only)."""
    n_c = sum(1 for f in forwards if f['position'] == 'C')
    n_w = sum(1 for f in forwards if f['position'] in ['W', 'LW', 'RW', 'L', 'R'])
    # Need at least 2C (or 1C + fill UTIL from C), and at least 3W (or 2W + fill UTIL from W)
    # With 6 forwards into 6 slots (2C, 3W, 1UTIL): need n_c >= 1 and n_w >= 2
    return n_c >= 1 and n_w >= 2


def build_all_stacks(dk_pool, all_logs, date_str, ml_models=None):
    """
    Build scored stack candidates — BOTH single-team and dual-team.

    Single-team stacks: L1+L2 from one team (6 fwds, original approach)
    Dual-team stacks: L1 from Team A + L1 from Team B (3+3 = 6 fwds)

    87% of $14 WTA winners use 2+ team stacks (avg 2.2 Tstacks).
    Dual-team stacks are the key to reaching 130+ FPTS.

    Returns list of stack dicts, sorted by combo_proj.
    """
    team_data = _get_team_lines(dk_pool)

    stacks = []

    # ─── PHASE 1: Single-team stacks (L1+L2 from same team) ───────────────
    for team, td in team_data.items():
        l1_fwds, l2_fwds = td['l1'], td['l2']
        team_impl, game_total = td['impl'], td['game_total']

        l1_proj_h = score_line_heuristic(l1_fwds, 1, team_impl, game_total, all_logs, date_str)
        l2_proj_h = score_line_heuristic(l2_fwds, 2, team_impl, game_total, all_logs, date_str)

        all_fwds = l1_fwds + l2_fwds
        if not _check_position_feasibility(all_fwds):
            continue

        fwd_salary = sum(f['salary'] for f in all_fwds)

        # PP1 overlap score: teams where L1 = PP1 get massive correlation bonus
        pp1_overlap = td.get('l1_pp1_count', 0)
        has_pp1_d = td.get('has_pp1_d', False)
        # PP1 score: each L1 forward on PP1 = +1, PP1 D available = +1
        pp1_score = pp1_overlap + (1 if has_pp1_d else 0)

        stack = {
            'team': team,
            'teams': [team],  # List of teams in stack
            'stack_type': 'single',
            'l1_proj_h': l1_proj_h,
            'l2_proj_h': l2_proj_h,
            'combo_proj': l1_proj_h + l2_proj_h,
            'fwd_salary': fwd_salary,
            'remaining_budget': SALARY_CAP - fwd_salary,
            'n_fwd': len(all_fwds),
            'n_c': sum(1 for f in all_fwds if f['position'] == 'C'),
            'n_w': sum(1 for f in all_fwds if f['position'] in ['W', 'LW', 'RW', 'L', 'R']),
            'forwards': all_fwds,
            'team_impl': team_impl,
            'game_total': game_total,
            'impl_rank': 0,
            'pp1_score': pp1_score,
            'pp1_overlap': pp1_overlap,
            'has_pp1_d': has_pp1_d,
            'ml_proj': None,
            'ml_ceiling': None,
        }

        # ML projections
        if ml_models is not None:
            scaler = ml_models['scaler']
            train_lines = ml_models['train_lines']

            f1 = build_slate_features(team, 1, l1_fwds, train_lines)
            f2 = build_slate_features(team, 2, l2_fwds, train_lines)
            f1 = np.nan_to_num(f1, nan=0.0)
            f2 = np.nan_to_num(f2, nan=0.0)

            f1_sc = scaler.transform(f1.reshape(1, -1))
            f2_sc = scaler.transform(f2.reshape(1, -1))

            l1_ml = float(ml_models['huber_l1'].predict(f1_sc)[0])
            l2_ml = float(ml_models['huber_l2'].predict(f2_sc)[0])
            ceil = float(ml_models['quantile_75'].predict(
                np.vstack([f1.reshape(1,-1), f2.reshape(1,-1)])).sum())

            stack['ml_proj'] = l1_ml + l2_ml
            stack['ml_ceiling'] = ceil

        stacks.append(stack)

    # ─── PHASE 2: Same-game dual stacks (optimizer.py 4+3 architecture) ────
    # $14 winners avg 2.2 team stacks. The key: both teams in the SAME GAME
    # for maximum correlated variance. When a game goes high-scoring, both
    # line stacks benefit simultaneously.
    #
    # Architecture: 3 L1 fwds from A + 3 L1 fwds from B = 6 forwards (2 teams)
    # Then fill_lineup adds 2D + 1G, ideally D from primary team (PP1) = 4+3 pattern
    #
    # Find game matchups: teams sharing the same game_total + matching opp_implied
    game_matchups = _find_game_matchups(dk_pool, team_data)

    for team_a, team_b, gt in game_matchups:
        if team_a not in team_data or team_b not in team_data:
            continue
        td_a, td_b = team_data[team_a], team_data[team_b]

        # Try both orderings (A primary + B secondary, B primary + A secondary)
        for primary, secondary, td_p, td_s in [
            (team_a, team_b, td_a, td_b),
            (team_b, team_a, td_b, td_a),
        ]:
            fwds_p = td_p['l1']  # Primary L1
            fwds_s = td_s['l1']  # Secondary L1
            all_fwds = fwds_p + fwds_s

            if not _check_position_feasibility(all_fwds):
                continue

            fwd_salary = sum(f['salary'] for f in all_fwds)
            remaining = SALARY_CAP - fwd_salary
            if remaining < 7500:  # Need at least $2500 per remaining (2D + 1G)
                continue

            proj_p = score_line_heuristic(fwds_p, 1, td_p['impl'], gt, all_logs, date_str)
            proj_s = score_line_heuristic(fwds_s, 1, td_s['impl'], gt, all_logs, date_str)

            # PP1 overlap for dual: primary team's PP1 score matters most
            pp1_overlap_p = td_p.get('l1_pp1_count', 0)
            has_pp1_d_p = td_p.get('has_pp1_d', False)
            pp1_score = pp1_overlap_p + (1 if has_pp1_d_p else 0)

            stack = {
                'team': f"{primary}+{secondary}",
                'teams': [primary, secondary],
                'stack_type': 'dual',
                'primary_team': primary,
                'secondary_team': secondary,
                'primary_line': 'l1',
                'secondary_line': 'l1',
                'l1_proj_h': proj_p,
                'l2_proj_h': proj_s,
                'combo_proj': proj_p + proj_s,
                'fwd_salary': fwd_salary,
                'remaining_budget': remaining,
                'n_fwd': len(all_fwds),
                'n_c': sum(1 for f in all_fwds if f['position'] == 'C'),
                'n_w': sum(1 for f in all_fwds if f['position'] in ['W', 'LW', 'RW', 'L', 'R']),
                'forwards': all_fwds,
                'team_impl': (td_p['impl'] + td_s['impl']) / 2,
                'game_total': gt,
                'impl_rank': 0,
                'pp1_score': pp1_score,
                'pp1_overlap': pp1_overlap_p,
                'has_pp1_d': has_pp1_d_p,
                'ml_proj': None,
                'ml_ceiling': None,
            }

            # ML projections for dual stacks
            if ml_models is not None:
                scaler = ml_models['scaler']
                train_lines = ml_models['train_lines']

                fp = build_slate_features(primary, 1, fwds_p, train_lines)
                fs = build_slate_features(secondary, 1, fwds_s, train_lines)
                fp = np.nan_to_num(fp, nan=0.0)
                fs = np.nan_to_num(fs, nan=0.0)

                fp_sc = scaler.transform(fp.reshape(1, -1))
                fs_sc = scaler.transform(fs.reshape(1, -1))

                ml_p = float(ml_models['huber_l1'].predict(fp_sc)[0])
                ml_s = float(ml_models['huber_l1'].predict(fs_sc)[0])
                ceil = float(ml_models['quantile_75'].predict(
                    np.vstack([fp.reshape(1,-1), fs.reshape(1,-1)])).sum())

                stack['ml_proj'] = ml_p + ml_s
                stack['ml_ceiling'] = ceil

            stacks.append(stack)

    # Rank by avg implied total (single stacks use team_impl, dual use avg)
    stacks.sort(key=lambda x: x['team_impl'], reverse=True)
    for i, s in enumerate(stacks):
        s['impl_rank'] = i + 1

    stacks.sort(key=lambda x: x['combo_proj'], reverse=True)
    return stacks


def _score_d_ceiling(d_row, stack_team=None, stack_game_teams=None, stack_teams=None):
    """Score a defenseman for ceiling potential (WTA optimization).

    PP1 D on high-implied teams in the same game as our stack = maximum
    correlated upside. This is the key to 130+ FPTS lineups.

    Architecture: D from primary team creates 4-player correlated stack.
    This applies to BOTH single and dual stacks.
    """
    score = 0.0

    # PP1 is the #1 D signal: 8.0 avg FPTS vs 2.9 non-PP (2.8x multiplier)
    pp = d_row.get('pp_unit', None)
    is_pp1 = pd.notna(pp) and str(pp).strip() in ['1', '1.0']
    if is_pp1:
        score += 40  # Heavy PP1 bonus
    elif pd.notna(pp) and str(pp).strip() in ['2', '2.0']:
        score += 15

    # DK ceiling (rho=0.356 with actual)
    ceiling = d_row.get('dk_ceiling', 0)
    if pd.notna(ceiling) and ceiling > 0:
        score += ceiling * 1.5

    # Salary (rho=0.387 — still the best single predictor)
    salary = d_row.get('salary', 3000)
    score += salary / 500  # ~6-16 points from salary range

    # Team implied total (rho=0.059 but matters for ceiling)
    impl = d_row.get('team_implied_total', 3.0)
    if pd.notna(impl) and impl > 0:
        score += impl * 3

    # Game correlation bonus: D from same game as our forwards
    # Correlated variance = when the game goes high-scoring, EVERYONE benefits
    d_team = norm(str(d_row.get('team', '')))
    if stack_game_teams is not None:
        if d_team in stack_game_teams:
            score += 20  # Same-game correlation bonus

    # ── PRIMARY TEAM D BONUS (applies to ALL stack types) ──
    # D from primary stack team creates correlated 4-player core.
    # PP1 D from primary = optimal ceiling construction.
    if stack_teams:
        primary_team = norm(str(stack_teams[0]))
        if d_team == primary_team:
            score += 30  # Primary team D bonus (creates 4-player stack)
            if is_pp1:
                score += 40  # PP1 D on primary team = maximum correlation
        # For dual stacks, secondary team D is also correlated (but less)
        elif len(stack_teams) > 1 and d_team == norm(str(stack_teams[1])):
            score += 15  # Secondary team D (still same-game correlated)
            if is_pp1:
                score += 20

    # fc_proj (rho=0.360)
    fc = d_row.get('fc_proj', 0)
    if pd.notna(fc) and fc > 0:
        score += fc * 2

    return score


def _score_g_ceiling(g_row, stack_team=None, stack_teams=None, opponent_teams=None):
    """Score a goalie for ceiling potential.

    Goalies are near-unpredictable (best rho=0.111). For WTA:
    - dk_ceiling is the best signal we have
    - PREFER goalie from primary stack team (positive correlation: stack scores → goalie wins)
    - PENALIZE goalie facing our primary stack (negative correlation — soft, not hard block)
    - Cheaper goalies free budget for better D (where signal is stronger)
    """
    score = 0.0
    g_team = norm(str(g_row.get('team', '')))
    all_stack_teams = [norm(str(t)) for t in (stack_teams or [stack_team] if stack_team else [])]

    # dk_ceiling (best predictor, rho=0.111)
    ceiling = g_row.get('dk_ceiling', 0)
    if pd.notna(ceiling) and ceiling > 0:
        score += ceiling * 2

    # fc_proj (rho=0.099)
    fc = g_row.get('fc_proj', 0)
    if pd.notna(fc) and fc > 0:
        score += fc * 2

    # ── POSITIVE CORRELATION: goalie from primary stack team ──
    # If our stack goes off (5 goals), the goalie gets the win + save bonus
    if all_stack_teams:
        if g_team == all_stack_teams[0]:
            score += 25  # Preference for primary team goalie
        elif len(all_stack_teams) > 1 and g_team == all_stack_teams[1]:
            score += 10  # Moderate preference for secondary team goalie

    # ── NEGATIVE CORRELATION PENALTY: goalie facing our stack ──
    # If we stack EDM and play the goalie who faces EDM, when EDM scores
    # the goalie gets shelled. Soft penalty (not hard block — sometimes
    # the opponent goalie is still the best option on thin slates).
    if opponent_teams and g_team in opponent_teams:
        score -= 20  # Penalty, not block

    # Opp implied (lower = fewer goals against = better for goalie)
    opp_impl = g_row.get('opp_implied_total', 3.0)
    if pd.notna(opp_impl):
        score += max(0, (4.0 - opp_impl)) * 5

    # dk_avg (rho=0.061)
    dk_avg = g_row.get('dk_avg_fpts', 0)
    if pd.notna(dk_avg) and dk_avg > 0:
        score += dk_avg

    return score


def fill_lineup(stack, dk_pool, actuals=None, fill_mode='ceiling'):
    """Fill a stack with 2D + 1G. Returns player list or None.

    fill_mode:
      'salary'  — Original method: pick most expensive (safe floor)
      'ceiling' — PP1-weighted ceiling scoring (WTA optimized)
      'game_corr' — Same as ceiling but extra weight on game correlation
    """
    fwd_names = set(f['player_name'] for f in stack['forwards'])
    remaining = stack['remaining_budget']
    stack_teams = [norm(str(t)) for t in stack.get('teams', [stack.get('team', '')])]
    stack_team = stack_teams[0]  # Primary team for goalie logic

    # Identify teams in the same game as our stack for correlation
    stack_game_teams = set()
    for f in stack['forwards']:
        f_team = norm(str(f.get('team', '')))
        stack_game_teams.add(f_team)
        # Also add opponent (same game)
        opp_impl = f.get('opp_implied_total', None)
        # We can identify opponent from the pool
    # Get all teams that play against our stack team
    opp_rows = dk_pool[dk_pool['team'].apply(norm) == stack_team]
    if not opp_rows.empty:
        # Find opponent team from game_total matching
        for _, row in dk_pool.iterrows():
            if norm(str(row['team'])) != stack_team and pd.notna(row.get('game_total')):
                # Check if this team plays vs our stack (same game total + opp impl match)
                pass
    # Simpler: just use all teams on the slate that share the same game_total
    stack_gt = stack.get('game_total', 0)
    if stack_gt > 0:
        for _, row in dk_pool.drop_duplicates('team').iterrows():
            if pd.notna(row.get('game_total')) and abs(row['game_total'] - stack_gt) < 0.01:
                stack_game_teams.add(norm(str(row['team'])))

    pool_d = dk_pool[(dk_pool['position'] == 'D') & (~dk_pool['player_name'].isin(fwd_names))].copy()
    pool_g = dk_pool[dk_pool['position'] == 'G'].copy()

    actual_keys = None
    if actuals is not None:
        actual_keys = set(actuals['_key'].tolist())
        pool_d['_k'] = pool_d['player_name'].str.lower().str.strip() + '_' + pool_d['team'].apply(norm)
        pool_g['_k'] = pool_g['player_name'].str.lower().str.strip() + '_' + pool_g['team'].apply(norm)
        pool_d = pool_d[pool_d['_k'].isin(actual_keys)]
        pool_g = pool_g[pool_g['_k'].isin(actual_keys)]

    if len(pool_d) < 2 or len(pool_g) < 1:
        return None

    # ── Identify opponent teams for goalie penalty (soft, not hard block) ──
    opponent_teams = set()
    for st in stack_teams:
        st_rows = dk_pool[dk_pool['team'].apply(norm) == st]
        if not st_rows.empty:
            st_gt = st_rows.iloc[0].get('game_total', 0)
            if st_gt > 0:
                for _, opp_row in dk_pool.drop_duplicates('team').iterrows():
                    opp_team = norm(str(opp_row['team']))
                    if opp_team not in stack_teams:
                        opp_gt = opp_row.get('game_total', 0)
                        if pd.notna(opp_gt) and abs(opp_gt - st_gt) < 0.1:
                            opponent_teams.add(opp_team)

    # Score and sort based on fill mode
    if fill_mode == 'salary':
        # Salary mode: sort by salary (rho=0.387 for D — strongest single predictor)
        # But still apply goalie correlation scoring (positive for stack team, penalty for opponents)
        pool_d = pool_d.sort_values('salary', ascending=False)
        pool_g['_ceil_score'] = pool_g.apply(
            lambda r: _score_g_ceiling(r, stack_team, stack_teams, opponent_teams), axis=1)
        pool_g = pool_g.sort_values('_ceil_score', ascending=False)
    else:
        # Ceiling/game_corr mode: PP1-weighted scoring with stack correlation
        game_teams = stack_game_teams
        pool_d['_ceil_score'] = pool_d.apply(
            lambda r: _score_d_ceiling(r, stack_team, game_teams, stack_teams), axis=1)
        pool_g['_ceil_score'] = pool_g.apply(
            lambda r: _score_g_ceiling(r, stack_team, stack_teams, opponent_teams), axis=1)
        pool_d = pool_d.sort_values('_ceil_score', ascending=False)
        pool_g = pool_g.sort_values('_ceil_score', ascending=False)

    best = None
    best_quality = 0

    for _, g_row in pool_g.head(8).iterrows():
        budget_after_g = remaining - g_row['salary']
        if budget_after_g < 5000:
            continue

        affordable_d = pool_d[pool_d['salary'] <= budget_after_g]
        if len(affordable_d) < 2:
            continue

        d1 = affordable_d.iloc[0]
        d2_pool = affordable_d[(affordable_d.index != d1.name) &
                                (affordable_d['salary'] <= budget_after_g - d1['salary'])]
        if d2_pool.empty:
            continue
        d2 = d2_pool.iloc[0]

        total_fill = g_row['salary'] + d1['salary'] + d2['salary']
        if total_fill > remaining:
            continue

        if fill_mode == 'salary':
            quality = d1['salary'] + d2['salary'] + g_row.get('_ceil_score', g_row['salary'])
        else:
            quality = d1['_ceil_score'] + d2['_ceil_score'] + g_row['_ceil_score']

        if quality > best_quality:
            best_quality = quality
            best = {'g': g_row, 'd1': d1, 'd2': d2}

    if best is None:
        return None

    players = []
    for f in stack['forwards']:
        players.append({'name': f['player_name'], 'team': f['team'],
                        'salary': f['salary'], 'position': f['position'], 'role': 'stack'})
    for dk in ['d1', 'd2']:
        d = best[dk]
        players.append({'name': d['player_name'], 'team': d['team'],
                        'salary': d['salary'], 'position': 'D', 'role': 'fill'})
    g = best['g']
    players.append({'name': g['player_name'], 'team': g['team'],
                    'salary': g['salary'], 'position': 'G', 'role': 'fill'})
    return players


# ==============================================================================
# MULTI-STACK SELECTION STRATEGIES (Heuristic + ML)
# ==============================================================================

HEURISTIC_STRATEGIES = ['chalk', 'contrarian_1', 'contrarian_2', 'value', 'ceiling', 'game_stack',
                        'pp1_stack', 'dual_chalk', 'dual_ceiling', 'dual_game']
ML_STRATEGIES = ['ml_chalk', 'ml_ceiling', 'ml_contrarian', 'ml_value',
                 'ml_dual_chalk', 'ml_dual_ceiling']
ALL_STRATEGIES = HEURISTIC_STRATEGIES + ML_STRATEGIES


def select_lineups(stacks, dk_pool, actuals, strategies=None, has_ml=False):
    """
    Generate multiple lineup candidates using different selection strategies.
    Returns dict of strategy -> (lineup, stack_info)
    """
    if strategies is None:
        strategies = ALL_STRATEGIES if has_ml else HEURISTIC_STRATEGIES

    results = {}

    for strategy in strategies:
        # Skip ML strategies if models weren't trained
        if strategy.startswith('ml_') and not has_ml:
            continue
        # Skip ML strategies if no stacks have ML projections
        if strategy.startswith('ml_') and all(s['ml_proj'] is None for s in stacks):
            continue

        # Separate single-team and dual-team stacks
        single_stacks = [s for s in stacks if s['stack_type'] == 'single']
        dual_stacks = [s for s in stacks if s['stack_type'] == 'dual']

        # ── Single-team strategies (original) ──
        if strategy == 'chalk':
            # Chalk with PP1 tiebreaker: projection first, then PP1 correlation
            candidates = sorted(single_stacks,
                                key=lambda s: (s['combo_proj'], s.get('pp1_score', 0)),
                                reverse=True)
        elif strategy == 'contrarian_1':
            candidates = [s for s in single_stacks if 3 <= s['impl_rank'] <= 5]
            candidates.sort(key=lambda s: s['combo_proj'], reverse=True)
        elif strategy == 'contrarian_2':
            candidates = [s for s in single_stacks if 5 <= s['impl_rank'] <= 8]
            candidates.sort(key=lambda s: s['combo_proj'], reverse=True)
        elif strategy == 'value':
            candidates = [s for s in single_stacks if s['combo_proj'] > 35]
            candidates.sort(key=lambda s: s['fwd_salary'])
        elif strategy == 'ceiling':
            def ceiling_score(s):
                max_sal = max(f['salary'] for f in s['forwards']) if s['forwards'] else 0
                pp1_bonus = s.get('pp1_score', 0) * 500  # PP1 overlap as ceiling tiebreaker
                return max_sal + s['combo_proj'] * 0.1 + pp1_bonus
            candidates = sorted(single_stacks, key=ceiling_score, reverse=True)
        elif strategy == 'game_stack':
            candidates = sorted(single_stacks, key=lambda s: (s['game_total'], s['combo_proj']), reverse=True)
        elif strategy == 'pp1_stack':
            # PP1 overlap: prioritize teams where L1 forwards ARE the PP1 unit
            # AND PP1 D is available = maximum 4-5 player correlated ceiling
            candidates = sorted(single_stacks,
                                key=lambda s: (s.get('pp1_score', 0), s['combo_proj']),
                                reverse=True)

        # ── Dual-team strategies (NEW — 2-team stacking) ──
        elif strategy == 'dual_chalk':
            candidates = sorted(dual_stacks, key=lambda s: s['combo_proj'], reverse=True)
        elif strategy == 'dual_ceiling':
            def dual_ceil_score(s):
                max_sal = max(f['salary'] for f in s['forwards']) if s['forwards'] else 0
                pp1_bonus = s.get('pp1_score', 0) * 500  # PP1 on primary team = asymmetric ceiling
                return max_sal + s['combo_proj'] * 0.1 + pp1_bonus
            candidates = sorted(dual_stacks, key=dual_ceil_score, reverse=True)
        elif strategy == 'dual_game':
            candidates = sorted(dual_stacks, key=lambda s: (s['game_total'], s['combo_proj']), reverse=True)

        # ── ML single-team strategies ──
        elif strategy == 'ml_chalk':
            candidates = sorted(
                [s for s in single_stacks if s['ml_proj'] is not None],
                key=lambda s: s['ml_proj'], reverse=True)
        elif strategy == 'ml_ceiling':
            candidates = sorted(
                [s for s in single_stacks if s['ml_ceiling'] is not None],
                key=lambda s: s['ml_ceiling'], reverse=True)
        elif strategy == 'ml_contrarian':
            candidates = [s for s in single_stacks if 3 <= s['impl_rank'] <= 6 and s['ml_proj'] is not None]
            candidates.sort(key=lambda s: s['ml_proj'], reverse=True)
        elif strategy == 'ml_value':
            candidates = [s for s in single_stacks if s['ml_proj'] is not None and s['ml_proj'] > 30]
            candidates.sort(key=lambda s: s['fwd_salary'])

        # ── ML dual-team strategies (NEW) ──
        elif strategy == 'ml_dual_chalk':
            candidates = sorted(
                [s for s in dual_stacks if s['ml_proj'] is not None],
                key=lambda s: s['ml_proj'], reverse=True)
        elif strategy == 'ml_dual_ceiling':
            candidates = sorted(
                [s for s in dual_stacks if s['ml_ceiling'] is not None],
                key=lambda s: s['ml_ceiling'], reverse=True)
        else:
            continue

        # Chalk/value use salary-weighted fill (salary rho=0.387 for D — best single predictor)
        # All others use ceiling-weighted PP1/correlation scoring
        if strategy in ['chalk', 'value', 'ml_chalk', 'ml_value']:
            fill_mode = 'salary'  # High-salary D = safe floor + strong predictor
        elif strategy in ['game_stack', 'dual_game']:
            fill_mode = 'game_corr'  # Extra weight on same-game correlation
        else:
            fill_mode = 'ceiling'  # PP1-weighted ceiling + stack correlation

        # Try to fill lineup for each candidate
        for stack in candidates[:5]:
            lineup = fill_lineup(stack, dk_pool, actuals, fill_mode=fill_mode)
            if lineup is not None:
                results[strategy] = {
                    'lineup': lineup,
                    'stack': stack,
                }
                break

    return results


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================

def run_multi_stack_backtest(start_date='2025-11-07', confirmed_only=True):
    conn = sqlite3.connect(str(DB_PATH))

    print("Loading game logs...")
    all_logs = pd.read_sql("""
        SELECT player_name, team, game_date, dk_fpts, position
        FROM game_logs_skaters ORDER BY game_date
    """, conn)
    all_logs['game_date'] = pd.to_datetime(all_logs['game_date'])
    all_logs['_key'] = all_logs['player_name'].str.lower().str.strip() + '_' + all_logs['team'].apply(norm)

    print("Building ML feature history...")
    line_history = build_line_history(conn)
    print(f"  Line records: {len(line_history)}, Features: {len(FEATURES)}")

    dates = pd.read_sql(f"""
        SELECT DISTINCT d.slate_date FROM dk_salaries d
        WHERE d.slate_date >= '{start_date}'
        AND EXISTS (SELECT 1 FROM game_logs_skaters g WHERE g.game_date = d.slate_date)
        ORDER BY d.slate_date
    """, conn)['slate_date'].tolist()

    print(f"\n{'='*90}")
    print(f"  UNIFIED MULTI-STACK BACKTEST | {len(dates)} dates")
    print(f"  Strategies: {', '.join(ALL_STRATEGIES)}")
    print(f"  ML min training: {MIN_ML_TRAIN} records | Confirmed only: {confirmed_only}")
    print(f"{'='*90}")

    all_results = []
    ml_active_count = 0

    for di, date_str in enumerate(dates, 1):
        dk_pool = pd.read_sql("SELECT * FROM dk_salaries WHERE slate_date = ?",
                               conn, params=(date_str,))
        actuals = load_actuals(date_str, conn)

        if dk_pool.empty or actuals.empty:
            continue

        # Contest info
        contest = conn.execute("""
            SELECT MIN(CASE WHEN n_cashed > 0 THEN score END),
                   MAX(CASE WHEN place = 1 THEN score END),
                   total_entries,
                   MAX(CASE WHEN place = 1 THEN profit END)
            FROM contest_results WHERE slate_date = ?
        """, (date_str,)).fetchone()
        cash_line = contest[0] if contest and contest[0] else 0
        first_score = contest[1] if contest and contest[1] else 0
        total_entries = contest[2] if contest and contest[2] else 0
        first_profit = contest[3] if contest and contest[3] else 0

        n_teams = dk_pool['team'].nunique()

        # Train ML models (walk-forward: only past data)
        ml_models = train_ml_models(line_history, date_str)
        has_ml = ml_models is not None
        if has_ml:
            ml_active_count += 1

        # Build all stacks (with ML if available)
        stacks = build_all_stacks(dk_pool, all_logs, date_str, ml_models)
        if not stacks:
            continue

        # Generate multi-strategy lineups
        lineups = select_lineups(
            stacks, dk_pool,
            actuals if confirmed_only else None,
            has_ml=has_ml
        )

        if not lineups:
            continue

        # Score each strategy
        best_strat = None
        best_actual = 0

        for strat, data in lineups.items():
            lineup = data['lineup']
            stack = data['stack']
            actual_total, n_matched, n_scratched = score_players(lineup, actuals)

            is_cash = actual_total >= cash_line if cash_line > 0 else None
            is_first = actual_total >= first_score if first_score > 0 else None

            # Get the projection used
            if strat.startswith('ml_'):
                projected = stack.get('ml_proj', 0) or 0
            else:
                projected = stack.get('combo_proj', 0) or 0

            # Count distinct teams in lineup
            lineup_teams = set(norm(str(p['team'])) for p in lineup)
            n_lineup_teams = len(lineup_teams)
            # Count teams among forwards only (stack teams)
            fwd_teams = set(norm(str(p['team'])) for p in lineup if p['role'] == 'stack')
            n_fwd_teams = len(fwd_teams)

            all_results.append({
                'date': date_str,
                'strategy': strat,
                'scoring': 'ML' if strat.startswith('ml_') else 'heuristic',
                'stack_type': stack.get('stack_type', 'single'),
                'stack_team': stack['team'],
                'impl_rank': stack['impl_rank'],
                'projected': projected,
                'combo_proj': stack['combo_proj'],
                'ml_proj': stack.get('ml_proj'),
                'ml_ceiling': stack.get('ml_ceiling'),
                'actual': actual_total,
                'matched': n_matched,
                'scratched': n_scratched,
                'salary': sum(p['salary'] for p in lineup),
                'cash_line': cash_line,
                'first_score': first_score,
                'first_profit': first_profit,
                'total_entries': total_entries,
                'n_teams': n_teams,
                'n_lineup_teams': n_lineup_teams,
                'n_fwd_teams': n_fwd_teams,
                'is_cash': is_cash,
                'is_first': is_first,
            })

            if actual_total > best_actual:
                best_actual = actual_total
                best_strat = strat

        # Print best result for this date
        if best_strat:
            best_row = [r for r in all_results if r['date'] == date_str and r['strategy'] == best_strat][-1]
            status = 'CASH' if best_row['is_cash'] else ('miss' if best_row['is_cash'] is not None else '  - ')
            first_flag = ' 1ST!' if best_row['is_first'] else ''
            ml_flag = ' [ML]' if best_strat.startswith('ml_') else ''
            print(f"  [{di:3d}] {date_str} ({n_teams:2d}t) | "
                  f"Best: {best_strat:16s} {best_row['stack_team']:4s} (impl#{best_row['impl_rank']}) | "
                  f"{best_actual:6.1f} FPTS | Cash:{cash_line:5.1f} 1st:{first_score:5.1f} | "
                  f"{status}{first_flag}{ml_flag}")

    conn.close()

    # ==============================================================================
    # ANALYSIS
    # ==============================================================================
    r = pd.DataFrame(all_results)

    if r.empty:
        print("No results!")
        return r

    print(f"\n{'='*90}")
    print(f"  UNIFIED RESULTS SUMMARY | ML active on {ml_active_count}/{len(dates)} dates")
    print(f"{'='*90}")

    # Per-strategy summary
    for strat in ALL_STRATEGIES:
        sr = r[r['strategy'] == strat]
        if sr.empty:
            continue

        with_cash = sr[sr['cash_line'] > 0]
        n_cash = int((with_cash['is_cash'] == True).sum()) if len(with_cash) > 0 else 0
        n_dates = len(with_cash)
        cash_rate = n_cash / max(1, n_dates) * 100

        n_first = int((with_cash['is_first'] == True).sum()) if len(with_cash) > 0 else 0
        first_rate = n_first / max(1, n_dates) * 100

        avg_impl_rank = sr['impl_rank'].mean()
        scoring_tag = ' [ML]' if strat.startswith('ml_') else ''

        print(f"\n  {strat.upper()}{scoring_tag} (n={len(sr)} dates, avg impl rank: {avg_impl_rank:.1f})")
        print(f"  Avg FPTS:     {sr['actual'].mean():.1f}")
        print(f"  Median FPTS:  {sr['actual'].median():.1f}")
        print(f"  90th pct:     {sr['actual'].quantile(0.9):.1f}")
        print(f"  Cash rate:    {cash_rate:.1f}% ({n_cash}/{n_dates})")
        print(f"  1st rate:     {first_rate:.1f}% ({n_first}/{n_dates})")

    # Best-of-all — heuristic only, ML only, and combined
    for label, subset in [('HEURISTIC ONLY', r[~r['strategy'].str.startswith('ml_')]),
                          ('ML ONLY', r[r['strategy'].str.startswith('ml_')]),
                          (f'ALL {len(ALL_STRATEGIES)} STRATEGIES', r)]:
        if subset.empty:
            continue
        best_per_date = subset.groupby('date').apply(
            lambda g: g.nlargest(1, 'actual').iloc[0], include_groups=False)
        with_cash_best = best_per_date[best_per_date['cash_line'] > 0]
        n_cash_best = int((with_cash_best['is_cash'] == True).sum())
        n_first_best = int((with_cash_best['is_first'] == True).sum())
        n_dates_best = len(with_cash_best)
        print(f"\n  BEST-OF {label}:")
        print(f"  Dates: {len(best_per_date)}")
        print(f"  Avg FPTS:     {best_per_date['actual'].mean():.1f}")
        print(f"  Cash rate:    {n_cash_best/max(1,n_dates_best)*100:.1f}% ({n_cash_best}/{n_dates_best})")
        print(f"  1st rate:     {n_first_best/max(1,n_dates_best)*100:.1f}% ({n_first_best}/{n_dates_best})")

    # Which strategy wins most often?
    print(f"\n  WHICH STRATEGY WINS MOST OFTEN?")
    best_strats = r.loc[r.groupby('date')['actual'].idxmax()]['strategy'].value_counts()
    for strat, count in best_strats.items():
        pct = count / r['date'].nunique() * 100
        ml_tag = ' [ML]' if strat.startswith('ml_') else ''
        print(f"    {strat:16s}{ml_tag}: {count:3d} times ({pct:.0f}%)")

    # ML vs Heuristic head-to-head
    r_ml = r[r['strategy'].str.startswith('ml_')]
    r_heur = r[~r['strategy'].str.startswith('ml_')]
    if not r_ml.empty and not r_heur.empty:
        print(f"\n  ML vs HEURISTIC HEAD-TO-HEAD:")
        ml_best = r_ml.groupby('date')['actual'].max()
        heur_best = r_heur.groupby('date')['actual'].max()
        common = ml_best.index.intersection(heur_best.index)
        if len(common) > 0:
            ml_wins = int((ml_best[common] > heur_best[common]).sum())
            heur_wins = int((heur_best[common] > ml_best[common]).sum())
            ties = int((ml_best[common] == heur_best[common]).sum())
            print(f"    Common dates: {len(common)}")
            print(f"    ML wins:       {ml_wins} ({ml_wins/len(common)*100:.0f}%)")
            print(f"    Heuristic wins: {heur_wins} ({heur_wins/len(common)*100:.0f}%)")
            print(f"    Ties:          {ties}")
            print(f"    ML avg:        {ml_best[common].mean():.1f}")
            print(f"    Heuristic avg: {heur_best[common].mean():.1f}")
            print(f"    ML advantage:  {(ml_best[common] - heur_best[common]).mean():+.1f} FPTS/night")

    # Single vs Dual stack comparison
    print(f"\n  SINGLE vs DUAL TEAM STACKS:")
    for stype, label in [('single', 'Single-team (L1+L2)'), ('dual', 'Dual-team (L1+L1)')]:
        sr = r[r['stack_type'] == stype]
        if sr.empty:
            continue
        with_cash = sr[sr['cash_line'] > 0]
        n_cash = int((with_cash['is_cash'] == True).sum())
        n_first = int((with_cash['is_first'] == True).sum())
        n_dates = len(with_cash)
        print(f"    {label}: {len(sr)} lineups, avg {sr['actual'].mean():.1f} FPTS")
        print(f"      Cash: {n_cash}/{n_dates} ({n_cash/max(1,n_dates)*100:.1f}%), "
              f"1st: {n_first}/{n_dates} ({n_first/max(1,n_dates)*100:.1f}%)")

    # Best-of single vs best-of dual per date
    for stype, label in [('single', 'SINGLE'), ('dual', 'DUAL')]:
        sr = r[r['stack_type'] == stype]
        if sr.empty:
            continue
        best = sr.groupby('date').apply(lambda g: g.nlargest(1, 'actual').iloc[0], include_groups=False)
        with_cash = best[best['cash_line'] > 0]
        n_cash = int((with_cash['is_cash'] == True).sum())
        n_first = int((with_cash['is_first'] == True).sum())
        n_dates = len(with_cash)
        print(f"    Best-of {label}/date: avg {best['actual'].mean():.1f} FPTS, "
              f"Cash {n_cash}/{n_dates} ({n_cash/max(1,n_dates)*100:.1f}%), "
              f"1st {n_first}/{n_dates} ({n_first/max(1,n_dates)*100:.1f}%)")

    # Contrarian value
    print(f"\n  CONTRARIAN VALUE:")
    chalk_cash_dates = set(r[(r['strategy'] == 'chalk') & (r['is_cash'] == True)]['date'].values)
    for strat in ['contrarian_1', 'contrarian_2', 'ml_contrarian']:
        sr = r[r['strategy'] == strat]
        with_cash = sr[sr['cash_line'] > 0]
        if with_cash.empty:
            continue
        this_cash_dates = set(with_cash[with_cash['is_cash'] == True]['date'].values)
        unique_to_this = this_cash_dates - chalk_cash_dates
        print(f"    {strat}: {len(unique_to_this)} dates where ONLY this cashed (chalk missed)")

    # Save full results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = BACKTESTS_DIR / f"unified_backtest_{ts}.csv"
    r.to_csv(str(output), index=False)
    print(f"\n  Saved: {output}")

    # Save per-date best
    best_all = r.groupby('date').apply(lambda g: g.nlargest(1, 'actual').iloc[0], include_groups=False)
    best_output = BACKTESTS_DIR / f"unified_best_per_date_{ts}.csv"
    best_all.to_csv(str(best_output))
    print(f"  Saved: {best_output}")

    return r


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2025-11-07')
    parser.add_argument('--all-players', action='store_true')
    args = parser.parse_args()

    run_multi_stack_backtest(
        start_date=args.start,
        confirmed_only=not args.all_players,
    )
