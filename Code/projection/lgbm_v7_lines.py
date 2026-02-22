#!/usr/bin/env python3
"""
lgbm_v7_lines.py — LightGBM v7: Chaos + Line Combination Features
====================================================================
Google Colab Edition

WHAT'S NEW vs v6 (chaos):
  Adds line combination context features derived from dk_salaries line assignments:

  LINE CONTEXT (from dk_salaries.start_line + pp_unit):
  - start_line: Current line assignment (1-4, encoded)
  - on_pp1: Binary PP1 assignment indicator
  - on_pp: Binary any-PP assignment indicator
  - line_changed: Did player's line change from last game?
  - line_stability: Consecutive games on same line (capped 10)

  LINEMATE QUALITY (rolling averages of same-line teammates):
  - linemate_fpts_avg: Avg recent FPTS of linemates on same line
  - linemate_fpts_std: Std of recent FPTS of linemates (volatility)
  - linemate_goals_avg: Avg recent goals of linemates
  - linemate_toi_avg: Avg recent TOI of linemates

  LINE-LEVEL PERFORMANCE:
  - line_total_fpts_avg: Rolling avg total FPTS of the full line
  - line_fpts_share: Player's share of their line's total FPTS
  - pp1_fpts_boost: Historical FPTS uplift when on PP1 vs not

  CHAOS × LINE INTERACTIONS:
  - chaos_x_line_stability: Stable lines dampen chaos, unstable amplify
  - hurst_x_pp1: Trending player on PP1 = compounding signal
  - lyap_x_linemate_quality: Chaotic player + elite linemates = regime shift
  - line_regime_chaos: Chaos features computed separately for PP1/non-PP1 games

  Everything from v6 preserved: SDE engine, Heston vol, chaos features,
  rolling stats, opponent FPTS, Optuna tuning.

Run: python lgbm_v7_lines.py --backtest --tune
"""

import argparse, sqlite3, time, warnings
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
warnings.filterwarnings('ignore')

import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

DB_PATH = Path('data/nhl_dfs_history.db')
BACKTEST_START = datetime(2025, 11, 7)
BACKTEST_END = datetime(2026, 2, 5)
RETRAIN_INTERVAL = 14
OU_MIN_GAMES = 15
OU_VOL_WINDOW = 5


# ============================================================================
# CHAOS THEORY ENGINE (from v6)
# ============================================================================
class ChaosAnalyzer:
    """Compute chaos-theoretic features for player FPTS time series."""

    @staticmethod
    def lyapunov_exponent(x, lag=1):
        x = np.array(x, dtype=float); n = len(x)
        if n < 15: return 0.0
        m = 3; tau = lag
        if n < m*tau+5: return 0.0
        N = n - (m-1)*tau
        embedded = np.array([x[i:i+(m*tau):tau] for i in range(N)])
        divergences = []
        for i in range(N-10):
            dists = np.sum((embedded-embedded[i])**2, axis=1)
            dists[max(0,i-2):i+3] = np.inf
            j = np.argmin(dists)
            if j+10<N and i+10<N:
                d0 = max(np.sqrt(dists[j]), 1e-10)
                for k in range(1, min(8, N-max(i,j))):
                    dk = np.sqrt(np.sum((embedded[i+k]-embedded[j+k])**2))
                    if dk > 0: divergences.append(np.log(dk/d0)/k)
        return float(np.median(divergences)) if len(divergences) >= 5 else 0.0

    @staticmethod
    def hurst_exponent(x):
        x = np.array(x, dtype=float); n = len(x)
        if n < 20: return 0.5
        max_k = min(n//4, 20); rs_vals = []; ns_vals = []
        for k in range(4, max_k+1):
            n_sub = n//k
            if n_sub < 4: break
            rs_list = []
            for i in range(k):
                sub = x[i*n_sub:(i+1)*n_sub]
                cumdev = np.cumsum(sub - sub.mean())
                R = cumdev.max() - cumdev.min()
                S = sub.std(ddof=1) if sub.std(ddof=1) > 0 else 1e-10
                rs_list.append(R/S)
            rs_vals.append(np.mean(rs_list)); ns_vals.append(n_sub)
        if len(rs_vals) < 3: return 0.5
        try:
            return float(np.clip(np.polyfit(np.log(ns_vals), np.log(rs_vals), 1)[0], 0, 1))
        except: return 0.5

    @staticmethod
    def recurrence_rate(x, threshold_pct=20):
        x = np.array(x, dtype=float); n = len(x)
        if n < 10: return 0.0
        embedded = np.column_stack([x[:-1], x[1:]])
        threshold = np.std(x) * threshold_pct / 100
        n_recur = 0; total = 0
        for i in range(min(100, len(embedded))):
            dists = np.sqrt(np.sum((embedded-embedded[i])**2, axis=1))
            dists[max(0,i-1):i+2] = np.inf
            n_recur += np.sum(dists < threshold)
            total += len(dists)-3
        return n_recur / max(total, 1)

    @staticmethod
    def autocorrelation_decay(x):
        x = np.array(x, dtype=float)
        if len(x) < 10: return 5.0
        x = x - x.mean(); var = np.var(x)
        if var == 0: return 5.0
        for lag in range(1, min(10, len(x)//2)):
            acf = np.mean(x[:-lag]*x[lag:])/var
            if acf < 0.5: return float(lag)
        return 10.0

    @staticmethod
    def embedding_dimension(x, max_dim=8):
        x = np.array(x, dtype=float); n = len(x)
        if n < 20: return 2
        threshold = 15.0; best_dim = 2
        for m in range(2, min(max_dim+1, n//4)):
            N = n - m
            if N < 10: break
            embedded = np.array([x[i:i+m] for i in range(N)])
            embedded_ext = np.array([x[i:i+m+1] for i in range(N-1)])
            fnn_count = 0; total = 0
            for i in range(min(50, N-1)):
                dists = np.sum((embedded[:N-1]-embedded[i])**2, axis=1)
                dists[max(0,i-1):i+2] = np.inf
                j = np.argmin(dists)
                d_m = max(np.sqrt(dists[j]), 1e-10)
                d_m1 = np.sqrt(np.sum((embedded_ext[i]-embedded_ext[j])**2))
                if abs(d_m1-d_m)/d_m > threshold: fnn_count += 1
                total += 1
            fnn_rate = fnn_count/max(total,1)
            if fnn_rate < 0.1: best_dim = m; break
            best_dim = m
        return best_dim

    def compute_all(self, fpts_series):
        x = np.array(fpts_series, dtype=float)
        if len(x) < 15:
            return {'lyapunov':0.0, 'hurst':0.5, 'recurrence':0.0,
                    'acf_decay':5.0, 'embed_dim':2, 'chaos_score':0.0}
        lyap = self.lyapunov_exponent(x)
        hurst = self.hurst_exponent(x)
        recur = self.recurrence_rate(x)
        acf_d = self.autocorrelation_decay(x)
        embed = self.embedding_dimension(x)
        chaos_score = (
            (1.0 if lyap > 0 else 0.0) * 0.3 +
            np.clip(embed/6.0, 0, 1) * 0.15 +
            (1.0 - recur) * 0.15 +
            (1.0 - acf_d/10.0) * 0.15 +
            abs(hurst - 0.5) * 2.0 * 0.25
        )
        return {'lyapunov':lyap, 'hurst':hurst, 'recurrence':recur,
                'acf_decay':acf_d, 'embed_dim':embed, 'chaos_score':chaos_score}


# ============================================================================
# SDE ENGINE (from v6)
# ============================================================================
class SDEEngine:
    def __init__(self):
        self.ou_params = {}
        self.heston_params = {}

    def fit_ou(self, x):
        x = np.array(x, dtype=float)
        if len(x) < 10: return None
        xp, xn = x[:-1], x[1:]
        xb, yb = xp.mean(), xn.mean()
        ss_xx = np.sum((xp-xb)**2)
        if ss_xx == 0: return None
        b = np.sum((xp-xb)*(xn-yb))/ss_xx; a = yb-b*xb; theta = 1.0-b
        if theta <= 0 or theta > 2: return None
        mu = a/theta
        if mu < 0: return None
        sigma = np.std(xn-(a+b*xp), ddof=2)
        if sigma <= 0: return None
        return {'theta':theta,'mu':mu,'sigma':sigma,
                'half_life':min(np.log(2)/theta,20.0),'n_games':len(x)}

    def fit_heston(self, x, w=5):
        x = np.array(x, dtype=float)
        if len(x) < w+15: return None
        ou = self.fit_ou(x)
        if ou is None: return None
        r = x[1:]-(x[:-1]+ou['theta']*(ou['mu']-x[:-1]))
        sq = r**2
        vt = np.array([np.mean(sq[i-w:i]) if i>=w else np.nan for i in range(len(sq))])
        vc = vt[~np.isnan(vt)]
        if len(vc)<10: return None
        vol_ou = self.fit_ou(vc)
        if vol_ou is None: return None
        return {'kappa':vol_ou['theta'],'theta_v':vol_ou['mu'],'xi':vol_ou['sigma']}

    def fit_all_players(self, df):
        print("  Fitting SDE parameters...")
        n_ou = n_h = 0
        for pid, g in df.groupby('player_id'):
            fpts = g.sort_values('game_date')['dk_fpts'].values
            if len(fpts) < OU_MIN_GAMES: continue
            ou = self.fit_ou(fpts)
            if ou:
                ou['player_name'] = g['player_name'].iloc[0]
                self.ou_params[pid] = ou; n_ou += 1
                h = self.fit_heston(fpts)
                if h: self.heston_params[pid] = h; n_h += 1
        print(f"  O-U: {n_ou} | Heston: {n_h}")

    def compute_sde_features(self, pid, hist):
        if pid not in self.ou_params or len(hist) < 3: return None
        ou = self.ou_params[pid]
        theta, mu_h, sigma = ou['theta'], ou['mu'], ou['sigma']
        n = len(hist)
        w = max(0.3, OU_MIN_GAMES/(OU_MIN_GAMES+n))
        mu_b = w*mu_h + (1-w)*np.mean(hist)
        x = hist[-1]; dist = mu_b-x; eb = theta*dist
        z = dist/sigma if sigma > 0 else 0
        gb = ga = 0
        for v in reversed(hist):
            if v < mu_b: gb += 1
            else: break
        for v in reversed(hist):
            if v > mu_b: ga += 1
            else: break
        rv = np.var(hist[-OU_VOL_WINDOW:]) if n >= OU_VOL_WINDOW else np.var(hist)
        lrv = sigma**2
        if pid in self.heston_params:
            tv = self.heston_params[pid]['theta_v']
            vr = 0 if rv < tv*0.67 else (2 if rv > tv*1.33 else 1)
        else:
            vr = 1 if rv <= lrv else 2
        if vr == 0: dc = 1 if z > 0.5 else 0
        elif vr == 2: dc = 3 if z > 0.5 else (4 if z < -0.5 else 2)
        else: dc = 2 if z > 0.5 else 5
        return {'sde_theta':theta,'sde_mu':mu_b,
                'sde_distance':np.clip(dist,-20,20),'sde_expected_bounce':np.clip(eb,-10,10),
                'sde_z_score':np.clip(z,-3,3),'sde_games_below':gb,'sde_games_above':ga,
                'sde_rolling_vol':rv,'sde_long_run_vol':lrv,
                'sde_vol_ratio':np.clip(rv/(lrv+1e-6),0,5),
                'sde_vol_regime':vr,'sde_half_life':ou['half_life'],'sde_dfs_class':dc}


# ============================================================================
# DATA LOADING
# ============================================================================
def load_boxscore_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT player_name, player_id, team, position, game_date, opponent,
               home_road, goals, assists, shots, hits, blocked_shots, plus_minus,
               pp_goals, toi_seconds, dk_fpts, game_id
        FROM boxscore_skaters ORDER BY game_date, player_id
    """, conn); conn.close()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['player_id','game_date']).reset_index(drop=True)

    stats = ['goals','assists','shots','blocked_shots','hits','pp_goals','dk_fpts','toi_seconds']
    for col in stats:
        s = df.groupby('player_id')[col].shift(1)
        for w in [3,5,10,20]:
            df[f'roll_{col}_{w}g'] = s.groupby(df['player_id']).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

    sf = df.groupby('player_id')['dk_fpts'].shift(1)
    for w in [5,10]:
        df[f'fpts_std_{w}g'] = sf.groupby(df['player_id']).rolling(w, min_periods=2).std().reset_index(level=0, drop=True).fillna(0)

    for col in stats:
        sc = df.groupby('player_id')[col].shift(1)
        cs = sc.groupby(df['player_id']).cumsum()
        gp = df.groupby('player_id').cumcount().clip(lower=1)
        df[f'season_avg_{col}'] = (cs/gp).fillna(0)

    df['season_gp'] = df.groupby('player_id').cumcount() + 1
    df['log_gp'] = np.log1p(df['season_gp'])
    df['is_home'] = (df['home_road'].str.upper()=='H').astype(int)
    for p in ['C','D','L','R']:
        df[f'pos_{p}'] = (df['position']==p).astype(int)

    st = df.groupby('player_id')['toi_seconds'].shift(1)
    rt5 = st.groupby(df['player_id']).rolling(5,min_periods=1).mean().reset_index(level=0,drop=True)
    ct = st.groupby(df['player_id']).cumsum()
    pgp = df.groupby('player_id').cumcount().clip(lower=1)
    sat = (ct/pgp).fillna(1)
    df['toi_trend'] = (rt5/(sat+1e-6)).fillna(1.0)

    df['last_game_fpts'] = df.groupby('player_id')['dk_fpts'].shift(1).fillna(0)
    df['last_game_toi'] = df.groupby('player_id')['toi_seconds'].shift(1).fillna(0)
    df['prev_game_date'] = df.groupby('player_id')['game_date'].shift(1)
    df['days_rest'] = (df['game_date']-df['prev_game_date']).dt.days.fillna(3).clip(0,7)

    if 'roll_dk_fpts_3g' in df.columns and 'roll_dk_fpts_10g' in df.columns:
        df['momentum_short'] = df['roll_dk_fpts_3g'] - df['roll_dk_fpts_10g']
    if 'roll_dk_fpts_5g' in df.columns and 'roll_dk_fpts_20g' in df.columns:
        df['momentum_long'] = df['roll_dk_fpts_5g'] - df['roll_dk_fpts_20g']
    df['consistency_5g'] = 1.0/(df['fpts_std_5g']+1.0)
    df['consistency_10g'] = 1.0/(df['fpts_std_10g']+1.0)
    return df


def load_historical_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT season,game_date,player_name,player_id,team,position,goals,assists,dk_fpts,shots,blocked_shots,hits,pp_goals,toi_seconds,opponent,home_road FROM historical_skaters WHERE dk_fpts IS NOT NULL ORDER BY player_id,game_date", conn)
        conn.close()
        df['game_date'] = pd.to_datetime(df['game_date'])
        print(f"  Historical: {len(df):,} rows")
        return df
    except:
        conn.close(); return None


def load_line_assignments():
    """Load line assignment data from dk_salaries table.

    Returns DataFrame with: player_id, player_name, slate_date, start_line, pp_unit, on_pp1
    Only includes rows where start_line is a valid number (1-4).
    Joins to roster_cache to get player_id for reliable matching to boxscore data.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT s.player_name, s.team, s.position, s.slate_date, s.start_line, s.pp_unit,
               r.player_id
        FROM dk_salaries s
        LEFT JOIN roster_cache r ON LOWER(TRIM(s.player_name)) = LOWER(TRIM(r.player_name))
        WHERE s.start_line IN ('1','2','3','4') AND s.position != 'G'
        ORDER BY s.slate_date, s.team, s.start_line
    """, conn)
    conn.close()

    df['slate_date'] = pd.to_datetime(df['slate_date'])
    df['start_line_num'] = df['start_line'].astype(int)
    df['on_pp1'] = (df['pp_unit'] == 1.0).astype(int)
    df['on_pp'] = df['pp_unit'].notna().astype(int)

    n_with_id = df['player_id'].notna().sum()
    print(f"  Line assignments: {len(df):,} rows across {df['slate_date'].nunique()} dates")
    print(f"  Matched to player_id: {n_with_id:,} / {len(df):,} ({100*n_with_id/len(df):.1f}%)")
    return df


# ============================================================================
# LINE COMBINATION FEATURES
# ============================================================================
def add_line_features(df, line_df):
    """Add line combination features to boxscore data.

    Joins dk_salaries line assignments to game_logs via player_name + date,
    then computes linemate-quality and line-stability features.
    All features use ONLY pre-game data (shifted/lagged) to prevent leakage.
    """
    print("  Computing line combination features...")
    t0 = time.time()

    # Initialize all line feature columns
    line_cols = [
        'start_line_num', 'on_pp1', 'on_pp',
        'line_changed', 'line_stability',
        'linemate_fpts_avg', 'linemate_fpts_std',
        'linemate_goals_avg', 'linemate_toi_avg',
        'line_total_fpts_avg', 'line_fpts_share',
        'pp1_fpts_boost',
    ]
    for c in line_cols:
        df[c] = 0.0

    # Join line assignments to boxscore data on player_id + date
    # dk_salaries joined to roster_cache gives player_id for reliable matching
    line_df_valid = line_df.dropna(subset=['player_id']).copy()
    line_df_valid['player_id'] = line_df_valid['player_id'].astype(int)
    line_lookup = line_df_valid.set_index(['player_id', 'slate_date'])[
        ['start_line_num', 'on_pp1', 'on_pp', 'team']
    ].to_dict('index')

    # Build per-game line rosters: (date, team, line_num) -> list of player_ids
    line_rosters = {}
    for _, row in line_df_valid.iterrows():
        key = (row['slate_date'], row['team'], row['start_line_num'])
        if key not in line_rosters:
            line_rosters[key] = []
        line_rosters[key].append(int(row['player_id']))

    # Track per-player line history for stability calculation
    player_line_history = {}  # player_id -> list of (date, line_num)

    # Sort by date for temporal processing
    df = df.sort_values(['game_date', 'player_id']).reset_index(drop=True)

    # Pre-build player game history indexed by player_id
    # for fast lookups during linemate quality computation
    player_game_fpts = {}  # player_id -> sorted list of (date, fpts, goals, toi)
    for _, row in df.iterrows():
        pid = row['player_id']
        if pid not in player_game_fpts:
            player_game_fpts[pid] = []
        player_game_fpts[pid].append((
            row['game_date'], row['dk_fpts'],
            row.get('goals', 0), row.get('toi_seconds', 0)
        ))

    # Now iterate through each row and compute features
    # Using pre-game data only (look at history BEFORE this game date)
    n_matched = 0
    n_with_linemates = 0

    for idx, row in df.iterrows():
        pid = row['player_id']
        gd = row['game_date']
        team = row['team']

        # Look up this player's line assignment for this date
        line_key = (pid, gd)
        if line_key not in line_lookup:
            continue

        line_info = line_lookup[line_key]
        line_num = line_info['start_line_num']
        pp1 = line_info['on_pp1']
        pp = line_info['on_pp']

        df.at[idx, 'start_line_num'] = line_num
        df.at[idx, 'on_pp1'] = pp1
        df.at[idx, 'on_pp'] = pp
        n_matched += 1

        # --- Line stability: consecutive games on same line ---
        if pid not in player_line_history:
            player_line_history[pid] = []

        # Count consecutive same-line assignments looking backward
        prev_history = player_line_history[pid]  # Already sorted by date
        stability = 0
        line_changed = 0
        if prev_history:
            for prev_date, prev_line in reversed(prev_history):
                if prev_line == line_num:
                    stability += 1
                else:
                    break
            line_changed = 1 if prev_history[-1][1] != line_num else 0

        df.at[idx, 'line_stability'] = min(stability, 10)
        df.at[idx, 'line_changed'] = line_changed

        # Update history AFTER computing features (no leakage)
        player_line_history[pid].append((gd, line_num))

        # --- Linemate quality (pre-game rolling averages) ---
        roster_key = (gd, team, line_num)
        linemates = line_rosters.get(roster_key, [])
        linemates = [lm for lm in linemates if lm != pid]  # Exclude self

        if linemates:
            lm_fpts = []
            lm_goals = []
            lm_toi = []

            for lm in linemates:
                if lm in player_game_fpts:
                    # Get last 5 games BEFORE this date
                    hist = [(d, f, g, t) for d, f, g, t in player_game_fpts[lm] if d < gd]
                    if hist:
                        recent = hist[-5:]  # Last 5 games
                        lm_fpts.append(np.mean([f for _, f, _, _ in recent]))
                        lm_goals.append(np.mean([g for _, _, g, _ in recent]))
                        lm_toi.append(np.mean([t for _, _, _, t in recent]))

            if lm_fpts:
                df.at[idx, 'linemate_fpts_avg'] = np.mean(lm_fpts)
                df.at[idx, 'linemate_fpts_std'] = np.std(lm_fpts) if len(lm_fpts) > 1 else 0
                df.at[idx, 'linemate_goals_avg'] = np.mean(lm_goals) if lm_goals else 0
                df.at[idx, 'linemate_toi_avg'] = np.mean(lm_toi) if lm_toi else 0
                n_with_linemates += 1

        # --- Line total FPTS and player's share ---
        all_on_line = line_rosters.get(roster_key, [])
        if all_on_line:
            line_fpts_history = []
            player_fpts_history = []

            for lm in all_on_line:
                if lm in player_game_fpts:
                    hist = [(d, f) for d, f, _, _ in player_game_fpts[lm] if d < gd]
                    if hist:
                        recent = hist[-5:]
                        line_fpts_history.append(np.mean([f for _, f in recent]))
                        if lm == pid:
                            player_fpts_history.append(np.mean([f for _, f in recent]))

            if line_fpts_history:
                line_total = sum(line_fpts_history)
                df.at[idx, 'line_total_fpts_avg'] = line_total
                if player_fpts_history and line_total > 0:
                    df.at[idx, 'line_fpts_share'] = player_fpts_history[0] / line_total

        # --- PP1 FPTS boost (historical uplift from PP1 assignment) ---
        if pid in player_game_fpts:
            hist_before = [(d, f) for d, f, _, _ in player_game_fpts[pid] if d < gd]
            if len(hist_before) >= 5:
                # Look up which past games had PP1
                pp1_fpts = []
                non_pp1_fpts = []
                for hd, hf in hist_before[-20:]:  # Last 20 games
                    hkey = (pid, hd)
                    if hkey in line_lookup and line_lookup[hkey]['on_pp1']:
                        pp1_fpts.append(hf)
                    else:
                        non_pp1_fpts.append(hf)

                if pp1_fpts and non_pp1_fpts:
                    boost = np.mean(pp1_fpts) - np.mean(non_pp1_fpts)
                    df.at[idx, 'pp1_fpts_boost'] = np.clip(boost, -10, 20)

    print(f"  Line features: {n_matched:,} matched, {n_with_linemates:,} with linemate data ({time.time()-t0:.1f}s)")
    return df


# ============================================================================
# OPPONENT / SDE / CHAOS FEATURES (from v6)
# ============================================================================
def compute_opponent_fpts_allowed(df):
    daily = df.groupby(['game_date','team'])['dk_fpts'].sum().reset_index()
    daily.columns = ['game_date','team','total_fpts']
    to = df[['game_date','team','opponent']].drop_duplicates()
    to = to.merge(daily.rename(columns={'team':'opponent','total_fpts':'opp_fpts'}), on=['game_date','opponent'], how='left')
    to['opp_fpts'] = to['opp_fpts'].fillna(0)
    to = to.sort_values(['team','game_date'])
    to['opp_fpts_allowed_10g'] = to.groupby('team')['opp_fpts'].rolling(10,min_periods=1).mean().reset_index(level=0,drop=True)
    for pos in ['C','D','L','R']:
        pd2 = df[df['position']==pos].groupby(['game_date','team'])['dk_fpts'].sum().reset_index()
        pd2.columns = ['game_date','team',f'opp_{pos}']
        po = df[['game_date','team','opponent']].drop_duplicates()
        po = po.merge(pd2.rename(columns={'team':'opponent'}), on=['game_date','opponent'], how='left')
        po[f'opp_{pos}'] = po[f'opp_{pos}'].fillna(0)
        po = po.sort_values(['team','game_date'])
        po[f'opp_{pos}_10g'] = po.groupby('team')[f'opp_{pos}'].rolling(10,min_periods=1).mean().reset_index(level=0,drop=True)
        to = to.merge(po[['game_date','team',f'opp_{pos}_10g']], on=['game_date','team'], how='left')
    df = df.merge(to[['game_date','team','opp_fpts_allowed_10g']+[f'opp_{p}_10g' for p in 'CDLR']], on=['game_date','team'], how='left')
    df['opp_fpts_allowed_10g'] = df['opp_fpts_allowed_10g'].fillna(0)
    df['opp_fpts_pos_10g'] = 0.0
    for pos in 'CDLR':
        m = df['position']==pos; c = f'opp_{pos}_10g'
        if c in df.columns: df.loc[m,'opp_fpts_pos_10g'] = df.loc[m,c].fillna(0)
    return df


def add_sde_features(df, sde):
    print("  Computing SDE features...")
    t0 = time.time()
    sc = ['sde_theta','sde_mu','sde_distance','sde_expected_bounce','sde_z_score',
          'sde_games_below','sde_games_above','sde_rolling_vol','sde_long_run_vol',
          'sde_vol_ratio','sde_vol_regime','sde_half_life','sde_dfs_class']
    for c in sc: df[c] = 0.0
    for pid, g in df.groupby('player_id'):
        gs = g.sort_values('game_date')
        idx = gs.index.tolist(); fv = gs['dk_fpts'].values
        for i in range(3, len(idx)):
            f = sde.compute_sde_features(pid, fv[:i])
            if f:
                for c in sc:
                    if c in f: df.loc[idx[i], c] = f[c]
    print(f"  SDE features in {time.time()-t0:.1f}s")
    return df


def add_chaos_features(df, chaos):
    print("  Computing chaos features...")
    t0 = time.time()
    cc = ['lyapunov','hurst','recurrence','acf_decay','embed_dim','chaos_score']
    for c in cc: df[c] = 0.0
    for pid, g in df.groupby('player_id'):
        gs = g.sort_values('game_date')
        idx = gs.index.tolist(); fv = gs['dk_fpts'].values
        for i in range(15, len(idx)):
            feat = chaos.compute_all(fv[:i])
            for c in cc:
                df.loc[idx[i], c] = feat.get(c, 0)
    print(f"  Chaos features in {time.time()-t0:.1f}s")
    return df


def add_interactions(df):
    """v6 interactions + new line×chaos interactions."""
    # Original v6 interactions
    df['sde_bounce_x_recent'] = df.get('sde_expected_bounce',0) * df.get('roll_dk_fpts_5g',0)
    df['sde_z_x_opp'] = df.get('sde_z_score',0) * df.get('opp_fpts_allowed_10g',0)
    df['sde_vol_x_z'] = df.get('sde_vol_ratio',0) * df.get('sde_z_score',0)
    df['sde_z_x_toi'] = df.get('sde_z_score',0) * df.get('toi_trend',0)
    df['sde_mu_x_opp'] = df.get('sde_mu',0) * df.get('opp_fpts_pos_10g',0)
    df['momentum_x_vol'] = df.get('momentum_short',0) * df.get('sde_vol_ratio',0)

    # v6 chaos x SDE interactions
    df['chaos_x_sde_z'] = df.get('chaos_score',0) * df.get('sde_z_score',0)
    df['hurst_x_momentum'] = df.get('hurst',0) * df.get('momentum_short',0)
    df['lyap_x_vol'] = df.get('lyapunov',0) * df.get('sde_rolling_vol',0)
    df['chaos_x_distance'] = df.get('chaos_score',0) * df.get('sde_distance',0)
    df['hurst_x_consistency'] = df.get('hurst',0) * df.get('consistency_10g',0)

    # NEW: Line × Chaos interactions
    # Stable line + low chaos = predictable (dampen variance estimate)
    df['chaos_x_line_stability'] = df.get('chaos_score',0) * (1.0 / (df.get('line_stability',0) + 1.0))
    # Trending player on PP1 = compounding upside
    df['hurst_x_pp1'] = df.get('hurst',0) * df.get('on_pp1',0)
    # Chaotic player with elite linemates = different regime
    df['lyap_x_linemate_quality'] = df.get('lyapunov',0) * df.get('linemate_fpts_avg',0)
    # SDE mu adjusted by linemate context
    df['sde_mu_x_linemate'] = df.get('sde_mu',0) * (df.get('linemate_fpts_avg',0) / 8.0)  # Normalize around ~8 avg fpts
    # Line change × momentum (promoted player with momentum = extra signal)
    df['line_changed_x_momentum'] = df.get('line_changed',0) * df.get('momentum_short',0)
    # PP1 boost × opponent weakness
    df['pp1_x_opp'] = df.get('on_pp1',0) * df.get('opp_fpts_allowed_10g',0)

    return df


def get_feature_columns():
    cols = []
    # v2 rolling stats (4 windows x 8 stats)
    for w in [3,5,10,20]:
        for s in ['goals','assists','shots','blocked_shots','hits','pp_goals','dk_fpts','toi_seconds']:
            cols.append(f'roll_{s}_{w}g')
    # v2 std
    cols += ['fpts_std_5g','fpts_std_10g']
    # v2 season averages
    for s in ['goals','assists','shots','blocked_shots','hits','pp_goals','dk_fpts','toi_seconds']:
        cols.append(f'season_avg_{s}')
    # v2 context
    cols += ['season_gp','log_gp','toi_trend','is_home','pos_C','pos_D','pos_L','pos_R',
             'momentum_short','momentum_long','consistency_5g','consistency_10g',
             'last_game_fpts','last_game_toi','days_rest','opp_fpts_allowed_10g','opp_fpts_pos_10g']
    # v2 SDE
    cols += ['sde_theta','sde_mu','sde_distance','sde_expected_bounce','sde_z_score',
             'sde_games_below','sde_games_above','sde_rolling_vol','sde_long_run_vol',
             'sde_vol_ratio','sde_vol_regime','sde_half_life','sde_dfs_class']
    # v2 interactions
    cols += ['sde_bounce_x_recent','sde_z_x_opp','sde_vol_x_z','sde_z_x_toi','sde_mu_x_opp','momentum_x_vol']
    # Chaos features (from v6)
    cols += ['lyapunov','hurst','recurrence','acf_decay','embed_dim','chaos_score']
    # Chaos interactions (from v6)
    cols += ['chaos_x_sde_z','hurst_x_momentum','lyap_x_vol','chaos_x_distance','hurst_x_consistency']
    # NEW: Line combination features
    cols += ['start_line_num','on_pp1','on_pp',
             'line_changed','line_stability',
             'linemate_fpts_avg','linemate_fpts_std',
             'linemate_goals_avg','linemate_toi_avg',
             'line_total_fpts_avg','line_fpts_share',
             'pp1_fpts_boost']
    # NEW: Line × Chaos interactions
    cols += ['chaos_x_line_stability','hurst_x_pp1','lyap_x_linemate_quality',
             'sde_mu_x_linemate','line_changed_x_momentum','pp1_x_opp']
    return cols


# ============================================================================
# OPTUNA TUNING (from v6)
# ============================================================================
def tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=50):
    if not HAS_OPTUNA:
        print("    Optuna not installed, using defaults")
        return get_default_params()
    def objective(trial):
        p = {
            'objective':'regression_l1','metric':'mae','verbose':-1,
            'num_leaves': trial.suggest_int('num_leaves',15,63),
            'learning_rate': trial.suggest_float('learning_rate',0.01,0.15,log=True),
            'feature_fraction': trial.suggest_float('feature_fraction',0.5,0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction',0.5,0.95),
            'bagging_freq': trial.suggest_int('bagging_freq',1,7),
            'min_child_samples': trial.suggest_int('min_child_samples',5,50),
            'reg_alpha': trial.suggest_float('reg_alpha',1e-3,10.0,log=True),
            'reg_lambda': trial.suggest_float('reg_lambda',1e-3,10.0,log=True),
            'min_split_gain': trial.suggest_float('min_split_gain',0.0,1.0),
            'max_depth': trial.suggest_int('max_depth',3,8),
        }
        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_val, label=y_val, reference=dt)
        m = lgb.train(p, dt, num_boost_round=500, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        return np.mean(np.abs(y_val - m.predict(X_val, num_iteration=m.best_iteration)))
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    bp.update({'objective':'regression_l1','metric':'mae','verbose':-1})
    print(f"    Best MAE: {study.best_value:.4f} | lr={bp['learning_rate']:.4f} leaves={bp['num_leaves']} depth={bp['max_depth']}")
    return bp


def get_default_params():
    return {'objective':'regression_l1','metric':'mae','num_leaves':31,'learning_rate':0.05,
            'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
            'min_child_samples':10,'reg_alpha':0.1,'reg_lambda':1.0,'verbose':-1}


# ============================================================================
# BACKTEST
# ============================================================================
def run_backtest(df_all, fcols, do_tune=False):
    print("\n" + "="*70)
    label = "TUNED+CHAOS+LINES" if do_tune else "DEFAULT+CHAOS+LINES"
    print(f"  LightGBM V7 ({label})")
    print("="*70)
    av = [c for c in fcols if c in df_all.columns]
    print(f"  Features: {len(av)}")

    # Report line feature coverage
    line_matched = (df_all['start_line_num'] > 0).sum()
    lm_matched = (df_all['linemate_fpts_avg'] > 0).sum()
    print(f"  Line assignment coverage: {line_matched:,} / {len(df_all):,} ({100*line_matched/len(df_all):.1f}%)")
    print(f"  Linemate quality coverage: {lm_matched:,} / {len(df_all):,} ({100*lm_matched/len(df_all):.1f}%)")

    rd = []; d = BACKTEST_START
    while d <= BACKTEST_END: rd.append(d); d += timedelta(days=RETRAIN_INTERVAL)
    results = []; bp = None; feat_imp = []

    for ri, rdate in enumerate(rd):
        print(f"\n  Retrain {ri+1}/{len(rd)}: {rdate.date()}")
        nr = rd[ri+1] if ri < len(rd)-1 else BACKTEST_END + timedelta(days=1)
        dtr = df_all[(df_all['game_date']<rdate)&(df_all['season_gp']>=8)].copy()
        if len(dtr)<100: continue

        sd = dtr['game_date'].quantile(0.8)
        dv = dtr[dtr['game_date']>=sd]; dt = dtr[dtr['game_date']<sd]
        Xt,yt = dt[av].fillna(0).values, dt['dk_fpts'].values
        Xv,yv = dv[av].fillna(0).values, dv['dk_fpts'].values
        print(f"    Train: {len(Xt):,} | Val: {len(Xv):,}")

        if do_tune and bp is None:
            print(f"    Tuning (50 trials)...")
            bp = tune_lgbm(Xt, yt, Xv, yv, n_trials=50)
        elif bp is None:
            bp = get_default_params()

        t0 = time.time()
        dtrain = lgb.Dataset(Xt,label=yt)
        dval = lgb.Dataset(Xv,label=yv,reference=dtrain)
        model = lgb.train(bp, dtrain, num_boost_round=1000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        print(f"    Trained {time.time()-t0:.1f}s ({model.best_iteration} rounds)")

        if ri == 0 or ri == len(rd)-1:
            feat_imp.append(dict(zip(av, model.feature_importance(importance_type='gain'))))

        dte = df_all[(df_all['game_date']>=rdate)&(df_all['game_date']<nr)&(df_all['season_gp']>=8)].copy()
        if len(dte)==0: continue
        Xe = dte[av].fillna(0).values
        preds = model.predict(Xe, num_iteration=model.best_iteration)
        actuals = dte['dk_fpts'].values
        mae = np.mean(np.abs(actuals-preds))
        print(f"    Test MAE: {mae:.4f} ({len(preds)} preds)")

        for i in range(len(preds)):
            results.append({'game_date':dte.iloc[i]['game_date'],'player_name':dte.iloc[i]['player_name'],
                           'position':dte.iloc[i]['position'],'actual':float(actuals[i]),
                           'predicted':float(preds[i]),'error':float(abs(actuals[i]-preds[i]))})

    if results:
        rdf = pd.DataFrame(results); mae = rdf['error'].mean()
        rmse = np.sqrt((rdf['error']**2).mean())
        print("\n" + "="*70)
        print(f"  LightGBM V7 ({label}) RESULTS")
        print("="*70)
        print(f"  Predictions: {len(rdf):,}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"\n  FULL LEADERBOARD:")
        print(f"  {'Model':<55} {'MAE':>8}")
        print(f"  {'-'*65}")
        models = [
            ('Chaos-Clustered (global, default params)', 4.488),
            ('LightGBM v2 (tuned, previous best)', 4.534),
            ('LightGBM v5 (DK features)', 4.538),
            ('LightGBM v6 (tuned+chaos, redundant)', 4.540),
            ('Season Average', 4.576),
            (f'LightGBM v7 ({label})', mae),
        ]
        for name, m in sorted(models, key=lambda x: x[1]):
            marker = " ★" if m == min(x[1] for x in models) else ""
            print(f"  {name:<55} {m:>8.4f}{marker}")

        print(f"\n  By Position:")
        for pos in 'CDLR':
            p = rdf[rdf['position']==pos]
            if len(p): print(f"    {pos}: MAE={p['error'].mean():.4f} (n={len(p)})")

        rdf['game_date'] = pd.to_datetime(rdf['game_date'])
        print(f"\n  By Window:")
        for ri, rdate in enumerate(rd):
            nr = rd[ri+1] if ri<len(rd)-1 else BACKTEST_END+timedelta(days=1)
            w = rdf[(rdf['game_date']>=rdate)&(rdf['game_date']<nr)]
            if len(w): print(f"    W{ri+1}: MAE={w['error'].mean():.4f} (n={len(w)})")

        if feat_imp:
            combined = {}
            for imp in feat_imp:
                for k,v in imp.items(): combined[k] = combined.get(k,0)+v
            si = sorted(combined.items(), key=lambda x:x[1], reverse=True)
            print(f"\n  Top 30 Features:")
            for r,(f,g) in enumerate(si[:30],1):
                line_flag = " ▲" if f in [
                    'start_line_num','on_pp1','on_pp','line_changed','line_stability',
                    'linemate_fpts_avg','linemate_fpts_std','linemate_goals_avg','linemate_toi_avg',
                    'line_total_fpts_avg','line_fpts_share','pp1_fpts_boost',
                    'chaos_x_line_stability','hurst_x_pp1','lyap_x_linemate_quality',
                    'sde_mu_x_linemate','line_changed_x_momentum','pp1_x_opp'
                ] else ""
                chaos_flag = " ◆" if f in ['lyapunov','hurst','recurrence','acf_decay','embed_dim','chaos_score',
                                            'chaos_x_sde_z','hurst_x_momentum','lyap_x_vol',
                                            'chaos_x_distance','hurst_x_consistency'] else ""
                print(f"    {r:>2}. {f:<45} {g:>10.1f}{line_flag}{chaos_flag}")
            print(f"\n  ▲ = line combination feature | ◆ = chaos-derived feature")

        rdf.to_csv(f'data/lgbm_v7_{label.lower().replace("+","_")}_results.csv', index=False)
        print(f"\n  Saved: data/lgbm_v7_{label.lower().replace('+','_')}_results.csv")
        return rdf, mae
    return None, None


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()
    if not args.backtest:
        print("Usage: python lgbm_v7_lines.py --backtest [--tune]"); return

    print("="*70)
    print("  LightGBM V7 — CHAOS + LINE COMBINATION FEATURES")
    print("="*70)

    print("\n[1] Loading data...")
    df = load_boxscore_data()
    dh = load_historical_data()
    print(f"  Boxscore: {len(df):,}")

    print("\n[2] Loading line assignments...")
    line_df = load_line_assignments()

    print("\n[3] Fitting SDE engine...")
    sde = SDEEngine()
    sde.fit_all_players(dh if dh is not None else df)

    print("\n[4] Feature engineering...")
    df = compute_opponent_fpts_allowed(df)
    df = add_sde_features(df, sde)

    print("\n[5] Computing chaos features...")
    chaos = ChaosAnalyzer()
    df = add_chaos_features(df, chaos)

    print("\n[6] Computing line combination features...")
    df = add_line_features(df, line_df)

    print("\n[7] Computing interactions...")
    df = add_interactions(df)
    fcols = get_feature_columns()

    print("\n[8] Running backtest...")
    t0 = time.time()
    rdf, mae = run_backtest(df, fcols, do_tune=args.tune)
    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
