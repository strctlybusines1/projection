#!/usr/bin/env python3
"""
goalie_lgbm_v3_lines.py — Goalie LightGBM v3: Chaos + Opponent Line Strength
==============================================================================
Google Colab Edition

WHAT'S NEW vs v2 (chaos):
  Adds opponent line strength features derived from dk_salaries line assignments.
  For goalies, line combinations matter on the OPPONENT side — a goalie facing
  a team with a dominant L1/PP1 has a very different FPTS distribution.

  OPPONENT LINE STRENGTH:
  - opp_l1_fpts_avg: Avg recent FPTS of opponent's L1 forwards
  - opp_l2_fpts_avg: Avg recent FPTS of opponent's L2 forwards
  - opp_pp1_fpts_avg: Avg recent FPTS of opponent's PP1-assigned players
  - opp_top6_fpts_avg: Combined avg of opponent's L1+L2 (top-6 forwards)

  OPPONENT LINE QUALITY:
  - opp_l1_goals_avg: Avg recent goals from opponent's L1
  - opp_pp1_count: Number of opponent skaters on PP1 (roster depth indicator)
  - opp_line_stability: Avg line stability of opponent's L1 (stable = more dangerous)
  - opp_top_heavy: Ratio of L1 FPTS to L2 FPTS (>1 = top-heavy offense)

  CHAOS × OPPONENT LINE INTERACTIONS:
  - chaos_x_opp_l1: Chaotic goalie vs elite L1 = amplified variance
  - hurst_x_opp_pp1: Trending goalie vs strong PP = regime indicator
  - sde_mu_x_opp_top6: True level adjusted by opponent offensive quality

  Everything from v2 preserved: SDE engine, chaos features, team strength,
  rolling goalie stats, Optuna tuning.

Run: python goalie_lgbm_v3_lines.py --backtest --tune
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
OU_MIN_GAMES = 10


# ============================================================================
# CHAOS ENGINE (from v2)
# ============================================================================
class ChaosAnalyzer:
    @staticmethod
    def lyapunov_exponent(x, lag=1):
        x = np.array(x, dtype=float); n = len(x)
        if n < 15: return 0.0
        m = 3; tau = lag
        if n < m*tau+5: return 0.0
        N = n-(m-1)*tau
        embedded = np.array([x[i:i+(m*tau):tau] for i in range(N)])
        divergences = []
        for i in range(N-10):
            dists = np.sum((embedded-embedded[i])**2, axis=1)
            dists[max(0,i-2):i+3] = np.inf
            j = np.argmin(dists)
            if j+10<N and i+10<N:
                d0 = max(np.sqrt(dists[j]),1e-10)
                for k in range(1, min(8, N-max(i,j))):
                    dk = np.sqrt(np.sum((embedded[i+k]-embedded[j+k])**2))
                    if dk > 0: divergences.append(np.log(dk/d0)/k)
        return float(np.median(divergences)) if len(divergences)>=5 else 0.0

    @staticmethod
    def hurst_exponent(x):
        x = np.array(x, dtype=float); n = len(x)
        if n < 20: return 0.5
        max_k = min(n//4,20); rs_vals=[]; ns_vals=[]
        for k in range(4, max_k+1):
            n_sub=n//k
            if n_sub<4: break
            rs_list=[]
            for i in range(k):
                sub=x[i*n_sub:(i+1)*n_sub]
                cumdev=np.cumsum(sub-sub.mean()); R=cumdev.max()-cumdev.min()
                S=sub.std(ddof=1) if sub.std(ddof=1)>0 else 1e-10
                rs_list.append(R/S)
            rs_vals.append(np.mean(rs_list)); ns_vals.append(n_sub)
        if len(rs_vals)<3: return 0.5
        try: return float(np.clip(np.polyfit(np.log(ns_vals),np.log(rs_vals),1)[0],0,1))
        except: return 0.5

    @staticmethod
    def recurrence_rate(x, threshold_pct=20):
        x = np.array(x, dtype=float); n = len(x)
        if n < 10: return 0.0
        embedded = np.column_stack([x[:-1],x[1:]])
        threshold = np.std(x)*threshold_pct/100
        n_recur=0; total=0
        for i in range(min(100,len(embedded))):
            dists = np.sqrt(np.sum((embedded-embedded[i])**2,axis=1))
            dists[max(0,i-1):i+2]=np.inf
            n_recur+=np.sum(dists<threshold); total+=len(dists)-3
        return n_recur/max(total,1)

    @staticmethod
    def autocorrelation_decay(x):
        x = np.array(x, dtype=float)
        if len(x)<10: return 5.0
        x=x-x.mean(); var=np.var(x)
        if var==0: return 5.0
        for lag in range(1, min(10,len(x)//2)):
            if np.mean(x[:-lag]*x[lag:])/var < 0.5: return float(lag)
        return 10.0

    def compute_all(self, fpts_series):
        x = np.array(fpts_series, dtype=float)
        if len(x)<15:
            return {'lyapunov':0,'hurst':0.5,'recurrence':0,'acf_decay':5,'chaos_score':0}
        lyap=self.lyapunov_exponent(x); hurst=self.hurst_exponent(x)
        recur=self.recurrence_rate(x); acf_d=self.autocorrelation_decay(x)
        chaos_score = ((1.0 if lyap>0 else 0.0)*0.3 + (1.0-recur)*0.2 +
                       (1.0-acf_d/10.0)*0.2 + abs(hurst-0.5)*2.0*0.3)
        return {'lyapunov':lyap,'hurst':hurst,'recurrence':recur,
                'acf_decay':acf_d,'chaos_score':chaos_score}


# ============================================================================
# GOALIE SDE (from v2)
# ============================================================================
class GoalieSDE:
    def __init__(self): self.ou_params = {}
    def fit_ou(self, x):
        x = np.array(x, dtype=float)
        if len(x)<8: return None
        xp,xn = x[:-1],x[1:]; xb,yb = xp.mean(),xn.mean()
        ss = np.sum((xp-xb)**2)
        if ss==0: return None
        b = np.sum((xp-xb)*(xn-yb))/ss; a=yb-b*xb; theta=1.0-b
        if theta<=0 or theta>2: return None
        mu=a/theta; sigma=np.std(xn-(a+b*xp), ddof=2)
        if sigma<=0: return None
        return {'theta':theta,'mu':mu,'sigma':sigma,'half_life':min(np.log(2)/theta,20.0)}
    def fit_all(self, df):
        n=0
        for pid,g in df.groupby('player_id'):
            fpts=g.sort_values('game_date')['dk_fpts'].values
            if len(fpts)<OU_MIN_GAMES: continue
            ou=self.fit_ou(fpts)
            if ou: self.ou_params[pid]=ou; n+=1
        print(f"  SDE fitted: {n} goalies")
    def compute_features(self, pid, hist):
        if pid not in self.ou_params or len(hist)<3: return None
        ou=self.ou_params[pid]; w=max(0.3,OU_MIN_GAMES/(OU_MIN_GAMES+len(hist)))
        mu_b=w*ou['mu']+(1-w)*np.mean(hist); dist=mu_b-hist[-1]; sigma=ou['sigma']
        z=dist/sigma if sigma>0 else 0
        rv=np.var(hist[-5:]) if len(hist)>=5 else np.var(hist)
        return {'sde_mu':mu_b,'sde_distance':np.clip(dist,-30,30),
                'sde_z_score':np.clip(z,-3,3),'sde_rolling_vol':rv,
                'sde_vol_ratio':np.clip(rv/(sigma**2+1e-6),0,5),
                'sde_half_life':ou['half_life']}


# ============================================================================
# DATA & FEATURES
# ============================================================================
def load_data():
    conn = sqlite3.connect(DB_PATH)
    gl = pd.read_sql("""
        SELECT player_id, player_name, team, game_date, opponent, home_road,
               games_started, decision, shots_against, goals_against, saves,
               save_pct, shutouts, goals, assists, toi_seconds, dk_fpts
        FROM game_logs_goalies ORDER BY game_date, player_id
    """, conn)
    gl['game_date'] = pd.to_datetime(gl['game_date'])
    print(f"  Current: {len(gl):,} rows ({gl['player_id'].nunique()} goalies)")
    try:
        hg = pd.read_sql("SELECT * FROM historical_goalies ORDER BY game_date, player_id", conn)
        hg['game_date'] = pd.to_datetime(hg['game_date'])
        if 'sv_pct' in hg.columns and 'save_pct' not in hg.columns:
            hg['save_pct'] = hg['sv_pct']
        hg = hg[hg['dk_fpts'].notna()].copy()
        print(f"  Historical: {len(hg):,} rows ({hg['player_id'].nunique()} goalies)")
    except Exception as e:
        hg = pd.DataFrame(); print(f"  No historical: {e}")
    sk = pd.read_sql("SELECT team, game_date, opponent, goals, shots FROM boxscore_skaters", conn)
    sk['game_date'] = pd.to_datetime(sk['game_date'])
    try:
        hsk = pd.read_sql("SELECT team, game_date, opponent, goals, shots FROM historical_skaters WHERE goals IS NOT NULL", conn)
        hsk['game_date'] = pd.to_datetime(hsk['game_date'])
    except: hsk = pd.DataFrame()
    conn.close()
    return gl, hg, sk, hsk


def load_line_assignments():
    """Load skater line assignments for opponent line strength features."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT ds.player_name, ds.team, ds.position, ds.slate_date, ds.start_line, ds.pp_unit,
               gl.dk_fpts, gl.goals, gl.toi_seconds
        FROM dk_salaries ds
        INNER JOIN game_logs_skaters gl
            ON ds.player_name = gl.player_name AND ds.slate_date = gl.game_date
        WHERE ds.start_line IN ('1','2','3','4') AND ds.position != 'G'
        ORDER BY ds.slate_date, ds.team, ds.start_line
    """, conn)
    conn.close()
    df['slate_date'] = pd.to_datetime(df['slate_date'])
    df['start_line_num'] = df['start_line'].astype(int)
    df['on_pp1'] = (df['pp_unit'] == 1.0).astype(int)
    print(f"  Line+outcome data: {len(df):,} rows across {df['slate_date'].nunique()} dates")
    return df


def compute_team_strength(skater_df):
    tg = skater_df.groupby(['game_date','team']).agg(
        team_goals=('goals','sum'), team_shots=('shots','sum')).reset_index()
    og = tg[['game_date','team','team_goals']].copy()
    og.columns = ['game_date','opponent','opp_goals_for']
    to = skater_df[['game_date','team','opponent']].drop_duplicates()
    tg = tg.merge(to, on=['game_date','team'], how='left')
    tg = tg.merge(og, on=['game_date','opponent'], how='left')
    tg['team_ga'] = tg['opp_goals_for'].fillna(0)
    tg = tg.sort_values(['team','game_date'])
    for c in ['team_goals','team_ga','team_shots']:
        tg[f'{c}_10g'] = tg.groupby('team')[c].transform(lambda x: x.rolling(10,min_periods=1).mean())
    os = tg[['game_date','team','team_goals_10g','team_ga_10g','team_shots_10g']].copy()
    os.columns = ['game_date','opponent','opp_goals_10g','opp_ga_10g','opp_shots_10g']
    tg = tg.merge(os, on=['game_date','opponent'], how='left')
    return tg[['game_date','team','opponent','team_goals_10g','team_ga_10g','team_shots_10g',
               'opp_goals_10g','opp_ga_10g','opp_shots_10g']]


def compute_opponent_line_strength(line_df):
    """Pre-compute per-team, per-date opponent line strength metrics.

    For each (team, date) pair, computes rolling averages of that team's
    L1, L2, PP1 FPTS from prior games. This is then used by goalies facing
    that team as the opponent.

    Returns dict: (team, date) -> {opp_l1_fpts_avg, opp_l2_fpts_avg, ...}
    """
    print("  Pre-computing opponent line strength...")
    t0 = time.time()

    # Build per-team, per-date, per-line FPTS totals
    line_daily = line_df.groupby(['slate_date', 'team', 'start_line_num']).agg(
        line_fpts=('dk_fpts', 'sum'),
        line_goals=('goals', 'sum'),
        n_players=('player_name', 'count'),
    ).reset_index()

    # Also PP1 aggregate
    pp1_daily = line_df[line_df['on_pp1'] == 1].groupby(['slate_date', 'team']).agg(
        pp1_fpts=('dk_fpts', 'sum'),
        pp1_count=('player_name', 'count'),
    ).reset_index()

    # Build per-team line history for rolling computation
    # For L1 and L2 separately
    team_line_history = {}  # (team) -> list of (date, l1_fpts, l2_fpts, pp1_fpts, l1_goals, pp1_count)

    all_dates = sorted(line_daily['slate_date'].unique())
    for date in all_dates:
        day_data = line_daily[line_daily['slate_date'] == date]
        pp1_data = pp1_daily[pp1_daily['slate_date'] == date]

        for team in day_data['team'].unique():
            team_day = day_data[day_data['team'] == team]
            l1_row = team_day[team_day['start_line_num'] == 1]
            l2_row = team_day[team_day['start_line_num'] == 2]
            pp1_row = pp1_data[pp1_data['team'] == team]

            l1_fpts = l1_row['line_fpts'].sum() if len(l1_row) else 0
            l2_fpts = l2_row['line_fpts'].sum() if len(l2_row) else 0
            l1_goals = l1_row['line_goals'].sum() if len(l1_row) else 0
            pp1_fpts = pp1_row['pp1_fpts'].sum() if len(pp1_row) else 0
            pp1_count = pp1_row['pp1_count'].sum() if len(pp1_row) else 0

            if team not in team_line_history:
                team_line_history[team] = []
            team_line_history[team].append({
                'date': date,
                'l1_fpts': l1_fpts, 'l2_fpts': l2_fpts,
                'pp1_fpts': pp1_fpts, 'l1_goals': l1_goals,
                'pp1_count': pp1_count,
            })

    # Now compute rolling averages: for each (team, date), look at last 5 games BEFORE that date
    opp_line_map = {}  # (team, date) -> feature dict

    for team, history in team_line_history.items():
        history.sort(key=lambda x: x['date'])
        for i, entry in enumerate(history):
            # Use prior games only (no leakage)
            prior = history[:i]
            if not prior:
                continue

            recent = prior[-5:]  # Last 5 games
            opp_line_map[(team, entry['date'])] = {
                'opp_l1_fpts_avg': np.mean([g['l1_fpts'] for g in recent]),
                'opp_l2_fpts_avg': np.mean([g['l2_fpts'] for g in recent]),
                'opp_pp1_fpts_avg': np.mean([g['pp1_fpts'] for g in recent]),
                'opp_top6_fpts_avg': np.mean([g['l1_fpts'] + g['l2_fpts'] for g in recent]),
                'opp_l1_goals_avg': np.mean([g['l1_goals'] for g in recent]),
                'opp_pp1_count': recent[-1]['pp1_count'] if recent else 0,
                'opp_top_heavy': (
                    np.mean([g['l1_fpts'] for g in recent]) /
                    max(np.mean([g['l2_fpts'] for g in recent]), 0.1)
                ),
            }

    # Compute line stability per team: how often does L1 roster change?
    # (Simplified: track unique L1 player sets per team)
    l1_players = line_df[line_df['start_line_num'] == 1].groupby(
        ['slate_date', 'team']
    )['player_name'].apply(frozenset).reset_index()
    l1_players = l1_players.sort_values(['team', 'slate_date'])

    team_stability = {}
    for team in l1_players['team'].unique():
        team_l1 = l1_players[l1_players['team'] == team].reset_index(drop=True)
        for i in range(len(team_l1)):
            date = team_l1.iloc[i]['slate_date']
            if i == 0:
                stability = 0
            else:
                # Count consecutive dates with same L1 roster
                stability = 0
                current_roster = team_l1.iloc[i]['player_name']
                for j in range(i-1, -1, -1):
                    if team_l1.iloc[j]['player_name'] == current_roster:
                        stability += 1
                    else:
                        break
            key = (team, date)
            if key in opp_line_map:
                opp_line_map[key]['opp_line_stability'] = min(stability, 10)

    print(f"  Opponent line strength: {len(opp_line_map):,} entries ({time.time()-t0:.1f}s)")
    return opp_line_map


def build_features(gl, team_stats, sde, chaos, opp_line_map=None):
    df = gl.sort_values(['player_id','game_date']).reset_index(drop=True)

    # Rolling stats (5g and 10g windows)
    for col in ['saves','goals_against','save_pct','dk_fpts','toi_seconds','shots_against']:
        if col not in df.columns: continue
        s = df.groupby('player_id')[col].shift(1)
        for w in [5,10]:
            df[f'g_{col}_{w}g'] = s.groupby(df['player_id']).rolling(w,min_periods=1).mean().reset_index(level=0,drop=True)

    # FPTS std
    sf = df.groupby('player_id')['dk_fpts'].shift(1)
    df['g_fpts_std_10'] = sf.groupby(df['player_id']).rolling(10,min_periods=2).std().reset_index(level=0,drop=True).fillna(0)

    # Season averages
    for col in ['dk_fpts','saves','goals_against','save_pct','toi_seconds']:
        if col not in df.columns: continue
        sc = df.groupby('player_id')[col].shift(1)
        cs = sc.groupby(df['player_id']).cumsum()
        gp = df.groupby('player_id').cumcount().clip(lower=1)
        df[f'g_avg_{col}'] = (cs/gp).fillna(0)

    # Win rates
    df['is_win'] = (df.get('decision','')=='W').astype(float)
    sw = df.groupby('player_id')['is_win'].shift(1)
    df['g_winrate_10'] = sw.groupby(df['player_id']).rolling(10,min_periods=1).mean().reset_index(level=0,drop=True).fillna(0.5)
    sc_w = sw.groupby(df['player_id']).cumsum()
    gpc = df.groupby('player_id').cumcount().clip(lower=1)
    df['g_season_winrate'] = (sc_w/gpc).fillna(0.5)

    # Context
    df['g_gp'] = df.groupby('player_id').cumcount()+1
    df['g_is_home'] = (df['home_road'].str.upper()=='H').astype(int)
    df['g_prev_date'] = df.groupby('player_id')['game_date'].shift(1)
    df['g_days_rest'] = (df['game_date']-df['g_prev_date']).dt.days.fillna(3).clip(0,14)
    df['g_is_b2b'] = (df['g_days_rest']<=1).astype(int)
    df['g_last_fpts'] = df.groupby('player_id')['dk_fpts'].shift(1).fillna(0)
    df['g_last_svpct'] = df.groupby('player_id')['save_pct'].shift(1).fillna(0.9) if 'save_pct' in df.columns else 0.9

    # Momentum
    if 'g_dk_fpts_5g' in df.columns and 'g_dk_fpts_10g' in df.columns:
        df['g_momentum'] = df['g_dk_fpts_5g'] - df['g_dk_fpts_10g']
    else: df['g_momentum'] = 0

    # Team strength
    if team_stats is not None and len(team_stats)>0:
        df = df.merge(team_stats, on=['game_date','team','opponent'], how='left')
        for c in ['team_goals_10g','team_ga_10g','team_shots_10g',
                   'opp_goals_10g','opp_ga_10g','opp_shots_10g']:
            if c in df.columns: df[c] = df[c].fillna(df[c].median())

    # SDE features
    sde_cols = ['sde_mu','sde_distance','sde_z_score','sde_rolling_vol','sde_vol_ratio','sde_half_life']
    for c in sde_cols: df[c] = 0.0
    if sde:
        for pid,g in df.groupby('player_id'):
            gs=g.sort_values('game_date'); idx=gs.index.tolist(); fv=gs['dk_fpts'].values
            for i in range(3,len(idx)):
                f=sde.compute_features(pid, fv[:i])
                if f:
                    for c in sde_cols:
                        if c in f: df.loc[idx[i],c]=f[c]

    # CHAOS features
    chaos_cols = ['lyapunov','hurst','recurrence','acf_decay','chaos_score']
    for c in chaos_cols: df[c] = 0.0
    if chaos:
        for pid,g in df.groupby('player_id'):
            gs=g.sort_values('game_date'); idx=gs.index.tolist(); fv=gs['dk_fpts'].values
            for i in range(15,len(idx)):
                feat=chaos.compute_all(fv[:i])
                for c in chaos_cols: df.loc[idx[i],c]=feat.get(c,0)

    # NEW: Opponent line strength features
    opp_line_cols = ['opp_l1_fpts_avg','opp_l2_fpts_avg','opp_pp1_fpts_avg',
                     'opp_top6_fpts_avg','opp_l1_goals_avg','opp_pp1_count',
                     'opp_top_heavy','opp_line_stability']
    for c in opp_line_cols: df[c] = 0.0

    if opp_line_map:
        n_matched = 0
        for idx_row, row in df.iterrows():
            opp = row.get('opponent', '')
            gd = row['game_date']
            key = (opp, gd)
            if key in opp_line_map:
                for c in opp_line_cols:
                    if c in opp_line_map[key]:
                        df.at[idx_row, c] = opp_line_map[key][c]
                n_matched += 1
        print(f"  Opponent line features matched: {n_matched:,} / {len(df):,}")

    # Interactions (v2 + chaos + NEW opponent line)
    df['sde_mu_x_opp'] = df['sde_mu'] * df.get('opp_goals_10g',0)
    df['saves_x_opp_shots'] = df.get('g_saves_5g',0) * df.get('opp_shots_10g',0)
    # Chaos interactions (from v2)
    df['chaos_x_sde_z'] = df.get('chaos_score',0) * df.get('sde_z_score',0)
    df['hurst_x_momentum'] = df.get('hurst',0) * df.get('g_momentum',0)
    df['lyap_x_vol'] = df.get('lyapunov',0) * df.get('sde_rolling_vol',0)
    df['chaos_x_opp'] = df.get('chaos_score',0) * df.get('opp_goals_10g',0)
    # NEW: Opponent line × chaos interactions
    df['chaos_x_opp_l1'] = df.get('chaos_score',0) * df.get('opp_l1_fpts_avg',0)
    df['hurst_x_opp_pp1'] = df.get('hurst',0) * df.get('opp_pp1_fpts_avg',0)
    df['sde_mu_x_opp_top6'] = df.get('sde_mu',0) * (df.get('opp_top6_fpts_avg',0) / 30.0)  # Normalize ~30 total

    return df


def get_feature_columns():
    cols = []
    for col in ['saves','goals_against','save_pct','dk_fpts','toi_seconds','shots_against']:
        for w in [5,10]: cols.append(f'g_{col}_{w}g')
    cols += ['g_fpts_std_10']
    for col in ['dk_fpts','saves','goals_against','save_pct','toi_seconds']:
        cols.append(f'g_avg_{col}')
    cols += ['g_winrate_10','g_season_winrate','g_gp','g_is_home',
             'g_days_rest','g_is_b2b','g_last_fpts','g_last_svpct','g_momentum']
    cols += ['team_goals_10g','team_ga_10g','team_shots_10g',
             'opp_goals_10g','opp_ga_10g','opp_shots_10g']
    cols += ['sde_mu','sde_distance','sde_z_score','sde_rolling_vol','sde_vol_ratio','sde_half_life']
    # Chaos features
    cols += ['lyapunov','hurst','recurrence','acf_decay','chaos_score']
    # v2 interactions
    cols += ['sde_mu_x_opp','saves_x_opp_shots',
             'chaos_x_sde_z','hurst_x_momentum','lyap_x_vol','chaos_x_opp']
    # NEW: Opponent line strength features
    cols += ['opp_l1_fpts_avg','opp_l2_fpts_avg','opp_pp1_fpts_avg',
             'opp_top6_fpts_avg','opp_l1_goals_avg','opp_pp1_count',
             'opp_top_heavy','opp_line_stability']
    # NEW: Opponent line × chaos interactions
    cols += ['chaos_x_opp_l1','hurst_x_opp_pp1','sde_mu_x_opp_top6']
    return cols


# ============================================================================
# OPTUNA (v2 anti-overfit ranges)
# ============================================================================
def tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=50):
    if not HAS_OPTUNA: return get_default_params()
    def objective(trial):
        p = {'objective':'regression_l1','metric':'mae','verbose':-1,
             'num_leaves': trial.suggest_int('num_leaves',8,30),
             'learning_rate': trial.suggest_float('learning_rate',0.01,0.08,log=True),
             'feature_fraction': trial.suggest_float('feature_fraction',0.5,0.9),
             'bagging_fraction': trial.suggest_float('bagging_fraction',0.5,0.9),
             'bagging_freq': trial.suggest_int('bagging_freq',1,5),
             'min_child_samples': trial.suggest_int('min_child_samples',10,40),
             'reg_alpha': trial.suggest_float('reg_alpha',0.1,10.0,log=True),
             'reg_lambda': trial.suggest_float('reg_lambda',0.1,10.0,log=True),
             'min_split_gain': trial.suggest_float('min_split_gain',0.1,2.0),
             'max_depth': trial.suggest_int('max_depth',3,5)}
        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_val, label=y_val, reference=dt)
        m = lgb.train(p, dt, num_boost_round=200, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)])
        return np.mean(np.abs(y_val - m.predict(X_val, num_iteration=m.best_iteration)))
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    bp.update({'objective':'regression_l1','metric':'mae','verbose':-1})
    print(f"    Best: lr={bp['learning_rate']:.4f} leaves={bp['num_leaves']} depth={bp['max_depth']} | val={study.best_value:.4f}")
    return bp

def get_default_params():
    return {'objective':'regression_l1','metric':'mae','num_leaves':20,'learning_rate':0.03,
            'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':3,
            'min_child_samples':15,'reg_alpha':1.0,'reg_lambda':1.0,
            'min_split_gain':0.5,'max_depth':4,'verbose':-1}


# ============================================================================
# BACKTEST
# ============================================================================
def run_backtest(df_current, df_hist, do_tune=False):
    label = "TUNED+CHAOS+LINES" if do_tune else "DEFAULT+CHAOS+LINES"
    print("\n" + "="*70)
    print(f"  Goalie LightGBM v3 ({label})")
    print("="*70)
    av = [c for c in get_feature_columns() if c in df_current.columns]
    if df_hist is not None and len(df_hist)>0:
        av = [c for c in av if c in df_hist.columns]
    print(f"  Features: {len(av)}")

    # Report opponent line feature coverage
    opp_matched = (df_current['opp_l1_fpts_avg'] > 0).sum()
    print(f"  Opponent line coverage: {opp_matched:,} / {len(df_current):,} ({100*opp_matched/len(df_current):.1f}%)")

    rd = []; d = BACKTEST_START
    while d <= BACKTEST_END: rd.append(d); d += timedelta(days=RETRAIN_INTERVAL)
    results = []; bp = None; feat_imp = []

    for ri, rdate in enumerate(rd):
        print(f"\n  Retrain {ri+1}/{len(rd)}: {rdate.date()}")
        nr = rd[ri+1] if ri<len(rd)-1 else BACKTEST_END+timedelta(days=1)
        train_mask = (df_current['game_date']<rdate)&(df_current['g_gp']>=3)
        test_mask = (df_current['game_date']>=rdate)&(df_current['game_date']<nr)&(df_current['g_gp']>=3)
        if train_mask.sum()<30 or test_mask.sum()==0: continue
        dtr = df_current[train_mask]

        # Historical + current training data
        if df_hist is not None and len(df_hist)>0:
            X_cur = dtr[av].fillna(0).values; y_cur = dtr['dk_fpts'].values
            dh = df_hist[df_hist['g_gp']>=3]
            av_h = [c for c in av if c in dh.columns]
            if len(av_h) == len(av):
                X_hist = dh[av].fillna(0).values; y_hist = dh['dk_fpts'].values
                X_all = np.vstack([X_cur, X_hist])
                y_all = np.concatenate([y_cur, y_hist])
                w_all = np.concatenate([np.ones(len(X_cur)), np.full(len(X_hist),0.3)])
            else:
                X_all = X_cur; y_all = y_cur; w_all = np.ones(len(X_all))
        else:
            X_all = dtr[av].fillna(0).values; y_all = dtr['dk_fpts'].values
            w_all = np.ones(len(X_all))

        n_val = max(30, int(len(dtr)*0.2))
        Xv = dtr[av].fillna(0).values[-n_val:]; yv = dtr['dk_fpts'].values[-n_val:]
        print(f"    Train: {len(X_all):,} | Val: {n_val}")

        if do_tune and bp is None and len(dtr)>=50:
            dt_tune = dtr[dtr['game_date']<dtr['game_date'].quantile(0.8)]
            if len(dt_tune)>=30:
                print(f"    Tuning (50 trials)...")
                bp = tune_lgbm(dt_tune[av].fillna(0).values, dt_tune['dk_fpts'].values, Xv, yv, n_trials=50)
        if bp is None:
            bp = get_default_params()

        dtrain = lgb.Dataset(X_all, label=y_all, weight=w_all)
        dval = lgb.Dataset(Xv, label=yv, reference=dtrain)
        model = lgb.train(bp, dtrain, num_boost_round=200, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

        if ri==0 or ri==len(rd)-1:
            feat_imp.append(dict(zip(av, model.feature_importance(importance_type='gain'))))

        dte = df_current[test_mask].copy()
        Xt = dte[av].fillna(0).values
        preds = model.predict(Xt, num_iteration=model.best_iteration)
        actuals = dte['dk_fpts'].values
        baseline = dte['g_avg_dk_fpts'].values
        mae = np.mean(np.abs(actuals-preds))
        mae_bl = np.mean(np.abs(actuals-baseline))
        print(f"    MAE: {mae:.4f} | BL: {mae_bl:.4f} ({model.best_iteration} rounds, n={len(dte)})")

        for i in range(len(preds)):
            results.append({
                'game_date':dte.iloc[i]['game_date'], 'player_name':dte.iloc[i]['player_name'],
                'actual':float(actuals[i]), 'predicted':float(preds[i]),
                'baseline':float(baseline[i]),
                'error':float(abs(actuals[i]-preds[i])),
                'error_bl':float(abs(actuals[i]-baseline[i])),
            })

    if results:
        rdf = pd.DataFrame(results)
        mae = rdf['error'].mean(); mae_bl = rdf['error_bl'].mean()
        print("\n" + "="*70)
        print(f"  GOALIE LightGBM v3 ({label}) RESULTS")
        print("="*70)
        print(f"  Predictions: {len(rdf):,}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  BL MAE:   {mae_bl:.4f}")
        print(f"\n  LEADERBOARD:")
        models = [
            ('Chaos-Cluster Global (default)', 7.201),
            ('Goalie LGB v2 (chaos, tuned)', 7.208),
            ('Goalie LGB v1b (previous best)', 7.252),
            ('XGBoost v1', 7.880),
            ('Season Average', mae_bl),
            (f'Goalie LGB v3 ({label})', mae),
        ]
        for name,m in sorted(models, key=lambda x:x[1]):
            marker = " ★" if m==min(x[1] for x in models) else ""
            print(f"  {name:<50} {m:>8.4f}{marker}")

        rdf['game_date'] = pd.to_datetime(rdf['game_date'])
        print(f"\n  By Window:")
        for ri,rdate in enumerate(rd):
            nr = rd[ri+1] if ri<len(rd)-1 else BACKTEST_END+timedelta(days=1)
            w = rdf[(rdf['game_date']>=rdate)&(rdf['game_date']<nr)]
            if len(w): print(f"    W{ri+1}: MAE={w['error'].mean():.4f} | BL={w['error_bl'].mean():.4f} (n={len(w)})")

        if feat_imp:
            combined = {}
            for imp in feat_imp:
                for k,v in imp.items(): combined[k]=combined.get(k,0)+v
            si = sorted(combined.items(), key=lambda x:x[1], reverse=True)
            print(f"\n  Top 25 Features:")
            for r,(f,g) in enumerate(si[:25],1):
                line_flag = " ▲" if f in [
                    'opp_l1_fpts_avg','opp_l2_fpts_avg','opp_pp1_fpts_avg',
                    'opp_top6_fpts_avg','opp_l1_goals_avg','opp_pp1_count',
                    'opp_top_heavy','opp_line_stability',
                    'chaos_x_opp_l1','hurst_x_opp_pp1','sde_mu_x_opp_top6'
                ] else ""
                chaos_flag = " ◆" if f in ['lyapunov','hurst','recurrence','acf_decay','chaos_score',
                                            'chaos_x_sde_z','hurst_x_momentum','lyap_x_vol','chaos_x_opp'] else ""
                print(f"    {r:>2}. {f:<40} {g:>10.1f}{line_flag}{chaos_flag}")
            print(f"\n  ▲ = opponent line feature | ◆ = chaos-derived feature")

        rdf.to_csv(f'data/goalie_lgbm_v3_{label.lower().replace("+","_")}_results.csv', index=False)
        print(f"\n  Saved results")
        return rdf,mae
    return None,None


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()
    if not args.backtest:
        print("Usage: python goalie_lgbm_v3_lines.py --backtest [--tune]"); return

    print("="*70)
    print("  GOALIE LightGBM v3 — CHAOS + OPPONENT LINE STRENGTH")
    print("="*70)
    print("\n[1] Loading data...")
    gl, hg, sk, hsk = load_data()

    print("\n[2] Loading line assignments...")
    line_df = load_line_assignments()

    print("\n[3] Computing team strength...")
    all_sk = pd.concat([hsk,sk], ignore_index=True) if len(hsk)>0 else sk
    team_stats = compute_team_strength(all_sk)

    print("\n[4] Computing opponent line strength...")
    opp_line_map = compute_opponent_line_strength(line_df)

    print("\n[5] Fitting SDE + Chaos...")
    sde = GoalieSDE(); chaos = ChaosAnalyzer()
    all_gl = pd.concat([hg,gl], ignore_index=True) if len(hg)>0 else gl
    sde.fit_all(all_gl)

    print("\n[6] Building current features...")
    t0 = time.time()
    df_current = build_features(gl, team_stats, sde, chaos, opp_line_map)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n[7] Building historical features...")
    df_hist = None
    if len(hg)>0:
        if 'season' not in hg.columns:
            hg['season'] = hg['game_date'].apply(lambda d: d.year if d.month>=9 else d.year-1)
        for col in ['saves','goals_against','save_pct','dk_fpts','toi_seconds','shots_against']:
            if col not in hg.columns: hg[col]=0
        if 'decision' not in hg.columns: hg['decision']=''
        if 'home_road' not in hg.columns: hg['home_road']='H'
        if 'opponent' not in hg.columns: hg['opponent']=''
        frames = []
        for season,sdf in hg.groupby('season'):
            frames.append(build_features(sdf.copy(), team_stats, sde, chaos))
        df_hist = pd.concat(frames, ignore_index=True) if frames else None
        if df_hist is not None: print(f"  Historical: {len(df_hist):,} rows")

    print("\n[8] Running backtest...")
    t0 = time.time()
    run_backtest(df_current, df_hist, do_tune=args.tune)
    print(f"\n  Total time: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
