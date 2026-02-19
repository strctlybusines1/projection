#!/usr/bin/env python3
"""
hmm_edge.py — Daily HMM Edge Tool (Production)
================================================
Integrates with the existing NHL DFS projection pipeline.
Uses ONLY backtested, confirmed signals:

  Signal 1: Opponent matchup regime  (+0.31 FPTS, p=0.003)
  Signal 2: 6-state downgrade bounce (+1.04 FPTS, p=0.005, 1.1% lower owned)
  Signal 3: 5-day transition window  (+0.27 FPTS, p=0.007)

Usage:
    # Standalone — print today's edges
    python hmm_edge.py

    # With DK salary file — merge edges into projections
    python hmm_edge.py --salary daily_salaries/DKSalaries_2.25.26.csv

    # Export edges to CSV for optimizer
    python hmm_edge.py --salary daily_salaries/DKSalaries_2.25.26.csv --csv

    # Called from main.py pipeline (returns DataFrame)
    from hmm_edge import HMMEdgeModel
    hmm = HMMEdgeModel()
    edges = hmm.get_slate_edges(salary_df)

Runs in <30 seconds. No GPU needed.
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.decomposition import PCA
except ImportError:
    print("Install: pip install hmmlearn scikit-learn")
    raise


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION — Calibrated from backtest
# ═══════════════════════════════════════════════════════════════

# ── Signal Classification (from combined backtest) ──
# These are SELECTION signals, not projection multipliers.
# Combined backtest showed optimal boost = x1.00 for MAE.
# But surprise effects are real:
#   Stacked (DN+WEAK):     +0.87 FPTS surprise (p=0.007) — SYNERGY
#   Downgrade + Strong:    -0.62 FPTS surprise — MATCHUP WINS
#   Downgrade alone:       +0.06 for depth, +0.93 for stars
#   Weak opponent alone:   +0.07

# Small projection boosts for optimizer guidance (won't hurt MAE)
OPP_WEAK_BOOST = 1.03       # +3% vs weak-regime team
OPP_MID_BOOST = 1.00        # neutral
OPP_STRONG_FADE = 0.97      # -3% vs strong-regime team

DOWNGRADE_BOOST = 1.03      # +3% for recent downgrades (conservative)
TRANSITION_BOOST = 1.01     # +1% for any recent transition (tiebreaker)

# Stacked signal gets extra (synergy confirmed p=0.007)
STACKED_DN_WEAK_BOOST = 1.06  # +6% when downgrade + weak opponent

# Conflict: downgrade + strong opponent — DON'T boost, net fade
CONFLICT_DN_STRONG_FADE = 0.97  # -3% when bounce meets brick wall

# Max combined adjustment
MAX_HMM_SWING = 0.08        # cap at ±8%

# Model parameters (from backtest: 6-state, basic features, diag cov)
N_STATES = 6
N_PCA = 8
WINDOW = 5
MIN_GAMES = 12
FEATURES = ['goals', 'assists', 'shots', 'hits', 'blocked_shots', 'dk_fpts']

# NST team features for opponent regime
TEAM_FEATURES = ['CF%', 'FF%', 'SF%', 'GF%', 'xGF%', 'SCF%', 'HDCF%',
                  'xGF/60', 'xGA/60', 'GF/60', 'GA/60', 'SH%', 'SV%', 'PDO']

NST_TEAM_MAP = {
    'Anaheim Ducks': 'ANA', 'Boston Bruins': 'BOS', 'Buffalo Sabres': 'BUF',
    'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR', 'Chicago Blackhawks': 'CHI',
    'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ', 'Dallas Stars': 'DAL',
    'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM', 'Florida Panthers': 'FLA',
    'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN', 'Montreal Canadiens': 'MTL',
    'Montréal Canadiens': 'MTL', 'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD',
    'New York Islanders': 'NYI', 'New York Rangers': 'NYR', 'Ottawa Senators': 'OTT',
    'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS',
    'Seattle Kraken': 'SEA', 'St. Louis Blues': 'STL', 'St Louis Blues': 'STL',
    'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR',
    'Utah Hockey Club': 'UTA', 'Utah Mammoth': 'UTA',
    'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG',
}

DB_PATH = Path(__file__).parent / 'data' / 'nhl_dfs_history.db'
NST_DIR = Path(__file__).parent / 'nht_daily'
MODEL_DIR = Path(__file__).parent / 'data' / 'hmm_models'


class HMMEdgeModel:
    """
    Production HMM edge model.
    Fits once per day, decodes player states, classifies opponent regimes.
    Returns multiplicative boost factors for the projection pipeline.
    """

    def __init__(self, db_path=None, nst_dir=None):
        self.db_path = db_path or DB_PATH
        self.nst_dir = nst_dir or NST_DIR
        self.player_model = None
        self.player_pca = None
        self.player_states = {}
        self.team_regimes = {}
        self.team_quality = {}
        self.season_avgs = {}
        self.pid_to_name = {}
        self._fitted = False

    # ───────────────────────────────────────────────────────────
    #  FIT (call once per day)
    # ───────────────────────────────────────────────────────────

    def fit(self):
        """Fit player HMM + team HMM on all available data."""
        print("  [HMM] Fitting models...")
        self._fit_player_hmm()
        self._fit_team_hmm()
        self._fitted = True
        print(f"  [HMM] Ready: {len(self.player_states)} players, "
              f"{len(self.team_regimes)} teams")

    def _fit_player_hmm(self):
        """Fit 6-state player HMM on boxscore game logs."""
        conn = sqlite3.connect(str(self.db_path))
        games = pd.read_sql_query("""
            SELECT player_id, player_name, team, position, game_date,
                   goals, assists, shots, hits, blocked_shots, dk_fpts
            FROM boxscore_skaters
            WHERE toi_seconds > 0
            ORDER BY player_id, game_date
        """, conn)
        conn.close()

        games['game_date'] = pd.to_datetime(games['game_date'])
        self.season_avgs = games.groupby('player_id')['dk_fpts'].mean().to_dict()
        self.pid_to_name = (games.drop_duplicates('player_id')
                            .set_index('player_id')['player_name'].to_dict())

        # Build observation sequences
        all_obs, lengths, pids = [], [], []
        self._player_game_dates = {}

        for pid, group in games.groupby('player_id'):
            group = group.sort_values('game_date').reset_index(drop=True)
            if len(group) < MIN_GAMES:
                continue

            rows = []
            for i in range(WINDOW, len(group)):
                win = group.iloc[i-WINDOW:i]
                row = []
                for f in FEATURES:
                    vals = win[f].values.astype(float)
                    row.append(np.mean(vals))
                    row.append(np.std(vals))
                for f in FEATURES:
                    row.append(float(group[f].iloc[i]))
                rows.append(row)

            if len(rows) >= 5:
                X = np.nan_to_num(np.array(rows, dtype=np.float64))
                all_obs.append(X)
                lengths.append(len(X))
                pids.append(pid)
                self._player_game_dates[pid] = group['game_date'].iloc[-1]

        X = np.vstack(all_obs)
        mu, sig = X.mean(0), X.std(0)
        sig[sig < 1e-8] = 1.0
        X = (X - mu) / sig

        self._player_mu = mu
        self._player_sig = sig

        # PCA + HMM
        pca = PCA(n_components=min(N_PCA, X.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X)

        best, best_s = None, -np.inf
        for seed in range(6):
            m = GaussianHMM(n_components=N_STATES, covariance_type='diag',
                            n_iter=300, random_state=seed * 37 + 7, tol=0.01)
            try:
                m.fit(X_pca, lengths)
                s = m.score(X_pca, lengths)
                if s > best_s:
                    best, best_s = m, s
            except:
                pass

        self.player_model = best
        self.player_pca = pca

        # Decode states
        idx = 0
        for i, pid in enumerate(pids):
            seq = X_pca[idx:idx + lengths[i]]
            idx += lengths[i]
            try:
                _, ss = best.decode(seq, algorithm='viterbi')
                curr = int(ss[-1])
                prev = int(ss[-3]) if len(ss) >= 3 else curr
                prev2 = int(ss[-6]) if len(ss) >= 6 else prev
                self.player_states[pid] = {
                    'current': curr,
                    'prev': prev,
                    'prev2': prev2,
                    'is_downgrade': curr < prev,
                    'is_upgrade': curr > prev,
                    'is_transition': curr != prev,
                    'last_game': self._player_game_dates.get(pid),
                }
            except:
                pass

        print(f"  [HMM] Player model: {len(self.player_states)} players, "
              f"{N_STATES} states, LL={best_s:,.0f}")

    def _fit_team_hmm(self):
        """Fit 4-state team HMM on NST 5v5 process metrics."""
        files = sorted(self.nst_dir.glob('*_team_5v5.csv'))
        if not files:
            print("  [HMM] WARNING: No NST team files found")
            return

        team_data = defaultdict(list)
        for f in files:
            date = f.name[:10]
            try:
                df = pd.read_csv(f)
                df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
                avail = [c for c in TEAM_FEATURES if c in df.columns]
                for _, row in df.iterrows():
                    team = NST_TEAM_MAP.get(str(row.get('Team', '')).strip(),
                                             str(row.get('Team', '')).strip())
                    if not team:
                        continue
                    rec = {'date': date}
                    for c in avail:
                        try:
                            rec[c] = float(row[c])
                        except:
                            rec[c] = np.nan
                    team_data[team].append(rec)
            except:
                continue

        # Build observation vectors
        all_obs, lengths, keys = [], [], []
        for team in sorted(team_data.keys()):
            df = pd.DataFrame(team_data[team]).sort_values('date').reset_index(drop=True)
            avail = [c for c in TEAM_FEATURES if c in df.columns]
            if not avail or len(df) < 8:
                continue
            rows = []
            for i in range(3, len(df)):
                row = []
                for c in avail:
                    curr, prev = df[c].iloc[i], df[c].iloc[i - 3]
                    row.append((curr - prev) if pd.notna(curr) and pd.notna(prev) else 0.0)
                for c in avail:
                    v = df[c].iloc[i]
                    row.append(v if pd.notna(v) else 50.0)
                rows.append(row)
            if len(rows) >= 5:
                X = np.nan_to_num(np.array(rows, dtype=np.float64))
                all_obs.append(X)
                lengths.append(len(X))
                keys.append(team)

        if not all_obs:
            return

        X = np.vstack(all_obs)
        mu, sig = X.mean(0), X.std(0)
        sig[sig < 1e-8] = 1.0
        X = (X - mu) / sig

        pca = PCA(n_components=min(8, X.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X)

        best, best_s = None, -np.inf
        for seed in range(6):
            m = GaussianHMM(n_components=4, covariance_type='diag',
                            n_iter=300, random_state=seed * 37 + 7, tol=0.01)
            try:
                m.fit(X_pca, lengths)
                s = m.score(X_pca, lengths)
                if s > best_s:
                    best, best_s = m, s
            except:
                pass

        if best is None:
            return

        # Decode + classify team quality
        state_xgf = defaultdict(list)
        idx = 0
        for i, team in enumerate(keys):
            seq = X_pca[idx:idx + lengths[i]]
            idx += lengths[i]
            try:
                _, ss = best.decode(seq, algorithm='viterbi')
                state = int(ss[-1])
                self.team_regimes[team] = state
                df = pd.DataFrame(team_data[team]).sort_values('date')
                if 'xGF%' in df.columns:
                    state_xgf[state].append(df['xGF%'].iloc[-1])
            except:
                pass

        # Rank states by quality
        state_avg = {s: np.mean(v) for s, v in state_xgf.items()}
        sorted_states = sorted(state_avg.items(), key=lambda x: x[1])

        self.team_quality = {}
        n = len(sorted_states)
        for rank, (state, xgf) in enumerate(sorted_states):
            if rank < n / 3:
                self.team_quality[state] = 'WEAK'
            elif rank < 2 * n / 3:
                self.team_quality[state] = 'MID'
            else:
                self.team_quality[state] = 'STRONG'

        print(f"  [HMM] Team model: {len(self.team_regimes)} teams")
        for qual in ['WEAK', 'MID', 'STRONG']:
            teams = [t for t, s in self.team_regimes.items()
                     if self.team_quality.get(s) == qual]
            print(f"    {qual}: {', '.join(sorted(teams))}")

    # ───────────────────────────────────────────────────────────
    #  GET EDGES (call per player on slate)
    # ───────────────────────────────────────────────────────────

    def get_player_edge(self, player_name, opponent=None):
        """
        Get HMM edge for a single player.

        Args:
            player_name: abbreviated name like "N. MacKinnon" or full "Nathan MacKinnon"
            opponent: team abbreviation like "DET"

        Returns:
            dict with boost factor and reasons
        """
        if not self._fitted:
            self.fit()

        # Find player by name
        pid = None
        name_lower = player_name.lower().strip()
        for p, n in self.pid_to_name.items():
            if n.lower() == name_lower:
                pid = p
                break
        # Try abbreviated match
        if pid is None:
            def abbrev(name):
                parts = str(name).strip().split()
                if len(parts) >= 2:
                    return f"{parts[0][0]}. {' '.join(parts[1:])}".lower()
                return name.lower()
            for p, n in self.pid_to_name.items():
                if abbrev(n) == name_lower or n.lower() == name_lower:
                    pid = p
                    break

        if pid is None or pid not in self.player_states:
            return {'boost': 1.0, 'reasons': [], 'state': None}

        info = self.player_states[pid]
        boost = 1.0
        reasons = []
        is_downgrade = info['is_downgrade']
        is_transition = info['is_transition']

        # Determine opponent quality first
        opp_qual = 'MID'
        if opponent and opponent in self.team_regimes:
            opp_state = self.team_regimes[opponent]
            opp_qual = self.team_quality.get(opp_state, 'MID')

        # Signal logic with conflict resolution (from combined backtest)
        if is_downgrade and opp_qual == 'WEAK':
            # STACKED: synergy confirmed p=0.007, +0.87 FPTS
            boost = STACKED_DN_WEAK_BOOST
            reasons.append(f"⭐ STACKED S{info['prev']}→S{info['current']} + WEAK_OPP {opponent} (+6%)")
        elif is_downgrade and opp_qual == 'STRONG':
            # CONFLICT: matchup wins, net fade
            boost = CONFLICT_DN_STRONG_FADE
            reasons.append(f"⚠️ CONFLICT DN S{info['prev']}→S{info['current']} vs STRONG {opponent} (-3%)")
        elif is_downgrade:
            # Downgrade only (mid opponent)
            boost = DOWNGRADE_BOOST
            reasons.append(f"DOWNGRADE_BOUNCE S{info['prev']}→S{info['current']} (+3%)")
        elif is_transition:
            boost *= TRANSITION_BOOST
            direction = "UP" if info['is_upgrade'] else "DN"
            reasons.append(f"TRANSITION_{direction} S{info['prev']}→S{info['current']} (+1%)")

        # Non-downgrade opponent matchup
        if not is_downgrade:
            if opp_qual == 'WEAK':
                boost *= OPP_WEAK_BOOST
                reasons.append(f"WEAK_OPP {opponent} (+3%)")
            elif opp_qual == 'STRONG':
                boost *= OPP_STRONG_FADE
                reasons.append(f"STRONG_OPP {opponent} (-3%)")

        # Clamp
        boost = max(1 - MAX_HMM_SWING, min(1 + MAX_HMM_SWING, boost))

        return {
            'boost': round(boost, 4),
            'reasons': reasons,
            'state': info['current'],
            'prev_state': info['prev'],
            'is_downgrade': info['is_downgrade'],
            'is_upgrade': info['is_upgrade'],
            'is_stacked': is_downgrade and opp_qual == 'WEAK',
            'is_conflict': is_downgrade and opp_qual == 'STRONG',
            'opp_quality': opp_qual,
            'season_avg': self.season_avgs.get(pid, 0),
            'signal_type': ('STACKED' if is_downgrade and opp_qual == 'WEAK'
                          else 'CONFLICT' if is_downgrade and opp_qual == 'STRONG'
                          else 'DOWNGRADE' if is_downgrade
                          else 'WEAK_OPP' if opp_qual == 'WEAK'
                          else 'STRONG_OPP' if opp_qual == 'STRONG'
                          else 'TRANSITION' if is_transition
                          else 'NONE'),
        }

    def get_slate_edges(self, salary_df=None):
        """
        Get edges for all players on a slate.

        Args:
            salary_df: DK salary DataFrame with Name, TeamAbbrev, Salary columns
                       (or None for all players in DB)

        Returns:
            DataFrame with player edges
        """
        if not self._fitted:
            self.fit()

        rows = []

        if salary_df is not None:
            # Parse opponent from Game Info or Opp column
            for _, sal in salary_df.iterrows():
                name = str(sal.get('Name', sal.get('Player', '')))
                team = str(sal.get('TeamAbbrev', sal.get('Team', '')))
                salary = sal.get('Salary', 0)

                # Get opponent
                opp = None
                if 'Opp' in sal.index:
                    opp = str(sal['Opp']).replace('vs ', '').replace('@ ', '').strip()
                elif 'Game Info' in sal.index:
                    gi = str(sal['Game Info'])
                    teams = gi.split(' ')[0].split('@')
                    if len(teams) == 2:
                        opp = teams[0] if teams[1].startswith(team) else teams[1]

                edge = self.get_player_edge(name, opponent=opp)

                rows.append({
                    'Name': name,
                    'Team': team,
                    'Salary': salary,
                    'Opponent': opp,
                    'HMM_Boost': edge['boost'],
                    'HMM_State': edge['state'],
                    'HMM_Prev': edge.get('prev_state'),
                    'Downgrade': edge.get('is_downgrade', False),
                    'Upgrade': edge.get('is_upgrade', False),
                    'Reasons': ' | '.join(edge['reasons']),
                    'Season_Avg': edge.get('season_avg', 0),
                })
        else:
            # All players
            for pid, info in self.player_states.items():
                name = self.pid_to_name.get(pid, f'PID_{pid}')
                edge = self.get_player_edge(name)
                rows.append({
                    'Name': name,
                    'HMM_Boost': edge['boost'],
                    'HMM_State': edge['state'],
                    'HMM_Prev': edge.get('prev_state'),
                    'Downgrade': edge.get('is_downgrade', False),
                    'Upgrade': edge.get('is_upgrade', False),
                    'Reasons': ' | '.join(edge['reasons']),
                    'Season_Avg': edge.get('season_avg', 0),
                })

        df = pd.DataFrame(rows)
        return df

    def print_report(self, edges_df=None):
        """Print formatted edge report for the slate."""
        if edges_df is None:
            edges_df = self.get_slate_edges()

        has_edges = edges_df[edges_df['HMM_Boost'] != 1.0].copy()

        print(f"\n{'='*65}")
        print(f"  HMM EDGE REPORT — {datetime.now().strftime('%Y-%m-%d')}")
        print(f"  {len(has_edges)} players with active signals")
        print(f"{'='*65}")

        # Downgrades (main signal)
        downgrades = has_edges[has_edges['Downgrade']].sort_values('Season_Avg', ascending=False)
        if len(downgrades) > 0:
            print(f"\n  🔥 DOWNGRADE BOUNCE — Buy the dip ({len(downgrades)} players)")
            print(f"  {'─'*55}")
            for _, r in downgrades.head(20).iterrows():
                sal = f"${r['Salary']:,}" if pd.notna(r.get('Salary')) and r.get('Salary') else ''
                opp = r.get('Opponent', '')
                print(f"    {r['Name']:<22s} {r.get('Team',''):>4s} {sal:>7s} "
                      f"x{r['HMM_Boost']:.2f}  {r['Reasons']}")

        # Opponent matchup boosts (non-downgrade)
        opp_boosts = has_edges[(~has_edges['Downgrade']) &
                               (has_edges['Reasons'].str.contains('WEAK_OPP', na=False))]
        if len(opp_boosts) > 0:
            print(f"\n  📈 WEAK OPPONENT MATCHUP ({len(opp_boosts)} players)")
            print(f"  {'─'*55}")
            for _, r in opp_boosts.sort_values('Season_Avg', ascending=False).head(15).iterrows():
                sal = f"${r['Salary']:,}" if pd.notna(r.get('Salary')) and r.get('Salary') else ''
                print(f"    {r['Name']:<22s} {r.get('Team',''):>4s} {sal:>7s} "
                      f"x{r['HMM_Boost']:.2f}  {r['Reasons']}")

        # Strong opponent fades
        fades = has_edges[has_edges['Reasons'].str.contains('STRONG_OPP', na=False)]
        if len(fades) > 0:
            print(f"\n  ⚠️  STRONG OPPONENT FADE ({len(fades)} players)")
            print(f"  {'─'*55}")
            for _, r in fades.sort_values('Season_Avg', ascending=False).head(10).iterrows():
                sal = f"${r['Salary']:,}" if pd.notna(r.get('Salary')) and r.get('Salary') else ''
                print(f"    {r['Name']:<22s} {r.get('Team',''):>4s} {sal:>7s} "
                      f"x{r['HMM_Boost']:.2f}  {r['Reasons']}")

        # STACKED signals (downgrade + weak opponent)
        stacked = downgrades[downgrades['Reasons'].str.contains('WEAK_OPP', na=False)]
        if len(stacked) > 0:
            print(f"\n  ⭐ STACKED SIGNAL — Downgrade + Weak Opponent ({len(stacked)})")
            print(f"  {'─'*55}")
            for _, r in stacked.iterrows():
                sal = f"${r['Salary']:,}" if pd.notna(r.get('Salary')) and r.get('Salary') else ''
                print(f"    {r['Name']:<22s} {r.get('Team',''):>4s} {sal:>7s} "
                      f"x{r['HMM_Boost']:.2f}  *** TOP TARGET ***")


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION: called from projections.py
# ═══════════════════════════════════════════════════════════════

def get_hmm_boost(player_name, opponent=None, _model={}):
    """
    Quick function for projections.py integration.
    Fits model on first call, caches for subsequent calls.

    Usage in projections.py:
        from hmm_edge import get_hmm_boost
        hmm_boost = get_hmm_boost(row['player_name'], row.get('opponent'))
        combined_mult *= hmm_boost
    """
    if 'model' not in _model:
        m = HMMEdgeModel()
        m.fit()
        _model['model'] = m

    edge = _model['model'].get_player_edge(player_name, opponent)
    return edge['boost']


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HMM Edge Tool')
    parser.add_argument('--salary', type=str, help='DK salary CSV file')
    parser.add_argument('--csv', action='store_true', help='Export edges to CSV')
    parser.add_argument('--db', type=str, default=None, help='Path to SQLite DB')
    parser.add_argument('--nst', type=str, default=None, help='Path to NST daily dir')
    args = parser.parse_args()

    model = HMMEdgeModel(
        db_path=args.db if args.db else DB_PATH,
        nst_dir=args.nst if args.nst else NST_DIR,
    )
    model.fit()

    if args.salary:
        sal_df = pd.read_csv(args.salary)
        edges = model.get_slate_edges(sal_df)
    else:
        edges = model.get_slate_edges()

    model.print_report(edges)

    if args.csv:
        out_path = f"hmm_edges_{datetime.now().strftime('%Y-%m-%d')}.csv"
        edges.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")
