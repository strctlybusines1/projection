#!/usr/bin/env python3
"""
Goalie Context Model
=====================

Adjusts goalie projections using pre-game context factors that
the base projection model misses:

    1. Opponent scoring rate (team GPG)
    2. Rest days / back-to-back detection
    3. Recent form (rolling FPTS, GA, SV%)
    4. Home/away split
    5. Team defensive context (D-man availability proxy)
    6. Opponent shot volume tendency

Designed to address the persistent goalie MAE problem.

Key findings from game log analysis:
    - goals_against: r = -0.775 with goalie FPTS (dominant factor)
    - saves: r = +0.642 (more shots = more save points)
    - dmen +/-: r = +0.544 (defense quality matters)
    - opponent scoring rate: r = -0.382 (stronger opponents = worse)
    - rest days: >6 days rest → sharp decline (9.5 FPTS vs 16.5 at 3 days)
    - shots against bucket: light=7.6, normal=13.9, heavy=16.2, extreme=20.9

Integration:
    Called from projection_blender.py or main.py to adjust goalie projections.

Usage:
    from goalie_context import GoalieContextModel
    model = GoalieContextModel()
    model.fit()  # trains on game log DB
    player_pool = model.adjust_projections(player_pool, slate_date, opponent_map)
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"


class GoalieContextModel:
    """Adjusts goalie projections using pre-game context."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.fitted = False

        # Learned parameters
        self.opp_scoring_rates = {}     # team → avg goals per game
        self.opp_shot_rates = {}        # team → avg shots per game
        self.goalie_profiles = {}       # goalie_name → season stats
        self.league_avg_gpg = 3.0
        self.league_avg_sa = 28.0

        # Model coefficients (from ridge regression on game logs)
        self.coefficients = {}
        self.intercept = 0.0
        self.scaler_means = {}
        self.scaler_stds = {}

        # Penalty discipline
        self.team_penalty_rates = {}    # team → {drawn_per60, taken_per60, net_per60}
        self.league_avg_drawn = 3.67
        self.league_avg_taken = 3.65

    def fit(self) -> 'GoalieContextModel':
        """Train on game log database."""
        if not Path(self.db_path).exists():
            print("  ⚠ No game log database found")
            return self

        conn = sqlite3.connect(self.db_path)

        # ── Build opponent scoring rates ──
        sk = pd.read_sql_query("""
            SELECT team, game_id, SUM(goals) as team_goals, SUM(shots) as team_shots
            FROM game_logs_skaters
            GROUP BY team, game_id
        """, conn)

        if len(sk) > 0:
            team_stats = sk.groupby('team').agg(
                gpg=('team_goals', 'mean'),
                spg=('team_shots', 'mean'),
                games=('game_id', 'nunique'),
            )
            self.opp_scoring_rates = team_stats['gpg'].to_dict()
            self.opp_shot_rates = team_stats['spg'].to_dict()
            self.league_avg_gpg = sk['team_goals'].mean()
            self.league_avg_sa = sk['team_shots'].mean() if 'team_shots' in sk else 28.0

        # ── Load PP/PK special teams data ──
        self.team_pp = {}   # team → PP stats (how dangerous their PP is)
        self.team_pk = {}   # team → PK stats (how exposed their goalie is)
        self.league_avg_pp = 0.20
        self.league_avg_pk = 0.80
        self.league_avg_ppga = 0.60

        try:
            st = pd.read_sql_query("SELECT * FROM team_special_teams", conn)
            if len(st) > 0:
                for _, r in st.iterrows():
                    abbrev = r.get('team_abbrev', '')
                    if not abbrev:
                        continue
                    self.team_pp[abbrev] = {
                        'pp_pct': r.get('pp_pct', 0.20),
                        'pp_goals_pg': r.get('pp_goals_pg', 0.60),
                        'pp_opp_pg': r.get('pp_opp_pg', 3.0),
                    }
                    self.team_pk[abbrev] = {
                        'pk_pct': r.get('pk_pct', 0.80),
                        'ppga_pg': r.get('ppga_pg', 0.60),
                        'times_sh_pg': r.get('times_sh_pg', 3.0),
                        'goals_against_pg': r.get('goals_against_pg', 3.0),
                        'shots_against_pg': r.get('shots_against_pg', 28.0),
                    }
                self.league_avg_pp = st['pp_pct'].mean()
                self.league_avg_pk = st['pk_pct'].mean()
                self.league_avg_ppga = st['ppga_pg'].mean()

                # Load penalty discipline rates
                if 'pen_drawn_per60' in st.columns:
                    for _, r in st.iterrows():
                        abbrev = r.get('team_abbrev', '')
                        if abbrev:
                            self.team_penalty_rates[abbrev] = {
                                'drawn_per60': r.get('pen_drawn_per60', 3.67),
                                'taken_per60': r.get('pen_taken_per60', 3.65),
                                'net_per60': r.get('net_pen_per60', 0.0),
                            }
                    self.league_avg_drawn = st['pen_drawn_per60'].mean()
                    self.league_avg_taken = st['pen_taken_per60'].mean()
        except Exception:
            pass  # Table may not exist yet

        # ── Build goalie profiles ──
        g = pd.read_sql_query("""
            SELECT player_name, team, game_date, dk_fpts, goals_against,
                   saves, shots_against, save_pct, decision, home_road,
                   toi_seconds, games_started
            FROM game_logs_goalies
            WHERE games_started = 1 OR toi_seconds > 1800
            ORDER BY player_name, game_date
        """, conn)

        if len(g) > 0:
            for name, group in g.groupby('player_name'):
                group = group.sort_values('game_date')
                self.goalie_profiles[name] = {
                    'team': group.iloc[-1]['team'],
                    'games': len(group),
                    'avg_fpts': group['dk_fpts'].mean(),
                    'avg_ga': group['goals_against'].mean(),
                    'avg_saves': group['saves'].mean(),
                    'avg_sa': group['shots_against'].mean(),
                    'avg_sv_pct': group['save_pct'].mean(),
                    'win_rate': (group['decision'] == 'W').mean(),
                    'last_date': group.iloc[-1]['game_date'],
                    'last3_fpts': group['dk_fpts'].tail(3).mean(),
                    'last3_ga': group['goals_against'].tail(3).mean(),
                    'last5_fpts': group['dk_fpts'].tail(5).mean() if len(group) >= 5 else group['dk_fpts'].mean(),
                    'fpts_std': group['dk_fpts'].std(),
                    'home_fpts': group[group['home_road'] == 'H']['dk_fpts'].mean() if (group['home_road'] == 'H').any() else group['dk_fpts'].mean(),
                    'away_fpts': group[group['home_road'] == 'R']['dk_fpts'].mean() if (group['home_road'] == 'R').any() else group['dk_fpts'].mean(),
                }

            # ── Train regression model ──
            self._fit_regression(g, conn)

        conn.close()
        self.fitted = True
        return self

    def _fit_regression(self, g: pd.DataFrame, conn: sqlite3.Connection):
        """Fit ridge regression on pre-game features → goalie FPTS."""
        from sklearn.linear_model import Ridge

        g = g.sort_values(['player_name', 'game_date']).copy()

        # Build features
        records = []
        for name, group in g.groupby('player_name'):
            group = group.sort_values('game_date').reset_index(drop=True)

            for i in range(1, len(group)):
                row = group.iloc[i]
                prev = group.iloc[:i]

                # Rest days
                curr_date = pd.to_datetime(row['game_date'])
                prev_date = pd.to_datetime(group.iloc[i-1]['game_date'])
                rest = (curr_date - prev_date).days

                opp = row.get('opponent', '') if 'opponent' in row.index else ''

                records.append({
                    'fpts': row['dk_fpts'],
                    'opp_gpg': self.opp_scoring_rates.get(opp, self.league_avg_gpg),
                    'opp_spg': self.opp_shot_rates.get(opp, self.league_avg_sa),
                    'home': 1 if row['home_road'] == 'H' else 0,
                    'rest_days': min(rest, 10),
                    'is_rusty': 1 if rest >= 7 else 0,
                    'roll3_fpts': prev['dk_fpts'].tail(3).mean(),
                    'roll3_ga': prev['goals_against'].tail(3).mean(),
                    'season_avg': prev['dk_fpts'].mean(),
                    'season_ga': prev['goals_against'].mean(),
                    'game_num': i,
                })

        if len(records) < 20:
            return

        df = pd.DataFrame(records)
        feature_cols = ['opp_gpg', 'opp_spg', 'home', 'rest_days', 'is_rusty',
                       'roll3_fpts', 'roll3_ga', 'season_avg', 'season_ga']

        X = df[feature_cols].values
        y = df['fpts'].values

        # Standardize
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0] = 1
        Xs = (X - means) / stds

        self.scaler_means = dict(zip(feature_cols, means))
        self.scaler_stds = dict(zip(feature_cols, stds))

        ridge = Ridge(alpha=10.0)
        ridge.fit(Xs, y)

        self.coefficients = dict(zip(feature_cols, ridge.coef_))
        self.intercept = ridge.intercept_
        self.feature_cols = feature_cols

    def predict_fpts(self, goalie_name: str, opponent: str,
                     is_home: bool = True, slate_date: str = None) -> Dict:
        """
        Predict goalie FPTS using all available context.

        Returns dict with predicted_fpts, adjustment, factors breakdown.
        """
        result = {
            'predicted_fpts': None,
            'adjustment': 0.0,
            'factors': {},
            'confidence': 'low',
        }

        profile = self.goalie_profiles.get(goalie_name)
        if not profile:
            # Try fuzzy match
            for name, prof in self.goalie_profiles.items():
                if goalie_name.split()[-1].lower() in name.lower():
                    profile = prof
                    break

        if not profile:
            return result

        # ── Gather pre-game features ──
        opp_gpg = self.opp_scoring_rates.get(opponent, self.league_avg_gpg)
        opp_spg = self.opp_shot_rates.get(opponent, self.league_avg_sa)

        # Rest days
        rest_days = 3  # default
        if slate_date and profile.get('last_date'):
            try:
                slate_dt = pd.to_datetime(slate_date)
                last_dt = pd.to_datetime(profile['last_date'])
                rest_days = min((slate_dt - last_dt).days, 10)
            except:
                pass

        is_rusty = 1 if rest_days >= 7 else 0

        features = {
            'opp_gpg': opp_gpg,
            'opp_spg': opp_spg,
            'home': 1 if is_home else 0,
            'rest_days': rest_days,
            'is_rusty': is_rusty,
            'roll3_fpts': profile.get('last3_fpts', profile['avg_fpts']),
            'roll3_ga': profile.get('last3_ga', profile['avg_ga']),
            'season_avg': profile['avg_fpts'],
            'season_ga': profile['avg_ga'],
        }

        # ── Use regression model if fitted ──
        if self.coefficients:
            X = np.array([[features[f] for f in self.feature_cols]])
            Xs = np.array([[(features[f] - self.scaler_means[f]) / self.scaler_stds[f]
                           for f in self.feature_cols]])
            predicted = float(self.intercept + Xs @ np.array([self.coefficients[f] for f in self.feature_cols]))
            predicted = max(0, predicted)

            result['predicted_fpts'] = round(predicted, 1)
            result['confidence'] = 'high' if profile['games'] >= 15 else 'medium'
        else:
            # Fallback: rule-based adjustments
            predicted = profile['avg_fpts']
            result['predicted_fpts'] = round(predicted, 1)

        # ── Calculate adjustment from baseline ──
        baseline = profile['avg_fpts']
        adjustment = 0.0

        # Factor 1: Opponent strength (even strength)
        opp_factor = (opp_gpg - self.league_avg_gpg) * -2.0
        adjustment += opp_factor
        result['factors']['opponent_es_strength'] = round(opp_factor, 1)

        # Factor 2: PP/PK matchup — opponent PP vs goalie's team PK
        pp_pk_factor = 0.0
        opp_pp = self.team_pp.get(opponent, {})
        team_pk = self.team_pk.get(profile.get('team', ''), {})

        if opp_pp and team_pk:
            # Opponent PP danger: how much above/below avg their PP is
            opp_pp_pct = opp_pp.get('pp_pct', self.league_avg_pp)
            opp_pp_gpg = opp_pp.get('pp_goals_pg', 0.60)

            # Team PK exposure: how many PPG against per game
            team_ppga = team_pk.get('ppga_pg', self.league_avg_ppga)
            team_pk_pct = team_pk.get('pk_pct', self.league_avg_pk)

            # Combined: strong opposing PP + weak team PK = danger
            pp_above_avg = (opp_pp_pct - self.league_avg_pp) * 100  # in percentage points
            pk_below_avg = (self.league_avg_pk - team_pk_pct) * 100

            # PP danger: opponent PP strength effect
            pp_pk_factor = pp_above_avg * -0.10  # each 1% better PP = -0.1 FPTS

            # PK exposure: own team PK weakness compounds it
            if pk_below_avg > 3:  # team PK is significantly worse than average
                pp_pk_factor -= pk_below_avg * 0.08  # each 1% worse PK = -0.08 FPTS extra

            # ── Penalty discipline matchup multiplier ──
            # When a team that draws lots of penalties faces a team that takes lots,
            # expect more PPs than season average → amplify PP/PK effect
            opp_drawn = self.team_penalty_rates.get(opponent, {}).get('drawn_per60', self.league_avg_drawn)
            team_taken = self.team_penalty_rates.get(profile.get('team', ''), {}).get('taken_per60', self.league_avg_taken)

            # How many more/fewer PPs than average this matchup should produce
            # Opponent draws above avg + goalie's team takes above avg = more PPs for opponent
            opp_draw_factor = opp_drawn / max(self.league_avg_drawn, 0.01)  # >1 = draws more than avg
            team_take_factor = team_taken / max(self.league_avg_taken, 0.01)  # >1 = takes more than avg

            # Combined: geometric mean of both factors
            penalty_multiplier = (opp_draw_factor * team_take_factor) ** 0.5
            # Clamp to reasonable range: 0.75x to 1.35x
            penalty_multiplier = max(0.75, min(1.35, penalty_multiplier))

            # Apply multiplier to the PP/PK factor
            pp_pk_factor *= penalty_multiplier

            pp_pk_factor = max(-3.5, min(2.5, pp_pk_factor))
            adjustment += pp_pk_factor

            result['factors']['opp_pp_strength'] = round(pp_above_avg * -0.10, 1)
            result['factors']['team_pk_weakness'] = round(
                -pk_below_avg * 0.08 if pk_below_avg > 3 else 0.0, 1)
            result['factors']['penalty_multiplier'] = round(penalty_multiplier, 2)
        result['factors']['pp_pk_matchup'] = round(pp_pk_factor, 1)

        # Factor 2: Rest/rustiness
        if rest_days >= 7:
            rust_penalty = -2.0 - (rest_days - 7) * 0.5
            adjustment += rust_penalty
            result['factors']['rust_penalty'] = round(rust_penalty, 1)
        elif rest_days <= 1:
            b2b_penalty = -1.0
            adjustment += b2b_penalty
            result['factors']['b2b_penalty'] = round(b2b_penalty, 1)

        # Factor 3: Recent form vs season average
        form_delta = profile.get('last3_fpts', baseline) - baseline
        form_adj = form_delta * 0.3  # 30% weight on recent form deviation
        adjustment += form_adj
        result['factors']['recent_form'] = round(form_adj, 1)

        # Factor 4: Home/away
        if is_home:
            ha_adj = (profile.get('home_fpts', baseline) - baseline) * 0.2
        else:
            ha_adj = (profile.get('away_fpts', baseline) - baseline) * 0.2
        adjustment += ha_adj
        result['factors']['home_away'] = round(ha_adj, 1)

        # Factor 5: Opponent shot volume (more shots = more save opportunities)
        shot_factor = (opp_spg - self.league_avg_sa) * 0.15
        adjustment += shot_factor
        result['factors']['opp_shot_volume'] = round(shot_factor, 1)

        result['adjustment'] = round(adjustment, 1)
        result['factors']['total'] = round(adjustment, 1)

        return result

    def adjust_projections(self, player_pool: pd.DataFrame,
                           slate_date: str = None,
                           opponent_map: Dict[str, str] = None,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Adjust goalie projections in a player pool.

        Args:
            player_pool: DataFrame with projections
            slate_date: Date string (YYYY-MM-DD)
            opponent_map: {team: opponent} mapping. If None, tries to infer.
        """
        if not self.fitted:
            self.fit()

        df = player_pool.copy()
        goalie_mask = df['position'] == 'G'
        goalies = df[goalie_mask].copy()

        if len(goalies) == 0:
            return df

        adjustments = []
        for idx, row in goalies.iterrows():
            name = row['name']
            team = row['team']
            opponent = opponent_map.get(team, '') if opponent_map else ''

            # Try to determine home/away
            is_home = True  # default

            result = self.predict_fpts(name, opponent, is_home, slate_date)
            adjustments.append({
                'idx': idx,
                'name': name,
                'team': team,
                'opponent': opponent,
                'adjustment': result['adjustment'],
                'predicted': result.get('predicted_fpts'),
                'factors': result['factors'],
                'confidence': result['confidence'],
            })

        if verbose and adjustments:
            print(f"\n  ── Goalie Context Adjustments ─────────────────")
            print(f"  {'Name':<22} {'Team':>4} {'Opp':>4} {'Adj':>5} {'Factors'}")
            print(f"  {'─' * 65}")
            for a in adjustments:
                factors_str = ', '.join(f"{k}={v:+.1f}" for k, v in a['factors'].items()
                                       if k != 'total' and abs(v) > 0.1)
                print(f"  {a['name']:<22} {a['team']:>4} {a['opponent']:>4} "
                      f"{a['adjustment']:>+5.1f} {factors_str}")

        # Apply adjustments
        for a in adjustments:
            current = df.loc[a['idx'], 'projected_fpts']
            adjusted = current + a['adjustment']
            df.loc[a['idx'], 'projected_fpts'] = max(0.5, adjusted)
            df.loc[a['idx'], 'goalie_context_adj'] = a['adjustment']

        # Update derived columns
        if 'salary' in df.columns:
            df.loc[goalie_mask, 'value'] = (
                df.loc[goalie_mask, 'projected_fpts'] / (df.loc[goalie_mask, 'salary'] / 1000)
            ).round(3)
        if 'dk_avg_fpts' in df.columns:
            df.loc[goalie_mask, 'edge'] = (
                df.loc[goalie_mask, 'projected_fpts'] - df.loc[goalie_mask, 'dk_avg_fpts']
            ).round(3)

        return df

    def print_goalie_report(self):
        """Print summary of all goalie profiles."""
        if not self.goalie_profiles:
            self.fit()

        print(f"\n  ── Goalie Profiles ({len(self.goalie_profiles)} goalies) ──")
        print(f"  {'Name':<24} {'Team':>4} {'GP':>3} {'Avg':>5} {'L3':>5} {'GA':>4} "
              f"{'SV%':>5} {'W%':>4} {'Std':>4}")
        print(f"  {'─' * 65}")

        sorted_goalies = sorted(self.goalie_profiles.items(),
                               key=lambda x: x[1]['avg_fpts'], reverse=True)
        for name, p in sorted_goalies:
            print(f"  {name:<24} {p['team']:>4} {p['games']:>3} {p['avg_fpts']:>5.1f} "
                  f"{p['last3_fpts']:>5.1f} {p['avg_ga']:>4.1f} "
                  f"{p['avg_sv_pct']:>5.3f} {p['win_rate']:>3.0%} {p['fpts_std']:>4.1f}")


# ================================================================
#  CLI
# ================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Goalie Context Model')
    parser.add_argument('--report', action='store_true', help='Print goalie profiles')
    parser.add_argument('--predict', type=str, help='Predict for a goalie (name)')
    parser.add_argument('--opponent', type=str, default='', help='Opponent team')
    parser.add_argument('--date', type=str, default=None, help='Slate date')
    args = parser.parse_args()

    model = GoalieContextModel()
    model.fit()

    if args.report:
        model.print_goalie_report()
    elif args.predict:
        result = model.predict_fpts(args.predict, args.opponent,
                                     slate_date=args.date)
        print(f"\n  {args.predict} vs {args.opponent or '???'}:")
        print(f"  Predicted: {result['predicted_fpts']} FPTS")
        print(f"  Adjustment: {result['adjustment']:+.1f}")
        for k, v in result['factors'].items():
            if k != 'total':
                print(f"    {k}: {v:+.1f}")
    else:
        model.print_goalie_report()
