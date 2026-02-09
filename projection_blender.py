#!/usr/bin/env python3
"""
Projection Blender v2 — Current + Bayesian + HMM + ESN
========================================================

Four-signal blend:
    1. Current model (rate-based projections)
    2. Bayesian event probabilities (P(goal), P(assist), etc.)
    3. HMM regime adjustment (hot/cold/neutral state from recent games)
    4. ESN time-series prediction (echo state network on game sequences)

Integration:
    Called from main.py after generate_projections() returns.

Usage:
    from projection_blender import blend_projections
    player_pool = blend_projections(player_pool, vegas_data, date_str)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parent

# ================================================================
#  Blend Parameters
# ================================================================

CURRENT_WEIGHT = 0.50
BAYESIAN_WEIGHT = 0.50

# Goalie-specific: lean toward current model (Bayesian is too generic for goalies)
GOALIE_CURRENT_WEIGHT = 0.80
GOALIE_BAYESIAN_WEIGHT = 0.20

# When ESN has a prediction, how much to blend it in
ESN_BLEND_WEIGHT = 0.20
ESN_GOALIE_BLEND_WEIGHT = 0.30  # Goalies get more ESN weight (captures recent form)

# HMM adjustment caps
HMM_MAX_BOOST = 2.0
HMM_MAX_PENALTY = -1.5
HMM_GOALIE_SCALE = 1.5

# Bias corrections
SKATER_BIAS_SHIFT = -2.50
GOALIE_BIAS_SHIFT = 0.0  # Goalies are bimodal — flat bias correction hurts

MIN_PROJECTION = 0.5
HMM_MIN_GAMES = 2
ESN_MIN_GAMES = 3


# ================================================================
#  HMM State Detector
# ================================================================

class PlayerStateDetector:
    """Detects hot/cold/neutral states from recent game sequences."""

    def __init__(self):
        self.skater_hmm = None
        self.goalie_hmm = None
        self.actuals = None

    def fit(self, actuals: pd.DataFrame):
        self.actuals = actuals.copy()
        try:
            from advanced_models import BayesianHMM
        except ImportError:
            return self

        sk = actuals[actuals['position'] != 'G']
        sk_seqs = self._build_sequences(sk)
        if len(sk_seqs) >= 5:
            self.skater_hmm = BayesianHMM(n_states=3)
            try:
                self.skater_hmm.fit(sk_seqs)
            except Exception:
                self.skater_hmm = None

        g = actuals[actuals['position'] == 'G']
        g_seqs = self._build_sequences(g)
        if len(g_seqs) >= 3:
            self.goalie_hmm = BayesianHMM(n_states=2)
            self.goalie_hmm.state_labels = ['BAD', 'GOOD']
            try:
                self.goalie_hmm.fit(g_seqs)
            except Exception:
                self.goalie_hmm = None

        return self

    def get_adjustment(self, player_name: str, team: str, position: str) -> Dict:
        default = {'adjustment': 0.0, 'state': 'UNKNOWN', 'confidence': 0.0}
        if self.actuals is None:
            return default

        def ln(n): return str(n).strip().split()[-1].lower()
        mask = self.actuals['name'].apply(ln) == ln(player_name)
        mask &= self.actuals['team'].str.lower() == team.lower()
        player_games = self.actuals[mask].sort_values('date')

        if len(player_games) < HMM_MIN_GAMES:
            return default

        recent = player_games['actual_fpts'].values
        hmm = self.goalie_hmm if position == 'G' else self.skater_hmm
        if hmm is None or hmm.model is None:
            return default

        try:
            result = hmm.predict_state(recent)
        except Exception:
            return default

        adj = result.get('expected_adjustment', 0.0)
        scale = HMM_GOALIE_SCALE if position == 'G' else 1.0
        adj_capped = np.clip(adj * scale, HMM_MAX_PENALTY, HMM_MAX_BOOST)
        confidence = min(1.0, len(recent) / 5.0)

        return {
            'adjustment': float(round(adj_capped * confidence, 2)),
            'state': result.get('state', 'NEUTRAL'),
            'confidence': float(round(confidence, 2)),
            'hot_prob': result.get('hot_prob', 0.33),
            'cold_prob': result.get('cold_prob', 0.33),
        }

    @staticmethod
    def _build_sequences(df, min_games=2):
        seqs = []
        for (name, team), group in df.groupby(['name', 'team']):
            games = group.sort_values('date')['actual_fpts'].values
            if len(games) >= min_games:
                seqs.append(games)
        return seqs


# ================================================================
#  ESN Predictor
# ================================================================

class ESNPredictor:
    """Echo State Network for per-player next-game FPTS prediction."""

    def __init__(self):
        self.esn = None
        self._cache = {}

    def fit_and_predict(self, actuals: pd.DataFrame) -> Dict[str, float]:
        self._cache = {}
        try:
            from advanced_models import EchoStateNetwork
        except ImportError:
            return self._cache

        X_seqs, y_seqs, keys = [], [], []
        def ln(n): return str(n).strip().split()[-1].lower()

        for (name, team), group in actuals.groupby(['name', 'team']):
            games = group.sort_values('date')['actual_fpts'].values
            if len(games) < ESN_MIN_GAMES:
                continue
            X = np.column_stack([
                games[:-1],
                np.arange(len(games) - 1, dtype=float) / max(len(games) - 1, 1),
            ])
            X_seqs.append(X)
            y_seqs.append(games[1:])
            keys.append(f"{ln(name)}_{team.lower()}")

        if not X_seqs:
            return self._cache

        esn = EchoStateNetwork(
            reservoir_size=150, spectral_radius=0.92,
            input_scaling=0.4, leak_rate=0.3, ridge_alpha=1.0,
        )
        try:
            esn.fit(X_seqs, y_seqs)
            self.esn = esn
        except Exception:
            return self._cache

        for key, X, y in zip(keys, X_seqs, y_seqs):
            try:
                next_input = np.array([[y[-1], len(y) / max(len(y), 1)]])
                full_X = np.vstack([X, next_input])
                preds = esn.predict(full_X)
                self._cache[key] = round(max(0, float(preds[-1])), 2)
            except Exception:
                continue

        return self._cache

    def get_prediction(self, player_name: str, team: str) -> Optional[float]:
        def ln(n): return str(n).strip().split()[-1].lower()
        return self._cache.get(f"{ln(player_name)}_{team.lower()}")


# ================================================================
#  Main Blender
# ================================================================

def blend_projections(player_pool: pd.DataFrame,
                      vegas: pd.DataFrame = None,
                      date_str: str = None,
                      replace: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """Four-signal projection blend: Current + Bayesian + HMM + ESN."""
    try:
        from bayesian_projections import BayesianProjector
    except ImportError:
        if verbose:
            print("  ⚠ bayesian_projections.py not found — skipping blend")
        return player_pool

    df = player_pool.copy()
    bp = BayesianProjector()

    # ── Signal 1 & 2: Current + Bayesian ──
    result = bp.project_player_pool(df, vegas, date_str)
    if 'bayes_expected_fpts' not in result.columns:
        if verbose:
            print("  ⚠ Bayesian projections failed — using current model only")
        return player_pool

    df['current_fpts'] = result['projected_fpts']
    df['bayes_fpts'] = result['bayes_expected_fpts']

    for col in ['bayes_floor', 'bayes_ceiling', 'bayes_std',
                'bayes_p_goal', 'bayes_p_assist', 'bayes_p_five_sog',
                'bayes_p_three_blocks', 'bayes_p_shutout', 'bayes_p_win',
                'bayes_median_fpts', 'bayes_p5', 'bayes_p95']:
        if col in result.columns:
            df[col] = result[col]

    # ── Signal 3: HMM ──
    hmm_adj = np.zeros(len(df))
    hmm_states = ['N/A'] * len(df)
    hmm_active = 0

    actuals = _load_actuals()
    if actuals is not None and len(actuals) > 0:
        try:
            detector = PlayerStateDetector()
            detector.fit(actuals)
            for i, (_, row) in enumerate(df.iterrows()):
                r = detector.get_adjustment(row['name'], row['team'], row['position'])
                hmm_adj[i] = r['adjustment']
                hmm_states[i] = r['state']
                if r['state'] != 'UNKNOWN':
                    hmm_active += 1
        except Exception as e:
            if verbose:
                print(f"  ⚠ HMM: {e}")

    df['hmm_adjustment'] = hmm_adj
    df['hmm_state'] = hmm_states

    # ── Signal 4: ESN ──
    esn_preds = np.full(len(df), np.nan)
    esn_active = 0

    if actuals is not None and len(actuals) > 0:
        try:
            esn = ESNPredictor()
            esn.fit_and_predict(actuals)
            for i, (_, row) in enumerate(df.iterrows()):
                pred = esn.get_prediction(row['name'], row['team'])
                if pred is not None:
                    esn_preds[i] = pred
                    esn_active += 1
        except Exception as e:
            if verbose:
                print(f"  ⚠ ESN: {e}")

    df['esn_fpts'] = esn_preds

    # ── Combine ──
    skater_mask = df['position'] != 'G'
    goalie_mask = df['position'] == 'G'

    blended = np.zeros(len(df))
    for i in range(len(df)):
        cur = df.iloc[i]['current_fpts']
        bay = df.iloc[i]['bayes_fpts']
        esn_val = esn_preds[i]
        hmm_val = hmm_adj[i]
        is_goalie = df.iloc[i]['position'] == 'G'

        has_esn = not np.isnan(esn_val) and esn_val > 0

        # Use position-specific weights
        w_cur = GOALIE_CURRENT_WEIGHT if is_goalie else CURRENT_WEIGHT
        w_bay = GOALIE_BAYESIAN_WEIGHT if is_goalie else BAYESIAN_WEIGHT
        w_esn = ESN_GOALIE_BLEND_WEIGHT if is_goalie else ESN_BLEND_WEIGHT

        if has_esn:
            base = (1.0 - w_esn) * (w_cur * cur + w_bay * bay)
            blended[i] = base + w_esn * esn_val
        else:
            blended[i] = w_cur * cur + w_bay * bay

        blended[i] += hmm_val

    blended[skater_mask.values] += SKATER_BIAS_SHIFT
    blended[goalie_mask.values] += GOALIE_BIAS_SHIFT
    blended = np.clip(blended, MIN_PROJECTION, None)

    df['blended_fpts'] = np.round(blended, 2)

    if replace:
        df['projected_fpts'] = df['blended_fpts']
        if 'salary' in df.columns:
            df['value'] = (df['projected_fpts'] / (df['salary'] / 1000)).round(3)
        if 'dk_avg_fpts' in df.columns:
            df['edge'] = (df['projected_fpts'] - df['dk_avg_fpts']).round(3)
        if 'bayes_floor' in df.columns:
            df['floor'] = df['bayes_floor']
        if 'bayes_ceiling' in df.columns:
            df['ceiling'] = df['bayes_ceiling']

    if verbose:
        n_sk = skater_mask.sum()
        n_g = goalie_mask.sum()
        sk_cur = df.loc[skater_mask, 'current_fpts'].mean()
        sk_bay = df.loc[skater_mask, 'bayes_fpts'].mean()
        sk_bld = df.loc[skater_mask, 'blended_fpts'].mean()

        print(f"\n  ── Projection Blend v2 (4-signal) ────────────────")
        print(f"  Base: {CURRENT_WEIGHT:.0%} current / {BAYESIAN_WEIGHT:.0%} Bayesian")
        print(f"  ESN: {ESN_BLEND_WEIGHT:.0%} weight ({esn_active} players with history)")
        print(f"  HMM: {hmm_active} players with state detection")
        print(f"  Bias: skaters {SKATER_BIAS_SHIFT:+.1f}, goalies {GOALIE_BIAS_SHIFT:+.1f}")
        print(f"  Players: {n_sk} skaters, {n_g} goalies")
        print(f"  Skater avg — Cur: {sk_cur:.1f}  Bay: {sk_bay:.1f}  Bld: {sk_bld:.1f}")

        if n_g > 0:
            g_cur = df.loc[goalie_mask, 'current_fpts'].mean()
            g_bay = df.loc[goalie_mask, 'bayes_fpts'].mean()
            g_bld = df.loc[goalie_mask, 'blended_fpts'].mean()
            print(f"  Goalie avg — Cur: {g_cur:.1f}  Bay: {g_bay:.1f}  Bld: {g_bld:.1f}")

        top = df.nlargest(12, 'blended_fpts')
        print(f"\n  {'Name':<22} {'Cur':>5} {'Bay':>5} {'ESN':>5} {'HMM':>5} {'Bld':>5} {'State':>8}")
        print(f"  {'─' * 60}")
        for _, p in top.iterrows():
            esn_s = f"{p['esn_fpts']:.1f}" if pd.notna(p['esn_fpts']) else "  —"
            hmm_s = f"{p['hmm_adjustment']:+.1f}" if p['hmm_state'] != 'N/A' else "  —"
            print(f"  {p['name']:<22} {p['current_fpts']:>5.1f} {p['bayes_fpts']:>5.1f} "
                  f"{esn_s:>5} {hmm_s:>5} {p['blended_fpts']:>5.1f} {p['hmm_state']:>8}")

        state_counts = df['hmm_state'].value_counts()
        active_states = {k: v for k, v in state_counts.items() if k not in ('N/A', 'UNKNOWN')}
        if active_states:
            print(f"\n  HMM States: {active_states}")
        print()

    return df


def _load_actuals() -> Optional[pd.DataFrame]:
    path = PROJECT_ROOT / 'backtests' / 'batch_backtest_details.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


def recalibrate(actuals_csv='backtests/batch_backtest_details.csv',
                vegas_csv='Vegas_Historical.csv', n_recent=10):
    """Re-optimize blend weights from latest backtests."""
    from datetime import datetime
    from bayesian_projections import BayesianProjector

    proj_dir = PROJECT_ROOT / 'daily_projections'
    actuals = pd.read_csv(PROJECT_ROOT / actuals_csv)
    vegas = None
    vpath = PROJECT_ROOT / vegas_csv
    if vpath.exists():
        vdf = pd.read_csv(vpath, encoding='utf-8-sig')
        vdf['date'] = vdf['Date'].apply(
            lambda d: f"20{d.split('.')[2]}-{int(d.split('.')[0]):02d}-{int(d.split('.')[1]):02d}"
        )
        vegas = vdf

    bp = BayesianProjector()
    def ln(n): return n.strip().split()[-1].lower()
    dates = sorted(actuals['date'].unique())[-n_recent:]
    all_records = []

    for date_str in dates:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        prefix = f'{dt.month:02d}_{dt.day:02d}_{dt.strftime("%y")}'
        proj_file = None
        for f in sorted(proj_dir.glob('*NHLprojections_*.csv')):
            if '_lineups' in f.name: continue
            if f.name.startswith(prefix): proj_file = f
        if not proj_file: continue
        pool = pd.read_csv(proj_file)
        result = bp.project_player_pool(pool, vegas, date_str)
        act_date = actuals[actuals['date'] == date_str].copy()
        if act_date.empty: continue
        act_date['_key'] = act_date['name'].apply(ln) + '_' + act_date['team'].str.lower()
        result['_key'] = result['name'].apply(ln) + '_' + result['team'].str.lower()
        merged = act_date.merge(
            result[['_key', 'bayes_expected_fpts', 'projected_fpts']].drop_duplicates('_key'),
            on='_key', how='inner', suffixes=('_actual', '_proj')
        )
        if not merged.empty:
            merged['date'] = date_str
            all_records.append(merged)

    if not all_records:
        print("No data found"); return
    combined = pd.concat(all_records, ignore_index=True)
    print(f"Recalibrating on {len(combined)} obs across {len(all_records)} slates")

    best_mae, best_params = 999, {}
    for w in np.arange(0, 1.01, 0.05):
        for shift in np.arange(-5.0, 1.0, 0.25):
            bl = w * combined['projected_fpts_proj'] + (1-w) * combined['bayes_expected_fpts']
            sk = combined['position'] != 'G'
            adj = bl.copy(); adj[sk] += shift; adj = adj.clip(lower=0.5)
            mae = (adj - combined['actual_fpts']).abs().mean()
            if mae < best_mae:
                best_mae = mae; best_params = {'w': w, 'shift': shift}

    p = best_params
    print(f"\n  OPTIMAL: CURRENT_WEIGHT={p['w']:.2f} SKATER_BIAS_SHIFT={p['shift']:.2f} MAE={best_mae:.3f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--recalibrate', action='store_true')
    parser.add_argument('--recent', type=int, default=10)
    args = parser.parse_args()
    if args.recalibrate:
        recalibrate(n_recent=args.recent)
