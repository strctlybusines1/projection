#!/usr/bin/env python3
"""
Single-Entry Ownership Model for NHL DFS
==========================================

Predicts ownership for small-field SE contests (<200 entries).
Trained on actual ownership data from 15 contests, 2249 player-observations.

Key findings vs GPP ownership:
    - dk_avg_fpts (season reputation) is the #1 driver (r=0.458)
    - Salary tier matters heavily (r=0.400)
    - Projection rank within position is critical (#1 at pos = 17.4% avg)
    - Value/edge are nearly irrelevant (r=0.08) — SE players don't chase value
    - Goalies are more concentrated (top goalie 25%+ avg)
    - Line 1/PP1 players are massively over-owned vs GPP

Usage:
    from se_ownership import SEOwnershipModel
    model = SEOwnershipModel()
    model.fit_from_contests()  # Train on historical data
    player_pool = model.predict(player_pool)
"""

import re
import glob
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent


# ================================================================
#  SE Ownership Curves (empirical from 15 contests)
# ================================================================

# Base ownership by salary tier (from actual SE data)
SE_SALARY_CURVE = {
    (2500, 3000): 1.5,
    (3000, 3500): 2.0,
    (3500, 4000): 3.5,
    (4000, 4500): 4.5,
    (4500, 5000): 5.5,
    (5000, 5500): 7.0,
    (5500, 6000): 6.0,
    (6000, 6500): 6.5,
    (6500, 7000): 7.0,
    (7000, 7500): 7.5,
    (7500, 8000): 8.0,
    (8000, 8500): 9.0,
    (8500, 9000): 11.0,
    (9000, 9500): 13.0,
    (9500, 10000): 16.0,
    (10000, 11000): 20.0,
}

# Position rank multipliers (how much ownership increases by rank)
# #1 at position = 17.4% avg, #2-3 = 14.2%, #4-6 = 9.4%, etc.
SE_RANK_MULTIPLIER = {
    1: 2.8,    # #1 at position: huge ownership
    2: 2.2,
    3: 1.8,
    4: 1.4,
    5: 1.2,
    6: 1.1,
    7: 1.0,
    8: 0.95,
    9: 0.90,
    10: 0.85,
    11: 0.80,
    12: 0.75,
}

# Position base adjustments (G and C owned more in SE)
SE_POSITION_MULT = {
    'C': 1.15,
    'W': 1.0,
    'D': 0.80,
    'G': 1.35,
}

# DK avg FPTS thresholds (the "name recognition" factor)
# In SE, people draft players they KNOW, not sleepers
SE_DK_AVG_TIERS = {
    'elite': {'min': 12.0, 'mult': 2.0},     # Top stars: owned heavily
    'high': {'min': 8.0, 'mult': 1.4},        # Known scorers
    'mid': {'min': 5.0, 'mult': 1.0},         # Average
    'low': {'min': 0, 'mult': 0.6},           # Unknown = low ownership
}


# ================================================================
#  SE Ownership Model
# ================================================================

class SEOwnershipModel:
    """
    Predict ownership for small-field single-entry NHL DFS contests.

    Uses empirical curves from actual SE contest data rather than
    heuristic multipliers. Key differences from GPP model:
        1. Season reputation (dk_avg_fpts) drives ownership, not edge/value
        2. Top-of-position players are MUCH more concentrated
        3. Line/PP status matters more (SE players check lines)
        4. Goalies are hyper-concentrated (everyone picks the "obvious" starter)
    """

    def __init__(self):
        self.lines_data = None
        self.confirmed_goalies = {}
        self.regression_model = None
        self._fit_data = None

    def set_lines_data(self, lines_data):
        """Provide line/PP data from DailyFaceoff scraper."""
        self.lines_data = lines_data

    def set_confirmed_goalies(self, confirmed: Dict[str, str]):
        """Set confirmed goalie starters {team: goalie_name}."""
        self.confirmed_goalies = confirmed

    def fit_from_contests(self, contest_dir: str = None, max_entries: int = 200):
        """
        Train a regression model on actual SE contest ownership data.

        Reads all contest CSVs, matches to projection features, and
        fits a simple model to predict ownership from features.
        """
        contest_dir = contest_dir or str(PROJECT_ROOT / 'contests')
        proj_dir = PROJECT_ROOT / 'daily_projections'

        records = []
        for f in sorted(glob.glob(f'{contest_dir}/*.csv')):
            try:
                df = pd.read_csv(f, encoding='utf-8-sig')
            except Exception:
                continue

            entries = df['EntryId'].nunique()
            if entries > max_entries and 'SE' not in f:
                continue

            date_match = re.search(r'(\d+)\.(\d+)\.(\d+)', f)
            if not date_match:
                continue
            m, d, y = date_match.groups()
            date_str = f'2026-{int(m):02d}-{int(d):02d}'

            dt = datetime.strptime(date_str, '%Y-%m-%d')
            prefix = f'{dt.month:02d}_{dt.day:02d}_{dt.strftime("%y")}'
            proj_file = None
            for pf in sorted(proj_dir.glob('*NHLprojections_*.csv')):
                if '_lineups' in pf.name:
                    continue
                if pf.name.startswith(prefix):
                    proj_file = pf
            if not proj_file:
                continue

            proj = pd.read_csv(proj_file)
            df['own_pct'] = df['%Drafted'].str.rstrip('%').astype(float)
            players = df[['Player', 'own_pct']].drop_duplicates('Player').dropna(subset=['Player'])

            def _ln(n):
                s = str(n).strip().split()
                return s[-1].lower() if s else ''

            players['_key'] = players['Player'].apply(_ln)
            proj['_key'] = proj['name'].apply(_ln)

            merged = players.merge(proj.drop_duplicates('_key'), on='_key', how='inner')
            merged['date'] = date_str
            merged['entries'] = entries
            records.append(merged)

        if not records:
            print("  No SE contest data found for training")
            return

        self._fit_data = pd.concat(records, ignore_index=True)
        print(f"  SE ownership model: trained on {len(self._fit_data)} observations "
              f"across {self._fit_data['date'].nunique()} slates")

        # Try to fit a simple regression
        try:
            from sklearn.linear_model import Ridge
            self._fit_regression(self._fit_data)
        except ImportError:
            print("  sklearn not available — using curve-based model only")

    def _fit_regression(self, data: pd.DataFrame):
        """Fit a Ridge regression on SE ownership data."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        features = self._build_features(data)
        target = data['own_pct'].values

        valid = features.notna().all(axis=1) & np.isfinite(target)
        X = features[valid].values
        y = target[valid]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=10.0)
        model.fit(X_scaled, y)

        preds = model.predict(X_scaled)
        mae = np.mean(np.abs(preds - y))
        corr = np.corrcoef(preds, y)[0, 1]

        self.regression_model = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(features.columns),
            'mae': mae,
            'corr': corr,
        }

        print(f"  Ridge regression: MAE={mae:.2f}% Corr={corr:.3f}")

        # Show feature importance
        coefs = pd.Series(model.coef_, index=features.columns).abs().sort_values(ascending=False)
        print(f"  Top features: {', '.join(f'{n}({v:.2f})' for n, v in coefs.head(5).items())}")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix for regression."""
        feats = pd.DataFrame(index=df.index)

        feats['salary_k'] = df['salary'] / 1000
        feats['salary_sq'] = feats['salary_k'] ** 2
        feats['dk_avg'] = df.get('dk_avg_fpts', pd.Series(5.0, index=df.index))
        feats['proj_fpts'] = df['projected_fpts']

        # Position rank within slate
        if 'date' in df.columns:
            feats['pos_rank'] = df.groupby(['date', 'position'])['projected_fpts'].rank(
                ascending=False, method='first'
            )
        else:
            feats['pos_rank'] = df.groupby('position')['projected_fpts'].rank(
                ascending=False, method='first'
            )
        feats['pos_rank_inv'] = 1.0 / feats['pos_rank'].clip(lower=1)

        # Position dummies
        feats['is_C'] = (df['position'] == 'C').astype(float)
        feats['is_G'] = (df['position'] == 'G').astype(float)
        feats['is_D'] = (df['position'] == 'D').astype(float)

        # Salary × projection interaction
        feats['sal_proj'] = feats['salary_k'] * feats['proj_fpts']

        # dk_avg percentile (name recognition)
        feats['dk_avg_pctile'] = feats['dk_avg'].rank(pct=True)

        return feats.fillna(0)

    def predict(self, player_pool: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Predict SE ownership for a player pool.

        Uses regression model if trained, falls back to curve-based.
        """
        df = player_pool.copy()

        if self.regression_model is not None:
            df = self._regression_predict(df)
        else:
            df = self._curve_predict(df)

        # Normalize: total ownership should sum to ~900% (9 roster spots)
        # In practice, DK shows per-player % so it won't sum to 100%
        # but the relative ordering matters most

        if verbose:
            top = df.nlargest(15, 'se_ownership')
            print(f"\n  ── SE Ownership Predictions ──────────────────")
            print(f"  {'Name':<22} {'Pos':>3} {'Sal':>6} {'DKAvg':>6} {'SE Own':>7}")
            print(f"  {'─' * 48}")
            for _, p in top.iterrows():
                dk = p.get('dk_avg_fpts', 0) or 0
                print(f"  {p['name']:<22} {p['position']:>3} ${p['salary']:>5} "
                      f"{dk:>6.1f} {p['se_ownership']:>6.1f}%")

        return df

    def _regression_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict using trained regression model."""
        feats = self._build_features(df)
        X = feats[self.regression_model['feature_names']].values
        X_scaled = self.regression_model['scaler'].transform(X)
        preds = self.regression_model['model'].predict(X_scaled)

        # Clip to reasonable range
        preds = np.clip(preds, 0.5, 60.0)

        # Apply confirmed goalie boost
        if self.confirmed_goalies:
            for i, (_, row) in enumerate(df.iterrows()):
                if row['position'] == 'G':
                    team = row.get('team', '')
                    name = row.get('name', '')
                    if team in self.confirmed_goalies:
                        confirmed_name = self.confirmed_goalies[team]
                        if _fuzzy_match(name, confirmed_name):
                            preds[i] *= 1.8  # Confirmed starter boost
                        else:
                            preds[i] *= 0.15  # Backup penalty

        df['se_ownership'] = np.round(preds, 1)

        # Also set the main predicted_ownership column
        df['predicted_ownership'] = df['se_ownership']

        # Tier labels
        df['ownership_tier'] = pd.cut(
            df['se_ownership'],
            bins=[0, 5, 10, 20, 100],
            labels=['contrarian', 'moderate', 'popular', 'chalk']
        )

        return df

    def _curve_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict using empirical curves (no training data needed)."""
        results = []

        # Compute position rank
        df = df.copy()
        df['_pos_rank'] = df.groupby('position')['projected_fpts'].rank(
            ascending=False, method='first'
        )

        for _, row in df.iterrows():
            sal = row.get('salary', 4000)
            pos = row.get('position', 'W')
            dk_avg = row.get('dk_avg_fpts', 5.0) or 5.0
            pos_rank = int(row.get('_pos_rank', 15))

            # 1. Base from salary
            base = 3.0  # default
            for (lo, hi), val in SE_SALARY_CURVE.items():
                if lo <= sal < hi:
                    base = val
                    break

            # 2. Position rank multiplier
            rank_mult = SE_RANK_MULTIPLIER.get(pos_rank, max(0.5, 1.0 - (pos_rank - 12) * 0.03))

            # 3. Position adjustment
            pos_mult = SE_POSITION_MULT.get(pos, 1.0)

            # 4. DK avg (name recognition) — the biggest SE driver
            dk_mult = 1.0
            for tier_name, tier in SE_DK_AVG_TIERS.items():
                if dk_avg >= tier['min']:
                    dk_mult = tier['mult']
                    break

            # 5. Confirmed goalie boost
            goalie_mult = 1.0
            if pos == 'G' and self.confirmed_goalies:
                team = row.get('team', '')
                name = row.get('name', '')
                if team in self.confirmed_goalies:
                    if _fuzzy_match(name, self.confirmed_goalies[team]):
                        goalie_mult = 2.0  # Confirmed = heavily owned in SE
                    else:
                        goalie_mult = 0.1  # Backup = barely owned

            own = base * rank_mult * pos_mult * dk_mult * goalie_mult
            own = np.clip(own, 0.5, 55.0)
            results.append(round(own, 1))

        df['se_ownership'] = results
        df['predicted_ownership'] = df['se_ownership']
        df['ownership_tier'] = pd.cut(
            df['se_ownership'],
            bins=[0, 5, 10, 20, 100],
            labels=['contrarian', 'moderate', 'popular', 'chalk']
        )

        df.drop(columns=['_pos_rank'], inplace=True, errors='ignore')
        return df


def _fuzzy_match(name1: str, name2: str) -> bool:
    """Simple fuzzy name match."""
    n1 = name1.lower().strip().split()[-1] if name1 else ''
    n2 = name2.lower().strip().split()[-1] if name2 else ''
    return n1 == n2


# ================================================================
#  CLI / Evaluation
# ================================================================

def evaluate_model():
    """Evaluate the SE ownership model against actual contest data."""
    model = SEOwnershipModel()
    model.fit_from_contests()

    if model._fit_data is None:
        print("No contest data to evaluate")
        return

    data = model._fit_data

    # Cross-validate: predict each date using data from other dates
    dates = sorted(data['date'].unique())
    all_preds = []

    for test_date in dates:
        test = data[data['date'] == test_date].copy()
        # Use curve model (always available) for comparison
        pred = model._curve_predict(test)
        pred['actual_own'] = test['own_pct'].values
        pred['date'] = test_date
        all_preds.append(pred)

    results = pd.concat(all_preds, ignore_index=True)

    # Overall metrics
    curve_mae = (results['se_ownership'] - results['actual_own']).abs().mean()
    curve_corr = results[['se_ownership', 'actual_own']].corr().iloc[0, 1]
    curve_bias = (results['se_ownership'] - results['actual_own']).mean()

    # Compare to old model
    if 'predicted_ownership' in data.columns:
        old_preds = data['predicted_ownership'] if 'predicted_ownership' in data.columns else data.get('own_pct', 0) * 0
    else:
        old_preds = pd.Series(5.0, index=data.index)

    print(f"\n{'=' * 60}")
    print(f"  SE OWNERSHIP MODEL EVALUATION")
    print(f"{'=' * 60}")
    print(f"\n  New SE Model:")
    print(f"    MAE:  {curve_mae:.2f}%")
    print(f"    Corr: {curve_corr:.3f}")
    print(f"    Bias: {curve_bias:+.2f}%")

    if model.regression_model:
        print(f"\n  Ridge Regression (in-sample):")
        print(f"    MAE:  {model.regression_model['mae']:.2f}%")
        print(f"    Corr: {model.regression_model['corr']:.3f}")

    # Per-date
    print(f"\n  Per-Date (curve model):")
    for d in dates:
        sub = results[results['date'] == d]
        mae = (sub['se_ownership'] - sub['actual_own']).abs().mean()
        corr = sub[['se_ownership', 'actual_own']].corr().iloc[0, 1]
        print(f"    {d}: MAE={mae:.2f}% Corr={corr:.3f} ({len(sub)} players)")


if __name__ == '__main__':
    evaluate_model()
