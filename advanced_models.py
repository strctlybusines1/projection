#!/usr/bin/env python3
"""
Advanced Projection Models for NHL DFS
========================================

Six specialized models designed to reduce MAE for both skaters and goalies.
Built for Google Colab execution with GPU acceleration where applicable.

Models:
    1. Bayesian Hidden Markov Model (HMM) — Player performance regimes
    2. Conditional Random Fields (CRF) — Team-context sequence labeling
    3. Markov Switching Model — Detect hot/cold streaks
    4. Regime Switching Model — Market regime-aware projections
    5. Change Point Detection — Identify breakout/breakdown moments
    6. Reservoir Computing (Echo State Network) — Time series projection

Run in Colab:
    !pip install hmmlearn pomegranate sklearn-crfsuite ruptures reservoirpy
    Then run each model section.

Data requirements:
    - backtests/batch_backtest_details.csv (actuals)
    - daily_projections/*NHLprojections_*.csv (features)
    - Vegas_Historical.csv (odds/totals)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats

# ================================================================
#  DATA LOADER (shared across all models)
# ================================================================

class NHLDataLoader:
    """Load and prepare data for all models."""

    def __init__(self, project_root: str = '.'):
        self.root = Path(project_root)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load actuals, projections, and Vegas data."""
        data = {}

        # Actuals
        act_path = self.root / 'backtests' / 'batch_backtest_details.csv'
        if act_path.exists():
            data['actuals'] = pd.read_csv(act_path)

        # Vegas
        for vp in [self.root / 'Vegas_Historical.csv',
                    self.root / 'vegas' / 'Vegas_Historical.csv']:
            if vp.exists():
                vdf = pd.read_csv(vp, encoding='utf-8-sig')
                vdf['date'] = vdf['Date'].apply(self._parse_vegas_date)
                vdf['win_pct'] = vdf['Win %'].str.rstrip('%').astype(float) / 100
                data['vegas'] = vdf
                break

        # Projections (latest per date)
        proj_dir = self.root / 'daily_projections'
        if proj_dir.exists():
            projs = {}
            for f in sorted(proj_dir.glob('*NHLprojections_*.csv')):
                if '_lineups' in f.name:
                    continue
                # Extract date prefix
                parts = f.name.split('NHLprojections')[0]
                projs[parts] = f

            all_projs = []
            for prefix, fpath in projs.items():
                df = pd.read_csv(fpath)
                # Parse date from prefix like "01_23_26"
                try:
                    parts = prefix.split('_')
                    if len(parts) >= 3:
                        m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
                        df['date'] = f"2026-{m:02d}-{d:02d}"
                except:
                    pass
                all_projs.append(df)

            if all_projs:
                data['projections'] = pd.concat(all_projs, ignore_index=True)

        return data

    def build_training_set(self, data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build matched training set: projection features → actual FPTS.
        Returns (skaters_df, goalies_df).
        """
        if 'actuals' not in data or 'projections' not in data:
            raise ValueError("Need both actuals and projections")

        act = data['actuals'].copy()
        proj = data['projections'].copy()

        def ln(n): return str(n).strip().split()[-1].lower()

        act['_key'] = act['name'].apply(ln) + '_' + act['team'].str.lower() + '_' + act['date']

        if 'date' in proj.columns:
            proj['_key'] = proj['name'].apply(ln) + '_' + proj['team'].str.lower() + '_' + proj['date']
        else:
            return pd.DataFrame(), pd.DataFrame()

        merged = act.merge(
            proj.drop_duplicates('_key'),
            on='_key', how='inner', suffixes=('_actual', '_proj')
        )

        skaters = merged[merged['position_actual'] != 'G'].copy()
        goalies = merged[merged['position_actual'] == 'G'].copy()

        return skaters, goalies

    @staticmethod
    def _parse_vegas_date(d):
        parts = d.strip().split('.')
        if len(parts) == 3:
            return f"20{parts[2]}-{int(parts[0]):02d}-{int(parts[1]):02d}"
        return d


# ================================================================
#  MODEL 1: BAYESIAN HIDDEN MARKOV MODEL
# ================================================================

class BayesianHMM:
    """
    Models player performance as transitions between hidden states:
        - HOT: scoring above expectation
        - NEUTRAL: scoring near expectation
        - COLD: scoring below expectation

    Uses game-log sequences to learn transition probabilities,
    then predicts which state a player is likely in for the next game.

    Key insight for goalies: goalies have much more volatile state transitions
    (a goalie can go from elite to terrible based on matchup/rest).
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = None
        self.state_means = None
        self.state_labels = ['COLD', 'NEUTRAL', 'HOT']

    def fit(self, sequences: List[np.ndarray]):
        """
        Fit HMM on player FPTS sequences.

        Args:
            sequences: List of 1D arrays, each a player's game-by-game FPTS
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            print("  Install: pip install hmmlearn")
            return self

        # Concatenate sequences with lengths
        all_data = np.concatenate(sequences).reshape(-1, 1)
        lengths = [len(s) for s in sequences]

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=200,
            random_state=42,
            init_params='stmc',
        )

        # Initialize with reasonable priors
        overall_mean = all_data.mean()
        overall_std = all_data.std()

        self.model.means_init = np.array([
            [overall_mean - overall_std],     # COLD
            [overall_mean],                   # NEUTRAL
            [overall_mean + overall_std],     # HOT
        ])

        self.model.fit(all_data, lengths)

        # Sort states by mean (so 0=COLD, 1=NEUTRAL, 2=HOT)
        order = np.argsort(self.model.means_.flatten())
        self.model.means_ = self.model.means_[order]
        self.model.covars_ = self.model.covars_[order]
        self.model.transmat_ = self.model.transmat_[order][:, order]
        self.model.startprob_ = self.model.startprob_[order]

        self.state_means = self.model.means_.flatten()

        return self

    def predict_state(self, recent_games: np.ndarray) -> Dict:
        """
        Given a player's recent game FPTS, predict their current state
        and probability of each state for next game.
        """
        if self.model is None:
            return {'state': 'NEUTRAL', 'state_probs': [0.33, 0.34, 0.33]}

        obs = recent_games.reshape(-1, 1)
        try:
            log_prob, state_seq = self.model.decode(obs, algorithm='viterbi')

            # Get current state
            current_state = state_seq[-1]

            # Next-game state probabilities from transition matrix
            next_probs = self.model.transmat_[current_state]

            return {
                'state': self.state_labels[current_state],
                'state_probs': next_probs.tolist(),
                'hot_prob': float(next_probs[2]),
                'cold_prob': float(next_probs[0]),
                'state_means': self.state_means.tolist(),
                'expected_adjustment': float(
                    np.dot(next_probs, self.state_means) - self.state_means[1]
                ),
            }
        except Exception:
            return {'state': 'NEUTRAL', 'state_probs': [0.33, 0.34, 0.33]}

    def build_sequences_from_actuals(self, actuals: pd.DataFrame,
                                     min_games: int = 3) -> Tuple[List[np.ndarray], Dict]:
        """
        Build per-player FPTS sequences from actuals data.
        Returns sequences list and player-to-index mapping.
        """
        sequences = []
        player_map = {}

        for (name, team), group in actuals.groupby(['name', 'team']):
            games = group.sort_values('date')['actual_fpts'].values
            if len(games) >= min_games:
                sequences.append(games)
                player_map[f"{name}_{team}"] = len(sequences) - 1

        return sequences, player_map


# ================================================================
#  MODEL 2: CONDITIONAL RANDOM FIELDS (CRF)
# ================================================================

class NHLConditionalRandomField:
    """
    Models player performance as a structured prediction problem:
    Given team context (line combos, matchup, home/away), predict
    performance labels for each player in a game.

    Unlike independent predictions, CRF considers team-level correlations:
    - If a goalie performs well, opposing skaters likely underperform
    - If a top line scores, other lines on the team may see less ice time
    - Team implied total affects all players on both sides

    Labels: LOW (<3 FPTS), MID (3-8), HIGH (8-15), BOOM (15+)
    """

    LABELS = ['LOW', 'MID', 'HIGH', 'BOOM']
    THRESHOLDS = [0, 3, 8, 15, 999]

    def __init__(self):
        self.model = None

    def fit(self, game_features: List[List[Dict]], game_labels: List[List[str]]):
        """
        Fit CRF on game-level sequences.

        Args:
            game_features: List of games, each game is list of player feature dicts
            game_labels: List of games, each game is list of labels (LOW/MID/HIGH/BOOM)
        """
        try:
            import sklearn_crfsuite
        except ImportError:
            print("  Install: pip install sklearn-crfsuite")
            return self

        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )

        self.model.fit(game_features, game_labels)
        return self

    def predict_proba(self, game_features: List[Dict]) -> List[Dict]:
        """Predict label probabilities for each player in a game."""
        if self.model is None:
            return [{'LOW': 0.4, 'MID': 0.35, 'HIGH': 0.15, 'BOOM': 0.1}] * len(game_features)

        try:
            marginals = self.model.predict_marginals_single(game_features)
            return marginals
        except Exception:
            return [{'LOW': 0.4, 'MID': 0.35, 'HIGH': 0.15, 'BOOM': 0.1}] * len(game_features)

    @staticmethod
    def fpts_to_label(fpts: float) -> str:
        """Convert FPTS to performance label."""
        if fpts < 3:
            return 'LOW'
        elif fpts < 8:
            return 'MID'
        elif fpts < 15:
            return 'HIGH'
        else:
            return 'BOOM'

    @staticmethod
    def label_to_expected_fpts(probs: Dict) -> float:
        """Convert label probabilities to expected FPTS."""
        midpoints = {'LOW': 1.5, 'MID': 5.5, 'HIGH': 11.5, 'BOOM': 22.0}
        return sum(probs.get(k, 0) * v for k, v in midpoints.items())

    def build_game_sequences(self, actuals: pd.DataFrame, projections: pd.DataFrame,
                             vegas: pd.DataFrame = None) -> Tuple[List, List]:
        """
        Build CRF training data: game-level player sequences with features.
        """
        def ln(n): return str(n).strip().split()[-1].lower()

        features_list = []
        labels_list = []

        for date in sorted(actuals['date'].unique()):
            day_act = actuals[actuals['date'] == date].copy()
            day_act['_key'] = day_act['name'].apply(ln) + '_' + day_act['team'].str.lower()

            # Get matching projections
            day_proj = projections[projections.get('date', '') == date].copy() if 'date' in projections.columns else pd.DataFrame()

            if not day_proj.empty:
                day_proj['_key'] = day_proj['name'].apply(ln) + '_' + day_proj['team'].str.lower()

            # Get Vegas data for this date
            vegas_map = {}
            if vegas is not None:
                day_vegas = vegas[vegas['date'] == date]
                for _, vrow in day_vegas.iterrows():
                    vegas_map[vrow['Team']] = {
                        'team_total': vrow['TeamGoal'],
                        'opp_total': vrow['OppGoal'],
                        'win_pct': vrow.get('win_pct', 0.5),
                        'game_total': vrow['Total'],
                    }

            # Group by team for context
            for team, team_players in day_act.groupby('team'):
                if len(team_players) < 3:
                    continue

                game_features = []
                game_labels = []

                for _, player in team_players.iterrows():
                    feat = {
                        'position': player['position'],
                        'team': team,
                        'salary_tier': 'high' if player.get('salary', 5000) > 7000 else
                                      'mid' if player.get('salary', 5000) > 4500 else 'low',
                    }

                    # Add projection features if available
                    if not day_proj.empty:
                        match = day_proj[day_proj['_key'] == player['_key']]
                        if not match.empty:
                            m = match.iloc[0]
                            feat['projected_fpts'] = str(round(m.get('projected_fpts', 5), 1))
                            feat['dk_avg'] = str(round(m.get('dk_avg_fpts', 5), 1))

                    # Add Vegas context
                    if team in vegas_map:
                        vm = vegas_map[team]
                        feat['team_implied'] = 'high' if vm['team_total'] > 3.3 else \
                                              'low' if vm['team_total'] < 2.7 else 'mid'
                        feat['game_total'] = 'high' if vm['game_total'] > 6.0 else 'low'

                    game_features.append(feat)
                    game_labels.append(self.fpts_to_label(player['actual_fpts']))

                if game_features:
                    features_list.append(game_features)
                    labels_list.append(game_labels)

        return features_list, labels_list


# ================================================================
#  MODEL 3: MARKOV SWITCHING MODEL
# ================================================================

class MarkovSwitchingModel:
    """
    Detects hot/cold streaks by modeling FPTS as switching between
    two (or more) Gaussian distributions with Markov transitions.

    Unlike simple moving averages, this explicitly models:
    - The probability of being in a hot vs cold regime
    - The persistence of each regime (how "sticky" streaks are)
    - Different variance in each regime

    Particularly useful for goalies, who have bimodal performance:
    a goalie either has a "good game" (15-25 FPTS) or a "bad game" (0-8 FPTS).
    """

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.transition_probs = None
        self.regime_params = None  # (mean, std) per regime
        self.fitted = False

    def fit(self, sequences: List[np.ndarray]):
        """
        Fit switching model using EM algorithm.
        Simplified implementation that doesn't require statsmodels.
        """
        all_data = np.concatenate(sequences)

        # Use k-means style initialization
        sorted_data = np.sort(all_data)
        n = len(sorted_data)
        split = n // self.n_regimes

        self.regime_params = []
        for i in range(self.n_regimes):
            start = i * split
            end = (i + 1) * split if i < self.n_regimes - 1 else n
            segment = sorted_data[start:end]
            self.regime_params.append({
                'mean': float(np.mean(segment)),
                'std': float(max(np.std(segment), 0.5)),
            })

        # Sort regimes by mean (cold first)
        self.regime_params.sort(key=lambda x: x['mean'])

        # Estimate transition matrix from sequences
        self.transition_probs = np.full((self.n_regimes, self.n_regimes), 1.0 / self.n_regimes)

        # EM-like iteration to refine
        for iteration in range(20):
            # E-step: assign observations to regimes
            all_assignments = []
            for seq in sequences:
                assignments = []
                for val in seq:
                    probs = []
                    for r in range(self.n_regimes):
                        p = stats.norm.pdf(val, self.regime_params[r]['mean'],
                                          self.regime_params[r]['std'])
                        probs.append(p)
                    total = sum(probs) + 1e-10
                    probs = [p / total for p in probs]
                    assignments.append(np.argmax(probs))
                all_assignments.append(assignments)

            # M-step: update regime params
            for r in range(self.n_regimes):
                vals = []
                for seq, assigns in zip(sequences, all_assignments):
                    for val, a in zip(seq, assigns):
                        if a == r:
                            vals.append(val)
                if vals:
                    self.regime_params[r]['mean'] = float(np.mean(vals))
                    self.regime_params[r]['std'] = float(max(np.std(vals), 0.5))

            # Update transition matrix
            trans_counts = np.zeros((self.n_regimes, self.n_regimes))
            for assigns in all_assignments:
                for t in range(len(assigns) - 1):
                    trans_counts[assigns[t]][assigns[t + 1]] += 1

            row_sums = trans_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            self.transition_probs = trans_counts / row_sums

        self.fitted = True
        return self

    def predict_regime(self, recent_games: np.ndarray) -> Dict:
        """Predict current regime and next-game expectations."""
        if not self.fitted:
            return {'regime': 'neutral', 'adjustment': 0.0}

        # Classify last game
        last_val = recent_games[-1]
        probs = []
        for r in range(self.n_regimes):
            p = stats.norm.pdf(last_val, self.regime_params[r]['mean'],
                              self.regime_params[r]['std'])
            probs.append(p)
        total = sum(probs) + 1e-10
        probs = [p / total for p in probs]
        current_regime = np.argmax(probs)

        # Next game probabilities
        next_probs = self.transition_probs[current_regime]
        expected_next = sum(
            next_probs[r] * self.regime_params[r]['mean']
            for r in range(self.n_regimes)
        )

        regime_labels = ['cold', 'hot'] if self.n_regimes == 2 else \
                       ['cold', 'neutral', 'hot']

        return {
            'regime': regime_labels[current_regime],
            'regime_probs': next_probs.tolist(),
            'expected_fpts': float(expected_next),
            'regime_means': [r['mean'] for r in self.regime_params],
            'persistence': float(self.transition_probs[current_regime][current_regime]),
            'adjustment': float(expected_next - np.mean([r['mean'] for r in self.regime_params])),
        }


# ================================================================
#  MODEL 4: REGIME SWITCHING MODEL (Market/Slate-Level)
# ================================================================

class RegimeSwitchingModel:
    """
    Unlike the Markov Switching Model (per-player), this operates at the
    SLATE level. Detects market regimes that affect all players:

    Regimes:
    - CHALK: High-owned players hit, favorites dominate → project conservatively
    - CHAOS: Upsets, low-owned booms, high variance → widen distributions
    - DEFENSE: Low-scoring games, goalies overperform → boost goalie projections

    Uses Vegas totals, team records, and slate-level scoring patterns
    to classify the regime before the slate.
    """

    def __init__(self):
        self.regime_classifiers = {}
        self.regime_history = []

    def fit(self, slate_data: List[Dict]):
        """
        Fit from historical slate-level features.

        Args:
            slate_data: List of dicts with keys:
                - avg_total: average game total for the slate
                - n_games: number of games
                - avg_actual_fpts: actual average FPTS across all players
                - pct_chalk_hit: % of top-projected players that hit (>median)
                - goalie_avg_fpts: average goalie FPTS
        """
        if not slate_data:
            return self

        df = pd.DataFrame(slate_data)

        # Classify regimes using clustering
        from sklearn.cluster import KMeans

        features = ['avg_total', 'avg_actual_fpts', 'goalie_avg_fpts']
        available = [f for f in features if f in df.columns]
        if not available:
            return self

        X = df[available].fillna(df[available].mean()).values

        if len(X) >= 3:
            km = KMeans(n_clusters=min(3, len(X)), random_state=42)
            df['regime'] = km.fit_predict(X)

            # Label regimes by characteristics
            for regime_id in df['regime'].unique():
                regime_data = df[df['regime'] == regime_id]
                self.regime_classifiers[regime_id] = {
                    'avg_total': regime_data['avg_total'].mean() if 'avg_total' in df else 6.0,
                    'avg_fpts': regime_data['avg_actual_fpts'].mean() if 'avg_actual_fpts' in df else 5.0,
                    'goalie_fpts': regime_data['goalie_avg_fpts'].mean() if 'goalie_avg_fpts' in df else 7.0,
                }

            self.km = km
            self.feature_cols = available
            self.feature_means = df[available].mean().to_dict()

        return self

    def predict_regime(self, slate_features: Dict) -> Dict:
        """
        Predict the regime for an upcoming slate.

        Args:
            slate_features: Dict with avg_total, n_games, etc.
        """
        if not self.regime_classifiers:
            return {
                'regime': 'NEUTRAL',
                'skater_adjustment': 0.0,
                'goalie_adjustment': 0.0,
                'variance_scale': 1.0,
            }

        # Classify
        X = np.array([[slate_features.get(f, self.feature_means.get(f, 0))
                       for f in self.feature_cols]])

        regime_id = self.km.predict(X)[0]
        regime_info = self.regime_classifiers[regime_id]

        # Determine adjustment
        avg_total = slate_features.get('avg_total', 6.0)
        if avg_total > 6.5:
            label = 'HIGH_SCORING'
            sk_adj = +0.5
            g_adj = -1.0
            var_scale = 1.2
        elif avg_total < 5.5:
            label = 'DEFENSE'
            sk_adj = -0.5
            g_adj = +1.5
            var_scale = 0.9
        else:
            label = 'NEUTRAL'
            sk_adj = 0.0
            g_adj = 0.0
            var_scale = 1.0

        return {
            'regime': label,
            'skater_adjustment': float(sk_adj),
            'goalie_adjustment': float(g_adj),
            'variance_scale': float(var_scale),
            'regime_info': regime_info,
        }

    def build_slate_features(self, actuals: pd.DataFrame, vegas: pd.DataFrame) -> List[Dict]:
        """Build slate-level feature vectors from historical data."""
        slates = []

        for date in sorted(actuals['date'].unique()):
            day = actuals[actuals['date'] == date]
            sk = day[day['position'] != 'G']
            g = day[day['position'] == 'G']

            slate = {
                'date': date,
                'n_players': len(day),
                'avg_actual_fpts': float(day['actual_fpts'].mean()),
                'sk_avg_fpts': float(sk['actual_fpts'].mean()) if len(sk) > 0 else 0,
                'goalie_avg_fpts': float(g['actual_fpts'].mean()) if len(g) > 0 else 0,
                'pct_zero': float((day['actual_fpts'] == 0).mean()),
            }

            # Add Vegas features
            if vegas is not None:
                day_vegas = vegas[vegas['date'] == date]
                if not day_vegas.empty:
                    slate['avg_total'] = float(day_vegas['Total'].mean())
                    slate['avg_team_total'] = float(day_vegas['TeamGoal'].mean())
                    slate['n_games'] = len(day_vegas) // 2

            slates.append(slate)

        return slates


# ================================================================
#  MODEL 5: CHANGE POINT DETECTION
# ================================================================

class ChangePointDetector:
    """
    Detects structural breaks in player performance sequences.

    Use cases:
    - Identify when a player "breaks out" (permanently higher production)
    - Detect when a player's role changes (more/less ice time)
    - Flag goalies who have lost their starting job
    - Catch players returning from injury at diminished capacity

    Uses PELT (Pruned Exact Linear Time) algorithm from ruptures library
    when available, falls back to CUSUM approach.
    """

    def __init__(self, min_size: int = 3, penalty: float = 3.0):
        self.min_size = min_size
        self.penalty = penalty

    def detect(self, sequence: np.ndarray) -> Dict:
        """
        Detect change points in a player's FPTS sequence.

        Returns:
            Dict with change_points (indices), segments (mean/std per segment),
            and current_trend
        """
        if len(sequence) < self.min_size * 2:
            return {
                'change_points': [],
                'n_changes': 0,
                'current_segment_mean': float(np.mean(sequence)),
                'trend': 'stable',
            }

        # Try ruptures library first
        try:
            import ruptures as rpt

            model = rpt.Pelt(model='rbf', min_size=self.min_size)
            result = model.fit_predict(sequence.reshape(-1, 1), pen=self.penalty)
            change_points = [cp for cp in result if cp < len(sequence)]

        except ImportError:
            # Fallback: CUSUM-based detection
            change_points = self._cusum_detect(sequence)

        # Analyze segments
        segments = []
        prev = 0
        all_cps = sorted(set(change_points))
        for cp in all_cps:
            seg = sequence[prev:cp]
            if len(seg) > 0:
                segments.append({
                    'start': prev,
                    'end': cp,
                    'mean': float(np.mean(seg)),
                    'std': float(np.std(seg)),
                    'n': len(seg),
                })
            prev = cp

        # Last segment
        if prev < len(sequence):
            seg = sequence[prev:]
            segments.append({
                'start': prev,
                'end': len(sequence),
                'mean': float(np.mean(seg)),
                'std': float(np.std(seg)),
                'n': len(seg),
            })

        # Determine trend
        trend = 'stable'
        if len(segments) >= 2:
            last_mean = segments[-1]['mean']
            prev_mean = segments[-2]['mean']
            if last_mean > prev_mean * 1.2:
                trend = 'breakout'
            elif last_mean < prev_mean * 0.8:
                trend = 'breakdown'

        return {
            'change_points': all_cps,
            'n_changes': len(all_cps),
            'segments': segments,
            'current_segment_mean': segments[-1]['mean'] if segments else float(np.mean(sequence)),
            'trend': trend,
        }

    def _cusum_detect(self, sequence: np.ndarray, threshold: float = 2.0) -> List[int]:
        """Fallback CUSUM change point detection."""
        mean = np.mean(sequence)
        std = max(np.std(sequence), 0.1)

        s_pos = 0
        s_neg = 0
        changes = []

        for i, val in enumerate(sequence):
            z = (val - mean) / std
            s_pos = max(0, s_pos + z - 0.5)
            s_neg = max(0, s_neg - z - 0.5)

            if s_pos > threshold or s_neg > threshold:
                changes.append(i)
                s_pos = 0
                s_neg = 0

        return changes


# ================================================================
#  MODEL 6: RESERVOIR COMPUTING (ECHO STATE NETWORK)
# ================================================================

class EchoStateNetwork:
    """
    Echo State Network for time-series FPTS prediction.

    Reservoir computing uses a fixed random recurrent network (reservoir)
    that maps input time series into a high-dimensional space, then fits
    only the output layer. This makes it:
    - Fast to train (only linear regression on output)
    - Good at capturing temporal patterns
    - Robust to overfitting with limited data

    Particularly effective for goalie projections where the sequence
    pattern (rest days, back-to-back, matchup alternation) matters.
    """

    def __init__(self, reservoir_size: int = 200, spectral_radius: float = 0.95,
                 input_scaling: float = 0.5, leak_rate: float = 0.3,
                 ridge_alpha: float = 1.0):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha

        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.fitted = False

    def _initialize_reservoir(self, input_dim: int):
        """Initialize reservoir weights."""
        np.random.seed(42)

        # Input weights
        self.W_in = (np.random.rand(self.reservoir_size, input_dim) - 0.5) * self.input_scaling

        # Reservoir weights (sparse random)
        W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        sparsity = 0.9
        mask = np.random.rand(self.reservoir_size, self.reservoir_size) > sparsity
        W *= mask

        # Scale to desired spectral radius
        eigenvalues = np.abs(np.linalg.eigvals(W))
        max_eig = max(eigenvalues) if len(eigenvalues) > 0 else 1.0
        if max_eig > 0:
            W *= self.spectral_radius / max_eig

        self.W_res = W

    def fit(self, X_sequences: List[np.ndarray], y_sequences: List[np.ndarray]):
        """
        Fit ESN on input-output sequence pairs.

        Args:
            X_sequences: List of input arrays (T x input_dim)
            y_sequences: List of target arrays (T x 1)
        """
        if not X_sequences:
            return self

        input_dim = X_sequences[0].shape[1] if X_sequences[0].ndim > 1 else 1
        self._initialize_reservoir(input_dim)

        # Collect reservoir states
        all_states = []
        all_targets = []

        for X, y in zip(X_sequences, y_sequences):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            states = self._run_reservoir(X)
            # Skip warmup
            warmup = min(5, len(states) // 3)
            all_states.append(states[warmup:])
            all_targets.append(y[warmup:])

        # Concatenate and solve
        S = np.vstack(all_states)
        Y = np.vstack(all_targets)

        # Ridge regression for output weights
        self.W_out = np.linalg.solve(
            S.T @ S + self.ridge_alpha * np.eye(S.shape[1]),
            S.T @ Y
        )

        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from input sequence."""
        if not self.fitted:
            return np.array([0.0])

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        states = self._run_reservoir(X)
        predictions = states @ self.W_out

        return predictions.flatten()

    def _run_reservoir(self, X: np.ndarray) -> np.ndarray:
        """Run input through reservoir, collect states."""
        T = X.shape[0]
        states = np.zeros((T, self.reservoir_size))
        x = np.zeros(self.reservoir_size)

        for t in range(T):
            u = X[t]
            x_new = np.tanh(self.W_in @ u + self.W_res @ x)
            x = (1 - self.leak_rate) * x + self.leak_rate * x_new
            states[t] = x

        # Augment with input (for direct connections)
        return np.hstack([states, X])


# ================================================================
#  UNIFIED EVALUATION
# ================================================================

def evaluate_all_models(project_root: str = '.'):
    """
    Run all models on historical data and report MAE improvements.
    """
    loader = NHLDataLoader(project_root)
    data = loader.load_all()

    if 'actuals' not in data:
        print("No actuals data found")
        return

    actuals = data['actuals']
    vegas = data.get('vegas')

    skaters = actuals[actuals['position'] != 'G']
    goalies = actuals[actuals['position'] == 'G']

    print("=" * 72)
    print("  ADVANCED MODEL EVALUATION")
    print("=" * 72)
    print(f"\n  Data: {len(actuals)} observations across {actuals['date'].nunique()} slates")
    print(f"  Skaters: {len(skaters)}, Goalies: {len(goalies)}")

    baseline_sk_mae = skaters['error'].abs().mean()
    baseline_g_mae = goalies['error'].abs().mean()
    print(f"  Baseline Skater MAE: {baseline_sk_mae:.3f}")
    print(f"  Baseline Goalie MAE: {baseline_g_mae:.3f}")

    # ── Model 1: Bayesian HMM ──
    print(f"\n{'─' * 50}")
    print("  MODEL 1: Bayesian Hidden Markov Model")
    try:
        hmm = BayesianHMM(n_states=3)
        sequences, player_map = hmm.build_sequences_from_actuals(actuals, min_games=2)
        if sequences:
            hmm.fit(sequences)
            print(f"    Trained on {len(sequences)} player sequences")
            print(f"    State means: {[f'{m:.1f}' for m in hmm.state_means]}")
            print(f"    Transition matrix:")
            for i, row in enumerate(hmm.model.transmat_):
                print(f"      {hmm.state_labels[i]}: {' '.join(f'{p:.2f}' for p in row)}")
        else:
            print("    Insufficient sequence data (need 2+ games per player)")
    except Exception as e:
        print(f"    ⚠ {e}")
        print(f"    Install: pip install hmmlearn")

    # ── Model 2: CRF ──
    print(f"\n{'─' * 50}")
    print("  MODEL 2: Conditional Random Fields")
    try:
        crf = NHLConditionalRandomField()
        proj = data.get('projections', pd.DataFrame())
        if not proj.empty:
            feats, labels = crf.build_game_sequences(actuals, proj, vegas)
            if feats:
                crf.fit(feats, labels)
                print(f"    Trained on {len(feats)} game sequences")
                # Show transition weights
                if crf.model and hasattr(crf.model, 'transition_features_'):
                    trans = crf.model.transition_features_
                    top_trans = sorted(trans.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    for (s1, s2), w in top_trans:
                        print(f"      {s1} → {s2}: {w:.3f}")
            else:
                print("    No game sequences built")
        else:
            print("    No projection data available")
    except Exception as e:
        print(f"    ⚠ {e}")
        print(f"    Install: pip install sklearn-crfsuite")

    # ── Model 3: Markov Switching ──
    print(f"\n{'─' * 50}")
    print("  MODEL 3: Markov Switching Model")
    try:
        # Separate skater and goalie models
        for label, subset in [('Skaters', skaters), ('Goalies', goalies)]:
            sequences = []
            for (name, team), group in subset.groupby(['name', 'team']):
                games = group.sort_values('date')['actual_fpts'].values
                if len(games) >= 2:
                    sequences.append(games)

            if sequences:
                msm = MarkovSwitchingModel(n_regimes=2)
                msm.fit(sequences)
                print(f"    {label} ({len(sequences)} sequences):")
                for i, rp in enumerate(msm.regime_params):
                    print(f"      Regime {i}: mean={rp['mean']:.1f} std={rp['std']:.1f}")
                print(f"      Persistence: {[f'{msm.transition_probs[i][i]:.2f}' for i in range(2)]}")
    except Exception as e:
        print(f"    ⚠ {e}")

    # ── Model 4: Regime Switching ──
    print(f"\n{'─' * 50}")
    print("  MODEL 4: Regime Switching (Slate-Level)")
    try:
        rsm = RegimeSwitchingModel()
        if vegas is not None:
            slates = rsm.build_slate_features(actuals, vegas)
            if slates:
                rsm.fit(slates)
                print(f"    Trained on {len(slates)} slates")
                for s in slates:
                    regime = rsm.predict_regime(s)
                    print(f"      {s['date']}: {regime['regime']} "
                          f"(sk_adj={regime['skater_adjustment']:+.1f}, "
                          f"g_adj={regime['goalie_adjustment']:+.1f})")
    except Exception as e:
        print(f"    ⚠ {e}")

    # ── Model 5: Change Point Detection ──
    print(f"\n{'─' * 50}")
    print("  MODEL 5: Change Point Detection")
    try:
        cpd = ChangePointDetector()
        changes_found = 0
        breakouts = []
        breakdowns = []

        for (name, team), group in actuals.groupby(['name', 'team']):
            games = group.sort_values('date')['actual_fpts'].values
            if len(games) >= 4:
                result = cpd.detect(games)
                if result['n_changes'] > 0:
                    changes_found += 1
                    if result['trend'] == 'breakout':
                        breakouts.append((name, team, result))
                    elif result['trend'] == 'breakdown':
                        breakdowns.append((name, team, result))

        print(f"    Players with detected changes: {changes_found}")
        if breakouts:
            print(f"    Breakouts ({len(breakouts)}):")
            for name, team, r in breakouts[:5]:
                print(f"      {name} ({team}): {r['segments'][-2]['mean']:.1f} → {r['segments'][-1]['mean']:.1f}")
        if breakdowns:
            print(f"    Breakdowns ({len(breakdowns)}):")
            for name, team, r in breakdowns[:5]:
                print(f"      {name} ({team}): {r['segments'][-2]['mean']:.1f} → {r['segments'][-1]['mean']:.1f}")
    except Exception as e:
        print(f"    ⚠ {e}")

    # ── Model 6: Echo State Network ──
    print(f"\n{'─' * 50}")
    print("  MODEL 6: Echo State Network")
    try:
        esn = EchoStateNetwork(reservoir_size=100, spectral_radius=0.9)

        X_seqs = []
        y_seqs = []
        for (name, team), group in actuals.groupby(['name', 'team']):
            games = group.sort_values('date')['actual_fpts'].values
            if len(games) >= 4:
                # Input: lagged features
                X = np.column_stack([
                    games[:-1],
                    np.arange(len(games) - 1) / len(games),  # position in sequence
                ])
                y = games[1:]
                X_seqs.append(X)
                y_seqs.append(y)

        if X_seqs:
            esn.fit(X_seqs, y_seqs)
            print(f"    Trained on {len(X_seqs)} sequences")

            # Test prediction
            total_mae = 0
            n = 0
            for X, y in zip(X_seqs, y_seqs):
                pred = esn.predict(X)
                total_mae += np.abs(pred[-1] - y[-1])
                n += 1

            if n > 0:
                print(f"    Last-game prediction MAE: {total_mae/n:.2f}")
        else:
            print("    Insufficient sequence data")
    except Exception as e:
        print(f"    ⚠ {e}")

    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  All models initialized. For full training with game logs,")
    print(f"  run on Colab with: !pip install hmmlearn sklearn-crfsuite ruptures reservoirpy")
    print(f"  Then use the individual model classes with richer data.")


# ================================================================
#  COLAB NOTEBOOK GENERATOR
# ================================================================

def generate_colab_notebook():
    """Generate a Jupyter notebook for Colab execution."""
    cells = []

    # Cell 1: Setup
    cells.append({
        'cell_type': 'code',
        'source': '''# Install dependencies
!pip install hmmlearn sklearn-crfsuite ruptures reservoirpy scikit-learn scipy pandas numpy -q
print("✓ Dependencies installed")''',
    })

    # Cell 2: Upload data
    cells.append({
        'cell_type': 'code',
        'source': '''# Upload your data files
from google.colab import files
import os

print("Upload these files from your projection/ directory:")
print("  1. backtests/batch_backtest_details.csv")
print("  2. Vegas_Historical.csv")
print("  3. All daily_projections/*NHLprojections_*.csv files")
print()

uploaded = files.upload()

# Create directory structure
os.makedirs('backtests', exist_ok=True)
os.makedirs('daily_projections', exist_ok=True)

for fname, content in uploaded.items():
    if 'backtest' in fname.lower():
        dest = f'backtests/{fname}'
    elif 'projection' in fname.lower() or 'NHL' in fname:
        dest = f'daily_projections/{fname}'
    else:
        dest = fname
    with open(dest, 'wb') as f:
        f.write(content)
    print(f"  Saved: {dest}")''',
    })

    # Cell 3: Run all models
    cells.append({
        'cell_type': 'code',
        'source': '''# Run all 6 models
# (Paste the entire advanced_models.py content here or upload it)
# Then run:
evaluate_all_models('.')''',
    })

    return cells


# ================================================================
#  CLI
# ================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Advanced NHL DFS Models')
    parser.add_argument('--evaluate', action='store_true', help='Run all models on historical data')
    parser.add_argument('--colab', action='store_true', help='Generate Colab notebook')
    parser.add_argument('--root', type=str, default='.', help='Project root directory')
    args = parser.parse_args()

    if args.colab:
        cells = generate_colab_notebook()
        print(f"Generated {len(cells)} notebook cells")
    else:
        evaluate_all_models(args.root)
