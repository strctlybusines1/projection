"""
NHL DFS Ownership Projection Model

Predicts ownership percentages for DraftKings NHL contests.
Optimized for top-heavy GPP structures ($5 entry, $5K to 1st).

Key insights from historical analysis:
- 47% of chalk (15%+ owned) busts (<10 FPTS)
- 20%+ owned players average only 7.1 FPTS (worst tier)
- Best leverage: 5-15% owned players with high ceilings
- Confirmed goalies critical (non-confirmed ~0% owned)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import spearmanr

from config import DAILY_SALARIES_DIR, CONTESTS_DIR, DAILY_PROJECTIONS_DIR


@dataclass
class OwnershipConfig:
    """Configuration for ownership model parameters."""
    # Salary curve parameters (based on historical analysis)
    # Adjusted: boosted mid-salary range where chalk concentrates
    salary_curve = {
        (2500, 3500): 4.0,    # Punt plays
        (3500, 4500): 12.0,   # Value sweet spot - BOOSTED (was 7.0)
        (4500, 5500): 14.0,   # Value sweet spot - BOOSTED (was 10.0)
        (5500, 6500): 12.0,   # Mid-range
        (6500, 7500): 11.0,   # Solid plays
        (7500, 8500): 13.0,   # Premium
        (8500, 9500): 16.0,   # Stars
        (9500, 11000): 22.0,  # Elite - BOOSTED (was 20.0)
    }

    # Multipliers
    pp1_boost: float = 1.5           # +50% for PP1 players (was 1.4)
    pp2_boost: float = 1.15          # +15% for PP2 players
    line1_boost: float = 1.25        # +25% for Line 1 players (was 1.2)
    confirmed_goalie_boost: float = 1.8  # +80% for confirmed starters (was 1.5)
    unconfirmed_goalie_penalty: float = 0.02  # 98% reduction - essentially 0% (was 0.1)

    # Value adjustments - BOOSTED for mid-salary value plays
    high_value_boost: float = 1.5    # +50% for top value plays (was 1.3)
    elite_value_boost: float = 1.8   # +80% for elite value (NEW)
    low_value_penalty: float = 0.6   # -40% for poor value (was 0.7)

    # Projection adjustments
    high_proj_boost: float = 1.3     # +30% for high projections (was 1.25)

    # Smash spot boost (high-value player in soft matchup)
    smash_spot_boost: float = 1.4    # +40% for smash spots (NEW)

    # Vegas implied team total multipliers (Feature 1)
    vegas_high_team_total_boost: float = 1.3     # implied_total >= 3.5
    vegas_mid_team_total_boost: float = 1.15     # implied_total >= 3.0
    vegas_low_team_total_penalty: float = 0.8    # implied_total < 2.5
    vegas_high_game_total_boost: float = 1.15    # game_total >= 6.5

    # DK average FPTS perceived value multipliers (Feature 2)
    dk_value_elite_boost: float = 1.4   # dk_value_ratio > 4.0
    dk_value_high_boost: float = 1.2    # dk_value_ratio > 3.0
    dk_value_low_penalty: float = 0.8   # dk_value_ratio < 2.0

    # Salary tier scarcity multipliers (Feature 3)
    scarcity_only_option_boost: float = 1.3   # only high-value option in tier
    scarcity_crowded_penalty: float = 0.85    # 3+ equally good options in tier

    # Return-from-injury buzz multipliers (Feature 4)
    injury_return_boost: float = 1.3    # IR player returning today
    injury_dtd_boost: float = 1.1       # DTD player on the slate

    # Individual recent game scoring multipliers (Feature 5)
    recency_blowup_boost: float = 1.5    # last_1_game > 25 FPTS
    recency_hot_boost: float = 1.3       # last_3_avg > 15
    recency_warm_boost: float = 1.15     # last_3_avg > 10
    recency_cold_penalty: float = 0.7    # last_3_avg < 5

    # TOI surge multipliers (Feature 6) — interaction with role
    toi_surge_role_boost: float = 1.5    # TOI surge (>2min) + Line1 + PP1
    toi_surge_pp1_boost: float = 1.25   # TOI surge + PP1 only
    toi_surge_line1_boost: float = 1.15  # TOI surge + Line1 only
    toi_surge_solo_boost: float = 1.1    # TOI surge alone (weak signal)
    toi_drop_penalty: float = 0.85       # TOI drop (>2min below avg) = demotion

    # Composite multiplier safety cap
    composite_cap: float = 5.0    # max composite multiplier
    composite_floor: float = 0.1  # min composite multiplier


class OwnershipRegressionModel:
    """Regression model for ownership prediction (Ridge or TabPFN).

    Trained on historical contest observations. Uses StandardScaler + model.
    Ridge alpha is tuned via leave-one-date-out cross-validation.
    TabPFN requires no hyperparameter tuning.
    """

    FEATURE_COLS = [
        # Core (from projection CSVs)
        'salary', 'projected_fpts', 'dk_avg_fpts', 'floor', 'ceiling', 'edge', 'value',
        # Derived ranks/percentiles
        'salary_rank_in_pos', 'proj_rank_in_pos', 'value_rank_in_pos',
        'salary_pctile', 'proj_pctile',
        'dk_value_ratio',
        'salary_bin',
        # Position one-hot
        'pos_C', 'pos_W', 'pos_D', 'pos_G',
        'is_goalie',
        # Slate context
        'slate_size', 'n_players_at_pos',
        # Conditional (lines JSON, 0 when missing)
        'is_pp1', 'is_pp2', 'is_line1', 'is_d1', 'is_confirmed_goalie',
    ]

    MODEL_PATH = Path(__file__).parent / "backtests" / "ownership_model.pkl"

    def __init__(self, model_type: str = 'ridge'):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.alpha_ = None
        self.model_type = model_type

    def train(self, training_df: pd.DataFrame, alpha: float = None) -> Dict:
        """Train model on training data. Returns metrics dict."""
        from sklearn.preprocessing import StandardScaler

        features = [c for c in self.FEATURE_COLS if c in training_df.columns]
        X = training_df[features].fillna(0).replace([np.inf, -np.inf], 0).values
        y = training_df['pct_drafted'].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == 'tabpfn':
            from tabpfn import TabPFNRegressor
            self.model = TabPFNRegressor(ignore_pretraining_limits=True)
            self.model.fit(X_scaled, y)
            self.alpha_ = None
        else:
            from sklearn.linear_model import Ridge
            if alpha is None:
                alpha = 1.0
            self.model = Ridge(alpha=alpha)
            self.model.fit(X_scaled, y)
            self.alpha_ = alpha

        self.feature_names = features

        y_pred = self.model.predict(X_scaled)
        mae = float(np.abs(y - y_pred).mean())
        rmse = float(np.sqrt(((y - y_pred) ** 2).mean()))
        corr, _ = spearmanr(y, y_pred)

        return {'mae': mae, 'rmse': rmse, 'spearman': float(corr), 'n': len(y),
                'alpha': alpha, 'model_type': self.model_type}

    def cross_validate(self, training_df: pd.DataFrame,
                       alphas: List[float] = None) -> Dict:
        """Leave-one-date-out CV. Returns best alpha and per-fold results.

        Dispatches to Ridge CV (multiple alphas) or TabPFN CV (single pass).
        """
        if self.model_type == 'tabpfn':
            return self._cross_validate_tabpfn(training_df)
        return self._cross_validate_ridge(training_df, alphas)

    def _cross_validate_ridge(self, training_df: pd.DataFrame,
                              alphas: List[float] = None) -> Dict:
        """Ridge LODOCV with alpha grid search."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        if alphas is None:
            alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

        if 'date_label' not in training_df.columns:
            raise ValueError("training_df must have 'date_label' column for LODOCV")

        features = [c for c in self.FEATURE_COLS if c in training_df.columns]
        dates = sorted(training_df['date_label'].unique())

        best_alpha = None
        best_mean_mae = float('inf')
        all_results = {}

        for alpha in alphas:
            fold_results = []
            for held_out in dates:
                train_mask = training_df['date_label'] != held_out
                test_mask = training_df['date_label'] == held_out

                X_train = training_df.loc[train_mask, features].fillna(0).replace([np.inf, -np.inf], 0).values
                y_train = training_df.loc[train_mask, 'pct_drafted'].values
                X_test = training_df.loc[test_mask, features].fillna(0).replace([np.inf, -np.inf], 0).values
                y_test = training_df.loc[test_mask, 'pct_drafted'].values

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                model = Ridge(alpha=alpha)
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)

                mae = float(np.abs(y_test - y_pred).mean())
                rmse = float(np.sqrt(((y_test - y_pred) ** 2).mean()))
                corr, _ = spearmanr(y_test, y_pred)

                fold_results.append({
                    'date': held_out, 'mae': mae, 'rmse': rmse,
                    'spearman': float(corr), 'n': len(y_test),
                })

            mean_mae = np.mean([f['mae'] for f in fold_results])
            all_results[alpha] = {'folds': fold_results, 'mean_mae': float(mean_mae)}

            if mean_mae < best_mean_mae:
                best_mean_mae = mean_mae
                best_alpha = alpha

        return {
            'best_alpha': best_alpha,
            'best_mean_mae': float(best_mean_mae),
            'results_by_alpha': all_results,
        }

    def _cross_validate_tabpfn(self, training_df: pd.DataFrame) -> Dict:
        """TabPFN LODOCV — single pass, no hyperparameters."""
        from tabpfn import TabPFNRegressor
        from sklearn.preprocessing import StandardScaler

        if 'date_label' not in training_df.columns:
            raise ValueError("training_df must have 'date_label' column for LODOCV")

        features = [c for c in self.FEATURE_COLS if c in training_df.columns]
        dates = sorted(training_df['date_label'].unique())

        fold_results = []
        for held_out in dates:
            train_mask = training_df['date_label'] != held_out
            test_mask = training_df['date_label'] == held_out

            X_train = training_df.loc[train_mask, features].fillna(0).replace([np.inf, -np.inf], 0).values
            y_train = training_df.loc[train_mask, 'pct_drafted'].values
            X_test = training_df.loc[test_mask, features].fillna(0).replace([np.inf, -np.inf], 0).values
            y_test = training_df.loc[test_mask, 'pct_drafted'].values

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = TabPFNRegressor(ignore_pretraining_limits=True)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            mae = float(np.abs(y_test - y_pred).mean())
            rmse = float(np.sqrt(((y_test - y_pred) ** 2).mean()))
            corr, _ = spearmanr(y_test, y_pred)

            fold_results.append({
                'date': held_out, 'mae': mae, 'rmse': rmse,
                'spearman': float(corr), 'n': len(y_test),
            })

        mean_mae = np.mean([f['mae'] for f in fold_results])

        return {
            'best_alpha': 'tabpfn',
            'best_mean_mae': float(mean_mae),
            'results_by_alpha': {
                'tabpfn': {'folds': fold_results, 'mean_mae': float(mean_mae)},
            },
        }

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict ownership percentages from feature DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        cols = [c for c in self.feature_names if c in features_df.columns]
        X = features_df.reindex(columns=self.feature_names, fill_value=0).fillna(0)
        X = X.replace([np.inf, -np.inf], 0).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: Path = None):
        """Pickle model + scaler + feature names + model_type."""
        path = path or self.MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'alpha': self.alpha_,
                'model_type': self.model_type,
            }, f)
        print(f"Saved ownership {self.model_type} model to {path}")

    @classmethod
    def load(cls, path: Path = None) -> Optional['OwnershipRegressionModel']:
        """Load from pickle. Returns None if file doesn't exist.

        Backward-compatible: pickles without model_type default to 'ridge'.
        """
        path = path or cls.MODEL_PATH
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            model_type = data.get('model_type', 'ridge')
            obj = cls(model_type=model_type)
            obj.model = data['model']
            obj.scaler = data['scaler']
            obj.feature_names = data['feature_names']
            obj.alpha_ = data.get('alpha')
            return obj
        except Exception as e:
            print(f"Warning: failed to load ownership model from {path}: {e}")
            return None

    def feature_importances(self) -> pd.DataFrame:
        """Return DataFrame of feature coefficients sorted by abs value.

        TabPFN has no .coef_ attribute — returns zeroed DataFrame.
        """
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        if self.model_type == 'tabpfn' or not hasattr(self.model, 'coef_'):
            return pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': 0.0,
                'abs_coef': 0.0,
            })
        coefs = self.model.coef_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefs,
            'abs_coef': np.abs(coefs),
        }).sort_values('abs_coef', ascending=False)
        return df


class OwnershipModel:
    """
    Predict ownership percentages for NHL DFS players.

    Designed for top-heavy GPP structures where differentiation matters.
    """

    def __init__(self, config: OwnershipConfig = None):
        self.config = config or OwnershipConfig()
        self.lines_data = None
        self.confirmed_goalies = None
        self.team_totals = None
        self.team_game_totals = None
        self.injury_data = None
        self.target_date = None
        self.recent_scores = None
        self.toi_surge_map = None

    def set_lines_data(self, lines_data: Dict, confirmed_goalies: Dict = None):
        """Set line combination data for PP1/Line1 boosts."""
        self.lines_data = lines_data
        self.confirmed_goalies = confirmed_goalies or {}

    def set_vegas_data(self, team_totals: Dict[str, float],
                       team_game_totals: Dict[str, float]):
        """Set Vegas implied team totals and game totals for ownership multipliers.

        Args:
            team_totals: team_abbrev -> implied team total (e.g. {'BOS': 3.4})
            team_game_totals: team_abbrev -> game total (e.g. {'BOS': 6.5})
        """
        self.team_totals = team_totals or {}
        self.team_game_totals = team_game_totals or {}

    def set_injury_data(self, injuries: pd.DataFrame, target_date: str):
        """Set injury data for return-from-injury buzz detection."""
        self.injury_data = injuries
        self.target_date = target_date

    def set_recent_scores(self, recent_scores: Dict[int, Dict[str, float]]):
        """Set individual recent game scoring data.

        Args:
            recent_scores: dict mapping player_id -> {
                'last_1_game_fpts': float,
                'last_3_avg_fpts': float,
                'last_5_avg_fpts': float,
            }
        """
        self.recent_scores = recent_scores or {}

    def set_toi_surge_data(self, toi_surge_map: Dict[str, float]):
        """Set per-player TOI delta (recent avg minus season avg, in minutes).

        Args:
            toi_surge_map: player name -> TOI delta in minutes (positive = surge)
        """
        self.toi_surge_map = toi_surge_map or {}

    def _get_base_ownership(self, salary: float) -> float:
        """Get base ownership from salary curve."""
        for (low, high), base_own in self.config.salary_curve.items():
            if low <= salary < high:
                return base_own
        # Default for outliers
        if salary >= 11000:
            return 22.0
        return 3.0

    def _is_pp1(self, player_name: str, team: str) -> bool:
        """Check if player is on PP1."""
        if not self.lines_data or team not in self.lines_data:
            return False

        team_data = self.lines_data.get(team, {})
        pp_units = team_data.get('pp_units', [])

        for pp in pp_units:
            if pp.get('unit') == 1:
                pp1_players = [p.lower() for p in pp.get('players', [])]
                if any(player_name.lower() in p or p in player_name.lower()
                       for p in pp1_players):
                    return True
        return False

    def _is_pp2(self, player_name: str, team: str) -> bool:
        """Check if player is on PP2."""
        if not self.lines_data or team not in self.lines_data:
            return False

        team_data = self.lines_data.get(team, {})
        pp_units = team_data.get('pp_units', [])

        for pp in pp_units:
            if pp.get('unit') == 2:
                pp2_players = [p.lower() for p in pp.get('players', [])]
                if any(player_name.lower() in p or p in player_name.lower()
                       for p in pp2_players):
                    return True
        return False

    def _is_line1(self, player_name: str, team: str) -> bool:
        """Check if player is on Line 1."""
        if not self.lines_data or team not in self.lines_data:
            return False

        team_data = self.lines_data.get(team, {})
        forward_lines = team_data.get('forward_lines', [])

        for line in forward_lines:
            if line.get('line') == 1:
                line1_players = [p.lower() for p in line.get('players', [])]
                if any(player_name.lower() in p or p in player_name.lower()
                       for p in line1_players):
                    return True
        return False

    def _is_d1(self, player_name: str, team: str) -> bool:
        """Check if player is on Defense Pair 1."""
        if not self.lines_data or team not in self.lines_data:
            return False

        team_data = self.lines_data.get(team, {})
        defense_pairs = team_data.get('defense_pairs', [])

        for pair in defense_pairs:
            if pair.get('pair') == 1:
                d1_players = [p.lower() for p in pair.get('players', [])]
                if any(player_name.lower() in p or p in player_name.lower()
                       for p in d1_players):
                    return True
        return False

    def _is_confirmed_goalie(self, player_name: str, team: str) -> bool:
        """
        Check if goalie is confirmed starter.

        CRITICAL: Non-confirmed goalies get essentially 0% ownership.
        Default to NOT confirmed unless explicitly in confirmed_goalies dict.
        """
        if not self.confirmed_goalies:
            return False  # Default to NOT confirmed (was True - caused over-prediction)

        confirmed = self.confirmed_goalies.get(team, '')
        if not confirmed:
            return False

        # Fuzzy match - be generous with name matching
        player_lower = player_name.lower().strip()
        confirmed_lower = confirmed.lower().strip()

        # Check various matching patterns
        if player_lower in confirmed_lower or confirmed_lower in player_lower:
            return True

        # Check last name match
        player_last = player_lower.split()[-1] if player_lower else ''
        confirmed_last = confirmed_lower.split()[-1] if confirmed_lower else ''

        if player_last and confirmed_last and player_last == confirmed_last:
            return True

        return False

    def _build_scarcity_map(self, df: pd.DataFrame) -> Dict[str, float]:
        """Precompute salary tier scarcity multipliers.

        Groups players into $500 salary buckets, counts high-value players
        per bucket, and returns a dict mapping player name -> scarcity multiplier.
        """
        scarcity = {}
        if 'value' not in df.columns:
            return scarcity

        median_value = df['value'].median()
        df_tmp = df.copy()
        df_tmp['salary_bucket'] = (df_tmp['salary'] // 500) * 500

        for bucket, group in df_tmp.groupby('salary_bucket'):
            high_value_players = group[group['value'] > median_value]
            n_high = len(high_value_players)

            for _, row in group.iterrows():
                if row['value'] > median_value:
                    if n_high == 1:
                        scarcity[row['name']] = self.config.scarcity_only_option_boost
                    elif n_high >= 3:
                        scarcity[row['name']] = self.config.scarcity_crowded_penalty
                    else:
                        scarcity[row['name']] = 1.0
                else:
                    scarcity[row['name']] = 1.0

        return scarcity

    def _build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix for regression model from player pool DataFrame.

        Uses self.lines_data and self.confirmed_goalies for line features.
        Returns DataFrame with columns matching OwnershipRegressionModel.FEATURE_COLS.
        """
        feat = df[['name', 'team']].copy()
        pos_col = 'dk_pos' if 'dk_pos' in df.columns else 'position'

        # Normalize positions: LW/RW/L/R -> W
        pos_map = {'LW': 'W', 'RW': 'W', 'L': 'W', 'R': 'W'}
        positions = df[pos_col].map(lambda p: pos_map.get(p, p))

        # Core features
        for col in ['salary', 'projected_fpts', 'dk_avg_fpts', 'floor', 'ceiling', 'edge', 'value']:
            feat[col] = df[col].fillna(0).astype(float) if col in df.columns else 0.0

        # Within-position ranks (ascending rank, 1 = best)
        feat['_pos'] = positions
        for col, rank_name in [('salary', 'salary_rank_in_pos'),
                                ('projected_fpts', 'proj_rank_in_pos'),
                                ('value', 'value_rank_in_pos')]:
            if col in df.columns:
                feat[rank_name] = feat.groupby('_pos')[col].rank(ascending=False, method='min')
            else:
                feat[rank_name] = 0.0

        # Percentile ranks (0-1)
        for col, pct_name in [('salary', 'salary_pctile'), ('projected_fpts', 'proj_pctile')]:
            if col in df.columns:
                feat[pct_name] = feat.groupby('_pos')[col].rank(pct=True)
            else:
                feat[pct_name] = 0.5

        # DK value ratio
        feat['dk_value_ratio'] = np.where(
            feat['salary'] > 0,
            feat['dk_avg_fpts'] / (feat['salary'] / 1000),
            0.0
        )

        # Salary bin: 0=punt, 1=low, 2=mid-low, 3=mid, 4=premium, 5=elite
        def _salary_bin(s):
            if s < 3500: return 0
            if s < 4500: return 1
            if s < 5500: return 2
            if s < 7000: return 3
            if s < 8500: return 4
            return 5
        feat['salary_bin'] = feat['salary'].apply(_salary_bin)

        # Position one-hot + is_goalie
        feat['pos_C'] = (positions == 'C').astype(int)
        feat['pos_W'] = (positions == 'W').astype(int)
        feat['pos_D'] = (positions == 'D').astype(int)
        feat['pos_G'] = (positions == 'G').astype(int)
        feat['is_goalie'] = feat['pos_G']

        # Slate context
        n_teams = df['team'].nunique() if 'team' in df.columns else 0
        feat['slate_size'] = n_teams / 2
        feat['n_players_at_pos'] = feat.groupby('_pos')['_pos'].transform('count')

        # Lines-based features (conditional on lines_data being available)
        feat['is_pp1'] = 0
        feat['is_pp2'] = 0
        feat['is_line1'] = 0
        feat['is_d1'] = 0
        feat['is_confirmed_goalie'] = 0

        if self.lines_data:
            pp1_vals, pp2_vals, line1_vals, d1_vals, cg_vals = [], [], [], [], []
            for i in range(len(df)):
                name = str(df.iloc[i]['name']) if 'name' in df.columns else ''
                team = str(df.iloc[i]['team']) if 'team' in df.columns else ''
                pos = positions.iloc[i] if i < len(positions) else ''
                pp1_vals.append(int(self._is_pp1(name, team)))
                pp2_vals.append(int(self._is_pp2(name, team)))
                line1_vals.append(int(self._is_line1(name, team)))
                d1_vals.append(int(self._is_d1(name, team)))
                cg_vals.append(int(self._is_confirmed_goalie(name, team)) if pos == 'G' else 0)
            feat['is_pp1'] = pp1_vals
            feat['is_pp2'] = pp2_vals
            feat['is_line1'] = line1_vals
            feat['is_d1'] = d1_vals
            feat['is_confirmed_goalie'] = cg_vals

        # Drop helper columns
        feat = feat.drop(columns=['name', 'team', '_pos'], errors='ignore')
        return feat

    def _is_returning_from_injury(self, player_name: str, team: str) -> str:
        """Check if player is returning from injury today.

        Returns:
            'IR_RETURN' if returning from IR today,
            'DTD' if day-to-day on the slate,
            '' otherwise
        """
        if self.injury_data is None or self.injury_data.empty or not self.target_date:
            return ''

        try:
            target = pd.Timestamp(self.target_date)
        except Exception:
            return ''

        player_lower = player_name.lower().strip()
        for _, row in self.injury_data.iterrows():
            inj_name = str(row.get('player_name', '')).lower().strip()
            # Fuzzy name match: check last name or substring
            player_last = player_lower.split()[-1] if player_lower else ''
            inj_last = inj_name.split()[-1] if inj_name else ''
            if not (player_lower in inj_name or inj_name in player_lower or
                    (player_last and inj_last and player_last == inj_last)):
                continue

            status = row.get('injury_status', '')
            return_date = row.get('return_date')

            # IR player returning today
            if status in ('IR', 'IR-LT', 'IR-NR') and pd.notna(return_date):
                ret = pd.Timestamp(return_date)
                if ret.date() == target.date():
                    return 'IR_RETURN'

            # DTD player on the slate
            if status == 'DTD':
                return 'DTD'

        return ''

    def _heuristic_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Original heuristic ownership prediction (12-factor multiplicative model).

        Used as fallback when no trained regression model exists.
        """
        # Calculate position averages for normalization
        pos_col = 'dk_pos' if 'dk_pos' in df.columns else 'position'
        pos_avg_proj = df.groupby(pos_col)['projected_fpts'].mean()
        pos_avg_value = df.groupby(pos_col)['value'].mean() if 'value' in df.columns else None

        # Precompute salary tier scarcity (Feature 3)
        scarcity_map = self._build_scarcity_map(df)

        ownership_predictions = []

        for idx, row in df.iterrows():
            name = row['name']
            team = row['team']
            salary = row['salary']
            position = row.get('dk_pos', row.get('position', 'C'))
            projected = row.get('projected_fpts', 10)
            value = row.get('value', 3)

            # 1. Base ownership from salary
            base_own = self._get_base_ownership(salary)

            # 2. Position/role multipliers
            multiplier = 1.0

            # Cache role checks for reuse in TOI surge (Feature 6)
            is_pp1 = self._is_pp1(name, team)
            is_line1 = self._is_line1(name, team)

            # PP1 boost (biggest ownership driver)
            if is_pp1:
                multiplier *= self.config.pp1_boost
            elif self._is_pp2(name, team):
                multiplier *= self.config.pp2_boost

            # Line 1 boost
            if is_line1:
                multiplier *= self.config.line1_boost

            # 3. Goalie confirmation (critical)
            if position == 'G':
                if self._is_confirmed_goalie(name, team):
                    multiplier *= self.config.confirmed_goalie_boost
                elif self.confirmed_goalies:
                    multiplier *= self.config.unconfirmed_goalie_penalty
                else:
                    if salary >= 8000:
                        multiplier *= 1.3
                    elif salary >= 7000:
                        multiplier *= 1.0
                    else:
                        multiplier *= 0.5

            # 4. Value adjustment
            if pos_avg_value is not None and position in pos_avg_value.index:
                avg_val = pos_avg_value[position]
                if avg_val > 0:
                    value_ratio = value / avg_val
                    if value_ratio > 1.5:
                        multiplier *= self.config.elite_value_boost
                    elif value_ratio > 1.2:
                        multiplier *= self.config.high_value_boost
                    elif value_ratio < 0.8:
                        multiplier *= self.config.low_value_penalty

            # 5. Projection adjustment
            if position in pos_avg_proj.index:
                avg_proj = pos_avg_proj[position]
                if avg_proj > 0:
                    proj_ratio = projected / avg_proj
                    if proj_ratio > 1.3:
                        multiplier *= self.config.high_proj_boost

            # 6. Smash spot detection
            if 3500 <= salary <= 5500 and value > 3.0:
                multiplier *= self.config.smash_spot_boost

            # 7. Vegas implied team total
            if self.team_totals and team in self.team_totals:
                implied_total = self.team_totals[team]
                if implied_total >= 3.5:
                    multiplier *= self.config.vegas_high_team_total_boost
                elif implied_total >= 3.0:
                    multiplier *= self.config.vegas_mid_team_total_boost
                elif implied_total < 2.5:
                    multiplier *= self.config.vegas_low_team_total_penalty
            if self.team_game_totals and team in self.team_game_totals:
                game_total = self.team_game_totals[team]
                if game_total is not None and game_total >= 6.5:
                    multiplier *= self.config.vegas_high_game_total_boost

            # 8. DK average FPTS perceived value
            dk_avg = row.get('dk_avg_fpts', 0)
            if dk_avg and salary > 0:
                dk_value_ratio = dk_avg / (salary / 1000)
                if dk_value_ratio > 4.0:
                    multiplier *= self.config.dk_value_elite_boost
                elif dk_value_ratio > 3.0:
                    multiplier *= self.config.dk_value_high_boost
                elif dk_value_ratio < 2.0:
                    multiplier *= self.config.dk_value_low_penalty

            # 9. Salary tier scarcity
            if name in scarcity_map:
                multiplier *= scarcity_map[name]

            # 10. Return-from-injury buzz
            injury_status = self._is_returning_from_injury(name, team)
            if injury_status == 'IR_RETURN':
                multiplier *= self.config.injury_return_boost
            elif injury_status == 'DTD':
                multiplier *= self.config.injury_dtd_boost

            # 11. Individual recent game scoring
            player_id = row.get('player_id')
            if self.recent_scores and player_id and player_id in self.recent_scores:
                scores = self.recent_scores[player_id]
                last_1 = scores.get('last_1_game_fpts', 0)
                last_3 = scores.get('last_3_avg_fpts', 0)
                recency_mult = 1.0
                if last_1 > 25:
                    recency_mult = max(recency_mult, self.config.recency_blowup_boost)
                if last_3 > 15:
                    recency_mult = max(recency_mult, self.config.recency_hot_boost)
                elif last_3 > 10:
                    recency_mult = max(recency_mult, self.config.recency_warm_boost)
                elif last_3 < 5:
                    recency_mult = min(recency_mult, self.config.recency_cold_penalty)
                multiplier *= recency_mult

            # 12. TOI surge — interaction with role
            if self.toi_surge_map:
                toi_delta = self.toi_surge_map.get(name, 0)
                if toi_delta > 2.0:
                    if is_pp1 and is_line1:
                        multiplier *= self.config.toi_surge_role_boost
                    elif is_pp1:
                        multiplier *= self.config.toi_surge_pp1_boost
                    elif is_line1:
                        multiplier *= self.config.toi_surge_line1_boost
                    else:
                        multiplier *= self.config.toi_surge_solo_boost
                elif toi_delta < -2.0:
                    multiplier *= self.config.toi_drop_penalty

            # Apply composite multiplier cap
            multiplier = max(self.config.composite_floor,
                             min(self.config.composite_cap, multiplier))

            predicted_own = base_own * multiplier
            predicted_own = max(0.5, min(45.0, predicted_own))

            ownership_predictions.append(predicted_own)

        df['predicted_ownership'] = ownership_predictions
        return df

    def predict_ownership(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """
        Predict ownership for all players in the pool.

        Tries regression model first; falls back to heuristic if no pickle exists.

        Args:
            player_pool: DataFrame with columns: name, team, position, salary,
                        projected_fpts, value, dk_avg_fpts

        Returns:
            DataFrame with added 'predicted_ownership' and 'leverage_score' columns
        """
        df = player_pool.copy()

        # Try regression path
        reg_model = OwnershipRegressionModel.load()
        if reg_model is not None:
            try:
                features_df = self._build_feature_matrix(df)
                preds = reg_model.predict(features_df)
                df['predicted_ownership'] = np.clip(preds, 0.1, 50.0)
                print("  [Ownership] Using regression model")
            except Exception as e:
                print(f"  [Ownership] Regression failed ({e}), falling back to heuristic")
                df = self._heuristic_predict(df)
        else:
            print("  [Ownership] No regression model found, using heuristic")
            df = self._heuristic_predict(df)

        # Normalize so total ownership is reasonable
        df = self._normalize_ownership(df)

        # Calculate leverage score (high projection + low ownership = good leverage)
        df['leverage_score'] = df['projected_fpts'] / (df['predicted_ownership'] + 1)

        # Ownership tier labels
        df['ownership_tier'] = df['predicted_ownership'].apply(self._get_ownership_tier)

        return df

    def _normalize_ownership(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ownership to realistic contest levels.

        Total ownership must sum to 900% (9 roster spots).
        Each position has a specific number of slots:
        - C: 2 slots = 200%
        - W (LW+RW): 3 slots = 300%
        - D: 2 slots = 200%
        - G: 1 slot = 100%
        - UTIL: 1 slot = 100% (can be any skater)

        We normalize within each position group to hit these targets.
        """
        df = df.copy()
        pos_col = 'dk_pos' if 'dk_pos' in df.columns else 'position'

        # Position slot allocations (how much total ownership per position)
        # UTIL is typically filled by best remaining skater, so we add it to skater pool
        position_targets = {
            'C': 200.0,      # 2 C slots
            'LW': 150.0,     # 1.5 W slots (3 W total, split)
            'RW': 150.0,     # 1.5 W slots
            'D': 200.0,      # 2 D slots
            'G': 100.0,      # 1 G slot
            'UTIL': 100.0,   # UTIL goes to highest value skater
        }

        # For positions we have, normalize to target
        for pos in df[pos_col].unique():
            pos_mask = df[pos_col] == pos
            pos_df = df[pos_mask]

            if len(pos_df) == 0:
                continue

            # Get target for this position
            target = position_targets.get(pos, 100.0)

            # Current sum for this position
            current_sum = pos_df['predicted_ownership'].sum()

            if current_sum > 0:
                scale_factor = target / current_sum
                df.loc[pos_mask, 'predicted_ownership'] = (
                    df.loc[pos_mask, 'predicted_ownership'] * scale_factor
                )

        # Ensure min/max bounds (after normalization)
        # Min 0.1% for non-confirmed goalies, max 35% for chalk
        df['predicted_ownership'] = df['predicted_ownership'].clip(0.1, 35.0)

        # Re-normalize slightly if clipping changed totals significantly
        total = df['predicted_ownership'].sum()
        if abs(total - 900) > 50:  # If off by more than 50%, adjust
            df['predicted_ownership'] = df['predicted_ownership'] * (900 / total)

        return df

    def _get_ownership_tier(self, ownership: float) -> str:
        """Categorize ownership into tiers."""
        if ownership >= 20:
            return 'Chalk (20%+)'
        elif ownership >= 15:
            return 'Popular (15-20%)'
        elif ownership >= 10:
            return 'Moderate (10-15%)'
        elif ownership >= 5:
            return 'Low (5-10%)'
        else:
            return 'Contrarian (<5%)'

    def get_leverage_plays(self, df: pd.DataFrame,
                          min_projection: float = 12.0,
                          max_ownership: float = 10.0,
                          top_n: int = 20) -> pd.DataFrame:
        """
        Find best leverage plays (high upside, low ownership).

        These are the GPP-winning differentiators in top-heavy structures.
        """
        leverage = df[
            (df['projected_fpts'] >= min_projection) &
            (df['predicted_ownership'] <= max_ownership)
        ].copy()

        leverage = leverage.sort_values('leverage_score', ascending=False)

        return leverage.head(top_n)

    def get_chalk_plays(self, df: pd.DataFrame,
                        min_ownership: float = 15.0) -> pd.DataFrame:
        """Get high-ownership chalk plays (potential fades in GPPs)."""
        return df[df['predicted_ownership'] >= min_ownership].sort_values(
            'predicted_ownership', ascending=False
        )

    def calculate_lineup_ownership(self, lineup: pd.DataFrame) -> Dict:
        """Calculate total ownership metrics for a lineup."""
        if 'predicted_ownership' not in lineup.columns:
            return {'error': 'No ownership predictions in lineup'}

        total_own = lineup['predicted_ownership'].sum()
        avg_own = lineup['predicted_ownership'].mean()
        max_own = lineup['predicted_ownership'].max()
        min_own = lineup['predicted_ownership'].min()

        # Count players by tier
        tier_counts = lineup['ownership_tier'].value_counts().to_dict()

        return {
            'total_ownership': total_own,
            'avg_ownership': avg_own,
            'max_single_ownership': max_own,
            'min_single_ownership': min_own,
            'tier_breakdown': tier_counts,
            'leverage_rating': 'High' if avg_own < 8 else 'Medium' if avg_own < 12 else 'Low'
        }


def analyze_historical_ownership(contest_files: List[str]) -> pd.DataFrame:
    """
    Analyze historical ownership from contest result files.

    Args:
        contest_files: List of paths to DK contest CSV files

    Returns:
        DataFrame with aggregated ownership statistics
    """
    all_data = []

    for file in contest_files:
        try:
            df = pd.read_csv(file)
            if '%Drafted' in df.columns:
                ownership = df[['Player', 'Roster Position', '%Drafted', 'FPTS']].dropna()
                ownership = ownership.drop_duplicates(subset=['Player'])
                ownership['%Drafted'] = ownership['%Drafted'].str.replace('%', '').astype(float)
                ownership['contest'] = file
                all_data.append(ownership)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def build_ownership_training_data() -> pd.DataFrame:
    """Build training DataFrame from historical contest + projection data.

    Matches ~1,200 player observations across 6 dates with actual %Drafted.
    Returns DataFrame with all regression features plus 'pct_drafted' target
    and 'date_label' for LODOCV.
    """
    from lines import fuzzy_match as _fm

    project_dir = Path(__file__).parent
    contests_dir = project_dir / CONTESTS_DIR
    proj_dir = project_dir / DAILY_PROJECTIONS_DIR

    # 6 matchable dates: (date_label, contest_csv, projection_csv, lines_json_or_None)
    TRAINING_DATES = [
        ('jan23', '$5main_NHL1.23.26.csv',
         '01_23_26NHLprojections_20260123_190750.csv', None),
        ('jan26', '$5SE_NHL1.26.26.csv',
         '01_26_26NHLprojections_20260126_184134.csv', None),
        ('jan28', '$5SE_NHL1.28.26.csv',
         '01_28_26NHLprojections_20260128_191024.csv', None),
        ('jan29', '$1SE_NHL_1.29.26.csv',
         '01_29_26NHLprojections_20260129_184650.csv', 'lines_2026_01_29.json'),
        ('jan31', '$5SE_NHL1.31.26.csv',
         '01_31_26NHLprojections_20260131_190255.csv', 'lines_2026_01_31.json'),
        ('feb01', '$5SE_NHL2.1.26.csv',
         '02_01_26NHLprojections_20260201_140426.csv', 'lines_2026_02_01.json'),
    ]

    all_frames = []

    for date_label, contest_file, proj_file, lines_file in TRAINING_DATES:
        contest_path = contests_dir / contest_file
        proj_path = proj_dir / proj_file

        if not contest_path.exists():
            print(f"  Warning: missing contest file {contest_path}, skipping {date_label}")
            continue
        if not proj_path.exists():
            print(f"  Warning: missing projection file {proj_path}, skipping {date_label}")
            continue

        # --- Load contest CSV and extract unique Player + %Drafted ---
        contest_df = pd.read_csv(contest_path)
        if '%Drafted' not in contest_df.columns or 'Player' not in contest_df.columns:
            print(f"  Warning: contest CSV missing required columns, skipping {date_label}")
            continue

        # Extract unique players with their %Drafted
        ownership_df = contest_df[['Player', '%Drafted']].dropna().drop_duplicates(subset=['Player'])
        ownership_df = ownership_df.copy()
        ownership_df['%Drafted'] = (
            ownership_df['%Drafted'].astype(str).str.replace('%', '').astype(float)
        )
        ownership_df = ownership_df.rename(columns={'Player': 'contest_name', '%Drafted': 'pct_drafted'})

        # --- Load projection CSV ---
        proj_df = pd.read_csv(proj_path)

        # Normalize positions
        pos_col = 'dk_pos' if 'dk_pos' in proj_df.columns else 'position'
        pos_map = {'LW': 'W', 'RW': 'W', 'L': 'W', 'R': 'W'}
        proj_df['norm_pos'] = proj_df[pos_col].map(lambda p: pos_map.get(p, p))

        # --- Load lines JSON if available ---
        lines_data = None
        confirmed_goalies = {}
        if lines_file:
            lines_path = proj_dir / lines_file
            if lines_path.exists():
                import json
                with open(lines_path, 'r') as f:
                    lines_data = json.load(f)
                # Extract confirmed goalies
                for team_abbrev, team_data in lines_data.items():
                    cg = team_data.get('confirmed_goalie', '')
                    if cg:
                        confirmed_goalies[team_abbrev] = cg

        # --- Match players by name ---
        proj_names = proj_df['name'].tolist()
        matched_rows = []

        for _, own_row in ownership_df.iterrows():
            contest_name = own_row['contest_name']
            pct_drafted = own_row['pct_drafted']

            # Exact match first
            exact = proj_df[proj_df['name'] == contest_name]
            if len(exact) == 1:
                matched_rows.append((exact.index[0], pct_drafted))
                continue

            # Fuzzy match fallback
            match = None
            best_ratio = 0
            from difflib import SequenceMatcher
            for i, pname in enumerate(proj_names):
                if _fm(contest_name, pname):
                    ratio = SequenceMatcher(None, contest_name.lower(), pname.lower()).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        match = proj_df.index[i]

            if match is not None:
                matched_rows.append((match, pct_drafted))

        if not matched_rows:
            print(f"  Warning: no players matched for {date_label}")
            continue

        # Build matched DataFrame (reset index to avoid duplicates)
        match_indices, match_pcts = zip(*matched_rows)
        date_df = proj_df.loc[list(match_indices)].copy().reset_index(drop=True)
        date_df['pct_drafted'] = list(match_pcts)
        date_df['date_label'] = date_label

        # --- Build features using OwnershipModel ---
        om = OwnershipModel()
        if lines_data:
            om.set_lines_data(lines_data, confirmed_goalies)

        features = om._build_feature_matrix(date_df)
        features['pct_drafted'] = date_df['pct_drafted'].values
        features['date_label'] = date_label

        all_frames.append(features)
        print(f"  {date_label}: {len(features)} players matched "
              f"(lines={'yes' if lines_data else 'no'})")

    if not all_frames:
        print("No training data built.")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal training data: {len(result)} observations across {len(all_frames)} dates")
    return result


def print_ownership_report(df: pd.DataFrame, top_n: int = 20):
    """Print a formatted ownership report."""
    print("\n" + "=" * 70)
    print("OWNERSHIP PROJECTIONS REPORT")
    print("=" * 70)

    # Top projected ownership
    print(f"\nTop {top_n} Projected Ownership:")
    print("-" * 70)
    cols = ['name', 'team', 'dk_pos', 'salary', 'projected_fpts',
            'predicted_ownership', 'ownership_tier']
    cols = [c for c in cols if c in df.columns]

    top_owned = df.nlargest(top_n, 'predicted_ownership')[cols]
    for _, row in top_owned.iterrows():
        pos = row.get('dk_pos', row.get('position', '?'))
        print(f"  {row['name']:<25} {row['team']:<4} {pos:<3} "
              f"${row['salary']:<6,} {row['projected_fpts']:>5.1f} pts  "
              f"{row['predicted_ownership']:>5.1f}% ({row.get('ownership_tier', '')})")

    # Leverage plays
    print(f"\nTop {top_n} Leverage Plays (High Proj, Low Own):")
    print("-" * 70)
    leverage = df[df['projected_fpts'] >= 10].nlargest(top_n, 'leverage_score')[cols + ['leverage_score']]
    for _, row in leverage.iterrows():
        pos = row.get('dk_pos', row.get('position', '?'))
        print(f"  {row['name']:<25} {row['team']:<4} {pos:<3} "
              f"${row['salary']:<6,} {row['projected_fpts']:>5.1f} pts  "
              f"{row['predicted_ownership']:>5.1f}% | Lev: {row['leverage_score']:.2f}")

    # Ownership distribution
    print("\nOwnership Distribution:")
    print("-" * 70)
    tier_dist = df['ownership_tier'].value_counts()
    for tier, count in tier_dist.items():
        pct = 100 * count / len(df)
        print(f"  {tier:<20} {count:>4} players ({pct:>5.1f}%)")


# Quick test
if __name__ == "__main__":
    from data_pipeline import NHLDataPipeline
    from projections import NHLProjectionModel
    from main import load_dk_salaries, merge_projections_with_salaries
    from lines import LinesScraper, StackBuilder
    from datetime import datetime

    print("Building Ownership Projections...")
    print("=" * 70)

    # Build projections
    pipeline = NHLDataPipeline()
    data = pipeline.build_projection_dataset(include_game_logs=False)
    model = NHLProjectionModel()
    today = datetime.now().strftime('%Y-%m-%d')
    projections = model.generate_projections(data, target_date=today)

    # Fetch lines for confirmed goalies and PP units
    print("\nFetching line combinations...")
    schedule = pipeline.fetch_schedule(today)
    teams_playing = set()
    teams_playing.update(schedule['home_team'].tolist())
    teams_playing.update(schedule['away_team'].tolist())
    teams_playing = sorted([t for t in teams_playing if t])

    scraper = LinesScraper()
    all_lines = scraper.get_multiple_teams(teams_playing)
    stack_builder = StackBuilder(all_lines)
    confirmed_goalies = stack_builder.get_all_starting_goalies()

    # Load salaries and merge (daily_salaries/ first, then project root)
    project_dir = Path(__file__).parent
    salaries_dir = project_dir / DAILY_SALARIES_DIR
    salary_files = list(salaries_dir.glob('DKSalaries*.csv')) if salaries_dir.exists() else []
    if not salary_files:
        salary_files = list(project_dir.glob('DKSalaries*.csv'))
    salary_files = sorted(salary_files)
    if salary_files:
        salary_file = str(salary_files[-1])
        dk_salaries = load_dk_salaries(salary_file)
        dk_skaters = dk_salaries[dk_salaries['position'].isin(['C', 'LW', 'RW', 'D'])]
        dk_goalies = dk_salaries[dk_salaries['position'] == 'G']

        projections['goalies']['position'] = 'G'

        skaters_merged = merge_projections_with_salaries(projections['skaters'], dk_skaters, 'skater')
        goalies_merged = merge_projections_with_salaries(projections['goalies'], dk_goalies, 'goalie')

        # Filter to confirmed starting goalies only
        if confirmed_goalies:
            from lines import fuzzy_match as _fm
            confirmed_names = list(confirmed_goalies.values())
            def _is_confirmed(name):
                return any(_fm(name, cn) for cn in confirmed_names)
            before = len(goalies_merged)
            goalies_merged = goalies_merged[goalies_merged['name'].apply(_is_confirmed)]
            filtered_count = before - len(goalies_merged)
            if filtered_count > 0:
                print(f"  Filtered {filtered_count} non-confirmed goalies from pool")

        player_pool = pd.concat([skaters_merged, goalies_merged], ignore_index=True)

        # Run ownership model
        ownership_model = OwnershipModel()
        ownership_model.set_lines_data(all_lines, confirmed_goalies)

        player_pool = ownership_model.predict_ownership(player_pool)

        # Print report
        print_ownership_report(player_pool)

        # Analyze historical for comparison
        print("\n" + "=" * 70)
        print("HISTORICAL OWNERSHIP COMPARISON")
        print("=" * 70)
        contests_dir = project_dir / CONTESTS_DIR
        contest_files = list(contests_dir.glob('$*main_NHL*.csv')) if contests_dir.exists() else []
        if not contest_files:
            contest_files = list(project_dir.glob('$*main_NHL*.csv'))
        contest_files = [str(p) for p in contest_files]
        if contest_files:
            historical = analyze_historical_ownership(contest_files)
            print(f"Analyzed {len(contest_files)} contests, {len(historical)} player observations")
            print(f"Historical avg ownership: {historical['%Drafted'].mean():.2f}%")
            print(f"Predicted avg ownership: {player_pool['predicted_ownership'].mean():.2f}%")
    else:
        print("No salary file found. Add DKSalaries*.csv to daily_salaries/ or project folder.")
