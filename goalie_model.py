#!/usr/bin/env python3
"""
Goalie Projection Model for NHL DFS
======================================

DK Scoring: Win(+6), Save(+0.7), GA(-3.5), Shutout(+3), OTL(+2)

Key findings from 1,836 starter-games across 113 slates:
  - DK Ceiling (r=0.49) encodes confirmed starter status
  - Individual rolling avg (r=0.09) provides quality signal  
  - Win probability adds small directional edge
  - Vegas lines have near-zero correlation with goalie FPTS

Model: Ridge regression on rolling features (no data leakage)
  MAE: 5.58 (vs 7.16 DK Avg baseline, 22% improvement)
  Selection: +2.2 FPTS/lineup over DK Avg pick

Usage:
    from goalie_model import GoalieModel, get_rolling_history
    model = GoalieModel()
    pool = model.project_goalies(goalie_df, goalie_history)
"""

import json
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MODEL_PATH = Path(__file__).parent / "data" / "goalie_model_coefficients.json"

DEFAULT_COEFFICIENTS = {
    'win_pct': 1.005,
    'Ceiling': 0.344,
    'OppGoal': 0.253,
    'goalie_rolling_avg': -0.241,
    'Avg': 0.096,
    'team_rolling_avg': -0.007,
}
DEFAULT_INTERCEPT = 3.793


def _make_goalie_key(name: str, team: str) -> str:
    return str(name).strip().lower() + '_' + str(team).strip().lower()


def get_rolling_history(history: dict, key: str, before_date: str, n: int = 15) -> List[float]:
    if key not in history:
        return []
    return [s for d, s in history[key] if d < before_date][-n:]


class GoalieModel:

    def __init__(self):
        self.coefficients = DEFAULT_COEFFICIENTS.copy()
        self.intercept = DEFAULT_INTERCEPT
        try:
            with open(MODEL_PATH) as f:
                data = json.load(f)
            self.coefficients = data.get('coefficients', self.coefficients)
            self.intercept = data.get('intercept', self.intercept)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def predict(self, goalie_row: dict,
                goalie_history: Dict[str, List[float]] = None,
                team_history: Dict[str, List[float]] = None) -> float:
        features = {}
        features['Avg'] = float(goalie_row.get('Avg', goalie_row.get('dk_avg_fpts', 10)))
        features['Ceiling'] = float(goalie_row.get('Ceiling', goalie_row.get('ceiling', 0)))
        features['win_pct'] = float(goalie_row.get('win_pct', goalie_row.get('Win %', 0.5)))
        features['OppGoal'] = float(goalie_row.get('OppGoal', goalie_row.get('opp_implied_total', 3.0)))

        g_key = _make_goalie_key(
            goalie_row.get('name', goalie_row.get('Player', '')),
            goalie_row.get('team', goalie_row.get('Team', '')))

        if goalie_history and g_key in goalie_history:
            recent = goalie_history[g_key][-15:]
            features['goalie_rolling_avg'] = np.mean(recent) if recent else features['Avg']
        else:
            features['goalie_rolling_avg'] = features['Avg']

        team = goalie_row.get('team', goalie_row.get('Team', ''))
        if team_history and team in team_history:
            recent = team_history[team][-20:]
            features['team_rolling_avg'] = np.mean(recent) if recent else 10.0
        else:
            features['team_rolling_avg'] = 10.0

        pred = self.intercept
        for feat, coef in self.coefficients.items():
            pred += coef * features.get(feat, 0)

        return max(0, pred)

    def project_goalies(self, goalies_df: pd.DataFrame,
                        goalie_history: Dict[str, List[float]] = None,
                        team_history: Dict[str, List[float]] = None,
                        verbose: bool = True) -> pd.DataFrame:
        df = goalies_df.copy()
        projections = []
        for _, row in df.iterrows():
            proj = self.predict(row.to_dict(), goalie_history, team_history)
            projections.append(proj)

        df['goalie_model_proj'] = projections

        if 'projected_fpts' in df.columns:
            df['projected_fpts_pre_goalie'] = df['projected_fpts'].copy()
            df['projected_fpts'] = 0.6 * df['goalie_model_proj'] + 0.4 * df['projected_fpts']
        else:
            df['projected_fpts'] = df['goalie_model_proj']

        if verbose and len(df) > 0:
            print(f"\n  Goalie Model Projections ({len(df)} goalies):")
            top = df.nlargest(min(5, len(df)), 'goalie_model_proj')
            for _, r in top.iterrows():
                name = r.get('name', r.get('Player', '?'))
                team = r.get('team', r.get('Team', '?'))
                proj = r['goalie_model_proj']
                old = r.get('projected_fpts_pre_goalie', proj)
                print(f"    {name:<22} {team:<4} model={proj:.1f} (was {old:.1f})")

        return df

    def select_best_goalie(self, goalies_df: pd.DataFrame) -> Tuple[str, float]:
        col = 'goalie_model_proj' if 'goalie_model_proj' in goalies_df.columns else 'projected_fpts'
        if col not in goalies_df.columns:
            return None, 0
        best = goalies_df.loc[goalies_df[col].idxmax()]
        return best.get('name', best.get('Player', '?')), best[col]


def build_goalie_history_from_season(dk_season_dir: str) -> Tuple[Dict, Dict]:
    """Build rolling histories from DK season files for backtesting."""
    files = sorted(glob.glob(f"{dk_season_dir}/draftkings_NHL_*.csv"))

    goalie_history = {}
    team_history = {}

    for f in files:
        date = os.path.basename(f).replace('draftkings_NHL_', '').replace('_players.csv', '')
        df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
        goalies = df[df['Pos'] == 'G'].copy()
        goalies['Score'] = pd.to_numeric(goalies['Score'], errors='coerce')

        for _, row in goalies[goalies['Score'].notna() & (goalies['Score'] > 0)].iterrows():
            key = _make_goalie_key(row['Player'], row['Team'])
            team = row['Team']

            goalie_history.setdefault(key, []).append((date, float(row['Score'])))
            team_history.setdefault(team, []).append((date, float(row['Score'])))

    return goalie_history, team_history
