#!/usr/bin/env python3
"""
Line Context Model for NHL DFS Projections
=============================================

Adjusts player projections based on line assignment, PP unit, and game
environment (team implied total, opponent implied GA).

Trained on 28,526 skater-games across 113 DK slates (Oct 2025-Feb 2026).

KEY FINDING: The biggest edge is in the INTERACTION between PP1 status
and high team implied totals. A PP1 player on a team with 3.5+ implied
total scores +1.5 FPTS above their DK average. DK prices in PP1 status
but underprices the upside in high-scoring game environments.

Adjustment magnitudes (from Ridge model, α=10):
  PP1 + High Total (3.5+):  +1.5 to +1.8 FPTS
  PP1 + Mid Total (3.0):    +0.1 to +0.3 FPTS
  PP1 + Low Total (<2.75):  +0.5 to +0.8 FPTS
  PP2 + Any Total:          -0.5 to +0.4 FPTS
  NoPP + Any Total:         -0.8 to +0.3 FPTS
  Line4 + NoPP:             -0.4 to -0.8 FPTS

Usage:
    from line_model import apply_line_adjustments
    
    # In main pipeline, after projections are built:
    skaters = apply_line_adjustments(skaters, team_totals)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

MODEL_PATH = Path(__file__).parent / "data" / "line_model_coefficients.json"
LOOKUP_PATH = Path(__file__).parent / "data" / "line_adjustment_lookup.json"

# ═══════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════

_MODEL = None
_LOOKUP = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        try:
            with open(MODEL_PATH) as f:
                _MODEL = json.load(f)
        except FileNotFoundError:
            _MODEL = {}
    return _MODEL


def _load_lookup():
    global _LOOKUP
    if _LOOKUP is None:
        try:
            with open(LOOKUP_PATH) as f:
                _LOOKUP = json.load(f)
        except FileNotFoundError:
            _LOOKUP = {}
    return _LOOKUP


# ═══════════════════════════════════════════════════════════════════
#  Core: Compute Adjustment for a Single Player
# ═══════════════════════════════════════════════════════════════════

def compute_adjustment(line_num: int, pp_unit: str, team_total: float,
                       opp_total: float, salary: float = 4000) -> float:
    """
    Compute FPTS adjustment for a player based on line context.
    
    Args:
        line_num: Even-strength line number (1-4)
        pp_unit: Power play unit ('1', '2', '' or None)
        team_total: Team implied goal total from Vegas
        opp_total: Opponent implied goal total from Vegas
        salary: Player DK salary
    
    Returns:
        FPTS adjustment to add to base projection
    """
    model = _load_model()
    if not model or 'coefficients' not in model:
        return 0.0
    
    coefs = model['coefficients']
    intercept = model['intercept']
    
    # Build feature vector
    is_pp1 = int(str(pp_unit).strip() in ('1', '1.0'))
    is_pp2 = int(str(pp_unit).strip() in ('2', '2.0'))
    no_pp = int(not is_pp1 and not is_pp2)
    game_total = team_total + opp_total
    high_total = int(team_total >= 3.5)
    low_total = int(team_total < 2.75)
    
    features = {
        'line_num': line_num,
        'is_pp1': is_pp1,
        'is_pp2': is_pp2,
        'no_pp': no_pp,
        'TeamGoal': team_total,
        'OppGoal': opp_total,
        'Total': game_total,
        'pp1_x_total': is_pp1 * team_total,
        'pp1_x_opp_ga': is_pp1 * opp_total,
        'pp2_x_total': is_pp2 * team_total,
        'line1_x_total': int(line_num == 1) * team_total,
        'line_x_pp1': line_num * is_pp1,
        'pp1_high_total': is_pp1 * high_total,
        'pp1_low_total': is_pp1 * low_total,
        'high_total': high_total,
        'low_total': low_total,
        'salary_k': salary / 1000,
    }
    
    adj = intercept
    for feat, val in features.items():
        adj += coefs.get(feat, 0) * val
    
    return adj


def compute_adjustment_fast(line_num: int, pp_unit: str,
                            team_total: float) -> float:
    """
    Fast lookup-based adjustment (no model computation).
    Uses pre-computed lookup table.
    """
    lookup = _load_lookup()
    if not lookup:
        return 0.0
    
    pp_label = 'NoPP'
    if str(pp_unit).strip() in ('1', '1.0'):
        pp_label = 'PP1'
    elif str(pp_unit).strip() in ('2', '2.0'):
        pp_label = 'PP2'
    
    if team_total >= 3.25:
        total_bucket = 'high'
    elif team_total < 2.75:
        total_bucket = 'low'
    else:
        total_bucket = 'mid'
    
    line_n = max(1, min(4, int(line_num))) if pd.notna(line_num) else 2
    key = f"L{line_n}_{pp_label}_{total_bucket}"
    
    return lookup.get(key, 0.0)


# ═══════════════════════════════════════════════════════════════════
#  Pipeline Integration: Apply to Full Player Pool
# ═══════════════════════════════════════════════════════════════════

def apply_line_adjustments(skaters_df: pd.DataFrame,
                           team_totals: Dict[str, float] = None,
                           opp_totals: Dict[str, float] = None,
                           use_full_model: bool = True,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Apply line context adjustments to skater projections.
    
    Args:
        skaters_df: DataFrame with columns: name, team, position, salary,
                    projected_fpts, and optionally start_line, pp_unit
        team_totals: Dict of team -> implied total (e.g. {'COL': 3.5})
        opp_totals: Dict of team -> opponent implied total
        use_full_model: Use Ridge model (True) or fast lookup (False)
        verbose: Print adjustment summary
    
    Returns:
        DataFrame with adjusted projected_fpts
    """
    df = skaters_df.copy()
    
    if team_totals is None:
        team_totals = {}
    if opp_totals is None:
        opp_totals = {}
    
    # Determine line/PP columns
    line_col = None
    pp_col = None
    for c in ['start_line', 'Start/Line', 'line', 'ev_line']:
        if c in df.columns:
            line_col = c
            break
    for c in ['pp_unit', 'PP Unit', 'pp']:
        if c in df.columns:
            pp_col = c
            break
    
    if line_col is None and pp_col is None:
        if verbose:
            print("  Line model: No line/PP data available — skipping")
        return df
    
    # Compute adjustments
    adjustments = []
    adj_fn = compute_adjustment if use_full_model else compute_adjustment_fast
    
    for idx, row in df.iterrows():
        line_num = pd.to_numeric(row.get(line_col, 2), errors='coerce')
        if pd.isna(line_num):
            line_num = 2  # Default to line 2
        
        pp = str(row.get(pp_col, '')).strip() if pp_col else ''
        team = row.get('team', '')
        tt = team_totals.get(team, 3.0)
        ot = opp_totals.get(team, 3.0)
        sal = row.get('salary', 4000)
        
        if use_full_model:
            adj = compute_adjustment(int(line_num), pp, tt, ot, sal)
        else:
            adj = compute_adjustment_fast(int(line_num), pp, tt)
        
        adjustments.append(adj)
    
    df['line_adjustment'] = adjustments
    df['projected_fpts_pre_line'] = df['projected_fpts'].copy()
    df['projected_fpts'] = df['projected_fpts'] + df['line_adjustment']
    
    # Recalculate value
    if 'salary' in df.columns:
        df['value'] = df['projected_fpts'] / (df['salary'] / 1000)
    
    if verbose:
        n_adj = (df['line_adjustment'].abs() > 0.01).sum()
        avg_adj = df['line_adjustment'].mean()
        max_up = df['line_adjustment'].max()
        max_down = df['line_adjustment'].min()
        
        print(f"\n  Line Context Adjustments ({n_adj}/{len(df)} players adjusted):")
        print(f"    Avg adjustment:  {avg_adj:+.2f} FPTS")
        print(f"    Max boost:       {max_up:+.2f} FPTS")
        print(f"    Max penalty:     {max_down:+.2f} FPTS")
        
        # Show biggest adjustments
        top_up = df.nlargest(5, 'line_adjustment')
        top_down = df.nsmallest(3, 'line_adjustment')
        
        print(f"\n    Top boosts:")
        for _, r in top_up.iterrows():
            ln = r.get(line_col, '?')
            pp = r.get(pp_col, '')
            tt = team_totals.get(r.get('team', ''), 0)
            print(f"      {r['name']:<22} L{ln} PP{pp or '-'} total={tt:.1f} "
                  f"adj={r['line_adjustment']:+.2f} → {r['projected_fpts']:.1f}")
        
        print(f"    Top penalties:")
        for _, r in top_down.iterrows():
            ln = r.get(line_col, '?')
            print(f"      {r['name']:<22} L{ln} adj={r['line_adjustment']:+.2f}")
    
    return df


# ═══════════════════════════════════════════════════════════════════
#  Goalie Line Context (opponent-focused)
# ═══════════════════════════════════════════════════════════════════

def goalie_opp_adjustment(opp_implied_total: float) -> float:
    """
    Adjust goalie projection based on opponent implied total.
    
    Lower opponent total = fewer shots against = better for goalie (fewer GA)
    but also fewer saves. Net effect from data:
    - Opp total < 2.5: goalie avg 10.2 (low GA, moderate saves)
    - Opp total 2.5-3.0: goalie avg 8.9 (baseline)
    - Opp total 3.0+: goalie avg 7.8 (high GA risk)
    """
    if opp_implied_total < 2.5:
        return +1.3
    elif opp_implied_total < 3.0:
        return 0.0
    elif opp_implied_total < 3.5:
        return -1.0
    else:
        return -2.0
