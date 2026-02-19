"""
Conformal Prediction Calibration for NHL DFS Ensemble
======================================================

Problem: MDN v3 std_fpts may be miscalibrated — the model's predicted
uncertainty intervals may not match actual coverage rates. If std_fpts is
too tight, the Monte Carlo sampler underestimates variance and generates
too-similar lineups. If too wide, it adds noise.

Solution: Conformal prediction computes calibrated scaling factors by position
so that the predicted intervals achieve correct coverage. This is done via
residual-based conformal calibration:

1. Compute residuals: r_i = |actual - predicted| for recent holdout data
2. For each position, compute the quantile adjustment factor:
   scale = quantile(r / std_predicted, 0.90) / z_0.90
   where z_0.90 = 1.282 (normal quantile for 90% coverage)
3. Apply: calibrated_std = std_fpts * scale[position]

If scale > 1: model is overconfident (needs wider intervals)
If scale < 1: model is underconfident (can tighten intervals)

Usage:
    from conformal_calibration import calibrate_std_fpts, compute_calibration_factors

    # Compute factors from backtest residuals
    factors = compute_calibration_factors(backtest_df)

    # Apply to pool
    pool = calibrate_std_fpts(pool, factors)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from scipy.stats import norm

BACKTEST_PATH = Path(__file__).parent / 'mdn_v3_backtest_results.csv'
CALIBRATION_PATH = Path(__file__).parent / 'data' / 'conformal_factors.json'

# Target coverage level (90% prediction interval)
TARGET_COVERAGE = 0.90
Z_TARGET = norm.ppf((1 + TARGET_COVERAGE) / 2)  # 1.6449 for 90% two-sided


def compute_calibration_factors(
    backtest_df: pd.DataFrame = None,
    min_samples: int = 100,
    recent_days: int = 60,
) -> Dict:
    """
    Compute per-position conformal calibration factors from backtest residuals.

    Args:
        backtest_df: DataFrame with actual_fpts, predicted_fpts, std_fpts, position, game_date
        min_samples: Minimum residuals needed per position
        recent_days: Only use this many recent days for calibration (recency weighting)

    Returns:
        Dict with 'factors' (position -> scale), 'coverage' (actual coverage stats),
        and 'overall_factor' (position-agnostic fallback)
    """
    if backtest_df is None:
        if not BACKTEST_PATH.exists():
            print("  Conformal: No backtest data found, using defaults")
            return _default_factors()
        backtest_df = pd.read_csv(BACKTEST_PATH)

    df = backtest_df.copy()

    # Ensure required columns
    required = ['actual_fpts', 'predicted_fpts', 'position']
    for col in required:
        if col not in df.columns:
            print(f"  Conformal: Missing column {col}, using defaults")
            return _default_factors()

    # Use std_fpts if available, otherwise estimate
    if 'std_fpts' not in df.columns:
        df['std_fpts'] = 5.5  # default

    df['std_fpts'] = df['std_fpts'].fillna(5.5).clip(lower=1.0)

    # Filter to valid predictions
    df = df.dropna(subset=['actual_fpts', 'predicted_fpts'])

    # Recency filter
    if 'game_date' in df.columns and recent_days > 0:
        df['game_date'] = pd.to_datetime(df['game_date'])
        cutoff = df['game_date'].max() - pd.Timedelta(days=recent_days)
        df = df[df['game_date'] >= cutoff]

    if len(df) < min_samples:
        print(f"  Conformal: Only {len(df)} samples, using defaults")
        return _default_factors()

    # Compute normalized residuals
    df['residual'] = np.abs(df['actual_fpts'] - df['predicted_fpts'])
    df['norm_residual'] = df['residual'] / df['std_fpts']

    # Per-position calibration
    factors = {}
    coverage = {}

    # Map positions to groups (L/R → W for DFS)
    pos_map = {'L': 'W', 'R': 'W', 'LW': 'W', 'RW': 'W'}
    df['pos_group'] = df['position'].map(lambda p: pos_map.get(p, p))

    for pos in ['C', 'W', 'D', 'G']:
        pos_df = df[df['pos_group'] == pos]
        if len(pos_df) < min_samples // 4:
            # Use overall factor as fallback
            continue

        # Conformal quantile: what scale makes 90% of residuals fall within 90% PI?
        # Target: P(|actual - predicted| <= scale * std * z_target) = target_coverage
        # Equivalent: quantile(|residual| / std, target_coverage) / z_target = scale
        q = np.quantile(pos_df['norm_residual'], TARGET_COVERAGE)
        scale = q / Z_TARGET
        factors[pos] = round(float(scale), 4)

        # Measure actual coverage before calibration
        within_90 = (pos_df['norm_residual'] <= Z_TARGET).mean()
        coverage[pos] = {
            'n': len(pos_df),
            'raw_coverage_90': round(float(within_90), 4),
            'calibration_factor': factors[pos],
            'mean_residual': round(float(pos_df['residual'].mean()), 3),
            'mean_std_fpts': round(float(pos_df['std_fpts'].mean()), 3),
        }

    # Overall factor (fallback)
    overall_q = np.quantile(df['norm_residual'], TARGET_COVERAGE)
    overall_factor = round(float(overall_q / Z_TARGET), 4)

    # Fill missing positions with overall
    for pos in ['C', 'W', 'D', 'G']:
        if pos not in factors:
            factors[pos] = overall_factor

    # Overall coverage
    overall_within = (df['norm_residual'] <= Z_TARGET).mean()
    coverage['overall'] = {
        'n': len(df),
        'raw_coverage_90': round(float(overall_within), 4),
        'overall_factor': overall_factor,
    }

    result = {
        'factors': factors,
        'coverage': coverage,
        'overall_factor': overall_factor,
        'target_coverage': TARGET_COVERAGE,
    }

    # Save to JSON
    import json
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Conformal factors saved to {CALIBRATION_PATH}")

    return result


def _default_factors() -> Dict:
    """Return default (uncalibrated) factors."""
    return {
        'factors': {'C': 1.0, 'W': 1.0, 'D': 1.0, 'G': 1.0},
        'coverage': {},
        'overall_factor': 1.0,
        'target_coverage': TARGET_COVERAGE,
    }


def load_calibration_factors() -> Dict:
    """Load saved calibration factors from JSON."""
    import json
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return _default_factors()


def calibrate_std_fpts(pool: pd.DataFrame, factors: Dict = None) -> pd.DataFrame:
    """
    Apply conformal calibration to std_fpts in a player pool.

    Maps pool positions to factor groups (L/R → W) and scales std_fpts.

    Args:
        pool: Player pool DataFrame with 'std_fpts' and position column
        factors: Calibration factors dict (from compute_calibration_factors)

    Returns:
        Pool with calibrated std_fpts
    """
    if factors is None:
        factors = load_calibration_factors()

    if 'std_fpts' not in pool.columns:
        return pool

    pool = pool.copy()
    factor_map = factors.get('factors', {})
    overall = factors.get('overall_factor', 1.0)

    # Determine position column
    pos_col = None
    for col in ['norm_pos', 'position', 'pos']:
        if col in pool.columns:
            pos_col = col
            break

    if pos_col is None:
        pool['std_fpts'] = pool['std_fpts'] * overall
        return pool

    # Map positions
    pos_mapping = {'L': 'W', 'R': 'W', 'LW': 'W', 'RW': 'W', 'C': 'C', 'D': 'D', 'G': 'G', 'W': 'W'}

    for pos_raw in pool[pos_col].unique():
        pos_group = pos_mapping.get(pos_raw, pos_raw)
        scale = factor_map.get(pos_group, overall)
        mask = pool[pos_col] == pos_raw
        pool.loc[mask, 'std_fpts'] = pool.loc[mask, 'std_fpts'] * scale

    return pool


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    import json

    print("=" * 70)
    print("CONFORMAL CALIBRATION ANALYSIS")
    print("=" * 70)

    # Load backtest results
    if not BACKTEST_PATH.exists():
        print(f"No backtest data at {BACKTEST_PATH}")
        exit(1)

    df = pd.read_csv(BACKTEST_PATH)
    print(f"\nBacktest data: {len(df)} predictions, {df['game_date'].nunique()} dates")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    # Compute calibration factors
    print("\n[1] Computing conformal calibration factors...")
    result = compute_calibration_factors(df, recent_days=60)

    print(f"\n  Target coverage: {result['target_coverage']*100:.0f}%")
    print(f"  Z-score target: {Z_TARGET:.4f}")
    print(f"\n  Per-position factors:")
    for pos, factor in result['factors'].items():
        cov = result['coverage'].get(pos, {})
        raw = cov.get('raw_coverage_90', 'N/A')
        n = cov.get('n', 'N/A')
        mean_res = cov.get('mean_residual', 'N/A')
        mean_std = cov.get('mean_std_fpts', 'N/A')
        print(f"    {pos}: scale={factor:.4f}  raw_90%_coverage={raw}  "
              f"mean_residual={mean_res}  mean_std={mean_std}  n={n}")

    overall = result['coverage'].get('overall', {})
    print(f"\n  Overall: scale={result['overall_factor']:.4f}  "
          f"raw_90%_coverage={overall.get('raw_coverage_90', 'N/A')}  n={overall.get('n', 'N/A')}")

    # Interpretation
    print("\n[2] Interpretation:")
    for pos, factor in result['factors'].items():
        if factor > 1.1:
            print(f"    {pos}: Model is OVERCONFIDENT — std_fpts needs {(factor-1)*100:.0f}% wider intervals")
        elif factor < 0.9:
            print(f"    {pos}: Model is UNDERCONFIDENT — std_fpts can be tightened by {(1-factor)*100:.0f}%")
        else:
            print(f"    {pos}: Well calibrated (scale={factor:.3f})")

    # Test calibration on a sample pool
    print("\n[3] Testing calibration on sample pool...")
    sample = pd.DataFrame({
        'player': ['McDavid', 'Draisaitl', 'Fox', 'Shesterkin'],
        'norm_pos': ['C', 'C', 'D', 'G'],
        'std_fpts': [6.5, 6.2, 4.1, 7.5],
        'projected_fpts': [8.5, 7.8, 5.2, 6.0],
    })
    calibrated = calibrate_std_fpts(sample, result)
    for _, row in calibrated.iterrows():
        orig = sample.loc[sample['player'] == row['player'], 'std_fpts'].values[0]
        print(f"    {row['player']:<15} std: {orig:.2f} → {row['std_fpts']:.2f} "
              f"(90% PI: ±{row['std_fpts'] * Z_TARGET:.1f} FPTS)")

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
