"""
Ledoit-Wolf shrinkage correlation matrix for NHL DFS stacking.

Replaces hardcoded heuristic correlations with empirically-derived values
from historical DK fantasy scoring data (game_logs_skaters).

Role slots (9 total, matching DK roster minus goalie):
  C1, C2, W1, W2, W3, W4, D1, D2, D3

Usage:
  python correlation_matrix.py            # Regenerate artifact + print comparison
  python correlation_matrix.py --force    # Regenerate even if artifact exists
"""

import json
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"
ARTIFACT_PATH = Path(__file__).parent / "data" / "lw_correlations.json"

ROLE_SLOTS = ["C1", "C2", "W1", "W2", "W3", "W4", "D1", "D2", "D3"]

# Stack-type definitions: each maps to a set of role-slot pairs whose
# pairwise correlations are averaged to produce the stack correlation.
STACK_TYPE_SLOTS = {
    "PP1": ["C1", "W1", "W2", "D1"],
    "Line1": ["C1", "W1", "W2"],
    "Line2": ["C2", "W3", "W4"],
    "Line1+D1": ["C1", "W1", "W2", "D1", "D2"],
    "Defense1": ["D1", "D2"],
    "Defense2": ["D2", "D3"],
}

# Old hardcoded values for comparison
HARDCODED = {
    "PP1": 0.95,
    "Line1": 0.85,
    "Line2": 0.70,
    "Line1+D1": 0.75,
    "Defense1": 0.50,
    "Defense2": 0.40,
}


def build_role_slot_matrix(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Query game_logs_skaters, assign role slots by TOI rank, pivot to wide format.

    Returns a DataFrame with one row per team-game and one column per role slot,
    values are dk_fpts. Rows with any missing slot are dropped.
    """
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT team, game_id, player_name, position, toi_seconds, dk_fpts
        FROM game_logs_skaters
        WHERE toi_seconds > 0 AND dk_fpts IS NOT NULL
        ORDER BY team, game_id, toi_seconds DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        raise ValueError("No data found in game_logs_skaters")

    # Map positions to groups: C stays C, L/R become W, D stays D
    pos_map = {"C": "C", "L": "W", "R": "W", "D": "D"}
    df["pos_group"] = df["position"].map(pos_map)
    df = df.dropna(subset=["pos_group"])

    # Rank within position group per team-game by TOI (descending)
    df["rank"] = df.groupby(["team", "game_id", "pos_group"])["toi_seconds"].rank(
        method="first", ascending=False
    ).astype(int)

    # Assign role slot labels
    def _slot_label(row):
        pg = row["pos_group"]
        r = row["rank"]
        if pg == "C" and r <= 2:
            return f"C{r}"
        elif pg == "W" and r <= 4:
            return f"W{r}"
        elif pg == "D" and r <= 3:
            return f"D{r}"
        return None

    df["slot"] = df.apply(_slot_label, axis=1)
    df = df.dropna(subset=["slot"])

    # Pivot: rows = (team, game_id), columns = slot, values = dk_fpts
    wide = df.pivot_table(
        index=["team", "game_id"],
        columns="slot",
        values="dk_fpts",
        aggfunc="first",
    )

    # Keep only rows that have all 9 slots populated
    wide = wide.reindex(columns=ROLE_SLOTS).dropna()

    print(f"  Built wide matrix: {len(wide)} team-games x {len(ROLE_SLOTS)} slots")
    return wide


def compute_lw_correlation(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Fit Ledoit-Wolf shrinkage estimator and convert covariance to correlation.

    Returns a 9x9 DataFrame (rows & columns = ROLE_SLOTS) of correlations.
    """
    X = wide_df[ROLE_SLOTS].values

    lw = LedoitWolf()
    lw.fit(X)
    cov = lw.covariance_

    # Convert covariance to correlation: corr_ij = cov_ij / (std_i * std_j)
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    # Avoid division by zero (shouldn't happen with real data)
    outer_std[outer_std == 0] = 1.0
    corr = cov / outer_std

    # Clip to [-1, 1] for numerical safety
    corr = np.clip(corr, -1.0, 1.0)

    corr_df = pd.DataFrame(corr, index=ROLE_SLOTS, columns=ROLE_SLOTS)
    print(f"  Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")
    return corr_df


def map_slots_to_stack_types(corr_df: pd.DataFrame) -> dict:
    """Average pairwise correlations into stack-type buckets.

    Returns dict like {"PP1": 0.42, "Line1": 0.38, ...}
    """
    result = {}
    for stack_type, slots in STACK_TYPE_SLOTS.items():
        pairs = list(combinations(slots, 2))
        if not pairs:
            continue
        values = [corr_df.loc[s1, s2] for s1, s2 in pairs]
        result[stack_type] = round(float(np.mean(values)), 4)
    return result


def save_correlations(stack_corr: dict, corr_df: pd.DataFrame,
                      path: Path = ARTIFACT_PATH):
    """Save correlation artifact to JSON."""
    artifact = {
        "stack_correlations": stack_corr,
        "pairwise_matrix": {
            slot: {s2: round(float(corr_df.loc[slot, s2]), 4) for s2 in ROLE_SLOTS}
            for slot in ROLE_SLOTS
        },
        "role_slots": ROLE_SLOTS,
        "stack_type_definitions": {k: v for k, v in STACK_TYPE_SLOTS.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Saved artifact to {path}")


def load_correlations(path: Path = ARTIFACT_PATH) -> dict | None:
    """Load correlation artifact from JSON. Returns None if not found."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def regenerate(db_path: Path = DB_PATH, artifact_path: Path = ARTIFACT_PATH):
    """Full pipeline: build matrix, fit LW, map to stack types, save."""
    print("Building role-slot matrix from game_logs_skaters...")
    wide = build_role_slot_matrix(db_path)

    print("Fitting Ledoit-Wolf shrinkage estimator...")
    corr_df = compute_lw_correlation(wide)

    print("Mapping pairwise correlations to stack types...")
    stack_corr = map_slots_to_stack_types(corr_df)

    save_correlations(stack_corr, corr_df, artifact_path)
    return stack_corr, corr_df


def print_comparison(stack_corr: dict):
    """Print comparison table: old hardcoded vs new empirical values."""
    print("\n" + "=" * 55)
    print(f"  {'Stack Type':<15} {'Hardcoded':>10} {'Empirical':>10} {'Delta':>10}")
    print("=" * 55)
    for st in STACK_TYPE_SLOTS:
        old = HARDCODED.get(st, 0.0)
        new = stack_corr.get(st, 0.0)
        delta = new - old
        sign = "+" if delta >= 0 else ""
        print(f"  {st:<15} {old:>10.4f} {new:>10.4f} {sign}{delta:>9.4f}")
    print("=" * 55)


def print_matrix(corr_df: pd.DataFrame):
    """Print the full 9x9 correlation matrix."""
    print("\n9x9 Ledoit-Wolf Correlation Matrix:")
    print("-" * 80)
    # Header
    header = "        " + "  ".join(f"{s:>6}" for s in ROLE_SLOTS)
    print(header)
    for slot in ROLE_SLOTS:
        row_vals = "  ".join(f"{corr_df.loc[slot, s]:6.3f}" for s in ROLE_SLOTS)
        print(f"  {slot:<5} {row_vals}")
    print()


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv

    if not force and ARTIFACT_PATH.exists():
        print(f"Artifact already exists at {ARTIFACT_PATH}")
        print("Use --force to regenerate.\n")
        data = load_correlations()
        stack_corr = data["stack_correlations"]
        # Reconstruct corr_df from pairwise_matrix
        corr_df = pd.DataFrame(data["pairwise_matrix"]).T
        corr_df = corr_df[ROLE_SLOTS].loc[ROLE_SLOTS]
    else:
        stack_corr, corr_df = regenerate()

    print_matrix(corr_df)
    print_comparison(stack_corr)
