"""
Load team danger stats (HD/MD/LD shots for) from NST-style CSVs in the test folder.
Used for goalie opponent shot quality: opponent's HDSF, MDSF, LDSF → per-60 and HD share.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    TEAM_DANGER_CSV_DIR,
    TEAM_DANGER_CSV,
    TEAM_FULL_NAME_TO_ABBREV,
)


def _normalize_team(team_name: str) -> str:
    """Map full team name to NHL abbrev (e.g. Chicago Blackhawks -> CHI)."""
    if pd.isna(team_name):
        return ""
    key = str(team_name).upper().strip()
    return TEAM_FULL_NAME_TO_ABBREV.get(key, "")


def _parse_filename_end_date(filename: str) -> Optional[tuple]:
    """Parse end date from filename like 1.25.26_1.27.26_nhl.csv -> (2026, 1, 27)."""
    base = os.path.basename(filename).replace("_nhl.csv", "").strip()
    parts = base.split("_")
    if len(parts) != 2:
        return None
    try:
        end_parts = parts[1].split(".")
        if len(end_parts) != 3:
            return None
        yy = int(end_parts[2])
        year = 2000 + yy if yy < 100 else yy
        return (year, int(end_parts[0]), int(end_parts[1]))
    except (ValueError, IndexError):
        return None


def load_team_danger_stats(
    csv_dir: Optional[str] = None,
    csv_file: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load team danger stats from NST-style CSV(s) in the test folder.

    Reads Team, TOI, HDSF, MDSF, LDSF, HDSV%, MDSV%, LDSV%; computes per-60 rates,
    shot shares, and team save % by type (for goalie projection).
    Returns DataFrame with team (abbrev) as index and columns:
    hdsf_60, mdsf_60, ldsf_60, hd_share, md_share, ld_share, hdsv_pct, mdsv_pct, ldsv_pct.

    Args:
        csv_dir: Directory containing *_nhl.csv files (default: TEAM_DANGER_CSV_DIR).
        csv_file: Specific filename (default: TEAM_DANGER_CSV). If None and csv_dir set, picks latest by end date.

    Returns:
        DataFrame keyed by team abbrev, or None if no config/dir or load fails.
    """
    csv_dir = csv_dir if csv_dir is not None else TEAM_DANGER_CSV_DIR
    csv_file = csv_file if csv_file is not None else TEAM_DANGER_CSV

    if not csv_dir:
        return None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.normpath(os.path.join(base_dir, csv_dir))
    if not os.path.isdir(dir_path):
        return None

    path = None
    if csv_file:
        path = os.path.join(dir_path, csv_file)
        if not os.path.isfile(path):
            return None
    else:
        # Find *_nhl.csv and pick latest by end date in filename
        candidates = [
            f for f in os.listdir(dir_path)
            if f.endswith("_nhl.csv") and os.path.isfile(os.path.join(dir_path, f))
        ]
        if not candidates:
            return None
        def sort_key(f):
            t = _parse_filename_end_date(f)
            return t if t else (0, 0, 0)
        candidates.sort(key=sort_key, reverse=True)
        path = os.path.join(dir_path, candidates[0])

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    required = ["Team", "TOI", "HDSF", "MDSF", "LDSF"]
    if not all(c in df.columns for c in required):
        return None

    out = pd.DataFrame(index=df.index)
    out["team"] = df["Team"].apply(_normalize_team)
    toi = df["TOI"].replace(0, np.nan)
    out["hdsf_60"] = df["HDSF"] / toi * 60
    out["mdsf_60"] = df["MDSF"] / toi * 60
    out["ldsf_60"] = df["LDSF"] / toi * 60
    total_sf = (df["HDSF"] + df["MDSF"] + df["LDSF"]).replace(0, np.nan)
    out["hd_share"] = df["HDSF"] / total_sf
    out["md_share"] = df["MDSF"] / total_sf
    out["ld_share"] = df["LDSF"] / total_sf

    # Team save % by shot type (HDSV%, MDSV%, LDSV% in CSV are 0–100)
    for col, out_col in [("HDSV%", "hdsv_pct"), ("MDSV%", "mdsv_pct"), ("LDSV%", "ldsv_pct")]:
        if col in df.columns:
            out[out_col] = (df[col].replace([np.nan, ""], np.nan).astype(float) / 100.0).clip(0, 1)
        else:
            out[out_col] = np.nan

    out = out[out["team"] != ""].copy()
    out = out.set_index("team")
    return out
