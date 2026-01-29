"""
NHL Season Signal vs Noise Backtest.

Loads team-level NST CSVs from test/, catalogs time period + GF, SF, GA, SA,
computes DK-style offense/defense outcomes, and runs every stat through
persistence and predictive-power analyses. See plan and
https://www.naturalstattrick.com/glossary.php?teams for stat definitions.
"""

import os
import glob
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import SKATER_SCORING

# DK weights for team totals (goals and shots only)
DK_GOALS = SKATER_SCORING["goals"]  # 8.5
DK_SHOTS = SKATER_SCORING["shots_on_goal"]  # 1.5

# Columns we add (not predictors)
WINDOW_COLS = ["period_id", "window_start", "window_end", "segment", "Team", "GP"]
DK_OUTCOME_COLS = ["team_dk_total", "team_dk_per_game", "team_dk_allowed_total", "team_dk_allowed_per_game", "team_dk_net_per_game"]

# Stats to test for regression to mean (percentage / luck). Use normalized names (pct not %).
REGRESSION_STATS = ["PDO", "SHpct", "SVpct", "SCSHpct", "SCSVpct", "HDSHpct", "HDSVpct", "MDSHpct", "MDSVpct", "LDSHpct", "LDSVpct", "GFpct"]

# NST glossary one-line definitions for report (https://www.naturalstattrick.com/glossary.php?teams)
STAT_DEFINITIONS = {
    "GP": "Games played",
    "TOI": "Total time on ice",
    "W": "Wins", "L": "Losses", "OTL": "Overtime losses", "ROW": "Regulation+OT wins",
    "Points": "Standings points", "Point_pct": "Point percentage",
    "CF": "Corsi for (shot attempts)", "CA": "Corsi against", "CF%": "Corsi share",
    "FF": "Fenwick for (unblocked attempts)", "FA": "Fenwick against", "FF%": "Fenwick share",
    "SF": "Shots for", "SA": "Shots against", "SF%": "Shot share",
    "GF": "Goals for", "GA": "Goals against", "GF%": "Goal share",
    "xGF": "Expected goals for", "xGA": "Expected goals against", "xGF%": "xG share",
    "SCF": "Scoring chances for", "SCA": "Scoring chances against", "SCF%": "SC share",
    "SCSF": "SC shots for", "SCSA": "SC shots against", "SCSF%": "SC shot share",
    "SCGF": "SC goals for", "SCGA": "SC goals against", "SCGF%": "SC goal share",
    "SCSH%": "SC shooting %", "SCSV%": "SC save %",
    "HDCF": "High-danger chances for", "HDCA": "HD chances against", "HDCF%": "HD share",
    "HDSF": "HD shots for", "HDSA": "HD shots against", "HDSF%": "HD shot share",
    "HDGF": "HD goals for", "HDGA": "HD goals against", "HDGF%": "HD goal share",
    "HDSH%": "HD shooting %", "HDSV%": "HD save %",
    "MDCF": "Medium-danger chances for", "MDCA": "MD chances against", "MDCF%": "MD share",
    "MDSF": "MD shots for", "MDSA": "MD shots against", "MDSF%": "MD shot share",
    "MDGF": "MD goals for", "MDGA": "MD goals against", "MDGF%": "MD goal share",
    "MDSH%": "MD shooting %", "MDSV%": "MD save %",
    "LDCF": "Low-danger chances for", "LDCA": "LD chances against", "LDCF%": "LD share",
    "LDSF": "LD shots for", "LDSA": "LD shots against", "LDSF%": "LD shot share",
    "LDGF": "LD goals for", "LDGA": "LD goals against", "LDGF%": "LD goal share",
    "LDSH%": "LD shooting %", "LDSV%": "LD save %",
    "SH%": "Shooting %", "SV%": "Save %", "PDO": "SH%+SV%, tends to regress to 100",
}
# Normalized names (pct suffix) for columns after load
for k, v in list(STAT_DEFINITIONS.items()):
    if "%" in k:
        STAT_DEFINITIONS[k.replace("%", "pct")] = v


def parse_filename_dates(filename: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse START_END from filename like 10.7.25_10.22.25_nhl.csv -> (start, end)."""
    basename = os.path.basename(filename).replace("_nhl.csv", "")
    parts = basename.split("_")
    if len(parts) != 2:
        return None, None
    try:
        start_parts = parts[0].split(".")
        end_parts = parts[1].split(".")
        if len(start_parts) != 3 or len(end_parts) != 3:
            return None, None
        # MM.DD.YY
        sy = int(start_parts[2])
        ey = int(end_parts[2])
        start_year = 2000 + sy if sy < 100 else sy
        end_year = 2000 + ey if ey < 100 else ey
        start_dt = datetime(start_year, int(start_parts[0]), int(start_parts[1]))
        end_dt = datetime(end_year, int(end_parts[0]), int(end_parts[1]))
        return start_dt, end_dt
    except (ValueError, IndexError):
        return None, None


def segment_from_dates(start_dt: datetime, end_dt: datetime) -> str:
    """Assign early/mid/late from window midpoint month. Oct–Nov=early, Dec=mid, Jan–Feb=late."""
    delta = (end_dt - start_dt).total_seconds() / 2
    mid = start_dt + timedelta(seconds=delta)
    month = mid.month
    if month in (10, 11):
        return "early"
    if month == 12:
        return "mid"
    if month in (1, 2):
        return "late"
    return "other"


def load_and_build_panel(csv_dir: str) -> pd.DataFrame:
    """Load all *_nhl.csv files, parse dates, assign segment, normalize columns, build panel with catalog."""
    csv_dir = Path(csv_dir)
    pattern = str(csv_dir / "*_nhl.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No *_nhl.csv files in {csv_dir}")

    all_dfs = []
    for f in files:
        start_dt, end_dt = parse_filename_dates(f)
        if start_dt is None or end_dt is None:
            continue
        period_id = Path(f).stem.replace("_nhl", "")
        seg = segment_from_dates(start_dt, end_dt)

        df = pd.read_csv(f)
        # Drop unnamed index column if present
        if df.columns[0].strip() == "" or df.columns[0].startswith("Unnamed"):
            df = df.drop(columns=[df.columns[0]])
        # Normalize column names: spaces and % -> underscore
        df.columns = [c.replace(" ", "_").replace("%", "pct").replace(".", "") for c in df.columns]
        # Ensure Point_pct exists (might be Point_pct from "Point %")
        if "Point_pct" not in df.columns and "Point" in df.columns:
            pass  # keep as is

        df["period_id"] = period_id
        df["window_start"] = start_dt
        df["window_end"] = end_dt
        df["segment"] = seg
        all_dfs.append(df)

    panel = pd.concat(all_dfs, ignore_index=True)
    return panel


def add_dk_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add team_dk_total, team_dk_per_game, team_dk_allowed_*, team_dk_net_per_game. Guard GP=0."""
    df = panel.copy()
    gp = df["GP"].replace(0, np.nan)
    df["team_dk_total"] = df["GF"] * DK_GOALS + df["SF"] * DK_SHOTS
    df["team_dk_per_game"] = df["team_dk_total"] / gp
    df["team_dk_allowed_total"] = df["GA"] * DK_GOALS + df["SA"] * DK_SHOTS
    df["team_dk_allowed_per_game"] = df["team_dk_allowed_total"] / gp
    df["team_dk_net_per_game"] = df["team_dk_per_game"] - df["team_dk_allowed_per_game"]
    return df


def get_predictor_columns(panel: pd.DataFrame) -> List[str]:
    """Numeric columns that are not identifiers or DK-derived."""
    exclude = set(WINDOW_COLS + DK_OUTCOME_COLS + ["Team"])
    numeric = panel.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


def build_window_pairs(panel: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return list of (period_id_A, period_id_B) where B is after A (by window_start)."""
    periods = panel[["period_id", "window_start"]].drop_duplicates().sort_values("window_start")
    ids = periods["period_id"].tolist()
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pairs.append((a, b))
    return pairs


def correlation_safe(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation; return np.nan if insufficient valid pairs."""
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    return x[valid].corr(y[valid])


def run_persistence_and_predictions(
    panel: pd.DataFrame,
    predictor_cols: List[str],
    window_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """For each stat: persistence (stat A vs stat B), corr with dk_per_game, dk_allowed_per_game, dk_net_per_game. By pair then average."""
    results = []
    for stat in predictor_cols:
        if stat not in panel.columns:
            continue
        pers_list = []
        corr_dk_off_list = []
        corr_dk_def_list = []
        corr_dk_net_list = []
        for (pid_a, pid_b) in window_pairs:
            df_a = panel[panel["period_id"] == pid_a][["Team", stat]].rename(columns={stat: "stat_a"})
            df_b = panel[panel["period_id"] == pid_b][["Team", stat, "team_dk_per_game", "team_dk_allowed_per_game", "team_dk_net_per_game"]]
            df_b = df_b.rename(columns={stat: "stat_b", "team_dk_per_game": "dk_off", "team_dk_allowed_per_game": "dk_def", "team_dk_net_per_game": "dk_net"})
            merged = df_a.merge(df_b, on="Team", how="inner")
            if len(merged) < 3:
                continue
            pers_list.append(correlation_safe(merged["stat_a"], merged["stat_b"]))
            corr_dk_off_list.append(correlation_safe(merged["stat_a"], merged["dk_off"]))
            corr_dk_def_list.append(correlation_safe(merged["stat_a"], merged["dk_def"]))
            corr_dk_net_list.append(correlation_safe(merged["stat_a"], merged["dk_net"]))

        pers = np.nanmean(pers_list) if pers_list else np.nan
        co = np.nanmean(corr_dk_off_list) if corr_dk_off_list else np.nan
        cd = np.nanmean(corr_dk_def_list) if corr_dk_def_list else np.nan
        cn = np.nanmean(corr_dk_net_list) if corr_dk_net_list else np.nan
        results.append({
            "stat": stat,
            "persistence": pers,
            "corr_team_dk_per_game": co,
            "corr_team_dk_allowed_per_game": cd,
            "corr_team_dk_net_per_game": cn,
        })
    return pd.DataFrame(results)


def run_regression_to_mean(
    panel: pd.DataFrame,
    predictor_cols: List[str],
    window_pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """For percentage/luck stats, correlate stat in A with change (B - A). Return dict stat -> correlation."""
    regress_stats = [c for c in predictor_cols if c in REGRESSION_STATS or "pct" in c.lower()]
    out = {}
    for stat in regress_stats:
        if stat not in panel.columns:
            continue
        corrs = []
        for (pid_a, pid_b) in window_pairs:
            df_a = panel[panel["period_id"] == pid_a][["Team", stat]].rename(columns={stat: "stat_a"})
            df_b = panel[panel["period_id"] == pid_b][["Team", stat]].rename(columns={stat: "stat_b"})
            merged = df_a.merge(df_b, on="Team", how="inner")
            merged["change"] = merged["stat_b"] - merged["stat_a"]
            if len(merged) < 3:
                continue
            corrs.append(correlation_safe(merged["stat_a"], merged["change"]))
        out[stat] = np.nanmean(corrs) if corrs else np.nan
    return out


def run_by_segment(
    panel: pd.DataFrame,
    predictor_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """Compute persistence and predictiveness for early->mid, mid->late, early->late pairs only."""
    period_starts = panel.groupby("period_id")["window_start"].min().to_dict()
    pair_segments = [("early", "mid"), ("mid", "late"), ("early", "late")]
    by_seg = {}
    for seg_a, seg_b in pair_segments:
        ids_a = panel[panel["segment"] == seg_a]["period_id"].unique().tolist()
        ids_b = panel[panel["segment"] == seg_b]["period_id"].unique().tolist()
        if not ids_a or not ids_b:
            continue
        pairs = [
            (a, b) for a in ids_a for b in ids_b
            if period_starts.get(a, datetime.min) < period_starts.get(b, datetime.max)
        ]
        if not pairs:
            continue
        sub = panel[panel["period_id"].isin([p[0] for p in pairs] + [p[1] for p in pairs])]
        res = run_persistence_and_predictions(sub, predictor_cols, pairs)
        by_seg[f"{seg_a}_to_{seg_b}"] = res
    return by_seg


def pattern_finding(panel: pd.DataFrame) -> Dict:
    """By segment: distributions and correlations of GF, SF, GA, SA. Cross-window and within-window."""
    core = ["GF", "SF", "GA", "SA"]
    for c in core:
        if c not in panel.columns:
            return {"by_segment": {}, "cross_window": {}, "error": "Missing core columns"}
    by_segment = {}
    for seg in panel["segment"].unique():
        sub = panel[panel["segment"] == seg]
        if len(sub) < 3:
            continue
        by_segment[seg] = {
            "mean": sub[core].mean().to_dict(),
            "std": sub[core].std().to_dict(),
            "corr_matrix": sub[core].corr().to_dict(),
        }
    # Cross-window: same team, window A -> window B (first two periods by time)
    periods_ordered = panel[["period_id", "window_start"]].drop_duplicates().sort_values("window_start")["period_id"].tolist()
    cross = {}
    if len(periods_ordered) >= 2:
        a, b = periods_ordered[0], periods_ordered[1]
        da = panel[panel["period_id"] == a][["Team"] + core].copy()
        da = da.rename(columns={x: f"{x}_a" for x in core})
        db = panel[panel["period_id"] == b][["Team"] + core].copy()
        db = db.rename(columns={x: f"{x}_b" for x in core})
        merged = da.merge(db, on="Team", how="inner")
        for col in core:
            key = f"{col}_a_vs_{col}_b"
            cross[key] = correlation_safe(merged[f"{col}_a"], merged[f"{col}_b"])
    return {"by_segment": by_segment, "cross_window": cross}


def main():
    parser = argparse.ArgumentParser(description="NHL Season Signal vs Noise Backtest")
    parser.add_argument("--input-dir", default="test", help="Directory containing *_nhl.csv files")
    parser.add_argument("--output-dir", default="test", help="Directory for output CSVs and report")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    input_dir = base.parent / args.input_dir if not os.path.isabs(args.input_dir) else Path(args.input_dir)
    output_dir = base.parent / args.output_dir if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and build panel with catalog
    print("Loading CSVs and building panel...")
    panel = load_and_build_panel(str(input_dir))
    print(f"  Loaded {len(panel)} rows, {panel['period_id'].nunique()} windows")

    # 2. Add DK columns
    panel = add_dk_columns(panel)
    predictor_cols = get_predictor_columns(panel)
    print(f"  Predictor stats: {len(predictor_cols)}")

    # 3. Window pairs and run every stat
    window_pairs = build_window_pairs(panel)
    print(f"  Window pairs: {len(window_pairs)}")

    print("Running persistence and predictive power for every stat...")
    results_df = run_persistence_and_predictions(panel, predictor_cols, window_pairs)
    regress = run_regression_to_mean(panel, predictor_cols, window_pairs)
    results_df["regression_to_mean_corr"] = results_df["stat"].map(regress)

    # Signal/noise: high persistence or high |corr| vs any DK outcome = signal
    results_df["max_abs_predictive"] = results_df[["corr_team_dk_per_game", "corr_team_dk_allowed_per_game", "corr_team_dk_net_per_game"]].abs().max(axis=1)
    results_df["signal_noise"] = np.where(
        (results_df["persistence"].abs() >= 0.3) | (results_df["max_abs_predictive"] >= 0.3),
        "signal",
        "noise",
    )
    results_df["definition"] = results_df["stat"].map(lambda s: STAT_DEFINITIONS.get(s, ""))

    # Sort by absolute predictiveness
    results_df = results_df.sort_values("max_abs_predictive", ascending=False, na_position="last")

    # 4. By-segment
    print("Running by-segment analysis...")
    by_seg = run_by_segment(panel, predictor_cols)

    # 5. Pattern-finding
    print("Running pattern-finding (catalog)...")
    patterns = pattern_finding(panel)

    # 6. Outputs
    # Catalog table: period_id, window_start, window_end, segment, Team, GP, GF, SF, GA, SA, ...
    catalog_cols = ["period_id", "window_start", "window_end", "segment", "Team", "GP", "GF", "SF", "GA", "SA"] + [c for c in panel.columns if c not in ["period_id", "window_start", "window_end", "segment", "Team", "GP", "GF", "SF", "GA", "SA"]]
    catalog_cols = [c for c in catalog_cols if c in panel.columns]
    catalog_df = panel[catalog_cols].copy()
    catalog_path = output_dir / "season_catalog.csv"
    catalog_df.to_csv(str(catalog_path), index=False)
    print(f"  Wrote {catalog_path}")

    # Full stat table
    stat_cols = ["stat", "definition", "persistence", "corr_team_dk_per_game", "corr_team_dk_allowed_per_game", "corr_team_dk_net_per_game", "regression_to_mean_corr", "signal_noise"]
    stat_cols = [c for c in stat_cols if c in results_df.columns]
    results_df[stat_cols].to_csv(str(output_dir / "signal_noise_report.csv"), index=False)
    print(f"  Wrote {output_dir / 'signal_noise_report.csv'}")

    # Markdown report
    md_path = output_dir / "SEASON_SIGNAL_REPORT.md"
    def df_to_md(df):
        try:
            return df.to_markdown(index=False)
        except AttributeError:
            return df.to_string(index=False)

    with open(md_path, "w") as f:
        f.write("# NHL Season Signal vs Noise Backtest Report\n\n")
        f.write("Stat definitions: [Natural Stat Trick - Glossary](https://www.naturalstattrick.com/glossary.php?teams).\n\n")
        f.write("## Full stat table (top 20 by predictiveness)\n\n")
        f.write(df_to_md(results_df.head(20)))
        f.write("\n\n## By-segment summaries\n\n")
        for seg_name, seg_df in by_seg.items():
            f.write(f"### {seg_name}\n\n")
            f.write(df_to_md(seg_df.head(15)))
            f.write("\n\n")
        f.write("## Pattern summary (catalog)\n\n")
        if "by_segment" in patterns and patterns["by_segment"]:
            f.write("Means by segment (GF, SF, GA, SA):\n\n")
            for seg, d in patterns["by_segment"].items():
                f.write(f"- **{seg}**: {d.get('mean', {})}\n")
        if "cross_window" in patterns and patterns["cross_window"]:
            f.write("\nCross-window correlations (first two periods):\n\n")
            for k, v in patterns["cross_window"].items():
                vstr = f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else str(v)
                f.write(f"- {k}: {vstr}\n")
    print(f"  Wrote {md_path}")

    # By-segment CSVs
    for seg_name, seg_df in by_seg.items():
        seg_df.to_csv(str(output_dir / f"signal_noise_by_segment_{seg_name}.csv"), index=False)
    print("Done.")
    return panel, results_df, by_seg, patterns


if __name__ == "__main__":
    main()
