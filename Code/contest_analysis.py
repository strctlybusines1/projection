"""
Contest analysis for NHL DFS: parse DraftKings contest standings,
compute winning-lineup exposure, stack size, team/game concentration,
and emit a strategy report for high-dollar slates.
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd


# Position tokens that start a roster slot in the Lineup string
POSITION_PREFIXES = ("C ", "D ", "G ", "UTIL ", "W ")


def parse_lineup(lineup_str: str) -> List[Tuple[str, str]]:
    """
    Parse DraftKings Lineup column into list of (position, player_name).

    Format: "C Macklin Celebrini C Matty Beniers D John Klingberg ..."
    Positions: C, D, G, UTIL, W (order typically C, C, D, D, G, UTIL, W, W, W).
    """
    if not lineup_str or not isinstance(lineup_str, str):
        return []
    s = lineup_str.strip()
    if not s:
        return []
    result = []
    # Match position followed by name (name runs until next position token)
    pattern = r"(C|D|G|UTIL|W)\s+(.*?)(?=\s+(?:C|D|G|UTIL|W)\s+|$)"
    for m in re.finditer(pattern, s, re.DOTALL):
        pos, name = m.group(1), m.group(2).strip()
        if name:
            result.append((pos, name))
    return result


def load_contest_csv(path: str) -> pd.DataFrame:
    """Load contest standings CSV; require Rank, Points, Lineup columns."""
    df = pd.read_csv(path)
    for col in ["Rank", "Points", "Lineup"]:
        if col not in df.columns:
            raise ValueError(f"Contest CSV must have column '{col}'")
    return df


def get_top_n_lineups(
    df: pd.DataFrame,
    top_n: int,
) -> List[List[Tuple[str, str]]]:
    """Return list of parsed lineups (each list of (position, name)) for top N by Rank."""
    df_sorted = df.sort_values("Rank").head(top_n)
    lineups = []
    for _, row in df_sorted.iterrows():
        parsed = parse_lineup(row["Lineup"])
        if len(parsed) == 9:
            lineups.append(parsed)
        elif parsed:
            lineups.append(parsed)  # allow partial if parse missed some
    return lineups


def get_lineups_from_df(df: pd.DataFrame) -> List[List[Tuple[str, str]]]:
    """Return list of parsed lineups from all rows of df (sorted by Rank)."""
    df_sorted = df.sort_values("Rank")
    lineups = []
    for _, row in df_sorted.iterrows():
        parsed = parse_lineup(row["Lineup"])
        if len(parsed) == 9:
            lineups.append(parsed)
        elif parsed:
            lineups.append(parsed)
    return lineups


def player_exposure(
    lineups: List[List[Tuple[str, str]]],
) -> pd.DataFrame:
    """Compute player exposure: % of lineups containing each player."""
    from collections import Counter
    player_counts = Counter()
    player_pos = {}
    for lineup in lineups:
        for pos, name in lineup:
            player_counts[name] += 1
            player_pos[name] = pos  # last position seen
    n = len(lineups)
    rows = [
        {"player": name, "position": player_pos.get(name, ""), "count": count, "pct": 100.0 * count / n if n else 0}
        for name, count in player_counts.most_common()
    ]
    return pd.DataFrame(rows)


def stack_sizes_for_lineup(
    lineup: List[Tuple[str, str]],
    name_to_team: Optional[Dict[str, str]],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    For one lineup: (primary_stack_size, secondary_stack_size, n_teams).
    primary = max players from one team, secondary = second-most.
    """
    if not name_to_team or not lineup:
        return None, None, None
    from collections import Counter
    teams = []
    for _, name in lineup:
        team = name_to_team.get(name) or name_to_team.get(name.strip())
        if team:
            teams.append(team)
    if not teams:
        return None, None, None
    counts = Counter(teams).most_common()
    primary = counts[0][1] if counts else 0
    secondary = counts[1][1] if len(counts) > 1 else 0
    n_teams = len(counts)
    return primary, secondary, n_teams


def stack_size_distribution(
    lineups: List[List[Tuple[str, str]]],
    name_to_team: Optional[Dict[str, str]],
) -> Dict:
    """Aggregate primary/secondary stack size and distinct teams per lineup."""
    primaries, secondaries, n_teams_list = [], [], []
    for lu in lineups:
        p, s, n = stack_sizes_for_lineup(lu, name_to_team)
        if p is not None:
            primaries.append(p)
        if s is not None:
            secondaries.append(s)
        if n is not None:
            n_teams_list.append(n)
    return {
        "primary_avg": sum(primaries) / len(primaries) if primaries else None,
        "primary_max": max(primaries) if primaries else None,
        "secondary_avg": sum(secondaries) / len(secondaries) if secondaries else None,
        "n_teams_avg": sum(n_teams_list) / len(n_teams_list) if n_teams_list else None,
        "n_lineups_with_team": len(primaries),
    }


def team_concentration(
    lineups: List[List[Tuple[str, str]]],
    name_to_team: Optional[Dict[str, str]],
) -> pd.DataFrame:
    """Count how many top-N lineups (and total player slots) each team appears in."""
    if not name_to_team:
        return pd.DataFrame(columns=["team", "lineup_count", "player_slots"])
    from collections import Counter
    team_lineup_count = Counter()  # number of lineups that used this team
    team_slots = Counter()         # total player slots
    for lineup in lineups:
        teams_in_lineup = set()
        for _, name in lineup:
            team = name_to_team.get(name) or name_to_team.get(name.strip())
            if team:
                teams_in_lineup.add(team)
                team_slots[team] += 1
        for t in teams_in_lineup:
            team_lineup_count[t] += 1
    rows = [
        {"team": t, "lineup_count": team_lineup_count[t], "player_slots": team_slots[t]}
        for t in sorted(team_slots.keys(), key=lambda x: -team_slots[x])
    ]
    return pd.DataFrame(rows)


def build_name_to_team_from_salaries(salaries_path: str) -> Dict[str, str]:
    """Build player name -> team from DraftKings salary CSV (Name, TeamAbbrev or similar)."""
    df = pd.read_csv(salaries_path)
    name_col = None
    team_col = None
    for c in df.columns:
        if "name" in c.lower() or c == "Name":
            name_col = c
        if "team" in c.lower() or "abbrev" in c.lower() or c == "TeamAbbrev":
            team_col = c
    if name_col is None or team_col is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        name = row.get(name_col)
        team = row.get(team_col)
        if pd.notna(name) and pd.notna(team):
            out[str(name).strip()] = str(team).strip()
    return out


def build_report(
    contest_path: str,
    df: pd.DataFrame,
    lineups: List[List[Tuple[str, str]]],
    name_to_team: Optional[Dict[str, str]],
    top_n_list: List[int],
) -> str:
    """Generate Markdown report with metadata, exposure, stack stats, strategy implications."""
    lines = []
    lines.append("# Contest Strategy Report")
    lines.append("")
    lines.append(f"**Contest file**: `{contest_path}`")
    lines.append(f"**Total entries**: {len(df)}")
    first = df.sort_values("Rank").iloc[0]
    lines.append(f"**1st place score**: {first['Points']:.1f} pts")
    lines.append("")
    lines.append("---")
    lines.append("")

    for top_n in top_n_list:
        if top_n > len(lineups):
            continue
        subset = lineups[:top_n]
        lines.append(f"## Top {top_n} lineups")
        lines.append("")

        # Player exposure
        exp = player_exposure(subset)
        lines.append(f"### Player exposure (top {top_n})")
        lines.append("")
        lines.append("| Player | Position | % in lineups |")
        lines.append("|--------|----------|--------------|")
        for _, r in exp.head(25).iterrows():
            lines.append(f"| {r['player']} | {r['position']} | {r['pct']:.1f}% |")
        lines.append("")
        lines.append("")

        # Stack / team stats (if we have name_to_team)
        if name_to_team:
            stack_dist = stack_size_distribution(subset, name_to_team)
            if stack_dist["primary_avg"] is not None:
                lines.append(f"### Stack size (top {top_n})")
                lines.append("")
                lines.append(f"- **Primary stack (avg)**: {stack_dist['primary_avg']:.1f} players")
                lines.append(f"- **Primary stack (max)**: {stack_dist['primary_max']}")
                if stack_dist.get("secondary_avg") is not None:
                    lines.append(f"- **Secondary stack (avg)**: {stack_dist['secondary_avg']:.1f}")
                lines.append(f"- **Distinct teams per lineup (avg)**: {stack_dist['n_teams_avg']:.1f}" if stack_dist.get("n_teams_avg") else "")
                lines.append("")
            team_conc = team_concentration(subset, name_to_team)
            if not team_conc.empty:
                lines.append(f"### Team concentration (top {top_n})")
                lines.append("")
                lines.append("| Team | Lineups used | Player slots |")
                lines.append("|------|--------------|--------------|")
                for _, r in team_conc.head(15).iterrows():
                    lines.append(f"| {r['team']} | {r['lineup_count']} | {r['player_slots']} |")
                lines.append("")
        lines.append("")

    # Strategy implications
    lines.append("---")
    lines.append("")
    lines.append("## Strategy implications")
    lines.append("")
    if lineups:
        exp_all = player_exposure(lineups[: min(100, len(lineups))])
        chalk = exp_all[exp_all["pct"] >= 15].head(5)
        leverage = exp_all[exp_all["pct"] >= 5].tail(5)
        lines.append("- **Chalk in winning lineups**: Players with highest exposure in top 100 (list above) indicate where the field was right.")
        lines.append("- **Concentration**: Use stack size and team concentration tables to align with winning structure (primary stack size, number of games).")
        if name_to_team:
            stack_dist = stack_size_distribution(lineups[: min(100, len(lineups))], name_to_team)
            if stack_dist.get("primary_avg") is not None:
                lines.append(f"- **Primary stack**: Top lineups averaged {stack_dist['primary_avg']:.1f} players from one team; consider targeting similar stack size for 10-game slates.")
            if stack_dist.get("n_teams_avg") is not None:
                lines.append(f"- **Game concentration**: ~{stack_dist['n_teams_avg']:.1f} distinct teams per lineup (proxy for games); concentrate in 2–3 games for GPP.")
        lines.append("- **High-dollar 10-game**: For similar slates, prioritize primary stack from team concentration table and match stack size distribution.")
    lines.append("")
    return "\n".join(lines)


def build_pro_report(
    contest_path: str,
    df_full: pd.DataFrame,
    pro_df: pd.DataFrame,
    entry_name: str,
    lineups: List[List[Tuple[str, str]]],
    name_to_team: Optional[Dict[str, str]],
) -> str:
    """Generate Markdown report for a single pro (EntryName filter): metadata, exposure, stack, team concentration."""
    lines = []
    lines.append(f"# Pro study: EntryName = {entry_name}")
    lines.append("")
    lines.append(f"**Contest file**: `{contest_path}`")
    lines.append(f"**Total entries (contest)**: {len(df_full)}")
    lines.append(f"**Filter**: EntryName = {entry_name}")
    lines.append(f"**Pro entries**: {len(pro_df)}")
    if not pro_df.empty:
        rank_min = int(pro_df["Rank"].min())
        rank_max = int(pro_df["Rank"].max())
        lines.append(f"**Rank range**: {rank_min} – {rank_max}")
        best = pro_df.sort_values("Points", ascending=False).iloc[0]
        lines.append(f"**Best score**: {best['Points']:.1f} pts")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not lineups:
        lines.append("No lineups to analyze.")
        lines.append("")
        return "\n".join(lines)

    # Player exposure across pro's lineups
    exp = player_exposure(lineups)
    lines.append("### Player exposure (across pro's lineups)")
    lines.append("")
    lines.append("| Player | Position | % in lineups |")
    lines.append("|--------|----------|--------------|")
    for _, r in exp.head(25).iterrows():
        lines.append(f"| {r['player']} | {r['position']} | {r['pct']:.1f}% |")
    lines.append("")
    lines.append("")

    # Stack size and team concentration
    if name_to_team:
        stack_dist = stack_size_distribution(lineups, name_to_team)
        if stack_dist["primary_avg"] is not None:
            lines.append("### Stack size (across pro's lineups)")
            lines.append("")
            lines.append(f"- **Primary stack (avg)**: {stack_dist['primary_avg']:.1f} players")
            lines.append(f"- **Primary stack (max)**: {stack_dist['primary_max']}")
            if stack_dist.get("secondary_avg") is not None:
                lines.append(f"- **Secondary stack (avg)**: {stack_dist['secondary_avg']:.1f}")
            if stack_dist.get("n_teams_avg") is not None:
                lines.append(f"- **Distinct teams per lineup (avg)**: {stack_dist['n_teams_avg']:.1f}")
            lines.append("")
        team_conc = team_concentration(lineups, name_to_team)
        if not team_conc.empty:
            lines.append("### Team concentration (across pro's lineups)")
            lines.append("")
            lines.append("| Team | Lineups used | Player slots |")
            lines.append("|------|--------------|--------------|")
            for _, r in team_conc.head(15).iterrows():
                lines.append(f"| {r['team']} | {r['lineup_count']} | {r['player_slots']} |")
            lines.append("")

    # Strategy implications
    lines.append("---")
    lines.append("")
    lines.append("## Strategy implications")
    lines.append("")
    lines.append("Use the exposure and stack tables to see this pro's tendencies (chalk vs leverage, stack size, game concentration); align or differentiate your builds accordingly.")
    lines.append("")
    return "\n".join(lines)


def _find_salary_file() -> Optional[str]:
    """Try to find a DraftKings salary CSV in daily_salaries/ or project root."""
    proj = Path(__file__).resolve().parent
    for d in (proj / "daily_salaries", proj):
        if not d.exists():
            continue
        files = sorted(d.glob("DKSalaries*.csv"))
        if files:
            return str(files[-1])
    return None


def run_analysis(
    contest_path: str,
    top_n_list: Optional[List[int]] = None,
    salaries_path: Optional[str] = None,
    out_path: Optional[str] = None,
    entry_name: Optional[str] = None,
) -> str:
    """Load contest, parse lineups, compute metrics, return report string. Optionally write to file.
    If entry_name is set, filter to that pro's entries and emit a pro-study report."""
    df = load_contest_csv(contest_path)
    path_to_use = salaries_path or _find_salary_file()
    name_to_team = None
    if path_to_use and Path(path_to_use).exists():
        name_to_team = build_name_to_team_from_salaries(path_to_use)

    if entry_name is not None and entry_name.strip():
        # Pro study: require EntryName column, filter by base name
        if "EntryName" not in df.columns:
            raise ValueError("Contest CSV has no 'EntryName' column; required for --entry-name")
        base = entry_name.strip()
        pro_df = df[df["EntryName"].astype(str).str.startswith(base)].copy()
        if pro_df.empty:
            report = f"# Pro study: EntryName = {base}\n\nNo entries found for EntryName = {base}.\n"
            if out_path:
                Path(out_path).write_text(report, encoding="utf-8")
            return report
        lineups = get_lineups_from_df(pro_df)
        report = build_pro_report(
            contest_path, df, pro_df, base, lineups, name_to_team
        )
    else:
        # Field report: top N by Rank
        if top_n_list is None:
            top_n_list = [20, 50, 100]
        max_n = max(top_n_list) if top_n_list else 100
        lineups = get_top_n_lineups(df, max_n)
        report = build_report(contest_path, df, lineups, name_to_team, top_n_list)

    if out_path:
        Path(out_path).write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze DraftKings NHL contest standings for strategy report.")
    parser.add_argument("contest", help="Path to contest CSV (e.g. contests/$360NHLSpin_1.27.26.csv)")
    parser.add_argument("--top-n", type=str, default="20,50,100", help="Comma-separated top-N values (default: 20,50,100)")
    parser.add_argument("--out", type=str, default=None, help="Write report to this path (Markdown)")
    parser.add_argument("--salaries", type=str, default=None, help="Optional salary CSV for name->team (stack/team stats)")
    parser.add_argument("--entry-name", type=str, default=None, help="Study a single pro: filter to entries where EntryName starts with this (e.g. bkreider)")
    args = parser.parse_args()
    top_n_list = [int(x.strip()) for x in args.top_n.split(",") if x.strip()]
    report = run_analysis(
        args.contest,
        top_n_list=top_n_list,
        salaries_path=args.salaries,
        out_path=args.out,
        entry_name=args.entry_name,
    )
    print(report)
    if args.out:
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
