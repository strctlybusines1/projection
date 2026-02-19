"""
Pre-Flight Validation for NHL DFS Pipeline.

Run before committing to a slate to catch data issues early.

Usage:
    python main.py --validate                    # Quick check with auto-detected files
    python main.py --validate --salaries DKSalaries_2.25.26.csv
    python validate.py                           # Standalone mode (no lineup generation)
    python validate.py --salaries DKSalaries_2.25.26.csv --vegas VegasNHL.csv
"""

import sys
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ================================================================
#  Result Classes
# ================================================================

class CheckResult:
    """Single validation check result."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    def __init__(self, name: str, status: str, message: str, details: str = ""):
        self.name = name
        self.status = status
        self.message = message
        self.details = details

    @property
    def icon(self):
        return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[self.status]


class ValidationReport:
    """Collection of check results."""

    def __init__(self):
        self.checks: List[CheckResult] = []

    def add(self, result: CheckResult):
        self.checks.append(result)

    @property
    def passes(self):
        return [c for c in self.checks if c.status == CheckResult.PASS]

    @property
    def warnings(self):
        return [c for c in self.checks if c.status == CheckResult.WARN]

    @property
    def failures(self):
        return [c for c in self.checks if c.status == CheckResult.FAIL]

    @property
    def is_go(self):
        return len(self.failures) == 0

    def print_report(self):
        print(f"\n{'=' * 70}")
        print("  PRE-FLIGHT VALIDATION REPORT")
        print(f"{'=' * 70}")

        for check in self.checks:
            print(f"  {check.icon}  {check.name}: {check.message}")
            if check.details:
                for line in check.details.strip().split("\n"):
                    print(f"        {line}")

        # Summary
        print(f"\n{'-' * 70}")
        total = len(self.checks)
        print(f"  {len(self.passes)}/{total} passed, "
              f"{len(self.warnings)} warnings, "
              f"{len(self.failures)} failures")

        if self.is_go:
            print(f"\n  🟢  GO — Pipeline is ready for this slate.")
        else:
            print(f"\n  🔴  NO-GO — Fix {len(self.failures)} failure(s) before running.")
        print(f"{'=' * 70}\n")


# ================================================================
#  Individual Checks
# ================================================================

def check_salary_file(salary_path: str, target_date: str) -> CheckResult:
    """Verify DK salary file exists and matches today's slate."""
    if not salary_path or not os.path.exists(salary_path):
        return CheckResult(
            "Salary File", CheckResult.FAIL,
            f"Not found: {salary_path}",
            "Download DKSalaries CSV from DraftKings and place in daily_salaries/"
        )

    # Check file is not empty
    size = os.path.getsize(salary_path)
    if size < 500:
        return CheckResult(
            "Salary File", CheckResult.FAIL,
            f"File too small ({size} bytes) — likely empty or corrupt"
        )

    # Try to parse date from filename (DKSalaries_M.D.YY.csv or similar)
    fname = Path(salary_path).name
    date_match = re.search(r'(\d{1,2})[._-](\d{1,2})[._-](\d{2,4})', fname)
    if date_match:
        m, d, y = date_match.groups()
        y = int(y)
        if y < 100:
            y += 2000
        try:
            file_date = datetime(y, int(m), int(d)).strftime('%Y-%m-%d')
            target_dt = datetime.strptime(target_date, '%Y-%m-%d').strftime('%Y-%m-%d')
            if file_date != target_dt:
                return CheckResult(
                    "Salary File", CheckResult.WARN,
                    f"Date mismatch — file is for {file_date}, target is {target_dt}",
                    "This will cause 0 player matches if games differ between dates."
                )
        except ValueError:
            pass

    # Load and do basic checks
    try:
        import pandas as pd
        df = pd.read_csv(salary_path)
        n_players = len(df)
        if n_players < 20:
            return CheckResult(
                "Salary File", CheckResult.WARN,
                f"Only {n_players} players — very small slate"
            )

        # Check required columns
        required = ['Name', 'Salary', 'TeamAbbrev']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return CheckResult(
                "Salary File", CheckResult.FAIL,
                f"Missing required columns: {missing}",
                f"Columns found: {list(df.columns)}"
            )

        teams = df['TeamAbbrev'].dropna().unique()
        n_games = len(teams) // 2
        return CheckResult(
            "Salary File", CheckResult.PASS,
            f"{n_players} players, {len(teams)} teams ({n_games} games) — {fname}"
        )
    except Exception as e:
        return CheckResult(
            "Salary File", CheckResult.FAIL,
            f"Could not parse CSV: {e}"
        )


def check_vegas_data(vegas_games: list, slate_teams: list) -> CheckResult:
    """Verify Vegas lines loaded and cover all slate teams."""
    if not vegas_games:
        return CheckResult(
            "Vegas Lines", CheckResult.FAIL,
            "No Vegas data loaded — projections will lack win probability and team totals",
            "Check Odds API key or add VegasNHL CSV to vegas/ folder."
        )

    # Extract teams from Vegas games
    from config import TEAM_FULL_NAME_TO_ABBREV
    name_to_abbrev = {k.upper(): v for k, v in TEAM_FULL_NAME_TO_ABBREV.items()}
    vegas_teams = set()
    for game in vegas_games:
        matchup = game.get('matchup', '')
        parts = matchup.split(' @ ')
        if len(parts) == 2:
            away = name_to_abbrev.get(parts[0].strip().upper())
            home = name_to_abbrev.get(parts[1].strip().upper())
            if away:
                vegas_teams.add(away)
            if home:
                vegas_teams.add(home)

    missing = [t for t in slate_teams if t not in vegas_teams]
    if missing:
        return CheckResult(
            "Vegas Lines", CheckResult.WARN,
            f"Missing Vegas data for {len(missing)} slate teams: {', '.join(missing)}",
            "These teams will use default implied totals — less accurate projections."
        )

    # Check for game totals
    games_with_totals = sum(1 for g in vegas_games if g.get('game_total') is not None)
    if games_with_totals == 0:
        return CheckResult(
            "Vegas Lines", CheckResult.WARN,
            f"{len(vegas_games)} games loaded but none have game totals"
        )

    return CheckResult(
        "Vegas Lines", CheckResult.PASS,
        f"{len(vegas_games)} games, {len(vegas_teams)} teams with lines"
    )


def check_nhl_schedule(target_date: str) -> CheckResult:
    """Verify there are NHL games today."""
    try:
        from nhl_api import NHLAPIClient
        client = NHLAPIClient(rate_limit_delay=0.2)
        schedule = client.get_schedule(target_date)

        games_today = []
        for day in schedule.get('gameWeek', []):
            if day.get('date') == target_date:
                games_today = day.get('games', [])
                break

        if not games_today:
            return CheckResult(
                "NHL Schedule", CheckResult.FAIL,
                f"No NHL games on {target_date}",
                "Check if date is correct. NHL may be on break (All-Star, Olympics, etc.)."
            )

        # Check game states
        n_scheduled = sum(1 for g in games_today if g.get('gameState') in ('FUT', 'PRE'))
        n_live = sum(1 for g in games_today if g.get('gameState') in ('LIVE', 'CRIT'))
        n_final = sum(1 for g in games_today if g.get('gameState') in ('FINAL', 'OFF'))

        detail_parts = []
        if n_scheduled:
            detail_parts.append(f"{n_scheduled} upcoming")
        if n_live:
            detail_parts.append(f"{n_live} live")
        if n_final:
            detail_parts.append(f"{n_final} final")
        detail = ", ".join(detail_parts) if detail_parts else ""

        if n_scheduled == 0 and n_final > 0:
            return CheckResult(
                "NHL Schedule", CheckResult.WARN,
                f"{len(games_today)} games but all are final — slate may be over",
                detail
            )

        return CheckResult(
            "NHL Schedule", CheckResult.PASS,
            f"{len(games_today)} games on {target_date}",
            detail
        )
    except Exception as e:
        return CheckResult(
            "NHL Schedule", CheckResult.WARN,
            f"Could not fetch schedule: {e}",
            "Pipeline will still run, but can't verify games exist."
        )


def check_data_pipeline(data: dict) -> CheckResult:
    """Verify the data pipeline returned usable data."""
    issues = []
    details = []

    # Skaters
    skaters = data.get('skaters')
    if skaters is None or (hasattr(skaters, 'empty') and skaters.empty):
        issues.append("No skater data")
    else:
        n = len(skaters)
        details.append(f"Skaters: {n}")
        if n < 100:
            issues.append(f"Only {n} skaters (expected 500+)")

    # Goalies
    goalies = data.get('goalies')
    if goalies is None or (hasattr(goalies, 'empty') and goalies.empty):
        issues.append("No goalie data")
    else:
        n = len(goalies)
        details.append(f"Goalies: {n}")
        if n < 20:
            issues.append(f"Only {n} goalies (expected 60+)")

    # Teams
    teams = data.get('teams')
    if teams is None or (hasattr(teams, 'empty') and teams.empty):
        issues.append("No team data")
    else:
        details.append(f"Teams: {len(teams)}")

    # Schedule
    schedule = data.get('schedule')
    if schedule is None or (hasattr(schedule, 'empty') and schedule.empty):
        issues.append("No schedule data")
    else:
        details.append(f"Schedule: {len(schedule)} games")

    if issues:
        return CheckResult(
            "Data Pipeline", CheckResult.FAIL,
            "; ".join(issues),
            "\n".join(details) if details else ""
        )

    return CheckResult(
        "Data Pipeline", CheckResult.PASS,
        "All data sources loaded",
        "\n".join(details)
    )


def check_projections(skaters_merged, goalies_merged, slate_teams: list) -> CheckResult:
    """Verify projections are reasonable."""
    import pandas as pd
    issues = []
    details = []

    # Skater checks
    n_sk = len(skaters_merged) if skaters_merged is not None else 0
    if n_sk == 0:
        issues.append("0 skaters matched to DK pool")
    else:
        details.append(f"Skaters matched: {n_sk}")

        if 'projected_fpts' in skaters_merged.columns:
            fpts = skaters_merged['projected_fpts']

            # Negative projections
            n_neg = (fpts < 0).sum()
            if n_neg > 0:
                issues.append(f"{n_neg} skaters with negative projections")

            # Absurdly high
            n_high = (fpts > 25).sum()
            if n_high > 0:
                details.append(f"⚠️  {n_high} skaters projected >25 FPTS")

            # Mean sanity
            mean_fpts = fpts.mean()
            details.append(f"Skater mean: {mean_fpts:.1f} FPTS")
            if mean_fpts < 2.0:
                issues.append(f"Skater mean too low ({mean_fpts:.1f}) — bias correction may be off")
            elif mean_fpts > 12.0:
                issues.append(f"Skater mean too high ({mean_fpts:.1f}) — bias correction may be off")

        # Team coverage
        if 'team' in skaters_merged.columns:
            proj_teams = set(skaters_merged['team'].unique())
            missing_teams = [t for t in slate_teams if t not in proj_teams]
            if missing_teams:
                details.append(f"Missing teams in projections: {', '.join(missing_teams)}")

    # Goalie checks
    n_g = len(goalies_merged) if goalies_merged is not None else 0
    if n_g == 0:
        issues.append("0 goalies matched to DK pool")
    else:
        details.append(f"Goalies matched: {n_g}")

        if 'projected_fpts' in goalies_merged.columns:
            g_fpts = goalies_merged['projected_fpts']
            n_neg_g = (g_fpts < 0).sum()
            if n_neg_g > 0:
                details.append(f"⚠️  {n_neg_g} goalies with negative projections (can happen)")
            details.append(f"Goalie range: {g_fpts.min():.1f} – {g_fpts.max():.1f} FPTS")

    # Salary checks
    if n_sk > 0 and 'salary' in skaters_merged.columns:
        no_salary = skaters_merged['salary'].isna().sum()
        if no_salary > 0:
            issues.append(f"{no_salary} skaters missing salary")

    # Duplicate check
    if n_sk > 0 and 'name' in skaters_merged.columns:
        dupes = skaters_merged['name'].duplicated().sum()
        if dupes > 0:
            dupe_names = skaters_merged[skaters_merged['name'].duplicated(keep=False)]['name'].unique()
            issues.append(f"{dupes} duplicate player(s): {', '.join(dupe_names[:5])}")

    if issues:
        return CheckResult(
            "Projections", CheckResult.FAIL if "0 skaters" in str(issues) or "0 goalies" in str(issues) else CheckResult.WARN,
            "; ".join(issues),
            "\n".join(details)
        )

    return CheckResult(
        "Projections", CheckResult.PASS,
        f"{n_sk} skaters + {n_g} goalies projected",
        "\n".join(details)
    )


def check_confirmed_goalies(stack_builder, goalies_merged, slate_teams: list) -> CheckResult:
    """Verify confirmed goalies are in the pool."""
    if stack_builder is None:
        return CheckResult(
            "Confirmed Goalies", CheckResult.WARN,
            "Line scraper not run — goalie confirmation unknown",
            "Run without --no-stacks to fetch goalie confirmations."
        )

    confirmed = stack_builder.get_all_starting_goalies()
    if not confirmed:
        return CheckResult(
            "Confirmed Goalies", CheckResult.WARN,
            "No confirmed goalies found on DailyFaceoff",
            "Goalies may not be confirmed yet. Check closer to game time."
        )

    n_confirmed = len(confirmed)
    n_games = len(slate_teams) // 2

    details = []
    for team, name in sorted(confirmed.items()):
        in_pool = "✓" if goalies_merged is not None and len(goalies_merged) > 0 and \
                         any(name.lower() in n.lower() for n in goalies_merged['name'].tolist()) else "✗"
        details.append(f"{team}: {name} [{in_pool}]")

    if n_confirmed < n_games:
        return CheckResult(
            "Confirmed Goalies", CheckResult.WARN,
            f"Only {n_confirmed}/{n_games * 2} goalies confirmed",
            "\n".join(details)
        )

    # Check if confirmed goalies are in DK pool
    if goalies_merged is not None and len(goalies_merged) > 0:
        in_pool = len(goalies_merged)
        if in_pool < n_confirmed:
            return CheckResult(
                "Confirmed Goalies", CheckResult.WARN,
                f"{n_confirmed} confirmed but only {in_pool} in DK pool after merge",
                "\n".join(details)
            )

    return CheckResult(
        "Confirmed Goalies", CheckResult.PASS,
        f"{n_confirmed} goalies confirmed",
        "\n".join(details)
    )


def check_lineup_feasibility(player_pool) -> CheckResult:
    """Verify a valid DK lineup can be built from the pool."""
    import pandas as pd

    if player_pool is None or len(player_pool) == 0:
        return CheckResult(
            "Lineup Feasibility", CheckResult.FAIL,
            "Empty player pool — no lineup possible"
        )

    # Check position coverage
    positions = player_pool['position'].str.upper() if 'position' in player_pool.columns else pd.Series()

    # Also check dk_pos if available (more accurate for DK eligibility)
    dk_pos = player_pool['dk_pos'].str.upper() if 'dk_pos' in player_pool.columns else positions

    n_c = ((dk_pos == 'C') | (positions == 'C')).sum()
    n_w = ((dk_pos == 'W') | (dk_pos == 'LW') | (dk_pos == 'RW') |
           (positions == 'W') | (positions == 'LW') | (positions == 'RW')).sum()
    n_d = ((dk_pos == 'D') | (positions == 'D')).sum()
    n_g = ((dk_pos == 'G') | (positions == 'G')).sum()

    issues = []
    if n_c < 2:
        issues.append(f"Only {n_c} Centers (need 2)")
    if n_w < 3:
        issues.append(f"Only {n_w} Wings (need 3)")
    if n_d < 2:
        issues.append(f"Only {n_d} Defensemen (need 2)")
    if n_g < 1:
        issues.append(f"No Goalies in pool")

    # Salary feasibility — can we fill a roster under $50k?
    if 'salary' in player_pool.columns:
        min_salaries = []
        for pos, need in [('C', 2), ('W', 3), ('D', 2), ('G', 1)]:
            pos_players = player_pool[
                (player_pool.get('dk_pos', player_pool.get('position', '')).str.upper().isin(
                    [pos] + (['LW', 'RW'] if pos == 'W' else [])
                ))
            ].nsmallest(need, 'salary') if 'salary' in player_pool.columns else pd.DataFrame()
            if len(pos_players) >= need:
                min_salaries.extend(pos_players['salary'].tolist())

        # UTIL = cheapest remaining skater
        skaters_only = player_pool[player_pool['position'].str.upper().isin(['C', 'W', 'LW', 'RW', 'D'])]
        if len(skaters_only) > 7:  # Need 7 skaters + 1 UTIL
            cheapest_remaining = skaters_only.nsmallest(8, 'salary')['salary'].iloc[-1] if len(skaters_only) >= 8 else 0
            min_salaries.append(cheapest_remaining)

        if min_salaries:
            min_total = sum(min_salaries)
            cap = 50000
            if min_total > cap:
                issues.append(f"Min salary ${min_total:,} exceeds ${cap:,} cap")

    details = f"Pool: {n_c}C, {n_w}W, {n_d}D, {n_g}G = {len(player_pool)} total"

    if issues:
        return CheckResult(
            "Lineup Feasibility", CheckResult.FAIL,
            "; ".join(issues),
            details
        )

    return CheckResult(
        "Lineup Feasibility", CheckResult.PASS,
        f"Roster requirements satisfiable",
        details
    )


def check_nst_freshness(target_date: str) -> CheckResult:
    """Check if NST danger-zone data can be fetched (goalie model dependency)."""
    try:
        from goalie_model import fetch_team_danger_stats
        from config import CURRENT_SEASON

        # Try fetching — just check it doesn't error
        season_start = "2025-10-07" if "2025" in CURRENT_SEASON else "2024-10-04"
        df = fetch_team_danger_stats(thru_date=target_date, gpf="410")
        if df is None or df.empty:
            return CheckResult(
                "NST Danger Data", CheckResult.WARN,
                "NST returned empty data — goalie model will use fallback rates",
                "Site may be down or season hasn't started."
            )

        n_teams = len(df)
        return CheckResult(
            "NST Danger Data", CheckResult.PASS,
            f"Fetched danger-zone stats for {n_teams} teams"
        )
    except ImportError:
        return CheckResult(
            "NST Danger Data", CheckResult.WARN,
            "goalie_model.py not found — using baseline goalie projections"
        )
    except Exception as e:
        return CheckResult(
            "NST Danger Data", CheckResult.WARN,
            f"NST fetch failed: {e}",
            "Goalie model will fall back to league-average shot distribution."
        )


def check_odds_api() -> CheckResult:
    """Check if Odds API is reachable and has NHL data."""
    try:
        api_key = os.environ.get('ODDS_API_KEY', '')
        if not api_key:
            # Check .env file
            env_path = Path(__file__).parent / '.env'
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith('ODDS_API_KEY='):
                            api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                            break

        if not api_key:
            return CheckResult(
                "Odds API", CheckResult.WARN,
                "No ODDS_API_KEY found — will rely on Vegas CSV fallback",
                "Set ODDS_API_KEY env var or add to .env file for live odds."
            )

        return CheckResult(
            "Odds API", CheckResult.PASS,
            "API key configured"
        )
    except Exception as e:
        return CheckResult(
            "Odds API", CheckResult.WARN,
            f"Could not check API: {e}"
        )


# ================================================================
#  Main Validation Runner
# ================================================================

def run_validation(salary_path: str = None,
                   target_date: str = None,
                   vegas_games: list = None,
                   data: dict = None,
                   skaters_merged=None,
                   goalies_merged=None,
                   player_pool=None,
                   stack_builder=None,
                   slate_teams: list = None,
                   quick: bool = False) -> ValidationReport:
    """
    Run all pre-flight checks.

    Can be called in two modes:
    1. Quick mode (quick=True): Only checks files and schedule, no data fetching.
       Good for a fast sanity check before running the full pipeline.
    2. Full mode (quick=False): Checks everything including projections.
       Called from main.py after data is loaded but before lineup generation.

    Args:
        salary_path: Path to DK salary CSV
        target_date: Target date (YYYY-MM-DD)
        vegas_games: List of game dicts from Vegas (if already loaded)
        data: Pipeline data dict (if already loaded)
        skaters_merged: Merged skater projections (if already generated)
        goalies_merged: Merged goalie projections (if already generated)
        player_pool: Combined player pool (if already built)
        stack_builder: StackBuilder instance (if already created)
        slate_teams: List of team abbreviations on the slate
        quick: If True, only run file/schedule checks (no data pipeline)

    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport()
    target_date = target_date or datetime.now().strftime('%Y-%m-%d')

    # 1. Schedule check (are there games?)
    report.add(check_nhl_schedule(target_date))

    # 2. Salary file check
    report.add(check_salary_file(salary_path, target_date))

    # 3. Odds API / Vegas check
    report.add(check_odds_api())
    if vegas_games is not None:
        report.add(check_vegas_data(vegas_games, slate_teams or []))

    # 4. NST accessibility
    if not quick:
        report.add(check_nst_freshness(target_date))

    # 5. Data pipeline check (if data already loaded)
    if data is not None:
        report.add(check_data_pipeline(data))

    # 6. Projection quality check
    if skaters_merged is not None or goalies_merged is not None:
        report.add(check_projections(skaters_merged, goalies_merged, slate_teams or []))

    # 7. Confirmed goalies
    if stack_builder is not None:
        report.add(check_confirmed_goalies(stack_builder, goalies_merged, slate_teams or []))

    # 8. Lineup feasibility
    if player_pool is not None:
        report.add(check_lineup_feasibility(player_pool))

    return report


# ================================================================
#  Standalone Mode
# ================================================================

def main():
    """Run validation standalone (quick mode — no full pipeline)."""
    import argparse

    parser = argparse.ArgumentParser(description='NHL DFS Pre-Flight Validation')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--salaries', type=str, default=None,
                        help='Path to DK salary CSV')
    parser.add_argument('--vegas', type=str, default=None,
                        help='Path to Vegas CSV')
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime('%Y-%m-%d')
    project_dir = Path(__file__).parent

    # Auto-detect salary file
    salary_path = args.salaries
    if not salary_path:
        from config import DAILY_SALARIES_DIR
        salaries_dir = project_dir / DAILY_SALARIES_DIR
        salary_files = list(salaries_dir.glob('DKSalaries*.csv')) if salaries_dir.exists() else []
        if not salary_files:
            salary_files = list(project_dir.glob('DKSalaries*.csv'))
        if salary_files:
            salary_path = str(sorted(salary_files)[-1])

    # Auto-detect Vegas file
    vegas_games = None
    if args.vegas:
        try:
            from main import _load_vegas_csv
            vegas_games = _load_vegas_csv(args.vegas)
        except Exception:
            pass

    report = run_validation(
        salary_path=salary_path,
        target_date=target_date,
        vegas_games=vegas_games,
        quick=True
    )
    report.print_report()

    sys.exit(0 if report.is_go else 1)


if __name__ == "__main__":
    main()
