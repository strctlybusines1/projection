"""
Goalie Model Backtest — Compare new danger-zone model vs old projections.

For each backtest date:
1. Fetch NST team + goalie stats up to the day BEFORE (no look-ahead)
2. Read Win%, Team, Opp from the FC sheet of the backtest xlsx
3. Run the new goalie model
4. Compare to actual DK scores and old projections from the xlsx

Usage:
    python goalie_backtest.py
"""

import pandas as pd
import numpy as np
import requests
import time
import zipfile
import tempfile
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from config import GOALIE_SCORING, GOALIE_BONUSES


# ================================================================
#  NST Fetching with Date Filters
# ================================================================

NST_BASE = "https://www.naturalstattrick.com"
SEASON_OPEN = "2025-10-07"  # Opening night 2025-26


def _nst_fetch(url: str) -> Optional[pd.DataFrame]:
    """Fetch first HTML table from NST URL."""
    time.sleep(2.0)  # Rate limit
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        return tables[0] if tables else None
    except Exception as e:
        print(f"    NST error: {e}")
        return None


def _normalize_team(team_str: str) -> str:
    """Normalize NST team names to standard 3-letter codes."""
    if pd.isna(team_str):
        return ''
    team_str = str(team_str).strip()
    mapping = {
        'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS',
        'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR',
        'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ',
        'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM',
        'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN',
        'Montréal Canadiens': 'MTL', 'Montreal Canadiens': 'MTL', 'Nashville Predators': 'NSH',
        'New Jersey Devils': 'NJD', 'New York Islanders': 'NYI', 'New York Rangers': 'NYR',
        'Ottawa Senators': 'OTT', 'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT',
        'San Jose Sharks': 'SJS', 'Seattle Kraken': 'SEA', 'St. Louis Blues': 'STL',
        'St Louis Blues': 'STL', 'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR',
        'Utah Hockey Club': 'UTA', 'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK',
        'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG',
        # Short codes
        'T.B': 'TBL', 'N.J': 'NJD', 'L.A': 'LAK', 'S.J': 'SJS',
        'ANA': 'ANA', 'ARI': 'ARI', 'BOS': 'BOS', 'BUF': 'BUF', 'CGY': 'CGY',
        'CAR': 'CAR', 'CHI': 'CHI', 'COL': 'COL', 'CBJ': 'CBJ', 'DAL': 'DAL',
        'DET': 'DET', 'EDM': 'EDM', 'FLA': 'FLA', 'LAK': 'LAK', 'MIN': 'MIN',
        'MTL': 'MTL', 'NSH': 'NSH', 'NJD': 'NJD', 'NYI': 'NYI', 'NYR': 'NYR',
        'OTT': 'OTT', 'PHI': 'PHI', 'PIT': 'PIT', 'SJS': 'SJS', 'SEA': 'SEA',
        'STL': 'STL', 'TBL': 'TBL', 'TOR': 'TOR', 'UTA': 'UTA', 'VAN': 'VAN',
        'VGK': 'VGK', 'WSH': 'WSH', 'WPG': 'WPG',
        # FC sheet abbreviations
        'TB': 'TBL', 'NJ': 'NJD', 'LA': 'LAK', 'SJ': 'SJS',
    }
    return mapping.get(team_str, team_str.upper())


def fetch_nst_team_stats(thru_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch team danger-zone shot stats from season open through thru_date.

    Returns DataFrame indexed by team with per-game shot rates:
        hdsf_pg, hdsa_pg, mdsf_pg, mdsa_pg, ldsf_pg, ldsa_pg
    """
    url = (
        f"{NST_BASE}/teamtable.php?"
        f"fromseason=20252026&thruseason=20252026"
        f"&stype=2&sit=all&score=all&rate=n&team=all&loc=B"
        f"&gpf=410&fd={SEASON_OPEN}&td={thru_date}"
    )
    df = _nst_fetch(url)
    if df is None:
        return None

    result = pd.DataFrame()
    result['team'] = df['Team'].apply(_normalize_team)
    gp = df['GP'].replace(0, 1)
    result['hdsf_pg'] = df['HDSF'] / gp
    result['hdsa_pg'] = df['HDSA'] / gp
    result['mdsf_pg'] = df['MDSF'] / gp
    result['mdsa_pg'] = df['MDSA'] / gp
    result['ldsf_pg'] = df['LDSF'] / gp
    result['ldsa_pg'] = df['LDSA'] / gp
    return result.set_index('team')


def fetch_nst_goalie_stats(thru_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch individual goalie danger-zone save % from season open through thru_date.

    Returns DataFrame with:
        name, team, gp, hdsv_pct, mdsv_pct, ldsv_pct, sa_pg
    """
    url = (
        f"{NST_BASE}/playerteams.php?"
        f"fromseason=20252026&thruseason=20252026"
        f"&stype=2&sit=all&score=all&stdoi=g&rate=n"
        f"&team=ALL&pos=S&loc=B&toi=0"
        f"&gpfilt=none&fd={SEASON_OPEN}&td={thru_date}"
        f"&tgp=82&lines=single&dession=false"
    )
    df = _nst_fetch(url)
    if df is None:
        return None

    result = pd.DataFrame()
    result['name'] = df['Player']
    result['team'] = df['Team'].apply(_normalize_team)
    result['gp'] = df['GP']
    result['hdsv_pct'] = pd.to_numeric(df['HDSV%'], errors='coerce')
    result['mdsv_pct'] = pd.to_numeric(df['MDSV%'], errors='coerce')
    result['ldsv_pct'] = pd.to_numeric(df['LDSV%'], errors='coerce')
    result['shots_against'] = df['Shots Against']
    result['sa_pg'] = result['shots_against'] / result['gp'].replace(0, 1)
    return result


# ================================================================
#  Backtest xlsx Reading
# ================================================================

def fix_strict_ooxml(filepath: str) -> str:
    """Fix strict OOXML namespace so openpyxl can read the file."""
    tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    tmp.close()
    with zipfile.ZipFile(filepath, 'r') as zin:
        with zipfile.ZipFile(tmp.name, 'w') as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith('.xml') or item.filename.endswith('.rels'):
                    text = data.decode('utf-8')
                    text = text.replace(
                        'http://purl.oclc.org/ooxml/spreadsheetml/main',
                        'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
                    )
                    text = text.replace(
                        'http://purl.oclc.org/ooxml/officeDocument/relationships',
                        'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
                    )
                    data = text.encode('utf-8')
                zout.writestr(item, data)
    return tmp.name


def read_backtest_goalies(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read goalie data from backtest xlsx.

    Returns:
        fc_goalies: DataFrame from FC sheet (confirmed starters with Win%, Opp, etc.)
        dk_actuals: DataFrame from Actual sheet (Player, FPTS)
    """
    fixed = fix_strict_ooxml(filepath)

    # Get available sheet names
    xl = pd.ExcelFile(fixed, engine='openpyxl')
    sheets = xl.sheet_names

    # Find FC sheet (case-insensitive)
    fc_sheet = None
    for s in sheets:
        if s.lower() == 'fc':
            fc_sheet = s
            break
    if fc_sheet is None:
        print(f"    Warning: No FC sheet found in {filepath} (sheets: {sheets})")
        return pd.DataFrame(), pd.DataFrame()

    # Find Actual sheet
    actual_sheet = None
    for s in sheets:
        if s.lower() == 'actual':
            actual_sheet = s
            break
    if actual_sheet is None:
        print(f"    Warning: No Actual sheet found in {filepath}")
        return pd.DataFrame(), pd.DataFrame()

    fc = pd.read_excel(fixed, sheet_name=fc_sheet, engine='openpyxl')
    actual = pd.read_excel(fixed, sheet_name=actual_sheet, engine='openpyxl')

    # Find Projection sheet for old model's projected_fpts
    proj_sheet = None
    for s in sheets:
        if s.lower() in ('projection', 'projections', 'projected'):
            proj_sheet = s
            break
    old_proj_df = None
    if proj_sheet:
        proj_df = pd.read_excel(fixed, sheet_name=proj_sheet, engine='openpyxl')
        if 'projected_fpts' in proj_df.columns and 'name' in proj_df.columns:
            # Filter to goalies
            if 'position' in proj_df.columns:
                old_proj_df = proj_df[proj_df['position'] == 'G'][['name', 'projected_fpts']].copy()
                old_proj_df = old_proj_df.rename(columns={'projected_fpts': 'old_projected_fpts'})

    # Filter FC to confirmed goalies only
    goalies = fc[
        (fc['Pos'] == 'G') &
        (fc['Start/Line'].str.lower().isin(['confirm', 'confirmed', 'likely', 'prob']))
    ].copy()

    # Clean Opp column: strip "vs " and "@ "
    goalies['is_home'] = goalies['Opp'].str.startswith('vs ')
    goalies['opponent'] = goalies['Opp'].str.replace(r'^(vs |@ )', '', regex=True).str.strip()
    goalies['opponent'] = goalies['opponent'].apply(_normalize_team)
    goalies['team_clean'] = goalies['Team'].apply(_normalize_team)
    goalies['win_pct'] = pd.to_numeric(goalies['Win %'], errors='coerce')

    # Get actuals for goalies
    dk_goalies = actual[actual['Roster Position'] == 'G'].copy()

    # Merge old model projections from Projection sheet into FC goalies
    if old_proj_df is not None and not old_proj_df.empty:
        goalies = goalies.merge(
            old_proj_df, left_on='Player', right_on='name', how='left'
        )
        if 'old_projected_fpts' in goalies.columns:
            goalies['old_proj'] = goalies['old_projected_fpts']
        goalies = goalies.drop(columns=['name_y', 'old_projected_fpts'], errors='ignore')
        if 'name_x' in goalies.columns:
            goalies = goalies.rename(columns={'name_x': 'name'})
    else:
        goalies['old_proj'] = goalies.get('My Proj', np.nan)

    return goalies, dk_goalies


# ================================================================
#  Single-Date Projection
# ================================================================

def project_date(
    date_str: str,
    team_stats: pd.DataFrame,
    goalie_stats: pd.DataFrame,
    fc_goalies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project all confirmed goalies for a single date using the danger-zone model.

    Returns DataFrame with: name, team, opponent, projected_fpts, old_proj, actual
    """
    results = []

    for _, g in fc_goalies.iterrows():
        name = g['Player']
        team = g['team_clean']
        opp = g['opponent']
        win_pct = g['win_pct'] if pd.notna(g['win_pct']) else 0.50
        is_home = g['is_home']
        old_proj = g.get('old_proj', np.nan)
        actual = g.get('Score', np.nan)

        # Look up team stats
        if opp not in team_stats.index or team not in team_stats.index:
            results.append({
                'name': name, 'team': team, 'opponent': opp,
                'projected_fpts': np.nan, 'old_proj': old_proj, 'actual': actual,
                'note': f'Team not found: opp={opp} team={team}',
            })
            continue

        opp_stats = team_stats.loc[opp]
        team_def = team_stats.loc[team]

        # Find goalie's individual save %
        goalie_rows = goalie_stats[
            goalie_stats['name'].str.lower().str.contains(
                name.lower().split()[-1]
            ) & (goalie_stats['team'] == team)
        ]
        if goalie_rows.empty:
            # Broader: just last name
            goalie_rows = goalie_stats[
                goalie_stats['name'].str.lower().str.contains(name.lower().split()[-1])
            ]

        if goalie_rows.empty:
            results.append({
                'name': name, 'team': team, 'opponent': opp,
                'projected_fpts': np.nan, 'old_proj': old_proj, 'actual': actual,
                'note': f'Goalie not found in NST: {name}',
            })
            continue

        goalie = goalie_rows.iloc[0]

        # ── Geometric mean shots by zone ──
        hd_shots = np.sqrt(opp_stats['hdsf_pg'] * team_def['hdsa_pg'])
        md_shots = np.sqrt(opp_stats['mdsf_pg'] * team_def['mdsa_pg'])
        ld_shots = np.sqrt(opp_stats['ldsf_pg'] * team_def['ldsa_pg'])
        total_shots = hd_shots + md_shots + ld_shots

        # ── Expected GA by zone ──
        hd_ga = hd_shots * (1 - goalie['hdsv_pct'])
        md_ga = md_shots * (1 - goalie['mdsv_pct'])
        ld_ga = ld_shots * (1 - goalie['ldsv_pct'])
        total_ga = hd_ga + md_ga + ld_ga
        total_saves = total_shots - total_ga

        # ── DK FPTS ──
        fpts = (
            total_saves * GOALIE_SCORING['save'] +
            total_ga * GOALIE_SCORING['goal_against'] +
            win_pct * GOALIE_SCORING['win']
        )

        # OT loss
        ot_loss_rate = (1 - win_pct) * 0.25
        fpts += ot_loss_rate * GOALIE_SCORING['overtime_loss']

        # Shutout
        shutout_prob = max(0, 0.15 - total_ga * 0.04)
        fpts += shutout_prob * GOALIE_SCORING['shutout_bonus']

        # 35+ saves bonus
        if total_saves > 25:
            std_saves = total_saves * 0.15
            try:
                from scipy.stats import norm
                prob_35 = 1 - norm.cdf(35, loc=total_saves, scale=max(std_saves, 1))
            except ImportError:
                prob_35 = max(0, (total_saves - 30) / 20)
            fpts += prob_35 * GOALIE_BONUSES['thirty_five_plus_saves']

        # Home ice
        if is_home:
            fpts *= 1.02

        results.append({
            'name': name,
            'team': team,
            'opponent': opp,
            'projected_fpts': round(fpts, 2),
            'old_proj': old_proj,
            'actual': actual,
            'exp_shots': round(total_shots, 1),
            'exp_saves': round(total_saves, 1),
            'exp_ga': round(total_ga, 2),
            'win_pct': win_pct,
            'note': '',
        })

    return pd.DataFrame(results)


# ================================================================
#  Main Backtest Loop
# ================================================================

def run_backtest():
    """Run the full backtest across all dates."""

    backtests_dir = Path('backtests')

    # Dates to backtest (M.D.YY format for filenames, YYYY-MM-DD for NST)
    dates = [
        ('1.23.26', '2026-01-23', '2026-01-22'),
        ('1.26.26', '2026-01-26', '2026-01-25'),
        ('1.28.26', '2026-01-28', '2026-01-27'),
        ('1.29.26', '2026-01-29', '2026-01-28'),
        ('1.31.26', '2026-01-31', '2026-01-30'),
        ('2.1.26',  '2026-02-01', '2026-01-31'),
        ('2.2.26',  '2026-02-02', '2026-02-01'),
        ('2.3.26',  '2026-02-03', '2026-02-02'),
        ('2.4.26',  '2026-02-04', '2026-02-03'),
        ('2.5.26',  '2026-02-05', '2026-02-04'),
    ]

    all_results = []
    date_summaries = []

    print("=" * 75)
    print("GOALIE MODEL BACKTEST — Danger Zone Matchup Approach")
    print("=" * 75)

    for file_date, slate_date, nst_thru_date in dates:
        bt_file = backtests_dir / f"{file_date}_nhl_backtest.xlsx"
        if not bt_file.exists():
            print(f"\n  {file_date}: File not found, skipping")
            continue

        print(f"\n{'─' * 75}")
        print(f"  DATE: {slate_date} (NST data through {nst_thru_date})")
        print(f"{'─' * 75}")

        # 1. Fetch NST data through the day before
        print(f"  Fetching NST team stats...")
        team_stats = fetch_nst_team_stats(nst_thru_date)
        if team_stats is None:
            print(f"    Failed to fetch team stats, skipping")
            continue

        print(f"  Fetching NST goalie stats...")
        goalie_stats = fetch_nst_goalie_stats(nst_thru_date)
        if goalie_stats is None:
            print(f"    Failed to fetch goalie stats, skipping")
            continue

        # 2. Read backtest xlsx
        fc_goalies, dk_actuals = read_backtest_goalies(str(bt_file))
        print(f"  Confirmed starters: {len(fc_goalies)}")

        if fc_goalies.empty:
            print(f"  No confirmed goalies, skipping")
            continue

        # 3. Project
        results = project_date(slate_date, team_stats, goalie_stats, fc_goalies)

        # Filter to goalies with actual scores
        scored = results[results['actual'].notna() & (results['projected_fpts'].notna())].copy()

        if scored.empty:
            print(f"  No scored goalies, skipping")
            continue

        # 4. Calculate errors
        scored['new_error'] = scored['projected_fpts'] - scored['actual']
        scored['new_abs_error'] = scored['new_error'].abs()
        scored['old_error'] = scored['old_proj'] - scored['actual']
        scored['old_abs_error'] = scored['old_error'].abs()
        scored['date'] = slate_date

        # Print per-date results
        new_mae = scored['new_abs_error'].mean()
        old_mae = scored['old_abs_error'].mean()
        new_bias = scored['new_error'].mean()
        old_bias = scored['old_error'].mean()

        print(f"\n  {'Name':<25} {'Team':<5} {'Opp':<5} {'New':>6} {'Old':>6} {'Act':>6} {'N_Err':>6} {'O_Err':>6}")
        print(f"  {'-' * 70}")
        for _, r in scored.iterrows():
            print(f"  {r['name']:<25} {r['team']:<5} {r['opponent']:<5} "
                  f"{r['projected_fpts']:6.1f} {r['old_proj']:6.1f} {r['actual']:6.1f} "
                  f"{r['new_error']:+6.1f} {r['old_error']:+6.1f}")

        print(f"\n  New Model MAE: {new_mae:.2f}  |  Old Model MAE: {old_mae:.2f}  |  Δ: {new_mae - old_mae:+.2f}")
        print(f"  New Model Bias: {new_bias:+.2f}  |  Old Model Bias: {old_bias:+.2f}")

        date_summaries.append({
            'date': slate_date,
            'n_goalies': len(scored),
            'new_mae': new_mae,
            'old_mae': old_mae,
            'new_bias': new_bias,
            'old_bias': old_bias,
            'delta_mae': new_mae - old_mae,
        })

        all_results.append(scored)

    # ================================================================
    #  Overall Summary
    # ================================================================
    if not all_results:
        print("\nNo results to summarize.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(date_summaries)

    print(f"\n{'=' * 75}")
    print(f"OVERALL RESULTS ({len(combined)} goalie starts across {len(summary)} dates)")
    print(f"{'=' * 75}")

    print(f"\n  {'Date':<12} {'N':>3} {'New MAE':>8} {'Old MAE':>8} {'Δ MAE':>8} {'New Bias':>9} {'Old Bias':>9}")
    print(f"  {'-' * 60}")
    for _, s in summary.iterrows():
        marker = '✓' if s['delta_mae'] < 0 else ' '
        print(f"  {s['date']:<12} {s['n_goalies']:3d} {s['new_mae']:8.2f} {s['old_mae']:8.2f} "
              f"{s['delta_mae']:+8.2f} {marker} {s['new_bias']:+9.2f} {s['old_bias']:+9.2f}")

    print(f"  {'-' * 60}")
    overall_new_mae = combined['new_abs_error'].mean()
    overall_old_mae = combined['old_abs_error'].mean()
    overall_new_bias = combined['new_error'].mean()
    overall_old_bias = combined['old_error'].mean()
    new_corr = combined[['projected_fpts', 'actual']].corr().iloc[0, 1]
    old_corr = combined[['old_proj', 'actual']].corr().iloc[0, 1]

    print(f"  {'TOTAL':<12} {len(combined):3d} {overall_new_mae:8.2f} {overall_old_mae:8.2f} "
          f"{overall_new_mae - overall_old_mae:+8.2f} {'✓' if overall_new_mae < overall_old_mae else ' '} "
          f"{overall_new_bias:+9.2f} {overall_old_bias:+9.2f}")

    print(f"\n  Correlation with actuals:")
    print(f"    New model: {new_corr:.3f}")
    print(f"    Old model: {old_corr:.3f}")

    wins_new = (summary['delta_mae'] < 0).sum()
    print(f"\n  New model wins on {wins_new}/{len(summary)} dates")

    if overall_new_mae < overall_old_mae:
        pct_improvement = (1 - overall_new_mae / overall_old_mae) * 100
        print(f"  New model is {pct_improvement:.1f}% better overall (lower MAE)")
    else:
        pct_worse = (overall_new_mae / overall_old_mae - 1) * 100
        print(f"  New model is {pct_worse:.1f}% worse overall (higher MAE)")
        print(f"  Consider adjustments or blending with old model")


if __name__ == '__main__':
    run_backtest()
