"""
Goalie Projection Model — Danger Zone Matchup Approach.

Projects goalie DK fantasy points using:
1. Geometric mean of opponent shot generation × team shot allowance per danger zone
2. Individual goalie save % by danger zone (HD, MD, LD) from Natural Stat Trick
3. Vegas-implied win probability for the win bonus

Data: Natural Stat Trick (all situations, per-game rates) + Vegas lines

Integration:
    In projections.py, generate_projections() calls:
        goalie_projections = self.project_goalies_baseline(goalie_features)

    Replace with:
        from goalie_model import GoalieProjectionModel
        goalie_model = GoalieProjectionModel()
        goalie_projections = goalie_model.project_goalies(
            goalie_features, data['schedule'], target_date,
            team_totals=team_totals, team_game_totals=team_game_totals,
        )

Standalone test:
    python goalie_model.py --goalie "Connor Hellebuyck" --team WPG --opp TOR --win 0.55 --home
    python goalie_model.py --all
"""

import pandas as pd
import numpy as np
import requests
import time
from io import StringIO
from typing import Dict, List, Optional

from config import GOALIE_SCORING, GOALIE_BONUSES


# ================================================================
#  NST Data Fetching
# ================================================================

NST_BASE = "https://www.naturalstattrick.com"
SEASON_OPEN = "2025-10-07"


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
        'T.B': 'TBL', 'N.J': 'NJD', 'L.A': 'LAK', 'S.J': 'SJS',
    }
    return mapping.get(team_str, team_str.upper().strip())


def _nst_fetch(url: str, rate_limit: float = 2.0) -> Optional[pd.DataFrame]:
    """Fetch first HTML table from NST URL with rate limiting."""
    time.sleep(rate_limit)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        return tables[0] if tables else None
    except Exception as e:
        print(f"    NST fetch error: {e}")
        return None


def fetch_team_danger_stats(thru_date: str = None) -> Optional[pd.DataFrame]:
    """
    Fetch team-level shot stats by danger zone (all situations, raw counts).

    Args:
        thru_date: If provided, only use data through this date (YYYY-MM-DD).
                   If None, uses full season to date.

    Returns:
        DataFrame indexed by team with per-game rates:
            hdsf_pg, hdsa_pg, mdsf_pg, mdsa_pg, ldsf_pg, ldsa_pg
    """
    print("  Fetching team danger-zone stats (NST, all situations)...")
    date_filter = f"&fd={SEASON_OPEN}&td={thru_date}" if thru_date else ""
    url = (
        f"{NST_BASE}/teamtable.php?"
        f"fromseason=20252026&thruseason=20252026"
        f"&stype=2&sit=all&score=all&rate=n&team=all&loc=B"
        f"&gpf=410{date_filter}"
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

    print(f"    {len(result)} teams loaded")
    return result.set_index('team')


def fetch_goalie_danger_stats(thru_date: str = None) -> Optional[pd.DataFrame]:
    """
    Fetch individual goalie save % by danger zone (all situations).

    Args:
        thru_date: If provided, only use data through this date (YYYY-MM-DD).
                   If None, uses full season to date.

    Returns:
        DataFrame with: name, team, gp, hdsv_pct, mdsv_pct, ldsv_pct
    """
    print("  Fetching goalie danger-zone stats (NST, all situations)...")
    date_filter = f"&fd={SEASON_OPEN}&td={thru_date}" if thru_date else ""
    url = (
        f"{NST_BASE}/playerteams.php?"
        f"fromseason=20252026&thruseason=20252026"
        f"&stype=2&sit=all&score=all&stdoi=g&rate=n"
        f"&team=ALL&pos=S&loc=B&toi=0"
        f"&gpfilt=none{date_filter}"
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
    result['sv_pct'] = pd.to_numeric(df['SV%'], errors='coerce')
    result['shots_against'] = df['Shots Against']
    result['sa_pg'] = result['shots_against'] / result['gp'].replace(0, 1)

    print(f"    {len(result)} goalies loaded")
    return result


# ================================================================
#  Goalie Projection Model
# ================================================================

class GoalieProjectionModel:
    """
    Project goalie DK FPTS using danger-zone matchup analysis.

    For each goalie on the slate:
    1. Get opponent's per-game shot generation by zone (HDSF, MDSF, LDSF)
    2. Get goalie's team per-game shots allowed by zone (HDSA, MDSA, LDSA)
    3. Expected shots = sqrt(Opp_SF * Team_SA) per zone (geometric mean)
    4. Expected GA per zone = Expected_Shots * (1 - Goalie's individual SV% for that zone)
    5. Expected saves per zone = Expected_Shots - Expected_GA
    6. DK FPTS = saves * 0.7 + GA * -3.5 + win% * 6.0 + bonuses
    """

    def __init__(self):
        self._team_stats = None
        self._goalie_stats = None

    def load_nst_data(self, thru_date: str = None, force_refresh: bool = False):
        """
        Fetch NST data. Called automatically by project_goalies if not loaded.

        Args:
            thru_date: Only use NST data through this date (for backtesting).
                       None = use full season to date (for live projections).
            force_refresh: Force re-fetch even if already loaded.
        """
        if self._team_stats is not None and not force_refresh:
            return
        self._team_stats = fetch_team_danger_stats(thru_date)
        self._goalie_stats = fetch_goalie_danger_stats(thru_date)

    def _find_goalie_stats(self, name: str, team: str) -> Optional[pd.Series]:
        """Find a goalie's danger-zone save percentages from NST data."""
        if self._goalie_stats is None:
            return None

        last_name = name.split()[-1].lower()

        # Match on last name + team
        matches = self._goalie_stats[
            self._goalie_stats['name'].str.lower().str.contains(last_name, na=False)
            & (self._goalie_stats['team'] == team)
        ]
        if not matches.empty:
            return matches.iloc[0]

        # Fallback: last name only (traded players)
        matches = self._goalie_stats[
            self._goalie_stats['name'].str.lower().str.contains(last_name, na=False)
        ]
        if not matches.empty:
            return matches.iloc[0]

        return None

    def _project_single(self, name: str, team: str, opponent: str,
                         win_prob: float, is_home: bool,
                         verbose: bool = False) -> Dict:
        """Project DK FPTS for one goalie matchup."""

        if self._team_stats is None or self._goalie_stats is None:
            return {'projected_fpts': np.nan, 'error': 'NST data unavailable'}

        if opponent not in self._team_stats.index:
            return {'projected_fpts': np.nan, 'error': f'Opponent {opponent} not in NST'}

        if team not in self._team_stats.index:
            return {'projected_fpts': np.nan, 'error': f'Team {team} not in NST'}

        goalie = self._find_goalie_stats(name, team)
        if goalie is None:
            return {'projected_fpts': np.nan, 'error': f'Goalie {name} not in NST'}

        opp = self._team_stats.loc[opponent]
        team_def = self._team_stats.loc[team]

        # ── Step 1: Expected shots by danger zone (geometric mean) ──
        hd_shots = np.sqrt(opp['hdsf_pg'] * team_def['hdsa_pg'])
        md_shots = np.sqrt(opp['mdsf_pg'] * team_def['mdsa_pg'])
        ld_shots = np.sqrt(opp['ldsf_pg'] * team_def['ldsa_pg'])
        total_shots = hd_shots + md_shots + ld_shots

        # ── Step 2: Expected GA by zone using goalie's individual save % ──
        hd_ga = hd_shots * (1 - goalie['hdsv_pct'])
        md_ga = md_shots * (1 - goalie['mdsv_pct'])
        ld_ga = ld_shots * (1 - goalie['ldsv_pct'])
        total_ga = hd_ga + md_ga + ld_ga

        # ── Step 3: Expected saves ──
        total_saves = total_shots - total_ga

        # ── Step 4: DK Fantasy Points ──
        fpts = (
            total_saves * GOALIE_SCORING['save'] +
            total_ga * GOALIE_SCORING['goal_against'] +
            win_prob * GOALIE_SCORING['win']
        )

        # OT loss (~25% of non-wins go to OT/SO)
        ot_loss_rate = (1 - win_prob) * 0.25
        fpts += ot_loss_rate * GOALIE_SCORING['overtime_loss']

        # Shutout probability
        shutout_prob = max(0, 0.15 - total_ga * 0.04)
        fpts += shutout_prob * GOALIE_SCORING['shutout_bonus']

        # 35+ saves bonus probability
        prob_35 = 0.0
        if total_saves > 25:
            std_saves = total_saves * 0.15
            try:
                from scipy.stats import norm
                prob_35 = 1 - norm.cdf(35, loc=total_saves, scale=max(std_saves, 1))
            except ImportError:
                prob_35 = max(0, (total_saves - 30) / 20)
            fpts += prob_35 * GOALIE_BONUSES['thirty_five_plus_saves']

        # Home ice advantage
        if is_home:
            fpts *= 1.02

        if verbose:
            print(f"\n  {'─' * 60}")
            print(f"  GOALIE PROJECTION: {name} ({team} vs {opponent})")
            print(f"  {'─' * 60}")
            print(f"  {'Zone':<6} {'Opp SF/g':>8} {'Team SA/g':>9} {'Exp Shots':>10} "
                  f"{'Goalie SV%':>10} {'Exp GA':>7} {'Exp SV':>7}")
            print(f"  {'HD':<6} {opp['hdsf_pg']:8.2f} {team_def['hdsa_pg']:9.2f} "
                  f"{hd_shots:10.2f} {goalie['hdsv_pct']:10.3f} {hd_ga:7.2f} {hd_shots-hd_ga:7.2f}")
            print(f"  {'MD':<6} {opp['mdsf_pg']:8.2f} {team_def['mdsa_pg']:9.2f} "
                  f"{md_shots:10.2f} {goalie['mdsv_pct']:10.3f} {md_ga:7.2f} {md_shots-md_ga:7.2f}")
            print(f"  {'LD':<6} {opp['ldsf_pg']:8.2f} {team_def['ldsa_pg']:9.2f} "
                  f"{ld_shots:10.2f} {goalie['ldsv_pct']:10.3f} {ld_ga:7.2f} {ld_shots-ld_ga:7.2f}")
            print(f"  {'─' * 60}")
            print(f"  {'Total':<6} {'':>8} {'':>9} {total_shots:10.2f} "
                  f"{'':>10} {total_ga:7.2f} {total_saves:7.2f}")
            print(f"\n  DK Scoring:")
            print(f"    Saves:    {total_saves:.1f} × {GOALIE_SCORING['save']}  = "
                  f"{total_saves * GOALIE_SCORING['save']:+.2f}")
            print(f"    GA:       {total_ga:.1f} × {GOALIE_SCORING['goal_against']}  = "
                  f"{total_ga * GOALIE_SCORING['goal_against']:+.2f}")
            print(f"    Win:      {win_prob:.0%} × {GOALIE_SCORING['win']}  = "
                  f"{win_prob * GOALIE_SCORING['win']:+.2f}")
            print(f"    OTL:      {ot_loss_rate:.0%} × {GOALIE_SCORING['overtime_loss']}  = "
                  f"{ot_loss_rate * GOALIE_SCORING['overtime_loss']:+.2f}")
            print(f"    Shutout:  {shutout_prob:.0%} × {GOALIE_SCORING['shutout_bonus']}  = "
                  f"{shutout_prob * GOALIE_SCORING['shutout_bonus']:+.2f}")
            print(f"    35+ SV:   {prob_35:.0%} × {GOALIE_BONUSES['thirty_five_plus_saves']}  = "
                  f"{prob_35 * GOALIE_BONUSES['thirty_five_plus_saves']:+.2f}")
            if is_home:
                print(f"    Home:     ×1.02")
            print(f"  {'─' * 60}")
            print(f"  PROJECTED FPTS: {fpts:.1f}")

        return {
            'projected_fpts': round(fpts, 2),
            'expected_shots': round(total_shots, 1),
            'expected_saves': round(total_saves, 1),
            'expected_ga': round(total_ga, 2),
            'hd_shots': round(hd_shots, 1),
            'md_shots': round(md_shots, 1),
            'ld_shots': round(ld_shots, 1),
            'hd_ga': round(hd_ga, 2),
            'md_ga': round(md_ga, 2),
            'ld_ga': round(ld_ga, 2),
            'win_prob': win_prob,
            'shutout_prob': round(shutout_prob, 3),
            'prob_35_saves': round(prob_35, 3),
        }

    def project_goalies(
        self,
        goalie_features: pd.DataFrame,
        schedule_df: pd.DataFrame,
        target_date: str,
        team_totals: Dict[str, float] = None,
        team_game_totals: Dict[str, float] = None,
        nst_thru_date: str = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Project all goalies for a slate. Drop-in replacement for project_goalies_baseline.

        Args:
            goalie_features: DataFrame from engineer_goalie_features (has name, team, etc.)
            schedule_df: Schedule DataFrame (date, home_team, away_team)
            target_date: Slate date YYYY-MM-DD
            team_totals: Dict of team -> Vegas implied team total
            team_game_totals: Dict of team -> Vegas game total
            nst_thru_date: NST data cutoff (None = full season, set for backtesting)
            verbose: Print per-goalie breakdown

        Returns:
            Same DataFrame with projected_fpts and breakdown columns added
        """
        self.load_nst_data(thru_date=nst_thru_date)

        if self._team_stats is None or self._goalie_stats is None:
            print("  Warning: NST data unavailable — falling back to feature-based projection")
            df = goalie_features.copy()
            df['projected_fpts'] = np.nan
            return df

        # Build matchup map from schedule
        games_today = schedule_df[schedule_df['date'] == target_date]
        matchups = {}  # team -> (opponent, is_home)
        for _, game in games_today.iterrows():
            home = game['home_team']
            away = game['away_team']
            matchups[home] = (away, True)
            matchups[away] = (home, False)

        df = goalie_features.copy()
        projections = []

        for idx, row in df.iterrows():
            name = row.get('name', '')
            team = row.get('team', '')

            if team not in matchups:
                projections.append({'projected_fpts': np.nan})
                continue

            opponent, is_home = matchups[team]

            # Get win probability from Vegas
            win_prob = self._get_win_prob(team, team_totals, team_game_totals)

            # Fallback to goalie's season win rate if no Vegas data
            season_win_rate = row.get('win_rate')
            if win_prob == 0.50 and pd.notna(season_win_rate):
                win_prob = season_win_rate

            result = self._project_single(
                name, team, opponent, win_prob, is_home, verbose=verbose
            )
            projections.append(result)

        proj_df = pd.DataFrame(projections, index=df.index)

        # Overwrite projected_fpts with new model
        df['projected_fpts'] = proj_df['projected_fpts']

        # Add breakdown columns
        for col in ['expected_shots', 'expected_saves', 'expected_ga',
                     'hd_shots', 'md_shots', 'ld_shots',
                     'hd_ga', 'md_ga', 'ld_ga',
                     'win_prob', 'shutout_prob', 'prob_35_saves']:
            if col in proj_df.columns:
                df[col] = proj_df[col]

        # Floor and ceiling
        df['floor'] = df['projected_fpts'] * 0.25
        df['ceiling'] = df['projected_fpts'] * 3.0

        df = df.sort_values('projected_fpts', ascending=False)

        n_projected = df['projected_fpts'].notna().sum()
        print(f"  Goalie danger-zone model: projected {n_projected} goalies")

        return df

    @staticmethod
    def _get_win_prob(team: str, team_totals: Dict, team_game_totals: Dict) -> float:
        """Derive win probability from Vegas implied totals."""
        if not team_totals or not team_game_totals:
            return 0.50
        my_total = team_totals.get(team)
        game_total = team_game_totals.get(team)
        if my_total and game_total and game_total > 0:
            share = my_total / game_total
            return min(0.80, max(0.20, share * 1.1 - 0.05))
        return 0.50


# ================================================================
#  CLI
# ================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Goalie danger-zone projection model')
    parser.add_argument('--goalie', type=str, help='Goalie name')
    parser.add_argument('--team', type=str, help='Goalie team abbreviation')
    parser.add_argument('--opp', type=str, help='Opponent abbreviation')
    parser.add_argument('--win', type=float, default=0.50, help='Win probability (0-1)')
    parser.add_argument('--home', action='store_true', help='Goalie is home team')
    parser.add_argument('--all', action='store_true', help='Show all team danger-zone stats')
    args = parser.parse_args()

    model = GoalieProjectionModel()

    if args.all:
        model.load_nst_data()
        if model._team_stats is not None:
            ts = model._team_stats.copy()
            ts['total_sf_pg'] = ts['hdsf_pg'] + ts['mdsf_pg'] + ts['ldsf_pg']
            ts['total_sa_pg'] = ts['hdsa_pg'] + ts['mdsa_pg'] + ts['ldsa_pg']
            print(f"\n  TEAM DANGER ZONE STATS (per game, all situations)")
            print(f"  {'=' * 75}")
            print(f"  {'Team':<5} {'HD SF':>6} {'MD SF':>6} {'LD SF':>6} {'Tot SF':>7} | "
                  f"{'HD SA':>6} {'MD SA':>6} {'LD SA':>6} {'Tot SA':>7}")
            print(f"  {'─' * 70}")
            for team in sorted(ts.index):
                r = ts.loc[team]
                print(f"  {team:<5} {r['hdsf_pg']:6.1f} {r['mdsf_pg']:6.1f} "
                      f"{r['ldsf_pg']:6.1f} {r['total_sf_pg']:7.1f} | "
                      f"{r['hdsa_pg']:6.1f} {r['mdsa_pg']:6.1f} "
                      f"{r['ldsa_pg']:6.1f} {r['total_sa_pg']:7.1f}")

    elif args.goalie and args.team and args.opp:
        model.load_nst_data()
        model._project_single(
            args.goalie, args.team.upper(), args.opp.upper(),
            args.win, args.home, verbose=True,
        )

    else:
        print("Usage:")
        print("  python goalie_model.py --goalie 'Connor Hellebuyck' --team WPG --opp TOR --win 0.55 --home")
        print("  python goalie_model.py --all")
