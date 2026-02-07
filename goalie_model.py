"""
Goalie Projection Model — Danger Zone Matchup Approach.

Projects goalie DK fantasy points using:
1. Geometric mean of opponent shot generation × team shot allowance per danger zone
2. Individual goalie save % by danger zone (HD, MD, LD)
3. Vegas win probability for the win bonus

Data source: Natural Stat Trick (all situations, per-game rates)

Usage:
    from goalie_model import GoalieProjectionModel

    model = GoalieProjectionModel()
    projections = model.project_goalies(
        goalie_df=data['goalies'],
        schedule_df=data['schedule'],
        target_date='2026-01-29',
        team_totals=team_totals,        # Vegas implied totals
        team_game_totals=team_game_totals,  # Vegas game totals
    )
"""

import pandas as pd
import numpy as np
import requests
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    GOALIE_SCORING, GOALIE_BONUSES,
    calculate_goalie_fantasy_points,
)


# ================================================================
#  NST Data Fetching
# ================================================================

class NSTGoalieData:
    """
    Fetch goalie and team danger-zone stats from Natural Stat Trick.

    All data is per-game, all situations (not 5v5).
    """

    BASE_URL = "https://www.naturalstattrick.com"

    def __init__(self, season: str = "20252026", rate_limit: float = 2.0):
        self.season = season
        self.rate_limit = rate_limit
        self._last_request = 0

    def _wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _fetch_table(self, url: str) -> Optional[pd.DataFrame]:
        """Fetch and return first HTML table from URL."""
        self._wait()
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            return tables[0] if tables else None
        except Exception as e:
            print(f"  NST fetch error: {e}")
            return None

    def fetch_team_danger_stats(self) -> Optional[pd.DataFrame]:
        """
        Fetch team-level shot stats by danger zone (all situations, raw counts).

        Returns DataFrame with per-game rates:
            team, gp, hdsf_pg, hdsa_pg, mdsf_pg, mdsa_pg, ldsf_pg, ldsa_pg
        """
        print("  Fetching team danger-zone stats (NST, all situations)...")
        url = (
            f"{self.BASE_URL}/teamtable.php?"
            f"fromseason={self.season}&thruseason={self.season}"
            f"&stype=2&sit=all&score=all&rate=n&team=all&loc=B"
            f"&gpf=410&fd=&td="
        )
        df = self._fetch_table(url)
        if df is None:
            return None

        # Normalize team names
        from scrapers import NaturalStatTrickScraper
        nst = NaturalStatTrickScraper()

        result = pd.DataFrame()
        result['team'] = df['Team'].apply(nst._normalize_team_code)
        result['gp'] = df['GP']

        # Per-game shot rates by danger zone
        gp = df['GP'].replace(0, 1)  # Avoid division by zero
        result['hdsf_pg'] = df['HDSF'] / gp   # HD Shots For per game
        result['hdsa_pg'] = df['HDSA'] / gp   # HD Shots Against per game
        result['mdsf_pg'] = df['MDSF'] / gp   # MD Shots For per game
        result['mdsa_pg'] = df['MDSA'] / gp   # MD Shots Against per game
        result['ldsf_pg'] = df['LDSF'] / gp   # LD Shots For per game
        result['ldsa_pg'] = df['LDSA'] / gp   # LD Shots Against per game

        # Total shots for/against per game (for reference)
        for col in ['SF', 'SA']:
            if col in df.columns:
                result[col.lower() + '_pg'] = df[col] / gp

        print(f"    {len(result)} teams loaded")
        return result.set_index('team')

    def fetch_goalie_stats(self) -> Optional[pd.DataFrame]:
        """
        Fetch individual goalie stats by danger zone (all situations).

        Returns DataFrame with:
            player, team, gp, sv_pct, hdsv_pct, mdsv_pct, ldsv_pct,
            hd_sa, md_sa, ld_sa, saves, goals_against
        """
        print("  Fetching goalie danger-zone stats (NST, all situations)...")
        url = (
            f"{self.BASE_URL}/playerteams.php?"
            f"fromseason={self.season}&thruseason={self.season}"
            f"&stype=2&sit=all&score=all&stdoi=g&rate=n"
            f"&team=ALL&pos=S&loc=B&toi=0"
            f"&gpfilt=none&fd=&td=&tgp=82&lines=single&dession=false"
        )
        df = self._fetch_table(url)
        if df is None:
            return None

        from scrapers import NaturalStatTrickScraper
        nst = NaturalStatTrickScraper()

        result = pd.DataFrame()
        result['name'] = df['Player']
        result['team'] = df['Team'].apply(nst._normalize_team_code)
        result['gp'] = df['GP']

        # Overall
        result['gaa'] = df['GAA']
        result['saves'] = df['Saves']
        result['goals_against'] = df['Goals Against']
        result['shots_against'] = df['Shots Against']

        # Danger zone save percentages (the key inputs)
        result['hdsv_pct'] = pd.to_numeric(df['HDSV%'], errors='coerce')
        result['mdsv_pct'] = pd.to_numeric(df['MDSV%'], errors='coerce')
        result['ldsv_pct'] = pd.to_numeric(df['LDSV%'], errors='coerce')

        # Overall save %
        result['sv_pct'] = pd.to_numeric(df['SV%'], errors='coerce')

        # Danger zone shot counts (for per-game reference)
        result['hd_sa'] = df['HD Shots Against']
        result['md_sa'] = df['MD Shots Against']
        result['ld_sa'] = df['LD Shots Against']

        # Per-game rates
        gp = result['gp'].replace(0, 1)
        result['sa_pg'] = result['shots_against'] / gp
        result['saves_pg'] = result['saves'] / gp
        result['ga_pg'] = result['goals_against'] / gp

        print(f"    {len(result)} goalies loaded")
        return result


# ================================================================
#  Projection Model
# ================================================================

class GoalieProjectionModel:
    """
    Project goalie DK FPTS using danger-zone matchup analysis.

    For each goalie:
    1. Get opponent's per-game shot generation by zone (HDSF, MDSF, LDSF)
    2. Get goalie's team per-game shots allowed by zone (HDSA, MDSA, LDSA)
    3. Expected shots = sqrt(Opp_SF * Team_SA) per zone (geometric mean)
    4. Expected GA per zone = Expected_Shots * (1 - Goalie_SV% for that zone)
    5. Expected saves per zone = Expected_Shots - Expected_GA
    6. DK FPTS = saves * 0.7 + GA * -3.5 + win% * 6.0 + bonuses
    """

    def __init__(self, season: str = "20252026"):
        self.season = season
        self.nst = NSTGoalieData(season=season)
        self._team_stats = None
        self._goalie_stats = None

    def _load_data(self, force_refresh: bool = False):
        """Fetch NST data if not already loaded."""
        if self._team_stats is None or force_refresh:
            self._team_stats = self.nst.fetch_team_danger_stats()
        if self._goalie_stats is None or force_refresh:
            self._goalie_stats = self.nst.fetch_goalie_stats()

    def project_single_goalie(
        self,
        goalie_name: str,
        goalie_team: str,
        opponent: str,
        win_prob: float = 0.50,
        is_home: bool = False,
        verbose: bool = False,
    ) -> Dict:
        """
        Project DK FPTS for a single goalie matchup.

        Args:
            goalie_name: Goalie name (must match NST)
            goalie_team: Goalie's team abbreviation
            opponent: Opponent team abbreviation
            win_prob: Vegas implied win probability (0-1)
            is_home: Whether goalie's team is home
            verbose: Print breakdown

        Returns:
            Dict with projected_fpts and breakdown
        """
        self._load_data()

        if self._team_stats is None or self._goalie_stats is None:
            return {'projected_fpts': 0, 'error': 'NST data unavailable'}

        # Get opponent's shots-for per game (what they generate)
        if opponent not in self._team_stats.index:
            return {'projected_fpts': 0, 'error': f'Opponent {opponent} not found in NST'}
        opp = self._team_stats.loc[opponent]

        # Get goalie's team shots-against per game (what they allow)
        if goalie_team not in self._team_stats.index:
            return {'projected_fpts': 0, 'error': f'Team {goalie_team} not found in NST'}
        team = self._team_stats.loc[goalie_team]

        # Find goalie's individual save percentages
        goalie_rows = self._goalie_stats[
            self._goalie_stats['name'].str.lower().str.contains(
                goalie_name.lower().split()[-1]  # Match on last name
            ) & (self._goalie_stats['team'] == goalie_team)
        ]

        if goalie_rows.empty:
            # Broader search: just last name
            goalie_rows = self._goalie_stats[
                self._goalie_stats['name'].str.lower().str.contains(
                    goalie_name.lower().split()[-1]
                )
            ]

        if goalie_rows.empty:
            return {'projected_fpts': 0, 'error': f'Goalie {goalie_name} not found in NST'}

        goalie = goalie_rows.iloc[0]

        # ── Step 1: Expected shots by danger zone (geometric mean) ──
        hd_shots = np.sqrt(opp['hdsf_pg'] * team['hdsa_pg'])
        md_shots = np.sqrt(opp['mdsf_pg'] * team['mdsa_pg'])
        ld_shots = np.sqrt(opp['ldsf_pg'] * team['ldsa_pg'])
        total_shots = hd_shots + md_shots + ld_shots

        # ── Step 2: Expected GA by zone using goalie's save % ──
        hd_ga = hd_shots * (1 - goalie['hdsv_pct'])
        md_ga = md_shots * (1 - goalie['mdsv_pct'])
        ld_ga = ld_shots * (1 - goalie['ldsv_pct'])
        total_ga = hd_ga + md_ga + ld_ga

        # ── Step 3: Expected saves ──
        total_saves = total_shots - total_ga

        # ── Step 4: DK Fantasy Points ──
        # Base scoring
        fpts = (
            total_saves * GOALIE_SCORING['save'] +
            total_ga * GOALIE_SCORING['goal_against'] +
            win_prob * GOALIE_SCORING['win']
        )

        # OT loss expected value (~25% of non-wins go to OT)
        ot_loss_rate = (1 - win_prob) * 0.25
        fpts += ot_loss_rate * GOALIE_SCORING['overtime_loss']

        # Shutout probability (decreases with expected GA)
        shutout_prob = max(0, 0.15 - total_ga * 0.04)
        fpts += shutout_prob * GOALIE_SCORING['shutout_bonus']

        # 35+ saves bonus probability
        # Rough estimate: if expected saves is near 35, there's a chance
        if total_saves > 25:
            # Use normal approximation: ~15% of variance
            std_saves = total_saves * 0.15
            from scipy.stats import norm
            try:
                prob_35 = 1 - norm.cdf(35, loc=total_saves, scale=max(std_saves, 1))
            except ImportError:
                # Fallback if scipy not available
                prob_35 = max(0, (total_saves - 30) / 20)
            fpts += prob_35 * GOALIE_BONUSES['thirty_five_plus_saves']
        else:
            prob_35 = 0.0

        # Home ice advantage
        if is_home:
            fpts *= 1.02

        if verbose:
            print(f"\n  {'─' * 60}")
            print(f"  GOALIE PROJECTION: {goalie_name} ({goalie_team} vs {opponent})")
            print(f"  {'─' * 60}")
            print(f"  {'Zone':<6} {'Opp SF/g':>8} {'Team SA/g':>9} {'Exp Shots':>10} {'Goalie SV%':>10} {'Exp GA':>7} {'Exp SV':>7}")
            print(f"  {'HD':<6} {opp['hdsf_pg']:8.2f} {team['hdsa_pg']:9.2f} {hd_shots:10.2f} {goalie['hdsv_pct']:10.3f} {hd_ga:7.2f} {hd_shots - hd_ga:7.2f}")
            print(f"  {'MD':<6} {opp['mdsf_pg']:8.2f} {team['mdsa_pg']:9.2f} {md_shots:10.2f} {goalie['mdsv_pct']:10.3f} {md_ga:7.2f} {md_shots - md_ga:7.2f}")
            print(f"  {'LD':<6} {opp['ldsf_pg']:8.2f} {team['ldsa_pg']:9.2f} {ld_shots:10.2f} {goalie['ldsv_pct']:10.3f} {ld_ga:7.2f} {ld_shots - ld_ga:7.2f}")
            print(f"  {'─' * 60}")
            print(f"  {'Total':<6} {'':>8} {'':>9} {total_shots:10.2f} {'':>10} {total_ga:7.2f} {total_saves:7.2f}")
            print(f"\n  DK Scoring:")
            print(f"    Saves:    {total_saves:.1f} × {GOALIE_SCORING['save']}  = {total_saves * GOALIE_SCORING['save']:+.2f}")
            print(f"    GA:       {total_ga:.1f} × {GOALIE_SCORING['goal_against']}  = {total_ga * GOALIE_SCORING['goal_against']:+.2f}")
            print(f"    Win:      {win_prob:.0%} × {GOALIE_SCORING['win']}  = {win_prob * GOALIE_SCORING['win']:+.2f}")
            print(f"    OTL:      {ot_loss_rate:.0%} × {GOALIE_SCORING['overtime_loss']}  = {ot_loss_rate * GOALIE_SCORING['overtime_loss']:+.2f}")
            print(f"    Shutout:  {shutout_prob:.0%} × {GOALIE_SCORING['shutout_bonus']}  = {shutout_prob * GOALIE_SCORING['shutout_bonus']:+.2f}")
            print(f"    35+ SV:   {prob_35:.0%} × {GOALIE_BONUSES['thirty_five_plus_saves']}  = {prob_35 * GOALIE_BONUSES['thirty_five_plus_saves']:+.2f}")
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
            'prob_35_saves': round(prob_35, 3) if prob_35 else 0,
        }

    def project_goalies(
        self,
        goalie_df: pd.DataFrame,
        schedule_df: pd.DataFrame,
        target_date: str,
        team_totals: Dict[str, float] = None,
        team_game_totals: Dict[str, float] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Project all goalies for a slate.

        This is the main entry point — call this from projections.py.

        Args:
            goalie_df: Goalie DataFrame from data_pipeline (needs 'name', 'team')
            schedule_df: Schedule DataFrame (needs 'date', 'home_team', 'away_team')
            target_date: Date string YYYY-MM-DD
            team_totals: Dict of team -> implied team total from Vegas
            team_game_totals: Dict of team -> game total from Vegas
            verbose: Print projection breakdowns

        Returns:
            DataFrame with projected_fpts added (same shape as goalie_df)
        """
        self._load_data()

        if self._team_stats is None or self._goalie_stats is None:
            print("  Warning: NST data unavailable, cannot run goalie model")
            return goalie_df

        # Build matchup map from schedule
        games_today = schedule_df[schedule_df['date'] == target_date]
        matchups = {}  # team -> (opponent, is_home)
        for _, game in games_today.iterrows():
            home = game['home_team']
            away = game['away_team']
            matchups[home] = (away, True)
            matchups[away] = (home, False)

        # Calculate win probability from Vegas
        def get_win_prob(team: str) -> float:
            if not team_totals or not team_game_totals:
                return 0.50
            my_total = team_totals.get(team)
            game_total = team_game_totals.get(team)
            if my_total and game_total and game_total > 0:
                # Implied win prob from team total / game total ratio
                # Higher share of game total = higher win probability
                share = my_total / game_total
                # Convert share to win prob (rough: 0.5 share = 50%, scaled)
                win_prob = min(0.80, max(0.20, share * 1.1 - 0.05))
                return win_prob
            return 0.50

        # Project each goalie
        df = goalie_df.copy()
        projections = []

        for idx, row in df.iterrows():
            name = row.get('name', '')
            team = row.get('team', '')

            if team not in matchups:
                projections.append({
                    'projected_fpts': 0,
                    'expected_shots': 0,
                    'expected_saves': 0,
                    'expected_ga': 0,
                })
                continue

            opponent, is_home = matchups[team]
            win_prob = get_win_prob(team)

            result = self.project_single_goalie(
                goalie_name=name,
                goalie_team=team,
                opponent=opponent,
                win_prob=win_prob,
                is_home=is_home,
                verbose=verbose,
            )

            projections.append(result)

        # Merge projections back into DataFrame
        proj_df = pd.DataFrame(projections, index=df.index)

        # Set projected_fpts from our model
        df['projected_fpts'] = proj_df['projected_fpts']

        # Add breakdown columns for analysis
        for col in ['expected_shots', 'expected_saves', 'expected_ga',
                     'hd_shots', 'md_shots', 'ld_shots',
                     'hd_ga', 'md_ga', 'ld_ga',
                     'win_prob', 'shutout_prob', 'prob_35_saves']:
            if col in proj_df.columns:
                df[col] = proj_df[col]

        # Floor and ceiling
        df['floor'] = df['projected_fpts'] * 0.25
        df['ceiling'] = df['projected_fpts'] * 3.0

        if verbose:
            print(f"\n  Goalie Model: Projected {len(df[df['projected_fpts'] > 0])} goalies")

        return df


# ================================================================
#  Quick Test
# ================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test goalie projection model')
    parser.add_argument('--goalie', type=str, help='Goalie name to project')
    parser.add_argument('--team', type=str, help='Goalie team')
    parser.add_argument('--opp', type=str, help='Opponent team')
    parser.add_argument('--win', type=float, default=0.50, help='Win probability')
    parser.add_argument('--home', action='store_true', help='Is home team')
    parser.add_argument('--all', action='store_true', help='Show all teams danger zone stats')
    args = parser.parse_args()

    model = GoalieProjectionModel()

    if args.all:
        model._load_data()
        if model._team_stats is not None:
            print("\n  TEAM DANGER ZONE STATS (per game, all situations)")
            print("  " + "=" * 75)
            ts = model._team_stats.copy()
            ts['total_sf_pg'] = ts['hdsf_pg'] + ts['mdsf_pg'] + ts['ldsf_pg']
            ts['total_sa_pg'] = ts['hdsa_pg'] + ts['mdsa_pg'] + ts['ldsa_pg']
            print(f"  {'Team':<5} {'HD SF':>6} {'MD SF':>6} {'LD SF':>6} {'Tot SF':>7} | {'HD SA':>6} {'MD SA':>6} {'LD SA':>6} {'Tot SA':>7}")
            print(f"  {'-' * 70}")
            for team in sorted(ts.index):
                r = ts.loc[team]
                print(f"  {team:<5} {r['hdsf_pg']:6.1f} {r['mdsf_pg']:6.1f} {r['ldsf_pg']:6.1f} {r['total_sf_pg']:7.1f} | "
                      f"{r['hdsa_pg']:6.1f} {r['mdsa_pg']:6.1f} {r['ldsa_pg']:6.1f} {r['total_sa_pg']:7.1f}")

    elif args.goalie and args.team and args.opp:
        result = model.project_single_goalie(
            goalie_name=args.goalie,
            goalie_team=args.team.upper(),
            opponent=args.opp.upper(),
            win_prob=args.win,
            is_home=args.home,
            verbose=True,
        )
    else:
        print("Usage:")
        print("  python goalie_model.py --goalie 'Connor Hellebuyck' --team WPG --opp TOR --win 0.60 --home")
        print("  python goalie_model.py --all")
