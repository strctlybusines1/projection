#!/usr/bin/env python3
"""
Advanced Game Environment Models
==================================

Three models that adjust projections based on game environment:

1. VEGAS IMPLIED SCORING MODEL
   Uses implied team totals to scale skater and goalie projections.
   A team implied for 3.8 goals should have skaters projected higher than
   a team implied for 2.3 goals. Currently the pipeline barely uses this.

2. PACE-OF-PLAY ADJUSTMENT
   Some matchups produce more total shots/chances. When two high-pace teams
   meet, goalies face more shots (more saves but more GA risk), and skaters
   get more opportunities. Built from shots-per-game rates of both teams.

3. RECENCY-WEIGHTED OPPONENT ADJUSTMENT
   The opponent profiling module uses full-season averages. But teams change
   through the season (injuries, trades, coaching changes). This weights
   the last 10 games more heavily to capture current defensive form.

Usage:
    from game_environment import GameEnvironmentModel
    env = GameEnvironmentModel()
    env.fit()
    player_pool = env.adjust_projections(player_pool, vegas_data, date_str)

CLI:
    python game_environment.py --report
    python game_environment.py --matchup COL vs EDM --total 6.5
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"


class GameEnvironmentModel:
    """Adjusts projections based on Vegas lines, pace, and recent opponent form."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.fitted = False

        # Team shot rates (for pace calculation)
        self.team_shots_for_pg = {}     # team → shots for per game
        self.team_shots_against_pg = {}  # team → shots against per game
        self.league_avg_shots = 29.0
        self.league_avg_goals = 3.1

        # Recency-weighted opponent defense
        self.team_recent_defense = {}    # team → {last10_ga_pg, last10_sa_pg, trend}
        self.team_season_defense = {}    # team → {season_ga_pg, season_sa_pg}

        # Vegas calibration (how much to trust implied totals)
        self.vegas_accuracy = {}         # learned from historical data

    def fit(self) -> 'GameEnvironmentModel':
        """Build models from game log database."""
        if not Path(self.db_path).exists():
            return self

        conn = sqlite3.connect(self.db_path)

        # ── Load team shot/goal rates from special teams table ──
        try:
            st = pd.read_sql_query("""
                SELECT team_abbrev, goals_for_pg, goals_against_pg,
                       shots_for_pg, shots_against_pg
                FROM team_special_teams
            """, conn)

            for _, r in st.iterrows():
                team = r['team_abbrev']
                if not team:
                    continue
                self.team_shots_for_pg[team] = r['shots_for_pg']
                self.team_shots_against_pg[team] = r['shots_against_pg']
                self.team_season_defense[team] = {
                    'ga_pg': r['goals_against_pg'],
                    'sa_pg': r['shots_against_pg'],
                    'gf_pg': r['goals_for_pg'],
                }

            if st['shots_for_pg'].notna().any():
                self.league_avg_shots = st['shots_for_pg'].mean()
            if st['goals_for_pg'].notna().any():
                self.league_avg_goals = st['goals_for_pg'].mean()
        except Exception:
            pass

        # ── Build recency-weighted opponent defense from game logs ──
        self._build_recent_defense(conn)

        conn.close()
        self.fitted = True
        return self

    def _build_recent_defense(self, conn: sqlite3.Connection):
        """
        Calculate each team's defensive performance over last 10 games
        vs their season average. Detects teams getting better or worse.
        """
        try:
            # Get goals against per game from goalie data
            g = pd.read_sql_query("""
                SELECT team, game_date, opponent, goals_against, shots_against
                FROM game_logs_goalies
                WHERE games_started = 1 OR toi_seconds > 1800
                ORDER BY team, game_date
            """, conn)

            if g.empty:
                return

            for team, group in g.groupby('team'):
                group = group.sort_values('game_date')

                season_ga = group['goals_against'].mean()
                season_sa = group['shots_against'].mean()

                # Last 10 games
                last10 = group.tail(10)
                recent_ga = last10['goals_against'].mean()
                recent_sa = last10['shots_against'].mean()

                # Last 5 games (more recent signal)
                last5 = group.tail(5)
                recent5_ga = last5['goals_against'].mean()

                # Trend: negative = defense improving, positive = getting worse
                trend = recent_ga - season_ga

                self.team_recent_defense[team] = {
                    'last10_ga_pg': round(recent_ga, 2),
                    'last10_sa_pg': round(recent_sa, 1),
                    'last5_ga_pg': round(recent5_ga, 2),
                    'season_ga_pg': round(season_ga, 2),
                    'season_sa_pg': round(season_sa, 1),
                    'trend': round(trend, 2),
                    'n_games': len(group),
                }
        except Exception:
            pass

    # ================================================================
    #  Model 1: Vegas Implied Scoring
    # ================================================================

    def get_vegas_multiplier(self, team: str, team_implied_total: float,
                              opp_implied_total: float = None) -> Dict:
        """
        Calculate projection multiplier from Vegas implied team total.

        The key insight: if Vegas implies a team scores 3.8 goals but the
        league average is 3.1, that team's skaters should be boosted ~22%.
        But not linearly — the relationship between team goals and individual
        skater FPTS has diminishing returns.

        Args:
            team: Team abbreviation
            team_implied_total: Vegas implied goals for this team
            opp_implied_total: Vegas implied goals for opponent

        Returns:
            Dict with skater_mult, goalie_mult, pace_factor, etc.
        """
        if not team_implied_total or team_implied_total <= 0:
            return {'skater_mult': 1.0, 'goalie_mult': 1.0, 'pace_factor': 1.0}

        # ── Skater multiplier ──
        # Relationship: +1 goal above avg ≈ +15% skater FPTS for the team
        # But with diminishing returns (sqrt scaling)
        goals_above_avg = team_implied_total - self.league_avg_goals
        if goals_above_avg >= 0:
            skater_mult = 1.0 + 0.12 * (goals_above_avg ** 0.8)
        else:
            skater_mult = 1.0 + 0.15 * goals_above_avg  # linear penalty for low totals

        skater_mult = max(0.75, min(1.35, skater_mult))

        # ── Goalie multiplier ──
        if opp_implied_total:
            # Opponent implied total directly predicts goals against
            # Lower opp total = fewer GA = better for goalie
            # But also fewer saves
            opp_above_avg = opp_implied_total - self.league_avg_goals
            # Net effect: low opponent total is slightly good for goalie (fewer GA dominates)
            goalie_mult = 1.0 - 0.08 * opp_above_avg
            goalie_mult = max(0.80, min(1.20, goalie_mult))
        else:
            goalie_mult = 1.0

        # ── Pace factor ──
        game_total = team_implied_total + (opp_implied_total or self.league_avg_goals)
        avg_game_total = self.league_avg_goals * 2
        pace_factor = game_total / avg_game_total if avg_game_total > 0 else 1.0

        return {
            'skater_mult': round(skater_mult, 3),
            'goalie_mult': round(goalie_mult, 3),
            'pace_factor': round(pace_factor, 3),
            'team_implied': team_implied_total,
            'goals_above_avg': round(goals_above_avg, 2),
        }

    # ================================================================
    #  Model 2: Pace-of-Play
    # ================================================================

    def get_pace_adjustment(self, team: str, opponent: str) -> Dict:
        """
        Calculate pace-of-play multiplier from both teams' shot rates.

        High-pace games (both teams shoot a lot) produce:
        - More save opportunities for goalies
        - More shot-based FPTS for skaters
        - But also more goals against for goalies

        Args:
            team: Team abbreviation
            opponent: Opponent abbreviation

        Returns:
            Dict with pace_mult, expected_shots, description
        """
        team_sf = self.team_shots_for_pg.get(team, self.league_avg_shots)
        team_sa = self.team_shots_against_pg.get(team, self.league_avg_shots)
        opp_sf = self.team_shots_for_pg.get(opponent, self.league_avg_shots)
        opp_sa = self.team_shots_against_pg.get(opponent, self.league_avg_shots)

        # Expected shots FOR this team = avg of (team's SF/G, opponent's SA/G)
        expected_shots_for = (team_sf + opp_sa) / 2
        # Expected shots AGAINST this team = avg of (team's SA/G, opponent's SF/G)
        expected_shots_against = (team_sa + opp_sf) / 2
        # Total expected shots in game
        total_shots = expected_shots_for + expected_shots_against

        # Pace relative to league average
        avg_total_shots = self.league_avg_shots * 2
        pace_mult = total_shots / avg_total_shots if avg_total_shots > 0 else 1.0

        # Skater shot boost: if team expected to generate more shots than avg
        skater_shot_mult = expected_shots_for / self.league_avg_shots if self.league_avg_shots > 0 else 1.0

        # Goalie workload: more shots against = more save opportunities
        goalie_workload = expected_shots_against / self.league_avg_shots if self.league_avg_shots > 0 else 1.0

        if pace_mult > 1.08:
            desc = f"high-pace ({total_shots:.0f} total shots expected)"
        elif pace_mult < 0.92:
            desc = f"low-pace ({total_shots:.0f} total shots expected)"
        else:
            desc = f"avg pace ({total_shots:.0f} shots)"

        return {
            'pace_mult': round(pace_mult, 3),
            'skater_shot_mult': round(skater_shot_mult, 3),
            'goalie_workload': round(goalie_workload, 3),
            'expected_shots_for': round(expected_shots_for, 1),
            'expected_shots_against': round(expected_shots_against, 1),
            'description': desc,
        }

    # ================================================================
    #  Model 3: Recency-Weighted Opponent Defense
    # ================================================================

    def get_recency_adjustment(self, opponent: str) -> Dict:
        """
        Adjust for opponent's recent defensive form vs season average.

        If a team's last 10 games show 3.5 GA/G but their season avg is 2.8,
        their defense is slipping — opposing skaters should be boosted.

        Returns:
            Dict with multiplier, trend direction, description
        """
        recent = self.team_recent_defense.get(opponent)
        season = self.team_season_defense.get(opponent)

        if not recent or not season:
            return {'multiplier': 1.0, 'trend': 0, 'description': 'no data'}

        # Compare last 10 to season
        recent_ga = recent['last10_ga_pg']
        season_ga = season['ga_pg']
        trend = recent_ga - season_ga  # positive = defense getting worse

        # Weight: 60% season + 40% recent (gives recent more than even weight)
        blended_ga = 0.60 * season_ga + 0.40 * recent_ga

        # Multiplier: how much to adjust opposing skater projections
        if season_ga > 0:
            multiplier = blended_ga / season_ga
        else:
            multiplier = 1.0

        # Also factor in last 5 (very recent signal)
        recent5 = recent.get('last5_ga_pg', recent_ga)
        if abs(recent5 - recent_ga) > 0.5:
            # Strong recent acceleration — give extra weight
            if recent5 > recent_ga:
                multiplier *= 1.02  # defense collapsing
            else:
                multiplier *= 0.98  # defense tightening

        multiplier = max(0.88, min(1.15, multiplier))

        if trend > 0.4:
            desc = f"defense slipping ({opponent} L10: {recent_ga:.1f} GA/G vs season {season_ga:.1f})"
        elif trend < -0.4:
            desc = f"defense tightening ({opponent} L10: {recent_ga:.1f} GA/G vs season {season_ga:.1f})"
        else:
            desc = f"stable defense ({opponent})"

        return {
            'multiplier': round(multiplier, 3),
            'trend': round(trend, 2),
            'blended_ga': round(blended_ga, 2),
            'description': desc,
        }

    # ================================================================
    #  Combined Adjustment
    # ================================================================

    def adjust_projections(self, player_pool: pd.DataFrame,
                           vegas: pd.DataFrame = None,
                           date_str: str = None,
                           opponent_map: Dict[str, str] = None,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Apply all three environment adjustments to player pool.

        Args:
            player_pool: DataFrame with projections
            vegas: Vegas data with Team, TeamGoal, OppGoal columns
            date_str: YYYY-MM-DD
            opponent_map: {team: opponent} (built from Vegas if not provided)
        """
        if not self.fitted:
            self.fit()

        df = player_pool.copy()

        # ── Build opponent map and Vegas totals from Vegas data ──
        team_totals = {}
        opp_totals = {}
        if opponent_map is None:
            opponent_map = {}

        if vegas is not None and date_str:
            day_v = vegas[vegas.get('date', '') == date_str] if 'date' in vegas.columns else vegas
            for _, vrow in day_v.iterrows():
                team = vrow.get('Team', '')
                opp = vrow.get('Opp', '').replace('vs ', '').replace('@ ', '').strip()
                if team:
                    team_totals[team] = vrow.get('TeamGoal', self.league_avg_goals)
                    opp_totals[team] = vrow.get('OppGoal', self.league_avg_goals)
                    if opp:
                        opponent_map[team] = opp

        # ── Apply adjustments ──
        vegas_adj_n = 0
        pace_adj_n = 0
        recency_adj_n = 0

        for idx, row in df.iterrows():
            team = row['team']
            opp = opponent_map.get(team, '')
            pos = row['position']

            total_mult = 1.0

            # Model 1: Vegas implied scoring
            if team in team_totals:
                vegas_result = self.get_vegas_multiplier(
                    team, team_totals[team], opp_totals.get(team)
                )
                if pos == 'G':
                    total_mult *= vegas_result['goalie_mult']
                else:
                    total_mult *= vegas_result['skater_mult']
                vegas_adj_n += 1

            # Model 2: Pace of play (skaters only — goalie pace handled via workload)
            if opp and pos != 'G':
                pace_result = self.get_pace_adjustment(team, opp)
                # Apply pace as a soft nudge: 20% weight
                pace_nudge = 0.80 + 0.20 * pace_result['skater_shot_mult']
                total_mult *= pace_nudge
                pace_adj_n += 1

            # Model 3: Recency-weighted opponent defense (skaters only)
            if opp and pos != 'G':
                recency_result = self.get_recency_adjustment(opp)
                if abs(recency_result['multiplier'] - 1.0) > 0.02:
                    # Apply at 25% weight
                    recency_nudge = 0.75 + 0.25 * recency_result['multiplier']
                    total_mult *= recency_nudge
                    recency_adj_n += 1

            # Apply combined multiplier
            if abs(total_mult - 1.0) > 0.01:
                df.loc[idx, 'projected_fpts'] = round(
                    df.loc[idx, 'projected_fpts'] * total_mult, 2
                )
                df.loc[idx, 'env_multiplier'] = round(total_mult, 3)

        # Update derived columns
        if 'salary' in df.columns:
            df['value'] = (df['projected_fpts'] / (df['salary'] / 1000)).round(3)
        if 'dk_avg_fpts' in df.columns:
            df['edge'] = (df['projected_fpts'] - df['dk_avg_fpts']).round(3)

        if verbose:
            print(f"\n  ── Game Environment Model ──────────────────")
            print(f"  Vegas adjustments: {vegas_adj_n} players")
            print(f"  Pace adjustments:  {pace_adj_n} skaters")
            print(f"  Recency defense:   {recency_adj_n} skaters")

            if team_totals:
                sorted_teams = sorted(team_totals.items(), key=lambda x: x[1], reverse=True)
                print(f"\n  Vegas implied totals:")
                for t, tot in sorted_teams[:5]:
                    opp = opponent_map.get(t, '?')
                    print(f"    {t} ({tot:.1f}) vs {opp} ({opp_totals.get(t,0):.1f})"
                          f"  game total: {tot + opp_totals.get(t,0):.1f}")

            # Show recency trends
            trending = [(t, d) for t, d in self.team_recent_defense.items()
                       if abs(d.get('trend', 0)) > 0.4]
            if trending:
                trending.sort(key=lambda x: x[1]['trend'], reverse=True)
                print(f"\n  Defensive trends (L10 vs season):")
                for t, d in trending[:5]:
                    direction = "↑ worse" if d['trend'] > 0 else "↓ better"
                    print(f"    {t}: {d['trend']:+.1f} GA/G {direction} "
                          f"(L10: {d['last10_ga_pg']:.1f} vs season: {d['season_ga_pg']:.1f})")

        return df

    def print_report(self):
        """Print full environment model report."""
        if not self.fitted:
            self.fit()

        print(f"\n  ── Game Environment Report ──")
        print(f"  League avg goals/game: {self.league_avg_goals:.2f}")
        print(f"  League avg shots/game: {self.league_avg_shots:.1f}")

        # Pace rankings
        if self.team_shots_for_pg:
            print(f"\n  PACE RANKINGS (shots for + against per game):")
            pace = []
            for team in self.team_shots_for_pg:
                sf = self.team_shots_for_pg.get(team, 0)
                sa = self.team_shots_against_pg.get(team, 0)
                pace.append((team, sf, sa, sf + sa))
            pace.sort(key=lambda x: x[3], reverse=True)

            print(f"  {'Team':<5} {'SF/G':>5} {'SA/G':>5} {'Total':>6} {'Pace':>5}")
            for t, sf, sa, total in pace[:10]:
                p = total / (self.league_avg_shots * 2)
                print(f"  {t:<5} {sf:>5.1f} {sa:>5.1f} {total:>6.1f} {p:>5.2f}x")

        # Recency defense
        if self.team_recent_defense:
            print(f"\n  DEFENSIVE TRENDS (L10 vs season GA/G):")
            trending = sorted(self.team_recent_defense.items(),
                            key=lambda x: x[1].get('trend', 0), reverse=True)
            print(f"  {'Team':<5} {'L10 GA':>7} {'Szn GA':>7} {'Trend':>6} {'Signal'}")
            for t, d in trending:
                signal = "WORSE" if d['trend'] > 0.4 else "BETTER" if d['trend'] < -0.4 else "STABLE"
                print(f"  {t:<5} {d['last10_ga_pg']:>7.2f} {d['season_ga_pg']:>7.2f} "
                      f"{d['trend']:>+5.2f}  {signal}")


# ================================================================
#  CLI
# ================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Game Environment Model')
    parser.add_argument('--report', action='store_true', help='Full report')
    parser.add_argument('--matchup', nargs=3, metavar=('TEAM', 'vs', 'OPP'),
                       help='Matchup analysis (e.g. --matchup COL vs EDM)')
    parser.add_argument('--total', type=float, default=6.0,
                       help='Game total for matchup analysis')
    args = parser.parse_args()

    env = GameEnvironmentModel()
    env.fit()

    if args.report:
        env.print_report()
    elif args.matchup:
        team, _, opp = args.matchup
        team_total = args.total / 2 + 0.3  # rough split favoring named team
        opp_total = args.total - team_total

        print(f"\n  MATCHUP: {team} vs {opp} (total: {args.total})")

        v = env.get_vegas_multiplier(team, team_total, opp_total)
        print(f"\n  Vegas: {team} implied {team_total:.1f} goals")
        print(f"    Skater mult: {v['skater_mult']:.3f}x")
        print(f"    Goalie mult: {v['goalie_mult']:.3f}x")

        p = env.get_pace_adjustment(team, opp)
        print(f"\n  Pace: {p['description']}")
        print(f"    Expected shots for {team}: {p['expected_shots_for']:.1f}")
        print(f"    Expected shots against: {p['expected_shots_against']:.1f}")

        r = env.get_recency_adjustment(opp)
        print(f"\n  Recency ({opp} defense): {r['description']}")
        print(f"    Multiplier: {r['multiplier']:.3f}x")
    else:
        env.print_report()
