#!/usr/bin/env python3
"""
Opponent Profiling Model
=========================

Two components that adjust projections based on opponent context:

1. POSITION-SPECIFIC SHOT ALLOWANCE
   Some teams allow way more D-man shots than others (e.g. bad gap control),
   while others funnel everything through the slot (more C/W goals against).
   This adjusts skater projections based on how the opponent defends each position.

2. MISSING STAR SKATER DETECTION
   When an opponent is missing their top scorers, their offensive threat drops.
   This benefits the opposing goalie. Detected by comparing a team's expected
   lineup (from game log frequency) to who actually played in recent games.

Data sources:
   - game_logs_skaters: who scored what, by position, vs each opponent
   - team_special_teams: PP/PK rates
   - Daily lineup information (from projections CSV or confirmed lineups)

Usage:
    from opponent_profiling import OpponentProfiler
    profiler = OpponentProfiler()
    profiler.fit()

    # Get position-specific adjustments
    adj = profiler.get_position_adjustment('C', opponent='VAN')

    # Check if opponent is missing stars
    missing = profiler.detect_missing_stars('EDM', slate_date='2026-02-05')

    # Adjust a full player pool
    pool = profiler.adjust_projections(pool, opponent_map, slate_date)

CLI:
    python opponent_profiling.py --report              # Show all team profiles
    python opponent_profiling.py --team EDM            # Show one team's profile
    python opponent_profiling.py --matchup COL vs EDM  # Show matchup analysis
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent / "data" / "nhl_dfs_history.db"


class OpponentProfiler:
    """Profiles opponents for position-specific adjustments and missing star detection."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.fitted = False

        # Position-specific allowance rates (opponent → position → multiplier)
        self.position_allowance = {}  # team → {C: mult, W: mult, D: mult}
        self.league_avg_by_pos = {}   # position → avg FPTS allowed per game

        # Star player profiles for missing detection
        self.team_rosters = {}        # team → [{player_id, name, avg_fpts, games, ...}]
        self.team_expected_stars = {}  # team → [top N player_ids by avg_fpts]

    def fit(self) -> 'OpponentProfiler':
        """Build opponent profiles from game log data."""
        if not Path(self.db_path).exists():
            print("  ⚠ No game log database found")
            return self

        conn = sqlite3.connect(self.db_path)

        sk = pd.read_sql_query("""
            SELECT player_id, player_name, team, position, game_id, game_date,
                   opponent, goals, assists, shots, pp_goals, pp_points,
                   toi_seconds, dk_fpts
            FROM game_logs_skaters
            ORDER BY team, game_date
        """, conn)

        if sk.empty:
            conn.close()
            return self

        # Normalize positions
        sk['pos_group'] = sk['position'].map(
            {'C': 'C', 'L': 'W', 'R': 'W', 'LW': 'W', 'RW': 'W', 'D': 'D'}
        ).fillna('W')

        self._build_position_allowance(sk)
        self._build_star_profiles(sk)

        conn.close()
        self.fitted = True
        return self

    def _build_position_allowance(self, sk: pd.DataFrame):
        """
        For each team, calculate how many FPTS they ALLOW to each position
        relative to league average.

        If team X allows 8.5 avg FPTS to centers but league avg is 7.0,
        then centers get a 1.21x multiplier when facing team X.
        """
        # Calculate what each team ALLOWS to opposing players by position
        # "opponent" column tells us who they were playing against
        # We want: for each team as the DEFENDING team, what did opposing positions score?

        # Group by opponent (defending team) and position of the scorer
        allowed = sk.groupby(['opponent', 'pos_group']).agg(
            total_fpts=('dk_fpts', 'sum'),
            total_shots=('shots', 'sum'),
            total_goals=('goals', 'sum'),
            total_assists=('assists', 'sum'),
            total_pp_goals=('pp_goals', 'sum'),
            n_player_games=('player_id', 'count'),
            n_games=('game_id', 'nunique'),
        ).reset_index()

        # Per-game averages (what each position scores against this team per game)
        allowed['fpts_pg'] = allowed['total_fpts'] / allowed['n_games']
        allowed['shots_pg'] = allowed['total_shots'] / allowed['n_games']
        allowed['goals_pg'] = allowed['total_goals'] / allowed['n_games']

        # League averages by position
        league = allowed.groupby('pos_group').agg(
            avg_fpts_pg=('fpts_pg', 'mean'),
            avg_shots_pg=('shots_pg', 'mean'),
            avg_goals_pg=('goals_pg', 'mean'),
        )
        self.league_avg_by_pos = league['avg_fpts_pg'].to_dict()

        # Build per-team multipliers
        for team in allowed['opponent'].unique():
            team_data = allowed[allowed['opponent'] == team]
            self.position_allowance[team] = {}

            for _, row in team_data.iterrows():
                pos = row['pos_group']
                league_avg = league.loc[pos, 'avg_fpts_pg'] if pos in league.index else 1.0

                if league_avg > 0:
                    multiplier = row['fpts_pg'] / league_avg
                else:
                    multiplier = 1.0

                # Regress toward 1.0 to avoid extreme values from small samples
                n_games = row['n_games']
                regression_weight = min(n_games / 30, 1.0)  # full weight at 30+ games
                multiplier = regression_weight * multiplier + (1 - regression_weight) * 1.0

                self.position_allowance[team][pos] = {
                    'multiplier': round(multiplier, 3),
                    'fpts_pg': round(row['fpts_pg'], 1),
                    'shots_pg': round(row['shots_pg'], 1),
                    'goals_pg': round(row['goals_pg'], 1),
                    'n_games': n_games,
                }

    def _build_star_profiles(self, sk: pd.DataFrame):
        """Build roster profiles for missing star detection."""
        for team, group in sk.groupby('team'):
            player_profiles = []
            for pid, pg in group.groupby('player_id'):
                pg = pg.sort_values('game_date')
                profile = {
                    'player_id': pid,
                    'name': pg.iloc[-1]['player_name'],
                    'position': pg.iloc[-1]['position'],
                    'pos_group': pg.iloc[-1]['pos_group'],
                    'games': len(pg),
                    'avg_fpts': pg['dk_fpts'].mean(),
                    'total_fpts': pg['dk_fpts'].sum(),
                    'avg_shots': pg['shots'].mean(),
                    'avg_goals': pg['goals'].mean(),
                    'avg_pp_goals': pg['pp_goals'].mean(),
                    'last_game': pg.iloc[-1]['game_date'],
                    'last5_fpts': pg['dk_fpts'].tail(5).mean(),
                    'game_ids': set(pg['game_id'].unique()),
                }
                player_profiles.append(profile)

            # Sort by avg_fpts descending
            player_profiles.sort(key=lambda x: x['avg_fpts'], reverse=True)
            self.team_rosters[team] = player_profiles

            # Top stars: players averaging 8+ FPTS with 10+ games
            stars = [p for p in player_profiles if p['avg_fpts'] >= 8.0 and p['games'] >= 10]
            self.team_expected_stars[team] = stars[:6]  # top 6 stars

    def get_position_adjustment(self, position: str, opponent: str) -> Dict:
        """
        Get the adjustment multiplier for a position vs a specific opponent.

        Returns:
            Dict with multiplier, fpts_pg, shots_pg, description
        """
        pos_group = {'C': 'C', 'L': 'W', 'R': 'W', 'LW': 'W', 'RW': 'W', 'D': 'D', 'W': 'W'}.get(position, 'W')

        opp_data = self.position_allowance.get(opponent, {}).get(pos_group)
        if not opp_data:
            return {'multiplier': 1.0, 'description': 'no data'}

        mult = opp_data['multiplier']
        league_avg = self.league_avg_by_pos.get(pos_group, 0)
        diff = opp_data['fpts_pg'] - league_avg

        if mult > 1.08:
            desc = f"allows +{diff:.1f} FPTS/G to {pos_group} (good matchup)"
        elif mult < 0.92:
            desc = f"allows {diff:.1f} FPTS/G to {pos_group} (bad matchup)"
        else:
            desc = f"neutral for {pos_group}"

        return {
            'multiplier': mult,
            'fpts_pg': opp_data['fpts_pg'],
            'shots_pg': opp_data['shots_pg'],
            'league_avg': league_avg,
            'diff': round(diff, 1),
            'n_games': opp_data['n_games'],
            'description': desc,
        }

    def detect_missing_stars(self, team: str, slate_date: str = None,
                              confirmed_out: List[str] = None) -> Dict:
        """
        Detect if a team is missing star players.

        Methods:
            1. confirmed_out: list of player names known to be out
            2. Game log gap: if a star hasn't played in recent games

        Returns:
            Dict with missing_stars list, offensive_impact estimate
        """
        stars = self.team_expected_stars.get(team, [])
        if not stars:
            return {'missing': [], 'impact': 0.0, 'note': 'no star data'}

        missing = []

        # Method 1: Confirmed out list
        if confirmed_out:
            for star in stars:
                for out_name in confirmed_out:
                    if out_name.lower() in star['name'].lower() or star['name'].lower() in out_name.lower():
                        missing.append(star)
                        break

        # Method 2: Game log gap detection
        if slate_date and not missing:
            try:
                slate_dt = pd.to_datetime(slate_date)
                for star in stars:
                    last_dt = pd.to_datetime(star['last_game'])
                    days_since = (slate_dt - last_dt).days

                    # If star hasn't played in 10+ days and the team has played recently,
                    # they're likely injured/out
                    if days_since > 10:
                        # Check if team has played more recently
                        roster = self.team_rosters.get(team, [])
                        team_last_game = max(
                            pd.to_datetime(p['last_game']) for p in roster
                        ) if roster else last_dt

                        if (slate_dt - team_last_game).days < days_since - 3:
                            missing.append(star)
            except Exception:
                pass

        # Calculate offensive impact
        total_star_fpts = sum(s['avg_fpts'] for s in stars)
        missing_fpts = sum(s['avg_fpts'] for s in missing)

        # Impact: fraction of star production missing → estimated goal reduction
        if total_star_fpts > 0:
            pct_missing = missing_fpts / total_star_fpts
        else:
            pct_missing = 0

        # Rough conversion: missing 30% of star power ≈ -0.5 goals per game
        goal_impact = pct_missing * 1.5  # max ~1.5 goals if ALL stars out
        fpts_impact = goal_impact * 3.5  # ~3.5 goalie FPTS per goal (saves + GA penalty)

        return {
            'missing': [(s['name'], s['avg_fpts'], s['pos_group']) for s in missing],
            'missing_fpts': round(missing_fpts, 1),
            'pct_missing': round(pct_missing, 3),
            'est_goal_reduction': round(goal_impact, 2),
            'goalie_boost': round(fpts_impact, 1),
            'total_stars': len(stars),
        }

    def adjust_projections(self, player_pool: pd.DataFrame,
                           opponent_map: Dict[str, str] = None,
                           slate_date: str = None,
                           confirmed_out: Dict[str, List[str]] = None,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Adjust player pool projections using opponent profiling.

        Args:
            player_pool: DataFrame with projections
            opponent_map: {team: opponent}
            slate_date: YYYY-MM-DD
            confirmed_out: {team: [player_names_out]}
        """
        if not self.fitted:
            self.fit()

        if not opponent_map:
            return player_pool

        df = player_pool.copy()

        # ── Position-specific adjustments for skaters ──
        skater_mask = df['position'] != 'G'
        adj_count = 0

        for idx, row in df[skater_mask].iterrows():
            team = row['team']
            opp = opponent_map.get(team, '')
            if not opp:
                continue

            pos = row['position']
            result = self.get_position_adjustment(pos, opp)
            mult = result['multiplier']

            if abs(mult - 1.0) > 0.02:  # only adjust if meaningful
                current = df.loc[idx, 'projected_fpts']
                # Apply as a soft adjustment: blend 70% current + 30% adjusted
                adjusted = current * (0.70 + 0.30 * mult)
                df.loc[idx, 'projected_fpts'] = round(adjusted, 2)
                df.loc[idx, 'opp_pos_adj'] = round(mult, 3)
                adj_count += 1

        # ── Missing star impact on goalies ──
        goalie_mask = df['position'] == 'G'
        goalie_adj_count = 0

        for idx, row in df[goalie_mask].iterrows():
            team = row['team']
            opp = opponent_map.get(team, '')
            if not opp:
                continue

            out_list = confirmed_out.get(opp, []) if confirmed_out else None
            missing = self.detect_missing_stars(opp, slate_date, out_list)

            if missing['goalie_boost'] > 0.5:
                current = df.loc[idx, 'projected_fpts']
                df.loc[idx, 'projected_fpts'] = round(current + missing['goalie_boost'], 2)
                df.loc[idx, 'opp_missing_stars'] = len(missing['missing'])
                df.loc[idx, 'opp_missing_boost'] = round(missing['goalie_boost'], 1)
                goalie_adj_count += 1

        # Update derived columns
        if 'salary' in df.columns:
            df['value'] = (df['projected_fpts'] / (df['salary'] / 1000)).round(3)
        if 'dk_avg_fpts' in df.columns:
            df['edge'] = (df['projected_fpts'] - df['dk_avg_fpts']).round(3)

        if verbose:
            print(f"\n  ── Opponent Profiling ──────────────────────")
            print(f"  Position adjustments: {adj_count} skaters modified")
            print(f"  Missing star boosts:  {goalie_adj_count} goalies boosted")

            # Show biggest position adjustments
            if 'opp_pos_adj' in df.columns:
                extremes = df[df['opp_pos_adj'].notna()].copy()
                if len(extremes) > 0:
                    top_boost = extremes.nlargest(5, 'opp_pos_adj')
                    top_nerf = extremes.nsmallest(5, 'opp_pos_adj')

                    print(f"\n  Best matchups:")
                    for _, r in top_boost.iterrows():
                        opp = opponent_map.get(r['team'], '?')
                        print(f"    {r['name']:<22} ({r['position']}) vs {opp}: "
                              f"{r['opp_pos_adj']:.2f}x")
                    print(f"  Worst matchups:")
                    for _, r in top_nerf.iterrows():
                        opp = opponent_map.get(r['team'], '?')
                        print(f"    {r['name']:<22} ({r['position']}) vs {opp}: "
                              f"{r['opp_pos_adj']:.2f}x")

        return df

    def print_report(self, team: str = None):
        """Print opponent profile report."""
        if not self.fitted:
            self.fit()

        teams = [team] if team else sorted(self.position_allowance.keys())

        print(f"\n  ── Opponent Position Allowance Report ──")
        print(f"  League averages: "
              + ', '.join(f"{p}={v:.1f}" for p, v in self.league_avg_by_pos.items()))
        print()
        print(f"  {'Team':<5} {'C mult':>7} {'C fpts':>7} {'W mult':>7} {'W fpts':>7} "
              f"{'D mult':>7} {'D fpts':>7} {'Games':>5}")
        print(f"  {'-' * 55}")

        for t in teams:
            opp = self.position_allowance.get(t, {})
            c = opp.get('C', {})
            w = opp.get('W', {})
            d = opp.get('D', {})
            games = c.get('n_games', 0)
            print(f"  {t:<5} {c.get('multiplier',1.0):>7.3f} {c.get('fpts_pg',0):>7.1f} "
                  f"{w.get('multiplier',1.0):>7.3f} {w.get('fpts_pg',0):>7.1f} "
                  f"{d.get('multiplier',1.0):>7.3f} {d.get('fpts_pg',0):>7.1f} "
                  f"{games:>5}")

        # Star profiles
        if team:
            stars = self.team_expected_stars.get(team, [])
            if stars:
                print(f"\n  {team} Star Players (top {len(stars)}):")
                for s in stars:
                    print(f"    {s['name']:<22} ({s['pos_group']}) "
                          f"avg={s['avg_fpts']:.1f} GP={s['games']} "
                          f"last={s['last_game']}")


# ================================================================
#  CLI
# ================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Opponent Profiling')
    parser.add_argument('--report', action='store_true', help='Print all team profiles')
    parser.add_argument('--team', type=str, help='Show profile for one team')
    parser.add_argument('--matchup', nargs=3, metavar=('TEAM', 'vs', 'OPP'),
                       help='Show matchup analysis (e.g. --matchup COL vs EDM)')
    args = parser.parse_args()

    profiler = OpponentProfiler()
    profiler.fit()

    if args.report:
        profiler.print_report()
    elif args.team:
        profiler.print_report(args.team)
    elif args.matchup:
        team, _, opp = args.matchup
        print(f"\n  MATCHUP: {team} vs {opp}")
        for pos in ['C', 'W', 'D']:
            adj = profiler.get_position_adjustment(pos, opp)
            print(f"    {pos}: {adj['multiplier']:.3f}x ({adj['description']})")

        missing = profiler.detect_missing_stars(opp)
        if missing['missing']:
            print(f"\n  {opp} MISSING STARS:")
            for name, fpts, pos in missing['missing']:
                print(f"    {name} ({pos}) avg={fpts:.1f}")
            print(f"  Est goalie boost: +{missing['goalie_boost']:.1f} FPTS")
        else:
            print(f"\n  No missing stars detected for {opp}")
    else:
        profiler.print_report()
