#!/usr/bin/env python3
"""
Single-Entry Lineup Selector for NHL DFS (DraftKings)
======================================================

Redesigned from 4 actual $121 SE contest results (2/2-2/5/26):

KEY FINDINGS FROM REAL CONTEST DATA:
  - Winners average 168 FPTS (need 5.3 players scoring >15 FPTS)
  - Goalie is +8.1 FPTS swing between winners and bottom half
  - Winners average 11.1% roster ownership (NOT max contrarian)
  - Being contrarian is a BYPRODUCT of finding the right players, not a goal
  - Cash line averages 116 FPTS across contests
  - Top 5 per-player average: 16.7 FPTS vs 8.6 for bottom half

CONTEST MODES:
  - satellite:  $14, 10 entries, winner-take-all ticket
  - se_gpp:     $121, ~80 entries, top-heavy payouts
  - custom:     User provides contest parameters

Usage:
    from single_entry import SingleEntrySelector, ContestProfile
    profile = ContestProfile.satellite()
    selector = SingleEntrySelector(player_pool, contest=profile)
    best = selector.select(candidate_lineups)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ContestProfile:
    """Defines contest structure to calibrate lineup selection."""
    name: str
    entry_fee: float
    total_entries: int
    total_prize: float
    first_place: float
    places_paid: int
    cash_line_pct: float
    top_heavy: bool
    target_score: float
    cash_score: float
    mode: str

    @classmethod
    def satellite(cls):
        return cls(name="Satellite ($14 WTA)", entry_fee=14, total_entries=10,
                   total_prize=0, first_place=0, places_paid=1, cash_line_pct=0.10,
                   top_heavy=True, target_score=140, cash_score=140, mode='satellite')

    @classmethod
    def se_gpp(cls, entries=80):
        return cls(name=f"$121 SE GPP ({entries} entries)", entry_fee=121,
                   total_entries=entries, total_prize=entries * 121 * 0.85,
                   first_place=2000, places_paid=20,
                   cash_line_pct=20 / entries, top_heavy=True,
                   target_score=150, cash_score=115, mode='se_gpp')

    @classmethod
    def custom(cls, entry_fee, total_entries, total_prize, first_place,
               places_paid, target_score=None, cash_score=None):
        cash_pct = places_paid / total_entries
        top_heavy = first_place > (total_prize * 0.15)
        if target_score is None:
            target_score = 155 if top_heavy else 130
        if cash_score is None:
            cash_score = 115 if total_entries > 50 else 120
        return cls(name=f"Custom (${entry_fee}, {total_entries} entries)",
                   entry_fee=entry_fee, total_entries=total_entries,
                   total_prize=total_prize, first_place=first_place,
                   places_paid=places_paid, cash_line_pct=cash_pct,
                   top_heavy=top_heavy, target_score=target_score,
                   cash_score=cash_score, mode='custom')


SE_WEIGHTS = {
    'ceiling_score':     0.30,
    'goalie_upside':     0.20,
    'projection_qual':   0.20,
    'stack_quality':     0.15,
    'differentiation':   0.15,
}

GOALIE_PARAMS = {
    'target_fpts': 18.0,
    'min_acceptable': 8.0,
    'ceiling_weight': 0.6,
    'projection_weight': 0.4,
}


class SingleEntrySelector:
    def __init__(self, player_pool, contest=None, stack_builder=None,
                 team_totals=None, weights=None):
        self.pool = player_pool.copy()
        self.contest = contest or ContestProfile.se_gpp()
        self.stack_builder = stack_builder
        self.team_totals = team_totals or {}
        self.weights = weights or SE_WEIGHTS.copy()

        skaters = self.pool[self.pool['position'] != 'G']
        self._proj_mean = skaters['projected_fpts'].mean() if len(skaters) > 0 else 8.0
        self._proj_std = skaters['projected_fpts'].std() if len(skaters) > 0 else 3.0
        self._ceil_mean = skaters['ceiling'].mean() if 'ceiling' in skaters.columns and len(skaters) > 0 else self._proj_mean * 2.5
        self._ceil_std = skaters['ceiling'].std() if 'ceiling' in skaters.columns and len(skaters) > 0 else self._proj_std * 2

        goalies = self.pool[self.pool['position'] == 'G']
        self._goalie_proj_mean = goalies['projected_fpts'].mean() if len(goalies) > 0 else 8.0
        self._goalie_proj_std = goalies['projected_fpts'].std() if len(goalies) > 0 else 3.0
        self._goalie_ceil_mean = goalies['ceiling'].mean() if 'ceiling' in goalies.columns and len(goalies) > 0 else self._goalie_proj_mean * 2.0

        self._linemate_sets = {}
        self._pp_sets = {}
        if stack_builder:
            self._build_linemate_lookup()

    def _build_linemate_lookup(self):
        if not self.stack_builder or not hasattr(self.stack_builder, 'lines_data'):
            return
        for team, data in self.stack_builder.lines_data.items():
            self._linemate_sets[team] = []
            self._pp_sets[team] = []
            for key in ['line1', 'line2', 'line3', 'line4']:
                line = data.get(key, {})
                players = [line.get(p) for p in ['lw', 'c', 'rw'] if line.get(p)]
                if len(players) >= 2:
                    self._linemate_sets[team].append(set(players))
            for key in ['pair1', 'pair2', 'pair3']:
                pair = data.get(key, {})
                players = [pair.get(p) for p in ['ld', 'rd'] if pair.get(p)]
                if len(players) >= 2:
                    self._linemate_sets[team].append(set(players))
            for key in ['pp1', 'pp2']:
                pp = data.get(key, {})
                players = [pp.get(p) for p in ['lw', 'c', 'rw', 'ld', 'rd'] if pp.get(p)]
                if len(players) >= 2:
                    self._pp_sets[team].append(set(players))

    @staticmethod
    def _fuzzy_in_set(name, name_set):
        name_lower = name.lower()
        for s in name_set:
            s_lower = s.lower()
            if name_lower == s_lower:
                return True
            n_last = name_lower.split()[-1] if name_lower.split() else name_lower
            s_last = s_lower.split()[-1] if s_lower.split() else s_lower
            if n_last == s_last and len(n_last) > 3:
                return True
        return False

    def _find_linemate_overlap(self, names, team):
        conns = 0
        for ls in self._linemate_sets.get(team, []):
            ov = sum(1 for n in names if self._fuzzy_in_set(n, ls))
            if ov >= 2: conns += ov * (ov - 1) // 2
        return conns

    def _find_pp_overlap(self, names, team):
        conns = 0
        for ps in self._pp_sets.get(team, []):
            ov = sum(1 for n in names if self._fuzzy_in_set(n, ps))
            if ov >= 2: conns += ov * (ov - 1) // 2
        return conns

    def score_ceiling(self, lineup):
        target = self.contest.target_score
        if 'ceiling' in lineup.columns:
            total_ceiling = lineup['ceiling'].sum()
        else:
            sk = lineup[lineup['position'] != 'G']
            g = lineup[lineup['position'] == 'G']
            total_ceiling = sk['projected_fpts'].sum() * 2.5 + g['projected_fpts'].sum() * 2.0

        if total_ceiling < target:
            return max((total_ceiling / target) * 0.3, 0.0)

        headroom = (total_ceiling - target) / target
        score = min(0.5 + headroom * 2, 1.0)

        if 'ceiling' in lineup.columns:
            high_ceil = (lineup['ceiling'] > 20).sum()
        else:
            high_ceil = (lineup['projected_fpts'] > 10).sum()
        score += min(high_ceil / 6, 1.0) * 0.3
        score = min(score, 1.0)

        total_proj = lineup['projected_fpts'].sum()
        proj_z = (total_proj - self._proj_mean * 9) / max(self._proj_std * 3, 1)
        score += min(max(proj_z * 0.05, -0.1), 0.15)

        return min(max(score, 0), 1)

    def score_goalie_upside(self, lineup):
        goalie = lineup[lineup['position'] == 'G']
        if goalie.empty:
            return 0.0
        g = goalie.iloc[0]
        proj = g.get('projected_fpts', 0)
        salary = g.get('salary', 7500)

        g_ceiling = g.get('ceiling', proj * 2.0 + 5)
        if pd.isna(g_ceiling):
            g_ceiling = proj * 2.0 + 5

        all_goalies = self.pool[self.pool['position'] == 'G']
        g_proj_z = 0
        if len(all_goalies) > 1 and self._goalie_proj_std > 0:
            g_proj_z = (proj - self._goalie_proj_mean) / self._goalie_proj_std

        ceil_comp = min(g_ceiling / 30, 1.0)
        proj_comp = min(max(g_proj_z / 3 + 0.5, 0), 1.0)
        score = 0.6 * ceil_comp + 0.4 * proj_comp

        goalie_team = g.get('team', '')
        game_info = g.get('game_info', '')
        opp_team = None
        if pd.notna(game_info) and isinstance(game_info, str):
            parts = str(game_info).split()
            if parts:
                teams_in_game = parts[0].replace('@', '/').split('/')
                for t in teams_in_game:
                    t = t.strip().upper()
                    if t != goalie_team.upper() and len(t) >= 2:
                        opp_team = t

        if opp_team and opp_team in self.team_totals:
            opp_implied = self.team_totals[opp_team]
            if opp_implied <= 2.8: score += 0.10
            elif opp_implied <= 3.2: score += 0.05
            elif opp_implied >= 3.5: score -= 0.05

        skater_teams = lineup[lineup['position'] != 'G']['team'].value_counts()
        if len(skater_teams) > 0 and opp_team:
            if opp_team == skater_teams.index[0]:
                score -= 0.15

        if salary >= 8000 and g_ceiling < 25:
            score -= 0.05

        return min(max(score, 0), 1)

    def score_projection_quality(self, lineup):
        per_player = lineup['projected_fpts'].mean()
        target = 14.0
        quality = min(per_player / target, 1.0)

        dead_weight = (lineup['projected_fpts'] < 5).sum()
        if dead_weight >= 3: quality -= 0.15
        elif dead_weight >= 2: quality -= 0.08

        elite_count = (lineup['projected_fpts'] > 15).sum()
        quality += elite_count * 0.03

        return min(max(quality, 0), 1)

    def score_stack_quality(self, lineup):
        skaters = lineup[lineup['position'] != 'G']
        if skaters.empty:
            return 0.5
        score = 0.5
        tc = skaters['team'].value_counts()
        pt = tc.index[0] if len(tc) > 0 else None
        pc = tc.iloc[0] if len(tc) > 0 else 0

        if pc >= 3:
            score += 0.15
            if pc == 4: score += 0.05
            elif pc >= 5: score -= 0.05
            if pt and self._linemate_sets:
                names = skaters[skaters['team'] == pt]['name'].tolist()
                score += self._find_linemate_overlap(names, pt) * 0.08
                score += self._find_pp_overlap(names, pt) * 0.05
        else:
            score -= 0.15

        if len(tc) >= 2 and tc.iloc[1] >= 2:
            score += 0.05

        one_offs = sum(1 for _, c in tc.items() if c == 1)
        if one_offs > 2:
            score -= (one_offs - 2) * 0.05

        return min(max(score, 0), 1)

    def score_differentiation(self, lineup):
        if 'predicted_ownership' not in lineup.columns:
            return 0.5
        owns = lineup['predicted_ownership'].fillna(5.0)
        avg_own = owns.mean()

        if 8 <= avg_own <= 15: score = 0.75
        elif 5 <= avg_own < 8 or 15 < avg_own <= 20: score = 0.55
        elif avg_own < 5: score = 0.35
        else: score = 0.30

        low_own_quality = lineup[
            (lineup['predicted_ownership'] < 8) &
            (lineup['projected_fpts'] > self._proj_mean)
        ]
        n_diff = len(low_own_quality)
        if 2 <= n_diff <= 4: score += 0.20
        elif n_diff == 1: score += 0.08
        elif n_diff >= 5: score += 0.10

        if (owns > 15).sum() >= 6: score -= 0.20

        goalie_own = lineup.loc[lineup['position'] == 'G', 'predicted_ownership']
        if not goalie_own.empty and goalie_own.iloc[0] < 10:
            score += 0.05

        return min(max(score, 0), 1)

    def score_lineup(self, lineup):
        components = {
            'ceiling_score': self.score_ceiling(lineup),
            'goalie_upside': self.score_goalie_upside(lineup),
            'projection_qual': self.score_projection_quality(lineup),
            'stack_quality': self.score_stack_quality(lineup),
            'differentiation': self.score_differentiation(lineup),
        }
        total = sum(self.weights.get(k, 0) * v for k, v in components.items())
        wsum = sum(self.weights.get(k, 0) for k in components)
        components['total'] = total / wsum if wsum > 0 else 0
        return components

    def select(self, candidate_lineups, verbose=True):
        if not candidate_lineups:
            raise ValueError("No candidate lineups provided")
        if len(candidate_lineups) == 1:
            scores = self.score_lineup(candidate_lineups[0])
            return candidate_lineups[0], scores

        scored = [(lu, self.score_lineup(lu)) for lu in candidate_lineups]
        scored.sort(key=lambda x: x[1]['total'], reverse=True)
        if verbose:
            self._print_scores(scored)
        return scored[0][0], scored[0][1]

    def _print_scores(self, scored, n_show=10):
        print(f"\n{'=' * 105}")
        print(f" SE LINEUP SELECTOR | {self.contest.name}")
        print(f" Target: {self.contest.target_score}+ FPTS to win | Cash: {self.contest.cash_score}+ | {self.contest.total_entries} entries")
        print(f"{'=' * 105}")

        comps = ['ceiling_score', 'goalie_upside', 'projection_qual', 'stack_quality', 'differentiation', 'total']
        header = f"{'#':>3} {'Proj':>6} {'Ceil':>6} {'Sal':>7} {'Goalie':<16} {'Stack':<10}"
        for c in comps:
            header += f" {c.split('_')[0][:5].title():>6}"
        print(f"\n{header}")
        print("-" * len(header))

        for i in range(min(len(scored), n_show)):
            lu, sc = scored[i]
            tp = lu['projected_fpts'].sum()
            tc = lu['ceiling'].sum() if 'ceiling' in lu.columns else 0
            ts = lu['salary'].sum()
            g = lu[lu['position'] == 'G']
            gn = g.iloc[0]['name'][:14] if not g.empty else "???"
            sk = lu[lu['position'] != 'G']
            stk = '+'.join(f"{t}{c}" for t, c in sk['team'].value_counts().items() if c >= 2)[:9]
            m = " <<<" if i == 0 else ""
            row = f"{i+1:>3} {tp:>6.1f} {tc:>6.0f} ${ts:>6,} {gn:<16} {stk:<10}"
            for c in comps:
                row += f" {sc.get(c, 0):>6.3f}"
            print(row + m)

        best_lu, best_sc = scored[0]
        print(f"\n  Scoring Breakdown:")
        for c in comps[:-1]:
            w = self.weights.get(c, 0)
            r = best_sc[c]
            bar = '\u2588' * int(r * 20) + '\u2591' * (20 - int(r * 20))
            print(f"    {c:<22} {bar} {r:.3f} x {w:.2f} = {w*r:.3f}")
        print(f"    {'TOTAL':<22} {'':>20} {best_sc['total']:.3f}")

        tp = best_lu['projected_fpts'].sum()
        tc = best_lu['ceiling'].sum() if 'ceiling' in best_lu.columns else tp * 2
        clr = 'CLEARS' if tc >= self.contest.target_score else 'BELOW'
        print(f"\n  Proj: {tp:.1f} | Ceiling: {tc:.0f} | Target: {self.contest.target_score} | {clr}")


def print_se_lineup(lineup, scores):
    print(f"\n{'=' * 80}")
    print(" SINGLE-ENTRY LINEUP (FINAL)")
    print(f"{'=' * 80}")
    ts = lineup['salary'].sum()
    tp = lineup['projected_fpts'].sum()
    tc = lineup['ceiling'].sum() if 'ceiling' in lineup.columns else 0
    print(f"  Salary: ${ts:,} / $50,000 (${50000 - ts:,} remaining)")
    print(f"  Projected: {tp:.1f} FPTS")
    if tc > 0: print(f"  Ceiling: {tc:.1f} FPTS")
    print(f"  SE Score: {scores['total']:.3f}")

    sk = lineup[lineup['position'] != 'G']
    tc_map = sk['team'].value_counts()
    stacks = [f"{t} x{c}" for t, c in tc_map.items() if c >= 2]
    ones = [t for t, c in tc_map.items() if c == 1]
    if stacks: print(f"  Stacks: {', '.join(stacks)}")
    if ones: print(f"  One-offs: {', '.join(ones)}")
    print()

    display = lineup.copy()
    if 'roster_slot' in display.columns:
        so = {'G':0,'C1':1,'C2':2,'W1':3,'W2':4,'W3':5,'D1':6,'D2':7,'UTIL':8}
        display['_o'] = display['roster_slot'].map(lambda x: so.get(x, 9))
        display = display.sort_values('_o')
        sc = 'roster_slot'
    else:
        po = {'C':0,'W':2,'D':3,'G':4}
        display['_o'] = display['position'].map(lambda x: po.get(x, 5))
        display = display.sort_values('_o')
        sc = 'position'

    oc = 'predicted_ownership' if 'predicted_ownership' in display.columns else None
    h = f"  {'Slot':<6} {'Name':<28} {'Team':<5} {'Salary':<9} {'Proj':>6} {'Ceil':>6} {'Value':>6}"
    if oc: h += f" {'Own%':>6}"
    print(h)
    print("  " + "-" * (len(h) - 2))
    for _, r in display.iterrows():
        cv = r.get('ceiling', 0)
        vl = r.get('value', 0)
        ln = f"  {r[sc]:<6} {r['name']:<28} {r['team']:<5} ${r['salary']:<8,} {r['projected_fpts']:>6.1f} {cv:>6.1f} {vl:>6.2f}"
        if oc:
            ln += f" {r.get(oc, 0):>5.1f}%"
        print(ln)
    print()


def prompt_contest_profile():
    """Interactive prompt for contest details."""
    print("\n  CONTEST PROFILE SETUP")
    print("  " + "-" * 40)
    print("    1. Satellite ($14, 10 entries, winner-take-all ticket)")
    print("    2. $121 SE GPP (~80 entries, top-heavy payouts)")
    print("    3. Custom contest")
    choice = input("\n  Select [1/2/3]: ").strip()
    if choice == '1':
        return ContestProfile.satellite()
    elif choice == '2':
        entries = input("  Total entries (default 80): ").strip()
        return ContestProfile.se_gpp(int(entries) if entries else 80)
    else:
        try:
            fee = float(input("  Entry fee ($): ").strip())
            entries = int(input("  Total entries: ").strip())
            prize = float(input("  Total prize pool ($): ").strip())
            first = float(input("  1st place prize ($): ").strip())
            paid = int(input("  Places paid: ").strip())
            return ContestProfile.custom(fee, entries, prize, first, paid)
        except (ValueError, EOFError):
            print("  Using default SE GPP")
            return ContestProfile.se_gpp()
