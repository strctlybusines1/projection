#!/usr/bin/env python3
"""
Leverage Score — Ownership-Aware Tournament Equity
====================================================

The sim_selector optimizes for ceiling (M+3σ). This module adds the
SECOND dimension: lineup uniqueness relative to the field.

P(win SE GPP) ≈ P(scoring 150+) × P(being the only one who scored 150+)

Key findings from 10 $121/$5 SE NHL contests:
  - Contrarian winners (7/10): avg score 168, avg lineup ownership 8.0%
  - Chalk winners (3/10):      avg score 177, avg lineup ownership 18.6%
  - Contrarian path needs ~10 fewer points to win

Key finding from 76 ownership slates (own.csv):
  - Chalk stack FPTS vs Opposing goalie FPTS: r = -0.581 (p < 0.0001)
  - When chalk busts (34% of slates): opp goalie avg 16.7 FPTS at 7.5% owned
  - When chalk hits (66% of slates):  opp goalie avg 1.6 FPTS
  - THE DOUBLE DOWN: stack against chalk + play opposing goalie

Usage:
    from leverage_score import LeverageScorer, detect_chalk_stack

    scorer = LeverageScorer(ownership_map, contest_type='se_gpp')
    scored = scorer.score_lineups(lineups, sim_results)
    # Returns lineups re-ranked by leverage-adjusted score

    chalk = detect_chalk_stack(ownership_map, player_pool)
    # Returns the chalk team, opposing goalie, and leverage recommendation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# Chalk Stack Detection
# ═══════════════════════════════════════════════════════════════

@dataclass
class ChalkAnalysis:
    """Result of chalk stack detection."""
    chalk_team: str                    # Most popular team to stack
    chalk_team_own: float             # Sum of skater ownership for chalk team
    chalk_goalie: Optional[str]       # Chalk team's goalie
    chalk_goalie_own: float           # Chalk goalie ownership %
    opp_team: Optional[str]           # Opponent of chalk team
    opp_goalie: Optional[str]         # Opposing goalie (THE LEVERAGE PLAY)
    opp_goalie_own: float             # Opposing goalie ownership %
    leverage_rating: str              # 'HIGH', 'MEDIUM', 'LOW'
    recommendation: str               # Human-readable recommendation


@dataclass
class SlateContext:
    """
    Slate-level features that determine whether contrarian or chalk
    is the +EV strategy. Computed once per slate.

    From 96 slates of ownership data:
      - Small slates (2-4 teams): chalk concentrates to 16.5% avg, 24% chance of 25%+ players
      - Large slates (12+ teams): only 6.6% avg, 4% chance of 25%+
      - $9K+ scores 40+ on 9.4% of slates — when they do, gap is unreachable
      - $8K+ busts below 10 FPTS 40% of the time
      - Slate top scorer is <10% owned 62% of the time
    """
    n_teams: int                       # Number of teams on slate
    is_small_slate: bool               # ≤6 teams → chalk concentrates
    must_have_players: List[str]       # Players with unreachable ceilings (large proj gap + PP1)
    n_must_haves: int                  # Count of must-have players
    chalk_intensity: str               # 'EXTREME', 'MODERATE', 'DISPERSED'
    contrarian_opportunity: str        # 'HIGH', 'MEDIUM', 'LOW'
    mode_recommendation: str           # What the system recommends


def detect_chalk_stack(
    player_pool: pd.DataFrame,
    ownership_map: Dict[str, float] = None,
) -> ChalkAnalysis:
    """
    Detect the chalk stack and identify the opposing goalie leverage play.

    Args:
        player_pool: DataFrame with [name, team, position, projected_fpts, ...]
        ownership_map: {player_name: ownership_%}. If None, uses projected_own
                       or ownership column from pool.

    Returns:
        ChalkAnalysis with chalk team, opposing goalie, and recommendation.
    """
    pool = player_pool.copy()

    # Standardize column names
    name_col = 'name' if 'name' in pool.columns else 'Player'
    team_col = 'team' if 'team' in pool.columns else 'Team'
    pos_col = 'position' if 'position' in pool.columns else 'Pos'

    # Get ownership
    if ownership_map:
        pool['_own'] = pool[name_col].map(ownership_map).fillna(1.0)
    else:
        for col in ['projected_own', 'own_proj', 'ownership', 'Ownership']:
            if col in pool.columns:
                pool['_own'] = pd.to_numeric(pool[col], errors='coerce').fillna(1.0)
                break
        else:
            pool['_own'] = 1.0

    # Find chalk team: highest total skater ownership
    skaters = pool[pool[pos_col] != 'G']
    team_own = skaters.groupby(team_col)['_own'].sum().sort_values(ascending=False)

    if len(team_own) == 0:
        return ChalkAnalysis('?', 0, None, 0, None, None, 0, 'LOW', 'No ownership data')

    chalk_team = team_own.index[0]
    chalk_own = team_own.iloc[0]

    # Second most popular for context
    second_team = team_own.index[1] if len(team_own) > 1 else None
    second_own = team_own.iloc[1] if len(team_own) > 1 else 0

    # Find chalk team's goalie
    chalk_goalies = pool[(pool[team_col] == chalk_team) & (pool[pos_col] == 'G')]
    chalk_goalie = chalk_goalies.nlargest(1, '_own').iloc[0][name_col] if len(chalk_goalies) > 0 else None
    chalk_goalie_own = chalk_goalies['_own'].max() if len(chalk_goalies) > 0 else 0

    # Find opposing team
    # Try to get from Game column or opponent column
    opp_team = None
    for col in ['opponent', 'Opp', 'opp']:
        if col in pool.columns:
            chalk_players = pool[pool[team_col] == chalk_team]
            opps = chalk_players[col].dropna().unique()
            if len(opps) > 0:
                opp_team = opps[0]
            break

    # Fallback: try Game column (format like "PIT_NYR" or "PIT@NYR")
    if not opp_team:
        for col in ['Game', 'game', 'Game Info']:
            if col in pool.columns:
                chalk_players = pool[pool[team_col] == chalk_team]
                games = chalk_players[col].dropna().unique()
                for g in games:
                    game_str = str(g).replace('@', '_').split('_')
                    for t in game_str:
                        t = t.strip().split()[0]  # Handle "PIT_NYR 07:00PM"
                        if t != chalk_team and len(t) >= 2 and len(t) <= 4:
                            opp_team = t
                            break
                    if opp_team:
                        break
                break

    # Find opposing goalie (THE LEVERAGE PLAY)
    opp_goalie = None
    opp_goalie_own = 0
    if opp_team:
        opp_goalies = pool[(pool[team_col] == opp_team) & (pool[pos_col] == 'G')]
        if len(opp_goalies) > 0:
            opp_g = opp_goalies.nlargest(1, '_own').iloc[0]
            opp_goalie = opp_g[name_col]
            opp_goalie_own = opp_g['_own']

    # Leverage rating
    own_gap = chalk_own - second_own if second_own else chalk_own
    if chalk_own > 80 and own_gap > 20:
        leverage = 'HIGH'
    elif chalk_own > 50:
        leverage = 'MEDIUM'
    else:
        leverage = 'LOW'

    # Build recommendation
    rec_parts = []
    rec_parts.append(f"Chalk team: {chalk_team} (total skater own: {chalk_own:.0f}%)")
    if opp_goalie:
        rec_parts.append(f"LEVERAGE PLAY: {opp_goalie} ({opp_team} G, {opp_goalie_own:.1f}% owned)")
        rec_parts.append("When chalk busts (34% of slates): opp goalie avg 16.7 FPTS")
        rec_parts.append(f"Stack {opp_team} skaters + {opp_goalie} = maximum field leverage")
    if leverage == 'HIGH':
        rec_parts.append("⚡ HIGH LEVERAGE: Chalk is heavily concentrated — contrarian path strongly favored")

    return ChalkAnalysis(
        chalk_team=chalk_team,
        chalk_team_own=chalk_own,
        chalk_goalie=chalk_goalie,
        chalk_goalie_own=chalk_goalie_own,
        opp_team=opp_team,
        opp_goalie=opp_goalie,
        opp_goalie_own=opp_goalie_own,
        leverage_rating=leverage,
        recommendation='\n'.join(rec_parts),
    )


def analyze_slate_context(
    player_pool: pd.DataFrame,
    ownership_map: Dict[str, float] = None,
) -> SlateContext:
    """
    Analyze slate structure to determine contrarian vs chalk strategy.

    Detects "must-have" chalk: players whose ceilings are so far above
    their salary tier that NOT having them when they hit is fatal.

    From 96 slates:
      - $9K+ scores 40+ on 9.4% of slates
      - When they do, best $8-9K player averages only 30.1 (gap of +13.7)
      - 0% of $5-6K players match a 40+ $9K+ player
      - BUT $8K+ busts below 10 FPTS 40% of the time
      - The top scorer on the slate is <10% owned 62% of the time

    "Must-have" criteria (all must be true):
      1. $8K+ salary (ceiling tier)
      2. PP1 (53% of separators are PP1)
      3. Projection gap ≥ 3.0 FPTS over #2 at their position
      4. On the team with highest implied total
      5. Ownership ≥ 25% (field already recognizes them — can't dodge)

    When must-haves exist: REDUCE the contrarian penalty for lineups
    containing them. Don't blindly swap them out for low-owned scrubs.
    """
    pool = player_pool.copy()
    name_col = 'name' if 'name' in pool.columns else 'Player'
    team_col = 'team' if 'team' in pool.columns else 'Team'
    pos_col = 'position' if 'position' in pool.columns else 'Pos'
    proj_col = 'projected_fpts' if 'projected_fpts' in pool.columns else 'FC Proj'
    sal_col = 'salary' if 'salary' in pool.columns else 'Salary'

    # Ensure numeric
    for col in [proj_col, sal_col]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors='coerce')

    # Get ownership
    if ownership_map:
        pool['_own'] = pool[name_col].map(ownership_map).fillna(1.0)
    else:
        pool['_own'] = 1.0

    # Count teams
    n_teams = pool[team_col].nunique()
    is_small = n_teams <= 6

    # Find team totals
    team_total_col = None
    for col in ['team_total', 'TeamTotal', 'TeamGoal']:
        if col in pool.columns:
            team_total_col = col
            break

    # Compute projection rank at each position
    must_haves = []
    skaters = pool[pool[pos_col] != 'G'].copy()

    if proj_col in skaters.columns and sal_col in skaters.columns:
        for pos in skaters[pos_col].unique():
            pos_players = skaters[skaters[pos_col] == pos].copy()
            if len(pos_players) < 2:
                continue

            pos_players = pos_players.sort_values(proj_col, ascending=False)
            top = pos_players.iloc[0]
            second = pos_players.iloc[1]
            gap = top[proj_col] - second[proj_col]

            top_name = top[name_col]
            top_sal = top[sal_col]
            top_own = top['_own']
            top_proj = top[proj_col]

            # Check PP1
            is_pp1 = False
            for pp_col in ['pp_unit', 'PP Unit', 'pp1']:
                if pp_col in pool.columns:
                    pp_val = top.get(pp_col)
                    if pp_val == 1 or pp_val == '1':
                        is_pp1 = True
                    break

            # Check if on best team
            is_best_team = False
            if team_total_col:
                team_totals = pool.groupby(team_col)[team_total_col].first()
                best_team = team_totals.idxmax() if len(team_totals) > 0 else None
                is_best_team = (top[team_col] == best_team)

            # MUST-HAVE criteria:
            # High salary + PP1 + large projection gap + high ownership
            # These are the players you CAN'T dodge
            is_must_have = (
                top_sal >= 8000 and
                is_pp1 and
                gap >= 3.0 and
                top_own >= 25
            )

            # SOFT MUST-HAVE: slightly relaxed criteria
            # High salary + projection gap + high ownership (no PP1 required)
            is_soft_must = (
                top_sal >= 7500 and
                gap >= 2.0 and
                top_own >= 20
            )

            if is_must_have or (is_soft_must and is_best_team):
                must_haves.append(top_name)

    # Also check goalies — if one goalie is 35%+ owned, they're must-have
    goalies = pool[pool[pos_col] == 'G'].copy()
    if len(goalies) > 0:
        top_goalie = goalies.nlargest(1, '_own').iloc[0]
        if top_goalie['_own'] >= 35:
            must_haves.append(top_goalie[name_col])

    # Chalk intensity
    max_own = pool['_own'].max()
    n_over_25 = (pool['_own'] >= 25).sum()
    if max_own >= 50 or n_over_25 >= 4:
        intensity = 'EXTREME'
    elif max_own >= 30 or n_over_25 >= 2:
        intensity = 'MODERATE'
    else:
        intensity = 'DISPERSED'

    # Contrarian opportunity
    if is_small and intensity == 'EXTREME':
        contra_opp = 'LOW'
        mode_rec = ('CHALK-LEAN: Small slate with extreme chalk concentration. '
                     'Include must-haves, find separation at 1-2 other positions.')
    elif is_small:
        contra_opp = 'MEDIUM'
        mode_rec = ('BALANCED: Small slate but chalk isn\'t extreme. '
                     'Mix must-haves with 1-2 separators.')
    elif intensity == 'DISPERSED':
        contra_opp = 'HIGH'
        mode_rec = ('FULL CONTRARIAN: Large slate with dispersed ownership. '
                     'Target separators + opposing goalie leverage.')
    elif len(must_haves) >= 2:
        contra_opp = 'MEDIUM'
        mode_rec = (f'SELECTIVE: {len(must_haves)} must-have players detected. '
                     'Include them but go contrarian everywhere else.')
    else:
        contra_opp = 'HIGH'
        mode_rec = ('CONTRARIAN: Standard slate. Target separators, '
                     'play opposing goalie, avoid chalk stacks.')

    return SlateContext(
        n_teams=n_teams,
        is_small_slate=is_small,
        must_have_players=must_haves,
        n_must_haves=len(must_haves),
        chalk_intensity=intensity,
        contrarian_opportunity=contra_opp,
        mode_recommendation=mode_rec,
    )


# ═══════════════════════════════════════════════════════════════
# Leverage Scorer
# ═══════════════════════════════════════════════════════════════

class LeverageScorer:
    """
    Re-rank lineups by ownership-adjusted tournament equity.

    Standard scoring: M+3σ (pure ceiling)
    Leverage scoring:  M+3σ × ownership_multiplier × separator_bonus

    The ownership multiplier rewards contrarian lineups:
      - Lineup avg ownership 5%:  multiplier ≈ 1.15 (+15%)
      - Lineup avg ownership 10%: multiplier ≈ 1.00 (neutral)
      - Lineup avg ownership 20%: multiplier ≈ 0.85 (-15%)

    The separator bonus rewards lineups with players who can
    SEPARATE from their salary tier at low ownership:
      - Separator candidate: PP1 + Proj≥10 + Own<10% (2.2x lift, 44% recall)
      - When separators hit, they avg 26.8 FPTS exclusively for YOUR lineup
      - 48% of all 20+ FPTS games come from players <5% owned

    Additional bonuses:
      - Opposing goalie bonus: +5% if lineup has the chalk team's opp goalie
      - Double down bonus: +3% for opp goalie + stack against chalk
    """

    # Calibrated from 10 SE contests
    OWNERSHIP_BASELINE = 10.0
    OWNERSHIP_PENALTY_PER_PCT = 0.015
    OWNERSHIP_BONUS_PER_PCT = 0.015

    # Opposing goalie vs chalk: r = -0.581
    OPP_GOALIE_BONUS = 0.05

    # Separator scoring (from 12,841 obs analysis)
    # Separator candidate: PP1 + Proj≥10 + Own<10% → 2.2x lift
    SEP_BONUS_PER_PLAYER = 0.025    # +2.5% per separator candidate
    SEP_OWN_THRESHOLD = 10.0        # % owned threshold for separator
    SEP_PROJ_THRESHOLD = 10.0       # FPTS projection threshold
    # Elite separator: <5% owned + Proj≥12 → highest separation power
    ELITE_SEP_BONUS = 0.035         # +3.5% per elite separator
    ELITE_SEP_OWN = 5.0
    ELITE_SEP_PROJ = 12.0

    # Chalk prediction features (from enhanced ownership model)
    # These features drive 25%+ ownership beyond raw projection:
    #   1. Overall projection rank (r=0.284 importance)
    #   2. PP1 status (r=0.114)
    #   3. Best team total (r=0.096)
    #   4. Slate size / n_teams (r=0.061)
    #   5. Position projection gap from #2 (r=0.022)
    # Players scoring high on these are likely to be OVER-owned

    # GPP Leverage framework (adapted from 4for4):
    # LEV = implied_ownership / projected_ownership
    # implied_ownership = P(target) / sum(P(target) at position) × total_pos_ownership
    # P(target) = 1 - Phi((target - projection) / std)
    #
    # Target scores per position (calibrated from 96 SE slates,
    # winning scores avg 168 contrarian / 177 chalk):
    TARGET_BY_POS = {'C': 19.0, 'W': 18.5, 'D': 16.5, 'G': 22.0}

    # Salary-adjusted volatility (std of projection error by pos/salary):
    # Computed from 12,841 $121 SE observations
    VOL_LOOKUP = {
        ('C', 'low'):  6.1,  ('C', 'mid'):  7.2,  ('C', 'high'): 9.2,  ('C', 'elite'): 11.2,
        ('W', 'low'):  6.3,  ('W', 'mid'):  7.2,  ('W', 'high'): 8.9,  ('W', 'elite'):  9.6,
        ('D', 'low'):  5.2,  ('D', 'mid'):  6.2,  ('D', 'high'): 6.9,  ('D', 'elite'): 10.1,
        ('G', 'low'):  9.9,  ('G', 'mid'):  9.9,  ('G', 'high'): 9.9,  ('G', 'elite'):  9.7,
    }

    def __init__(
        self,
        ownership_map: Dict[str, float],
        chalk_analysis: ChalkAnalysis = None,
        contest_type: str = 'se_gpp',
        player_pool: pd.DataFrame = None,
        slate_context: SlateContext = None,
    ):
        """
        Args:
            ownership_map: {player_name: ownership_%}
            chalk_analysis: Result from detect_chalk_stack()
            contest_type: 'se_gpp', 'wta', 'cash'. Only applies to GPP types.
            player_pool: Full player pool DataFrame for separator/chalk detection.
            slate_context: Result from analyze_slate_context(). Controls adaptive behavior.
        """
        self.ownership = ownership_map
        self.chalk = chalk_analysis
        self.contest_type = contest_type
        self.pool = player_pool
        self.slate = slate_context

        # For cash games, ownership leverage doesn't matter
        self.enabled = contest_type in ('se_gpp', 'wta', 'gpp', '3max', '20max')

        # Adaptive multiplier strength based on slate context
        # Default: full contrarian lean
        self._ownership_penalty = self.OWNERSHIP_PENALTY_PER_PCT
        self._ownership_bonus = self.OWNERSHIP_BONUS_PER_PCT
        self._must_have_names = set()

        if slate_context:
            self._must_have_names = set(slate_context.must_have_players)

            if slate_context.contrarian_opportunity == 'LOW':
                # Small slate, extreme chalk: REDUCE contrarian lean
                # Don't penalize chalk as much, don't bonus contrarian as much
                self._ownership_penalty = 0.008   # Halved from 0.015
                self._ownership_bonus = 0.008
            elif slate_context.contrarian_opportunity == 'MEDIUM':
                # Mixed: moderate adjustment
                self._ownership_penalty = 0.012
                self._ownership_bonus = 0.012
            # HIGH = keep defaults (full contrarian)

        # Pre-compute separator candidates from pool
        self._separator_names = set()
        self._elite_separator_names = set()
        self._pp1_players = set()
        self._player_leverage = {}
        if player_pool is not None:
            self._precompute_separators(player_pool)
            self.compute_player_leverage()

    def _precompute_separators(self, pool: pd.DataFrame):
        """
        Identify separator candidates from the player pool.

        Separator candidate (2.2x lift, 44% recall of actual 20+ FPTS):
            PP1 + Proj ≥ 10 + Own < 10%

        Elite separator (highest exclusive FPTS when they hit):
            Own < 5% + Proj ≥ 12

        From 12,841 observations:
            - 48% of all 20+ FPTS games come from players <5% owned
            - Separators come disproportionately from $6.5-8K tier (12.1% sep rate)
            - PP1 players are 53% of separators vs 44% of all players
            - Defensemen are UNDER-represented (23% of seps vs 30% of pool)
              but when they separate they tend to be true differentiators
        """
        name_col = 'name' if 'name' in pool.columns else 'Player'
        pos_col = 'position' if 'position' in pool.columns else 'Pos'
        proj_col = 'projected_fpts' if 'projected_fpts' in pool.columns else 'FC Proj'

        for _, row in pool.iterrows():
            name = row[name_col]
            own = self.ownership.get(name, 2.0)
            proj = row.get(proj_col, 0)
            if pd.isna(proj):
                proj = 0

            # Check PP1 status
            is_pp1 = False
            for col in ['pp_unit', 'PP Unit', 'pp1']:
                if col in pool.columns:
                    val = row.get(col)
                    if val == 1 or val == '1':
                        is_pp1 = True
                    break

            # Separator candidate: PP1 + Proj≥10 + Own<10%
            if is_pp1 and proj >= self.SEP_PROJ_THRESHOLD and own < self.SEP_OWN_THRESHOLD:
                self._separator_names.add(name)

            # Elite separator: Own<5% + Proj≥12 (doesn't require PP1)
            if own < self.ELITE_SEP_OWN and proj >= self.ELITE_SEP_PROJ:
                self._elite_separator_names.add(name)

            if is_pp1:
                self._pp1_players.add(name)

    def _get_salary_tier(self, salary: float) -> str:
        """Map salary to volatility tier."""
        if salary >= 8000: return 'elite'
        elif salary >= 6000: return 'high'
        elif salary >= 4000: return 'mid'
        else: return 'low'

    def compute_player_leverage(self) -> Dict[str, float]:
        """
        Compute per-player GPP Leverage Score using the 4for4 framework:
          LEV = implied_ownership / projected_ownership

        Where:
          implied_own = P(target) / sum(P(target) at position) × total_pos_own
          P(target) = 1 - Phi((target - projection) / std)

        Returns:
            {player_name: leverage_score}
        """
        from scipy.stats import norm

        if self.pool is None:
            return {}

        pool = self.pool.copy()
        name_col = 'name' if 'name' in pool.columns else 'Player'
        pos_col = 'position' if 'position' in pool.columns else 'Pos'
        proj_col = 'projected_fpts' if 'projected_fpts' in pool.columns else 'FC Proj'
        sal_col = 'salary' if 'salary' in pool.columns else 'Salary'

        # Ensure numeric
        for col in [proj_col, sal_col]:
            if col in pool.columns:
                pool[col] = pd.to_numeric(pool[col], errors='coerce')

        # Compute P(target) for each player
        p_targets = {}
        player_positions = {}

        for _, row in pool.iterrows():
            name = row[name_col]
            pos = row[pos_col]
            proj = row.get(proj_col, 0)
            sal = row.get(sal_col, 5000)
            if pd.isna(proj) or pd.isna(sal):
                continue

            # Target score
            target = self.TARGET_BY_POS.get(pos, 18.0)
            sal_adj = max(0, (sal - 5000) / 1000) * 0.5
            target += sal_adj

            # Salary-adjusted volatility
            tier = self._get_salary_tier(sal)
            std = self.VOL_LOOKUP.get((pos, tier), 7.0)

            # P(hitting target)
            if std > 0:
                z = (target - proj) / std
                p = 1 - norm.cdf(z)
            else:
                p = 0.5

            p_targets[name] = max(p, 0.001)  # Floor to avoid div by zero
            player_positions[name] = pos

        # Compute implied ownership per position
        leverage_scores = {}

        # Group by position
        pos_groups = {}
        for name, pos in player_positions.items():
            if pos not in pos_groups:
                pos_groups[pos] = []
            pos_groups[pos].append(name)

        for pos, names in pos_groups.items():
            total_p = sum(p_targets.get(n, 0) for n in names)
            total_pos_own = sum(self.ownership.get(n, 1.0) for n in names)

            if total_p <= 0 or total_pos_own <= 0:
                for n in names:
                    leverage_scores[n] = 1.0
                continue

            for n in names:
                p_share = p_targets.get(n, 0) / total_p
                implied_own = p_share * total_pos_own
                projected_own = max(self.ownership.get(n, 1.0), 0.5)
                leverage_scores[n] = implied_own / projected_own

        self._player_leverage = leverage_scores
        return leverage_scores

    def _get_lineup_ownership(self, lineup: pd.DataFrame) -> Dict:
        """Compute ownership and separator metrics for a lineup."""
        name_col = 'name' if 'name' in lineup.columns else 'Player'
        pos_col = 'position' if 'position' in lineup.columns else 'Pos'
        proj_col = 'projected_fpts' if 'projected_fpts' in lineup.columns else 'FC Proj'

        player_owns = []
        for _, row in lineup.iterrows():
            own = self.ownership.get(row[name_col], 2.0)  # Default 2% for unknown
            player_owns.append(own)

        avg_own = np.mean(player_owns)
        max_own = np.max(player_owns)
        n_chalk = sum(1 for o in player_owns if o > 20)
        n_contrarian = sum(1 for o in player_owns if o < 5)

        # Count separator candidates in the lineup
        n_separators = 0
        n_elite_separators = 0
        # Projection-weighted average leverage
        # A LEV=11 player projected for 6 FPTS shouldn't dominate
        # over a LEV=2.0 player projected for 25 FPTS.
        proj_col = 'projected_fpts' if 'projected_fpts' in lineup.columns else 'FC Proj'
        weighted_lev_num = 0
        weighted_lev_den = 0
        player_leverages = []
        for _, row in lineup.iterrows():
            name = row[name_col]
            if name in self._separator_names:
                n_separators += 1
            if name in self._elite_separator_names:
                n_elite_separators += 1
            lev = self._player_leverage.get(name, 1.0)
            player_leverages.append(lev)
            proj = row.get(proj_col, 10.0)
            if pd.isna(proj):
                proj = 10.0
            weighted_lev_num += lev * proj
            weighted_lev_den += proj

        avg_leverage = weighted_lev_num / weighted_lev_den if weighted_lev_den > 0 else 1.0

        # Check for opposing goalie
        has_opp_goalie = False
        if self.chalk and self.chalk.opp_goalie:
            for _, row in lineup.iterrows():
                if row[name_col] == self.chalk.opp_goalie:
                    has_opp_goalie = True
                    break

        # Check if lineup stacks against chalk
        stacks_against_chalk = False
        if self.chalk and self.chalk.opp_team:
            team_col = 'team' if 'team' in lineup.columns else 'Team'
            opp_team_count = (lineup[team_col] == self.chalk.opp_team).sum()
            if opp_team_count >= 3:
                stacks_against_chalk = True

        return {
            'avg_own': avg_own,
            'max_own': max_own,
            'n_chalk': n_chalk,
            'n_contrarian': n_contrarian,
            'n_separators': n_separators,
            'n_elite_separators': n_elite_separators,
            'avg_leverage': avg_leverage,
            'player_leverages': player_leverages,
            'has_opp_goalie': has_opp_goalie,
            'stacks_against_chalk': stacks_against_chalk,
        }

    def compute_leverage_multiplier(self, lineup: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Compute the ownership leverage multiplier for a lineup.

        PRIMARY SIGNAL: Average player-level GPP Leverage Score (4for4 framework)
          LEV = implied_ownership / projected_ownership per player
          avg_leverage > 1.0 → lineup is under-owned relative to winning probability
          avg_leverage < 1.0 → lineup is over-owned relative to winning probability

        The old system used a linear ownership penalty (high own% = bad).
        The new system asks: is each player's ownership JUSTIFIED by their
        P(being a winning play)? A 30% owned player with LEV=1.2 is fine.
        A 10% owned player with LEV=0.3 is over-owned.

        STRUCTURAL BONUSES (additive, on top of leverage signal):
          - Opposing goalie: +5% (r=-0.581 with chalk stack)
          - Double down (opp goalie + anti-chalk stack): +3%
          - Must-have combo (must-have chalk + separators): +2% per must-have

        Returns:
            (multiplier, details_dict)
        """
        if not self.enabled:
            return 1.0, {'reason': 'cash game — leverage disabled'}

        own_info = self._get_lineup_ownership(lineup)
        avg_own = own_info['avg_own']
        avg_lev = own_info['avg_leverage']
        name_col = 'name' if 'name' in lineup.columns else 'Player'

        # ══════════════════════════════════════════════════════
        # PRIMARY SIGNAL: Average GPP Leverage Score
        #
        # avg_lev = 1.0 → lineup is fairly owned (neutral)
        # avg_lev = 2.0 → lineup is under-owned 2x (strong contrarian)
        # avg_lev = 0.5 → lineup is over-owned 2x (strong chalk)
        #
        # Convert to multiplier using log scale to avoid
        # extreme values from very high leverage players.
        # log2(1.0) = 0, log2(2.0) = 1, log2(0.5) = -1
        #
        # Sensitivity: LEV_WEIGHT controls how much leverage
        # tilts the multiplier. At 0.06:
        #   avg_lev=2.0 → mult=1.06 (+6%)
        #   avg_lev=4.0 → mult=1.12 (+12%)
        #   avg_lev=0.5 → mult=0.94 (-6%)
        # ══════════════════════════════════════════════════════
        LEV_WEIGHT = 0.06  # % adjustment per doubling of leverage

        # Clamp avg_lev to prevent extreme outliers from dominating
        clamped_lev = np.clip(avg_lev, 0.1, 20.0)
        lev_signal = np.log2(clamped_lev) * LEV_WEIGHT
        multiplier = 1.0 + lev_signal

        details = {
            'avg_leverage': avg_lev,
            'clamped_lev': clamped_lev,
            'lev_signal': lev_signal,
            'base_mult': multiplier,
            'avg_own': avg_own,
            'scoring_mode': 'LEVERAGE',
        }

        # ══════════════════════════════════════════════════════
        # MUST-HAVE ADJUSTMENT
        #
        # If must-haves exist, check if this lineup has them.
        # Must-haves have LEV ≥ 1.0 by definition (their
        # winning probability justifies their ownership), so
        # the leverage signal already handles them correctly.
        # But we add a small combo bonus when must-haves are
        # paired with separators — that's the ideal lineup.
        # ══════════════════════════════════════════════════════
        n_must_haves_in = 0
        for _, row in lineup.iterrows():
            if row[name_col] in self._must_have_names:
                n_must_haves_in += 1

        details['n_must_haves_in'] = n_must_haves_in

        # Combo bonus: must-have + separators = ceiling + exclusivity
        if n_must_haves_in > 0 and (own_info['n_separators'] >= 1 or own_info['n_elite_separators'] >= 1):
            combo_bonus = 0.02 * n_must_haves_in
            multiplier += combo_bonus
            details['combo_bonus'] = combo_bonus

        # Small penalty for missing must-haves when they exist
        # (tail risk: 9.4% of slates the must-have scores 40+)
        if self._must_have_names and n_must_haves_in == 0:
            missing_penalty = 0.015 * len(self._must_have_names)
            multiplier -= missing_penalty
            details['missing_mh_penalty'] = missing_penalty

        # ══════════════════════════════════════════════════════
        # STRUCTURAL BONUSES (unchanged from original)
        # These are real edges from the data, independent of
        # the leverage signal.
        # ══════════════════════════════════════════════════════

        # Opposing goalie bonus: r=-0.581 with chalk stack
        if own_info['has_opp_goalie']:
            multiplier += self.OPP_GOALIE_BONUS
            details['opp_goalie_bonus'] = self.OPP_GOALIE_BONUS

        # Double down: opp goalie + stacking against chalk
        if own_info['has_opp_goalie'] and own_info['stacks_against_chalk']:
            multiplier += 0.03
            details['double_down_bonus'] = 0.03

        # Cap multiplier to avoid extremes
        multiplier = np.clip(multiplier, 0.70, 1.30)

        details.update(own_info)
        details['final_mult'] = multiplier
        return multiplier, details

    def score_lineups(
        self,
        lineups: List[pd.DataFrame],
        sim_results: List[Dict],
        verbose: bool = True,
    ) -> List[Tuple[pd.DataFrame, Dict, float, Dict]]:
        """
        Re-rank lineups by leverage-adjusted M+3σ.

        Args:
            lineups: Candidate lineups from optimizer
            sim_results: Matching sim results from SimSelector
            verbose: Print ranking table

        Returns:
            List of (lineup, sim_result, leverage_score, leverage_details)
            sorted by leverage_score descending.
        """
        scored = []

        for lu, res in zip(lineups, sim_results):
            m3s = res.get('m3s', res['mean'] + 3 * res['std'])
            mult, details = self.compute_leverage_multiplier(lu)
            leverage_score = m3s * mult
            scored.append((lu, res, leverage_score, details))

        scored.sort(key=lambda x: x[2], reverse=True)

        if verbose:
            self._print_results(scored, lineups, sim_results)

        return scored

    def _print_results(self, scored, orig_lineups, orig_results, n_show=15):
        """Print leverage-scored ranking."""
        # Build original rank map
        orig_m3s = [(r.get('m3s', r['mean'] + 3 * r['std']), i) 
                     for i, r in enumerate(orig_results)]
        orig_m3s.sort(reverse=True)
        orig_rank_map = {idx: rank + 1 for rank, (_, idx) in enumerate(orig_m3s)}

        if self.chalk:
            print(f"\n  ╔══ CHALK ANALYSIS ═══════════════════════════════════════╗")
            print(f"  ║  Chalk team: {self.chalk.chalk_team} "
                  f"(total own: {self.chalk.chalk_team_own:.0f}%)")
            if self.chalk.opp_goalie:
                print(f"  ║  Leverage play: {self.chalk.opp_goalie} "
                      f"({self.chalk.opp_team} G, {self.chalk.opp_goalie_own:.1f}% owned)")
            print(f"  ║  Leverage rating: {self.chalk.leverage_rating}")
            print(f"  ╚═══════════════════════════════════════════════════════════╝")

        if self.slate:
            print(f"\n  ╔══ SLATE CONTEXT ════════════════════════════════════════╗")
            print(f"  ║  Teams: {self.slate.n_teams} | "
                  f"Chalk intensity: {self.slate.chalk_intensity} | "
                  f"Contrarian opp: {self.slate.contrarian_opportunity}")
            if self.slate.must_have_players:
                mh_list = ', '.join(self.slate.must_have_players[:3])
                print(f"  ║  Must-haves: {mh_list}")
                print(f"  ║  (ceiling gap too large to fade — include + separate elsewhere)")
            print(f"  ║  Mode: {self.slate.mode_recommendation}")
            print(f"  ╚═══════════════════════════════════════════════════════════╝")

        print(f"\n{'═' * 100}")
        print(f"  LEVERAGE-ADJUSTED LINEUP RANKING — {len(scored)} candidates")
        print(f"  Formula: LevScore = M+3σ × ownership_multiplier")
        print(f"{'═' * 100}")
        print(f"  {'#':>3} {'SimRk':>6} {'M+3σ':>6} {'Mult':>5} {'LevScr':>7} "
              f"{'AvgOwn':>7} {'AvgLEV':>7} {'Chalk':>5} {'MH':>3} {'OppG':>4} {'Stacks':>8}")
        print(f"  {'-' * 90}")

        for i, (lu, res, lev_score, details) in enumerate(scored[:n_show]):
            m3s = res.get('m3s', res['mean'] + 3 * res['std'])
            mult = details['final_mult']
            avg_own = details['avg_own']
            avg_lev = details.get('avg_leverage', 1.0)
            n_chalk = details['n_chalk']
            n_mh = details.get('n_must_haves_in', 0)
            opp_g = '✓' if details.get('has_opp_goalie') else ''
            double = '💰' if details.get('stacks_against_chalk') and details.get('has_opp_goalie') else ''

            # Find original sim rank
            orig_idx = None
            for j, orig_lu in enumerate(orig_lineups):
                if lu is orig_lu or (hasattr(lu, 'equals') and lu.equals(orig_lu)):
                    orig_idx = j
                    break
            sim_rank = orig_rank_map.get(orig_idx, '?') if orig_idx is not None else '?'

            # Stack description
            team_col = 'team' if 'team' in lu.columns else 'Team'
            counts = lu[team_col].value_counts()
            stacks = '-'.join(str(c) for c in sorted(counts.values, reverse=True) if c >= 2)

            changed = ' ←' if sim_rank != (i + 1) else ''
            print(f"  {i+1:>3} {sim_rank:>6} {m3s:>6.0f} {mult:>5.2f} {lev_score:>7.0f} "
                  f"{avg_own:>6.1f}% {avg_lev:>6.2f}x {n_chalk:>5} {n_mh:>3} {opp_g:>4} "
                  f"{stacks:>8}{double}{changed}")

        # How many re-ranked?
        n_moved = sum(1 for i, (lu, _, _, _) in enumerate(scored)
                      if orig_rank_map.get(
                          next((j for j, ol in enumerate(orig_lineups) if lu is ol), None), i+1
                      ) != i + 1)
        print(f"\n  Lineups re-ranked: {n_moved}/{min(n_show, len(scored))}")

        # Top pick summary
        best_lu, best_res, best_lev, best_det = scored[0]
        name_col = 'name' if 'name' in best_lu.columns else 'Player'
        goalie = best_lu[best_lu[
            'position' if 'position' in best_lu.columns else 'Pos'
        ] == 'G']
        g_name = goalie.iloc[0][name_col] if len(goalie) > 0 else '?'
        g_own = self.ownership.get(g_name, 0)

        print(f"\n  ✅ Top pick: LevScore={best_lev:.0f} "
              f"(M+3σ={best_res.get('m3s', 0):.0f} × {best_det['final_mult']:.2f})")
        print(f"     Goalie: {g_name} ({g_own:.1f}% owned)")
        print(f"     Avg lineup ownership: {best_det['avg_own']:.1f}%")
        print(f"     Avg GPP leverage: {best_det.get('avg_leverage', 1.0):.2f}x "
              f"({'under-owned ✓' if best_det.get('avg_leverage', 1.0) > 1.0 else 'over-owned'})")
        print(f"     Separators: {best_det.get('n_separators', 0)} "
              f"(elite: {best_det.get('n_elite_separators', 0)})")
        if best_det.get('n_must_haves_in', 0) > 0:
            print(f"     Must-haves: {best_det['n_must_haves_in']} included")
        if best_det.get('has_opp_goalie'):
            print(f"     ⚡ Has opposing goalie — maximum leverage play!")


# ═══════════════════════════════════════════════════════════════
# Convenience: One-call function
# ═══════════════════════════════════════════════════════════════

def leverage_select(
    lineups: List[pd.DataFrame],
    sim_results: List[Dict],
    player_pool: pd.DataFrame,
    ownership_map: Dict[str, float] = None,
    contest_type: str = 'se_gpp',
    verbose: bool = True,
) -> List[Tuple[pd.DataFrame, Dict, float, Dict]]:
    """
    One-call leverage scoring. Drop-in addition after sim_selector.

    Usage in main.py:
        # After sim_selector picks top 20...
        from leverage_score import leverage_select
        ranked = leverage_select(top_lineups, top_results, player_pool, ownership_map)
        best_lineup = ranked[0][0]
    """
    # Build ownership map if not provided
    if ownership_map is None:
        ownership_map = {}
        name_col = 'name' if 'name' in player_pool.columns else 'Player'
        for col in ['projected_own', 'own_proj', 'ownership', 'Ownership']:
            if col in player_pool.columns:
                ownership_map = dict(zip(
                    player_pool[name_col],
                    pd.to_numeric(player_pool[col], errors='coerce').fillna(1.0)
                ))
                break

    # Detect chalk stack
    chalk = detect_chalk_stack(player_pool, ownership_map)

    # Analyze slate context (must-haves, slate size, chalk intensity)
    slate = analyze_slate_context(player_pool, ownership_map)

    if verbose:
        if chalk:
            print(f"\n  Chalk detected: {chalk.chalk_team} ({chalk.chalk_team_own:.0f}% total own)")
            if chalk.opp_goalie:
                print(f"  Opposing goalie leverage: {chalk.opp_goalie} ({chalk.opp_goalie_own:.1f}% owned)")
        if slate:
            print(f"  Slate: {slate.n_teams} teams | {slate.chalk_intensity} chalk | "
                  f"{slate.contrarian_opportunity} contrarian opp")
            if slate.must_have_players:
                print(f"  Must-haves: {', '.join(slate.must_have_players)}")
            print(f"  Strategy: {slate.mode_recommendation}")

    # Score with full context
    scorer = LeverageScorer(
        ownership_map, chalk, contest_type,
        player_pool=player_pool, slate_context=slate,
    )
    return scorer.score_lineups(lineups, sim_results, verbose=verbose)
