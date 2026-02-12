#!/usr/bin/env python3
"""
Agent Panel — Simons-Style Qualitative Overlay for DFS Lineup Selection
=========================================================================

Three specialist AI agents review the top sim-ranked lineups and apply
qualitative filters the quantitative model can't capture:

  Agent 1 (Contrarian): ownership leverage, field differentiation
  Agent 2 (Narrative):  injuries, B2B, revenge games, news, streaks
  Agent 3 (Structure):  stack quality, goalie matchup, game environment

Architecture:
  Simulation Engine → Top 20 lineups → Agent Panel → Final 1-3 picks

Key constraint: Agents can VETO but not PROMOTE. They only filter
lineups the sim engine already ranked highly. The math leads.

Backtest tracking:
  Every slate logs sim_only_fpts vs agent_panel_fpts.
  After 20 slates, if mean(agent_delta) < 0, agents get removed.

Usage:
    from agent_panel import AgentPanel, run_agent_panel

    panel = AgentPanel(player_pool, slate_context)
    final_picks = panel.review(top_20_lineups, scored_results)
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
#  Agent Prompts
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are part of a 3-agent panel evaluating NHL DFS lineups for DraftKings.
You are {agent_name} — {agent_role}.

You will receive:
1. Top 20 sim-ranked lineups with statistics (mean, std, M+3σ, P(cash), P(140), stacks)
2. Slate context (injuries, B2B, confirmed goalies, Vegas lines, news)
3. Ownership projections

Your job: Review each lineup and provide a VERDICT for each:
- APPROVE: No issues found
- FLAG: Minor concern, reduce score by 5 points
- VETO: Major disqualifying issue, reduce score by 15 points (use sparingly)
- BOOST: Qualitative advantage the model can't see, add 3 points

CRITICAL RULES:
- You CANNOT override the simulation math. If a lineup is sim-ranked #1, you need
  strong qualitative evidence to VETO it (and even then, 2 of 3 agents must agree).
- Be SPECIFIC. Don't say "I don't like this stack." Say "EDM stack is on back-to-back
  with travel after playing last night in Vancouver."
- You are looking for information ASYMMETRY — things the quantitative model doesn't know.

Respond in JSON format:
{{
  "agent": "{agent_name}",
  "reviews": [
    {{
      "lineup_rank": 1,
      "verdict": "APPROVE|FLAG|VETO|BOOST",
      "reason": "specific reason",
      "confidence": 0.0-1.0
    }},
    ...
  ],
  "top_3_picks": [1, 5, 3],
  "reasoning": "brief overall assessment"
}}"""

CONTRARIAN_ROLE = """the Contrarian Agent. Your mandate: maximize differentiation from the field.

You analyze:
- Player ownership concentrations (flag lineups where 3+ players are >25% owned with no low-owned contrarian)
- Stack popularity (flag if primary stack is the most popular team in the field)
- Goalie ownership (prefer goalies <20% owned if sim-ranked well)
- Leverage spots (identify players who are underowned relative to their projection)

You believe: In GPP/WTA, being different when right is worth more than being right when chalk.
In a 10-person WTA, you need to beat 9 people. Playing the same lineup as everyone = no edge."""

NARRATIVE_ROLE = """the Narrative Agent. Your mandate: catch qualitative factors the model can't see.

You analyze:
- Injury status (GTD players with no backup plan, just-returned players)
- Back-to-back games (team played last night, especially with travel)
- Recent momentum beyond what rolling averages capture
- News (trade deadline moves, coaching changes, line shuffles announced after data pull)
- Revenge games (former player vs old team — historically boosts performance)
- Goalie situations (backup in net, goalie controversy, hot/cold streaks)

You believe: The model uses yesterday's data. You see today's reality."""

STRUCTURE_ROLE = """the Structural Agent. Your mandate: evaluate lineup construction quality.

You analyze:
- Stack composition (are stack players on the SAME LINE? Same PP unit? Mixed lines = fake correlation)
- Goalie-skater alignment (goalie opposing our primary stack = BAD, r=-0.34 works against us)
- Salary distribution (top-heavy vs balanced — for GPP, slight top-heavy is fine)
- Game environment (are stacks in the highest O/U game? Do Vegas lines support the stack?)
- Correlation structure (does the lineup's correlation profile match what wins GPPs?)

You believe: Lineup construction matters as much as player selection. A 4-man stack from
the same PP1 unit is worth far more than 4 random players from the same team.

Key correlation facts (from 29,339 game analysis):
- Same line skaters: r = 0.124
- Different line, same team: r = 0.034
- Goalie ↔ own skaters: r = +0.191
- Goalie ↔ opponent skaters: r = -0.340 (STRONGEST — stack AGAINST your goalie)"""


# ═══════════════════════════════════════════════════════════════════
#  Slate Context Builder
# ═══════════════════════════════════════════════════════════════════

class SlateContext:
    """Collects all qualitative context for agent review."""

    def __init__(self):
        self.injuries: List[Dict] = []          # {name, team, status, note}
        self.back_to_back: List[str] = []        # team abbreviations
        self.confirmed_goalies: List[Dict] = []  # {name, team, opponent}
        self.news: List[str] = []                # free text news items
        self.vegas_lines: List[Dict] = []        # {home, away, total, home_ml, away_ml}
        self.ownership: Dict[str, float] = {}    # name → projected own%
        self.contest_type: str = 'se_gpp'        # se_gpp, wta, cash

    def add_injury(self, name: str, team: str, status: str, note: str = ''):
        self.injuries.append({'name': name, 'team': team, 'status': status, 'note': note})

    def add_b2b(self, team: str):
        self.back_to_back.append(team)

    def add_goalie(self, name: str, team: str, opponent: str):
        self.confirmed_goalies.append({'name': name, 'team': team, 'opponent': opponent})

    def add_news(self, item: str):
        self.news.append(item)

    def add_vegas(self, home: str, away: str, total: float,
                  home_ml: int = 0, away_ml: int = 0):
        self.vegas_lines.append({
            'home': home, 'away': away, 'total': total,
            'home_ml': home_ml, 'away_ml': away_ml,
        })

    def set_ownership(self, ownership_map: Dict[str, float]):
        self.ownership = ownership_map

    def to_text(self) -> str:
        """Render context as text for agent prompts."""
        lines = []
        lines.append("=== SLATE CONTEXT ===\n")
        lines.append(f"Contest type: {self.contest_type}\n")

        if self.vegas_lines:
            lines.append("VEGAS LINES:")
            for v in self.vegas_lines:
                lines.append(f"  {v['away']} @ {v['home']}  O/U: {v['total']}")

        if self.confirmed_goalies:
            lines.append("\nCONFIRMED GOALIES:")
            for g in self.confirmed_goalies:
                lines.append(f"  {g['name']} ({g['team']}) vs {g['opponent']}")

        if self.back_to_back:
            lines.append(f"\nBACK-TO-BACK TEAMS: {', '.join(self.back_to_back)}")
            lines.append("  (These teams played last night — fatigue risk)")

        if self.injuries:
            lines.append("\nINJURY REPORT:")
            for inj in self.injuries:
                note = f" — {inj['note']}" if inj['note'] else ''
                lines.append(f"  {inj['name']} ({inj['team']}): {inj['status']}{note}")

        if self.news:
            lines.append("\nNEWS:")
            for n in self.news:
                lines.append(f"  • {n}")

        return '\n'.join(lines)

    @classmethod
    def from_pipeline(cls, player_pool: pd.DataFrame, stack_builder=None,
                      injuries_df=None, vegas_games=None):
        """Auto-build context from pipeline data."""
        ctx = cls()

        # Extract ownership from pool
        own_col = None
        for c in ['projected_own', 'own_proj', 'ownership']:
            if c in player_pool.columns:
                own_col = c
                break
        if own_col:
            name_col = 'name' if 'name' in player_pool.columns else 'Player'
            ctx.ownership = dict(zip(player_pool[name_col], player_pool[own_col]))

        # Extract confirmed goalies from stack builder
        if stack_builder and hasattr(stack_builder, 'get_all_starting_goalies'):
            try:
                goalies = stack_builder.get_all_starting_goalies()
                if goalies:
                    for g_name, g_info in goalies.items():
                        ctx.add_goalie(g_name,
                                       g_info.get('team', '?'),
                                       g_info.get('opponent', '?'))
            except Exception:
                pass

        # Vegas lines
        if vegas_games:
            for game in vegas_games:
                ctx.add_vegas(
                    home=game.get('home', '?'),
                    away=game.get('away', '?'),
                    total=game.get('total', 0),
                )

        return ctx


# ═══════════════════════════════════════════════════════════════════
#  Agent Panel
# ═══════════════════════════════════════════════════════════════════

class AgentPanel:
    """
    Three-agent review panel for lineup selection.

    Can run in two modes:
    1. API mode: Calls Claude API for each agent (automated)
    2. Manual mode: Generates prompts for human to paste into Claude chat

    Args:
        slate_context: SlateContext with injuries, B2B, goalies, news
        api_key: Anthropic API key (if None, runs in manual mode)
        model: Claude model to use
    """

    AGENTS = [
        ('Contrarian', CONTRARIAN_ROLE),
        ('Narrative', NARRATIVE_ROLE),
        ('Structure', STRUCTURE_ROLE),
    ]

    def __init__(
        self,
        slate_context: SlateContext,
        api_key: str = None,
        model: str = 'claude-sonnet-4-20250514',
    ):
        self.context = slate_context
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model = model

    def _format_lineups_for_agents(
        self,
        lineups: List[pd.DataFrame],
        results: List[Dict],
        n_show: int = 20,
    ) -> str:
        """Format top N lineups as text for agent review."""
        lines = []
        lines.append("=== TOP LINEUPS (sorted by M+3σ) ===\n")
        lines.append(f"{'Rank':>4} {'Mean':>6} {'Std':>5} {'M+3σ':>6} "
                      f"{'P(111)':>7} {'P(140)':>7} {'P95':>5} {'Stacks':>8}")
        lines.append('-' * 55)

        n = min(n_show, len(lineups))
        for i in range(n):
            lu = lineups[i]
            r = results[i]
            m3s = r.get('m3s', r['mean'] + 3 * r['std'])
            tc = lu['team'].value_counts() if 'team' in lu.columns else lu['Team'].value_counts()
            stacks = '-'.join(str(v) for v in sorted(tc.values, reverse=True) if v >= 2)

            lines.append(f"{i+1:>4} {r['mean']:>6.1f} {r['std']:>5.1f} {m3s:>6.0f} "
                          f"{r['p_cash']:>6.1%} {r['p_top5']:>6.1%} "
                          f"{r['p95']:>5.0f} {stacks:>8}")

            # Player details
            name_col = 'name' if 'name' in lu.columns else 'Player'
            team_col = 'team' if 'team' in lu.columns else 'Team'
            pos_col = 'position' if 'position' in lu.columns else 'Pos'
            sal_col = 'salary' if 'salary' in lu.columns else 'Salary'
            proj_col = 'projected_fpts' if 'projected_fpts' in lu.columns else 'Avg'

            for _, row in lu.iterrows():
                own = self.context.ownership.get(row[name_col], 0)
                own_str = f"{own:.0f}%" if own > 0 else '?%'
                lines.append(f"      {row[pos_col]:<4} {row[name_col]:<25} "
                              f"{row[team_col]:<5} ${row[sal_col]:>5,.0f} "
                              f"proj={row[proj_col]:.1f} own={own_str}")
            lines.append('')

        return '\n'.join(lines)

    def _call_agent(self, agent_name: str, agent_role: str,
                    lineup_text: str) -> Optional[Dict]:
        """Call Claude API for one agent. Returns parsed JSON response."""
        if not self.api_key:
            return None

        try:
            import requests
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': self.api_key,
                    'anthropic-version': '2023-06-01',
                },
                json={
                    'model': self.model,
                    'max_tokens': 4000,
                    'system': SYSTEM_PROMPT.format(
                        agent_name=agent_name, agent_role=agent_role),
                    'messages': [{
                        'role': 'user',
                        'content': f"{self.context.to_text()}\n\n{lineup_text}",
                    }],
                },
                timeout=60,
            )
            data = response.json()
            text = data['content'][0]['text']

            # Parse JSON from response (handle markdown fences)
            text = text.strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1]
                text = text.rsplit('```', 1)[0]

            return json.loads(text)
        except Exception as e:
            print(f"    ⚠ Agent {agent_name} API call failed: {e}")
            return None

    def review(
        self,
        lineups: List[pd.DataFrame],
        results: List[Dict],
        n_review: int = 20,
        verbose: bool = True,
    ) -> List[Tuple[pd.DataFrame, Dict, float]]:
        """
        Run all 3 agents on top N lineups, compute consensus scores.

        Returns: List of (lineup, sim_result, consensus_score) sorted by consensus.
        """
        n = min(n_review, len(lineups))
        lineup_text = self._format_lineups_for_agents(lineups[:n], results[:n])

        # Score adjustments
        ADJUSTMENTS = {'APPROVE': 0, 'FLAG': -5, 'VETO': -15, 'BOOST': +3}

        # Initialize consensus scores from sim ranking
        consensus = [n - i for i in range(n)]  # rank 1 = n points, rank n = 1 point

        agent_reviews = {}

        if self.api_key:
            # Automated mode: call API for each agent
            for agent_name, agent_role in self.AGENTS:
                if verbose:
                    print(f"  Calling Agent: {agent_name}...")

                response = self._call_agent(agent_name, agent_role, lineup_text)
                if response and 'reviews' in response:
                    agent_reviews[agent_name] = response
                    for review in response['reviews']:
                        rank = review.get('lineup_rank', 0) - 1
                        if 0 <= rank < n:
                            verdict = review.get('verdict', 'APPROVE').upper()
                            adj = ADJUSTMENTS.get(verdict, 0)
                            consensus[rank] += adj

                            if verbose and verdict != 'APPROVE':
                                conf = review.get('confidence', 0)
                                print(f"    #{rank+1}: {verdict} ({conf:.0%}) — {review.get('reason', '?')}")
        else:
            # Manual mode: print prompts for human to run in Claude chat
            if verbose:
                print(f"\n{'═' * 80}")
                print(f"  AGENT PANEL — Manual Mode (no API key)")
                print(f"  Copy the prompts below into Claude chat for each agent.")
                print(f"{'═' * 80}")

                for agent_name, agent_role in self.AGENTS:
                    print(f"\n{'─' * 60}")
                    print(f"  AGENT: {agent_name}")
                    print(f"{'─' * 60}")
                    print(f"\n{SYSTEM_PROMPT.format(agent_name=agent_name, agent_role=agent_role)}")
                    print(f"\n{self.context.to_text()}")
                    print(f"\n{lineup_text}")

        # Build final ranked list
        ranked = []
        for i in range(n):
            ranked.append((lineups[i], results[i], consensus[i]))

        ranked.sort(key=lambda x: x[2], reverse=True)

        if verbose and self.api_key:
            print(f"\n  ── Consensus Results ──")
            print(f"  {'Rank':>4} {'SimRank':>8} {'Score':>6} {'Mean':>6} {'Std':>5} {'M+3σ':>6}")
            print(f"  {'-' * 40}")
            for final_rank, (lu, r, score) in enumerate(ranked[:10], 1):
                orig_rank = results.index(r) + 1 if r in results else '?'
                m3s = r.get('m3s', r['mean'] + 3 * r['std'])
                print(f"  {final_rank:>4} {orig_rank:>8} {score:>6.0f} "
                      f"{r['mean']:>6.1f} {r['std']:>5.1f} {m3s:>6.0f}")

        return ranked

    def generate_manual_prompt(
        self,
        lineups: List[pd.DataFrame],
        results: List[Dict],
        n_review: int = 20,
    ) -> str:
        """
        Generate a single comprehensive prompt for manual Claude chat review.
        Includes all 3 agent perspectives in one prompt.
        """
        n = min(n_review, len(lineups))
        lineup_text = self._format_lineups_for_agents(lineups[:n], results[:n])

        prompt = f"""You are running the Simons Agent Panel — a 3-agent review system for NHL DFS lineups.

Review the top {n} simulation-ranked lineups below from THREE perspectives:

**Agent 1 — Contrarian:** Evaluate ownership leverage and field differentiation.
Flag lineups where 3+ players are >25% owned with no contrarian plays.
Flag if the primary stack is the chalk team.

**Agent 2 — Narrative:** Evaluate qualitative factors the model can't see.
Flag GTD players, back-to-back teams, recent news not in the data.
Boost revenge narratives or favorable goalie situations.

**Agent 3 — Structure:** Evaluate lineup construction quality.
Flag fake stacks (players on different lines). Flag goalie opposing our primary stack.
Prefer stacks in highest O/U games with same-line/PP-unit correlation.

{self.context.to_text()}

{lineup_text}

For each lineup, give a verdict (APPROVE/FLAG/VETO/BOOST) with a specific reason.
Then recommend your TOP 3 picks with reasoning.

Scoring: Each lineup starts with points equal to (21 - sim_rank).
FLAGS subtract 5 points. VETOES subtract 15. BOOSTS add 3.
Final ranking is by total score across all 3 agent perspectives.
"""
        return prompt


# ═══════════════════════════════════════════════════════════════════
#  Tracking / Logging
# ═══════════════════════════════════════════════════════════════════

TRACKER_PATH = Path(__file__).parent / "data" / "agent_tracker.json"


def log_slate_result(
    date: str,
    contest_type: str,
    sim_only_fpts: float,
    agent_panel_fpts: float,
    sim_rank: int,
    agent_rank: int,
    notes: str = '',
):
    """Log a slate result for measuring agent value over time."""
    entry = {
        'date': date,
        'contest_type': contest_type,
        'sim_only_fpts': sim_only_fpts,
        'agent_panel_fpts': agent_panel_fpts,
        'agent_delta': agent_panel_fpts - sim_only_fpts,
        'sim_rank': sim_rank,
        'agent_rank': agent_rank,
        'notes': notes,
        'logged_at': datetime.now().isoformat(),
    }

    # Load existing
    history = []
    if TRACKER_PATH.exists():
        try:
            with open(TRACKER_PATH) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(entry)

    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKER_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"  Logged: {date} | sim={sim_only_fpts:.0f} agent={agent_panel_fpts:.0f} "
          f"delta={entry['agent_delta']:+.0f}")

    return entry


def print_agent_tracker_summary():
    """Print running summary of agent panel performance."""
    if not TRACKER_PATH.exists():
        print("  No agent tracking data yet.")
        return

    with open(TRACKER_PATH) as f:
        history = json.load(f)

    if not history:
        print("  No agent tracking data yet.")
        return

    deltas = [h['agent_delta'] for h in history]
    n = len(deltas)
    avg_delta = np.mean(deltas)
    wins = sum(1 for d in deltas if d > 0)

    print(f"\n  ── Agent Panel Tracker ({n} slates) ──")
    print(f"  Avg agent delta: {avg_delta:+.1f} FPTS/slate")
    print(f"  Win rate: {wins}/{n} ({wins/n:.0%})")
    print(f"  Total edge: {sum(deltas):+.0f} FPTS")

    if n >= 20:
        if avg_delta > 2.0:
            print(f"  STATUS: ✅ AGENTS ADD VALUE — keep them")
        elif avg_delta < 0:
            print(f"  STATUS: ❌ AGENTS HURT — consider removing")
        else:
            print(f"  STATUS: ⚠ INCONCLUSIVE — need more data")
    else:
        print(f"  STATUS: 📊 {20 - n} more slates needed for evaluation")


# ═══════════════════════════════════════════════════════════════════
#  Convenience function
# ═══════════════════════════════════════════════════════════════════

def run_agent_panel(
    lineups: List[pd.DataFrame],
    results: List[Dict],
    player_pool: pd.DataFrame,
    stack_builder=None,
    vegas_games=None,
    injuries_df=None,
    contest_type: str = 'se_gpp',
    api_key: str = None,
    verbose: bool = True,
) -> List[Tuple[pd.DataFrame, Dict, float]]:
    """
    One-call convenience function to run the full agent panel.

    Returns: Ranked list of (lineup, result, consensus_score).
    """
    # Build context
    ctx = SlateContext.from_pipeline(player_pool, stack_builder, injuries_df, vegas_games)
    ctx.contest_type = contest_type

    # Run panel
    panel = AgentPanel(ctx, api_key=api_key)
    ranked = panel.review(lineups, results, verbose=verbose)

    return ranked
