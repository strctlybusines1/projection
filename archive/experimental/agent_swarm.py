#!/usr/bin/env python3
"""
Multi-Model Agent Swarm
========================

Extends the AgentPanel to route each specialist agent to a different
LLM provider, eliminating shared blind spots.

Default routing:
  Contrarian → Claude Sonnet 4  (best at nuanced ownership reasoning)
  Narrative  → GPT-4o mini      (good at news/injury pattern matching, cheap)
  Structure  → Claude Sonnet 4  (strong at correlation/math analysis)
  Consensus  → Claude Sonnet 4  (synthesizes all verdicts)

If a provider is unavailable, falls back to the next available.
If only one provider is available, all agents use it (same as current).

Cost: ~$0.05/slate total (same as single-model panel)

Usage:
    from agent_swarm import SwarmPanel

    panel = SwarmPanel(slate_context)
    ranked = panel.review(lineups, results)
"""

import json
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from agent_panel import (
    AgentPanel, SlateContext,
    SYSTEM_PROMPT, CONTRARIAN_ROLE, NARRATIVE_ROLE, STRUCTURE_ROLE,
)
from model_factory import ModelFactory, BaseModel


# ═══════════════════════════════════════════════════════════════
# Agent → Model Routing
# ═══════════════════════════════════════════════════════════════

DEFAULT_ROUTING = {
    'Contrarian': 'anthropic',   # Best at nuanced ownership reasoning
    'Narrative':  'openai',      # Good at news/injury pattern matching
    'Structure':  'anthropic',   # Strong at correlation/structural analysis
}

CONSENSUS_PROVIDER = 'anthropic'  # Final synthesis always uses Claude

CONSENSUS_PROMPT = """You are the Consensus Agent for an NHL DFS lineup review panel.

You have received independent reviews from 3 specialist agents, each using a DIFFERENT
AI model to eliminate shared blind spots:

- Contrarian Agent (Claude): Evaluated ownership leverage and field differentiation
- Narrative Agent (GPT): Evaluated injuries, B2B, news, qualitative factors
- Structure Agent (Claude): Evaluated stack quality, goalie matchup, correlations

Your job: Synthesize their verdicts into a final ranking of the top 20 lineups.

Scoring rules:
- Each lineup starts with points = 21 - sim_rank (rank 1 = 20 pts, rank 20 = 1 pt)
- APPROVE: +0 points
- FLAG: -5 points
- VETO: -15 points (only valid if 2+ agents agree)
- BOOST: +3 points

CRITICAL: A single agent's VETO is downgraded to FLAG unless another agent also
flagged/vetoed the same lineup. This prevents one model's quirk from overriding the math.

Respond in JSON:
{
  "final_ranking": [
    {"lineup_rank": 1, "final_score": 20, "key_factors": "brief reason"},
    ...
  ],
  "top_pick": 1,
  "top_pick_reasoning": "why this lineup wins after all agent input",
  "dissent_noted": "any significant disagreement between agents"
}"""


class SwarmPanel(AgentPanel):
    """
    Multi-model agent panel. Each specialist agent runs on a different LLM.

    Inherits all lineup formatting and context from AgentPanel.
    Overrides the review method to route agents to different providers.
    """

    def __init__(
        self,
        slate_context: SlateContext,
        routing: Dict[str, str] = None,
        verbose: bool = True,
    ):
        # Don't need a single api_key — each model has its own
        super().__init__(slate_context, api_key='swarm')
        self.routing = routing or DEFAULT_ROUTING
        self.verbose = verbose

        # Check which providers are available
        available = ModelFactory.available()
        self.models: Dict[str, BaseModel] = {}

        # Build fallback chain: anthropic → openai → deepseek
        fallback_order = ['anthropic', 'openai', 'deepseek']
        fallbacks = [p for p in fallback_order if available.get(p)]

        if not fallbacks:
            raise RuntimeError("No LLM API keys found. Set at least ANTHROPIC_API_KEY in .env")

        for agent_name, preferred_provider in self.routing.items():
            if available.get(preferred_provider):
                self.models[agent_name] = ModelFactory.create(preferred_provider)
            else:
                # Fall back to first available
                fb = fallbacks[0]
                if verbose:
                    print(f"  ⚠ {agent_name}: {preferred_provider} unavailable, using {fb}")
                self.models[agent_name] = ModelFactory.create(fb)

        # Consensus model
        if available.get(CONSENSUS_PROVIDER):
            self.consensus_model = ModelFactory.create(CONSENSUS_PROVIDER)
        else:
            self.consensus_model = ModelFactory.create(fallbacks[0])

        if verbose:
            print(f"\n  ╔══ SWARM PANEL CONFIGURATION ═══════════════════╗")
            for agent_name, model in self.models.items():
                print(f"  ║  {agent_name:<12} → {model.provider:<10} ({model.model_name})")
            print(f"  ║  {'Consensus':<12} → {self.consensus_model.provider:<10} ({self.consensus_model.model_name})")
            n_unique = len(set(m.provider for m in self.models.values()))
            print(f"  ║  Model diversity: {n_unique} unique providers")
            print(f"  ╚════════════════════════════════════════════════╝")

    def _call_agent_swarm(
        self,
        agent_name: str,
        agent_role: str,
        lineup_text: str,
    ) -> Optional[Dict]:
        """Call the appropriate model for this agent."""
        model = self.models.get(agent_name)
        if not model:
            print(f"    ⚠ No model for {agent_name}")
            return None

        system = SYSTEM_PROMPT.format(agent_name=agent_name, agent_role=agent_role)
        user = f"{self.context.to_text()}\n\n{lineup_text}"

        t0 = time.time()
        result = model.generate_json(system, user)
        elapsed = time.time() - t0

        if result and self.verbose:
            print(f"    {agent_name} ({model.provider}): responded in {elapsed:.1f}s")

        return result

    def review(
        self,
        lineups: List[pd.DataFrame],
        results: List[Dict],
        n_review: int = 20,
        verbose: bool = True,
    ) -> List[Tuple[pd.DataFrame, Dict, float]]:
        """
        Run multi-model swarm review.

        Each agent runs on its assigned model provider independently.
        Consensus model synthesizes all verdicts.
        """
        n = min(n_review, len(lineups))
        lineup_text = self._format_lineups_for_agents(lineups[:n], results[:n])

        ADJUSTMENTS = {'APPROVE': 0, 'FLAG': -5, 'VETO': -15, 'BOOST': +3}
        consensus = [n - i for i in range(n)]  # sim rank scoring

        agent_reviews = {}
        all_verdicts = {}  # track per-lineup verdicts for VETO validation

        if verbose:
            print(f"\n  ── Swarm Agent Panel ({n} lineups) ──")

        # Phase 1: Run each specialist on its own model
        for agent_name, agent_role in self.AGENTS:
            if verbose:
                model = self.models.get(agent_name)
                provider = model.provider if model else '?'
                print(f"  Calling {agent_name} ({provider})...")

            response = self._call_agent_swarm(agent_name, agent_role, lineup_text)
            if response and 'reviews' in response:
                agent_reviews[agent_name] = response

                for review in response['reviews']:
                    rank = review.get('lineup_rank', 0) - 1
                    if 0 <= rank < n:
                        verdict = review.get('verdict', 'APPROVE').upper()

                        # Track verdicts per lineup for VETO validation
                        if rank not in all_verdicts:
                            all_verdicts[rank] = {}
                        all_verdicts[rank][agent_name] = verdict

        # Phase 2: Apply scoring with VETO validation
        # Single-agent VETOs are downgraded to FLAGs (prevents one model's quirk)
        for rank in range(n):
            verdicts = all_verdicts.get(rank, {})
            n_negative = sum(1 for v in verdicts.values() if v in ('VETO', 'FLAG'))

            for agent_name, verdict in verdicts.items():
                if verdict == 'VETO' and n_negative < 2:
                    # Single VETO → downgrade to FLAG
                    adj = ADJUSTMENTS['FLAG']
                    if verbose:
                        print(f"    #{rank+1}: {agent_name} VETO → FLAG (single agent, need 2+)")
                else:
                    adj = ADJUSTMENTS.get(verdict, 0)

                consensus[rank] += adj

                if verbose and verdict != 'APPROVE':
                    review = next(
                        (r for r in agent_reviews.get(agent_name, {}).get('reviews', [])
                         if r.get('lineup_rank', 0) - 1 == rank),
                        {}
                    )
                    reason = review.get('reason', '?')
                    conf = review.get('confidence', 0)
                    model = self.models.get(agent_name)
                    provider = model.provider if model else '?'
                    if verdict != 'VETO' or n_negative >= 2:  # Don't double-print downgraded
                        print(f"    #{rank+1}: {verdict} by {agent_name} ({provider}, "
                              f"{conf:.0%}) — {reason}")

        # Phase 3: Optional consensus synthesis
        # If we have multiple agent reviews, ask consensus model to synthesize
        if len(agent_reviews) >= 2 and self.consensus_model:
            if verbose:
                print(f"  Running Consensus ({self.consensus_model.provider})...")

            reviews_summary = json.dumps({
                name: {
                    'top_3_picks': r.get('top_3_picks', []),
                    'reasoning': r.get('reasoning', ''),
                    'reviews': [
                        {'rank': rev.get('lineup_rank'), 'verdict': rev.get('verdict'),
                         'reason': rev.get('reason', '')}
                        for rev in r.get('reviews', [])
                        if rev.get('verdict', 'APPROVE') != 'APPROVE'
                    ]
                }
                for name, r in agent_reviews.items()
            }, indent=2)

            consensus_input = (
                f"Agent reviews:\n{reviews_summary}\n\n"
                f"Current consensus scores (after adjustments):\n"
                + '\n'.join(f"  Lineup #{i+1}: {consensus[i]} pts" for i in range(min(10, n)))
            )

            consensus_response = self.consensus_model.generate_json(
                CONSENSUS_PROMPT, consensus_input
            )
            if consensus_response and verbose:
                top = consensus_response.get('top_pick_reasoning', '')
                dissent = consensus_response.get('dissent_noted', '')
                if top:
                    print(f"    Consensus top pick reasoning: {top[:200]}")
                if dissent:
                    print(f"    Dissent: {dissent[:200]}")

        # Build final ranked list
        ranked = []
        for i in range(n):
            ranked.append((lineups[i], results[i], consensus[i]))
        ranked.sort(key=lambda x: x[2], reverse=True)

        if verbose:
            print(f"\n  ── Final Swarm Consensus ──")
            print(f"  {'Rank':>4} {'SimRank':>8} {'Score':>6} {'Mean':>6} {'Std':>5} {'M+3σ':>6}")
            print(f"  {'-' * 40}")
            for final_rank, (lu, r, score) in enumerate(ranked[:10], 1):
                orig_rank = results.index(r) + 1 if r in results else '?'
                m3s = r.get('m3s', r['mean'] + 3 * r['std'])
                changed = ' ←' if final_rank != orig_rank else ''
                print(f"  {final_rank:>4} {orig_rank:>8} {score:>6.0f} "
                      f"{r['mean']:>6.1f} {r['std']:>5.1f} {m3s:>6.0f}{changed}")

        return ranked


# ═══════════════════════════════════════════════════════════════
# Convenience function (drop-in replacement for run_agent_panel)
# ═══════════════════════════════════════════════════════════════

def run_swarm_panel(
    lineups: List[pd.DataFrame],
    results: List[Dict],
    player_pool: pd.DataFrame,
    stack_builder=None,
    vegas_games=None,
    injuries_df=None,
    contest_type: str = 'se_gpp',
    routing: Dict[str, str] = None,
    verbose: bool = True,
) -> List[Tuple[pd.DataFrame, Dict, float]]:
    """
    One-call convenience function to run the multi-model swarm panel.

    Drop-in replacement for run_agent_panel() from agent_panel.py.
    """
    ctx = SlateContext.from_pipeline(player_pool, stack_builder, injuries_df, vegas_games)
    ctx.contest_type = contest_type

    try:
        panel = SwarmPanel(ctx, routing=routing, verbose=verbose)
        return panel.review(lineups, results, verbose=verbose)
    except RuntimeError as e:
        # No API keys — fall back to original panel
        print(f"  ⚠ Swarm unavailable ({e}), falling back to single-model panel")
        from agent_panel import run_agent_panel
        return run_agent_panel(
            lineups, results, player_pool,
            stack_builder, vegas_games, injuries_df,
            contest_type, verbose=verbose,
        )


if __name__ == '__main__':
    """Test model availability and routing."""
    from model_factory import ModelFactory

    print("Multi-Model Swarm — Status Check")
    print("=" * 50)

    available = ModelFactory.available()
    for provider, has_key in available.items():
        icon = "✅" if has_key else "❌"
        print(f"  {icon} {provider}")

    n_available = sum(1 for v in available.values() if v)
    print(f"\n  {n_available}/3 providers available")

    if n_available == 0:
        print("\n  ❌ No API keys found. Add to .env:")
        print("    ANTHROPIC_API_KEY=sk-ant-...")
        print("    OPENAI_API_KEY=sk-...")
        print("    DEEPSEEK_API_KEY=sk-...")
    elif n_available == 1:
        provider = [k for k, v in available.items() if v][0]
        print(f"\n  ⚠ Single model mode — all agents use {provider}")
        print("  Add more API keys for true model diversity.")
    else:
        print(f"\n  ✅ Multi-model swarm ready!")
        print("  Agent routing:")
        for agent, provider in DEFAULT_ROUTING.items():
            actual = provider if available.get(provider) else [k for k, v in available.items() if v][0]
            fb = " (fallback)" if actual != provider else ""
            print(f"    {agent:<12} → {actual}{fb}")
