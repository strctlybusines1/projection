"""
Export all ranked single-entry candidate lineups to JSON for the website.

Called from main.py after SE selection. Saves all 40 (or N) candidates
with their scores and player details.

Usage from main.py:
    from lineup_export import export_se_candidates
    export_se_candidates(scored_candidates, date_str, output_dir)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd


def export_se_candidates(
    scored: List[Tuple[pd.DataFrame, Dict[str, float]]],
    date_str: str,
    output_dir: str = "daily_projections",
    actuals: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Export all ranked SE candidate lineups to a JSON file.

    Args:
        scored: List of (lineup_df, scores_dict) tuples, sorted by total score desc
        date_str: Slate date as YYYY-MM-DD
        output_dir: Directory to save to
        actuals: Optional DataFrame with actual_fpts column for post-slate grading

    Returns:
        Path to saved JSON file
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = datetime.strptime(date_str, '%Y-%m-%d')
    filename = f"{dt.strftime('%m_%d_%y')}NHL_se_lineups.json"
    filepath = out_dir / filename

    # Build actual FPTS lookup if available
    actual_lookup = {}
    if actuals is not None and not actuals.empty and 'actual_fpts' in actuals.columns:
        for _, row in actuals.iterrows():
            actual_lookup[row['name']] = float(row['actual_fpts'])

    lineups_data = []
    for rank, (lineup, scores) in enumerate(scored, 1):
        players = []
        for _, p in lineup.iterrows():
            player = {
                'name': p.get('name', ''),
                'team': p.get('team', ''),
                'position': p.get('position', ''),
                'roster_slot': p.get('roster_slot', ''),
                'salary': int(p.get('salary', 0)),
                'projected_fpts': round(float(p.get('projected_fpts', 0)), 1),
                'floor': round(float(p.get('floor', 0)), 1) if pd.notna(p.get('floor')) else None,
                'ceiling': round(float(p.get('ceiling', 0)), 1) if pd.notna(p.get('ceiling')) else None,
                'predicted_ownership': round(float(p.get('predicted_ownership', 0)), 1) if pd.notna(p.get('predicted_ownership')) else None,
            }
            # Add actual if available
            if actual_lookup:
                player['actual_fpts'] = actual_lookup.get(p.get('name'), None)
            players.append(player)

        # Compute lineup totals
        total_salary = sum(p['salary'] for p in players)
        total_proj = sum(p['projected_fpts'] for p in players)
        total_own = sum(p['predicted_ownership'] or 0 for p in players)
        total_actual = None
        if actual_lookup:
            actuals_found = [p['actual_fpts'] for p in players if p.get('actual_fpts') is not None]
            if len(actuals_found) == len(players):
                total_actual = round(sum(actuals_found), 1)

        # Stack summary
        from collections import Counter
        skater_teams = [p['team'] for p in players if p['position'] != 'G']
        team_counts = Counter(skater_teams)
        stacks = [f"{team}{count}" for team, count in team_counts.most_common() if count >= 2]

        # Goalie name
        goalie = next((p for p in players if p['position'] == 'G'), None)

        lineups_data.append({
            'rank': rank,
            'players': players,
            'total_salary': total_salary,
            'total_projected': round(total_proj, 1),
            'total_ownership': round(total_own, 1),
            'total_actual': total_actual,
            'goalie': goalie['name'] if goalie else None,
            'stacks': stacks,
            'scores': {k: round(v, 4) for k, v in scores.items()},
        })

    output = {
        'slate_date': date_str,
        'generated_at': datetime.now().isoformat(),
        'n_candidates': len(lineups_data),
        'lineups': lineups_data,
    }

    filepath.write_text(json.dumps(output, indent=2))
    print(f"  SE lineup details exported to: {filepath}")
    return filepath


def load_se_lineups(date_str: str, projections_dir: str = "daily_projections") -> Optional[Dict]:
    """Load SE lineup JSON for a given date."""
    proj_dir = Path(projections_dir)
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    filename = f"{dt.strftime('%m_%d_%y')}NHL_se_lineups.json"
    filepath = proj_dir / filename

    if filepath.exists():
        return json.loads(filepath.read_text())
    return None
