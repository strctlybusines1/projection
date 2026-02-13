"""
Integration test: baseline probability flows through the mini-pipeline.

Simulates the main.py flow: build pool → compute baseline → verify column
is present and sane before ownership prediction.
"""

import pandas as pd
import numpy as np

from baseline_probability import compute_baseline_probabilities


def test_end_to_end_pipeline():
    """Simulate a realistic mini-pipeline: pool → baseline → ownership-ready.

    Verifies:
    1. baseline_prob column is added
    2. All values are in [0, 1]
    3. Total sum ≈ 9 (roster slots)
    4. Original columns preserved
    5. Goalies get lower baseline than scarce skater positions
    """
    # Build a realistic 6-game slate pool
    players = []
    teams = ['BOS', 'TOR', 'NYR', 'MTL', 'DET', 'BUF',
             'OTT', 'FLA', 'TBL', 'CAR', 'WSH', 'PHI']

    for team in teams:
        # Typical DK pool: ~3 C, 4 W, 3 D, 1 G per team
        for j in range(3):
            players.append({
                'name': f'Center_{team}_{j}', 'team': team, 'position': 'C',
                'salary': 4500 + j * 500,
                'projected_fpts': 8.0 + np.random.uniform(-2, 4),
                'dk_avg_fpts': 7.5 + np.random.uniform(-2, 4),
            })
        for j in range(4):
            players.append({
                'name': f'Wing_{team}_{j}', 'team': team, 'position': 'W',
                'salary': 4000 + j * 500,
                'projected_fpts': 7.0 + np.random.uniform(-2, 4),
                'dk_avg_fpts': 6.5 + np.random.uniform(-2, 4),
            })
        for j in range(3):
            players.append({
                'name': f'Defense_{team}_{j}', 'team': team, 'position': 'D',
                'salary': 4000 + j * 500,
                'projected_fpts': 5.0 + np.random.uniform(-1, 3),
                'dk_avg_fpts': 4.5 + np.random.uniform(-1, 3),
            })
        players.append({
            'name': f'Goalie_{team}', 'team': team, 'position': 'G',
            'salary': 7500,
            'projected_fpts': 15.0 + np.random.uniform(-3, 5),
            'dk_avg_fpts': 14.0 + np.random.uniform(-3, 5),
        })

    pool = pd.DataFrame(players)

    # Add derived columns (mimicking main.py pipeline)
    pool['value'] = pool['projected_fpts'] / (pool['salary'] / 1000)
    pool['edge'] = pool['projected_fpts'] - pool['dk_avg_fpts']
    pool['floor'] = pool['projected_fpts'] * 0.3
    pool['ceiling'] = pool['projected_fpts'] * 2.5 + 5

    # Run baseline probability
    result = compute_baseline_probabilities(pool)

    # 1. Column exists
    assert 'baseline_prob' in result.columns

    # 2. All values in [0, 1]
    assert (result['baseline_prob'] >= 0).all()
    assert (result['baseline_prob'] <= 1.0).all()

    # 3. Total sum ≈ 9 (roster slots)
    total = result['baseline_prob'].sum()
    assert 8.0 <= total <= 10.0, f"Total {total:.2f} not near 9"

    # 4. Original columns preserved
    for col in ['name', 'team', 'position', 'salary', 'projected_fpts',
                'dk_avg_fpts', 'value', 'edge', 'floor', 'ceiling']:
        assert col in result.columns

    # 5. Goalies get lower aggregate baseline (1 slot) vs C (2 slots)
    g_sum = result[result['position'] == 'G']['baseline_prob'].sum()
    c_sum = result[result['position'] == 'C']['baseline_prob'].sum()
    assert g_sum < c_sum, f"Goalie sum {g_sum:.2f} should be < Center sum {c_sum:.2f}"

    # 6. Length preserved
    assert len(result) == len(pool)
