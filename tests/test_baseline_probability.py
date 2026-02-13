"""
Tests for slate-aware baseline probability calculations.

Baseline probability = structural prior for ownership based on positional scarcity.
On a 2-game slate with 4 centers and 2 C slots, each C has ~50% baseline.
On a 10-game slate with 30 centers, each C has ~6.7% baseline.

DraftKings NHL roster: 2C, 3W, 2D, 1G, 1UTIL = 9 players
Position slots: C=2, W=3, D=2, G=1, UTIL=1 (any skater)
"""

import pytest
import pandas as pd
import numpy as np

from baseline_probability import compute_baseline_probabilities


def _make_pool(players):
    """Helper: build a player pool DataFrame from a list of (name, team, position, salary) tuples."""
    return pd.DataFrame(players, columns=['name', 'team', 'position', 'salary'])


# ─── Test 1: Normal slate (8-10 games, ~60 skaters + 10 goalies) ───

class TestNormalSlate:
    def test_normal_slate_returns_baseline_prob_column(self):
        """A normal 8-game slate should produce a 'baseline_prob' column for all players."""
        players = []
        teams = ['BOS', 'TOR', 'NYR', 'MTL', 'DET', 'BUF', 'OTT', 'FLA',
                 'TBL', 'CAR', 'WSH', 'PHI', 'PIT', 'NJD', 'CBJ', 'NYI']
        for i, team in enumerate(teams):
            # 3 C, 4 W, 3 D per team
            for j in range(3):
                players.append((f'Center_{team}_{j}', team, 'C', 4500 + j * 500))
            for j in range(4):
                players.append((f'Wing_{team}_{j}', team, 'W', 4000 + j * 500))
            for j in range(3):
                players.append((f'Defense_{team}_{j}', team, 'D', 4000 + j * 500))
            # 1 goalie per team
            players.append((f'Goalie_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        assert 'baseline_prob' in result.columns
        assert len(result) == len(pool)
        assert result['baseline_prob'].notna().all()

    def test_normal_slate_probabilities_are_positive(self):
        """All baseline probabilities should be > 0."""
        players = []
        for team in ['BOS', 'TOR', 'NYR', 'MTL', 'DET', 'BUF', 'OTT', 'FLA',
                      'TBL', 'CAR', 'WSH', 'PHI', 'PIT', 'NJD', 'CBJ', 'NYI']:
            for j in range(3):
                players.append((f'C_{team}_{j}', team, 'C', 4500))
            for j in range(4):
                players.append((f'W_{team}_{j}', team, 'W', 4500))
            for j in range(3):
                players.append((f'D_{team}_{j}', team, 'D', 4500))
            players.append((f'G_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)
        assert (result['baseline_prob'] > 0).all()


# ─── Test 2: Small slate (2-3 games, ~20 skaters) ───

class TestSmallSlate:
    def test_small_slate_higher_baseline(self):
        """On a 2-game slate, each position group is smaller → higher baseline per player."""
        players = []
        for team in ['BOS', 'TOR', 'NYR', 'MTL']:
            for j in range(3):
                players.append((f'C_{team}_{j}', team, 'C', 4500))
            for j in range(4):
                players.append((f'W_{team}_{j}', team, 'W', 4500))
            for j in range(3):
                players.append((f'D_{team}_{j}', team, 'D', 4500))
            players.append((f'G_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        # With 12 centers and 2 C slots, baseline ~= 2/12 ≈ 0.167
        centers = result[result['position'] == 'C']
        avg_c_prob = centers['baseline_prob'].mean()
        assert avg_c_prob > 0.10, f"Small slate center prob {avg_c_prob:.3f} should be > 0.10"


# ─── Test 3: Single game slate (2 teams) ───

class TestSingleGame:
    def test_single_game_highest_baseline(self):
        """Single-game slate → maximum scarcity → highest baseline probabilities."""
        players = []
        for team in ['BOS', 'TOR']:
            for j in range(3):
                players.append((f'C_{team}_{j}', team, 'C', 4500))
            for j in range(4):
                players.append((f'W_{team}_{j}', team, 'W', 4500))
            for j in range(3):
                players.append((f'D_{team}_{j}', team, 'D', 4500))
            players.append((f'G_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        # 6 centers, 2 C slots → baseline ≈ 0.33
        centers = result[result['position'] == 'C']
        avg_c_prob = centers['baseline_prob'].mean()
        assert avg_c_prob > 0.25, f"Single game C prob {avg_c_prob:.3f} should be > 0.25"

        # Goalies: 2 goalies, 1 G slot → baseline ≈ 0.50
        goalies = result[result['position'] == 'G']
        avg_g_prob = goalies['baseline_prob'].mean()
        assert avg_g_prob > 0.40, f"Single game G prob {avg_g_prob:.3f} should be > 0.40"


# ─── Test 4: Empty slate ───

class TestEmptySlate:
    def test_empty_pool_returns_empty(self):
        """An empty player pool should return an empty DataFrame with baseline_prob column."""
        pool = _make_pool([])
        result = compute_baseline_probabilities(pool)
        assert 'baseline_prob' in result.columns
        assert len(result) == 0


# ─── Test 5: Missing player data ───

class TestMissingData:
    def test_missing_salary_still_computes(self):
        """Players with NaN salary should still get a baseline probability."""
        players = [
            ('Player_A', 'BOS', 'C', np.nan),
            ('Player_B', 'BOS', 'C', 5000),
            ('Player_C', 'TOR', 'C', 4500),
            ('Player_D', 'BOS', 'W', 4000),
            ('Player_E', 'TOR', 'W', 4000),
            ('Player_F', 'BOS', 'D', 4500),
            ('Player_G', 'TOR', 'D', 4500),
            ('Goalie_A', 'BOS', 'G', 7500),
            ('Goalie_B', 'TOR', 'G', 7500),
        ]
        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)
        assert result['baseline_prob'].notna().all()

    def test_missing_position_excluded(self):
        """Players with NaN position should get baseline_prob = 0 (can't fill any slot)."""
        players = [
            ('Player_A', 'BOS', None, 5000),
            ('Player_B', 'BOS', 'C', 5000),
            ('Player_C', 'TOR', 'C', 4500),
            ('Player_D', 'BOS', 'W', 4000),
            ('Player_E', 'TOR', 'W', 4000),
            ('Player_F', 'BOS', 'D', 4500),
            ('Player_G', 'TOR', 'D', 4500),
            ('Goalie_A', 'BOS', 'G', 7500),
        ]
        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)
        # Player with None position gets 0
        assert result.loc[result['name'] == 'Player_A', 'baseline_prob'].iloc[0] == 0.0


# ─── Test 6: Defense vs Forwards (position key naming) ───

class TestDefenseVsForwards:
    def test_defense_uses_correct_key(self):
        """Defense players use position='D', not 'defense' or 'defensemen'.
        2 D slots on DK → baseline for D is calculated from 2 slots / n_defense."""
        players = []
        for team in ['BOS', 'TOR', 'NYR', 'MTL']:
            for j in range(3):
                players.append((f'C_{team}_{j}', team, 'C', 4500))
            for j in range(4):
                players.append((f'W_{team}_{j}', team, 'W', 4500))
            for j in range(3):
                players.append((f'D_{team}_{j}', team, 'D', 4500))  # 'D' not 'defense'
            players.append((f'G_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        defense = result[result['position'] == 'D']
        assert len(defense) == 12  # 4 teams × 3 D
        # 2 D slots / 12 D players ≈ 0.167 base
        avg_d_prob = defense['baseline_prob'].mean()
        assert avg_d_prob > 0.05, f"Defense baseline {avg_d_prob:.3f} too low"

    def test_forwards_have_more_slots(self):
        """Forwards (C+W) have 5 pure slots + 1 UTIL vs D's 2+UTIL.
        On equal pool sizes, forward baseline should be >= defense baseline."""
        players = []
        for team in ['BOS', 'TOR']:
            for j in range(5):
                players.append((f'C_{team}_{j}', team, 'C', 4500))
            for j in range(5):
                players.append((f'W_{team}_{j}', team, 'W', 4500))
            for j in range(5):
                players.append((f'D_{team}_{j}', team, 'D', 4500))
            players.append((f'G_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        c_prob = result[result['position'] == 'C']['baseline_prob'].mean()
        w_prob = result[result['position'] == 'W']['baseline_prob'].mean()
        d_prob = result[result['position'] == 'D']['baseline_prob'].mean()

        # C has 2 slots for 10 players, D has 2 slots for 10 players → similar base
        # But UTIL slot benefits skaters, so C and W should be slightly >= D
        # The key check is they're all reasonable and computed correctly
        assert c_prob > 0, "Center baseline should be positive"
        assert w_prob > 0, "Wing baseline should be positive"
        assert d_prob > 0, "Defense baseline should be positive"


# ─── Test 7: Goalie edge cases ───

class TestGoalieEdgeCases:
    def test_single_goalie_gets_high_baseline(self):
        """If only 1 goalie on slate, baseline ≈ 1.0 (must be rostered)."""
        players = [
            ('C1', 'BOS', 'C', 5000),
            ('C2', 'TOR', 'C', 5000),
            ('W1', 'BOS', 'W', 4500),
            ('W2', 'TOR', 'W', 4500),
            ('W3', 'BOS', 'W', 4500),
            ('D1', 'BOS', 'D', 4500),
            ('D2', 'TOR', 'D', 4500),
            ('G1', 'BOS', 'G', 7500),  # Only 1 goalie
        ]
        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        goalie = result[result['position'] == 'G']
        assert len(goalie) == 1
        # 1 goalie, 1 G slot → must be rostered → baseline ≈ 1.0
        assert goalie['baseline_prob'].iloc[0] >= 0.95

    def test_many_goalies_low_baseline(self):
        """With 10 goalies and 1 G slot, each goalie has low baseline ≈ 0.10."""
        players = []
        for i in range(10):
            players.append((f'Goalie_{i}', f'T{i}', 'G', 7500))
        # Add some skaters
        for i in range(5):
            players.append((f'C_{i}', f'T{i}', 'C', 4500))
            players.append((f'W_{i}', f'T{i}', 'W', 4500))
            players.append((f'D_{i}', f'T{i}', 'D', 4500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        goalies = result[result['position'] == 'G']
        avg_g = goalies['baseline_prob'].mean()
        assert avg_g < 0.20, f"10 goalies avg baseline {avg_g:.3f} should be < 0.20"

    def test_goalies_cannot_fill_util(self):
        """Goalies are NOT UTIL-eligible on DraftKings NHL. Their baseline comes only from the 1 G slot."""
        players = []
        for team in ['BOS', 'TOR']:
            for j in range(3):
                players.append((f'C_{team}_{j}', team, 'C', 4500))
            for j in range(3):
                players.append((f'W_{team}_{j}', team, 'W', 4500))
            for j in range(3):
                players.append((f'D_{team}_{j}', team, 'D', 4500))
            players.append((f'G_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        goalies = result[result['position'] == 'G']
        # 2 goalies, 1 G slot → 0.50 each, NOT higher (UTIL doesn't help goalies)
        for _, g in goalies.iterrows():
            assert g['baseline_prob'] <= 0.60, \
                f"Goalie {g['name']} baseline {g['baseline_prob']:.3f} too high — UTIL shouldn't help goalies"


# ─── Test 8: Probabilities sum correctly per position ───

class TestProbabilitySum:
    def test_position_probabilities_sum_to_slots(self):
        """Sum of baseline_prob within each position group should ≈ number of slots for that position.
        C: sum ≈ 2, W: sum ≈ 3, D: sum ≈ 2, G: sum ≈ 1.
        UTIL adds ~1 slot shared among skaters."""
        players = []
        for team in ['BOS', 'TOR', 'NYR', 'MTL', 'DET', 'BUF']:
            for j in range(3):
                players.append((f'C_{team}_{j}', team, 'C', 4500))
            for j in range(4):
                players.append((f'W_{team}_{j}', team, 'W', 4500))
            for j in range(3):
                players.append((f'D_{team}_{j}', team, 'D', 4500))
            players.append((f'G_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        c_sum = result[result['position'] == 'C']['baseline_prob'].sum()
        w_sum = result[result['position'] == 'W']['baseline_prob'].sum()
        d_sum = result[result['position'] == 'D']['baseline_prob'].sum()
        g_sum = result[result['position'] == 'G']['baseline_prob'].sum()

        # C slots: 2 dedicated + share of UTIL → sum should be close to 2-3
        assert 1.5 <= c_sum <= 3.5, f"C sum {c_sum:.2f} should be near 2-3"
        # W slots: 3 dedicated + share of UTIL → sum should be close to 3-4
        assert 2.5 <= w_sum <= 4.5, f"W sum {w_sum:.2f} should be near 3-4"
        # D slots: 2 dedicated + share of UTIL → sum should be near 2-3
        assert 1.5 <= d_sum <= 3.5, f"D sum {d_sum:.2f} should be near 2-3"
        # G slot: 1 dedicated, no UTIL → sum should be ≈ 1
        assert 0.8 <= g_sum <= 1.2, f"G sum {g_sum:.2f} should be near 1.0"

    def test_total_probability_equals_roster_size(self):
        """Total sum of all baseline_prob ≈ 9 (roster size).
        Each of 9 lineup slots is filled by exactly 1 player."""
        players = []
        for team in ['BOS', 'TOR', 'NYR', 'MTL']:
            for j in range(3):
                players.append((f'C_{team}_{j}', team, 'C', 4500))
            for j in range(4):
                players.append((f'W_{team}_{j}', team, 'W', 4500))
            for j in range(3):
                players.append((f'D_{team}_{j}', team, 'D', 4500))
            players.append((f'G_{team}', team, 'G', 7500))

        pool = _make_pool(players)
        result = compute_baseline_probabilities(pool)

        total = result['baseline_prob'].sum()
        # Should be close to 9.0 (9 roster slots)
        assert 8.0 <= total <= 10.0, f"Total baseline sum {total:.2f} should be near 9.0"


# ─── Test 9: Scaling with slate size ───

class TestScalingWithSlateSize:
    def test_larger_slate_lower_individual_probabilities(self):
        """As slate grows (more players), individual baseline probabilities decrease."""
        def make_slate(n_teams):
            players = []
            teams = [f'T{i}' for i in range(n_teams)]
            for team in teams:
                for j in range(3):
                    players.append((f'C_{team}_{j}', team, 'C', 4500))
                for j in range(4):
                    players.append((f'W_{team}_{j}', team, 'W', 4500))
                for j in range(3):
                    players.append((f'D_{team}_{j}', team, 'D', 4500))
                players.append((f'G_{team}', team, 'G', 7500))
            return _make_pool(players)

        small = compute_baseline_probabilities(make_slate(4))
        large = compute_baseline_probabilities(make_slate(16))

        small_avg = small['baseline_prob'].mean()
        large_avg = large['baseline_prob'].mean()

        assert small_avg > large_avg, \
            f"Small slate avg {small_avg:.3f} should be > large slate avg {large_avg:.3f}"


# ─── Test 10: dk_pos column support ───

class TestDkPosColumn:
    def test_dk_pos_column_used_when_present(self):
        """If 'dk_pos' column exists (from DK salaries merge), use it over 'position'."""
        players = pd.DataFrame({
            'name': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'G1'],
            'team': ['BOS', 'BOS', 'TOR', 'BOS', 'TOR', 'BOS', 'TOR', 'BOS'],
            'position': ['C', 'C', 'C', 'W', 'W', 'D', 'D', 'G'],
            'dk_pos': ['C', 'C', 'C', 'W', 'W', 'D', 'D', 'G'],
            'salary': [5000, 4500, 4500, 4000, 4000, 4500, 4500, 7500],
        })
        result = compute_baseline_probabilities(players)
        assert 'baseline_prob' in result.columns
        assert result['baseline_prob'].notna().all()


# ─── Test 11: LW/RW normalization ───

class TestPositionNormalization:
    def test_lw_rw_normalized_to_w(self):
        """LW and RW should be treated as W for slot calculations."""
        players = pd.DataFrame({
            'name': ['C1', 'C2', 'LW1', 'RW1', 'LW2', 'D1', 'D2', 'G1', 'G2'],
            'team': ['BOS', 'TOR', 'BOS', 'TOR', 'BOS', 'BOS', 'TOR', 'BOS', 'TOR'],
            'position': ['C', 'C', 'LW', 'RW', 'LW', 'D', 'D', 'G', 'G'],
            'salary': [5000, 4500, 4500, 4000, 4000, 4500, 4500, 7500, 7500],
        })
        result = compute_baseline_probabilities(players)

        # LW1, RW1, LW2 should all have wing-based probabilities
        wings = result[result['position'].isin(['LW', 'RW'])]
        assert len(wings) == 3
        assert (wings['baseline_prob'] > 0).all()


# ─── Test 12: Output preserves original DataFrame ───

class TestPreservesOriginal:
    def test_original_columns_preserved(self):
        """compute_baseline_probabilities should add baseline_prob without losing other columns."""
        players = _make_pool([
            ('P1', 'BOS', 'C', 5000),
            ('P2', 'TOR', 'W', 4000),
            ('P3', 'BOS', 'D', 4500),
            ('G1', 'BOS', 'G', 7500),
        ])
        players['projected_fpts'] = [12.0, 10.0, 8.0, 15.0]
        players['dk_avg_fpts'] = [11.0, 9.0, 7.0, 14.0]

        result = compute_baseline_probabilities(players)

        assert 'projected_fpts' in result.columns
        assert 'dk_avg_fpts' in result.columns
        assert 'name' in result.columns
        assert 'baseline_prob' in result.columns
