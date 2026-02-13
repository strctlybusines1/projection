"""
Tests for NHL DFS Lineup Optimizer.

NOTE: NHLLineupOptimizer uses generate_gpp_lineups(), not generate_lineups().
"""
import pytest
import pandas as pd
import numpy as np


class TestOptimizerBasics:
    """Basic optimizer functionality tests."""

    def test_optimizer_initializes_without_error(self, sample_player_pool):
        """Optimizer should initialize successfully."""
        from optimizer import NHLLineupOptimizer

        pool = sample_player_pool.copy()
        pool['dk_pos'] = pool['position'].replace({'LW': 'W', 'RW': 'W'})

        # Should not raise
        optimizer = NHLLineupOptimizer()
        assert optimizer is not None


class TestOptimizerConstraints:
    """Tests for lineup constraints (documented behavior)."""

    def test_min_teams_excludes_goalie(self):
        """
        CRITICAL: min_teams constraint applies to SKATERS ONLY.

        Goalie is never counted toward min_teams requirement.
        """
        # This is tested in test_regressions.py with actual lineup generation
        # Here we just document the expected behavior
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
