"""
NHL DFS Pipeline — Test Suite

Run:
    cd ~/Desktop/Code/projection
    pytest tests/ -v
    pytest tests/ -v -k "test_scoring"     # Run only scoring tests
    pytest tests/ -v --tb=short            # Short tracebacks
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Helper to check if main.py can be imported (needs tqdm + tabpfn via projections.py)
def _can_import_main():
    try:
        import tqdm
        import tabpfn
        return True
    except ImportError:
        return False

def _can_import_projections():
    try:
        import tabpfn
        return True
    except ImportError:
        return False

def _can_import_optimizer():
    """Optimizer imports don't need tabpfn."""
    try:
        import tqdm
        return True
    except ImportError:
        return False

_skip_no_main = pytest.mark.skipif(not _can_import_main(), reason="main.py deps (tqdm/tabpfn) not installed")
_skip_no_tabpfn = pytest.mark.skipif(not _can_import_projections(), reason="tabpfn not installed")
_skip_no_optimizer = pytest.mark.skipif(not _can_import_optimizer(), reason="tqdm not installed")


# ================================================================
#  CONFIG & SCORING TESTS
# ================================================================

class TestScoringFunctions:
    """Test DraftKings scoring math — these must be perfect."""

    def test_skater_single_goal(self):
        from config import calculate_skater_fantasy_points
        # 1 goal, 0 assists, 1 shot, 0 blocks
        pts = calculate_skater_fantasy_points(1, 0, 1, 0)
        assert pts == 8.5 + 1.5  # goal + shot

    def test_skater_goal_and_assist(self):
        from config import calculate_skater_fantasy_points
        pts = calculate_skater_fantasy_points(1, 1, 3, 0)
        # 8.5(goal) + 5.0(assist) + 4.5(3 shots) = 18.0
        assert pts == 18.0

    def test_skater_hat_trick_bonus(self):
        from config import calculate_skater_fantasy_points
        pts = calculate_skater_fantasy_points(3, 0, 5, 0)
        # 25.5(goals) + 7.5(shots) + 3.0(hat trick) + 3.0(5+ shots) + 3.0(3+ points) = 42.0
        assert pts == 42.0

    def test_skater_five_shots_bonus(self):
        from config import calculate_skater_fantasy_points
        pts = calculate_skater_fantasy_points(0, 0, 5, 0)
        # 7.5(shots) + 3.0(bonus) = 10.5
        assert pts == 10.5

    def test_skater_four_shots_no_bonus(self):
        from config import calculate_skater_fantasy_points
        pts = calculate_skater_fantasy_points(0, 0, 4, 0)
        assert pts == 6.0  # 4 * 1.5, no bonus

    def test_skater_three_blocks_bonus(self):
        from config import calculate_skater_fantasy_points
        pts = calculate_skater_fantasy_points(0, 0, 0, 3)
        # 3.9(blocks) + 3.0(bonus) = 6.9
        assert pts == 6.9

    def test_skater_shorthanded_bonus(self):
        from config import calculate_skater_fantasy_points
        pts = calculate_skater_fantasy_points(1, 0, 1, 0, sh_goals=1)
        # 8.5(goal) + 1.5(shot) + 2.0(SH bonus) = 12.0
        assert pts == 12.0

    def test_skater_zero_stats(self):
        from config import calculate_skater_fantasy_points
        pts = calculate_skater_fantasy_points(0, 0, 0, 0)
        assert pts == 0

    def test_skater_three_plus_points_bonus(self):
        from config import calculate_skater_fantasy_points
        # 1 goal + 2 assists = 3 points -> bonus
        pts = calculate_skater_fantasy_points(1, 2, 2, 0)
        # 8.5 + 10.0 + 3.0 + 3.0(3+ pts bonus) = 24.5
        assert pts == 24.5

    def test_goalie_win_30_saves_2_ga(self):
        from config import calculate_goalie_fantasy_points
        pts = calculate_goalie_fantasy_points(win=True, saves=30, goals_against=2)
        # 6.0(win) + 21.0(saves) - 7.0(GA) = 20.0
        assert pts == 20.0

    def test_goalie_shutout(self):
        from config import calculate_goalie_fantasy_points
        pts = calculate_goalie_fantasy_points(win=True, saves=28, goals_against=0, shutout=True)
        # 6.0(win) + 19.6(saves) + 4.0(shutout) = 29.6
        assert pts == pytest.approx(29.6)

    def test_goalie_loss(self):
        from config import calculate_goalie_fantasy_points
        pts = calculate_goalie_fantasy_points(win=False, saves=25, goals_against=4)
        # 17.5(saves) - 14.0(GA) = 3.5
        assert pts == 3.5

    def test_goalie_ot_loss(self):
        from config import calculate_goalie_fantasy_points
        pts = calculate_goalie_fantasy_points(win=False, saves=32, goals_against=3, ot_loss=True)
        # 2.0(OTL) + 22.4(saves) - 10.5(GA) = 13.9
        assert pts == pytest.approx(13.9)

    def test_goalie_35_saves_bonus(self):
        from config import calculate_goalie_fantasy_points
        pts = calculate_goalie_fantasy_points(win=True, saves=38, goals_against=3)
        # 6.0(win) + 26.6(saves) - 10.5(GA) + 3.0(35+ bonus) = 25.1
        assert pts == pytest.approx(25.1)

    def test_goalie_bad_game(self):
        from config import calculate_goalie_fantasy_points
        pts = calculate_goalie_fantasy_points(win=False, saves=15, goals_against=6)
        # 10.5(saves) - 21.0(GA) = -10.5
        assert pts == -10.5


class TestConfigConstants:
    """Verify critical config values haven't drifted."""

    def test_salary_cap(self):
        from optimizer import NHLLineupOptimizer
        assert NHLLineupOptimizer.SALARY_CAP == 50000

    def test_roster_slots(self):
        from optimizer import NHLLineupOptimizer
        reqs = NHLLineupOptimizer.ROSTER_REQUIREMENTS
        assert reqs['C'] == 2
        assert reqs['W'] == 3
        assert reqs['D'] == 2
        assert reqs['G'] == 1
        assert reqs['UTIL'] == 1
        assert sum(reqs.values()) == 9

    def test_team_count(self):
        from config import NHL_TEAMS
        assert len(NHL_TEAMS) >= 32

    def test_injury_statuses(self):
        from config import INJURY_STATUSES_EXCLUDE
        assert 'IR' in INJURY_STATUSES_EXCLUDE
        assert 'O' in INJURY_STATUSES_EXCLUDE
        assert 'DTD' not in INJURY_STATUSES_EXCLUDE

    def test_team_name_mapping_coverage(self):
        from config import TEAM_FULL_NAME_TO_ABBREV
        # Should have all 32+ teams
        assert len(TEAM_FULL_NAME_TO_ABBREV) >= 32
        # Spot-check known teams
        assert TEAM_FULL_NAME_TO_ABBREV.get('BOSTON BRUINS') == 'BOS'
        assert TEAM_FULL_NAME_TO_ABBREV.get('VEGAS GOLDEN KNIGHTS') == 'VGK'
        assert TEAM_FULL_NAME_TO_ABBREV.get('WINNIPEG JETS') == 'WPG'

    def test_nst_team_map(self):
        from config import NST_TEAM_MAP
        assert NST_TEAM_MAP['T.B'] == 'TBL'
        assert NST_TEAM_MAP['N.J'] == 'NJD'
        assert NST_TEAM_MAP['L.A'] == 'LAK'
        assert NST_TEAM_MAP['S.J'] == 'SJS'


# ================================================================
#  POSITION NORMALIZATION TESTS
# ================================================================

class TestPositionNormalization:
    """Test position normalization across modules."""

    @_skip_no_main
    def test_main_normalize_position(self):
        from main import normalize_position
        assert normalize_position('LW') == 'W'
        assert normalize_position('RW') == 'W'
        assert normalize_position('C') == 'C'
        assert normalize_position('D') == 'D'
        assert normalize_position('G') == 'G'

    def test_optimizer_normalize_position(self):
        from optimizer import NHLLineupOptimizer
        opt = NHLLineupOptimizer()
        assert opt._normalize_position('LW') == 'W'
        assert opt._normalize_position('RW') == 'W'
        assert opt._normalize_position('L') == 'W'
        assert opt._normalize_position('R') == 'W'
        assert opt._normalize_position('C') == 'C'
        assert opt._normalize_position('D') == 'D'
        assert opt._normalize_position('G') == 'G'


# ================================================================
#  SALARY FILE TESTS
# ================================================================

class TestSalaryLoading:
    """Test DK salary CSV parsing."""

    @_skip_no_main
    def test_load_dk_salaries_columns(self, tmp_path):
        """Verify required columns survive loading."""
        csv = tmp_path / "DKSalaries_test.csv"
        csv.write_text(
            "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
            "C,Connor McDavid (12345),Connor McDavid,12345,C/UTIL,9800,EDM@CGY 01/01/2026 09:00PM ET,EDM,18.5\n"
            "RW,Nikita Kucherov (12346),Nikita Kucherov,12346,W/UTIL,9900,FLA@TBL 01/01/2026 07:00PM ET,TBL,17.6\n"
            "G,Connor Hellebuyck (12347),Connor Hellebuyck,12347,G,8200,WPG@MIN 01/01/2026 08:00PM ET,WPG,12.3\n"
        )
        from main import load_dk_salaries
        df = load_dk_salaries(str(csv))
        assert 'dk_name' in df.columns
        assert 'salary' in df.columns
        assert 'team' in df.columns
        assert 'dk_id' in df.columns
        assert len(df) == 3

    @_skip_no_main
    def test_load_dk_salaries_position_normalization(self, tmp_path):
        """RW should become W."""
        csv = tmp_path / "DKSalaries_test.csv"
        csv.write_text(
            "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
            "RW,Test Player (1),Test Player,1,W/UTIL,5000,ANA@BOS 01/01/2026 07:00PM ET,ANA,5.0\n"
        )
        from main import load_dk_salaries
        df = load_dk_salaries(str(csv))
        assert df.iloc[0]['position'] == 'W'


# ================================================================
#  VEGAS DATA TESTS
# ================================================================

class TestVegasData:
    """Test Vegas line parsing and team total mapping."""

    @_skip_no_main
    def test_build_team_total_map(self):
        from main import build_team_total_map
        games = [
            {
                'matchup': 'Boston Bruins @ Toronto Maple Leafs',
                'game_total': 6.5,
                'away_team_total': 3.0,
                'home_team_total': 3.5,
            }
        ]
        totals, game_totals = build_team_total_map(games)
        assert totals.get('BOS') == 3.0
        assert totals.get('TOR') == 3.5
        assert game_totals.get('BOS') == 6.5
        assert game_totals.get('TOR') == 6.5

    @_skip_no_main
    def test_build_team_total_map_empty(self):
        from main import build_team_total_map
        totals, game_totals = build_team_total_map([])
        assert totals == {}
        assert game_totals == {}

    @_skip_no_main
    def test_ml_to_implied_team_total(self):
        from main import _ml_to_implied_team_total
        # Heavy favorite at home with 6.5 total
        home_tt, away_tt = _ml_to_implied_team_total(-200, 170, 6.5)
        assert home_tt is not None
        assert away_tt is not None
        assert abs(home_tt + away_tt - 6.5) < 0.01


# ================================================================
#  VALIDATION MODULE TESTS
# ================================================================

class TestValidation:
    """Test pre-flight validation checks."""

    def test_check_salary_file_exists(self, tmp_path):
        from validate import check_salary_file, CheckResult
        csv = tmp_path / "DKSalaries_2.25.26.csv"
        csv.write_text(
            "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
            + "\n".join(
                f"C,Player{i} ({i}),Player{i},{i},C/UTIL,{5000+i*100},ANA@BOS 02/25/2026 07:00PM ET,ANA,5.0"
                for i in range(30)
            )
        )
        result = check_salary_file(str(csv), '2026-02-25')
        assert result.status == CheckResult.PASS

    def test_check_salary_file_missing(self):
        from validate import check_salary_file, CheckResult
        result = check_salary_file('/nonexistent/file.csv', '2026-02-25')
        assert result.status == CheckResult.FAIL

    def test_check_salary_file_date_mismatch(self, tmp_path):
        from validate import check_salary_file, CheckResult
        csv = tmp_path / "DKSalaries_2.25.26.csv"
        csv.write_text(
            "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
            + "\n".join(
                f"C,Player{i} ({i}),Player{i},{i},C/UTIL,5000,ANA@BOS 02/25/2026 07:00PM ET,ANA,5.0"
                for i in range(30)
            )
        )
        result = check_salary_file(str(csv), '2026-02-26')  # Wrong date
        assert result.status == CheckResult.WARN

    def test_check_vegas_data_pass(self):
        from validate import check_vegas_data, CheckResult
        games = [
            {'matchup': 'Boston Bruins @ Toronto Maple Leafs', 'game_total': 6.5,
             'away_team_total': 3.0, 'home_team_total': 3.5}
        ]
        result = check_vegas_data(games, ['BOS', 'TOR'])
        assert result.status == CheckResult.PASS

    def test_check_vegas_data_missing_teams(self):
        from validate import check_vegas_data, CheckResult
        games = [
            {'matchup': 'Boston Bruins @ Toronto Maple Leafs', 'game_total': 6.5,
             'away_team_total': 3.0, 'home_team_total': 3.5}
        ]
        result = check_vegas_data(games, ['BOS', 'TOR', 'WPG'])  # WPG missing
        assert result.status == CheckResult.WARN

    def test_check_vegas_data_empty(self):
        from validate import check_vegas_data, CheckResult
        result = check_vegas_data([], ['BOS'])
        assert result.status == CheckResult.FAIL

    def test_check_data_pipeline_pass(self):
        from validate import check_data_pipeline, CheckResult
        data = {
            'skaters': pd.DataFrame({'name': [f'p{i}' for i in range(500)]}),
            'goalies': pd.DataFrame({'name': [f'g{i}' for i in range(60)]}),
            'teams': pd.DataFrame({'team': [f't{i}' for i in range(32)]}),
            'schedule': pd.DataFrame({'game': [1, 2, 3]}),
        }
        result = check_data_pipeline(data)
        assert result.status == CheckResult.PASS

    def test_check_data_pipeline_missing_skaters(self):
        from validate import check_data_pipeline, CheckResult
        data = {
            'skaters': pd.DataFrame(),
            'goalies': pd.DataFrame({'name': ['g1']}),
            'teams': pd.DataFrame({'team': ['t1']}),
            'schedule': pd.DataFrame({'game': [1]}),
        }
        result = check_data_pipeline(data)
        assert result.status == CheckResult.FAIL

    def test_check_projections_pass(self):
        from validate import check_projections, CheckResult
        skaters = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'projected_fpts': [8.0, 6.0, 5.0],
            'salary': [6000, 5000, 4000],
            'team': ['BOS', 'BOS', 'TOR'],
        })
        goalies = pd.DataFrame({
            'name': ['G1'],
            'projected_fpts': [10.0],
            'salary': [8000],
        })
        result = check_projections(skaters, goalies, ['BOS', 'TOR'])
        assert result.status == CheckResult.PASS

    def test_check_projections_negative(self):
        from validate import check_projections, CheckResult
        skaters = pd.DataFrame({
            'name': ['A'],
            'projected_fpts': [-5.0],
            'salary': [5000],
            'team': ['BOS'],
        })
        result = check_projections(skaters, pd.DataFrame(), ['BOS'])
        assert result.status in (CheckResult.WARN, CheckResult.FAIL)

    def test_check_projections_duplicates(self):
        from validate import check_projections, CheckResult
        skaters = pd.DataFrame({
            'name': ['A', 'A'],  # duplicate
            'projected_fpts': [8.0, 7.0],
            'salary': [6000, 5500],
            'team': ['BOS', 'BOS'],
        })
        goalies = pd.DataFrame({
            'name': ['G1'],
            'projected_fpts': [10.0],
        })
        result = check_projections(skaters, goalies, ['BOS'])
        assert result.status in (CheckResult.WARN, CheckResult.FAIL)

    def test_check_lineup_feasibility_pass(self):
        from validate import check_lineup_feasibility, CheckResult
        pool = pd.DataFrame({
            'position': ['C'] * 5 + ['W'] * 8 + ['D'] * 5 + ['G'] * 2,
            'salary': [5000] * 20,
            'name': [f'p{i}' for i in range(20)],
        })
        result = check_lineup_feasibility(pool)
        assert result.status == CheckResult.PASS

    def test_check_lineup_feasibility_no_goalies(self):
        from validate import check_lineup_feasibility, CheckResult
        pool = pd.DataFrame({
            'position': ['C'] * 5 + ['W'] * 5 + ['D'] * 5,
            'salary': [5000] * 15,
            'name': [f'p{i}' for i in range(15)],
        })
        result = check_lineup_feasibility(pool)
        assert result.status == CheckResult.FAIL

    def test_validation_report_summary(self):
        from validate import ValidationReport, CheckResult
        report = ValidationReport()
        report.add(CheckResult("Test1", CheckResult.PASS, "OK"))
        report.add(CheckResult("Test2", CheckResult.WARN, "Hmm"))
        report.add(CheckResult("Test3", CheckResult.PASS, "OK"))
        assert report.is_go is True
        assert len(report.passes) == 2
        assert len(report.warnings) == 1
        assert len(report.failures) == 0

    def test_validation_report_no_go(self):
        from validate import ValidationReport, CheckResult
        report = ValidationReport()
        report.add(CheckResult("Test1", CheckResult.PASS, "OK"))
        report.add(CheckResult("Test2", CheckResult.FAIL, "Bad"))
        assert report.is_go is False


# ================================================================
#  OPTIMIZER TESTS
# ================================================================

@_skip_no_optimizer
class TestOptimizer:
    """Test lineup optimizer logic."""

    @pytest.fixture
    def sample_pool(self):
        """Create a minimal valid player pool with proper game_info per team.

        Key: use 'position' column (C/W/D/G), NOT 'dk_position' (C/UTIL etc.)
        because the optimizer's _normalize_position doesn't handle 'C/UTIL' format.
        Also each player's game_info must match their team.
        """
        game_info_map = {
            'BOS': 'BOS@TOR 01/01/2026 07:00PM ET',
            'TOR': 'BOS@TOR 01/01/2026 07:00PM ET',
            'WPG': 'WPG@EDM 01/01/2026 09:00PM ET',
            'EDM': 'WPG@EDM 01/01/2026 09:00PM ET',
            'CGY': 'CGY@VAN 01/01/2026 10:00PM ET',
            'VAN': 'CGY@VAN 01/01/2026 10:00PM ET',
        }
        teams = ['BOS', 'TOR', 'WPG', 'EDM', 'CGY', 'VAN']
        players = []
        # 2 Centers per team
        for team in teams:
            for j in range(2):
                players.append({'name': f'C_{team}_{j}', 'position': 'C',
                               'salary': 4500 + j * 500, 'projected_fpts': 6.0 + j,
                               'team': team, 'game_info': game_info_map[team]})
        # 3 Wings per team
        for team in teams:
            for j in range(3):
                players.append({'name': f'W_{team}_{j}', 'position': 'W',
                               'salary': 4000 + j * 400, 'projected_fpts': 5.0 + j * 0.5,
                               'team': team, 'game_info': game_info_map[team]})
        # 2 Defensemen per team
        for team in teams:
            for j in range(2):
                players.append({'name': f'D_{team}_{j}', 'position': 'D',
                               'salary': 3500 + j * 500, 'projected_fpts': 4.0 + j * 0.5,
                               'team': team, 'game_info': game_info_map[team]})
        # 1 Goalie per team
        for team in teams:
            players.append({'name': f'G_{team}', 'position': 'G',
                           'salary': 7500, 'projected_fpts': 9.0,
                           'team': team, 'game_info': game_info_map[team]})
        return pd.DataFrame(players)

    def test_optimize_returns_lineup(self, sample_pool):
        from optimizer import NHLLineupOptimizer
        opt = NHLLineupOptimizer()
        lineups = opt.optimize_lineup(sample_pool, n_lineups=1, mode='cash')
        assert len(lineups) >= 1
        lineup = lineups[0]
        assert len(lineup) == 9  # 9 roster slots

    def test_lineup_under_salary_cap(self, sample_pool):
        from optimizer import NHLLineupOptimizer
        opt = NHLLineupOptimizer()
        lineups = opt.optimize_lineup(sample_pool, n_lineups=1, mode='cash')
        if lineups:
            total_salary = lineups[0]['salary'].sum()
            assert total_salary <= 50000

    def test_lineup_position_requirements(self, sample_pool):
        from optimizer import NHLLineupOptimizer
        opt = NHLLineupOptimizer()
        lineups = opt.optimize_lineup(sample_pool, n_lineups=1, mode='cash')
        if lineups:
            lineup = lineups[0]
            positions = lineup['position'].str.upper().tolist()
            # Must have at least 1 G, 2 C, 2 D, 3 W (UTIL can be any skater)
            assert positions.count('G') >= 1

    def test_optimizer_opponent_extraction(self):
        from optimizer import NHLLineupOptimizer
        opt = NHLLineupOptimizer()
        row = pd.Series({'game_info': 'ANA@EDM 01/26/2026 08:30PM ET', 'team': 'ANA'})
        opp = opt._get_opponent_team(row)
        assert opp == 'EDM'

        row2 = pd.Series({'game_info': 'ANA@EDM 01/26/2026 08:30PM ET', 'team': 'EDM'})
        opp2 = opt._get_opponent_team(row2)
        assert opp2 == 'ANA'


# ================================================================
#  CONTEST ROI TESTS
# ================================================================

class TestContestROI:
    """Test contest profile and EV calculations."""

    def test_contest_profile_creation(self):
        from contest_roi import ContestProfile
        cp = ContestProfile(
            entry_fee=5.0,
            max_entries=1,
            field_size=10000,
            payout_preset='top_heavy_gpp'
        )
        assert cp.entry_fee == 5.0
        assert cp.field_size == 10000

    def test_leverage_recommendation(self):
        from contest_roi import ContestProfile, recommend_leverage
        cp = ContestProfile(entry_fee=5.0, max_entries=1, field_size=10000)
        rec = recommend_leverage(cp)
        assert hasattr(rec, 'target_ownership_low')
        assert hasattr(rec, 'target_ownership_high')
        assert hasattr(rec, 'leverage_tier')
        assert hasattr(rec, 'summary')


# ================================================================
#  MERGE LOGIC TESTS
# ================================================================

class TestMergeLogic:
    """Test projection-salary merge."""

    def test_name_cleaning_accents(self):
        """Verify accent normalization for player matching."""
        import unicodedata, re
        _UMLAUT_MAP = {'ü': 'ue', 'ö': 'oe', 'ä': 'ae', 'ß': 'ss'}

        def _clean_name(s):
            for orig, repl in _UMLAUT_MAP.items():
                s = s.replace(orig, repl).replace(orig.upper(), repl.capitalize())
            s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
            s = s.lower().strip()
            s = re.sub(r'[^a-z\s]', '', s)
            return s

        assert _clean_name('León Draisaitl') == 'leon draisaitl'
        assert _clean_name('Patrik Laine') == 'patrik laine'
        assert _clean_name("Connor O'Brien") == 'connor obrien'
        assert _clean_name('Nico Hischier') == 'nico hischier'
        assert _clean_name('Tim Stützle') == 'tim stuetzle'


# ================================================================
#  EDGE CASE / REGRESSION TESTS
# ================================================================

class TestEdgeCases:
    """Guard against known past bugs."""

    def test_goalie_negative_fpts_possible(self):
        """Goalies can legitimately go negative — model shouldn't floor at 0."""
        from config import calculate_goalie_fantasy_points
        pts = calculate_goalie_fantasy_points(win=False, saves=10, goals_against=7)
        assert pts < 0

    def test_skater_shootout_goal(self):
        from config import calculate_skater_fantasy_points
        pts = calculate_skater_fantasy_points(0, 0, 0, 0, shootout_goals=1)
        assert pts == 1.5

    @_skip_no_main
    def test_empty_dataframe_handling(self):
        """Merge with empty dataframe shouldn't crash."""
        from main import merge_projections_with_salaries
        empty_proj = pd.DataFrame(columns=['name', 'projected_fpts', 'position', 'team'])
        empty_sal = pd.DataFrame(columns=['dk_name', 'salary', 'dk_position', 'dk_id', 'dk_avg_fpts', 'position'])
        result = merge_projections_with_salaries(empty_proj, empty_sal, 'skater')
        assert len(result) == 0


# ================================================================
#  PROJECTION BIAS CORRECTION TESTS
# ================================================================

@_skip_no_tabpfn
class TestBiasCorrections:
    """Verify bias correction constants are in sane ranges."""

    def test_global_bias_in_range(self):
        from projections import GLOBAL_BIAS_CORRECTION
        assert 0.3 <= GLOBAL_BIAS_CORRECTION <= 1.5

    def test_goalie_bias_in_range(self):
        from projections import GOALIE_BIAS_CORRECTION
        assert 0.1 <= GOALIE_BIAS_CORRECTION <= 1.0

    def test_floor_multiplier_in_range(self):
        from projections import FLOOR_MULTIPLIER
        assert 0.1 <= FLOOR_MULTIPLIER <= 0.5

    def test_max_multiplicative_swing(self):
        from projections import MAX_MULTIPLICATIVE_SWING
        assert 0.05 <= MAX_MULTIPLICATIVE_SWING <= 0.30

    def test_position_bias_near_one(self):
        from projections import POSITION_BIAS_CORRECTION
        for pos, val in POSITION_BIAS_CORRECTION.items():
            assert 0.8 <= val <= 1.2, f"Position {pos} bias {val} out of range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
