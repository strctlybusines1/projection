"""
Pytest configuration for NHL DFS Pipeline tests.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import json

# Add the projection directory to Python path so tests can import modules
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


@pytest.fixture
def sample_dk_salaries():
    """Realistic DraftKings salary CSV data."""
    return pd.DataFrame({
        'Name': ['Connor McDavid', 'Nathan MacKinnon', 'Cale Makar', 'Connor Hellebuyck',
                 'Brad Marchand', 'David Pastrnak', 'Charlie McAvoy', 'Jeremy Swayman'],
        'Position': ['C', 'C', 'D', 'G', 'LW', 'RW', 'D', 'G'],
        'Name + ID': ['Connor McDavid (8478402)', 'Nathan MacKinnon (8477492)',
                      'Cale Makar (8480069)', 'Connor Hellebuyck (8476432)',
                      'Brad Marchand (8473419)', 'David Pastrnak (8477956)',
                      'Charlie McAvoy (8479325)', 'Jeremy Swayman (8480280)'],
        'Salary': [9500, 9000, 8500, 8000, 7500, 8000, 6500, 7000],
        'Game Info': ['EDM@COL', 'COL@EDM', 'COL@EDM', 'WPG@BOS',
                      'BOS@WPG', 'BOS@WPG', 'BOS@WPG', 'BOS@WPG'],
        'TeamAbbrev': ['EDM', 'COL', 'COL', 'WPG', 'BOS', 'BOS', 'BOS', 'BOS'],
        'AvgPointsPerGame': [12.5, 11.8, 9.2, 8.5, 8.9, 9.5, 7.2, 8.1],
    })


@pytest.fixture
def sample_projections():
    """Sample projection data for skaters and goalies."""
    skaters = pd.DataFrame({
        'name': ['Connor McDavid', 'Nathan MacKinnon', 'Cale Makar', 'Brad Marchand',
                 'David Pastrnak', 'Charlie McAvoy'],
        'team': ['EDM', 'COL', 'COL', 'BOS', 'BOS', 'BOS'],
        'position': ['C', 'C', 'D', 'LW', 'RW', 'D'],
        'projected_fpts': [14.2, 13.5, 10.8, 10.2, 11.1, 8.5],
        'floor': [5.0, 5.5, 4.2, 4.8, 5.2, 3.5],
        'ceiling': [28.0, 26.5, 22.0, 20.5, 24.0, 18.0],
        'std_dev': [6.5, 6.2, 5.8, 5.4, 6.0, 5.0],
    })

    goalies = pd.DataFrame({
        'name': ['Connor Hellebuyck', 'Jeremy Swayman'],
        'team': ['WPG', 'BOS'],
        'position': ['G', 'G'],
        'projected_fpts': [9.5, 8.8],
        'floor': [0.0, 0.0],
        'ceiling': [20.0, 18.5],
        'std_dev': [7.0, 6.5],
    })

    return {'skaters': skaters, 'goalies': goalies}


@pytest.fixture
def sample_lines_data():
    """Sample line combination data from DailyFaceoff."""
    return {
        'BOS': {
            'forward_lines': [
                {'line': 1, 'players': ['Brad Marchand', 'Elias Lindholm', 'David Pastrnak']},
                {'line': 2, 'players': ['Pavel Zacha', 'Charlie Coyle', 'Trent Frederic']},
            ],
            'defense_pairs': [
                {'pair': 1, 'players': ['Charlie McAvoy', 'Hampus Lindholm']},
                {'pair': 2, 'players': ['Mason Lohrei', 'Brandon Carlo']},
            ],
            'pp_units': [
                {'unit': 1, 'players': ['Brad Marchand', 'David Pastrnak', 'Elias Lindholm',
                                         'Charlie McAvoy', 'Hampus Lindholm']},
                {'unit': 2, 'players': ['Pavel Zacha', 'Trent Frederic', 'Charlie Coyle',
                                         'Mason Lohrei', 'Brandon Carlo']},
            ],
            'confirmed_goalie': 'Jeremy Swayman',
        },
        'COL': {
            'forward_lines': [
                {'line': 1, 'players': ['Artturi Lehkonen', 'Nathan MacKinnon', 'Mikko Rantanen']},
            ],
            'defense_pairs': [
                {'pair': 1, 'players': ['Cale Makar', 'Devon Toews']},
            ],
            'pp_units': [
                {'unit': 1, 'players': ['Nathan MacKinnon', 'Mikko Rantanen', 'Cale Makar',
                                         'Devon Toews', 'Artturi Lehkonen']},
            ],
            'confirmed_goalie': 'Alexandar Georgiev',
        },
    }


@pytest.fixture
def sample_historical_scores():
    """Sample historical player scores for distribution fitting."""
    return pd.DataFrame({
        'Player': ['Connor McDavid'] * 20 + ['Nathan MacKinnon'] * 20,
        'Team': ['EDM'] * 20 + ['COL'] * 20,
        'Score': [12.5, 15.0, 8.5, 0.0, 22.0, 11.0, 9.5, 1.5, 18.0, 14.5,
                  10.0, 7.5, 0.5, 16.0, 13.0, 9.0, 2.0, 20.0, 12.0, 11.5] * 2,
        'slate_date': ['2026-01-01'] * 40,
    })


@pytest.fixture
def sample_player_pool(sample_projections, sample_dk_salaries):
    """Merged player pool with projections and salaries."""
    skaters = sample_projections['skaters'].copy()
    goalies = sample_projections['goalies'].copy()

    # Add salary info
    salaries_dict = dict(zip(sample_dk_salaries['Name'], sample_dk_salaries['Salary']))
    dk_avg_dict = dict(zip(sample_dk_salaries['Name'], sample_dk_salaries['AvgPointsPerGame']))

    for df in [skaters, goalies]:
        df['salary'] = df['name'].map(salaries_dict).fillna(3000)
        df['dk_avg_fpts'] = df['name'].map(dk_avg_dict).fillna(5.0)
        df['value'] = df['projected_fpts'] / (df['salary'] / 1000)
        df['edge'] = df['projected_fpts'] - df['dk_avg_fpts']

    return pd.concat([skaters, goalies], ignore_index=True)


@pytest.fixture
def mock_nhl_api():
    """Mock NHL API responses."""
    with patch('requests.get') as mock_get:
        # Mock schedule response
        mock_schedule = Mock()
        mock_schedule.json.return_value = {
            'dates': [{
                'games': [{
                    'teams': {
                        'home': {'team': {'abbreviation': 'BOS'}},
                        'away': {'team': {'abbreviation': 'WPG'}},
                    },
                    'venue': {'name': 'TD Garden'},
                }]
            }]
        }
        mock_schedule.status_code = 200
        mock_get.return_value = mock_schedule
        yield mock_get


@pytest.fixture
def temp_test_dir(tmp_path):
    """Temporary directory for test file I/O."""
    test_dir = tmp_path / "nhl_dfs_test"
    test_dir.mkdir()

    # Create subdirectories
    (test_dir / "data").mkdir()
    (test_dir / "daily_salaries").mkdir()
    (test_dir / "daily_projections").mkdir()
    (test_dir / "contests").mkdir()
    (test_dir / "backtests").mkdir()

    return test_dir
