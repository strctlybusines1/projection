"""
Pytest configuration for NHL DFS Pipeline tests.
"""
import sys
from pathlib import Path

# Add the projection directory to Python path so tests can import modules
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
