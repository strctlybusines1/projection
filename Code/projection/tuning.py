"""
Centralized tuning parameter loader.

Reads calibrated constants from tuning_params.json so all magic numbers
live in one human-readable, version-controllable file instead of being
scattered across 10+ Python modules.

Usage:
    from tuning import params

    bias = params['projections']['global_bias_correction']
    elite_oz = params['edge_stats']['skater_boosts']['elite_oz_time']

    # Or use the helper:
    from tuning import get_param
    bias = get_param('projections.global_bias_correction', default=0.80)
"""

import json
from pathlib import Path
from typing import Any

_PARAMS_FILE = Path(__file__).parent / "tuning_params.json"

# Load once at import time
with open(_PARAMS_FILE, 'r') as f:
    params: dict = json.load(f)


def get_param(dotted_path: str, default: Any = None) -> Any:
    """
    Get a tuning parameter by dotted path.

    Example:
        get_param('projections.global_bias_correction')
        get_param('edge_stats.skater_boosts.elite_oz_time')
        get_param('nonexistent.key', default=1.0)
    """
    keys = dotted_path.split('.')
    obj = params
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return default
    return obj


def reload_params():
    """Reload parameters from disk (useful after editing tuning_params.json)."""
    global params
    with open(_PARAMS_FILE, 'r') as f:
        params = json.load(f)
