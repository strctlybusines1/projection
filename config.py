"""
DraftKings NHL Scoring Configuration and API Settings
"""

# DraftKings NHL Classic Scoring - Skaters
SKATER_SCORING = {
    "goals": 8.5,
    "assists": 5.0,
    "shots_on_goal": 1.5,
    "blocked_shots": 1.3,
    "shorthanded_points_bonus": 2.0,  # Per SH goal or assist
    "shootout_goal": 1.5,
}

# Skater Bonuses
SKATER_BONUSES = {
    "hat_trick": 3.0,           # 3+ goals
    "five_plus_shots": 3.0,     # 5+ shots on goal
    "three_plus_blocks": 3.0,   # 3+ blocked shots
    "three_plus_points": 3.0,   # 3+ points (G+A)
}

# DraftKings NHL Classic Scoring - Goalies
GOALIE_SCORING = {
    "win": 6.0,
    "save": 0.7,
    "goal_against": -3.5,
    "shutout_bonus": 4.0,
    "overtime_loss": 2.0,
}

# Goalie Bonuses
GOALIE_BONUSES = {
    "thirty_five_plus_saves": 3.0,  # 35+ saves
}

# NHL API Configuration
NHL_API_WEB_BASE = "https://api-web.nhle.com"
NHL_API_STATS_BASE = "https://api.nhle.com/stats/rest"

# Current season format: YYYYYYYY (e.g., 20242025)
CURRENT_SEASON = "20242025"
GAME_TYPE_REGULAR = 2
GAME_TYPE_PLAYOFFS = 3

# Team codes for NHL
NHL_TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WSH", "WPG"
]


def calculate_skater_fantasy_points(goals, assists, shots, blocks, sh_goals=0, sh_assists=0, shootout_goals=0):
    """Calculate DraftKings fantasy points for a skater."""
    points = 0

    # Base scoring
    points += goals * SKATER_SCORING["goals"]
    points += assists * SKATER_SCORING["assists"]
    points += shots * SKATER_SCORING["shots_on_goal"]
    points += blocks * SKATER_SCORING["blocked_shots"]
    points += (sh_goals + sh_assists) * SKATER_SCORING["shorthanded_points_bonus"]
    points += shootout_goals * SKATER_SCORING["shootout_goal"]

    # Bonuses
    if goals >= 3:
        points += SKATER_BONUSES["hat_trick"]
    if shots >= 5:
        points += SKATER_BONUSES["five_plus_shots"]
    if blocks >= 3:
        points += SKATER_BONUSES["three_plus_blocks"]
    if (goals + assists) >= 3:
        points += SKATER_BONUSES["three_plus_points"]

    return points


def calculate_goalie_fantasy_points(win, saves, goals_against, shutout=False, ot_loss=False):
    """Calculate DraftKings fantasy points for a goalie."""
    points = 0

    # Base scoring
    if win:
        points += GOALIE_SCORING["win"]
    if ot_loss:
        points += GOALIE_SCORING["overtime_loss"]

    points += saves * GOALIE_SCORING["save"]
    points += goals_against * GOALIE_SCORING["goal_against"]

    if shutout:
        points += GOALIE_SCORING["shutout_bonus"]

    # Bonuses
    if saves >= 35:
        points += GOALIE_BONUSES["thirty_five_plus_saves"]

    return points
