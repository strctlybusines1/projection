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

# Current season format: YYYYYYYY (e.g., 20252026)
CURRENT_SEASON = "20252026"  # FIX: was incorrectly set to 20242025
GAME_TYPE_REGULAR = 2
GAME_TYPE_PLAYOFFS = 3

# Team codes for NHL
NHL_TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WSH", "WPG"
]

# ==================== MoneyPuck Configuration ====================
MONEYPUCK_INJURIES_URL = "https://moneypuck.com/moneypuck/playerData/playerNews/current_injuries.csv"

# Injury statuses that should exclude players from projections
# DTD = Day-to-Day (might play, include by default)
# IR = Injured Reserve
# IR-LT = Injured Reserve Long-Term
# IR-NR = Injured Reserve Non-Roster
# O = Out
INJURY_STATUSES_EXCLUDE = ['IR', 'IR-LT', 'IR-NR', 'O']
INJURY_STATUSES_ALL = ['DTD', 'IR', 'IR-LT', 'IR-NR', 'O']

# ==================== Natural Stat Trick Configuration ====================
NST_BASE_URL = "https://www.naturalstattrick.com"
NST_RATE_LIMIT_DELAY = 2.0  # Seconds between requests
NST_RECENT_FORM_GAMES = 10  # Number of games for recent form analysis

# Team code mappings (NST format -> standard NHL format)
NST_TEAM_MAP = {
    'T.B': 'TBL',
    'N.J': 'NJD',
    'L.A': 'LAK',
    'S.J': 'SJS',
}

# Standard team code mappings (for reverse lookup)
STANDARD_TO_NST_MAP = {v: k for k, v in NST_TEAM_MAP.items()}

# ==================== xG and Advanced Stats Configuration ====================
# League average values for normalization
LEAGUE_AVG_XGF_60 = 2.5  # Expected goals for per 60 at 5v5
LEAGUE_AVG_XGA_60 = 2.5  # Expected goals against per 60 at 5v5
LEAGUE_AVG_CF_PCT = 50.0  # Corsi for percentage
LEAGUE_AVG_PDO = 100.0   # PDO (SH% + SV%)

# PDO regression thresholds
PDO_HIGH_THRESHOLD = 102.0  # Above this, expect regression down
PDO_LOW_THRESHOLD = 98.0    # Below this, expect regression up
PDO_REGRESSION_FACTOR = 0.05  # How much to adjust projections

# Hot/cold streak thresholds (based on recent xG vs season avg)
HOT_STREAK_THRESHOLD = 1.15   # 15% above season average
COLD_STREAK_THRESHOLD = 0.85  # 15% below season average
STREAK_ADJUSTMENT_FACTOR = 0.10  # Max adjustment for streaks


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
