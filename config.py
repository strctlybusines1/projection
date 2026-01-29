"""
DraftKings NHL Scoring Configuration and API Settings
"""

# Data directories (relative to projection/)
DAILY_SALARIES_DIR = "daily_salaries"
VEGAS_DIR = "vegas"
BACKTESTS_DIR = "backtests"
CONTESTS_DIR = "contests"
DAILY_PROJECTIONS_DIR = "daily_projections"

# Team danger (HD/MD/LD) from test-folder NST CSVs - optional for goalie shot quality
TEAM_DANGER_CSV_DIR = None   # e.g. "../test" (relative to projection/) to use CSVs; None = skip
TEAM_DANGER_CSV = None        # specific file e.g. "1.25.26_1.27.26_nhl.csv"; None = pick latest by date
LEAGUE_AVG_HD_SHARE = 0.33    # ~league avg share of shots that are high-danger
SHOT_QUALITY_ADJ_CAP = 0.08   # cap opponent shot-quality adjustment at ±8%

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

# Goalie Quality Tiers
# Based on Jan 26 backtest: Korpisalo (backup-quality) recommended as starter, scored 4.8 FPTS
GOALIE_TIERS = {
    'ELITE': {'min_save_pct': 0.915, 'min_games_started': 20, 'projection_mult': 1.0},
    'STARTER': {'min_save_pct': 0.900, 'min_games_started': 15, 'projection_mult': 0.95},
    'BACKUP': {'min_save_pct': 0.0, 'min_games_started': 0, 'projection_mult': 0.80},
}

# Injury Opportunity Boost
# Based on Jan 26 backtest: ANA lost key players, Granlund (43.3 FPTS) exploded
INJURY_OPPORTUNITY = {
    'key_player_boost': 0.05,      # +5% per key injured player (top-6 F, top-4 D)
    'regular_player_boost': 0.02,  # +2% per other injured player
    'max_boost': 0.20,             # Cap at 20% total boost
    'key_player_threshold_ppg': 0.5,  # Points per game threshold for "key player"
}

# DK per TOI / Expected TOI (plan: dk_per_toi_and_expected_toi)
USE_DK_PER_TOI_PROJECTION = False   # If True, use rate-based: dk_pts_per_60 * (expected_toi_minutes/60)
USE_EXPECTED_TOI_INJURY_BUMP = False  # If True, bump expected_toi_minutes when key teammates out
EXPECTED_TOI_BUMP_CAP = 0.15        # Cap TOI bump at 15%
# When rate-based is on but expected_toi_minutes == toi_minutes (no situation TOI), scale base so comparison differs
RATE_BASED_SAME_TOI_SCALE = 0.97    # 3% lower when data doesn't differentiate (backtest comparison)

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

# Full team name -> abbrev (for NST CSVs that use e.g. "Chicago Blackhawks")
TEAM_FULL_NAME_TO_ABBREV = {
    'ANAHEIM DUCKS': 'ANA',
    'ARIZONA COYOTES': 'ARI',
    'BOSTON BRUINS': 'BOS',
    'BUFFALO SABRES': 'BUF',
    'CALGARY FLAMES': 'CGY',
    'CAROLINA HURRICANES': 'CAR',
    'CHICAGO BLACKHAWKS': 'CHI',
    'COLORADO AVALANCHE': 'COL',
    'COLUMBUS BLUE JACKETS': 'CBJ',
    'DALLAS STARS': 'DAL',
    'DETROIT RED WINGS': 'DET',
    'EDMONTON OILERS': 'EDM',
    'FLORIDA PANTHERS': 'FLA',
    'LOS ANGELES KINGS': 'LAK',
    'MINNESOTA WILD': 'MIN',
    'MONTREAL CANADIENS': 'MTL',
    'NASHVILLE PREDATORS': 'NSH',
    'NEW JERSEY DEVILS': 'NJD',
    'NEW YORK ISLANDERS': 'NYI',
    'NEW YORK RANGERS': 'NYR',
    'OTTAWA SENATORS': 'OTT',
    'PHILADELPHIA FLYERS': 'PHI',
    'PITTSBURGH PENGUINS': 'PIT',
    'SAN JOSE SHARKS': 'SJS',
    'SEATTLE KRAKEN': 'SEA',
    'ST. LOUIS BLUES': 'STL',
    'ST LOUIS BLUES': 'STL',
    'TAMPA BAY LIGHTNING': 'TBL',
    'TORONTO MAPLE LEAFS': 'TOR',
    'UTAH HOCKEY CLUB': 'UTA',
    'VANCOUVER CANUCKS': 'VAN',
    'VEGAS GOLDEN KNIGHTS': 'VGK',
    'WASHINGTON CAPITALS': 'WSH',
    'WINNIPEG JETS': 'WPG',
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
PDO_REGRESSION_FACTOR = 0.02  # Reduced from 0.05 - backtest confirms PDO is noise (corr 0.107)

# Hot/cold streak thresholds (based on recent xG vs season avg)
HOT_STREAK_THRESHOLD = 1.15   # 15% above season average
COLD_STREAK_THRESHOLD = 0.85  # 15% below season average
STREAK_ADJUSTMENT_FACTOR = 0.10  # Max adjustment for streaks

# ==================== Signal-Weighted Team Quality Configuration ====================
# Derived from season_signal_backtest.py results (signal_noise_report.csv)
# Weights = max absolute predictive correlation with DK net pts/game
SIGNAL_WEIGHTS = {
    'sf_pct':   0.455,   # Shot share - persistence 0.527, corr 0.455
    'xgf_pct':  0.455,   # xG share - persistence 0.454, corr 0.455
    'scf_pct':  0.438,   # Scoring chance share - persistence 0.478, corr 0.438
    'ff_pct':   0.432,   # Fenwick share - persistence 0.546, corr 0.432
    'cf_pct':   0.402,   # Corsi share - persistence 0.581, corr 0.402
}
_total = sum(SIGNAL_WEIGHTS.values())
SIGNAL_WEIGHTS_NORMALIZED = {k: v / _total for k, v in SIGNAL_WEIGHTS.items()}

SIGNAL_COMPOSITE_SENSITIVITY = 0.30   # Pct-point deviation -> projection % change
SIGNAL_COMPOSITE_CLIP_LOW = 0.92      # Worst matchup: -8%
SIGNAL_COMPOSITE_CLIP_HIGH = 1.10     # Best matchup: +10%
LEAGUE_AVG_SHARE_PCT = 50.0           # Neutral share percentage

# ==================== GPP Optimizer Configuration ====================
# Based on analysis of winning GPP lineups from 1/22/26 slate

# Stack size constraints
GPP_MIN_STACK_SIZE = 3       # Minimum players from one team for GPP
GPP_MAX_FROM_TEAM = 6        # Allow up to 6 from one team (winning lineup had NSH 6-stack)
CASH_MAX_FROM_TEAM = 4       # Conservative limit for cash games

# Stack correlation boosts
PRIMARY_STACK_BOOST = 0.20   # 20% boost for primary stack players
SECONDARY_STACK_BOOST = 0.12 # 12% boost for secondary stack
LINEMATE_BOOST = 0.15        # 15% boost for confirmed linemates
GOALIE_CORRELATION_BOOST = 0.10  # 10% boost for skaters with same-team goalie

# Game environment targeting
HIGH_TOTAL_THRESHOLD = 6.0   # Vegas total above this = smash spot
BLOWOUT_SPREAD_THRESHOLD = 2.5  # Spread above this = potential blowout (target underdog)

# Stack structure preferences (based on winning lineup patterns)
# 68% of top 100 had MIN stacks, 29% had NSH stacks
# Winning lineup: NSH(6) + MIN(2)
PREFERRED_PRIMARY_STACK_SIZE = 4  # Target 4 players in primary stack
PREFERRED_SECONDARY_STACK_SIZE = 3  # Target 3 players in secondary stack (based on $360 Spin analysis)

# Player pair correlations from GPP analysis
# Kaprizov + Zuccarello appeared in 59% of winning lineups
# These are linemates who should be stacked together
TOP_CORRELATION_PAIRS = [
    # Format: (player1, player2, team, boost_multiplier)
    # Add known high-correlation pairs here as discovered
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
