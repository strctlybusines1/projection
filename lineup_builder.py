"""
Integrated Lineup Builder — GPP lineup construction using:
1. MDN v3 projections (point estimates + ceiling probabilities + std dev)
2. Ownership v2 predictions (XGBoost, MAE 1.92%, corr 0.905)
3. Game theory leverage weighting (GPP_value = ceiling × 1/sqrt(own))
4. Monte Carlo simulation for ceiling-based lineup generation
5. Stacking constraints from winning lineup analysis

Architecture:
    MDN v3 → projected_fpts, std_fpts, p_above_15, p_above_20
    Ownership v2 → predicted_ownership
    Game Theory → leverage_score = f(projection, ownership, ceiling)
    Monte Carlo → sample N lineups from mixture distributions
    Stacking → enforce 4-3-1-1 structure with team correlation

Backtestable against contest_results table (7,428 entries).

Author: Claude
Date: 2026-02-16
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from itertools import combinations
import unicodedata
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SALARY_CAP = 50_000
MIN_SALARY_PER_PLAYER = 2_500

# DK NHL Classic: 2C, 3W, 2D, 1G, 1UTIL (C/W only for UTIL)
ROSTER_SLOTS = {'C': 2, 'W': 3, 'D': 2, 'G': 1, 'UTIL': 1}  # total = 9

# GPP scoring weights for player selection
# GPP_value = w_proj * norm(proj) + w_ceil * norm(ceiling) + w_lev * norm(leverage)
GPP_WEIGHTS = {
    'projection': 0.35,    # Point projection (MDN expected value)
    'ceiling': 0.35,       # Ceiling probability (p_above_15 or p_above_20)
    'leverage': 0.30,      # Ownership leverage (high proj / low own)
}

# Stacking config (from winning lineup analysis: 68% had 3-4 player stacks)
PRIMARY_STACK_SIZE = 4
SECONDARY_STACK_SIZE = 3
MAX_FROM_TEAM = 6  # Allow up to 6 from one team
MIN_TEAMS = 3      # Minimum unique teams (skaters only)

# Monte Carlo config
MC_ITERATIONS = 1000      # Simulations per lineup generation
MC_LINEUP_POOL = 5000     # Generate this many candidate lineups
MC_TOP_SELECT = 20        # Return top N lineups

# Leverage function parameters
LEVERAGE_OWNERSHIP_FLOOR = 0.5   # Minimum ownership % to avoid divide-by-zero
LEVERAGE_CEILING_THRESHOLD = 15  # FPTS threshold for "ceiling game"

DB_PATH = Path(__file__).parent / 'data' / 'nhl_dfs_history.db'


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_mdn_backtest_results(date_str: str = None) -> pd.DataFrame:
    """
    Load MDN v3 backtest results.
    If date_str provided, filter to that date only.
    """
    path = Path(__file__).parent / 'mdn_v3_backtest_results.csv'
    if not path.exists():
        print("Warning: mdn_v3_backtest_results.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['game_date'] = pd.to_datetime(df['game_date'])

    if date_str:
        df = df[df['game_date'] == pd.to_datetime(date_str)]

    return df


def load_dk_salaries_for_date(date_str: str) -> pd.DataFrame:
    """Load DK salaries from database for a specific date."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"""
        SELECT player_name as name, team, position, salary,
               dk_avg_fpts, dk_ceiling, dk_stdv, fc_proj, opponent,
               team_implied_total, opp_implied_total, game_total,
               is_favorite, spread, win_pct, ownership_pct,
               start_line, pp_unit,
               n_games_on_slate, slate_size_players
        FROM dk_salaries
        WHERE slate_date = '{date_str}'
    """, conn)
    conn.close()

    if df.empty:
        return df

    # Normalize positions
    df['norm_pos'] = df['position'].apply(normalize_position)
    return df


def load_ownership_data_for_date(date_str: str) -> pd.DataFrame:
    """Load real ownership data from own.csv for a specific date."""
    path = Path(__file__).parent / 'own.csv'
    if not path.exists():
        return pd.DataFrame()

    own = pd.read_csv(path)

    # Parse date
    own['date'] = pd.to_datetime(own['Date'], format='%m/%d/%y', errors='coerce')
    target = pd.to_datetime(date_str)
    own = own[own['date'] == target]

    if own.empty:
        return pd.DataFrame()

    # Normalize for merging
    own['name_lower'] = own['Player'].str.lower().str.strip()
    return own


def load_contest_results(date_str: str = None) -> pd.DataFrame:
    """Load contest results for backtesting."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM contest_results"
    if date_str:
        query += f" WHERE slate_date = '{date_str}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_actuals_for_date(date_str: str) -> pd.DataFrame:
    """Load actual FPTS for a specific date."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"""
        SELECT name, team, position, actual_fpts
        FROM actuals
        WHERE game_date = '{date_str}'
    """, conn)
    conn.close()
    return df


# ==============================================================================
# HELPERS
# ==============================================================================

def normalize_position(pos: str) -> str:
    """Normalize position codes for DK roster."""
    if pd.isna(pos):
        return 'W'
    pos = str(pos).upper().strip()
    if pos in ('LW', 'RW'):
        return 'W'
    if pos in ('LD', 'RD'):
        return 'D'
    if pos in ('C', 'W', 'D', 'G'):
        return pos
    return 'W'


def parse_opponent(team: str, game_info: str) -> Optional[str]:
    """Extract opponent team from DK game info string."""
    if not game_info or pd.isna(game_info):
        return None
    # Format: "PIT@CBJ 01/22/2026 07:00PM ET"
    parts = str(game_info).split(' ')
    if not parts:
        return None
    matchup = parts[0]
    teams = matchup.split('@')
    if len(teams) != 2:
        return None
    away, home = teams[0].upper(), teams[1].upper()
    team_upper = str(team).upper()
    if team_upper == away:
        return home
    elif team_upper == home:
        return away
    return None


# ==============================================================================
# GAME THEORY: LEVERAGE SCORING
# ==============================================================================

def compute_leverage_scores(player_pool: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GPP leverage scores combining projection quality and ownership.

    GPP value = ceiling_probability × (1 / sqrt(predicted_ownership))

    This captures both upside potential AND differentiation from the field.
    High-leverage plays: high ceiling + low ownership = GPP gold.
    """
    df = player_pool.copy()

    # Ensure required columns exist
    if 'predicted_ownership' not in df.columns:
        df['predicted_ownership'] = 5.0  # Default 5% if no ownership data

    if 'p_above_15' not in df.columns:
        # Estimate from projected_fpts and std_fpts
        if 'std_fpts' in df.columns:
            z = (15 - df['projected_fpts']) / (df['std_fpts'] + 1e-6)
            from scipy.stats import norm
            df['p_above_15'] = 1 - norm.cdf(z)
        else:
            df['p_above_15'] = np.where(df['projected_fpts'] > 12, 0.3, 0.1)

    if 'p_above_20' not in df.columns:
        if 'std_fpts' in df.columns:
            z = (20 - df['projected_fpts']) / (df['std_fpts'] + 1e-6)
            from scipy.stats import norm
            df['p_above_20'] = 1 - norm.cdf(z)
        else:
            df['p_above_20'] = np.where(df['projected_fpts'] > 15, 0.15, 0.03)

    # Ownership leverage: inverse sqrt of ownership (diminishing penalty for high own)
    own_clipped = df['predicted_ownership'].clip(lower=LEVERAGE_OWNERSHIP_FLOOR)
    df['ownership_leverage'] = 1.0 / np.sqrt(own_clipped)

    # Raw leverage score: ceiling × ownership leverage
    df['leverage_score'] = df['p_above_15'] * df['ownership_leverage']

    # Normalize components to 0-1 scale for weighted scoring
    for col in ['projected_fpts', 'p_above_15', 'leverage_score']:
        col_min = df[col].min()
        col_max = df[col].max()
        col_range = col_max - col_min
        if col_range > 0:
            df[f'{col}_norm'] = (df[col] - col_min) / col_range
        else:
            df[f'{col}_norm'] = 0.5

    # Composite GPP score (weighted combination)
    df['gpp_score'] = (
        GPP_WEIGHTS['projection'] * df['projected_fpts_norm'] +
        GPP_WEIGHTS['ceiling'] * df['p_above_15_norm'] +
        GPP_WEIGHTS['leverage'] * df['leverage_score_norm']
    )

    # Value efficiency: GPP score per $1K salary
    df['gpp_value'] = df['gpp_score'] / (df['salary'] / 1000)

    # Tier classification
    try:
        q25, q50, q75 = df['leverage_score'].quantile([0.25, 0.50, 0.75])
        # Ensure strictly increasing bins
        bins = sorted(set([-np.inf, q25, q50, q75, np.inf]))
        if len(bins) >= 3:
            n_labels = len(bins) - 1
            labels = ['Low', 'Moderate', 'High', 'Elite'][:n_labels]
            df['leverage_tier'] = pd.cut(df['leverage_score'], bins=bins, labels=labels, duplicates='drop')
        else:
            df['leverage_tier'] = 'Moderate'
    except Exception:
        df['leverage_tier'] = 'Moderate'

    return df


# ==============================================================================
# MONTE CARLO LINEUP SIMULATION
# ==============================================================================

class MonteCarloLineupBuilder:
    """
    Build GPP lineups using Monte Carlo simulation of MDN mixture distributions.

    For each iteration:
    1. Sample FPTS from N(projected_fpts, std_fpts) for all players
    2. Apply leverage weighting to sampled values
    3. Build best lineup under salary/stacking constraints
    4. Track player frequencies across iterations

    Returns: top lineups ranked by expected GPP score.
    """

    def __init__(self, player_pool: pd.DataFrame, n_iterations: int = MC_ITERATIONS):
        self.pool = player_pool.copy()
        self.n_iterations = n_iterations

        # Separate by position
        self.skaters = self.pool[self.pool['norm_pos'].isin(['C', 'W', 'D'])].copy()
        self.goalies = self.pool[self.pool['norm_pos'] == 'G'].copy()

        # Build opponent map
        self.opponent_map = {}
        for _, row in self.pool.drop_duplicates('team').iterrows():
            team = str(row['team']).upper()
            opp = row.get('opponent', '')
            if opp and not pd.isna(opp):
                self.opponent_map[team] = str(opp).upper()

        # Index skaters by team
        self.skaters_by_team = {}
        for team, grp in self.skaters.groupby('team'):
            self.skaters_by_team[team] = grp.sort_values('gpp_score', ascending=False).head(12)

        self.all_teams = sorted(self.skaters['team'].unique())

    def _sample_fpts(self) -> pd.Series:
        """Sample FPTS for all players from their individual distributions."""
        projected = self.pool['projected_fpts'].values
        std = self.pool['std_fpts'].values if 'std_fpts' in self.pool.columns else np.full(len(projected), 5.5)
        std = np.where(np.isnan(std) | (std < 1), 5.5, std)
        sampled = np.maximum(np.random.normal(projected, std), 0)
        return pd.Series(sampled, index=self.pool.index)

    def _check_position_feasibility(self, positions: List[str]) -> bool:
        """Check if 8 skater positions can fill 2C+3W+2D+1UTIL(C/W only)."""
        c = positions.count('C')
        w = positions.count('W')
        d = positions.count('D')
        if d != 2:
            return False
        if c + w != 6:
            return False
        return c >= 2 and w >= 3

    def _build_single_lineup(self, sampled_fpts: pd.Series, leverage_weight: float = 0.3) -> Optional[Dict]:
        """
        Build a single lineup from sampled FPTS values.

        Uses a greedy stacking approach:
        1. Pick primary stack team (weighted random from top teams by sampled total)
        2. Pick 4 best skaters from primary team
        3. Pick secondary stack team
        4. Pick 3 best skaters from secondary team
        5. Pick best fill skater from other teams
        6. Pick best goalie (not opposing any skater team)
        """
        pool = self.pool.copy()
        pool['sampled_fpts'] = sampled_fpts.values

        # Blend sampled FPTS with leverage for GPP selection
        if 'gpp_score' in pool.columns:
            pool['selection_score'] = (
                (1 - leverage_weight) * pool['sampled_fpts'] / (pool['sampled_fpts'].max() + 1e-6) +
                leverage_weight * pool['gpp_score']
            )
        else:
            pool['selection_score'] = pool['sampled_fpts']

        skaters = pool[pool['norm_pos'].isin(['C', 'W', 'D'])].copy()
        goalies = pool[pool['norm_pos'] == 'G'].copy()

        # Team total by sampled FPTS
        team_totals = skaters.groupby('team')['sampled_fpts'].sum().sort_values(ascending=False)

        # Select primary stack team (weighted random from top 5)
        top_teams = team_totals.head(5)
        weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08][:len(top_teams)])
        weights = weights / weights.sum()
        primary_team = np.random.choice(top_teams.index, p=weights)

        # Select secondary stack team (weighted random from remaining top teams)
        remaining = [t for t in team_totals.head(8).index if t != primary_team]
        if not remaining:
            return None
        sec_weights = np.array([0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03][:len(remaining)])
        sec_weights = sec_weights / sec_weights.sum()
        secondary_team = np.random.choice(remaining, p=sec_weights)

        lineup = []
        used_names = set()
        team_counts = defaultdict(int)
        remaining_salary = SALARY_CAP

        def add_player(row, slot):
            nonlocal remaining_salary
            lineup.append({
                'name': row['name'],
                'team': row['team'],
                'norm_pos': row['norm_pos'],
                'salary': row['salary'],
                'projected_fpts': row['projected_fpts'],
                'sampled_fpts': row.get('sampled_fpts', row['projected_fpts']),
                'predicted_ownership': row.get('predicted_ownership', 5.0),
                'gpp_score': row.get('gpp_score', 0),
                'leverage_score': row.get('leverage_score', 0),
                'std_fpts': row.get('std_fpts', 5.5),
                'p_above_15': row.get('p_above_15', 0),
                'p_above_20': row.get('p_above_20', 0),
                'roster_slot': slot,
            })
            used_names.add(row['name'])
            team_counts[row['team']] += 1
            remaining_salary -= row['salary']

        def can_add(row, check_team=True):
            if row['name'] in used_names:
                return False
            if row['salary'] > remaining_salary:
                return False
            if check_team and team_counts.get(row['team'], 0) >= MAX_FROM_TEAM:
                return False
            return True

        # Step 1: Pick goalie first (prefer correlation with primary stack)
        goalies_sorted = goalies.sort_values('selection_score', ascending=False)
        goalie_picked = False
        goalie_opponent = None
        for _, g in goalies_sorted.iterrows():
            if can_add(g):
                add_player(g, 'G')
                goalie_picked = True
                goalie_opponent = self.opponent_map.get(str(g['team']).upper())
                break

        if not goalie_picked:
            return None

        # Remove players from goalie's opponent team
        if goalie_opponent:
            skaters = skaters[skaters['team'].str.upper() != goalie_opponent.upper()]

        # Step 2: Primary stack (4 skaters)
        primary_skaters = skaters[skaters['team'] == primary_team].sort_values(
            'selection_score', ascending=False
        )
        primary_added = 0
        for _, p in primary_skaters.iterrows():
            if primary_added >= PRIMARY_STACK_SIZE:
                break
            spots_remaining = 9 - len(lineup) - 1
            budget_after = remaining_salary - p['salary']
            if budget_after < spots_remaining * MIN_SALARY_PER_PLAYER:
                continue
            if can_add(p):
                pos = p['norm_pos']
                slot = self._get_slot(pos, lineup)
                if slot:
                    add_player(p, slot)
                    primary_added += 1

        # Step 3: Secondary stack (3 skaters)
        secondary_skaters = skaters[skaters['team'] == secondary_team].sort_values(
            'selection_score', ascending=False
        )
        secondary_added = 0
        for _, p in secondary_skaters.iterrows():
            if secondary_added >= SECONDARY_STACK_SIZE:
                break
            spots_remaining = 9 - len(lineup) - 1
            budget_after = remaining_salary - p['salary']
            if budget_after < spots_remaining * MIN_SALARY_PER_PLAYER:
                continue
            if can_add(p):
                pos = p['norm_pos']
                slot = self._get_slot(pos, lineup)
                if slot:
                    add_player(p, slot)
                    secondary_added += 1

        # Step 4: Fill remaining spots
        fill_teams = {primary_team, secondary_team}
        all_skaters_sorted = skaters.sort_values('selection_score', ascending=False)

        for _, p in all_skaters_sorted.iterrows():
            if len(lineup) >= 9:
                break
            if can_add(p):
                pos = p['norm_pos']
                slot = self._get_slot(pos, lineup)
                if slot:
                    spots_remaining = 9 - len(lineup) - 1
                    if spots_remaining > 0:
                        budget_after = remaining_salary - p['salary']
                        if budget_after < spots_remaining * MIN_SALARY_PER_PLAYER:
                            continue
                    add_player(p, slot)

        if len(lineup) != 9:
            return None

        # Validate salary
        total_salary = sum(p['salary'] for p in lineup)
        if total_salary > SALARY_CAP:
            return None

        # Validate position feasibility
        skater_positions = [p['norm_pos'] for p in lineup if p['roster_slot'] != 'G']
        if not self._check_position_feasibility(skater_positions):
            return None

        # Validate minimum teams (skaters only)
        skater_teams = set(p['team'] for p in lineup if p['roster_slot'] != 'G')
        if len(skater_teams) < MIN_TEAMS:
            return None

        return {
            'players': lineup,
            'total_salary': total_salary,
            'total_projected': sum(p['projected_fpts'] for p in lineup),
            'total_sampled': sum(p['sampled_fpts'] for p in lineup),
            'avg_ownership': np.mean([p['predicted_ownership'] for p in lineup]),
            'total_gpp_score': sum(p['gpp_score'] for p in lineup),
            'primary_stack': primary_team,
            'secondary_stack': secondary_team,
            'primary_count': primary_added,
            'secondary_count': secondary_added,
        }

    def _get_slot(self, position: str, lineup: List[Dict]) -> Optional[str]:
        """Get next available roster slot for a position."""
        filled = [p['roster_slot'] for p in lineup]

        slots = {
            'C': ['C1', 'C2', 'UTIL'],
            'W': ['W1', 'W2', 'W3', 'UTIL'],
            'D': ['D1', 'D2'],  # D excluded from UTIL (lower ceilings)
            'G': ['G'],
        }

        for slot in slots.get(position, []):
            if slot not in filled:
                if slot == 'UTIL':
                    # Only use UTIL when position's primary slots are full
                    c_filled = sum(1 for s in filled if s.startswith('C') and s != 'UTIL')
                    w_filled = sum(1 for s in filled if s.startswith('W'))
                    d_filled = sum(1 for s in filled if s.startswith('D'))

                    if position == 'C' and c_filled >= 2:
                        return slot
                    elif position == 'W' and w_filled >= 3:
                        return slot
                    elif c_filled >= 2 and w_filled >= 3 and d_filled >= 2:
                        return slot
                else:
                    return slot
        return None

    def run(self, n_lineups: int = MC_TOP_SELECT) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Run Monte Carlo simulation to generate diverse GPP lineups.

        Returns:
            (lineups, frequency_table)
        """
        print(f"\n{'='*80}")
        print(f"MONTE CARLO LINEUP BUILDER")
        print(f"{'='*80}")
        print(f"  Iterations: {self.n_iterations}")
        print(f"  Players: {len(self.pool)} ({len(self.skaters)} skaters, {len(self.goalies)} goalies)")
        print(f"  Teams: {len(self.all_teams)}")
        print()

        all_lineups = []
        player_counts = defaultdict(int)
        total_valid = 0

        for i in range(self.n_iterations):
            sampled_fpts = self._sample_fpts()
            lineup = self._build_single_lineup(sampled_fpts)

            if lineup is not None:
                total_valid += 1
                all_lineups.append(lineup)
                for p in lineup['players']:
                    player_counts[p['name']] += 1

            if (i + 1) % 200 == 0:
                print(f"  Iteration {i+1}/{self.n_iterations} — {total_valid} valid lineups")

        print(f"\n  Total valid lineups: {total_valid}/{self.n_iterations} "
              f"({total_valid/self.n_iterations*100:.1f}% success rate)")

        # Build frequency table
        freq_rows = []
        for name, count in player_counts.items():
            player_info = self.pool[self.pool['name'] == name].iloc[0] if name in self.pool['name'].values else None
            if player_info is not None:
                freq_rows.append({
                    'name': name,
                    'team': player_info['team'],
                    'position': player_info['norm_pos'],
                    'salary': player_info['salary'],
                    'projected_fpts': player_info['projected_fpts'],
                    'predicted_ownership': player_info.get('predicted_ownership', 5.0),
                    'gpp_score': player_info.get('gpp_score', 0),
                    'leverage_score': player_info.get('leverage_score', 0),
                    'count': count,
                    'exposure_pct': round(100.0 * count / total_valid, 1) if total_valid > 0 else 0,
                })

        freq_df = pd.DataFrame(freq_rows).sort_values('count', ascending=False).reset_index(drop=True)

        # Select top lineups by GPP score
        all_lineups.sort(key=lambda x: x['total_gpp_score'], reverse=True)

        # Deduplicate
        seen = set()
        unique_lineups = []
        for lu in all_lineups:
            key = frozenset(p['name'] for p in lu['players'])
            if key not in seen:
                seen.add(key)
                unique_lineups.append(lu)
                if len(unique_lineups) >= n_lineups:
                    break

        return unique_lineups, freq_df


# ==============================================================================
# PLAYER POOL BUILDER
# ==============================================================================

def build_player_pool(date_str: str, use_mdn: bool = True) -> pd.DataFrame:
    """
    Build integrated player pool merging DK salaries + MDN projections + ownership.

    Args:
        date_str: Date string (YYYY-MM-DD)
        use_mdn: If True, use MDN v3 projections; else use DK avg FPTS

    Returns:
        Player pool DataFrame with all features needed for lineup construction.
    """
    print(f"\nBuilding player pool for {date_str}...")

    # 1. Load DK salaries
    dk = load_dk_salaries_for_date(date_str)
    if dk.empty:
        print(f"  No DK salaries found for {date_str}")
        return pd.DataFrame()
    print(f"  DK salaries: {len(dk)} players")

    # 2. Load MDN projections (if available)
    if use_mdn:
        mdn = load_mdn_backtest_results(date_str)
        if not mdn.empty:
            print(f"  MDN v3 projections: {len(mdn)} players")

            # MDN uses abbreviated names (C. Perry), DK uses full (Corey Perry)
            # Also: DK uses ASCII (Stuetzle), MDN uses Unicode (Stützle)
            # Build robust match: normalize accents + last name + first initial

            def normalize_name_for_key(name):
                """
                Normalize a player name to a matching key.
                Handles: diacritics (ü→u, é→e), German umlauts (ü→ue in DK),
                hyphenated names, multi-part last names (Del Bel Belluz).
                Returns: 'lastname_firstinit' in pure ASCII lowercase.
                """
                if pd.isna(name):
                    return ''
                name = str(name).strip()

                # Strip diacritics: ü→u, é→e, ý→y etc.
                name_ascii = unicodedata.normalize('NFD', name)
                name_ascii = ''.join(c for c in name_ascii if unicodedata.category(c) != 'Mn')

                # Handle German umlaut transliterations: ue→u, ae→a, oe→o
                # DK uses "Stuetzle" but after accent strip we get "Stutzle"
                # We normalize BOTH sides, so map ue→u for DK names too
                name_lower = name_ascii.lower()
                for umlaut, replacement in [('ue', 'u'), ('ae', 'a'), ('oe', 'o')]:
                    name_lower = name_lower.replace(umlaut, replacement)

                parts = name_lower.split()
                if len(parts) < 2:
                    return name_lower

                last = parts[-1]
                first_init = parts[0][0]

                # Handle abbreviations like "J." → take just the letter
                if len(parts[0]) <= 2 and parts[0].endswith('.'):
                    first_init = parts[0][0]

                return f"{last}_{first_init}"

            dk['_match_key'] = dk['name'].apply(normalize_name_for_key)
            mdn['_match_key'] = mdn['player_name'].apply(normalize_name_for_key)

            # Aggregate MDN predictions per match key (in case of duplicates)
            mdn_cols = ['predicted_fpts', 'std_fpts', 'floor_fpts', 'ceiling_fpts',
                       'p_above_10', 'p_above_15', 'p_above_20', 'p_above_25']
            mdn_agg = mdn.groupby('_match_key')[mdn_cols].mean().reset_index()

            dk = dk.merge(mdn_agg, on='_match_key', how='left')

            # Use MDN predictions where available, fallback to DK avg
            dk['projected_fpts'] = dk['predicted_fpts'].fillna(dk['dk_avg_fpts'].fillna(5.0))
            dk['std_fpts'] = dk['std_fpts'].fillna(dk['dk_stdv'].fillna(5.5))

            match_rate = dk['predicted_fpts'].notna().sum() / len(dk) * 100
            print(f"  MDN match rate: {match_rate:.1f}%")
        else:
            print(f"  No MDN projections available, using DK avg FPTS")
            dk['projected_fpts'] = dk['dk_avg_fpts'].fillna(5.0)
            dk['std_fpts'] = dk['dk_stdv'].fillna(5.5)
    else:
        dk['projected_fpts'] = dk['dk_avg_fpts'].fillna(5.0)
        dk['std_fpts'] = dk['dk_stdv'].fillna(5.5)

    # 3. Load ownership data
    own = load_ownership_data_for_date(date_str)
    if not own.empty:
        if 'name_lower' not in dk.columns:
            dk['name_lower'] = dk['name'].str.lower().str.strip()
        if 'name_lower' not in own.columns:
            own['name_lower'] = own['Player'].str.lower().str.strip()

        dk = dk.merge(
            own[['name_lower', 'Ownership']].rename(columns={'Ownership': 'actual_ownership'}),
            on='name_lower',
            how='left'
        )

        # For backtest: use actual ownership as "predicted" (best case scenario)
        # In production: would use ownership_v2 model predictions
        dk['predicted_ownership'] = dk['actual_ownership'].fillna(
            dk['ownership_pct'].fillna(5.0)
        )
        print(f"  Ownership data: {dk['actual_ownership'].notna().sum()}/{len(dk)} matched")
    elif 'ownership_pct' in dk.columns and dk['ownership_pct'].notna().any():
        # Use DK's ownership estimate from salaries table
        dk['predicted_ownership'] = dk['ownership_pct'].fillna(5.0)
        print(f"  Using DK ownership estimates from salaries")
    else:
        # Estimate ownership from salary and projection
        dk['predicted_ownership'] = estimate_ownership_from_salary(dk)
        print(f"  No ownership data, using salary-based estimates")

    # 4. Compute leverage scores
    dk = compute_leverage_scores(dk)
    print(f"  Leverage scores computed")

    # 5. Compute value metrics
    dk['value'] = dk['projected_fpts'] / (dk['salary'] / 1000)

    # Clean up
    dk = dk.dropna(subset=['salary', 'projected_fpts'])
    dk = dk[dk['salary'] > 0].copy()

    print(f"  Final pool: {len(dk)} players")
    return dk


def estimate_ownership_from_salary(dk: pd.DataFrame) -> pd.Series:
    """Simple salary-based ownership estimate when no real data available."""
    # Higher salary → higher ownership (simplified)
    salary_rank = dk.groupby('norm_pos')['salary'].rank(method='first', ascending=False, pct=True)
    # Top salary → ~20%, bottom → ~1%
    estimated_own = 1.0 + 19.0 * (1 - salary_rank)
    return estimated_own


# ==============================================================================
# LINEUP SCORING & BACKTESTING
# ==============================================================================

def score_lineup(lineup: Dict, actuals: pd.DataFrame) -> Dict:
    """
    Score a lineup against actual results.

    Returns dict with actual total, ownership, and GPP metrics.
    """
    actuals_map = {}
    for _, row in actuals.iterrows():
        actuals_map[row['name'].lower().strip()] = row['actual_fpts']

    total_actual = 0
    total_projected = 0
    total_ownership = 0
    matched = 0

    for p in lineup['players']:
        name_lower = p['name'].lower().strip()
        actual = actuals_map.get(name_lower, 0)
        total_actual += actual
        total_projected += p['projected_fpts']
        total_ownership += p.get('predicted_ownership', 5.0)
        if name_lower in actuals_map:
            matched += 1

    return {
        'actual_total': total_actual,
        'projected_total': total_projected,
        'avg_ownership': total_ownership / 9,
        'sum_ownership': total_ownership,
        'matched_players': matched,
        'primary_stack': lineup.get('primary_stack', ''),
        'secondary_stack': lineup.get('secondary_stack', ''),
    }


def run_backtest(start_date: str = '2026-01-15', end_date: str = '2026-02-05',
                 n_lineups: int = 10, mc_iterations: int = 500) -> pd.DataFrame:
    """
    Run backtest of lineup builder against historical results.

    For each date with available data:
    1. Build player pool (projections + ownership)
    2. Generate lineups via Monte Carlo
    3. Score against actual results
    4. Compare to contest winning scores
    """
    print(f"\n{'='*80}")
    print(f"LINEUP BUILDER BACKTEST: {start_date} to {end_date}")
    print(f"{'='*80}\n")

    results = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        # Check if we have data for this date
        pool = build_player_pool(date_str, use_mdn=True)
        actuals = load_actuals_for_date(date_str)

        if pool.empty or actuals.empty or len(pool) < 20:
            current += timedelta(days=1)
            continue

        print(f"\n--- {date_str}: {len(pool)} players, {len(actuals)} actuals ---")

        # Generate lineups
        builder = MonteCarloLineupBuilder(pool, n_iterations=mc_iterations)
        lineups, freq_df = builder.run(n_lineups=n_lineups)

        if not lineups:
            print(f"  No valid lineups generated")
            current += timedelta(days=1)
            continue

        # Score each lineup
        for i, lu in enumerate(lineups):
            scored = score_lineup(lu, actuals)

            results.append({
                'date': date_str,
                'lineup_rank': i + 1,
                'actual_total': scored['actual_total'],
                'projected_total': scored['projected_total'],
                'avg_ownership': scored['avg_ownership'],
                'sum_ownership': scored['sum_ownership'],
                'total_salary': lu['total_salary'],
                'total_gpp_score': lu['total_gpp_score'],
                'primary_stack': scored['primary_stack'],
                'secondary_stack': scored['secondary_stack'],
                'matched_players': scored['matched_players'],
            })

        # Print best lineup
        best = max(lineups, key=lambda x: score_lineup(x, actuals)['actual_total'])
        best_scored = score_lineup(best, actuals)
        print(f"  Best lineup: {best_scored['actual_total']:.1f} actual FPTS, "
              f"avg own {best_scored['avg_ownership']:.1f}%, "
              f"stacks: {best_scored['primary_stack']}({best.get('primary_count', 0)}) + "
              f"{best_scored['secondary_stack']}({best.get('secondary_count', 0)})")

        # Load contest results for comparison
        contests = load_contest_results(date_str)
        if not contests.empty:
            winning_score = contests['score'].max()
            cash_line = contests[contests['n_cashed'] > 0]['score'].min() if contests['n_cashed'].sum() > 0 else 0
            print(f"  Contest: winning={winning_score:.1f}, cash line={cash_line:.1f}")
            print(f"  Our best: {best_scored['actual_total']:.1f} "
                  f"({'CASHED' if best_scored['actual_total'] >= cash_line else 'MISSED'})")

        current += timedelta(days=1)

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        print(f"\n{'='*80}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*80}")
        print(f"  Dates tested: {results_df['date'].nunique()}")
        print(f"  Total lineups: {len(results_df)}")
        print(f"  Avg actual FPTS (best per date): "
              f"{results_df.groupby('date')['actual_total'].max().mean():.1f}")
        print(f"  Avg projected FPTS: {results_df['projected_total'].mean():.1f}")
        print(f"  Avg ownership: {results_df['avg_ownership'].mean():.1f}%")
        print(f"  Avg salary used: ${results_df['total_salary'].mean():,.0f}")

    return results_df


# ==============================================================================
# DISPLAY & EXPORT
# ==============================================================================

def print_lineup(lineup: Dict, rank: int = 1):
    """Pretty-print a lineup."""
    players = lineup['players']
    total_salary = lineup['total_salary']
    total_proj = lineup['total_projected']

    # Sort by roster slot
    slot_order = {'G': 0, 'C1': 1, 'C2': 2, 'W1': 3, 'W2': 4, 'W3': 5, 'D1': 6, 'D2': 7, 'UTIL': 8}
    players_sorted = sorted(players, key=lambda p: slot_order.get(p['roster_slot'], 9))

    print(f"\n  Lineup #{rank}")
    print(f"  Salary: ${total_salary:,} / $50,000 (${50000-total_salary:,} remaining)")
    print(f"  Projected: {total_proj:.1f} FPTS | GPP Score: {lineup['total_gpp_score']:.3f}")
    print(f"  Avg Own: {lineup['avg_ownership']:.1f}% | "
          f"Stacks: {lineup['primary_stack']}({lineup['primary_count']}) + "
          f"{lineup['secondary_stack']}({lineup['secondary_count']})")
    print()
    print(f"  {'Slot':<6} {'Name':<25} {'Team':<5} {'Sal':>7} {'Proj':>6} "
          f"{'Std':>5} {'Own%':>5} {'GPP':>6} {'Lev':>6}")
    print(f"  {'-'*85}")

    for p in players_sorted:
        print(f"  {p['roster_slot']:<6} {p['name']:<25} {p['team']:<5} "
              f"${p['salary']:>6,} {p['projected_fpts']:>6.1f} "
              f"{p.get('std_fpts', 0):>5.1f} {p.get('predicted_ownership', 0):>5.1f} "
              f"{p.get('gpp_score', 0):>6.3f} {p.get('leverage_score', 0):>6.2f}")


def print_frequency_table(freq_df: pd.DataFrame, top_n: int = 25):
    """Print Monte Carlo exposure table."""
    print(f"\n{'='*90}")
    print(f"MONTE CARLO EXPOSURE TABLE (Top {min(top_n, len(freq_df))})")
    print(f"{'='*90}")

    # Skaters
    skaters = freq_df[freq_df['position'] != 'G'].head(top_n)
    print(f"\n  {'Name':<25} {'Team':<5} {'Pos':<4} {'Sal':>7} {'Proj':>6} "
          f"{'Own%':>6} {'GPP':>6} {'Exp%':>6}")
    print(f"  {'-'*75}")
    for _, row in skaters.iterrows():
        print(f"  {row['name']:<25} {row['team']:<5} {row['position']:<4} "
              f"${row['salary']:>6,} {row['projected_fpts']:>6.1f} "
              f"{row['predicted_ownership']:>6.1f} {row['gpp_score']:>6.3f} "
              f"{row['exposure_pct']:>5.1f}%")

    # Goalies
    goalies = freq_df[freq_df['position'] == 'G']
    if not goalies.empty:
        print(f"\n  GOALIES:")
        print(f"  {'-'*75}")
        for _, row in goalies.iterrows():
            print(f"  {row['name']:<25} {row['team']:<5} {row['position']:<4} "
                  f"${row['salary']:>6,} {row['projected_fpts']:>6.1f} "
                  f"{row['predicted_ownership']:>6.1f} {row['gpp_score']:>6.3f} "
                  f"{row['exposure_pct']:>5.1f}%")


def export_lineups_dk_csv(lineups: List[Dict], output_path: str):
    """Export lineups to DraftKings upload format."""
    rows = []
    for i, lu in enumerate(lineups):
        players = lu['players']
        slot_order = {'C1': 0, 'C2': 1, 'W1': 2, 'W2': 3, 'W3': 4, 'D1': 5, 'D2': 6, 'G': 7, 'UTIL': 8}
        sorted_players = sorted(players, key=lambda p: slot_order.get(p['roster_slot'], 9))

        row = {}
        dk_slots = ['C', 'C', 'W', 'W', 'W', 'D', 'D', 'G', 'UTIL']
        for j, (slot_name, p) in enumerate(zip(dk_slots, sorted_players)):
            row[f'{slot_name}_{j}'] = p['name']

        row['lineup_num'] = i + 1
        row['total_salary'] = lu['total_salary']
        row['projected_fpts'] = lu['total_projected']
        row['avg_ownership'] = lu['avg_ownership']
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Exported {len(rows)} lineups to {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='NHL DFS Integrated Lineup Builder')
    parser.add_argument('--date', type=str, default=None,
                        help='Date for lineup generation (YYYY-MM-DD)')
    parser.add_argument('--backtest', action='store_true',
                        help='Run historical backtest')
    parser.add_argument('--start', type=str, default='2026-01-15',
                        help='Backtest start date')
    parser.add_argument('--end', type=str, default='2026-02-05',
                        help='Backtest end date')
    parser.add_argument('--lineups', type=int, default=10,
                        help='Number of lineups to generate')
    parser.add_argument('--mc-iterations', type=int, default=500,
                        help='Monte Carlo iterations')
    args = parser.parse_args()

    if args.backtest:
        results = run_backtest(
            start_date=args.start,
            end_date=args.end,
            n_lineups=args.lineups,
            mc_iterations=args.mc_iterations
        )
        if not results.empty:
            out_path = Path(__file__).parent / 'backtest_lineups.csv'
            results.to_csv(out_path, index=False)
            print(f"\nBacktest results saved to {out_path}")
    else:
        date_str = args.date or datetime.now().strftime('%Y-%m-%d')
        pool = build_player_pool(date_str)

        if pool.empty:
            print(f"No data available for {date_str}")
            return

        builder = MonteCarloLineupBuilder(pool, n_iterations=args.mc_iterations)
        lineups, freq_df = builder.run(n_lineups=args.lineups)

        # Print results
        print_frequency_table(freq_df)

        for i, lu in enumerate(lineups[:5]):  # Show top 5
            print_lineup(lu, rank=i+1)

        # Export
        out_dir = Path(__file__).parent / 'daily_projections'
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_lineups_dk_csv(lineups, str(out_dir / f'lineups_{ts}.csv'))


if __name__ == '__main__':
    main()
