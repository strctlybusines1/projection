"""
IceEdge DFS — Flask Backend

Features:
    - Login/register auth wall
    - Lineup review dashboard (40 ranked SE candidates)
    - Retro hockey rink aesthetic

Run:
    cd website/
    pip install flask pandas werkzeug
    python app.py
"""

import os
import re
import json
import sqlite3
import glob
from datetime import datetime
from pathlib import Path
from functools import wraps
from collections import Counter

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, g
)
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-production-to-a-random-string')

# Path to your projection directory
PROJECTION_DIR = Path(__file__).resolve().parent.parent
DAILY_PROJ_DIR = PROJECTION_DIR / "daily_projections"
DB_PATH = Path(__file__).resolve().parent / "users.db"


# ================================================================
#  Database
# ================================================================

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db:
        db.close()


def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            is_admin INTEGER DEFAULT 0
        )
    """)
    db.commit()
    db.close()


# ================================================================
#  Auth
# ================================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not username or not password:
            flash('Username and password required', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
        elif password != confirm:
            flash('Passwords do not match', 'error')
        else:
            db = get_db()
            try:
                db.execute(
                    'INSERT INTO users (username, password_hash) VALUES (?, ?)',
                    (username, generate_password_hash(password))
                )
                db.commit()
                flash('Account created! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already taken', 'error')
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ================================================================
#  Data Loading
# ================================================================

def _available_dates():
    if not DAILY_PROJ_DIR.exists():
        return []
    dates = set()
    for f in DAILY_PROJ_DIR.glob('*NHL_se_lineups.json'):
        match = re.match(r'(\d{2})_(\d{2})_(\d{2})NHL', f.name)
        if match:
            m, d, y = match.groups()
            dates.add(f"20{y}-{m}-{d}")
    for f in DAILY_PROJ_DIR.glob('*NHLprojections_*.csv'):
        if '_lineups' in f.name:
            continue
        match = re.match(r'(\d{2})_(\d{2})_(\d{2})NHL', f.name)
        if match:
            m, d, y = match.groups()
            dates.add(f"20{y}-{m}-{d}")
    return sorted(dates, reverse=True)


def _load_actuals_for_date(date_str):
    """Load actual FPTS from contest CSVs matching the given date.

    Contest filenames use M.DD.YY format (e.g., $5SE_NHL1.28.26.csv).
    Returns dict of {player_name: actual_fpts}.
    """
    contests_dir = PROJECTION_DIR / "contests"
    if not contests_dir.exists():
        return {}

    dt = datetime.strptime(date_str, '%Y-%m-%d')
    target_m = dt.month
    target_d = dt.day
    target_y = dt.year % 100  # 2-digit year

    actuals = {}
    for f in contests_dir.glob('*.csv'):
        # Extract date from filename: look for M.DD.YY or M.D.YY pattern
        match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{2})', f.name)
        if not match:
            continue
        fm, fd, fy = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if fm != target_m or fd != target_d or fy != target_y:
            continue

        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            if 'Player' in df.columns and 'FPTS' in df.columns:
                for _, row in df.iterrows():
                    name = str(row['Player']).strip()
                    try:
                        fpts = float(row['FPTS'])
                    except (ValueError, TypeError):
                        continue
                    if name and name != 'nan':
                        actuals[name] = fpts
        except Exception:
            continue

    return actuals


def _merge_actuals(lineup_data, actuals):
    """Merge actual FPTS into lineup data dict (modifies in place)."""
    if not actuals or not lineup_data:
        return lineup_data

    for lineup in lineup_data.get('lineups', []):
        actuals_found = []
        for player in lineup.get('players', []):
            name = player.get('name', '')
            if name in actuals:
                player['actual_fpts'] = actuals[name]
                actuals_found.append(actuals[name])
            else:
                player['actual_fpts'] = None
        if len(actuals_found) == len(lineup.get('players', [])):
            lineup['total_actual'] = round(sum(actuals_found), 1)

    return lineup_data


def _load_se_lineups(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    filepath = DAILY_PROJ_DIR / f"{dt.strftime('%m_%d_%y')}NHL_se_lineups.json"
    if filepath.exists():
        data = json.loads(filepath.read_text())
        actuals = _load_actuals_for_date(date_str)
        return _merge_actuals(data, actuals)
    return None


def _build_lineups_from_csv(date_str):
    """Build lineup data from projection + lineup CSVs when JSON doesn't exist."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    prefixes = [
        f"{dt.month:02d}_{dt.day:02d}_{dt.strftime('%y')}",
        f"{dt.month}_{dt.day}_{dt.strftime('%y')}",
    ]

    proj_file = None
    for f in sorted(DAILY_PROJ_DIR.glob('*NHLprojections_*.csv')):
        if '_lineups' in f.name:
            continue
        for prefix in prefixes:
            if f.name.startswith(prefix):
                proj_file = f

    if not proj_file:
        return None

    proj = pd.read_csv(proj_file)
    if 'dk_id' not in proj.columns:
        return None

    lineup_files = []
    for f in sorted(DAILY_PROJ_DIR.glob('*_lineups.csv')):
        for prefix in prefixes:
            if f.name.startswith(prefix):
                lineup_files.append(f)

    if not lineup_files:
        return None

    lineup_df = pd.read_csv(lineup_files[-1])
    slots = list(lineup_df.columns)
    lineups_data = []

    for idx, row in lineup_df.iterrows():
        players = []
        for slot in slots:
            dk_id = str(row[slot])
            match = proj[proj['dk_id'].astype(str) == dk_id]
            if match.empty:
                continue
            p = match.iloc[0]
            players.append({
                'name': p.get('name', ''),
                'team': p.get('team', ''),
                'position': p.get('position', ''),
                'roster_slot': slot,
                'salary': int(p.get('salary', 0)),
                'projected_fpts': round(float(p.get('projected_fpts', 0)), 1),
                'floor': round(float(p.get('floor', 0)), 1) if pd.notna(p.get('floor')) else None,
                'ceiling': round(float(p.get('ceiling', 0)), 1) if pd.notna(p.get('ceiling')) else None,
                'predicted_ownership': round(float(p.get('predicted_ownership', 0)), 1) if pd.notna(p.get('predicted_ownership')) else None,
            })

        if len(players) == 9:
            total_salary = sum(p['salary'] for p in players)
            total_proj = sum(p['projected_fpts'] for p in players)
            total_own = sum(p['predicted_ownership'] or 0 for p in players)
            skater_teams = [p['team'] for p in players if p['position'] != 'G']
            team_counts = Counter(skater_teams)
            stacks = [f"{t}{c}" for t, c in team_counts.most_common() if c >= 2]
            goalie = next((p for p in players if p['position'] == 'G'), None)

            lineups_data.append({
                'rank': idx + 1,
                'players': players,
                'total_salary': total_salary,
                'total_projected': round(total_proj, 1),
                'total_ownership': round(total_own, 1),
                'total_actual': None,
                'goalie': goalie['name'] if goalie else None,
                'stacks': stacks,
                'scores': {},
            })

    if not lineups_data:
        return None

    data = {
        'slate_date': date_str,
        'n_candidates': len(lineups_data),
        'lineups': lineups_data,
    }
    actuals = _load_actuals_for_date(date_str)
    return _merge_actuals(data, actuals)


# ================================================================
#  Routes
# ================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    date = request.args.get('date')
    dates = _available_dates()
    if not date and dates:
        date = dates[0]

    lineup_data = None
    if date:
        lineup_data = _load_se_lineups(date)
        if not lineup_data:
            lineup_data = _build_lineups_from_csv(date)

    lineups = lineup_data.get('lineups', []) if lineup_data else []

    return render_template('dashboard.html',
                         lineups=lineups,
                         slate_date=date,
                         dates=dates,
                         n_lineups=len(lineups),
                         username=session.get('username'))


@app.route('/api/lineups')
@login_required
def api_lineups():
    date = request.args.get('date')
    if not date:
        dates = _available_dates()
        date = dates[0] if dates else None
    if not date:
        return jsonify({'error': 'No date', 'lineups': []})
    lineup_data = _load_se_lineups(date)
    if not lineup_data:
        lineup_data = _build_lineups_from_csv(date)
    return jsonify(lineup_data or {'error': 'No lineups', 'lineups': []})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    init_db()
    print(f"\n  IceEdge DFS Website")
    print(f"  {'─' * 40}")
    print(f"  Projection dir: {DAILY_PROJ_DIR}")
    print(f"  User database:  {DB_PATH}")
    print(f"\n  http://localhost:{args.port}\n")
    app.run(debug=True, port=args.port)
