"""
Local NHL DFS Dashboard - Flask backend.

Serves: GET / (dashboard), GET /api/odds, /api/projections, /api/lines, /api/lineup.
Odds from The Odds API (if ODDS_API_KEY set) or latest Vegas CSV. Load .env from projection/.
"""

import os
import sys
import json
import time
from pathlib import Path

# Project root = projection/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from projection/ so ODDS_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import pandas as pd
import requests
from flask import Flask, send_from_directory, jsonify

from config import DAILY_PROJECTIONS_DIR, VEGAS_DIR, BACKTESTS_DIR

app = Flask(__name__, static_folder="static", template_folder="templates")

# In-memory cache for odds API: (data, expires_at)
_odds_cache = None
_odds_cache_expires = 0
ODDS_CACHE_MINUTES = 12


def _projections_dir():
    return PROJECT_ROOT / DAILY_PROJECTIONS_DIR


def _vegas_dir():
    return PROJECT_ROOT / VEGAS_DIR


def _latest_mae_path():
    """Path to backtests/latest_mae.json."""
    return PROJECT_ROOT / BACKTESTS_DIR / "latest_mae.json"


def _latest_projections_path():
    """Latest CSV matching *NHLprojections_*.csv, excluding *_lineups.csv."""
    d = _projections_dir()
    if not d.exists():
        return None
    files = [f for f in d.glob("*NHLprojections_*.csv") if "_lineups" not in f.name]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def _latest_lineup_path():
    """Latest *_lineups.csv."""
    d = _projections_dir()
    if not d.exists():
        return None
    files = list(d.glob("*_lineups.csv"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def _parse_slate_date_from_lineup_filename(path):
    """Parse slate date YYYY-MM-DD from lineup filename like 01_29_26NHLprojections_20260129_132647_lineups.csv."""
    try:
        name = path.name
        if "_lineups.csv" not in name:
            return None
        # MM_DD_YY at start (e.g. 01_29_26)
        parts = name.split("NHLprojections_")
        if not parts:
            return None
        prefix = parts[0]
        if "_" not in prefix or len(prefix) < 8:
            return None
        mm, dd, yy = prefix.split("_")[:3]
        yy_int = int(yy)
        year = 2000 + yy_int if yy_int < 100 else yy_int
        month, day = int(mm), int(dd)
        from datetime import date
        d = date(year, month, day)
        return d.strftime("%Y-%m-%d")
    except (ValueError, IndexError, TypeError):
        return None


def _last_night_lineup_path():
    """Return (path, slate_date) for the most recent past-slate lineup file, or (None, None)."""
    d = _projections_dir()
    if not d.exists():
        return None, None
    from datetime import date
    today = date.today().isoformat()
    files = list(d.glob("*_lineups.csv"))
    past = []
    for f in files:
        slate = _parse_slate_date_from_lineup_filename(f)
        if slate and slate < today:
            past.append((f, slate))
    if not past:
        return None, None
    # Most recent slate date, then latest mtime
    past.sort(key=lambda x: (x[1], x[0].stat().st_mtime), reverse=True)
    return past[0][0], past[0][1]


def _projections_path_for_slate_date(slate_date):
    """Return path to a projections CSV for slate_date (YYYY-MM-DD), or None."""
    try:
        from datetime import datetime
        dt = datetime.strptime(slate_date, "%Y-%m-%d")
        prefix = dt.strftime("%m") + "_" + dt.strftime("%d") + "_" + dt.strftime("%y")
    except ValueError:
        return None
    d = _projections_dir()
    if not d.exists():
        return None
    files = [f for f in d.glob(f"*{prefix}*NHLprojections_*.csv") if "_lineups" not in f.name]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def _backtest_xlsx_for_slate(slate_date):
    """Find backtests/M.DD.YY_nhl_backtest.xlsx for a given YYYY-MM-DD slate date."""
    try:
        from datetime import datetime as _dt
        dt = _dt.strptime(slate_date, "%Y-%m-%d")
        # File format: 1.28.26_nhl_backtest.xlsx (no zero-padding on month)
        tag = f"{dt.month}.{dt.day:02d}.{dt.strftime('%y')}"
    except (ValueError, TypeError):
        return None
    bt_dir = PROJECT_ROOT / BACKTESTS_DIR
    if not bt_dir.exists():
        return None
    candidate = bt_dir / f"{tag}_nhl_backtest.xlsx"
    if candidate.exists():
        return candidate
    # Fallback: try zero-padded month
    tag2 = f"{dt.month:02d}.{dt.day:02d}.{dt.strftime('%y')}"
    candidate2 = bt_dir / f"{tag2}_nhl_backtest.xlsx"
    return candidate2 if candidate2.exists() else None


def _load_backtest_actuals(xlsx_path):
    """Load the Projection sheet from a backtest xlsx and return a dict keyed by lowercase name.

    Each value has keys: actual_fpts, actual_ownership (as percentage, e.g. 18.2).
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name="Projection")
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        actual = float(r["actual"]) if pd.notna(r.get("actual")) else None
        own_raw = r.get("own")
        own_pct = round(float(own_raw) * 100, 1) if pd.notna(own_raw) and own_raw != 0 else None
        out[name.lower()] = {"actual_fpts": actual, "actual_ownership": own_pct}
    return out


def _latest_lines_path():
    """Latest lines_*.json or lines_{today}.json."""
    d = _projections_dir()
    if not d.exists():
        return None
    files = list(d.glob("lines_*.json"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def _latest_vegas_path():
    """Latest Vegas*.csv or VegasNHL*.csv."""
    d = _vegas_dir()
    if not d.exists():
        return None
    files = list(d.glob("Vegas*.csv")) + list(d.glob("VegasNHL*.csv"))
    files = list(dict.fromkeys(files))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def _ml_to_implied_team_total(home_ml, away_ml, game_total):
    """Derive per-team implied totals from game total + moneylines.

    Convert American moneylines to implied win probabilities, normalize
    to remove the vig, then multiply each side's probability by the
    game total.  Returns (away_team_total, home_team_total) rounded to
    one decimal place, or (None, None) when any input is missing.
    """
    if home_ml is None or away_ml is None or game_total is None:
        return None, None

    def _ml_to_prob(ml):
        ml = float(ml)
        if ml > 0:
            return 100.0 / (ml + 100.0)
        else:
            return (-ml) / (-ml + 100.0)

    home_prob = _ml_to_prob(home_ml)
    away_prob = _ml_to_prob(away_ml)
    total_prob = home_prob + away_prob  # > 1 due to vig
    home_fair = home_prob / total_prob
    away_fair = away_prob / total_prob
    home_tt = round(home_fair * game_total, 1)
    away_tt = round(away_fair * game_total, 1)
    return away_tt, home_tt


def _fetch_odds_api():
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        return None, "no_key"
    url = (
        "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
        "?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey=" + key
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data, None
    except Exception as e:
        return None, str(e)


def _normalize_odds_api(events):
    """Convert Odds API response to list of { matchup, game_total, home_ml, away_ml, spread_home, spread_away, commence_time }."""
    out = []
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        # API uses full names; optionally shorten (e.g. "Boston Bruins" -> "BOS")
        matchup = f"{away} @ {home}"
        game_total = None
        home_ml = away_ml = None
        spread_home = spread_away = None
        for bm in ev.get("bookmakers", [])[:1]:  # first bookmaker (e.g. DraftKings)
            for m in bm.get("markets", []):
                if m.get("key") == "totals" and m.get("outcomes"):
                    for o in m["outcomes"]:
                        if o.get("name") == "Over":
                            game_total = o.get("point")
                            break
                elif m.get("key") == "h2h" and m.get("outcomes"):
                    for o in m["outcomes"]:
                        if o.get("name") == home:
                            home_ml = o.get("price")
                        elif o.get("name") == away:
                            away_ml = o.get("price")
                elif m.get("key") == "spreads" and m.get("outcomes"):
                    for o in m["outcomes"]:
                        if o.get("name") == home:
                            spread_home = o.get("point")
                        elif o.get("name") == away:
                            spread_away = o.get("point")
        away_tt, home_tt = _ml_to_implied_team_total(home_ml, away_ml, game_total)
        out.append({
            "matchup": matchup,
            "game_total": game_total,
            "away_team_total": away_tt,
            "home_team_total": home_tt,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "spread_home": spread_home,
            "spread_away": spread_away,
            "commence_time": ev.get("commence_time"),
        })
    # Sort by game_total desc (nulls last)
    out.sort(key=lambda x: (x["game_total"] is None, -(x["game_total"] or 0)))
    return out


def _normalize_vegas_csv(path):
    """Read Vegas CSV and return same shape as odds API."""
    df = pd.read_csv(path)
    if df.empty:
        return []
    # CSV has team, opp, moneyline, spread, game_total (each game two rows)
    if "team" not in df.columns or "opp" not in df.columns:
        return []
    total_col = None
    for c in ["game_total", "total", "Total"]:
        if c in df.columns:
            total_col = c
            break
    if not total_col:
        return []
    games = {}
    for _, row in df.iterrows():
        t, o = str(row["team"]).strip(), str(row["opp"]).strip()
        key = tuple(sorted([t, o]))
        if key not in games:
            away, home = key[0], key[1]
            games[key] = {"away": away, "home": home, "rows": []}
        games[key]["rows"].append(row)
    out = []
    for key, g in games.items():
        rows = g["rows"]
        away, home = g["away"], g["home"]
        # Find which row is away perspective (team==away) and which is home
        away_row = next((r for _, r in enumerate(rows) if r["team"] == away), None)
        home_row = next((r for _, r in enumerate(rows) if r["team"] == home), None)
        game_total = float(away_row[total_col]) if away_row is not None and pd.notna(away_row.get(total_col)) else None
        away_ml = int(away_row["moneyline"]) if away_row is not None and pd.notna(away_row.get("moneyline")) else None
        home_ml = int(home_row["moneyline"]) if home_row is not None and pd.notna(home_row.get("moneyline")) else None
        spread_away = float(away_row["spread"]) if away_row is not None and pd.notna(away_row.get("spread")) else None
        spread_home = float(home_row["spread"]) if home_row is not None and pd.notna(home_row.get("spread")) else None
        away_tt, home_tt = _ml_to_implied_team_total(home_ml, away_ml, game_total)
        out.append({
            "matchup": f"{away} @ {home}",
            "game_total": game_total,
            "away_team_total": away_tt,
            "home_team_total": home_tt,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "spread_home": spread_home,
            "spread_away": spread_away,
            "commence_time": None,
        })
    out.sort(key=lambda x: (x["game_total"] is None, -(x["game_total"] or 0)))
    return out


@app.route("/api/mae")
def api_mae():
    """Return latest overall, skater, and goalie MAE from backtests/latest_mae.json."""
    path = _latest_mae_path()
    if not path.exists():
        return jsonify({
            "overall_mae": None,
            "skater_mae": None,
            "goalie_mae": None,
            "updated": None,
            "error": "No MAE file. Run backtest.py (e.g. python backtest.py --players 75) or slate backtests.",
        })
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({
            "overall_mae": data.get("overall_mae"),
            "skater_mae": data.get("skater_mae"),
            "goalie_mae": data.get("goalie_mae"),
            "updated": data.get("updated"),
            "error": None,
        })
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({
            "overall_mae": None,
            "skater_mae": None,
            "goalie_mae": None,
            "updated": None,
            "error": str(e),
        })


@app.route("/api/odds")
def api_odds():
    global _odds_cache, _odds_cache_expires
    now = time.time()
    if _odds_cache is not None and now < _odds_cache_expires:
        return jsonify(_odds_cache)
    data, err = _fetch_odds_api()
    if data is not None:
        normalized = _normalize_odds_api(data)
        _odds_cache = {"source": "api", "games": normalized, "updated": now}
        _odds_cache_expires = now + ODDS_CACHE_MINUTES * 60
        return jsonify(_odds_cache)
    # Fallback to Vegas CSV; include api_error so the UI can show why API wasn't used
    path = _latest_vegas_path()
    if path:
        normalized = _normalize_vegas_csv(path)
        payload = {"source": "vegas_csv", "games": normalized, "updated": now}
        if err:
            payload["api_error"] = err
        _odds_cache = payload
        _odds_cache_expires = now + 60  # cache 1 min for CSV
        return jsonify(_odds_cache)
    return jsonify({"source": "none", "games": [], "error": err or "No Vegas file or API key"})


@app.route("/api/projections")
def api_projections():
    path = _latest_projections_path()
    if not path:
        return jsonify({"projections": [], "error": "No projections file. Run main.py."})
    df = pd.read_csv(path)
    cols = ["name", "team", "position", "salary", "projected_fpts", "value", "predicted_ownership", "ownership_tier", "leverage_score", "player_type"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df = df.where(pd.notna(df), None)
    return jsonify({"projections": df.to_dict(orient="records")})


@app.route("/api/lines")
def api_lines():
    path = _latest_lines_path()
    if not path:
        return jsonify({"lines": {}, "error": "No lines file. Run main.py with --stacks."})
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({"lines": data})


# DraftKings NHL Classic slot order: C, C, W, W, W, D, D, G, UTIL
DK_SLOT_ORDER = ["C", "C", "W", "W", "W", "D", "D", "G", "UTIL"]


def _position_to_slot_type(pos):
    """Map projection position to slot type (C, W, D, G) for correct slot assignment."""
    if pos is None or pd.isna(pos):
        return None
    p = str(pos).strip().upper()
    if p == "G":
        return "G"
    if p == "C":
        return "C"
    if p in ("LW", "RW", "W", "L", "R"):
        return "W"
    if p == "D":
        return "D"
    return None


def _assign_slots_by_position(lineup_out):
    """
    Reassign roster slots so the goalie gets G (not C) and skaters fill
    C/W/D/UTIL correctly.  Handles CSVs where columns don't match DK slot
    order and lineups with more of one position than there are natural slots
    (e.g. 4 centers → 2C + overflow into W/UTIL).
    """
    # Separate goalie from skaters
    goalie = None
    skaters = []
    for rec in lineup_out:
        if _position_to_slot_type(rec.get("position")) == "G" and goalie is None:
            goalie = rec
        else:
            skaters.append(rec)

    # DK slot order for 8 skater slots: C, C, W, W, W, D, D, UTIL
    SKATER_SLOTS = ["C", "C", "W", "W", "W", "D", "D", "UTIL"]

    # Greedily assign: try to put each skater in its natural slot first
    slot_counts = {"C": 0, "W": 0, "D": 0}
    slot_limits = {"C": 2, "W": 3, "D": 2}
    assigned = []
    overflow = []

    for sk in skaters:
        st = _position_to_slot_type(sk.get("position"))
        if st and st in slot_counts and slot_counts[st] < slot_limits[st]:
            sk["slot"] = st
            slot_counts[st] += 1
            assigned.append(sk)
        else:
            overflow.append(sk)

    # Fill remaining empty natural slots with overflow, then UTIL
    for st in ["C", "W", "D"]:
        while slot_counts[st] < slot_limits[st] and overflow:
            p = overflow.pop(0)
            p["slot"] = st
            slot_counts[st] += 1
            assigned.append(p)

    # First remaining overflow player becomes UTIL
    if overflow:
        overflow[0]["slot"] = "UTIL"
        assigned.append(overflow.pop(0))

    # Any further overflow (shouldn't happen with 8 skaters) gets appended
    for p in overflow:
        p["slot"] = "UTIL"
        assigned.append(p)

    # Build final list with display_order: C, C, W, W, W, D, D, G, UTIL
    out = []
    idx = 0
    for slot_type in ["C", "W", "D"]:
        for p in assigned:
            if p["slot"] == slot_type:
                p["display_order"] = idx
                idx += 1
                out.append(p)
        assigned = [p for p in assigned if p not in out or p["slot"] != slot_type]

    if goalie:
        goalie["slot"] = "G"
        goalie["display_order"] = 7
        out.append(goalie)

    for p in assigned:
        if p not in out:
            p["display_order"] = 8
            out.append(p)

    out.sort(key=lambda x: (x.get("display_order", 9), x.get("name", "")))
    return out


@app.route("/api/lineup")
def api_lineup():
    lineup_path = _latest_lineup_path()
    proj_path = _latest_projections_path()
    if not lineup_path:
        return jsonify({"lineup": [], "totals": {}, "error": "No lineup file. Run main.py with --lineups 1."})
    df_lineup = pd.read_csv(lineup_path)
    if df_lineup.empty:
        return jsonify({"lineup": [], "totals": {}})
    row = df_lineup.iloc[0]
    ncols = min(9, len(row))
    id_by_slot = []
    for i in range(ncols):
        val = row.iloc[i]
        try:
            val = int(val) if pd.notna(val) and str(val).replace(".0", "").isdigit() else val
        except (ValueError, TypeError):
            pass
        id_by_slot.append({"slot": DK_SLOT_ORDER[i], "id": val})
    if not proj_path:
        return jsonify({"lineup": id_by_slot, "totals": {}, "error": "No projections file to enrich lineup."})
    proj = pd.read_csv(proj_path)
    id_to_proj = {}
    if "dk_id" in proj.columns:
        for _, r in proj.iterrows():
            rid = r.get("dk_id")
            if pd.notna(rid):
                try:
                    id_to_proj[int(rid)] = r
                except (ValueError, TypeError):
                    id_to_proj[str(rid)] = r
    lineup_out = []
    for i, s in enumerate(id_by_slot):
        slot, pid = s["slot"], s["id"]
        rec = {"slot": slot, "name": None, "team": None, "position": None, "salary": None, "projected_fpts": None, "value": None, "predicted_ownership": None, "display_order": i}
        r = None
        if pid is not None:
            try:
                pid_int = int(float(pid)) if isinstance(pid, (int, float)) else int(pid)
            except (ValueError, TypeError):
                pid_int = None
            if pid_int is not None and pid_int in id_to_proj:
                r = id_to_proj[pid_int]
            elif pid_int is not None:
                r = id_to_proj.get(pid_int)
            else:
                r = id_to_proj.get(pid)
            if r is not None:
                rec["name"] = r.get("name")
                rec["team"] = r.get("team")
                rec["position"] = r.get("position")
                rec["salary"] = float(r["salary"]) if pd.notna(r.get("salary")) else None
                rec["projected_fpts"] = float(r["projected_fpts"]) if pd.notna(r.get("projected_fpts")) else None
                rec["value"] = float(r["value"]) if pd.notna(r.get("value")) else None
                rec["predicted_ownership"] = float(r["predicted_ownership"]) if pd.notna(r.get("predicted_ownership")) else None
        lineup_out.append(rec)
    # Assign slots by actual position so goalie shows as G, not C (handles bad CSV column order)
    lineup_out = _assign_slots_by_position(lineup_out)
    total_salary = sum(x["salary"] or 0 for x in lineup_out)
    total_proj = sum(x["projected_fpts"] or 0 for x in lineup_out)
    return jsonify({
        "lineup": lineup_out,
        "totals": {"salary": total_salary, "projected_fpts": total_proj},
    })


def _match_lineup_player_to_actual(rec, actuals_df):
    """Match a lineup player (name, team) to actuals row; return actual_fpts or None."""
    if actuals_df.empty or rec.get("name") is None or rec.get("team") is None:
        return None
    name = str(rec["name"]).strip()
    team = str(rec["team"]).strip()
    last_name = name.split()[-1] if name else ""
    if not last_name:
        return None
    cand = actuals_df[(actuals_df["team"] == team) & (actuals_df["last_name"] == last_name)]
    if len(cand) == 1:
        return float(cand.iloc[0]["actual_fpts"])
    if len(cand) > 1:
        for _, row in cand.iterrows():
            if row["name"] and name.endswith(" " + row["last_name"]):
                return float(row["actual_fpts"])
        return float(cand.iloc[0]["actual_fpts"])
    return None


def _dk_salary_path_for_slate(slate_date):
    """Find daily_salaries/DKSalaries_M.DD.YY.csv for a YYYY-MM-DD slate date."""
    try:
        from datetime import datetime as _dt
        dt = _dt.strptime(slate_date, "%Y-%m-%d")
        tag = f"{dt.month}.{dt.day:02d}.{dt.strftime('%y')}"
    except (ValueError, TypeError):
        return None
    sal_dir = PROJECT_ROOT / "daily_salaries"
    if not sal_dir.exists():
        return None
    candidate = sal_dir / f"DKSalaries_{tag}.csv"
    return candidate if candidate.exists() else None


@app.route("/api/last_night_lineup")
def api_last_night_lineup():
    """Return last night's lineup with projections, actuals, and ownership from the backtest xlsx."""
    lineup_path, slate_date = _last_night_lineup_path()
    if not lineup_path or not slate_date:
        return jsonify({
            "date": None, "lineup": [], "totals": {},
            "error": "No past-slate lineup file found.",
        })

    # Read lineup CSV (DK IDs)
    df_lineup = pd.read_csv(lineup_path)
    if df_lineup.empty:
        return jsonify({"date": slate_date, "lineup": [], "totals": {}, "error": "Lineup file empty."})
    row = df_lineup.iloc[0]
    dk_ids = []
    for i in range(min(9, len(row))):
        val = row.iloc[i]
        try:
            dk_ids.append(int(val) if pd.notna(val) and str(val).replace(".0", "").isdigit() else None)
        except (ValueError, TypeError):
            dk_ids.append(None)

    # Resolve DK IDs to names via salary file
    id_to_name = {}
    sal_path = _dk_salary_path_for_slate(slate_date)
    if sal_path:
        try:
            dk_sal = pd.read_csv(sal_path)
            if "ID" in dk_sal.columns and "Name" in dk_sal.columns:
                for _, sr in dk_sal.iterrows():
                    if pd.notna(sr["ID"]):
                        id_to_name[int(sr["ID"])] = str(sr["Name"]).strip()
        except Exception:
            pass

    # Load backtest xlsx for actuals + ownership (primary source)
    xlsx_path = _backtest_xlsx_for_slate(slate_date)
    bt_data = {}  # keyed by lowercase name
    if xlsx_path:
        bt_data = _load_backtest_actuals(xlsx_path)
        # Also load full projection data from the xlsx Projection sheet
        try:
            bt_proj = pd.read_excel(xlsx_path, sheet_name="Projection")
            for _, r in bt_proj.iterrows():
                name = str(r.get("name", "")).strip()
                if not name:
                    continue
                key = name.lower()
                if key not in bt_data:
                    bt_data[key] = {}
                bt_data[key]["team"] = r.get("team")
                bt_data[key]["position"] = r.get("position")
                bt_data[key]["salary"] = float(r["salary"]) if pd.notna(r.get("salary")) else None
                bt_data[key]["projected_fpts"] = float(r["projected_fpts"]) if pd.notna(r.get("projected_fpts")) else None
        except Exception:
            pass

    # Build lineup records
    lineup_out = []
    for i, dk_id in enumerate(dk_ids):
        rec = {
            "slot": DK_SLOT_ORDER[i] if i < len(DK_SLOT_ORDER) else "UTIL",
            "name": None, "team": None, "position": None, "salary": None,
            "projected_fpts": None, "predicted_ownership": None,
            "actual_fpts": None, "actual_ownership": None,
            "display_order": i,
        }
        name = id_to_name.get(dk_id) if dk_id else None
        if name:
            rec["name"] = name
            entry = bt_data.get(name.lower(), {})
            rec["team"] = entry.get("team")
            rec["position"] = entry.get("position")
            rec["salary"] = entry.get("salary")
            rec["projected_fpts"] = entry.get("projected_fpts")
            rec["actual_fpts"] = entry.get("actual_fpts")
            rec["actual_ownership"] = entry.get("actual_ownership")
        lineup_out.append(rec)

    lineup_out = _assign_slots_by_position(lineup_out)
    total_proj = sum(x["projected_fpts"] or 0 for x in lineup_out)
    total_actual = sum(x["actual_fpts"] or 0 for x in lineup_out)
    return jsonify({
        "date": slate_date,
        "lineup": lineup_out,
        "totals": {"projected_fpts": total_proj, "actual_fpts": total_actual},
        "error": None,
    })


@app.route("/")
def index():
    return send_from_directory(Path(__file__).resolve().parent, "index.html")


@app.route("/static/<path:path>")
def static_file(path):
    return send_from_directory(Path(__file__).resolve().parent / "static", path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
