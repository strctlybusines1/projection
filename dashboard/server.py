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
    Reassign roster slots from each player's actual position so the goalie
    gets G (not C) and every player is in the correct slot. Handles CSVs
    that were exported with columns in the wrong order.
    """
    # Bucket by slot type from position
    goalies = []
    centers = []
    wings = []
    defense = []
    other = []
    for rec in lineup_out:
        pos = rec.get("position")
        slot_type = _position_to_slot_type(pos)
        if slot_type == "G":
            goalies.append(rec)
        elif slot_type == "C":
            centers.append(rec)
        elif slot_type == "W":
            wings.append(rec)
        elif slot_type == "D":
            defense.append(rec)
        else:
            other.append(rec)
    # Assign slots: 1 G, 2 C, 3 W, 2 D, 1 UTIL (any skater)
    out = []
    idx = 0
    for g in goalies[:1]:
        g["slot"] = "G"
        g["display_order"] = 7
        out.append(g)
    for c in centers[:2]:
        c["slot"] = "C"
        c["display_order"] = idx
        idx += 1
        out.append(c)
    for w in wings[:3]:
        w["slot"] = "W"
        w["display_order"] = idx
        idx += 1
        out.append(w)
    for d in defense[:2]:
        d["slot"] = "D"
        d["display_order"] = idx
        idx += 1
        out.append(d)
    # UTIL = one remaining skater (C, W, or D)
    util_pool = centers[2:] + wings[3:] + defense[2:] + other
    for u in util_pool[:1]:
        u["slot"] = "UTIL"
        u["display_order"] = 8
        out.append(u)
    # Sort by display order: C, C, W, W, W, D, D, G, UTIL
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


@app.route("/")
def index():
    return send_from_directory(Path(__file__).resolve().parent, "index.html")


@app.route("/static/<path:path>")
def static_file(path):
    return send_from_directory(Path(__file__).resolve().parent / "static", path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
