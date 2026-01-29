#!/usr/bin/env bash
# NHL DFS Daily Workflow — automated pipeline + dashboard
# Scheduled via launchd to run at 6 PM ET daily.

set -euo pipefail

PROJ_DIR="/Users/brendanhorlbeck/Desktop/Code/projection"
PYTHON="/opt/miniconda3/bin/python"
LOG_DIR="$PROJ_DIR/logs"
LOG="$LOG_DIR/daily_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=========================================="
echo "NHL DFS Daily Run — $(date)"
echo "=========================================="

cd "$PROJ_DIR"

# --- Phase 1: Fetch line combinations & confirmed goalies ---
echo ""
echo "[1/4] Fetching lines & goalies..."
$PYTHON lines.py

# --- Phase 3: Full projections with stacks & injury report ---
echo ""
echo "[2/4] Generating projections..."
$PYTHON main.py --stacks --show-injuries

# --- Phase 4: Generate lineups ---
echo ""
echo "[3/4] Generating lineups..."
$PYTHON main.py --lineups 5 --stacks

# --- Start dashboard (kill existing first) ---
echo ""
echo "[4/4] Starting dashboard..."
EXISTING_PID=$(lsof -ti:5000 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Killing existing dashboard (PID $EXISTING_PID)..."
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 1
fi
# Run dashboard in background; it stays up until next run or manual stop
nohup $PYTHON dashboard/server.py >> "$LOG_DIR/dashboard.log" 2>&1 &
DASH_PID=$!
echo "Dashboard started (PID $DASH_PID) at http://127.0.0.1:5000"

echo ""
echo "=========================================="
echo "Daily run complete — $(date)"
echo "Log: $LOG"
echo "=========================================="
