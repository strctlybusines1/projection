#!/bin/bash
# ============================================================
# NHL DFS Quick Commands
# Usage: ./run.sh <command> [options]
#
# Place in: ~/Desktop/Code/projection/
# Setup:   chmod +x run.sh
# ============================================================

# cd to the script's directory so it works from anywhere
cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

show_help() {
    echo ""
    echo -e "${CYAN}NHL DFS Quick Commands${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${GREEN}Daily Workflow:${NC}"
    echo "  ./run.sh se                Single-entry (40 candidates, edge, stacks)"
    echo "  ./run.sh se 60             Single-entry with 60 candidates"
    echo "  ./run.sh gpp               GPP multi-entry (5 lineups, edge, stacks)"
    echo "  ./run.sh gpp 20            GPP with 20 lineups"
    echo "  ./run.sh base              Quick base projections (no edge, no linemates)"
    echo "  ./run.sh full              Full pipeline (edge + linemates + stacks)"
    echo ""
    echo -e "${GREEN}Backtesting:${NC}"
    echo "  ./run.sh bt 2026-01-29     Backtest a past date (auto-matches salary+vegas)"
    echo "  ./run.sh actuals 2026-01-29          Fetch actual DK scores"
    echo "  ./run.sh actuals 2026-01-29 --compare   Fetch + compare to projections"
    echo "  ./run.sh actuals-all       Backfill actuals for all saved salary dates"
    echo ""
    echo -e "${GREEN}Analysis:${NC}"
    echo "  ./run.sh stacks            Show stacking recommendations"
    echo "  ./run.sh injuries          Show injury report"
    echo "  ./run.sh sim               Run simulator (deterministic)"
    echo "  ./run.sh sim 100           Run simulator with 100 MC iterations"
    echo ""
    echo -e "${GREEN}Contest-Specific:${NC}"
    echo "  ./run.sh hdse              High-dollar single entry (\$20+ SE)"
    echo "  ./run.sh smallgpp          Small-field GPP"
    echo ""
    echo -e "${GREEN}Options (append to any command):${NC}"
    echo "  --date 2026-01-29          Run for specific date"
    echo "  --force-stack TBL          Force primary stack team"
    echo "  --no-edge                  Skip Edge stats"
    echo "  --refresh-edge             Force refresh Edge cache"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  ./run.sh se --force-stack COL"
    echo "  ./run.sh gpp 10 --date 2026-01-29"
    echo "  ./run.sh bt 2026-02-03"
    echo ""
}

CMD="${1:-help}"
shift 2>/dev/null  # Consume first arg, rest passed through

case "$CMD" in

    # ── Single Entry ──────────────────────────────────────
    se)
        N_CANDIDATES="${1:-40}"
        # If first remaining arg is a number, consume it; otherwise keep it as a flag
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            shift
        fi
        echo -e "${YELLOW}▶ Single-Entry Mode (${N_CANDIDATES} candidates)${NC}"
        python main.py --single-entry --lineups "$N_CANDIDATES" --edge --stacks "$@"
        ;;

    # ── GPP Multi-Entry ───────────────────────────────────
    gpp)
        N_LINEUPS="${1:-5}"
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            shift
        fi
        echo -e "${YELLOW}▶ GPP Mode (${N_LINEUPS} lineups)${NC}"
        python main.py --lineups "$N_LINEUPS" --edge --stacks "$@"
        ;;

    # ── Base Projections (fast) ───────────────────────────
    base)
        echo -e "${YELLOW}▶ Base Projections (no edge, no linemates)${NC}"
        python main.py --no-edge "$@"
        ;;

    # ── Full Pipeline ─────────────────────────────────────
    full)
        echo -e "${YELLOW}▶ Full Pipeline (edge + linemates + stacks)${NC}"
        python main.py --edge --linemates --stacks --show-injuries "$@"
        ;;

    # ── Backtest a Past Date ──────────────────────────────
    bt|backtest)
        DATE="$1"
        if [ -z "$DATE" ]; then
            echo "Usage: ./run.sh bt 2026-01-29"
            exit 1
        fi
        shift
        echo -e "${YELLOW}▶ Backtest: ${DATE}${NC}"
        python main.py --date "$DATE" --edge --stacks "$@"
        ;;

    # ── Fetch Actual Scores ───────────────────────────────
    actuals)
        DATE="$1"
        if [ -z "$DATE" ]; then
            echo "Usage: ./run.sh actuals 2026-01-29 [--save] [--compare]"
            exit 1
        fi
        shift
        echo -e "${YELLOW}▶ Fetching actual scores: ${DATE}${NC}"
        python actual_scores.py "$DATE" "$@"
        ;;

    # ── Backfill Actuals for All Saved Dates ──────────────
    actuals-all)
        echo -e "${YELLOW}▶ Backfilling actuals for all saved salary dates${NC}"
        for f in daily_salaries/DKSalaries_*.csv; do
            # Extract date from filename: DKSalaries_1.29.26.csv -> 2026-01-29
            DATE_PART=$(echo "$f" | sed 's/.*DKSalaries_//' | sed 's/.csv//' | sed 's/_filtered//')
            # Parse M.D.YY format
            M=$(echo "$DATE_PART" | cut -d. -f1)
            D=$(echo "$DATE_PART" | cut -d. -f2)
            Y=$(echo "$DATE_PART" | cut -d. -f3)
            FULL_DATE="20${Y}-$(printf '%02d' $M)-$(printf '%02d' $D)"
            echo ""
            echo -e "${CYAN}── ${FULL_DATE} ──${NC}"
            python actual_scores.py "$FULL_DATE" --save --compare
        done
        ;;

    # ── Stacking Recommendations ──────────────────────────
    stacks)
        echo -e "${YELLOW}▶ Stacking Recommendations${NC}"
        python main.py --stacks --no-edge "$@"
        ;;

    # ── Injury Report ─────────────────────────────────────
    injuries)
        echo -e "${YELLOW}▶ Injury Report${NC}"
        python main.py --show-injuries --no-edge "$@"
        ;;

    # ── Simulator ─────────────────────────────────────────
    sim)
        ITERS="${1:-0}"
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            shift
        fi
        echo -e "${YELLOW}▶ Simulator (${ITERS} iterations)${NC}"
        python main.py --simulate --sim-iterations "$ITERS" --edge "$@"
        ;;

    # ── High-Dollar Single Entry ──────────────────────────
    hdse)
        echo -e "${YELLOW}▶ High-Dollar Single Entry${NC}"
        python main.py --single-entry --lineups 50 --edge --stacks \
            --contest-payout high_dollar_single "$@"
        ;;

    # ── Small GPP ─────────────────────────────────────────
    smallgpp)
        echo -e "${YELLOW}▶ Small-Field GPP${NC}"
        python main.py --lineups 3 --edge --stacks \
            --contest-payout small_se_gpp "$@"
        ;;

    # ── Help ──────────────────────────────────────────────
    help|-h|--help|"")
        show_help
        ;;

    # ── Unknown Command ───────────────────────────────────
    *)
        echo -e "Unknown command: ${CMD}"
        echo "Run ./run.sh help for available commands"
        exit 1
        ;;
esac
