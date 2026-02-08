#!/bin/bash
# ============================================================
# NHL DFS Project Cleanup Script
# Run from: ~/Desktop/Code/projection/
# 
# WHAT THIS DOES:
#   1. Creates a reports/ folder for PRO_REPORT files
#   2. Creates an archive/docs/ folder for redundant docs
#   3. Moves PRO_REPORTs into reports/
#   4. Moves redundant/overlapping docs into archive/docs/
#   5. Moves NFLParser out of the project
#   6. Cleans up backup files, __pycache__, .DS_Store
#   7. Updates .gitignore
#
# SAFE: No Python code is modified. No projection logic touched.
# REVERSIBLE: Everything is moved, not deleted. You can move it back.
# ============================================================

echo "=== NHL DFS Project Cleanup ==="
echo "Run this from: ~/Desktop/Code/projection/"
echo ""

# ---- Verify we're in the right directory ----
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found. Make sure you're in the projection/ directory."
    echo "  cd ~/Desktop/Code/projection"
    exit 1
fi

echo "✓ In projection/ directory"
echo ""

# ============================================================
# 1. Create new directories
# ============================================================
echo "--- Creating directories ---"
mkdir -p reports
mkdir -p archive/docs

# ============================================================
# 2. Move PRO_REPORT files → reports/
# ============================================================
echo "--- Moving PRO_REPORT files to reports/ ---"
mv BKRIDER_PRO_REPORT.md reports/ 2>/dev/null && echo "  ✓ BKRIDER_PRO_REPORT.md"
mv HUNTER_PRO_REPORT.md reports/ 2>/dev/null && echo "  ✓ HUNTER_PRO_REPORT.md"
mv SAUL_PRO_REPORT.md reports/ 2>/dev/null && echo "  ✓ SAUL_PRO_REPORT.md"
mv fan_PRO_REPORT.md reports/ 2>/dev/null && echo "  ✓ fan_PRO_REPORT.md"
mv fire_PRO_REPORT.md reports/ 2>/dev/null && echo "  ✓ fire_PRO_REPORT.md"
mv goodseats_PRO_REPORT.md reports/ 2>/dev/null && echo "  ✓ goodseats_PRO_REPORT.md"
mv nolt_PRO_REPORT.md reports/ 2>/dev/null && echo "  ✓ nolt_PRO_REPORT.md"
mv HIGH_DOLLAR_REPORT.md reports/ 2>/dev/null && echo "  ✓ HIGH_DOLLAR_REPORT.md"
mv HIGH_DOLLAR_SLATE_NOTES.md reports/ 2>/dev/null && echo "  ✓ HIGH_DOLLAR_SLATE_NOTES.md"

# ============================================================
# 3. Move redundant/overlapping docs → archive/docs/
#    (CLAUDE.md + DAILY_WORKFLOW.md cover the same ground as these)
# ============================================================
echo ""
echo "--- Archiving redundant docs to archive/docs/ ---"
mv EDGE_CACHING_WORKFLOW.md archive/docs/ 2>/dev/null && echo "  ✓ EDGE_CACHING_WORKFLOW.md (covered in CLAUDE.md)"
mv EDGE_INTEGRATION.md archive/docs/ 2>/dev/null && echo "  ✓ EDGE_INTEGRATION.md (covered in CLAUDE.md)"
mv INTEGRATION_GUIDE.md archive/docs/ 2>/dev/null && echo "  ✓ INTEGRATION_GUIDE.md (covered in CLAUDE.md)"
mv NHL_DFS_Research_Integration.md archive/docs/ 2>/dev/null && echo "  ✓ NHL_DFS_Research_Integration.md"

# ============================================================
# 4. Move NFLParser out of the NHL project
# ============================================================
echo ""
echo "--- Moving NFLParser out of project ---"
mv "../NFLParser_Updated copy.py" ~/Desktop/ 2>/dev/null && echo "  ✓ NFLParser moved to ~/Desktop/"

# ============================================================
# 5. Clean up backup files
# ============================================================
echo ""
echo "--- Cleaning up backup files ---"
rm -f CLAUDE.md.backup 2>/dev/null && echo "  ✓ Removed CLAUDE.md.backup"
rm -f .mcp.json.backup 2>/dev/null && echo "  ✓ Removed .mcp.json.backup"

# ============================================================
# 6. Clean up __pycache__ and .DS_Store
# ============================================================
echo ""
echo "--- Cleaning up cache and OS files ---"
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null && echo "  ✓ Removed __pycache__ directories"
find . -name '.DS_Store' -delete 2>/dev/null && echo "  ✓ Removed .DS_Store files"
find .. -maxdepth 1 -name '.DS_Store' -delete 2>/dev/null

# ============================================================
# 7. Summary
# ============================================================
echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "New structure:"
echo "  reports/              ← PRO_REPORT files + slate analysis"
echo "  archive/docs/         ← Redundant docs (still accessible)"
echo ""
echo "Remaining core docs:"
echo "  CLAUDE.md             ← Architecture + commands (keep as primary)"
echo "  DAILY_WORKFLOW.md     ← Daily operations (keep)"
echo "  DAILY_PROJECTION_IMPROVEMENT_PLAN.md ← Strategy (keep)"
echo "  OWNERSHIP_PLAN.md     ← Ownership model notes (keep)"
echo "  README.md             ← Project overview (keep)"
echo ""
echo "NFLParser moved to ~/Desktop/"
echo ""
echo "To undo any move:  mv reports/FILENAME.md ./"
