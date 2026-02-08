#!/usr/bin/env bash
# git_backup.sh — Quick commit and push to GitHub
# Usage: ./git_backup.sh [optional commit message]

set -e
cd "$(dirname "$0")"

# Default commit message with date
MSG="${1:-slate update $(date +%m/%d/%y)}"

# Show what's changed
echo "=== Changed files ==="
git status --short
echo ""

# Stage everything (respects .gitignore)
git add -A

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo "Nothing to commit — working tree clean."
    exit 0
fi

# Commit and push
git commit -m "$MSG"
git push origin main

echo ""
echo "✅ Pushed to origin/main"
