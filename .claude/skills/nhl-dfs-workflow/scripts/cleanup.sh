#!/bin/bash
# NHL DFS Project Cleanup Script
# Run from projection/ directory

set -e

echo "NHL DFS Project Cleanup"
echo "======================="

# Create archive directories
echo "Creating archive directories..."
mkdir -p daily_salaries/archive
mkdir -p daily_projections/archive
mkdir -p vegas/archive
mkdir -p backtests/archive

# Archive files older than 7 days
echo "Archiving files older than 7 days..."

# Salaries
count=$(find daily_salaries -maxdepth 1 -name "*.csv" -mtime +7 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    find daily_salaries -maxdepth 1 -name "*.csv" -mtime +7 -exec mv {} daily_salaries/archive/ \;
    echo "  Archived $count salary files"
else
    echo "  No old salary files to archive"
fi

# Projections
count=$(find daily_projections -maxdepth 1 -name "*.csv" -mtime +7 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    find daily_projections -maxdepth 1 -name "*.csv" -mtime +7 -exec mv {} daily_projections/archive/ \;
    echo "  Archived $count projection files"
else
    echo "  No old projection files to archive"
fi

# Vegas
count=$(find vegas -maxdepth 1 -name "*.csv" -mtime +7 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    find vegas -maxdepth 1 -name "*.csv" -mtime +7 -exec mv {} vegas/archive/ \;
    echo "  Archived $count vegas files"
else
    echo "  No old vegas files to archive"
fi

# Remove empty __pycache__ and .pyc files
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Remove .DS_Store files (macOS)
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -delete 2>/dev/null || true

# Summary
echo ""
echo "Cleanup complete!"
echo ""
echo "Current file counts:"
echo "  daily_salaries/: $(ls -1 daily_salaries/*.csv 2>/dev/null | wc -l) active"
echo "  daily_projections/: $(ls -1 daily_projections/*.csv 2>/dev/null | wc -l) active"
echo "  vegas/: $(ls -1 vegas/*.csv 2>/dev/null | wc -l) active"
echo "  backtests/: $(ls -1 backtests/*.xlsx 2>/dev/null | wc -l) backtest files"
