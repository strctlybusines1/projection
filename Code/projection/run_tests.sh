#!/usr/bin/env bash
#
# NHL DFS Pipeline Test Runner
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh -k test_optimizer  # Run specific test
#   ./run_tests.sh --verbose    # Verbose output
#

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  NHL DFS Pipeline Test Suite                                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗ pytest not found${NC}"
    echo "Install with: pip install pytest pytest-cov pytest-mock"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in a git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    # Check for uncommitted changes (informational only)
    if [[ -n $(git status --porcelain) ]]; then
        echo -e "${YELLOW}⚠ Uncommitted changes detected${NC}"
    fi
fi

echo "Running tests..."
echo ""

# Run pytest with coverage
pytest tests/ \
    --verbose \
    --tb=short \
    --color=yes \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed${NC}"
    echo ""
    echo "Coverage report: file://$(pwd)/htmlcov/index.html"
else
    echo -e "${RED}✗ Tests failed (exit code: $EXIT_CODE)${NC}"
    echo ""
    echo "Review failures above and fix before committing."
fi

exit $EXIT_CODE
