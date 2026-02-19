#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

# Only run tests on git commit commands
if [[ ! "$COMMAND" =~ ^git.*commit ]]; then
  exit 0
fi

cd /Users/brendanhorlbeck/Desktop/Code/projection
python -m pytest tests/ --timeout=30 -q 2>&1 | tail -5

if [ $? -eq 0 ]; then
  exit 0
else
  echo "Tests failed — blocking commit"
  exit 2  # Block commit if tests fail
fi
