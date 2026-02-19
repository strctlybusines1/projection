#!/bin/bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only check Python files
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

if python -m py_compile "$FILE_PATH" 2>&1; then
  exit 0
else
  echo "SYNTAX ERROR in $FILE_PATH"
  exit 2  # Block the edit if syntax check fails
fi
