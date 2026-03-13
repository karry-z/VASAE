#!/bin/bash
# Block direct pip install — use uv instead
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
if echo "$COMMAND" | grep -qE 'pip install|pip3 install'; then
  echo "BLOCKED: Use 'uv add' instead of 'pip install'" >&2
  exit 2
fi
