#!/bin/bash
# Block venv creation — uv manages the virtualenv
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
if echo "$COMMAND" | grep -qE 'python -m venv|python3 -m venv|virtualenv'; then
  echo "BLOCKED: Use 'uv' to manage virtual environments" >&2
  exit 2
fi
