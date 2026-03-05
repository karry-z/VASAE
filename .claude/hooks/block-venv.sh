#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

if echo "$COMMAND" | grep -qE '(^|[;&|]\s*)python3?\s+-m\s+venv\b'; then
    echo "Blocked: use 'uv venv --python <version>' instead of 'python -m venv'" >&2
    exit 2
fi

if echo "$COMMAND" | grep -qE '(^|[;&|]\s*)virtualenv\b'; then
    echo "Blocked: use 'uv venv --python <version>' instead of 'virtualenv'" >&2
    exit 2
fi

exit 0
