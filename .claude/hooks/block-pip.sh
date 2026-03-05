#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

if echo "$COMMAND" | grep -qE '(^|[;&|]\s*)(pip3?)\s+install\b'; then
    echo "Blocked: use 'uv add <package>' instead of 'pip install'" >&2
    exit 2
fi

if echo "$COMMAND" | grep -qE '(^|[;&|]\s*)(pip3?)\s+uninstall\b'; then
    echo "Blocked: use 'uv remove <package>' instead of 'pip uninstall'" >&2
    exit 2
fi

exit 0
