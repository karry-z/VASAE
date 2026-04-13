#!/usr/bin/env bash
# PostToolUse hook: run isort + black on the file Claude just edited.
# Non-Python files: no-op. Failures: warn but don't block.

set -u

payload="$(cat)"
file="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty' 2>/dev/null)"

[[ -z "$file" ]] && exit 0
[[ "$file" != *.py ]] && exit 0
[[ ! -f "$file" ]] && exit 0

if ! uvx --quiet isort --profile black --quiet "$file" 2>&1; then
    echo "[format-python] isort failed on $file" >&2
fi

if ! uvx --quiet black --quiet "$file" 2>&1; then
    echo "[format-python] black failed on $file" >&2
fi

exit 0
