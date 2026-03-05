#!/bin/bash
INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name')
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""')

# Only care about pyproject.toml
if ! echo "$FILE" | grep -q 'pyproject\.toml$'; then
    exit 0
fi

if [ "$TOOL" = "Edit" ]; then
    OLD=$(echo "$INPUT" | jq -r '.tool_input.old_string // ""')
    NEW=$(echo "$INPUT" | jq -r '.tool_input.new_string // ""')
    COMBINED="$OLD
$NEW"
elif [ "$TOOL" = "Write" ]; then
    COMBINED=$(echo "$INPUT" | jq -r '.tool_input.content // ""')
else
    exit 0
fi

# Check if the edit touches dependency sections
if echo "$COMBINED" | grep -qE '^\s*\[project\.dependencies\]|^\s*\[project\.optional-dependencies\]|^\s*dependencies\s*=\s*\['; then
    echo "Blocked: do not manually edit dependencies in pyproject.toml. Use 'uv add <package>' or 'uv remove <package>' instead." >&2
    exit 2
fi

exit 0
