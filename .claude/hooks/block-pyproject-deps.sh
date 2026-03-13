#!/bin/bash
# Block direct edits to pyproject.toml dependencies — use uv add instead
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
if echo "$FILE_PATH" | grep -qE 'pyproject\.toml'; then
  CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')
  if echo "$CONTENT" | grep -qE 'dependencies'; then
    echo "BLOCKED: Use 'uv add <package>' instead of editing dependencies directly" >&2
    exit 2
  fi
fi
