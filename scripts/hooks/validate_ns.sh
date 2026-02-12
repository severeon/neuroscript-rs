#!/bin/bash
# PostToolUse hook: Validate .ns files after edits
# Catches NeuroScript syntax/validation errors immediately.
# Also persists results to .claude/context/ns-validation-status.md for later reference.
#
# Input: JSON on stdin from Claude Code (PostToolUse event)
# Output: Validation errors as stdout (injected as Claude context)

set -o pipefail

INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

# Only run for .ns files
[[ "$FILE_PATH" != *.ns ]] && exit 0

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

CONTEXT_DIR="$CLAUDE_PROJECT_DIR/.claude/context"
STATUS_FILE="$CONTEXT_DIR/ns-validation-status.md"
mkdir -p "$CONTEXT_DIR"

# Check if the binary exists
BINARY="./target/release/neuroscript"
if [ ! -f "$BINARY" ]; then
  echo "neuroscript binary not found — skipping .ns validation (run cargo build --release)"
  exit 0
fi

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Run validation on the edited file
RESULT=$("$BINARY" validate "$FILE_PATH" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  ERRORS=$(echo "$RESULT" | head -30)
  # Immediate feedback via stdout
  echo "NeuroScript validation failed for $(basename "$FILE_PATH"):"
  echo "$ERRORS"

  # Persist to context file
  {
    echo "# NeuroScript Validation Status"
    echo ""
    echo "Updated: $TIMESTAMP"
    echo ""
    echo "## Last edited file: FAIL"
    echo ""
    echo "File: \`$FILE_PATH\`"
    echo ""
    echo '```'
    echo "$ERRORS"
    echo '```'
  } > "$STATUS_FILE.tmp" && mv "$STATUS_FILE.tmp" "$STATUS_FILE"
else
  # Immediate success feedback
  echo "NeuroScript validation passed for $(basename "$FILE_PATH")"

  # Persist success
  {
    echo "# NeuroScript Validation Status"
    echo ""
    echo "Updated: $TIMESTAMP"
    echo ""
    echo "## Last edited file: OK"
    echo ""
    echo "File: \`$FILE_PATH\` validates successfully."
  } > "$STATUS_FILE.tmp" && mv "$STATUS_FILE.tmp" "$STATUS_FILE"
fi

# Always exit 0 — informational
exit 0
