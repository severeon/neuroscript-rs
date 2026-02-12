#!/bin/bash
# PostToolUse hook: Run cargo check after .rs file edits
# Catches compilation errors immediately instead of letting them cascade.
# Also persists results to .claude/context/build-status.md for later reference.
#
# Input: JSON on stdin from Claude Code (PostToolUse event)
# Output: Compilation errors as stdout (injected as Claude context)

set -o pipefail

INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

# Only run for .rs files
[[ "$FILE_PATH" != *.rs ]] && exit 0

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

CONTEXT_DIR="$CLAUDE_PROJECT_DIR/.claude/context"
STATUS_FILE="$CONTEXT_DIR/build-status.md"
mkdir -p "$CONTEXT_DIR"

# Run cargo check (incremental, usually fast)
RESULT=$(cargo check --message-format=short 2>&1)
EXIT_CODE=$?

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

if [ $EXIT_CODE -ne 0 ]; then
  ERRORS=$(echo "$RESULT" | head -40)
  # Immediate feedback via stdout
  echo "cargo check failed after editing $(basename "$FILE_PATH"):"
  echo "$ERRORS"

  # Persist to context file
  {
    echo "# Build Status"
    echo ""
    echo "Updated: $TIMESTAMP"
    echo ""
    echo "## cargo check: FAIL"
    echo ""
    echo "Triggered by edit to \`$(basename "$FILE_PATH")\`"
    echo ""
    echo '```'
    echo "$ERRORS"
    echo '```'
  } > "$STATUS_FILE.tmp" && mv "$STATUS_FILE.tmp" "$STATUS_FILE"
else
  # Persist success
  {
    echo "# Build Status"
    echo ""
    echo "Updated: $TIMESTAMP"
    echo ""
    echo "## cargo check: OK"
    echo ""
    echo "All Rust code compiles cleanly."
  } > "$STATUS_FILE.tmp" && mv "$STATUS_FILE.tmp" "$STATUS_FILE"
fi

# Always exit 0 — this is informational, not blocking
exit 0
