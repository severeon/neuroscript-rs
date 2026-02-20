#!/bin/bash
# Stop hook: Flag sessions that made commits for reflection
# If commits were made during this session, creates a pending-reflection marker
# so the next session can prompt for /reflect.
#
# Input: JSON on stdin from Claude Code (Stop event)
# Output: None (just writes marker file)

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

CONTEXT_DIR="$CLAUDE_PROJECT_DIR/.claude/context"
MARKER="$CONTEXT_DIR/pending-reflection.md"
SESSION_START_FILE="$CONTEXT_DIR/.session-start-time"

# Check if we have a session start timestamp
if [ ! -f "$SESSION_START_FILE" ]; then
  exit 0
fi

SESSION_START=$(cat "$SESSION_START_FILE" 2>/dev/null)

# Count commits made since session start
COMMIT_COUNT=$(git log --after="$SESSION_START" --oneline 2>/dev/null | wc -l | tr -d ' ')

if [ "$COMMIT_COUNT" -gt 0 ]; then
  mkdir -p "$CONTEXT_DIR"

  # Detect neurons created during this session (stdlib/*.ns files in commits)
  NEURONS_CREATED=$(git log --after="$SESSION_START" --diff-filter=A --name-only --pretty=format: 2>/dev/null \
    | grep '^stdlib/.*\.ns$' \
    | sed 's|stdlib/||;s|\.ns$||' \
    | sort -u \
    | paste -sd ',' - 2>/dev/null || echo "")
  CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

  cat > "$MARKER" <<EOF
---
session_end: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
commits: $COMMIT_COUNT
neurons_created: [${NEURONS_CREATED}]
branch: $CURRENT_BRANCH
---

# Pending Session Reflection
Session ended: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Commits made: $COMMIT_COUNT

## Commits This Session
$(git log --after="$SESSION_START" --oneline 2>/dev/null)

## Files Changed
$(git diff --stat "$(git log --after="$SESSION_START" --format=%H 2>/dev/null | tail -1)^..HEAD" 2>/dev/null | tail -5)

Run /reflect to capture learnings from this session.
EOF
fi

# Clean up session start marker
rm -f "$SESSION_START_FILE"

exit 0
