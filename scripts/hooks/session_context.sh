#!/bin/bash
# SessionStart hook: Inject project state + context bundle references
# Gives Claude immediate awareness of branch, build status, recent work,
# and pointers to generated context files for deeper exploration.
#
# Also runs cargo check and neuroscript validate at session start to
# populate .claude/context/ with fresh build/validation status.
#
# Input: JSON on stdin from Claude Code (SessionStart event)
# Output: Project context as stdout (injected into Claude's context)

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

CONTEXT_DIR="$CLAUDE_PROJECT_DIR/.claude/context"
GEN_SCRIPT="$CLAUDE_PROJECT_DIR/scripts/generate-context.sh"

mkdir -p "$CONTEXT_DIR"

# Regenerate context if artifacts are stale (older than last commit)
if [ -f "$GEN_SCRIPT" ]; then
  NEEDS_REGEN=false

  # Check if context files exist at all
  if [ ! -f "$CONTEXT_DIR/ir-types-summary.md" ] || [ ! -f "$CONTEXT_DIR/recent-changes.md" ]; then
    NEEDS_REGEN=true
  else
    # Check if any context file is older than the last commit
    LAST_COMMIT_TIME=$(git log -1 --format=%ct 2>/dev/null || echo "0")
    for f in "$CONTEXT_DIR"/*.md; do
      if [ -f "$f" ]; then
        FILE_TIME=$(stat -f %m "$f" 2>/dev/null || echo "0")
        if [ "$FILE_TIME" -lt "$LAST_COMMIT_TIME" ]; then
          NEEDS_REGEN=true
          break
        fi
      fi
    done
  fi

  if [ "$NEEDS_REGEN" = true ]; then
    "$GEN_SCRIPT" all >/dev/null 2>&1 || true
  fi
fi

echo "Project state:"

# Current branch
BRANCH=$(git branch --show-current 2>/dev/null)
[ -n "$BRANCH" ] && echo "  Branch: $BRANCH"

# Uncommitted changes summary
CHANGES=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [ "$CHANGES" -gt 0 ]; then
  echo "  Uncommitted changes: $CHANGES file(s)"
fi

# ----- Run cargo check and persist results -----
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
BUILD_STATUS_FILE="$CONTEXT_DIR/build-status.md"

CARGO_RESULT=$(cargo check --message-format=short 2>&1)
CARGO_EXIT=$?

if [ $CARGO_EXIT -ne 0 ]; then
  CARGO_ERRORS=$(echo "$CARGO_RESULT" | head -40)
  echo "  Build: FAIL (cargo check errors)"
  {
    echo "# Build Status"
    echo ""
    echo "Updated: $TIMESTAMP (session start)"
    echo ""
    echo "## cargo check: FAIL"
    echo ""
    echo '```'
    echo "$CARGO_ERRORS"
    echo '```'
  } > "$BUILD_STATUS_FILE.tmp" && mv "$BUILD_STATUS_FILE.tmp" "$BUILD_STATUS_FILE"
else
  echo "  Build: OK (cargo check passes)"
  {
    echo "# Build Status"
    echo ""
    echo "Updated: $TIMESTAMP (session start)"
    echo ""
    echo "## cargo check: OK"
    echo ""
    echo "All Rust code compiles cleanly."
  } > "$BUILD_STATUS_FILE.tmp" && mv "$BUILD_STATUS_FILE.tmp" "$BUILD_STATUS_FILE"
fi

# ----- Validate .ns files and persist results -----
BINARY="./target/release/neuroscript"
NS_STATUS_FILE="$CONTEXT_DIR/ns-validation-status.md"

if [ -f "$BINARY" ]; then
  # Check if binary is stale
  STALE_SRC=$(find src/ -name '*.rs' -newer "$BINARY" -print -quit 2>/dev/null)
  if [ -n "$STALE_SRC" ]; then
    echo "  Binary: STALE (source changed since last build)"
  fi

  # Validate all .ns files in examples/ and stdlib/
  FAIL_COUNT=0
  PASS_COUNT=0
  FAIL_DETAILS=""

  for ns_file in examples/*.ns stdlib/*.ns; do
    [ -f "$ns_file" ] || continue
    VRESULT=$("$BINARY" validate "$ns_file" 2>&1)
    if [ $? -ne 0 ]; then
      FAIL_COUNT=$((FAIL_COUNT + 1))
      FAIL_DETAILS="${FAIL_DETAILS}\n### $(basename "$ns_file")\n\n\`\`\`\n$(echo "$VRESULT" | head -10)\n\`\`\`\n"
    else
      PASS_COUNT=$((PASS_COUNT + 1))
    fi
  done

  TOTAL=$((PASS_COUNT + FAIL_COUNT))
  if [ $FAIL_COUNT -eq 0 ]; then
    echo "  NS validation: OK ($TOTAL/$TOTAL files pass)"
    {
      echo "# NeuroScript Validation Status"
      echo ""
      echo "Updated: $TIMESTAMP (session start — full scan)"
      echo ""
      echo "## All files: OK"
      echo ""
      echo "$PASS_COUNT/$TOTAL .ns files validate successfully."
    } > "$NS_STATUS_FILE.tmp" && mv "$NS_STATUS_FILE.tmp" "$NS_STATUS_FILE"
  else
    echo "  NS validation: $FAIL_COUNT/$TOTAL files FAIL"
    {
      echo "# NeuroScript Validation Status"
      echo ""
      echo "Updated: $TIMESTAMP (session start — full scan)"
      echo ""
      echo "## $FAIL_COUNT/$TOTAL files FAIL"
      echo ""
      echo "$PASS_COUNT passed, $FAIL_COUNT failed."
      echo ""
      echo "## Failures"
      echo -e "$FAIL_DETAILS"
    } > "$NS_STATUS_FILE.tmp" && mv "$NS_STATUS_FILE.tmp" "$NS_STATUS_FILE"
  fi
else
  echo "  Binary: not found (needs cargo build --release)"
  {
    echo "# NeuroScript Validation Status"
    echo ""
    echo "Updated: $TIMESTAMP (session start)"
    echo ""
    echo "## Skipped"
    echo ""
    echo "neuroscript binary not found. Run \`cargo build --release\` first."
  } > "$NS_STATUS_FILE.tmp" && mv "$NS_STATUS_FILE.tmp" "$NS_STATUS_FILE"
fi

# Recent commits (last 3)
echo "  Recent commits:"
git log --oneline -3 2>/dev/null | while read -r line; do
  echo "    $line"
done

# Context bundle references
echo ""
echo "Context files (.claude/context/):"
if [ -f "$CONTEXT_DIR/build-status.md" ]; then
  echo "  - build-status.md — cargo check results (auto-updated on .rs edits)"
fi
if [ -f "$CONTEXT_DIR/ns-validation-status.md" ]; then
  echo "  - ns-validation-status.md — .ns validation results (auto-updated on .ns edits)"
fi
if [ -f "$CONTEXT_DIR/ir-types-summary.md" ]; then
  echo "  - ir-types-summary.md — Core IR data types (enums, structs, type aliases)"
fi
if [ -f "$CONTEXT_DIR/recent-changes.md" ]; then
  echo "  - recent-changes.md — Recent git activity and diff stats"
fi
if [ -f "$CONTEXT_DIR/source-index.md" ]; then
  echo "  - source-index.md — Function/type catalog with line numbers"
fi
if [ -f "$CONTEXT_DIR/call-graph.md" ]; then
  echo "  - call-graph.md — Caller/callee cross-references"
fi
if [ -f "$CONTEXT_DIR/project-status.md" ]; then
  echo "  - project-status.md — Phase, language gaps, stdlib progress"
fi
if [ -f "$CONTEXT_DIR/architecture.md" ]; then
  echo "  - architecture.md — Mermaid diagrams (pipeline, types, module deps)"
fi
if [ -f "$CONTEXT_DIR/session-reflections.md" ]; then
  echo "  - session-reflections.md — Accumulated session learnings"
fi
echo "  Read these files before exploring the codebase."

# Check for pending reflection from previous session
MARKER="$CONTEXT_DIR/pending-reflection.md"
if [ -f "$MARKER" ]; then
  COMMIT_COUNT=$(grep '^Commits made:' "$MARKER" 2>/dev/null | awk '{print $3}')
  echo ""
  echo "Previous session made $COMMIT_COUNT commit(s). Run /reflect to capture learnings."
fi

# Record session start time for the Stop hook
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$CONTEXT_DIR/.session-start-time"

exit 0
