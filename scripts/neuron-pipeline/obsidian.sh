#!/bin/bash
#
# Obsidian Note Creator — Creates structured vault notes for completed neurons
#
# Creates a note with YAML frontmatter and structured content for each neuron.
# Write method:
#   - If $OBSIDIAN_VAULT is set → direct file write
#   - Otherwise → claude -p with mcp__obsidian__write_note
#
# Usage: obsidian.sh <neuron_name> <result_json_path>
#
# Sources pipeline.sh for helper functions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source pipeline.sh for helpers
source "$SCRIPT_DIR/pipeline.sh"

# ── Usage ─────────────────────────────────────────────────────────

usage() {
  echo "Usage: obsidian.sh <neuron_name> <result_json_path>"
  echo ""
  echo "Creates an Obsidian vault note for a completed neuron."
  echo "Set OBSIDIAN_VAULT env var for direct file write."
  exit 1
}

[[ $# -lt 2 ]] && usage

NAME="$1"
RESULT_JSON="$2"

# ── Gather metadata ──────────────────────────────────────────────

INFO=$(get_neuron_info "$NAME")
if [ -z "$INFO" ]; then
  die "Neuron '$NAME' not found in manifest"
fi

PARAMS=$(echo "$INFO" | jq -r '.params')
IN_SHAPE=$(echo "$INFO" | jq -r '.input_shape')
OUT_SHAPE=$(echo "$INFO" | jq -r '.output_shape')
PATTERN=$(echo "$INFO" | jq -r '.pattern')
LEVEL=$(echo "$INFO" | jq -r '.level')
DESC=$(echo "$INFO" | jq -r '.description')
DEPS=$(echo "$INFO" | jq -r '.dependencies | join(", ")')
TARGET=$(echo "$INFO" | jq -r '.target_file')

# Read result JSON if it exists
TURNS_USED="unknown"
LESSONS=""
STATUS="unknown"
PAPER=""

if [ -f "$RESULT_JSON" ]; then
  TURNS_USED=$(jq -r '.turns_used // "unknown"' "$RESULT_JSON" 2>/dev/null || echo "unknown")
  LESSONS=$(jq -r '.lessons_learned // ""' "$RESULT_JSON" 2>/dev/null || echo "")
  STATUS=$(jq -r '.status // "unknown"' "$RESULT_JSON" 2>/dev/null || echo "unknown")
fi

# Check for paper reference in research file
RESEARCH_FILE="$ROOT_DIR/.context/research/${NAME}.md"
if [ -f "$RESEARCH_FILE" ]; then
  PAPER=$(grep -m1 -iE '^\*?\*?paper|citation|reference' "$RESEARCH_FILE" 2>/dev/null | head -1 || echo "")
fi

COMPLETED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ── Build note content ───────────────────────────────────────────

NOTE_CONTENT=$(cat <<EOF
---
tags:
  - neuroscript
  - stdlib
  - ${PATTERN}
status: ${STATUS}
neuron: ${NAME}
params: "${PARAMS}"
input_shape: "${IN_SHAPE}"
output_shape: "${OUT_SHAPE}"
level: ${LEVEL}
turns_used: ${TURNS_USED}
completed_at: ${COMPLETED_AT}
---

# ${NAME}

${DESC}

## Parameters

\`${NAME}(${PARAMS})\`

| Port | Shape |
|------|-------|
| Input | \`${IN_SHAPE}\` |
| Output | \`${OUT_SHAPE}\` |

## Dependencies

${DEPS}

## Source

\`${TARGET}\`

## Implementation Notes

${LESSONS:-No lessons recorded.}

## Related Neurons

See also: ${DEPS}
EOF
)

# ── Write to vault ───────────────────────────────────────────────

NOTE_PATH="neuroscript/stdlib/${NAME}.md"

if [ -n "${OBSIDIAN_VAULT:-}" ]; then
  # Direct file write
  FULL_PATH="${OBSIDIAN_VAULT}/${NOTE_PATH}"
  mkdir -p "$(dirname "$FULL_PATH")"
  echo "$NOTE_CONTENT" > "$FULL_PATH"
  log "Obsidian note written to $FULL_PATH"
else
  # Use Claude with Obsidian MCP tool
  OBSIDIAN_PROMPT=$(cat <<PROMPT
Write this exact content as an Obsidian note at path "${NOTE_PATH}" using the mcp__obsidian__write_note tool. Do not modify the content.

${NOTE_CONTENT}
PROMPT
)

  OBSIDIAN_EXIT=0
  cd "$ROOT_DIR" && claude -p "$OBSIDIAN_PROMPT" \
    --allowedTools "mcp__obsidian__write_note" \
    --max-turns 3 \
    2>&1 || OBSIDIAN_EXIT=$?

  if [ $OBSIDIAN_EXIT -eq 0 ]; then
    log "Obsidian note created via MCP: $NOTE_PATH"
  else
    echo -e "${YELLOW}Warning: Obsidian note creation failed (exit $OBSIDIAN_EXIT)${NC}" >&2
    echo -e "${YELLOW}Note content saved to .context/neurons/${NAME}-note.md${NC}" >&2
    mkdir -p "$ROOT_DIR/.context/neurons"
    echo "$NOTE_CONTENT" > "$ROOT_DIR/.context/neurons/${NAME}-note.md"
  fi
fi
