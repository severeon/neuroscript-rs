#!/bin/bash
#
# Research Generator — Produces research summaries and reference tests for neurons
#
# Two-step process using Claude headless mode:
#   Step 1: Generate a research summary (.context/research/<name>.md)
#   Step 2: Generate a reference PyTorch test (tests/test_<name>.py)
#
# Usage: research.sh <neuron_name>
#
# Sources pipeline.sh for helper functions (get_neuron_info, manifest path, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source pipeline.sh for helpers (guarded, won't trigger CLI)
source "$SCRIPT_DIR/pipeline.sh"

# ── Config ────────────────────────────────────────────────────────

RESEARCH_DIR="$ROOT_DIR/.context/research"
RESEARCH_MAX_TURNS=5
TEST_MAX_TURNS=5

# ── Usage ─────────────────────────────────────────────────────────

usage() {
  echo "Usage: research.sh <neuron_name>"
  echo ""
  echo "Generates research summary and reference test for a neuron."
  echo "  Output: .context/research/<name>.md"
  echo "  Output: tests/test_<name>.py"
  exit 1
}

[[ $# -lt 1 ]] && usage

NAME="$1"

# ── Validate neuron exists ────────────────────────────────────────

INFO=$(get_neuron_info "$NAME")
if [ -z "$INFO" ]; then
  die "Neuron '$NAME' not found in manifest"
fi

DESC=$(echo "$INFO" | jq -r '.description')
PARAMS=$(echo "$INFO" | jq -r '.params')
IN_SHAPE=$(echo "$INFO" | jq -r '.input_shape')
OUT_SHAPE=$(echo "$INFO" | jq -r '.output_shape')
PATTERN=$(echo "$INFO" | jq -r '.pattern')
DEPS=$(echo "$INFO" | jq -r '.dependencies | join(", ")')
TARGET=$(echo "$INFO" | jq -r '.target_file')

mkdir -p "$RESEARCH_DIR"

# ── Step 1: Research Summary ──────────────────────────────────────

RESEARCH_FILE="$RESEARCH_DIR/${NAME}.md"

log "Generating research summary for $NAME..."

RESEARCH_PROMPT=$(cat <<PROMPT
Write a concise research summary for the "${NAME}" neural network component. Output ONLY markdown content, no code fences around the whole thing.

## Neuron: ${NAME}(${PARAMS})
- Input: ${IN_SHAPE}
- Output: ${OUT_SHAPE}
- Pattern: ${PATTERN}
- Dependencies: ${DEPS}
- Description: ${DESC}

Write a markdown document covering:

1. **Paper Reference**: Original paper where this architecture was introduced (authors, year, title). If it's a well-known component, cite the most relevant paper.

2. **Mathematical Formulation**: The core math in LaTeX notation. For example, for a gated unit: \$y = \sigma(W_g x) \odot (W_v x)\$

3. **Usage Rationale**: When and why you'd use this over alternatives. What problem does it solve?

4. **Implementation Gotchas**: Common mistakes, numerical stability concerns, initialization tips.

5. **Related Architectures**: How this relates to similar components. For example, GLU vs SwiGLU vs GeGLU.

Keep it concise — aim for 50-100 lines. This will be used as context by an agent creating the NeuroScript implementation.
PROMPT
)

RESEARCH_EXIT=0
RESEARCH_OUTPUT=$(cd "$ROOT_DIR" && claude -p "$RESEARCH_PROMPT" \
  --allowedTools "WebSearch,WebFetch" \
  --max-turns "$RESEARCH_MAX_TURNS" \
  2>&1) || RESEARCH_EXIT=$?

if [ $RESEARCH_EXIT -eq 0 ] && [ -n "$RESEARCH_OUTPUT" ]; then
  echo "$RESEARCH_OUTPUT" > "$RESEARCH_FILE"
  log "  Research summary written to $RESEARCH_FILE"
else
  echo -e "${YELLOW}Warning: Research generation failed for $NAME (exit $RESEARCH_EXIT), continuing without it${NC}" >&2
  # Write a minimal fallback
  cat > "$RESEARCH_FILE" <<EOF
# ${NAME}

Research generation failed. Use manifest description as reference:

${DESC}

Parameters: ${PARAMS}
Input: ${IN_SHAPE}
Output: ${OUT_SHAPE}
Pattern: ${PATTERN}
Dependencies: ${DEPS}
EOF
fi

# ── Step 2: Reference PyTorch Test ────────────────────────────────

# Convert neuron name to snake_case for test filename
SNAKE_NAME=$(echo "$NAME" | sed -E 's/([a-z])([A-Z])/\1_\2/g' | tr '[:upper:]' '[:lower:]')
TEST_FILE="tests/test_${SNAKE_NAME}.py"
FULL_TEST_PATH="$ROOT_DIR/$TEST_FILE"

log "Generating reference test for $NAME..."

TEST_PROMPT=$(cat <<PROMPT
Write a standalone pytest file for testing a "${NAME}" neural network component.

## Neuron: ${NAME}(${PARAMS})
- Input: ${IN_SHAPE}
- Output: ${OUT_SHAPE}
- Pattern: ${PATTERN}
- Description: ${DESC}

Write a Python test file with:

1. **test_${SNAKE_NAME}_shapes()**: Build a reference PyTorch implementation of this component using nn.Module. Create sample input tensors and verify the output shape matches ${OUT_SHAPE}.

2. **test_${SNAKE_NAME}_gradient()**: Verify gradients flow through the component by calling .backward() on the output sum and checking that input.grad is not None.

Requirements:
- Use pytest (no unittest)
- Use torch and torch.nn
- The reference implementation should be a simple nn.Module class called Reference${NAME}
- Use concrete dimension values (e.g., dim=64, hidden_dim=128, batch=2, seq=16)
- Include appropriate shape assertions with descriptive messages
- File should be self-contained (no imports from the project)

Write the file to: ${TEST_FILE}
PROMPT
)

TEST_EXIT=0
TEST_OUTPUT=$(cd "$ROOT_DIR" && claude -p "$TEST_PROMPT" \
  --allowedTools "Write,Read" \
  --max-turns "$TEST_MAX_TURNS" \
  2>&1) || TEST_EXIT=$?

if [ $TEST_EXIT -eq 0 ] && [ -f "$FULL_TEST_PATH" ]; then
  log "  Reference test written to $TEST_FILE"
else
  echo -e "${YELLOW}Warning: Test generation failed for $NAME (exit $TEST_EXIT), continuing without it${NC}" >&2
  TEST_FILE=""
fi

# ── Report ────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}Research generation complete for ${BOLD}$NAME${NC}"
echo -e "  Research: ${RESEARCH_FILE}"
if [ -n "$TEST_FILE" ]; then
  echo -e "  Test:     ${TEST_FILE}"
else
  echo -e "  Test:     ${YELLOW}(not generated)${NC}"
fi
