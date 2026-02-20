#!/bin/bash
#
# Neuron Pipeline — Dependency-aware automated neuron creation
#
# Creates NeuroScript neurons one at a time using Claude Code headless mode.
# Respects dependency ordering and tracks progress in manifest.json.
#
# Prerequisites: jq, claude CLI
#
# Usage:
#   ./pipeline.sh              Run the full pipeline
#   ./pipeline.sh --status     Show current progress
#   ./pipeline.sh --next       Show next neuron to create
#   ./pipeline.sh --reset NAME Reset a failed neuron to pending
#   ./pipeline.sh --dry-run    Show creation order without executing
#   ./pipeline.sh --one        Create only the next single neuron
#   ./pipeline.sh --create N   Create a specific neuron by name

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$SCRIPT_DIR/manifest.json"
LOG="$SCRIPT_DIR/progress.log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ──────────────────────────────────────────────────────

log() {
  local msg
  msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$LOG"
}

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

require_jq() {
  command -v jq &>/dev/null || die "jq is required. Install with: brew install jq"
}

require_claude() {
  command -v claude &>/dev/null || die "claude CLI not found in PATH"
}

ensure_binary() {
  local binary="$ROOT_DIR/target/release/neuroscript"
  if [ ! -f "$binary" ]; then
    log "Building neuroscript binary..."
    (cd "$ROOT_DIR" && cargo build --release 2>&1)
  fi
}

# ── Manifest Queries ─────────────────────────────────────────────

# Find the next neuron ready to create:
#   - status == "pending"
#   - attempts < max_attempts
#   - all dependencies either completed in manifest OR not in manifest (already exist)
get_next_neuron() {
  local max_attempts
  max_attempts=$(jq -r '.config.max_attempts // 3' "$MANIFEST")

  jq -r --argjson max "$max_attempts" '
    [.neurons[] | select(.status == "completed") | .name] as $completed |
    [.neurons[] | .name] as $managed |
    [
      .neurons[] |
      select(.status == "pending") |
      select(.attempts < $max) |
      select(
        (.dependencies // []) | all(
          . as $dep |
          if ($managed | index($dep)) then
            ($completed | index($dep)) != null
          else
            true
          end
        )
      )
    ] | first // empty | .name // empty
  ' "$MANIFEST"
}

get_neuron_info() {
  local name="$1"
  jq --arg name "$name" '.neurons[] | select(.name == $name)' "$MANIFEST"
}

update_status() {
  local name="$1" status="$2"
  local tmp
  tmp=$(mktemp)
  jq --arg name "$name" --arg status "$status" \
    '(.neurons[] | select(.name == $name)).status = $status' \
    "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
}

record_attempt() {
  local name="$1" error="${2:-}"
  local tmp
  tmp=$(mktemp)
  jq --arg name "$name" --arg error "$error" \
    '(.neurons[] | select(.name == $name)).attempts += 1 |
     (.neurons[] | select(.name == $name)).last_error = (if $error == "" then null else $error end)' \
    "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
}

# ── Status Display ───────────────────────────────────────────────

show_status() {
  echo ""
  echo -e "${BOLD}Neuron Pipeline Status${NC}"
  echo "-------------------------------------------"

  local total completed pending failed in_progress
  total=$(jq '.neurons | length' "$MANIFEST")
  completed=$(jq '[.neurons[] | select(.status == "completed")] | length' "$MANIFEST")
  pending=$(jq '[.neurons[] | select(.status == "pending")] | length' "$MANIFEST")
  failed=$(jq '[.neurons[] | select(.status == "failed")] | length' "$MANIFEST")
  in_progress=$(jq '[.neurons[] | select(.status == "in_progress")] | length' "$MANIFEST")

  echo -e "  Total:       ${BOLD}$total${NC}"
  echo -e "  ${GREEN}Completed:   $completed${NC}"
  echo -e "  ${YELLOW}Pending:     $pending${NC}"
  echo -e "  ${BLUE}In Progress: $in_progress${NC}"
  echo -e "  ${RED}Failed:      $failed${NC}"
  echo ""

  # Group by level
  local max_level
  max_level=$(jq '[.neurons[].level] | max' "$MANIFEST")

  for level in $(seq 0 "$max_level"); do
    local level_neurons
    level_neurons=$(jq --argjson l "$level" '[.neurons[] | select(.level == $l)] | length' "$MANIFEST")
    if [ "$level_neurons" -gt 0 ]; then
      echo -e "  ${BOLD}Level $level:${NC}"
      jq -r --argjson l "$level" '
        .neurons[] | select(.level == $l) |
        if .status == "completed" then "    \u2705 \(.name)"
        elif .status == "failed" then "    \u274c \(.name) \u2014 \(.last_error // "unknown") [\(.attempts) attempts]"
        elif .status == "in_progress" then "    \ud83d\udd04 \(.name)"
        else "    \u2b1c \(.name)"
        end
      ' "$MANIFEST"
      echo ""
    fi
  done

  local next
  next=$(get_next_neuron)
  if [ -n "$next" ]; then
    echo -e "  ${BLUE}Next up: ${BOLD}$next${NC}"
  else
    if [ "$completed" -eq "$total" ]; then
      echo -e "  ${GREEN}All neurons completed!${NC}"
    else
      echo -e "  ${YELLOW}No neurons ready (check dependencies or reset failed neurons)${NC}"
    fi
  fi
  echo ""
}

# ── Prompt Builder ───────────────────────────────────────────────

build_prompt() {
  local name="$1"
  local info
  info=$(get_neuron_info "$name")

  local desc params in_shape out_shape pattern target deps
  desc=$(echo "$info" | jq -r '.description')
  params=$(echo "$info" | jq -r '.params')
  in_shape=$(echo "$info" | jq -r '.input_shape')
  out_shape=$(echo "$info" | jq -r '.output_shape')
  pattern=$(echo "$info" | jq -r '.pattern')
  target=$(echo "$info" | jq -r '.target_file')
  deps=$(echo "$info" | jq -r '.dependencies | join(", ")')

  cat <<PROMPT
IMPORTANT: Do NOT use plan mode. Do NOT ask for approval. Write the file IMMEDIATELY.

Create a NeuroScript neuron file. Write ONLY this single neuron. Start by writing the file, then validate it.

## Neuron: ${name}(${params})
- Input: ${in_shape}
- Output: ${out_shape}
- Pattern: ${pattern}
- Write to: ${target}

## Description
${desc}

## Available neurons (already exist, use these)
${deps}

## NeuroScript syntax rules
- Shapes: only ONE variadic per shape ([*shape, dim] OK, [*a, *b] INVALID)
- Tuple unpacking supports arbitrary length: -> (a, b, c, ...)
- Fork = 2 outputs: in -> Fork() -> (main, skip)
- Fork3 = 3 outputs: in -> Fork3() -> (a, b, c)
- Add = 2 inputs via tuple: (main, skip) -> Add() -> result
- Multiply = 2 inputs via tuple: (a, b) -> Multiply() -> result
- Concat = exactly 2 inputs. Chain for 3+.
- Pipeline continuation uses indentation (like Python)
- Doc comments use ///
- No let blocks, no underscore in identifiers
- Parameters with defaults: param=default_value

## Example: Simple pipeline
\`\`\`
neuron FFN(dim, expansion):
    in: [*shape, dim]
    out: [*shape, dim]
    graph:
        in ->
            Linear(dim, expansion)
            GELU()
            Linear(expansion, dim)
            out
\`\`\`

## Example: Residual with Fork/Add
\`\`\`
neuron TransformerBlock(dim, num_heads, d_ff):
    in: [batch, seq, dim]
    out: [batch, seq, dim]
    graph:
        in -> Fork() -> (skip1, attn_path)
        attn_path ->
            LayerNorm(dim)
            MultiHeadSelfAttention(dim, num_heads)
            Dropout(0.1)
            attn_out
        (skip1, attn_out) -> Add() -> attn_residual
        attn_residual -> Fork() -> (skip2, ffn_path)
        ffn_path ->
            LayerNorm(dim)
            FFN(dim, d_ff)
            Dropout(0.1)
            ffn_out
        (skip2, ffn_out) -> Add() -> out
\`\`\`

## Example: Gating with Fork/Multiply
\`\`\`
neuron GatedUnit(dim, hidden):
    in: [*shape, dim]
    out: [*shape, hidden]
    graph:
        in -> Fork() -> (gate_path, value_path)
        gate_path ->
            Linear(dim, hidden)
            Sigmoid()
            gate
        value_path ->
            Linear(dim, hidden)
            value
        (gate, value) -> Multiply() -> out
\`\`\`

## Validation (REQUIRED)
After writing the file, run BOTH commands:
  ./target/release/neuroscript validate ${target}
  ./target/release/neuroscript compile ${target} --neuron ${name}

If either fails, read the error, fix the .ns file, and re-run.
Do NOT finish until BOTH pass. This is critical.
PROMPT
}

# ── Neuron Creation ──────────────────────────────────────────────

create_single_neuron() {
  local name="$1"
  local max_turns
  max_turns=$(jq -r '.config.max_turns // 25' "$MANIFEST")

  log "--------------------------------------------"
  log "Creating: $name"

  update_status "$name" "in_progress"

  local prompt
  prompt=$(build_prompt "$name")

  local exit_code=0

  # Run Claude in headless mode with explicit tool permissions
  (cd "$ROOT_DIR" && claude -p "$prompt" \
    --allowedTools "Write,Edit,Read,Bash,Glob,Grep" \
    --max-turns "$max_turns" \
    2>&1) || exit_code=$?

  if [ $exit_code -ne 0 ]; then
    log "Claude exited with code $exit_code for $name"
    record_attempt "$name" "claude exit code $exit_code"
    update_status "$name" "pending"
    return 1
  fi

  # Verify the file was actually created
  local target
  target=$(get_neuron_info "$name" | jq -r '.target_file')
  local full_path="$ROOT_DIR/$target"

  if [ ! -f "$full_path" ]; then
    log "File not created: $target"
    record_attempt "$name" "file not created"
    update_status "$name" "pending"
    return 1
  fi

  # Independent validation (don't trust Claude's self-report)
  local binary="$ROOT_DIR/target/release/neuroscript"
  local val_output val_exit=0
  val_output=$("$binary" validate "$full_path" 2>&1) || val_exit=$?

  if [ $val_exit -ne 0 ]; then
    log "Validation FAILED for $name"
    local err_line
    err_line=$(echo "$val_output" | grep -i "error" | head -1)
    log "  $err_line"
    record_attempt "$name" "validate: $err_line"
    update_status "$name" "pending"
    return 1
  fi

  local comp_output comp_exit=0
  comp_output=$("$binary" compile "$full_path" --neuron "$name" 2>&1) || comp_exit=$?

  if [ $comp_exit -ne 0 ]; then
    log "Compile FAILED for $name"
    local err_line
    err_line=$(echo "$comp_output" | grep -i "error" | head -1)
    log "  $err_line"
    record_attempt "$name" "compile: $err_line"
    update_status "$name" "pending"
    return 1
  fi

  # All checks passed
  log "  $name created and validated"
  update_status "$name" "completed"
  record_attempt "$name" ""
  return 0
}

# ── Dry Run ──────────────────────────────────────────────────────

dry_run() {
  echo -e "\n${BOLD}Dry run — dependency-resolved creation order:${NC}\n"

  # Work on a temporary copy
  local tmp_manifest
  tmp_manifest=$(mktemp)
  cp "$MANIFEST" "$tmp_manifest"

  local order=1
  local max_attempts
  max_attempts=$(jq -r '.config.max_attempts // 3' "$tmp_manifest")

  while true; do
    local next
    next=$(jq -r --argjson max "$max_attempts" '
      [.neurons[] | select(.status == "completed") | .name] as $completed |
      [.neurons[] | .name] as $managed |
      [
        .neurons[] |
        select(.status == "pending") |
        select(.attempts < $max) |
        select(
          (.dependencies // []) | all(
            . as $dep |
            if ($managed | index($dep)) then
              ($completed | index($dep)) != null
            else true end
          )
        )
      ] | first // empty | .name // empty
    ' "$tmp_manifest")

    if [ -z "$next" ]; then break; fi

    local level deps
    level=$(jq -r --arg n "$next" '.neurons[] | select(.name == $n) | .level' "$tmp_manifest")
    deps=$(jq -r --arg n "$next" '.neurons[] | select(.name == $n) | .dependencies | join(", ")' "$tmp_manifest")

    echo -e "  ${BOLD}${order}.${NC} ${BOLD}$next${NC} (L${level})  deps: ${deps}"

    # Simulate completion
    local tmp2
    tmp2=$(mktemp)
    jq --arg n "$next" '(.neurons[] | select(.name == $n)).status = "completed"' \
      "$tmp_manifest" > "$tmp2" && mv "$tmp2" "$tmp_manifest"

    order=$((order + 1))
  done

  rm -f "$tmp_manifest"
  echo -e "\n  ${YELLOW}$((order - 1)) neurons in dependency order${NC}\n"
}

# ── Main Pipeline Loop ───────────────────────────────────────────

run_pipeline() {
  local single_only="${1:-false}"

  ensure_binary

  log "==========================================="
  log "Neuron Pipeline Started"
  log "==========================================="

  local created=0
  local failed=0

  while true; do
    local next
    next=$(get_next_neuron)

    if [ -z "$next" ]; then
      log "No more neurons ready to create"
      break
    fi

    if create_single_neuron "$next"; then
      created=$((created + 1))
    else
      failed=$((failed + 1))

      # Check if max attempts reached
      local attempts max_attempts
      attempts=$(jq -r --arg n "$next" '.neurons[] | select(.name == $n) | .attempts' "$MANIFEST")
      max_attempts=$(jq -r '.config.max_attempts // 3' "$MANIFEST")

      if [ "$attempts" -ge "$max_attempts" ]; then
        log "$next exceeded max attempts ($max_attempts), marking failed"
        update_status "$next" "failed"
      fi
    fi

    # Stop after one if --one mode
    if [ "$single_only" = "true" ]; then
      break
    fi

    # Brief pause between neurons
    sleep 2
  done

  log "==========================================="
  log "Pipeline finished: $created created, $failed failed"
  log "==========================================="

  show_status
}

# ── CLI Entrypoint ───────────────────────────────────────────────
# Guard: only run CLI when executed directly (not when sourced)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  require_jq

  case "${1:-}" in
    --status|-s)
      show_status
      ;;

    --next|-n)
      next=$(get_next_neuron)
      if [ -n "$next" ]; then
        echo -e "${BOLD}$next${NC}"
        get_neuron_info "$next" | jq .
      else
        echo "No neurons ready to create"
      fi
      ;;

    --reset|-r)
      name="${2:?Usage: pipeline.sh --reset NEURON_NAME}"
      update_status "$name" "pending"
      tmp=$(mktemp)
      jq --arg n "$name" \
        '(.neurons[] | select(.name == $n)).attempts = 0 |
         (.neurons[] | select(.name == $n)).last_error = null' \
        "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
      echo "Reset $name to pending"
      ;;

    --dry-run|-d)
      dry_run
      ;;

    --one|-1)
      require_claude
      run_pipeline true
      ;;

    --create|-c)
      name="${2:?Usage: pipeline.sh --create NEURON_NAME}"
      require_claude
      ensure_binary
      # Verify neuron exists in manifest
      info=$(get_neuron_info "$name")
      if [ -z "$info" ]; then
        die "$name not found in manifest"
      fi
      create_single_neuron "$name"
      show_status
      ;;

    --help|-h)
      echo "Neuron Pipeline - Dependency-aware neuron creation"
      echo ""
      echo "Usage: pipeline.sh [OPTION]"
      echo ""
      echo "Options:"
      echo "  (none)              Run the full pipeline (all pending neurons)"
      echo "  --one,    -1        Create only the next single neuron"
      echo "  --create, -c NAME   Create a specific neuron by name"
      echo "  --status, -s        Show current progress"
      echo "  --next,   -n        Show next neuron to create (with details)"
      echo "  --dry-run,-d        Show creation order without executing"
      echo "  --reset,  -r NAME   Reset a failed neuron to pending"
      echo "  --help,   -h        Show this help"
      echo ""
      echo "The manifest (manifest.json) tracks:"
      echo "  - Neuron name, level, dependencies"
      echo "  - Creation status (pending/in_progress/completed/failed)"
      echo "  - Attempt count and last error"
      echo ""
      echo "To add new neurons, edit manifest.json directly."
      ;;

    "")
      require_claude
      run_pipeline false
      ;;

    *)
      die "Unknown option: $1. Use --help for usage."
      ;;
  esac
fi
