#!/bin/bash
#
# Orchestrator — Multi-agent neuron creation pipeline
#
# Replaces run_pipeline() with a research-first, agent-based workflow:
#   1. Refresh compiler context
#   2. Generate research + reference test
#   3. Create feature branch
#   4. Dev agent creates and validates neuron
#   5. Independent validation (don't trust agent)
#   6. Axon agent registers package
#   7. Git commit + merge to main
#   8. Update manifest with results
#   9. Create Obsidian vault note
#  10. Docs agent updates documentation (once, after all neurons)
#
# Usage:
#   ./orchestrate.sh             Run full pipeline (all pending neurons)
#   ./orchestrate.sh --one       Create only the next single neuron
#   ./orchestrate.sh --dry-run   Show creation order without executing
#   ./orchestrate.sh --status    Show current progress

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source pipeline.sh for all helper functions
source "$SCRIPT_DIR/pipeline.sh"

# ── Config ────────────────────────────────────────────────────────

CONTEXT_DIR="$ROOT_DIR/.context"
NEURONS_DIR="$CONTEXT_DIR/neurons"
RESEARCH_DIR="$CONTEXT_DIR/research"

# ── Orchestrator Helpers ──────────────────────────────────────────

refresh_context() {
  local gen_script="$ROOT_DIR/scripts/generate-context.sh"
  if [ -f "$gen_script" ] && [ -x "$gen_script" ]; then
    log "Refreshing compiler context..."
    "$gen_script" all 2>&1 || echo -e "${YELLOW}Warning: context generation had errors${NC}" >&2
  fi
}

build_dev_prompt() {
  local name="$1"
  local base_prompt
  base_prompt=$(build_prompt "$name")

  local research_file="$RESEARCH_DIR/${name}.md"
  local snake_name
  snake_name=$(echo "$name" | sed -E 's/([a-z])([A-Z])/\1_\2/g' | tr '[:upper:]' '[:lower:]')
  local test_file="tests/test_${snake_name}.py"

  local research_section=""
  if [ -f "$research_file" ]; then
    research_section="
## Research Context
A research summary is available at: .context/research/${name}.md
Read it before writing the neuron for mathematical formulation and gotchas."
  fi

  local test_section=""
  if [ -f "$ROOT_DIR/$test_file" ]; then
    test_section="
## Reference Test
A reference PyTorch test exists at: ${test_file}
After creating and validating the .ns file, compile it and run:
  source ~/.venv_ai/bin/activate && pytest ${test_file} -v
The test verifies output shapes and gradient flow."
  fi

  local result_section="
## Result Reporting
After completing (or failing), write a JSON result to .context/neurons/${name}-result.json:
{
  \"status\": \"success\" or \"failed\",
  \"turns_used\": <number of turns you used>,
  \"lessons_learned\": \"<brief notes on what was tricky>\",
  \"shape_confirmed\": true or false,
  \"deviations\": \"<any changes from the description above>\"
}"

  echo "${base_prompt}${research_section}${test_section}${result_section}"
}

independent_validate() {
  local name="$1"
  local target="$2"
  local full_path="$ROOT_DIR/$target"
  local binary="$ROOT_DIR/target/release/neuroscript"

  # File exists?
  if [ ! -f "$full_path" ]; then
    echo "file_not_created"
    return 1
  fi

  # Validate
  local val_output val_exit=0
  val_output=$("$binary" validate "$full_path" 2>&1) || val_exit=$?
  if [ $val_exit -ne 0 ]; then
    local err
    err=$(echo "$val_output" | grep -i "error" | head -1)
    echo "validate: $err"
    return 1
  fi

  # Compile
  local comp_output comp_exit=0
  comp_output=$("$binary" compile "$full_path" --neuron "$name" 2>&1) || comp_exit=$?
  if [ $comp_exit -ne 0 ]; then
    local err
    err=$(echo "$comp_output" | grep -i "error" | head -1)
    echo "compile: $err"
    return 1
  fi

  echo "ok"
  return 0
}

update_manifest_result() {
  local name="$1"
  local result_json="$NEURONS_DIR/${name}-result.json"

  local turns_used="null"
  local lessons_learned="null"
  local paper="null"
  local snake_name
  snake_name=$(echo "$name" | sed -E 's/([a-z])([A-Z])/\1_\2/g' | tr '[:upper:]' '[:lower:]')
  local test_file="null"
  local completed_at
  completed_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  if [ -f "$result_json" ]; then
    turns_used=$(jq '.turns_used // null' "$result_json" 2>/dev/null || echo "null")
    local ll
    ll=$(jq -r '.lessons_learned // ""' "$result_json" 2>/dev/null || echo "")
    if [ -n "$ll" ]; then
      lessons_learned="\"$ll\""
    fi
  fi

  # Check for test file
  if [ -f "$ROOT_DIR/tests/test_${snake_name}.py" ]; then
    test_file="\"tests/test_${snake_name}.py\""
  fi

  # Check for paper in research
  local research_file="$RESEARCH_DIR/${name}.md"
  if [ -f "$research_file" ]; then
    local paper_line
    paper_line=$(grep -m1 -oE '\([^)]*\d{4}[^)]*\)' "$research_file" 2>/dev/null | head -1 || echo "")
    if [ -n "$paper_line" ]; then
      paper="\"$paper_line\""
    fi
  fi

  local tmp
  tmp=$(mktemp)
  jq --arg name "$name" \
     --arg completed_at "$completed_at" \
     --argjson turns_used "$turns_used" \
     --argjson lessons_learned "$lessons_learned" \
     --argjson paper "$paper" \
     --argjson test_file "$test_file" \
    '(.neurons[] | select(.name == $name)) |=
      .completed_at = $completed_at |
      .turns_used = $turns_used |
      .lessons_learned = $lessons_learned |
      .paper = $paper |
      .test_file = $test_file' \
    "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
}

# ── Per-Neuron Orchestration ─────────────────────────────────────

orchestrate_neuron() {
  local name="$1"
  local max_turns
  max_turns=$(jq -r '.config.max_turns // 25' "$MANIFEST")

  log "============================================"
  log "Orchestrating: $name"
  log "============================================"

  local info target
  info=$(get_neuron_info "$name")
  target=$(echo "$info" | jq -r '.target_file')

  update_status "$name" "in_progress"

  # Step 1: Research + reference test
  log "Step 1: Research generation"
  "$SCRIPT_DIR/research.sh" "$name" 2>&1 || {
    echo -e "${YELLOW}Warning: research generation had errors, continuing${NC}" >&2
  }

  # Step 2: Feature branch
  log "Step 2: Creating feature branch"
  local branch="neuron/${name}"
  (cd "$ROOT_DIR" && git checkout -b "$branch" 2>&1) || {
    # Branch may already exist
    (cd "$ROOT_DIR" && git checkout "$branch" 2>&1) || {
      log "Warning: Could not create/checkout branch $branch, working on current branch"
    }
  }

  # Step 3: Dev agent
  log "Step 3: Running dev agent"
  local dev_prompt
  dev_prompt=$(build_dev_prompt "$name")

  local dev_exit=0
  (cd "$ROOT_DIR" && claude --agent dev -p "$dev_prompt" \
    --max-turns "$max_turns" \
    2>&1) || dev_exit=$?

  if [ $dev_exit -ne 0 ]; then
    log "Dev agent exited with code $dev_exit for $name"
  fi

  # Step 4: Independent validation
  log "Step 4: Independent validation"
  local val_result
  val_result=$(independent_validate "$name" "$target")

  if [ "$val_result" != "ok" ]; then
    log "Validation FAILED: $val_result"
    record_attempt "$name" "$val_result"

    # Check max attempts
    local attempts max_attempts
    attempts=$(jq -r --arg n "$name" '.neurons[] | select(.name == $n) | .attempts' "$MANIFEST")
    max_attempts=$(jq -r '.config.max_attempts // 3' "$MANIFEST")

    if [ "$attempts" -ge "$max_attempts" ]; then
      log "$name exceeded max attempts ($max_attempts), marking failed"
      update_status "$name" "failed"
    else
      update_status "$name" "pending"
    fi

    # Return to main branch
    (cd "$ROOT_DIR" && git checkout main 2>/dev/null) || true
    # Clean up failed branch files
    (cd "$ROOT_DIR" && git checkout -- . 2>/dev/null) || true
    return 1
  fi

  log "  Validation passed!"

  # Step 5: Axon agent (package registration)
  log "Step 5: Running axon agent"
  local axon_prompt="Register the neuron '${name}' in Axon.toml. The neuron file is at ${target}. Write status to .context/neurons/${name}-axon.json"
  (cd "$ROOT_DIR" && claude --agent axon -p "$axon_prompt" \
    --max-turns 5 \
    2>&1) || {
    echo -e "${YELLOW}Warning: Axon agent had errors, continuing${NC}" >&2
  }

  # Step 6: Git commit + merge
  log "Step 6: Git commit and merge"
  (cd "$ROOT_DIR" && {
    git add "$target" 2>/dev/null || true

    # Also add test file if it exists
    local snake_name
    snake_name=$(echo "$name" | sed -E 's/([a-z])([A-Z])/\1_\2/g' | tr '[:upper:]' '[:lower:]')
    [ -f "tests/test_${snake_name}.py" ] && git add "tests/test_${snake_name}.py" 2>/dev/null || true

    # Add Axon.toml if modified
    git diff --quiet Axon.toml 2>/dev/null || git add Axon.toml 2>/dev/null || true

    git commit -m "feat(stdlib): add ${name} neuron

Created by neuron pipeline orchestrator.
Validates and compiles successfully.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" 2>/dev/null || true

    git checkout main 2>/dev/null
    git merge "$branch" --no-edit 2>/dev/null || {
      log "Warning: merge had conflicts, staying on branch"
    }
  })

  # Step 7: Update manifest
  log "Step 7: Updating manifest"
  update_status "$name" "completed"
  record_attempt "$name" ""
  update_manifest_result "$name"

  # Step 8: Obsidian note
  log "Step 8: Creating Obsidian note"
  local result_json="$NEURONS_DIR/${name}-result.json"
  if [ -f "$result_json" ]; then
    "$SCRIPT_DIR/obsidian.sh" "$name" "$result_json" 2>&1 || {
      echo -e "${YELLOW}Warning: Obsidian note creation failed${NC}" >&2
    }
  else
    # Create minimal result for obsidian
    mkdir -p "$NEURONS_DIR"
    echo '{"status":"success","turns_used":null,"lessons_learned":""}' > "$result_json"
    "$SCRIPT_DIR/obsidian.sh" "$name" "$result_json" 2>&1 || {
      echo -e "${YELLOW}Warning: Obsidian note creation failed${NC}" >&2
    }
  fi

  log "  $name completed successfully!"
  return 0
}

# ── Docs Agent (runs once after all neurons) ─────────────────────

run_docs_agent() {
  local completed_list="$1"

  if [ -z "$completed_list" ]; then
    log "No new neurons to document"
    return 0
  fi

  log "Running docs agent for: $completed_list"

  local docs_prompt="Update the stdlib documentation for these newly created neurons: ${completed_list}. Read each neuron's .ns file in stdlib/ and update stdlib/README.md with their entries."

  (cd "$ROOT_DIR" && claude --agent docs -p "$docs_prompt" \
    --max-turns 15 \
    2>&1) || {
    echo -e "${YELLOW}Warning: Docs agent had errors${NC}" >&2
  }

  # Commit docs changes if any
  (cd "$ROOT_DIR" && {
    if ! git diff --quiet stdlib/README.md 2>/dev/null; then
      git add stdlib/README.md
      git commit -m "docs(stdlib): update README for ${completed_list}

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" 2>/dev/null || true
    fi
  })
}

# ── Main Orchestration Loop ──────────────────────────────────────

run_orchestrator() {
  local single_only="${1:-false}"

  require_jq
  require_claude
  ensure_binary

  # Refresh context at start
  refresh_context

  log "============================================"
  log "Neuron Pipeline Orchestrator Started"
  log "============================================"

  local created=0
  local failed=0
  local completed_neurons=""

  while true; do
    local next
    next=$(get_next_neuron)

    if [ -z "$next" ]; then
      log "No more neurons ready to create"
      break
    fi

    if orchestrate_neuron "$next"; then
      created=$((created + 1))
      if [ -n "$completed_neurons" ]; then
        completed_neurons="${completed_neurons}, ${next}"
      else
        completed_neurons="$next"
      fi
    else
      failed=$((failed + 1))
    fi

    # Stop after one if --one mode
    if [ "$single_only" = "true" ]; then
      break
    fi

    # Brief pause between neurons
    sleep 2
  done

  # Run docs agent once for all completed neurons
  if [ -n "$completed_neurons" ]; then
    run_docs_agent "$completed_neurons"
  fi

  log "============================================"
  log "Orchestrator finished: $created created, $failed failed"
  log "============================================"

  show_status
}

# ── CLI Entrypoint ───────────────────────────────────────────────

case "${1:-}" in
  --status|-s)
    require_jq
    show_status
    ;;

  --next|-n)
    require_jq
    next=$(get_next_neuron)
    if [ -n "$next" ]; then
      echo -e "${BOLD}$next${NC}"
      get_neuron_info "$next" | jq .
    else
      echo "No neurons ready to create"
    fi
    ;;

  --dry-run|-d)
    require_jq
    dry_run
    ;;

  --one|-1)
    run_orchestrator true
    ;;

  --help|-h)
    echo "Neuron Pipeline Orchestrator - Multi-agent neuron creation"
    echo ""
    echo "Usage: orchestrate.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  (none)              Run the full pipeline (all pending neurons)"
    echo "  --one,    -1        Create only the next single neuron"
    echo "  --status, -s        Show current progress"
    echo "  --next,   -n        Show next neuron to create (with details)"
    echo "  --dry-run,-d        Show creation order without executing"
    echo "  --help,   -h        Show this help"
    echo ""
    echo "Per-neuron workflow:"
    echo "  1. Refresh compiler context"
    echo "  2. Generate research summary + reference test"
    echo "  3. Create feature branch"
    echo "  4. Dev agent creates and validates neuron"
    echo "  5. Independent validation (compile + validate)"
    echo "  6. Axon agent registers package"
    echo "  7. Git commit + merge to main"
    echo "  8. Update manifest with results"
    echo "  9. Create Obsidian vault note"
    echo " 10. Docs agent updates documentation (once, at end)"
    ;;

  "")
    run_orchestrator false
    ;;

  *)
    die "Unknown option: $1. Use --help for usage."
    ;;
esac
