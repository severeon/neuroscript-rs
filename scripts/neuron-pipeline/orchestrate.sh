#!/bin/bash
#
# Orchestrator — Multi-agent neuron creation pipeline with batch PRs
#
# Workflow:
#   1. Scan existing stdlib files (--restart) to preserve valid neurons
#   2. Process neurons in batches (3-5 per PR)
#   3. Per batch: create branch, create neurons, commit+push each, open PR
#   4. Per neuron: research → dev agent → validate → commit → push
#   5. Ctrl+C cleanly stops the pipeline at any point
#
# Usage:
#   ./orchestrate.sh              Run full pipeline (all batches)
#   ./orchestrate.sh --one        Create only the next single neuron
#   ./orchestrate.sh --batch N    Run a specific batch by number
#   ./orchestrate.sh --restart    Scan stdlib, preserve valid, reset rest
#   ./orchestrate.sh --dry-run    Show creation order and batches
#   ./orchestrate.sh --status     Show current progress
#   ./orchestrate.sh --help       Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source pipeline.sh for all helper functions (includes signal handling)
source "$SCRIPT_DIR/pipeline.sh"

# ── Config ────────────────────────────────────────────────────────

CONTEXT_DIR="$ROOT_DIR/.context"
NEURONS_DIR="$CONTEXT_DIR/neurons"
RESEARCH_DIR="$CONTEXT_DIR/research"

# Override cleanup to handle batch branch state
CURRENT_BATCH_BRANCH=""
ALL_COMPLETED_NEURONS=""

orchestrator_cleanup() {
  if [ "$INTERRUPTED" = true ]; then return; fi
  INTERRUPTED=true
  echo ""
  echo -e "${YELLOW}Interrupted! Cleaning up...${NC}"

  # Reset any in_progress neurons back to pending
  if [ -f "$MANIFEST" ]; then
    local tmp
    tmp=$(mktemp)
    if jq '(.neurons[] | select(.status == "in_progress")).status = "pending"' \
         "$MANIFEST" > "$tmp" 2>/dev/null; then
      mv "$tmp" "$MANIFEST"
      echo -e "${YELLOW}  Reset in-progress neurons to pending${NC}"
    else
      rm -f "$tmp"
    fi
  fi

  # Push any uncommitted work on the batch branch before leaving
  local active_branch
  active_branch=$(cd "$ROOT_DIR" && git branch --show-current 2>/dev/null) || true

  if [ -n "$CURRENT_BATCH_BRANCH" ] && [ "$active_branch" = "$CURRENT_BATCH_BRANCH" ]; then
    # Stage and commit any work in progress
    (cd "$ROOT_DIR" && {
      git add stdlib/*.ns tests/test_*.py 2>/dev/null || true
      git diff --cached --quiet 2>/dev/null || {
        git commit -m "wip: interrupted pipeline — partial progress

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" 2>/dev/null || true
        git push -u origin "$CURRENT_BATCH_BRANCH" 2>/dev/null || true
        echo -e "${YELLOW}  Pushed partial work to $CURRENT_BATCH_BRANCH${NC}"
      }
    })
  fi

  # Return to main branch
  if [ -n "$active_branch" ] && [ "$active_branch" != "main" ]; then
    echo -e "${YELLOW}  Returning to main branch (was on $active_branch)${NC}"
    (cd "$ROOT_DIR" && git checkout main 2>/dev/null) || true
  fi

  echo -e "${YELLOW}Pipeline stopped. Run --status to see progress.${NC}"
  exit 130
}

trap orchestrator_cleanup SIGINT SIGTERM

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

  if [ ! -f "$full_path" ]; then
    echo "file_not_created"
    return 1
  fi

  local val_output val_exit=0
  val_output=$("$binary" validate "$full_path" 2>&1) || val_exit=$?
  if [ $val_exit -ne 0 ]; then
    local err
    err=$(echo "$val_output" | grep -i "error" | head -1)
    echo "validate: $err"
    return 1
  fi

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

  if [ -f "$ROOT_DIR/tests/test_${snake_name}.py" ]; then
    test_file="\"tests/test_${snake_name}.py\""
  fi

  local research_file="$RESEARCH_DIR/${name}.md"
  if [ -f "$research_file" ]; then
    local paper_line
    paper_line=$(grep -m1 -oE '\([^)]*[0-9]{4}[^)]*\)' "$research_file" 2>/dev/null | head -1 || echo "")
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

# ── Batch Helpers ────────────────────────────────────────────────

get_batch_count() {
  jq '.config.batches | length' "$MANIFEST"
}

get_batch_info() {
  local batch_num="$1"
  jq --argjson n "$batch_num" '.config.batches[$n - 1]' "$MANIFEST"
}

get_batch_neurons() {
  local batch_num="$1"
  jq -r --argjson n "$batch_num" '.config.batches[$n - 1].neurons[]' "$MANIFEST"
}

get_batch_name() {
  local batch_num="$1"
  jq -r --argjson n "$batch_num" '.config.batches[$n - 1].name' "$MANIFEST"
}

# Find the next batch that has at least one pending neuron
get_next_batch() {
  local count
  count=$(get_batch_count)

  for i in $(seq 1 "$count"); do
    local has_pending=false
    while read -r neuron_name; do
      local status
      status=$(jq -r --arg n "$neuron_name" '.neurons[] | select(.name == $n) | .status' "$MANIFEST")
      if [ "$status" = "pending" ]; then
        has_pending=true
        break
      fi
    done < <(get_batch_neurons "$i")

    if [ "$has_pending" = true ]; then
      echo "$i"
      return 0
    fi
  done
  echo ""
}

show_batches() {
  local count
  count=$(get_batch_count)

  echo -e "\n${BOLD}Batch Configuration${NC}"
  echo "-------------------------------------------"

  for i in $(seq 1 "$count"); do
    local name neurons_list
    name=$(get_batch_name "$i")
    echo -e "\n  ${BOLD}Batch $i: $name${NC}"

    while read -r neuron_name; do
      local status
      status=$(jq -r --arg n "$neuron_name" '.neurons[] | select(.name == $n) | .status' "$MANIFEST")
      case "$status" in
        completed)  echo -e "    ${GREEN}✅ $neuron_name${NC}" ;;
        failed)     echo -e "    ${RED}❌ $neuron_name${NC}" ;;
        in_progress) echo -e "    ${BLUE}🔄 $neuron_name${NC}" ;;
        *)          echo -e "    ⬜ $neuron_name" ;;
      esac
    done < <(get_batch_neurons "$i")
  done
  echo ""
}

# ── Per-Neuron Creation (within a batch) ─────────────────────────

create_neuron_in_batch() {
  local name="$1"
  local max_turns
  max_turns=$(jq -r '.config.max_turns // 25' "$MANIFEST")

  log "--------------------------------------------"
  log "Creating: $name"
  log "--------------------------------------------"

  local info target
  info=$(get_neuron_info "$name")
  target=$(echo "$info" | jq -r '.target_file')

  update_status "$name" "in_progress"

  # Step 1: Research + reference test
  log "  [1/5] Research generation"
  "$SCRIPT_DIR/research.sh" "$name" 2>&1 || {
    echo -e "${YELLOW}  Warning: research generation had errors, continuing${NC}" >&2
  }

  # Step 2: Dev agent
  log "  [2/5] Running dev agent"
  local dev_prompt
  dev_prompt=$(build_dev_prompt "$name")

  local dev_exit=0
  (cd "$ROOT_DIR" && claude --agent dev -p "$dev_prompt" \
    --max-turns "$max_turns" \
    2>&1) || dev_exit=$?

  if [ $dev_exit -ne 0 ]; then
    log "  Dev agent exited with code $dev_exit for $name"
  fi

  # Step 3: Independent validation
  log "  [3/5] Independent validation"
  local val_result
  val_result=$(independent_validate "$name" "$target")

  if [ "$val_result" != "ok" ]; then
    log "  Validation FAILED: $val_result"
    record_attempt "$name" "$val_result"

    local attempts max_attempts
    attempts=$(jq -r --arg n "$name" '.neurons[] | select(.name == $n) | .attempts' "$MANIFEST")
    max_attempts=$(jq -r '.config.max_attempts // 3' "$MANIFEST")

    if [ "$attempts" -ge "$max_attempts" ]; then
      log "  $name exceeded max attempts ($max_attempts), marking failed"
      update_status "$name" "failed"
    else
      update_status "$name" "pending"
    fi
    return 1
  fi

  log "  Validation passed!"

  # Step 4: Commit + push
  log "  [4/5] Committing and pushing"
  local snake_name
  snake_name=$(echo "$name" | sed -E 's/([a-z])([A-Z])/\1_\2/g' | tr '[:upper:]' '[:lower:]')

  (cd "$ROOT_DIR" && {
    git add "$target" 2>/dev/null || true
    [ -f "tests/test_${snake_name}.py" ] && git add "tests/test_${snake_name}.py" 2>/dev/null || true
    git diff --cached --quiet 2>/dev/null || {
      git commit -m "feat(stdlib): add ${name} neuron

Validates and compiles successfully.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    }
    git push -u origin "$CURRENT_BATCH_BRANCH" 2>&1 || true
  })

  # Step 5: Update manifest
  log "  [5/5] Updating manifest"
  update_status "$name" "completed"
  record_attempt "$name" ""
  update_manifest_result "$name"

  log "  ✅ $name completed and pushed!"
  return 0
}

# ── Batch Orchestration ──────────────────────────────────────────

orchestrate_batch() {
  local batch_num="$1"
  local batch_name
  batch_name=$(get_batch_name "$batch_num")

  log "============================================"
  log "Batch $batch_num: $batch_name"
  log "============================================"

  # Create batch branch from main
  CURRENT_BATCH_BRANCH="stdlib/batch-${batch_num}-${batch_name}"
  log "Creating branch: $CURRENT_BATCH_BRANCH"

  (cd "$ROOT_DIR" && {
    git checkout main 2>/dev/null || true
    git pull --ff-only origin main 2>/dev/null || true
    git checkout -b "$CURRENT_BATCH_BRANCH" 2>/dev/null || {
      # Branch exists — check it out and rebase on main
      git checkout "$CURRENT_BATCH_BRANCH" 2>/dev/null || {
        log "Warning: Could not create/checkout $CURRENT_BATCH_BRANCH"
      }
    }
  })

  local created=0 failed=0 skipped=0
  local completed_neurons=""

  while read -r neuron_name; do
    # Check if interrupted
    if [ "$INTERRUPTED" = true ]; then
      log "Interrupted, stopping batch"
      break
    fi

    # Skip already completed neurons
    local status
    status=$(jq -r --arg n "$neuron_name" '.neurons[] | select(.name == $n) | .status' "$MANIFEST")
    if [ "$status" = "completed" ]; then
      log "  Skipping $neuron_name (already completed)"
      skipped=$((skipped + 1))
      continue
    fi
    if [ "$status" = "failed" ]; then
      log "  Skipping $neuron_name (failed, use --reset to retry)"
      skipped=$((skipped + 1))
      continue
    fi

    # Check dependencies are met
    local deps_met=true
    while read -r dep; do
      local dep_in_manifest dep_status
      dep_in_manifest=$(jq -r --arg d "$dep" '[.neurons[] | select(.name == $d)] | length' "$MANIFEST")
      if [ "$dep_in_manifest" -gt 0 ]; then
        dep_status=$(jq -r --arg d "$dep" '.neurons[] | select(.name == $d) | .status' "$MANIFEST")
        if [ "$dep_status" != "completed" ]; then
          deps_met=false
          log "  Skipping $neuron_name (dependency $dep not completed)"
          break
        fi
      fi
    done < <(jq -r --arg n "$neuron_name" '.neurons[] | select(.name == $n) | .dependencies[]' "$MANIFEST")

    if [ "$deps_met" = false ]; then
      skipped=$((skipped + 1))
      continue
    fi

    if create_neuron_in_batch "$neuron_name"; then
      created=$((created + 1))
      if [ -n "$completed_neurons" ]; then
        completed_neurons="${completed_neurons}, ${neuron_name}"
      else
        completed_neurons="$neuron_name"
      fi
    else
      failed=$((failed + 1))
    fi

    # Brief pause between neurons
    sleep 2
  done < <(get_batch_neurons "$batch_num")

  # Create PR if we created any neurons
  if [ "$created" -gt 0 ]; then
    log "Creating PR for batch $batch_num"

    # Push final state
    (cd "$ROOT_DIR" && git push -u origin "$CURRENT_BATCH_BRANCH" 2>&1) || true

    local pr_url
    pr_url=$(cd "$ROOT_DIR" && gh pr create \
      --base main \
      --head "$CURRENT_BATCH_BRANCH" \
      --title "feat(stdlib): batch ${batch_num} — ${batch_name}" \
      --body "$(cat <<EOF
## Summary
- **Batch ${batch_num}**: ${batch_name}
- **Neurons created**: ${completed_neurons}
- **Results**: ${created} created, ${failed} failed, ${skipped} skipped

Each neuron has been independently validated (parse + compile).

## Neurons
$(while read -r n; do
    s=$(jq -r --arg n "$n" '.neurons[] | select(.name == $n) | .status' "$MANIFEST")
    case "$s" in
      completed) echo "- [x] $n" ;;
      failed)    echo "- [ ] $n (failed)" ;;
      *)         echo "- [ ] $n (pending)" ;;
    esac
  done < <(get_batch_neurons "$batch_num"))

## Test plan
- [ ] All neurons validate: \`./target/release/neuroscript validate stdlib/<Name>.ns\`
- [ ] All neurons compile: \`./target/release/neuroscript compile stdlib/<Name>.ns\`
- [ ] Reference pytest tests pass (if generated)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)" 2>&1) || {
      log "Warning: PR creation failed (may already exist)"
      pr_url="(failed to create)"
    }

    log "PR created: $pr_url"
  fi

  # Return to main
  (cd "$ROOT_DIR" && git checkout main 2>/dev/null) || true
  CURRENT_BATCH_BRANCH=""

  # Accumulate completed neurons for docs agent
  if [ -n "$completed_neurons" ]; then
    if [ -n "$ALL_COMPLETED_NEURONS" ]; then
      ALL_COMPLETED_NEURONS="${ALL_COMPLETED_NEURONS}, ${completed_neurons}"
    else
      ALL_COMPLETED_NEURONS="$completed_neurons"
    fi
  fi

  log "Batch $batch_num done: $created created, $failed failed, $skipped skipped"
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
  local target_batch="${2:-}"

  require_jq
  require_claude
  ensure_binary

  # Refresh context at start
  refresh_context

  log "============================================"
  log "Neuron Pipeline Orchestrator Started"
  log "============================================"

  show_batches

  ALL_COMPLETED_NEURONS=""

  if [ -n "$target_batch" ]; then
    # Run a specific batch
    orchestrate_batch "$target_batch"
  elif [ "$single_only" = "true" ]; then
    # Create just the next single neuron on its batch branch
    local next
    next=$(get_next_neuron)
    if [ -z "$next" ]; then
      log "No neurons ready to create"
      show_status
      return 0
    fi

    # Find which batch this neuron belongs to
    local count
    count=$(get_batch_count)
    local found_batch=""
    for i in $(seq 1 "$count"); do
      if get_batch_neurons "$i" | grep -qx "$next"; then
        found_batch="$i"
        break
      fi
    done

    if [ -z "$found_batch" ]; then
      log "Neuron $next not assigned to any batch"
      return 1
    fi

    # Set up batch branch for the single neuron
    local batch_name
    batch_name=$(get_batch_name "$found_batch")
    CURRENT_BATCH_BRANCH="stdlib/batch-${found_batch}-${batch_name}"
    log "Creating/checking out branch: $CURRENT_BATCH_BRANCH"

    (cd "$ROOT_DIR" && {
      git checkout main 2>/dev/null || true
      git pull --ff-only origin main 2>/dev/null || true
      git checkout -b "$CURRENT_BATCH_BRANCH" 2>/dev/null || {
        git checkout "$CURRENT_BATCH_BRANCH" 2>/dev/null || true
      }
    })

    # Create just this one neuron
    log "Creating single neuron: $next (batch $found_batch: $batch_name)"
    create_neuron_in_batch "$next" || true

    # Push and return to main
    (cd "$ROOT_DIR" && {
      git push -u origin "$CURRENT_BATCH_BRANCH" 2>&1 || true
      git checkout main 2>/dev/null || true
    })
    CURRENT_BATCH_BRANCH=""
  else
    # Run all batches in order
    local count
    count=$(get_batch_count)

    for i in $(seq 1 "$count"); do
      if [ "$INTERRUPTED" = true ]; then break; fi

      local next_batch
      next_batch=$(get_next_batch)
      if [ -z "$next_batch" ]; then
        log "No more batches with pending neurons"
        break
      fi

      orchestrate_batch "$next_batch"
    done
  fi

  # Run docs agent if we completed any neurons in a full run
  if [ "$single_only" = "false" ] && [ -z "$target_batch" ] && [ -n "$ALL_COMPLETED_NEURONS" ]; then
    run_docs_agent "$ALL_COMPLETED_NEURONS"
  fi

  log "============================================"
  log "Orchestrator finished"
  log "============================================"

  show_status
  show_batches
}

# ── CLI Entrypoint ───────────────────────────────────────────────

case "${1:-}" in
  --status|-s)
    require_jq
    show_status
    show_batches
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
    echo ""
    show_batches
    ;;

  --restart)
    require_jq
    restart_manifest
    show_status
    show_batches
    ;;

  --batch|-b)
    batch_num="${2:?Usage: orchestrate.sh --batch N}"
    run_orchestrator false "$batch_num"
    ;;

  --one|-1)
    run_orchestrator true
    ;;

  --help|-h)
    echo "Neuron Pipeline Orchestrator - Multi-agent neuron creation with batch PRs"
    echo ""
    echo "Usage: orchestrate.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  (none)              Run the full pipeline (all batches)"
    echo "  --one,    -1        Create only the next single neuron"
    echo "  --batch,  -b N      Run a specific batch by number"
    echo "  --restart           Scan stdlib, preserve valid files, reset rest"
    echo "  --status, -s        Show current progress and batch status"
    echo "  --next,   -n        Show next neuron to create (with details)"
    echo "  --dry-run,-d        Show creation order and batch assignments"
    echo "  --help,   -h        Show this help"
    echo ""
    echo "Per-neuron workflow (within a batch):"
    echo "  1. Generate research summary + reference test"
    echo "  2. Dev agent creates and validates neuron"
    echo "  3. Independent validation (parse + compile)"
    echo "  4. Git commit + push to batch branch"
    echo "  5. Update manifest with results"
    echo ""
    echo "Per-batch workflow:"
    echo "  1. Create branch: stdlib/batch-N-name"
    echo "  2. Process all pending neurons in the batch"
    echo "  3. Create PR via gh CLI"
    echo ""
    echo "Ctrl+C cleanly stops the pipeline (saves progress, pushes partial work)"
    ;;

  "")
    run_orchestrator false
    ;;

  *)
    die "Unknown option: $1. Use --help for usage."
    ;;
esac
