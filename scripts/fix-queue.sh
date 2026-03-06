#!/usr/bin/env bash
# fix-queue.sh — Automated fix queue orchestrator
#
# Processes review findings from fix-queue.json:
#   1. Reads next pending task
#   2. Creates feature branch from dev
#   3. Invokes claude (haiku by default) to verify + fix
#   4. Creates PR targeting dev
#   5. Prompts for terminal approval
#   6. Merges on approve, updates context
#
# Usage:
#   ./scripts/fix-queue.sh              # Process all pending tasks
#   ./scripts/fix-queue.sh --dry-run    # Show what would be done
#   ./scripts/fix-queue.sh --skip-to ID # Start from specific task ID
#   ./scripts/fix-queue.sh --one        # Process only the next task

set -euo pipefail

# ── Cleanup on Ctrl+C / exit ──────────────────────────────────────────
CHILD_PIDS=()
CURRENT_TASK_IDX=""
cleanup() {
  printf '\r\033[K'  # Clear any spinner line
  # Kill background monitors
  for pid in "${CHILD_PIDS[@]}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
  done
  wait 2>/dev/null || true
  # Reset in_progress task back to pending
  if [[ -n "$CURRENT_TASK_IDX" ]]; then
    queue_set "$CURRENT_TASK_IDX" status "pending" 2>/dev/null || true
  fi
  # Revert any uncommitted changes
  cd "$PROJECT_DIR" 2>/dev/null
  git checkout -- . 2>/dev/null || true
  git clean -fd 2>/dev/null || true
  git checkout "$DEV_BRANCH" --quiet 2>/dev/null || true
  rm -f /tmp/fix-queue-output.* /tmp/fix-queue-status.txt /tmp/fix-queue-status.txt.events
  echo ""
  err "Interrupted. Task reset to pending — re-run to resume."
  exit 130
}
trap cleanup INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
QUEUE_FILE="$PROJECT_DIR/docs/reviews/fix-queue.json"
FIX_LOG="$PROJECT_DIR/docs/reviews/fix-log.md"
DEV_BRANCH="dev"
REPO="severeon/neuroscript-rs"

HAIKU_MODEL="claude-haiku-4-5-20251001"
SONNET_MODEL="claude-sonnet-4-6"
OPUS_MODEL="claude-opus-4-6"
TASK_TIMEOUT=600  # 10 minutes per attempt

CONTEXT_FILE="$PROJECT_DIR/.claude/context/fix-context.md"
STATUS_FILE="/tmp/fix-queue-status.txt"

DRY_RUN=false
ONE_ONLY=false
SKIP_TO=""

# Session-level timing
SESSION_AGENT_SECS=0
SESSION_REVIEW_SECS=0

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=true; shift ;;
    --one)      ONE_ONLY=true; shift ;;
    --skip-to)  SKIP_TO="$2"; shift 2 ;;
    *)          echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────
log() { echo -e "\033[1;36m▸\033[0m $*"; }
warn() { echo -e "\033[1;33m⚠\033[0m $*"; }
err() { echo -e "\033[1;31m✗\033[0m $*" >&2; }
ok() { echo -e "\033[1;32m✓\033[0m $*"; }
divider() {
  echo ""
  echo -e "\033[1;90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
}

fmt_duration() {
  local secs=$1
  if (( secs >= 3600 )); then
    printf '%dh%02dm%02ds' $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
  elif (( secs >= 60 )); then
    printf '%dm%02ds' $((secs/60)) $((secs%60))
  else
    printf '%ds' "$secs"
  fi
}

# Write status to file + update terminal spinner line
status_update() {
  local id="$1" tier="$2" elapsed="$3" msg="$4"
  # Write to status file (watchable externally)
  printf '%s | %s | %s | %s | %s\n' \
    "$(date '+%H:%M:%S')" "$id" "$tier" "$(fmt_duration "$elapsed")" "$msg" \
    > "$STATUS_FILE"
  # Update in-place terminal line
  printf '\r\033[K  \033[90m⏱ %s\033[0m \033[36m[%s]\033[0m %s' \
    "$(fmt_duration "$elapsed")" "$tier" "$msg"
}

# Background monitor: tails output file for stream-json tool events
start_monitor() {
  local id="$1" tier="$2" output_file="$3" start_time="$4"
  : > "$STATUS_FILE"
  (
    # Long-running python process that reads stream-json from tail -f
    tail -f "$output_file" 2>/dev/null | python3 -u -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        obj = json.loads(line)
        if obj.get('type') == 'tool_use':
            name = obj.get('name', '')
            inp = obj.get('input', {})
            if name == 'Bash':
                msg = f'Running: {inp.get(\"command\", \"?\")[:55]}'
            elif name in ('Read', 'Edit', 'Write'):
                fp = inp.get('file_path', '?')
                msg = f'{name}: {fp.split(\"/\")[-1]}'
            elif name == 'Grep':
                msg = f'Searching: {inp.get(\"pattern\", \"?\")[:40]}'
            elif name == 'Glob':
                msg = f'Finding: {inp.get(\"pattern\", \"?\")[:40]}'
            else:
                msg = name
            print(msg, flush=True)
    except: pass
" 2>/dev/null > "$STATUS_FILE.events" &
    local parser_pid=$!

    # Timer loop: update terminal with elapsed time + last event
    while kill -0 $parser_pid 2>/dev/null; do
      sleep 2
      local elapsed=$(( $(date +%s) - start_time ))
      local last_msg="working..."
      if [[ -s "$STATUS_FILE.events" ]]; then
        last_msg=$(tail -1 "$STATUS_FILE.events" 2>/dev/null | head -c 60)
      fi
      status_update "$id" "$tier" "$elapsed" "${last_msg:-working...}"
    done
  ) &
  local pid=$!
  CHILD_PIDS+=("$pid")
  echo "$pid"
}

stop_monitor() {
  local pid="$1"
  kill "$pid" 2>/dev/null
  # Also kill any children of the monitor (tail, python)
  pkill -P "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  printf '\r\033[K'  # Clear the status line
  rm -f "$STATUS_FILE.events"
  # Remove from CHILD_PIDS
  CHILD_PIDS=("${CHILD_PIDS[@]/$pid/}")
}

# Read a field from the queue for a given index
queue_get() {
  python3 -c "
import json, sys
q = json.load(open('$QUEUE_FILE'))
print(q[$1].get('$2', '') or '')
"
}

# Update a field in the queue
queue_set() {
  local idx="$1" field="$2" value="$3"
  python3 -c "
import json
q = json.load(open('$QUEUE_FILE'))
q[$idx]['$field'] = '$value'
json.dump(q, open('$QUEUE_FILE', 'w'), indent=2)
"
}

queue_set_null() {
  local idx="$1" field="$2"
  python3 -c "
import json
q = json.load(open('$QUEUE_FILE'))
q[$idx]['$field'] = None
json.dump(q, open('$QUEUE_FILE', 'w'), indent=2)
"
}

# Get the total count and next pending index
queue_stats() {
  python3 -c "
import json
q = json.load(open('$QUEUE_FILE'))
total = len(q)
done = sum(1 for i in q if i['status'] in ('done', 'skipped', 'rejected', 'needs_review'))
pending = [i for i, t in enumerate(q) if t['status'] == 'pending']
skip_to = '$SKIP_TO'
if skip_to:
    pending = [i for i in pending if q[i]['id'] == skip_to] + [i for i in pending if q[i]['id'] != skip_to]
next_idx = pending[0] if pending else -1
print(f'{total} {done} {next_idx}')
"
}

# ── Ensure dev branch exists ──────────────────────────────────────────
ensure_dev_branch() {
  cd "$PROJECT_DIR"
  if ! git rev-parse --verify "$DEV_BRANCH" &>/dev/null; then
    log "Creating $DEV_BRANCH branch from main..."
    git branch "$DEV_BRANCH" main
  fi
}

# ── Regenerate context ────────────────────────────────────────────────
regen_context() {
  log "Regenerating context..."
  "$PROJECT_DIR/scripts/generate-context.sh" recent &>/dev/null || true
  rebuild_fix_context 2>/dev/null || true
}

# ── Rebuild shared context ────────────────────────────────────────────
rebuild_fix_context() {
  "$SCRIPT_DIR/build-fix-context.sh" 2>/dev/null
}

# ── Build the claude prompt for a task ────────────────────────────────
build_prompt() {
  local idx="$1"
  local id summary file line detail
  id=$(queue_get "$idx" id)
  summary=$(queue_get "$idx" summary)
  file=$(queue_get "$idx" file)
  line=$(queue_get "$idx" line)
  detail=$(queue_get "$idx" detail)

  cat <<PROMPT
## Task: $id — $summary

**File:** ${file:-"(not specified)"}
**Line:** ${line:-"(not specified)"}

**Details:**
$detail

## Instructions

DO NOT explore the codebase. Go directly to the file(s) listed above.

1. **Verify** (1 read): Read the referenced file at the specified line range. If the issue is already fixed, output EXACTLY: SKIP: <reason>

2. **Fix**: Make the minimal change. Only touch files directly related to this issue.

3. **Test**: Run \`cargo check\`. If test code changed, run \`cargo test <module>\`. For .ns files, run \`./target/release/neuroscript validate <file>\`.

4. **Final check**: Run \`cargo test\` to confirm nothing broke. If tests fail, fix or revert.

## Hard Rules

- Do NOT read files unrelated to this task
- Do NOT refactor surrounding code or add comments to unchanged code
- Do NOT add documentation unless the task requires it
- Keep the diff as small as possible
- If you are uncertain about the fix, make the conservative choice

PROMPT
}

# ── Process one task ──────────────────────────────────────────────────
process_task() {
  local idx="$1"
  local id severity summary file model_tier
  id=$(queue_get "$idx" id)
  severity=$(queue_get "$idx" severity)
  summary=$(queue_get "$idx" summary)
  file=$(queue_get "$idx" file)
  model_tier=$(queue_get "$idx" model)

  # Pick model
  local model="$HAIKU_MODEL"
  if [[ "$model_tier" == "sonnet" ]]; then
    model="$SONNET_MODEL"
  fi

  local stats
  stats=$(queue_stats)
  local total done _next
  total=$(echo "$stats" | cut -d' ' -f1)
  done=$(echo "$stats" | cut -d' ' -f2)
  local progress="$((done + 1))/$total"

  divider
  echo -e "\033[1mTask $progress: $id\033[0m ($severity) [$model_tier]"
  echo "  $summary"
  if [[ -n "$file" ]]; then
    echo "  File: $file"
  fi
  divider

  if $DRY_RUN; then
    log "[dry-run] Would process $id with $model_tier model"
    return 0
  fi

  # Mark in progress (and track for cleanup)
  CURRENT_TASK_IDX="$idx"
  queue_set "$idx" status "in_progress"

  # Create feature branch
  local branch="fix/${id}"
  cd "$PROJECT_DIR"
  git checkout "$DEV_BRANCH" --quiet || {
    err "Failed to checkout $DEV_BRANCH branch"
    return 1
  }
  git pull --ff-only origin "$DEV_BRANCH" --quiet 2>/dev/null || true
  if git rev-parse --verify "$branch" &>/dev/null; then
    # Branch exists, reset it to dev
    git checkout "$branch" --quiet
    git reset --hard "$DEV_BRANCH" --quiet
  else
    git checkout -b "$branch" "$DEV_BRANCH" --quiet
  fi

  queue_set "$idx" branch "$branch"

  # Build prompt and context
  local prompt
  prompt=$(build_prompt "$idx")

  # Rebuild shared context (includes fix log, recent commits, type signatures)
  rebuild_fix_context

  local system_prompt
  system_prompt=$(cat "$CONTEXT_FILE")

  # ── Escalation chain: haiku → sonnet → opus → needs_review ──────
  # Determine starting model tier and build escalation chain
  local -a model_chain
  case "$model_tier" in
    haiku)  model_chain=("$HAIKU_MODEL" "$SONNET_MODEL" "$OPUS_MODEL") ;;
    sonnet) model_chain=("$SONNET_MODEL" "$OPUS_MODEL") ;;
    opus)   model_chain=("$OPUS_MODEL") ;;
    *)      model_chain=("$HAIKU_MODEL" "$SONNET_MODEL" "$OPUS_MODEL") ;;
  esac

  local tier_names
  case "$model_tier" in
    haiku)  tier_names=("haiku" "sonnet" "opus") ;;
    sonnet) tier_names=("sonnet" "opus") ;;
    opus)   tier_names=("opus") ;;
    *)      tier_names=("haiku" "sonnet" "opus") ;;
  esac

  local attempt=0
  local succeeded=false
  local prior_output=""
  local task_agent_secs=0

  for curr_model in "${model_chain[@]}"; do
    local curr_tier="${tier_names[$attempt]}"
    attempt=$((attempt + 1))

    # Build escalated prompt if retrying
    local full_prompt="$prompt"
    if [[ -n "$prior_output" ]]; then
      full_prompt="$prompt

## Prior Attempt (${tier_names[$((attempt-2))]} model — timed out or failed)

The previous model attempted this fix but ran out of time. Here is its output so far — continue from where it left off. Do NOT restart from scratch.

<prior-attempt>
$(echo "$prior_output" | tail -100)
</prior-attempt>"
    fi

    log "Attempt $attempt: running $curr_tier (${TASK_TIMEOUT}s timeout)..."

    # Write prompt to temp file (claude reads stdin)
    local output_file
    output_file=$(mktemp /tmp/fix-queue-output.XXXXXX)

    local attempt_start
    attempt_start=$(date +%s)

    # Start background monitor for live status
    local monitor_pid
    monitor_pid=$(start_monitor "$id" "$curr_tier" "$output_file" "$attempt_start")

    # Run claude with stream-json so monitor can parse tool events live
    local claude_exit=0
    timeout "$TASK_TIMEOUT" env CLAUDECODE="" claude \
      --model "$curr_model" \
      --system-prompt "$system_prompt" \
      --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
      --output-format stream-json \
      --max-turns 25 \
      -p "$full_prompt" \
      > "$output_file" 2>&1 || claude_exit=$?

    # Stop monitor, clear status line
    stop_monitor "$monitor_pid"

    local attempt_elapsed=$(( $(date +%s) - attempt_start ))
    task_agent_secs=$((task_agent_secs + attempt_elapsed))

    # Extract text content from stream-json output
    local claude_raw
    claude_raw=$(cat "$output_file")
    local claude_output
    claude_output=$(echo "$claude_raw" | python3 -c "
import sys, json
lines = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        obj = json.loads(line)
        if obj.get('type') == 'assistant' and 'content' in obj:
            for block in obj['content']:
                if isinstance(block, dict) and block.get('type') == 'text':
                    lines.append(block['text'])
        elif obj.get('type') == 'result' and 'content' in obj:
            lines.append(str(obj['content']))
    except: pass
print('\n'.join(lines))
" 2>/dev/null || echo "$claude_raw")
    rm -f "$output_file"

    # timeout returns 124 on timeout
    if [[ $claude_exit -eq 124 ]]; then
      warn "$curr_tier timed out after $(fmt_duration $attempt_elapsed)."
      prior_output="$claude_output"
      # Reset any partial changes before escalating
      git checkout -- . 2>/dev/null || true
      git clean -fd 2>/dev/null || true
      continue
    fi

    ok "$curr_tier finished in $(fmt_duration $attempt_elapsed)."

    # Check for SKIP
    if echo "$claude_output" | grep -q "SKIP:"; then
      local skip_reason
      skip_reason=$(echo "$claude_output" | grep "SKIP:" | head -1 | sed 's/.*SKIP: *//')
      ok "Skipped: $skip_reason"
      queue_set "$idx" status "skipped"
      queue_set "$idx" skip_reason "$skip_reason"
      SESSION_AGENT_SECS=$((SESSION_AGENT_SECS + task_agent_secs))
      CURRENT_TASK_IDX=""
      git checkout "$DEV_BRANCH" --quiet
      git branch -D "$branch" --quiet 2>/dev/null || true
      return 0
    fi

    # Check if any files were actually changed
    if [[ -z "$(git status --porcelain)" ]]; then
      warn "$curr_tier made no changes."
      prior_output="$claude_output"
      continue
    fi

    # Check if cargo check passes
    if ! cargo check --quiet 2>/dev/null; then
      warn "$curr_tier changes don't compile."
      prior_output="$claude_output
(Note: cargo check failed after these changes)"
      git checkout -- . 2>/dev/null || true
      git clean -fd 2>/dev/null || true
      continue
    fi

    succeeded=true
    ok "Fix applied by $curr_tier model in $(fmt_duration $task_agent_secs) total."
    break
  done

  SESSION_AGENT_SECS=$((SESSION_AGENT_SECS + task_agent_secs))

  # All models failed — mark for manual review
  if ! $succeeded; then
    err "All models failed for $id. Marking for manual review."
    queue_set "$idx" status "needs_review"
    queue_set "$idx" skip_reason "All models failed (haiku/sonnet/opus)"
    CURRENT_TASK_IDX=""
    git checkout -- . 2>/dev/null || true
    git clean -fd 2>/dev/null || true
    git checkout "$DEV_BRANCH" --quiet
    git branch -D "$branch" --quiet 2>/dev/null || true
    return 0
  fi

  # Commit changes
  git add -A
  git commit -m "fix($id): $summary

Automated fix from review queue.
Model: $model_tier

Co-Authored-By: Claude <noreply@anthropic.com>" --quiet

  # Push and create PR
  git push -u origin "$branch" --quiet 2>/dev/null
  local pr_url
  pr_url=$(gh pr create \
    --base "$DEV_BRANCH" \
    --title "fix($id): $summary" \
    --body "$(cat <<EOF
## Fix: $id

**Severity:** $severity
**Source:** $(queue_get "$idx" source)

### Issue
$(queue_get "$idx" detail)

### Changes
$(git diff "$DEV_BRANCH"..."$branch" --stat)

---
Automated fix from review queue. Model: $model_tier
EOF
)" 2>/dev/null) || {
    err "Failed to create PR"
    queue_set "$idx" status "pending"
    git checkout "$DEV_BRANCH" --quiet
    return 1
  }

  local pr_num
  pr_num=$(echo "$pr_url" | grep -oE '[0-9]+$')
  queue_set "$idx" pr_number "$pr_num"

  ok "PR #$pr_num created: $pr_url"
  echo ""

  # ── Terminal approval loop (time NOT counted toward agent execution) ──
  local review_start
  review_start=$(date +%s)
  echo -e "  \033[90m⏱ Agent time: $(fmt_duration $task_agent_secs) | Review timer started...\033[0m"
  while true; do
    echo -en "\033[1mCommands:\033[0m [a]pprove  [r]eject  [d]iff  [s]kip  > "
    read -r cmd </dev/tty

    case "$cmd" in
      a|approve)
        log "Merging PR #$pr_num..."
        gh pr merge "$pr_num" --squash --delete-branch --repo "$REPO" 2>/dev/null || {
          # Try merge commit if squash fails
          gh pr merge "$pr_num" --merge --delete-branch --repo "$REPO" 2>/dev/null || {
            err "Merge failed. Please merge manually."
            break
          }
        }
        ok "Merged!"
        queue_set "$idx" status "done"

        # Update fix log
        if [[ ! -f "$FIX_LOG" ]]; then
          echo "# Fix Log" > "$FIX_LOG"
          echo "" >> "$FIX_LOG"
          echo "Automated fixes from review queue." >> "$FIX_LOG"
          echo "" >> "$FIX_LOG"
        fi
        echo "- **$id** ($severity): $summary — PR #$pr_num" >> "$FIX_LOG"

        # Update dev branch locally
        git checkout "$DEV_BRANCH" --quiet
        git pull --ff-only origin "$DEV_BRANCH" --quiet 2>/dev/null || true

        # Regen context for next task
        regen_context
        break
        ;;

      r|reject)
        log "Closing PR #$pr_num..."
        gh pr close "$pr_num" --repo "$REPO" 2>/dev/null || true
        queue_set "$idx" status "rejected"
        git checkout "$DEV_BRANCH" --quiet
        git branch -D "$branch" --quiet 2>/dev/null || true
        git push origin --delete "$branch" --quiet 2>/dev/null || true
        warn "Rejected."
        break
        ;;

      d|diff)
        gh pr diff "$pr_num" --repo "$REPO" 2>/dev/null | ${PAGER:-less}
        ;;

      s|skip)
        log "Skipping (leaving PR open)..."
        queue_set "$idx" status "skipped"
        queue_set "$idx" skip_reason "Manually skipped during review"
        git checkout "$DEV_BRANCH" --quiet
        warn "Skipped. PR #$pr_num left open."
        break
        ;;

      *)
        echo "  Unknown command. Use: a(pprove), r(eject), d(iff), s(kip)"
        ;;
    esac
  done

  local review_elapsed=$(( $(date +%s) - review_start ))
  SESSION_REVIEW_SECS=$((SESSION_REVIEW_SECS + review_elapsed))
  echo -e "  \033[90m⏱ Task: agent $(fmt_duration $task_agent_secs) + review $(fmt_duration $review_elapsed)\033[0m"
  CURRENT_TASK_IDX=""
}

# ── Main loop ─────────────────────────────────────────────────────────
main() {
  cd "$PROJECT_DIR"

  if [[ ! -f "$QUEUE_FILE" ]]; then
    err "Queue file not found: $QUEUE_FILE"
    exit 1
  fi

  ensure_dev_branch

  # Push dev branch if it doesn't exist on remote
  if ! git ls-remote --heads origin "$DEV_BRANCH" | grep -q "$DEV_BRANCH"; then
    log "Pushing $DEV_BRANCH to origin..."
    git push -u origin "$DEV_BRANCH" --quiet
  fi

  log "Fix Queue Orchestrator"
  local stats
  stats=$(queue_stats)
  local total done next_idx
  total=$(echo "$stats" | cut -d' ' -f1)
  done=$(echo "$stats" | cut -d' ' -f2)
  next_idx=$(echo "$stats" | cut -d' ' -f3)

  log "$done/$total complete, $((total - done)) remaining"

  if [[ "$next_idx" == "-1" ]]; then
    ok "All tasks processed!"
    exit 0
  fi

  # Process tasks
  if $DRY_RUN; then
    # In dry-run, iterate all pending tasks (status won't change)
    local -a pending_indices
    IFS=' ' read -ra pending_indices <<< "$(python3 -c "
import json
q = json.load(open('$QUEUE_FILE'))
skip_to = '$SKIP_TO'
pending = [i for i, t in enumerate(q) if t['status'] == 'pending']
if skip_to:
    pending = [i for i in pending if q[i]['id'] == skip_to] + [i for i in pending if q[i]['id'] != skip_to]
print(' '.join(str(i) for i in pending))
")"
    for idx in "${pending_indices[@]}"; do
      [[ -z "$idx" ]] && continue
      process_task "$idx"
      if $ONE_ONLY; then
        log "Stopping after one task (--one flag)."
        break
      fi
    done
  else
    while true; do
      stats=$(queue_stats)
      next_idx=$(echo "$stats" | cut -d' ' -f3)

      if [[ "$next_idx" == "-1" ]]; then
        ok "All tasks processed!"
        break
      fi

      process_task "$next_idx"

      if $ONE_ONLY; then
        log "Stopping after one task (--one flag)."
        break
      fi

      # Clear skip-to after first iteration
      SKIP_TO=""
    done
  fi

  # Final summary
  divider
  stats=$(queue_stats)
  total=$(echo "$stats" | cut -d' ' -f1)
  done=$(echo "$stats" | cut -d' ' -f2)
  log "Session complete: $done/$total tasks processed."

  local done_count skipped_count rejected_count needs_review_count
  done_count=$(python3 -c "import json; q=json.load(open('$QUEUE_FILE')); print(sum(1 for i in q if i['status']=='done'))")
  skipped_count=$(python3 -c "import json; q=json.load(open('$QUEUE_FILE')); print(sum(1 for i in q if i['status']=='skipped'))")
  rejected_count=$(python3 -c "import json; q=json.load(open('$QUEUE_FILE')); print(sum(1 for i in q if i['status']=='rejected'))")
  needs_review_count=$(python3 -c "import json; q=json.load(open('$QUEUE_FILE')); print(sum(1 for i in q if i['status']=='needs_review'))")

  echo "  Done: $done_count | Skipped: $skipped_count | Rejected: $rejected_count | Needs Review: $needs_review_count"
  echo -e "  \033[90m⏱ Agent: $(fmt_duration $SESSION_AGENT_SECS) | Review: $(fmt_duration $SESSION_REVIEW_SECS) | Total: $(fmt_duration $((SESSION_AGENT_SECS + SESSION_REVIEW_SECS)))\033[0m"

  # Clean up status file
  rm -f "$STATUS_FILE"
}

main "$@"
