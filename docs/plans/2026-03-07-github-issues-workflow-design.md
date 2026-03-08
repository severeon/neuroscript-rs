# GitHub Issues-Centric Sprint Workflow

**Date:** 2026-03-07
**Status:** Approved

## Overview

Replace the Obsidian kanban-centric workflow with a GitHub Issues-centric process that is public, auditable, and optimized for AI-assisted development. The workflow has four phases: autonomous planning, collaborative approval, parallel agent execution, and human-gated merge.

## Phase 1: Plan (Autonomous)

A **planner agent** runs autonomously over a batch of open issues:

1. Reads issue description, labels, and any existing comments
2. Reads relevant source files referenced in the issue
3. Posts an implementation plan as an issue comment, covering:
   - Approach (what to change and why)
   - Files to modify
   - Test strategy
   - Risk/complexity estimate
   - Conflict notes (files shared with other issues)
4. For issues with testable behavior: creates a feature branch, writes failing acceptance tests, pushes a **draft PR** with the plan as the PR description
5. For plan-only issues (docs, config, refactors without clear test targets): posts the plan as a comment only

The planner runs through all selected issues in one session without human interaction.

## Phase 2: Approve & Schedule (Collaborative)

Human and orchestrator review together in a live session:

1. Review draft PRs and plan comments
2. Discuss priorities, adjust plans, resolve conflicts between issues touching the same files
3. Select the batch for execution (3-5 issues, based on Sprint 2 learnings)
4. Pre-create worktrees and prepare agent assignments
5. Human gives the go-ahead

## Phase 3: Execute (Agent Team)

Dispatched as a team of 3 parallel agents (proven sweet spot from Sprint 2):

1. Each agent picks up an assigned issue with an existing branch and failing tests
2. Implements the fix/feature to make tests pass
3. Pushes commits to the existing draft PR branch
4. Posts implementation summary as a PR comment
5. Waits for CI — addresses review feedback autonomously (formatting, naming, missing tests)
6. Escalates design questions to the issue thread
7. Agent shuts down when CI passes

## Phase 4: Review & Merge (Human)

1. Human reviews the draft PRs (now with implementation)
2. Converts draft to ready-for-review (human-only action)
3. Merges when satisfied
4. Repeats with next batch

## Guard Rails

| Rule | Enforcement |
|------|------------|
| Draft PRs can't be merged | GitHub branch protection (require review) |
| Agents can't promote draft to ready | Convention in agent def + GitHub Action guard |
| Agents can't merge | Convention in agent def (no `gh pr merge`) |
| All discussion on issue threads | Agent def: post progress/questions as issue comments |
| No Obsidian dependency | Agent def removes all `mcp__obsidian__*` tools |
| Worktree isolation | Pre-created worktrees, explicit paths |

## Agent Definitions

### planner.md

Reads issues, posts plans, creates draft PRs with acceptance tests. No implementation code.

**Tools:** `Read`, `Bash`, `Glob`, `Grep`, `Write` (for test files only)

**Inputs:**
- List of issue numbers to plan
- Base branch (usually `_dev`)

**Outputs per issue:**
- Implementation plan posted as issue comment
- Feature branch with failing acceptance tests (if applicable)
- Draft PR with plan as description (if applicable)

**Hard rules:**
- No implementation code — tests only
- No PR merge, no draft-to-ready conversion
- Post all discussion on the issue thread

### implementer.md

Picks up an issue with existing branch and tests, implements the solution, pushes to the draft PR.

**Tools:** `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`

**Inputs:**
- Issue number with existing branch and draft PR
- Worktree path (pre-created by orchestrator)
- Conflict warnings (files other agents are touching)

**Outputs:**
- Implementation commits pushed to draft PR branch
- Implementation summary posted as PR comment
- CI passing

**Hard rules:**
- No PR merge, no draft-to-ready conversion
- Stay in scope — only fix what the issue describes
- Post design questions on the issue thread
- Never work outside assigned worktree

## What Gets Retired

- Obsidian kanban board as source of truth (replaced by GitHub Issues)
- Story notes in Obsidian (issue descriptions serve this role)
- `backlog-worker.md` agent (split into planner + implementer)
- Manual story-to-agent assignment (planner handles this)
- `mcp__obsidian__*` tool dependencies in agent definitions

## GitHub Action: Draft PR Guard

A lightweight workflow that fires on `pull_request.ready_for_review` and auto-reverts to draft if the actor is not in an allowlist of human maintainers. Prevents automated promotion of draft PRs.

```yaml
name: Draft PR Guard
on:
  pull_request:
    types: [ready_for_review]

jobs:
  check-actor:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - name: Revert to draft if not human
        if: github.actor != 'severeon'
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
          REPO: ${{ github.repository }}
        run: |
          gh pr ready --undo "${PR_NUMBER}" -R "${REPO}" \
            || { echo "::error::Failed to revert PR to draft!"; exit 1; }
```
