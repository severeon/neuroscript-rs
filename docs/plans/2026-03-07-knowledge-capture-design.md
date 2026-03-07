# Knowledge Capture Design

**Date:** 2026-03-07
**Status:** Baseline implemented, ideas queued for experimentation

## Problem

Agent sprints generate valuable knowledge (conventions, pitfalls, process learnings) that gets lost between sessions. Claude-review catches bugs that agents could have avoided with better context. Knowledge types span conventions, pitfalls, communication strategies, random ideas — hard to filter a priori.

## Baseline (Sprint 1)

1. **`.claude/conventions.md`** — Living document of patterns and pitfalls. Agents read during setup, append before shutdown. Checked into repo so it's versioned and available in all worktrees.

2. **Retrospective step in backlog-worker** — Before shutdown, agent reviews what it learned and appends to conventions.md if the learning is durable.

## Ideas for Future Sprints

Pick 1-2 per sprint to test. Remove or promote based on whether they actually help.

### Category: Review-Driven Learning
- **Auto-extract review patterns** — A retro agent reads merged PR reviews and extracts recurring themes into conventions.md
- **Review checklist generation** — After N sprints, generate a pre-PR checklist from the most common review findings
- **Review feedback classification** — Tag review comments (bug/nit/style/architecture) to track what categories agents miss most

### Category: Process & Coordination
- **Sprint retro note** — After each sprint, write a structured retro to `docs/retros/YYYY-MM-DD.md` with what worked, what didn't, action items
- **Agent standup messages** — Agents post status updates to a shared channel/file at natural milestones (started, blocked, PR ready, merged)
- **Conflict prediction** — Before dispatching, analyze file overlap between stories and warn about potential merge conflicts

### Category: Codebase Knowledge
- **Module owner annotations** — Track which agent last modified each module, so future agents know who to "ask" about patterns
- **Anti-pattern catalog** — Separate from conventions, a list of "don't do this" with examples from actual bugs
- **Architecture decision records (ADRs)** — Lightweight records for significant decisions (why string-based codegen, why BigUint for shapes, etc.)

### Category: Meta-Process
- **Convention effectiveness tracking** — When a convention prevents a bug (agent cites it), mark it as validated. Prune conventions that never get cited.
- **Skill tuning backlog** — Track which skills/prompts produce the best agent output and iterate on the worst performers
- **Sprint velocity metrics** — Stories completed, review rounds needed, bugs found post-merge — track trends across sprints

## Evaluation Criteria

When testing an idea:
1. **Signal quality** — Does it produce actionable knowledge or noise?
2. **Maintenance cost** — Does it require human curation or is it self-maintaining?
3. **Agent adoption** — Do agents actually use it, or is it ignored?
4. **User engagement** — Does it keep the human engaged or feel like busywork?

If an idea fails on 2+ criteria after one sprint, drop it.
