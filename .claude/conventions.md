# Conventions & Patterns

Living document of lessons learned across sprints. Agents should read this during setup and append learnings before shutdown.

## Rust / Cargo

- **build.rs `rerun-if-changed`**: Emit per-file, not per-directory. `cargo:rerun-if-changed=stdlib` only watches the top-level dir — subdirectory changes are missed. Always emit for each discovered file.
- **`expect()` over `unwrap()` in build scripts**: Build script panics produce opaque errors. Use `expect("context")` so contributors can diagnose failures.

## Error Handling

- **Guard clauses that gate on prior state can silently drop items** when iteration order isn't guaranteed. If you accumulate results in a loop, don't condition early items on flags set by later items. Collect first, then process.
- **Validation collects ALL errors** rather than failing fast. New validation checks should push to the `errors` vector, not return early.
- **Don't mix stderr side-effects with error return values.** If you print errors to stderr in a loop AND embed them in a returned error, users see double output.

## Codegen

- **Codegen is string-based**, not AST-based. Generated Python is emitted as format strings.
- **Always sanitize user-provided identifiers** before emitting them in generated code (`sanitize_python_ident`).
- **Escape string content** in generated Python — backslashes, quotes, newlines, tabs, null bytes.

## CI / GitHub Actions

- **`pull_request` trigger evaluates from the base branch config**, not the PR head. Adding a branch to the trigger in a PR targeting that branch won't take effect until after merge.
- **When removing CI gate jobs**, update branch protection rules to reference the individual checks directly. Otherwise the branch is silently unprotected.

## Agent Coordination

- **Worktrees must be isolated per agent.** Shared worktrees cause branch contamination and leaked changes in PRs.
- **Always fix release bugs in separate PRs**, not in the release PR itself. Add non-trivial nits to the board as stories.
- **Dogfood new tooling early.** If we build a script or workflow, use it immediately in the same sprint.

## NeuroScript Language

- **Comments use `#` (inline) and `///` (doc).** Never `//`.
- **`impl:` field uses comma notation**: `core,attention/ScaledDotProductAttention`, not dot notation.
