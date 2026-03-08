# Sprint 4 Design — "Architecture Sprint"

**Date**: 2026-03-08
**Version**: 0.6.2 (target)
**Branch**: `_dev`
**Theme**: Pay down structural debt to unlock Phase 3 features, ship one language feature, clear PR review debt.

## Scope

15 issues across 4 batches, 11 agent slots. Builds on Sprint 3 lessons: 3 agents max per batch, no overlapping files between agents, pre-created worktrees, version bump before branching.

## Batch 1 — Quick Wins

Small, isolated changes. No file conflicts between agents.

| Agent | Issues | Description | Key Files |
|-------|--------|-------------|-----------|
| **Samantha** | #155, #159, #160 | Remove unused `_gen` param from `process_reshape_reduce`; remove dead unreachable check in same fn; add SYNC comment to `EXPECTED_VARIANT_COUNT` | `src/codegen/forward.rs`, `src/optimizer/core.rs` |
| **Sonny** | #162, #163 | `Binding::default()` factory to reduce `span: None` boilerplate; rename `call_to_result` → `endpoint_to_result` | `src/interfaces.rs`, `src/codegen/forward.rs` |
| **HAL** | #75, #165 | Implement miette `Diagnostic` for `ValidationError`; document leading-only wildcard constraint in language reference | `src/interfaces.rs`, docs |

**Conflict zone**: Samantha and Sonny both touch `codegen/forward.rs` but different functions (`process_reshape_reduce` vs `call_to_result`). Safe if changes don't overlap. HAL and Sonny both touch `interfaces.rs` but different sections (error types vs binding constructors).

## Batch 2 — Structural Refactors

The three biggest tech debt items in the backlog. Each is self-contained.

| Agent | Issues | Description | Key Files |
|-------|--------|-------------|-----------|
| **Bishop** | #127 | Decompose `contract_resolver.rs` (1,776 lines) into submodules: detection, resolution, substitution, tests | `src/passes/contract_resolver.rs` → `src/passes/contract_resolver/` |
| **Chappie** | #126 | Create `IRPass`/`EndpointVisitor` trait to unify 5 duplicate endpoint walkers (~500 LOC dedup) | New `src/visitor.rs`, touches `cycles.rs`, `utils.rs`, `contract_resolver.rs`, `desugar.rs`, `inference.rs` |
| **Roy** | #130 | Deduplicate dimension solving — `validator/shapes.rs` delegates to `shape/inference.rs` engine | `src/shape/inference.rs`, `src/validator/shapes.rs` |

**Dependency note**: Chappie (#126) touches `contract_resolver.rs` which Bishop (#127) is decomposing. These MUST run in different batches or Bishop must land first. Since they're in the same batch, Bishop decomposes the file first, then Chappie's PR targets the new submodule structure. **Alternative**: serialize Bishop → Chappie if conflicts arise.

**Risk mitigation**: If Bishop and Chappie conflict, merge Bishop first, then rebase Chappie. The visitor trait (#126) can target the decomposed submodules.

## Batch 3 — Features + Codegen

One language feature, one validator enhancement, two codegen fixes.

| Agent | Issues | Description | Key Files |
|-------|--------|-------------|-----------|
| **Vision** | #158 | Support implicit fork inside match arms (grammar + AST + codegen) | `src/grammar/neuroscript.pest`, `src/grammar/ast.rs`, `src/codegen/forward.rs` |
| **Dolores** | #153, #154 | Fix `process_match` silent `None` propagation; confirm if/else snapshot coverage in `process_branch_pipeline` | `src/codegen/forward.rs`, `src/codegen/tests.rs` |
| **Cortana** | #131 | Cross-neuron mutual `@lazy` recursion detection via whole-program call graph | `src/validator/bindings.rs`, `src/validator/tests/` |

**Dependency**: Dolores (#153) touches `codegen/forward.rs` which Sonny (#163) modified in Batch 1. Must land after Batch 1.

## Batch 4 — PR Review Debt

Clear accumulated review items from two major PRs.

| Agent | Issues | Description | Key Files |
|-------|--------|-------------|-----------|
| **Ava** | #122 | Address 19 unresolved review items from fat arrow PR #34 | `src/codegen/forward.rs`, `src/grammar/ast.rs` |
| **JARVIS** | #123 | Address 13 unresolved review items from wrap/hyper PR #39 | `src/passes/desugar.rs`, `src/interfaces.rs` |

**Reference**: Both agents need `docs/reviews/pr-comment-audit-2026-03-05.md` for the full item lists.

## Execution Protocol

1. **Version bump** to 0.6.2 on `_dev` before any branching
2. **Pre-create worktrees** for each batch (3 at a time)
3. **Each agent**: worktree → branch → implement → PR → CI green → review
4. **Merge order**: Batch 1 → Batch 2 → Batch 3 → Batch 4
5. **Between batches**: merge completed PRs to `_dev`, resolve any conflicts, verify CI
6. **Release**: merge `_dev` → `main` via release PR after all 4 batches

## Agent Persona Assignments

Per established specializations from `docs/AGENT-SCOREBOARD.md`:

- **Samantha**: Error handling, diagnostics, safety → codegen cleanup
- **Sonny**: Testing, infrastructure → interfaces refactoring
- **HAL**: String processing, edge cases → diagnostics + docs
- **Bishop**: Codegen refactoring, large decompositions → contract_resolver split
- **Chappie**: Code quality, error propagation → visitor trait abstraction
- **Roy**: Shape algebra, validation → dimension dedup
- **Vision**: Grammar, cross-cutting features → implicit fork in match arms
- **Dolores**: Shape system, type inference → process_match fixes
- **Cortana**: Integration testing → cross-neuron recursion detection
- **Ava**: AST/IR, source spans → fat arrow review debt
- **JARVIS**: Type system cleanup → wrap/hyper review debt

## Success Criteria

- All 15 issues closed with merged PRs
- `cargo test` passes after each batch merge
- `./test_examples.sh` passes after each batch merge
- No new snapshot failures (or all reviewed and accepted)
- `contract_resolver.rs` no longer a monolith (< 500 lines per submodule)
- Endpoint walker duplication reduced from 5 to 1-2 locations
- Zero remaining PR review debt from PRs #34 and #39

## What's Next (Sprint 5 candidates)

- #132: Full cross-neuron dimension type inference (needs design doc)
- #133: Loop/repeat construct (needs language design)
- #134: Backend trait for multi-target codegen (Phase 4)
- #135: LSP server (Phase 4)
- #131 cross-neuron lazy recursion unblocked by #126
