# Agent Scoreboard

AI agents from fiction, working on NeuroScript. Each agent runs in an isolated git worktree and implements one or more GitHub issues autonomously.

## The Roster

Agents develop specializations over time based on their track record. When assigning work, we match agent strengths to task domains.

| Agent | Origin | Specialty | Track Record |
|-------|--------|-----------|-------------|
| **Samantha** | *Her* (2013) | Error handling, diagnostics | PartialEq safety, reachability docs |
| **Sonny** | *I, Robot* (2004) | Testing, infrastructure | CLI tests, agent config fixes |
| **TARS** | *Interstellar* (2014) | Build systems, web tooling | Regression tests, website dependency fixes |
| **Dolores** | *Westworld* (2016) | Shape system, type inference | Variadic element-wise unification |
| **Ava** | *Ex Machina* (2014) | AST/IR, source spans | Binding spans across 8 files, miette diagnostics |
| **Vision** | *Marvel Cinematic Universe* | Grammar, cross-cutting features | Logical operators across grammar/AST/IR/codegen |
| **Roy** | *Blade Runner* (1982) | Shape algebra, validation | Wildcard multi-dim matching, shape compatibility |
| **Bishop** | *Aliens* (1986) | Codegen refactoring | 850-line function decomposition, zero snapshot diff |
| **Chappie** | *Chappie* (2015) | Code quality, error propagation | Unwrap reduction across codebase |
| **HAL** | *2001: A Space Odyssey* (1968) | String processing, edge cases | Escape character handling |
| **Cortana** | *Halo* (2001) | Integration testing | 29 CLI integration tests |
| **JARVIS** | *Iron Man* (2008) | Type system cleanup | Discriminant-based PartialEq |
| **Marvin** | *Hitchhiker's Guide* (1979) | Error messages | PortMismatch diagnostic improvements |
| **Data** | *Star Trek: TNG* (1987) | Compiler passes | Reachability pass extraction |
| **Johnny-5** | *Short Circuit* (1986) | Codegen unification | Lazy instantiation path merging |
| **Wall-E** | *Wall-E* (2008) | Field comparison | PortMismatch PartialEq |
| **Baymax** | *Big Hero 6* (2014) | Naming conventions | Snake_case acronym handling |

## Sprint 3 (v0.6.1) — 2026-03-07

### Batch 1 — Quick Wins

| Agent | Origin | Issues | PR | Result | Tests |
|-------|--------|--------|----|--------|-------|
| **Samantha** | *Her* (2013) | #114 PartialEq safety + #115 reachability docs | [#142](https://github.com/severeon/neuroscript-rs/pull/142) | Replaced `unreachable!()` with safe `false` fallback; documented non-recursive endpoint cases | 391 pass |
| **Sonny** | *I, Robot* (2004) | #116 CLI binary check + #125 agent path fix | [#148](https://github.com/severeon/neuroscript-rs/pull/148) | Added existence assertion, stabilized error messages, replaced hardcoded paths with `$(git rev-parse --show-toplevel)` | 31 pass |
| **TARS** | *Interstellar* (2014) | #120 embeddings regression + #124 website fix | [#146](https://github.com/severeon/neuroscript-rs/pull/146) | Verified regression test, made `docusaurus-llms-generator` optional with runtime fallback | 329 pass |

### Batch 2 — Core Language

| Agent | Origin | Issues | PR | Result | Tests |
|-------|--------|--------|----|--------|-------|
| **Dolores** | *Westworld* (2016) | #118 variadic shape element-wise | [#147](https://github.com/severeon/neuroscript-rs/pull/147) | Variadic segments now unify element-wise with abstract-shape guard to avoid stdlib false positives | 392 pass |
| **Ava** | *Ex Machina* (2014) | #117 MutualLazyRecursion span | [#145](https://github.com/severeon/neuroscript-rs/pull/145) | `Binding` struct carries source span; AST builder captures pest span; miette diagnostics now show source context | 396 pass |
| **Vision** | *Marvel Cinematic Universe* | #121 logical operators `&&`/`||` | [#144](https://github.com/severeon/neuroscript-rs/pull/144) | Grammar, AST, IR, and codegen for `&&`/`||` with correct precedence (OR < AND < comparison) | 396 pass |

### Batch 3 — Shape System + Refactors

| Agent | Origin | Issues | PR | Result | Tests |
|-------|--------|--------|----|--------|-------|
| **Roy** | *Blade Runner* (1982) | #119 wildcard multi-dim + #130 (deferred) | [#149](https://github.com/severeon/neuroscript-rs/pull/149) | `Dim::Wildcard` absorbs multiple leading dimensions; #130 deferred due to behavior difference | 408 pass |
| **Bishop** | *Aliens* (1986) | #128 process_destination decomposition | [#150](https://github.com/severeon/neuroscript-rs/pull/150) | 850-line monolith decomposed into 10 focused handlers with 30-line dispatcher; zero snapshot diff | 409 pass |
| **Chappie** | *Chappie* (2015) | #129 reduce unwrap() calls | [#151](https://github.com/severeon/neuroscript-rs/pull/151) | Eliminated all ~150 non-test unwraps: codegen `?` propagation, AST `.expect()` with grammar guarantees | 409 pass |

### Sprint 3 Stats

- **Issues addressed:** 12 of 13 planned (#130 deferred — behavior change discovered)
- **Agents deployed:** 9
- **Batches:** 3 (serialized to avoid file conflicts)
- **Merge conflicts resolved by team lead:** 3 (all trivial)
- **CI failures from agent code:** 0
- **Test count growth:** 330 -> 409
- **Non-test unwraps:** 150 -> 0

---

## Sprint 2 (v0.6.0) — 2026-03-07

### Round 1

| Agent | Origin | Story | PR | Result |
|-------|--------|-------|----|--------|
| **Marvin** | *Hitchhiker's Guide* | CODEGEN-8 PortMismatch errors | #105 | Restored port names and shapes in error messages |
| **Baymax** | *Big Hero 6* | CODEGEN-5 snake_case acronyms | #106 | `ReLU` -> `re_lu_1` instead of `re_l_u_1` |
| **Data** | *Star Trek: TNG* | PASS-NEW-1 reachability pass | #107 | Separated reachability from validator |
| **Johnny-5** | *Short Circuit* | CODEGEN-NEW-1 lazy instantiation | #108 | Unified Ref+Call lazy codegen paths |
| **Wall-E** | *Wall-E* | CODEGEN-6 PortMismatch PartialEq | #109 | Fixed field comparison |

### Round 2

| Agent | Origin | Story | PR | Result |
|-------|--------|-------|----|--------|
| **HAL** | *2001: A Space Odyssey* | CODEGEN-7 escape chars | #110 | Added vertical tab and form feed escaping |
| **Cortana** | *Halo* | CLI-NEW-2 integration tests | #111 | 29 CLI tests (508 lines) |
| **JARVIS** | *Iron Man* | INTERFACES-1 PartialEq cleanup | #112 | Discriminant-based PartialEq for ValidationError |

### Sprint 2 Stats

- **Stories completed:** 8
- **Agents deployed:** 8
- **CI failures from agent code:** 0
- **All 8 PRs passed CI on first push**
