# Sprint 4 Implementation Plan — "Architecture Sprint"

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close 15 issues across tech debt, structural refactors, a language feature, and PR review debt — unlocking Phase 3 language features.

**Architecture:** 4 sequential batches of 2-3 parallel agents, each in pre-created worktrees. Batch N merges before Batch N+1 branches. Version bump to 0.6.2 before any branching.

**Tech Stack:** Rust (pest, miette, thiserror), PyTorch codegen, cargo test, insta snapshots

---

## Pre-Sprint Setup

### Step 0: Version bump and worktree prep

**Step 0.1: Bump version on `_dev`**

```bash
# On _dev branch, update Cargo.toml
sed -i '' 's/^version = "0.6.1"/version = "0.6.2"/' Cargo.toml
cargo check  # verify
git add Cargo.toml Cargo.lock
git commit -m "chore: bump version to 0.6.2 for Sprint 4"
```

**Step 0.2: Create worktrees for Batch 1**

```bash
# From repo root, create worktrees for each agent
git worktree add ../neuroscript-rs-samantha _dev -b sprint4/samantha-155-159-160
git worktree add ../neuroscript-rs-sonny _dev -b sprint4/sonny-162-163
git worktree add ../neuroscript-rs-hal _dev -b sprint4/hal-75-165
```

Repeat pattern for each batch after the previous one merges.

---

## Batch 1 — Quick Wins

### Task 1: Samantha — #155, #159, #160

**Issues:**
- #155: Remove unused `_gen` parameter from `process_reshape_reduce`
- #159: Dead unreachable check in `process_reshape_reduce`
- #160: Add SYNC comment to `EXPECTED_VARIANT_COUNT`

**Files:**
- Modify: `src/codegen/forward.rs:1129-1138` (process_reshape_reduce signature)
- Modify: `src/interfaces.rs:1467-1475` (EXPECTED_VARIANT_COUNT)

**Step 1.1: Write test confirming current behavior**

Run: `cargo test codegen -- --nocapture`
Expected: All pass (baseline)

**Step 1.2: Remove `_gen` from `process_reshape_reduce`**

In `src/codegen/forward.rs`:

1. Line 1131: Remove `_gen: &CodeGenerator,` parameter
2. Line 1089: Update call site to remove `gen` argument:
   ```rust
   // Before:
   process_reshape_reduce(output, gen, reshape, &result_var, &source_var, indent, strategy, reshape_source_shape)?;
   // After:
   process_reshape_reduce(output, reshape, &result_var, &source_var, indent, strategy, reshape_source_shape)?;
   ```

**Step 1.3: Address dead unreachable in process_reshape_reduce**

Review the function body (lines 1139-1178). The `TransformStrategy::Intrinsic` match arm at line 1148 has a catch-all `_ =>` that returns an error — this is NOT unreachable (it handles unknown intrinsic names).

Check if there's another unreachable pattern. Look for any `unreachable!()` macro calls or dead match arms in the function. The issue says "dead unreachable check" — examine whether a match arm can never be reached due to prior validation.

**Step 1.4: Add SYNC comment to EXPECTED_VARIANT_COUNT**

In `src/interfaces.rs` around line 1469:

```rust
// SYNC: This count must match the number of variants in `ValidationError` enum (line ~696).
// When adding a new variant, update this constant AND the `PartialEq` impl (line ~807).
const EXPECTED_VARIANT_COUNT: usize = 16;
```

**Step 1.5: Run tests and commit**

```bash
cargo test
cargo test --test integration_tests
git add src/codegen/forward.rs src/interfaces.rs
git commit -m "fix: remove unused _gen param, add SYNC comment (#155, #159, #160)"
```

---

### Task 2: Sonny — #162, #163

**Issues:**
- #162: `Binding::default()` factory to reduce `span: None` boilerplate
- #163: Rename `call_to_result` → `endpoint_to_result`

**Files:**
- Modify: `src/interfaces.rs:460-471` (Binding struct — add factory method)
- Modify: `src/codegen/forward.rs` (rename `call_to_result` throughout — ~25 occurrences)

**Step 2.1: Add `Binding::new()` factory**

In `src/interfaces.rs`, after the `Binding` struct definition (line ~471), add:

```rust
impl Binding {
    /// Create a new binding with common defaults (span: None, not frozen, no unroll group).
    pub fn new(name: impl Into<String>, call_name: impl Into<String>, args: Vec<Value>) -> Self {
        Self {
            name: name.into(),
            call_name: call_name.into(),
            args,
            kwargs: Vec::new(),
            scope: Scope::Instance,
            frozen: false,
            unroll_group: None,
            span: None,
        }
    }
}
```

**Step 2.2: Replace `span: None` boilerplate with `Binding::new()`**

Search all 81 `span: None` occurrences related to `Binding` construction across:
- `src/passes/desugar.rs` (5 occurrences)
- `src/passes/unroll.rs` (12 occurrences)
- `src/passes/contract_resolver.rs` (6 occurrences)
- `src/codegen/tests.rs` (3 occurrences)
- `src/validator/tests/` (various)

Only replace where the binding uses default values for kwargs, scope, frozen, and unroll_group. Leave complex constructions (non-default scope, frozen=true, etc.) as-is.

**Step 2.3: Rename `call_to_result` → `endpoint_to_result`**

In `src/codegen/forward.rs`, do a global rename. The variable appears:
- Line 206: `let mut call_to_result: HashMap<String, String>` → `let mut endpoint_to_result`
- Lines 257, 259, 286, 386, 398, 401, 436, 459, 464, 469, 783, 892, 901, 904, 945, 968, 980, 1005, 1029, 1044, 1070, 1098: All references

Also update comments referencing "call_to_result" (lines 205, 398, 901).

**Step 2.4: Run tests and commit**

```bash
cargo test
cargo test --test integration_tests
git add src/interfaces.rs src/codegen/forward.rs src/passes/ src/validator/
git commit -m "refactor: add Binding::new() factory, rename call_to_result (#162, #163)"
```

---

### Task 3: HAL — #75, #165

**Issues:**
- #75: Add miette `Diagnostic` impl for `ValidationError`
- #165: Document leading-only wildcard constraint

**Files:**
- Modify: `src/interfaces.rs:696-802` (ValidationError enum)
- Create: `docs/language/wildcard-constraints.md` (or add to existing reference)

**Step 3.1: Check current Diagnostic derive status**

`ValidationError` already has `#[derive(Diagnostic)]` from miette (verify by checking derives on the enum). The issue may be about adding `#[diagnostic(code(...))]` attributes for machine-readable error codes.

Look at `src/interfaces.rs:696` — the enum derives `Error` from thiserror but check if `Diagnostic` is also derived. If not, add it:

```rust
#[derive(Debug, Clone, Error, Diagnostic)]
pub enum ValidationError {
    #[error("Neuron '{name}' not found (in {context})")]
    #[diagnostic(code(neuroscript::missing_neuron))]
    MissingNeuron { ... },
    // ... add #[diagnostic(code(...))] to each variant
}
```

Each variant should get a diagnostic code following the pattern `neuroscript::<snake_case_variant>`.

**Step 3.2: Write test for Diagnostic impl**

```rust
#[test]
fn validation_error_implements_diagnostic() {
    use miette::Diagnostic;
    let err = ValidationError::MissingNeuron {
        name: "Foo".into(),
        context: "test".into(),
        span: None,
    };
    // Verify code is present
    assert!(err.code().is_some());
}
```

**Step 3.3: Document wildcard constraint**

The constraint: wildcards (`*`) only match **leading** dimensions. `[*, dim]` matches `[batch, seq, dim]` but `[dim, *]` is NOT supported. This is enforced in `src/validator/shapes.rs:138` (`shorter.dims.first() != Some(&Dim::Wildcard)`).

Document this in the language reference. Check if `docs/language/` exists; if not, add the note to CLAUDE.md's "Key Language Concepts > Shapes" section, or create a dedicated doc.

**Step 3.4: Run tests and commit**

```bash
cargo test
git add src/interfaces.rs docs/
git commit -m "feat: add Diagnostic codes to ValidationError, document wildcard constraint (#75, #165)"
```

---

### Batch 1 Merge

```bash
# After all 3 PRs pass CI and review:
# Merge each PR to _dev
# Verify clean build on _dev
cargo test && ./test_examples.sh
```

---

## Batch 2 — Structural Refactors

### Task 4: Bishop — #127

**Issue:** Decompose `contract_resolver.rs` (1,776 lines) into submodules

**Files:**
- Modify: `src/passes/contract_resolver.rs` → split into `src/passes/contract_resolver/mod.rs` + submodules

**Current function map (16 functions):**

| Function | Lines | Category |
|----------|-------|----------|
| `resolve_neuron_contracts` | 50-97 | Entry point |
| `collect_unresolved_contracts` | 99-145 | Detection |
| `has_named_match` | 147-159 | Detection |
| `endpoint_has_named_match` | 161-201 | Detection |
| `resolve_contracts_for_neuron` | 203-314 | Resolution |
| `build_default_bindings` | 316-325 | Resolution helpers |
| `collect_named_match_params` | 327-338 | Detection |
| `collect_named_params_from_endpoint` | 340-376 | Detection |
| `find_call_sites` | 378-437 | Call site discovery |
| `find_call_sites_in_endpoint` | 439-521 | Call site discovery |
| `resolve_match_in_neuron` | 523-617 | Resolution |
| `resolve_match_in_endpoint` | 619-731 | Resolution |
| `find_matching_arm` | 733-756 | Matching |
| `contract_matches` | 758-773 | Matching |
| `ports_match` | 775-824 | Matching |
| `shape_pattern_matches` | 826-EOF | Matching |

**Proposed decomposition:**

```
src/passes/contract_resolver/
├── mod.rs          # Re-exports, resolve_neuron_contracts entry point (~100 lines)
├── detection.rs    # has_named_match, endpoint_has_named_match, collect_named_*, collect_unresolved_contracts (~250 lines)
├── call_sites.rs   # find_call_sites, find_call_sites_in_endpoint (~160 lines)
├── resolution.rs   # resolve_contracts_for_neuron, resolve_match_in_*, build_default_bindings (~420 lines)
├── matching.rs     # find_matching_arm, contract_matches, ports_match, shape_pattern_matches (~100 lines)
└── tests.rs        # Move existing tests (if inline)
```

**Step 4.1: Create module directory**

```bash
mkdir -p src/passes/contract_resolver
```

**Step 4.2: Create `mod.rs` with entry point**

Move `resolve_neuron_contracts` and the module doc comment. Re-export submodule items needed externally (only `resolve_neuron_contracts` is `pub`).

**Step 4.3: Create `detection.rs`**

Move: `collect_unresolved_contracts`, `has_named_match`, `endpoint_has_named_match`, `collect_named_match_params`, `collect_named_params_from_endpoint`

These are all `fn` (not `pub fn`) — make them `pub(super)` for access from `mod.rs` and `resolution.rs`.

**Step 4.4: Create `call_sites.rs`**

Move: `find_call_sites`, `find_call_sites_in_endpoint`

**Step 4.5: Create `resolution.rs`**

Move: `resolve_contracts_for_neuron`, `resolve_match_in_neuron`, `resolve_match_in_endpoint`, `build_default_bindings`

This module needs to call functions from `detection.rs`, `call_sites.rs`, and `matching.rs`.

**Step 4.6: Create `matching.rs`**

Move: `find_matching_arm`, `contract_matches`, `ports_match`, `shape_pattern_matches`

**Step 4.7: Move tests**

If there are `#[cfg(test)]` blocks, move to `tests.rs` within the new module.

**Step 4.8: Verify**

```bash
cargo test
cargo test --test integration_tests
# Verify no submodule exceeds ~500 lines
wc -l src/passes/contract_resolver/*.rs
git add src/passes/
git commit -m "refactor: decompose contract_resolver.rs into submodules (#127)"
```

---

### Task 5: Chappie — #126

**Issue:** Create `EndpointVisitor` trait to unify 5 duplicate endpoint walkers (~500 LOC)

**Files:**
- Create: `src/visitor.rs` (new module)
- Modify: `src/lib.rs` (add `mod visitor`)
- Modify: `src/validator/cycles.rs:74-291` (3 extract functions)
- Modify: `src/codegen/utils.rs:230-310` (endpoint_key_impl, collect_calls_impl)
- Modify: `src/codegen/generator.rs:414-450` (collect_calls_from_endpoint)
- Potentially modify: `src/passes/contract_resolver/detection.rs` (after Bishop's decomposition)
- Potentially modify: `src/passes/desugar.rs:83+` (desugar_endpoint_wraps)

**Step 5.1: Design the trait**

```rust
/// Trait for visiting all endpoints in a recursive endpoint tree.
/// Implementors handle specific endpoint types; the default walk
/// implementation handles recursion into nested endpoints.
pub trait EndpointVisitor {
    type Output;

    fn visit_ref(&mut self, port_ref: &PortRef) -> Self::Output;
    fn visit_call(&mut self, name: &str, args: &[Value], kwargs: &[Kwarg], endpoint: &Endpoint) -> Self::Output;
    fn visit_tuple(&mut self, endpoints: &[Endpoint]) -> Self::Output;
    fn visit_match(&mut self, match_expr: &MatchExpr) -> Self::Output;
    fn visit_reshape(&mut self, reshape: &ReshapeExpr) -> Self::Output;
    fn visit_if(&mut self, if_expr: &IfExpr) -> Self::Output;
    fn visit_wrap(&mut self, wrap: &WrapExpr) -> Self::Output;

    /// Walk an endpoint, dispatching to the appropriate visit method.
    fn walk(&mut self, endpoint: &Endpoint) {
        match endpoint {
            Endpoint::Ref(r) => { self.visit_ref(r); }
            Endpoint::Call { name, args, kwargs, .. } => { self.visit_call(name, args, kwargs, endpoint); }
            Endpoint::Tuple(eps) => { self.visit_tuple(eps); }
            Endpoint::Match(m) => { self.visit_match(m); }
            Endpoint::Reshape(r) => { self.visit_reshape(r); }
            Endpoint::If(i) => { self.visit_if(i); }
            Endpoint::Wrap(w) => { self.visit_wrap(w); }
        }
    }
}
```

**Step 5.2: Implement for cycles.rs extract functions**

Refactor `extract_node_names_from_sources`, `extract_node_names_from_destinations`, and `extract_simple_node_names` to use the visitor. These collect `Vec<String>` from endpoint trees.

**Step 5.3: Implement for codegen collect_calls**

Refactor `collect_calls_from_endpoint` (generator.rs) and `collect_calls_impl` (utils.rs) to use the visitor.

**Step 5.4: Assess remaining walkers**

`desugar_endpoint_wraps` and `collect_unresolved_contracts` are **mutable** walkers — they modify endpoints. The visitor trait may need a separate `EndpointVisitorMut` variant. Start with read-only visitor for the 3 simplest cases, document the mutable ones as future work.

**Step 5.5: Run tests and commit**

```bash
cargo test
cargo test --test integration_tests
git add src/visitor.rs src/lib.rs src/validator/cycles.rs src/codegen/
git commit -m "refactor: introduce EndpointVisitor trait, unify 3 walkers (#126)"
```

---

### Task 6: Roy — #130

**Issue:** Deduplicate dimension solving between `shape/` and `validator/`

**Files:**
- Modify: `src/validator/shapes.rs` (215 lines of compatibility logic)
- Modify: `src/shape/inference.rs:1226` (`shapes_compatible` method)

**Analysis:**

`validator/shapes.rs` has its own `shapes_compatible()` (line 14) with local `dim_bindings`. The `InferenceContext` in `shape/inference.rs` has a more sophisticated `shapes_compatible()` (line 1226) that creates a temporary context.

The validator function is simpler (no expression solving, conservative on expressions) but handles variadics and wildcard multi-dim. The inference engine handles expressions but delegates variadics differently.

**Step 6.1: Compare behavior**

Write a test that exercises both paths with the same inputs and documents where they diverge:

```rust
#[test]
fn both_compatibility_functions_agree() {
    // Cases where both should agree
    let cases = vec![
        // [*, dim] vs [batch, seq, 512] — both should say compatible
        (Shape { dims: vec![Dim::Wildcard, Dim::Named("dim".into())] },
         Shape { dims: vec![Dim::Named("batch".into()), Dim::Named("seq".into()), Dim::Literal(512)] }),
        // ... more cases
    ];
}
```

**Step 6.2: Make validator delegate to inference engine**

Replace `shapes_compatible` in `validator/shapes.rs` with a thin wrapper:

```rust
pub(crate) fn shapes_compatible(source: &Shape, dest: &Shape) -> bool {
    let ctx = InferenceContext::new();
    ctx.shapes_compatible(source, dest)
}
```

This requires `InferenceContext::shapes_compatible` to handle all cases that `validator/shapes.rs` currently handles (variadics, wildcard multi-dim). Verify this by checking that `InferenceContext::shapes_compatible` at line 1226 handles:
- Variadic matching ✓ (via `unify_shapes_with_variadic`)
- Wildcard multi-dim ✓ (check — may need to port `wildcard_multi_dim_compatible`)
- Named dimension binding tracking ✓ (via temporary context)

If `InferenceContext` is missing wildcard multi-dim handling, port that logic into it first.

**Step 6.3: Keep substitute_* and pattern functions**

`substitute_params`, `substitute_shape`, `substitute_dim`, `is_catch_all_pattern`, `pattern_subsumes`, etc. stay in `validator/shapes.rs` — they're not duplicated.

**Step 6.4: Run tests and commit**

```bash
cargo test shape
cargo test validator
cargo test --test integration_tests
git add src/validator/shapes.rs src/shape/inference.rs
git commit -m "refactor: deduplicate dimension solving, validator delegates to inference (#130)"
```

---

### Batch 2 Merge

```bash
# Merge Bishop first (file structure change), then Chappie (visitor), then Roy
# After all 3:
cargo test && ./test_examples.sh
```

---

## Batch 3 — Features + Codegen

### Task 7: Vision — #158

**Issue:** Implicit fork inside match arms not supported

**Files:**
- Modify: `src/grammar/neuroscript.pest` (match arm pipeline rule)
- Modify: `src/grammar/ast.rs` (parse multi-connection pipelines in arms)
- Modify: `src/codegen/forward.rs` (generate fork code inside match arms)
- Create: `examples/match_implicit_fork.ns` (test case)

**Step 7.1: Understand the limitation**

Currently, match arm pipelines only support linear chains: `in -> A() -> B() -> out`. They don't support fork syntax like `in -> (main, skip)` followed by separate connections.

The desired syntax:
```neuroscript
in -> match: ->
    [*, d] where d == target:
        in -> (main, skip)
        main -> Linear(d, d) -> processed
        (processed, skip) -> Add() -> out
```

**Step 7.2: Check grammar rule for match arm body**

Look at `match_arm` and `pipeline` rules in `neuroscript.pest`. The arm body likely only accepts a single pipeline, not multiple connections. The grammar needs to accept a block of connections (like a mini graph body).

**Step 7.3: Write failing test**

Create `examples/match_implicit_fork.ns`:
```neuroscript
# Test: implicit fork inside match arm
neuron ConditionalResidual(target):
    in: [*, dim]
    out: [*, dim]

    graph:
        in -> match: ->
            [*, d] where d == target:
                in -> (main, skip)
                main -> Linear(d, d) -> processed
                (processed, skip) -> Add() -> out
            [*, d]:
                in -> Linear(d, d) -> out
```

Run: `./target/release/neuroscript parse examples/match_implicit_fork.ns`
Expected: Parse error (confirms the limitation)

**Step 7.4: Update grammar**

Modify the match arm body rule to accept either a single pipeline or a connection block. This is a grammar design decision — consult with the user if the approach is unclear.

**Step 7.5: Update AST builder**

Handle multi-connection match arm bodies in `src/grammar/ast.rs`.

**Step 7.6: Update codegen**

In `process_match` and `process_branch_pipeline`, handle the case where an arm body contains fork/multi-connection patterns.

**Step 7.7: Test**

```bash
cargo build --release
./target/release/neuroscript compile examples/match_implicit_fork.ns --verbose
cargo test
cargo test --test integration_tests
cargo insta review  # Accept new snapshots
git add src/grammar/ src/codegen/ examples/
git commit -m "feat: support implicit fork inside match arms (#158)"
```

---

### Task 8: Dolores — #153, #154

**Issues:**
- #153: `process_match` — `match_out = None` can propagate silently
- #154: Confirm if/else snapshot coverage in `process_branch_pipeline`

**Files:**
- Modify: `src/codegen/forward.rs:776-910` (process_match)
- Modify: `src/codegen/tests.rs` (add snapshot tests)

**Step 8.1: Analyze the None propagation**

In `process_match` (line 792): `match_out = None` is initialized. If no match arm runs at runtime (non-exhaustive match that passed validation), the None propagates. After the match block, there should be a runtime safety check.

**Step 8.2: Add runtime guard**

After the match arms, before returning `result_var`, add:

```python
if match_out is None:
    raise RuntimeError("No match arm matched the input shape")
```

This should be emitted by the codegen only when there's no guaranteed catch-all arm.

**Step 8.3: Write codegen test for the guard**

```rust
#[test]
fn test_match_none_guard() {
    // Compile a neuron with match arms, verify the generated code includes
    // the None guard
}
```

**Step 8.4: Add if/else snapshot tests (#154)**

Add snapshot tests for `process_branch_pipeline` covering:
- Simple if/else
- if/elif/else chain
- Nested if inside match arm

```rust
#[test]
fn test_if_else_codegen_snapshot() {
    // Create a neuron with if/else, compile, snapshot the output
}
```

Run: `cargo insta test --review`

**Step 8.5: Commit**

```bash
cargo test
cargo insta review
git add src/codegen/forward.rs src/codegen/tests.rs tests/snapshots/
git commit -m "fix: add match_out None guard, add if/else snapshot tests (#153, #154)"
```

---

### Task 9: Cortana — #131

**Issue:** Cross-neuron mutual `@lazy` recursion detection

**Files:**
- Modify: `src/validator/bindings.rs` (currently only within-neuron, line 143-228)
- Modify: `src/validator/core.rs` (add cross-neuron check call)
- Create: `src/validator/tests/cross_neuron_lazy.rs` (new test file)

**Step 9.1: Write failing test**

```rust
#[test]
fn test_cross_neuron_mutual_lazy_recursion() {
    let source = r#"
neuron A(dim):
    in: [*, dim]
    out: [*, dim]
    context:
        @lazy b = B(dim)
    graph:
        in -> b -> out

neuron B(dim):
    in: [*, dim]
    out: [*, dim]
    context:
        @lazy a = A(dim)
    graph:
        in -> a -> out
"#;
    let program = parse_program(source).unwrap();
    let errors = validate(&program);
    assert!(errors.iter().any(|e| matches!(e, ValidationError::MutualLazyRecursion { .. })));
}
```

**Step 9.2: Build cross-neuron call graph**

In `src/validator/bindings.rs`, add a new function:

```rust
/// Detect cross-neuron mutual @lazy recursion.
///
/// Builds a whole-program call graph from @lazy bindings:
/// - Node = neuron name
/// - Edge = neuron A has a @lazy binding that calls neuron B
///
/// Then runs cycle detection on this graph.
pub(super) fn detect_cross_neuron_lazy_cycles(
    program: &Program,
) -> Vec<ValidationError> {
    // 1. Build adjacency list: for each neuron, collect names of neurons
    //    referenced by @lazy bindings
    // 2. Run DFS cycle detection (reuse CycleDfs pattern from within-neuron check)
    // 3. Return MutualLazyRecursion errors for each cycle found
}
```

**Step 9.3: Call from validator**

In `src/validator/core.rs`, after per-neuron validation, call the cross-neuron check:

```rust
// After individual neuron validation loops:
errors.extend(detect_cross_neuron_lazy_cycles(program));
```

**Step 9.4: Add non-cycle test (no false positives)**

```rust
#[test]
fn test_cross_neuron_no_false_positive() {
    // A calls B (lazy), B does NOT call A — should pass
    let source = r#"
neuron A(dim):
    in: [*, dim]
    out: [*, dim]
    context:
        @lazy b = B(dim)
    graph:
        in -> b -> out

neuron B(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> Linear(dim, dim) -> out
"#;
    let program = parse_program(source).unwrap();
    let errors = validate(&program);
    assert!(errors.is_empty());
}
```

**Step 9.5: Run tests and commit**

```bash
cargo test validator
cargo test --test integration_tests
git add src/validator/
git commit -m "feat: detect cross-neuron mutual @lazy recursion (#131)"
```

---

### Batch 3 Merge

```bash
# Merge Dolores first (codegen fix), then Cortana (validator), then Vision (language feature)
cargo test && ./test_examples.sh
```

---

## Batch 4 — PR Review Debt

### Task 10: Ava — #122

**Issue:** Address 19 unresolved review items from fat arrow PR #34

**Reference:** `docs/reviews/pr-comment-audit-2026-03-05.md`, items prefixed `PR34-`

**Priority order (medium severity first):**

1. **PR34-57**: `Others` in @repeat(copy) misclassified — `src/codegen/forward.rs`
2. **PR34-58**: `eprintln!` in production codegen for @reduce fallback — `src/codegen/forward.rs`
3. **PR34-59**: Two distinct dim-to-Python conversion paths — `src/codegen/forward.rs`
4. **PR34-29**: `Others` → `Dim::Wildcard` loses rank-collapsing semantics — `src/interfaces.rs`
5. **PR34-30**: `to_shape()` drops Binding expression constraint — `src/interfaces.rs`
6. **PR34-39**: InvalidReshape/InvalidAnnotation lack miette source spans — `src/interfaces.rs`
7. **PR34-21**: No validation that @reduce target dims reachable from source — `src/validator/`
8. **PR34-51**: Rank-delta fallback generates wrong code for non-trailing reductions — `src/codegen/forward.rs`
9. **PR34-55**: @reduce doesn't validate rank actually decreases — `src/validator/`

Then low severity:
10-19: PR34-10, PR34-28, PR34-41, PR34-49, PR34-60, PR34-62, PR34-63, PR34-64, PR34-20, PR34-53

**Approach:** For each item, assess whether it's a real bug (fix it), a valid concern (add test + fix), or a non-issue (add comment explaining why and close). Document each decision in the PR description.

**Step 10.1: Triage all 19 items**

Read each item in the audit doc. For each, determine: fix / test-only / document-and-close.

**Step 10.2: Fix medium-severity items**

Address items 1-9 above. Each fix should include a test.

**Step 10.3: Address low-severity items**

Items 10-19: fix where practical, document-and-close where the fix would be disproportionate.

**Step 10.4: Run tests and commit**

```bash
cargo test
cargo test --test integration_tests
cargo insta review
git add -A  # review staged files carefully
git commit -m "fix: address 19 PR review items from fat arrow PR #34 (#122)"
```

Note: This is a large task. The agent should make incremental commits (one per logical group of fixes).

---

### Task 11: JARVIS — #123

**Issue:** Address 13 unresolved review items from wrap/hyper PR #39

**Reference:** `docs/reviews/pr-comment-audit-2026-03-05.md`, items prefixed `PR39-`

**Items (by severity):**

Medium:
1. **PR39-13**: No semantic validation that @wrap target accepts Neuron-typed first param — `src/passes/desugar.rs`
2. **PR39-20**: `__sequential__` can collide with user-defined neuron names — `src/validator/core.rs`
3. **PR39-33**: No integration test for @wrap inside unroll block

Low:
4. **PR39-14**: Fragile grammar test assertions — `src/grammar/tests.rs`
5. **PR39-15**: `__sequential__` tracking may classify composites as primitives — `src/codegen/instantiation.rs`
6. **PR39-22**: Nested @wrap silently dropped — `src/passes/desugar.rs`
7. **PR39-26**: Silent Value::Call fallback in build_wrap_endpoint — `src/grammar/ast.rs`
8. **PR39-29**: test_parse_wrap_inline_pipeline only checks non-empty — `src/grammar/tests.rs`
9. **PR39-32**: Missing @wrap kwargs test — `src/grammar/tests.rs`
10. **PR39-34**: `__sequential__` bypasses arity/shape validation — `src/validator/symbol_table.rs`
11. **PR39-35**: Spurious primitive insertion for binding-reference args — `src/codegen/instantiation.rs`
12. **PR39-36**: Fragile token skipping in build_wrap_endpoint — `src/grammar/ast.rs`
13. **PR39-28**: WrapExpr.id reused by desugared Call — `src/passes/desugar.rs`

**Step 11.1: Fix PR39-13 (semantic validation)**

Add a validation check in `src/validator/core.rs` or `src/passes/desugar.rs` that verifies the @wrap target neuron's first parameter has type `: Neuron`.

**Step 11.2: Fix PR39-20 (name collision)**

Either:
- Reserve `__sequential__` as a keyword (add grammar rule), OR
- Use a UUID-based name like `__ns_sequential_{id}__`

**Step 11.3: Add @wrap in unroll integration test (PR39-33)**

Create an example file or inline test.

**Step 11.4: Address remaining low-severity items**

Fix where practical, document-and-close otherwise.

**Step 11.5: Run tests and commit**

```bash
cargo test
cargo test --test integration_tests
git add -A  # review staged files
git commit -m "fix: address 13 PR review items from wrap/hyper PR #39 (#123)"
```

---

### Batch 4 Merge

```bash
# Merge Ava first, then JARVIS
cargo test && ./test_examples.sh
```

---

## Post-Sprint

### Release

```bash
# On _dev, verify everything
cargo test && ./test_examples.sh
# Update CHANGELOG.md with Sprint 4 changes
# Create release PR: _dev → main
gh pr create --base main --head _dev --title "Sprint 4: v0.6.2 release" --body "..."
```

### Cleanup

```bash
# Remove all worktrees
git worktree list | grep sprint4 | awk '{print $1}' | xargs -I{} git worktree remove {}
# Delete merged branches
git branch | grep sprint4 | xargs git branch -d
```

---

## Worktree Isolation Protocol

**CRITICAL**: Every agent MUST run in a pre-created worktree. The `isolation: "worktree"` Agent parameter is unreliable.

For each batch:

```bash
# 1. Pre-create worktrees from _dev (after previous batch merged)
git worktree add ../neuroscript-rs-<agent> _dev -b sprint4/<agent>-<issues>

# 2. Agent prompt MUST include:
#    "Your working directory is /path/to/neuroscript-rs-<agent>.
#     Prefix ALL bash commands with: cd /path/to/neuroscript-rs-<agent> &&"

# 3. After agent completes, push from worktree:
cd ../neuroscript-rs-<agent> && git push -u origin sprint4/<agent>-<issues>

# 4. Create PR from branch
gh pr create --base _dev --head sprint4/<agent>-<issues> --title "..."

# 5. After merge, cleanup worktree
git worktree remove ../neuroscript-rs-<agent>
```

**No agent shares a worktree. No agent works in the main repo checkout.**
