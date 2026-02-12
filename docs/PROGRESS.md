# NeuroScript Progress Log

Tracks what was built, what's left, known bugs, and key decisions. Referenced from the root CLAUDE.md.

---

## Session: 2026-02-11 — Variadic Input Ports (#18)

### What Was Done

Implemented variadic input ports as a general language feature (`in *inputs: [*shape]`), resolving language-gap #3 (Concat limited to 2 inputs).

**Files changed (23 total):**

| Layer | Files | Change |
|-------|-------|--------|
| IR | `src/interfaces.rs` | Added `pub variadic: bool` to `Port` struct |
| Grammar | `src/grammar/neuroscript.pest` | Added `star ~ ident ~ colon ~ shape` as first alternative in `in_section` |
| AST Builder | `src/grammar/ast.rs` | Added `is_variadic` state, `Rule::star` match, propagation to Port |
| Validator | `src/validator/core.rs` | Variadic port declaration validation (no variadic outputs, must have explicit name, single variadic input only) |
| Validator | `src/validator/symbol_table.rs` | Variadic port registered as "in" alias; variadic arity bypass in `check_port_compatibility` |
| Validator | `src/validator/shapes.rs` | Propagated `variadic` field through `substitute_params` |
| Shape Inference | `src/shape/inference.rs` | Per-element validation of source shapes against variadic port; skips arity check |
| Codegen | `src/codegen/generator.rs` | Variadic neuron gets single tuple parameter in `forward()` |
| CLI | `src/main.rs` | `*` prefix for variadic ports in all 4 display locations |
| Snapshots | `tests/integration_tests.rs` | `*` prefix in snapshot formatter |
| Stdlib | `stdlib/primitives/Concat.ns` | Changed from `in a`/`in b` to `in *inputs: [*shape]` |
| Examples | `examples/primitives/structural.ns`, `examples/tutorials/02_fork_join.ns` | Updated local Concat defs |
| New example | `examples/variadic_concat.ns` | Demonstrates 2-, 3-, 4-input concat and ConcatNorm wrapper |
| Tests | 6 files | 9 new tests (grammar, AST, validator) |

**Test counts:** 191 unit tests, 13 integration/snapshot tests, all passing.

### Key Design Decisions

1. **Variadic is input-only.** Output ports cannot be variadic. Rationale: variable-length outputs have no clear semantics in the dataflow graph (how would you connect them?).

2. **Single variadic port per neuron.** A neuron with `in *inputs: [*shape]` must have exactly one `in` declaration. Multiple variadic ports or mixed variadic + fixed ports are rejected.

3. **Explicit name required.** `in *inputs: [shape]` works; `in *: [shape]` (unnamed/default) is rejected. This avoids ambiguity with the default port alias "in".

4. **Per-element shape validation only.** Each tuple element is validated against the variadic port's shape individually. Cross-input compatibility (e.g., non-concat dimensions must match) is left to the Python runtime. This is intentional — the type system validates the declared constraint, runtime catches inter-element mismatches.

5. **Composite variadic neurons pass tuple as-is.** In `neuron ConcatNorm: in *inputs: [*shape]`, the `in` node carries the full tuple. `in -> Concat(1)` passes the entire tuple to Concat. There is no per-element indexing syntax (`inputs[0]`).

6. **Symbol table registers variadic under "in" alias.** A single variadic input port gets the "in" node alias (same as a default port), so `in -> Neuron()` works in graph bodies. The two cases (default name vs variadic) are mutually exclusive because validation enforces variadic ports have `name != "default"`.

### Exploration Findings (Persistable Knowledge)

**How the "in" node gets resolved in composite neurons:**
- Symbol table (`src/validator/symbol_table.rs:60-76`): If the neuron has a single input port named "default" OR a single variadic port, it's registered as `"in"`. Otherwise each named port gets its own node.
- Shape inference (`src/shape/inference.rs:320-332`): Always registers `"in"` from `neuron.inputs` shapes, plus individual named ports.
- The validator and shape inference have separate resolution paths — both need updating when changing port semantics.

**Port compatibility check ordering matters:**
- In `check_port_compatibility`, the variadic check must come before the implicit fork check. A tuple like `(a, b, c)` flowing into a variadic neuron should match the variadic path (N→1), not fall through to arity mismatch.

**Squash merge gotcha:**
- PR #18 was merged in two squash commits (`71d2f87`, `8ca8e68`). The second contained review-round tests but the validation code from `core.rs` was lost. Fixed on 2026-02-12 by re-adding the validation. Always verify tests pass after squash merge.

---

## Known Bugs / Pre-existing Issues

### Parse Failures (Pre-existing, Not Caused by Variadic Work)

| File | Error | Notes |
|------|-------|-------|
| `examples/primitives/embeddings.ns` | `Expected COMMENT, rparen, or call_args` | Likely uses unsupported syntax (kwargs or string args) |
| `examples/tutorials/03_match_guards.ns` | `Expected COMMENT, colon, slash, plus, minus, star, or comparison_op` at line 123 | Possibly uses `%` modulo or other unsupported operator in shape expression |

### Stdlib Validation Errors (Pre-existing)

Running `neuroscript validate` on any file that loads stdlib produces 18 errors from complex stdlib neurons (MultiQueryAttention, CrossAttention, GroupedQueryAttention, TransformerEncoderBlock, TransformerDecoderBlock, ViTBlock, ConvNeXtBlock). All are shape mismatches like `[batch, seq, dim] -> [*, in_dim]` where the validator's wildcard matching doesn't unify 3-dim shapes against `[*, x]` patterns. These predate the variadic work.

---

## What's Left (Language Gaps from `notes/language-gaps.md`)

| # | Gap | Status | Difficulty |
|---|-----|--------|------------|
| 1 | Subtract/Divide primitives | **Done** (PR #16) | Easy |
| 2 | Doc comment blank line | **Done** (PR #17) | Easy |
| 3 | Concat limited to 2 inputs | **Done** (PR #18) | Medium-Hard |
| 4 | Loop/repeat construct | Open | Hard |
| 5 | `context` reserved keyword | Open | Easy-Medium |
| 6 | Inter-neuron stdlib refs | Open (works, needs testing) | N/A |

### Phase 3 Roadmap Items (from CLAUDE.md)

- Full dimension variable type inference across programs
- Loop constructs for repeated layers (gap #4)
- Higher-order neurons (neuron parameters)
- Graph simplification and fusion optimizations
- Multiple backends (ONNX, JAX, TorchScript)
