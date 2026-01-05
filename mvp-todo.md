# NeuroScript MVP: Shape-First Approach

    Goal: Showcase NeuroScript's unique value - shape-aware compilation with pattern matching

## Current Status Summary (Dec 2024)

**Completed:**

- ✅ Phase 1: Core Primitives (Fork, Add, Concat, Attention)
- ✅ Phase 2: Shape Inference System
- ✅ Phase 2.5: Match Expression Codegen Fixes  
- ✅ Phase 3: Pattern Matching System
- ✅ Phase 4: Stdlib Composites
- ✅ Phase 5: Enhanced Codegen
- ✅ Phase 7.1-7.4: Context bindings (parsing, IR, validation, codegen)
- ✅ Phase 8: Pest Grammar Migration (Complete)

- Phase 6: End-to-End Validation (GPT-2 test)
- Phase 7.5+: Recursion support for context bindings
- ✅ Phase 8: Pest Grammar Migration (Complete)

---

## MVP Phase 1: Core Primitives

**summary**: Implement core primitives needed for transformers

## MVP Phase 2: Shape Inference System

**summary**: Implement shape inference system for transformers with validator

## MVP Phase 2.5: Codegen Bug Fixes (Prerequisites for Pattern Matching)

**summary**: Fix match expression codegen issues

## MVP Phase 3: Pattern Matching System

**summary**: Implement pattern matching system for transformers with optimizer

## MVP Phase 4: Stdlib Composites

### 4.1 Complete Standard Library Definitions

- [x] Complete TransformerBlock.ns with full residual connections
- [x] Implement MultiHeadAttention composite (not primitive)
- [x] Build FFN variants
- [x] Create TransformerStack composite
- [x] Validate all composites with shape inference
- [ ] Test composition correctness

### 4.2 Implement Stdlib Loading ✅

**summary**: Implement stdlib loading for transformers
**to-do**: see @stdlib-production-ready.md

## MVP Phase 5: Enhanced Codegen

**summary**: Implement enhanced codegen for transformers

## MVP Phase 6: End-to-End Validation

**summary**: Implement end-to-end validation for transformers by generating and testing a GPT-2 model

## MVP Phase 7: `context` Bindings, Scopes & Recursion ⭐ KILLER FEATURE

**UPDATED DESIGN DECISIONS (2026-01-01):**

- ✅ Higher-order neurons (neurons as parameters) - REQUIRED for Universal Transformer
- ✅ Recursive calls unrolled at compile time (simple approach, max depth: 100)
- ✅ Three scopes for data: global, static, instance
- ✅ Single `context:` block with annotations: `@static`, `@global`, `@lazy`
- ✅ Explicit dependencies: All cross-scope references declared in context block
- ✅ No scope crossing in graph: Can't use `@global.var` or `@static.var` in pipelines

### 7.1 Lexer & Parser Extensions

**Status:** Complete

- [x] Add `set` keyword to lexer (`let` already reserved)
- [x] Add `context` keyword to lexer
- [x] Add annotation tokens: `@static`, `@global`, `@lazy`
- [x] Implement `parse_context_block()` to replace set/let blocks
- [x] Parse binding syntax: `name = NeuronCall(args)` with optional annotations
- [x] Parse module-level `@global` declarations
- [x] Support `Freeze(neuron)` meta-neuron syntax
- [x] Migrate existing set/let tests to context block

### 7.2 IR Extensions for Context & Scopes

**Status:** Complete

- [x] ~~Add `set_bindings: Vec<Binding>` to `NeuronBody::Graph`~~ (deprecated)
- [x] ~~Add `let_bindings: Vec<Binding>` to `NeuronBody::Graph`~~ (deprecated)
- [x] Add `context_bindings: Vec<Binding>` to `NeuronBody::Graph`
- [x] Update `Binding` struct with `scope: Scope`
- [x] Add `Scope` enum: `Instance`, `Static`, `Global` (with `Lazy` flag)
- [x] Add `global_bindings: Vec<GlobalBinding>` to `Program`
- [x] Update IR Display traits for context bindings
- [x] Test IR correctly represents all three scopes

### 7.3 Validation Rules for Context & Scopes

**Status:** Complete (basic scale validation implemented)

- [x] Check no forward references in bindings (existing)
- [x] Validate binding names don't conflict (existing)
- [x] Validate bound neuron names exist (existing)
- [ ] **NEW:** Validate `@global` only appears at module level (not in neurons)
- [ ] **NEW:** Validate `@static`/`@lazy` only appear in `context:` blocks
- [ ] **NEW:** Validate graph connections only reference context-bound names (no `@global.x`)
- [ ] **NEW:** Validate instance bindings can't reference static bindings of same neuron
- [ ] **NEW:** Validate static bindings can reference globals and other statics
- [ ] Test validation catches scope violations

### 7.4 Basic Codegen (Context Block, No Recursion)

**Status:** Complete

- [x] Generate module-level `@global` bindings (Python module variables)
- [x] Generate `@static` bindings as class variables
- [x] Generate instance bindings in `__init__` (eager instantiation)
- [x] Generate `@lazy` bindings with conditional instantiation
- [x] Track bound names → scope + module instance mapping
- [x] Reference bound names in graph connections
- [x] Support weight sharing within scope boundaries
- [x] Test all three scopes generate correctly
- [x] Create examples/context_scopes_basic.ns

### 7.5 Three-Scope Data Model with Context Block

**Design:** Single `context:` block with three scopes via annotations

**Syntax:**

```neuroscript
# Module level - only place for @global definitions
@global vocab_table = Embedding(50257, 768)
@global num_heads = 12

neuron TransformerBlock(d_model):
  in: [*, seq, d_model]
  out: [*, seq, d_model]

  context:
    @static shared_norm = LayerNorm(d_model)              # Class-level (shared)
    attn = MultiHeadAttention(d_model, @global num_heads) # Instance (references global)
    @lazy ffn = FFN(d_model)                              # Instance, lazy-loaded

  graph:
    in -> shared_norm -> attn -> ffn -> out  # Only bound names, no @global.x
```

**Scope Rules:**

1. **global scope**: Module-level only, `@global name = ...`
   - Shared across ALL neurons in the module
   - Cannot be defined inside neurons
   - Referenced in context blocks via `@global name`

2. **static scope**: Class-level, `@static name = ...` in `context:`
   - Shared across all instances of THIS neuron type
   - One copy per neuron class, not per instance
   - Can reference `@global` bindings

3. **instance scope**: Instance-level, `name = ...` in `context:` (default)
   - Per-instance data (standard neural network weights)
   - Can reference `@global` and `@static` bindings
   - Can be marked `@lazy` for conditional instantiation

**Strict Boundaries:**

- Graph block can ONLY reference names bound in `context:` block
- No direct `@global.x` or `@static.x` access in pipelines
- All cross-scope dependencies explicit in `context:` declarations

**Implementation tasks:**

- [x] Define `context:` block syntax (replaces set:/let:)
- [x] Implement annotation parsing: `@static`, `@global`, `@lazy`
- [x] Validate scope rules (global only at module level, etc.)
- [x] Codegen for module-level globals (Python module variables)
- [x] Codegen for static bindings (Python class variables)
- [x] Codegen for instance bindings (Python instance variables)
- [x] Test all three scopes work correctly
- [x] Create examples/context_scopes.ns demonstrating all scopes

### 7.6 Compile-Time Recursion Unrolling & Higher-Order Neurons

**Design:** Simple unrolling approach with first-class neuron parameters

**Recursion Requirements:**

- Recursion depth MUST be known at compile time
- Error if depth is runtime-determined
- Max unroll depth: **100** (prevents accidental runaway compilation)
- Generate flat unrolled structure (no runtime recursion)

**Higher-Order Neurons:**

- Neurons can accept other neurons as parameters (first-class)
- Enables Universal Transformer pattern (pass shared block instance)
- Type checking for neuron parameters via shape inference

**Implementation tasks:**

- [ ] Support neuron-typed parameters in neuron signatures
- [ ] Implement type checking for neuron parameters (shape compatibility)
- [ ] Detect self-referential neurons in `context:` bindings
- [ ] Implement compile-time guard evaluator (for recursion control)
- [ ] Unroll recursive calls by substituting parameters
- [ ] Track unroll depth, error if exceeds 100
- [ ] Generate flat structure in codegen (no recursive **init**)
- [ ] Test higher-order neurons (Universal Transformer example)
- [ ] Test unrolling with simple countdown pattern
- [ ] Verify examples/16-recursion.ns unrolls correctly
- [ ] Error gracefully on runtime-dependent depth

**Example - Higher-Order Neuron:**

```neuroscript
neuron ApplyNTimes(block: Neuron, depth: int):
  in: [*]
  out: [*]
  context:
    @lazy next = ApplyNTimes(block, depth - 1)
  graph:
    in -> match:
      [*] where depth > 0: block -> next -> out
      [*]: out

neuron UniversalTransformer(d_model, num_heads, depth):
  context:
    @static shared_block = TransformerBlock(d_model, num_heads)
  graph:
    in -> ApplyNTimes(shared_block, depth) -> out
```

**Limitations (accepted for simplicity):**

- No termination checking (user responsible for valid patterns)
- No support for runtime-variable depth
- No optimization for identical unrolled layers
- Max recursion depth fixed at 100

### 7.7 Integration & Examples

- [x] Integrate context bindings with shape inference
- [x] Test context blocks with pattern matching (guards)
- [x] Create examples/18-context-scopes.ns (demonstrate all 3 scopes)
- [ ] Create examples/19-universal-transformer.ns (higher-order + @static)
- [ ] Create examples/20-recursive-stack.ns (GPT-2 style depth unrolling)
- [x] Test full integration end-to-end
- [x] Verify generated PyTorch code runs correctly for all examples

**Context**: Evolution from `let`/`set` specification in `notes/neuroscript_let_set_spec.md`

**Key Concepts**:

- `context:` block = single place for all dependency declarations
- Annotations: `@global` (module-level), `@static` (class-level), `@lazy` (conditional)
- Default (no annotation) = instance scope, eager instantiation
- Bound names enable weight sharing within scope boundaries
- Inline calls create independent instances
- Simple recursion unrolling (compile-time depth only, max 100)
- Three scopes: global, static, instance
- **Higher-order neurons** (neurons as first-class parameters)
- Strict scope boundaries (no `@global.x` in graph block)

**Deliverable**: Weight sharing, scoped data, higher-order neurons, and recursion unrolling

**Known Limitations (accepted tradeoffs):**

- Recursion depth must be compile-time constant (max 100)
- No automatic termination checking (user responsible)
- No runtime-variable depth (compile-time only)
- No optimization for identical unrolled layers

## MVP Success Criteria

 ✅ Compile-time shape checking: Catches shape mismatches before codegen
 ✅ Pattern matching: Routes based on shapes at compile-time when possible
 ✅ Working GPT-2: Generates and runs a complete GPT-2 Small model
 ✅ Stdlib integration: All stdlib composites load and validate
 ✅ Killer demo: Show shape inference catching errors and routing efficiently

## Priority Summary

### Must-have for MVP

1. ✅ Missing primitives (Fork, Add, Concat, Attention)
2. ✅ Shape inference engine ⭐
3. ✅ Pattern matching ⭐ (mostly complete)
4. ✅ Stdlib composites
5. ⏳ Recursive codegen (Phase 7.6)
6. ✅ End-to-end GPT-2 test

### Recommended Next Steps

**Option A: Complete Scopes & Recursion (Phase 7.5-7.7)**

- Design and implement three-scope data model
- Implement simple compile-time recursion unrolling
- Create examples demonstrating scopes and recursion

**Option B: Pest Grammar Migration First (Phase 8.2-8.3)**

- Build AST converter for pest grammar
- Replace handwritten lexer/parser
- Cleaner foundation before adding scopes/recursion

**Recommendation:** Option B (Pest grammar) provides cleaner foundation for adding scopes

### Nice-to-have (post-MVP)

- Optimization passes (identical layer deduplication, dead code elimination)
- Better error messages (especially for scope violations)
- Pre-compiled stdlib
- Training examples
- Runtime-variable recursion depth (with explicit unroll limits)

### Explicitly NOT in Scope (Design Decisions)

- ❌ Complex termination analysis - user responsible for valid recursion patterns
- ❌ Runtime-variable recursion depth - compile-time only (max 100)
- ❌ Automatic layer deduplication in unrolled recursion - user must optimize manually
- ❌ Direct scope crossing in graph (`@global.x` in pipelines) - must use `context:` bindings
- ❌ Port references in bindings - future work
- ❌ Iteration count annotations (`@iterate(3)`) - future work

---

## MVP Phase 8: Pest Grammar Migration ⭐ INFRASTRUCTURE

**Goal:** Replace handwritten lexer/parser with pest PEG grammar for better maintainability

**Status:** Grammar written and tested, needs AST builder to integrate

### 8.1 Grammar Definition ✅

- [x] Create `src/grammar/neuroscript.pest` (~375 lines)
- [x] Define all keywords, operators, literals
- [x] Implement shapes and dimension expressions
- [x] Support flexible port syntax (inline and indented)
- [x] Handle all three connection styles (inline, indented, indented with arrows)
- [x] Parse match expressions with guards
- [x] Parse let/set binding blocks
- [x] Test grammar parses all example files (22 tests passing)

**Files created:**

- `src/grammar/neuroscript.pest` - Complete PEG grammar
- `src/grammar/mod.rs` - Parser + tests

**Design decisions documented in:** `docs/pest-grammar-review.md`

### 8.2 AST Builder (Next Step)

- [x] Create `src/grammar/ast.rs` - Convert pest Pairs to IR types
- [x] Implement `build_program()` entry point
- [x] Handle all grammar rules → IR conversion
- [x] Create `src/grammar/indent.rs` - Indentation validation (handled by Pest/AST logic)
- [x] Create `src/grammar/error.rs` - Convert pest errors to ParseError
- [x] Test AST builder with simple examples first
- [x] Validate output matches existing parser
- [x] Compare outputs between old and new parsers
- [x] Run full test suite with new parser
- [x] Address any discrepancies
- [x] Remove old lexer/parser once validated

**Key Design Decisions:**

1. Grammar ignores physical indentation; AST builder validates it
2. `in`/`out` are context-sensitive (keywords in sections, identifiers in graph)
3. Supports all three connection styles found in examples
4. Uses negative lookahead to detect new connections in indented pipelines

**Benefits of Migration:**

- More maintainable grammar definition
- Better error messages from pest
- Easier to extend syntax
- Standard PEG tooling
