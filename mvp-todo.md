# NeuroScript MVP: Shape-First Approach

    Goal: Showcase NeuroScript's unique value - shape-aware compilation with pattern matching

## Current Status Summary (Dec 2024)

**Completed:**
- ✅ Phase 1: Core Primitives (Fork, Add, Concat, Attention)
- ✅ Phase 2: Shape Inference System
- ✅ Phase 2.5: Match Expression Codegen Fixes  
- ✅ Phase 3: Pattern Matching System (mostly complete, 1 item remaining)
- ✅ Phase 4: Stdlib Composites (mostly complete, 1 item remaining)
- ✅ Phase 5: Enhanced Codegen (complete)
- ✅ Phase 7.1-7.4: Basic let/set bindings (parsing, IR, validation, basic codegen)

**In Progress:**
- Phase 6: End-to-End Validation (GPT-2 test)
- Phase 7.5+: Recursion support for let/set

**New Work:**
- Phase 8: Pest Grammar Migration (grammar written, needs AST builder)

---

## MVP Phase 1: Core Primitives

    Note: Phase 1 and Phase 2 can be developed concurrently. Shape inference can be built and tested using existing primitives, then validated with new primitives as they're added.

### 1.1 Add Missing Critical Primitives ✅ 

  - [x] Implement Fork primitive (split tensor into multiple references)
  - [x] Implement Add primitive (element-wise addition for residual connections)
  - [x] Implement Concat primitive (concatenate tensors)
  - [x] Implement ScaledDotProductAttention primitive (base attention primitive)
  - [x] Verify Softmax implementation (already registered)
  - [x] Test all new primitives
  - [x] Validate all primitives needed for transformers are working

**Deliverable**: All primitives needed for transformers implemented and tested

## MVP Phase 2: Shape Inference System ⭐ KILLER FEATURE

### 2.1 Implement Shape Inference Engine ✅

  - [x] Create src/shape/ module with inference engine
  - [x] Implement ShapeInferenceEngine struct
  - [x] Add shape constraint tracking from connections
  - [x] Implement dimension variable unification (e.g., d_model consistency)
  - [x] Build forward shape propagation through graph
  - [x] Add shape compatibility validation for all connections
  - [x] Implement constraint solving for expressions (dim * 4, seq - 1)
  - [x] Create detailed error reporting with shape context
  - [x] Test shape inference with simple examples
  - [x] Test dimension unification across multiple neurons
 
### 2.2 Integrate with Validator

  - [x] Run shape inference after parsing
  - [x] Validate all connections for shape compatibility
  - [x] Report shape errors with full context
  - [x] Display inferred shapes in validation output
  - [x] Test validation catches shape mismatches
  - [x] Create example showing caught shape errors (examples/shape_inference_demo.ns)

**Deliverable**: Compile-time shape validation catches all mismatches

## MVP Phase 2.5: Codegen Bug Fixes (Prerequisites for Pattern Matching)

### 2.5.1 Fix Match Expression Codegen Issues ✅

  - [x] Fix guard condition generation (where clauses not emitted)
  - [x] Fix return variable tracking in match expressions
  - [x] Fix parameter storage for guard evaluation
  - [x] Test guard conditions work correctly
  - [x] Verify examples/16-recursion.ns catches depth guard

**Context**: Three bugs in current codegen prevent match expressions from working correctly:
1. Guard conditions (`where depth > 0`) are parsed but not generated in Python
2. Match expressions return wrong variable (last temp instead of result)
3. Parameters not stored as instance variables, so guards can't reference them

**Files to modify**:
- `src/codegen.rs` - `generate_shape_check()`, `process_destination()`, `generate_init()`

**Deliverable**: Match expressions with guards work correctly

## MVP Phase 3: Pattern Matching System ⭐ KILLER FEATURE

### 3.1 Enhance Match Expression Support

  - [x] Implement compile-time match resolution for known shapes
  - [x] Support runtime match generation (Python if/elif) for dynamic shapes
  - [x] Add exhaustiveness checking for pattern coverage
  - [x] Implement dead branch elimination
  - [x] Create AdaptiveProjection example neuron
  - [x] Test compile-time pattern selection
  - [ ] Test runtime pattern fallback
  - [x] Validate exhaustiveness checking
 
### 3.2 Pattern Matching Optimizations

  - [x] Generate direct path when all shapes known at compile-time
  - [x] Reorder patterns for efficient checking (most specific first)
  - [x] Cache pattern checks to avoid redundant computation
  - [x] Test optimization with various pattern combinations

**Deliverable**: Match expressions work for both compile-time and runtime routing ✅

## MVP Phase 4: Stdlib Composites

### 4.1 Complete Standard Library Definitions

  - [x] Complete TransformerBlock.ns with full residual connections
  - [x] Implement MultiHeadAttention composite (not primitive)
  - [x] Build FFN variants
  - [x] Create TransformerStack composite
  - [x] Validate all composites with shape inference
  - [ ] Test composition correctness
 
### 4.2 Implement Stdlib Loading ✅

  - [x] Load and parse all .ns files from stdlib/
  - [x] Merge stdlib with user program
  - [x] Validate entire combined program
  - [x] Run shape inference on everything
  - [x] Test stdlib loading with example programs

**Deliverable**: Stdlib neurons available for codegen ✅

**Status**: Core functionality complete! Successfully loads 46 stdlib neurons.
**Note**: See `stdlib-production-ready.md` for remaining tasks to make this production-ready.
  - Known issues: Some stdlib files have validation errors (need fixing)
  - Primitives need proper shape definitions (currently using dummy ports)

## MVP Phase 5: Enhanced Codegen

### 5.1 Shape-Aware Code Generation ✅

  - [x] Emit shape assertions in forward()
  - [x] based on inferred shapes
  - [x] Optimize based on static shapes (fixed-size operations)
  - [x] Add inferred shape documentation in generated code comments
  - [x] Test generated code includes proper assertions
  - [x] Validate shape documentation is accurate

**Status**: Complete! Shape inference now runs during codegen and:
- Emits shape documentation comments for all operations
- Generates runtime assertions for concrete shapes (parameters, literals)
- Skips assertions for wildcards and unresolved dimensions
- Example output shows `# Linear() output shape: [*, out_dim]` comments
 
### 5.2 Recursive Dependency Resolution ✅

  - [x] Generate all dependencies recursively
  - [x] Implement topological sort for correct order
  - [x] Emit all classes in single Python file
  - [x] Test dependency resolution with complex graphs
  - [x] Validate generated code order is correct

**Status**: Complete! Implementation in `src/codegen/generator.rs`:
- `collect_dependencies()` uses post-order DFS for topological sort
- Tested with `examples/dependency_chain_test.ns` (linear dependencies)
- Tested with `examples/diamond_dependency_test.ns` (shared dependency)
- Dependencies correctly emitted before dependents
- Deduplication prevents duplicate class generation

### 5.3 Composite Neuron Handling ✅

  - [x] Detect composite vs primitive neurons
  - [x] Generate nested modules for composites
  - [x] Import only primitives from runtime
  - [x] Test composite generation
  - [x] Validate imports are minimal and correct

**Status**: Complete! Implementation verified:
- `NeuronDef::is_primitive()` correctly distinguishes primitive/composite
- Composite neurons generated as Python classes in output file
- Only primitives imported from `neuroscript_runtime.primitives.*`
- `StdlibRegistry::generate_imports()` deduplicates and sorts imports
- Verified with dependency test examples (only primitives in imports)

**Deliverable**: Generate complete, self-contained PyTorch modules ✅

## MVP Phase 6: End-to-End Validation

### 6.1 Generate GPT-2 Small ✅

  - [x] Create examples/gpt2_small.ns (self-contained single-layer GPT-2)
  - [x] Run codegen: ./target/release/neuroscript compile --neuron GPT2Small --output gpt2_small.py examples/gpt2_small.ns
  - [x] Verify complete gpt2_small.py is generated (includes FFN, GPTTransformerBlock, GPT2Small classes)
  - [x] Validate all dependencies are included (primitives imported from neuroscript_runtime)

### 6.2 Test in PyTorch ✅

  - [x] Import generated GPT-2 model in Python
  - [x] Instantiate GPT2Small(vocab_size=50257)
  - [x] Create test input: torch.randint(0, 50257, (2, 128))
  - [x] Run forward pass and verify output shape [2, 128, 50257]
  - [x] Validate model runs without errors
  - [x] Test with different batch sizes and sequence lengths

**Deliverable**: Fully functional GPT-2 generated from NeuroScript ✅

## MVP Phase 7: `let`/`set` Bindings, Scopes & Recursion ⭐ KILLER FEATURE

**UPDATED DESIGN DECISIONS (2026-01-01):**
- ✅ Higher-order neurons (neurons as parameters) - REQUIRED for Universal Transformer
- ✅ Recursive calls unrolled at compile time (simple approach, max depth: 100)
- ✅ Three scopes for data: global, static, instance
- ✅ Single `context:` block with annotations: `@static`, `@global`, `@lazy`
- ✅ Explicit dependencies: All cross-scope references declared in context block
- ✅ No scope crossing in graph: Can't use `@global.var` or `@static.var` in pipelines

### 7.1 Lexer & Parser Extensions

**Status:** Partially complete (set/let blocks exist, need `context:` migration)

  - [x] Add `set` keyword to lexer (`let` already reserved)
  - [ ] Add `context` keyword to lexer
  - [ ] Add annotation tokens: `@static`, `@global`, `@lazy`
  - [ ] Implement `parse_context_block()` to replace set/let blocks
  - [ ] Parse binding syntax: `name = NeuronCall(args)` with optional annotations
  - [ ] Parse module-level `@global` declarations
  - [ ] Support `Freeze(neuron)` meta-neuron syntax
  - [ ] Migrate existing set/let tests to context block

### 7.2 IR Extensions for Context & Scopes

**Status:** Needs refactor from set/let bindings to context model

  - [x] ~~Add `set_bindings: Vec<Binding>` to `NeuronBody::Graph`~~ (deprecated)
  - [x] ~~Add `let_bindings: Vec<Binding>` to `NeuronBody::Graph`~~ (deprecated)
  - [ ] Add `context: Vec<ContextBinding>` to `NeuronBody::Graph`
  - [ ] Define `ContextBinding` struct with name, neuron call, and scope annotation
  - [ ] Add `Scope` enum: `Instance`, `Static`, `Global` (with `Lazy` flag)
  - [ ] Add `global_bindings: Vec<GlobalBinding>` to `Program`
  - [ ] Update IR Display traits for context bindings
  - [ ] Test IR correctly represents all three scopes

### 7.3 Validation Rules for Context & Scopes

**Status:** Needs extension for scope validation

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

**Status:** Needs refactor from set/let to context model

  - [ ] Generate module-level `@global` bindings (Python module variables)
  - [ ] Generate `@static` bindings as class variables
  - [ ] Generate instance bindings in `__init__` (eager instantiation)
  - [ ] Generate `@lazy` bindings with conditional instantiation
  - [ ] Track bound names → scope + module instance mapping
  - [ ] Reference bound names in graph connections
  - [ ] Support weight sharing within scope boundaries
  - [ ] Test all three scopes generate correctly
  - [ ] Create examples/context_scopes_basic.ns

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
  - [ ] Define `@global` syntax at module level (outside neurons)
  - [ ] Define `context:` block syntax (replaces set:/let:)
  - [ ] Implement annotation parsing: `@static`, `@global`, `@lazy`
  - [ ] Validate scope rules (global only at module level, etc.)
  - [ ] Codegen for module-level globals (Python module variables)
  - [ ] Codegen for static bindings (Python class variables)
  - [ ] Codegen for instance bindings (Python instance variables)
  - [ ] Test all three scopes work correctly
  - [ ] Create examples/context_scopes.ns demonstrating all scopes

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
  - [ ] Generate flat structure in codegen (no recursive __init__)
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

  - [ ] Integrate context bindings with shape inference
  - [ ] Test context blocks with pattern matching (guards)
  - [ ] Create examples/18-context-scopes.ns (demonstrate all 3 scopes)
  - [ ] Create examples/19-universal-transformer.ns (higher-order + @static)
  - [ ] Create examples/20-recursive-stack.ns (GPT-2 style depth unrolling)
  - [ ] Test full integration end-to-end
  - [ ] Verify generated PyTorch code runs correctly for all examples

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
4. ✅ Stdlib composites (mostly complete)
5. ⏳ Recursive codegen (Phase 7.5+)
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

* Optimization passes (identical layer deduplication, dead code elimination)
* Better error messages (especially for scope violations)
* Pre-compiled stdlib
* Training examples
* Runtime-variable recursion depth (with explicit unroll limits)

### Explicitly NOT in Scope (Design Decisions)

* ❌ Complex termination analysis - user responsible for valid recursion patterns
* ❌ Runtime-variable recursion depth - compile-time only (max 100)
* ❌ Automatic layer deduplication in unrolled recursion - user must optimize manually
* ❌ Direct scope crossing in graph (`@global.x` in pipelines) - must use `context:` bindings
* ❌ Port references in bindings - future work
* ❌ Iteration count annotations (`@iterate(3)`) - future work

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

  - [ ] Create `src/grammar/ast.rs` - Convert pest Pairs to IR types
  - [ ] Implement `build_program()` entry point
  - [ ] Handle all grammar rules → IR conversion
  - [ ] Create `src/grammar/indent.rs` - Indentation validation
  - [ ] Create `src/grammar/error.rs` - Convert pest errors to ParseError
  - [ ] Test AST builder with simple examples first
  - [ ] Validate output matches existing parser

### 8.3 Integration

  - [ ] Create feature flag for gradual migration (e.g., `--use-pest`)
  - [ ] Compare outputs between old and new parsers
  - [ ] Run full test suite with new parser
  - [ ] Benchmark performance comparison
  - [ ] Address any discrepancies
  - [ ] Remove old lexer/parser once validated

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
