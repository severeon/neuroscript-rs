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
- ✅ First-order neurons only (no neuron-as-parameter support needed)
- ✅ Recursive calls unrolled at compile time (simple approach)
- ✅ Three scopes for data: model/global, neuron/static, node/instance
- ✅ Lazy loading via keyword/operator (syntax TBD)

### 7.1 Lexer & Parser Extensions ✅

  - [x] Add `set` keyword to lexer (`let` already reserved)
  - [x] Implement `parse_set_block()` for eager bindings
  - [x] Implement `parse_let_block()` for lazy bindings
  - [x] Parse binding syntax: `name = NeuronCall(args)`
  - [ ] Support `Freeze(neuron)` meta-neuron syntax
  - [x] Test parsing of let/set blocks

### 7.2 IR Extensions for Bindings ✅

  - [x] Add `set_bindings: Vec<Binding>` to `NeuronBody::Graph`
  - [x] Add `let_bindings: Vec<Binding>` to `NeuronBody::Graph`
  - [x] Define `Binding` struct with name + neuron call
  - [x] Update IR Display traits for bindings
  - [x] Test IR correctly represents bindings

### 7.3 Validation Rules for Bindings ✅

  - [x] Check no forward references in bindings
  - [x] Validate binding names don't conflict (duplicate binding check)
  - [x] Validate bound neuron names exist
  - [x] Check for recursive set: bindings (not allowed)
  - [x] Check for let: bindings with parameters (recursion control)
  - [x] Test validation catches binding errors

### 7.4 Basic Codegen (Weight Sharing, No Recursion) ✅

  - [x] Generate `set:` bindings in `__init__` (eager instantiation)
  - [x] Generate `let:` bindings with lazy instantiation markers
  - [x] Track bound names → module instance mapping
  - [x] Reference bound names in graph connections
  - [x] Support weight sharing (same instance multiple uses)
  - [x] Test basic set bindings generate correctly
  - [x] Test basic let bindings generate correctly with lazy instantiation
  - [x] Create examples/let_set_basic.ns

### 7.5 Three-Scope Data Model (NEW)

**Design:** Three distinct scopes for neuron data/weights:

1. **model/global scope**: Shared across ALL instances of ALL neurons
   - Use case: Global statistics, shared embeddings, vocabulary
   - Syntax TBD (e.g., `global:` block or `@global` annotation)

2. **neuron/static scope**: Shared across all instances of a SPECIFIC neuron type
   - Use case: Weight-shared layers (Universal Transformer)
   - Syntax TBD (e.g., `static:` block or `@static` annotation)

3. **node/instance scope**: Per-instance data (current default)
   - Use case: Regular neural network weights
   - Current `set:` and `let:` blocks map here

**Implementation tasks:**
  - [ ] Design syntax for three scopes
  - [ ] Extend IR to track scope annotations
  - [ ] Update parser to handle new scope keywords/annotations
  - [ ] Implement scope validation (e.g., global can't reference instance data)
  - [ ] Codegen for model/global scope (module-level variables)
  - [ ] Codegen for neuron/static scope (class-level variables)
  - [ ] Test all three scopes work correctly
  - [ ] Create example demonstrating all three scopes

**Open questions:**
  - How does model/global interact with multiple top-level neurons?
  - Should lazy loading apply to all three scopes?
  - How does this interact with shape inference?

### 7.6 Compile-Time Recursion Unrolling (SIMPLIFIED)

**Design:** Simple unrolling approach - no complex termination analysis

**Requirements:**
  - Recursion depth MUST be known at compile time
  - Error if depth is runtime-determined
  - Configurable max unroll depth (default: 100)
  - Generate flat unrolled structure (no runtime recursion)

**Implementation tasks:**
  - [ ] Detect self-referential neurons in `let:` bindings
  - [ ] Implement compile-time guard evaluator (only for recursion control)
  - [ ] Unroll recursive calls by substituting parameters
  - [ ] Track unroll depth, error if exceeds limit
  - [ ] Generate flat structure in codegen (no recursive __init__)
  - [ ] Test unrolling with simple countdown pattern
  - [ ] Verify examples/16-recursion.ns unrolls correctly
  - [ ] Error gracefully on runtime-dependent depth

**Limitations (accepted for simplicity):**
  - No termination checking (user responsible for valid patterns)
  - No support for runtime-variable depth
  - No optimization for identical unrolled layers

### 7.7 Integration & Examples

  - [ ] Integrate bindings with shape inference
  - [ ] Test let/set with pattern matching (guards)
  - [ ] Create examples/18-scopes.ns (demonstrate 3 scopes)
  - [ ] Create examples/19-recursive-stack.ns (GPT-2 style depth unrolling)
  - [ ] Test full integration end-to-end
  - [ ] Verify generated PyTorch code runs

**Context**: Simplified from full `let`/`set` specification in `notes/neuroscript_let_set_spec.md`

**Key Concepts**:
- `set:` = eager instantiation (always created in __init__)
- `let:` = lazy instantiation (only if referenced in active path)
- Bound names enable weight sharing (same instance reused)
- Inline calls create independent instances
- Simple recursion unrolling (compile-time depth only)
- Three scopes: model/global, neuron/static, node/instance
- **NO higher-order neurons** (first-order only)

**Deliverable**: Weight sharing, scoped data, and simple recursion unrolling

**Known Limitations (accepted tradeoffs):**
- Cannot pass neurons as parameters (blocks Universal Transformer pattern from spec)
- Recursion depth must be compile-time constant
- No automatic termination checking

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

* ❌ Higher-order neurons (neurons as parameters) - first-order only
* ❌ Complex termination analysis - user responsible for valid recursion
* ❌ Port references in bindings - future work
* ❌ Iteration count annotations - future work

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
