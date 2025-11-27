# NeuroScript MVP: Shape-First Approach

    Goal: Showcase NeuroScript's unique value - shape-aware compilation with pattern matching

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

  - [x] Create src/shape_inference.rs module
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

  - [ ] Implement compile-time match resolution for known shapes
  - [ ] Support runtime match generation (Python if/elif) for dynamic shapes
  - [ ] Add exhaustiveness checking for pattern coverage
  - [ ] Implement dead branch elimination
  - [ ] Create AdaptiveProjection example neuron
  - [ ] Test compile-time pattern selection
  - [ ] Test runtime pattern fallback
  - [ ] Validate exhaustiveness checking
 
### 3.2 Pattern Matching Optimizations

  - [ ] Generate direct path when all shapes known at compile-time
  - [ ] Reorder patterns for efficient checking (most specific first)
  - [ ] Cache pattern checks to avoid redundant computation
  - [ ] Test optimization with various pattern combinations

**Deliverable**: Match expressions work for both compile-time and runtime routing

## MVP Phase 4: Stdlib Composites

### 4.1 Complete Standard Library Definitions

  - [ ] Complete TransformerBlock.ns with full residual connections
  - [ ] Implement MultiHeadAttention composite (not primitive)
  - [ ] Build FFN variants
  - [ ] Create TransformerStack composite
  - [ ] Validate all composites with shape inference
  - [ ] Test composition correctness
 
### 4.2 Implement Stdlib Loading

  - [ ] Load and parse all .ns files from stdlib/
  - [ ] Merge stdlib with user program
  - [ ] Validate entire combined program
  - [ ] Run shape inference on everything
  - [ ] Test stdlib loading with example programs

**Deliverable**: Stdlib neurons available for codegen

## MVP Phase 5: Enhanced Codegen

### 5.1 Shape-Aware Code Generation

  - [ ] Emit shape assertions in forward()
  - [ ] based on inferred shapes
  - [ ] Optimize based on static shapes (fixed-size operations)
  - [ ] Add inferred shape documentation in generated code comments
  - [ ] Test generated code includes proper assertions
  - [ ] Validate shape documentation is accurate
 
### 5.2 Recursive Dependency Resolution

  - [ ] Generate all dependencies recursively
  - [ ] Implement topological sort for correct order
  - [ ] Emit all classes in single Python file
  - [ ] Test dependency resolution with complex graphs
  - [ ] Validate generated code order is correct
 
### 5.3 Composite Neuron Handling

  - [ ] Detect composite vs primitive neurons
  - [ ] Generate nested modules for composites
  - [ ] Import only primitives from runtime
  - [ ] Test composite generation
  - [ ] Validate imports are minimal and correct

**Deliverable**: Generate complete, self-contained PyTorch modules

## MVP Phase 6: End-to-End Validation

### 6.1 Generate GPT-2 Small

  - [ ] Create examples/transformer_from_stdlib.ns using stdlib
  - [ ] Run codegen: ./target/release/neuroscript --codegen GPT2Small --output gpt2_small.py
  - [ ] Verify complete gpt2_small.py is generated
  - [ ] Validate all dependencies are included
 
### 6.2 Test in PyTorch

  - [ ] Import generated GPT-2 model in Python
  - [ ] Instantiate GPT2Small(vocab_size=50257)
  - [ ] Create test input: torch.randint(0, 50257, (2, 128))
  - [ ] Run forward pass and verify output shape [2, 128, 50257]
  - [ ] Validate model runs without errors
  - [ ] Test with different batch sizes and sequence lengths
 
### 6.3 Shape Validation Examples

  - [ ] Create examples/shape_mismatch.ns (compile-time error demo)
  - [ ] Create examples/adaptive_routing.ns (pattern matching demo)
  - [ ] Create examples/shape_inference.ns (shows inferred shapes)
  - [ ] Document each example with expected behavior
  - [ ] Test all examples produce expected results

**Deliverable**: Fully functional GPT-2 generated from NeuroScript

## MVP Phase 7: `let`/`set` Bindings & Structural Recursion ⭐ KILLER FEATURE

### 7.1 Lexer & Parser Extensions

  - [ ] Add `set` keyword to lexer (`let` already reserved)
  - [ ] Implement `parse_set_block()` for eager bindings
  - [ ] Implement `parse_let_block()` for lazy bindings
  - [ ] Parse binding syntax: `name = NeuronCall(args)`
  - [ ] Support `Freeze(neuron)` meta-neuron syntax
  - [ ] Test parsing of let/set blocks

### 7.2 IR Extensions for Bindings

  - [ ] Add `set_bindings: Vec<SetBinding>` to `NeuronDef`
  - [ ] Add `let_bindings: Vec<LetBinding>` to `NeuronDef`
  - [ ] Define `SetBinding` struct with name + neuron call
  - [ ] Define `LetBinding` struct with name + neuron call
  - [ ] Update IR Display traits for bindings
  - [ ] Test IR correctly represents bindings

### 7.3 Validation Rules for Bindings

  - [ ] Check no forward references in bindings
  - [ ] Validate binding names don't conflict with parameters
  - [ ] Validate binding names don't conflict with ports
  - [ ] Check lazy bindings not used in eager (set) context
  - [ ] Validate bound neuron names exist
  - [ ] Test validation catches binding errors

### 7.4 Basic Codegen (Weight Sharing, No Recursion)

  - [ ] Generate `set:` bindings in `__init__` (eager instantiation)
  - [ ] Track bound names → module instance mapping
  - [ ] Reference bound names in graph connections
  - [ ] Support weight sharing (same instance multiple uses)
  - [ ] Test basic set bindings generate correctly
  - [ ] Test weight sharing with repeated references
  - [ ] Create examples/17-let-set-basics.ns

### 7.5 Recursion Detection & Analysis

  - [ ] Detect self-referential neurons in `let:` bindings
  - [ ] Identify recursion control parameter (e.g., depth)
  - [ ] Detect parameter decrease pattern (depth - 1, n - 2)
  - [ ] Identify base case (match arm without recursive call)
  - [ ] Test recursion detection on simple examples

### 7.6 Compile-Time Guard Evaluator

  - [ ] Implement expression evaluator for guard conditions
  - [ ] Support arithmetic: +, -, *, /
  - [ ] Support comparisons: <, >, <=, >=, ==, !=
  - [ ] Support parameter substitution in expressions
  - [ ] Test guard evaluation with various expressions
  - [ ] Handle evaluation errors gracefully

### 7.7 Recursive Expansion Algorithm

  - [ ] Implement compile-time expansion for recursive neurons
  - [ ] Evaluate guards with parameter substitution
  - [ ] Instantiate lazy bindings only in active code paths
  - [ ] Track expansion depth with configurable limit (default: 100)
  - [ ] Generate detailed error on expansion limit
  - [ ] Test expansion on countdown pattern

### 7.8 Termination Checking

  - [ ] Implement simple termination checker:
    - Single parameter decreases by constant
    - Guard is simple comparison (>, >=, etc.)
    - Base case exists (non-recursive arm)
  - [ ] Error on non-terminating patterns with helpful message
  - [ ] Error on complex patterns (multiple params, non-linear)
  - [ ] Test termination checker catches infinite recursion
  - [ ] Test termination checker allows valid patterns

### 7.9 Recursive Codegen

  - [ ] Generate conditional instantiation for `let:` bindings
  - [ ] Emit recursive module creation in `__init__`
  - [ ] Add depth/parameter tracking for termination
  - [ ] Generate proper forward pass routing
  - [ ] Test generated recursive code works
  - [ ] Verify examples/16-recursion.ns generates correctly

### 7.10 Integration & Examples

  - [ ] Integrate bindings with shape inference
  - [ ] Test let/set with pattern matching (guards)
  - [ ] Create examples/18-weight-sharing.ns (Universal Transformer)
  - [ ] Create examples/19-recursive-stack.ns (GPT-2 style depth)
  - [ ] Test full integration end-to-end
  - [ ] Verify generated PyTorch code runs

**Context**: Implements full `let`/`set` specification from `notes/neuroscript_let_set_spec.md`

**Key Concepts**:
- `set:` = eager instantiation (always created in __init__)
- `let:` = lazy instantiation (only if referenced in active path)
- Bound names enable weight sharing (same instance reused)
- Inline calls create independent instances
- Structural recursion via self-reference + guards
- Compile-time expansion (no runtime recursion)

**Deliverable**: Full support for weight sharing and recursive neuron definitions

## MVP Success Criteria

 ✅ Compile-time shape checking: Catches shape mismatches before codegen
 ✅ Pattern matching: Routes based on shapes at compile-time when possible
 ✅ Working GPT-2: Generates and runs a complete GPT-2 Small model
 ✅ Stdlib integration: All stdlib composites load and validate
 ✅ Killer demo: Show shape inference catching errors and routing efficiently

## Priority Summary

### Must-have for MVP

1. Missing primitives (Fork, Add, Concat, Attention)
2. Shape inference engine ⭐
3. Enhanced pattern matching ⭐
4. Complete stdlib composites
5. Recursive codegen
6. End-to-end GPT-2 test

### Nice-to-have (post-MVP)

* Optimization passes
* Better error messages
* Pre-compiled stdlib
* Training examples
