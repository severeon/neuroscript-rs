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

### 2.1 Implement Shape Inference Engine

  - [x] Create src/shape_inference.rs module
  - [x] Implement ShapeInferenceEngine struct
  - [x] Add shape constraint tracking from connections
  - [x] Implement dimension variable unification (e.g., d_model consistency)
  - [x] Build forward shape propagation through graph
  - [ ] Add shape compatibility validation for all connections
  - [ ] Implement constraint solving for expressions (dim * 4, seq - 1)
  - [ ] Create detailed error reporting with shape context
  - [ ] Test shape inference with simple examples
  - [ ] Test dimension unification across multiple neurons
 
### 2.2 Integrate with Validator

  - [ ] Run shape inference after parsing
  - [ ] Validate all connections for shape compatibility
  - [ ] Report shape errors with full context
  - [ ] Display inferred shapes in validation output
  - [ ] Test validation catches shape mismatches
  - [ ] Create example showing caught shape errors

**Deliverable**: Compile-time shape validation catches all mismatches

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
