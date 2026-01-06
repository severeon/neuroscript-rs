# Stdlib Loading: Production-Ready Tasks

**Status**: Core functionality complete, needs refinement for production use

## Priority 1: Critical Issues (Blocking Production)

### 1.1 Fix Validation Errors in Stdlib Files

**Why**: 126+ validation errors prevent stdlib neurons from being used
**Tasks**:

- [x] Audit all 6 stdlib files for validation errors:
  - [x] FFN.ns - Port mismatch issues with primitives
  - [x] Residual.ns - Arity mismatch in tuple unpacking
  - [x] MultiHeadAttention.ns - Unknown node references
  - [x] TransformerBlock.ns - Cycle detection issues
  - [x] TransformerStack.ns - Shape mismatches
  - [x] MetaNeurons.ns - Arity mismatches
- [x] Fix or remove broken neurons
- [x] Create test cases for each stdlib neuron
- [x] Document known limitations/requirements

**Acceptance Criteria**: `./target/release/neuroscript validate examples/test_stdlib.ns` passes without errors

### 1.2 Define Primitive Neuron Signatures

**Why**: Current dummy port `[]` causes shape validation failures
**Options**:

- **Option A**: Create .ns files with primitive signatures in stdlib/primitives/
- **Option B**: Hard-code primitive signatures in StdlibRegistry
- **Option C**: Add primitive_signatures.toml config file

**Recommended**: Option A (most flexible, user-facing)

**Tasks**:

- [x] Create stdlib/primitives/ directory
- [x] Define signatures for core primitives:
  - [x] Linear.ns - `in: [*, in_dim], out: [*, out_dim]`
  - [x] GELU.ns, ReLU.ns, etc. - `in: [*shape], out: [*shape]`
  - [x] LayerNorm.ns - `in: [*, dim], out: [*, dim]`
  - [x] Dropout.ns - `in: [*shape], out: [*shape]`
  - [x] Softmax.ns - `in: [*, dim], out: [*, dim]`
  - [x] ScaledDotProductAttention.ns - proper Q/K/V signatures
  - [x] All other primitives from StdlibRegistry (30+ total)
- [x] Load primitive definitions automatically with stdlib
- [x] Test that primitives have proper shape validation

**Acceptance Criteria**: `SimpleModel` using `Linear -> GELU -> Linear` validates without port mismatches

### 1.3 Update Integration Tests

**Why**: Snapshot test is failing due to validator changes
**Tasks**:

- [x] Review snapshot diff for `example_28-let_set_basic`
- [x] Run `cargo insta review` and accept/reject changes
- [x] Ensure all 10 integration tests pass
- [x] Add new integration test for stdlib loading

**Acceptance Criteria**: `cargo test --test integration_tests` passes 100%

## Priority 2: User Experience (Critical for Usability)

### 2.1 Improve Error Messages

**Tasks**:

- [ ] Better stdlib loading errors:
  - [ ] Show which stdlib file failed to parse (with line numbers)
  - [ ] Suggest running with `--no-stdlib` if stdlib is broken
  - [ ] Differentiate between "stdlib not found" vs "stdlib parse error"
- [ ] Better validation errors for stdlib neurons:
  - [ ] Mark errors as coming from stdlib vs user code
  - [ ] Add `--skip-stdlib-validation` flag for development
  - [ ] Show which stdlib neuron has issues when user tries to use it

**Acceptance Criteria**: Users can quickly identify if error is in their code or stdlib

### 2.2 Add CLI Improvements

**Tasks**:

- [ ] Add `--stdlib-path <DIR>` flag to use custom stdlib location
- [ ] Add `neuroscript list-stdlib` command to show available stdlib neurons
- [ ] Show stdlib loading status in verbose mode (already done ✓)
- [ ] Cache parsed stdlib (optional performance optimization)

**Acceptance Criteria**: Users can easily discover and use stdlib neurons

### 2.3 Documentation

**Tasks**:

- [ ] Update CLAUDE.md with stdlib usage examples
- [ ] Create stdlib/README.md documenting:
  - [ ] How to use stdlib neurons
  - [ ] How to extend stdlib
  - [ ] Available neurons by category
  - [ ] Known issues/limitations
- [ ] Add docstrings to stdlib.rs functions
- [ ] Update CLI help text to mention stdlib loading

**Acceptance Criteria**: New users can understand and use stdlib without reading code

## Priority 3: Testing & Validation (Quality Assurance)

### 3.1 Comprehensive Test Suite

**Tasks**:

- [ ] Add unit tests for stdlib.rs:
  - [ ] Test stdlib directory discovery
  - [ ] Test parsing all stdlib files
  - [ ] Test merge_programs logic
  - [ ] Test duplicate neuron detection
  - [ ] Test error handling
- [ ] Add integration tests:
  - [ ] Test compiling a neuron that uses stdlib
  - [ ] Test that generated PyTorch code works
  - [ ] Test --no-stdlib flag
  - [ ] Test custom stdlib path
- [ ] Add validation tests for each stdlib neuron:
  - [ ] Create examples/stdlib_tests/ directory
  - [ ] One test file per stdlib neuron
  - [ ] Verify each neuron validates and compiles

**Acceptance Criteria**: 95%+ test coverage for stdlib loading code

### 3.2 End-to-End Validation

**Tasks**:

- [ ] Create working examples that use stdlib:
  - [ ] examples/gpt2_from_stdlib.ns (using TransformerStack)
  - [ ] examples/bert_from_stdlib.ns (using BERT neurons)
  - [ ] examples/simple_mlp.ns (using FFN)
- [ ] Compile each example to PyTorch
- [ ] Run generated PyTorch code with test inputs
- [ ] Verify output shapes match expectations

**Acceptance Criteria**: At least 3 end-to-end examples work from .ns → PyTorch → execution

## Priority 4: Performance & Scalability (Optimization)

### 4.1 Optimize Stdlib Loading

**Tasks**:

- [ ] Profile stdlib loading time
- [ ] Add caching mechanism:
  - [ ] Cache parsed stdlib in ~/.cache/neuroscript/
  - [ ] Invalidate cache when stdlib files change
  - [ ] Add --force-reload flag to bypass cache
- [ ] Parallelize stdlib file parsing (if >10 files)
- [ ] Lazy-load stdlib (only load when first neuron is used)

**Acceptance Criteria**: Stdlib loading adds <50ms to compilation time

### 4.2 Reduce Validation Overhead

**Tasks**:

- [ ] Skip validation of stdlib neurons if already validated
- [ ] Only validate stdlib neurons that are actually used
- [ ] Add --trust-stdlib flag to skip stdlib validation
- [ ] Cache validation results

**Acceptance Criteria**: Large programs with stdlib validate in <100ms

## Priority 5: Advanced Features (Nice-to-Have)

### 5.1 Stdlib Versioning

**Tasks**:

- [ ] Add version to stdlib directory (e.g., stdlib/v1/)
- [ ] Support multiple stdlib versions
- [ ] Add `use stdlib::v2` syntax for version selection
- [ ] Deprecation warnings for old stdlib APIs

**Acceptance Criteria**: Users can pin to specific stdlib version

### 5.2 Stdlib Package Manager

**Tasks**:

- [ ] Define .neuroscript package format
- [ ] Support downloading stdlib from remote sources
- [ ] Add `neuroscript install <package>` command
- [ ] Local package cache and dependency resolution

**Acceptance Criteria**: Users can share and reuse neuron libraries

### 5.3 Stdlib Precompilation

**Tasks**:

- [ ] Add --precompile-stdlib flag
- [ ] Generate optimized IR for stdlib neurons
- [ ] Ship precompiled stdlib with binary
- [ ] Validate that precompiled stdlib matches source

**Acceptance Criteria**: Zero stdlib parsing overhead in production

## Quick Wins (Can Do Now)

### Immediate Tasks (1-2 hours each)

- [x] Accept failing snapshot test: `cargo insta accept`
- [ ] Fix top 5 most-used stdlib neurons (FFN, Residual, etc.)
- [ ] Add examples/stdlib_demo.ns showing stdlib usage
- [ ] Document --no-stdlib flag in help text
- [ ] Add warning when stdlib validation fails

### Medium Tasks (half day each)

- [ ] Create primitive neuron definitions for top 10 primitives
- [ ] Add `neuroscript list-stdlib` command
- [ ] Write stdlib/README.md with usage guide
- [ ] Add 10 unit tests for stdlib.rs

### Large Tasks (1-2 days each)

- [ ] Fix all validation errors in stdlib files
- [ ] Create complete primitive definitions (30+ files)
- [ ] End-to-end testing with PyTorch execution
- [ ] Performance optimization with caching

## Metrics for "Production Ready"

- ✅ Core loading functionality works (DONE)
- ⬜ 0 validation errors in stdlib files
- ⬜ All primitive signatures defined
- ⬜ 95%+ test coverage
- ⬜ 3+ working end-to-end examples
- ⬜ Complete documentation
- ⬜ <50ms stdlib loading overhead
- ⬜ Helpful error messages for all failure modes

## Current Status Summary

**Working**:

- ✅ Stdlib discovery and loading
- ✅ Program merging
- ✅ Primitive neuron recognition
- ✅ CLI integration with --no-stdlib
- ✅ Error handling and graceful fallback

**Needs Work**:

- ⬜ Stdlib validation errors (126+ errors)
- ⬜ Primitive shape validation (dummy ports)
- ⬜ Integration test snapshots
- ⬜ User documentation
- ⬜ End-to-end examples

**Priority Order for Implementation**:

1. Fix primitive shape validation (blocks everything else)
2. Fix stdlib validation errors (makes stdlib usable)
3. Update integration tests (ensures quality)
4. Add documentation (enables users)
5. Performance optimization (scales to production)
