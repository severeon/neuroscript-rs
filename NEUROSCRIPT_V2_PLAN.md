# NeuroScript v2: Architecture Redesign Plan

**Status**: Planning Phase
**Date**: 2025-12-05
**Objective**: Design a modular, extensible neural architecture DSL with first-class neurons, multi-backend support, and efficient recursion

---

## Executive Summary

NeuroScript v2 will be a complete architectural redesign focused on:
- **Multi-backend IR**: Target PyTorch, JAX, ONNX, XLA from a single source
- **First-class neurons**: Neurons as values that can be passed, stored, and composed
- **Efficient recursion**: Native support for recursive architectures with weight sharing
- **Parser library**: Use pest/lalrpop instead of hand-written parser
- **Modular design**: All files ≤300 lines, clear separation of concerns
- **Strong type system**: Dimension types, shape inference, and neuron types from day one

---

## Part 1: Lessons from v1

### What Worked ✅
1. **Indentation-based syntax** - Clean, readable, Pythonic
2. **Shape algebra with wildcards** - Powerful pattern matching for tensors
3. **Composite neurons via graph** - Natural dataflow composition
4. **miette diagnostics** - Beautiful error messages
5. **Comprehensive test suite** - 126+ examples caught regressions

### What Needs Improvement 🔧
1. **Hand-written parser** - Maintenance burden, hard to extend grammar
2. **String-based codegen** - Brittle, no optimization, backend-locked
3. **Late shape inference** - Retrofitted instead of core to type system
4. **Monolithic files** - codegen/ modules grew too large
5. **No real recursion** - Let bindings are a workaround, not a solution
6. **Neurons not first-class** - Can't pass as parameters or return from functions
7. **Backend coupling** - PyTorch-only, hard to add new targets

---

## Part 2: Recursion in Neural Architectures

### Recursive Model Patterns

Based on research into recursive neural architectures (TreeLSTM, Universal Transformers, Neural Turing Machines):

#### Weight Sharing Strategies
1. **Static recursion** (compile-time depth):
   - Unroll fixed number of iterations
   - Each iteration shares same weights
   - Example: Universal Transformer with N layers

2. **Dynamic recursion** (runtime depth):
   - Halting condition based on input
   - Requires dynamic computation graph
   - Example: Adaptive Computation Time (ACT)

3. **Structural recursion** (tree/graph):
   - Recurse over data structure (parse tree, scene graph)
   - Different recursion depth per sample
   - Example: TreeLSTM, Graph Neural Networks

#### Samsung's Likely Approach
While I couldn't access specific papers, tiny recursive models typically use:
- **Weight tying**: Same module applied at each recursive step
- **Residual connections**: Preserve gradient flow through deep recursion
- **Dynamic halting**: Learn when to stop recursing (ACT mechanism)
- **Shared parameters**: Minimize memory footprint via weight reuse

### NeuroScript v2 Recursion Design

We should support all three patterns:

```neuroscript
# Static recursion (compile-time unrolling)
neuron RecurrentBlock(depth: Nat):
  in: [batch, seq, dim]
  out: [batch, seq, dim]
  graph:
    in -> match:
      [*] where depth > 0:
        -> SelfAttention(dim)
        -> RecurrentBlock(depth - 1)  # Recursive call
        -> out
      [*]:
        -> Identity() -> out

# Dynamic recursion (runtime halting)
neuron AdaptiveProcessor:
  in: [batch, seq, dim]
  out: [batch, seq, dim]
  param halting_threshold: Float = 0.95

  let:
    process_step = TransformerBlock(dim)
    halting_unit = Linear(dim, 1) -> Sigmoid()

  graph:
    in -> recurse(halting_threshold):
      step: process_step
      halt: halting_unit
      max_steps: 16
    -> out

# Structural recursion (over trees)
neuron TreeEncoder<T>:
  in: Tree<T>
  out: [embed_dim]
  graph:
    in -> match:
      Leaf(x): -> Embed(x) -> out
      Node(left, right):
        -> (TreeEncoder(left), TreeEncoder(right))  # Recurse on children
        -> Concat(dim=-1)
        -> Linear(embed_dim * 2, embed_dim)
        -> out
```

---

## Part 3: Multi-Backend Architecture

### Three-Stage IR Design (MLIR-inspired)

```
Source → High IR → Mid IR → Low IR → Backend Code
         (HIR)     (MIR)     (LIR)
```

#### High-Level IR (HIR)
- **Purpose**: Semantic representation close to source
- **Features**:
  - Neuron definitions with generic types
  - First-class neuron values
  - Symbolic shape algebra
  - Type variables and constraints
- **Transformations**:
  - Type inference and checking
  - Shape inference
  - Macro expansion
  - Dead code elimination

#### Mid-Level IR (MIR)
- **Purpose**: Backend-agnostic computation graph
- **Features**:
  - Static Single Assignment (SSA) form
  - Explicit tensor operations
  - Control flow as basic blocks
  - Resolved types and shapes
- **Transformations**:
  - Inlining
  - Constant folding
  - Common subexpression elimination
  - Loop fusion/tiling
  - Recursion lowering (unrolling or dynamic)

#### Low-Level IR (LIR)
- **Purpose**: Backend-specific optimized representation
- **Features**:
  - Target-specific ops (aten ops for PyTorch, XLA HLO for JAX)
  - Memory layout decisions
  - Kernel fusion annotations
  - Quantization hints
- **Backends**:
  - `pytorch_backend/`: Generates `nn.Module` + `torch.fx`
  - `jax_backend/`: Generates `flax.linen` modules
  - `onnx_backend/`: Generates ONNX graphs
  - `xla_backend/`: Generates XLA HLO (for TPU)

### Backend Trait System

```rust
// src/backend/traits.rs (~50 lines)
trait Backend {
    type IR: LowIR;
    type Output;

    fn lower(mir: &MidIR) -> Result<Self::IR>;
    fn optimize(ir: &mut Self::IR) -> Result<()>;
    fn codegen(ir: &Self::IR) -> Result<Self::Output>;
}

// src/backend/pytorch.rs (~250 lines)
struct PyTorchBackend;
impl Backend for PyTorchBackend {
    type IR = TorchFxGraph;
    type Output = String; // Python code
    // ...
}

// src/backend/jax.rs (~250 lines)
struct JaxBackend;
impl Backend for JaxBackend {
    type IR = JaxprGraph;
    type Output = String; // Python code
    // ...
}
```

---

## Part 4: Parser Library Selection

### Comparison of Rust Parser Libraries

| Library | Approach | Pros | Cons | Recommendation |
|---------|----------|------|------|----------------|
| **pest** | PEG | Declarative grammar, great errors | Runtime parsing | ⭐ Best for v2 |
| **lalrpop** | LALR | Fast, compile-time | Harder to debug | Good alternative |
| **nom** | Combinators | Flexible, zero-copy | Manual error handling | Too low-level |
| **tree-sitter** | GLR | Incremental parsing, LSP-ready | C dependency | Future (for LSP) |

### Recommendation: **pest**

**Rationale**:
1. **Declarative grammar** matches our needs (indentation, operators, pipelines)
2. **Built-in error reporting** integrates well with miette
3. **PEG parsing** handles our syntax naturally (no ambiguity)
4. **Active development** and good docs
5. **Easy migration path** from hand-written parser

**Example grammar** (`grammar.pest`):
```pest
program = { SOI ~ use_stmt* ~ neuron_def+ ~ EOI }

neuron_def = {
    "neuron" ~ ident ~ type_params? ~ params? ~ ":" ~ NEWLINE
    ~ INDENT ~ port_decls ~ neuron_body ~ DEDENT
}

pipeline = { endpoint ~ ("->" ~ endpoint)* }
endpoint = { match_expr | call_expr | tuple_expr | port_ref }

shape = { "[" ~ shape_elem ~ ("," ~ shape_elem)* ~ "]" }
shape_elem = { wildcard | dim_expr }
```

---

## Part 5: First-Class Neurons

### Type System for Neurons

```neuroscript
# Neuron types are function types over tensors
type NeuronType = Signature {
    params: Map<String, Type>,
    inputs: PortMap<Shape>,
    outputs: PortMap<Shape>,
}

# Example: Linear is a neuron type constructor
Linear: (Nat, Nat) -> Neuron<[*, in_dim] -> [*, out_dim]>

# Higher-order neuron: takes a neuron as parameter
neuron Residual<F: Neuron<[*s] -> [*s]>>(inner: F):
  in: [*s]
  out: [*s]
  graph:
    in -> (Identity(), inner) -> Add() -> out

# Usage: Residual can wrap any neuron with matching signature
let my_block = Residual(Linear(256, 256))
let nested = Residual(Residual(Linear(256, 256)))

# Map: apply neuron to list
neuron MapSeq<F: Neuron<[*s] -> [*t]>>(f: F, n: Nat):
  in: List<Tensor<[*s]>, n>
  out: List<Tensor<[*t]>, n>
  graph:
    in -> map(f) -> out

# Fold: reduce list with neuron
neuron FoldSeq<F: Neuron<([*s], [*s]) -> [*s]>>(f: F, n: Nat):
  in: List<Tensor<[*s]>, n>
  out: [*s]
  graph:
    in -> fold(f) -> out
```

### Implementation Strategy

1. **HIR**: Represent neuron types as first-class `Type::Neuron`
2. **Type checker**: Unification-based inference (Hindley-Milner style)
3. **MIR**: Monomorphization - specialize generic neurons to concrete types
4. **LIR**: Neuron values become module instances or closures

---

## Part 6: Modular File Structure

### Enforced Size Limits
- **Hard limit**: 300 lines per file
- **Soft limit**: 200 lines preferred
- **CI check**: Fail build if any file >300 lines

### Proposed Directory Layout

```
neuroscript-v2/
├── src/
│   ├── main.rs                      # CLI entry (~100 lines)
│   ├── lib.rs                       # Public API (~80 lines)
│   │
│   ├── syntax/                      # Frontend (parsing)
│   │   ├── mod.rs                   # Re-exports (~20 lines)
│   │   ├── grammar.pest             # Pest grammar (~150 lines)
│   │   ├── lexer.rs                 # Token types (~100 lines)
│   │   ├── parser.rs                # Pest parser (~250 lines)
│   │   └── span.rs                  # Source location (~50 lines)
│   │
│   ├── hir/                         # High-level IR
│   │   ├── mod.rs                   # Re-exports (~30 lines)
│   │   ├── types.rs                 # Type definitions (~200 lines)
│   │   ├── neuron.rs                # Neuron HIR (~150 lines)
│   │   ├── expr.rs                  # Expression HIR (~200 lines)
│   │   ├── shape.rs                 # Shape HIR (~150 lines)
│   │   └── visitor.rs               # HIR traversal (~100 lines)
│   │
│   ├── typeck/                      # Type checking
│   │   ├── mod.rs                   # Re-exports (~20 lines)
│   │   ├── infer.rs                 # Type inference (~250 lines)
│   │   ├── unify.rs                 # Unification (~200 lines)
│   │   ├── shape_infer.rs           # Shape inference (~250 lines)
│   │   ├── constraint.rs            # Constraint solving (~200 lines)
│   │   └── errors.rs                # Type errors (~150 lines)
│   │
│   ├── mir/                         # Mid-level IR
│   │   ├── mod.rs                   # Re-exports (~30 lines)
│   │   ├── types.rs                 # MIR types (~200 lines)
│   │   ├── ops.rs                   # Tensor operations (~250 lines)
│   │   ├── cfg.rs                   # Control flow graph (~200 lines)
│   │   ├── ssa.rs                   # SSA construction (~250 lines)
│   │   └── lower.rs                 # HIR→MIR lowering (~250 lines)
│   │
│   ├── transform/                   # IR transformations
│   │   ├── mod.rs                   # Re-exports (~30 lines)
│   │   ├── inline.rs                # Inlining pass (~200 lines)
│   │   ├── dce.rs                   # Dead code elimination (~150 lines)
│   │   ├── cse.rs                   # Common subexpression (~200 lines)
│   │   ├── fold.rs                  # Constant folding (~180 lines)
│   │   ├── fusion.rs                # Op fusion (~250 lines)
│   │   └── recursion.rs             # Recursion lowering (~250 lines)
│   │
│   ├── backend/                     # Code generation
│   │   ├── mod.rs                   # Re-exports (~40 lines)
│   │   ├── traits.rs                # Backend trait (~100 lines)
│   │   │
│   │   ├── pytorch/
│   │   │   ├── mod.rs               # PyTorch backend (~50 lines)
│   │   │   ├── lower.rs             # MIR→TorchFX (~250 lines)
│   │   │   ├── codegen.rs           # Python generation (~250 lines)
│   │   │   ├── optimize.rs          # PT optimizations (~200 lines)
│   │   │   └── runtime.rs           # Runtime helpers (~150 lines)
│   │   │
│   │   ├── jax/
│   │   │   ├── mod.rs               # JAX backend (~50 lines)
│   │   │   ├── lower.rs             # MIR→Jaxpr (~250 lines)
│   │   │   ├── codegen.rs           # Python generation (~250 lines)
│   │   │   └── runtime.rs           # Flax helpers (~150 lines)
│   │   │
│   │   └── onnx/
│   │       ├── mod.rs               # ONNX backend (~50 lines)
│   │       ├── lower.rs             # MIR→ONNX (~250 lines)
│   │       └── serialize.rs         # Protobuf serialization (~200 lines)
│   │
│   ├── stdlib/                      # Standard library
│   │   ├── mod.rs                   # Registry (~50 lines)
│   │   ├── core.ns                  # Core neurons (source)
│   │   ├── primitives.rs            # Primitive mapping (~200 lines)
│   │   └── signatures.rs            # Type signatures (~150 lines)
│   │
│   └── util/                        # Utilities
│       ├── mod.rs                   # Re-exports (~20 lines)
│       ├── errors.rs                # Error types (~150 lines)
│       ├── intern.rs                # String interning (~200 lines)
│       └── arena.rs                 # Arena allocator (~150 lines)
│
├── tests/
│   ├── integration/                 # Integration tests
│   │   ├── parser.rs                # Parser tests (~250 lines)
│   │   ├── typeck.rs                # Type checking tests (~250 lines)
│   │   ├── codegen_pytorch.rs       # PyTorch codegen (~250 lines)
│   │   ├── codegen_jax.rs           # JAX codegen (~250 lines)
│   │   └── recursion.rs             # Recursion tests (~250 lines)
│   │
│   └── snapshots/                   # Insta snapshots
│       └── *.snap
│
└── examples/
    ├── basic/                       # Basic examples
    ├── recursion/                   # Recursive architectures
    ├── stdlib/                      # Standard library examples
    └── backends/                    # Multi-backend demos
```

### File Organization Principles

1. **One concept per file**: Each file has a single clear purpose
2. **Depth over breadth**: Prefer deeper directory trees to large files
3. **Module hierarchy**: `mod.rs` only for re-exports, logic in named files
4. **Test colocation**: Unit tests in same file, integration in `tests/`
5. **Backend isolation**: Each backend is independent module

---

## Part 7: Development Phases

### Phase 0: Infrastructure (Weeks 1-2)
**Goal**: Set up project skeleton and tooling

- [ ] Create new repo: `neuroscript-v2/`
- [ ] Set up pest grammar for basic syntax
- [ ] Implement HIR types (neuron, expr, shape)
- [ ] Add miette error infrastructure
- [ ] Create file size CI check (300 line limit)
- [ ] Port 10 core examples from v1

**Deliverable**: Can parse basic neuron definitions into HIR

### Phase 1: Type System (Weeks 3-5)
**Goal**: First-class neuron types and shape inference

- [ ] Implement type inference engine (Hindley-Milner)
- [ ] Shape algebra with dimension variables
- [ ] Constraint solver for shape compatibility
- [ ] Type error reporting with miette
- [ ] Support for generic neurons `Neuron<T>`

**Deliverable**: Full type checking for composite neurons

### Phase 2: MIR and Recursion (Weeks 6-8)
**Goal**: Lower to MIR with recursion support

- [ ] Define MIR (SSA form, CFG, tensor ops)
- [ ] Implement HIR→MIR lowering
- [ ] Recursion lowering strategies:
  - [ ] Static unrolling (compile-time depth)
  - [ ] Dynamic recursion (halting condition)
  - [ ] Structural recursion (tree traversal)
- [ ] Basic optimization passes (DCE, constant folding)

**Deliverable**: Recursive neurons compile to MIR

### Phase 3: PyTorch Backend (Weeks 9-11)
**Goal**: Generate working PyTorch code from MIR

- [ ] Define PyTorch LIR (TorchFX-based)
- [ ] Implement MIR→PyTorch lowering
- [ ] Code generation (nn.Module classes)
- [ ] Runtime support for dynamic recursion
- [ ] Integration tests with PyTorch execution

**Deliverable**: v1 parity for PyTorch codegen

### Phase 4: Multi-Backend (Weeks 12-14)
**Goal**: Add JAX and ONNX backends

- [ ] JAX backend (MIR→Jaxpr→Flax)
- [ ] ONNX backend (MIR→ONNX graph)
- [ ] Backend-agnostic test suite
- [ ] Cross-backend validation (same outputs)

**Deliverable**: Can compile same source to 3 backends

### Phase 5: Advanced Features (Weeks 15-18)
**Goal**: Higher-order neurons and metaprogramming

- [ ] Higher-order neurons (neuron parameters)
- [ ] Macros and compile-time execution
- [ ] Loop constructs (`for`, `while`)
- [ ] Conditional compilation
- [ ] Module system and imports

**Deliverable**: Full language feature set

### Phase 6: Optimization (Weeks 19-21)
**Goal**: Production-quality performance

- [ ] Advanced MIR optimizations:
  - [ ] Inlining heuristics
  - [ ] Loop fusion and tiling
  - [ ] Operator fusion (backend-specific)
  - [ ] Memory layout optimization
- [ ] Benchmarking suite
- [ ] Performance comparison with v1

**Deliverable**: Optimized compilation pipeline

### Phase 7: Tooling (Weeks 22-24)
**Goal**: Developer experience

- [ ] LSP server (using tree-sitter)
- [ ] Syntax highlighting (VSCode, Vim)
- [ ] REPL with type inference display
- [ ] Documentation generator
- [ ] Package manager (for sharing neurons)

**Deliverable**: Complete development environment

---

## Part 8: Key Technical Decisions

### 1. Parser: pest (PEG)
- **Why**: Declarative, good error messages, handles indentation
- **Alternative**: lalrpop (faster but harder to debug)

### 2. Type System: Hindley-Milner + Constraints
- **Why**: Proven for functional languages, supports generics
- **Shape inference**: Separate constraint solver for dimensions

### 3. IR: Three-level (HIR/MIR/LIR)
- **Why**: Clean separation of concerns, enables multi-backend
- **Inspired by**: MLIR, Relay IR, Swift IR

### 4. Recursion: Multi-strategy
- **Static**: Unroll at compile time (simple, efficient)
- **Dynamic**: Runtime halting (flexible, complex)
- **Structural**: Over data structures (for trees/graphs)

### 5. Backend Trait: Extensible
- **Why**: Third-party backends without forking
- **Example**: Users can add TensorFlow, IREE, Mojo backends

### 6. Memory: Arena allocation
- **Why**: IR nodes are immutable, arena is fast
- **Library**: `typed-arena` or custom

### 7. String Interning: Reduce allocation
- **Why**: Symbol names repeated frequently
- **Library**: `string-interner` crate

---

## Part 9: Migration from v1

### What to Port
1. **Examples**: All 126+ `.ns` files (regression suite)
2. **Stdlib**: Core neurons (Linear, Conv, Attention, etc.)
3. **Tests**: Integration tests, snapshot tests
4. **Documentation**: Update for new syntax/features

### What to Rewrite
1. **Parser**: pest grammar (don't port hand-written parser)
2. **IR**: New HIR/MIR/LIR (don't port old IR)
3. **Codegen**: Backend-agnostic (don't port PyTorch-only)
4. **Type system**: Built-in from start (don't retrofit)

### Compatibility
- **Syntax**: Mostly backward compatible
- **Semantics**: Match v1 behavior where possible
- **New features**: Opt-in (e.g., higher-order neurons)

---

## Part 10: Success Metrics

### Functional Requirements
- ✅ Parse all v1 examples without syntax changes
- ✅ Type check with informative error messages
- ✅ Generate correct PyTorch code (match v1 output)
- ✅ Support at least 3 backends (PyTorch, JAX, ONNX)
- ✅ Handle recursive neurons efficiently

### Non-Functional Requirements
- ✅ All files ≤300 lines (enforced by CI)
- ✅ Compile 100+ neuron file in <1 second
- ✅ Type check in <100ms (for LSP responsiveness)
- ✅ Zero unsafe Rust (unless in isolated modules)
- ✅ >90% test coverage

### Developer Experience
- ✅ Error messages point to exact source location
- ✅ Type errors suggest fixes
- ✅ LSP provides instant feedback
- ✅ Documentation with examples for every feature

---

## Part 11: Risk Mitigation

### Risk 1: Scope Creep
- **Mitigation**: Strict phase boundaries, MVP first
- **Fallback**: Ship PyTorch-only v2.0, add backends in 2.1+

### Risk 2: Type System Complexity
- **Mitigation**: Start simple (no higher-order in phase 1)
- **Fallback**: Limit to first-order neurons initially

### Risk 3: Backend Compatibility
- **Mitigation**: Design MIR conservatively (lowest common denominator)
- **Fallback**: Backend-specific extensions via attributes

### Risk 4: Performance Regression
- **Mitigation**: Continuous benchmarking against v1
- **Fallback**: Optimize hot paths incrementally

### Risk 5: File Size Limits Too Strict
- **Mitigation**: 300 lines is generous; enforce early
- **Fallback**: Increase to 400 if truly necessary

---

## Part 12: Research Questions

Before starting implementation, investigate:

1. **Recursion Efficiency**: How do PyTorch/JAX handle dynamic depth?
   - Research `torch.fx` for dynamic graphs
   - Study JAX's `jax.lax.while_loop` and `jax.lax.cond`
   - Look at ONNX Loop operator

2. **Shape Inference**: What algorithm scales best?
   - Compare unification vs. constraint solving vs. SMT
   - Study TensorFlow's shape inference (C++ implementation)
   - Investigate Relay's type system

3. **Backend Abstraction**: What's the right IR level?
   - Study MLIR dialects (linalg, tensor, arith)
   - Look at TVM Relay IR
   - Review XLA HLO

4. **Parser Performance**: pest vs lalrpop benchmarks
   - Parse 1000+ line file
   - Incremental reparsing for LSP

5. **Memory Efficiency**: Arena vs. Rc vs. Box
   - Profile IR allocation patterns
   - Benchmark arena allocators

---

## Part 13: Open Questions

1. **Syntax for higher-order neurons**: How to pass neuron values?
   ```neuroscript
   # Option A: Explicit type annotation
   neuron Wrapper<F: Neuron<[*] -> [*]>>(inner: F):
     ...

   # Option B: Inferred from usage
   neuron Wrapper(inner):
     in -> inner -> out
   ```

2. **Dynamic recursion syntax**: Runtime halting condition?
   ```neuroscript
   # Option A: Built-in construct
   in -> recurse(max_depth=16, halt_fn=should_stop):
     step: process
   -> out

   # Option B: Library function
   in -> recursive_apply(process, should_stop, 16) -> out
   ```

3. **Backend selection**: Compile-time or runtime?
   ```bash
   # Option A: Compile-time flag
   neuroscript compile --backend pytorch program.ns
   neuroscript compile --backend jax program.ns

   # Option B: Multiple outputs
   neuroscript compile --all-backends program.ns
   # Generates: program.pytorch.py, program.jax.py, program.onnx
   ```

4. **Module system**: How to import neurons?
   ```neuroscript
   # Option A: Python-style
   from stdlib.attention import MultiHeadAttention

   # Option B: Rust-style
   use stdlib::attention::MultiHeadAttention;

   # Option C: Custom
   import MultiHeadAttention from "stdlib/attention.ns"
   ```

---

## Conclusion

NeuroScript v2 will be a **production-grade neural architecture DSL** with:
- ✅ Clean, modular codebase (300 line file limit)
- ✅ Multi-backend compilation (PyTorch, JAX, ONNX, extensible)
- ✅ First-class neurons (higher-order composition)
- ✅ Efficient recursion (static, dynamic, structural)
- ✅ Strong type system (shape inference, dimension checking)
- ✅ Excellent developer experience (LSP, REPL, great errors)

**Next steps**:
1. Review this plan with team
2. Answer open questions (syntax decisions)
3. Prototype pest grammar
4. Set up v2 repo skeleton
5. Begin Phase 0 implementation

**Estimated timeline**: 24 weeks to full feature parity + advanced features
**MVP (PyTorch-only)**: 11 weeks
**Multi-backend**: 14 weeks
**Production-ready**: 21 weeks
