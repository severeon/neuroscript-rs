# NeuroScript v2: Executive Summary

## TL;DR

NeuroScript v2 will be a **modular, multi-backend neural architecture DSL** that learns from v1's successes and addresses its limitations.

### Key Changes from v1

| Aspect | v1 | v2 |
|--------|----|----|
| **Parser** | Hand-written | pest (PEG grammar) |
| **Backend** | PyTorch only | Multi-backend (PyTorch, JAX, ONNX, extensible) |
| **IR** | Single-level | Three-level (HIR/MIR/LIR) |
| **Recursion** | Limited (let bindings) | First-class (static, dynamic, structural) |
| **Neurons** | Values only | First-class types (higher-order) |
| **File size** | Some >500 lines | Hard limit: 300 lines |
| **Type system** | Retrofitted | Built-in from start |
| **Codegen** | String-based | IR-based with optimizations |

---

## Architecture Overview

```
┌─────────────┐
│  Source.ns  │
└──────┬──────┘
       │ pest parser
       ▼
┌─────────────────────┐
│  High-Level IR      │  ← Type checking, shape inference
│  (HIR)              │  ← Neuron types, generics, symbolic shapes
└──────┬──────────────┘
       │ Lowering + Monomorphization
       ▼
┌─────────────────────┐
│  Mid-Level IR       │  ← Optimizations (inline, DCE, CSE, fusion)
│  (MIR)              │  ← SSA form, control flow, tensor ops
└──────┬──────────────┘
       │ Backend selection
       ▼
    ┌──┴──┐
    │     │
    ▼     ▼
┌────────┐  ┌────────┐  ┌────────┐
│PyTorch │  │  JAX   │  │  ONNX  │
│ (LIR)  │  │ (LIR)  │  │ (LIR)  │
└───┬────┘  └───┬────┘  └───┬────┘
    │           │            │
    ▼           ▼            ▼
  .py         .py         .onnx
```

---

## Three Questions Answered

### 1. How to handle recursion efficiently?

**Strategy**: Multi-level support matching use cases

- **Static recursion** (Phase 1 - MVP):
  - Compile-time unrolling with weight sharing
  - Covers 80% of use cases (ResNet, Transformer stacks)
  - Simple to implement, efficient to execute

- **Dynamic recursion** (Phase 2):
  - Runtime halting with learned stop condition
  - For adaptive models (ACT, Neural Turing Machines)
  - Requires accumulation logic and dynamic graphs

- **Structural recursion** (Phase 3):
  - Recurse over data structures (trees, graphs)
  - For TreeLSTM, Graph Neural Networks
  - Needs batching optimizations

**Key insight**: Weight sharing is what makes recursive models "tiny" - same parameters reused at each step.

### 2. How to make neurons first-class?

**Strategy**: Neuron types as first-class types

```neuroscript
# Neurons have function types
Linear: (in: Nat, out: Nat) -> Neuron<[*, in] -> [*, out]>

# Higher-order neurons take neurons as parameters
neuron Residual<F: Neuron<[*s] -> [*s]>>(inner: F):
  in: [*s]
  out: [*s]
  graph:
    in -> (Identity(), inner) -> Add() -> out

# Can be partially applied, composed, mapped
let block = Residual(Linear(256, 256))
let stack = Compose([block, block, block])
```

**Implementation**:
1. **HIR**: Represent neuron types (`Type::Neuron`)
2. **Type checker**: Hindley-Milner inference with neuron types
3. **MIR**: Monomorphization (specialize generics to concrete types)
4. **Backend**: Module instances or closures

### 3. How to stay modular with 300-line limit?

**Strategy**: Deep directory tree, single-purpose files

**Example**: Codegen split into small modules
```
backend/
├── traits.rs         # Backend trait (~100 lines)
├── pytorch/
│   ├── mod.rs        # Exports (~50 lines)
│   ├── lower.rs      # MIR→TorchFX (~250 lines)
│   ├── codegen.rs    # Code generation (~250 lines)
│   ├── optimize.rs   # PT optimizations (~200 lines)
│   └── runtime.rs    # Runtime helpers (~150 lines)
├── jax/
│   ├── mod.rs        # Exports (~50 lines)
│   ├── lower.rs      # MIR→Jaxpr (~250 lines)
│   └── codegen.rs    # Code generation (~250 lines)
└── onnx/
    ├── mod.rs        # Exports (~50 lines)
    └── lower.rs      # MIR→ONNX (~250 lines)
```

**Enforced by CI**: Build fails if any file >300 lines

---

## Why pest over hand-written parser?

| Criterion | Hand-written | pest |
|-----------|--------------|------|
| **Maintainability** | ❌ Hard to modify | ✅ Declarative grammar |
| **Error messages** | ❌ Manual | ✅ Built-in |
| **Grammar clarity** | ❌ Implicit | ✅ Explicit |
| **Performance** | ✅ Fast | ⚠️ Slightly slower |
| **Indentation** | ⚠️ Manual tracking | ✅ Native support |
| **Testing** | ❌ Test parser code | ✅ Test grammar |

**Verdict**: pest wins on maintainability, which is critical for v2 longevity

**Example grammar**:
```pest
// grammar.pest
neuron_def = {
    "neuron" ~ ident ~ type_params? ~ params? ~ ":" ~ NEWLINE
    ~ INDENT ~ port_decls ~ neuron_body ~ DEDENT
}

pipeline = { endpoint ~ ("->" ~ endpoint)* }
```

---

## Development Timeline

### MVP (11 weeks): PyTorch-only with static recursion
- ✅ pest parser
- ✅ HIR with neuron types
- ✅ Type inference + shape checking
- ✅ MIR with SSA form
- ✅ PyTorch backend
- ✅ Static recursion (unrolling)
- ✅ Basic optimizations (DCE, inlining)

### Multi-Backend (14 weeks): Add JAX and ONNX
- ✅ Backend trait system
- ✅ JAX backend (Flax modules)
- ✅ ONNX backend (graph export)
- ✅ Cross-backend validation tests

### Full Features (21 weeks): Higher-order + dynamic recursion
- ✅ First-class neuron values
- ✅ Higher-order neurons
- ✅ Dynamic recursion (ACT)
- ✅ Loop constructs
- ✅ Advanced optimizations

### Production (24 weeks): Tooling + polish
- ✅ LSP server
- ✅ REPL
- ✅ Documentation generator
- ✅ Package manager

---

## Migration Path from v1

### Keep
- ✅ All 126+ example files (syntax mostly compatible)
- ✅ Test suite structure
- ✅ Standard library neurons
- ✅ Python runtime package

### Rewrite
- ❌ Parser (use pest)
- ❌ IR (new three-level design)
- ❌ Validator (integrated with type checker)
- ❌ Codegen (backend-agnostic)

### Extend
- 🔧 Syntax (add higher-order neurons, loop constructs)
- 🔧 Semantics (recursion, first-class values)
- 🔧 Type system (neuron types, constraints)

---

## Open Design Questions

Before starting implementation, decide:

1. **Syntax for higher-order neurons**:
   ```neuroscript
   # Option A: Explicit bounds
   neuron Wrapper<F: Neuron<[*] -> [*]>>(inner: F): ...

   # Option B: Inferred
   neuron Wrapper(inner): ...
   ```

2. **Dynamic recursion syntax**:
   ```neuroscript
   # Option A: Built-in construct
   in -> loop.adaptive(halt_fn, threshold): step: process

   # Option B: Library function
   in -> adaptive_loop(process, halt_fn, threshold) -> out
   ```

3. **Backend selection**:
   ```bash
   # Option A: Compile-time flag
   neuroscript compile --backend pytorch program.ns

   # Option B: All at once
   neuroscript compile --all-backends program.ns
   ```

4. **Module imports**:
   ```neuroscript
   # Option A: Python-style
   from stdlib.attention import MultiHeadAttention

   # Option B: Rust-style
   use stdlib::attention::MultiHeadAttention;
   ```

---

## Risk Mitigation

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Scope creep | Strict phases, MVP first | Ship PyTorch-only v2.0 |
| Type system complexity | Start simple, add features incrementally | Limit to first-order initially |
| Backend compatibility | Conservative MIR design | Backend-specific extensions |
| Performance regression | Continuous benchmarking vs. v1 | Optimize hot paths |
| 300-line limit too strict | Enforce early, adjust if needed | Increase to 400 if necessary |

---

## Success Metrics

### Must-Have (MVP)
- ✅ Parse all v1 examples
- ✅ Type check with clear errors
- ✅ Generate correct PyTorch code
- ✅ All files ≤300 lines
- ✅ Compile time <1s for 100-neuron file

### Should-Have (Multi-Backend)
- ✅ 3+ backends working
- ✅ Same semantics across backends
- ✅ Efficient recursion support

### Nice-to-Have (Production)
- ✅ LSP with <100ms latency
- ✅ REPL for experimentation
- ✅ >90% test coverage
- ✅ Documentation for every feature

---

## Next Steps

1. **Review this plan** - Discuss open questions with team
2. **Prototype pest grammar** - Validate syntax can be parsed
3. **Design HIR types** - Sketch out type system
4. **Set up v2 repo** - Start with clean slate
5. **Begin Phase 0** - Infrastructure and pest parser

---

## Key Takeaways

1. **Three-level IR** enables clean backend separation
2. **pest parser** reduces maintenance burden
3. **Static recursion first** covers most use cases simply
4. **First-class neurons** unlock higher-order composition
5. **300-line files** force good architecture
6. **Ship incrementally** - MVP → Multi-backend → Full features

**NeuroScript v2 will be production-ready, extensible, and maintainable for years to come.**

---

## Additional Resources

- **Full plan**: `NEUROSCRIPT_V2_PLAN.md` (detailed phases and architecture)
- **Recursion research**: `RECURSION_RESEARCH.md` (deep dive on recursive models)
- **v1 codebase**: Current implementation for reference

**Questions?** Open an issue or discuss in team meeting.
