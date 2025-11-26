# Shape Inference – Critical Realization & Action Plan

## The Core Realization

- **Shape inference does not need to be a compile‑time problem.**  Tensors already carry their shape (`tensor.shape`).  The language only needs to expose this metadata at **runtime** and use it for dynamic dispatch.
- **Match expressions become runtime pattern‑matching on `x.shape`.**  A match arm is simply:
  1. Check if the current shape matches the pattern.
  2. Optionally evaluate a guard.
  3. Route the tensor to the appropriate sub‑graph/module.
- **Static type‑level dependent types are unnecessary.**  The difficulty lies in generating code that can *lazily* instantiate modules when a dimension is unknown, not in proving shapes ahead of time.

## Why the Original Thinking Was Wrong

| Assumption | Reality |
|------------|---------|
| Need a full dependent‑type system to prove shapes at compile time. | Shapes are runtime data; PyTorch already tracks them. |
| All shapes must be known when the model class is constructed. | Many dimensions (e.g., `d` in `[*, d]`) are only known on the first forward pass. |
| Shape inference is a hard PL research problem. | It is a tractable runtime dispatch problem plus lazy module creation. |

## What This Means for Implementation

1. **Architecture** – The **Rust compiler** parses NeuroScript and generates **Python/PyTorch** code.
2. **Pattern‑compilation** – The Rust codegen converts DSL patterns like `[*, d] where d > 512` into Python code that checks `shape[1]`.
3. **Lazy instantiation** – The generated Python code handles lazy module creation on the first forward pass.
4. **Module caching** – The generated code caches created modules.
5. **Shape probing** – The generated Python model supports a `probe()` method to trace execution.
6. **Validator** – The Rust compiler performs static graph checks (connections, port existence) before codegen.

## Action Plan (ordered steps)

- [ ] **Runtime Library**: implement `Linear`, `GELU`, `Add`, `Fork`, `Identity` in the runtime package (e.g., `neuroscript_runtime`).
- [x] **Primitive codegen**: map `impl: core,nn/Linear` → `nn.Linear(...)` (implemented in `src/codegen.rs` and `src/stdlib_registry.rs`).
- [x] **Composite codegen**: generate `__init__` and `forward` for neuron bodies (implemented in `src/codegen.rs`).
- [ ] **Tests**: parse a simple neuron (e.g., `residual.ns`), generate Python, run it.
- [ ] **Shape probing API**: `model.probe(example_input)` that traces execution and validates shapes.
- [ ] **Clear error messages** for mismatched shapes.
- [ ] Build a real model (tiny transformer or ResNet‑like), train, and demonstrate the full pipeline.

## TL;DR

- **Runtime shape tracking** replaces compile‑time inference.
- Implement **pattern matching**, **lazy module creation**, and **caching**.
- Use **validator + probing** to guarantee shape correctness before full training.
- Follow the incremental plan above to get a working code‑generator quickly.
