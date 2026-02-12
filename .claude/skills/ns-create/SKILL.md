---
name: ns-create
description: Create NeuroScript neurons. Covers primitives (impl references), composites (graph connections), residual (Fork/Add), match-based (shape routing), and recursive (@lazy) neurons.
context: fork
agent: general-purpose
disable-model-invocation: true
---

# NeuroScript Neuron Creator

## Available Primitives (live — check before creating duplicates)

!`grep -A1 'self\.register(' src/stdlib_registry.rs | grep '"' | sed 's/.*"\([^"]*\)".*/\1/' | sort`

## Workflow

Given: `$ARGUMENTS` — description of the neuron to create.

### 1. Understand Requirements

From the description, determine:
- **What** the neuron does (transform, route, compose, etc.)
- **Shape signature** — input/output tensor shapes
- **Parameters** — what's configurable
- **Dependencies** — which existing primitives/neurons it uses

### 2. Check Existing Neurons

Before creating, verify the neuron doesn't already exist:
- Check the primitives list above
- Check `stdlib/*.ns` for composite library neurons
- Check `examples/` for similar patterns

**Reuse stdlib composites**: If the neuron you're building contains a sub-pattern that already exists in `stdlib/`, reference it by name instead of inlining the logic. All stdlib neurons (primitives and composites) are automatically loaded and available by name in any `.ns` file. For example, use `SEBlock(channels, reduction)` instead of inlining the squeeze-and-excitation pattern. See `examples/stdlib_composition.ns` for examples of composing stdlib neurons.

### 3. Choose Pattern

| Pattern | When to Use | Template |
|---------|------------|----------|
| **Primitive** | Wrapping external Python/PyTorch code | `templates/primitive.ns` |
| **Composite** | Connecting existing neurons in a pipeline | `templates/composite.ns` |
| **Residual** | Skip connection (Fork → process → Add) | `templates/residual.ns` |
| **Match-based** | Shape-dependent routing | `templates/match-pattern.ns` |
| **Recursive** | Repeated/stacked layers with depth param | `templates/recursive.ns` |

### 4. Implement

Use the appropriate template from `templates/`. Key rules:

**Shapes**:
- Only one variadic per shape (`[*shape, dim]` OK, `[*a, *b]` invalid)
- Use named dims for polymorphism: `[*, dim]` not `[*, 512]`
- Expression dims: `[dim * 4]`, `[d_model / heads]` (division must be exact)

**Structural**:
- Implicit fork (preferred): `in -> (a, b, c)` — any single output → N-way tuple
- Explicit Fork/Fork3/ForkN only needed for named port access
- Add/Multiply = 2 inputs (main, skip). For 3+ inputs, chain: `Add() -> Add()`
- Concat takes 2 inputs — chain for 3+

**Impl references**:
- Format: `core,<category>/<ClassName>` — NOT dot notation
- Example: `core,nn/Linear`, `core,attention/ScaledDotProductAttention`

**Context bindings**:
- `@lazy` for match-captured dims or recursion
- `@static` for eager init from parameters only
- `@global` for shared singletons

**Match expressions**:
- Last arm must be catch-all (no guard)
- Order: most specific → most general
- Captured dims usable in guards and call args

### 5. Validate

Run the validation script:

```bash
./scripts/validate_neuron.sh <file.ns>
```

Or manually:
```bash
./target/release/neuroscript parse <file.ns>
./target/release/neuroscript validate <file.ns>
./target/release/neuroscript compile <file.ns> --neuron <NeuronName>
```

All three stages must pass before the neuron is complete.

### 6. Common Pitfalls

See `references/common-pitfalls.md` for detailed gotchas. Quick summary:

- **Concat arity**: Concat takes exactly 2 inputs via named ports, not inline tuple
- **Match exhaustiveness**: Always end with a catch-all pattern
- **Impl format**: `core,nn/Linear` not `core.nn.Linear`
- **Shape solver limits**: Only solves single-unknown linear equations
- **Prefer implicit fork**: `in -> (a, b, c)` over explicit `Fork()`/`Fork3()` — no limit on outputs
- **Recursive base case**: `@lazy` recursion needs `where depth > 0` guard + Identity fallback

## Output Location

Place created neurons in:
- `examples/` — for standalone examples and tests
- `stdlib/` — for reusable library neurons
- Project `src/` — for package-specific neurons
