# NeuroScript Language Gaps

Issues discovered during batch neuron generation (25 neurons created). Ordered by impact and fix difficulty.

## 1. Add Subtract and Divide primitives

**Impact**: High | **Difficulty**: Easy

Only `Add` and `Multiply` exist as element-wise binary operations. There is no `Subtract` or `Divide`. This prevents expressing patterns like `1 - gate` (used in true highway connections) or normalized attention weights.

**What needs to change**:
- Register `Subtract` and `Divide` in `src/stdlib_registry.rs` with the same pattern as `Add`/`Multiply`
- Add Python runtime implementations in `neuroscript_runtime/primitives/`
- Both take 2 inputs via tuple: `(a, b) -> Subtract() -> result`

**Neurons blocked**: HighwayConnection (currently uses Add as approximation), any gating mechanism needing complement operations.

## 2. Doc comment blank line parse failure

**Impact**: Medium | **Difficulty**: Easy

A blank line between a `///` doc comment block and the `neuron` keyword causes a parse error. The grammar rule `doc_block = { (DOC_COMMENT ~ NEWLINE)+ }` treats the blank line as terminating the doc block, then fails to find `neuron` immediately after.

**What needs to change**:
- Adjust the grammar in `src/grammar/neuroscript.pest` to allow optional blank lines between `doc_block` and `neuron_def`
- Alternatively, make `neuron_def` tolerate whitespace between the doc block and the keyword

**Current workaround**: Don't leave blank lines between doc comments and the `neuron` keyword. The GeGLU and GatedFFN agents both hit this and had to remove the blank line.

## 3. Concat limited to 2 inputs

**Impact**: Medium | **Difficulty**: Medium-Hard

Concat is defined with exactly 2 named input ports (`in a`, `in b`). The Python runtime already supports variadic inputs (`Tuple[torch.Tensor, ...]`), but the NeuroScript grammar has no way to declare variadic input ports.

**What needs to change**:
- Option A: Add a variadic input port syntax to the grammar (e.g., `in *inputs: [*shape]`)
- Option B: Register `Concat3`, `Concat4` variants (simple but inelegant)
- Option C: Allow Concat to accept N-element tuples by special-casing it in the validator

**Current workaround**: Chain Concat calls — `(a, b) -> Concat(1) -> ab` then `(ab, c) -> Concat(1) -> abc`.

**Neurons affected**: InceptionBlock (3+ branches), DenseNet variants, any multi-branch architecture.

## 4. No loop/repeat construct

**Impact**: High | **Difficulty**: Hard

There is no way to repeat a block N times. TransformerStack and similar layered architectures must use `@lazy` recursive bindings with a depth parameter and match-based base case. This works but is fragile and verbose.

**What needs to change**:
- New IR node for loop/repeat
- Grammar rule: something like `repeat(N): block` or `stack(depth): NeuronCall()`
- Validator support for loop unrolling or dynamic depth
- Codegen to emit `nn.ModuleList` + loop in PyTorch

**Current workaround**: `@lazy` recursion:
```neuroscript
context:
    @lazy recurse = TransformerStack(dim, heads, d_ff, depth - 1)
graph:
    in -> match:
        [*] where depth > 0: Block() -> recurse -> out
        [*]: Identity() -> out
```

**Already on roadmap**: Phase 3 ("Loop constructs for repeated layers").

## 5. `context` is a reserved keyword

**Impact**: Low | **Difficulty**: Easy-Medium

The word `context` cannot be used as a port name or identifier because it's reserved for the `context:` binding block. This is a minor ergonomic issue — CrossAttention had to rename its encoder input port from `context` to `memory`.

**What needs to change**:
- Option A: Allow `context` as an identifier in port-name position (grammar context-sensitivity)
- Option B: Rename the binding block keyword (breaking change, not recommended)
- Option C: Accept as a known limitation and document it

**Neurons affected**: CrossAttention (worked around by using `memory`).

## 6. Inter-neuron stdlib references need `use` statements

**Impact**: Low | **Difficulty**: Already implemented

This turned out to not be a limitation — `use` statements work. However, none of the generated neurons currently use `use` statements to reference other stdlib neurons. MBConvBlock inlined the SE pattern instead of importing SEBlock.

**What needs to change**:
- No language change needed
- Update the ns-create skill to recommend `use stdlib, SEBlock/*` when a dependency exists in stdlib
- Verify that `use` + compilation works end-to-end for stdlib-to-stdlib references
