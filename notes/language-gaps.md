# NeuroScript Language Gaps

Issues discovered during batch neuron generation (25 neurons created). Ordered by impact and fix difficulty.

## 1. ~~Add Subtract and Divide primitives~~ (RESOLVED)

**Impact**: High | **Difficulty**: Easy | **Status**: Done

`Subtract` and `Divide` are now registered in `src/stdlib_registry.rs` with Python runtime implementations in `neuroscript_runtime/primitives/structural.py`. Both take 2 inputs via tuple, matching `Add`/`Multiply`. HighwayConnection updated to use the true `(1-gate)` formula.

## 2. ~~Doc comment blank line parse failure~~

**Impact**: Medium | **Difficulty**: Easy

A blank line between a `///` doc comment block and the `neuron` keyword causes a parse error. The grammar rule `doc_block = { (DOC_COMMENT ~ NEWLINE)+ }` treats the blank line as terminating the doc block, then fails to find `neuron` immediately after.

**What needs to change**:

- Adjust the grammar in `src/grammar/neuroscript.pest` to allow optional blank lines between `doc_block` and `neuron_def`
- Alternatively, make `neuron_def` tolerate whitespace between the doc block and the keyword

**Current workaround**: Don't leave blank lines between doc comments and the `neuron` keyword. The GeGLU and GatedFFN agents both hit this and had to remove the blank line.

## 3. ~~Concat limited to 2 inputs~~ (RESOLVED)

**Impact**: Medium | **Difficulty**: Medium-Hard | **Status**: Done

Added variadic input ports as a general language feature: `in *inputs: [*shape]`. Changes span grammar, AST builder, IR (`Port.variadic` field), validator (variadic arity bypass), shape inference (per-element validation against variadic port shape), and codegen (single tuple parameter for forward signature). Concat in stdlib and examples updated to use variadic syntax. Neurons like InceptionBlock can now do `(a, b, c, d) -> Concat(1)` directly.

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

## ~~6. Inter-neuron stdlib references need `use` statements~~ (RESOLVED)

**Impact**: Low | **Difficulty**: Already implemented | **Status**: Done

Stdlib neurons can reference other stdlib neurons directly by name — all `.ns` files in `stdlib/` are loaded and merged automatically. No `use` statement is needed for stdlib-to-stdlib references. MBConvBlock has been refactored to use `SEBlock(channels * expansion, reduction)` instead of inlining the SE pattern. The ns-create skill now recommends reusing stdlib composites. See `examples/stdlib_composition.ns` for a demonstration of composing stdlib neurons (PreNormFFNBlock uses FFN; EfficientStage uses MBConvBlock which uses SEBlock).
