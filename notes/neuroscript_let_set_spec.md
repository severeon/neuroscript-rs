# NeuroScript Feature Spec: `let`/`set` Bindings and Structural Recursion

**Status:** Proposed for v0.4  
**Author:** Thomas + Claude  
**Date:** 2025-11-27  
**Prerequisites:** Core language (v0.1), Codegen (v0.2), Shape inference (v0.3)

---

## Motivation

NeuroScript currently lacks two capabilities essential for expressing modern neural architectures:

1. **Named instance reuse (weight sharing):** No way to apply the same neuron instance multiple times with shared weights. Each call in a graph creates an independent instance.

2. **Parameterized depth:** No way to define a neuron whose structure depends on a depth/count parameter without manually writing N identical lines.

These limitations prevent clean expression of:
- Universal Transformers (weight-shared layers)
- Tiny Recursive Models (TRM) with iterative refinement
- Any architecture where layer count is a parameter
- Transfer learning with frozen pretrained components

---

## Design Goals

1. **No dedicated loop syntax** — recursion via self-reference, terminated by guards
2. **Explicit weight sharing** — clear distinction between shared and independent instances
3. **Lazy vs eager instantiation** — necessary for recursive definitions to terminate
4. **Compositional** — works with existing features (match, guards, pipelines)
5. **Compile-time resolution** — no runtime recursion, unrolls to flat structure

---

## Syntax

### Grammar Additions

```ebnf
body ::= (port | impl_stmt | let_stmt | set_stmt | graph_stmt)+

let_stmt ::= "let" ":" NEWLINE INDENT let_binding+ DEDENT

set_stmt ::= "set" ":" NEWLINE INDENT set_binding+ DEDENT

let_binding ::= identifier "=" neuron_expr NEWLINE

set_binding ::= identifier "=" neuron_expr NEWLINE

neuron_expr ::= identifier "(" call_args ")"
              | "Freeze" "(" neuron_expr ")"
```

### Keywords Added

```
let, set
```

Note: `let` is already reserved in v0.1 spec.

---

## Semantics

### `set:` Block — Eager Instantiation

Bindings in `set:` are instantiated when the containing neuron is instantiated. Use for components that are always needed.

```neuroscript
neuron MyNetwork(dim):
  in: [*, dim]
  out: [*, dim]
  set:
    norm = LayerNorm(dim)
    ffn = FFN(dim, dim * 4)
  graph:
    in -> norm -> ffn -> out
```

**Codegen (PyTorch):**

```python
class MyNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)      # Instantiated in __init__
        self.ffn = FFN(dim, dim * 4)
    
    def forward(self, x):
        return self.ffn(self.norm(x))
```

### `let:` Block — Lazy Instantiation

Bindings in `let:` are instantiated only if referenced in an active code path. Essential for recursive definitions where some paths don't use the binding.

```neuroscript
neuron ConditionalNetwork(dim, use_attention):
  in: [*, dim]
  out: [*, dim]
  let:
    attn = MultiHeadAttention(dim, 8)  # Only created if use_attention true
  set:
    ffn = FFN(dim, dim * 4)            # Always created
  graph:
    in -> match:
      [*, dim] where use_attention: attn -> ffn -> out
      [*, dim]: ffn -> out
```

**Codegen:** Compiler analyzes which code paths are selected (based on compile-time-known guards) and only emits instantiation for reachable bindings.

### Instance Reuse (Weight Sharing)

A bound name references the same instance wherever it appears:

```neuroscript
neuron WeightSharedStack(dim):
  in: [*, dim]
  out: [*, dim]
  set:
    block = TransformerBlock(dim, 8, dim * 4)  # ONE instance
  graph:
    in ->
      block    # Same weights
      block    # Same weights  
      block    # Same weights
      out
```

**Codegen:**

```python
class WeightSharedStack(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = TransformerBlock(dim, 8, dim * 4)
    
    def forward(self, x):
        x = self.block(x)
        x = self.block(x)  # Same self.block
        x = self.block(x)  # Same self.block
        return x
```

### Contrast: Inline Calls (Independent Weights)

Inline calls without binding create independent instances:

```neuroscript
neuron IndependentStack(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in ->
      TransformerBlock(dim, 8, dim * 4)  # Instance 1
      TransformerBlock(dim, 8, dim * 4)  # Instance 2 (different weights)
      TransformerBlock(dim, 8, dim * 4)  # Instance 3 (different weights)
      out
```

---

## Structural Recursion

### Self-Referential Neurons

A neuron may reference itself in a `let:` binding. Combined with match guards, this enables compile-time recursive expansion:

```neuroscript
neuron RecursiveStack(d_model, num_heads, d_ff, depth):
  in: [*, seq, d_model]
  out: [*, seq, d_model]
  let:
    deeper = RecursiveStack(d_model, num_heads, d_ff, depth - 1)
  graph:
    in -> match:
      [*, seq, d_model] where depth > 0:
        TransformerBlock(d_model, num_heads, d_ff)
        deeper
        out
      [*, seq, d_model]:
        Identity()
        out
```

### Expansion Rules

When instantiating `RecursiveStack(768, 12, 3072, 3)`:

1. **Evaluate guard:** `depth > 0` → `3 > 0` → true
2. **Select arm:** First match arm (with `TransformerBlock` and `deeper`)
3. **Instantiate lazy binding:** `deeper = RecursiveStack(768, 12, 3072, 2)`
4. **Recurse:** Repeat for depth=2, depth=1, depth=0
5. **Base case:** At depth=0, guard fails, select `Identity()` arm
6. **Unroll complete:** Flat chain of 3 TransformerBlocks + Identity

**Resulting structure:**

```
RecursiveStack(depth=3)
  ├─ TransformerBlock(768, 12, 3072)  [instance 1]
  └─ RecursiveStack(depth=2)
       ├─ TransformerBlock(768, 12, 3072)  [instance 2]
       └─ RecursiveStack(depth=1)
            ├─ TransformerBlock(768, 12, 3072)  [instance 3]
            └─ RecursiveStack(depth=0)
                 └─ Identity()
```

**Codegen (flattened):**

```python
class RecursiveStack(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, depth):
        super().__init__()
        if depth > 0:
            self.block = TransformerBlock(d_model, num_heads, d_ff)
            self.deeper = RecursiveStack(d_model, num_heads, d_ff, depth - 1)
            self._use_deeper = True
        else:
            self._use_deeper = False
    
    def forward(self, x):
        if self._use_deeper:
            return self.deeper(self.block(x))
        else:
            return x
```

Or fully unrolled (optimization):

```python
class RecursiveStack_depth3(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.block_0 = TransformerBlock(d_model, num_heads, d_ff)
        self.block_1 = TransformerBlock(d_model, num_heads, d_ff)
        self.block_2 = TransformerBlock(d_model, num_heads, d_ff)
    
    def forward(self, x):
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        return x
```

### Weight Sharing Across Recursion Levels

To share weights across all recursion levels, pass the neuron instance as a parameter:

```neuroscript
neuron SharedRecursiveStack(block, depth):
  in: [*, seq, d_model]
  out: [*, seq, d_model]
  let:
    deeper = SharedRecursiveStack(block, depth - 1)
  graph:
    in -> match:
      [*, seq, d_model] where depth > 0:
        block
        deeper
        out
      [*, seq, d_model]:
        Identity()
        out

neuron UniversalTransformer(d_model, num_heads, d_ff, depth):
  in: [*, seq, d_model]
  out: [*, seq, d_model]
  set:
    block = TransformerBlock(d_model, num_heads, d_ff)
  graph:
    in -> SharedRecursiveStack(block, depth) -> out
```

**Note:** This requires neurons-as-parameters (see Future Work).

---

## `Freeze()` Meta-Neuron

For transfer learning, wrap a neuron to disable gradient updates:

```neuroscript
neuron FineTuned(d_model):
  in: [*, seq, d_model]
  out: [*, seq, d_model]
  set:
    backbone = Freeze(PretrainedEncoder(d_model))
    head = Linear(d_model, d_model)
  graph:
    in -> backbone -> head -> out
```

**Codegen:**

```python
self.backbone = PretrainedEncoder(d_model)
for param in self.backbone.parameters():
    param.requires_grad = False
self.head = Linear(d_model, d_model)
```

`Freeze` is a stdlib meta-neuron, not syntax. It wraps any neuron and sets `requires_grad=False` on all parameters.

---

## Validation Rules

### 1. Recursion Must Terminate

The compiler must prove that recursive `let:` bindings eventually reach a base case.

**Valid:** Guard depends on parameter that decreases toward base case
```neuroscript
let:
  deeper = Self(depth - 1)  # depth decreases
graph:
  in -> match:
    [...] where depth > 0: deeper -> out  # eventually false
    [...]: Identity() -> out               # base case exists
```

**Invalid:** No base case
```neuroscript
let:
  deeper = Self(depth - 1)
graph:
  in -> deeper -> out  # No termination condition!
```

**Invalid:** Non-terminating guard
```neuroscript
let:
  deeper = Self(depth + 1)  # depth increases!
graph:
  in -> match:
    [...] where depth > 0: deeper -> out  # Always true, infinite recursion
    [...]: Identity() -> out
```

### 2. Guards Must Be Compile-Time Evaluable

Recursive expansion happens at compile time. Guards must depend only on:
- Literal values
- Neuron parameters
- Arithmetic on the above

**Valid:**
```neuroscript
where depth > 0
where dim == 512
where num_layers >= 1
```

**Invalid:**
```neuroscript
where input_shape[0] > 32  # Runtime value
where is_training           # Runtime state
```

### 3. Lazy Bindings Cannot Be Used Eagerly

A `let:` binding cannot be referenced in a `set:` binding (would force eager evaluation of lazy binding).

### 4. No Forward References

Bindings can only reference previously defined bindings or parameters:

```neuroscript
set:
  a = Linear(dim, dim)
  b = Sequential(a, a)  # OK: a is defined above
  c = Sequential(d, a)  # ERROR: d not yet defined
  d = Linear(dim, dim)
```

---

## Interaction with Existing Features

### With `match` Guards

Guards control which arm is selected, which determines which lazy bindings are instantiated:

```neuroscript
neuron AdaptiveDepth(dim, depth):
  in: [*, dim]
  out: [*, dim]
  let:
    shallow = ShallowNet(dim)
    deep = DeepNet(dim)
  graph:
    in -> match:
      [*, dim] where depth <= 2: shallow -> out
      [*, dim]: deep -> out
```

Only one of `shallow` or `deep` is instantiated based on compile-time `depth` value.

### With Multi-Port Neurons

Bindings work with any neuron, including multi-port:

```neuroscript
neuron MultiPathNetwork(dim):
  in: [*, dim]
  out: [*, dim]
  set:
    split = Fork()
    merge = Add()
  graph:
    in -> split -> (a, b)
    a -> Linear(dim, dim) -> a_out
    b -> Linear(dim, dim) -> b_out
    (a_out, b_out) -> merge -> out
```

### With Pipelines

Bound names integrate seamlessly into pipelines:

```neuroscript
set:
  norm = LayerNorm(dim)
  ffn = FFN(dim)
graph:
  in -> norm -> ffn -> norm -> out  # norm used twice (same instance)
```

---

## Examples

### Example 1: GPT-2 with Parameterized Depth

```neuroscript
neuron GPTStack(d_model, num_heads, d_ff, num_layers):
  in: [*, seq, d_model]
  out: [*, seq, d_model]
  let:
    rest = GPTStack(d_model, num_heads, d_ff, num_layers - 1)
  graph:
    in -> match:
      [*, seq, d_model] where num_layers > 0:
        TransformerBlock(d_model, num_heads, d_ff)
        rest
        out
      [*, seq, d_model]:
        Identity()
        out

neuron GPT2(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq):
  in: [batch, seq]
  out: [batch, seq, vocab_size]
  graph:
    in ->
      Embedding(vocab_size, d_model)
      PositionalEncoding(d_model, max_len=max_seq)
      GPTStack(d_model, num_heads, d_ff, num_layers)
      LayerNorm(d_model)
      Linear(d_model, vocab_size)
      out

# Instantiate specific models
# GPT2(50257, 768, 12, 3072, 12, 1024)   -> GPT-2 Small
# GPT2(50257, 1024, 16, 4096, 24, 1024)  -> GPT-2 Medium
# GPT2(50257, 1280, 20, 5120, 36, 1024)  -> GPT-2 Large
```

### Example 2: Universal Transformer (Weight-Shared)

```neuroscript
neuron RecurseShared(block, depth):
  in: [*, seq, d_model]
  out: [*, seq, d_model]
  let:
    deeper = RecurseShared(block, depth - 1)
  graph:
    in -> match:
      [*, seq, d_model] where depth > 0:
        block
        deeper
        out
      [*, seq, d_model]:
        Identity()
        out

neuron UniversalTransformer(d_model, num_heads, d_ff, depth):
  in: [*, seq, d_model]
  out: [*, seq, d_model]
  set:
    block = TransformerBlock(d_model, num_heads, d_ff)
  graph:
    in -> RecurseShared(block, depth) -> out
```

### Example 3: Frozen Backbone with Trainable Head

```neuroscript
neuron ImageClassifier(num_classes):
  in: [batch, 3, 224, 224]
  out: [batch, num_classes]
  set:
    backbone = Freeze(ResNet50Pretrained())
    head = Linear(2048, num_classes)
  graph:
    in -> backbone -> Flatten() -> head -> out
```

### Example 4: Tiny Recursive Model (TRM)

```neuroscript
neuron ReasoningStep(d_model):
  in x: [*, seq, d_model]
  in y: [*, seq, d_model]
  in z: [*, seq, d_model]
  out: [*, seq, d_model]
  graph:
    (x, y, z) -> Concat(dim=-1) -> combined
    combined ->
      Linear(d_model * 3, d_model)
      TRMBlock(d_model)
      out

neuron AnswerStep(d_model):
  in y: [*, seq, d_model]
  in z: [*, seq, d_model]
  out: [*, seq, d_model]
  graph:
    (y, z) -> Concat(dim=-1) -> combined
    combined ->
      Linear(d_model * 2, d_model)
      TRMBlock(d_model)
      out

neuron RecursionCycle(d_model):
  in x: [*, seq, d_model]
  in y: [*, seq, d_model]
  in z: [*, seq, d_model]
  out y_out: [*, seq, d_model]
  out z_out: [*, seq, d_model]
  set:
    step = ReasoningStep(d_model)
    answer = AnswerStep(d_model)
  graph:
    # 6 reasoning steps with shared weights
    (x, y, z) -> step -> z1
    (x, y, z1) -> step -> z2
    (x, y, z2) -> step -> z3
    (x, y, z3) -> step -> z4
    (x, y, z4) -> step -> z5
    (x, y, z5) -> step -> z_out
    
    (y, z_out) -> answer -> y_out

# Full TRM with T=3 cycles (also weight-shared)
neuron TRM(d_model, vocab_size, max_seq):
  in: [batch, seq]
  out: [batch, seq, vocab_size]
  set:
    cycle = RecursionCycle(d_model)
  graph:
    in ->
      Embedding(vocab_size, d_model)
      PositionalEncoding(d_model, max_len=max_seq)
      x_embedded
    
    x_embedded -> Fork() -> (x, y_init, z_init)
    
    # 3 cycles with shared weights
    (x, y_init, z_init) -> cycle -> (y1, z1)
    (x, y1, z1) -> cycle -> (y2, z2)
    (x, y2, z2) -> cycle -> (y_final, z_final)
    
    y_final ->
      RMSNorm(d_model)
      Linear(d_model, vocab_size)
      out
```

---

## Future Work

### Neurons as First-Class Parameters

The weight-sharing-across-recursion pattern requires passing neurons as parameters:

```neuroscript
neuron RecurseShared(block, depth):  # block is a neuron instance
  ...
```

This requires:
- Type system extension for neuron-typed parameters
- Codegen to pass `nn.Module` instances
- Shape inference through neuron parameters

### Port References

For more advanced feedback patterns, allow referencing ports of bound neurons:

```neuroscript
set:
  cycle = RecursionCycle(d_model)
graph:
  in -> cycle.in.x
  cycle.out.y -> cycle.in.y  # Feedback loop
  cycle.out.z -> cycle.in.z
  cycle.out.y -> out
```

This enables true iterative architectures but requires careful cycle semantics.

### Iteration Count Annotation

Shorthand for "apply N times with feedback":

```neuroscript
(x, y, z) -> cycle @ iterate(3) -> (y_final, z_final)
```

---

## Implementation Notes

### Compiler Phases

1. **Parse:** Recognize `let:`/`set:` blocks, build AST
2. **Validate:** Check recursion termination, guard evaluability
3. **Expand:** Recursively instantiate, evaluating guards at compile time
4. **Flatten:** Convert recursive structure to flat module graph
5. **Codegen:** Emit PyTorch with `__init__` bindings and `forward` flow

### Guard Evaluation

Guards are evaluated in a compile-time interpreter supporting:
- Integer arithmetic: `+`, `-`, `*`, `/`
- Comparisons: `<`, `>`, `<=`, `>=`, `==`, `!=`
- Boolean logic: `and`, `or` (if added)
- Parameter substitution

### Lazy Binding Implementation

Track which lazy bindings are reachable from the selected match arms. Only emit instantiation for reachable bindings.

### Cycle Detection

With `let:` introducing potential self-reference, cycle detection must distinguish:
- **Structural cycles:** A references B references A (still forbidden)
- **Recursive definitions:** A references A with decreasing parameter (allowed if terminates)

---

## Changelog

- **2025-11-27:** Initial draft

---

## References

- [TRM Paper: Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
- [NeuroScript Language Spec v0.1](./neuroscript_spec.md)
- Clojure `let` bindings
- Scala `val`/`lazy val` distinction
