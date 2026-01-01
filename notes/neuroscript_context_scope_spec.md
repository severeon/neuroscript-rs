# NeuroScript Feature Spec: Context Blocks, Scopes & Higher-Order Neurons

**Status:** Proposed for v0.4
**Author:** Thomas + Claude
**Date:** 2026-01-01
**Prerequisites:** Core language (v0.1), Codegen (v0.2), Shape inference (v0.3)

---

## Motivation

NeuroScript currently lacks three capabilities essential for expressing modern neural architectures:

1. **Explicit dependency management:** No way to clearly declare what external dependencies a neuron requires (global state, shared weights, etc.)

2. **Weight sharing at different scopes:** Need to share weights at different levels:
   - Across all instances of all neurons (global embeddings)
   - Across all instances of a specific neuron type (Universal Transformer)
   - Within a single instance (reusing a layer multiple times)

3. **Higher-order composition:** No way to pass neuron instances as parameters, preventing patterns like Universal Transformers or generic layer repeaters.

These limitations prevent clean expression of:
- Universal Transformers (weight-shared layers)
- Tiny Recursive Models (TRM) with iterative refinement
- Shared vocabularies across encoder/decoder architectures
- Generic meta-neurons like "ApplyNTimes"
- Any architecture where depth is parameterized

**Prior Art:** The `let`/`set` specification explored these concepts but had issues:
- Too many separate blocks (`let:`, `set:`, `get:`) made order-of-operations confusing
- Scope boundaries were implicit and hard to validate
- No clear distinction between dependency injection and instance creation

---

## Design Goals

1. **Single dependency block** — One `context:` block where ALL dependencies are declared
2. **Explicit scope control** — Clear annotations for global/static/instance scope
3. **Strict boundaries** — Graph block cannot cross scope boundaries directly
4. **Higher-order neurons** — Neurons as first-class parameters
5. **Lazy instantiation** — Conditional creation for recursive patterns
6. **Compile-time resolution** — Recursion unrolls to flat structure
7. **Compositional** — Works with existing features (match, guards, pipelines)
8. **Footgun prevention** — Invalid scope access caught at compile time

---

## Syntax

### Grammar Additions

```ebnf
program ::= (global_declaration | neuron_def)+

global_declaration ::= "@global" identifier "=" neuron_expr NEWLINE

neuron_def ::= "neuron" identifier "(" params ")" ":" NEWLINE INDENT body DEDENT

body ::= (port | impl_stmt | context_stmt | graph_stmt)+

context_stmt ::= "context" ":" NEWLINE INDENT context_binding+ DEDENT

context_binding ::= annotation* identifier "=" neuron_expr NEWLINE

annotation ::= "@static" | "@global" | "@lazy"

neuron_expr ::= identifier "(" call_args ")"
              | "Freeze" "(" neuron_expr ")"

params ::= param ("," param)*

param ::= identifier ":" type

type ::= "int" | "float" | "Neuron" | shape_type
```

### Keywords Added

```
context
```

### Annotations Added

```
@global   - Module-level scope (only valid at module level)
@static   - Class-level scope (only valid in context: blocks)
@lazy     - Lazy instantiation modifier (only valid in context: blocks)
```

Note: `@` prefix for annotations is new syntax.

---

## Semantics

### Module-Level `@global` Declarations

Global bindings are declared at module level (outside any neuron definition). They are instantiated once when the module loads and shared across all neurons.

```neuroscript
@global vocab_embedding = Embedding(50257, 768)
@global num_attention_heads = 12

neuron Encoder(d_model):
  in: [*, seq, d_model]
  out: [*, seq, d_model]
  context:
    attn = MultiHeadAttention(d_model, @global num_attention_heads)
  graph:
    in -> attn -> out
```

**Codegen (PyTorch):**

```python
# Module-level globals
vocab_embedding = Embedding(50257, 768)
num_attention_heads = 12

class Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_attention_heads)

    def forward(self, x):
        return self.attn(x)
```

### `context:` Block — Single Dependency Declaration Point

The `context:` block is the **only** place where a neuron declares its dependencies. All bindings used in the graph must be declared here.

```neuroscript
neuron TransformerBlock(d_model, num_heads):
  in: [*, seq, d_model]
  out: [*, seq, d_model]

  context:
    @static shared_norm = LayerNorm(d_model)  # Class-level (shared)
    attn = MultiHeadAttention(d_model, num_heads)  # Instance (default)
    @lazy ffn = FFN(d_model)  # Instance, lazy-loaded

  graph:
    in -> shared_norm -> attn -> ffn -> out
```

**Key properties:**
- Appears at most once per neuron
- Must come before `graph:` block
- All names used in `graph:` must be bound in `context:`
- Bindings can reference `@global` names and neuron parameters

---

## Scope System

### 1. Global Scope (`@global`)

**Declaration:** Module level only, outside neuron definitions

**Lifetime:** Instantiated when module loads, exists for program lifetime

**Sharing:** Shared across ALL neurons in the module

**Use cases:**
- Shared vocabularies (encoder/decoder sharing embedding table)
- Global configuration constants
- Shared positional encodings

**Example:**
```neuroscript
@global vocab = Embedding(50257, 768)
@global pos_encoding = PositionalEncoding(768, max_len=2048)

neuron Encoder(d_model):
  context:
    tok_emb = @global vocab
    pos_emb = @global pos_encoding
  graph:
    in -> tok_emb -> tok_emb_out
    in -> pos_emb -> pos_emb_out
    (tok_emb_out, pos_emb_out) -> Add() -> out
```

**Codegen:**
```python
# Module level
vocab = Embedding(50257, 768)
pos_encoding = PositionalEncoding(768, max_len=2048)

class Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.tok_emb = vocab  # Reference, not copy
        self.pos_emb = pos_encoding
```

### 2. Static Scope (`@static`)

**Declaration:** In `context:` block with `@static` annotation

**Lifetime:** Instantiated once per neuron class (not per instance)

**Sharing:** Shared across all instances of THIS neuron type

**Use cases:**
- Universal Transformers (all instances share same weights)
- Weight-tied layers
- Shared normalization layers

**Example:**
```neuroscript
neuron TransformerBlock(d_model):
  in: [*, seq, d_model]
  out: [*, seq, d_model]

  context:
    @static shared_norm = LayerNorm(d_model)
    attn = MultiHeadAttention(d_model, 8)

  graph:
    in -> shared_norm -> attn -> out

# Two instances share the same shared_norm
neuron TwoBlocks(d_model):
  context:
    block1 = TransformerBlock(d_model)
    block2 = TransformerBlock(d_model)
  graph:
    in -> block1 -> block2 -> out
    # block1.shared_norm and block2.shared_norm are THE SAME instance
```

**Codegen:**
```python
class TransformerBlock(nn.Module):
    # Class variable (shared across all instances)
    _shared_norm = None

    def __init__(self, d_model):
        super().__init__()
        # Initialize shared_norm once for all instances
        if TransformerBlock._shared_norm is None:
            TransformerBlock._shared_norm = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, 8)

    def forward(self, x):
        return self.attn(TransformerBlock._shared_norm(x))
```

### 3. Instance Scope (default)

**Declaration:** In `context:` block without annotation (or with `@lazy`)

**Lifetime:** Instantiated when neuron instance is created

**Sharing:** Per-instance, not shared

**Use cases:**
- Standard neural network weights (most common case)
- Per-instance state

**Example:**
```neuroscript
neuron MyNeuron(dim):
  context:
    layer1 = Linear(dim, dim)  # Instance scope (default)
    layer2 = Linear(dim, dim)  # Different from layer1
  graph:
    in -> layer1 -> layer2 -> out
```

**Codegen:**
```python
class MyNeuron(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = Linear(dim, dim)  # Each instance gets its own
        self.layer2 = Linear(dim, dim)
```

---

## Lazy Instantiation (`@lazy`)

The `@lazy` annotation marks an instance-scope binding as conditionally instantiated. The binding is only created if it's referenced in an active code path (determined at compile time via match pattern analysis).

```neuroscript
neuron ConditionalNetwork(dim, use_attention):
  in: [*, dim]
  out: [*, dim]

  context:
    @lazy attn = MultiHeadAttention(dim, 8)  # Only if use_attention true
    ffn = FFN(dim, dim * 4)  # Always created

  graph:
    in -> match:
      [*, dim] where use_attention: attn -> ffn -> out
      [*, dim]: ffn -> out
```

**Compile-time analysis:**
- If `use_attention` is known at compile time, compiler selects one branch
- Only bindings referenced in the selected branch are instantiated
- If both branches possible, both bindings instantiated

**Codegen (when use_attention known at compile time):**
```python
class ConditionalNetwork(nn.Module):
    def __init__(self, dim, use_attention):
        super().__init__()
        if use_attention:
            self.attn = MultiHeadAttention(dim, 8)
        self.ffn = FFN(dim, dim * 4)
        self.use_attention = use_attention

    def forward(self, x):
        if self.use_attention:
            return self.ffn(self.attn(x))
        else:
            return self.ffn(x)
```

---

## Higher-Order Neurons

Neurons can accept other neurons as parameters, enabling generic meta-patterns.

### Neuron Parameters

Parameters can be typed as `Neuron`, indicating they accept a neuron instance:

```neuroscript
neuron ApplyNTimes(block: Neuron, depth: int):
  in: [*]
  out: [*]

  context:
    @lazy next = ApplyNTimes(block, depth - 1)

  graph:
    in -> match:
      [*] where depth > 0: block -> next -> out
      [*]: out
```

**Usage:**
```neuroscript
neuron UniversalTransformer(d_model, num_heads, depth):
  in: [*, seq, d_model]
  out: [*, seq, d_model]

  context:
    @static shared_block = TransformerBlock(d_model, num_heads)

  graph:
    in -> ApplyNTimes(shared_block, depth) -> out
```

### Shape Compatibility Checking

When a neuron parameter is used in a pipeline, the compiler validates shape compatibility:

```neuroscript
neuron Apply(transform: Neuron):
  in: [*, dim]
  out: [*]
  graph:
    in -> transform -> out
    # Compiler checks: transform.in compatible with [*, dim]
    #                  transform.out compatible with [*]
```

**Type checking rules:**
1. Neuron parameter shapes must be compatible with usage context
2. Input shape of neuron param ≥ shape at call site
3. Output shape of neuron param ≤ expected shape at call site
4. Dimension variables unified across the graph

---

## Recursion & Compile-Time Unrolling

### Self-Referential Neurons

A neuron may reference itself in a `context:` binding (typically marked `@lazy`). Combined with match guards, this enables parameterized depth.

```neuroscript
neuron RecursiveStack(d_model, num_heads, d_ff, depth):
  in: [*, seq, d_model]
  out: [*, seq, d_model]

  context:
    @lazy deeper = RecursiveStack(d_model, num_heads, d_ff, depth - 1)

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

### Unrolling Process

When instantiating `RecursiveStack(768, 12, 3072, 3)`:

1. **Evaluate guard:** `depth > 0` → `3 > 0` → true
2. **Select arm:** First match arm (with `TransformerBlock` and `deeper`)
3. **Instantiate lazy binding:** `deeper = RecursiveStack(768, 12, 3072, 2)`
4. **Recurse:** Repeat for depth=2, depth=1, depth=0
5. **Base case:** At depth=0, guard fails → select `Identity()` arm (no `deeper` binding)
6. **Unroll complete:** Flat chain of 3 TransformerBlocks + Identity

**Max depth:** 100 (compiler error if exceeded)

**Codegen (flattened structure):**
```python
class RecursiveStack(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, depth):
        super().__init__()
        self.depth = depth
        if depth > 0:
            self.block = TransformerBlock(d_model, num_heads, d_ff)
            self.deeper = RecursiveStack(d_model, num_heads, d_ff, depth - 1)

    def forward(self, x):
        if self.depth > 0:
            return self.deeper(self.block(x))
        else:
            return x
```

---

## Strict Scope Boundaries

### Rule: Graph Block Cannot Cross Scopes

The graph block can **only** reference names bound in the `context:` block. Direct scope access (e.g., `@global.vocab`) is forbidden.

**Invalid:**
```neuroscript
@global vocab = Embedding(50257, 768)

neuron Encoder(d_model):
  graph:
    in -> @global.vocab -> out  # ERROR: Cannot reference @global in graph
```

**Valid:**
```neuroscript
@global vocab = Embedding(50257, 768)

neuron Encoder(d_model):
  context:
    embedding = @global vocab  # Bind in context
  graph:
    in -> embedding -> out  # Reference bound name
```

### Rationale

Strict boundaries provide:
1. **Clear dependencies:** All external dependencies visible in `context:` block
2. **Validation:** Compiler can check all references exist
3. **Namespace control:** No accidental name collisions
4. **Refactoring safety:** Easy to see what breaks when changing scopes

---

## Validation Rules

### 1. Scope Declaration Rules

- `@global` declarations MUST appear at module level (outside neuron definitions)
- `@global` declarations CANNOT appear in `context:` blocks
- `@static` annotations MUST appear in `context:` blocks
- `@lazy` annotations MUST appear in `context:` blocks
- `@lazy` CANNOT be combined with `@static` (static is always eager)

### 2. Reference Rules

- Graph connections MUST only reference names from `context:` block
- Graph connections CANNOT use `@global.name` or `@static.name` syntax
- Context bindings MAY reference `@global name` (read global value)
- Context bindings MAY reference neuron parameters
- Context bindings CANNOT have forward references (order matters)

### 3. Scope Interaction Rules

- Instance bindings CAN reference `@global` names
- Instance bindings CAN reference `@static` names from the SAME neuron
- `@static` bindings CAN reference `@global` names
- `@static` bindings CANNOT reference instance bindings
- `@global` declarations CANNOT reference anything (must be self-contained)

### 4. Higher-Order Neuron Rules

- Neuron parameters MUST be typed as `Neuron` in signature
- Neuron parameters CAN be passed to other neurons
- Neuron parameters CAN be used in graph pipelines
- Neuron parameter usage MUST satisfy shape compatibility
- Compiler MUST validate neuron parameter shapes at compile time

### 5. Recursion Rules

- Recursive `context:` bindings MUST be marked `@lazy`
- Recursion depth MUST be computable at compile time
- Recursion depth MUST NOT exceed 100
- Recursive bindings MUST have a base case (guard that terminates)
- Runtime-variable depth → compile error

### 6. Name Collision Rules

- Names in `context:` block MUST be unique within that neuron
- Global names MUST be unique within the module
- Static names MAY shadow global names (static takes precedence)
- Instance names MAY shadow global/static names (instance takes precedence)

---

## Interaction with Existing Features

### With `match` Guards

Guards control which match arm is selected, which determines which `@lazy` bindings are instantiated:

```neuroscript
neuron AdaptiveDepth(dim, depth):
  in: [*, dim]
  out: [*, dim]

  context:
    @lazy shallow = ShallowNet(dim)
    @lazy deep = DeepNet(dim)

  graph:
    in -> match:
      [*, dim] where depth <= 2: shallow -> out
      [*, dim]: deep -> out
```

Only one of `shallow` or `deep` is instantiated based on compile-time `depth` value.

### With Multi-Port Neurons

Context bindings work with any neuron, including multi-port:

```neuroscript
neuron ResidualBlock(dim):
  in: [*, dim]
  out: [*, dim]

  context:
    split = Fork()
    transform = Linear(dim, dim)
    merge = Add()

  graph:
    in -> split -> (main, skip)
    main -> transform -> transformed
    (transformed, skip) -> merge -> out
```

### With Pipelines

Bound names integrate seamlessly into pipelines:

```neuroscript
neuron RepeatedNorm(dim):
  context:
    norm = LayerNorm(dim)
  graph:
    in -> norm -> norm -> norm -> out  # Same instance used 3 times
```

### With Shape Inference

Shape inference works across scopes:

```neuroscript
@global embedding_dim = 768

neuron Encoder(seq_len):
  in: [batch, seq_len]
  out: [batch, seq_len, @global embedding_dim]  # Reference in shape
  # ERROR: Cannot reference @global in port shapes
  # Must use parameter instead
```

**Correct approach:**
```neuroscript
@global embedding_dim = 768

neuron Encoder(seq_len, d_model):
  in: [batch, seq_len]
  out: [batch, seq_len, d_model]

  context:
    # Validation: d_model must match @global embedding_dim if used together
    embed = Embedding(50257, @global embedding_dim)
```

---

## Examples

### Example 1: Universal Transformer with Higher-Order Neurons

```neuroscript
# Generic meta-neuron for applying a block N times
neuron ApplyNTimes(block: Neuron, depth: int):
  in: [*]
  out: [*]

  context:
    @lazy next = ApplyNTimes(block, depth - 1)

  graph:
    in -> match:
      [*] where depth > 0: block -> next -> out
      [*]: out

# Universal Transformer with weight-shared blocks
neuron UniversalTransformer(d_model, num_heads, depth):
  in: [*, seq, d_model]
  out: [*, seq, d_model]

  context:
    @static shared_block = TransformerBlock(d_model, num_heads)

  graph:
    in -> ApplyNTimes(shared_block, depth) -> out
```

**Key points:**
- `ApplyNTimes` is higher-order (takes `block: Neuron`)
- `shared_block` is `@static` → shared across all UniversalTransformer instances
- Recursion unrolls at compile time (depth must be known)

### Example 2: Shared Vocabulary Across Encoder/Decoder

```neuroscript
@global vocab_table = Embedding(50257, 768)
@global pos_encoding = PositionalEncoding(768, max_len=2048)

neuron Encoder(d_model):
  in: [batch, seq]
  out: [batch, seq, d_model]

  context:
    tok_embed = @global vocab_table
    pos_embed = @global pos_encoding

  graph:
    in -> tok_embed -> tok_out
    in -> pos_embed -> pos_out
    (tok_out, pos_out) -> Add() -> out

neuron Decoder(d_model):
  in: [batch, seq]
  out: [batch, seq, vocab_size]

  context:
    tok_embed = @global vocab_table  # Same embedding as Encoder
    pos_embed = @global pos_encoding
    projection = Linear(d_model, 50257)

  graph:
    in -> tok_embed -> tok_out
    in -> pos_embed -> pos_out
    (tok_out, pos_out) -> Add() -> projection -> out
```

**Key points:**
- `@global` ensures encoder and decoder share exact same embedding weights
- Explicit binding in `context:` makes dependency clear
- Validation ensures vocab_table is only created once

### Example 3: GPT-2 with Parameterized Depth

```neuroscript
neuron GPTStack(d_model, num_heads, d_ff, num_layers):
  in: [*, seq, d_model]
  out: [*, seq, d_model]

  context:
    @lazy rest = GPTStack(d_model, num_heads, d_ff, num_layers - 1)

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

  context:
    stack = GPTStack(d_model, num_heads, d_ff, num_layers)

  graph:
    in ->
      Embedding(vocab_size, d_model)
      PositionalEncoding(d_model, max_len=max_seq)
      stack
      LayerNorm(d_model)
      Linear(d_model, vocab_size)
      out

# Instantiate specific models
# GPT2(50257, 768, 12, 3072, 12, 1024)   -> GPT-2 Small (12 layers)
# GPT2(50257, 1024, 16, 4096, 24, 1024)  -> GPT-2 Medium (24 layers)
```

**Key points:**
- `GPTStack` recursively unrolls based on `num_layers`
- Each layer gets its own weights (not weight-shared)
- Max 100 layers (compiler enforces limit)

### Example 4: Conditional Feature with Lazy Loading

```neuroscript
neuron AdaptiveModel(dim, use_attention, use_residual):
  in: [*, dim]
  out: [*, dim]

  context:
    @lazy attn = MultiHeadAttention(dim, 8)
    @lazy residual_path = Identity()
    ffn = FFN(dim, dim * 4)

  graph:
    in -> match:
      [*, dim] where use_attention and use_residual:
        Fork() -> (main, skip)
        main -> attn -> ffn -> processed
        (processed, skip) -> Add() -> out

      [*, dim] where use_attention:
        attn -> ffn -> out

      [*, dim]:
        ffn -> out
```

**Key points:**
- `@lazy` prevents instantiating unused modules
- Compiler analyzes which path is taken based on compile-time parameters
- Only necessary modules are created

### Example 5: Tiny Recursive Model (TRM) with Static Shared Weights

```neuroscript
neuron ReasoningStep(d_model):
  in x: [*, seq, d_model]
  in y: [*, seq, d_model]
  in z: [*, seq, d_model]
  out: [*, seq, d_model]

  context:
    proj = Linear(d_model * 3, d_model)
    block = TRMBlock(d_model)

  graph:
    (x, y, z) -> Concat(dim=-1) -> proj -> block -> out

neuron RecursionCycle(d_model, num_steps):
  in x: [*, seq, d_model]
  in y: [*, seq, d_model]
  in z: [*, seq, d_model]
  out y_out: [*, seq, d_model]
  out z_out: [*, seq, d_model]

  context:
    @static step = ReasoningStep(d_model)  # Shared across all cycles
    answer = AnswerStep(d_model)

  graph:
    # Apply step num_steps times (weight-shared)
    in -> ApplyNTimes(step, num_steps) -> z_out
    (y, z_out) -> answer -> y_out
```

**Key points:**
- `@static step` is shared across all RecursionCycle instances
- Combines higher-order neurons with static scope
- Enables TRM's iterative reasoning pattern

---

## Implementation Notes

### Compiler Phases

1. **Parse:** Recognize `@global`, `context:`, and annotations
2. **IR Construction:** Build scope-aware IR with annotations
3. **Validation:**
   - Check scope rules (global at module level, etc.)
   - Check reference rules (graph uses only context names)
   - Check recursion depth limits
   - Type-check neuron parameters
4. **Scope Resolution:** Resolve all `@global`/`@static` references
5. **Recursion Expansion:** Unroll recursive neurons (max depth: 100)
6. **Lazy Analysis:** Determine which `@lazy` bindings are reachable
7. **Codegen:** Emit Python with proper scope handling

### Scope Implementation in Codegen

**Global scope:**
```python
# Module level, before any class definitions
global_var = SomeNeuron(args)
```

**Static scope:**
```python
class MyNeuron(nn.Module):
    # Class variable, initialized on first instance
    _static_var = None

    def __init__(self, ...):
        if MyNeuron._static_var is None:
            MyNeuron._static_var = SomeNeuron(args)
        # ...
```

**Instance scope:**
```python
class MyNeuron(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.instance_var = SomeNeuron(args)
        # ...
```

**Lazy instantiation:**
```python
class MyNeuron(nn.Module):
    def __init__(self, condition):
        super().__init__()
        if condition:
            self.lazy_var = SomeNeuron(args)
        self.condition = condition

    def forward(self, x):
        if self.condition:
            return self.lazy_var(x)
        else:
            return x
```

### Higher-Order Neuron Implementation

**Neuron parameters in Python:**
```python
class ApplyNTimes(nn.Module):
    def __init__(self, block: nn.Module, depth: int):
        super().__init__()
        self.block = block  # Store neuron instance
        self.depth = depth
        if depth > 0:
            self.next = ApplyNTimes(block, depth - 1)

    def forward(self, x):
        if self.depth > 0:
            return self.next(self.block(x))
        else:
            return x
```

### Recursion Unrolling Strategy

**Compile-time evaluation:**
1. Start with initial parameters (e.g., `depth=3`)
2. Evaluate guard expression with parameters
3. If guard true, instantiate binding and recurse with decremented parameter
4. Track unroll depth, error if exceeds 100
5. Continue until base case reached
6. Generate nested Python __init__ structure

### Error Messages

**Scope violation:**
```
Error: Cannot reference @global.vocab directly in graph block
  --> example.ns:15:8
   |
15 |     in -> @global.vocab -> out
   |           ^^^^^^^^^^^^^
   |
   = note: Global bindings must be imported in context: block
   = help: Add 'embedding = @global vocab' to context: block
```

**Recursion depth exceeded:**
```
Error: Recursion depth exceeded limit of 100
  --> example.ns:23:12
   |
23 |     deeper = RecursiveStack(d_model, num_heads, d_ff, depth - 1)
   |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: Unrolling stopped at depth=100
   = help: Reduce the initial depth parameter or increase the limit
```

**Invalid scope annotation:**
```
Error: @static annotation not allowed at module level
  --> example.ns:3:1
   |
3  | @static shared_layer = Linear(768, 768)
   | ^^^^^^^
   |
   = note: @static is only valid in context: blocks
   = help: Use @global for module-level bindings
```

---

## Migration from `let`/`set`

### Mapping Old Syntax to New

| Old Syntax | New Syntax | Notes |
|------------|------------|-------|
| `set: a = ...` | `context: a = ...` | Default is instance scope, eager |
| `let: a = ...` | `context: @lazy a = ...` | Lazy instantiation |
| `get: a = @global.x` | `context: a = @global x` | Simplified reference syntax |
| (none) | `@global x = ...` | New: module-level declarations |
| (none) | `context: @static a = ...` | New: class-level scope |

### Example Migration

**Old (let/set/get):**
```neuroscript
neuron OldStyle(dim):
  set:
    layer1 = Linear(dim, dim)
  let:
    layer2 = Linear(dim, dim)
  get:
    vocab = @global.vocab_table
  graph:
    in -> layer1 -> layer2 -> vocab -> out
```

**New (context):**
```neuroscript
neuron NewStyle(dim):
  context:
    layer1 = Linear(dim, dim)         # Eager instance (was set:)
    @lazy layer2 = Linear(dim, dim)   # Lazy instance (was let:)
    vocab = @global vocab_table       # Reference global (was get:)
  graph:
    in -> layer1 -> layer2 -> vocab -> out
```

---

## Future Work

### Runtime-Variable Depth

Currently recursion depth must be compile-time constant. Future versions could support:

```neuroscript
neuron DynamicStack(d_model, depth: runtime):
  # depth not known at compile time
  # Could generate Python recursion or loop
```

This requires either:
- Python-level recursion (with depth limit checking)
- Loop-based implementation
- JIT compilation based on runtime depth

### Port References in Context

Allow referencing specific ports of bound neurons:

```neuroscript
context:
  cycle = RecursionCycle(d_model)
graph:
  in -> cycle.in.x
  cycle.out.y -> cycle.in.y  # Feedback loop
  cycle.out.z -> out
```

Enables explicit feedback patterns and multi-port routing.

### Optimization: Layer Deduplication

When recursion unrolls to identical layers, automatically deduplicate:

```neuroscript
neuron Stack(d_model, depth):
  context:
    @lazy rest = Stack(d_model, depth - 1)
  graph:
    in -> match:
      [*] where depth > 0: Linear(d_model, d_model) -> rest -> out
      [*]: out

# Stack(768, 5) creates 5 identical Linear layers
# Optimizer could detect and weight-share them
```

### Type System for Neuron Parameters

More precise types for neuron parameters:

```neuroscript
neuron Apply(transform: Neuron[in: [*, dim], out: [*, dim]]):
  # Type constraint: transform must have specific signature
```

Enables better compile-time validation and documentation.

---

## Changelog

- **2026-01-01:** Initial draft based on let/set spec evolution
- Replaced separate `let:`/`set:`/`get:` blocks with single `context:` block
- Added `@global`, `@static`, `@lazy` annotations
- Added strict scope boundary rules
- Added higher-order neuron support
- Set recursion limit to 100

---

## References

- [TRM Paper: Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
- [NeuroScript Language Spec v0.1](./neuroscript_spec.md)
- [NeuroScript Let/Set Spec (predecessor)](./neuroscript_let_set_spec.md)
- Python class variables and module-level state
- PyTorch nn.Module design patterns
