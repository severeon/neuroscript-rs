# NeuroScript Language Specification v0.1

**Status**: Current implementation as of December 2025
**Compiler Version**: Phase 2 Complete (Parse → Validate → Codegen)

---

## Table of Contents

1. [Lexical Structure](#1-lexical-structure)
2. [Type System](#2-type-system)
3. [Program Structure](#3-program-structure)
4. [Graph Syntax](#4-graph-syntax)
5. [Bindings](#5-bindings)
6. [Shape System](#6-shape-system)
7. [Primitives](#7-primitives)
8. [Validation](#8-validation)
9. [Code Generation](#9-code-generation)
10. [Standard Library](#10-standard-library)
11. [Error Handling](#11-error-handling)
12. [Optimization](#12-optimization)
13. [Language Features Status](#13-language-features-status)
14. [Examples](#14-examples)
15. [CLI Usage](#15-cli-usage)
16. [Implementation Notes](#16-implementation-notes)

---

## 1. LEXICAL STRUCTURE

### 1.1 Keywords

```
neuron, use, in, out, impl, graph, match, where, external, and, or, let, set, true, false
```

**Reserved for future use**: `loop`, `for`, `while`, `if`, `else`

### 1.2 Operators

| Category | Operators | Usage |
|----------|-----------|-------|
| Pipeline | `->` | Connection flow |
| Arithmetic | `+`, `-`, `*`, `/` | Dimension expressions |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` | Match guards |
| Logical | `and`, `or` | Guard composition |
| Assignment | `=` | Bindings, parameters |
| Structural | `:`, `,`, `.`, `/` | Syntax elements |

### 1.3 Delimiters

- **Parentheses** `( )`: Parameter lists, expressions, tuple unpacking
- **Brackets** `[ ]`: Tensor shapes
- **Indentation**: Python-style significant whitespace (2 spaces or tabs)

### 1.4 Literals

```neuroscript
# Integers
512, -1, 0, 1024

# Floats
0.1, 3.14, -0.5, 1e-3

# Strings (backtick-quoted)
`hello`, `model.weights`, `path/to/file`

# Booleans
true, false
```

### 1.5 Comments

```neuroscript
# Single-line comments only
# Multi-line comments: use multiple # lines
```

### 1.6 Indentation Rules

```neuroscript
# 2-space indentation (or tabs = 2 spaces)
neuron Example:
  in: [dim]      # 2-space indent
  out: [dim]     # Consistent throughout block
  graph:         # Same level
    in -> out    # Nested block
```

**Rules:**
- Indentation creates block structure
- Dedentation ends blocks
- Blank lines and comment-only lines ignored during indentation tracking
- Inconsistent indentation is a lexer error

### 1.7 Identifiers

```
[a-zA-Z_][a-zA-Z0-9_]*
```

**Naming conventions:**
- Neuron names: `PascalCase` (e.g., `TransformerBlock`, `MultiHeadAttention`)
- Parameters: `snake_case` (e.g., `d_model`, `num_heads`)
- Dimensions: `snake_case` (e.g., `batch`, `seq`, `dim`)
- Port names: `snake_case` (e.g., `left`, `right`, `query_out`)

---

## 2. TYPE SYSTEM

### 2.1 Tensor Shapes

NeuroScript's type system is primarily shape-based. All data flowing through neurons is tensors with explicit shapes.

#### Dimension Types

**1. Literal Dimensions**
```neuroscript
[512, 256, 128]    # Concrete integer dimensions
[1, 1024]          # Batch size 1, sequence length 1024
```

**2. Named Dimensions**
```neuroscript
[batch, seq, dim]       # Named dimension variables
[b, heads, seq, d_k]    # Can be bound through inference
```

Named dimensions enable:
- Shape polymorphism
- Dimension unification across connections
- Expression in guards and calls

**3. Wildcard (Single)**
```neuroscript
[*, dim]        # First dimension is any value, second is 'dim'
[batch, *]      # 'batch' followed by any dimension
```

The wildcard `*` matches exactly one dimension but doesn't capture its value.

**4. Variadic Wildcard**
```neuroscript
[*shape]              # Match any rank (0 or more dimensions)
[*batch, seq, dim]    # Variadic prefix: any leading dims, then seq, dim
[batch, seq, *rest]   # Variadic suffix: batch, seq, then any trailing dims
```

Variadic wildcards match zero or more dimensions and capture the matched sequence.

**5. Expression Dimensions**
```neuroscript
[dim * 4]           # Computed from 'dim'
[seq - 1]           # Arithmetic expressions
[heads * d_k]       # Product of named dims
[d_model / heads]   # Division (must be exact)
```

Supported operators: `+`, `-`, `*`, `/`

#### Shape Examples

```neuroscript
# Scalar (rank 0)
[]

# Vector (rank 1)
[dim]

# Matrix
[rows, cols]

# Batched sequences
[batch, seq, d_model]

# Multi-head attention
[batch, heads, seq, d_k]

# Polymorphic
[*, dim]              # Any batch shape, fixed feature dim
[*shape, d_model]     # Any prefix, fixed final dim

# Expressions
[batch, seq, heads * d_k]    # Computed dimension
```

### 2.2 Parameter Types

#### Standard Parameters
```neuroscript
neuron Linear(in_dim, out_dim):              # Required params
neuron Dropout(p=0.1):                       # With default
neuron LayerNorm(dim, eps=1e-5):            # Mixed
```

Parameters accept:
- Integers: `512`, `1024`
- Floats: `0.1`, `1e-5`
- Strings: `` `relu` ``, `` `path/to/file` ``
- Booleans: `true`, `false`
- Expressions: `dim * 4`, `heads * d_k`

#### Type Annotations

**NeuronType**: Meta-parameter accepting neuron type references

```neuroscript
neuron Sequential(count, neuron_type: NeuronType):
    in: [*shape]
    out: [*shape]
    # 'neuron_type' is a neuron class, not an instance
```

Usage:
```neuroscript
let:
    stack = Sequential(12, TransformerBlock)  # Pass neuron type
```

The validator automatically converts `Value::Name` to `Value::NeuronRef` for `NeuronType` parameters.

---

## 3. PROGRAM STRUCTURE

### 3.1 Use Statements

Import neurons from external sources:

```neuroscript
use core,nn/*                           # Wildcard import
use core,nn/Linear                      # Specific neuron
use core,activations/GELU               # From submodule
use neuroscript_runtime,primitives/*    # Runtime primitives
```

**Format**: `use <source>,<path>/<item>`

- **Source**: Provider/package name (e.g., `core`, `neuroscript_runtime`)
- **Path**: Module path with `/` separators (e.g., `nn`, `activations`)
- **Item**: Neuron name or `*` for all neurons in module

### 3.2 Neuron Definitions

#### Basic Structure

```neuroscript
neuron <Name>(<params>):
    in: <port_spec>
    out: <port_spec>
    <body>
```

#### Parameters

```neuroscript
# No parameters
neuron GELU:

# Single parameter
neuron Dropout(p):

# Multiple parameters
neuron Linear(in_dim, out_dim):

# With defaults
neuron LayerNorm(dim, eps=1e-5):

# Type-annotated
neuron Sequential(count, neuron_type: NeuronType):
```

#### Port Specifications

**Inline (single default port):**
```neuroscript
neuron Linear(in_dim, out_dim):
    in: [*, in_dim]
    out: [*, out_dim]
```

**Named ports:**
```neuroscript
neuron Fork:
    in: [*shape]
    out left: [*shape]
    out right: [*shape]
```

**Multiple ports (indented):**
```neuroscript
neuron MultiInput:
    in:
        query: [batch, seq, dim]
        key: [batch, seq, dim]
        value: [batch, seq, dim]
    out:
        attention: [batch, seq, dim]
        weights: [batch, seq, seq]
```

**Port naming:**
- Default port name: `in` or `out` (maps to internal name "default")
- Named ports: any identifier (e.g., `left`, `right`, `query`, `key`)

### 3.3 Neuron Bodies

#### Primitive Neurons

Reference external implementations:

```neuroscript
neuron Linear(in_dim, out_dim):
    in: [*, in_dim]
    out: [*, out_dim]
    impl: neuroscript_runtime,primitives/Linear
```

**Format**: `impl: <source>,<library>/<class>`

Maps to Python import:
```python
from neuroscript_runtime.primitives.linear import Linear
```

#### Composite Neurons

Define internal connection graphs:

```neuroscript
neuron MLP(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            Linear(dim, dim * 4)
            GELU()
            Linear(dim * 4, dim)
            out
```

**Optional bindings before graph:**
```neuroscript
neuron ResidualMLP(dim, depth):
    let:
        recurse = ResidualMLP(dim, depth - 1)
    set:
        norm = LayerNorm(dim)
    graph:
        in -> match:
            [*] where depth > 0:
                norm
                MLP(dim)
                recurse
                out
            [*]:
                Identity()
                out
```

---

## 4. GRAPH SYNTAX

The `graph:` section defines dataflow connections between neurons.

### 4.1 Connections

#### Basic Pipeline

```neuroscript
in -> Linear(512, 256) -> GELU() -> out
```

Single-line pipeline with `->` operator connecting stages.

#### Multi-line Pipeline

```neuroscript
in ->
    Linear(512, 256)
    GELU()
    Dropout(0.1)
    Linear(256, 512)
    out
```

Indentation indicates continuation. Each indented line is implicitly connected to the previous with `->`.

#### Named References

```neuroscript
in -> Linear(512, 256) -> hidden
hidden -> GELU() -> activated
activated -> Dropout(0.1) -> dropped
dropped -> out
```

Intermediate results can be named and referenced later.

### 4.2 Tuple Unpacking

Split multi-output neurons into named references:

```neuroscript
in -> Fork() -> (left, right)
left -> Linear(dim, dim) -> processed
right -> Identity() -> skip
(processed, skip) -> Add() -> out
```

**Rules:**
- Tuple arity must match source output count
- Creates named references for each output
- Can be used as connection sources
- Can be recombined with tuple syntax `(a, b) -> Target()`

**Multi-output example:**
```neuroscript
in -> Fork3() -> (q, k, v)
(q, k) -> ScaledDotProductAttention() -> (attn, weights)
```

### 4.3 Port Access

#### Default Ports

Single-port neurons automatically use the default port:
```neuroscript
in -> Linear(dim, dim) -> out  # Uses default ports
```

#### Named Port Access

```neuroscript
in -> Fork() -> fork_ref
fork_ref.left -> Linear(dim, dim) -> processed
fork_ref.right -> Identity() -> skip
```

Dot notation: `<reference>.<port_name>`

#### Port Reference Examples

```neuroscript
# Default port
in -> Linear(512, 256) -> hidden

# Named output ports
in -> Fork() -> fork
fork.left -> out_left
fork.right -> out_right

# Named input ports (if neuron has multiple inputs)
query -> mha.query
key -> mha.key
value -> mha.value
mha.output -> out
```

### 4.4 Match Expressions

Pattern match on input tensor shapes with dimension capture:

```neuroscript
in -> match:
    <pattern>: <pipeline>
    <pattern> where <guard>: <pipeline>
    ...
```

#### Basic Pattern Matching

```neuroscript
in -> match:
    [*, 512]: Identity() -> out          # Exactly 512 in last dim
    [*, 256]: Linear(256, 512) -> out    # Exactly 256 in last dim
    [*, d]: Linear(d, 512) -> out        # Any other dimension
```

#### Dimension Capture

Named dimensions in patterns are captured and can be used:

```neuroscript
in -> match:
    [batch, seq, d]: Linear(d, 512) -> out  # Captures batch, seq, d
```

Captured dimensions available in:
- Guard expressions
- Neuron call arguments
- Downstream connections

#### Guards

Boolean expressions filtering patterns:

```neuroscript
in -> match:
    [*, d] where d > 512: Linear(d, 512) -> out
    [*, d] where d == 256: Identity() -> out
    [*, d] where d < 256: Linear(d, 256) -> Linear(256, 512) -> out
    [*, d]: Linear(d, 512) -> out  # Catch-all
```

**Guard operators:**
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `and`, `or`
- Operands: captured dimensions, literals

#### Complex Match Example

```neuroscript
neuron AdaptiveNorm:
    in: [batch, seq, dim]
    out: [batch, seq, dim]
    graph:
        in -> match:
            [b, s, d] where d <= 512:
                LayerNorm(d)
                out
            [b, s, d] where d > 512 and d <= 2048:
                RMSNorm(d)
                out
            [b, s, d]:
                GroupNorm(d, 8)
                out
```

#### Validation Rules

1. **Exhaustiveness**: Must have a catch-all pattern (e.g., `[*]`, `[*, d]`)
2. **Reachability**: Warns on patterns shadowed by earlier patterns
3. **Type consistency**: Guard expressions must be boolean
4. **Dimension usage**: Captured dimensions must be used consistently

---

## 5. BINDINGS

Bindings allow reusable neuron instantiation within a neuron definition.

### 5.1 Let Bindings (Lazy)

**Lazy instantiation**: Modules created on first use in `forward()`.

```neuroscript
let:
    name = NeuronCall(args)
```

**Characteristics:**
- Created in a dictionary during first access
- Supports dimension capture from match patterns
- Enables controlled recursion
- Multiple instances can exist with different captured dimensions

**Example:**
```neuroscript
neuron RecursiveBlock(depth):
    let:
        recurse = RecursiveBlock(depth - 1)
        processor = MLP(512)
    graph:
        in -> match:
            [*] where depth > 0:
                processor
                recurse
                out
            [*]:
                Identity()
                out
```

Generated code:
```python
def forward(self, x):
    # Lazy instantiation
    if 'processor' not in self._let_cache:
        self._let_cache['processor'] = MLP(512)

    if x.shape satisfies depth > 0:
        x = self._let_cache['processor'](x)
        if 'recurse' not in self._let_cache:
            self._let_cache['recurse'] = RecursiveBlock(depth - 1)
        x = self._let_cache['recurse'](x)
```

### 5.2 Set Bindings (Eager)

**Eager instantiation**: Modules created in `__init__()`.

```neuroscript
set:
    name = NeuronCall(args)
```

**Characteristics:**
- Instantiated once during initialization
- Cannot capture match dimensions (must use parameters)
- Cannot be recursive (would cause infinite initialization)
- More efficient than `let` when dimensions are known

**Example:**
```neuroscript
neuron TransformerBlock(d_model):
    set:
        norm1 = LayerNorm(d_model)
        norm2 = LayerNorm(d_model)
        ffn = FFN(d_model, d_model * 4)
    graph:
        in -> Fork() -> (main, skip1)
        main ->
            norm1
            MultiHeadSelfAttention(d_model)
            attn_out
        (attn_out, skip1) -> Add() -> residual1
        residual1 -> Fork() -> (main2, skip2)
        main2 -> norm2 -> ffn -> ffn_out
        (ffn_out, skip2) -> Add() -> out
```

Generated code:
```python
def __init__(self, d_model):
    super().__init__()
    self.norm1 = LayerNorm(d_model)
    self.norm2 = LayerNorm(d_model)
    self.ffn = FFN(d_model, d_model * 4)
```

### 5.3 Binding Validation

**Errors:**
- **Duplicate names**: Same name used twice in `let`/`set`
- **Recursive `set`**: `set: recurse = MyNeuron(depth - 1)` → Error
- **Unresolved parameters**: `let: norm = LayerNorm(unknown_dim)` → Error if `unknown_dim` not captured

---

## 6. SHAPE SYSTEM

NeuroScript's shape system provides compile-time shape checking and inference.

### 6.1 Shape Algebra Operations

Implemented operations on shapes (using `BigUint` to prevent overflow):

| Operation | Description | Example |
|-----------|-------------|---------|
| `size()` | Total element count | `[10, 20, 30].size() = 6000` |
| `rank()` | Number of dimensions | `[batch, seq, dim].rank() = 3` |
| `axiswise_le(other)` | Element-wise ≤ | `[10, 20] ≤ [10, 30] = true` |
| `axiswise_divides(other)` | Divisibility check | `[2, 4] divides [10, 8] = true` |
| `axiswise_gcd(other)` | GCD per dimension | `gcd([12, 18], [8, 24]) = [4, 6]` |
| `axiswise_lcm(other)` | LCM per dimension | `lcm([4, 6], [6, 9]) = [12, 18]` |
| `broadcastable(other)` | NumPy broadcasting | `[1, 20] ⊕ [10, 20] = true` |
| `tiles(other)` | Tiling compatibility | `[10, 20] tiles [2, 4]` |
| `permutes(other)` | Same dims, diff order | `[2, 3, 4] permutes [3, 4, 2]` |
| `flatten()` | Reduce to rank 1 | `[10, 20, 30] → [6000]` |
| `refine_axis(i, factors)` | Split dimension | `[6] → [2, 3]` |
| `coarsen_axes(i, j)` | Merge dimensions | `[10, 20, 30] → [200, 30]` |

### 6.2 Pattern Matching

Shapes can be matched against patterns with wildcards and captures.

#### Pattern Tokens

```rust
pub enum PatternToken {
    Lit(BigUint),       // Literal: 512
    Any(Option<Name>),  // Wildcard: * or *batch (captures if named)
    Ignore,             // Ignore: _ (doesn't capture)
    Rest,               // Variadic: ... (matches remaining)
}
```

#### Matching Rules

**Non-variadic patterns:**
```neuroscript
[*, 512]     # Rank must be 2, last dim must be 512
[batch, seq] # Rank must be 2, both dims captured
```

**Variadic patterns:**
```neuroscript
[*shape]           # Any rank, captures entire shape
[*batch, seq, d]   # Prefix variadic: (..., seq, d)
[batch, seq, *rest] # Suffix variadic: (batch, seq, ...)
```

**Literal matching:**
```neuroscript
[512, 256] matches [512, 256]  # ✓
[512, 256] matches [512, 128]  # ✗
```

**Named dimension unification:**
```neuroscript
# First match: dim = 512
[batch, seq, dim] matches [32, 128, 512]

# Later use: dim must still be 512
Linear(dim, dim * 2)  # Linear(512, 1024)
```

### 6.3 Shape Inference

The shape inference engine validates shape compatibility across connections.

#### Inference Context

Tracks:
- **Resolved dimensions**: `dim → 512`, `batch → 32`
- **Equivalences**: `d_k == d_v`, `heads * d_k == d_model`
- **Node outputs**: `hidden_node → [batch, seq, 768]`

#### Unification

Pattern-based dimension resolution:

```neuroscript
# Connection: [batch, seq, dim] -> Linear(dim, 512) -> [batch, seq, 512]

# Unification:
# 1. Input shape [batch, seq, dim] captures batch, seq, dim
# 2. Linear expects [*, dim] and produces [*, 512]
# 3. Verify input compatible with [*, dim]
# 4. Output shape: [batch, seq, 512]
```

#### Expression Solving

When possible, solve for unknowns:

```neuroscript
# Given: dim * 4 = 2048
# Solve: dim = 512

# Given: heads * d_k = d_model, d_model = 768, heads = 12
# Solve: d_k = 64
```

**Solver capabilities:**
- Linear equations: `a * x = b` → `x = b / a` (if exact)
- Addition/subtraction: `x + a = b` → `x = b - a`
- Multiple constraints: builds system of equations

**Limitations:**
- Non-linear equations may not solve
- Ambiguous systems return errors
- Integer division must be exact

#### Validation Flow

```
1. Parse shape expressions into AST
2. Initialize inference context
3. For each connection:
   a. Unify source output with sink input
   b. Resolve dimension variables
   c. Check compatibility
   d. Record node outputs
4. Report errors for:
   - Incompatible shapes
   - Unresolvable dimensions
   - Contradictory constraints
```

---

## 7. PRIMITIVES

### 7.1 Registered Primitives

NeuroScript includes 26 built-in primitives in the stdlib registry:

#### Core Layers

| Neuron | Parameters | Shape Signature |
|--------|------------|-----------------|
| `Linear` | `in_dim, out_dim` | `[*, in_dim] → [*, out_dim]` |
| `Identity` | — | `[*shape] → [*shape]` |

#### Activations

| Neuron | Parameters | Shape Signature |
|--------|------------|-----------------|
| `GELU` | — | `[*shape] → [*shape]` |
| `ReLU` | — | `[*shape] → [*shape]` |
| `Tanh` | — | `[*shape] → [*shape]` |
| `Sigmoid` | — | `[*shape] → [*shape]` |
| `SiLU` | — | `[*shape] → [*shape]` |
| `Softmax` | `dim=-1` | `[*shape] → [*shape]` |

#### Normalization

| Neuron | Parameters | Shape Signature |
|--------|------------|-----------------|
| `LayerNorm` | `dim, eps=1e-5` | `[*, dim] → [*, dim]` |
| `RMSNorm` | `dim` | `[*, dim] → [*, dim]` |
| `GroupNorm` | `dim, groups` | `[*, dim] → [*, dim]` |

#### Regularization

| Neuron | Parameters | Shape Signature |
|--------|------------|-----------------|
| `Dropout` | `p=0.1` | `[*shape] → [*shape]` |
| `DropPath` | `p=0.1` | `[*shape] → [*shape]` |
| `DropConnect` | `p=0.1` | `[*shape] → [*shape]` |

#### Embeddings

| Neuron | Parameters | Shape Signature |
|--------|------------|-----------------|
| `Embedding` | `vocab, dim` | `[*batch] → [*batch, dim]` |
| `PositionalEncoding` | `seq_len, dim` | `[batch, seq, dim] → [batch, seq, dim]` |
| `LearnedPositionalEmbedding` | `seq_len, dim` | `[batch, seq] → [batch, seq, dim]` |

#### Structural

| Neuron | Parameters | Shape Signature |
|--------|------------|-----------------|
| `Fork` | — | `[*shape] → ([*shape], [*shape])` |
| `Fork3` | — | `[*shape] → ([*shape], [*shape], [*shape])` |
| `Add` | — | `([*shape], [*shape]) → [*shape]` |
| `Multiply` | — | `([*shape], [*shape]) → [*shape]` |
| `Concat` | `dim=-1` | `([*a], [*b]) → [*c]` |
| `Reshape` | `*target_shape` | `[*source] → [*target]` |
| `Transpose` | `dim0, dim1` | `[*shape] → [*shape]` |

#### Attention

| Neuron | Parameters | Shape Signature |
|--------|------------|-----------------|
| `ScaledDotProductAttention` | — | `([b,s,d], [b,s,d], [b,s,d]) → [b,s,d]` |
| `MultiHeadSelfAttention` | `d_model, heads` | `[b, s, d_model] → [b, s, d_model]` |

### 7.2 Implementation References

#### Format

```neuroscript
impl: <source>,<module>/<class>
```

Examples:
- `impl: neuroscript_runtime,primitives/Linear`
- `impl: core,nn/Linear`
- `impl: torch,nn/Dropout`

#### Python Mapping

The implementation reference maps to Python imports:

```neuroscript
impl: neuroscript_runtime,primitives/Linear
```

Generates:
```python
from neuroscript_runtime.primitives.linear import Linear
```

#### External Implementations

Primitives can reference external PyTorch modules:

```neuroscript
neuron Dropout(p=0.1):
    in: [*shape]
    out: [*shape]
    impl: torch,nn/Dropout
```

### 7.3 Adding Custom Primitives

To add a new primitive:

1. **Define NeuroScript interface:**
   ```neuroscript
   neuron CustomLayer(dim):
       in: [*, dim]
       out: [*, dim * 2]
       impl: neuroscript_runtime,primitives/CustomLayer
   ```

2. **Implement Python class:**
   ```python
   # neuroscript_runtime/primitives/custom.py
   import torch.nn as nn

   class CustomLayer(nn.Module):
       def __init__(self, dim):
           super().__init__()
           self.proj = nn.Linear(dim, dim * 2)

       def forward(self, x):
           return self.proj(x)
   ```

3. **Register in stdlib registry** (optional for stdlib inclusion):
   ```rust
   // src/stdlib_registry.rs
   registry.insert(
       "CustomLayer".to_string(),
       ImplRef::Source {
           module: "neuroscript_runtime.primitives.custom".to_string(),
           class: "CustomLayer".to_string(),
       },
   );
   ```

---

## 8. VALIDATION

### 8.1 Validation Phases

The validator runs after parsing and before code generation:

```
Parse → IR → Validate → Codegen
                ↓
         Shape Inference
```

### 8.2 Validation Checks

#### 1. Existence Checks

Verify all referenced neurons exist:

```neuroscript
# ERROR: UnknownLayer not defined
in -> UnknownLayer(512) -> out
```

**Checks:**
- Neuron calls reference defined neurons or registered primitives
- Port references are valid for the neuron
- Variable references are in scope

#### 2. Arity Checks

Verify connection endpoint compatibility:

```neuroscript
# ERROR: Fork produces 2 outputs, tuple expects 3
in -> Fork() -> (a, b, c)
```

**Checks:**
- Tuple unpacking arity matches source outputs
- Multi-input neurons receive correct number of inputs
- Port counts match on both ends of connection

#### 3. Cycle Detection

Prevent circular dependencies:

```neuroscript
# ERROR: Cycle detected (A → B → C → A)
neuron A: graph: in -> B() -> out
neuron B: graph: in -> C() -> out
neuron C: graph: in -> A() -> out  # Cycle!
```

**Algorithm:**
- Build dependency graph
- Depth-first search for cycles
- Report all cycles found

**Note:** Self-edges within a single neuron are allowed (pipelines in graph).

#### 4. Shape Compatibility

Verify shapes across connections via inference:

```neuroscript
# ERROR: Linear expects [*, 512] but receives [*, 256]
in: [*, 256]
graph:
    in -> Linear(512, 1024) -> out  # Dimension mismatch!
```

**Checks:**
- Input shapes match neuron expectations
- Dimension variables unify consistently
- Expression dimensions are solvable
- Broadcasting rules are followed

#### 5. Match Validation

Validate match expression completeness and reachability:

```neuroscript
# ERROR: Non-exhaustive match (no catch-all)
in -> match:
    [*, 512]: Identity() -> out
    [*, 256]: Linear(256, 512) -> out
    # Missing catch-all pattern!
```

**Checks:**
- **Exhaustiveness**: Must have a catch-all pattern (`[*]`, `[*, d]`, etc.)
- **Reachability**: Later patterns not shadowed by earlier ones
- **Guard types**: Guard expressions are boolean
- **Dimension usage**: Captured dims used correctly

Example of shadowing:
```neuroscript
# WARNING: Second pattern is unreachable
in -> match:
    [*, d]: Linear(d, 512) -> out        # Catches everything
    [*, 256]: Identity() -> out          # Never reached!
```

#### 6. Binding Validation

Validate `let` and `set` bindings:

```neuroscript
# ERROR: Duplicate binding name
let:
    norm = LayerNorm(512)
    norm = RMSNorm(512)  # Duplicate!
```

**Checks:**
- No duplicate names in `let`/`set`
- `set` bindings are not recursive
- Binding arguments are resolvable

### 8.3 Error Reporting

Errors include:
- **Source spans**: Exact location in source file
- **Context**: Relevant code snippet
- **Suggestion**: How to fix (when possible)
- **Error code**: For documentation reference

Example:
```
Error: Arity mismatch in tuple unpacking
  ┌─ examples/error.ns:5:15
  │
5 │     in -> Fork() -> (a, b, c)
  │               ^^ produces 2 outputs
  │                   ^^^^^^^^^ but expects 3

Help: Fork produces 2 outputs (left, right).
      Either use (a, b) or use Fork3() for 3 outputs.
```

### 8.4 Validation Error Types

```rust
pub enum ValidationError {
    MissingNeuron { name, span },
    PortMismatch { expected, actual, span },
    CycleDetected { path, span },
    ArityMismatch { expected, actual, span },
    UnknownNode { name, span },
    NonExhaustiveMatch { span },
    UnreachableMatchArm { pattern, span },
    DuplicateBinding { name, first_span, second_span },
    InvalidRecursion { name, span },
    ShapeIncompatible { source, target, span },
    UnresolvableDimension { expr, span },
}
```

---

## 9. CODE GENERATION

### 9.1 Target: PyTorch nn.Module

NeuroScript compiles to PyTorch modules:

```python
import torch
import torch.nn as nn
from neuroscript_runtime.primitives.linear import Linear
from neuroscript_runtime.primitives.activations import GELU

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_0 = Linear(dim, dim * 4)
        self.gelu_1 = GELU()
        self.linear_2 = Linear(dim * 4, dim)

    def forward(self, x):
        # Shape: [*, dim]
        x = self.linear_0(x)  # [*, dim * 4]
        x = self.gelu_1(x)    # [*, dim * 4]
        x = self.linear_2(x)  # [*, dim]
        return x
```

### 9.2 Code Generation Phases

```
IR → Generator → PyTorch Code
      ├── Imports
      ├── __init__ (instantiation)
      └── forward (connections)
```

### 9.3 Import Generation

**Automatic import collection:**
- Scan all neuron calls in graph
- Lookup implementation in stdlib registry
- Deduplicate imports
- Generate import statements

**Example:**
```python
# From neurons used in graph
from neuroscript_runtime.primitives.linear import Linear
from neuroscript_runtime.primitives.activations import GELU
from neuroscript_runtime.primitives.norm import LayerNorm
```

### 9.4 Module Instantiation (`__init__`)

**Set bindings:**
```neuroscript
set:
    norm = LayerNorm(dim)
    ffn = FFN(dim, dim * 4)
```

Generates:
```python
def __init__(self, dim):
    super().__init__()
    self.norm = LayerNorm(dim)
    self.ffn = FFN(dim, dim * 4)
```

**Let bindings:**
```neuroscript
let:
    processor = MLP(512)
```

Generates:
```python
def __init__(self, ...):
    super().__init__()
    self._let_cache = {}  # Lazy instantiation dictionary
```

### 9.5 Forward Pass Generation

**Simple pipeline:**
```neuroscript
in -> Linear(dim, dim) -> GELU() -> out
```

Generates:
```python
def forward(self, x):
    x = self.linear_0(x)
    x = self.gelu_1(x)
    return x
```

**Named references:**
```neuroscript
in -> Linear(dim, dim) -> hidden
hidden -> GELU() -> activated
activated -> out
```

Generates:
```python
def forward(self, x):
    hidden = self.linear_0(x)
    activated = self.gelu_1(hidden)
    return activated
```

**Tuple unpacking:**
```neuroscript
in -> Fork() -> (left, right)
left -> Linear(dim, dim) -> processed
(processed, right) -> Add() -> out
```

Generates:
```python
def forward(self, x):
    left, right = self.fork_0(x)
    processed = self.linear_1(left)
    x = self.add_2(processed, right)
    return x
```

### 9.6 Match Expression Codegen

**Shape-based dispatch:**

```neuroscript
in -> match:
    [*, 512]: Identity() -> out
    [*, d] where d > 512: Linear(d, 512) -> out
    [*, d]: Linear(d, 512) -> out
```

Generates:
```python
def forward(self, x):
    # Pattern: [*, 512]
    if x.ndim == 2 and x.shape[-1] == 512:
        x = self.identity_0(x)
        return x

    # Pattern: [*, d] where d > 512
    if x.ndim == 2:
        d = x.shape[-1]
        if d > 512:
            # Lazy instantiation
            key = f'linear_{d}_512'
            if key not in self._let_cache:
                self._let_cache[key] = Linear(d, 512)
            x = self._let_cache[key](x)
            return x

    # Pattern: [*, d]
    if x.ndim == 2:
        d = x.shape[-1]
        key = f'linear_{d}_512'
        if key not in self._let_cache:
            self._let_cache[key] = Linear(d, 512)
        x = self._let_cache[key](x)
        return x
```

**Key features:**
- Runtime shape checking
- Dimension binding from tensor shape
- Guard evaluation after dimension binding
- Lazy module instantiation for captured dimensions

### 9.7 Shape Comments

Generated code includes shape comments:

```python
def forward(self, x):
    # Shape: [batch, seq, dim]
    x = self.norm(x)      # [batch, seq, dim]
    x = self.linear(x)    # [batch, seq, dim * 4]
    x = self.gelu(x)      # [batch, seq, dim * 4]
    return x
```

### 9.8 Variable Naming

**Automatic naming:**
- Modules: `<neuron>_<index>` (e.g., `linear_0`, `gelu_1`)
- Intermediate vars: Use reference name or `x`
- Tuple elements: Use unpacking names

**Collision avoidance:**
- Track used names
- Append suffix for collisions
- Reserved names: `self`, `super`, `x`, Python keywords

---

## 10. STANDARD LIBRARY

### 10.1 Library Organization

Located in `stdlib/` directory:

```
stdlib/
├── FFN.ns                    # Feed-forward networks
├── Residual.ns               # Skip connections
├── MultiHeadAttention.ns     # Attention mechanisms
├── TransformerBlock.ns       # Complete transformer layers
├── TransformerStack.ns       # Stacked transformers
├── MetaNeurons.ns            # Routing and composition
└── Sequential.ns             # Meta-neuron for repetition
```

### 10.2 Core Library Neurons

#### Feed-Forward Networks (FFN.ns)

```neuroscript
# Basic FFN with expansion
neuron FFN(dim, expansion):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            Linear(dim, expansion)
            GELU()
            Linear(expansion, dim)
            out

# FFN with dropout
neuron FFNWithDropout(dim, expansion, dropout=0.1):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            Linear(dim, expansion)
            GELU()
            Dropout(dropout)
            Linear(expansion, dim)
            Dropout(dropout)
            out
```

#### Residual Connections (Residual.ns)

```neuroscript
# Basic residual with any inner neuron
neuron Residual(inner_neuron):
    in: [*shape]
    out: [*shape]
    graph:
        in -> Fork() -> (main, skip)
        main -> inner_neuron -> processed
        (processed, skip) -> Add() -> out

# Pre-norm residual
neuron PreNormResidual(dim, inner_neuron):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> Fork() -> (main, skip)
        main ->
            LayerNorm(dim)
            inner_neuron
            processed
        (processed, skip) -> Add() -> out
```

#### Multi-Head Attention (MultiHeadAttention.ns)

```neuroscript
# Self-attention with multiple heads
neuron MultiHeadSelfAttention(d_model, heads):
    in: [batch, seq, d_model]
    out: [batch, seq, d_model]
    graph:
        in ->
            MultiHeadSelfAttention(d_model, heads)
            out
```

#### Transformer Block (TransformerBlock.ns)

```neuroscript
neuron TransformerBlock(d_model, heads, expansion=4):
    in: [batch, seq, d_model]
    out: [batch, seq, d_model]

    set:
        attn = MultiHeadSelfAttention(d_model, heads)
        ffn = FFN(d_model, d_model * expansion)
        norm1 = LayerNorm(d_model)
        norm2 = LayerNorm(d_model)

    graph:
        # Self-attention with residual
        in -> Fork() -> (attn_in, skip1)
        attn_in -> norm1 -> attn -> attn_out
        (attn_out, skip1) -> Add() -> residual1

        # FFN with residual
        residual1 -> Fork() -> (ffn_in, skip2)
        ffn_in -> norm2 -> ffn -> ffn_out
        (ffn_out, skip2) -> Add() -> out
```

#### Meta-Neurons (MetaNeurons.ns)

```neuroscript
# Route input to one of N branches based on shape
neuron ShapeRouter:
    in: [*shape]
    out: [*shape]
    graph:
        in -> match:
            [batch, seq, dim] where dim < 512:
                SmallModel(dim)
                out
            [batch, seq, dim]:
                LargeModel(dim)
                out

# Parallel processing with fan-out/fan-in
neuron Parallel(neuron_a, neuron_b):
    in: [*shape]
    out: [*shape]
    graph:
        in -> Fork() -> (branch_a, branch_b)
        branch_a -> neuron_a -> out_a
        branch_b -> neuron_b -> out_b
        (out_a, out_b) -> Concat() -> out
```

#### Sequential (Sequential.ns)

```neuroscript
# Meta-neuron for repeated layers
neuron Sequential(count, neuron_type: NeuronType):
    in: [*shape]
    out: [*shape]
    graph:
        in -> out  # Placeholder; expanded at compile time
```

**Status**: Syntax and detection implemented, compile-time expansion TODO.

**Intended usage:**
```neuroscript
neuron DeepTransformer(depth, d_model):
    let:
        stack = Sequential(depth, TransformerBlock)
    graph:
        in -> stack(d_model) -> out
```

Will expand to:
```python
class DeepTransformer(nn.Module):
    def __init__(self, depth, d_model):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

### 10.3 Using the Standard Library

**Import entire module:**
```neuroscript
use stdlib,FFN/*
```

**Import specific neurons:**
```neuroscript
use stdlib,Residual/PreNormResidual
use stdlib,TransformerBlock/TransformerBlock
```

**Compose stdlib neurons:**
```neuroscript
neuron MyModel(d_model):
    let:
        block = TransformerBlock(d_model, 8)
        residual = PreNormResidual(d_model, block)
    graph:
        in -> residual -> out
```

---

## 11. ERROR HANDLING

### 11.1 Error Philosophy

NeuroScript errors are:
- **Precise**: Source spans pinpoint exact location
- **Helpful**: Suggestions for common mistakes
- **Beautiful**: Colored terminal output with miette
- **Structured**: Typed errors for programmatic handling

### 11.2 Error Types by Phase

#### Lexer Errors

| Error | Cause | Example |
|-------|-------|---------|
| `UnexpectedChar` | Invalid character | `@` in source |
| `UnterminatedString` | Missing backtick | `` `hello `` (no closing `` ` ``) |
| `InvalidNumber` | Malformed numeric | `123.456.789` |
| `InconsistentIndent` | Indentation mismatch | 2 spaces then 3 spaces |

#### Parser Errors

| Error | Cause | Example |
|-------|-------|---------|
| `Expected` | Missing expected token | Missing `:` after `in` |
| `Unexpected` | Unexpected token | `out` when expecting `graph` |
| `DuplicateNeuron` | Same neuron defined twice | Two `neuron Linear` |

#### Validation Errors

| Error | Cause | Example |
|-------|-------|---------|
| `MissingNeuron` | Undefined neuron reference | `UnknownLayer(512)` |
| `PortMismatch` | Incompatible port shapes | `[*, 512]` → `[*, 256]` |
| `CycleDetected` | Circular dependency | A → B → A |
| `ArityMismatch` | Wrong port count | `Fork() -> (a, b, c)` (Fork has 2 outputs) |
| `UnknownNode` | Invalid node reference | `unknown_ref.port` |
| `NonExhaustiveMatch` | No catch-all pattern | Match without `[*, d]` |
| `UnreachableMatchArm` | Shadowed pattern | Specific after general |
| `DuplicateBinding` | Same name twice | `let: x = A(); x = B()` |
| `InvalidRecursion` | Illegal recursive binding | `set: rec = Neuron(rec)` |
| `ShapeIncompatible` | Shape mismatch | Expression dimension mismatch |
| `UnresolvableDimension` | Can't solve expression | `dim * x = 100` (x unknown, 100 not divisible) |

### 11.3 Error Examples

#### Detailed Error Message

```
Error: Shape incompatibility in connection
  ┌─ examples/shapes.ns:8:15
  │
7 │     in: [batch, seq, 512]
8 │         in -> Linear(256, 512) -> out
  │               ^^^^^^^^^^^^^^^^ expects input [*, 256]
  │
  = note: Input shape is [batch, seq, 512] but Linear expects [*, 256]
  = help: Linear's first parameter should match the last dimension of input
          Try: Linear(512, 512) to match input dimension
```

#### Multiple Errors

The validator collects all errors before reporting:

```
Error: Multiple validation errors found:

1. Missing neuron 'UnknownLayer'
  ┌─ examples/errors.ns:5:11
  │
5 │     in -> UnknownLayer(512) -> out
  │           ^^^^^^^^^^^^
  │
  = help: Did you mean: Linear, LayerNorm?

2. Arity mismatch in tuple unpacking
  ┌─ examples/errors.ns:7:15
  │
7 │     in -> Fork() -> (a, b, c)
  │               ^^ produces 2 outputs
  │                   ^^^^^^^^^ expects 3
  │
  = help: Fork produces 2 outputs. Use Fork3() for 3 outputs.

3. Non-exhaustive match expression
  ┌─ examples/errors.ns:10:11
  │
10│     in -> match:
11│         [*, 512]: Identity() -> out
  │         ^^^^^^^
  │
  = help: Add a catch-all pattern like [*, d] as the last arm
```

---

## 12. OPTIMIZATION

### 12.1 Implemented Optimizations

NeuroScript includes compile-time optimizations enabled by default.

#### Pattern Reordering

Reorder match arms for efficiency:

**Before:**
```neuroscript
match:
    [*, d]: Linear(d, 512) -> out           # General
    [*, 512]: Identity() -> out              # Specific
```

**After:**
```neuroscript
match:
    [*, 512]: Identity() -> out              # Specific first
    [*, d]: Linear(d, 512) -> out           # General last
```

**Benefits:**
- Specific patterns matched first (faster path)
- Reduces unnecessary runtime checks
- Improves generated code clarity

#### Dead Branch Elimination

Remove unreachable match arms:

**Before:**
```neuroscript
match:
    [*, d]: Linear(d, 512) -> out           # Catches all
    [*, 256]: Identity() -> out              # Unreachable
```

**After:**
```neuroscript
match:
    [*, d]: Linear(d, 512) -> out           # Only reachable pattern
```

**Analysis:**
- Detect pattern shadowing
- Mark arms with `is_reachable = false`
- Skip codegen for dead arms
- Emit warnings during compilation

### 12.2 CLI Optimization Flags

```bash
# Default: All optimizations enabled
neuroscript compile examples/model.ns

# Disable all optimizations
neuroscript compile examples/model.ns --no-optimize

# Disable only dead branch elimination
neuroscript compile examples/model.ns --no-dead-elim

# Verbose output shows optimization stats
neuroscript compile examples/model.ns --verbose
```

**Verbose output:**
```
Optimizations applied:
  - Pattern reordering: 3 match expressions
  - Dead branch elimination: 2 unreachable arms removed

Generated 45 lines of Python code.
```

### 12.3 Future Optimizations (Planned)

- **Graph simplification**: Merge adjacent neurons
- **Fusion**: Combine operations (e.g., Linear + GELU)
- **Constant folding**: Evaluate compile-time constants
- **Shape specialization**: Generate specialized code for known shapes
- **Memory planning**: Optimize tensor allocations

---

## 13. LANGUAGE FEATURES STATUS

### 13.1 Fully Implemented ✅

**Lexer:**
- [x] Indentation-aware tokenization
- [x] All keywords and operators
- [x] Literals (int, float, string, bool)
- [x] Comments
- [x] Error recovery with spans

**Parser:**
- [x] Use statements
- [x] Neuron definitions
- [x] Port specifications (inline and multi-line)
- [x] Primitive neurons (`impl:`)
- [x] Composite neurons (`graph:`)
- [x] Connections and pipelines
- [x] Tuple unpacking
- [x] Match expressions with guards
- [x] Let/set bindings
- [x] Shape expressions

**IR:**
- [x] Complete algebraic type system
- [x] Modular organization (`interfaces.rs`)
- [x] Display implementations for debugging

**Validator:**
- [x] Existence checks
- [x] Arity checks
- [x] Cycle detection
- [x] Shape inference integration
- [x] Match exhaustiveness
- [x] Reachability analysis
- [x] Binding validation

**Shape System:**
- [x] Five dimension types
- [x] Pattern matching
- [x] Shape algebra operations
- [x] Inference engine
- [x] Unification
- [x] Expression solving
- [x] BigUint arithmetic (no overflow)

**Codegen:**
- [x] PyTorch module generation
- [x] Import collection
- [x] `__init__` generation
- [x] `forward` pass generation
- [x] Match expression codegen
- [x] Let/set binding codegen
- [x] Lazy instantiation
- [x] Shape comments

**Standard Library:**
- [x] 26 registered primitives
- [x] Stdlib loading from `stdlib/` directory
- [x] 6 library modules (FFN, Residual, Attention, etc.)
- [x] 30+ composable neurons

**Tooling:**
- [x] CLI with 4 subcommands (parse, validate, compile, list)
- [x] Verbose mode for all commands
- [x] Optimization flags
- [x] Beautiful error messages (miette)
- [x] Snapshot testing (insta)

### 13.2 Partially Implemented ⏳

**Meta-Neurons:**
- [x] `NeuronType` parameter annotation
- [x] Detection of meta-neuron patterns
- [x] Sequential definition in stdlib
- [ ] Compile-time expansion of Sequential
- [ ] Higher-order neuron instantiation

**Type Inference:**
- [x] Dimension variable unification within neurons
- [ ] Global type inference across entire program
- [ ] Polymorphic neuron signatures

### 13.3 Planned 📋

**Language Features:**
- [ ] Loop constructs (`loop`, `for`)
- [ ] Conditional branches (`if`/`else`)
- [ ] First-class functions
- [ ] Generic neurons (beyond current shape polymorphism)
- [ ] Effects system (randomness, state, etc.)

**Backends:**
- [ ] ONNX export
- [ ] JAX backend
- [ ] TorchScript compilation
- [ ] Custom IR for multi-backend support

**Tooling:**
- [ ] LSP server
- [ ] Syntax highlighting
- [ ] REPL
- [ ] Package manager
- [ ] Documentation generator
- [ ] Graph visualization

**Optimizations:**
- [ ] Operation fusion
- [ ] Memory planning
- [ ] Kernel specialization
- [ ] Quantization hints

---

## 14. EXAMPLES

### 14.1 Hello World

```neuroscript
neuron HelloLinear(in_dim, out_dim):
    in: [*, in_dim]
    out: [*, out_dim]
    graph:
        in -> Linear(in_dim, out_dim) -> out
```

### 14.2 Multi-Layer Perceptron

```neuroscript
neuron MLP(dim, expansion=4):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            Linear(dim, dim * expansion)
            GELU()
            Linear(dim * expansion, dim)
            out
```

### 14.3 Residual Connection

```neuroscript
neuron Residual(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> Fork() -> (main, skip)
        main ->
            Linear(dim, dim * 4)
            GELU()
            Linear(dim * 4, dim)
            processed
        (processed, skip) -> Add() -> out
```

### 14.4 Multi-Head Attention

```neuroscript
neuron SimpleAttention(d_model, heads):
    in: [batch, seq, d_model]
    out: [batch, seq, d_model]

    set:
        mha = MultiHeadSelfAttention(d_model, heads)

    graph:
        in -> mha -> out
```

### 14.5 Transformer Block

```neuroscript
neuron TransformerBlock(d_model, heads, expansion=4):
    in: [batch, seq, d_model]
    out: [batch, seq, d_model]

    set:
        norm1 = LayerNorm(d_model)
        norm2 = LayerNorm(d_model)
        attn = MultiHeadSelfAttention(d_model, heads)
        ffn = FFN(d_model, d_model * expansion)

    graph:
        # Self-attention block
        in -> Fork() -> (attn_input, skip1)
        attn_input ->
            norm1
            attn
            attn_output
        (attn_output, skip1) -> Add() -> residual1

        # Feed-forward block
        residual1 -> Fork() -> (ffn_input, skip2)
        ffn_input ->
            norm2
            ffn
            ffn_output
        (ffn_output, skip2) -> Add() -> out
```

### 14.6 Adaptive Shape Handling

```neuroscript
neuron AdaptiveProjection(target_dim):
    in: [*, dim]
    out: [*, target_dim]
    graph:
        in -> match:
            [*, d] where d == target_dim:
                Identity()
                out
            [*, d] where d < target_dim:
                Linear(d, target_dim)
                out
            [*, d]:
                Linear(d, target_dim)
                out
```

### 14.7 Recursive Depth Control

```neuroscript
neuron RecursiveBlock(dim, depth):
    let:
        recurse = RecursiveBlock(dim, depth - 1)
    set:
        layer = MLP(dim)
    graph:
        in -> match:
            [*shape] where depth > 0:
                layer
                recurse
                out
            [*shape]:
                Identity()
                out
```

### 14.8 Multi-Branch Router

```neuroscript
neuron SizeBasedRouter(small_dim, large_dim):
    set:
        small_path = MLP(small_dim)
        large_path = DeepMLP(large_dim)
    graph:
        in -> match:
            [batch, seq, d] where d <= 512:
                small_path
                out
            [batch, seq, d]:
                large_path
                out
```

### 14.9 Using Standard Library

```neuroscript
use stdlib,FFN/FFN
use stdlib,Residual/PreNormResidual

neuron MyModel(d_model):
    let:
        ffn = FFN(d_model, d_model * 4)
        block = PreNormResidual(d_model, ffn)
    graph:
        in -> block -> out
```

### 14.10 Complete Model

```neuroscript
use stdlib,TransformerBlock/TransformerBlock

neuron GPT(vocab_size, d_model, heads, depth):
    set:
        embed = Embedding(vocab_size, d_model)
        pos_enc = LearnedPositionalEmbedding(2048, d_model)
        norm = LayerNorm(d_model)
        proj = Linear(d_model, vocab_size)

    let:
        stack = Sequential(depth, TransformerBlock)

    graph:
        # Embedding
        in -> embed -> embedded
        embedded -> pos_enc -> positioned

        # Transformer stack (once Sequential expansion is implemented)
        positioned -> stack(d_model, heads) -> transformed

        # Output projection
        transformed ->
            norm
            proj
            out
```

---

## 15. CLI USAGE

### 15.1 Installation

```bash
# Build release binary
cargo build --release

# Install Python runtime
pip install -e .

# Verify installation
./target/release/neuroscript --help
```

### 15.2 Commands

#### Parse

Display IR structure:

```bash
# Quiet mode (default)
./target/release/neuroscript parse examples/residual.ns

# Verbose mode (detailed IR)
./target/release/neuroscript parse --verbose examples/residual.ns
./target/release/neuroscript parse -v examples/residual.ns
```

**Output:**
- Program structure
- Neuron definitions
- Connection graph
- Shape information (verbose only)

#### Validate

Run validation checks:

```bash
# Standard validation
./target/release/neuroscript validate examples/residual.ns

# Verbose validation
./target/release/neuroscript validate --verbose examples/residual.ns
./target/release/neuroscript validate -v examples/residual.ns
```

**Checks:**
- Neuron existence
- Port arity
- Cycle detection
- Shape compatibility
- Match exhaustiveness

#### Compile

Generate PyTorch code:

```bash
# Auto-detect neuron name from filename
./target/release/neuroscript compile examples/residual.ns

# Specify neuron explicitly
./target/release/neuroscript compile examples/residual.ns --neuron ResidualBlock
./target/release/neuroscript compile examples/residual.ns -n ResidualBlock

# Output to file
./target/release/neuroscript compile examples/residual.ns -o residual.py
./target/release/neuroscript compile examples/residual.ns --output residual.py

# Verbose (show optimization stats)
./target/release/neuroscript compile examples/residual.ns --verbose
./target/release/neuroscript compile examples/residual.ns -v

# Disable optimizations
./target/release/neuroscript compile examples/residual.ns --no-optimize

# Disable only dead branch elimination
./target/release/neuroscript compile examples/residual.ns --no-dead-elim
```

**Features:**
- Pattern reordering (default on)
- Dead branch elimination (default on)
- Import generation
- Shape comments

#### List

Show all neurons in file:

```bash
# Basic listing
./target/release/neuroscript list examples/residual.ns

# Verbose (with connection details)
./target/release/neuroscript list --verbose examples/residual.ns
./target/release/neuroscript list -v examples/residual.ns
```

**Output:**
- Neuron names
- Parameter signatures
- Input/output ports
- Primitive vs. composite
- Connection graph (verbose only)

### 15.3 Common Workflows

#### Development Iteration

```bash
# 1. Check syntax
cargo check

# 2. Parse to verify structure
./target/release/neuroscript parse examples/mymodel.ns

# 3. Validate
./target/release/neuroscript validate examples/mymodel.ns

# 4. Compile
./target/release/neuroscript compile examples/mymodel.ns -o mymodel.py

# 5. Test generated code
source ~/.venv_ai/bin/activate
python -c "from mymodel import MyModel; print(MyModel(512))"
```

#### Testing

```bash
# Unit tests
cargo test

# Specific module
cargo test parser
cargo test validator
cargo test codegen

# Integration tests
cargo test --test integration_tests

# Snapshot review
cargo insta review

# All examples
./test_examples.sh
```

#### Debugging

```bash
# Verbose parse
./target/release/neuroscript parse -v examples/debug.ns

# Verbose validate (see inference details)
./target/release/neuroscript validate -v examples/debug.ns

# Verbose compile (optimization stats)
./target/release/neuroscript compile -v examples/debug.ns

# Disable optimizations for debugging
./target/release/neuroscript compile --no-optimize examples/debug.ns
```

---

## 16. IMPLEMENTATION NOTES

### 16.1 BigUint for Shape Arithmetic

Shapes use `num_bigint::BigUint` to prevent integer overflow:

```rust
use num_bigint::BigUint;

// Shape: [1000, 1000, 1000]
// Size: 1,000,000,000 elements
// Would overflow u32, but BigUint handles it
```

**Why:**
- Tensor sizes can be enormous (billions of elements)
- Dimension expressions can produce large values
- Prevents silent overflow bugs

### 16.2 Indentation-Aware Lexing

The lexer generates `Indent` and `Dedent` tokens:

```neuroscript
graph:          # Newline
  in -> out     # Indent + tokens + Newline
                # Dedent (at EOF or next unindented line)
```

**Rules:**
- Tab = 2 spaces
- Blank lines ignored
- Comment lines ignored
- Inconsistent indentation = error

### 16.3 Discriminant-Based Token Matching

Parser uses `std::mem::discriminant` for token kind comparison:

```rust
// Correct: Handles enum variants with data
if discriminant(&self.peek().kind) == discriminant(&TokenKind::Arrow) {
    // ...
}

// Incorrect: Fails for enums with data
if self.peek().kind == TokenKind::Arrow {
    // Won't compile if TokenKind has data variants
}
```

### 16.4 Modular Architecture

Clean phase separation:

```
src/
├── lexer/      # Tokenization
├── parser/     # IR construction
├── validator/  # Semantic checks
├── shape/      # Shape algebra and inference
├── codegen/    # PyTorch generation
└── stdlib_registry.rs  # Primitive lookup
```

**Benefits:**
- Each phase independently testable
- Clear phase boundaries
- Easy to add new passes

### 16.5 Error Collection

Validator collects all errors before reporting:

```rust
let mut errors = Vec::new();

// Check 1
if let Some(err) = check_existence() {
    errors.push(err);
}

// Check 2 (runs even if check 1 failed)
if let Some(err) = check_arity() {
    errors.push(err);
}

// Report all errors at once
if !errors.is_empty() {
    return Err(errors);
}
```

**Why:**
- Better developer experience (fix multiple errors at once)
- Faster iteration
- More information per compilation

### 16.6 Snapshot Testing

Uses `insta` crate for regression detection:

```rust
#[test]
fn test_parse_residual() {
    let input = read_file("examples/residual.ns");
    let program = parse(input).unwrap();
    insta::assert_yaml_snapshot!(program);
}
```

**Benefits:**
- Comprehensive regression coverage
- Human-readable diffs
- Easy to review changes
- Catches unintended changes

**Files:**
- Tests: `tests/integration_tests.rs`
- Snapshots: `tests/snapshots/*.snap`
- Review: `cargo insta review`

### 16.7 Python Runtime

NeuroScript requires a Python runtime package:

```bash
pip install -e .
```

Provides:
- `neuroscript_runtime.primitives.*`: Primitive implementations
- Base classes and utilities
- Shape helpers

Generated code imports from this package:
```python
from neuroscript_runtime.primitives.linear import Linear
```

**Important**: Always activate venv before testing:
```bash
source ~/.venv_ai/bin/activate
```

---

## Appendix A: Grammar (EBNF)

```ebnf
Program      ::= Use* NeuronDef+
Use          ::= 'use' Source ',' Path '/' Item
NeuronDef    ::= 'neuron' Name '(' Params? ')' ':' INDENT Ports Body DEDENT

Params       ::= Param (',' Param)* ','?
Param        ::= Name (':' Type)? ('=' Value)?
Type         ::= 'NeuronType'

Ports        ::= 'in' ':' PortSpec NEWLINE
                 'out' ':' PortSpec NEWLINE
PortSpec     ::= Shape                                    # Inline
               | INDENT NamedPort+ DEDENT                  # Multi-line
NamedPort    ::= Name ':' Shape NEWLINE

Body         ::= PrimitiveBody | CompositeBody
PrimitiveBody::= 'impl' ':' ImplRef NEWLINE
CompositeBody::= Bindings? 'graph' ':' INDENT Connection+ DEDENT

Bindings     ::= ('let' | 'set') ':' INDENT Binding+ DEDENT
Binding      ::= Name '=' Call NEWLINE

Connection   ::= Endpoint '->' Pipeline NEWLINE
Pipeline     ::= Endpoint ('->' Endpoint)*               # Inline
               | INDENT Endpoint+ DEDENT                  # Multi-line

Endpoint     ::= Ref | Tuple | Call | Match
Ref          ::= Name ('.' Name)?
Tuple        ::= '(' Ref (',' Ref)* ')'
Call         ::= Name '(' Args? ')'
Match        ::= 'match' ':' INDENT MatchArm+ DEDENT
MatchArm     ::= Shape Guard? ':' Pipeline NEWLINE
Guard        ::= 'where' Expr

Shape        ::= '[' DimExpr (',' DimExpr)* ']'
               | '[' '*' Name? ']'                        # Variadic
DimExpr      ::= Literal | Name | Expr | '*'
Expr         ::= DimExpr Op DimExpr
Op           ::= '+' | '-' | '*' | '/'

Args         ::= Value (',' Value)* ','?
Value        ::= Literal | Name | Expr
Literal      ::= Integer | Float | String | Boolean

Name         ::= [a-zA-Z_][a-zA-Z0-9_]*
Integer      ::= [0-9]+
Float        ::= [0-9]+ '.' [0-9]+
String       ::= '`' [^`]* '`'
Boolean      ::= 'true' | 'false'
Source       ::= Name
Path         ::= Name ('/' Name)*
Item         ::= Name | '*'
ImplRef      ::= Source ',' Path '/' Name
```

---

## Appendix B: Primitive Reference

See [Section 7: Primitives](#7-primitives) for complete list of 26 built-in primitives.

---

## Appendix C: Example Programs

See [Section 14: Examples](#14-examples) for 10 complete example programs.

---

## Appendix D: Further Reading

- **Project README**: `/README.md`
- **CLAUDE.md**: `/CLAUDE.md` (development guide)
- **Examples**: `/examples/*.ns` (126+ test cases)
- **Standard Library**: `/stdlib/*.ns` (6 library modules)
- **Tests**: `/tests/integration_tests.rs` (snapshot tests)
- **Source Code**: `/src/**/*.rs` (well-documented implementation)

---

**Document Version**: 0.1
**Last Updated**: December 2025
**NeuroScript Version**: Phase 2 Complete
