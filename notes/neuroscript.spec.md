# NeuroScript Language Specification v0.1

**Target Audience:** Developers and AI systems implementing or using NeuroScript

This document fully specifies the NeuroScript language as currently implemented, including syntax, semantics, type system, and standard library.

---

## Table of Contents

1. [Philosophy & Core Concepts](#philosophy--core-concepts)
2. [Lexical Structure](#lexical-structure)
3. [Syntax Grammar](#syntax-grammar)
4. [Type System & Shapes](#type-system--shapes)
5. [Neuron Definitions](#neuron-definitions)
6. [Graph Composition](#graph-composition)
7. [Standard Library](#standard-library)
8. [Validation Rules](#validation-rules)
9. [Code Examples](#code-examples)
10. [Implementation Notes](#implementation-notes)

---

## Philosophy & Core Concepts

### Everything is a Neuron

NeuroScript treats neural network components uniformly:

- **Primitive neurons** wrap external implementations (PyTorch modules, external APIs)
- **Composite neurons** define internal computation graphs
- **All neurons** have typed input/output ports with shape contracts
- **Composition** is the primary abstraction mechanism

### Design Goals

1. **Declarative** - Describe architecture, not training loops
2. **Compositional** - Build complex from simple via clean interfaces
3. **Type-safe** - Catch shape mismatches before runtime
4. **Readable** - Python-like syntax with explicit structure
5. **Portable** - Target multiple backends (PyTorch, ONNX, JAX)

### Key Abstractions

```neuroscript
Neuron = Primitive | Composite
Port = (name: string, shape: Shape)
Shape = [Dim, Dim, ...]
Dim = Literal(int) | Named(id) | Wildcard | Variadic(id) | Expr(dim op dim)
Graph = [Connection, ...]
Connection = source -> destination
```

---

## Lexical Structure

### Comments

```neuroscript
# Single-line comment
# Comments can appear anywhere

neuron Test:  # Inline comments allowed
  in: [*, dim]  # After any statement
  out: [*, dim]
  impl: core,nn/Identity
```

### Identifiers

```
identifier ::= [a-zA-Z_][a-zA-Z0-9_]*
```

**Keywords (reserved):**

```
neuron, use, in, out, impl, graph, match, where, external, and, or, true, false
```

### Literals

**Integers:**

```neuroscript
42
-17
0
1000000
```

**Floats:**

```neuroscript
3.14
-0.5
2.7e-3
```

**Strings (backtick-delimited):**

```neuroscript
`hello world`
`model-name-with-hyphens`
`path/to/file`
```

**Booleans:**

```neuroscript
true
false
```

### Operators

```
->    Arrow (pipeline)
:     Colon (type annotation)
,     Comma (separator)
.     Dot (port access)
/     Slash (path separator)
*     Star (multiplication, wildcard)
+     Plus (addition)
-     Minus (subtraction, negation)
==    Equal
!=    Not equal
<     Less than
>     Greater than
<=    Less than or equal
>=    Greater than or equal
=     Assignment (default values, kwargs)
```

### Delimiters

```
( )   Parentheses (calls, tuples, grouping)
[ ]   Brackets (shapes)
```

### Whitespace & Indentation

NeuroScript uses **significant whitespace** (Python-style):

- Indentation must be consistent (spaces or tabs, not mixed)
- Indent/dedent tokens track block structure
- Blank lines and comments are ignored for indentation purposes

**Indentation Example:**

```neuroscript
neuron Example:        # Start of neuron block
  in: [*, 512]         # Indented body
  out: [*, 512]        # Same indent level
  graph:               # Start of graph block
    in -> out          # Indented graph body
                       # Dedent ends blocks
```

---

## Syntax Grammar

### Complete Grammar (EBNF-style)

```ebnf
program ::= (use_stmt | neuron_def | NEWLINE)*

use_stmt ::= "use" identifier "," path NEWLINE
path ::= (identifier | "*") ("/" (identifier | "*"))*

neuron_def ::= "neuron" identifier params? ":" NEWLINE INDENT body DEDENT

params ::= "(" param ("," param)* ")"
param ::= identifier ("=" expr)?

body ::= (port | impl_stmt | graph_stmt)+

port ::= ("in" | "out") (identifier ":")? shape NEWLINE

shape ::= "[" (dim ("," dim)*)? "]"

dim ::= literal
      | identifier
      | "*" identifier?         # wildcard or variadic
      | dim binop dim           # expression
      | "-" dim                 # negation
      | "(" dim ")"             # grouping

binop ::= "+" | "-" | "*" | "/" | "<" | ">" | "<=" | ">=" | "==" | "!="

impl_stmt ::= "impl" ":" impl_ref NEWLINE

impl_ref ::= identifier "," path                    # source,path reference
           | "external" "(" kwargs ")"              # external API

graph_stmt ::= "graph" ":" NEWLINE INDENT connection+ DEDENT

connection ::= endpoint "->" pipeline NEWLINE
             | endpoint "->" NEWLINE INDENT endpoint+ DEDENT  # indented

pipeline ::= endpoint ("->" endpoint)*

endpoint ::= identifier ("(" call_args ")")?        # call or ref
           | identifier "." identifier              # port access
           | "(" identifier ("," identifier)* ")"   # tuple
           | match_expr                             # pattern match

match_expr ::= "match" ":" NEWLINE INDENT match_arm+ DEDENT

match_arm ::= shape ("where" expr)? ":" endpoint ("->" endpoint)* NEWLINE

call_args ::= (expr ("," expr)*)? ("," kwargs)?

kwargs ::= identifier "=" expr ("," identifier "=" expr)*

expr ::= literal
       | identifier
       | identifier "(" call_args ")"     # function call
       | expr binop expr
       | "-" expr                         # negation
       | "(" expr ")"                     # grouping

literal ::= INTEGER | FLOAT | STRING | "true" | "false"
```

### Parsing Notes

1. **No semicolons** - Newlines are significant statement terminators
2. **Indentation-based blocks** - `INDENT`/`DEDENT` tokens from lexer
3. **Left-to-right pipelines** - `a -> b -> c` means `c(b(a(x)))`
4. **Operator precedence:**
   - Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
   - Additive: `+`, `-`
   - Multiplicative: `*`, `/`
   - Unary: `-` (negation)

---

## Type System & Shapes

### Shape Specification

Shapes are lists of dimensions representing tensor shapes.

**Syntax:**

```neuroscript
[dim, dim, ...]
```

### Dimension Types

#### 1. Literal Dimensions

Concrete integer values:

```neuroscript
[512]              # 1D tensor of size 512
[32, 512]          # Batch of 32, dimension 512
[3, 224, 224]      # RGB image (CHW format)
```

#### 2. Named Dimensions

Variables that bind to actual values at instantiation:

```neuroscript
neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: ...

# When called: Linear(512, 256)
# Binds: in_dim=512, out_dim=256
```

#### 3. Wildcard `*`

Matches any single dimension (like `_` in pattern matching):

```neuroscript
in: [*, 512]       # Batch size is flexible, feature dim is 512
in: [*, *, dim]    # 2D batch structure + feature dim
```

#### 4. Variadic `*name`

Captures zero or more dimensions:

```neuroscript
in: [*batch, seq, dim]   # Batch can be [], [32], [8, 4], etc.
in: [*shape]             # Matches any shape whatsoever
```

**Variadic Rules:**

- Can appear at any position
- Only one variadic per shape allowed (current implementation)
- Commonly used for shape-polymorphic operations

#### 5. Dimension Expressions

Arithmetic on dimensions:

```neuroscript
neuron Expand(dim):
  in: [batch, dim]
  out: [batch, dim * 4]      # Quadruple the dimension
  impl: ...

neuron Halve(dim):
  in: [batch, dim]
  out: [batch, dim / 2]      # Halve the dimension
  impl: ...

neuron Offset(dim, delta):
  in: [batch, dim]
  out: [batch, dim + delta]  # Add offset
  impl: ...
```

**Supported operations:** `+`, `-`, `*`, `/`

### Shape Compatibility

Shapes are compatible if:

1. **Exact match:** `[32, 512]` ≡ `[32, 512]`
2. **Wildcard match:** `[*, 512]` ≡ `[32, 512]`
3. **Variadic match:** `[*batch, 512]` ≡ `[8, 4, 512]`
4. **Named match:** `[*, dim]` ≡ `[32, 512]` (binds `dim=512`)
5. **Expression match:** Expressions assumed compatible (requires inference)

### Empty Shape

```neuroscript
in: []      # Scalar (rank-0 tensor)
```

Represents a single scalar value.

---

## Neuron Definitions

### Anatomy of a Neuron

```neuroscript
neuron NeuronName(param1, param2=default):
  in [port_name]: [shape]
  out [port_name]: [shape]
  impl: implementation_reference
  # OR
  graph:
    # connection definitions
```

### Parameters

**Syntax:**

```neuroscript
neuron Name(required, optional=default, another=42):
```

**Rules:**

- Required parameters come before optional ones
- Default values can be: integers, floats, strings, booleans, names
- Parameters are substituted in shapes and expressions

**Example:**

```neuroscript
neuron Configurable(dim, dropout=0.1, bias=true):
  in: [*, dim]
  out: [*, dim]
  graph:
    in ->
      Linear(dim, dim, bias=bias)
      Dropout(p=dropout)
      out
```

### Ports

Ports define input/output interfaces.

#### Unnamed Ports (Default)

```neuroscript
in: [*, 512]
out: [*, 512]
```

Port name defaults to `"default"`.

#### Named Ports

```neuroscript
in query: [batch, seq, dim]
in key: [batch, seq, dim]
in value: [batch, seq, dim]
out: [batch, seq, dim]
```

**Usage:** Required for multi-port neurons.

#### Multiple Ports

```neuroscript
neuron Fork:
  in: [*shape]
  out a: [*shape]
  out b: [*shape]
  impl: core,builtin/Fork
```

Creates multiple output ports with distinct names.

### Implementation References

Two types: **source references** and **external APIs**.

#### Source References

Format: `source,path/to/implementation`

```neuroscript
impl: core,nn/Linear
impl: neuroscript_runtime.primitives.Linear
impl: models,transformers/BertLayer
```

**Interpretation:**

- `source`: Module/package name
- `path`: Relative path within source
- Maps to Python import via `stdlib_registry.rs`

#### External References

Format: `external(kwargs...)`

```neuroscript
impl: external(provider=`lmstudio`, model=`qwen3-vl-12b`)
impl: external(provider=`openai`, model=`gpt-4`)
```

**Usage:** Integrate external APIs or services as neurons.

### Primitive vs Composite

**Primitive:** Has `impl:` clause, wraps external code

```neuroscript
neuron GELU:
  in: [*shape]
  out: [*shape]
  impl: neuroscript_runtime.primitives.GELU
```

**Composite:** Has `graph:` clause, defines internal connections

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

---

## Graph Composition

Composite neurons define computation graphs via connections.

### Connection Syntax

**Inline pipeline:**

```neuroscript
graph:
  in -> A() -> B() -> C() -> out
```

**Indented pipeline (no intermediate arrows):**

```neuroscript
graph:
  in ->
    A()
    B()
    C()
    out
```

**Multiple connections:**

```neuroscript
graph:
  in -> A() -> intermediate
  intermediate -> B() -> out
```

### Endpoints

Endpoints are connection sources/destinations:

#### 1. Simple Reference

```neuroscript
in              # Input port
out             # Output port
intermediate    # Intermediate node
```

#### 2. Neuron Call

```neuroscript
Linear(512, 256)
GELU()
Dropout(p=0.1)
```

Creates a neuron instance with arguments.

#### 3. Port Access

```neuroscript
fork.left       # Access 'left' port of 'fork' node
fork.right
```

**Note:** Current syntax uses tuple unpacking instead (see below).

#### 4. Tuple (Multi-port)

**Unpacking multiple outputs:**

```neuroscript
Fork() -> (branch_a, branch_b)
```

**Packing multiple inputs:**

```neuroscript
(processed, skip) -> Add() -> out
```

**Rules:**

- Tuple arity must match port count
- Creates named intermediate nodes

#### 5. Match Expression

Pattern match on input shape:

```neuroscript
in -> match:
  [*, 512]: Identity() -> out
  [*, 256]: Linear(256, 512) -> out
  [*, d]: Linear(d, 512) -> out
```

**Semantics:**

- First matching pattern wins
- Can include `where` guards (expressions)
- Each arm is a pipeline

### Special Nodes

**Input node:** `in`

- Automatically available in all composite neurons
- Corresponds to neuron's input ports

**Output node:** `out`

- Automatically available in all composite neurons
- Corresponds to neuron's output ports

**Intermediate nodes:**

- Created via tuple unpacking or simple references
- Scoped to the graph

### Graph Examples

#### Residual Connection

```neuroscript
neuron Residual(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    # Fork input into two paths
    in -> Fork() -> (main, skip)
    
    # Process main path
    main -> MLP(dim) -> processed
    
    # Merge processed with skip
    (processed, skip) -> Add() -> out
```

#### Multi-Stage Pipeline

```neuroscript
neuron Pipeline(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> LayerNorm(dim) -> stage1
    stage1 -> Linear(dim, dim * 2) -> stage2
    stage2 -> GELU() -> stage3
    stage3 -> Linear(dim * 2, dim) -> out
```

#### Parallel Branches

```neuroscript
neuron ParallelPaths(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Fork() -> (path_a, path_b)
    
    path_a ->
      Linear(dim, dim)
      GELU()
      result_a
    
    path_b ->
      LayerNorm(dim)
      result_b
    
    (result_a, result_b) -> Add() -> out
```

---

## Standard Library

NeuroScript ships with a hybrid standard library:

### Level 0: Primitives (Python)

Located in `neuroscript_runtime/primitives/`:

**Core Operations:**

- `Linear(in_features, out_features, bias=True)`

**Activations:**

- `GELU(approximate='none')`
- `ReLU(inplace=False)`
- `Tanh()`
- `Sigmoid()`
- `SiLU()`
- `Softmax(dim=-1)`

**Normalization:**

- `LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`
- `RMSNorm(dim, eps=1e-6)`
- `GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)`

**Regularization:**

- `Dropout(p=0.5, inplace=False)`
- `DropPath(drop_prob=0.1, scale_by_keep=True)`
- `DropConnect(p=0.5)`

**Embeddings:**

- `Embedding(vocab_size, embedding_dim, padding_idx=None, ...)`
- `PositionalEncoding(d_model, max_len=5000, dropout=0.1)`
- `LearnedPositionalEmbedding(max_len, d_model)`

### Level 1+: Composites (NeuroScript)

Located in `stdlib/`:

**Feed-Forward Networks (`FFN.ns`):**

- `FFN(dim, expansion)` - Standard two-layer FFN
- `FFNWithHidden(in_dim, hidden_dim, out_dim)`
- `GatedFFN(dim, expansion)` - GLU-style gating

**Residual Connections (`Residual.ns`):**

- `Residual(f)` - Generic residual wrapper
- `PreNormResidual(f, dim)` - LayerNorm before transformation
- `PostNormResidual(f, dim)` - LayerNorm after addition
- `HighwayResidual(dim, f)` - Gated residual
- `ResidualFFN(dim, expansion)` - Concrete FFN residual

**Multi-Head Attention (`MultiHeadAttention.ns`):**

- `ScaledDotProductAttention(d_k)` - Single-head attention
- `MultiHeadSelfAttention(d_model, num_heads)` - Standard MHA
- `MultiHeadCrossAttention(d_model, num_heads)` - Encoder-decoder attention
- `GroupedQueryAttention(d_model, num_query_heads, num_kv_heads)` - GQA
- `MultiQueryAttention(d_model, num_heads)` - MQA

**Transformer Blocks (`TransformerBlock.ns`):**

- `TransformerBlock(d_model, num_heads, d_ff)` - Pre-norm GPT-style
- `PostNormTransformerBlock(d_model, num_heads, d_ff)` - Post-norm
- `TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)` - BERT-style
- `TransformerDecoderBlock(d_model, num_heads, d_ff)` - Causal decoder
- `SimpleTransformerBlock(dim)` - Minimal test block

**Transformer Stacks (`TransformerStack.ns`):**

- `TransformerStack(num_layers, d_model, num_heads, d_ff)` - N identical layers
- `SmallTransformerStack(d_model, num_heads, d_ff)` - 3 layers explicit
- `GPTStack(num_layers, d_model, num_heads, d_ff)` - With final LayerNorm
- `BERTEncoder(num_layers, d_model, num_heads, d_ff)` - Bidirectional
- `EncoderDecoderTransformer(...)` - Full seq2seq architecture
- `DepthScaledTransformerStack(...)` - Depth-scaled for stability

**Meta-Neurons (`MetaNeurons.ns`):**

- `Sequential(neurons)` - Chain of neurons
- `Parallel(neurons)` - Parallel execution
- `Concatenate(dim)` - Tensor concatenation
- `Add()`, `Multiply()` - Element-wise ops
- `Split(dim, num_splits)` - Split tensor
- `Reshape(target_shape)`, `Transpose(dim0, dim1)`
- `Identity()` - Pass-through
- `Repeat(f, n)` - Recurrent application
- `Fork(num_paths)`, `Merge(operation)`, `Switch(...)`

### Registry System

Primitives are registered in `src/stdlib_registry.rs`:

```rust
pub struct ImplRef {
    pub module_path: String,  // e.g., "neuroscript_runtime.primitives.linear"
    pub class_name: String,   // e.g., "Linear"
    pub description: String,
}

let registry = StdlibRegistry::new();
registry.lookup("Linear");  // Returns ImplRef
registry.generate_imports(&["Linear", "GELU"]);  // Python imports
```

**Usage in codegen:**

- Map `impl: ...` references to actual Python imports
- Generate `from neuroscript_runtime.primitives.linear import Linear`

---

## Validation Rules

The validator (`src/validator.rs`) enforces correctness before codegen.

### 1. Neuron Existence

**Rule:** All referenced neurons must be defined.

**Example violation:**

```neuroscript
graph:
  in -> UndefinedNeuron() -> out  # Error: UndefinedNeuron not found
```

### 2. Arity Matching

**Rule:** Tuple arity must match port count.

**Example violations:**

```neuroscript
# Fork produces 2 outputs
graph:
  Fork() -> (a, b, c)  # Error: Expected 2, got 3

# Add requires 2 inputs
graph:
  (a) -> Add() -> out  # Error: Expected 2, got 1
```

### 3. Shape Compatibility

**Rule:** Connected ports must have compatible shapes.

**Compatible:**

```neuroscript
[*, 512] -> [*, 512]           # Exact match
[*, dim] -> [*, 512]           # Named matches literal
[*, 512] -> [*batch, 512]      # Variadic matches wildcard
```

**Incompatible:**

```neuroscript
[*, 512] -> [*, 256]           # Literal mismatch
[batch, seq, dim] -> [*, dim]  # Rank mismatch
```

### 4. No Cycles

**Rule:** Dependency graph must be acyclic.

**Example violation:**

```neuroscript
graph:
  A() -> B() -> intermediate
  intermediate -> A() -> out   # Error: Cycle A -> B -> A
```

**Exception:** Self-edges within single connection are allowed (e.g., `Linear -> Linear` in one pipeline).

### Valid Patterns

**Residual (no cycle):**

```neuroscript
graph:
  in -> Fork() -> (main, skip)
  main -> Process() -> processed
  (processed, skip) -> Add() -> out
  # Valid: No cycle, just parallel paths
```

### Running Validation

```bash
./target/release/neuroscript --validate examples/residual.ns
```

Output:

```
✓ Program is valid!
```

Or:

```
✗ Validation failed with 2 errors:
  Neuron 'Undefined' not found (in Composite)
  Arity mismatch: expected 2 ports, got 1 (in Composite: tuple unpacking)
```

---

## Code Examples

### Complete Residual Block

```neuroscript
use core,nn/*

neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear

neuron GELU:
  in: [*shape]
  out: [*shape]
  impl: core,activations/GELU

neuron Add:
  in left: [*shape]
  in right: [*shape]
  out: [*shape]
  impl: core,builtin/Add

neuron Fork:
  in: [*shape]
  out a: [*shape]
  out b: [*shape]
  impl: core,builtin/Fork

neuron MLP(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in ->
      Linear(dim, dim * 4)
      GELU()
      Linear(dim * 4, dim)
      out

neuron Residual(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Fork() -> (main, skip)
    main -> MLP(dim) -> processed
    (processed, skip) -> Add() -> out
```

### GPT-2 Small Architecture

```neuroscript
neuron GPT2Small(vocab_size):
    in: [batch, seq]
    out: [batch, seq, vocab_size]
    graph:
        # Embedding + positional encoding
        in ->
            Embedding(vocab_size, 768)
            PositionalEncoding(768, max_len=1024)
            embedded

        # 12 transformer layers
        embedded ->
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            features

        # Final projection
        features ->
            LayerNorm(768)
            Linear(768, vocab_size)
            out
```

### Pattern Matching

```neuroscript
neuron AdaptiveProjection:
  in: [*, dim]
  out: [*, 512]
  graph:
    in -> match:
      [*, 512]: Identity() -> out
      [*, 256]: Linear(256, 512) -> out
      [*, 1024]: Linear(1024, 512) -> out
      [*, d] where d > 512: 
        Linear(d, 512)
        out
      [*, d]:
        Linear(d, 512)
        out
```

### Multi-Port Composition

```neuroscript
neuron Attention(d_model):
  in query: [batch, seq_q, d_model]
  in key: [batch, seq_k, d_model]
  in value: [batch, seq_k, d_model]
  out: [batch, seq_q, d_model]
  graph:
    # Project Q, K, V
    query -> Linear(d_model, d_model) -> q
    key -> Linear(d_model, d_model) -> k
    value -> Linear(d_model, d_model) -> v
    
    # Compute attention (placeholder)
    (q, k, v) -> ScaledDotProductAttention(d_model) -> attended
    
    # Output projection
    attended -> Linear(d_model, d_model) -> out
```

---

## Implementation Notes

### For Developers

#### Lexer (`src/lexer.rs`)

- Tracks indentation with stack for `INDENT`/`DEDENT` tokens
- Handles comments by skipping to end of line
- Backtick strings can contain any characters
- Negative numbers via unary minus operator

#### Parser (`src/parser.rs`)

- Recursive descent with operator precedence
- Indented pipelines require `source ->` then indented steps (no arrows)
- Tuple syntax requires explicit parentheses
- Match expressions have their own block structure

#### IR (`src/ir.rs`)

- Algebraic types map cleanly to Rust enums
- `Shape` and `Dim` are recursive structures
- `Connection` links `Endpoint` nodes
- `NeuronBody` distinguishes primitive vs composite

#### Validator (`src/validator.rs`)

- Builds symbol table tracking intermediate nodes
- Resolves endpoints to port signatures
- DFS cycle detection with path reconstruction
- Handles tuple unpacking arity

#### Shape Algebra (`src/shape_algebra.rs`)

- Uses `BigUint` to prevent overflow on large products
- Pattern matching with wildcards, literals, rest
- Broadcasting, reshaping, tiling checks
- Axis refinement/coarsening operations

### For AI Systems

When generating NeuroScript code:

1. **Always define dependencies first** - Reference only already-defined neurons
2. **Match port counts in tuples** - `Fork() -> (a, b)` requires Fork to have 2 outputs
3. **Use indented pipelines for readability** - Multi-step transformations clearer indented
4. **Respect shape contracts** - Dimension expressions must make mathematical sense
5. **Start simple, compose up** - Build primitives → composites → architectures
6. **Leverage stdlib** - Reuse standard components rather than reimplementing

**Common Patterns:**

```neuroscript
# Residual connection pattern
in -> Fork() -> (main, skip)
main -> Transform() -> transformed
(transformed, skip) -> Add() -> out

# Multi-head pattern
in -> (q, k, v)
q -> ProjectQ() -> query
k -> ProjectK() -> key  
v -> ProjectV() -> value
(query, key, value) -> Attention() -> out

# Sequential transformation
in ->
  Normalize()
  Transform()
  Activate()
  Project()
  out
```

---

## Appendix: Reserved for Future Features

These features are planned but not yet implemented:

### Higher-Order Neurons

```neuroscript
# Neuron as parameter
neuron Repeat(f: Neuron, n: int):
  in: [*shape]
  out: [*shape]
  graph:
    # Apply f n times
```

### List Parameters

```neuroscript
neuron Sequential(layers: list[Neuron]):
  in: [*shape]
  out: [*shape]
  graph:
    # Chain layers
```

### Loop Constructs

```neuroscript
graph:
  for i in range(num_layers):
    current -> TransformerBlock(dim) -> current
```

### Type Annotations

```neuroscript
neuron Linear(in_dim: int, out_dim: int):
  ...
```

### Dynamic Shapes

```neuroscript
# Runtime-determined dimensions
in: [batch, ?seq, dim]
```

---

## Changelog

### **v0.1 (Current)**

- Core language: lexer, parser, IR, validator
- Shape algebra with pattern matching
- Standard library (primitives + composites)
- Python runtime package
- Comprehensive test suite

### **Upcoming**

- v0.2: Codegen → PyTorch
- v0.3: Shape inference
- v0.4: Advanced features (higher-order, loops)
- v0.5: Multiple backends

---

## References

- **Repository:** `neuroscript-rs/`
- **Examples:** `examples/*.ns`
- **Standard Library:** `stdlib/*.ns`
- **Tests:** `cargo test`, `./test_examples.sh`
- **Runtime:** `neuroscript_runtime/`

---

## **End of Specification**

This document describes NeuroScript as implemented in commit [current]. For the latest updates, see the repository README and source code.
