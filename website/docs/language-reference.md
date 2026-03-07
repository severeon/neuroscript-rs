---
sidebar_position: 2
---

# Language Reference

Complete syntax reference for NeuroScript. This page covers every language construct — if you need to write or understand `.ns` files, start here.

## Program Structure

A NeuroScript program is a collection of neuron definitions, optionally preceded by `use` imports:

```neuroscript
use core,nn/*                    # import all from core nn library
use core,attention/ScaledDotProductAttention  # import specific neuron

/// Doc comment for MyNeuron
neuron MyNeuron(param1, param2):
    in: [*batch, dim]
    out: [*batch, dim]
    graph:
        in -> Linear(dim, dim) -> out
```

## Comments

```neuroscript
# Inline comment — use anywhere
/// Doc comment — must precede a neuron definition
/// Can span multiple lines
///
/// Blank lines within doc comments are allowed
```

**Important:** `//` is NOT valid syntax. Always use `#` for inline comments.

## Neurons

Neurons are the fundamental unit. Every neuron has a name, optional parameters, input/output port declarations, and a body.

### Primitive Neurons

Wrap external implementations (e.g., PyTorch modules):

```neuroscript
neuron Linear(in_dim, out_dim):
    in: [*batch, in_dim]
    out: [*batch, out_dim]
    impl: core,nn/Linear
```

The `impl:` field uses comma-separated provider and slash-separated path: `<provider>,<library>/<class>`.

### Composite Neurons

Define dataflow graphs that connect other neurons:

```neuroscript
neuron FFN(dim, expansion):
    in: [*batch, seq, dim]
    out: [*batch, seq, dim]
    graph:
        in ->
            Linear(dim, dim * expansion)
            GELU()
            Linear(dim * expansion, dim)
            out
```

### Parameters

```neuroscript
neuron Basic(dim, heads):             # positional parameters
neuron WithDefaults(dim, layers=10):  # default values
neuron HigherOrder(block: Neuron):    # neuron type parameter
neuron Combined(block: Neuron, dim, layers=6):  # mixed
```

Parameters with `: Neuron` type annotation accept neuron constructors as arguments, enabling generic architectures:

```neuroscript
neuron Stack(block: Neuron, d_model, count=6):
    in: [*, seq, d_model]
    out: [*, seq, d_model]
    context:
        blocks = unroll(count):
            layer = block(d_model)
    graph:
        in -> blocks -> out
```

## Ports

### Input/Output Declarations

```neuroscript
# Default ports (named "default" internally)
in: [batch, seq, dim]
out: [batch, seq, dim]

# Named ports
in left: [*shape]
in right: [*shape]
out a: [*shape]
out b: [*shape]

# Variadic input port — accepts any number of inputs as a tuple
in *inputs: [*shape]
```

**Variadic port rules:**
- Only input ports can be variadic (not output)
- A neuron with a variadic port must have exactly one `in` declaration
- Must have an explicit name (e.g., `*inputs`)
- Each tuple element is validated against the port's shape individually

### Port References

```neuroscript
in                  # default input port
out                 # default output port
fork.left           # named port on a binding
fork.a              # another named port
```

## Shapes

Tensor shapes are first-class citizens with compile-time validation.

### Shape Syntax

```neuroscript
[512, 256]              # literal dimensions
[batch, seq, dim]       # named dimensions (variables)
[*, dim]                # single wildcard dimension
[*batch, seq, dim]      # variadic wildcard (any number of leading dims)
[dim * 4]               # arithmetic expression
[seq - 1]               # subtraction
[dim / heads]           # division
```

### Shape Expressions

Dimension expressions support `+`, `-`, `*`, `/` with standard precedence. The shape solver handles simple single-unknown equations.

### Shape Inference

The compiler automatically infers and validates tensor shapes across all connections. Named dimensions create equivalence constraints:

```neuroscript
# 'dim' in the input shape must equal 'dim' in Linear's input
neuron Example(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> Linear(dim, dim) -> out
```

## Connections and Pipelines

Connections define dataflow between neurons using the `->` operator.

### Basic Connections

```neuroscript
graph:
    in -> Linear(dim, dim) -> out           # single line
    in -> Linear(dim, 512) -> ReLU() -> out # chained

    # Multi-line pipeline (indentation continues the chain)
    in ->
        Linear(dim, 512)
        ReLU()
        Linear(512, dim)
        out
```

### Implicit Fork (Preferred)

Split a single output into multiple named bindings:

```neuroscript
graph:
    in -> (main, skip)                      # fork input into two paths
    main -> Linear(dim, dim) -> processed
    (processed, skip) -> Add() -> out       # join paths
```

### Explicit Fork

Use `Fork()` or `Fork3()` when you need named port access:

```neuroscript
graph:
    in -> Fork() -> f
    f.left -> Linear(dim, dim) -> a
    f.right -> Linear(dim, dim) -> b
    (a, b) -> Add() -> out
```

### Tuple Assembly

Combine multiple outputs into a tuple for multi-input neurons:

```neuroscript
(processed, skip) -> Add() -> out
(a, b, c) -> Concat(1) -> out
```

## Context Bindings

The `context:` block pre-instantiates neurons for reuse in the graph. Bindings use `=` for assignment.

```neuroscript
neuron MyBlock(dim):
    in: [*, dim]
    out: [*, dim]
    context:
        linear1 = Linear(dim, dim * 4)
        linear2 = Linear(dim * 4, dim)
        norm = LayerNorm(dim)
    graph:
        in -> norm -> linear1 -> GELU() -> linear2 -> out
```

### Annotations

```neuroscript
context:
    regular = Linear(dim, dim)              # instantiated immediately
    @lazy heavy = TransformerBlock(dim)     # instantiated only when used
    @static shared = LayerNorm(dim)         # shared across instances
    @global vocab = Embedding(vocab_size)   # global singleton
```

**`@lazy`**: Deferred instantiation — the neuron is only created when the binding is actually reached during forward execution. Required for recursive bindings where arguments change per call.

**`@static`**: Shared across all instances of the enclosing neuron — useful for parameter-free modules like normalization.

**`@global`**: A process-wide singleton — all neurons that reference this binding share the same instance. Useful for shared vocabularies or embeddings.

`@lazy` is required for recursive bindings — the arguments that change must be expressions:

```neuroscript
context:
    @lazy recurse = RecursiveStack(dim, depth - 1)
```

### Unroll Blocks

Repeat a binding pattern multiple times with `unroll(count):`:

```neuroscript
context:
    blocks = unroll(num_layers):
        block = TransformerBlock(d_model, num_heads, d_ff)
```

Each iteration creates a unique instance. The aggregate name (`blocks`) can be used in the graph as a sequential pipeline.

## Match Expressions

Route tensors based on their shape at runtime.

### Data-Threading Match

```neuroscript
graph:
    in -> match: ->
        [batch, d] where d > 1024:         # shape pattern with guard
            Linear(d, 512) ->
            Linear(512, out_dim)
        [batch, d] where d > 256:           # second arm
            Linear(d, out_dim)
        [batch, d]:                         # fallback (no guard)
            Linear(d, out_dim)
    -> out
```

**Match arm syntax:** `[pattern] where guard: pipeline`

- **Pattern:** Shape pattern with dimension captures (e.g., `d` captures the actual dimension value)
- **Guard** (optional): Boolean expression using captured dimensions
- **Pipeline:** The processing to apply when matched

### Inline Match Arms

```neuroscript
in -> match: ->
    [*, d] where d > 512: Linear(d, 256)
    [*, d]: Linear(d, d)
-> out
```

### Contract Match (Compile-Time Dispatch)

Dispatch based on a neuron parameter's port shapes, resolved at compile time:

```neuroscript
# Selects wiring strategy based on the block's input/output shapes
neuron SmartStack(block: Neuron, d_model, count=6):
    in: [*, seq, d_model]
    out: [*, seq, d_model]
    context:
        blocks = unroll(count):
            layer = block(d_model)
    graph:
        in ->
            match(block):
                # Block expects sequence dimension — pass through directly
                in [*, seq, d_model] -> out [*, seq, d_model]:
                    blocks
                    out
                # Block operates per-token — reshape around it
                in [*, d_model] -> out [*, d_model]:
                    blocks
                    out
```

## Conditionals

Route based on parameter values (compile-time branching).

### Inline Syntax

```neuroscript
graph:
    in ->
        if has_pool: pool
        else: Identity()
        out
```

### Block Syntax

```neuroscript
graph:
    in ->
        if depth > 0:
            Linear(dim, dim)
            GELU()
            recurse
        elif depth == 0:
            Linear(dim, dim)
        else: Identity()
        out
```

Guards support comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`.

## Fat Arrow (`=>`) Shape Transform

Inline reshape/view operator that compiles to `torch.reshape()`:

Dimension binding (`name=expr`) assigns a name to the result of an expression, making it available in subsequent reshape steps.

```neuroscript
# Basic reshape: [batch, seq, dim] -> [batch, seq, heads, dim/heads]
# dh=dim/heads creates a new named dimension 'dh' with value dim/heads
in => [batch, seq, heads, dh=dim/heads] -> out

# Transpose via reshape: [batch, seq, heads, dh] -> [batch, heads, seq, dh]
in => [batch, seq, heads, dh=dim/heads] => [batch, heads, seq, dh] -> out

# Flatten: [b, c, h, w] -> [b, c, h*w] -> [b, h*w, c]
in => [b, c, hw=h*w] => [b, hw, c] -> out

# Flatten tail dimensions
in => [b, others] -> out
```

Fat arrow can appear anywhere in a pipeline as a reshape step. Dimension binding (`dh=dim/heads`) creates new named dimensions from expressions.

## Use/Import Declarations

```neuroscript
use core,nn/*                                    # wildcard import
use core,nn/Linear                               # specific import
use core,attention/ScaledDotProductAttention      # from attention library
```

Import path format: `<provider>,<library>/<neuron>` or `<provider>,<library>/*`.

Wildcard imports (`/*`) make all neurons in that library available without qualification. Specific imports are preferred when you only need a few neurons to keep dependencies explicit.

## Complete Examples

### Residual Connection

```neuroscript
neuron Residual(dim):
    in: [*shape, dim]
    out: [*shape, dim]
    graph:
        in -> (main, skip)
        main ->
            Linear(dim, dim)
            ReLU()
            Linear(dim, dim)
            processed
        (processed, skip) -> Add() -> out
```

### Transformer Block

```neuroscript
neuron TransformerBlock(dim, num_heads, d_ff):
    in: [*batch, seq, dim]
    out: [*batch, seq, dim]
    graph:
        in -> (skip1, attn_path)

        attn_path ->
            LayerNorm(dim)
            MultiHeadSelfAttention(dim, num_heads)
            Dropout(0.1)
            attn_out

        (skip1, attn_out) -> Add() -> attn_residual

        attn_residual -> (skip2, ffn_path)

        ffn_path ->
            LayerNorm(dim)
            FFN(dim, d_ff)
            Dropout(0.1)
            ffn_out

        (skip2, ffn_out) -> Add() -> out
```

### Recursive Architecture

```neuroscript
neuron RecursiveStack(dim, depth):
    in: [*, dim]
    out: [*, dim]
    context:
        @lazy recurse = RecursiveStack(dim, depth - 1)
    graph:
        in ->
            if depth > 0:
                Linear(dim, dim)
                GELU()
                recurse
            else: Identity()
            out
```

### Multi-Head Reshape with Fat Arrow

```neuroscript
neuron MultiHeadReshape(dim, heads):
    in: [batch, seq, dim]
    out: [batch, heads, seq, dim / heads]
    graph:
        in => [batch, seq, heads, dh=dim/heads] => [batch, heads, seq, dh] -> out
```

### Variadic Concat

```neuroscript
neuron ConcatNorm(dim):
    in *inputs: [*shape]
    out: [*, dim]
    graph:
        in -> Concat(1) -> LayerNorm(dim) -> out
```

### Shape-Based Routing

```neuroscript
neuron DimensionRouter(out_dim):
    in: [batch, in_dim]
    out: [batch, out_dim]
    graph:
        in -> match: ->
            [batch, d] where d > 1024:
                Linear(d, 512) ->
                Linear(512, out_dim)
            [batch, d] where d > 256:
                Linear(d, out_dim)
            [batch, d]:
                Linear(d, out_dim)
        -> out
```

## Primitive Catalog (Quick Reference)

For detailed documentation on each primitive, see [Primitives](/docs/primitives).

### Basics
- `Linear(in_dim, out_dim)` — fully connected layer

### Activations
- `ReLU()`, `GELU()`, `SiLU()`, `Sigmoid()`, `Tanh()`, `Softmax(dim)`, `ELU(alpha)`, `PReLU()`, `Mish()`

### Convolutions
- `Conv1d(in_ch, out_ch, kernel)`, `Conv2d(...)`, `Conv3d(...)`
- `DepthwiseConv(channels, kernel)`, `SeparableConv(in_ch, out_ch, kernel)`
- `TransposedConv(in_ch, out_ch, kernel)`

### Normalization
- `LayerNorm(dim)`, `BatchNorm(features)`, `GroupNorm(groups, channels)`
- `InstanceNorm(features)`, `RMSNorm(dim)`

### Pooling
- `MaxPool(kernel)`, `AvgPool(kernel)`, `GlobalAvgPool()`, `GlobalMaxPool()`
- `AdaptiveAvgPool(output_size)`, `AdaptiveMaxPool(output_size)`

### Attention
- `ScaledDotProductAttention(dim, num_heads)`
- `MultiHeadSelfAttention(dim, num_heads)`

### Embeddings
- `Embedding(vocab_size, dim)`, `PositionalEncoding(dim, max_len)`
- `LearnedPositionalEmbedding(max_len, dim)`, `RotaryEmbedding(dim)`

### Structural
- `Fork()`, `Fork3()` — explicit split (prefer implicit fork with tuples)
- `Concat(dim)`, `Split(dim, sections)`, `Add()`, `Multiply()`
- `Reshape(shape)`, `Flatten(start, end)`, `Transpose(dim0, dim1)`
- `Slice(dim, start, end)`, `Pad(padding)`

### Operations
- `MatMul()`, `Einsum(equation)`, `Bias(dim)`, `Scale(dim)`, `Identity()`

### Regularization
- `Dropout(p)`, `DropConnect(p)`, `DropPath(p)`

## Standard Library (Composite Neurons)

For full documentation, see [Standard Library](/docs/stdlib).

**Feed-Forward:** FFN, GatedFFN, GLU, GeGLU, SwiGLU

**Residual:** Residual, PreNormResidual, PostNormResidual, DenseConnection, HighwayConnection

**Attention:** MultiHeadAttention, MultiQueryAttention, GroupedQueryAttention, CrossAttention, RelativePositionBias

**Transformer:** TransformerBlock, TransformerEncoderBlock, TransformerDecoderBlock, TransformerStack

**Vision:** PatchEmbedding, ViTBlock, InceptionBlock, SEBlock

**ConvNets:** ResNetBasicBlock, BottleneckBlock, ResNeXtBlock, ConvNeXtBlock, MBConvBlock, FusedMBConv, DenseBlock

**Audio/Sequence:** Conformer, WaveNetBlock

**Routing:** MetaNeurons (16 routing/composition patterns), Expert

## Syntax Summary

| Construct | Syntax |
|-----------|--------|
| Inline comment | `# comment` |
| Doc comment | `/// comment` |
| Neuron definition | `neuron Name(params):` |
| Default parameter | `param=value` |
| Neuron type param | `param: Neuron` |
| Input port | `in: [shape]` or `in name: [shape]` |
| Output port | `out: [shape]` or `out name: [shape]` |
| Variadic input | `in *name: [shape]` |
| Primitive body | `impl: provider,library/Class` |
| Composite body | `graph:` |
| Connection | `source -> dest` |
| Implicit fork | `source -> (a, b, c)` |
| Tuple join | `(a, b) -> Neuron()` |
| Context binding | `context:` block with `name = Neuron(args)` |
| Lazy binding | `@lazy name = Neuron(args)` |
| Static binding | `@static name = Neuron(args)` |
| Global binding | `@global name = Neuron(args)` |
| Unroll | `name = unroll(count): binding = Neuron(args)` |
| Match (runtime) | `match: -> [pattern] where guard: pipeline` |
| Match (contract) | `match(param): in [...] -> out [...]: pipeline` |
| Conditional | `if cond: pipeline elif cond: pipeline else: pipeline` |
| Fat arrow reshape | `=> [new_shape]` |
| Import | `use provider,library/Neuron` or `use provider,library/*` |
