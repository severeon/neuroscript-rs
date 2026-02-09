---
sidebar_position: 2
title: Creating Packages
description: Initialize and structure NeuroScript neuron packages
---

# Creating Packages

A NeuroScript package is a collection of neuron definitions bundled with an `Axon.toml` manifest. Packages can be shared as git repositories and consumed as dependencies by other projects.

## Initializing a Package

Use `neuroscript init` to scaffold a new package:

```bash
neuroscript init attention-mechanisms \
  --author "Your Name <you@example.com>" \
  --license MIT
```

This creates:

```
attention-mechanisms/
├── Axon.toml          # Package manifest
├── README.md          # Basic documentation
├── .gitignore         # Common ignore patterns
└── src/
    └── attention_mechanisms.ns  # Starter neuron definition
```

Use `--bin` to include an `examples/` directory:

```bash
neuroscript init attention-mechanisms --bin
```

Use `--path` to create the package in a specific location:

```bash
neuroscript init my-neurons --path ./packages/my-neurons
```

## The Axon.toml Manifest

The manifest is the central configuration file for your package.

### Package Metadata

```toml
[package]
name = "attention-mechanisms"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
license = "MIT"
description = "Self-attention neurons for transformer architectures"
repository = "https://github.com/user/attention-mechanisms"
```

**Name rules**: lowercase, alphanumeric + hyphens only. No leading, trailing, or consecutive hyphens.

```toml
# Valid names
name = "core-primitives"
name = "attention123"
name = "my-cool-neurons"

# Invalid names
name = "Invalid_Name"        # no underscores
name = "-leading-hyphen"     # no leading hyphen
name = "double--hyphen"      # no consecutive hyphens
```

**Version**: must be valid [semver](https://semver.org/) (e.g., `"0.1.0"`, `"1.2.3-alpha"`).

### Exported Neurons

The `neurons` list declares which neurons your package provides to consumers:

```toml
neurons = [
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    "CrossAttention"
]
```

If `neurons` is empty or omitted, **all** neuron definitions in `src/*.ns` are exported.

### Dependencies

```toml
[dependencies]
core-blocks = { git = "https://github.com/org/core-blocks.git", branch = "main" }
local-utils = { path = "../local-utils" }
```

See [Working with Dependencies](dependencies) for details.

### Python Runtime

Declare Python dependencies that your compiled neurons need at runtime:

```toml
[python-runtime]
requires = ["torch>=2.0", "einops>=0.6"]
```

## Writing Neuron Definitions

Place your `.ns` files in the `src/` directory. Each file can contain one or more neuron definitions.

### Example: SwiGLU Feed-Forward Network

**`src/swiglu_ffn.ns`**:

```neuroscript
neuron SwiGLU_FFN(dim, expansion):
    in: [*batch, seq, dim]
    out: [*batch, seq, dim]
    graph:
        in -> Fork() -> (gate_path, value_path)
        gate_path ->
            Linear(dim, dim * expansion)
            SiLU()
            gate
        value_path ->
            Linear(dim, dim * expansion)
            value
        (gate, value) -> Multiply() ->
            Linear(dim * expansion, dim) ->
            out
```

### Example: Mamba-Style Block

**`src/mamba_block.ns`**:

```neuroscript
neuron MambaBlock(d_model, d_inner):
    in: [*batch, seq, d_model]
    out: [*batch, seq, d_model]
    graph:
        in -> Fork() -> (main, skip)
        main ->
            Linear(d_model, d_inner)
            SiLU()
            Linear(d_inner, d_inner)
            SiLU()
            Linear(d_inner, d_model)
            processed
        (processed, skip) -> Add() -> out
```

### Validating Your Package

Always validate your neurons before publishing:

```bash
# Validate a single file
neuroscript validate src/swiglu_ffn.ns

# Compile to PyTorch to verify codegen
neuroscript compile src/swiglu_ffn.ns

# List all neurons in a file
neuroscript list src/swiglu_ffn.ns
```

## Complete Example: Building a Package

Here is a full walkthrough of creating a package with two neurons:

```bash
# 1. Initialize the package
neuroscript init transformer-blocks \
  --author "Your Name <you@example.com>" \
  --license MIT

cd transformer-blocks
```

**2. Edit `Axon.toml`** to declare your exports:

```toml
[package]
name = "transformer-blocks"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
license = "MIT"
description = "Reusable transformer building blocks"

neurons = ["PreNormBlock", "SwiGLU_FFN"]

[dependencies]

[python-runtime]
requires = ["torch>=2.0"]
```

**3. Create `src/blocks.ns`** with your neuron definitions:

```neuroscript
neuron SwiGLU_FFN(dim, expansion):
    in: [*batch, seq, dim]
    out: [*batch, seq, dim]
    graph:
        in -> Fork() -> (gate_path, value_path)
        gate_path ->
            Linear(dim, dim * expansion)
            SiLU()
            gate
        value_path ->
            Linear(dim, dim * expansion)
            value
        (gate, value) -> Multiply() ->
            Linear(dim * expansion, dim) ->
            out

neuron PreNormBlock(dim, heads, expansion):
    in: [*batch, seq, dim]
    out: [*batch, seq, dim]
    graph:
        in -> Fork() -> (main, skip)
        main ->
            LayerNorm(dim)
            MultiHeadSelfAttention(dim, heads)
            processed
        (processed, skip) -> Add() -> out
```

**4. Validate and test:**

```bash
neuroscript validate src/blocks.ns
neuroscript compile src/blocks.ns --neuron SwiGLU_FFN
neuroscript compile src/blocks.ns --neuron PreNormBlock
```

**5. Push to git** to make it available as a dependency:

```bash
git init && git add -A && git commit -m "Initial package"
git remote add origin https://github.com/you/transformer-blocks.git
git push -u origin main
```

Other projects can now depend on this package using the git URL. See [Working with Dependencies](dependencies) for how consumers import your neurons.
