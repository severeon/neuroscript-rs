# NeuroScript Code Generation

This document describes the PyTorch code generation feature added to NeuroScript.

## Overview

The codegen feature translates NeuroScript IR into executable PyTorch Python code. This implements **Phase 0** from the `codegen_guide.md` - direct lowering without shape inference or optimizations.

## Usage

### Basic Usage

Generate PyTorch code to stdout:

```bash
neuroscript --codegen <NeuronName> <file.ns>
```

Example:

```bash
./target/release/neuroscript --codegen ResidualFFN examples/codegen_demo.ns
```

### Save to File

Generate PyTorch code and save to a file:

```bash
neuroscript --codegen <NeuronName> --output <output.py> <file.ns>
```

Example:

```bash
./target/release/neuroscript --codegen ResidualFFN --output residual.py examples/codegen_demo.ns
```

### Combine with Validation

You can validate the program before generating code:

```bash
neuroscript --validate --codegen <NeuronName> <file.ns>
```

## Supported Features

### ✅ Phase 0 Features

1. **Sequential Pipelines**
   - Simple chains of operations
   - Example: `in -> LayerNorm(dim) -> GELU() -> out`

2. **Composite Neurons**
   - Graph-based neuron definitions
   - Nested module instantiation

3. **Parameters**
   - Required parameters: `neuron FFN(dim):`
   - Default values: `neuron FFN(dim, expansion=4):`

4. **Arithmetic Expressions**
   - Binary operations in parameters
   - Example: `Linear(dim, dim * 4)`
   - Supports: `+`, `-`, `*`, `/`

5. **Tuple Unpacking**
   - Multi-output operations
   - Example: `Fork() -> (path_a, path_b)`

6. **Multi-Input Operations**
   - Operations with multiple inputs
   - Example: `(result1, result2) -> Add() -> out`

7. **Intermediate Nodes**
   - Named references for complex graphs
   - Example: `in -> LayerNorm(dim) -> normalized`

8. **Import Generation**
   - Automatic imports for primitives
   - From `neuroscript_runtime.primitives.*`

## Generated Code Structure

Every generated neuron becomes a PyTorch `nn.Module`:

```python
import torch
import torch.nn as nn
from neuroscript_runtime.primitives.* import *

class NeuronName(nn.Module):
    def __init__(self, param1, param2=default):
        super().__init__()
        # Module instantiations
        self.module_0 = Module(...)
        self.module_1 = Module(...)
    
    def forward(self, x):
        # Sequential operations
        x0 = self.module_0(x)
        x1 = self.module_1(x0)
        return x1
```

## Examples

See `examples/codegen_demo.ns` for a comprehensive demonstration of all supported features.

### Simple Pipeline

**Input:**

```neuroscript
neuron SimplePipeline(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> LayerNorm(dim) -> GELU() -> out
```

**Output:**

```python
class SimplePipeline(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm_0 = LayerNorm(dim)
        self.g_e_l_u_1 = GELU()
    
    def forward(self, x):
        x0 = self.layer_norm_0(x)
        x1 = self.g_e_l_u_1(x0)
        return x1
```

### Residual Connection

**Input:**

```neuroscript
neuron ResidualFFN(dim, expansion=4, dropout_rate=0.1):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Fork() -> (main_path, skip_path)
    main_path ->
      LayerNorm(dim)
      Linear(dim, dim * expansion)
      GELU()
      Dropout(p=dropout_rate)
      Linear(dim * expansion, dim)
      transformed
    (transformed, skip_path) -> Add() -> out
```

**Output:**

```python
class ResidualFFN(nn.Module):
    def __init__(self, dim, expansion=4, dropout_rate=0.1):
        super().__init__()
        self.fork_0 = Fork()
        self.layer_norm_1 = LayerNorm(dim)
        self.linear_2 = Linear(dim, dim * expansion)
        self.g_e_l_u_3 = GELU()
        self.dropout_4 = Dropout(p=dropout_rate)
        self.linear_5 = Linear(dim * expansion, dim)
        self.add_6 = Add()
    
    def forward(self, x):
        x0 = self.fork_0(x)
        x1, x2 = x0                              # Tuple unpacking
        x3 = self.layer_norm_1(x1)
        x4 = self.linear_2(x3)
        x5 = self.g_e_l_u_3(x4)
        x6 = self.dropout_4(x5)
        x7 = self.linear_5(x6)
        x8 = self.add_6((x7, x2))                # Multi-input
        return x8
```

## Known Limitations

These features are **not yet supported** (planned for future phases):

- ❌ Match expressions (pattern matching on shapes)
- ❌ Automatic shape inference
- ❌ Graph optimizations
- ❌ Nested composite generation (generating all dependencies)
- ❌ Multi-backend support (JAX, ONNX, etc.)
- ❌ Docstring generation
- ❌ Shape comments in generated code

## Runtime Requirements

The generated code imports from `neuroscript_runtime.primitives.*`. You need:

1. A Python environment with PyTorch installed
2. The `neuroscript_runtime` package (to be implemented)

The runtime package should provide implementations for:

- `neuroscript_runtime.primitives.linear.Linear`
- `neuroscript_runtime.primitives.activations.GELU`
- `neuroscript_runtime.primitives.normalization.LayerNorm`
- `neuroscript_runtime.primitives.regularization.Dropout`
- And other primitives...

## Architecture

The codegen is implemented in `src/codegen.rs` and follows this pipeline:

1. **Parse** NeuroScript source → IR
2. **Collect** all Call endpoints from connections
3. **Assign** unique IDs and module names
4. **Generate** `__init__` with module instantiations
5. **Generate** `forward` with sequential operations
6. **Generate** imports based on used primitives
7. **Combine** everything into final Python code

## Future Phases

See `src/codegen_guide.md` for the roadmap:

- **Phase 1**: Polish (imports, docstrings, shape comments)
- **Phase 2**: Fancy (shape inference, caching, multi-backend)
- **Phase 3**: Wild (backward graphs, staged execution, apoptosis integration!)
