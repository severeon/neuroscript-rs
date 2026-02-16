---
sidebar_position: 2
title: Compiler Reference
description: Complete reference for parse, validate, compile, and list commands
---

# Compiler Reference

The NeuroScript CLI provides four commands for working with `.ns` source files: **parse**, **validate**, **compile**, and **list**.

## `neuroscript parse`

Parse a NeuroScript file and display its internal IR structure. Useful for understanding how the compiler sees your code.

```bash
neuroscript parse <FILE> [OPTIONS]
```

### Arguments [parse]

| Argument | Description |
| -------- | ----------- |
| `<FILE>` | Path to a `.ns` source file |

### Options [parse]

| Option | Description |
| ------ | ----------- |
| `-v, --verbose` | Show detailed IR structure output |

### Example [parse]

```bash
# Quick check that a file parses
neuroscript parse my_model.ns

# See the full IR
neuroscript parse my_model.ns --verbose
```

---

## `neuroscript validate`

Parse a file and run all validation checks: neuron existence, connection arity, cycle detection, and shape compatibility.

```bash
neuroscript validate <FILE> [OPTIONS]
```

### Arguments [validate]

| Argument | Description |
| -------- | ----------- |
| `<FILE>` | Path to a `.ns` source file |

### Options [validate]

| Option | Description |
| ------ | ----------- |
| `-v, --verbose` | Show detailed validation output |
| `--no-stdlib` | Skip loading the standard library |
| `--no-deps` | Skip loading fetched dependencies |

### Example [validate]

```bash
neuroscript validate my_model.ns --verbose
```

---

## `neuroscript compile`

Full compilation pipeline: parse, validate, optimize, and generate a PyTorch `nn.Module`.

```bash
neuroscript compile <FILE> [OPTIONS]
```

### Arguments [compile]

| Argument | Description |
| -------- | ----------- |
| `<FILE>` | Path to a `.ns` source file |

### Options [compile]

| Option | Description |
| ------ | ----------- |
| `-n, --neuron <NAME>` | Neuron to compile (defaults to PascalCase of filename) |
| `-o, --output <FILE>` | Write output to a file instead of stdout |
| `-v, --verbose` | Show optimization stats and detailed output |
| `--bundle` | Inline primitive class definitions for a self-contained output |
| `--no-optimize` | Disable all optimizations |
| `--no-dead-elim` | Disable dead branch elimination only |
| `--no-stdlib` | Skip loading the standard library |
| `--no-deps` | Skip loading fetched dependencies |

### Examples [compile]

```bash
# Compile to stdout (auto-detects neuron from filename)
neuroscript compile my_model.ns

# Compile a specific neuron to a file
neuroscript compile my_model.ns --neuron Encoder -o encoder.py

# See optimization details
neuroscript compile my_model.ns --verbose
```

---

## `neuroscript list`

List all neurons defined in a file, the standard library, or fetched packages.

```bash
neuroscript list [FILE] [OPTIONS]
```

### Arguments [list]

| Argument | Description |
| -------- | ----------- |
| `[FILE]` | NeuroScript file (optional when using `--stdlib` or `--available`) |

### Options [list]

| Option | Description |
| ------ | ----------- |
| `-v, --verbose` | Show connection details |
| `--stdlib` | List all primitives and stdlib composites |
| `--package <NAME>` | List neurons from a specific fetched dependency |
| `--available` | List everything: stdlib + all dependencies |

### Examples [list]

```bash
neuroscript list my_model.ns
neuroscript list --stdlib --verbose
neuroscript list --available
```

---

## Bundle Mode

By default, compiled output imports primitives from the `neuroscript_runtime` Python package:

```python
from neuroscript_runtime.primitives.linear import Linear
from neuroscript_runtime.primitives.activations import GELU
```

This requires `pip install neuroscript-runtime` in the target environment.

The **`--bundle`** flag eliminates that dependency by inlining the required primitive class definitions directly into the generated file. The output is a single, self-contained Python module that only needs `torch`:

```bash
neuroscript compile my_model.ns --bundle -o model.py
```

The bundled file includes:

- A unified import block (`torch`, `torch.nn`, `torch.nn.functional`, `math`, `typing`)
- Only the primitive classes actually used by your neuron (not the entire library)
- Your compiled neuron class(es) at the bottom

### When to use `--bundle`

| Scenario | Recommended |
| -------- | ----------- |
| Development with the NeuroScript toolchain installed | Default (no flag) |
| Sharing a generated file with collaborators | `--bundle` |
| Deploying to an environment without `neuroscript_runtime` | `--bundle` |
| Embedding generated code in another project | `--bundle` |
| Minimizing output size | Default (imports are smaller) |

### Example

```bash
# Generate a self-contained model file
neuroscript compile transformer.ns --bundle -o transformer.py

# Verify it works without neuroscript_runtime installed
python -c "
import importlib.util, torch
spec = importlib.util.spec_from_file_location('model', 'transformer.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('Classes:', [c for c in dir(mod) if not c.startswith('_')])
"
```
