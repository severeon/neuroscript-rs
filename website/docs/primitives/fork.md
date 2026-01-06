---
sidebar_label: Fork
---

# Fork

Fork (Duplicate)

Splits a single input into two identical outputs. Essential building block
for residual connections, parallel processing paths, and skip connections.

Shape Contract:
- Input: [*shape] arbitrary shape
- Output main: [*shape] first copy of input
- Output skip: [*shape] second copy of input

Notes:
- No learnable parameters (pure structural operation)
- Zero-copy in most implementations (shares memory)
- Named outputs (main, skip) for clarity in residual patterns
- Both outputs are identical references to the input
- Fundamental for any architecture with parallel paths

## Signature

```neuroscript
neuron Fork()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `main`: `[*shape]`
- `skip`: `[*shape]`

## Implementation

```
Source { source: "core", path: "structural/Fork" }
```

