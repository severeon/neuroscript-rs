---
sidebar_label: Identity
---

# Identity

Identity Operation

Returns input unchanged. Useful as a placeholder, in conditional branches,
or when an operation slot requires a neuron but no transform is needed.

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input (unchanged)

Notes:
- No learnable parameters
- Zero computational cost (just returns input reference)
- Useful in pattern matching for do-nothing branches
- Can serve as placeholder during architecture development
- Used in residual connections when skip path needs no transform

## Signature

```neuroscript
neuron Identity()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
Source { source: "core", path: "structural/Identity" }
```

