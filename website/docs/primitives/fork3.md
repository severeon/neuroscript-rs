---
sidebar_label: Fork3
---

# Fork3

Fork3 (Triple Duplicate)

Splits a single input into three identical outputs. Used for attention
mechanisms where Q, K, V projections start from the same input.

Shape Contract:
- Input: [*shape] arbitrary shape
- Output a: [*shape] first copy
- Output b: [*shape] second copy
- Output c: [*shape] third copy

Notes:
- No learnable parameters (pure structural operation)
- Zero-copy in most implementations (shares memory)
- Primary use: self-attention Q/K/V generation
- All three outputs are identical references to input
- Can be composed with Fork for more branches

## Signature

```neuroscript
neuron Fork3()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `a`: `[*shape]`
- `b`: `[*shape]`
- `c`: `[*shape]`

## Implementation

```
"from core import structural/Fork3"
```

```
Source { source: "core", path: "structural/Fork3" }
```

