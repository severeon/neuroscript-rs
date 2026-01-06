---
sidebar_label: Pad
---

# Pad

Pad

Pads tensor with specified value along boundaries.

Parameters:
- padding: Padding sizes (left, right) or (left, right, top, bottom) etc.
- value: Padding value (default: 0)
- mode: Padding mode - constant, reflect, replicate, circular (default: constant)

Shape Contract:
- Input: [*shape_in] tensor to pad
- Output: [*shape_out] padded tensor (larger along padded dimensions)

Notes:
- No learnable parameters (pure structural operation)
- padding specified from last dim backward: (left, right, top, bottom, ...)
- Modes:
- constant: fill with value
- reflect: reflect values at boundary
- replicate: repeat edge values
- circular: wrap around
- Essential for maintaining spatial size in convolutions

## Signature

```neuroscript
neuron Pad(padding, value=Int(0), mode=Name("constant"))
```

## Ports

**Inputs:**
- `default`: `[*shape_in]`

**Outputs:**
- `default`: `[*shape_out]`

## Implementation

```
Source { source: "core", path: "structural/Pad" }
```

