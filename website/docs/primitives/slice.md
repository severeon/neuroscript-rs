---
sidebar_label: Slice
---

# Slice

Slice

Extracts a contiguous slice from a tensor along specified dimensions.

Parameters:
- dim: Dimension to slice along
- start: Starting index (inclusive)
- end: Ending index (exclusive, use -1 for end)

Shape Contract:
- Input: [*shape_in] tensor to slice
- Output: [*shape_out] sliced tensor (smaller along dim)

Notes:
- No learnable parameters (pure structural operation)
- Supports negative indexing (Python-style)
- end=-1 means slice to the end
- Commonly used to extract specific positions or ranges
- Memory efficient (view operation when possible)

## Signature

```neuroscript
neuron Slice(dim, start, end)
```

## Ports

**Inputs:**
- `default`: `[*shape_in]`

**Outputs:**
- `default`: `[*shape_out]`

## Implementation

```
"from core import structural/Slice"
```

```
Source { source: "core", path: "structural/Slice" }
```

