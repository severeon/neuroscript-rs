---
sidebar_label: GroupNorm
---

# GroupNorm

Group Normalization

Divides channels into groups and normalizes within each group. Works well
with small batch sizes where BatchNorm becomes unstable.

Parameters:
- num_groups: Number of groups to divide channels into
- num_channels: Total number of channels (must be divisible by num_groups)

Shape Contract:
- Input: [*, num_channels, *, *] feature maps with spatial dimensions
- Output: [*, num_channels, *, *] same shape as input

Notes:
- num_channels must be divisible by num_groups
- Common choices: 32 groups (GPT-style) or num_groups = num_channels (LayerNorm equivalent)
- Independent of batch size (works with batch_size=1)
- Includes learnable scale (gamma) and shift (beta) per channel
- Used in ResNeXt, BigGAN, and many generative models
- Interpolates between LayerNorm (1 group) and InstanceNorm (channels groups)

## Signature

```neuroscript
neuron GroupNorm(num_groups, num_channels)
```

## Ports

**Inputs:**
- `default`: `[*, num_channels, *, *]`

**Outputs:**
- `default`: `[*, num_channels, *, *]`

## Implementation

```
Source { source: "core", path: "normalization/GroupNorm" }
```

