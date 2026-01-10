---
sidebar_label: InstanceNorm
---

# InstanceNorm

Instance Normalization

Normalizes each sample independently across spatial dimensions.
Commonly used in style transfer and image generation.

Parameters:
- num_features: Number of channels (C dimension)
- eps: Small constant for numerical stability (default: 1e-5)
- affine: If true, learnable scale and shift (default: true)

Shape Contract:
- Input: [batch, num_features, *spatial] feature maps
- Output: [batch, num_features, *spatial] same shape as input

Notes:
- Normalizes across spatial dimensions only (not batch or channel)
- Each instance (sample) normalized independently
- Equivalent to GroupNorm with num_groups=num_features
- Removes style information from features
- Used in neural style transfer, GANs, image-to-image translation

## Signature

```neuroscript
neuron InstanceNorm(num_features, eps=Float(1e-5), affine=Bool(true))
```

## Ports

**Inputs:**
- `default`: `[batch, num_features, *spatial]`

**Outputs:**
- `default`: `[batch, num_features, *spatial]`

## Implementation

```
Source { source: "core", path: "normalization/InstanceNorm" }
```

