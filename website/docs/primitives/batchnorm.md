---
sidebar_label: BatchNorm
---

# BatchNorm

Batch normalization layer

Normalizes inputs across the batch dimension, computing mean and variance
over the batch. Helps stabilize training and enables higher learning rates.

## Signature

```neuroscript
neuron BatchNorm(num_features)
```

## Parameters

- num_features: Number of features (channels for CNNs, dimensions for MLPs)

## Shape Contract

- Input: [*shape, num_features] where num_features is the normalized dimension
- Output: [*shape, num_features] same shape as input

## Ports

**Inputs:**
- `default`: `[*shape, num_features]`

**Outputs:**
- `default`: `[*shape, num_features]`

## Example

```neuroscript
neuron ConvBlock(in_channels, out_channels):
graph:
in -> Conv2d(in_channels, out_channels, 3) ->
BatchNorm(out_channels) -> ReLU() -> out
```

## Notes

- Normalizes over batch and spatial dimensions (for CNNs)
- Includes learnable scale (γ) and shift (β) parameters
- Maintains running statistics (mean, variance) for inference
- Behavior differs between training and evaluation modes
- Less stable than LayerNorm for variable batch sizes or sequential data

## Implementation

```
Source { source: "core", path: "normalization/BatchNorm" }
```

