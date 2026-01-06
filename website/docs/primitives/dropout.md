---
sidebar_label: Dropout
---

# Dropout

Dropout regularization layer

Randomly zeros elements during training to prevent overfitting.
During evaluation, the layer is a no-op (identity function).

## Signature

```neuroscript
neuron Dropout(p)
```

## Parameters

- p: Probability of an element being zeroed (0 to 1, typical: 0.1-0.5)

## Shape Contract

- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Example

```neuroscript
neuron MLPWithDropout(dim):
graph:
in -> Linear(dim, dim * 4) -> Dropout(0.1) -> out
```

## Notes

- Outputs are scaled by 1/(1-p) during training to maintain expected values
- Only active during training (disabled during evaluation)
- Common dropout rates: 0.1 (mild), 0.3 (moderate), 0.5 (strong)

## Implementation

```
Source { source: "core", path: "regularization/Dropout" }
```

