# Shape System

## Wildcard Dimensions

NeuroScript supports wildcard dimensions in shape declarations to match
variable-rank tensors.

### Leading-only constraint

The wildcard `*` must appear as the **first** dimension in a shape. It absorbs
zero or more leading dimensions during shape matching, following PyTorch
broadcasting semantics.

```neuroscript
# Valid -- wildcard is the first dimension
in: [*, dim]         # matches [dim], [batch, dim], [batch, seq, dim], ...
in: [*shape, dim]    # named variadic -- same rule, first position only

# INVALID -- wildcard in non-leading position
in: [dim, *]         # NOT supported
in: [batch, *, dim]  # NOT supported
```

When two shapes of different rank are compared, the shorter shape must begin
with a wildcard for the match to succeed. For example, `[*, dim]` can match
against `[batch, seq, dim]` because the wildcard absorbs the extra leading
dimensions `batch` and `seq`.

If both shapes start with a wildcard, matching is rejected to avoid ambiguity
about which wildcard absorbs the extra dimensions.

This constraint is enforced during validation (`src/validator/shapes.rs`).
