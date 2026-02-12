# Common Pitfalls

## 1. Concat Arity

**Wrong**: Passing 3+ inputs to Concat
```neuroscript
(a, b, c) -> Concat() -> out  # ERROR: Concat takes exactly 2 inputs
```

**Right**: Chain Concat calls
```neuroscript
(a, b) -> Concat() -> ab
(ab, c) -> Concat() -> out
```

## 2. Splitting Tensors

**Preferred**: Implicit fork (v0.3.0+)
```neuroscript
in -> (a, b)                    # 2-way implicit fork
in -> (a, b, c, d)             # N-way implicit fork (no limit)
```

**Also valid**: Explicit Fork/Fork3 (only needed for named port access)
```neuroscript
in -> Fork() -> f
f.left -> ...                   # named port access
f.right -> ...
```

**No longer needed**: Chaining Fork calls for 4+ splits.

## 3. Impl Format

**Wrong**: Dot notation
```neuroscript
impl: core.nn.Linear           # ERROR: wrong format
impl: core,nn.Linear           # ERROR: wrong format
```

**Right**: Comma + slash
```neuroscript
impl: core,nn/Linear           # Correct
impl: core,attention/ScaledDotProductAttention
```

## 4. Multiple Variadics

**Wrong**: Two variadic patterns in one shape
```neuroscript
in: [*batch, *features]        # ERROR: only one variadic allowed
```

**Right**: Single variadic
```neuroscript
in: [*batch, dim]              # variadic prefix + fixed trailing
in: [batch, seq, *rest]        # fixed prefix + variadic trailing
```

## 5. Match Exhaustiveness

**Wrong**: No catch-all pattern
```neuroscript
in -> match:
  [*, 512]: Identity() -> out
  [*, 256]: Linear(256, 512) -> out
  # ERROR: non-exhaustive — what about other dims?
```

**Right**: End with catch-all
```neuroscript
in -> match:
  [*, 512]: Identity() -> out
  [*, 256]: Linear(256, 512) -> out
  [*, d]: Linear(d, 512) -> out    # catch-all
```

## 6. Match Arm Ordering

**Wrong**: Catch-all before specific arms
```neuroscript
in -> match:
  [*, d]: Linear(d, 512) -> out      # catches everything
  [*, 512]: Identity() -> out        # UNREACHABLE
```

**Right**: Most specific first
```neuroscript
in -> match:
  [*, 512]: Identity() -> out        # specific
  [*, d]: Linear(d, 512) -> out      # catch-all last
```

## 7. Recursive Binding Scope

**Wrong**: `@static` for recursion
```neuroscript
context:
  @static recurse = MyNeuron(depth - 1)  # ERROR: infinite init
```

**Right**: `@lazy` for recursion
```neuroscript
context:
  @lazy recurse = MyNeuron(depth - 1)    # created on first use
```

## 8. Missing Base Case

**Wrong**: No termination condition
```neuroscript
context:
  @lazy recurse = Stack(depth - 1)
graph:
  in -> recurse -> out  # infinite recursion
```

**Right**: Guard with base case
```neuroscript
graph:
  in -> match:
    [*] where depth > 0: ... -> recurse -> out
    [*]: Identity() -> out              # base case
```

## 9. Shape Solver Limitations

The shape solver handles single-unknown linear equations only.

**Solvable**:
- `dim * 4 = 2048` → `dim = 512`
- `x + 128 = 640` → `x = 512`

**Not solvable**:
- `a * b = 512` (two unknowns)
- `dim^2 = 256` (non-linear)

If the solver can't resolve a dimension, provide it explicitly as a parameter.

## 10. Port Name Mismatch

Add inputs are named `main` and `skip`.

When using tuple unpacking (implicit or explicit), the names you assign override the port names:
```neuroscript
in -> (processed, residual)            # implicit fork — your names
(processed, residual) -> Add() -> out  # matched by position, not name
```

## 11. Inlining Instead of Reusing Stdlib

**Wrong**: Duplicating a pattern that already exists in stdlib
```neuroscript
# Inlining the SE pattern manually
se_input -> (features, attn_path)
attn_path ->
    GlobalAvgPool()
    Flatten()
    Linear(channels, channels / reduction)
    ReLU()
    Linear(channels / reduction, channels)
    Sigmoid()
    attention
(features, attention) -> Multiply() -> se_out
```

**Right**: Reference the stdlib neuron by name
```neuroscript
se_input -> SEBlock(channels, reduction) -> se_out
```

All stdlib neurons (primitives and composites) are loaded automatically. Check `stdlib/*.ns` before inlining a sub-pattern.
