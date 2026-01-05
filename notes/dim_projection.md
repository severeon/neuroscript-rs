# Dimensional Projection & The 0D Reduction

Your brain tickle is onto something profound - you're intuiting a **dimensional collapse** where everything reduces to 0D objects (scalars/modules) with internal tensor state, and brackets describe **replication topology** rather than geometric arrangement.

## The Core Insight

```text
// These are all 0D objects with different internal structure
node              // scalar: single value
node[512]         // vector: 512 values internally  
node[512][8]      // matrix: 512×8 structure internally
node[512][8][4]   // 3-tensor: 512×8×4 internally

// But from the OUTSIDE, they're all just "a thing that processes input"
```

This is actually **elegant** because:

- Every module has the same interface: `forward(input) -> output`
- Brackets don't mean spatial topology, they mean **internal parallelism**
- The compiler can decide whether to materialize as separate objects or fused tensors

## Dimensional Projection Semantics

```text
// Single neuron (0D)
node             // forward: scalar -> scalar

// Project to 1D
node[512]        // forward: vector[512] -> vector[512]
                 // OR: scalar -> vector[512] (broadcast input)
                 // Internally: 512 parallel copies

// Project to 2D  
node[512][8]     // forward: vector[512] -> matrix[512,8]?
                 // OR: matrix[512,8] -> matrix[512,8]?
                 // Internally: 512×8 parallel copies

// The key question: what does indexing MEAN?
```

## Two Interpretation Models

### Model A: Nested Containers (Current Intuition)

```text
node[512][8]
  = 8 clusters, each cluster has 512 nodes
  = outer dimension is "batch" or "ensemble"
  = inner dimension is "feature"

Linear(512)[8]
  = 8 separate Linear layers, each with 512 outputs
  = input broadcasts to all 8
  = outputs stack to [8, 512]
```

### Model B: Dimensional Collapse (Your Brain Tickle)

```text
node[512][8]
  = single 0D object
  = internally holds tensor[512, 8]
  = forward just does tensor ops

Linear(512)[8]
  = single 0D module  
  = weight: tensor[8, 512, in_features]
  = forward does batched matmul
```

## The Elegance of Model B

If everything is 0D with internal tensors, then:

```text
// Base case
mod Neuron(in):
  weight: Tensor[in]
  bias: Scalar

// 1D projection
mod Neuron(in)[out]
  weight: Tensor[out, in]  // automatically broadcast
  bias: Tensor[out]

// 2D projection
mod Neuron(in)[out_x][out_y] {
  weight: Tensor[out_x, out_y, in]  // shape just extends
  bias: Tensor[out_x, out_y]
}

// Nested modules
mod MLP(in, hidden)[out] {
  // Each of 'out' copies has same internal structure
  layer1: Neuron(in)[hidden]   // shape: [out, hidden, in]
  layer2: Neuron(hidden)[in]   // shape: [out, in, hidden]
}
```

The brackets **always** prepend dimensions to internal tensors. Everything compiles to efficient batched operations.

## Your Examples Decoded

```text
// "8 layers of linears would be Linear[1][8]"
Linear[1][8]
  = 8 copies (outer bracket)
  = each copy has 1 output (inner bracket)
  = essentially 8 scalar outputs running in parallel
  = weight shape: [8, 1, in_features]

// "a node[512][8] is essentially a loose Linear(512)[8]"
node[512][8]
  = 512 features × 8 parallel instances
  = weight shape: [8, 512, in_features]
  
Linear(in, 512)[8]
  = 8 parallel linears, each outputting 512 features
  = weight shape: [8, 512, in_features]
  = SAME THING!
```

## Implementation Reality Check

This is actually **easier** to implement than the compositional version:

```rust
// In your IR
struct ModInstance {
    base_type: ModuleDef,
    dimensions: Vec<usize>,  // [8, 512] for node[512][8]
}

// Codegen
fn generate_parameters(&self, mod: &ModInstance) -> Vec<TensorShape> {
    mod.base_type.parameters.iter().map(|param| {
        let mut shape = mod.dimensions.clone();
        shape.extend(param.shape.iter());
        TensorShape(shape)
    }).collect()
}

// Example
Neuron(128)[512][8]
  -> base param: weight[128], bias[]
  -> generated params: weight[8, 512, 128], bias[8, 512]
```

## The PITA Factor

**Not a PITA because:**

- Brackets just prepend dimensions (simple rule)
- Everything becomes batched tensor ops (fast)
- No special cases for 1D vs 2D vs 3D
- Pattern matching unnecessary - it's all the same codegen

**Potential issues:**

- Semantic clarity: does `node[512][8]` mean 512-then-8 or 8-then-512?
- Broadcasting rules: when input dimension doesn't match
- Indexing semantics: what does `layer[3]` mean?

## Bracket Ordering Convention

```text
// Right-to-left interpretation (like function application)
node[8][512]
  = start with node
  = replicate 512 times (innermost/fastest varying)
  = then replicate those 8 times (outermost/slowest varying)
  = tensor shape: [8, 512, ...]

// This matches PyTorch convention
x = torch.randn(8, 512, 128)
  # 8 batches of 512 sequences of 128-dim vectors
```

## Concrete Proposal

```text
// Bracket rule: prepend dimensions left-to-right
node              -> 0D module
node[512]         -> params get [512, ...] prepended
node[512][8]      -> params get [512, 8, ...] prepended  
node[512][8][4]   -> params get [512, 8, 4, ...] prepended

// Example
mod Neuron(in_features) {
  weight: Tensor[in_features]
  bias: Scalar
}

Neuron(128)           -> weight[128], bias[]
Neuron(128)[512]      -> weight[512, 128], bias[512]
Neuron(128)[512][8]   -> weight[512, 8, 128], bias[512, 8]

// Forward pass just does batched ops
forward(x: Tensor[...]) {
  // x gets broadcast to match param dimensions
  return batched_dot(x, weight) + bias
}
```

## Should You Build This Now?

**Implement it** - it's actually simpler than the alternatives:

1. Single codegen path for all dimensionalities
2. No pattern matching needed
3. Directly maps to efficient tensor ops
4. Naturally handles your CTM use case

```text
// CTM as dimensional projection
mod DeepNeuron(in, depth) {
  layers: Neuron(in)[depth]
}

DeepNeuron(128, 4)[512]
  -> layers.weight: [512, 4, 128]
  -> layers.bias: [512, 4]
  -> forward does batched sequential MLP
```

The only question: do you need custom traversal/aggregation, or is batched tensor math enough for now? If the latter, this is straightforward to implement.

# Dimensional Projection Design Specification

## Core Principle

Every module is a 0D object with internal tensor state. Brackets specify replication topology by prepending dimensions to all internal parameters.

## Bracket Semantics

```text
mod[d1][d2]...[dn]
```

Prepends dimensions `[d1, d2, ..., dn]` to all parameter tensors, left-to-right. Forward pass performs batched operations across these dimensions.

## Rules

1. **Base case**: `mod` has parameters with shapes `[s1, s2, ...]`
2. **Projection**: `mod[d1][d2]` transforms parameters to `[d1, d2, s1, s2, ...]`
3. **Forward**: Input broadcasts across prepended dimensions, operations batch naturally
4. **No special cases**: Same codegen for 1D, 2D, nD projections

## Examples

```text
Neuron(128)           -> weight[128], bias[]
Neuron(128)[512]      -> weight[512, 128], bias[512]  
Neuron(128)[512][8]   -> weight[512, 8, 128], bias[512, 8]
```

## Implementation Tasks

1. Parse bracket notation into `Vec<usize>` dimension list
2. During parameter generation, prepend dimensions to base shapes
3. Codegen batched tensor operations (einsum/batched_matmul)
4. No traversal/aggregation customization (defer to later)
