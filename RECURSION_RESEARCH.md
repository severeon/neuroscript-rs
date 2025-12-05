# Recursion Strategies for Neural Architectures

## Overview

This document explores how to implement efficient recursion in NeuroScript v2, drawing from research in recursive neural networks, dynamic computation, and compiler techniques.

---

## 1. Types of Neural Recursion

### 1.1 Static Recursion (Compile-Time Unrolling)

**Pattern**: Fixed depth known at compile time

**Example**: Universal Transformer with N=6 layers
```neuroscript
neuron UniversalTransformer(depth: Nat = 6, dim: Nat = 512):
  in: [batch, seq, dim]
  out: [batch, seq, dim]

  let:
    layer = TransformerBlock(dim)

  graph:
    in -> match:
      [*] where depth > 0:
        -> layer
        -> UniversalTransformer(depth - 1, dim)
        -> out
      [*]:
        -> Identity() -> out
```

**Compilation Strategy**:
- **Unroll at compile time**: Generate N copies of the module
- **Weight sharing**: All copies reference same parameters
- **Optimization**: Fuse operations across unrolled layers

**PyTorch Lowering**:
```python
class UniversalTransformer(nn.Module):
    def __init__(self, depth=6, dim=512):
        super().__init__()
        self.depth = depth
        # Single module instance, reused
        self.layer = TransformerBlock(dim)

    def forward(self, x):
        # Unrolled loop
        for _ in range(self.depth):
            x = self.layer(x)
        return x
```

**Pros**:
- ✅ Simple to implement
- ✅ Efficient (no overhead)
- ✅ Easy to optimize (compiler knows loop bound)

**Cons**:
- ❌ Depth must be compile-time constant
- ❌ Can't adapt depth per input

---

### 1.2 Dynamic Recursion (Runtime Halting)

**Pattern**: Recurse until learned halting condition

**Example**: Adaptive Computation Time (ACT)
```neuroscript
neuron AdaptiveProcessor(dim: Nat = 512, threshold: Float = 0.95):
  in: [batch, seq, dim]
  out: [batch, seq, dim]

  let:
    process = TransformerBlock(dim)
    halt_net = Linear(dim, 1) -> Sigmoid()

  graph:
    in -> adaptive_loop:
      step: process
      halt: halt_net
      threshold: threshold
      max_steps: 32
    -> out
```

**Compilation Strategy**:
- **While loop with condition**: Generate dynamic control flow
- **Accumulation**: Weighted sum of outputs at each step
- **Ponder cost**: Regularization to encourage early stopping

**PyTorch Lowering**:
```python
class AdaptiveProcessor(nn.Module):
    def __init__(self, dim=512, threshold=0.95):
        super().__init__()
        self.process = TransformerBlock(dim)
        self.halt_net = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.threshold = threshold
        self.max_steps = 32

    def forward(self, x):
        batch, seq, dim = x.shape

        # Initialize accumulators
        output = torch.zeros_like(x)
        halting_prob = torch.zeros(batch, seq, 1, device=x.device)
        remainders = torch.ones(batch, seq, 1, device=x.device)
        n_updates = torch.zeros(batch, seq, 1, device=x.device)

        for step in range(self.max_steps):
            # Process
            x = self.process(x)

            # Compute halting probability
            p = self.halt_net(x)

            # Still running mask
            still_running = (halting_prob < 1.0).float()

            # Update accumulation weights
            new_halted = (halting_prob + p * still_running) > self.threshold
            p_masked = torch.where(
                new_halted,
                remainders,  # Use remainder if newly halted
                p * still_running  # Otherwise use p
            )

            # Accumulate output
            output += x * p_masked
            halting_prob += p_masked
            remainders -= p_masked
            n_updates += still_running

            # Early exit if all sequences halted
            if (halting_prob >= self.threshold).all():
                break

        # Ponder cost (regularization)
        self.ponder_cost = (n_updates + remainders).mean()

        return output
```

**Pros**:
- ✅ Adaptive depth per input
- ✅ Can learn to stop early (efficiency)
- ✅ More expressive than fixed depth

**Cons**:
- ❌ Complex implementation (accumulation logic)
- ❌ Harder to optimize (dynamic control flow)
- ❌ Training instability (halting probability)

---

### 1.3 Structural Recursion (Tree/Graph Traversal)

**Pattern**: Recurse over input data structure

**Example**: TreeLSTM
```neuroscript
neuron TreeLSTM(embed_dim: Nat = 256):
  in: Tree<Token>
  out: [embed_dim]

  let:
    embedder = Embedding(vocab_size, embed_dim)
    composer = ChildSumTreeLSTMCell(embed_dim)

  graph:
    in -> match:
      Leaf(token):
        -> embedder(token)
        -> out

      Node(children: List<Tree>):
        -> children.map(TreeLSTM(embed_dim))  # Recursive on children
        -> composer  # Compose child states
        -> out
```

**Compilation Strategy**:
- **Recursive function**: Generate actual recursive forward pass
- **Memoization**: Cache results for shared subtrees
- **Batch processing**: Process trees of same structure together

**PyTorch Lowering**:
```python
class TreeLSTM(nn.Module):
    def __init__(self, embed_dim=256, vocab_size=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.composer = ChildSumTreeLSTMCell(embed_dim)

    def forward(self, tree):
        if tree.is_leaf():
            # Base case: embed token
            return self.embedder(tree.token)
        else:
            # Recursive case: process children then compose
            child_states = [self.forward(child) for child in tree.children]
            return self.composer(child_states)

    def forward_batch(self, trees):
        """Batch processing for efficiency"""
        # Group trees by structure for vectorized ops
        # ... (complex batching logic)
        pass
```

**Pros**:
- ✅ Natural for hierarchical data
- ✅ Theoretically elegant
- ✅ Can handle arbitrary structures

**Cons**:
- ❌ Hard to batch (trees have different shapes)
- ❌ Slower than sequential models
- ❌ Requires structured input (not just tensors)

---

## 2. Weight Sharing Strategies

### 2.1 Full Weight Sharing (Same Module)

All recursive calls use **identical parameters**.

```python
class RecursiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = SomeLayer()  # Single instance

    def forward(self, x, depth):
        for _ in range(depth):
            x = self.layer(x)  # Reuse same layer
        return x
```

**Benefits**:
- Minimal parameter count
- Strong regularization (forced to be general)
- Easy to train

**Drawbacks**:
- Less expressive (can't specialize per depth)
- May need more steps to converge

---

### 2.2 Partial Weight Sharing (Parameter Tying)

Some parameters shared, others depth-specific.

```python
class PartiallySharedRecursive(nn.Module):
    def __init__(self, depth):
        super().__init__()
        # Shared parameters
        self.shared_layer = SomeLayer()

        # Depth-specific parameters
        self.depth_specific = nn.ModuleList([
            DepthSpecificLayer() for _ in range(depth)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.depth_specific):
            x = self.shared_layer(x)  # Shared
            x = layer(x)  # Depth-specific
        return x
```

**Benefits**:
- More expressive than full sharing
- Still regularized
- Can adapt per depth

**Drawbacks**:
- More parameters
- Need to decide what to share

---

### 2.3 No Weight Sharing (Separate Modules)

Each depth has **separate parameters**.

```python
class UnsharedRecursive(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            SomeLayer() for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

**Benefits**:
- Most expressive
- Each layer can specialize
- Standard approach (ResNet, Transformer)

**Drawbacks**:
- Most parameters
- No recursion (just a stack)

---

## 3. Implementation Strategies for NeuroScript v2

### 3.1 Recursion as First-Class Construct

Add `recurse` keyword to language:

```neuroscript
neuron RecursiveBlock(depth: Nat):
  in: [batch, seq, dim]
  out: [batch, seq, dim]

  graph:
    in -> recurse(depth):
      step: TransformerBlock(dim)
    -> out
```

**Lowering**:
```python
# Compile to:
def forward(self, x):
    for _ in range(self.depth):
        x = self.step(x)
    return x
```

---

### 3.2 Match-Based Recursion (Current Approach)

Use pattern matching with guards:

```neuroscript
neuron RecursiveBlock(depth: Nat):
  in: [batch, seq, dim]
  out: [batch, seq, dim]

  graph:
    in -> match:
      [*] where depth > 0:
        -> TransformerBlock(dim)
        -> RecursiveBlock(depth - 1)
        -> out
      [*]:
        -> Identity() -> out
```

**Lowering**:
- **Option A**: Unroll at compile time (if depth known)
- **Option B**: Generate loop (if depth dynamic)

---

### 3.3 Adaptive Loop Construct

Built-in support for learned halting:

```neuroscript
neuron AdaptiveBlock:
  in: [batch, seq, dim]
  out: [batch, seq, dim]

  let:
    process = TransformerBlock(dim)
    halt = Linear(dim, 1) -> Sigmoid()

  graph:
    in -> loop.adaptive:
      step: process
      halt: halt
      threshold: 0.95
      max_steps: 32
    -> out
```

**Lowering**: Generate full ACT logic (see section 1.2)

---

### 3.4 Tree/Graph Recursion

Native support for structured data:

```neuroscript
# Define recursive data type
type Tree<T> = Leaf(T) | Node(List<Tree<T>>)

neuron TreeEncoder:
  in: Tree<Token>
  out: [dim]

  graph:
    in -> match:
      Leaf(x): Embed(x) -> out
      Node(children):
        children -> map(TreeEncoder) -> Sum() -> out
```

**Lowering**: Compile to recursive Python function with memoization

---

## 4. Optimization Strategies

### 4.1 Loop Unrolling

**When applicable**: Static recursion with small depth

```python
# Before optimization:
for _ in range(3):
    x = layer(x)

# After unrolling:
x = layer(x)
x = layer(x)
x = layer(x)
```

**Benefits**:
- Eliminates loop overhead
- Enables instruction pipelining
- Better for XLA/TorchScript

---

### 4.2 Loop Fusion

**When applicable**: Multiple sequential recursions

```python
# Before:
for _ in range(N):
    x = layer1(x)
for _ in range(N):
    x = layer2(x)

# After fusion:
for _ in range(N):
    x = layer1(x)
    x = layer2(x)
```

**Benefits**:
- Fewer memory reads/writes
- Better cache locality

---

### 4.3 Operator Fusion

**When applicable**: Within each recursive step

```python
# Before:
x = layer_norm(x)
x = linear(x)
x = relu(x)

# After fusion (custom kernel):
x = fused_ln_linear_relu(x)
```

**Benefits**:
- Fewer kernel launches (GPU)
- Less memory movement

---

### 4.4 Gradient Checkpointing

**When applicable**: Deep recursion (memory-bound)

```python
# Use PyTorch's checkpoint to trade compute for memory
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for _ in range(self.depth):
        x = checkpoint(self.layer, x)  # Don't store activations
    return x
```

**Benefits**:
- Reduces memory usage O(N) → O(1)
- Allows deeper recursion

**Drawbacks**:
- Recomputes activations on backward pass (slower)

---

## 5. Backend Considerations

### 5.1 PyTorch

**Supports**:
- ✅ Dynamic control flow (Python loops)
- ✅ `torch.jit.script` for optimization
- ✅ `torch.fx` for graph transformations

**Strategy**:
- Use Python loops for dynamic recursion
- Compile with TorchScript for static recursion

---

### 5.2 JAX

**Supports**:
- ✅ `jax.lax.while_loop` for dynamic loops
- ✅ `jax.lax.fori_loop` for static loops
- ✅ `jax.lax.scan` for accumulation

**Strategy**:
```python
# Static recursion:
def body(carry, _):
    return layer(carry), None

final, _ = jax.lax.scan(body, x, None, length=depth)

# Dynamic recursion:
def cond(state):
    return state['halt_prob'] < threshold

def body(state):
    x = layer(state['x'])
    halt = halt_net(x)
    return {'x': x, 'halt_prob': state['halt_prob'] + halt}

final = jax.lax.while_loop(cond, body, init_state)
```

---

### 5.3 ONNX

**Supports**:
- ✅ `Loop` operator (while/for loops)
- ❌ Limited dynamic control flow

**Strategy**:
- Static recursion: Unroll to sequential ops
- Dynamic recursion: Use `Loop` operator (limited adoption)

**Challenge**: Many runtimes don't support dynamic shapes well

---

## 6. Recommendations for NeuroScript v2

### Phase 1: Static Recursion Only
**Start simple**: Only support compile-time depth

```neuroscript
neuron Stack(depth: Nat = 4):
  in -> match:
    [*] where depth > 0:
      -> Layer() -> Stack(depth - 1) -> out
    [*]:
      -> Identity() -> out
```

**Lowering**: Always unroll to loop

---

### Phase 2: Add Dynamic Recursion
**Add built-in**: `loop.while` construct

```neuroscript
neuron Dynamic:
  in -> loop.while(condition):
    step: Layer()
    max_steps: 100
  -> out
```

**Lowering**:
- PyTorch: Python while loop
- JAX: `jax.lax.while_loop`
- ONNX: `Loop` operator (warn if runtime doesn't support)

---

### Phase 3: Add Adaptive Loops
**Add built-in**: `loop.adaptive` with ACT

```neuroscript
neuron Adaptive:
  in -> loop.adaptive(halt_net, threshold):
    step: Layer()
  -> out
```

**Lowering**: Generate full ACT logic per backend

---

### Phase 4: Structural Recursion
**Add recursive types**: Trees, graphs, lists

```neuroscript
type Tree<T> = Leaf(T) | Node(List<Tree<T>>)

neuron TreeRNN:
  in: Tree<Token> -> out: [dim]
```

**Lowering**: Recursive functions with batching optimizations

---

## 7. Example: Universal Transformer in v2

```neuroscript
# Static version (compile-time depth)
neuron UniversalTransformerStatic(
  depth: Nat = 6,
  dim: Nat = 512,
  heads: Nat = 8
):
  in: [batch, seq, dim]
  out: [batch, seq, dim]

  let:
    layer = TransformerBlock(dim, heads)

  graph:
    in -> loop.fixed(depth):
      step: layer
    -> out


# Dynamic version (learned halting)
neuron UniversalTransformerDynamic(
  dim: Nat = 512,
  heads: Nat = 8,
  threshold: Float = 0.95
):
  in: [batch, seq, dim]
  out: [batch, seq, dim]

  let:
    layer = TransformerBlock(dim, heads)
    halt = Linear(dim, 1) -> Sigmoid()

  graph:
    in -> loop.adaptive:
      step: layer
      halt: halt
      threshold: threshold
      max_steps: 16
    -> out
```

---

## Conclusion

**Key Insights**:
1. **Start with static recursion** (simplest, covers 80% of use cases)
2. **Add dynamic recursion** for adaptive models (ACT, Neural Turing Machines)
3. **Consider structural recursion** for tree/graph data
4. **Optimize per backend** (PyTorch loops, JAX scan, ONNX unrolling)
5. **Weight sharing** is the key to "tiny recursive models"

**Implementation Priority**:
1. ✅ **Phase 1**: Match-based recursion with unrolling (v2 MVP)
2. ✅ **Phase 2**: `loop.fixed(n)` built-in
3. ⏳ **Phase 3**: `loop.while(cond)` for dynamic depth
4. ⏳ **Phase 4**: `loop.adaptive(halt)` for ACT
5. ⏳ **Phase 5**: Recursive types (Tree, Graph)

This progression allows us to ship useful features early while building toward full expressiveness.
