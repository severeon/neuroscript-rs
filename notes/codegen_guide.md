# **Starting Codegen (What to aim for first — the minimal viable direction)**

You want to resist the temptation to go architecturally huge.
The right approach is:

## **Stage 1 — “Direct Lowering” (no smartness, no inference)**

Translation pipeline:

```neuroscript
NeuroScript IR
    ↓ Lowering
PyTorch nn.Module skeleton
    ↓ Emit Python source
Generated Python file
```

### **The Minimal Thing to Lower**

Lower only:

* **Primitive nodes** → instantiate referenced PyTorch modules
* **Composite nodes** → generate a class with a `.forward` containing sequential/inlined calls
* **Connections** → generate variable names and call structure
* **Tuple unpacking** → Python tuple assignment
* **Match expressions** → `if/elif/else` shape guards (initially: best-effort, or static order)

This is enough to generate working PyTorch code for every example in your README.

### **You do *not* need to solve**

* shape inference
* graph optimizations
* lazy references
* multi-backend support
* kwargs marshalling
* module caching
* recursion
* batching semantics
* loop constructs

Those come later.

For the first pass, the philosophy should be:

> **"Take the IR and mechanically map each node to a Python snippet."**

---

## **Concrete: Minimal Codegen Architecture**

### **1. Generate a PyTorch class per top-level neuron**

```python
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = Linear(dim, dim * 4)
        self.gelu_2 = GELU()
        self.linear_3 = Linear(dim * 4, dim)
```

### **2. Forward pass as direct call graph**

```python
def forward(self, x):
    x0 = x
    x1 = self.linear_1(x0)
    x2 = self.gelu_2(x1)
    x3 = self.linear_3(x2)
    return x3
```

You already have all information needed to generate this perfectly.

### **3. Node naming strategy**

A deterministic naming scheme keeps debugging crisp:

```txt
<neuron_name>_<instance_id>_<port_name?>
```

Example:

```txt
Linear_1
Linear_2
GELU_3
```

Since your IR already has opaque node references, that’s simple.

### **4. Imports from stdlib registry**

You already built this!

Just emit:

```python
from neuroscript_runtime.primitives import Linear, GELU, LayerNorm
```

Based on the ImplRefs encountered.

---

## **Progression Path for the Codegen**

### **Phase 0 — Bare bones**

* [x] support primitives
* [x] support sequential pipelines
* [x] support multi-step graphs
* [x] support tuple unpacking
* [x] support match arms (linear `if` chains)

**This alone unlocks all your examples.**

### **Phase 1 — Polish**

* [x] factor out repeated imports
* [x] ensure deterministic order
* [ ] generate docstrings
* [ ] add shape comments to generated code
* [x] handle primitive kwargs properly

### **Phase 2 — Fancy**

* [ ] automatic shape inference
* [ ] module caching
* [ ] hoisting subgraphs
* [ ] multi-backend support
* [ ] exporting ONNX/JAX
* [ ] optimization passes

### **Phase 3 — Wild**

* [ ] generate backward graphs
* [ ] speculative compilation
* [ ] staged execution
* [ ] mixed-backend execution plans
* [ ] integrating with the apoptosis experiments (!)

But not now.
