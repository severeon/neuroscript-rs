# First-Order Neurons: Refactoring Plan

## Executive Summary

To support first-order neurons (neurons as parameters), we need to extend NeuroScript to allow:
1. **Neuron references as values** - `Sequential(12, MyNeuron)` where `MyNeuron` is a type, not a call
2. **Deferred parameter application** - `pipeline(d_model, num_heads)` where `pipeline` was bound to `Sequential(12, MyNeuron)` and the params are passed through to each MyNeuron instance
3. **Meta-neurons** - Higher-order neurons that accept neuron types as parameters and instantiate them

## Current Architecture Analysis

### 1. Value System (src/interfaces.rs:134-152)

**Current State:**
```rust
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Name(String),       // Variable reference
    BinOp { ... },
    Call { ... },       // Eager call with args
}
```

**Gap:** No `NeuronRef` or `Type` variant to represent a neuron as a first-class value.

**Impact:**
- Cannot pass `MyNeuron` as a parameter
- Cannot distinguish between `MyNeuron` (type reference) and `myvar` (variable reference)
- No way to represent partial application or deferred instantiation

**Refactoring Needed:**
```rust
pub enum Value {
    // ... existing variants ...

    /// Reference to a neuron type (not instantiated)
    NeuronRef(String),

    /// Partial application - neuron with some but not all parameters bound
    PartialCall {
        neuron: Box<Value>,  // Could be NeuronRef or another PartialCall
        args: Vec<Value>,
        kwargs: Vec<(String, Value)>,
    },
}
```

### 2. Parameter System (src/interfaces.rs:229-233)

**Current State:**
```rust
pub struct Param {
    pub name: String,
    pub default: Option<Value>,  // Only Value-typed defaults
}
```

**Gap:** Parameters have no type annotations. Cannot specify "this parameter expects a neuron type".

**Impact:**
- No way to declare `neuron Sequential(n, layer_type): ...`
- No validation that a parameter is a neuron vs a scalar
- Cannot enforce constraints on neuron parameters

**Refactoring Needed:**
```rust
pub enum ParamType {
    Value,              // Any scalar value (int, float, string, etc.)
    Neuron,             // A neuron type
    Shape,              // A shape pattern
    Any,                // No constraint (default for backward compat)
}

pub struct Param {
    pub name: String,
    pub param_type: ParamType,
    pub default: Option<Value>,
}
```

### 3. Binding System (src/interfaces.rs:206-213)

**Current State:**
```rust
pub struct Binding {
    pub name: String,
    pub call_name: String,     // Must be a neuron name
    pub args: Vec<Value>,      // Must provide all args now
    pub kwargs: Vec<(String, Value)>,
}
```

**Gap:** Bindings are always full calls. Cannot bind partial applications.

**Example:**
```neuroscript
let:
    layer = TransformerBlock  # ERROR: Not a call!

# Want to write:
let:
    layer = TransformerBlock(d_model)  # Partial - still needs num_heads

# Or even:
let:
    layer = Sequential(12, TransformerBlock)  # Meta-neuron with neuron param
```

**Refactoring Needed:**
```rust
pub struct Binding {
    pub name: String,
    pub value: Value,  // Can be NeuronRef, PartialCall, or full Call
}
```

This allows:
- `layer = TransformerBlock` → `Value::NeuronRef("TransformerBlock")`
- `layer = TransformerBlock(512)` → `Value::PartialCall { neuron: NeuronRef("TransformerBlock"), args: [512], ... }`
- `layer = Sequential(12, TransformerBlock)` → `Value::Call { name: "Sequential", args: [12, NeuronRef("TransformerBlock")], ... }`

### 4. Call System (Endpoint::Call)

**Current State:**
```rust
Endpoint::Call {
    name: String,
    args: Vec<Value>,
    kwargs: Vec<(String, Value)>,
    id: usize,
}
```

**Gap:** Calls are always immediate - cannot defer argument application.

**Example Need:**
```neuroscript
neuron Sequential(n, layer_type):
    in: [*shape]
    out: [*shape]
    # Need to instantiate `layer_type` n times
    # But layer_type might need parameters from Sequential's caller!
```

**Refactoring Needed:**
- When a parameter is a `NeuronRef`, the meta-neuron needs to instantiate it
- If the referenced neuron requires parameters, they must be threaded through
- Two approaches:
  1. **Explicit parameter passing**: `Sequential(12, MyNeuron, d_model=512)`
  2. **Parameter currying**: Bind once, apply params at call site

### 5. Neuron Definition (src/interfaces.rs:237-243)

**Current State:**
```rust
pub struct NeuronDef {
    pub name: String,
    pub params: Vec<Param>,    // No type annotations
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub body: NeuronBody,
}
```

**Gap:** Cannot distinguish regular neurons from meta-neurons (those accepting neuron params).

**Impact:**
- No validation that neuron parameters are used correctly
- Codegen cannot generate different instantiation logic for meta-neurons
- No way to express "this neuron is parameterized by another neuron"

**Refactoring Needed:**
Add type info to params (see Param refactoring above). Optionally add a flag:

```rust
pub struct NeuronDef {
    // ... existing fields ...
    pub is_meta: bool,  // Auto-set if any param has type Neuron
}
```

### 6. Codegen System (src/codegen/)

**Current State:**
- `generate_module_instantiations()` assumes all Calls have concrete neuron names
- `value_to_python_impl()` converts Values to Python, but only handles scalars and names
- No support for generating loops or repeated instantiations

**Gap:** Cannot generate code for meta-neurons that:
1. Accept neuron type parameters
2. Instantiate them multiple times
3. Thread parameters through to inner neurons

**Example Target:**
```neuroscript
neuron Sequential(n, layer_type):
    in: [*, dim]
    out: [*, dim]
    # Implementation TBD
```

Should generate something like:
```python
class Sequential(nn.Module):
    def __init__(self, n, layer_type, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            layer_type(**layer_kwargs) for _ in range(n)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

**Refactoring Needed:**
- Detect when a parameter is a `Value::NeuronRef`
- Generate `nn.ModuleList` for repeated instantiations
- Thread kwargs through to inner neuron instantiations
- Handle partial application (some args bound, some deferred)

## Design Questions to Resolve

### Q1: Syntax for Neuron References

How should users write neuron references?

**Option A: Implicit (preferred)**
```neuroscript
neuron Sequential(n, layer):
    # `layer` is recognized as neuron because it's used as a type
    let:
        instance = layer(d_model=512)
```

**Option B: Explicit annotation**
```neuroscript
neuron Sequential(n, layer: Neuron):
    # Explicit type annotation
```

**Option C: Sigil**
```neuroscript
neuron Sequential(n, &layer):
    # & indicates type parameter
```

**Recommendation:** Start with Option A, add Option B later for clarity/validation.

### Q2: Parameter Threading

How do parameters flow to inner neurons?

**Option A: Explicit passing (simpler)**
```neuroscript
neuron Sequential(n, layer, **layer_params):
    # User must explicitly pass layer params
    graph:
        in -> layer(**layer_params) -> ...
```

**Option B: Automatic currying (more magical)**
```neuroscript
let:
    pipeline = Sequential(12, TransformerBlock)

# Later...
graph:
    in -> pipeline(d_model=512, num_heads=8)  # Auto-threads to TransformerBlock
```

**Recommendation:** Start with Option A, consider Option B as sugar later.

### Q3: Validation Strategy

How do we validate that neuron parameters are used correctly?

**Challenges:**
- At parse time, we don't know if a `Name` refers to a neuron param or value param
- Need whole-program analysis to track parameter types
- Codegen needs to know which params are neurons to generate different code

**Recommendation:**
1. **Parser:** Accept both `layer` and `layer(args)` as valid endpoints
2. **Validator:** Add new pass to track parameter types:
   - If param is used in a Call position → likely Value
   - If param is used as a type in binding/instantiation → Neuron
   - Infer types, report ambiguities
3. **Codegen:** Check param types to generate different instantiation code

### Q4: Scope and Binding

How do neuron references interact with let/set bindings?

```neuroscript
neuron Meta(layer_type):
    let:
        layer = layer_type  # Is this valid?
        instance = layer_type(512)  # Is this valid?
    graph:
        in -> layer(d_model=512)  # Which layer?
        in -> instance  # Which instance?
```

**Recommendation:**
- `layer = layer_type` → Binds `layer` to the same neuron reference (alias)
- `instance = layer_type(512)` → Binds `instance` to a partial call
- First graph line: Error - cannot call unbound neuron ref without args
- Second graph line: Valid - calls the partial application, needs remaining params

## Implementation Phases

### Phase 1: Extend Value System ✓ Core Types

**Tasks:**
1. Add `Value::NeuronRef(String)` variant
2. Add `Value::PartialCall { neuron, args, kwargs }` variant
3. Update `value_to_python_impl()` to handle new variants
4. Update parser to create `NeuronRef` when appropriate
5. Add tests for new value types

**Files to modify:**
- `src/interfaces.rs` - Add enum variants
- `src/parser/core.rs` - Parse neuron references
- `src/codegen/utils.rs` - Convert to Python
- Tests

**Success criteria:**
- Can parse `let: x = MyNeuron` (not a call)
- Can parse `let: y = MyNeuron(512)` (partial call)
- IR correctly represents these distinctions

### Phase 2: Parameter Type System ✓ Type Annotations

**Tasks:**
1. Add `ParamType` enum
2. Extend `Param` struct with `param_type` field
3. Add parser support for type annotations (optional)
4. Update examples with type annotations
5. Add validation pass to infer/check parameter types

**Files to modify:**
- `src/interfaces.rs` - Add ParamType, extend Param
- `src/parser/core.rs` - Parse type annotations if present
- `src/validator/core.rs` - Add type inference pass
- Examples

**Success criteria:**
- Can annotate parameters with types
- Validator infers types when not annotated
- Errors on type mismatches (e.g., passing int where neuron expected)

### Phase 3: Binding System Refactor ✓ Flexible Bindings

**Tasks:**
1. Change `Binding.value` from separate fields to single `Value`
2. Update parser to create bindings with any Value
3. Update validator to check binding values
4. Update codegen to handle different binding value types
5. Add tests for neuron ref bindings

**Files to modify:**
- `src/interfaces.rs` - Modify Binding struct
- `src/parser/core.rs` - Parse flexible bindings
- `src/validator/core.rs` - Validate new binding types
- `src/codegen/instantiation.rs` - Handle different value types
- Tests

**Success criteria:**
- Can bind neuron references: `let: x = MyNeuron`
- Can bind partial calls: `let: y = MyNeuron(512)`
- Bindings work in graph connections

### Phase 4: Meta-Neuron Codegen ✓ Code Generation

**Tasks:**
1. Detect meta-neurons (neurons with Neuron-typed params)
2. Generate `nn.ModuleList` for repeated instantiations
3. Handle parameter threading to inner neurons
4. Generate loops in forward() for repeated layers
5. Add primitive `Sequential` neuron to stdlib

**Files to modify:**
- `src/codegen/generator.rs` - Detect meta-neurons
- `src/codegen/instantiation.rs` - Generate ModuleList code
- `src/codegen/forward.rs` - Generate loop code
- `src/stdlib_registry.rs` - Add Sequential primitive (or implement in NS)
- `stdlib/MetaNeurons.ns` - Add Sequential definition

**Example target:**
```neuroscript
neuron Sequential(n, layer_type):
    in: [*shape]
    out: [*shape]
    impl: external(repeat_count=n, layer=layer_type)
```

Or pure NeuroScript implementation (requires loops - Phase 5):
```neuroscript
neuron Sequential(n, layer_type, **layer_kwargs):
    in: [*shape]
    out: [*shape]
    set:
        # Eager instantiation of n layers
        layers = [layer_type(**layer_kwargs) for _ in range(n)]
    graph:
        in -> layers[0] -> layers[1] -> ... -> layers[n-1] -> out
```

**Success criteria:**
- Can define Sequential(n, layer_type)
- Generates correct PyTorch code with ModuleList
- Can use: `pipeline = Sequential(12, TransformerBlock(512, 8))`

### Phase 5: Full First-Order Support ✓ Advanced Features

**Tasks:**
1. Add loop constructs to graph syntax
2. Support parameter currying at call sites
3. Add higher-order neuron combinators (Map, Fold, etc.)
4. Optimize repeated neuron instantiations
5. Add comprehensive examples and docs

**Files to modify:**
- All of the above
- `CLAUDE.md` - Update with first-order neuron docs
- `examples/` - Add meta-neuron examples
- `stdlib/MetaNeurons.ns` - Expand with combinators

**Success criteria:**
- Can write complex meta-neurons
- Can curry parameters: `pipeline = Sequential(12, TransformerBlock); pipeline(512, 8)`
- Standard library has rich meta-neuron vocabulary

## Key Files to Modify

### Core IR (src/interfaces.rs)
- [ ] Add `Value::NeuronRef(String)`
- [ ] Add `Value::PartialCall { ... }`
- [ ] Add `ParamType` enum
- [ ] Extend `Param` with `param_type: ParamType`
- [ ] Modify `Binding` structure
- [ ] Add `NeuronDef.is_meta()` helper method

### Parser (src/parser/core.rs)
- [ ] Parse neuron references in bindings
- [ ] Parse partial calls
- [ ] Parse type annotations on parameters (optional)
- [ ] Distinguish `MyNeuron` from `myvar` in context

### Validator (src/validator/core.rs)
- [ ] Add parameter type inference pass
- [ ] Validate neuron reference usage
- [ ] Check that neuron params exist
- [ ] Validate partial call arities

### Codegen (src/codegen/)
- [ ] `utils.rs`: Handle `NeuronRef` and `PartialCall` in `value_to_python_impl()`
- [ ] `instantiation.rs`: Detect meta-neurons, generate ModuleList
- [ ] `forward.rs`: Generate loops for repeated layers
- [ ] `generator.rs`: Track neuron-typed parameters

### Standard Library
- [ ] Add `Sequential(n, layer_type)` meta-neuron
- [ ] Add `Parallel(layer_type, ...)` meta-neuron
- [ ] Add examples using meta-neurons

## Example Usage Evolution

### Current (Phase 2 - Recursion):
```neuroscript
neuron Stack(depth):
    in: [*, dim]
    out: [*, dim]
    let:
        recurse = Stack(depth - 1)
    graph:
        in -> match:
            [*, d] where depth > 0: Layer(d) -> recurse -> out
            [*, d]: out
```

**Limitation:** Must manually write recursion logic. Hard-coded to specific layer type.

### Target (Phase 5 - First-Order):
```neuroscript
neuron Sequential(n, layer_type):
    in: [*shape]
    out: [*shape]
    graph:
        in -> [layer_type() for _ in range(n)] -> out

# Usage:
neuron MyModel:
    let:
        stack = Sequential(12, TransformerBlock(512, 8))
    graph:
        in -> stack -> out
```

**Benefits:**
- Generic over layer type
- Cleaner syntax
- Reusable meta-neurons
- Composable building blocks

## Testing Strategy

### Unit Tests
- [ ] Value enum variant parsing
- [ ] Parameter type inference
- [ ] Binding with neuron refs
- [ ] Meta-neuron detection
- [ ] Code generation for ModuleList

### Integration Tests
- [ ] Simple Sequential neuron
- [ ] Nested meta-neurons (Sequential of Sequential)
- [ ] Partial application
- [ ] Parameter threading
- [ ] Currying at call sites

### Example Programs
- [ ] `examples/30-first-order-basic.ns` - Simple Sequential
- [ ] `examples/31-first-order-nested.ns` - Nested meta-neurons
- [ ] `examples/32-first-order-partial.ns` - Partial application
- [ ] `stdlib/Sequential.ns` - Full Sequential implementation

## Open Questions

1. **Syntax bikeshedding:** How to spell type annotations? `layer: Neuron` vs `layer: neuron` vs implicit?

2. **Runtime overhead:** How much overhead does parameter threading add? Can we optimize it away?

3. **Type inference:** How aggressive should type inference be? Should we require annotations for clarity?

4. **Backward compatibility:** Can we add this without breaking existing code? (Answer: likely yes if we default param_type to Any)

5. **Scope of Phase 1:** Should we implement full first-order support in one shot, or incrementally? (Recommendation: incremental per phases above)

## Related Future Work

- **Generic neurons:** `neuron Transformer<T: TransformerBlock>` - full parametric polymorphism
- **Higher-kinded types:** Neurons that return neurons
- **Type classes/traits:** Constraints on neuron parameters (e.g., "must have shape [*, dim]")
- **Compile-time instantiation:** Unroll loops at compile time for known `n`

## Summary

This refactoring enables treating neurons as first-class values, unlocking:
- ✅ Generic meta-neurons like `Sequential(n, layer)`
- ✅ Higher-order composition patterns
- ✅ Cleaner abstractions for repeated structures
- ✅ Reduced boilerplate in user code
- ✅ Foundation for advanced type system features

**Complexity:** Moderate to High
- Touches all major systems (parser, IR, validator, codegen)
- Requires careful type tracking
- Backward compatibility manageable with defaults

**Value:** Very High
- Core feature for practical neural architecture composition
- Aligns with "neurons all the way down" philosophy
- Enables powerful abstractions users expect from composition language
