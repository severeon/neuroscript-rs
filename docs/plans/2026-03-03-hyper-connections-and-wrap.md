# Hyper-Connections & `@wrap` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable higher-order neuron codegen, add the `@wrap` annotation for inline higher-order composition, implement HC runtime primitives, and add HC to the standard library.

**Architecture:** Fix the codegen/validator gap for `Neuron`-typed parameters so context bindings like `inner = layer(dim)` (where `layer: Neuron`) produce correct Python. Then add `@wrap` as a grammar+desugaring feature that rewrites to standard higher-order calls before validation. Finally, implement the HC primitives in Python and add `.ns` stdlib files.

**Tech Stack:** Rust (pest PEG grammar, AST builder, codegen), Python (PyTorch nn.Module primitives), NeuroScript (.ns stdlib)

---

## Phase 1: Close the Higher-Order Codegen Gap

### Task 1: Track Neuron-typed params in CodeGenerator

**Files:**
- Modify: `src/interfaces.rs:113` (change `current_neuron_params` type)
- Modify: `src/codegen/generator.rs:134,147` (populate new field)

**Step 1: Change `current_neuron_params` from `HashSet<String>` to carry type info**

In `src/interfaces.rs`, the `CodeGenerator` struct has:
```rust
pub current_neuron_params: HashSet<String>,
```

Add a new field alongside it:
```rust
/// Names of parameters with `: Neuron` type annotation (higher-order neuron params)
pub neuron_typed_params: HashSet<String>,
```

**Step 2: Initialize in constructor**

In `src/codegen/generator.rs:134`, add:
```rust
neuron_typed_params: HashSet::new(),
```

**Step 3: Populate in `generate_neuron`**

In `src/codegen/generator.rs:147`, after the existing `current_neuron_params` line, add:
```rust
self.neuron_typed_params = neuron.params.iter()
    .filter(|p| p.type_annotation.as_ref() == Some(&ParamType::Neuron))
    .map(|p| p.name.clone())
    .collect();
```

**Step 4: Verify it compiles**

Run: `cargo check`
Expected: PASS (no behavior change yet)

**Step 5: Commit**

```
feat: track neuron-typed params in CodeGenerator
```

---

### Task 2: Fix standalone binding codegen for Neuron-typed params

**Files:**
- Modify: `src/codegen/instantiation.rs:62-136` (standalone bindings loop)
- Test: `src/codegen/tests.rs` (new test)

**Step 1: Write the failing test**

Add to `src/codegen/tests.rs`:
```rust
#[test]
fn test_higher_order_neuron_passthrough() {
    let source = r#"
neuron Wrapper(layer: Neuron, dim):
    in: [*, dim]
    out: [*, dim]
    context:
        inner = layer
    graph:
        in -> inner -> out
"#;
    let program = parse(source).unwrap();
    let code = generate_pytorch(&program, "Wrapper").unwrap();
    assert!(code.contains("self.inner = layer"), "Expected pass-through assignment, got:\n{}", code);
    // Should NOT try to import "layer" as a primitive
    assert!(!code.contains("from "), "Should not generate import for neuron param:\n{}", code);
}

#[test]
fn test_higher_order_neuron_construct() {
    let source = r#"
neuron Wrapper(layer: Neuron, dim):
    in: [*, dim]
    out: [*, dim]
    context:
        inner = layer(dim)
    graph:
        in -> inner -> out
"#;
    let program = parse(source).unwrap();
    let code = generate_pytorch(&program, "Wrapper").unwrap();
    // Should call the class with args: self.inner = layer(dim)
    assert!(code.contains("self.inner = layer(dim)"), "Expected construct-from-type, got:\n{}", code);
    assert!(!code.contains("from "), "Should not generate import for neuron param:\n{}", code);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test test_higher_order_neuron_passthrough test_higher_order_neuron_construct -- --nocapture`
Expected: FAIL (validation errors because the validator produces dummy `[]` shapes that cause PortMismatch)

**Step 3: Fix validator — propagate neuron-typed param shapes through bindings**

The validator in `src/validator/symbol_table.rs:306-316` returns a dummy port with empty shape `Shape { dims: vec![] }` for neuron-typed params. When a context binding calls a neuron-typed param (e.g., `inner = layer` or `inner = layer(dim)`), the binding gets these empty-shape ports, and subsequent connections fail shape compatibility.

In `src/validator/symbol_table.rs`, modify the `Endpoint::Call` case in `resolve_endpoint` (around line 306). When the call target is a neuron-typed param, instead of returning an empty shape, return a wildcard shape `[*]` that matches anything:

```rust
// 4. Check if this is a neuron-typed parameter (higher-order neuron)
let is_neuron_param = ctx.neuron.params.iter().any(|p| {
    p.name == *name && p.type_annotation.as_ref() == Some(&ParamType::Neuron)
});
if is_neuron_param {
    // Neuron-typed param: return a wildcard port to allow validation to continue.
    // The actual shape is unknown until the caller provides a concrete neuron.
    return Ok(vec![Port {
        name: "default".to_string(),
        shape: Shape { dims: vec![Dim::Wildcard] },
        variadic: false,
    }]);
}
```

Also fix `shapes_compatible` in `src/validator/shapes.rs` to ensure shapes with `Dim::Wildcard` are compatible with anything (this likely already works, but verify).

**Step 4: Fix codegen — detect neuron-typed param bindings in instantiation**

In `src/codegen/instantiation.rs`, in the standalone bindings loop (around line 63), add a check before the `is_primitive` lookup:

```rust
for binding in &standalone_bindings {
    let module_name = binding.name.clone();
    let name = &binding.call_name;
    let args = &binding.args;
    let kwargs = &binding.kwargs;

    // Check if the call target is a Neuron-typed parameter
    if gen.neuron_typed_params.contains(name) {
        // Neuron-typed param: register as submodule without importing
        match &binding.scope {
            Scope::Instance { lazy: false } => {
                if args.is_empty() && kwargs.is_empty() {
                    // Pass-through: the param is already an nn.Module instance
                    writeln!(output, "        self.{} = {}", module_name, name).unwrap();
                } else {
                    // Construct from type: the param is a class reference
                    let (args_str, kwargs_str) = extract_kwargs(args, kwargs);
                    writeln!(output, "        self.{} = {}({}{})", module_name, name, args_str, kwargs_str).unwrap();
                }
                gen.var_names.insert(module_name.clone(), format!("self.{}", module_name));
                instantiated_count += 1;
            }
            // TODO: handle Scope::Static and lazy for neuron-typed params if needed
            _ => {
                // For now, treat non-Instance scopes the same as pass-through
                writeln!(output, "        self.{} = {}", module_name, name).unwrap();
                gen.var_names.insert(module_name.clone(), format!("self.{}", module_name));
                instantiated_count += 1;
            }
        }
        continue; // Skip the normal instantiation path
    }

    // ... existing is_primitive check and instantiation logic ...
```

**Step 5: Also fix anonymous call codegen for neuron-typed params**

In the anonymous calls section (around line 267), add a similar check:

```rust
if let Some(neuron) = gen.program.neurons.get(name.as_str()) {
    if neuron.is_primitive() {
        gen.used_primitives.insert(name.clone());
    }
} else if !gen.neuron_typed_params.contains(name) {
    // Only add to used_primitives if NOT a neuron-typed param
    gen.used_primitives.insert(name.clone());
}
```

**Step 6: Run tests to verify they pass**

Run: `cargo test test_higher_order_neuron -- --nocapture`
Expected: PASS

**Step 7: Run full test suite to check for regressions**

Run: `cargo test`
Expected: All existing tests pass

**Step 8: Test with CLI**

Create `/tmp/test_ho.ns`:
```neuroscript
neuron Wrapper(layer: Neuron, dim):
    in: [*, dim]
    out: [*, dim]
    context:
        inner = layer(dim)
    graph:
        in -> inner -> out
```

Run: `./target/release/neuroscript compile /tmp/test_ho.ns --neuron Wrapper`
Expected: Valid Python output with `self.inner = layer(dim)`

**Step 9: Commit**

```
feat: fix higher-order neuron codegen for Neuron-typed params
```

---

### Task 3: Verify the existing `higher_order_neuron.ns` example still works

**Step 1: Run existing example**

Run: `cargo build --release && ./target/release/neuroscript compile examples/higher_order_neuron.ns --neuron Stack`
Expected: Same output as before (unroll path is unchanged)

**Step 2: Run full integration test suite**

Run: `./test_examples.sh`
Expected: All examples pass

**Step 3: Run snapshot tests**

Run: `cargo test --test integration_tests`
Expected: All snapshots match (or review any changes with `cargo insta review`)

---

## Phase 2: Implement `@wrap` Annotation

### Task 4: Add `@wrap` grammar rules

**Files:**
- Modify: `src/grammar/neuroscript.pest`

**Step 1: Add the `keyword_wrap` and `wrap_annotation` rules**

Add keyword:
```pest
keyword_wrap = @{ "wrap" ~ !ident_cont }
```

Add to the keyword union:
```pest
keyword = _{
    ... | keyword_wrap
}
```

Add the `@wrap` grammar rule. This should slot into `endpoint` as a new alternative:
```pest
# @wrap annotation: inline higher-order neuron composition
# Reference form: @wrap(Wrapper, args): binding_name
# Pipeline form: @wrap(Wrapper, args): -> pipeline
wrap_endpoint = {
    at ~ keyword_wrap ~ lparen ~ call_args ~ rparen ~ colon
    ~ (
        # Pipeline form: -> followed by inline or indented pipeline
        (arrow ~ (NEWLINE ~ indented_pipeline | endpoint ~ ((arrow ~ endpoint) | fat_arrow_step)*))
        # Reference form: just a binding name
      | ident
    )
}
```

Add `wrap_endpoint` to the `endpoint` rule (before `call_endpoint` to avoid ambiguity):
```pest
endpoint = {
    match_eval_expr
  | match_expr
  | if_expr
  | wrap_endpoint
  | tuple_endpoint
  | ref_endpoint
  | call_endpoint
}
```

**Step 2: Verify grammar compiles**

Run: `cargo check`
Expected: PASS (pest grammar compiles but AST builder doesn't handle it yet)

**Step 3: Commit**

```
feat: add @wrap grammar rules to pest
```

---

### Task 5: Add `@wrap` IR representation

**Files:**
- Modify: `src/interfaces.rs` (add WrapExpr struct, extend Endpoint enum)

**Step 1: Add IR types**

```rust
/// A @wrap annotation expression
#[derive(Debug, Clone, PartialEq)]
pub struct WrapExpr {
    /// The higher-order neuron to wrap with (first arg in call_args is the wrapper name)
    pub wrapper_name: String,
    /// Arguments to the wrapper (excluding the first Neuron-typed param)
    pub wrapper_args: Vec<Value>,
    /// Keyword arguments to the wrapper
    pub wrapper_kwargs: Vec<Kwarg>,
    /// The wrapped content: either a reference to an existing binding or an anonymous pipeline
    pub content: WrapContent,
    /// Unique ID for deduplication
    pub id: usize,
}

/// What @wrap wraps
#[derive(Debug, Clone, PartialEq)]
pub enum WrapContent {
    /// Reference form: @wrap(Wrapper, args): existing_binding
    Ref(String),
    /// Pipeline form: @wrap(Wrapper, args): -> X -> Y -> Z
    Pipeline(Vec<Endpoint>),
}
```

Add `Wrap(WrapExpr)` variant to the `Endpoint` enum.

**Step 2: Add Display impl for WrapExpr**

Add to the existing Display implementations in interfaces.rs.

**Step 3: Verify it compiles**

Run: `cargo check`
Expected: PASS (exhaustive match warnings in codegen/validator may appear — that's expected)

**Step 4: Fix exhaustive match warnings**

Add `Endpoint::Wrap(_)` arms to all match statements that handle `Endpoint` variants (in validator, codegen, etc.). For now, these can be `todo!()` or empty matches.

**Step 5: Commit**

```
feat: add WrapExpr IR types and Endpoint::Wrap variant
```

---

### Task 6: Parse `@wrap` in AST builder

**Files:**
- Modify: `src/grammar/ast.rs` (add `build_wrap_endpoint` method)

**Step 1: Add parsing for `wrap_endpoint` rule**

In the endpoint dispatch in `build_endpoint` (or wherever endpoints are matched), add:
```rust
Rule::wrap_endpoint => self.build_wrap_endpoint(inner),
```

Implement `build_wrap_endpoint`:
```rust
fn build_wrap_endpoint(&mut self, pair: Pair<Rule>) -> Endpoint {
    let mut inner = pair.into_inner();
    // Skip @ and "wrap" tokens (consumed by grammar)
    // Parse call_args (wrapper name + extra args)
    let call_args_pair = inner.next().unwrap(); // call_args
    let (args, kwargs) = self.build_call_args(call_args_pair);

    // First positional arg must be a Name (the wrapper neuron)
    let wrapper_name = match &args[0] {
        Value::Name(n) => n.clone(),
        _ => panic!("@wrap first argument must be a neuron name"),
    };
    let wrapper_args = args[1..].to_vec();

    // Parse content: either ident (reference) or arrow + pipeline
    let content_pair = inner.next().unwrap();
    let content = match content_pair.as_rule() {
        Rule::ident => WrapContent::Ref(content_pair.as_str().to_string()),
        Rule::arrow => {
            // Pipeline form: collect subsequent endpoints
            let mut pipeline = Vec::new();
            for ep_pair in inner {
                pipeline.push(self.build_endpoint(ep_pair));
            }
            WrapContent::Pipeline(pipeline)
        }
        _ => { /* handle indented_pipeline */ }
    };

    let id = self.next_id();
    Endpoint::Wrap(WrapExpr {
        wrapper_name,
        wrapper_args,
        wrapper_kwargs: kwargs,
        content,
        id,
    })
}
```

Note: The exact parsing logic will depend on how pest structures the parse tree for the grammar rule. The implementer should inspect the pairs with `pair.as_rule()` and adjust accordingly.

**Step 2: Write a parse test**

```rust
#[test]
fn test_parse_wrap_ref() {
    let source = r#"
neuron Test(dim):
    in: [*, dim]
    out: [*, dim]
    context:
        attn = MultiHeadSelfAttention(dim, 8)
    graph:
        in -> @wrap(HyperConnect, 4, dim, 0): attn -> out
"#;
    let program = parse(source).unwrap();
    let neuron = program.neurons.get("Test").unwrap();
    // Verify the wrap endpoint was parsed
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        // The connection should contain a Wrap endpoint
        assert!(connections.iter().any(|c| matches!(c.destination, Endpoint::Wrap(_))
            || matches!(c.source, Endpoint::Wrap(_))));
    }
}
```

**Step 3: Run test**

Run: `cargo test test_parse_wrap -- --nocapture`

**Step 4: Commit**

```
feat: parse @wrap annotation in AST builder
```

---

### Task 7: Implement `@wrap` desugaring pass

**Files:**
- Create: `src/desugar.rs` (new module for AST rewrites)
- Modify: `src/lib.rs` (add module, call desugar before validation)

**Step 1: Create the desugaring module**

```rust
//! AST desugaring passes
//!
//! Rewrites syntactic sugar (like @wrap) into standard IR before validation.

use crate::interfaces::*;

/// Desugar all @wrap annotations in a program.
/// Must be called after parsing but before validation.
pub fn desugar_wraps(program: &mut Program) {
    let neuron_names: Vec<String> = program.neurons.keys().cloned().collect();
    for name in &neuron_names {
        if let Some(neuron) = program.neurons.get_mut(name) {
            desugar_neuron_wraps(neuron);
        }
    }
}

fn desugar_neuron_wraps(neuron: &mut NeuronDef) {
    if let NeuronBody::Graph {
        ref mut context_bindings,
        ref mut connections,
        ..
    } = neuron.body
    {
        let mut new_bindings = Vec::new();
        let mut wrap_counter = 0;

        for conn in connections.iter_mut() {
            desugar_endpoint_wraps(&mut conn.source, context_bindings, &mut new_bindings, &mut wrap_counter);
            desugar_endpoint_wraps(&mut conn.destination, context_bindings, &mut new_bindings, &mut wrap_counter);
        }

        // Prepend synthesized bindings to context
        context_bindings.splice(0..0, new_bindings);
    }
}

fn desugar_endpoint_wraps(
    endpoint: &mut Endpoint,
    _existing_bindings: &[Binding],
    new_bindings: &mut Vec<Binding>,
    counter: &mut usize,
) {
    match endpoint {
        Endpoint::Wrap(wrap_expr) => {
            let wrap_id = *counter;
            *counter += 1;

            match &wrap_expr.content {
                WrapContent::Ref(binding_name) => {
                    // Reference form: rewrite to Call endpoint
                    // @wrap(Wrapper, a, b): existing → Wrapper(existing, a, b)
                    let mut call_args = vec![Value::Name(binding_name.clone())];
                    call_args.extend(wrap_expr.wrapper_args.clone());

                    *endpoint = Endpoint::Call {
                        name: wrap_expr.wrapper_name.clone(),
                        args: call_args,
                        kwargs: wrap_expr.wrapper_kwargs.clone(),
                        id: wrap_expr.id,
                        frozen: false,
                    };
                }
                WrapContent::Pipeline(pipeline_endpoints) => {
                    // Pipeline form: synthesize nn.Sequential binding, then rewrite
                    // @wrap(Wrapper, a, b): -> X -> Y →
                    //   context: _wrap_N = __sequential__(X, Y)
                    //   Wrapper(_wrap_N, a, b)
                    let anon_name = format!("_wrap_{}", wrap_id);

                    // Create synthetic binding for the sequential pipeline
                    // The pipeline endpoints are Call endpoints; extract them as args
                    // For codegen, we'll use a special call_name "__sequential__"
                    let seq_args: Vec<Value> = pipeline_endpoints.iter().map(|ep| {
                        match ep {
                            Endpoint::Call { name, args, kwargs, .. } => {
                                Value::Call {
                                    name: name.clone(),
                                    args: args.clone(),
                                    kwargs: kwargs.clone(),
                                }
                            }
                            Endpoint::Ref(port_ref) => Value::Name(port_ref.node.clone()),
                            _ => Value::Name("__unknown__".to_string()),
                        }
                    }).collect();

                    new_bindings.push(Binding {
                        name: anon_name.clone(),
                        call_name: "__sequential__".to_string(),
                        args: seq_args,
                        kwargs: vec![],
                        scope: Scope::Instance { lazy: false },
                        frozen: false,
                        unroll_group: None,
                    });

                    // Rewrite the endpoint to a Call
                    let mut call_args = vec![Value::Name(anon_name)];
                    call_args.extend(wrap_expr.wrapper_args.clone());

                    *endpoint = Endpoint::Call {
                        name: wrap_expr.wrapper_name.clone(),
                        args: call_args,
                        kwargs: wrap_expr.wrapper_kwargs.clone(),
                        id: wrap_expr.id,
                        frozen: false,
                    };
                }
            }
        }
        Endpoint::Match(match_expr) => {
            for arm in &mut match_expr.arms {
                for ep in &mut arm.pipeline {
                    desugar_endpoint_wraps(ep, _existing_bindings, new_bindings, counter);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &mut if_expr.branches {
                for ep in &mut branch.pipeline {
                    desugar_endpoint_wraps(ep, _existing_bindings, new_bindings, counter);
                }
            }
            if let Some(else_branch) = &mut if_expr.else_branch {
                for ep in else_branch {
                    desugar_endpoint_wraps(ep, _existing_bindings, new_bindings, counter);
                }
            }
        }
        _ => {}
    }
}
```

**Step 2: Wire into the compilation pipeline**

In `src/lib.rs`, after `parse()` and before `validate()`, call:
```rust
desugar::desugar_wraps(&mut program);
```

**Step 3: Handle `__sequential__` in codegen instantiation**

In `src/codegen/instantiation.rs`, add a special case for `__sequential__` bindings:
```rust
if name == "__sequential__" {
    // Synthesized by @wrap pipeline form: emit nn.Sequential(...)
    let items: Vec<String> = args.iter().map(|arg| {
        match arg {
            Value::Call { name, args, kwargs } => {
                let (a, k) = extract_kwargs(args, kwargs);
                if a.is_empty() && k.is_empty() {
                    format!("{}()", name)
                } else {
                    format!("{}({}{})", name, a, k)
                }
            }
            Value::Name(n) => n.clone(),
            _ => value_to_python_impl(arg),
        }
    }).collect();

    writeln!(output, "        self.{} = nn.Sequential(", module_name).unwrap();
    for (i, item) in items.iter().enumerate() {
        let comma = if i < items.len() - 1 { "," } else { "," };
        writeln!(output, "            {}{}", item, comma).unwrap();
    }
    writeln!(output, "        )").unwrap();

    // Track primitives used in the sequential
    for arg in args {
        if let Value::Call { name, .. } = arg {
            gen.used_primitives.insert(name.clone());
        }
    }

    gen.var_names.insert(module_name.clone(), format!("self.{}", module_name));
    instantiated_count += 1;
    continue;
}
```

**Step 4: Write tests for @wrap desugaring**

```rust
#[test]
fn test_wrap_ref_codegen() {
    let source = r#"
neuron HyperConnect(layer: Neuron, n, dim, layer_idx):
    in: [*, n, dim]
    out: [*, n, dim]
    graph:
        in -> layer -> out

neuron Test(dim):
    in: [*, dim]
    out: [*, dim]
    context:
        attn = MultiHeadSelfAttention(dim, 8)
    graph:
        in -> @wrap(HyperConnect, 4, dim, 0): attn -> out
"#;
    let program = parse(source).unwrap();
    let code = generate_pytorch(&program, "Test").unwrap();
    assert!(code.contains("HyperConnect(self.attn, 4, dim, 0)"),
        "Expected wrapper call with attn ref, got:\n{}", code);
}

#[test]
fn test_wrap_pipeline_codegen() {
    let source = r#"
neuron HyperConnect(layer: Neuron, n, dim, layer_idx):
    in: [*, n, dim]
    out: [*, n, dim]
    graph:
        in -> layer -> out

neuron Test(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> @wrap(HyperConnect, 4, dim, 0): ->
            LayerNorm(dim)
            Linear(dim, dim)
        -> out
"#;
    let program = parse(source).unwrap();
    let code = generate_pytorch(&program, "Test").unwrap();
    assert!(code.contains("nn.Sequential"),
        "Expected nn.Sequential for pipeline form, got:\n{}", code);
}
```

**Step 5: Run all tests**

Run: `cargo test`
Expected: All pass

**Step 6: Commit**

```
feat: implement @wrap annotation with desugaring pass
```

---

## Phase 3: HC Runtime Primitives

### Task 8: Implement HCWidth, HCDepth, HyperExpand, HyperCollapse in Python

**Files:**
- Create: `neuroscript_runtime/primitives/connections.py`

**Step 1: Implement the primitives**

```python
"""Hyper-Connection primitives for NeuroScript.

Reference: Zhu et al., "Hyper-Connections" (ICLR 2025) — arXiv:2409.19606v3

These modules implement the core hyper-connection operations:
- HyperExpand: Expand single hidden to n copies (network entry)
- HyperCollapse: Collapse n copies via sum (network exit)
- HCWidth: Width connection (mix n hidden vectors, extract layer input + state)
- HCDepth: Depth connection (merge layer output back into hyper state)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperExpand(nn.Module):
    """Expand a single hidden vector to n copies along a new dimension.

    Input:  [*batch, dim]
    Output: [*batch, n, dim]
    """
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        # x: [*batch, dim] -> [*batch, 1, dim] -> [*batch, n, dim]
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.n, x.shape[-1]).clone()


class HyperCollapse(nn.Module):
    """Collapse n copies by summing along the expansion dimension.

    Input:  [*batch, n, dim]
    Output: [*batch, dim]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [*batch, n, dim] -> [*batch, dim]
        return x.sum(dim=-2)


class HCWidth(nn.Module):
    """Width connection: mix n hidden vectors and extract layer input + state.

    Implements Equations 10-13 (dynamic) or direct matrix multiply (static)
    from Algorithm 2 of the paper.

    Input:  [*batch, n, dim]
    Output: (layer_in: [*batch, dim], state: [*batch, n, dim])

    Args:
        n: Number of hidden copies (expansion rate)
        dim: Hidden dimension
        layer_idx: Layer index (for round-robin initialization per Eq. 14)
        dynamic: If True, use input-dependent mixing (Eqs. 10-13).
                 If False, use static learned matrix.
    """
    def __init__(self, n, dim, layer_idx, dynamic=True):
        super().__init__()
        self.n = n
        self.dim = dim
        self.layer_idx = layer_idx
        self.dynamic = dynamic

        if dynamic:
            # Dynamic: project input to mixing weights (Eq. 10-11)
            # alpha has shape (n+1, n) — row 0 is for layer_in extraction,
            # rows 1..n+1 are for state pass-through
            self.proj = nn.Linear(dim, (n + 1) * n, bias=False)
        else:
            # Static: learnable (n+1) x n matrix
            self.alpha = nn.Parameter(torch.zeros(n + 1, n))

        # Initialize per Eq. 14: round-robin based on layer_idx
        self._init_weights()

    def _init_weights(self):
        """Initialize to Pre-Norm residual equivalence (Eq. 14)."""
        n = self.n
        # Am initialization: round-robin assignment
        # For layer m, the "active" copy index is m mod n
        active = self.layer_idx % n

        if self.dynamic:
            # Initialize projection bias so initial alpha matches Eq. 14
            with torch.no_grad():
                self.proj.weight.zero_()
                # We'll set initial bias through the weight matrix
                # so that the output alpha matrix has the round-robin pattern
        else:
            with torch.no_grad():
                self.alpha.zero_()
                # Row 0 (layer input extraction): take from active copy
                self.alpha[0, active] = 1.0
                # Rows 1..n+1 (state): identity pass-through
                for i in range(n):
                    self.alpha[i + 1, i] = 1.0

    def forward(self, x):
        # x: [*batch, n, dim]
        batch_shape = x.shape[:-2]
        n, dim = x.shape[-2], x.shape[-1]

        if self.dynamic:
            # Eq. 10-11: compute input-dependent mixing weights
            # Average across copies to get a summary
            x_mean = x.mean(dim=-2)  # [*batch, dim]
            raw = self.proj(x_mean)  # [*batch, (n+1)*n]
            raw = raw.view(*batch_shape, n + 1, n)
            alpha = torch.tanh(raw)  # Eq. 12: stabilize with tanh
        else:
            alpha = self.alpha  # [n+1, n]

        # Eq. 13: apply mixing
        # alpha[0, :] extracts layer input, alpha[1:, :] produces state
        # layer_in = sum_j(alpha[0,j] * x[..., j, :])
        layer_in = torch.einsum('...jd, ...j -> ...d', x, alpha[..., 0, :])

        # state[i] = sum_j(alpha[i+1, j] * x[..., j, :])
        state = torch.einsum('...jd, ...ij -> ...id', x, alpha[..., 1:, :].transpose(-2, -1))

        return layer_in, state


class HCDepth(nn.Module):
    """Depth connection: merge layer output back into hyper hidden state.

    Implements Equation 5 from the paper.

    Input:  (layer_out: [*batch, dim], state: [*batch, n, dim])
    Output: [*batch, n, dim]

    Args:
        n: Number of hidden copies
        dim: Hidden dimension
        dynamic: If True, use input-dependent depth weights.
                 If False, use static learned weights.
    """
    def __init__(self, n, dim, dynamic=True):
        super().__init__()
        self.n = n
        self.dim = dim
        self.dynamic = dynamic

        if dynamic:
            self.proj = nn.Linear(dim, n, bias=False)
        else:
            # Static: learnable weight vector of length n
            self.beta = nn.Parameter(torch.ones(n))

        self._init_weights()

    def _init_weights(self):
        """Initialize to standard residual (add layer output equally)."""
        if self.dynamic:
            with torch.no_grad():
                self.proj.weight.zero_()
        else:
            with torch.no_grad():
                self.beta.fill_(1.0)

    def forward(self, inputs):
        layer_out, state = inputs
        # layer_out: [*batch, dim], state: [*batch, n, dim]

        if self.dynamic:
            beta = torch.tanh(self.proj(layer_out))  # [*batch, n]
        else:
            beta = self.beta  # [n]

        # Eq. 5: h_out[i] = state[i] + beta[i] * layer_out
        # Expand layer_out: [*batch, dim] -> [*batch, 1, dim]
        layer_expanded = layer_out.unsqueeze(-2)
        # Expand beta: [*batch, n] -> [*batch, n, 1]
        if self.dynamic:
            beta_expanded = beta.unsqueeze(-1)
        else:
            beta_expanded = beta.view(*([1] * (state.dim() - 2)), self.n, 1)

        return state + beta_expanded * layer_expanded
```

**Step 2: Run Python tests**

```bash
source ~/.venv_ai/bin/activate
cd neuroscript_runtime/primitives
python -c "
import torch
from connections import HyperExpand, HyperCollapse, HCWidth, HCDepth

# Test HyperExpand
expand = HyperExpand(4)
x = torch.randn(2, 32, 256)  # [batch, seq, dim]
y = expand(x)
assert y.shape == (2, 32, 4, 256), f'HyperExpand: expected [2,32,4,256], got {y.shape}'

# Test HyperCollapse
collapse = HyperCollapse()
z = collapse(y)
assert z.shape == (2, 32, 256), f'HyperCollapse: expected [2,32,256], got {z.shape}'

# Test HCWidth
width = HCWidth(4, 256, layer_idx=0)
layer_in, state = width(y)
assert layer_in.shape == (2, 32, 256), f'HCWidth layer_in: expected [2,32,256], got {layer_in.shape}'
assert state.shape == (2, 32, 4, 256), f'HCWidth state: expected [2,32,4,256], got {state.shape}'

# Test HCDepth
depth = HCDepth(4, 256)
out = depth((layer_in, state))
assert out.shape == (2, 32, 4, 256), f'HCDepth: expected [2,32,4,256], got {out.shape}'

print('All HC primitive tests passed!')
"
```

**Step 3: Commit**

```
feat: implement HC runtime primitives (HCWidth, HCDepth, HyperExpand, HyperCollapse)
```

---

### Task 9: Register HC primitives in stdlib registry

**Files:**
- Modify: `src/stdlib_registry.rs` (register 4 new primitives)
- Modify: `src/codegen/generator.rs` (add `connections` to embedded_primitives)

**Step 1: Register primitives**

In `src/stdlib_registry.rs`, in `register_all_primitives()`, add:

```rust
// Level 0: Connections (Hyper-Connections)
self.register(
    "HCWidth",
    ImplRef::with_desc(
        "neuroscript_runtime.primitives.connections",
        "HCWidth",
        "Width connection for hyper-connections (mix hidden copies)",
    ),
);

self.register(
    "HCDepth",
    ImplRef::with_desc(
        "neuroscript_runtime.primitives.connections",
        "HCDepth",
        "Depth connection for hyper-connections (merge layer output)",
    ),
);

self.register(
    "HyperExpand",
    ImplRef::with_desc(
        "neuroscript_runtime.primitives.connections",
        "HyperExpand",
        "Expand single hidden to n copies for hyper-connections",
    ),
);

self.register(
    "HyperCollapse",
    ImplRef::with_desc(
        "neuroscript_runtime.primitives.connections",
        "HyperCollapse",
        "Collapse n copies via sum for hyper-connections",
    ),
);
```

**Step 2: Add embedded primitive for bundle mode**

In `src/codegen/generator.rs`, add to the `embedded_primitives!` macro:
```rust
CONNECTIONS,    "connections"    => "connections.py",
```

**Step 3: Verify**

Run: `cargo test stdlib_registry`
Expected: PASS (registry count increases by 4)

**Step 4: Commit**

```
feat: register HC primitives in stdlib registry
```

---

## Phase 4: Standard Library `.ns` Files

### Task 10: Create HC primitive `.ns` definitions

**Files:**
- Create: `stdlib/primitives/HCWidth.ns`
- Create: `stdlib/primitives/HCDepth.ns`
- Create: `stdlib/primitives/HyperExpand.ns`
- Create: `stdlib/primitives/HyperCollapse.ns`

**Step 1: Write the primitive definitions**

`stdlib/primitives/HCWidth.ns`:
```neuroscript
/// Width connection: mix n hidden vectors, extract layer input + state.
/// Part of Hyper-Connections (Zhu et al., ICLR 2025).
neuron HCWidth(n, dim, layer_idx, dynamic=true):
    in: [*batch, n, dim]
    out layer_in: [*batch, dim]
    out state: [*batch, n, dim]
    impl: core,connections/HCWidth
```

`stdlib/primitives/HCDepth.ns`:
```neuroscript
/// Depth connection: merge layer output back into hyper hidden state.
/// Part of Hyper-Connections (Zhu et al., ICLR 2025).
neuron HCDepth(n, dim, dynamic=true):
    in layer_out: [*batch, dim]
    in state: [*batch, n, dim]
    out: [*batch, n, dim]
    impl: core,connections/HCDepth
```

`stdlib/primitives/HyperExpand.ns`:
```neuroscript
/// Expand single hidden to n copies (network entry point for hyper-connections).
neuron HyperExpand(n):
    in: [*batch, dim]
    out: [*batch, n, dim]
    impl: core,connections/HyperExpand
```

`stdlib/primitives/HyperCollapse.ns`:
```neuroscript
/// Collapse n copies via sum (network exit point for hyper-connections).
neuron HyperCollapse:
    in: [*batch, n, dim]
    out: [*batch, dim]
    impl: core,connections/HyperCollapse
```

**Step 2: Validate all primitive files**

Run:
```bash
for f in stdlib/primitives/HC*.ns stdlib/primitives/Hyper*.ns; do
    echo "Validating $f..."
    ./target/release/neuroscript validate "$f"
done
```

**Step 3: Commit**

```
feat: add HC primitive .ns definitions to stdlib
```

---

### Task 11: Create HyperConnect composite neuron

**Files:**
- Create: `stdlib/HyperConnect.ns`

**Step 1: Write the composite neuron**

```neuroscript
/// Hyper-Connection wrapper.
/// Replaces residual connections with learnable depth+width connections.
/// Reference: Zhu et al., "Hyper-Connections" (ICLR 2025) — arXiv:2409.19606v3
neuron HyperConnect(layer: Neuron, n, dim, layer_idx, dynamic=true):
    in: [*batch, n, dim]
    out: [*batch, n, dim]
    graph:
        in -> HCWidth(n, dim, layer_idx, dynamic) -> (layer_in, state)
        layer_in -> layer -> layer_out
        (layer_out, state) -> HCDepth(n, dim, dynamic) -> out
```

**Step 2: Validate**

Run: `./target/release/neuroscript validate stdlib/HyperConnect.ns`
Expected: PASS

**Step 3: Compile and inspect output**

Run: `./target/release/neuroscript compile stdlib/HyperConnect.ns --neuron HyperConnect`
Expected: Valid Python with proper HCWidth/HCDepth instantiation and layer pass-through

**Step 4: Commit**

```
feat: add HyperConnect composite neuron to stdlib
```

---

### Task 12: Create HCTransformerBlock example

**Files:**
- Create: `examples/hc_transformer.ns`

**Step 1: Write the example**

```neuroscript
# Hyper-Connected Transformer Block
# Uses learnable hyper-connections instead of fixed residual connections.
# Reference: Zhu et al., "Hyper-Connections" (ICLR 2025)

neuron HCTransformerBlock(dim, heads, d_ff, n=4, layer_idx=0):
    in: [*batch, n, dim]
    out: [*batch, n, dim]
    context:
        attn = MultiHeadSelfAttention(dim, heads)
        ffn_block = FFN(dim, d_ff)
    graph:
        in ->
            HyperConnect(attn, n, dim, layer_idx * 2)
            HyperConnect(ffn_block, n, dim, layer_idx * 2 + 1)
            out
```

**Step 2: Validate and compile**

Run:
```bash
./target/release/neuroscript validate examples/hc_transformer.ns
./target/release/neuroscript compile examples/hc_transformer.ns --neuron HCTransformerBlock
```

**Step 3: Run full test suite**

Run: `./test_examples.sh && cargo test`
Expected: All pass

**Step 4: Commit**

```
feat: add HC transformer example
```

---

### Task 13: Final integration test and snapshot update

**Step 1: Run all tests**

```bash
cargo test
cargo test --test integration_tests
./test_examples.sh
```

**Step 2: Review and accept snapshot changes**

```bash
cargo insta review
```

Accept all valid new/changed snapshots.

**Step 3: Commit snapshots**

```
test: update snapshots for HC and @wrap features
```

---

## Summary of Files Changed

| Phase | Files | Type |
|-------|-------|------|
| 1 | `src/interfaces.rs` | Modify (add `neuron_typed_params` field) |
| 1 | `src/codegen/generator.rs` | Modify (populate new field) |
| 1 | `src/codegen/instantiation.rs` | Modify (neuron-param binding logic) |
| 1 | `src/validator/symbol_table.rs` | Modify (wildcard shape for neuron params) |
| 1 | `src/codegen/tests.rs` | Modify (new tests) |
| 2 | `src/grammar/neuroscript.pest` | Modify (add @wrap rules) |
| 2 | `src/grammar/ast.rs` | Modify (parse @wrap) |
| 2 | `src/desugar.rs` | Create (desugaring pass) |
| 2 | `src/lib.rs` | Modify (wire desugar) |
| 2 | `src/interfaces.rs` | Modify (WrapExpr, Endpoint::Wrap) |
| 3 | `neuroscript_runtime/primitives/connections.py` | Create |
| 3 | `src/stdlib_registry.rs` | Modify (register HC) |
| 3 | `src/codegen/generator.rs` | Modify (embedded primitive) |
| 4 | `stdlib/primitives/HC*.ns`, `stdlib/HyperConnect.ns` | Create |
| 4 | `examples/hc_transformer.ns` | Create |

## Dependency Order

```
Task 1 → Task 2 → Task 3 (Phase 1: codegen gap)
Task 4 → Task 5 → Task 6 → Task 7 (Phase 2: @wrap)
Task 8 → Task 9 (Phase 3: runtime, depends on Phase 1)
Task 10 → Task 11 → Task 12 → Task 13 (Phase 4: stdlib, depends on Phases 1-3)
```

Phase 1 and Phase 2 (Tasks 4-7) can proceed in parallel since they're independent until integration testing.
