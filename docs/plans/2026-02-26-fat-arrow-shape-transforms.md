# Fat Arrow (`=>`) Shape Transforms Implementation Plan

**Goal:** Add `=>` as a first-class shape transformation operator in NeuroScript, enabling element-preserving reshapes (permute, split, merge, flatten) and annotated transforms (`@reduce`, `@repeat`) directly in pipelines.

**Architecture:** The fat arrow is implemented as a new `Endpoint::Reshape` variant in the IR. In grammar, `=> [shape]` and `=> @annotation [shape]` are parsed as reshape endpoints within pipelines. The AST builder creates `Reshape` endpoints that flow through the existing `Connection` flattening machinery. Validation enforces element preservation for bare `=>` and annotation requirements for non-preserving transforms. Codegen emits `view`/`permute`/`reshape`/`mean`/`sum`/etc. PyTorch calls.

**Tech Stack:** Rust, pest PEG parser, miette diagnostics, PyTorch codegen

**Design Doc:** `notes/fat_arrow_shape_transforms.md`

---

## Dependency Graph

```
Task 1 (IR Types) ──┐
                     ├── Task 3 (AST Builder) ── Task 5 (Validator) ── Task 6 (Codegen) ── Task 7 (Integration Tests)
Task 2 (Grammar) ───┘                                                                         │
Task 4 (Display) ────────────────────────────────────────────────────────────────────────── Task 7
```

Tasks 1+2 can run in parallel. Task 4 can run after Task 1. Task 3 depends on 1+2. Task 5 depends on 3. Task 6 depends on 5. Task 7 depends on all.

---

### Task 1: Add IR Types for Fat Arrow

**Files:**
- Modify: `src/interfaces.rs`

**Context:** The `Endpoint` enum at line 180 currently has 5 variants: `Ref`, `Tuple`, `Call`, `Match`, `If`. We add a `Reshape` variant. We also add supporting types: `ReshapeExpr`, `ReshapeDim`, `TransformAnnotation`, `TransformStrategy`.

**Step 1: Add new IR types to `src/interfaces.rs`**

Add after the `IfExpr` struct (around line 273):

```rust
/// A reshape expression: [dim_spec, dim_spec, ...]
#[derive(Debug, Clone, PartialEq)]
pub struct ReshapeExpr {
    pub dims: Vec<ReshapeDim>,
    pub annotation: Option<TransformAnnotation>,
    pub id: usize,
}

/// A dimension spec in a reshape expression
#[derive(Debug, Clone, PartialEq)]
pub enum ReshapeDim {
    /// Named dimension reference: b, seq, dim
    Named(String),
    /// Literal value: 1, 5, 512
    Literal(i64),
    /// Decomposition binding: h=dim/heads
    Binding { name: String, expr: Box<Value> },
    /// Others keyword: flattens remaining dims
    Others,
    /// Dimension expression: h*w, dim/heads (uses existing Dim::Expr)
    Expr(Box<DimExpr>),
}

/// Transform annotation: @reduce(mean), @repeat(copy)
#[derive(Debug, Clone, PartialEq)]
pub enum TransformAnnotation {
    Reduce(TransformStrategy),
    Repeat(TransformStrategy),
}

/// Strategy for a transform: intrinsic name or neuron call
#[derive(Debug, Clone, PartialEq)]
pub enum TransformStrategy {
    /// Built-in: mean, sum, min, max, prod, logsumexp, copy
    Intrinsic(String),
    /// Neuron call: AttentionPool(dim)
    Neuron {
        name: String,
        args: Vec<Value>,
        kwargs: Vec<Kwarg>,
    },
}
```

**Step 2: Add `Reshape` variant to `Endpoint` enum**

In `src/interfaces.rs` at line 180, add to the `Endpoint` enum:

```rust
pub enum Endpoint {
    Ref(PortRef),
    Tuple(Vec<PortRef>),
    Call {
        name: String,
        args: Vec<Value>,
        kwargs: Vec<Kwarg>,
        id: usize,
        frozen: bool,
    },
    Match(MatchExpr),
    If(IfExpr),
    /// Shape transformation: => [shape] or => @annotation [shape]
    Reshape(ReshapeExpr),
}
```

**Step 3: Run `cargo check`**

Expected: Compiler errors about non-exhaustive patterns in match statements on `Endpoint`. This is expected — we'll fix these incrementally in later tasks. For now, add `Endpoint::Reshape(_) => todo!("fat arrow reshape")` stubs in all match arms that error.

Files with match on Endpoint that will need stubs:
- `src/codegen/forward.rs` (multiple match arms in `generate_forward_body` and `process_destination`)
- `src/codegen/instantiation.rs` (match on endpoints for anonymous calls)
- `src/codegen/utils.rs` (`endpoint_key_impl` function)
- `src/validator/core.rs` (endpoint resolution)
- `src/validator/symbol_table.rs` (endpoint resolution)
- `src/validator/cycles.rs` (cycle detection)
- `src/shape/inference.rs` (shape inference)
- `src/ir.rs` (Display impl)
- `tests/integration_tests.rs` (format_endpoint)

Add `todo!()` stubs to all of these so the project compiles.

**Step 4: Run `cargo check` to verify compilation**

Expected: PASS (compiles with todo stubs)

**Step 5: Commit**

```bash
git add src/interfaces.rs src/codegen/ src/validator/ src/shape/ src/ir.rs tests/integration_tests.rs
git commit -m "feat: add IR types for fat arrow shape transforms (ReshapeExpr, ReshapeDim, TransformAnnotation)"
```

---

### Task 2: Add Grammar Rules for Fat Arrow

**Files:**
- Modify: `src/grammar/neuroscript.pest`

**Context:** The grammar at line 47 defines `arrow = { "->" }`. Connection rules at lines 311-333 use `arrow` as the pipe separator. We need to add `fat_arrow`, `reshape_expr`, `reshape_dim`, `transform_annotation` rules, and integrate fat arrow steps into the connection and pipeline rules.

**Step 1: Add fat arrow operator and keyword**

After `arrow = { "->" }` (line 47), add:

```pest
fat_arrow = { "=>" }
```

After `keyword_unroll` (line 41), add:

```pest
keyword_others = @{ "others" ~ !ident_cont }
```

Update the `keyword` rule (line 95-103) to include `keyword_others`:

```pest
keyword = _{
    keyword_neuron | keyword_use | keyword_in | keyword_out
  | keyword_impl | keyword_graph | keyword_match | keyword_where
  | keyword_external | keyword_and | keyword_or
  | keyword_true | keyword_false
  | keyword_context | keyword_static | keyword_global | keyword_lazy
  | keyword_if | keyword_elif | keyword_else
  | keyword_unroll | keyword_others
}
```

**Step 2: Add reshape and annotation rules**

Add a new section after the "Shapes and Dimensions" section (after line 136):

```pest
// ============================================================================
// Fat Arrow Reshape
// ============================================================================

// Transform annotation: @reduce(mean), @repeat(copy), @reduce(AttentionPool(dim))
transform_annotation = {
    at ~ ident ~ lparen ~ annotation_arg ~ rparen
}

annotation_arg = {
    call_expr
  | ident
}

// Reshape expression: [dim_spec, dim_spec, ...]
reshape_expr = {
    lbracket ~ (reshape_dim ~ (comma ~ reshape_dim)*)? ~ rbracket
}

// Individual dimension in a reshape
// Order matters: binding must come before plain dim to avoid ident consuming the name
reshape_dim = {
    ident ~ assign ~ dim           // h=dim/heads (decomposition binding)
  | keyword_others                 // others
  | dim                            // named, literal, expr, wildcard, variadic
}

// Fat arrow step: => [shape] or => @reduce(mean) [shape]
fat_arrow_step = {
    fat_arrow ~ transform_annotation? ~ reshape_expr
}
```

**Step 3: Update connection rules to support fat arrow**

Replace the connection and pipeline rules (lines 311-333) with:

```pest
// Connection: source -> dest or pipeline
// Two styles:
// Inline: a -> b -> c => [shape] -> d
// Indented: a ->\n  b\n  => [shape]\n  d  (implicit arrows between lines)
connection = {
    endpoint ~ arrow ~ connection_tail
  | endpoint ~ fat_arrow_step ~ connection_tail_after_reshape
}

// After a fat arrow step, continue with more steps or end
connection_tail_after_reshape = {
    // More inline steps
    ((arrow ~ endpoint) | fat_arrow_step)* ~ NEWLINE*
    // Indented continuation
  | NEWLINE ~ indented_pipeline
}

connection_tail = {
    // Inline: endpoints separated by arrows or fat arrow steps
    (endpoint ~ (((arrow ~ endpoint) | fat_arrow_step))* ~ NEWLINE*)
    // Indented: newline, then indented pipeline
  | (NEWLINE ~ indented_pipeline)
}

// Indented pipeline: can have arrows between lines or not
// Stops when we see an endpoint followed by arrow at start of line (next connection)
indented_pipeline = {
    indented_pipeline_item+
}

indented_pipeline_item = {
    // Fat arrow step on its own line
    fat_arrow_step ~ (arrow ~ NEWLINE*)?
    // Item with trailing arrow means continue to next line
  | (!( (ref_endpoint | tuple_endpoint) ~ arrow) ~ endpoint ~ arrow ~ NEWLINE*)
    // Item without trailing arrow (may be followed by next connection or end)
  | (!( (ref_endpoint | tuple_endpoint) ~ arrow) ~ endpoint ~ NEWLINE*)
}
```

**Step 4: Update match_pipeline and branch_pipeline to support fat arrow**

Replace `match_pipeline` (lines 410-415):

```pest
// Pipeline in match arm (supports both inline and indented, with fat arrows)
match_pipeline = {
    // Inline: endpoints/fat-arrow-steps separated by arrows on one line
    (pipeline_step ~ ((arrow ~ pipeline_step) | fat_arrow_step)* ~ NEWLINE*)
    // Indented: newline, then indented pipeline items
  | (NEWLINE ~ indented_pipeline)
}

pipeline_step = { endpoint }
```

Replace `branch_pipeline` (lines 427-432):

```pest
branch_pipeline = {
    // Indented pipeline
    (NEWLINE ~ indented_pipeline)
    // Inline pipeline with fat arrow support
  | (pipeline_step ~ ((arrow ~ pipeline_step) | fat_arrow_step)*)
}
```

Also update `neuron_match_pipeline` (lines 388-393):

```pest
neuron_match_pipeline = {
    (pipeline_step ~ ((arrow ~ pipeline_step) | fat_arrow_step)* ~ NEWLINE*)
  | (NEWLINE ~ neuron_match_indented_pipeline)
}
```

**Step 5: Run `cargo check`**

Expected: Should compile (grammar changes are only active when used by parser). May get warnings about unused rules. The parser won't actually create Reshape endpoints yet (that's Task 3).

**Step 6: Commit**

```bash
git add src/grammar/neuroscript.pest
git commit -m "feat: add grammar rules for fat arrow (=>) shape transforms"
```

---

### Task 3: AST Builder — Parse Fat Arrow Steps

**Files:**
- Modify: `src/grammar/ast.rs`

**Context:** The AST builder at lines 839-919 handles connection parsing. `build_connection()` parses `endpoint ~ arrow ~ connection_tail`. `build_connection_tail()` flattens inline pipelines. `build_indented_pipeline()` handles multi-line pipelines. All need to handle fat arrow steps that create `Endpoint::Reshape` nodes.

**Step 1: Add `build_reshape_expr` and `build_fat_arrow_step` methods**

Add these new methods to the `AstBuilder` impl:

```rust
/// Build a fat arrow step into a Reshape endpoint
fn build_fat_arrow_step(&mut self, pair: Pair<Rule>) -> Result<Endpoint, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::fat_arrow_step);

    let mut annotation = None;
    let mut reshape = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::fat_arrow => {} // skip the => token
            Rule::transform_annotation => {
                annotation = Some(self.build_transform_annotation(inner)?);
            }
            Rule::reshape_expr => {
                reshape = Some(self.build_reshape_expr(inner)?);
            }
            _ => {}
        }
    }

    let mut reshape_expr = reshape.expect("fat_arrow_step must have reshape_expr");
    reshape_expr.annotation = annotation;
    reshape_expr.id = self.next_id();

    Ok(Endpoint::Reshape(reshape_expr))
}

/// Build a reshape expression: [dim_spec, dim_spec, ...]
fn build_reshape_expr(&mut self, pair: Pair<Rule>) -> Result<ReshapeExpr, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::reshape_expr);

    let mut dims = vec![];

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::reshape_dim => {
                dims.push(self.build_reshape_dim(inner)?);
            }
            _ => {} // skip brackets, commas
        }
    }

    Ok(ReshapeExpr {
        dims,
        annotation: None,
        id: 0, // will be set by caller
    })
}

/// Build a single dimension in a reshape expression
fn build_reshape_dim(&mut self, pair: Pair<Rule>) -> Result<ReshapeDim, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::reshape_dim);

    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();

    match first.as_rule() {
        Rule::keyword_others => Ok(ReshapeDim::Others),
        Rule::ident => {
            let name = first.as_str().to_string();
            // Check if this is a binding: ident = dim
            if let Some(next) = inner.next() {
                if next.as_rule() == Rule::assign {
                    // This is a binding: name = expr
                    let dim_pair = inner.next().unwrap();
                    let dim = self.build_dim(dim_pair)?;
                    // Convert Dim to Value for the binding expr
                    let expr = dim_to_value(dim);
                    Ok(ReshapeDim::Binding {
                        name,
                        expr: Box::new(expr),
                    })
                } else {
                    // Shouldn't happen given grammar
                    Ok(ReshapeDim::Named(name))
                }
            } else {
                Ok(ReshapeDim::Named(name))
            }
        }
        Rule::dim => {
            let dim = self.build_dim(first)?;
            match dim {
                Dim::Named(name) => Ok(ReshapeDim::Named(name)),
                Dim::Literal(n) => Ok(ReshapeDim::Literal(n)),
                Dim::Wildcard => Ok(ReshapeDim::Named("*".to_string())),
                Dim::Variadic(name) => Ok(ReshapeDim::Named(format!("*{}", name))),
                Dim::Expr(expr) => Ok(ReshapeDim::Expr(expr)),
                Dim::Global(name) => Ok(ReshapeDim::Named(format!("@{}", name))),
            }
        }
        _ => Err(error::expected("reshape dimension", first.as_str(), 0)),
    }
}

/// Build a transform annotation: @reduce(mean), @repeat(AttentionPool(dim))
fn build_transform_annotation(
    &mut self,
    pair: Pair<Rule>,
) -> Result<TransformAnnotation, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::transform_annotation);

    let mut inner = pair.into_inner();
    inner.next(); // skip @
    let annotation_name = inner.next().unwrap().as_str().to_string();
    inner.next(); // skip (
    let arg_pair = inner.next().unwrap();

    let strategy = self.build_annotation_arg(arg_pair)?;

    match annotation_name.as_str() {
        "reduce" => Ok(TransformAnnotation::Reduce(strategy)),
        "repeat" => Ok(TransformAnnotation::Repeat(strategy)),
        other => Err(error::expected(
            "reduce or repeat",
            other,
            0,
        )),
    }
}

/// Build an annotation argument: either an intrinsic name or a neuron call
fn build_annotation_arg(&mut self, pair: Pair<Rule>) -> Result<TransformStrategy, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::annotation_arg);

    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::call_expr => {
            let (name, args, kwargs) = self.build_call_expr(inner)?;
            Ok(TransformStrategy::Neuron { name, args, kwargs })
        }
        Rule::ident => Ok(TransformStrategy::Intrinsic(inner.as_str().to_string())),
        _ => Err(error::expected("intrinsic or call", inner.as_str(), 0)),
    }
}
```

Also add a helper function to convert `Dim` to `Value`:

```rust
/// Convert a Dim to a Value (for reshape bindings like h=dim/heads)
fn dim_to_value(dim: Dim) -> Value {
    match dim {
        Dim::Literal(n) => Value::Int(n),
        Dim::Named(name) => Value::Name(name),
        Dim::Expr(expr) => {
            let op = expr.op;
            let left = dim_to_value(expr.left);
            let right = dim_to_value(expr.right);
            Value::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        Dim::Global(name) => Value::Global(name),
        Dim::Wildcard => Value::Name("*".to_string()),
        Dim::Variadic(name) => Value::Name(format!("*{}", name)),
    }
}
```

**Step 2: Update `build_connection` to handle fat arrow start**

The `build_connection` method (line 839) needs to handle connections that start with `endpoint => reshape_expr`:

```rust
fn build_connection(&mut self, pair: Pair<Rule>) -> Result<Vec<Connection>, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::connection);

    let mut inner = pair.into_inner();

    // First endpoint
    let first_endpoint = self.build_endpoint(inner.next().unwrap())?;

    // Second element: either arrow (->) or fat_arrow_step
    let second = inner.next().unwrap();

    match second.as_rule() {
        Rule::arrow => {
            // Standard: endpoint -> connection_tail
            let tail = inner.next().unwrap();
            self.build_connection_tail(first_endpoint, tail)
        }
        Rule::fat_arrow_step => {
            // Fat arrow: endpoint => [shape] connection_tail_after_reshape
            let reshape_endpoint = self.build_fat_arrow_step(second)?;
            let mut connections = vec![Connection {
                source: first_endpoint,
                destination: reshape_endpoint.clone(),
            }];

            // Process remaining tail if present
            if let Some(tail) = inner.next() {
                let tail_conns =
                    self.build_connection_tail_after_reshape(reshape_endpoint, tail)?;
                connections.extend(tail_conns);
            }

            Ok(connections)
        }
        _ => Ok(vec![]),
    }
}
```

**Step 3: Update `build_connection_tail` to handle mixed arrows**

Update the `build_connection_tail` method (line 856) to handle fat arrow steps within inline pipelines:

```rust
fn build_connection_tail(
    &mut self,
    first: Endpoint,
    pair: Pair<Rule>,
) -> Result<Vec<Connection>, ParseError> {
    if pair.as_rule() != Rule::connection_tail {
        return Ok(vec![]);
    }

    let mut connections = vec![];
    let mut prev = first;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::endpoint | Rule::pipeline_step => {
                let next = self.build_endpoint(inner)?;
                connections.push(Connection {
                    source: prev,
                    destination: next.clone(),
                });
                prev = next;
            }
            Rule::fat_arrow_step => {
                let next = self.build_fat_arrow_step(inner)?;
                connections.push(Connection {
                    source: prev,
                    destination: next.clone(),
                });
                prev = next;
            }
            Rule::indented_pipeline => {
                let pipeline_conns = self.build_indented_pipeline(prev, inner)?;
                connections.extend(pipeline_conns);
                return Ok(connections);
            }
            _ => {}
        }
    }

    Ok(connections)
}
```

**Step 4: Add `build_connection_tail_after_reshape`**

```rust
fn build_connection_tail_after_reshape(
    &mut self,
    first: Endpoint,
    pair: Pair<Rule>,
) -> Result<Vec<Connection>, ParseError> {
    // Same logic as build_connection_tail but for connection_tail_after_reshape rule
    let mut connections = vec![];
    let mut prev = first;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::endpoint | Rule::pipeline_step => {
                let next = self.build_endpoint(inner)?;
                connections.push(Connection {
                    source: prev,
                    destination: next.clone(),
                });
                prev = next;
            }
            Rule::fat_arrow_step => {
                let next = self.build_fat_arrow_step(inner)?;
                connections.push(Connection {
                    source: prev,
                    destination: next.clone(),
                });
                prev = next;
            }
            Rule::indented_pipeline => {
                let pipeline_conns = self.build_indented_pipeline(prev, inner)?;
                connections.extend(pipeline_conns);
                return Ok(connections);
            }
            _ => {}
        }
    }

    Ok(connections)
}
```

**Step 5: Update `build_indented_pipeline` to handle fat arrow items**

Update the `build_indented_pipeline` method (line 893):

```rust
fn build_indented_pipeline(
    &mut self,
    first: Endpoint,
    pair: Pair<Rule>,
) -> Result<Vec<Connection>, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::indented_pipeline);

    let mut connections = vec![];
    let mut prev = first;

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::indented_pipeline_item {
            for item_inner in inner.into_inner() {
                match item_inner.as_rule() {
                    Rule::endpoint => {
                        let next = self.build_endpoint(item_inner)?;
                        connections.push(Connection {
                            source: prev,
                            destination: next.clone(),
                        });
                        prev = next;
                    }
                    Rule::fat_arrow_step => {
                        let next = self.build_fat_arrow_step(item_inner)?;
                        connections.push(Connection {
                            source: prev,
                            destination: next.clone(),
                        });
                        prev = next;
                    }
                    _ => {} // skip arrows, newlines
                }
            }
        }
    }

    Ok(connections)
}
```

**Step 6: Update match pipeline builders**

The `build_match_pipeline` and similar functions that build `Vec<Endpoint>` for match arms and if branches need to handle fat_arrow_step. Find every function that processes match_pipeline, branch_pipeline, or neuron_match_pipeline and add fat_arrow_step handling.

In `build_match_arm` (or wherever match arm pipelines are built), when iterating pipeline items:

```rust
Rule::fat_arrow_step => {
    pipeline.push(self.build_fat_arrow_step(inner)?);
}
```

**Step 7: Run `cargo check`**

Expected: PASS

**Step 8: Write a basic parse test**

Add to `src/grammar/tests.rs`:

```rust
#[test]
fn test_parse_fat_arrow_basic() {
    let source = r#"
neuron Reshape(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dim / heads]
  graph:
    in => [batch, seq, heads, dh=dim/heads] => [batch, heads, seq, dh] -> out
"#;
    let program = parse(source).expect("should parse");
    let neuron = program.neurons.get("Reshape").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        assert_eq!(connections.len(), 3); // in=>reshape, reshape=>reshape, reshape->out
    } else {
        panic!("expected graph body");
    }
}

#[test]
fn test_parse_fat_arrow_with_annotation() {
    let source = r#"
neuron Pool:
  in: [b, c, h, w]
  out: [b, c]
  graph:
    in => @reduce(mean) [b, c] -> out
"#;
    let program = parse(source).expect("should parse");
    let neuron = program.neurons.get("Pool").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        assert_eq!(connections.len(), 2); // in=>reduce, reduce->out
    } else {
        panic!("expected graph body");
    }
}

#[test]
fn test_parse_fat_arrow_indented() {
    let source = r#"
neuron VitFlatten:
  in: [b, c, h, w]
  out: [b, seq, c]
  graph:
    in ->
      Linear(512, 256)
      => [b, c, hw=h*w]
      => [b, hw, c]
      out
"#;
    let program = parse(source).expect("should parse");
    let neuron = program.neurons.get("VitFlatten").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        assert!(connections.len() >= 4); // in->Linear, Linear=>reshape, reshape=>reshape, reshape->out
    } else {
        panic!("expected graph body");
    }
}

#[test]
fn test_parse_fat_arrow_others() {
    let source = r#"
neuron Flatten:
  in: [b, c, h, w]
  out: [b, flat]
  graph:
    in => [b, others] -> out
"#;
    let program = parse(source).expect("should parse");
    assert!(program.neurons.contains_key("Flatten"));
}
```

**Step 9: Run `cargo test grammar`**

Expected: PASS

**Step 10: Commit**

```bash
git add src/grammar/ast.rs src/grammar/tests.rs
git commit -m "feat: parse fat arrow (=>) steps in AST builder with tests"
```

---

### Task 4: Display Implementations for New Types

**Files:**
- Modify: `src/ir.rs` (Display impls)
- Modify: `tests/integration_tests.rs` (snapshot formatting)

**Step 1: Add Display for new IR types in `src/ir.rs`**

Add Display impls after the existing ones:

```rust
impl std::fmt::Display for ReshapeDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReshapeDim::Named(name) => write!(f, "{}", name),
            ReshapeDim::Literal(n) => write!(f, "{}", n),
            ReshapeDim::Binding { name, expr } => write!(f, "{}={}", name, expr),
            ReshapeDim::Others => write!(f, "others"),
            ReshapeDim::Expr(expr) => write!(f, "{}", expr),
        }
    }
}

impl std::fmt::Display for ReshapeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref ann) = self.annotation {
            write!(f, "{} ", ann)?;
        }
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl std::fmt::Display for TransformAnnotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformAnnotation::Reduce(s) => write!(f, "@reduce({})", s),
            TransformAnnotation::Repeat(s) => write!(f, "@repeat({})", s),
        }
    }
}

impl std::fmt::Display for TransformStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformStrategy::Intrinsic(name) => write!(f, "{}", name),
            TransformStrategy::Neuron { name, args, kwargs } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                for (i, (k, v)) in kwargs.iter().enumerate() {
                    if !args.is_empty() || i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}={}", k, v)?;
                }
                write!(f, ")")
            }
        }
    }
}
```

**Step 2: Update Endpoint Display in `src/ir.rs`**

In the `Display for Endpoint` impl (line 146), add the Reshape case:

```rust
Endpoint::Reshape(r) => write!(f, "=> {}", r),
```

**Step 3: Update Connection Display in `src/ir.rs`**

Update the `Display for Connection` impl (line 299) to handle reshape destinations:

```rust
impl std::fmt::Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.destination {
            Endpoint::Reshape(_) => write!(f, "{} => {}", self.source, self.destination),
            _ => write!(f, "{} -> {}", self.source, self.destination),
        }
    }
}
```

Wait — the Endpoint::Reshape Display already includes "=> ", so the Connection Display should just use the destination's display minus the "=> " prefix, or we should change how it works. Let me simplify:

Actually, have `Endpoint::Reshape` display as just the reshape expression (no "=> " prefix), and let `Connection::Display` handle the arrow:

```rust
// In Display for Endpoint:
Endpoint::Reshape(r) => write!(f, "{}", r),

// In Display for Connection:
impl std::fmt::Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let arrow = match &self.destination {
            Endpoint::Reshape(_) => "=>",
            _ => "->",
        };
        write!(f, "{} {} {}", self.source, arrow, self.destination)
    }
}
```

**Step 4: Update `format_endpoint` in `tests/integration_tests.rs`**

Add the Reshape case to the `format_endpoint` function:

```rust
Endpoint::Reshape(reshape) => {
    let mut result = String::new();
    if let Some(ref ann) = reshape.annotation {
        result.push_str(&format!("{} ", ann));
    }
    result.push('[');
    let dims: Vec<String> = reshape.dims.iter().map(|d| format!("{}", d)).collect();
    result.push_str(&dims.join(", "));
    result.push(']');
    result
}
```

Update `format_connection` to use `=>` for reshape connections:

```rust
fn format_connection(conn: &Connection) -> String {
    let arrow = match &conn.destination {
        Endpoint::Reshape(_) => "=>",
        _ => "->",
    };
    format!(
        "{} {} {}",
        format_endpoint(&conn.source),
        arrow,
        format_endpoint(&conn.destination)
    )
}
```

**Step 5: Run `cargo test`**

Expected: PASS (existing tests still pass, todo stubs don't fire since no existing .ns files use `=>`)

**Step 6: Commit**

```bash
git add src/ir.rs tests/integration_tests.rs
git commit -m "feat: add Display implementations for fat arrow IR types"
```

---

### Task 5: Validator Support for Fat Arrow

**Files:**
- Modify: `src/validator/core.rs`
- Modify: `src/validator/symbol_table.rs`
- Modify: `src/validator/cycles.rs`
- Modify: `src/validator/shapes.rs`
- Modify: `src/shape/inference.rs`
- Modify: `src/interfaces.rs` (add validation error variants)

**Context:** The validator needs to handle `Endpoint::Reshape` in symbol resolution, cycle detection, and shape checking. For this initial implementation, focus on making fat arrow pass through validation without errors. Full element-preservation checking and annotation validation can be enhanced later.

**Step 1: Add validation error variants**

In `src/interfaces.rs`, add to the `ValidationError` enum:

```rust
InvalidReshape {
    message: String,
    context: String,
},
InvalidAnnotation {
    annotation: String,
    reason: String,
    context: String,
},
```

Add Display arms for these in the `Display for ValidationError` impl.

**Step 2: Update symbol_table.rs**

In `src/validator/symbol_table.rs`, find `resolve_endpoint` and add:

```rust
Endpoint::Reshape(reshape) => {
    // Reshape endpoints don't reference neurons — they're shape transforms
    // Validate dimension names are in scope (basic check)
    // For now, just pass through — full validation in shape inference
    Ok(())
}
```

In `build_symbol_table` or wherever endpoints are collected, add the Reshape case.

**Step 3: Update cycles.rs**

In `src/validator/cycles.rs`, add handling for Reshape endpoints in cycle detection:

```rust
Endpoint::Reshape(_) => {
    // Reshape is a pass-through — no new dependencies introduced
    // Continue with the same node
}
```

**Step 4: Update core.rs**

In `src/validator/core.rs`, update `check_neurons_exist` to handle Reshape:

```rust
Endpoint::Reshape(reshape) => {
    // Check if annotation references a valid neuron (for Neuron strategies)
    if let Some(ref annotation) = reshape.annotation {
        let strategy = match annotation {
            TransformAnnotation::Reduce(s) => s,
            TransformAnnotation::Repeat(s) => s,
        };
        if let TransformStrategy::Neuron { name, .. } = strategy {
            // Verify the neuron exists
            if !program.neurons.contains_key(name)
                && !stdlib_registry.primitives.contains_key(name.as_str())
            {
                errors.push(ValidationError::MissingNeuron {
                    name: name.clone(),
                    context: format!("transform annotation in {}", neuron_name),
                });
            }
        }
        // Validate intrinsic names
        if let TransformStrategy::Intrinsic(name) = strategy {
            let valid_intrinsics = match annotation {
                TransformAnnotation::Reduce(_) => {
                    vec!["mean", "sum", "min", "max", "prod", "logsumexp"]
                }
                TransformAnnotation::Repeat(_) => vec!["copy"],
            };
            if !valid_intrinsics.contains(&name.as_str()) {
                errors.push(ValidationError::InvalidAnnotation {
                    annotation: format!("{}", annotation),
                    reason: format!(
                        "unknown intrinsic '{}', expected one of: {}",
                        name,
                        valid_intrinsics.join(", ")
                    ),
                    context: format!("in {}", neuron_name),
                });
            }
        }
    }
}
```

**Step 5: Update shape inference**

In `src/shape/inference.rs`, update `check_connection` to handle Reshape destinations:

```rust
Endpoint::Reshape(reshape) => {
    // For now, propagate a shape derived from the reshape expression
    // The reshape dims define the output shape
    let output_shape = reshape_dims_to_shape(&reshape.dims);

    // Store the output shape for this reshape endpoint
    ctx.call_outputs.insert(reshape.id, vec![output_shape]);

    Ok(())
}
```

Add helper:

```rust
/// Convert reshape dimensions to an IR Shape
fn reshape_dims_to_shape(dims: &[ReshapeDim]) -> Shape {
    Shape {
        dims: dims.iter().map(|d| match d {
            ReshapeDim::Named(name) => Dim::Named(name.clone()),
            ReshapeDim::Literal(n) => Dim::Literal(*n),
            ReshapeDim::Binding { name, .. } => Dim::Named(name.clone()),
            ReshapeDim::Others => Dim::Wildcard, // placeholder
            ReshapeDim::Expr(expr) => Dim::Expr(expr.clone()),
        }).collect(),
    }
}
```

Also update `resolve_endpoint_shape` to handle Reshape:

```rust
Endpoint::Reshape(reshape) => {
    // Output shape comes from the reshape expression itself
    ctx.call_outputs
        .get(&reshape.id)
        .cloned()
        .unwrap_or_else(|| vec![reshape_dims_to_shape(&reshape.dims)])
}
```

**Step 6: Remove all `todo!()` stubs from Task 1**

Replace all `todo!("fat arrow reshape")` stubs with the real implementations from above.

**Step 7: Run `cargo test`**

Expected: PASS (all existing tests, plus new parse tests from Task 3)

**Step 8: Write validator tests**

Add tests in `src/validator/tests/` for fat arrow validation:

```rust
#[test]
fn test_validate_fat_arrow_basic() {
    let source = r#"
neuron Reshape(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dim / heads]
  graph:
    in => [batch, seq, heads, dh=dim/heads] => [batch, heads, seq, dh] -> out
"#;
    let program = parse(source).unwrap();
    let result = validate(&program);
    assert!(result.is_ok(), "basic fat arrow should validate: {:?}", result);
}

#[test]
fn test_validate_fat_arrow_reduce() {
    let source = r#"
neuron Pool:
  in: [b, c, h, w]
  out: [b, c]
  graph:
    in => @reduce(mean) [b, c] -> out
"#;
    let program = parse(source).unwrap();
    let result = validate(&program);
    assert!(result.is_ok(), "reduce annotation should validate: {:?}", result);
}

#[test]
fn test_validate_fat_arrow_invalid_intrinsic() {
    let source = r#"
neuron Bad:
  in: [b, c]
  out: [b, c]
  graph:
    in => @reduce(foobar) [b] -> out
"#;
    let program = parse(source).unwrap();
    let result = validate(&program);
    assert!(result.is_err(), "invalid intrinsic should fail validation");
}
```

**Step 9: Run `cargo test validator`**

Expected: PASS

**Step 10: Commit**

```bash
git add src/validator/ src/shape/ src/interfaces.rs
git commit -m "feat: validator and shape inference support for fat arrow transforms"
```

---

### Task 6: Codegen for Fat Arrow Transforms

**Files:**
- Modify: `src/codegen/forward.rs`
- Modify: `src/codegen/instantiation.rs`
- Modify: `src/codegen/utils.rs`

**Context:** Codegen needs to handle `Endpoint::Reshape` as both source and destination in connections. As a destination, it generates reshape/permute/reduce/repeat code. As a source, it provides the result variable from the previous reshape operation.

**Step 1: Update `endpoint_key_impl` in `src/codegen/utils.rs`**

```rust
Endpoint::Reshape(r) => format!("reshape_{}", r.id),
```

**Step 2: Update `generate_module_instantiations` in `src/codegen/instantiation.rs`**

Add handling for Reshape endpoints that have Neuron strategies (they need module instantiation):

```rust
// After collecting anonymous calls, also collect neuron strategies from reshapes
for conn in connections {
    if let Endpoint::Reshape(reshape) = &conn.destination {
        if let Some(ref annotation) = reshape.annotation {
            let strategy = match annotation {
                TransformAnnotation::Reduce(s) => s,
                TransformAnnotation::Repeat(s) => s,
            };
            if let TransformStrategy::Neuron { name, args, kwargs } = strategy {
                let module_name = format!("_transform_{}", reshape.id);
                // Generate: self._transform_N = NeuronName(args)
                let args_str = format_call_args(gen, args, kwargs);
                writeln!(output, "        self.{} = {}({})", module_name, name, args_str).unwrap();
                gen.used_primitives.insert(name.clone());
            }
        }
    }
    // Also check source (reshapes can appear as sources in chained connections)
}
```

**Step 3: Update `generate_forward_body` in `src/codegen/forward.rs`**

Add Reshape handling in the source resolution block (around line 180):

```rust
Endpoint::Reshape(reshape) => {
    // Look up result from previous reshape processing
    let key = endpoint_key_impl(&conn.source);
    call_to_result.get(&key).cloned().ok_or_else(|| {
        CodegenError::InvalidConnection(format!(
            "Reshape used as source before being processed"
        ))
    })?
}
```

Add Reshape handling in the destination processing. Create a new function `generate_reshape_code`:

```rust
/// Generate PyTorch code for a reshape operation
fn generate_reshape_code(
    gen: &mut CodeGenerator,
    output: &mut String,
    reshape: &ReshapeExpr,
    source_var: &str,
    indent: &str,
    used_var_names: &mut HashSet<String>,
) -> Result<String, CodegenError> {
    let result_var = make_var_name(used_var_names, "x");

    match &reshape.annotation {
        None => {
            // Bare => : element-preserving view/permute/reshape
            generate_bare_reshape(gen, output, reshape, source_var, &result_var, indent)?;
        }
        Some(TransformAnnotation::Reduce(strategy)) => {
            generate_reduce(gen, output, strategy, reshape, source_var, &result_var, indent)?;
        }
        Some(TransformAnnotation::Repeat(strategy)) => {
            generate_repeat(gen, output, strategy, reshape, source_var, &result_var, indent)?;
        }
    }

    Ok(result_var)
}

/// Generate code for bare reshape (view/reshape)
fn generate_bare_reshape(
    gen: &mut CodeGenerator,
    output: &mut String,
    reshape: &ReshapeExpr,
    source_var: &str,
    result_var: &str,
    indent: &str,
) -> Result<(), CodegenError> {
    // Build the target shape expression
    let shape_args = reshape_dims_to_python(gen, &reshape.dims);

    // For bindings, emit them first
    for dim in &reshape.dims {
        if let ReshapeDim::Binding { name, expr } = dim {
            let expr_str = format_value_for_codegen(gen, expr);
            writeln!(output, "{}{} = {}", indent, name, expr_str).unwrap();
        }
    }

    // Use .view() for simple reshapes, .reshape() when contiguity might be an issue
    // For now, use .reshape() which handles both cases
    writeln!(
        output,
        "{}{} = {}.reshape({})",
        indent, result_var, source_var, shape_args
    )
    .unwrap();

    Ok(())
}

/// Generate code for @reduce(strategy)
fn generate_reduce(
    gen: &mut CodeGenerator,
    output: &mut String,
    strategy: &TransformStrategy,
    reshape: &ReshapeExpr,
    source_var: &str,
    result_var: &str,
    indent: &str,
) -> Result<(), CodegenError> {
    match strategy {
        TransformStrategy::Intrinsic(name) => {
            // For intrinsics, we need to determine which dimensions to reduce
            // For now, emit a general reduction pattern
            // TODO: Determine exact reduction dims from source vs dest shape comparison
            let method = match name.as_str() {
                "mean" => "mean",
                "sum" => "sum",
                "min" => "min",
                "max" => "max",
                "prod" => "prod",
                "logsumexp" => "logsumexp",
                _ => return Err(CodegenError::UnsupportedFeature(
                    format!("unknown reduce intrinsic: {}", name)
                )),
            };

            // For min/max, we need .values
            let suffix = if name == "min" || name == "max" { ".values" } else { "" };

            // Build the target shape for reshaping after reduction
            let shape_args = reshape_dims_to_python(gen, &reshape.dims);

            // Emit: result = source.method(dim=...).values  or  source.method(dim=...)
            // For now, reduce all dims not in target shape
            // Simple approach: reduce then reshape
            writeln!(
                output,
                "{}{} = {}.{}(dim=TODO_REDUCE_DIMS){}",
                indent, result_var, source_var, method, suffix
            )
            .unwrap();

            // If target shape differs from reduction result, reshape
            // writeln!(output, "{}{} = {}.reshape({})", indent, result_var, result_var, shape_args).unwrap();
        }
        TransformStrategy::Neuron { .. } => {
            let module_name = format!("self._transform_{}", reshape.id);
            writeln!(
                output,
                "{}{} = {}({})",
                indent, result_var, module_name, source_var
            )
            .unwrap();
        }
    }

    Ok(())
}

/// Generate code for @repeat(strategy)
fn generate_repeat(
    gen: &mut CodeGenerator,
    output: &mut String,
    strategy: &TransformStrategy,
    reshape: &ReshapeExpr,
    source_var: &str,
    result_var: &str,
    indent: &str,
) -> Result<(), CodegenError> {
    match strategy {
        TransformStrategy::Intrinsic(name) if name == "copy" => {
            let shape_args = reshape_dims_to_python(gen, &reshape.dims);
            writeln!(
                output,
                "{}{} = {}.expand({})",
                indent, result_var, source_var, shape_args
            )
            .unwrap();
        }
        TransformStrategy::Neuron { .. } => {
            let module_name = format!("self._transform_{}", reshape.id);
            writeln!(
                output,
                "{}{} = {}({})",
                indent, result_var, module_name, source_var
            )
            .unwrap();
        }
        _ => {
            return Err(CodegenError::UnsupportedFeature(
                format!("unknown repeat strategy")
            ));
        }
    }

    Ok(())
}

/// Convert reshape dims to Python argument string
fn reshape_dims_to_python(gen: &CodeGenerator, dims: &[ReshapeDim]) -> String {
    dims.iter()
        .map(|d| match d {
            ReshapeDim::Named(name) => name.clone(),
            ReshapeDim::Literal(n) => n.to_string(),
            ReshapeDim::Binding { name, .. } => name.clone(),
            ReshapeDim::Others => "-1".to_string(),
            ReshapeDim::Expr(expr) => format!("{}", expr),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Format a Value for use in codegen expressions
fn format_value_for_codegen(gen: &CodeGenerator, value: &Value) -> String {
    match value {
        Value::Int(n) => n.to_string(),
        Value::Name(name) => {
            gen.binding_context.get(name).cloned().unwrap_or_else(|| name.clone())
        }
        Value::BinOp { op, left, right } => {
            let l = format_value_for_codegen(gen, left);
            let r = format_value_for_codegen(gen, right);
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "//", // integer division in Python
                _ => "?",
            };
            format!("{} {} {}", l, op_str, r)
        }
        Value::Global(name) => name.clone(),
        _ => format!("{}", value),
    }
}
```

**Step 4: Wire up Reshape in `process_destination`**

In the `process_destination` function, add the Reshape arm:

```rust
Endpoint::Reshape(reshape) => {
    let result = generate_reshape_code(
        gen, output, reshape, source_var.clone(), indent, used_var_names,
    )?;
    // Store result for lookup when this reshape is used as a source
    let key = endpoint_key_impl(dest);
    call_to_result.insert(key, result.clone());
    Ok(result)
}
```

(Note: you'll need to pass `call_to_result` through or store it appropriately based on the existing function signature.)

**Step 5: Run `cargo check`**

Expected: PASS

**Step 6: Write codegen test**

Add a test that compiles a fat arrow neuron to PyTorch:

```rust
#[test]
fn test_codegen_fat_arrow_reshape() {
    let source = r#"
neuron MultiHeadReshape(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dim / heads]
  graph:
    in => [batch, seq, heads, dh=dim/heads] -> out
"#;
    let program = parse(source).unwrap();
    validate(&program).unwrap();
    let result = generate_pytorch(&program, "MultiHeadReshape");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    assert!(code.contains(".reshape("), "should contain reshape call");
}
```

**Step 7: Run `cargo test codegen`**

Expected: PASS

**Step 8: Commit**

```bash
git add src/codegen/
git commit -m "feat: codegen support for fat arrow shape transforms"
```

---

### Task 7: Integration Tests and Example Files

**Files:**
- Create: `examples/fat_arrow_basic.ns`
- Create: `examples/fat_arrow_reduce.ns`
- Create: `examples/fat_arrow_repeat.ns`
- Modify: `tests/integration_tests.rs` (add snapshot tests)

**Step 1: Create example files**

`examples/fat_arrow_basic.ns`:
```neuroscript
# Basic fat arrow reshape examples

neuron MultiHeadReshape(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dim / heads]
  graph:
    in => [batch, seq, heads, dh=dim/heads] => [batch, heads, seq, dh] -> out

neuron VitFlatten:
  in: [b, c, h, w]
  out: [b, seq, c]
  graph:
    in => [b, c, hw=h*w] => [b, hw, c] -> out

neuron FlattenTail:
  in: [b, c, h, w]
  out: [b, flat]
  graph:
    in => [b, others] -> out
```

`examples/fat_arrow_reduce.ns`:
```neuroscript
# Fat arrow with @reduce annotation

neuron GlobalAvgPool:
  in: [b, c, h, w]
  out: [b, c]
  graph:
    in => @reduce(mean) [b, c] -> out
```

`examples/fat_arrow_repeat.ns`:
```neuroscript
# Fat arrow with @repeat annotation

neuron ExpandDim:
  in: [b, c, h, w]
  out: [b, c, 1, h, w]
  graph:
    in => [b, c, 1, h, w] -> out
```

**Step 2: Validate all example files**

```bash
cargo build --release
./target/release/neuroscript validate examples/fat_arrow_basic.ns
./target/release/neuroscript validate examples/fat_arrow_reduce.ns
./target/release/neuroscript validate examples/fat_arrow_repeat.ns
```

Expected: All validate successfully

**Step 3: Add snapshot tests**

In `tests/integration_tests.rs`, add snapshot tests for the new example files:

```rust
#[test]
fn snapshot_fat_arrow_basic() {
    let path = example_path("fat_arrow_basic.ns");
    let source = fs::read_to_string(&path).unwrap();
    let program = parse(&source).expect("should parse");
    insta::assert_snapshot!(format_program_ir(&program));
}

#[test]
fn snapshot_fat_arrow_reduce() {
    let path = example_path("fat_arrow_reduce.ns");
    let source = fs::read_to_string(&path).unwrap();
    let program = parse(&source).expect("should parse");
    insta::assert_snapshot!(format_program_ir(&program));
}
```

**Step 4: Run snapshot tests and accept**

```bash
cargo test --test integration_tests
cargo insta accept
```

**Step 5: Run the full test suite**

```bash
cargo test
./test_examples.sh
```

Expected: ALL PASS

**Step 6: Compile a fat arrow example to PyTorch**

```bash
./target/release/neuroscript compile examples/fat_arrow_basic.ns --neuron MultiHeadReshape
```

Verify the output looks reasonable.

**Step 7: Commit**

```bash
git add examples/ tests/
git commit -m "feat: integration tests and examples for fat arrow shape transforms"
```

---

## Execution Notes

**Parallelization opportunities:**
- Tasks 1 and 2 are independent (IR types vs grammar rules) — run in parallel
- Task 4 (Display) can start after Task 1 completes
- Tasks 1+2 must complete before Task 3 (AST builder needs both)
- Task 5 depends on Task 3
- Task 6 depends on Task 5
- Task 7 depends on all

**Key risks:**
- Grammar changes in Task 2 are the trickiest part — pest PEG grammars can be finicky with ambiguous alternatives. If `connection_tail_after_reshape` causes issues, fall back to making the entire connection rule use a generalized `pipe_step` approach
- The reshape dimension binding (`h=dim/heads`) parsing in Task 3 may need iteration if the grammar is ambiguous with the existing `dim` rule
- Codegen for reduce (Task 6) needs to determine which dimensions are being reduced. The initial implementation can use a simple heuristic (compare source/dest dim count) and be refined later

**Testing strategy:**
- Each task has unit/integration tests
- The final task runs the full suite including `test_examples.sh`
- Snapshot tests catch unintended regressions in existing parsing
