//! Endpoint visitor utilities for traversing nested IR structures.
//!
//! Provides two complementary APIs:
//!
//! 1. **`EndpointVisitor` trait** — A trait-based visitor pattern for walking `Endpoint` trees.
//!    Implementors override specific `visit_*` methods; default implementations handle recursive
//!    descent into nested structures (match arms, if branches, wrap pipelines).
//!
//! 2. **`walk_endpoints` / `walk_endpoints_mut`** — Closure-based walkers for simpler one-off
//!    traversals that don't need the full visitor pattern.

use crate::interfaces::*;

// ---------------------------------------------------------------------------
// Trait-based EndpointVisitor
// ---------------------------------------------------------------------------

/// Trait for walking `Endpoint` trees with a visitor pattern.
///
/// Provides default recursive walking — implementors override specific visit
/// methods to collect data or perform analysis.  The default `visit_endpoint`
/// dispatches to variant-specific methods, and compound variants (`Tuple`,
/// `Match`, `If`, `Wrap`) recurse into their children automatically.
///
/// # Example
///
/// ```ignore
/// use neuroscript::visitor::{EndpointVisitor, walk_connections};
///
/// struct CallCounter(usize);
///
/// impl EndpointVisitor for CallCounter {
///     fn visit_call(&mut self, _name: &str, _args: &[Value], _kwargs: &[Kwarg], _ep: &Endpoint) {
///         self.0 += 1;
///     }
/// }
///
/// let mut counter = CallCounter(0);
/// walk_connections(&mut counter, &connections);
/// println!("Found {} calls", counter.0);
/// ```
pub trait EndpointVisitor {
    /// Called for each `Endpoint` node. Default implementation dispatches to
    /// variant-specific methods.
    fn visit_endpoint(&mut self, endpoint: &Endpoint) {
        match endpoint {
            Endpoint::Ref(port_ref) => self.visit_ref(port_ref),
            Endpoint::Call {
                name,
                args,
                kwargs,
                ..
            } => self.visit_call(name, args, kwargs, endpoint),
            Endpoint::Tuple(port_refs) => self.visit_tuple(port_refs),
            Endpoint::Match(match_expr) => self.visit_match(match_expr),
            Endpoint::Reshape(reshape) => self.visit_reshape(reshape),
            Endpoint::If(if_expr) => self.visit_if(if_expr),
            Endpoint::Wrap(wrap_expr) => self.visit_wrap(wrap_expr),
        }
    }

    /// Visit a `Ref` endpoint (e.g., `in`, `out`, `my_binding`).
    fn visit_ref(&mut self, _port_ref: &PortRef) {}

    /// Visit a `Call` endpoint (e.g., `Linear(512, 256)`).
    /// The full `endpoint` is passed for callers that need to clone it.
    fn visit_call(
        &mut self,
        _name: &str,
        _args: &[Value],
        _kwargs: &[Kwarg],
        _endpoint: &Endpoint,
    ) {
    }

    /// Visit a `Tuple` endpoint (e.g., `(a, b)`).
    /// Default: no-op (tuples contain `PortRef`, not nested `Endpoint`s).
    fn visit_tuple(&mut self, _port_refs: &[PortRef]) {}

    /// Visit a `Match` endpoint. Default: recurse into each arm's pipeline.
    fn visit_match(&mut self, match_expr: &MatchExpr) {
        for arm in &match_expr.arms {
            for ep in &arm.pipeline {
                self.visit_endpoint(ep);
            }
        }
    }

    /// Visit a `Reshape` endpoint (e.g., `=> [batch, seq, heads, dim]`).
    fn visit_reshape(&mut self, _reshape: &ReshapeExpr) {}

    /// Visit an `If` endpoint. Default: recurse into branches and else_branch.
    fn visit_if(&mut self, if_expr: &IfExpr) {
        for branch in &if_expr.branches {
            for ep in &branch.pipeline {
                self.visit_endpoint(ep);
            }
        }
        if let Some(else_branch) = &if_expr.else_branch {
            for ep in else_branch {
                self.visit_endpoint(ep);
            }
        }
    }

    /// Visit a `Wrap` endpoint (e.g., `@wrap(Wrapper, args): content`).
    /// Default: recurse into pipeline content if present.
    fn visit_wrap(&mut self, wrap_expr: &WrapExpr) {
        if let WrapContent::Pipeline(pipeline) = &wrap_expr.content {
            for ep in pipeline {
                self.visit_endpoint(ep);
            }
        }
    }
}

/// Walk all endpoints in a slice of connections using an `EndpointVisitor`.
pub fn walk_connections(visitor: &mut impl EndpointVisitor, connections: &[Connection]) {
    for conn in connections {
        visitor.visit_endpoint(&conn.source);
        visitor.visit_endpoint(&conn.destination);
    }
}

// ---------------------------------------------------------------------------
// Built-in visitor implementations
// ---------------------------------------------------------------------------

/// Collects all `Call` endpoints (cloned) from an endpoint tree.
///
/// Used by codegen to enumerate module instantiations.
pub struct CallCollector {
    pub calls: Vec<Endpoint>,
}

impl CallCollector {
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }

    /// Collect all `Call` endpoints from a slice of connections.
    pub fn from_connections(connections: &[Connection]) -> Vec<Endpoint> {
        let mut collector = Self::new();
        walk_connections(&mut collector, connections);
        collector.calls
    }
}

impl EndpointVisitor for CallCollector {
    fn visit_call(
        &mut self,
        _name: &str,
        _args: &[Value],
        _kwargs: &[Kwarg],
        endpoint: &Endpoint,
    ) {
        self.calls.push(endpoint.clone());
    }

    // Reshape annotations may contain neuron calls, but those are instantiated
    // separately by collect_reshape_transforms in instantiation.rs. We must NOT
    // include them here to avoid duplicate module instantiation.
    fn visit_reshape(&mut self, _reshape: &ReshapeExpr) {}

    // @wrap should be desugared before codegen — skip.
    fn visit_wrap(&mut self, _wrap_expr: &WrapExpr) {}
}

/// Collects unique neuron names from `Call` endpoints (and reshape annotations).
///
/// Used by codegen to determine which imports are needed.
pub struct NeuronNameCollector {
    pub names: std::collections::HashSet<String>,
}

impl NeuronNameCollector {
    pub fn new() -> Self {
        Self {
            names: std::collections::HashSet::new(),
        }
    }

    /// Collect all neuron names from a slice of connections.
    pub fn from_connections(connections: &[Connection]) -> std::collections::HashSet<String> {
        let mut collector = Self::new();
        walk_connections(&mut collector, connections);
        collector.names
    }
}

impl EndpointVisitor for NeuronNameCollector {
    fn visit_call(
        &mut self,
        name: &str,
        _args: &[Value],
        _kwargs: &[Kwarg],
        _endpoint: &Endpoint,
    ) {
        self.names.insert(name.to_string());
    }

    fn visit_reshape(&mut self, reshape: &ReshapeExpr) {
        if let Some(ref annotation) = reshape.annotation {
            let strategy = match annotation {
                TransformAnnotation::Reduce { strategy, .. } => strategy,
                TransformAnnotation::Repeat { strategy, .. } => strategy,
            };
            if let TransformStrategy::Neuron { name, .. } = strategy {
                self.names.insert(name.clone());
            }
        }
    }

    // @wrap should be desugared before codegen — skip.
    fn visit_wrap(&mut self, _wrap_expr: &WrapExpr) {}
}

/// Collects node names for simple cycle detection.
///
/// Calls are identified by name + args to distinguish different instances.
pub struct SimpleNodeNameCollector {
    pub names: Vec<String>,
}

impl SimpleNodeNameCollector {
    pub fn new() -> Self {
        Self { names: Vec::new() }
    }

    /// Collect node names from a single endpoint.
    pub fn from_endpoint(endpoint: &Endpoint) -> Vec<String> {
        let mut collector = Self::new();
        collector.visit_endpoint(endpoint);
        collector.names
    }
}

impl EndpointVisitor for SimpleNodeNameCollector {
    fn visit_ref(&mut self, port_ref: &PortRef) {
        self.names.push(port_ref.node.clone());
    }

    fn visit_call(
        &mut self,
        name: &str,
        args: &[Value],
        _kwargs: &[Kwarg],
        _endpoint: &Endpoint,
    ) {
        let args_str = args
            .iter()
            .map(|v| format!("{:?}", v))
            .collect::<Vec<_>>()
            .join(",");
        self.names.push(format!("{}({})", name, args_str));
    }

    fn visit_tuple(&mut self, port_refs: &[PortRef]) {
        for r in port_refs {
            self.names.push(r.node.clone());
        }
    }

    // Match, If, Reshape, Wrap all return empty for cycle detection — override
    // defaults to prevent recursion into children.
    fn visit_match(&mut self, _match_expr: &MatchExpr) {}
    fn visit_if(&mut self, _if_expr: &IfExpr) {}
    fn visit_reshape(&mut self, _reshape: &ReshapeExpr) {}
    fn visit_wrap(&mut self, _wrap_expr: &WrapExpr) {}
}

// ---------------------------------------------------------------------------
// Closure-based walkers (pre-existing API, retained for simpler use cases)
// ---------------------------------------------------------------------------

/// Walk all endpoints in a program immutably, calling `f` for each one.
///
/// The callback receives `(endpoint, neuron_name)` for context.
/// This walks into match arms, if branches, and other nested structures.
pub fn walk_endpoints(program: &Program, f: &mut impl FnMut(&Endpoint, &str)) {
    for (name, neuron) in &program.neurons {
        if let NeuronBody::Graph { connections, .. } = &neuron.body {
            for conn in connections {
                walk_endpoint_recursive(&conn.source, name, f);
                walk_endpoint_recursive(&conn.destination, name, f);
            }
        }
    }
}

fn walk_endpoint_recursive(
    endpoint: &Endpoint,
    neuron_name: &str,
    f: &mut impl FnMut(&Endpoint, &str),
) {
    f(endpoint, neuron_name);
    match endpoint {
        Endpoint::Match(match_expr) => {
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    walk_endpoint_recursive(ep, neuron_name, f);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    walk_endpoint_recursive(ep, neuron_name, f);
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    walk_endpoint_recursive(ep, neuron_name, f);
                }
            }
        }
        Endpoint::Wrap(wrap_expr) => {
            if let WrapContent::Pipeline(pipeline) = &wrap_expr.content {
                for ep in pipeline {
                    walk_endpoint_recursive(ep, neuron_name, f);
                }
            }
        }
        _ => {}
    }
}

/// Walk all endpoints in a program mutably, calling `f` for each one.
///
/// The callback receives `(endpoint, neuron_name)` for context.
/// This walks into match arms, if branches, and other nested structures.
pub fn walk_endpoints_mut(program: &mut Program, f: &mut impl FnMut(&mut Endpoint, &str)) {
    let neuron_names: Vec<String> = program.neurons.keys().cloned().collect();
    for name in &neuron_names {
        if let Some(neuron) = program.neurons.get_mut(name) {
            if let NeuronBody::Graph { connections, .. } = &mut neuron.body {
                for conn in connections {
                    walk_endpoint_recursive_mut(&mut conn.source, name, f);
                    walk_endpoint_recursive_mut(&mut conn.destination, name, f);
                }
            }
        }
    }
}

fn walk_endpoint_recursive_mut(
    endpoint: &mut Endpoint,
    neuron_name: &str,
    f: &mut impl FnMut(&mut Endpoint, &str),
) {
    f(endpoint, neuron_name);
    match endpoint {
        Endpoint::Match(match_expr) => {
            for arm in &mut match_expr.arms {
                for ep in &mut arm.pipeline {
                    walk_endpoint_recursive_mut(ep, neuron_name, f);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &mut if_expr.branches {
                for ep in &mut branch.pipeline {
                    walk_endpoint_recursive_mut(ep, neuron_name, f);
                }
            }
            if let Some(else_branch) = &mut if_expr.else_branch {
                for ep in else_branch {
                    walk_endpoint_recursive_mut(ep, neuron_name, f);
                }
            }
        }
        Endpoint::Wrap(wrap_expr) => {
            if let WrapContent::Pipeline(pipeline) = &mut wrap_expr.content {
                for ep in pipeline {
                    walk_endpoint_recursive_mut(ep, neuron_name, f);
                }
            }
        }
        _ => {}
    }
}
