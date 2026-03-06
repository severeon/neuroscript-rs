//! Utility functions for code generation
//!
//! This module provides pure utility functions for value conversion,
//! naming transformations, call analysis, and dimension checking.

use super::generator::CodeGenerator;
use crate::interfaces::*;
use std::collections::HashSet;

/// Map a BinOp to its Python operator string.
/// When `int_div` is true, `Div` maps to `//` (integer division for dimension arithmetic).
///
/// Design note: Dimension expressions (reshape targets, shape computations) use integer
/// division `//` because tensor dimensions are always integers. General value expressions
/// (neuron parameter arithmetic) use float division `/` to preserve Python semantics.
/// The caller chooses via `int_div` depending on context.
pub(super) fn binop_to_str(op: &BinOp, int_div: bool) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => if int_div { "//" } else { "/" },
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
    }
}

/// Convert a Value to Python code, resolving names through var_names map.
/// Used by lazy instantiation where context bindings need `self.` prefix.
/// Recurses into BinOp operands and Call args/kwargs to resolve all names.
pub(super) fn value_to_python_with_vars(
    value: &Value,
    var_names: &std::collections::HashMap<String, String>,
) -> String {
    match value {
        Value::Name(n) => {
            if let Some(resolved) = var_names.get(n) {
                resolved.clone()
            } else {
                n.clone()
            }
        }
        Value::BinOp { op, left, right } => {
            format!(
                "{} {} {}",
                value_to_python_with_vars(left, var_names),
                binop_to_str(op, false),
                value_to_python_with_vars(right, var_names)
            )
        }
        Value::Call { name, args, kwargs } => {
            let args_str = args
                .iter()
                .map(|v| value_to_python_with_vars(v, var_names))
                .collect::<Vec<_>>()
                .join(", ");

            let kwargs_str = if kwargs.is_empty() {
                String::new()
            } else {
                let kw: Vec<String> = kwargs
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, value_to_python_with_vars(v, var_names)))
                    .collect();
                if args.is_empty() {
                    kw.join(", ")
                } else {
                    format!(", {}", kw.join(", "))
                }
            };

            format!("{}({}{})", name, args_str, kwargs_str)
        }
        // For other value types (Int, Float, String, Bool, Global),
        // delegate to the standard converter
        _ => value_to_python_impl(value),
    }
}

/// Convert a Value to Python code
pub(super) fn value_to_python_impl(value: &Value) -> String {
    match value {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => format!("\"{}\"", s),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Name(n) => n.clone(),
        Value::Global(n) => n.clone(),
        Value::BinOp { op, left, right } => {
            format!(
                "{} {} {}",
                value_to_python_impl(left),
                binop_to_str(op, false),
                value_to_python_impl(right)
            )
        }
        Value::Call { name, args, kwargs } => {
            let args_str = args
                .iter()
                .map(value_to_python_impl)
                .collect::<Vec<_>>()
                .join(", ");

            let kwargs_str = if kwargs.is_empty() {
                String::new()
            } else {
                let kw: Vec<String> = kwargs
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, value_to_python_impl(v)))
                    .collect();
                if args.is_empty() {
                    kw.join(", ")
                } else {
                    format!(", {}", kw.join(", "))
                }
            };

            format!("{}({}{})", name, args_str, kwargs_str)
        }
    }
}

/// Convert CamelCase to snake_case
pub(super) fn snake_case_impl(name: &str) -> String {
    let mut result = String::new();

    for c in name.chars() {
        if c.is_uppercase() {
            if !result.is_empty() {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }

    result
}

/// Generate a unique key for an endpoint (for call deduplication)
///
/// Each call is uniquely identified by its id, so modules with learnable
/// parameters get separate instances (e.g., 12 transformer layers should
/// each have their own weights).
pub(super) fn endpoint_key_impl(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Call {
            name,
            args,
            kwargs,
            id,
            frozen: _,
        } => {
            let args_str = args
                .iter()
                .map(|v| format!("{:?}", v))
                .collect::<Vec<_>>()
                .join(",");
            let kwargs_str = kwargs
                .iter()
                .map(|(k, v)| format!("{}={:?}", k, v))
                .collect::<Vec<_>>()
                .join(",");
            // Include id to ensure each call gets its own module instance
            format!("{}({};{})@{}", name, args_str, kwargs_str, id)
        }
        Endpoint::If(if_expr) => format!("if@{}", if_expr.id),
        Endpoint::Reshape(r) => format!("reshape@{}", r.id),
        Endpoint::Wrap(w) => format!("wrap@{}", w.id),
        _ => format!("{:?}", endpoint),
    }
}

/// Check if a Value contains references to captured dimensions (not parameters)
pub(super) fn has_captured_dimensions_impl(value: &Value, params: &HashSet<String>) -> bool {
    match value {
        Value::Name(n) => !params.contains(n),
        Value::BinOp { left, right, .. } => {
            has_captured_dimensions_impl(left, params)
                || has_captured_dimensions_impl(right, params)
        }
        Value::Call { args, kwargs, .. } => {
            args.iter().any(|v| has_captured_dimensions_impl(v, params))
                || kwargs
                    .iter()
                    .any(|(_, v)| has_captured_dimensions_impl(v, params))
        }
        Value::Global(_) => false,
        _ => false,
    }
}

/// Collect all Call endpoints recursively from connections
pub(super) fn collect_calls_impl(connections: &[Connection], calls: &mut Vec<Endpoint>) {
    for conn in connections {
        collect_calls_from_endpoint_impl(&conn.source, calls);
        collect_calls_from_endpoint_impl(&conn.destination, calls);
    }
}

/// Collect calls from a single endpoint
fn collect_calls_from_endpoint_impl(endpoint: &Endpoint, calls: &mut Vec<Endpoint>) {
    match endpoint {
        Endpoint::Call { .. } => calls.push(endpoint.clone()),
        Endpoint::Match(match_expr) => {
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    collect_calls_from_endpoint_impl(ep, calls);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    collect_calls_from_endpoint_impl(ep, calls);
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    collect_calls_from_endpoint_impl(ep, calls);
                }
            }
        }
        Endpoint::Tuple(_refs) => {
            // Tuple unpacking doesn't contain calls in current IR
        }
        Endpoint::Ref(_) => {}
        Endpoint::Reshape(_) => {
            // Reshape neuron annotations are instantiated separately by
            // collect_reshape_transforms in instantiation.rs (as self._transform_{id}).
            // Do NOT create synthetic Call endpoints here — that would cause
            // duplicate module instantiation.
        }
        Endpoint::Wrap(_) => {
            // @wrap should be desugared before codegen
        }
        // Endpoint::Unroll removed — expanded before codegen
    }
}

// CodeGenerator wrapper methods for backward compatibility
impl<'a> CodeGenerator<'a> {
    /// Convert a Value to Python code
    pub(super) fn value_to_python(&self, value: &Value) -> String {
        value_to_python_impl(value)
    }

    /// Convert a Value to Python, replacing parameter names with self.param
    pub(super) fn value_to_python_with_self(&self, value: &Value) -> String {
        match value {
            Value::Name(n) => {
                if self.current_neuron_params.contains(n) {
                    format!("self.{}", n)
                } else {
                    n.clone()
                }
            }
            Value::BinOp { op, left, right } => {
                format!(
                    "{} {} {}",
                    self.value_to_python_with_self(left),
                    binop_to_str(op, false),
                    self.value_to_python_with_self(right)
                )
            }
            _ => self.value_to_python(value),
        }
    }

    /// Convert a Value to Python for dimension arithmetic in reshape bindings.
    /// Uses `//` for integer division and resolves names via neuron params
    /// and binding_context.
    pub(super) fn value_to_python_dim_expr(&self, value: &Value) -> String {
        match value {
            Value::Name(n) => {
                if self.current_neuron_params.contains(n) {
                    format!("self.{}", n)
                } else if let Some(resolved) = self.binding_context.get(n) {
                    resolved.clone()
                } else {
                    n.clone()
                }
            }
            Value::BinOp { op, left, right } => {
                format!(
                    "{} {} {}",
                    self.value_to_python_dim_expr(left),
                    binop_to_str(op, true),
                    self.value_to_python_dim_expr(right)
                )
            }
            _ => value_to_python_impl(value),
        }
    }

    /// Generate a unique node ID
    pub(super) fn next_node_id(&mut self) -> usize {
        let id = self.node_counter;
        self.node_counter += 1;
        id
    }

    /// Format a shape for use in a Python assertion
    /// Converts [batch, seq, dim] to a tuple expression like (batch, seq, dim)
    /// Returns None if the shape contains wildcards or other non-concrete dimensions
    pub(super) fn format_shape_for_assertion(&self, shape: &Shape) -> Option<String> {
        let mut dims = Vec::new();

        for dim in &shape.dims {
            let dim_str = match dim {
                Dim::Literal(n) => n.to_string(),
                Dim::Named(name) => {
                    // Check if this is resolved in the inference context
                    if let Some(value) = self.inference_ctx.resolved_dims.get(name) {
                        value.to_string()
                    } else if self.current_neuron_params.contains(name) {
                        // It's a parameter - use self.param
                        format!("self.{}", name)
                    } else {
                        // Not resolved - can't create concrete assertion
                        return None;
                    }
                }
                Dim::Global(name) => name.clone(),
                Dim::Wildcard => return None, // Can't assert on wildcard
                Dim::Inferred => return None, // Can't assert on inferred (-1) dim
                Dim::Variadic(_) => return None, // Can't assert on variadic
                Dim::Expr(expr) => {
                    // Try to evaluate the expression
                    if let Some(value) = self.inference_ctx.evaluate_expr(expr) {
                        value.to_string()
                    } else {
                        // Build expression with parameters
                        format!(
                            "({})",
                            self.value_to_python_with_self(&Value::BinOp {
                                op: expr.op,
                                left: Box::new(dim_to_value(&expr.left)),
                                right: Box::new(dim_to_value(&expr.right)),
                            })
                        )
                    }
                }
            };
            dims.push(dim_str);
        }

        Some(format!("({})", dims.join(", ")))
    }

    /// Format a shape for use in a comment
    /// Converts [batch, seq, dim] to a readable string like [batch, seq, dim]
    pub(super) fn format_shape_for_comment(&self, shape: &Shape) -> String {
        let dims: Vec<String> = shape
            .dims
            .iter()
            .map(|dim| match dim {
                Dim::Literal(n) => n.to_string(),
                Dim::Named(name) => {
                    if let Some(value) = self.inference_ctx.resolved_dims.get(name) {
                        format!("{}={}", name, value)
                    } else {
                        name.clone()
                    }
                }
                Dim::Global(name) => format!("@global {}", name),
                Dim::Wildcard => "*".to_string(),
                Dim::Inferred => "...".to_string(),
                Dim::Variadic(name) => format!("*{}", name),
                Dim::Expr(expr) => {
                    format!("{}", expr.left)
                        + match expr.op {
                            BinOp::Add => " + ",
                            BinOp::Sub => " - ",
                            BinOp::Mul => " * ",
                            BinOp::Div => " / ",
                            _ => " ? ",
                        }
                        + &format!("{}", expr.right)
                }
            })
            .collect();

        format!("[{}]", dims.join(", "))
    }

    /// Check if a shape should have a runtime assertion
    /// Returns true if the shape is concrete enough to assert on
    pub(super) fn should_assert_shape(&self, shape: &Shape) -> bool {
        // Don't assert if shape has wildcards or variadics
        if shape
            .dims
            .iter()
            .any(|d| matches!(d, Dim::Wildcard | Dim::Inferred | Dim::Variadic(_)))
        {
            return false;
        }

        // Don't assert on empty shapes
        if shape.dims.is_empty() {
            return false;
        }

        // Check if all dimensions are either:
        // - Literals
        // - Named dimensions that are resolved or are parameters
        // - Expressions that can be evaluated or converted to Python
        for dim in &shape.dims {
            match dim {
                Dim::Literal(_) => continue,
                Dim::Named(name) => {
                    if !self.inference_ctx.resolved_dims.contains_key(name)
                        && !self.current_neuron_params.contains(name)
                    {
                        return false; // Unresolved dimension
                    }
                }
                Dim::Expr(_) => {
                    // We can try to generate an expression assertion
                    continue;
                }
                _ => return false,
            }
        }

        true
    }
}

/// Convert a Dim to a Value for expression building
fn dim_to_value(dim: &Dim) -> Value {
    match dim {
        Dim::Literal(n) => Value::Int(*n),
        Dim::Named(name) => Value::Name(name.clone()),
        Dim::Expr(expr) => Value::BinOp {
            op: expr.op,
            left: Box::new(dim_to_value(&expr.left)),
            right: Box::new(dim_to_value(&expr.right)),
        },
        Dim::Global(name) => Value::Global(name.clone()),
        _ => Value::Name("None".to_string()), // Shouldn't happen
    }
}

#[cfg(test)]
mod tests;
