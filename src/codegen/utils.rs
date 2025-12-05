//! Utility functions for code generation
//!
//! This module provides pure utility functions for value conversion,
//! naming transformations, call analysis, and dimension checking.

use std::collections::HashSet;
use crate::interfaces::*;

/// Convert a Value to Python code
pub(super) fn value_to_python_impl(value: &Value) -> String {
    match value {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => format!("\"{}\"", s),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Name(n) => n.clone(),
        Value::BinOp { op, left, right } => {
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Lt => "<",
                BinOp::Gt => ">",
                BinOp::Le => "<=",
                BinOp::Ge => ">=",
                BinOp::Eq => "==",
                BinOp::Ne => "!=",
            };
            format!("{} {} {}", value_to_python_impl(left), op_str, value_to_python_impl(right))
        }
        Value::Call { name, args, kwargs } => {
            let args_str = args.iter()
                .map(|v| value_to_python_impl(v))
                .collect::<Vec<_>>()
                .join(", ");

            let kwargs_str = if kwargs.is_empty() {
                String::new()
            } else {
                let kw: Vec<String> = kwargs.iter()
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
        Value::NeuronRef(name) => {
            // A neuron reference is just the class name in Python
            name.clone()
        }
        Value::PartialCall { neuron, args, kwargs } => {
            // For partial application, use functools.partial
            // Example: functools.partial(MyNeuron, 512, d_ff=2048)
            let neuron_str = value_to_python_impl(neuron);

            let args_str = args.iter()
                .map(|v| value_to_python_impl(v))
                .collect::<Vec<_>>()
                .join(", ");

            let kwargs_str = if kwargs.is_empty() {
                String::new()
            } else {
                let kw: Vec<String> = kwargs.iter()
                    .map(|(k, v)| format!("{}={}", k, value_to_python_impl(v)))
                    .collect();
                if args.is_empty() {
                    kw.join(", ")
                } else {
                    format!(", {}", kw.join(", "))
                }
            };

            if args.is_empty() && kwargs.is_empty() {
                // No partial application - just the neuron ref
                neuron_str
            } else {
                // Use functools.partial for partial application
                format!("functools.partial({}, {}{})", neuron_str, args_str, kwargs_str)
            }
        }
    }
}

/// Convert CamelCase to snake_case
pub(super) fn snake_case_impl(name: &str) -> String {
    let mut result = String::new();
    let mut chars = name.chars().peekable();

    while let Some(c) = chars.next() {
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
pub(super) fn endpoint_key_impl(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Call { name, args, kwargs, .. } => {
            let args_str = args.iter()
                .map(|v| format!("{:?}", v))
                .collect::<Vec<_>>()
                .join(",");
            let kwargs_str = kwargs.iter()
                .map(|(k, v)| format!("{}={:?}", k, v))
                .collect::<Vec<_>>()
                .join(",");
            format!("{}({};{})", name, args_str, kwargs_str)
        }
        _ => format!("{:?}", endpoint),
    }
}

/// Check if a Value contains references to captured dimensions (not parameters)
pub(super) fn has_captured_dimensions_impl(value: &Value, params: &HashSet<String>) -> bool {
    match value {
        Value::Name(n) => !params.contains(n),
        Value::BinOp { left, right, .. } => {
            has_captured_dimensions_impl(left, params) || has_captured_dimensions_impl(right, params)
        }
        Value::Call { args, kwargs, .. } => {
            args.iter().any(|v| has_captured_dimensions_impl(v, params)) ||
            kwargs.iter().any(|(_, v)| has_captured_dimensions_impl(v, params))
        }
        Value::NeuronRef(_) => {
            // Neuron references themselves don't contain captured dimensions
            false
        }
        Value::PartialCall { neuron, args, kwargs } => {
            // Check if the neuron or any of its arguments contain captured dimensions
            has_captured_dimensions_impl(neuron, params) ||
            args.iter().any(|v| has_captured_dimensions_impl(v, params)) ||
            kwargs.iter().any(|(_, v)| has_captured_dimensions_impl(v, params))
        }
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
        Endpoint::Tuple(_refs) => {
            // Tuple unpacking doesn't contain calls in current IR
        }
        Endpoint::Ref(_) => {}
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
                let op_str = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                    BinOp::Lt => "<",
                    BinOp::Gt => ">",
                    BinOp::Le => "<=",
                    BinOp::Ge => ">=",
                    BinOp::Eq => "==",
                    BinOp::Ne => "!=",
                };
                format!("{} {} {}", self.value_to_python_with_self(left), op_str, self.value_to_python_with_self(right))
            }
            Value::Call { name, args, kwargs } => {
                // Handle calls with self-prefixed params
                let args_str = args.iter()
                    .map(|v| self.value_to_python_with_self(v))
                    .collect::<Vec<_>>()
                    .join(", ");

                let kwargs_str = if kwargs.is_empty() {
                    String::new()
                } else {
                    let kw: Vec<String> = kwargs.iter()
                        .map(|(k, v)| format!("{}={}", k, self.value_to_python_with_self(v)))
                        .collect();
                    if args.is_empty() {
                        kw.join(", ")
                    } else {
                        format!(", {}", kw.join(", "))
                    }
                };

                format!("{}({}{})", name, args_str, kwargs_str)
            }
            Value::PartialCall { neuron, args, kwargs } => {
                // Handle partial calls with self-prefixed params
                let neuron_str = self.value_to_python_with_self(neuron);

                let args_str = args.iter()
                    .map(|v| self.value_to_python_with_self(v))
                    .collect::<Vec<_>>()
                    .join(", ");

                let kwargs_str = if kwargs.is_empty() {
                    String::new()
                } else {
                    let kw: Vec<String> = kwargs.iter()
                        .map(|(k, v)| format!("{}={}", k, self.value_to_python_with_self(v)))
                        .collect();
                    if args.is_empty() {
                        kw.join(", ")
                    } else {
                        format!(", {}", kw.join(", "))
                    }
                };

                if args.is_empty() && kwargs.is_empty() {
                    neuron_str
                } else {
                    format!("functools.partial({}, {}{})", neuron_str, args_str, kwargs_str)
                }
            }
            _ => self.value_to_python(value)
        }
    }

    /// Convert CamelCase to snake_case
    // pub(super) fn snake_case(&self, name: &str) -> String {
    //     snake_case_impl(name)
    // }

    /// Generate a unique key for an endpoint
    // pub(super) fn endpoint_key(&self, endpoint: &Endpoint) -> String {
    //     endpoint_key_impl(endpoint)
    // }

    /// Check if a Value contains references to captured dimensions
    // pub(super) fn has_captured_dimensions(&self, value: &Value) -> bool {
    //     has_captured_dimensions_impl(value, &self.current_neuron_params)
    // }

    /// Collect all Call endpoints recursively
    // pub(super) fn collect_calls(&self, connections: &[Connection], calls: &mut Vec<Endpoint>) {
    //     collect_calls_impl(connections, calls)
    // }

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
                Dim::Wildcard => return None,  // Can't assert on wildcard
                Dim::Variadic(_) => return None,  // Can't assert on variadic
                Dim::Expr(expr) => {
                    // Try to evaluate the expression
                    if let Some(value) = self.inference_ctx.evaluate_expr(expr) {
                        value.to_string()
                    } else {
                        // Build expression with parameters
                        format!("({})", self.value_to_python_with_self(&Value::BinOp {
                            op: expr.op,
                            left: Box::new(dim_to_value(&expr.left)),
                            right: Box::new(dim_to_value(&expr.right)),
                        }))
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
        let dims: Vec<String> = shape.dims.iter().map(|dim| {
            match dim {
                Dim::Literal(n) => n.to_string(),
                Dim::Named(name) => {
                    if let Some(value) = self.inference_ctx.resolved_dims.get(name) {
                        format!("{}={}", name, value)
                    } else {
                        name.clone()
                    }
                }
                Dim::Wildcard => "*".to_string(),
                Dim::Variadic(name) => format!("*{}", name),
                Dim::Expr(expr) => {
                    format!("{}", expr.left) + match expr.op {
                        BinOp::Add => " + ",
                        BinOp::Sub => " - ",
                        BinOp::Mul => " * ",
                        BinOp::Div => " / ",
                        _ => " ? ",
                    } + &format!("{}", expr.right)
                }
            }
        }).collect();

        format!("[{}]", dims.join(", "))
    }

    /// Check if a shape should have a runtime assertion
    /// Returns true if the shape is concrete enough to assert on
    pub(super) fn should_assert_shape(&self, shape: &Shape) -> bool {
        // Don't assert if shape has wildcards or variadics
        if shape.dims.iter().any(|d| matches!(d, Dim::Wildcard | Dim::Variadic(_))) {
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
                       && !self.current_neuron_params.contains(name) {
                        return false;  // Unresolved dimension
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
        _ => Value::Name("None".to_string()),  // Shouldn't happen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_to_python_primitives() {
        assert_eq!(value_to_python_impl(&Value::Int(42)), "42");
        assert_eq!(value_to_python_impl(&Value::Float(3.14)), "3.14");
        assert_eq!(value_to_python_impl(&Value::String("hello".to_string())), "\"hello\"");
        assert_eq!(value_to_python_impl(&Value::Bool(true)), "True");
        assert_eq!(value_to_python_impl(&Value::Bool(false)), "False");
        assert_eq!(value_to_python_impl(&Value::Name("dim".to_string())), "dim");
    }

    #[test]
    fn test_value_to_python_binop() {
        let binop = Value::BinOp {
            op: BinOp::Mul,
            left: Box::new(Value::Name("dim".to_string())),
            right: Box::new(Value::Int(4)),
        };
        assert_eq!(value_to_python_impl(&binop), "dim * 4");
    }

    #[test]
    fn test_snake_case() {
        assert_eq!(snake_case_impl("Linear"), "linear");
        assert_eq!(snake_case_impl("GELU"), "g_e_l_u");
        assert_eq!(snake_case_impl("LayerNorm"), "layer_norm");
        assert_eq!(snake_case_impl("MultiHeadAttention"), "multi_head_attention");
    }

    #[test]
    fn test_has_captured_dimensions() {
        let mut params = HashSet::new();
        params.insert("dim".to_string());

        // Parameter reference - not captured
        assert!(!has_captured_dimensions_impl(&Value::Name("dim".to_string()), &params));

        // Non-parameter reference - captured
        assert!(has_captured_dimensions_impl(&Value::Name("d".to_string()), &params));

        // BinOp with captured
        let binop = Value::BinOp {
            op: BinOp::Mul,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(4)),
        };
        assert!(has_captured_dimensions_impl(&binop, &params));
    }

    #[test]
    fn test_endpoint_key_deduplication() {
        let call1 = Endpoint::Call {
            name: "Linear".to_string(),
            args: vec![Value::Int(512), Value::Int(256)],
            kwargs: vec![],
            id: 0,
        };

        let call2 = Endpoint::Call {
            name: "Linear".to_string(),
            args: vec![Value::Int(512), Value::Int(256)],
            kwargs: vec![],
            id: 1,
        };

        // Same signature should have same key (id doesn't matter)
        assert_eq!(endpoint_key_impl(&call1), endpoint_key_impl(&call2));
    }

    #[test]
    fn test_value_to_python_neuron_ref() {
        // Simple neuron reference
        let neuron_ref = Value::NeuronRef("TransformerBlock".to_string());
        assert_eq!(value_to_python_impl(&neuron_ref), "TransformerBlock");
    }

    #[test]
    fn test_value_to_python_partial_call() {
        // Partial call with args
        let partial = Value::PartialCall {
            neuron: Box::new(Value::NeuronRef("MyNeuron".to_string())),
            args: vec![Value::Int(512)],
            kwargs: vec![],
        };
        assert_eq!(value_to_python_impl(&partial), "functools.partial(MyNeuron, 512)");

        // Partial call with kwargs
        let partial_kwargs = Value::PartialCall {
            neuron: Box::new(Value::NeuronRef("MyNeuron".to_string())),
            args: vec![],
            kwargs: vec![("d_ff".to_string(), Value::Int(2048))],
        };
        assert_eq!(value_to_python_impl(&partial_kwargs), "functools.partial(MyNeuron, d_ff=2048)");

        // Partial call with both
        let partial_both = Value::PartialCall {
            neuron: Box::new(Value::NeuronRef("MyNeuron".to_string())),
            args: vec![Value::Int(512)],
            kwargs: vec![("d_ff".to_string(), Value::Int(2048))],
        };
        assert_eq!(value_to_python_impl(&partial_both), "functools.partial(MyNeuron, 512, d_ff=2048)");

        // Empty partial call (should just return neuron ref)
        let empty_partial = Value::PartialCall {
            neuron: Box::new(Value::NeuronRef("MyNeuron".to_string())),
            args: vec![],
            kwargs: vec![],
        };
        assert_eq!(value_to_python_impl(&empty_partial), "MyNeuron");
    }

    #[test]
    fn test_has_captured_dimensions_with_new_values() {
        let mut params = HashSet::new();
        params.insert("d_model".to_string());

        // NeuronRef has no captured dimensions
        let neuron_ref = Value::NeuronRef("MyNeuron".to_string());
        assert!(!has_captured_dimensions_impl(&neuron_ref, &params));

        // PartialCall with parameter - not captured
        let partial_param = Value::PartialCall {
            neuron: Box::new(Value::NeuronRef("MyNeuron".to_string())),
            args: vec![Value::Name("d_model".to_string())],
            kwargs: vec![],
        };
        assert!(!has_captured_dimensions_impl(&partial_param, &params));

        // PartialCall with captured dimension
        let partial_captured = Value::PartialCall {
            neuron: Box::new(Value::NeuronRef("MyNeuron".to_string())),
            args: vec![Value::Name("d".to_string())],
            kwargs: vec![],
        };
        assert!(has_captured_dimensions_impl(&partial_captured, &params));
    }
}
