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
            _ => self.value_to_python(value)
        }
    }

    /// Convert CamelCase to snake_case
    pub(super) fn snake_case(&self, name: &str) -> String {
        snake_case_impl(name)
    }

    /// Generate a unique key for an endpoint
    pub(super) fn endpoint_key(&self, endpoint: &Endpoint) -> String {
        endpoint_key_impl(endpoint)
    }

    /// Check if a Value contains references to captured dimensions
    pub(super) fn has_captured_dimensions(&self, value: &Value) -> bool {
        has_captured_dimensions_impl(value, &self.current_neuron_params)
    }

    /// Collect all Call endpoints recursively
    pub(super) fn collect_calls(&self, connections: &[Connection], calls: &mut Vec<Endpoint>) {
        collect_calls_impl(connections, calls)
    }

    /// Generate a unique node ID
    pub(super) fn next_node_id(&mut self) -> usize {
        let id = self.node_counter;
        self.node_counter += 1;
        id
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
}
