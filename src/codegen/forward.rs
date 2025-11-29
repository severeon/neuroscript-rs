//! Forward method generation
//!
//! This module generates the forward() method body from the connection graph.
//! It handles all endpoint types including calls, match expressions, tuple unpacking,
//! and references.

use std::collections::HashMap;
use std::fmt::Write;
use crate::interfaces::*;
use super::utils::*;

/// Generate forward method body from connections
pub(super) fn generate_forward_body(
    gen: &mut CodeGenerator,
    output: &mut String,
    connections: &[Connection],
    inputs: &[&str]
) -> Result<(), CodegenError> {
    // Clear var names for this forward pass
    gen.var_names.clear();

    // Map input ports to initial variables
    if inputs.len() == 1 && inputs[0] == "default" {
        gen.var_names.insert("in".to_string(), "x".to_string());
    } else {
        for input in inputs {
            gen.var_names.insert((*input).to_string(), (*input).to_string());
        }
    }

    let mut temp_var_counter = 0;
    let indent = "        ";

    // Build a map from Call endpoints to their result variable names
    let mut call_to_result: HashMap<String, String> = HashMap::new();

    // Track the last result variable (for implicit output)
    let mut last_result = None;

    // Process each connection
    for conn in connections {
        // Resolve the source to a variable name
        let source_var = match &conn.source {
            Endpoint::Ref(port_ref) => {
                gen.var_names.get(&port_ref.node)
                    .cloned()
                    .unwrap_or_else(|| port_ref.node.clone())
            }
            Endpoint::Tuple(refs) => {
                let vars: Vec<String> = refs.iter()
                    .map(|r| gen.var_names.get(&r.node).cloned().unwrap_or_else(|| r.node.clone()))
                    .collect();
                format!("({})", vars.join(", "))
            }
            Endpoint::Call { name, .. } => {
                // This Call should have been processed in a previous connection
                // Look it up in our call_to_result map
                let key = endpoint_key_impl(&conn.source);
                call_to_result.get(&key)
                    .cloned()
                    .ok_or_else(|| CodegenError::InvalidConnection(
                        format!("Call to {} used as source before being defined", name)
                    ))?
            }
            Endpoint::Match(_) => {
                return Err(CodegenError::UnsupportedFeature("Match expressions as source".to_string()));
            }
        };

        // Process the destination
        let result_var = process_destination(output, gen, &conn.destination, source_var, indent, &mut temp_var_counter, &mut call_to_result)?;

        // Track the last result for implicit output
        last_result = Some(result_var.clone());

        // If destination was a Call, store result in call_to_result
        if let Endpoint::Call { .. } = &conn.destination {
             let key = endpoint_key_impl(&conn.destination);
             call_to_result.insert(key, result_var);
        }
    }

    // Return the output variable
    // Priority: explicit "out" port > last result > last temp variable
    let output_var = gen.var_names.get("out")
        .cloned()
        .or(last_result)
        .unwrap_or_else(|| format!("x{}", temp_var_counter - 1));
    writeln!(output, "        return {}", output_var).unwrap();

    Ok(())
}

/// Process a destination endpoint, generating code and returning the result variable name
fn process_destination(
    output: &mut String,
    gen: &mut CodeGenerator,
    endpoint: &Endpoint,
    source_var: String,
    indent: &str,
    temp_var_counter: &mut usize,
    call_to_result: &mut HashMap<String, String>
) -> Result<String, CodegenError> {
    match endpoint {
        Endpoint::Ref(port_ref) => {
            // Simple assignment - the source becomes this port's variable
            gen.var_names.insert(port_ref.node.clone(), source_var.clone());
            Ok(source_var)
        }
        Endpoint::Tuple(refs) => {
            // Tuple unpacking
            let var_names: Vec<String> = refs.iter().map(|r| {
                let v = format!("x{}", *temp_var_counter);
                *temp_var_counter += 1;
                gen.var_names.insert(r.node.clone(), v.clone());
                v
            }).collect();

            writeln!(output, "{}{} = {}", indent, var_names.join(", "), source_var).unwrap();
            Ok(source_var) // Return tuple as result
        }
        Endpoint::Call { name, args, kwargs, .. } => {
            // Generate a call to the module
            let key = endpoint_key_impl(endpoint);
            let module_name = gen.call_to_module.get(&key)
                .cloned()
                .ok_or_else(|| CodegenError::InvalidConnection(
                    format!("Module for call to {} not found", name)
                ))?;

            let result_var = format!("x{}", *temp_var_counter);
            *temp_var_counter += 1;

            // Check if this call has captured dimensions (needs lazy instantiation)
            let has_captured = args.iter().any(|v| has_captured_dimensions_impl(v, &gen.current_neuron_params)) ||
                               kwargs.iter().any(|(_, v)| has_captured_dimensions_impl(v, &gen.current_neuron_params));

            if has_captured {
                // Lazy instantiation: check cache, instantiate if needed
                writeln!(output, "{}if self._{} is None:", indent, module_name).unwrap();

                // Generate instantiation with current dimension values
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

                // Mark as primitive for imports
                if let Some(neuron) = gen.program.neurons.get(name.as_str()) {
                    if neuron.is_primitive() {
                        gen.used_primitives.insert(name.clone());
                    }
                } else {
                    gen.used_primitives.insert(name.clone());
                }

                writeln!(output, "{}    self._{} = {}({}{})", indent, module_name, name, args_str, kwargs_str).unwrap();

                // Call the lazily-instantiated module
                writeln!(output, "{}{} = self._{}({})", indent, result_var, module_name, source_var).unwrap();
            } else {
                // Normal call to pre-instantiated module
                writeln!(output, "{}{} = self.{}({})", indent, result_var, module_name, source_var).unwrap();
            }

            Ok(result_var)
        }
        Endpoint::Match(match_expr) => {
            let result_var = format!("x{}", *temp_var_counter);
            *temp_var_counter += 1;

            // Initialize result_var to None for safety (though not strictly needed if all paths return)
            writeln!(output, "{}{} = None", indent, result_var).unwrap();

            let mut first = true;
            let mut prev_condition = String::new();
            for arm in &match_expr.arms {
                let shape_check = generate_shape_check(gen, &arm.pattern, arm.guard.as_ref(), &source_var);

                // Determine prefix: use "else:" if pattern condition is same as previous
                let prefix = if first {
                    "if"
                } else if shape_check.condition == prev_condition {
                    // Same pattern, different guard (or no guard) -> use else
                    "else"
                } else {
                    "elif"
                };
                first = false;

                // Only output condition if it's not "else"
                if prefix == "else" {
                    writeln!(output, "{}{}:", indent, prefix).unwrap();
                } else {
                    writeln!(output, "{}{} {}:", indent, prefix, shape_check.condition).unwrap();
                    prev_condition = shape_check.condition.clone();
                }

                // Process pipeline - save var_names to avoid pollution from match arm scope
                let saved_var_names = gen.var_names.clone();
                let arm_indent = format!("{}    ", indent);

                // Emit dimension bindings before processing pipeline
                for binding in &shape_check.bindings {
                    writeln!(output, "{}{}", arm_indent, binding).unwrap();
                }

                // If guard uses captured dimensions, check it after binding
                let pipeline_indent = if let Some(guard_cond) = &shape_check.guard_condition {
                    writeln!(output, "{}if {}:", arm_indent, guard_cond).unwrap();
                    format!("{}    ", arm_indent)
                } else {
                    arm_indent.clone()
                };

                let mut current_var = source_var.clone();

                for ep in &arm.pipeline {
                     current_var = process_destination(output, gen, ep, current_var, &pipeline_indent, temp_var_counter, call_to_result)?;

                     // If endpoint was a Call, store result in call_to_result
                    if let Endpoint::Call { .. } = ep {
                         let key = endpoint_key_impl(ep);
                         call_to_result.insert(key, current_var.clone());
                    }
                }

                writeln!(output, "{}{} = {}", pipeline_indent, result_var, current_var).unwrap();

                // Restore var_names to prevent match arm scope from leaking
                gen.var_names = saved_var_names;
            }

            // Else clause
            writeln!(output, "{}else:", indent).unwrap();
            writeln!(output, "{}    raise ValueError(f'No match found for shape {{ {}.shape }}')", indent, source_var).unwrap();

            Ok(result_var)
        }
    }
}

/// Generate a runtime shape check condition and dimension bindings
///
/// Returns a ShapeCheckResult containing:
/// - condition: Boolean expression for runtime check (shape checks only, no guard)
/// - bindings: Dimension variable assignments (e.g., "d = x.shape[1]")
/// - guard_condition: Separate guard check if it references captured dimensions
///
/// Named dimensions in patterns are handled as follows:
/// - If the name is a neuron parameter: Check equality (no binding needed)
/// - Otherwise: Capture the dimension value for use in the pipeline
pub(super) fn generate_shape_check(
    gen: &CodeGenerator,
    pattern: &Shape,
    guard: Option<&Value>,
    var_name: &str
) -> ShapeCheckResult {
    let mut checks = Vec::new();
    let mut bindings = Vec::new();

    // Rank check (unless variadic)
    let has_variadic = pattern.dims.iter().any(|d| matches!(d, Dim::Variadic(_)));
    if !has_variadic {
        checks.push(format!("{}.ndim == {}", var_name, pattern.dims.len()));
    }

    for (i, dim) in pattern.dims.iter().enumerate() {
        match dim {
            Dim::Literal(n) => {
                checks.push(format!("{}.shape[{}] == {}", var_name, i, n));
            }
            Dim::Named(n) => {
                // Check if it's a parameter
                if gen.current_neuron_params.contains(n) {
                    // Parameter: check equality with self.param
                    checks.push(format!("{}.shape[{}] == self.{}", var_name, i, n));
                } else {
                    // Pattern capture: bind dimension for use in pipeline
                    bindings.push(format!("{} = {}.shape[{}]", n, var_name, i));
                }
            }
            _ => {} // Skip Wildcard, Variadic, Expr for now
        }
    }

    // Handle guard: if it references captured dimensions, defer to after binding
    let guard_condition = if let Some(guard_expr) = guard {
        if !bindings.is_empty() && has_captured_dimensions_impl(guard_expr, &gen.current_neuron_params) {
            // Guard uses captured dims - check it separately after binding
            Some(gen.value_to_python_with_self(guard_expr))
        } else {
            // Guard doesn't use captured dims - include in main condition
            let guard_str = gen.value_to_python_with_self(guard_expr);
            checks.push(format!("({})", guard_str));
            None
        }
    } else {
        None
    };

    let condition = if checks.is_empty() {
        "True".to_string()
    } else {
        checks.join(" and ")
    };

    ShapeCheckResult { condition, bindings, guard_condition }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    #[test]
    fn test_shape_check_literals() {
        let program = Program::new();
        let gen = CodeGenerator::new(&program);

        let shape = Shape::new(vec![
            Dim::Wildcard,
            Dim::Literal(512),
        ]);

        let result = generate_shape_check(&gen, &shape, None, "x");
        assert_eq!(result.condition, "x.ndim == 2 and x.shape[1] == 512");
        assert!(result.bindings.is_empty());
        assert!(result.guard_condition.is_none());
    }

    #[test]
    fn test_shape_check_with_capture() {
        let program = Program::new();
        let gen = CodeGenerator::new(&program);

        let shape = Shape::new(vec![
            Dim::Wildcard,
            Dim::Named("d".to_string()),
        ]);

        let result = generate_shape_check(&gen, &shape, None, "x");
        assert_eq!(result.condition, "x.ndim == 2");
        assert_eq!(result.bindings, vec!["d = x.shape[1]"]);
        assert!(result.guard_condition.is_none());
    }

    #[test]
    fn test_shape_check_with_guard() {
        let program = Program::new();
        let gen = CodeGenerator::new(&program);

        let shape = Shape::new(vec![
            Dim::Wildcard,
            Dim::Named("d".to_string()),
        ]);

        let guard = Value::BinOp {
            op: BinOp::Gt,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(512)),
        };

        let result = generate_shape_check(&gen, &shape, Some(&guard), "x");
        assert_eq!(result.condition, "x.ndim == 2");
        assert_eq!(result.bindings, vec!["d = x.shape[1]"]);
        assert_eq!(result.guard_condition, Some("d > 512".to_string()));
    }
}
