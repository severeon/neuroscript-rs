//! Forward method generation
//!
//! This module generates the forward() method body from the connection graph.
//! It handles all endpoint types including calls, match expressions, tuple unpacking,
//! and references.

use super::utils::*;
use crate::interfaces::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Generate a unique, sanitized variable name from a hint.
/// Tracks used names to avoid collisions.
fn make_var_name(used: &mut HashSet<String>, hint: &str) -> String {
    // Sanitize: replace non-alphanumeric with underscore, ensure starts with letter
    let sanitized: String = hint
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let sanitized = if sanitized.is_empty()
        || sanitized.starts_with(|c: char| c.is_ascii_digit())
    {
        format!("v_{}", sanitized)
    } else {
        sanitized
    };

    if !used.contains(&sanitized) {
        used.insert(sanitized.clone());
        return sanitized;
    }

    // Append counter to deduplicate
    let mut counter = 2;
    loop {
        let candidate = format!("{}_{}", sanitized, counter);
        if !used.contains(&candidate) {
            used.insert(candidate.clone());
            return candidate;
        }
        counter += 1;
    }
}

/// Emit a shape comment for a bound module call, using the called neuron's output shape.
/// Only emits if the shape is different from the last emitted shape.
fn emit_bound_module_shape_comment(
    gen: &mut CodeGenerator,
    output: &mut String,
    binding_name: &str,
    indent: &str,
) {
    // Look up the neuron being called via binding_to_call_name
    if let Some(call_name) = gen.binding_to_call_name.get(binding_name).cloned() {
        if let Some(neuron_def) = gen.program.neurons.get(&call_name) {
            if !neuron_def.outputs.is_empty() {
                let shape = &neuron_def.outputs[0].shape;
                let shape_comment = gen.format_shape_for_comment(shape);

                // Only emit if shape changed
                if gen.last_emitted_shape.as_ref() != Some(&shape_comment) {
                    writeln!(
                        output,
                        "{}# {}() output shape: {}",
                        indent, call_name, shape_comment
                    )
                    .unwrap();
                    gen.last_emitted_shape = Some(shape_comment);
                }
            }
        }
    }
}

/// Emit a shape comment and optional assertion for a variable at a Ref destination.
/// Suppresses comments for _unroll temp refs and when shape hasn't changed.
fn emit_shape_comment_and_assertion(
    gen: &mut CodeGenerator,
    output: &mut String,
    var_name: &str,
    node_name: &str,
    indent: &str,
) {
    // Skip shape comments for unroll temp refs (they carry wrong shapes)
    if node_name.starts_with("_unroll") {
        return;
    }

    // Try to get the inferred shape for this node
    if let Some(shapes) = gen.inference_ctx.node_outputs.get(node_name) {
        if !shapes.is_empty() {
            let shape = &shapes[0];
            let shape_comment = gen.format_shape_for_comment(shape);

            // Only emit comment if shape changed
            if gen.last_emitted_shape.as_ref() != Some(&shape_comment) {
                writeln!(output, "{}# Expected shape: {}", indent, shape_comment).unwrap();
                gen.last_emitted_shape = Some(shape_comment.clone());
            }

            // Assertions still always emitted (runtime checking, not documentation)
            if gen.should_assert_shape(shape) {
                if let Some(expected_shape) = gen.format_shape_for_assertion(shape) {
                    writeln!(
                        output,
                        "{}assert {}.shape == {}, f\"Shape mismatch: expected {}, got {{{}.shape}}\"",
                        indent, var_name, expected_shape, shape_comment, var_name
                    ).unwrap();
                }
            }
        }
    }
}

/// Generate forward method body from connections
pub(super) fn generate_forward_body(
    gen: &mut CodeGenerator,
    output: &mut String,
    connections: &[Connection],
    inputs: &[&str],
) -> Result<(), CodegenError> {
    // Don't clear var_names - preserve module bindings from __init__
    // Only add/override with input port mappings

    // Map global names (available in all neurons)
    for global in &gen.program.globals {
        gen.var_names
            .insert(global.name.clone(), global.name.clone());
    }

    // Map input ports to initial variables
    if inputs.len() == 1 && inputs[0] == "default" {
        gen.var_names.insert("in".to_string(), "x".to_string());
    } else {
        for input in inputs {
            gen.var_names
                .insert((*input).to_string(), (*input).to_string());
        }
    }

    // Track used variable names for semantic naming
    let mut used_var_names: HashSet<String> = HashSet::new();
    // Reserve input variable names
    if inputs.len() == 1 && inputs[0] == "default" {
        used_var_names.insert("x".to_string());
    } else {
        for input in inputs {
            used_var_names.insert((*input).to_string());
        }
    }

    let indent = "        ";

    // Build a map from Call endpoints to their result variable names
    let mut call_to_result: HashMap<String, String> = HashMap::new();

    // Build a map from Match endpoints to their result variable names
    let mut match_to_result: HashMap<String, String> = HashMap::new();

    // Build a map from If endpoints to their result variable names
    let mut if_to_result: HashMap<String, String> = HashMap::new();

    // Track the last call result for bound modules (context bindings).
    // This is separate from var_names so the module reference is preserved
    // for subsequent calls (e.g., @static bindings called multiple times).
    let mut binding_call_results: HashMap<String, String> = HashMap::new();

    // Track which unroll groups have already been emitted as for loops
    let mut emitted_unroll_groups: HashSet<String> = HashSet::new();

    // Track the last result variable (for implicit output)
    let mut last_result = None;

    // Reset last_emitted_shape for this forward method
    gen.last_emitted_shape = None;

    // Process each connection
    for conn in connections {
        // Resolve the source to a variable name
        let source_var = match &conn.source {
            Endpoint::Ref(port_ref) => {
                // Check binding call results first (for modules called multiple times)
                if let Some(result) = binding_call_results.get(&port_ref.node) {
                    result.clone()
                } else {
                    gen.var_names
                        .get(&port_ref.node)
                        .cloned()
                        .unwrap_or_else(|| port_ref.node.clone())
                }
            }
            Endpoint::Tuple(refs) => {
                let vars: Vec<String> = refs
                    .iter()
                    .map(|r| {
                        gen.var_names
                            .get(&r.node)
                            .cloned()
                            .unwrap_or_else(|| r.node.clone())
                    })
                    .collect();
                format!("({})", vars.join(", "))
            }
            Endpoint::Call { name, .. } => {
                // This Call should have been processed in a previous connection
                // Look it up in our call_to_result map
                let key = endpoint_key_impl(&conn.source);
                call_to_result.get(&key).cloned().ok_or_else(|| {
                    CodegenError::InvalidConnection(format!(
                        "Call to {} used as source before being defined",
                        name
                    ))
                })?
            }
            Endpoint::Match(_) => {
                // Look it up in our match_to_result map
                let key = endpoint_key_impl(&conn.source);
                match_to_result.get(&key).cloned().ok_or_else(|| {
                    CodegenError::InvalidConnection(format!(
                        "Match expression used as source before being defined"
                    ))
                })?
            }
            Endpoint::If(_) => {
                // Look it up in our if_to_result map
                let key = endpoint_key_impl(&conn.source);
                if_to_result.get(&key).cloned().ok_or_else(|| {
                    CodegenError::InvalidConnection(format!(
                        "If expression used as source before being defined"
                    ))
                })?
            }
            Endpoint::Reshape(_) => {
                let key = endpoint_key_impl(&conn.source);
                call_to_result.get(&key).cloned().ok_or_else(|| {
                    CodegenError::InvalidConnection(format!(
                        "Reshape used as source before being processed"
                    ))
                })?
            }
        };

        // Determine source output count for implicit fork detection
        let source_output_count = match &conn.source {
            Endpoint::Ref(port_ref) => gen
                .inference_ctx
                .node_outputs
                .get(&port_ref.node)
                .map(|s| s.len())
                .unwrap_or(1),
            Endpoint::Call { id, .. } => gen
                .inference_ctx
                .call_outputs
                .get(id)
                .map(|s| s.len())
                .unwrap_or(1),
            Endpoint::Tuple(refs) => refs.len(),
            _ => 1,
        };

        // For Reshape destinations, emit dimension bindings from source shape
        // before processing the reshape itself
        let mut reshape_source_shape_for_dest: Option<Shape> = None;
        if let Endpoint::Reshape(reshape) = &conn.destination {
            // Look up the source shape from inference context
            let source_shape = match &conn.source {
                Endpoint::Ref(port_ref) => gen
                    .inference_ctx
                    .node_outputs
                    .get(&port_ref.node)
                    .and_then(|shapes| shapes.first().cloned()),
                Endpoint::Call { id, .. } => gen
                    .inference_ctx
                    .call_outputs
                    .get(id)
                    .and_then(|shapes| shapes.first().cloned()),
                Endpoint::Reshape(r) => gen
                    .inference_ctx
                    .call_outputs
                    .get(&r.id)
                    .and_then(|shapes| shapes.first().cloned()),
                _ => None,
            };

            if let Some(ref src_shape) = source_shape {
                // Build a map of dim_name -> source_index from the source shape
                let mut src_dim_indices: HashMap<String, usize> = HashMap::new();
                for (i, dim) in src_shape.dims.iter().enumerate() {
                    if let Dim::Named(name) = dim {
                        src_dim_indices.insert(name.clone(), i);
                    }
                }

                // Emit bindings for Named dims in reshape target that aren't params
                // and aren't Binding expressions (those are handled in process_destination)
                for dim in &reshape.dims {
                    if let ReshapeDim::Named(name) = dim {
                        if !gen.current_neuron_params.contains(name)
                            && !gen.binding_context.contains_key(name)
                        {
                            if let Some(&idx) = src_dim_indices.get(name) {
                                writeln!(
                                    output,
                                    "{}{} = {}.size({})",
                                    indent, name, source_var, idx
                                )
                                .unwrap();
                            }
                        }
                    }
                }
            }

            // Store source shape for passing to process_destination
            reshape_source_shape_for_dest = source_shape;
        }

        // Process the destination
        let result_var = process_destination(
            output,
            gen,
            &conn.destination,
            source_var,
            indent,
            &mut used_var_names,
            &mut call_to_result,
            &mut match_to_result,
            &mut if_to_result,
            &mut binding_call_results,
            &mut emitted_unroll_groups,
            source_output_count,
            reshape_source_shape_for_dest.as_ref(),
        )?;

        // Track the last result for implicit output
        last_result = Some(result_var.clone());

        // If destination was a Call, store result in call_to_result
        if let Endpoint::Call { .. } = &conn.destination {
            let key = endpoint_key_impl(&conn.destination);
            call_to_result.insert(key, result_var.clone());
        } else if let Endpoint::Match(_) = &conn.destination {
            let key = endpoint_key_impl(&conn.destination);
            match_to_result.insert(key, result_var.clone());
        } else if let Endpoint::If(_) = &conn.destination {
            let key = endpoint_key_impl(&conn.destination);
            if_to_result.insert(key, result_var.clone());
        }
    }

    // Return the output variable
    // Priority: explicit "out" port > last result > fallback
    let output_var = gen
        .var_names
        .get("out")
        .cloned()
        .or(last_result)
        .unwrap_or_else(|| "x".to_string());
    writeln!(output, "        return {}", output_var).unwrap();

    Ok(())
}

/// Process a destination endpoint, generating code and returning the result variable name
///
/// `source_output_count` indicates how many outputs the source produces.
/// When 1 and destination is a multi-element tuple, we emit individual assignments
/// (implicit fork) instead of tuple unpacking.
fn process_destination(
    output: &mut String,
    gen: &mut CodeGenerator,
    endpoint: &Endpoint,
    source_var: String,
    indent: &str,
    used_var_names: &mut HashSet<String>,
    call_to_result: &mut HashMap<String, String>,
    match_to_result: &mut HashMap<String, String>,
    if_to_result: &mut HashMap<String, String>,
    binding_call_results: &mut HashMap<String, String>,
    emitted_unroll_groups: &mut HashSet<String>,
    source_output_count: usize,
    reshape_source_shape: Option<&Shape>,
) -> Result<String, CodegenError> {
    match endpoint {
        Endpoint::Ref(port_ref) => {
            // Check if this is a reference to a bound module (from context:)
            // Bound modules have var_names entries like "norm" -> "self.norm" or "extra" -> "self._extra"
            if let Some(module_ref) = gen.var_names.get(&port_ref.node) {
                if module_ref.starts_with("self.") {
                    // Check if this is a direct reference to an aggregate name (e.g., "blocks")
                    if let Some((base_name, count, is_static)) = gen.aggregate_to_group.get(&port_ref.node).cloned() {
                        let list_name = &port_ref.node;

                        if !emitted_unroll_groups.contains(list_name) {
                            emitted_unroll_groups.insert(list_name.clone());

                            if is_static {
                                // @static: call same class-level instance N times
                                let range_expr = match &count {
                                    Value::Name(param_name) => format!("self.{}", param_name),
                                    Value::Int(n) => n.to_string(),
                                    _ => "1".to_string(),
                                };
                                writeln!(
                                    output,
                                    "{}for _ in range({}):",
                                    indent, range_expr
                                )
                                .unwrap();
                                writeln!(
                                    output,
                                    "{}    {} = self.__class__.{}({})",
                                    indent, source_var, base_name, source_var
                                )
                                .unwrap();
                            } else {
                                // Instance: iterate over nn.ModuleList
                                writeln!(
                                    output,
                                    "{}for {} in self.{}:",
                                    indent, base_name, list_name
                                )
                                .unwrap();
                                writeln!(
                                    output,
                                    "{}    {} = {}({})",
                                    indent, source_var, base_name, source_var
                                )
                                .unwrap();
                            }

                            binding_call_results
                                .insert(port_ref.node.clone(), source_var.clone());
                            return Ok(source_var);
                        } else {
                            binding_call_results
                                .insert(port_ref.node.clone(), source_var.clone());
                            return Ok(source_var);
                        }
                    }

                    // Check if this is part of an unroll group (individual member)
                    if let Some(group_info) = gen.binding_to_unroll_group.get(&port_ref.node).cloned() {
                        let base_name = &group_info.base_name;
                        let list_name = group_info.aggregate_name.clone();

                        if !emitted_unroll_groups.contains(&list_name) {
                            // First encounter of this group: emit a for loop
                            emitted_unroll_groups.insert(list_name.clone());

                            // Use the source var as the loop variable (mutated in-place)
                            writeln!(
                                output,
                                "{}for {} in self.{}:",
                                indent, base_name, list_name
                            )
                            .unwrap();
                            writeln!(
                                output,
                                "{}    {} = {}({})",
                                indent, source_var, base_name, source_var
                            )
                            .unwrap();

                            // Emit shape comment once after the loop
                            emit_bound_module_shape_comment(gen, output, &port_ref.node, indent);

                            // The result is the source var (mutated in-place through the loop)
                            binding_call_results
                                .insert(port_ref.node.clone(), source_var.clone());
                            return Ok(source_var);
                        } else {
                            // Subsequent member of already-emitted group: no-op
                            // Just update binding_call_results to point to source_var
                            binding_call_results
                                .insert(port_ref.node.clone(), source_var.clone());
                            return Ok(source_var);
                        }
                    }

                    // Check if this is a lazy binding (starts with "self._")
                    if module_ref.starts_with("self._")
                        && gen.lazy_bindings.contains_key(&port_ref.node)
                    {
                        // Generate lazy instantiation code
                        let (call_name, args, kwargs) =
                            gen.lazy_bindings.get(&port_ref.node).unwrap();

                        let args_str = args
                            .iter()
                            .map(|v| gen.value_to_python_with_self(v))
                            .collect::<Vec<_>>()
                            .join(", ");

                        let kwargs_str = if kwargs.is_empty() {
                            String::new()
                        } else {
                            let kw: Vec<String> = kwargs
                                .iter()
                                .map(|(k, v)| format!("{}={}", k, gen.value_to_python_with_self(v)))
                                .collect();
                            if args.is_empty() {
                                kw.join(", ")
                            } else {
                                format!(", {}", kw.join(", "))
                            }
                        };

                        // Generate lazy instantiation check
                        writeln!(output, "{}if {} is None:", indent, module_ref).unwrap();
                        writeln!(
                            output,
                            "{}    {} = {}({}{})",
                            indent, module_ref, call_name, args_str, kwargs_str
                        )
                        .unwrap();
                    }

                    // This is a bound module - generate a call with semantic name
                    let result_var = make_var_name(used_var_names, &port_ref.node);
                    writeln!(
                        output,
                        "{}{} = {}({})",
                        indent, result_var, module_ref, source_var
                    )
                    .unwrap();

                    // Emit shape comment using the called neuron's output shape
                    emit_bound_module_shape_comment(gen, output, &port_ref.node, indent);

                    // Store the call result separately so the module reference is preserved
                    // in var_names for subsequent calls (e.g., @static called N times).
                    binding_call_results
                        .insert(port_ref.node.clone(), result_var.clone());
                    return Ok(result_var);
                }
            }

            // Simple port reference - the source becomes this port's variable
            gen.var_names
                .insert(port_ref.node.clone(), source_var.clone());

            // Emit shape comment/assertion for intermediate nodes
            if port_ref.node != "in" && port_ref.node != "out" {
                emit_shape_comment_and_assertion(gen, output, &source_var, &port_ref.node, indent);
            }

            Ok(source_var)
        }
        Endpoint::Tuple(refs) => {
            // Tuple unpacking with semantic names
            let var_names: Vec<String> = refs
                .iter()
                .map(|r| {
                    let v = make_var_name(used_var_names, &r.node);
                    gen.var_names.insert(r.node.clone(), v.clone());
                    v
                })
                .collect();

            if source_output_count == 1 && var_names.len() > 1 {
                // Implicit fork: assign source to each binding individually
                for var_name in &var_names {
                    writeln!(output, "{}{} = {}", indent, var_name, source_var).unwrap();
                }
            } else {
                // Multi-output source: standard tuple unpacking
                writeln!(
                    output,
                    "{}{} = {}",
                    indent,
                    var_names.join(", "),
                    source_var
                )
                .unwrap();
            }
            Ok(source_var) // Return tuple as result
        }
        Endpoint::Call {
            name,
            args,
            kwargs,
            id,
            frozen: _,
        } => {
            // Generate a call to the module
            let key = endpoint_key_impl(endpoint);
            let module_name = gen.call_to_module.get(&key).cloned().ok_or_else(|| {
                CodegenError::InvalidConnection(format!("Module for call to {} not found", name))
            })?;

            // Semantic name from module attribute name
            let result_var = make_var_name(used_var_names, &module_name);

            // Check if this call has captured dimensions (needs lazy instantiation)
            let has_captured = args
                .iter()
                .any(|v| has_captured_dimensions_impl(v, &gen.current_neuron_params))
                || kwargs
                    .iter()
                    .any(|(_, v)| has_captured_dimensions_impl(v, &gen.current_neuron_params));

            if has_captured {
                // Lazy instantiation: check cache, instantiate if needed
                writeln!(output, "{}if self._{} is None:", indent, module_name).unwrap();

                // Generate instantiation with current dimension values
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

                // Mark as primitive for imports
                if let Some(neuron) = gen.program.neurons.get(name.as_str()) {
                    if neuron.is_primitive() {
                        gen.used_primitives.insert(name.clone());
                    }
                } else {
                    gen.used_primitives.insert(name.clone());
                }

                writeln!(
                    output,
                    "{}    self._{} = {}({}{})",
                    indent, module_name, name, args_str, kwargs_str
                )
                .unwrap();

                // Call the lazily-instantiated module
                writeln!(
                    output,
                    "{}{} = self._{}({})",
                    indent, result_var, module_name, source_var
                )
                .unwrap();
            } else {
                // Normal call to pre-instantiated module
                writeln!(
                    output,
                    "{}{} = self.{}({})",
                    indent, result_var, module_name, source_var
                )
                .unwrap();
            }

            // Emit shape comment/assertion for the call result
            // Look up the output shape from call_outputs using the call ID
            if let Some(shapes) = gen.inference_ctx.call_outputs.get(id) {
                if !shapes.is_empty() {
                    let shape = &shapes[0];
                    let shape_comment = gen.format_shape_for_comment(shape);

                    // Only emit if shape changed
                    if gen.last_emitted_shape.as_ref() != Some(&shape_comment) {
                        writeln!(
                            output,
                            "{}# {}() output shape: {}",
                            indent, name, shape_comment
                        )
                        .unwrap();
                        gen.last_emitted_shape = Some(shape_comment.clone());
                    }

                    if gen.should_assert_shape(shape) {
                        if let Some(expected_shape) = gen.format_shape_for_assertion(shape) {
                            writeln!(
                                output,
                                "{}assert {}.shape == {}, f\"{}() shape mismatch: expected {}, got {{{}.shape}}\"",
                                indent, result_var, expected_shape, name, shape_comment, result_var
                            ).unwrap();
                        }
                    }
                }
            }

            Ok(result_var)
        }
        Endpoint::Match(match_expr) => {
            let result_var = make_var_name(used_var_names, "match_out");

            // Initialize result_var to None for safety (though not strictly needed if all paths return)
            writeln!(output, "{}{} = None", indent, result_var).unwrap();

            let mut first = true;
            let mut prev_condition = String::new();

            // Check if there's a reachable catch-all arm (all wildcards, no guard)
            let has_reachable_catchall = match_expr.arms.iter().any(|arm| {
                arm.is_reachable
                    && arm.guard.is_none()
                    && match &arm.pattern {
                        MatchPattern::Shape(shape) => shape
                            .dims
                            .iter()
                            .all(|d| matches!(d, Dim::Wildcard | Dim::Variadic(_))),
                        MatchPattern::NeuronContract(_) => false,
                    }
            });

            for arm in &match_expr.arms {
                // Skip unreachable arms (pruned by optimizer)
                if !arm.is_reachable {
                    continue;
                }

                let shape = match &arm.pattern {
                    MatchPattern::Shape(s) => s,
                    MatchPattern::NeuronContract(_) => {
                        // NeuronContract should be resolved before codegen
                        return Err(CodegenError::UnsupportedFeature(
                            "NeuronContract patterns must be resolved before codegen".to_string(),
                        ));
                    }
                };
                let shape_check =
                    generate_shape_check(gen, shape, arm.guard.as_ref(), &source_var);

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

                let mut prev_endpoint: Option<&Endpoint> = None;
                for ep in &arm.pipeline {
                    // For Reshape endpoints, try to resolve source shape from previous endpoint
                    let arm_reshape_src = if matches!(ep, Endpoint::Reshape(_)) {
                        resolve_endpoint_source_shape(prev_endpoint, gen)
                    } else {
                        None
                    };

                    current_var = process_destination(
                        output,
                        gen,
                        ep,
                        current_var,
                        &pipeline_indent,
                        used_var_names,
                        call_to_result,
                        match_to_result,
                        if_to_result,
                        binding_call_results,
                        emitted_unroll_groups,
                        1,
                        arm_reshape_src.as_ref(),
                    )?;

                    // If endpoint was a Call, store result in call_to_result
                    if let Endpoint::Call { .. } = ep {
                        let key = endpoint_key_impl(ep);
                        call_to_result.insert(key, current_var.clone());
                    }
                    prev_endpoint = Some(ep);
                }

                writeln!(
                    output,
                    "{}{} = {}",
                    pipeline_indent, result_var, current_var
                )
                .unwrap();

                // Restore var_names to prevent match arm scope from leaking
                gen.var_names = saved_var_names;
            }

            // Else clause: only raise if no reachable catch-all exists
            if !has_reachable_catchall {
                writeln!(output, "{}else:", indent).unwrap();
                writeln!(
                    output,
                    "{}    raise ValueError(f'No match found for shape {{ {}.shape }}')",
                    indent, source_var
                )
                .unwrap();
            }

            Ok(result_var)
        }
        Endpoint::If(if_expr) => {
            let result_var = make_var_name(used_var_names, "cond_out");

            // Initialize result variable to None
            writeln!(output, "{}{} = None", indent, result_var).unwrap();

            // Handle branches
            for (i, branch) in if_expr.branches.iter().enumerate() {
                let prefix = if i == 0 { "if" } else { "elif" };
                // Ensure self.param is used in conditions
                let cond_str = gen.value_to_python_with_self(&branch.condition);

                writeln!(output, "{}{} {}:", indent, prefix, cond_str).unwrap();
                let branch_indent = format!("{}    ", indent);

                // Process pipeline
                let saved_var_names = gen.var_names.clone();
                let mut current_var = source_var.clone();

                let mut prev_ep: Option<&Endpoint> = None;
                for ep in &branch.pipeline {
                    let branch_reshape_src = if matches!(ep, Endpoint::Reshape(_)) {
                        resolve_endpoint_source_shape(prev_ep, gen)
                    } else {
                        None
                    };

                    current_var = process_destination(
                        output,
                        gen,
                        ep,
                        current_var,
                        &branch_indent,
                        used_var_names,
                        call_to_result,
                        match_to_result,
                        if_to_result,
                        binding_call_results,
                        emitted_unroll_groups,
                        1,
                        branch_reshape_src.as_ref(),
                    )?;
                    // Cache call/match/if results inside branch
                    if let Endpoint::Call { .. } = ep {
                        let key = endpoint_key_impl(ep);
                        call_to_result.insert(key, current_var.clone());
                    } else if let Endpoint::Match(_) = ep {
                        let key = endpoint_key_impl(ep);
                        match_to_result.insert(key, current_var.clone());
                    } else if let Endpoint::If(_) = ep {
                        let key = endpoint_key_impl(ep);
                        if_to_result.insert(key, current_var.clone());
                    }
                    prev_ep = Some(ep);
                }

                writeln!(output, "{}{} = {}", branch_indent, result_var, current_var).unwrap();
                gen.var_names = saved_var_names;
            }

            // Else branch
            if let Some(else_branch) = &if_expr.else_branch {
                writeln!(output, "{}else:", indent).unwrap();
                let branch_indent = format!("{}    ", indent);
                let saved_var_names = gen.var_names.clone();
                let mut current_var = source_var.clone();

                let mut prev_ep: Option<&Endpoint> = None;
                for ep in else_branch {
                    let else_reshape_src = if matches!(ep, Endpoint::Reshape(_)) {
                        resolve_endpoint_source_shape(prev_ep, gen)
                    } else {
                        None
                    };

                    current_var = process_destination(
                        output,
                        gen,
                        ep,
                        current_var,
                        &branch_indent,
                        used_var_names,
                        call_to_result,
                        match_to_result,
                        if_to_result,
                        binding_call_results,
                        emitted_unroll_groups,
                        1,
                        else_reshape_src.as_ref(),
                    )?;
                    if let Endpoint::Call { .. } = ep {
                        let key = endpoint_key_impl(ep);
                        call_to_result.insert(key, current_var.clone());
                    } else if let Endpoint::Match(_) = ep {
                        let key = endpoint_key_impl(ep);
                        match_to_result.insert(key, current_var.clone());
                    } else if let Endpoint::If(_) = ep {
                        let key = endpoint_key_impl(ep);
                        if_to_result.insert(key, current_var.clone());
                    }
                    prev_ep = Some(ep);
                }
                writeln!(output, "{}{} = {}", branch_indent, result_var, current_var).unwrap();
                gen.var_names = saved_var_names;
            } else {
                // Implicit else: pass-through/Identity
                writeln!(output, "{}else:", indent).unwrap();
                writeln!(output, "{}    {} = {}", indent, result_var, source_var).unwrap();
            }

            Ok(result_var)
        }
        Endpoint::Reshape(reshape) => {
            let result_var = make_var_name(used_var_names, "x");

            // Helper closure: resolve a ReshapeDim name to a Python expression.
            // Priority: neuron param -> self.param, binding_context, else use as-is
            let resolve_dim_name = |name: &str| -> String {
                if gen.current_neuron_params.contains(name) {
                    format!("self.{}", name)
                } else if let Some(resolved) = gen.binding_context.get(name) {
                    resolved.clone()
                } else {
                    // Assume it's already in scope (from match capture, prior reshape, etc.)
                    name.to_string()
                }
            };

            // Convert a ReshapeDim to Python expression
            let dim_to_py = |d: &ReshapeDim| -> String {
                match d {
                    ReshapeDim::Named(name) => resolve_dim_name(name),
                    ReshapeDim::Literal(n) => n.to_string(),
                    ReshapeDim::Binding { name, .. } => name.clone(),
                    ReshapeDim::Others => "-1".to_string(),
                    ReshapeDim::Expr(expr) => dim_expr_to_python(gen, expr),
                }
            };

            match &reshape.annotation {
                None => {
                    // Bare => : element-preserving reshape
                    // Emit binding assignments first (e.g., dh = self.dim // self.heads)
                    for dim in &reshape.dims {
                        if let ReshapeDim::Binding { name, expr } = dim {
                            let expr_str = value_to_python_int_div(gen, expr);
                            writeln!(output, "{}{} = {}", indent, name, expr_str).unwrap();
                        }
                    }

                    // Build target shape args
                    let shape_args = reshape
                        .dims
                        .iter()
                        .map(|d| dim_to_py(d))
                        .collect::<Vec<_>>()
                        .join(", ");

                    writeln!(
                        output,
                        "{}{} = {}.reshape({})",
                        indent, result_var, source_var, shape_args
                    )
                    .unwrap();
                }
                Some(TransformAnnotation::Reduce(strategy)) => match strategy {
                    TransformStrategy::Intrinsic(name) => {
                        let method = match name.as_str() {
                            "mean" => "mean",
                            "sum" => "sum",
                            "min" => "amin",
                            "max" => "amax",
                            "prod" => "prod",
                            "logsumexp" => "logsumexp",
                            _ => {
                                return Err(CodegenError::UnsupportedFeature(format!(
                                    "unknown reduce intrinsic: {}",
                                    name
                                )))
                            }
                        };

                        // Determine which source dims to reduce by comparing
                        // source shape dim names with target dim names
                        let target_dim_names: HashSet<String> = reshape
                            .dims
                            .iter()
                            .filter_map(|d| match d {
                                ReshapeDim::Named(n) => Some(n.clone()),
                                ReshapeDim::Binding { name, .. } => Some(name.clone()),
                                _ => None,
                            })
                            .collect();

                        // NOTE: This only matches Dim::Named entries in the source shape.
                        // Wildcard (*) and variadic (*name) dims are skipped, which means
                        // @reduce on shapes with wildcards may produce incorrect dim indices.
                        // Source shapes must be fully named for correct reduce codegen.
                        let reduce_dims: Vec<usize> =
                            if let Some(src_shape) = reshape_source_shape {
                                // Find source dims not present in target
                                src_shape
                                    .dims
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(i, dim)| {
                                        if let Dim::Named(name) = dim {
                                            if !target_dim_names.contains(name) {
                                                return Some(i);
                                            }
                                        }
                                        None
                                    })
                                    .collect()
                            } else {
                                return Err(CodegenError::InvalidConnection(
                                    "cannot determine reduce dimensions: source shape unavailable".to_string(),
                                ));
                            };

                        if reduce_dims.is_empty() {
                            return Err(CodegenError::InvalidConnection(
                                "cannot determine reduce dimensions: source shape has no named dims to reduce (may contain wildcards/variadics)".to_string(),
                            ));
                        }

                        if reduce_dims.len() == 1 {
                            writeln!(
                                output,
                                "{}{} = {}.{}(dim={})",
                                indent, result_var, source_var, method, reduce_dims[0]
                            )
                            .unwrap();
                        } else {
                            let dims_str = reduce_dims
                                .iter()
                                .map(|d| d.to_string())
                                .collect::<Vec<_>>()
                                .join(", ");
                            writeln!(
                                output,
                                "{}{} = {}.{}(dim=({}))",
                                indent, result_var, source_var, method, dims_str
                            )
                            .unwrap();
                        }
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
                },
                Some(TransformAnnotation::Repeat(strategy)) => match strategy {
                    TransformStrategy::Intrinsic(name) if name == "copy" => {
                        // Determine which target dims are new (not in source shape).
                        // These require unsqueeze before expand.
                        let source_dim_names: HashSet<String> =
                            if let Some(src_shape) = reshape_source_shape {
                                src_shape
                                    .dims
                                    .iter()
                                    .filter_map(|d| {
                                        if let Dim::Named(n) = d {
                                            Some(n.clone())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect()
                            } else {
                                HashSet::new()
                            };

                        // Find indices of new dims in target that need unsqueeze
                        let mut unsqueeze_indices: Vec<usize> = Vec::new();
                        for (i, dim) in reshape.dims.iter().enumerate() {
                            let is_new = match dim {
                                ReshapeDim::Literal(_) => {
                                    // Literal dims like `1` are new dimensions
                                    true
                                }
                                ReshapeDim::Named(n) => !source_dim_names.contains(n),
                                _ => false,
                            };
                            if is_new {
                                unsqueeze_indices.push(i);
                            }
                        }

                        // Emit unsqueeze for each new dimension (in order)
                        let mut current = source_var.clone();
                        for (offset, &idx) in unsqueeze_indices.iter().enumerate() {
                            let unsqueezed = if offset == unsqueeze_indices.len() - 1
                                && unsqueeze_indices.len() > 0
                            {
                                // Last unsqueeze can write to result_var if no expand needed
                                // But we always need expand, so use a temp
                                make_var_name(used_var_names, "x")
                            } else {
                                make_var_name(used_var_names, "x")
                            };
                            writeln!(
                                output,
                                "{}{} = {}.unsqueeze({})",
                                indent,
                                unsqueezed,
                                current,
                                idx + offset // Account for previous unsqueezes shifting indices
                            )
                            .unwrap();
                            current = unsqueezed;
                        }

                        let shape_args = reshape
                            .dims
                            .iter()
                            .map(|d| dim_to_py(d))
                            .collect::<Vec<_>>()
                            .join(", ");

                        if unsqueeze_indices.is_empty() {
                            // No new dims — just expand existing size-1 dims
                            writeln!(
                                output,
                                "{}{} = {}.expand({})",
                                indent, result_var, source_var, shape_args
                            )
                            .unwrap();
                        } else {
                            // Expand after unsqueezing
                            writeln!(
                                output,
                                "{}{} = {}.expand({})",
                                indent, result_var, current, shape_args
                            )
                            .unwrap();
                        }
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
                        // Validator rejects unknown intrinsics, so this is unreachable
                        unreachable!("unknown repeat strategy should be caught by validator")
                    }
                },
            }

            // Store result for when this reshape is used as a source
            let key = endpoint_key_impl(endpoint);
            call_to_result.insert(key, result_var.clone());

            Ok(result_var)
        }
    }
}
/// Convert a Value to Python code for dimension arithmetic (uses // for division)
/// Resolves neuron parameter names to self.{name}
fn value_to_python_int_div(gen: &CodeGenerator, value: &Value) -> String {
    match value {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => format!("\"{}\"", s),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Name(n) => {
            if gen.current_neuron_params.contains(n) {
                format!("self.{}", n)
            } else if let Some(resolved) = gen.binding_context.get(n) {
                resolved.clone()
            } else {
                n.clone()
            }
        }
        Value::Global(n) => n.clone(),
        Value::BinOp { op, left, right } => {
            let op_str = match op {
                BinOp::Div => "//",
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Lt => "<",
                BinOp::Gt => ">",
                BinOp::Le => "<=",
                BinOp::Ge => ">=",
                BinOp::Eq => "==",
                BinOp::Ne => "!=",
            };
            format!(
                "{} {} {}",
                value_to_python_int_div(gen, left),
                op_str,
                value_to_python_int_div(gen, right)
            )
        }
        Value::Call { name, args, kwargs } => {
            let args_str = args
                .iter()
                .map(|v| value_to_python_int_div(gen, v))
                .collect::<Vec<_>>()
                .join(", ");
            let kwargs_str = if kwargs.is_empty() {
                String::new()
            } else {
                let kw: Vec<String> = kwargs
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, value_to_python_int_div(gen, v)))
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

/// Convert a DimExpr to Python code for dimension arithmetic (uses // for division)
fn dim_expr_to_python(gen: &CodeGenerator, expr: &DimExpr) -> String {
    let op_str = match expr.op {
        BinOp::Div => "//",
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        // Grammar only allows +, -, *, / in dimension expressions.
        // Comparison operators cannot reach codegen through valid programs.
        other => unreachable!(
            "operator {:?} should not appear in reshape dimension expression",
            other
        ),
    };
    let left = reshape_dim_ref_to_python(gen, &expr.left);
    let right = reshape_dim_ref_to_python(gen, &expr.right);
    format!("{} {} {}", left, op_str, right)
}

/// Convert a Dim (when used as part of a DimExpr in a reshape) to Python
fn reshape_dim_ref_to_python(gen: &CodeGenerator, dim: &Dim) -> String {
    match dim {
        Dim::Literal(n) => n.to_string(),
        Dim::Named(n) => {
            if gen.current_neuron_params.contains(n) {
                format!("self.{}", n)
            } else if let Some(resolved) = gen.binding_context.get(n) {
                resolved.clone()
            } else {
                n.clone()
            }
        }
        Dim::Global(n) => n.clone(),
        Dim::Expr(e) => dim_expr_to_python(gen, e),
        Dim::Wildcard => "-1".to_string(),
        Dim::Variadic(_) => "-1".to_string(),
    }
}

/// Resolve the output shape from a previous endpoint in a pipeline.
///
/// Used when a Reshape follows another endpoint inside a Match/If arm pipeline,
/// so the reshape knows its source shape for @reduce/@repeat dim calculations.
fn resolve_endpoint_source_shape(
    prev: Option<&Endpoint>,
    gen: &CodeGenerator,
) -> Option<Shape> {
    match prev? {
        Endpoint::Call { id, .. } => gen
            .inference_ctx
            .call_outputs
            .get(id)
            .and_then(|shapes| shapes.first().cloned()),
        Endpoint::Ref(port_ref) => gen
            .inference_ctx
            .node_outputs
            .get(&port_ref.node)
            .and_then(|shapes| shapes.first().cloned()),
        Endpoint::Reshape(r) => gen
            .inference_ctx
            .call_outputs
            .get(&r.id)
            .and_then(|shapes| shapes.first().cloned()),
        _ => None,
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
    var_name: &str,
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
        if !bindings.is_empty()
            && has_captured_dimensions_impl(guard_expr, &gen.current_neuron_params)
        {
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

    ShapeCheckResult {
        condition,
        bindings,
        guard_condition,
    }
}

#[cfg(test)]
mod tests;
