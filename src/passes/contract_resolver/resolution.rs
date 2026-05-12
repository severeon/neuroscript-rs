//! Core contract resolution logic.

use crate::interfaces::*;
use std::collections::HashMap;

use super::call_sites::find_call_sites;
use super::detection::collect_named_match_params;
use super::matching::find_matching_arm;
use super::MAX_CONTRACT_RESOLUTION_DEPTH;

/// Resolve contract matches for a specific neuron by examining its call sites.
/// Returns any errors encountered during resolution.
pub(super) fn resolve_contracts_for_neuron(
    program: &mut Program,
    neuron_name: &str,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Extract only the data we need from the neuron definition before taking a
    // mutable borrow on `program` for `resolve_match_in_neuron`. This avoids
    // cloning the entire NeuronDef.
    let (match_params, param_info) = {
        let neuron = match program.neurons.get(neuron_name) {
            Some(n) => n,
            None => return errors,
        };

        let match_params = collect_named_match_params(neuron);
        if match_params.is_empty() {
            return errors;
        }

        // Extract (name, index, type_annotation) for each param used in a named match
        let param_info: Vec<(String, usize, Option<ParamType>)> = match_params
            .iter()
            .filter_map(|name| {
                neuron.params.iter().position(|p| &p.name == name).map(|idx| {
                    (
                        name.clone(),
                        idx,
                        neuron.params[idx].type_annotation.clone(),
                    )
                })
            })
            .collect();

        (match_params, param_info)
    };

    let _ = match_params; // consumed via param_info

    // For each parameter used in a named match, find call sites that pass a concrete neuron
    // and resolve the match at that call site
    for (param_name, param_idx, type_annotation) in &param_info {
        // Verify the parameter is typed as Neuron. The validator should have already
        // caught this, but we check defensively to avoid resolving contracts on
        // non-Neuron parameters.
        if *type_annotation != Some(ParamType::Neuron) {
            errors.push(ValidationError::Custom(format!(
                "Parameter '{}' in neuron '{}' is used in a contract match but is not \
                 typed as `: Neuron`. Add the type annotation to enable contract resolution.",
                param_name, neuron_name
            )));
            continue;
        }

        // Find all call sites in the program that instantiate this neuron.
        // Only `Value::Name` arguments are detected — complex expressions (conditionals,
        // nested calls) are intentionally skipped, as they cannot be resolved at compile
        // time. Such cases are left for runtime resolution or will produce a codegen error
        // if they contain NeuronContract patterns.
        let call_sites: Vec<(String, String)> =
            find_call_sites(program, neuron_name, param_name, *param_idx);

        // For each call site, resolve the contract
        for (_caller_name, concrete_neuron_name) in &call_sites {
            // Look up the concrete neuron's port declarations
            if let Some(concrete_neuron) = program.neurons.get(concrete_neuron_name) {
                // Build a substitution map from the concrete neuron's parameter defaults.
                // This resolves named dimensions (e.g., `d_model`) in port shapes when
                // the neuron has default values for those parameters.
                let bindings = build_default_bindings(&concrete_neuron.params);

                let input_ports: Vec<(String, Shape)> = concrete_neuron
                    .inputs
                    .iter()
                    .map(|p| {
                        let shape = if bindings.is_empty() {
                            p.shape.clone()
                        } else {
                            crate::validator::shapes::substitute_shape(&p.shape, &bindings)
                        };
                        (p.name.clone(), shape)
                    })
                    .collect();
                let output_ports: Vec<(String, Shape)> = concrete_neuron
                    .outputs
                    .iter()
                    .map(|p| {
                        let shape = if bindings.is_empty() {
                            p.shape.clone()
                        } else {
                            crate::validator::shapes::substitute_shape(&p.shape, &bindings)
                        };
                        (p.name.clone(), shape)
                    })
                    .collect();

                // Try to resolve the match expression
                errors.extend(resolve_match_in_neuron(
                    program,
                    neuron_name,
                    param_name,
                    &input_ports,
                    &output_ports,
                ));
            }
        }
    }

    errors
}

/// Build a bindings map from parameter default values.
/// Maps parameter names to their integer default values for shape substitution.
pub(super) fn build_default_bindings(params: &[Param]) -> HashMap<String, i64> {
    let mut bindings = HashMap::new();
    for param in params {
        if let Some(Value::Int(val)) = &param.default {
            bindings.insert(param.name.clone(), *val);
        }
    }
    bindings
}

/// Resolve a named match expression in a neuron by checking concrete port shapes
/// against the match arm contracts. Returns any errors encountered.
///
/// When a multi-endpoint pipeline is resolved, the first endpoint replaces the
/// match expression and additional connections are spliced into the connection list.
fn resolve_match_in_neuron(
    program: &mut Program,
    neuron_name: &str,
    param_name: &str,
    input_ports: &[(String, Shape)],
    output_ports: &[(String, Shape)],
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let neuron = match program.neurons.get_mut(neuron_name) {
        Some(n) => n,
        None => return errors,
    };

    if let NeuronBody::Graph { connections, .. } = &mut neuron.body {
        // Collect pending insertions: (insert_after_index, new_connections)
        let mut pending_insertions: Vec<(usize, Vec<Connection>)> = Vec::new();

        for (conn_idx, conn) in connections.iter_mut().enumerate() {
            // Resolve source endpoint
            let (src_errors, src_pipeline) = resolve_match_in_endpoint(
                &mut conn.source,
                param_name,
                input_ports,
                output_ports,
                neuron_name,
                0,
            );
            errors.extend(src_errors);
            if let Some(remaining) = src_pipeline {
                // Multi-endpoint resolution in source: conn.source is now the first
                // endpoint. Chain: conn.source -> remaining[0] -> remaining[1] -> ...
                let mut new_conns = Vec::new();
                // Connect the replaced endpoint to the first remaining
                new_conns.push(Connection {
                    source: conn.source.clone(),
                    destination: remaining[0].clone(),
                });
                // Connect remaining endpoints to each other
                for i in 0..remaining.len() - 1 {
                    new_conns.push(Connection {
                        source: remaining[i].clone(),
                        destination: remaining[i + 1].clone(),
                    });
                }
                pending_insertions.push((conn_idx, new_conns));
            }

            // Resolve destination endpoint
            let (dst_errors, dst_pipeline) = resolve_match_in_endpoint(
                &mut conn.destination,
                param_name,
                input_ports,
                output_ports,
                neuron_name,
                0,
            );
            errors.extend(dst_errors);
            if let Some(remaining) = dst_pipeline {
                // Multi-endpoint resolution in destination: conn.destination is now
                // the first endpoint. Chain: conn.dest -> remaining[0] -> remaining[1] -> ...
                let mut new_conns = Vec::new();
                // Connect the replaced endpoint to the first remaining
                new_conns.push(Connection {
                    source: conn.destination.clone(),
                    destination: remaining[0].clone(),
                });
                // Connect remaining endpoints to each other
                for i in 0..remaining.len() - 1 {
                    new_conns.push(Connection {
                        source: remaining[i].clone(),
                        destination: remaining[i + 1].clone(),
                    });
                }
                pending_insertions.push((conn_idx, new_conns));
            }
        }

        // Insert new connections in reverse order to preserve indices
        pending_insertions.sort_by(|a, b| b.0.cmp(&a.0));
        for (after_idx, new_conns) in pending_insertions {
            for (i, new_conn) in new_conns.into_iter().enumerate() {
                connections.insert(after_idx + 1 + i, new_conn);
            }
        }
    }
    errors
}

/// Recursively resolve named match expressions in an endpoint.
///
/// Returns `(errors, Option<pipeline>)`:
/// - `errors`: any validation errors encountered
/// - `Some(pipeline)`: when a multi-endpoint pipeline was resolved. The first
///   endpoint has already been placed into `*endpoint`; the returned vec contains
///   the remaining endpoints that need to be spliced into the connection graph.
/// - `None`: single-endpoint or no resolution occurred (handled inline)
fn resolve_match_in_endpoint(
    endpoint: &mut Endpoint,
    param_name: &str,
    input_ports: &[(String, Shape)],
    output_ports: &[(String, Shape)],
    neuron_name: &str,
    depth: usize,
) -> (Vec<ValidationError>, Option<Vec<Endpoint>>) {
    let mut errors = Vec::new();

    if depth >= MAX_CONTRACT_RESOLUTION_DEPTH {
        errors.push(ValidationError::Custom(format!(
            "Contract resolution depth limit ({}) exceeded in neuron '{}'. \
             This may indicate circular or excessively nested contract definitions.",
            MAX_CONTRACT_RESOLUTION_DEPTH, neuron_name
        )));
        return (errors, None);
    }

    match endpoint {
        Endpoint::Match(match_expr) => {
            let should_resolve = matches!(
                &match_expr.subject,
                MatchSubject::Named(name) if name == param_name
            );

            if should_resolve {
                match find_matching_arm(&match_expr.arms, input_ports, output_ports) {
                    Some(matching_arm_idx) => {
                        let pipeline = match_expr.arms[matching_arm_idx].pipeline.clone();
                        match pipeline.len() {
                            0 => {
                                // Empty pipeline — leave match in place
                            }
                            1 => {
                                // Single endpoint: replace the match directly
                                *endpoint = pipeline
                                    .into_iter()
                                    .next()
                                    .expect("pipeline verified to have exactly 1 element");
                                return (errors, None);
                            }
                            _ => {
                                // Multi-endpoint pipeline: replace match with first endpoint
                                // and return remaining endpoints for splicing
                                let mut pipeline_iter = pipeline.into_iter();
                                *endpoint = pipeline_iter.next().expect("multi-endpoint pipeline has at least one element");
                                let remaining: Vec<Endpoint> = pipeline_iter.collect();
                                return (errors, Some(remaining));
                            }
                        }
                    }
                    None => {
                        errors.push(ValidationError::Custom(format!(
                            "No contract arm in neuron '{}' matches the port shapes of the \
                             neuron passed as '{}'. Check that the concrete neuron's input/output \
                             shapes match at least one arm's port contract.",
                            neuron_name, param_name
                        )));
                    }
                }
            }

            // Recurse into arms for nested matches
            if let Endpoint::Match(match_expr) = endpoint {
                for arm in &mut match_expr.arms {
                    for ep in &mut arm.pipeline {
                        let (nested_errors, _) = resolve_match_in_endpoint(
                            ep,
                            param_name,
                            input_ports,
                            output_ports,
                            neuron_name,
                            depth + 1,
                        );
                        errors.extend(nested_errors);
                    }
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &mut if_expr.branches {
                for ep in &mut branch.pipeline {
                    let (nested_errors, _) = resolve_match_in_endpoint(
                        ep,
                        param_name,
                        input_ports,
                        output_ports,
                        neuron_name,
                        depth + 1,
                    );
                    errors.extend(nested_errors);
                }
            }
            if let Some(else_branch) = &mut if_expr.else_branch {
                for ep in else_branch {
                    let (nested_errors, _) = resolve_match_in_endpoint(
                        ep,
                        param_name,
                        input_ports,
                        output_ports,
                        neuron_name,
                        depth + 1,
                    );
                    errors.extend(nested_errors);
                }
            }
        }
        _ => {}
    }
    (errors, None)
}
