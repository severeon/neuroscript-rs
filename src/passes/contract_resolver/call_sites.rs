//! Call site discovery for contract resolution.

use crate::interfaces::*;

use super::MAX_CONTRACT_RESOLUTION_DEPTH;

/// Find call sites in the program that instantiate the given neuron.
/// Returns `(caller_neuron_name, concrete_arg_value)` pairs.
///
/// Checks both positional arguments (by `param_idx`) and keyword arguments
/// (by `param_name`). Only `Value::Name` arguments are detected — complex
/// expressions (conditionals, nested calls, arithmetic) are not resolvable
/// at compile time and are intentionally skipped.
pub(super) fn find_call_sites(
    program: &Program,
    target_neuron: &str,
    param_name: &str,
    param_idx: usize,
) -> Vec<(String, String)> {
    let mut sites = Vec::new();

    for (caller_name, caller_def) in &program.neurons {
        if let NeuronBody::Graph {
            context_bindings,
            connections,
            ..
        } = &caller_def.body
        {
            // Check context bindings
            for binding in context_bindings {
                if binding.call_name == target_neuron {
                    // Check positional args first
                    if let Some(Value::Name(arg_name)) = binding.args.get(param_idx) {
                        sites.push((caller_name.clone(), arg_name.clone()));
                    } else {
                        // Check kwargs by parameter name
                        for (kw_name, kw_value) in &binding.kwargs {
                            if kw_name == param_name {
                                if let Value::Name(arg_name) = kw_value {
                                    sites.push((caller_name.clone(), arg_name.clone()));
                                }
                            }
                        }
                    }
                }
            }

            // Check inline calls in connections
            for conn in connections {
                find_call_sites_in_endpoint(
                    &conn.source,
                    target_neuron,
                    param_name,
                    param_idx,
                    caller_name,
                    &mut sites,
                    0,
                );
                find_call_sites_in_endpoint(
                    &conn.destination,
                    target_neuron,
                    param_name,
                    param_idx,
                    caller_name,
                    &mut sites,
                    0,
                );
            }
        }
    }

    sites
}

fn find_call_sites_in_endpoint(
    endpoint: &Endpoint,
    target_neuron: &str,
    param_name: &str,
    param_idx: usize,
    caller_name: &str,
    sites: &mut Vec<(String, String)>,
    depth: usize,
) {
    if depth >= MAX_CONTRACT_RESOLUTION_DEPTH {
        return;
    }
    match endpoint {
        Endpoint::Call {
            name, args, kwargs, ..
        } => {
            if name == target_neuron {
                // Check positional args first
                if let Some(Value::Name(arg_name)) = args.get(param_idx) {
                    sites.push((caller_name.to_string(), arg_name.clone()));
                } else {
                    // Check kwargs by parameter name
                    for (kw_name, kw_value) in kwargs {
                        if kw_name == param_name {
                            if let Value::Name(arg_name) = kw_value {
                                sites.push((caller_name.to_string(), arg_name.clone()));
                            }
                        }
                    }
                }
            }
        }
        Endpoint::Match(match_expr) => {
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    find_call_sites_in_endpoint(
                        ep,
                        target_neuron,
                        param_name,
                        param_idx,
                        caller_name,
                        sites,
                        depth + 1,
                    );
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    find_call_sites_in_endpoint(
                        ep,
                        target_neuron,
                        param_name,
                        param_idx,
                        caller_name,
                        sites,
                        depth + 1,
                    );
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    find_call_sites_in_endpoint(
                        ep,
                        target_neuron,
                        param_name,
                        param_idx,
                        caller_name,
                        sites,
                        depth + 1,
                    );
                }
            }
        }
        _ => {}
    }
}
