//! Detection of named match expressions and unresolved contracts.

use crate::interfaces::*;

use super::MAX_CONTRACT_RESOLUTION_DEPTH;

/// Check if a neuron definition contains any `MatchExpr` with `MatchSubject::Named`
pub(super) fn has_named_match(neuron: &NeuronDef) -> bool {
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        for conn in connections {
            if endpoint_has_named_match(&conn.source, 0)
                || endpoint_has_named_match(&conn.destination, 0)
            {
                return true;
            }
        }
    }
    false
}

/// Recursively check if an endpoint contains a named match
fn endpoint_has_named_match(endpoint: &Endpoint, depth: usize) -> bool {
    if depth >= MAX_CONTRACT_RESOLUTION_DEPTH {
        return false;
    }
    match endpoint {
        Endpoint::Match(match_expr) => {
            if matches!(match_expr.subject, MatchSubject::Named(_)) {
                return true;
            }
            // Also check nested endpoints in arms
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    if endpoint_has_named_match(ep, depth + 1) {
                        return true;
                    }
                }
            }
            false
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    if endpoint_has_named_match(ep, depth + 1) {
                        return true;
                    }
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    if endpoint_has_named_match(ep, depth + 1) {
                        return true;
                    }
                }
            }
            false
        }
        Endpoint::Call { .. } | Endpoint::Ref(_) | Endpoint::Tuple(_) | Endpoint::Reshape(_) | Endpoint::Wrap(_) => false,
    }
}

/// Check for remaining MatchSubject::Named patterns that were not resolved.
/// These would cause codegen failures since NeuronContract patterns cannot be
/// lowered to runtime shape checks.
pub(super) fn collect_unresolved_contracts(
    endpoint: &Endpoint,
    neuron_name: &str,
    errors: &mut Vec<ValidationError>,
    depth: usize,
) {
    if depth >= MAX_CONTRACT_RESOLUTION_DEPTH {
        errors.push(ValidationError::Custom(format!(
            "Contract resolution depth limit ({}) exceeded in neuron '{}'. \
             This may indicate deeply nested or circular contract definitions.",
            MAX_CONTRACT_RESOLUTION_DEPTH, neuron_name
        )));
        return;
    }
    match endpoint {
        Endpoint::Match(match_expr) => {
            if let MatchSubject::Named(param_name) = &match_expr.subject {
                errors.push(ValidationError::Custom(format!(
                    "Unresolved contract match on '{}' in neuron '{}'. Contract dispatch \
                     requires a concrete neuron name at the call site — ensure the argument \
                     for '{}' is a direct neuron name, not a complex expression.",
                    param_name, neuron_name, param_name
                )));
            }
            // Also check nested endpoints in arms
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    collect_unresolved_contracts(ep, neuron_name, errors, depth + 1);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    collect_unresolved_contracts(ep, neuron_name, errors, depth + 1);
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    collect_unresolved_contracts(ep, neuron_name, errors, depth + 1);
                }
            }
        }
        _ => {}
    }
}

/// Collect parameter names used as subjects in named match expressions
pub(super) fn collect_named_match_params(neuron: &NeuronDef) -> Vec<String> {
    let mut params = Vec::new();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        for conn in connections {
            collect_named_params_from_endpoint(&conn.source, &mut params, 0);
            collect_named_params_from_endpoint(&conn.destination, &mut params, 0);
        }
    }
    params.sort();
    params.dedup();
    params
}

fn collect_named_params_from_endpoint(endpoint: &Endpoint, params: &mut Vec<String>, depth: usize) {
    if depth >= MAX_CONTRACT_RESOLUTION_DEPTH {
        return;
    }
    match endpoint {
        Endpoint::Match(match_expr) => {
            if let MatchSubject::Named(name) = &match_expr.subject {
                params.push(name.clone());
            }
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    collect_named_params_from_endpoint(ep, params, depth + 1);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    collect_named_params_from_endpoint(ep, params, depth + 1);
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    collect_named_params_from_endpoint(ep, params, depth + 1);
                }
            }
        }
        _ => {}
    }
}
