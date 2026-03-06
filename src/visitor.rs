//! Endpoint visitor utilities for traversing nested IR structures.
//!
//! Provides `walk_endpoints` and `walk_endpoints_mut` to recursively visit
//! all endpoints in a program, including those nested inside match arms,
//! if/elif/else branches, and wrap expressions.

use crate::interfaces::*;

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

fn walk_endpoint_recursive(endpoint: &Endpoint, neuron_name: &str, f: &mut impl FnMut(&Endpoint, &str)) {
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
