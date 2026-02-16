//! Compile-time neuron contract resolution.
//!
//! Resolves `match(param): in [...] -> out [...]: ...` expressions by inspecting
//! the concrete neuron passed as the parameter at each call site. The matching
//! arm's pipeline replaces the entire match expression.
//!
//! Called after validation/shape inference, before codegen.
//!
//! # Example
//!
//! Given a higher-order neuron:
//! ```text
//! neuron SmartStack(block: Neuron, d_model, count=6):
//!     ...
//!     graph:
//!         in -> match(block):
//!             in [*, seq, d] -> out [*, seq, d]: blocks -> out
//!             in [*, d] -> out [*, d]: blocks -> out
//! ```
//!
//! When `SmartStack` is called with `SmartStack(TransformerBlock, 512)`, the
//! resolver looks up `TransformerBlock`'s port declarations and matches them
//! against each arm's contract. The first matching arm's pipeline replaces the
//! match expression.
//!
//! # Multi-endpoint pipelines
//!
//! When a matching arm contains a multi-step pipeline (e.g., `blocks -> out`),
//! the resolver splices the pipeline into the parent connection graph:
//! - The first endpoint replaces the match expression inline
//! - Additional connections are inserted for the remaining pipeline steps

use crate::interfaces::*;
use std::collections::HashMap;

/// Maximum recursion depth for contract resolution to prevent infinite loops
/// in pathological cases (e.g., mutually recursive contract definitions).
const MAX_CONTRACT_RESOLUTION_DEPTH: usize = 32;

/// Resolve all neuron contract match expressions in the program.
///
/// For each composite neuron containing a `MatchExpr` with `MatchSubject::Named(param)`:
/// 1. Find all call sites where this neuron is instantiated with a concrete neuron argument
/// 2. Look up the concrete neuron's port declarations
/// 3. Match ports against each arm's `NeuronPortContract`
/// 4. Select the first matching arm's pipeline
///
/// Returns `Err` with collected errors if any contracts cannot be resolved.
#[must_use]
pub fn resolve_neuron_contracts(program: &mut Program) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Collect neurons that have Named match subjects
    let neurons_with_contracts: Vec<String> = program
        .neurons
        .iter()
        .filter(|(_, neuron)| has_named_match(neuron))
        .map(|(name, _)| name.clone())
        .collect();

    if neurons_with_contracts.is_empty() {
        return Ok(());
    }

    // For each neuron with contract matches, find call sites and resolve
    for neuron_name in &neurons_with_contracts {
        errors.extend(resolve_contracts_for_neuron(program, neuron_name));
    }

    // Post-resolution check: detect any remaining MatchSubject::Named patterns
    // that weren't resolved (e.g., because the argument was a complex expression
    // rather than a simple neuron name). These will cause codegen failures, so
    // report them here with a clear message.
    //
    // Only run this check when resolution itself didn't produce errors — if
    // resolution already failed (no matching arm, etc.), the Named match is
    // still present and would be redundantly flagged here.
    if errors.is_empty() {
        for (neuron_name, neuron) in &program.neurons {
            if let NeuronBody::Graph { connections, .. } = &neuron.body {
                for conn in connections {
                    collect_unresolved_contracts(&conn.source, neuron_name, &mut errors, 0);
                    collect_unresolved_contracts(&conn.destination, neuron_name, &mut errors, 0);
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Check for remaining MatchSubject::Named patterns that were not resolved.
/// These would cause codegen failures since NeuronContract patterns cannot be
/// lowered to runtime shape checks.
fn collect_unresolved_contracts(
    endpoint: &Endpoint,
    neuron_name: &str,
    errors: &mut Vec<ValidationError>,
    depth: usize,
) {
    if depth >= MAX_CONTRACT_RESOLUTION_DEPTH {
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

/// Check if a neuron definition contains any `MatchExpr` with `MatchSubject::Named`
fn has_named_match(neuron: &NeuronDef) -> bool {
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
        Endpoint::Call { .. } | Endpoint::Ref(_) | Endpoint::Tuple(_) => false,
    }
}

/// Resolve contract matches for a specific neuron by examining its call sites.
/// Returns any errors encountered during resolution.
fn resolve_contracts_for_neuron(
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
fn build_default_bindings(params: &[Param]) -> HashMap<String, i64> {
    let mut bindings = HashMap::new();
    for param in params {
        if let Some(Value::Int(val)) = &param.default {
            bindings.insert(param.name.clone(), *val);
        }
    }
    bindings
}

/// Collect parameter names used as subjects in named match expressions
fn collect_named_match_params(neuron: &NeuronDef) -> Vec<String> {
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

/// Find call sites in the program that instantiate the given neuron.
/// Returns `(caller_neuron_name, concrete_arg_value)` pairs.
///
/// Checks both positional arguments (by `param_idx`) and keyword arguments
/// (by `param_name`). Only `Value::Name` arguments are detected — complex
/// expressions (conditionals, nested calls, arithmetic) are not resolvable
/// at compile time and are intentionally skipped.
fn find_call_sites(
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
                                *endpoint = pipeline_iter.next().unwrap();
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

/// Find the first arm whose NeuronPortContract matches the given ports
fn find_matching_arm(
    arms: &[MatchArm],
    input_ports: &[(String, Shape)],
    output_ports: &[(String, Shape)],
) -> Option<usize> {
    for (idx, arm) in arms.iter().enumerate() {
        if !arm.is_reachable {
            continue;
        }
        match &arm.pattern {
            MatchPattern::NeuronContract(contract) => {
                if contract_matches(contract, input_ports, output_ports) {
                    return Some(idx);
                }
            }
            MatchPattern::Shape(_) => {
                // Shape patterns don't apply to neuron contract matching
                continue;
            }
        }
    }
    None
}

/// Check if a neuron port contract matches the given concrete ports
fn contract_matches(
    contract: &NeuronPortContract,
    input_ports: &[(String, Shape)],
    output_ports: &[(String, Shape)],
) -> bool {
    // Check input ports
    if !ports_match(&contract.input_ports, input_ports) {
        return false;
    }
    // Check output ports
    if !ports_match(&contract.output_ports, output_ports) {
        return false;
    }
    true
}

/// Check if contract port patterns match concrete port shapes
fn ports_match(
    contract_ports: &[(String, Shape)],
    concrete_ports: &[(String, Shape)],
) -> bool {
    // If contract has no ports specified, it matches anything
    if contract_ports.is_empty() {
        return true;
    }

    // Each contract port must find a matching concrete port
    for (contract_name, contract_shape) in contract_ports {
        let matching_concrete = if contract_name == "default" {
            // Default port: match against the default port or any single port
            concrete_ports
                .iter()
                .find(|(name, _)| name == "default")
                .or_else(|| {
                    if concrete_ports.len() == 1 {
                        concrete_ports.first()
                    } else {
                        None
                    }
                })
        } else {
            concrete_ports
                .iter()
                .find(|(name, _)| name == contract_name)
        };

        match matching_concrete {
            Some((_, concrete_shape)) => {
                if !shape_pattern_matches(contract_shape, concrete_shape) {
                    return false;
                }
            }
            None => return false,
        }
    }

    true
}

/// Check if a contract shape pattern matches a concrete shape.
///
/// Delegates to the validator's `shapes_compatible`, which checks structural
/// compatibility: wildcards match any single dimension, variadics match zero
/// or more, and named dimensions match any concrete dimension. This is the
/// same semantics used for connection shape validation — contract patterns
/// are intentionally treated identically to port shape patterns, since a
/// contract arm like `in [*, seq, d]` means "this block accepts any shape
/// matching `[*, seq, d]`", which is exactly what shapes_compatible checks.
fn shape_pattern_matches(pattern: &Shape, concrete: &Shape) -> bool {
    crate::validator::shapes::shapes_compatible(pattern, concrete)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shape(dims: Vec<Dim>) -> Shape {
        Shape { dims }
    }

    fn make_port(name: &str, dims: Vec<Dim>) -> Port {
        Port {
            name: name.to_string(),
            shape: make_shape(dims),
            variadic: false,
        }
    }

    fn make_neuron_contract_arm(
        input_dims: Vec<Dim>,
        output_dims: Vec<Dim>,
        pipeline: Vec<Endpoint>,
    ) -> MatchArm {
        MatchArm {
            pattern: MatchPattern::NeuronContract(NeuronPortContract {
                input_ports: vec![("default".to_string(), make_shape(input_dims))],
                output_ports: vec![("default".to_string(), make_shape(output_dims))],
            }),
            guard: None,
            pipeline,
            is_reachable: true,
        }
    }

    fn ref_endpoint(name: &str) -> Endpoint {
        Endpoint::Ref(PortRef {
            node: name.to_string(),
            port: "default".to_string(),
        })
    }

    #[test]
    fn test_has_named_match_empty() {
        let neuron = NeuronDef {
            name: "Test".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![],
            },
            max_cycle_depth: Some(10),
            doc: None,
        };
        assert!(!has_named_match(&neuron));
    }

    #[test]
    fn test_has_named_match_with_implicit() {
        let neuron = NeuronDef {
            name: "Test".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Implicit,
                        arms: vec![],
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        };
        assert!(!has_named_match(&neuron));
    }

    #[test]
    fn test_has_named_match_with_named() {
        let neuron = NeuronDef {
            name: "Test".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Named("block".to_string()),
                        arms: vec![],
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        };
        assert!(has_named_match(&neuron));
    }

    #[test]
    fn test_resolve_no_contracts() {
        let mut program = Program::new();
        program.neurons.insert(
            "Simple".to_string(),
            NeuronDef {
                name: "Simple".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = resolve_neuron_contracts(&mut program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_no_matching_arm_reports_error() {
        // Create a concrete neuron with [*, dim] ports
        let mut program = Program::new();
        program.neurons.insert(
            "ConcreteBlock".to_string(),
            NeuronDef {
                name: "ConcreteBlock".to_string(),
                params: vec![],
                inputs: vec![make_port(
                    "default",
                    vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                )],
                outputs: vec![make_port(
                    "default",
                    vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                )],
                body: NeuronBody::Primitive(ImplRef::Source {
                    source: "test".to_string(),
                    path: "test".to_string(),
                }),
                max_cycle_depth: None,
                doc: None,
            },
        );

        // Create a higher-order neuron with a contract that expects 3D shapes
        program.neurons.insert(
            "HigherOrder".to_string(),
            NeuronDef {
                name: "HigherOrder".to_string(),
                params: vec![Param {
                    name: "block".to_string(),
                    default: None,
                    type_annotation: Some(ParamType::Neuron),
                }],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: ref_endpoint("in"),
                        destination: Endpoint::Match(MatchExpr {
                            subject: MatchSubject::Named("block".to_string()),
                            arms: vec![make_neuron_contract_arm(
                                // Expects 3D input: [*, seq, dim]
                                vec![
                                    Dim::Wildcard,
                                    Dim::Named("seq".to_string()),
                                    Dim::Named("dim".to_string()),
                                ],
                                vec![
                                    Dim::Wildcard,
                                    Dim::Named("seq".to_string()),
                                    Dim::Named("dim".to_string()),
                                ],
                                vec![ref_endpoint("blocks")],
                            )],
                            id: 0,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        // Create a caller that passes ConcreteBlock (2D) to HigherOrder (expects 3D)
        program.neurons.insert(
            "Caller".to_string(),
            NeuronDef {
                name: "Caller".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![Binding {
                        name: "ho".to_string(),
                        call_name: "HigherOrder".to_string(),
                        args: vec![Value::Name("ConcreteBlock".to_string())],
                        kwargs: vec![],
                        scope: Scope::Instance { lazy: false },
                        frozen: false,
                        unroll_group: None,
                    }],
                    context_unrolls: vec![],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = resolve_neuron_contracts(&mut program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        let msg = format!("{}", errors[0]);
        assert!(
            msg.contains("No contract arm"),
            "Expected 'No contract arm' error, got: {}",
            msg
        );
    }

    #[test]
    fn test_multi_endpoint_pipeline_splices_connections() {
        // Create a concrete neuron with matching 2D ports
        let mut program = Program::new();
        program.neurons.insert(
            "Block2D".to_string(),
            NeuronDef {
                name: "Block2D".to_string(),
                params: vec![],
                inputs: vec![make_port(
                    "default",
                    vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                )],
                outputs: vec![make_port(
                    "default",
                    vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                )],
                body: NeuronBody::Primitive(ImplRef::Source {
                    source: "test".to_string(),
                    path: "test".to_string(),
                }),
                max_cycle_depth: None,
                doc: None,
            },
        );

        // Create higher-order neuron where matching arm has 2 endpoints (multi-step)
        program.neurons.insert(
            "MultiStep".to_string(),
            NeuronDef {
                name: "MultiStep".to_string(),
                params: vec![Param {
                    name: "block".to_string(),
                    default: None,
                    type_annotation: Some(ParamType::Neuron),
                }],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: ref_endpoint("in"),
                        destination: Endpoint::Match(MatchExpr {
                            subject: MatchSubject::Named("block".to_string()),
                            arms: vec![make_neuron_contract_arm(
                                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                                // Multi-endpoint pipeline: blocks -> out
                                vec![ref_endpoint("blocks"), ref_endpoint("out")],
                            )],
                            id: 0,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        // Caller passes Block2D to MultiStep
        program.neurons.insert(
            "Caller".to_string(),
            NeuronDef {
                name: "Caller".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![Binding {
                        name: "ms".to_string(),
                        call_name: "MultiStep".to_string(),
                        args: vec![Value::Name("Block2D".to_string())],
                        kwargs: vec![],
                        scope: Scope::Instance { lazy: false },
                        frozen: false,
                        unroll_group: None,
                    }],
                    context_unrolls: vec![],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = resolve_neuron_contracts(&mut program);
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        // Verify the match was replaced and connections were spliced
        let multi_step = program.neurons.get("MultiStep").unwrap();
        if let NeuronBody::Graph { connections, .. } = &multi_step.body {
            // Original: in -> match(block)
            // After resolution with pipeline [blocks, out]:
            //   in -> blocks (match replaced with first endpoint)
            //   blocks -> out (spliced connection)
            assert_eq!(
                connections.len(),
                2,
                "Expected 2 connections after splicing, got {}",
                connections.len()
            );

            // First connection: in -> blocks
            match &connections[0].destination {
                Endpoint::Ref(port_ref) => {
                    assert_eq!(port_ref.node, "blocks");
                }
                other => panic!("Expected Ref(blocks) destination, got {:?}", other),
            }

            // Second connection: blocks -> out
            match (&connections[1].source, &connections[1].destination) {
                (Endpoint::Ref(src), Endpoint::Ref(dst)) => {
                    assert_eq!(src.node, "blocks");
                    assert_eq!(dst.node, "out");
                }
                other => panic!(
                    "Expected blocks -> out connection, got {:?}",
                    other
                ),
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_multi_endpoint_pipeline_three_steps() {
        // Test a 3-step pipeline: blocks -> norm -> out
        let mut program = Program::new();
        program.neurons.insert(
            "Block3D".to_string(),
            NeuronDef {
                name: "Block3D".to_string(),
                params: vec![],
                inputs: vec![make_port(
                    "default",
                    vec![
                        Dim::Wildcard,
                        Dim::Named("seq".to_string()),
                        Dim::Named("dim".to_string()),
                    ],
                )],
                outputs: vec![make_port(
                    "default",
                    vec![
                        Dim::Wildcard,
                        Dim::Named("seq".to_string()),
                        Dim::Named("dim".to_string()),
                    ],
                )],
                body: NeuronBody::Primitive(ImplRef::Source {
                    source: "test".to_string(),
                    path: "test".to_string(),
                }),
                max_cycle_depth: None,
                doc: None,
            },
        );

        program.neurons.insert(
            "ThreeStep".to_string(),
            NeuronDef {
                name: "ThreeStep".to_string(),
                params: vec![Param {
                    name: "block".to_string(),
                    default: None,
                    type_annotation: Some(ParamType::Neuron),
                }],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: ref_endpoint("in"),
                        destination: Endpoint::Match(MatchExpr {
                            subject: MatchSubject::Named("block".to_string()),
                            arms: vec![make_neuron_contract_arm(
                                vec![
                                    Dim::Wildcard,
                                    Dim::Named("seq".to_string()),
                                    Dim::Named("dim".to_string()),
                                ],
                                vec![
                                    Dim::Wildcard,
                                    Dim::Named("seq".to_string()),
                                    Dim::Named("dim".to_string()),
                                ],
                                // 3-step pipeline
                                vec![
                                    ref_endpoint("blocks"),
                                    ref_endpoint("norm"),
                                    ref_endpoint("out"),
                                ],
                            )],
                            id: 0,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        program.neurons.insert(
            "Caller".to_string(),
            NeuronDef {
                name: "Caller".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![Binding {
                        name: "ts".to_string(),
                        call_name: "ThreeStep".to_string(),
                        args: vec![Value::Name("Block3D".to_string())],
                        kwargs: vec![],
                        scope: Scope::Instance { lazy: false },
                        frozen: false,
                        unroll_group: None,
                    }],
                    context_unrolls: vec![],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = resolve_neuron_contracts(&mut program);
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let three_step = program.neurons.get("ThreeStep").unwrap();
        if let NeuronBody::Graph { connections, .. } = &three_step.body {
            // in -> blocks, blocks -> norm, norm -> out
            assert_eq!(connections.len(), 3);
            // Verify chain
            match &connections[0].destination {
                Endpoint::Ref(r) => assert_eq!(r.node, "blocks"),
                o => panic!("Expected blocks, got {:?}", o),
            }
            match (&connections[1].source, &connections[1].destination) {
                (Endpoint::Ref(s), Endpoint::Ref(d)) => {
                    assert_eq!(s.node, "blocks");
                    assert_eq!(d.node, "norm");
                }
                o => panic!("Expected blocks->norm, got {:?}", o),
            }
            match (&connections[2].source, &connections[2].destination) {
                (Endpoint::Ref(s), Endpoint::Ref(d)) => {
                    assert_eq!(s.node, "norm");
                    assert_eq!(d.node, "out");
                }
                o => panic!("Expected norm->out, got {:?}", o),
            }
        }
    }

    #[test]
    fn test_single_endpoint_resolution_succeeds() {
        // Create a concrete neuron with matching 2D ports
        let mut program = Program::new();
        program.neurons.insert(
            "Block2D".to_string(),
            NeuronDef {
                name: "Block2D".to_string(),
                params: vec![],
                inputs: vec![make_port(
                    "default",
                    vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                )],
                outputs: vec![make_port(
                    "default",
                    vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                )],
                body: NeuronBody::Primitive(ImplRef::Source {
                    source: "test".to_string(),
                    path: "test".to_string(),
                }),
                max_cycle_depth: None,
                doc: None,
            },
        );

        // Higher-order neuron with single-endpoint arm
        program.neurons.insert(
            "Wrapper".to_string(),
            NeuronDef {
                name: "Wrapper".to_string(),
                params: vec![Param {
                    name: "block".to_string(),
                    default: None,
                    type_annotation: Some(ParamType::Neuron),
                }],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: ref_endpoint("in"),
                        destination: Endpoint::Match(MatchExpr {
                            subject: MatchSubject::Named("block".to_string()),
                            arms: vec![make_neuron_contract_arm(
                                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                                vec![ref_endpoint("blocks")],
                            )],
                            id: 0,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        // Caller passes Block2D
        program.neurons.insert(
            "Caller".to_string(),
            NeuronDef {
                name: "Caller".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![Binding {
                        name: "w".to_string(),
                        call_name: "Wrapper".to_string(),
                        args: vec![Value::Name("Block2D".to_string())],
                        kwargs: vec![],
                        scope: Scope::Instance { lazy: false },
                        frozen: false,
                        unroll_group: None,
                    }],
                    context_unrolls: vec![],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = resolve_neuron_contracts(&mut program);
        assert!(result.is_ok());

        // Verify the match was replaced with the ref endpoint
        let wrapper = program.neurons.get("Wrapper").unwrap();
        if let NeuronBody::Graph { connections, .. } = &wrapper.body {
            assert_eq!(connections.len(), 1);
            match &connections[0].destination {
                Endpoint::Ref(port_ref) => {
                    assert_eq!(port_ref.node, "blocks");
                }
                other => panic!("Expected Ref endpoint, got {:?}", other),
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_contract_match_nested_in_if_branch() {
        // Contract match inside an if-expression branch should still be resolved
        let mut program = Program::new();
        program.neurons.insert(
            "Block2D".to_string(),
            NeuronDef {
                name: "Block2D".to_string(),
                params: vec![],
                inputs: vec![make_port(
                    "default",
                    vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                )],
                outputs: vec![make_port(
                    "default",
                    vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                )],
                body: NeuronBody::Primitive(ImplRef::Source {
                    source: "test".to_string(),
                    path: "test".to_string(),
                }),
                max_cycle_depth: None,
                doc: None,
            },
        );

        // Higher-order neuron with contract match nested inside an if branch pipeline
        let contract_match = Endpoint::Match(MatchExpr {
            subject: MatchSubject::Named("block".to_string()),
            arms: vec![make_neuron_contract_arm(
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                vec![ref_endpoint("blocks")],
            )],
            id: 0,
        });

        program.neurons.insert(
            "Conditional".to_string(),
            NeuronDef {
                name: "Conditional".to_string(),
                params: vec![
                    Param {
                        name: "block".to_string(),
                        default: None,
                        type_annotation: Some(ParamType::Neuron),
                    },
                    Param {
                        name: "use_block".to_string(),
                        default: Some(Value::Int(1)),
                        type_annotation: None,
                    },
                ],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: ref_endpoint("in"),
                        destination: Endpoint::If(IfExpr {
                            branches: vec![IfBranch {
                                condition: Value::Name("use_block".to_string()),
                                pipeline: vec![contract_match],
                            }],
                            else_branch: Some(vec![ref_endpoint("out")]),
                            id: 0,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        // Caller
        program.neurons.insert(
            "Caller".to_string(),
            NeuronDef {
                name: "Caller".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![Binding {
                        name: "c".to_string(),
                        call_name: "Conditional".to_string(),
                        args: vec![
                            Value::Name("Block2D".to_string()),
                            Value::Int(1),
                        ],
                        kwargs: vec![],
                        scope: Scope::Instance { lazy: false },
                        frozen: false,
                        unroll_group: None,
                    }],
                    context_unrolls: vec![],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = resolve_neuron_contracts(&mut program);
        assert!(result.is_ok());

        // Verify the nested contract match was resolved inside the if branch
        let conditional = program.neurons.get("Conditional").unwrap();
        if let NeuronBody::Graph { connections, .. } = &conditional.body {
            if let Endpoint::If(if_expr) = &connections[0].destination {
                assert_eq!(if_expr.branches.len(), 1);
                // The contract match should have been replaced with Ref("blocks")
                assert_eq!(if_expr.branches[0].pipeline.len(), 1);
                match &if_expr.branches[0].pipeline[0] {
                    Endpoint::Ref(port_ref) => {
                        assert_eq!(port_ref.node, "blocks");
                    }
                    other => panic!(
                        "Expected Ref endpoint in if branch, got {:?}",
                        other
                    ),
                }
            } else {
                panic!("Expected If endpoint");
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_unresolved_contract_detected_post_resolution() {
        // A higher-order neuron with a Named match but no call sites should
        // be flagged as unresolved in the post-resolution check
        let mut program = Program::new();

        program.neurons.insert(
            "Uncalled".to_string(),
            NeuronDef {
                name: "Uncalled".to_string(),
                params: vec![Param {
                    name: "block".to_string(),
                    default: None,
                    type_annotation: Some(ParamType::Neuron),
                }],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: ref_endpoint("in"),
                        destination: Endpoint::Match(MatchExpr {
                            subject: MatchSubject::Named("block".to_string()),
                            arms: vec![make_neuron_contract_arm(
                                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                                vec![ref_endpoint("blocks")],
                            )],
                            id: 0,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = resolve_neuron_contracts(&mut program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        let msg = format!("{}", errors[0]);
        assert!(
            msg.contains("Unresolved contract match"),
            "Expected 'Unresolved contract match' error, got: {}",
            msg
        );
    }

    #[test]
    fn test_parameter_default_substitution_enables_matching() {
        // A concrete neuron with named dimensions and default values should
        // have those dimensions substituted before matching
        let mut program = Program::new();

        // TransformerBlock with d_model parameter defaulting to 512
        program.neurons.insert(
            "TransformerBlock".to_string(),
            NeuronDef {
                name: "TransformerBlock".to_string(),
                params: vec![Param {
                    name: "d_model".to_string(),
                    default: Some(Value::Int(512)),
                    type_annotation: None,
                }],
                inputs: vec![make_port(
                    "default",
                    vec![
                        Dim::Wildcard,
                        Dim::Named("seq".to_string()),
                        Dim::Named("d_model".to_string()),
                    ],
                )],
                outputs: vec![make_port(
                    "default",
                    vec![
                        Dim::Wildcard,
                        Dim::Named("seq".to_string()),
                        Dim::Named("d_model".to_string()),
                    ],
                )],
                body: NeuronBody::Primitive(ImplRef::Source {
                    source: "core".to_string(),
                    path: "transformer/TransformerBlock".to_string(),
                }),
                max_cycle_depth: None,
                doc: None,
            },
        );

        // Higher-order neuron matching on 3D shapes
        program.neurons.insert(
            "Stack".to_string(),
            NeuronDef {
                name: "Stack".to_string(),
                params: vec![Param {
                    name: "block".to_string(),
                    default: None,
                    type_annotation: Some(ParamType::Neuron),
                }],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: ref_endpoint("in"),
                        destination: Endpoint::Match(MatchExpr {
                            subject: MatchSubject::Named("block".to_string()),
                            arms: vec![make_neuron_contract_arm(
                                vec![
                                    Dim::Wildcard,
                                    Dim::Named("seq".to_string()),
                                    Dim::Named("d".to_string()),
                                ],
                                vec![
                                    Dim::Wildcard,
                                    Dim::Named("seq".to_string()),
                                    Dim::Named("d".to_string()),
                                ],
                                vec![ref_endpoint("layers")],
                            )],
                            id: 0,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        // Caller passes TransformerBlock
        program.neurons.insert(
            "Caller".to_string(),
            NeuronDef {
                name: "Caller".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![Binding {
                        name: "s".to_string(),
                        call_name: "Stack".to_string(),
                        args: vec![Value::Name("TransformerBlock".to_string())],
                        kwargs: vec![],
                        scope: Scope::Instance { lazy: false },
                        frozen: false,
                        unroll_group: None,
                    }],
                    context_unrolls: vec![],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = resolve_neuron_contracts(&mut program);
        assert!(
            result.is_ok(),
            "Expected Ok after default substitution, got: {:?}",
            result
        );

        // Verify the match was resolved
        let stack = program.neurons.get("Stack").unwrap();
        if let NeuronBody::Graph { connections, .. } = &stack.body {
            match &connections[0].destination {
                Endpoint::Ref(port_ref) => {
                    assert_eq!(port_ref.node, "layers");
                }
                other => panic!("Expected Ref(layers), got {:?}", other),
            }
        }
    }

    #[test]
    fn test_build_default_bindings() {
        let params = vec![
            Param {
                name: "d_model".to_string(),
                default: Some(Value::Int(512)),
                type_annotation: None,
            },
            Param {
                name: "block".to_string(),
                default: None,
                type_annotation: Some(ParamType::Neuron),
            },
            Param {
                name: "num_heads".to_string(),
                default: Some(Value::Int(8)),
                type_annotation: None,
            },
        ];

        let bindings = build_default_bindings(&params);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings["d_model"], 512);
        assert_eq!(bindings["num_heads"], 8);
        assert!(!bindings.contains_key("block"));
    }
}
