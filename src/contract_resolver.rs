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
//! # Limitations
//!
//! - Only `Value::Name` arguments are resolved. Complex expressions (conditionals,
//!   calls) passed as neuron parameters are left unresolved for runtime handling.
//! - Single-endpoint pipelines in matching arms are replaced inline. Multi-endpoint
//!   pipelines require connection graph restructuring and are reported as errors.

use crate::interfaces::*;

/// Resolve all neuron contract match expressions in the program.
///
/// For each composite neuron containing a `MatchExpr` with `MatchSubject::Named(param)`:
/// 1. Find all call sites where this neuron is instantiated with a concrete neuron argument
/// 2. Look up the concrete neuron's port declarations
/// 3. Match ports against each arm's `NeuronPortContract`
/// 4. Select the first matching arm's pipeline
///
/// Returns `Err` with collected errors if any contracts cannot be resolved
/// (no matching arm, or multi-endpoint pipeline limitation).
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
    // resolution already failed (no matching arm, multi-step pipeline), the
    // Named match is still present and would be redundantly flagged here.
    if errors.is_empty() {
        for (neuron_name, neuron) in &program.neurons {
            if let NeuronBody::Graph { connections, .. } = &neuron.body {
                for conn in connections {
                    collect_unresolved_contracts(&conn.source, neuron_name, &mut errors);
                    collect_unresolved_contracts(&conn.destination, neuron_name, &mut errors);
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
) {
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
                    collect_unresolved_contracts(ep, neuron_name, errors);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    collect_unresolved_contracts(ep, neuron_name, errors);
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    collect_unresolved_contracts(ep, neuron_name, errors);
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
            if endpoint_has_named_match(&conn.source) || endpoint_has_named_match(&conn.destination)
            {
                return true;
            }
        }
    }
    false
}

/// Recursively check if an endpoint contains a named match
fn endpoint_has_named_match(endpoint: &Endpoint) -> bool {
    match endpoint {
        Endpoint::Match(match_expr) => {
            if matches!(match_expr.subject, MatchSubject::Named(_)) {
                return true;
            }
            // Also check nested endpoints in arms
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    if endpoint_has_named_match(ep) {
                        return true;
                    }
                }
            }
            false
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    if endpoint_has_named_match(ep) {
                        return true;
                    }
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    if endpoint_has_named_match(ep) {
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

    // Clone the neuron definition to release the immutable borrow on `program`.
    // This is necessary because we later call `resolve_match_in_neuron` which
    // borrows `program` mutably. The clone cost is acceptable since this only
    // runs for neurons containing Named match subjects (typically few).
    let neuron = match program.neurons.get(neuron_name) {
        Some(n) => n.clone(),
        None => return errors,
    };

    // Find named match subjects and their parameter names
    let match_params: Vec<String> = collect_named_match_params(&neuron);
    if match_params.is_empty() {
        return errors;
    }

    // For each parameter used in a named match, find call sites that pass a concrete neuron
    // and resolve the match at that call site
    for param_name in &match_params {
        // Find the parameter index
        let param_idx = match neuron.params.iter().position(|p| &p.name == param_name) {
            Some(idx) => idx,
            None => continue,
        };

        // Find all call sites in the program that instantiate this neuron.
        // Only `Value::Name` arguments are detected — complex expressions (conditionals,
        // nested calls) are intentionally skipped, as they cannot be resolved at compile
        // time. Such cases are left for runtime resolution or will produce a codegen error
        // if they contain NeuronContract patterns.
        let call_sites: Vec<(String, String)> = find_call_sites(program, neuron_name, param_idx);

        // For each call site, resolve the contract
        for (_caller_name, concrete_neuron_name) in &call_sites {
            // Look up the concrete neuron's port declarations
            if let Some(concrete_neuron) = program.neurons.get(concrete_neuron_name) {
                let input_ports: Vec<(String, Shape)> = concrete_neuron
                    .inputs
                    .iter()
                    .map(|p| (p.name.clone(), p.shape.clone()))
                    .collect();
                let output_ports: Vec<(String, Shape)> = concrete_neuron
                    .outputs
                    .iter()
                    .map(|p| (p.name.clone(), p.shape.clone()))
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

/// Collect parameter names used as subjects in named match expressions
fn collect_named_match_params(neuron: &NeuronDef) -> Vec<String> {
    let mut params = Vec::new();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        for conn in connections {
            collect_named_params_from_endpoint(&conn.source, &mut params);
            collect_named_params_from_endpoint(&conn.destination, &mut params);
        }
    }
    params.sort();
    params.dedup();
    params
}

fn collect_named_params_from_endpoint(endpoint: &Endpoint, params: &mut Vec<String>) {
    match endpoint {
        Endpoint::Match(match_expr) => {
            if let MatchSubject::Named(name) = &match_expr.subject {
                params.push(name.clone());
            }
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    collect_named_params_from_endpoint(ep, params);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    collect_named_params_from_endpoint(ep, params);
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    collect_named_params_from_endpoint(ep, params);
                }
            }
        }
        _ => {}
    }
}

/// Find call sites in the program that instantiate the given neuron.
/// Returns `(caller_neuron_name, concrete_arg_value)` pairs.
///
/// Only detects `Value::Name` arguments at the neuron parameter position.
/// Complex expressions (conditionals, nested calls, arithmetic) are not
/// resolvable at compile time and are intentionally skipped.
fn find_call_sites(
    program: &Program,
    target_neuron: &str,
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
                    if let Some(Value::Name(arg_name)) = binding.args.get(param_idx) {
                        sites.push((caller_name.clone(), arg_name.clone()));
                    }
                }
            }

            // Check inline calls in connections
            for conn in connections {
                find_call_sites_in_endpoint(
                    &conn.source,
                    target_neuron,
                    param_idx,
                    caller_name,
                    &mut sites,
                );
                find_call_sites_in_endpoint(
                    &conn.destination,
                    target_neuron,
                    param_idx,
                    caller_name,
                    &mut sites,
                );
            }
        }
    }

    sites
}

fn find_call_sites_in_endpoint(
    endpoint: &Endpoint,
    target_neuron: &str,
    param_idx: usize,
    caller_name: &str,
    sites: &mut Vec<(String, String)>,
) {
    match endpoint {
        Endpoint::Call { name, args, .. } => {
            if name == target_neuron {
                if let Some(Value::Name(arg_name)) = args.get(param_idx) {
                    sites.push((caller_name.to_string(), arg_name.clone()));
                }
            }
        }
        Endpoint::Match(match_expr) => {
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    find_call_sites_in_endpoint(ep, target_neuron, param_idx, caller_name, sites);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    find_call_sites_in_endpoint(ep, target_neuron, param_idx, caller_name, sites);
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    find_call_sites_in_endpoint(ep, target_neuron, param_idx, caller_name, sites);
                }
            }
        }
        _ => {}
    }
}

/// Resolve a named match expression in a neuron by checking concrete port shapes
/// against the match arm contracts. Returns any errors encountered.
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
        for conn in connections.iter_mut() {
            errors.extend(resolve_match_in_endpoint(
                &mut conn.source,
                param_name,
                input_ports,
                output_ports,
                neuron_name,
            ));
            errors.extend(resolve_match_in_endpoint(
                &mut conn.destination,
                param_name,
                input_ports,
                output_ports,
                neuron_name,
            ));
        }
    }
    errors
}

/// Recursively resolve named match expressions in an endpoint.
/// Returns errors for unresolvable cases (no matching arm, multi-endpoint pipeline).
fn resolve_match_in_endpoint(
    endpoint: &mut Endpoint,
    param_name: &str,
    input_ports: &[(String, Shape)],
    output_ports: &[(String, Shape)],
    neuron_name: &str,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
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
                                return errors;
                            }
                            n => {
                                // Multi-endpoint pipeline requires connection graph
                                // restructuring that is not yet implemented.
                                errors.push(ValidationError::Custom(format!(
                                    "Contract match on '{}' in neuron '{}' resolved to a \
                                     multi-step pipeline ({} endpoints). Currently only \
                                     single-endpoint replacement is supported. Use a single \
                                     endpoint in the matching arm's pipeline, or wrap the \
                                     pipeline in a separate neuron.",
                                    param_name, neuron_name, n
                                )));
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
                        errors.extend(resolve_match_in_endpoint(
                            ep,
                            param_name,
                            input_ports,
                            output_ports,
                            neuron_name,
                        ));
                    }
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &mut if_expr.branches {
                for ep in &mut branch.pipeline {
                    errors.extend(resolve_match_in_endpoint(
                        ep,
                        param_name,
                        input_ports,
                        output_ports,
                        neuron_name,
                    ));
                }
            }
            if let Some(else_branch) = &mut if_expr.else_branch {
                for ep in else_branch {
                    errors.extend(resolve_match_in_endpoint(
                        ep,
                        param_name,
                        input_ports,
                        output_ports,
                        neuron_name,
                    ));
                }
            }
        }
        _ => {}
    }
    errors
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
    fn test_multi_endpoint_pipeline_reports_error() {
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
                                // Multi-endpoint pipeline
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
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        let msg = format!("{}", errors[0]);
        assert!(
            msg.contains("multi-step pipeline"),
            "Expected multi-step pipeline error, got: {}",
            msg
        );
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
}
