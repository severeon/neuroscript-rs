//! Compile-time neuron contract resolution.
//!
//! Resolves `match(param): in [...] -> out [...]: ...` expressions by inspecting
//! the concrete neuron passed as the parameter at each call site. The matching
//! arm's pipeline replaces the entire match expression.
//!
//! Called after validation/shape inference, before codegen.

use crate::interfaces::*;

/// Resolve all neuron contract match expressions in the program.
///
/// For each composite neuron containing a `MatchExpr` with `MatchSubject::Named(param)`:
/// 1. Find all call sites where this neuron is instantiated with a concrete neuron argument
/// 2. Look up the concrete neuron's port declarations
/// 3. Match ports against each arm's `NeuronPortContract`
/// 4. Select the first matching arm's pipeline
///
/// This is a best-effort pass: if the parameter isn't resolvable at compile time
/// (e.g., it's a runtime value), the match is left in place for runtime resolution.
pub fn resolve_neuron_contracts(program: &mut Program) -> Result<(), Vec<ValidationError>> {
    let errors = Vec::new();

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
        resolve_contracts_for_neuron(program, neuron_name);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
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

/// Resolve contract matches for a specific neuron by examining its call sites
fn resolve_contracts_for_neuron(program: &mut Program, neuron_name: &str) {
    // Get the neuron definition to find which param is the subject
    let neuron = match program.neurons.get(neuron_name) {
        Some(n) => n.clone(),
        None => return,
    };

    // Find named match subjects and their parameter names
    let match_params: Vec<String> = collect_named_match_params(&neuron);
    if match_params.is_empty() {
        return;
    }

    // For each parameter used in a named match, find call sites that pass a concrete neuron
    // and resolve the match at that call site
    for param_name in &match_params {
        // Find the parameter index
        let param_idx = match neuron.params.iter().position(|p| &p.name == param_name) {
            Some(idx) => idx,
            None => continue,
        };

        // Find all call sites in the program that instantiate this neuron
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
                resolve_match_in_neuron(
                    program,
                    neuron_name,
                    param_name,
                    &input_ports,
                    &output_ports,
                );
            }
        }
    }
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

/// Find call sites in the program that instantiate the given neuron
/// Returns (caller_neuron_name, concrete_arg_value) pairs
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
/// against the match arm contracts
fn resolve_match_in_neuron(
    program: &mut Program,
    neuron_name: &str,
    param_name: &str,
    input_ports: &[(String, Shape)],
    output_ports: &[(String, Shape)],
) {
    let neuron = match program.neurons.get_mut(neuron_name) {
        Some(n) => n,
        None => return,
    };

    if let NeuronBody::Graph { connections, .. } = &mut neuron.body {
        for conn in connections.iter_mut() {
            resolve_match_in_endpoint(
                &mut conn.source,
                param_name,
                input_ports,
                output_ports,
            );
            resolve_match_in_endpoint(
                &mut conn.destination,
                param_name,
                input_ports,
                output_ports,
            );
        }
    }
}

/// Recursively resolve named match expressions in an endpoint
fn resolve_match_in_endpoint(
    endpoint: &mut Endpoint,
    param_name: &str,
    input_ports: &[(String, Shape)],
    output_ports: &[(String, Shape)],
) {
    match endpoint {
        Endpoint::Match(match_expr) => {
            if let MatchSubject::Named(name) = &match_expr.subject {
                if name == param_name {
                    // Try to find a matching arm
                    if let Some(matching_arm_idx) =
                        find_matching_arm(&match_expr.arms, input_ports, output_ports)
                    {
                        // Replace the match with the matched arm's pipeline
                        let pipeline = match_expr.arms[matching_arm_idx].pipeline.clone();
                        if pipeline.len() == 1 {
                            *endpoint = pipeline.into_iter().next().unwrap();
                        }
                        // If multiple endpoints in pipeline, we'd need to restructure
                        // For now, only handle single-endpoint replacement
                    }
                }
            }
            // Also recurse into arms for nested matches
            if let Endpoint::Match(match_expr) = endpoint {
                for arm in &mut match_expr.arms {
                    for ep in &mut arm.pipeline {
                        resolve_match_in_endpoint(ep, param_name, input_ports, output_ports);
                    }
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &mut if_expr.branches {
                for ep in &mut branch.pipeline {
                    resolve_match_in_endpoint(ep, param_name, input_ports, output_ports);
                }
            }
            if let Some(else_branch) = &mut if_expr.else_branch {
                for ep in else_branch {
                    resolve_match_in_endpoint(ep, param_name, input_ports, output_ports);
                }
            }
        }
        _ => {}
    }
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

/// Check if a contract shape pattern matches a concrete shape
/// Uses the existing shape compatibility check from the validator
fn shape_pattern_matches(pattern: &Shape, concrete: &Shape) -> bool {
    crate::validator::shapes::shapes_compatible(pattern, concrete)
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    source: Endpoint::Ref(PortRef {
                        node: "in".to_string(),
                        port: "default".to_string(),
                    }),
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
                    source: Endpoint::Ref(PortRef {
                        node: "in".to_string(),
                        port: "default".to_string(),
                    }),
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
}
