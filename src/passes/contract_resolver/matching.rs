//! Contract pattern matching against concrete neuron ports.

use crate::interfaces::*;

/// Find the first arm whose NeuronPortContract matches the given ports
pub(super) fn find_matching_arm(
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
