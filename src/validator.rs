//! NeuroScript Graph Validator
//!
//! Validates that NeuroScript programs are well-formed:
//! 1. All referenced neurons exist
//! 2. Connection endpoints match (tuple arity, port names, shapes)
//! 3. No cycles in the dependency graph

use std::collections::{HashMap, HashSet};
use crate::ir::*;

/// Validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    MissingNeuron {
        name: String,
        context: String,
        referenced_in: String,
    },
    PortMismatch {
        source_info: String,
        dest_info: String,
        context: String,
        details: String,
    },
    CycleDetected {
        cycle: Vec<String>,
        context: String,
    },
    TupleArityMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingNeuron { name, context, referenced_in } => {
                write!(f, "Neuron '{}' not found (referenced in {}, context: {})",
                       name, referenced_in, context)
            }
            ValidationError::PortMismatch { source_info, dest_info, context, details } => {
                write!(f, "Port mismatch: {} -> {} (context: {}, details: {})",
                       source_info, dest_info, context, details)
            }
            ValidationError::CycleDetected { cycle, context } => {
                write!(f, "Cycle detected in {}: {} -> ... -> {}",
                       context, cycle.join(" -> "), cycle[0])
            }
            ValidationError::TupleArityMismatch { expected, got, context } => {
                write!(f, "Tuple arity mismatch: expected {} ports, got {} (context: {})",
                       expected, got, context)
            }
        }
    }
}

/// Graph validator
pub struct Validator;

impl Validator {
    /// Validate an entire program
    pub fn validate(program: &Program) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check each neuron
        for (neuron_name, neuron) in &program.neurons {
            // Validate connections within this neuron if it's composite
            if let NeuronBody::Graph(connections) = &neuron.body {
                errors.extend(Self::validate_connections(
                    connections, neuron_name, program
                ));

                // Check for cycles in this neuron's graph
                errors.extend(Self::detect_cycles(connections, neuron_name));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check that all neuron references in connections exist
    fn validate_connections(
        connections: &[Connection],
        context_neuron: &str,
        program: &Program,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for connection in connections {
            // Check source endpoint for neuron references
            errors.extend(Self::check_endpoint_neuron_exists(
                &connection.source, context_neuron, program
            ));

            // Check destination endpoint for neuron references
            errors.extend(Self::check_endpoint_neuron_exists(
                &connection.destination, context_neuron, program
            ));

            // Check connection compatibility
            errors.extend(Self::check_connection_compatibility(
                connection, context_neuron, program
            ));
        }

        errors
    }

    /// Check if endpoint refers to existing neurons
    fn check_endpoint_neuron_exists(
        endpoint: &Endpoint,
        context_neuron: &str,
        program: &Program,
    ) -> Vec<ValidationError> {
        match endpoint {
            Endpoint::Call { name, .. } => {
                if !program.neurons.contains_key(name) {
                    vec![ValidationError::MissingNeuron {
                        name: name.clone(),
                        context: "neuron instantiation".to_string(),
                        referenced_in: context_neuron.to_string(),
                    }]
                } else {
                    vec![]
                }
            }
            Endpoint::Tuple(refs) => {
                refs.iter().flat_map(|port_ref| {
                    if let Some(interneuron) = Self::is_internal_neuron_call(port_ref.node.as_str(), context_neuron, program) {
                        Self::check_endpoint_neuron_exists(&interneuron, context_neuron, program)
                    } else {
                        vec![]
                    }
                }).collect()
            }
            Endpoint::Match(match_expr) => {
                match_expr.arms.iter().flat_map(|arm| {
                    arm.pipeline.iter().flat_map(|ep| {
                        Self::check_endpoint_neuron_exists(ep, context_neuron, program)
                    }).collect::<Vec<_>>()
                }).collect()
            }
            Endpoint::Ref(_) => vec![],
        }
    }

    /// Check if a port reference refers to an interneuron call
    /// Returns the interneuron's output endpoint if it does
    fn is_internal_neuron_call(node: &str, context_neuron: &str, _program: &Program) -> Option<Endpoint> {
        // This is a simplified check - in a full implementation we'd track
        // interneuron temporaries in the graph. For now, assume refs are to ports.
        None
    }

    /// Check connection compatibility between endpoints
    /// For now, only validates direct neuron-to-neuron connections and basic tuple structure
    fn check_connection_compatibility(
        connection: &Connection,
        context_neuron: &str,
        program: &Program,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Only validate connections between neuron calls, or from neuron calls to inputs/outputs
        // Skip intermediate port reference connections for now
        match (&connection.source, &connection.destination) {
            // Neuron call to neuron call: check port compatibility
            (Endpoint::Call { name: src_name, .. }, Endpoint::Call { name: dst_name, .. }) => {
                if let (Some(src_neuron), Some(dst_neuron)) = (
                    program.neurons.get(src_name),
                    program.neurons.get(dst_name)
                ) {
                    // Check outputs of source match inputs of destination
                    if src_neuron.outputs.len() != dst_neuron.inputs.len() {
                        errors.push(ValidationError::TupleArityMismatch {
                            expected: dst_neuron.inputs.len(),
                            got: src_neuron.outputs.len(),
                            context: format!("{}: {} -> {}", context_neuron, src_name, dst_name),
                        });
                    } else {
                        // Check each port pair for shape compatibility
                        for (i, (src_port, dst_port)) in src_neuron.outputs.iter().zip(&dst_neuron.inputs).enumerate() {
                            if src_port.shape != dst_port.shape {
                                errors.push(ValidationError::PortMismatch {
                                    source_info: format!("{}: {}", src_name, Self::port_desc(src_port)),
                                    dest_info: format!("{}: {}", dst_name, Self::port_desc(dst_port)),
                                    context: format!("{} (port {})", context_neuron, i),
                                    details: format!("shape mismatch: {} vs {}", src_port.shape, dst_port.shape),
                                });
                            }
                        }
                    }
                }
            }
            // Neuron call to input port: validate shapes match neuron's input ports
            (Endpoint::Call { name, .. }, Endpoint::Ref(port_ref)) if port_ref.node == "in" => {
                if let Some(src_neuron) = program.neurons.get(name) {
                    let context_neuron_def = program.neurons.get(context_neuron).unwrap();

                    // Check if port exists and shapes match
                    if let Some(input_port) = context_neuron_def.inputs.iter().find(|p| p.name == port_ref.port) {
                        if src_neuron.outputs.len() == 1 {
                            if src_neuron.outputs[0].shape != input_port.shape {
                                errors.push(ValidationError::PortMismatch {
                                    source_info: format!("{}: {}", name, Self::port_desc(&src_neuron.outputs[0])),
                                    dest_info: format!("input port {}.{}", port_ref.node, port_ref.port),
                                    context: format!("{}: {} -> in.{}", context_neuron, name, port_ref.port),
                                    details: format!("shape mismatch: {} vs {}", src_neuron.outputs[0].shape, input_port.shape),
                                });
                            }
                        } else {
                            errors.push(ValidationError::TupleArityMismatch {
                                expected: 1,
                                got: src_neuron.outputs.len(),
                                context: format!("{}: {} -> in.{}", context_neuron, name, port_ref.port),
                            });
                        }
                    }
                }
            }
            // Output port to neuron call: validate shapes match neuron's input ports
            (Endpoint::Ref(port_ref), Endpoint::Call { name, .. }) if port_ref.node == "out" => {
                if let Some(dst_neuron) = program.neurons.get(name) {
                    let context_neuron_def = program.neurons.get(context_neuron).unwrap();

                    // Check if port exists and shapes match
                    if let Some(output_port) = context_neuron_def.outputs.iter().find(|p| p.name == port_ref.port) {
                        if dst_neuron.inputs.len() == 1 {
                            if output_port.shape != dst_neuron.inputs[0].shape {
                                errors.push(ValidationError::PortMismatch {
                                    source_info: format!("output port {}.{}", port_ref.node, port_ref.port),
                                    dest_info: format!("{}: {}", name, Self::port_desc(&dst_neuron.inputs[0])),
                                    context: format!("{}: out.{} -> {}", context_neuron, port_ref.port, name),
                                    details: format!("shape mismatch: {} vs {}", output_port.shape, dst_neuron.inputs[0].shape),
                                });
                            }
                        } else {
                            errors.push(ValidationError::TupleArityMismatch {
                                expected: 1,
                                got: dst_neuron.inputs.len(),
                                context: format!("{}: out.{} -> {}", context_neuron, port_ref.port, name),
                            });
                        }
                    }
                }
            }
            // Tuple validation - basic arity check
            (Endpoint::Tuple(src_refs), endpoint) => {
                match endpoint {
                    Endpoint::Call { name, .. } => {
                        if let Some(dst_neuron) = program.neurons.get(name) {
                            if src_refs.len() != dst_neuron.inputs.len() {
                                errors.push(ValidationError::TupleArityMismatch {
                                    expected: dst_neuron.inputs.len(),
                                    got: src_refs.len(),
                                    context: format!("{}: tuple({}) -> {}",
                                        context_neuron,
                                        src_refs.iter().map(|r| format!("{}.{}", r.node, r.port)).collect::<Vec<_>>().join(", "),
                                        name
                                    ),
                                });
                            }
                        }
                    }
                    Endpoint::Tuple(dst_refs) => {
                        if src_refs.len() != dst_refs.len() {
                            errors.push(ValidationError::TupleArityMismatch {
                                expected: dst_refs.len(),
                                got: src_refs.len(),
                                context: format!("{}: tuple to tuple connection", context_neuron),
                            });
                        }
                    }
                    _ => {} // Other tuple connections not validated
                }
            }
            // Skip validation for intermediate port-to-port and call-to-intermediate connections
            _ => {}
        }

        errors
    }

    /// Get port information from an endpoint
    /// Returns (ports, description) tuple
    fn get_endpoint_port_info(
        endpoint: &Endpoint,
        context_neuron: &str,
        program: &Program,
        is_source: bool,
    ) -> (Option<Vec<Port>>, String) {
        match endpoint {
            Endpoint::Call { name, .. } => {
                if let Some(neuron) = program.neurons.get(name) {
                    let ports = if is_source { &neuron.outputs } else { &neuron.inputs };
                    (Some(ports.clone()), format!("neuron {}", name))
                } else {
                    (None, format!("unknown neuron {}", name))
                }
            }
            Endpoint::Ref(port_ref) => {
                // For port refs, we need to find the corresponding port in the context neuron
                let context_neuron_def = program.neurons.get(context_neuron).unwrap();
                let ports = if is_source { &context_neuron_def.inputs } else { &context_neuron_def.outputs };

                let matching_port = ports.iter().find(|p| p.name == port_ref.port);
                match matching_port {
                    Some(port) => (Some(vec![port.clone()]), format!("port {}.{}", port_ref.node, port_ref.port)),
                    None => (None, format!("unknown port {}.{}", port_ref.node, port_ref.port)),
                }
            }
            Endpoint::Tuple(port_refs) => {
                // Resolve each port in the tuple
                let context_neuron_def = program.neurons.get(context_neuron).unwrap();
                let ports = if is_source { &context_neuron_def.inputs } else { &context_neuron_def.outputs };

                let mut resolved_ports = Vec::new();
                for port_ref in port_refs {
                    if let Some(port) = ports.iter().find(|p| p.name == port_ref.port) {
                        resolved_ports.push(port.clone());
                    } else {
                        // Unknown port - return with error info
                        return (None, format!("tuple with unknown port {}.{}", port_ref.node, port_ref.port));
                    }
                }
                (Some(resolved_ports), format!("tuple({})", port_refs.iter().map(|r| format!("{}.{}", r.node, r.port)).collect::<Vec<_>>().join(", ")))
            }
            Endpoint::Match(_) => {
                // Match expressions have variable outputs depending on paths
                // For now, skip compatibility checking
                (None, "match expression".to_string())
            }
        }
    }

    /// Check if two ports are compatible
    fn ports_compatible(source: &Port, dest: &Port) -> Result<(), String> {
        // For now, simple shape equality check
        // In a full implementation, this would handle wildcard matching,
        // dimension computation, etc.
        if source.shape == dest.shape {
            Ok(())
        } else {
            Err(format!("shape mismatch: {} vs {}", source.shape, dest.shape))
        }
    }

    /// Get a description of a port for error messages
    fn port_desc(port: &Port) -> String {
        format!("{}: {}", port.name, port.shape)
    }

    /// Detect cycles in the connection graph
    fn detect_cycles(connections: &[Connection], context_neuron: &str) -> Vec<ValidationError> {
        // Build dependency graph: node -> set of dependencies
        let mut graph: HashMap<String, HashSet<String>> = HashMap::new();
        let mut all_nodes = HashSet::new();

        // Collect all node names
        for connection in connections {
            Self::collect_nodes_from_endpoint(&connection.source, &mut all_nodes);
            Self::collect_nodes_from_endpoint(&connection.destination, &mut all_nodes);
        }

        // Initialize graph
        for node in &all_nodes {
            graph.insert(node.clone(), HashSet::new());
        }

        // Add edges
        for connection in connections {
            let deps = Self::endpoint_dependencies(&connection.source);
            for source_node in deps {
                let dest_deps = Self::endpoint_dependencies(&connection.destination);
                for dest_node in dest_deps {
                    if let Some(node_deps) = graph.get_mut(&dest_node) {
                        node_deps.insert(source_node.clone());
                    }
                }
            }
        }

        // Detect cycles using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut errors = Vec::new();

        for node in &all_nodes {
            if !visited.contains(node) {
                if let Some(cycle) = Self::dfs_cycle_detect(
                    node, &graph, &mut visited, &mut rec_stack, vec![]
                ) {
                    errors.push(ValidationError::CycleDetected {
                        cycle,
                        context: format!("neuron {}", context_neuron),
                    });
                }
            }
        }

        errors
    }

    /// Recursively collect all unique node names from an endpoint
    /// Only includes actual neuron references, not internal port references
    fn collect_nodes_from_endpoint(endpoint: &Endpoint, nodes: &mut HashSet<String>) {
        match endpoint {
            Endpoint::Call { name, .. } => {
                nodes.insert(name.clone());
            }
            Endpoint::Ref(port_ref) => {
                // Only add "in" and "out" as they represent external connections
                // Don't add intermediate variables like "intermediate", "expanded", etc.
                if port_ref.node == "in" || port_ref.node == "out" {
                    nodes.insert(port_ref.node.clone());
                }
            }
            Endpoint::Tuple(refs) => {
                for port_ref in refs {
                    // Only add "in" and "out" references
                    if port_ref.node == "in" || port_ref.node == "out" {
                        nodes.insert(port_ref.node.clone());
                    }
                }
            }
            Endpoint::Match(match_expr) => {
                for arm in &match_expr.arms {
                    for ep in &arm.pipeline {
                        Self::collect_nodes_from_endpoint(ep, nodes);
                    }
                }
            }
        }
    }

    /// Get the node dependencies of an endpoint (what it depends on)
    fn endpoint_dependencies(endpoint: &Endpoint) -> HashSet<String> {
        let mut deps = HashSet::new();
        Self::collect_nodes_from_endpoint(endpoint, &mut deps);
        deps
    }

    /// DFS cycle detection
    fn dfs_cycle_detect(
        node: &str,
        graph: &HashMap<String, HashSet<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        mut current_path: Vec<String>,
    ) -> Option<Vec<String>> {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        current_path.push(node.to_string());

        if let Some(dependencies) = graph.get(node) {
            for neighbor in dependencies {
                if !visited.contains(neighbor) {
                    let cycle = Self::dfs_cycle_detect(
                        neighbor, graph, visited, rec_stack, current_path.clone()
                    );
                    if cycle.is_some() {
                        return cycle;
                    }
                } else if rec_stack.contains(neighbor) {
                    // Found a cycle - reconstruct the cycle path
                    let cycle_start = current_path.iter().position(|n| n == neighbor).unwrap();
                    let mut cycle = current_path[cycle_start..].to_vec();
                    cycle.push(neighbor.to_string()); // Close the cycle
                    return Some(cycle);
                }
            }
        }

        rec_stack.remove(node);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_missing_neuron() {
        let mut program = Program::new();
        program.neurons.insert("Existing".to_string(), NeuronDef {
            name: "Existing".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard]) }],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
        });

        let mut composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard]) }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call {
                    name: "Missing".to_string(),
                    args: vec![],
                    kwargs: vec![],
                },
                destination: Endpoint::Ref(PortRef::new("out")),
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            ValidationError::MissingNeuron { name, .. } => assert_eq!(name, "Missing"),
            _ => panic!("Expected MissingNeuron error"),
        }
    }

    #[test]
    fn test_cycle_detection() {
        let mut program = Program::new();

        // Create neurons for cycle A -> B -> C -> A
        let cycle_neurons = ["A", "B", "C"];
        for name in &cycle_neurons {
            program.neurons.insert(name.to_string(), NeuronDef {
                name: name.to_string(),
                params: vec![],
                inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard]) }],
                outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard]) }],
                body: NeuronBody::Primitive(ImplRef::Source {
                    source: "test".to_string(),
                    path: "test".to_string(),
                }),
            });
        }

        // Create a composite neuron with cycle
        let mut cycle_composite = NeuronDef {
            name: "CycleComposite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard]) }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![] },
                },
                Connection {
                    source: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![] },
                    destination: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![] },
                },
                Connection {
                    source: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![] },
                    destination: Endpoint::Call { name: "C".to_string(), args: vec![], kwargs: vec![] },
                },
                Connection {
                    source: Endpoint::Call { name: "C".to_string(), args: vec![], kwargs: vec![] },
                    destination: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![] },
                },
            ]),
        };
        program.neurons.insert("CycleComposite".to_string(), cycle_composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::CycleDetected { .. })));
    }
}
