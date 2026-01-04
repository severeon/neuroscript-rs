use crate::interfaces::*;
use std::collections::HashMap;

/// Symbol table tracking intermediate nodes in a composite neuron graph
/// Maps node names to their resolved port signatures
#[derive(Debug, Clone)]
pub(super) struct SymbolTable {
    /// Map from node name -> ports
    nodes: HashMap<String, Vec<Port>>,
}

impl SymbolTable {
    pub(super) fn new() -> Self {
        SymbolTable {
            nodes: HashMap::new(),
        }
    }

    /// Add a node with its ports
    pub(super) fn add_node(&mut self, name: String, ports: Vec<Port>) {
        self.nodes.insert(name, ports);
    }

    /// Get ports for a node
    pub(super) fn get_ports(&self, name: &str) -> Option<&Vec<Port>> {
        self.nodes.get(name)
    }

    /// Get all node names in the symbol table
    pub(super) fn node_names(&self) -> impl Iterator<Item = &String> {
        self.nodes.keys()
    }
}

/// Build symbol table from connections, tracking intermediate nodes
pub(super) fn build_symbol_table(
    neuron: &NeuronDef,
    connections: &[Connection],
    program: &Program,
    registry: &StdlibRegistry,
    errors: &mut Vec<ValidationError>,
    substitute_params_fn: impl Fn(&[Port], &[Param], &[Value]) -> Vec<Port>,
) -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add input ports as nodes
    // - If only one input and it's named "default", add as "in"
    // - Otherwise, add each named input port as a separate node
    if neuron.inputs.len() == 1 && neuron.inputs[0].name == "default" {
        table.add_node("in".to_string(), neuron.inputs.clone());
    } else {
        // Add each named input port as a separate node
        for input_port in &neuron.inputs {
            table.add_node(input_port.name.clone(), vec![input_port.clone()]);
        }
        // Also add "in" for backward compatibility if there's a default port
        if let Some(default_port) = neuron.inputs.iter().find(|p| p.name == "default") {
            table.add_node("in".to_string(), vec![default_port.clone()]);
        }
    }

    // Add output ports as nodes
    // - If only one output and it's named "default", add as "out"
    // - Otherwise, add each named output port as a separate node
    if neuron.outputs.len() == 1 && neuron.outputs[0].name == "default" {
        table.add_node("out".to_string(), neuron.outputs.clone());
    } else {
        // Add each named output port as a separate node
        for output_port in &neuron.outputs {
            table.add_node(output_port.name.clone(), vec![output_port.clone()]);
        }
        // Also add "out" for backward compatibility if there's a default port
        if let Some(default_port) = neuron.outputs.iter().find(|p| p.name == "default") {
            table.add_node("out".to_string(), vec![default_port.clone()]);
        }
    }

    // Scan connections for intermediate node creation
    for connection in connections {
        match &connection.destination {
            // Tuple unpacking: source -> (a, b, c)
            Endpoint::Tuple(port_refs) => {
                // Source must resolve to multiple ports
                match resolve_endpoint_partial(
                    &connection.source,
                    neuron,
                    &table,
                    program,
                    registry,
                    true,
                    errors,
                    &substitute_params_fn,
                ) {
                    Some(source_ports) => {
                        if source_ports.len() != port_refs.len() {
                            errors.push(ValidationError::ArityMismatch {
                                expected: port_refs.len(),
                                got: source_ports.len(),
                                context: format!("{}:  tuple unpacking", neuron.name),
                            });
                        } else {
                            // Add each unpacked reference as a single-port node
                            for (port_ref, port) in port_refs.iter().zip(source_ports.iter()) {
                                table.add_node(port_ref.node.clone(), vec![port.clone()]);
                            }
                        }
                    }
                    None => {
                        // Error already added by resolve_endpoint_partial
                    }
                }
            }
            // Single intermediate node: source -> intermediate
            Endpoint::Ref(port_ref) if port_ref.node != "in" && port_ref.node != "out" => {
                // This creates an intermediate node
                match resolve_endpoint_partial(
                    &connection.source,
                    neuron,
                    &table,
                    program,
                    registry,
                    true,
                    errors,
                    &substitute_params_fn,
                ) {
                    Some(source_ports) => {
                        // Add the intermediate node with the source's output ports
                        table.add_node(port_ref.node.clone(), source_ports);
                    }
                    None => {
                        // Error already added by resolve_endpoint_partial
                    }
                }
            }
            _ => {
                // Not an intermediate node creation
            }
        }
    }

    table
}

/// Partially resolve endpoint (used during symbol table building)
/// Returns None if resolution fails (errors added to errors vec)
pub(super) fn resolve_endpoint_partial(
    endpoint: &Endpoint,
    neuron: &NeuronDef,
    table: &SymbolTable,
    program: &Program,
    registry: &StdlibRegistry,
    is_source: bool,
    errors: &mut Vec<ValidationError>,
    substitute_params_fn: &impl Fn(&[Port], &[Param], &[Value]) -> Vec<Port>,
) -> Option<Vec<Port>> {
    match resolve_endpoint(
        endpoint,
        neuron,
        table,
        program,
        registry,
        is_source,
        substitute_params_fn,
    ) {
        Ok(ports) => Some(ports),
        Err(e) => {
            errors.push(e);
            None
        }
    }
}

/// Resolve an endpoint to a vector of ports
pub(super) fn resolve_endpoint(
    endpoint: &Endpoint,
    neuron: &NeuronDef,
    symbol_table: &SymbolTable,
    program: &Program,
    registry: &StdlibRegistry,
    is_source: bool,
    substitute_params_fn: &impl Fn(&[Port], &[Param], &[Value]) -> Vec<Port>,
) -> Result<Vec<Port>, ValidationError> {
    match endpoint {
        Endpoint::Call { name, args, .. } => {
            // Look up neuron definition
            if let Some(called_neuron) = program.neurons.get(name) {
                let ports = if is_source {
                    &called_neuron.outputs
                } else {
                    &called_neuron.inputs
                };

                // Substitute parameters in port shapes
                let substituted_ports = substitute_params_fn(ports, &called_neuron.params, args);
                Ok(substituted_ports)
            } else if registry.contains(name) {
                // Primitive neuron - skip detailed port validation
                // Primitives are validated at codegen time
                // Return a dummy port to allow validation to continue
                Ok(vec![Port {
                    name: "default".to_string(),
                    shape: Shape { dims: vec![] },
                }])
            } else {
                Err(ValidationError::MissingNeuron {
                    name: name.clone(),
                    context: neuron.name.clone(),
                })
            }
        }
        Endpoint::Ref(port_ref) => {
            // Check symbol table first (for intermediate nodes)
            if let Some(ports) = symbol_table.get_ports(&port_ref.node) {
                // Find the specific port
                if port_ref.port == "default" && ports.len() == 1 {
                    Ok(vec![ports[0].clone()])
                } else if let Some(port) = ports.iter().find(|p| p.name == port_ref.port) {
                    Ok(vec![port.clone()])
                } else {
                    Err(ValidationError::UnknownNode {
                        node: format!("{}.{}", port_ref.node, port_ref.port),
                        context: neuron.name.clone(),
                    })
                }
            } else {
                Err(ValidationError::UnknownNode {
                    node: port_ref.node.clone(),
                    context: neuron.name.clone(),
                })
            }
        }
        Endpoint::Tuple(port_refs) => {
            // Resolve each port reference
            let mut ports = Vec::new();
            for port_ref in port_refs {
                let resolved = resolve_endpoint(
                    &Endpoint::Ref(port_ref.clone()),
                    neuron,
                    symbol_table,
                    program,
                    registry,
                    is_source,
                    substitute_params_fn,
                )?;
                ports.extend(resolved);
            }
            Ok(ports)
        }
        Endpoint::Match(_match_expr) => {
            // Match expressions are complex - conservatively skip for now
            // A full implementation would need to handle all possible arms
            Ok(vec![])
        }
    }
}

/// Check if two port lists are compatible
pub(super) fn check_port_compatibility(
    source_ports: &[Port],
    dest_ports: &[Port],
    source_endpoint: &Endpoint,
    dest_endpoint: &Endpoint,
    context_neuron: &str,
    shapes_compatible_fn: impl Fn(&Shape, &Shape) -> bool,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Skip Match endpoints
    if matches!(source_endpoint, Endpoint::Match(_))
        || matches!(dest_endpoint, Endpoint::Match(_))
    {
        return errors;
    }

    // Check arity
    if source_ports.len() != dest_ports.len() {
        errors.push(ValidationError::ArityMismatch {
            expected: dest_ports.len(),
            got: source_ports.len(),
            context: format!(
                "{}: {} -> {}",
                context_neuron,
                endpoint_desc(source_endpoint),
                endpoint_desc(dest_endpoint)
            ),
        });
        return errors;
    }

    // Check each port pair
    for (src_port, dst_port) in source_ports.iter().zip(dest_ports.iter()) {
        // Check shapes (exact match for now - wildcards/exprs need inference)
        if !shapes_compatible_fn(&src_port.shape, &dst_port.shape) {
            errors.push(ValidationError::PortMismatch {
                source_node: extract_node_name(source_endpoint),
                source_port: src_port.name.clone(),
                source_shape: src_port.shape.clone(),
                dest_node: extract_node_name(dest_endpoint),
                dest_port: dst_port.name.clone(),
                dest_shape: dst_port.shape.clone(),
                context: context_neuron.to_string(),
            });
        }
    }

    errors
}

/// Extract node name from endpoint for error messages
pub(super) fn extract_node_name(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Call { name, .. } => name.clone(),
        Endpoint::Ref(port_ref) => port_ref.node.clone(),
        Endpoint::Tuple(_) => "Tuple".to_string(), // Simplification for now
        Endpoint::Match(_) => "Match".to_string(),
    }
}

/// Get a description of an endpoint for error messages
pub(super) fn endpoint_desc(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Call { name, .. } => name.clone(),
        Endpoint::Ref(port_ref) => {
            if port_ref.port == "default" {
                port_ref.node.clone()
            } else {
                format!("{}.{}", port_ref.node, port_ref.port)
            }
        }
        Endpoint::Tuple(refs) => {
            format!(
                "({})",
                refs.iter()
                    .map(|r| if r.port == "default" {
                        r.node.clone()
                    } else {
                        format!("{}.{}", r.node, r.port)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Endpoint::Match(_) => "match".to_string(),
    }
}
