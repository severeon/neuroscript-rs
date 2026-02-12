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

/// Context for partially resolving endpoints during symbol table building
pub(super) struct ResolutionContext<'a, F>
where
    F: Fn(&[Port], &[Param], &[Value]) -> Vec<Port>,
{
    pub neuron: &'a NeuronDef,
    pub program: &'a Program,
    pub registry: &'a StdlibRegistry,
    pub substitute_params_fn: F,
}

/// Build symbol table from connections, tracking intermediate nodes
pub(super) fn build_symbol_table<F>(
    neuron: &NeuronDef,
    connections: &[Connection],
    program: &Program,
    registry: &StdlibRegistry,
    errors: &mut Vec<ValidationError>,
    substitute_params_fn: F,
) -> SymbolTable
where
    F: Fn(&[Port], &[Param], &[Value]) -> Vec<Port>,
{
    let mut table = SymbolTable::new();

    // Add input ports as nodes
    // - Single unnamed ("default") port → register as "in"
    // - Single variadic port → register as "in" (carries whole tuple; validation
    //   in core.rs guarantees variadic ports always have an explicit name != "default",
    //   so the two branches here are mutually exclusive)
    // - Otherwise, register each named port separately
    if neuron.inputs.len() == 1
        && (neuron.inputs[0].name == "default" || neuron.inputs[0].variadic)
    {
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

    let ctx = ResolutionContext {
        neuron,
        program,
        registry,
        substitute_params_fn,
    };

    // Add context bindings as nodes
    if let NeuronBody::Graph {
        context_bindings, ..
    } = &neuron.body
    {
        // Process unified context bindings
        for binding in context_bindings {
            let endpoint = Endpoint::Call {
                name: binding.call_name.clone(),
                args: binding.args.clone(),
                kwargs: binding.kwargs.clone(),
                id: 0,
                frozen: binding.frozen,
            };
            if let Some(ports) = resolve_endpoint_partial(&endpoint, &ctx, &table, true, errors) {
                table.add_node(binding.name.clone(), ports);
            }
        }
    }

    // Scan connections for intermediate node creation
    for connection in connections {
        process_destination_for_symbol_table(
            &connection.destination,
            &connection.source,
            &ctx,
            &mut table,
            errors,
        );
    }

    table
}

/// Helper to recursively process destinations for symbol table building
fn process_destination_for_symbol_table<F>(
    dest: &Endpoint,
    source: &Endpoint,
    ctx: &ResolutionContext<F>,
    table: &mut SymbolTable,
    errors: &mut Vec<ValidationError>,
) where
    F: Fn(&[Port], &[Param], &[Value]) -> Vec<Port>,
{
    match dest {
        // Tuple unpacking: source -> (a, b, c)
        Endpoint::Tuple(port_refs) => {
            // Source must resolve to ports
            if let Some(source_ports) = resolve_endpoint_partial(source, ctx, table, true, errors) {
                if source_ports.len() == 1 && port_refs.len() > 1 {
                    // Implicit fork: single source replicates to all tuple bindings
                    let single_port = &source_ports[0];
                    for port_ref in port_refs {
                        table.add_node(port_ref.node.clone(), vec![single_port.clone()]);
                    }
                } else if source_ports.len() != port_refs.len() {
                    errors.push(ValidationError::ArityMismatch {
                        expected: port_refs.len(),
                        got: source_ports.len(),
                        context: format!("{}: tuple unpacking", ctx.neuron.name),
                    });
                } else {
                    // Add each unpacked reference as a single-port node
                    for (port_ref, port) in port_refs.iter().zip(source_ports.iter()) {
                        table.add_node(port_ref.node.clone(), vec![port.clone()]);
                    }
                }
            }
        }
        // Single intermediate node: source -> intermediate
        Endpoint::Ref(port_ref) if port_ref.node != "in" && port_ref.node != "out" => {
            // This creates an intermediate node if it's not already in the table
            if table.get_ports(&port_ref.node).is_none() {
                if let Some(source_ports) =
                    resolve_endpoint_partial(source, ctx, table, true, errors)
                {
                    // Add the intermediate node with the source's output ports
                    table.add_node(port_ref.node.clone(), source_ports);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                let mut current_source = source.clone();
                for ep in &branch.pipeline {
                    process_destination_for_symbol_table(ep, &current_source, ctx, table, errors);
                    current_source = ep.clone();
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                let mut current_source = source.clone();
                for ep in else_branch {
                    process_destination_for_symbol_table(ep, &current_source, ctx, table, errors);
                    current_source = ep.clone();
                }
            }
        }
        Endpoint::Match(match_expr) => {
            for arm in &match_expr.arms {
                let mut current_source = source.clone();
                for ep in &arm.pipeline {
                    process_destination_for_symbol_table(ep, &current_source, ctx, table, errors);
                    current_source = ep.clone();
                }
            }
        }
        _ => {
            // Not an intermediate node creation
        }
    }
}

/// Partially resolve endpoint (used during symbol table building)
/// Returns None if resolution fails (errors added to errors vec)
pub(super) fn resolve_endpoint_partial<F>(
    endpoint: &Endpoint,
    ctx: &ResolutionContext<F>,
    table: &SymbolTable,
    is_source: bool,
    errors: &mut Vec<ValidationError>,
) -> Option<Vec<Port>>
where
    F: Fn(&[Port], &[Param], &[Value]) -> Vec<Port>,
{
    match resolve_endpoint(endpoint, ctx, table, is_source) {
        Ok(ports) => Some(ports),
        Err(e) => {
            errors.push(*e);
            None
        }
    }
}

/// Resolve an endpoint to a vector of ports
pub(super) fn resolve_endpoint<F>(
    endpoint: &Endpoint,
    ctx: &ResolutionContext<F>,
    symbol_table: &SymbolTable,
    is_source: bool,
) -> Result<Vec<Port>, Box<ValidationError>>
where
    F: Fn(&[Port], &[Param], &[Value]) -> Vec<Port>,
{
    match endpoint {
        Endpoint::Call { name, args, .. } => {
            // 1. Look up neuron definition
            if let Some(called_neuron) = ctx.program.neurons.get(name) {
                let ports = if is_source {
                    &called_neuron.outputs
                } else {
                    &called_neuron.inputs
                };

                // Substitute parameters in port shapes
                let substituted_ports =
                    (ctx.substitute_params_fn)(ports, &called_neuron.params, args);
                return Ok(substituted_ports);
            }

            // 2. Look up global names
            if let Some(global) = ctx.program.globals.iter().find(|g| &g.name == name) {
                match &global.value {
                    Value::Call {
                        name: c_name,
                        args: c_args,
                        kwargs: _,
                    } => {
                        // Nested call resolution for global neurons
                        let sub_endpoint = Endpoint::Call {
                            name: c_name.clone(),
                            args: c_args.clone(),
                            kwargs: vec![],
                            id: 0,
                            frozen: false,
                        };
                        return resolve_endpoint(&sub_endpoint, ctx, symbol_table, is_source);
                    }
                    _ => {
                        // For simple global values (int, float, etc.),
                        // they don't have ports. This should probably error
                        // if used as a neuron call.
                        return Err(Box::new(ValidationError::Custom(format!(
                            "Global name '{}' is a value, not a neuron",
                            name
                        ))));
                    }
                }
            }

            // 3. Look up registry
            if ctx.registry.contains(name) {
                // Primitive neuron - skip detailed port validation
                // Primitives are validated at codegen time
                // Return a dummy port to allow validation to continue
                Ok(vec![Port {
                    name: "default".to_string(),
                    shape: Shape { dims: vec![] },
                    variadic: false,
                }])
            } else {
                Err(Box::new(ValidationError::MissingNeuron {
                    name: name.clone(),
                    context: ctx.neuron.name.clone(),
                }))
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
                    Err(Box::new(ValidationError::UnknownNode {
                        node: format!("{}.{}", port_ref.node, port_ref.port),
                        context: ctx.neuron.name.clone(),
                    }))
                }
            } else {
                Err(Box::new(ValidationError::UnknownNode {
                    node: port_ref.node.clone(),
                    context: ctx.neuron.name.clone(),
                }))
            }
        }
        Endpoint::Tuple(port_refs) => {
            // Resolve each port reference
            let mut ports = Vec::new();
            for port_ref in port_refs {
                let resolved = resolve_endpoint(
                    &Endpoint::Ref(port_ref.clone()),
                    ctx,
                    symbol_table,
                    is_source,
                )?;
                ports.extend(resolved);
            }
            Ok(ports)
        }
        Endpoint::Match(match_expr) => {
            if match_expr.arms.is_empty() {
                return Ok(vec![]);
            }

            let mut all_arm_ports = Vec::new();
            for arm in &match_expr.arms {
                let ep = if is_source {
                    arm.pipeline.last()
                } else {
                    arm.pipeline.first()
                };
                if let Some(ep) = ep {
                    all_arm_ports.push(resolve_endpoint(ep, ctx, symbol_table, is_source)?);
                } else {
                    all_arm_ports.push(vec![]);
                }
            }

            // Return ports from first arm (they should be consistent in number)
            Ok(all_arm_ports[0].clone())
        }
        Endpoint::If(if_expr) => {
            if if_expr.branches.is_empty() {
                return Ok(vec![]);
            }

            let mut all_branch_ports = Vec::new();
            for branch in &if_expr.branches {
                let ep = if is_source {
                    branch.pipeline.last()
                } else {
                    branch.pipeline.first()
                };
                if let Some(ep) = ep {
                    all_branch_ports.push(resolve_endpoint(ep, ctx, symbol_table, is_source)?);
                } else {
                    all_branch_ports.push(vec![]);
                }
            }

            if let Some(else_branch) = &if_expr.else_branch {
                let ep = if is_source {
                    else_branch.last()
                } else {
                    else_branch.first()
                };
                if let Some(ep) = ep {
                    all_branch_ports.push(resolve_endpoint(ep, ctx, symbol_table, is_source)?);
                } else {
                    all_branch_ports.push(vec![]);
                }
            }

            Ok(all_branch_ports[0].clone())
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

    // For Match and If, we check consistency but typically they are destinations
    // and shape inference does the heavy lifting.
    // We'll allow them through but we might need more specific checks later.
    /*
    if matches!(source_endpoint, Endpoint::Match(_))
        || matches!(dest_endpoint, Endpoint::Match(_))
        || matches!(source_endpoint, Endpoint::If(_))
        || matches!(dest_endpoint, Endpoint::If(_))
    {
        return errors;
    }
    */

    // Variadic input port: accept any number of source ports.
    // This check must precede the implicit fork check below, because a tuple
    // source like (a, b, c) flowing into a variadic neuron should match here
    // (N→1 variadic), not fall through to the arity mismatch path.
    if dest_ports.len() == 1 && dest_ports[0].variadic {
        // Validate each source port's shape against the variadic port's shape individually
        for src_port in source_ports {
            if !shapes_compatible_fn(&src_port.shape, &dest_ports[0].shape) {
                errors.push(ValidationError::PortMismatch {
                    source_node: extract_node_name(source_endpoint),
                    source_port: src_port.name.clone(),
                    source_shape: src_port.shape.clone(),
                    dest_node: extract_node_name(dest_endpoint),
                    dest_port: dest_ports[0].name.clone(),
                    dest_shape: dest_ports[0].shape.clone(),
                    context: context_neuron.to_string(),
                });
            }
        }
        return errors;
    }

    // Check arity - allow implicit fork (1→N) for tuple destinations
    if source_ports.len() == 1 && dest_ports.len() > 1 {
        if let Endpoint::Tuple(_) = dest_endpoint {
            // Implicit fork: validate single source shape against all destinations
            for dst_port in dest_ports {
                if !shapes_compatible_fn(&source_ports[0].shape, &dst_port.shape) {
                    errors.push(ValidationError::PortMismatch {
                        source_node: extract_node_name(source_endpoint),
                        source_port: source_ports[0].name.clone(),
                        source_shape: source_ports[0].shape.clone(),
                        dest_node: extract_node_name(dest_endpoint),
                        dest_port: dst_port.name.clone(),
                        dest_shape: dst_port.shape.clone(),
                        context: context_neuron.to_string(),
                    });
                }
            }
            return errors;
        }
    }

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
        Endpoint::If(_) => "If".to_string(),
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
        Endpoint::If(_) => "if".to_string(),
    }
}
