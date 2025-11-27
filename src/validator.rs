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
    },
    PortMismatch {
        source_ports: String,
        dest_ports: String,
        context: String,
        details: String,
    },
    CycleDetected {
        cycle: Vec<String>,
        context: String,
    },
    ArityMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
    UnknownNode {
        node: String,
        context: String,
    },
    Custom(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingNeuron { name, context } => {
                write!(f, "Neuron '{}' not found (in {})", name, context)
            }
            ValidationError::PortMismatch { source_ports, dest_ports, context, details } => {
                write!(f, "Port mismatch: {} -> {} (in {}: {})",
                       source_ports, dest_ports, context, details)
            }
            ValidationError::CycleDetected { cycle, context } => {
                write!(f, "Cycle detected in {}: {}", context, cycle.join(" -> "))
            }
            ValidationError::ArityMismatch { expected, got, context } => {
                write!(f, "Arity mismatch: expected {} ports, got {} (in {})",
                       expected, got, context)
            }
            ValidationError::UnknownNode { node, context } => {
                write!(f, "Unknown node '{}' (in {})", node, context)
            }
            ValidationError::Custom(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}

/// Symbol table tracking intermediate nodes in a composite neuron graph
/// Maps node names to their resolved port signatures
#[derive(Debug, Clone)]
struct SymbolTable {
    /// Map from node name -> ports
    nodes: HashMap<String, Vec<Port>>,
}

impl SymbolTable {
    fn new() -> Self {
        SymbolTable {
            nodes: HashMap::new(),
        }
    }

    /// Add a node with its ports
    fn add_node(&mut self, name: String, ports: Vec<Port>) {
        self.nodes.insert(name, ports);
    }

    /// Get ports for a node
    fn get_ports(&self, name: &str) -> Option<&Vec<Port>> {
        self.nodes.get(name)
    }

}

/// Graph validator
pub struct Validator;

impl Validator {
    /// Validate an entire program
    pub fn validate(program: &Program) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check each neuron
        for (_neuron_name, neuron) in &program.neurons {
            // Validate connections within this neuron if it's composite
            if let NeuronBody::Graph(connections) = &neuron.body {
                errors.extend(Self::validate_neuron_graph(
                    neuron, connections, program
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate a single neuron's graph
    fn validate_neuron_graph(
        neuron: &NeuronDef,
        connections: &[Connection],
        program: &Program,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Build symbol table for this neuron's graph
        let symbol_table = Self::build_symbol_table(neuron, connections, program, &mut errors);

        // Validate each connection
        for connection in connections {
            // Check that neurons exist
            errors.extend(Self::check_neurons_exist(&connection.source, &neuron.name, program));
            errors.extend(Self::check_neurons_exist(&connection.destination, &neuron.name, program));

            // Resolve source and destination endpoints
            let source_resolution = Self::resolve_endpoint(
                &connection.source,
                neuron,
                &symbol_table,
                program,
                true, // is_source
            );
            let dest_resolution = Self::resolve_endpoint(
                &connection.destination,
                neuron,
                &symbol_table,
                program,
                false, // is_source
            );

            match (source_resolution, dest_resolution) {
                (Ok(source_ports), Ok(dest_ports)) => {
                    // Check compatibility
                    errors.extend(Self::check_port_compatibility(
                        &source_ports,
                        &dest_ports,
                        &connection.source,
                        &connection.destination,
                        &neuron.name,
                    ));
                }
                (Err(e), _) | (_, Err(e)) => {
                    errors.push(e);
                }
            }
        }

        // Check for cycles
        errors.extend(Self::detect_cycles(connections, &neuron.name, &symbol_table, program));

        errors
    }

    /// Build symbol table by scanning connections for tuple unpackings
    /// and neuron calls that create intermediate nodes
    fn build_symbol_table(
        neuron: &NeuronDef,
        connections: &[Connection],
        program: &Program,
        errors: &mut Vec<ValidationError>,
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
                    match Self::resolve_endpoint_partial(&connection.source, neuron, &table, program, true, errors) {
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
                    match Self::resolve_endpoint_partial(&connection.source, neuron, &table, program, true, errors) {
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
    fn resolve_endpoint_partial(
        endpoint: &Endpoint,
        neuron: &NeuronDef,
        table: &SymbolTable,
        program: &Program,
        is_source: bool,
        errors: &mut Vec<ValidationError>,
    ) -> Option<Vec<Port>> {
        match Self::resolve_endpoint(endpoint, neuron, table, program, is_source) {
            Ok(ports) => Some(ports),
            Err(e) => {
                errors.push(e);
                None
            }
        }
    }

    /// Resolve an endpoint to a vector of ports
    fn resolve_endpoint(
        endpoint: &Endpoint,
        neuron: &NeuronDef,
        symbol_table: &SymbolTable,
        program: &Program,
        is_source: bool,
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
                    let substituted_ports = Self::substitute_params(ports, &called_neuron.params, args);
                    Ok(substituted_ports)
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
                    let resolved = Self::resolve_endpoint(
                        &Endpoint::Ref(port_ref.clone()),
                        neuron,
                        symbol_table,
                        program,
                        is_source,
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
    fn check_port_compatibility(
        source_ports: &[Port],
        dest_ports: &[Port],
        source_endpoint: &Endpoint,
        dest_endpoint: &Endpoint,
        context_neuron: &str,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Skip Match endpoints
        if matches!(source_endpoint, Endpoint::Match(_)) || matches!(dest_endpoint, Endpoint::Match(_)) {
            return errors;
        }

        // Check arity
        if source_ports.len() != dest_ports.len() {
            errors.push(ValidationError::ArityMismatch {
                expected: dest_ports.len(),
                got: source_ports.len(),
                context: format!("{}: {} -> {}",
                    context_neuron,
                    Self::endpoint_desc(source_endpoint),
                    Self::endpoint_desc(dest_endpoint)
                ),
            });
            return errors;
        }

        // Check each port pair
        for (src_port, dst_port) in source_ports.iter().zip(dest_ports.iter()) {
            // Check shapes (exact match for now - wildcards/exprs need inference)
            if !Self::shapes_compatible(&src_port.shape, &dst_port.shape) {
                errors.push(ValidationError::PortMismatch {
                    source_ports: Self::port_desc(src_port),
                    dest_ports: Self::port_desc(dst_port),
                    context: context_neuron.to_string(),
                    details: format!("shape mismatch: {} vs {}", src_port.shape, dst_port.shape),
                });
            }
        }

        errors
    }

    /// Check if two shapes are compatible
    fn shapes_compatible(source: &Shape, dest: &Shape) -> bool {
        // Find variadic dimensions in both shapes
        let source_variadic_pos = source.dims.iter().position(|d| matches!(d, Dim::Variadic(_)));
        let dest_variadic_pos = dest.dims.iter().position(|d| matches!(d, Dim::Variadic(_)));

        match (source_variadic_pos, dest_variadic_pos) {
            // Both have variadics - complex case, for now assume compatible
            (Some(_), Some(_)) => {
                // Would need sophisticated matching, for now allow it
                true
            }
            // Source has variadic, dest does not
            (Some(var_pos), None) => {
                Self::match_variadic_shape(&source.dims, var_pos, &dest.dims)
            }
            // Dest has variadic, source does not
            (None, Some(var_pos)) => {
                Self::match_variadic_shape(&dest.dims, var_pos, &source.dims)
            }
            // Neither has variadic - must match exactly
            (None, None) => {
                if source.dims.len() != dest.dims.len() {
                    return false;
                }
                // Check each dimension pair
                for (src_dim, dst_dim) in source.dims.iter().zip(dest.dims.iter()) {
                    if !Self::dims_compatible(src_dim, dst_dim) {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Substitute parameter values in port shapes
    /// For example: Linear(512, 256) binds in_dim=512, out_dim=256
    /// Then [*, in_dim] becomes [*, 512]
    fn substitute_params(ports: &[Port], params: &[Param], args: &[Value]) -> Vec<Port> {
        // Build parameter binding map
        let mut bindings: HashMap<String, i64> = HashMap::new();
        for (param, arg) in params.iter().zip(args.iter()) {
            if let Value::Int(val) = arg {
                bindings.insert(param.name.clone(), *val);
            } else if let Value::Name(name) = arg {
                // Named arguments remain as named dimensions
                // We could handle this better with a more sophisticated type system
                // For now, we don't substitute named parameters
                continue;
            }
        }

        // Substitute in each port
        ports.iter().map(|port| {
            Port {
                name: port.name.clone(),
                shape: Self::substitute_shape(&port.shape, &bindings),
            }
        }).collect()
    }

    /// Substitute parameter values in a shape
    fn substitute_shape(shape: &Shape, bindings: &HashMap<String, i64>) -> Shape {
        Shape {
            dims: shape.dims.iter().map(|dim| Self::substitute_dim(dim, bindings)).collect()
        }
    }

    /// Substitute parameter values in a dimension
    fn substitute_dim(dim: &Dim, bindings: &HashMap<String, i64>) -> Dim {
        match dim {
            Dim::Named(name) => {
                if let Some(val) = bindings.get(name) {
                    Dim::Literal(*val)
                } else {
                    dim.clone()
                }
            }
            Dim::Expr(expr) => {
                // Recursively substitute in expressions
                let left = Self::substitute_dim(&expr.left, bindings);
                let right = Self::substitute_dim(&expr.right, bindings);

                // Try to evaluate if both sides are now literals
                if let (Dim::Literal(l), Dim::Literal(r)) = (&left, &right) {
                    let result = match expr.op {
                        BinOp::Add => l + r,
                        BinOp::Sub => l - r,
                        BinOp::Mul => l * r,
                        BinOp::Div => l / r,
                        _ => {
                            // Non-arithmetic operations, keep as expression
                            return Dim::Expr(Box::new(DimExpr {
                                op: expr.op,
                                left,
                                right,
                            }));
                        }
                    };
                    Dim::Literal(result)
                } else {
                    Dim::Expr(Box::new(DimExpr {
                        op: expr.op,
                        left,
                        right,
                    }))
                }
            }
            _ => dim.clone(),
        }
    }

    /// Match a shape with a variadic against a concrete shape
    /// pattern_dims: the dims with a variadic at var_pos
    /// concrete_dims: the dims without variadic
    fn match_variadic_shape(pattern_dims: &[Dim], var_pos: usize, concrete_dims: &[Dim]) -> bool {
        // Pattern: [prefix..., *variadic, suffix...]
        // Concrete: [concrete_dims...]

        // Count non-variadic dimensions in pattern
        let pattern_fixed_count = pattern_dims.len() - 1; // Subtract 1 for the variadic

        // The variadic must match at least 0 dimensions
        // So concrete must have at least as many dims as pattern's fixed dims
        if concrete_dims.len() < pattern_fixed_count {
            return false;
        }

        // Match prefix (before variadic)
        for i in 0..var_pos {
            if !Self::dims_compatible(&pattern_dims[i], &concrete_dims[i]) {
                return false;
            }
        }

        // Match suffix (after variadic)
        let suffix_count = pattern_dims.len() - var_pos - 1;
        let concrete_suffix_start = concrete_dims.len() - suffix_count;

        for i in 0..suffix_count {
            let pattern_idx = var_pos + 1 + i;
            let concrete_idx = concrete_suffix_start + i;
            if !Self::dims_compatible(&pattern_dims[pattern_idx], &concrete_dims[concrete_idx]) {
                return false;
            }
        }

        // Everything matches - the variadic captures the middle portion
        true
    }

    /// Check if two dimensions are compatible
    fn dims_compatible(source: &Dim, dest: &Dim) -> bool {
        match (source, dest) {
            // Wildcards match anything
            (Dim::Wildcard, _) | (_, Dim::Wildcard) => true,
            // Variadics match anything (shouldn't reach here due to check above, but safe)
            (Dim::Variadic(_), _) | (_, Dim::Variadic(_)) => true,
            // Exact matches
            (Dim::Literal(a), Dim::Literal(b)) => a == b,
            // Named dimensions: assume compatible (parameter binding handles unification)
            // Full shape inference would need parameter context
            (Dim::Named(_), Dim::Named(_)) => true,
            // Expressions: would need evaluation context, for now assume compatible
            (Dim::Expr(_), _) | (_, Dim::Expr(_)) => true,
            // Mixed named/literal: incompatible (can't unify 512 with a variable)
            (Dim::Named(_), Dim::Literal(_)) | (Dim::Literal(_), Dim::Named(_)) => false,
        }
    }

    /// Get a description of a port for error messages
    fn port_desc(port: &Port) -> String {
        if port.name == "default" {
            format!("{}", port.shape)
        } else {
            format!("{}: {}", port.name, port.shape)
        }
    }

    /// Get a description of an endpoint for error messages
    fn endpoint_desc(endpoint: &Endpoint) -> String {
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
                format!("({})", refs.iter()
                    .map(|r| if r.port == "default" {
                        r.node.clone()
                    } else {
                        format!("{}.{}", r.node, r.port)
                    })
                    .collect::<Vec<_>>()
                    .join(", "))
            }
            Endpoint::Match(_) => "match".to_string(),
        }
    }

    /// Check that all neurons referenced in an endpoint exist
    fn check_neurons_exist(
        endpoint: &Endpoint,
        context_neuron: &str,
        program: &Program,
    ) -> Vec<ValidationError> {
        match endpoint {
            Endpoint::Call { name, .. } => {
                if !program.neurons.contains_key(name) {
                    vec![ValidationError::MissingNeuron {
                        name: name.clone(),
                        context: context_neuron.to_string(),
                    }]
                } else {
                    vec![]
                }
            }
            Endpoint::Match(match_expr) => {
                match_expr.arms.iter()
                    .flat_map(|arm| {
                        arm.pipeline.iter()
                            .flat_map(|ep| Self::check_neurons_exist(ep, context_neuron, program))
                    })
                    .collect()
            }
            _ => vec![],
        }
    }

    /// Detect cycles in the connection graph
    /// Uses neuron names + symbol table nodes, allows self-edges within a single connection
    fn detect_cycles(
        connections: &[Connection],
        context_neuron: &str,
        symbol_table: &SymbolTable,
        _program: &Program,
    ) -> Vec<ValidationError> {
        // Build dependency graph: which nodes flow to which others
        let mut graph: HashMap<String, HashSet<String>> = HashMap::new();

        // Add symbol table nodes
        for node in symbol_table.nodes.keys() {
            graph.insert(node.clone(), HashSet::new());
        }

        // Track Call destinations: each destination creates a new unique instance
        // Track Call sources: sources map to their most recent destination instance
        let mut call_last_instance: HashMap<String, String> = HashMap::new();
        let mut call_instance_counter: HashMap<String, usize> = HashMap::new();

        // Build edges from connections
        for connection in connections {
            // Extract source nodes - these reference existing instances
            let source_nodes = Self::extract_node_names_from_sources(
                &connection.source,
                &mut call_last_instance,
                &mut call_instance_counter,
            );

            // Extract destination nodes - these CREATE new instances
            let dest_nodes = Self::extract_node_names_from_destinations(
                &connection.destination,
                &mut call_instance_counter,
                &mut call_last_instance,
            );

            // Add nodes
            for node in &source_nodes {
                graph.entry(node.clone()).or_insert_with(HashSet::new);
            }
            for node in &dest_nodes {
                graph.entry(node.clone()).or_insert_with(HashSet::new);
            }

            // Add edges, but skip self-edges within the same connection
            // (e.g., Linear -> Linear in same connection is OK for pipeline)
            for src in &source_nodes {
                for dst in &dest_nodes {
                    if src != dst || source_nodes.len() > 1 || dest_nodes.len() > 1 {
                        graph.get_mut(src).unwrap().insert(dst.clone());
                    }
                }
            }
        }

        // Detect cycles using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut errors = Vec::new();

        for node in graph.keys() {
            if !visited.contains(node) {
                if let Some(cycle) = Self::dfs_cycle_detect(
                    node,
                    &graph,
                    &mut visited,
                    &mut rec_stack,
                    Vec::new(),
                ) {
                    errors.push(ValidationError::CycleDetected {
                        cycle,
                        context: context_neuron.to_string(),
                    });
                    break; // Report first cycle only
                }
            }
        }

        errors
    }

    /// Extract node names from SOURCE endpoints - these reference existing instances
    fn extract_node_names_from_sources(
        endpoint: &Endpoint,
        call_last_instance: &mut HashMap<String, String>,
        call_counter: &mut HashMap<String, usize>,
    ) -> Vec<String> {
        match endpoint {
            Endpoint::Call { name, args, .. } => {
                // Create base signature for this call
                let args_str = args.iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join(",");
                let base_sig = format!("{}({})", name, args_str);

                // Look up or create an instance for this call
                if let Some(existing) = call_last_instance.get(&base_sig) {
                    // Reuse existing instance
                    vec![existing.clone()]
                } else {
                    // First time seeing this call - create an instance
                    let instance_id = call_counter.entry(base_sig.clone()).or_insert(0);
                    let unique_name = format!("{}#{}", base_sig, instance_id);
                    *instance_id += 1;
                    call_last_instance.insert(base_sig, unique_name.clone());
                    vec![unique_name]
                }
            }
            Endpoint::Ref(port_ref) => vec![port_ref.node.clone()],
            Endpoint::Tuple(refs) => refs.iter().map(|r| r.node.clone()).collect(),
            Endpoint::Match(_) => vec![], // Skip Match for cycle detection
        }
    }

    /// Extract node names from DESTINATION endpoints - these CREATE new instances
    fn extract_node_names_from_destinations(
        endpoint: &Endpoint,
        call_counter: &mut HashMap<String, usize>,
        call_last_instance: &mut HashMap<String, String>,
    ) -> Vec<String> {
        match endpoint {
            Endpoint::Call { name, args, .. } => {
                // Create base signature for this call
                let args_str = args.iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join(",");
                let base_sig = format!("{}({})", name, args_str);

                // Check if we already have an instance from a source
                if let Some(existing) = call_last_instance.get(&base_sig) {
                    // Reuse the existing instance (this creates a cycle if used again later)
                    vec![existing.clone()]
                } else {
                    // Create a new unique instance
                    let instance_id = call_counter.entry(base_sig.clone()).or_insert(0);
                    let unique_name = format!("{}#{}", base_sig, instance_id);
                    *instance_id += 1;

                    // Record this as the instance for this call signature
                    call_last_instance.insert(base_sig, unique_name.clone());

                    vec![unique_name]
                }
            }
            Endpoint::Ref(port_ref) => vec![port_ref.node.clone()],
            Endpoint::Tuple(refs) => refs.iter().map(|r| r.node.clone()).collect(),
            Endpoint::Match(_) => vec![], // Skip Match for cycle detection
        }
    }

    /// Extract node names for cycle detection (legacy version for tests)
    /// Calls are identified by name + args to distinguish different instances
    #[allow(dead_code)]
    fn extract_simple_node_names(endpoint: &Endpoint) -> Vec<String> {
        match endpoint {
            Endpoint::Call { name, args, .. } => {
                // Include args in node ID to distinguish different call instances
                // Format: "NeuronName(arg1,arg2,...)"
                let args_str = args.iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join(",");
                vec![format!("{}({})", name, args_str)]
            }
            Endpoint::Ref(port_ref) => vec![port_ref.node.clone()],
            Endpoint::Tuple(refs) => refs.iter().map(|r| r.node.clone()).collect(),
            Endpoint::Match(_) => vec![], // Skip Match for cycle detection
        }
    }

    /// DFS cycle detection
    fn dfs_cycle_detect(
        node: &str,
        graph: &HashMap<String, HashSet<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        mut path: Vec<String>,
    ) -> Option<Vec<String>> {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        path.push(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if let Some(cycle) = Self::dfs_cycle_detect(
                        neighbor,
                        graph,
                        visited,
                        rec_stack,
                        path.clone(),
                    ) {
                        return Some(cycle);
                    }
                } else if rec_stack.contains(neighbor) {
                    // Found cycle - extract cycle path
                    if let Some(start_idx) = path.iter().position(|n| n == neighbor) {
                        let mut cycle = path[start_idx..].to_vec();
                        cycle.push(neighbor.to_string());
                        return Some(cycle);
                    }
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

    // Helper to create simple neuron
    fn simple_neuron(name: &str, in_shape: Shape, out_shape: Shape) -> NeuronDef {
        NeuronDef {
            name: name.to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: in_shape }],
            outputs: vec![Port { name: "default".to_string(), shape: out_shape }],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
        }
    }

    // Helper to create multi-port neuron
    fn multi_port_neuron(name: &str, inputs: Vec<Port>, outputs: Vec<Port>) -> NeuronDef {
        NeuronDef {
            name: name.to_string(),
            params: vec![],
            inputs,
            outputs,
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
        }
    }

    // Helper shapes
    fn wildcard() -> Shape {
        Shape::new(vec![Dim::Wildcard])
    }

    fn shape_512() -> Shape {
        Shape::new(vec![Dim::Literal(512)])
    }

    fn shape_256() -> Shape {
        Shape::new(vec![Dim::Literal(256)])
    }

    fn shape_batch_512() -> Shape {
        Shape::new(vec![Dim::Wildcard, Dim::Literal(512)])
    }

    fn shape_batch_256() -> Shape {
        Shape::new(vec![Dim::Wildcard, Dim::Literal(256)])
    }

    // ========== MISSING NEURON TESTS ==========

    #[test]
    fn test_missing_neuron_in_call() {
        let mut program = Program::new();
        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Call {
                    name: "MissingNeuron".to_string(),
                    args: vec![],
                    kwargs: vec![],
                    id: 0
                },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingNeuron { name, .. } if name == "MissingNeuron"
        )));
    }

    #[test]
    fn test_missing_neuron_in_match() {
        let mut program = Program::new();
        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    arms: vec![MatchArm {
                        pattern: wildcard(),
                        guard: None,
                        pipeline: vec![
                            Endpoint::Call {
                                name: "MissingInMatch".to_string(),
                                args: vec![],
                                kwargs: vec![],
                                id: 0
                            }
                        ],
                    }],
                }),
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingNeuron { name, .. } if name == "MissingInMatch"
        )));
    }

    // ========== ARITY MISMATCH TESTS ==========

    #[test]
    fn test_arity_mismatch_call_to_call() {
        let mut program = Program::new();

        // TwoOut: 1 input -> 2 outputs
        program.neurons.insert("TwoOut".to_string(), multi_port_neuron(
            "TwoOut",
            vec![Port { name: "default".to_string(), shape: wildcard() }],
            vec![
                Port { name: "a".to_string(), shape: wildcard() },
                Port { name: "b".to_string(), shape: wildcard() },
            ],
        ));

        // OneIn: 1 input -> 1 output
        program.neurons.insert("OneIn".to_string(), simple_neuron("OneIn", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "TwoOut".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Call { name: "OneIn".to_string(), args: vec![], kwargs: vec![], id: 0 },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::ArityMismatch { expected: 1, got: 2, .. }
        )));
    }

    #[test]
    fn test_arity_mismatch_tuple_unpacking() {
        let mut program = Program::new();

        // OneOut: 1 input -> 1 output
        program.neurons.insert("OneOut".to_string(), simple_neuron("OneOut", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "OneOut".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Tuple(vec![
                    PortRef::new("a"),
                    PortRef::new("b"),
                ]),
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::ArityMismatch { expected: 2, got: 1, .. }
        )));
    }

    #[test]
    fn test_arity_mismatch_tuple_to_call() {
        let mut program = Program::new();

        // TwoIn: 2 inputs -> 1 output
        program.neurons.insert("TwoIn".to_string(), multi_port_neuron(
            "TwoIn",
            vec![
                Port { name: "left".to_string(), shape: wildcard() },
                Port { name: "right".to_string(), shape: wildcard() },
            ],
            vec![Port { name: "default".to_string(), shape: wildcard() }],
        ));

        // Fork: 1 input -> 2 outputs
        program.neurons.insert("Fork".to_string(), multi_port_neuron(
            "Fork",
            vec![Port { name: "default".to_string(), shape: wildcard() }],
            vec![
                Port { name: "a".to_string(), shape: wildcard() },
                Port { name: "b".to_string(), shape: wildcard() },
            ],
        ));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                // Fork creates (a, b)
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Tuple(vec![PortRef::new("a"), PortRef::new("b")]),
                },
                // Only send (a) to TwoIn - arity mismatch
                Connection {
                    source: Endpoint::Tuple(vec![PortRef::new("a")]),
                    destination: Endpoint::Call { name: "TwoIn".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
            ]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::ArityMismatch { expected: 2, got: 1, .. }
        )));
    }

    // ========== SHAPE MISMATCH TESTS ==========

    #[test]
    fn test_shape_mismatch_literal() {
        let mut program = Program::new();
        program.neurons.insert("Out512".to_string(), simple_neuron("Out512", wildcard(), shape_512()));
        program.neurons.insert("In256".to_string(), simple_neuron("In256", shape_256(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "Out512".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Call { name: "In256".to_string(), args: vec![], kwargs: vec![], id: 0 },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::PortMismatch { .. })));
    }

    #[test]
    fn test_shape_mismatch_multi_dim() {
        let mut program = Program::new();
        program.neurons.insert("Out512".to_string(), simple_neuron("Out512", wildcard(), shape_batch_512()));
        program.neurons.insert("In256".to_string(), simple_neuron("In256", shape_batch_256(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "Out512".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Call { name: "In256".to_string(), args: vec![], kwargs: vec![], id: 0 },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::PortMismatch { .. })));
    }

    #[test]
    fn test_shape_match_exact() {
        let mut program = Program::new();
        program.neurons.insert("Out512".to_string(), simple_neuron("Out512", wildcard(), shape_512()));
        program.neurons.insert("In512".to_string(), simple_neuron("In512", shape_512(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "Out512".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Call { name: "In512".to_string(), args: vec![], kwargs: vec![], id: 0 },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok());
    }

    // ========== CYCLE DETECTION TESTS ==========

    #[test]
    fn test_simple_cycle() {
        let mut program = Program::new();
        program.neurons.insert("A".to_string(), simple_neuron("A", wildcard(), wildcard()));
        program.neurons.insert("B".to_string(), simple_neuron("B", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
            ]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::CycleDetected { .. })));
    }

    #[test]
    fn test_cycle_through_unpacked_ports() {
        let mut program = Program::new();
        program.neurons.insert("Fork".to_string(), multi_port_neuron(
            "Fork",
            vec![Port { name: "default".to_string(), shape: wildcard() }],
            vec![
                Port { name: "a".to_string(), shape: wildcard() },
                Port { name: "b".to_string(), shape: wildcard() },
            ],
        ));
        program.neurons.insert("A".to_string(), simple_neuron("A", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Tuple(vec![PortRef::new("main"), PortRef::new("skip")]),
                },
                // main -> A creates cycle: A -> Fork -> main -> A
                Connection {
                    source: Endpoint::Ref(PortRef::new("main")),
                    destination: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
            ]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::CycleDetected { .. })));
    }

    #[test]
    fn test_no_cycle_valid_residual() {
        let mut program = Program::new();

        // Fork: 1 -> 2
        program.neurons.insert("Fork".to_string(), multi_port_neuron(
            "Fork",
            vec![Port { name: "default".to_string(), shape: wildcard() }],
            vec![
                Port { name: "a".to_string(), shape: wildcard() },
                Port { name: "b".to_string(), shape: wildcard() },
            ],
        ));

        // Add: 2 -> 1
        program.neurons.insert("Add".to_string(), multi_port_neuron(
            "Add",
            vec![
                Port { name: "left".to_string(), shape: wildcard() },
                Port { name: "right".to_string(), shape: wildcard() },
            ],
            vec![Port { name: "default".to_string(), shape: wildcard() }],
        ));

        program.neurons.insert("Process".to_string(), simple_neuron("Process", wildcard(), wildcard()));

        // Residual: in -> Fork -> (main, skip), main -> Process -> processed, (processed, skip) -> Add -> out
        let composite = NeuronDef {
            name: "Residual".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Tuple(vec![PortRef::new("main"), PortRef::new("skip")]),
                },
                Connection {
                    source: Endpoint::Ref(PortRef::new("main")),
                    destination: Endpoint::Call { name: "Process".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Process".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Ref(PortRef::new("processed")),
                },
                Connection {
                    source: Endpoint::Tuple(vec![PortRef::new("processed"), PortRef::new("skip")]),
                    destination: Endpoint::Call { name: "Add".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Add".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Ref(PortRef::new("out")),
                },
            ]),
        };
        program.neurons.insert("Residual".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok(), "Valid residual pattern should not have cycles: {:?}", result);
    }

    // ========== VALID CASES ==========

    #[test]
    fn test_empty_graph() {
        let mut program = Program::new();
        let composite = NeuronDef {
            name: "Empty".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![]),
        };
        program.neurons.insert("Empty".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simple_passthrough() {
        let mut program = Program::new();
        let composite = NeuronDef {
            name: "Passthrough".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Ref(PortRef::new("out")),
            }]),
        };
        program.neurons.insert("Passthrough".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_pipeline() {
        let mut program = Program::new();
        program.neurons.insert("A".to_string(), simple_neuron("A", wildcard(), wildcard()));
        program.neurons.insert("B".to_string(), simple_neuron("B", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Pipeline".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Ref(PortRef::new("out")),
                },
            ]),
        };
        program.neurons.insert("Pipeline".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok());
    }
}
