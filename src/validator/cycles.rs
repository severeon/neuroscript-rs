use crate::interfaces::*;
use super::symbol_table::SymbolTable;
use std::collections::{HashMap, HashSet};

/// Detect cycles in the neuron dependency graph
pub(super) fn detect_cycles(
    connections: &[Connection],
    neuron: &NeuronDef,
    symbol_table: &SymbolTable,
    _program: &Program,
) -> Vec<ValidationError> {
    let context_neuron = &neuron.name;
    // Build dependency graph: which nodes flow to which others
    let mut graph: HashMap<String, HashSet<String>> = HashMap::new();

    // Add symbol table nodes
    for node in symbol_table.node_names() {
        graph.insert(node.clone(), HashSet::new());
    }

    // Track Call destinations: each destination creates a new unique instance
    // Track Call sources: sources map to their most recent destination instance
    let mut call_last_instance: HashMap<String, String> = HashMap::new();
    let mut call_instance_counter: HashMap<String, usize> = HashMap::new();

    // Build edges from connections
    for connection in connections {
        // Extract source nodes - these reference existing instances
        let source_nodes = extract_node_names_from_sources(
            &connection.source,
            &mut call_last_instance,
            &mut call_instance_counter,
        );

        // Extract destination nodes - these CREATE new instances
        let dest_nodes = extract_node_names_from_destinations(
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
            if let Some(cycle) =
                dfs_cycle_detect(node, &graph, &mut visited, &mut rec_stack, Vec::new())
            {
                // Check if this cycle is allowed by max_cycle_depth
                let cycle_depth = cycle.len() - 1; // Subtract 1 because cycle includes start node twice

                if let Some(max_depth) = neuron.max_cycle_depth {
                    if cycle_depth <= max_depth {
                        // Cycle is within allowed depth, skip error
                        continue;
                    }
                }

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
pub(super) fn extract_node_names_from_sources(
    endpoint: &Endpoint,
    call_last_instance: &mut HashMap<String, String>,
    call_counter: &mut HashMap<String, usize>,
) -> Vec<String> {
    match endpoint {
        Endpoint::Call { name, args, .. } => {
            // Create base signature for this call
            let args_str = args
                .iter()
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
pub(super) fn extract_node_names_from_destinations(
    endpoint: &Endpoint,
    call_counter: &mut HashMap<String, usize>,
    call_last_instance: &mut HashMap<String, String>,
) -> Vec<String> {
    match endpoint {
        Endpoint::Call { name, args, .. } => {
            // Create base signature for this call
            let args_str = args
                .iter()
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
pub(super) fn extract_simple_node_names(endpoint: &Endpoint) -> Vec<String> {
    match endpoint {
        Endpoint::Call { name, args, .. } => {
            // Include args in node ID to distinguish different call instances
            // Format: "NeuronName(arg1,arg2,...)"
            let args_str = args
                .iter()
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
pub(super) fn dfs_cycle_detect(
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
                if let Some(cycle) =
                    dfs_cycle_detect(neighbor, graph, visited, rec_stack, path.clone())
                {
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
