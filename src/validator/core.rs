use super::bindings;
use super::cycles;
use super::shapes;
use super::symbol_table;
use crate::interfaces::*;

/// Graph validator
pub struct Validator;

impl Validator {
    /// Check if a neuron exists (either in the program or as a primitive)
    fn neuron_exists(name: &str, program: &Program, registry: &StdlibRegistry) -> bool {
        program.neurons.contains_key(name) || registry.contains(name)
    }

    /// Validate an entire program
    pub fn validate(program: &mut Program) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        let registry = StdlibRegistry::new();

        // Check each neuron (read-only pass for structure)
        // We use a scope to limit the borrow of program
        {
            for neuron in program.neurons.values() {
                // Validate connections within this neuron if it's composite
                if let NeuronBody::Graph {
                    let_bindings,
                    set_bindings,
                    context_bindings: _,
                    connections,
                } = &neuron.body
                {
                    // Validate bindings first
                    errors.extend(bindings::validate_bindings(
                        neuron,
                        let_bindings,
                        set_bindings,
                        program,
                        &registry,
                        Self::neuron_exists,
                    ));

                    errors.extend(Self::validate_neuron_graph(
                        neuron,
                        connections,
                        program,
                        &registry,
                    ));
                }
            }
        }

        // Check match expressions (mutable pass for reachability)
        for (neuron_name, neuron) in &mut program.neurons {
            if let NeuronBody::Graph { connections, .. } = &mut neuron.body {
                for connection in connections {
                    if let Endpoint::Match(match_expr) = &mut connection.destination {
                        errors.extend(Self::validate_match_expression(match_expr, neuron_name));
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

    /// Validate a single neuron's graph
    fn validate_neuron_graph(
        neuron: &NeuronDef,
        connections: &[Connection],
        program: &Program,
        registry: &StdlibRegistry,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Build symbol table for this neuron's graph
        let symbol_table = symbol_table::build_symbol_table(
            neuron,
            connections,
            program,
            registry,
            &mut errors,
            shapes::substitute_params,
        );

        // Validate each connection
        for connection in connections {
            // Check that neurons exist
            errors.extend(Self::check_neurons_exist(
                &connection.source,
                &neuron.name,
                program,
                registry,
            ));
            errors.extend(Self::check_neurons_exist(
                &connection.destination,
                &neuron.name,
                program,
                registry,
            ));

            // Resolve source and destination endpoints
            let source_resolution = symbol_table::resolve_endpoint(
                &connection.source,
                neuron,
                &symbol_table,
                program,
                registry,
                true, // is_source
                &shapes::substitute_params,
            );
            let dest_resolution = symbol_table::resolve_endpoint(
                &connection.destination,
                neuron,
                &symbol_table,
                program,
                registry,
                false, // is_source
                &shapes::substitute_params,
            );

            match (source_resolution, dest_resolution) {
                (Ok(source_ports), Ok(dest_ports)) => {
                    // Check compatibility
                    errors.extend(symbol_table::check_port_compatibility(
                        &source_ports,
                        &dest_ports,
                        &connection.source,
                        &connection.destination,
                        &neuron.name,
                        shapes::shapes_compatible,
                    ));
                }
                (Err(e), _) | (_, Err(e)) => {
                    errors.push(e);
                }
            }
        }

        // Check for cycles (respecting max_cycle_depth if set)
        errors.extend(cycles::detect_cycles(
            connections,
            neuron,
            &symbol_table,
            program,
        ));

        errors
    }

    /// Check that all neurons referenced in an endpoint exist
    fn check_neurons_exist(
        endpoint: &Endpoint,
        context_neuron: &str,
        program: &Program,
        registry: &StdlibRegistry,
    ) -> Vec<ValidationError> {
        match endpoint {
            Endpoint::Call { name, .. } => {
                if !Self::neuron_exists(name, program, registry) {
                    vec![ValidationError::MissingNeuron {
                        name: name.clone(),
                        context: context_neuron.to_string(),
                    }]
                } else {
                    vec![]
                }
            }
            Endpoint::Match(match_expr) => match_expr
                .arms
                .iter()
                .flat_map(|arm| {
                    arm.pipeline.iter().flat_map(|ep| {
                        Self::check_neurons_exist(ep, context_neuron, program, registry)
                    })
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Validate a match expression for exhaustiveness and pattern shadowing
    /// Marks unreachable arms by setting is_reachable = false
    fn validate_match_expression(
        match_expr: &mut MatchExpr,
        context_neuron: &str,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Check exhaustiveness: last pattern should be a catch-all
        if !match_expr.arms.is_empty() {
            let last_pattern = &match_expr.arms.last().unwrap().pattern;
            if !Self::is_catch_all_pattern(last_pattern) {
                errors.push(ValidationError::NonExhaustiveMatch {
                    context: context_neuron.to_string(),
                    suggestion: "Add a catch-all pattern as the last arm, e.g., [*shape] or [*, d]"
                        .to_string(),
                });
            }
        }

        // Check for pattern shadowing and mark unreachable arms
        for i in 0..match_expr.arms.len() {
            for j in (i + 1)..match_expr.arms.len() {
                // Check if arm i subsumes arm j (making j unreachable)
                // A pattern with a guard does NOT subsume any pattern (guard can fail)
                // We need to access arms carefully to avoid multiple mutable borrows
                let subsumes = {
                    let arm_i = &match_expr.arms[i];
                    let arm_j = &match_expr.arms[j];
                    arm_i.guard.is_none() && Self::pattern_subsumes(&arm_i.pattern, &arm_j.pattern)
                };

                if subsumes {
                    // Mark as unreachable
                    match_expr.arms[j].is_reachable = false;

                    // We don't error on shadowing anymore, just mark it
                    // errors.push(ValidationError::UnreachableMatchArm { ... });
                }
            }
        }

        errors
    }

    /// Delegation method for is_catch_all_pattern (for backward compatibility)
    /// Used by tests that call Validator::is_catch_all_pattern()
    pub fn is_catch_all_pattern(pattern: &Shape) -> bool {
        shapes::is_catch_all_pattern(pattern)
    }

    /// Delegation method for pattern_subsumes (for backward compatibility)
    /// Used by tests that call Validator::pattern_subsumes()
    pub fn pattern_subsumes(general: &Shape, specific: &Shape) -> bool {
        shapes::pattern_subsumes(general, specific)
    }
}
