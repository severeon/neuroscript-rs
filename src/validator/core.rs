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
        // __sequential__ is a synthetic pseudo-neuron created by @wrap desugaring
        program.neurons.contains_key(name)
            || registry.contains(name)
            || name == crate::interfaces::SEQUENTIAL_PSEUDO_NEURON
    }

    /// Validate an entire program
    pub fn validate(program: &mut Program) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        let registry = StdlibRegistry::new();

        // 1. Validate global declarations
        for global in &program.globals {
            // Globals can be calls or simple values
            if let Value::Call { name, .. } = &global.value {
                if !Self::neuron_exists(name, program, &registry) {
                    errors.push(ValidationError::MissingNeuron {
                        name: name.clone(),
                        context: format!("global declaration '{}'", global.name),
                    });
                }
            }
        }

        // 2. Validate variadic port declarations
        for neuron in program.neurons.values() {
            // Variadic ports on outputs are not supported
            for port in &neuron.outputs {
                if port.variadic {
                    errors.push(ValidationError::Custom(format!(
                        "Variadic output ports are not supported: out *{} in neuron '{}'",
                        port.name, neuron.name
                    )));
                }
            }
            // A neuron with a variadic input must have exactly one input port
            let variadic_count = neuron.inputs.iter().filter(|p| p.variadic).count();
            if variadic_count > 0 && neuron.inputs.len() != 1 {
                errors.push(ValidationError::Custom(format!(
                    "A neuron with a variadic input port must have exactly one input declaration, \
                     but '{}' has {} inputs",
                    neuron.name,
                    neuron.inputs.len()
                )));
            }
            // Variadic port must have an explicit name (not "default")
            for port in &neuron.inputs {
                if port.variadic && port.name == "default" {
                    errors.push(ValidationError::Custom(format!(
                        "Variadic input port in neuron '{}' needs an explicit name. \
                         Write `in *inputs: [shape]` (or another name) instead of an unnamed variadic port.",
                        neuron.name
                    )));
                }
            }
        }

        // 3. Check each neuron (read-only pass for structure)
        // We use a scope to limit the borrow of program
        {
            for neuron in program.neurons.values() {
                // Validate connections within this neuron if it's composite
                if let NeuronBody::Graph {
                    context_bindings,
                    connections,
                    ..
                } = &neuron.body
                {
                    // Validate bindings first
                    errors.extend(bindings::validate_bindings(
                        neuron,
                        context_bindings,
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
            let params = neuron.params.clone();
            if let NeuronBody::Graph { connections, .. } = &mut neuron.body {
                for connection in connections {
                    if let Endpoint::Match(match_expr) = &mut connection.destination {
                        errors.extend(Self::validate_match_expression(match_expr, neuron_name, &params));
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

        // Collect neuron-typed parameter names for higher-order neuron support
        let neuron_param_names: std::collections::HashSet<&str> = neuron
            .params
            .iter()
            .filter(|p| p.type_annotation.as_ref() == Some(&ParamType::Neuron))
            .map(|p| p.name.as_str())
            .collect();

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
            // Check that neurons exist (skip neuron-typed params)
            errors.extend(Self::check_neurons_exist(
                &connection.source,
                &neuron.name,
                program,
                registry,
                &neuron_param_names,
            ));
            errors.extend(Self::check_neurons_exist(
                &connection.destination,
                &neuron.name,
                program,
                registry,
                &neuron_param_names,
            ));

            let res_ctx = symbol_table::ResolutionContext {
                neuron,
                program,
                registry,
                substitute_params_fn: shapes::substitute_params,
            };

            // Resolve source and destination endpoints
            let source_resolution = symbol_table::resolve_endpoint(
                &connection.source,
                &res_ctx,
                &symbol_table,
                true, // is_source
            );
            let dest_resolution = symbol_table::resolve_endpoint(
                &connection.destination,
                &res_ctx,
                &symbol_table,
                false, // is_source
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
                    errors.push(*e);
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
        neuron_param_names: &std::collections::HashSet<&str>,
    ) -> Vec<ValidationError> {
        match endpoint {
            Endpoint::Call { name, .. } => {
                // Skip check if the name is a neuron-typed parameter (higher-order neuron)
                if neuron_param_names.contains(name.as_str()) {
                    vec![]
                } else if !Self::neuron_exists(name, program, registry) {
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
                        Self::check_neurons_exist(ep, context_neuron, program, registry, neuron_param_names)
                    })
                })
                .collect(),
            Endpoint::If(if_expr) => {
                let mut errors = Vec::new();
                for branch in &if_expr.branches {
                    for ep in &branch.pipeline {
                        errors.extend(Self::check_neurons_exist(
                            ep,
                            context_neuron,
                            program,
                            registry,
                            neuron_param_names,
                        ));
                    }
                }
                if let Some(else_branch) = &if_expr.else_branch {
                    for ep in else_branch {
                        errors.extend(Self::check_neurons_exist(
                            ep,
                            context_neuron,
                            program,
                            registry,
                            neuron_param_names,
                        ));
                    }
                }
                errors
            }
            Endpoint::Reshape(reshape) => {
                let mut errors = Vec::new();
                // Reject empty reshape expressions (=> [])
                if reshape.dims.is_empty() {
                    errors.push(ValidationError::InvalidReshape {
                        message: "reshape expression must have at least one dimension".to_string(),
                        context: format!("in {}", context_neuron),
                        span: None, // TODO: propagate source span from ReshapeExpr
                    });
                }
                // Validate at most one 'others' dimension (PyTorch allows only one -1)
                let others_count = reshape
                    .dims
                    .iter()
                    .filter(|d| matches!(d, ReshapeDim::Others))
                    .count();
                if others_count > 1 {
                    errors.push(ValidationError::InvalidReshape {
                        message: format!(
                            "reshape expression has {} 'others' dimensions, but only one is allowed",
                            others_count
                        ),
                        context: format!("in {}", context_neuron),
                        span: None, // TODO: propagate source span from ReshapeExpr
                    });
                }
                if let Some(ref annotation) = reshape.annotation {
                    let strategy = match annotation {
                        TransformAnnotation::Reduce(s) => s,
                        TransformAnnotation::Repeat(s) => s,
                    };
                    match strategy {
                        TransformStrategy::Neuron { name, .. } => {
                            // Skip check if the name is a neuron-typed parameter
                            if !neuron_param_names.contains(name.as_str())
                                && !Self::neuron_exists(name, program, registry)
                            {
                                errors.push(ValidationError::MissingNeuron {
                                    name: name.clone(),
                                    context: format!(
                                        "transform annotation in {}",
                                        context_neuron
                                    ),
                                });
                            }
                        }
                        TransformStrategy::Intrinsic(name) => {
                            let valid_intrinsics: &[&str] = match annotation {
                                TransformAnnotation::Reduce(_) => {
                                    &["mean", "sum", "min", "max", "prod", "logsumexp"]
                                }
                                TransformAnnotation::Repeat(_) => &["copy"],
                            };
                            if !valid_intrinsics.contains(&name.as_str()) {
                                errors.push(ValidationError::InvalidAnnotation {
                                    annotation: format!("{}", annotation),
                                    reason: format!(
                                        "unknown intrinsic '{}', expected one of: {}",
                                        name,
                                        valid_intrinsics.join(", ")
                                    ),
                                    context: format!("in {}", context_neuron),
                                    span: None, // TODO: propagate source span from annotation
                                });
                            }
                        }
                    }
                }
                errors
            }
            _ => vec![],
        }
    }

    /// Validate a match expression for exhaustiveness and pattern shadowing
    /// Marks unreachable arms by setting is_reachable = false
    fn validate_match_expression(
        match_expr: &mut MatchExpr,
        context_neuron: &str,
        neuron_params: &[Param],
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // For Named subject (neuron contract dispatch), validate parameter and skip shape checks
        if let MatchSubject::Named(param_name) = &match_expr.subject {
            // Check that the parameter exists
            let param = neuron_params.iter().find(|p| &p.name == param_name);
            match param {
                Some(p) => {
                    // Check that the parameter has Neuron type annotation
                    if p.type_annotation.as_ref() != Some(&ParamType::Neuron) {
                        errors.push(ValidationError::Custom(format!(
                            "match({}) in '{}': parameter '{}' must have type annotation ': Neuron'",
                            param_name, context_neuron, param_name
                        )));
                    }
                }
                None => {
                    errors.push(ValidationError::Custom(format!(
                        "match({}) in '{}': parameter '{}' not found",
                        param_name, context_neuron, param_name
                    )));
                }
            }
            // Validate that all arms use NeuronContract patterns
            for (i, arm) in match_expr.arms.iter().enumerate() {
                if matches!(arm.pattern, MatchPattern::Shape(_)) {
                    errors.push(ValidationError::Custom(format!(
                        "match({}) in '{}': arm {} uses shape pattern but neuron contract pattern expected",
                        param_name, context_neuron, i + 1
                    )));
                }
            }
            return errors;
        }

        // Check exhaustiveness: last pattern should be a catch-all
        if !match_expr.arms.is_empty() {
            let last_pattern = &match_expr.arms.last().unwrap().pattern;
            if let Some(shape) = last_pattern.as_shape() {
                if !Self::is_catch_all_pattern(shape) {
                    errors.push(ValidationError::NonExhaustiveMatch {
                        context: context_neuron.to_string(),
                        suggestion: "Add a catch-all pattern as the last arm, e.g., [*shape] or [*, d]"
                            .to_string(),
                    });
                }
            }
        }

        // Check for pattern shadowing and mark unreachable arms
        for i in 0..match_expr.arms.len() {
            for j in (i + 1)..match_expr.arms.len() {
                // Check if arm i subsumes arm j (making j unreachable)
                // A pattern with a guard does NOT subsume any pattern (guard can fail)
                let subsumes = {
                    let arm_i = &match_expr.arms[i];
                    let arm_j = &match_expr.arms[j];
                    match (arm_i.pattern.as_shape(), arm_j.pattern.as_shape()) {
                        (Some(shape_i), Some(shape_j)) => {
                            arm_i.guard.is_none() && Self::pattern_subsumes(shape_i, shape_j)
                        }
                        _ => false,
                    }
                };

                if subsumes {
                    // Mark as unreachable
                    match_expr.arms[j].is_reachable = false;
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
