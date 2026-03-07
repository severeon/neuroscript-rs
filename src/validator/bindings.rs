use crate::interfaces::*;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Validate context:, let: and set: bindings
pub(super) fn validate_bindings(
    neuron: &NeuronDef,
    context_bindings: &[Binding],
    program: &Program,
    registry: &StdlibRegistry,
    neuron_exists_fn: impl Fn(&str, &Program, &StdlibRegistry) -> bool,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut defined_bindings = HashSet::new();

    // Collect neuron-typed parameter names for higher-order neuron support
    let neuron_param_names: HashSet<&str> = neuron
        .params
        .iter()
        .filter(|p| p.type_annotation.as_ref() == Some(&ParamType::Neuron))
        .map(|p| p.name.as_str())
        .collect();

    // 1. Validate unified context: bindings
    for binding in context_bindings {
        // Check if the neuron being called exists
        // (This could be a neuron in the program, a primitive, a global name,
        // or a neuron-typed parameter like `block: Neuron`)
        let is_neuron_param = neuron_param_names.contains(binding.call_name.as_str());
        if !is_neuron_param && !neuron_exists_fn(&binding.call_name, program, registry) {
            errors.push(ValidationError::MissingNeuron {
                name: binding.call_name.clone(),
                context: format!(
                    "context binding '{}' in neuron '{}'",
                    binding.name, neuron.name
                ),
            });
        }

        // Validate arguments to Neuron-typed parameters at call sites.
        // When calling a user-defined neuron, check that any arg at a `: Neuron` parameter
        // position is a valid neuron name (not an arbitrary value).
        if !is_neuron_param {
            if let Some(callee) = program.neurons.get(&binding.call_name) {
                for (idx, param) in callee.params.iter().enumerate() {
                    if param.type_annotation.as_ref() == Some(&ParamType::Neuron) {
                        if let Some(arg) = binding.args.get(idx) {
                            match arg {
                                Value::Name(name) => {
                                    if !neuron_exists_fn(name, program, registry)
                                        && !neuron_param_names.contains(name.as_str())
                                    {
                                        errors.push(ValidationError::Custom(format!(
                                            "Argument '{}' for Neuron-typed parameter '{}' of \
                                             '{}' in neuron '{}' is not a known neuron",
                                            name, param.name, binding.call_name, neuron.name
                                        )));
                                    }
                                }
                                _ => {
                                    errors.push(ValidationError::Custom(format!(
                                        "Parameter '{}' of '{}' expects a neuron type, but got \
                                         a non-name value in neuron '{}'",
                                        param.name, binding.call_name, neuron.name
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check for duplicate binding names inside the same neuron
        if defined_bindings.contains(&binding.name) {
            errors.push(ValidationError::DuplicateBinding {
                name: binding.name.clone(),
                neuron: neuron.name.clone(),
            });
        }
        defined_bindings.insert(binding.name.clone());

        // Scope-specific validation
        match &binding.scope {
            Scope::Instance { lazy } => {
                // If eager (lazy=false), check for recursion (not allowed)
                if !lazy && binding.call_name == neuron.name {
                    errors.push(ValidationError::InvalidRecursion {
                        binding: binding.name.clone(),
                        neuron: neuron.name.clone(),
                        reason:
                            "Eager instance bindings cannot be recursive (use @lazy for recursion)"
                                .to_string(),
                    });
                }

                // If lazy, allowed to be recursive
                if *lazy && binding.call_name == neuron.name {
                    // Check for infinite recursion (no parameters)
                    if binding.args.is_empty() && binding.kwargs.is_empty() {
                        errors.push(ValidationError::InvalidRecursion {
                            binding: binding.name.clone(),
                            neuron: neuron.name.clone(),
                            reason: "@lazy binding to self without arguments may cause infinite recursion"
                                .to_string(),
                        });
                    }
                }
            }
            Scope::Static => {
                // Static bindings are always eager, cannot be recursive
                if binding.call_name == neuron.name {
                    errors.push(ValidationError::InvalidRecursion {
                        binding: binding.name.clone(),
                        neuron: neuron.name.clone(),
                        reason:
                            "@static bindings cannot be recursive (static weights must be finite)"
                                .to_string(),
                    });
                }
            }
            Scope::Global => {
                // @global annotations should not appear in context: blocks
                // (They are only for module-level @global declarations)
                // Wait, the spec says: "context bindings MAY reference @global names"
                // But the binding itself shouldn't have Scope::Global if it's IN a context: block.
                // It should probably be Scope::Instance or Scope::Static.

                // Actually, if a context: block has `@global vocab = Embedding(...)`,
                // it's an error. It should be `vocab = @global vocab_table`.
                // Wait, `neuroscript.pest` allows `at ~ keyword_global ~ binding` in `context_binding`.
                // But `mvp-todo.md` says: "Validate @global only appears at module level (not in neurons)"
                // So Scope::Global in a binding in a neuron is an ERROR.
                errors.push(ValidationError::Custom(format!(
                    "Binding '{}' in neuron '{}' cannot be marked @global. Use @global at module level instead.",
                    binding.name, neuron.name
                )));
            }
        }
    }

    // Mutual @lazy recursion detection within the same context block.
    // Build a dependency graph: for each @lazy binding, find which other bindings
    // it references (via Value::Name args that match another binding's name).
    // Then detect cycles in this graph.
    let lazy_bindings: BTreeSet<&str> = context_bindings
        .iter()
        .filter(|b| matches!(b.scope, Scope::Instance { lazy: true }))
        .map(|b| b.name.as_str())
        .collect();

    if lazy_bindings.len() >= 2 {
        let binding_names: HashSet<&str> = context_bindings
            .iter()
            .map(|b| b.name.as_str())
            .collect();

        // Build adjacency list: binding name -> set of binding names it references
        let mut deps: HashMap<&str, BTreeSet<&str>> = HashMap::new();
        for binding in context_bindings {
            if !lazy_bindings.contains(binding.name.as_str()) {
                continue;
            }
            let mut refs = BTreeSet::new();
            for arg in &binding.args {
                collect_name_refs(arg, &binding_names, &mut refs);
            }
            for (_key, val) in &binding.kwargs {
                collect_name_refs(val, &binding_names, &mut refs);
            }
            refs.remove(binding.name.as_str());
            deps.insert(binding.name.as_str(), refs);
        }

        // DFS cycle detection -- collect all independent cycles
        let mut visited: HashSet<&str> = HashSet::new();
        let mut stack: HashSet<&str> = HashSet::new();
        let mut path: Vec<&str> = Vec::new();
        let mut reported_in_cycle: HashSet<&str> = HashSet::new();

        for &node in &lazy_bindings {
            if !visited.contains(node) {
                find_cycles(
                    node,
                    &deps,
                    &mut visited,
                    &mut stack,
                    &mut path,
                    &mut reported_in_cycle,
                    &neuron.name,
                    &mut errors,
                );
            }
        }
    }

    errors
}

/// Collect Value::Name references that match known binding names
fn collect_name_refs<'a>(
    value: &'a Value,
    binding_names: &HashSet<&str>,
    refs: &mut BTreeSet<&'a str>,
) {
    match value {
        Value::Int(_) | Value::Float(_) | Value::String(_) | Value::Bool(_) | Value::Global(_) => {}
        Value::Name(name) if binding_names.contains(name.as_str()) => {
            refs.insert(name.as_str());
        }
        Value::Name(_) => {}
        Value::BinOp { left, right, .. } => {
            collect_name_refs(left, binding_names, refs);
            collect_name_refs(right, binding_names, refs);
        }
        Value::Call { args, kwargs, .. } => {
            for arg in args {
                collect_name_refs(arg, binding_names, refs);
            }
            for (_key, val) in kwargs {
                collect_name_refs(val, binding_names, refs);
            }
        }
    }
}

/// DFS-based cycle detection. Pushes errors for all independent cycles found.
fn find_cycles<'a>(
    node: &'a str,
    deps: &HashMap<&'a str, BTreeSet<&'a str>>,
    visited: &mut HashSet<&'a str>,
    stack: &mut HashSet<&'a str>,
    path: &mut Vec<&'a str>,
    reported_in_cycle: &mut HashSet<&'a str>,
    neuron_name: &str,
    errors: &mut Vec<ValidationError>,
) {
    visited.insert(node);
    stack.insert(node);
    path.push(node);

    if let Some(neighbors) = deps.get(node) {
        for &neighbor in neighbors {
            if !visited.contains(neighbor) {
                find_cycles(
                    neighbor,
                    deps,
                    visited,
                    stack,
                    path,
                    reported_in_cycle,
                    neuron_name,
                    errors,
                );
            } else if stack.contains(neighbor) && !reported_in_cycle.contains(neighbor) {
                let cycle_start = path
                    .iter()
                    .position(|&n| n == neighbor)
                    .expect("cycle node must be in path since it is on the stack");
                let mut cycle: Vec<String> =
                    path[cycle_start..].iter().map(|s| s.to_string()).collect();
                cycle.push(neighbor.to_string());
                for &n in &path[cycle_start..] {
                    reported_in_cycle.insert(n);
                }
                let cycle_str = cycle.join(" -> ");
                errors.push(ValidationError::Custom(format!(
                    "Mutual @lazy recursion detected between bindings: {} (in {})",
                    cycle_str, neuron_name
                )));
            }
        }
    }

    stack.remove(node);
    path.pop();
}
