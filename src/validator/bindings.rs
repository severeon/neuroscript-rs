use crate::interfaces::*;
use std::collections::{BTreeMap, BTreeSet, HashSet};

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
    // Build a dependency graph: for each @lazy binding, find which other @lazy
    // bindings it references (via Value::Name args). Then detect cycles.
    //
    // NOTE: This only detects mutual recursion within a single neuron's context
    // bindings. Cross-neuron mutual recursion (neuron A's binding calls neuron B,
    // neuron B's binding calls neuron A) requires whole-program call graph
    // analysis and is not handled here.
    let lazy_bindings: BTreeSet<&str> = context_bindings
        .iter()
        .filter(|b| matches!(b.scope, Scope::Instance { lazy: true }))
        .map(|b| b.name.as_str())
        .collect();

    // A single @lazy binding cannot form mutual recursion with itself;
    // self-recursion is already checked in the per-binding loop above.
    if lazy_bindings.len() >= 2 {
        // Build adjacency list: only edges between @lazy bindings.
        // BTreeMap ensures deterministic DFS traversal order.
        let mut deps: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for binding in context_bindings {
            if !lazy_bindings.contains(binding.name.as_str()) {
                continue;
            }
            let mut refs = BTreeSet::new();
            for arg in &binding.args {
                collect_name_refs(arg, &lazy_bindings, &mut refs);
            }
            for (_key, val) in &binding.kwargs {
                collect_name_refs(val, &lazy_bindings, &mut refs);
            }
            refs.remove(binding.name.as_str());
            deps.insert(binding.name.as_str(), refs);
        }

        let mut dfs = CycleDfs {
            deps: &deps,
            visited: HashSet::new(),
            stack: HashSet::new(),
            path: Vec::new(),
            reported_cycles: HashSet::new(),
        };

        for &node in &lazy_bindings {
            if !dfs.visited.contains(node) {
                dfs.find_cycles(node, &neuron.name, &mut errors);
            }
        }
    }

    errors
}

/// Collect Value::Name references that match a set of target names
fn collect_name_refs<'a>(
    value: &'a Value,
    targets: &BTreeSet<&str>,
    refs: &mut BTreeSet<&'a str>,
) {
    match value {
        Value::Int(_) | Value::Float(_) | Value::String(_) | Value::Bool(_) | Value::Global(_) => {}
        Value::Name(name) if targets.contains(name.as_str()) => {
            refs.insert(name.as_str());
        }
        Value::Name(_) => {}
        Value::BinOp { left, right, .. } => {
            collect_name_refs(left, targets, refs);
            collect_name_refs(right, targets, refs);
        }
        Value::Call { args, kwargs, .. } => {
            for arg in args {
                collect_name_refs(arg, targets, refs);
            }
            for (_key, val) in kwargs {
                collect_name_refs(val, targets, refs);
            }
        }
    }
}

/// DFS state for cycle detection across @lazy bindings.
struct CycleDfs<'a> {
    deps: &'a BTreeMap<&'a str, BTreeSet<&'a str>>,
    visited: HashSet<&'a str>,
    stack: HashSet<&'a str>,
    path: Vec<&'a str>,
    /// Set of already-reported cycle strings to avoid exact duplicates.
    reported_cycles: HashSet<String>,
}

impl<'a> CycleDfs<'a> {
    fn find_cycles(
        &mut self,
        node: &'a str,
        neuron_name: &str,
        errors: &mut Vec<ValidationError>,
    ) {
        self.visited.insert(node);
        self.stack.insert(node);
        self.path.push(node);

        if let Some(neighbors) = self.deps.get(node) {
            for &neighbor in neighbors {
                if !self.visited.contains(neighbor) {
                    self.find_cycles(neighbor, neuron_name, errors);
                } else if self.stack.contains(neighbor) {
                    let cycle_start = self
                        .path
                        .iter()
                        .position(|&n| n == neighbor)
                        .expect("cycle node must be in path since it is on the DFS stack");
                    let mut cycle: Vec<String> =
                        self.path[cycle_start..].iter().map(|s| s.to_string()).collect();
                    cycle.push(neighbor.to_string());
                    // Normalize: rotate so the lexicographically smallest node is first,
                    // to deduplicate cycles found from different starting points.
                    let key = normalize_cycle(&cycle);
                    if self.reported_cycles.insert(key) {
                        errors.push(ValidationError::MutualLazyRecursion {
                            cycle,
                            neuron: neuron_name.to_string(),
                        });
                    }
                }
            }
        }

        self.stack.remove(node);
        self.path.pop();
    }
}

/// Normalize a cycle for deduplication: rotate the nodes (excluding the
/// repeated tail) so the lexicographically smallest node comes first.
fn normalize_cycle(cycle: &[String]) -> String {
    // cycle is [a, b, ..., a] -- strip the repeated tail for rotation
    let nodes = &cycle[..cycle.len() - 1];
    if nodes.is_empty() {
        return String::new();
    }
    let min_pos = nodes
        .iter()
        .enumerate()
        .min_by_key(|(_, n)| n.as_str())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut rotated: Vec<&str> = nodes[min_pos..].iter().map(|s| s.as_str()).collect();
    rotated.extend(nodes[..min_pos].iter().map(|s| s.as_str()));
    rotated.join(",")
}
