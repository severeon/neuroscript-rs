use crate::interfaces::*;
use std::collections::HashSet;

/// Validate let: and set: bindings
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

    errors
}
