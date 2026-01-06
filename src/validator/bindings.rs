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

    // 1. Validate unified context: bindings
    for binding in context_bindings {
        // Check if the neuron being called exists
        // (This could be a neuron in the program, a primitive, or a global name)
        if !neuron_exists_fn(&binding.call_name, program, registry) {
            errors.push(ValidationError::MissingNeuron {
                name: binding.call_name.clone(),
                context: format!(
                    "context binding '{}' in neuron '{}'",
                    binding.name, neuron.name
                ),
            });
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

    // 2. Validate scope dependencies
    for binding in context_bindings {
        // Find what this binding is calling
        // It could be a neuron name or another binding name
        if let Some(called_binding) = context_bindings
            .iter()
            .find(|b| b.name == binding.call_name)
        {
            match (&binding.scope, &called_binding.scope) {
                // Task 7.3.4: Instance bindings can't reference static bindings
                (Scope::Instance { .. }, Scope::Static) => {
                    errors.push(ValidationError::Custom(format!(
                        "Instance binding '{}' in neuron '{}' cannot reference @static binding '{}'.",
                        binding.name, neuron.name, called_binding.name
                    )));
                }
                // Static bindings can reference globals and other statics
                (Scope::Static, Scope::Instance { .. }) => {
                    errors.push(ValidationError::Custom(format!(
                        "Static binding '{}' in neuron '{}' cannot reference instance binding '{}'.",
                        binding.name, neuron.name, called_binding.name
                    )));
                }
                _ => {}
            }
        }

        // Check arguments for scope violations
        for arg in &binding.args {
            check_value_scope(arg, &binding.scope, context_bindings, neuron, &mut errors);
        }
        for (_, val) in &binding.kwargs {
            check_value_scope(val, &binding.scope, context_bindings, neuron, &mut errors);
        }
    }

    errors
}

/// Check if a value is compatible with the given scope
fn check_value_scope(
    value: &Value,
    current_scope: &Scope,
    context_bindings: &[Binding],
    neuron: &NeuronDef,
    errors: &mut Vec<ValidationError>,
) {
    match value {
        Value::Name(name) => {
            if let Some(called_binding) = context_bindings.iter().find(|b| b.name == *name) {
                match (current_scope, &called_binding.scope) {
                    (Scope::Instance { .. }, Scope::Static) => {
                        errors.push(ValidationError::Custom(format!(
                            "Instance scope in neuron '{}' cannot reference @static binding '{}'.",
                            neuron.name, name
                        )));
                    }
                    (Scope::Static, Scope::Instance { .. }) => {
                        errors.push(ValidationError::Custom(format!(
                            "Static scope in neuron '{}' cannot reference instance binding '{}'.",
                            neuron.name, name
                        )));
                    }
                    _ => {}
                }
            }
        }
        Value::BinOp { left, right, .. } => {
            check_value_scope(left, current_scope, context_bindings, neuron, errors);
            check_value_scope(right, current_scope, context_bindings, neuron, errors);
        }
        Value::Call { args, kwargs, .. } => {
            for arg in args {
                check_value_scope(arg, current_scope, context_bindings, neuron, errors);
            }
            for (_, val) in kwargs {
                check_value_scope(val, current_scope, context_bindings, neuron, errors);
            }
        }
        _ => {}
    }
}
