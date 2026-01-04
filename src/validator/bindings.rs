use crate::interfaces::*;
use std::collections::HashSet;

/// Validate let: and set: bindings
pub(super) fn validate_bindings(
    neuron: &NeuronDef,
    let_bindings: &[Binding],
    set_bindings: &[Binding],
    program: &Program,
    registry: &StdlibRegistry,
    neuron_exists_fn: impl Fn(&str, &Program, &StdlibRegistry) -> bool,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut defined_bindings = HashSet::new();

    // Check set: bindings first (they're evaluated eagerly in __init__)
    for binding in set_bindings {
        // Check if the neuron being called exists
        if !neuron_exists_fn(&binding.call_name, program, registry) {
            errors.push(ValidationError::MissingNeuron {
                name: binding.call_name.clone(),
                context: format!("set binding '{}' in neuron '{}'", binding.name, neuron.name),
            });
        }

        // Check for duplicate binding names
        if defined_bindings.contains(&binding.name) {
            errors.push(ValidationError::DuplicateBinding {
                name: binding.name.clone(),
                neuron: neuron.name.clone(),
            });
        }
        defined_bindings.insert(binding.name.clone());

        // Check for recursion in set: bindings (not allowed for eager instantiation)
        if binding.call_name == neuron.name {
            errors.push(ValidationError::InvalidRecursion {
                binding: binding.name.clone(),
                neuron: neuron.name.clone(),
                reason: "set: bindings cannot be recursive (use let: for lazy recursion)"
                    .to_string(),
            });
        }
    }

    // Check let: bindings (lazy instantiation)
    for binding in let_bindings {
        // Check if the neuron being called exists
        if !neuron_exists_fn(&binding.call_name, program, registry) {
            errors.push(ValidationError::MissingNeuron {
                name: binding.call_name.clone(),
                context: format!("let binding '{}' in neuron '{}'", binding.name, neuron.name),
            });
        }

        // Check for duplicate binding names
        if defined_bindings.contains(&binding.name) {
            errors.push(ValidationError::DuplicateBinding {
                name: binding.name.clone(),
                neuron: neuron.name.clone(),
            });
        }
        defined_bindings.insert(binding.name.clone());

        // Allow recursion in let: bindings (lazy instantiation)
        // but warn if it's the same neuron without parameters (infinite recursion)
        if binding.call_name == neuron.name {
            // Check if there are parameters that could control recursion
            if binding.args.is_empty() && binding.kwargs.is_empty() {
                errors.push(ValidationError::InvalidRecursion {
                    binding: binding.name.clone(),
                    neuron: neuron.name.clone(),
                    reason:
                        "let: binding to self without arguments may cause infinite recursion"
                            .to_string(),
                });
            }
            // Otherwise, we allow it and trust the user to have termination conditions
        }
    }

    errors
}
