//! Module instantiation logic for __init__ method generation
//!
//! This module handles generating module instantiations in the __init__ method,
//! including deduplication of calls and lazy instantiation for modules with
//! captured dimensions.

use super::utils::*;
use crate::interfaces::*;
use std::collections::HashMap;
use std::fmt::Write;

/// Generate module instantiations in __init__
pub(super) fn generate_module_instantiations(
    gen: &mut CodeGenerator,
    output: &mut String,
    let_bindings: &[Binding],
    set_bindings: &[Binding],
    connections: &[Connection],
) -> Result<(), CodegenError> {
    // First, generate set: bindings (eager instantiation)
    for binding in set_bindings {
        let module_name = binding.name.clone();
        let name = &binding.call_name;
        let args = &binding.args;
        let kwargs = &binding.kwargs;

        // Check if this is a primitive
        let is_primitive = if let Some(neuron) = gen.program.neurons.get(name.as_str()) {
            neuron.is_primitive()
        } else {
            true // Assume primitive if not in program
        };

        if is_primitive {
            gen.used_primitives.insert(name.clone());
        }

        // Generate instantiation for set binding
        let args_str = args
            .iter()
            .map(|v| value_to_python_impl(v))
            .collect::<Vec<_>>()
            .join(", ");

        let kwargs_str = if kwargs.is_empty() {
            String::new()
        } else {
            let kw: Vec<String> = kwargs
                .iter()
                .map(|(k, v)| format!("{}={}", k, value_to_python_impl(v)))
                .collect();
            if args.is_empty() {
                kw.join(", ")
            } else {
                format!(", {}", kw.join(", "))
            }
        };

        writeln!(
            output,
            "        self.{} = {}({}{})",
            module_name, name, args_str, kwargs_str
        )
        .unwrap();

        // Register the binding name as a variable for use in graph
        gen.var_names
            .insert(module_name.clone(), format!("self.{}", module_name));
    }

    // TODO: Handle let: bindings - these need lazy instantiation
    // For now, we'll mark them for lazy instantiation
    for binding in let_bindings {
        let module_name = binding.name.clone();
        let name = &binding.call_name;

        // Check if this is a primitive
        let is_primitive = if let Some(neuron) = gen.program.neurons.get(name.as_str()) {
            neuron.is_primitive()
        } else {
            true // Assume primitive if not in program
        };

        if is_primitive {
            gen.used_primitives.insert(name.clone());
        }

        // Initialize as None for lazy instantiation
        writeln!(
            output,
            "        self._{} = None  # Lazy instantiation (let binding)",
            binding.name
        )
        .unwrap();

        // Store binding info for lazy instantiation in forward()
        gen.lazy_bindings.insert(
            module_name.clone(),
            (name.clone(), binding.args.clone(), binding.kwargs.clone()),
        );
        gen.var_names
            .insert(module_name.clone(), format!("self._{}", module_name));
    }

    // Collect all unique Call endpoints and assign them IDs
    let mut seen_calls: HashMap<String, (String, String, Vec<Value>, Vec<(String, Value)>)> =
        HashMap::new();
    let mut all_endpoints = Vec::new();
    collect_calls_impl(connections, &mut all_endpoints);

    for endpoint in &all_endpoints {
        if let Endpoint::Call {
            name, args, kwargs, ..
        } = endpoint
        {
            let key = endpoint_key_impl(&endpoint);
            if !seen_calls.contains_key(&key) {
                let id = gen.next_node_id();
                let module_name = format!("{}_{}", snake_case_impl(&name), id);
                // Store the mapping for use in forward generation
                gen.call_to_module.insert(key.clone(), module_name.clone());
                seen_calls.insert(
                    key,
                    (name.clone(), module_name, args.clone(), kwargs.clone()),
                );
            }
        }
    }

    // Generate instantiations in deterministic order
    let mut calls: Vec<_> = seen_calls.into_iter().collect();
    calls.sort_by(|a, b| a.1 .1.cmp(&b.1 .1)); // Sort by module_name for determinism

    let mut instantiated_count = 0;
    for (_key, (name, module_name, args, kwargs)) in &calls {
        // Check if any arguments contain captured dimensions
        let has_captured = args
            .iter()
            .any(|v| has_captured_dimensions_impl(v, &gen.current_neuron_params))
            || kwargs
                .iter()
                .any(|(_, v)| has_captured_dimensions_impl(v, &gen.current_neuron_params));

        if has_captured {
            // Skip instantiation in __init__ for modules with captured dimensions
            // They will be instantiated lazily in forward()
            // Initialize cache variable to None
            writeln!(
                output,
                "        self._{} = None  # Lazy instantiation (has captured dimensions)",
                module_name
            )
            .unwrap();
            instantiated_count += 1;
            continue;
        }

        // Check if this is a primitive
        let is_primitive = if let Some(neuron) = gen.program.neurons.get(name.as_str()) {
            neuron.is_primitive()
        } else {
            // Assume it's a primitive if not in program
            true
        };

        if is_primitive {
            gen.used_primitives.insert(name.clone());
        }

        // Generate instantiation
        let args_str = args
            .iter()
            .map(|v| value_to_python_impl(v))
            .collect::<Vec<_>>()
            .join(", ");

        let kwargs_str = if kwargs.is_empty() {
            String::new()
        } else {
            let kw: Vec<String> = kwargs
                .iter()
                .map(|(k, v)| format!("{}={}", k, value_to_python_impl(v)))
                .collect();
            if args.is_empty() {
                kw.join(", ")
            } else {
                format!(", {}", kw.join(", "))
            }
        };

        writeln!(
            output,
            "        self.{} = {}({}{})",
            module_name, name, args_str, kwargs_str
        )
        .unwrap();
        instantiated_count += 1;
    }

    // If no modules were instantiated, add pass
    if instantiated_count == 0 {
        writeln!(output, "        pass").unwrap();
    }

    Ok(())
}

#[cfg(test)]
mod tests;
