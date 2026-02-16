//! Module instantiation logic for __init__ method generation
//!
//! This module handles generating module instantiations in the __init__ method,
//! including deduplication of calls and lazy instantiation for modules with
//! captured dimensions.

use super::utils::*;
use crate::interfaces::Kwarg;
use crate::interfaces::*;
use std::collections::{hash_map, HashMap};
use std::fmt::Write;

/// Generate module instantiations in __init__
pub(super) fn generate_module_instantiations(
    gen: &mut CodeGenerator,
    output: &mut String,
    context_bindings: &[Binding],
    connections: &[Connection],
) -> Result<(), CodegenError> {
    let mut instantiated_count = 0;

    // Partition bindings into standalone and unroll groups
    // Collect unroll groups: base_name -> (call_name, args, kwargs, count, members, aggregate_name, scope)
    let mut unroll_groups: HashMap<String, (String, Vec<Value>, Vec<Kwarg>, Value, Vec<(String, usize)>, String, Scope)> =
        HashMap::new();
    let mut standalone_bindings: Vec<&Binding> = Vec::new();

    for binding in context_bindings {
        // Track binding_to_call_name for all bindings
        gen.binding_to_call_name
            .insert(binding.name.clone(), binding.call_name.clone());

        if let Some(ref group_info) = binding.unroll_group {
            // Track unroll group info
            gen.binding_to_unroll_group
                .insert(binding.name.clone(), group_info.clone());

            let entry = unroll_groups
                .entry(group_info.base_name.clone())
                .or_insert_with(|| {
                    (
                        binding.call_name.clone(),
                        binding.args.clone(),
                        binding.kwargs.clone(),
                        group_info.count.clone(),
                        Vec::new(),
                        group_info.aggregate_name.clone(),
                        binding.scope.clone(),
                    )
                });
            entry.4.push((binding.name.clone(), group_info.index));
        } else {
            standalone_bindings.push(binding);
        }
    }

    // Sort each unroll group's members by index
    for (_base, group) in unroll_groups.iter_mut() {
        group.4.sort_by_key(|(_, idx)| *idx);
    }

    // 1. Process standalone bindings (non-unrolled)
    for binding in &standalone_bindings {
        let module_name = binding.name.clone();
        let name = &binding.call_name;
        let args = &binding.args;
        let kwargs = &binding.kwargs;

        let is_primitive = if let Some(neuron) = gen.program.neurons.get(name.as_str()) {
            neuron.is_primitive()
        } else {
            true // Assume primitive if not in program
        };

        if is_primitive {
            gen.used_primitives.insert(name.clone());
        }

        match &binding.scope {
            Scope::Static => {
                let (args_str, kwargs_str) = extract_kwargs(args, kwargs);

                writeln!(
                    output,
                    "        if not hasattr(self.__class__, '{}'):",
                    module_name
                )
                .unwrap();
                writeln!(
                    output,
                    "            self.__class__.{} = {}({}{})",
                    module_name, name, args_str, kwargs_str
                )
                .unwrap();

                gen.var_names.insert(
                    module_name.clone(),
                    format!("self.__class__.{}", module_name),
                );
                instantiated_count += 1;
            }
            Scope::Instance { lazy: true } => {
                writeln!(
                    output,
                    "        self._{} = None  # Lazy instantiation (@lazy)",
                    module_name
                )
                .unwrap();

                gen.lazy_bindings.insert(
                    module_name.clone(),
                    (name.clone(), args.clone(), kwargs.clone()),
                );
                gen.var_names
                    .insert(module_name.clone(), format!("self._{}", module_name));
                instantiated_count += 1;
            }
            Scope::Instance { lazy: false } => {
                let (args_str, kwargs_str) = extract_kwargs(args, kwargs);

                writeln!(
                    output,
                    "        self.{} = {}({}{})",
                    module_name, name, args_str, kwargs_str
                )
                .unwrap();

                gen.var_names
                    .insert(module_name.clone(), format!("self.{}", module_name));
                instantiated_count += 1;
            }
            Scope::Global => {
                gen.var_names.insert(module_name.clone(), name.clone());
            }
        }
    }

    // 2. Process unroll groups
    // Sort groups by their first member's position in the original binding list
    // to maintain declaration order
    let mut sorted_groups: Vec<_> = unroll_groups.into_iter().collect();
    sorted_groups.sort_by_key(|(_, group)| group.4.first().map(|(_, idx)| *idx).unwrap_or(0));

    for (base_name, (call_name, args, kwargs, count, members, aggregate_name, scope)) in &sorted_groups {
        let is_primitive = if let Some(neuron) = gen.program.neurons.get(call_name.as_str()) {
            neuron.is_primitive()
        } else {
            true
        };

        if is_primitive {
            gen.used_primitives.insert(call_name.clone());
        }

        let (args_str, kwargs_str) = extract_kwargs(args, kwargs);
        let list_name = aggregate_name.clone();
        let is_static = matches!(scope, Scope::Static);

        // Determine the range expression
        let range_expr = match count {
            Value::Name(param_name) => param_name.clone(),
            Value::Int(n) => n.to_string(),
            _ => members.len().to_string(),
        };

        if is_static {
            // @static: single shared class-level instance called N times
            writeln!(
                output,
                "        if not hasattr(self.__class__, '{}'):",
                base_name
            )
            .unwrap();
            writeln!(
                output,
                "            self.__class__.{} = {}({}{})",
                base_name, call_name, args_str, kwargs_str
            )
            .unwrap();

            // Register the class-level var name for the base binding
            gen.var_names.insert(
                base_name.clone(),
                format!("self.__class__.{}", base_name),
            );

            // Register aggregate name pointing to (base_name, count, is_static=true)
            gen.var_names
                .insert(list_name.clone(), format!("self.__class__.{}", base_name));
        } else {
            // Instance scope: nn.ModuleList with N separate instances
            writeln!(
                output,
                "        self.{} = nn.ModuleList([",
                list_name
            )
            .unwrap();
            writeln!(
                output,
                "            {}({}{}) for _ in range({})",
                call_name, args_str, kwargs_str, range_expr
            )
            .unwrap();
            writeln!(output, "        ])").unwrap();

            // Register the ModuleList var name
            gen.var_names
                .insert(list_name.clone(), format!("self.{}", list_name));
        }

        // Register aggregate name for forward() to detect and emit for-loops
        gen.aggregate_to_group
            .insert(list_name.clone(), (base_name.clone(), count.clone(), is_static));

        // Also register each individual member so forward() can look them up
        for (member_name, _) in members {
            gen.var_names
                .insert(member_name.clone(), format!("self.{}", member_name));
        }

        instantiated_count += 1;
    }

    // 3. Collect and instantiate anonymous calls from connections
    let mut seen_calls: HashMap<String, (String, String, Vec<Value>, Vec<Kwarg>)> = HashMap::new();
    let mut all_endpoints = Vec::new();
    collect_calls_impl(connections, &mut all_endpoints);

    for endpoint in &all_endpoints {
        if let Endpoint::Call { .. } = endpoint {
            let key = endpoint_key_impl(endpoint);
            if let hash_map::Entry::Vacant(e) = seen_calls.entry(key.clone()) {
                let id = gen.next_node_id();
                let name = extract_call_name(endpoint);
                let module_name = format!("{}_{}", snake_case_impl(&name), id);
                let args = extract_call_args(endpoint);
                let kwargs = extract_call_kwargs(endpoint);

                gen.call_to_module.insert(key.clone(), module_name.clone());
                e.insert((name, module_name, args, kwargs));
            }
        }
    }

    let mut calls: Vec<_> = seen_calls.into_iter().collect();
    calls.sort_by(|a, b| a.1 .1.cmp(&b.1 .1));

    for (_key, (name, module_name, args, kwargs)) in &calls {
        let has_captured = args
            .iter()
            .any(|v| has_captured_dimensions_impl(v, &gen.current_neuron_params))
            || kwargs
                .iter()
                .any(|(_, v)| has_captured_dimensions_impl(v, &gen.current_neuron_params));

        if has_captured {
            writeln!(
                output,
                "        self._{} = None  # Lazy instantiation (captured)",
                module_name
            )
            .unwrap();
            instantiated_count += 1;
            continue;
        }

        if let Some(neuron) = gen.program.neurons.get(name.as_str()) {
            if neuron.is_primitive() {
                gen.used_primitives.insert(name.clone());
            }
        } else {
            gen.used_primitives.insert(name.clone());
        }

        let (args_str, kwargs_str) = extract_kwargs(args, kwargs);

        writeln!(
            output,
            "        self.{} = {}({}{})",
            module_name, name, args_str, kwargs_str
        )
        .unwrap();
        instantiated_count += 1;
    }

    if instantiated_count == 0 {
        writeln!(output, "        pass").unwrap();
    }

    Ok(())
}

fn extract_kwargs(args: &[Value], kwargs: &[(String, Value)]) -> (String, String) {
    let args_str = args
        .iter()
        .map(value_to_python_impl)
        .collect::<Vec<_>>()
        .join(", ");
    let kwargs_str = format_kwargs_impl(kwargs);
    (args_str, kwargs_str)
}

fn format_kwargs_impl(kwargs: &[(String, Value)]) -> String {
    if kwargs.is_empty() {
        String::new()
    } else {
        let kw: Vec<String> = kwargs
            .iter()
            .map(|(k, v)| format!("{}={}", k, value_to_python_impl(v)))
            .collect();
        format!(", {}", kw.join(", "))
    }
}

fn extract_call_name(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Call { name, .. } => name.clone(),
        _ => String::new(),
    }
}

fn extract_call_args(endpoint: &Endpoint) -> Vec<Value> {
    match endpoint {
        Endpoint::Call { args, .. } => args.clone(),
        _ => vec![],
    }
}

fn extract_call_kwargs(endpoint: &Endpoint) -> Vec<Kwarg> {
    match endpoint {
        Endpoint::Call { kwargs, .. } => kwargs.clone(),
        _ => vec![],
    }
}

#[cfg(test)]
mod tests;
