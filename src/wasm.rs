use crate::interfaces::{Endpoint, NeuronBody, Program};
use crate::{generate_pytorch, parse, validate};
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn compile(source: &str) -> Result<String, String> {
    let mut program = parse(source).map_err(|e| format!("{}", e))?;

    validate(&mut program).map_err(|errors| {
        errors
            .iter()
            .map(|e| format!("{}", e))
            .collect::<Vec<_>>()
            .join("\n")
    })?;

    let entry_point = find_entry_point(&program).ok_or("No suitable entry point neuron found")?;

    let py_code = generate_pytorch(&program, &entry_point).map_err(|e| format!("{}", e))?;

    Ok(py_code)
}

#[wasm_bindgen]
pub fn validate_source(source: &str) -> Result<(), String> {
    let mut program = parse(source).map_err(|e| format!("{}", e))?;
    validate(&mut program).map_err(|errors| {
        errors
            .iter()
            .map(|e| format!("{}", e))
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn find_entry_point(program: &Program) -> Option<String> {
    let preferred = ["Model", "Main", "GPT", "Transformer"];
    for name in preferred {
        if program.neurons.contains_key(name) {
            return Some(name.to_string());
        }
    }

    let mut called = HashSet::new();
    for neuron in program.neurons.values() {
        if let NeuronBody::Graph {
            context_bindings,
            connections,
            ..
        } = &neuron.body
        {
            for binding in context_bindings {
                called.insert(binding.call_name.clone());
            }
            for conn in connections {
                collect_calls(&conn.source, &mut called);
                collect_calls(&conn.destination, &mut called);
            }
        }
    }

    let mut roots: Vec<String> = program
        .neurons
        .keys()
        .filter(|name| !called.contains(*name))
        .filter(|name| !program.neurons.get(*name).unwrap().is_primitive())
        .cloned()
        .collect();

    roots.sort();

    if !roots.is_empty() {
        return Some(roots.last().unwrap().clone());
    }

    program
        .neurons
        .iter()
        .filter(|(_, n)| !n.is_primitive())
        .map(|(n, _)| n.clone())
        .next()
}

fn collect_calls(endpoint: &Endpoint, called: &mut HashSet<String>) {
    match endpoint {
        Endpoint::Call { name, .. } => {
            called.insert(name.clone());
        }
        Endpoint::Tuple(_) => {}
        Endpoint::Match(expr) => {
            for arm in &expr.arms {
                for ep in &arm.pipeline {
                    collect_calls(ep, called);
                }
            }
        }
        Endpoint::If(expr) => {
            for branch in &expr.branches {
                for ep in &branch.pipeline {
                    collect_calls(ep, called);
                }
            }
            if let Some(else_branch) = &expr.else_branch {
                for ep in else_branch {
                    collect_calls(ep, called);
                }
            }
        }
        Endpoint::Ref(_) => {}
    }
}
