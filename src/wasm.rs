use crate::interfaces::{Dim, Endpoint, NeuronBody, Program, Shape};
use crate::{generate_pytorch, parse, stdlib, validate};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use wasm_bindgen::prelude::*;

/// Load embedded stdlib and merge with user program.
/// User neurons take precedence over stdlib neurons.
fn load_and_merge(user_program: Program) -> Result<Program, String> {
    let stdlib_program = stdlib::load_stdlib_embedded()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;
    Ok(stdlib::merge_programs(stdlib_program, user_program))
}

/// Analysis result for the WASM API
#[derive(Serialize, Deserialize)]
pub struct AnalysisResult {
    pub neurons: Vec<NeuronInfo>,
    pub inferred_dims: HashMap<String, i64>,
    pub match_expressions: Vec<MatchExprInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct NeuronInfo {
    pub name: String,
    pub params: Vec<ParamInfo>,
    pub inputs: Vec<PortInfo>,
    pub outputs: Vec<PortInfo>,
    pub is_primitive: bool,
    pub connections: Vec<ConnectionInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct ParamInfo {
    pub name: String,
    pub default_value: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct PortInfo {
    pub name: String,
    pub shape: String,
}

#[derive(Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub source: String,
    pub destination: String,
}

#[derive(Serialize, Deserialize)]
pub struct MatchExprInfo {
    pub neuron: String,
    pub arms: Vec<MatchArmInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct MatchArmInfo {
    pub pattern: String,
    pub guard: Option<String>,
    pub is_reachable: bool,
}

#[wasm_bindgen]
pub fn compile(source: &str, neuron_name: Option<String>) -> Result<String, String> {
    let user_program = parse(source).map_err(|e| format!("{}", e))?;
    let mut program = load_and_merge(user_program)?;

    validate(&mut program).map_err(|errors| {
        errors
            .iter()
            .map(|e| format!("{}", e))
            .collect::<Vec<_>>()
            .join("\n")
    })?;

    // Use provided name or fall back to heuristic
    let entry_point = if let Some(name) = neuron_name {
        if program.neurons.contains_key(&name) {
            name
        } else {
            return Err(format!("Neuron '{}' not found in program", name));
        }
    } else {
        find_entry_point(&program).ok_or("No suitable entry point neuron found")?
    };

    let py_code = generate_pytorch(&program, &entry_point).map_err(|e| format!("{}", e))?;

    Ok(py_code)
}

#[wasm_bindgen]
pub fn validate_source(source: &str) -> Result<(), String> {
    let user_program = parse(source).map_err(|e| format!("{}", e))?;
    let mut program = load_and_merge(user_program)?;
    validate(&mut program).map_err(|errors| {
        errors
            .iter()
            .map(|e| format!("{}", e))
            .collect::<Vec<_>>()
            .join("\n")
    })
}

/// List all non-primitive neurons in a program
#[wasm_bindgen]
pub fn list_neurons(source: &str) -> Result<String, String> {
    let program = parse(source).map_err(|e| format!("{}", e))?;
    let neurons: Vec<String> = program
        .neurons
        .iter()
        .filter(|(_, neuron)| !neuron.is_primitive())
        .map(|(name, _)| name.clone())
        .collect();
    serde_json::to_string(&neurons).map_err(|e| format!("JSON serialization error: {}", e))
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
        Endpoint::Unroll(_) => {}
    }
}

/// Analyze a NeuroScript program and return structured information
/// about neurons, shapes, and match expressions
#[wasm_bindgen]
pub fn analyze(source: &str) -> Result<String, String> {
    let user_program = parse(source).map_err(|e| format!("{}", e))?;
    let mut program = load_and_merge(user_program)?;

    validate(&mut program).map_err(|errors| {
        errors
            .iter()
            .map(|e| format!("{}", e))
            .collect::<Vec<_>>()
            .join("\n")
    })?;

    let mut result = AnalysisResult {
        neurons: Vec::new(),
        inferred_dims: HashMap::new(),
        match_expressions: Vec::new(),
    };

    // Extract neuron information
    let mut neuron_names: Vec<_> = program.neurons.keys().cloned().collect();
    neuron_names.sort();

    for name in neuron_names {
        let neuron = program.neurons.get(&name).unwrap();

        let params: Vec<ParamInfo> = neuron
            .params
            .iter()
            .map(|p| ParamInfo {
                name: p.name.clone(),
                default_value: p.default.as_ref().map(|v| format_value(v)),
            })
            .collect();

        let inputs: Vec<PortInfo> = neuron
            .inputs
            .iter()
            .map(|p| PortInfo {
                name: p.name.clone(),
                shape: format_shape(&p.shape),
            })
            .collect();

        let outputs: Vec<PortInfo> = neuron
            .outputs
            .iter()
            .map(|p| PortInfo {
                name: p.name.clone(),
                shape: format_shape(&p.shape),
            })
            .collect();

        let is_primitive = neuron.is_primitive();

        let connections = match &neuron.body {
            NeuronBody::Graph { connections, .. } => connections
                .iter()
                .map(|c| ConnectionInfo {
                    source: format_endpoint(&c.source),
                    destination: format_endpoint(&c.destination),
                })
                .collect(),
            NeuronBody::Primitive(_) => Vec::new(),
        };

        // Extract match expressions from this neuron
        if let NeuronBody::Graph { connections, .. } = &neuron.body {
            for conn in connections {
                collect_match_exprs(&name, &conn.source, &mut result.match_expressions);
                collect_match_exprs(&name, &conn.destination, &mut result.match_expressions);
            }
        }

        result.neurons.push(NeuronInfo {
            name,
            params,
            inputs,
            outputs,
            is_primitive,
            connections,
        });
    }

    serde_json::to_string(&result).map_err(|e| format!("JSON serialization error: {}", e))
}

fn format_shape(shape: &Shape) -> String {
    let dims: Vec<String> = shape.dims.iter().map(format_dim).collect();
    format!("[{}]", dims.join(", "))
}

fn format_dim(dim: &Dim) -> String {
    match dim {
        Dim::Literal(n) => n.to_string(),
        Dim::Named(name) => name.clone(),
        Dim::Wildcard => "*".to_string(),
        Dim::Variadic(name) => format!("*{}", name),
        Dim::Expr(expr) => {
            let op = match expr.op {
                crate::interfaces::BinOp::Add => "+",
                crate::interfaces::BinOp::Sub => "-",
                crate::interfaces::BinOp::Mul => "*",
                crate::interfaces::BinOp::Div => "/",
                crate::interfaces::BinOp::Lt => "<",
                crate::interfaces::BinOp::Gt => ">",
                crate::interfaces::BinOp::Le => "<=",
                crate::interfaces::BinOp::Ge => ">=",
                crate::interfaces::BinOp::Eq => "==",
                crate::interfaces::BinOp::Ne => "!=",
            };
            format!("{} {} {}", format_dim(&expr.left), op, format_dim(&expr.right))
        }
        Dim::Global(name) => format!("@{}", name),
    }
}

fn format_value(value: &crate::interfaces::Value) -> String {
    match value {
        crate::interfaces::Value::Int(n) => n.to_string(),
        crate::interfaces::Value::Float(f) => f.to_string(),
        crate::interfaces::Value::String(s) => format!("\"{}\"", s),
        crate::interfaces::Value::Bool(b) => b.to_string(),
        crate::interfaces::Value::Name(name) => name.clone(),
        crate::interfaces::Value::Global(name) => format!("@{}", name),
        crate::interfaces::Value::BinOp { op, left, right } => {
            let op_str = match op {
                crate::interfaces::BinOp::Add => "+",
                crate::interfaces::BinOp::Sub => "-",
                crate::interfaces::BinOp::Mul => "*",
                crate::interfaces::BinOp::Div => "/",
                crate::interfaces::BinOp::Lt => "<",
                crate::interfaces::BinOp::Gt => ">",
                crate::interfaces::BinOp::Le => "<=",
                crate::interfaces::BinOp::Ge => ">=",
                crate::interfaces::BinOp::Eq => "==",
                crate::interfaces::BinOp::Ne => "!=",
            };
            format!("{} {} {}", format_value(left), op_str, format_value(right))
        }
        crate::interfaces::Value::Call { name, args, kwargs } => {
            let args_str: Vec<String> = args.iter().map(format_value).collect();
            let kwargs_str: Vec<String> = kwargs
                .iter()
                .map(|(k, v)| format!("{}={}", k, format_value(v)))
                .collect();
            let all_args = [args_str, kwargs_str].concat().join(", ");
            format!("{}({})", name, all_args)
        }
    }
}

fn format_endpoint(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Ref(port_ref) => {
            if port_ref.port == "default" {
                port_ref.node.clone()
            } else {
                format!("{}.{}", port_ref.node, port_ref.port)
            }
        }
        Endpoint::Tuple(refs) => {
            let names: Vec<String> = refs
                .iter()
                .map(|r| {
                    if r.port == "default" {
                        r.node.clone()
                    } else {
                        format!("{}.{}", r.node, r.port)
                    }
                })
                .collect();
            format!("({})", names.join(", "))
        }
        Endpoint::Call { name, args, kwargs, .. } => {
            let args_str: Vec<String> = args.iter().map(format_value).collect();
            let kwargs_str: Vec<String> = kwargs
                .iter()
                .map(|(k, v)| format!("{}={}", k, format_value(v)))
                .collect();
            let all_args = [args_str, kwargs_str].concat().join(", ");
            format!("{}({})", name, all_args)
        }
        Endpoint::Match(_) => "match { ... }".to_string(),
        Endpoint::If(_) => "if { ... }".to_string(),
        Endpoint::Unroll(_) => "unroll { ... }".to_string(),
    }
}

fn collect_match_exprs(neuron_name: &str, endpoint: &Endpoint, result: &mut Vec<MatchExprInfo>) {
    match endpoint {
        Endpoint::Match(expr) => {
            let arms: Vec<MatchArmInfo> = expr
                .arms
                .iter()
                .map(|arm| MatchArmInfo {
                    pattern: format_shape(&arm.pattern),
                    guard: arm.guard.as_ref().map(format_value),
                    is_reachable: arm.is_reachable,
                })
                .collect();

            result.push(MatchExprInfo {
                neuron: neuron_name.to_string(),
                arms,
            });

            // Recurse into arm pipelines
            for arm in &expr.arms {
                for ep in &arm.pipeline {
                    collect_match_exprs(neuron_name, ep, result);
                }
            }
        }
        Endpoint::If(expr) => {
            for branch in &expr.branches {
                for ep in &branch.pipeline {
                    collect_match_exprs(neuron_name, ep, result);
                }
            }
            if let Some(else_branch) = &expr.else_branch {
                for ep in else_branch {
                    collect_match_exprs(neuron_name, ep, result);
                }
            }
        }
        Endpoint::Tuple(_) | Endpoint::Call { .. } | Endpoint::Ref(_) | Endpoint::Unroll(_) => {}
    }
}
