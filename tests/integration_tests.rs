//! Integration tests for NeuroScript with snapshot testing
//!
//! This module contains comprehensive snapshot tests for:
//! - Parser IR output (complete AST structures)
//! - Codegen output (generated PyTorch code)
//! - Error messages (formatted diagnostics)
//! - Validation results
//!
//! Snapshots are stored in tests/snapshots/ and managed by the `insta` crate.

use neuroscript::{parse, validate, generate_pytorch};
use neuroscript::interfaces::*;
use std::fs;
use std::path::PathBuf;

/// Format a Program IR for snapshot testing
///
/// This provides a cleaner, more readable format than Debug output by:
/// - Omitting spans and source text (unstable across runs)
/// - Pretty-printing nested structures
/// - Focusing on semantic content
fn format_program_ir(program: &Program) -> String {
    let mut output = String::new();

    // Format use statements
    if !program.uses.is_empty() {
        output.push_str("=== Uses ===\n");
        for use_stmt in &program.uses {
            output.push_str(&format!("use {} (source: {})\n",
                use_stmt.path.join("::"), use_stmt.source));
        }
        output.push('\n');
    }

    // Format neurons
    output.push_str("=== Neurons ===\n");
    let mut neuron_names: Vec<_> = program.neurons.keys().collect();
    neuron_names.sort();

    for name in neuron_names {
        let neuron = &program.neurons[name];
        output.push_str(&format!("\n{}", format_neuron(neuron)));
    }

    output
}

fn format_neuron(neuron: &NeuronDef) -> String {
    let mut output = String::new();

    // Neuron name and parameters
    output.push_str(&format!("neuron {}", neuron.name));
    if !neuron.params.is_empty() {
        output.push('(');
        let param_strs: Vec<String> = neuron.params.iter().map(|p| {
            if let Some(default) = &p.default {
                format!("{} = {}", p.name, format_value(default))
            } else {
                p.name.clone()
            }
        }).collect();
        output.push_str(&param_strs.join(", "));
        output.push(')');
    }
    output.push_str(":\n");

    // Input ports
    if !neuron.inputs.is_empty() {
        output.push_str("  inputs:\n");
        for port in &neuron.inputs {
            output.push_str(&format!("    {}: {}\n", port.name, format_shape(&port.shape)));
        }
    }

    // Output ports
    if !neuron.outputs.is_empty() {
        output.push_str("  outputs:\n");
        for port in &neuron.outputs {
            output.push_str(&format!("    {}: {}\n", port.name, format_shape(&port.shape)));
        }
    }

    // Body
    match &neuron.body {
        NeuronBody::Primitive(impl_ref) => {
            output.push_str("  impl:\n");
            output.push_str(&format!("    {}\n", format_impl_ref(impl_ref)));
        }
        NeuronBody::Graph { let_bindings, set_bindings, connections } => {
            if !let_bindings.is_empty() {
                output.push_str("  let:\n");
                for binding in let_bindings {
                    output.push_str(&format!("    {} = {}(...)\n", binding.name, binding.call_name));
                }
            }
            if !set_bindings.is_empty() {
                output.push_str("  set:\n");
                for binding in set_bindings {
                    output.push_str(&format!("    {} = {}(...)\n", binding.name, binding.call_name));
                }
            }
            output.push_str("  graph:\n");
            for conn in connections {
                output.push_str(&format!("    {}\n", format_connection(conn)));
            }
        }
    }

    output
}

fn format_shape(shape: &Shape) -> String {
    let dims: Vec<String> = shape.dims.iter().map(|dim| match dim {
        Dim::Literal(n) => n.to_string(),
        Dim::Named(name) => name.clone(),
        Dim::Wildcard => "*".to_string(),
        Dim::Variadic(name) => format!("*{}", name),
        Dim::Expr(expr) => format_dim_expr(expr),
    }).collect();

    format!("[{}]", dims.join(", "))
}

fn format_dim_expr(expr: &DimExpr) -> String {
    format!("({} {} {})",
        format_dim(&expr.left),
        format_binop(&expr.op),
        format_dim(&expr.right))
}

fn format_dim(dim: &Dim) -> String {
    match dim {
        Dim::Literal(n) => n.to_string(),
        Dim::Named(name) => name.clone(),
        Dim::Wildcard => "*".to_string(),
        Dim::Variadic(name) => format!("*{}", name),
        Dim::Expr(expr) => format_dim_expr(expr),
    }
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => format!("\"{}\"", s),
        Value::Bool(b) => b.to_string(),
        Value::Name(name) => name.clone(),
        Value::BinOp { op, left, right } => {
            format!("({} {} {})", format_value(left), format_binop(op), format_value(right))
        }
        Value::Call { name, args, kwargs } => {
            let mut result = name.clone();
            result.push('(');
            let mut params = Vec::new();
            params.extend(args.iter().map(format_value));
            params.extend(kwargs.iter().map(|(k, v)| format!("{}={}", k, format_value(v))));
            result.push_str(&params.join(", "));
            result.push(')');
            result
        }
    }
}

fn format_binop(op: &BinOp) -> &str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
    }
}

fn format_impl_ref(impl_ref: &ImplRef) -> String {
    match impl_ref {
        ImplRef::External { kwargs } => {
            if kwargs.is_empty() {
                "External".to_string()
            } else {
                let kwargs_str: Vec<String> = kwargs.iter()
                    .map(|(k, v)| format!("{}={}", k, format_value(v)))
                    .collect();
                format!("External with kwargs: {}", kwargs_str.join(", "))
            }
        }
        ImplRef::Source { source, path } => {
            format!("Source: {},{}", source, path)
        }
    }
}

fn format_connection(conn: &Connection) -> String {
    format!("{} -> {}", format_endpoint(&conn.source), format_endpoint(&conn.destination))
}

fn format_endpoint(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Ref(port_ref) => format_port_ref(port_ref),
        Endpoint::Tuple(port_refs) => {
            let items: Vec<String> = port_refs.iter().map(format_port_ref).collect();
            format!("({})", items.join(", "))
        }
        Endpoint::Call { name, args, kwargs, id } => {
            let mut result = format!("{}#{}", name, id);
            if !args.is_empty() || !kwargs.is_empty() {
                result.push('(');
                let mut params = Vec::new();
                params.extend(args.iter().map(format_value));
                params.extend(kwargs.iter().map(|(k, v)| format!("{}={}", k, format_value(v))));
                result.push_str(&params.join(", "));
                result.push(')');
            } else {
                result.push_str("()");
            }
            result
        }
        Endpoint::Match(match_expr) => {
            let mut result = String::from("match:\n");
            for arm in &match_expr.arms {
                result.push_str(&format!("      {}", format_shape(&arm.pattern)));
                if let Some(guard) = &arm.guard {
                    result.push_str(&format!(" where {}", format_value(guard)));
                }
                result.push_str(": ");
                // Format pipeline
                let pipeline_str: Vec<String> = arm.pipeline.iter()
                    .map(format_endpoint)
                    .collect();
                result.push_str(&pipeline_str.join(" -> "));
                if !arm.is_reachable {
                    result.push_str(" [UNREACHABLE]");
                }
                result.push('\n');
            }
            result.trim_end().to_string()
        }
    }
}

fn format_port_ref(port_ref: &PortRef) -> String {
    if port_ref.port == "default" {
        port_ref.node.clone()
    } else {
        format!("{}.{}", port_ref.node, port_ref.port)
    }
}

/// Get all .ns files from examples/ and stdlib/ directories
fn get_test_files() -> Vec<PathBuf> {
    let mut files = Vec::new();

    // Collect examples
    if let Ok(entries) = fs::read_dir("examples") {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "ns" {
                    files.push(entry.path());
                }
            }
        }
    }

    // Collect stdlib
    if let Ok(entries) = fs::read_dir("stdlib") {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "ns" {
                    files.push(entry.path());
                }
            }
        }
    }

    files.sort();
    files
}

// ============================================================================
// Parser IR Snapshot Tests
// ============================================================================

#[test]
fn snapshot_parser_ir_residual() {
    let source = fs::read_to_string("examples/residual.ns")
        .expect("Failed to read residual.ns");

    let program = parse(&source).expect("Parse failed");
    let formatted = format_program_ir(&program);

    insta::assert_snapshot!("parser_ir_residual", formatted);
}

// TODO: Re-enable when 'let' bindings are supported
// #[test]
// fn snapshot_parser_ir_transformer() {
//     let source = fs::read_to_string("examples/transformer_from_stdlib.ns")
//         .expect("Failed to read transformer_from_stdlib.ns");
//
//     let program = parse(&source).expect("Parse failed");
//     let formatted = format_program_ir(&program);
//
//     insta::assert_snapshot!("parser_ir_transformer", formatted);
// }

#[test]
fn snapshot_parser_ir_match_basic() {
    let source = fs::read_to_string("examples/10-match.ns")
        .expect("Failed to read 10-match.ns");

    let program = parse(&source).expect("Parse failed");
    let formatted = format_program_ir(&program);

    insta::assert_snapshot!("parser_ir_match_basic", formatted);
}

#[test]
fn snapshot_parser_ir_match_dimension_binding() {
    let source = fs::read_to_string("examples/17-match-dimension-binding.ns")
        .expect("Failed to read 17-match-dimension-binding.ns");

    let program = parse(&source).expect("Parse failed");
    let formatted = format_program_ir(&program);

    insta::assert_snapshot!("parser_ir_match_dimension_binding", formatted);
}

#[test]
fn snapshot_parser_ir_ffn_stdlib() {
    let source = fs::read_to_string("stdlib/FFN.ns")
        .expect("Failed to read FFN.ns");

    let program = parse(&source).expect("Parse failed");
    let formatted = format_program_ir(&program);

    insta::assert_snapshot!("parser_ir_ffn_stdlib", formatted);
}

// ============================================================================
// Codegen Output Snapshot Tests
// ============================================================================

#[test]
fn snapshot_codegen_simple_linear() {
    let source = r#"
use core,nn/*

neuron SimpleLinear:
    in: [*, 512]
    out: [*, 256]
    graph:
        in -> Linear(512, 256) -> out

neuron Linear(in_dim, out_dim):
    in: [*, in_dim]
    out: [*, out_dim]
    impl: core,nn/Linear
"#;

    let mut program = parse(source).expect("Parse failed");
    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "SimpleLinear")
        .expect("Codegen failed");

    insta::assert_snapshot!("codegen_simple_linear", code);
}

// TODO: Re-enable when match expression validation allows 'out' in pipelines
// This test has valid syntax but fails validation due to current limitations
// See examples/17-match-dimension-binding.ns for working match expressions
// #[test]
// fn snapshot_codegen_match_with_guards() {
//     let source = r#"
// use core,nn/*
//
// neuron AdaptiveProjection:
//     in: [*, d]
//     out: [*, 512]
//     graph:
//         in -> match:
//             [*, dim] where dim > 512: Linear(dim, 512) -> out
//             [*, dim] where dim < 512: Linear(dim, 512) -> out
//             [*, d]: Identity() -> out
//
// neuron Linear(in_dim, out_dim):
//     in: [*, in_dim]
//     out: [*, out_dim]
//     impl: core,nn/Linear
//
// neuron Identity:
//     in: [*shape]
//     out: [*shape]
//     impl: core,nn/Identity
// "#;
//
//     let mut program = parse(source).expect("Parse failed");
//     validate(&mut program).expect("Validation failed");
//
//     let code = generate_pytorch(&program, "AdaptiveProjection")
//         .expect("Codegen failed");
//
//     insta::assert_snapshot!("codegen_match_with_guards", code);
// }

#[test]
fn snapshot_codegen_residual_block() {
    let source = fs::read_to_string("examples/residual.ns")
        .expect("Failed to read residual.ns");

    let mut program = parse(&source).expect("Parse failed");
    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "Residual")
        .expect("Codegen failed");

    insta::assert_snapshot!("codegen_residual_block", code);
}

// ============================================================================
// Error Message Snapshot Tests
// ============================================================================

#[test]
fn snapshot_error_missing_neuron() {
    let source = r#"
neuron Test:
    graph:
        in -> MissingNeuron() -> out
"#;

    let mut program = parse(source).expect("Parse should succeed");
    let errors = validate(&mut program).unwrap_err();

    let error_text = errors.iter()
        .map(|e| format!("{:?}", e))
        .collect::<Vec<_>>()
        .join("\n\n");

    insta::assert_snapshot!("error_missing_neuron", error_text);
}

#[test]
fn snapshot_error_arity_mismatch() {
    let source = r#"
neuron Test:
    graph:
        in -> TwoOutputs() -> (a, b, c)
        a -> out

neuron TwoOutputs:
    out left: [*]
    out right: [*]
    impl: test,test/TwoOutputs
"#;

    let mut program = parse(source).expect("Parse should succeed");
    let errors = validate(&mut program).unwrap_err();

    let error_text = errors.iter()
        .map(|e| format!("{:?}", e))
        .collect::<Vec<_>>()
        .join("\n\n");

    insta::assert_snapshot!("error_arity_mismatch", error_text);
}

#[test]
fn snapshot_error_parse_failure() {
    let source = r#"
neuron Test
    graph
        in -> out
"#;

    let result = parse(source);

    let error_text = match result {
        Ok(_) => "Expected parse error but succeeded".to_string(),
        Err(e) => format!("{:?}", e),
    };

    insta::assert_snapshot!("error_parse_failure", error_text);
}

// ============================================================================
// Comprehensive Snapshot Tests (iterate all examples)
// ============================================================================

/// Test that generates snapshots for all example files
///
/// This creates individual snapshots for each .ns file to track changes
/// across the entire test suite.
#[test]
fn snapshot_all_examples() {
    let files = get_test_files();

    for file_path in files {
        let file_name = file_path.file_name().unwrap().to_str().unwrap();

        // Skip if we already have a specific test for this file
        if file_name == "residual.ns" || file_name == "10-match.ns" ||
           file_name == "17-match-dimension-binding.ns" || file_name == "FFN.ns" {
            continue;
        }

        let source = fs::read_to_string(&file_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", file_name));

        match parse(&source) {
            Ok(program) => {
                let formatted = format_program_ir(&program);
                let snapshot_name = format!("example_{}", file_name.replace(".ns", ""));
                insta::assert_snapshot!(snapshot_name, formatted);
            }
            Err(e) => {
                // Some examples might be intentionally invalid for error testing
                let error_text = format!("Parse error: {:?}", e);
                let snapshot_name = format!("example_{}_error", file_name.replace(".ns", ""));
                insta::assert_snapshot!(snapshot_name, error_text);
            }
        }
    }
}
