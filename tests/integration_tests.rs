//! Integration tests for NeuroScript with snapshot testing
//!
//! This module contains comprehensive snapshot tests for:
//! - Parser IR output (complete AST structures)
//! - Codegen output (generated PyTorch code)
//! - Error messages (formatted diagnostics)
//! - Validation results
//!
//! Snapshots are stored in tests/snapshots/ and managed by the `insta` crate.

use neuroscript::interfaces::*;
use neuroscript::{generate_pytorch, generate_pytorch_with_options, CodegenOptions, parse, validate};
use std::fs;
use std::path::{Path, PathBuf};

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
            output.push_str(&format!(
                "use {} (source: {})\n",
                use_stmt.path.join("::"),
                use_stmt.source
            ));
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
        let param_strs: Vec<String> = neuron
            .params
            .iter()
            .map(|p| {
                if let Some(default) = &p.default {
                    format!("{} = {}", p.name, format_value(default))
                } else {
                    p.name.clone()
                }
            })
            .collect();
        output.push_str(&param_strs.join(", "));
        output.push(')');
    }
    output.push_str(":\n");

    // Input ports
    if !neuron.inputs.is_empty() {
        output.push_str("  inputs:\n");
        for port in &neuron.inputs {
            output.push_str(&format!(
                "    {}{}: {}\n",
                if port.variadic { "*" } else { "" },
                port.name,
                format_shape(&port.shape)
            ));
        }
    }

    // Output ports
    if !neuron.outputs.is_empty() {
        output.push_str("  outputs:\n");
        for port in &neuron.outputs {
            output.push_str(&format!(
                "    {}: {}\n",
                port.name,
                format_shape(&port.shape)
            ));
        }
    }

    // Body
    match &neuron.body {
        NeuronBody::Primitive(impl_ref) => {
            output.push_str("  impl:\n");
            output.push_str(&format!("    {}\n", format_impl_ref(impl_ref)));
        }
        NeuronBody::Graph {
            context_bindings,
            connections,
            ..
        } => {
            if !context_bindings.is_empty() {
                output.push_str("  context:\n");
                for binding in context_bindings {
                    let prefix = match binding.scope {
                        Scope::Instance { lazy: true } => "@lazy ",
                        Scope::Static => "@static ",
                        Scope::Global => "@global ",
                        _ => "",
                    };
                    output.push_str(&format!(
                        "    {}{} = {}(...)\n",
                        prefix, binding.name, binding.call_name
                    ));
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
    let dims: Vec<String> = shape
        .dims
        .iter()
        .map(|dim| match dim {
            Dim::Literal(n) => n.to_string(),
            Dim::Named(name) => name.clone(),
            Dim::Wildcard => "*".to_string(),
            Dim::Inferred => "...".to_string(),
            Dim::Variadic(name) => format!("*{}", name),
            Dim::Expr(expr) => format_dim_expr(expr),
            Dim::Global(name) => format!("@global {}", name),
        })
        .collect();

    format!("[{}]", dims.join(", "))
}

fn format_dim_expr(expr: &DimExpr) -> String {
    format!(
        "({} {} {})",
        format_dim(&expr.left),
        format_binop(&expr.op),
        format_dim(&expr.right)
    )
}

fn format_dim(dim: &Dim) -> String {
    match dim {
        Dim::Literal(n) => n.to_string(),
        Dim::Named(name) => name.clone(),
        Dim::Wildcard => "*".to_string(),
        Dim::Inferred => "...".to_string(),
        Dim::Variadic(name) => format!("*{}", name),
        Dim::Expr(expr) => format_dim_expr(expr),
        Dim::Global(name) => format!("@global {}", name),
    }
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => format!("\"{}\"", s),
        Value::Bool(b) => b.to_string(),
        Value::Name(name) => name.clone(),
        Value::Global(name) => format!("@global {}", name),
        Value::BinOp { op, left, right } => {
            format!(
                "({} {} {})",
                format_value(left),
                format_binop(op),
                format_value(right)
            )
        }
        Value::Call { name, args, kwargs } => {
            let mut result = name.clone();
            result.push('(');
            let mut params = Vec::new();
            params.extend(args.iter().map(format_value));
            params.extend(
                kwargs
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, format_value(v))),
            );
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
                let kwargs_str: Vec<String> = kwargs
                    .iter()
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
    let arrow = match &conn.destination {
        Endpoint::Reshape(_) => "=>",
        _ => "->",
    };
    format!(
        "{} {} {}",
        format_endpoint(&conn.source),
        arrow,
        format_endpoint(&conn.destination)
    )
}

fn format_endpoint(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Ref(port_ref) => format_port_ref(port_ref),
        Endpoint::Tuple(port_refs) => {
            let items: Vec<String> = port_refs.iter().map(format_port_ref).collect();
            format!("({})", items.join(", "))
        }
        Endpoint::Call {
            name,
            args,
            kwargs,
            id,
            frozen: _,
        } => {
            let mut result = format!("{}#{}", name, id);
            if !args.is_empty() || !kwargs.is_empty() {
                result.push('(');
                let mut params = Vec::new();
                params.extend(args.iter().map(format_value));
                params.extend(
                    kwargs
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, format_value(v))),
                );
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
                result.push_str(&format!("      {}", match &arm.pattern {
                    MatchPattern::Shape(shape) => format_shape(shape),
                    MatchPattern::NeuronContract(contract) => format!("{:?}", contract),
                }));
                if let Some(guard) = &arm.guard {
                    result.push_str(&format!(" where {}", format_value(guard)));
                }
                result.push_str(": ");
                // Format pipeline
                let pipeline_str: Vec<String> = arm.pipeline.iter().map(format_endpoint).collect();
                result.push_str(&pipeline_str.join(" -> "));
                if !arm.is_reachable {
                    result.push_str(" [UNREACHABLE]");
                }
                result.push('\n');
            }
            result.trim_end().to_string()
        }
        Endpoint::If(if_expr) => {
            let mut result = String::from("if:\n");
            for (i, branch) in if_expr.branches.iter().enumerate() {
                let prefix = if i == 0 { "if" } else { "elif" };
                result.push_str(&format!(
                    "      {} {}: ",
                    prefix,
                    format_value(&branch.condition)
                ));
                // Format pipeline
                let pipeline_str: Vec<String> =
                    branch.pipeline.iter().map(format_endpoint).collect();
                result.push_str(&pipeline_str.join(" -> "));
                result.push('\n');
            }
            if let Some(else_branch) = &if_expr.else_branch {
                result.push_str("      else: ");
                let pipeline_str: Vec<String> = else_branch.iter().map(format_endpoint).collect();
                result.push_str(&pipeline_str.join(" -> "));
                result.push('\n');
            }
            result.trim_end().to_string()
        }
        Endpoint::Reshape(reshape) => {
            let mut result = String::new();
            if let Some(ref ann) = reshape.annotation {
                result.push_str(&format!("{} ", ann));
            }
            result.push('[');
            let dims: Vec<String> = reshape.dims.iter().map(|d| format!("{}", d)).collect();
            result.push_str(&dims.join(", "));
            result.push(']');
            result
        }
        Endpoint::Wrap(wrap_expr) => {
            let mut result = format!("@wrap({}", wrap_expr.wrapper_name);
            for arg in &wrap_expr.wrapper_args {
                result.push_str(&format!(", {}", format_value(arg)));
            }
            result.push_str("): ");
            match &wrap_expr.content {
                neuroscript::WrapContent::Ref(name) => result.push_str(name),
                neuroscript::WrapContent::Pipeline(pipeline) => {
                    result.push_str("-> ");
                    let pipeline_str: Vec<String> =
                        pipeline.iter().map(format_endpoint).collect();
                    result.push_str(&pipeline_str.join(" -> "));
                }
            }
            result
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

/// Get all .ns files from examples/ and stdlib/ directories (recursive)
fn get_test_files() -> Vec<PathBuf> {
    let mut files = Vec::new();

    fn collect_ns_files(dir: &Path, files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                // Skip __scratch and dot-prefixed directories
                if path.is_dir() {
                    if !name_str.starts_with('.') && !name_str.starts_with("__") {
                        collect_ns_files(&path, files);
                    }
                } else if let Some(ext) = path.extension() {
                    if ext == "ns" {
                        files.push(path);
                    }
                }
            }
        }
    }

    collect_ns_files(Path::new("examples"), &mut files);
    collect_ns_files(Path::new("stdlib"), &mut files);

    files.sort();
    files
}

// ============================================================================
// Parser IR Snapshot Tests
// ============================================================================

#[test]
fn snapshot_parser_ir_residual() {
    let source = fs::read_to_string("examples/codegen_demo/residual.ns").expect("Failed to read residual.ns");

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
    let source = fs::read_to_string("examples/match_inline_test.ns").expect("Failed to read match_inline_test.ns");

    let program = parse(&source).expect("Parse failed");
    let formatted = format_program_ir(&program);

    insta::assert_snapshot!("parser_ir_match_basic", formatted);
}

#[test]
fn snapshot_parser_ir_match_dimension_binding() {
    let source = fs::read_to_string("examples/codegen_demo/match_dimension_binding.ns")
        .expect("Failed to read match_dimension_binding.ns");

    let program = parse(&source).expect("Parse failed");
    let formatted = format_program_ir(&program);

    insta::assert_snapshot!("parser_ir_match_dimension_binding", formatted);
}

#[test]
fn snapshot_parser_ir_ffn_stdlib() {
    let source = fs::read_to_string("stdlib/FFN.ns").expect("Failed to read FFN.ns");

    let program = parse(&source).expect("Parse failed");
    let formatted = format_program_ir(&program);

    insta::assert_snapshot!("parser_ir_ffn_stdlib", formatted);
}

#[test]
fn test_stdlib_loading_with_primitives() {
    use neuroscript::stdlib::load_stdlib;

    // Load stdlib (should include primitives from stdlib/primitives/)
    let stdlib = load_stdlib().expect("Failed to load stdlib");

    // Verify we loaded a reasonable number of neurons
    // Should have at least the 26 primitives + stdlib neurons
    assert!(
        stdlib.neurons.len() >= 26,
        "Expected at least 26 neurons (primitives), got {}",
        stdlib.neurons.len()
    );

    // Verify key primitives are present and have proper shapes
    assert!(
        stdlib.neurons.contains_key("Linear"),
        "Linear primitive not found"
    );
    assert!(
        stdlib.neurons.contains_key("GELU"),
        "GELU primitive not found"
    );
    assert!(
        stdlib.neurons.contains_key("LayerNorm"),
        "LayerNorm primitive not found"
    );
    assert!(
        stdlib.neurons.contains_key("Dropout"),
        "Dropout primitive not found"
    );
    assert!(
        stdlib.neurons.contains_key("Fork"),
        "Fork primitive not found"
    );
    assert!(
        stdlib.neurons.contains_key("Add"),
        "Add primitive not found"
    );

    // Verify Linear has proper shape signature
    let linear = &stdlib.neurons["Linear"];
    assert_eq!(linear.inputs.len(), 1);
    assert_eq!(linear.outputs.len(), 1);
    assert_eq!(linear.inputs[0].shape.dims.len(), 2);
    assert_eq!(linear.outputs[0].shape.dims.len(), 2);

    // Verify Fork has two output ports
    let fork = &stdlib.neurons["Fork"];
    assert_eq!(fork.outputs.len(), 2);

    // Verify stdlib composite neurons are also present
    assert!(
        stdlib.neurons.contains_key("FFN"),
        "FFN stdlib neuron not found"
    );
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

    let code = generate_pytorch(&program, "SimpleLinear").expect("Codegen failed");

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
    let source = fs::read_to_string("examples/codegen_demo/residual.ns").expect("Failed to read residual.ns");

    let mut program = parse(&source).expect("Parse failed");
    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "Residual").expect("Codegen failed");

    insta::assert_snapshot!("codegen_residual_block", code);
}

#[test]
fn snapshot_codegen_cnn_demo() {
    let source = fs::read_to_string("examples/real_world/cnn_demo.ns").expect("Failed to read cnn_demo.ns");

    let mut program = parse(&source).expect("Parse failed");
    // We need to load stdlib primitives for this to pass validation
    let stdlib = neuroscript::stdlib::load_stdlib().expect("Failed to load stdlib");
    program = neuroscript::stdlib::merge_programs(stdlib, program);

    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "CNN").expect("Codegen failed");

    insta::assert_snapshot!("codegen_cnn_demo", code);
}

#[test]
fn snapshot_codegen_cnn_demo_2() {
    let source =
        fs::read_to_string("examples/real_world/cnn_demo_2.ns").expect("Failed to read cnn_demo_2.ns");

    let mut program = parse(&source).expect("Parse failed");
    // We need to load stdlib primitives for this to pass validation
    let stdlib = neuroscript::stdlib::load_stdlib().expect("Failed to load stdlib");
    program = neuroscript::stdlib::merge_programs(stdlib, program);

    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "CNN").expect("Codegen failed");

    insta::assert_snapshot!("codegen_cnn_demo_2", code);
}

// ============================================================================
// Unroll Codegen Snapshot Tests
// ============================================================================

#[test]
fn snapshot_codegen_unroll_context() {
    let source =
        fs::read_to_string("examples/unroll_context.ns").expect("Failed to read unroll_context.ns");

    let mut program = parse(&source).expect("Parse failed");
    let stdlib = neuroscript::stdlib::load_stdlib().expect("Failed to load stdlib");
    program = neuroscript::stdlib::merge_programs(stdlib, program);

    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "NamedStack").expect("Codegen failed");

    insta::assert_snapshot!("codegen_unroll_context", code);
}

#[test]
fn snapshot_codegen_unroll_gpt2() {
    let source =
        fs::read_to_string("examples/unroll_gpt2.ns").expect("Failed to read unroll_gpt2.ns");

    let mut program = parse(&source).expect("Parse failed");
    let stdlib = neuroscript::stdlib::load_stdlib().expect("Failed to load stdlib");
    program = neuroscript::stdlib::merge_programs(stdlib, program);

    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "GPT2Small").expect("Codegen failed");

    insta::assert_snapshot!("codegen_unroll_gpt2", code);
}

#[test]
fn snapshot_codegen_unroll_static() {
    let source =
        fs::read_to_string("examples/unroll_static.ns").expect("Failed to read unroll_static.ns");

    let mut program = parse(&source).expect("Parse failed");
    let stdlib = neuroscript::stdlib::load_stdlib().expect("Failed to load stdlib");
    program = neuroscript::stdlib::merge_programs(stdlib, program);

    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "SharedLayers").expect("Codegen failed");

    insta::assert_snapshot!("codegen_unroll_static", code);
}

#[test]
fn snapshot_codegen_unroll_threaded() {
    let source = fs::read_to_string("examples/unroll_threaded.ns")
        .expect("Failed to read unroll_threaded.ns");

    let mut program = parse(&source).expect("Parse failed");
    let stdlib = neuroscript::stdlib::load_stdlib().expect("Failed to load stdlib");
    program = neuroscript::stdlib::merge_programs(stdlib, program);

    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "TransformerStack").expect("Codegen failed");

    insta::assert_snapshot!("codegen_unroll_threaded", code);
}

// ============================================================================
// Fat Arrow Reshape Snapshot Tests
// ============================================================================

#[test]
fn snapshot_fat_arrow_basic() {
    let source = fs::read_to_string("examples/fat_arrow_basic.ns")
        .expect("Failed to read fat_arrow_basic.ns");
    let program = parse(&source).expect("should parse");
    insta::assert_snapshot!("parser_ir_fat_arrow_basic", format_program_ir(&program));
}

#[test]
fn snapshot_fat_arrow_reduce() {
    let source = fs::read_to_string("examples/fat_arrow_reduce.ns")
        .expect("Failed to read fat_arrow_reduce.ns");
    let program = parse(&source).expect("should parse");
    insta::assert_snapshot!("parser_ir_fat_arrow_reduce", format_program_ir(&program));
}

#[test]
fn snapshot_fat_arrow_repeat() {
    let source = fs::read_to_string("examples/fat_arrow_repeat.ns")
        .expect("Failed to read fat_arrow_repeat.ns");
    let program = parse(&source).expect("should parse");
    insta::assert_snapshot!("parser_ir_fat_arrow_repeat", format_program_ir(&program));
}

#[test]
fn snapshot_codegen_fat_arrow_basic() {
    let source = fs::read_to_string("examples/fat_arrow_basic.ns")
        .expect("Failed to read fat_arrow_basic.ns");
    let mut program = parse(&source).expect("Parse failed");
    validate(&mut program).expect("Validation failed");
    let code = generate_pytorch(&program, "MultiHeadReshape").expect("Codegen failed");
    insta::assert_snapshot!("codegen_fat_arrow_basic", code);
}

#[test]
fn snapshot_codegen_fat_arrow_reduce() {
    let source = fs::read_to_string("examples/fat_arrow_reduce.ns")
        .expect("Failed to read fat_arrow_reduce.ns");
    let mut program = parse(&source).expect("Parse failed");
    validate(&mut program).expect("Validation failed");
    let code = generate_pytorch(&program, "GlobalAvgPool").expect("Codegen failed");
    insta::assert_snapshot!("codegen_fat_arrow_reduce", code);
}

#[test]
fn snapshot_codegen_fat_arrow_repeat() {
    let source = fs::read_to_string("examples/fat_arrow_repeat.ns")
        .expect("Failed to read fat_arrow_repeat.ns");
    let mut program = parse(&source).expect("Parse failed");
    validate(&mut program).expect("Validation failed");
    let code = generate_pytorch(&program, "ExpandDim").expect("Codegen failed");
    insta::assert_snapshot!("codegen_fat_arrow_repeat", code);
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

    let error_text = errors
        .iter()
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

    let error_text = errors
        .iter()
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

    // Paths that already have dedicated snapshot tests
    let skip_paths: Vec<&str> = vec![
        "examples/codegen_demo/residual.ns",
        "examples/codegen_demo/match_dimension_binding.ns",
        "examples/tutorials/03_match_guards.ns",
        "examples/real_world/cnn_demo.ns",
        "examples/real_world/cnn_demo_2.ns",
    ];

    for file_path in files {
        let path_str = file_path.to_string_lossy();

        // Skip if we already have a specific test for this file
        if skip_paths.iter().any(|s| path_str.ends_with(s)) {
            continue;
        }

        // Also skip stdlib FFN.ns (has dedicated test)
        if path_str.ends_with("stdlib/FFN.ns") || path_str.ends_with("stdlib\\FFN.ns") {
            continue;
        }

        let file_name = file_path.file_name().unwrap().to_str().unwrap();

        let source = fs::read_to_string(&file_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", file_name));

        // Create a snapshot name from the relative path
        let snapshot_name = if path_str.starts_with("stdlib/") {
            // Keep "stdlib_" prefix for top-level stdlib files
            path_str.replace("/", "_").replace(".ns", "")
        } else {
            let trimmed = path_str.trim_start_matches("examples/");
            format!("example_{}", trimmed.replace("/", "_").replace(".ns", ""))
        };

        match parse(&source) {
            Ok(program) => {
                let formatted = format_program_ir(&program);
                insta::assert_snapshot!(snapshot_name, formatted);
            }
            Err(e) => {
                // Some examples might be intentionally invalid for error testing
                let error_text = format!("Parse error: {:?}", e);
                let snapshot_name = format!("{}_error", snapshot_name);
                insta::assert_snapshot!(snapshot_name, error_text);
            }
        }
    }
}

// ============================================================================
// Bundle Mode Tests
// ============================================================================

#[test]
fn bundle_mode_no_runtime_imports() {
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

    let options = CodegenOptions { bundle: true };
    let code =
        generate_pytorch_with_options(&program, "SimpleLinear", &options).expect("Codegen failed");

    // Must not contain any neuroscript_runtime import statements (comments are fine)
    let has_runtime_import = code.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.starts_with("from neuroscript_runtime") || trimmed.starts_with("import neuroscript_runtime")
    });
    assert!(
        !has_runtime_import,
        "Bundle mode output must not import from neuroscript_runtime.\nGot:\n{}",
        code
    );

    // Must contain inlined class definitions
    assert!(
        code.contains("class Linear(nn.Module):"),
        "Bundle mode output must inline the Linear class definition"
    );

    // Must contain the unified import block
    assert!(code.contains("import torch"));
    assert!(code.contains("import torch.nn as nn"));
    assert!(code.contains("import torch.nn.functional as F"));
}

#[test]
fn bundle_mode_strips_module_docstrings() {
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

    let options = CodegenOptions { bundle: true };
    let code =
        generate_pytorch_with_options(&program, "SimpleLinear", &options).expect("Codegen failed");

    // The linear.py module docstring mentions "Linear (dense/fully-connected) layer primitive."
    // It must be stripped in bundle mode.
    assert!(
        !code.contains("Linear (dense/fully-connected) layer primitive"),
        "Bundle mode must strip module-level docstrings.\nGot:\n{}",
        code
    );
}

#[test]
fn bundle_mode_default_mode_unchanged() {
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

    // Default options (bundle: false) must produce the same output as generate_pytorch
    let default_code = generate_pytorch(&program, "SimpleLinear").expect("Codegen failed");
    let options = CodegenOptions { bundle: false };
    let explicit_code =
        generate_pytorch_with_options(&program, "SimpleLinear", &options).expect("Codegen failed");

    assert_eq!(default_code, explicit_code);

    // Default mode must use neuroscript_runtime imports
    assert!(
        default_code.contains("from neuroscript_runtime"),
        "Default mode should import from neuroscript_runtime"
    );
}

#[test]
fn bundle_mode_snapshot() {
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

    let options = CodegenOptions { bundle: true };
    let code =
        generate_pytorch_with_options(&program, "SimpleLinear", &options).expect("Codegen failed");

    insta::assert_snapshot!("codegen_bundle_simple_linear", code);
}
