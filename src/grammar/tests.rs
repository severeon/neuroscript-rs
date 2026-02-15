use super::*;
use pest::Parser;

#[test]
fn test_parse_simple_literal() {
    let input = "42";
    let result = NeuroScriptParser::parse(Rule::integer, input);
    assert!(
        result.is_ok(),
        "Failed to parse integer: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_identifier() {
    let input = "my_neuron";
    let result = NeuroScriptParser::parse(Rule::ident, input);
    assert!(
        result.is_ok(),
        "Failed to parse identifier: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_shape() {
    let input = "[batch, seq, dim]";
    let result = NeuroScriptParser::parse(Rule::shape, input);
    assert!(result.is_ok(), "Failed to parse shape: {:?}", result.err());
}

#[test]
fn test_parse_shape_with_wildcard() {
    let input = "[*, 512]";
    let result = NeuroScriptParser::parse(Rule::shape, input);
    assert!(
        result.is_ok(),
        "Failed to parse shape with wildcard: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_shape_with_expr() {
    let input = "[dim * 4]";
    let result = NeuroScriptParser::parse(Rule::shape, input);
    assert!(
        result.is_ok(),
        "Failed to parse shape with expression: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_use_stmt() {
    let input = "use core,nn/*";
    let result = NeuroScriptParser::parse(Rule::use_stmt, input);
    assert!(
        result.is_ok(),
        "Failed to parse use statement: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_simple_neuron() {
    let input = r#"neuron Linear(in_dim, out_dim):
in:
default: [*, in_dim]
out:
default: [*, out_dim]
impl:
core,nn/Linear
"#;
    let result = NeuroScriptParser::parse(Rule::neuron_def, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse simple neuron");
}

#[test]
fn test_parse_program() {
    let input = r#"use core,nn/*

neuron Linear(in_dim, out_dim):
in:
default: [*, in_dim]
out:
default: [*, out_dim]
impl:
core,nn/Linear
"#;
    let result = NeuroScriptParser::parse(Rule::program, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse program");
}

#[test]
fn test_parse_residual_example() {
    let input = include_str!("../../examples/stdlib/residual.ns");
    let result = NeuroScriptParser::parse(Rule::program, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse stdlib/residual.ns example");
}

// Test suite for example files
macro_rules! test_example_file {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let input = include_str!(concat!("../../examples/", $file));
            let result = NeuroScriptParser::parse(Rule::program, input);
            if let Err(e) = &result {
                eprintln!("Parse error in {}: {}", $file, e);
            }
            assert!(result.is_ok(), "Failed to parse {}", $file);
        }
    };
}

test_example_file!(test_primitives_basics, "primitives/basics.ns");
test_example_file!(test_primitives_activations, "primitives/activations.ns");
test_example_file!(test_primitives_structural, "primitives/structural.ns");
test_example_file!(test_primitives_operations, "primitives/operations.ns");
test_example_file!(test_primitives_attention, "primitives/attention.ns");
test_example_file!(test_stdlib_feedforward, "stdlib/feedforward.ns");
test_example_file!(test_stdlib_attention, "stdlib/attention.ns");
test_example_file!(test_stdlib_transformer_block, "stdlib/transformer_block.ns");
test_example_file!(test_tutorials_fork_join, "tutorials/02_fork_join.ns");
test_example_file!(test_match_multiline, "match_multiline.ns");
test_example_file!(test_real_world_resnet, "real_world/resnet.ns");
test_example_file!(test_dropout, "dropout.ns");
test_example_file!(test_unroll_threaded, "unroll_threaded.ns");
test_example_file!(test_unroll_context, "unroll_context.ns");
test_example_file!(test_unroll_static, "unroll_static.ns");
test_example_file!(test_unroll_gpt2, "unroll_gpt2.ns");

#[test]
fn test_multiple_connections_in_graph() {
    let input = r#"neuron Test:
in: [*shape]
out: [*shape]
graph:
in ->
    A()
    x

x ->
    B()
    out
"#;
    let result = NeuroScriptParser::parse(Rule::program, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse multiple connections");
}

#[test]
fn test_parse_if_expr() {
    let input = "if has_pool: pool else: Identity() -> out";
    let result = NeuroScriptParser::parse(Rule::if_expr, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse if expr");
}

#[test]
fn test_if_is_not_ident() {
    let input = "if";
    let result = NeuroScriptParser::parse(Rule::ident, input);
    assert!(result.is_err(), "if should not parse as identifier");
}

#[test]
fn test_if_as_endpoint() {
    let input = "if true: pool";
    let result = NeuroScriptParser::parse(Rule::endpoint, input);
    assert!(result.is_ok(), "Failed to parse if as endpoint");
}

#[test]
fn test_branch_pipeline_with_arrow() {
    let input = "Identity() -> out";
    let result = NeuroScriptParser::parse(Rule::branch_pipeline, input);
    assert!(result.is_ok());
}

#[test]
fn test_parse_connection_with_if() {
    let input = "act -> if has_pool: pool else: Identity() -> out\n";
    let result = NeuroScriptParser::parse(Rule::connection, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok());
}

#[test]
fn test_parse_simple_neuron_with_if() {
    let input = r#"neuron Test:
    graph:
        act -> if has_pool: pool else: Identity() -> out
"#;
    let result = NeuroScriptParser::parse(Rule::neuron_def, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok());
}

#[test]
fn test_doc_comment_blank_line_before_neuron() {
    // Single blank line between doc comment and neuron keyword
    let input = r#"/// A documented neuron

neuron Test:
  in: [*, dim]
  out: [*, dim]
  impl: core,nn/Linear
"#;
    let result = NeuroScriptParser::parse(Rule::neuron_def, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse neuron_def with single blank line after doc comment");

    // Multiple blank lines between doc comment and neuron keyword
    let input_multi = r#"/// A documented neuron


neuron Test:
  in: [*, dim]
  out: [*, dim]
  impl: core,nn/Linear
"#;
    let result_multi = NeuroScriptParser::parse(Rule::neuron_def, input_multi);
    if let Err(e) = &result_multi {
        eprintln!("Error: {}", e);
    }
    assert!(result_multi.is_ok(), "Failed to parse neuron_def with multiple blank lines after doc comment");
}

#[test]
fn test_parse_variadic_input_port() {
    // Variadic input port syntax: in *inputs: [*shape]
    let input = r#"neuron Concat(dim):
    in *inputs: [*shape]
    out: [*shape_out]
    impl: core,structural/Concat
"#;
    let result = NeuroScriptParser::parse(Rule::neuron_def, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse neuron with variadic input port");
}

// ============================================================================
// Unroll grammar tests
// ============================================================================

#[test]
fn test_parse_named_unroll_context_block() {
    // Named context-level unroll block
    let input = "blocks = unroll(3):\n    block = TransformerBlock(d_model)\n";
    let result = NeuroScriptParser::parse(Rule::named_unroll_context_block, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse named unroll context block");
}

#[test]
fn test_parse_named_unroll_context_with_annotation() {
    // Named context-level unroll with @static annotation
    let input = "layers = unroll(num_layers):\n    @static block = TransformerBlock(d_model)\n";
    let result = NeuroScriptParser::parse(Rule::named_unroll_context_block, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse named unroll context block with @static");
}

#[test]
fn test_parse_named_unroll_context_with_param() {
    // Named context-level unroll with parameter reference
    let input = "blocks = unroll(num_layers):\n    block = Block(dim)\n";
    let result = NeuroScriptParser::parse(Rule::named_unroll_context_block, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse named unroll context block with param");
}

#[test]
fn test_unroll_is_not_ident() {
    // "unroll" should not parse as an identifier (it's a keyword)
    let input = "unroll";
    let result = NeuroScriptParser::parse(Rule::ident, input);
    assert!(result.is_err(), "unroll should not parse as identifier");
}
