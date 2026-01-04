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
    let input = include_str!("../../examples/residual.ns");
    let result = NeuroScriptParser::parse(Rule::program, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse residual.ns example");
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

test_example_file!(test_01_comments, "01-comments.ns");
test_example_file!(test_02_imports, "02-imports.ns");
test_example_file!(test_03_parameters, "03-parameters.ns");
test_example_file!(test_04_shapes, "04-shapes.ns");
test_example_file!(test_05_ports, "05-ports.ns");
test_example_file!(test_06_impl_refs, "06-impl-refs.ns");
test_example_file!(test_07_pipelines, "07-pipelines.ns");
test_example_file!(test_08_tuples, "08-tuples.ns");
test_example_file!(test_09_port_access, "09-port-access.ns");
test_example_file!(test_10_match, "10-match.ns");
test_example_file!(test_xor, "22-xor.ns");
test_example_file!(test_addition, "27-addition.ns");

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
